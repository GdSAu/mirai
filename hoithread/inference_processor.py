"""
hoithread/inference_processor.py — InferenceProcessor sin hilo.

Misma lógica que InferenceThread pero expuesta como llamada directa
(process_frame) en lugar de un bucle con queue. Diseñado para uso
offline en postprocess.py.

El estado YOLO (tracking IDs), el historial cinemático y el mapa de
entorno persisten entre llamadas — instanciar una vez por sesión y
llamar process_frame() en orden.
"""

import cv2
import torch
import numpy as np
import time

from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from hoithread.environmentbuild import EnvironmentBuilderThread


class InferenceProcessor:
    """
    Procesador de inferencia síncrono (sin cola, sin hilo).

    Parámetros
    ----------
    cam_id      : identificador de la cámara (ej. "RS_123456")
    has_depth   : True si la cámara tiene stream de profundidad
    env_map     : dict con llaves "background_depth", "reliability",
                  "surface_types", "cam_id" (cargado desde .npy en offline).
                  Si es None, InferenceProcessor arranca en modo calibración
                  igual que InferenceThread — los primeros N_FRAMES_CALIB
                  frames no producen contactos con método "depth".
    """

    PERSON_CLASS   = 0
    ANIMAL_CLASSES = [15, 16]
    OBJECT_CLASSES = [24,25,26,28,39,41,45,56,57,60,62,63,64,67,73,74,75,76]
    ALL_CLASSES    = [PERSON_CLASS] + ANIMAL_CLASSES + OBJECT_CLASSES

    CONTACT_GAP_M     = 0.06
    CONTACT_CONF_FULL = 0.95
    CONTACT_CONF_LOW  = 0.50
    RELIABILITY_MIN   = 0.4
    ROI_PAD           = 15

    VEL_CONTACT_PX = 3.0
    STAB_FRAMES    = 5

    N_FRAMES_CALIB = 60   # warm-up si no se inyecta env_map

    def __init__(self, cam_id: str, has_depth: bool,
                 env_map: dict | None = None):
        self.cam_id    = cam_id
        self.has_depth = has_depth
        self._env_map  = env_map   # None → modo calibración

        # Warm-up interno (usado solo si env_map=None)
        self._calib_buffer: list[np.ndarray] = []
        self._calib_done = env_map is not None

        # ── Modelos ───────────────────────────────────────────────────────
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model  = YOLO("yolo26x.pt").to(self.device)

        base_opts  = mp_python.BaseOptions(model_asset_path="hand_landmarker.task")
        hand_opts  = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self.hands_model = mp_vision.HandLandmarker.create_from_options(hand_opts)

        # ── Intrínsecos (iguales que InferenceThread) ─────────────────────
        self.FRAME_W = 320;  self.FRAME_H = 240
        self.FX = 525.0 * (self.FRAME_W / 640)
        self.FY = 525.0 * (self.FRAME_H / 480)
        self.CX = self.FRAME_W / 2
        self.CY = self.FRAME_H / 2

        # ── Estado cinemático ─────────────────────────────────────────────
        self._prev_hand_pts: list = []
        self._hand_history:  list = []

        if env_map is not None:
            print(f"[InfProc:{cam_id}] mapa de entorno inyectado — sin warm-up")
        else:
            status = "depth activo" if has_depth else "solo RGB"
            print(f"[InfProc:{cam_id}] iniciado ({status}) — "
                  f"warm-up: {self.N_FRAMES_CALIB} frames")

    # ── API principal ─────────────────────────────────────────────────────

    def process_frame(self, color: np.ndarray,
                      depth: np.ndarray | None) -> dict:
        """
        Procesa un frame y devuelve:
            {
              "frame_annotated": np.ndarray (BGR),
              "contacts":        list[dict],
              "has_person":      bool,
              "calibrating":     bool,   # True mientras el mapa aún no está listo
              "calib_pct":       int,    # 0-100, porcentaje de calibración
            }
        """
        h_orig, w_orig = color.shape[:2]
        annotated      = color.copy()
        all_contacts   = []
        has_person     = False

        # ── Warm-up interno si no se inyectó env_map ──────────────────────
        calibrating = False
        calib_pct   = 100
        if not self._calib_done and self.has_depth and depth is not None:
            self._calib_buffer.append(depth.copy())
            calib_pct   = int(len(self._calib_buffer) / self.N_FRAMES_CALIB * 100)
            calibrating = True

            if len(self._calib_buffer) >= self.N_FRAMES_CALIB:
                self._env_map    = self._build_env_map(self._calib_buffer)
                self._calib_done = True
                self._calib_buffer.clear()
                print(f"[InfProc:{self.cam_id}] mapa construido ✓")

        # Mostrar progreso en el frame anotado
        if calibrating:
            cv2.putText(annotated, f"Calibrando... {calib_pct}%",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 2)
            cv2.rectangle(annotated, (10, 28), (10 + calib_pct * 2, 40),
                          (0,165,255), -1)

        # ── YOLO ──────────────────────────────────────────────────────────
        f_yolo  = cv2.resize(color, (640, 480))
        results = self.model.track(
            f_yolo, persist=True,
            classes=self.ALL_CLASSES,
            device=self.device,
            verbose=False, conf=0.35,
        )

        sx = w_orig / 640
        sy = h_orig / 480

        persons_raw = []   # [(tid, (x1,y1,x2,y2), conf)]
        per_person_hand_pts = {}   # {tid: [pts]}

        if results[0].boxes is not None and len(results[0].boxes):
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            clss  = results[0].boxes.cls.cpu().numpy().astype(int)
            ids   = results[0].boxes.id

            for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, clss)):
                x1 = int(box[0]*sx); y1 = int(box[1]*sy)
                x2 = int(box[2]*sx); y2 = int(box[3]*sy)
                tid = int(ids[i].item()) if ids is not None else -1

                if cls_id == self.PERSON_CLASS:
                    has_person = True
                    persons_raw.append((tid, (x1,y1,x2,y2), conf))

                draw_color = (0,255,0) if cls_id == self.PERSON_CLASS else (140,140,140)
                cv2.rectangle(annotated, (x1,y1), (x2,y2), draw_color, 1)
                cv2.putText(annotated,
                            f"{self.model.names[cls_id]}#{tid}",
                            (x1, max(y1-4, 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, draw_color, 1)

        # ── MediaPipe + contactos ─────────────────────────────────────────
        for tid, (x1,y1,x2,y2), _ in persons_raw:
            rx1 = max(0, x1 - self.ROI_PAD)
            ry1 = max(0, y1 - self.ROI_PAD)
            rx2 = min(w_orig, x2 + self.ROI_PAD)
            ry2 = min(h_orig, y2 + self.ROI_PAD)
            roi = color[ry1:ry2, rx1:rx2]
            if roi.size == 0:
                continue

            hand_pts = self._extract_hands(roi, rx1, ry1)
            per_person_hand_pts[tid] = hand_pts

            self._hand_history.append(hand_pts)
            if len(self._hand_history) > self.STAB_FRAMES + 2:
                self._hand_history.pop(0)

            contacts = self._detect_contacts(hand_pts, depth, color)

            for c in contacts:
                c["tid"] = tid
                all_contacts.append(c)

            # Anotar manos
            for (hx, hy) in hand_pts:
                cv2.circle(annotated, (hx,hy), 3, (0,255,255), -1)

            for c in contacts:
                hx, hy     = c["pixel"]
                draw_color = (0,0,255) if c["confidence"] > 0.7 else (0,165,255)
                cv2.circle(annotated, (hx,hy), 6, draw_color, 2)
                cv2.putText(annotated,
                            f"{c['method']} {c['confidence']:.2f}",
                            (hx+8, hy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, draw_color, 1)

        # Actualizar estado cinemático con los puntos ya calculados
        self._prev_hand_pts = [
            pt
            for pts in per_person_hand_pts.values()
            for pt in pts
        ]

        return {
            "frame_annotated": annotated,
            "contacts":        all_contacts,
            "has_person":      has_person,
            "calibrating":     calibrating,
            "calib_pct":       calib_pct,
        }

    def close(self):
        """Libera los modelos de GPU/CPU."""
        try:
            del self.model
        except Exception:
            pass
        try:
            self.hands_model.close()
            del self.hands_model
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Detección de contacto (idéntica a InferenceThread) ────────────────

    def _detect_contacts(self, hand_pts, depth_data, frame) -> list:
        contacts = []
        env_map  = self._env_map
        has_env  = env_map is not None
        has_d    = depth_data is not None and self.has_depth

        for (hx, hy) in hand_pts:
            contact = None
            if has_d and has_env:
                contact = self._contact_from_depth(hx, hy, depth_data, env_map)
            elif has_d and not has_env:
                contact = self._contact_from_depth_raw(hx, hy, depth_data)
            if contact is None:
                contact = self._contact_from_kinematics(hx, hy, frame)
            if contact is not None:
                contacts.append(contact)
        return contacts

    def _contact_from_depth(self, hx, hy, depth_data, env_map) -> dict | None:
        if depth_data.ndim == 3:
            depth_data = depth_data[:, :, 0]
        reliability = env_map["reliability"][hy, hx]
        if reliability < self.RELIABILITY_MIN:
            return None
        d_hand = self._get_depth_m(depth_data, hx, hy)
        if d_hand is None:
            return None
        d_bg = env_map["background_depth"][hy, hx] / 1000.0
        if d_bg <= 0:
            return None
        gap = d_bg - d_hand
        if 0 <= gap < self.CONTACT_GAP_M:
            surface_map = {0: "unknown", 1: "horizontal", 2: "vertical"}
            return {
                "pixel":        (hx, hy),
                "pos_3d":       self._pixel_to_3d(depth_data, hx, hy),
                "confidence":   float(self.CONTACT_CONF_FULL * reliability),
                "method":       "depth",
                "surface_type": surface_map.get(int(env_map["surface_types"][hy, hx]), "unknown"),
                "gap_m":        round(gap, 4),
            }
        return None

    def _contact_from_depth_raw(self, hx, hy, depth_data) -> dict | None:
        if depth_data.ndim == 3:
            depth_data = depth_data[:, :, 0]
        d_hand = self._get_depth_m(depth_data, hx, hy, k=3)
        if d_hand is None:
            return None
        h, w = depth_data.shape
        y, x  = np.ogrid[:h, :w]
        dist  = np.sqrt((x-hx)**2 + (y-hy)**2)
        ring  = (dist >= 8) & (dist < 20)
        vals  = depth_data[ring].astype(np.float32)
        valid = vals[vals > 0]
        if not len(valid):
            return None
        d_surf = float(np.median(valid)) / 1000.0
        gap    = d_surf - d_hand
        if 0 <= gap < self.CONTACT_GAP_M:
            return {
                "pixel":        (hx, hy),
                "pos_3d":       self._pixel_to_3d(depth_data, hx, hy),
                "confidence":   0.7,
                "method":       "depth_raw",
                "surface_type": "unknown",
                "gap_m":        round(gap, 4),
            }
        return None

    def _contact_from_kinematics(self, hx, hy, frame) -> dict | None:
        scores = []
        if self._prev_hand_pts:
            dists     = [np.linalg.norm(np.array([hx,hy]) - np.array(p))
                         for p in self._prev_hand_pts]
            vel_score = float(np.clip(1.0 - min(dists) / self.VEL_CONTACT_PX, 0, 1))
            scores.append(("velocity", vel_score, 0.5))
        if len(self._hand_history) >= self.STAB_FRAMES:
            recent  = list(self._hand_history)[-self.STAB_FRAMES:]
            all_pts = [p for frame_pts in recent for p in frame_pts]
            if all_pts:
                pts_arr = np.array(all_pts)
                var     = pts_arr.var(axis=0).mean()
                stab    = float(np.clip(1.0 - var / 4.0, 0, 1))
                scores.append(("stability", stab, 0.5))
        if not scores:
            return None
        total_w = sum(w for _, _, w in scores)
        prob    = sum(v * w for _, v, w in scores) / total_w
        if prob < 0.65:
            return None
        return {
            "pixel":        (hx, hy),
            "pos_3d":       None,
            "confidence":   float(prob * self.CONTACT_CONF_LOW),
            "method":       "kinematic",
            "surface_type": "unknown",
            "gap_m":        None,
        }

    # ── MediaPipe ─────────────────────────────────────────────────────────

    def _extract_hands(self, roi, roi_x0, roi_y0) -> list:
        rgb    = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.hands_model.detect(mp_img)
        pts    = []
        if not result.hand_landmarks:
            return pts
        h, w    = roi.shape[:2]
        TIP_IDS = [4, 8, 12, 16, 20]
        for hand_lms in result.hand_landmarks:
            for idx in TIP_IDS:
                lm = hand_lms[idx]
                fx = max(0, min(self.FRAME_W-1, int(lm.x * w) + roi_x0))
                fy = max(0, min(self.FRAME_H-1, int(lm.y * h) + roi_y0))
                pts.append((fx, fy))
        return pts

    # ── Utils depth (idénticos a InferenceThread) ─────────────────────────

    def _get_depth_m(self, depth_data, px, py, k=3) -> float | None:
        if depth_data is None:
            return None
        if depth_data.ndim == 3:
            depth_data = depth_data[:, :, 0]
        h, w = depth_data.shape
        x1 = max(0, px-k//2);  x2 = min(w, px+k//2+1)
        y1 = max(0, py-k//2);  y2 = min(h, py+k//2+1)
        patch = depth_data[y1:y2, x1:x2].astype(np.float32)
        valid = patch[patch > 0]
        return float(np.median(valid)) / 1000.0 if len(valid) else None

    def _pixel_to_3d(self, depth_data, px, py) -> np.ndarray | None:
        d = self._get_depth_m(depth_data, px, py)
        if d is None or d <= 0:
            return None
        return np.array([
            (px - self.CX) * d / self.FX,
            (py - self.CY) * d / self.FY,
            d,
        ])

    # ── Construcción del mapa de entorno (warm-up offline) ────────────────

    def _build_env_map(self, buffer: list) -> dict:
        """Replica la lógica de EnvironmentBuilderThread._build_map."""
        import cv2 as _cv2

        normalized = []
        for frame in buffer:
            f = frame[:, :, 0] if frame.ndim == 3 else frame
            normalized.append(f)

        stack      = np.stack(normalized, axis=0).astype(np.float32)
        background = np.median(stack, axis=0)
        std        = np.std(stack, axis=0)
        zero_ratio = (stack == 0).mean(axis=0)

        reliability = (1.0 - np.clip(std / 100.0, 0, 1))
        reliability *= (1.0 - np.clip(zero_ratio / 0.3, 0, 1))

        # Clasificar superficies (horizontal/vertical/unknown)
        depth_m = background / 1000.0
        depth_m[depth_m == 0] = np.nan
        grad_x   = _cv2.Sobel(depth_m, _cv2.CV_32F, 1, 0, ksize=5)
        grad_y   = _cv2.Sobel(depth_m, _cv2.CV_32F, 0, 1, ksize=5)
        norm_len = np.sqrt(grad_x**2 + grad_y**2 + 1)
        ny = -grad_y / norm_len
        nz =  1.0   / norm_len
        surface = np.zeros(depth_m.shape, dtype=np.uint8)
        surface[np.abs(ny) > 0.7] = 1
        surface[np.abs(nz) > 0.7] = 2

        return {
            "background_depth": background,
            "reliability":      reliability.astype(np.float32),
            "surface_types":    surface,
            "cam_id":           self.cam_id,
        }
