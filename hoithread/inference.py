import threading
import queue
import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

class InferenceThread(threading.Thread):

    PERSON_CLASS   = 0
    ANIMAL_CLASSES = [15, 16]
    OBJECT_CLASSES = [24,25,26,28,39,41,45,56,57,60,62,63,64,67,73,74,75,76]
    ALL_CLASSES    = [PERSON_CLASS] + ANIMAL_CLASSES + OBJECT_CLASSES

    CONTACT_GAP_M     = 0.06   # metros — mano a menos de 6cm del fondo = contacto
    CONTACT_CONF_FULL = 0.95   # confianza cuando hay depth confiable
    CONTACT_CONF_LOW  = 0.50   # confianza con fallback cinemático
    RELIABILITY_MIN   = 0.4    # mínimo de confiabilidad del depth para usarlo
    ROI_PAD           = 15

    # Cinemático
    VEL_CONTACT_PX    = 3.0    # px — velocidad bajo este valor = mano quieta
    STAB_FRAMES       = 5      # frames para medir estabilidad

    def __init__(self, cam_id: str, in_queue: queue.Queue,
                 fusion_queue: queue.Queue, q_env: queue.Queue,
                 has_depth: bool, env_builder: "EnvironmentBuilderThread"):
        super().__init__(daemon=True)
        self.cam_id       = cam_id
        self.in_queue     = in_queue
        self.fusion_queue = fusion_queue
        self.q_env        = q_env
        self.has_depth    = has_depth
        self.env_builder  = env_builder
        self.running      = True

        # Mapa del entorno — se llena via callback cuando está listo
        self._env_map     = None
        self._env_lock    = threading.Lock()

        # Registrar en EnvironmentBuilder si tiene depth
        if has_depth:
            env_builder.register_camera(cam_id, self._on_env_ready)

        # ── Modelos ───────────────────────────────────────────────────
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model  = YOLO("yolo26x.pt").to(self.device)

        base_opts  = mp_python.BaseOptions(model_asset_path="hand_landmarker.task")
        hand_opts  = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.IMAGE
        )
        self.hands_model = mp_vision.HandLandmarker.create_from_options(hand_opts)

        # ── Intrínsecos ───────────────────────────────────────────────
        self.FRAME_W = 320;  self.FRAME_H = 240
        self.FX = 525.0 * (self.FRAME_W / 640)
        self.FY = 525.0 * (self.FRAME_H / 480)
        self.CX = self.FRAME_W / 2
        self.CY = self.FRAME_H / 2

        # ── Estado cinemático ─────────────────────────────────────────
        self._prev_hand_pts  = []          # frame anterior
        self._hand_history   = []          # últimos N frames para estabilidad
        self._prev_frame     = None

        #print(f"[Inf:{cam_id}] listo — depth={'sí' if has_depth else 'no'}")

    # ──────────────────────────────────────────────────────────────────
    # CALLBACK — mapa listo
    # ──────────────────────────────────────────────────────────────────
    def _on_env_ready(self, env_map: dict):
        with self._env_lock:
            self._env_map = env_map
        #print(f"[Inf:{self.cam_id}] entorno listo, iniciando detección de contacto")

    def _get_env_map(self):
        with self._env_lock:
            return self._env_map

    # ──────────────────────────────────────────────────────────────────
    # UTILS DEPTH
    # ──────────────────────────────────────────────────────────────────
    def _get_depth_m(self, depth_data, px, py, k=3):
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

    def _pixel_to_3d(self, depth_data, px, py):
        d = self._get_depth_m(depth_data, px, py)
        if d is None or d <= 0:
            return None
        return np.array([
            (px - self.CX) * d / self.FX,
            (py - self.CY) * d / self.FY,
            d
        ])

    # ──────────────────────────────────────────────────────────────────
    # DETECCIÓN DE CONTACTO
    # ──────────────────────────────────────────────────────────────────
    def _detect_contacts(self, hand_pts, depth_data, frame) -> list:
        """
        Detecta contactos combinando depth+fondo (RGBD) y/o cinemática (RGB).
        Retorna lista de contactos con posición 3D y confianza.
        """
        contacts     = []
        env_map      = self._get_env_map()
        has_env      = env_map is not None
        has_depth_ok = depth_data is not None and self.has_depth

        for (hx, hy) in hand_pts:

            contact = None

            # ── MÉTODO 1: depth + mapa base (RGBD con entorno calibrado) ──
            if has_depth_ok and has_env:
                contact = self._contact_from_depth(
                    hx, hy, depth_data, env_map
                )

            # ── MÉTODO 2: depth sin mapa base (RGBD, aún calibrando) ──
            elif has_depth_ok and not has_env:
                contact = self._contact_from_depth_raw(
                    hx, hy, depth_data
                )

            # ── MÉTODO 3: fallback cinemático (RGB sin depth) ──────────
            if contact is None:
                contact = self._contact_from_kinematics(hx, hy, frame)

            if contact is not None:
                contacts.append(contact)

        return contacts

    def _contact_from_depth(self, hx, hy, depth_data, env_map) -> dict:
        """Contacto físico comparando mano con mapa base."""
        if depth_data.ndim == 3:
            depth_data = depth_data[:, :, 0]

        reliability = env_map["reliability"][hy, hx]
        if reliability < self.RELIABILITY_MIN:
            return None   # zona poco confiable del sensor

        d_hand = self._get_depth_m(depth_data, hx, hy)
        if d_hand is None:
            return None

        d_bg = env_map["background_depth"][hy, hx] / 1000.0
        if d_bg <= 0:
            return None

        gap = d_bg - d_hand   # positivo = mano más cerca que el fondo

        if 0 <= gap < self.CONTACT_GAP_M:
            p3d          = self._pixel_to_3d(depth_data, hx, hy)
            surface_code = env_map["surface_types"][hy, hx]
            surface_map  = {0: "unknown", 1: "horizontal", 2: "vertical"}

            return {
                "pixel":        (hx, hy),
                "pos_3d":       p3d,
                "confidence":   float(self.CONTACT_CONF_FULL * reliability),
                "method":       "depth",
                "surface_type": surface_map.get(int(surface_code), "unknown"),
                "gap_m":        round(gap, 4),
            }
        return None

    def _contact_from_depth_raw(self, hx, hy, depth_data) -> dict:
        """
        Contacto estimado comparando mano con superficie circundante.
        Usado mientras el mapa base aún se está construyendo.
        """
        if depth_data.ndim == 3:
            depth_data = depth_data[:, :, 0]

        d_hand = self._get_depth_m(depth_data, hx, hy, k=3)
        if d_hand is None:
            return None

        # Superficie en anillo exterior a la mano
        h, w   = depth_data.shape
        y, x   = np.ogrid[:h, :w]
        dist   = np.sqrt((x-hx)**2 + (y-hy)**2)
        ring   = (dist >= 8) & (dist < 20)
        vals   = depth_data[ring].astype(np.float32)
        valid  = vals[vals > 0]
        if not len(valid):
            return None

        d_surf = float(np.median(valid)) / 1000.0
        gap    = d_surf - d_hand

        if 0 <= gap < self.CONTACT_GAP_M:
            p3d = self._pixel_to_3d(depth_data, hx, hy)
            return {
                "pixel":        (hx, hy),
                "pos_3d":       p3d,
                "confidence":   0.7,        # menos confiable sin mapa base
                "method":       "depth_raw",
                "surface_type": "unknown",
                "gap_m":        round(gap, 4),
            }
        return None

    def _contact_from_kinematics(self, hx, hy, frame) -> dict:
        """
        Fallback cinemático para cámaras RGB sin depth.
        Combina velocidad + estabilidad del landmark.
        """
        scores = []

        # Señal 1: velocidad baja → mano quieta
        if self._prev_hand_pts:
            dists = [np.linalg.norm(np.array([hx,hy]) - np.array(p))
                     for p in self._prev_hand_pts]
            min_dist = min(dists) if dists else 999
            vel_score = float(np.clip(1.0 - min_dist / self.VEL_CONTACT_PX, 0, 1))
            scores.append(("velocity", vel_score, 0.5))

        # Señal 2: estabilidad en últimos N frames
        if len(self._hand_history) >= self.STAB_FRAMES:
            recent = list(self._hand_history)[-self.STAB_FRAMES:]
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

        # Sin depth no hay posición 3D real
        return {
            "pixel":        (hx, hy),
            "pos_3d":       None,       # FusionThread lo proyecta via rayo
            "confidence":   float(prob * self.CONTACT_CONF_LOW),
            "method":       "kinematic",
            "surface_type": "unknown",
            "gap_m":        None,
        }

    def _build_contact_mask(self, contacts, shape) -> np.ndarray:
        """Máscara de pixels donde hubo contacto (para actualizar el fondo)."""
        mask = np.zeros(shape, dtype=bool)
        for c in contacts:
            hx, hy = c["pixel"]
            r = 20
            y1 = max(0, hy-r);  y2 = min(shape[0], hy+r)
            x1 = max(0, hx-r);  x2 = min(shape[1], hx+r)
            mask[y1:y2, x1:x2] = True
        return mask

    # ──────────────────────────────────────────────────────────────────
    # MEDIAPIPE
    # ──────────────────────────────────────────────────────────────────
    def _extract_hands(self, roi, roi_x0, roi_y0):
        rgb     = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result  = self.hands_model.detect(mp_img)
        pts     = []
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

    # ──────────────────────────────────────────────────────────────────
    # LOOP PRINCIPAL
    # ──────────────────────────────────────────────────────────────────
    def run(self):
        try:
            while self.running:
                try:
                    item       = self.in_queue.get(timeout=1)
                    frame      = item.get("color")
                    depth_data = item.get("depth")
                    now        = time.time()

                    if frame is None:
                        continue

                    h_orig, w_orig = frame.shape[:2]
                    annotated      = frame.copy()

                    # ── YOLO ───────────────────────────────────────────
                    f_yolo  = cv2.resize(frame, (640, 480))
                    results = self.model.track(
                        f_yolo, persist=True,
                        classes=self.ALL_CLASSES,
                        device=self.device,
                        verbose=False, conf=0.35
                    )

                    sx = w_orig / 640
                    sy = h_orig / 480

                    persons_raw = []
                    has_person  = False

                    if results[0].boxes is not None and len(results[0].boxes):
                        boxes  = results[0].boxes.xyxy.cpu().numpy()
                        confs  = results[0].boxes.conf.cpu().numpy()
                        clss   = results[0].boxes.cls.cpu().numpy().astype(int)
                        ids    = results[0].boxes.id

                        for i, (box, conf, cid) in enumerate(zip(boxes, confs, clss)):
                            x1 = int(box[0]*sx); y1 = int(box[1]*sy)
                            x2 = int(box[2]*sx); y2 = int(box[3]*sy)
                            tid = int(ids[i].item()) if ids is not None else -1

                            if cid == self.PERSON_CLASS:
                                has_person = True
                                persons_raw.append((tid, (x1,y1,x2,y2), conf))

                            # Anotar
                            color = (0,255,0) if cid == self.PERSON_CLASS else (140,140,140)
                            cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 1)
                            cv2.putText(annotated, f"{self.model.names[cid]}#{tid}",
                                        (x1, max(y1-4,8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

                    # ── Enviar al EnvironmentBuilder (solo RGBD) ───────
                    if self.has_depth and depth_data is not None:
                        try:
                            self.q_env.put_nowait({
                                "cam_id":     self.cam_id,
                                "depth":      depth_data,
                                "has_person": has_person,
                            })
                        except queue.Full:
                            pass

                    # ── Mostrar estado de calibración si no está listo ─
                    env_ready = self._get_env_map() is not None
                    if self.has_depth and not env_ready:
                        pct = int(len(
                            self.env_builder._state.get(
                                self.cam_id, {}
                            ).get("buffer", [])
                        ) / self.env_builder.N_FRAMES_CALIB * 100)

                        cv2.putText(annotated,
                            f"Calibrando... {pct}%",
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,165,255), 2)
                        cv2.rectangle(annotated,
                            (10, 28), (10 + pct*2, 40),
                            (0,165,255), -1)

                    # ── MediaPipe + contactos por persona ──────────────
                    all_contacts = []

                    for tid, (x1,y1,x2,y2), conf in persons_raw:
                        rx1 = max(0, x1 - self.ROI_PAD)
                        ry1 = max(0, y1 - self.ROI_PAD)
                        rx2 = min(w_orig, x2 + self.ROI_PAD)
                        ry2 = min(h_orig, y2 + self.ROI_PAD)
                        roi = frame[ry1:ry2, rx1:rx2]
                        if roi.size == 0:
                            continue

                        hand_pts = self._extract_hands(roi, rx1, ry1)

                        # Actualizar historial cinemático
                        self._hand_history.append(hand_pts)
                        if len(self._hand_history) > self.STAB_FRAMES + 2:
                            self._hand_history.pop(0)

                        # Detectar contactos
                        contacts = self._detect_contacts(
                            hand_pts, depth_data, frame
                        )

                        # Agregar track_id a cada contacto
                        for c in contacts:
                            c["tid"] = tid
                            all_contacts.append(c)

                        # En InferenceThread.run(), después de detectar contactos: QUITAR
                        #if all_contacts:
                            #print(f"[Inf:{self.cam_id}] {len(all_contacts)} contactos detectados")
                            #for c in all_contacts[:2]:  # mostrar max 2
                                #print(f"  → {c['method']} conf={c['confidence']:.2f} pos={c.get('pos_3d')}")
                        # Actualizar fondo en zonas de contacto
                        if self.has_depth and depth_data is not None:
                            mask = self._build_contact_mask(
                                contacts, (h_orig, w_orig)
                            )
                            self.env_builder.update_background(
                                self.cam_id, depth_data, mask
                            )

                        # Anotar manos y contactos
                        for (hx, hy) in hand_pts:
                            cv2.circle(annotated, (hx,hy), 3, (0,255,255), -1)

                        for c in contacts:
                            hx, hy = c["pixel"]
                            color  = (0,0,255) if c["confidence"] > 0.7 \
                                     else (0,165,255)
                            cv2.circle(annotated, (hx,hy), 6, color, 2)
                            cv2.putText(annotated,
                                f"{c['method']} {c['confidence']:.2f}",
                                (hx+8, hy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

                    # Actualizar estado cinemático
                    self._prev_hand_pts = [p for pts in
                                           [h for _, (x1,y1,x2,y2), _ in persons_raw
                                            for h in [self._extract_hands(
                                                frame[max(0,y1-self.ROI_PAD):
                                                      min(h_orig,y2+self.ROI_PAD),
                                                      max(0,x1-self.ROI_PAD):
                                                      min(w_orig,x2+self.ROI_PAD)],
                                                max(0,x1-self.ROI_PAD),
                                                max(0,y1-self.ROI_PAD)
                                            )]]
                                           for p in pts]
                    self._prev_frame = frame.copy()

                    # ── Publicar al FusionThread ───────────────────────
                    payload = {
                        "cam_id":          self.cam_id,
                        "timestamp":       now,
                        "has_depth":       self.has_depth,
                        "contacts":        all_contacts,
                        "frame_annotated": annotated,
                        "frame_shape":     (h_orig, w_orig),
                    }
                    try:
                        self.fusion_queue.put_nowait(payload)
                    except queue.Full:
                        pass

                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"[Inf:{self.cam_id}] ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        finally:
            self._cleanup()

    def _cleanup(self):
        print(f"[Inf:{self.cam_id}] liberando modelos...")
        try:
            del self.model
        except Exception:
            pass
        try:
            self.hands_model.close()
            del self.hands_model
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
        import gc
        gc.collect()
        print(f"[Inf:{self.cam_id}] memoria liberada")

    def stop(self):
        self.running = False
