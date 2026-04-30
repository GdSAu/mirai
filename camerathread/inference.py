import threading
import queue
import time
import cv2
import numpy as np
from collections import defaultdict

import torch
from ultralytics import YOLO

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


class InferenceThread(threading.Thread):
    def __init__(self, q):
        super().__init__()
        self.q = q
        self.running = True

        # ─────────────────────────────
        # MODELOS
        # ─────────────────────────────
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model  = YOLO("yolo26x.pt").to(self.device)

        base_options = mp_python.BaseOptions(model_asset_path="hand_landmarker.task")
        hand_options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            running_mode=mp_vision.RunningMode.IMAGE
        )
        self.hands_model = mp_vision.HandLandmarker.create_from_options(hand_options)

        # ─────────────────────────────
        # PARAMS
        # ─────────────────────────────
        self.FRAME_W, self.FRAME_H = 320, 240
        self.FX = 525.0 * (self.FRAME_W / 640)
        self.FY = 525.0 * (self.FRAME_H / 480)
        self.CX = self.FRAME_W / 2
        self.CY = self.FRAME_H / 2

        self.PERSON_CLASS = 0
        self.ALL_CLASSES = [0]

        self.ROI_PAD = 15
        self.INTENSITY = 35
        self.DECAY_VISUAL = 0.93

        self.HOI_HAND_DIST_3D = 0.1
        self.HOI_HAND_DIST_2D = 80

        # ─────────────────────────────
        # ESTADO
        # ─────────────────────────────
        self.heatmap_visual = np.zeros((self.FRAME_H, self.FRAME_W), dtype=np.float32)
        self.track_history = defaultdict(lambda: {"path": []})

        print("[INFO] InferenceThread inicializado")

    # ─────────────────────────────
    # UTILS
    # ─────────────────────────────
    def get_depth(self, depth, x, y):
        if depth is None:
            return None
        d = depth[y, x]
        return None if d == 0 else d / 1000.0

    def pixel_to_3d(self, depth, x, y):
        d = self.get_depth(depth, x, y)
        if d is None:
            return None
        X = (x - self.CX) * d / self.FX
        Y = (y - self.CY) * d / self.FY
        return np.array([X, Y, d])

    def add_heat(self, heatmap, cx, cy):
        h, w = heatmap.shape
        y, x = np.ogrid[:h, :w]
        mask = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * 10**2))
        heatmap += mask * self.INTENSITY

    def draw_heat(self, frame):
        heatmap = self.heatmap_visual

        # Normalizar
        norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        color = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)

        # 🔥 FIX CLAVE: resize al tamaño del frame
        color = cv2.resize(color, (frame.shape[1], frame.shape[0]))

        return cv2.addWeighted(frame, 0.7, color, 0.3, 0)

    def extract_hands(self, roi, x0, y0):
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)
        result = self.hands_model.detect(mp_image)

        pts = []
        if result.hand_landmarks:
            h, w = roi.shape[:2]
            for hand in result.hand_landmarks:
                for lm in hand:
                    px = int(lm.x * w) + x0
                    py = int(lm.y * h) + y0
                    pts.append((px, py))
        return pts

    # ─────────────────────────────
    # LOOP
    # ─────────────────────────────
    def run(self):
        while self.running:
            
            try:
                item = self.q.get(timeout=1)

                frame = item["color"]
                depth = item["depth"]
                cam_id = item["cam_id"]
                print("Inference recibiendo frame de:", item["cam_id"])
                if frame is None:
                    continue

                self.heatmap_visual *= self.DECAY_VISUAL

                # YOLO
                f = cv2.resize(frame, (640, 480))
                results = self.model.track(f, persist=True, device=self.device, verbose=False)

                if results[0].boxes is not None and len(results[0].boxes):

                    boxes  = results[0].boxes.xyxy.cpu().numpy()
                    confs  = results[0].boxes.conf.cpu().numpy()
                    clss   = results[0].boxes.cls.cpu().numpy().astype(int)
                    ids    = results[0].boxes.id

                    sx = frame.shape[1] / 640
                    sy = frame.shape[0] / 480

                    for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, clss)):
                        x1, y1, x2, y2 = box

                        x1, x2 = int(x1 * sx), int(x2 * sx)
                        y1, y2 = int(y1 * sy), int(y2 * sy)

                        label = self.model.names[cls_id]

                        if ids is not None:
                            tid = int(ids[i].item())
                            label = f"{label}#{tid}"

                        label = f"{label} {conf:.2f}"

                        # Bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                        # Label
                        cv2.putText(frame, label,
                                    (x1, max(y1-5,10)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4, (0,255,0), 1)

                        # Centro → heatmap
                        cx, cy = (x1+x2)//2, (y1+y2)//2
                        self.add_heat(self.heatmap_visual, cx, cy)


                # ───────── Heatmap overlay ─────────
                output = self.draw_heat(frame)

                # ───────── Ventana final ─────────
                cv2.imshow(f"INF_{cam_id}", output)

                if cv2.waitKey(1) == 27:
                    self.running = False

            except queue.Empty:
                continue

    def stop(self):
        self.running = False
        cv2.destroyAllWindows()