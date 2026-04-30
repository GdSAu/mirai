import threading
import time

class CameraThread(threading.Thread):
    def __init__(self, cam, cam_id, q_infer, q_display):
        super().__init__()
        self.cam = cam
        self.cam_id = cam_id
        self.q_infer = q_infer
        self.q_display = q_display
        self.running = True

    def run(self):
        while self.running:
            data = self.cam.read()

            if data is None:
                continue

            # 🔥 NORMALIZACIÓN DE FORMATO
            if isinstance(data, dict):
                color = data["color"]
                depth = data.get("depth", None)
            else:
                # USB / OpenCV
                color = data
                depth = None

            payload = {
                "cam_id": self.cam_id,
                "color": color,
                "depth": depth
            }
            # Inferencia (prioridad tiempo real)
            if not self.q_infer.full():
                self.q_infer.put(payload)

            # Visualización (puede perder frames sin problema)
            if not self.q_display.full():
                self.q_display.put(payload)
            
            if not self.running:
                break

    def stop(self):
        self.running = False
        self.cam.stop()