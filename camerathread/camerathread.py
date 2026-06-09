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
        try:
            while self.running:
                data = self.cam.read()

                if data is None:
                    continue

                if isinstance(data, dict):
                    color     = data["color"]
                    depth     = data.get("depth", None)      # uint16 mm (RealSense/OpenNI)
                    depth_viz = data.get("depth_viz", None)  # BGR colormap para display
                else:
                    color     = data
                    depth     = None
                    depth_viz = None

                payload = {
                    "cam_id":       self.cam_id,
                    "color":        color,
                    "depth":        depth,
                    "depth_viz":    depth_viz,
                    "timestamp_ns": time.time_ns(),
                }
                if not self.q_infer.full():
                    self.q_infer.put(payload)

                if not self.q_display.full():
                    self.q_display.put(payload)
        finally:
            # pipeline.stop() se llama desde el propio thread para evitar deadlock
            self.cam.stop()

    def stop(self):
        self.running = False