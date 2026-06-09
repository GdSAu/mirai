import threading
import cv2
import queue
import sys
import numpy as np
class DisplayThread(threading.Thread):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        self.running = True
        self.frames = {}  # 🔥 buffer por cámara

    def run(self):
        while self.running:
            try:
                # consumir TODO lo disponible (no solo 1)
                while True:
                    item = self.queue.get_nowait()

                    cam_id    = item["cam_id"]
                    color     = item["color"]
                    depth_viz = item.get("depth_viz", None)

                    self.frames[cam_id] = (color, depth_viz)

            except queue.Empty:
                pass

            # ───────── MOSTRAR TODAS ─────────
            for cam_id, (color, depth) in self.frames.items():

                if color is not None:
                    cv2.imshow(f"RAW_{cam_id}", color)

                if depth is not None:
                    #depth_vis = self.process_depth(depth)
                    cv2.imshow(f"DEPTH_{cam_id}", depth)

            if cv2.waitKey(1) == 27:
                self.running = False
                cv2.destroyAllWindows()
                sys.exit(0)

    def process_depth(self, depth):
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_norm.astype(np.uint8)
        return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

    def stop(self):
        self.running = False
        cv2.destroyAllWindows()