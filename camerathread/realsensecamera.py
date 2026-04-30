from .basecamera import BaseCamera
import pyrealsense2 as rs
import cv2
import numpy as np

class RealSenseCamera(BaseCamera):
    def __init__(self, serial):
        self.serial = serial
        self.pipeline = rs.pipeline()
        config = rs.config()

        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.pipeline.start(config)

    def read(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame:
            return None
        # Convertir a numpy
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Normalizar depth para visualización
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )
        return {
            "color": color_image,
            "depth": depth_colormap
        }
    def stop(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()
