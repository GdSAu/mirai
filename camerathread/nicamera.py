
from .basecamera import BaseCamera
import cv2
import numpy as np
from openni import openni2

class OpenNICamera(BaseCamera):
    def __init__(self):
        openni2.initialize()

        self.dev = openni2.Device.open_any()

        self.rgb_stream = self.dev.create_color_stream()
        self.depth_stream = self.dev.create_depth_stream()
        # Registro depth → color
        if self.dev.is_image_registration_mode_supported(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR):
            self.dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
            print("Registration enabled")
        else:
            print("Registration not supported")

        # Configuración opcional
        self.rgb_stream.start()
        self.depth_stream.start()

        

    def read(self):
        # RGB
        rgb_frame = self.rgb_stream.read_frame()
        rgb = np.frombuffer(
            rgb_frame.get_buffer_as_uint8(),
            dtype=np.uint8
        ).reshape((rgb_frame.height, rgb_frame.width, 3))

        color = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Depth
        depth_frame = self.depth_stream.read_frame()
        depth = np.frombuffer(
            depth_frame.get_buffer_as_uint16(),
            dtype=np.uint16
        ).reshape((depth_frame.height, depth_frame.width))

        depth_uint8 = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

        return {
            "color": color,
            "depth": depth_colormap
        }

    def stop(self):
        self.rgb_stream.stop()
        self.depth_stream.stop()
        openni2.unload()