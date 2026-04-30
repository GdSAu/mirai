import cv2 
from .basecamera import BaseCamera

class USBCamera(BaseCamera):
    def __init__(self, index):
        self.index = index
        self.cap = cv2.VideoCapture(index)

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def stop(self):
        self.cap.release()