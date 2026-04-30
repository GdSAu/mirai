import cv2
import sys
import time
import threading
import queue
from thread.displaythread import DisplayThread
from thread.utils import get_realsense_devices, get_usb_cameras
from thread.realsensecamera import RealSenseCamera
from thread.camerathread import CameraThread
from thread.usbcamera import USBCamera
from thread.nicamera import OpenNICamera

# MAIN
# =========================
def main():
    threads = []
    q = queue.Queue(maxsize=10)
    q_display = queue.Queue(maxsize=5)
    display_thread = DisplayThread(q)
    display_thread.start()
    # RealSense
    
    for serial in get_realsense_devices():
        cam = RealSenseCamera(serial)
        t = CameraThread(cam, f"RS_{serial}",q, q_display)
        t.start()
        threads.append(t)

    # USB
    for idx in get_usb_cameras():
        cam = USBCamera(idx)
        t = CameraThread(cam, f"USB_{idx}",q, q_display)
        t.start()
        threads.append(t)

    # OpenNI (opcional)
    try:
        cam = OpenNICamera()
        t = CameraThread(cam, "OPENNI",q, q_display)
        t.start()
        threads.append(t)
    except:
        print("OpenNI no disponible")

    try:
        while display_thread.running:
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("[INFO] Ctrl+C detectado")

    print("[INFO] Cerrando sistema...")

    for t in threads:
        t.stop()

    display_thread.stop()

    for t in threads:
        t.join(timeout=2)

    display_thread.join(timeout=2)

    print("[INFO] Threads restantes:")
    for t in threading.enumerate():
        print(t)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()