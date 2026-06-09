import queue
import time
import threading
import cv2
import torch
from collections import defaultdict
from camerathread.displaythread import DisplayThread
from camerathread.utils import get_realsense_devices, get_usb_cameras
from camerathread.realsensecamera import RealSenseCamera
from camerathread.camerathread import CameraThread
from camerathread.usbcamera import USBCamera
from camerathread.nicamera import OpenNICamera
from hoithread.environmentbuild import EnvironmentBuilderThread
from hoithread.fusion import FusionThread
from hoithread.inference import InferenceThread


def main():
    threads = []
    # ── Queues ────────────────────────────────────────────────────────
    # Cámaras → Inferencia (una queue por cámara para no mezclar frames)
    q_cameras = {}# {cam_id: Queue}

    # Inferencia → Fusión
    q_fusion = queue.Queue(maxsize=30)

    # Fusión → Display (ya la tenías)
    q_display = queue.Queue(maxsize=5)

     ##BUILDER
    q_env = queue.Queue(maxsize=10)
    env_builder = EnvironmentBuilderThread(q_env)
    env_builder.start()

    # ── Detectar cámaras y crear queues individuales ──────────────────
    cam_ids = []
    cam_depths = {} # {cam_id: bool} — tiene depth o no

    for serial in get_realsense_devices():
        cid = f"RS_{serial}"
        q_cameras[cid]= queue.Queue(maxsize=10)
        cam_ids.append(cid)
        cam_depths[cid]= True

    for idx in get_usb_cameras():
        cid = f"USB_{idx}"
        q_cameras[cid]= queue.Queue(maxsize=10)
        cam_ids.append(cid)
        cam_depths[cid]= False

    try:
        cid = "OPENNI"
        q_cameras[cid] = queue.Queue(maxsize=10)
        cam_ids.append(cid)
        cam_depths[cid] = True
    except Exception:
        print("OpenNI no disponible")

     # ── CameraThreads ─────────────────────────────────────────────────
     # Cada CameraThread ahora escribe en su queue individual
    # (cambia q → q_cameras[cam_id] en tu CameraThread)
    for serial in get_realsense_devices():
        cid = f"RS_{serial}"
        cam = RealSenseCamera(serial)
        t = CameraThread(cam, cid, q_cameras[cid], q_display)
        t.start()
        threads.append(t)

    for idx in get_usb_cameras():
        cid = f"USB_{idx}"
        cam = USBCamera(idx)
        t = CameraThread(cam, cid, q_cameras[cid], q_display)
        t.start()
        threads.append(t)
    try:
        cid = "OPENNI"
        cam = OpenNICamera()
        t = CameraThread(cam, cid, q_cameras[cid], q_display)
        t.start()
        threads.append(t)
    except Exception:
        print("OpenNI no disponible")
        cam_ids = [c for c in cam_ids if c != "OPENNI"]
        cam_depths.pop("OPENNI", None)
        q_cameras.pop("OPENNI", None)

    # ── InferenceThreads — uno por cámara ─────────────────────────────
    inf_threads = []
    # ── InferenceThreads — ahora reciben q_env y env_builder ──────────────
    for cid in cam_ids:
        inf = InferenceThread(cam_id= cid, in_queue= q_cameras[cid],fusion_queue = q_fusion,q_env= q_env,
                              has_depth= cam_depths[cid], env_builder= env_builder)
        inf.start()
        inf_threads.append(inf)

    # ── FusionThread ──────────────────────────────────────────────────
    fusion = FusionThread(
        fusion_queue  = q_fusion,
        display_queue = q_display,
        cam_ids       = cam_ids,
    )
    fusion.start()
    # fusion se gestiona separado del loop de cámaras en el shutdown

    # ── DisplayThread — recibe de q_display igual que antes ───────────
    display_thread = DisplayThread(q_display)
    display_thread.start()

    # ── Calibración opcional — llama después de que las cámaras estén ─
    # estabilizadas (puedes moverlo a un hilo separado o CLI)
    # _calibrate(fusion, cam_ids)
    # # ── Loop principal ────────────────────────────────────────────────
    try:
        while display_thread.running:
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("[INFO] Ctrl+C detectado")

    # ── Shutdown ordenado ─────────────────────────────────────────────
    print("[INFO] Cerrando sistema...")

    # 1. Detener cámaras primero — corta el flujo de datos
    for t in threads:
        t.stop()
    for t in threads:
        t.join(timeout=3)

    # 2. Detener inferencia — vacía lo que quedó en las queues
    for t in inf_threads:
        t.stop()
    for t in inf_threads:
        t.join(timeout=5)

    # 3. Detener env_builder — ya no recibirá datos nuevos
    env_builder.stop()
    env_builder.join(timeout=3)

    # 4. Detener fusión y display
    fusion.stop()
    display_thread.stop()
    fusion.join(timeout=2)
    display_thread.join(timeout=2)

    print("[INFO] Threads restantes:")
    for t in threading.enumerate():
        print(f"  {t.name}  daemon={t.daemon}  clase={type(t).__name__}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
