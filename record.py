"""
record.py — grabación multicámara a disco.

Uso:
    python record.py [--dir recordings] [--preview]

Salida (por sesión):
    recordings/session_YYYYMMDD_HHMMSS/
    ├── metadata.yaml
    ├── cam_RS_<serial>/
    │   ├── color.mp4
    │   ├── timestamps.csv
    │   └── depth/<timestamp_ns>.png
    └── cam_USB_<idx>/
        ├── color.mp4
        └── timestamps.csv

Detener: Ctrl+C  (escribe metadata.yaml al finalizar)
"""

import argparse
import os
import queue
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import yaml

from camerathread.camerathread import CameraThread
from camerathread.realsensecamera import RealSenseCamera
from camerathread.usbcamera import USBCamera
from camerathread.utils import get_realsense_devices, get_usb_cameras


# ──────────────────────────────────────────────────────────────────────────────
# RecordThread
# ──────────────────────────────────────────────────────────────────────────────

class RecordThread(threading.Thread):
    """
    Consume payloads de una queue y los escribe a disco:
      - color.mp4        (H.264 si disponible, mp4v como fallback)
      - timestamps.csv   (frame_idx, timestamp_ns)
      - depth/<ts_ns>.png  (uint16 mm, solo cámaras con depth)

    La queue tiene maxsize=60 (≈ 2 s a 30 fps). Si CameraThread la
    encuentra llena, descarta el frame silenciosamente — el hueco
    queda reflejado en timestamps.csv.
    """

    QUEUE_MAXSIZE = 60

    def __init__(self, cam_id: str, cam_dir: str,
                 fps: int, resolution: tuple, has_depth: bool):
        super().__init__(daemon=True, name=f"RecordThread-{cam_id}")
        self.cam_id     = cam_id
        self.resolution = resolution   # (W, H)
        self.has_depth  = has_depth
        self.running    = True
        self._written   = 0
        self._dropped   = 0

        os.makedirs(cam_dir, exist_ok=True)
        self._depth_dir = os.path.join(cam_dir, "depth")
        if has_depth:
            os.makedirs(self._depth_dir, exist_ok=True)

        self._ts_file = open(os.path.join(cam_dir, "timestamps.csv"), "w")
        self._ts_file.write("frame_idx,timestamp_ns\n")

        # Intentar H.264, fallback a mp4v
        video_path = os.path.join(cam_dir, "color.mp4")
        self.codec = "avc1"
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        self._writer = cv2.VideoWriter(video_path, fourcc, fps, resolution)
        if not self._writer.isOpened():
            self._writer.release()   # liberar handle antes de reintentar
            self.codec = "mp4v"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(video_path, fourcc, fps, resolution)

        # Queue pública — CameraThread escribe aquí (como q_infer)
        self.q = queue.Queue(maxsize=self.QUEUE_MAXSIZE)

        print(f"[Record:{cam_id}] codec={self.codec}  salida={cam_dir}")

    # ── propiedades ───────────────────────────────────────────────────────────

    @property
    def written(self) -> int:
        return self._written

    @property
    def dropped(self) -> int:
        return self._dropped

    # ── hilo ──────────────────────────────────────────────────────────────────

    def run(self):
        while self.running or not self.q.empty():
            try:
                payload = self.q.get(timeout=0.5)
            except queue.Empty:
                continue

            color = payload["color"]
            depth = payload.get("depth")
            ts_ns = payload.get("timestamp_ns", time.time_ns())

            # Resize si el frame llega con resolución distinta a la configurada
            h, w = color.shape[:2]
            if (w, h) != self.resolution:
                color = cv2.resize(color, self.resolution)

            self._writer.write(color)
            self._ts_file.write(f"{self._written},{ts_ns}\n")

            if depth is not None and self.has_depth:
                cv2.imwrite(
                    os.path.join(self._depth_dir, f"{ts_ns}.png"),
                    depth,
                )

            self._written += 1

        self._cleanup()

    def stop(self):
        self.running = False

    def _cleanup(self):
        self._writer.release()
        self._ts_file.flush()
        self._ts_file.close()
        print(
            f"[Record:{self.cam_id}] cerrado — "
            f"{self._written} escritos, {self._dropped} descartados"
        )


# ──────────────────────────────────────────────────────────────────────────────
# PreviewThread  (solo activo con --preview)
# ──────────────────────────────────────────────────────────────────────────────

class PreviewThread(threading.Thread):
    """Muestra un mosaico simple con el último frame de cada cámara."""

    def __init__(self, q_display: queue.Queue):
        super().__init__(daemon=True, name="PreviewThread")
        self.q       = q_display
        self.running = True
        self._frames = {}   # {cam_id: np.ndarray}

    def run(self):
        while self.running:
            # Vaciar todo lo disponible en la queue
            try:
                while True:
                    payload = self.q.get_nowait()
                    self._frames[payload["cam_id"]] = payload["color"]
            except queue.Empty:
                pass

            if self._frames:
                tiles  = [cv2.resize(f, (320, 240)) for f in self._frames.values()]
                mosaic = np.hstack(tiles)
                cv2.imshow("Grabando  —  q para salir", mosaic)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.running = False

            time.sleep(0.03)   # ~30 fps de refresco en preview

    def stop(self):
        self.running = False


# ──────────────────────────────────────────────────────────────────────────────
# Metadata
# ──────────────────────────────────────────────────────────────────────────────

def _save_metadata(session_dir: str, session_id: str,
                   cam_meta: dict, started_at: str) -> None:
    meta = {
        "session_id":       session_id,
        "recorded_at":      started_at,
        "cameras":          cam_meta,
        "background_saved": False,
    }
    path = os.path.join(session_dir, "metadata.yaml")
    with open(path, "w") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)
    print(f"[Record] metadata → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Grabación multicámara MIRAI")
    parser.add_argument(
        "--dir", default="recordings",
        help="Directorio raíz de sesiones (default: recordings/)",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Mostrar ventana de preview en tiempo real",
    )
    parser.add_argument(
        "--no-usb", action="store_true",
        help="Ignorar cámaras USB (solo grabar RealSense)",
    )
    args = parser.parse_args()

    now         = datetime.now()
    session_id  = f"session_{now.strftime('%Y%m%d_%H%M%S')}"
    session_dir = os.path.join(args.dir, session_id)
    os.makedirs(session_dir, exist_ok=True)
    print(f"[Record] Sesión: {session_dir}")

    threads        = []
    record_threads = {}
    cam_meta       = {}
    q_display      = queue.Queue(maxsize=10)

    # ── RealSense ─────────────────────────────────────────────────────────────
    for serial in get_realsense_devices():
        cid = f"RS_{serial}"
        cam = RealSenseCamera(serial)
        rec = RecordThread(
            cid,
            os.path.join(session_dir, cid),
            fps=30, resolution=(640, 480), has_depth=True,
        )
        rec.start()
        record_threads[cid] = rec

        ct = CameraThread(cam, cid, rec.q, q_display)
        ct.start()
        threads.append(ct)

        cam_meta[cid] = {
            "type":        "realsense",
            "serial":      serial,
            "fps":         30,
            "resolution":  [640, 480],
            "depth_scale": 0.001,
            "has_depth":   True,
            "homography":  None,
            "video_codec": rec.codec,
        }

    # ── USB ───────────────────────────────────────────────────────────────────
    for idx in ([] if args.no_usb else get_usb_cameras()):
        cid = f"USB_{idx}"
        cam = USBCamera(idx)
        rec = RecordThread(
            cid,
            os.path.join(session_dir, cid),
            fps=30, resolution=(640, 480), has_depth=False,
        )
        rec.start()
        record_threads[cid] = rec

        ct = CameraThread(cam, cid, rec.q, q_display)
        ct.start()
        threads.append(ct)

        cam_meta[cid] = {
            "type":        "usb",
            "index":       idx,
            "fps":         30,
            "resolution":  [640, 480],
            "depth_scale": None,
            "has_depth":   False,
            "homography":  None,
            "video_codec": rec.codec,
        }

    if not threads:
        print("[Record] No se detectaron cámaras. Saliendo.")
        return

    # ── Preview opcional ──────────────────────────────────────────────────────
    preview = None
    if args.preview:
        preview = PreviewThread(q_display)
        preview.start()

    # ── Loop principal ────────────────────────────────────────────────────────
    print("[Record] Grabando... (Ctrl+C para detener)")
    try:
        while True:
            time.sleep(5)
            total   = sum(r.written  for r in record_threads.values())
            dropped = sum(r.dropped  for r in record_threads.values())
            print(f"[Record] frames: {total}  descartados: {dropped}")
    except KeyboardInterrupt:
        print("\n[Record] Ctrl+C — cerrando...")

    # ── Shutdown ──────────────────────────────────────────────────────────────
    # 1. Cortar flujo de captura
    for t in threads:
        t.stop()
    for t in threads:
        t.join(timeout=3)

    # 2. Drenar y cerrar writers (timeout generoso para disco lento)
    for rec in record_threads.values():
        rec.stop()
    for rec in record_threads.values():
        rec.join(timeout=15)

    # 3. Preview
    if preview:
        preview.stop()
        preview.join(timeout=2)
        cv2.destroyAllWindows()

    # 4. Metadata
    _save_metadata(session_dir, session_id, cam_meta,
                   started_at=now.isoformat())

    print(f"\n[Record] Grabación guardada en: {session_dir}")


if __name__ == "__main__":
    main()
