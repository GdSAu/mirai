"""
camerathread/filecamera.py — lector de sesiones grabadas.

Simula una cámara física leyendo los archivos producidos por record.py.
La alineación depth/color se hace por timestamp (nombre del PNG),
no por índice secuencial, para tolerar frames de video descartados.
"""

import os
import cv2
import numpy as np
import csv

from .basecamera import BaseCamera


class FileCamera(BaseCamera):
    """
    Lee una carpeta de sesión grabada:
        cam_dir/
        ├── color.mp4
        ├── timestamps.csv   (frame_idx, timestamp_ns)
        └── depth/
            └── <timestamp_ns>.png   (uint16 mm)

    read() devuelve el mismo dict que RealSenseCamera:
        {"color": BGR, "depth": uint16|None, "depth_viz": BGR|None,
         "timestamp_ns": int}

    Retorna None cuando se llega al final del video (EOF limpio).
    """

    def __init__(self, cam_dir: str):
        color_path = os.path.join(cam_dir, "color.mp4")
        ts_path    = os.path.join(cam_dir, "timestamps.csv")
        depth_dir  = os.path.join(cam_dir, "depth")

        if not os.path.isfile(color_path):
            raise FileNotFoundError(f"No se encontró: {color_path}")
        if not os.path.isfile(ts_path):
            raise FileNotFoundError(f"No se encontró: {ts_path}")

        self._cap        = cv2.VideoCapture(color_path)
        self._depth_dir  = depth_dir if os.path.isdir(depth_dir) else None
        self._timestamps = self._load_timestamps(ts_path)
        self._frame_idx  = 0

    # ── interfaz pública ──────────────────────────────────────────────────

    @property
    def total_frames(self) -> int:
        return len(self._timestamps)

    def read(self) -> dict | None:
        """
        Lee el siguiente frame. Devuelve None en EOF o si el video falla.
        El campo 'depth' es None si no hay carpeta depth o falta el PNG
        para ese timestamp — sin contagiar frames siguientes.
        """
        if self._frame_idx >= len(self._timestamps):
            return None

        ret, color = self._cap.read()
        if not ret:
            return None

        ts_ns = self._timestamps[self._frame_idx]
        self._frame_idx += 1

        depth       = None
        depth_viz   = None

        if self._depth_dir is not None:
            depth_path = os.path.join(self._depth_dir, f"{ts_ns}.png")
            if os.path.isfile(depth_path):
                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)   # uint16 mm
                if depth is not None:
                    depth_uint8 = cv2.normalize(
                        depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
                    )
                    depth_viz = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

        return {
            "color":        color,
            "depth":        depth,
            "depth_viz":    depth_viz,
            "timestamp_ns": ts_ns,
        }

    def stop(self):
        self._cap.release()

    # ── interno ───────────────────────────────────────────────────────────

    @staticmethod
    def _load_timestamps(ts_path: str) -> list:
        timestamps = []
        with open(ts_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamps.append(int(row["timestamp_ns"]))
        return timestamps
