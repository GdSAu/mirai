"""
postprocess.py — postprocesamiento offline de sesiones grabadas.

Uso — una sesión:
    python postprocess.py recordings/session_A [--out output/] [--preview]

Uso — múltiples sesiones (mapa integrado):
    python postprocess.py recordings/session_A recordings/session_B [--out output/]

Las sesiones se procesan en el orden indicado. El FusionThread acumula
el heatmap a través de todas ellas, produciendo un único mapa integrado.
Los InferenceProcessors (tracker YOLO) se reinician entre sesiones.

Salida:
    output/
    ├── output_mosaic.mp4   video completo del mosaico + heatmap
    ├── hoi_stats.json      conteo acumulado de interacciones
    └── hoi_heatmap.png     imagen final del top-view

El script detecta automáticamente si fue interrumpido y reanuda desde
el último frame procesado (gracias a postprocess.lock).
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import yaml

# ── Dependencia opcional: tqdm ────────────────────────────────────────────────
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    print("[INFO] tqdm no instalado — sin barra de progreso. "
          "Instalar con: pip install tqdm")

from camerathread.filecamera import FileCamera
from hoithread.inference_processor import InferenceProcessor
from hoithread.fusion import FusionThread


# ──────────────────────────────────────────────────────────────────────────────
# Carga de sesión
# ──────────────────────────────────────────────────────────────────────────────

def load_session(session_dir: str) -> tuple[dict, dict, dict]:
    """
    Lee metadata.yaml y devuelve:
        meta     — dict completo del YAML
        cameras  — {cam_id: FileCamera}
        cam_info — {cam_id: dict}  (tipo, depth_scale, has_depth…)
    """
    meta_path = os.path.join(session_dir, "metadata.yaml")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"No se encontró metadata.yaml en: {session_dir}")

    with open(meta_path) as f:
        meta = yaml.safe_load(f)

    cameras  = {}
    cam_info = {}

    for cam_id, info in meta["cameras"].items():
        cam_dir = os.path.join(session_dir, cam_id)
        if not os.path.isdir(cam_dir):
            print(f"[WARN] Carpeta no encontrada, se omite: {cam_dir}")
            continue
        cameras[cam_id]  = FileCamera(cam_dir)
        cam_info[cam_id] = info

    if not cameras:
        raise RuntimeError("No se encontraron cámaras válidas en la sesión.")

    return meta, cameras, cam_info


def load_env_maps(session_dir: str, cam_info: dict) -> dict:
    """
    Carga mapas de background desde background/ si existen.
    Devuelve {cam_id: env_map_dict | None}.
    """
    bg_dir   = os.path.join(session_dir, "background")
    env_maps = {}

    for cam_id, info in cam_info.items():
        if not info.get("has_depth", False):
            env_maps[cam_id] = None
            continue
        bg_path  = os.path.join(bg_dir, f"{cam_id}.npy")
        rel_path = os.path.join(bg_dir, f"{cam_id}_rel.npy")
        if os.path.isfile(bg_path) and os.path.isfile(rel_path):
            background  = np.load(bg_path)
            reliability = np.load(rel_path)
            # Reconstruir surface_types a partir del background
            env_maps[cam_id] = {
                "background_depth": background,
                "reliability":      reliability,
                "surface_types":    _compute_surface_types(background),
                "cam_id":           cam_id,
            }
            print(f"[Post] mapa de background cargado: {cam_id}")
        else:
            env_maps[cam_id] = None
            print(f"[Post] sin mapa de background para {cam_id} "
                  f"— warm-up de {InferenceProcessor.N_FRAMES_CALIB} frames")

    return env_maps


def _compute_surface_types(background_depth: np.ndarray) -> np.ndarray:
    depth_m = background_depth / 1000.0
    depth_m[depth_m == 0] = np.nan
    grad_x   = cv2.Sobel(depth_m, cv2.CV_32F, 1, 0, ksize=5)
    grad_y   = cv2.Sobel(depth_m, cv2.CV_32F, 0, 1, ksize=5)
    norm_len = np.sqrt(grad_x**2 + grad_y**2 + 1)
    ny = -grad_y / norm_len
    nz =  1.0   / norm_len
    surface = np.zeros(depth_m.shape, dtype=np.uint8)
    surface[np.abs(ny) > 0.7] = 1
    surface[np.abs(nz) > 0.7] = 2
    return surface


# ──────────────────────────────────────────────────────────────────────────────
# Resume
# ──────────────────────────────────────────────────────────────────────────────

def _lock_path(out_dir: str) -> str:
    return os.path.join(out_dir, "postprocess.lock")


def read_resume_frame(out_dir: str) -> int:
    lock = _lock_path(out_dir)
    if os.path.isfile(lock):
        with open(lock) as f:
            content = f.read().strip()
        if content.isdigit():
            frame = int(content)
            print(f"[Post] Reanudando desde frame {frame}")
            return frame
    return 0


def write_resume_frame(out_dir: str, frame_idx: int):
    with open(_lock_path(out_dir), "w") as f:
        f.write(str(frame_idx))


def clear_resume(out_dir: str):
    lock = _lock_path(out_dir)
    if os.path.isfile(lock):
        os.remove(lock)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def _process_session(session_dir: str, fusion: "FusionThread",
                     out_dir: str, writer_ref: list,
                     fps_out: int, args,
                     global_frame_idx: int,
                     pbar) -> int:
    """
    Procesa una sesión completa acumulando en el FusionThread dado.
    Devuelve el número total de frames procesados en esta sesión.

    writer_ref es una lista de un elemento [writer|None] para poder
    inicializar el VideoWriter en la primera sesión y reutilizarlo.
    """
    print(f"\n[Post] ── Sesión: {session_dir} ──")
    meta, cameras, cam_info = load_session(session_dir)
    env_maps     = load_env_maps(session_dir, cam_info)
    cam_ids      = list(cameras.keys())
    total_frames = min(cam.total_frames for cam in cameras.values())
    print(f"[Post] Cámaras: {cam_ids}  |  Frames: {total_frames}")

    # Asegurar que el FusionThread conoce las cámaras de esta sesión
    for cid in cam_ids:
        if cid not in fusion.last_frames:
            fusion.last_frames[cid] = None
        if cid not in fusion.homographies:
            fusion.homographies[cid] = None

    # InferenceProcessors nuevos por sesión (tracker YOLO se reinicia)
    processors = {
        cam_id: InferenceProcessor(
            cam_id    = cam_id,
            has_depth = cam_info[cam_id].get("has_depth", False),
            env_map   = env_maps[cam_id],
        )
        for cam_id in cam_ids
    }

    prev_ts   = {cam_id: None for cam_id in cam_ids}
    frame_idx = 0

    try:
        while frame_idx < total_frames:
            frames = {}
            for cam_id, cam in cameras.items():
                data = cam.read()
                if data is None:
                    print(f"\n[Post] EOF en {cam_id} frame {frame_idx}")
                    return frame_idx
                frames[cam_id] = data

            for cam_id, data in frames.items():
                color = data["color"]
                depth = data.get("depth")
                ts_ns = data["timestamp_ns"]

                p_ts       = prev_ts[cam_id]
                virtual_dt = (ts_ns - p_ts) / 1e9 if p_ts is not None else 1/fps_out
                prev_ts[cam_id] = ts_ns

                result = processors[cam_id].process_frame(color, depth)

                fusion.fuse_frame(
                    cam_id          = cam_id,
                    frame_annotated = result["frame_annotated"],
                    contacts        = result["contacts"],
                    frame_shape     = color.shape[:2],
                    virtual_dt      = virtual_dt,
                )

            mosaic = fusion.build_mosaic()

            # Inicializar writer en el primer frame de la primera sesión
            if writer_ref[0] is None:
                mh, mw = mosaic.shape[:2]
                mosaic_path = os.path.join(out_dir, "output_mosaic.mp4")
                fourcc      = cv2.VideoWriter_fourcc(*"mp4v")
                writer_ref[0] = cv2.VideoWriter(mosaic_path, fourcc, fps_out, (mw, mh))
                print(f"[Post] Escribiendo mosaico: {mosaic_path} ({mw}×{mh})")

            writer_ref[0].write(mosaic)

            if args.preview:
                cv2.imshow("Postprocesamiento", mosaic)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n[Post] Preview cerrado.")
                    return frame_idx

            frame_idx += 1

            if (global_frame_idx + frame_idx) % 100 == 0:
                write_resume_frame(out_dir, global_frame_idx + frame_idx)

            if pbar:
                pbar.update(1)
            elif frame_idx % 50 == 0:
                pct = frame_idx / total_frames * 100
                print(f"[Post] {frame_idx}/{total_frames} ({pct:.1f}%)")

    finally:
        for proc in processors.values():
            proc.close()
        for cam in cameras.values():
            cam.stop()

    return frame_idx


def main():
    parser = argparse.ArgumentParser(
        description="Postprocesamiento offline de sesiones MIRAI"
    )
    parser.add_argument(
        "session_dirs", nargs="+",
        help="Una o más rutas de sesión grabada (contienen metadata.yaml). "
             "Si se pasan varias, el heatmap se acumula a través de todas.",
    )
    parser.add_argument("--out", default="output",
                        help="Directorio de salida (default: output/)")
    parser.add_argument("--preview", action="store_true",
                        help="Mostrar mosaico en tiempo real mientras procesa")
    args = parser.parse_args()

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    # ── Calcular total de frames para la barra de progreso ────────────────
    total_all = 0
    for sd in args.session_dirs:
        try:
            _, cams, _ = load_session(sd)
            total_all += min(c.total_frames for c in cams.values())
            for c in cams.values():
                c.stop()
        except Exception as e:
            print(f"[WARN] No se pudo leer {sd}: {e}")

    # ── Resume global (por si fue interrumpido en multi-sesión) ──────────
    start_frame = read_resume_frame(out_dir)
    if start_frame > 0:
        print(f"[Post] Reanudando desde frame global {start_frame} "
              f"— avanzando sesiones...")

    # ── FusionThread compartido entre todas las sesiones ──────────────────
    import queue as _queue
    # Descubrir cam_ids de la primera sesión para inicializar FusionThread
    first_meta, _, first_cam_info = load_session(args.session_dirs[0])
    first_cam_ids = list(first_cam_info.keys())
    fps_out       = first_meta["cameras"][first_cam_ids[0]].get("fps", 30)

    fusion = FusionThread(
        fusion_queue  = _queue.Queue(),
        display_queue = _queue.Queue(),
        cam_ids       = first_cam_ids,
    )

    # ── Barra de progreso ─────────────────────────────────────────────────
    if _HAS_TQDM:
        pbar = tqdm(total=total_all - start_frame, desc="Postprocesando",
                    unit="frame")
    else:
        pbar = None

    writer_ref        = [None]   # lista mutable para pasar por referencia
    global_frame_idx  = 0
    frames_to_skip    = start_frame

    try:
        for i, session_dir in enumerate(args.session_dirs):
            print(f"\n[Post] Sesión {i+1}/{len(args.session_dirs)}: {session_dir}")

            # Si esta sesión ya fue completada en una ejecución anterior, saltarla
            if frames_to_skip > 0:
                try:
                    _, cams, _ = load_session(session_dir)
                    sess_frames = min(c.total_frames for c in cams.values())
                    for c in cams.values():
                        c.stop()
                except Exception:
                    sess_frames = 0

                if frames_to_skip >= sess_frames:
                    print(f"[Post] Sesión ya procesada, omitiendo.")
                    global_frame_idx += sess_frames
                    frames_to_skip   -= sess_frames
                    continue
                else:
                    # Sesión parcialmente procesada — no implementamos skip
                    # parcial por simplicidad; reprocesamos desde el inicio
                    frames_to_skip = 0

            processed = _process_session(
                session_dir     = session_dir,
                fusion          = fusion,
                out_dir         = out_dir,
                writer_ref      = writer_ref,
                fps_out         = fps_out,
                args            = args,
                global_frame_idx= global_frame_idx,
                pbar            = pbar,
            )
            global_frame_idx += processed

    except KeyboardInterrupt:
        print(f"\n[Post] Interrumpido en frame global {global_frame_idx}.")
        write_resume_frame(out_dir, global_frame_idx)

    finally:
        if pbar:
            pbar.close()
        if writer_ref[0]:
            writer_ref[0].release()
        if args.preview:
            cv2.destroyAllWindows()

    # ── Exportar estadísticas finales (todas las sesiones acumuladas) ─────
    if global_frame_idx >= total_all:
        clear_resume(out_dir)

    stats_path = os.path.join(out_dir, "hoi_stats.json")
    with open(stats_path, "w") as f:
        json.dump(dict(fusion.hoi_counts), f, indent=2)
    print(f"\n[Post] Estadísticas → {stats_path}")

    heatmap_path = os.path.join(out_dir, "hoi_heatmap.png")
    cv2.imwrite(heatmap_path, fusion.get_top_view())
    print(f"[Post] Heatmap integrado → {heatmap_path}")

    n = len(args.session_dirs)
    print(f"\n[Post] Completado: {global_frame_idx} frames de {n} sesión(es).")
    print(f"[Post] Salida en:  {out_dir}/")


if __name__ == "__main__":
    main()
