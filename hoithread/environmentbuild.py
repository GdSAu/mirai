import threading
import queue
import numpy as np
import cv2

class EnvironmentBuilderThread(threading.Thread):
    """
    Thread separado que construye el mapa del entorno automáticamente.
    - Recibe frames de TODAS las cámaras RGBD via q_env
    - Acumula solo cuando no hay personas en escena
    - Publica env_map listo a cada InferenceThread via callbacks
    - Actualiza el fondo incrementalmente durante la sesión
    """

    N_FRAMES_CALIB  = 60      # frames para construir el mapa base
    ALPHA_UPDATE    = 0.002   # velocidad de actualización incremental
    MAX_WAIT_SECS   = 60      # tiempo máximo esperando escena vacía

    def __init__(self, q_env: queue.Queue):
        super().__init__(daemon=True)
        self.q_env   = q_env
        self.running = True

        # Estado por cámara RGBD
        # { cam_id: { "buffer": [], "map": {...}, "ready": bool } }
        self._state  = {}
        self._lock   = threading.Lock()

        # Callbacks registrados por cada InferenceThread
        # { cam_id: callable(env_map) }
        self._callbacks = {}

        print("[EnvBuilder] iniciado")

    # ──────────────────────────────────────────────────────────────────
    # API PÚBLICA
    # ──────────────────────────────────────────────────────────────────
    def register_camera(self, cam_id: str, callback):
        """
        Registra una cámara RGBD y el callback que se llama
        cuando su mapa esté listo.
        callback: fn(env_map: dict) → None
        """
        with self._lock:
            self._state[cam_id] = {
                "buffer":   [],
                "map":      None,
                "ready":    False,
                "started":  False,
            }
            self._callbacks[cam_id] = callback
        print(f"[EnvBuilder] cámara registrada: {cam_id}")

    def get_map(self, cam_id: str) -> dict:
        """Retorna el mapa actual de una cámara (None si no está listo)."""
        with self._lock:
            state = self._state.get(cam_id)
            return state["map"] if state else None

    def is_ready(self, cam_id: str) -> bool:
        with self._lock:
            state = self._state.get(cam_id)
            return state["ready"] if state else False

    def update_background(self, cam_id: str, depth_data: np.ndarray,
                      contact_mask: np.ndarray):
        with self._lock:
            state = self._state.get(cam_id)
            if state is None or not state["ready"]:
                return
            
            # ── FIX: normalizar depth a 2D ────────────────────────────
            if depth_data.ndim == 3:
                depth_data = depth_data[:, :, 0]
            
            bg    = state["map"]["background_depth"]
            curr  = depth_data.astype(np.float32)
            valid = (curr > 0) & ~contact_mask
            bg[valid] = ((1 - self.ALPHA_UPDATE) * bg[valid] +
                        self.ALPHA_UPDATE * curr[valid])

    # ──────────────────────────────────────────────────────────────────
    # CONSTRUCCIÓN DEL MAPA
    # ──────────────────────────────────────────────────────────────────
    def _build_map(self, cam_id: str, buffer: list) -> dict:
        normalized = []
        for frame in buffer:
            if frame.ndim == 3:
                frame = frame[:, :, 0]
            normalized.append(frame)
        
        stack = np.stack(normalized, axis=0).astype(np.float32)

        background = np.median(stack, axis=0)
        std        = np.std(stack, axis=0)
        zero_ratio = (stack == 0).mean(axis=0)

        # Confiabilidad: 1.0 = muy confiable, 0.0 = no usar
        reliability = (1.0 - np.clip(std / 100.0, 0, 1))
        reliability *= (1.0 - np.clip(zero_ratio / 0.3, 0, 1))

        surface_types = self._compute_surface_types(background)

        env_map = {
            "background_depth": background,
            "reliability":      reliability.astype(np.float32),
            "surface_types":    surface_types,
            "cam_id":           cam_id,
        }

        print(f"[EnvBuilder:{cam_id}] mapa construido ✓")
        return env_map

    def _compute_surface_types(self, background_depth: np.ndarray) -> np.ndarray:
        """
        Clasifica cada pixel como horizontal/vertical/unknown
        usando las normales locales del depth.
        """
        depth_m  = background_depth / 1000.0
        depth_m[depth_m == 0] = np.nan

        grad_x   = cv2.Sobel(depth_m, cv2.CV_32F, 1, 0, ksize=5)
        grad_y   = cv2.Sobel(depth_m, cv2.CV_32F, 0, 1, ksize=5)
        norm_len = np.sqrt(grad_x**2 + grad_y**2 + 1)

        ny = -grad_y / norm_len   # componente Y de la normal
        nz =  1.0   / norm_len   # componente Z de la normal

        # 0=unknown, 1=horizontal, 2=vertical
        surface = np.zeros(depth_m.shape, dtype=np.uint8)
        surface[np.abs(ny) > 0.7] = 1   # normal apunta arriba → mesa/suelo
        surface[np.abs(nz) > 0.7] = 2   # normal apunta al frente → pared/pizarrón

        return surface

    # ──────────────────────────────────────────────────────────────────
    # LOOP PRINCIPAL
    # ──────────────────────────────────────────────────────────────────
    def run(self):
        try:
            while self.running:
                try:
                    item       = self.q_env.get(timeout=1)
                    cam_id     = item["cam_id"]
                    depth_data = item.get("depth")
                    has_person = item.get("has_person", False)

                    if depth_data is None:
                        continue

                    with self._lock:
                        state = self._state.get(cam_id)
                        if state is None or state["ready"]:
                            continue

                    # ── Esperar escena vacía ───────────────────────────────
                    if has_person:
                        with self._lock:
                            state["buffer"] = []   # resetear si hay persona
                        continue

                    # ── Acumular frames ────────────────────────────────────
                    with self._lock:
                        state["buffer"].append(depth_data.copy())
                        n       = len(state["buffer"])
                        total   = self.N_FRAMES_CALIB

                    print(f"[EnvBuilder:{cam_id}] {n}/{total} frames", end="\r")

                    # ── Construir mapa cuando hay suficientes frames ────────
                    if n >= total:
                        with self._lock:
                            env_map          = self._build_map(cam_id, state["buffer"])
                            state["map"]     = env_map
                            state["ready"]   = True
                            state["buffer"]  = []   # liberar memoria

                        # Notificar al InferenceThread correspondiente
                        cb = self._callbacks.get(cam_id)
                        if cb:
                            cb(env_map)

                except queue.Empty:
                    continue
        finally:
            self._cleanup()
    
    def _cleanup(self):
        with self._lock:
            self._state.clear()
            self._callbacks.clear()
        print("[EnvBuilder] memoria liberada")

    def stop(self):
        self.running = False
