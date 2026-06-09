import threading
import queue
import numpy as np
from collections import defaultdict
import cv2

class FusionThread(threading.Thread):
    """
    Recibe payloads de todos los InferenceThreads y mantiene:

    1. heatmap_accum  — presencia acumulativa por cámara, proyectada al
                        plano canónico via homografía (o suma directa
                        si no hay homografía calibrada aún).

    2. hoi_accum      — mapa acumulativo de interacciones por objeto.

    3. top_view       — imagen 2D top-down renderizable en tiempo real.

    Calibración por homografía:
    - Llama a fusion.set_homography(cam_id, H) con una matriz 3×3
      calculada externamente (4 puntos del suelo son suficientes).
    - Sin homografía, la cámara suma en su propio espacio escalado.
    """

    GRID_W   = 400          # píxeles del canvas canónico
    GRID_H   = 400
    INTENSITY = 30
    DECAY_HOI = 0.9995      # muy lento — casi permanente

    def __init__(self, fusion_queue: queue.Queue,
             display_queue: queue.Queue,   # ← agregar este
             cam_ids: list):
        super().__init__(daemon=True)
        self.fusion_queue  = fusion_queue
        self.display_queue = display_queue    # ← y este
        self.cam_ids       = cam_ids
        self.running       = True

        # Homografías por cámara (None = sin calibrar)
        self.homographies = {cid: None for cid in cam_ids}
        self._H_lock      = threading.Lock()

        # ── Heatmaps canónicos ────────────────────────────────────────
        self.heatmap_presence = np.zeros((self.GRID_H, self.GRID_W), dtype=np.float32)
        self.heatmap_hoi      = defaultdict(
            lambda: np.zeros((self.GRID_H, self.GRID_W), dtype=np.float32)
        )

        # Resumen de interacciones globales
        self.hoi_counts  = defaultdict(int)   # {obj_name: total_eventos}
        self.hoi_by_cam  = defaultdict(lambda: defaultdict(int))

        # Último frame anotado por cámara (para mosaic)
        self.last_frames = {cid: None for cid in cam_ids}
        self._frame_lock = threading.Lock()

        # Top-view renderizado (actualizado cada ciclo)
        self.top_view    = np.zeros((self.GRID_H, self.GRID_W, 3), dtype=np.uint8)
        self._tv_lock    = threading.Lock()

        print(f"[FusionThread] listo — cámaras: {cam_ids}")

    # ──────────────────────────────────────────────────────────────────
    # API PÚBLICA — calibración
    # ──────────────────────────────────────────────────────────────────
    def set_homography(self, cam_id: str, H: np.ndarray):
        """
        Registra la homografía de cam_id al plano canónico.
        H es una matriz 3×3 obtenida con cv2.findHomography(
            pts_camara, pts_canonico
        ) donde pts_canonico son coordenadas en el canvas GRID_W×GRID_H.

        Ejemplo mínimo de calibración:
            pts_cam = np.float32([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
            pts_can = np.float32([[u1,v1],[u2,v2],[u3,v3],[u4,v4]])
            H, _    = cv2.findHomography(pts_cam, pts_can)
            fusion.set_homography("cam_0", H)
        """
        with self._H_lock:
            self.homographies[cam_id] = H
        print(f"[FusionThread] homografía de {cam_id} registrada")

    def get_top_view(self) -> np.ndarray:
        with self._tv_lock:
            return self.top_view.copy()

    def get_hoi_summary(self) -> dict:
        return dict(self.hoi_counts)

    # ──────────────────────────────────────────────────────────────────
    # UTILS
    # ──────────────────────────────────────────────────────────────────
    def _project_point(self, cam_id, px, py, src_w, src_h):
        """
        Proyecta (px,py) del espacio de cam_id al canvas canónico.
        Si hay homografía la usa; si no, escala directamente.
        """
        with self._H_lock:
            H = self.homographies.get(cam_id)

        if H is not None:
            pt  = np.array([[[float(px), float(py)]]], dtype=np.float32)
            dst = cv2.perspectiveTransform(pt, H)
            gx, gy = int(dst[0,0,0]), int(dst[0,0,1])
        else:
            # Fallback: escalar al canvas (sin corrección de perspectiva)
            gx = int(px / src_w * self.GRID_W)
            gy = int(py / src_h * self.GRID_H)

        gx = max(0, min(self.GRID_W-1, gx))
        gy = max(0, min(self.GRID_H-1, gy))
        return gx, gy

    def _add_heat(self, heatmap, gx, gy, radius=12, intensity=None):
        intensity = intensity or self.INTENSITY
        h, w = heatmap.shape
        y, x = np.ogrid[:h, :w]
        mask  = np.exp(-((x-gx)**2 + (y-gy)**2) / (2*(radius/2)**2))
        heatmap += mask * intensity

    def _render_top_view(self):
        """Renderiza el canvas canónico con ambos heatmaps + etiquetas HOI."""
        # Presencia (azul→rojo)
        norm_p = cv2.normalize(self.heatmap_presence, None, 0, 255, cv2.NORM_MINMAX)
        canvas = cv2.applyColorMap(norm_p.astype(np.uint8), cv2.COLORMAP_INFERNO)

        # HOI acumulado — overlay en amarillo-verde
        hoi_combined = np.zeros((self.GRID_H, self.GRID_W), dtype=np.float32)
        for h in self.heatmap_hoi.values():
            hoi_combined += h

        if hoi_combined.max() > 0:
            norm_h  = cv2.normalize(hoi_combined, None, 0, 200, cv2.NORM_MINMAX)
            hoi_col = cv2.applyColorMap(norm_h.astype(np.uint8), cv2.COLORMAP_SUMMER)
            mask    = (norm_h > 10).astype(np.uint8)
            hoi_col = cv2.bitwise_and(hoi_col, hoi_col, mask=mask)
            canvas  = cv2.addWeighted(canvas, 0.6, hoi_col, 0.4, 0)

        # Grilla de referencia
        step = self.GRID_W // 4
        for i in range(1, 4):
            cv2.line(canvas, (i*step, 0), (i*step, self.GRID_H), (40,40,40), 1)
            cv2.line(canvas, (0, i*step), (self.GRID_W, i*step), (40,40,40), 1)

        # Top HOI como texto
        top_hoi = sorted(self.hoi_counts.items(), key=lambda x: -x[1])[:5]
        for i, (name, cnt) in enumerate(top_hoi):
            cv2.putText(canvas, f"{name}: {cnt}", (6, 16 + 14*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,255,200), 1)

        cv2.putText(canvas, "TOP-VIEW FUSIONADO", (self.GRID_W//2 - 70, self.GRID_H - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1)

        with self._tv_lock:
            self.top_view = canvas

    # ──────────────────────────────────────────────────────────────────
    # LOOP PRINCIPAL
    # ──────────────────────────────────────────────────────────────────
    def run(self):
        while self.running:
            try:
                payload  = self.fusion_queue.get(timeout=1)
                cam_id   = payload["cam_id"]
                src_h, src_w = payload["frame_shape"]
                now      = payload["timestamp"]

                # ── Guardar frame anotado ──────────────────────────────
                with self._frame_lock:
                    self.last_frames[cam_id] = payload["frame_annotated"]

                # ── Decay suave del heatmap de presencia ───────────────
                self.heatmap_presence *= 0.9999

                # ── Procesar CONTACTOS (formato nuevo) ─────────────────
                contacts = payload.get("contacts", [])
                # En FusionThread.run(), después de recibir payload:

                #if contacts:
                #    print(f"[Fusion] {cam_id}: {len(contacts)} contactos — acumulando en mapa")
                for contact in contacts:
                    pos_3d     = contact.get("pos_3d")
                    pixel      = contact.get("pixel")
                    confidence = contact.get("confidence", 0.5)
                    method     = contact.get("method", "unknown")

                    # Proyectar al canvas canónico
                    if pos_3d is not None:
                        # Hay posición 3D real — proyectar directamente
                        # (por ahora sin extrínsecos, solo usa homografía en pixel)
                        gx, gy = self._project_point(
                            cam_id, pixel[0], pixel[1], src_w, src_h
                        )
                    elif pixel is not None:
                        # Solo pixel (RGB sin depth)
                        gx, gy = self._project_point(
                            cam_id, pixel[0], pixel[1], src_w, src_h
                        )
                    else:
                        continue

                    # Acumular en heatmap con peso por confianza
                    self._add_heat(
                        self.heatmap_presence, gx, gy,
                        radius=12,
                        intensity=self.INTENSITY * confidence
                    )

                    # Contador simple de contactos
                    self.hoi_counts["contactos_totales"] += 1

                # ── Renderizar top-view ────────────────────────────────
                self._render_top_view()

                # ── Enviar mosaico al display ──────────────────────────
                mosaic = self.build_mosaic()
                try:
                    self.display_queue.put_nowait({
                        "cam_id": "FUSION",
                        "color":  mosaic,
                        "depth":  None,
                    })
                except queue.Full:
                    pass

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[FusionThread] ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

    def stop(self):
        self.running = False

    # ──────────────────────────────────────────────────────────────────
    # API OFFLINE — llamada directa para postprocess.py
    # ──────────────────────────────────────────────────────────────────
    def fuse_frame(self, cam_id: str, frame_annotated: np.ndarray,
                   contacts: list, frame_shape: tuple,
                   virtual_dt: float = 1/30) -> None:
        """
        Versión síncrona del bucle interno de run(). Acumula un frame
        de una cámara en los heatmaps sin usar queues ni hilos.

        Parámetros
        ----------
        cam_id          : identificador de la cámara
        frame_annotated : frame BGR ya anotado por InferenceProcessor
        contacts        : lista de dicts de contacto (mismo formato que run())
        frame_shape     : (H, W) del frame original (para proyección)
        virtual_dt      : tiempo real en segundos entre este frame y el
                          anterior (obtenido de timestamps.csv). Permite
                          reproducir la misma constante de decay que en
                          el modo online a 30 fps, independientemente de
                          la velocidad de procesamiento offline.
        """
        src_h, src_w = frame_shape

        # ── Guardar frame anotado ──────────────────────────────────────
        with self._frame_lock:
            if cam_id not in self.last_frames:
                self.last_frames[cam_id] = None
            self.last_frames[cam_id] = frame_annotated

        # ── Decay basado en tiempo real, no en frames ──────────────────
        # En online a 30 fps: decay_per_frame = 0.9999
        # decay_per_second = 0.9999^30 ≈ 0.9970
        # Para cualquier virtual_dt: decay = 0.9999^(30 * virtual_dt)
        decay = 0.9999 ** (30.0 * virtual_dt)
        self.heatmap_presence *= decay

        # ── Acumular contactos ─────────────────────────────────────────
        for contact in contacts:
            pixel      = contact.get("pixel")
            confidence = contact.get("confidence", 0.5)

            if pixel is None:
                continue

            gx, gy = self._project_point(cam_id, pixel[0], pixel[1],
                                         src_w, src_h)
            self._add_heat(self.heatmap_presence, gx, gy,
                           radius=12, intensity=self.INTENSITY * confidence)
            self.hoi_counts["contactos_totales"] += 1

        # ── Renderizar top-view ────────────────────────────────────────
        self._render_top_view()

    # ──────────────────────────────────────────────────────────────────
    # MOSAIC — helper para visualizar las 3 cámaras + top-view
    # ──────────────────────────────────────────────────────────────────
    def build_mosaic(self, display_w=320, display_h=240) -> np.ndarray:
        with self._frame_lock:
            frames = [self.last_frames.get(cid) for cid in self.cam_ids]

        # ── Fila superior: frames anotados de cada cámara ─────────────────
        row_cams = []
        for cid, f in zip(self.cam_ids, frames):
            if f is not None:
                tile = cv2.resize(f, (display_w, display_h))
            else:
                tile = np.zeros((display_h, display_w, 3), dtype=np.uint8)
                cv2.putText(tile, f"Sin señal", (10, display_h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80,80,80), 1)
            # Etiqueta de cámara
            cv2.putText(tile, cid, (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1)
            row_cams.append(tile)

        top_row = np.hstack(row_cams)
        total_w = display_w * len(self.cam_ids)

        # ── Fila inferior: top-view fusionado ─────────────────────────────
        tv = self.get_top_view()

        # Dividir el espacio inferior: top-view + panel HOI
        tv_w      = total_w * 2 // 3
        panel_w   = total_w - tv_w

        tv_resized = cv2.resize(tv, (tv_w, display_h))

        # Panel lateral con resumen HOI
        panel = np.zeros((display_h, panel_w, 3), dtype=np.uint8)
        cv2.putText(panel, "HOI Global", (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1)

        top_hoi = sorted(self.hoi_counts.items(), key=lambda x: -x[1])[:8]
        for i, (name, cnt) in enumerate(top_hoi):
            # Barra de progreso proporcional
            max_cnt = top_hoi[0][1] if top_hoi else 1
            bar_w   = int((cnt / max_cnt) * (panel_w - 10))
            bar_y   = 30 + i * 22
            cv2.rectangle(panel, (5, bar_y), (5 + bar_w, bar_y + 14),
                        (0, 180, 100), -1)
            cv2.putText(panel, f"{name}: {cnt}", (7, bar_y + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255,255,255), 1)

        # Totales
        total_hoi = sum(self.hoi_counts.values())
        cv2.putText(panel, f"Total: {total_hoi}", (5, display_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
        cv2.putText(panel, f"Cams: {len(self.cam_ids)}", (5, display_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

        bottom_row = np.hstack([tv_resized, panel])

        return np.vstack([top_row, bottom_row])
    