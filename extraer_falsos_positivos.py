"""
extraer_falsos_positivos.py
───────────────────────────
Corre el modelo YOLO sobre un video y guarda TODOS los crops detectados
para que el usuario los revise y clasifique manualmente en:

    dataset_revision/
        bache/      ← mover aquí los que SÍ son baches reales
        no_bache/   ← mover aquí los que son falsos positivos

Luego esas carpetas se suben a Roboflow para reentrenar el modelo.

Uso:
    python extraer_falsos_positivos.py --video videos/holes.mp4
    python extraer_falsos_positivos.py --video videos/holes.mp4 --conf 0.40
    python extraer_falsos_positivos.py --video videos/holes.mp4 --max_crops 300
"""

import cv2
import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO

# ─── Configuración ──────────────────────────────────────────────────────────
MODELO_LOCAL   = Path(__file__).parent / "mi_modelo_baches.pt"
SALIDA_DIR     = Path(__file__).parent / "dataset_revision"
ROI_INICIO     = 0.35   # ignorar parte superior del frame (cielo, horizonte)
INFERENCIA_CADA = 3     # procesar 1 de cada N frames (acelera el proceso)
# ─────────────────────────────────────────────────────────────────────────────


def extraer_crops(video_path: str, conf_min: float, max_crops: int):
    # ── Preparar directorios de salida ───────────────────────────────────────
    bache_dir    = SALIDA_DIR / "bache"
    no_bache_dir = SALIDA_DIR / "no_bache"
    bache_dir.mkdir(parents=True, exist_ok=True)
    no_bache_dir.mkdir(parents=True, exist_ok=True)

    # Carpeta para el frame completo de contexto (opcional, ayuda al anotar)
    contexto_dir = SALIDA_DIR / "_contexto_frames"
    contexto_dir.mkdir(exist_ok=True)

    print(f"[INFO] Modelo : {MODELO_LOCAL}")
    print(f"[INFO] Video  : {video_path}")
    print(f"[INFO] Conf   : ≥ {conf_min:.0%}")
    print(f"[INFO] Salida : {SALIDA_DIR}")
    print()

    # ── Cargar modelo ────────────────────────────────────────────────────────
    model = YOLO(str(MODELO_LOCAL))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] No se puede abrir: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Video: {total_frames} frames @ {fps:.1f} fps")

    crops_guardados = 0
    frame_idx       = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Saltar frames para ir más rápido
        if frame_idx % INFERENCIA_CADA != 0:
            continue

        if crops_guardados >= max_crops:
            print(f"\n[INFO] Límite alcanzado: {max_crops} crops. Terminando.")
            break

        h, w = frame.shape[:2]

        # ── Recortar ROI (ignorar cielo/capó) ───────────────────────────────
        roi_y = int(h * ROI_INICIO)
        roi   = frame[roi_y:, :]

        # Redimensionar a máx 640px para inferencia rápida
        max_w = 640
        if roi.shape[1] > max_w:
            escala = max_w / roi.shape[1]
            roi_inf = cv2.resize(roi, (max_w, int(roi.shape[0] * escala)))
        else:
            escala  = 1.0
            roi_inf = roi

        # ── Inferencia ───────────────────────────────────────────────────────
        results = model.predict(
            roi_inf,
            conf=conf_min,
            iou=0.45,
            verbose=False,
            device="cuda" if _cuda_disponible() else "cpu",
        )

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Escalar coords de vuelta a roi original
                if escala != 1.0:
                    x1 = int(x1 / escala); y1 = int(y1 / escala)
                    x2 = int(x2 / escala); y2 = int(y2 / escala)

                # Clip a tamaño roi
                rh, rw = roi.shape[:2]
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(rw, x2); y2 = min(rh, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                # ── Guardar crop con padding ─────────────────────────────────
                pad = 20
                cx1 = max(0, x1 - pad); cy1 = max(0, y1 - pad)
                cx2 = min(rw, x2 + pad); cy2 = min(rh, y2 + pad)
                crop = roi[cy1:cy2, cx1:cx2].copy()

                # Dibujar bbox sobre el crop para que sea fácil de revisar
                bx1, by1 = x1 - cx1, y1 - cy1
                bx2, by2 = x2 - cx1, y2 - cy1
                cv2.rectangle(crop, (bx1, by1), (bx2, by2), (0, 140, 255), 2)
                cv2.putText(crop, f"{conf:.0%}", (bx1, max(by1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

                nombre = f"f{frame_idx:06d}_c{conf:.2f}_x{x1}y{y1}.jpg"
                # Por defecto va a no_bache/ — el usuario moverá los reales a bache/
                cv2.imwrite(str(no_bache_dir / nombre), crop)

                # Guardar frame de contexto (frame completo con bbox marcado)
                ctx = frame.copy()
                # Ajustar coords al frame completo
                fy1, fy2 = y1 + roi_y, y2 + roi_y
                cv2.rectangle(ctx, (x1, fy1), (x2, fy2), (0, 140, 255), 3)
                cv2.putText(ctx, f"DETECCION {conf:.0%}",
                            (x1, max(fy1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
                cv2.imwrite(str(contexto_dir / nombre), ctx)

                crops_guardados += 1
                pct = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
                print(f"  [{pct:5.1f}%] Frame {frame_idx:5d} | "
                      f"conf {conf:.0%} | crop #{crops_guardados:03d} → {nombre}")

    cap.release()
    print(f"\n{'='*60}")
    print(f"✓ Extracción completa: {crops_guardados} crops guardados")
    print(f"  Carpeta: {SALIDA_DIR.absolute()}")
    print()
    print("PRÓXIMOS PASOS:")
    print("  1. Abre la carpeta  dataset_revision/no_bache/")
    print("  2. MUEVE a          dataset_revision/bache/")
    print("     los crops que SÍ son baches reales")
    print("  3. Los que quedan en no_bache/ son los falsos positivos")
    print("  4. Sube ambas carpetas a Roboflow para anotar y reentrenar")


def _cuda_disponible() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrae crops de detecciones para revisión")
    parser.add_argument("--video",      required=True,        help="Ruta al video")
    parser.add_argument("--conf",       type=float, default=0.40, help="Confianza mínima (default 0.40)")
    parser.add_argument("--max_crops",  type=int,   default=500,  help="Máximo de crops a guardar (default 500)")
    args = parser.parse_args()

    extraer_crops(args.video, args.conf, args.max_crops)
