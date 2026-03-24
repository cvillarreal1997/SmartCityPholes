"""
Detector de baches con YOLOv8
Modelo: keremberke/yolov8m-pothole-segmentation (HuggingFace)
Se descarga automáticamente la primera vez (~50 MB).
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# ─────────────────────────────────────────────────────────────
#  PARÁMETROS
# ─────────────────────────────────────────────────────────────
MODELO_REPO     = "keremberke/yolov8n-pothole-segmentation"
MODELO_ARCHIVO  = "best.pt"
# Cuando tengas tu modelo entrenado, ponlo aquí (deja None para usar el de HuggingFace):
MODELO_LOCAL = "mi_modelo_baches.pt"
ESCALA_VENTANA  = 0.55   # tamaño de la ventana en pantalla
CONFIANZA_MIN   = 0.45   # 0.0–1.0
IOU_NMS         = 0.45
FRAMES_CONFIRM  = 2      # frames consecutivos para confirmar y guardar imagen
ROI_INICIO      = 0.45   # fracción Y desde donde empieza la zona de carretera
INFERENCIA_CADA = 2      # correr YOLO 1 de cada N frames para mantener velocidad real

# Filtro de color para descartar falsos positivos
VERDE_MAX           = 0.25   # más del 25% verde → descartar
BRILLO_MAX          = 180    # brillo máximo del bbox

# Filtros anti-pintura de carretera
PINTURA_BLANCA_PCT  = 0.32   # >32% píxeles blanco/gris-claro (V>172, S<40) → es pintura
PINTURA_AMARILLA_PCT= 0.25   # >25% píxeles amarillos (H 15–38, S>80, V>150) → es pintura amarilla
ASPECTO_MAX         = 5.0    # ratio ancho/alto o alto/ancho > 5 → línea/banda larga
UNIFORMIDAD_BRILLO  = 26     # std del gris < 26 con brillo medio > 138 → superficie uniforme (pintura)
                             # baches brillosos: textura rugosa → std > 26 → pasan el filtro
                             # pintura desgastada: color plano → std < 26 → rechazada
# ─────────────────────────────────────────────────────────────


def es_bache_valido(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> tuple[bool, str]:
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return False, "region vacia"

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    total_px = h.size

    # ── Filtro: vegetación verde ─────────────────────────────────
    mascara_verde = cv2.inRange(hsv, (25, 30, 30), (95, 255, 255))
    porcentaje_verde = float(mascara_verde.sum()) / 255 / total_px
    if porcentaje_verde > VERDE_MAX:
        return False, f"verde {porcentaje_verde:.0%}"

    sat_media = float(s.mean())
    if sat_media > 60:
        return False, f"saturacion alta {sat_media:.0f}"

    gris = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    brillo_medio = float(gris.mean())
    if brillo_medio > BRILLO_MAX:
        return False, f"muy brillante {brillo_medio:.0f}"

    # ── Filtro: pintura blanca/gris-claro (cruces, líneas, flechas, texto) ──
    # V>172 y S<40 captura tanto blanco puro como pintura gris-blanca desgastada
    mascara_blanca = cv2.inRange(hsv, (0, 0, 172), (180, 40, 255))
    pct_blanco = float(mascara_blanca.sum()) / 255 / total_px
    if pct_blanco > PINTURA_BLANCA_PCT:
        return False, f"pintura blanca {pct_blanco:.0%}"

    # ── Filtro: pintura amarilla (líneas centrales, bordes) ──────
    mascara_amarilla = cv2.inRange(hsv, (15, 80, 150), (38, 255, 255))
    pct_amarillo = float(mascara_amarilla.sum()) / 255 / total_px
    if pct_amarillo > PINTURA_AMARILLA_PCT:
        return False, f"pintura amarilla {pct_amarillo:.0%}"

    # ── Filtro: forma muy alargada (líneas y bandas de carretera) ─
    ancho_bbox = x2 - x1
    alto_bbox  = y2 - y1
    if alto_bbox > 0 and ancho_bbox > 0:
        aspecto = max(ancho_bbox / alto_bbox, alto_bbox / ancho_bbox)
        if aspecto > ASPECTO_MAX:
            return False, f"forma alargada {aspecto:.1f}"

    # ── Filtro: superficie uniforme (pintura lisa/desgastada) ────────
    # Pintura: brillo medio + textura plana (std bajo)
    # Bache brilloso: aunque sea claro, tiene grietas y textura rugosa (std alto)
    std_brillo = float(gris.std())
    if brillo_medio > 138 and std_brillo < UNIFORMIDAD_BRILLO:
        return False, f"uniforme brillante brillo={brillo_medio:.0f} std={std_brillo:.0f}"

    return True, ""


def color_por_confianza(conf: float):
    if conf >= 0.80:
        return (0, 200, 0)
    elif conf >= 0.65:
        return (0, 165, 255)
    else:
        return (0, 0, 255)


def guardar_recorte(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                    conf: float, carpeta: Path, num: int) -> str:
    pad  = 25
    h, w = frame.shape[:2]
    rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
    rx2, ry2 = min(w, x2 + pad), min(h, y2 + pad)
    recorte  = frame[ry1:ry2, rx1:rx2].copy()

    color = color_por_confianza(conf)
    bx1, by1 = x1 - rx1, y1 - ry1
    bx2, by2 = x2 - rx1, y2 - ry1
    cv2.rectangle(recorte, (bx1, by1), (bx2, by2), color, 2)
    cv2.putText(recorte, f"Bache {conf:.0%}",
                (bx1, max(by1 - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    nombre = f"bache_{num:04d}_conf{int(conf * 100)}.jpg"
    cv2.imwrite(str(carpeta / nombre), recorte)
    return nombre


def detectar_baches(video_path: str, output_path: str, mostrar_ventana: bool = True):
    if MODELO_LOCAL and Path(MODELO_LOCAL).exists():
        print(f"Cargando modelo propio: {MODELO_LOCAL}\n")
        modelo = YOLO(MODELO_LOCAL)
    else:
        print(f"Cargando modelo: {MODELO_REPO}")
        print("(Primera vez: descarga ~50 MB desde HuggingFace, espera un momento...)\n")
        modelo_path = hf_hub_download(repo_id=MODELO_REPO, filename=MODELO_ARCHIVO)
        modelo = YOLO(modelo_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir {video_path}")
        return

    fps   = cap.get(cv2.CAP_PROP_FPS)
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {ancho}x{alto} @ {fps:.1f} fps | {total} frames")

    carpeta_imgs = Path(output_path).parent / "baches_detectados"
    carpeta_imgs.mkdir(exist_ok=True)
    print(f"Imágenes confirmadas → {carpeta_imgs}\n")

    writer = None
    if output_path:
        Path(output_path).parent.mkdir(exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (ancho, alto))

    tracked: dict = {}
    img_guardadas   = 0
    frame_idx       = 0
    roi_y           = int(alto * ROI_INICIO)
    ultimas_dets    = []   # detecciones del último frame procesado por YOLO

    print("Procesando... (presiona 'q' para salir)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        output_frame = frame.copy()

        # ── Inferencia solo cada INFERENCIA_CADA frames ─────────
        if frame_idx % INFERENCIA_CADA == 0:
            roi_frame  = frame[roi_y:alto, 0:ancho]
            resultados = modelo.track(
                roi_frame,
                persist=True,
                conf=CONFIANZA_MIN,
                iou=IOU_NMS,
                verbose=False,
            )

            nuevas_dets = []
            if resultados and resultados[0].boxes is not None:
                boxes = resultados[0].boxes
                for i in range(len(boxes)):
                    conf = float(boxes.conf[i])
                    if conf < CONFIANZA_MIN:
                        continue

                    xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    y1 += roi_y
                    y2 += roi_y

                    valido, _ = es_bache_valido(frame, x1, y1, x2, y2)
                    if not valido:
                        continue

                    track_id = int(boxes.id[i]) if boxes.id is not None else -(frame_idx * 100 + i)

                    if track_id not in tracked:
                        tracked[track_id] = {
                            "frames_visto": 0,
                            "mejor_conf":   0.0,
                            "mejor_frame":  None,
                            "mejor_bbox":   None,
                            "guardado":     False,
                        }

                    t = tracked[track_id]
                    t["frames_visto"] += 1

                    if conf > t["mejor_conf"]:
                        t["mejor_conf"]  = conf
                        t["mejor_frame"] = frame.copy()
                        t["mejor_bbox"]  = (x1, y1, x2, y2)

                    if (t["frames_visto"] >= FRAMES_CONFIRM
                            and not t["guardado"]
                            and t["mejor_frame"] is not None):
                        img_guardadas += 1
                        bx1, by1, bx2, by2 = t["mejor_bbox"]
                        nombre = guardar_recorte(
                            t["mejor_frame"], bx1, by1, bx2, by2,
                            t["mejor_conf"], carpeta_imgs, img_guardadas
                        )
                        t["guardado"] = True
                        print(f"  ✓ Bache ID {track_id:4d}  conf={t['mejor_conf']:.0%}  → {nombre}")

                    mask_data = None
                    if resultados[0].masks is not None:
                        try:
                            mask_data = resultados[0].masks.data[i].cpu().numpy()
                        except Exception:
                            pass

                    nuevas_dets.append({
                        "bbox": (x1, y1, x2, y2), "conf": conf,
                        "track_id": track_id, "mask": mask_data,
                    })

            ultimas_dets = nuevas_dets

        # ── Dibujar últimas detecciones en TODOS los frames ─────
        cv2.line(output_frame, (0, roi_y), (ancho, roi_y), (60, 60, 60), 1)

        for det in ultimas_dets:
            x1, y1, x2, y2 = det["bbox"]
            conf     = det["conf"]
            track_id = det["track_id"]
            color    = color_por_confianza(conf)

            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
            label = f"Bache #{track_id}  {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(output_frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(output_frame, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if det["mask"] is not None:
                try:
                    roi_alto      = alto - roi_y
                    mask_resized  = cv2.resize(det["mask"], (ancho, roi_alto))
                    mask_bool_roi = mask_resized > 0.5
                    overlay       = output_frame.copy()
                    overlay[roi_y:alto, :][mask_bool_roi] = (
                        overlay[roi_y:alto, :][mask_bool_roi] * 0.5
                        + np.array(color) * 0.5
                    ).astype(np.uint8)
                    output_frame = overlay
                except Exception:
                    pass

        cv2.rectangle(output_frame, (0, 0), (340, 90), (20, 20, 20), -1)
        cv2.putText(output_frame, f"Baches en pantalla : {len(ultimas_dets)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(output_frame, f"Baches confirmados : {img_guardadas}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 255, 100), 2)

        cv2.rectangle(output_frame, (ancho - 200, 10), (ancho - 5, 90), (20, 20, 20), -1)
        for i, (txt, col) in enumerate([
            (">= 80%  alta",  (0, 200, 0)),
            (">= 65%  media", (0, 165, 255)),
            (f">= {int(CONFIANZA_MIN*100)}%   baja", (0, 0, 255)),
        ]):
            cv2.putText(output_frame, txt, (ancho - 190, 32 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

        if writer:
            writer.write(output_frame)

        if mostrar_ventana:
            ventana = cv2.resize(output_frame, (
                int(ancho * ESCALA_VENTANA),
                int(alto  * ESCALA_VENTANA)
            ))
            cv2.imshow("Detector de Baches - YOLOv8", ventana)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nDetenido por el usuario.")
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print("\n── Resultado ───────────────────────────────")
    print(f"  Baches únicos confirmados : {img_guardadas}")
    print(f"  Imágenes en               : {carpeta_imgs}")
    if output_path:
        print(f"  Video resultado           : {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Detector de baches con YOLOv8")
    parser.add_argument("--video",      "-v", type=str, default=None)
    parser.add_argument("--output",     "-o", type=str, default=None)
    parser.add_argument("--no-ventana", action="store_true")
    args = parser.parse_args()

    if args.video is None:
        carpeta = Path(__file__).parent / "videos"
        exts    = {".mp4", ".avi", ".mov", ".mkv", ".MOV", ".MP4", ".AVI"}
        videos  = sorted([f for f in carpeta.iterdir() if f.suffix in exts])
        if not videos:
            print("No hay videos en 'videos/'.")
            print("Coloca tu video ahí o usa:  python detector_baches.py --video ruta.mp4")
            return
        video_path = str(videos[0])
        print(f"Video encontrado: {video_path}")
    else:
        video_path = args.video

    if args.output is None:
        nombre      = Path(video_path).stem
        output_path = str(Path(__file__).parent / "output" / f"{nombre}_baches.mp4")
    else:
        output_path = args.output

    detectar_baches(video_path, output_path, not args.no_ventana)


if __name__ == "__main__":
    main()
