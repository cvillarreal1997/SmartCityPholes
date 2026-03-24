"""
Extrae frames del video para etiquetar baches.
Guarda imágenes en la carpeta 'frames_para_etiquetar/'.

Uso:
    python extraer_frames.py
    python extraer_frames.py --cada 30        # 1 frame cada 30 frames
    python extraer_frames.py --cada 15        # más frames (video con muchos baches)
"""

import cv2
import argparse
from pathlib import Path

def extraer(video_path: str, cada_n: int, carpeta_out: Path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir {video_path}")
        return

    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    carpeta_out.mkdir(parents=True, exist_ok=True)

    print(f"Video   : {ancho}x{alto} @ {fps:.1f} fps | {total} frames")
    print(f"Extrayendo 1 de cada {cada_n} frames...")
    print(f"Destino : {carpeta_out}\n")

    guardados = 0
    idx       = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1

        if idx % cada_n == 0:
            nombre = carpeta_out / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(nombre), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            guardados += 1
            if guardados % 20 == 0:
                print(f"  {guardados} imágenes guardadas...")

    cap.release()
    print(f"\nListo. {guardados} imágenes en: {carpeta_out}")
    print("\nSiguiente paso:")
    print("  1. Ve a https://roboflow.com y crea una cuenta gratuita")
    print("  2. Nuevo proyecto → Object Detection → sube la carpeta 'frames_para_etiquetar'")
    print("  3. Etiqueta los baches (especialmente los pequeños que el modelo no detecta)")
    print("  4. Export → YOLOv8 → Download ZIP")
    print("  5. Abre el notebook 'entrenar_colab.ipynb' en Google Colab")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", "-v", type=str, default=None)
    parser.add_argument("--cada",  "-c", type=int, default=20,
                        help="Extraer 1 frame de cada N (default: 20)")
    args = parser.parse_args()

    if args.video is None:
        carpeta = Path(__file__).parent / "videos"
        exts    = {".mp4", ".avi", ".mov", ".mkv", ".MOV", ".MP4", ".AVI"}
        videos  = sorted([f for f in carpeta.iterdir() if f.suffix in exts])
        if not videos:
            print("No hay videos en 'videos/'. Usa: python extraer_frames.py --video ruta.mp4")
            return
        video_path = str(videos[0])
        print(f"Video encontrado: {video_path}")
    else:
        video_path = args.video

    carpeta_out = Path(__file__).parent / "frames_para_etiquetar"
    extraer(video_path, args.cada, carpeta_out)


if __name__ == "__main__":
    main()
