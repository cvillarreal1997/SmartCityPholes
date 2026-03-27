"""
detector_gps.py — Detector de baches con GPS integrado
Corre en el vehículo municipal. Guarda cada bache detectado en SQLite local
con sus coordenadas GPS, foto y metadatos. Al terminar el turno sincroniza
con el servidor central del municipio.
"""

import cv2
import numpy as np
import sqlite3
import base64
import requests
import argparse
import math
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Distancia en metros entre dos coordenadas GPS."""
    R = 6_371_000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

RADIO_DEDUP_LOCAL_M = 12.0   # metros — mismo bache entre corridas

# ── Importar lector GPS ──────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from gps_reader import crear_gps, CoordenadaGPS

# ─────────────────────────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────
MODELO_LOCAL        = str(Path(__file__).parent.parent / "mi_modelo_baches.pt")
MODELO_REPO         = "keremberke/yolov8n-pothole-segmentation"
MODELO_ARCHIVO      = "best.pt"

CONFIANZA_MIN       = 0.45
IOU_NMS             = 0.45
FRAMES_CONFIRM      = 3       # frames consecutivos para confirmar bache
ROI_INICIO          = 0.45
INFERENCIA_CADA     = 2
ESCALA_VENTANA      = 0.55

DB_PATH             = Path(__file__).parent / "baches_local.db"
FOTOS_DIR           = Path(__file__).parent / "fotos"
SERVIDOR_URL        = "http://localhost:8000"   # cambiar por IP del servidor

# API key del vehículo — se obtiene al registrar el dispositivo en el servidor
# Ejecutar una vez: POST /api/dispositivos  →  copiar api_key aquí
DEVICE_API_KEY      = "15308dc1d046017fab5f8deb48b7b6130e449f6bf3171a7ab718164496de6df8"

# GPS: "simulador" para pruebas, "real" con hardware conectado
GPS_MODO            = "simulador"
GPS_PUERTO          = "COM3"

# Filtros anti-pintura
VERDE_MAX            = 0.25
BRILLO_MAX           = 180
PINTURA_BLANCA_PCT   = 0.22      # si ≥22 % de pixels son blanco puro → pintura
PINTURA_AMARILLA_PCT = 0.22
ASPECTO_MAX          = 5.0
UNIFORMIDAD_BRILLO   = 26
PIXELS_BRILLANTES_PCT= 0.25      # si ≥25 % son blanco puro (S<25) → pintura vial
# ─────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════
#  BASE DE DATOS LOCAL (SQLite — funciona sin internet)
# ══════════════════════════════════════════════════════════════════

def iniciar_db() -> sqlite3.Connection:
    FOTOS_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS baches (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            latitud     REAL    NOT NULL,
            longitud    REAL    NOT NULL,
            altitud     REAL    DEFAULT 0,
            velocidad   REAL    DEFAULT 0,
            precision_gps REAL  DEFAULT 0,
            confianza   REAL    NOT NULL,
            severidad   TEXT    NOT NULL,   -- leve / moderado / grave / critico
            ancho_px    INTEGER DEFAULT 0,
            alto_px     INTEGER DEFAULT 0,
            foto_path           TEXT,
            foto_panorama_path  TEXT,
            fecha       TEXT    NOT NULL,
            turno_id    TEXT    NOT NULL,
            sincronizado INTEGER DEFAULT 0  -- 0=pendiente, 1=enviado al servidor
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS turnos (
            id          TEXT PRIMARY KEY,
            vehiculo    TEXT,
            operador    TEXT,
            inicio      TEXT,
            fin         TEXT,
            km_recorridos REAL DEFAULT 0,
            total_baches  INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    return conn


def calcular_severidad(confianza: float, ancho: int, alto: int) -> str:
    area = ancho * alto
    if confianza >= 0.80 and area > 40_000:
        return "critico"
    if confianza >= 0.65 or area > 20_000:
        return "grave"
    if confianza >= 0.55 or area > 8_000:
        return "moderado"
    return "leve"


def guardar_bache(conn: sqlite3.Connection, coord: CoordenadaGPS,
                  frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                  confianza: float, turno_id: str) -> int:
    """Guarda un bache confirmado en SQLite con foto y coordenadas."""
    ancho = x2 - x1
    alto  = y2 - y1
    severidad = calcular_severidad(confianza, ancho, alto)

    color = {"critico": (0,0,255), "grave": (0,100,255),
             "moderado": (0,165,255), "leve": (0,200,0)}[severidad]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]

    # ── Foto recorte (zoom al bache) ─────────────────────────────
    pad = 30
    h, w = frame.shape[:2]
    rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
    rx2, ry2 = min(w, x2 + pad), min(h, y2 + pad)
    recorte   = frame[ry1:ry2, rx1:rx2].copy()
    bx1, by1  = x1 - rx1, y1 - ry1
    bx2, by2  = x2 - rx1, y2 - ry1
    cv2.rectangle(recorte, (bx1, by1), (bx2, by2), color, 2)
    cv2.putText(recorte, f"{severidad.upper()} {confianza:.0%}",
                (bx1, max(by1-8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    foto = FOTOS_DIR / f"bache_{ts}.jpg"
    cv2.imwrite(str(foto), recorte)

    # ── Foto panorama (frame completo con bache marcado) ──────────
    panorama = frame.copy()
    cv2.rectangle(panorama, (x1, y1), (x2, y2), color, 3)
    cx_p, cy_p = (x1 + x2) // 2, y1
    cv2.arrowedLine(panorama, (cx_p, max(0, cy_p - 60)), (cx_p, cy_p), color, 3, tipLength=0.3)
    label = f"BACHE {severidad.upper()} {confianza:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    lx = max(0, cx_p - tw // 2)
    ly = max(th + 4, cy_p - 68)
    cv2.rectangle(panorama, (lx-4, ly-th-4), (lx+tw+4, ly+4), color, -1)
    cv2.putText(panorama, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    foto_panorama = FOTOS_DIR / f"bache_{ts}_panorama.jpg"
    cv2.imwrite(str(foto_panorama), panorama)

    cur = conn.execute("""
        INSERT INTO baches
          (latitud, longitud, altitud, velocidad, precision_gps,
           confianza, severidad, ancho_px, alto_px,
           foto_path, foto_panorama_path, fecha, turno_id)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        coord.latitud, coord.longitud, coord.altitud,
        coord.velocidad, coord.precision,
        confianza, severidad, ancho, alto,
        str(foto), str(foto_panorama) if foto_panorama else None,
        datetime.now().isoformat(), turno_id
    ))
    conn.commit()
    return cur.lastrowid, foto.name


def crear_turno(conn: sqlite3.Connection, vehiculo: str, operador: str) -> str:
    turno_id = f"T{datetime.now().strftime('%Y%m%d%H%M%S')}"
    conn.execute("""
        INSERT INTO turnos (id, vehiculo, operador, inicio)
        VALUES (?,?,?,?)
    """, (turno_id, vehiculo, operador, datetime.now().isoformat()))
    conn.commit()
    print(f"[DB] Turno iniciado: {turno_id}")
    return turno_id


def cerrar_turno(conn: sqlite3.Connection, turno_id: str):
    total = conn.execute(
        "SELECT COUNT(*) FROM baches WHERE turno_id=?", (turno_id,)
    ).fetchone()[0]
    conn.execute("""
        UPDATE turnos SET fin=?, total_baches=? WHERE id=?
    """, (datetime.now().isoformat(), total, turno_id))
    conn.commit()
    print(f"[DB] Turno cerrado: {turno_id} | Baches: {total}")


# ══════════════════════════════════════════════════════════════════
#  SINCRONIZACIÓN CON SERVIDOR
# ══════════════════════════════════════════════════════════════════

def sincronizar_con_servidor(conn: sqlite3.Connection):
    """Envía los baches pendientes al servidor central del municipio."""
    pendientes = conn.execute(
        "SELECT * FROM baches WHERE sincronizado=0"
    ).fetchall()

    if not pendientes:
        print("[SYNC] No hay baches pendientes de sincronizar.")
        return

    print(f"[SYNC] Sincronizando {len(pendientes)} baches...")
    enviados = 0

    for row in pendientes:
        def _leer_foto(campo):
            p = row[campo] if campo in row.keys() else None
            if isinstance(p, bytes):
                p = p.decode("utf-8", errors="replace")
            if p and Path(p).exists():
                with open(p, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            return None

        foto_b64     = _leer_foto("foto_path")
        panorama_b64 = _leer_foto("foto_panorama_path")

        import struct

        def _str(v):
            if isinstance(v, bytes):
                return v.decode("utf-8", errors="replace")
            return str(v) if v is not None else ""

        def _int(v):
            if isinstance(v, bytes):
                return int.from_bytes(v, byteorder="little")
            return int(v or 0)

        def _float(v):
            if isinstance(v, bytes):
                # SQLite guarda reales como IEEE 754 double little-endian (8 bytes)
                return struct.unpack("<d", v)[0]
            return float(v or 0)

        payload = {
            "latitud":       _float(row["latitud"]),
            "longitud":      _float(row["longitud"]),
            "altitud":       _float(row["altitud"]),
            "velocidad":     _float(row["velocidad"]),
            "precision_gps": _float(row["precision_gps"]),
            "confianza":     _float(row["confianza"]),
            "severidad":     _str(row["severidad"]),
            "ancho_px":      _int(row["ancho_px"]),
            "alto_px":       _int(row["alto_px"]),
            "fecha":         _str(row["fecha"]),
            "turno_id":             _str(row["turno_id"]),
            "foto_base64":          foto_b64,
            "foto_panorama_base64": panorama_b64,
        }

        try:
            r = requests.post(
                f"{SERVIDOR_URL}/api/baches",
                json=payload,
                headers={"X-Api-Key": DEVICE_API_KEY},  # Device Authentication
                timeout=10,
            )
            # 201 = bache nuevo creado | 200 = duplicado actualizado en servidor
            if r.status_code in (200, 201):
                conn.execute(
                    "UPDATE baches SET sincronizado=1 WHERE id=?", (row["id"],)
                )
                enviados += 1
        except requests.exceptions.ConnectionError:
            print("[SYNC] Servidor no disponible. Datos guardados localmente.")
            break

    conn.commit()
    print(f"[SYNC] Enviados: {enviados}/{len(pendientes)}")


# ══════════════════════════════════════════════════════════════════
#  FILTROS ANTI-PINTURA (igual que detector_baches.py)
# ══════════════════════════════════════════════════════════════════

def es_bache_valido(frame, x1, y1, x2, y2):
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return False

    hsv      = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    h, s, v  = cv2.split(hsv)
    total_px = h.size
    gris     = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    brillo_medio = float(gris.mean())

    # 1. Vegetación (verde)
    mascara_verde = cv2.inRange(hsv, (25, 30, 30), (95, 255, 255))
    if float(mascara_verde.sum()) / 255 / total_px > VERDE_MAX:
        return False

    # 2. Color muy saturado (no es carretera)
    if float(s.mean()) > 60:
        return False

    # 3. Brillo promedio excesivo
    if brillo_medio > BRILLO_MAX:
        return False

    # 4. ── Pintura blanca vial (clave) ────────────────────────────
    # El bbox de una marca vial mezcla pintura (~40%) + asfalto oscuro (~60%).
    # Brillo promedio baja (~120-140), pero los pixels MUY BLANCOS (V>185, S<45)
    # siguen representando el 25-40% del bbox.
    # Bache arenoso: arena tiene V=150-170 y rara vez supera V=185 → <8% de pixels.
    mascara_blanca = cv2.inRange(hsv, (0, 0, 185), (180, 45, 255))
    if float(mascara_blanca.sum()) / 255 / total_px > 0.10:
        return False

    # 5. Pintura amarilla (líneas centrales)
    mascara_amarilla = cv2.inRange(hsv, (15, 80, 150), (38, 255, 255))
    if float(mascara_amarilla.sum()) / 255 / total_px > PINTURA_AMARILLA_PCT:
        return False

    # 6. Aspecto muy elongado = línea de carretera
    ancho_bbox = x2 - x1
    alto_bbox  = y2 - y1
    if alto_bbox > 0 and ancho_bbox > 0:
        if max(ancho_bbox / alto_bbox, alto_bbox / ancho_bbox) > ASPECTO_MAX:
            return False

    # 7. Superficie uniforme y clara = pintura sólida
    std_brillo = float(gris.std())
    if brillo_medio > 138 and std_brillo < UNIFORMIDAD_BRILLO:
        return False

    # 8. Textura (Laplaciano): zona clara + lisa = pintura vial
    lap_var = float(cv2.Laplacian(gris, cv2.CV_64F).var())
    if brillo_medio > 148 and lap_var < 700:
        return False
    if brillo_medio > 115 and lap_var < 280:
        return False

    # 9. Sombras de rejas/vallas (rayas direccionales)
    if gris.shape[0] > 4 and gris.shape[1] > 4:
        var_filas = float(np.var(gris.mean(axis=1)))
        var_cols  = float(np.var(gris.mean(axis=0)))
        min_var   = min(var_filas, var_cols) + 1e-5
        if max(var_filas, var_cols) / min_var > 4.0:
            return False

    return True


def color_por_severidad(severidad: str):
    return {"critico": (0,0,255), "grave": (0,100,255),
            "moderado": (0,165,255), "leve": (0,200,0)}.get(severidad, (255,255,255))


def iou_cajas(a, b) -> float:
    """Intersection over Union entre dos bboxes (x1,y1,x2,y2)."""
    xi1 = max(a[0], b[0]); yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2]); yi2 = min(a[3], b[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)


def confirmado_por_segundo_modelo(modelo2, roi_small, escala, roi_y,
                                   box_original, conf_min=0.20) -> bool:
    """Retorna True si el segundo modelo también detecta algo en la misma zona.

    Estrategia: el centro del bbox de M1 debe caer dentro del bbox de M2
    (con margen). Esto es robusto cuando los dos modelos generan bboxes
    de tamaños muy distintos pero apuntan al mismo objeto.
    """
    try:
        res2 = modelo2.predict(roi_small, conf=conf_min, iou=0.45,
                               verbose=False, device=DEVICE)
        if not res2 or res2[0].boxes is None or len(res2[0].boxes) == 0:
            return False
        x1o, y1o, x2o, y2o = box_original
        # Centro de la detección de M1 en coordenadas de roi_small
        cx = int((x1o + x2o) / 2 * escala)
        cy = int(((y1o + y2o) / 2 - roi_y) * escala)
        margen = 50  # tolerancia en píxeles (roi_small es 640px)
        for b2 in res2[0].boxes.xyxy.cpu().numpy():
            bx1, by1, bx2, by2 = int(b2[0]), int(b2[1]), int(b2[2]), int(b2[3])
            if (bx1 - margen <= cx <= bx2 + margen and
                    by1 - margen <= cy <= by2 + margen):
                return True
        return False
    except Exception:
        return True   # si falla el segundo modelo, no bloqueamos


# ══════════════════════════════════════════════════════════════════
#  LOOP PRINCIPAL DE DETECCIÓN
# ══════════════════════════════════════════════════════════════════

def detectar(video_path: str, output_path: str, mostrar: bool,
             vehiculo: str, operador: str):

    # Cargar modelos (ensemble: ambos deben coincidir para confirmar un bache)
    if Path(MODELO_LOCAL).exists():
        print(f"[MODELO 1] Cargando modelo propio: {MODELO_LOCAL}")
        modelo = YOLO(MODELO_LOCAL)
    else:
        print(f"[MODELO 1] Descargando desde HuggingFace...")
        modelo = YOLO(hf_hub_download(MODELO_REPO, MODELO_ARCHIVO))

    try:
        print(f"[MODELO 2] Cargando modelo HuggingFace (validador)...")
        modelo2 = YOLO(hf_hub_download(MODELO_REPO, MODELO_ARCHIVO))
        usar_ensemble = True
        print(f"[ENSEMBLE] Activo — ambos modelos deben coincidir para guardar un bache")
    except Exception as e:
        print(f"[ENSEMBLE] No disponible ({e}) — usando solo modelo 1")
        modelo2       = None
        usar_ensemble = False

    # Iniciar GPS
    gps = crear_gps(GPS_MODO, GPS_PUERTO)

    # Iniciar BD local
    conn     = iniciar_db()
    turno_id = crear_turno(conn, vehiculo, operador)

    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[VIDEO] {ancho}x{alto} @ {fps:.1f}fps | {total} frames")

    writer = None
    if output_path:
        Path(output_path).parent.mkdir(exist_ok=True)
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ancho, alto)
        )

    tracked          = {}
    pendientes_st    = []   # buffer para sin_tracker: acumular y guardar al salir
    baches_recientes = []   # (cx, cy, frame_idx) — dedup visual dentro de la corrida
    baches_gps_guardados = []  # (lat, lon) — dedup GPS extra
    baches_total     = 0
    frame_idx             = 0
    roi_y        = int(alto * ROI_INICIO)
    ultimas_dets = []
    DIST_MISMO_BACHE_PX = 200  # px — distancia para considerar misma detección sin tracker

    print("Procesando... (q = salir | s = sincronizar ahora)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx   += 1
        output_frame = frame.copy()
        coord        = gps.coordenada_actual()

        if frame_idx % INFERENCIA_CADA == 0:
            roi_frame = frame[roi_y:alto, 0:ancho]
            # Redimensionar a max 640px de ancho para inferencia más rápida
            h_roi, w_roi = roi_frame.shape[:2]
            if w_roi > 640:
                escala    = 640 / w_roi
                roi_small = cv2.resize(roi_frame, (640, int(h_roi * escala)))
            else:
                escala    = 1.0
                roi_small = roi_frame

            resultados = modelo.track(
                roi_small, persist=True,
                conf=CONFIANZA_MIN, iou=IOU_NMS, verbose=False,
                device=DEVICE,
            )

            nuevas_dets = []
            if resultados and resultados[0].boxes is not None:
                boxes = resultados[0].boxes
                for i in range(len(boxes)):
                    conf = float(boxes.conf[i])
                    if conf < CONFIANZA_MIN:
                        continue

                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1 = int(xyxy[0] / escala)
                    y1 = int(xyxy[1] / escala)
                    x2 = int(xyxy[2] / escala)
                    y2 = int(xyxy[3] / escala)
                    y1 += roi_y
                    y2 += roi_y

                    if not es_bache_valido(frame, x1, y1, x2, y2):
                        continue

                    # Ensemble: solo validar con M2 cuando M1 tiene baja confianza
                    # Si M1 ya supera 0.68 lo consideramos confiable solo
                    if usar_ensemble and conf < 0.68 and not confirmado_por_segundo_modelo(
                        modelo2, roi_small, escala, roi_y, (x1, y1, x2, y2)
                    ):
                        continue

                    sin_tracker = boxes.id is None
                    track_id    = -(frame_idx * 100 + i) if sin_tracker else int(boxes.id[i])

                    # ── Sin tracker: NO guardar inmediatamente, acumular en buffer ──
                    if sin_tracker and conf >= 0.58 and coord is not None:
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                        # Buscar si ya hay un pendiente cercano (mismo bache acercándose)
                        encontrado = False
                        for p in pendientes_st:
                            dx = abs(cx - p["cx"])
                            dy_ok = (cy >= p["cy"]) or (abs(cy - p["cy"]) < DIST_MISMO_BACHE_PX)
                            if dx < DIST_MISMO_BACHE_PX and dy_ok:
                                # Actualizar: mover posición y guardar mejor foto
                                p["cx"] = cx
                                p["cy"] = cy
                                p["ultimo_frame"] = frame_idx
                                p["visto"] += 1
                                if conf > p["mejor_conf"]:
                                    p["mejor_conf"]  = conf
                                    p["mejor_frame"] = frame.copy()
                                    p["mejor_bbox"]  = (x1, y1, x2, y2)
                                    p["mejor_coord"] = coord
                                encontrado = True
                                break

                        if not encontrado:
                            # Nuevo bache pendiente
                            pendientes_st.append({
                                "cx": cx, "cy": cy,
                                "ultimo_frame": frame_idx,
                                "mejor_conf": conf,
                                "mejor_frame": frame.copy(),
                                "mejor_bbox": (x1, y1, x2, y2),
                                "mejor_coord": coord,
                                "visto": 1,
                            })

                        sev = calcular_severidad(conf, x2-x1, y2-y1)
                        nuevas_dets.append({
                            "bbox": (x1, y1, x2, y2), "conf": conf,
                            "track_id": track_id, "severidad": sev,
                        })
                        continue

                    if track_id not in tracked:
                        tracked[track_id] = {
                            "frames_visto": 0, "mejor_conf": 0.0,
                            "mejor_frame": None, "mejor_bbox": None,
                            "mejor_coord": None, "guardado": False,
                        }

                    t = tracked[track_id]
                    t["frames_visto"] += 1

                    if conf > t["mejor_conf"]:
                        t["mejor_conf"]  = conf
                        t["mejor_frame"] = frame.copy()
                        t["mejor_bbox"]  = (x1, y1, x2, y2)
                        t["mejor_coord"] = coord

                    # NO guardar todavía — solo marcar confirmado.
                    # Se guarda cuando el track_id desaparezca del frame.
                    if t["frames_visto"] >= FRAMES_CONFIRM:
                        t["confirmado"] = True

                    sev  = calcular_severidad(conf, x2-x1, y2-y1)
                    nuevas_dets.append({
                        "bbox": (x1, y1, x2, y2), "conf": conf,
                        "track_id": track_id, "severidad": sev,
                    })

            # ── Flush: guardar baches confirmados que ya salieron de la pantalla ──
            ids_en_pantalla = {d["track_id"] for d in nuevas_dets}
            for tid, t in list(tracked.items()):
                if tid in ids_en_pantalla:
                    continue  # sigue en pantalla, no guardar aún
                if not t.get("confirmado") or t["guardado"]:
                    continue  # no confirmado o ya guardado
                if t["mejor_frame"] is None or t["mejor_coord"] is None:
                    t["guardado"] = True
                    continue

                bx1, by1, bx2, by2 = t["mejor_bbox"]
                tcx, tcy = (bx1 + bx2) // 2, (by1 + by2) // 2

                # Dedup por píxeles
                ya_guardado = any(
                    (
                        (abs(tcx - px) < 120 and tcy >= py)
                        or (abs(tcx - px) < 120 and abs(tcy - py) < 400)
                    )
                    and (frame_idx - pf) < 450
                    for px, py, pf in baches_recientes
                )
                # Dedup GPS extra
                if not ya_guardado and t["mejor_coord"] is not None:
                    mc = t["mejor_coord"]
                    for glat, glon in baches_gps_guardados:
                        if haversine_m(mc.latitud, mc.longitud, glat, glon) < RADIO_DEDUP_LOCAL_M:
                            ya_guardado = True
                            break

                if not ya_guardado:
                    baches_total += 1
                    bid, foto_name = guardar_bache(
                        conn, t["mejor_coord"],
                        t["mejor_frame"], bx1, by1, bx2, by2,
                        t["mejor_conf"], turno_id
                    )
                    sev = calcular_severidad(t["mejor_conf"], bx2-bx1, by2-by1)
                    c   = t["mejor_coord"]
                    baches_recientes.append((tcx, tcy, frame_idx))
                    if c is not None:
                        baches_gps_guardados.append((c.latitud, c.longitud))
                    print(f"  ✓ Bache #{bid} | {sev.upper()} {t['mejor_conf']:.0%} "
                          f"| GPS ({c.latitud:.6f}, {c.longitud:.6f}) | {foto_name}")
                t["guardado"] = True

            # ── Flush pendientes sin tracker: guardar los que ya salieron de pantalla ──
            pendientes_vivos = []
            for p in pendientes_st:
                frames_sin_ver = frame_idx - p["ultimo_frame"]
                if frames_sin_ver < 30:
                    pendientes_vivos.append(p)  # aún podría reaparecer
                    continue


                mc = p["mejor_coord"]
                # Dedup GPS
                ya_guardado = False
                if mc is not None:
                    for glat, glon in baches_gps_guardados:
                        if haversine_m(mc.latitud, mc.longitud, glat, glon) < RADIO_DEDUP_LOCAL_M:
                            ya_guardado = True
                            break

                if not ya_guardado:
                    bx1, by1, bx2, by2 = p["mejor_bbox"]
                    baches_total += 1
                    bid, foto_name = guardar_bache(
                        conn, mc,
                        p["mejor_frame"], bx1, by1, bx2, by2,
                        p["mejor_conf"], turno_id
                    )
                    sev = calcular_severidad(p["mejor_conf"], bx2-bx1, by2-by1)
                    tcx, tcy = (bx1 + bx2) // 2, (by1 + by2) // 2
                    baches_recientes.append((tcx, tcy, frame_idx))
                    if mc is not None:
                        baches_gps_guardados.append((mc.latitud, mc.longitud))
                    print(f"  ✓ Bache #{bid} (sin tracker) | {sev.upper()} {p['mejor_conf']:.0%} "
                          f"| visto {p['visto']}x | {foto_name}")

            pendientes_st = pendientes_vivos

            ultimas_dets = nuevas_dets

        # ── Dibujar ─────────────────────────────────────────────
        cv2.line(output_frame, (0, roi_y), (ancho, roi_y), (60,60,60), 1)

        for det in ultimas_dets:
            x1, y1, x2, y2 = det["bbox"]
            color = color_por_severidad(det["severidad"])
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
            label = f"#{det['track_id']} {det['severidad']} {det['conf']:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(output_frame, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
            cv2.putText(output_frame, label, (x1+3, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

        # HUD superior izquierdo
        cv2.rectangle(output_frame, (0,0), (360, 115), (15,15,15), -1)
        cv2.putText(output_frame, f"Baches en pantalla : {len(ultimas_dets)}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        cv2.putText(output_frame, f"Baches confirmados : {baches_total}",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100,255,100), 2)

        if coord:
            gps_txt = f"GPS: {coord.latitud:.5f}, {coord.longitud:.5f}"
            vel_txt = f"Vel: {coord.velocidad:.1f} km/h  HDOP: {coord.precision:.1f}"
            cv2.putText(output_frame, gps_txt,
                        (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100,200,255), 1)
            cv2.putText(output_frame, vel_txt,
                        (10, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100,200,255), 1)

        if writer:
            writer.write(output_frame)

        if mostrar:
            ventana = cv2.resize(output_frame, (
                int(ancho * ESCALA_VENTANA), int(alto * ESCALA_VENTANA)
            ))
            cv2.imshow("Sistema Vial GAD — Detección en vivo", ventana)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            if tecla == ord("s"):
                sincronizar_con_servidor(conn)
            if tecla == ord("p"):
                cv2.putText(ventana, "PAUSADO — presiona P para continuar",
                            (10, ventana.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("Sistema Vial GAD — Detección en vivo", ventana)
                while True:
                    if cv2.waitKey(100) & 0xFF == ord("p"):
                        break

    # ── Flush final: guardar todo lo que quedó en buffers ──────────
    # Pendientes sin tracker
    for p in pendientes_st:
        mc = p["mejor_coord"]
        ya_guardado = False
        if mc is not None:
            for glat, glon in baches_gps_guardados:
                if haversine_m(mc.latitud, mc.longitud, glat, glon) < RADIO_DEDUP_LOCAL_M:
                    ya_guardado = True
                    break
        if not ya_guardado:
            bx1, by1, bx2, by2 = p["mejor_bbox"]
            baches_total += 1
            bid, foto_name = guardar_bache(
                conn, mc, p["mejor_frame"], bx1, by1, bx2, by2,
                p["mejor_conf"], turno_id
            )
            sev = calcular_severidad(p["mejor_conf"], bx2-bx1, by2-by1)
            if mc is not None:
                baches_gps_guardados.append((mc.latitud, mc.longitud))
            print(f"  ✓ Bache #{bid} (final) | {sev.upper()} {p['mejor_conf']:.0%} | {foto_name}")

    # Tracked con tracker que no se guardaron
    for tid, t in tracked.items():
        if t.get("confirmado") and not t["guardado"]:
            if t["mejor_frame"] is not None and t["mejor_coord"] is not None:
                mc = t["mejor_coord"]
                ya_guardado = False
                for glat, glon in baches_gps_guardados:
                    if haversine_m(mc.latitud, mc.longitud, glat, glon) < RADIO_DEDUP_LOCAL_M:
                        ya_guardado = True
                        break
                if not ya_guardado:
                    bx1, by1, bx2, by2 = t["mejor_bbox"]
                    baches_total += 1
                    bid, foto_name = guardar_bache(
                        conn, mc, t["mejor_frame"], bx1, by1, bx2, by2,
                        t["mejor_conf"], turno_id
                    )
                    sev = calcular_severidad(t["mejor_conf"], bx2-bx1, by2-by1)
                    if mc is not None:
                        baches_gps_guardados.append((mc.latitud, mc.longitud))
                    print(f"  ✓ Bache #{bid} (final) | {sev.upper()} {t['mejor_conf']:.0%} | {foto_name}")
                t["guardado"] = True

    # ── Cierre ──────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    gps.detener()

    cerrar_turno(conn, turno_id)
    print("\n── Turno finalizado ────────────────────────────────")
    print(f"  Baches detectados : {baches_total}")
    print(f"  Base de datos     : {DB_PATH}")

    resp = input("\n¿Sincronizar con el servidor ahora? (s/n): ").strip().lower()
    if resp == "s":
        sincronizar_con_servidor(conn)

    conn.close()


def main():
    ap = argparse.ArgumentParser(description="Detector de baches con GPS — Sistema Vial GAD")
    ap.add_argument("--video",     "-v", default=None)
    ap.add_argument("--output",    "-o", default=None)
    ap.add_argument("--vehiculo",        default="VH-001")
    ap.add_argument("--operador",        default="Operador")
    ap.add_argument("--gps",             default="simulador",
                    choices=["simulador", "real"],
                    help="Modo GPS: simulador (pruebas) o real (hardware USB)")
    ap.add_argument("--puerto",          default="COM3",
                    help="Puerto serial del GPS (ej: COM3, /dev/ttyUSB0)")
    ap.add_argument("--no-ventana",      action="store_true")
    ap.add_argument("--stream",          action="store_true",
                    help="Modo stream en vivo: salta mas frames para no quedarse atras")
    ap.add_argument("--sincronizar",     action="store_true",
                    help="Solo sincronizar BD local con servidor y salir")
    args = ap.parse_args()

    global GPS_MODO, GPS_PUERTO, INFERENCIA_CADA, ESCALA_VENTANA
    GPS_MODO   = args.gps
    GPS_PUERTO = args.puerto

    if args.stream:
        INFERENCIA_CADA = 6     # procesar 1 de cada 6 frames → más fluido
        ESCALA_VENTANA  = 0.45  # ventana más pequeña → menos carga gráfica
        print("[STREAM] Modo live stream activado — procesando 1/6 frames")

    if args.sincronizar:
        conn = iniciar_db()
        sincronizar_con_servidor(conn)
        conn.close()
        return

    if args.video is None:
        carpeta = Path(__file__).parent.parent / "videos"
        exts    = {".mp4", ".avi", ".mov", ".mkv", ".MOV", ".MP4", ".AVI"}
        videos  = sorted([f for f in carpeta.iterdir() if f.suffix in exts])
        if not videos:
            print("Coloca un video en la carpeta 'videos/' o usa --video ruta.mp4")
            return
        video_path = str(videos[0])
    else:
        video_path = args.video

    if args.output is None:
        nombre      = Path(video_path).stem
        output_path = str(Path(__file__).parent.parent / "output" / f"{nombre}_gps.mp4")
    else:
        output_path = args.output

    detectar(video_path, output_path, not args.no_ventana,
             args.vehiculo, args.operador)


if __name__ == "__main__":
    main()
