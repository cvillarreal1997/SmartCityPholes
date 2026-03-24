"""
rutas/baches.py — Endpoints REST para gestión de baches.
"""

import base64
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Header
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from database import get_db
from models import Bache, Dispositivo, haversine_metros, RADIO_DUPLICADO_M

router = APIRouter(prefix="/api/baches", tags=["Baches"])

FOTOS_DIR = Path(__file__).parent.parent / "fotos_servidor"
FOTOS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════
#  DEVICE AUTHENTICATION
# ══════════════════════════════════════════════════════════════════

def verificar_dispositivo(
    x_api_key: Optional[str] = Header(None),
    db: Session = Depends(get_db)
) -> Dispositivo:
    """
    Valida el header X-Api-Key contra la tabla dispositivos.
    Todos los endpoints de escritura deben usar esta dependencia.
    """
    if not x_api_key:
        raise HTTPException(401, "Se requiere header X-Api-Key")
    dispositivo = db.query(Dispositivo).filter(
        Dispositivo.api_key == x_api_key,
        Dispositivo.activo  == True
    ).first()
    if not dispositivo:
        raise HTTPException(403, "API key inválida o dispositivo inactivo")
    dispositivo.ultimo_sync = datetime.utcnow()
    db.commit()
    return dispositivo


# ══════════════════════════════════════════════════════════════════
#  SCHEMAS PYDANTIC
# ══════════════════════════════════════════════════════════════════

class BacheEntrada(BaseModel):
    latitud:       float
    longitud:      float
    altitud:       float = 0.0
    velocidad:     float = 0.0
    precision_gps: float = 0.0
    confianza:     float
    severidad:     str
    ancho_px:      int   = 0
    alto_px:       int   = 0
    fecha:         str
    turno_id:      str
    foto_base64:          Optional[str] = None
    foto_panorama_base64: Optional[str] = None
    zona:                 Optional[str] = None
    observaciones:        Optional[str] = None

class BacheActualizacion(BaseModel):
    estado:          Optional[str] = None
    observaciones:   Optional[str] = None
    zona:            Optional[str] = None


# ── Endpoints ────────────────────────────────────────────────────

@router.post("", status_code=201)
def crear_bache(
    data: BacheEntrada,
    dispositivo: Dispositivo = Depends(verificar_dispositivo),
    db: Session = Depends(get_db),
):
    """
    Recibe un bache detectado desde el vehículo.
    1. Verifica autenticación del dispositivo (X-Api-Key).
    2. Duplicate Detection: si ya existe un bache a menos de RADIO_DUPLICADO_M
       metros, incrementa su contador en lugar de crear uno nuevo.
    3. Si es nuevo, guarda foto y crea el registro.
    """
    # ── Duplicate Detection ──────────────────────────────────────
    # Buscar baches existentes en un bounding box aproximado (~50 m)
    # para no recorrer toda la tabla. Luego calcular distancia exacta.
    delta = 0.0005   # ~55 metros en grados de latitud/longitud
    candidatos = db.query(Bache).filter(
        Bache.latitud  >= data.latitud  - delta,
        Bache.latitud  <= data.latitud  + delta,
        Bache.longitud >= data.longitud - delta,
        Bache.longitud <= data.longitud + delta,
        Bache.estado   != "reparado",
    ).all()

    for existente in candidatos:
        distancia = haversine_metros(
            data.latitud, data.longitud,
            existente.latitud, existente.longitud,
        )
        if distancia <= RADIO_DUPLICADO_M:
            # Es el mismo bache — actualizar si la nueva detección es más confiable
            existente.veces_detectado += 1
            if data.confianza > existente.confianza:
                existente.confianza = data.confianza
                # Actualizar severidad si mejoró
                from models import Bache as B
                existente.severidad = data.severidad
            db.commit()
            return {
                "id":         existente.id,
                "duplicado":  True,
                "distancia_m": round(distancia, 1),
                "veces_detectado": existente.veces_detectado,
                "mensaje": f"Bache ya registrado a {distancia:.1f} m — contador actualizado",
            }

    # ── Bache nuevo ──────────────────────────────────────────────
    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:19]

    foto_path = None
    if data.foto_base64:
        foto_path = str(FOTOS_DIR / f"bache_{ts}.jpg")
        with open(foto_path, "wb") as f:
            f.write(base64.b64decode(data.foto_base64))

    if data.foto_panorama_base64:
        panorama_path = str(FOTOS_DIR / f"bache_{ts}_panorama.jpg")
        with open(panorama_path, "wb") as f:
            f.write(base64.b64decode(data.foto_panorama_base64))

    bache = Bache(
        latitud       = data.latitud,
        longitud      = data.longitud,
        altitud       = data.altitud,
        velocidad     = data.velocidad,
        precision_gps = data.precision_gps,
        confianza     = data.confianza,
        severidad     = data.severidad,
        ancho_px      = data.ancho_px,
        alto_px       = data.alto_px,
        fecha         = datetime.fromisoformat(data.fecha),
        turno_id      = data.turno_id,
        device_id     = dispositivo.id,
        foto_path     = foto_path,
        zona          = data.zona,
        observaciones = data.observaciones,
        estado        = "nuevo",
    )
    bache.costo_estimado = bache.calcular_costo()

    db.add(bache)
    db.commit()
    db.refresh(bache)
    return {"id": bache.id, "costo_estimado": bache.costo_estimado}


@router.get("")
def listar_baches(
    severidad: Optional[str] = Query(None),
    estado:    Optional[str] = Query(None),
    zona:      Optional[str] = Query(None),
    limit:     int           = Query(500, le=2000),
    db: Session = Depends(get_db)
):
    """Lista baches con filtros opcionales. Retorna formato GeoJSON para el mapa."""
    q = db.query(Bache)
    if severidad:
        q = q.filter(Bache.severidad == severidad)
    if estado:
        q = q.filter(Bache.estado == estado)
    if zona:
        q = q.filter(Bache.zona.ilike(f"%{zona}%"))

    baches = q.order_by(Bache.fecha.desc()).limit(limit).all()

    # Formato GeoJSON para Leaflet
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [b.longitud, b.latitud]},
            "properties": b.to_dict(),
        }
        for b in baches
    ]
    return {"type": "FeatureCollection", "features": features}


@router.get("/mapa-calor")
def datos_mapa_calor(db: Session = Depends(get_db)):
    """
    Retorna puntos para el mapa de calor de Leaflet.heat.
    Formato: [[lat, lon, intensidad], ...]
    La intensidad considera severidad y confianza.
    """
    baches = db.query(Bache).filter(Bache.estado != "reparado").all()

    peso = {"critico": 1.0, "grave": 0.75, "moderado": 0.5, "leve": 0.25}
    puntos = [
        [b.latitud, b.longitud, peso.get(b.severidad, 0.5) * b.confianza]
        for b in baches
    ]
    return {"puntos": puntos, "total": len(puntos)}


@router.get("/estadisticas")
def estadisticas(db: Session = Depends(get_db)):
    """KPIs para el dashboard del municipio."""
    todos     = db.query(Bache).all()
    total     = len(todos)
    reparados = sum(1 for b in todos if b.estado == "reparado")
    criticos  = sum(1 for b in todos if b.severidad == "critico" and b.estado != "reparado")
    costo_total = sum(b.costo_estimado or 0 for b in todos if b.estado != "reparado")

    por_severidad = {}
    por_estado    = {}
    for b in todos:
        por_severidad[b.severidad] = por_severidad.get(b.severidad, 0) + 1
        por_estado[b.estado]       = por_estado.get(b.estado, 0) + 1

    return {
        "total":            total,
        "reparados":        reparados,
        "pendientes":       total - reparados,
        "criticos_activos": criticos,
        "pct_resolucion":   round(reparados / total * 100, 1) if total else 0,
        "costo_estimado_total": round(costo_total, 2),
        "por_severidad":    por_severidad,
        "por_estado":       por_estado,
    }


@router.patch("/{bache_id}")
def actualizar_bache(bache_id: int, data: BacheActualizacion,
                     db: Session = Depends(get_db)):
    """Actualiza el estado de un bache (en_reparacion, reparado, etc.)."""
    bache = db.query(Bache).filter(Bache.id == bache_id).first()
    if not bache:
        raise HTTPException(404, "Bache no encontrado")

    if data.estado:
        bache.estado = data.estado
        if data.estado == "reparado":
            bache.fecha_reparacion = datetime.utcnow()
    if data.observaciones:
        bache.observaciones = data.observaciones
    if data.zona:
        bache.zona = data.zona

    db.commit()
    return bache.to_dict()


@router.get("/{bache_id}/foto")
def obtener_foto(bache_id: int, db: Session = Depends(get_db)):
    """Retorna el recorte del bache (zoom) como imagen JPEG."""
    bache = db.query(Bache).filter(Bache.id == bache_id).first()
    if not bache:
        raise HTTPException(404, "Bache no encontrado")
    if bache.foto_path and Path(bache.foto_path).exists():
        return FileResponse(bache.foto_path, media_type="image/jpeg")
    raise HTTPException(404, "Foto no disponible")


@router.get("/{bache_id}/panorama")
def obtener_panorama(bache_id: int, db: Session = Depends(get_db)):
    """Retorna el frame completo con el bache marcado (panorama)."""
    bache = db.query(Bache).filter(Bache.id == bache_id).first()
    if not bache:
        raise HTTPException(404, "Bache no encontrado")
    if bache.foto_path:
        # panorama = mismo nombre pero con sufijo _panorama
        panorama = Path(bache.foto_path).with_suffix("").as_posix() + "_panorama.jpg"
        # también buscar en fotos_servidor si el path apunta a otra carpeta
        panorama_local = FOTOS_DIR / (Path(bache.foto_path).stem + "_panorama.jpg")
        for ruta in [panorama, str(panorama_local)]:
            if Path(ruta).exists():
                return FileResponse(ruta, media_type="image/jpeg")
    raise HTTPException(404, "Panorama no disponible")
