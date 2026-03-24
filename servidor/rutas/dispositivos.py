"""
rutas/dispositivos.py — Gestión de vehículos/dispositivos municipales.
El administrador registra cada vehículo aquí y obtiene su API key.
"""

from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from database import get_db
from models import Dispositivo

router = APIRouter(prefix="/api/dispositivos", tags=["Dispositivos"])


class DispositivoEntrada(BaseModel):
    nombre: str
    placa:  Optional[str] = None


@router.post("", status_code=201)
def registrar_dispositivo(data: DispositivoEntrada, db: Session = Depends(get_db)):
    """
    Registra un vehículo municipal nuevo y genera su API key.
    Guardar la api_key — no se puede recuperar después.
    """
    api_key = Dispositivo.generar_api_key()
    disp = Dispositivo(nombre=data.nombre, placa=data.placa, api_key=api_key)
    db.add(disp)
    db.commit()
    db.refresh(disp)
    return {
        "id":      disp.id,
        "nombre":  disp.nombre,
        "placa":   disp.placa,
        "api_key": api_key,   # mostrar solo una vez al crear
        "mensaje": "Guarda la api_key en el vehículo. No se puede recuperar después.",
    }


@router.get("")
def listar_dispositivos(db: Session = Depends(get_db)):
    """Lista todos los vehículos registrados (sin mostrar api_key)."""
    return [d.to_dict() for d in db.query(Dispositivo).all()]


@router.patch("/{device_id}/activar")
def activar_desactivar(device_id: int, activo: bool, db: Session = Depends(get_db)):
    """Activa o desactiva un vehículo (su api_key deja de funcionar si está inactivo)."""
    d = db.query(Dispositivo).filter(Dispositivo.id == device_id).first()
    if not d:
        raise HTTPException(404, "Dispositivo no encontrado")
    d.activo = activo
    db.commit()
    return {"id": d.id, "activo": d.activo}
