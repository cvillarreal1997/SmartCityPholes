"""
models.py — Modelos de base de datos para el servidor central del GAD.
Usa SQLAlchemy con SQLite (desarrollo) o PostgreSQL (producción).
"""

import secrets
import math
from sqlalchemy import (
    Column, Integer, Float, String, Text, DateTime, Boolean, ForeignKey
)
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()

# Radio máximo para considerar que dos baches son el MISMO (metros)
# 35m cubre variación del simulador GPS; en producción con GPS real se puede bajar a 15m
RADIO_DUPLICADO_M = 35.0


def haversine_metros(lat1, lon1, lat2, lon2) -> float:
    """Distancia en metros entre dos coordenadas GPS (fórmula de Haversine)."""
    R = 6_371_000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class Dispositivo(Base):
    """
    Representa un vehículo/dispositivo municipal registrado.
    Cada vehículo tiene un api_key único para autenticarse al sincronizar.
    """
    __tablename__ = "dispositivos"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    nombre      = Column(String(100), nullable=False)          # ej: "Camioneta VH-001"
    placa       = Column(String(20),  nullable=True)
    api_key     = Column(String(64),  unique=True, nullable=False)
    activo      = Column(Boolean, default=True)
    fecha_alta  = Column(DateTime, default=datetime.utcnow)
    ultimo_sync = Column(DateTime, nullable=True)

    turnos      = relationship("Turno", back_populates="dispositivo")

    @staticmethod
    def generar_api_key() -> str:
        return secrets.token_hex(32)   # 64 caracteres hex seguros

    def to_dict(self) -> dict:
        return {
            "id":          self.id,
            "nombre":      self.nombre,
            "placa":       self.placa,
            "activo":      self.activo,
            "ultimo_sync": self.ultimo_sync.isoformat() if self.ultimo_sync else None,
        }


class Bache(Base):
    __tablename__ = "baches"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    latitud         = Column(Float,  nullable=False)
    longitud        = Column(Float,  nullable=False)
    altitud         = Column(Float,  default=0.0)
    velocidad       = Column(Float,  default=0.0)
    precision_gps   = Column(Float,  default=0.0)

    confianza       = Column(Float,  nullable=False)
    severidad       = Column(String(20), nullable=False)   # leve/moderado/grave/critico
    ancho_px        = Column(Integer, default=0)
    alto_px         = Column(Integer, default=0)

    foto_path       = Column(Text,   nullable=True)
    foto_base64     = Column(Text,   nullable=True)        # foto embebida para sincronización

    fecha           = Column(DateTime, default=datetime.utcnow)
    turno_id        = Column(String(50), ForeignKey("turnos.id"))
    device_id       = Column(Integer, ForeignKey("dispositivos.id"), nullable=True)
    veces_detectado = Column(Integer, default=1)   # cuántos vehículos lo vieron

    # Estado de reparación
    estado          = Column(String(20), default="nuevo")  # nuevo/en_reparacion/reparado
    fecha_reparacion= Column(DateTime, nullable=True)
    costo_estimado  = Column(Float, nullable=True)
    zona            = Column(String(100), nullable=True)   # parroquia/barrio
    observaciones   = Column(Text, nullable=True)

    turno           = relationship("Turno", back_populates="baches")
    dispositivo     = relationship("Dispositivo")

    def to_dict(self) -> dict:
        return {
            "id":           self.id,
            "latitud":      self.latitud,
            "longitud":     self.longitud,
            "altitud":      self.altitud,
            "confianza":    round(self.confianza, 3),
            "severidad":    self.severidad,
            "ancho_px":     self.ancho_px,
            "alto_px":      self.alto_px,
            "fecha":        self.fecha.isoformat() if self.fecha else None,
            "turno_id":     self.turno_id,
            "estado":       self.estado,
            "costo_estimado": self.costo_estimado,
            "zona":         self.zona,
            "tiene_foto":       bool(self.foto_path or self.foto_base64),
            "veces_detectado":  self.veces_detectado,
        }

    def calcular_costo(self) -> float:
        """Estima el costo de reparación según el área del bache."""
        area_m2 = (self.ancho_px * self.alto_px) / (640 * 480)  # normalizado
        if area_m2 > 0.15:
            return round(120 + area_m2 * 280, 2)
        if area_m2 > 0.06:
            return round(30 + area_m2 * 150, 2)
        return round(15 + area_m2 * 100, 2)


class Turno(Base):
    __tablename__ = "turnos"

    id              = Column(String(50), primary_key=True)
    vehiculo        = Column(String(50))
    operador        = Column(String(100))
    inicio          = Column(DateTime, default=datetime.utcnow)
    fin             = Column(DateTime, nullable=True)
    km_recorridos   = Column(Float, default=0.0)
    total_baches    = Column(Integer, default=0)
    zona_operacion  = Column(String(100), nullable=True)
    device_id       = Column(Integer, ForeignKey("dispositivos.id"), nullable=True)

    baches          = relationship("Bache", back_populates="turno")
    dispositivo     = relationship("Dispositivo", back_populates="turnos")

    def to_dict(self) -> dict:
        return {
            "id":           self.id,
            "vehiculo":     self.vehiculo,
            "operador":     self.operador,
            "inicio":       self.inicio.isoformat() if self.inicio else None,
            "fin":          self.fin.isoformat() if self.fin else None,
            "km_recorridos":self.km_recorridos,
            "total_baches": self.total_baches,
            "zona_operacion": self.zona_operacion,
        }
