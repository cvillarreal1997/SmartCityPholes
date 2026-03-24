"""
database.py — Configuración de base de datos.
SQLite para desarrollo local, PostgreSQL para producción en el municipio.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from models import Base

# ── Configuración ───────────────────────────────────────────────
# Para producción, cambiar a:
# postgresql://usuario:password@servidor/vial_gad
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./vial_gad.db"
)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def crear_tablas():
    Base.metadata.create_all(bind=engine)
    print("[DB] Tablas creadas/verificadas.")


def get_db():
    """Dependencia FastAPI para obtener sesión de BD."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
