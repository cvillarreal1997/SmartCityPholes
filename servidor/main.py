"""
main.py — Servidor central del Sistema de Gestión Vial Municipal.
Corre en el servidor del municipio. Los vehículos sincronizan datos aquí.
El dashboard web también se sirve desde aquí.

Ejecutar: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import sys
sys.path.insert(0, str(Path(__file__).parent))
from database import crear_tablas
from rutas.baches       import router as router_baches
from rutas.reportes     import router as router_reportes
from rutas.dispositivos import router as router_dispositivos

# ── Crear app ────────────────────────────────────────────────────
app = FastAPI(
    title="Sistema de Gestión Vial Municipal",
    description="Detección y gestión de baches con GPS para GADs",
    version="1.0.0",
    docs_url="/api/docs",
)

# CORS: permite que el dashboard (mismo servidor o dominio del municipio) acceda
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # en producción, restringir al dominio del municipio
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rutas API ────────────────────────────────────────────────────
app.include_router(router_baches)
app.include_router(router_reportes)
app.include_router(router_dispositivos)

# ── Servir dashboard estático ────────────────────────────────────
DASHBOARD_DIR = Path(__file__).parent.parent / "dashboard"
if DASHBOARD_DIR.exists():
    static_dir = DASHBOARD_DIR / "static"
    static_dir.mkdir(exist_ok=True)   # crea la carpeta si no existe
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", include_in_schema=False)
    def dashboard_mapa():
        return FileResponse(str(DASHBOARD_DIR / "index.html"))

    @app.get("/kpi", include_in_schema=False)
    def dashboard_kpi():
        return FileResponse(str(DASHBOARD_DIR / "kpi.html"))


# ── Inicialización ───────────────────────────────────────────────
@app.on_event("startup")
def startup():
    crear_tablas()
    print("=" * 55)
    print("  Sistema de Gestión Vial Municipal — Servidor activo")
    print("  Dashboard : http://localhost:8000")
    print("  KPIs      : http://localhost:8000/kpi")
    print("  API docs  : http://localhost:8000/api/docs")
    print("=" * 55)


@app.get("/api/salud")
def salud():
    return {"estado": "activo", "sistema": "Gestión Vial GAD"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
