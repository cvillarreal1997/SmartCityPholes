# SmartCityPotholes — Sistema de Gestión Vial Municipal

Detección automática de baches con YOLOv8 + GPS para vehículos municipales (GADs Ecuador).

## Arquitectura

```
Vehículo (cámara + GPS)  →  FastAPI Servidor  →  Dashboard Web (mapa + KPIs)
```

---

## Requisitos

- Python 3.10+
- GPU NVIDIA recomendada (funciona sin GPU también)
- CUDA 11.8+ (opcional, para GPU)

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/cvillarreal1997/SmartCityPholes.git
cd SmartCityPholes
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Descargar el modelo entrenado

El modelo `mi_modelo_baches.pt` no está en el repositorio por su tamaño.
Descárgalo desde: **[Google Drive / HuggingFace — agregar link aquí]**
y colócalo en la raíz del proyecto.

> Si no tienes el modelo, el sistema usará automáticamente el modelo público de HuggingFace como respaldo.

---

## Uso

### Paso 1 — Iniciar el servidor

```bash
cd servidor
python main.py
```

El servidor queda corriendo en: `http://localhost:8000`
Dashboard: `http://localhost:8000/dashboard`

### Paso 2 — Registrar el vehículo

```bash
python registrar_vehiculo.py
```

Esto genera una API key y la configura automáticamente en el detector.

### Paso 3 — Ejecutar el detector

**Con video de prueba:**
```bash
python vehiculo/detector_gps.py --video videos/holes.mp4
```

**Con cámara en tiempo real:**
```bash
python vehiculo/detector_gps.py
```

**Con GPS real (puerto COM):**
```bash
python vehiculo/detector_gps.py --gps real --puerto COM3
```

**Controles durante la detección:**
- `Q` — salir
- `S` — sincronizar con servidor ahora
- `P` — pausar/reanudar

### Paso 4 — Ver el dashboard

Abre el navegador en: `http://localhost:8000/dashboard`

---

## Estructura del proyecto

```
├── vehiculo/
│   ├── detector_gps.py     # Detector principal (YOLO + GPS + SQLite)
│   └── gps_reader.py       # Lector GPS real y simulador
├── servidor/
│   ├── main.py             # FastAPI servidor
│   ├── models.py           # Modelos de BD (SQLAlchemy)
│   ├── database.py         # Conexión BD (SQLite dev / PostgreSQL prod)
│   └── rutas/              # Endpoints REST
│       ├── baches.py
│       ├── dispositivos.py
│       └── reportes.py
├── dashboard/
│   ├── index.html          # Mapa de calor (Leaflet.js)
│   └── kpi.html            # Panel KPIs (Chart.js)
├── registrar_vehiculo.py   # Registro de vehículo y API key
├── requirements.txt
└── mi_modelo_baches.pt     # Modelo YOLOv8 (descargar aparte)
```

---

## Base de datos

Por defecto usa **SQLite** (sin configuración). Para producción con PostgreSQL:

```bash
export DATABASE_URL="postgresql://usuario:password@localhost/vial_gad"
```

---

## Variables de entorno (opcionales)

| Variable | Descripción | Default |
|----------|-------------|---------|
| `DATABASE_URL` | URL de base de datos | SQLite local |
| `SERVER_URL` | URL del servidor desde el vehículo | `http://localhost:8000` |

---

## Scripts de inicio rápido (Windows)

- `INICIAR_SERVIDOR.bat` — inicia el servidor
- `INICIAR_VEHICULO.bat` — inicia el detector con simulador GPS
- `INICIAR_SIMULACION.bat` — abre servidor + registro en terminales separadas
