# Cómo simular el sistema completo en local

## Lo que vas a tener corriendo

```
Terminal 1: Servidor FastAPI  → http://localhost:8000
Terminal 2: Detector vehículo → procesa holes.mp4 con GPS simulado
Navegador:  Dashboard         → http://localhost:8000
                                http://localhost:8000/kpi
```

---

## PASO 1 — Instalar dependencias (una sola vez)

```bash
pip install fastapi uvicorn sqlalchemy openpyxl requests pyserial pynmea2
```

---

## PASO 2 — Iniciar el servidor (Terminal 1)

```bash
cd "c:/Users/terry/Desktop/ProyectosExtras/INTENTOS IA/CARRETERA/servidor"
python main.py
```

Verás:
```
Sistema de Gestión Vial Municipal — Servidor activo
Dashboard : http://localhost:8000
KPIs      : http://localhost:8000/kpi
API docs  : http://localhost:8000/api/docs
```

---

## PASO 3 — Registrar el vehículo y obtener API key (una sola vez)

Abre otra terminal y ejecuta:

```bash
cd "c:/Users/terry/Desktop/ProyectosExtras/INTENTOS IA/CARRETERA"
python registrar_vehiculo.py
```

Recibirás algo así:
```
Vehículo registrado:
  ID      : 1
  Nombre  : Camioneta VH-001
  API Key : a3f9c2d1e8b7...   ← se guarda automáticamente
```

El script guarda la API key en vehiculo/detector_gps.py automáticamente.

---

## PASO 4 — Correr el detector (Terminal 2)

```bash
cd "c:/Users/terry/Desktop/ProyectosExtras/INTENTOS IA/CARRETERA/vehiculo"
python detector_gps.py --gps simulador
```

Verás la ventana de video con los baches detectados en tiempo real.
Al terminar el video (o presionar Q), escribe `s` para sincronizar con el servidor.

---

## PASO 5 — Ver el dashboard

Abre el navegador en:
- **Mapa de calor** → http://localhost:8000
- **KPIs / Estadísticas** → http://localhost:8000/kpi
- **API interactiva** → http://localhost:8000/api/docs

---

## Teclas durante el detector

| Tecla | Acción |
|-------|--------|
| `Q`   | Detener el video |
| `S`   | Sincronizar ahora con el servidor (sin esperar al final) |
