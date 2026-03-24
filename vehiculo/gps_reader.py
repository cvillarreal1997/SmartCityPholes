"""
gps_reader.py — Lector de GPS para el vehículo municipal
Soporta:
  - GPS real via puerto serial (USB NMEA, ej: u-blox, GlobalSat)
  - Modo simulador para pruebas sin hardware GPS
"""

import threading
import time
import random
from dataclasses import dataclass, field
from datetime import datetime

try:
    import serial
    import pynmea2
    GPS_LIBS_OK = True
except ImportError:
    GPS_LIBS_OK = False


@dataclass
class CoordenadaGPS:
    latitud:   float
    longitud:  float
    altitud:   float  = 0.0
    velocidad: float  = 0.0   # km/h
    precision: float  = 0.0   # HDOP (menor = mejor)
    timestamp: datetime = field(default_factory=datetime.now)
    valida:    bool   = True

    def to_dict(self) -> dict:
        return {
            "latitud":   self.latitud,
            "longitud":  self.longitud,
            "altitud":   self.altitud,
            "velocidad": self.velocidad,
            "precision": self.precision,
            "timestamp": self.timestamp.isoformat(),
        }


class LectorGPS:
    """Lee coordenadas GPS desde puerto serial (dispositivo USB NMEA)."""

    def __init__(self, puerto: str = "COM3", baudrate: int = 9600):
        if not GPS_LIBS_OK:
            raise ImportError("Instala: pip install pyserial pynmea2")
        self.puerto   = puerto
        self.baudrate = baudrate
        self._coord   = None
        self._lock    = threading.Lock()
        self._activo  = False
        self._hilo    = None

    def iniciar(self):
        self._activo = True
        self._hilo   = threading.Thread(target=self._leer_loop, daemon=True)
        self._hilo.start()
        print(f"[GPS] Leyendo desde {self.puerto} @ {self.baudrate} baud")

    def detener(self):
        self._activo = False

    def coordenada_actual(self) -> CoordenadaGPS | None:
        with self._lock:
            return self._coord

    def _leer_loop(self):
        try:
            with serial.Serial(self.puerto, self.baudrate, timeout=1) as ser:
                while self._activo:
                    linea = ser.readline().decode("ascii", errors="replace").strip()
                    if linea.startswith("$GNGGA") or linea.startswith("$GPGGA"):
                        self._parsear_gga(linea)
                    elif linea.startswith("$GNRMC") or linea.startswith("$GPRMC"):
                        self._parsear_rmc(linea)
        except Exception as e:
            print(f"[GPS] Error serial: {e}")

    def _parsear_gga(self, linea: str):
        try:
            msg = pynmea2.parse(linea)
            if msg.gps_qual == 0:
                return   # sin fix
            with self._lock:
                if self._coord is None:
                    self._coord = CoordenadaGPS(latitud=msg.latitude,
                                                longitud=msg.longitude)
                self._coord.latitud   = msg.latitude
                self._coord.longitud  = msg.longitude
                self._coord.altitud   = float(msg.altitude or 0)
                self._coord.precision = float(msg.horizontal_dil or 99)
                self._coord.timestamp = datetime.now()
                self._coord.valida    = True
        except Exception:
            pass

    def _parsear_rmc(self, linea: str):
        try:
            msg = pynmea2.parse(linea)
            if msg.status != "A":
                return
            with self._lock:
                if self._coord:
                    self._coord.velocidad = float(msg.spd_over_grnd or 0) * 1.852  # nudos→km/h
        except Exception:
            pass


class SimuladorGPS:
    """
    Simula un recorrido GPS para desarrollo y pruebas sin hardware.
    Traza una ruta por calles de Quito por defecto.
    """

    RUTA_QUITO = [
        (-0.2201, -78.5123), (-0.2195, -78.5115), (-0.2188, -78.5107),
        (-0.2180, -78.5098), (-0.2172, -78.5089), (-0.2165, -78.5080),
        (-0.2158, -78.5072), (-0.2150, -78.5063), (-0.2143, -78.5055),
        (-0.2136, -78.5047), (-0.2128, -78.5038), (-0.2120, -78.5030),
        (-0.2113, -78.5022), (-0.2105, -78.5013), (-0.2098, -78.5005),
    ]

    def __init__(self, intervalo_seg: float = 0.5, variacion_m: float = 3.0):
        self._intervalo  = intervalo_seg
        self._variacion  = variacion_m / 111_000   # metros → grados aprox.
        self._coord      = None
        self._lock       = threading.Lock()
        self._activo     = False
        self._hilo       = None
        self._idx        = 0

    def iniciar(self):
        self._activo = True
        self._hilo   = threading.Thread(target=self._simular_loop, daemon=True)
        self._hilo.start()
        print("[GPS-SIM] Simulador GPS iniciado - ruta por Quito")

    def detener(self):
        self._activo = False

    def coordenada_actual(self) -> CoordenadaGPS | None:
        with self._lock:
            return self._coord

    def _simular_loop(self):
        ruta = self.RUTA_QUITO
        while self._activo:
            lat, lon = ruta[self._idx % len(ruta)]
            # pequeña variación aleatoria (simula ruido GPS real)
            lat += random.gauss(0, self._variacion)
            lon += random.gauss(0, self._variacion)

            with self._lock:
                self._coord = CoordenadaGPS(
                    latitud   = lat,
                    longitud  = lon,
                    altitud   = 2850.0 + random.uniform(-5, 5),  # Quito ~2850 msnm
                    velocidad = random.uniform(20, 50),
                    precision = random.uniform(0.8, 2.0),
                    timestamp = datetime.now(),
                    valida    = True,
                )
            self._idx += 1
            time.sleep(self._intervalo)


def crear_gps(modo: str = "simulador", puerto: str = "COM3") -> LectorGPS | SimuladorGPS:
    """
    Fábrica de GPS.

    modo="simulador"  → sin hardware, para desarrollo
    modo="real"       → GPS USB conectado al puerto indicado
    """
    if modo == "real":
        gps = LectorGPS(puerto=puerto)
    else:
        gps = SimuladorGPS()
    gps.iniciar()
    return gps


if __name__ == "__main__":
    print("Probando simulador GPS...")
    gps = crear_gps("simulador")
    for _ in range(5):
        time.sleep(1)
        c = gps.coordenada_actual()
        if c:
            print(f"  Lat: {c.latitud:.6f}  Lon: {c.longitud:.6f}  "
                  f"Vel: {c.velocidad:.1f} km/h  HDOP: {c.precision:.1f}")
    gps.detener()
    print("OK")
