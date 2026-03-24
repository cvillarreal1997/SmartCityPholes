"""
registrar_vehiculo.py
Registra el vehículo en el servidor y guarda la API key automáticamente
en vehiculo/detector_gps.py para que no tengas que copiarla a mano.

Uso: python registrar_vehiculo.py
"""

import re
import requests
from pathlib import Path

SERVIDOR = "http://localhost:8000"
DETECTOR = Path(__file__).parent / "vehiculo" / "detector_gps.py"


def registrar():
    print("=" * 50)
    print("  Registro de vehículo municipal")
    print("=" * 50)

    nombre = input("Nombre del vehículo [Camioneta VH-001]: ").strip() or "Camioneta VH-001"
    placa  = input("Placa [ABC-1234]: ").strip() or "ABC-1234"

    try:
        r = requests.post(
            f"{SERVIDOR}/api/dispositivos",
            json={"nombre": nombre, "placa": placa},
            timeout=5,
        )
    except requests.exceptions.ConnectionError:
        print("\n❌ No se puede conectar al servidor.")
        print("   Asegúrate de que el servidor esté corriendo:")
        print("   cd servidor && python main.py")
        return

    if r.status_code != 201:
        print(f"\n❌ Error del servidor: {r.text}")
        return

    datos   = r.json()
    api_key = datos["api_key"]

    print(f"\n✅ Vehículo registrado:")
    print(f"   ID      : {datos['id']}")
    print(f"   Nombre  : {datos['nombre']}")
    print(f"   Placa   : {datos['placa']}")
    print(f"   API Key : {api_key}")

    # Parchear automáticamente detector_gps.py
    if DETECTOR.exists():
        contenido  = DETECTOR.read_text(encoding="utf-8")
        nuevo      = re.sub(
            r'DEVICE_API_KEY\s*=\s*"[^"]*"',
            f'DEVICE_API_KEY      = "{api_key}"',
            contenido,
        )
        DETECTOR.write_text(nuevo, encoding="utf-8")
        print(f"\n✅ API key guardada en {DETECTOR}")
        print("\nAhora puedes correr el detector:")
        print("   cd vehiculo && python detector_gps.py --gps simulador")
    else:
        print(f"\n⚠️  No se encontró {DETECTOR}")
        print(f"   Copia la API key manualmente: DEVICE_API_KEY = \"{api_key}\"")


if __name__ == "__main__":
    registrar()
