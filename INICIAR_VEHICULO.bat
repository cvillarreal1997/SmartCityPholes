@echo off
echo ============================================
echo  Sistema de Gestion Vial Municipal
echo  Iniciando detector en vehiculo...
echo ============================================
cd /d "%~dp0vehiculo"
pip install pyserial pynmea2 requests -q
python detector_gps.py --gps simulador
pause
