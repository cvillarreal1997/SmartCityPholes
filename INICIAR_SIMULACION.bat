@echo off
echo ============================================================
echo   Sistema de Gestion Vial Municipal - Simulacion completa
echo ============================================================
echo.

REM Terminal 1: Servidor
start "SERVIDOR" cmd /k "cd /d "%~dp0servidor" && pip install fastapi uvicorn sqlalchemy openpyxl -q && python main.py"

REM Esperar 3 segundos a que el servidor arranque
timeout /t 3 /nobreak >nul

REM Terminal 2: Registrar vehiculo (si no tiene API key)
start "REGISTRO VEHICULO" cmd /k "cd /d "%~dp0" && python registrar_vehiculo.py && pause"

echo.
echo Cuando termines el registro, abre una tercera terminal y ejecuta:
echo   cd vehiculo
echo   python detector_gps.py --gps simulador
echo.
echo Dashboard: http://localhost:8000
echo KPIs:      http://localhost:8000/kpi
echo.
pause
