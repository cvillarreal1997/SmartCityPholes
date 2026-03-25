@echo off
echo ============================================
echo  Sistema de Gestion Vial Municipal
echo  Iniciando servidor central...
echo ============================================
cd /d "%~dp0servidor"
pip install fastapi uvicorn sqlalchemy openpyxl requests -q

REM Abrir servidor en ventana separada
start "Servidor Vial" python main.py

REM Esperar 3 segundos a que arranque
timeout /t 3 /nobreak >nul

REM Registrar vehiculo automaticamente
cd /d "%~dp0"
python registrar_vehiculo.py

echo.
echo Servidor listo en http://localhost:8000
echo.
pause
