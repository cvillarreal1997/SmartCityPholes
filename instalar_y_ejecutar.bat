@echo off
echo ========================================
echo   Detector de Baches - YOLOv8 + IA
echo ========================================
echo.
echo Instalando dependencias...
pip install -r requirements.txt
echo.
echo Ejecutando detector...
echo (Primera vez descarga el modelo ~50 MB automaticamente)
echo.
python detector_baches.py
pause
