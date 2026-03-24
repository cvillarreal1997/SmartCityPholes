@echo off
echo ============================================
echo  Sistema de Gestion Vial Municipal
echo  Iniciando servidor central...
echo ============================================
cd /d "%~dp0servidor"
pip install fastapi uvicorn sqlalchemy openpyxl requests -q
python main.py
pause
