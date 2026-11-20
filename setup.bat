@echo off
echo =====================================================
echo  FaceVerify - Setup Instalasi
echo =====================================================
echo.
echo Langkah 1: Membuat virtual environment Python...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Python tidak ditemukan. Install Python 3.10+ dulu.
    pause
    exit /b 1
)

echo.
echo Langkah 2: Aktivasi virtual environment...
call venv\Scripts\activate.bat

echo.
echo Langkah 3: Update pip...
python -m pip install --upgrade pip

echo.
echo Langkah 4: Install semua dependensi...
pip install -r requirements.txt

echo.
echo =====================================================
echo  Instalasi selesai!
echo =====================================================
echo.
echo Cara menjalankan:
echo   1. Aktifkan venv  : venv\Scripts\activate.bat
echo   2. Jalankan server: python run.py
echo   3. Buka browser   : http://localhost:8000
echo.
echo CATATAN: InsightFace model buffalo_l (~326MB) akan
echo          diunduh otomatis saat server pertama kali jalan.
echo.
pause
