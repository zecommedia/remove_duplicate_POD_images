@echo off
REM ============================================
REM POD Duplicate Detector - Windows Setup Script
REM ============================================
REM Script này giúp cài đặt đúng cách trên Windows
REM Chạy: setup_windows.bat
REM ============================================

echo.
echo ============================================
echo   POD DUPLICATE DETECTOR - WINDOWS SETUP
echo ============================================
echo.

REM Kiểm tra Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python chua duoc cai dat!
    echo Tai Python tu: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/5] Kiem tra Python version...
python --version

echo.
echo [2/5] Tao virtual environment...
if exist venv (
    echo Virtual environment da ton tai, skip...
) else (
    python -m venv venv
)

echo.
echo [3/5] Kich hoat virtual environment...
call venv\Scripts\activate.bat

echo.
echo [4/5] Cai dat dependencies...
echo.

REM Gỡ PyTorch cũ nếu có
echo Dang go PyTorch cu (neu co)...
pip uninstall torch torchvision torchaudio -y >nul 2>&1

REM Kiểm tra GPU NVIDIA
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo.
    echo [INFO] Phat hien GPU NVIDIA, cai dat PyTorch CUDA...
    echo.
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
) else (
    echo.
    echo [INFO] Khong phat hien GPU NVIDIA, cai dat PyTorch CPU...
    echo.
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

echo.
echo Cai dat cac package con lai...
pip install open-clip-torch Pillow imagehash opencv-python numpy scikit-learn requests tqdm

echo.
echo [5/5] Kiem tra cai dat...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo.
echo ============================================
echo   CAI DAT HOAN TAT!
echo ============================================
echo.
echo Chay detector: python run_detector.py
echo Kiem tra chi tiet: python setup_check.py
echo.
pause
