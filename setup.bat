@echo off
REM Multi-task Deep Learning for Oral Cancer Analysis - Environment Setup
REM =====================================================================

echo 🔬 Setting up Multi-task Deep Learning Environment for Oral Cancer Analysis
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Check if we're in a virtual environment
python -c "import sys; exit(0 if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo 🔄 Creating virtual environment...
    python -m venv venv
    echo ✅ Virtual environment created
    echo.
    echo 🔄 Activating virtual environment...
    call venv\Scripts\activate.bat
    echo ✅ Virtual environment activated
) else (
    echo ✅ Virtual environment detected
)

echo.
echo 🔄 Upgrading pip...
python -m pip install --upgrade pip

echo.
echo 🔄 Installing local development requirements...
pip install -r requirements-local.txt

echo.
echo 🔄 Verifying PyTorch GPU support...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}')"

echo.
echo 🔄 Testing configuration...
python src/config.py

echo.
echo ✅ Setup complete! 
echo.
echo 📝 Next steps:
echo   1. Activate the environment: call venv\Scripts\activate.bat
echo   2. Start Jupyter Lab: jupyter lab
echo   3. Open the Data.ipynb notebook to begin
echo.
echo 💡 Tips:
echo   - Use 'requirements-local.txt' for local development (RTX 4050)
echo   - Use 'requirements.txt' for cloud environments (Colab/Kaggle)
echo   - Check GPU memory usage with: nvidia-smi
echo.

pause