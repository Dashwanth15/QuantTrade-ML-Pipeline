@echo off
:: ============================================================
:: QuantTrade ML Pipeline — One-click venv setup
:: Requirements: Python 3.14+ from https://www.python.org/
:: ============================================================
echo [1/4] Removing old .venv (if any)...
rmdir /s /q .venv 2>nul

echo [2/4] Creating fresh virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo ERROR: Could not create virtual environment.
    echo Make sure Python is installed from https://www.python.org/
    pause & exit /b 1
)

echo [3/4] Installing all dependencies (binary wheels only, no compiler needed)...
.venv\Scripts\pip install --upgrade pip --quiet
.venv\Scripts\pip install -r requirements.txt --only-binary=:all:
if errorlevel 1 (
    :: ta is pure-Python and only ships as sdist — install separately
    .venv\Scripts\pip install ta
)

echo [4/4] Verifying installation...
.venv\Scripts\python -c "import xgboost, shap, optuna, streamlit, pandas, numpy; print('All core packages OK')"
if errorlevel 1 (
    echo WARNING: Some packages may not have installed correctly.
    pause & exit /b 1
)

echo.
echo ============================================================
echo  Setup complete!  Activate with: .venv\Scripts\activate
echo  Run tests:       .venv\Scripts\pytest tests\
echo  Run dashboard:   .venv\Scripts\streamlit run app\Main.py
echo ============================================================
pause
