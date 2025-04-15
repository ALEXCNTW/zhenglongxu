@echo off
echo -------------------------------------
echo Checking if virtual environment exists...
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

echo -------------------------------------
echo Activating virtual environment...
call .venv\Scripts\activate

echo -------------------------------------
echo Installing dependencies from requirements.txt...
pip install --upgrade pip
pip install -r requirements.txt

echo -------------------------------------
echo Starting Streamlit app...
set STREAMLIT_WATCH_FILE_SYSTEM=false
set STREAMLIT_SERVER_RUN_ON_SAVE=false
streamlit run main_streamlit.py

pause
