@echo off
echo Activating virtual environment...
call .venv\Scripts\activate

echo Installing required packages from requirements.txt...
pip install -r requirements.txt

echo Starting Streamlit app without file watch...
set STREAMLIT_WATCH_FILE_SYSTEM=false
streamlit run main_streamlit.py --server.runOnSave false

pause


