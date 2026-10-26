@echo off
echo Starting Noise-Robust Music Recommendation System Web App...
cd /d "%~dp0"
"%~dp0venv\Scripts\streamlit.exe" run app.py
pause
