@echo off
cd /d "%~dp0"
"%~dp0sam3\.venv\Scripts\python.exe" -m uvicorn server.main:app --host 0.0.0.0 --port 8765
