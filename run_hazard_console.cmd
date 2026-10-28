@echo off
chcp 65001 >nul
set "ADAS_PROJECT=%~dp0"
"%ADAS_PROJECT%.venv\Scripts\python.exe" "%ADAS_PROJECT%tools\run_competition_console.py" --hazards --mock-hud --paused --window-title "Tesla-BevCam | Hazard Events PC" --profile "%ADAS_PROJECT%configs\competition_native_official.json"
if errorlevel 1 pause
