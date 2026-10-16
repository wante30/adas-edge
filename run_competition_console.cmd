@echo off
chcp 65001 >nul
set "ADAS_PROJECT=%~dp0"
"%ADAS_PROJECT%.venv\Scripts\python.exe" "%ADAS_PROJECT%tools\run_competition_console.py" --video "D:\BaiduNetdiskDownload\紫光同创杯赛题三参考素材\video_20260929_112739.mp4" --paused --mode navigation --mock-hud --split-navigation --profile "%ADAS_PROJECT%configs\competition_native_navigation_112739.json" --window-title "ADAS | Same-frame Navigation"
if errorlevel 1 pause
