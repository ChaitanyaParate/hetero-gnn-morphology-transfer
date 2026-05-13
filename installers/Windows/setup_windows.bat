@echo off
:: ==============================================================================
:: setup_windows.bat — Entry point launcher for setup_windows.ps1
:: Double-click this file to start the setup. Requests Admin rights automatically.
:: ==============================================================================
title Hetero-GNN Morphology Transfer — Windows Setup

:: Request Administrator privileges if not already elevated
net session >nul 2>&1
if %errorLevel% NEQ 0 (
    echo Requesting Administrator privileges...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

:: Run the PowerShell script located in the same folder as this .bat file
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1"
pause
