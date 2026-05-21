@echo off
:: vgre-discover.bat -- Windows launcher for vgre-discover.ps1
:: Allows running  vgre-discover [--show|--set-master|--register|--find|--unregister]
:: from any terminal once %LOCALAPPDATA%\VGRE\scripts is on PATH.

setlocal EnableExtensions EnableDelayedExpansion

set "INSTALL_DIR=%LOCALAPPDATA%\VGRE"
set "SCRIPT_DIR=%~dp0"

:: Locate vgre-discover.ps1
set "PS1=%SCRIPT_DIR%vgre-discover.ps1"
if not exist "!PS1!" set "PS1=%INSTALL_DIR%\scripts\vgre-discover.ps1"
if not exist "!PS1!" set "PS1=%INSTALL_DIR%\vgre-discover.ps1"

if not exist "!PS1!" (
    echo [ERROR] vgre-discover.ps1 not found.
    echo         Run  .\scripts\vgre_sync.bat  from the VGRE repository to install.
    exit /b 1
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "!PS1!" %*

exit /b %ERRORLEVEL%
