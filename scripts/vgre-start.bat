@echo off
:: vgre-start.bat — Windows launcher for the VGRE cluster start command.
:: Wraps Start-VGRE.ps1 so the command is callable from any terminal
:: (cmd.exe, PowerShell 5.1, PowerShell 7+, Windows Terminal) as simply
:: 'vgre-start' once %LOCALAPPDATA%\VGRE\scripts is on PATH (set by vgre_sync.bat).
::
:: Usage:
::   vgre-start --master
::   vgre-start --worker
::   vgre-start --worker --master-ip <IP>               LAN: explicit master IP
::   vgre-start --worker --master-address <IP>:<PORT>   WAN: public hostname or IPv6
::   vgre-start --test                                   Local self-test
::   vgre-start --help

setlocal EnableExtensions EnableDelayedExpansion

if not defined VGRE_INSTALL_DIR set "VGRE_INSTALL_DIR=%LOCALAPPDATA%\VGRE"
set "INSTALL_DIR=%VGRE_INSTALL_DIR%"
set "SCRIPTS_DIR=%INSTALL_DIR%\scripts"
set "SCRIPT_DIR=%~dp0"

:: Locate Start-VGRE.ps1 — check next to this .bat, then in installed dir
set "PS1=%SCRIPT_DIR%Start-VGRE.ps1"
if not exist "!PS1!" set "PS1=%SCRIPTS_DIR%\Start-VGRE.ps1"
if not exist "!PS1!" set "PS1=%INSTALL_DIR%\Start-VGRE.ps1"

if not exist "!PS1!" (
    echo [ERROR] Start-VGRE.ps1 not found.
    echo         Run  .\scripts\vgre_sync.bat  from the VGRE repository to install.
    exit /b 1
)

:: Ensure VGRE lib dir is on PATH so vgre-worker.exe finds vgre.dll
set "_HAS_LIB=0"
echo "!PATH!" | findstr /I /C:"%INSTALL_DIR%\lib" >nul 2>&1 && set "_HAS_LIB=1"
if "!_HAS_LIB!"=="0" (
    set "PATH=%INSTALL_DIR%\lib;%INSTALL_DIR%;!PATH!"
)

:: Auto-load auth token from default location if not already set
if not defined VGRE_TCP_AUTH_TOKEN_FILE (
    if exist "%USERPROFILE%\.vgre\token" (
        set "VGRE_TCP_AUTH_TOKEN_FILE=%USERPROFILE%\.vgre\token"
    )
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "!PS1!" %*

exit /b %ERRORLEVEL%
