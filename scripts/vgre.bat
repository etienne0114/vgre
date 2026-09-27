@echo off
:: vgre.bat — Windows launcher for vgre.ps1 (the unified `vgre` command).
:: Allows running  vgre <command>  from cmd.exe or any terminal that has
:: %LOCALAPPDATA%\VGRE\scripts on PATH (set by the setup scripts).
::
:: Usage:  vgre version | info | generate | train | tokenize
::         vgre start | worker | token | discover | dashboard

setlocal

:: Locate the PowerShell script next to this .bat file
set "SCRIPT_DIR=%~dp0"
set "PS1=%SCRIPT_DIR%vgre.ps1"

if not exist "%PS1%" (
    :: Fall-back: look in %LOCALAPPDATA%\VGRE\scripts
    set "PS1=%LOCALAPPDATA%\VGRE\scripts\vgre.ps1"
)

if not exist "%PS1%" (
    echo [ERROR] vgre.ps1 not found. Re-run the VGRE installer.
    exit /b 1
)

:: Forward all arguments to the PowerShell script.
:: -ExecutionPolicy Bypass: runs unsigned scripts without changing system policy.
:: -NoLogo -NoProfile:      faster startup.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ^
    -File "%PS1%" %*

exit /b %ERRORLEVEL%
