@echo off
:: vgre-token.bat — Windows launcher for vgre-token.ps1
:: Allows running  vgre-token <command>  from cmd.exe or any terminal that
:: has %LOCALAPPDATA%\VGRE\scripts in PATH (set by setup scripts).
::
:: Usage:  vgre-token generate | show | fingerprint | set TOKEN | verify | copy | revoke

setlocal

:: Locate the PowerShell script next to this .bat file
set "SCRIPT_DIR=%~dp0"
set "PS1=%SCRIPT_DIR%vgre-token.ps1"

if not exist "%PS1%" (
    :: Fall-back: look in %LOCALAPPDATA%\VGRE\scripts
    set "PS1=%LOCALAPPDATA%\VGRE\scripts\vgre-token.ps1"
)

if not exist "%PS1%" (
    echo [ERROR] vgre-token.ps1 not found. Re-run the VGRE installer.
    exit /b 1
)

:: Forward all arguments to the PowerShell script.
:: -ExecutionPolicy Bypass: runs unsigned scripts without changing system policy.
:: -NoLogo -NoProfile:      faster startup.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ^
    -File "%PS1%" %*

exit /b %ERRORLEVEL%
