@echo off
setlocal enabledelayedexpansion

:: VGRE Sync & Install Script (Windows)
:: This script builds the VGRE engine and dashboard and installs it to the local user profile.

echo 🚀 Starting VGRE Global Sync (Windows)...

:: 1. Build Native Engine
echo 📦 Building VGRE Native Engine...
if not exist build mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release --target vgre vgre_cudart vgre-worker
cd ..

:: 2. Build Flutter Dashboard
echo 🎨 Building VGRE Dashboard (Release)...
cd vgre_dashboard
call flutter build windows --release
cd ..

:: 3. Prepare Installation Directory
set INSTALL_DIR=%LOCALAPPDATA%\VGRE
echo 📂 Deploying to %INSTALL_DIR%...
if not exist "%INSTALL_DIR%\lib" mkdir "%INSTALL_DIR%\lib"

:: Windows deployment
set BUNDLE_DIR=vgre_dashboard\build\windows\x64\runner\Release
xcopy /E /Y /I "%BUNDLE_DIR%" "%INSTALL_DIR%"
copy /Y build\Release\vgre.dll "%INSTALL_DIR%\lib\"
copy /Y build\Release\vgre_cudart.dll "%INSTALL_DIR%\lib\"
copy /Y build\Release\vgre-worker.exe "%INSTALL_DIR%\"

:: Copy essential JIT headers
if not exist "%INSTALL_DIR%\include" mkdir "%INSTALL_DIR%\include"
xcopy /E /Y /I "include\vgre" "%INSTALL_DIR%\include\vgre"

:: Create Desktop Shortcut using PowerShell
set SCRIPT_PATH=%INSTALL_DIR%\vgre_dashboard.exe
set SHORTCUT_PATH=%USERPROFILE%\Desktop\VGRE Dashboard.lnk
powershell -Command "$s=(New-Object -COM WScript.Shell).CreateShortcut('%SHORTCUT_PATH%');$s.TargetPath='%SCRIPT_PATH%';$s.IconLocation='%SCRIPT_PATH%,0';$s.Save()"

echo ✅ VGRE Sync Complete!
echo 📍 Installed to: %INSTALL_DIR%
echo 🚀 A shortcut has been created on your Desktop.
pause
