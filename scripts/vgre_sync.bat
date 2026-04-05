@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"
set "BUILD_DIR=%PROJECT_ROOT%\build"
set "DASHBOARD_DIR=%PROJECT_ROOT%\vgre_dashboard"
set "INSTALL_DIR=%LOCALAPPDATA%\VGRE"
set "BUNDLE_DIR=%DASHBOARD_DIR%\build\windows\x64\runner\Release"
set "TOOLS_ROOT=%LOCALAPPDATA%\VGRE\BuildTools"
set "CMAKE_EXE=cmake"
set "CLANG_EXE=clang"
set "FLUTTER_CMD=flutter"
set "VCVARS64="
for %%D in (
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools"
    "C:\Program Files\Microsoft Visual Studio\2022\Community"
    "C:\Program Files\Microsoft Visual Studio\2022\Professional"
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise"
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\Community"
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\Professional"
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\Enterprise"
) do (
    if not defined VCVARS64 if exist "%%~D\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS64=%%~D\VC\Auxiliary\Build\vcvars64.bat"
)

if exist "%TOOLS_ROOT%\cmake\bin\cmake.exe" (
    set "CMAKE_EXE=%TOOLS_ROOT%\cmake\bin\cmake.exe"
    set "PATH=%TOOLS_ROOT%\cmake\bin;%PATH%"
)
if exist "%TOOLS_ROOT%\llvm\bin\clang.exe" (
    set "CLANG_EXE=%TOOLS_ROOT%\llvm\bin\clang.exe"
    set "PATH=%TOOLS_ROOT%\llvm\bin;%PATH%"
)

call :find_flutter

echo === VGRE Global Sync (Windows) ===
echo Project root: %PROJECT_ROOT%

echo Cleaning up stale VGRE processes...
taskkill /F /IM vgre-worker.exe /IM vgre_dashboard.exe /T 2>NUL
rem Ignore errorlevel as processes might not be running

echo Checking CMake...
if exist "%CMAKE_EXE%" (
    rem found explicit executable path
) else (
    where %CMAKE_EXE% >nul 2>&1
    if errorlevel 1 (
        echo ERROR: CMake is required. Run .\Install-BuildTools.ps1 first.
        exit /b 1
    )
)
echo Checking Flutter...
if exist "%FLUTTER_CMD%" (
    rem found explicit executable path
) else (
    where %FLUTTER_CMD% >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Flutter is required to build the dashboard.
        exit /b 1
    )
)

echo Resolving LLVM_DIR...
if "!LLVM_DIR!"=="" (
    for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('LLVM_DIR','User')"`) do set "LLVM_DIR=%%I"
)

if "!LLVM_DIR!"=="" (
    set "LLVM_DIR=%LOCALAPPDATA%\VGRE\BuildTools\llvm\lib\cmake\llvm"
)

if "!LLVM_DIR!"=="" (
    echo ERROR: LLVM_DIR is not set.
    echo Fix: run .\Install-BuildTools.ps1 and open a fresh terminal.
    exit /b 1
)

echo Checking LLVM configuration...
if not exist "!LLVM_DIR!\LLVMConfig.cmake" (
    echo ERROR: LLVM_DIR does not point to a valid LLVM CMake package.
    echo Current LLVM_DIR: !LLVM_DIR!
    exit /b 1
)

if defined VCVARS64 (
    echo Loading Visual Studio Build Tools environment...
    call "!VCVARS64!" >nul
    if errorlevel 1 (
        echo ERROR: Failed to initialize Visual Studio Build Tools.
        exit /b 1
    )
) else (
    echo WARNING: vcvars64.bat was not found at:
    echo   Visual Studio 2022 Build Tools / Community / Professional / Enterprise
    echo Continuing, but CMake may fail to find the Visual Studio toolchain.
)

echo Checking Clang...
if exist "%CLANG_EXE%" (
    rem found explicit executable path
) else (
    where %CLANG_EXE% >nul 2>&1
    if errorlevel 1 (
        echo ERROR: LLVM was not found on PATH after setup. Run .\Install-BuildTools.ps1 again.
        exit /b 1
    )
)

echo.
echo === Tool Versions ===
echo cmake: %CMAKE_EXE%
echo clang: %CLANG_EXE%
echo flutter: %FLUTTER_CMD%
echo LLVM_DIR: !LLVM_DIR!

echo.
echo === Building VGRE Native Engine ===
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
pushd "%BUILD_DIR%" || exit /b 1

"%CMAKE_EXE%" "%PROJECT_ROOT%" -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -DLLVM_DIR="!LLVM_DIR!" -DCMAKE_DISABLE_FIND_PACKAGE_LibXml2=TRUE
if errorlevel 1 (
    popd
    echo ERROR: CMake configure failed.
    exit /b 1
)

"%CMAKE_EXE%" --build . --config Release --target vgre vgre_cudart vgre-worker -- /m:1
if errorlevel 1 (
    popd
    echo ERROR: Native build failed.
    exit /b 1
)
popd

echo.
echo === Building VGRE Dashboard ===
pushd "%DASHBOARD_DIR%" || exit /b 1
powershell -NoProfile -Command "& '%FLUTTER_CMD%' build windows --release"
if errorlevel 1 (
    popd
    echo ERROR: Flutter build failed.
    exit /b 1
)
popd

echo.
echo === Deploying ===
echo Stopping running VGRE processes...
taskkill /IM vgre_dashboard.exe /F >nul 2>&1
taskkill /IM vgre-worker.exe /F >nul 2>&1
timeout /t 1 /nobreak >nul

if not exist "%INSTALL_DIR%\lib" mkdir "%INSTALL_DIR%\lib"
if not exist "%INSTALL_DIR%\include" mkdir "%INSTALL_DIR%\include"

if not exist "%BUNDLE_DIR%\vgre_dashboard.exe" (
    echo ERROR: Dashboard bundle not found at:
    echo   %BUNDLE_DIR%
    exit /b 1
)

xcopy /E /Y /I "%BUNDLE_DIR%" "%INSTALL_DIR%" >nul
if errorlevel 1 (
    echo ERROR: Failed to copy dashboard bundle.
    exit /b 1
)

copy /Y "%BUILD_DIR%\Release\vgre.dll" "%INSTALL_DIR%\" >nul
copy /Y "%BUILD_DIR%\Release\vgre.dll" "%INSTALL_DIR%\lib\" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Failed to copy vgre.dll
    exit /b 1
)

copy /Y "%BUILD_DIR%\Release\vgre_cudart.dll" "%INSTALL_DIR%\" >nul
copy /Y "%BUILD_DIR%\Release\vgre_cudart.dll" "%INSTALL_DIR%\lib\" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Failed to copy vgre_cudart.dll
    exit /b 1
)

copy /Y "%BUILD_DIR%\src\advanced\Release\vgre-worker.exe" "%INSTALL_DIR%\" >nul
if errorlevel 1 (
    echo ERROR: Failed to copy vgre-worker.exe
    exit /b 1
)

xcopy /E /Y /I "%PROJECT_ROOT%\include\vgre" "%INSTALL_DIR%\include\vgre" >nul
if errorlevel 1 (
    echo ERROR: Failed to copy JIT headers.
    exit /b 1
)

echo.
echo === Creating Launcher ===
set "LAUNCHER_PATH=%INSTALL_DIR%\Launch-VGRE-Dashboard.cmd"
(
    echo @echo off
    echo setlocal EnableExtensions
    echo set "APP_DIR=%%~dp0"
    echo set "TOOLS_ROOT=%%LOCALAPPDATA%%\VGRE\BuildTools"
    echo set "PATH=%%APP_DIR%%lib;%%APP_DIR%%;%%TOOLS_ROOT%%\llvm\bin;%%PATH%%"
    echo cd /d "%%APP_DIR%%"
    echo start "" "%%APP_DIR%%vgre_dashboard.exe"
) > "%LAUNCHER_PATH%"
if errorlevel 1 (
    echo ERROR: Failed to create launcher script.
    exit /b 1
)

echo.
echo === Creating Desktop Shortcut ===
set "SHORTCUT_PATH="
set "VGRE_EXE_PATH=%INSTALL_DIR%\vgre_dashboard.exe"
set "VGRE_LAUNCHER_PATH=%LAUNCHER_PATH%"
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "$desktop=[Environment]::GetFolderPath('Desktop'); if([string]::IsNullOrWhiteSpace($desktop) -and $env:OneDrive){$desktop=Join-Path $env:OneDrive 'Desktop'}; if([string]::IsNullOrWhiteSpace($desktop)){$desktop=Join-Path $env:USERPROFILE 'Desktop'}; $target=$env:VGRE_LAUNCHER_PATH; $icon=$env:VGRE_EXE_PATH; $workdir=$env:INSTALL_DIR; if (!(Test-Path -LiteralPath $desktop)) { New-Item -ItemType Directory -Path $desktop -Force | Out-Null }; $shortcut=Join-Path $desktop 'VGRE Dashboard.lnk'; $shell=New-Object -ComObject WScript.Shell; $s=$shell.CreateShortcut($shortcut); $s.TargetPath=$target; $s.WorkingDirectory=$workdir; $s.IconLocation=($icon + ',0'); $s.Save(); Write-Output $shortcut"`) do set "SHORTCUT_PATH=%%I"
if errorlevel 1 (
    echo WARNING: Failed to create desktop shortcut.
)

echo.
echo === Updating System Path ===
for /f "usebackq" %%I in (`powershell -NoProfile -Command "$dir='%INSTALL_DIR%'; $path=[Environment]::GetEnvironmentVariable('Path','User'); if($path -notlike '*'+$dir+'*'){ [Environment]::SetEnvironmentVariable('Path', $path+';'+$dir, 'User'); Write-Output 'CHANGED' } else { Write-Output 'EXISTS' }"`) do set "PATH_STATUS=%%I"

if "%PATH_STATUS%"=="CHANGED" (
    echo ✅ Added %INSTALL_DIR% to your User PATH.
    echo ℹ️  Please RESTART your terminal for the changes to take effect.
) else (
    echo ✅ %INSTALL_DIR% is already in your PATH.
)

echo.
echo VGRE Sync Complete.
echo Installed to: %INSTALL_DIR%
echo Desktop shortcut: %SHORTCUT_PATH%
echo.
echo 🚀 You can now run 'vgre-worker' from any NEW terminal.
pause
exit /b 0

:find_flutter
where flutter >nul 2>&1
if not errorlevel 1 exit /b 0

for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "$roots=@('$env:USERPROFILE\Downloads\Compressed','$env:USERPROFILE\Downloads','$env:USERPROFILE','C:\src','C:\tools'); $hit=Get-ChildItem -Path $roots -Filter flutter.bat -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty DirectoryName; if($hit){$hit}"`) do set "FLUTTER_BIN=%%I"

if defined FLUTTER_BIN (
    set "FLUTTER_CMD=!FLUTTER_BIN!\flutter.bat"
    set "PATH=!FLUTTER_BIN!;%PATH%"
)
exit /b 0
