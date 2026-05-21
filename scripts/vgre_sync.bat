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
set "VGRE_ENABLE_NATIVE_SIMD_FLAG=OFF"
set "VCVARS64="
set "VS_YEAR="

rem -- Cache (x86) path prefix at top level so the literal ')' in "(x86)" never
rem    appears inside a parenthesised block — the batch parser counts parens even
rem    inside quoted strings and misidentifies the close-paren as a block end.
set "_PF86=%ProgramFiles(x86)%"
set "_PF64=%ProgramFiles%"

rem -- vswhere.exe (ships with VS 2017+ Installer) gives the most reliable hit -
set "_VSWHERE=!_PF86!\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "!_VSWHERE!" set "_VSWHERE=!_PF64!\Microsoft Visual Studio\Installer\vswhere.exe"

if exist "!_VSWHERE!" (
    for /f "usebackq delims=" %%I in (`"!_VSWHERE!" -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find VC\Auxiliary\Build\vcvars64.bat 2^>nul`) do (
        if not defined VCVARS64 set "VCVARS64=%%I"
    )
    if defined VCVARS64 (
        for /f "usebackq delims=" %%V in (`"!_VSWHERE!" -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationVersion 2^>nul`) do set "_VS_VER=%%V"
        for /f "tokens=1 delims=." %%M in ("!_VS_VER!") do set "VS_MAJOR=%%M"
        if "!VS_MAJOR!"=="15" set "VS_YEAR=2017"
        if "!VS_MAJOR!"=="16" set "VS_YEAR=2019"
        if "!VS_MAJOR!"=="17" set "VS_YEAR=2022"
        if "!VS_MAJOR!"=="18" set "VS_YEAR=2025"
    )
)

rem -- Fallback: enumerate 2022 / 2019 / 2017 in both Program Files locations --
if not defined VCVARS64 (
    for %%Y in (2022 2019 2017) do (
        if not defined VCVARS64 (
            for %%E in (BuildTools Community Professional Enterprise) do (
                if not defined VCVARS64 if exist "!_PF64!\Microsoft Visual Studio\%%Y\%%E\VC\Auxiliary\Build\vcvars64.bat" (
                    set "VCVARS64=!_PF64!\Microsoft Visual Studio\%%Y\%%E\VC\Auxiliary\Build\vcvars64.bat"
                    set "VS_YEAR=%%Y"
                )
                if not defined VCVARS64 if exist "!_PF86!\Microsoft Visual Studio\%%Y\%%E\VC\Auxiliary\Build\vcvars64.bat" (
                    set "VCVARS64=!_PF86!\Microsoft Visual Studio\%%Y\%%E\VC\Auxiliary\Build\vcvars64.bat"
                    set "VS_YEAR=%%Y"
                )
            )
        )
    )
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
if /I "%VGRE_ENABLE_NATIVE_SIMD%"=="1" set "VGRE_ENABLE_NATIVE_SIMD_FLAG=ON"
if /I "%VGRE_ENABLE_NATIVE_SIMD%"=="ON" set "VGRE_ENABLE_NATIVE_SIMD_FLAG=ON"
echo SIMD tuning: %VGRE_ENABLE_NATIVE_SIMD_FLAG% (set VGRE_ENABLE_NATIVE_SIMD=1 to opt-in)

echo.
echo === Auto-loading Cluster Auth Token ===
rem Priority: VGRE_TCP_AUTH_TOKEN_FILE env var > %USERPROFILE%\.vgre\token > VGRE_TCP_AUTH_TOKEN
set "DEFAULT_TOKEN_FILE=%USERPROFILE%\.vgre\token"
if not defined VGRE_TCP_AUTH_TOKEN_FILE if exist "!DEFAULT_TOKEN_FILE!" (
    set "VGRE_TCP_AUTH_TOKEN_FILE=!DEFAULT_TOKEN_FILE!"
    echo Auto-loaded token from !DEFAULT_TOKEN_FILE!
)
if defined VGRE_TCP_AUTH_TOKEN_FILE (
    echo Cluster auth token: ready  ^(!VGRE_TCP_AUTH_TOKEN_FILE!^)
) else if defined VGRE_TCP_AUTH_TOKEN (
    echo VGRE_TCP_AUTH_TOKEN set; will embed in launcher.
) else (
    echo WARNING: No auth token configured. Run Setup-VGRECluster.ps1 to set one up.
    echo          Cluster will use hardware secure storage ^(TPM/CredMan^) or allow manual input.
)

echo.
echo === Verifying Dependencies ===

rem ── Helper: detect winget (Windows Package Manager) ──────────────────────
set "HAS_WINGET=0"
winget --version >nul 2>&1 && set "HAS_WINGET=1"

rem ── Helper: detect chocolatey ─────────────────────────────────────────────
set "HAS_CHOCO=0"
choco --version >nul 2>&1 && set "HAS_CHOCO=1"

rem ── Check and auto-install CMake ──────────────────────────────────────────
if exist "%CMAKE_EXE%" (
    echo [OK] cmake (explicit path: %CMAKE_EXE%)
    goto :cmake_ok
)
where cmake >nul 2>&1
if not errorlevel 1 (
    echo [OK] cmake found on PATH
    goto :cmake_ok
)
echo [MISSING] cmake - attempting auto-install...
if "%HAS_WINGET%"=="1" (
    winget install --id Kitware.CMake --silent --accept-package-agreements --accept-source-agreements
    if not errorlevel 1 (
        set "PATH=C:\Program Files\CMake\bin;!PATH!"
        echo [OK] cmake installed via winget
        goto :cmake_ok
    )
)
if "%HAS_CHOCO%"=="1" (
    choco install cmake --confirm --install-arguments="ADD_CMAKE_TO_PATH=System"
    if not errorlevel 1 (
        echo [OK] cmake installed via chocolatey
        goto :cmake_ok
    )
)
echo [ERROR] cmake not found and auto-install failed.
echo         Install manually: https://cmake.org/download/
echo         Or run: winget install Kitware.CMake
exit /b 1
:cmake_ok

rem ── Check and auto-install LLVM ──────────────────────────────────────────
echo Resolving LLVM_DIR...
if "!LLVM_DIR!"=="" (
    for /f "usebackq delims=" %%I in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "[Environment]::GetEnvironmentVariable('LLVM_DIR','User')"`) do set "LLVM_DIR=%%I"
)
if "!LLVM_DIR!"=="" (
    set "LLVM_DIR=%LOCALAPPDATA%\VGRE\BuildTools\llvm\lib\cmake\llvm"
)
if not exist "!LLVM_DIR!\LLVMConfig.cmake" (
    echo [MISSING] LLVM dev libraries - attempting auto-install...
    if "%HAS_WINGET%"=="1" (
        winget install --id LLVM.LLVM --version 18.1.8 --silent ^
            --accept-package-agreements --accept-source-agreements
        if not errorlevel 1 (
            set "LLVM_DIR=C:\Program Files\LLVM\lib\cmake\llvm"
            echo [OK] LLVM installed via winget
            goto :llvm_ok
        )
    )
    if "%HAS_CHOCO%"=="1" (
        choco install llvm --confirm
        if not errorlevel 1 (
            set "LLVM_DIR=C:\Program Files\LLVM\lib\cmake\llvm"
            echo [OK] LLVM installed via chocolatey
            goto :llvm_ok
        )
    )
    rem Fall back to Install-BuildTools.ps1 in the repo
    if exist "%SCRIPT_DIR%Install-BuildTools.ps1" (
        echo [INFO] Running Install-BuildTools.ps1 to download LLVM...
        powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%Install-BuildTools.ps1" -LLVM
        if not errorlevel 1 (
            set "LLVM_DIR=%LOCALAPPDATA%\VGRE\BuildTools\llvm\lib\cmake\llvm"
            goto :llvm_ok
        )
    )
    echo [ERROR] LLVM not found and auto-install failed.
    echo         Run:  powershell -File scripts\Install-BuildTools.ps1
    echo         Or:   winget install LLVM.LLVM
    exit /b 1
)
:llvm_ok
echo [OK] LLVM at !LLVM_DIR!

rem ── Check Visual Studio Build Tools ──────────────────────────────────────
if defined VCVARS64 (
    echo [OK] Visual Studio Build Tools at !VCVARS64!
) else (
    call :install_vs_buildtools
)

rem ── Check and auto-install Flutter ─────────────────────────────────────────
if exist "%FLUTTER_CMD%" (
    echo [OK] Flutter (explicit path: %FLUTTER_CMD%)
    goto :flutter_ok
)
where flutter >nul 2>&1
if not errorlevel 1 (
    echo [OK] Flutter found on PATH
    goto :flutter_ok
)
echo [MISSING] Flutter - attempting auto-install...
if "%HAS_WINGET%"=="1" (
    winget install --id Google.FlutterSDK --silent ^
        --accept-package-agreements --accept-source-agreements
    if not errorlevel 1 (
        rem Refresh PATH to pick up flutter
        for /f "usebackq delims=" %%I in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:Path"`) do set "PATH=%%I"
        where flutter >nul 2>&1
        if not errorlevel 1 (
            echo [OK] Flutter installed via winget
            goto :flutter_ok
        )
    )
)
if "%HAS_CHOCO%"=="1" (
    choco install flutter --confirm
    if not errorlevel 1 (
        echo [OK] Flutter installed via chocolatey
        goto :flutter_ok
    )
)
echo [WARN] Flutter not found and auto-install failed.
echo        Dashboard build will be SKIPPED.
echo        Install manually: https://flutter.dev/docs/get-started/install/windows
echo        Or: winget install Google.FlutterSDK
set "SKIP_DASHBOARD=1"
:flutter_ok

echo.
echo === Dependency check complete ===

echo.
echo Cleaning up stale VGRE processes...
taskkill /F /IM vgre-worker.exe /IM vgre_dashboard.exe /T 2>NUL
rem Ignore errorlevel as processes might not be running

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
    echo WARNING: vcvars64.bat was not found for any detected Visual Studio version.
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
echo === Installing VGRE CLI Tools (vgre-token, vgre-start) ===
rem Install CLI tools NOW — before the build — so they are available even if
rem the native build fails.  Also updates the current session PATH immediately
rem so the user does not need to restart their terminal.
set "TOKEN_SCRIPT_DIR=%INSTALL_DIR%\scripts"
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"
if not exist "%TOKEN_SCRIPT_DIR%" mkdir "%TOKEN_SCRIPT_DIR%"
copy /Y "%SCRIPT_DIR%vgre-token.ps1"        "%TOKEN_SCRIPT_DIR%\vgre-token.ps1"        >nul 2>&1
copy /Y "%SCRIPT_DIR%vgre-token.bat"        "%TOKEN_SCRIPT_DIR%\vgre-token.bat"        >nul 2>&1
copy /Y "%SCRIPT_DIR%vgre-start.bat"        "%TOKEN_SCRIPT_DIR%\vgre-start.bat"        >nul 2>&1
copy /Y "%SCRIPT_DIR%Start-VGRE.ps1"        "%TOKEN_SCRIPT_DIR%\Start-VGRE.ps1"        >nul 2>&1
copy /Y "%SCRIPT_DIR%Setup-VGRECluster.ps1" "%TOKEN_SCRIPT_DIR%\Setup-VGRECluster.ps1" >nul 2>&1
copy /Y "%SCRIPT_DIR%vgre_env.ps1"          "%TOKEN_SCRIPT_DIR%\vgre_env.ps1"          >nul 2>&1
copy /Y "%SCRIPT_DIR%Install-VGRETools.ps1" "%TOKEN_SCRIPT_DIR%\Install-VGRETools.ps1" >nul 2>&1
echo [OK] vgre-token + vgre-start installed to %TOKEN_SCRIPT_DIR%

rem Update User PATH (persistent across new terminals)
for /f "usebackq" %%I in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$d='%TOKEN_SCRIPT_DIR%';$d2='%INSTALL_DIR%';$p=[Environment]::GetEnvironmentVariable('Path','User');$changed=$false;foreach($dir in @($d2,$d)){if($p -notlike '*'+$dir+'*'){$p+=';'+$dir;$changed=$true}};if($changed){[Environment]::SetEnvironmentVariable('Path',$p,'User');'CHANGED'}else{'EXISTS'}"`) do set "_CLI_PATH_STATUS=%%I"
if "!_CLI_PATH_STATUS!"=="CHANGED" echo [OK] Added CLI tools to User PATH.

rem Update current session PATH too — vgre-token works WITHOUT restarting terminal
if "!PATH!" neq "!PATH:%TOKEN_SCRIPT_DIR%=!" goto :cli_path_ok
set "PATH=%TOKEN_SCRIPT_DIR%;%INSTALL_DIR%;!PATH!"
:cli_path_ok
echo [INFO] vgre-token is available in this terminal and all new terminals.
echo        Run: vgre-token generate

echo.
echo === Building VGRE Native Engine ===
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
pushd "%BUILD_DIR%" || exit /b 1

rem -- Locate Ninja: check PATH first, then the LLVM bin dir we installed -------
set "NINJA_FOUND=0"
where ninja >nul 2>&1
if not errorlevel 1 set "NINJA_FOUND=1"
if "!NINJA_FOUND!"=="0" (
    if exist "%TOOLS_ROOT%\llvm\bin\ninja.exe" (
        set "NINJA_FOUND=1"
        set "PATH=%TOOLS_ROOT%\llvm\bin;!PATH!"
        echo [INFO] Found ninja in LLVM build tools — added to PATH
    )
)

rem -- Detect host architecture for the VS -A platform flag ------------------
set "_CMAKE_VSARCH=x64"
if /I "!PROCESSOR_ARCHITECTURE!"=="ARM64" set "_CMAKE_VSARCH=ARM64"

rem -- Select CMake generator ---------------------------------------------------
rem   Priority: Ninja (if found) > VS MSBuild (if VS detected) > NMake Makefiles
set "CMAKE_GENERATOR=Ninja"
set "CMAKE_ARCH_FLAG="
set "CMAKE_COMPILER_FLAGS="
if "!NINJA_FOUND!"=="0" (
    rem Ninja not available — try VS MSBuild or NMake
    if "!VS_YEAR!"=="2025" ( set "CMAKE_GENERATOR=Visual Studio 18 2025" & set "CMAKE_ARCH_FLAG=-A !_CMAKE_VSARCH!" )
    if "!VS_YEAR!"=="2022" ( set "CMAKE_GENERATOR=Visual Studio 17 2022" & set "CMAKE_ARCH_FLAG=-A !_CMAKE_VSARCH!" )
    if "!VS_YEAR!"=="2019" ( set "CMAKE_GENERATOR=Visual Studio 16 2019" & set "CMAKE_ARCH_FLAG=-A !_CMAKE_VSARCH!" )
    if "!VS_YEAR!"=="2017" ( set "CMAKE_GENERATOR=Visual Studio 15 2017 Win64" )
    rem VS_YEAR empty = no VS found; fall back to NMake (needs cl.exe on PATH)
    if "!VS_YEAR!"=="" set "CMAKE_GENERATOR=NMake Makefiles"
)
rem Ninja without MSVC: tell CMake to use Clang as the C/C++ compiler
if "!CMAKE_GENERATOR!"=="Ninja" if "!VS_YEAR!"=="" (
    set "CMAKE_COMPILER_FLAGS=-DCMAKE_C_COMPILER=clang.exe -DCMAKE_CXX_COMPILER=clang++.exe"
)
echo CMake generator: !CMAKE_GENERATOR!

rem -- Wipe stale CMakeCache when the generator has changed --------------------
rem A mismatch between the cached and requested generator is a hard CMake error.
if exist "%BUILD_DIR%\CMakeCache.txt" (
    set "_cached_gen="
    for /f "usebackq tokens=2 delims==" %%G in (`findstr /B /C:"CMAKE_GENERATOR:INTERNAL=" "%BUILD_DIR%\CMakeCache.txt" 2^>nul`) do set "_cached_gen=%%G"
    if defined _cached_gen (
        if not "!_cached_gen!"=="!CMAKE_GENERATOR!" (
            echo [INFO] Generator changed from "!_cached_gen!" to "!CMAKE_GENERATOR!" — cleaning stale build cache.
            del /f /q "%BUILD_DIR%\CMakeCache.txt" >nul 2>&1
            rd /s /q "%BUILD_DIR%\CMakeFiles" >nul 2>&1
        )
    )
)

rem -- On Windows LAPACK is never searched by default (no system LAPACK ships).
rem    Explicitly pass OFF so that a stale CMakeCache from a Linux build
rem    (which may have VGRE_USE_LAPACK=ON cached) does not cause configure failure.
rem    Users who have Intel oneAPI MKL or vcpkg OpenBLAS installed can override
rem    by running:  cmake ... -DVGRE_USE_LAPACK=ON
rem
rem    Also detect and clear any stale cache where VGRE_USE_LAPACK was cached ON.
if exist "%BUILD_DIR%\CMakeCache.txt" (
    findstr /C:"VGRE_USE_LAPACK:BOOL=ON" "%BUILD_DIR%\CMakeCache.txt" >nul 2>&1
    if not errorlevel 1 (
        echo [INFO] Stale cache has VGRE_USE_LAPACK=ON — clearing to avoid LAPACK search on Windows.
        del /f /q "%BUILD_DIR%\CMakeCache.txt" >nul 2>&1
        rd /s /q "%BUILD_DIR%\CMakeFiles" >nul 2>&1
    )
)

rem -- If a previous configure succeeded but the build failed (vgre.dll absent),
rem    wipe CMakeCache + CMakeFiles so CMake re-reads LLVMTargets-release.cmake
rem    and our DIA SDK patch re-applies cleanly on the next configure.
rem    Without this, Ninja can retain a stale link.txt with the old DIA path.
if exist "%BUILD_DIR%\CMakeCache.txt" (
    set "_vgre_dll_found=0"
    if exist "%BUILD_DIR%\vgre.dll"         set "_vgre_dll_found=1"
    if exist "%BUILD_DIR%\Release\vgre.dll" set "_vgre_dll_found=1"
    if "!_vgre_dll_found!"=="0" (
        echo [INFO] Previous configure exists but vgre.dll is absent — cleaning stale cache
        echo        so LLVM imported-target DIA SDK paths are patched from scratch.
        del /f /q "%BUILD_DIR%\CMakeCache.txt" >nul 2>&1
        rd /s /q "%BUILD_DIR%\CMakeFiles"      >nul 2>&1
    )
)

"%CMAKE_EXE%" "%PROJECT_ROOT%" -G "!CMAKE_GENERATOR!" !CMAKE_ARCH_FLAG! !CMAKE_COMPILER_FLAGS! -DCMAKE_BUILD_TYPE=Release -DLLVM_DIR="!LLVM_DIR!" -DCMAKE_DISABLE_FIND_PACKAGE_LibXml2=TRUE -DVGRE_ENABLE_NATIVE_SIMD=%VGRE_ENABLE_NATIVE_SIMD_FLAG% -DVGRE_USE_LAPACK=OFF
if errorlevel 1 (
    popd
    echo ERROR: CMake configure failed.
    exit /b 1
)

rem --parallel works for all generators (CMake 3.12+):
rem   Ninja / NMake → maps to -j N,  VS MSBuild → maps to /m:N
"%CMAKE_EXE%" --build . --config Release --target vgre vgre_cudart vgre-worker --parallel %NUMBER_OF_PROCESSORS%
if errorlevel 1 (
    popd
    echo ERROR: Native build failed.
    exit /b 1
)
popd

rem -- Resolve DLL and exe output locations based on selected generator --------
rem Ninja and NMake (single-config) emit outputs directly into the build root.
rem VS MSBuild (multi-config) places Release outputs into a Release\ subdir.
set "BUILD_OUT_DIR=!BUILD_DIR!"
set "WORKER_OUT_DIR=!BUILD_DIR!\src\advanced"
if not "!CMAKE_GENERATOR!"=="Ninja" if not "!CMAKE_GENERATOR!"=="NMake Makefiles" (
    set "BUILD_OUT_DIR=!BUILD_DIR!\Release"
    set "WORKER_OUT_DIR=!BUILD_DIR!\src\advanced\Release"
)

echo.
echo === Building VGRE Dashboard ===
if "%SKIP_DASHBOARD%"=="1" (
    echo [SKIP] Flutter not available - dashboard build skipped.
    goto :dashboard_skip
)

rem -- Step 1: precache Windows engine artifacts (flutter_windows.dll, icudtl.dat,
rem    cpp_client_wrapper sources).  Without this, tool_backend.bat cannot find
rem    them and emits "The system cannot find the path specified." (3x) before
rem    failing.  --no-analytics avoids a consent prompt in CI/automation.
echo [INFO] Pre-caching Flutter Windows engine artifacts...
call "!FLUTTER_CMD!" precache --windows --no-analytics
if errorlevel 1 (
    echo [WARN] flutter precache failed - engine DLLs may be missing.
    echo        Run manually: flutter precache --windows
)

rem -- Step 2: resolve Dart packages in the Windows environment.  pubspec.lock
rem    was created on Linux; the local Dart pub cache on this machine may be
rem    empty.  Running pub get here ensures every dependency is downloaded
rem    before cmake configures the Windows build.
pushd "%DASHBOARD_DIR%" || exit /b 1
echo [INFO] Fetching Dart packages (flutter pub get)...
call "!FLUTTER_CMD!" pub get --no-analytics
if errorlevel 1 (
    popd
    echo WARNING: flutter pub get failed. Dashboard build skipped.
    echo          Check internet connectivity and try again.
    set "SKIP_DASHBOARD=1"
    goto :dashboard_skip
)

rem -- Step 3: build the Windows release bundle.
echo [INFO] Building Flutter Windows release bundle...
call "!FLUTTER_CMD!" build windows --release --no-analytics
if errorlevel 1 (
    popd
    echo WARNING: Flutter build failed. Continuing without dashboard.
    echo          Run  flutter build windows --release  from %DASHBOARD_DIR%
    echo          for full build output.
    set "SKIP_DASHBOARD=1"
    goto :dashboard_skip
)
popd
:dashboard_skip

rem -- Detect actual Flutter build output arch (x64 is the default; fall back to
rem    arm64 when running on a Windows-on-ARM machine or when Flutter chose arm64).
rem    This runs outside any block so %VAR% expansion is not affected by parsing.
if "!SKIP_DASHBOARD!"=="1" goto :bundle_dir_ok
if exist "!BUNDLE_DIR!\vgre_dashboard.exe" goto :bundle_dir_ok
if exist "%DASHBOARD_DIR%\build\windows\arm64\runner\Release\vgre_dashboard.exe" (
    set "BUNDLE_DIR=%DASHBOARD_DIR%\build\windows\arm64\runner\Release"
    echo [INFO] Flutter build output detected at arm64 path.
)
:bundle_dir_ok

echo.
echo === Deploying ===
echo Stopping running VGRE processes...
taskkill /IM vgre_dashboard.exe /F >nul 2>&1
taskkill /IM vgre-worker.exe /F >nul 2>&1
timeout /t 1 /nobreak >nul

if not exist "%INSTALL_DIR%\lib" mkdir "%INSTALL_DIR%\lib"
if not exist "%INSTALL_DIR%\include" mkdir "%INSTALL_DIR%\include"

if not "!SKIP_DASHBOARD!"=="1" (
    if not exist "!BUNDLE_DIR!\vgre_dashboard.exe" (
        echo ERROR: Dashboard bundle not found at:
        echo   !BUNDLE_DIR!
        echo         The Flutter build may have failed. Run:
        echo           flutter build windows --release
        echo         from the vgre_dashboard directory for full error output.
        exit /b 1
    )
    xcopy /E /Y /I "!BUNDLE_DIR!" "%INSTALL_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to copy dashboard bundle from !BUNDLE_DIR!
        exit /b 1
    )
    if not exist "%INSTALL_DIR%\vgre_dashboard.exe" (
        echo ERROR: vgre_dashboard.exe missing from %INSTALL_DIR% after xcopy.
        exit /b 1
    )
    echo [OK] Dashboard deployed to %INSTALL_DIR%
) else (
    echo [SKIP] Dashboard bundle deployment skipped ^(Flutter was unavailable^).
)

copy /Y "!BUILD_OUT_DIR!\vgre.dll" "%INSTALL_DIR%\" >nul
copy /Y "!BUILD_OUT_DIR!\vgre.dll" "%INSTALL_DIR%\lib\" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Failed to copy vgre.dll from !BUILD_OUT_DIR!\vgre.dll
    exit /b 1
)

copy /Y "!BUILD_OUT_DIR!\vgre_cudart.dll" "%INSTALL_DIR%\" >nul
copy /Y "!BUILD_OUT_DIR!\vgre_cudart.dll" "%INSTALL_DIR%\lib\" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Failed to copy vgre_cudart.dll from !BUILD_OUT_DIR!\vgre_cudart.dll
    exit /b 1
)

echo.
echo === Validating Deployment ===
if not exist "%INSTALL_DIR%\vgre.dll" (
    echo ERROR: vgre.dll not found in %INSTALL_DIR%
    exit /b 1
) else (
    echo [OK] vgre.dll deployed
)

if not exist "%INSTALL_DIR%\vgre_cudart.dll" (
    echo ERROR: vgre_cudart.dll not found in %INSTALL_DIR%
    exit /b 1
) else (
    echo [OK] vgre_cudart.dll deployed
)

if not exist "%INSTALL_DIR%\lib\vgre.dll" (
    echo WARNING: vgre.dll not found in lib subdirectory ^(may cause issues^)
) else (
    echo [OK] vgre.dll deployed to lib/
)

echo.
echo === Attempting to copy LLVM/OpenMP Runtime Dependencies ===
if exist "%TOOLS_ROOT%\llvm\bin\libomp.dll" (
    echo Copying OpenMP runtime...
    copy /Y "%TOOLS_ROOT%\llvm\bin\libomp.dll" "%INSTALL_DIR%\lib\" >nul 2>&1
    copy /Y "%TOOLS_ROOT%\llvm\bin\libomp.dll" "%INSTALL_DIR%\" >nul 2>&1
) else (
    echo WARNING: OpenMP runtime ^(libomp.dll^) not found. Runtime errors may occur.
)

if exist "%TOOLS_ROOT%\llvm\bin\*.dll" (
    echo Copying LLVM support libraries...
    for %%F in ("%TOOLS_ROOT%\llvm\bin\*.dll") do (
        copy /Y "%%F" "%INSTALL_DIR%\lib\" >nul 2>&1
        copy /Y "%%F" "%INSTALL_DIR%\" >nul 2>&1
    )
)

copy /Y "!WORKER_OUT_DIR!\vgre-worker.exe" "%INSTALL_DIR%\" >nul
if errorlevel 1 (
    echo ERROR: Failed to copy vgre-worker.exe from !WORKER_OUT_DIR!\vgre-worker.exe
    exit /b 1
) else (
    echo [OK] vgre-worker.exe deployed
)

echo.
echo === Configuring vgre-worker PATH ===
rem Ensure vgre-worker can find its DLL dependencies
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"
echo @echo off > "%INSTALL_DIR%\vgre-worker.cmd"
echo setlocal EnableExtensions EnableDelayedExpansion >> "%INSTALL_DIR%\vgre-worker.cmd"
echo set "TOOLS_ROOT=%%LOCALAPPDATA%%\VGRE\BuildTools" >> "%INSTALL_DIR%\vgre-worker.cmd"
echo set "PATH=%%~dp0lib;%%~dp0;%%TOOLS_ROOT%%\llvm\bin;%%PATH%%" >> "%INSTALL_DIR%\vgre-worker.cmd"
echo "%%~dp0vgre-worker.exe" %%* >> "%INSTALL_DIR%\vgre-worker.cmd"

echo.
echo === Creating Launcher ===
rem Write the launcher BEFORE the worker self-check so it is always up-to-date
rem even if the worker fails its startup validation.
set "LAUNCHER_PATH=%INSTALL_DIR%\Launch-VGRE-Dashboard.cmd"
(
    echo @echo off
    echo setlocal EnableExtensions EnableDelayedExpansion
    echo set "APP_DIR=%%~dp0"
    echo set "TOOLS_ROOT=%%LOCALAPPDATA%%\VGRE\BuildTools"
    echo.
    echo rem -- Ensure LLVM, OpenMP, and VGRE libs are found first --
    echo set "PATH=%%APP_DIR%%lib;%%APP_DIR%%;%%TOOLS_ROOT%%\llvm\bin;%%TOOLS_ROOT%%\llvm\lib;%%PATH%%"
    echo.
    echo rem -- Load cluster auth token if not already set --
    echo if not defined VGRE_TCP_AUTH_TOKEN_FILE ^(
    echo     if exist "%%USERPROFILE%%\.vgre\token" set "VGRE_TCP_AUTH_TOKEN_FILE=%%USERPROFILE%%\.vgre\token"
    echo ^)
    echo.
    echo if not exist "%%APP_DIR%%vgre_dashboard.exe" ^(
    echo     echo [ERROR] vgre_dashboard.exe not found in %%APP_DIR%%
    echo     echo         The dashboard was not built or deployed.
    echo     echo         Fix: run  .\scripts\vgre_sync.bat  from the VGRE repository root.
    echo     pause
    echo     exit /b 1
    echo ^)
    echo.
    echo cd /d "%%APP_DIR%%"
    echo start "" "%%APP_DIR%%vgre_dashboard.exe"
) > "%LAUNCHER_PATH%"
if errorlevel 1 (
    echo ERROR: Failed to create launcher script.
    exit /b 1
)
echo [OK] Launcher written to %LAUNCHER_PATH%

echo.
echo === Creating Desktop Shortcut ===
set "SHORTCUT_PATH="
set "VGRE_EXE_PATH=%INSTALL_DIR%\vgre_dashboard.exe"
set "VGRE_LAUNCHER_PATH=%LAUNCHER_PATH%"
for /f "usebackq delims=" %%I in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$desktop=[Environment]::GetFolderPath('Desktop'); if([string]::IsNullOrWhiteSpace($desktop) -and $env:OneDrive){$desktop=Join-Path $env:OneDrive 'Desktop'}; if([string]::IsNullOrWhiteSpace($desktop)){$desktop=Join-Path $env:USERPROFILE 'Desktop'}; $target=$env:VGRE_LAUNCHER_PATH; $icon=$env:VGRE_EXE_PATH; $workdir=$env:INSTALL_DIR; if (!(Test-Path -LiteralPath $desktop)) { New-Item -ItemType Directory -Path $desktop -Force | Out-Null }; $shortcut=Join-Path $desktop 'VGRE Dashboard.lnk'; $shell=New-Object -ComObject WScript.Shell; $s=$shell.CreateShortcut($shortcut); $s.TargetPath=$target; $s.WorkingDirectory=$workdir; $s.IconLocation=($icon + ',0'); $s.Save(); Write-Output $shortcut"`) do set "SHORTCUT_PATH=%%I"
if errorlevel 1 (
    echo WARNING: Failed to create desktop shortcut.
)
if not "!SHORTCUT_PATH!"=="" echo [OK] Desktop shortcut updated: !SHORTCUT_PATH!

echo.
echo === Validating vgre-worker ===
if not exist "%INSTALL_DIR%\vgre-worker.exe" (
    echo ERROR: vgre-worker.exe not found after deployment
    exit /b 1
) else (
    echo [OK] vgre-worker.exe verified
)
set "PATH=%INSTALL_DIR%\lib;%INSTALL_DIR%;%TOOLS_ROOT%\llvm\bin;%PATH%"
"%INSTALL_DIR%\vgre-worker.exe" --help >nul 2>&1
if errorlevel 1 (
    echo ERROR: vgre-worker failed startup self-check.
    echo ERROR: see docs/TROUBLESHOOTING_WINDOWS.md
    exit /b 1
) else (
    echo [OK] vgre-worker startup self-check passed
)

xcopy /E /Y /I "%PROJECT_ROOT%\include\vgre" "%INSTALL_DIR%\include\vgre" >nul
if errorlevel 1 (
    echo ERROR: Failed to copy JIT headers.
    exit /b 1
)

echo.
echo === Updating System Path ===
for /f "usebackq" %%I in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$dir='%INSTALL_DIR%'; $path=[Environment]::GetEnvironmentVariable('Path','User'); if($path -notlike '*'+$dir+'*'){ [Environment]::SetEnvironmentVariable('Path', $path+';'+$dir, 'User'); Write-Output 'CHANGED' } else { Write-Output 'EXISTS' }"`) do set "PATH_STATUS=%%I"

if "%PATH_STATUS%"=="CHANGED" (
    echo [OK] Added %INSTALL_DIR% to your User PATH.
    echo [INFO] Please RESTART your terminal for the changes to take effect.
) else (
    echo [OK] %INSTALL_DIR% is already in your PATH.
)

rem ── Refresh CLI scripts (already installed before the build; this updates them) --
copy /Y "%SCRIPT_DIR%vgre-token.ps1"        "%TOKEN_SCRIPT_DIR%\vgre-token.ps1"        >nul 2>&1
copy /Y "%SCRIPT_DIR%vgre-token.bat"        "%TOKEN_SCRIPT_DIR%\vgre-token.bat"        >nul 2>&1
copy /Y "%SCRIPT_DIR%vgre-start.bat"        "%TOKEN_SCRIPT_DIR%\vgre-start.bat"        >nul 2>&1
copy /Y "%SCRIPT_DIR%Start-VGRE.ps1"        "%TOKEN_SCRIPT_DIR%\Start-VGRE.ps1"        >nul 2>&1
copy /Y "%SCRIPT_DIR%Setup-VGRECluster.ps1" "%TOKEN_SCRIPT_DIR%\Setup-VGRECluster.ps1" >nul 2>&1

echo.
echo ============================================================
echo  VGRE Sync Complete
echo ============================================================
echo  Installed to:  %INSTALL_DIR%
if not "%SHORTCUT_PATH%"=="" echo  Desktop shortcut: %SHORTCUT_PATH%
echo.
if exist "!DEFAULT_TOKEN_FILE!" (
    echo  Auth token:  READY ^(run  vgre-token fingerprint  to verify^)
) else (
    echo  Auth token:  NOT configured
    echo  Run from any NEW terminal:  vgre-token generate
)
echo.
echo  TOKEN MANAGEMENT ^(run from any terminal after restart^):
echo    vgre-token generate          create / rotate auth token
echo    vgre-token fingerprint       show SHA-256 for comparison
echo    vgre-token set ^<TOKEN^>       paste token from master node
echo    vgre-token copy              show copy command for workers
echo    vgre-token verify            check master/worker match
echo.
echo  START COMMANDS:
echo    vgre-start --master                              launch master + dashboard
echo    vgre-start --worker                              launch worker (LAN auto-discover)
echo    vgre-start --worker --master-ip ^<IP^>             connect to specific LAN master
echo    vgre-start --worker --master-address ^<HOST:PORT^>  WAN / hostname / IPv6
echo    vgre-start --test                                local self-test
echo    vgre-start --help                                show all options
echo.
echo  Open a NEW terminal for PATH changes to take effect.
echo ============================================================
pause
exit /b 0

rem =============================================================================
rem  SUBROUTINE: install_vs_buildtools
rem  Called when VCVARS64 is not defined.  Attempts winget install; on success
rem  calls :vs_rescan to locate vcvars64.bat.  Must stay outside all blocks so
rem  that ^ continuation and nested FOR loops work without parser ambiguity.
rem =============================================================================
:install_vs_buildtools
echo [MISSING] Visual Studio Build Tools (C++ workload) - attempting auto-install...
if not "%HAS_WINGET%"=="1" goto :vs_install_done
rem Store the --override value in a variable so the winget call is one line.
rem Putting ^ inside "..." inside a block causes "- was unexpected at this time."
set "_VS_OV=--wait --quiet --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.22621 --includeRecommended"
winget install --id Microsoft.VisualStudio.2022.BuildTools --silent --override "!_VS_OV!" --accept-package-agreements --accept-source-agreements
if errorlevel 1 goto :vs_install_done
echo [OK] Build Tools installed - re-scanning with vswhere...
call :vs_rescan
:vs_install_done
if not defined VCVARS64 (
    echo [WARN] Visual Studio Build Tools not found after install attempt.
    echo        CMake may fail to find the toolchain.
    echo        Install manually: https://visualstudio.microsoft.com/visual-cpp-build-tools/
)
exit /b 0

rem =============================================================================
rem  SUBROUTINE: vs_rescan
rem  Scans for vcvars64.bat via vswhere, then by path enumeration.
rem =============================================================================
:vs_rescan
if exist "!_VSWHERE!" (
    for /f "usebackq delims=" %%I in (`"!_VSWHERE!" -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find VC\Auxiliary\Build\vcvars64.bat 2^>nul`) do (
        if not defined VCVARS64 set "VCVARS64=%%I"
    )
)
if defined VCVARS64 exit /b 0
for %%Y in (2022 2019 2017) do (
    if not defined VCVARS64 (
        for %%E in (BuildTools Community Professional Enterprise) do (
            if not defined VCVARS64 if exist "!_PF64!\Microsoft Visual Studio\%%Y\%%E\VC\Auxiliary\Build\vcvars64.bat" (
                set "VCVARS64=!_PF64!\Microsoft Visual Studio\%%Y\%%E\VC\Auxiliary\Build\vcvars64.bat"
                set "VS_YEAR=%%Y"
            )
            if not defined VCVARS64 if exist "!_PF86!\Microsoft Visual Studio\%%Y\%%E\VC\Auxiliary\Build\vcvars64.bat" (
                set "VCVARS64=!_PF86!\Microsoft Visual Studio\%%Y\%%E\VC\Auxiliary\Build\vcvars64.bat"
                set "VS_YEAR=%%Y"
            )
        )
    )
)
exit /b 0

:find_flutter
where flutter >nul 2>&1
if not errorlevel 1 exit /b 0

for /f "usebackq delims=" %%I in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$up=$env:USERPROFILE; $roots=@($up+'\Downloads\Compressed',$up+'\Downloads',$up,'C:\src','C:\tools'); $hit=Get-ChildItem -Path $roots -Filter flutter.bat -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty DirectoryName; if($hit){$hit}"`) do set "FLUTTER_BIN=%%I"

if defined FLUTTER_BIN (
    set "FLUTTER_CMD=!FLUTTER_BIN!\flutter.bat"
    set "PATH=!FLUTTER_BIN!;%PATH%"
)
exit /b 0
