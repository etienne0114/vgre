@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"
set "BUILD_DIR=%PROJECT_ROOT%\build"
set "DASHBOARD_DIR=%PROJECT_ROOT%\vgre_dashboard"
if not defined VGRE_INSTALL_DIR set "VGRE_INSTALL_DIR=%LOCALAPPDATA%\VGRE"
set "INSTALL_DIR=%VGRE_INSTALL_DIR%"
set "BUNDLE_DIR=%DASHBOARD_DIR%\build\windows\x64\runner\Release"
if not defined VGRE_TOOLS_ROOT set "VGRE_TOOLS_ROOT=%INSTALL_DIR%\BuildTools"
set "TOOLS_ROOT=%VGRE_TOOLS_ROOT%"
set "CMAKE_EXE=cmake"
set "CLANG_EXE=clang"
set "FLUTTER_CMD=flutter"
set "VGRE_ENABLE_NATIVE_SIMD_FLAG=OFF"
set "VCVARS64="
set "VS_YEAR="

rem -- Cache (x86) path prefix at top level so the literal ')' in "(x86)" never
rem    appears inside a parenthesised block - the batch parser counts parens even
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

rem -- Early PATH enrichment: add known-good VGRE BuildTools only if present --
rem    (full auto-detection happens in the Verifying Dependencies section below)
if exist "%TOOLS_ROOT%\cmake\bin\cmake.exe" (
    set "CMAKE_EXE=%TOOLS_ROOT%\cmake\bin\cmake.exe"
    set "PATH=%TOOLS_ROOT%\cmake\bin;%PATH%"
)
if exist "%TOOLS_ROOT%\llvm\bin\clang.exe" (
    set "CLANG_EXE=%TOOLS_ROOT%\llvm\bin\clang.exe"
    set "PATH=%TOOLS_ROOT%\llvm\bin;%PATH%"
)
rem -- Scoop installs (user-local, often on PATH already) ---------------------
for /f "usebackq delims=" %%P in (`where cmake 2^>nul`) do (
    if not defined CMAKE_EXE set "CMAKE_EXE=%%P"
)
for /f "usebackq delims=" %%P in (`where clang 2^>nul`) do (
    if not defined CLANG_EXE set "CLANG_EXE=%%P"
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

rem -- Helper: detect winget (Windows Package Manager) ----------------------
set "HAS_WINGET=0"
winget --version >nul 2>&1 && set "HAS_WINGET=1"

rem -- Helper: detect chocolatey ---------------------------------------------
set "HAS_CHOCO=0"
choco --version >nul 2>&1 && set "HAS_CHOCO=1"

rem -- Check and auto-install CMake ------------------------------------------
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
        set "_CMAKE_INSTALL_DIR="
        for /f "usebackq skip=2 tokens=2,*" %%A in (`reg query "HKLM\SOFTWARE\Kitware\CMake" /v "InstallLocation" 2^>nul`) do set "_CMAKE_INSTALL_DIR=%%B"
        if not defined _CMAKE_INSTALL_DIR for /f "usebackq skip=2 tokens=2,*" %%A in (`reg query "HKCU\SOFTWARE\Kitware\CMake" /v "InstallLocation" 2^>nul`) do set "_CMAKE_INSTALL_DIR=%%B"
        if defined _CMAKE_INSTALL_DIR (
            set "PATH=!_CMAKE_INSTALL_DIR!\bin;!PATH!"
            set "CMAKE_EXE=!_CMAKE_INSTALL_DIR!\bin\cmake.exe"
            echo [OK] cmake installed via winget at !_CMAKE_INSTALL_DIR!
        ) else (
            if exist "%ProgramFiles%\CMake\bin\cmake.exe" (
                set "PATH=%ProgramFiles%\CMake\bin;!PATH!"
                set "CMAKE_EXE=%ProgramFiles%\CMake\bin\cmake.exe"
                echo [OK] cmake installed via winget at %ProgramFiles%\CMake
            ) else if exist "%ProgramFiles(x86)%\CMake\bin\cmake.exe" (
                set "PATH=%ProgramFiles(x86)%\CMake\bin;!PATH!"
                set "CMAKE_EXE=%ProgramFiles(x86)%\CMake\bin\cmake.exe"
                echo [OK] cmake installed via winget at %ProgramFiles(x86)%\CMake
            ) else (
                echo [WARN] cmake installed via winget but install location could not be detected.
            )
        )
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

rem -- Auto-detect LLVM_DIR (no hardcoded paths) ----------------------------
echo Resolving LLVM_DIR...

rem 1. User/machine environment variable (highest precedence)
if "!LLVM_DIR!"=="" (
    for /f "usebackq delims=" %%I in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "[Environment]::GetEnvironmentVariable('LLVM_DIR','User')" 2^>nul`) do (
        if not "%%I"=="" set "LLVM_DIR=%%I"
    )
)
if "!LLVM_DIR!" neq "" if exist "!LLVM_DIR!\LLVMConfig.cmake" goto :llvm_ok

rem 2. Derived from CLANG_EXE if clang is on PATH
if defined CLANG_EXE (
    for %%I in ("!CLANG_EXE!") do set "_clang_bin_dir=%%~dpI"
    for %%I in ("!_clang_bin_dir!..") do set "_clang_root=%%~fI"
    if exist "!_clang_root!\lib\cmake\llvm\LLVMConfig.cmake" (
        set "LLVM_DIR=!_clang_root!\lib\cmake\llvm"
        goto :llvm_ok
    )
)

rem 3. llvm-config on PATH -> ask it directly (works for system llvm, Scoop, Homebrew-on-WSL)
for /f "usebackq delims=" %%I in (`llvm-config-18 --cmakedir 2^>nul`) do (
    if exist "%%I\LLVMConfig.cmake" ( set "LLVM_DIR=%%I" & goto :llvm_ok )
)
for /f "usebackq delims=" %%I in (`llvm-config --cmakedir 2^>nul`) do (
    if exist "%%I\LLVMConfig.cmake" ( set "LLVM_DIR=%%I" & goto :llvm_ok )
)

rem 4. Windows Registry - official LLVM installer writes install root here
for /f "usebackq skip=2 tokens=2,*" %%A in (`reg query "HKLM\SOFTWARE\WOW6432Node\LLVM\LLVM" /v "" 2^>nul`) do (
    if exist "%%B\lib\cmake\llvm\LLVMConfig.cmake" (
        set "LLVM_DIR=%%B\lib\cmake\llvm" & goto :llvm_ok
    )
)
for /f "usebackq skip=2 tokens=2,*" %%A in (`reg query "HKLM\SOFTWARE\LLVM\LLVM" /v "" 2^>nul`) do (
    if exist "%%B\lib\cmake\llvm\LLVMConfig.cmake" (
        set "LLVM_DIR=%%B\lib\cmake\llvm" & goto :llvm_ok
    )
)
for /f "usebackq skip=2 tokens=2,*" %%A in (`reg query "HKCU\SOFTWARE\WOW6432Node\LLVM\LLVM" /v "" 2^>nul`) do (
    if exist "%%B\lib\cmake\llvm\LLVMConfig.cmake" (
        set "LLVM_DIR=%%B\lib\cmake\llvm" & goto :llvm_ok
    )
)
for /f "usebackq skip=2 tokens=2,*" %%A in (`reg query "HKCU\SOFTWARE\LLVM\LLVM" /v "" 2^>nul`) do (
    if exist "%%B\lib\cmake\llvm\LLVMConfig.cmake" (
        set "LLVM_DIR=%%B\lib\cmake\llvm" & goto :llvm_ok
    )
)

rem 5. Visual Studio LLVM component (via vswhere)
if exist "!_VSWHERE!" (
    for /f "usebackq delims=" %%I in (`"!_VSWHERE!" -latest -requires Microsoft.VisualStudio.Component.VC.Llvm.Clang -find VC\Tools\Llvm 2^>nul`) do (
        if exist "%%I\lib\cmake\llvm\LLVMConfig.cmake" (
            set "LLVM_DIR=%%I\lib\cmake\llvm" & goto :llvm_ok
        )
    )
    rem More general search for LLVMConfig.cmake under Visual Studio installations
    for /f "usebackq delims=" %%I in (`"!_VSWHERE!" -latest -find **\LLVMConfig.cmake 2^>nul`) do (
        for %%D in ("%%~dpI.") do (
            if exist "%%~fD\LLVMConfig.cmake" (
                set "LLVM_DIR=%%~fD" & goto :llvm_ok
            )
        )
    )
)

rem 6. Common install locations (winget default, scoop, chocolatey)
for %%D in (
    "%ProgramFiles%\LLVM\lib\cmake\llvm"
    "%ProgramFiles(x86)%\LLVM\lib\cmake\llvm"
    "%USERPROFILE%\scoop\apps\llvm\current\lib\cmake\llvm"
    "%USERPROFILE%\AppData\Local\Programs\LLVM\lib\cmake\llvm"
    "!TOOLS_ROOT!\llvm\lib\cmake\llvm"
) do (
    if exist "%%~D\LLVMConfig.cmake" ( set "LLVM_DIR=%%~D" & goto :llvm_ok )
)

rem 7. Last resort: attempt auto-install
echo [MISSING] LLVM dev libraries ^(LLVM 18 required^) - attempting auto-install...
if "%HAS_WINGET%"=="1" (
    winget install --id LLVM.LLVM --version 18.1.8 --silent ^
        --accept-package-agreements --accept-source-agreements >nul 2>&1
    if not errorlevel 1 (
        set "LLVM_DIR=%ProgramFiles%\LLVM\lib\cmake\llvm"
        if exist "!LLVM_DIR!\LLVMConfig.cmake" goto :llvm_ok
    )
)
if "%HAS_CHOCO%"=="1" (
    choco install llvm --confirm >nul 2>&1
    if not errorlevel 1 (
        set "LLVM_DIR=%ProgramFiles%\LLVM\lib\cmake\llvm"
        if exist "!LLVM_DIR!\LLVMConfig.cmake" goto :llvm_ok
    )
)
if exist "%SCRIPT_DIR%Install-BuildTools.ps1" (
    echo [INFO] Running Install-BuildTools.ps1 to download LLVM...
    set "VGRE_INSTALL_DIR=%INSTALL_DIR%"
    set "VGRE_TOOLS_ROOT=%TOOLS_ROOT%"
    powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%Install-BuildTools.ps1" -LLVM
    set "LLVM_DIR=!TOOLS_ROOT!\llvm\lib\cmake\llvm"
    if exist "!LLVM_DIR!\LLVMConfig.cmake" goto :llvm_ok
)
echo [ERROR] LLVM 18 not found. Install with one of:
echo         winget install LLVM.LLVM --version 18.1.8
echo         choco install llvm
echo         powershell -File scripts\Install-BuildTools.ps1
exit /b 1

:llvm_ok
echo [OK] LLVM at !LLVM_DIR!
rem -- Derive LLVM bin dir for runtime DLL copying (no hardcoded path) ---------
rem    LLVM_DIR is   <root>/lib/cmake/llvm  ->  go up 3 levels to get <root>
for %%D in ("!LLVM_DIR!\..\..\..") do set "_LLVM_ROOT=%%~fD"
set "_LLVM_BIN=!_LLVM_ROOT!\bin"
if not exist "!_LLVM_BIN!\clang.exe" (
    rem fallback: search for bin folder from LLVM_DIR up to 4 levels
    for %%D in ("!LLVM_DIR!\..") do (
        if exist "%%~fD\bin\clang.exe" set "_LLVM_BIN=%%~fD\bin"
    )
    for %%D in ("!LLVM_DIR!\..\..") do (
        if exist "%%~fD\bin\clang.exe" set "_LLVM_BIN=%%~fD\bin"
    )
    for %%D in ("!LLVM_DIR!\..\..\..") do (
        if exist "%%~fD\bin\clang.exe" set "_LLVM_BIN=%%~fD\bin"
    )
)

rem -- Check Visual Studio Build Tools --------------------------------------
if defined VCVARS64 (
    echo [OK] Visual Studio Build Tools at !VCVARS64!
) else (
    call :install_vs_buildtools
)

rem -- Check and auto-install Flutter -----------------------------------------
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
echo flutter: !FLUTTER_CMD!
echo LLVM_DIR: !LLVM_DIR!

echo.
echo === Installing VGRE CLI Tools (vgre-token, vgre-start) ===
rem Install CLI tools NOW - before the build - so they are available even if
rem the native build fails.  Also updates the current session PATH immediately
rem so the user does not need to restart their terminal.
set "TOKEN_SCRIPT_DIR=%INSTALL_DIR%\scripts"
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"
if not exist "%TOKEN_SCRIPT_DIR%" mkdir "%TOKEN_SCRIPT_DIR%"
copy /Y "%SCRIPT_DIR%vgre-token.ps1"        "%TOKEN_SCRIPT_DIR%\vgre-token.ps1"        >nul 2>&1
copy /Y "%SCRIPT_DIR%vgre-token.bat"        "%TOKEN_SCRIPT_DIR%\vgre-token.bat"        >nul 2>&1
copy /Y "%SCRIPT_DIR%vgre-start.bat"        "%TOKEN_SCRIPT_DIR%\vgre-start.bat"        >nul 2>&1
copy /Y "%SCRIPT_DIR%Start-VGRE.ps1"        "%TOKEN_SCRIPT_DIR%\Start-VGRE.ps1"        >nul 2>&1
copy /Y "%SCRIPT_DIR%vgre-discover.bat"     "%TOKEN_SCRIPT_DIR%\vgre-discover.bat"     >nul 2>&1
copy /Y "%SCRIPT_DIR%vgre-discover.ps1"     "%TOKEN_SCRIPT_DIR%\vgre-discover.ps1"     >nul 2>&1
copy /Y "%SCRIPT_DIR%Setup-VGRECluster.ps1" "%TOKEN_SCRIPT_DIR%\Setup-VGRECluster.ps1" >nul 2>&1
copy /Y "%SCRIPT_DIR%vgre_env.ps1"          "%TOKEN_SCRIPT_DIR%\vgre_env.ps1"          >nul 2>&1
copy /Y "%SCRIPT_DIR%Install-VGRETools.ps1" "%TOKEN_SCRIPT_DIR%\Install-VGRETools.ps1" >nul 2>&1
echo [OK] vgre-token + vgre-start + vgre-discover installed to %TOKEN_SCRIPT_DIR%

rem Update User PATH (persistent across new terminals)
for /f "usebackq" %%I in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$d='%TOKEN_SCRIPT_DIR%';$d2='%INSTALL_DIR%';$p=[Environment]::GetEnvironmentVariable('Path','User');$changed=$false;foreach($dir in @($d2,$d)){if($p -notlike '*'+$dir+'*'){$p+=';'+$dir;$changed=$true}};if($changed){[Environment]::SetEnvironmentVariable('Path',$p,'User');'CHANGED'}else{'EXISTS'}"`) do set "_CLI_PATH_STATUS=%%I"
if "!_CLI_PATH_STATUS!"=="CHANGED" echo [OK] Added CLI tools to User PATH.

rem Update current session PATH too - vgre-token works WITHOUT restarting terminal
if "!PATH!" neq "!PATH:%TOKEN_SCRIPT_DIR%=!" goto :cli_path_ok
set "PATH=%TOKEN_SCRIPT_DIR%;%INSTALL_DIR%;!PATH!"
:cli_path_ok
echo [INFO] vgre-token is available in this terminal and all new terminals.
echo        Run: vgre-token generate

echo.
echo === Pulling Latest Source ===
git -C "%PROJECT_ROOT%" pull --rebase --autostash >nul 2>&1
if errorlevel 1 (
    echo [WARN] git pull failed - building from current local source.
) else (
    echo [OK] Source is up to date.
)

echo.
echo === Building VGRE Native Engine ===
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
pushd "%BUILD_DIR%" || exit /b 1

rem -- Locate Ninja: check PATH first, then the LLVM bin dir we installed -------
set "NINJA_FOUND=0"
where ninja >nul 2>&1
if not errorlevel 1 set "NINJA_FOUND=1"
if "!NINJA_FOUND!"=="0" (
    if exist "!_LLVM_BIN!\ninja.exe" (
        set "NINJA_FOUND=1"
        set "PATH=!_LLVM_BIN!;!PATH!"
        echo [INFO] Found ninja in LLVM build tools - added to PATH
    ) else if exist "!TOOLS_ROOT!\llvm\bin\ninja.exe" (
        set "NINJA_FOUND=1"
        set "PATH=!TOOLS_ROOT!\llvm\bin;!PATH!"
        echo [INFO] Found ninja in LLVM build tools - added to PATH
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
    rem Ninja not available - try VS MSBuild or NMake
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
            echo [INFO] Generator changed from "!_cached_gen!" to "!CMAKE_GENERATOR!" - cleaning stale build cache.
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
        echo [INFO] Stale cache has VGRE_USE_LAPACK=ON - clearing to avoid LAPACK search on Windows.
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
        echo [INFO] Previous configure exists but vgre.dll is absent - cleaning stale cache
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
rem   Ninja / NMake -> maps to -j N,  VS MSBuild -> maps to /m:N
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

rem -- Step 1: derive the Flutter SDK root directly from FLUTTER_CMD path.
rem    FLUTTER_CMD is always a full path to flutter.bat (set by :find_flutter).
rem    SDK root = parent of the bin\ directory that contains flutter.bat.
set "_flutter_sdk="
for %%F in ("!FLUTTER_CMD!") do (
    for %%D in ("%%~dpF..") do set "_flutter_sdk=%%~fD"
)
if not defined _flutter_sdk (
    echo [WARN] Cannot derive Flutter SDK root from FLUTTER_CMD=!FLUTTER_CMD!. Dashboard build skipped.
    set "SKIP_DASHBOARD=1"
    goto :dashboard_skip
)
if not exist "!_flutter_sdk!\bin\flutter.bat" (
    echo [WARN] Flutter SDK root invalid: !_flutter_sdk!. Dashboard build skipped.
    set "SKIP_DASHBOARD=1"
    goto :dashboard_skip
)
rem -- Suppress Flutter analytics - prevents interactive consent prompts
set "FLUTTER_SUPPRESS_ANALYTICS=1"
set "DART_SUPPRESS_ANALYTICS=1"
set "PUB_ENVIRONMENT=bot.ci"
set "_flutter_log=%TEMP%\vgre_flutter_build.log"

rem -- Step 1: Pre-cache engine artifacts (non-blocking, 60s timeout).
echo [INFO] Pre-caching Flutter Windows engine artifacts...
pushd "!_flutter_sdk!"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p=Start-Process '!FLUTTER_CMD!' -ArgumentList 'precache','--windows' -RedirectStandardOutput '%TEMP%\vgre_precache.log' -RedirectStandardError '%TEMP%\vgre_precache_err.log' -PassThru; if(-not $p.WaitForExit(60000)){$p.Kill()}"
popd
echo [INFO] flutter precache done (errors above are non-fatal).

rem -- Step 2: resolve Dart packages (120s timeout).
set "_dart_bin=!_flutter_sdk!\bin\cache\dart-sdk\bin\dart.exe"
if not exist "!_dart_bin!" set "_dart_bin=!_flutter_sdk!\bin\dart.exe"
if not exist "!_dart_bin!" set "_dart_bin="

pushd "!DASHBOARD_DIR!"
if errorlevel 1 (
    echo [WARN] Cannot enter dashboard directory. Dashboard build skipped.
    set "SKIP_DASHBOARD=1"
    goto :dashboard_skip
)

if defined _dart_bin (
    echo [INFO] Fetching Dart packages via dart pub get...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$p=Start-Process '!_dart_bin!' -ArgumentList 'pub','get' -RedirectStandardOutput '%TEMP%\vgre_pubget.log' -RedirectStandardError '%TEMP%\vgre_pubget_err.log' -PassThru; if(-not $p.WaitForExit(120000)){$p.Kill();exit 1}; exit $p.ExitCode"
) else (
    echo [INFO] Fetching Dart packages via flutter pub get...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$p=Start-Process '!FLUTTER_CMD!' -ArgumentList 'pub','get' -RedirectStandardOutput '%TEMP%\vgre_pubget.log' -RedirectStandardError '%TEMP%\vgre_pubget_err.log' -PassThru; if(-not $p.WaitForExit(120000)){$p.Kill();exit 1}; exit $p.ExitCode"
)
if errorlevel 1 (
    popd
    echo [WARN] pub get failed. Dashboard build skipped.
    echo        Run manually from !DASHBOARD_DIR!:  flutter pub get
    set "SKIP_DASHBOARD=1"
    goto :dashboard_skip
)

rem -- Step 3: build the Windows bundle (release preferred; debug fallback, 5min timeout each).
set "_flutter_build_ok=0"
echo [INFO] Attempting Flutter Windows release build...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p=Start-Process '!FLUTTER_CMD!' -ArgumentList 'build','windows','--release' -RedirectStandardOutput '!_flutter_log!' -RedirectStandardError '%TEMP%\vgre_flutter_build_err.log' -PassThru; if(-not $p.WaitForExit(300000)){$p.Kill();exit 1}; exit $p.ExitCode"
if not errorlevel 1 set "_flutter_build_ok=1"

if "!_flutter_build_ok!"=="0" (
    echo [INFO] Release build unavailable. Trying debug build...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$p=Start-Process '!FLUTTER_CMD!' -ArgumentList 'build','windows','--debug' -RedirectStandardOutput '!_flutter_log!' -RedirectStandardError '%TEMP%\vgre_flutter_build_err.log' -PassThru; if(-not $p.WaitForExit(300000)){$p.Kill();exit 1}; exit $p.ExitCode"
    if not errorlevel 1 set "_flutter_build_ok=1"
)

if "!_flutter_build_ok!"=="0" (
    echo [WARN] Flutter build failed ^(both release and debug^). Continuing without dashboard.
    echo.
    echo        To diagnose, run manually from !DASHBOARD_DIR!:
    echo          flutter doctor
    echo          flutter build windows --release
    echo.
    echo        Common causes on Windows:
    echo          - Flutter not installed as a git clone ^(run: git clone -b stable https://github.com/flutter/flutter.git^)
    echo          - Visual Studio 2022 desktop workload missing ^(run: flutter doctor^)
    echo          - Engine artifacts not cached ^(run: flutter precache --windows^)
    popd
    set "SKIP_DASHBOARD=1"
    goto :dashboard_skip
)
popd
:dashboard_skip

rem -- Detect actual Flutter build output arch/mode.
rem    Priority: x64 Release -> x64 Debug -> arm64 Release -> arm64 Debug
if "!SKIP_DASHBOARD!"=="1" goto :bundle_dir_ok
if exist "!BUNDLE_DIR!\vgre_dashboard.exe" goto :bundle_dir_ok
if exist "%DASHBOARD_DIR%\build\windows\x64\runner\Debug\vgre_dashboard.exe" (
    set "BUNDLE_DIR=%DASHBOARD_DIR%\build\windows\x64\runner\Debug"
    echo [INFO] Flutter build output detected at x64 Debug path.
    goto :bundle_dir_ok
)
if exist "%DASHBOARD_DIR%\build\windows\arm64\runner\Release\vgre_dashboard.exe" (
    set "BUNDLE_DIR=%DASHBOARD_DIR%\build\windows\arm64\runner\Release"
    echo [INFO] Flutter build output detected at arm64 Release path.
    goto :bundle_dir_ok
)
if exist "%DASHBOARD_DIR%\build\windows\arm64\runner\Debug\vgre_dashboard.exe" (
    set "BUNDLE_DIR=%DASHBOARD_DIR%\build\windows\arm64\runner\Debug"
    echo [INFO] Flutter build output detected at arm64 Debug path.
)
:bundle_dir_ok

rem === Recover flutter_windows.dll if missing from build output ===
rem Flutter incremental builds do not re-emit the engine DLL when the compiled
rem Dart code is unchanged.  If it is absent, copy it from the Flutter SDK
rem artifact cache (downloaded by 'flutter precache --windows').
if not "!SKIP_DASHBOARD!"=="1" if "!_flutter_build_ok!"=="1" if defined _flutter_sdk (
    if not exist "!BUNDLE_DIR!\flutter_windows.dll" (
        echo [INFO] flutter_windows.dll absent from build output - checking Flutter SDK engine cache...
        set "_fwdll_src="
        if exist "!_flutter_sdk!\bin\cache\artifacts\engine\windows-x64-release\flutter_windows.dll" (
            set "_fwdll_src=!_flutter_sdk!\bin\cache\artifacts\engine\windows-x64-release\flutter_windows.dll"
        )
        if not defined _fwdll_src if exist "!_flutter_sdk!\bin\cache\artifacts\engine\windows-x64\flutter_windows.dll" (
            set "_fwdll_src=!_flutter_sdk!\bin\cache\artifacts\engine\windows-x64\flutter_windows.dll"
        )
        if not defined _fwdll_src if exist "!_flutter_sdk!\bin\cache\artifacts\engine\windows-arm64-release\flutter_windows.dll" (
            set "_fwdll_src=!_flutter_sdk!\bin\cache\artifacts\engine\windows-arm64-release\flutter_windows.dll"
        )
        if defined _fwdll_src (
            copy /Y "!_fwdll_src!" "!BUNDLE_DIR!\" >nul 2>&1
            if exist "!BUNDLE_DIR!\flutter_windows.dll" (
                echo [OK] flutter_windows.dll restored from Flutter engine cache.
            ) else (
                echo [WARN] Could not copy flutter_windows.dll from engine cache.
            )
        )
        if not defined _fwdll_src (
            echo [WARN] flutter_windows.dll not found in Flutter SDK engine cache.
            echo        Run:  flutter precache --windows
            echo        Then: .\scripts\vgre_sync.bat
        )
    )
)

echo.
echo === Deploying ===
echo Stopping running VGRE processes...
taskkill /IM vgre_dashboard.exe /F /T >nul 2>&1
taskkill /IM vgre-worker.exe /F /T >nul 2>&1

rem Ensure all target directories exist before any copy operations
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"
if not exist "%INSTALL_DIR%\lib" mkdir "%INSTALL_DIR%\lib"
if not exist "%INSTALL_DIR%\include" mkdir "%INSTALL_DIR%\include"
if not exist "%INSTALL_DIR%\data" mkdir "%INSTALL_DIR%\data"
if not exist "%INSTALL_DIR%\data\flutter_assets" mkdir "%INSTALL_DIR%\data\flutter_assets"

rem Add INSTALL_DIR to Avast exclusions so deployed executables are not quarantined.
rem This uses the HKCU registry path which does not require elevation.
reg add "HKCU\SOFTWARE\AVAST Software\Avast\exclusions" /v "vgre_install" /t REG_SZ /d "%INSTALL_DIR%" /f >nul 2>&1
reg add "HKCU\SOFTWARE\AVAST Software\Avast\exclusions" /v "vgre_install_lib" /t REG_SZ /d "%INSTALL_DIR%\lib" /f >nul 2>&1

if not "!SKIP_DASHBOARD!"=="1" (
    if not exist "!BUNDLE_DIR!\vgre_dashboard.exe" (
        echo ERROR: Dashboard bundle not found at:
        echo   !BUNDLE_DIR!
        echo         The Flutter build may have failed. Run:
        echo           flutter build windows --release
        echo         from the vgre_dashboard directory for full error output.
        exit /b 1
    )
    rem Delete the old exe first to release any stale handle
    if exist "%INSTALL_DIR%\vgre_dashboard.exe" del /f /q "%INSTALL_DIR%\vgre_dashboard.exe" >nul 2>&1
    rem Copy root-level files (exe + DLLs) individually - copy /Y never blocks
    for %%F in ("!BUNDLE_DIR!\*.*") do copy /Y "%%F" "%INSTALL_DIR%\" >nul 2>&1
    rem Explicitly copy flutter_windows.dll by name so it is guaranteed to be present
    rem even when the wildcard form above is incomplete on some cmd.exe versions.
    if exist "!BUNDLE_DIR!\flutter_windows.dll" (
        copy /Y "!BUNDLE_DIR!\flutter_windows.dll" "%INSTALL_DIR%\" >nul 2>&1
    )
    rem Copy data directory files
    if exist "!BUNDLE_DIR!\data\*.*" (
        for %%F in ("!BUNDLE_DIR!\data\*.*") do copy /Y "%%F" "%INSTALL_DIR%\data\" >nul 2>&1
    )
    rem Copy flutter_assets recursively using a subroutine (no xcopy/robocopy)
    if exist "!BUNDLE_DIR!\data\flutter_assets" (
        call :copy_dir "!BUNDLE_DIR!\data\flutter_assets" "%INSTALL_DIR%\data\flutter_assets"
    )
    if not exist "%INSTALL_DIR%\vgre_dashboard.exe" (
        echo ERROR: vgre_dashboard.exe missing from %INSTALL_DIR% after copy.
        exit /b 1
    )
    if not exist "%INSTALL_DIR%\flutter_windows.dll" (
        echo ERROR: flutter_windows.dll missing from %INSTALL_DIR% after copy.
        echo        Source: !BUNDLE_DIR!\flutter_windows.dll
        echo        Dashboard will fail at startup with "flutter_windows.dll was not found".
        exit /b 1
    )
    echo [OK] Dashboard deployed to %INSTALL_DIR%
) else (
    echo [SKIP] Dashboard bundle deployment skipped ^(Flutter was unavailable^).
)

copy /Y "!BUILD_OUT_DIR!\vgre.dll" "%INSTALL_DIR%\" >nul 2>&1
copy /Y "!BUILD_OUT_DIR!\vgre.dll" "%INSTALL_DIR%\lib\" >nul 2>&1
if not exist "%INSTALL_DIR%\vgre.dll" (
    echo ERROR: Failed to copy vgre.dll from !BUILD_OUT_DIR!\vgre.dll
    exit /b 1
)

copy /Y "!BUILD_OUT_DIR!\vgre_cudart.dll" "%INSTALL_DIR%\" >nul 2>&1
copy /Y "!BUILD_OUT_DIR!\vgre_cudart.dll" "%INSTALL_DIR%\lib\" >nul 2>&1
if not exist "%INSTALL_DIR%\vgre_cudart.dll" (
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
echo === Copying LLVM/OpenMP Runtime Dependencies ===
rem _LLVM_BIN is derived from LLVM_DIR (set in the auto-detection section above).
rem No hardcoded TOOLS_ROOT path - follows wherever LLVM was actually installed.
rem
rem IMPORTANT: VC runtime DLLs (msvcp140*, vcruntime140*, concrt140, ucrtbase)
rem must NEVER be copied here. Bundling old versions causes STATUS_DLL_INIT_FAILED
rem (0xC0000142) when the binary was compiled with a newer toolchain. Windows
rem always resolves these from System32 automatically.
if exist "!_LLVM_BIN!\*.dll" (
    echo Copying LLVM runtime DLLs from !_LLVM_BIN!...
    for %%F in ("!_LLVM_BIN!\*.dll") do (
        rem Skip VC runtime DLLs - must come from System32, not bundled
        set "_skip_dll=0"
        for %%R in (msvcp140 msvcp140_1 msvcp140_2 msvcp140_atomic_wait msvcp140_codecvt_ids vcruntime140 vcruntime140_1 concrt140 ucrtbase) do (
            if /I "%%~nF"=="%%R" set "_skip_dll=1"
        )
        if "!_skip_dll!"=="0" (
            copy /Y "%%F" "%INSTALL_DIR%\lib\" >nul 2>&1
            copy /Y "%%F" "%INSTALL_DIR%\" >nul 2>&1
        )
    )
    rem Also purge any stale VC runtime DLLs left from previous syncs
    for %%R in (msvcp140.dll msvcp140_1.dll msvcp140_2.dll msvcp140_atomic_wait.dll msvcp140_codecvt_ids.dll vcruntime140.dll vcruntime140_1.dll concrt140.dll ucrtbase.dll) do (
        if exist "%INSTALL_DIR%\lib\%%R" del /f /q "%INSTALL_DIR%\lib\%%R" >nul 2>&1
        if exist "%INSTALL_DIR%\%%R"     del /f /q "%INSTALL_DIR%\%%R"     >nul 2>&1
    )
    if exist "%INSTALL_DIR%\libomp.dll" (
        echo [OK] libomp.dll deployed
    ) else (
        echo [WARN] libomp.dll not found in !_LLVM_BIN! - vgre-worker may crash.
        echo        Ensure LLVM was built with OpenMP support.
    )
) else (
    echo [WARN] No LLVM DLLs found in !_LLVM_BIN!
    echo        libomp.dll is required; vgre-worker.exe may crash on startup.
    echo        Run:  powershell -File scripts\Install-BuildTools.ps1
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
rem Embed the actual detected LLVM bin dir so the wrapper works on any machine.
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"
echo @echo off > "%INSTALL_DIR%\vgre-worker.cmd"
echo setlocal EnableExtensions EnableDelayedExpansion >> "%INSTALL_DIR%\vgre-worker.cmd"
echo set "PATH=%%~dp0lib;%%~dp0;!_LLVM_BIN!;%%PATH%%" >> "%INSTALL_DIR%\vgre-worker.cmd"
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
    echo.
    echo rem -- Ensure VGRE libs are found first; LLVM bin recorded at build time --
    echo set "PATH=%%APP_DIR%%lib;%%APP_DIR%%;!_LLVM_BIN!;%%PATH%%"
    echo.
    echo rem -- Load cluster auth token if not already set --
    echo if not defined VGRE_TCP_AUTH_TOKEN_FILE ^(
    echo     if exist "%%USERPROFILE%%\.vgre\token" set "VGRE_TCP_AUTH_TOKEN_FILE=%%USERPROFILE%%\.vgre\token"
    echo ^)
    echo.
    echo rem -- Launch dashboard if built, otherwise start master worker headlessly --
    echo if exist "%%APP_DIR%%vgre_dashboard.exe" ^(
    echo     cd /d "%%APP_DIR%%"
    echo     start "" "%%APP_DIR%%vgre_dashboard.exe"
    echo ^) else ^(
    echo     echo [INFO] Dashboard not built - starting VGRE master worker headlessly.
    echo     echo        Workers can connect via LAN auto-discovery or --master-address.
    echo     echo        To build the dashboard: git clone -b stable https://github.com/flutter/flutter.git C:\flutter
    echo     echo        Then re-run: .\scripts\vgre_sync.bat from the VGRE repository root.
    echo     echo.
    echo     powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%%APP_DIR%%scripts\Start-VGRE.ps1" --master
    echo ^)
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
rem Add INSTALL_DIR and detected LLVM bin to PATH so vgre.dll and libomp.dll
rem are found by Windows loader during the self-check (no hardcoded TOOLS_ROOT).
set "PATH=%INSTALL_DIR%\lib;%INSTALL_DIR%;!_LLVM_BIN!;%PATH%"
"%INSTALL_DIR%\vgre-worker.exe" --help >nul 2>&1
rem  IMPORTANT: use %ERRORLEVEL% neq 0, NOT "if errorlevel 1".
rem  STATUS_DLL_INIT_FAILED = -1073741502 and STATUS_ACCESS_VIOLATION = -1073741819
rem  are both NEGATIVE - "if errorlevel 1" only triggers for exit codes >= 1 and
rem  silently misreports a crash as "passed".
if %ERRORLEVEL% neq 0 (
    echo ERROR: vgre-worker failed startup self-check ^(exit code %ERRORLEVEL%^).
    echo        Possible causes:
    echo          - STATUS_DLL_INIT_FAILED ^(0xC0000142^): DLL global constructor fault
    echo          - STATUS_ACCESS_VIOLATION ^(0xC0000005^): invalid memory access
    echo          - Missing dependency: check %INSTALL_DIR%\lib for libomp.dll
    echo        See docs/TROUBLESHOOTING_WINDOWS.md for full diagnostics.
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

rem -- Refresh CLI scripts (already installed before the build; this updates them) --
copy /Y "%SCRIPT_DIR%vgre-token.ps1"        "%TOKEN_SCRIPT_DIR%\vgre-token.ps1"        >nul 2>&1
copy /Y "%SCRIPT_DIR%vgre-token.bat"        "%TOKEN_SCRIPT_DIR%\vgre-token.bat"        >nul 2>&1
copy /Y "%SCRIPT_DIR%vgre-start.bat"        "%TOKEN_SCRIPT_DIR%\vgre-start.bat"        >nul 2>&1
copy /Y "%SCRIPT_DIR%Start-VGRE.ps1"        "%TOKEN_SCRIPT_DIR%\Start-VGRE.ps1"        >nul 2>&1
copy /Y "%SCRIPT_DIR%vgre-discover.bat"     "%TOKEN_SCRIPT_DIR%\vgre-discover.bat"     >nul 2>&1
copy /Y "%SCRIPT_DIR%vgre-discover.ps1"     "%TOKEN_SCRIPT_DIR%\vgre-discover.ps1"     >nul 2>&1
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

:wait_for_file_release
rem Subroutine: no-op kept for compatibility. taskkill /F is synchronous.
exit /b 0

:copy_dir
rem Subroutine: recursively copy a directory using only copy /Y (never blocks).
rem Usage: call :copy_dir "source_dir" "dest_dir"
rem Creates dest_dir if it does not exist.
if not exist %2 mkdir %2 >nul 2>&1
for %%F in (%1\*.*) do copy /Y "%%F" %2\ >nul 2>&1
for /d %%D in (%1\*) do (
    call :copy_dir "%%D" "%~2\%%~nxD"
)
exit /b 0

:find_flutter
rem -- Priority 1: FLUTTER_CMD already set by caller (env var override)
if defined FLUTTER_CMD (
    if exist "!FLUTTER_CMD!" exit /b 0
    set "FLUTTER_CMD="
)

rem -- Priority 2: resolve from PATH - prefer .bat so cmd /c works reliably
set "_flutter_on_path="
for /f "usebackq tokens=*" %%F in (`where flutter.bat 2^>nul`) do (
    if not defined _flutter_on_path set "_flutter_on_path=%%F"
)
if not defined _flutter_on_path (
    for /f "usebackq tokens=*" %%F in (`where flutter 2^>nul`) do (
        if not defined _flutter_on_path set "_flutter_on_path=%%F"
    )
)
if defined _flutter_on_path (
    rem Prefer the one that is a git clone (has a .git folder two levels up)
    for /f "usebackq tokens=*" %%F in (`where flutter.bat 2^>nul`) do (
        set "_candidate_bat=%%F"
        rem %%~dpF = directory of the .bat, parent of that = SDK root
        for %%D in ("%%~dpF..") do (
            if exist "%%~fD\.git" (
                if not defined FLUTTER_CMD set "FLUTTER_CMD=%%F"
            )
        )
    )
    rem If no git-clone found, just take the first .bat on PATH
    if not defined FLUTTER_CMD (
        for /f "usebackq tokens=*" %%F in (`where flutter.bat 2^>nul`) do (
            if not defined FLUTTER_CMD set "FLUTTER_CMD=%%F"
        )
    )
    if not defined FLUTTER_CMD set "FLUTTER_CMD=!_flutter_on_path!"
    set "PATH=!FLUTTER_CMD:~0,-11!;%PATH%"
    exit /b 0
)

rem -- Priority 3: check registry Path variables (covers recently added entries)
set "_reg_paths="
for /f "usebackq tokens=2*" %%A in (`reg query "HKCU\Environment" /v Path 2^>nul`) do set "_reg_paths=%%B"
for /f "usebackq tokens=2*" %%A in (`reg query "HKLM\System\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul`) do set "_reg_paths=!_reg_paths!;%%B"

if defined _reg_paths (
    for %%P in ("!_reg_paths:;=" "!") do (
        if not defined FLUTTER_CMD (
            set "_p=%%~P"
            if exist "!_p!\flutter.bat" (
                set "FLUTTER_CMD=!_p!\flutter.bat"
                set "PATH=!_p!;%PATH%"
            ) else if exist "!_p!\bin\flutter.bat" (
                set "FLUTTER_CMD=!_p!\bin\flutter.bat"
                set "PATH=!_p!\bin;%PATH%"
            )
        )
    )
)
if defined FLUTTER_CMD exit /b 0

rem -- Priority 4: well-known install locations (no hardcoded user names)
for %%D in (
    "%USERPROFILE%\flutter\bin"
    "%USERPROFILE%\Documents\flutter\bin"
    "%USERPROFILE%\development\flutter\bin"
    "%LOCALAPPDATA%\Programs\flutter\bin"
    "!TOOLS_ROOT!\flutter\bin"
    "%LOCALAPPDATA%\Microsoft\WinGet\Packages\Google.FlutterSDK\bin"
    "C:\src\flutter\bin"
    "C:\tools\flutter\bin"
    "C:\flutter\bin"
) do (
    if not defined FLUTTER_CMD (
        if exist "%%~D\flutter.bat" (
            set "FLUTTER_CMD=%%~D\flutter.bat"
            set "PATH=%%~D;%PATH%"
        )
    )
)
if defined FLUTTER_CMD exit /b 0

rem -- Priority 5: slow PowerShell search under USERPROFILE and common roots
for /f "usebackq delims=" %%I in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$roots=@($env:USERPROFILE,'C:\src','C:\tools','C:\flutter'); $hit=Get-ChildItem -Path $roots -Filter flutter.bat -Recurse -Depth 4 -ErrorAction SilentlyContinue | Where-Object { Test-Path (Join-Path (Split-Path (Split-Path $_.FullName)) '.git') } | Select-Object -First 1 -ExpandProperty DirectoryName; if(-not $hit){$hit=Get-ChildItem -Path $roots -Filter flutter.bat -Recurse -Depth 4 -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty DirectoryName}; if($hit){$hit}"`) do (
    if not defined FLUTTER_CMD (
        set "FLUTTER_CMD=%%I\flutter.bat"
        set "PATH=%%I;%PATH%"
    )
)
exit /b 0
