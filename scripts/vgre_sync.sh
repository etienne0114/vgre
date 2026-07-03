#!/bin/bash
set -e

# VGRE Sync & Install Script (Linux/macOS)
# Verifies all required dependencies, installs missing ones automatically,
# builds the native engine + Flutter dashboard, and installs to the user profile.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INSTALL_DIR="$HOME/.local/share/VGRE"
BIN_DIR="$HOME/.local/bin"
VGRE_ENABLE_NATIVE_SIMD_FLAG="${VGRE_ENABLE_NATIVE_SIMD:-0}"
OS="$(uname -s)"
ARCH="$(uname -m)"

# ── Dependency verification + auto-install ────────────────────────────────────
_pkg_install() {
    local PKGS=("$@")
    if command -v apt-get >/dev/null 2>&1; then
        echo "  [apt] Installing: ${PKGS[*]}"
        sudo apt-get update -qq && sudo apt-get install -y "${PKGS[@]}"
    elif command -v dnf >/dev/null 2>&1; then
        echo "  [dnf] Installing: ${PKGS[*]}"
        sudo dnf install -y "${PKGS[@]}"
    elif command -v pacman >/dev/null 2>&1; then
        echo "  [pacman] Installing: ${PKGS[*]}"
        sudo pacman -Sy --noconfirm "${PKGS[@]}"
    elif command -v zypper >/dev/null 2>&1; then
        echo "  [zypper] Installing: ${PKGS[*]}"
        sudo zypper install -y "${PKGS[@]}"
    elif command -v brew >/dev/null 2>&1; then
        echo "  [brew] Installing: ${PKGS[*]}"
        brew install "${PKGS[@]}"
    else
        echo "ERROR: No package manager found. Install manually: ${PKGS[*]}" >&2
        exit 1
    fi
}

_check_and_install_deps() {
    echo ""
    echo "=== Verifying Dependencies ==="
    local MISSING_APT=()
    local MISSING_DNF=()
    local MISSING_BREW=()
    local NEED_INSTALL=0

    # cmake
    if ! command -v cmake >/dev/null 2>&1; then
        echo "  [MISSING] cmake"
        MISSING_APT+=(cmake); MISSING_DNF+=(cmake); MISSING_BREW+=(cmake)
        NEED_INSTALL=1
    else
        echo "  [OK] cmake $(cmake --version | head -1 | awk '{print $3}')"
    fi

    # C++ compiler
    if ! command -v g++ >/dev/null 2>&1 && ! command -v clang++ >/dev/null 2>&1; then
        echo "  [MISSING] C++ compiler (g++ or clang++)"
        MISSING_APT+=(build-essential g++); MISSING_DNF+=("gcc-c++"); MISSING_BREW+=(gcc)
        NEED_INSTALL=1
    else
        echo "  [OK] C++ compiler present"
    fi

    # make
    if ! command -v make >/dev/null 2>&1 && ! command -v ninja >/dev/null 2>&1; then
        echo "  [MISSING] make / ninja"
        MISSING_APT+=(make); MISSING_DNF+=(make); MISSING_BREW+=(make)
        NEED_INSTALL=1
    else
        echo "  [OK] Build driver present"
    fi

    # LLVM dev libraries — VGRE requires exactly LLVM 18
    local LLVM_FOUND=0
    for cfg in llvm-config-18 llvm-config; do
        if command -v "$cfg" >/dev/null 2>&1; then
            local VER
            VER=$("$cfg" --version 2>/dev/null | grep -oE '^[0-9]+' || echo "0")
            if [[ "$VER" == "18" ]]; then
                LLVM_FOUND=1
                echo "  [OK] LLVM $("$cfg" --version) ($cfg)"
                break
            else
                echo "  [WARN] Found LLVM $("$cfg" --version) ($cfg) — VGRE requires LLVM 18; ignoring"
            fi
        fi
    done
    # Homebrew llvm@18 is keg-only: llvm-config is not on PATH by default.
    if [[ $LLVM_FOUND -eq 0 && "$OS" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
        local _brew_cfg
        _brew_cfg="$(brew --prefix llvm@18 2>/dev/null)/bin/llvm-config"
        if [[ -x "$_brew_cfg" ]]; then
            local VER
            VER=$("$_brew_cfg" --version 2>/dev/null | grep -oE '^[0-9]+' || echo "0")
            if [[ "$VER" == "18" ]]; then
                LLVM_FOUND=1
                echo "  [OK] LLVM $("$_brew_cfg" --version) ($_brew_cfg)"
            fi
        fi
    fi
    if [[ $LLVM_FOUND -eq 0 ]]; then
        echo "  [MISSING] LLVM 18 dev libraries (llvm-18, clang-18, libclang-18-dev)"
        MISSING_APT+=(llvm-18-dev clang-18 libclang-18-dev); MISSING_DNF+=(llvm18-devel clang18-devel)
        MISSING_BREW+=(llvm@18)
        NEED_INSTALL=1
    fi

    # OpenMP
    local OMP_FOUND=0
    if [[ "$OS" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
        _OMP_PREFIX="$(brew --prefix libomp 2>/dev/null || true)"
        if [[ -n "$_OMP_PREFIX" && -f "${_OMP_PREFIX}/lib/libomp.dylib" ]]; then
            OMP_FOUND=1
            echo "  [OK] OpenMP present (${_OMP_PREFIX}/lib/libomp.dylib)"
        fi
    fi
    for lib in /usr/lib/x86_64-linux-gnu/libomp.so \
               /usr/lib/aarch64-linux-gnu/libomp.so \
               /usr/lib/libomp.so \
               /usr/lib64/libomp.so; do
        [[ -f "$lib" ]] && OMP_FOUND=1 && break
    done
    command -v dpkg >/dev/null 2>&1 && dpkg -l libomp-dev >/dev/null 2>&1 && OMP_FOUND=1
    if [[ $OMP_FOUND -eq 0 ]]; then
        echo "  [MISSING] OpenMP (libomp-dev)"
        MISSING_APT+=(libomp-dev); MISSING_DNF+=(libomp-devel); MISSING_BREW+=(libomp)
        NEED_INSTALL=1
    elif [[ "$OS" != "Darwin" ]]; then
        echo "  [OK] OpenMP present"
    fi

    # openssl
    if ! command -v openssl >/dev/null 2>&1; then
        echo "  [MISSING] openssl"
        MISSING_APT+=(libssl-dev openssl); MISSING_DNF+=(openssl-devel openssl)
        MISSING_BREW+=(openssl)
        NEED_INSTALL=1
    else
        echo "  [OK] openssl $(openssl version | awk '{print $2}')"
    fi

    # curl
    if ! command -v curl >/dev/null 2>&1; then
        echo "  [MISSING] curl"
        MISSING_APT+=(curl); MISSING_DNF+=(curl); MISSING_BREW+=(curl)
        NEED_INSTALL=1
    else
        echo "  [OK] curl present"
    fi

    # jemalloc — required at runtime to avoid glibc malloc arena assertions
    # when Flutter (libc++) and VGRE (libstdc++) threads share the same heap.
    local JEMALLOC_FOUND=0
    if [[ "$OS" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
        _JEM_PREFIX="$(brew --prefix jemalloc 2>/dev/null || true)"
        if [[ -n "$_JEM_PREFIX" && -f "${_JEM_PREFIX}/lib/libjemalloc.2.dylib" ]]; then
            JEMALLOC_FOUND=1
            echo "  [OK] jemalloc (${_JEM_PREFIX}/lib/libjemalloc.2.dylib)"
        fi
    fi
    for lib in /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
               /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 \
               /usr/lib64/libjemalloc.so.2 \
               /usr/lib/libjemalloc.so.2 \
               /usr/local/lib/libjemalloc.so.2; do
        [[ -f "$lib" ]] && JEMALLOC_FOUND=1 && echo "  [OK] jemalloc ($lib)" && break
    done
    if [[ $JEMALLOC_FOUND -eq 0 ]]; then
        echo "  [MISSING] jemalloc (required for dashboard runtime stability)"
        MISSING_APT+=(libjemalloc2); MISSING_DNF+=(jemalloc); MISSING_BREW+=(jemalloc)
        NEED_INSTALL=1
    fi

    # Auto-install if anything missing
    if [[ $NEED_INSTALL -eq 1 ]]; then
        echo ""
        echo "  Installing missing packages..."
        if [[ "$OS" == "Linux" ]]; then
            if command -v apt-get >/dev/null 2>&1; then
                _pkg_install "${MISSING_APT[@]}"
            elif command -v dnf >/dev/null 2>&1; then
                _pkg_install "${MISSING_DNF[@]}"
            elif command -v pacman >/dev/null 2>&1; then
                _pkg_install base-devel cmake llvm clang openssl libomp git curl
            fi
        elif [[ "$OS" == "Darwin" ]]; then
            if ! command -v brew >/dev/null 2>&1; then
                echo "  Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            _pkg_install "${MISSING_BREW[@]}"
        fi
    fi

    # Flutter — special install path
    if ! command -v flutter >/dev/null 2>&1; then
        echo "  [MISSING] Flutter"
        if [[ "$OS" == "Linux" ]] && command -v snap >/dev/null 2>&1; then
            echo "  Installing Flutter via snap..."
            sudo snap install flutter --classic
        elif [[ "$OS" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
            echo "  Installing Flutter via Homebrew..."
            brew install --cask flutter
        else
            local FLUTTER_VERSION="3.24.5"
            local FLUTTER_ARCH="x64"
            [[ "$ARCH" == "arm64" || "$ARCH" == "aarch64" ]] && FLUTTER_ARCH="arm64"
            mkdir -p "$HOME/.local"
            if [[ "$OS" == "Darwin" ]]; then
                local FLUTTER_URL="https://storage.googleapis.com/flutter_infra_release/releases/stable/macos/flutter_macos_${FLUTTER_ARCH}_${FLUTTER_VERSION}-stable.zip"
                echo "  Downloading Flutter ${FLUTTER_VERSION} (macOS ${FLUTTER_ARCH})..."
                curl -fsSL "$FLUTTER_URL" -o "/tmp/flutter_macos_$$.zip"
                unzip -q "/tmp/flutter_macos_$$.zip" -d "$HOME/.local"
                rm -f "/tmp/flutter_macos_$$.zip"
            else
                local FLUTTER_URL="https://storage.googleapis.com/flutter_infra_release/releases/stable/linux/flutter_linux_${FLUTTER_VERSION}-stable.tar.xz"
                echo "  Downloading Flutter ${FLUTTER_VERSION} (Linux ${FLUTTER_ARCH})..."
                curl -fsSL "$FLUTTER_URL" | tar -xJ -C "$HOME/.local"
            fi
            ln -sf "$HOME/.local/flutter/bin/flutter" "$BIN_DIR/flutter" 2>/dev/null || true
            export PATH="$HOME/.local/flutter/bin:$PATH"
        fi
    fi
    if command -v flutter >/dev/null 2>&1; then
        echo "  [OK] Flutter $(flutter --version 2>/dev/null | head -1 | awk '{print $2}')"
    else
        echo "  [WARN] Flutter unavailable — dashboard build will be skipped"
    fi

    # macOS Flutter desktop requires full Xcode (Command Line Tools alone are insufficient).
    if [[ "$OS" == "Darwin" ]]; then
        if xcrun xcodebuild -version >/dev/null 2>&1; then
            echo "  [OK] Xcode $(xcrun xcodebuild -version 2>/dev/null | head -1)"
        else
            echo "  [WARN] Xcode not installed — Flutter macOS dashboard cannot be built"
            echo "         Install from https://developer.apple.com/xcode/ then run:"
            echo "           sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer"
            echo "           sudo xcodebuild -runFirstLaunch"
        fi
    fi

    # Final hard check
    local HARD_FAIL=0
    for CMD in cmake git; do
        command -v "$CMD" >/dev/null 2>&1 || { echo "  [FAIL] $CMD still missing"; HARD_FAIL=1; }
    done
    command -v g++ >/dev/null 2>&1 || command -v clang++ >/dev/null 2>&1 || \
        { echo "  [FAIL] C++ compiler still missing"; HARD_FAIL=1; }
    if [[ $HARD_FAIL -eq 1 ]]; then
        echo ""
        echo "ERROR: Critical dependencies missing. Install them manually and re-run." >&2
        exit 1
    fi
    echo "=== All dependencies satisfied ==="
    echo ""
}

_check_and_install_deps

# ── Auth Token Auto-load ──────────────────────────────────────────────────────
# Priority order:
#   1. VGRE_TCP_AUTH_TOKEN_FILE env var (explicit file path)
#   2. ~/.vgre/token              (written by setup-cluster.sh — zero config after setup)
#   3. VGRE_TCP_AUTH_TOKEN env var (raw token — visible in process list, avoid in production)
#   4. None — dashboard falls back to hardware secure storage (TPM/Keyring)
#
# For full cluster env var reference, see docs/USER_GUIDE.md section 4.5.
DEFAULT_TOKEN_FILE="$HOME/.vgre/token"
if [[ -z "$VGRE_TCP_AUTH_TOKEN_FILE" && -f "$DEFAULT_TOKEN_FILE" ]]; then
    export VGRE_TCP_AUTH_TOKEN_FILE="$DEFAULT_TOKEN_FILE"
    echo "🔑 Auto-loaded token from $DEFAULT_TOKEN_FILE"
fi

if [[ -n "$VGRE_TCP_AUTH_TOKEN_FILE" ]]; then
    TOKEN_FILE="$VGRE_TCP_AUTH_TOKEN_FILE"
    echo "🔐 Secure cluster mode active (token file: $TOKEN_FILE)"
elif [[ -n "$VGRE_TCP_AUTH_TOKEN" ]]; then
    echo "🔐 VGRE_TCP_AUTH_TOKEN set; writing to ~/.vgre/token for the launch wrapper..."
    mkdir -p "$HOME/.vgre"
    TOKEN_FILE="$HOME/.vgre/token"
    printf '%s' "$VGRE_TCP_AUTH_TOKEN" > "$TOKEN_FILE"
    chmod 600 "$TOKEN_FILE"
    export VGRE_TCP_AUTH_TOKEN_FILE="$TOKEN_FILE"
else
    echo "🔑 No auth token found — generating one automatically..."
    mkdir -p "$HOME/.vgre"
    TOKEN_FILE="$HOME/.vgre/token"
    if command -v openssl >/dev/null 2>&1; then
        _TOKEN=$(openssl rand -hex 32)
    else
        _TOKEN=$(od -An -tx1 /dev/urandom | tr -d ' \n' | head -c 64)
    fi
    printf '%s' "$_TOKEN" > "$TOKEN_FILE"
    chmod 600 "$TOKEN_FILE"
    export VGRE_TCP_AUTH_TOKEN_FILE="$TOKEN_FILE"
    echo "🔐 Auth token saved to $TOKEN_FILE"
fi

FLUTTER_CACHE_PATH="${FLUTTER_CACHE_PATH:-$HOME/.cache/flutter}"
mkdir -p "$FLUTTER_CACHE_PATH"
export FLUTTER_CACHE_PATH

if pgrep -f vgre_dashboard >/dev/null 2>&1; then
    echo "🛑 Stopping running VGRE Dashboard (pkg conflict)..."
    pkill -f vgre_dashboard || true
    sleep 1
fi

cd "$PROJECT_ROOT"

echo "🚀 Starting VGRE Global Sync..."
if [[ "$VGRE_ENABLE_NATIVE_SIMD_FLAG" == "1" || "$VGRE_ENABLE_NATIVE_SIMD_FLAG" == "ON" || "$VGRE_ENABLE_NATIVE_SIMD_FLAG" == "on" ]]; then
    VGRE_ENABLE_NATIVE_SIMD_FLAG=ON
else
    VGRE_ENABLE_NATIVE_SIMD_FLAG=OFF
fi
echo "SIMD tuning: $VGRE_ENABLE_NATIVE_SIMD_FLAG (set VGRE_ENABLE_NATIVE_SIMD=1 to opt-in)"

# 1. Build Native Engine
echo "📦 Building VGRE Native Engine..."
mkdir -p build && cd build

# Hint GCC on Linux to avoid OpenMP detection issues with Clang 18 without libomp-dev
if [[ "$(uname)" == "Linux" ]] && [[ -z "$CC" ]] && [[ -z "$CXX" ]]; then
    export CC=gcc
    export CXX=g++
    rm -f CMakeCache.txt
fi

# Auto-detect LLVM cmake dir (no hardcoded paths).
# Priority: LLVM_DIR env var → llvm-config-18 → llvm-config → cmake will search itself.
_LLVM_CMAKE_ARG=""
if [[ -n "${LLVM_DIR:-}" ]] && [[ -f "$LLVM_DIR/LLVMConfig.cmake" ]]; then
    _LLVM_CMAKE_ARG="-DLLVM_DIR=$LLVM_DIR"
    echo "  [LLVM] Using LLVM_DIR from environment: $LLVM_DIR"
else
    for _cfg in llvm-config-18 llvm-config; do
        if command -v "$_cfg" >/dev/null 2>&1; then
            _ver=$("$_cfg" --version 2>/dev/null | grep -oE '^[0-9]+' || echo "0")
            if [[ "$_ver" == "18" ]]; then
                _cmake_dir=$("$_cfg" --cmakedir 2>/dev/null || true)
                if [[ -n "$_cmake_dir" ]] && [[ -f "$_cmake_dir/LLVMConfig.cmake" ]]; then
                    _LLVM_CMAKE_ARG="-DLLVM_DIR=$_cmake_dir"
                    echo "  [LLVM] Detected via $_cfg: $_cmake_dir"
                    break
                fi
            fi
        fi
    done
    if [[ -z "$_LLVM_CMAKE_ARG" && "$OS" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
        _brew_cfg="$(brew --prefix llvm@18 2>/dev/null)/bin/llvm-config"
        if [[ -x "$_brew_cfg" ]]; then
            _ver=$("$_brew_cfg" --version 2>/dev/null | grep -oE '^[0-9]+' || echo "0")
            if [[ "$_ver" == "18" ]]; then
                _cmake_dir=$("$_brew_cfg" --cmakedir 2>/dev/null || true)
                if [[ -n "$_cmake_dir" ]] && [[ -f "$_cmake_dir/LLVMConfig.cmake" ]]; then
                    _LLVM_CMAKE_ARG="-DLLVM_DIR=$_cmake_dir"
                    echo "  [LLVM] Detected via $_brew_cfg: $_cmake_dir"
                fi
            fi
        fi
    fi
fi

# macOS: mirror CI configure flags (Homebrew llvm@18 + libomp; no hardcoded prefixes).
_MACOS_CMAKE_EXTRA=()
if [[ "$OS" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
    _LLVM18_PREFIX="$(brew --prefix llvm@18 2>/dev/null || true)"
    _LIBOMP_PREFIX="$(brew --prefix libomp 2>/dev/null || true)"
    if [[ -n "$_LLVM18_PREFIX" && -x "$_LLVM18_PREFIX/bin/clang++" ]]; then
        echo "  [macOS] Using Homebrew llvm@18 at $_LLVM18_PREFIX"
        _MACOS_CMAKE_EXTRA+=(
            -DCMAKE_C_COMPILER="${_LLVM18_PREFIX}/bin/clang"
            -DCMAKE_CXX_COMPILER="${_LLVM18_PREFIX}/bin/clang++"
            -DLLVM_DIR="${_LLVM18_PREFIX}/lib/cmake/llvm"
        )
        [[ -n "$_LIBOMP_PREFIX" ]] && _MACOS_CMAKE_EXTRA+=(
            -DOpenMP_ROOT="${_LIBOMP_PREFIX}"
            -DCMAKE_PREFIX_PATH="${_LLVM18_PREFIX};${_LIBOMP_PREFIX}"
        )
        _SDK="$(xcrun --show-sdk-path 2>/dev/null || true)"
        [[ -n "$_SDK" ]] && _MACOS_CMAKE_EXTRA+=(-DCMAKE_OSX_SYSROOT="$_SDK")
    fi
fi

# Prefer Ninja for faster incremental builds; fall back to make.
_BUILD_CMD="make"
_GEN_ARG=""
if command -v ninja >/dev/null 2>&1; then
    _BUILD_CMD="ninja"
    _GEN_ARG="-G Ninja"
fi

cmake .. $_GEN_ARG -DCMAKE_BUILD_TYPE=Release \
    -DVGRE_ENABLE_NATIVE_SIMD="$VGRE_ENABLE_NATIVE_SIMD_FLAG" \
    $_LLVM_CMAKE_ARG "${_MACOS_CMAKE_EXTRA[@]}"

_NCPU=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
if [[ "$_BUILD_CMD" == "ninja" ]]; then
    ninja -j"$_NCPU" vgre vgre_cudart vgre-worker
else
    make -j"$_NCPU" vgre vgre_cudart vgre-worker
fi

# Verify the built library is a Release build (no ASAN symbols).
# An ASan build is 2-3x larger and will abort on the first heap error it detects.
_MIN_LIB_BYTES=1048576
if [[ "$OS" == "Darwin" ]]; then
    _LIB="libvgre.dylib"
    _LIB_SIZE=$(stat -Lf%z "$_LIB" 2>/dev/null || echo 0)
    if otool -L "$_LIB" 2>/dev/null | grep -qE "libasan|libclang_rt.asan"; then
        echo "❌ ERROR: $_LIB is an ASAN build — re-run cmake with -DCMAKE_BUILD_TYPE=Release"
        exit 1
    fi
else
    _LIB="libvgre.so"
    # -L follows the SONAME symlink chain (libvgre.so -> .so.0 -> .so.0.1.0) so we
    # measure the real 53 MB library, not the 12-byte symlink. GNU stat first,
    # then BSD stat as a fallback.
    _LIB_SIZE=$(stat -Lc%s "$_LIB" 2>/dev/null || stat -Lf%z "$_LIB" 2>/dev/null || echo 0)
    if ldd "$_LIB" 2>/dev/null | grep -q "libasan"; then
        echo "❌ ERROR: $_LIB is an ASAN build — re-run cmake with -DCMAKE_BUILD_TYPE=Release"
        echo "   Run: cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j\$(nproc) vgre"
        exit 1
    fi
fi
if [[ "$_LIB_SIZE" -lt "$_MIN_LIB_BYTES" ]]; then
    echo "❌ ERROR: $_LIB is only ${_LIB_SIZE} bytes (expected ≥ ${_MIN_LIB_BYTES}) — build may have failed or symlink was measured"
    exit 1
fi
echo "✅ $_LIB verified: Release build, ${_LIB_SIZE} bytes"
cd ..

# 2. Build Flutter Dashboard
_DASHBOARD_BUILT=0
if command -v flutter >/dev/null 2>&1; then
    if [[ "$OS" == "Darwin" ]] && ! xcrun xcodebuild -version >/dev/null 2>&1; then
        echo "⚠️  Skipping dashboard build: full Xcode is required on macOS."
        echo "   Native engine, worker, and libraries will still be deployed."
        echo "   After installing Xcode, re-run: ./scripts/vgre_sync.sh"
    else
        echo "🎨 Building VGRE Dashboard (Release)..."
        cd vgre_dashboard

        # Unset CC/CXX to allow Flutter to use its own system-appropriate toolchain (usually clang)
        # and avoid linker resolution issues in LLVM directories.
        unset CC
        unset CXX

        # Workaround for Flutter/Dart linker resolution issue on Ubuntu with LLVM-18.
        # Dart's AOT compiler often expects 'ld' to be in the same dir as the compiler.
        # If 'clang' is version 18, it looks in /usr/lib/llvm-18/bin where 'ld' is missing.
        # We create a local bin dir and symlink gcc/g++/ld there so Dart finds a consistent
        # toolchain. Linux-only: macOS uses Apple Clang which does not have this issue.
        if [[ "$OS" == "Linux" ]]; then
            mkdir -p "$PROJECT_ROOT/build/vgre_bin"
            _GCC_BIN="$(command -v gcc 2>/dev/null || true)"
            _GPP_BIN="$(command -v g++ 2>/dev/null || true)"
            _LD_BIN="$(command -v ld 2>/dev/null || true)"
            [[ -x "$_GCC_BIN" ]] && ln -sf "$_GCC_BIN" "$PROJECT_ROOT/build/vgre_bin/clang"
            [[ -x "$_GPP_BIN" ]] && ln -sf "$_GPP_BIN" "$PROJECT_ROOT/build/vgre_bin/clang++"
            [[ -x "$_LD_BIN" ]] && ln -sf "$_LD_BIN" "$PROJECT_ROOT/build/vgre_bin/ld"
            [[ -x "$_LD_BIN" ]] && ln -sf "$_LD_BIN" "$PROJECT_ROOT/build/vgre_bin/ld.lld"
            export PATH="$PROJECT_ROOT/build/vgre_bin:$PATH"
        fi

        if [[ "$OS" == "Darwin" && ! -d macos ]]; then
            flutter create --platforms=macos .
        fi
        flutter build $(uname -s | tr '[:upper:]' '[:lower:]' | sed 's/darwin/macos/') --release
        _DASHBOARD_BUILT=1
        cd ..
    fi
else
    echo "⚠️  Flutter not found — dashboard build skipped"
fi
cd "$PROJECT_ROOT"

# 3. Prepare Installation Directory
echo "📂 Deploying to $INSTALL_DIR..."
mkdir -p "$INSTALL_DIR/lib"
mkdir -p "$BIN_DIR"

if [[ "$(uname)" == "Darwin" ]]; then
    # macOS deployment — always install native libraries; dashboard is optional.
    APP_BUNDLE="$PROJECT_ROOT/vgre_dashboard/build/macos/Build/Products/Release/vgre_dashboard.app"
    mkdir -p "$INSTALL_DIR/lib"
    for _dylib in build/libvgre*.dylib build/libvgre_cudart*.dylib; do
        [[ -e "$_dylib" ]] || continue
        cp -Lf "$_dylib" "$INSTALL_DIR/lib/"
    done
    echo "✅ Native libraries installed to $INSTALL_DIR/lib/"

    if [[ -d "$APP_BUNDLE" ]]; then
        cp -R "$APP_BUNDLE" "$INSTALL_DIR/"
        mkdir -p "$INSTALL_DIR/vgre_dashboard.app/Contents/Frameworks"
        for _dylib in build/libvgre*.dylib build/libvgre_cudart*.dylib; do
            [[ -e "$_dylib" ]] || continue
            cp -Lf "$_dylib" "$INSTALL_DIR/vgre_dashboard.app/Contents/Frameworks/"
        done
        echo "✅ Flutter dashboard (.app) installed."
    else
        echo "⚠️  Dashboard .app not found — install Xcode and re-run sync to build the GUI."
    fi

    # Deploy vgre-worker
    WORKER_SRC="$PROJECT_ROOT/build/src/advanced/vgre-worker"
    if [[ -f "$WORKER_SRC" ]]; then
        cp "$WORKER_SRC" "$INSTALL_DIR/vgre-worker"
        chmod +x "$INSTALL_DIR/vgre-worker"
        # Launch wrapper that injects DYLD_LIBRARY_PATH so the worker finds dylibs
        WORKER_CMD="$INSTALL_DIR/vgre-worker.sh"
        {
            echo '#!/bin/bash'
            echo "export DYLD_LIBRARY_PATH=\"$INSTALL_DIR/lib:\${DYLD_LIBRARY_PATH:-}\""
            echo "exec \"$INSTALL_DIR/vgre-worker\" \"\$@\""
        } > "$WORKER_CMD"
        chmod +x "$WORKER_CMD"
        ln -sf "$WORKER_CMD" "$BIN_DIR/vgre-worker"
        echo "✅ vgre-worker deployed to $INSTALL_DIR"
    else
        echo "⚠️  vgre-worker binary not found at $WORKER_SRC — skipping"
    fi

    # CLI symlinks installed below via vgre_install_cli_symlinks

    # Copy JIT headers
    mkdir -p "$INSTALL_DIR/include"
    cp -r "$PROJECT_ROOT/include/vgre" "$INSTALL_DIR/include/" 2>/dev/null || true

    # Dashboard launcher symlink (only when .app was built)
    if [[ -x "$INSTALL_DIR/vgre_dashboard.app/Contents/MacOS/vgre_dashboard" ]]; then
        ln -sf "$INSTALL_DIR/vgre_dashboard.app/Contents/MacOS/vgre_dashboard" "$BIN_DIR/vgre-dashboard"
    fi

    # macOS launch wrapper (DYLD_LIBRARY_PATH + optional jemalloc preload)
    LAUNCH_SCRIPT="$INSTALL_DIR/vgre-launch.sh"
    {
        echo '#!/bin/bash'
        echo '[ -f "$HOME/.profile" ] && source "$HOME/.profile" 2>/dev/null'
        echo '[ -f "$HOME/.zprofile" ] && source "$HOME/.zprofile" 2>/dev/null'
        echo "export DYLD_LIBRARY_PATH=\"$INSTALL_DIR/lib:\${DYLD_LIBRARY_PATH:-}\""
        echo "export VGRE_INSTALL_DIR=\"$INSTALL_DIR\""
        if [[ -n "$VGRE_TCP_AUTH_TOKEN_FILE" ]]; then
            echo "export VGRE_TCP_AUTH_TOKEN_FILE=\"$VGRE_TCP_AUTH_TOKEN_FILE\""
        elif [[ -f "${TOKEN_FILE:-$HOME/.vgre/token}" ]]; then
            echo "export VGRE_TCP_AUTH_TOKEN_FILE=\"${TOKEN_FILE:-$HOME/.vgre/token}\""
        fi
        _JEM_PREFIX=""
        command -v brew >/dev/null 2>&1 && _JEM_PREFIX="$(brew --prefix jemalloc 2>/dev/null || true)"
        if [[ -n "$_JEM_PREFIX" && -f "${_JEM_PREFIX}/lib/libjemalloc.2.dylib" ]]; then
            echo "export DYLD_INSERT_LIBRARIES=\"${_JEM_PREFIX}/lib/libjemalloc.2.dylib\${DYLD_INSERT_LIBRARIES:+:\$DYLD_INSERT_LIBRARIES}\""
        fi
        echo 'DASH_EXEC="'"$INSTALL_DIR"'/vgre_dashboard.app/Contents/MacOS/vgre_dashboard"'
        echo 'if [[ -x "$DASH_EXEC" ]]; then'
        echo '  exec "$DASH_EXEC" "$@"'
        echo 'fi'
        echo 'echo "VGRE dashboard not installed. Install Xcode from https://developer.apple.com/xcode/ and re-run ./scripts/vgre_sync.sh" >&2'
        echo 'echo "Native engine is ready at '"$INSTALL_DIR"'/lib/ — use vgre-worker for cluster mode." >&2'
        echo 'exit 1'
    } > "$LAUNCH_SCRIPT"
    chmod +x "$LAUNCH_SCRIPT"
    ln -sf "$LAUNCH_SCRIPT" "$BIN_DIR/vgre-launch.sh" 2>/dev/null || true
else
    # Linux deployment — Flutter's bundle directory is per-arch (x64 / arm64).
    _BUNDLE_ARCH="x64"
    case "$(uname -m)" in aarch64|arm64) _BUNDLE_ARCH="arm64" ;; esac
    BUNDLE_DIR="$PROJECT_ROOT/vgre_dashboard/build/linux/$_BUNDLE_ARCH/release/bundle"
    [[ -d "$BUNDLE_DIR" ]] || { echo "❌ Flutter bundle not found at $BUNDLE_DIR — run 'flutter build linux --release' first."; exit 1; }
    cp -r "$BUNDLE_DIR"/* "$INSTALL_DIR/"

    # Always install the freshly-built Release libraries AFTER copying the Flutter
    # bundle. Flutter's bundle build does not include libvgre.so; installing it
    # here ensures the dashboard always loads the correct non-debug, non-ASAN build.
    # Use cp -P to preserve symlinks so the loader finds both libvgre.so and libvgre.so.0.
    cp -P build/libvgre.so* "$INSTALL_DIR/lib/"
    cp -P build/libvgre_cudart.so* "$INSTALL_DIR/lib/"
    echo "✅ libvgre.so $(md5sum build/libvgre.so | awk '{print $1}') installed to $INSTALL_DIR/lib/"

    # Deploy vgre-worker.
    # The binary lives at build/src/advanced/vgre-worker (CMake output path).
    # If the destination is busy (worker process still running), stop it first,
    # then copy to a temp file and atomically rename to avoid ETXTBSY.
    WORKER_SRC="$PROJECT_ROOT/build/src/advanced/vgre-worker"
    WORKER_DST="$INSTALL_DIR/vgre-worker"
    if [[ ! -f "$WORKER_SRC" ]]; then
        echo "⚠️  vgre-worker binary not found at $WORKER_SRC — skipping worker deploy"
    else
        # Stop any running worker so the destination file is not busy.
        if pgrep -x vgre-worker >/dev/null 2>&1; then
            echo "🛑 Stopping running vgre-worker before deploy..."
            pkill -x vgre-worker || true
            sleep 1
        fi
        # Copy to a temp file then rename — atomic on Linux, avoids ETXTBSY
        # even if a stale process still holds the old inode open.
        cp "$WORKER_SRC" "${WORKER_DST}.new"
        mv -f "${WORKER_DST}.new" "$WORKER_DST"
        echo "✅ vgre-worker deployed to $WORKER_DST"
    fi
    
    # Copy essential JIT headers
    mkdir -p "$INSTALL_DIR/include"
    cp -r "$PROJECT_ROOT/include/vgre" "$INSTALL_DIR/include/"
    
    # Copy icon for desktop integration
    cp "$PROJECT_ROOT/vgre_dashboard/assets/icon.png" "$INSTALL_DIR/vgre_icon.png" 2>/dev/null || true
    
    # Create a robust launch wrapper. We inject the known token directly into the 
    # wrapper to bypass unreliability with GUI launchers sourcing .bashrc without TTYs.
    LAUNCH_SCRIPT="$INSTALL_DIR/vgre-launch.sh"
    echo '#!/bin/bash' > "$LAUNCH_SCRIPT"
    echo '# Source user profiles to inherit env vars if possible' >> "$LAUNCH_SCRIPT"
    echo '[ -f "$HOME/.profile" ] && source "$HOME/.profile" 2>/dev/null' >> "$LAUNCH_SCRIPT"
    echo '[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc" 2>/dev/null' >> "$LAUNCH_SCRIPT"
    echo '' >> "$LAUNCH_SCRIPT"
    
    # Use token file instead of raw token to keep the secret out of the process list.
    if [[ -n "$VGRE_TCP_AUTH_TOKEN_FILE" ]]; then
        echo "# Token file path set at install time" >> "$LAUNCH_SCRIPT"
        echo "export VGRE_TCP_AUTH_TOKEN_FILE=\"$VGRE_TCP_AUTH_TOKEN_FILE\"" >> "$LAUNCH_SCRIPT"
    elif [[ -f "${TOKEN_FILE:-$HOME/.vgre/token}" ]]; then
        echo "# Read auth token from file (written during sync)" >> "$LAUNCH_SCRIPT"
        echo "export VGRE_TCP_AUTH_TOKEN_FILE=\"${TOKEN_FILE:-$HOME/.vgre/token}\"" >> "$LAUNCH_SCRIPT"
    fi
    
    # Set LD_LIBRARY_PATH with proper precedence (new path first)
    echo "export LD_LIBRARY_PATH=\"$INSTALL_DIR/lib:\$LD_LIBRARY_PATH\"" >> "$LAUNCH_SCRIPT"
    echo "export VGRE_CACHE_DIR=\"\$HOME/.vgre/cache\"" >> "$LAUNCH_SCRIPT"
    echo "mkdir -p \"\$VGRE_CACHE_DIR\"" >> "$LAUNCH_SCRIPT"
    echo "cd \"$INSTALL_DIR\"" >> "$LAUNCH_SCRIPT"
    echo '' >> "$LAUNCH_SCRIPT"
    echo '# ── Memory allocator: use jemalloc to avoid glibc malloc arena' >> "$LAUNCH_SCRIPT"
    echo '#    contention between Flutter (libc++) and VGRE (libstdc++) threads.' >> "$LAUNCH_SCRIPT"
    echo '#    glibc malloc triggers internal assertions when many threads from' >> "$LAUNCH_SCRIPT"
    echo '#    both runtimes allocate concurrently against shared per-thread arenas.' >> "$LAUNCH_SCRIPT"
    echo '#    jemalloc resolves this with its own arena management strategy.' >> "$LAUNCH_SCRIPT"
    echo 'JEMALLOC_LIB=""' >> "$LAUNCH_SCRIPT"
    echo 'for candidate in \' >> "$LAUNCH_SCRIPT"
    echo '    /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \' >> "$LAUNCH_SCRIPT"
    echo '    /usr/lib64/libjemalloc.so.2 \' >> "$LAUNCH_SCRIPT"
    echo '    /usr/lib/libjemalloc.so.2 \' >> "$LAUNCH_SCRIPT"
    echo '    /usr/local/lib/libjemalloc.so.2; do' >> "$LAUNCH_SCRIPT"
    echo '    if [ -f "$candidate" ]; then JEMALLOC_LIB="$candidate"; break; fi' >> "$LAUNCH_SCRIPT"
    echo 'done' >> "$LAUNCH_SCRIPT"
    echo 'if [ -n "$JEMALLOC_LIB" ]; then' >> "$LAUNCH_SCRIPT"
    echo '    export LD_PRELOAD="$JEMALLOC_LIB${LD_PRELOAD:+:$LD_PRELOAD}"' >> "$LAUNCH_SCRIPT"
    echo 'else' >> "$LAUNCH_SCRIPT"
    echo '    echo "[VGRE] Warning: jemalloc not found; falling back to glibc malloc." >&2' >> "$LAUNCH_SCRIPT"
    echo '    echo "[VGRE] Install with: sudo apt-get install libjemalloc2  (Debian/Ubuntu)" >&2' >> "$LAUNCH_SCRIPT"
    echo '    echo "[VGRE] Install with: sudo dnf install jemalloc          (Fedora/RHEL)" >&2' >> "$LAUNCH_SCRIPT"
    echo 'fi' >> "$LAUNCH_SCRIPT"
    echo '' >> "$LAUNCH_SCRIPT"
    echo '# Ensure dashboard stays running even if parent shell exits' >> "$LAUNCH_SCRIPT"
    echo 'nohup ./vgre_dashboard "$@" > /tmp/vgre_dashboard.log 2>&1 &' >> "$LAUNCH_SCRIPT"
    echo 'DASHBOARD_PID=$!' >> "$LAUNCH_SCRIPT"
    echo 'wait $DASHBOARD_PID' >> "$LAUNCH_SCRIPT"
    
    chmod +x "$LAUNCH_SCRIPT"
    
    # Desktop Integration
    DESKTOP_DIR="$HOME/.local/share/applications"
    mkdir -p "$DESKTOP_DIR"
    
    # Remove old conflicting entries
    rm -f "$DESKTOP_DIR/vgre.desktop"
    rm -f "$DESKTOP_DIR/vgre-dashboard.desktop"
    
    cat <<EOF > "$DESKTOP_DIR/vgre-dashboard.desktop"
[Desktop Entry]
Version=1.0
Type=Application
Name=VGRE Dashboard
Comment=Virtual GPU Runtime Engine
Exec=bash -ic "$INSTALL_DIR/vgre-launch.sh"
Icon=$INSTALL_DIR/vgre_icon.png
Terminal=false
Categories=Development;System;Utility;
Path=$INSTALL_DIR
EOF
    
    chmod +x "$INSTALL_DIR/vgre_dashboard"
    chmod +x "$INSTALL_DIR/vgre-worker"
    ln -sf "$INSTALL_DIR/vgre-launch.sh" "$BIN_DIR/vgre-dashboard"
    ln -sf "$INSTALL_DIR/vgre-worker" "$BIN_DIR/vgre-worker"
    # CLI symlinks installed below via vgre_install_cli_symlinks
fi

# ── vgre-worker self-check ────────────────────────────────────────────────────
# Run vgre-worker --help with proper library paths.  Exit code must be exactly 0.
# Any non-zero code (including negative values from OS signals or DLL init failures)
# is treated as a hard error.
_WORKER_BIN="$INSTALL_DIR/vgre-worker"
if [[ -x "$_WORKER_BIN" ]]; then
    echo ""
    echo "=== Validating vgre-worker ==="
    if [[ "$OS" == "Darwin" ]]; then
        DYLD_LIBRARY_PATH="$INSTALL_DIR/lib:${DYLD_LIBRARY_PATH:-}" "$_WORKER_BIN" --help >/dev/null 2>&1
    else
        LD_LIBRARY_PATH="$INSTALL_DIR/lib:${LD_LIBRARY_PATH:-}" "$_WORKER_BIN" --help >/dev/null 2>&1
    fi
    _WORKER_RC=$?
    if [[ $_WORKER_RC -ne 0 ]]; then
        echo "ERROR: vgre-worker failed startup self-check (exit code $_WORKER_RC)."
        echo "       Check that libvgre.so / libvgre.dylib is in $INSTALL_DIR/lib"
        exit 1
    else
        echo "[OK] vgre-worker startup self-check passed"
    fi
fi

# Ensure VGRE CLI tools are on PATH (~/.local/bin)
# shellcheck source=vgre-cli-install.sh
. "$SCRIPT_DIR/vgre-cli-install.sh"
vgre_install_cli_symlinks "$SCRIPT_DIR"
vgre_ensure_cli_path

echo ""
echo "✅ VGRE Sync Complete!"
echo "📍 Installed to: $INSTALL_DIR"
echo ""
if [[ -f "$HOME/.vgre/token" ]]; then
    echo "🔐 Cluster auth token: ready  ($HOME/.vgre/token)"
    echo ""
    echo "   Start master:            vgre-start --master"
    echo "   Start worker (LAN):      vgre-start --worker"
    echo "   Start worker (WAN):      vgre-start --worker --master-address <IP>:<PORT>"
    echo "   Local test:              vgre-start --test"
    echo ""
    echo "   Detect public IP:        vgre-discover"
    echo "   Register for discovery:  vgre-discover --register"
    echo "   Find master (worker):    vgre-discover --find <BUCKET_ID>"
else
    echo "⚙️  Cluster not yet configured. Run the setup to enable secure clustering:"
    echo "   bash scripts/setup-cluster.sh"
fi
echo ""
echo "🚀 Launch dashboard: vgre-dashboard  (or from your application menu)"
