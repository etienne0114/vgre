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

    # LLVM dev libraries
    local LLVM_FOUND=0
    for cfg in llvm-config llvm-config-18 llvm-config-17 llvm-config-16; do
        command -v "$cfg" >/dev/null 2>&1 && LLVM_FOUND=1 && \
            echo "  [OK] LLVM $($cfg --version) ($cfg)" && break
    done
    if [[ $LLVM_FOUND -eq 0 ]]; then
        echo "  [MISSING] LLVM dev libraries"
        MISSING_APT+=(llvm-dev clang libclang-dev); MISSING_DNF+=(llvm-devel clang-devel)
        MISSING_BREW+=(llvm)
        NEED_INSTALL=1
    fi

    # OpenMP
    local OMP_FOUND=0
    for lib in /usr/lib/x86_64-linux-gnu/libomp.so \
               /usr/lib/aarch64-linux-gnu/libomp.so \
               /usr/local/lib/libomp.dylib; do
        [[ -f "$lib" ]] && OMP_FOUND=1 && break
    done
    command -v dpkg >/dev/null 2>&1 && dpkg -l libomp-dev >/dev/null 2>&1 && OMP_FOUND=1
    if [[ $OMP_FOUND -eq 0 ]]; then
        echo "  [MISSING] OpenMP (libomp-dev)"
        MISSING_APT+=(libomp-dev); MISSING_DNF+=(libomp-devel); MISSING_BREW+=(libomp)
        NEED_INSTALL=1
    else
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
            local FLUTTER_URL="https://storage.googleapis.com/flutter_infra_release/releases/stable/linux/flutter_linux_${FLUTTER_VERSION}-stable.tar.xz"
            echo "  Downloading Flutter ${FLUTTER_VERSION}..."
            mkdir -p "$HOME/.local"
            curl -fsSL "$FLUTTER_URL" | tar -xJ -C "$HOME/.local"
            ln -sf "$HOME/.local/flutter/bin/flutter" "$BIN_DIR/flutter" 2>/dev/null || true
            export PATH="$HOME/.local/flutter/bin:$PATH"
        fi
    fi
    if command -v flutter >/dev/null 2>&1; then
        echo "  [OK] Flutter $(flutter --version 2>/dev/null | head -1 | awk '{print $2}')"
    else
        echo "  [WARN] Flutter unavailable — dashboard build will be skipped"
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
    TOKEN_FILE=""
    echo "⚠️  No auth token configured."
    echo "   Run  vgre-token generate  after install to create one (recommended)."
    echo "   Or:  bash install_local.sh  to build and generate the token in one step."
    echo "   Cluster will use hardware secure storage (TPM/Keyring) or allow manual input."
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
    # Clear cache to force Re-detection with correct compiler
    rm -f CMakeCache.txt
fi

cmake .. -DCMAKE_BUILD_TYPE=Release -DVGRE_ENABLE_NATIVE_SIMD="$VGRE_ENABLE_NATIVE_SIMD_FLAG"
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu) vgre vgre_cudart vgre-worker

# Verify the built library is a Release build (no ASAN symbols).
# An ASan build is 2-3x larger and will abort on the first heap error it detects.
_LIB_SIZE=$(stat -c%s libvgre.so 2>/dev/null || stat -f%z libvgre.so 2>/dev/null || echo 0)
if ldd libvgre.so 2>/dev/null | grep -q "libasan"; then
    echo "❌ ERROR: libvgre.so is an ASAN build — re-run cmake with -DCMAKE_BUILD_TYPE=Release"
    echo "   Run: cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j\$(nproc) vgre"
    exit 1
fi
echo "✅ libvgre.so verified: Release build, ${_LIB_SIZE} bytes"
cd ..

# 2. Build Flutter Dashboard
echo "🎨 Building VGRE Dashboard (Release)..."
cd vgre_dashboard

# Unset CC/CXX to allow Flutter to use its own system-appropriate toolchain (usually clang)
# and avoid linker resolution issues in LLVM directories.
unset CC
unset CXX

# Workaround for Flutter/Dart linker resolution issue on Ubuntu with LLVM-18
# Dart's AOT compiler often expects 'ld' to be in the same dir as the compiler.
# If 'clang' is version 18, it looks in /usr/lib/llvm-18/bin where 'ld' is missing.
# We create a local bin dir and provide 'clang'/'clang++' symlinks to 'gcc'/'g++'
# alongside a symlinked 'ld' and 'ld.lld'. This tricks Dart into using GCC.
mkdir -p "$PROJECT_ROOT/build/vgre_bin"
ln -sf /usr/bin/gcc "$PROJECT_ROOT/build/vgre_bin/clang"
ln -sf /usr/bin/g++ "$PROJECT_ROOT/build/vgre_bin/clang++"
ln -sf /usr/bin/ld "$PROJECT_ROOT/build/vgre_bin/ld"
ln -sf /usr/bin/ld "$PROJECT_ROOT/build/vgre_bin/ld.lld"
export PATH="$PROJECT_ROOT/build/vgre_bin:$PATH"

flutter build $(uname -s | tr '[:upper:]' '[:lower:]' | sed 's/darwin/macos/') --release
cd ..

# 3. Prepare Installation Directory
echo "📂 Deploying to $INSTALL_DIR..."
mkdir -p "$INSTALL_DIR/lib"
mkdir -p "$BIN_DIR"

if [[ "$(uname)" == "Darwin" ]]; then
    # macOS deployment
    APP_BUNDLE="$PROJECT_ROOT/vgre_dashboard/build/macos/Build/Products/Release/vgre_dashboard.app"
    cp -r "$APP_BUNDLE" "$INSTALL_DIR/"
    cp build/libvgre.dylib "$INSTALL_DIR/vgre_dashboard.app/Contents/Frameworks/" 2>/dev/null || \
    cp build/libvgre.dylib "$INSTALL_DIR/lib/"
    
    # Create a symlink in bin
    ln -sf "$INSTALL_DIR/vgre_dashboard.app/Contents/MacOS/vgre_dashboard" "$BIN_DIR/vgre-dashboard"
else
    # Linux deployment
    BUNDLE_DIR="$PROJECT_ROOT/vgre_dashboard/build/linux/x64/release/bundle"
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
    # Install vgre-start for easy cluster management
    if [[ -f "$SCRIPT_DIR/vgre-start.sh" ]]; then
        chmod +x "$SCRIPT_DIR/vgre-start.sh"
        ln -sf "$SCRIPT_DIR/vgre-start.sh" "$BIN_DIR/vgre-start"
    fi
    # Install vgre-token token manager (works from any directory)
    if [[ -f "$SCRIPT_DIR/vgre-token.sh" ]]; then
        chmod +x "$SCRIPT_DIR/vgre-token.sh"
        ln -sf "$SCRIPT_DIR/vgre-token.sh" "$BIN_DIR/vgre-token"
    fi
fi

echo ""
echo "✅ VGRE Sync Complete!"
echo "📍 Installed to: $INSTALL_DIR"
echo ""
if [[ -f "$HOME/.vgre/token" ]]; then
    echo "🔐 Cluster auth token: ready  ($HOME/.vgre/token)"
    echo ""
    echo "   Start master:  vgre-start --master"
    echo "   Start worker:  vgre-start --worker"
    echo "   Local test:    vgre-start --test"
else
    echo "⚙️  Cluster not yet configured. Run the setup to enable secure clustering:"
    echo "   bash scripts/setup-cluster.sh"
fi
echo ""
echo "🚀 Launch dashboard: vgre-dashboard  (or from your application menu)"
