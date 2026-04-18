#!/bin/bash
set -e

# VGRE Sync & Install Script (Linux/macOS)
# This script builds the VGRE engine and dashboard and installs it to the local user profile.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INSTALL_DIR="$HOME/.local/share/VGRE"
BIN_DIR="$HOME/.local/bin"
VGRE_ENABLE_NATIVE_SIMD_FLAG="${VGRE_ENABLE_NATIVE_SIMD:-0}"

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
    echo "   Run  bash scripts/setup-cluster.sh  to set one up (recommended)."
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
    cp build/libvgre.so "$INSTALL_DIR/lib/"
    cp build/libvgre_cudart.so "$INSTALL_DIR/lib/"
    cp build/src/advanced/vgre-worker "$INSTALL_DIR/" || cp build/bin/vgre-worker "$INSTALL_DIR/" || cp build/vgre-worker "$INSTALL_DIR/"
    
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
    echo '[ -f "$HOME/.profile" ] && source "$HOME/.profile"' >> "$LAUNCH_SCRIPT"
    echo '[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"' >> "$LAUNCH_SCRIPT"
    echo '' >> "$LAUNCH_SCRIPT"
    
    # Use token file instead of raw token to keep the secret out of the process list.
    if [[ -n "$VGRE_TCP_AUTH_TOKEN_FILE" ]]; then
        echo "# Token file path set at install time" >> "$LAUNCH_SCRIPT"
        echo "export VGRE_TCP_AUTH_TOKEN_FILE=\"$VGRE_TCP_AUTH_TOKEN_FILE\"" >> "$LAUNCH_SCRIPT"
    elif [[ -f "${TOKEN_FILE:-$HOME/.vgre/token}" ]]; then
        echo "# Read auth token from file (written during sync)" >> "$LAUNCH_SCRIPT"
        echo "export VGRE_TCP_AUTH_TOKEN_FILE=\"${TOKEN_FILE:-$HOME/.vgre/token}\"" >> "$LAUNCH_SCRIPT"
    fi
    
    echo "export LD_LIBRARY_PATH=\"\$LD_LIBRARY_PATH:$INSTALL_DIR/lib\"" >> "$LAUNCH_SCRIPT"
    echo "cd \"$INSTALL_DIR\"" >> "$LAUNCH_SCRIPT"
    echo 'exec ./vgre_dashboard "$@"' >> "$LAUNCH_SCRIPT"
    
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
