#!/bin/bash
set -e

# VGRE Sync & Install Script (Linux/macOS)
# This script builds the VGRE engine and dashboard and installs it to the local user profile.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INSTALL_DIR="$HOME/.local/share/VGRE"
BIN_DIR="$HOME/.local/bin"
VGRE_ENABLE_NATIVE_SIMD_FLAG="${VGRE_ENABLE_NATIVE_SIMD:-0}"

# Phase 10: Dynamic Auth Token Handling
# If VGRE_TCP_AUTH_TOKEN is not in env, we do NOT default to a hardcoded string.
# This ensures truth in the security model.
if [[ -z "$VGRE_TCP_AUTH_TOKEN" ]]; then
    echo "⚠️  VGRE_TCP_AUTH_TOKEN is not set."
    echo "   Dashboard will fallback to Hardware Secure Storage (TPM) or allow manual input."
else
    echo "🔐 VGRE_TCP_AUTH_TOKEN provided in environment; secure cluster mode active."
    # Sync to local cache for dashboard dev fallback
    mkdir -p "$HOME/.vgre"
    echo "$VGRE_TCP_AUTH_TOKEN" > "$HOME/.vgre/token"
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
    
    if [[ -n "$VGRE_TCP_AUTH_TOKEN" ]]; then
        echo "# Injecting known install-time auth token for guaranteed Dashboard visibility" >> "$LAUNCH_SCRIPT"
        echo "export VGRE_TCP_AUTH_TOKEN=\"$VGRE_TCP_AUTH_TOKEN\"" >> "$LAUNCH_SCRIPT"
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
fi

echo "✅ VGRE Sync Complete!"
echo "📍 Installed to: $INSTALL_DIR"
echo "🚀 You can now launch 'VGRE Dashboard' from your application menu or run 'vgre-dashboard' in your terminal."
