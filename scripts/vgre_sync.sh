#!/bin/bash
set -e

# VGRE Sync & Install Script (Linux/macOS)
# This script builds the VGRE engine and dashboard and installs it to the local user profile.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INSTALL_DIR="$HOME/.local/share/VGRE"
BIN_DIR="$HOME/.local/bin"

cd "$PROJECT_ROOT"

echo "🚀 Starting VGRE Global Sync..."

# 1. Build Native Engine
echo "📦 Building VGRE Native Engine..."
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu) vgre vgre_cudart
cd ..

# 2. Build Flutter Dashboard
echo "🎨 Building VGRE Dashboard (Release)..."
cd vgre_dashboard
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
    
    # Copy essential JIT headers
    mkdir -p "$INSTALL_DIR/include"
    cp -r "$PROJECT_ROOT/include/vgre" "$INSTALL_DIR/include/"
    
    # Copy icon for desktop integration
    cp "$PROJECT_ROOT/vgre_dashboard/assets/icon.png" "$INSTALL_DIR/vgre_icon.png" 2>/dev/null || true
    
    # Create a robust launch wrapper
    cat <<EOF > "$INSTALL_DIR/vgre-launch.sh"
#!/bin/bash
export LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:$INSTALL_DIR/lib"
cd "$INSTALL_DIR"
exec ./vgre_dashboard "\$@"
EOF
    chmod +x "$INSTALL_DIR/vgre-launch.sh"
    
    # Desktop Integration
    DESKTOP_DIR="$HOME/.local/share/applications"
    mkdir -p "$DESKTOP_DIR"
    
    # Remove old conflicting entry
    rm -f "$DESKTOP_DIR/vgre.desktop"
    
    cat <<EOF > "$DESKTOP_DIR/vgre-dashboard.desktop"
[Desktop Entry]
Version=1.0
Type=Application
Name=VGRE Dashboard
Comment=Virtual GPU Runtime Engine
Exec=$INSTALL_DIR/vgre-launch.sh
Icon=$INSTALL_DIR/vgre_icon.png
Terminal=false
Categories=Development;System;Utility;
Path=$INSTALL_DIR
EOF
    
    chmod +x "$INSTALL_DIR/vgre_dashboard"
    ln -sf "$INSTALL_DIR/vgre-launch.sh" "$BIN_DIR/vgre-dashboard"
fi

echo "✅ VGRE Sync Complete!"
echo "📍 Installed to: $INSTALL_DIR"
echo "🚀 You can now launch 'VGRE Dashboard' from your application menu or run 'vgre-dashboard' in your terminal."
