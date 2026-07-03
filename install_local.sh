#!/usr/bin/env bash
# VGRE Local Deployment Script
# Verifies all required dependencies (and installs missing ones automatically),
# builds the native engine + Flutter dashboard, installs everything under
# ~/.local/share/VGRE, and writes all required environment variables to
# ~/.vgre/env so they are sourced automatically on every new terminal.
#
# Usage:
#   bash install_local.sh                 # verify deps, full build + install
#   bash install_local.sh --no-build      # skip rebuild (install already-built artifacts)
#   bash install_local.sh --dev           # development mode: run from repo, no install
#   bash install_local.sh --check         # only verify deps, do not build or install
#   bash install_local.sh --prefix DIR    # install somewhere other than ~/.local/share/VGRE
#   bash install_local.sh --skip-profile  # don't touch shell profiles (CI/containers)
#
# All paths are also overridable via env: VGRE_INSTALL_DIR, VGRE_BIN_DIR, VGRE_DIR.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
# Every location is overridable (env var or --prefix) — nothing is hard-wired
# to a specific user layout.
INSTALL_DIR="${VGRE_INSTALL_DIR:-$HOME/.local/share/VGRE}"
BIN_DIR="${VGRE_BIN_DIR:-$HOME/.local/bin}"
VGRE_DIR="${VGRE_DIR:-$HOME/.vgre}"
DESKTOP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"

NO_BUILD=0
DEV_MODE=0
CHECK_ONLY=0
SKIP_PROFILE="${VGRE_SKIP_PROFILE:-0}"
_prev=""
for arg in "$@"; do
    if [[ "$_prev" == "--prefix" ]]; then
        INSTALL_DIR="$arg"
        _prev=""
        continue
    fi
    case "$arg" in
        --no-build)     NO_BUILD=1 ;;
        --dev)          DEV_MODE=1 ;;
        --check)        CHECK_ONLY=1 ;;
        --prefix)       _prev="--prefix" ;;
        --prefix=*)     INSTALL_DIR="${arg#--prefix=}" ;;
        --skip-profile) SKIP_PROFILE=1 ;;
    esac
done
ENV_FILE="$VGRE_DIR/env"
TOKEN_FILE="$VGRE_DIR/token"

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
ok()   { echo -e "${GREEN}✓${RESET} $*"; }
info() { echo -e "${CYAN}▸${RESET} $*"; }
warn() { echo -e "${YELLOW}⚠${RESET} $*"; }
fail() { echo -e "${RED}✗${RESET} $*" >&2; exit 1; }

echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║          VGRE Local Deployment                           ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# ── Dependency verification + auto-install ────────────────────────────────────
# Detects the OS package manager and installs any missing required tools.
# Requires sudo for Linux package installation; macOS uses Homebrew (no sudo).

OS="$(uname -s)"     # Linux | Darwin
ARCH="$(uname -m)"   # x86_64 | arm64 | aarch64

_pkg_install_linux() {
    local PKGS=("$@")
    if command -v apt-get >/dev/null 2>&1; then
        info "Installing via apt-get: ${PKGS[*]}"
        sudo apt-get update -qq && sudo apt-get install -y "${PKGS[@]}"
    elif command -v dnf >/dev/null 2>&1; then
        info "Installing via dnf: ${PKGS[*]}"
        sudo dnf install -y "${PKGS[@]}"
    elif command -v pacman >/dev/null 2>&1; then
        info "Installing via pacman: ${PKGS[*]}"
        sudo pacman -Sy --noconfirm "${PKGS[@]}"
    elif command -v zypper >/dev/null 2>&1; then
        info "Installing via zypper: ${PKGS[*]}"
        sudo zypper install -y "${PKGS[@]}"
    else
        fail "No supported package manager found. Please install manually: ${PKGS[*]}"
    fi
}

_install_flutter_linux() {
    if command -v flutter >/dev/null 2>&1; then return 0; fi
    info "Flutter not found. Attempting install..."
    # Try snap first (Ubuntu/Debian with snapd)
    if command -v snap >/dev/null 2>&1; then
        sudo snap install flutter --classic
        return 0
    fi
    # Fall back to direct download
    local FLUTTER_VERSION="3.24.5"
    local FLUTTER_ARCH="x64"
    [[ "$ARCH" == "arm64" || "$ARCH" == "aarch64" ]] && FLUTTER_ARCH="arm64"
    local FLUTTER_URL="https://storage.googleapis.com/flutter_infra_release/releases/stable/linux/flutter_linux_${FLUTTER_VERSION}-stable.tar.xz"
    local FLUTTER_DIR="$HOME/.local/flutter"
    info "Downloading Flutter ${FLUTTER_VERSION} (${FLUTTER_ARCH})..."
    mkdir -p "$HOME/.local"
    curl -fsSL "$FLUTTER_URL" | tar -xJ -C "$HOME/.local"
    mkdir -p "$BIN_DIR"
    ln -sf "$FLUTTER_DIR/bin/flutter" "$BIN_DIR/flutter"
    export PATH="$FLUTTER_DIR/bin:$PATH"
    ok "Flutter installed to $FLUTTER_DIR"
}

_install_flutter_macos() {
    if command -v flutter >/dev/null 2>&1; then return 0; fi
    if command -v brew >/dev/null 2>&1; then
        info "Installing Flutter via Homebrew..."
        brew install --cask flutter
    else
        warn "Flutter not found and Homebrew is unavailable."
        warn "Install Flutter from https://flutter.dev/docs/get-started/install/macos"
        fail "Flutter is required to build the dashboard."
    fi
}

check_and_install_deps() {
    info "Verifying required dependencies..."
    local APT_MISSING=()
    local DNF_MISSING=()
    local BREW_MISSING=()
    local FAIL_MSGS=()
    local ALL_OK=1

    # ── cmake ──────────────────────────────────────────────────────────────────
    if ! command -v cmake >/dev/null 2>&1; then
        warn "cmake not found"
        APT_MISSING+=(cmake); DNF_MISSING+=(cmake); BREW_MISSING+=(cmake)
        ALL_OK=0
    else
        ok "cmake $(cmake --version | head -1 | awk '{print $3}')"
    fi

    # ── C++ compiler ──────────────────────────────────────────────────────────
    if ! command -v g++ >/dev/null 2>&1 && ! command -v clang++ >/dev/null 2>&1; then
        warn "No C++ compiler found (g++ or clang++)"
        APT_MISSING+=(build-essential g++); DNF_MISSING+=("gcc-c++" make); BREW_MISSING+=()
        ALL_OK=0
    else
        local CXX_VER
        CXX_VER=$(g++ --version 2>/dev/null | head -1 || clang++ --version 2>/dev/null | head -1)
        ok "C++ compiler: $CXX_VER"
    fi

    # ── make ──────────────────────────────────────────────────────────────────
    if ! command -v make >/dev/null 2>&1 && ! command -v ninja >/dev/null 2>&1; then
        warn "Neither make nor ninja found"
        APT_MISSING+=(make); DNF_MISSING+=(make); BREW_MISSING+=(make)
        ALL_OK=0
    else
        ok "Build driver: $(command -v make 2>/dev/null || command -v ninja)"
    fi

    # ── LLVM 18 (required for JIT) ───────────────────────────────────────────
    local LLVM_OK=0 LLVM_CFG=""
    for llvm_cfg in llvm-config-18 llvm-config; do
        if command -v "$llvm_cfg" >/dev/null 2>&1; then
            local _llvm_ver
            _llvm_ver=$("$llvm_cfg" --version 2>/dev/null | grep -oE '^[0-9]+' || echo "0")
            if [[ "$_llvm_ver" == "18" ]]; then
                LLVM_CFG="$llvm_cfg"
                LLVM_OK=1
                break
            fi
        fi
    done
    if [[ $LLVM_OK -eq 0 && "$OS" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
        local _brew_cfg
        _brew_cfg="$(brew --prefix llvm@18 2>/dev/null)/bin/llvm-config"
        if [[ -x "$_brew_cfg" ]]; then
            local _llvm_ver
            _llvm_ver=$("$_brew_cfg" --version 2>/dev/null | grep -oE '^[0-9]+' || echo "0")
            if [[ "$_llvm_ver" == "18" ]]; then
                LLVM_CFG="$_brew_cfg"
                LLVM_OK=1
            fi
        fi
    fi
    if [[ $LLVM_OK -eq 1 ]]; then
        ok "LLVM: $($LLVM_CFG --version) (${LLVM_CFG})"
    else
        warn "LLVM 18 dev libraries not found (required for JIT compilation)"
        APT_MISSING+=(llvm-18-dev clang-18 libclang-18-dev)
        DNF_MISSING+=(llvm18-devel clang18-devel)
        BREW_MISSING+=(llvm@18)
        ALL_OK=0
    fi

    # ── OpenMP ────────────────────────────────────────────────────────────────
    local OMP_FOUND=0
    for omp_lib in /usr/lib/x86_64-linux-gnu/libomp.so \
                   /usr/lib/aarch64-linux-gnu/libomp.so \
                   /usr/lib/llvm-*/lib/libomp.so \
                   /usr/local/lib/libomp.dylib \
                   "$(brew --prefix libomp 2>/dev/null)/lib/libomp.dylib"; do
        # shellcheck disable=SC2086
        ls $omp_lib >/dev/null 2>&1 && OMP_FOUND=1 && break
    done
    if command -v dpkg >/dev/null 2>&1 && dpkg -l libomp-dev >/dev/null 2>&1; then OMP_FOUND=1; fi
    if [[ $OMP_FOUND -eq 0 ]]; then
        warn "OpenMP (libomp-dev) not found"
        APT_MISSING+=(libomp-dev); DNF_MISSING+=(libomp-devel); BREW_MISSING+=(libomp)
        ALL_OK=0
    else
        ok "OpenMP library present"
    fi

    # ── openssl ───────────────────────────────────────────────────────────────
    if ! command -v openssl >/dev/null 2>&1; then
        warn "openssl not found"
        APT_MISSING+=(libssl-dev openssl); DNF_MISSING+=(openssl-devel openssl); BREW_MISSING+=(openssl)
        ALL_OK=0
    else
        ok "openssl $(openssl version | awk '{print $2}')"
    fi

    # ── git ───────────────────────────────────────────────────────────────────
    if ! command -v git >/dev/null 2>&1; then
        warn "git not found"
        APT_MISSING+=(git); DNF_MISSING+=(git); BREW_MISSING+=(git)
        ALL_OK=0
    else
        ok "git $(git --version | awk '{print $3}')"
    fi

    # ── curl (for Flutter download fallback) ──────────────────────────────────
    if ! command -v curl >/dev/null 2>&1; then
        warn "curl not found"
        APT_MISSING+=(curl); DNF_MISSING+=(curl); BREW_MISSING+=(curl)
        ALL_OK=0
    fi

    # ── Auto-install missing packages ─────────────────────────────────────────
    if [[ $ALL_OK -eq 0 ]]; then
        echo ""
        warn "Missing dependencies detected. Attempting automatic installation..."
        echo ""
        if [[ "$OS" == "Linux" ]]; then
            if [[ ${#APT_MISSING[@]} -gt 0 ]] && command -v apt-get >/dev/null 2>&1; then
                _pkg_install_linux "${APT_MISSING[@]}"
            elif [[ ${#DNF_MISSING[@]} -gt 0 ]] && command -v dnf >/dev/null 2>&1; then
                _pkg_install_linux "${DNF_MISSING[@]}"
            elif command -v pacman >/dev/null 2>&1; then
                # Map to pacman package names
                local PAC_PKGS=(base-devel cmake llvm clang openssl libomp git curl)
                _pkg_install_linux "${PAC_PKGS[@]}"
            fi
        elif [[ "$OS" == "Darwin" ]]; then
            if ! command -v brew >/dev/null 2>&1; then
                info "Homebrew not found — installing Homebrew first..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            if [[ ${#BREW_MISSING[@]} -gt 0 ]]; then
                info "Installing via Homebrew: ${BREW_MISSING[*]}"
                brew install "${BREW_MISSING[@]}"
            fi
        fi
    fi

    # ── Flutter ───────────────────────────────────────────────────────────────
    if ! command -v flutter >/dev/null 2>&1; then
        if [[ "$OS" == "Linux" ]]; then
            _install_flutter_linux
        elif [[ "$OS" == "Darwin" ]]; then
            _install_flutter_macos
        fi
    fi
    if command -v flutter >/dev/null 2>&1; then
        ok "Flutter $(flutter --version --machine 2>/dev/null | python3 -c \
            'import json,sys; d=json.load(sys.stdin); print(d.get("frameworkVersion","?"))' \
            2>/dev/null || flutter --version | head -1 | awk '{print $2}')"
    else
        warn "Flutter not available — dashboard build will be skipped."
    fi
    if [[ "$OS" == "Darwin" ]] && ! xcrun xcodebuild -version >/dev/null 2>&1; then
        warn "Xcode not installed — Flutter macOS dashboard cannot be built."
        warn "Install from https://developer.apple.com/xcode/ then: sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer"
    fi

    # ── Final pass: re-verify critical tools ──────────────────────────────────
    echo ""
    info "Final dependency check:"
    local HARD_FAIL=0
    for REQUIRED_CMD in cmake make git; do
        if ! command -v "$REQUIRED_CMD" >/dev/null 2>&1; then
            warn "$REQUIRED_CMD still missing after auto-install"
            HARD_FAIL=1
        fi
    done
    if ! command -v g++ >/dev/null 2>&1 && ! command -v clang++ >/dev/null 2>&1; then
        warn "C++ compiler still missing"
        HARD_FAIL=1
    fi
    if [[ $HARD_FAIL -eq 1 ]]; then
        fail "One or more critical dependencies could not be installed automatically."$'\n'"Please install them manually and re-run this script."
    fi
    ok "All required dependencies present."

    if [[ $CHECK_ONLY -eq 1 ]]; then
        ok "Dependency check complete (--check mode)."
        exit 0
    fi
    # Explicit success: without this, the [[ ]] above returning non-zero would
    # make the function's exit status 1 and set -e would abort the installer.
    return 0
}

check_and_install_deps

# ── Step 1: Build native engine ───────────────────────────────────────────────
if [[ $NO_BUILD -eq 0 ]]; then
    info "Building native VGRE engine..."
    mkdir -p "$REPO_DIR/build"
    cmake -S "$REPO_DIR" -B "$REPO_DIR/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -Wno-dev \
        2>&1 | tail -5
    cmake --build "$REPO_DIR/build" -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)" 2>&1 | tail -5
    ok "Native engine built."
else
    info "Skipping build (--no-build)."
fi

BUILD_DIR="$REPO_DIR/build"

# Detect the native shared library
LIB_NAME="libvgre.so"
if [[ "$(uname)" == "Darwin" ]]; then LIB_NAME="libvgre.dylib"; fi

LIB_SRC=""
for candidate in \
    "$BUILD_DIR/$LIB_NAME" \
    "$BUILD_DIR/lib/$LIB_NAME" \
    "$BUILD_DIR/src/api/$LIB_NAME"; do
    if [[ -f "$candidate" ]]; then LIB_SRC="$candidate"; break; fi
done

# Also check for libvgre_cudart (the canonical loadable library)
CUDART_NAME="libvgre_cudart.so"
if [[ "$(uname)" == "Darwin" ]]; then CUDART_NAME="libvgre_cudart.dylib"; fi
CUDART_SRC=""
for candidate in \
    "$BUILD_DIR/$CUDART_NAME" \
    "$BUILD_DIR/lib/$CUDART_NAME"; do
    if [[ -f "$candidate" ]]; then CUDART_SRC="$candidate"; break; fi
done

[[ -z "$LIB_SRC" && -z "$CUDART_SRC" ]] && \
    warn "Could not find libvgre.so — dashboard may fail to load FFI. Run cmake --build first."

# ── Step 2: Build Flutter dashboard ──────────────────────────────────────────
# Platform/arch-aware: flutter targets differ (linux vs macos) and the Linux
# bundle directory is per-arch (x64 vs arm64).
FLUTTER_TARGET="linux"
[[ "$OS" == "Darwin" ]] && FLUTTER_TARGET="macos"
FLUTTER_ARCH="x64"
[[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]] && FLUTTER_ARCH="arm64"
if [[ $NO_BUILD -eq 0 ]]; then
    if command -v flutter >/dev/null 2>&1; then
        if [[ "$OS" == "Darwin" ]] && ! xcrun xcodebuild -version >/dev/null 2>&1; then
            warn "Skipping dashboard build: full Xcode is required on macOS."
        else
            info "Building Flutter dashboard ($FLUTTER_TARGET/$FLUTTER_ARCH)..."
            cd "$REPO_DIR/vgre_dashboard"
            flutter pub get --quiet
            [[ "$OS" == "Darwin" && ! -d macos ]] && flutter create --platforms=macos .
            flutter build "$FLUTTER_TARGET" --release 2>&1 | tail -5
            ok "Flutter dashboard built."
            cd "$REPO_DIR"
        fi
    else
        warn "flutter not found — skipping dashboard build. Install from https://flutter.dev/docs/get-started/install"
    fi
fi

# ── Step 3: Set up ~/.vgre directory ─────────────────────────────────────────
info "Setting up ~/.vgre config directory..."
mkdir -p "$VGRE_DIR"
chmod 700 "$VGRE_DIR"

# ── Step 4: Auth token — delegate to vgre-token.sh if already linked, ────────
#            or generate directly on first install.
if [[ ! -f "$TOKEN_FILE" ]]; then
    info "No auth token found — generating one automatically..."
    if command -v openssl >/dev/null 2>&1; then
        TOKEN=$(openssl rand -hex 32)
    else
        TOKEN=$(od -An -tx1 /dev/urandom | tr -d ' \n' | head -c 64)
    fi
    mkdir -p "$VGRE_DIR"
    chmod 700 "$VGRE_DIR"
    printf '%s' "$TOKEN" > "$TOKEN_FILE"
    chmod 600 "$TOKEN_FILE"
    ok "Auth token saved to $TOKEN_FILE"
    warn "Share with workers:  vgre-token copy  (after install completes)"
else
    ok "Auth token already exists (run  vgre-token fingerprint  to verify)"
fi

# ── Step 5: Write ~/.vgre/env with ALL environment variables ─────────────────
info "Writing environment configuration to $ENV_FILE..."

# Determine final lib path (prefer installed, fall back to build)
if [[ $DEV_MODE -eq 1 ]]; then
    VGRE_LIB_DIR="$BUILD_DIR"
else
    VGRE_LIB_DIR="$INSTALL_DIR/lib"
fi

cat > "$ENV_FILE" <<ENVEOF
# VGRE Runtime Environment — auto-generated by install_local.sh
# Source this file in your shell profile: source "$ENV_FILE"
# Or use the launch wrappers which source it automatically.

# ── Library path ─────────────────────────────────────────────────────────────
export VGRE_INSTALL_DIR="$INSTALL_DIR"
export VGRE_LIB_PATH="$VGRE_LIB_DIR/$LIB_NAME"
export LD_LIBRARY_PATH="$VGRE_LIB_DIR:\${LD_LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="$VGRE_LIB_DIR:\${DYLD_LIBRARY_PATH:-}"
export PATH="$BIN_DIR:\${PATH}"

# ── Auth token ────────────────────────────────────────────────────────────────
export VGRE_TCP_AUTH_TOKEN_FILE="$TOKEN_FILE"

# ── Cluster / networking ──────────────────────────────────────────────────────
export VGRE_PORT="7777"
# Uncomment to connect worker to a specific master:
# export VGRE_CLUSTER_NODES="MASTER_IP:7777"

# ── Logging ───────────────────────────────────────────────────────────────────
export VGRE_LOG_LEVEL="INFO"
# DEBUG | INFO | WARN | ERROR

# ── GPU cache model ───────────────────────────────────────────────────────────
export VGRE_L1_CACHE_KB="32"
export VGRE_L2_CACHE_MB="6"

# ── MPS multi-process service ─────────────────────────────────────────────────
# Uncomment to enable the VGRE MPS daemon (multi-process GPU sharing):
# export VGRE_MPS_PIPE="/tmp/vgre_mps.sock"

# ── RDMA transport ────────────────────────────────────────────────────────────
# Uncomment when libibverbs is available and RDMA NICs are present:
# export VGRE_RDMA_DEVICE="mlx5_0"

# ── gRPC cluster transport ────────────────────────────────────────────────────
# Uncomment to expose a gRPC endpoint (requires -DVGRE_ENABLE_GRPC=ON build):
# export VGRE_GRPC_PORT="50051"
ENVEOF
chmod 600 "$ENV_FILE"
ok "Environment file written to $ENV_FILE"

# ── Step 6: Add to shell profile (idempotent; skipped with --skip-profile) ────
SOURCE_LINE="# VGRE environment"$'\n'"[[ -f \"$ENV_FILE\" ]] && source \"$ENV_FILE\""
PROFILE_UPDATED=$SKIP_PROFILE
[[ $SKIP_PROFILE -eq 1 ]] && info "Skipping shell-profile update (--skip-profile)."
for PROFILE in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.zprofile" "$HOME/.bash_profile" "$HOME/.profile"; do
    [[ $SKIP_PROFILE -eq 1 ]] && break
    if [[ -f "$PROFILE" ]] && ! grep -q "vgre/env" "$PROFILE" 2>/dev/null; then
        printf '\n%s\n' "$SOURCE_LINE" >> "$PROFILE"
        ok "Added env sourcing to $PROFILE"
        PROFILE_UPDATED=1
    fi
done
if [[ $PROFILE_UPDATED -eq 0 ]] && ! grep -qr "vgre/env" \
        "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.zprofile" "$HOME/.bash_profile" "$HOME/.profile" 2>/dev/null; then
    printf '\n%s\n' "$SOURCE_LINE" >> "$HOME/.profile"
    ok "Added env sourcing to ~/.profile"
fi

# Apply to the current shell session immediately
# shellcheck disable=SC1090
source "$ENV_FILE"

# ── Step 7: Install artifacts ─────────────────────────────────────────────────
if [[ $DEV_MODE -eq 0 ]]; then
    info "Installing to $INSTALL_DIR..."
    mkdir -p "$INSTALL_DIR/lib" "$INSTALL_DIR/bin" "$DESKTOP_DIR" "$BIN_DIR"

    # Copy native shared libraries (.so on Linux, .dylib on macOS; -P keeps
    # version symlinks like libvgre.so.0 intact).
    for lib in \
        "$BUILD_DIR"/*.so* \
        "$BUILD_DIR"/*.dylib \
        "$BUILD_DIR"/lib/*.so* \
        "$BUILD_DIR"/lib/*.dylib; do
        [[ -e "$lib" ]] && cp -P "$lib" "$INSTALL_DIR/lib/" && true
    done
    # Copy worker binary. Canonical install location is $INSTALL_DIR/vgre-worker
    # (what vgre-start.sh and vgre_sync.sh use); the CMake output path is
    # build/src/advanced/vgre-worker.
    for bin in \
        "$BUILD_DIR/src/advanced/vgre-worker" \
        "$BUILD_DIR/vgre-worker" \
        "$BUILD_DIR/bin/vgre-worker"; do
        if [[ -x "$bin" ]]; then
            cp "$bin" "$INSTALL_DIR/vgre-worker.new"
            mv -f "$INSTALL_DIR/vgre-worker.new" "$INSTALL_DIR/vgre-worker"  # atomic; avoids ETXTBSY
            break
        fi
    done
    [[ -x "$INSTALL_DIR/vgre-worker" ]] || \
        warn "vgre-worker binary not found in $BUILD_DIR — cluster worker mode unavailable."

    # Copy Flutter bundle (per-platform layout).
    if [[ "$OS" == "Darwin" ]]; then
        FLUTTER_BUNDLE="$REPO_DIR/vgre_dashboard/build/macos/Build/Products/Release/vgre_dashboard.app"
        if [[ -d "$FLUTTER_BUNDLE" ]]; then
            cp -R "$FLUTTER_BUNDLE" "$INSTALL_DIR/"
            ok "Flutter dashboard (.app) installed."
        else
            warn "Flutter bundle not found — dashboard install skipped."
        fi
    else
        FLUTTER_BUNDLE="$REPO_DIR/vgre_dashboard/build/linux/$FLUTTER_ARCH/release/bundle"
        if [[ -d "$FLUTTER_BUNDLE" ]]; then
            cp -r "$FLUTTER_BUNDLE"/. "$INSTALL_DIR/"
            ok "Flutter dashboard installed."
        else
            warn "Flutter bundle not found at $FLUTTER_BUNDLE — dashboard install skipped."
        fi
    fi

    # Create launch wrapper that sources env and starts the dashboard.
    # All paths are expanded NOW (install time) so the wrapper never depends on
    # variables that don't exist at runtime. On Linux, jemalloc is preloaded:
    # glibc malloc asserts under concurrent Flutter(libc++)/VGRE(libstdc++)
    # per-thread arena contention.
    if [[ "$OS" == "Darwin" ]]; then
        DASH_EXEC="$INSTALL_DIR/vgre_dashboard.app/Contents/MacOS/vgre_dashboard"
        cat > "$INSTALL_DIR/vgre-launch.sh" <<LAUNCHEOF
#!/usr/bin/env bash
# Auto-generated VGRE launch wrapper — sources all runtime env vars.
[[ -f "$ENV_FILE" ]] && source "$ENV_FILE"
export DYLD_LIBRARY_PATH="$INSTALL_DIR/lib:\${DYLD_LIBRARY_PATH:-}"
export VGRE_LIB_PATH="$INSTALL_DIR/lib/$LIB_NAME"
cd "$INSTALL_DIR"
exec "$DASH_EXEC" "\$@"
LAUNCHEOF
    else
        cat > "$INSTALL_DIR/vgre-launch.sh" <<LAUNCHEOF
#!/usr/bin/env bash
# Auto-generated VGRE launch wrapper — sources all runtime env vars.
[[ -f "$ENV_FILE" ]] && source "$ENV_FILE"
export LD_LIBRARY_PATH="$INSTALL_DIR/lib:\${LD_LIBRARY_PATH:-}"
export VGRE_LIB_PATH="$INSTALL_DIR/lib/$LIB_NAME"
cd "$INSTALL_DIR"

# jemalloc preload (see comment in install_local.sh); warn-and-continue if absent.
JEMALLOC_LIB=""
for candidate in \\
    /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \\
    /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 \\
    /usr/lib64/libjemalloc.so.2 \\
    /usr/lib/libjemalloc.so.2 \\
    /usr/local/lib/libjemalloc.so.2; do
    [[ -f "\$candidate" ]] && JEMALLOC_LIB="\$candidate" && break
done
if [[ -n "\$JEMALLOC_LIB" ]]; then
    export LD_PRELOAD="\$JEMALLOC_LIB\${LD_PRELOAD:+:\$LD_PRELOAD}"
else
    echo "[VGRE] Warning: jemalloc not found; glibc malloc may assert under load." >&2
    echo "[VGRE] Install with: sudo apt-get install libjemalloc2 | sudo dnf install jemalloc" >&2
fi

exec "$INSTALL_DIR/vgre_dashboard" "\$@"
LAUNCHEOF
    fi
    chmod +x "$INSTALL_DIR/vgre-launch.sh"
    ln -sf "$INSTALL_DIR/vgre-launch.sh" "$BIN_DIR/vgre-dashboard" 2>/dev/null || true

    # Symlink worker binary into PATH
    [[ -x "$INSTALL_DIR/vgre-worker" ]] && \
        ln -sf "$INSTALL_DIR/vgre-worker" "$BIN_DIR/vgre-worker" 2>/dev/null || true

    # Desktop shortcut
    cat > "$DESKTOP_DIR/vgre-dashboard.desktop" <<DESKEOF
[Desktop Entry]
Version=1.0
Type=Application
Name=VGRE Dashboard
Comment=Virtual GPU Runtime Engine — real-time performance monitor
Exec=$INSTALL_DIR/vgre-launch.sh
Icon=utilities-system-monitor
Terminal=false
Categories=System;Utility;Development;
DESKEOF
    update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true

    ok "Installed to $INSTALL_DIR"
else
    # Dev mode: create a thin launch wrapper pointing at the build tree
    mkdir -p "$BIN_DIR"
    DEV_DEVICE="linux"
    [[ "$OS" == "Darwin" ]] && DEV_DEVICE="macos"
    cat > "$BIN_DIR/vgre-dashboard" <<DEVLAUNCHEOF
#!/usr/bin/env bash
[[ -f "$ENV_FILE" ]] && source "$ENV_FILE"
export LD_LIBRARY_PATH="$BUILD_DIR:\${LD_LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="$BUILD_DIR:\${DYLD_LIBRARY_PATH:-}"
export VGRE_LIB_PATH="$BUILD_DIR/$LIB_NAME"
cd "$REPO_DIR/vgre_dashboard"
exec flutter run -d $DEV_DEVICE "\$@"
DEVLAUNCHEOF
    chmod +x "$BIN_DIR/vgre-dashboard"
    ok "Dev-mode launcher written to $BIN_DIR/vgre-dashboard"
fi

# ── Link CLI commands into PATH ───────────────────────────────────────────────
mkdir -p "$BIN_DIR"
if [[ -f "$REPO_DIR/scripts/vgre-start.sh" ]]; then
    chmod +x "$REPO_DIR/scripts/vgre-start.sh"
    ln -sf "$REPO_DIR/scripts/vgre-start.sh" "$BIN_DIR/vgre-start" 2>/dev/null || true
    ok "vgre-start  → $BIN_DIR/vgre-start"
fi
if [[ -f "$REPO_DIR/scripts/vgre-token.sh" ]]; then
    chmod +x "$REPO_DIR/scripts/vgre-token.sh"
    ln -sf "$REPO_DIR/scripts/vgre-token.sh" "$BIN_DIR/vgre-token" 2>/dev/null || true
    ok "vgre-token  → $BIN_DIR/vgre-token"
fi
if [[ -f "$REPO_DIR/scripts/vgre-discover.sh" ]]; then
    chmod +x "$REPO_DIR/scripts/vgre-discover.sh"
    ln -sf "$REPO_DIR/scripts/vgre-discover.sh" "$BIN_DIR/vgre-discover" 2>/dev/null || true
    ok "vgre-discover → $BIN_DIR/vgre-discover"
fi

# Ensure ~/.local/bin is on PATH in shell profiles and ~/.vgre/env
# shellcheck source=scripts/vgre-cli-install.sh
. "$REPO_DIR/scripts/vgre-cli-install.sh"
vgre_ensure_cli_path

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════════╗"
echo "║   VGRE installed successfully!                           ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║                                                          ║"
echo "║  Environment is active NOW (sourced into this shell).    ║"
echo "║  New terminals will load it automatically from:          ║"
echo "║    $ENV_FILE"
echo "║                                                          ║"
echo "║  Quick start:                                            ║"
echo "║    vgre-dashboard               launch the dashboard     ║"
echo "║    vgre-start --master          start master node        ║"
echo "║    vgre-start --worker          start worker node        ║"
echo "║    vgre-start --test            local self-test          ║"
echo "║                                                          ║"
echo "║  Token management (run from any terminal):               ║"
echo "║    vgre-token generate          create new token         ║"
echo "║    vgre-token fingerprint       show SHA-256 fingerprint ║"
echo "║    vgre-token set <TOKEN>       paste token from master  ║"
echo "║    vgre-token copy              show scp copy command    ║"
echo "║    vgre-token verify            check master/worker match║"
echo "║                                                          ║"
echo "║  Current token fingerprint (share with workers):         ║"
if command -v sha256sum >/dev/null 2>&1; then
    FP=$(sha256sum "$TOKEN_FILE" | awk '{print $1}')
elif command -v shasum >/dev/null 2>&1; then
    FP=$(shasum -a 256 "$TOKEN_FILE" | awk '{print $1}')
else
    FP="(sha256sum unavailable)"
fi
echo "║    ${FP:0:32}..."
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${RESET}"
