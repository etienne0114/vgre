# VGRE User Guide

**Version 1.2.0** — Virtual GPU Runtime Engine

VGRE lets you run unmodified CUDA applications on any x86-64 or ARM64 CPU by intercepting the CUDA runtime at load time. No GPU required. Includes a real-time Flutter dashboard, distributed cluster support, and a full token management CLI.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [System Requirements](#2-system-requirements)
3. [Installation](#3-installation)
4. [Running CUDA Applications](#4-running-cuda-applications)
5. [Environment Variables Reference](#5-environment-variables-reference)
6. [Token Management (`vgre-token`)](#6-token-management-vgre-token)
7. [Cluster Setup (`vgre-start`)](#7-cluster-setup-vgre-start)
8. [Dashboard](#8-dashboard)
9. [Advanced Features](#9-advanced-features)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Quick Start

### Linux / macOS — one command

```bash
git clone https://github.com/vgre-org/vgre-runtime.git
cd vgre-runtime
bash install_local.sh
```

`install_local.sh` automatically:
- Detects and installs missing dependencies (cmake, LLVM, OpenMP, Flutter, …)
- Builds the native engine and Flutter dashboard
- Writes `~/.vgre/env` with all environment variables
- Adds `source ~/.vgre/env` to your shell profile (bashrc / zshrc / profile)
- Creates the auth token at `~/.vgre/token`
- Links `vgre-dashboard`, `vgre-start`, and `vgre-token` into `~/.local/bin`

After install, open a **new terminal** and run:

```bash
vgre-dashboard          # launch the real-time monitor
vgre-start --test       # local master + worker self-test
```

### Windows — one command

```powershell
git clone https://github.com/vgre-org/vgre-runtime.git
cd vgre-runtime
.\scripts\vgre_sync.bat
```

`vgre_sync.bat` automatically:
- Detects and installs cmake / LLVM / Visual Studio Build Tools / Flutter via **winget** (falls back to **chocolatey**)
- Builds native engine and dashboard
- Creates a Desktop shortcut and launcher
- Installs `vgre-token.bat` into `%LOCALAPPDATA%\VGRE\scripts` and adds it to PATH

Open a **new terminal** and run:

```powershell
vgre-token generate     # create auth token
vgre-dashboard          # launch dashboard (or double-click Desktop shortcut)
```

---

## 2. System Requirements

### All Platforms

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Linux 5.4 / Win 10 / macOS 11 | Ubuntu 22.04 / Win 11 / macOS 14 |
| CPU | x86-64, 2 cores | x86-64 with AVX-512 or ARM64, 8+ cores |
| RAM | 4 GB | 16 GB+ |
| CMake | 3.18 | 3.25+ |
| LLVM / Clang | 16 | 18 |
| C++ standard | C++17 | C++20 |
| Flutter SDK | 3.16 | 3.24+ |

### Optional — unlocks additional features

| Package | Feature unlocked |
|---------|-----------------|
| `libomp-dev` | Multi-threaded kernel execution (required for OpenMP) |
| `libssl-dev` | Secure cluster transport (TLS) |
| `libtpm2-tss-dev` | Hardware TPM token storage (Linux) |
| `libibverbs-dev` | RDMA/RoCE zero-copy transport (`-DVGRE_ENABLE_RDMA=ON`) |
| `libgrpc++-dev` | gRPC cluster transport (`-DVGRE_ENABLE_GRPC=ON`) |
| Intel AMX CPU | AMX tile acceleration (auto-detected at build time) |

---

## 3. Installation

### 3.1 Linux — step by step

```bash
# Install build tools (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y \
    cmake build-essential git curl \
    llvm-18 clang-18 libclang-dev \
    libomp-dev libssl-dev

# Install Flutter (choose one)
sudo snap install flutter --classic          # Ubuntu with snapd
# OR: brew install --cask flutter           # via Homebrew
# OR: install_local.sh auto-installs it

# Build + install
bash install_local.sh
```

Verify the install:

```bash
source ~/.vgre/env        # load env in current shell (new terminals load it automatically)
vgre-dashboard --version  # prints: VGRE x.y.z
vgre-token fingerprint    # prints: SHA-256 of your token
```

### 3.2 macOS — step by step

```bash
# Install Homebrew if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake llvm libomp openssl git
brew install --cask flutter

# Build + install
bash install_local.sh
```

> **Apple Silicon (M1/M2/M3):** VGRE builds natively for ARM64. AMX instructions
> on Apple Silicon use a different interface and are not yet supported; the AVX-512
> path is disabled. Performance comes from OpenMP parallelism.

### 3.3 Windows — step by step

1. Open **PowerShell** (no admin needed for user-scope install):

```powershell
# Auto-installs all dependencies via winget:
.\scripts\vgre_sync.bat
```

2. If winget is not available, install dependencies manually first:
   - [CMake](https://cmake.org/download/) — tick "Add to PATH"
   - [LLVM 18](https://github.com/llvm/llvm-project/releases/tag/llvmorg-18.1.8)
   - [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) with "Desktop development with C++"
   - [Flutter SDK](https://flutter.dev/docs/get-started/install/windows)

3. Then re-run `vgre_sync.bat`.

### 3.4 Build options (cmake flags)

| Flag | Default | Effect |
|------|---------|--------|
| `-DCMAKE_BUILD_TYPE=Release` | Release | Optimized build |
| `-DVGRE_ENABLE_NATIVE_SIMD=ON` | OFF | Enable `-march=native` (max SIMD, not portable) |
| `-DVGRE_ENABLE_RDMA=ON` | OFF | Enable RDMA/RoCE transport (requires `libibverbs-dev`) |
| `-DVGRE_ENABLE_GRPC=ON` | OFF | Enable gRPC cluster transport (requires `libgrpc++-dev`) |

```bash
# Example: Release + RDMA
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DVGRE_ENABLE_RDMA=ON
cmake --build build -j$(nproc)
```

### 3.5 Dev-mode install (no copy to system)

```bash
bash install_local.sh --dev   # runs from repo, wraps flutter run -d linux
bash install_local.sh --check # only verify deps, no build
```

---

## 4. Running CUDA Applications

### 4.1 Intercept mode (LD_PRELOAD)

Redirect an existing CUDA application to VGRE without recompiling it:

```bash
# Linux
export LD_PRELOAD="$HOME/.local/share/VGRE/lib/libvgre_cudart.so"
python my_pytorch_script.py

# macOS
export DYLD_INSERT_LIBRARIES="$HOME/.local/share/VGRE/lib/libvgre_cudart.dylib"
python my_pytorch_script.py
```

> **Windows:** Copy `%LOCALAPPDATA%\VGRE\vgre_cudart.dll` into the application
> directory and rename it `cudart64_120.dll` (match your CUDA version suffix).

### 4.2 Python bindings

```python
import vgre
import numpy as np

rt = vgre.Runtime()
rt.init(enable_profiling=True)

# Allocate GPU memory
a = rt.alloc(1024 * np.float32().itemsize)
b = rt.alloc(1024 * np.float32().itemsize)

# Upload data
data = np.ones(1024, dtype=np.float32)
rt.memcpy_h2d(a, data)

# Launch a kernel
kernel = vgre.Kernel("vector_scale", """
    __global__ void vector_scale(float* a, float scale, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) a[i] *= scale;
    }
""")
rt.launch(kernel, grid=(4, 1, 1), block=(256, 1, 1), args=[a, 2.0, 1024])
rt.synchronize()

result = rt.memcpy_d2h(a, 1024 * 4)
print(np.frombuffer(result, dtype=np.float32)[:8])
```

### 4.3 Using the C API

```c
#include <vgre/api/vgre_c_api.h>

int main(void) {
    vgre_init();

    void* ptr = NULL;
    vgre_malloc(&ptr, 1024 * sizeof(float));

    // ... launch kernels ...

    vgre_free(ptr);
    vgre_shutdown();
}
```

---

## 5. Environment Variables Reference

All variables are written to `~/.vgre/env` by `install_local.sh` and loaded automatically.  
On Windows they are set in User scope by `Setup-VGRECluster.ps1` / `vgre_sync.bat`.

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `VGRE_LIB_PATH` | auto-detected | Absolute path to `libvgre.so` / `vgre.dll` |
| `LD_LIBRARY_PATH` | extended | Directory containing VGRE shared libraries |
| `VGRE_LOG_LEVEL` | `INFO` | Verbosity: `DEBUG` \| `INFO` \| `WARN` \| `ERROR` |
| `VGRE_INSTALL_DIR` | `~/.local/share/VGRE` | Installation directory |

### Cluster / networking

| Variable | Default | Description |
|----------|---------|-------------|
| `VGRE_PORT` | `7777` | TCP port for master and worker nodes |
| `VGRE_CLUSTER_NODES` | _(empty)_ | `IP:PORT` of master to connect to (worker only) |
| `VGRE_TCP_AUTH_TOKEN_FILE` | `~/.vgre/token` | Path to the shared cluster auth token file |
| `VGRE_TCP_AUTH_TOKEN` | _(empty)_ | Raw token string (less secure; avoid in production) |

### GPU cache model

| Variable | Default | Description |
|----------|---------|-------------|
| `VGRE_L1_CACHE_KB` | `32` | Per-block L1 cache size in KB (16 \| 32 \| 64 \| 128) |
| `VGRE_L2_CACHE_MB` | `6` | Per-device L2 cache size in MB (2 \| 6 \| 20 \| 40) |

### Optional features

| Variable | Default | Description |
|----------|---------|-------------|
| `VGRE_MPS_PIPE` | _(empty)_ | Unix socket path to enable CUDA MPS daemon (e.g. `/tmp/vgre_mps.sock`) |
| `VGRE_RDMA_DEVICE` | _(empty)_ | RDMA device name (e.g. `mlx5_0`). Requires `-DVGRE_ENABLE_RDMA=ON` build |
| `VGRE_GRPC_PORT` | _(empty)_ | Port for gRPC cluster transport. Requires `-DVGRE_ENABLE_GRPC=ON` build |

### Performance tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `VGRE_ENABLE_NUMA` | `0` | Enable NUMA-aware thread scheduling (Linux) |
| `VGRE_WORKER_THREADS` | auto (nproc) | Override worker thread count |
| `VGRE_SIMD_LEVEL` | auto-detected | Force SIMD level: `SSE4` \| `AVX` \| `AVX2` \| `AVX512` |

---

## 6. Token Management (`vgre-token`)

`vgre-token` is a standalone CLI that works from **any directory** after install. It manages the shared auth token that secures cluster communication between master and worker nodes.

### Commands

```
vgre-token generate       — generate a new secure 256-bit token
vgre-token show           — print the stored token value
vgre-token fingerprint    — print the SHA-256 fingerprint
vgre-token set <TOKEN>    — store a token pasted from another machine
vgre-token verify         — check that the env-var matches the stored file
vgre-token copy           — print the scp / robocopy command for workers
vgre-token revoke         — delete the stored token (with confirmation)
```

### Typical workflow

```bash
# ── On the MASTER machine ─────────────────────────────────────────────────
vgre-token generate
# Output:
#   ✓ Token saved to ~/.vgre/token
#   Token fingerprint (SHA-256):
#     a3f9bcd2... (64 chars)

# Share the token with workers (choose one):
vgre-token copy               # prints the scp command
# OR manually:
scp ~/.vgre/token worker@192.168.1.50:~/.vgre/token

# ── On every WORKER machine ───────────────────────────────────────────────
# Option A: scp the file (see above)
# Option B: paste the token value
vgre-token set a3f9bcd2xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# ── Verify both machines match ────────────────────────────────────────────
# Run on BOTH master and worker — output must be identical:
vgre-token fingerprint
```

### Token storage

| Platform | Token file | Env var set in |
|----------|-----------|----------------|
| Linux/macOS | `~/.vgre/token` (chmod 600) | `~/.vgre/env` → shell profile |
| Windows | `%USERPROFILE%\.vgre\token` | User-scope env (no restart needed) |

---

## 7. Cluster Setup (`vgre-start`)

`vgre-start` launches master or worker nodes. It automatically sources `~/.vgre/env` before starting, so no manual environment setup is needed.

### Commands

```bash
vgre-start --master                        # start master node (launches dashboard)
vgre-start --worker                        # start worker (auto-discovers master on LAN)
vgre-start --worker --master-ip 10.0.1.5  # connect to specific master IP
vgre-start --worker --port 7778           # use custom port
vgre-start --test                          # local self-test (master + worker, same machine)
```

### Complete cluster setup (4 steps)

**Step 1 — Install on every machine**

```bash
bash install_local.sh   # Linux/macOS
vgre_sync.bat           # Windows
```

**Step 2 — Generate token on master**

```bash
vgre-token generate
```

**Step 3 — Copy token to workers**

```bash
# Method A: scp
vgre-token copy         # shows the exact command
scp ~/.vgre/token worker@192.168.1.50:~/.vgre/token

# Method B: paste
vgre-token show         # copy the token value
# On worker:
vgre-token set <paste-token-here>
```

**Step 4 — Start**

```bash
# On master:
vgre-start --master

# On each worker:
vgre-start --worker
# or, for a different subnet:
vgre-start --worker --master-ip 192.168.1.10
```

### Verify the cluster is working

```bash
# On any node — fingerprints must match:
vgre-token fingerprint
```

The dashboard automatically shows discovered cluster nodes in the **Cluster Topology** tab.

### Cross-platform cluster

| Master | Worker | Supported |
|--------|--------|-----------|
| Linux | Linux | ✓ Full |
| Linux | macOS | ✓ Full |
| Linux | Windows | ✓ Full |
| macOS | Windows | ✓ Full |
| Windows | Linux | ✓ Full |

### Cross-Platform Cluster Setup Guide

VGRE supports heterogeneous clusters mixing Windows, Linux, and macOS nodes. This section provides detailed setup instructions and troubleshooting for cross-platform deployments.

#### Prerequisites for Cross-Platform Clusters

1. **Network Connectivity**: All nodes must be on the same network or have direct IP connectivity
2. **Firewall Configuration**: 
   - TCP port 7777 (default cluster port) must be open between all nodes
   - UDP port 7778 (default discovery port) must be open for auto-discovery
3. **Time Synchronization**: Ensure all nodes have synchronized clocks (NTP recommended)
4. **Same Auth Token**: All nodes must use the same `VGRE_TCP_AUTH_TOKEN`

#### Step-by-Step Cross-Platform Setup

**Step 1: Install VGRE on all platforms**

```bash
# Linux/macOS
bash install_local.sh

# Windows (PowerShell as Administrator)
.\scripts\vgre_sync.bat
```

**Step 2: Configure firewall rules**

```bash
# Linux (Ubuntu/Debian)
sudo ufw allow 7777/tcp
sudo ufw allow 7778/udp

# Linux (CentOS/RHEL)
sudo firewall-cmd --permanent --add-port=7777/tcp
sudo firewall-cmd --permanent --add-port=7778/udp
sudo firewall-cmd --reload

# macOS
# System Preferences > Security & Privacy > Firewall > Options
# Allow incoming connections for VGRE applications

# Windows (PowerShell as Administrator)
New-NetFirewallRule -DisplayName "VGRE TCP Cluster" -Direction Inbound -Protocol TCP -LocalPort 7777 -Action Allow
New-NetFirewallRule -DisplayName "VGRE UDP Discovery" -Direction Inbound -Protocol UDP -LocalPort 7778 -Action Allow
```

**Step 3: Generate and distribute auth token**

```bash
# On master node (any platform):
vgre-token generate

# Copy token to all worker nodes:
vgre-token show    # Copy the displayed token

# On each worker node:
vgre-token set <paste-token-here>

# Verify all nodes have matching tokens:
vgre-token fingerprint
```

**Step 4: Start cluster services**

```bash
# On master node:
vgre-start --master

# On worker nodes (auto-discovery):
vgre-start --worker

# On worker nodes (manual IP if auto-discovery fails):
vgre-start --worker --master-ip <master-ip-address>
```

#### Cross-Platform Environment Variables

Set these environment variables for optimal cross-platform compatibility:

```bash
# Recommended for all platforms
export VGRE_TCP_AUTH_TOKEN="your-shared-token-here"
export VGRE_CLUSTER_PORT=7777
export VGRE_CLUSTER_UDP_ANNOUNCE_PORT=7778

# Windows-specific (PowerShell)
$env:VGRE_TCP_AUTH_TOKEN="your-shared-token-here"
$env:VGRE_CLUSTER_PORT=7777
$env:VGRE_CLUSTER_UDP_ANNOUNCE_PORT=7778

# Optional: Restrict master discovery to specific IPs
export VGRE_CLUSTER_MASTER_IP="192.168.1.10,192.168.1.11"

# Optional: Increase timeouts for slower networks
export VGRE_CLUSTER_HANDSHAKE_TIMEOUT_SEC=10
export VGRE_CLUSTER_PEEK_TIMEOUT_MS=15000
```

#### Troubleshooting Cross-Platform Issues

**Problem: Windows worker crashes with exit code -1073741819 (0xC0000005)**

This indicates a memory access violation during cross-platform connection.

*Solution:*
1. Ensure both master and worker have the same auth token:
   ```bash
   vgre-token fingerprint  # Run on both nodes, compare output
   ```

2. Check Windows Defender/antivirus isn't blocking VGRE:
   ```powershell
   # Add VGRE to Windows Defender exclusions
   Add-MpPreference -ExclusionPath "C:\Users\%USERNAME%\.vgre"
   Add-MpPreference -ExclusionProcess "vgre-start.exe"
   ```

3. Verify network connectivity:
   ```bash
   # From worker, test TCP connection to master
   telnet <master-ip> 7777
   
   # Test UDP discovery
   nc -u <master-ip> 7778
   ```

4. Enable debug logging:
   ```bash
   export VGRE_LOG_LEVEL=DEBUG
   vgre-start --worker --master-ip <master-ip>
   ```

**Problem: UDP auto-discovery not working across platforms**

*Solution:*
1. Check firewall rules allow UDP port 7778
2. Verify nodes are on same subnet or configure manual IP:
   ```bash
   vgre-start --worker --master-ip <master-ip>
   ```
3. Check for network segmentation/VLANs blocking broadcast packets

**Problem: Security handshake fails between platforms**

*Solution:*
1. Verify identical auth tokens:
   ```bash
   vgre-token show  # Compare exact token values
   ```
2. Check system clocks are synchronized (±30 seconds)
3. Disable strict auth mode temporarily for testing:
   ```bash
   export VGRE_ALLOW_AUTH_FALLBACK=1
   vgre-start --worker
   ```

**Problem: Connection established but worker shows as inactive**

*Solution:*
1. Check platform-specific socket buffer sizes:
   ```bash
   # Linux: Increase socket buffers
   echo 'net.core.rmem_max = 16777216' | sudo tee -a /etc/sysctl.conf
   echo 'net.core.wmem_max = 16777216' | sudo tee -a /etc/sysctl.conf
   sudo sysctl -p
   ```

2. Verify CPU architecture compatibility:
   ```bash
   # Check SIMD instruction support matches
   vgre-dashboard  # View Hardware Tuning tab
   ```

**Problem: Performance degradation in cross-platform clusters**

*Solution:*
1. Enable platform-specific optimizations:
   ```bash
   # Linux/macOS: Enable native SIMD
   export VGRE_ENABLE_NATIVE_SIMD=ON
   
   # Windows: Use Windows-optimized threading
   set VGRE_WINDOWS_THREAD_POOL=1
   ```

2. Adjust cluster topology for network latency:
   ```bash
   # Prefer local workers for latency-sensitive kernels
   export VGRE_PREFER_LOCAL_WORKERS=1
   ```

#### Platform-Specific Notes

**Windows Nodes:**
- Run PowerShell as Administrator for initial setup
- Windows Defender may quarantine VGRE binaries - add exclusions
- Use `vgre-token.bat` instead of `vgre-token` command
- WSL2 nodes are supported but require additional network configuration

**macOS Nodes:**
- Allow VGRE through macOS Firewall in System Preferences
- Grant network permissions when prompted
- Apple Silicon (M1/M2) nodes provide excellent performance as workers

**Linux Nodes:**
- Most stable as master nodes
- Support all VGRE features including RDMA and gRPC transports
- Recommended for high-performance computing clusters

#### Monitoring Cross-Platform Clusters

Use the dashboard to monitor cross-platform cluster health:

1. **Cluster Topology Tab**: Shows all connected nodes with platform indicators
2. **Overview Tab**: Displays aggregate performance across all platforms  
3. **Hardware Tuning Tab**: Compare SIMD capabilities across different architectures

```bash
# Launch dashboard from any node
vgre-dashboard
```

The dashboard automatically detects platform differences and provides platform-specific optimization recommendations.

---

## 8. Dashboard

The Flutter dashboard provides real-time visibility into the VGRE runtime.

### Launch

```bash
vgre-dashboard          # Linux/macOS
# Windows: double-click "VGRE Dashboard" Desktop shortcut
```

### Dashboard tabs

| Tab | What it shows |
|-----|---------------|
| **Overview** | Compute utilization gauge, memory bandwidth, temperature, **L2 cache hit rate** (live from `vgre_get_cache_stats`) |
| **Kernel Explorer** | Top-N kernels by total time, per-kernel invocation history, PTX / IR source viewer |
| **Cluster Topology** | All connected nodes, latency heat-map, per-node credits/debits |
| **Hardware Tuning** | SIMD capabilities (SSE4 / AVX2 / AVX-512 / AMX), thread count controls, background compute toggle |
| **Memory Analysis** | UVM page map, pool allocator stats, active allocation list |
| **Settings** | Log level, profiler on/off, service mode, auth token display |

### Auto-connect to backend

The dashboard reads `~/.vgre/env` at startup and calls `setenv()` in the native
process so the C++ backend receives `VGRE_TCP_AUTH_TOKEN_FILE`, `VGRE_PORT`, and
all cache/RDMA variables without any manual configuration.

Library resolution order:
1. `VGRE_LIB_PATH` env var
2. `<bundle>/lib/libvgre.so` (installed bundle)
3. `~/.local/share/VGRE/lib/libvgre.so`
4. `build/libvgre.so` (repo build, developer mode)

---

## 9. Advanced Features

### 9.1 GPU L1/L2 cache model

VGRE emulates a software-managed GPU cache hierarchy (Ampere Ampere-class defaults):

- **L1**: 32 KB per block, 4-way set-associative, 128-byte lines, per-block (not shared)
- **L2**: 6 MB per device, 8-way set-associative, per-device singleton, mutex-protected

Size is configurable at runtime:

```bash
export VGRE_L1_CACHE_KB=64    # 16 | 32 | 64 | 128
export VGRE_L2_CACHE_MB=20    # 2 | 6 | 20 | 40
```

Cache statistics are exposed via the C API and displayed live in the dashboard:

```c
vgre_cache_stats_t cs;
vgre_get_cache_stats(&cs);
printf("L2 hit rate: %.1f%%  hits=%llu  misses=%llu\n",
       cs.l2_hit_rate * 100.0, cs.l2_hits, cs.l2_misses);
```

### 9.2 AMX + AVX-512 WMMA acceleration

VGRE auto-detects Intel AMX (Sapphire Rapids+) and AVX-512 at startup:

- **AMX path**: used for 16×16×16 BF16 matrix tiles (`mma_sync` + `wgmma`)
- **AVX-512 path**: used for N=16 tiles on any AVX-512 CPU (16× speedup vs scalar)
- **Scalar fallback**: always available

Check what's active:

```bash
VGRE_LOG_LEVEL=DEBUG vgre-dashboard 2>&1 | grep -E "AMX|AVX-512|SIMD"
```

### 9.3 Hopper PTX emulation

VGRE translates Hopper-generation PTX instructions to CPU code:

| PTX instruction | CPU emulation |
|----------------|---------------|
| `wgmma.mma_async.*` | Full M×N×K GEMM via `vgre_wgmma_*` (AVX-512 vectorized when N=256) |
| `cp.async.bulk.tensor.*` | Synchronous `memcpy` (no async staging needed on CPU) |
| `wgmma.fence` / `wgmma.wait_group` | `__atomic_thread_fence(SEQ_CST)` |
| `mma.sync.aligned.m16n8k16` | `vgre_mma_m16n8k16_f32_f16` scalar tile GEMM |

### 9.4 CUDA MPS (Multi-Process Sharing)

Enable VGRE MPS to let multiple processes share one VGRE context (mirrors NVIDIA MPS):

```bash
# Start MPS daemon (in background):
export VGRE_MPS_PIPE=/tmp/vgre_mps.sock

# Each client process automatically connects when VGRE_MPS_PIPE is set.
# The daemon serializes kernel launches from all clients.
```

### 9.5 RDMA / RoCE transport

Build with RDMA support and VGRE uses zero-copy transfers for large payloads (>64 KB):

```bash
cmake -S . -B build -DVGRE_ENABLE_RDMA=ON
# Requires: sudo apt-get install libibverbs-dev rdma-core

# Enable soft-RoCE loopback for testing (no physical RDMA NIC needed):
sudo rdma link add rxe0 type rxe netdev eth0
export VGRE_RDMA_DEVICE=rxe0
```

### 9.6 gRPC cluster transport

Expose a gRPC endpoint for Ray Serve / Horovod / DeepSpeed integration:

```bash
cmake -S . -B build -DVGRE_ENABLE_GRPC=ON
# Requires: sudo apt-get install libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc

export VGRE_GRPC_PORT=50051
vgre-start --master   # starts gRPC service alongside cluster TCP
```

Clients connect via standard gRPC:

```python
import grpc
# Generated stubs from proto/vgre_cluster.proto
channel = grpc.insecure_channel("master-ip:50051")
stub = vgre_cluster_pb2_grpc.VGREClusterStub(channel)
resp = stub.AllocMemory(MemAllocRequest(size_bytes=1024*1024))
```

---

## 10. Troubleshooting

### 10.1 Dashboard fails to load ("Failed to load native library")

```
Possible causes:
1. libvgre.so not found
2. LD_LIBRARY_PATH not set
3. Library missing dependencies (LLVM, OpenMP)
```

**Fix:**

```bash
# Reload environment
source ~/.vgre/env

# Check the library exists
ls -la ~/.local/share/VGRE/lib/libvgre.so

# Check its dependencies
ldd ~/.local/share/VGRE/lib/libvgre.so | grep "not found"

# If libomp.so is missing:
sudo apt-get install libomp-dev
```

### 10.2 Cluster worker cannot connect to master

```bash
# Verify both machines use the same token
vgre-token fingerprint      # run on BOTH — must be identical

# Test network reachability
nc -zv MASTER_IP 7777        # Linux/macOS
Test-NetConnection MASTER_IP -Port 7777  # Windows PowerShell

# Check firewall (Linux)
sudo ufw allow 7777/tcp

# Check firewall (Windows PowerShell — run as admin)
New-NetFirewallRule -DisplayName "VGRE" -Direction Inbound -Protocol TCP -LocalPort 7777 -Action Allow
```

### 10.3 Mismatched token fingerprints

```bash
# Re-copy the token from master to worker
vgre-token copy          # on master — prints scp command
scp ~/.vgre/token worker@WORKER_IP:~/.vgre/token

# Verify after copy
vgre-token fingerprint   # must match on both machines
```

### 10.4 CMake cannot find LLVM

```bash
# Linux: install and export
sudo apt-get install llvm-18 llvm-18-dev clang-18
export LLVM_DIR=$(llvm-config-18 --cmakedir)

# macOS
brew install llvm
export LLVM_DIR=$(brew --prefix llvm)/lib/cmake/llvm

# Windows
winget install LLVM.LLVM
set LLVM_DIR=C:\Program Files\LLVM\lib\cmake\llvm
```

### 10.5 Flutter build fails

```bash
# Update Flutter and clear caches
flutter upgrade
flutter clean
cd vgre_dashboard && flutter pub get

# Linux: ensure correct linker is available
sudo apt-get install clang lld

# macOS: install Xcode command-line tools
xcode-select --install
```

### 10.6 Low compute performance

```bash
# Check SIMD level being used
VGRE_LOG_LEVEL=DEBUG ./build/examples/matrix_multiply 2>&1 | grep "SIMD\|AVX\|AMX"

# Force maximum SIMD
export VGRE_SIMD_LEVEL=AVX512

# Enable native CPU tuning (rebuild required)
cmake -S . -B build -DVGRE_ENABLE_NATIVE_SIMD=ON
cmake --build build -j$(nproc)
```

### 10.7 Windows: error 0xC000001D / 1114 (DLL load failure)

This means a DLL dependency is missing or the wrong architecture is loaded.

```powershell
# Show missing dependencies
dumpbin /dependents %LOCALAPPDATA%\VGRE\vgre.dll

# Ensure all runtime DLLs are in place
Get-ChildItem "$env:LOCALAPPDATA\VGRE\lib" -Filter "*.dll" | Select-Object Name

# Re-run the sync script
.\scripts\vgre_sync.bat
```

### 10.8 OpenMP not working (all kernels run on 1 thread)

```bash
# Check OpenMP is linked
ldd ~/.local/share/VGRE/lib/libvgre.so | grep omp

# Install missing library
sudo apt-get install libomp-dev         # Ubuntu/Debian
sudo dnf install libomp-devel           # Fedora
brew install libomp                     # macOS

# Rebuild
cmake --build build -j$(nproc)
```

---

## Production Status

| Feature | Status |
|---------|--------|
| CUDA runtime intercept (cudart) | ✅ Production |
| LLVM JIT kernel compilation | ✅ Production |
| cuBLAS INT8 / FP32 / FP64 | ✅ Production |
| cuDNN convolution (INT8, FP32) | ✅ Production |
| NCCL AllReduce (ring + barrier-tree) | ✅ Production |
| CUDA Graphs (IF / WHILE / SWITCH) | ✅ Production |
| Virtual memory (cuMemCreate / cuMemMap) | ✅ Production |
| External semaphores (eventfd / Win32) | ✅ Production |
| UVM managed memory | ✅ Production |
| NVTX profiling markers | ✅ Production |
| AVX-512 WMMA acceleration | ✅ Production |
| Intel AMX tile acceleration | ✅ Production (Sapphire Rapids+) |
| Hopper PTX (wgmma, TMA, cp.async.bulk) | ✅ Production |
| GPU L1/L2 cache model | ✅ Production |
| TCP cluster + TLS | ✅ Production |
| RDMA / RoCE transport | ✅ Optional (`-DVGRE_ENABLE_RDMA=ON`) |
| gRPC cluster transport | ✅ Optional (`-DVGRE_ENABLE_GRPC=ON`) |
| CUDA MPS multi-process | ✅ Production (Unix socket) |
| cuMemMulticast | ✅ Production |
| Flutter real-time dashboard | ✅ Production |
| `vgre-token` CLI | ✅ Production (Linux/macOS/Windows) |
| `vgre-start` cluster launcher | ✅ Production (Linux/macOS/Windows) |
| OTel hw.gpu.* telemetry export | ✅ Production |
