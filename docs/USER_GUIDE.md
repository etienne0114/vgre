# VGRE User Guide

**Version 1.4.0** — Virtual GPU Runtime Engine

VGRE lets you run unmodified CUDA applications on any x86-64 or ARM64 CPU by intercepting the CUDA runtime at load time. No GPU required. Includes a real-time Flutter dashboard, distributed cluster support with **full WAN connectivity**, and a complete token-management + discovery CLI.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [System Requirements](#2-system-requirements)
3. [Installation](#3-installation)
4. [Running CUDA Applications](#4-running-cuda-applications)
5. [Environment Variables Reference](#5-environment-variables-reference)
6. [Token Management (`vgre-token`)](#6-token-management-vgre-token)
7. [Cluster Setup (`vgre-start`)](#7-cluster-setup-vgre-start)
8. [Public IP Discovery (`vgre-discover`)](#8-public-ip-discovery-vgre-discover)
9. [Dashboard](#9-dashboard)
10. [Advanced Features](#10-advanced-features)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Quick Start

### Automated Deployment (Linux / macOS)

```bash
git clone https://github.com/vgre-org/vgre-runtime.git
cd vgre-runtime
bash install_local.sh
```

`install_local.sh` automatically:
- Detects and installs missing dependencies (cmake, LLVM, OpenMP, Flutter, …)
- Builds the native engine and Flutter dashboard
- Writes `~/.vgre/env` with all environment variables
- Adds `source ~/.vgre/env` to your shell profile
- Creates the auth token at `~/.vgre/token`
- Links `vgre-dashboard`, `vgre-start`, `vgre-token`, and `vgre-discover` into `~/.local/bin`

After install, open a **new terminal** and run:

```bash
vgre-dashboard          # launch the real-time monitor
vgre-start --test       # local master + worker self-test
```

### Automated Deployment (Windows)

```powershell
git clone https://github.com/vgre-org/vgre-runtime.git
cd vgre-runtime
.\scripts\vgre_sync.bat
```

`vgre_sync.bat` automatically:
- Detects and installs cmake / LLVM / Visual Studio Build Tools / Flutter via **winget** (falls back to **chocolatey**)
- Builds native engine and dashboard
- Creates a Desktop shortcut and launcher
- Installs **`vgre-token.bat`**, **`vgre-start.bat`**, and **`vgre-discover.bat`** into `%LOCALAPPDATA%\VGRE\scripts` and adds them to PATH (current session + future terminals)

Open a **new terminal** (or use the same one — PATH is updated immediately) and run:

```powershell
vgre-token generate       # create auth token
vgre-start --master       # launch master + dashboard
```

---

## 2. System Requirements

### All Platforms

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Linux 5.4 / Win 10 22H2 / macOS 11 | Ubuntu 22.04 / Win 11 / macOS 14 |
| CPU | x86-64, 2 cores | x86-64 with AVX-512 or ARM64, 8+ cores |
| RAM | 4 GB | 16 GB+ |
| CMake | 3.18 | 3.25+ |
| LLVM / Clang | 16 | 18 |
| C++ standard | C++17 | C++20 |
| Flutter SDK | 3.16 | 3.24+ |
| PowerShell | 5.1 (Windows built-in) | 7+ |

### Optional — unlocks additional features

| Package | Feature unlocked | Detection |
|---------|-----------------|-----------|
| `libomp-dev` | Multi-threaded kernel execution | Required |
| `libssl-dev` | Secure cluster transport (TLS) | Required |
| `libtss2-dev` | Hardware TPM 2.0 token storage (Linux) | **Auto-detected** |
| `libibverbs-dev` | RDMA/RoCE zero-copy transport | **Auto-detected** |
| `libsecret-1-dev` | GNOME Keyring token storage (Linux) | **Auto-detected** |
| `libgrpc++-dev` | gRPC cluster transport | Manual (`-DVGRE_ENABLE_GRPC=ON`) |
| Intel AMX CPU | AMX tile acceleration | **Auto-detected** (CPUID at build time) |

---

## 3. Installation

### 3.1 Linux Manual Installation

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

Verify:

```bash
source ~/.vgre/env        # load env in current shell (new terminals load it automatically)
vgre-dashboard --version  # prints: VGRE x.y.z
vgre-token fingerprint    # prints: SHA-256 of your token
vgre-discover             # shows your public IP + worker connection command
```

### 3.2 macOS Manual Installation

```bash
brew install cmake llvm libomp openssl git
brew install --cask flutter
bash install_local.sh
```

> **Apple Silicon (M1/M2/M3):** VGRE builds natively for ARM64. AVX-512 is disabled; performance comes from OpenMP parallelism.

### 3.3 Windows Manual Installation

1. Open **PowerShell** (no admin needed):

```powershell
.\scripts\vgre_sync.bat
```

2. If winget is unavailable, install manually first:
   - [CMake](https://cmake.org/download/) — tick "Add to PATH"
   - [LLVM 18](https://github.com/llvm/llvm-project/releases/tag/llvmorg-18.1.8)
   - [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) with "Desktop development with C++"
   - [Flutter SDK](https://flutter.dev/docs/get-started/install/windows)

3. Then re-run `vgre_sync.bat`.

### 3.4 Build options (cmake flags)

| Flag | Default | Effect |
|------|---------|--------|
| `-DCMAKE_BUILD_TYPE=Release` | Release | Optimized build |
| `-DVGRE_ENABLE_NATIVE_SIMD` | **auto** (ON if CPU supports it) | Enable `-march=native` (max SIMD, not portable) |
| `-DVGRE_ENABLE_RDMA` | **auto** (ON if `libibverbs-dev` found) | RDMA/RoCE zero-copy transport |
| `-DVGRE_ENABLE_TPM2` | **auto** (ON if `libtss2-dev` found) | TPM 2.0 hardware token storage |
| `-DVGRE_ENABLE_LIBSECRET` | **auto** (ON if `libsecret-1-dev` found) | GNOME Keyring token storage |
| `-DVGRE_ENABLE_GRPC=ON` | OFF | gRPC cluster transport (requires `libgrpc++-dev`) |
| `-DVGRE_WARNINGS_AS_ERRORS=ON` | ON | Treat all compiler warnings as errors |

Auto-detected features activate silently when their libraries are installed. The build never fails due to a missing optional dependency — it downgrades gracefully with a CMake warning.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

---

## 4. Running CUDA Applications

### 4.1 Intercept mode (LD_PRELOAD)

```bash
# Linux
export LD_PRELOAD="$HOME/.local/share/VGRE/lib/libvgre_cudart.so"
python my_pytorch_script.py

# macOS
export DYLD_INSERT_LIBRARIES="$HOME/.local/share/VGRE/lib/libvgre_cudart.dylib"
python my_pytorch_script.py
```

> **Windows:** Copy `%LOCALAPPDATA%\VGRE\vgre_cudart.dll` into the application directory and rename it `cudart64_120.dll` (match your CUDA version suffix).

### 4.2 Python bindings

```python
import vgre
import numpy as np

rt = vgre.Runtime()
rt.init(enable_profiling=True)

a = rt.alloc(1024 * np.float32().itemsize)
data = np.ones(1024, dtype=np.float32)
rt.memcpy_h2d(a, data)

kernel = vgre.Kernel("vector_scale", """
    __global__ void vector_scale(float* a, float scale, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) a[i] *= scale;
    }
""")
rt.launch(kernel, grid=(4, 1, 1), block=(256, 1, 1), args=[a, 2.0, 1024])
rt.synchronize()
print(np.frombuffer(rt.memcpy_d2h(a, 1024 * 4), dtype=np.float32)[:8])
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
On Windows they are set in User scope by `vgre_sync.bat` / `vgre_env.ps1`.

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `VGRE_LIB_PATH` | auto-detected | Absolute path to `libvgre.so` / `vgre.dll` |
| `LD_LIBRARY_PATH` | extended | Directory containing VGRE shared libraries |
| `VGRE_LOG_LEVEL` | `INFO` | Verbosity: `DEBUG` \| `INFO` \| `WARN` \| `ERROR` |
| `VGRE_INSTALL_DIR` | `~/.local/share/VGRE` | Installation directory |

### Kubernetes / Container

| Variable | Default | Description |
|----------|---------|-------------|
| `VGRE_DEVICE_PLUGIN_PATH` | `/var/lib/kubelet/device-plugins/kubelet.sock` | Override kubelet socket path for K8s device plugin registration |

### Cluster / networking

| Variable | Default | Description |
|----------|---------|-------------|
| `VGRE_PORT` | `7777` | TCP port for master and worker nodes |
| `VGRE_CLUSTER_MASTER_ADDRESS` | _(empty)_ | **WAN direct connect**: `IP:PORT`, hostname, or `[::1]:PORT`. Worker connects directly without UDP broadcast. Supports IPv4, IPv6, and hostnames. |
| `VGRE_CLUSTER_ADVERTISED_ADDRESS` | _(empty)_ | **Master public address**: included in UDP pings so workers behind NAT receive the correct address. Set automatically by `vgre-start --master` via `vgre-discover --set-master`. |
| `VGRE_CLUSTER_NODES` | _(empty)_ | Comma-separated `IP:PORT` list of worker addresses (master side). |
| `VGRE_MESH_PEERS` | _(empty)_ | Comma-separated `IP:PORT` list for full-mesh topology (all nodes list all peers). |
| `VGRE_CLUSTER_DISCOVERY` | _(enabled)_ | Set to `OFF` to disable UDP broadcast (pure WAN/explicit-address mode). |
| `VGRE_TCP_AUTH_TOKEN_FILE` | `~/.vgre/token` | Path to the shared cluster auth token file. |
| `VGRE_TCP_AUTH_TOKEN` | _(empty)_ | Raw token string (less secure; prefer the file). |

### WAN / connection tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `VGRE_CLUSTER_CONNECT_TIMEOUT_SEC` | `10` | TCP connect timeout in seconds (1–120). Increase for high-latency WAN links. |
| `VGRE_CLUSTER_MAX_BACKOFF_SEC` | `120` | Maximum retry interval for proactive reconnection (10–3600). |
| `VGRE_CLUSTER_IDLE_EVICT_SEC` | `300` | Seconds of silence before an idle connection is evicted (0 = disabled). |

### Discovery

| Variable | Default | Description |
|----------|---------|-------------|
| `VGRE_DISCOVERY_BUCKET_ID` | _(auto-created)_ | kvdb.io bucket ID for cross-LAN token-keyed discovery. Created automatically on first `vgre-discover --register`. |
| `VGRE_CLUSTER_UDP_ANNOUNCE_PORT` | `7778` | UDP port master broadcasts on (workers listen). Must match on all nodes. |
| `VGRE_CLUSTER_UDP_WORKER_PORT` | `7779` | UDP port workers broadcast on (master listens). Must match on all nodes. |

### GPU cache model

| Variable | Default | Description |
|----------|---------|-------------|
| `VGRE_L1_CACHE_KB` | `32` | Per-block L1 cache size in KB (16 \| 32 \| 64 \| 128) |
| `VGRE_L2_CACHE_MB` | `6` | Per-device L2 cache size in MB (2 \| 6 \| 20 \| 40) |

### Performance tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `VGRE_ENABLE_NUMA` | `0` | Enable NUMA-aware thread scheduling (Linux) |
| `VGRE_WORKER_THREADS` | auto (nproc) | Override worker thread count |
| `VGRE_SIMD_LEVEL` | auto-detected | Force SIMD level: `SSE4` \| `AVX` \| `AVX2` \| `AVX512` |

### Configuration Management

| Variable | Default | Description |
|----------|---------|-------------|
| `VGRE_CONFIG_FILE` | _(empty)_ | Path to a JSON or YAML configuration file. If defined, loads configuration variables from the file. |
| `VGRE_CONFIG_HOT_RELOAD` | `false` | If set to `true`, the system monitors the configuration file for changes and hot-reloads config on the fly. |
| `VGRE_DEPLOYMENT_PROFILE` | `development` | Active configuration profile: `development` \| `staging` \| `production` \| `custom`. |

---

## 6. Token Management (`vgre-token`)

`vgre-token` works from **any directory** on all three platforms after install.

### Commands

```
vgre-token generate          generate a new secure 256-bit token
vgre-token show              print the stored token value
vgre-token fingerprint       print the SHA-256 fingerprint
vgre-token set [TOKEN]       store a token pasted from another machine
vgre-token verify            check env-var matches the stored file
vgre-token copy              print the scp / robocopy command for workers
vgre-token revoke            delete the stored token (with confirmation)
vgre-token install           install vgre-token to User PATH (first-time)
```

### Typical workflow

```bash
# ── On the MASTER machine ─────────────────────────────────────────────────
vgre-token generate
# Output: Token saved to ~/.vgre/token
#         Token fingerprint (SHA-256): a3f9bcd2...

# Share the token with workers (choose one method):
vgre-token copy               # prints scp/robocopy command
scp ~/.vgre/token worker@192.168.1.50:~/.vgre/token

# ── On every WORKER machine ───────────────────────────────────────────────
# Option A — copy file via scp (see above)
# Option B — paste the token value
vgre-token set a3f9bcd2xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# ── Verify both machines match ────────────────────────────────────────────
vgre-token fingerprint    # run on BOTH — output must be identical
```

### Token storage

| Platform | Token file | Env var set in |
|----------|-----------|----------------|
| Linux/macOS | `~/.vgre/token` (chmod 600) | `~/.vgre/env` → shell profile |
| Windows | `%USERPROFILE%\.vgre\token` | User-scope env (no restart needed) |

---

## 7. Cluster Setup (`vgre-start`)

`vgre-start` launches master or worker nodes. It automatically sources `~/.vgre/env` (Linux/macOS) or sets up the DLL PATH (Windows) before starting. Available on **all platforms** via `vgre-start.bat` (Windows), `vgre-start.sh` symlinked as `vgre-start` (Linux/macOS).

### Commands

```
vgre-start --master                              Start master node (launches dashboard)
vgre-start --worker                              Start worker (LAN auto-discovery)
vgre-start --worker --is-master                  Start as headless master (no dashboard; K8s/container mode)
vgre-start --worker --master-ip <IP>             LAN: connect to specific master IP
vgre-start --worker --master-address <HOST:PORT> WAN: hostname, IPv4, or IPv6 direct connect
vgre-start --test                                Local self-test (master + worker, same machine)
vgre-start --port <N>                            Use custom TCP port (default 7777)
vgre-start --threads <N>                         Worker thread count (default: auto)
vgre-start --help                                Show all options
```

### Local Area Network (LAN) Cluster Initialization

**Phase 1: Node Installation**

```bash
bash install_local.sh   # Linux/macOS
.\scripts\vgre_sync.bat # Windows (installs all CLI tools including vgre-start)
```

**Phase 2: Master Token Generation**

```bash
vgre-token generate
```

**Phase 3: Worker Token Distribution**

```bash
# scp (Linux/macOS)
vgre-token copy
scp ~/.vgre/token worker@192.168.1.50:~/.vgre/token

# Windows
vgre-token copy         # prints the robocopy command
# Or: vgre-token set <paste-token-here> on the worker
```

**Phase 4: Service Initialization**

```bash
# Master:
vgre-start --master

# Workers (LAN auto-discovery):
vgre-start --worker

# Workers (explicit LAN IP):
vgre-start --worker --master-ip 192.168.1.10
```

### Wide Area Network (WAN) Cluster Initialization

For workers on a **different network** than the master, UDP broadcast cannot cross routers. Use direct TCP connection instead.

**Method A: Explicit IP Configuration**

```bash
# On master: find your public IP
vgre-discover
# Output: Public IP: 78.45.12.99
#         vgre-start --worker --master-address 78.45.12.99:7777

# On worker (any network):
vgre-start --worker --master-address 78.45.12.99:7777
```

**Method B: Automated Token-Keyed Discovery**

```bash
# On master (run once, or after IP changes):
vgre-discover --register
# Output: Bucket ID: aBcDeFgH   <- share this with workers

# On each worker (same token required):
vgre-discover --find aBcDeFgH
# Output: Master found: 78.45.12.99:7777
#         Run: vgre-start --worker --master-address 78.45.12.99:7777

vgre-start --worker --master-address 78.45.12.99:7777
```

**Method C: Virtual Private Network (VPN) Overlay**

```bash
# Both machines on the same Tailscale network
# Master's Tailscale IP: 100.64.0.1

vgre-start --worker --master-ip 100.64.0.1
# OR with explicit port:
vgre-start --worker --master-address 100.64.0.1:7777
```

### Firewall rules required for WAN

```bash
# Linux (UFW)
sudo ufw allow 7777/tcp     # TCP cluster
sudo ufw allow 7778/udp     # UDP discovery (LAN only)

# Windows PowerShell (admin)
New-NetFirewallRule -DisplayName "VGRE TCP" -Direction Inbound -Protocol TCP -LocalPort 7777 -Action Allow

# macOS: System Settings > Network > Firewall > Options
# Allow incoming connections for vgre-worker / vgre_dashboard
```

### Automatic reconnection

Workers use exponential backoff to automatically reconnect after master restarts or network drops:

| State | Behavior |
|-------|----------|
| Connection drops | Worker retries immediately, then 1 s → 2 → 4 → … → 120 s |
| Master restarts | Worker reconnects within one backoff cycle |
| IP changes | Re-run `vgre-discover --register` on master; workers re-run `--find` |

Backoff ceiling is configurable: `export VGRE_CLUSTER_MAX_BACKOFF_SEC=60`

### Dashboard worker disconnect detection

The dashboard detects worker disconnections automatically (TCP keepalive, ~12 s after drop):

- **Red card** with `DISCONNECTED` + countdown timer appears for the node
- **Floating notification** banner: "Worker 192.168.x.x disconnected"
- Node is **removed** from the topology view after 30 seconds
- 3D topology canvas updates in real time

### Cross-platform compatibility

| Master | Worker | Notes |
|--------|--------|-------|
| Linux | Linux / macOS / Windows | Fully supported |
| macOS | Linux / Windows | Fully supported |
| Windows | Linux / macOS | Fully supported |

---

## 8. Public IP Discovery (`vgre-discover`)

`vgre-discover` is a standalone CLI available on all platforms after `vgre_sync.bat` / `install_local.sh`. It solves the WAN cluster bootstrap problem: detecting the real public IP and making it available to workers without manual configuration.

### Commands

```
vgre-discover                  Show real public IP + worker connection command
vgre-discover --set-master     Set VGRE_CLUSTER_ADVERTISED_ADDRESS for this session
                               (called automatically by vgre-start --master)
vgre-discover --register       Register master's public IP to a token-keyed KV store
                               (free kvdb.io bucket, auto-created on first use)
vgre-discover --find [BUCKET]  Retrieve master's IP using the local token
                               (worker side — verifies token fingerprint)
vgre-discover --unregister     Remove registration from the KV store
```

### How it works

**Public IP detection** — tries 5 HTTP APIs in order, uses the first valid IPv4 response:
1. `api.ipify.org`
2. `ipecho.net/plain`
3. `icanhazip.com`
4. `checkip.amazonaws.com`
5. `api4.my-ip.io/ip`

**Master auto-broadcast** — `vgre-start --master` calls `vgre-discover --set-master` before launching, which sets `VGRE_CLUSTER_ADVERTISED_ADDRESS=<public-ip>:7777`. The C++ UDP announcer loop reads this variable and embeds the real public IP in every broadcast packet. Workers on the same LAN (or VPN) receive the correct address instead of the NAT-internal sender IP.

**Cross-LAN token-keyed discovery** — the KV store key is derived as:
```
key = "vgre-" + first_32_hex_chars_of( SHA256(token) )
```
Only nodes with the same token can compute or look up the same key. The bucket ID is a non-secret opaque identifier that must be shared once with workers (like a meeting room ID, not a password).

### Quick Start: WAN Cluster with Automated Discovery

```bash
# ── Master ────────────────────────────────────────────────────────────────
# Generate token (if not already done)
vgre-token generate

# Register public IP — prints the BUCKET_ID to share
vgre-discover --register
# Output:
#   [OK] Master registered successfully.
#   Bucket ID : aBcDeFgH
#   Master IP : 78.45.12.99:7777
#   Token FP  : a3f9bcd2...
#
#   Share with workers:
#     vgre-discover --find aBcDeFgH

# Start master (auto-detects IP and sets advertised address)
vgre-start --master

# ── Workers (any network, same token) ─────────────────────────────────────
# Copy token from master first (scp or vgre-token set)

# Find the master automatically
vgre-discover --find aBcDeFgH
# Output: Master found: 78.45.12.99:7777
#         Connect: vgre-start --worker --master-address 78.45.12.99:7777

vgre-start --worker --master-address 78.45.12.99:7777
```

### Discovery Modes: LAN vs WAN

| Scenario | How discovery works |
|----------|-------------------|
| Same LAN | UDP broadcast — automatic, no configuration |
| Different LAN, same token | `vgre-discover --register` → `--find` |
| Different LAN, manual | `vgre-discover` shows IP → share the one-liner |
| VPN (Tailscale/ZeroTier) | Same as LAN — auto-discovery works |
| Dynamic IP (home ISP) | Re-run `vgre-discover --register` after IP changes |

### NAT / firewall note

`vgre-discover` detects your **external** IP, but workers can only reach the master if:
- Your router port-forwards TCP `7777` → this machine, **OR**
- You use a VPN like [Tailscale](https://tailscale.com) or [ZeroTier](https://www.zerotier.com) (no port-forward needed), **OR**
- The master runs on a VPS / cloud VM with a direct public IP

---

## 9. Dashboard

The Flutter dashboard provides real-time visibility into the VGRE runtime.

### Launch

```bash
vgre-dashboard          # Linux/macOS
# Windows: double-click "VGRE Dashboard" Desktop shortcut
# Or: vgre-start --master (launches dashboard + master node)
```

### Dashboard tabs

| Tab | What it shows |
|-----|---------------|
| **Overview** | Compute utilization, memory bandwidth, temperature, L2 cache hit rate (live) |
| **Kernel Explorer** | Top-N kernels by total time, per-kernel invocation history, PTX / IR source viewer |
| **Cluster Topology** | Connected nodes, 3D animated topology, latency heat-map, **live disconnect notifications**, per-node credits |
| **Hardware Tuning** | SIMD capabilities, thread count controls, background compute toggle |
| **Memory Analysis** | UVM page map, pool allocator stats, active allocation list |
| **Settings** | Log level, profiler on/off, service mode, secure cluster VPN toggle |

### Worker disconnect notifications

The dashboard polls worker state every 500 ms. When a worker's TCP connection drops:

1. Node card turns **red** with a `DISCONNECTED` label and a live countdown (`Removing in Ns`)
2. A floating **SnackBar notification** appears: "Worker 192.168.x.x disconnected"
3. After 30 seconds the node is removed from the list and 3D canvas
4. When the worker reconnects, the node reappears automatically

### Library resolution order

1. `VGRE_LIB_PATH` env var
2. `<bundle>/lib/libvgre.so` (installed bundle)
3. `~/.local/share/VGRE/lib/libvgre.so`
4. `build/libvgre.so` (repo build, developer mode)

---

## 10. Advanced Features

### 10.1 GPU L1/L2 cache model

```bash
export VGRE_L1_CACHE_KB=64    # 16 | 32 | 64 | 128
export VGRE_L2_CACHE_MB=20    # 2 | 6 | 20 | 40
```

```c
vgre_cache_stats_t cs;
vgre_get_cache_stats(&cs);
printf("L2 hit rate: %.1f%%  hits=%llu  misses=%llu\n",
       cs.l2_hit_rate * 100.0, cs.l2_hits, cs.l2_misses);
```

### 10.2 AMX + AVX-512 WMMA acceleration

- **AMX path**: 16×16×16 BF16 matrix tiles (Sapphire Rapids+)
- **AVX-512 path**: N=16 tiles, 16× speedup vs scalar
- **Scalar fallback**: always available

```bash
VGRE_LOG_LEVEL=DEBUG vgre-dashboard 2>&1 | grep -E "AMX|AVX-512|SIMD"
```

### 10.3 Hopper PTX emulation

| PTX instruction | CPU emulation |
|----------------|---------------|
| `wgmma.mma_async.*` | Full M×N×K GEMM via `vgre_wgmma_*` (AVX-512 when N=256) |
| `cp.async.bulk.tensor.*` | Synchronous `memcpy` |
| `wgmma.fence` / `wgmma.wait_group` | `__atomic_thread_fence(SEQ_CST)` |
| `mma.sync.aligned.m16n8k16` | `vgre_mma_m16n8k16_f32_f16` scalar tile GEMM |

### 10.4 CUDA MPS

```bash
export VGRE_MPS_PIPE=/tmp/vgre_mps.sock
# Each client process connects automatically when VGRE_MPS_PIPE is set.
```

### 10.5 RDMA / RoCE transport

```bash
cmake -S . -B build -DVGRE_ENABLE_RDMA=ON
# Requires: sudo apt-get install libibverbs-dev rdma-core

# Soft-RoCE loopback for testing:
sudo rdma link add rxe0 type rxe netdev eth0
export VGRE_RDMA_DEVICE=rxe0
```

### 10.6 gRPC cluster transport

```bash
cmake -S . -B build -DVGRE_ENABLE_GRPC=ON
export VGRE_GRPC_PORT=50051
vgre-start --master
```

---

## 11. Troubleshooting

### 11.1 Dashboard fails to load ("Failed to load native library")

```bash
source ~/.vgre/env
ls -la ~/.local/share/VGRE/lib/libvgre.so
ldd ~/.local/share/VGRE/lib/libvgre.so | grep "not found"
# If libomp.so is missing:
sudo apt-get install libomp-dev
```

### 11.2 `vgre-start` not recognised (Windows)

`vgre-start.bat` is installed to `%LOCALAPPDATA%\VGRE\scripts` by `vgre_sync.bat`.

```powershell
# Check it is installed
Test-Path "$env:LOCALAPPDATA\VGRE\scripts\vgre-start.bat"

# If PATH is not updated yet in current terminal:
$env:PATH = "$env:LOCALAPPDATA\VGRE\scripts;$env:PATH"
vgre-start --help

# Reinstall (re-run sync):
.\scripts\vgre_sync.bat
```

### 11.3 Worker cannot connect to master

```bash
# 1. Verify tokens match
vgre-token fingerprint      # run on BOTH — must be identical

# 2. Test network reachability
nc -zv MASTER_IP 7777                            # Linux/macOS
Test-NetConnection MASTER_IP -Port 7777          # Windows PowerShell

# 3. Open firewall
sudo ufw allow 7777/tcp                          # Linux
New-NetFirewallRule -DisplayName "VGRE" -Direction Inbound -Protocol TCP -LocalPort 7777 -Action Allow  # Windows admin

# 4. WAN: verify correct public IP
vgre-discover

# 5. Increase connect timeout for slow WAN links
export VGRE_CLUSTER_CONNECT_TIMEOUT_SEC=30
```

### 11.4 UDP auto-discovery not working across subnets

UDP broadcast (`255.255.255.255`) cannot cross routers. Use direct connection instead:

```bash
# Worker — explicit master address (LAN or WAN)
vgre-start --worker --master-ip 192.168.1.10          # same LAN, different subnet
vgre-start --worker --master-address 78.45.12.99:7777 # WAN / internet

# Or use cross-LAN token-keyed discovery:
vgre-discover --register   # on master
vgre-discover --find <ID>  # on worker
```

### 11.5 Mismatched token fingerprints

```bash
# Re-copy the token from master
vgre-token copy          # on master — prints scp command
scp ~/.vgre/token worker@WORKER_IP:~/.vgre/token
vgre-token fingerprint   # verify on both machines
```

### 11.6 CMake cannot find LLVM

```bash
# Linux
sudo apt-get install llvm-18 llvm-18-dev clang-18
export LLVM_DIR=$(llvm-config-18 --cmakedir)

# macOS
brew install llvm
export LLVM_DIR=$(brew --prefix llvm)/lib/cmake/llvm

# Windows
winget install LLVM.LLVM
set LLVM_DIR=C:\Program Files\LLVM\lib\cmake\llvm
```

### 11.7 Flutter build fails (Windows)

```powershell
# Re-run the sync script — it now pre-caches engine artifacts automatically
.\scripts\vgre_sync.bat

# Or manually:
flutter precache --windows
cd vgre_dashboard
flutter pub get
flutter build windows --release
```

### 11.8 Windows: "Windows cannot find vgre_dashboard.exe"

The Flutter build did not complete. The launcher now checks for the exe and shows a clear message:

```
[ERROR] vgre_dashboard.exe not found. Run .\scripts\vgre_sync.bat to build and deploy.
```

Check that `flutter precache --windows` and `flutter build windows --release` succeed without errors in the sync output.

### 11.9 Low compute performance

SIMD is auto-detected at configure time. Check which level was selected:

```bash
# See which SIMD features CMake detected
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release 2>&1 | grep -i "simd\|avx\|amx"

# Force a specific SIMD level at runtime
export VGRE_SIMD_LEVEL=AVX512

# Debug-confirm which path VGRE is using
VGRE_LOG_LEVEL=DEBUG ./build/examples/matrix_multiply 2>&1 | grep "SIMD\|AVX\|AMX"
```

If SIMD was not auto-enabled, install the dev tools and rebuild:

```bash
# Ubuntu/Debian — nothing extra needed; /proc/cpuinfo probed at configure time
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### 11.10 Windows: DLL load failure (0xC000001D / error 1114)

```powershell
dumpbin /dependents $env:LOCALAPPDATA\VGRE\vgre.dll
Get-ChildItem "$env:LOCALAPPDATA\VGRE\lib" -Filter "*.dll" | Select-Object Name
.\scripts\vgre_sync.bat    # re-run to restore missing DLLs
```

### 11.11 OpenMP not working (all kernels run on 1 thread)

```bash
ldd ~/.local/share/VGRE/lib/libvgre.so | grep omp
sudo apt-get install libomp-dev       # Ubuntu/Debian
sudo dnf install libomp-devel         # Fedora
brew install libomp                   # macOS
cmake --build build -j$(nproc)
### 11.12 Windows: Socket and Port Bind Failures (ACCESS_VIOLATION 0xC0000005)

When setting up or testing VGRE TCP clusters on Windows, you may encounter Winsock errors or dynamic-link crashes. These are resolved in the runtime through the following configurations:

#### 1. WSAStartup Lifecycle (WSANOTINITIALIZED)
- **Symptom**: Socket creations fail with WinSock error `10093`.
- **Cause**: Multithreaded execution starting socket operations before Winsock is initialized, or static destruction cleaning up Winsock too early.
- **Solution**: VGRE utilizes a thread-safe `WindowsSocketManager` with Meyers singletons and dynamic `std::call_once` bounds to manage WSAStartup/WSACleanup lifecycles automatically. If a custom application experiences this, ensure Winsock is not closed prematurely by external libraries.

#### 2. Cross-Platform Struct Packing Inconsistencies
- **Symptom**: Buffer overruns or ACCESS_VIOLATION crashes when connecting a Linux master to a Windows worker (or vice-versa).
- **Cause**: Structural memory alignment discrepancies. On Linux, 64-bit offsets in network headers default to 8-byte alignment; Windows MSVC compiler defaults may align differently, causing packet headers to mismatch in size (e.g., `SecurePacketHeader` or `VSBPHeader`).
- **Solution**: All cross-platform network packet headers are compiled using strict `#pragma pack(push, 1)` and `__attribute__((packed))` directives to guarantee bit-level structural identity across compilers.

#### 3. Socket Port Conflicts
- **Symptom**: Master fails to start, logging "Address already in use" or crashing on bind.
- **Cause**: Multiple tests or workers executing concurrently on ports `7777` (cluster) or `7778` (discovery) before the OS has fully released the socket from a previous run (TIME_WAIT state).
- **Solution**: Set the environment variable `VGRE_PORT` to allocate a distinct TCP port, or increase `VGRE_CLUSTER_CONNECT_TIMEOUT_SEC` to allow lingering connections to close naturally.

---

## Production Status

| Feature | Status |
|---------|--------|
| CUDA runtime intercept (cudart) | Production |
| LLVM JIT kernel compilation | Production |
| cuBLAS INT8 / FP32 / FP64 | Production |
| cuDNN convolution (INT8, FP32) | Production |
| NCCL AllReduce (ring + barrier-tree) | Production |
| CUDA Graphs (IF / WHILE / SWITCH) | Production |
| Virtual memory (cuMemCreate / cuMemMap) | Production |
| External semaphores (eventfd / Win32) | Production |
| UVM managed memory | Production |
| NVTX profiling markers | Production |
| AVX-512 WMMA acceleration | Production |
| Intel AMX tile acceleration | Production (Sapphire Rapids+) |
| Hopper PTX (wgmma, TMA, cp.async.bulk) | Production |
| GPU L1/L2 cache model | Production |
| TCP cluster + TLS | Production |
| **WAN cluster (hostname/IPv6/NAT)** | **Production** |
| **Worker auto-disconnect detection** | **Production** |
| RDMA / RoCE transport | Optional (`-DVGRE_ENABLE_RDMA=ON`) |
| gRPC cluster transport | Optional (`-DVGRE_ENABLE_GRPC=ON`) |
| CUDA MPS multi-process | Production (Unix socket) |
| cuMemMulticast | Production |
| Flutter real-time dashboard | Production |
| `vgre-token` CLI | Production (Linux / macOS / Windows) |
| `vgre-start` cluster launcher | Production (Linux / macOS / Windows) |
| **`vgre-discover` IP discovery** | **Production** |
| OTel hw.gpu.* telemetry export | Production |
