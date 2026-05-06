# VGRE User Guide

Welcome to the Virtual GPU Runtime Engine (VGRE). This guide will help you set up and use VGRE to run CUDA-accelerated applications on non-NVIDIA hardware across Windows, Linux, and macOS.

---

## 1. Getting Started

### 1.1 System Requirements

**All Platforms:**
- **Compiler**: LLVM/Clang 16+ (required for kernel compilation)
- **Build System**: CMake 3.16+
- **Memory**: 4GB+ RAM recommended
- **CPU**: x86_64 or ARM64 (Apple Silicon supported)

**Platform-Specific Requirements:**

**Linux (Ubuntu 22.04+, Fedora 38+, or equivalent):**
- `build-essential` or equivalent development tools
- `libssl-dev` (for secure networking)
- `libtpm2-tss-dev` (optional, for hardware token storage)
- `libsecret-1-dev` (optional, for GNOME Keyring integration)
- `keyutils` (optional, for Linux kernel keyring)

**Windows (Windows 10/11):**
- Visual Studio 2022 Build Tools or Community Edition with C++ support
- Windows SDK 10.0.19041.0 or later
- PowerShell 5.1+ or PowerShell Core 7+

**macOS (macOS 11.0+):**
- Xcode Command Line Tools (`xcode-select --install`)
- Homebrew (recommended for dependencies)
- `brew install libomp` (for OpenMP support)

### 1.2 Installation

#### Quick Installation (All Platforms)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vgre-org/vgre-runtime.git
   cd vgre-runtime
   ```

#### Linux Installation

2. **Install dependencies:**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install build-essential cmake llvm-18 clang-18 libssl-dev libtpm2-tss-dev libsecret-1-dev keyutils

   # Fedora/RHEL
   sudo dnf install gcc-c++ cmake llvm clang openssl-devel tpm2-tss-devel libsecret-devel keyutils
   ```

3. **Build and install:**
   ```bash
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   cmake --build . --parallel $(nproc)
   sudo cmake --install .
   ```
   This installs libraries to `/usr/local/lib/vgre/` and headers to `/usr/local/include/vgre/`.

#### macOS Installation

2. **Install dependencies:**
   ```bash
   # Install Homebrew if not already installed
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   
   # Install dependencies
   brew install cmake llvm libomp
   ```

3. **Build and install:**
   ```bash
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   cmake --build . --parallel $(sysctl -n hw.ncpu)
   sudo cmake --install .
   ```
   This installs libraries to `/usr/local/lib/vgre/` and headers to `/usr/local/include/vgre/`.

#### Windows Installation

2. **Install build tools:**
   - Download and install [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
   - Or install Visual Studio 2022 Community with "Desktop development with C++" workload
   - Install [CMake](https://cmake.org/download/) and add to PATH
   - Install [LLVM](https://github.com/llvm/llvm-project/releases) and add to PATH

3. **Build the project:**
   ```powershell
   # Open "Developer PowerShell for VS 2022" or run vcvars64.bat first
   mkdir build
   cd build
   cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
   cmake --build . --config Release --parallel
   ```

4. **Install (optional):**
   ```powershell
   # Run as Administrator
   cmake --install . --config Release
   ```
   This installs to `C:\Program Files\VGRE\` by default.

---

## 2. Using VGRE with Applications

VGRE supports two primary modes of operation: **Native Integration** and **Framework Interception**.

### 2.1 Framework Interception (PyTorch, TensorFlow, etc.)

#### Linux/macOS
To run an existing CUDA application (like a Python script using PyTorch) on VGRE, use `LD_PRELOAD` to redirect CUDA calls:

```bash
export LD_PRELOAD=/usr/local/lib/vgre/libvgre_cudart.so
export VGRE_LOG_LEVEL=INFO
python my_pytorch_script.py
```

#### Windows
On Windows, use DLL replacement or PATH manipulation:

```powershell
# Method 1: Copy VGRE DLLs to application directory
copy "C:\Program Files\VGRE\bin\vgre_cudart.dll" ".\cudart64_110.dll"
copy "C:\Program Files\VGRE\bin\vgre.dll" ".\"

# Method 2: Add VGRE to PATH (system-wide)
$env:PATH = "C:\Program Files\VGRE\bin;$env:PATH"
python my_pytorch_script.py
```

### 2.2 Native Python Integration
For custom scripts, use the `vgre` Python package (all platforms):

```python
import vgre
import numpy as np

# Initialize the VGRE Runtime
rt = vgre.Runtime()
rt.init(enable_profiling=True)

# Launch a kernel
kernel = vgre.Kernel("vector_add", my_source_code)
rt.launch(kernel, grid=(1024, 1, 1), block=(256, 1, 1), args=[a_ptr, b_ptr, c_ptr])

rt.synchronize()
```

### 2.3 Environment Variables (All Platforms)

**Common Configuration:**
```bash
# Linux/macOS
export VGRE_LOG_LEVEL=INFO              # DEBUG, INFO, WARN, ERROR
export VGRE_CACHE_DIR=~/.vgre/cache     # Kernel compilation cache
export VGRE_DEVICE_COUNT=4              # Override auto-detected device count

# Windows (PowerShell)
$env:VGRE_LOG_LEVEL="INFO"
$env:VGRE_CACHE_DIR="$env:USERPROFILE\.vgre\cache"
$env:VGRE_DEVICE_COUNT="4"
```

**Performance Tuning:**
```bash
# Linux/macOS
export VGRE_ENABLE_NUMA=1               # Enable NUMA awareness (Linux only)
export VGRE_WORKER_THREADS=16           # Override worker thread count
export VGRE_SIMD_LEVEL=AVX2             # Force SIMD level (SSE4, AVX, AVX2, AVX512)

# Windows (PowerShell)
$env:VGRE_WORKER_THREADS="16"
$env:VGRE_SIMD_LEVEL="AVX2"
```

---

## 3. VGRE Dashboard & Monitoring

VGRE includes a real-time monitor to visualize compute and memory utilization across all platforms.

### 3.1 Dashboard Installation

**Prerequisites:**
- Flutter SDK 3.0+ ([installation guide](https://flutter.dev/docs/get-started/install))

**Build Dashboard:**
```bash
# All platforms - run from project root
cd vgre_dashboard
flutter build [platform]
```

Where `[platform]` is:
- `linux` for Linux
- `windows` for Windows  
- `macos` for macOS

### 3.2 Launch Dashboard

**Linux:**
```bash
./vgre_dashboard/build/linux/x64/release/bundle/vgre_dashboard
# Or if installed system-wide:
vgre-dashboard
```

**macOS:**
```bash
open ./vgre_dashboard/build/macos/Build/Products/Release/vgre_dashboard.app
# Or if installed:
open -a "VGRE Dashboard"
```

**Windows:**
```powershell
.\vgre_dashboard\build\windows\x64\runner\Release\vgre_dashboard.exe
# Or if installed:
Start-Process "VGRE Dashboard"
```

### 3.3 Dashboard Features
- **GFLOPS Meter**: Real-time compute performance across all devices
- **UVM Residency Map**: Visualizes page migrations in managed memory
- **3D Topology**: View your virtual device cluster in a 3D orbital space
- **Cross-Platform**: Native look and feel on each operating system
- **Auto-Discovery**: Automatically connects to active VGRE sessions via IPC

---

## 4. Distributed Cluster Setup

VGRE lets you aggregate multiple network-connected CPUs into a single Virtual GPU cluster. This works across mixed platforms (Linux, Windows, macOS).

> [!IMPORTANT]
> Run the scripts **in the exact order shown below**. Each step must finish successfully before moving to the next. You only run steps 1–3 once per machine; after that you only need step 4 every time you want to use the cluster.

---

### 4.0 Script Execution Order

```
Every machine (master + every worker):
  STEP 1 → Build & install     (vgre_sync.sh / vgre_sync.bat)
  STEP 2 → Set up cluster token (setup-cluster.sh / Setup-VGRECluster.ps1)

Master machine only:
  STEP 3 → Copy token to workers

Then to run the cluster:
  STEP 4a → Start master        (vgre-start --master / Start-VGRE.ps1 --master)
  STEP 4b → Start each worker   (vgre-start --worker / Start-VGRE.ps1 --worker)
```

---

### Step 1 — Build and Install (run once per machine)

This compiles the VGRE engine, worker binary, and dashboard, then installs them to your local profile.

**Linux / macOS:**
```bash
bash scripts/vgre_sync.sh
```

**Windows (run from Developer PowerShell or a terminal with VS build tools in PATH):**
```
scripts\vgre_sync.bat
```

Expected output: ends with `✅ VGRE Sync Complete!` (Linux/macOS) or `VGRE Sync Complete.` (Windows).  
If the build fails, fix the reported error before continuing.

---

### Step 2 — Set Up the Cluster Token (run once per machine)

Every machine in the cluster — master and every worker — must run this step. The script generates a secure 64-character random token, saves it to a local file, and wires it into your shell/environment automatically so you never have to type it again.

**Linux / macOS:**
```bash
bash scripts/setup-cluster.sh
```

**Windows (PowerShell):**
```powershell
.\scripts\Setup-VGRECluster.ps1
```

The script will:
1. Ask whether to generate a new token or paste one from another node
2. Save the token to `~/.vgre/token` (Linux/macOS) or `%USERPROFILE%\.vgre\token` (Windows)
3. Add `VGRE_TCP_AUTH_TOKEN_FILE` to your shell profile / User environment — permanent, loads on every new terminal
4. Display a **SHA256 fingerprint** (first 16 characters) — write this down

> [!WARNING]
> **Do not run the setup script more than once** unless you intend to replace the token. Each run generates a new random token. If you regenerate on the master, all workers become disconnected until they get the new token file.

To check the current token fingerprint on any node at any time without regenerating:

```bash
# Linux/macOS
bash scripts/setup-cluster.sh --show-fingerprint
```
```powershell
# Windows
.\scripts\Setup-VGRECluster.ps1 -ShowFingerprint
```

---

### Step 3 — Copy the Token to Every Worker (run on master only)

The master and all workers **must have the identical token file**. Only the master generates the token in Step 2; all workers receive a copy of it.

**Linux/macOS — copy from master to each worker over SSH:**
```bash
# Replace user@WORKER_IP with actual credentials for each worker machine
scp ~/.vgre/token  user@WORKER_IP:~/.vgre/token
```

**Windows — copy from master to worker (PowerShell with SSH):**
```powershell
scp "$env:USERPROFILE\.vgre\token"  user@WORKER_IP:"~/.vgre/token"
```

**If SSH is not available** (USB, shared drive, etc.):
1. On the master: run `bash scripts/setup-cluster.sh --show-fingerprint` and note the full SHA256
2. On the worker: run `bash scripts/setup-cluster.sh` (Linux) or `.\scripts\Setup-VGRECluster.ps1` (Windows), choose option **2 (Enter your own token)**, and paste the token printed by the master's setup script

**After copying, verify the fingerprints match on both machines:**
```bash
# Run on BOTH master and worker — output must be identical
bash scripts/setup-cluster.sh --show-fingerprint          # Linux/macOS
.\scripts\Setup-VGRECluster.ps1 -ShowFingerprint          # Windows
```

If the fingerprints differ, the handshake will fail. Re-copy the token file and verify again before proceeding.

---

### Step 4a — Start the Master

Run this on the machine that will run your CUDA application. The master is embedded in the VGRE Dashboard.

**Linux/macOS:**
```bash
vgre-start --master
```

**Windows (PowerShell):**
```powershell
.\scripts\Start-VGRE.ps1 --master
```

The script will print the token fingerprint before launching. The dashboard opens and begins listening for worker connections on port 7777. Keep this terminal / process running.

---

### Step 4b — Start Each Worker

Run this on every compute node that will contribute CPU resources. Start the master (Step 4a) before starting workers.

**Linux/macOS — same subnet (auto-discovery):**
```bash
vgre-start --worker
```

**Linux/macOS — different subnet (specify master IP):**
```bash
vgre-start --worker --master-ip 192.168.1.50
```

**Windows — same subnet:**
```powershell
.\scripts\Start-VGRE.ps1 --worker
```

**Windows — different subnet:**
```powershell
.\scripts\Start-VGRE.ps1 --worker --master-ip 192.168.1.50
```

The worker prints its token fingerprint, then enters a scanning/discovery loop. Once connected, the master dashboard shows the worker's CPU and RAM stats.

**Optional flags (all platforms):**
```bash
# Custom port (must match master port)
vgre-start --worker --port 7778

# Limit thread count
vgre-start --worker --threads 8

# Windows equivalents:
.\scripts\Start-VGRE.ps1 --worker --port 7778 --threads 8
```

---

### Quick Local Test (same machine — no worker needed)

To verify your installation without a second machine:

**Linux/macOS:**
```bash
vgre-start --test
```

**Windows:**
```powershell
.\scripts\Start-VGRE.ps1 --test
```

This starts both master and worker on the same machine and shuts them down cleanly when you press Enter / Ctrl+C.

---

### 4.1 Full Script Reference

| Order | Script | Platform | When to run |
|-------|--------|----------|-------------|
| 1 | `scripts/vgre_sync.sh` | Linux/macOS | Once per machine — build & install |
| 1 | `scripts\vgre_sync.bat` | Windows | Once per machine — build & install |
| 2 | `scripts/setup-cluster.sh` | Linux/macOS | Once per machine — generate/configure token |
| 2 | `.\scripts\Setup-VGRECluster.ps1` | Windows | Once per machine — generate/configure token |
| 3 | `scp ~/.vgre/token user@WORKER:~/.vgre/token` | Linux/macOS | Once — copy token from master to each worker |
| 4a | `vgre-start --master` | Linux/macOS | Every session — start the master dashboard |
| 4a | `.\scripts\Start-VGRE.ps1 --master` | Windows | Every session — start the master dashboard |
| 4b | `vgre-start --worker` | Linux/macOS | Every session — start a worker node |
| 4b | `.\scripts\Start-VGRE.ps1 --worker` | Windows | Every session — start a worker node |

> [!NOTE]
> AES-256 HMAC-authenticated encryption is always active when a token is configured. There is no separate `VGRE_CLUSTER_SECURE` variable — security is on by default.

---

### 4.2 How Auto-Discovery Works

The master broadcasts its presence over UDP on the local subnet every 2 seconds. Workers listen and connect automatically — **no manual IP configuration required** on the same subnet.

```
[Master machine]          UDP broadcast (port 7778)
  vgre-start --master  ─────────────────────────────►  [Worker machine]
                        ◄─────────────────────────────  vgre-start --worker
                         TCP connection (port 7777)
                           HMAC handshake + AES-256
                         ◄─────────────────────────────
                         Shared memory + kernel dispatch
```

For **cross-subnet clusters** where UDP broadcast does not reach, use `--master-ip` to specify the master address directly.

### 4.3 Cross-Platform Cluster Support

**Mixed Platform Clusters:**
- ✅ Linux Master + Windows Workers
- ✅ Windows Master + Linux Workers
- ✅ macOS Master + Linux/Windows Workers
- ✅ Any combination of platforms

**Security Features:**
- Hardware-backed token storage on all platforms:
  - **Linux**: Kernel keyring + GNOME Keyring + TPM 2.0
  - **Windows**: Credential Manager + TPM 2.0
  - **macOS**: Keychain + TPM 2.0
- AES-256-CTR encryption for all cluster communication
- HMAC-SHA256 handshake authentication

**Recent Fixes (April 2026):**
- ✅ Windows worker crash (exit code -1073741819) - Fixed WSA error handling
- ✅ WSAStartup/WSACleanup pairing - Fixed on all error paths
- ✅ TCP cluster authentication bypass - Fixed with HMAC-SHA256
- ✅ Secure channel token validation - Fixed with session key verification
- ✅ UDP discovery ports - Made configurable via env vars

### 4.4 Hybrid Authentication Mode (VGRE_CLUSTER_STRICT_AUTH)

VGRE supports two authentication modes for handling token mismatches between master and worker nodes:

**Fallback Mode (Default):**
- When a token mismatch is detected, the connection retries using a shared default key. The channel remains AES-256 encrypted at all times — no plaintext connection is ever used.
- Leave `VGRE_CLUSTER_STRICT_AUTH` unset or set to `0`.

**Strict Mode:**
- When a token mismatch is detected, the connection is rejected immediately.
- Set `VGRE_CLUSTER_STRICT_AUTH=1` on all nodes for production.

---

### 4.5 Cross-Platform Cluster Support

**Mixed Platform Clusters:**
- ✅ Linux Master + Windows Workers
- ✅ Windows Master + Linux Workers
- ✅ macOS Master + Linux/Windows Workers
- ✅ Any combination of platforms

**Testing Cross-Platform Connections:**
1. Ensure all machines have the same token (see Section 4.4)
2. Run `vgre-start --master` on the master machine
3. Run `vgre-start --worker` on each worker machine
4. Verify the master dashboard shows worker CPU and RAM stats

**Platform-Specific Notes:**

**Linux → Windows:**
- No special configuration needed
- Windows worker must have `vgre-worker.exe` allowed through Windows Firewall

**Windows → Linux:**
- No special configuration needed
- Linux worker must have port 7777 open in firewall

**Windows → Windows:**
- Both machines must have `vgre-worker.exe` allowed through Windows Firewall
- Ensure both machines are on the same subnet for auto-discovery

**macOS → Any Platform:**
- No special configuration needed
- macOS Keychain integration for secure token storage

### 4.6 Troubleshooting Token Mismatches

The most common cluster problem is mismatched tokens (different token on master vs worker). Use the fingerprint to diagnose — **do not compare the raw token strings**; compare the SHA256 fingerprint which is shorter and harder to misread.

**Step 1 — Check the fingerprint on each node:**

```bash
# Linux/macOS (run on master, then on each worker)
bash scripts/setup-cluster.sh --show-fingerprint
```
```powershell
# Windows (run on master, then on each worker)
.\scripts\Setup-VGRECluster.ps1 -ShowFingerprint
```

All nodes must print **identical** SHA256 output. If any differ:

**Step 2 — Re-copy the master's token to the worker:**
```bash
# Linux/macOS
scp ~/.vgre/token  user@WORKER_IP:~/.vgre/token

# Windows
scp "$env:USERPROFILE\.vgre\token"  user@WORKER_IP:"~/.vgre/token"
```

**Step 3 — If you pasted the token manually** (e.g., via Notepad), re-run setup option 2 on the worker and paste carefully. The token must be exactly 64 hex characters, no spaces, no newline. The setup script strips stray whitespace automatically.

**Step 4 — Verify again**, then restart the worker:
```bash
bash scripts/setup-cluster.sh --show-fingerprint   # must match master
vgre-start --worker
```

**Step 5 — Worker exits immediately on Windows:**
1. Run `.\scripts\vgre_sync.bat` to ensure all DLLs are installed
2. Check that `%LOCALAPPDATA%\VGRE\lib\libvgre.dll` exists
3. Allow `vgre-worker.exe` through Windows Firewall
4. Run with full diagnostic output:
   ```powershell
   & "$env:LOCALAPPDATA\VGRE\vgre-worker.exe" --port 7777
   ```

---

### 4.7 Manual Token Setup (Advanced — CI/containers)

If you cannot use the setup script (e.g., Docker container, CI pipeline), set the token inline. This token is only active for the current terminal session.

**Linux/macOS:**
```bash
export VGRE_TCP_AUTH_TOKEN="my-secret-key"
vgre-worker --port 7777
```

**Windows:**
```powershell
$env:VGRE_TCP_AUTH_TOKEN = "my-secret-key"
.\scripts\Start-VGRE.ps1 --worker
```

> [!WARNING]
> The inline token is visible in the process list (`ps aux` / Task Manager) and is lost when the terminal closes. Use the setup script for persistent, secure storage.

**Cross-subnet with manual token:**
```bash
# Linux/macOS
export VGRE_TCP_AUTH_TOKEN="my-secret-key"
export VGRE_CLUSTER_NODES="192.168.1.50:7777"
./my_cuda_app
```

---

### 4.7 Complete Environment Variable Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `VGRE_TCP_AUTH_TOKEN` | — | Cluster auth token value (visible in process list — prefer the file) |
| `VGRE_TCP_AUTH_TOKEN_FILE` | — | Path to file containing auth token (set automatically by setup script) |
| `VGRE_CLUSTER_NODES` | — | Manual node list: `192.168.1.50:7777,10.0.0.100:7777` |
| `VGRE_CLUSTER_STRICT_AUTH` | `0` | Set to `1` to reject mismatched-token connections (production) |
| `VGRE_ALLOW_AUTH_FALLBACK` | `0` | Set to `1` to allow fallback to default key on mismatch (development) |
| `VGRE_CLUSTER_UDP_ANNOUNCE_PORT` | `7778` | UDP port master broadcasts presence on |
| `VGRE_CLUSTER_UDP_WORKER_PORT` | `7779` | UDP port workers broadcast presence on |
| `VGRE_CLUSTER_MASTER_IP` | — | Comma-separated IP allowlist for worker security |
| `VGRE_CLUSTER_BANDWIDTH_REPROBE_SEC` | `300` | Bandwidth re-probe interval in seconds |
| `VGRE_PBKDF2_ITERATIONS` | `600000` | PBKDF2 iteration count for session key derivation |
| `VGRE_LOG_LEVEL` | `INFO` | Log verbosity: `DEBUG`, `INFO`, `WARN`, `ERROR` |
| `VGRE_CACHE_DIR` | `~/.vgre/cache` | JIT kernel compilation cache directory |
| `VGRE_DEVICE_COUNT` | auto | Override auto-detected virtual device count |
| `VGRE_ENABLE_NUMA` | `0` | Set to `1` to enable NUMA-aware scheduling (Linux only) |
| `VGRE_WORKER_THREADS` | auto | Override worker thread count (0 = auto-detect) |
| `VGRE_SIMD_LEVEL` | auto | Force SIMD level: `SSE4`, `AVX`, `AVX2`, `AVX512`, `native` |
| `VGRE_ADAPTIVE_ALPHA` | `0.3` | Exponential moving-average alpha for adaptive engine |
| `VGRE_HYBRID_REBALANCE_INTERVAL_MS` | — | Auto-start hybrid rebalancing loop with this interval in ms |
| `VGRE_ENABLE_NATIVE_SIMD` | `0` | Set to `1` during build to enable `-march=native` SIMD tuning |

---

## 5. Platform-Specific Features

### 5.1 Linux-Specific Features

**NUMA Awareness:**
```bash
export VGRE_ENABLE_NUMA=1
export VGRE_NUMA_POLICY=interleave  # local, interleave, preferred
```

**Hardware Token Storage Priority:**
1. Linux Keyring (kernel keyutils) - Highest security
2. GNOME Keyring/KWallet (libsecret) - Desktop integration
3. TPM 2.0 - Hardware security module
4. Encrypted file fallback - Cross-platform compatibility

**Performance Monitoring:**
- Real instruction counting via `perf_event` API
- CPU temperature monitoring from `/sys/class/thermal/`
- NUMA topology detection and optimization

### 5.2 Windows-Specific Features

**Hardware Token Storage:**
- Windows Credential Manager integration
- TPM 2.0 support for enterprise environments
- Encrypted file fallback for compatibility

**Networking:**
- WinSock2 optimizations for cluster communication
- Windows-specific socket options and error handling
- Vectored Exception Handler for UVM page faults

**Build Integration:**
- Visual Studio project generation
- MSBuild integration
- Windows SDK compatibility

### 5.3 macOS-Specific Features

**Apple Silicon Optimization:**
- Native ARM64 support with NEON SIMD
- Unified memory model optimizations
- Metal Performance Shaders integration (future)

**Hardware Token Storage:**
- macOS Keychain integration via Security framework
- Touch ID/Face ID authentication support (future)
- TPM 2.0 support on Intel Macs

**System Integration:**
- IOKit temperature monitoring
- macOS-specific CPU detection via sysctl
- App Bundle support for dashboard

### 5.4 Cross-Platform Features

**Available on All Platforms:**
- LLVM JIT kernel compilation
- OpenMP parallelization
- TCP cluster networking with AES-256 encryption
- Unified Virtual Memory (UVM) with page fault handling
- CUDA Graphs support
- Texture and surface memory operations
- Event synchronization
- Stream management

---

## 6. Troubleshooting

### 6.1 Common Issues (All Platforms)

| Issue | Potential Cause | Solution |
|-------|----------------|----------|
| `cudaErrorNoDevice` | VGRE Driver not found | Ensure library path is configured correctly |
| `cudaErrorInvalidValue` | Pointer out of bounds | Check kernel arguments; VGRE performs strict bounds checking |
| Low Performance | JIT Cache miss | First run is slower due to compilation; subsequent runs use cache |
| Dashboard not updating | IPC Socket error | Check IPC socket permissions and connection limits |

### 6.2 Linux-Specific Issues

| Issue | Potential Cause | Solution |
|-------|----------------|----------|
| `libvgre_cudart.so: not found` | Library path not set | `export LD_LIBRARY_PATH=/usr/local/lib/vgre:$LD_LIBRARY_PATH` |
| NUMA warnings | NUMA not available | Normal on single-socket systems; can be ignored |
| Permission denied on keyring | User keyring not initialized | `keyctl new_session` or use fallback storage |

### 6.3 Windows-Specific Issues

| Issue | Potential Cause | Solution |
|-------|----------------|----------|
| `vgre.dll not found` | DLL not in PATH | Add `C:\Program Files\VGRE\bin` to PATH or copy to app directory |
| Build fails with MSVC | Wrong Visual Studio version | Use Visual Studio 2022 with C++ workload |
| Worker connection fails | Windows Firewall | Allow `vgre-worker.exe` through Windows Firewall |

### 6.4 macOS-Specific Issues

| Issue | Potential Cause | Solution |
|-------|----------------|----------|
| `dylib not loaded` | Library not signed | `codesign -s - /usr/local/lib/vgre/libvgre.dylib` |
| OpenMP not found | Homebrew libomp missing | `brew install libomp` |
| Keychain access denied | App not authorized | Grant keychain access in System Preferences |

### 6.5 Cluster-Specific Issues

| Issue | Potential Cause | Solution |
|-------|----------------|----------|
| Workers not discovered | Different subnets | Use `--master-ip` flag or `VGRE_CLUSTER_NODES` |
| Authentication failures / handshake error | Token mismatch | Run `setup-cluster.sh --show-fingerprint` on each node — fingerprints must match |
| Worker exits immediately on Windows | Missing DLLs | Run `vgre_sync.bat` to reinstall; allow `vgre-worker.exe` through Firewall |
| Different fingerprint after copy-paste | Extra whitespace in token file | Re-run `Setup-VGRECluster.ps1` option 2 — script strips whitespace automatically |
| Worker shows CPU/RAM as 0 after connect | Reconnected before handshake finished | Wait 2–3 seconds after connect; values update automatically |
| Mixed platform issues | Network/firewall | All platforms are protocol-compatible; check port 7777 is open |

### 6.6 Performance Optimization

**Linux:**
```bash
# Enable all optimizations
export VGRE_ENABLE_NUMA=1
export VGRE_SIMD_LEVEL=native
export VGRE_WORKER_THREADS=$(nproc)
```

**Windows:**
```powershell
# Enable all optimizations
$env:VGRE_SIMD_LEVEL="native"
$env:VGRE_WORKER_THREADS=[Environment]::ProcessorCount
```

**macOS:**
```bash
# Enable all optimizations
export VGRE_SIMD_LEVEL=native
export VGRE_WORKER_THREADS=$(sysctl -n hw.ncpu)
```

---

## 7. Advanced Configuration

### 7.1 Build Options

**Cross-Platform CMake Options:**
```bash
# Enable native SIMD optimizations (build machine specific)
cmake .. -DVGRE_ENABLE_NATIVE_SIMD=ON

# Disable OpenCL backend (Windows default)
cmake .. -DVGRE_USE_OPENCL_BACKEND=OFF

# Enable debug build with full logging
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Custom installation prefix
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/vgre
```

**Platform-Specific Build Options:**

**Linux:**
```bash
# Enable all optional features
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DVGRE_ENABLE_NATIVE_SIMD=ON \
         -DVGRE_USE_OPENCL_BACKEND=ON

# Minimal build (no optional dependencies)
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DVGRE_USE_OPENCL_BACKEND=OFF
```

**Windows:**
```powershell
# Full feature build
cmake .. -G "Visual Studio 17 2022" -A x64 `
         -DCMAKE_BUILD_TYPE=Release `
         -DVGRE_ENABLE_NATIVE_SIMD=ON

# Static runtime linking
cmake .. -G "Visual Studio 17 2022" -A x64 `
         -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
```

**macOS:**
```bash
# Universal binary (Intel + Apple Silicon)
cmake .. -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" \
         -DCMAKE_BUILD_TYPE=Release

# Apple Silicon only
cmake .. -DCMAKE_OSX_ARCHITECTURES=arm64 \
         -DCMAKE_BUILD_TYPE=Release
```

### 7.2 Runtime Configuration Files

**Linux/macOS:** `~/.vgre/config.json`
**Windows:** `%USERPROFILE%\.vgre\config.json`

```json
{
  "runtime": {
    "log_level": "INFO",
    "cache_dir": "~/.vgre/cache",
    "max_devices": 8,
    "enable_profiling": true
  },
  "cluster": {
    "auto_discovery": true,
    "port": 7777,
    "secure_by_default": true,
    "timeout_ms": 5000
  },
  "performance": {
    "numa_enabled": true,
    "simd_level": "auto",
    "worker_threads": 0
  }
}
```

---

For more details, refer to the [Technical Architecture](architecture.md) or the [API Reference](api_reference.md).
---

---

## Production Status

**Status**: ✅ **PRODUCTION READY** as of May 6, 2026

All critical issues have been fixed. The system is ready for production deployment.

**See docs/PROJECT_STATUS.md for canonical project status and test results.**