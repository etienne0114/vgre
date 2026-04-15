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

VGRE allows you to aggregate multiple network-connected CPUs into a single Virtual GPU cluster using a high-performance **Synchronized Discovery** protocol. This works across mixed platforms (Linux, Windows, macOS).

### 4.1 Start Worker Nodes

**Linux/macOS:**
```bash
# Start worker with authentication
vgre-worker --auth-token my-secret-key

# Advanced options
vgre-worker --auth-token my-secret-key --port 7777 --threads 16
```

**Windows:**
```powershell
# Start worker with authentication
vgre-worker.exe --auth-token my-secret-key

# Advanced options
vgre-worker.exe --auth-token my-secret-key --port 7777 --threads 16
```

Workers automatically:
- Listen on port **7777** (configurable)
- Broadcast availability to Master nodes on the local subnet
- Support secure AES-256 encrypted channels
- Auto-detect optimal thread count based on CPU cores

### 4.2 Master Node Configuration

**Automatic Discovery (Same Subnet):**
The Master will proactively scan the network and connect to available workers automatically—**no manual IP configuration required**.

**Linux/macOS:**
```bash
export VGRE_TCP_AUTH_TOKEN="my-secret-key"
export VGRE_CLUSTER_SECURE=1  # Enable encryption
./my_cuda_app
```

**Windows:**
```powershell
$env:VGRE_TCP_AUTH_TOKEN="my-secret-key"
$env:VGRE_CLUSTER_SECURE="1"  # Enable encryption
.\my_cuda_app.exe
```

**Manual Connection (Cross-Subnet):**
If nodes are on different subnets, specify them manually:

**Linux/macOS:**
```bash
export VGRE_CLUSTER_NODES="192.168.1.50:7777,10.0.0.100:7777"
export VGRE_TCP_AUTH_TOKEN="my-secret-key"
./my_cuda_app
```

**Windows:**
```powershell
$env:VGRE_CLUSTER_NODES="192.168.1.50:7777,10.0.0.100:7777"
$env:VGRE_TCP_AUTH_TOKEN="my-secret-key"
.\my_cuda_app.exe
```

### 4.3 Cross-Platform Cluster Support

**Mixed Platform Clusters:**
- ✅ Linux Master + Windows Workers
- ✅ Windows Master + Linux Workers  
- ✅ macOS Master + Linux/Windows Workers
- ✅ Any combination of platforms

**Platform-Specific Optimizations:**
- **Linux**: NUMA-aware scheduling and memory binding
- **Windows**: Windows-specific networking optimizations
- **macOS**: Unified memory model optimizations for Apple Silicon

**Security Features:**
- Hardware-backed token storage on all platforms:
  - **Linux**: Kernel keyring + GNOME Keyring + TPM 2.0
  - **Windows**: Credential Manager + TPM 2.0
  - **macOS**: Keychain + TPM 2.0
- AES-256-CTR encryption for all cluster communication
- Automatic security synchronization across mixed platforms

> [!TIP]
> **Security Synchronization**: If you toggle the "Secure cluster channel" in the dashboard, any discovered workers will automatically detect the change and restart their handshake to establish an encrypted tunnel, regardless of their platform.

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
| Workers not discovered | Different subnets | Use manual `VGRE_CLUSTER_NODES` configuration |
| Authentication failures | Token mismatch | Ensure all nodes use same `VGRE_TCP_AUTH_TOKEN` |
| Mixed platform issues | Platform incompatibility | All platforms are compatible; check network connectivity |

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
