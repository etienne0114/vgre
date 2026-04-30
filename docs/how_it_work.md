# How VGRE Works

**Version**: 0.1.2 — Last updated: 2026-04-30

---

## What Is VGRE?

**VGRE** (Virtual GPU Runtime Engine) lets you run GPU applications **without a physical GPU**. Instead of buying an expensive NVIDIA graphics card, VGRE translates GPU code into CPU code and runs it natively on your processor.

**Think of it like this**: If a GPU application is a recipe written in French (CUDA), VGRE is a translator that rewrites it into English (CPU instructions) so your computer can follow the recipe — and it actually cooks the meal, not just pretends to.

> **Key principle**: VGRE is NOT a simulator. It does not fake results. Every computation runs for real on your CPU, producing identical outputs to a physical GPU.

---

## Glossary — Abbreviations Used in the Codebase

| Abbreviation | Full Name | What It Means |
|---|---|---|
| **VGRE** | Virtual GPU Runtime Engine | The project itself — runs GPU code on CPUs |
| **CUDA** | Compute Unified Device Architecture | NVIDIA's programming language for GPUs |
| **JIT** | Just-In-Time (Compilation) | Compiling code right before it runs, not ahead of time |
| **LLVM** | Low Level Virtual Machine | A compiler framework that translates code into native CPU instructions |
| **ORC** | On-Request Compilation | LLVM's JIT engine that compiles code when needed |
| **IR** | Intermediate Representation | A middle-step format between source code and machine code |
| **UVM** | Unified Virtual Memory | Memory shared between "GPU" and "CPU" — accessed by both without manual copying |
| **SHM** | Shared Memory | Memory visible to all threads within a block (fast scratchpad) |
| **NUMA** | Non-Uniform Memory Access | Multi-socket CPU architecture where memory access speed depends on which CPU socket owns the memory |
| **IPC** | Inter-Process Communication | Mechanisms for separate processes to share data (e.g., shared memory segments) |
| **SIMD** | Single Instruction, Multiple Data | CPU feature that processes multiple values in one operation (e.g., 8 floats at once with AVX2) |
| **AVX** | Advanced Vector Extensions | Intel/AMD SIMD instruction sets (AVX, AVX2, AVX-512) |
| **TLS** | Thread-Local Storage | Data private to each thread — prevents conflicts between parallel workers |
| **VSBP** | VGRE Structured Binary Protocol | Network protocol for cluster communication between VGRE nodes |
| **HMAC** | Hash-based Message Authentication Code | Cryptographic signature proving a message hasn't been tampered with |
| **AES** | Advanced Encryption Standard | Industry-standard encryption algorithm (AES-256-CTR used for cluster traffic) |
| **PBKDF2** | Password-Based Key Derivation Function 2 | Algorithm that derives a strong encryption key from a password/token |
| **EWMA** | Exponential Weighted Moving Average | Smoothing technique that gives more weight to recent measurements |
| **GFLOPS** | Giga Floating-Point Operations Per Second | Measure of computation speed (billions of math operations per second) |
| **VEH** | Vectored Exception Handler | Windows mechanism for catching memory access violations (like SIGSEGV on Linux) |
| **OTLP** | OpenTelemetry Protocol | Standard format for exporting performance traces to monitoring tools |
| **DAG** | Directed Acyclic Graph | A workflow where tasks have dependencies but no circular loops |
| **RCU** | Read-Copy-Update | Lock-free technique for safely reading shared data while another thread updates it |
| **LOD** | Level of Detail | Selecting coarser/finer texture resolution based on distance |
| **SM** | Streaming Multiprocessor | GPU hardware unit that runs groups of threads — VGRE maps this to CPU core capability |

---

## The Big Picture — How Everything Connects

```mermaid
graph TB
    subgraph "Your Application"
        APP["PyTorch / TensorFlow / Custom CUDA App"]
    end

    subgraph "VGRE Engine"
        SHIM["API Layer\n(Intercepts CUDA Calls)"]
        PARSE["Kernel Parser\n(Understands GPU Code)"]
        COMPILE["JIT Compiler\n(Translates to CPU Code)"]
        EXEC["Parallel Executor\n(Runs on CPU Cores)"]
        MEM["Memory Manager\n(Handles Allocations)"]
        SCHED["Scheduler\n(Organizes Work)"]
        DEVICE["Virtual GPU Device\n(Reports Hardware Info)"]
        ADAPT["Adaptive Engine\n(Measures Performance)"]
    end

    subgraph "Output"
        RESULT["Real Computed Results"]
        DASH["Dashboard\n(Live Performance Metrics)"]
        CLUSTER["Cluster Nodes\n(Distributed Execution)"]
    end

    APP --> SHIM
    SHIM --> PARSE
    PARSE --> COMPILE
    COMPILE --> EXEC
    EXEC --> RESULT
    SHIM --> MEM
    SHIM --> DEVICE
    EXEC --> SCHED
    EXEC --> ADAPT
    ADAPT --> DASH
    EXEC --> CLUSTER

    style APP fill:#4a90d9,color:#fff
    style SHIM fill:#e74c3c,color:#fff
    style PARSE fill:#f39c12,color:#fff
    style COMPILE fill:#f39c12,color:#fff
    style EXEC fill:#27ae60,color:#fff
    style MEM fill:#8e44ad,color:#fff
    style RESULT fill:#2ecc71,color:#fff
    style DASH fill:#3498db,color:#fff
    style CLUSTER fill:#3498db,color:#fff
```

---

## Step-by-Step: What Happens When You Run a GPU Program

### Step 1 — Your App Calls CUDA

Your application calls `cudaLaunchKernel(...)` thinking it's talking to a real GPU.

VGRE intercepts this call using `LD_PRELOAD` (Linux) or DLL injection (Windows). The app never knows the difference.

### Step 2 — Parsing the Kernel

The **Kernel Parser** reads the GPU code (written in CUDA C) and extracts:
- What arguments the function takes (pointers, integers, floats)
- Whether it uses shared memory (fast scratchpad memory within a block)
- Whether it uses `__syncthreads()` (barrier synchronization between threads)

### Step 3 — Compiling to CPU Code (JIT)

The **LLVM Translation Engine** converts GPU code into native CPU code through 4 stages:

```mermaid
flowchart LR
    A["CUDA Kernel\n(GPU Code)"] --> B["Clang Parser\n(Extract Structure)"]
    B --> C["Wrapper Generator\n(Bridge GPU→CPU)"]
    C --> D["Clang Compiler\n(-O3 Optimization)"]
    D --> E["LLVM JIT\n(Native Machine Code)"]
    E --> F["Ready to Execute"]

    style A fill:#e74c3c,color:#fff
    style B fill:#f39c12,color:#fff
    style C fill:#f39c12,color:#fff
    style D fill:#f39c12,color:#fff
    style E fill:#27ae60,color:#fff
    style F fill:#2ecc71,color:#fff
```

**Caching**: Once compiled, the result is saved to disk (`~/.vgre/cache/`). Next time the same kernel runs, it loads instantly (0ms) instead of recompiling.

### Step 4 — Executing on CPU Cores

The **CPU Parallel Executor** maps GPU concepts to CPU resources:

```mermaid
flowchart TB
    subgraph "GPU Concept"
        GRID["Grid\n(thousands of blocks)"]
        BLOCK["Block\n(group of threads)"]
        THREAD["Thread\n(single worker)"]
        SMEM["Shared Memory\n(per-block scratchpad)"]
    end

    subgraph "CPU Reality"
        OMP["OpenMP Parallel Loop\n(spread across CPU cores)"]
        SIMDL["SIMD Vectorization\n(AVX2/AVX-512 lanes)"]
        WORKER["Worker Thread\n(OS thread)"]
        BUFFER["Allocated Buffer\n(per-thread memory)"]
    end

    GRID --> OMP
    BLOCK --> SIMDL
    THREAD --> WORKER
    SMEM --> BUFFER

    style GRID fill:#e74c3c,color:#fff
    style BLOCK fill:#e74c3c,color:#fff
    style THREAD fill:#e74c3c,color:#fff
    style SMEM fill:#e74c3c,color:#fff
    style OMP fill:#27ae60,color:#fff
    style SIMDL fill:#27ae60,color:#fff
    style WORKER fill:#27ae60,color:#fff
    style BUFFER fill:#27ae60,color:#fff
```

### Step 5 — Results

The computed results are written directly to host memory. Since VGRE runs on the CPU, there's no "GPU-to-CPU transfer" — the results are already where your app expects them.

---

## How Memory Works

### Regular Memory (`cudaMalloc`)

Simple: allocates aligned memory on the host. Your app writes to it, reads from it — no tricks needed.

### Unified Virtual Memory — UVM (`cudaMallocManaged`)

UVM is the clever part. It lets both "GPU code" and "CPU code" access the same memory address without manual copying.

```mermaid
flowchart TB
    ALLOC["1. App calls cudaMallocManaged()"] --> RESERVE["2. VGRE reserves memory\n(marked as NO ACCESS)"]
    RESERVE --> ACCESS["3. App tries to read/write"]
    ACCESS --> FAULT["4. Page Fault!\n(OS catches the violation)"]
    FAULT --> HANDLER["5. VGRE's Signal Handler\n(SIGSEGV on Linux / VEH on Windows)"]
    HANDLER --> UNLOCK["6. Unlock the page\n(mark as READ+WRITE)"]
    UNLOCK --> TRACK["7. Record access\n(dirty bitmap + timestamp)"]
    TRACK --> CONTINUE["8. App continues normally\n(doesn't know anything happened)"]

    style ALLOC fill:#3498db,color:#fff
    style FAULT fill:#e74c3c,color:#fff
    style HANDLER fill:#f39c12,color:#fff
    style UNLOCK fill:#27ae60,color:#fff
    style CONTINUE fill:#2ecc71,color:#fff
```

**Why?** This gives VGRE precise knowledge of which memory pages are being used, when, and by which part of the program — enabling smart decisions about where to place memory on multi-socket (NUMA) systems.

### Shared Memory — SHM (`__shared__`)

In real GPUs, shared memory is a fast scratchpad visible to all threads in a block. VGRE implements this as a pre-allocated buffer that's pointer-swizzled into the kernel at compile time:

- `__shared__ float s[256]` → becomes a pointer into the buffer: `float* s = (float*)(buffer + offset)`
- Each OpenMP thread gets its own buffer — no conflicts between blocks running in parallel

### Memory Pools (`cudaMallocAsync`)

For apps that allocate/free memory rapidly, VGRE maintains free-lists that reuse blocks instead of calling the OS allocator each time. This avoids the overhead of thousands of `malloc`/`free` calls per second.

---

## How Rendering and Graphics Are Supported

VGRE supports texture and surface operations used in graphics and machine learning:

```mermaid
flowchart LR
    subgraph "Texture Pipeline"
        CREATE["Create Texture\n(cudaCreateTextureObject)"]
        UPLOAD["Upload Image Data\n(cudaMallocArray + memcpy)"]
        SAMPLE["Sample in Kernel\n(tex2D, tex3D)"]
        FILTER["Filtering\n(Nearest / Bilinear / Cubic)"]
    end

    subgraph "Advanced Features"
        MIP["Mipmaps\n(Multi-resolution)"]
        ANISO["Anisotropic Filtering\n(1x-16x quality)"]
        TRILI["Trilinear Blending\n(Between mip levels)"]
        SURF["Surface Read/Write\n(surf2Dread, surf2Dwrite)"]
    end

    CREATE --> UPLOAD --> SAMPLE --> FILTER
    FILTER --> MIP
    FILTER --> ANISO
    MIP --> TRILI
    SAMPLE --> SURF

    style CREATE fill:#8e44ad,color:#fff
    style SAMPLE fill:#27ae60,color:#fff
    style MIP fill:#3498db,color:#fff
    style ANISO fill:#3498db,color:#fff
```

**Texture fetching** (e.g., `tex2D(texture, x, y)`) is implemented through built-in functions that the JIT compiler resolves at compile time. The `TextureManager` stores image data in host memory and performs real filtering math (bilinear interpolation, cubic Catmull-Rom, anisotropic multi-sampling).

---

## How the Cluster Works (Multi-Machine Execution)

VGRE can distribute work across multiple machines over a network:

```mermaid
flowchart TB
    subgraph "Master Node"
        MASTER["Master\n(Coordinates Work)"]
        PART["Workload Partitioner\n(Splits Grid by CPU Power)"]
    end

    subgraph "Network (VSBP Protocol)"
        SEC["Encrypted Channel\n(AES-256 + HMAC-SHA256)"]
    end

    subgraph "Worker Nodes"
        W1["Worker 1\n(8 cores, 16GB RAM)"]
        W2["Worker 2\n(16 cores, 32GB RAM)"]
        W3["Worker 3\n(4 cores, 8GB RAM)"]
    end

    MASTER --> PART
    PART --> SEC
    SEC --> W1
    SEC --> W2
    SEC --> W3
    W1 --> |"Results + Dirty Pages"| MASTER
    W2 --> |"Results + Dirty Pages"| MASTER
    W3 --> |"Results + Dirty Pages"| MASTER

    style MASTER fill:#e74c3c,color:#fff
    style SEC fill:#f39c12,color:#fff
    style W1 fill:#27ae60,color:#fff
    style W2 fill:#27ae60,color:#fff
    style W3 fill:#27ae60,color:#fff
```

**How it works**:
1. **Discovery**: Nodes find each other via UDP broadcast (or manual `VGRE_CLUSTER_NODES` config)
2. **Authentication**: HMAC-SHA256 handshake verifies both sides have the same auth token
3. **Encryption**: All traffic encrypted with AES-256-CTR (keys rotate every 10,000 packets)
4. **Partitioning**: The master splits the kernel grid proportionally to each node's CPU power
5. **Execution**: Each node runs its slice and sends back results + modified memory pages
6. **SHM Fast Path**: When master and worker are on the same machine, data bypasses the network and uses shared memory (SHM) instead

---

## How CUDA Graphs Work

CUDA Graphs let apps define a **workflow once** and replay it many times without re-submitting individual operations:

```mermaid
flowchart LR
    subgraph "Graph Definition"
        K1["Kernel A"] --> K2["Kernel B"]
        K1 --> K3["Memcpy C"]
        K2 --> K4["Kernel D"]
        K3 --> K4
    end

    subgraph "Execution"
        INST["Instantiate Graph\n(Validate + Optimize)"]
        LAUNCH["Launch\n(Execute all nodes in order)"]
        REPLAY["Replay\n(Re-execute without overhead)"]
    end

    K4 --> INST --> LAUNCH --> REPLAY

    style K1 fill:#e74c3c,color:#fff
    style K2 fill:#e74c3c,color:#fff
    style K3 fill:#8e44ad,color:#fff
    style K4 fill:#e74c3c,color:#fff
    style LAUNCH fill:#27ae60,color:#fff
    style REPLAY fill:#2ecc71,color:#fff
```

VGRE supports: graph creation, node dependencies, cloning, serialization/deserialization (JSON), stream capture, conditional nodes (IF/WHILE), and graph updates.

---

## How Performance Measurement Works

VGRE doesn't guess performance — it **measures** it:

```mermaid
flowchart TB
    subgraph "Data Sources"
        PERF["perf_event API\n(Linux: real instruction count)"]
        TIMER["Wall-Clock Timer\n(Actual elapsed time)"]
        LLVMA["LLVM IR Analysis\n(Count math operations)"]
        BW["Bandwidth Probe\n(4MB memcpy benchmark)"]
    end

    subgraph "Adaptive Engine"
        EWMA_E["EWMA Smoothing\n(Reduces noise)"]
        AUTO["Auto-Tuning\n(Adjusts sensitivity)"]
        CALIB["Calibration Cache\n(Saved per-process)"]
    end

    subgraph "Output"
        GFLOPS_O["GFLOPS\n(Computation Speed)"]
        BWOUT["Bandwidth GB/s\n(Memory Throughput)"]
        LATENCY["Latency ms\n(Per-Kernel Timing)"]
        OTLP_O["OTLP Export\n(Jaeger / Grafana)"]
    end

    PERF --> EWMA_E
    TIMER --> EWMA_E
    LLVMA --> EWMA_E
    BW --> CALIB
    EWMA_E --> AUTO
    AUTO --> GFLOPS_O
    CALIB --> BWOUT
    EWMA_E --> LATENCY
    LATENCY --> OTLP_O

    style PERF fill:#27ae60,color:#fff
    style TIMER fill:#27ae60,color:#fff
    style LLVMA fill:#f39c12,color:#fff
    style GFLOPS_O fill:#3498db,color:#fff
    style BWOUT fill:#3498db,color:#fff
```

---

## How Synchronization Works

GPU programming has multiple synchronization barriers. Here's how VGRE implements each:

| GPU Concept | What It Does | VGRE Implementation |
|---|---|---|
| `__syncthreads()` | Waits for all threads in a block | Sense-reversing barrier in BlockWorkerPool |
| `this_grid().sync()` | Waits for all blocks in the entire grid | GridBarrierState — atomic counter + condition variable |
| `cudaStreamSynchronize()` | Waits for all work in a stream to finish | Scheduler drains the per-stream task queue |
| `cudaDeviceSynchronize()` | Waits for ALL work on ALL streams | Scheduler drains every queue |
| `cudaEventSynchronize()` | Waits for a specific recorded event | Event flag checked via atomic variable |

---

## Cross-Platform Support

VGRE runs on **Linux**, **Windows**, and **macOS**. The engine automatically adapts to each platform:

```mermaid
flowchart TB
    subgraph "VGRE Engine (Cross-Platform Core)"
        CORE["Shared Logic\n(Compilation, Execution, Scheduling)"]
    end

    subgraph "Linux"
        L1["SIGSEGV handler\n(UVM page faults)"]
        L2["perf_event\n(Instruction counting)"]
        L3["NUMA + mbind\n(Memory placement)"]
        L4["Linux Keyring\n(Token storage)"]
    end

    subgraph "Windows"
        W1["VEH handler\n(UVM page faults)"]
        W2["Performance counters"]
        W3["WinSock2\n(Networking)"]
        W4["Credential Manager\n(Token storage)"]
    end

    subgraph "macOS"
        M1["SIGSEGV handler\n(UVM page faults)"]
        M2["IOKit\n(Temperature monitoring)"]
        M3["SO_NOSIGPIPE\n(Socket safety)"]
        M4["Keychain\n(Token storage)"]
    end

    CORE --> L1 & L2 & L3 & L4
    CORE --> W1 & W2 & W3 & W4
    CORE --> M1 & M2 & M3 & M4

    style CORE fill:#3498db,color:#fff
    style L1 fill:#27ae60,color:#fff
    style L2 fill:#27ae60,color:#fff
    style L3 fill:#27ae60,color:#fff
    style L4 fill:#27ae60,color:#fff
    style W1 fill:#e74c3c,color:#fff
    style W2 fill:#e74c3c,color:#fff
    style W3 fill:#e74c3c,color:#fff
    style W4 fill:#e74c3c,color:#fff
    style M1 fill:#f39c12,color:#fff
    style M2 fill:#f39c12,color:#fff
    style M3 fill:#f39c12,color:#fff
    style M4 fill:#f39c12,color:#fff
```

---

## Key Configuration Options

| Variable | Default | What It Controls |
|---|---|---|
| `VGRE_DEVICE_COUNT` | auto | How many virtual GPUs to create |
| `VGRE_LOG_LEVEL` | `INFO` | How much detail to print (`DEBUG` for troubleshooting) |
| `VGRE_CACHE_DIR` | `~/.vgre/cache` | Where compiled kernels are cached |
| `VGRE_UVM_MIGRATION_MS` | `500` | How often to check for memory migration opportunities |
| `VGRE_ADAPTIVE_ALPHA` | `0.3` | How quickly performance predictions adapt to changes |
| `VGRE_TCP_AUTH_TOKEN_FILE` | — | Path to the cluster authentication token file |
| `VGRE_MESH_PEERS` | — | List of peer nodes for mesh networking |
| `VGRE_IPC_MODE` | `ON` | Enable/disable inter-process shared memory |
| `VGRE_BLOCK_THREADS` | `false` | Force multi-threaded block execution |
| `VGRE_GPU_PEAK_BANDWIDTH_GBPS` | `900` | Expected GPU bandwidth for telemetry comparison |

---

## Summary — Why VGRE Is Different

| Feature | Traditional Emulator | VGRE |
|---|---|---|
| Execution | Simulates GPU behavior | **Runs real code on CPU** |
| Results | May be approximate | **Bit-identical to GPU** |
| Performance data | Estimated/faked | **Measured from hardware** |
| Memory tracking | Simulated | **Real OS page faults** |
| Multi-machine | Rarely supported | **Encrypted cluster with auto-discovery** |
| Platform support | Usually Linux only | **Linux + Windows + macOS** |

---

## What VGRE Can and Cannot Do

### ✅ What VGRE Is Designed For

VGRE intercepts **CUDA and OpenCL compute APIs** — the math/science side of GPU programming:

| Use Case | How VGRE Helps |
|---|---|
| **AI / Machine Learning** (PyTorch, TensorFlow) | Run training and inference without a GPU — ideal for development, testing, CI/CD |
| **Scientific Computing** (CUDA kernels, simulations) | Develop and debug GPU code on any machine |
| **Data Processing** (cuBLAS, custom CUDA) | Process data with CUDA APIs on CPU hardware |
| **Education & Learning** | Learn CUDA programming without buying expensive hardware |
| **CI/CD Pipelines** | Run GPU test suites on cloud servers that have no GPUs |
| **Cluster Computing** | Distribute compute work across multiple CPU-only machines |
| **Algorithm Prototyping** | Test CUDA algorithms for correctness before deploying to real GPUs |

### ⚠️ What VGRE Can Run But Will Be Slower

For pure CUDA compute renderers, VGRE can execute the kernels — but a CPU with 16 cores **cannot match** a GPU with 10,000+ cores for massively parallel rendering:

| Software | Renderer | VGRE Status |
|---|---|---|
| **Blender** | Cycles (CUDA mode) | Runs correctly but **10-100x slower** than a real GPU. Blender already has a built-in CPU renderer — use that instead. |
| **V-Ray** | CUDA RT engine | Same — runs but impractical for production renders |
| **Arnold** | GPU mode (CUDA) | Same — use Arnold's native CPU mode instead |
| **OctaneRender** | CUDA only | Could run but extremely slow — not practical |

> **Bottom line**: If a renderer already has a "CPU mode", use that — it will be faster than routing CUDA through VGRE, because the CPU mode is optimized for CPU architecture while CUDA code is optimized for GPU architecture.

### ❌ What VGRE Cannot Do (Not Supported)

VGRE does **not** intercept graphics rendering APIs:

```mermaid
flowchart TB
    subgraph "GPU Work Types"
        direction LR
        COMPUTE["CUDA / OpenCL\n(Math & Compute)"]
        GRAPHICS["OpenGL / Vulkan / DirectX\n(Graphics & Rendering)"]
        VIDEO["NVENC / NVDEC\n(Video Encode/Decode)"]
        RT["RT Cores\n(Hardware Ray Tracing)"]
    end

    subgraph "VGRE Coverage"
        YES["✅ Fully Supported"]
        NO["❌ Not Supported"]
    end

    COMPUTE --> YES
    GRAPHICS --> NO
    VIDEO --> NO
    RT --> NO

    style COMPUTE fill:#27ae60,color:#fff
    style GRAPHICS fill:#e74c3c,color:#fff
    style VIDEO fill:#e74c3c,color:#fff
    style RT fill:#e74c3c,color:#fff
    style YES fill:#2ecc71,color:#fff
    style NO fill:#c0392b,color:#fff
```

| Not Supported | Why | Impact |
|---|---|---|
| **OpenGL / Vulkan / DirectX** | These are graphics APIs for drawing — VGRE doesn't intercept them | AutoCAD, SolidWorks, Blender viewport, games will NOT work through VGRE |
| **NVENC / NVDEC** | Hardware video encode/decode — no CPU equivalent | Video editing GPU acceleration won't work |
| **RT Cores** | Hardware ray tracing units — CPU has no equivalent | Hardware-accelerated ray tracing not available |
| **CUDA↔OpenGL Interop** | Sharing buffers between CUDA and OpenGL | Apps that mix compute and graphics won't work |

### Rendering Software Compatibility

| Software | Viewport (3D display) | Render Engine | VGRE Verdict |
|---|---|---|---|
| **AutoCAD** | DirectX/OpenGL ❌ | CPU-based ✅ | **Cannot support viewport.** Use CPU rendering (already built-in). |
| **SolidWorks** | DirectX ❌ | CPU-based ✅ | **Cannot support viewport.** Use built-in CPU mode. |
| **Blender** | OpenGL ❌ | Cycles has CPU mode ✅ | **Cannot support viewport.** Use Cycles CPU renderer directly. |
| **3ds Max** | DirectX ❌ | V-Ray/Arnold CPU ✅ | **Cannot support viewport.** Use CPU render plugins. |
| **Maya** | OpenGL ❌ | Arnold CPU ✅ | **Cannot support viewport.** Use Arnold CPU mode. |
| **DaVinci Resolve** | OpenGL + CUDA ❌ | GPU-accelerated ❌ | **Not compatible.** Requires real GPU. |

> **In simple terms**: VGRE is a **compute engine**, not a **graphics engine**. It excels at running math-heavy CUDA programs (AI, simulations, data science) on CPUs — but it cannot display 3D graphics or accelerate rendering viewports.

---

## Cluster Scenarios — Can I Use a Friend's GPU?

### Scenario: "My machine has no GPU. My friend's machine has an NVIDIA GPU. Can I use VGRE clustering to run my CUDA code on their GPU?"

**Answer: Not currently.** Here's what happens today vs. what would be needed:

```mermaid
flowchart LR
    subgraph "Today (CPU-to-CPU Cluster)"
        M1["Your Machine\n(Master, CPU only)"] -->|"Sends kernel\nvia TCP"| W1["Friend's Machine\n(Worker)"]
        W1 -->|"JIT compiles\nto CPU code"| CPU1["Runs on CPU\n(GPU is ignored ❌)"]
        CPU1 -->|"Results"| M1
    end

    style M1 fill:#3498db,color:#fff
    style W1 fill:#f39c12,color:#fff
    style CPU1 fill:#e74c3c,color:#fff
```

```mermaid
flowchart LR
    subgraph "Future Goal (CPU-to-GPU Cluster)"
        M2["Your Machine\n(Master, CPU only)"] -->|"Sends kernel\nvia TCP"| W2["Friend's Machine\n(Worker + GPU)"]
        W2 -->|"Detects NVIDIA GPU\nUses real CUDA"| GPU2["Runs on GPU ✅"]
        GPU2 -->|"Results"| M2
    end

    style M2 fill:#3498db,color:#fff
    style W2 fill:#27ae60,color:#fff
    style GPU2 fill:#2ecc71,color:#fff
```

### What the Cluster CAN Do Today

| Scenario | Supported | Benefit |
|---|---|---|
| CPU machine → CPU machine(s) | ✅ Yes | Spread work across multiple CPU-only machines |
| 2-core laptop → 64-core server (CPU) | ✅ Yes | Use a powerful server's CPU for heavy compute |
| Multiple office PCs combined | ✅ Yes | Pool idle machines into a compute cluster |
| Encrypted communication | ✅ Yes | All traffic is AES-256 encrypted + HMAC authenticated |

### What the Cluster CANNOT Do Yet

| Scenario | Supported | Why Not |
|---|---|---|
| CPU machine → Friend's GPU | ❌ Not yet | Worker always JIT-compiles to CPU, ignores physical GPU |
| Remote GPU rendering | ❌ Not yet | Would need real CUDA runtime on the worker side |
| GPU-to-GPU across network | ❌ Not yet | No GPU passthrough implemented |

### SHM Clarification

**SHM (Shared Memory) is NOT for cross-network GPU sharing.** It is a local optimization:

| SHM Scenario | What Happens |
|---|---|
| Master and Worker on **same machine** (127.0.0.1) | ✅ SHM used — data bypasses TCP, transferred via shared memory segment (256MB) for speed |
| Master and Worker on **different machines** | ❌ SHM not used — data transferred via TCP (encrypted) |
| Cross-network GPU access | ❌ SHM cannot do this — it's local-only |

### Future: GPU Passthrough Worker (Roadmap)

To enable the "use a friend's GPU" scenario, the VGRE worker would need:

1. **GPU Detection** — Check if the worker machine has a real NVIDIA GPU (`nvidia-smi`)
2. **Real CUDA Runtime Loading** — Dynamically load `libcuda.so` / `nvcuda.dll` 
3. **GPU Execution Path** — When a kernel arrives, send it to the real GPU instead of JIT-compiling to CPU
4. **Memory Transfer** — Copy input data to GPU memory, run kernel, copy results back, send to master via TCP

This is a significant feature that is **not yet implemented** but is architecturally possible within the existing cluster framework.

---

For more details, see:
- [PROJECT_STATUS.md](PROJECT_STATUS.md) — Component completion status
- [CROSS_PLATFORM_STATUS.md](CROSS_PLATFORM_STATUS.md) — Platform-specific implementation details
- [IMPLEMENTATION_ACTION_PLAN.md](IMPLEMENTATION_ACTION_PLAN.md) — Phase 2 roadmap
- [USER_GUIDE.md](USER_GUIDE.md) — How to install and use VGRE
