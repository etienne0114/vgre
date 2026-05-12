# VGRE Documentation

Welcome to the VGRE (Virtual GPU Runtime Engine) documentation. This guide will help you understand, install, use, and extend VGRE.

---

## Quick Navigation

### For Users
- **[USER_GUIDE.md](USER_GUIDE.md)** - Installation, setup, and usage instructions for all platforms
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Current project status, test results, and capabilities
- **[missingFeatures.md](missingFeatures.md)** - Exhaustive list of implemented vs missing features
- **[implementationPlan.md](implementationPlan.md)** - Phased roadmap for implementing all missing features

### For Developers
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design, components, and internal architecture
- **[how_it_work.md](how_it_work.md)** - Detailed explanation of how VGRE works internally
- **[api_reference.md](api_reference.md)** - API documentation and function reference

---

## Documentation Overview

### 1. PROJECT_STATUS.md (Canonical)
**Purpose**: Single source of truth for project status  
**Contains**:
- Executive summary and key metrics
- What works and what doesn't
- Known limitations and security issues
- Test coverage and platform support
- Performance baselines
- Recent improvements and roadmap

**When to read**: To understand the current state of VGRE and what features are available.

### 2. USER_GUIDE.md
**Purpose**: Complete guide for installing and using VGRE  
**Contains**:
- System requirements for all platforms
- Installation instructions (Linux, Windows, macOS)
- Framework integration (PyTorch, TensorFlow)
- Cluster setup and configuration
- Troubleshooting guide
- Environment variables reference

**When to read**: When setting up VGRE or configuring it for your use case.

### 3. ARCHITECTURE.md
**Purpose**: Deep dive into VGRE's internal design  
**Contains**:
- System overview and component descriptions
- Execution flow and data structures
- Memory management and UVM implementation
- Kernel compilation pipeline
- Scheduler and parallel executor design
- Security architecture
- Performance optimizations
- Cross-platform implementation details

**When to read**: When extending VGRE, debugging issues, or understanding how it works internally.

### 4. how_it_work.md
**Purpose**: Accessible explanation of VGRE's operation  
**Contains**:
- High-level overview with diagrams
- Step-by-step kernel execution flow
- Memory management explanation
- Cluster networking details
- CUDA Graphs support
- Advanced features (warp shuffles, FP16, WMMA, CDP, PTX translation)
- Cross-platform support details
- Configuration options
- Use cases and limitations

**When to read**: When you want to understand how VGRE works without diving into code.

### 5. api_reference.md
**Purpose**: API documentation and function reference  
**Contains**:
- CUDA Runtime API functions
- Memory management functions
- Stream and event management
- Kernel launch functions
- Device management
- Texture and surface operations
- Graph operations
- Cluster-specific APIs

**When to read**: When looking up specific API functions or their parameters.

---

## Getting Started

### First Time Users
1. Read [PROJECT_STATUS.md](PROJECT_STATUS.md) to understand what VGRE can do
2. Follow [USER_GUIDE.md](USER_GUIDE.md) Section 1 for installation
3. Try the quick examples in [USER_GUIDE.md](USER_GUIDE.md) Section 2

### Cluster Setup
1. Read [USER_GUIDE.md](USER_GUIDE.md) Section 4 for step-by-step cluster setup
2. Use the provided scripts (`vgre_sync.sh`, `setup-cluster.sh`, `vgre-start`)
3. Refer to troubleshooting section if issues arise

### Developers
1. Read [ARCHITECTURE.md](ARCHITECTURE.md) for system design
2. Read [how_it_work.md](how_it_work.md) for detailed operation
3. Explore the source code in `src/` directory
4. Check [api_reference.md](api_reference.md) for API details

---

## Key Features

✅ **Partial CUDA Runtime API** (~94 of ~214 functions) — memory, streams, events, kernel launch, basic graphs  
✅ **OpenCL 1.2 Compatibility**  
✅ **JIT Kernel Compilation** with persistent caching  
✅ **Unified Virtual Memory (UVM)** with page-fault handling  
✅ **Partial CUDA Graphs** — capture, instantiation, replay; memcpy/conditional/external-semaphore nodes. Kernel/memset/host/child nodes missing from shim.  
✅ **Distributed Cluster Networking** with AES-256 encryption  
✅ **Cross-Platform** (Linux, Windows, macOS)  
✅ **Hardware-Backed Token Storage** (keyring, Keychain, CredMan, TPM)  
✅ **NCCL Collective Operations** (~55% coverage) — AllReduce, Broadcast, AllGather, ReduceScatter  
✅ **cuBLAS & cuDNN Shims** (partial) — cuBLAS ~13% (Gemm/Gemv/Axpy/Dot/Nrm2/Scal), cuDNN ~24% (forward-only conv/pool/activation/softmax/BN inference)  

**See `missingFeatures.md` for the exhaustive list of implemented vs missing APIs.**  

---

## Performance Expectations

| Workload | VGRE vs GPU | Notes |
|----------|------------|-------|
| Compute-bound (FP32 matrix math) | 30–50× slower | Expected for CPU emulation |
| Memory-bound (bandwidth-limited) | 5–15× slower | Mitigated by NUMA binding |
| Vectorizable (SIMD-friendly) | 10–20× slower | Mitigated by AVX-512 auto-vectorization |
| DDP training (multi-GPU) | 50–100× slower | NCCL shim available; vendor NCCL faster on large clusters |

**Bottom line**: VGRE is ideal for development, testing, and CI/CD. For production training on real GPUs, use actual hardware.

---

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| **Linux** | ✅ Full | NUMA support, Linux Keyring, perf_event profiling |
| **Windows** | ✅ Full | Credential Manager, WinSock2, VEH for UVM |
| **macOS** | ✅ Full | Keychain, SO_NOSIGPIPE, IOKit temperature |

---

## Common Tasks

### Run a PyTorch Model
```bash
export LD_PRELOAD=/usr/local/lib/vgre/libvgre_cudart.so
python my_pytorch_script.py
```

### Set Up a Cluster
```bash
# On master
bash scripts/setup-cluster.sh
vgre-start --master

# On each worker
bash scripts/setup-cluster.sh
vgre-start --worker
```

### Enable Profiling
```bash
export VGRE_LOG_LEVEL=DEBUG
export VGRE_PROFILER_DUMP=traces.json
python my_script.py
# Open traces.json in Chrome DevTools (chrome://tracing/)
```

### Optimize Performance
```bash
export VGRE_ENABLE_NUMA=1
export VGRE_SIMD_LEVEL=native
export VGRE_WORKER_THREADS=$(nproc)
```

---

## Troubleshooting

### Common Issues

**"libvgre_cudart.so: not found"**
- Set `LD_LIBRARY_PATH=/usr/local/lib/vgre:$LD_LIBRARY_PATH`

**"Worker connection fails"**
- Check token fingerprint matches on all nodes
- Verify port 7777 is open in firewall
- Use `--master-ip` flag for cross-subnet clusters

**"Low performance"**
- First run is slower due to JIT compilation; subsequent runs use cache
- Enable NUMA awareness: `VGRE_ENABLE_NUMA=1`
- Use native SIMD: `VGRE_SIMD_LEVEL=native`

See [USER_GUIDE.md](USER_GUIDE.md) Section 6 for detailed troubleshooting.

---

## Security

VGRE uses industry-standard security practices:
- **Authentication**: HMAC-SHA256 handshake
- **Encryption**: AES-256-CTR (hardware-accelerated via AES-NI)
- **Token Storage**: Hardware-backed (keyring, Keychain, CredMan, TPM)
- **Replay Protection**: 256-bit sequence bitmap

See [ARCHITECTURE.md](ARCHITECTURE.md) for security architecture details.

---

## Contributing

To extend VGRE:
1. Read [ARCHITECTURE.md](ARCHITECTURE.md) to understand the design
2. Explore the source code in `src/` directory
3. Follow the existing code style and patterns
4. Add tests for new features
5. Update documentation

---

## License

VGRE is licensed under the MIT License. See LICENSE file for details.

---

## Support

- **Issues**: Report bugs on GitHub
- **Questions**: Check the troubleshooting section in [USER_GUIDE.md](USER_GUIDE.md)
- **Documentation**: Refer to the appropriate guide above

---

**Version**: 1.0.0  
**Last Updated**: 2026-05-12  
**Status**: Development / CI-Ready (see `missingFeatures.md` for API coverage gaps)
