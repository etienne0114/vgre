# VGRE — Missing Features & Next-Release Roadmap

**Last Updated**: 2026-09-25

This file lists **only what is not yet done** — the targets for the next release and the
items blocked on external hardware, accounts, or content. It is deliberately short.
Everything already delivered is documented elsewhere (do not duplicate it here):

- **Execution stack** — the from-scratch, LLVM-free four-tier CPU backend: [`zeroBurdenRoadmap.md`](zeroBurdenRoadmap.md)
- **CUDA-C support matrix** — per tier, bit-exact: [`supportedCudaSubset.md`](supportedCudaSubset.md)
- **Capabilities, test metrics, platform/CI status**: [`PROJECT_STATUS.md`](PROJECT_STATUS.md)
- **ML tracks (T1–T6)** — build steps + success criteria: [`implementationPlan.md`](implementationPlan.md)

**Current baseline (2026-09-25):** the full `ctest` suite is green on all three platforms in CI —
**394/394 with LLVM, 374/374 in the LLVM-free build** on Linux x86-64, and the whole suite passes
on macOS (Apple Silicon) and Windows (clang-cl). Released **v0.1.0** ships self-contained,
LLVM-free wheels for Linux, macOS (arm64) and Windows.

---

## 1. Next release — in-tree, buildable now

Net-new or breadth work we can do without external hardware. (Items marked *correctness done*
already run on some tier and are bit-exact — what's left is a native path, a loader, or breadth.)

### 1.1 Packaging & distribution
- **macOS Intel (x86-64) wheel** — the v0.1.0 macOS wheel bundles an **arm64-only** dylib, so Intel
  Macs must build from source. Add an x86-64 build (or a true universal2 fat dylib) to the
  release-wheels matrix.
- **PyPI publish** — wheels currently live on the GitHub Release only. Add a trusted-publisher
  workflow so `pip install vgre` resolves from PyPI directly.
- **Docker image** — the multi-arch build (arm64 via QEMU) is slow. Ship the **amd64** image fast,
  and build arm64 on a native ARM runner instead of emulation.

### 1.2 SSA / native codegen (Tier-2)
- **Native AArch64 codegen for the cooperative ops** — shared-memory/`__syncthreads` and the warp
  intrinsics emit native machine code on x86-64 but fall to the portable evaluator on ARM. Emit them
  natively on AArch64 (the `Arm64Asm` encoders and the ucontext-fiber scheduler already exist).
- **Flip `VGRE_SSA_ARM_NATIVE` on by default** — after validating native AArch64 *execution* on real
  Apple-Silicon / ARM hardware (the encodings are already llvm-mc-verified and execution is
  CI-checked on `macos-arm64`).

### 1.3 Front-end breadth (from-scratch tiers)
- **Texture / surface**: `tex2DLayeredLod` / cubemap-array variants (scalar `tex1D/2D/3D`,
  `tex1Dfetch`, `surf2Dread/write`, vector fetches `texND<float4>`/`float2`, `tex2DLod`,
  layered `tex1DLayered`/`tex2DLayered`, and **cubemap `texCubemap`** already run on both tiers,
  bit-exact vs the shared `TextureManager`).
- **Recursion** in `__device__` helpers (currently rejected; templates + inlining are done).
- *(The SSA tier now runs the full scalar CUDA-C subset: multi-dimensional arrays
  — `float As[H][W]` —, dynamic `extern __shared__` — launch-sized shared buffers —,
  and by-value struct kernel params are all supported, bit-exact vs the other tiers.)*

### 1.4 Serving / KV cache
- *(Done — `KVCacheManager` now stores K/V as symmetric-absmax **int8 or packed int4** with a
  per-head scale (`KVDType::{F32,I8,I4}`), sharing the exact codec (`vgre/core/kv_quant.h`) with
  the generation path. `pagedAttention` dequantizes on read; the continuous-batching scheduler
  runs unchanged over a quantized pool. Measured **3.8× (int8) / 7.1× (int4)** KV-memory reduction
  at headDim=64, bit-close to fp32 — `test_kv_cache.cpp`.)*

### 1.5 ML track breadth (correctness delivered; breadth/perf left)
| Track | Left | Nature |
|-------|------|--------|
| **T3** Speculative decoding | early-exit self-speculative drafting | throughput optimization (greedy + sampler-exact linear speculative decode, **SpecInfer/Medusa-style tree verification** — accept the longest valid root→leaf path, distribution-exact — KV rollback, prompt-lookup drafter already land) |
| **T4** State-space models | a Mamba safetensors/GGUF loader | breadth (single-state selective scan, depthwise conv1d, **and the Mamba-3 MIMO matrix-state scan** — `H∈R^{N×P}` outer-product update, parallel==sequential, P=1≡SISO — already done) |

---

## 2. Blocked on external content (in-tree code is ready; needs a gated download)

| Item | What is already built | Blocker |
|------|-----------------------|---------|
| **T1** GGUF `I2_S` / TL1 / TL2 ternary tensor loader | absmean ternary codec + multiplication-free GEMM + `BitLinear` QAT | a real **BitNet-b1.58** checkpoint (gated download) to verify end-to-end |
| **Frontier-scale checkpoints** (Llama-3-8B/70B/405B, GPT-3) | identical code path to the verified **GPT-2 (124M)** run (matches Hugging Face) | a **license-gated, multi-GB download** |

---

## 3. Hardware / account / auditor gated (cannot be built in-tree)

The in-tree primitives already exist where applicable; only the externally-gated piece remains.

| § | Track | What's left | Blocker |
|---|-------|-------------|---------|
| 3.1 | GPU security framework | SEV-SNP/TDX enclaves, HSM, FIPS-140 cert | confidential-computing **hardware** + external **auditor** |
| 3.2 | Cryptography | homomorphic / threshold crypto, Intel QAT offload | research-grade scope / crypto-accelerator **hardware** |
| 3.3 | Windows deployment | DirectML backend, AD/Kerberos, Windows containers | Windows-specific **APIs/SDKs** (the engine already builds + tests green on windows-2022) |
| 3.4 | macOS / Apple Silicon | Metal Performance Shaders backend | **Apple Silicon + Metal** hardware (the CPU path is CI-green — full `ctest` passes on `macos-14`) |
| 3.5 | ML frameworks | device-level `jax.jit(backend='vgre')` PJRT plugin | upstream `pjrt_c_api.h` + MLIR C++ libs **not in the wheels** (the StableHLO path already runs JAX/TF/PyTorch) |
| 3.6 | Model serving | TensorRT-LLM / vLLM *compatibility layers* | those external **runtimes** / a live **fleet** |
| 3.7 | Multi-cloud | apply to live AWS/Azure/GCP | cloud **accounts + credentials** (the Terraform module is built) |
| 3.8 | Multi-vendor | Intel oneAPI (SYCL/DPC++), Apple Metal; ROCm library shims | those **SDKs / hardware** (the AMD HIP core runtime is done) |
| 3.9 | Edge / CDN | physical edge nodes, CDN providers | external **infrastructure** (latency-aware routing is built) |
| 3.10 | Post-Blackwell (Rubin) | Rubin / HBM4 emulation | **unreleased** hardware, no public ISA |

---

## 4. Physical multi-node run (external only)

The transport, collectives, tensor/pipeline executor, and GSPMD auto-partitioner are all built and
verified across real OS processes (multi-step data-parallel with zero cross-step drift;
cross-process tensor + expert-parallel MoE). What remains is outside the code:

| Remaining | Why it isn't in-tree |
|-----------|----------------------|
| **Physical multi-node run** (Llama-3-70B tensor-parallel across N machines; 175B/405B pipeline) | needs **actual networked machines** — everything below the wire is built and verified across real OS processes |

---

*Delivered capabilities (the LLVM-optional build, the native x86-64 JIT, the full cooperative
surface, the Tier-2 SSA backend, the T1–T6 ML tracks, and the §2 serving/quantization frontier)
are recorded in the cross-referenced documents above and are intentionally not repeated here.*
