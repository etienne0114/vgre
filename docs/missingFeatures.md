# VGRE — Missing Features & Next-Release Roadmap

**Last Updated**: 2026-10-07

This file lists **only what is not yet done** — the targets for the next release and the
items blocked on external hardware, accounts, or content. It is deliberately short.
Everything already delivered is documented elsewhere (do not duplicate it here):

- **Execution stack** — the from-scratch, LLVM-free four-tier CPU backend: [`zeroBurdenRoadmap.md`](zeroBurdenRoadmap.md)
- **CUDA-C support matrix** — per tier, bit-exact: [`supportedCudaSubset.md`](supportedCudaSubset.md)
- **Capabilities, test metrics, platform/CI status**: [`PROJECT_STATUS.md`](PROJECT_STATUS.md)
- **ML tracks (T1–T6)** — build steps + success criteria: [`implementationPlan.md`](implementationPlan.md)

**Current baseline (2026-10-07):** [GitHub Actions run 37652962427](https://github.com/etienne0114/vgre/actions/runs/37652962427)
passed all four hosted jobs: Linux x86-64, Linux LLVM-free, macOS ARM64, and Windows x86-64.
PyPI serves **v0.1.4** for Linux, macOS universal2, and Windows. The [v0.1.4 release workflow](https://github.com/etienne0114/vgre/actions/runs/37045306062)
built and smoke-tested each wheel, then published the GitHub Release and PyPI package.

---

## 1. Next release — in-tree, buildable now

Net-new or breadth work we can do without external hardware. (Items marked *correctness done*
already run on some tier and are bit-exact — what's left is a native path, a loader, or breadth.)

### 1.1 Packaging & distribution
- *(Done — PyPI trusted publishing is configured and verified: [v0.1.4](https://pypi.org/project/vgre/0.1.4/)
  was uploaded by GitHub Actions using OIDC; see successful publish run
  [37045306062](https://github.com/etienne0114/vgre/actions/runs/37045306062). No API token is stored.
  Pushing a matching `vX.Y.Z` tag builds all platform wheels and publishes them. A release-tag
  version guard checks the Python package declarations and Linux/Windows source CLI versions, then
  checks the tag before wheel builds. Ordinary pushes to `main` run CI but do not publish a new
  version.)*
- *(Done — the release-wheels matrix builds a **macOS universal2 wheel** on `macos-14` for both
  Apple Silicon and Intel, plus Linux x86-64 and Windows x86-64 wheels. The v0.1.4 release built
  and smoke-tested all three. The Docker release builds **amd64 and arm64 each on its own native
  runner** in parallel — no QEMU emulation — then merges them into one multi-arch manifest.)*

### 1.2 SSA / native codegen (Tier-2)
- *(Done — native AArch64 codegen now covers the **full cooperative surface**: shared-memory /
  `__syncthreads` (already native) **and the warp intrinsics** (`__shfl_*`/vote/`__reduce_*`/match/
  `__activemask`) now emit AArch64 machine code — `bl vgre_ssa_warp`/`vgre_ssa_activemask` per
  AAPCS64, mirroring the x86-64 path and reusing the same portable helpers + ucontext-fiber
  scheduler. It composes only encoders already verified by `arm64EncSelfTest` against llvm-mc.)*
- *(Done — `VGRE_SSA_ARM_NATIVE` is now **ON by default**. Native AArch64 *execution* is validated on
  real Apple Silicon: the `macos-arm64` CI job runs the `SsaIr` differential test + `SsaBackend`
  with native codegen and diffs the emitted machine code bit-for-bit against the portable evaluator
  (step "SSA AArch64 native execution (real Apple Silicon)"). Set `VGRE_SSA_ARM_NATIVE=0` to force
  the evaluator (escape hatch); on any unsupported op the builder bails to the identical evaluator.)*

### 1.3 Front-end breadth (from-scratch tiers)
- *(Done — the full texture/surface surface runs on both from-scratch tiers, bit-exact vs the
  shared `TextureManager`: scalar `tex1D/2D/3D`, `tex1Dfetch`, `surf2Dread/write`, vector fetches
  `texND<float4>`/`float2`, `tex2DLod`, layered `tex1DLayered`/`tex2DLayered`, cubemap `texCubemap`,
  cubemap-array `texCubemapLayered`, and **`tex2DLayeredLod`** — trilinear across the mip levels of
  a mipmapped **layered array** (`createMipmappedLayeredArray`); the LOD genuinely selects levels.)*
- **Recursion** in `__device__` helpers now runs on the **Tier-1 compiled** backend (a
  recursive helper is compiled once into a shared body over a fixed slot frame that is
  saved/restored around each call, so the native stack carries the nested invocations —
  direct + mutual recursion). The interpreter/SSA tiers still inline and reject it, so a
  recursive kernel is dispatched to the compiled tier. **Mutual recursion** works too —
  function prototypes / forward declarations (`ret name(params);`) now parse, so
  `isEven`↔`isOdd` runs end-to-end.
- *(The SSA tier now runs the full scalar CUDA-C subset: multi-dimensional arrays
  — `float As[H][W]` —, dynamic `extern __shared__` — launch-sized shared buffers —,
  and by-value struct kernel params are all supported, bit-exact vs the other tiers.)*
- *(Done — the **legacy (pre-CUDA-9) warp intrinsics without a mask** now run on all
  three tiers: `__shfl`/`__shfl_up`/`__shfl_down`/`__shfl_xor`/`__ballot`/`__any`/`__all`.
  The parser desugars each to its `_sync` form with an implicit full-warp mask
  (0xffffffff), exactly as CUDA does on sm_70+, so the backends only ever see the
  already-supported `_sync` ops — one shared change, no per-tier codegen. `test_cuda_warp_legacy`.)*
- *(The CUDA-C **language** subset is now at its practical ceiling for VGRE's model.
  The last "not yet supported" item — **nested template arguments** `foo<bar<int>>` —
  requires **class templates**, which the from-scratch front-end does not implement by
  design (it targets **function** templates on `__device__` helpers). Kernels needing
  nested class templates (CUB / Cooperative Groups / Thrust) use VGRE's fallback headers
  or the LLVM/Clang path. So the remaining "CUDA coverage" gap is the **hardware-bound
  runtime-API** functions below in §3, not the kernel language.)*

### 1.4 Serving / KV cache
- *(Done — `KVCacheManager` now stores K/V as symmetric-absmax **int8 or packed int4** with a
  per-head scale (`KVDType::{F32,I8,I4}`), sharing the exact codec (`vgre/core/kv_quant.h`) with
  the generation path. `pagedAttention` dequantizes on read; the continuous-batching scheduler
  runs unchanged over a quantized pool. Measured **3.8× (int8) / 7.1× (int4)** KV-memory reduction
  at headDim=64, bit-close to fp32 — `test_kv_cache.cpp`.)*

### 1.5 ML track breadth (correctness delivered; breadth/perf left)
| Track | Left | Nature |
|-------|------|--------|
| **T3** Speculative decoding | — (complete) | throughput optimization — all delivered: greedy + sampler-exact linear speculative decode, SpecInfer/Medusa-style tree verification (distribution-exact), KV rollback, prompt-lookup drafter, **and early-exit self-speculative decoding** (draft with the first E layers, verify with the full model via the batched path — lossless, output == greedy; `SampleConfig::early_exit_draft`) |
| **T4** State-space models | — (complete) | single-state selective scan, depthwise conv1d, the Mamba-3 MIMO matrix-state scan, a full Mamba-1 model + safetensors loader (`mamba.h`/`mamba_loader.cpp`), **and a verified end-to-end run of a real checkpoint** — `tools/mamba_run` loads the open **`state-spaces/mamba-130m-hf`** (130M params, 24 layers) and its logits match an independent NumPy Mamba-1 forward at **cosine sim 1.0** (identical argmax, 10/10 top-10, max abs diff 2e-4) |

### 1.6 Math and tensor-core coverage
- The complete-fragment reference coverage for FP16/BF16/TF32/FP8 `mma.sync`, FP16/TF32 WGMMA, and all currently mapped tcgen05 shapes is in commit `2857b114`; the latest reported Linux, Linux LLVM-free, macOS, and Windows CI jobs are green.
- **Delivered and CI-green:** FP8 E4M3/E5M2 narrowing implements round-to-nearest-even and round-toward-zero, ReLU and finite-saturation modes, and PTX source-to-byte order for packed conversions. The FP16 narrowing and f64-to-f32 rounding follow-on also passed the four-lane hosted CI matrix in run `37652962427`.
- **Current follow-on:** Float-to-integer PTX conversions cover signed/unsigned 8-, 16-, 32-, and 64-bit destinations from f16, bf16, f32, and f64, using `.rni`, `.rzi`, `.rmi`, and `.rpi`. Integer-to-f32/f64 conversions cover all integer widths and five rounding modes with explicit significand rounding. Integer-to-integer conversion covers every signed/unsigned 8-, 16-, 32-, and 64-bit source/destination pair with source-width interpretation and defined narrowing. FP32-to-FP16 narrowing covers all five explicit rounding modes; exact FP16-to-FP32 and FP32-to-FP64 widening uses unrounded PTX forms. The generated JIT source includes the emitted helper implementations. C++17 syntax checks and focused local `PTXTranslate`, `CudaHalf`, `CudaFuzzCast`, and `CudaFuzzToInt` tests pass; they do not exhaustively execute the new conversion matrix, and hosted CI remains pending. Remaining math breadth is beyond explicitly mapped PTX shapes and formats.
- **Remaining:** broaden coverage beyond explicitly mapped PTX shapes and formats. Unsupported MMA variants remain unsupported; current tests establish correctness only for translator entries listed in `src/compiler/ptx/ptx_translator_map.cpp` and `src/compiler/ptx/ptx_conversion.cpp`.
- **Current cuDNN backend follow-on (workspace):** the signal operation now performs atomic flag SET/WAIT with pass-through tensor copying; unrecognized pointwise modes report `CUDNN_STATUS_NOT_SUPPORTED` rather than producing identity output. Pointwise `GELU_FWD` uses the exact erf expression, while the tanh form is available only through the explicit approximate mode; invalid-domain arithmetic preserves IEEE results. The focused `CudnnBackendV8` regression passes locally, including signal and exact GELU checks; hosted CI is pending. Signal semantics follow NVIDIA's [cuDNN backend signal operation](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.11.1/api/cudnn-graph-library.html#cudnn-backend-operation-signal-descriptor) contract.
- **Current cuDNN activation follow-on (workspace):** direct activation GELU forward/backward now implement the exact erf definition and analytical derivative. Descriptor initialization and enum values are validated; invalid modes cannot silently become identity/gradient-one, NaN propagation is honored, and ReLU uses its configured cap. Float32 supports the in-tree activation modes except identity in the direct APIs (identity remains available to fused operations); plain INT8 supports ReLU and clipped ReLU with alpha/beta blending and saturating rounding. Other tensor types and INT8 modes return `CUDNN_STATUS_NOT_SUPPORTED`. Normalization fusion now adds its optional residual before activation; batch-normalization backward rejects modes that require unavailable pre-activation values. Local `ExactGELUActivation`, `CUDNNActivationBackward`, `CudnnBackendV8`, and `CuDNNNormalization` tests pass; hosted CI remains pending. cuDNN defines GELU as `x Φ(x)` in the [original GELU paper](https://arxiv.org/abs/1606.08415); the [normalization API](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-ops-library.html) specifies residual addition before activation.
- **Current SASS decoder safety follow-on (workspace):** the decoder no longer synthesizes no-op PTX entries for kernels with metadata but no code. Unknown SASS opcodes or modules with a kernel missing its code section now fail translation as an empty result, allowing the caller to report unsupported SASS instead of silently running a truncated kernel. The decoder still covers only its explicitly mapped opcodes; full cubin/SASS execution remains unsupported.
- **Current surface-object introspection follow-on (workspace):** `cudaGetSurfaceObjectResourceDesc` now returns the exact resource descriptor retained at object creation and reports invalid/destroyed handles instead of returning a fabricated linear descriptor. The `SASSDetection` integration test covers create/query/destroy behavior locally; hosted CI remains pending.
- **Current texture-object introspection boundary (workspace):** texture resources currently retain only the sampling fields VGRE needs, not the full CUDA resource/texture/view descriptors. Their getters now return `cudaErrorNotSupported` for live objects instead of returning fabricated defaults; descriptor-preserving texture handles remain unimplemented.
- **Current PTX translator follow-on (workspace):** numbered/named inline-assembly operands are now substituted from actual output/input constraints, nested expressions are parsed through the complete asm call, and predicated or malformed operands fail explicitly. `PTXTranslate` passes locally; hosted CI is pending.
- **Current CUDA device-attribute follow-on (workspace):** numeric IDs now follow CUDA Runtime 11.0, and reserved enum gaps/unknown IDs fail with `cudaErrorInvalidValue` instead of succeeding with a fabricated zero. Physical GPU memory clock, bus width, L2 size, and measured FP32/FP64 throughput remain unavailable on the CPU-backed virtual device and return `cudaErrorNotSupported`; hosted CI for this correction is pending.
- **Current CUPTI follow-on (workspace):** the VGRE-specific, non-NVIDIA-ABI interface implements event groups for measured kernel launch count, elapsed nanoseconds, and virtual threads; metrics calculate from collected event values, callbacks receive completed API-tagged launches, and activity records preserve per-launch dimensions and measured duration. NVTX and other non-kernel profile entries are excluded. The test executes real driver and runtime-interceptor launches, checks computed output, then verifies counters, callbacks, and activity records. Full local CTest passes in both normal and LLVM-free builds (410/410 and 390/390). Physical GPU PMU metrics, register counts, utilization, and device/context/stream IDs remain unavailable and are never inferred from host counters or static estimates. Hosted CI is pending.

---

## 2. Blocked on external content (in-tree code is ready; needs a gated download)

| Item | What is already built | Blocker |
|------|-----------------------|---------|
| **T1** GGUF `I2_S` ternary loader **+ BitNet SubLN transformer** | **DONE (loader + model)** — (a) `I2_S` (ggml type 36) dequantizes in the GGUF loader (`quant.h::dequant_i2_s` + `gguf.cpp`): 2-bit MSB-packed `(c0<<6)|(c1<<4)|(c2<<2)|c3`, codes→{-1,0,+1}, trailing f32 scale = 1/mean\|w\| ⇒ `w=(code-1)/scale` (microsoft/BitNet reference doc; validated by `test_i2s_quant` + bit-identical to a reference dequant on the real `bitnet-b1.58-2B` tensor bytes). (b) The **BitNet-b1.58 SubLN transformer** is wired into the runtime: `Config::sub_norm` adds the two per-block RMSNorm sub-layers (`attn_sub_norm` on the attention output before `o_proj`, `ffn_sub_norm` on the SwiGLU intermediate before `down_proj`) across all three forward paths (autograd `forward`, KV-cached decode, batched prefill), and `load_gguf_bitnet` loads the `bitnet-b1.58` GGUF (GQA 20/5 + RoPE-500000 + tied head + I2_S linears). `test_bitnet_subln` checks the forward is **bit-exact (max\|Δ\|≈1e-7) vs an independent reference** and that decode==prefill; `tools/bitnet_run` runs the real 1.2 GB checkpoint end-to-end. TL1/TL2 are the T-MAC LUT variants (separate). | — (in-tree code complete) |
| **Frontier-scale checkpoints** (Llama-3-8B/70B/405B, GPT-3) | identical code path to the verified **GPT-2 (124M)** run (matches Hugging Face) | a **license-gated, multi-GB download** |

*(Mamba is no longer here — the loader + forward were run end-to-end on the real
open `mamba-130m-hf` checkpoint and match a reference at cosine sim 1.0; see the T4 row.
BitNet/frontier remain because their checkpoints are license-**gated**, not merely large.)*

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
