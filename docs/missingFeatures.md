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
- **PyPI publish** — the trusted-publisher **workflow is now in place** (`release-wheels.yml` → the
  opt-in `pypi` job, `pypa/gh-action-pypi-publish` via OIDC, gated behind `publish_pypi=true` so it
  never breaks the auto-tag release). The only thing left is the **one-time PyPI-side setup** — create
  the `vgre` project and add a trusted publisher for this repo/workflow/`pypi` environment — which is
  an **account action** on pypi.org, not in-tree code.
- *(Done — the release-wheels matrix now builds a **macOS Intel (x86-64) wheel** on the `macos-13`
  runner alongside the arm64 one, `build_wheel.sh` tags each by the host platform. The Docker
  release now builds **amd64 and arm64 each on its own native runner** in parallel — no QEMU
  emulation — then merges them into one multi-arch manifest. Both are validated by CI on the
  next release tag.)*

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
