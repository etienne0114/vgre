# VGRE — Phase 3: Innovation & Frontier Targets

**Last Updated**: 2026-06-12

This is the authoritative, forward-looking ledger for VGRE's **Phase 3**. It is a
clean slate: **Phase 1 (tracks A–W) and Phase 2 (tracks 1–27) are complete and
verified in-tree** (see git history `feat(track-*)`/`fix(track-*)` and
`docs/implementationPlan.md`). Every previously-listed gap (simplified Flash
Attention, NCCL barrier-per-round, flat-dot-product WMMA, cuSPARSE conversions,
graph DCE, PTX carry chains, cuRAND skip-ahead, FP8/FP4 + cuBLASLt, paged
attention, continuous batching, CI matrix, metrics/health, …) has been
implemented and tested; this document does **not** re-list them.

> **Verification done 2026-06-12** (before opening Phase 3):
> - Every cited Phase-1/2 gap location was re-read — the real algorithm is in
>   place (online-softmax, NCCL reduce-scatter/all-gather, warp-collective
>   `mma.sync`, GF(2) cuRAND jump matrix, liveness DCE, sparse conversions, …).
> - **Wiring**: every `src/**.cpp` is referenced by a `CMakeLists.txt`; every
>   `tests/**/test_*.cpp` is registered; no genuine `TODO/FIXME/stub` remains
>   (the residual "simplified"/"placeholder" strings are boundary docs, Windows
>   `MEM_*_PLACEHOLDER` API flag names, and CUDA empty-node semantics).
> - Two **dead headers** were found and resolved: `fp16_ops.h` (redundant with
>   `vgre_cuda::__half`, and its unconditional `<immintrin.h>` broke arm64) was
>   removed; `gradient_checkpointing.h` was wired with `test_gradient_checkpointing`.

Phase 3 is graded against the **2026–2027** GPU/AI frontier (Blackwell/NVFP4,
disaggregated serving, MoE, SSMs, the Triton/cuda.core stack) and asks: *what
would a CPU GPU-emulation runtime need to credibly emulate the next two years of
CUDA workloads?* Each item is concrete and cites where it hooks into the tree.
Priorities: **P1** = major capability frameworks actually hit; **P2** =
fidelity / research-frontier.

---

## 1. Blackwell-era quantized compute

### 1.1 FP4 (NVFP4 / MXFP4) through cuBLASLt — P1 — ✅ DONE
- **Have**: `vgre::quant::fp4_gemm_block_scaled` (block-scaled E2M1) and FP8 in
  `cublasLtMatmul` (Track 17).
- **Gap**: no `CUDA_R_4F_E2M1` operand path in `cublasLtMatmul`, and no
  block-scale-mode descriptor attribute (`CUBLASLT_MATMUL_DESC_A_SCALE_MODE` /
  per-K-block scale vectors). FP4 is Blackwell's headline throughput lever and
  the default for NVFP4-quantized inference.
- **Design**: add `CUDA_R_4F_E2M1` + the MX block-scale mode; in
  `src/api/cublaslt/cublaslt_matmul.cpp` dequantize FP4 operands per K-block via
  the existing `fp4_gemm_block_scaled` contract (one UE8M0/FP8 scale per 16- or
  32-element block). Acceptance: scaled FP4 matmul == dequant→FP32 GEMM within
  the format epsilon, through the public API.

### 1.2 Weight-only INT4 (AWQ / GPTQ, W4A16) — P1 — ✅ DONE
- **Gap**: the dominant *deployed* LLM-weight format is 4-bit group-quantized
  weights with FP16 activations (AWQ/GPTQ) — VGRE has no W4A16 path.
- **Design**: a group-wise INT4 codec (per-group scale + zero-point, group size
  64/128) in `vgre::quant`, and a `dequant→FP16 GEMM` op wired so a cuBLASLt /
  custom-kernel call with INT4 weights + FP16 activations works. Acceptance:
  matches a reference dequant-then-GEMM; reproduces an AWQ checkpoint's logits.

### 1.3 Microscaling (MX) format family — P2 — ✅ DONE
- **Design**: generalize the FP4 block-scale machinery to MXFP8/MXFP6/MXFP4 and
  MXINT8 (OCP MX spec: 32-element blocks, shared UE8M0 power-of-two scale) as
  `vgre::quant` codecs + a uniform `mx_gemm` entry. Acceptance: round-trip +
  scaled-GEMM equivalence per format.

---

## 2. Next-gen tensor & memory architecture emulation

### 2.1 Hopper `wgmma.mma_async` warp-group MMA — P1
- **Location**: `include/vgre/compiler/wmma_emulation.h` (wgmma helpers, with the
  "simplified descriptor" note); `src/compiler/ptx/ptx_translator_map.cpp`.
- **Gap**: the Hopper warp-group MMA (`wgmma.mma_async.sync.aligned.mNkNxN`) is
  emulated as a single-thread full GEMM from a raw descriptor pointer — it does
  not model the 128-lane warp-group collective or the SMEM swizzle/descriptor.
- **Design**: reuse the warp-collective fragment buffer built for Track 9 at
  *warp-group* (4-warp / 128-lane) scope; decode the 64-bit SMEM matrix
  descriptor (base, leading-dim, swizzle mode) properly. Used by CUTLASS 3.x and
  Triton Hopper kernels. Acceptance: bit-comparable to a reference GEMM for the
  supported wgmma shapes/precisions.

### 2.2 TMA — `cp.async.bulk` / `cp.async.bulk.tensor` — P1
- **Gap**: no Tensor Memory Accelerator. Hopper/Blackwell kernels move tiles with
  asynchronous bulk + tensor-descriptor copies gated by an mbarrier; VGRE's PTX
  translator has no `cp.async.bulk*` and no `cuTensorMapEncodeTiled` descriptor.
- **Design**: a host-side tensor-map object (`CUtensorMap`) + PTX lowerings for
  `cp.async.bulk.tensor.Nd.shared::cluster.global` to strided memcpy with
  mbarrier-arrival completion (the JIT already has a block barrier). Acceptance:
  a TMA-based tiled GEMM/attention kernel produces correct results.

### 2.3 Thread-block clusters + distributed shared memory — P2
- **Gap**: SM90 thread-block clusters (`cluster.sync`, `mapa`,
  `cluster.map_shared_rank`, cluster dims in the launch) and cross-block DSMEM
  are unsupported — the block-worker model is per-block only.
- **Design**: a cluster grouping over `BlockWorkerPool`, a cluster barrier, and a
  rank→SMEM map so one block can address a peer block's shared memory.
  Acceptance: a 2-block-cluster reduction reads a peer's DSMEM correctly.

### 2.4 Blackwell `tcgen05` 5th-gen tensor cores + tensor memory — P2
- **Location**: `src/compiler/ptx/ptx_conversion.cpp` already lists
  `tcgen05.mma.*` shapes (decoded, but the MMA + tensor-memory (`tmem`) semantics
  are placeholder-level).
- **Design**: model `tmem` as a per-CTA scratch region and implement the
  `tcgen05.mma`/`tcgen05.ld`/`tcgen05.st` data path. Acceptance: a tcgen05 GEMM
  matches the reference.

---

## 3. Serving-layer innovation (on Tracks 18/22)

### 3.1 Speculative decoding (draft + verify) — P1 — ✅ DONE
- **Gap**: the continuous-batch scheduler (`ContinuousBatchScheduler`,
  `include/vgre/core/kv_cache.h`) advances one token/step. Modern serving uses a
  small draft model (or Medusa/EAGLE heads) to propose K tokens, then the target
  model verifies and accepts the longest matching prefix — 2–3× decode speedup.
- **Design**: a draft/verify step that proposes K tokens, runs a single batched
  target forward over the K positions, accepts via the rejection-sampling rule,
  and rolls the KV-cache back to the accepted length. Acceptance: accepted
  sequence is distribution-identical to plain greedy/sampled decode.

### 3.2 MoE: top-k router + grouped expert GEMM — P1 — ✅ DONE
- **Gap**: Mixture-of-Experts is the dominant frontier-LLM scaling axis (Mixtral,
  DeepSeek, GPT-OSS) and VGRE has no expert routing or grouped/batched GEMM.
- **Design**: a top-k softmax router producing per-token expert assignments, a
  permutation/scatter to group tokens by expert, and a grouped GEMM (variable
  per-expert M) in the fusion engine, then un-permute + combine. Acceptance:
  output matches a dense reference that runs every token through its top-k
  experts.

### 3.3 Prefix caching + chunked prefill — P2
- **Gap**: `KVCacheManager` allocates per-sequence blocks with no cross-sequence
  sharing; long shared system prompts re-compute/re-store identical KV.
- **Design**: content-hash the prompt's block table and share read-only KV blocks
  across sequences (copy-on-write on divergence); chunk long prefills so they
  interleave with decode. Acceptance: two requests with a shared prefix reuse the
  prefix's physical blocks; outputs unchanged.

### 3.4 Structured attention masks — P2 — ✅ DONE
- **Design**: sliding-window, attention-sink, and ALiBi-bias variants of the
  paged-attention kernel (the masks modern long-context models use). Acceptance:
  each matches a naive masked-softmax reference.

---

## 4. New model architectures

### 4.1 Mamba / SSM selective-scan — P2 — ✅ DONE
- **Gap**: state-space models (Mamba-2, Jamba) use a parallel *selective scan*,
  not attention — VGRE has no associative-scan primitive for it.
- **Design**: a chunked parallel associative scan (Blelloch-style, which VGRE
  already has for prefix-sum) specialized to the SSM recurrence
  `h_t = A_t h_{t-1} + B_t x_t`. Acceptance: matches a sequential reference scan.

---

## 5. Frontends & graph fidelity

### 5.1 Triton IR frontend — P2
- **Gap**: VGRE consumes Triton's *emitted PTX*; it cannot ingest Triton's own IR
  (TTIR/TTGIR MLIR). A direct frontend would avoid PTX round-trips and support
  Triton features that lower poorly to PTX.
- **Design**: an MLIR/TTIR → VGRE-IR lowering path feeding the existing JIT.
  Acceptance: a representative Triton kernel runs from IR with matching output.

### 5.2 Full CUDA-Graph capture-from-stream fidelity — P2
- **Gap**: conditional graph nodes and explicit graph build work (Track 16 note),
  but capturing an *arbitrary* stream of library calls (`torch.compile`'s CUDA
  graphs) into a replayable graph is only partially faithful.
- **Design**: record every stream op during capture into graph nodes with their
  real dependencies; replay must be bit-identical to eager. Acceptance: a
  captured+replayed multi-kernel torch graph equals eager execution.

---

## 6. Multi-GPU / distributed realism

### 6.1 Virtual NVLink/NVSwitch topology + collective cost model — P1 — ✅ DONE
- **Gap**: multi-vGPU collectives (NCCL) compute correct results but model no
  interconnect topology — ranks share host memory at host bandwidth, so
  collective *timing* is unrealistic for scaling studies.
- **Design**: a configurable virtual topology (NVLink domains, NVSwitch, PCIe
  fallback) and a bandwidth/latency cost model applied to ring/tree collectives
  so reported timings track real multi-GPU behavior. Acceptance: AllReduce timing
  scales with the configured topology; numerics unchanged.

### 6.2 NVSHMEM symmetric memory (one-sided put/get) — P2 — ✅ DONE
- **Design**: a symmetric heap + one-sided `put/get/atomic` over the existing
  RDMA/TCP transport, the substrate for NVSHMEM-style collectives and
  fine-grained overlap. Acceptance: a one-sided ring AllReduce matches the
  two-phase reference.

### 6.3 Tensor / pipeline parallel primitives — P2 — ✅ DONE
- **Design**: column/row-parallel linear + all-reduce/all-gather helpers and a
  P2P send/recv pipeline stage over existing P2P, so a sharded model runs.
  Acceptance: a 2-way tensor-parallel MLP equals the unsharded result.

---

## 7. Determinism & differential verification

### 7.1 Bit-deterministic mode — P1
- **Gap**: reductions (GEMM, attention, NCCL) and RNG depend on thread
  scheduling, so runs are not bit-reproducible — a real blocker for debugging and
  regression gating.
- **Design**: `VGRE_DETERMINISTIC=1` forces fixed reduction order (serial or
  tree with a fixed shape), a fixed worker count, and counter-based RNG seeding.
  Acceptance: two runs of a non-trivial workload are byte-identical.

### 7.2 Differential testing harness — P2
- **Design**: extend the golden-vector suite (Track 26) into a property-based
  differential tester that fuzzes shapes/dtypes for each op and compares against
  an independent oracle (NumPy/SciPy/OpenBLAS) at tight tolerance, gated in CI.
  Acceptance: a CI job that fails on any op diverging from its oracle.

---

## 8. Memory & observability

### 8.1 UVM oversubscription with disk-backed eviction — P2
- **Gap**: managed memory is capped by host RAM. Real UVM oversubscribes device
  memory and migrates cold pages; VGRE's UVM (`memory_manager`) has dirty-page
  tracking but no eviction-to-disk.
- **Design**: an LRU page evictor that spills cold managed pages to a backing
  file and faults them back on access (reusing the existing SIGSEGV/SIGBUS fault
  handler). Acceptance: a workload allocating > host RAM completes correctly.

### 8.2 Occupancy calculator + roofline + flame graphs — P2 — ✅ DONE
- **Design**: an SM occupancy model (registers/SMEM/warps per block → active
  warps), a roofline (FLOP/byte vs. measured) view, and a flame-graph export from
  the existing trace, surfaced via the metrics/Nsight exporter. Acceptance: the
  occupancy calculator matches CUDA's spreadsheet for known kernels.

---

## Out of scope (environment-blocked, not missing code)
- **Windows full build/test green** — needs a Windows CI runner to iterate
  (configure is unblocked; build/link items remain).
- **Multi-arch Docker publish** — needs `buildx`/CI (artifacts + workflow ready).
- **perf_event cache-hit proxies** — this host has `perf_event_paranoid=4`;
  unverifiable here (the IPC counter + proxy fallback already exist).
