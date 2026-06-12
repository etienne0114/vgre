# VGRE Implementation Plan — Phase 3 (Innovation Frontier)

**Last Updated**: 2026-06-12

**Phase 1 (tracks A–W)** and **Phase 2 (tracks 1–27)** are complete and verified
in-tree (see git history `feat(track-*)`/`fix(track-*)`); they are not repeated
here. Phase 2 closed the production-readiness and the "no stubs/heuristics" gaps:
real warp-collective tensor-core `mma.sync`, FlashAttention-2 online softmax,
pipelined NCCL, FP8/FP4 + cuBLASLt FP8 matmul, paged attention, continuous
batching, sparse-format conversions, GF(2) cuRAND skip-ahead, liveness DCE, the
Prometheus/health endpoints, the Linux+macOS CI matrix, etc.

**Audit (2026-06-12), before opening Phase 3** — every Phase-2 "done" track was
re-verified at its source location; **all `src/**.cpp` are wired into CMake**,
all tests registered, no genuine `TODO/FIXME/stub` remains. Two dead headers were
resolved: `fp16_ops.h` removed (redundant + arm64-breaking), and
`gradient_checkpointing.h` wired with `test_gradient_checkpointing` (13/13).

Phase 3 is the **2026–2027 frontier**: Blackwell-class quantized compute,
next-gen tensor/memory ISA (wgmma/TMA/clusters/tcgen05), advanced serving
(speculative decode, MoE), new architectures (SSM), richer frontends, distributed
realism, and determinism. Each track maps to a `docs/missingFeatures.md` section.
Priorities: **P1** = capability real frameworks hit; **P2** = fidelity/frontier.
All tracks start **🔴 Planned**.

---

## Completion Tracker

| # | Track | Pri | Maps to | Status |
|---|---|---|---|---|
| P3-1 | FP4 (NVFP4/MXFP4) through cuBLASLt | P1 | §1.1 | ✅ Done |
| P3-2 | Weight-only INT4 (AWQ/GPTQ, W4A16) | P1 | §1.2 | ✅ Done |
| P3-3 | Microscaling (MX) format family | P2 | §1.3 | ✅ Done |
| P3-4 | Hopper `wgmma.mma_async` warp-group MMA | P1 | §2.1 | ✅ Done |
| P3-5 | TMA `cp.async.bulk(.tensor)` | P1 | §2.2 | ✅ Done |
| P3-6 | Thread-block clusters + distributed SMEM | P2 | §2.3 | ✅ Done |
| P3-7 | Blackwell `tcgen05` + tensor memory | P2 | §2.4 | 🔴 Planned |
| P3-8 | Speculative decoding (draft+verify) | P1 | §3.1 | ✅ Done |
| P3-9 | MoE: top-k router + grouped GEMM | P1 | §3.2 | ✅ Done |
| P3-10 | Prefix caching + chunked prefill | P2 | §3.3 | ✅ Done |
| P3-11 | Structured attention masks | P2 | §3.4 | ✅ Done |
| P3-12 | Mamba/SSM selective-scan | P2 | §4.1 | ✅ Done |
| P3-13 | Triton IR frontend | P2 | §5.1 | 🔴 Planned |
| P3-14 | CUDA-graph capture-from-stream fidelity | P2 | §5.2 | 🔴 Planned |
| P3-15 | Virtual NVLink topology + collective cost model | P1 | §6.1 | ✅ Done |
| P3-16 | NVSHMEM symmetric memory (one-sided) | P2 | §6.2 | ✅ Done |
| P3-17 | Tensor/pipeline parallel primitives | P2 | §6.3 | ✅ Done |
| P3-18 | Bit-deterministic mode | P1 | §7.1 | ✅ Done |
| P3-19 | Differential testing harness | P2 | §7.2 | ✅ Done |
| P3-20 | UVM oversubscription + disk eviction | P2 | §8.1 | 🔴 Planned |
| P3-21 | Occupancy + roofline + flame graphs | P2 | §8.2 | ✅ Done |

---

## P1 — Capability frameworks actually hit

### P3-1 — FP4 (NVFP4/MXFP4) through cuBLASLt
**File**: `src/api/cublaslt/cublaslt_matmul.cpp`, `include/vgre/core/math/fp_quant_gemm.h`.
Add `CUDA_R_4F_E2M1` operands and an MX block-scale mode
(`CUBLASLT_MATMUL_DESC_A_SCALE_MODE` + per-K-block scale vectors). Reuse
`fp4_gemm_block_scaled` to dequantize FP4 operands per 16/32-element block.
**Acceptance**: scaled FP4 matmul == dequant→FP32 GEMM within format epsilon,
through the public cuBLASLt API (mirrors the Track-17 FP8 test).

### P3-2 — Weight-only INT4 (AWQ/GPTQ, W4A16)
**New**: an INT4 group-quant codec in `vgre::quant` (per-group scale + zero-point,
group 64/128) + a `dequant→FP16 GEMM` op. The dominant deployed LLM-weight
format. **Acceptance**: matches a reference dequant-then-GEMM; reproduces an AWQ
checkpoint's logits within tolerance.

### P3-4 — Hopper `wgmma.mma_async` warp-group MMA
**File**: `include/vgre/compiler/wmma_emulation.h` (wgmma helpers),
`src/compiler/ptx/ptx_translator_map.cpp`. Promote the single-thread descriptor
GEMM to a real **128-lane warp-group collective** by reusing the per-warp
fragment buffer + barrier from Track 9 at warp-group scope; decode the 64-bit
SMEM matrix descriptor (base/leading-dim/swizzle). **Acceptance**: bit-comparable
to a reference GEMM for the supported wgmma shapes/precisions (CUTLASS-3.x/Triton
Hopper kernels run).

### P3-5 — TMA `cp.async.bulk(.tensor)`
**New**: a host-side `CUtensorMap` (`cuTensorMapEncodeTiled`) + PTX lowerings for
`cp.async.bulk.tensor.Nd.shared::cluster.global` → strided memcpy with mbarrier
arrival completion (the JIT block barrier is the substrate). **Acceptance**: a
TMA-tiled GEMM/attention kernel produces correct results.
**Fix (store path)**: the `cp.async.bulk.tensor.Nd.global.shared::cta.bulk_group`
**store** opcodes were mis-lowered to TMA *loads* (the 2d form was additionally
shadowed across maps and its store helper had a swapped descriptor/source argument),
so every TMA store silently became a load — results never written back. Now all
1d–5d store forms lower through one `tma_store_emit` parser (real
`[tensorMap,{coords}], [src]` operand layout) to `vgre_tma_store_<rank>d_b`, the
exact inverse of the load, with out-of-bounds box clipping (no overrun, matching
real TMA store boundary semantics). Regression-guarded by a store-direction
translation check + an interior/OOB/round-trip runtime test.

### P3-8 — Speculative decoding (draft + verify)
**File**: `ContinuousBatchScheduler` in `include/vgre/core/kv_cache.h` +
`src/core/kv_cache.cpp`. A draft step proposes K tokens; a single batched target
forward over the K positions verifies; accept via the rejection-sampling rule and
roll the KV-cache back to the accepted length. **Acceptance**: accepted output is
distribution-identical to plain decode; >1 token/step amortized.

### P3-9 — MoE: top-k router + grouped expert GEMM
**File**: `src/compiler/kernel_fusion_engine.cpp` + a grouped-GEMM op. Top-k
softmax router → permutation/scatter grouping tokens by expert → grouped GEMM
(variable per-expert M) → un-permute + weighted combine. **Acceptance**: equals a
dense reference that routes every token through its top-k experts.

### P3-15 — Virtual NVLink topology + collective cost model
**File**: `src/api/nccl/*`, scheduler. A configurable virtual topology (NVLink
domains/NVSwitch/PCIe) + a bandwidth/latency cost model on ring/tree collectives
so reported timings track real multi-GPU scaling. **Acceptance**: AllReduce
timing scales with the configured topology; numerics unchanged.

### P3-18 — Bit-deterministic mode
**Files**: reductions (cuBLAS/attention/NCCL), RNG, `BlockWorkerPool`,
`config_registry`. `VGRE_DETERMINISTIC=1` forces a fixed reduction order/shape, a
fixed worker count, and counter-based RNG seeding. **Acceptance**: two runs of a
non-trivial workload are byte-identical.

---

## P2 — Fidelity & research frontier

### P3-3 — Microscaling (MX) format family
Generalize the FP4 block-scale machinery to MXFP8/MXFP6/MXFP4 + MXINT8 (OCP MX:
32-element blocks, shared UE8M0 scale) as `vgre::quant` codecs + a uniform
`mx_gemm`. **Acceptance**: round-trip + scaled-GEMM equivalence per format.

### P3-6 — Thread-block clusters + distributed shared memory ✅ DONE
`include/vgre/core/cluster.h`: a rank→SMEM-base table with `map_shared_rank`
(`mapa.shared::cluster` — retarget an address to a peer CTA's shared memory) and a
sense-reversing cluster barrier (`cluster.sync`). Wired end-to-end:
`CPUParallelExecutor::executeClustered` tiles the grid into clusters whose CTAs run
concurrently on the `BlockWorkerPool`, registering each CTA's shared buffer and
installing the per-CTA cluster context (`vgre_jit_set_cluster`); the JIT helpers
`vgre_jit_cluster_sync` / `vgre_jit_mapa_shared_cluster` / `vgre_jit_cluster_{c,nc}tarank`
read it. PTX `barrier.cluster.arrive/wait` + `mapa.shared::cluster.{u32,u64}` lower
to those helpers; the prelude exposes `cooperative_groups::this_cluster()`
(`sync` / `map_shared_rank` / `block_rank` / `num_blocks`); `cuLaunchKernelEx`
parses the cluster-dimension launch attribute and routes to
`launchClusteredKernel`. **Acceptance**: a 2-block-cluster reduction reads a peer's
DSMEM correctly — plus 4/8-CTA and 2×2 cluster DSMEM reductions, barrier ordering,
and the executor path (`test_cluster`, `test_cluster_exec`).

### P3-7 — Blackwell `tcgen05` + tensor memory
**File**: `src/compiler/ptx/ptx_conversion.cpp` (tcgen05 shapes already decoded).
Model `tmem` as per-CTA scratch and implement the `tcgen05.mma/.ld/.st` data
path. **Acceptance**: a tcgen05 GEMM matches the reference.

### P3-10 — Prefix caching + chunked prefill
**File**: `include/vgre/core/kv_cache.h`. Content-hash a prompt's block table and
share read-only KV blocks across sequences (copy-on-write on divergence); chunk
long prefills to interleave with decode. **Acceptance**: two requests with a
shared prefix reuse the prefix's physical blocks; outputs unchanged.

### P3-11 — Structured attention masks
Sliding-window, attention-sink, and ALiBi-bias variants of the paged-attention
kernel. **Acceptance**: each matches a naive masked-softmax reference.

### P3-12 — Mamba / SSM selective-scan
A chunked parallel associative scan (Blelloch — VGRE already has prefix-sum)
specialized to `h_t = A_t h_{t-1} + B_t x_t`. **Acceptance**: matches a sequential
reference scan.

### P3-13 — Triton IR frontend
An MLIR/TTIR → VGRE-IR lowering feeding the existing JIT (vs. consuming emitted
PTX). **Acceptance**: a representative Triton kernel runs from IR with matching
output.

### P3-14 — CUDA-graph capture-from-stream fidelity
**File**: `src/api/cudart/cudart_shim_capture.cpp`, `src/core/runtime_engine_graph.cpp`.
Record every stream op during capture into graph nodes with real dependencies;
replay bit-identical to eager. **Acceptance**: a captured+replayed multi-kernel
torch graph equals eager.

### P3-16 — NVSHMEM symmetric memory (one-sided)
A symmetric heap + one-sided `put/get/atomic` over the existing RDMA/TCP
transport. **Acceptance**: a one-sided ring AllReduce matches the two-phase
reference.

### P3-17 — Tensor / pipeline parallel primitives
Column/row-parallel linear + all-reduce/all-gather helpers and a P2P send/recv
pipeline stage over existing P2P. **Acceptance**: a 2-way tensor-parallel MLP
equals the unsharded result.

### P3-19 — Differential testing harness
Extend the golden-vector suite (Track 26) into a property-based differential
tester that fuzzes shapes/dtypes per op vs. an independent oracle
(NumPy/SciPy/OpenBLAS), gated in CI. **Acceptance**: a CI job failing on any op
that diverges from its oracle.

### P3-20 — UVM oversubscription + disk-backed eviction
**File**: `src/core/memory/memory_manager*.cpp`. An LRU page evictor that spills
cold managed pages to a backing file and faults them back via the existing
SIGSEGV/SIGBUS handler. **Acceptance**: a workload allocating > host RAM completes
correctly.

### P3-21 — Occupancy + roofline + flame graphs
**File**: profiler + `src/api/nsight_exporter.cpp` / metrics server. An SM
occupancy model (regs/SMEM/warps → active warps), a roofline view, and a
flame-graph export from the trace. **Acceptance**: occupancy matches CUDA's
calculator for known kernels.

---

## Environment-blocked (carried from Phase 2 — not Phase-3 code work)
- **Windows build/test green** — needs a Windows CI runner (configure unblocked).
- **Multi-arch Docker publish** — needs `buildx`/CI (artifacts + workflow ready).
- **perf_event cache-hit proxies** — host `perf_event_paranoid=4` blocks
  verification (IPC counter + proxy fallback already present).
