# vNext Audit & LLVM-Reduction Plan

**Date:** 2026-09-06 · **Scope:** whole-repo audit, doc-truth reconciliation, and a
concrete plan to make VGRE lightweight by removing the mandatory LLVM dependency.

> **Direction note (2026-09-06):** the chosen strategy is a **from-scratch
> replacement** of the LLVM path, *not* an optional `VGRE_ENABLE_JIT` flag — see
> **[`zeroBurdenRoadmap.md`](zeroBurdenRoadmap.md)** for the authoritative plan.
> The **footprint measurements, repo-hygiene findings, and doc-truth table below
> remain valid** and feed that plan; where §2 here mentions an "optional flag",
> read it as superseded by the layered from-scratch `ExecutionBackend`.

This document is the research deliverable behind the next version. It records
(1) what is actually true in the tree today, (2) issues that need attention, and
(3) a staged plan to build the parts we need from scratch so LLVM is optional —
fixing the large download / slow-and-incomplete Windows build.

---

## 1. LLVM: exactly what it costs and what it buys

### 1.1 Where LLVM is actually used (measured, not assumed)

LLVM/Clang headers are included in **only 8 source files**; the rest of the tree
(runtime, api shims, cluster, `libvgre_nn`, xla) is already LLVM-free.

| File | Uses LLVM for | Weight |
|---|---|---|
| `src/compiler/llvm_translation_engine.cpp` | build LLVM IR, ORC JIT | **core JIT** |
| `src/compiler/llvm_translation_codegen.cpp` | native codegen | **core JIT** |
| `src/compiler/clang_kernel_parser.cpp` | Clang AST parse of CUDA C | **core JIT** |
| `src/compiler/clang_kernel_analysis.cpp` | Clang AST analysis | **core JIT** |
| `src/compiler/kernel_cache.cpp` | cache compiled IR | core JIT |
| `src/compiler/llvm_state_internal.h` | shared LLVM state | core JIT |
| `src/advanced/mps_control.cpp` | **`llvm::json::parse` only** | trivial |
| `src/advanced/tcp_cluster/configuration_manager_file_io.cpp` | **`llvm::json::parse` only** | trivial |

LLVM components linked (`CMakeLists.txt` §7): `Core ExecutionEngine OrcJIT
Support Target native`. It is pulled in unconditionally —
`find_package(LLVM REQUIRED CONFIG)` — so **there is no way to build without it
today**.

### 1.2 What that costs

- **Download/instal:** `llvm-18-dev` + `libclang-18-dev` ≈ 1 GB unpacked; the
  Windows CI pulls the ~500 MB `clang+llvm-18.1.8` tarball and unpacks to ~2 GB.
  This is the direct cause of the reported *"Windows build takes very long and
  never completes."*
- **Build time:** every TU that includes LLVM headers is heavy; the LLVM link is
  the long pole.
- **Version lock:** VGRE targets the LLVM-18 C++ API (`ThreadSafeContext::
  getContext`, `getHostCPUFeatures(StringMap&)`), which changed in 19+, so users
  must pin exactly 18 (macOS `llvm@18`, Windows tarball, Linux `llvm-18-dev`).

### 1.3 What we already have that is LLVM-free

The machinery to **execute** GPU code without LLVM already exists in-tree:

- **`src/compiler/ptx/ptx_translator.cpp`** — a hand-written PTX→host translation
  map (no LLVM). Translates PTX instructions to host C/inline-asm.
- **`src/debug/ptx_interpreter.cpp`** (693 LOC) — a real PTX interpreter:
  registers, memory spaces, barriers, predication, breakpoints, single-step. It
  executes PTX with **zero compilation**. Today it is wired only to the GDB RSP
  server (`src/debug/gdb_rsp_server.cpp`) for debugging — **not** to the runtime
  kernel-launch path.
- **`src/common/json.cpp`** — an in-tree JSON parser already used by
  `jwt_verifier`, `safetensors`, `tokenizer`.

**Conclusion:** the only thing LLVM is truly required for is the *CUDA-C →
native* JIT frontend. Execution and PTX handling can already be done without it.

---

## 2. LLVM-reduction plan (staged, lowest-risk first)

### Stage 0 — Drop LLVM from the cluster/advanced module (quick win, ~½ day)

`mps_control.cpp` and `configuration_manager_file_io.cpp` include *all of LLVM*
only for `llvm::json::parse`. Replace with the in-tree `vgre::common::json`
(already used elsewhere). This is the **only** reason `src/advanced` links
`vgre_llvm_iface` (see `src/advanced/CMakeLists.txt:92`); after the swap the
whole advanced/cluster module is LLVM-free. **No behaviour change, no new deps.**

### Stage 1 — Make LLVM optional behind a build flag (~2–3 days)

Add `option(VGRE_ENABLE_JIT "Compile CUDA C via the LLVM ORC JIT" ON)`.

- `ON` (default): today's behaviour — full CUDA-C JIT, best performance.
- `OFF`: skip `find_package(LLVM)`, exclude the 6 JIT files from `vgre_compiler`,
  and compile a **PTX-execution backend** instead. No LLVM download at all →
  small, fast build that completes on Windows in minutes.

Guard the 6 JIT files and the `find_package(LLVM REQUIRED)` block on the flag.
`vgre_compiler` keeps `kernel_parser.cpp`, all of `ptx/`, `sass/`, fusion, and
texture builtins (all LLVM-free).

### Stage 2 — Promote the PTX interpreter to a runtime execution backend (~3–5 days)

Move `ptx_interpreter` from `src/debug` into a first-class `ExecutionBackend`
consumed by `runtime_engine`'s launch path (today launch → ORC JIT symbol). Add
a backend interface with two implementations:

- `JitBackend` (LLVM, `VGRE_ENABLE_JIT=ON`) — unchanged fast path.
- `InterpreterBackend` (LLVM-free) — runs PTX through `ptx_interpreter`.

Select at runtime (`VGRE_EXEC_BACKEND=jit|interp`) with the interpreter as the
automatic fallback when JIT is compiled out. Correctness is shared by running the
existing kernel tests against both backends.

### Stage 3 — A from-scratch CUDA-C → PTX frontend for the common subset (research)

The remaining gap in `VGRE_ENABLE_JIT=OFF` mode is *CUDA C → PTX* without Clang.
Two options, in order of effort:

1. **PTX-in mode (cheap, ship first):** in LLVM-free builds accept PTX directly
   (inline PTX, `.ptx` modules, or `nvcc -ptx` output) and run it through the
   interpreter. Covers "I already have PTX/SASS" and CI-on-Windows use.
2. **Hand-written CUDA-C-subset compiler (the real prize):** extend the existing
   regex `kernel_parser.cpp` into a small recursive-descent front-end that emits
   PTX for the common kernel subset (arithmetic, `threadIdx/blockIdx`, indexing,
   `if`, `for`, `__syncthreads`, global/shared loads/stores). This is the
   "build it from scratch for the features we need" path. It will **not** cover
   all of CUDA C — but it removes LLVM for the workloads VGRE actually
   demonstrates (vector add, matrix math, reductions, the docs' examples).

**Recommended:** ship Stages 0–2 + Stage 3.1 first (they already unblock a
lightweight, Windows-friendly, LLVM-free build). Treat Stage 3.2 as an ongoing
track with a documented, growing "supported CUDA-C subset" list.

### Effort / payoff summary

| Stage | Effort | Removes LLVM from | Payoff |
|---|---|---|---|
| 0 | ½ day | cluster/advanced | smaller link, cleaner deps |
| 1 | 2–3 d | optional everywhere | **no LLVM download**, fast Windows build |
| 2 | 3–5 d | execution | run kernels with zero compiler |
| 3.1 | 1–2 d | — | LLVM-free path usable (PTX-in) |
| 3.2 | ongoing | CUDA-C frontend | LLVM-free for the common CUDA subset |

---

## 3. Repository hygiene issues (attention needed)

1. **~4.9 GB of stale local build trees** (not tracked, good): `build/`,
   `build-asan/`, `build_asan/`, `build_jit/`, `build_local/`, `build_rdma_fix/`,
   `build_test/`, `build-tsan/`, plus `venv/` (82 MB), stray root artifacts
   (`libvgre.so`, `libvgre_cudart.so`, `opencl_adapter.o`, `vgre_trace.json`).
   *Action:* delete the ones you're not using; extend `.gitignore` from the
   current `build/` + `build_local/` to `build*/` so the ASan/TSan/test trees are
   ignored too.
2. **`src/backup/`** exists as a source dir — confirm it's a real module and not a
   leftover; if leftover, remove.
3. **Two JSON parsers in play** (`vgre::common::json` vs `llvm::json`) — Stage 0
   consolidates onto the in-tree one.

---

## 4. Documentation truth reconciliation

Several top-level docs describe a world that predates the recent work and are now
**inaccurate**. Corrected in this pass (README, PROJECT_STATUS); the rest of the
list is tracked here.

| Claim in docs (old) | Reality (2026-09) |
|---|---|
| PROJECT_STATUS: *"single hard blocker is GitHub Actions billing; every CI run since 2026-06-22 fails"* | **False now.** CI runs free on the public repo on every push; Linux green (required), macOS green, Windows builds (informational). |
| README/STATUS: *"macOS full serial ctest bring-up in progress"* | macOS CI is **green** (fixed `test_mamba` FMA tolerance + `PythonCAPIVectorAdd` ctypes ABI). |
| README: *"Windows code-complete but CI-unverified"* | Windows now **builds in CI** (LLVM-18 tarball cached); still informational until fully green. |
| No mention of a public demo | **Live free demo:** `https://vgrengine.streamlit.app` (Streamlit Community Cloud; HF now requires PRO for server-side Spaces). |
| Test count "293" | ~300 locally (see memory / latest run). |
| Dates stamped "July 2026" | It is **September 2026**. |

**Also stale but not yet rewritten (follow-up):** the generated HTML site under
`docs/site/*.html` — regenerate with `docs/site/build_site.py` after the
markdown is settled so `installation.html`, `running-cuda.html`, `faq.html`, and
`index.html` reflect the LLVM-optional build and the Streamlit demo.

---

## 5. Suggested vNext milestone

1. Stage 0 (JSON) + `.gitignore` fix + doc-truth pass (this document + README +
   PROJECT_STATUS). *(docs done in this commit.)*
2. Stage 1 `VGRE_ENABLE_JIT` flag + Stage 2 interpreter backend, with the kernel
   test-suite run against both backends in CI.
3. Stage 3.1 PTX-in mode; publish a "supported CUDA-C subset" doc and begin 3.2.
4. Regenerate the docs site; add a "Lightweight / no-LLVM build" page.
