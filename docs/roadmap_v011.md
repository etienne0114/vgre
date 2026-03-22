# VGRE Version 0.1.1 Roadmap

Based on the **Zero-Simulation Audit** conducted in Phase 7, the following enhancements are planned for the v0.1.1 release to further harden the authoritative nature of the VGRE system.

## 1. Static IR FLOP Counting (High Priority)
Currently, some kernels (like those on iGPU) fallback to an estimated 10 FLOPs/thread baseline if metadata is unavailable.
- **Goal**: Implement an LLVM-based static analysis pass to count actual floating-point operations within the IR before execution.
- **Benefit**: 100% precise GFLOPS reporting for all kernels without heuristics.

## 2. Dynamic JIT Fusion
Currently, kernels are executed sequentially even if they have simple data dependencies.
- **Goal**: Implement a fusion engine that identifies sequential kernels sharing the same memory buffers and fuses them into a single JIT-compiled task.
- **Benefit**: Reduced host-side launch overhead and improved data locality.

## 3. Advanced Cluster Load Balancing
The current "Proportional Load Balancing" is cores-based.
- **Goal**: Incorporate real-time "Ground Truth" performance metrics (latencies, BW) into the partitioning algorithm.
- **Benefit**: Dynamic adaptation to varying network conditions and node performance fluctuations.

## 4. Dashboard Enhancements
- **Kernel Comparison**: Side-by-side comparison of different kernel versions (Source vs. Transpiled IR).
- **Trace Export (chrome://tracing)**: Export authoritative execution traces for analysis in external tools.
- **Cluster Heatmap**: Visualizing thermal and power distribution across the entire cluster.

## 5. Security Hardening
- **Dynamic Key Rotation**: Periodically rotate the session key for long-running cluster compute tasks.
- **Hardware-Backed Tokens**: Integration with TPM/Secure Enclave for auth token storage.
