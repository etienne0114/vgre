# VGRE — Positioning & Go-to-Market

**Last Updated**: 2026-07-02

## Where VGRE sits in the 2026 local-AI market

The dominant local runtimes are Ollama (~174k stars, llama.cpp/MLX wrapper), LM Studio
(GUI), llama.cpp (engine), vLLM (multi-user serving), and MLX (Apple Silicon). All of
them do **inference only**. VGRE's differentiation is that it is one dependency-free
stack that no competitor covers end-to-end:

| Capability | llama.cpp / Ollama | vLLM | **VGRE** |
|---|---|---|---|
| CUDA/HIP runtime emulation (run GPU code with no GPU) | ✗ | ✗ | ✅ core product |
| Inference (GGUF Q4/Q8, safetensors, bf16/int4/int8) | ✅ | ✅ | ✅ in-tree |
| **Training/fine-tuning on CPU** (autograd, AdamW, LoRA) | ✗ (llama.cpp dropped training) | ✗ | ✅ in-tree |
| RAG retrieval (vector index) | ✗ (needs FAISS/Chroma) | ✗ | ✅ in-tree HNSW |
| Tokenizers (HF tokenizer.json, exact) | partial (GGUF-embedded) | via HF libs | ✅ in-tree, UCD-exact |
| Distributed (data/tensor/pipeline parallel, RDMA-style collectives) | ✗ | ✅ (GPU) | ✅ CPU/emulated |
| Enterprise (RBAC/OIDC, audit, PQC crypto, K8s operator, MIG) | ✗ | partial | ✅ in-tree |

**One-line pitch:** *"The whole local-AI stack — GPU emulation, training, fine-tuning,
RAG, and serving — in one lightweight, zero-dependency runtime. No GPU required, no
cloud bill, no vendor lock-in."*

## Who buys it, and why

1. **CUDA developers without NVIDIA hardware** (students, CI pipelines, ARM/laptop
   fleets): run and test real CUDA/HIP code anywhere. *Nobody else does this.*
2. **Privacy/cost-constrained teams** (legal, health, gov, finance): fully offline
   embed→index→retrieve→generate→fine-tune loop; data never leaves the machine.
3. **Edge/OEM integrators**: single small native library (no Python/torch runtime) —
   embeddable via the C ABI; per-device licensing.
4. **Enterprises needing auditability**: in-tree audit log, RBAC, ML-KEM-768 PQC,
   backup/DR — the compliance story local rivals lack.

## Adoption levers (ordered by cost-effectiveness)

- **Free core, paid enterprise** (open-core): engine + CLI free; RBAC/audit/operator/
  support paid. Matches buyer expectations set by Ollama-free / vLLM-enterprise.
- **Benchmarks as content**: publish reproducible CPU tokens/sec + recall@k + LoRA
  fine-tune-time numbers vs llama.cpp on the same hardware (scripts already in-tree:
  `scripts/run_benchmarks.sh`). Benchmarks are the #1 discovery channel in this market.
- **"Runs CUDA without a GPU" demos**: short screencasts of real CUDA kernels + GPT-2
  generation on a bare laptop; this is the hook no competitor can copy quickly.
- **One-command experience**: `pip install vgre` wheel (already built) + a model-pull
  command; friction parity with Ollama is table stakes.
- **CI marketplace**: GitHub Action "test your CUDA code without GPU runners" — turns
  the emulator into a recurring-spend developer tool (GPU CI minutes are expensive).
- **Education channel**: universities teaching CUDA without lab hardware; free academic
  tier seeds the future buyer base.

## Pricing sketch

- **Community** (free): engine, CLI, Python wheel, single node.
- **Pro** (per-seat): IDE/debug tooling, priority builds, Windows/macOS binaries.
- **Enterprise** (per-node/yr): RBAC/OIDC, audit, MIG partitioning, K8s operator,
  backup/DR, PQC, support SLA.
- **OEM/Edge** (per-device royalty): C-ABI embedding license.

## Near-term blockers to revenue (from missingFeatures.md)

- CI is dead (GitHub Actions billing) — Win/macOS binaries unverifiable → blocks Pro tier.
- A physical multi-node demo run + one frontier checkpoint (Llama-3-8B int4, ~4.5 GB)
  would complete the flagship demo story; both are hardware/download-gated, not code-gated.
