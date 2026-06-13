# VGRE TensorFlow backend (XLA StableHLO → VGRE HLO)

Runs **real TensorFlow (XLA) graphs** on the VGRE HLO engine, validated against
TensorFlow's own output. A `tf.function(..., jit_compile=True)` is XLA-compiled
and its StableHLO is translated to VGRE HLO and executed — the same translator
used for the JAX backend (`frameworks/jax/vgre_jax`).

## Two-process design (required)

TensorFlow and jaxlib each bundle their **own** MLIR/LLVM; importing both into one
process and parsing MLIR segfaults. So lowering is isolated:

```
tf_worker.py  (TF process)         test_tf_e2e.py  (TF-free driver)
  tf.function(jit_compile=True)
  experimental_get_compiler_ir       parse StableHLO (jaxlib MLIR)
    -> StableHLO text  ───file───►    -> Translator -> VGRE HLO
  run eager -> reference  ──────►      -> execute on VGRE, compare to reference
```

`tf_worker.py` dumps `m.mlir` + `in_*.npy` + `ref.npy` per case; the driver parses
and runs them on VGRE. The driver never imports TensorFlow.

## Run

```bash
.venv/bin/python frameworks/tf/test_tf_e2e.py     # 16/16 vs TF
# or:  ctest -R TensorFlowStableHloBackend
```

Covered: matmul, dense+relu, elementwise, tanh/sigmoid/softmax, reductions
(sum/mean/max), transpose, rsqrt-norm, concat, **conv2d**, **max-pool**,
**avg-pool** (`reduce_window`). TF emits `stablehlo.dot` and `stablehlo.reduce_window`
where JAX uses `dot_general`/explicit forms — both are handled.

## Scope

Same as the JAX backend: this is the XLA-HLO **compiler-backend** path (consume
the StableHLO a framework emits, run it on VGRE). A device-level TF plugin would
need the in-process PJRT toolchain (upstream `pjrt_c_api.h` + MLIR C++ libs) that
the wheels don't ship.
