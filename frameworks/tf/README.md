# VGRE TensorFlow backend (XLA StableHLO → VGRE HLO)

Runs **real TensorFlow (XLA) graphs** on the VGRE HLO engine, validated against
TensorFlow's own output. A `tf.function(..., jit_compile=True)` is XLA-compiled
and its StableHLO is translated to VGRE HLO and executed — the same translator
used for the JAX backend (`frameworks/jax/vgre_jax`).

## In-process (TensorFlow + VGRE in one process)

```
tf.function(jit_compile=True).experimental_get_compiler_ir(stage='stablehlo')
  -> parse StableHLO (jaxlib MLIR) -> Translator -> VGRE HLO engine -> result
                                                      (vs TF eager output)
```

libvgre statically links LLVM (for the JIT) and dynamically links system
protobuf/abseil (gRPC transport); TensorFlow bundles its **own** LLVM + protobuf.
To coexist in one process, libvgre is loaded with **`RTLD_DEEPBIND`** (see
`vgre_jax/runtime.py`) so it binds to its own copies of those libraries instead of
TensorFlow's globals, and it is linked with `--exclude-libs ALL` + a version
script (`cmake/llvm_isolation.map`) so its bundled LLVM symbols stay private. No
subprocess isolation is needed. (`tf_worker.py` holds the case definitions and can
still dump StableHLO to a directory for standalone debugging.)

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
