# VGRE JAX backend (StableHLO → VGRE HLO)

Runs **real JAX-jitted functions** on the VGRE HLO engine. No mocks: JAX lowers a
`jax.jit` function to StableHLO exactly as it would for any backend; VGRE then
translates that StableHLO and executes it, and the result is checked against
JAX's own CPU backend.

## Path

```
jax.jit(f).lower(*args).compiler_ir('stablehlo')      # real JAX lowering → StableHLO MLIR
   └─ vgre_jax.Translator: walk the MLIR ops, rebuild as a VGRE HLO module
        via the builder C ABI (vgre_xla_b_* in libvgre.so)
        └─ vgre_xla_b_compile  → executable on the VGRE engine
             └─ vgre_xla_execute → float32 result
```

The translator (`vgre_jax/translate.py`) maps StableHLO ops onto VGRE HLO:
elementwise add/sub/mul/div/max/min/pow, unary negate/exp/log/tanh/abs/rsqrt,
`dot_general` (2-D matmul), `broadcast_in_dim` (incl. degenerate size-1
stretching), `reshape`, `transpose`, `compare`, `select`, `reduce`
(sum/max/min/prod), `constant`, `convert` (f32 identity), and inlines
`func.call` (e.g. the `@_where` helper JAX outlines). Unsupported ops raise —
nothing is silently approximated.

## Run

```bash
.venv/bin/python frameworks/jax/test_jax_e2e.py      # 13/13 vs jax CPU
# or via ctest:  ctest -R JaxStableHloBackend
```

```python
import jax, jax.numpy as jnp, numpy as np, vgre_jax
f = jax.jit(lambda x, W, b: jnp.maximum(x @ W + b, 0.0))
out = vgre_jax.run(f, x, W, b)        # executed on the VGRE engine
```

## Scope / what this is not

This is the **compiler-backend** half of an XLA/PJRT integration: it consumes the
StableHLO a framework emits and runs it on VGRE. It is **not** an in-process PJRT
*plugin* registering VGRE as a `jax` device for transparent
`jax.jit(..., backend='vgre')` dispatch. That additionally requires:

- the version-pinned upstream `pjrt_c_api.h` (matching the installed `jaxlib`),
  which is **not shipped** in the jaxlib wheel, and
- the MLIR C++ libraries to parse the StableHLO **bytecode** inside the plugin's
  `Compile` entry point (the wheel ships only Python MLIR extension modules, not
  linkable dev libraries).

Both would need XLA built from source. The StableHLO-level path here delivers the
identical compute without that toolchain.
