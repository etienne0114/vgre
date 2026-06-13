# VGRE Keras backend (high-level models on VGRE)

Runs **complete Keras 3 neural networks** on the VGRE HLO engine, validated
against Keras' own inference. A Keras model's forward pass is jax-traceable
(Keras JAX backend), so it lowers to StableHLO and runs on VGRE unchanged.

```
keras_worker.py  (subprocess)          test_keras_e2e.py  (libvgre driver)
  KERAS_BACKEND=jax; build model
  jax.jit(model).lower -> StableHLO ─file─► parse (jaxlib MLIR) -> VGRE HLO
  run -> reference ──────────────────────►  execute on VGRE, compare to Keras
```

Lowering runs in an isolated subprocess: some layers (e.g. `Bidirectional`) pull
in **TensorFlow**, whose bundled LLVM collides with libvgre.so's LLVM if loaded
in the same process. The driver never imports Keras/TF.

## Verified models (test_keras_e2e.py, 10/10 vs Keras)

| Model | Layers exercised |
|-------|------------------|
| MLP / CNN | Dense, Conv2D, Max/AveragePooling2D, Flatten, softmax |
| BatchNorm MLP | Dense, **BatchNormalization** (inference moving stats) |
| LayerNorm + GELU MLP | Dense, **LayerNormalization**, **GELU** (erf) |
| Transformer block | **MultiHeadAttention**, LayerNorm, residual + FFN |
| SimpleRNN / LSTM / GRU | recurrence via **while** + dynamic-slice/update |
| stacked LSTM | `return_sequences` + a second recurrent layer |
| **Bidirectional LSTM** | forward + reversed (`reverse`) passes, concatenated |

Transformers, conv nets **and** recurrent nets all run on the CPU emulator with
no model changes. The recurrent family exercises the engine's control flow
(`while` with embedded cond/body sub-computations, `dynamic_slice`,
`dynamic_update_slice`, `reverse`).

## Run

```bash
.venv/bin/python frameworks/keras/test_keras_e2e.py     # 10/10
# or:  ctest -R KerasModelsBackend
```

`vgre_keras.run_model(model, x)` also runs a Keras model in-process for models
that don't import TensorFlow.
