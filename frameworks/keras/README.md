# VGRE Keras backend (high-level models on VGRE)

Runs **complete Keras 3 neural networks** on the VGRE HLO engine, validated
against Keras' own inference. A Keras model's forward pass is jax-traceable
(Keras JAX backend), so it lowers to StableHLO and runs on VGRE unchanged.

```
KERAS_BACKEND=jax; keras model(x, training=False)
  -> jax.jit -> StableHLO -> VGRE translator -> VGRE HLO engine -> result
                                                  (vs keras model(x))
```

Runs in one process. Some layers (e.g. `Bidirectional`) pull in **TensorFlow**,
which bundles its own LLVM + protobuf; libvgre is loaded with **`RTLD_DEEPBIND`**
(see `vgre_jax/runtime.py`) plus a linker version script so it binds to its own
copies and coexists with TensorFlow without the LLVM/protobuf symbol collision.
(`keras_worker.py` holds the model builders.)

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
