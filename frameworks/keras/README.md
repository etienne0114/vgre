# VGRE Keras backend (high-level models on VGRE)

Runs **complete Keras 3 neural networks** on the VGRE HLO engine, validated
against Keras' own inference. This is the high-level DL framework layer on top of
the JAX/StableHLO path: a Keras model's forward pass is jax-traceable (Keras JAX
backend), so it lowers to StableHLO and runs on VGRE unchanged.

```
KERAS_BACKEND=jax
keras.Model(x, training=False)
  -> jax.jit -> StableHLO  -> VGRE translator -> VGRE HLO engine -> result
                                                   (vs keras model(x))
```

## Verified models (test_keras_e2e.py, 5/5 vs Keras)

| Model | Layers exercised |
|-------|------------------|
| MLP classifier | Dense, relu/tanh, softmax |
| CNN classifier | Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, softmax |
| BatchNorm MLP | Dense, **BatchNormalization** (inference moving stats), relu |
| LayerNorm + GELU MLP | Dense, **LayerNormalization**, **GELU** (erf) |
| Transformer block | **MultiHeadAttention**, LayerNormalization, residual + FFN |

A full transformer encoder block and a conv net both run on the CPU emulator with
no model changes — every op they emit (convolution, dot_general/attention,
reduce_window pooling, batch/layer norm, gelu/erf, softmax) is in the engine.

## Run

```bash
.venv/bin/python frameworks/keras/test_keras_e2e.py     # 5/5
# or:  ctest -R KerasModelsBackend
```

```python
import os; os.environ["KERAS_BACKEND"] = "jax"
import keras, numpy as np, vgre_keras
model = keras.Sequential([...])
out = vgre_keras.run_model(model, x)     # executed on VGRE
```
