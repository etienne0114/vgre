"""End-to-end: run real Keras 3 models on the VGRE HLO engine.

Builds complete neural networks with the high-level Keras API — CNN classifier,
BatchNorm/LayerNorm MLPs, a Transformer MultiHeadAttention block — runs each on
VGRE (Keras JAX backend → jax.jit → StableHLO → VGRE HLO) and asserts the output
matches Keras' own inference. Real models, real layers, no mocks.
"""
import os
import sys

os.environ["KERAS_BACKEND"] = "jax"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

try:
    import keras
    from keras import layers
    import vgre_keras
except ImportError as e:
    print(f"SKIP: prerequisites missing ({e})")
    sys.exit(0)

np.random.seed(0)
PASS = FAIL = 0


def check(name, model, x):
    global PASS, FAIL
    x = np.asarray(x, np.float32)
    model(x)  # build / initialise variables
    ref = np.asarray(model(x, training=False), np.float32).ravel()
    try:
        got = vgre_keras.run_model(model, x)
    except Exception as e:
        print(f"  FAIL  {name}: {type(e).__name__}: {str(e)[:80]}")
        FAIL += 1
        return
    ok = got.shape == ref.shape and np.allclose(got, ref, rtol=1e-3, atol=1e-3)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}"
          + ("" if ok else f"  (maxdiff {np.max(np.abs(got-ref)):.3g})"))
    PASS += ok
    FAIL += (not ok)


def mlp():
    return keras.Sequential([
        keras.Input((8,)), layers.Dense(16, activation="relu"),
        layers.Dense(8, activation="tanh"), layers.Dense(3, activation="softmax")])


def cnn():
    return keras.Sequential([
        keras.Input((1, 8, 8)),
        layers.Conv2D(4, 3, padding="same", activation="relu", data_format="channels_first"),
        layers.MaxPooling2D(2, data_format="channels_first"),
        layers.Conv2D(8, 3, padding="same", activation="relu", data_format="channels_first"),
        layers.AveragePooling2D(2, data_format="channels_first"),
        layers.Flatten(), layers.Dense(10, activation="softmax")])


def bn_mlp():
    return keras.Sequential([
        keras.Input((8,)), layers.Dense(16), layers.BatchNormalization(),
        layers.Activation("relu"), layers.Dense(4)])


def ln_mlp():
    return keras.Sequential([
        keras.Input((8,)), layers.Dense(16), layers.LayerNormalization(),
        layers.Activation("gelu"), layers.Dense(4)])


class TransformerBlock(keras.Model):
    def __init__(self):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=2, key_dim=8)
        self.ln1 = layers.LayerNormalization()
        self.ln2 = layers.LayerNormalization()
        self.ff1 = layers.Dense(32, activation="relu")
        self.ff2 = layers.Dense(16)

    def call(self, x, training=False):
        h = self.ln1(x + self.att(x, x, training=training))
        return self.ln2(h + self.ff2(self.ff1(h)))


def main():
    print("=== Keras 3 models → StableHLO → VGRE HLO (vs Keras) ===")
    check("MLP classifier", mlp(), np.random.randn(4, 8))
    check("CNN classifier", cnn(), np.random.randn(2, 1, 8, 8))
    check("BatchNorm MLP", bn_mlp(), np.random.randn(4, 8))
    check("LayerNorm+GELU MLP", ln_mlp(), np.random.randn(4, 8))
    check("Transformer block", TransformerBlock(), np.random.randn(2, 6, 16))
    print(f"\n{PASS} / {PASS + FAIL} passed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
