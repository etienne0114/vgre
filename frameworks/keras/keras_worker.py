"""Keras model lowering worker (runs in its own process).

Some Keras layers (e.g. Bidirectional) pull in TensorFlow, whose bundled LLVM
collides with libvgre.so's LLVM if co-loaded. So model construction + lowering
runs here, isolated: build the model (Keras JAX backend), jit + lower its
inference pass to StableHLO, and dump the text + inputs + Keras' own reference
output. The driver (test_keras_e2e.py, which loads libvgre) never imports Keras.

Usage:  python keras_worker.py <case-name> <out-dir>
"""
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

import jax
import keras
from keras import layers


def build(name):
    """Return (model, input ndarray) for a case. Fixed seed for reproducibility."""
    np.random.seed(0)
    keras.utils.set_random_seed(0)
    if name == "mlp":
        m = keras.Sequential([keras.Input((8,)), layers.Dense(16, activation="relu"),
                              layers.Dense(8, activation="tanh"), layers.Dense(3, activation="softmax")])
        x = np.random.randn(4, 8)
    elif name == "cnn":
        m = keras.Sequential([
            keras.Input((1, 8, 8)),
            layers.Conv2D(4, 3, padding="same", activation="relu", data_format="channels_first"),
            layers.MaxPooling2D(2, data_format="channels_first"),
            layers.Conv2D(8, 3, padding="same", activation="relu", data_format="channels_first"),
            layers.AveragePooling2D(2, data_format="channels_first"),
            layers.Flatten(), layers.Dense(10, activation="softmax")])
        x = np.random.randn(2, 1, 8, 8)
    elif name == "batchnorm":
        m = keras.Sequential([keras.Input((8,)), layers.Dense(16), layers.BatchNormalization(),
                              layers.Activation("relu"), layers.Dense(4)])
        x = np.random.randn(4, 8)
    elif name == "layernorm_gelu":
        m = keras.Sequential([keras.Input((8,)), layers.Dense(16), layers.LayerNormalization(),
                              layers.Activation("gelu"), layers.Dense(4)])
        x = np.random.randn(4, 8)
    elif name == "transformer":
        inp = keras.Input((6, 16))
        a = layers.MultiHeadAttention(num_heads=2, key_dim=8)(inp, inp)
        h = layers.LayerNormalization()(inp + a)
        f = layers.Dense(32, activation="relu")(h)
        out = layers.LayerNormalization()(h + layers.Dense(16)(f))
        m = keras.Model(inp, out)
        x = np.random.randn(2, 6, 16)
    elif name == "simple_rnn":
        m = keras.Sequential([keras.Input((5, 3)), layers.SimpleRNN(6),
                              layers.Dense(3, activation="softmax")])
        x = np.random.randn(2, 5, 3)
    elif name == "lstm":
        m = keras.Sequential([keras.Input((5, 3)), layers.LSTM(6),
                              layers.Dense(3, activation="softmax")])
        x = np.random.randn(2, 5, 3)
    elif name == "gru":
        m = keras.Sequential([keras.Input((5, 3)), layers.GRU(6),
                              layers.Dense(3, activation="softmax")])
        x = np.random.randn(2, 5, 3)
    elif name == "stacked_lstm":
        m = keras.Sequential([keras.Input((5, 3)), layers.LSTM(8, return_sequences=True),
                              layers.LSTM(4)])
        x = np.random.randn(2, 5, 3)
    elif name == "bidirectional":
        m = keras.Sequential([keras.Input((5, 3)), layers.Bidirectional(layers.LSTM(4)),
                              layers.Dense(3, activation="softmax")])
        x = np.random.randn(2, 5, 3)
    else:
        raise ValueError(f"unknown case {name}")
    return m, x.astype(np.float32)


def main():
    name, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    model, x = build(name)
    model(x)  # initialise variables
    call = jax.jit(lambda z: model(z, training=False))
    txt = call.lower(x).compiler_ir("stablehlo")
    with open(os.path.join(out_dir, "m.mlir"), "w") as f:
        f.write(str(txt))
    np.save(os.path.join(out_dir, "in_0.npy"), x)
    ref = np.asarray(call(x), np.float32).ravel()
    np.save(os.path.join(out_dir, "ref.npy"), ref)
    print(f"dumped {name}: in {x.shape}, ref {ref.size}")


if __name__ == "__main__":
    main()
