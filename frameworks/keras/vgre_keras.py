"""Run high-level Keras models on the VGRE HLO engine.

Keras 3 is a high-level DL framework; with the JAX backend a model's forward pass
is a normal jax-traceable function, so it lowers to StableHLO and runs on VGRE via
the existing JAX path (frameworks/jax/vgre_jax). This module is a thin convenience
wrapper: jit the model in inference mode, lower, translate, execute on VGRE.

Import order matters: set KERAS_BACKEND=jax before importing keras.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("KERAS_BACKEND", "jax")

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "jax"))

import jax  # noqa: E402
import numpy as np  # noqa: E402

import vgre_jax  # noqa: E402


def run_model(model, x, lib_path: str | None = None) -> np.ndarray:
    """Execute a Keras model's inference forward pass on VGRE.

    Returns the flat float32 output; compare against np.asarray(model(x))."""
    call = jax.jit(lambda z: model(z, training=False))
    return vgre_jax.run(call, np.asarray(x, np.float32), lib_path=lib_path)
