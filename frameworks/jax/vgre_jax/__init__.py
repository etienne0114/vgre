"""VGRE JAX backend: run JAX-jitted functions on the VGRE HLO engine.

End-to-end path (no mocks):
    jax.jit(f).lower(*args).compiler_ir('stablehlo')   # real JAX lowering
        -> Translator: walk the StableHLO MLIR, emit VGRE HLO via the builder ABI
        -> vgre_xla_b_compile  -> executable on the VGRE engine
        -> vgre_xla_execute    -> result, compared against jax.jit(f)(*args)

This is the XLA/HLO compiler-backend half. A full in-process PJRT *plugin*
(registering VGRE as a `jax` device for transparent dispatch) additionally needs
the version-pinned upstream pjrt_c_api.h plus the MLIR C++ libraries to parse the
StableHLO bytecode inside the plugin's Compile entry point — neither ships in the
jaxlib wheel. This module delivers the same compute path at the StableHLO level.
"""
from __future__ import annotations

import numpy as np

from .translate import Executable, Translator

__all__ = ["Translator", "Executable", "compile_jit", "run"]


def compile_jit(jitted, *args, lib_path: str | None = None) -> Executable:
    """Lower a jax.jit function at the given example args and compile for VGRE."""
    module = jitted.lower(*args).compiler_ir("stablehlo")
    return Translator(lib_path).compile_module(module)


def run(jitted, *args, lib_path: str | None = None) -> np.ndarray:
    """Compile + execute on VGRE, returning a flat float32 result."""
    exe = compile_jit(jitted, *args, lib_path=lib_path)
    try:
        return exe(*args)
    finally:
        exe.close()
