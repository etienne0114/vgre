"""ctypes bindings to VGRE's XLA HLO builder + executable C ABI (libvgre.so).

This is the real bridge: the StableHLO translator constructs an HLO module
op-by-op through these calls (into the actual C++ HloModule), compiles it to an
executable, and runs it on the VGRE engine. No reimplementation of the IR on the
Python side — Python only orchestrates.
"""
from __future__ import annotations

import ctypes
import os
import sys
from ctypes import c_char_p, c_float, c_int, c_int64, c_uint64, c_void_p, POINTER

import numpy as np

# HloOp enum ordinals (must match include/vgre/xla/hlo.h).
OP = {
    "Parameter": 0, "Constant": 1, "Iota": 2,
    "Add": 3, "Subtract": 4, "Multiply": 5, "Divide": 6,
    "Maximum": 7, "Minimum": 8, "Power": 9,
    "Negate": 10, "Exp": 11, "Log": 12, "Tanh": 13, "Abs": 14, "Rsqrt": 15,
    "Sqrt": 16, "Sign": 17, "Floor": 18, "Ceil": 19, "Logistic": 20,
    "Compare": 21, "Select": 22,
    "Broadcast": 23, "Reshape": 24, "Transpose": 25,
    "Dot": 26, "Reduce": 27,
    "DotGeneral": 28, "Concatenate": 29, "Slice": 30, "Pad": 31,
    "Convolution": 32, "Gather": 33, "ReduceWindow": 34,
    "Erf": 35, "Erfc": 36, "Cos": 37, "Sin": 38,
    "While": 39, "Tuple": 40, "GetTupleElement": 41,
    "DynamicSlice": 42, "DynamicUpdateSlice": 43, "Reverse": 44,
    "Sort": 45, "ReduceGeneral": 46, "And": 47, "Or": 48, "Xor": 49, "Not": 50,
    "Scatter": 51, "IsFinite": 52, "RoundNearestEven": 53, "RoundNearestAfz": 54,
}

_I64P = POINTER(c_int64)
_F32P = POINTER(c_float)


def _lib_names() -> tuple:
    """Platform-specific shared-library file names for libvgre."""
    if sys.platform == "win32":
        return ("vgre.dll", "libvgre.dll")
    if sys.platform == "darwin":
        return ("libvgre.dylib", "libvgre.so")
    return ("libvgre.so",)


def _find_lib() -> str:
    env = os.environ.get("VGRE_LIB")
    if env and os.path.exists(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    # frameworks/jax/vgre_jax -> repo root
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    build_dir = os.environ.get("VGRE_BUILD_DIR", os.path.join(root, "build"))
    for d in (build_dir, os.path.join(root, "build")):
        for name in _lib_names():
            cand = os.path.join(d, name)
            if os.path.exists(cand):
                return cand
    raise FileNotFoundError(
        f"VGRE shared library ({'/'.join(_lib_names())}) not found; "
        "set VGRE_LIB or VGRE_BUILD_DIR, or build the project")


def _load_libvgre(path: str):
    """Load libvgre so it coexists in-process with TensorFlow / jaxlib.

    libvgre statically links LLVM (for the JIT) and dynamically links system
    protobuf/abseil (gRPC transport). TensorFlow bundles its OWN LLVM + protobuf
    with global symbols; on glibc, if TF is loaded first, libvgre's references
    would bind to TF's copies (different state) and crash at dlopen. RTLD_DEEPBIND
    makes libvgre prefer its own dependency tree over already-loaded globals.
    RTLD_LOCAL keeps libvgre's symbols out of the global scope.

    The RTLD_* flags are POSIX-only and absent on Windows, where the default
    loader semantics already isolate each DLL's imports — so we just load plainly.
    """
    if sys.platform == "win32":            # Windows: no dlopen flags; default isolation
        return ctypes.CDLL(path)
    mode = getattr(os, "RTLD_NOW", 0) | getattr(os, "RTLD_LOCAL", 0)
    deepbind = getattr(os, "RTLD_DEEPBIND", 0)  # Linux/glibc only; 0 on macOS
    try:
        return ctypes.CDLL(path, mode=mode | deepbind)
    except OSError:
        return ctypes.CDLL(path, mode=mode)  # fallback if DEEPBIND unsupported


def _i64arr(xs):
    arr = (c_int64 * len(xs))(*[int(v) for v in xs])
    return arr, len(xs)


class VgreHlo:
    """Thin owner of one libvgre handle, exposing the builder + execute ABI."""

    def __init__(self, lib_path: str | None = None):
        self.lib = _load_libvgre(lib_path or _find_lib())
        L = self.lib
        L.vgre_xla_last_error.restype = c_char_p
        L.vgre_xla_builder_new.restype = c_uint64
        L.vgre_xla_b_parameter.argtypes = [c_uint64, c_int, _I64P, c_int]
        L.vgre_xla_b_parameter.restype = c_int
        L.vgre_xla_b_constant.argtypes = [c_uint64, _I64P, c_int, _F32P, c_int]
        L.vgre_xla_b_constant.restype = c_int
        L.vgre_xla_b_binary.argtypes = [c_uint64, c_int, c_int, c_int]
        L.vgre_xla_b_binary.restype = c_int
        L.vgre_xla_b_unary.argtypes = [c_uint64, c_int, c_int]
        L.vgre_xla_b_unary.restype = c_int
        L.vgre_xla_b_broadcast.argtypes = [c_uint64, c_int, _I64P, c_int, _I64P, c_int]
        L.vgre_xla_b_broadcast.restype = c_int
        L.vgre_xla_b_reshape.argtypes = [c_uint64, c_int, _I64P, c_int]
        L.vgre_xla_b_reshape.restype = c_int
        L.vgre_xla_b_transpose.argtypes = [c_uint64, c_int, _I64P, c_int]
        L.vgre_xla_b_transpose.restype = c_int
        L.vgre_xla_b_dot.argtypes = [c_uint64, c_int, c_int]
        L.vgre_xla_b_dot.restype = c_int
        L.vgre_xla_b_reduce.argtypes = [c_uint64, c_int, _I64P, c_int, c_char_p, c_float]
        L.vgre_xla_b_reduce.restype = c_int
        L.vgre_xla_b_compare.argtypes = [c_uint64, c_int, c_int, c_char_p]
        L.vgre_xla_b_compare.restype = c_int
        L.vgre_xla_b_select.argtypes = [c_uint64, c_int, c_int, c_int]
        L.vgre_xla_b_select.restype = c_int
        L.vgre_xla_b_dot_general.argtypes = [c_uint64, c_int, c_int, _I64P, c_int, _I64P, c_int,
                                             _I64P, c_int, _I64P, c_int, _I64P, c_int]
        L.vgre_xla_b_dot_general.restype = c_int
        L.vgre_xla_b_concatenate.argtypes = [c_uint64, POINTER(c_int), c_int, c_int64, _I64P, c_int]
        L.vgre_xla_b_concatenate.restype = c_int
        L.vgre_xla_b_slice.argtypes = [c_uint64, c_int, _I64P, _I64P, _I64P, c_int, _I64P, c_int]
        L.vgre_xla_b_slice.restype = c_int
        L.vgre_xla_b_pad.argtypes = [c_uint64, c_int, c_int, _I64P, _I64P, _I64P, c_int, _I64P, c_int]
        L.vgre_xla_b_pad.restype = c_int
        L.vgre_xla_b_convolution.argtypes = [c_uint64, c_int, c_int, _I64P, c_int,
                                             c_int, c_int, c_int, c_int, c_int, c_int,
                                             _I64P, _I64P, _I64P, c_int,
                                             _I64P, _I64P, _I64P, _I64P, c_int]
        L.vgre_xla_b_convolution.restype = c_int
        L.vgre_xla_b_gather.argtypes = [c_uint64, c_int, c_int, _I64P, c_int, _I64P, c_int,
                                        _I64P, c_int, _I64P, c_int, _I64P, c_int, c_int]
        L.vgre_xla_b_gather.restype = c_int
        L.vgre_xla_b_reduce_window.argtypes = [c_uint64, c_int, c_int, c_char_p, _I64P, c_int,
                                               _I64P, _I64P, _I64P, _I64P, _I64P, _I64P, c_int]
        L.vgre_xla_b_reduce_window.restype = c_int
        _IP = POINTER(c_int)
        L.vgre_xla_b_while.argtypes = [c_uint64, _IP, c_int, c_uint64, c_uint64, _I64P, c_int]
        L.vgre_xla_b_while.restype = c_int
        L.vgre_xla_b_tuple.argtypes = [c_uint64, _IP, c_int]
        L.vgre_xla_b_tuple.restype = c_int
        L.vgre_xla_b_get_tuple_element.argtypes = [c_uint64, c_int, c_int, _I64P, c_int]
        L.vgre_xla_b_get_tuple_element.restype = c_int
        L.vgre_xla_b_dynamic_slice.argtypes = [c_uint64, c_int, _IP, c_int, _I64P, c_int, _I64P, c_int]
        L.vgre_xla_b_dynamic_slice.restype = c_int
        L.vgre_xla_b_dynamic_update_slice.argtypes = [c_uint64, c_int, c_int, _IP, c_int, _I64P, c_int]
        L.vgre_xla_b_dynamic_update_slice.restype = c_int
        L.vgre_xla_b_reverse.argtypes = [c_uint64, c_int, _I64P, c_int]
        L.vgre_xla_b_reverse.restype = c_int
        L.vgre_xla_b_iota.argtypes = [c_uint64, _I64P, c_int, c_int64]
        L.vgre_xla_b_iota.restype = c_int
        L.vgre_xla_b_sort.argtypes = [c_uint64, POINTER(c_int), c_int, c_int64, c_uint64]
        L.vgre_xla_b_sort.restype = c_int
        L.vgre_xla_b_reduce_general.argtypes = [c_uint64, POINTER(c_int), c_int, _I64P, c_int,
                                                c_uint64, _I64P, c_int]
        L.vgre_xla_b_reduce_general.restype = c_int
        L.vgre_xla_b_scatter.argtypes = [c_uint64, c_int, c_int, c_int, c_uint64,
                                         _I64P, c_int, _I64P, c_int, _I64P, c_int, c_int64]
        L.vgre_xla_b_scatter.restype = c_int
        L.vgre_xla_b_set_root.argtypes = [c_uint64, c_int]
        L.vgre_xla_b_compile.argtypes = [c_uint64]
        L.vgre_xla_b_compile.restype = c_uint64
        L.vgre_xla_b_free.argtypes = [c_uint64]
        L.vgre_xla_execute.argtypes = [c_uint64, POINTER(_F32P), _I64P, c_int, _F32P, c_int64]
        L.vgre_xla_execute.restype = c_int64
        L.vgre_xla_output_numel.argtypes = [c_uint64]
        L.vgre_xla_output_numel.restype = c_int64
        L.vgre_xla_execute_multi.argtypes = [c_uint64, POINTER(_F32P), _I64P, c_int, _F32P, c_int64]
        L.vgre_xla_execute_multi.restype = c_int64
        L.vgre_xla_free.argtypes = [c_uint64]

    # ── builder ──────────────────────────────────────────────────────────────
    def new_builder(self) -> int:
        return self.lib.vgre_xla_builder_new()

    def parameter(self, b, index, dims):
        a, n = _i64arr(dims)
        return _ck(self.lib.vgre_xla_b_parameter(b, index, a, n), "parameter")

    def constant(self, b, dims, data):
        a, n = _i64arr(dims)
        flat = np.asarray(data, dtype=np.float32).ravel()
        fa = (c_float * len(flat))(*flat.tolist())
        return _ck(self.lib.vgre_xla_b_constant(b, a, n, fa, len(flat)), "constant")

    def binary(self, b, op, lhs, rhs):
        return _ck(self.lib.vgre_xla_b_binary(b, OP[op], lhs, rhs), "binary")

    def unary(self, b, op, x):
        return _ck(self.lib.vgre_xla_b_unary(b, OP[op], x), "unary")

    def broadcast(self, b, x, out_dims, bcast_dims):
        oa, on = _i64arr(out_dims)
        ba, bn = _i64arr(bcast_dims)
        return _ck(self.lib.vgre_xla_b_broadcast(b, x, oa, on, ba, bn), "broadcast")

    def reshape(self, b, x, out_dims):
        a, n = _i64arr(out_dims)
        return _ck(self.lib.vgre_xla_b_reshape(b, x, a, n), "reshape")

    def transpose(self, b, x, perm):
        a, n = _i64arr(perm)
        return _ck(self.lib.vgre_xla_b_transpose(b, x, a, n), "transpose")

    def dot(self, b, lhs, rhs):
        return _ck(self.lib.vgre_xla_b_dot(b, lhs, rhs), "dot")

    def reduce(self, b, x, dims, kind, init):
        a, n = _i64arr(dims)
        return _ck(self.lib.vgre_xla_b_reduce(b, x, a, n, kind.encode(), float(init)), "reduce")

    def compare(self, b, lhs, rhs, direction):
        return _ck(self.lib.vgre_xla_b_compare(b, lhs, rhs, direction.encode()), "compare")

    def select(self, b, pred, on_true, on_false):
        return _ck(self.lib.vgre_xla_b_select(b, pred, on_true, on_false), "select")

    def dot_general(self, b, lhs, rhs, lb, rb, lc, rc, out_dims):
        la, ln = _i64arr(lb); ra, rn = _i64arr(rb)
        ca, cn = _i64arr(lc); da, dn = _i64arr(rc)
        oa, on = _i64arr(out_dims)
        return _ck(self.lib.vgre_xla_b_dot_general(b, lhs, rhs, la, ln, ra, rn, ca, cn, da, dn, oa, on),
                   "dot_general")

    def concatenate(self, b, xs, dim, out_dims):
        xa = (c_int * len(xs))(*[int(v) for v in xs])
        oa, on = _i64arr(out_dims)
        return _ck(self.lib.vgre_xla_b_concatenate(b, xa, len(xs), int(dim), oa, on), "concatenate")

    def slice(self, b, x, starts, limits, strides, out_dims):
        sa, n = _i64arr(starts); la, _ = _i64arr(limits); ta, _ = _i64arr(strides)
        oa, on = _i64arr(out_dims)
        return _ck(self.lib.vgre_xla_b_slice(b, x, sa, la, ta, n, oa, on), "slice")

    def pad(self, b, x, pad_val, low, high, interior, out_dims):
        la, n = _i64arr(low); ha, _ = _i64arr(high); ia, _ = _i64arr(interior)
        oa, on = _i64arr(out_dims)
        return _ck(self.lib.vgre_xla_b_pad(b, x, pad_val, la, ha, ia, n, oa, on), "pad")

    def convolution(self, b, lhs, rhs, out_dims, dn, strides, pad_lo, pad_hi, rhs_dil, groups):
        oa, on = _i64arr(out_dims)
        isa, nsp = _i64arr(dn["in_sp"]); ksa, _ = _i64arr(dn["k_sp"]); osa, _ = _i64arr(dn["out_sp"])
        sta, _ = _i64arr(strides); pla, _ = _i64arr(pad_lo); pha, _ = _i64arr(pad_hi); rda, _ = _i64arr(rhs_dil)
        return _ck(self.lib.vgre_xla_b_convolution(
            b, lhs, rhs, oa, on,
            dn["in_batch"], dn["in_feat"], dn["k_out"], dn["k_in"], dn["out_batch"], dn["out_feat"],
            isa, ksa, osa, nsp, sta, pla, pha, rda, int(groups)), "convolution")

    def gather(self, b, operand, indices, out_dims, offset_dims, collapsed, start_map,
               slice_sizes, index_vector_dim):
        oa, on = _i64arr(out_dims)
        ofa, nof = _i64arr(offset_dims); ca, nc = _i64arr(collapsed)
        sma, nsm = _i64arr(start_map); ssa, nss = _i64arr(slice_sizes)
        return _ck(self.lib.vgre_xla_b_gather(b, operand, indices, oa, on, ofa, nof, ca, nc,
                                              sma, nsm, ssa, nss, int(index_vector_dim)), "gather")

    def reduce_window(self, b, x, init, kind, out_dims, win_dims, win_strides,
                      pad_lo, pad_hi, base_dil, win_dil):
        oa, on = _i64arr(out_dims)
        wd, n = _i64arr(win_dims); ws, _ = _i64arr(win_strides)
        pl, _ = _i64arr(pad_lo); ph, _ = _i64arr(pad_hi)
        bd, _ = _i64arr(base_dil); wl, _ = _i64arr(win_dil)
        return _ck(self.lib.vgre_xla_b_reduce_window(b, x, init, kind.encode(), oa, on,
                                                     wd, ws, pl, ph, bd, wl, n), "reduce_window")

    def _intarr(self, xs):
        return (c_int * len(xs))(*[int(v) for v in xs])

    def while_op(self, b, inits, cond_builder, body_builder, primary_dims):
        ia = self._intarr(inits)
        pa, pn = _i64arr(primary_dims)
        return _ck(self.lib.vgre_xla_b_while(b, ia, len(inits), cond_builder, body_builder, pa, pn),
                   "while")

    def tuple(self, b, elems):
        ea = self._intarr(elems)
        return _ck(self.lib.vgre_xla_b_tuple(b, ea, len(elems)), "tuple")

    def get_tuple_element(self, b, src, index, out_dims):
        oa, on = _i64arr(out_dims)
        return _ck(self.lib.vgre_xla_b_get_tuple_element(b, src, index, oa, on), "get_tuple_element")

    def dynamic_slice(self, b, operand, starts, sizes, out_dims):
        sa = self._intarr(starts)
        za, zn = _i64arr(sizes)
        oa, on = _i64arr(out_dims)
        return _ck(self.lib.vgre_xla_b_dynamic_slice(b, operand, sa, len(starts), za, zn, oa, on),
                   "dynamic_slice")

    def dynamic_update_slice(self, b, operand, update, starts, out_dims):
        sa = self._intarr(starts)
        oa, on = _i64arr(out_dims)
        return _ck(self.lib.vgre_xla_b_dynamic_update_slice(b, operand, update, sa, len(starts), oa, on),
                   "dynamic_update_slice")

    def reverse(self, b, x, dims):
        a, n = _i64arr(dims)
        return _ck(self.lib.vgre_xla_b_reverse(b, x, a, n), "reverse")

    def iota(self, b, out_dims, dim):
        a, n = _i64arr(out_dims)
        return _ck(self.lib.vgre_xla_b_iota(b, a, n, int(dim)), "iota")

    def sort(self, b, operands, dim, cmp_builder):
        oa = self._intarr(operands)
        return _ck(self.lib.vgre_xla_b_sort(b, oa, len(operands), int(dim), cmp_builder), "sort")

    def reduce_general(self, b, operands, dims, body_builder, primary_dims):
        oa = self._intarr(operands)
        da, dn = _i64arr(dims)
        pa, pn = _i64arr(primary_dims)
        return _ck(self.lib.vgre_xla_b_reduce_general(b, oa, len(operands), da, dn,
                                                      body_builder, pa, pn), "reduce_general")

    def scatter(self, b, operand, indices, updates, combiner, uwd, iwd, dto, index_vector_dim):
        ua, un = _i64arr(uwd)
        ia, in_ = _i64arr(iwd)
        da, dn = _i64arr(dto)
        return _ck(self.lib.vgre_xla_b_scatter(b, operand, indices, updates, combiner,
                                               ua, un, ia, in_, da, dn, int(index_vector_dim)),
                   "scatter")

    def set_root(self, b, idx):
        self.lib.vgre_xla_b_set_root(b, idx)

    def compile(self, b) -> int:
        exe = self.lib.vgre_xla_b_compile(b)
        if exe == 0:
            raise RuntimeError("vgre_xla_b_compile failed")
        return exe

    def free_builder(self, b):
        self.lib.vgre_xla_b_free(b)

    # ── execute ──────────────────────────────────────────────────────────────
    def output_numel(self, exe) -> int:
        return self.lib.vgre_xla_output_numel(exe)

    def execute(self, exe, inputs: list[np.ndarray]) -> np.ndarray:
        bufs = [np.ascontiguousarray(x, dtype=np.float32).ravel() for x in inputs]
        ptr_arr = (_F32P * len(bufs))()
        numel_arr = (c_int64 * len(bufs))()
        keep = []
        for i, buf in enumerate(bufs):
            ca = (c_float * buf.size)(*buf.tolist())
            keep.append(ca)
            ptr_arr[i] = ctypes.cast(ca, _F32P)
            numel_arr[i] = buf.size
        cap = self.output_numel(exe)
        out = (c_float * cap)()
        n = self.lib.vgre_xla_execute(exe, ptr_arr, numel_arr, len(bufs), out, cap)
        if n < 0:
            raise RuntimeError(f"vgre_xla_execute failed: {self._last_error()}")
        return np.array(out[:n], dtype=np.float32)

    def execute_multi(self, exe, inputs, out_numels) -> list:
        """Execute a tuple-root module; returns one ndarray per output, split by
        the given per-output element counts."""
        bufs = [np.ascontiguousarray(x, dtype=np.float32).ravel() for x in inputs]
        ptr_arr = (_F32P * len(bufs))()
        numel_arr = (c_int64 * len(bufs))()
        keep = []
        for i, buf in enumerate(bufs):
            ca = (c_float * buf.size)(*buf.tolist())
            keep.append(ca)
            ptr_arr[i] = ctypes.cast(ca, _F32P)
            numel_arr[i] = buf.size
        total = int(sum(out_numels))
        out = (c_float * total)()
        n = self.lib.vgre_xla_execute_multi(exe, ptr_arr, numel_arr, len(bufs), out, total)
        if n < 0:
            raise RuntimeError(f"vgre_xla_execute_multi failed: {self._last_error()}")
        flat = np.array(out[:n], dtype=np.float32)
        res, off = [], 0
        for k in out_numels:
            res.append(flat[off:off + k]); off += k
        return res

    def _last_error(self) -> str:
        e = self.lib.vgre_xla_last_error()
        return e.decode() if e else "(no detail)"

    def free(self, exe):
        self.lib.vgre_xla_free(exe)


def _ck(rc: int, what: str) -> int:
    if rc < 0:
        raise RuntimeError(f"vgre builder op '{what}' failed (rc={rc})")
    return rc
