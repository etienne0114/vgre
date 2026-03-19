"""
VGRE Graph — Python-friendly CUDA Graphs/DAG API.
"""

from typing import List, Optional, Sequence, Tuple, Any

import ctypes
import numpy as np

from vgre.memory import DeviceArray
from vgre._native import (
    native_graph_create,
    native_graph_destroy,
    native_graph_instantiate,
    native_graph_exec_destroy,
    native_graph_launch,
    native_graph_add_kernel_node,
    native_graph_add_memcpy_node,
    native_graph_add_dependency,
    native_graph_update_kernel_node,
    native_graph_update_memcpy_node,
    VGRE_MEMCPY_HOST_TO_DEVICE,
    VGRE_MEMCPY_DEVICE_TO_HOST,
    VGRE_MEMCPY_DEVICE_TO_DEVICE,
    VGRE_ARG_POINTER,
    VGRE_ARG_INT32,
    VGRE_ARG_INT64,
    VGRE_ARG_FLOAT32,
    VGRE_ARG_FLOAT64,
    VGRE_ARG_UINT32,
    VGRE_ARG_UINT64,
)


def _infer_arg_type(value) -> int:
    if isinstance(value, DeviceArray):
        return VGRE_ARG_POINTER
    if isinstance(value, np.ndarray):
        return VGRE_ARG_POINTER
    if isinstance(value, int):
        return VGRE_ARG_INT32
    if isinstance(value, float):
        return VGRE_ARG_FLOAT32
    if isinstance(value, ctypes.c_int64):
        return VGRE_ARG_INT64
    if isinstance(value, ctypes.c_uint64):
        return VGRE_ARG_UINT64
    if isinstance(value, ctypes.c_double):
        return VGRE_ARG_FLOAT64
    return VGRE_ARG_POINTER


def _pack_args(
    args: Sequence,
    arg_types: Optional[Sequence[int]] = None,
) -> Tuple[List[ctypes.c_void_p], List[int], List[Any]]:
    packed: List[ctypes.c_void_p] = []
    types: List[int] = []
    keep_alive: List[Any] = []

    if arg_types is not None and len(arg_types) != len(args):
        raise ValueError("arg_types length must match args length")

    for idx, a in enumerate(args):
        t = arg_types[idx] if arg_types is not None else _infer_arg_type(a)

        if t == VGRE_ARG_POINTER:
            if isinstance(a, DeviceArray):
                ptr_val = ctypes.c_void_p(a.as_pointer().value)
            elif isinstance(a, np.ndarray):
                ptr_val = ctypes.c_void_p(a.ctypes.data)
            elif isinstance(a, ctypes.c_void_p):
                ptr_val = a
            else:
                ptr_val = ctypes.c_void_p(int(a))
            keep_alive.append(ptr_val)
            packed.append(ctypes.cast(ctypes.byref(ptr_val), ctypes.c_void_p))
        elif t == VGRE_ARG_INT64:
            val = a if isinstance(a, ctypes.c_int64) else ctypes.c_int64(int(a))
            keep_alive.append(val)
            packed.append(ctypes.cast(ctypes.byref(val), ctypes.c_void_p))
        elif t == VGRE_ARG_UINT64:
            val = a if isinstance(a, ctypes.c_uint64) else ctypes.c_uint64(int(a))
            keep_alive.append(val)
            packed.append(ctypes.cast(ctypes.byref(val), ctypes.c_void_p))
        elif t == VGRE_ARG_FLOAT64:
            val = a if isinstance(a, ctypes.c_double) else ctypes.c_double(float(a))
            keep_alive.append(val)
            packed.append(ctypes.cast(ctypes.byref(val), ctypes.c_void_p))
        elif t == VGRE_ARG_UINT32:
            val = ctypes.c_uint32(int(a))
            keep_alive.append(val)
            packed.append(ctypes.cast(ctypes.byref(val), ctypes.c_void_p))
        elif t == VGRE_ARG_INT32:
            val = ctypes.c_int32(int(a))
            keep_alive.append(val)
            packed.append(ctypes.cast(ctypes.byref(val), ctypes.c_void_p))
        elif t == VGRE_ARG_FLOAT32:
            val = ctypes.c_float(float(a))
            keep_alive.append(val)
            packed.append(ctypes.cast(ctypes.byref(val), ctypes.c_void_p))
        else:
            val = ctypes.c_void_p(int(a))
            keep_alive.append(val)
            packed.append(ctypes.cast(ctypes.byref(val), ctypes.c_void_p))
            t = VGRE_ARG_POINTER

        types.append(t)

    return packed, types, keep_alive


class Graph:
    """Python wrapper for VGRE CUDA Graphs (DAG)."""

    def __init__(self):
        self._id = native_graph_create()
        self._exec_id: Optional[int] = None

    @property
    def id(self) -> int:
        return self._id

    def add_kernel_node(
        self,
        kernel_id: int,
        name: str,
        grid_dim: Sequence[int],
        block_dim: Sequence[int],
        args: Sequence,
        arg_types: Optional[Sequence[int]] = None,
        deps: Optional[Sequence[int]] = None,
    ) -> int:
        packed, types, _keep_alive = _pack_args(args, arg_types)
        return native_graph_add_kernel_node(
            self._id, kernel_id, name, grid_dim, block_dim, packed, types, deps
        )

    def add_memcpy_node(
        self,
        dst: int,
        src: int,
        count: int,
        kind: int,
        deps: Optional[Sequence[int]] = None,
    ) -> int:
        return native_graph_add_memcpy_node(self._id, dst, src, count, kind, deps)

    def add_dependency(self, node_id: int, depends_on: int) -> None:
        native_graph_add_dependency(self._id, node_id, depends_on)

    def update_kernel_node(
        self, node_id: int, args: Sequence, arg_types: Optional[Sequence[int]] = None
    ) -> None:
        packed, types, _keep_alive = _pack_args(args, arg_types)
        native_graph_update_kernel_node(self._id, node_id, packed, types)

    def update_memcpy_node(
        self, node_id: int, dst: int, src: int, count: int, kind: int
    ) -> None:
        native_graph_update_memcpy_node(self._id, node_id, dst, src, count, kind)

    def instantiate(self) -> int:
        self._exec_id = native_graph_instantiate(self._id)
        return self._exec_id

    def launch(self, stream_id: int = 0) -> None:
        if self._exec_id is None:
            self.instantiate()
        native_graph_launch(self._exec_id, stream_id)

    def destroy_exec(self) -> None:
        if self._exec_id is not None:
            native_graph_exec_destroy(self._exec_id)
            self._exec_id = None

    def destroy(self) -> None:
        self.destroy_exec()
        native_graph_destroy(self._id)

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass


# Convenience re-exports
MEMCPY_HOST_TO_DEVICE = VGRE_MEMCPY_HOST_TO_DEVICE
MEMCPY_DEVICE_TO_HOST = VGRE_MEMCPY_DEVICE_TO_HOST
MEMCPY_DEVICE_TO_DEVICE = VGRE_MEMCPY_DEVICE_TO_DEVICE

ARG_POINTER = VGRE_ARG_POINTER
ARG_INT32 = VGRE_ARG_INT32
ARG_INT64 = VGRE_ARG_INT64
ARG_FLOAT32 = VGRE_ARG_FLOAT32
ARG_FLOAT64 = VGRE_ARG_FLOAT64
ARG_UINT32 = VGRE_ARG_UINT32
ARG_UINT64 = VGRE_ARG_UINT64
