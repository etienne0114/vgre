"""
VGRE Native Bridge — ctypes FFI to libvgre.so

Loads the shared library and wraps all C API functions with proper
argtypes/restype declarations. Falls back gracefully when the library
is not found.
"""

import ctypes
import ctypes.util
import os
import sys
from pathlib import Path
from typing import Optional


# ── Status codes (must match vgre_c_api.h) ──────────────────────────────────
VGRE_SUCCESS = 0
VGRE_ERROR_NOT_INIT = -1
VGRE_ERROR_INVALID_VALUE = -2
VGRE_ERROR_OUT_OF_MEMORY = -3
VGRE_ERROR_INVALID_KERNEL = -4
VGRE_ERROR_LAUNCH_FAILURE = -5
VGRE_ERROR_IO = -6
VGRE_ERROR_GENERIC = -99

# Memcpy directions
VGRE_MEMCPY_HOST_TO_DEVICE = 0
VGRE_MEMCPY_DEVICE_TO_HOST = 1
VGRE_MEMCPY_DEVICE_TO_DEVICE = 2

# Kernel argument types
VGRE_ARG_POINTER = 0
VGRE_ARG_INT32 = 1
VGRE_ARG_INT64 = 2
VGRE_ARG_FLOAT32 = 3
VGRE_ARG_FLOAT64 = 4
VGRE_ARG_UINT32 = 5
VGRE_ARG_UINT64 = 6


# ── Device properties struct ────────────────────────────────────────────────
class VGREDeviceProperties(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 256),
        ("total_global_mem", ctypes.c_uint64),
        ("shared_mem_per_block", ctypes.c_uint64),
        ("max_threads_per_block", ctypes.c_int),
        ("max_threads_dim", ctypes.c_int * 3),
        ("max_grid_size", ctypes.c_int * 3),
        ("warp_size", ctypes.c_int),
        ("multi_processor_count", ctypes.c_int),
        ("major", ctypes.c_int),
        ("minor", ctypes.c_int),
        ("clock_rate", ctypes.c_int),
        ("total_const_mem", ctypes.c_uint64),
    ]


# ── Library loading ─────────────────────────────────────────────────────────
_lib: Optional[ctypes.CDLL] = None
NATIVE_AVAILABLE = False


def _find_library() -> Optional[str]:
    """Search for libvgre.so in common locations."""
    # 1. Environment variable
    env_path = os.environ.get("VGRE_LIB_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2. Relative to this file (project build dir)
    this_dir = Path(__file__).resolve().parent
    search_dirs = [
        this_dir / ".." / ".." / ".." / "build",         # bindings/python/vgre -> build
        this_dir / ".." / ".." / ".." / "build" / "lib",
        Path.cwd() / "build",
        Path.cwd(),
    ]
    for d in search_dirs:
        for name in ("libvgre.so", "libvgre.dylib", "vgre.dll"):
            candidate = d / name
            if candidate.is_file():
                return str(candidate)

    # 3. System search
    found = ctypes.util.find_library("vgre")
    if found:
        return found

    return None


def _load_library() -> Optional[ctypes.CDLL]:
    """Load libvgre and set up function signatures."""
    path = _find_library()
    if path is None:
        return None

    try:
        mode = getattr(ctypes, "RTLD_GLOBAL", None)
        if mode is None:
            lib = ctypes.CDLL(path)
        else:
            lib = ctypes.CDLL(path, mode=mode)
    except OSError:
        return None

    # ── Declare function signatures ─────────────────────────────────────────

    # Init / shutdown
    lib.vgre_init.argtypes = []
    lib.vgre_init.restype = ctypes.c_int

    lib.vgre_shutdown.argtypes = []
    lib.vgre_shutdown.restype = ctypes.c_int

    # Device management
    lib.vgre_get_device_count.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.vgre_get_device_count.restype = ctypes.c_int

    lib.vgre_set_device.argtypes = [ctypes.c_int]
    lib.vgre_set_device.restype = ctypes.c_int

    lib.vgre_get_device.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.vgre_get_device.restype = ctypes.c_int

    lib.vgre_get_device_properties.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(VGREDeviceProperties),
    ]
    lib.vgre_get_device_properties.restype = ctypes.c_int

    lib.vgre_synchronize.argtypes = []
    lib.vgre_synchronize.restype = ctypes.c_int

    # Memory management
    lib.vgre_malloc.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
    ]
    lib.vgre_malloc.restype = ctypes.c_int

    lib.vgre_free.argtypes = [ctypes.c_void_p]
    lib.vgre_free.restype = ctypes.c_int

    lib.vgre_memcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.vgre_memcpy.restype = ctypes.c_int

    lib.vgre_memset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    lib.vgre_memset.restype = ctypes.c_int

    # Kernel registration & launch
    lib.vgre_register_kernel.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.vgre_register_kernel.restype = ctypes.c_int

    lib.vgre_launch_kernel.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint32 * 3,
        ctypes.c_uint32 * 3,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_uint64,
    ]
    lib.vgre_launch_kernel.restype = ctypes.c_int

    # Stream management
    lib.vgre_stream_create.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
    lib.vgre_stream_create.restype = ctypes.c_int

    lib.vgre_stream_synchronize.argtypes = [ctypes.c_uint64]
    lib.vgre_stream_synchronize.restype = ctypes.c_int

    lib.vgre_stream_destroy.argtypes = [ctypes.c_uint64]
    lib.vgre_stream_destroy.restype = ctypes.c_int

    # Graphs (DAG)
    lib.vgre_graphCreate.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
    lib.vgre_graphCreate.restype = ctypes.c_int

    lib.vgre_graphDestroy.argtypes = [ctypes.c_uint64]
    lib.vgre_graphDestroy.restype = ctypes.c_int

    lib.vgre_graphInstantiate.argtypes = [
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.vgre_graphInstantiate.restype = ctypes.c_int

    lib.vgre_graphExecDestroy.argtypes = [ctypes.c_uint64]
    lib.vgre_graphExecDestroy.restype = ctypes.c_int

    lib.vgre_graphLaunch.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
    lib.vgre_graphLaunch.restype = ctypes.c_int

    lib.vgre_graphAddKernelNodeEx.argtypes = [
        ctypes.c_uint64,  # graph
        ctypes.c_uint64,  # kernel_id
        ctypes.c_char_p,  # name
        ctypes.c_uint32 * 3,  # grid
        ctypes.c_uint32 * 3,  # block
        ctypes.POINTER(ctypes.c_void_p),  # args
        ctypes.POINTER(ctypes.c_uint8),  # arg types
        ctypes.c_int,  # num args
        ctypes.POINTER(ctypes.c_uint64),  # deps
        ctypes.c_int,  # num deps
        ctypes.POINTER(ctypes.c_uint64),  # out node id
    ]
    lib.vgre_graphAddKernelNodeEx.restype = ctypes.c_int

    lib.vgre_graphAddMemcpyNodeEx.argtypes = [
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.vgre_graphAddMemcpyNodeEx.restype = ctypes.c_int

    lib.vgre_graphAddDependency.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
    ]
    lib.vgre_graphAddDependency.restype = ctypes.c_int

    lib.vgre_graphUpdateKernelNode.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
    ]
    lib.vgre_graphUpdateKernelNode.restype = ctypes.c_int

    lib.vgre_graphUpdateMemcpyNode.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.vgre_graphUpdateMemcpyNode.restype = ctypes.c_int

    # Version
    lib.vgre_get_version.argtypes = []
    lib.vgre_get_version.restype = ctypes.c_char_p

    return lib


# Load on import
_lib = _load_library()
NATIVE_AVAILABLE = _lib is not None


# ── Error handling ──────────────────────────────────────────────────────────

_STATUS_MESSAGES = {
    VGRE_SUCCESS: "Success",
    VGRE_ERROR_NOT_INIT: "Runtime not initialized",
    VGRE_ERROR_INVALID_VALUE: "Invalid value",
    VGRE_ERROR_OUT_OF_MEMORY: "Out of memory",
    VGRE_ERROR_INVALID_KERNEL: "Invalid kernel",
    VGRE_ERROR_LAUNCH_FAILURE: "Launch failure",
    VGRE_ERROR_IO: "I/O error",
    VGRE_ERROR_GENERIC: "Unknown error",
}


class VGREError(RuntimeError):
    """Exception raised when a VGRE C API call fails."""

    def __init__(self, status: int, context: str = ""):
        msg = _STATUS_MESSAGES.get(status, f"Error code {status}")
        if context:
            msg = f"{context}: {msg}"
        super().__init__(msg)
        self.status = status


def _check(status: int, context: str = "") -> None:
    """Raise VGREError if status is not SUCCESS."""
    if status != VGRE_SUCCESS:
        raise VGREError(status, context)


# ── Python-friendly wrappers ────────────────────────────────────────────────

def native_init() -> None:
    """Initialize the VGRE runtime engine."""
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT, "libvgre not loaded")
    _check(_lib.vgre_init(), "vgre_init")


def native_shutdown() -> None:
    """Shutdown the VGRE runtime engine."""
    if _lib is None:
        return
    _check(_lib.vgre_shutdown(), "vgre_shutdown")


def native_get_device_count() -> int:
    """Return the number of virtual GPU devices."""
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    count = ctypes.c_int(0)
    _check(_lib.vgre_get_device_count(ctypes.byref(count)),
           "vgre_get_device_count")
    return count.value


def native_set_device(device_id: int) -> None:
    """Set the active virtual GPU device."""
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    _check(_lib.vgre_set_device(device_id), "vgre_set_device")


def native_get_device_properties(device_id: int) -> VGREDeviceProperties:
    """Get properties of the specified device."""
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    props = VGREDeviceProperties()
    _check(_lib.vgre_get_device_properties(device_id, ctypes.byref(props)),
           "vgre_get_device_properties")
    return props


def native_synchronize() -> None:
    """Synchronize the device."""
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    _check(_lib.vgre_synchronize(), "vgre_synchronize")


def native_graph_create() -> int:
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    gid = ctypes.c_uint64(0)
    _check(_lib.vgre_graphCreate(ctypes.byref(gid)), "vgre_graphCreate")
    return int(gid.value)


def native_graph_destroy(graph_id: int) -> None:
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    _check(_lib.vgre_graphDestroy(ctypes.c_uint64(graph_id)),
           "vgre_graphDestroy")


def native_graph_instantiate(graph_id: int) -> int:
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    exec_id = ctypes.c_uint64(0)
    _check(_lib.vgre_graphInstantiate(ctypes.c_uint64(graph_id),
                                      ctypes.byref(exec_id)),
           "vgre_graphInstantiate")
    return int(exec_id.value)


def native_graph_exec_destroy(exec_id: int) -> None:
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    _check(_lib.vgre_graphExecDestroy(ctypes.c_uint64(exec_id)),
           "vgre_graphExecDestroy")


def native_graph_launch(exec_id: int, stream_id: int = 0) -> None:
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    _check(_lib.vgre_graphLaunch(ctypes.c_uint64(exec_id),
                                 ctypes.c_uint64(stream_id)),
           "vgre_graphLaunch")


def native_graph_add_kernel_node(graph_id: int, kernel_id: int, name: str,
                                 grid_dim, block_dim, args,
                                 arg_types, deps=None) -> int:
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    deps = deps or []
    grid = (ctypes.c_uint32 * 3)(*grid_dim)
    block = (ctypes.c_uint32 * 3)(*block_dim)
    args_arr = (ctypes.c_void_p * len(args))(*args)
    arg_types_arr = (ctypes.c_uint8 * len(arg_types))(*arg_types)
    deps_arr = (ctypes.c_uint64 * len(deps))(*deps) if deps else None
    out_id = ctypes.c_uint64(0)
    _check(
        _lib.vgre_graphAddKernelNodeEx(
            ctypes.c_uint64(graph_id),
            ctypes.c_uint64(kernel_id),
            name.encode("utf-8"),
            grid,
            block,
            args_arr,
            arg_types_arr,
            ctypes.c_int(len(arg_types)),
            deps_arr,
            ctypes.c_int(len(deps)),
            ctypes.byref(out_id),
        ),
        "vgre_graphAddKernelNodeEx",
    )
    return int(out_id.value)


def native_graph_add_memcpy_node(graph_id: int, dst, src, count, kind,
                                 deps=None) -> int:
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    deps = deps or []
    deps_arr = (ctypes.c_uint64 * len(deps))(*deps) if deps else None
    out_id = ctypes.c_uint64(0)
    _check(
        _lib.vgre_graphAddMemcpyNodeEx(
            ctypes.c_uint64(graph_id),
            ctypes.c_void_p(dst),
            ctypes.c_void_p(src),
            ctypes.c_size_t(count),
            ctypes.c_int(kind),
            deps_arr,
            ctypes.c_int(len(deps)),
            ctypes.byref(out_id),
        ),
        "vgre_graphAddMemcpyNodeEx",
    )
    return int(out_id.value)


def native_graph_add_dependency(graph_id: int, node_id: int,
                                depends_on: int) -> None:
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    _check(_lib.vgre_graphAddDependency(ctypes.c_uint64(graph_id),
                                        ctypes.c_uint64(node_id),
                                        ctypes.c_uint64(depends_on)),
           "vgre_graphAddDependency")


def native_graph_update_kernel_node(graph_id: int, node_id: int, args,
                                    arg_types) -> None:
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    args_arr = (ctypes.c_void_p * len(args))(*args)
    arg_types_arr = (ctypes.c_uint8 * len(arg_types))(*arg_types)
    _check(
        _lib.vgre_graphUpdateKernelNode(
            ctypes.c_uint64(graph_id),
            ctypes.c_uint64(node_id),
            args_arr,
            arg_types_arr,
            ctypes.c_int(len(arg_types)),
        ),
        "vgre_graphUpdateKernelNode",
    )


def native_graph_update_memcpy_node(graph_id: int, node_id: int, dst, src,
                                    count, kind) -> None:
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    _check(
        _lib.vgre_graphUpdateMemcpyNode(
            ctypes.c_uint64(graph_id),
            ctypes.c_uint64(node_id),
            ctypes.c_void_p(dst),
            ctypes.c_void_p(src),
            ctypes.c_size_t(count),
            ctypes.c_int(kind),
        ),
        "vgre_graphUpdateMemcpyNode",
    )


def native_malloc(size: int) -> ctypes.c_void_p:
    """Allocate device memory. Returns a void pointer."""
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    ptr = ctypes.c_void_p(0)
    _check(_lib.vgre_malloc(ctypes.byref(ptr), ctypes.c_size_t(size)),
           "vgre_malloc")
    return ptr


def native_free(ptr: ctypes.c_void_p) -> None:
    """Free device memory."""
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    _check(_lib.vgre_free(ptr), "vgre_free")


def native_memcpy(dst, src, count: int, direction: int) -> None:
    """Copy memory between host and device."""
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    _check(_lib.vgre_memcpy(dst, src, ctypes.c_size_t(count), direction),
           "vgre_memcpy")


def native_memset(ptr, value: int, count: int) -> None:
    """Set device memory to a value."""
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    _check(_lib.vgre_memset(ptr, value, ctypes.c_size_t(count)),
           "vgre_memset")


def native_register_kernel(name: str, source: str) -> int:
    """Register a kernel from source. Returns kernel ID."""
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    kid = ctypes.c_uint64(0)
    _check(_lib.vgre_register_kernel(
        name.encode("utf-8"),
        source.encode("utf-8"),
        ctypes.byref(kid)
    ), "vgre_register_kernel")
    return kid.value


def native_launch_kernel(kernel_id: int,
                         grid_dim: tuple,
                         block_dim: tuple,
                         args: list,
                         shared_mem: int = 0,
                         stream_id: int = 0) -> None:
    """Launch a registered kernel on the native engine."""
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)

    gd = (ctypes.c_uint32 * 3)(grid_dim[0], grid_dim[1], grid_dim[2])
    bd = (ctypes.c_uint32 * 3)(block_dim[0], block_dim[1], block_dim[2])

    num_args = len(args)
    arg_array = (ctypes.c_void_p * num_args)()
    
    import numpy as np
    # Must keep references alive so byref() pointers don't dangle
    _keep_alive = []
    
    for i, a in enumerate(args):
        if isinstance(a, np.ndarray):
            # The argument is a pointer to the array data
            val = ctypes.c_void_p(a.ctypes.data)
            _keep_alive.append(val)
            # We pass the address of that pointer
            arg_array[i] = ctypes.cast(ctypes.byref(val), ctypes.c_void_p)
        elif isinstance(a, int):
            # Pass 32-bit int by reference
            val = ctypes.c_int32(a)
            _keep_alive.append(val)
            arg_array[i] = ctypes.cast(ctypes.byref(val), ctypes.c_void_p)
        elif isinstance(a, float):
            # Pass 32-bit float by reference
            val = ctypes.c_float(a)
            _keep_alive.append(val)
            arg_array[i] = ctypes.cast(ctypes.byref(val), ctypes.c_void_p)
        else:
            val = ctypes.c_void_p(a)
            _keep_alive.append(val)
            arg_array[i] = ctypes.cast(ctypes.byref(val), ctypes.c_void_p)

    _check(_lib.vgre_launch_kernel(
        ctypes.c_uint64(kernel_id),
        gd, bd,
        arg_array, num_args,
        ctypes.c_size_t(shared_mem),
        ctypes.c_uint64(stream_id)
    ), "vgre_launch_kernel")


def native_stream_create() -> int:
    """Create a new stream. Returns stream ID."""
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    sid = ctypes.c_uint64(0)
    _check(_lib.vgre_stream_create(ctypes.byref(sid)), "vgre_stream_create")
    return sid.value


def native_stream_synchronize(stream_id: int) -> None:
    """Synchronize a stream."""
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    _check(_lib.vgre_stream_synchronize(ctypes.c_uint64(stream_id)),
           "vgre_stream_synchronize")


def native_stream_destroy(stream_id: int) -> None:
    """Destroy a stream."""
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    _check(_lib.vgre_stream_destroy(ctypes.c_uint64(stream_id)),
           "vgre_stream_destroy")


def native_get_version() -> str:
    """Get the VGRE library version string."""
    if _lib is None:
        return "0.0.0-python-only"
    return _lib.vgre_get_version().decode("utf-8")
