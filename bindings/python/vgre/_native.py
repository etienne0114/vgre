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
from typing import Optional, Any, List, Type, cast


# ── Status codes (must match vgre_c_api.h) ──────────────────────────────────
VGRE_SUCCESS = 0
VGRE_ERROR_NOT_INIT = -1
VGRE_ERROR_INVALID_VALUE = -2
VGRE_ERROR_OUT_OF_MEMORY = -3
VGRE_ERROR_INVALID_KERNEL = -4
VGRE_ERROR_LAUNCH_FAILURE = -5
VGRE_ERROR_IO = -6
VGRE_ERROR_AUTH_FAILED = -7
VGRE_ERROR_CRYPTO = -8
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
VGRE_ARG_STRUCT = 7

# ── Type Aliases (for linter compatibility) ────────────────────────────────
Char256 = cast(Any, ctypes.c_char * 256)
Char64 = cast(Any, ctypes.c_char * 64)
Char65 = cast(Any, ctypes.c_char * 65)
Int3 = cast(Any, ctypes.c_int * 3)
Uint3 = cast(Any, ctypes.c_uint32 * 3)
Uint8_1024 = cast(Any, ctypes.c_uint8 * 1024)


# ── Device properties struct ────────────────────────────────────────────────
class VGREDeviceProperties(ctypes.Structure):
    _fields_ = [
        ("name", Char256),
        ("total_global_mem", ctypes.c_uint64),
        ("shared_mem_per_block", ctypes.c_uint64),
        ("max_threads_per_block", ctypes.c_int),
        ("max_threads_dim", Int3),
        ("max_grid_size", Int3),
        ("warp_size", ctypes.c_int),
        ("multi_processor_count", ctypes.c_int),
        ("major", ctypes.c_int),
        ("minor", ctypes.c_int),
        ("clock_rate", ctypes.c_int),
        ("total_const_mem", ctypes.c_uint64),
    ]


class VGREClusterNode(ctypes.Structure):
    _fields_ = [
        ("address", Char64),
        ("port", ctypes.c_int),
        ("cpu_cores", ctypes.c_int),
        ("memory_bytes", ctypes.c_uint64),
        ("latency_ms", ctypes.c_double),
        ("available", ctypes.c_int),
        ("igpu_name", Char64),
    ]


class VGRETelemetry(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("timestamp", ctypes.c_uint64),
        ("version_major", ctypes.c_uint64),
        ("version_minor", ctypes.c_uint64),
        ("gflops", ctypes.c_double),
        ("max_gflops", ctypes.c_double),
        ("compute_utilization", ctypes.c_double),
        ("memory_bandwidth_gbps", ctypes.c_double),
        ("max_memory_bandwidth_gbps", ctypes.c_double),
        ("memory_bus_utilization", ctypes.c_double),
        ("memory_used_bytes", ctypes.c_uint64),
        ("memory_total_bytes", ctypes.c_uint64),
        ("total_pages", ctypes.c_uint64),
        ("resident_pages", ctypes.c_uint64),
        ("evicted_pages", ctypes.c_uint64),
        ("page_faults_per_sec", ctypes.c_double),
        ("active_kernels", ctypes.c_int64),
        ("active_threads", ctypes.c_int64),
        ("device_clock_mhz", ctypes.c_double),
        ("avg_kernel_latency_ms", ctypes.c_double),
        ("device_temperature", ctypes.c_double),
        ("ecc_enabled", ctypes.c_int64),
        ("background_compute_active", ctypes.c_int64),
        ("uvm_map", Uint8_1024),
    ]


class VGRESecurityInfo(ctypes.Structure):
    _fields_ = [
        ("cipher_name", Char64),
        ("key_fingerprint", Char65),
        ("session_seconds", ctypes.c_double),
        ("is_encrypted", ctypes.c_int),
        ("packets_sent", ctypes.c_uint64),
        ("packets_received", ctypes.c_uint64),
        ("bytes_sent", ctypes.c_uint64),
        ("bytes_received", ctypes.c_uint64),
    ]


class VGRECreditInfo(ctypes.Structure):
    _fields_ = [
        ("address", Char64),
        ("total_credits", ctypes.c_double),
        ("total_debits", ctypes.c_double),
        ("balance", ctypes.c_double),
        ("last_activity", ctypes.c_uint64),
        ("transaction_count", ctypes.c_int),
    ]


# ── Library loading ─────────────────────────────────────────────────────────
_lib: Optional[ctypes.CDLL] = None
NATIVE_AVAILABLE = False


def _find_library() -> Optional[str]:
    """Search for libvgre.so in common locations."""
    # 1. Environment variable override
    env_path = os.environ.get("VGRE_LIB_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2. VGRE_INSTALL_DIR environment variable
    vgre_install_dir = os.environ.get("VGRE_INSTALL_DIR")
    if vgre_install_dir:
        install_path = Path(vgre_install_dir)
        for name in ("libvgre.so", "libvgre.dylib", "vgre.dll"):
            for sub in ("", "lib"):
                candidate = install_path / sub / name
                if candidate.is_file():
                    return str(candidate)

    # 3. Windows LOCALAPPDATA default installation path
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            install_path = Path(local_app_data) / "VGRE"
            for name in ("vgre.dll",):
                for sub in ("", "lib"):
                    candidate = install_path / sub / name
                    if candidate.is_file():
                        return str(candidate)

    # 4. Relative to this file (bundled-in-wheel, then project build dir)
    this_dir = Path(__file__).resolve().parent
    search_dirs = [
        this_dir / "lib",                                 # libs bundled inside the wheel
        this_dir,
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

    # 5. System search
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

    lib.vgre_free.restype = ctypes.c_int

    lib.vgre_malloc_managed.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
    ]
    lib.vgre_malloc_managed.restype = ctypes.c_int

    # P2P
    lib.vgre_device_can_access_peer.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.vgre_device_can_access_peer.restype = ctypes.c_int

    lib.vgre_device_enable_peer_access.argtypes = [ctypes.c_int]
    lib.vgre_device_enable_peer_access.restype = ctypes.c_int

    lib.vgre_device_disable_peer_access.argtypes = [ctypes.c_int]
    lib.vgre_device_disable_peer_access.restype = ctypes.c_int

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

    lib.vgre_launch_kernel.argtypes = cast(Any, [
        ctypes.c_uint64,
        Uint3,
        Uint3,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_uint64,
    ])
    lib.vgre_launch_kernel.restype = ctypes.c_int

    # Stream management
    lib.vgre_stream_create.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
    lib.vgre_stream_create.restype = ctypes.c_int

    lib.vgre_stream_synchronize.argtypes = [ctypes.c_uint64]
    lib.vgre_stream_synchronize.restype = ctypes.c_int

    lib.vgre_stream_destroy.restype = ctypes.c_int

    lib.vgre_stream_create_with_priority.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_int,
    ]
    lib.vgre_stream_create_with_priority.restype = ctypes.c_int

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

    lib.vgre_graphExecDestroy.argtypes = cast(Any, [ctypes.c_uint64])
    lib.vgre_graphExecDestroy.restype = ctypes.c_int

    lib.vgre_graphLaunch.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
    lib.vgre_graphLaunch.restype = ctypes.c_int

    lib.vgre_graphAddKernelNodeEx.argtypes = cast(Any, [
        ctypes.c_uint64,  # graph
        ctypes.c_uint64,  # kernel_id
        ctypes.c_char_p,  # name
        Uint3,            # grid
        Uint3,            # block
        ctypes.POINTER(ctypes.c_void_p),  # args
        ctypes.POINTER(ctypes.c_uint8),  # arg types
        ctypes.c_int,  # num args
        ctypes.POINTER(ctypes.c_uint64),  # deps
        ctypes.c_int,  # num deps
        ctypes.POINTER(ctypes.c_uint64),  # out node id
    ])
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
    lib.vgre_get_version.restype = ctypes.c_char_p

    # Telemetry & Info
    lib.vgre_get_telemetry.argtypes = [ctypes.POINTER(VGRETelemetry)]
    lib.vgre_get_telemetry.restype = ctypes.c_int

    lib.vgre_get_profiler_json.argtypes = [
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_int,
    ]
    lib.vgre_get_profiler_json.restype = ctypes.c_int

    lib.vgre_free_string.argtypes = [ctypes.c_char_p]
    lib.vgre_free_string.restype = None

    lib.vgre_set_profiler_enabled.argtypes = [ctypes.c_int]
    lib.vgre_set_profiler_enabled.restype = ctypes.c_int

    lib.vgre_get_memory_info_json.argtypes = [ctypes.POINTER(ctypes.c_char_p)]
    lib.vgre_get_memory_info_json.restype = ctypes.c_int

    lib.vgre_get_kernel_history_json.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    lib.vgre_get_kernel_history_json.restype = ctypes.c_int

    lib.vgre_get_logs.argtypes = [
        ctypes.POINTER(ctypes.POINTER(ctypes.c_char_p)),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.vgre_get_logs.restype = ctypes.c_int

    lib.vgre_free_logs.argtypes = [
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_int,
    ]
    lib.vgre_free_logs.restype = None

    lib.vgre_get_cluster_nodes.argtypes = [
        ctypes.POINTER(VGREClusterNode),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.vgre_get_cluster_nodes.restype = ctypes.c_int

    # Control
    lib.vgre_set_background_compute.argtypes = [ctypes.c_int]
    lib.vgre_set_background_compute.restype = ctypes.c_int

    lib.vgre_set_service_mode.argtypes = [ctypes.c_int]
    lib.vgre_set_service_mode.restype = ctypes.c_int

    lib.vgre_set_block_threads.argtypes = [ctypes.c_int]
    lib.vgre_set_block_threads.restype = ctypes.c_int

    # Phase 5: Global Compute Network
    lib.vgre_cluster_set_security.argtypes = [ctypes.c_int]
    lib.vgre_cluster_set_security.restype = ctypes.c_int

    lib.vgre_cluster_get_security_info.argtypes = [ctypes.POINTER(VGRESecurityInfo)]
    lib.vgre_cluster_get_security_info.restype = ctypes.c_int

    lib.vgre_credits_get_balance.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(VGRECreditInfo),
    ]
    lib.vgre_credits_get_balance.restype = ctypes.c_int

    lib.vgre_credits_get_all.argtypes = [
        ctypes.POINTER(VGRECreditInfo),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.vgre_credits_get_all.restype = ctypes.c_int

    lib.vgre_credits_reset.argtypes = []
    lib.vgre_credits_reset.restype = ctypes.c_int

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
    VGRE_ERROR_AUTH_FAILED: "Authentication failed",
    VGRE_ERROR_CRYPTO: "Cryptographic error",
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
    grid = Uint3(*grid_dim)
    block = Uint3(*block_dim)
    args_arr = cast(Any, ctypes.c_void_p * len(args))(*args)
    arg_types_arr = cast(Any, ctypes.c_uint8 * len(arg_types))(*arg_types)
    deps_arr = cast(Any, ctypes.c_uint64 * len(deps))(*deps) if deps else None
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
    deps_arr = cast(Any, ctypes.c_uint64 * len(deps))(*deps) if deps else None
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
    args_arr = cast(Any, ctypes.c_void_p * len(args))(*args)
    arg_types_arr = cast(Any, ctypes.c_uint8 * len(arg_types))(*arg_types)
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


def native_malloc_managed(size: int) -> ctypes.c_void_p:
    """Allocate unified managed memory."""
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    ptr = ctypes.c_void_p(0)
    _check(_lib.vgre_malloc_managed(ctypes.byref(ptr), ctypes.c_size_t(size)),
           "vgre_malloc_managed")
    return ptr


def native_device_can_access_peer(device: int, peer: int) -> bool:
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    can = ctypes.c_int(0)
    _check(_lib.vgre_device_can_access_peer(ctypes.byref(can), device, peer),
           "vgre_device_can_access_peer")
    return bool(can.value)


def native_device_enable_peer_access(peer: int) -> None:
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    _check(_lib.vgre_device_enable_peer_access(peer),
           "vgre_device_enable_peer_access")


def native_device_disable_peer_access(peer: int) -> None:
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    _check(_lib.vgre_device_disable_peer_access(peer),
           "vgre_device_disable_peer_access")


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

    import numpy as np  # type: ignore

    gd = Uint3(grid_dim[0], grid_dim[1], grid_dim[2])
    bd = Uint3(block_dim[0], block_dim[1], block_dim[2])

    num_args = len(args)
    arg_array = cast(Any, ctypes.c_void_p * num_args)()
    
    # Must keep references alive so byref() pointers don't dangle
    _keep_alive: List[Any] = []
    
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


def native_stream_create_with_priority(priority: int) -> int:
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    sid = ctypes.c_uint64(0)
    _check(_lib.vgre_stream_create_with_priority(ctypes.byref(sid), priority),
           "vgre_stream_create_with_priority")
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
    res = _lib.vgre_get_version()
    return res.decode("utf-8") if res else "0.0.0"


def native_get_telemetry() -> VGRETelemetry:
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    tel = VGRETelemetry()
    _check(_lib.vgre_get_telemetry(ctypes.byref(tel)), "vgre_get_telemetry")
    return tel


def native_get_profiler_json(top_n: int = 0) -> str:
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    ptr = ctypes.c_char_p()
    _check(_lib.vgre_get_profiler_json(ctypes.byref(ptr), top_n),
           "vgre_get_profiler_json")
    val = ptr.value
    if val is None:
        return "{}"
    res = val.decode("utf-8")
    _lib.vgre_free_string(ptr)
    return res


def native_set_profiler_enabled(enabled: bool) -> None:
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    _check(_lib.vgre_set_profiler_enabled(1 if enabled else 0),
           "vgre_set_profiler_enabled")


def native_get_memory_info_json() -> str:
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    ptr = ctypes.c_char_p()
    _check(_lib.vgre_get_memory_info_json(ctypes.byref(ptr)),
           "vgre_get_memory_info_json")
    val = ptr.value
    if val is None:
        return "{}"
    res = val.decode("utf-8")
    _lib.vgre_free_string(ptr)
    return res


def native_get_kernel_history_json(kernel_name: str) -> str:
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    ptr = ctypes.c_char_p()
    _check(_lib.vgre_get_kernel_history_json(kernel_name.encode("utf-8"),
                                             ctypes.byref(ptr)),
           "vgre_get_kernel_history_json")
    val = ptr.value
    if val is None:
        return "[]"
    res = val.decode("utf-8")
    _lib.vgre_free_string(ptr)
    return res


def native_get_logs() -> list[str]:
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    buffer = ctypes.POINTER(ctypes.c_char_p)()
    count = ctypes.c_int(0)
    _check(_lib.vgre_get_logs(ctypes.byref(buffer), ctypes.byref(count)),
           "vgre_get_logs")
    logs = []
    for i in range(count.value):
        entry = buffer[i]
        if entry:
            logs.append(entry.decode("utf-8"))
    _lib.vgre_free_logs(buffer, count)
    return logs


def native_get_cluster_nodes() -> list[VGREClusterNode]:
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    count = ctypes.c_int(32)  # Initial capacity
    node_type = cast(Any, VGREClusterNode * count.value)  # type: ignore
    nodes = node_type()
    _check(_lib.vgre_get_cluster_nodes(nodes, ctypes.byref(count)),
           "vgre_get_cluster_nodes")
    return [nodes[i] for i in range(count.value)]


def native_set_background_compute(enabled: bool) -> None:
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    _check(_lib.vgre_set_background_compute(1 if enabled else 0),
           "vgre_set_background_compute")


def native_set_service_mode(is_master: bool) -> None:
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    _check(_lib.vgre_set_service_mode(1 if is_master else 0),
           "vgre_set_service_mode")


def native_set_block_threads(enabled: bool) -> None:
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    _check(_lib.vgre_set_block_threads(1 if enabled else 0),
           "vgre_set_block_threads")


def native_cluster_set_security(enabled: bool) -> None:
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    _check(_lib.vgre_cluster_set_security(1 if enabled else 0),
           "vgre_cluster_set_security")


def native_cluster_get_security_info() -> VGRESecurityInfo:
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    info = VGRESecurityInfo()
    _check(_lib.vgre_cluster_get_security_info(ctypes.byref(info)),
           "vgre_cluster_get_security_info")
    return info


def native_credits_get_balance(address: str) -> VGRECreditInfo:
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    info = VGRECreditInfo()
    _check(_lib.vgre_credits_get_balance(address.encode("utf-8"),
                                         ctypes.byref(info)),
           "vgre_credits_get_balance")
    return info


def native_credits_get_all() -> list[VGRECreditInfo]:
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    count = ctypes.c_int(64)
    credit_type = cast(Any, VGRECreditInfo * count.value)  # type: ignore
    nodes = credit_type()
    _check(_lib.vgre_credits_get_all(nodes, ctypes.byref(count)),
           "vgre_credits_get_all")
    return [nodes[i] for i in range(count.value)]


def native_credits_reset() -> None:
    if _lib is None:
        raise VGREError(VGRE_ERROR_NOT_INIT)
    _check(_lib.vgre_credits_reset(), "vgre_credits_reset")
