"""
VGRE Runtime — High-level Python runtime API.

Provides init/launch/sync/profiling for the VGRE virtual GPU system.
"""

import numpy as np
import time
import json
import os
from typing import Optional, Dict, List, Any

from vgre.device import VirtualDevice, DeviceProperties
from vgre.kernel import Kernel, Dim3
from vgre.graph import Graph

try:
    from vgre._native import (
        NATIVE_AVAILABLE,
        native_init,
        native_shutdown,
        native_synchronize,
    )
except ImportError:
    NATIVE_AVAILABLE = False


class ProfilingReport:
    """Stores profiling data for executed kernels."""

    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def add_event(self, kernel_name: str, duration_ms: float,
                  grid_dim: Dim3, block_dim: Dim3,
                  memory_bytes: int = 0, flops: int = 0) -> None:
        throughput_gbps = 0.0
        gflops = 0.0
        if duration_ms > 0:
            if memory_bytes > 0:
                throughput_gbps = (memory_bytes / 1e9) / (duration_ms / 1000.0)
            if flops > 0:
                gflops = (flops / 1e9) / (duration_ms / 1000.0)

        self.events.append({
            "kernel_name": kernel_name,
            "duration_ms": duration_ms,
            "grid_dim": f"{grid_dim.x}x{grid_dim.y}x{grid_dim.z}",
            "block_dim": f"{block_dim.x}x{block_dim.y}x{block_dim.z}",
            "throughput_gbps": throughput_gbps,
            "gflops": gflops,
        })

    def to_json(self) -> str:
        return json.dumps({
            "profiler": "VGRE Python Runtime Profiler",
            "total_events": len(self.events),
            "events": self.events,
        }, indent=2)

    def export_to_file(self, filepath: str) -> None:
        with open(filepath, "w") as f:
            f.write(self.to_json())

    def summary(self) -> str:
        if not self.events:
            return "No profiling data recorded."

        lines = ["VGRE Profiling Summary",
                 "=" * 60]

        # Aggregate by kernel name
        kernel_stats: Dict[str, Dict[str, Any]] = {}
        for ev in self.events:
            name = ev["kernel_name"]
            if name not in kernel_stats:
                kernel_stats[name] = {
                    "count": 0, "total_ms": 0.0,
                    "min_ms": float("inf"), "max_ms": 0.0
                }
            s = kernel_stats[name]
            s["count"] += 1
            s["total_ms"] += ev["duration_ms"]
            s["min_ms"] = min(s["min_ms"], ev["duration_ms"])
            s["max_ms"] = max(s["max_ms"], ev["duration_ms"])

        for name, s in kernel_stats.items():
            avg = s["total_ms"] / s["count"]
            lines.append(
                f"  {name}: {s['count']} calls, "
                f"avg={avg:.3f}ms, "
                f"min={s['min_ms']:.3f}ms, "
                f"max={s['max_ms']:.3f}ms"
            )

        return "\n".join(lines)


class Runtime:
    """
    High-level VGRE runtime for Python.

    Usage:
        rt = Runtime()
        rt.init()

        a = np.random.randn(1024).astype(np.float32)
        b = np.random.randn(1024).astype(np.float32)
        c = np.zeros(1024, dtype=np.float32)

        from vgre.kernel import vector_add_kernel, Dim3
        kernel = vector_add_kernel()
        rt.launch(kernel, Dim3(4), Dim3(256), [a, b, c])

        print(rt.get_profiling_report().summary())
        rt.shutdown()
    """

    def __init__(self):
        self._device: Optional[VirtualDevice] = None
        self._initialized = False
        self._use_native = False
        self._profiling_enabled = False
        self._profile = ProfilingReport()
        self._kernels: Dict[str, Kernel] = {}

    def init(self, device_id: int = 0,
             enable_profiling: bool = False) -> None:
        """Initialize the runtime."""
        if self._initialized:
            return

        if not NATIVE_AVAILABLE:
            raise RuntimeError(
                "VGRE native execution engine is required. "
                "Pure Python simulations are no longer supported."
            )

        native_init()
        self._use_native = True

        self._device = VirtualDevice(device_id)
        self._device.create_context()
        self._profiling_enabled = enable_profiling
        self._initialized = True

        props = self._device.get_properties()
        print(f"[VGRE] Initialized virtual GPU: {props.name} (native)")
        print(f"[VGRE] Cores: {props.multi_processor_count}, "
              f"VRAM: {props.total_global_mem // (1024*1024)} MB")

    def shutdown(self) -> None:
        """Shutdown the runtime."""
        if not self._initialized:
            return
        if self._device and self._device.has_context():
            self._device.destroy_context()
        if self._use_native:
            try:
                native_shutdown()
            except Exception:
                pass
        self._initialized = False
        self._use_native = False
        print("[VGRE] Runtime shut down")

    def is_initialized(self) -> bool:
        return self._initialized

    def get_device(self) -> VirtualDevice:
        """Get the current virtual device."""
        if not self._initialized:
            raise RuntimeError("Runtime not initialized")
        return self._device

    def create_graph(self) -> Graph:
        """Create a native CUDA graph (DAG) wrapper."""
        if not self._initialized:
            raise RuntimeError("Runtime not initialized")
        return Graph()

    def launch(self, kernel: Kernel, grid_dim: Dim3, block_dim: Dim3,
               args: list, shared_mem: int = 0,
               parallel: bool = True) -> float:
        """
        Launch a kernel on the virtual GPU.

        Returns execution time in milliseconds.
        """
        if not self._initialized:
            raise RuntimeError("Runtime not initialized")

        start = time.perf_counter()
        
        if not self._use_native:
            raise RuntimeError("VGRE native execution engine is required. Pure Python simulations are no longer supported.")

        from vgre._native import native_register_kernel, native_launch_kernel
        if kernel._kernel_id is None:
            kernel._kernel_id = native_register_kernel(kernel.name, getattr(kernel, "source", ""))
        
        native_launch_kernel(kernel._kernel_id, 
                             (grid_dim.x, grid_dim.y, grid_dim.z), 
                             (block_dim.x, block_dim.y, block_dim.z), 
                             args, shared_mem)
        if not parallel:
            # Caller requested serialized semantics for this launch.
            native_synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        kernel._launch_count += 1
        kernel._total_time_ms += elapsed_ms

        # Record profiling
        if self._profiling_enabled:
            # Estimate memory accessed from numpy arrays
            mem_bytes = sum(a.nbytes for a in args
                          if isinstance(a, np.ndarray))
            self._profile.add_event(
                kernel.name, elapsed_ms, grid_dim, block_dim,
                memory_bytes=mem_bytes
            )

        return elapsed_ms

    def synchronize(self) -> None:
        """Synchronize the device."""
        if self._use_native:
            native_synchronize()
        elif self._device:
            self._device.synchronize()

    def enable_profiling(self) -> None:
        self._profiling_enabled = True

    def disable_profiling(self) -> None:
        self._profiling_enabled = False

    def get_profiling_report(self) -> ProfilingReport:
        return self._profile

    def clear_profiling(self) -> None:
        self._profile = ProfilingReport()

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        return f"Runtime({status})"


# ── Module-level convenience ────────────────────────────────────────────────
_global_runtime: Optional[Runtime] = None


def init(device_id: int = 0, enable_profiling: bool = False) -> Runtime:
    """Initialize the global VGRE runtime."""
    global _global_runtime
    if _global_runtime is None:
        _global_runtime = Runtime()
    _global_runtime.init(device_id, enable_profiling)
    return _global_runtime


def get_runtime() -> Runtime:
    """Get the global runtime instance."""
    global _global_runtime
    if _global_runtime is None:
        _global_runtime = Runtime()
    return _global_runtime


def shutdown() -> None:
    """Shutdown the global runtime."""
    global _global_runtime
    if _global_runtime:
        _global_runtime.shutdown()
