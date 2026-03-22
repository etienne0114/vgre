"""
VGRE Virtual Device — Python wrapper for the virtual GPU device.

Provides device enumeration, property queries, and context management
via ctypes FFI to the libvgre shared library.
"""

import ctypes
import ctypes.util
import os
from typing import Optional, Dict, Any

try:
    from vgre._native import (  # type: ignore
        NATIVE_AVAILABLE,
        native_init,
        native_get_device_properties,
        native_synchronize,
        native_device_can_access_peer,
        native_device_enable_peer_access,
        native_device_disable_peer_access,
        native_get_telemetry,
    )
except (ImportError, ModuleNotFoundError):
    # This might fail during linting if PYTHONPATH is not set
    NATIVE_AVAILABLE = False

_NATIVE_REQUIRED_MSG = (
    "VGRE native backend is required. Pure Python device simulation is not supported."
)


def _ensure_native_init() -> None:
    if not NATIVE_AVAILABLE:
        raise RuntimeError(_NATIVE_REQUIRED_MSG)
    native_init()


class DeviceProperties:
    """Virtual GPU device properties."""

    def __init__(self):
        self.name: str = "VGRE Virtual GPU"
        self.total_global_mem: int = 4 * 1024 * 1024 * 1024  # 4 GB
        self.shared_mem_per_block: int = 48 * 1024
        self.max_threads_per_block: int = 1024
        self.max_threads_dim: list = [1024, 1024, 64]
        self.max_grid_size: list = [2147483647, 65535, 65535]
        self.warp_size: int = 32
        self.multi_processor_count: int = os.cpu_count() or 4
        self.major: int = 8
        self.minor: int = 6
        self.clock_rate: int = 1500000
        self.total_const_mem: int = 64 * 1024
        self.compute_capability: int = 86

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total_global_mem_mb": self.total_global_mem // (1024 * 1024),
            "shared_mem_per_block_kb": self.shared_mem_per_block // 1024,
            "max_threads_per_block": self.max_threads_per_block,
            "max_threads_dim": self.max_threads_dim,
            "max_grid_size": self.max_grid_size,
            "warp_size": self.warp_size,
            "multi_processor_count": self.multi_processor_count,
            "compute_capability": f"{self.major}.{self.minor}",
            "clock_rate_mhz": self.clock_rate // 1000,
        }

    def __repr__(self) -> str:
        d = self.to_dict()
        lines = [f"  {k}: {v}" for k, v in d.items()]
        return "DeviceProperties(\n" + "\n".join(lines) + "\n)"


class VirtualDevice:
    """
    Represents a VGRE virtual GPU device.

    Usage:
        device = VirtualDevice()
        props = device.get_properties()
        print(props)
    """

    _lib: Optional[ctypes.CDLL] = None
    _devices: list = []

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._properties = DeviceProperties()
        self._context_active = False
        self._detect_hardware()

    def _detect_hardware(self) -> None:
        """Detect actual CPU hardware to populate device properties."""
        _ensure_native_init()

        props = native_get_device_properties(self.device_id)
        self._properties.name = props.name.decode("utf-8", errors="replace")
        self._properties.total_global_mem = props.total_global_mem
        self._properties.shared_mem_per_block = props.shared_mem_per_block
        self._properties.max_threads_per_block = props.max_threads_per_block
        self._properties.max_threads_dim = list(props.max_threads_dim)
        self._properties.max_grid_size = list(props.max_grid_size)
        self._properties.warp_size = props.warp_size
        self._properties.multi_processor_count = props.multi_processor_count
        self._properties.major = props.major
        self._properties.minor = props.minor
        self._properties.clock_rate = props.clock_rate
        self._properties.total_const_mem = props.total_const_mem

    @staticmethod
    def get_device_count() -> int:
        """Return the number of virtual GPU devices."""
        _ensure_native_init()
        from vgre._native import native_get_device_count  # type: ignore
        return native_get_device_count()

    def get_properties(self) -> DeviceProperties:
        """Return the device properties."""
        return self._properties

    def create_context(self) -> None:
        """Create a device context."""
        if self._context_active:
            raise RuntimeError("Context already active")
        self._context_active = True

    def destroy_context(self) -> None:
        """Destroy the device context."""
        if not self._context_active:
            raise RuntimeError("No active context")
        self._context_active = False

    def has_context(self) -> bool:
        """Check if a context is active."""
        return self._context_active

    def synchronize(self) -> None:
        """Synchronize the device."""
        _ensure_native_init()
        native_synchronize()

    def can_access_peer(self, peer: int) -> bool:
        """Check if this device can access the peer's memory."""
        _ensure_native_init()
        return native_device_can_access_peer(self.device_id, peer)

    def enable_peer_access(self, peer: int) -> None:
        """Enable memory access to a peer device."""
        _ensure_native_init()
        native_device_enable_peer_access(peer)

    def disable_peer_access(self, peer: int) -> None:
        """Disable memory access to a peer device."""
        _ensure_native_init()
        native_device_disable_peer_access(peer)

    def get_telemetry(self) -> Dict[str, Any]:
        """Get live telemetry from the virtual device."""
        _ensure_native_init()
        t = native_get_telemetry()
        return {
            "timestamp": t.timestamp,
            "gflops": t.gflops,
            "max_gflops": t.max_gflops,
            "compute_utilization": t.compute_utilization,
            "memory_bandwidth_gbps": t.memory_bandwidth_gbps,
            "memory_bus_utilization": t.memory_bus_utilization,
            "memory_used_bytes": t.memory_used_bytes,
            "memory_total_bytes": t.memory_total_bytes,
            "active_kernels": t.active_kernels,
            "active_threads": t.active_threads,
            "device_temperature": t.device_temperature,
            "background_compute_active": bool(t.background_compute_active),
            "resident_pages": t.resident_pages,
            "page_faults_per_sec": t.page_faults_per_sec,
        }

    def __repr__(self) -> str:
        return (f"VirtualDevice(id={self.device_id}, "
                f"name='{self._properties.name}', "
                f"cores={self._properties.multi_processor_count})")


# ── Module-level convenience ────────────────────────────────────────────────
def get_device(device_id: int = 0) -> VirtualDevice:
    """Get a virtual device by ID."""
    return VirtualDevice(device_id)


def get_device_count() -> int:
    """Get the number of virtual devices."""
    return VirtualDevice.get_device_count()


if __name__ == "__main__":
    # Self-test
    dev = VirtualDevice()
    print(f"Device: {dev}")
    print(f"Properties:\n{dev.get_properties()}")
    print(f"Device count: {get_device_count()}")
