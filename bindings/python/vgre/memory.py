"""
VGRE Memory — Device memory management with numpy interop.

Provides DeviceArray for allocating, transferring, and managing memory
on the virtual GPU device, with seamless numpy conversion.
"""

import ctypes
import numpy as np
from typing import Optional

try:
    from vgre._native import (
        NATIVE_AVAILABLE,
        native_malloc, native_free, native_memcpy, native_memset,
        VGRE_MEMCPY_HOST_TO_DEVICE, VGRE_MEMCPY_DEVICE_TO_HOST,
        VGRE_MEMCPY_DEVICE_TO_DEVICE,
    )
except ImportError:
    NATIVE_AVAILABLE = False


class DeviceArray:
    """
    A device-side array that mirrors a numpy array.

    Manages GPU memory lifecycle and provides host ↔ device transfers.

    Usage:
        import numpy as np
        from vgre.memory import DeviceArray

        host_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        d_arr = DeviceArray.from_numpy(host_arr)
        result = d_arr.to_numpy()
        d_arr.free()
    """

    def __init__(self, size: int, dtype: np.dtype = np.float32):
        """
        Allocate a device array of the given size (in elements).

        Args:
            size: Number of elements.
            dtype: Numpy data type.
        """
        self.size = size
        self.dtype = np.dtype(dtype)
        self.nbytes = size * self.dtype.itemsize
        self._freed = False

        if not NATIVE_AVAILABLE:
            raise RuntimeError(
                "VGRE native execution engine is required. "
                "Pure Python simulations are no longer supported."
            )

        self._ptr = native_malloc(self.nbytes)
        self._native = True

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "DeviceArray":
        """
        Create a DeviceArray from a numpy array, copying data to device.

        Args:
            arr: Source numpy array.

        Returns:
            DeviceArray with a copy of the data on device.
        """
        d_arr = cls(arr.size, arr.dtype)
        d_arr.copy_from_host(arr)
        return d_arr

    @classmethod
    def zeros(cls, size: int, dtype: np.dtype = np.float32) -> "DeviceArray":
        """Create a zero-initialized device array."""
        d_arr = cls(size, dtype)
        if d_arr._native:
            native_memset(d_arr._ptr, 0, d_arr.nbytes)
        return d_arr

    @classmethod
    def empty(cls, size: int, dtype: np.dtype = np.float32) -> "DeviceArray":
        """Create an uninitialized device array."""
        return cls(size, dtype)

    def copy_from_host(self, host_arr: np.ndarray) -> None:
        """Copy data from a host numpy array to this device array."""
        if self._freed:
            raise RuntimeError("DeviceArray has been freed")

        src = host_arr.astype(self.dtype, copy=False)
        if src.nbytes > self.nbytes:
            raise ValueError(
                f"Source ({src.nbytes} bytes) exceeds device "
                f"allocation ({self.nbytes} bytes)"
            )

        if self._native:
            src_ptr = src.ctypes.data_as(ctypes.c_void_p)
            native_memcpy(self._ptr, src_ptr, src.nbytes,
                          VGRE_MEMCPY_HOST_TO_DEVICE)

    def copy_to_host(self, host_arr: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Copy data from this device array to a host numpy array.

        Args:
            host_arr: Optional destination array. If None, a new one is created.

        Returns:
            Numpy array with the device data.
        """
        if self._freed:
            raise RuntimeError("DeviceArray has been freed")

        if host_arr is None:
            host_arr = np.empty(self.size, dtype=self.dtype)

        if self._native:
            dst_ptr = host_arr.ctypes.data_as(ctypes.c_void_p)
            native_memcpy(dst_ptr, self._ptr,
                          min(host_arr.nbytes, self.nbytes),
                          VGRE_MEMCPY_DEVICE_TO_HOST)

        return host_arr

    def to_numpy(self) -> np.ndarray:
        """Convenience: copy device data to a new numpy array."""
        return self.copy_to_host()

    def copy_from_device(self, other: "DeviceArray") -> None:
        """Copy data from another DeviceArray to this one."""
        if self._freed or other._freed:
            raise RuntimeError("DeviceArray has been freed")

        count = min(self.nbytes, other.nbytes)
        if self._native and other._native:
            native_memcpy(self._ptr, other._ptr, count,
                          VGRE_MEMCPY_DEVICE_TO_DEVICE)

    def as_pointer(self) -> ctypes.c_void_p:
        """Get the raw device pointer for kernel arguments."""
        if self._freed:
            raise RuntimeError("DeviceArray has been freed")
        return self._ptr

    def free(self) -> None:
        """Free device memory."""
        if self._freed:
            return
        if self._native:
            native_free(self._ptr)
        self._freed = True

    @property
    def is_native(self) -> bool:
        """Whether this array uses the native C++ backend."""
        return self._native

    def __len__(self) -> int:
        return self.size

    def __del__(self):
        self.free()

    def __repr__(self) -> str:
        backend = "native"
        state = "freed" if self._freed else f"{self.size} x {self.dtype}"
        return f"DeviceArray({state}, backend={backend})"
