"""
VGRE Stream — Asynchronous stream management.

Provides a Python Stream class that wraps the native VGRE stream API
for ordering and synchronizing kernel launches.
"""

from typing import Optional

try:
    from vgre._native import (
        NATIVE_AVAILABLE,
        native_stream_create,
        native_stream_synchronize,
        native_stream_destroy,
    )
except ImportError:
    NATIVE_AVAILABLE = False


class Stream:
    """
    Represents a VGRE execution stream for ordering GPU operations.

    Usage:
        from vgre.stream import Stream

        stream = Stream()
        # Launch kernels on this stream ...
        stream.synchronize()
        stream.destroy()

    Context manager usage:
        with Stream() as s:
            # Launch kernels on s ...
            pass  # auto-synchronized and destroyed
    """

    # Default stream ID (stream 0)
    DEFAULT_STREAM_ID = 0

    def __init__(self, create: bool = True):
        """
        Create a new stream.

        Args:
            create: If True, immediately allocates a native stream.
                    If False, represents the default stream (ID 0).
        """
        self._destroyed = False

        if create and NATIVE_AVAILABLE:
            self._stream_id = native_stream_create()
            self._native = True
        else:
            self._stream_id = Stream.DEFAULT_STREAM_ID
            self._native = NATIVE_AVAILABLE

    @classmethod
    def default(cls) -> "Stream":
        """Get the default stream (stream 0)."""
        s = cls(create=False)
        return s

    @property
    def stream_id(self) -> int:
        """The native stream ID."""
        return self._stream_id

    @property
    def is_native(self) -> bool:
        """Whether this stream uses the native C++ backend."""
        return self._native

    def synchronize(self) -> None:
        """Block until all operations on this stream are complete."""
        if self._destroyed:
            raise RuntimeError("Stream has been destroyed")

        if self._native and self._stream_id != Stream.DEFAULT_STREAM_ID:
            native_stream_synchronize(self._stream_id)

    def destroy(self) -> None:
        """Destroy the stream and release resources."""
        if self._destroyed:
            return
        if self._native and self._stream_id != Stream.DEFAULT_STREAM_ID:
            native_stream_destroy(self._stream_id)
        self._destroyed = True

    def __enter__(self) -> "Stream":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.synchronize()
        self.destroy()
        return False

    def __del__(self):
        self.destroy()

    def __repr__(self) -> str:
        if self._destroyed:
            return "Stream(destroyed)"
        backend = "native" if self._native else "python"
        return f"Stream(id={self._stream_id}, backend={backend})"
