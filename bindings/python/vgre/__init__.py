"""
VGRE — Virtual GPU Runtime Engine
Python Bindings Package
"""

from vgre.device import VirtualDevice
from vgre.kernel import Kernel
from vgre.runtime import Runtime
from vgre.memory import DeviceArray
from vgre.stream import Stream

try:
    from vgre._native import NATIVE_AVAILABLE
except ImportError:
    NATIVE_AVAILABLE = False

__version__ = "0.1.0"
__all__ = [
    "VirtualDevice", "Kernel", "Runtime",
    "DeviceArray", "Stream",
    "NATIVE_AVAILABLE",
]
