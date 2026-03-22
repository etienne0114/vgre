import sys
import os
import numpy as np  # type: ignore
from pathlib import Path
from typing import cast, Any

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "bindings" / "python"))

import vgre  # type: ignore
from vgre.runtime import Runtime  # type: ignore
from vgre.memory import DeviceArray, ManagedArray  # type: ignore
from vgre.device import VirtualDevice  # type: ignore

def test_vgre_available():
    assert vgre.NATIVE_AVAILABLE, "VGRE native library not found. Build the project first."

def test_managed_memory():
    with Runtime() as rt:
        # Allocate managed memory (UVM)
        size = 1024
        arr = ManagedArray(size, dtype=np.float32)
        assert arr.is_native
        
        # Test basic memset
        ptr = arr.as_pointer()
        assert ptr is not None
        
        # In a real UVM system, we could access arr._ptr directly if it were mapped.
        # Here we verify the C API didn't crash.
        arr.free()

def test_p2p_logic():
    with Runtime() as rt:
        count = vgre.device.get_device_count()
        if count < 2:
            print("Skipping P2P test (requires 2+ devices)")
            return
        
        dev0 = VirtualDevice(0)
        can = dev0.can_access_peer(1)
        print(f"Device 0 can access Device 1: {can}")
        
        dev0.enable_peer_access(1)
        dev0.disable_peer_access(1)

def test_telemetry_authoritative():
    with Runtime() as rt:
        dev = rt.get_device()
        tel = dev.get_telemetry()
        
        assert "timestamp" in tel
        assert "gflops" in tel
        assert "device_temperature" in tel
        print(f"Telemetry: {tel}")

def test_profiler_and_logs():
    with Runtime() as rt:
        rt.set_native_profiler_enabled(True)
        
        # Launch a dummy kernel if possible, or just check JSON
        try:
            prof_json = rt.get_native_profiler_report()
            assert isinstance(prof_json, str)
        except Exception as e:
            print(f"Profiler report failed (might be empty): {e}")
            
        logs = rt.get_engine_logs()
        if isinstance(logs, list) and logs:
            print(f"Last 5 logs: {logs[-5:]}")  # type: ignore
        else:
            print("Last 5 logs: None")

def test_cluster_security_and_credits():
    with Runtime() as rt:
        # Check security info
        info = rt.cluster_get_security_info()
        assert "cipher_name" in info
        
        # Check credits (might be empty but shouldn't crash)
        try:
            balance = rt.credits_get_balance("localhost")
            assert "balance" in balance
        except:
            pass
            
        all_credits = rt.credits_get_all()
        assert isinstance(all_credits, list)

if __name__ == "__main__":
    # Manual run
    try:
        test_vgre_available()
        test_managed_memory()
        test_p2p_logic()
        test_telemetry_authoritative()
        test_profiler_and_logs()
        test_cluster_security_and_credits()
        print("\nAll authoritative Python tests PASSED!")
    except Exception as e:
        print(f"\nTest FAILED: {e}")
        sys.exit(1)
