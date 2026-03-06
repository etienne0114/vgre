import sys
import os
import time
import numpy as np

# Ensure we use libvgre from the build directory
# Normally this would be handled by LD_LIBRARY_PATH
vgre_path = os.path.abspath("../build")
sys.path.append(os.path.abspath("../bindings/python"))

import vgre

def run_external_workload():
    print("--- VGRE External Workload (IPC Test) ---")
    try:
        # Initialize the virtual GPU
        dev = vgre.VirtualDevice(0)
        runtime = vgre.Runtime()
        
        print(f"Connecting to VGRE Service on {dev.get_properties()['name']}")
        
        # Matrix size: 1024x1024
        N = 1024
        A = np.random.rand(N, N).astype(np.float32)
        B = np.random.rand(N, N).astype(np.float32)
        
        # Managed memory
        dA = vgre.DeviceArray.from_numpy(A)
        dB = vgre.DeviceArray.from_numpy(B)
        dC = vgre.DeviceArray(A.shape, A.dtype)
        
        print("Starting continuous workload loop...")
        print("Switch to the VGRE Dashboard to see aggregated GFLOPS!")
        
        loop_count = 0
        while True:
            # Launch matrix multiplication
            runtime.matMul(dA, dB, dC)
            runtime.synchronize()
            
            loop_count += 1
            if loop_count % 10 == 0:
                print(f"Completed {loop_count} iterations...")
            
            # small sleep to avoid 100% CPU starvation of the UI
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nWorkload stopped.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_external_workload()
