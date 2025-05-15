#!/usr/bin/env python3
"""
Minimal GPU check for QuantRS2
"""

import sys

def check_gpu_minimal():
    """Minimal check for GPU support"""
    try:
        # Try to import the module
        import _quantrs2 as qr
        print("✓ Successfully imported _quantrs2 module")
        
        # Create a simple circuit
        print("Creating a 2-qubit circuit...")
        circuit = qr.PyCircuit(2)
        circuit.h(0)
        circuit.cnot(0, 1)
        
        # First try with CPU
        print("Running on CPU...")
        cpu_result = circuit.run(use_gpu=False)
        print("CPU result:", cpu_result.state_probabilities())
        
        # Now try with GPU flag - this will show if GPU is compiled in
        # even if it falls back to CPU internally
        try:
            print("\nAttempting GPU simulation (will use CPU fallback if needed)...")
            gpu_result = circuit.run(use_gpu=True)
            print("GPU/fallback result:", gpu_result.state_probabilities())
            print("\n✅ GPU code path is working!")
            return True
        except Exception as e:
            print(f"\n❌ Error when requesting GPU: {e}")
            if "GPU acceleration requested but not compiled in" in str(e):
                print("The package was not compiled with GPU support.")
            return False
    except ImportError as e:
        print(f"❌ Could not import module: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("QuantRS2 Minimal GPU Check")
    print("=========================\n")
    success = check_gpu_minimal()
    sys.exit(0 if success else 1)
