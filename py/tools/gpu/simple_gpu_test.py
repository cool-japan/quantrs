#!/usr/bin/env python3
"""
Simple GPU Support Test for QuantRS2

This script provides a basic test for GPU support in QuantRS2.
"""

import sys

def test_gpu_support():
    """Test if GPU support is working in QuantRS2."""
    try:
        # Try to import the module
        import _quantrs2 as qr
        
        print("✓ Successfully imported _quantrs2 module")
        
        # Create a simple circuit
        print("Creating test circuit...")
        circuit = qr.PyCircuit(2)
        
        # Add gates to create a Bell state
        circuit.h(0)
        circuit.cnot(0, 1)
        
        # First run with CPU
        print("Running simulation with CPU...")
        cpu_result = circuit.run(use_gpu=False)
        print(f"✓ CPU simulation completed")
        print(f"CPU state probabilities: {cpu_result.state_probabilities()}")
        
        # Check if GPU is available in this build
        try:
            # Try with GPU
            print("\nRunning simulation with GPU...")
            gpu_result = circuit.run(use_gpu=True)
            print(f"✓ GPU simulation completed")
            print(f"GPU state probabilities: {gpu_result.state_probabilities()}")
            print("\n✅ GPU support is working correctly!")
            return True
        except RuntimeError as e:
            if "GPU acceleration requested but not compiled in" in str(e):
                print("\n❌ GPU support was not enabled during compilation.")
                print("Please rebuild using: ./build_with_gpu.sh")
            elif "GPU acceleration requested but not available" in str(e):
                print("\n❌ GPU support was compiled in, but no compatible GPU was found.")
                print("This could be because you don't have a supported GPU or the drivers are not properly installed.")
            else:
                print(f"\n❌ Error during GPU simulation: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import _quantrs2 module: {e}")
        print("Make sure you've installed the package with:")
        print("maturin develop --release")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    print("QuantRS2 Simple GPU Support Test")
    print("===============================\n")
    success = test_gpu_support()
    sys.exit(0 if success else 1)