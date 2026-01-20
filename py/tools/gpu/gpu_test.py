#!/usr/bin/env python3
"""
GPU Support Test for QuantRS2

This script tests if the GPU support in QuantRS2 is properly enabled and working.
"""

import os
import sys
import time


def test_gpu_support():
    """Test if GPU support is working in QuantRS2."""
    try:
        # Try to import the module
        import _quantrs2 as qr
        
        print("✓ Successfully imported _quantrs2 module")
        
        # Create a simple circuit
        print("Creating test circuit...")
        circuit = qr.PyCircuit(10)  # Use 10 qubits to make GPU advantageous
        
        # Add some gates to make the simulation non-trivial
        for i in range(10):
            circuit.h(i)
        
        for i in range(9):
            circuit.cnot(i, i+1)
        
        # First run with CPU
        print("Running simulation with CPU...")
        start_time = time.time()
        cpu_result = circuit.run(use_gpu=False)
        cpu_time = time.time() - start_time
        print(f"✓ CPU simulation completed in {cpu_time:.4f} seconds")
        
        # Then try with GPU
        try:
            print("Running simulation with GPU...")
            start_time = time.time()
            gpu_result = circuit.run(use_gpu=True)
            gpu_time = time.time() - start_time
            print(f"✓ GPU simulation completed in {gpu_time:.4f} seconds")
            
            # Compare results
            print(f"Speedup: {cpu_time/gpu_time:.2f}x")
            
            # Verify the results are consistent
            cpu_probs = cpu_result.state_probabilities()
            gpu_probs = gpu_result.state_probabilities()
            
            # Check if the top states are the same
            cpu_top = sorted([(i, p) for i, p in enumerate(cpu_probs) if p > 0.01], 
                            key=lambda x: x[1], reverse=True)[:5]
            gpu_top = sorted([(i, p) for i, p in enumerate(gpu_probs) if p > 0.01], 
                            key=lambda x: x[1], reverse=True)[:5]
            
            print("\nTop states from CPU simulation:")
            for state, prob in cpu_top:
                print(f"|{state:010b}⟩: {prob:.4f}")
            
            print("\nTop states from GPU simulation:")
            for state, prob in gpu_top:
                print(f"|{state:010b}⟩: {prob:.4f}")
            
            # If we got here, GPU support is working!
            print("\n✅ GPU support is working correctly!")
            return True
            
        except Exception as e:
            print(f"❌ GPU simulation failed with error: {e}")
            print("\nThis could indicate that:")
            print("1. The GPU feature was not enabled during compilation")
            print("2. Your system doesn't have a compatible GPU")
            print("3. There's an issue with the GPU implementation")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import _quantrs2 module: {e}")
        print("Make sure you've installed the package with:")
        print("maturin develop --release --features gpu")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    print("QuantRS2 GPU Support Test")
    print("========================\n")
    test_gpu_support()