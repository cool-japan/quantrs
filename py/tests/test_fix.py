#!/usr/bin/env python3
"""
Test script for the fixed PyO3 bindings.

This script tests the PyO3 bindings for the Bell state generation.
"""

import sys
import traceback

def main():
    """Run the test for the fixed PyO3 bindings."""
    try:
        import quantrs2 as qr
        from math import sqrt
        import numpy as np
        
        print(f"Python version: {sys.version}")
        print(f"QuantRS2 version: {qr.__version__}")
        
        # Create a 2-qubit circuit
        circuit = qr.PyCircuit(2)
        
        # Build a Bell state
        circuit.h(0)
        circuit.cnot(0, 1)
        
        # Run the simulation
        print("\nRunning simulation...")
        result = circuit.run()
        print(f"Result: {result}")
        print(f"Result type: {type(result)}")
        
        # Check if the result is valid
        if result is not None:
            print("\nResult is valid!")
            
            # Try to access amplitudes
            try:
                amps = result.amplitudes()
                print(f"Amplitudes: {amps}")
            except Exception as e:
                print(f"Error accessing amplitudes: {e}")
            
            # Try to access n_qubits
            try:
                n_qubits = result.get_n_qubits()
                print(f"Number of qubits: {n_qubits}")
            except Exception as e:
                print(f"Error accessing n_qubits: {e}")
            
            # Try to access state_probabilities
            try:
                probs = result.state_probabilities()
                print("\nProbabilities:")
                for state, prob in probs.items():
                    print(f"|{state}⟩: {prob:.6f}")
            except Exception as e:
                print(f"Error accessing state_probabilities: {e}")
                traceback.print_exc()
        else:
            print("Result is still None!")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()