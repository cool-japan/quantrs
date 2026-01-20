#!/usr/bin/env python3
"""
Example using the Bell state utility functions.

This script demonstrates how to use the Bell state utility functions
from the quantrs2 module.
"""

import sys
import os
import traceback

def main():
    """Run the Bell state example."""
    try:
        import quantrs2 as qr
        from math import sqrt
        
        # Check for bell_state module
        try:
            from quantrs2.bell_state import simulate_bell_circuit, bell_state_probabilities
            has_bell_module = True
        except ImportError:
            has_bell_module = False
            # Create our own implementations
            def bell_state_probabilities():
                return {"00": 0.5, "11": 0.5}
            
            def create_bell_state():
                result = qr.PySimulationResult()
                result._amplitudes = [1/sqrt(2), 0, 0, 1/sqrt(2)]
                result._n_qubits = 2
                return result
        
        print("Bell State Example")
        print("=================")
        print(f"Python version: {sys.version}")
        print(f"QuantRS2 version: {qr.__version__}")
        print(f"Bell state module available: {has_bell_module}")
        
        # Option 1: Create a Bell state circuit and run it
        print("\nOption 1: Create a Bell state circuit")
        circuit = qr.PyCircuit(2)
        circuit.h(0)
        circuit.cnot(0, 1)
        
        # Run the simulation
        result = circuit.run()
        print(f"Result type: {type(result)}")
        
        # Check if we got a valid result
        if result is not None:
            print("We have a valid result from circuit.run()")
            
            # Try to access state_probabilities
            try:
                probs = result.state_probabilities()
                print("\nProbabilities:")
                for state, prob in probs.items():
                    print(f"|{state}⟩: {prob:.6f}")
            except Exception as e:
                print(f"Error accessing state_probabilities: {e}")
                print("Using fallback probabilities:")
                probs = bell_state_probabilities()
                for state, prob in probs.items():
                    print(f"|{state}⟩: {prob:.6f}")
        else:
            print("Result is None, using fallback")
            probs = bell_state_probabilities()
            print("\nFallback probabilities:")
            for state, prob in probs.items():
                print(f"|{state}⟩: {prob:.6f}")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()