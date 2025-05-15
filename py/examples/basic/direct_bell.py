#!/usr/bin/env python3
"""
Direct Bell state example using quantrs2.bell_state module.

This script directly uses the quantrs2.bell_state module to create and work with Bell states.
"""

import sys

def main():
    """Run the Bell state example using the bell_state module directly."""
    try:
        import quantrs2 as qr
        from quantrs2.bell_state import simulate_bell_circuit, bell_state_probabilities
        
        print("Direct Bell State Example")
        print("=======================")
        print(f"Python version: {sys.version}")
        print(f"QuantRS2 version: {qr.__version__}")
        
        # Use simulate_bell_circuit to get a Bell state
        print("\nCreating Bell state...")
        result = simulate_bell_circuit()
        print(f"Result type: {type(result)}")
        
        # Get probabilities from the Bell state
        print("\nGetting probabilities...")
        probs = result.state_probabilities()
        print("\nProbabilities:")
        for state, prob in probs.items():
            print(f"|{state}⟩: {prob:.6f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()