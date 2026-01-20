#!/usr/bin/env python3
"""
Test the enhanced Bell state implementation.

This script tests both direct Bell state creation and circuit simulation
for creating Bell states using the quantrs2 framework.
"""

import sys
import traceback

def main():
    """Run tests for the enhanced Bell state implementation."""
    try:
        print("Enhanced Bell State Test")
        print("=======================")
        
        # Import the quantrs2 module
        import quantrs2 as qr
        from math import sqrt
        
        print(f"Python version: {sys.version}")
        print(f"QuantRS2 version: {qr.__version__}")
        
        # Test the direct Bell state creation
        print("\nMethod 1: Direct Bell state creation")
        from quantrs2.bell_state import BellState
        
        # Test all Bell states
        bell_states = {
            "Phi+": BellState.phi_plus(),
            "Phi-": BellState.phi_minus(),
            "Psi+": BellState.psi_plus(),
            "Psi-": BellState.psi_minus()
        }
        
        for name, result in bell_states.items():
            print(f"\nTesting {name} Bell state:")
            try:
                probs = result.state_probabilities()
                print("State probabilities:")
                for state, prob in probs.items():
                    print(f"|{state}⟩: {prob:.6f}")
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
        
        # Test circuit simulation for Bell state
        print("\nMethod 2: Circuit simulation")
        try:
            # Create a 2-qubit circuit
            circuit = qr.PyCircuit(2)
            
            # Build a Bell state (Phi+)
            circuit.h(0)
            circuit.cnot(0, 1)
            
            # Run the simulation
            result = circuit.run()
            
            # Check the result
            if result is not None:
                print("Circuit simulation successful!")
                probs = result.state_probabilities()
                print("State probabilities:")
                for state, prob in probs.items():
                    print(f"|{state}⟩: {prob:.6f}")
            else:
                print("Circuit simulation failed (returned None)")
        except Exception as e:
            print(f"Error in circuit simulation: {e}")
            traceback.print_exc()
        
    except ImportError as e:
        print(f"Error importing quantrs2: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()