#!/usr/bin/env python3
"""
Bell State Demonstration for QuantRS2.

This script demonstrates how to create and work with Bell states
using the quantrs2 framework, particularly focusing on the enhanced
Bell state utilities provided in the bell_state module.
"""

import sys
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

def main():
    """Demonstrate Bell state creation and analysis."""
    try:
        # Import QuantRS2
        import quantrs2 as qr
        
        print("Bell State Demonstration")
        print("=======================")
        print(f"Python version: {sys.version}")
        print(f"QuantRS2 version: {qr.__version__}")
        
        # Method 1: Create Bell state by direct simulation
        print("\n1. Creating Bell state using circuit simulation:")
        circuit = qr.PyCircuit(2)
        circuit.h(0)
        circuit.cnot(0, 1)
        result = circuit.run()
        
        # Display result
        probs = result.state_probabilities()
        print("State probabilities:")
        for state, prob in probs.items():
            print(f"|{state}⟩: {prob:.6f}")
        
        # Method 2: Use Bell state utilities
        print("\n2. Using Bell state utilities:")
        
        # Import Bell state utilities
        from quantrs2.bell_state import BellState
        
        # Create Bell states
        phi_plus = BellState.phi_plus()   # |00⟩ + |11⟩)/√2
        phi_minus = BellState.phi_minus() # |00⟩ - |11⟩)/√2
        psi_plus = BellState.psi_plus()   # |01⟩ + |10⟩)/√2
        psi_minus = BellState.psi_minus() # |01⟩ - |10⟩)/√2
        
        # Show all Bell states
        bell_states = {
            "Φ⁺ (|00⟩ + |11⟩)/√2": phi_plus,
            "Φ⁻ (|00⟩ - |11⟩)/√2": phi_minus,
            "Ψ⁺ (|01⟩ + |10⟩)/√2": psi_plus,
            "Ψ⁻ (|01⟩ - |10⟩)/√2": psi_minus
        }
        
        # Display results
        for name, state in bell_states.items():
            probs = state.state_probabilities()
            print(f"\n{name}:")
            for basis, prob in probs.items():
                print(f"|{basis}⟩: {prob:.6f}")
        
        # Visualize Bell state probabilities
        try:
            # Set up the plot
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            axes = axes.flatten()
            
            for i, (name, state) in enumerate(bell_states.items()):
                probs = state.state_probabilities()
                
                # Sort states for consistent visualization
                sorted_states = sorted(probs.items())
                states = [s[0] for s in sorted_states]
                values = [s[1] for s in sorted_states]
                
                # Plot the probabilities
                ax = axes[i]
                ax.bar(states, values)
                ax.set_title(name)
                ax.set_ylim(0, 1)
                ax.set_ylabel("Probability")
                ax.set_xlabel("Basis State")
            
            plt.tight_layout()
            plt.savefig("bell_states.png")
            print("\nVisualization saved to bell_states.png")
        except Exception as e:
            print(f"Visualization failed: {e}")
        
    except ImportError as e:
        print(f"Error importing quantrs2: {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()