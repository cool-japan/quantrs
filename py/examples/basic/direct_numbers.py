#!/usr/bin/env python3
"""
Direct Bell state calculation.

This script directly calculates Bell state probabilities.
"""

from math import sqrt

def main():
    """Calculate Bell state probabilities without using the package."""
    # A Bell state is (|00⟩ + |11⟩)/√2
    # Create the amplitudes directly
    amplitudes = [1/sqrt(2), 0, 0, 1/sqrt(2)]
    
    # Calculate probabilities
    probabilities = [abs(amp)**2 for amp in amplitudes]
    
    # Print the results
    print("Bell State Probabilities")
    print("=======================")
    
    states = ["00", "01", "10", "11"]
    for i, (state, prob) in enumerate(zip(states, probabilities)):
        print(f"|{state}⟩: {prob:.6f}")
    
    # Summary
    print("\nThis is a direct calculation of probabilities for a Bell state.")
    print("The Bell state (|00⟩ + |11⟩)/√2 should have:")
    print("  - 50% probability of measuring |00⟩")
    print("  - 50% probability of measuring |11⟩")
    print("  - 0% probability of measuring |01⟩ or |10⟩")

if __name__ == "__main__":
    main()