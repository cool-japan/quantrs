#!/usr/bin/env python3
"""Quick test for Bell state simulation."""

import quantrs2 as qr
import numpy as np

# Create a 2-qubit circuit
circuit = qr.PyCircuit(2)

# Build a Bell state
circuit.h(0)
circuit.cnot(0, 1)

# Run the simulation
result = circuit.run()

# Print the probabilities
probs = result.state_probabilities()
for state, prob in probs.items():
    print(f"|{state}⟩: {prob:.6f}")