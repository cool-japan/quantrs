#!/usr/bin/env python3
"""
Variational quantum algorithms using parametric gates.

This example demonstrates:
- Creating variational quantum circuits
- Using symbolic parameters
- Parameter optimization workflows
- Common variational ansätze
"""

import numpy as np
from quantrs2 import Circuit
from quantrs2 import gates

def create_single_qubit_variational_layer(circuit, qubit, param_prefix):
    """Create a general single-qubit variational layer."""
    # Most general single-qubit gate: U3(θ, φ, λ)
    # For now, use RY-RZ decomposition
    circuit.ry(qubit, 0.0)  # Will be replaced with parametric version
    circuit.rz(qubit, 0.0)  # Will be replaced with parametric version
    
    print(f"Added variational layer to qubit {qubit} with prefix {param_prefix}")

def create_ry_ansatz(n_qubits, depth):
    """Create a hardware-efficient RY ansatz."""
    print(f"\n=== Creating RY Ansatz (n_qubits={n_qubits}, depth={depth}) ===")
    
    circuit = Circuit(n_qubits)
    param_count = 0
    
    for layer in range(depth):
        # Single-qubit rotations
        for q in range(n_qubits):
            angle = np.random.uniform(0, 2*np.pi)
            circuit.ry(q, angle)
            param_count += 1
            
        # Entangling layer (if not last layer)
        if layer < depth - 1:
            for q in range(n_qubits - 1):
                circuit.cnot(q, q + 1)
            # Add wrap-around CNOT for periodic boundary
            if n_qubits > 2:
                circuit.cnot(n_qubits - 1, 0)
    
    print(f"Created ansatz with {param_count} parameters")
    return circuit

def create_ry_rz_ansatz(n_qubits, depth):
    """Create a more expressive RY-RZ ansatz."""
    print(f"\n=== Creating RY-RZ Ansatz (n_qubits={n_qubits}, depth={depth}) ===")
    
    circuit = Circuit(n_qubits)
    param_count = 0
    
    for layer in range(depth):
        # RY rotations
        for q in range(n_qubits):
            angle = np.random.uniform(0, 2*np.pi)
            circuit.ry(q, angle)
            param_count += 1
            
        # RZ rotations
        for q in range(n_qubits):
            angle = np.random.uniform(0, 2*np.pi)
            circuit.rz(q, angle)
            param_count += 1
            
        # Entangling layer
        if layer < depth - 1:
            for q in range(0, n_qubits - 1, 2):
                circuit.cnot(q, q + 1)
            for q in range(1, n_qubits - 1, 2):
                circuit.cnot(q, q + 1)
    
    print(f"Created ansatz with {param_count} parameters")
    return circuit

def demo_parametric_gates():
    """Demonstrate parametric gate functionality."""
    print("=== Parametric Gates Demo ===")
    
    # Create parametric rotation gates
    param_rx = gates.ParametricRX(0, "theta_x")
    param_ry = gates.ParametricRY(0, "theta_y")
    param_rz = gates.ParametricRZ(0, "theta_z")
    
    print(f"\nParametric gates created:")
    print(f"  RX: {param_rx}")
    print(f"  RY: {param_ry}")
    print(f"  RZ: {param_rz}")
    
    # Show parameter information
    print(f"\nRX parameters: {param_rx.parameters()}")
    print(f"RX parameter names: {param_rx.parameter_names()}")
    
    # Assign values
    param_rx_bound = param_rx.assign({"theta_x": np.pi/2})
    print(f"\nAfter binding theta_x = π/2:")
    print(f"  Parameters: {param_rx_bound.parameters()}")
    
    # Create parametric U gate
    param_u = gates.ParametricU(0, "alpha", "beta", "gamma")
    print(f"\nParametric U gate: {param_u}")
    print(f"  Parameter names: {param_u.parameter_names()}")

def create_qaoa_ansatz(n_qubits, p):
    """Create a QAOA-style ansatz."""
    print(f"\n=== Creating QAOA Ansatz (n_qubits={n_qubits}, p={p}) ===")
    
    circuit = Circuit(n_qubits)
    
    # Initial state: uniform superposition
    for q in range(n_qubits):
        circuit.h(q)
    
    # p rounds of alternating operators
    for round in range(p):
        # Problem Hamiltonian layer (ZZ interactions)
        # Using random parameters for demonstration
        for q in range(n_qubits - 1):
            gamma = np.random.uniform(0, np.pi)
            circuit.cnot(q, q + 1)
            circuit.rz(q + 1, 2 * gamma)
            circuit.cnot(q, q + 1)
        
        # Mixer Hamiltonian layer (X rotations)
        for q in range(n_qubits):
            beta = np.random.uniform(0, np.pi)
            circuit.rx(q, 2 * beta)
    
    print(f"Created QAOA ansatz with {2 * p * n_qubits - p} parameters")
    return circuit

def create_ucc_inspired_ansatz(n_qubits):
    """Create a chemistry-inspired ansatz."""
    print(f"\n=== Creating UCC-Inspired Ansatz (n_qubits={n_qubits}) ===")
    
    circuit = Circuit(n_qubits)
    
    # Reference state (e.g., Hartree-Fock)
    # Put electrons in lowest orbitals
    n_electrons = n_qubits // 2
    for q in range(n_electrons):
        circuit.x(q)
    
    # Single excitations
    for i in range(n_electrons):
        for a in range(n_electrons, n_qubits):
            theta = np.random.uniform(-np.pi, np.pi)
            
            # Simplified excitation operator
            circuit.cnot(i, a)
            circuit.ry(a, theta)
            circuit.cnot(i, a)
    
    # Double excitations (simplified)
    if n_qubits >= 4:
        for i in range(0, n_electrons - 1):
            for j in range(i + 1, n_electrons):
                for a in range(n_electrons, n_qubits - 1):
                    for b in range(a + 1, n_qubits):
                        theta = np.random.uniform(-np.pi/2, np.pi/2)
                        
                        # Very simplified double excitation
                        circuit.cnot(i, a)
                        circuit.cnot(j, b)
                        circuit.rz(b, theta)
                        circuit.cnot(j, b)
                        circuit.cnot(i, a)
    
    return circuit

def measure_expressibility(circuit, n_samples=1000):
    """Measure the expressibility of a variational circuit."""
    print("\n=== Measuring Circuit Expressibility ===")
    
    # This is a simplified version - real expressibility requires
    # computing fidelities between random instances
    
    states = []
    for _ in range(min(n_samples, 10)):  # Limit for demonstration
        result = circuit.run()
        probs = result.state_probabilities()
        states.append(probs)
    
    # Simple diversity measure
    unique_states = len(set(str(s) for s in states))
    print(f"Unique state distributions: {unique_states}/{len(states)}")

def demo_variational_workflow():
    """Demonstrate a typical variational algorithm workflow."""
    print("\n=== Variational Algorithm Workflow ===")
    
    # Step 1: Choose ansatz
    n_qubits = 3
    depth = 2
    circuit = create_ry_rz_ansatz(n_qubits, depth)
    
    # Step 2: Prepare measurement
    # Add measurements (in real VQE, measure in different bases)
    print("\nPreparing measurements...")
    
    # Step 3: Run circuit with current parameters
    result = circuit.run()
    print(f"\nInitial state probabilities:")
    print(result.state_probabilities())
    
    # Step 4: Compute expectation value
    # For demo, just compute <Z0>
    z0_expectation = 0.0
    for state, prob in result.state_probabilities().items():
        # Z eigenvalue is +1 for |0⟩ and -1 for |1⟩ on first qubit
        if state[0] == '0':
            z0_expectation += prob
        else:
            z0_expectation -= prob
    
    print(f"\nExpectation value <Z0>: {z0_expectation:.4f}")
    
    # Step 5: Update parameters (would use optimizer in practice)
    print("\nIn a real VQE, parameters would be updated by optimizer")
    print("Common optimizers: COBYLA, L-BFGS-B, SPSA, Adam")

def demo_entanglement_patterns():
    """Demonstrate different entanglement patterns in ansätze."""
    print("\n=== Entanglement Patterns ===")
    
    n_qubits = 6
    
    # Linear entanglement
    print("\n1. Linear Entanglement:")
    circuit1 = Circuit(n_qubits)
    for q in range(n_qubits):
        circuit1.ry(q, np.random.uniform(0, 2*np.pi))
    for q in range(n_qubits - 1):
        circuit1.cnot(q, q + 1)
    print("   0-1-2-3-4-5")
    
    # Circular entanglement
    print("\n2. Circular Entanglement:")
    circuit2 = Circuit(n_qubits)
    for q in range(n_qubits):
        circuit2.ry(q, np.random.uniform(0, 2*np.pi))
    for q in range(n_qubits - 1):
        circuit2.cnot(q, q + 1)
    circuit2.cnot(n_qubits - 1, 0)  # Close the loop
    print("   0-1-2-3-4-5-0")
    
    # All-to-all entanglement
    print("\n3. All-to-All Entanglement:")
    circuit3 = Circuit(n_qubits)
    for q in range(n_qubits):
        circuit3.ry(q, np.random.uniform(0, 2*np.pi))
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            circuit3.cz(i, j)
    print("   Fully connected")
    
    # Alternating layers
    print("\n4. Alternating Layer Entanglement:")
    circuit4 = Circuit(n_qubits)
    for q in range(n_qubits):
        circuit4.ry(q, np.random.uniform(0, 2*np.pi))
    # Even pairs
    for q in range(0, n_qubits - 1, 2):
        circuit4.cnot(q, q + 1)
    # Odd pairs
    for q in range(1, n_qubits - 1, 2):
        circuit4.cnot(q, q + 1)
    print("   Layer 1: (0,1) (2,3) (4,5)")
    print("   Layer 2: (1,2) (3,4)")

def main():
    """Run all demonstrations."""
    print("QuantRS2 Variational Gates Demonstration")
    print("=" * 50)
    
    demo_parametric_gates()
    
    # Create different ansätze
    create_ry_ansatz(4, 3)
    create_ry_rz_ansatz(4, 2)
    create_qaoa_ansatz(4, 2)
    create_ucc_inspired_ansatz(4)
    
    # Demonstrate workflow
    demo_variational_workflow()
    
    # Show entanglement patterns
    demo_entanglement_patterns()
    
    print("\n" + "=" * 50)
    print("Variational gates demonstration completed!")

if __name__ == "__main__":
    main()