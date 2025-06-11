#!/usr/bin/env python3
"""
Explore quantum gate properties and decompositions.

This example demonstrates:
- Gate matrix properties (unitarity, hermiticity)
- Gate compositions and commutation relations
- Common gate decompositions
- Gate fidelity calculations
"""

import numpy as np
from quantrs2 import Circuit
from quantrs2 import gates

def check_unitarity(gate):
    """Check if a gate is unitary (U†U = I)."""
    matrix = gate.matrix()
    matrix_dag = np.conj(matrix.T)
    product = matrix_dag @ matrix
    identity = np.eye(matrix.shape[0])
    
    is_unitary = np.allclose(product, identity)
    error = np.max(np.abs(product - identity))
    
    print(f"{gate.name} gate unitarity check:")
    print(f"  Is unitary: {is_unitary}")
    print(f"  Max error: {error:.2e}")
    return is_unitary

def check_hermiticity(gate):
    """Check if a gate is Hermitian (H† = H)."""
    matrix = gate.matrix()
    matrix_dag = np.conj(matrix.T)
    
    is_hermitian = np.allclose(matrix, matrix_dag)
    error = np.max(np.abs(matrix - matrix_dag))
    
    print(f"{gate.name} gate Hermiticity check:")
    print(f"  Is Hermitian: {is_hermitian}")
    print(f"  Max error: {error:.2e}")
    return is_hermitian

def demo_gate_properties():
    """Demonstrate various gate properties."""
    print("=== Gate Properties ===\n")
    
    # Check unitarity of various gates
    gates_to_check = [
        gates.H(0),
        gates.X(0),
        gates.Y(0),
        gates.Z(0),
        gates.RX(0, np.pi/3),
        gates.CNOT(0, 1),
        gates.SWAP(0, 1)
    ]
    
    for gate in gates_to_check:
        check_unitarity(gate)
        print()
    
    # Check Hermiticity (self-adjoint gates)
    print("\n=== Hermitian Gates ===\n")
    hermitian_gates = [
        gates.X(0),
        gates.Y(0),
        gates.Z(0),
        gates.H(0),
        gates.SWAP(0, 1)
    ]
    
    for gate in hermitian_gates:
        check_hermiticity(gate)
        print()

def demo_gate_identities():
    """Demonstrate common gate identities."""
    print("=== Gate Identities ===\n")
    
    # X² = I
    circuit = Circuit(1)
    circuit.x(0)
    circuit.x(0)
    result = circuit.run()
    print("X² = I check:")
    print(f"  Final state: {result.state_probabilities()}")
    print(f"  Should be |0⟩: {result.state_probabilities().get('0', 0):.4f}")
    
    # H² = I
    circuit = Circuit(1)
    circuit.h(0)
    circuit.h(0)
    result = circuit.run()
    print("\nH² = I check:")
    print(f"  Final state: {result.state_probabilities()}")
    print(f"  Should be |0⟩: {result.state_probabilities().get('0', 0):.4f}")
    
    # S² = Z
    circuit1 = Circuit(1)
    circuit1.h(0)  # Create superposition for testing
    circuit1.s(0)
    circuit1.s(0)
    result1 = circuit1.run()
    
    circuit2 = Circuit(1)
    circuit2.h(0)
    circuit2.z(0)
    result2 = circuit2.run()
    
    print("\nS² = Z check:")
    print(f"  S² result: {result1.state_probabilities()}")
    print(f"  Z result:  {result2.state_probabilities()}")
    
    # T⁴ = Z
    circuit = Circuit(1)
    circuit.h(0)
    circuit.t(0)
    circuit.t(0)
    circuit.t(0)
    circuit.t(0)
    result = circuit.run()
    
    print("\nT⁴ = Z check:")
    print(f"  T⁴ result: {result.state_probabilities()}")

def demo_commutation_relations():
    """Demonstrate gate commutation relations."""
    print("\n=== Commutation Relations ===\n")
    
    # Pauli gates anti-commute
    # XY = -YX
    circuit1 = Circuit(1)
    circuit1.x(0)
    circuit1.y(0)
    result1 = circuit1.run()
    
    circuit2 = Circuit(1)
    circuit2.y(0)
    circuit2.x(0)
    result2 = circuit2.run()
    
    print("XY vs YX (should differ by phase):")
    print(f"  XY result: {result1.amplitudes()}")
    print(f"  YX result: {result2.amplitudes()}")
    
    # RZ gates commute
    circuit1 = Circuit(1)
    circuit1.h(0)  # Create superposition
    circuit1.rz(0, np.pi/4)
    circuit1.rz(0, np.pi/3)
    result1 = circuit1.run()
    
    circuit2 = Circuit(1)
    circuit2.h(0)
    circuit2.rz(0, np.pi/3)
    circuit2.rz(0, np.pi/4)
    result2 = circuit2.run()
    
    print("\nRZ commutation (should be identical):")
    print(f"  RZ(π/4)RZ(π/3): {result1.state_probabilities()}")
    print(f"  RZ(π/3)RZ(π/4): {result2.state_probabilities()}")

def demo_gate_decompositions():
    """Demonstrate common gate decompositions."""
    print("\n=== Gate Decompositions ===\n")
    
    # Arbitrary single-qubit gate as ZYZ decomposition
    # U = RZ(α)RY(β)RZ(γ)
    alpha, beta, gamma = np.pi/4, np.pi/3, np.pi/6
    
    circuit = Circuit(1)
    circuit.rz(0, gamma)
    circuit.ry(0, beta)
    circuit.rz(0, alpha)
    
    print("ZYZ decomposition of arbitrary single-qubit gate:")
    print(f"  U = RZ({alpha:.3f})RY({beta:.3f})RZ({gamma:.3f})")
    
    # CNOT decomposition in terms of CZ and Hadamards
    # CNOT = H(target) · CZ · H(target)
    circuit1 = Circuit(2)
    circuit1.h(0)
    circuit1.cnot(0, 1)
    result1 = circuit1.run()
    
    circuit2 = Circuit(2)
    circuit2.h(0)
    circuit2.h(1)
    circuit2.cz(0, 1)
    circuit2.h(1)
    result2 = circuit2.run()
    
    print("\nCNOT = H·CZ·H decomposition:")
    print(f"  Direct CNOT: {result1.state_probabilities()}")
    print(f"  H·CZ·H:      {result2.state_probabilities()}")
    
    # Toffoli gate approximation using single and two-qubit gates
    # This is a simplified version - full decomposition uses 6 CNOTs
    circuit = Circuit(3)
    circuit.h(2)
    circuit.cnot(1, 2)
    circuit.tdg(2)
    circuit.cnot(0, 2)
    circuit.t(2)
    circuit.cnot(1, 2)
    circuit.tdg(2)
    circuit.cnot(0, 2)
    circuit.t(1)
    circuit.t(2)
    circuit.h(2)
    circuit.cnot(0, 1)
    circuit.t(0)
    circuit.tdg(1)
    circuit.cnot(0, 1)
    
    print("\nToffoli gate decomposition uses 6 CNOTs and single-qubit gates")

def demo_controlled_gates():
    """Demonstrate controlled gate constructions."""
    print("\n=== Controlled Gates ===\n")
    
    # Controlled-H gate
    circuit1 = Circuit(2)
    circuit1.x(0)  # Set control to |1⟩
    circuit1.ch(0, 1)
    result1 = circuit1.run()
    
    circuit2 = Circuit(2)
    circuit2.x(0)
    circuit2.h(1)
    result2 = circuit2.run()
    
    print("Controlled-H gate:")
    print(f"  CH with control=1: {result1.state_probabilities()}")
    print(f"  Just H on target:  {result2.state_probabilities()}")
    
    # Controlled rotation
    angle = np.pi/3
    circuit = Circuit(2)
    circuit.h(0)  # Superposition on control
    circuit.x(1)  # Target in |1⟩
    circuit.crx(0, 1, angle)
    result = circuit.run()
    
    print(f"\nControlled-RX({angle:.3f}) with control in superposition:")
    print(f"  Result: {result.state_probabilities()}")

def calculate_gate_fidelity(gate1, gate2):
    """Calculate fidelity between two gates."""
    m1 = gate1.matrix()
    m2 = gate2.matrix()
    
    # For unitary gates, fidelity = |Tr(U1† U2)|² / d²
    d = m1.shape[0]
    trace = np.trace(np.conj(m1.T) @ m2)
    fidelity = np.abs(trace)**2 / d**2
    
    print(f"Fidelity between {gate1.name} and {gate2.name}: {fidelity:.6f}")
    return fidelity

def demo_gate_fidelity():
    """Demonstrate gate fidelity calculations."""
    print("\n=== Gate Fidelity ===\n")
    
    # Compare different rotation angles
    rx1 = gates.RX(0, np.pi/4)
    rx2 = gates.RX(0, np.pi/4 + 0.01)  # Small error
    calculate_gate_fidelity(rx1, rx2)
    
    # Compare S gate with two T gates
    s_gate = gates.S(0)
    
    # Create custom gate for T²
    t_matrix = gates.T(0).matrix()
    t2_matrix = t_matrix @ t_matrix
    t2_gate = gates.CustomGate("T²", [0], t2_matrix.reshape(2, 2))
    
    calculate_gate_fidelity(s_gate, t2_gate)
    
    # Compare identity with small rotation
    i_matrix = np.eye(2, dtype=complex)
    i_gate = gates.CustomGate("I", [0], i_matrix)
    small_rot = gates.RZ(0, 0.001)
    
    calculate_gate_fidelity(i_gate, small_rot)

def main():
    """Run all demonstrations."""
    print("QuantRS2 Gate Properties and Decompositions")
    print("=" * 50)
    
    demo_gate_properties()
    demo_gate_identities()
    demo_commutation_relations()
    demo_gate_decompositions()
    demo_controlled_gates()
    demo_gate_fidelity()
    
    print("\n" + "=" * 50)
    print("Gate property demonstrations completed!")

if __name__ == "__main__":
    main()