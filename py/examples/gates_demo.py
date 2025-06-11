#!/usr/bin/env python3
"""
Demonstration of quantum gates in QuantRS2.

This example shows how to use various quantum gates including:
- Standard single-qubit gates
- Parameterized rotation gates
- Multi-qubit gates
- Custom gates from matrices
"""

import numpy as np
from quantrs2 import Circuit
from quantrs2 import gates

def demo_basic_gates():
    """Demonstrate basic single-qubit gates."""
    print("=== Basic Single-Qubit Gates ===")
    
    # Create individual gates
    h_gate = gates.H(0)
    x_gate = gates.X(0)
    y_gate = gates.Y(0)
    z_gate = gates.Z(0)
    
    # Print gate information
    print(f"Hadamard gate: {h_gate}")
    print(f"Gate name: {h_gate.name}")
    print(f"Acts on qubits: {h_gate.qubits}")
    print(f"Is parameterized: {h_gate.is_parameterized}")
    
    # Get gate matrices
    h_matrix = h_gate.matrix()
    print(f"\nHadamard matrix shape: {h_matrix.shape}")
    print(f"Hadamard matrix:\n{h_matrix}")
    
    # Apply gates to a circuit
    circuit = Circuit(2)
    circuit.h(0)
    circuit.x(1)
    
    result = circuit.run()
    print(f"\nCircuit result after H(0) and X(1):")
    print(f"State probabilities: {result.state_probabilities()}")

def demo_rotation_gates():
    """Demonstrate rotation gates."""
    print("\n=== Rotation Gates ===")
    
    # Create rotation gates with different angles
    rx_gate = gates.RX(0, np.pi/2)
    ry_gate = gates.RY(0, np.pi/4)
    rz_gate = gates.RZ(0, np.pi/3)
    
    print(f"RX(π/2) gate: {rx_gate}")
    print(f"RY(π/4) gate: {ry_gate}")
    print(f"RZ(π/3) gate: {rz_gate}")
    
    # Apply rotation gates
    circuit = Circuit(1)
    circuit.rx(0, np.pi/2)  # Rotate around X by π/2
    
    result = circuit.run()
    print(f"\nAfter RX(π/2) on |0⟩:")
    print(f"Amplitudes: {result.amplitudes()}")
    print(f"Probabilities: {result.probabilities()}")

def demo_multi_qubit_gates():
    """Demonstrate multi-qubit gates."""
    print("\n=== Multi-Qubit Gates ===")
    
    # Create multi-qubit gates
    cnot_gate = gates.CNOT(0, 1)
    cz_gate = gates.CZ(0, 1)
    swap_gate = gates.SWAP(0, 1)
    
    print(f"CNOT gate: {cnot_gate}")
    print(f"CZ gate: {cz_gate}")
    print(f"SWAP gate: {swap_gate}")
    
    # Bell state using CNOT
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cnot(0, 1)
    
    result = circuit.run()
    print(f"\nBell state |Φ+⟩:")
    print(f"State probabilities: {result.state_probabilities()}")
    
    # Three-qubit gates
    toffoli_gate = gates.Toffoli(0, 1, 2)
    print(f"\nToffoli gate: {toffoli_gate}")
    print(f"Acts on qubits: {toffoli_gate.qubits}")

def demo_parametric_gates():
    """Demonstrate parametric gates for variational algorithms."""
    print("\n=== Parametric Gates ===")
    
    # Create parametric gates with symbolic parameters
    param_rx = gates.ParametricRX(0, "theta1")
    param_ry = gates.ParametricRY(0, "theta2")
    
    print(f"Parametric RX: {param_rx}")
    print(f"Parameters: {param_rx.parameters()}")
    print(f"Parameter names: {param_rx.parameter_names()}")
    
    # Assign values to parameters
    param_rx_assigned = param_rx.assign({"theta1": np.pi/2})
    print(f"\nAfter assigning theta1=π/2:")
    print(f"Parameters: {param_rx_assigned.parameters()}")
    
    # Create parametric U gate
    param_u = gates.ParametricU(0, "theta", "phi", "lambda")
    print(f"\nParametric U gate: {param_u}")
    print(f"Parameter names: {param_u.parameter_names()}")
    
    # Bind all parameters
    param_u_bound = param_u.bind({
        "theta": np.pi/2,
        "phi": np.pi/4,
        "lambda": 0
    })
    print(f"After binding all parameters: {param_u_bound.parameters()}")

def demo_custom_gates():
    """Demonstrate custom gates from matrices."""
    print("\n=== Custom Gates ===")
    
    # Create a custom single-qubit gate (π/8 gate)
    pi_8_matrix = np.array([
        [1, 0],
        [0, np.exp(1j * np.pi/8)]
    ], dtype=complex)
    
    custom_gate = gates.CustomGate("Pi/8", [0], pi_8_matrix)
    print(f"Custom π/8 gate: {custom_gate}")
    print(f"Gate name: {custom_gate.name}")
    
    # Create a custom two-qubit gate
    # This is an imaginary SWAP gate
    iswap_matrix = np.array([
        [1, 0, 0, 0],
        [0, 0, 1j, 0],
        [0, 1j, 0, 0],
        [0, 0, 0, 1]
    ], dtype=complex)
    
    iswap_gate = gates.CustomGate("iSWAP", [0, 1], iswap_matrix)
    print(f"\nCustom iSWAP gate: {iswap_gate}")
    print(f"Acts on qubits: {iswap_gate.qubits}")
    
    # Get the matrix back
    retrieved_matrix = iswap_gate.matrix()
    print(f"Retrieved matrix shape: {retrieved_matrix.shape}")

def demo_gate_sequences():
    """Demonstrate common gate sequences."""
    print("\n=== Gate Sequences ===")
    
    # QFT-like sequence on 2 qubits
    circuit = Circuit(2)
    circuit.h(0)
    circuit.crz(0, 1, np.pi/2)
    circuit.h(1)
    circuit.swap(0, 1)
    
    result = circuit.run()
    print("QFT-like sequence result:")
    print(f"State probabilities: {result.state_probabilities()}")
    
    # Grover-like sequence
    circuit2 = Circuit(2)
    # Initialize in uniform superposition
    circuit2.h(0)
    circuit2.h(1)
    # Oracle (mark |11⟩)
    circuit2.cz(0, 1)
    # Diffusion operator
    circuit2.h(0)
    circuit2.h(1)
    circuit2.x(0)
    circuit2.x(1)
    circuit2.cz(0, 1)
    circuit2.x(0)
    circuit2.x(1)
    circuit2.h(0)
    circuit2.h(1)
    
    result2 = circuit2.run()
    print("\nGrover-like sequence result:")
    print(f"State probabilities: {result2.state_probabilities()}")

def demo_phase_gates():
    """Demonstrate phase gates."""
    print("\n=== Phase Gates ===")
    
    # S and T gates
    s_gate = gates.S(0)
    t_gate = gates.T(0)
    sdg_gate = gates.SDagger(0)
    tdg_gate = gates.TDagger(0)
    
    print(f"S gate: {s_gate}")
    print(f"T gate: {t_gate}")
    
    # Apply phase gates
    circuit = Circuit(1)
    circuit.h(0)  # Create superposition
    circuit.s(0)  # Apply S gate
    circuit.h(0)  # Back to computational basis
    
    result = circuit.run()
    print(f"\nH-S-H sequence:")
    print(f"State probabilities: {result.state_probabilities()}")
    
    # T gate decomposition demonstration
    circuit2 = Circuit(1)
    circuit2.h(0)
    circuit2.t(0)
    circuit2.t(0)  # T² = S
    circuit2.h(0)
    
    result2 = circuit2.run()
    print(f"\nH-T-T-H sequence (should be same as H-S-H):")
    print(f"State probabilities: {result2.state_probabilities()}")

def main():
    """Run all demonstrations."""
    print("QuantRS2 Gates Demonstration")
    print("=" * 40)
    
    demo_basic_gates()
    demo_rotation_gates()
    demo_multi_qubit_gates()
    demo_parametric_gates()
    demo_custom_gates()
    demo_gate_sequences()
    demo_phase_gates()
    
    print("\n" + "=" * 40)
    print("Gate demonstrations completed!")

if __name__ == "__main__":
    main()