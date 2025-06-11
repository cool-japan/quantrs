#!/usr/bin/env python3
"""
Demonstration of custom gate definitions in QuantRS2.

This example shows how to:
1. Define custom gates from unitary matrices
2. Create parametric gates with Python functions
3. Build gates from decompositions
4. Register and use custom gates
5. Create controlled versions of gates
"""

import numpy as np
from quantrs2.custom_gates import (
    CustomGate, GateBuilder, GateRegistry,
    create_phase_gate, create_rotation_gate
)

def custom_gate_from_matrix():
    """Example of creating a custom gate from a unitary matrix."""
    print("=== Custom Gate from Matrix ===")
    
    # Define a custom single-qubit gate (sqrt(Y) gate)
    sqrt_y_matrix = np.array([
        [(1+1j)/2, -(1+1j)/2],
        [(1+1j)/2, (1+1j)/2]
    ], dtype=np.complex128)
    
    # Create the custom gate
    sqrt_y_gate = CustomGate.from_matrix("SqrtY", sqrt_y_matrix)
    
    print(f"Created gate: {sqrt_y_gate.name}")
    print(f"Number of qubits: {sqrt_y_gate.num_qubits}")
    print(f"Number of parameters: {sqrt_y_gate.num_params}")
    
    # Get the matrix back
    matrix = sqrt_y_gate.get_matrix()
    print(f"Gate matrix:\n{matrix}")


def parametric_gate_example():
    """Example of creating a parametric custom gate."""
    print("\n=== Parametric Custom Gate ===")
    
    # Define a parametric rotation gate around arbitrary axis
    def rotation_matrix(params):
        theta, phi, lam = params
        # General single-qubit rotation
        return np.array([
            [np.cos(theta/2), -np.exp(1j*lam)*np.sin(theta/2)],
            [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lam))*np.cos(theta/2)]
        ], dtype=np.complex128)
    
    # Create parametric gate
    u3_gate = CustomGate.from_function(
        "U3",
        num_qubits=1,
        num_params=3,
        matrix_fn=rotation_matrix
    )
    
    print(f"Created parametric gate: {u3_gate.name}")
    print(f"Number of parameters: {u3_gate.num_params}")
    
    # Get matrix for specific parameters
    params = [np.pi/4, np.pi/6, np.pi/8]
    matrix = u3_gate.get_matrix(params)
    print(f"Gate matrix for params {params}:\n{matrix}")


def decomposition_example():
    """Example of creating a gate from decomposition."""
    print("\n=== Gate from Decomposition ===")
    
    # Create a Toffoli gate decomposition
    toffoli_decomp = [
        ("h", [2], None),
        ("cx", [1, 2], None),
        ("tdg", [2], None),
        ("cx", [0, 2], None),
        ("t", [2], None),
        ("cx", [1, 2], None),
        ("tdg", [2], None),
        ("cx", [0, 2], None),
        ("t", [1], None),
        ("t", [2], None),
        ("h", [2], None),
        ("cx", [0, 1], None),
        ("t", [0], None),
        ("tdg", [1], None),
        ("cx", [0, 1], None),
    ]
    
    # Create the custom gate
    toffoli_custom = CustomGate.from_decomposition(
        "ToffoliDecomp",
        num_qubits=3,
        gates=toffoli_decomp
    )
    
    print(f"Created decomposed gate: {toffoli_custom.name}")
    print(f"Number of qubits: {toffoli_custom.num_qubits}")
    print(f"Decomposition has {len(toffoli_decomp)} gates")


def gate_builder_example():
    """Example using the GateBuilder fluent API."""
    print("\n=== Gate Builder Example ===")
    
    # Build a custom phase gate
    builder = GateBuilder()
    builder.with_name("CustomPhase")
    
    # Create phase matrix
    phase = np.pi/3
    phase_matrix = np.array([
        [1, 0],
        [0, np.exp(1j*phase)]
    ], dtype=np.complex128)
    
    builder.with_matrix(phase_matrix)
    custom_phase = builder.build()
    
    print(f"Built gate: {custom_phase.name}")
    print(f"Gate matrix:\n{custom_phase.get_matrix()}")
    
    # Build gate from decomposition
    builder2 = GateBuilder()
    builder2.with_name("MyGate")
    builder2.add_gate("h", [0], None)
    builder2.add_gate("rz", [0], [np.pi/4])
    builder2.add_gate("h", [0], None)
    
    # Note: Need to set num_qubits for decomposition
    # This is a limitation of the current implementation
    # decomposed_gate = builder2.build()


def gate_registry_example():
    """Example of using the gate registry."""
    print("\n=== Gate Registry Example ===")
    
    # Create registry
    registry = GateRegistry()
    
    # Register some custom gates
    sqrt_x = CustomGate.from_matrix("SqrtX", np.array([
        [(1+1j)/2, (1-1j)/2],
        [(1-1j)/2, (1+1j)/2]
    ], dtype=np.complex128))
    
    sqrt_z = CustomGate.from_matrix("SqrtZ", np.array([
        [1, 0],
        [0, 1j]
    ], dtype=np.complex128))
    
    registry.register(sqrt_x)
    registry.register(sqrt_z)
    
    # List registered gates
    print(f"Registered gates: {registry.list_gates()}")
    
    # Check if gates exist
    print(f"Has SqrtX: {registry.contains('SqrtX')}")
    print(f"Has SqrtY: {registry.contains('SqrtY')}")
    
    # Retrieve a gate
    retrieved = registry.get("SqrtX")
    if retrieved:
        print(f"Retrieved gate: {retrieved.name}")


def controlled_gate_example():
    """Example of creating controlled versions of gates."""
    print("\n=== Controlled Gate Example ===")
    
    # Create a controlled version of a custom gate
    controlled_sqrt_x = CustomGate.controlled(
        "CSqrtX",
        base_gate="SqrtX",
        num_controls=1
    )
    
    print(f"Created controlled gate: {controlled_sqrt_x.name}")
    print(f"Number of qubits: {controlled_sqrt_x.num_qubits}")
    
    # Create doubly-controlled gate
    cc_phase = CustomGate.controlled(
        "CCPhase",
        base_gate="Phase",
        num_controls=2
    )
    
    print(f"Created doubly-controlled gate: {cc_phase.name}")


def template_gates_example():
    """Example using gate templates."""
    print("\n=== Gate Templates Example ===")
    
    # Create a phase gate
    phase_gate = create_phase_gate(np.pi/4)
    print(f"Created phase gate: {phase_gate.name}")
    matrix = phase_gate.get_matrix()
    print(f"Phase gate matrix:\n{matrix}")
    
    # Create rotation around arbitrary axis
    axis = [1, 1, 0]  # X+Y axis
    angle = np.pi/3
    rot_gate = create_rotation_gate(axis, angle)
    print(f"\nCreated rotation gate: {rot_gate.name}")
    matrix = rot_gate.get_matrix()
    print(f"Rotation gate matrix:\n{matrix}")


def two_qubit_custom_gate():
    """Example of a custom two-qubit gate."""
    print("\n=== Two-Qubit Custom Gate ===")
    
    # Create a custom entangling gate
    # This is the sqrt(iSWAP) gate
    sqrt_iswap_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1/np.sqrt(2), 1j/np.sqrt(2), 0],
        [0, 1j/np.sqrt(2), 1/np.sqrt(2), 0],
        [0, 0, 0, 1]
    ], dtype=np.complex128)
    
    sqrt_iswap = CustomGate.from_matrix("SqrtISWAP", sqrt_iswap_matrix)
    
    print(f"Created gate: {sqrt_iswap.name}")
    print(f"Number of qubits: {sqrt_iswap.num_qubits}")
    print(f"Gate matrix shape: {sqrt_iswap.get_matrix().shape}")


def verify_unitarity():
    """Example showing unitary validation."""
    print("\n=== Unitary Validation ===")
    
    # Try to create a non-unitary gate (should fail)
    non_unitary = np.array([
        [1, 0],
        [0, 2]  # Not unitary!
    ], dtype=np.complex128)
    
    try:
        bad_gate = CustomGate.from_matrix("BadGate", non_unitary)
        print("ERROR: Non-unitary gate was accepted!")
    except ValueError as e:
        print(f"Correctly rejected non-unitary matrix: {e}")
    
    # Create a properly unitary gate
    unitary = np.array([
        [1/np.sqrt(2), 1/np.sqrt(2)],
        [1/np.sqrt(2), -1/np.sqrt(2)]
    ], dtype=np.complex128)
    
    good_gate = CustomGate.from_matrix("GoodGate", unitary)
    print(f"Successfully created unitary gate: {good_gate.name}")


if __name__ == "__main__":
    custom_gate_from_matrix()
    parametric_gate_example()
    decomposition_example()
    gate_builder_example()
    gate_registry_example()
    controlled_gate_example()
    template_gates_example()
    two_qubit_custom_gate()
    verify_unitarity()
    
    print("\n=== Custom Gates Demo Complete ===")