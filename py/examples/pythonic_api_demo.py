#!/usr/bin/env python3
"""
Demonstration of Pythonic API with Qiskit/Cirq-style conventions.

This example shows how to use familiar APIs from:
1. Qiskit-style QuantumCircuit
2. Cirq-style circuit construction
3. Register-based qubit management
4. Compatible gate naming conventions
"""

import numpy as np
from quantrs2 import QuantumCircuit, QuantumRegister, ClassicalRegister
from quantrs2.cirq_compat import Circuit as CirqCircuit, Gates, LineQubit, GridQubit

def qiskit_style_example():
    """Example using Qiskit-style API."""
    print("=== Qiskit-Style API Example ===")
    
    # Create quantum and classical registers
    qreg = QuantumRegister(3, 'q')
    creg = ClassicalRegister(3, 'c')
    
    # Create circuit with registers
    circuit = QuantumCircuit(qreg, creg)
    
    # Build a GHZ state using Qiskit conventions
    circuit.h(0)  # Can use integer indices
    circuit.cx(0, 1)  # cx is CNOT in Qiskit
    circuit.cx(1, 2)
    
    # Add measurements
    circuit.measure(0, 0)
    circuit.measure(1, 1)
    circuit.measure(2, 2)
    
    # Draw the circuit
    print("Circuit diagram:")
    print(circuit.draw())
    
    # Execute the circuit
    result = circuit.execute()
    print(f"\nExecution result: {result}")


def qiskit_style_advanced():
    """Advanced Qiskit-style features."""
    print("\n=== Advanced Qiskit-Style Features ===")
    
    # Create circuit with direct qubit count
    qc = QuantumCircuit(5)
    
    # Use various gates with Qiskit naming
    qc.h(0)
    qc.x(1)
    qc.y(2)
    qc.z(3)
    
    # Rotation gates
    qc.rx(np.pi/4, 0)
    qc.ry(np.pi/3, 1)
    qc.rz(np.pi/2, 2)
    
    # Phase gates
    qc.s(0)
    qc.sdg(1)  # S-dagger
    qc.t(2)
    qc.tdg(3)  # T-dagger
    
    # Two-qubit gates
    qc.cx(0, 1)  # CNOT
    qc.cy(1, 2)  # Controlled-Y
    qc.cz(2, 3)  # Controlled-Z
    qc.swap(3, 4)
    
    # Three-qubit gates
    qc.ccx(0, 1, 2)  # Toffoli
    qc.toffoli(1, 2, 3)  # Alias
    
    # Add barrier (for visualization)
    qc.barrier()
    
    # Measure all qubits
    qc.measure_all()
    
    print(f"Circuit depth: {qc.depth()}")
    print(f"Number of qubits: {qc.num_qubits}")


def cirq_style_example():
    """Example using Cirq-style API."""
    print("\n=== Cirq-Style API Example ===")
    
    # Create a circuit
    circuit = CirqCircuit()
    
    # Define qubits
    q0, q1, q2 = 0, 1, 2  # Simple indices
    
    # Add gates in moments (Cirq style)
    circuit.append([
        Gates.H(q0),
        Gates.X(q1),
    ])
    
    circuit.append([
        Gates.CNOT(q0, q1),
        Gates.Z(q2),
    ])
    
    circuit.append([
        Gates.rx(np.pi/4, q0),
        Gates.ry(np.pi/3, q1),
        Gates.rz(np.pi/2, q2),
    ])
    
    # Show circuit structure
    print("Circuit moments:")
    for i, moment in enumerate(circuit.moments()):
        print(f"  Moment {i}: {moment}")
    
    print(f"\nAll qubits used: {circuit.all_qubits()}")
    
    # Convert to QuantRS circuit for execution
    quantrs_circuit = circuit.to_quantrs()
    print("\nConverted to QuantRS circuit for execution")


def cirq_style_grid_qubits():
    """Example using Cirq-style grid qubits."""
    print("\n=== Cirq-Style Grid Qubits ===")
    
    circuit = CirqCircuit()
    
    # Define qubits on a 2D grid
    q00 = GridQubit(0, 0, n_cols=3)
    q01 = GridQubit(0, 1, n_cols=3)
    q10 = GridQubit(1, 0, n_cols=3)
    q11 = GridQubit(1, 1, n_cols=3)
    
    print(f"Grid qubit mapping:")
    print(f"  (0,0) -> qubit {q00}")
    print(f"  (0,1) -> qubit {q01}")
    print(f"  (1,0) -> qubit {q10}")
    print(f"  (1,1) -> qubit {q11}")
    
    # Build a 2D entangled state
    circuit.append([Gates.H(q00)])
    circuit.append([
        Gates.CNOT(q00, q01),
        Gates.CNOT(q00, q10),
    ])
    circuit.append([Gates.CNOT(q10, q11)])
    
    print(f"\nCircuit uses qubits: {circuit.all_qubits()}")


def line_qubit_example():
    """Example using Cirq-style line qubits."""
    print("\n=== Cirq-Style Line Qubits ===")
    
    circuit = CirqCircuit()
    
    # Define line qubits (1D chain)
    qubits = [LineQubit(i) for i in range(5)]
    
    # Create a chain of entanglement
    circuit.append([Gates.H(qubits[0])])
    
    for i in range(4):
        circuit.append([Gates.CNOT(qubits[i], qubits[i+1])])
    
    print("Created entanglement chain on line qubits 0-4")


def mixed_api_example():
    """Example mixing different API styles."""
    print("\n=== Mixed API Example ===")
    
    # Start with Qiskit-style circuit
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    
    # Can mix integer and register indexing
    qreg = QuantumRegister(3, 'q')
    qc.add_register(qreg)
    
    # Now can use register indexing
    # qc.x(('q', 2))  # Would work with full implementation
    
    print("Created circuit mixing API styles")


def register_based_example():
    """Example using register-based qubit addressing."""
    print("\n=== Register-Based Addressing ===")
    
    # Create multiple registers
    qreg1 = QuantumRegister(2, 'control')
    qreg2 = QuantumRegister(3, 'target')
    creg = ClassicalRegister(5, 'meas')
    
    # Create circuit with registers
    circuit = QuantumCircuit(qreg1.size + qreg2.size)
    circuit.add_register(qreg1)
    circuit.add_register(qreg2)
    
    # Access qubits through registers
    print(f"Control register size: {qreg1.size}")
    print(f"Target register size: {qreg2.size}")
    
    # Use register[index] notation
    control_qubit = qreg1[0]
    target_qubit = qreg2[1]
    
    print(f"Control qubit: {control_qubit.register}[{control_qubit.index}]")
    print(f"Target qubit: {target_qubit.register}[{target_qubit.index}]")


if __name__ == "__main__":
    qiskit_style_example()
    qiskit_style_advanced()
    cirq_style_example()
    cirq_style_grid_qubits()
    line_qubit_example()
    mixed_api_example()
    register_based_example()
    
    print("\n=== Pythonic API Demo Complete ===")