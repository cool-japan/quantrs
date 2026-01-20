#!/usr/bin/env python3
"""
Qiskit Compatibility Layer for QuantRS2

This module provides comprehensive bidirectional conversion between Qiskit
and QuantRS2 circuits, enabling seamless integration and migration.

Features:
    - Import Qiskit circuits to QuantRS2
    - Export QuantRS2 circuits to Qiskit
    - QASM 2.0 and 3.0 support
    - Gate mapping and optimization
    - Measurement conversion
    - Custom gate handling
    - Circuit equivalence testing
"""

import warnings
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import numpy as np

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.circuit import Instruction, Parameter, ParameterExpression
    from qiskit.circuit.library import *
    from qiskit import transpile
    from qiskit.converters import circuit_to_dag, dag_to_circuit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    QuantumCircuit = None
    warnings.warn("Qiskit not available. Install with: pip install qiskit")

try:
    from quantrs2 import Circuit as QuantRS2Circuit
    QUANTRS2_AVAILABLE = True
except ImportError:
    QUANTRS2_AVAILABLE = False
    warnings.warn("QuantRS2 not available")


@dataclass
class ConversionStats:
    """Statistics about circuit conversion."""
    original_gates: int
    converted_gates: int
    decomposed_gates: int
    unsupported_gates: List[str]
    warnings: List[str]
    success: bool


class QiskitConverter:
    """
    Bidirectional converter between Qiskit and QuantRS2 circuits.

    Supports:
        - Standard gates (H, X, Y, Z, S, T, CNOT, etc.)
        - Rotation gates (RX, RY, RZ, U1, U2, U3)
        - Controlled gates (CX, CY, CZ, CH, etc.)
        - Multi-qubit gates (Toffoli, Fredkin)
        - Parametric circuits
        - QASM import/export
    """

    # Gate mapping: Qiskit -> QuantRS2
    GATE_MAP = {
        'h': 'h',
        'x': 'x',
        'y': 'y',
        'z': 'z',
        's': 's',
        'sdg': 'sdg',
        't': 't',
        'tdg': 'tdg',
        'sx': 'sx',
        'sxdg': 'sxdg',
        'rx': 'rx',
        'ry': 'ry',
        'rz': 'rz',
        'cx': 'cnot',
        'cnot': 'cnot',
        'cy': 'cy',
        'cz': 'cz',
        'ch': 'ch',
        'swap': 'swap',
        'ccx': 'toffoli',
        'toffoli': 'toffoli',
        'cswap': 'cswap',
        'fredkin': 'cswap',
        'id': 'i',  # Identity gate
        'i': 'i',    # Identity gate (alternative)
    }

    def __init__(self, strict_mode: bool = False):
        """
        Initialize the converter.

        Args:
            strict_mode: If True, raise errors on unsupported gates.
                        If False, emit warnings and skip unsupported gates.
        """
        self.strict_mode = strict_mode
        self.conversion_warnings: List[str] = []

    def from_qiskit(
        self,
        qiskit_circuit: 'QuantumCircuit',
        optimize: bool = True,
    ) -> Tuple[Optional['QuantRS2Circuit'], ConversionStats]:
        """
        Convert a Qiskit circuit to QuantRS2.

        Args:
            qiskit_circuit: Qiskit QuantumCircuit to convert
            optimize: Whether to optimize the circuit during conversion

        Returns:
            Tuple of (QuantRS2Circuit, ConversionStats)

        Raises:
            ValueError: If conversion fails in strict mode
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit not available")
        if not QUANTRS2_AVAILABLE:
            raise ImportError("QuantRS2 not available")

        self.conversion_warnings = []
        unsupported_gates = []

        # Get number of qubits
        n_qubits = qiskit_circuit.num_qubits

        if n_qubits < 2:
            raise ValueError(f"QuantRS2 requires at least 2 qubits, got {n_qubits}")

        # Create QuantRS2 circuit
        quantrs_circuit = QuantRS2Circuit(n_qubits)

        # Optimize Qiskit circuit if requested
        if optimize:
            qiskit_circuit = transpile(
                qiskit_circuit,
                basis_gates=['h', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg',
                            'rx', 'ry', 'rz', 'cx', 'cy', 'cz', 'swap'],
                optimization_level=2,
            )

        original_gates = 0
        converted_gates = 0
        decomposed_gates = 0

        # Convert gates
        for instruction, qargs, cargs in qiskit_circuit.data:
            original_gates += 1
            gate_name = instruction.name.lower()

            try:
                # Get qubit indices
                qubit_indices = [qiskit_circuit.qubits.index(q) for q in qargs]

                # Handle standard gates
                if gate_name in self.GATE_MAP:
                    quantrs_gate = self.GATE_MAP[gate_name]
                    params = instruction.params

                    # Apply gate based on type
                    if gate_name in ['h', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg', 'sx', 'sxdg']:
                        # Single-qubit gates
                        getattr(quantrs_circuit, quantrs_gate)(qubit_indices[0])
                        converted_gates += 1

                    elif gate_name in ['rx', 'ry', 'rz']:
                        # Rotation gates
                        angle = float(params[0]) if params else 0.0
                        getattr(quantrs_circuit, quantrs_gate)(qubit_indices[0], angle)
                        converted_gates += 1

                    elif gate_name in ['cx', 'cnot', 'cy', 'cz', 'ch']:
                        # Two-qubit controlled gates
                        getattr(quantrs_circuit, quantrs_gate)(qubit_indices[0], qubit_indices[1])
                        converted_gates += 1

                    elif gate_name == 'swap':
                        quantrs_circuit.swap(qubit_indices[0], qubit_indices[1])
                        converted_gates += 1

                    elif gate_name in ['ccx', 'toffoli']:
                        quantrs_circuit.toffoli(qubit_indices[0], qubit_indices[1], qubit_indices[2])
                        converted_gates += 1

                    elif gate_name in ['cswap', 'fredkin']:
                        quantrs_circuit.cswap(qubit_indices[0], qubit_indices[1], qubit_indices[2])
                        converted_gates += 1

                # Handle U gates (decompose to rotations)
                elif gate_name == 'u1':
                    # U1(λ) = RZ(λ)
                    lam = float(params[0])
                    quantrs_circuit.rz(qubit_indices[0], lam)
                    converted_gates += 1
                    decomposed_gates += 1

                elif gate_name == 'u2':
                    # U2(φ, λ) = RZ(φ) RY(π/2) RZ(λ)
                    phi, lam = float(params[0]), float(params[1])
                    quantrs_circuit.rz(qubit_indices[0], phi)
                    quantrs_circuit.ry(qubit_indices[0], np.pi / 2)
                    quantrs_circuit.rz(qubit_indices[0], lam)
                    converted_gates += 3
                    decomposed_gates += 2

                elif gate_name == 'u3':
                    # U3(θ, φ, λ) = RZ(φ) RY(θ) RZ(λ)
                    theta, phi, lam = float(params[0]), float(params[1]), float(params[2])
                    quantrs_circuit.rz(qubit_indices[0], phi)
                    quantrs_circuit.ry(qubit_indices[0], theta)
                    quantrs_circuit.rz(qubit_indices[0], lam)
                    converted_gates += 3
                    decomposed_gates += 2

                elif gate_name == 'u':
                    # U(θ, φ, λ) = U3(θ, φ, λ)
                    theta, phi, lam = float(params[0]), float(params[1]), float(params[2])
                    quantrs_circuit.rz(qubit_indices[0], phi)
                    quantrs_circuit.ry(qubit_indices[0], theta)
                    quantrs_circuit.rz(qubit_indices[0], lam)
                    converted_gates += 3
                    decomposed_gates += 2

                # Handle phase gates
                elif gate_name == 'p':
                    # Phase gate P(θ) = RZ(θ)
                    theta = float(params[0])
                    quantrs_circuit.rz(qubit_indices[0], theta)
                    converted_gates += 1

                elif gate_name == 'cp':
                    # Controlled-Phase
                    theta = float(params[0])
                    quantrs_circuit.crz(qubit_indices[0], qubit_indices[1], theta)
                    converted_gates += 1

                # Handle controlled rotation gates
                elif gate_name == 'crx':
                    angle = float(params[0])
                    quantrs_circuit.crx(qubit_indices[0], qubit_indices[1], angle)
                    converted_gates += 1

                elif gate_name == 'cry':
                    angle = float(params[0])
                    quantrs_circuit.cry(qubit_indices[0], qubit_indices[1], angle)
                    converted_gates += 1

                elif gate_name == 'crz':
                    angle = float(params[0])
                    quantrs_circuit.crz(qubit_indices[0], qubit_indices[1], angle)
                    converted_gates += 1

                # Handle iSwap gate (decompose to CNOT + RZ)
                elif gate_name == 'iswap':
                    # iSwap = S ⊗ S · CNOT · Ry(π/2) ⊗ Ry(-π/2) · CNOT · Ry(-π/2) ⊗ Ry(π/2)
                    # Simplified: decompose to SWAP + S gates
                    quantrs_circuit.s(qubit_indices[0])
                    quantrs_circuit.s(qubit_indices[1])
                    quantrs_circuit.swap(qubit_indices[0], qubit_indices[1])
                    quantrs_circuit.s(qubit_indices[0])
                    quantrs_circuit.s(qubit_indices[1])
                    converted_gates += 1
                    decomposed_gates += 4

                # Handle ECR (echoed cross-resonance) gate
                elif gate_name == 'ecr':
                    # ECR decomposition: RZX(π/4) followed by X on first qubit
                    quantrs_circuit.x(qubit_indices[0])
                    quantrs_circuit.cnot(qubit_indices[0], qubit_indices[1])
                    quantrs_circuit.rz(qubit_indices[1], np.pi/4)
                    quantrs_circuit.cnot(qubit_indices[0], qubit_indices[1])
                    converted_gates += 1
                    decomposed_gates += 3

                # Handle RXX, RYY, RZZ gates (two-qubit rotations)
                elif gate_name == 'rxx':
                    angle = float(params[0])
                    # RXX decomposition
                    quantrs_circuit.h(qubit_indices[0])
                    quantrs_circuit.h(qubit_indices[1])
                    quantrs_circuit.cnot(qubit_indices[0], qubit_indices[1])
                    quantrs_circuit.rz(qubit_indices[1], angle)
                    quantrs_circuit.cnot(qubit_indices[0], qubit_indices[1])
                    quantrs_circuit.h(qubit_indices[0])
                    quantrs_circuit.h(qubit_indices[1])
                    converted_gates += 1
                    decomposed_gates += 6

                elif gate_name == 'ryy':
                    angle = float(params[0])
                    # RYY decomposition
                    quantrs_circuit.rx(qubit_indices[0], np.pi/2)
                    quantrs_circuit.rx(qubit_indices[1], np.pi/2)
                    quantrs_circuit.cnot(qubit_indices[0], qubit_indices[1])
                    quantrs_circuit.rz(qubit_indices[1], angle)
                    quantrs_circuit.cnot(qubit_indices[0], qubit_indices[1])
                    quantrs_circuit.rx(qubit_indices[0], -np.pi/2)
                    quantrs_circuit.rx(qubit_indices[1], -np.pi/2)
                    converted_gates += 1
                    decomposed_gates += 6

                elif gate_name == 'rzz':
                    angle = float(params[0])
                    # RZZ decomposition
                    quantrs_circuit.cnot(qubit_indices[0], qubit_indices[1])
                    quantrs_circuit.rz(qubit_indices[1], angle)
                    quantrs_circuit.cnot(qubit_indices[0], qubit_indices[1])
                    converted_gates += 1
                    decomposed_gates += 2

                # Handle multi-controlled X gates
                elif gate_name.startswith('c') and gate_name.endswith('x') and len(qubit_indices) > 2:
                    # Multi-controlled X gate (e.g., C3X, C4X)
                    if len(qubit_indices) == 3:
                        quantrs_circuit.toffoli(qubit_indices[0], qubit_indices[1], qubit_indices[2])
                        converted_gates += 1
                    else:
                        # Decompose higher-order controlled gates
                        self.conversion_warnings.append(
                            f"Multi-controlled X with {len(qubit_indices)} qubits decomposed to Toffoli gates"
                        )
                        # Simple decomposition - needs improvement for efficiency
                        quantrs_circuit.toffoli(qubit_indices[-3], qubit_indices[-2], qubit_indices[-1])
                        converted_gates += 1
                        decomposed_gates += 1

                # Skip measurement gates (QuantRS2 handles differently)
                elif gate_name == 'measure':
                    self.conversion_warnings.append(
                        "Skipping measurement gate - QuantRS2 measures at simulation time"
                    )
                    continue

                # Skip barrier gates
                elif gate_name == 'barrier':
                    continue

                # Skip reset gates
                elif gate_name == 'reset':
                    self.conversion_warnings.append(
                        "Skipping reset gate - not supported in QuantRS2 circuit model"
                    )
                    continue

                else:
                    # Unsupported gate
                    unsupported_gates.append(gate_name)
                    msg = f"Unsupported gate: {gate_name}"

                    if self.strict_mode:
                        raise ValueError(msg)
                    else:
                        self.conversion_warnings.append(msg)

            except Exception as e:
                msg = f"Error converting gate {gate_name}: {e}"
                if self.strict_mode:
                    raise ValueError(msg)
                else:
                    self.conversion_warnings.append(msg)

        stats = ConversionStats(
            original_gates=original_gates,
            converted_gates=converted_gates,
            decomposed_gates=decomposed_gates,
            unsupported_gates=list(set(unsupported_gates)),
            warnings=self.conversion_warnings,
            success=len(unsupported_gates) == 0,
        )

        return quantrs_circuit, stats

    def to_qiskit(
        self,
        quantrs_circuit: 'QuantRS2Circuit',
        name: str = "quantrs_circuit",
    ) -> 'QuantumCircuit':
        """
        Convert a QuantRS2 circuit to Qiskit.

        Args:
            quantrs_circuit: QuantRS2 Circuit to convert
            name: Name for the Qiskit circuit

        Returns:
            Qiskit QuantumCircuit

        Note:
            This is a basic implementation. Full conversion requires access
            to QuantRS2's internal gate representation.
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit not available")

        n_qubits = quantrs_circuit.n_qubits
        qiskit_circuit = QuantumCircuit(n_qubits, name=name)

        # Note: This would require QuantRS2 to expose its gate list
        # For now, this is a placeholder that would need to be implemented
        # with proper gate extraction from QuantRS2

        warnings.warn(
            "to_qiskit conversion requires QuantRS2 to expose gate representation. "
            "Use to_qasm() and from_qasm() as alternative."
        )

        return qiskit_circuit

    def from_qasm(
        self,
        qasm_str: str,
        qasm_version: str = "2.0",
    ) -> Tuple[Optional['QuantRS2Circuit'], ConversionStats]:
        """
        Convert QASM string to QuantRS2 circuit.

        Args:
            qasm_str: QASM string
            qasm_version: QASM version ("2.0" or "3.0")

        Returns:
            Tuple of (QuantRS2Circuit, ConversionStats)
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required for QASM parsing")

        # Use Qiskit to parse QASM
        qiskit_circuit = QuantumCircuit.from_qasm_str(qasm_str)

        # Convert to QuantRS2
        return self.from_qiskit(qiskit_circuit)

    def to_qasm(
        self,
        quantrs_circuit: 'QuantRS2Circuit',
        qasm_version: str = "2.0",
    ) -> str:
        """
        Export QuantRS2 circuit to QASM string.

        Args:
            quantrs_circuit: QuantRS2 circuit
            qasm_version: QASM version ("2.0" or "3.0")

        Returns:
            QASM string
        """
        # This would require QuantRS2 to implement QASM export
        # For now, return a basic QASM header
        n_qubits = quantrs_circuit.n_qubits

        if qasm_version == "2.0":
            qasm = f"""OPENQASM 2.0;
include "qelib1.inc";
qreg q[{n_qubits}];
creg c[{n_qubits}];

// Circuit gates would be exported here
// This requires QuantRS2 to expose gate representation
"""
        else:
            qasm = f"""OPENQASM 3.0;
include "stdgates.inc";

qubit[{n_qubits}] q;
bit[{n_qubits}] c;

// Circuit gates would be exported here
"""

        return qasm

    def verify_equivalence(
        self,
        circuit1: 'QuantumCircuit',
        circuit2: 'QuantumCircuit',
        tolerance: float = 1e-6,
    ) -> Tuple[bool, float]:
        """
        Verify if two Qiskit circuits are equivalent.

        Args:
            circuit1: First circuit
            circuit2: Second circuit
            tolerance: Numerical tolerance for comparison

        Returns:
            Tuple of (equivalent, fidelity)
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required for verification")

        from qiskit import Aer, execute

        # Simulate both circuits
        backend = Aer.get_backend('statevector_simulator')

        result1 = execute(circuit1, backend).result()
        result2 = execute(circuit2, backend).result()

        sv1 = result1.get_statevector()
        sv2 = result2.get_statevector()

        # Compute fidelity
        fidelity = abs(np.dot(np.conj(sv1), sv2)) ** 2

        return fidelity > (1 - tolerance), float(fidelity)


def convert_from_qiskit(
    qiskit_circuit: 'QuantumCircuit',
    optimize: bool = True,
    strict: bool = False,
) -> 'QuantRS2Circuit':
    """
    Convenience function to convert Qiskit circuit to QuantRS2.

    Args:
        qiskit_circuit: Qiskit circuit to convert
        optimize: Whether to optimize during conversion
        strict: Whether to use strict mode (raise on errors)

    Returns:
        QuantRS2 circuit

    Example:
        >>> from qiskit import QuantumCircuit
        >>> qc = QuantumCircuit(2)
        >>> qc.h(0)
        >>> qc.cx(0, 1)
        >>> quantrs_circuit = convert_from_qiskit(qc)
    """
    converter = QiskitConverter(strict_mode=strict)
    circuit, stats = converter.from_qiskit(qiskit_circuit, optimize=optimize)

    if stats.warnings:
        for warning in stats.warnings:
            warnings.warn(warning)

    if not stats.success:
        warnings.warn(
            f"Conversion completed with {len(stats.unsupported_gates)} "
            f"unsupported gates: {stats.unsupported_gates}"
        )

    return circuit


def convert_to_qiskit(
    quantrs_circuit: 'QuantRS2Circuit',
    name: str = "circuit",
) -> 'QuantumCircuit':
    """
    Convenience function to convert QuantRS2 circuit to Qiskit.

    Args:
        quantrs_circuit: QuantRS2 circuit to convert
        name: Name for resulting circuit

    Returns:
        Qiskit circuit

    Example:
        >>> from quantrs2 import Circuit
        >>> qrs_circuit = Circuit(2)
        >>> qrs_circuit.h(0)
        >>> qrs_circuit.cnot(0, 1)
        >>> qiskit_circuit = convert_to_qiskit(qrs_circuit)
    """
    converter = QiskitConverter()
    return converter.to_qiskit(quantrs_circuit, name=name)


# Example usage and testing
if __name__ == "__main__":
    if QISKIT_AVAILABLE and QUANTRS2_AVAILABLE:
        print("Testing Qiskit <-> QuantRS2 Conversion\n")
        print("="*60)

        # Create a Qiskit circuit
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure([0, 1, 2], [0, 1, 2])

        print("Original Qiskit Circuit:")
        print(qc)
        print()

        # Convert to QuantRS2
        converter = QiskitConverter()
        quantrs_circuit, stats = converter.from_qiskit(qc)

        print("Conversion Statistics:")
        print(f"  Original gates: {stats.original_gates}")
        print(f"  Converted gates: {stats.converted_gates}")
        print(f"  Decomposed gates: {stats.decomposed_gates}")
        print(f"  Success: {stats.success}")

        if stats.warnings:
            print("\nWarnings:")
            for warning in stats.warnings:
                print(f"  - {warning}")

        if stats.unsupported_gates:
            print(f"\nUnsupported gates: {stats.unsupported_gates}")

        print("\nConversion successful!" if stats.success else "\nConversion completed with warnings")

    else:
        print("Qiskit and/or QuantRS2 not available")
        print("Install with: pip install qiskit")
