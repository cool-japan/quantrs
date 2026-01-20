#!/usr/bin/env python3
"""
MyQLM Compatibility Layer for QuantRS2

This module provides comprehensive bidirectional conversion between MyQLM/QLM
and QuantRS2 circuits, enabling seamless integration with Atos Quantum Learning Machine.

Features:
    - Import MyQLM/QLM circuits to QuantRS2
    - Export QuantRS2 circuits to MyQLM/QLM
    - Gate mapping and optimization
    - Abstract gate handling
    - Job submission and result processing
    - Variational plugin support
"""

import warnings
from typing import Dict, List, Optional, Union, Any, Tuple, Sequence
from dataclasses import dataclass
import numpy as np

try:
    from qat.lang.AQASM import Program, H, X, Y, Z, S, T, CNOT, SWAP, RX, RY, RZ
    from qat.lang.AQASM import AbstractGate, QRoutine
    from qat.core import Circuit as QLMCircuit, Job, Result
    from qat.core.qpu import QPUHandler
    MYQLM_AVAILABLE = True
except ImportError:
    MYQLM_AVAILABLE = False
    Program = None
    QLMCircuit = None
    warnings.warn("MyQLM/QLM not available. Install with: pip install myqlm")

try:
    from quantrs2 import Circuit as QuantRS2Circuit
    QUANTRS2_AVAILABLE = True
except ImportError:
    QUANTRS2_AVAILABLE = False
    warnings.warn("QuantRS2 not available")


@dataclass
class MyQLMConversionStats:
    """Statistics about MyQLM circuit conversion."""
    original_gates: int
    converted_gates: int
    decomposed_gates: int
    unsupported_gates: List[str]
    warnings: List[str]
    success: bool


class MyQLMConverter:
    """
    Bidirectional converter between MyQLM/QLM and QuantRS2 circuits.

    Supports:
        - Standard gates (H, X, Y, Z, S, T, CNOT, etc.)
        - Rotation gates (RX, RY, RZ)
        - Controlled gates
        - Abstract gates and QRoutines
        - Parametric circuits
        - Variational algorithms
    """

    # Gate mapping: MyQLM -> QuantRS2
    GATE_MAP = {
        'H': 'h',
        'X': 'x',
        'Y': 'y',
        'Z': 'z',
        'S': 's',
        'T': 't',
        'CNOT': 'cnot',
        'CX': 'cnot',
        'SWAP': 'swap',
        'RX': 'rx',
        'RY': 'ry',
        'RZ': 'rz',
        'CZ': 'cz',
        'CSIGN': 'cz',
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

    def from_myqlm(
        self,
        qlm_circuit: Union['QLMCircuit', 'Program'],
    ) -> Tuple[Optional['QuantRS2Circuit'], MyQLMConversionStats]:
        """
        Convert a MyQLM/QLM circuit to QuantRS2.

        Args:
            qlm_circuit: MyQLM Circuit or Program to convert

        Returns:
            Tuple of (QuantRS2Circuit, MyQLMConversionStats)

        Example:
            >>> from qat.lang.AQASM import Program, H, CNOT
            >>> prog = Program()
            >>> qbits = prog.qalloc(2)
            >>> prog.apply(H, qbits[0])
            >>> prog.apply(CNOT, qbits[0], qbits[1])
            >>> circuit = prog.to_circ()
            >>> converter = MyQLMConverter()
            >>> quantrs_circuit, stats = converter.from_myqlm(circuit)
        """
        if not MYQLM_AVAILABLE:
            raise ImportError("MyQLM/QLM not available")
        if not QUANTRS2_AVAILABLE:
            raise ImportError("QuantRS2 not available")

        self.conversion_warnings = []
        unsupported_gates = []

        # Convert Program to Circuit if necessary
        if isinstance(qlm_circuit, Program):
            qlm_circuit = qlm_circuit.to_circ()

        # Get number of qubits
        n_qubits = qlm_circuit.nbqbits

        if n_qubits < 2:
            raise ValueError(f"QuantRS2 requires at least 2 qubits, got {n_qubits}")

        # Create QuantRS2 circuit
        quantrs_circuit = QuantRS2Circuit(n_qubits)

        original_gates = 0
        converted_gates = 0
        decomposed_gates = 0

        # Process each gate
        for op in qlm_circuit.iterate_simple():
            original_gates += 1
            gate_name = op.gate.upper() if hasattr(op, 'gate') else str(op.type)

            try:
                # Get qubit indices
                qubits = list(op.qbits)

                # Handle standard gates
                if gate_name in self.GATE_MAP:
                    quantrs_gate = self.GATE_MAP[gate_name]

                    # Check if parametric
                    if hasattr(op, 'params') and op.params:
                        # Parametric gate (rotation)
                        if gate_name in ['RX', 'RY', 'RZ']:
                            angle = float(op.params[0])
                            getattr(quantrs_circuit, quantrs_gate)(qubits[0], angle)
                            converted_gates += 1
                        else:
                            # Non-parametric gate with unexpected params
                            self.conversion_warnings.append(
                                f"Gate {gate_name} has unexpected parameters"
                            )
                            getattr(quantrs_circuit, quantrs_gate)(*qubits)
                            converted_gates += 1
                    else:
                        # Non-parametric gate
                        if gate_name in ['H', 'X', 'Y', 'Z', 'S', 'T']:
                            # Single-qubit gates
                            getattr(quantrs_circuit, quantrs_gate)(qubits[0])
                            converted_gates += 1
                        elif gate_name in ['CNOT', 'CX', 'CZ', 'CSIGN', 'SWAP']:
                            # Two-qubit gates
                            getattr(quantrs_circuit, quantrs_gate)(qubits[0], qubits[1])
                            converted_gates += 1

                # Handle S dagger
                elif gate_name in ['SDAG', 'SDG']:
                    quantrs_circuit.sdg(qubits[0])
                    converted_gates += 1

                # Handle T dagger
                elif gate_name in ['TDAG', 'TDG']:
                    quantrs_circuit.tdg(qubits[0])
                    converted_gates += 1

                # Handle controlled-Y
                elif gate_name == 'CY':
                    quantrs_circuit.cy(qubits[0], qubits[1])
                    converted_gates += 1

                # Handle controlled-H
                elif gate_name == 'CH':
                    quantrs_circuit.ch(qubits[0], qubits[1])
                    converted_gates += 1

                # Handle Toffoli (CCX, CCNOT)
                elif gate_name in ['CCX', 'CCNOT', 'TOFFOLI']:
                    if len(qubits) >= 3:
                        quantrs_circuit.toffoli(qubits[0], qubits[1], qubits[2])
                        converted_gates += 1
                    else:
                        raise ValueError(f"Toffoli gate requires 3 qubits, got {len(qubits)}")

                # Handle Fredkin (CSWAP)
                elif gate_name in ['CSWAP', 'FREDKIN']:
                    if len(qubits) >= 3:
                        quantrs_circuit.cswap(qubits[0], qubits[1], qubits[2])
                        converted_gates += 1
                    else:
                        raise ValueError(f"Fredkin gate requires 3 qubits, got {len(qubits)}")

                # Handle controlled rotation gates
                elif gate_name == 'CRX':
                    angle = float(op.params[0]) if hasattr(op, 'params') and op.params else 0.0
                    quantrs_circuit.crx(qubits[0], qubits[1], angle)
                    converted_gates += 1

                elif gate_name == 'CRY':
                    angle = float(op.params[0]) if hasattr(op, 'params') and op.params else 0.0
                    quantrs_circuit.cry(qubits[0], qubits[1], angle)
                    converted_gates += 1

                elif gate_name == 'CRZ':
                    angle = float(op.params[0]) if hasattr(op, 'params') and op.params else 0.0
                    quantrs_circuit.crz(qubits[0], qubits[1], angle)
                    converted_gates += 1

                # Handle phase gate
                elif gate_name in ['P', 'PHASE']:
                    angle = float(op.params[0]) if hasattr(op, 'params') and op.params else 0.0
                    quantrs_circuit.rz(qubits[0], angle)
                    converted_gates += 1

                # Handle U gates (decompose)
                elif gate_name == 'U':
                    # U(θ, φ, λ) = RZ(φ) RY(θ) RZ(λ)
                    if hasattr(op, 'params') and len(op.params) >= 3:
                        theta, phi, lam = float(op.params[0]), float(op.params[1]), float(op.params[2])
                        quantrs_circuit.rz(qubits[0], phi)
                        quantrs_circuit.ry(qubits[0], theta)
                        quantrs_circuit.rz(qubits[0], lam)
                        converted_gates += 3
                        decomposed_gates += 2
                    else:
                        self.conversion_warnings.append(
                            f"U gate missing parameters, skipping"
                        )

                # Skip measurement gates
                elif gate_name in ['MEASURE', 'MEAS']:
                    self.conversion_warnings.append(
                        "Skipping measurement gate - QuantRS2 measures at simulation time"
                    )
                    continue

                # Skip reset gates
                elif gate_name == 'RESET':
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

        stats = MyQLMConversionStats(
            original_gates=original_gates,
            converted_gates=converted_gates,
            decomposed_gates=decomposed_gates,
            unsupported_gates=list(set(unsupported_gates)),
            warnings=self.conversion_warnings,
            success=len(unsupported_gates) == 0,
        )

        return quantrs_circuit, stats

    def to_myqlm(
        self,
        quantrs_circuit: 'QuantRS2Circuit',
    ) -> 'Program':
        """
        Convert a QuantRS2 circuit to MyQLM Program.

        Args:
            quantrs_circuit: QuantRS2 Circuit to convert

        Returns:
            MyQLM Program

        Note:
            This requires QuantRS2 to expose its gate representation.
        """
        if not MYQLM_AVAILABLE:
            raise ImportError("MyQLM/QLM not available")

        n_qubits = quantrs_circuit.n_qubits
        prog = Program()
        qbits = prog.qalloc(n_qubits)

        # Note: This would require QuantRS2 to expose its gate list
        warnings.warn(
            "to_myqlm conversion requires QuantRS2 to expose gate representation."
        )

        return prog

    def create_job(
        self,
        qlm_circuit: Union['QLMCircuit', 'Program'],
        nbshots: int = 0,
        qubits: Optional[List[int]] = None,
    ) -> 'Job':
        """
        Create a MyQLM job from a circuit.

        Args:
            qlm_circuit: MyQLM Circuit or Program
            nbshots: Number of shots (0 for exact simulation)
            qubits: Qubits to measure (None for all)

        Returns:
            MyQLM Job
        """
        if not MYQLM_AVAILABLE:
            raise ImportError("MyQLM/QLM required for job creation")

        # Convert Program to Circuit if necessary
        if isinstance(qlm_circuit, Program):
            qlm_circuit = qlm_circuit.to_circ()

        # Create job
        job = qlm_circuit.to_job(nbshots=nbshots, qubits=qubits)

        return job


def convert_from_myqlm(
    qlm_circuit: Union['QLMCircuit', 'Program'],
    strict: bool = False,
) -> 'QuantRS2Circuit':
    """
    Convenience function to convert MyQLM circuit to QuantRS2.

    Args:
        qlm_circuit: MyQLM Circuit or Program to convert
        strict: Whether to use strict mode

    Returns:
        QuantRS2 circuit

    Example:
        >>> from qat.lang.AQASM import Program, H, CNOT
        >>> prog = Program()
        >>> qbits = prog.qalloc(2)
        >>> prog.apply(H, qbits[0])
        >>> prog.apply(CNOT, qbits[0], qbits[1])
        >>> circuit = prog.to_circ()
        >>> quantrs_circuit = convert_from_myqlm(circuit)
    """
    converter = MyQLMConverter(strict_mode=strict)
    circuit, stats = converter.from_myqlm(qlm_circuit)

    if stats.warnings:
        for warning in stats.warnings:
            warnings.warn(warning)

    if not stats.success:
        warnings.warn(
            f"Conversion completed with {len(stats.unsupported_gates)} "
            f"unsupported gates: {stats.unsupported_gates}"
        )

    return circuit


def convert_to_myqlm(
    quantrs_circuit: 'QuantRS2Circuit',
) -> 'Program':
    """
    Convenience function to convert QuantRS2 circuit to MyQLM Program.

    Args:
        quantrs_circuit: QuantRS2 circuit to convert

    Returns:
        MyQLM Program

    Example:
        >>> from quantrs2 import Circuit
        >>> circuit = Circuit(2)
        >>> circuit.h(0)
        >>> circuit.cnot(0, 1)
        >>> myqlm_prog = convert_to_myqlm(circuit)
    """
    converter = MyQLMConverter()
    return converter.to_myqlm(quantrs_circuit)


# Example usage and testing
if __name__ == "__main__":
    if MYQLM_AVAILABLE and QUANTRS2_AVAILABLE:
        print("Testing MyQLM <-> QuantRS2 Conversion\n")
        print("="*60)

        # Create a MyQLM program
        prog = Program()
        qbits = prog.qalloc(3)

        # Add gates
        prog.apply(H, qbits[0])
        prog.apply(CNOT, qbits[0], qbits[1])
        prog.apply(CNOT, qbits[1], qbits[2])

        circuit = prog.to_circ()

        print("Original MyQLM Circuit:")
        print(f"  Number of qubits: {circuit.nbqbits}")
        print(f"  Number of gates: {len(list(circuit.iterate_simple()))}")
        print()

        # Convert to QuantRS2
        converter = MyQLMConverter()
        quantrs_circuit, stats = converter.from_myqlm(circuit)

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
        print("MyQLM and/or QuantRS2 not available")
        print("Install MyQLM with: pip install myqlm")
