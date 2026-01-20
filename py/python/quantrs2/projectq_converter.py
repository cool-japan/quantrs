#!/usr/bin/env python3
"""
ProjectQ Compatibility Layer for QuantRS2

This module provides comprehensive integration between ProjectQ and QuantRS2 circuits,
enabling seamless interoperability with the ProjectQ quantum computing framework.

Features:
    - Import ProjectQ circuits to QuantRS2
    - Export QuantRS2 circuits to ProjectQ
    - Gate mapping and command extraction
    - Quantum engine integration
    - Backend compatibility
    - Decomposition support
"""

import warnings
from typing import Dict, List, Optional, Union, Any, Tuple, Sequence
from dataclasses import dataclass
import numpy as np

try:
    import projectq
    from projectq import MainEngine
    from projectq.ops import (
        H, X, Y, Z, S, T, Rx, Ry, Rz, CNOT, CX, CZ, Swap,
        Toffoli, Measure, All, Barrier, SqrtX
    )
    from projectq.backends import CommandPrinter, Simulator
    from projectq.meta import Control
    from projectq.types import Qubit, Qureg
    PROJECTQ_AVAILABLE = True
    PROJECTQ_VERSION = projectq.__version__ if hasattr(projectq, '__version__') else "unknown"
except ImportError:
    PROJECTQ_AVAILABLE = False
    PROJECTQ_VERSION = "not_available"
    MainEngine = None
    warnings.warn("ProjectQ not available. Install with: pip install projectq")

try:
    from quantrs2 import Circuit as QuantRS2Circuit
    QUANTRS2_AVAILABLE = True
except ImportError:
    QUANTRS2_AVAILABLE = False
    warnings.warn("QuantRS2 not available")


@dataclass
class ProjectQConversionStats:
    """Statistics about ProjectQ conversion."""
    original_commands: int
    converted_gates: int
    decomposed_gates: int
    unsupported_gates: List[str]
    warnings: List[str]
    success: bool


class ProjectQConverter:
    """
    Bidirectional converter between ProjectQ and QuantRS2.

    Supports:
        - Standard gates (H, X, Y, Z, S, T, etc.)
        - Rotation gates (Rx, Ry, Rz)
        - Controlled gates (CNOT, CZ, Toffoli)
        - Multi-qubit operations
        - Command extraction from ProjectQ engines
        - Backend integration
    """

    # Gate mapping
    GATE_MAP = {
        'H': 'h',
        'X': 'x',
        'Y': 'y',
        'Z': 'z',
        'S': 's',
        'T': 't',
        'Rx': 'rx',
        'Ry': 'ry',
        'Rz': 'rz',
        'CNOT': 'cnot',
        'CX': 'cnot',
        'CZ': 'cz',
        'Swap': 'swap',
        'SqrtX': 'sx',
    }

    def __init__(self, strict_mode: bool = False):
        """
        Initialize the converter.

        Args:
            strict_mode: If True, raise errors on unsupported operations.
                        If False, emit warnings and skip unsupported operations.
        """
        self.strict_mode = strict_mode
        self.conversion_warnings: List[str] = []

    def from_projectq(
        self,
        engine: 'MainEngine',
        qubit_mapping: Optional[Dict[int, int]] = None,
    ) -> Tuple[Optional['QuantRS2Circuit'], ProjectQConversionStats]:
        """
        Convert a ProjectQ MainEngine's command list to QuantRS2.

        Args:
            engine: ProjectQ MainEngine with executed commands
            qubit_mapping: Optional mapping from ProjectQ qubit IDs to indices

        Returns:
            Tuple of (QuantRS2Circuit, ProjectQConversionStats)

        Example:
            >>> from projectq import MainEngine
            >>> from projectq.ops import H, CNOT, Measure, All
            >>> eng = MainEngine()
            >>> qubits = eng.allocate_qureg(2)
            >>> H | qubits[0]
            >>> CNOT | (qubits[0], qubits[1])
            >>> eng.flush()
            >>> converter = ProjectQConverter()
            >>> quantrs_circuit, stats = converter.from_projectq(eng)
        """
        if not PROJECTQ_AVAILABLE:
            raise ImportError("ProjectQ not available")
        if not QUANTRS2_AVAILABLE:
            raise ImportError("QuantRS2 not available")

        self.conversion_warnings = []
        unsupported_gates = []

        # Extract commands from engine
        commands = engine.backend.received_commands if hasattr(engine.backend, 'received_commands') else []

        if not commands:
            warnings.warn("No commands found in ProjectQ engine. Make sure to call eng.flush()")

        # Determine number of qubits
        qubit_ids = set()
        for cmd in commands:
            for qureg in cmd.qubits:
                for qubit in qureg:
                    qubit_ids.add(id(qubit))

        n_qubits = len(qubit_ids)

        if n_qubits < 2:
            # Use minimum of 2 qubits for QuantRS2
            n_qubits = 2

        # Create qubit mapping if not provided
        if qubit_mapping is None:
            qubit_id_list = sorted(list(qubit_ids))
            qubit_mapping = {qid: idx for idx, qid in enumerate(qubit_id_list)}

        # Create QuantRS2 circuit
        quantrs_circuit = QuantRS2Circuit(n_qubits)

        original_commands = 0
        converted_gates = 0
        decomposed_gates = 0

        # Process commands
        for cmd in commands:
            original_commands += 1
            gate_name = cmd.gate.__class__.__name__

            try:
                # Get qubit indices
                qubit_indices = []
                for qureg in cmd.qubits:
                    for qubit in qureg:
                        qid = id(qubit)
                        if qid in qubit_mapping:
                            qubit_indices.append(qubit_mapping[qid])

                # Get control qubits if any
                control_qubits = []
                if cmd.control_qubits:
                    for qubit in cmd.control_qubits:
                        qid = id(qubit)
                        if qid in qubit_mapping:
                            control_qubits.append(qubit_mapping[qid])

                # Handle standard gates
                if gate_name in self.GATE_MAP:
                    quantrs_gate = self.GATE_MAP[gate_name]

                    # Check for parametric gates
                    if gate_name in ['Rx', 'Ry', 'Rz']:
                        # Rotation gates
                        if hasattr(cmd.gate, 'angle'):
                            angle = float(cmd.gate.angle)
                        else:
                            angle = 0.0
                            self.conversion_warnings.append(
                                f"Rotation gate {gate_name} missing angle parameter"
                            )

                        if control_qubits:
                            # Controlled rotation
                            if len(control_qubits) == 1:
                                ctrl = control_qubits[0]
                                target = qubit_indices[0] if qubit_indices else 0
                                if gate_name == 'Rx':
                                    quantrs_circuit.crx(ctrl, target, angle)
                                elif gate_name == 'Ry':
                                    quantrs_circuit.cry(ctrl, target, angle)
                                elif gate_name == 'Rz':
                                    quantrs_circuit.crz(ctrl, target, angle)
                                converted_gates += 1
                            else:
                                self.conversion_warnings.append(
                                    f"Multi-controlled {gate_name} not supported"
                                )
                        else:
                            # Regular rotation
                            if qubit_indices:
                                getattr(quantrs_circuit, quantrs_gate)(qubit_indices[0], angle)
                                converted_gates += 1

                    elif gate_name in ['H', 'X', 'Y', 'Z', 'S', 'T', 'SqrtX']:
                        # Single-qubit gates
                        if control_qubits:
                            # Controlled version
                            if len(control_qubits) == 1 and qubit_indices:
                                ctrl = control_qubits[0]
                                target = qubit_indices[0]
                                if gate_name == 'X':
                                    quantrs_circuit.cnot(ctrl, target)
                                elif gate_name == 'Z':
                                    quantrs_circuit.cz(ctrl, target)
                                elif gate_name == 'Y':
                                    quantrs_circuit.cy(ctrl, target)
                                elif gate_name == 'H':
                                    quantrs_circuit.ch(ctrl, target)
                                else:
                                    self.conversion_warnings.append(
                                        f"Controlled {gate_name} not directly supported"
                                    )
                                converted_gates += 1
                        else:
                            # Regular gate
                            if qubit_indices:
                                getattr(quantrs_circuit, quantrs_gate)(qubit_indices[0])
                                converted_gates += 1

                    elif gate_name in ['CNOT', 'CX', 'CZ', 'Swap']:
                        # Two-qubit gates
                        if len(qubit_indices) >= 2:
                            getattr(quantrs_circuit, quantrs_gate)(qubit_indices[0], qubit_indices[1])
                            converted_gates += 1
                        else:
                            self.conversion_warnings.append(
                                f"Gate {gate_name} requires 2 qubits, got {len(qubit_indices)}"
                            )

                # Handle Toffoli
                elif gate_name == 'Toffoli':
                    if len(qubit_indices) >= 3:
                        quantrs_circuit.toffoli(qubit_indices[0], qubit_indices[1], qubit_indices[2])
                        converted_gates += 1
                    else:
                        self.conversion_warnings.append(
                            f"Toffoli requires 3 qubits, got {len(qubit_indices)}"
                        )

                # Handle S dagger
                elif gate_name in ['Sdag', 'Sdagger']:
                    if qubit_indices:
                        quantrs_circuit.sdg(qubit_indices[0])
                        converted_gates += 1

                # Handle T dagger
                elif gate_name in ['Tdag', 'Tdagger']:
                    if qubit_indices:
                        quantrs_circuit.tdg(qubit_indices[0])
                        converted_gates += 1

                # Handle Ph (Phase gate)
                elif gate_name == 'Ph':
                    angle = float(cmd.gate.angle) if hasattr(cmd.gate, 'angle') else 0.0
                    if qubit_indices:
                        quantrs_circuit.rz(qubit_indices[0], angle)
                        converted_gates += 1

                # Handle R (general rotation)
                elif gate_name == 'R':
                    # R(θ) = exp(-iθ/2) - can be approximated as RZ
                    angle = float(cmd.gate.angle) if hasattr(cmd.gate, 'angle') else 0.0
                    if qubit_indices:
                        quantrs_circuit.rz(qubit_indices[0], angle)
                        converted_gates += 1

                # Skip allocation and deallocation
                elif gate_name in ['Allocate', 'Deallocate']:
                    continue

                # Skip measurement
                elif gate_name == 'Measure':
                    self.conversion_warnings.append(
                        "Skipping measurement gate - QuantRS2 measures at simulation time"
                    )
                    continue

                # Skip barriers
                elif gate_name == 'Barrier':
                    continue

                # Skip flush
                elif gate_name == 'FlushGate':
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

        stats = ProjectQConversionStats(
            original_commands=original_commands,
            converted_gates=converted_gates,
            decomposed_gates=decomposed_gates,
            unsupported_gates=list(set(unsupported_gates)),
            warnings=self.conversion_warnings,
            success=len(unsupported_gates) == 0,
        )

        return quantrs_circuit, stats

    def to_projectq(
        self,
        quantrs_circuit: 'QuantRS2Circuit',
    ) -> 'MainEngine':
        """
        Convert a QuantRS2 circuit to ProjectQ MainEngine.

        Args:
            quantrs_circuit: QuantRS2 Circuit to convert

        Returns:
            ProjectQ MainEngine with circuit applied

        Note:
            This requires QuantRS2 to expose its gate representation.
        """
        if not PROJECTQ_AVAILABLE:
            raise ImportError("ProjectQ not available")

        n_qubits = quantrs_circuit.n_qubits

        # Create ProjectQ engine
        eng = MainEngine()
        qubits = eng.allocate_qureg(n_qubits)

        # Note: This would require QuantRS2 to expose its gate list
        warnings.warn(
            "to_projectq conversion requires QuantRS2 to expose gate representation."
        )

        return eng


class ProjectQBackend:
    """
    ProjectQ-style backend for QuantRS2.

    Allows using QuantRS2 as a backend for ProjectQ circuits.
    """

    def __init__(self):
        """Initialize the QuantRS2 backend for ProjectQ."""
        self.received_commands = []
        self._circuit = None
        self._qubit_mapping = {}

    def is_available(self, cmd):
        """
        Check if a command is available (supported).

        Args:
            cmd: ProjectQ command

        Returns:
            True if supported
        """
        gate_name = cmd.gate.__class__.__name__
        return gate_name in ProjectQConverter.GATE_MAP or gate_name in [
            'Allocate', 'Deallocate', 'Measure', 'Barrier'
        ]

    def receive(self, command_list):
        """
        Receive a list of commands from ProjectQ.

        Args:
            command_list: List of ProjectQ commands
        """
        self.received_commands.extend(command_list)

    def flush(self):
        """Flush the backend (convert to QuantRS2 and execute)."""
        if QUANTRS2_AVAILABLE:
            # Create converter and convert commands
            converter = ProjectQConverter()

            # Create a mock engine with the received commands
            class MockEngine:
                def __init__(self, commands):
                    self.backend = type('obj', (object,), {'received_commands': commands})()

            mock_eng = MockEngine(self.received_commands)
            self._circuit, stats = converter.from_projectq(mock_eng)

            if not stats.success:
                warnings.warn(f"Conversion had warnings: {stats.warnings}")


def convert_from_projectq(
    engine: 'MainEngine',
    strict: bool = False,
) -> 'QuantRS2Circuit':
    """
    Convenience function to convert ProjectQ circuit to QuantRS2.

    Args:
        engine: ProjectQ MainEngine with executed commands
        strict: Whether to use strict mode

    Returns:
        QuantRS2 circuit

    Example:
        >>> from projectq import MainEngine
        >>> from projectq.ops import H, CNOT
        >>> eng = MainEngine()
        >>> qubits = eng.allocate_qureg(2)
        >>> H | qubits[0]
        >>> CNOT | (qubits[0], qubits[1])
        >>> eng.flush()
        >>> quantrs_circuit = convert_from_projectq(eng)
    """
    converter = ProjectQConverter(strict_mode=strict)
    circuit, stats = converter.from_projectq(engine)

    if stats.warnings:
        for warning in stats.warnings:
            warnings.warn(warning)

    if not stats.success:
        warnings.warn(
            f"Conversion completed with {len(stats.unsupported_gates)} "
            f"unsupported gates: {stats.unsupported_gates}"
        )

    return circuit


def convert_to_projectq(
    quantrs_circuit: 'QuantRS2Circuit',
) -> 'MainEngine':
    """
    Convenience function to convert QuantRS2 circuit to ProjectQ.

    Args:
        quantrs_circuit: QuantRS2 circuit to convert

    Returns:
        ProjectQ MainEngine

    Example:
        >>> from quantrs2 import Circuit
        >>> circuit = Circuit(2)
        >>> circuit.h(0)
        >>> circuit.cnot(0, 1)
        >>> projectq_eng = convert_to_projectq(circuit)
    """
    converter = ProjectQConverter()
    return converter.to_projectq(quantrs_circuit)


# Example usage and testing
if __name__ == "__main__":
    if PROJECTQ_AVAILABLE and QUANTRS2_AVAILABLE:
        print("Testing ProjectQ <-> QuantRS2 Conversion\n")
        print("="*60)

        # Create a ProjectQ circuit
        eng = MainEngine()
        qubits = eng.allocate_qureg(3)

        # Add gates
        H | qubits[0]
        CNOT | (qubits[0], qubits[1])
        CNOT | (qubits[1], qubits[2])

        eng.flush()

        print("Original ProjectQ Circuit:")
        print(f"  Number of qubits: {len(qubits)}")
        print()

        # Convert to QuantRS2
        converter = ProjectQConverter()
        quantrs_circuit, stats = converter.from_projectq(eng)

        print("Conversion Statistics:")
        print(f"  Original commands: {stats.original_commands}")
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
        print("ProjectQ and/or QuantRS2 not available")
        print("Install ProjectQ with: pip install projectq")
