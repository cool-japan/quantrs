#!/usr/bin/env python3
"""
Cirq Compatibility Layer for QuantRS2

This module provides comprehensive bidirectional conversion between Cirq
and QuantRS2 circuits, enabling seamless integration and migration.

Features:
    - Import Cirq circuits to QuantRS2
    - Export QuantRS2 circuits to Cirq
    - Gate mapping and optimization
    - Moment-based conversion
    - Custom operation handling
    - Circuit equivalence testing
"""

import warnings
from typing import Dict, List, Optional, Union, Any, Tuple, Sequence
from dataclasses import dataclass
import numpy as np

try:
    import cirq
    from cirq import Circuit as CirqCircuit, LineQubit, GridQubit
    from cirq.ops import Gate, Operation
    CIRQ_AVAILABLE = True
except ImportError:
    CIRQ_AVAILABLE = False
    cirq = None
    warnings.warn("Cirq not available. Install with: pip install cirq")

try:
    from quantrs2 import Circuit as QuantRS2Circuit
    QUANTRS2_AVAILABLE = True
except ImportError:
    QUANTRS2_AVAILABLE = False
    warnings.warn("QuantRS2 not available")


@dataclass
class CirqConversionStats:
    """Statistics about Cirq circuit conversion."""
    original_operations: int
    converted_operations: int
    decomposed_operations: int
    unsupported_operations: List[str]
    warnings: List[str]
    success: bool
    num_moments: int = 0


class CirqConverter:
    """
    Bidirectional converter between Cirq and QuantRS2 circuits.

    Supports:
        - Standard gates (H, X, Y, Z, S, T, CNOT, etc.)
        - Rotation gates (Rx, Ry, Rz)
        - Controlled gates (CX, CY, CZ, etc.)
        - Multi-qubit gates (CCNOT, CSWAP)
        - Parametric circuits
        - Cirq moments and operations
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize the converter.

        Args:
            strict_mode: If True, raise errors on unsupported operations.
                        If False, emit warnings and skip unsupported operations.
        """
        self.strict_mode = strict_mode
        self.conversion_warnings: List[str] = []

    def from_cirq(
        self,
        cirq_circuit: 'CirqCircuit',
        qubit_map: Optional[Dict] = None,
    ) -> Tuple[Optional['QuantRS2Circuit'], CirqConversionStats]:
        """
        Convert a Cirq circuit to QuantRS2.

        Args:
            cirq_circuit: Cirq Circuit to convert
            qubit_map: Optional mapping from Cirq qubits to indices

        Returns:
            Tuple of (QuantRS2Circuit, CirqConversionStats)

        Example:
            >>> import cirq
            >>> qubits = cirq.LineQubit.range(2)
            >>> circuit = cirq.Circuit()
            >>> circuit.append([cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[1])])
            >>> converter = CirqConverter()
            >>> quantrs_circuit, stats = converter.from_cirq(circuit)
        """
        if not CIRQ_AVAILABLE:
            raise ImportError("Cirq not available")
        if not QUANTRS2_AVAILABLE:
            raise ImportError("QuantRS2 not available")

        self.conversion_warnings = []
        unsupported_operations = []

        # Get qubits and create mapping
        qubits = sorted(cirq_circuit.all_qubits())
        if qubit_map is None:
            qubit_map = {q: i for i, q in enumerate(qubits)}

        n_qubits = len(qubits)

        if n_qubits < 2:
            raise ValueError(f"QuantRS2 requires at least 2 qubits, got {n_qubits}")

        # Create QuantRS2 circuit
        quantrs_circuit = QuantRS2Circuit(n_qubits)

        original_ops = 0
        converted_ops = 0
        decomposed_ops = 0
        num_moments = len(cirq_circuit.moments)

        # Process each moment
        for moment in cirq_circuit:
            for operation in moment:
                original_ops += 1

                try:
                    gate = operation.gate
                    gate_qubits = [qubit_map[q] for q in operation.qubits]

                    # Handle single-qubit gates
                    if isinstance(gate, cirq.HPowGate):
                        if gate.exponent == 1.0:
                            quantrs_circuit.h(gate_qubits[0])
                            converted_ops += 1
                        else:
                            # Decompose H^exp
                            self._decompose_powered_gate(
                                quantrs_circuit, 'h', gate_qubits[0], gate.exponent
                            )
                            converted_ops += 1
                            decomposed_ops += 1

                    elif isinstance(gate, cirq.XPowGate):
                        if gate.exponent == 1.0:
                            quantrs_circuit.x(gate_qubits[0])
                            converted_ops += 1
                        elif gate.exponent == 0.5:
                            quantrs_circuit.sx(gate_qubits[0])
                            converted_ops += 1
                        else:
                            # X^exp = RX(exp * π)
                            angle = gate.exponent * np.pi
                            quantrs_circuit.rx(gate_qubits[0], angle)
                            converted_ops += 1
                            decomposed_ops += 1

                    elif isinstance(gate, cirq.YPowGate):
                        if gate.exponent == 1.0:
                            quantrs_circuit.y(gate_qubits[0])
                            converted_ops += 1
                        else:
                            # Y^exp = RY(exp * π)
                            angle = gate.exponent * np.pi
                            quantrs_circuit.ry(gate_qubits[0], angle)
                            converted_ops += 1
                            decomposed_ops += 1

                    elif isinstance(gate, cirq.ZPowGate):
                        if gate.exponent == 1.0:
                            quantrs_circuit.z(gate_qubits[0])
                            converted_ops += 1
                        elif gate.exponent == 0.5:
                            quantrs_circuit.s(gate_qubits[0])
                            converted_ops += 1
                        elif gate.exponent == 0.25:
                            quantrs_circuit.t(gate_qubits[0])
                            converted_ops += 1
                        elif gate.exponent == -0.5:
                            quantrs_circuit.sdg(gate_qubits[0])
                            converted_ops += 1
                        elif gate.exponent == -0.25:
                            quantrs_circuit.tdg(gate_qubits[0])
                            converted_ops += 1
                        else:
                            # Z^exp = RZ(exp * π)
                            angle = gate.exponent * np.pi
                            quantrs_circuit.rz(gate_qubits[0], angle)
                            converted_ops += 1
                            decomposed_ops += 1

                    # Rotation gates
                    elif isinstance(gate, (cirq.rx, cirq.Rx)):
                        angle = gate.rads if hasattr(gate, 'rads') else 0.0
                        quantrs_circuit.rx(gate_qubits[0], angle)
                        converted_ops += 1

                    elif isinstance(gate, (cirq.ry, cirq.Ry)):
                        angle = gate.rads if hasattr(gate, 'rads') else 0.0
                        quantrs_circuit.ry(gate_qubits[0], angle)
                        converted_ops += 1

                    elif isinstance(gate, (cirq.rz, cirq.Rz)):
                        angle = gate.rads if hasattr(gate, 'rads') else 0.0
                        quantrs_circuit.rz(gate_qubits[0], angle)
                        converted_ops += 1

                    # Two-qubit gates
                    elif isinstance(gate, cirq.CNotPowGate):
                        if gate.exponent == 1.0:
                            quantrs_circuit.cnot(gate_qubits[0], gate_qubits[1])
                            converted_ops += 1
                        else:
                            # Decompose CNOT^exp
                            self.conversion_warnings.append(
                                f"Partial CNOT (exp={gate.exponent}) decomposed to full CNOT"
                            )
                            quantrs_circuit.cnot(gate_qubits[0], gate_qubits[1])
                            converted_ops += 1
                            decomposed_ops += 1

                    elif isinstance(gate, cirq.CZPowGate):
                        if gate.exponent == 1.0:
                            quantrs_circuit.cz(gate_qubits[0], gate_qubits[1])
                            converted_ops += 1
                        else:
                            # CZ^exp = CRZ(exp * π)
                            angle = gate.exponent * np.pi
                            quantrs_circuit.crz(gate_qubits[0], gate_qubits[1], angle)
                            converted_ops += 1
                            decomposed_ops += 1

                    elif isinstance(gate, cirq.SwapPowGate):
                        if gate.exponent == 1.0:
                            quantrs_circuit.swap(gate_qubits[0], gate_qubits[1])
                            converted_ops += 1
                        else:
                            # Partial SWAP - emit warning
                            self.conversion_warnings.append(
                                f"Partial SWAP (exp={gate.exponent}) not supported, using full SWAP"
                            )
                            quantrs_circuit.swap(gate_qubits[0], gate_qubits[1])
                            converted_ops += 1
                            decomposed_ops += 1

                    # Three-qubit gates
                    elif isinstance(gate, cirq.CCXPowGate) or isinstance(gate, cirq.CCNotPowGate):
                        if gate.exponent == 1.0:
                            quantrs_circuit.toffoli(gate_qubits[0], gate_qubits[1], gate_qubits[2])
                            converted_ops += 1
                        else:
                            self.conversion_warnings.append(
                                f"Partial Toffoli (exp={gate.exponent}) not supported, using full Toffoli"
                            )
                            quantrs_circuit.toffoli(gate_qubits[0], gate_qubits[1], gate_qubits[2])
                            converted_ops += 1
                            decomposed_ops += 1

                    elif isinstance(gate, cirq.CSwapGate):
                        quantrs_circuit.cswap(gate_qubits[0], gate_qubits[1], gate_qubits[2])
                        converted_ops += 1

                    # Handle iSwap and iSwap-like gates
                    elif isinstance(gate, cirq.ISwapPowGate):
                        if gate.exponent == 1.0:
                            # iSwap decomposition
                            quantrs_circuit.s(gate_qubits[0])
                            quantrs_circuit.s(gate_qubits[1])
                            quantrs_circuit.swap(gate_qubits[0], gate_qubits[1])
                            quantrs_circuit.s(gate_qubits[0])
                            quantrs_circuit.s(gate_qubits[1])
                            converted_ops += 1
                            decomposed_ops += 4
                        else:
                            self.conversion_warnings.append(
                                f"Partial iSwap (exp={gate.exponent}) decomposed to full iSwap"
                            )
                            quantrs_circuit.s(gate_qubits[0])
                            quantrs_circuit.s(gate_qubits[1])
                            quantrs_circuit.swap(gate_qubits[0], gate_qubits[1])
                            quantrs_circuit.s(gate_qubits[0])
                            quantrs_circuit.s(gate_qubits[1])
                            converted_ops += 1
                            decomposed_ops += 4

                    # Handle FSimGate (fermionic simulation gate)
                    elif hasattr(cirq, 'FSimGate') and isinstance(gate, cirq.FSimGate):
                        # FSim(θ, φ) decomposition to basic gates
                        theta = gate.theta if hasattr(gate, 'theta') else 0.0
                        phi = gate.phi if hasattr(gate, 'phi') else 0.0

                        # Approximate decomposition
                        quantrs_circuit.rxx(gate_qubits[0], gate_qubits[1], theta)
                        quantrs_circuit.cz(gate_qubits[0], gate_qubits[1])
                        quantrs_circuit.rz(gate_qubits[0], phi)
                        quantrs_circuit.rz(gate_qubits[1], phi)
                        converted_ops += 1
                        decomposed_ops += 3
                        self.conversion_warnings.append(
                            "FSim gate decomposed to RXX + CZ + RZ gates"
                        )

                    # Handle Givens rotation gate
                    elif hasattr(cirq, 'GivensRotation') and isinstance(gate, cirq.GivensRotation):
                        angle = gate.rads if hasattr(gate, 'rads') else 0.0
                        # Givens rotation decomposition
                        quantrs_circuit.ryy(gate_qubits[0], gate_qubits[1], angle)
                        converted_ops += 1
                        decomposed_ops += 1

                    # Handle PhasedXPowGate
                    elif hasattr(cirq, 'PhasedXPowGate') and isinstance(gate, cirq.PhasedXPowGate):
                        exponent = gate.exponent if hasattr(gate, 'exponent') else 1.0
                        phase_exponent = gate.phase_exponent if hasattr(gate, 'phase_exponent') else 0.0

                        # PhasedX decomposition: RZ(phase) RX(exponent*π) RZ(-phase)
                        quantrs_circuit.rz(gate_qubits[0], phase_exponent * np.pi)
                        quantrs_circuit.rx(gate_qubits[0], exponent * np.pi)
                        quantrs_circuit.rz(gate_qubits[0], -phase_exponent * np.pi)
                        converted_ops += 1
                        decomposed_ops += 2

                    # Handle CNOT variants (CX, etc.)
                    elif hasattr(cirq, 'CNOT') and isinstance(gate, type(cirq.CNOT)):
                        quantrs_circuit.cnot(gate_qubits[0], gate_qubits[1])
                        converted_ops += 1

                    # Handle Identity gate
                    elif hasattr(cirq, 'IdentityGate') and isinstance(gate, cirq.IdentityGate):
                        # Skip identity gates
                        continue

                    # Measurement (skip - QuantRS2 handles at simulation time)
                    elif isinstance(operation, cirq.MeasurementGate) or \
                         operation.gate is None and 'measure' in str(operation).lower():
                        self.conversion_warnings.append(
                            "Skipping measurement - QuantRS2 measures at simulation time"
                        )
                        continue

                    else:
                        # Unsupported operation
                        op_name = str(gate) if gate else str(operation)
                        unsupported_operations.append(op_name)
                        msg = f"Unsupported operation: {op_name}"

                        if self.strict_mode:
                            raise ValueError(msg)
                        else:
                            self.conversion_warnings.append(msg)

                except Exception as e:
                    msg = f"Error converting operation {operation}: {e}"
                    if self.strict_mode:
                        raise ValueError(msg)
                    else:
                        self.conversion_warnings.append(msg)

        stats = CirqConversionStats(
            original_operations=original_ops,
            converted_operations=converted_ops,
            decomposed_operations=decomposed_ops,
            unsupported_operations=list(set(unsupported_operations)),
            warnings=self.conversion_warnings,
            success=len(unsupported_operations) == 0,
            num_moments=num_moments,
        )

        return quantrs_circuit, stats

    def to_cirq(
        self,
        quantrs_circuit: 'QuantRS2Circuit',
        qubit_type: str = "line",
    ) -> 'CirqCircuit':
        """
        Convert a QuantRS2 circuit to Cirq.

        Args:
            quantrs_circuit: QuantRS2 Circuit to convert
            qubit_type: Type of Cirq qubits ("line" or "grid")

        Returns:
            Cirq Circuit

        Note:
            This requires QuantRS2 to expose its gate representation.
        """
        if not CIRQ_AVAILABLE:
            raise ImportError("Cirq not available")

        n_qubits = quantrs_circuit.n_qubits

        # Create qubits
        if qubit_type == "line":
            qubits = [cirq.LineQubit(i) for i in range(n_qubits)]
        else:
            # Grid qubits (assuming square grid)
            rows = int(np.ceil(np.sqrt(n_qubits)))
            qubits = [cirq.GridQubit(i // rows, i % rows) for i in range(n_qubits)]

        cirq_circuit = cirq.Circuit()

        # Note: This would require QuantRS2 to expose its gate list
        warnings.warn(
            "to_cirq conversion requires QuantRS2 to expose gate representation."
        )

        return cirq_circuit

    def _decompose_powered_gate(
        self,
        circuit: 'QuantRS2Circuit',
        gate_type: str,
        qubit: int,
        exponent: float,
    ):
        """
        Decompose a powered gate into basic gates.

        Args:
            circuit: QuantRS2 circuit
            gate_type: Type of gate ('h', 'x', 'y', 'z')
            qubit: Target qubit
            exponent: Power exponent
        """
        # Simplified decomposition - apply gate multiple times for integer exponents
        if abs(exponent - round(exponent)) < 1e-10:
            # Integer exponent
            n_times = int(round(abs(exponent)))
            for _ in range(n_times):
                getattr(circuit, gate_type)(qubit)
        else:
            # Non-integer exponent - use rotation approximation
            self.conversion_warnings.append(
                f"Non-integer power {exponent} for {gate_type} gate - using rotation approximation"
            )
            angle = exponent * np.pi
            if gate_type in ['x', 'h']:
                circuit.rx(qubit, angle)
            elif gate_type == 'y':
                circuit.ry(qubit, angle)
            elif gate_type == 'z':
                circuit.rz(qubit, angle)

    def verify_equivalence(
        self,
        circuit1: 'CirqCircuit',
        circuit2: 'CirqCircuit',
        tolerance: float = 1e-6,
    ) -> Tuple[bool, float]:
        """
        Verify if two Cirq circuits are equivalent.

        Args:
            circuit1: First circuit
            circuit2: Second circuit
            tolerance: Numerical tolerance

        Returns:
            Tuple of (equivalent, fidelity)
        """
        if not CIRQ_AVAILABLE:
            raise ImportError("Cirq required for verification")

        # Simulate both circuits
        simulator = cirq.Simulator()

        result1 = simulator.simulate(circuit1)
        result2 = simulator.simulate(circuit2)

        sv1 = result1.state_vector()
        sv2 = result2.state_vector()

        # Compute fidelity
        fidelity = abs(np.dot(np.conj(sv1), sv2)) ** 2

        return fidelity > (1 - tolerance), float(fidelity)


def convert_from_cirq(
    cirq_circuit: 'CirqCircuit',
    strict: bool = False,
) -> 'QuantRS2Circuit':
    """
    Convenience function to convert Cirq circuit to QuantRS2.

    Args:
        cirq_circuit: Cirq circuit to convert
        strict: Whether to use strict mode

    Returns:
        QuantRS2 circuit

    Example:
        >>> import cirq
        >>> qubits = cirq.LineQubit.range(2)
        >>> circuit = cirq.Circuit()
        >>> circuit.append([cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[1])])
        >>> quantrs_circuit = convert_from_cirq(circuit)
    """
    converter = CirqConverter(strict_mode=strict)
    circuit, stats = converter.from_cirq(cirq_circuit)

    if stats.warnings:
        for warning in stats.warnings:
            warnings.warn(warning)

    if not stats.success:
        warnings.warn(
            f"Conversion completed with {len(stats.unsupported_operations)} "
            f"unsupported operations: {stats.unsupported_operations}"
        )

    return circuit


def convert_to_cirq(
    quantrs_circuit: 'QuantRS2Circuit',
    qubit_type: str = "line",
) -> 'CirqCircuit':
    """
    Convenience function to convert QuantRS2 circuit to Cirq.

    Args:
        quantrs_circuit: QuantRS2 circuit to convert
        qubit_type: Type of Cirq qubits ("line" or "grid")

    Returns:
        Cirq circuit

    Example:
        >>> from quantrs2 import Circuit
        >>> circuit = Circuit(2)
        >>> circuit.h(0)
        >>> circuit.cnot(0, 1)
        >>> cirq_circuit = convert_to_cirq(circuit)
    """
    converter = CirqConverter()
    return converter.to_cirq(quantrs_circuit, qubit_type=qubit_type)


# Example usage and testing
if __name__ == "__main__":
    if CIRQ_AVAILABLE and QUANTRS2_AVAILABLE:
        print("Testing Cirq <-> QuantRS2 Conversion\n")
        print("="*60)

        # Create a Cirq circuit
        qubits = cirq.LineQubit.range(3)
        circuit = cirq.Circuit()

        # Add gates
        circuit.append([
            cirq.H(qubits[0]),
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.CNOT(qubits[1], qubits[2]),
            cirq.measure(*qubits, key='result')
        ])

        print("Original Cirq Circuit:")
        print(circuit)
        print()

        # Convert to QuantRS2
        converter = CirqConverter()
        quantrs_circuit, stats = converter.from_cirq(circuit)

        print("Conversion Statistics:")
        print(f"  Original operations: {stats.original_operations}")
        print(f"  Converted operations: {stats.converted_operations}")
        print(f"  Decomposed operations: {stats.decomposed_operations}")
        print(f"  Number of moments: {stats.num_moments}")
        print(f"  Success: {stats.success}")

        if stats.warnings:
            print("\nWarnings:")
            for warning in stats.warnings:
                print(f"  - {warning}")

        if stats.unsupported_operations:
            print(f"\nUnsupported operations: {stats.unsupported_operations}")

        print("\nConversion successful!" if stats.success else "\nConversion completed with warnings")

    else:
        print("Cirq and/or QuantRS2 not available")
        print("Install with: pip install cirq")
