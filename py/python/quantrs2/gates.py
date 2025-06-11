"""
Quantum gate operations for QuantRS2.

This module provides comprehensive support for quantum gates including:
- Standard single-qubit gates (X, Y, Z, H, S, T, etc.)
- Parameterized rotation gates (RX, RY, RZ)
- Multi-qubit gates (CNOT, CZ, SWAP, Toffoli, etc.)
- Controlled gates
- Custom gate creation from matrices
- Symbolic parameters for variational algorithms
"""

from typing import Union, List, Optional, Dict, Tuple
import numpy as np

try:
    from quantrs2._quantrs2 import gates as _gates
    HAS_NATIVE_GATES = True
except ImportError:
    HAS_NATIVE_GATES = False

if HAS_NATIVE_GATES:
    # Re-export base classes
    Gate = _gates.Gate
    GateParameter = _gates.GateParameter
    ParametricGateBase = _gates.ParametricGateBase

    # Standard single-qubit gates
    class H(_gates.HadamardGate):
        """Hadamard gate - creates superposition.
        
        Matrix: 1/√2 * [[1, 1], [1, -1]]
        
        Args:
            qubit: Index of the qubit to apply the gate to
            
        Example:
            >>> gate = H(0)  # Apply Hadamard to qubit 0
        """
        pass

    class X(_gates.PauliXGate):
        """Pauli-X gate (bit flip).
        
        Matrix: [[0, 1], [1, 0]]
        
        Args:
            qubit: Index of the qubit to apply the gate to
            
        Example:
            >>> gate = X(0)  # Apply X gate to qubit 0
        """
        pass

    class Y(_gates.PauliYGate):
        """Pauli-Y gate.
        
        Matrix: [[0, -i], [i, 0]]
        
        Args:
            qubit: Index of the qubit to apply the gate to
            
        Example:
            >>> gate = Y(0)  # Apply Y gate to qubit 0
        """
        pass

    class Z(_gates.PauliZGate):
        """Pauli-Z gate (phase flip).
        
        Matrix: [[1, 0], [0, -1]]
        
        Args:
            qubit: Index of the qubit to apply the gate to
            
        Example:
            >>> gate = Z(0)  # Apply Z gate to qubit 0
        """
        pass

    class S(_gates.SGate):
        """S gate (quarter-turn phase gate).
        
        Matrix: [[1, 0], [0, i]]
        
        Args:
            qubit: Index of the qubit to apply the gate to
            
        Example:
            >>> gate = S(0)  # Apply S gate to qubit 0
        """
        pass

    class T(_gates.TGate):
        """T gate (eighth-turn phase gate).
        
        Matrix: [[1, 0], [0, e^(iπ/4)]]
        
        Args:
            qubit: Index of the qubit to apply the gate to
            
        Example:
            >>> gate = T(0)  # Apply T gate to qubit 0
        """
        pass

    # Rotation gates
    class RX(_gates.RXGate):
        """Rotation around X-axis.
        
        Matrix: [[cos(θ/2), -i*sin(θ/2)], [-i*sin(θ/2), cos(θ/2)]]
        
        Args:
            qubit: Index of the qubit to apply the gate to
            angle: Rotation angle in radians
            
        Example:
            >>> gate = RX(0, np.pi/2)  # π/2 rotation around X
        """
        pass

    class RY(_gates.RYGate):
        """Rotation around Y-axis.
        
        Matrix: [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
        
        Args:
            qubit: Index of the qubit to apply the gate to
            angle: Rotation angle in radians
            
        Example:
            >>> gate = RY(0, np.pi/2)  # π/2 rotation around Y
        """
        pass

    class RZ(_gates.RZGate):
        """Rotation around Z-axis.
        
        Matrix: [[e^(-iθ/2), 0], [0, e^(iθ/2)]]
        
        Args:
            qubit: Index of the qubit to apply the gate to
            angle: Rotation angle in radians
            
        Example:
            >>> gate = RZ(0, np.pi/2)  # π/2 rotation around Z
        """
        pass

    # Two-qubit gates
    class CNOT(_gates.CNOTGate):
        """Controlled-NOT gate.
        
        Matrix: [[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]]
        
        Args:
            control: Index of the control qubit
            target: Index of the target qubit
            
        Example:
            >>> gate = CNOT(0, 1)  # Control on 0, target on 1
        """
        pass

    class CZ(_gates.CZGate):
        """Controlled-Z gate.
        
        Matrix: [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,-1]]
        
        Args:
            control: Index of the control qubit
            target: Index of the target qubit
            
        Example:
            >>> gate = CZ(0, 1)  # Control on 0, target on 1
        """
        pass

    class SWAP(_gates.SWAPGate):
        """SWAP gate - exchanges two qubits.
        
        Matrix: [[1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1]]
        
        Args:
            qubit1: Index of the first qubit
            qubit2: Index of the second qubit
            
        Example:
            >>> gate = SWAP(0, 1)  # Swap qubits 0 and 1
        """
        pass

    # Three-qubit gates
    class Toffoli(_gates.ToffoliGate):
        """Toffoli gate (CCNOT) - controlled-controlled-NOT.
        
        Args:
            control1: Index of the first control qubit
            control2: Index of the second control qubit
            target: Index of the target qubit
            
        Example:
            >>> gate = Toffoli(0, 1, 2)  # Controls on 0,1, target on 2
        """
        pass

    class Fredkin(_gates.FredkinGate):
        """Fredkin gate (CSWAP) - controlled-SWAP.
        
        Args:
            control: Index of the control qubit
            target1: Index of the first target qubit
            target2: Index of the second target qubit
            
        Example:
            >>> gate = Fredkin(0, 1, 2)  # Control on 0, swap 1 and 2
        """
        pass

    # Parametric gates
    class ParametricGate(ParametricGateBase):
        """Base class for parametric gates with symbolic parameters.
        
        Args:
            name: Name of the gate
            qubits: List of qubit indices
            parameters: List of parameters (can be symbolic)
            
        Example:
            >>> from quantrs2.gates import GateParameter
            >>> theta = GateParameter("theta", np.pi/4)
            >>> gate = ParametricGate("RY", [0], [theta])
        """
        
        def __init__(self, name: str, qubits: List[int], parameters: List[GateParameter]):
            super().__init__(name)
            self.qubits = qubits
            self.parameters = parameters

    # Measurement gate
    class Measure(_gates.MeasureGate):
        """Measurement gate.
        
        Args:
            qubit: Index of the qubit to measure
            classical_bit: Index of the classical bit to store the result
            
        Example:
            >>> gate = Measure(0, 0)  # Measure qubit 0, store in bit 0
        """
        pass

    # Custom gate creation utilities
    def custom_gate(matrix: np.ndarray, qubits: List[int], name: str = "Custom") -> Gate:
        """Create a custom gate from a unitary matrix.
        
        Args:
            matrix: Unitary matrix representing the gate
            qubits: List of qubit indices the gate acts on
            name: Optional name for the gate
            
        Returns:
            Gate object that can be added to circuits
            
        Example:
            >>> # Create a custom rotation gate
            >>> angle = np.pi/3
            >>> matrix = np.array([[np.cos(angle/2), -1j*np.sin(angle/2)],
            ...                   [-1j*np.sin(angle/2), np.cos(angle/2)]])
            >>> gate = custom_gate(matrix, [0], "CustomRX")
        """
        return _gates.custom_gate(matrix, qubits, name)

    def controlled_gate(base_gate: Gate, control_qubits: List[int]) -> Gate:
        """Create a controlled version of any gate.
        
        Args:
            base_gate: The base gate to control
            control_qubits: List of control qubit indices
            
        Returns:
            Controlled gate object
            
        Example:
            >>> # Create a controlled-Y gate
            >>> y_gate = Y(1)  # Target qubit 1
            >>> cy_gate = controlled_gate(y_gate, [0])  # Control on qubit 0
        """
        return _gates.controlled_gate(base_gate, control_qubits)

    # Gate utilities
    def dagger(gate: Gate) -> Gate:
        """Get the Hermitian conjugate (dagger) of a gate.
        
        Args:
            gate: Input gate
            
        Returns:
            Hermitian conjugate of the input gate
            
        Example:
            >>> h_gate = H(0)
            >>> h_dagger = dagger(h_gate)  # H† = H for Hadamard
        """
        return _gates.dagger(gate)

    def gate_power(gate: Gate, power: float) -> Gate:
        """Raise a gate to a given power.
        
        Args:
            gate: Input gate
            power: Power to raise the gate to
            
        Returns:
            Gate raised to the specified power
            
        Example:
            >>> x_gate = X(0)
            >>> sqrt_x = gate_power(x_gate, 0.5)  # √X gate
        """
        return _gates.gate_power(gate, power)

    __all__ = [
        "Gate", "GateParameter", "ParametricGateBase", "ParametricGate",
        "H", "X", "Y", "Z", "S", "T",
        "RX", "RY", "RZ", 
        "CNOT", "CZ", "SWAP",
        "Toffoli", "Fredkin",
        "Measure",
        "custom_gate", "controlled_gate", "dagger", "gate_power"
    ]

else:
    # Fallback implementations when native gates are not available
    class Gate:
        """Fallback gate implementation."""
        def __init__(self, name: str):
            self.name = name
            
    class GateParameter:
        """Fallback gate parameter implementation."""
        def __init__(self, name: str, value=None):
            self.name = name
            self.value = value
            
    class ParametricGateBase:
        """Fallback parametric gate base implementation."""
        def __init__(self, name: str):
            self.name = name

    # Minimal gate implementations for testing
    class H(Gate):
        def __init__(self, qubit: int):
            super().__init__("H")
            self.qubit = qubit
            
    class X(Gate):
        def __init__(self, qubit: int):
            super().__init__("X")
            self.qubit = qubit
            
    class Y(Gate):
        def __init__(self, qubit: int):
            super().__init__("Y")
            self.qubit = qubit
            
    class Z(Gate):
        def __init__(self, qubit: int):
            super().__init__("Z")
            self.qubit = qubit
            
    class CNOT(Gate):
        def __init__(self, control: int, target: int):
            super().__init__("CNOT")
            self.control = control
            self.target = target

    def custom_gate(matrix, qubits, name="Custom"):
        return Gate(name)
        
    def controlled_gate(base_gate, control_qubits):
        return Gate(f"C{base_gate.name}")
        
    def dagger(gate):
        return Gate(f"{gate.name}†")
        
    def gate_power(gate, power):
        return Gate(f"{gate.name}^{power}")

    __all__ = [
        "Gate", "GateParameter", "ParametricGateBase",
        "H", "X", "Y", "Z", "CNOT",
        "custom_gate", "controlled_gate", "dagger", "gate_power"
    ]