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
from quantrs2._quantrs2 import gates as _gates

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
        >>> gate = H(0)  # Hadamard on qubit 0
        >>> matrix = gate.matrix()  # Get the gate matrix
    """
    pass

class X(_gates.PauliXGate):
    """Pauli-X (NOT) gate - bit flip.
    
    Matrix: [[0, 1], [1, 0]]
    
    Args:
        qubit: Index of the qubit to apply the gate to
    """
    pass

class Y(_gates.PauliYGate):
    """Pauli-Y gate.
    
    Matrix: [[0, -i], [i, 0]]
    
    Args:
        qubit: Index of the qubit to apply the gate to
    """
    pass

class Z(_gates.PauliZGate):
    """Pauli-Z gate - phase flip.
    
    Matrix: [[1, 0], [0, -1]]
    
    Args:
        qubit: Index of the qubit to apply the gate to
    """
    pass

class S(_gates.SGate):
    """S (phase) gate - π/2 phase.
    
    Matrix: [[1, 0], [0, i]]
    
    Args:
        qubit: Index of the qubit to apply the gate to
    """
    pass

class SDagger(_gates.SDaggerGate):
    """S-dagger gate - conjugate of S gate.
    
    Matrix: [[1, 0], [0, -i]]
    
    Args:
        qubit: Index of the qubit to apply the gate to
    """
    pass

class T(_gates.TGate):
    """T gate - π/4 phase.
    
    Matrix: [[1, 0], [0, exp(iπ/4)]]
    
    Args:
        qubit: Index of the qubit to apply the gate to
    """
    pass

class TDagger(_gates.TDaggerGate):
    """T-dagger gate - conjugate of T gate.
    
    Matrix: [[1, 0], [0, exp(-iπ/4)]]
    
    Args:
        qubit: Index of the qubit to apply the gate to
    """
    pass

class SX(_gates.SqrtXGate):
    """Square root of X gate.
    
    Matrix: [[0.5+0.5i, 0.5-0.5i], [0.5-0.5i, 0.5+0.5i]]
    
    Args:
        qubit: Index of the qubit to apply the gate to
    """
    pass

class SXDagger(_gates.SqrtXDaggerGate):
    """Square root of X dagger gate.
    
    Matrix: [[0.5-0.5i, 0.5+0.5i], [0.5+0.5i, 0.5-0.5i]]
    
    Args:
        qubit: Index of the qubit to apply the gate to
    """
    pass

# Rotation gates
class RX(_gates.RXGate):
    """Rotation around X-axis.
    
    Matrix: [[cos(θ/2), -i*sin(θ/2)], [-i*sin(θ/2), cos(θ/2)]]
    
    Args:
        qubit: Index of the qubit to apply the gate to
        theta: Rotation angle in radians
        
    Example:
        >>> gate = RX(0, np.pi/2)  # π/2 rotation on qubit 0
    """
    pass

class RY(_gates.RYGate):
    """Rotation around Y-axis.
    
    Matrix: [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
    
    Args:
        qubit: Index of the qubit to apply the gate to
        theta: Rotation angle in radians
    """
    pass

class RZ(_gates.RZGate):
    """Rotation around Z-axis.
    
    Matrix: [[exp(-iθ/2), 0], [0, exp(iθ/2)]]
    
    Args:
        qubit: Index of the qubit to apply the gate to
        theta: Rotation angle in radians
    """
    pass

# Two-qubit gates
class CNOT(_gates.CNOTGate):
    """Controlled-NOT (CX) gate.
    
    Flips the target qubit if control is |1⟩.
    
    Args:
        control: Index of the control qubit
        target: Index of the target qubit
        
    Example:
        >>> gate = CNOT(0, 1)  # Control on qubit 0, target on qubit 1
    """
    pass

class CY(_gates.CYGate):
    """Controlled-Y gate.
    
    Applies Y gate to target if control is |1⟩.
    
    Args:
        control: Index of the control qubit
        target: Index of the target qubit
    """
    pass

class CZ(_gates.CZGate):
    """Controlled-Z gate.
    
    Applies Z gate to target if control is |1⟩.
    
    Args:
        control: Index of the control qubit
        target: Index of the target qubit
    """
    pass

class CH(_gates.CHGate):
    """Controlled-Hadamard gate.
    
    Applies H gate to target if control is |1⟩.
    
    Args:
        control: Index of the control qubit
        target: Index of the target qubit
    """
    pass

class CS(_gates.CSGate):
    """Controlled-S gate.
    
    Applies S gate to target if control is |1⟩.
    
    Args:
        control: Index of the control qubit
        target: Index of the target qubit
    """
    pass

class SWAP(_gates.SWAPGate):
    """SWAP gate - exchanges two qubits.
    
    Args:
        qubit1: Index of the first qubit
        qubit2: Index of the second qubit
    """
    pass

class CRX(_gates.CRXGate):
    """Controlled rotation around X-axis.
    
    Args:
        control: Index of the control qubit
        target: Index of the target qubit
        theta: Rotation angle in radians
    """
    pass

class CRY(_gates.CRYGate):
    """Controlled rotation around Y-axis.
    
    Args:
        control: Index of the control qubit
        target: Index of the target qubit
        theta: Rotation angle in radians
    """
    pass

class CRZ(_gates.CRZGate):
    """Controlled rotation around Z-axis.
    
    Args:
        control: Index of the control qubit
        target: Index of the target qubit
        theta: Rotation angle in radians
    """
    pass

# Three-qubit gates
class Toffoli(_gates.ToffoliGate):
    """Toffoli (CCNOT) gate - doubly-controlled NOT.
    
    Flips target if both controls are |1⟩.
    
    Args:
        control1: Index of the first control qubit
        control2: Index of the second control qubit
        target: Index of the target qubit
    """
    pass

class Fredkin(_gates.FredkinGate):
    """Fredkin (CSWAP) gate - controlled SWAP.
    
    Swaps targets if control is |1⟩.
    
    Args:
        control: Index of the control qubit
        target1: Index of the first target qubit
        target2: Index of the second target qubit
    """
    pass

# Parametric gates
class ParametricRX(_gates.ParametricRX):
    """Parametric rotation around X-axis.
    
    Supports both numeric and symbolic parameters for variational algorithms.
    
    Args:
        qubit: Index of the qubit
        theta: Rotation angle (float) or parameter name (str)
        
    Example:
        >>> # With numeric parameter
        >>> gate1 = ParametricRX(0, np.pi/2)
        >>> 
        >>> # With symbolic parameter
        >>> gate2 = ParametricRX(0, "theta1")
        >>> gate2 = gate2.assign({"theta1": np.pi/2})
    """
    pass

class ParametricRY(_gates.ParametricRY):
    """Parametric rotation around Y-axis.
    
    Args:
        qubit: Index of the qubit
        theta: Rotation angle (float) or parameter name (str)
    """
    pass

class ParametricRZ(_gates.ParametricRZ):
    """Parametric rotation around Z-axis.
    
    Args:
        qubit: Index of the qubit
        theta: Rotation angle (float) or parameter name (str)
    """
    pass

class ParametricU(_gates.ParametricUGate):
    """Parametric general single-qubit gate.
    
    U(θ, φ, λ) = [[cos(θ/2), -e^(iλ)sin(θ/2)], 
                   [e^(iφ)sin(θ/2), e^(i(φ+λ))cos(θ/2)]]
    
    Args:
        qubit: Index of the qubit
        theta: First rotation angle (float/str)
        phi: Second rotation angle (float/str)
        lambda_: Third rotation angle (float/str)
    """
    pass

class CustomGate(_gates.CustomGate):
    """Custom gate from matrix.
    
    Create a gate from an arbitrary unitary matrix.
    
    Args:
        name: Name of the gate
        qubits: List of qubit indices
        matrix: Unitary matrix as numpy array
        
    Example:
        >>> # Create a custom 2-qubit gate
        >>> matrix = np.array([[1, 0, 0, 0],
        ...                    [0, 0, 1j, 0],
        ...                    [0, 1j, 0, 0],
        ...                    [0, 0, 0, 1]], dtype=complex)
        >>> gate = CustomGate("MyGate", [0, 1], matrix)
    """
    pass

# Utility functions
def create_controlled_gate(gate: Gate, control_qubits: List[int]) -> Gate:
    """Create a controlled version of any gate.
    
    Args:
        gate: The base gate to control
        control_qubits: List of control qubit indices
        
    Returns:
        Controlled version of the gate
    """
    # This would need implementation in Rust
    raise NotImplementedError("Controlled gate creation not yet implemented")

def decompose_gate(gate: Gate) -> List[Gate]:
    """Decompose a gate into simpler gates.
    
    Args:
        gate: The gate to decompose
        
    Returns:
        List of simpler gates
    """
    # This would need implementation in Rust
    raise NotImplementedError("Gate decomposition not yet implemented")

# Convenience functions matching the circuit API
def h(qubit: int) -> H:
    """Create a Hadamard gate."""
    return H(qubit)

def x(qubit: int) -> X:
    """Create a Pauli-X gate."""
    return X(qubit)

def y(qubit: int) -> Y:
    """Create a Pauli-Y gate."""
    return Y(qubit)

def z(qubit: int) -> Z:
    """Create a Pauli-Z gate."""
    return Z(qubit)

def s(qubit: int) -> S:
    """Create an S gate."""
    return S(qubit)

def sdg(qubit: int) -> SDagger:
    """Create an S-dagger gate."""
    return SDagger(qubit)

def t(qubit: int) -> T:
    """Create a T gate."""
    return T(qubit)

def tdg(qubit: int) -> TDagger:
    """Create a T-dagger gate."""
    return TDagger(qubit)

def sx(qubit: int) -> SX:
    """Create a square root of X gate."""
    return SX(qubit)

def sxdg(qubit: int) -> SXDagger:
    """Create a square root of X dagger gate."""
    return SXDagger(qubit)

def rx(qubit: int, theta: float) -> RX:
    """Create an X-rotation gate."""
    return RX(qubit, theta)

def ry(qubit: int, theta: float) -> RY:
    """Create a Y-rotation gate."""
    return RY(qubit, theta)

def rz(qubit: int, theta: float) -> RZ:
    """Create a Z-rotation gate."""
    return RZ(qubit, theta)

def cnot(control: int, target: int) -> CNOT:
    """Create a CNOT gate."""
    return CNOT(control, target)

def cy(control: int, target: int) -> CY:
    """Create a controlled-Y gate."""
    return CY(control, target)

def cz(control: int, target: int) -> CZ:
    """Create a controlled-Z gate."""
    return CZ(control, target)

def ch(control: int, target: int) -> CH:
    """Create a controlled-H gate."""
    return CH(control, target)

def cs(control: int, target: int) -> CS:
    """Create a controlled-S gate."""
    return CS(control, target)

def swap(qubit1: int, qubit2: int) -> SWAP:
    """Create a SWAP gate."""
    return SWAP(qubit1, qubit2)

def crx(control: int, target: int, theta: float) -> CRX:
    """Create a controlled X-rotation gate."""
    return CRX(control, target, theta)

def cry(control: int, target: int, theta: float) -> CRY:
    """Create a controlled Y-rotation gate."""
    return CRY(control, target, theta)

def crz(control: int, target: int, theta: float) -> CRZ:
    """Create a controlled Z-rotation gate."""
    return CRZ(control, target, theta)

def toffoli(control1: int, control2: int, target: int) -> Toffoli:
    """Create a Toffoli gate."""
    return Toffoli(control1, control2, target)

def fredkin(control: int, target1: int, target2: int) -> Fredkin:
    """Create a Fredkin gate."""
    return Fredkin(control, target1, target2)

# Gate aliases
CX = CNOT  # Common alias for CNOT
CCX = Toffoli  # Common alias for Toffoli
CSWAP = Fredkin  # Common alias for Fredkin

__all__ = [
    # Base classes
    'Gate', 'GateParameter', 'ParametricGateBase',
    
    # Single-qubit gates
    'H', 'X', 'Y', 'Z', 'S', 'SDagger', 'T', 'TDagger', 'SX', 'SXDagger',
    'RX', 'RY', 'RZ',
    
    # Two-qubit gates
    'CNOT', 'CY', 'CZ', 'CH', 'CS', 'SWAP', 'CRX', 'CRY', 'CRZ',
    
    # Three-qubit gates
    'Toffoli', 'Fredkin',
    
    # Parametric gates
    'ParametricRX', 'ParametricRY', 'ParametricRZ', 'ParametricU',
    
    # Custom gate
    'CustomGate',
    
    # Aliases
    'CX', 'CCX', 'CSWAP',
    
    # Convenience functions
    'h', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg', 'sx', 'sxdg',
    'rx', 'ry', 'rz', 'cnot', 'cy', 'cz', 'ch', 'cs', 'swap',
    'crx', 'cry', 'crz', 'toffoli', 'fredkin',
    
    # Utility functions
    'create_controlled_gate', 'decompose_gate',
]