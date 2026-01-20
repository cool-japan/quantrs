"""
Validated Quantum Gates for QuantRS2

This module provides security-hardened quantum gates with comprehensive
input validation and sanitization for production environments.
"""

import logging
import numpy as np
from typing import Union, List, Optional, Dict, Tuple, Any
from functools import wraps

# Import the base gates module
from . import gates
from .security import validate_quantum_input, get_quantum_validator, QuantumValidationConfig

logger = logging.getLogger(__name__)

# Global validation configuration
_validation_config = QuantumValidationConfig()

def set_validation_config(config: QuantumValidationConfig) -> None:
    """Set global validation configuration for gates."""
    global _validation_config
    _validation_config = config

def validate_gate_input(func):
    """Decorator to validate gate inputs."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Validate positional arguments
            validated_args = []
            for i, arg in enumerate(args):
                if isinstance(arg, (int, float)) and i > 0:  # Skip self/cls
                    if 'qubit' in func.__name__.lower() or i == 1:  # Likely qubit index
                        validated_arg = validate_quantum_input(arg, "qubit_index", strict=False)
                    else:  # Likely angle or parameter
                        validated_arg = validate_quantum_input(arg, "rotation_angle", strict=False)
                    validated_args.append(validated_arg)
                else:
                    validated_args.append(arg)
            
            # Validate keyword arguments
            validated_kwargs = {}
            for key, value in kwargs.items():
                if key in ['qubit', 'control', 'target', 'qubit1', 'qubit2']:
                    validated_value = validate_quantum_input(value, "qubit_index", strict=False)
                elif key in ['angle', 'theta', 'phi', 'lambda', 'parameter']:
                    validated_value = validate_quantum_input(value, "rotation_angle", strict=False)
                elif key == 'matrix':
                    validated_value = validate_quantum_input(value, "gate_matrix", strict=False)
                else:
                    validated_value = validate_quantum_input(value, "gate_parameter", strict=False)
                validated_kwargs[key] = validated_value
            
            return func(*validated_args, **validated_kwargs)
            
        except Exception as e:
            logger.error(f"Gate validation failed for {func.__name__}: {e}")
            raise ValueError(f"Invalid gate parameters: {e}")
    
    return wrapper

class ValidatedGateFactory:
    """Factory class for creating validated quantum gates."""
    
    def __init__(self, config: Optional[QuantumValidationConfig] = None):
        self.config = config or _validation_config
        self.validator = get_quantum_validator(self.config)
    
    @validate_gate_input
    def H(self, qubit: int) -> gates.Gate:
        """Create validated Hadamard gate."""
        return gates.H(qubit)
    
    @validate_gate_input  
    def X(self, qubit: int) -> gates.Gate:
        """Create validated Pauli-X gate."""
        return gates.X(qubit)
    
    @validate_gate_input
    def Y(self, qubit: int) -> gates.Gate:
        """Create validated Pauli-Y gate."""
        return gates.Y(qubit)
    
    @validate_gate_input
    def Z(self, qubit: int) -> gates.Gate:
        """Create validated Pauli-Z gate."""
        return gates.Z(qubit)
    
    @validate_gate_input
    def S(self, qubit: int) -> gates.Gate:
        """Create validated S gate."""
        return gates.S(qubit)
    
    @validate_gate_input
    def T(self, qubit: int) -> gates.Gate:
        """Create validated T gate."""
        return gates.T(qubit)
    
    @validate_gate_input
    def RX(self, qubit: int, angle: float) -> gates.Gate:
        """Create validated X-rotation gate."""
        return gates.RX(qubit, angle)
    
    @validate_gate_input
    def RY(self, qubit: int, angle: float) -> gates.Gate:
        """Create validated Y-rotation gate."""
        return gates.RY(qubit, angle)
    
    @validate_gate_input
    def RZ(self, qubit: int, angle: float) -> gates.Gate:
        """Create validated Z-rotation gate."""
        return gates.RZ(qubit, angle)
    
    @validate_gate_input
    def CNOT(self, control: int, target: int) -> gates.Gate:
        """Create validated CNOT gate."""
        return gates.CNOT(control, target)
    
    @validate_gate_input
    def CZ(self, control: int, target: int) -> gates.Gate:
        """Create validated controlled-Z gate."""
        return gates.CZ(control, target)
    
    @validate_gate_input
    def SWAP(self, qubit1: int, qubit2: int) -> gates.Gate:
        """Create validated SWAP gate."""
        return gates.SWAP(qubit1, qubit2)
    
    @validate_gate_input
    def Toffoli(self, control1: int, control2: int, target: int) -> gates.Gate:
        """Create validated Toffoli gate."""
        return gates.Toffoli(control1, control2, target)
    
    def custom_gate(self, matrix: np.ndarray, qubits: List[int], name: str = "Custom") -> gates.Gate:
        """Create validated custom gate from unitary matrix."""
        try:
            # Validate matrix
            validated_matrix = self.validator.validate_and_sanitize(matrix, "gate_matrix", strict=True)
            
            # Validate qubits
            validated_qubits = self.validator.validate_qubit_indices(qubits)
            
            # Validate name
            validated_name = self.validator.validate_and_sanitize(name, "circuit_name", strict=False)
            
            return gates.custom_gate(validated_matrix, validated_qubits, validated_name)
            
        except Exception as e:
            logger.error(f"Custom gate validation failed: {e}")
            raise ValueError(f"Invalid custom gate parameters: {e}")
    
    def controlled_gate(self, base_gate: gates.Gate, control_qubits: List[int]) -> gates.Gate:
        """Create validated controlled gate."""
        try:
            # Validate control qubits
            validated_controls = self.validator.validate_qubit_indices(control_qubits)
            
            return gates.controlled_gate(base_gate, validated_controls)
            
        except Exception as e:
            logger.error(f"Controlled gate validation failed: {e}")
            raise ValueError(f"Invalid controlled gate parameters: {e}")

# Global validated gate factory instance
_gate_factory = ValidatedGateFactory()

# Export validated gate creation functions
def H(qubit: int) -> gates.Gate:
    """Create validated Hadamard gate."""
    return _gate_factory.H(qubit)

def X(qubit: int) -> gates.Gate:
    """Create validated Pauli-X gate."""
    return _gate_factory.X(qubit)

def Y(qubit: int) -> gates.Gate:
    """Create validated Pauli-Y gate."""
    return _gate_factory.Y(qubit)

def Z(qubit: int) -> gates.Gate:
    """Create validated Pauli-Z gate."""
    return _gate_factory.Z(qubit)

def S(qubit: int) -> gates.Gate:
    """Create validated S gate."""
    return _gate_factory.S(qubit)

def T(qubit: int) -> gates.Gate:
    """Create validated T gate."""
    return _gate_factory.T(qubit)

def RX(qubit: int, angle: float) -> gates.Gate:
    """Create validated X-rotation gate."""
    return _gate_factory.RX(qubit, angle)

def RY(qubit: int, angle: float) -> gates.Gate:
    """Create validated Y-rotation gate."""
    return _gate_factory.RY(qubit, angle)

def RZ(qubit: int, angle: float) -> gates.Gate:
    """Create validated Z-rotation gate."""
    return _gate_factory.RZ(qubit, angle)

def CNOT(control: int, target: int) -> gates.Gate:
    """Create validated CNOT gate."""
    return _gate_factory.CNOT(control, target)

def CZ(control: int, target: int) -> gates.Gate:
    """Create validated controlled-Z gate."""
    return _gate_factory.CZ(control, target)

def SWAP(qubit1: int, qubit2: int) -> gates.Gate:
    """Create validated SWAP gate."""
    return _gate_factory.SWAP(qubit1, qubit2)

def Toffoli(control1: int, control2: int, target: int) -> gates.Gate:
    """Create validated Toffoli gate."""
    return _gate_factory.Toffoli(control1, control2, target)

def custom_gate(matrix: np.ndarray, qubits: List[int], name: str = "Custom") -> gates.Gate:
    """Create validated custom gate from unitary matrix."""
    return _gate_factory.custom_gate(matrix, qubits, name)

def controlled_gate(base_gate: gates.Gate, control_qubits: List[int]) -> gates.Gate:
    """Create validated controlled gate."""
    return _gate_factory.controlled_gate(base_gate, control_qubits)

# Circuit validation utilities
class CircuitValidator:
    """Validator for quantum circuits."""
    
    def __init__(self, config: Optional[QuantumValidationConfig] = None):
        self.config = config or _validation_config
        self.validator = get_quantum_validator(self.config)
    
    def validate_circuit_structure(self, circuit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate complete circuit structure."""
        try:
            validated_circuit = {}
            
            # Validate circuit metadata
            if 'name' in circuit_data:
                validated_circuit['name'] = self.validator.validate_and_sanitize(
                    circuit_data['name'], "circuit_name", strict=False
                )
            
            if 'num_qubits' in circuit_data:
                validated_circuit['num_qubits'] = self.validator.validate_and_sanitize(
                    circuit_data['num_qubits'], "qubit_index", strict=False
                )
            
            # Validate gates
            if 'gates' in circuit_data:
                validated_circuit['gates'] = []
                for gate_data in circuit_data['gates']:
                    validated_gate = self.validate_gate_data(gate_data)
                    validated_circuit['gates'].append(validated_gate)
            
            # Validate measurements
            if 'measurements' in circuit_data:
                validated_circuit['measurements'] = []
                for measurement in circuit_data['measurements']:
                    validated_measurement = self.validate_measurement_data(measurement)
                    validated_circuit['measurements'].append(validated_measurement)
            
            return validated_circuit
            
        except Exception as e:
            logger.error(f"Circuit validation failed: {e}")
            raise ValueError(f"Invalid circuit structure: {e}")
    
    def validate_gate_data(self, gate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate individual gate data."""
        validated_gate = {}
        
        # Validate gate type
        if 'type' in gate_data:
            validated_gate['type'] = self.validator.validate_and_sanitize(
                gate_data['type'], "circuit_name", strict=False
            )
        
        # Validate qubits
        if 'qubits' in gate_data:
            validated_gate['qubits'] = self.validator.validate_qubit_indices(gate_data['qubits'])
        
        # Validate parameters
        if 'parameters' in gate_data:
            validated_gate['parameters'] = self.validator.validate_gate_parameters(gate_data['parameters'])
        
        return validated_gate
    
    def validate_measurement_data(self, measurement_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate measurement data."""
        validated_measurement = {}
        
        if 'qubits' in measurement_data:
            validated_measurement['qubits'] = self.validator.validate_qubit_indices(
                measurement_data['qubits']
            )
        
        if 'classical_bits' in measurement_data:
            validated_measurement['classical_bits'] = [
                self.validator.validate_and_sanitize(bit, "qubit_index", strict=False)
                for bit in measurement_data['classical_bits']
            ]
        
        return validated_measurement

# Global circuit validator
_circuit_validator = CircuitValidator()

def validate_circuit(circuit_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate complete circuit structure."""
    return _circuit_validator.validate_circuit_structure(circuit_data)

def validate_gate_sequence(gates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate a sequence of gates."""
    return [_circuit_validator.validate_gate_data(gate) for gate in gates]

# Export all validated components
__all__ = [
    # Validated gate functions
    "H", "X", "Y", "Z", "S", "T",
    "RX", "RY", "RZ",
    "CNOT", "CZ", "SWAP", "Toffoli",
    "custom_gate", "controlled_gate",
    
    # Validation classes and utilities
    "ValidatedGateFactory",
    "CircuitValidator", 
    "validate_circuit",
    "validate_gate_sequence",
    "set_validation_config",
]