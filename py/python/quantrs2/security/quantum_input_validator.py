"""
Quantum-Specific Input Validation for QuantRS2

This module provides specialized input validation for quantum computing operations,
extending the base input validator with quantum-specific validation rules.
"""

import logging
import math
import re
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from .input_validator import InputValidator, ValidationRule, ValidationError

logger = logging.getLogger(__name__)

@dataclass
class QuantumValidationConfig:
    """Configuration for quantum input validation."""
    
    max_qubits: int = 1024
    max_circuit_depth: int = 10000
    max_angle_magnitude: float = 100 * math.pi  # Allow up to 100π
    max_matrix_dimension: int = 2**20  # 2^20 x 2^20 matrices max
    allow_symbolic_parameters: bool = True
    strict_gate_validation: bool = True

class QuantumInputValidator(InputValidator):
    """
    Specialized input validator for quantum computing operations.
    
    Adds quantum-specific validation rules for:
    - Qubit indices and ranges
    - Rotation angles and parameters
    - Quantum gate matrices
    - Circuit structures
    - QASM code
    - Quantum states and amplitudes
    """
    
    def __init__(self, config: Optional[QuantumValidationConfig] = None):
        super().__init__()
        self.config = config or QuantumValidationConfig()
        self._initialize_quantum_rules()
    
    def _initialize_quantum_rules(self) -> None:
        """Initialize quantum-specific validation rules."""
        
        # Qubit index validation
        self.add_field_rule("qubit_index", ValidationRule(
            name="valid_qubit_index",
            validator=self._validate_qubit_index,
            error_message=f"Qubit index must be non-negative integer < {self.config.max_qubits}",
            sanitizer=self._sanitize_qubit_index,
        ))
        
        # Angle validation for rotation gates
        self.add_field_rule("rotation_angle", ValidationRule(
            name="valid_rotation_angle",
            validator=self._validate_rotation_angle,
            error_message=f"Rotation angle must be real number with magnitude < {self.config.max_angle_magnitude}",
            sanitizer=self._sanitize_rotation_angle,
        ))
        
        # Matrix validation for custom gates
        self.add_field_rule("gate_matrix", ValidationRule(
            name="valid_unitary_matrix",
            validator=self._validate_unitary_matrix,
            error_message="Gate matrix must be square unitary matrix",
            sanitizer=self._sanitize_matrix,
        ))
        
        # QASM code validation
        self.add_field_rule("qasm_code", ValidationRule(
            name="safe_qasm_code",
            validator=self._validate_qasm_code,
            error_message="QASM code contains unsafe patterns",
            sanitizer=self._sanitize_qasm_code,
        ))
        
        # Circuit name validation
        self.add_field_rule("circuit_name", ValidationRule(
            name="valid_circuit_name",
            validator=self._validate_circuit_name,
            error_message="Circuit name contains invalid characters",
            sanitizer=self._sanitize_circuit_name,
        ))
        
        # Parameter validation
        self.add_field_rule("gate_parameter", ValidationRule(
            name="valid_gate_parameter",
            validator=self._validate_gate_parameter,
            error_message="Gate parameter must be numeric or valid symbolic expression",
            sanitizer=self._sanitize_gate_parameter,
        ))
    
    def _validate_qubit_index(self, value: Any) -> bool:
        """Validate qubit index is within acceptable range."""
        try:
            if isinstance(value, str) and value.isdigit():
                value = int(value)
            
            if not isinstance(value, int):
                return False
            
            return 0 <= value < self.config.max_qubits
        except (ValueError, TypeError):
            return False
    
    def _sanitize_qubit_index(self, value: Any) -> int:
        """Sanitize qubit index to valid value."""
        try:
            if isinstance(value, str) and value.isdigit():
                value = int(value)
            
            if isinstance(value, (int, float)):
                value = int(abs(value))  # Ensure non-negative
                return min(value, self.config.max_qubits - 1)
            
            return 0  # Default fallback
        except (ValueError, TypeError):
            return 0
    
    def _validate_rotation_angle(self, value: Any) -> bool:
        """Validate rotation angle is a valid real number."""
        try:
            if isinstance(value, str):
                # Check for symbolic parameters if allowed
                if self.config.allow_symbolic_parameters and self._is_symbolic_parameter(value):
                    return True
                value = float(value)
            
            if not isinstance(value, (int, float)):
                return False
            
            if math.isnan(value) or math.isinf(value):
                return False
            
            return abs(value) <= self.config.max_angle_magnitude
        except (ValueError, TypeError):
            return False
    
    def _sanitize_rotation_angle(self, value: Any) -> float:
        """Sanitize rotation angle to valid value."""
        try:
            if isinstance(value, str):
                if self.config.allow_symbolic_parameters and self._is_symbolic_parameter(value):
                    return value  # Keep symbolic parameters as-is
                value = float(value)
            
            if isinstance(value, (int, float)):
                if math.isnan(value) or math.isinf(value):
                    return 0.0
                
                # Clamp to valid range
                return max(-self.config.max_angle_magnitude, 
                          min(self.config.max_angle_magnitude, float(value)))
            
            return 0.0  # Default fallback
        except (ValueError, TypeError):
            return 0.0
    
    def _validate_unitary_matrix(self, value: Any) -> bool:
        """Validate that matrix is a valid unitary matrix."""
        try:
            if isinstance(value, (list, tuple)):
                matrix = np.array(value, dtype=complex)
            elif isinstance(value, np.ndarray):
                matrix = value.astype(complex)
            else:
                return False
            
            # Check if square
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                return False
            
            # Check size limits
            if matrix.shape[0] > self.config.max_matrix_dimension:
                return False
            
            # Check if unitary (U† U = I) within tolerance
            n = matrix.shape[0]
            identity = np.eye(n, dtype=complex)
            product = matrix.conj().T @ matrix
            
            return np.allclose(product, identity, atol=1e-10)
            
        except (ValueError, TypeError, np.linalg.LinAlgError):
            return False
    
    def _sanitize_matrix(self, value: Any) -> np.ndarray:
        """Sanitize matrix to valid form."""
        try:
            if isinstance(value, (list, tuple)):
                matrix = np.array(value, dtype=complex)
            elif isinstance(value, np.ndarray):
                matrix = value.astype(complex)
            else:
                # Return identity matrix as fallback
                return np.eye(2, dtype=complex)
            
            # Ensure square
            if matrix.ndim != 2:
                return np.eye(2, dtype=complex)
            
            if matrix.shape[0] != matrix.shape[1]:
                n = min(matrix.shape)
                matrix = matrix[:n, :n]
            
            # Clamp size
            if matrix.shape[0] > self.config.max_matrix_dimension:
                return np.eye(2, dtype=complex)
            
            # If not unitary, try to make it unitary via QR decomposition
            if not self._validate_unitary_matrix(matrix):
                try:
                    Q, R = np.linalg.qr(matrix)
                    # Adjust phases to make R have positive diagonal
                    phases = np.diag(R) / np.abs(np.diag(R))
                    matrix = Q * phases
                except np.linalg.LinAlgError:
                    return np.eye(matrix.shape[0], dtype=complex)
            
            return matrix
            
        except (ValueError, TypeError):
            return np.eye(2, dtype=complex)
    
    def _validate_qasm_code(self, value: Any) -> bool:
        """Validate QASM code for safety."""
        if not isinstance(value, str):
            return False
        
        # Check for dangerous patterns in QASM
        dangerous_patterns = [
            r'include\s*"[^"]*\.\./[^"]*"',  # Path traversal in includes
            r'pragma\s+.*shell',            # Shell pragma
            r'eval\s*\(',                    # Eval calls
            r'exec\s*\(',                    # Exec calls
            r'__[a-zA-Z]',                   # Python dunder methods
            r'import\s+',                    # Import statements
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return False
        
        return True
    
    def _sanitize_qasm_code(self, value: Any) -> str:
        """Sanitize QASM code by removing dangerous patterns."""
        if not isinstance(value, str):
            return ""
        
        # Remove dangerous patterns
        sanitized = re.sub(r'include\s*"[^"]*\.\./[^"]*"', '', value)
        sanitized = re.sub(r'pragma\s+.*shell[^;]*;', '', sanitized)
        sanitized = re.sub(r'eval\s*\([^)]*\)', '', sanitized)
        sanitized = re.sub(r'exec\s*\([^)]*\)', '', sanitized)
        sanitized = re.sub(r'__[a-zA-Z][a-zA-Z0-9_]*', '', sanitized)
        sanitized = re.sub(r'import\s+[^;]*;', '', sanitized)
        
        return sanitized.strip()
    
    def _validate_circuit_name(self, value: Any) -> bool:
        """Validate circuit name contains only safe characters."""
        if not isinstance(value, str):
            return False
        
        # Allow alphanumeric, underscore, hyphen, period
        return re.match(r'^[a-zA-Z0-9_.-]+$', value) is not None
    
    def _sanitize_circuit_name(self, value: Any) -> str:
        """Sanitize circuit name to safe characters."""
        if not isinstance(value, str):
            return "unnamed_circuit"
        
        # Keep only safe characters
        sanitized = re.sub(r'[^a-zA-Z0-9_.-]', '_', value)
        
        # Ensure not empty
        if not sanitized:
            sanitized = "unnamed_circuit"
        
        return sanitized
    
    def _validate_gate_parameter(self, value: Any) -> bool:
        """Validate gate parameter is numeric or valid symbolic."""
        if isinstance(value, (int, float)):
            return not (math.isnan(value) or math.isinf(value))
        elif isinstance(value, str):
            if self.config.allow_symbolic_parameters:
                return self._is_symbolic_parameter(value)
            else:
                try:
                    float(value)
                    return True
                except ValueError:
                    return False
        elif isinstance(value, complex):
            return not (math.isnan(value.real) or math.isnan(value.imag) or
                       math.isinf(value.real) or math.isinf(value.imag))
        
        return False
    
    def _sanitize_gate_parameter(self, value: Any) -> Union[float, str]:
        """Sanitize gate parameter to valid value."""
        if isinstance(value, (int, float)):
            if math.isnan(value) or math.isinf(value):
                return 0.0
            return float(value)
        elif isinstance(value, str):
            if self.config.allow_symbolic_parameters and self._is_symbolic_parameter(value):
                return value
            try:
                return float(value)
            except ValueError:
                return 0.0
        elif isinstance(value, complex):
            if math.isnan(value.real) or math.isnan(value.imag) or \
               math.isinf(value.real) or math.isinf(value.imag):
                return 0.0
            return value
        
        return 0.0
    
    def _is_symbolic_parameter(self, value: str) -> bool:
        """Check if string represents a valid symbolic parameter."""
        # Simple check for basic symbolic parameters
        # In full implementation, would use a proper expression parser
        if not value:
            return False
        
        # Allow simple variable names and mathematical expressions
        safe_pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*(\s*[\+\-\*/]\s*[a-zA-Z0-9_\.]*)*$'
        return re.match(safe_pattern, value) is not None
    
    def validate_qubit_indices(self, indices: List[int]) -> List[int]:
        """Validate and sanitize a list of qubit indices."""
        validated_indices = []
        for idx in indices:
            validated_idx = self.validate_and_sanitize(idx, "qubit_index", strict=False)
            validated_indices.append(validated_idx)
        return validated_indices
    
    def validate_gate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize gate parameters dictionary."""
        validated_params = {}
        for key, value in parameters.items():
            validated_key = self.validate_and_sanitize(key, "circuit_name", strict=False)
            validated_value = self.validate_and_sanitize(value, "gate_parameter", strict=False)
            validated_params[validated_key] = validated_value
        return validated_params
    
    def validate_quantum_state(self, state: Union[List, np.ndarray]) -> np.ndarray:
        """Validate and normalize a quantum state vector."""
        try:
            if isinstance(state, (list, tuple)):
                state = np.array(state, dtype=complex)
            elif not isinstance(state, np.ndarray):
                raise ValueError("State must be array-like")
            
            state = state.astype(complex)
            
            # Check if state is a valid quantum state
            if state.ndim != 1:
                raise ValueError("State must be 1-dimensional")
            
            # Check if dimension is power of 2
            n_qubits = int(np.log2(len(state)))
            if 2**n_qubits != len(state):
                raise ValueError("State dimension must be power of 2")
            
            # Check if too large
            if n_qubits > int(np.log2(self.config.max_matrix_dimension)):
                raise ValueError("State too large")
            
            # Normalize state
            norm = np.linalg.norm(state)
            if norm == 0:
                # Return |0> state as fallback
                normalized_state = np.zeros(len(state), dtype=complex)
                normalized_state[0] = 1.0
                return normalized_state
            
            return state / norm
            
        except (ValueError, TypeError):
            # Return |0> state as fallback
            return np.array([1.0] + [0.0] * (len(state) - 1 if hasattr(state, '__len__') else 1), 
                           dtype=complex)

# Global quantum validator instance
_quantum_validator = None

def get_quantum_validator(config: Optional[QuantumValidationConfig] = None) -> QuantumInputValidator:
    """Get the global quantum validator instance."""
    global _quantum_validator
    if _quantum_validator is None:
        _quantum_validator = QuantumInputValidator(config)
    return _quantum_validator

def validate_quantum_input(
    data: Any, 
    field_type: str, 
    strict: bool = True
) -> Any:
    """Convenience function for validating quantum inputs."""
    validator = get_quantum_validator()
    return validator.validate_and_sanitize(data, field_type, strict)