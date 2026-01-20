"""
Secure Interactive Circuit Builder for QuantRS2

This module provides a security-hardened interactive GUI for building quantum circuits
with comprehensive input validation and sanitization.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
import threading
import time

# Import base circuit builder
from . import circuit_builder
from .security import validate_quantum_input, get_quantum_validator, QuantumValidationConfig

logger = logging.getLogger(__name__)

@dataclass 
class SecureBuilderConfig:
    """Configuration for secure circuit builder."""
    
    max_circuit_size: int = 1000
    max_gates_per_operation: int = 100
    allow_custom_gates: bool = True
    allow_file_operations: bool = True
    validate_user_input: bool = True
    sandbox_execution: bool = True
    max_session_duration: int = 3600  # 1 hour

class CircuitBuilderSecurityError(Exception):
    """Exception raised for circuit builder security violations."""
    pass

class SecureGateInfo(circuit_builder.GateInfo):
    """Security-enhanced gate information."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validator = get_quantum_validator()
    
    def validate_parameters(self, params: List[Any]) -> List[Any]:
        """Validate gate parameters."""
        validated_params = []
        for param in params:
            validated_param = self.validator.validate_and_sanitize(
                param, "gate_parameter", strict=False
            )
            validated_params.append(validated_param)
        return validated_params

class SecureCircuitBuilder:
    """Security-hardened circuit builder with input validation."""
    
    def __init__(self, config: Optional[SecureBuilderConfig] = None):
        self.config = config or SecureBuilderConfig()
        self.validator = get_quantum_validator()
        self.session_start = time.time()
        self.operations_count = 0
        
        # Initialize base builder
        self.base_builder = circuit_builder.CircuitBuilder()
        
        # Security tracking
        self.validated_operations = []
        self.security_log = []
    
    def add_gate(self, gate_type: str, qubits: List[int], 
                parameters: Optional[List[Any]] = None) -> bool:
        """
        Add gate to circuit with security validation.
        
        Args:
            gate_type: Type of gate to add
            qubits: List of qubit indices
            parameters: Optional gate parameters
            
        Returns:
            True if gate was added successfully
            
        Raises:
            CircuitBuilderSecurityError: If validation fails
        """
        try:
            # Check session limits
            self._check_session_limits()
            
            # Validate gate type
            validated_gate_type = self._validate_gate_type(gate_type)
            
            # Validate qubits
            validated_qubits = self._validate_qubits(qubits)
            
            # Validate parameters
            validated_parameters = self._validate_parameters(parameters or [])
            
            # Check circuit size limits
            if len(self.validated_operations) >= self.config.max_circuit_size:
                raise CircuitBuilderSecurityError("Circuit size limit exceeded")
            
            # Log operation
            operation = {
                'type': 'add_gate',
                'gate_type': validated_gate_type,
                'qubits': validated_qubits,
                'parameters': validated_parameters,
                'timestamp': time.time()
            }
            
            self.validated_operations.append(operation)
            self.operations_count += 1
            
            # Add to base builder
            success = self.base_builder.add_gate(
                validated_gate_type, validated_qubits, validated_parameters
            )
            
            if success:
                self._log_security_event("gate_added", operation)
            
            return success
            
        except Exception as e:
            self._log_security_event("gate_add_failed", {"error": str(e)})
            if isinstance(e, CircuitBuilderSecurityError):
                raise
            raise CircuitBuilderSecurityError(f"Gate addition failed: {e}")
    
    def load_circuit(self, circuit_data: Union[str, Dict[str, Any]], 
                    trusted_source: bool = False) -> bool:
        """
        Load circuit with security validation.
        
        Args:
            circuit_data: Circuit data as JSON string or dictionary
            trusted_source: Whether to trust the source
            
        Returns:
            True if circuit was loaded successfully
        """
        try:
            # Parse circuit data if string
            if isinstance(circuit_data, str):
                validated_data = self._validate_json_input(circuit_data)
            else:
                validated_data = circuit_data
            
            # Validate circuit structure
            validated_circuit = self._validate_circuit_data(validated_data, trusted_source)
            
            # Check circuit size
            if self._estimate_circuit_size(validated_circuit) > self.config.max_circuit_size:
                raise CircuitBuilderSecurityError("Circuit too large to load")
            
            # Load into base builder
            success = self.base_builder.load_circuit(validated_circuit)
            
            if success:
                self.validated_operations = validated_circuit.get('operations', [])
                self._log_security_event("circuit_loaded", {
                    "gates": len(self.validated_operations),
                    "trusted": trusted_source
                })
            
            return success
            
        except Exception as e:
            self._log_security_event("circuit_load_failed", {"error": str(e)})
            if isinstance(e, CircuitBuilderSecurityError):
                raise
            raise CircuitBuilderSecurityError(f"Circuit loading failed: {e}")
    
    def export_circuit(self, format: str = "json", validate_output: bool = True) -> str:
        """
        Export circuit with output validation.
        
        Args:
            format: Export format ('json', 'qasm', etc.)
            validate_output: Whether to validate exported data
            
        Returns:
            Exported circuit data
        """
        try:
            # Validate format
            validated_format = self._validate_export_format(format)
            
            # Export using base builder
            exported_data = self.base_builder.export_circuit(validated_format)
            
            # Validate output if requested
            if validate_output:
                exported_data = self._validate_exported_data(exported_data, validated_format)
            
            self._log_security_event("circuit_exported", {"format": validated_format})
            
            return exported_data
            
        except Exception as e:
            self._log_security_event("circuit_export_failed", {"error": str(e)})
            raise CircuitBuilderSecurityError(f"Circuit export failed: {e}")
    
    def save_circuit_file(self, filepath: Union[str, Path], 
                         format: str = "json") -> bool:
        """
        Save circuit to file with security validation.
        
        Args:
            filepath: Output file path
            format: File format
            
        Returns:
            True if save was successful
        """
        try:
            if not self.config.allow_file_operations:
                raise CircuitBuilderSecurityError("File operations not allowed")
            
            # Validate file path
            validated_path = self._validate_file_path(Path(filepath), for_write=True)
            
            # Export circuit
            circuit_data = self.export_circuit(format, validate_output=True)
            
            # Write to file securely
            with open(validated_path, 'w', encoding='utf-8') as f:
                f.write(circuit_data)
            
            self._log_security_event("circuit_saved", {
                "path": str(validated_path),
                "format": format
            })
            
            return True
            
        except Exception as e:
            self._log_security_event("circuit_save_failed", {"error": str(e)})
            if isinstance(e, CircuitBuilderSecurityError):
                raise
            raise CircuitBuilderSecurityError(f"Circuit save failed: {e}")
    
    def load_circuit_file(self, filepath: Union[str, Path], 
                         trusted_source: bool = False) -> bool:
        """
        Load circuit from file with security validation.
        
        Args:
            filepath: Input file path
            trusted_source: Whether to trust the file source
            
        Returns:
            True if load was successful
        """
        try:
            if not self.config.allow_file_operations:
                raise CircuitBuilderSecurityError("File operations not allowed")
            
            # Validate file path
            validated_path = self._validate_file_path(Path(filepath), for_write=False)
            
            # Read file securely
            with open(validated_path, 'r', encoding='utf-8') as f:
                circuit_data = f.read()
            
            # Load circuit
            success = self.load_circuit(circuit_data, trusted_source)
            
            if success:
                self._log_security_event("circuit_loaded_from_file", {
                    "path": str(validated_path),
                    "trusted": trusted_source
                })
            
            return success
            
        except Exception as e:
            self._log_security_event("circuit_file_load_failed", {"error": str(e)})
            if isinstance(e, CircuitBuilderSecurityError):
                raise
            raise CircuitBuilderSecurityError(f"Circuit file load failed: {e}")
    
    def _check_session_limits(self) -> None:
        """Check session duration and operation limits."""
        current_time = time.time()
        session_duration = current_time - self.session_start
        
        if session_duration > self.config.max_session_duration:
            raise CircuitBuilderSecurityError("Session duration limit exceeded")
        
        if self.operations_count > self.config.max_gates_per_operation:
            raise CircuitBuilderSecurityError("Operation count limit exceeded")
    
    def _validate_gate_type(self, gate_type: str) -> str:
        """Validate gate type name."""
        validated_type = self.validator.validate_and_sanitize(
            gate_type, "circuit_name", strict=False
        )
        
        # Check against allowed gate types
        allowed_gates = [
            'h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg',
            'rx', 'ry', 'rz', 'u1', 'u2', 'u3',
            'cx', 'cy', 'cz', 'ch', 'crx', 'cry', 'crz',
            'ccx', 'cswap', 'swap', 'iswap', 'measure'
        ]
        
        if validated_type.lower() not in allowed_gates and not self.config.allow_custom_gates:
            raise CircuitBuilderSecurityError(f"Gate type not allowed: {validated_type}")
        
        return validated_type
    
    def _validate_qubits(self, qubits: List[int]) -> List[int]:
        """Validate qubit indices."""
        validated_qubits = []
        for qubit in qubits:
            validated_qubit = self.validator.validate_and_sanitize(
                qubit, "qubit_index", strict=False
            )
            validated_qubits.append(validated_qubit)
        
        return validated_qubits
    
    def _validate_parameters(self, parameters: List[Any]) -> List[Any]:
        """Validate gate parameters."""
        validated_params = []
        for param in parameters:
            validated_param = self.validator.validate_and_sanitize(
                param, "gate_parameter", strict=False
            )
            validated_params.append(validated_param)
        
        return validated_params
    
    def _validate_json_input(self, json_str: str) -> Dict[str, Any]:
        """Validate JSON input."""
        try:
            # Basic validation through security validator
            validated_json = self.validator.validate_and_sanitize(
                json_str, "circuit_name", strict=False  # Will sanitize basic patterns
            )
            
            # Parse JSON
            data = json.loads(validated_json)
            
            return data
            
        except json.JSONDecodeError as e:
            raise CircuitBuilderSecurityError(f"Invalid JSON: {e}")
    
    def _validate_circuit_data(self, circuit_data: Dict[str, Any], 
                              trusted_source: bool) -> Dict[str, Any]:
        """Validate circuit data structure."""
        validated_data = {}
        
        # Validate metadata
        if 'name' in circuit_data:
            validated_data['name'] = self.validator.validate_and_sanitize(
                circuit_data['name'], "circuit_name", strict=not trusted_source
            )
        
        if 'num_qubits' in circuit_data:
            validated_data['num_qubits'] = self.validator.validate_and_sanitize(
                circuit_data['num_qubits'], "qubit_index", strict=not trusted_source
            )
        
        # Validate operations/gates
        if 'operations' in circuit_data:
            validated_operations = []
            for op in circuit_data['operations']:
                validated_op = self._validate_operation_data(op, trusted_source)
                validated_operations.append(validated_op)
            validated_data['operations'] = validated_operations
        
        return validated_data
    
    def _validate_operation_data(self, operation: Dict[str, Any], 
                                trusted_source: bool) -> Dict[str, Any]:
        """Validate individual operation data."""
        validated_op = {}
        
        if 'type' in operation:
            validated_op['type'] = self._validate_gate_type(operation['type'])
        
        if 'qubits' in operation:
            validated_op['qubits'] = self._validate_qubits(operation['qubits'])
        
        if 'parameters' in operation:
            validated_op['parameters'] = self._validate_parameters(operation['parameters'])
        
        return validated_op
    
    def _validate_export_format(self, format: str) -> str:
        """Validate export format."""
        allowed_formats = ['json', 'qasm', 'xml']
        
        validated_format = self.validator.validate_and_sanitize(
            format, "circuit_name", strict=False
        )
        
        if validated_format.lower() not in allowed_formats:
            raise CircuitBuilderSecurityError(f"Export format not allowed: {validated_format}")
        
        return validated_format.lower()
    
    def _validate_exported_data(self, data: str, format: str) -> str:
        """Validate exported data."""
        if format == 'json':
            # Validate JSON structure
            try:
                json.loads(data)
            except json.JSONDecodeError:
                raise CircuitBuilderSecurityError("Exported JSON is invalid")
        
        elif format == 'qasm':
            # Use QASM validator
            validated_data = self.validator.validate_and_sanitize(
                data, "qasm_code", strict=False
            )
            return validated_data
        
        return data
    
    def _validate_file_path(self, filepath: Path, for_write: bool) -> Path:
        """Validate file path for security."""
        try:
            resolved_path = filepath.resolve()
        except OSError:
            raise CircuitBuilderSecurityError(f"Cannot resolve path: {filepath}")
        
        # Check for path traversal
        cwd = Path.cwd()
        try:
            resolved_path.relative_to(cwd)
        except ValueError:
            raise CircuitBuilderSecurityError(f"Path outside allowed directory: {filepath}")
        
        if not for_write:
            # For reading, file must exist
            if not resolved_path.exists():
                raise CircuitBuilderSecurityError(f"File does not exist: {filepath}")
            
            if not resolved_path.is_file():
                raise CircuitBuilderSecurityError(f"Path is not a file: {filepath}")
        
        else:
            # For writing, ensure parent directory exists
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
        
        return resolved_path
    
    def _estimate_circuit_size(self, circuit_data: Dict[str, Any]) -> int:
        """Estimate circuit size from data."""
        size = 0
        
        if 'operations' in circuit_data:
            size += len(circuit_data['operations'])
        
        if 'gates' in circuit_data:
            size += len(circuit_data['gates'])
        
        return size
    
    def _log_security_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log security event."""
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'details': details
        }
        
        self.security_log.append(event)
        logger.info(f"Security event: {event_type} - {details}")
    
    def get_security_report(self) -> Dict[str, Any]:
        """Get security report for this session."""
        return {
            'session_start': self.session_start,
            'session_duration': time.time() - self.session_start,
            'operations_count': self.operations_count,
            'security_events': len(self.security_log),
            'validated_operations': len(self.validated_operations),
            'config': {
                'max_circuit_size': self.config.max_circuit_size,
                'allow_custom_gates': self.config.allow_custom_gates,
                'allow_file_operations': self.config.allow_file_operations,
                'validate_user_input': self.config.validate_user_input,
            }
        }

class SecureWebCircuitBuilder(SecureCircuitBuilder):
    """Security-hardened web-based circuit builder."""
    
    def __init__(self, config: Optional[SecureBuilderConfig] = None):
        super().__init__(config)
        
        # Additional web security
        self.session_tokens = {}
        self.rate_limits = {}
    
    def create_session(self, user_id: str) -> str:
        """Create secure session for web interface."""
        import secrets
        
        session_token = secrets.token_urlsafe(32)
        self.session_tokens[session_token] = {
            'user_id': user_id,
            'created': time.time(),
            'last_activity': time.time()
        }
        
        return session_token
    
    def validate_session(self, session_token: str) -> bool:
        """Validate session token."""
        if session_token not in self.session_tokens:
            return False
        
        session = self.session_tokens[session_token]
        current_time = time.time()
        
        # Check session expiry
        if current_time - session['created'] > self.config.max_session_duration:
            del self.session_tokens[session_token]
            return False
        
        # Update last activity
        session['last_activity'] = current_time
        return True
    
    def check_rate_limit(self, user_id: str, operation: str) -> bool:
        """Check rate limiting for user operations."""
        current_time = time.time()
        key = f"{user_id}:{operation}"
        
        if key not in self.rate_limits:
            self.rate_limits[key] = []
        
        # Clean old entries (1 minute window)
        self.rate_limits[key] = [
            t for t in self.rate_limits[key] 
            if current_time - t < 60
        ]
        
        # Check limit (max 60 operations per minute)
        if len(self.rate_limits[key]) >= 60:
            return False
        
        # Add current operation
        self.rate_limits[key].append(current_time)
        return True

# Convenience functions for secure circuit building
def create_secure_builder(config: Optional[SecureBuilderConfig] = None) -> SecureCircuitBuilder:
    """Create a secure circuit builder instance."""
    return SecureCircuitBuilder(config)

def create_secure_web_builder(config: Optional[SecureBuilderConfig] = None) -> SecureWebCircuitBuilder:
    """Create a secure web-based circuit builder instance."""
    return SecureWebCircuitBuilder(config)

# Export secure circuit builder components
__all__ = [
    "SecureCircuitBuilder",
    "SecureWebCircuitBuilder",
    "SecureBuilderConfig",
    "SecureGateInfo",
    "CircuitBuilderSecurityError",
    "create_secure_builder",
    "create_secure_web_builder",
]