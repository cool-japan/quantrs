"""
Comprehensive Input Validation for QuantRS2

This module provides production-grade input validation and sanitization
to prevent injection attacks and ensure data integrity.
"""

import re
import html
import json
import shlex
import subprocess
from typing import Any, Dict, List, Optional, Union, Callable, Pattern
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        self.field = field
        self.value = value
        super().__init__(message)

@dataclass
class ValidationRule:
    """Defines a validation rule."""
    
    name: str
    validator: Callable[[Any], bool]
    error_message: str
    sanitizer: Optional[Callable[[Any], Any]] = None

class InputValidator:
    """
    Comprehensive input validation and sanitization system.
    
    Features:
    - Command injection prevention
    - SQL injection prevention  
    - XSS prevention
    - Path traversal prevention
    - Data type validation
    - Custom validation rules
    - Automatic sanitization
    """
    
    def __init__(self):
        self._rules: Dict[str, List[ValidationRule]] = {}
        self._global_rules: List[ValidationRule] = []
        self._initialize_default_rules()
    
    def _initialize_default_rules(self) -> None:
        """Initialize default validation rules."""
        
        # Command injection prevention
        self.add_global_rule(ValidationRule(
            name="no_command_injection",
            validator=self._validate_no_command_injection,
            error_message="Input contains potentially dangerous command characters",
            sanitizer=self._sanitize_command_injection,
        ))
        
        # Path traversal prevention  
        self.add_global_rule(ValidationRule(
            name="no_path_traversal",
            validator=self._validate_no_path_traversal,
            error_message="Input contains path traversal sequences",
            sanitizer=self._sanitize_path_traversal,
        ))
        
        # SQL injection prevention
        self.add_global_rule(ValidationRule(
            name="no_sql_injection",
            validator=self._validate_no_sql_injection,
            error_message="Input contains potential SQL injection patterns",
            sanitizer=self._sanitize_sql_injection,
        ))
    
    def add_global_rule(self, rule: ValidationRule) -> None:
        """Add a global validation rule applied to all inputs."""
        self._global_rules.append(rule)
    
    def add_field_rule(self, field_name: str, rule: ValidationRule) -> None:
        """Add a validation rule for a specific field."""
        if field_name not in self._rules:
            self._rules[field_name] = []
        self._rules[field_name].append(rule)
    
    def validate_and_sanitize(
        self, 
        data: Union[str, Dict[str, Any], List[Any]], 
        field_name: Optional[str] = None,
        strict: bool = True,
    ) -> Union[str, Dict[str, Any], List[Any]]:
        """
        Validate and sanitize input data.
        
        Args:
            data: Input data to validate
            field_name: Optional field name for specific validation rules
            strict: If True, raise exceptions on validation failures
            
        Returns:
            Sanitized data
            
        Raises:
            ValidationError: If validation fails and strict=True
        """
        try:
            if isinstance(data, dict):
                return self._validate_dict(data, strict)
            elif isinstance(data, list):
                return self._validate_list(data, strict)
            else:
                return self._validate_value(data, field_name, strict)
        except Exception as e:
            if strict:
                raise ValidationError(f"Validation failed: {e}", field_name, data)
            logger.warning(f"Validation warning for {field_name}: {e}")
            return self._safe_sanitize(data)
    
    def _validate_dict(self, data: Dict[str, Any], strict: bool) -> Dict[str, Any]:
        """Validate and sanitize dictionary data."""
        sanitized = {}
        for key, value in data.items():
            # Validate key
            sanitized_key = self._validate_value(key, "dict_key", strict)
            # Validate value
            sanitized_value = self.validate_and_sanitize(value, key, strict)
            sanitized[sanitized_key] = sanitized_value
        return sanitized
    
    def _validate_list(self, data: List[Any], strict: bool) -> List[Any]:
        """Validate and sanitize list data."""
        return [self.validate_and_sanitize(item, "list_item", strict) for item in data]
    
    def _validate_value(self, value: Any, field_name: Optional[str], strict: bool) -> Any:
        """Validate and sanitize a single value."""
        if not isinstance(value, (str, int, float, bool)):
            # For non-primitive types, convert to string for validation
            if value is None:
                return None
            value = str(value)
        
        # Apply global rules
        for rule in self._global_rules:
            if not rule.validator(value):
                if strict:
                    raise ValidationError(rule.error_message, field_name, value)
                if rule.sanitizer:
                    value = rule.sanitizer(value)
        
        # Apply field-specific rules
        if field_name and field_name in self._rules:
            for rule in self._rules[field_name]:
                if not rule.validator(value):
                    if strict:
                        raise ValidationError(rule.error_message, field_name, value)
                    if rule.sanitizer:
                        value = rule.sanitizer(value)
        
        return value
    
    def _validate_no_command_injection(self, value: Any) -> bool:
        """Check for command injection patterns."""
        if not isinstance(value, str):
            return True
        
        # Dangerous patterns
        dangerous_patterns = [
            r'[;&|`$()]',  # Shell metacharacters
            r'\.\./',      # Directory traversal
            r'rm\s+',      # Remove commands
            r'eval\s*\(',  # Eval functions
            r'exec\s*\(',  # Exec functions
            r'system\s*\(',# System calls
            r'__import__', # Python imports
            r'subprocess', # Subprocess calls
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return False
        
        return True
    
    def _sanitize_command_injection(self, value: Any) -> str:
        """Sanitize command injection attempts."""
        if not isinstance(value, str):
            return str(value)
        
        # Remove or escape dangerous characters
        sanitized = re.sub(r'[;&|`$()]', '', value)
        sanitized = re.sub(r'\.\./', '', sanitized)
        sanitized = re.sub(r'\b(rm|eval|exec|system|__import__|subprocess)\b', 
                          '', sanitized, flags=re.IGNORECASE)
        
        return sanitized.strip()
    
    def _validate_no_path_traversal(self, value: Any) -> bool:
        """Check for path traversal patterns."""
        if not isinstance(value, str):
            return True
        
        dangerous_patterns = [
            r'\.\./',
            r'\.\.\\',
            r'/etc/',
            r'/proc/',
            r'/sys/',
            r'C:\\Windows',
            r'%SYSTEMROOT%',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return False
        
        return True
    
    def _sanitize_path_traversal(self, value: Any) -> str:
        """Sanitize path traversal attempts."""
        if not isinstance(value, str):
            return str(value)
        
        # Remove path traversal sequences
        sanitized = re.sub(r'\.\.[\\/]', '', value)
        sanitized = re.sub(r'[\\/]etc[\\/]', '/safe/', sanitized)
        sanitized = re.sub(r'[\\/]proc[\\/]', '/safe/', sanitized)
        sanitized = re.sub(r'[\\/]sys[\\/]', '/safe/', sanitized)
        
        return sanitized
    
    def _validate_no_sql_injection(self, value: Any) -> bool:
        """Check for SQL injection patterns."""
        if not isinstance(value, str):
            return True
        
        dangerous_patterns = [
            r"'.*(?:OR|AND).*'",
            r';\s*DROP\s+TABLE',
            r';\s*DELETE\s+FROM',
            r';\s*INSERT\s+INTO',
            r';\s*UPDATE\s+.*SET',
            r'UNION\s+SELECT',
            r'exec\s*\(',
            r'sp_executesql',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return False
        
        return True
    
    def _sanitize_sql_injection(self, value: Any) -> str:
        """Sanitize SQL injection attempts."""
        if not isinstance(value, str):
            return str(value)
        
        # Escape quotes and remove dangerous SQL keywords
        sanitized = value.replace("'", "''")  # SQL quote escaping
        sanitized = re.sub(r'\b(DROP|DELETE|INSERT|UPDATE|UNION|EXEC)\b', 
                          '', sanitized, flags=re.IGNORECASE)
        
        return sanitized.strip()
    
    def _safe_sanitize(self, value: Any) -> Any:
        """Fallback sanitization for any input."""
        if value is None:
            return None
        
        if isinstance(value, str):
            # HTML escape for XSS prevention
            sanitized = html.escape(value)
            # Remove null bytes
            sanitized = sanitized.replace('\x00', '')
            # Limit length
            if len(sanitized) > 10000:
                sanitized = sanitized[:10000] + "..."
            return sanitized
        
        return value
    
    def validate_quantum_circuit_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate quantum circuit parameters."""
        validated = {}
        
        # Validate qubit count
        if 'n_qubits' in params:
            n_qubits = params['n_qubits']
            if not isinstance(n_qubits, int) or n_qubits <= 0 or n_qubits > 1000:
                raise ValidationError("Invalid qubit count", "n_qubits", n_qubits)
            validated['n_qubits'] = n_qubits
        
        # Validate gate parameters
        if 'gates' in params:
            gates = params['gates']
            if not isinstance(gates, list):
                raise ValidationError("Gates must be a list", "gates", gates)
            
            validated_gates = []
            for i, gate in enumerate(gates):
                if not isinstance(gate, dict):
                    raise ValidationError(f"Gate {i} must be a dictionary", f"gates[{i}]", gate)
                
                # Validate gate name
                if 'name' not in gate:
                    raise ValidationError(f"Gate {i} missing name", f"gates[{i}].name", gate)
                
                gate_name = gate['name']
                if not isinstance(gate_name, str) or not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', gate_name):
                    raise ValidationError(f"Invalid gate name", f"gates[{i}].name", gate_name)
                
                # Validate qubits
                if 'qubits' in gate:
                    qubits = gate['qubits']
                    if not isinstance(qubits, list):
                        raise ValidationError(f"Gate {i} qubits must be a list", f"gates[{i}].qubits", qubits)
                    
                    for qubit in qubits:
                        if not isinstance(qubit, int) or qubit < 0:
                            raise ValidationError(f"Invalid qubit index", f"gates[{i}].qubits", qubit)
                
                validated_gates.append(gate)
            
            validated['gates'] = validated_gates
        
        return validated
    
    def validate_file_path(self, file_path: Union[str, Path], allowed_extensions: Optional[List[str]] = None) -> Path:
        """
        Validate and sanitize file paths.
        
        Args:
            file_path: Path to validate
            allowed_extensions: List of allowed file extensions
            
        Returns:
            Validated Path object
            
        Raises:
            ValidationError: If path is invalid or dangerous
        """
        if isinstance(file_path, str):
            path = Path(file_path)
        else:
            path = file_path
        
        # Check for path traversal
        try:
            resolved = path.resolve()
        except Exception as e:
            raise ValidationError(f"Invalid path: {e}", "file_path", file_path)
        
        # Ensure path doesn't escape allowed directories
        cwd = Path.cwd().resolve()
        try:
            resolved.relative_to(cwd)
        except ValueError:
            raise ValidationError("Path escapes current directory", "file_path", file_path)
        
        # Check file extension if specified
        if allowed_extensions and resolved.suffix.lower() not in allowed_extensions:
            raise ValidationError(
                f"File extension not allowed. Allowed: {allowed_extensions}",
                "file_path",
                file_path
            )
        
        return resolved
    
    def sanitize_for_shell(self, command: str) -> str:
        """
        Safely sanitize a string for shell execution.
        
        Args:
            command: Command string to sanitize
            
        Returns:
            Shell-safe command string
        """
        # Use shlex.quote for proper shell escaping
        return shlex.quote(command)
    
    def validate_json_data(self, json_str: str, max_size: int = 1024 * 1024) -> Dict[str, Any]:
        """
        Validate and parse JSON data safely.
        
        Args:
            json_str: JSON string to validate
            max_size: Maximum allowed size in bytes
            
        Returns:
            Parsed JSON data
            
        Raises:
            ValidationError: If JSON is invalid or too large
        """
        if len(json_str.encode()) > max_size:
            raise ValidationError(f"JSON data too large (max {max_size} bytes)", "json_data", len(json_str))
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON: {e}", "json_data", json_str)
        
        # Recursively validate the data
        return self.validate_and_sanitize(data)

# Global validator instance
_input_validator: Optional[InputValidator] = None

def get_input_validator() -> InputValidator:
    """Get global input validator instance."""
    global _input_validator
    if _input_validator is None:
        _input_validator = InputValidator()
    return _input_validator

def validate_input(
    data: Union[str, Dict[str, Any], List[Any]], 
    field_name: Optional[str] = None,
    strict: bool = True,
) -> Union[str, Dict[str, Any], List[Any]]:
    """Convenience function for input validation."""
    return get_input_validator().validate_and_sanitize(data, field_name, strict)