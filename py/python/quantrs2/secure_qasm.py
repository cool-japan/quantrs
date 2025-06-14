"""
Secure OpenQASM 3.0 Support for QuantRS2

This module provides security-hardened QASM parsing and export capabilities
with comprehensive input validation and sanitization.
"""

import os
import re
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass

# Import base QASM functionality
from . import qasm
from .security import validate_quantum_input, get_quantum_validator, QuantumValidationConfig

logger = logging.getLogger(__name__)

@dataclass
class SecureQasmConfig:
    """Configuration for secure QASM operations."""
    
    max_file_size: int = 10 * 1024 * 1024  # 10MB max
    max_qubits: int = 1024
    max_gates: int = 100000
    allow_includes: bool = False
    allow_external_files: bool = False
    sandbox_execution: bool = True
    validate_syntax: bool = True
    strip_comments: bool = False

class QasmSecurityError(Exception):
    """Exception raised for QASM security violations."""
    pass

class SecureQasmParser:
    """Security-hardened QASM parser with input validation."""
    
    def __init__(self, config: Optional[SecureQasmConfig] = None):
        self.config = config or SecureQasmConfig()
        self.validator = get_quantum_validator()
        self.base_parser = qasm.QasmParser()
    
    def parse(self, qasm_code: str, trusted_source: bool = False) -> 'QuantumCircuit':
        """
        Parse QASM code with security validation.
        
        Args:
            qasm_code: QASM code to parse
            trusted_source: If True, applies less strict validation
            
        Returns:
            Parsed quantum circuit
            
        Raises:
            QasmSecurityError: If security validation fails
            QasmParseError: If parsing fails
        """
        try:
            # Validate input
            validated_code = self._validate_and_sanitize_qasm(qasm_code, trusted_source)
            
            # Parse with validated code
            return self.base_parser.parse(validated_code)
            
        except QasmSecurityError:
            raise
        except Exception as e:
            logger.error(f"QASM parsing failed: {e}")
            raise qasm.QasmParseError(f"Failed to parse QASM: {e}")
    
    def parse_file(self, filepath: Union[str, Path], trusted_source: bool = False) -> 'QuantumCircuit':
        """
        Parse QASM file with security validation.
        
        Args:
            filepath: Path to QASM file
            trusted_source: If True, applies less strict validation
            
        Returns:
            Parsed quantum circuit
            
        Raises:
            QasmSecurityError: If security validation fails
        """
        filepath = Path(filepath)
        
        # Validate file path
        self._validate_file_path(filepath)
        
        # Read and validate file
        try:
            file_size = filepath.stat().st_size
            if file_size > self.config.max_file_size:
                raise QasmSecurityError(f"File too large: {file_size} bytes > {self.config.max_file_size}")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                qasm_code = f.read()
            
            return self.parse(qasm_code, trusted_source)
            
        except (OSError, IOError) as e:
            raise QasmSecurityError(f"Cannot read file {filepath}: {e}")
    
    def _validate_and_sanitize_qasm(self, qasm_code: str, trusted_source: bool) -> str:
        """Validate and sanitize QASM code for security."""
        if not isinstance(qasm_code, str):
            raise QasmSecurityError("QASM code must be a string")
        
        # Basic validation
        if len(qasm_code) > self.config.max_file_size:
            raise QasmSecurityError(f"QASM code too long: {len(qasm_code)} chars")
        
        # Use quantum validator for basic sanitization
        validated_code = self.validator.validate_and_sanitize(
            qasm_code, "qasm_code", strict=not trusted_source
        )
        
        # Additional QASM-specific validation
        validated_code = self._validate_qasm_structure(validated_code, trusted_source)
        
        return validated_code
    
    def _validate_qasm_structure(self, qasm_code: str, trusted_source: bool) -> str:
        """Validate QASM code structure and content."""
        lines = qasm_code.split('\n')
        validated_lines = []
        
        gate_count = 0
        qubit_count = 0
        found_version = False
        
        for line_num, line in enumerate(lines, 1):
            original_line = line
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('//'):
                if not self.config.strip_comments or not line.startswith('//'):
                    validated_lines.append(original_line)
                continue
            
            # Check for version declaration
            if line.startswith('OPENQASM'):
                if found_version:
                    raise QasmSecurityError(f"Multiple version declarations at line {line_num}")
                found_version = True
                validated_lines.append(original_line)
                continue
            
            # Validate include statements
            if line.startswith('include'):
                if not self.config.allow_includes or not trusted_source:
                    raise QasmSecurityError(f"Include statements not allowed at line {line_num}")
                
                # Validate include path
                include_match = re.match(r'include\s*"([^"]+)"', line)
                if include_match:
                    include_path = include_match.group(1)
                    if not self._is_safe_include_path(include_path):
                        raise QasmSecurityError(f"Unsafe include path at line {line_num}: {include_path}")
                
                validated_lines.append(original_line)
                continue
            
            # Check for qubit declarations
            qubit_match = re.match(r'qubit\[(\d+)\]', line)
            if qubit_match:
                qubits = int(qubit_match.group(1))
                qubit_count = max(qubit_count, qubits)
                
                if qubit_count > self.config.max_qubits:
                    raise QasmSecurityError(f"Too many qubits at line {line_num}: {qubit_count}")
                
                validated_lines.append(original_line)
                continue
            
            # Check for gate operations
            if self._is_gate_operation(line):
                gate_count += 1
                if gate_count > self.config.max_gates:
                    raise QasmSecurityError(f"Too many gates at line {line_num}: {gate_count}")
                
                # Validate gate parameters
                validated_line = self._validate_gate_line(line, line_num)
                validated_lines.append(validated_line)
                continue
            
            # Check for measurement operations
            if 'measure' in line.lower():
                validated_lines.append(original_line)
                continue
            
            # Check for other statements
            if self._is_safe_statement(line):
                validated_lines.append(original_line)
            else:
                if not trusted_source:
                    raise QasmSecurityError(f"Potentially unsafe statement at line {line_num}: {line}")
                validated_lines.append(original_line)
        
        return '\n'.join(validated_lines)
    
    def _validate_gate_line(self, line: str, line_num: int) -> str:
        """Validate a gate operation line."""
        # Extract gate parameters and validate them
        # This is a simplified validation - full implementation would parse the gate syntax
        
        # Look for angle parameters in parentheses
        angle_pattern = r'\(([^)]+)\)'
        angles = re.findall(angle_pattern, line)
        
        for angle_str in angles:
            # Split multiple parameters
            params = [p.strip() for p in angle_str.split(',')]
            for param in params:
                try:
                    # Try to validate as numeric or symbolic parameter
                    validated_param = self.validator.validate_and_sanitize(
                        param, "gate_parameter", strict=False
                    )
                except Exception as e:
                    raise QasmSecurityError(f"Invalid gate parameter at line {line_num}: {param}")
        
        # Look for qubit indices
        qubit_pattern = r'\[(\d+)\]'
        qubits = re.findall(qubit_pattern, line)
        
        for qubit_str in qubits:
            try:
                qubit_idx = int(qubit_str)
                validated_qubit = self.validator.validate_and_sanitize(
                    qubit_idx, "qubit_index", strict=False
                )
            except Exception as e:
                raise QasmSecurityError(f"Invalid qubit index at line {line_num}: {qubit_str}")
        
        return line
    
    def _is_gate_operation(self, line: str) -> bool:
        """Check if line contains a gate operation."""
        # Common gate names
        gate_names = [
            'h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg',
            'rx', 'ry', 'rz', 'u1', 'u2', 'u3',
            'cx', 'cy', 'cz', 'ch', 'crx', 'cry', 'crz',
            'ccx', 'cswap', 'swap', 'iswap'
        ]
        
        for gate in gate_names:
            if re.match(rf'\b{gate}\b', line, re.IGNORECASE):
                return True
        
        return False
    
    def _is_safe_statement(self, line: str) -> bool:
        """Check if a statement is safe to execute."""
        # Allow common QASM statements
        safe_patterns = [
            r'^qubit\[',
            r'^bit\[',
            r'^creg\s+',
            r'^qreg\s+',
            r'^barrier\b',
            r'^reset\b',
            r'^gate\s+\w+',
            r'^\w+\s*\(',  # Custom gate calls
        ]
        
        for pattern in safe_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        
        return False
    
    def _is_safe_include_path(self, path: str) -> bool:
        """Check if include path is safe."""
        # Prevent path traversal
        if '..' in path or path.startswith('/'):
            return False
        
        # Only allow standard library includes
        safe_includes = [
            'stdgates.inc',
            'qelib1.inc',
        ]
        
        return path in safe_includes
    
    def _validate_file_path(self, filepath: Path) -> None:
        """Validate file path for security."""
        # Resolve path to prevent traversal
        try:
            resolved_path = filepath.resolve()
        except OSError:
            raise QasmSecurityError(f"Cannot resolve path: {filepath}")
        
        # Check if file exists and is readable
        if not resolved_path.exists():
            raise QasmSecurityError(f"File does not exist: {filepath}")
        
        if not resolved_path.is_file():
            raise QasmSecurityError(f"Path is not a file: {filepath}")
        
        # Prevent access to system files if not explicitly allowed
        if not self.config.allow_external_files:
            # Only allow files in current directory or subdirectories
            cwd = Path.cwd()
            try:
                resolved_path.relative_to(cwd)
            except ValueError:
                raise QasmSecurityError(f"File outside allowed directory: {filepath}")

class SecureQasmExporter:
    """Security-hardened QASM exporter."""
    
    def __init__(self, options: Optional[qasm.QasmExportOptions] = None,
                 config: Optional[SecureQasmConfig] = None):
        self.options = options or qasm.QasmExportOptions()
        self.config = config or SecureQasmConfig()
        self.validator = get_quantum_validator()
        self.base_exporter = qasm.QasmExporter(options)
    
    def export(self, circuit: 'QuantumCircuit', validate_output: bool = True) -> str:
        """
        Export circuit to QASM with optional output validation.
        
        Args:
            circuit: Quantum circuit to export
            validate_output: Whether to validate the exported QASM
            
        Returns:
            QASM code string
        """
        try:
            # Export using base exporter
            qasm_code = self.base_exporter.export(circuit)
            
            # Validate output if requested
            if validate_output:
                qasm_code = self._validate_exported_qasm(qasm_code)
            
            return qasm_code
            
        except Exception as e:
            logger.error(f"QASM export failed: {e}")
            raise qasm.QasmExportError(f"Failed to export QASM: {e}")
    
    def export_to_file(self, circuit: 'QuantumCircuit', filepath: Union[str, Path],
                      validate_output: bool = True) -> None:
        """
        Export circuit to QASM file with security validation.
        
        Args:
            circuit: Quantum circuit to export
            filepath: Output file path
            validate_output: Whether to validate the exported QASM
        """
        filepath = Path(filepath)
        
        # Validate output path
        self._validate_output_path(filepath)
        
        # Export QASM
        qasm_code = self.export(circuit, validate_output)
        
        # Write to file securely
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(qasm_code)
        except (OSError, IOError) as e:
            raise QasmSecurityError(f"Cannot write to file {filepath}: {e}")
    
    def _validate_exported_qasm(self, qasm_code: str) -> str:
        """Validate exported QASM code."""
        # Use validator to sanitize output
        validated_code = self.validator.validate_and_sanitize(
            qasm_code, "qasm_code", strict=False
        )
        
        return validated_code
    
    def _validate_output_path(self, filepath: Path) -> None:
        """Validate output file path."""
        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if we can write to the location
        if filepath.exists() and not filepath.is_file():
            raise QasmSecurityError(f"Output path is not a file: {filepath}")

# Convenience functions with default security settings
def secure_parse_qasm(qasm_code: str, trusted_source: bool = False) -> 'QuantumCircuit':
    """Parse QASM code with security validation."""
    parser = SecureQasmParser()
    return parser.parse(qasm_code, trusted_source)

def secure_parse_qasm_file(filepath: Union[str, Path], trusted_source: bool = False) -> 'QuantumCircuit':
    """Parse QASM file with security validation."""
    parser = SecureQasmParser()
    return parser.parse_file(filepath, trusted_source)

def secure_export_qasm(circuit: 'QuantumCircuit', validate_output: bool = True) -> str:
    """Export circuit to QASM with security validation."""
    exporter = SecureQasmExporter()
    return exporter.export(circuit, validate_output)

def secure_export_qasm_file(circuit: 'QuantumCircuit', filepath: Union[str, Path],
                           validate_output: bool = True) -> None:
    """Export circuit to QASM file with security validation."""
    exporter = SecureQasmExporter()
    exporter.export_to_file(circuit, filepath, validate_output)

# Export security-hardened components
__all__ = [
    "SecureQasmParser",
    "SecureQasmExporter", 
    "SecureQasmConfig",
    "QasmSecurityError",
    "secure_parse_qasm",
    "secure_parse_qasm_file",
    "secure_export_qasm",
    "secure_export_qasm_file",
]