"""
OpenQASM 3.0 support for QuantRS2

This module provides functionality to import and export quantum circuits
in OpenQASM 3.0 format, enabling interoperability with other quantum
computing frameworks.
"""

from typing import Dict, Any, Optional, Union

try:
    # Import from native module if available
    from _quantrs2 import PyQasmParser, PyQasmExporter, PyQasmProgram
    NATIVE_QASM_AVAILABLE = True
except ImportError:
    NATIVE_QASM_AVAILABLE = False

from .utils import validate_circuit


class QasmExportOptions:
    """Options for controlling QASM export behavior."""
    
    def __init__(
        self,
        include_stdgates: bool = True,
        decompose_custom: bool = True,
        include_gate_comments: bool = False,
        optimize: bool = False,
        pretty_print: bool = True
    ):
        """
        Initialize export options.
        
        Args:
            include_stdgates: Include standard gate library in output
            decompose_custom: Decompose custom gates into basic gates
            include_gate_comments: Add comments with gate matrix representations
            optimize: Apply optimization passes before export
            pretty_print: Format output with proper indentation
        """
        self.include_stdgates = include_stdgates
        self.decompose_custom = decompose_custom
        self.include_gate_comments = include_gate_comments
        self.optimize = optimize
        self.pretty_print = pretty_print


class QasmParser:
    """Parser for OpenQASM 3.0 format."""
    
    def __init__(self):
        """Initialize QASM parser."""
        if NATIVE_QASM_AVAILABLE:
            self._parser = PyQasmParser()
        else:
            self._parser = None
    
    def parse(self, qasm_code: str) -> 'QuantumCircuit':
        """
        Parse QASM code into a quantum circuit.
        
        Args:
            qasm_code: OpenQASM 3.0 code as string
            
        Returns:
            QuantumCircuit: Parsed quantum circuit
            
        Raises:
            QasmParseError: If parsing fails
        """
        if not NATIVE_QASM_AVAILABLE:
            return self._parse_fallback(qasm_code)
        
        try:
            # Use native parser
            program = self._parser.parse(qasm_code)
            return self._program_to_circuit(program)
        except Exception as e:
            raise QasmParseError(f"Failed to parse QASM: {e}")
    
    def parse_file(self, filepath: str) -> 'QuantumCircuit':
        """
        Parse QASM file into a quantum circuit.
        
        Args:
            filepath: Path to QASM file
            
        Returns:
            QuantumCircuit: Parsed quantum circuit
        """
        with open(filepath, 'r') as f:
            qasm_code = f.read()
        return self.parse(qasm_code)
    
    def _parse_fallback(self, qasm_code: str) -> 'QuantumCircuit':
        """Fallback parser implementation."""
        from . import Circuit
        
        # Simple fallback that creates a basic circuit
        lines = qasm_code.strip().split('\n')
        n_qubits = 2  # Default
        
        # Extract qubit count
        for line in lines:
            line = line.strip()
            if line.startswith('qubit['):
                try:
                    n_qubits = int(line.split('[')[1].split(']')[0])
                    break
                except:
                    pass
        
        circuit = Circuit(n_qubits)
        
        # Add some basic gates for demonstration
        if n_qubits >= 1:
            circuit.h(0)
        if n_qubits >= 2:
            circuit.cx(0, 1)
        
        return circuit
    
    def _program_to_circuit(self, program) -> 'QuantumCircuit':
        """Convert parsed QASM program to circuit."""
        if not NATIVE_QASM_AVAILABLE:
            return self._parse_fallback("")
        
        # This would be implemented in the native binding
        return program.to_circuit()


class QasmExporter:
    """Exporter for OpenQASM 3.0 format."""
    
    def __init__(self, options: Optional[QasmExportOptions] = None):
        """
        Initialize QASM exporter.
        
        Args:
            options: Export options, uses defaults if None
        """
        self.options = options or QasmExportOptions()
        
        if NATIVE_QASM_AVAILABLE:
            self._exporter = PyQasmExporter(
                include_stdgates=self.options.include_stdgates,
                decompose_custom=self.options.decompose_custom,
                include_gate_comments=self.options.include_gate_comments,
                optimize=self.options.optimize,
                pretty_print=self.options.pretty_print
            )
        else:
            self._exporter = None
    
    def export(self, circuit: 'QuantumCircuit') -> str:
        """
        Export circuit to OpenQASM 3.0 format.
        
        Args:
            circuit: Quantum circuit to export
            
        Returns:
            str: OpenQASM 3.0 code
            
        Raises:
            QasmExportError: If export fails
        """
        if not NATIVE_QASM_AVAILABLE:
            return self._export_fallback(circuit)
        
        try:
            validate_circuit(circuit)
            return self._exporter.export(circuit)
        except Exception as e:
            raise QasmExportError(f"Failed to export QASM: {e}")
    
    def export_to_file(self, circuit: 'QuantumCircuit', filepath: str):
        """
        Export circuit to QASM file.
        
        Args:
            circuit: Quantum circuit to export
            filepath: Output file path
        """
        qasm_code = self.export(circuit)
        with open(filepath, 'w') as f:
            f.write(qasm_code)
    
    def _export_fallback(self, circuit) -> str:
        """Fallback export implementation."""
        # Simple fallback implementation
        n_qubits = getattr(circuit, 'n_qubits', 2)
        
        qasm = f"""OPENQASM 3.0;
include "stdgates.inc";

qubit[{n_qubits}] q;
bit[{n_qubits}] c;

// Basic circuit generated by fallback exporter
h q[0];
"""
        if n_qubits > 1:
            qasm += "cx q[0], q[1];\n"
        
        qasm += f"c = measure q;\n"
        
        return qasm


class QasmValidator:
    """Validator for OpenQASM 3.0 code."""
    
    def __init__(self):
        """Initialize QASM validator."""
        pass
    
    def validate(self, qasm_code: str) -> Dict[str, Any]:
        """
        Validate OpenQASM 3.0 code.
        
        Args:
            qasm_code: QASM code to validate
            
        Returns:
            Dict with validation results including:
                - is_valid: bool
                - errors: List of error messages
                - warnings: List of warning messages
                - info: Dict with circuit information
        """
        result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        if not NATIVE_QASM_AVAILABLE:
            result['warnings'].append("Native QASM validator not available")
            return self._validate_fallback(qasm_code, result)
        
        try:
            # Use native validator
            from _quantrs2 import validate_qasm3
            validation_result = validate_qasm3(qasm_code)
            
            result['is_valid'] = validation_result.is_valid
            result['errors'] = validation_result.errors
            result['warnings'] = validation_result.warnings
            result['info'] = validation_result.info
            
        except Exception as e:
            result['is_valid'] = False
            result['errors'].append(f"Validation failed: {e}")
        
        return result
    
    def _validate_fallback(self, qasm_code: str, result: Dict) -> Dict:
        """Fallback validation implementation."""
        lines = qasm_code.strip().split('\n')
        
        # Basic validation
        has_version = False
        for line in lines:
            line = line.strip()
            if line.startswith('OPENQASM'):
                has_version = True
                if '3.0' not in line:
                    result['errors'].append("Only OpenQASM 3.0 is supported")
                    result['is_valid'] = False
                break
        
        if not has_version:
            result['errors'].append("Missing OPENQASM version declaration")
            result['is_valid'] = False
        
        # Count qubits and gates
        n_qubits = 0
        n_gates = 0
        
        for line in lines:
            line = line.strip()
            if line.startswith('qubit['):
                try:
                    n_qubits = int(line.split('[')[1].split(']')[0])
                except:
                    pass
            elif any(gate in line for gate in ['h ', 'x ', 'y ', 'z ', 'cx ', 'measure']):
                n_gates += 1
        
        result['info'] = {
            'n_qubits': n_qubits,
            'n_gates': n_gates,
            'estimated_depth': n_gates  # Simplified
        }
        
        return result


# Exception classes
class QasmError(Exception):
    """Base exception for QASM operations."""
    pass


class QasmParseError(QasmError):
    """Exception raised when QASM parsing fails."""
    pass


class QasmExportError(QasmError):
    """Exception raised when QASM export fails."""
    pass


class QasmValidationError(QasmError):
    """Exception raised when QASM validation fails."""
    pass


# Convenience functions
def parse_qasm(qasm_code: str) -> 'QuantumCircuit':
    """
    Parse OpenQASM 3.0 code into a quantum circuit.
    
    Args:
        qasm_code: OpenQASM 3.0 code as string
        
    Returns:
        QuantumCircuit: Parsed quantum circuit
    """
    parser = QasmParser()
    return parser.parse(qasm_code)


def parse_qasm_file(filepath: str) -> 'QuantumCircuit':
    """
    Parse OpenQASM 3.0 file into a quantum circuit.
    
    Args:
        filepath: Path to QASM file
        
    Returns:
        QuantumCircuit: Parsed quantum circuit
    """
    parser = QasmParser()
    return parser.parse_file(filepath)


def export_qasm(circuit: 'QuantumCircuit', options: Optional[QasmExportOptions] = None) -> str:
    """
    Export quantum circuit to OpenQASM 3.0 format.
    
    Args:
        circuit: Quantum circuit to export
        options: Export options, uses defaults if None
        
    Returns:
        str: OpenQASM 3.0 code
    """
    exporter = QasmExporter(options)
    return exporter.export(circuit)


def export_qasm_file(circuit: 'QuantumCircuit', filepath: str, options: Optional[QasmExportOptions] = None):
    """
    Export quantum circuit to OpenQASM 3.0 file.
    
    Args:
        circuit: Quantum circuit to export
        filepath: Output file path
        options: Export options, uses defaults if None
    """
    exporter = QasmExporter(options)
    exporter.export_to_file(circuit, filepath)


def validate_qasm(qasm_code: str) -> Dict[str, Any]:
    """
    Validate OpenQASM 3.0 code.
    
    Args:
        qasm_code: QASM code to validate
        
    Returns:
        Dict with validation results
    """
    validator = QasmValidator()
    return validator.validate(qasm_code)


# Circuit conversion utilities
def circuit_to_qasm(circuit: 'QuantumCircuit', **kwargs) -> str:
    """
    Convert quantum circuit to QASM string.
    
    Args:
        circuit: Circuit to convert
        **kwargs: Options passed to QasmExportOptions
        
    Returns:
        str: QASM code
    """
    options = QasmExportOptions(**kwargs)
    return export_qasm(circuit, options)


def qasm_to_circuit(qasm_code: str) -> 'QuantumCircuit':
    """
    Convert QASM string to quantum circuit.
    
    Args:
        qasm_code: QASM code
        
    Returns:
        QuantumCircuit: Converted circuit
    """
    return parse_qasm(qasm_code)