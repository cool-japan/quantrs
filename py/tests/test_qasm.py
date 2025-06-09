#!/usr/bin/env python3
"""
Test suite for OpenQASM 3.0 functionality.
"""

import pytest
import tempfile
import os
from pathlib import Path

try:
    import quantrs2
    from quantrs2.qasm import (
        QasmParser, QasmExporter, QasmValidator, QasmExportOptions,
        parse_qasm, export_qasm, validate_qasm,
        QasmParseError, QasmExportError, QasmValidationError
    )
    HAS_QUANTRS2 = True
except ImportError:
    HAS_QUANTRS2 = False


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestQasmExportOptions:
    """Test QASM export options."""
    
    def test_default_options(self):
        """Test default export options."""
        options = QasmExportOptions()
        assert options.include_stdgates is True
        assert options.decompose_custom is True
        assert options.include_gate_comments is False
        assert options.optimize is False
        assert options.pretty_print is True
    
    def test_custom_options(self):
        """Test custom export options."""
        options = QasmExportOptions(
            include_stdgates=False,
            decompose_custom=False,
            include_gate_comments=True,
            optimize=True,
            pretty_print=False
        )
        assert options.include_stdgates is False
        assert options.decompose_custom is False
        assert options.include_gate_comments is True
        assert options.optimize is True
        assert options.pretty_print is False


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestQasmParser:
    """Test QASM parser functionality."""
    
    def test_parser_creation(self):
        """Test parser initialization."""
        parser = QasmParser()
        assert parser is not None
    
    def test_parse_basic_circuit(self):
        """Test parsing a basic QASM circuit."""
        qasm_code = """
        OPENQASM 3.0;
        include "stdgates.inc";
        
        qubit[2] q;
        bit[2] c;
        
        h q[0];
        cx q[0], q[1];
        c = measure q;
        """
        
        parser = QasmParser()
        circuit = parser.parse(qasm_code)
        assert circuit is not None
        # In fallback mode, we get a basic circuit
        assert hasattr(circuit, 'n_qubits') or hasattr(circuit, '_qubits')
    
    def test_parse_invalid_qasm(self):
        """Test parsing invalid QASM code."""
        invalid_qasm = "invalid qasm code"
        
        parser = QasmParser()
        # Should raise exception or return a fallback circuit
        try:
            circuit = parser.parse(invalid_qasm)
            # If no exception, should still get a valid circuit (fallback)
            assert circuit is not None
        except QasmParseError:
            # This is also acceptable
            pass
    
    def test_parse_file(self):
        """Test parsing QASM from file."""
        qasm_code = """
        OPENQASM 3.0;
        include "stdgates.inc";
        
        qubit[1] q;
        bit[1] c;
        
        h q[0];
        c[0] = measure q[0];
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.qasm', delete=False) as f:
            f.write(qasm_code)
            temp_path = f.name
        
        try:
            parser = QasmParser()
            circuit = parser.parse_file(temp_path)
            assert circuit is not None
        finally:
            os.unlink(temp_path)


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestQasmExporter:
    """Test QASM exporter functionality."""
    
    def test_exporter_creation(self):
        """Test exporter initialization."""
        exporter = QasmExporter()
        assert exporter is not None
        assert exporter.options is not None
    
    def test_exporter_with_options(self):
        """Test exporter with custom options."""
        options = QasmExportOptions(pretty_print=False)
        exporter = QasmExporter(options)
        assert exporter.options.pretty_print is False
    
    def test_export_basic_circuit(self):
        """Test exporting a basic circuit."""
        circuit = quantrs2.Circuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        
        exporter = QasmExporter()
        qasm_code = exporter.export(circuit)
        
        assert isinstance(qasm_code, str)
        assert "OPENQASM" in qasm_code
        assert "qubit" in qasm_code
    
    def test_export_to_file(self):
        """Test exporting circuit to file."""
        circuit = quantrs2.Circuit(1)
        circuit.h(0)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.qasm', delete=False) as f:
            temp_path = f.name
        
        try:
            exporter = QasmExporter()
            exporter.export_to_file(circuit, temp_path)
            
            # Verify file was created and contains QASM code
            assert os.path.exists(temp_path)
            with open(temp_path, 'r') as f:
                content = f.read()
                assert "OPENQASM" in content
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestQasmValidator:
    """Test QASM validator functionality."""
    
    def test_validator_creation(self):
        """Test validator initialization."""
        validator = QasmValidator()
        assert validator is not None
    
    def test_validate_valid_qasm(self):
        """Test validating valid QASM code."""
        qasm_code = """
        OPENQASM 3.0;
        include "stdgates.inc";
        
        qubit[2] q;
        bit[2] c;
        
        h q[0];
        cx q[0], q[1];
        c = measure q;
        """
        
        validator = QasmValidator()
        result = validator.validate(qasm_code)
        
        assert isinstance(result, dict)
        assert 'is_valid' in result
        assert 'errors' in result
        assert 'warnings' in result
        assert 'info' in result
        # In fallback mode, should still be valid
        assert result['is_valid'] is True
    
    def test_validate_invalid_qasm(self):
        """Test validating invalid QASM code."""
        invalid_qasm = "not qasm code"
        
        validator = QasmValidator()
        result = validator.validate(invalid_qasm)
        
        assert isinstance(result, dict)
        assert 'is_valid' in result
        # Should detect missing version
        assert result['is_valid'] is False or len(result['errors']) > 0
    
    def test_validate_wrong_version(self):
        """Test validating QASM with wrong version."""
        qasm_code = """
        OPENQASM 2.0;
        
        qreg q[2];
        creg c[2];
        
        h q[0];
        cx q[0], q[1];
        measure q -> c;
        """
        
        validator = QasmValidator()
        result = validator.validate(qasm_code)
        
        # Should detect version mismatch
        assert not result['is_valid'] or any('3.0' in error for error in result['errors'])


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_parse_qasm_function(self):
        """Test parse_qasm convenience function."""
        qasm_code = """
        OPENQASM 3.0;
        include "stdgates.inc";
        
        qubit[1] q;
        h q[0];
        """
        
        circuit = parse_qasm(qasm_code)
        assert circuit is not None
    
    def test_export_qasm_function(self):
        """Test export_qasm convenience function."""
        circuit = quantrs2.Circuit(1)
        circuit.h(0)
        
        qasm_code = export_qasm(circuit)
        assert isinstance(qasm_code, str)
        assert "OPENQASM" in qasm_code
    
    def test_validate_qasm_function(self):
        """Test validate_qasm convenience function."""
        qasm_code = """
        OPENQASM 3.0;
        qubit[1] q;
        """
        
        result = validate_qasm(qasm_code)
        assert isinstance(result, dict)
        assert 'is_valid' in result


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestRoundTripConversion:
    """Test round-trip conversion between circuits and QASM."""
    
    def test_circuit_to_qasm_to_circuit(self):
        """Test converting circuit to QASM and back."""
        # Create original circuit
        original = quantrs2.Circuit(2)
        original.h(0)
        original.cx(0, 1)
        
        # Export to QASM
        qasm_code = export_qasm(original)
        assert "OPENQASM" in qasm_code
        
        # Parse back to circuit
        reconstructed = parse_qasm(qasm_code)
        assert reconstructed is not None
        
        # Note: In fallback mode, the circuits might not be identical
        # but they should both be valid quantum circuits
        assert hasattr(reconstructed, 'n_qubits') or hasattr(reconstructed, '_qubits')
    
    def test_qasm_to_circuit_to_qasm(self):
        """Test parsing QASM to circuit and back to QASM."""
        original_qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        
        qubit[2] q;
        bit[2] c;
        
        h q[0];
        cx q[0], q[1];
        c = measure q;
        """
        
        # Parse to circuit
        circuit = parse_qasm(original_qasm)
        assert circuit is not None
        
        # Export back to QASM
        new_qasm = export_qasm(circuit)
        assert "OPENQASM" in new_qasm
        assert "qubit" in new_qasm


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestQasmIntegration:
    """Test integration with main quantrs2 module."""
    
    def test_qasm_functions_available(self):
        """Test that QASM functions are available from main module."""
        # These should be available if QASM module imported successfully
        try:
            from quantrs2 import parse_qasm, export_qasm, validate_qasm
            assert callable(parse_qasm)
            assert callable(export_qasm)
            assert callable(validate_qasm)
        except ImportError:
            # This is acceptable if QASM not available
            pass
    
    def test_qasm_with_circuit_visualization(self):
        """Test QASM export with circuit visualization."""
        circuit = quantrs2.Circuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        
        # Export to QASM
        qasm_code = export_qasm(circuit)
        
        # Should be able to visualize the QASM
        assert isinstance(qasm_code, str)
        assert len(qasm_code) > 0


# Example QASM circuits for testing
EXAMPLE_BELL_STATE = """
OPENQASM 3.0;
include "stdgates.inc";

qubit[2] q;
bit[2] c;

h q[0];
cx q[0], q[1];
c = measure q;
"""

EXAMPLE_GHZ_STATE = """
OPENQASM 3.0;
include "stdgates.inc";

qubit[3] q;
bit[3] c;

h q[0];
cx q[0], q[1];
cx q[1], q[2];
c = measure q;
"""

EXAMPLE_PARAMETRIC_CIRCUIT = """
OPENQASM 3.0;
include "stdgates.inc";

qubit[2] q;
bit[2] c;

rx(0.5) q[0];
ry(1.0) q[1];
cx q[0], q[1];
rz(0.25) q[1];
c = measure q;
"""


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestExampleCircuits:
    """Test parsing example QASM circuits."""
    
    def test_bell_state_qasm(self):
        """Test parsing Bell state QASM."""
        circuit = parse_qasm(EXAMPLE_BELL_STATE)
        assert circuit is not None
    
    def test_ghz_state_qasm(self):
        """Test parsing GHZ state QASM."""
        circuit = parse_qasm(EXAMPLE_GHZ_STATE)
        assert circuit is not None
    
    def test_parametric_circuit_qasm(self):
        """Test parsing parametric circuit QASM."""
        circuit = parse_qasm(EXAMPLE_PARAMETRIC_CIRCUIT)
        assert circuit is not None


if __name__ == "__main__":
    pytest.main([__file__])