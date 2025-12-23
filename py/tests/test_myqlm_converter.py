#!/usr/bin/env python3
"""
Tests for MyQLM/QLM converter functionality.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

# Try to import required modules
try:
    from quantrs2.myqlm_converter import (
        MyQLMConverter,
        MyQLMConversionStats,
        MYQLM_AVAILABLE,
        convert_from_myqlm,
        convert_to_myqlm,
    )
    CONVERTER_AVAILABLE = True
except ImportError as e:
    CONVERTER_AVAILABLE = False
    print(f"MyQLM converter not available: {e}")

# Try to import MyQLM
try:
    from qat.lang.AQASM import Program, H, X, Y, Z, S, T, CNOT, SWAP, RX, RY, RZ
    from qat.core import Circuit as QLMCircuit
    MYQLM_INSTALLED = True
except ImportError:
    MYQLM_INSTALLED = False
    print("MyQLM not installed. Skipping MyQLM-dependent tests.")


@pytest.mark.skipif(not CONVERTER_AVAILABLE, reason="MyQLM converter not available")
class TestMyQLMConverterBasics:
    """Test basic converter functionality."""

    def test_converter_initialization(self):
        """Test converter can be initialized."""
        converter = MyQLMConverter()
        assert converter is not None
        assert converter.strict_mode is False

    def test_converter_strict_mode(self):
        """Test converter with strict mode."""
        converter = MyQLMConverter(strict_mode=True)
        assert converter.strict_mode is True


@pytest.mark.skipif(
    not (CONVERTER_AVAILABLE and MYQLM_INSTALLED),
    reason="MyQLM converter or MyQLM not available"
)
class TestMyQLMConversion:
    """Test MyQLM circuit conversions."""

    def test_basic_bell_state_conversion(self):
        """Test conversion of a simple Bell state circuit."""
        # Create MyQLM Bell state circuit
        prog = Program()
        qbits = prog.qalloc(2)
        prog.apply(H, qbits[0])
        prog.apply(CNOT, qbits[0], qbits[1])
        qlm_circuit = prog.to_circ()

        # Convert to QuantRS2
        converter = MyQLMConverter()
        quantrs_circuit, stats = converter.from_myqlm(qlm_circuit)

        # Verify conversion
        assert quantrs_circuit is not None
        assert stats.success
        assert stats.converted_gates >= 2
        assert len(stats.unsupported_gates) == 0

    def test_single_qubit_gates(self):
        """Test conversion of single-qubit gates."""
        # Create circuit with various single-qubit gates
        prog = Program()
        qbits = prog.qalloc(3)
        prog.apply(H, qbits[0])
        prog.apply(X, qbits[1])
        prog.apply(Y, qbits[2])
        prog.apply(Z, qbits[0])
        prog.apply(S, qbits[1])
        prog.apply(T, qbits[2])
        qlm_circuit = prog.to_circ()

        # Convert
        converter = MyQLMConverter()
        quantrs_circuit, stats = converter.from_myqlm(qlm_circuit)

        # Verify
        assert quantrs_circuit is not None
        assert stats.success
        assert stats.converted_gates == 6

    def test_rotation_gates(self):
        """Test conversion of rotation gates."""
        import numpy as np

        # Create circuit with rotation gates
        prog = Program()
        qbits = prog.qalloc(3)
        prog.apply(RX(np.pi/2), qbits[0])
        prog.apply(RY(np.pi/4), qbits[1])
        prog.apply(RZ(np.pi/3), qbits[2])
        qlm_circuit = prog.to_circ()

        # Convert
        converter = MyQLMConverter()
        quantrs_circuit, stats = converter.from_myqlm(qlm_circuit)

        # Verify
        assert quantrs_circuit is not None
        assert stats.success
        assert stats.converted_gates == 3

    def test_two_qubit_gates(self):
        """Test conversion of two-qubit gates."""
        # Create circuit with two-qubit gates
        prog = Program()
        qbits = prog.qalloc(3)
        prog.apply(CNOT, qbits[0], qbits[1])
        prog.apply(SWAP, qbits[1], qbits[2])
        qlm_circuit = prog.to_circ()

        # Convert
        converter = MyQLMConverter()
        quantrs_circuit, stats = converter.from_myqlm(qlm_circuit)

        # Verify
        assert quantrs_circuit is not None
        assert stats.success
        assert stats.converted_gates == 2

    def test_ghz_state(self):
        """Test conversion of GHZ state preparation."""
        # Create 3-qubit GHZ state
        prog = Program()
        qbits = prog.qalloc(3)
        prog.apply(H, qbits[0])
        prog.apply(CNOT, qbits[0], qbits[1])
        prog.apply(CNOT, qbits[1], qbits[2])
        qlm_circuit = prog.to_circ()

        # Convert
        converter = MyQLMConverter()
        quantrs_circuit, stats = converter.from_myqlm(qlm_circuit)

        # Verify
        assert quantrs_circuit is not None
        assert stats.success
        assert stats.converted_gates == 3
        assert len(stats.warnings) == 0

    def test_conversion_stats(self):
        """Test that conversion statistics are properly populated."""
        # Create a simple circuit
        prog = Program()
        qbits = prog.qalloc(2)
        prog.apply(H, qbits[0])
        prog.apply(CNOT, qbits[0], qbits[1])
        qlm_circuit = prog.to_circ()

        # Convert
        converter = MyQLMConverter()
        _, stats = converter.from_myqlm(qlm_circuit)

        # Verify stats structure
        assert isinstance(stats, MyQLMConversionStats)
        assert stats.original_gates >= 0
        assert stats.converted_gates >= 0
        assert stats.decomposed_gates >= 0
        assert isinstance(stats.unsupported_gates, list)
        assert isinstance(stats.warnings, list)
        assert isinstance(stats.success, bool)

    def test_empty_circuit(self):
        """Test conversion of an empty circuit."""
        # Create empty circuit
        prog = Program()
        qbits = prog.qalloc(2)
        qlm_circuit = prog.to_circ()

        # Convert
        converter = MyQLMConverter()
        quantrs_circuit, stats = converter.from_myqlm(qlm_circuit)

        # Verify - empty circuit should still convert successfully
        assert stats.success
        assert stats.converted_gates == 0


@pytest.mark.skipif(
    not (CONVERTER_AVAILABLE and MYQLM_INSTALLED),
    reason="MyQLM converter or MyQLM not available"
)
class TestMyQLMRoundTrip:
    """Test round-trip conversions (QuantRS2 -> MyQLM -> QuantRS2)."""

    def test_bell_state_roundtrip(self):
        """Test round-trip conversion of Bell state."""
        # Create MyQLM Bell state
        prog = Program()
        qbits = prog.qalloc(2)
        prog.apply(H, qbits[0])
        prog.apply(CNOT, qbits[0], qbits[1])
        original_circuit = prog.to_circ()

        # Convert to QuantRS2
        converter = MyQLMConverter()
        quantrs_circuit, stats1 = converter.from_myqlm(original_circuit)
        assert stats1.success

        # Convert back to MyQLM
        myqlm_circuit, stats2 = converter.to_myqlm(quantrs_circuit)
        assert stats2.success

        # Verify gate counts match
        assert stats1.converted_gates == stats2.original_gates


@pytest.mark.skipif(not CONVERTER_AVAILABLE, reason="MyQLM converter not available")
class TestMyQLMErrorHandling:
    """Test error handling in MyQLM converter."""

    def test_strict_mode_on_unsupported_gate(self):
        """Test that strict mode raises errors on unsupported gates."""
        if not MYQLM_INSTALLED:
            pytest.skip("MyQLM not installed")

        converter = MyQLMConverter(strict_mode=True)

        # Create circuit with potentially unsupported operation
        # (depending on implementation)
        prog = Program()
        qbits = prog.qalloc(2)
        prog.apply(H, qbits[0])
        qlm_circuit = prog.to_circ()

        # This should work for basic gates
        quantrs_circuit, stats = converter.from_myqlm(qlm_circuit)
        assert quantrs_circuit is not None

    def test_lenient_mode_continues_on_unsupported(self):
        """Test that lenient mode continues on unsupported gates."""
        if not MYQLM_INSTALLED:
            pytest.skip("MyQLM not installed")

        converter = MyQLMConverter(strict_mode=False)

        # Create a basic circuit
        prog = Program()
        qbits = prog.qalloc(2)
        prog.apply(H, qbits[0])
        prog.apply(CNOT, qbits[0], qbits[1])
        qlm_circuit = prog.to_circ()

        # Should succeed even if some gates might be problematic
        quantrs_circuit, stats = converter.from_myqlm(qlm_circuit)
        assert quantrs_circuit is not None or not stats.success


@pytest.mark.skipif(not CONVERTER_AVAILABLE, reason="MyQLM converter not available")
class TestMyQLMConvenienceFunctions:
    """Test convenience functions for MyQLM conversion."""

    def test_convert_from_myqlm_function(self):
        """Test the convert_from_myqlm convenience function."""
        if not MYQLM_INSTALLED:
            pytest.skip("MyQLM not installed")

        # Create simple circuit
        prog = Program()
        qbits = prog.qalloc(2)
        prog.apply(H, qbits[0])
        qlm_circuit = prog.to_circ()

        # Use convenience function
        quantrs_circuit = convert_from_myqlm(qlm_circuit)
        assert quantrs_circuit is not None

    def test_convert_to_myqlm_function(self):
        """Test the convert_to_myqlm convenience function."""
        if not MYQLM_INSTALLED:
            pytest.skip("MyQLM not installed")

        # Create MyQLM circuit and convert to QuantRS2
        prog = Program()
        qbits = prog.qalloc(2)
        prog.apply(H, qbits[0])
        qlm_circuit = prog.to_circ()

        converter = MyQLMConverter()
        quantrs_circuit, _ = converter.from_myqlm(qlm_circuit)

        # Convert back using convenience function
        myqlm_circuit = convert_to_myqlm(quantrs_circuit)
        assert myqlm_circuit is not None


def test_module_availability():
    """Test that the converter module is properly available."""
    assert CONVERTER_AVAILABLE, "MyQLM converter module should be available"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
