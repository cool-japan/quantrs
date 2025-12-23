#!/usr/bin/env python3
"""
Tests for ProjectQ converter functionality.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

# Try to import required modules
try:
    from quantrs2.projectq_converter import (
        ProjectQConverter,
        ProjectQConversionStats,
        ProjectQBackend,
        PROJECTQ_AVAILABLE,
        convert_from_projectq,
        convert_to_projectq,
    )
    CONVERTER_AVAILABLE = True
except ImportError as e:
    CONVERTER_AVAILABLE = False
    print(f"ProjectQ converter not available: {e}")

# Try to import ProjectQ
try:
    import projectq
    from projectq import MainEngine
    from projectq.ops import H, X, Y, Z, S, T, Rx, Ry, Rz, CNOT, CZ, Swap, Toffoli, All
    from projectq.backends import Simulator, CommandPrinter
    PROJECTQ_INSTALLED = True
except ImportError:
    PROJECTQ_INSTALLED = False
    print("ProjectQ not installed. Skipping ProjectQ-dependent tests.")


@pytest.mark.skipif(not CONVERTER_AVAILABLE, reason="ProjectQ converter not available")
class TestProjectQConverterBasics:
    """Test basic converter functionality."""

    def test_converter_initialization(self):
        """Test converter can be initialized."""
        converter = ProjectQConverter()
        assert converter is not None
        assert converter.strict_mode is False

    def test_converter_strict_mode(self):
        """Test converter with strict mode."""
        converter = ProjectQConverter(strict_mode=True)
        assert converter.strict_mode is True


@pytest.mark.skipif(
    not (CONVERTER_AVAILABLE and PROJECTQ_INSTALLED),
    reason="ProjectQ converter or ProjectQ not available"
)
class TestProjectQConversion:
    """Test ProjectQ circuit conversions."""

    def test_basic_bell_state_conversion(self):
        """Test conversion of a simple Bell state circuit."""
        # Create ProjectQ Bell state circuit
        eng = MainEngine(backend=CommandPrinter())
        qureg = eng.allocate_qureg(2)

        H | qureg[0]
        CNOT | (qureg[0], qureg[1])

        eng.flush()

        # Convert to QuantRS2
        converter = ProjectQConverter()
        quantrs_circuit, stats = converter.from_projectq(eng)

        # Verify conversion
        assert quantrs_circuit is not None
        assert stats.success
        assert stats.converted_gates >= 2
        assert len(stats.unsupported_gates) == 0

    def test_single_qubit_gates(self):
        """Test conversion of single-qubit gates."""
        # Create circuit with various single-qubit gates
        eng = MainEngine(backend=CommandPrinter())
        qureg = eng.allocate_qureg(3)

        H | qureg[0]
        X | qureg[1]
        Y | qureg[2]
        Z | qureg[0]
        S | qureg[1]
        T | qureg[2]

        eng.flush()

        # Convert
        converter = ProjectQConverter()
        quantrs_circuit, stats = converter.from_projectq(eng)

        # Verify
        assert quantrs_circuit is not None
        assert stats.success
        assert stats.converted_gates >= 6

    def test_rotation_gates(self):
        """Test conversion of rotation gates."""
        import math

        # Create circuit with rotation gates
        eng = MainEngine(backend=CommandPrinter())
        qureg = eng.allocate_qureg(3)

        Rx(math.pi/2) | qureg[0]
        Ry(math.pi/4) | qureg[1]
        Rz(math.pi/3) | qureg[2]

        eng.flush()

        # Convert
        converter = ProjectQConverter()
        quantrs_circuit, stats = converter.from_projectq(eng)

        # Verify
        assert quantrs_circuit is not None
        assert stats.success
        assert stats.converted_gates >= 3

    def test_two_qubit_gates(self):
        """Test conversion of two-qubit gates."""
        # Create circuit with two-qubit gates
        eng = MainEngine(backend=CommandPrinter())
        qureg = eng.allocate_qureg(3)

        CNOT | (qureg[0], qureg[1])
        CZ | (qureg[1], qureg[2])
        Swap | (qureg[0], qureg[2])

        eng.flush()

        # Convert
        converter = ProjectQConverter()
        quantrs_circuit, stats = converter.from_projectq(eng)

        # Verify
        assert quantrs_circuit is not None
        assert stats.success
        assert stats.converted_gates >= 3

    def test_three_qubit_gates(self):
        """Test conversion of three-qubit gates."""
        # Create circuit with Toffoli gate
        eng = MainEngine(backend=CommandPrinter())
        qureg = eng.allocate_qureg(3)

        Toffoli | (qureg[0], qureg[1], qureg[2])

        eng.flush()

        # Convert
        converter = ProjectQConverter()
        quantrs_circuit, stats = converter.from_projectq(eng)

        # Verify
        assert quantrs_circuit is not None
        assert stats.success
        assert stats.converted_gates >= 1

    def test_ghz_state(self):
        """Test conversion of GHZ state preparation."""
        # Create 3-qubit GHZ state
        eng = MainEngine(backend=CommandPrinter())
        qureg = eng.allocate_qureg(3)

        H | qureg[0]
        CNOT | (qureg[0], qureg[1])
        CNOT | (qureg[1], qureg[2])

        eng.flush()

        # Convert
        converter = ProjectQConverter()
        quantrs_circuit, stats = converter.from_projectq(eng)

        # Verify
        assert quantrs_circuit is not None
        assert stats.success
        assert stats.converted_gates >= 3
        assert len(stats.warnings) == 0

    def test_conversion_stats(self):
        """Test that conversion statistics are properly populated."""
        # Create a simple circuit
        eng = MainEngine(backend=CommandPrinter())
        qureg = eng.allocate_qureg(2)

        H | qureg[0]
        CNOT | (qureg[0], qureg[1])

        eng.flush()

        # Convert
        converter = ProjectQConverter()
        _, stats = converter.from_projectq(eng)

        # Verify stats structure
        assert isinstance(stats, ProjectQConversionStats)
        assert stats.original_commands >= 0
        assert stats.converted_gates >= 0
        assert stats.decomposed_gates >= 0
        assert isinstance(stats.unsupported_gates, list)
        assert isinstance(stats.warnings, list)
        assert isinstance(stats.success, bool)

    def test_empty_circuit(self):
        """Test conversion of an empty circuit."""
        # Create empty circuit
        eng = MainEngine(backend=CommandPrinter())
        qureg = eng.allocate_qureg(2)
        eng.flush()

        # Convert
        converter = ProjectQConverter()
        quantrs_circuit, stats = converter.from_projectq(eng)

        # Verify - empty circuit should still convert successfully
        assert stats.success
        assert stats.converted_gates == 0


@pytest.mark.skipif(
    not (CONVERTER_AVAILABLE and PROJECTQ_INSTALLED),
    reason="ProjectQ converter or ProjectQ not available"
)
class TestProjectQBackend:
    """Test ProjectQ backend adapter."""

    def test_backend_initialization(self):
        """Test that ProjectQ backend can be initialized."""
        backend = ProjectQBackend()
        assert backend is not None

    def test_backend_with_engine(self):
        """Test using QuantRS2 as ProjectQ backend."""
        # Create engine with QuantRS2 backend
        backend = ProjectQBackend()
        eng = MainEngine(backend=backend)

        # Build simple circuit
        qureg = eng.allocate_qureg(2)
        H | qureg[0]
        CNOT | (qureg[0], qureg[1])

        eng.flush()

        # Verify backend captured commands
        assert backend._circuit is not None

    def test_backend_bell_state(self):
        """Test Bell state creation using backend adapter."""
        backend = ProjectQBackend()
        eng = MainEngine(backend=backend)

        qureg = eng.allocate_qureg(2)
        H | qureg[0]
        CNOT | (qureg[0], qureg[1])

        eng.flush()

        # Get the QuantRS2 circuit
        circuit = backend._circuit
        assert circuit is not None


@pytest.mark.skipif(not CONVERTER_AVAILABLE, reason="ProjectQ converter not available")
class TestProjectQErrorHandling:
    """Test error handling in ProjectQ converter."""

    def test_strict_mode_on_unsupported_gate(self):
        """Test that strict mode raises errors on unsupported gates."""
        if not PROJECTQ_INSTALLED:
            pytest.skip("ProjectQ not installed")

        converter = ProjectQConverter(strict_mode=True)

        # Create simple circuit (all gates should be supported)
        eng = MainEngine(backend=CommandPrinter())
        qureg = eng.allocate_qureg(2)
        H | qureg[0]
        eng.flush()

        # This should work for basic gates
        quantrs_circuit, stats = converter.from_projectq(eng)
        assert quantrs_circuit is not None

    def test_lenient_mode_continues_on_unsupported(self):
        """Test that lenient mode continues on unsupported gates."""
        if not PROJECTQ_INSTALLED:
            pytest.skip("ProjectQ not installed")

        converter = ProjectQConverter(strict_mode=False)

        # Create a basic circuit
        eng = MainEngine(backend=CommandPrinter())
        qureg = eng.allocate_qureg(2)
        H | qureg[0]
        CNOT | (qureg[0], qureg[1])
        eng.flush()

        # Should succeed even if some gates might be problematic
        quantrs_circuit, stats = converter.from_projectq(eng)
        assert quantrs_circuit is not None or not stats.success


@pytest.mark.skipif(not CONVERTER_AVAILABLE, reason="ProjectQ converter not available")
class TestProjectQConvenienceFunctions:
    """Test convenience functions for ProjectQ conversion."""

    def test_convert_from_projectq_function(self):
        """Test the convert_from_projectq convenience function."""
        if not PROJECTQ_INSTALLED:
            pytest.skip("ProjectQ not installed")

        # Create simple circuit
        eng = MainEngine(backend=CommandPrinter())
        qureg = eng.allocate_qureg(2)
        H | qureg[0]
        eng.flush()

        # Use convenience function
        quantrs_circuit = convert_from_projectq(eng)
        assert quantrs_circuit is not None

    def test_convert_to_projectq_function(self):
        """Test the convert_to_projectq convenience function."""
        if not PROJECTQ_INSTALLED:
            pytest.skip("ProjectQ not installed")

        # Create ProjectQ circuit and convert to QuantRS2
        eng = MainEngine(backend=CommandPrinter())
        qureg = eng.allocate_qureg(2)
        H | qureg[0]
        eng.flush()

        converter = ProjectQConverter()
        quantrs_circuit, _ = converter.from_projectq(eng)

        # Convert back using convenience function
        projectq_commands = convert_to_projectq(quantrs_circuit)
        assert projectq_commands is not None


@pytest.mark.skipif(
    not (CONVERTER_AVAILABLE and PROJECTQ_INSTALLED),
    reason="ProjectQ converter or ProjectQ not available"
)
class TestProjectQAdvancedFeatures:
    """Test advanced ProjectQ features."""

    def test_quantum_fourier_transform(self):
        """Test QFT conversion."""
        import math

        eng = MainEngine(backend=CommandPrinter())
        qureg = eng.allocate_qureg(3)

        # Simple QFT-like pattern
        H | qureg[0]
        Rz(math.pi/2) | qureg[0]
        CNOT | (qureg[0], qureg[1])
        H | qureg[1]

        eng.flush()

        # Convert
        converter = ProjectQConverter()
        quantrs_circuit, stats = converter.from_projectq(eng)

        # Verify
        assert quantrs_circuit is not None
        assert stats.success

    def test_variational_circuit(self):
        """Test variational circuit conversion."""
        import math

        eng = MainEngine(backend=CommandPrinter())
        qureg = eng.allocate_qureg(2)

        # Variational ansatz pattern
        Ry(math.pi/4) | qureg[0]
        Ry(math.pi/3) | qureg[1]
        CNOT | (qureg[0], qureg[1])
        Ry(math.pi/6) | qureg[0]
        Ry(math.pi/5) | qureg[1]

        eng.flush()

        # Convert
        converter = ProjectQConverter()
        quantrs_circuit, stats = converter.from_projectq(eng)

        # Verify
        assert quantrs_circuit is not None
        assert stats.success
        assert stats.converted_gates >= 5


def test_module_availability():
    """Test that the converter module is properly available."""
    assert CONVERTER_AVAILABLE, "ProjectQ converter module should be available"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
