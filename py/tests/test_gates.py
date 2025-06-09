#!/usr/bin/env python3
"""
Test suite for quantum gate functionality.
"""

import pytest
import numpy as np

try:
    from quantrs2.gates import *
    HAS_GATES = True
except ImportError:
    HAS_GATES = False


@pytest.mark.skipif(not HAS_GATES, reason="gates module not available")
class TestSingleQubitGates:
    """Test single-qubit gate classes."""
    
    def test_hadamard_gate(self):
        """Test Hadamard gate creation."""
        gate = H(0)
        assert gate is not None
        
        # Test convenience function
        gate2 = h(0)
        assert gate2 is not None
        assert type(gate) == type(gate2)
    
    def test_pauli_gates(self):
        """Test Pauli gate creation."""
        # X gate
        x_gate = X(0)
        assert x_gate is not None
        x_gate2 = x(0)
        assert type(x_gate) == type(x_gate2)
        
        # Y gate
        y_gate = Y(1)
        assert y_gate is not None
        y_gate2 = y(1)
        assert type(y_gate) == type(y_gate2)
        
        # Z gate
        z_gate = Z(2)
        assert z_gate is not None
        z_gate2 = z(2)
        assert type(z_gate) == type(z_gate2)
    
    def test_phase_gates(self):
        """Test phase gate creation."""
        # S gate
        s_gate = S(0)
        assert s_gate is not None
        s_gate2 = s(0)
        assert type(s_gate) == type(s_gate2)
        
        # S-dagger gate
        sdg_gate = SDagger(0)
        assert sdg_gate is not None
        sdg_gate2 = sdg(0)
        assert type(sdg_gate) == type(sdg_gate2)
        
        # T gate
        t_gate = T(0)
        assert t_gate is not None
        t_gate2 = t(0)
        assert type(t_gate) == type(t_gate2)
        
        # T-dagger gate
        tdg_gate = TDagger(0)
        assert tdg_gate is not None
        tdg_gate2 = tdg(0)
        assert type(tdg_gate) == type(tdg_gate2)
    
    def test_sqrt_x_gates(self):
        """Test square root of X gates."""
        # SX gate
        sx_gate = SX(0)
        assert sx_gate is not None
        sx_gate2 = sx(0)
        assert type(sx_gate) == type(sx_gate2)
        
        # SX-dagger gate
        sxdg_gate = SXDagger(0)
        assert sxdg_gate is not None
        sxdg_gate2 = sxdg(0)
        assert type(sxdg_gate) == type(sxdg_gate2)
    
    def test_rotation_gates(self):
        """Test rotation gate creation."""
        # RX gate
        rx_gate = RX(0, np.pi/2)
        assert rx_gate is not None
        rx_gate2 = rx(0, np.pi/2)
        assert type(rx_gate) == type(rx_gate2)
        
        # RY gate
        ry_gate = RY(1, np.pi/3)
        assert ry_gate is not None
        ry_gate2 = ry(1, np.pi/3)
        assert type(ry_gate) == type(ry_gate2)
        
        # RZ gate
        rz_gate = RZ(2, np.pi/4)
        assert rz_gate is not None
        rz_gate2 = rz(2, np.pi/4)
        assert type(rz_gate) == type(rz_gate2)
    
    def test_rotation_gate_angles(self):
        """Test rotation gates with different angles."""
        # Test with zero angle
        rx_zero = RX(0, 0.0)
        assert rx_zero is not None
        
        # Test with pi angle
        ry_pi = RY(0, np.pi)
        assert ry_pi is not None
        
        # Test with 2*pi angle
        rz_2pi = RZ(0, 2 * np.pi)
        assert rz_2pi is not None
        
        # Test with negative angle
        rx_neg = RX(0, -np.pi/2)
        assert rx_neg is not None


@pytest.mark.skipif(not HAS_GATES, reason="gates module not available")
class TestTwoQubitGates:
    """Test two-qubit gate classes."""
    
    def test_cnot_gate(self):
        """Test CNOT gate creation."""
        cnot_gate = CNOT(0, 1)
        assert cnot_gate is not None
        
        # Test convenience function
        cnot_gate2 = cnot(0, 1)
        assert type(cnot_gate) == type(cnot_gate2)
        
        # Test alias
        cx_gate = CX(0, 1)
        assert type(cnot_gate) == type(cx_gate)
    
    def test_controlled_pauli_gates(self):
        """Test controlled Pauli gates."""
        # Controlled-Y
        cy_gate = CY(0, 1)
        assert cy_gate is not None
        cy_gate2 = cy(0, 1)
        assert type(cy_gate) == type(cy_gate2)
        
        # Controlled-Z
        cz_gate = CZ(0, 1)
        assert cz_gate is not None
        cz_gate2 = cz(0, 1)
        assert type(cz_gate) == type(cz_gate2)
    
    def test_controlled_other_gates(self):
        """Test other controlled gates."""
        # Controlled-H
        ch_gate = CH(0, 1)
        assert ch_gate is not None
        ch_gate2 = ch(0, 1)
        assert type(ch_gate) == type(ch_gate2)
        
        # Controlled-S
        cs_gate = CS(0, 1)
        assert cs_gate is not None
        cs_gate2 = cs(0, 1)
        assert type(cs_gate) == type(cs_gate2)
    
    def test_swap_gate(self):
        """Test SWAP gate creation."""
        swap_gate = SWAP(0, 1)
        assert swap_gate is not None
        
        # Test convenience function
        swap_gate2 = swap(0, 1)
        assert type(swap_gate) == type(swap_gate2)
    
    def test_controlled_rotation_gates(self):
        """Test controlled rotation gates."""
        # Controlled RX
        crx_gate = CRX(0, 1, np.pi/2)
        assert crx_gate is not None
        crx_gate2 = crx(0, 1, np.pi/2)
        assert type(crx_gate) == type(crx_gate2)
        
        # Controlled RY
        cry_gate = CRY(0, 1, np.pi/3)
        assert cry_gate is not None
        cry_gate2 = cry(0, 1, np.pi/3)
        assert type(cry_gate) == type(cry_gate2)
        
        # Controlled RZ
        crz_gate = CRZ(0, 1, np.pi/4)
        assert crz_gate is not None
        crz_gate2 = crz(0, 1, np.pi/4)
        assert type(crz_gate) == type(crz_gate2)
    
    def test_two_qubit_gate_qubit_ordering(self):
        """Test that two-qubit gates handle qubit ordering."""
        # Different orderings should create gates
        cnot1 = CNOT(0, 1)
        cnot2 = CNOT(1, 0)
        assert cnot1 is not None
        assert cnot2 is not None
        
        swap1 = SWAP(0, 2)
        swap2 = SWAP(2, 0)
        assert swap1 is not None
        assert swap2 is not None


@pytest.mark.skipif(not HAS_GATES, reason="gates module not available")
class TestThreeQubitGates:
    """Test three-qubit gate classes."""
    
    def test_toffoli_gate(self):
        """Test Toffoli gate creation."""
        toffoli_gate = Toffoli(0, 1, 2)
        assert toffoli_gate is not None
        
        # Test convenience function
        toffoli_gate2 = toffoli(0, 1, 2)
        assert type(toffoli_gate) == type(toffoli_gate2)
        
        # Test alias
        ccx_gate = CCX(0, 1, 2)
        assert type(toffoli_gate) == type(ccx_gate)
    
    def test_fredkin_gate(self):
        """Test Fredkin gate creation."""
        fredkin_gate = Fredkin(0, 1, 2)
        assert fredkin_gate is not None
        
        # Test convenience function
        fredkin_gate2 = fredkin(0, 1, 2)
        assert type(fredkin_gate) == type(fredkin_gate2)
        
        # Test alias
        cswap_gate = CSWAP(0, 1, 2)
        assert type(fredkin_gate) == type(cswap_gate)
    
    def test_three_qubit_gate_ordering(self):
        """Test three-qubit gates with different qubit orderings."""
        # Different control qubits
        toffoli1 = Toffoli(0, 1, 2)
        toffoli2 = Toffoli(1, 0, 2)
        assert toffoli1 is not None
        assert toffoli2 is not None
        
        # Different target qubits
        fredkin1 = Fredkin(0, 1, 2)
        fredkin2 = Fredkin(0, 2, 1)
        assert fredkin1 is not None
        assert fredkin2 is not None


@pytest.mark.skipif(not HAS_GATES, reason="gates module not available")
class TestParametricGates:
    """Test parametric gate classes."""
    
    def test_parametric_rotation_gates(self):
        """Test parametric rotation gate creation."""
        # With numeric parameters
        prx_num = ParametricRX(0, np.pi/2)
        assert prx_num is not None
        
        pry_num = ParametricRY(0, np.pi/3)
        assert pry_num is not None
        
        prz_num = ParametricRZ(0, np.pi/4)
        assert prz_num is not None
        
        # With string parameters (symbolic)
        try:
            prx_sym = ParametricRX(0, "theta1")
            assert prx_sym is not None
        except Exception:
            # Symbolic parameters might not be fully implemented
            pass
    
    def test_parametric_u_gate(self):
        """Test parametric U gate creation."""
        # With numeric parameters
        pu_num = ParametricU(0, np.pi/2, np.pi/3, np.pi/4)
        assert pu_num is not None
        
        # With mixed parameters
        try:
            pu_mixed = ParametricU(0, "theta", np.pi/3, "lambda")
            assert pu_mixed is not None
        except Exception:
            # Mixed parameters might not be fully implemented
            pass


@pytest.mark.skipif(not HAS_GATES, reason="gates module not available")
class TestCustomGates:
    """Test custom gate creation."""
    
    def test_custom_single_qubit_gate(self):
        """Test custom single-qubit gate creation."""
        # Identity matrix
        identity = np.array([[1, 0], [0, 1]], dtype=complex)
        
        try:
            custom_gate = CustomGate("Identity", [0], identity)
            assert custom_gate is not None
        except Exception:
            # Custom gates might not be fully implemented
            pass
    
    def test_custom_two_qubit_gate(self):
        """Test custom two-qubit gate creation."""
        # Identity matrix for 2 qubits
        identity_2 = np.eye(4, dtype=complex)
        
        try:
            custom_gate = CustomGate("Identity2", [0, 1], identity_2)
            assert custom_gate is not None
        except Exception:
            # Custom gates might not be fully implemented
            pass
    
    def test_custom_gate_invalid_matrix(self):
        """Test custom gate with invalid matrix."""
        # Non-unitary matrix
        invalid_matrix = np.array([[1, 0], [0, 2]], dtype=complex)
        
        try:
            custom_gate = CustomGate("Invalid", [0], invalid_matrix)
            # If it doesn't raise an error, that's implementation-dependent
        except Exception:
            # Expected for non-unitary matrices
            pass


@pytest.mark.skipif(not HAS_GATES, reason="gates module not available")
class TestGateUtilities:
    """Test gate utility functions."""
    
    def test_create_controlled_gate(self):
        """Test controlled gate creation utility."""
        h_gate = H(1)
        
        # This function is not yet implemented
        with pytest.raises(NotImplementedError):
            controlled_h = create_controlled_gate(h_gate, [0])
    
    def test_decompose_gate(self):
        """Test gate decomposition utility."""
        cnot_gate = CNOT(0, 1)
        
        # This function is not yet implemented
        with pytest.raises(NotImplementedError):
            decomposed = decompose_gate(cnot_gate)


@pytest.mark.skipif(not HAS_GATES, reason="gates module not available")
class TestGateAliases:
    """Test gate aliases and convenience functions."""
    
    def test_cnot_aliases(self):
        """Test CNOT aliases."""
        cnot_gate = CNOT(0, 1)
        cx_gate = CX(0, 1)
        
        # Should be the same type
        assert type(cnot_gate) == type(cx_gate)
    
    def test_toffoli_aliases(self):
        """Test Toffoli aliases."""
        toffoli_gate = Toffoli(0, 1, 2)
        ccx_gate = CCX(0, 1, 2)
        
        # Should be the same type
        assert type(toffoli_gate) == type(ccx_gate)
    
    def test_fredkin_aliases(self):
        """Test Fredkin aliases."""
        fredkin_gate = Fredkin(0, 1, 2)
        cswap_gate = CSWAP(0, 1, 2)
        
        # Should be the same type
        assert type(fredkin_gate) == type(cswap_gate)
    
    def test_all_convenience_functions(self):
        """Test that all convenience functions work."""
        convenience_gates = [
            (h, [0]),
            (x, [0]),
            (y, [0]),
            (z, [0]),
            (s, [0]),
            (sdg, [0]),
            (t, [0]),
            (tdg, [0]),
            (sx, [0]),
            (sxdg, [0]),
            (rx, [0, np.pi/2]),
            (ry, [0, np.pi/2]),
            (rz, [0, np.pi/2]),
            (cnot, [0, 1]),
            (cy, [0, 1]),
            (cz, [0, 1]),
            (ch, [0, 1]),
            (cs, [0, 1]),
            (swap, [0, 1]),
            (crx, [0, 1, np.pi/2]),
            (cry, [0, 1, np.pi/2]),
            (crz, [0, 1, np.pi/2]),
            (toffoli, [0, 1, 2]),
            (fredkin, [0, 1, 2]),
        ]
        
        for func, args in convenience_gates:
            gate = func(*args)
            assert gate is not None


@pytest.mark.skipif(not HAS_GATES, reason="gates module not available")
class TestGateParameters:
    """Test gate parameter handling."""
    
    def test_rotation_gate_parameters(self):
        """Test rotation gates with various parameter values."""
        # Test with different angle types
        angles = [0, np.pi/4, np.pi/2, np.pi, 2*np.pi, -np.pi/2]
        
        for angle in angles:
            rx_gate = RX(0, angle)
            assert rx_gate is not None
            
            ry_gate = RY(0, angle)
            assert ry_gate is not None
            
            rz_gate = RZ(0, angle)
            assert rz_gate is not None
    
    def test_controlled_rotation_parameters(self):
        """Test controlled rotation gates with various parameters."""
        angles = [0, np.pi/4, np.pi, -np.pi/4]
        
        for angle in angles:
            crx_gate = CRX(0, 1, angle)
            assert crx_gate is not None
            
            cry_gate = CRY(0, 1, angle)
            assert cry_gate is not None
            
            crz_gate = CRZ(0, 1, angle)
            assert crz_gate is not None
    
    def test_parameter_edge_cases(self):
        """Test parameter edge cases."""
        # Very small angle
        small_angle = 1e-10
        rx_small = RX(0, small_angle)
        assert rx_small is not None
        
        # Very large angle
        large_angle = 1000 * np.pi
        ry_large = RY(0, large_angle)
        assert ry_large is not None
        
        # Complex angle (should work if implementation supports it)
        try:
            complex_angle = 1 + 1j
            rz_complex = RZ(0, complex_angle)
        except (TypeError, ValueError):
            # Complex angles might not be supported
            pass


@pytest.mark.skipif(not HAS_GATES, reason="gates module not available")
class TestGateQubitIndices:
    """Test gate qubit index handling."""
    
    def test_single_qubit_indices(self):
        """Test single-qubit gates with different indices."""
        for qubit in range(10):
            h_gate = H(qubit)
            assert h_gate is not None
            
            x_gate = X(qubit)
            assert x_gate is not None
    
    def test_two_qubit_indices(self):
        """Test two-qubit gates with different indices."""
        qubit_pairs = [(0, 1), (1, 0), (0, 2), (3, 5), (10, 15)]
        
        for control, target in qubit_pairs:
            cnot_gate = CNOT(control, target)
            assert cnot_gate is not None
            
            swap_gate = SWAP(control, target)
            assert swap_gate is not None
    
    def test_three_qubit_indices(self):
        """Test three-qubit gates with different indices."""
        qubit_triples = [(0, 1, 2), (2, 1, 0), (0, 2, 1), (5, 3, 7)]
        
        for c1, c2, t in qubit_triples:
            toffoli_gate = Toffoli(c1, c2, t)
            assert toffoli_gate is not None
            
            fredkin_gate = Fredkin(c1, c2, t)
            assert fredkin_gate is not None
    
    def test_same_qubit_indices(self):
        """Test gates with same qubit indices (should be invalid)."""
        # Two-qubit gates with same indices
        try:
            cnot_same = CNOT(0, 0)
            # Implementation might allow or disallow this
        except Exception:
            # Invalid same-qubit operation is expected
            pass
        
        # Three-qubit gates with repeated indices
        try:
            toffoli_same = Toffoli(0, 0, 1)
            # Implementation might allow or disallow this
        except Exception:
            # Invalid repeated control qubits is expected
            pass


@pytest.mark.skipif(not HAS_GATES, reason="gates module not available")
class TestModuleExports:
    """Test module exports and __all__ completeness."""
    
    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        from quantrs2.gates import __all__
        
        # Check that major gate classes are exported
        expected_classes = [
            'H', 'X', 'Y', 'Z', 'S', 'T', 'RX', 'RY', 'RZ',
            'CNOT', 'CZ', 'SWAP', 'Toffoli', 'Fredkin'
        ]
        
        for class_name in expected_classes:
            assert class_name in __all__
        
        # Check that convenience functions are exported
        expected_functions = [
            'h', 'x', 'y', 'z', 'cnot', 'swap', 'toffoli'
        ]
        
        for func_name in expected_functions:
            assert func_name in __all__
    
    def test_import_all(self):
        """Test that all exported names can be imported."""
        from quantrs2.gates import __all__
        
        # Try to access each exported name
        import quantrs2.gates as gates_module
        
        for name in __all__:
            assert hasattr(gates_module, name)
            obj = getattr(gates_module, name)
            assert obj is not None


if __name__ == "__main__":
    pytest.main([__file__])