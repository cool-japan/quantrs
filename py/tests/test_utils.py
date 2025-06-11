#!/usr/bin/env python3
"""
Test suite for utility functions.
"""

import pytest
import numpy as np
import math

try:
    from quantrs2.utils import (
        binary_to_int, int_to_binary, state_index, get_basis_states,
        state_to_vector, vector_to_state, fidelity, entropy,
        measure_qubit, bell_state, ghz_state, w_state, uniform_superposition
    )
    HAS_UTILS = True
except ImportError:
    HAS_UTILS = False


@pytest.mark.skipif(not HAS_UTILS, reason="utils module not available")
class TestBinaryConversions:
    """Test binary string conversion functions."""
    
    def test_binary_to_int(self):
        """Test binary string to integer conversion."""
        assert binary_to_int('0') == 0
        assert binary_to_int('1') == 1
        assert binary_to_int('10') == 2
        assert binary_to_int('11') == 3
        assert binary_to_int('100') == 4
        assert binary_to_int('101') == 5
        assert binary_to_int('110') == 6
        assert binary_to_int('111') == 7
        assert binary_to_int('1101') == 13
    
    def test_int_to_binary(self):
        """Test integer to binary string conversion."""
        assert int_to_binary(0, 1) == '0'
        assert int_to_binary(1, 1) == '1'
        assert int_to_binary(2, 2) == '10'
        assert int_to_binary(3, 2) == '11'
        assert int_to_binary(5, 3) == '101'
        assert int_to_binary(7, 3) == '111'
        assert int_to_binary(13, 4) == '1101'
        
        # Test padding
        assert int_to_binary(1, 3) == '001'
        assert int_to_binary(2, 4) == '0010'
    
    def test_state_index(self):
        """Test bit string to state index conversion."""
        assert state_index('0') == 0
        assert state_index('1') == 1
        assert state_index('00') == 0
        assert state_index('01') == 1
        assert state_index('10') == 2
        assert state_index('11') == 3
        assert state_index('000') == 0
        assert state_index('101') == 5
        assert state_index('111') == 7
    
    def test_get_basis_states(self):
        """Test generation of all basis states."""
        # 1 qubit
        states_1 = get_basis_states(1)
        assert states_1 == ['0', '1']
        
        # 2 qubits
        states_2 = get_basis_states(2)
        assert states_2 == ['00', '01', '10', '11']
        
        # 3 qubits
        states_3 = get_basis_states(3)
        expected_3 = ['000', '001', '010', '011', '100', '101', '110', '111']
        assert states_3 == expected_3
        
        # Check length
        assert len(get_basis_states(4)) == 16
        assert len(get_basis_states(5)) == 32


@pytest.mark.skipif(not HAS_UTILS, reason="utils module not available")
class TestStateConversions:
    """Test quantum state conversion functions."""
    
    def test_state_to_vector(self):
        """Test state dictionary to vector conversion."""
        # Single qubit |0⟩
        state_0 = {'0': 1.0}
        vec_0 = state_to_vector(state_0, 1)
        expected_0 = np.array([1.0, 0.0])
        np.testing.assert_array_almost_equal(vec_0, expected_0)
        
        # Single qubit |1⟩
        state_1 = {'1': 1.0}
        vec_1 = state_to_vector(state_1, 1)
        expected_1 = np.array([0.0, 1.0])
        np.testing.assert_array_almost_equal(vec_1, expected_1)
        
        # Superposition (|0⟩ + |1⟩)/√2
        state_super = {'0': 1/math.sqrt(2), '1': 1/math.sqrt(2)}
        vec_super = state_to_vector(state_super, 1)
        expected_super = np.array([1/math.sqrt(2), 1/math.sqrt(2)])
        np.testing.assert_array_almost_equal(vec_super, expected_super)
        
        # Two qubits |00⟩
        state_00 = {'00': 1.0}
        vec_00 = state_to_vector(state_00, 2)
        expected_00 = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(vec_00, expected_00)
        
        # Bell state (|00⟩ + |11⟩)/√2
        bell_dict = {'00': 1/math.sqrt(2), '11': 1/math.sqrt(2)}
        vec_bell = state_to_vector(bell_dict, 2)
        expected_bell = np.array([1/math.sqrt(2), 0.0, 0.0, 1/math.sqrt(2)])
        np.testing.assert_array_almost_equal(vec_bell, expected_bell)
    
    def test_vector_to_state(self):
        """Test vector to state dictionary conversion."""
        # Single qubit |0⟩
        vec_0 = np.array([1.0, 0.0])
        state_0 = vector_to_state(vec_0, 1)
        assert '0' in state_0
        assert abs(state_0['0'] - 1.0) < 1e-10
        assert len(state_0) == 1  # Should only contain non-zero amplitudes
        
        # Single qubit |1⟩
        vec_1 = np.array([0.0, 1.0])
        state_1 = vector_to_state(vec_1, 1)
        assert '1' in state_1
        assert abs(state_1['1'] - 1.0) < 1e-10
        assert len(state_1) == 1
        
        # Superposition
        vec_super = np.array([1/math.sqrt(2), 1/math.sqrt(2)])
        state_super = vector_to_state(vec_super, 1)
        assert '0' in state_super and '1' in state_super
        assert abs(state_super['0'] - 1/math.sqrt(2)) < 1e-10
        assert abs(state_super['1'] - 1/math.sqrt(2)) < 1e-10
        
        # Bell state
        vec_bell = np.array([1/math.sqrt(2), 0.0, 0.0, 1/math.sqrt(2)])
        state_bell = vector_to_state(vec_bell, 2)
        assert '00' in state_bell and '11' in state_bell
        assert '01' not in state_bell and '10' not in state_bell  # Zero amplitudes filtered out
        assert abs(state_bell['00'] - 1/math.sqrt(2)) < 1e-10
        assert abs(state_bell['11'] - 1/math.sqrt(2)) < 1e-10
    
    def test_round_trip_conversion(self):
        """Test round-trip state conversion."""
        # Start with a state dictionary
        original_state = {
            '000': 0.5,
            '001': 0.5,
            '110': 1/math.sqrt(2) * 0.5,
            '111': 1/math.sqrt(2) * 0.5
        }
        
        # Convert to vector and back
        vector = state_to_vector(original_state, 3)
        reconstructed_state = vector_to_state(vector, 3)
        
        # Check that non-zero amplitudes match
        for basis in original_state:
            assert basis in reconstructed_state
            assert abs(original_state[basis] - reconstructed_state[basis]) < 1e-10


@pytest.mark.skipif(not HAS_UTILS, reason="utils module not available")
class TestQuantumMeasures:
    """Test quantum measurement and information functions."""
    
    def test_fidelity_identical_states(self):
        """Test fidelity between identical states."""
        # Dictionary states
        state1 = {'0': 1.0}
        state2 = {'0': 1.0}
        assert abs(fidelity(state1, state2, 1) - 1.0) < 1e-10
        
        # Vector states
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([1.0, 0.0])
        assert abs(fidelity(vec1, vec2) - 1.0) < 1e-10
        
        # Bell states
        bell1 = {'00': 1/math.sqrt(2), '11': 1/math.sqrt(2)}
        bell2 = {'00': 1/math.sqrt(2), '11': 1/math.sqrt(2)}
        assert abs(fidelity(bell1, bell2, 2) - 1.0) < 1e-10
    
    def test_fidelity_orthogonal_states(self):
        """Test fidelity between orthogonal states."""
        # |0⟩ and |1⟩
        state_0 = {'0': 1.0}
        state_1 = {'1': 1.0}
        assert abs(fidelity(state_0, state_1, 1) - 0.0) < 1e-10
        
        # |00⟩ and |11⟩
        state_00 = {'00': 1.0}
        state_11 = {'11': 1.0}
        assert abs(fidelity(state_00, state_11, 2) - 0.0) < 1e-10
    
    def test_fidelity_superposition_states(self):
        """Test fidelity between superposition states."""
        # |+⟩ = (|0⟩ + |1⟩)/√2 and |-⟩ = (|0⟩ - |1⟩)/√2
        plus_state = {'0': 1/math.sqrt(2), '1': 1/math.sqrt(2)}
        minus_state = {'0': 1/math.sqrt(2), '1': -1/math.sqrt(2)}
        
        # These should be orthogonal
        assert abs(fidelity(plus_state, minus_state, 1) - 0.0) < 1e-10
        
        # |+⟩ and |0⟩ should have fidelity 0.5
        zero_state = {'0': 1.0}
        assert abs(fidelity(plus_state, zero_state, 1) - 0.5) < 1e-10
    
    def test_fidelity_mixed_formats(self):
        """Test fidelity between dictionary and vector formats."""
        state_dict = {'0': 1/math.sqrt(2), '1': 1/math.sqrt(2)}
        state_vec = np.array([1/math.sqrt(2), 1/math.sqrt(2)])
        
        assert abs(fidelity(state_dict, state_vec, 1) - 1.0) < 1e-10
        assert abs(fidelity(state_vec, state_dict, 1) - 1.0) < 1e-10
    
    def test_fidelity_dimension_mismatch(self):
        """Test fidelity with mismatched dimensions."""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0, 0.0])
        
        with pytest.raises(ValueError):
            fidelity(vec1, vec2)
    
    def test_entropy_pure_states(self):
        """Test entropy of pure states."""
        # Pure states should have zero entropy
        pure_state_0 = {'0': 1.0}
        assert abs(entropy(pure_state_0) - 0.0) < 1e-10
        
        pure_state_bell = {'00': 1/math.sqrt(2), '11': 1/math.sqrt(2)}
        assert abs(entropy(pure_state_bell) - 0.0) < 1e-10
        
        # Vector format
        pure_vec = np.array([1.0, 0.0, 0.0, 0.0])
        assert abs(entropy(pure_vec) - 0.0) < 1e-10
    
    def test_entropy_mixed_states(self):
        """Test entropy of mixed states."""
        # Maximally mixed single qubit (uniform distribution)
        mixed_state = {'0': 1/math.sqrt(2), '1': 1/math.sqrt(2)}
        # This is actually a pure state in superposition, so entropy should be 0
        assert abs(entropy(mixed_state) - 0.0) < 1e-10
        
        # For testing mixed state entropy, we need classical mixtures
        # which aren't directly representable as pure quantum states
        # But we can test the entropy formula with probabilities
        
        # Equal probabilities should give maximum entropy
        uniform_probs = np.array([0.5, 0.5])
        expected_entropy = 1.0  # log2(2) = 1
        calculated_entropy = -sum(p * math.log2(p) for p in uniform_probs)
        assert abs(calculated_entropy - expected_entropy) < 1e-10


@pytest.mark.skipif(not HAS_UTILS, reason="utils module not available")
class TestMeasurement:
    """Test quantum measurement simulation."""
    
    def test_measure_qubit_computational_basis(self):
        """Test measurement in computational basis states."""
        # Measure |0⟩ state
        state_0 = {'0': 1.0}
        post_0, post_1, prob_1 = measure_qubit(state_0, 0)
        
        assert prob_1 == 0.0  # Should never measure 1
        assert '0' in post_0
        assert len(post_1) == 0  # No amplitudes for |1⟩ outcome
        
        # Measure |1⟩ state
        state_1 = {'1': 1.0}
        post_0, post_1, prob_1 = measure_qubit(state_1, 0)
        
        assert prob_1 == 1.0  # Should always measure 1
        assert len(post_0) == 0  # No amplitudes for |0⟩ outcome
        assert '1' in post_1
    
    def test_measure_qubit_superposition(self):
        """Test measurement in superposition state."""
        # Measure |+⟩ = (|0⟩ + |1⟩)/√2
        plus_state = {'0': 1/math.sqrt(2), '1': 1/math.sqrt(2)}
        post_0, post_1, prob_1 = measure_qubit(plus_state, 0)
        
        assert abs(prob_1 - 0.5) < 1e-10  # 50% probability
        
        # Post-measurement states should be normalized
        if post_0:
            assert abs(post_0['0'] - 1.0) < 1e-10
        if post_1:
            assert abs(post_1['1'] - 1.0) < 1e-10
    
    def test_measure_qubit_two_qubit_system(self):
        """Test measurement in two-qubit system."""
        # Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        bell_state_dict = {'00': 1/math.sqrt(2), '11': 1/math.sqrt(2)}
        
        # Measure first qubit (rightmost in our convention)
        post_0, post_1, prob_1 = measure_qubit(bell_state_dict, 0)
        
        assert abs(prob_1 - 0.5) < 1e-10  # 50% probability
        
        # If we measure first qubit as 0, we should get |00⟩
        if post_0:
            assert '00' in post_0
            assert abs(post_0['00'] - 1.0) < 1e-10
        
        # If we measure first qubit as 1, we should get |11⟩
        if post_1:
            assert '11' in post_1
            assert abs(post_1['11'] - 1.0) < 1e-10
        
        # Measure second qubit
        post_0_2, post_1_2, prob_1_2 = measure_qubit(bell_state_dict, 1)
        
        assert abs(prob_1_2 - 0.5) < 1e-10  # 50% probability


@pytest.mark.skipif(not HAS_UTILS, reason="utils module not available")
class TestQuantumStates:
    """Test quantum state creation functions."""
    
    def test_bell_state_variants(self):
        """Test all Bell state variants."""
        # |Φ+⟩ = (|00⟩ + |11⟩)/√2
        phi_plus = bell_state('phi_plus')
        assert '00' in phi_plus and '11' in phi_plus
        assert abs(phi_plus['00'] - 1/math.sqrt(2)) < 1e-10
        assert abs(phi_plus['11'] - 1/math.sqrt(2)) < 1e-10
        
        # |Φ-⟩ = (|00⟩ - |11⟩)/√2
        phi_minus = bell_state('phi_minus')
        assert '00' in phi_minus and '11' in phi_minus
        assert abs(phi_minus['00'] - 1/math.sqrt(2)) < 1e-10
        assert abs(phi_minus['11'] - (-1/math.sqrt(2))) < 1e-10
        
        # |Ψ+⟩ = (|01⟩ + |10⟩)/√2
        psi_plus = bell_state('psi_plus')
        assert '01' in psi_plus and '10' in psi_plus
        assert abs(psi_plus['01'] - 1/math.sqrt(2)) < 1e-10
        assert abs(psi_plus['10'] - 1/math.sqrt(2)) < 1e-10
        
        # |Ψ-⟩ = (|01⟩ - |10⟩)/√2
        psi_minus = bell_state('psi_minus')
        assert '01' in psi_minus and '10' in psi_minus
        assert abs(psi_minus['01'] - 1/math.sqrt(2)) < 1e-10
        assert abs(psi_minus['10'] - (-1/math.sqrt(2))) < 1e-10
    
    def test_bell_state_invalid_variant(self):
        """Test invalid Bell state variant."""
        with pytest.raises(ValueError):
            bell_state('invalid_variant')
    
    def test_bell_state_normalization(self):
        """Test Bell state normalization."""
        for variant in ['phi_plus', 'phi_minus', 'psi_plus', 'psi_minus']:
            state = bell_state(variant)
            # Calculate total probability
            total_prob = sum(abs(amp)**2 for amp in state.values())
            assert abs(total_prob - 1.0) < 1e-10
    
    def test_ghz_state(self):
        """Test GHZ state creation."""
        # 2-qubit GHZ (same as Bell |Φ+⟩)
        ghz_2 = ghz_state(2)
        assert '00' in ghz_2 and '11' in ghz_2
        assert abs(ghz_2['00'] - 1/math.sqrt(2)) < 1e-10
        assert abs(ghz_2['11'] - 1/math.sqrt(2)) < 1e-10
        
        # 3-qubit GHZ
        ghz_3 = ghz_state(3)
        assert '000' in ghz_3 and '111' in ghz_3
        assert len(ghz_3) == 2  # Only two non-zero amplitudes
        assert abs(ghz_3['000'] - 1/math.sqrt(2)) < 1e-10
        assert abs(ghz_3['111'] - 1/math.sqrt(2)) < 1e-10
        
        # Check normalization
        total_prob = sum(abs(amp)**2 for amp in ghz_3.values())
        assert abs(total_prob - 1.0) < 1e-10
        
        # 4-qubit GHZ
        ghz_4 = ghz_state(4)
        assert '0000' in ghz_4 and '1111' in ghz_4
        assert len(ghz_4) == 2
    
    def test_w_state(self):
        """Test W state creation."""
        # 2-qubit W state should be like |Ψ+⟩
        w_2 = w_state(2)
        assert len(w_2) == 2
        expected_amp = 1/math.sqrt(2)
        assert abs(w_2['10'] - expected_amp) < 1e-10
        assert abs(w_2['01'] - expected_amp) < 1e-10
        
        # 3-qubit W state
        w_3 = w_state(3)
        assert len(w_3) == 3
        expected_amp = 1/math.sqrt(3)
        assert abs(w_3['100'] - expected_amp) < 1e-10
        assert abs(w_3['010'] - expected_amp) < 1e-10
        assert abs(w_3['001'] - expected_amp) < 1e-10
        
        # Check normalization
        total_prob = sum(abs(amp)**2 for amp in w_3.values())
        assert abs(total_prob - 1.0) < 1e-10
        
        # 4-qubit W state
        w_4 = w_state(4)
        assert len(w_4) == 4
        expected_amp = 1/math.sqrt(4)
        for basis_state in ['1000', '0100', '0010', '0001']:
            assert basis_state in w_4
            assert abs(w_4[basis_state] - expected_amp) < 1e-10
    
    def test_uniform_superposition(self):
        """Test uniform superposition state."""
        # 1-qubit uniform superposition |+⟩
        uniform_1 = uniform_superposition(1)
        assert len(uniform_1) == 2
        expected_amp = 1/math.sqrt(2)
        assert abs(uniform_1['0'] - expected_amp) < 1e-10
        assert abs(uniform_1['1'] - expected_amp) < 1e-10
        
        # 2-qubit uniform superposition
        uniform_2 = uniform_superposition(2)
        assert len(uniform_2) == 4
        expected_amp = 1/math.sqrt(4)
        for basis_state in ['00', '01', '10', '11']:
            assert basis_state in uniform_2
            assert abs(uniform_2[basis_state] - expected_amp) < 1e-10
        
        # Check normalization
        total_prob = sum(abs(amp)**2 for amp in uniform_2.values())
        assert abs(total_prob - 1.0) < 1e-10
        
        # 3-qubit uniform superposition
        uniform_3 = uniform_superposition(3)
        assert len(uniform_3) == 8
        expected_amp = 1/math.sqrt(8)
        
        # Check normalization
        total_prob = sum(abs(amp)**2 for amp in uniform_3.values())
        assert abs(total_prob - 1.0) < 1e-10


@pytest.mark.skipif(not HAS_UTILS, reason="utils module not available")
class TestUtilsIntegration:
    """Test integration between utility functions."""
    
    def test_state_conversion_with_quantum_states(self):
        """Test state conversions with quantum states."""
        # Create Bell state and convert to vector
        bell_dict = bell_state('phi_plus')
        bell_vec = state_to_vector(bell_dict, 2)
        
        # Convert back to dictionary
        bell_dict_2 = vector_to_state(bell_vec, 2)
        
        # Should be the same (within numerical precision)
        for basis in bell_dict:
            assert basis in bell_dict_2
            assert abs(bell_dict[basis] - bell_dict_2[basis]) < 1e-10
    
    def test_fidelity_with_quantum_states(self):
        """Test fidelity calculations with quantum states."""
        # Same Bell states should have fidelity 1
        bell1 = bell_state('phi_plus')
        bell2 = bell_state('phi_plus')
        assert abs(fidelity(bell1, bell2, 2) - 1.0) < 1e-10
        
        # Different Bell states
        phi_plus = bell_state('phi_plus')
        phi_minus = bell_state('phi_minus')
        # These should be orthogonal
        assert abs(fidelity(phi_plus, phi_minus, 2) - 0.0) < 1e-10
        
        # GHZ and W states
        ghz = ghz_state(3)
        w = w_state(3)
        # These should have some overlap but not be identical
        overlap = fidelity(ghz, w, 3)
        assert 0.0 < overlap < 1.0
    
    def test_entropy_with_quantum_states(self):
        """Test entropy calculations with quantum states."""
        # Pure states should have zero entropy
        bell = bell_state('phi_plus')
        assert abs(entropy(bell) - 0.0) < 1e-10
        
        ghz = ghz_state(3)
        assert abs(entropy(ghz) - 0.0) < 1e-10
        
        w = w_state(3)
        assert abs(entropy(w) - 0.0) < 1e-10
        
        uniform = uniform_superposition(2)
        assert abs(entropy(uniform) - 0.0) < 1e-10
    
    def test_measurement_with_quantum_states(self):
        """Test measurement with quantum states."""
        # Measure Bell state
        bell = bell_state('phi_plus')
        post_0, post_1, prob_1 = measure_qubit(bell, 0)
        
        assert abs(prob_1 - 0.5) < 1e-10
        
        # Measure W state
        w = w_state(3)
        post_0, post_1, prob_1 = measure_qubit(w, 0)
        
        # First qubit should be |1⟩ with probability 1/3
        assert abs(prob_1 - 1/3) < 1e-10


@pytest.mark.skipif(not HAS_UTILS, reason="utils module not available")
class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_states(self):
        """Test handling of empty states."""
        empty_state = {}
        
        # Converting empty state should give zero vector
        vec = state_to_vector(empty_state, 1)
        expected = np.array([0.0, 0.0])
        np.testing.assert_array_almost_equal(vec, expected)
        
        # Entropy of empty state
        assert entropy(empty_state) == 0.0
    
    def test_zero_qubits(self):
        """Test handling of zero qubits."""
        # Should handle gracefully
        states = get_basis_states(0)
        assert states == ['']  # Empty string for 0 qubits
        
        ghz_0 = ghz_state(0)
        assert ghz_0 == {'': 1/math.sqrt(2), '': 1/math.sqrt(2)}  # Degenerate case
    
    def test_large_qubit_numbers(self):
        """Test with larger qubit numbers."""
        # This tests that the functions can handle reasonable sized systems
        states_5 = get_basis_states(5)
        assert len(states_5) == 32
        
        uniform_5 = uniform_superposition(5)
        assert len(uniform_5) == 32
        
        # Check normalization
        total_prob = sum(abs(amp)**2 for amp in uniform_5.values())
        assert abs(total_prob - 1.0) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__])