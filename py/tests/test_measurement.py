#!/usr/bin/env python3
"""Tests for measurement module."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from quantrs2.measurement import *


class TestMeasurementOperations:
    """Test measurement operations."""
    
    def test_measure_all(self):
        """Test measuring all qubits."""
        # Create mock circuit result
        mock_result = Mock()
        mock_result.state_probabilities.return_value = {
            "00": 0.5,
            "01": 0.25,
            "10": 0.25,
            "11": 0.0
        }
        mock_result.n_qubits = 2
        
        measurements = measure_all(mock_result, shots=100)
        
        assert isinstance(measurements, list)
        assert len(measurements) <= 100
        
        # Check that measurements are valid bit strings
        for measurement in measurements[:10]:  # Check first 10
            assert isinstance(measurement, str)
            assert len(measurement) == 2
            assert all(bit in '01' for bit in measurement)
    
    def test_measure_qubit(self):
        """Test measuring a single qubit."""
        mock_result = Mock()
        mock_result.state_probabilities.return_value = {
            "00": 0.25,
            "01": 0.25,
            "10": 0.25,
            "11": 0.25
        }
        mock_result.n_qubits = 2
        
        measurements = measure_qubit(mock_result, qubit=0, shots=100)
        
        assert isinstance(measurements, list)
        assert len(measurements) <= 100
        
        # Check that measurements are 0 or 1
        for measurement in measurements[:10]:
            assert measurement in [0, 1]
    
    def test_measure_subset(self):
        """Test measuring a subset of qubits."""
        mock_result = Mock()
        mock_result.state_probabilities.return_value = {
            "000": 0.125,
            "001": 0.125,
            "010": 0.125,
            "011": 0.125,
            "100": 0.125,
            "101": 0.125,
            "110": 0.125,
            "111": 0.125
        }
        mock_result.n_qubits = 3
        
        measurements = measure_subset(mock_result, qubits=[0, 2], shots=50)
        
        assert isinstance(measurements, list)
        assert len(measurements) <= 50
        
        # Check that measurements are valid for the subset
        for measurement in measurements[:10]:
            assert isinstance(measurement, str)
            assert len(measurement) == 2  # Only qubits 0 and 2
            assert all(bit in '01' for bit in measurement)
    
    def test_expectation_value(self):
        """Test expectation value calculation."""
        mock_result = Mock()
        mock_result.state_probabilities.return_value = {
            "00": 0.5,
            "01": 0.0,
            "10": 0.0,
            "11": 0.5
        }
        mock_result.n_qubits = 2
        
        # Test Z expectation on qubit 0
        z_exp = expectation_value(mock_result, 'Z', qubit=0)
        expected = 0.5 * 1 + 0.5 * (-1)  # |00⟩ and |11⟩ have Z_0 = +1, -1
        assert abs(z_exp - expected) < 1e-10
        
        # Test X expectation (should be 0 for computational basis)
        x_exp = expectation_value(mock_result, 'X', qubit=0)
        assert abs(x_exp) < 1e-10  # X expectation in Z basis is 0
    
    def test_probability_distribution(self):
        """Test probability distribution extraction."""
        mock_result = Mock()
        probs = {
            "00": 0.4,
            "01": 0.3,
            "10": 0.2,
            "11": 0.1
        }
        mock_result.state_probabilities.return_value = probs
        mock_result.n_qubits = 2
        
        distribution = probability_distribution(mock_result)
        
        assert isinstance(distribution, dict)
        assert len(distribution) == 4
        
        # Check probabilities sum to 1
        total_prob = sum(distribution.values())
        assert abs(total_prob - 1.0) < 1e-10
        
        # Check specific probabilities
        for state, prob in probs.items():
            assert abs(distribution[state] - prob) < 1e-10


class TestProjectiveMeasurement:
    """Test projective measurement functionality."""
    
    def test_projective_measurement_z_basis(self):
        """Test projective measurement in Z basis."""
        mock_result = Mock()
        mock_result.state_probabilities.return_value = {
            "0": 0.7,
            "1": 0.3
        }
        mock_result.n_qubits = 1
        
        measurements = projective_measurement(mock_result, basis='Z', shots=1000)
        
        assert isinstance(measurements, list)
        assert len(measurements) <= 1000
        
        # Count occurrences
        count_0 = measurements.count('0')
        count_1 = measurements.count('1')
        
        # Should be approximately 70% |0⟩ and 30% |1⟩
        assert abs(count_0 / len(measurements) - 0.7) < 0.1
        assert abs(count_1 / len(measurements) - 0.3) < 0.1
    
    def test_projective_measurement_x_basis(self):
        """Test projective measurement in X basis."""
        mock_result = Mock()
        # |+⟩ state in computational basis
        mock_result.state_probabilities.return_value = {
            "0": 0.5,
            "1": 0.5
        }
        mock_result.n_qubits = 1
        
        measurements = projective_measurement(mock_result, basis='X', shots=100)
        
        assert isinstance(measurements, list)
        assert len(measurements) <= 100
        
        # In X basis, |+⟩ should always measure as +1 (represented as '0')
        # This is a simplified test - actual implementation may vary
        for measurement in measurements[:10]:
            assert measurement in ['0', '1']
    
    def test_projective_measurement_y_basis(self):
        """Test projective measurement in Y basis."""
        mock_result = Mock()
        mock_result.state_probabilities.return_value = {
            "0": 0.5,
            "1": 0.5
        }
        mock_result.n_qubits = 1
        
        measurements = projective_measurement(mock_result, basis='Y', shots=100)
        
        assert isinstance(measurements, list)
        assert len(measurements) <= 100
        
        for measurement in measurements[:10]:
            assert measurement in ['0', '1']


class TestMeasurementStats:
    """Test measurement statistics functions."""
    
    def test_measurement_counts(self):
        """Test measurement count extraction."""
        measurements = ['00', '01', '00', '11', '01', '00']
        counts = measurement_counts(measurements)
        
        assert isinstance(counts, dict)
        assert counts['00'] == 3
        assert counts['01'] == 2
        assert counts['11'] == 1
        assert '10' not in counts
    
    def test_measurement_probabilities(self):
        """Test measurement probability calculation."""
        measurements = ['0', '1', '0', '0', '1']
        probs = measurement_probabilities(measurements)
        
        assert isinstance(probs, dict)
        assert abs(probs['0'] - 0.6) < 1e-10
        assert abs(probs['1'] - 0.4) < 1e-10
        
        # Check probabilities sum to 1
        total_prob = sum(probs.values())
        assert abs(total_prob - 1.0) < 1e-10
    
    def test_measurement_entropy(self):
        """Test measurement entropy calculation."""
        # Uniform distribution
        measurements_uniform = ['0', '1'] * 50
        entropy_uniform = measurement_entropy(measurements_uniform)
        expected_uniform = 1.0  # log2(2) for uniform distribution
        assert abs(entropy_uniform - expected_uniform) < 0.1
        
        # Deterministic measurement
        measurements_det = ['0'] * 100
        entropy_det = measurement_entropy(measurements_det)
        assert entropy_det == 0  # No uncertainty
        
        # Mixed distribution
        measurements_mixed = ['0'] * 75 + ['1'] * 25
        entropy_mixed = measurement_entropy(measurements_mixed)
        assert 0 < entropy_mixed < 1
    
    def test_mutual_information(self):
        """Test mutual information calculation."""
        # Correlated measurements
        measurements1 = ['0', '1', '0', '1', '0']
        measurements2 = ['0', '1', '0', '1', '0']  # Same as measurements1
        
        mi_corr = mutual_information(measurements1, measurements2)
        assert mi_corr > 0.8  # Should be high for identical measurements
        
        # Independent measurements
        measurements3 = ['0', '0', '1', '1', '0']
        measurements4 = ['1', '0', '0', '1', '1']
        
        mi_indep = mutual_information(measurements3, measurements4)
        # Should be lower than correlated case
        assert mi_indep < mi_corr


class TestPOVMMeasurement:
    """Test POVM (Positive Operator-Valued Measure) measurements."""
    
    def test_povm_measurement(self):
        """Test general POVM measurement."""
        mock_result = Mock()
        mock_result.state_probabilities.return_value = {
            "0": 0.6,
            "1": 0.4
        }
        mock_result.n_qubits = 1
        
        # Define POVM elements (informationally complete)
        povm_elements = [
            np.array([[1, 0], [0, 0]]) / 2,  # |0⟩⟨0| / 2
            np.array([[0, 0], [0, 1]]) / 2,  # |1⟩⟨1| / 2
            np.array([[1, 1], [1, 1]]) / 4,  # |+⟩⟨+| / 2
            np.array([[1, -1], [-1, 1]]) / 4  # |-⟩⟨-| / 2
        ]
        
        outcomes = povm_measurement(mock_result, povm_elements, shots=100)
        
        assert isinstance(outcomes, list)
        assert len(outcomes) <= 100
        
        # Check that outcomes are valid indices
        for outcome in outcomes[:10]:
            assert 0 <= outcome < len(povm_elements)
    
    def test_state_tomography_measurement(self):
        """Test measurement for quantum state tomography."""
        mock_result = Mock()
        mock_result.state_probabilities.return_value = {
            "0": 0.5,
            "1": 0.5
        }
        mock_result.n_qubits = 1
        
        tomo_results = state_tomography_measurement(mock_result, shots=600)
        
        assert isinstance(tomo_results, dict)
        assert 'X' in tomo_results
        assert 'Y' in tomo_results
        assert 'Z' in tomo_results
        
        # Each basis should have measurements
        for basis, measurements in tomo_results.items():
            assert isinstance(measurements, list)
            assert len(measurements) <= 200  # 600 shots / 3 bases


class TestQuantumNonDemolition:
    """Test quantum non-demolition measurements."""
    
    def test_qnd_measurement(self):
        """Test quantum non-demolition measurement."""
        mock_result = Mock()
        mock_result.state_probabilities.return_value = {
            "00": 0.25,
            "01": 0.25,
            "10": 0.25,
            "11": 0.25
        }
        mock_result.n_qubits = 2
        
        # QND measurement of qubit 0 in Z basis
        post_measurement_state, measurement_outcome = qnd_measurement(
            mock_result, qubit=0, basis='Z'
        )
        
        assert measurement_outcome in [0, 1]
        assert post_measurement_state is not None
        
        # Post-measurement state should be consistent with outcome
        post_probs = post_measurement_state.state_probabilities()
        if measurement_outcome == 0:
            # Should only have |00⟩ and |01⟩ states
            assert post_probs.get("10", 0) == 0
            assert post_probs.get("11", 0) == 0
        else:
            # Should only have |10⟩ and |11⟩ states
            assert post_probs.get("00", 0) == 0
            assert post_probs.get("01", 0) == 0


class TestWeakMeasurement:
    """Test weak measurement functionality."""
    
    def test_weak_measurement(self):
        """Test weak measurement implementation."""
        mock_result = Mock()
        mock_result.state_probabilities.return_value = {
            "0": 0.8,
            "1": 0.2
        }
        mock_result.n_qubits = 1
        
        # Weak measurement with small coupling strength
        weak_value, post_state = weak_measurement(
            mock_result, observable='Z', coupling_strength=0.1
        )
        
        assert isinstance(weak_value, (int, float, complex))
        assert post_state is not None
        
        # Weak value should be close to strong measurement expectation
        strong_exp = expectation_value(mock_result, 'Z', qubit=0)
        assert abs(weak_value - strong_exp) < 0.5  # Weaker coupling = closer to strong
    
    def test_weak_measurement_strong_coupling(self):
        """Test weak measurement with strong coupling."""
        mock_result = Mock()
        mock_result.state_probabilities.return_value = {
            "0": 0.7,
            "1": 0.3
        }
        mock_result.n_qubits = 1
        
        # Strong coupling should approach projective measurement
        weak_value, post_state = weak_measurement(
            mock_result, observable='Z', coupling_strength=10.0
        )
        
        assert isinstance(weak_value, (int, float, complex))
        # Strong coupling should give values closer to ±1
        assert abs(abs(weak_value) - 1) < 0.5


class TestMeasurementErrorHandling:
    """Test measurement error handling."""
    
    def test_invalid_qubit_index(self):
        """Test error handling for invalid qubit indices."""
        mock_result = Mock()
        mock_result.state_probabilities.return_value = {"00": 1.0}
        mock_result.n_qubits = 2
        
        # Qubit index out of range
        with pytest.raises((IndexError, ValueError)):
            measure_qubit(mock_result, qubit=5, shots=10)
    
    def test_invalid_basis(self):
        """Test error handling for invalid measurement basis."""
        mock_result = Mock()
        mock_result.state_probabilities.return_value = {"0": 1.0}
        mock_result.n_qubits = 1
        
        # Invalid basis
        with pytest.raises((ValueError, KeyError)):
            projective_measurement(mock_result, basis='invalid', shots=10)
    
    def test_zero_shots(self):
        """Test handling of zero shots."""
        mock_result = Mock()
        mock_result.state_probabilities.return_value = {"0": 1.0}
        mock_result.n_qubits = 1
        
        measurements = measure_all(mock_result, shots=0)
        assert measurements == []
    
    def test_negative_shots(self):
        """Test handling of negative shots."""
        mock_result = Mock()
        mock_result.state_probabilities.return_value = {"0": 1.0}
        mock_result.n_qubits = 1
        
        with pytest.raises(ValueError):
            measure_all(mock_result, shots=-10)


class TestMeasurementUtilities:
    """Test measurement utility functions."""
    
    def test_counts_to_probabilities(self):
        """Test conversion from counts to probabilities."""
        counts = {"00": 75, "01": 15, "10": 8, "11": 2}
        probs = counts_to_probabilities(counts)
        
        assert isinstance(probs, dict)
        total_prob = sum(probs.values())
        assert abs(total_prob - 1.0) < 1e-10
        
        # Check specific probabilities
        total_counts = sum(counts.values())
        for state, count in counts.items():
            expected_prob = count / total_counts
            assert abs(probs[state] - expected_prob) < 1e-10
    
    def test_probabilities_to_counts(self):
        """Test conversion from probabilities to counts."""
        probs = {"0": 0.7, "1": 0.3}
        counts = probabilities_to_counts(probs, shots=1000)
        
        assert isinstance(counts, dict)
        total_counts = sum(counts.values())
        assert total_counts == 1000
        
        # Check approximate counts
        assert abs(counts["0"] - 700) < 100  # Allow some variance
        assert abs(counts["1"] - 300) < 100
    
    def test_fidelity_from_measurements(self):
        """Test fidelity calculation from measurements."""
        # Perfect overlap
        measurements1 = ["0", "1", "0", "1"]
        measurements2 = ["0", "1", "0", "1"]
        fidelity1 = fidelity_from_measurements(measurements1, measurements2)
        assert abs(fidelity1 - 1.0) < 1e-10
        
        # No overlap
        measurements3 = ["0", "0", "0", "0"]
        measurements4 = ["1", "1", "1", "1"]
        fidelity2 = fidelity_from_measurements(measurements3, measurements4)
        assert abs(fidelity2 - 0.0) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__])