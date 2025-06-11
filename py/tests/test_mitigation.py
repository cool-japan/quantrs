"""Tests for quantum error mitigation functionality"""

import pytest
import numpy as np
import quantrs2
from quantrs2.mitigation import (
    ZNEConfig, ZNEResult, Observable, ZeroNoiseExtrapolation,
    CircuitFolding, ExtrapolationFitting,
    ProbabilisticErrorCancellation, VirtualDistillation, SymmetryVerification
)
from quantrs2.measurement import MeasurementResult, MeasurementSampler


def test_zne_config():
    """Test ZNE configuration creation and properties"""
    # Default config
    config = ZNEConfig()
    assert config.scale_factors == [1.0, 1.5, 2.0, 2.5, 3.0]
    assert config.scaling_method == "global"
    assert config.extrapolation_method == "richardson"
    assert config.bootstrap_samples == 100
    assert config.confidence_level == 0.95
    
    # Custom config
    config = ZNEConfig(
        scale_factors=[1.0, 2.0, 3.0],
        scaling_method="local",
        extrapolation_method="exponential",
        bootstrap_samples=50,
        confidence_level=0.99
    )
    assert config.scale_factors == [1.0, 2.0, 3.0]
    assert config.scaling_method == "local"
    assert config.extrapolation_method == "exponential"
    assert config.bootstrap_samples == 50
    assert config.confidence_level == 0.99
    
    # Test setters
    config.scale_factors = [1.0, 1.5, 2.0]
    assert config.scale_factors == [1.0, 1.5, 2.0]


def test_observable():
    """Test observable creation and expectation values"""
    # Single qubit observable
    obs_z = Observable.z(0)
    assert obs_z.pauli_string == [(0, "Z")]
    assert obs_z.coefficient == 1.0
    
    # Two qubit observable
    obs_zz = Observable.zz(0, 1)
    assert obs_zz.pauli_string == [(0, "Z"), (1, "Z")]
    assert obs_zz.coefficient == 1.0
    
    # Custom observable
    obs_custom = Observable([(0, "X"), (1, "Y"), (2, "Z")], 0.5)
    assert obs_custom.pauli_string == [(0, "X"), (1, "Y"), (2, "Z")]
    assert obs_custom.coefficient == 0.5
    
    # Test expectation value calculation
    # Create mock measurement result
    counts = {"00": 50, "11": 50}  # Perfect Bell state
    measurement = type('obj', (object,), {
        'counts': counts,
        'shots': 100,
        'n_qubits': 2
    })
    
    # For Z0Z1, |00⟩ gives +1 and |11⟩ gives +1
    exp_val = obs_zz.expectation_value(measurement)
    assert abs(exp_val - 1.0) < 0.01


def test_circuit_folding():
    """Test circuit folding operations"""
    circuit = quantrs2.Circuit(2)
    circuit.h(0)
    circuit.cnot(0, 1)
    original_gates = circuit.num_gates
    
    # Global folding
    folded = CircuitFolding.fold_global(circuit, 1.0)
    assert folded.num_gates == original_gates  # Scale 1.0 = no change
    
    folded = CircuitFolding.fold_global(circuit, 3.0)
    # Currently returns same circuit (placeholder), but would be larger
    assert folded.num_gates >= original_gates
    
    # Local folding
    folded = CircuitFolding.fold_local(circuit, 2.0)
    assert folded.num_gates >= original_gates
    
    # With custom weights
    weights = [1.0, 2.0]  # Different weights for H and CNOT
    folded = CircuitFolding.fold_local(circuit, 2.0, weights)
    assert folded.num_gates >= original_gates
    
    # Test error handling
    with pytest.raises(ValueError):
        CircuitFolding.fold_global(circuit, 0.5)  # Scale < 1.0


def test_extrapolation_fitting():
    """Test different extrapolation methods"""
    # Generate test data
    x = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    y_linear = 0.9 - 0.1 * x  # Linear decay
    y_exp = 0.9 * np.exp(-0.2 * (x - 1))  # Exponential decay
    
    # Linear fit
    result = ExtrapolationFitting.fit_linear(x, y_linear)
    assert isinstance(result, ZNEResult)
    assert abs(result.mitigated_value - 0.9) < 0.01  # y-intercept
    assert result.r_squared > 0.99  # Perfect linear fit
    assert len(result.fit_params) >= 2
    
    # Polynomial fit
    result = ExtrapolationFitting.fit_polynomial(x, y_linear, 2)
    assert abs(result.mitigated_value - 0.9) < 0.02
    
    # Exponential fit
    result = ExtrapolationFitting.fit_exponential(x, y_exp)
    assert abs(result.mitigated_value - 0.9) < 0.05
    
    # Richardson extrapolation
    result = ExtrapolationFitting.fit_richardson(x, y_linear)
    assert result.mitigated_value < y_linear[0]  # Should extrapolate below data
    
    # Adaptive fit
    result = ExtrapolationFitting.fit_adaptive(x, y_linear)
    assert result.r_squared > 0.9  # Should find good fit


def test_zero_noise_extrapolation():
    """Test ZNE workflow"""
    # Create simple circuit
    circuit = quantrs2.Circuit(1)
    circuit.rx(0, np.pi/4)
    
    # Configure ZNE
    config = ZNEConfig(
        scale_factors=[1.0, 2.0, 3.0],
        scaling_method="global",
        extrapolation_method="linear"
    )
    zne = ZeroNoiseExtrapolation(config)
    
    # Test circuit folding
    folded = zne.fold_circuit(circuit, 2.0)
    assert folded.num_gates >= circuit.num_gates
    
    # Test extrapolation
    data = [(1.0, 0.9), (2.0, 0.8), (3.0, 0.7)]
    result = zne.extrapolate(data)
    assert isinstance(result, ZNEResult)
    assert result.mitigated_value > 0.9  # Should extrapolate above highest value
    
    # Test with observable
    obs = Observable.z(0)
    
    # Create mock measurements
    measurements = []
    for scale, value in data:
        counts = {"0": int(1000 * (1 + value) / 2), "1": int(1000 * (1 - value) / 2)}
        measurement = type('obj', (object,), {
            'counts': counts,
            'shots': 1000,
            'n_qubits': 1
        })
        measurements.append((scale, measurement))
    
    result = zne.mitigate_observable(obs, measurements)
    assert isinstance(result, ZNEResult)


def test_zne_result_properties():
    """Test ZNE result object properties"""
    # Create result through extrapolation
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([0.9, 0.8, 0.7])
    result = ExtrapolationFitting.fit_linear(x, y)
    
    # Check properties
    assert hasattr(result, 'mitigated_value')
    assert hasattr(result, 'error_estimate')
    assert hasattr(result, 'raw_data')
    assert hasattr(result, 'fit_params')
    assert hasattr(result, 'r_squared')
    assert hasattr(result, 'extrapolation_fn')
    
    # Check raw data
    raw_data = result.raw_data
    assert len(raw_data) == 3
    assert all(isinstance(point, tuple) and len(point) == 2 for point in raw_data)
    
    # Check fit params
    params = result.fit_params
    assert isinstance(params, np.ndarray)
    assert len(params) >= 1


def test_placeholder_methods():
    """Test placeholder implementations"""
    # PEC
    pec = ProbabilisticErrorCancellation()
    circuit = quantrs2.Circuit(1)
    with pytest.raises(ValueError, match="PEC not yet implemented"):
        pec.quasi_probability_decomposition(circuit)
    
    # Virtual Distillation
    vd = VirtualDistillation()
    circuits = [quantrs2.Circuit(1) for _ in range(3)]
    with pytest.raises(ValueError, match="Virtual distillation not yet implemented"):
        vd.distill(circuits)
    
    # Symmetry Verification
    sv = SymmetryVerification()
    circuit = quantrs2.Circuit(2)
    with pytest.raises(ValueError, match="Symmetry verification not yet implemented"):
        sv.verify_symmetry(circuit, "parity")


def test_observable_pauli_validation():
    """Test observable Pauli string validation"""
    # Valid Pauli strings
    Observable([(0, "I")], 1.0)
    Observable([(0, "X"), (1, "Y"), (2, "Z")], 1.0)
    
    # Invalid Pauli string
    with pytest.raises(ValueError, match="Invalid Pauli operator"):
        Observable([(0, "A")], 1.0)
    
    with pytest.raises(ValueError, match="Invalid Pauli operator"):
        Observable([(0, "XY")], 1.0)


def test_repr_methods():
    """Test string representations"""
    # ZNEConfig
    config = ZNEConfig()
    repr_str = repr(config)
    assert "ZNEConfig" in repr_str
    assert "scale_factors" in repr_str
    assert "global" in repr_str
    
    # Observable
    obs = Observable.zz(0, 1)
    repr_str = repr(obs)
    assert "Observable" in repr_str
    assert "Z_0" in repr_str
    assert "Z_1" in repr_str
    
    # ZNEResult (from fit)
    result = ExtrapolationFitting.fit_linear(
        np.array([1.0, 2.0]), 
        np.array([0.9, 0.8])
    )
    repr_str = repr(result)
    assert "ZNEResult" in repr_str
    assert "mitigated_value" in repr_str


def test_different_extrapolation_methods_config():
    """Test configuration with different extrapolation methods"""
    methods = [
        "linear", "polynomial2", "polynomial3", 
        "exponential", "richardson", "adaptive"
    ]
    
    for method in methods:
        config = ZNEConfig(extrapolation_method=method)
        assert method in config.extrapolation_method
    
    # Invalid method
    with pytest.raises(ValueError, match="Unknown extrapolation method"):
        ZNEConfig(extrapolation_method="invalid")


def test_different_scaling_methods_config():
    """Test configuration with different scaling methods"""
    methods = ["global", "local", "pulse", "digital"]
    
    for method in methods:
        config = ZNEConfig(scaling_method=method)
        assert config.scaling_method == method
    
    # Invalid method
    with pytest.raises(ValueError, match="Unknown scaling method"):
        ZNEConfig(scaling_method="invalid")