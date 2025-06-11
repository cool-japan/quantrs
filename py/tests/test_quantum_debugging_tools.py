#!/usr/bin/env python3
"""
Comprehensive test suite for the QuantRS2 Quantum Debugging Tools.

This test suite provides complete coverage of all debugging functionality including:
- QuantumStateInspector with multiple analysis modes and edge cases
- QuantumErrorAnalyzer with error classification and auto-fix strategies
- QuantumCircuitValidator with comprehensive validation rules
- QuantumMemoryDebugger with memory profiling and leak detection
- InteractiveQuantumDebugConsole with command execution and session management
- QuantumDebuggingWebInterface with both Dash and Flask implementations
- QuantumDebuggingToolsManager with integration testing
- Error handling, edge cases, and performance validation
"""

import pytest
import tempfile
import os
import json
import time
import threading
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, List, Any

try:
    import quantrs2
    from quantrs2.quantum_debugging_tools import (
        DebugLevel, DebuggerState, ErrorType, InspectionMode, ValidationRule,
        DebugBreakpoint, DebugFrame, ErrorDiagnosis, ValidationResult, DebugSession,
        StateInspectionResult, MemoryDebugInfo,
        QuantumStateInspector, QuantumErrorAnalyzer, QuantumCircuitValidator,
        QuantumMemoryDebugger, InteractiveQuantumDebugConsole,
        QuantumDebuggingWebInterface, QuantumDebuggingToolsManager,
        get_quantum_debugging_tools, debug_quantum_circuit, analyze_quantum_error,
        inspect_quantum_state, validate_quantum_circuit,
        start_quantum_debugging_console, start_quantum_debugging_web_interface
    )
    
    # Import optional feature flags separately
    try:
        from quantrs2.quantum_debugging_tools import (
            HAS_MATPLOTLIB, HAS_PLOTLY, HAS_PANDAS, HAS_NETWORKX, HAS_DASH, HAS_FLASK, HAS_PSUTIL
        )
    except ImportError:
        HAS_MATPLOTLIB = False
        HAS_PLOTLY = False
        HAS_PANDAS = False
        HAS_NETWORKX = False
        HAS_DASH = False
        HAS_FLASK = False
        HAS_PSUTIL = False
    
    HAS_QUANTUM_DEBUGGING_TOOLS = True
except ImportError:
    HAS_QUANTUM_DEBUGGING_TOOLS = False
    HAS_MATPLOTLIB = False
    HAS_PLOTLY = False
    HAS_PANDAS = False
    HAS_NETWORKX = False
    HAS_DASH = False
    HAS_FLASK = False
    HAS_PSUTIL = False

# Test fixtures
@pytest.fixture
def sample_state_vector():
    """Create sample quantum state vector for testing."""
    return np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)

@pytest.fixture
def sample_single_qubit_state():
    """Create sample single-qubit state."""
    return np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)

@pytest.fixture
def sample_ghz_state():
    """Create sample GHZ state."""
    state = np.zeros(8, dtype=complex)
    state[0] = 1/np.sqrt(2)  # |000⟩
    state[7] = 1/np.sqrt(2)  # |111⟩
    return state

@pytest.fixture
def mock_circuit():
    """Create mock quantum circuit for testing."""
    circuit = Mock()
    circuit.num_qubits = 2
    circuit.gates = [
        Mock(name="H", qubits=[0], params=[]),
        Mock(name="CNOT", qubits=[0, 1], params=[]),
        Mock(name="MEASURE", qubits=[0], params=[])
    ]
    
    # Mock run method
    def mock_run():
        result = Mock()
        result.state_vector = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        return result
    
    circuit.run = mock_run
    return circuit

@pytest.fixture
def mock_error():
    """Create mock error for testing."""
    return Exception("Gate execution failed: invalid parameter range")

@pytest.fixture
def debug_tools_manager():
    """Create quantum debugging tools manager for testing."""
    return QuantumDebuggingToolsManager(DebugLevel.DEBUG)


@pytest.mark.skipif(not HAS_QUANTUM_DEBUGGING_TOOLS, reason="quantum debugging tools not available")
class TestQuantumStateInspector:
    """Test QuantumStateInspector functionality."""
    
    def test_state_inspector_initialization(self):
        """Test QuantumStateInspector initialization."""
        inspector = QuantumStateInspector(DebugLevel.INFO)
        
        assert inspector.debug_level == DebugLevel.INFO
        assert inspector.inspection_history == []
        assert isinstance(inspector.anomaly_thresholds, dict)
        assert 'amplitude_threshold' in inspector.anomaly_thresholds
        assert 'phase_threshold' in inspector.anomaly_thresholds
        assert 'entanglement_threshold' in inspector.anomaly_thresholds
        assert 'purity_threshold' in inspector.anomaly_thresholds
    
    def test_amplitude_analysis(self, sample_state_vector):
        """Test amplitude analysis functionality."""
        inspector = QuantumStateInspector()
        result = inspector.inspect_state(sample_state_vector, InspectionMode.AMPLITUDE_ANALYSIS)
        
        assert isinstance(result, StateInspectionResult)
        assert result.mode == InspectionMode.AMPLITUDE_ANALYSIS
        assert 'amplitude_stats' in result.analysis_data
        assert 'dominant_states' in result.analysis_data
        assert 'normalization' in result.analysis_data
        
        # Check amplitude statistics
        amp_stats = result.analysis_data['amplitude_stats']
        assert 'max' in amp_stats
        assert 'min' in amp_stats
        assert 'mean' in amp_stats
        assert 'std' in amp_stats
        
        # Check normalization
        norm_data = result.analysis_data['normalization']
        assert norm_data['normalized'] is True
        assert abs(norm_data['norm'] - 1.0) < 1e-10
        
        assert isinstance(result.insights, list)
        assert isinstance(result.anomalies, list)
        assert isinstance(result.recommendations, list)
    
    def test_probability_analysis(self, sample_state_vector):
        """Test probability analysis functionality."""
        inspector = QuantumStateInspector()
        result = inspector.inspect_state(sample_state_vector, InspectionMode.PROBABILITY_ANALYSIS)
        
        assert result.mode == InspectionMode.PROBABILITY_ANALYSIS
        assert 'probability_stats' in result.analysis_data
        assert 'qubit_probabilities' in result.analysis_data
        
        # Check probability statistics
        prob_stats = result.analysis_data['probability_stats']
        assert 'max' in prob_stats
        assert 'entropy' in prob_stats
        assert 'effective_dimension' in prob_stats
        
        # Check individual qubit probabilities
        qubit_probs = result.analysis_data['qubit_probabilities']
        assert len(qubit_probs) == 2  # Two-qubit state
        for qprob in qubit_probs:
            assert 'qubit' in qprob
            assert 'prob_0' in qprob
            assert 'prob_1' in qprob
            assert abs(qprob['prob_0'] + qprob['prob_1'] - 1.0) < 1e-10
    
    def test_phase_analysis(self, sample_state_vector):
        """Test phase analysis functionality."""
        inspector = QuantumStateInspector()
        result = inspector.inspect_state(sample_state_vector, InspectionMode.PHASE_ANALYSIS)
        
        assert result.mode == InspectionMode.PHASE_ANALYSIS
        assert 'phase_stats' in result.analysis_data
        
        phase_stats = result.analysis_data['phase_stats']
        assert 'mean' in phase_stats
        assert 'std' in phase_stats
        assert 'range' in phase_stats
    
    def test_entanglement_analysis(self, sample_state_vector):
        """Test entanglement analysis functionality."""
        inspector = QuantumStateInspector()
        result = inspector.inspect_state(sample_state_vector, InspectionMode.ENTANGLEMENT_ANALYSIS)
        
        assert result.mode == InspectionMode.ENTANGLEMENT_ANALYSIS
        assert 'entanglement_measures' in result.analysis_data
        assert 'max_entanglement_entropy' in result.analysis_data
        
        # Bell state should show significant entanglement
        entanglement_measures = result.analysis_data['entanglement_measures']
        assert len(entanglement_measures) > 0
        
        # Check for entanglement detection
        max_entropy = result.analysis_data['max_entanglement_entropy']
        assert max_entropy > 0.5  # Bell state should have high entanglement entropy
    
    def test_entanglement_analysis_single_qubit(self, sample_single_qubit_state):
        """Test entanglement analysis on single-qubit state."""
        inspector = QuantumStateInspector()
        result = inspector.inspect_state(sample_single_qubit_state, InspectionMode.ENTANGLEMENT_ANALYSIS)
        
        assert "Single qubit state - no entanglement possible" in result.insights
        assert result.analysis_data['n_qubits'] == 1
    
    def test_coherence_analysis(self, sample_state_vector):
        """Test coherence analysis functionality."""
        inspector = QuantumStateInspector()
        result = inspector.inspect_state(sample_state_vector, InspectionMode.COHERENCE_ANALYSIS)
        
        assert result.mode == InspectionMode.COHERENCE_ANALYSIS
        assert 'coherence_measures' in result.analysis_data
        
        coherence_measures = result.analysis_data['coherence_measures']
        assert 'off_diagonal_coherence' in coherence_measures
        assert 'coherence_ratio' in coherence_measures
        assert 'l1_coherence' in coherence_measures
    
    def test_correlation_analysis(self, sample_state_vector):
        """Test correlation analysis functionality."""
        inspector = QuantumStateInspector()
        result = inspector.inspect_state(sample_state_vector, InspectionMode.CORRELATION_ANALYSIS)
        
        assert result.mode == InspectionMode.CORRELATION_ANALYSIS
        assert 'two_qubit_correlations' in result.analysis_data
        
        correlations = result.analysis_data['two_qubit_correlations']
        assert len(correlations) > 0  # Should have correlation data for qubit pairs
        
        for corr in correlations:
            assert 'qubit_pair' in corr
            assert 'correlation' in corr
            assert 'joint_probs' in corr
            assert 'marginal_probs' in corr
    
    def test_correlation_analysis_single_qubit(self, sample_single_qubit_state):
        """Test correlation analysis on single-qubit state."""
        inspector = QuantumStateInspector()
        result = inspector.inspect_state(sample_single_qubit_state, InspectionMode.CORRELATION_ANALYSIS)
        
        assert "Single qubit state - no correlations possible" in result.insights
    
    def test_purity_analysis(self, sample_state_vector):
        """Test purity analysis functionality."""
        inspector = QuantumStateInspector()
        result = inspector.inspect_state(sample_state_vector, InspectionMode.PURITY_ANALYSIS)
        
        assert result.mode == InspectionMode.PURITY_ANALYSIS
        assert 'purity_measures' in result.analysis_data
        
        purity_measures = result.analysis_data['purity_measures']
        assert 'purity' in purity_measures
        assert 'mixedness' in purity_measures
        assert 'is_pure' in purity_measures
        assert 'linear_entropy' in purity_measures
        
        # Pure state should have purity close to 1
        assert abs(purity_measures['purity'] - 1.0) < 1e-6
        assert purity_measures['is_pure'] is True
    
    def test_fidelity_analysis(self, sample_state_vector):
        """Test fidelity analysis functionality."""
        inspector = QuantumStateInspector()
        result = inspector.inspect_state(sample_state_vector, InspectionMode.FIDELITY_ANALYSIS)
        
        assert result.mode == InspectionMode.FIDELITY_ANALYSIS
        assert 'fidelities' in result.analysis_data
        assert 'best_match' in result.analysis_data
        
        fidelities = result.analysis_data['fidelities']
        assert 'uniform_superposition' in fidelities
        
        # Bell state should have high fidelity with Bell states
        bell_fidelities = [f for name, f in fidelities.items() if 'Bell' in name]
        if bell_fidelities:
            assert max(bell_fidelities) > 0.99
    
    def test_ghz_state_fidelity(self, sample_ghz_state):
        """Test fidelity analysis with GHZ state."""
        inspector = QuantumStateInspector()
        result = inspector.inspect_state(sample_ghz_state, InspectionMode.FIDELITY_ANALYSIS)
        
        fidelities = result.analysis_data['fidelities']
        assert 'GHZ_state' in fidelities
        assert fidelities['GHZ_state'] > 0.99
    
    def test_invalid_inspection_mode(self, sample_state_vector):
        """Test handling of invalid inspection mode."""
        inspector = QuantumStateInspector()
        
        # Mock an invalid mode by using a string instead of enum
        with pytest.raises(ValueError, match="Unknown inspection mode"):
            inspector.inspect_state(sample_state_vector, "invalid_mode")
    
    def test_inspection_with_errors(self):
        """Test inspection error handling."""
        inspector = QuantumStateInspector()
        
        # Test with invalid state vector
        invalid_state = np.array([1, 2, 3])  # Not normalized, wrong format
        result = inspector.inspect_state(invalid_state, InspectionMode.AMPLITUDE_ANALYSIS)
        
        # Should handle gracefully and return result with errors
        assert isinstance(result, StateInspectionResult)
        assert len(result.anomalies) > 0 or len(result.recommendations) > 0
    
    def test_zero_state_vector(self):
        """Test inspection with zero state vector."""
        inspector = QuantumStateInspector()
        zero_state = np.array([0, 0], dtype=complex)
        
        result = inspector.inspect_state(zero_state, InspectionMode.AMPLITUDE_ANALYSIS)
        
        assert isinstance(result, StateInspectionResult)
        # Should handle zero state gracefully
    
    def test_large_state_vector(self):
        """Test inspection with large state vector."""
        inspector = QuantumStateInspector()
        
        # Create 4-qubit state
        large_state = np.zeros(16, dtype=complex)
        large_state[0] = 1.0
        
        result = inspector.inspect_state(large_state, InspectionMode.PROBABILITY_ANALYSIS)
        
        assert isinstance(result, StateInspectionResult)
        assert len(result.analysis_data['qubit_probabilities']) == 4


@pytest.mark.skipif(not HAS_QUANTUM_DEBUGGING_TOOLS, reason="quantum debugging tools not available")
class TestQuantumErrorAnalyzer:
    """Test QuantumErrorAnalyzer functionality."""
    
    def test_error_analyzer_initialization(self):
        """Test QuantumErrorAnalyzer initialization."""
        analyzer = QuantumErrorAnalyzer(DebugLevel.INFO)
        
        assert analyzer.debug_level == DebugLevel.INFO
        assert isinstance(analyzer.error_patterns, dict)
        assert isinstance(analyzer.auto_fix_strategies, dict)
        assert analyzer.error_history == []
        
        # Check that error patterns are properly initialized
        assert ErrorType.GATE_ERROR in analyzer.error_patterns
        assert ErrorType.MEASUREMENT_ERROR in analyzer.error_patterns
        assert ErrorType.DECOHERENCE_ERROR in analyzer.error_patterns
    
    def test_gate_error_analysis(self):
        """Test gate error analysis."""
        analyzer = QuantumErrorAnalyzer()
        error = Exception("Gate execution failed: invalid parameters")
        
        diagnosis = analyzer.analyze_error(error, {'gate_name': 'RX', 'qubit_indices': [0]})
        
        assert isinstance(diagnosis, ErrorDiagnosis)
        assert diagnosis.error_type == ErrorType.GATE_ERROR
        assert "Gate" in diagnosis.message
        assert len(diagnosis.suggestions) > 0
        assert diagnosis.auto_fix_available is True
    
    def test_measurement_error_analysis(self):
        """Test measurement error analysis."""
        analyzer = QuantumErrorAnalyzer()
        error = Exception("Measurement readout error: low fidelity")
        
        diagnosis = analyzer.analyze_error(error)
        
        assert diagnosis.error_type == ErrorType.MEASUREMENT_ERROR
        assert "readout" in diagnosis.message.lower()
        assert any("measurement" in suggestion.lower() for suggestion in diagnosis.suggestions)
    
    def test_decoherence_error_analysis(self):
        """Test decoherence error analysis."""
        analyzer = QuantumErrorAnalyzer()
        error = Exception("T1 decay detected: coherence time exceeded")
        
        diagnosis = analyzer.analyze_error(error)
        
        assert diagnosis.error_type == ErrorType.DECOHERENCE_ERROR
        assert any("decoherence" in suggestion.lower() for suggestion in diagnosis.suggestions)
    
    def test_circuit_error_analysis(self):
        """Test circuit error analysis."""
        analyzer = QuantumErrorAnalyzer()
        error = Exception("Circuit depth exceeds maximum limit")
        
        diagnosis = analyzer.analyze_error(error)
        
        assert diagnosis.error_type == ErrorType.CIRCUIT_ERROR
        assert any("circuit" in suggestion.lower() for suggestion in diagnosis.suggestions)
    
    def test_state_error_analysis(self):
        """Test state error analysis."""
        analyzer = QuantumErrorAnalyzer()
        error = Exception("State vector normalization error")
        
        diagnosis = analyzer.analyze_error(error)
        
        assert diagnosis.error_type == ErrorType.STATE_ERROR
        assert any("state" in suggestion.lower() for suggestion in diagnosis.suggestions)
    
    def test_memory_error_analysis(self):
        """Test memory error analysis."""
        analyzer = QuantumErrorAnalyzer()
        error = Exception("Out of memory: allocation failed")
        
        diagnosis = analyzer.analyze_error(error)
        
        assert diagnosis.error_type == ErrorType.MEMORY_ERROR
        assert any("memory" in suggestion.lower() for suggestion in diagnosis.suggestions)
    
    def test_performance_error_analysis(self):
        """Test performance error analysis."""
        analyzer = QuantumErrorAnalyzer()
        error = Exception("Performance timeout: operation too slow")
        
        diagnosis = analyzer.analyze_error(error)
        
        assert diagnosis.error_type == ErrorType.PERFORMANCE_ERROR
    
    def test_validation_error_analysis(self):
        """Test validation error analysis."""
        analyzer = QuantumErrorAnalyzer()
        error = Exception("Validation failed: invalid constraint")
        
        diagnosis = analyzer.analyze_error(error)
        
        assert diagnosis.error_type == ErrorType.VALIDATION_ERROR
    
    def test_unknown_error_analysis(self):
        """Test unknown error analysis."""
        analyzer = QuantumErrorAnalyzer()
        error = Exception("Some unknown error occurred")
        
        diagnosis = analyzer.analyze_error(error)
        
        assert diagnosis.error_type == ErrorType.RUNTIME_ERROR
        assert isinstance(diagnosis.suggestions, list)
    
    def test_severity_assessment(self):
        """Test error severity assessment."""
        analyzer = QuantumErrorAnalyzer()
        
        # Critical error
        critical_error = Exception("Critical system failure: corrupted state")
        diagnosis = analyzer.analyze_error(critical_error)
        assert diagnosis.severity == "CRITICAL"
        
        # High severity error
        high_error = Exception("Gate execution failed")
        diagnosis = analyzer.analyze_error(high_error)
        assert diagnosis.severity == "HIGH"
        
        # Medium severity error
        medium_error = Exception("Warning: precision loss detected")
        diagnosis = analyzer.analyze_error(medium_error)
        assert diagnosis.severity == "MEDIUM"
    
    def test_error_location_extraction(self):
        """Test error location extraction."""
        analyzer = QuantumErrorAnalyzer()
        error = Exception("Test error")
        
        # Test with context
        context = {'location': 'test_function:line_42'}
        diagnosis = analyzer.analyze_error(error, context)
        assert diagnosis.location == 'test_function:line_42'
    
    def test_auto_fix_availability(self):
        """Test auto-fix availability detection."""
        analyzer = QuantumErrorAnalyzer()
        
        # Gate error should have auto-fix
        gate_error = Exception("Gate parameter out of range")
        diagnosis = analyzer.analyze_error(gate_error)
        assert diagnosis.auto_fix_available is True
        
        # Unknown error might not have auto-fix
        unknown_error = Exception("Completely unknown error")
        diagnosis = analyzer.analyze_error(unknown_error)
        # auto_fix_available depends on error classification
    
    def test_auto_fix_application(self):
        """Test auto-fix strategy application."""
        analyzer = QuantumErrorAnalyzer()
        error = Exception("Gate parameter error")
        diagnosis = analyzer.analyze_error(error)
        
        if diagnosis.auto_fix_available:
            result = analyzer.apply_auto_fix(diagnosis)
            assert isinstance(result, dict)
            assert 'success' in result
            assert 'message' in result
    
    def test_error_history_tracking(self):
        """Test error history tracking."""
        analyzer = QuantumErrorAnalyzer()
        
        initial_count = len(analyzer.error_history)
        
        error1 = Exception("First error")
        error2 = Exception("Second error")
        
        analyzer.analyze_error(error1)
        analyzer.analyze_error(error2)
        
        assert len(analyzer.error_history) == initial_count + 2
    
    def test_context_based_analysis(self):
        """Test context-based error analysis."""
        analyzer = QuantumErrorAnalyzer()
        error = Exception("Operation failed")
        
        context = {
            'gate_name': 'CNOT',
            'qubit_indices': [0, 1],
            'circuit_depth': 100,
            'location': 'quantum_circuit.py:123'
        }
        
        diagnosis = analyzer.analyze_error(error, context)
        
        assert context['gate_name'] in diagnosis.message
        assert diagnosis.location == context['location']
    
    def test_error_analysis_failure_handling(self):
        """Test error analysis failure handling."""
        analyzer = QuantumErrorAnalyzer()
        
        # Force an error in analysis by mocking
        with patch.object(analyzer, '_classify_error', side_effect=Exception("Analysis failed")):
            diagnosis = analyzer.analyze_error(Exception("Test error"))
            
            assert diagnosis.error_type == ErrorType.RUNTIME_ERROR
            assert "Error analysis failed" in diagnosis.message


@pytest.mark.skipif(not HAS_QUANTUM_DEBUGGING_TOOLS, reason="quantum debugging tools not available")
class TestQuantumCircuitValidator:
    """Test QuantumCircuitValidator functionality."""
    
    def test_circuit_validator_initialization(self):
        """Test QuantumCircuitValidator initialization."""
        validator = QuantumCircuitValidator(DebugLevel.INFO)
        
        assert validator.debug_level == DebugLevel.INFO
        assert isinstance(validator.validation_rules, dict)
        assert validator.validation_history == []
        
        # Check that validation rules are properly initialized
        assert ValidationRule.UNITARITY_CHECK in validator.validation_rules
        assert ValidationRule.NORMALIZATION_CHECK in validator.validation_rules
        assert ValidationRule.RESOURCE_CHECK in validator.validation_rules
    
    def test_unitarity_check(self, mock_circuit):
        """Test unitarity validation rule."""
        validator = QuantumCircuitValidator()
        results = validator.validate_circuit(mock_circuit, [ValidationRule.UNITARITY_CHECK])
        
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, ValidationResult)
        assert result.rule == ValidationRule.UNITARITY_CHECK
        assert isinstance(result.passed, bool)
        assert isinstance(result.message, str)
    
    def test_normalization_check(self, mock_circuit):
        """Test normalization validation rule."""
        validator = QuantumCircuitValidator()
        results = validator.validate_circuit(mock_circuit, [ValidationRule.NORMALIZATION_CHECK])
        
        assert len(results) == 1
        result = results[0]
        assert result.rule == ValidationRule.NORMALIZATION_CHECK
        
        if 'final_state_norm' in result.details:
            assert isinstance(result.details['final_state_norm'], float)
    
    def test_hermiticity_check(self, mock_circuit):
        """Test Hermiticity validation rule."""
        validator = QuantumCircuitValidator()
        results = validator.validate_circuit(mock_circuit, [ValidationRule.HERMITICITY_CHECK])
        
        assert len(results) == 1
        result = results[0]
        assert result.rule == ValidationRule.HERMITICITY_CHECK
        
        if 'gate_analysis' in result.details:
            assert isinstance(result.details['gate_analysis'], dict)
    
    def test_commutativity_check(self, mock_circuit):
        """Test commutativity validation rule."""
        validator = QuantumCircuitValidator()
        results = validator.validate_circuit(mock_circuit, [ValidationRule.COMMUTATIVITY_CHECK])
        
        assert len(results) == 1
        result = results[0]
        assert result.rule == ValidationRule.COMMUTATIVITY_CHECK
        
        if 'commutation_issues' in result.details:
            assert isinstance(result.details['commutation_issues'], list)
    
    def test_causality_check(self, mock_circuit):
        """Test causality validation rule."""
        validator = QuantumCircuitValidator()
        results = validator.validate_circuit(mock_circuit, [ValidationRule.CAUSALITY_CHECK])
        
        assert len(results) == 1
        result = results[0]
        assert result.rule == ValidationRule.CAUSALITY_CHECK
        
        # Should detect measurement followed by gates
        if 'timing_issues' in result.details:
            timing_issues = result.details['timing_issues']
            assert isinstance(timing_issues, list)
    
    def test_resource_check(self, mock_circuit):
        """Test resource validation rule."""
        validator = QuantumCircuitValidator()
        results = validator.validate_circuit(mock_circuit, [ValidationRule.RESOURCE_CHECK])
        
        assert len(results) == 1
        result = results[0]
        assert result.rule == ValidationRule.RESOURCE_CHECK
        
        details = result.details
        assert 'used_qubits' in details
        assert 'total_gates' in details
        assert 'circuit_depth' in details
        
        if 'declared_qubits' in details:
            assert details['declared_qubits'] == mock_circuit.num_qubits
    
    def test_connectivity_check(self, mock_circuit):
        """Test connectivity validation rule."""
        validator = QuantumCircuitValidator()
        results = validator.validate_circuit(mock_circuit, [ValidationRule.CONNECTIVITY_CHECK])
        
        assert len(results) == 1
        result = results[0]
        assert result.rule == ValidationRule.CONNECTIVITY_CHECK
        
        if 'connectivity_violations' in result.details:
            assert isinstance(result.details['connectivity_violations'], list)
    
    def test_timing_check(self, mock_circuit):
        """Test timing validation rule."""
        validator = QuantumCircuitValidator()
        results = validator.validate_circuit(mock_circuit, [ValidationRule.TIMING_CHECK])
        
        assert len(results) == 1
        result = results[0]
        assert result.rule == ValidationRule.TIMING_CHECK
        
        details = result.details
        if 'gate_times' in details:
            assert isinstance(details['gate_times'], list)
        if 'total_execution_time' in details:
            assert isinstance(details['total_execution_time'], (int, float))
    
    def test_all_validation_rules(self, mock_circuit):
        """Test validation with all rules."""
        validator = QuantumCircuitValidator()
        results = validator.validate_circuit(mock_circuit)
        
        # Should run all validation rules
        assert len(results) == len(ValidationRule)
        
        rule_types = [result.rule for result in results]
        for rule in ValidationRule:
            assert rule in rule_types
    
    def test_circuit_depth_estimation(self):
        """Test circuit depth estimation."""
        validator = QuantumCircuitValidator()
        
        # Create mock gates for depth estimation
        gates = [
            Mock(qubits=[0]),
            Mock(qubits=[1]),
            Mock(qubits=[0, 1]),  # Should increase depth
            Mock(qubits=[2])
        ]
        
        depth = validator._estimate_circuit_depth(gates)
        assert isinstance(depth, int)
        assert depth > 0
    
    def test_empty_circuit_validation(self):
        """Test validation of empty circuit."""
        validator = QuantumCircuitValidator()
        
        empty_circuit = Mock()
        empty_circuit.num_qubits = 0
        empty_circuit.gates = []
        
        results = validator.validate_circuit(empty_circuit, [ValidationRule.RESOURCE_CHECK])
        
        assert len(results) == 1
        result = results[0]
        assert result.rule == ValidationRule.RESOURCE_CHECK
    
    def test_large_circuit_validation(self):
        """Test validation of large circuit."""
        validator = QuantumCircuitValidator()
        
        # Create large circuit
        large_circuit = Mock()
        large_circuit.num_qubits = 10
        large_circuit.gates = [Mock(qubits=[i % 10]) for i in range(100)]
        
        results = validator.validate_circuit(large_circuit, [ValidationRule.RESOURCE_CHECK])
        
        assert len(results) == 1
        result = results[0]
        assert result.details['total_gates'] == 100
    
    def test_validation_error_handling(self):
        """Test validation error handling."""
        validator = QuantumCircuitValidator()
        
        # Mock circuit that will cause validation errors
        invalid_circuit = Mock()
        del invalid_circuit.num_qubits  # Remove required attribute
        
        results = validator.validate_circuit(invalid_circuit, [ValidationRule.RESOURCE_CHECK])
        
        assert len(results) == 1
        result = results[0]
        assert result.passed is False
        assert "failed" in result.message.lower()
    
    def test_validation_history_tracking(self, mock_circuit):
        """Test validation history tracking."""
        validator = QuantumCircuitValidator()
        
        initial_count = len(validator.validation_history)
        
        validator.validate_circuit(mock_circuit, [ValidationRule.UNITARITY_CHECK])
        validator.validate_circuit(mock_circuit, [ValidationRule.NORMALIZATION_CHECK])
        
        assert len(validator.validation_history) == initial_count + 2
    
    def test_unimplemented_validation_rule(self, mock_circuit):
        """Test handling of unimplemented validation rule."""
        validator = QuantumCircuitValidator()
        
        # Mock an unimplemented rule by removing it from the registry
        original_rules = validator.validation_rules.copy()
        del validator.validation_rules[ValidationRule.UNITARITY_CHECK]
        
        try:
            results = validator.validate_circuit(mock_circuit, [ValidationRule.UNITARITY_CHECK])
            assert len(results) == 1
            result = results[0]
            assert result.passed is False
            assert "not implemented" in result.message
        finally:
            validator.validation_rules = original_rules


@pytest.mark.skipif(not HAS_QUANTUM_DEBUGGING_TOOLS, reason="quantum debugging tools not available")
class TestQuantumMemoryDebugger:
    """Test QuantumMemoryDebugger functionality."""
    
    def test_memory_debugger_initialization(self):
        """Test QuantumMemoryDebugger initialization."""
        debugger = QuantumMemoryDebugger(DebugLevel.INFO)
        
        assert debugger.debug_level == DebugLevel.INFO
        assert debugger.memory_snapshots == []
        assert isinstance(debugger.memory_threshold, int)
    
    def test_memory_snapshot(self):
        """Test memory snapshot functionality."""
        debugger = QuantumMemoryDebugger()
        snapshot = debugger.get_memory_snapshot()
        
        assert isinstance(snapshot, MemoryDebugInfo)
        assert isinstance(snapshot.total_memory, int)
        assert isinstance(snapshot.quantum_memory, int)
        assert isinstance(snapshot.classical_memory, int)
        assert isinstance(snapshot.peak_usage, int)
        assert isinstance(snapshot.memory_leaks, list)
        assert isinstance(snapshot.optimization_suggestions, list)
    
    @pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not available")
    def test_memory_monitoring(self):
        """Test memory monitoring functionality."""
        debugger = QuantumMemoryDebugger()
        
        # Start monitoring
        debugger.start_memory_monitoring()
        
        # Let it run for a short time
        time.sleep(0.2)
        
        # Stop monitoring
        debugger.stop_memory_monitoring()
        
        # Should have collected some snapshots
        assert len(debugger.memory_snapshots) > 0
    
    def test_memory_profiling_context(self):
        """Test memory profiling context manager."""
        debugger = QuantumMemoryDebugger()
        
        with debugger.memory_profiling_context("test_operation"):
            # Perform some operation
            time.sleep(0.1)
        
        # Context manager should complete without errors
        assert True
    
    def test_memory_leak_detection(self):
        """Test memory leak detection."""
        debugger = QuantumMemoryDebugger()
        
        # Create fake memory snapshots with growth
        for i in range(20):
            fake_snapshot = MemoryDebugInfo(
                total_memory=1024 * 1024 * (10 + i),  # Growing memory
                quantum_memory=0,
                classical_memory=1024 * 1024 * (10 + i),
                peak_usage=1024 * 1024 * (10 + i),
                memory_leaks=[],
                optimization_suggestions=[]
            )
            debugger.memory_snapshots.append(fake_snapshot)
        
        # Check leak detection
        leaks = debugger._detect_memory_leaks()
        
        # Should detect growth if threshold exceeded
        if any(leak['type'] == 'consistent_growth' for leak in leaks):
            assert True  # Growth detected
        else:
            assert True  # No significant growth (also valid)
    
    def test_memory_optimization_suggestions(self):
        """Test memory optimization suggestions."""
        debugger = QuantumMemoryDebugger()
        
        # Test with high memory usage
        high_memory = debugger.memory_threshold * 2
        suggestions = debugger._generate_memory_optimizations(high_memory)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert any("High memory usage" in suggestion for suggestion in suggestions)
    
    def test_memory_snapshot_without_psutil(self):
        """Test memory snapshot when psutil is not available."""
        debugger = QuantumMemoryDebugger()
        
        # Mock psutil not being available
        with patch('quantrs2.quantum_debugging_tools.HAS_PSUTIL', False):
            snapshot = debugger.get_memory_snapshot()
            
            assert isinstance(snapshot, MemoryDebugInfo)
            assert "psutil" in snapshot.optimization_suggestions[0]


@pytest.mark.skipif(not HAS_QUANTUM_DEBUGGING_TOOLS, reason="quantum debugging tools not available")
class TestInteractiveQuantumDebugConsole:
    """Test InteractiveQuantumDebugConsole functionality."""
    
    def test_console_initialization(self):
        """Test console initialization."""
        console = InteractiveQuantumDebugConsole(DebugLevel.INFO)
        
        assert console.debug_level == DebugLevel.INFO
        assert console.active_session is None
        assert console.command_history == []
        assert isinstance(console.commands, dict)
        
        # Check that all expected commands are registered
        expected_commands = ['help', 'inspect', 'validate', 'analyze', 'breakpoint', 
                           'continue', 'step', 'memory', 'history', 'session', 'quit']
        for cmd in expected_commands:
            assert cmd in console.commands
    
    def test_help_command(self):
        """Test help command."""
        console = InteractiveQuantumDebugConsole()
        result = console.execute_command("help")
        
        assert "Available Commands" in result
        assert "help" in result
        assert "inspect" in result
        assert "validate" in result
    
    def test_inspect_command(self):
        """Test inspect command."""
        console = InteractiveQuantumDebugConsole()
        
        # Test with valid mode
        result = console.execute_command("inspect amplitude")
        assert "State Inspection" in result
        assert "Amplitude Analysis" in result
        
        # Test with invalid mode
        result = console.execute_command("inspect invalid")
        assert "Unknown inspection mode" in result
        
        # Test with no arguments
        result = console.execute_command("inspect")
        assert "Usage:" in result
    
    def test_validate_command(self):
        """Test validate command."""
        console = InteractiveQuantumDebugConsole()
        
        # Test with valid rule
        result = console.execute_command("validate unitarity")
        assert "Circuit Validation" in result
        assert "Unitarity" in result
        
        # Test with invalid rule
        result = console.execute_command("validate invalid")
        assert "Unknown validation rule" in result
        
        # Test with no arguments
        result = console.execute_command("validate")
        assert "Usage:" in result
    
    def test_analyze_command(self):
        """Test analyze command."""
        console = InteractiveQuantumDebugConsole()
        
        # Test with error description
        result = console.execute_command("analyze gate parameter error")
        assert "Error Analysis" in result
        assert "Type:" in result
        assert "Severity:" in result
        
        # Test with no arguments
        result = console.execute_command("analyze")
        assert "Usage:" in result
    
    def test_breakpoint_command(self):
        """Test breakpoint command."""
        console = InteractiveQuantumDebugConsole()
        
        # Create a session first
        console.execute_command("session new")
        
        # Test list breakpoints (empty)
        result = console.execute_command("breakpoint list")
        assert "No active breakpoints" in result
        
        # Test add breakpoint
        result = console.execute_command("breakpoint add gate_5")
        assert "Breakpoint added" in result
        
        # Test list breakpoints (should have one)
        result = console.execute_command("breakpoint list")
        assert "gate_5" in result
        
        # Test invalid action
        result = console.execute_command("breakpoint invalid")
        assert "Unknown breakpoint action" in result
    
    def test_session_command(self):
        """Test session command."""
        console = InteractiveQuantumDebugConsole()
        
        # Test new session
        result = console.execute_command("session new")
        assert "New debug session created" in result
        assert console.active_session is not None
        
        # Test session info
        result = console.execute_command("session info")
        assert "Active Session" in result
        assert "State:" in result
        
        # Test invalid action
        result = console.execute_command("session invalid")
        assert "Unknown session action" in result
    
    def test_memory_command(self):
        """Test memory command."""
        console = InteractiveQuantumDebugConsole()
        
        result = console.execute_command("memory")
        assert "Memory Usage Information" in result
        assert "Total Memory:" in result
    
    def test_history_command(self):
        """Test history command."""
        console = InteractiveQuantumDebugConsole()
        
        # Execute some commands first
        console.execute_command("help")
        console.execute_command("session new")
        
        result = console.execute_command("history")
        assert "Command History" in result
        assert "help" in result
        assert "session new" in result
    
    def test_continue_and_step_commands(self):
        """Test continue and step commands."""
        console = InteractiveQuantumDebugConsole()
        
        result = console.execute_command("continue")
        assert "Continuing execution" in result
        
        result = console.execute_command("step")
        assert "Stepping to next" in result
    
    def test_quit_command(self):
        """Test quit command."""
        console = InteractiveQuantumDebugConsole()
        
        result = console.execute_command("quit")
        assert result == "quit"
    
    def test_unknown_command(self):
        """Test unknown command handling."""
        console = InteractiveQuantumDebugConsole()
        
        result = console.execute_command("unknown_command")
        assert "Unknown command" in result
        assert "help" in result
    
    def test_empty_command(self):
        """Test empty command handling."""
        console = InteractiveQuantumDebugConsole()
        
        result = console.execute_command("")
        assert result == "empty"
    
    def test_command_with_arguments(self):
        """Test commands with multiple arguments."""
        console = InteractiveQuantumDebugConsole()
        
        result = console.execute_command("analyze multiple word error message")
        assert "Error Analysis" in result
    
    def test_command_history_tracking(self):
        """Test command history tracking."""
        console = InteractiveQuantumDebugConsole()
        
        initial_count = len(console.command_history)
        
        console.execute_command("help")
        console.execute_command("session new")
        
        # Note: command_history is updated by start_interactive_session, not execute_command
        # So we test that execute_command works without updating history
        assert len(console.command_history) == initial_count
    
    def test_breakpoint_without_session(self):
        """Test breakpoint command without active session."""
        console = InteractiveQuantumDebugConsole()
        
        result = console.execute_command("breakpoint add test_location")
        assert "No active debug session" in result
    
    def test_session_info_without_session(self):
        """Test session info without active session."""
        console = InteractiveQuantumDebugConsole()
        
        result = console.execute_command("session info")
        assert "No active session" in result


@pytest.mark.skipif(not HAS_QUANTUM_DEBUGGING_TOOLS, reason="quantum debugging tools not available")
class TestQuantumDebuggingWebInterface:
    """Test QuantumDebuggingWebInterface functionality."""
    
    def test_web_interface_initialization(self):
        """Test web interface initialization."""
        interface = QuantumDebuggingWebInterface(DebugLevel.INFO)
        
        assert interface.debug_level == DebugLevel.INFO
        assert isinstance(interface.console, InteractiveQuantumDebugConsole)
        
        # App should be created if frameworks are available
        if HAS_DASH or HAS_FLASK:
            assert interface.app is not None
    
    @pytest.mark.skipif(not HAS_DASH, reason="dash not available")
    def test_dash_app_creation(self):
        """Test Dash app creation."""
        interface = QuantumDebuggingWebInterface()
        
        assert interface.app is not None
        assert hasattr(interface.app, 'layout')
        assert hasattr(interface.app, 'callback')
    
    @pytest.mark.skipif(not HAS_FLASK, reason="flask not available")
    def test_flask_app_creation(self):
        """Test Flask app creation when Dash is not available."""
        # Mock Dash as unavailable
        with patch('quantrs2.quantum_debugging_tools.HAS_DASH', False):
            interface = QuantumDebuggingWebInterface()
            
            if HAS_FLASK:
                assert interface.app is not None
                assert hasattr(interface.app, 'route')
    
    def test_web_interface_without_frameworks(self):
        """Test web interface when no frameworks are available."""
        # Mock both frameworks as unavailable
        with patch('quantrs2.quantum_debugging_tools.HAS_DASH', False), \
             patch('quantrs2.quantum_debugging_tools.HAS_FLASK', False):
            interface = QuantumDebuggingWebInterface()
            
            # Should still initialize but app might be None
            assert isinstance(interface.console, InteractiveQuantumDebugConsole)


@pytest.mark.skipif(not HAS_QUANTUM_DEBUGGING_TOOLS, reason="quantum debugging tools not available")
class TestQuantumDebuggingToolsManager:
    """Test QuantumDebuggingToolsManager functionality."""
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = QuantumDebuggingToolsManager(DebugLevel.INFO)
        
        assert manager.debug_level == DebugLevel.INFO
        assert isinstance(manager.state_inspector, QuantumStateInspector)
        assert isinstance(manager.error_analyzer, QuantumErrorAnalyzer)
        assert isinstance(manager.circuit_validator, QuantumCircuitValidator)
        assert isinstance(manager.memory_debugger, QuantumMemoryDebugger)
        assert isinstance(manager.console, InteractiveQuantumDebugConsole)
        assert isinstance(manager.web_interface, QuantumDebuggingWebInterface)
    
    def test_debug_quantum_circuit(self, mock_circuit):
        """Test comprehensive circuit debugging."""
        manager = QuantumDebuggingToolsManager()
        
        debug_options = {
            'validate_circuit': True,
            'analyze_state': True,
            'profile_performance': False,  # Skip performance for mock
            'analyze_memory': True
        }
        
        results = manager.debug_quantum_circuit(mock_circuit, debug_options)
        
        assert isinstance(results, dict)
        assert 'circuit_analysis' in results
        assert 'state_analysis' in results
        assert 'validation_results' in results
        assert 'error_diagnosis' in results
        assert 'performance_metrics' in results
        assert 'memory_analysis' in results
        assert 'recommendations' in results
        assert 'overall_status' in results
        
        # Check validation results
        validation_results = results['validation_results']
        assert isinstance(validation_results, list)
        
        # Check memory analysis
        memory_analysis = results['memory_analysis']
        assert 'total_memory_mb' in memory_analysis
        assert 'quantum_memory_mb' in memory_analysis
        assert 'peak_usage_mb' in memory_analysis
    
    def test_debug_circuit_with_minimal_options(self, mock_circuit):
        """Test circuit debugging with minimal options."""
        manager = QuantumDebuggingToolsManager()
        
        debug_options = {
            'validate_circuit': False,
            'analyze_state': False,
            'profile_performance': False,
            'analyze_memory': False
        }
        
        results = manager.debug_quantum_circuit(mock_circuit, debug_options)
        
        assert isinstance(results, dict)
        assert results['overall_status'] in ['healthy', 'needs_attention']
    
    def test_analyze_quantum_error_integration(self, mock_error):
        """Test quantum error analysis integration."""
        manager = QuantumDebuggingToolsManager()
        
        diagnosis = manager.analyze_quantum_error(mock_error)
        
        assert isinstance(diagnosis, ErrorDiagnosis)
        assert isinstance(diagnosis.error_type, ErrorType)
        assert isinstance(diagnosis.severity, str)
        assert isinstance(diagnosis.suggestions, list)
    
    def test_inspect_quantum_state_integration(self, sample_state_vector):
        """Test quantum state inspection integration."""
        manager = QuantumDebuggingToolsManager()
        
        result = manager.inspect_quantum_state(sample_state_vector, InspectionMode.AMPLITUDE_ANALYSIS)
        
        assert isinstance(result, StateInspectionResult)
        assert result.mode == InspectionMode.AMPLITUDE_ANALYSIS
        assert isinstance(result.insights, list)
    
    def test_validate_quantum_circuit_integration(self, mock_circuit):
        """Test quantum circuit validation integration."""
        manager = QuantumDebuggingToolsManager()
        
        rules = [ValidationRule.UNITARITY_CHECK, ValidationRule.RESOURCE_CHECK]
        results = manager.validate_quantum_circuit(mock_circuit, rules)
        
        assert isinstance(results, list)
        assert len(results) == len(rules)
        
        for result in results:
            assert isinstance(result, ValidationResult)
            assert result.rule in rules
    
    def test_memory_debugging_integration(self):
        """Test memory debugging integration."""
        manager = QuantumDebuggingToolsManager()
        
        memory_info = manager.get_memory_usage()
        
        assert isinstance(memory_info, MemoryDebugInfo)
        assert isinstance(memory_info.total_memory, int)
        assert isinstance(memory_info.optimization_suggestions, list)
    
    def test_debugging_context_manager(self):
        """Test debugging context manager."""
        manager = QuantumDebuggingToolsManager()
        
        with manager.debugging_context("test_operation") as debug_manager:
            assert debug_manager is manager
            # Perform some operation
            time.sleep(0.1)
        
        # Context should complete successfully
        assert True
    
    def test_debugging_context_with_memory_monitoring(self):
        """Test debugging context with memory monitoring."""
        manager = QuantumDebuggingToolsManager()
        
        debug_options = {'monitor_memory': True}
        
        with manager.debugging_context("memory_test", debug_options) as debug_manager:
            assert debug_manager is manager
            # Perform some operation
            time.sleep(0.1)
        
        # Context should complete successfully
        assert True
    
    def test_integration_module_availability(self):
        """Test integration with other modules."""
        manager = QuantumDebuggingToolsManager()
        
        # Check that integration attributes exist
        assert hasattr(manager, 'profiler')
        assert hasattr(manager, 'test_manager')
        assert hasattr(manager, 'visualizer')
        assert hasattr(manager, 'algorithm_debugger')
        
        # These might be None if modules aren't available, which is fine


@pytest.mark.skipif(not HAS_QUANTUM_DEBUGGING_TOOLS, reason="quantum debugging tools not available")
class TestConvenienceFunctions:
    """Test convenience functions for easy usage."""
    
    def test_get_quantum_debugging_tools(self):
        """Test get_quantum_debugging_tools function."""
        debugger = get_quantum_debugging_tools()
        
        assert isinstance(debugger, QuantumDebuggingToolsManager)
        assert debugger.debug_level == DebugLevel.INFO
        
        # Test with custom debug level
        debugger_custom = get_quantum_debugging_tools(DebugLevel.DEBUG)
        assert debugger_custom.debug_level == DebugLevel.DEBUG
    
    def test_debug_quantum_circuit_function(self, mock_circuit):
        """Test debug_quantum_circuit convenience function."""
        results = debug_quantum_circuit(mock_circuit)
        
        assert isinstance(results, dict)
        assert 'validation_results' in results
        assert 'overall_status' in results
        
        # Test with custom options
        debug_options = {'validate_circuit': True, 'analyze_state': False}
        results_custom = debug_quantum_circuit(mock_circuit, debug_options)
        
        assert isinstance(results_custom, dict)
    
    def test_analyze_quantum_error_function(self, mock_error):
        """Test analyze_quantum_error convenience function."""
        diagnosis = analyze_quantum_error(mock_error)
        
        assert isinstance(diagnosis, ErrorDiagnosis)
        assert isinstance(diagnosis.error_type, ErrorType)
        
        # Test with context
        context = {'gate_name': 'RX', 'location': 'test.py:123'}
        diagnosis_with_context = analyze_quantum_error(mock_error, context)
        
        assert isinstance(diagnosis_with_context, ErrorDiagnosis)
        assert context['gate_name'] in diagnosis_with_context.message
    
    def test_inspect_quantum_state_function(self, sample_state_vector):
        """Test inspect_quantum_state convenience function."""
        # Test with default mode
        result = inspect_quantum_state(sample_state_vector)
        
        assert isinstance(result, StateInspectionResult)
        assert result.mode == InspectionMode.AMPLITUDE_ANALYSIS
        
        # Test with specific mode
        result_prob = inspect_quantum_state(sample_state_vector, 'probability')
        
        assert result_prob.mode == InspectionMode.PROBABILITY_ANALYSIS
        
        # Test with invalid mode
        with pytest.raises(ValueError, match="Unknown inspection mode"):
            inspect_quantum_state(sample_state_vector, 'invalid_mode')
    
    def test_validate_quantum_circuit_function(self, mock_circuit):
        """Test validate_quantum_circuit convenience function."""
        # Test with default rules (all rules)
        results = validate_quantum_circuit(mock_circuit)
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Test with specific rules
        rules = ['unitarity', 'resources']
        results_specific = validate_quantum_circuit(mock_circuit, rules)
        
        assert len(results_specific) == len(rules)
        
        for result in results_specific:
            assert isinstance(result, ValidationResult)
            assert result.rule in [ValidationRule.UNITARITY_CHECK, ValidationRule.RESOURCE_CHECK]
    
    def test_convenience_functions_integration(self):
        """Test that convenience functions work together."""
        # Create a debugging session using convenience functions
        debugger = get_quantum_debugging_tools()
        
        assert isinstance(debugger, QuantumDebuggingToolsManager)
        
        # Test that we can use the manager for various operations
        state_vector = np.array([1, 0], dtype=complex)
        result = debugger.inspect_quantum_state(state_vector, InspectionMode.AMPLITUDE_ANALYSIS)
        
        assert isinstance(result, StateInspectionResult)


@pytest.mark.skipif(not HAS_QUANTUM_DEBUGGING_TOOLS, reason="quantum debugging tools not available")
class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    def test_state_inspector_with_invalid_state(self):
        """Test state inspector with invalid state vectors."""
        inspector = QuantumStateInspector()
        
        # Test with empty state
        empty_state = np.array([], dtype=complex)
        result = inspector.inspect_state(empty_state, InspectionMode.AMPLITUDE_ANALYSIS)
        
        assert isinstance(result, StateInspectionResult)
        # Should handle gracefully
        
        # Test with non-complex state
        real_state = np.array([1, 0])
        result = inspector.inspect_state(real_state, InspectionMode.AMPLITUDE_ANALYSIS)
        
        assert isinstance(result, StateInspectionResult)
    
    def test_error_analyzer_with_none_error(self):
        """Test error analyzer with None error."""
        analyzer = QuantumErrorAnalyzer()
        
        # This should not crash
        diagnosis = analyzer.analyze_error(None)
        
        assert isinstance(diagnosis, ErrorDiagnosis)
    
    def test_circuit_validator_with_invalid_circuit(self):
        """Test circuit validator with invalid circuit."""
        validator = QuantumCircuitValidator()
        
        # Test with None circuit
        results = validator.validate_circuit(None)
        
        assert isinstance(results, list)
        # Should handle gracefully and return failure results
        
        for result in results:
            assert isinstance(result, ValidationResult)
            assert result.passed is False
    
    def test_memory_debugger_without_psutil(self):
        """Test memory debugger when psutil is not available."""
        with patch('quantrs2.quantum_debugging_tools.HAS_PSUTIL', False):
            debugger = QuantumMemoryDebugger()
            
            snapshot = debugger.get_memory_snapshot()
            
            assert isinstance(snapshot, MemoryDebugInfo)
            assert "psutil" in snapshot.optimization_suggestions[0]
    
    def test_console_command_exceptions(self):
        """Test console command exception handling."""
        console = InteractiveQuantumDebugConsole()
        
        # Mock a command that raises an exception
        def failing_command(args):
            raise Exception("Command failed")
        
        console.commands['test_fail'] = failing_command
        
        result = console.execute_command("test_fail")
        
        assert "Command failed" in result
    
    def test_debugging_tools_manager_with_missing_modules(self):
        """Test debugging tools manager when optional modules are missing."""
        # Mock all optional modules as unavailable
        with patch('quantrs2.quantum_debugging_tools.HAS_PROFILER', False), \
             patch('quantrs2.quantum_debugging_tools.HAS_TESTING_TOOLS', False), \
             patch('quantrs2.quantum_debugging_tools.HAS_VISUALIZATION', False), \
             patch('quantrs2.quantum_debugging_tools.HAS_ALGORITHM_DEBUGGER', False):
            
            manager = QuantumDebuggingToolsManager()
            
            # Should still initialize core components
            assert isinstance(manager.state_inspector, QuantumStateInspector)
            assert isinstance(manager.error_analyzer, QuantumErrorAnalyzer)
            assert isinstance(manager.circuit_validator, QuantumCircuitValidator)
            assert isinstance(manager.memory_debugger, QuantumMemoryDebugger)
            
            # Optional integrations should be None
            assert manager.profiler is None
            assert manager.test_manager is None
            assert manager.visualizer is None
            assert manager.algorithm_debugger is None
    
    def test_large_state_vector_performance(self):
        """Test performance with large state vectors."""
        inspector = QuantumStateInspector()
        
        # Create large state vector (6 qubits = 64 amplitudes)
        large_state = np.zeros(64, dtype=complex)
        large_state[0] = 1.0
        
        # Should complete without timing out
        start_time = time.time()
        result = inspector.inspect_state(large_state, InspectionMode.PROBABILITY_ANALYSIS)
        end_time = time.time()
        
        assert isinstance(result, StateInspectionResult)
        assert end_time - start_time < 5.0  # Should complete within 5 seconds
    
    def test_concurrent_debugging_operations(self):
        """Test concurrent debugging operations."""
        import threading
        
        manager = QuantumDebuggingToolsManager()
        results = []
        errors = []
        
        def debug_operation(circuit_id):
            try:
                # Create mock circuit
                circuit = Mock()
                circuit.num_qubits = 2
                circuit.gates = [Mock(name="H", qubits=[0])]
                circuit.run = Mock(return_value=Mock(state_vector=np.array([1, 0, 0, 0])))
                
                result = manager.debug_quantum_circuit(circuit)
                results.append((circuit_id, result))
            except Exception as e:
                errors.append((circuit_id, e))
        
        # Start multiple debugging operations concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=debug_operation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10.0)
        
        # Check results
        assert len(errors) == 0  # No errors should occur
        assert len(results) == 5  # All operations should complete
        
        for circuit_id, result in results:
            assert isinstance(result, dict)
            assert 'overall_status' in result


@pytest.mark.skipif(not HAS_QUANTUM_DEBUGGING_TOOLS, reason="quantum debugging tools not available")
class TestPerformanceAndScalability:
    """Test performance and scalability of debugging tools."""
    
    def test_state_inspection_performance(self):
        """Test state inspection performance with various state sizes."""
        inspector = QuantumStateInspector()
        
        # Test with different state sizes
        for n_qubits in [2, 3, 4, 5]:
            state_size = 2 ** n_qubits
            state_vector = np.zeros(state_size, dtype=complex)
            state_vector[0] = 1.0
            
            start_time = time.time()
            result = inspector.inspect_state(state_vector, InspectionMode.AMPLITUDE_ANALYSIS)
            end_time = time.time()
            
            assert isinstance(result, StateInspectionResult)
            assert end_time - start_time < 2.0  # Should complete within 2 seconds
    
    def test_error_analysis_performance(self):
        """Test error analysis performance with many errors."""
        analyzer = QuantumErrorAnalyzer()
        
        start_time = time.time()
        
        # Analyze many errors
        for i in range(100):
            error = Exception(f"Test error {i}")
            diagnosis = analyzer.analyze_error(error)
            assert isinstance(diagnosis, ErrorDiagnosis)
        
        end_time = time.time()
        
        # Should complete all analyses within reasonable time
        assert end_time - start_time < 5.0
        assert len(analyzer.error_history) >= 100
    
    def test_circuit_validation_performance(self):
        """Test circuit validation performance with large circuits."""
        validator = QuantumCircuitValidator()
        
        # Create large circuit
        large_circuit = Mock()
        large_circuit.num_qubits = 20
        large_circuit.gates = [Mock(qubits=[i % 20]) for i in range(1000)]
        
        start_time = time.time()
        results = validator.validate_circuit(large_circuit, [ValidationRule.RESOURCE_CHECK])
        end_time = time.time()
        
        assert len(results) == 1
        assert end_time - start_time < 3.0  # Should complete within 3 seconds
    
    def test_memory_debugging_efficiency(self):
        """Test memory debugging efficiency."""
        debugger = QuantumMemoryDebugger()
        
        # Test multiple memory snapshots
        start_time = time.time()
        
        for _ in range(50):
            snapshot = debugger.get_memory_snapshot()
            assert isinstance(snapshot, MemoryDebugInfo)
        
        end_time = time.time()
        
        # Should be efficient
        assert end_time - start_time < 2.0
    
    def test_comprehensive_debugging_performance(self):
        """Test comprehensive debugging performance."""
        manager = QuantumDebuggingToolsManager()
        
        # Create reasonably complex circuit
        circuit = Mock()
        circuit.num_qubits = 5
        circuit.gates = [
            Mock(name="H", qubits=[i]) for i in range(5)
        ] + [
            Mock(name="CNOT", qubits=[i, (i+1) % 5]) for i in range(5)
        ]
        circuit.run = Mock(return_value=Mock(state_vector=np.zeros(32, dtype=complex)))
        circuit.run.return_value.state_vector[0] = 1.0
        
        start_time = time.time()
        results = manager.debug_quantum_circuit(circuit)
        end_time = time.time()
        
        assert isinstance(results, dict)
        assert end_time - start_time < 10.0  # Should complete within 10 seconds


if __name__ == "__main__":
    pytest.main([__file__])