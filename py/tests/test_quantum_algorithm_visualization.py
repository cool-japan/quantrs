#!/usr/bin/env python3
"""
Comprehensive test suite for the QuantRS2 Quantum Algorithm Visualization System.

This test suite provides complete coverage of all visualization functionality including:
- VisualizationConfig dataclass and configuration management
- CircuitVisualizationData and StateVisualizationData structures
- CircuitVisualizer with circuit diagram rendering and performance integration
- StateVisualizer with 3D Bloch spheres and state evolution animations
- PerformanceVisualizer with analytics charts and dashboard creation
- QuantumAlgorithmVisualizer main orchestrator with comprehensive workflows
- GUI and web interfaces (if dependencies available)
- Export capabilities and format conversions
- Integration with performance profiling system
"""

import pytest
import tempfile
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, List, Any

try:
    import quantrs2
    from quantrs2.quantum_algorithm_visualization import (
        VisualizationConfig, CircuitVisualizationData, StateVisualizationData,
        CircuitVisualizer, StateVisualizer, PerformanceVisualizer,
        QuantumAlgorithmVisualizer, visualize_quantum_circuit,
        visualize_quantum_state, create_bloch_sphere_visualization,
        compare_quantum_algorithms, HAS_TKINTER, HAS_DASH, HAS_JUPYTER, HAS_PROFILER
    )
    HAS_QUANTRS2 = True
except ImportError:
    HAS_QUANTRS2 = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Test fixtures and mock objects
@pytest.fixture
def sample_config():
    """Create sample VisualizationConfig for testing."""
    return VisualizationConfig(
        figure_size=(10, 8),
        dpi=150,
        style="seaborn-v0_8",
        color_scheme="quantum",
        animation_speed=2.0,
        gate_spacing=1.5,
        qubit_spacing=1.2
    )

@pytest.fixture
def sample_circuit_data():
    """Create sample CircuitVisualizationData for testing."""
    circuit_data = CircuitVisualizationData()
    circuit_data.qubits = [0, 1, 2]
    circuit_data.add_gate("H", [0], execution_time=0.001, fidelity=0.99)
    circuit_data.add_gate("CNOT", [0, 1], execution_time=0.005, fidelity=0.98)
    circuit_data.add_gate("RZ", [2], [np.pi/4], execution_time=0.002, fidelity=0.995)
    circuit_data.add_gate("MEASURE", [0])
    return circuit_data

@pytest.fixture
def sample_state_data():
    """Create sample StateVisualizationData for testing."""
    # Bell state
    state_vector = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
    state_data = StateVisualizationData(state_vector=state_vector)
    
    # Add time evolution data
    for i in range(5):
        t = i * 0.1
        evolved_state = state_vector * np.exp(1j * t)
        state_data.time_evolution.append((t, evolved_state))
    
    return state_data

@pytest.fixture
def mock_performance_metrics():
    """Create mock performance metrics for testing."""
    metrics = []
    for i in range(10):
        metric = Mock()
        metric.execution_time = 0.1 + i * 0.01
        metric.memory_usage = 100 + i * 10
        metric.gate_count = 5 + i
        metric.circuit_depth = 3 + i // 2
        metric.error_rate = 0.01 + i * 0.001
        metric.fidelity = 0.99 - i * 0.001
        metric.cpu_usage = 50 + i * 2
        metric.timestamp = i * 1.0
        metrics.append(metric)
    return metrics

@pytest.fixture
def mock_circuit():
    """Create mock quantum circuit for testing."""
    circuit = Mock()
    circuit.num_qubits = 2
    circuit.gates = [
        Mock(name="H", qubits=[0], params=[]),
        Mock(name="CNOT", qubits=[0, 1], params=[])
    ]
    circuit.run = Mock(return_value=Mock(state_vector=np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])))
    return circuit


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestVisualizationConfig:
    """Test VisualizationConfig dataclass functionality."""
    
    def test_default_initialization(self):
        """Test default VisualizationConfig initialization."""
        config = VisualizationConfig()
        assert config.figure_size == (12, 8)
        assert config.dpi == 100
        assert config.style == "seaborn-v0_8"
        assert config.color_scheme == "quantum"
        assert config.animation_speed == 1.0
        assert config.gate_spacing == 1.0
        assert config.qubit_spacing == 1.0
        assert config.enable_profiling_overlay is True
        assert config.export_formats == ["png", "pdf", "svg", "html"]
    
    def test_custom_initialization(self, sample_config):
        """Test VisualizationConfig with custom values."""
        assert sample_config.figure_size == (10, 8)
        assert sample_config.dpi == 150
        assert sample_config.animation_speed == 2.0
        assert sample_config.gate_spacing == 1.5
        assert sample_config.qubit_spacing == 1.2
    
    def test_color_palette_quantum(self):
        """Test quantum color palette."""
        config = VisualizationConfig(color_scheme="quantum")
        palette = config.get_color_palette()
        
        assert isinstance(palette, dict)
        assert "qubit" in palette
        assert "gate" in palette
        assert "measurement" in palette
        assert "classical" in palette
        assert "entanglement" in palette
        assert "background" in palette
        assert "text" in palette
        assert "grid" in palette
        assert palette["qubit"] == "#2E86AB"
        assert palette["gate"] == "#A23B72"
    
    def test_color_palette_classical(self):
        """Test classical color palette."""
        config = VisualizationConfig(color_scheme="classical")
        palette = config.get_color_palette()
        
        assert palette["qubit"] == "#34495E"
        assert palette["gate"] == "#7F8C8D"
    
    def test_color_palette_colorblind(self):
        """Test colorblind-friendly palette."""
        config = VisualizationConfig(color_scheme="colorblind")
        palette = config.get_color_palette()
        
        assert palette["qubit"] == "#0173B2"
        assert palette["gate"] == "#DE8F05"
    
    def test_invalid_color_scheme(self):
        """Test handling of invalid color scheme."""
        config = VisualizationConfig(color_scheme="invalid")
        palette = config.get_color_palette()
        
        # Should fall back to quantum palette
        assert palette["qubit"] == "#2E86AB"


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestCircuitVisualizationData:
    """Test CircuitVisualizationData dataclass functionality."""
    
    def test_default_initialization(self):
        """Test default CircuitVisualizationData initialization."""
        data = CircuitVisualizationData()
        assert data.gates == []
        assert data.qubits == []
        assert data.classical_bits == []
        assert data.measurements == []
        assert data.barriers == []
        assert data.gate_execution_times == {}
        assert data.gate_fidelities == {}
        assert data.gate_error_rates == {}
        assert data.circuit_depth == 0
        assert data.total_gates == 0
        assert data.entangling_gates == 0
        assert data.single_qubit_gates == 0
    
    def test_add_single_qubit_gate(self):
        """Test adding single-qubit gates."""
        data = CircuitVisualizationData()
        
        data.add_gate("H", [0], execution_time=0.001, fidelity=0.99)
        
        assert len(data.gates) == 1
        assert data.gates[0]["type"] == "H"
        assert data.gates[0]["qubits"] == [0]
        assert data.gates[0]["id"] == 0
        assert data.total_gates == 1
        assert data.single_qubit_gates == 1
        assert data.entangling_gates == 0
        assert 0 in data.qubits
        assert data.gate_execution_times[0] == 0.001
        assert data.gate_fidelities[0] == 0.99
    
    def test_add_two_qubit_gate(self):
        """Test adding two-qubit gates."""
        data = CircuitVisualizationData()
        
        data.add_gate("CNOT", [0, 1], execution_time=0.005, fidelity=0.98)
        
        assert len(data.gates) == 1
        assert data.gates[0]["type"] == "CNOT"
        assert data.gates[0]["qubits"] == [0, 1]
        assert data.total_gates == 1
        assert data.single_qubit_gates == 0
        assert data.entangling_gates == 1
        assert set(data.qubits) == {0, 1}
    
    def test_add_parametric_gate(self):
        """Test adding parametric gates."""
        data = CircuitVisualizationData()
        
        data.add_gate("RZ", [0], [np.pi/4])
        
        assert data.gates[0]["params"] == [np.pi/4]
    
    def test_multiple_gates(self, sample_circuit_data):
        """Test adding multiple gates."""
        assert len(sample_circuit_data.gates) == 4
        assert sample_circuit_data.total_gates == 4
        assert sample_circuit_data.single_qubit_gates == 3  # H, RZ, MEASURE
        assert sample_circuit_data.entangling_gates == 1   # CNOT
        assert set(sample_circuit_data.qubits) == {0, 1, 2}
    
    def test_qubit_ordering(self):
        """Test that qubits are kept in sorted order."""
        data = CircuitVisualizationData()
        
        data.add_gate("H", [2])
        data.add_gate("X", [0])
        data.add_gate("Y", [1])
        
        assert data.qubits == [0, 1, 2]


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestStateVisualizationData:
    """Test StateVisualizationData dataclass functionality."""
    
    def test_default_initialization(self):
        """Test default StateVisualizationData initialization."""
        data = StateVisualizationData()
        assert len(data.state_vector) == 0
        assert data.density_matrix is None
        assert data.measurement_probabilities == {}
        assert data.time_evolution == []
        assert data.measurement_history == []
        assert data.bloch_coordinates == {}
        assert data.entanglement_entropy == 0.0
        assert data.schmidt_coefficients == []
        assert data.purity == 1.0
    
    def test_initialization_with_state_vector(self):
        """Test initialization with state vector."""
        state_vector = np.array([1, 0], dtype=complex)
        data = StateVisualizationData(state_vector=state_vector)
        
        assert np.array_equal(data.state_vector, state_vector)
    
    def test_bloch_coordinates_single_qubit(self):
        """Test Bloch coordinate calculation for single qubit."""
        # |0⟩ state
        state_vector = np.array([1, 0], dtype=complex)
        data = StateVisualizationData(state_vector=state_vector)
        
        x, y, z = data.calculate_bloch_coordinates(0)
        assert abs(x) < 1e-10
        assert abs(y) < 1e-10
        assert abs(z - 1.0) < 1e-10
        
        # |1⟩ state
        state_vector = np.array([0, 1], dtype=complex)
        data = StateVisualizationData(state_vector=state_vector)
        
        x, y, z = data.calculate_bloch_coordinates(0)
        assert abs(x) < 1e-10
        assert abs(y) < 1e-10
        assert abs(z - (-1.0)) < 1e-10
        
        # |+⟩ state
        state_vector = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        data = StateVisualizationData(state_vector=state_vector)
        
        x, y, z = data.calculate_bloch_coordinates(0)
        assert abs(x - 1.0) < 1e-10
        assert abs(y) < 1e-10
        assert abs(z) < 1e-10
    
    def test_bloch_coordinates_two_qubit(self, sample_state_data):
        """Test Bloch coordinate calculation for two-qubit state."""
        # Bell state: should give bloch coordinates for reduced states
        x0, y0, z0 = sample_state_data.calculate_bloch_coordinates(0)
        x1, y1, z1 = sample_state_data.calculate_bloch_coordinates(1)
        
        # For maximally entangled state, reduced states should be maximally mixed
        assert abs(x0) < 1e-10
        assert abs(y0) < 1e-10
        assert abs(z0) < 1e-10
        assert abs(x1) < 1e-10
        assert abs(y1) < 1e-10
        assert abs(z1) < 1e-10
    
    def test_bloch_coordinates_invalid_qubit(self):
        """Test Bloch coordinates for invalid qubit index."""
        state_vector = np.array([1, 0], dtype=complex)
        data = StateVisualizationData(state_vector=state_vector)
        
        # Request coordinates for non-existent qubit
        x, y, z = data.calculate_bloch_coordinates(5)
        assert (x, y, z) == (0.0, 0.0, 1.0)  # Default |0⟩ state
    
    def test_bloch_coordinates_empty_state(self):
        """Test Bloch coordinates with empty state vector."""
        data = StateVisualizationData()
        
        x, y, z = data.calculate_bloch_coordinates(0)
        assert (x, y, z) == (0.0, 0.0, 1.0)  # Default |0⟩ state
    
    def test_reduced_density_matrix(self, sample_state_data):
        """Test reduced density matrix calculation."""
        rho = sample_state_data.get_reduced_density_matrix(0)
        
        assert rho.shape == (2, 2)
        assert np.allclose(rho, rho.conj().T)  # Hermitian
        assert abs(np.trace(rho) - 1.0) < 1e-10  # Trace 1
        
        # For Bell state, reduced state should be maximally mixed
        expected_rho = 0.5 * np.eye(2)
        assert np.allclose(rho, expected_rho, atol=1e-10)
    
    def test_time_evolution_data(self, sample_state_data):
        """Test time evolution data handling."""
        assert len(sample_state_data.time_evolution) == 5
        
        for i, (time, state) in enumerate(sample_state_data.time_evolution):
            assert time == i * 0.1
            assert len(state) == 4  # Two-qubit state
            assert abs(np.linalg.norm(state) - 1.0) < 1e-10  # Normalized


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestCircuitVisualizer:
    """Test CircuitVisualizer functionality."""
    
    def test_circuit_visualizer_initialization(self, sample_config):
        """Test CircuitVisualizer initialization."""
        visualizer = CircuitVisualizer(sample_config)
        
        assert visualizer.config == sample_config
        assert isinstance(visualizer.colors, dict)
        assert "qubit" in visualizer.colors
        assert len(visualizer.gate_renderers) > 0
        assert "H" in visualizer.gate_renderers
        assert "CNOT" in visualizer.gate_renderers
    
    def test_default_config_initialization(self):
        """Test CircuitVisualizer with default config."""
        visualizer = CircuitVisualizer()
        
        assert isinstance(visualizer.config, VisualizationConfig)
        assert visualizer.config.color_scheme == "quantum"
    
    def test_visualize_circuit_basic(self, sample_circuit_data):
        """Test basic circuit visualization."""
        visualizer = CircuitVisualizer()
        
        fig = visualizer.visualize_circuit(sample_circuit_data, title="Test Circuit")
        
        assert isinstance(fig, plt.Figure)
        axes = fig.get_axes()
        assert len(axes) >= 1
        
        ax = axes[0]
        assert ax.get_title() == "Test Circuit"
        assert ax.get_xlabel() == "Circuit Depth"
        assert ax.get_ylabel() == "Qubits"
        
        plt.close(fig)
    
    def test_circuit_layout_calculation(self, sample_circuit_data):
        """Test circuit layout calculation."""
        visualizer = CircuitVisualizer()
        layout = visualizer._calculate_circuit_layout(sample_circuit_data)
        
        assert "n_qubits" in layout
        assert "n_layers" in layout
        assert "qubit_positions" in layout
        assert "layer_positions" in layout
        assert "gate_layers" in layout
        assert "circuit_width" in layout
        assert "circuit_height" in layout
        
        assert layout["n_qubits"] == 3
        assert layout["n_layers"] >= 1
        assert len(layout["qubit_positions"]) == 3
        assert len(layout["gate_layers"]) == len(sample_circuit_data.gates)
    
    def test_gate_layer_calculation(self, sample_circuit_data):
        """Test gate layer calculation for optimal placement."""
        visualizer = CircuitVisualizer()
        gate_layers = visualizer._calculate_gate_layers(sample_circuit_data)
        
        assert isinstance(gate_layers, dict)
        assert len(gate_layers) == len(sample_circuit_data.gates)
        
        # All gates should have valid layer assignments
        for gate_id, layer in gate_layers.items():
            assert isinstance(layer, int)
            assert layer >= 0
    
    def test_gate_rendering_methods(self):
        """Test that all gate rendering methods exist."""
        visualizer = CircuitVisualizer()
        
        required_gates = ["H", "X", "Y", "Z", "CNOT", "CX", "CZ", 
                         "RX", "RY", "RZ", "SWAP", "MEASURE", "BARRIER"]
        
        for gate in required_gates:
            assert gate in visualizer.gate_renderers
            assert callable(visualizer.gate_renderers[gate])
    
    def test_performance_overlay(self, sample_circuit_data):
        """Test performance data overlay."""
        visualizer = CircuitVisualizer()
        
        # Test with performance data
        fig = visualizer.visualize_circuit(sample_circuit_data, 
                                         show_performance=True)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_animated_execution(self, sample_circuit_data):
        """Test animated circuit execution visualization."""
        visualizer = CircuitVisualizer()
        
        # Create mock execution trace
        execution_trace = [{"step": i, "gate_id": i} for i in range(len(sample_circuit_data.gates))]
        
        anim = visualizer.create_animated_execution(sample_circuit_data, execution_trace)
        
        # Check that animation was created
        assert hasattr(anim, 'fig')
        assert hasattr(anim, 'event_source')
        
        plt.close(anim.fig)
    
    def test_empty_circuit_data(self):
        """Test visualization with empty circuit data."""
        visualizer = CircuitVisualizer()
        empty_data = CircuitVisualizationData()
        
        fig = visualizer.visualize_circuit(empty_data)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_single_gate_circuit(self):
        """Test visualization with single gate."""
        visualizer = CircuitVisualizer()
        circuit_data = CircuitVisualizationData()
        circuit_data.add_gate("H", [0])
        
        fig = visualizer.visualize_circuit(circuit_data)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_large_circuit(self):
        """Test visualization with many gates."""
        visualizer = CircuitVisualizer()
        circuit_data = CircuitVisualizationData()
        
        # Create circuit with many gates
        for i in range(10):
            circuit_data.add_gate("H", [i % 3])
            if i > 0:
                circuit_data.add_gate("CNOT", [i % 3, (i + 1) % 3])
        
        fig = visualizer.visualize_circuit(circuit_data)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestStateVisualizer:
    """Test StateVisualizer functionality."""
    
    def test_state_visualizer_initialization(self, sample_config):
        """Test StateVisualizer initialization."""
        visualizer = StateVisualizer(sample_config)
        
        assert visualizer.config == sample_config
        assert isinstance(visualizer.colors, dict)
    
    def test_visualize_amplitudes(self, sample_state_data):
        """Test state vector amplitude visualization."""
        visualizer = StateVisualizer()
        
        fig = visualizer.visualize_state_vector(sample_state_data, 
                                              title="Test State",
                                              visualization_type="amplitudes")
        
        assert isinstance(fig, plt.Figure)
        assert "Real Parts" in fig.axes[0].get_title()
        assert "Imaginary Parts" in fig.axes[1].get_title()
        
        plt.close(fig)
    
    def test_visualize_probabilities(self, sample_state_data):
        """Test state probability visualization."""
        visualizer = StateVisualizer()
        
        fig = visualizer.visualize_state_vector(sample_state_data,
                                              visualization_type="probabilities")
        
        assert isinstance(fig, plt.Figure)
        assert "Measurement Probabilities" in fig.axes[0].get_title()
        
        plt.close(fig)
    
    def test_visualize_phases(self, sample_state_data):
        """Test state phase visualization."""
        visualizer = StateVisualizer()
        
        fig = visualizer.visualize_state_vector(sample_state_data,
                                              visualization_type="phase")
        
        assert isinstance(fig, plt.Figure)
        # Check that polar subplots were created
        assert len(fig.axes) >= 2
        
        plt.close(fig)
    
    def test_visualize_bloch_spheres(self, sample_state_data):
        """Test multi-qubit Bloch sphere visualization."""
        visualizer = StateVisualizer()
        
        fig = visualizer.visualize_state_vector(sample_state_data,
                                              visualization_type="bloch")
        
        assert isinstance(fig, plt.Figure)
        # Should have 2 subplots for 2-qubit state
        assert len(fig.axes) == 2
        
        plt.close(fig)
    
    def test_visualize_density_matrix(self, sample_state_data):
        """Test density matrix visualization."""
        visualizer = StateVisualizer()
        
        fig = visualizer.visualize_state_vector(sample_state_data,
                                              visualization_type="density_matrix")
        
        assert isinstance(fig, plt.Figure)
        # Should have 4 subplots (real, imaginary, absolute, eigenvalues)
        assert len(fig.axes) == 4
        
        plt.close(fig)
    
    def test_single_bloch_sphere(self, sample_state_data):
        """Test single Bloch sphere creation."""
        visualizer = StateVisualizer()
        
        fig = visualizer.create_bloch_sphere(0, sample_state_data, "Test Bloch")
        
        assert isinstance(fig, plt.Figure)
        assert "Test Bloch" in fig.axes[0].get_title()
        
        plt.close(fig)
    
    def test_state_evolution_animation(self, sample_state_data):
        """Test state evolution animation."""
        visualizer = StateVisualizer()
        
        anim = visualizer.create_state_evolution_animation(sample_state_data, 
                                                         qubit_index=0)
        
        assert hasattr(anim, 'fig')
        assert hasattr(anim, 'event_source')
        
        plt.close(anim.fig)
    
    def test_empty_state_vector(self):
        """Test visualization with empty state vector."""
        visualizer = StateVisualizer()
        empty_state_data = StateVisualizationData()
        
        with pytest.raises(ValueError, match="State vector is empty"):
            visualizer.visualize_state_vector(empty_state_data)
    
    def test_invalid_visualization_type(self, sample_state_data):
        """Test with invalid visualization type."""
        visualizer = StateVisualizer()
        
        with pytest.raises(ValueError, match="Unknown visualization type"):
            visualizer.visualize_state_vector(sample_state_data, 
                                            visualization_type="invalid")
    
    def test_single_qubit_state(self):
        """Test visualization with single-qubit state."""
        visualizer = StateVisualizer()
        state_vector = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        state_data = StateVisualizationData(state_vector=state_vector)
        
        fig = visualizer.visualize_state_vector(state_data, 
                                              visualization_type="amplitudes")
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_large_state_vector(self):
        """Test visualization with large state vector."""
        visualizer = StateVisualizer()
        # 3-qubit state (8 amplitudes)
        state_vector = np.zeros(8, dtype=complex)
        state_vector[0] = 1.0  # |000⟩ state
        state_data = StateVisualizationData(state_vector=state_vector)
        
        fig = visualizer.visualize_state_vector(state_data,
                                              visualization_type="probabilities")
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_bloch_sphere_drawing_components(self):
        """Test Bloch sphere drawing helper methods."""
        visualizer = StateVisualizer()
        
        # Create 3D subplot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Test sphere surface drawing
        visualizer._draw_bloch_sphere_surface(ax)
        
        # Test axes drawing
        visualizer._draw_bloch_sphere_axes(ax)
        
        # Check that elements were added to the plot
        assert len(ax.collections) > 0  # Surface was added
        
        plt.close(fig)


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestPerformanceVisualizer:
    """Test PerformanceVisualizer functionality."""
    
    def test_performance_visualizer_initialization(self, sample_config):
        """Test PerformanceVisualizer initialization."""
        visualizer = PerformanceVisualizer(sample_config)
        
        assert visualizer.config == sample_config
        assert isinstance(visualizer.colors, dict)
    
    def test_visualize_performance_metrics(self, mock_performance_metrics):
        """Test performance metrics visualization."""
        visualizer = PerformanceVisualizer()
        
        fig = visualizer.visualize_performance_metrics(mock_performance_metrics,
                                                     title="Test Performance")
        
        assert isinstance(fig, plt.Figure)
        assert "Test Performance" in fig._suptitle.get_text()
        
        # Should have 2x3 subplots
        assert len(fig.axes) == 6
        
        plt.close(fig)
    
    def test_empty_performance_data(self):
        """Test visualization with empty performance data."""
        visualizer = PerformanceVisualizer()
        
        with pytest.raises(ValueError, match="No performance metrics provided"):
            visualizer.visualize_performance_metrics([])
    
    def test_single_metric_data(self):
        """Test visualization with single performance metric."""
        visualizer = PerformanceVisualizer()
        
        single_metric = Mock()
        single_metric.execution_time = 0.1
        single_metric.memory_usage = 100
        single_metric.gate_count = 5
        single_metric.circuit_depth = 3
        single_metric.error_rate = 0.01
        single_metric.fidelity = 0.99
        
        fig = visualizer.visualize_performance_metrics([single_metric])
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_performance_dashboard(self, mock_performance_metrics):
        """Test performance dashboard creation."""
        visualizer = PerformanceVisualizer()
        
        fig = visualizer.create_performance_dashboard(mock_performance_metrics)
        
        assert isinstance(fig, plt.Figure)
        assert "Performance Dashboard" in fig.axes[0].get_title()
        
        plt.close(fig)
    
    def test_performance_correlation_analysis(self, mock_performance_metrics):
        """Test that correlation analysis is included in visualization."""
        visualizer = PerformanceVisualizer()
        
        fig = visualizer.visualize_performance_metrics(mock_performance_metrics)
        
        # Check that correlation heatmap subplot exists
        correlation_ax = None
        for ax in fig.axes:
            if "Correlation" in ax.get_title():
                correlation_ax = ax
                break
        
        assert correlation_ax is not None
        plt.close(fig)


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")  
class TestQuantumAlgorithmVisualizer:
    """Test QuantumAlgorithmVisualizer main orchestrator."""
    
    def test_main_visualizer_initialization(self, sample_config):
        """Test QuantumAlgorithmVisualizer initialization."""
        visualizer = QuantumAlgorithmVisualizer(sample_config)
        
        assert visualizer.config == sample_config
        assert isinstance(visualizer.circuit_visualizer, CircuitVisualizer)
        assert isinstance(visualizer.state_visualizer, StateVisualizer)
        assert isinstance(visualizer.performance_visualizer, PerformanceVisualizer)
        assert visualizer.circuit_data is None
        assert visualizer.state_data is None
        assert visualizer.performance_data == []
    
    def test_default_initialization(self):
        """Test default initialization without config."""
        visualizer = QuantumAlgorithmVisualizer()
        
        assert isinstance(visualizer.config, VisualizationConfig)
        assert visualizer.config.color_scheme == "quantum"
    
    def test_extract_circuit_data(self, mock_circuit):
        """Test circuit data extraction from mock circuit."""
        visualizer = QuantumAlgorithmVisualizer()
        
        circuit_data = visualizer._extract_circuit_data(mock_circuit)
        
        assert isinstance(circuit_data, CircuitVisualizationData)
        assert len(circuit_data.gates) >= 2  # H and CNOT
        assert 0 in circuit_data.qubits
        assert 1 in circuit_data.qubits
    
    def test_extract_state_data(self, mock_circuit):
        """Test state data extraction from mock circuit."""
        visualizer = QuantumAlgorithmVisualizer()
        
        state_data = visualizer._extract_state_data(mock_circuit)
        
        assert isinstance(state_data, StateVisualizationData)
        assert len(state_data.state_vector) == 4  # 2-qubit state
    
    def test_visualize_algorithm_execution(self, mock_circuit):
        """Test comprehensive algorithm execution visualization."""
        visualizer = QuantumAlgorithmVisualizer()
        
        figures = visualizer.visualize_algorithm_execution(
            mock_circuit,
            include_state_evolution=True,
            include_performance=False,  # No performance data available
            title="Test Algorithm"
        )
        
        assert isinstance(figures, dict)
        assert "circuit" in figures
        assert isinstance(figures["circuit"], plt.Figure)
        
        if "state_amplitudes" in figures:
            assert isinstance(figures["state_amplitudes"], plt.Figure)
        if "state_probabilities" in figures:
            assert isinstance(figures["state_probabilities"], plt.Figure)
        
        # Clean up figures
        for fig in figures.values():
            plt.close(fig)
    
    def test_comparative_visualization(self):
        """Test comparative visualization of multiple algorithms."""
        visualizer = QuantumAlgorithmVisualizer()
        
        # Create mock algorithms
        algorithms = []
        for i in range(3):
            algorithm = Mock()
            algorithm.num_qubits = 2 + i
            algorithm.gates = [Mock(name="H"), Mock(name="CNOT")] * (i + 1)
            algorithms.append(algorithm)
        
        algorithm_names = ["Algorithm 1", "Algorithm 2", "Algorithm 3"]
        
        fig = visualizer.create_comparative_visualization(algorithms, algorithm_names)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4  # 2x2 subplots
        
        plt.close(fig)
    
    def test_export_visualization(self, mock_circuit):
        """Test visualization export functionality."""
        visualizer = QuantumAlgorithmVisualizer()
        
        # Create a simple visualization
        circuit_data = visualizer._extract_circuit_data(mock_circuit)
        fig = visualizer.circuit_visualizer.visualize_circuit(circuit_data)
        
        # Test PNG export
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
        
        try:
            visualizer.export_visualization(fig, temp_path, format="png")
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            plt.close(fig)
    
    def test_export_html_format(self, mock_circuit):
        """Test HTML export format."""
        visualizer = QuantumAlgorithmVisualizer()
        
        circuit_data = visualizer._extract_circuit_data(mock_circuit)
        fig = visualizer.circuit_visualizer.visualize_circuit(circuit_data)
        
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            temp_path = f.name
        
        try:
            visualizer.export_visualization(fig, temp_path, format="html")
            assert os.path.exists(temp_path)
            
            with open(temp_path, 'r') as f:
                content = f.read()
                assert "<!DOCTYPE html>" in content
                assert "QuantRS2 Visualization System" in content
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            plt.close(fig)
    
    def test_unsupported_export_format(self, mock_circuit):
        """Test error handling for unsupported export format."""
        visualizer = QuantumAlgorithmVisualizer()
        
        circuit_data = visualizer._extract_circuit_data(mock_circuit)
        fig = visualizer.circuit_visualizer.visualize_circuit(circuit_data)
        
        # Override supported formats to test error
        visualizer.config.export_formats = ["png"]
        
        with pytest.raises(ValueError, match="Unsupported export format"):
            visualizer.export_visualization(fig, "test.gif", format="gif")
        
        plt.close(fig)
    
    def test_profiler_integration(self, mock_performance_metrics):
        """Test integration with performance profiler."""
        visualizer = QuantumAlgorithmVisualizer()
        
        if HAS_PROFILER:
            # Mock profiler
            mock_profiler = Mock()
            mock_profiler.all_metrics = mock_performance_metrics
            
            visualizer.integrate_with_profiler(mock_profiler)
            
            assert visualizer.profiler == mock_profiler
            assert visualizer.performance_data == mock_performance_metrics
    
    def test_circuit_data_with_no_gates(self):
        """Test circuit data extraction with circuit that has no gates."""
        visualizer = QuantumAlgorithmVisualizer()
        
        empty_circuit = Mock()
        empty_circuit.num_qubits = 2
        del empty_circuit.gates  # No gates attribute
        
        circuit_data = visualizer._extract_circuit_data(empty_circuit)
        
        assert isinstance(circuit_data, CircuitVisualizationData)
        assert len(circuit_data.qubits) >= 0
    
    def test_state_data_extraction_failure(self):
        """Test state data extraction when circuit run fails."""
        visualizer = QuantumAlgorithmVisualizer()
        
        failing_circuit = Mock()
        failing_circuit.run.side_effect = Exception("Circuit execution failed")
        
        state_data = visualizer._extract_state_data(failing_circuit)
        
        # Should return None or mock data gracefully
        assert state_data is None or isinstance(state_data, StateVisualizationData)


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestConvenienceFunctions:
    """Test convenience functions for easy usage."""
    
    def test_visualize_quantum_circuit_function(self, mock_circuit):
        """Test visualize_quantum_circuit convenience function."""
        fig = visualize_quantum_circuit(mock_circuit, title="Test Circuit")
        
        assert isinstance(fig, plt.Figure)
        assert "Test Circuit" in fig.axes[0].get_title()
        
        plt.close(fig)
    
    def test_visualize_quantum_state_function(self):
        """Test visualize_quantum_state convenience function."""
        state_vector = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        
        fig = visualize_quantum_state(state_vector, visualization_type="amplitudes")
        
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)
    
    def test_create_bloch_sphere_visualization_function(self):
        """Test create_bloch_sphere_visualization convenience function."""
        qubit_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        
        fig = create_bloch_sphere_visualization(qubit_state, title="Test Bloch")
        
        assert isinstance(fig, plt.Figure)
        assert "Test Bloch" in fig.axes[0].get_title()
        
        plt.close(fig)
    
    def test_bloch_sphere_invalid_state_size(self):
        """Test Bloch sphere with invalid state vector size."""
        invalid_state = np.array([1, 0, 0], dtype=complex)  # 3D state
        
        with pytest.raises(ValueError, match="2-dimensional state vector"):
            create_bloch_sphere_visualization(invalid_state)
    
    def test_compare_quantum_algorithms_function(self):
        """Test compare_quantum_algorithms convenience function."""
        # Create mock algorithms
        algorithms = []
        for i in range(2):
            algorithm = Mock()
            algorithm.num_qubits = 2
            algorithm.gates = [Mock()] * (i + 1)
            algorithms.append(algorithm)
        
        algorithm_names = ["Algo 1", "Algo 2"]
        
        fig = compare_quantum_algorithms(algorithms, algorithm_names)
        
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestGUIComponents:
    """Test GUI components (if available)."""
    
    @pytest.mark.skipif(not HAS_TKINTER, reason="tkinter not available")
    def test_visualization_gui_creation(self):
        """Test VisualizationGUI creation."""
        from quantrs2.quantum_algorithm_visualization import VisualizationGUI
        
        # Create GUI instance (don't start mainloop)
        gui = VisualizationGUI()
        
        assert gui.root is not None
        assert isinstance(gui.visualizer, QuantumAlgorithmVisualizer)
        
        # Clean up
        gui.root.destroy()
    
    @pytest.mark.skipif(not HAS_DASH, reason="dash not available")
    def test_web_app_creation(self):
        """Test web application creation."""
        from quantrs2.quantum_algorithm_visualization import create_quantum_visualization_app
        
        app = create_quantum_visualization_app()
        
        assert app is not None
        assert hasattr(app, 'layout')


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestIntegrationScenarios:
    """Integration tests for complete visualization workflows."""
    
    def test_complete_visualization_workflow(self, mock_circuit, mock_performance_metrics):
        """Test complete end-to-end visualization workflow."""
        visualizer = QuantumAlgorithmVisualizer()
        
        # Set up performance data
        visualizer.performance_data = mock_performance_metrics
        
        # Generate comprehensive visualization
        figures = visualizer.visualize_algorithm_execution(
            mock_circuit,
            include_state_evolution=True,
            include_performance=True,
            title="Complete Workflow Test"
        )
        
        assert isinstance(figures, dict)
        assert len(figures) >= 1
        
        # Test export of one figure
        if figures:
            first_fig = list(figures.values())[0]
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                temp_path = f.name
            
            try:
                visualizer.export_visualization(first_fig, temp_path)
                assert os.path.exists(temp_path)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        # Clean up figures
        for fig in figures.values():
            plt.close(fig)
    
    def test_configuration_customization_workflow(self):
        """Test workflow with custom configuration."""
        # Create custom config
        config = VisualizationConfig(
            figure_size=(15, 10),
            color_scheme="colorblind",
            animation_speed=0.5,
            export_quality="high"
        )
        
        visualizer = QuantumAlgorithmVisualizer(config)
        
        # Verify config propagation
        assert visualizer.circuit_visualizer.config == config
        assert visualizer.state_visualizer.config == config
        assert visualizer.performance_visualizer.config == config
        
        # Test visualization with custom config
        circuit_data = CircuitVisualizationData()
        circuit_data.add_gate("H", [0])
        
        fig = visualizer.circuit_visualizer.visualize_circuit(circuit_data)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_multi_format_export_workflow(self, mock_circuit):
        """Test exporting visualizations in multiple formats."""
        visualizer = QuantumAlgorithmVisualizer()
        circuit_data = visualizer._extract_circuit_data(mock_circuit)
        fig = visualizer.circuit_visualizer.visualize_circuit(circuit_data)
        
        formats = ["png", "pdf", "svg", "html"]
        temp_files = []
        
        try:
            for fmt in formats:
                with tempfile.NamedTemporaryFile(suffix=f'.{fmt}', delete=False) as f:
                    temp_path = f.name
                    temp_files.append(temp_path)
                
                visualizer.export_visualization(fig, temp_path, format=fmt)
                assert os.path.exists(temp_path)
                assert os.path.getsize(temp_path) > 0
                
        finally:
            # Clean up
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            plt.close(fig)
    
    def test_state_evolution_visualization_workflow(self):
        """Test complete state evolution visualization workflow."""
        visualizer = QuantumAlgorithmVisualizer()
        
        # Create state with time evolution
        state_data = StateVisualizationData()
        
        # Add time evolution: rotating qubit on Bloch sphere
        times = np.linspace(0, 2*np.pi, 20)
        for t in times:
            state_vector = np.array([np.cos(t/2), np.sin(t/2)], dtype=complex)
            state_data.time_evolution.append((t, state_vector))
        
        # Create Bloch sphere animation
        anim = visualizer.state_visualizer.create_state_evolution_animation(
            state_data, qubit_index=0, title="Rotating Qubit"
        )
        
        assert hasattr(anim, 'fig')
        assert hasattr(anim, 'event_source')
        
        plt.close(anim.fig)
    
    def test_performance_integration_workflow(self, mock_performance_metrics):
        """Test performance profiler integration workflow."""
        if not HAS_PROFILER:
            pytest.skip("Performance profiler not available")
        
        visualizer = QuantumAlgorithmVisualizer()
        
        # Mock profiler integration
        mock_profiler = Mock()
        mock_profiler.all_metrics = mock_performance_metrics
        
        visualizer.integrate_with_profiler(mock_profiler)
        
        # Create performance visualization
        fig = visualizer.performance_visualizer.visualize_performance_metrics(
            mock_performance_metrics,
            title="Integrated Performance Analysis"
        )
        
        assert isinstance(fig, plt.Figure)
        assert "Integrated Performance Analysis" in fig._suptitle.get_text()
        
        plt.close(fig)


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_state_vector_visualization(self):
        """Test visualization with invalid state vector."""
        visualizer = StateVisualizer()
        
        # Non-normalized state vector
        invalid_state = np.array([1, 1], dtype=complex)  # Not normalized
        state_data = StateVisualizationData(state_vector=invalid_state)
        
        # Should still work (visualization doesn't enforce normalization)
        fig = visualizer.visualize_state_vector(state_data, visualization_type="amplitudes")
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_zero_state_vector_visualization(self):
        """Test visualization with zero state vector."""
        visualizer = StateVisualizer()
        
        zero_state = np.array([0, 0], dtype=complex)
        state_data = StateVisualizationData(state_vector=zero_state)
        
        fig = visualizer.visualize_state_vector(state_data, visualization_type="probabilities")
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_circuit_visualization_with_invalid_gates(self):
        """Test circuit visualization with invalid gate data."""
        visualizer = CircuitVisualizer()
        circuit_data = CircuitVisualizationData()
        
        # Add gate with invalid qubit index
        circuit_data.gates.append({
            "id": 0,
            "type": "INVALID_GATE",
            "qubits": [-1],  # Invalid qubit index
            "params": [],
            "position": 0
        })
        circuit_data.total_gates = 1
        
        # Should handle gracefully
        layout = visualizer._calculate_circuit_layout(circuit_data)
        
        assert isinstance(layout, dict)
    
    def test_performance_visualization_with_missing_attributes(self):
        """Test performance visualization with metrics missing attributes."""
        visualizer = PerformanceVisualizer()
        
        # Create metric with missing attributes
        incomplete_metric = Mock()
        del incomplete_metric.execution_time  # Remove required attribute
        incomplete_metric.memory_usage = 100
        incomplete_metric.gate_count = 5
        incomplete_metric.circuit_depth = 3
        incomplete_metric.error_rate = 0.01
        incomplete_metric.fidelity = 0.99
        
        # Should handle gracefully (getattr will return 0 for missing attributes)
        try:
            fig = visualizer.visualize_performance_metrics([incomplete_metric])
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
        except AttributeError:
            # This is acceptable behavior
            pass
    
    def test_export_to_nonexistent_directory(self, mock_circuit):
        """Test export to non-existent directory."""
        visualizer = QuantumAlgorithmVisualizer()
        circuit_data = visualizer._extract_circuit_data(mock_circuit)
        fig = visualizer.circuit_visualizer.visualize_circuit(circuit_data)
        
        nonexistent_path = "/nonexistent/directory/test.png"
        
        # Should handle gracefully and log error
        try:
            visualizer.export_visualization(fig, nonexistent_path)
        except (OSError, IOError):
            # Expected behavior for non-existent directory
            pass
        
        plt.close(fig)
    
    def test_matplotlib_backend_issues(self):
        """Test handling of matplotlib backend issues."""
        # This test would check graceful handling of matplotlib backend issues
        # In practice, this is hard to test without actually changing backends
        
        # Just verify that basic visualization doesn't crash
        visualizer = CircuitVisualizer()
        circuit_data = CircuitVisualizationData()
        circuit_data.add_gate("H", [0])
        
        fig = visualizer.visualize_circuit(circuit_data)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestPerformanceAndScalability:
    """Test performance and scalability of visualization system."""
    
    def test_large_circuit_visualization_performance(self):
        """Test visualization performance with large circuits."""
        import time
        
        visualizer = CircuitVisualizer()
        circuit_data = CircuitVisualizationData()
        
        # Create large circuit (100 gates)
        n_gates = 100
        start_time = time.time()
        
        for i in range(n_gates):
            circuit_data.add_gate("H", [i % 5])  # 5 qubits, many gates
        
        # Measure layout calculation time
        layout_start = time.time()
        layout = visualizer._calculate_circuit_layout(circuit_data)
        layout_time = time.time() - layout_start
        
        # Should complete in reasonable time (< 1 second)
        assert layout_time < 1.0
        assert len(layout["gate_layers"]) == n_gates
        
        creation_time = time.time() - start_time
        assert creation_time < 2.0  # Total time should be reasonable
    
    def test_large_state_vector_visualization_performance(self):
        """Test visualization performance with large state vectors."""
        import time
        
        visualizer = StateVisualizer()
        
        # Create 4-qubit state (16 amplitudes)
        n_qubits = 4
        state_size = 2 ** n_qubits
        state_vector = np.zeros(state_size, dtype=complex)
        state_vector[0] = 1.0  # |0000⟩ state
        
        state_data = StateVisualizationData(state_vector=state_vector)
        
        start_time = time.time()
        fig = visualizer.visualize_state_vector(state_data, visualization_type="probabilities")
        visualization_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert visualization_time < 5.0
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)
    
    def test_many_performance_metrics_visualization(self):
        """Test visualization with many performance metrics."""
        visualizer = PerformanceVisualizer()
        
        # Create many performance metrics
        metrics = []
        for i in range(1000):
            metric = Mock()
            metric.execution_time = 0.1 + i * 0.001
            metric.memory_usage = 100 + i
            metric.gate_count = 5 + i % 10
            metric.circuit_depth = 3 + i % 5
            metric.error_rate = 0.01 + (i % 100) * 0.0001
            metric.fidelity = 0.99 - (i % 100) * 0.0001
            metrics.append(metric)
        
        # Should handle large dataset efficiently
        import time
        start_time = time.time()
        
        fig = visualizer.visualize_performance_metrics(metrics)
        
        visualization_time = time.time() - start_time
        assert visualization_time < 10.0  # Should complete in reasonable time
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)
    
    def test_memory_usage_during_visualization(self):
        """Test memory usage during visualization."""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        visualizer = QuantumAlgorithmVisualizer()
        
        # Create and visualize multiple circuits
        figures = []
        for i in range(10):
            circuit_data = CircuitVisualizationData()
            for j in range(10):
                circuit_data.add_gate("H", [j % 3])
            
            fig = visualizer.circuit_visualizer.visualize_circuit(circuit_data)
            figures.append(fig)
        
        mid_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Clean up figures
        for fig in figures:
            plt.close(fig)
        
        gc.collect()  # Force garbage collection
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory usage should not grow excessively
        memory_growth = mid_memory - initial_memory
        memory_cleanup = mid_memory - final_memory
        
        # These are rough heuristics - actual values depend on system
        assert memory_growth < 500  # Less than 500MB growth
        assert memory_cleanup > 0  # Some memory should be freed


if __name__ == "__main__":
    pytest.main([__file__])