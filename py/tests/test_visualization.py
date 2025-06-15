#!/usr/bin/env python3
"""
Test suite for quantum circuit visualization functionality.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

try:
    import quantrs2
    from quantrs2.visualization import (
        CircuitVisualizer, ProbabilityHistogram,
        visualize_circuit, visualize_probabilities
    )
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False


# Mock objects for testing when quantrs2 is not available
class MockCircuit:
    """Mock circuit for testing."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
    
    def visualize(self):
        # Simulate native visualizer not available
        raise AttributeError("Native visualizer not available")


class MockResult:
    """Mock simulation result for testing."""
    def __init__(self, n_qubits: int, probabilities: dict = None):
        self.n_qubits = n_qubits
        self._probabilities = probabilities or {}
    
    def state_probabilities(self):
        return self._probabilities


@pytest.mark.skipif(not HAS_VIZ, reason="visualization module not available")
class TestCircuitVisualizer:
    """Test CircuitVisualizer functionality."""
    
    def test_visualizer_initialization(self):
        """Test circuit visualizer initialization."""
        mock_circuit = MockCircuit(n_qubits=3)
        visualizer = CircuitVisualizer(mock_circuit)
        
        assert visualizer.circuit == mock_circuit
        assert visualizer.n_qubits == 3
        assert visualizer._has_native is False  # Mock doesn't have native
    
    def test_visualizer_with_quantrs2_circuit(self):
        """Test visualizer with real quantrs2 circuit if available."""
        if not hasattr(quantrs2, 'Circuit'):
            pytest.skip("quantrs2.Circuit not available")
        
        circuit = quantrs2.Circuit(2)
        circuit.h(0)
        circuit.cnot(0, 1)
        
        visualizer = CircuitVisualizer(circuit)
        
        assert visualizer.circuit == circuit
        assert visualizer.n_qubits == 2
    
    def test_text_representation(self):
        """Test text representation of circuit."""
        mock_circuit = MockCircuit(n_qubits=2)
        visualizer = CircuitVisualizer(mock_circuit)
        
        text = visualizer.text()
        
        assert isinstance(text, str)
        assert len(text) > 0
        assert "q0:" in text
        assert "q1:" in text
        assert "─" in text  # Should contain wire characters
    
    def test_html_representation(self):
        """Test HTML representation of circuit."""
        mock_circuit = MockCircuit(n_qubits=3)
        visualizer = CircuitVisualizer(mock_circuit)
        
        html = visualizer.html()
        
        assert isinstance(html, str)
        assert len(html) > 0
        assert "<div" in html
        assert "qc-container" in html
        assert "q0:" in html
        assert "q1:" in html
        assert "q2:" in html
    
    def test_repr_html(self):
        """Test HTML representation for Jupyter notebooks."""
        mock_circuit = MockCircuit(n_qubits=2)
        visualizer = CircuitVisualizer(mock_circuit)
        
        html_repr = visualizer._repr_html_()
        
        # Should be the same as html() method
        assert html_repr == visualizer.html()
        assert isinstance(html_repr, str)
        assert "<div" in html_repr
    
    def test_draw_text_style(self):
        """Test draw method with text style."""
        mock_circuit = MockCircuit(n_qubits=2)
        visualizer = CircuitVisualizer(mock_circuit)
        
        result = visualizer.draw(style='text')
        
        assert isinstance(result, str)
        assert result == visualizer.text()
    
    def test_draw_html_style(self):
        """Test draw method with HTML style."""
        mock_circuit = MockCircuit(n_qubits=2)
        visualizer = CircuitVisualizer(mock_circuit)
        
        result = visualizer.draw(style='html')
        
        # Should return HTML object (from IPython.display)
        assert hasattr(result, 'data')  # HTML objects have a data attribute
    
    def test_draw_invalid_style(self):
        """Test draw method with invalid style."""
        mock_circuit = MockCircuit(n_qubits=2)
        visualizer = CircuitVisualizer(mock_circuit)
        
        with pytest.raises(ValueError):
            visualizer.draw(style='invalid')
    
    def test_plot_circuit(self):
        """Test matplotlib plotting of circuit."""
        mock_circuit = MockCircuit(n_qubits=3)
        visualizer = CircuitVisualizer(mock_circuit)
        
        fig = visualizer.plot()
        
        assert isinstance(fig, plt.Figure)
        
        # Check that the plot has the expected structure
        ax = fig.get_axes()[0]
        assert ax is not None
        
        # Check y-axis limits (should accommodate all qubits)
        ylim = ax.get_ylim()
        assert ylim[0] <= -0.5
        assert ylim[1] >= 2.5  # For 3 qubits (0, 1, 2)
        
        # Clean up
        plt.close(fig)
    
    def test_plot_with_custom_figsize(self):
        """Test plotting with custom figure size."""
        mock_circuit = MockCircuit(n_qubits=2)
        visualizer = CircuitVisualizer(mock_circuit)
        
        fig = visualizer.plot(figsize=(12, 8))
        
        assert isinstance(fig, plt.Figure)
        assert fig.get_size_inches()[0] == 12
        assert fig.get_size_inches()[1] == 8
        
        plt.close(fig)
    
    def test_zero_qubit_circuit(self):
        """Test visualizer with zero qubits."""
        mock_circuit = MockCircuit(n_qubits=0)
        visualizer = CircuitVisualizer(mock_circuit)
        
        # Should handle gracefully
        text = visualizer.text()
        assert isinstance(text, str)
        
        html = visualizer.html()
        assert isinstance(html, str)
        
        fig = visualizer.plot()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_large_qubit_circuit(self):
        """Test visualizer with many qubits."""
        mock_circuit = MockCircuit(n_qubits=10)
        visualizer = CircuitVisualizer(mock_circuit)
        
        text = visualizer.text()
        assert "q9:" in text  # Should handle 10 qubits (0-9)
        
        html = visualizer.html()
        assert "q9:" in html
        
        fig = visualizer.plot()
        assert isinstance(fig, plt.Figure)
        
        # Figure height should adapt to number of qubits
        assert fig.get_size_inches()[1] > 5  # Should be taller for more qubits
        
        plt.close(fig)


@pytest.mark.skipif(not HAS_VIZ, reason="visualization module not available")
class TestProbabilityHistogram:
    """Test ProbabilityHistogram functionality."""
    
    def test_histogram_initialization(self):
        """Test probability histogram initialization."""
        probabilities = {'00': 0.5, '11': 0.5}
        mock_result = MockResult(n_qubits=2, probabilities=probabilities)
        
        histogram = ProbabilityHistogram(mock_result)
        
        assert histogram.result == mock_result
        assert histogram.n_qubits == 2
        assert histogram.probabilities == probabilities
    
    def test_histogram_with_quantrs2_result(self):
        """Test histogram with real quantrs2 result if available."""
        if not hasattr(quantrs2, 'Circuit'):
            pytest.skip("quantrs2.Circuit not available")
        
        circuit = quantrs2.Circuit(2)
        circuit.h(0)
        circuit.cnot(0, 1)
        result = circuit.run()
        
        histogram = ProbabilityHistogram(result)
        
        assert histogram.result == result
        assert histogram.n_qubits >= 0
        assert isinstance(histogram.probabilities, dict)
    
    def test_plot_histogram(self):
        """Test plotting probability histogram."""
        probabilities = {'00': 0.6, '01': 0.1, '10': 0.1, '11': 0.2}
        mock_result = MockResult(n_qubits=2, probabilities=probabilities)
        histogram = ProbabilityHistogram(mock_result)
        
        fig = histogram.plot()
        
        assert isinstance(fig, plt.Figure)
        
        # Check that the plot contains bars
        ax = fig.get_axes()[0]
        bars = ax.patches
        assert len(bars) > 0
        
        # Check labels
        assert ax.get_xlabel() == 'Basis State'
        assert ax.get_ylabel() == 'Probability'
        assert 'Quantum State Probabilities' in ax.get_title()
        
        plt.close(fig)
    
    def test_plot_with_threshold(self):
        """Test plotting with probability threshold."""
        probabilities = {'00': 0.8, '01': 0.05, '10': 0.05, '11': 0.1}
        mock_result = MockResult(n_qubits=2, probabilities=probabilities)
        histogram = ProbabilityHistogram(mock_result)
        
        # Plot with high threshold - should filter out small probabilities
        fig = histogram.plot(threshold=0.1)
        
        ax = fig.get_axes()[0]
        bars = ax.patches
        
        # Should have fewer bars due to threshold
        assert len(bars) <= 2  # Only '00' and '11' should remain
        
        plt.close(fig)
    
    def test_plot_with_max_states(self):
        """Test plotting with maximum states limit."""
        # Create many states
        probabilities = {f'{i:02b}': 0.125 for i in range(8)}  # 3-qubit uniform
        mock_result = MockResult(n_qubits=3, probabilities=probabilities)
        histogram = ProbabilityHistogram(mock_result)
        
        # Limit to 4 states
        fig = histogram.plot(max_states=4)
        
        ax = fig.get_axes()[0]
        bars = ax.patches
        
        # Should have at most 4 bars
        assert len(bars) <= 4
        
        plt.close(fig)
    
    def test_plot_empty_probabilities(self):
        """Test plotting with no probabilities."""
        mock_result = MockResult(n_qubits=2, probabilities={})
        histogram = ProbabilityHistogram(mock_result)
        
        fig = histogram.plot()
        
        assert isinstance(fig, plt.Figure)
        
        # Should handle empty data gracefully
        ax = fig.get_axes()[0]
        bars = ax.patches
        assert len(bars) == 0
        
        plt.close(fig)
    
    def test_html_representation(self):
        """Test HTML representation of histogram."""
        probabilities = {'00': 0.7, '01': 0.1, '10': 0.1, '11': 0.1}
        mock_result = MockResult(n_qubits=2, probabilities=probabilities)
        histogram = ProbabilityHistogram(mock_result)
        
        html = histogram.html()
        
        assert isinstance(html, str)
        assert len(html) > 0
        assert "qp-container" in html
        assert "|00⟩" in html
        assert "|11⟩" in html
        assert "0.7000" in html  # Should show probability values
    
    def test_html_with_filtering(self):
        """Test HTML representation with probability filtering."""
        probabilities = {'00': 0.95, '01': 0.02, '10': 0.02, '11': 0.01}
        mock_result = MockResult(n_qubits=2, probabilities=probabilities)
        histogram = ProbabilityHistogram(mock_result)
        
        html = histogram.html()
        
        # Small probabilities (< 0.001) should be filtered out
        # In this case, 0.01 and 0.02 should still be included
        assert "|00⟩" in html
        assert "|01⟩" in html
        assert "|10⟩" in html
        assert "|11⟩" in html
    
    def test_repr_html(self):
        """Test HTML representation for Jupyter notebooks."""
        probabilities = {'0': 0.6, '1': 0.4}
        mock_result = MockResult(n_qubits=1, probabilities=probabilities)
        histogram = ProbabilityHistogram(mock_result)
        
        html_repr = histogram._repr_html_()
        
        # Should be the same as html() method
        assert html_repr == histogram.html()
        assert isinstance(html_repr, str)
        assert "qp-container" in html_repr
    
    def test_custom_figsize(self):
        """Test plotting with custom figure size."""
        probabilities = {'00': 0.5, '11': 0.5}
        mock_result = MockResult(n_qubits=2, probabilities=probabilities)
        histogram = ProbabilityHistogram(mock_result)
        
        fig = histogram.plot(figsize=(15, 10))
        
        assert isinstance(fig, plt.Figure)
        assert fig.get_size_inches()[0] == 15
        assert fig.get_size_inches()[1] == 10
        
        plt.close(fig)


@pytest.mark.skipif(not HAS_VIZ, reason="visualization module not available")
class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_visualize_circuit_function(self):
        """Test visualize_circuit convenience function."""
        mock_circuit = MockCircuit(n_qubits=2)
        
        visualizer = visualize_circuit(mock_circuit)
        
        assert isinstance(visualizer, CircuitVisualizer)
        assert visualizer.circuit == mock_circuit
        assert visualizer.n_qubits == 2
    
    def test_visualize_probabilities_function(self):
        """Test visualize_probabilities convenience function."""
        probabilities = {'0': 0.3, '1': 0.7}
        mock_result = MockResult(n_qubits=1, probabilities=probabilities)
        
        histogram = visualize_probabilities(mock_result)
        
        assert isinstance(histogram, ProbabilityHistogram)
        assert histogram.result == mock_result
        assert histogram.n_qubits == 1
        assert histogram.probabilities == probabilities


@pytest.mark.skipif(not HAS_VIZ, reason="visualization module not available")
class TestVisualizationIntegration:
    """Test integration between visualization components."""
    
    def test_circuit_and_result_visualization(self):
        """Test visualizing both circuit and its results."""
        if not hasattr(quantrs2, 'Circuit'):
            pytest.skip("quantrs2.Circuit not available")
        
        # Create circuit
        circuit = quantrs2.Circuit(2)
        circuit.h(0)
        
        # Visualize circuit
        circuit_viz = visualize_circuit(circuit)
        circuit_text = circuit_viz.text()
        
        assert isinstance(circuit_text, str)
        assert len(circuit_text) > 0
        
        # Run circuit and visualize results
        result = circuit.run()
        result_viz = visualize_probabilities(result)
        result_html = result_viz.html()
        
        assert isinstance(result_html, str)
        assert len(result_html) > 0
    
    def test_visualization_consistency(self):
        """Test consistency between different visualization methods."""
        mock_circuit = MockCircuit(n_qubits=3)
        visualizer = CircuitVisualizer(mock_circuit)
        
        # Different representations should be consistent
        text1 = visualizer.text()
        text2 = visualizer.draw(style='text')
        
        assert text1 == text2
        
        html1 = visualizer.html()
        html2 = visualizer._repr_html_()
        
        assert html1 == html2
    
    def test_error_handling_in_visualization(self):
        """Test error handling in visualization components."""
        # Circuit with missing attributes
        class BrokenCircuit:
            pass  # No n_qubits attribute
        
        broken_circuit = BrokenCircuit()
        visualizer = CircuitVisualizer(broken_circuit)
        
        # Should handle gracefully with default values
        assert visualizer.n_qubits == 0
        text = visualizer.text()
        assert isinstance(text, str)
        
        # Result with broken state_probabilities method
        class BrokenResult:
            def __init__(self):
                self.n_qubits = 2
            
            def state_probabilities(self):
                raise Exception("Broken method")
        
        broken_result = BrokenResult()
        histogram = ProbabilityHistogram(broken_result)
        
        # Should handle gracefully with empty probabilities
        assert histogram.probabilities == {}
        html = histogram.html()
        assert isinstance(html, str)


@pytest.mark.skipif(not HAS_VIZ, reason="visualization module not available")
class TestVisualizationPerformance:
    """Test performance characteristics of visualization."""
    
    def test_large_state_visualization(self):
        """Test visualization with large quantum states."""
        # Create large probability distribution
        n_states = 64  # 6 qubits worth of states
        probabilities = {f'{i:06b}': 1.0/n_states for i in range(n_states)}
        mock_result = MockResult(n_qubits=6, probabilities=probabilities)
        
        histogram = ProbabilityHistogram(mock_result)
        
        # Should handle large states gracefully
        html = histogram.html()
        assert isinstance(html, str)
        
        # Plot should limit number of displayed states
        fig = histogram.plot()
        ax = fig.get_axes()[0]
        bars = ax.patches
        assert len(bars) <= 16  # Default max_states
        
        plt.close(fig)
    
    def test_visualization_memory_efficiency(self):
        """Test that visualization doesn't consume excessive memory."""
        # Create multiple visualizers
        circuits = [MockCircuit(n_qubits=i) for i in range(1, 6)]
        visualizers = [CircuitVisualizer(circuit) for circuit in circuits]
        
        # Generate multiple visualizations
        texts = [viz.text() for viz in visualizers]
        htmls = [viz.html() for viz in visualizers]
        
        # Should complete without memory issues
        assert len(texts) == 5
        assert len(htmls) == 5
        assert all(isinstance(text, str) for text in texts)
        assert all(isinstance(html, str) for html in htmls)


if __name__ == "__main__":
    pytest.main([__file__])