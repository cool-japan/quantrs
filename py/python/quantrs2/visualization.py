"""
Quantum circuit visualization for QuantRS2.

This module provides Python-side visualization capabilities for quantum circuits,
complementing the Rust-side visualization implementation.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from IPython.display import HTML

class CircuitVisualizer:
    """
    Quantum circuit visualization helper.
    
    This class provides methods to visualize quantum circuits using
    matplotlib and HTML rendering for Jupyter notebooks.
    """
    
    def __init__(self, circuit: Any):
        """
        Initialize a new circuit visualizer.
        
        Args:
            circuit: The PyCircuit object to visualize
        """
        self.circuit = circuit
        self.n_qubits = getattr(circuit, 'n_qubits', 0)
        
        # Try to access the native visualizer
        try:
            self._native_visualizer = circuit.visualize()
            self._has_native = True
        except (AttributeError, Exception):
            self._has_native = False
    
    def text(self) -> str:
        """
        Get a text representation of the circuit.
        
        Returns:
            ASCII text representation of the circuit
        """
        if self._has_native:
            try:
                return self._native_visualizer.to_text()
            except Exception:
                pass
        
        # Fallback text representation
        text = []
        for q in range(self.n_qubits):
            line = f"q{q}: " + "─" * 20
            text.append(line)
        
        return "\n".join(text)
    
    def html(self) -> str:
        """
        Get an HTML representation of the circuit for Jupyter notebooks.
        
        Returns:
            HTML representation of the circuit
        """
        if self._has_native:
            try:
                return self._native_visualizer.to_html()
            except Exception:
                pass
        
        # Fallback HTML representation
        html = """
        <style>
            .qc-container {
                font-family: monospace;
                margin: 10px 0;
                display: grid;
            }
            .qc-qubit-labels {
                grid-column: 1;
                display: grid;
                align-items: center;
                margin-right: 10px;
            }
            .qc-qubit-label {
                text-align: right;
                padding-right: 5px;
                height: 30px;
                line-height: 30px;
            }
            .qc-wire {
                height: 2px;
                background-color: #000;
                width: 100%;
                margin: 15px 0;
            }
        </style>
        <div class="qc-container">
          <div class="qc-qubit-labels">
        """
        
        # Add qubit labels
        for q in range(self.n_qubits):
            html += f'<div class="qc-qubit-label">q{q}:</div>\n'
        
        html += """
          </div>
          <div class="qc-circuit">
        """
        
        # Add qubit wires
        for q in range(self.n_qubits):
            html += '<div class="qc-wire"></div>\n'
        
        html += """
          </div>
        </div>
        """
        
        return html
    
    def _repr_html_(self) -> str:
        """
        HTML representation for Jupyter notebook display.
        
        Returns:
            HTML representation of the circuit
        """
        return self.html()
    
    def draw(self, style: str = 'text') -> Union[str, HTML]:
        """
        Draw the circuit in the specified style.
        
        Args:
            style: 'text' for ASCII text or 'html' for HTML
            
        Returns:
            Circuit visualization in the specified format
        """
        if style == 'text':
            return self.text()
        elif style == 'html':
            return HTML(self.html())
        else:
            raise ValueError(f"Unknown style: {style}. Use 'text' or 'html'")
    
    def plot(self, figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot the circuit using matplotlib.
        
        Args:
            figsize: Optional figure size (width, height)
            
        Returns:
            Matplotlib figure with the circuit visualization
        """
        if figsize is None:
            figsize = (8, self.n_qubits * 0.7 + 1)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw qubit lines
        for q in range(self.n_qubits):
            ax.axhline(y=q, color='k', lw=1)
            ax.text(-0.5, q, f'q{q}: ', ha='right', va='center')
        
        # Remove axis ticks and spines
        ax.set_yticks([])
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Set limits with some padding (ensure non-identical values to avoid warnings)
        y_max = max(self.n_qubits - 0.5, 0.5)  # Ensure at least 1 unit height
        ax.set_ylim(-0.5, y_max)
        ax.set_xlim(-0.5, 10.5)
        
        # Add title
        ax.set_title(f'Quantum Circuit ({self.n_qubits} qubits)')
        
        plt.tight_layout()
        return fig


class ProbabilityHistogram:
    """
    Probability histogram visualizer for quantum states.
    
    This class provides methods to visualize the probability distribution
    of a quantum state from simulation results.
    """
    
    def __init__(self, result: Any):
        """
        Initialize a new probability histogram visualizer.
        
        Args:
            result: The PySimulationResult object to visualize
        """
        self.result = result
        self.n_qubits = getattr(result, 'n_qubits', 0)
        
        # Get probabilities
        try:
            self.probabilities = result.state_probabilities()
        except (AttributeError, Exception):
            self.probabilities = {}
    
    def plot(self, figsize: Optional[Tuple[int, int]] = None, 
             threshold: float = 0.01, max_states: int = 16) -> plt.Figure:
        """
        Plot the probability histogram.
        
        Args:
            figsize: Optional figure size (width, height)
            threshold: Minimum probability to include in the plot
            max_states: Maximum number of states to display
            
        Returns:
            Matplotlib figure with the probability histogram
        """
        if figsize is None:
            figsize = (10, 6)
        
        # Filter probabilities by threshold
        filtered_probs = {k: v for k, v in self.probabilities.items() if v >= threshold}
        
        # Sort by probability (descending) and limit to max_states
        sorted_items = sorted(filtered_probs.items(), key=lambda x: x[1], reverse=True)[:max_states]
        states, probs = zip(*sorted_items) if sorted_items else ([], [])
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create the bar plot
        bars = ax.bar(states, probs)
        
        # Add value labels above bars
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.4f}', ha='center', va='bottom', rotation=0)
        
        # Add labels and title
        ax.set_xlabel('Basis State')
        ax.set_ylabel('Probability')
        ax.set_title(f'Quantum State Probabilities ({self.n_qubits} qubits)')
        
        # Ensure y-axis starts at 0 and goes slightly above max probability
        ax.set_ylim(0, max(probs) * 1.15 if probs else 1.0)
        
        plt.tight_layout()
        return fig
    
    def html(self) -> str:
        """
        Get an HTML representation of the probability histogram.
        
        Returns:
            HTML representation of the probability histogram
        """
        # Filter and sort probabilities
        filtered_probs = {k: v for k, v in self.probabilities.items() if v >= 0.001}
        sorted_items = sorted(filtered_probs.items(), key=lambda x: x[1], reverse=True)[:16]
        
        # Start HTML
        html = """
        <style>
            .qp-container {
                font-family: sans-serif;
                margin: 10px 0;
                max-width: 800px;
            }
            .qp-title {
                font-weight: bold;
                margin-bottom: 10px;
            }
            .qp-bar-container {
                height: 20px;
                background-color: #eee;
                margin: 5px 0;
                position: relative;
                width: 70%;
                display: inline-block;
            }
            .qp-bar {
                height: 100%;
                background-color: #3498db;
                position: absolute;
                top: 0;
                left: 0;
            }
            .qp-label {
                display: inline-block;
                width: 15%;
                text-align: right;
                padding-right: 10px;
            }
            .qp-value {
                display: inline-block;
                width: 10%;
                text-align: left;
                padding-left: 10px;
            }
        </style>
        <div class="qp-container">
          <div class="qp-title">Quantum State Probabilities</div>
        """
        
        # Add bars for each state
        for state, prob in sorted_items:
            # Calculate width as percentage
            width = int(prob * 100)
            html += f"""
            <div>
              <span class="qp-label">|{state}⟩:</span>
              <div class="qp-bar-container">
                <div class="qp-bar" style="width: {width}%;"></div>
              </div>
              <span class="qp-value">{prob:.4f}</span>
            </div>
            """
        
        html += "</div>"
        return html
    
    def _repr_html_(self) -> str:
        """
        HTML representation for Jupyter notebook display.
        
        Returns:
            HTML representation of the probability histogram
        """
        return self.html()


def visualize_circuit(circuit: Any) -> CircuitVisualizer:
    """
    Create a circuit visualizer for the given circuit.
    
    Args:
        circuit: PyCircuit object to visualize
        
    Returns:
        CircuitVisualizer for the circuit
    """
    return CircuitVisualizer(circuit)


def visualize_probabilities(result: Any) -> ProbabilityHistogram:
    """
    Create a probability histogram visualizer for the given result.
    
    Args:
        result: PySimulationResult object to visualize
        
    Returns:
        ProbabilityHistogram for the result
    """
    return ProbabilityHistogram(result)