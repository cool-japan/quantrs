"""
QuantRS2 Quantum Algorithm Visualization System

A comprehensive visualization framework for quantum algorithms, circuits, and quantum states.
Provides interactive plots, circuit diagrams, state evolution visualization, and seamless
integration with performance analytics and profiling systems.

Features:
- Interactive quantum circuit diagram visualization
- Real-time quantum state evolution visualization 
- 3D Bloch sphere and state space visualizations
- Performance analytics integration with profiling data
- Animation capabilities for algorithm execution
- Comparative visualization tools for multiple algorithms
- Export capabilities for presentations and publications
- Integration with QuantRS2 profiling and debugging tools

Author: QuantRS2 Team
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.offline as pyo
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
import time
import threading
import json
import warnings
from pathlib import Path
import logging
from collections import defaultdict, deque
import pandas as pd

# Optional dependencies with graceful fallbacks
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False

try:
    import ipywidgets as widgets
    from IPython.display import display, HTML
    HAS_JUPYTER = True
except ImportError:
    HAS_JUPYTER = False

try:
    import dash
    from dash import dcc, html, Input, Output, State, callback
    import dash_bootstrap_components as dbc
    HAS_DASH = True
except ImportError:
    HAS_DASH = False

try:
    from quantum_performance_profiler import PerformanceMetrics, QuantumPerformanceProfiler
    HAS_PROFILER = True
except ImportError:
    HAS_PROFILER = False

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for quantum algorithm visualization."""
    
    # Display settings
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 100
    style: str = "seaborn-v0_8"  # matplotlib style
    color_scheme: str = "quantum"  # quantum, classical, colorblind
    animation_speed: float = 1.0  # seconds per frame
    
    # Circuit visualization
    gate_spacing: float = 1.0
    qubit_spacing: float = 1.0
    gate_width: float = 0.8
    gate_height: float = 0.6
    wire_thickness: float = 2.0
    
    # State visualization  
    bloch_sphere_resolution: int = 50
    state_vector_threshold: float = 1e-10
    probability_threshold: float = 1e-6
    max_displayed_states: int = 16
    
    # Performance integration
    enable_profiling_overlay: bool = True
    show_execution_times: bool = True
    show_memory_usage: bool = True
    performance_colormap: str = "viridis"
    
    # Export settings
    export_formats: List[str] = field(default_factory=lambda: ["png", "pdf", "svg", "html"])
    export_quality: str = "high"  # low, medium, high
    include_metadata: bool = True
    
    # Interactive features
    enable_tooltips: bool = True
    enable_zoom: bool = True
    enable_pan: bool = True
    enable_selection: bool = True
    
    def get_color_palette(self) -> Dict[str, str]:
        """Get color palette based on color scheme."""
        palettes = {
            "quantum": {
                "qubit": "#2E86AB",
                "gate": "#A23B72", 
                "measurement": "#F18F01",
                "classical": "#C73E1D",
                "entanglement": "#8E44AD",
                "background": "#FFFFFF",
                "text": "#2C3E50",
                "grid": "#ECF0F1"
            },
            "classical": {
                "qubit": "#34495E",
                "gate": "#7F8C8D",
                "measurement": "#E67E22",
                "classical": "#C0392B",
                "entanglement": "#9B59B6",
                "background": "#FFFFFF", 
                "text": "#2C3E50",
                "grid": "#BDC3C7"
            },
            "colorblind": {
                "qubit": "#0173B2",
                "gate": "#DE8F05",
                "measurement": "#CC78BC",
                "classical": "#CA9161",
                "entanglement": "#029E73",
                "background": "#FFFFFF",
                "text": "#000000",
                "grid": "#D3D3D3"
            }
        }
        return palettes.get(self.color_scheme, palettes["quantum"])


@dataclass
class CircuitVisualizationData:
    """Data structure for circuit visualization."""
    
    gates: List[Dict[str, Any]] = field(default_factory=list)
    qubits: List[int] = field(default_factory=list)
    classical_bits: List[int] = field(default_factory=list)
    measurements: List[Dict[str, Any]] = field(default_factory=list)
    barriers: List[int] = field(default_factory=list)
    
    # Timing and performance data
    gate_execution_times: Dict[int, float] = field(default_factory=dict)
    gate_fidelities: Dict[int, float] = field(default_factory=dict)
    gate_error_rates: Dict[int, float] = field(default_factory=dict)
    
    # Layout information
    circuit_depth: int = 0
    total_gates: int = 0
    entangling_gates: int = 0
    single_qubit_gates: int = 0
    
    def add_gate(self, gate_type: str, qubits: List[int], params: Optional[List[float]] = None,
                 execution_time: Optional[float] = None, fidelity: Optional[float] = None):
        """Add a gate to the visualization data."""
        gate_id = len(self.gates)
        gate_data = {
            "id": gate_id,
            "type": gate_type,
            "qubits": qubits,
            "params": params or [],
            "position": gate_id,  # Position in circuit
            "layer": None  # Will be calculated during layout
        }
        
        self.gates.append(gate_data)
        
        if execution_time is not None:
            self.gate_execution_times[gate_id] = execution_time
        if fidelity is not None:
            self.gate_fidelities[gate_id] = fidelity
            
        # Update statistics
        self.total_gates += 1
        if len(qubits) > 1:
            self.entangling_gates += 1
        else:
            self.single_qubit_gates += 1
            
        # Update qubit list
        for qubit in qubits:
            if qubit not in self.qubits:
                self.qubits.append(qubit)
                
        self.qubits.sort()


@dataclass
class StateVisualizationData:
    """Data structure for quantum state visualization."""
    
    state_vector: np.ndarray = field(default_factory=lambda: np.array([]))
    density_matrix: Optional[np.ndarray] = None
    measurement_probabilities: Dict[str, float] = field(default_factory=dict)
    
    # State evolution over time
    time_evolution: List[Tuple[float, np.ndarray]] = field(default_factory=list)
    measurement_history: List[Tuple[float, Dict[str, float]]] = field(default_factory=list)
    
    # Bloch sphere coordinates for single qubits
    bloch_coordinates: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)
    
    # Entanglement measures
    entanglement_entropy: float = 0.0
    schmidt_coefficients: List[float] = field(default_factory=list)
    purity: float = 1.0
    
    def calculate_bloch_coordinates(self, qubit_index: int) -> Tuple[float, float, float]:
        """Calculate Bloch sphere coordinates for a specific qubit."""
        if len(self.state_vector) == 0:
            return (0.0, 0.0, 1.0)  # |0⟩ state
            
        n_qubits = int(np.log2(len(self.state_vector)))
        if qubit_index >= n_qubits:
            return (0.0, 0.0, 1.0)
            
        # Calculate reduced density matrix for the qubit
        rho = self.get_reduced_density_matrix(qubit_index)
        
        # Extract Bloch vector coordinates
        x = 2 * np.real(rho[0, 1])
        y = 2 * np.imag(rho[0, 1])
        z = np.real(rho[0, 0] - rho[1, 1])
        
        self.bloch_coordinates[qubit_index] = (x, y, z)
        return (x, y, z)
    
    def get_reduced_density_matrix(self, qubit_index: int) -> np.ndarray:
        """Get reduced density matrix for a specific qubit."""
        if self.density_matrix is not None:
            # Use provided density matrix
            rho = self.density_matrix
        else:
            # Create density matrix from state vector
            rho = np.outer(self.state_vector, np.conj(self.state_vector))
            
        n_qubits = int(np.log2(rho.shape[0]))
        
        # Trace out all qubits except the target
        qubits_to_trace = [i for i in range(n_qubits) if i != qubit_index]
        
        reduced_rho = rho
        for qubit in sorted(qubits_to_trace, reverse=True):
            reduced_rho = self._partial_trace(reduced_rho, qubit, n_qubits - len([q for q in qubits_to_trace if q > qubit]))
            
        return reduced_rho
    
    def _partial_trace(self, rho: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
        """Compute partial trace over specified qubit."""
        dim = 2 ** n_qubits
        reduced_dim = dim // 2
        
        # Reshape for partial trace
        rho_reshaped = rho.reshape([2] * (2 * n_qubits))
        
        # Move traced qubit axes to the end
        axes_order = list(range(2 * n_qubits))
        axes_order.remove(qubit)
        axes_order.remove(qubit + n_qubits)
        axes_order.extend([qubit, qubit + n_qubits])
        
        rho_reordered = np.transpose(rho_reshaped, axes_order)
        rho_matrix = rho_reordered.reshape(reduced_dim, reduced_dim, 2, 2)
        
        # Trace over the last two dimensions
        return np.trace(rho_matrix, axis1=2, axis2=3)


class CircuitVisualizer:
    """Advanced quantum circuit visualization with performance integration."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.colors = self.config.get_color_palette()
        
        # Set matplotlib style
        plt.style.use(self.config.style)
        
        # Gate rendering functions
        self.gate_renderers = {
            "H": self._render_hadamard,
            "X": self._render_pauli_x,
            "Y": self._render_pauli_y,
            "Z": self._render_pauli_z,
            "CNOT": self._render_cnot,
            "CX": self._render_cnot,
            "CZ": self._render_cz,
            "RX": self._render_rotation_x,
            "RY": self._render_rotation_y,
            "RZ": self._render_rotation_z,
            "SWAP": self._render_swap,
            "MEASURE": self._render_measurement,
            "BARRIER": self._render_barrier
        }
        
    def visualize_circuit(self, circuit_data: CircuitVisualizationData, 
                         title: str = "Quantum Circuit",
                         show_performance: bool = True,
                         interactive: bool = False) -> plt.Figure:
        """Create comprehensive circuit visualization."""
        
        # Calculate circuit layout
        layout = self._calculate_circuit_layout(circuit_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        # Draw circuit elements
        self._draw_qubit_wires(ax, circuit_data, layout)
        self._draw_gates(ax, circuit_data, layout, show_performance)
        self._draw_measurements(ax, circuit_data, layout)
        self._draw_barriers(ax, circuit_data, layout)
        
        # Add performance overlay if enabled
        if show_performance and self.config.enable_profiling_overlay:
            self._add_performance_overlay(ax, circuit_data, layout)
            
        # Customize appearance
        self._customize_circuit_plot(ax, circuit_data, title)
        
        # Add interactive features if requested
        if interactive:
            self._add_interactive_features(fig, ax, circuit_data)
            
        return fig
    
    def create_animated_execution(self, circuit_data: CircuitVisualizationData,
                                execution_trace: List[Dict[str, Any]],
                                title: str = "Circuit Execution Animation") -> animation.FuncAnimation:
        """Create animated visualization of circuit execution."""
        
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        layout = self._calculate_circuit_layout(circuit_data)
        
        # Initialize static elements
        self._draw_qubit_wires(ax, circuit_data, layout)
        self._customize_circuit_plot(ax, circuit_data, title)
        
        # Animation state
        current_step = 0
        executed_gates = set()
        
        def animate(frame):
            nonlocal current_step
            
            # Clear previous gate highlights
            for artist in ax.artists + ax.patches:
                if hasattr(artist, '_animation_highlight'):
                    artist.remove()
                    
            # Draw gates up to current step
            for i, gate in enumerate(circuit_data.gates):
                if i <= frame:
                    executed_gates.add(i)
                    alpha = 1.0 if i == frame else 0.7
                    self._draw_single_gate(ax, gate, layout, alpha=alpha, 
                                         highlight=(i == frame))
                                         
            current_step = frame
            return ax.artists + ax.patches
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=len(circuit_data.gates),
            interval=1000 / self.config.animation_speed,
            blit=False, repeat=True
        )
        
        return anim
    
    def _calculate_circuit_layout(self, circuit_data: CircuitVisualizationData) -> Dict[str, Any]:
        """Calculate optimal layout for circuit visualization."""
        
        n_qubits = len(circuit_data.qubits)
        n_gates = len(circuit_data.gates)
        
        # Calculate gate layers to minimize visual overlap
        gate_layers = self._calculate_gate_layers(circuit_data)
        n_layers = max(gate_layers.values()) + 1 if gate_layers else 0
        
        # Layout parameters
        layout = {
            "n_qubits": n_qubits,
            "n_layers": n_layers,
            "qubit_positions": {q: i for i, q in enumerate(circuit_data.qubits)},
            "layer_positions": {i: i * self.config.gate_spacing for i in range(n_layers)},
            "gate_layers": gate_layers,
            "circuit_width": n_layers * self.config.gate_spacing,
            "circuit_height": n_qubits * self.config.qubit_spacing
        }
        
        return layout
    
    def _calculate_gate_layers(self, circuit_data: CircuitVisualizationData) -> Dict[int, int]:
        """Calculate which layer each gate should be placed in."""
        
        gate_layers = {}
        qubit_last_layer = {q: -1 for q in circuit_data.qubits}
        
        for gate in circuit_data.gates:
            # Find the earliest layer where this gate can be placed
            min_layer = max(qubit_last_layer[q] for q in gate["qubits"]) + 1
            
            gate_layers[gate["id"]] = min_layer
            
            # Update last layer for affected qubits
            for qubit in gate["qubits"]:
                qubit_last_layer[qubit] = min_layer
                
        return gate_layers
    
    def _draw_qubit_wires(self, ax: plt.Axes, circuit_data: CircuitVisualizationData, layout: Dict[str, Any]):
        """Draw qubit wires."""
        
        for i, qubit in enumerate(circuit_data.qubits):
            y = i * self.config.qubit_spacing
            ax.plot([0, layout["circuit_width"]], [y, y], 
                   color=self.colors["qubit"], linewidth=self.config.wire_thickness,
                   solid_capstyle='round')
            
            # Add qubit labels
            ax.text(-0.5, y, f"q[{qubit}]", ha="right", va="center",
                   fontsize=12, color=self.colors["text"])
    
    def _draw_gates(self, ax: plt.Axes, circuit_data: CircuitVisualizationData, 
                   layout: Dict[str, Any], show_performance: bool = True):
        """Draw all gates in the circuit."""
        
        for gate in circuit_data.gates:
            self._draw_single_gate(ax, gate, layout, 
                                 show_performance=show_performance)
    
    def _draw_single_gate(self, ax: plt.Axes, gate: Dict[str, Any], layout: Dict[str, Any],
                         alpha: float = 1.0, highlight: bool = False, show_performance: bool = True):
        """Draw a single gate."""
        
        gate_type = gate["type"]
        qubits = gate["qubits"]
        layer = layout["gate_layers"][gate["id"]]
        
        # Get gate position
        x = layout["layer_positions"][layer]
        
        # Get performance data if available
        performance_color = None
        if show_performance and hasattr(self, 'performance_data'):
            performance_color = self._get_performance_color(gate["id"])
        
        # Render specific gate type
        if gate_type in self.gate_renderers:
            self.gate_renderers[gate_type](ax, gate, x, qubits, layout, 
                                         alpha=alpha, highlight=highlight,
                                         performance_color=performance_color)
        else:
            # Generic gate rendering
            self._render_generic_gate(ax, gate, x, qubits, layout, 
                                    alpha=alpha, highlight=highlight,
                                    performance_color=performance_color)
    
    def _render_hadamard(self, ax: plt.Axes, gate: Dict[str, Any], x: float, 
                        qubits: List[int], layout: Dict[str, Any], **kwargs):
        """Render Hadamard gate."""
        
        qubit_y = layout["qubit_positions"][qubits[0]] * self.config.qubit_spacing
        color = kwargs.get('performance_color', self.colors["gate"])
        alpha = kwargs.get('alpha', 1.0)
        
        # Draw gate box
        rect = patches.Rectangle(
            (x - self.config.gate_width/2, qubit_y - self.config.gate_height/2),
            self.config.gate_width, self.config.gate_height,
            facecolor=color, edgecolor='black', alpha=alpha, linewidth=2
        )
        ax.add_patch(rect)
        
        # Add gate label
        ax.text(x, qubit_y, "H", ha="center", va="center", 
               fontsize=14, fontweight="bold", color="white")
        
        # Add highlight if requested
        if kwargs.get('highlight', False):
            highlight_rect = patches.Rectangle(
                (x - self.config.gate_width/2 - 0.1, qubit_y - self.config.gate_height/2 - 0.1),
                self.config.gate_width + 0.2, self.config.gate_height + 0.2,
                facecolor='none', edgecolor='red', linewidth=3
            )
            highlight_rect._animation_highlight = True
            ax.add_patch(highlight_rect)
    
    def _render_pauli_x(self, ax: plt.Axes, gate: Dict[str, Any], x: float,
                       qubits: List[int], layout: Dict[str, Any], **kwargs):
        """Render Pauli-X gate."""
        
        qubit_y = layout["qubit_positions"][qubits[0]] * self.config.qubit_spacing
        color = kwargs.get('performance_color', self.colors["gate"])
        alpha = kwargs.get('alpha', 1.0)
        
        rect = patches.Rectangle(
            (x - self.config.gate_width/2, qubit_y - self.config.gate_height/2),
            self.config.gate_width, self.config.gate_height,
            facecolor=color, edgecolor='black', alpha=alpha, linewidth=2
        )
        ax.add_patch(rect)
        
        ax.text(x, qubit_y, "X", ha="center", va="center",
               fontsize=14, fontweight="bold", color="white")
    
    def _render_pauli_y(self, ax: plt.Axes, gate: Dict[str, Any], x: float,
                       qubits: List[int], layout: Dict[str, Any], **kwargs):
        """Render Pauli-Y gate."""
        
        qubit_y = layout["qubit_positions"][qubits[0]] * self.config.qubit_spacing
        color = kwargs.get('performance_color', self.colors["gate"])
        alpha = kwargs.get('alpha', 1.0)
        
        rect = patches.Rectangle(
            (x - self.config.gate_width/2, qubit_y - self.config.gate_height/2),
            self.config.gate_width, self.config.gate_height,
            facecolor=color, edgecolor='black', alpha=alpha, linewidth=2
        )
        ax.add_patch(rect)
        
        ax.text(x, qubit_y, "Y", ha="center", va="center",
               fontsize=14, fontweight="bold", color="white")
    
    def _render_pauli_z(self, ax: plt.Axes, gate: Dict[str, Any], x: float,
                       qubits: List[int], layout: Dict[str, Any], **kwargs):
        """Render Pauli-Z gate."""
        
        qubit_y = layout["qubit_positions"][qubits[0]] * self.config.qubit_spacing
        color = kwargs.get('performance_color', self.colors["gate"])
        alpha = kwargs.get('alpha', 1.0)
        
        rect = patches.Rectangle(
            (x - self.config.gate_width/2, qubit_y - self.config.gate_height/2),
            self.config.gate_width, self.config.gate_height,
            facecolor=color, edgecolor='black', alpha=alpha, linewidth=2
        )
        ax.add_patch(rect)
        
        ax.text(x, qubit_y, "Z", ha="center", va="center",
               fontsize=14, fontweight="bold", color="white")
    
    def _render_cnot(self, ax: plt.Axes, gate: Dict[str, Any], x: float,
                    qubits: List[int], layout: Dict[str, Any], **kwargs):
        """Render CNOT gate."""
        
        control_y = layout["qubit_positions"][qubits[0]] * self.config.qubit_spacing
        target_y = layout["qubit_positions"][qubits[1]] * self.config.qubit_spacing
        alpha = kwargs.get('alpha', 1.0)
        
        # Draw control dot
        control_circle = patches.Circle((x, control_y), 0.15, 
                                      facecolor='black', alpha=alpha)
        ax.add_patch(control_circle)
        
        # Draw target circle with X
        target_circle = patches.Circle((x, target_y), 0.25,
                                     facecolor='white', edgecolor='black',
                                     alpha=alpha, linewidth=2)
        ax.add_patch(target_circle)
        
        # Draw X in target
        ax.plot([x-0.15, x+0.15], [target_y-0.15, target_y+0.15], 
               'k-', linewidth=2, alpha=alpha)
        ax.plot([x-0.15, x+0.15], [target_y+0.15, target_y-0.15],
               'k-', linewidth=2, alpha=alpha)
        
        # Draw connecting line
        ax.plot([x, x], [min(control_y, target_y), max(control_y, target_y)],
               'k-', linewidth=2, alpha=alpha)
    
    def _render_cz(self, ax: plt.Axes, gate: Dict[str, Any], x: float,
                  qubits: List[int], layout: Dict[str, Any], **kwargs):
        """Render CZ gate."""
        
        control_y = layout["qubit_positions"][qubits[0]] * self.config.qubit_spacing
        target_y = layout["qubit_positions"][qubits[1]] * self.config.qubit_spacing
        alpha = kwargs.get('alpha', 1.0)
        
        # Draw control dots for both qubits
        for qubit in qubits:
            qubit_y = layout["qubit_positions"][qubit] * self.config.qubit_spacing
            control_circle = patches.Circle((x, qubit_y), 0.15,
                                          facecolor='black', alpha=alpha)
            ax.add_patch(control_circle)
        
        # Draw connecting line
        ax.plot([x, x], [min(control_y, target_y), max(control_y, target_y)],
               'k-', linewidth=2, alpha=alpha)
    
    def _render_rotation_x(self, ax: plt.Axes, gate: Dict[str, Any], x: float,
                          qubits: List[int], layout: Dict[str, Any], **kwargs):
        """Render RX rotation gate."""
        
        qubit_y = layout["qubit_positions"][qubits[0]] * self.config.qubit_spacing
        color = kwargs.get('performance_color', self.colors["gate"])
        alpha = kwargs.get('alpha', 1.0)
        params = gate.get("params", [])
        
        rect = patches.Rectangle(
            (x - self.config.gate_width/2, qubit_y - self.config.gate_height/2),
            self.config.gate_width, self.config.gate_height,
            facecolor=color, edgecolor='black', alpha=alpha, linewidth=2
        )
        ax.add_patch(rect)
        
        # Show parameter if available
        if params:
            label = f"RX({params[0]:.2f})"
        else:
            label = "RX"
            
        ax.text(x, qubit_y, label, ha="center", va="center",
               fontsize=10, fontweight="bold", color="white")
    
    def _render_rotation_y(self, ax: plt.Axes, gate: Dict[str, Any], x: float,
                          qubits: List[int], layout: Dict[str, Any], **kwargs):
        """Render RY rotation gate."""
        
        qubit_y = layout["qubit_positions"][qubits[0]] * self.config.qubit_spacing
        color = kwargs.get('performance_color', self.colors["gate"])
        alpha = kwargs.get('alpha', 1.0)
        params = gate.get("params", [])
        
        rect = patches.Rectangle(
            (x - self.config.gate_width/2, qubit_y - self.config.gate_height/2),
            self.config.gate_width, self.config.gate_height,
            facecolor=color, edgecolor='black', alpha=alpha, linewidth=2
        )
        ax.add_patch(rect)
        
        if params:
            label = f"RY({params[0]:.2f})"
        else:
            label = "RY"
            
        ax.text(x, qubit_y, label, ha="center", va="center",
               fontsize=10, fontweight="bold", color="white")
    
    def _render_rotation_z(self, ax: plt.Axes, gate: Dict[str, Any], x: float,
                          qubits: List[int], layout: Dict[str, Any], **kwargs):
        """Render RZ rotation gate."""
        
        qubit_y = layout["qubit_positions"][qubits[0]] * self.config.qubit_spacing
        color = kwargs.get('performance_color', self.colors["gate"])
        alpha = kwargs.get('alpha', 1.0)
        params = gate.get("params", [])
        
        rect = patches.Rectangle(
            (x - self.config.gate_width/2, qubit_y - self.config.gate_height/2),
            self.config.gate_width, self.config.gate_height,
            facecolor=color, edgecolor='black', alpha=alpha, linewidth=2
        )
        ax.add_patch(rect)
        
        if params:
            label = f"RZ({params[0]:.2f})"
        else:
            label = "RZ"
            
        ax.text(x, qubit_y, label, ha="center", va="center",
               fontsize=10, fontweight="bold", color="white")
    
    def _render_swap(self, ax: plt.Axes, gate: Dict[str, Any], x: float,
                    qubits: List[int], layout: Dict[str, Any], **kwargs):
        """Render SWAP gate."""
        
        qubit1_y = layout["qubit_positions"][qubits[0]] * self.config.qubit_spacing
        qubit2_y = layout["qubit_positions"][qubits[1]] * self.config.qubit_spacing
        alpha = kwargs.get('alpha', 1.0)
        
        # Draw X symbols on both qubits
        for qubit_y in [qubit1_y, qubit2_y]:
            ax.plot([x-0.15, x+0.15], [qubit_y-0.15, qubit_y+0.15],
                   'k-', linewidth=3, alpha=alpha)
            ax.plot([x-0.15, x+0.15], [qubit_y+0.15, qubit_y-0.15],
                   'k-', linewidth=3, alpha=alpha)
        
        # Draw connecting line
        ax.plot([x, x], [min(qubit1_y, qubit2_y), max(qubit1_y, qubit2_y)],
               'k-', linewidth=2, alpha=alpha)
    
    def _render_measurement(self, ax: plt.Axes, gate: Dict[str, Any], x: float,
                           qubits: List[int], layout: Dict[str, Any], **kwargs):
        """Render measurement operation."""
        
        qubit_y = layout["qubit_positions"][qubits[0]] * self.config.qubit_spacing
        alpha = kwargs.get('alpha', 1.0)
        
        # Draw measurement box
        rect = patches.Rectangle(
            (x - self.config.gate_width/2, qubit_y - self.config.gate_height/2),
            self.config.gate_width, self.config.gate_height,
            facecolor=self.colors["measurement"], edgecolor='black',
            alpha=alpha, linewidth=2
        )
        ax.add_patch(rect)
        
        # Draw measurement arc
        arc = patches.Arc((x, qubit_y - 0.1), 0.4, 0.4, angle=0, theta1=0, theta2=180,
                         linewidth=2, color='white')
        ax.add_patch(arc)
        
        # Draw measurement arrow
        ax.annotate('', xy=(x + 0.1, qubit_y + 0.1), xytext=(x, qubit_y - 0.1),
                   arrowprops=dict(arrowstyle='->', color='white', lw=2))
    
    def _render_barrier(self, ax: plt.Axes, gate: Dict[str, Any], x: float,
                       qubits: List[int], layout: Dict[str, Any], **kwargs):
        """Render barrier."""
        
        alpha = kwargs.get('alpha', 1.0)
        
        # Draw vertical dashed line across all qubits
        y_min = -0.5 * self.config.qubit_spacing
        y_max = (len(qubits) - 0.5) * self.config.qubit_spacing
        
        ax.plot([x, x], [y_min, y_max], '--', color='gray',
               linewidth=2, alpha=alpha)
    
    def _render_generic_gate(self, ax: plt.Axes, gate: Dict[str, Any], x: float,
                           qubits: List[int], layout: Dict[str, Any], **kwargs):
        """Render generic gate type."""
        
        qubit_y = layout["qubit_positions"][qubits[0]] * self.config.qubit_spacing
        color = kwargs.get('performance_color', self.colors["gate"])
        alpha = kwargs.get('alpha', 1.0)
        
        rect = patches.Rectangle(
            (x - self.config.gate_width/2, qubit_y - self.config.gate_height/2),
            self.config.gate_width, self.config.gate_height,
            facecolor=color, edgecolor='black', alpha=alpha, linewidth=2
        )
        ax.add_patch(rect)
        
        # Use gate type as label, truncate if too long
        label = gate["type"][:6]
        ax.text(x, qubit_y, label, ha="center", va="center",
               fontsize=10, fontweight="bold", color="white")
    
    def _draw_measurements(self, ax: plt.Axes, circuit_data: CircuitVisualizationData, layout: Dict[str, Any]):
        """Draw measurement operations."""
        
        for measurement in circuit_data.measurements:
            # Measurements are typically at the end of the circuit
            x = layout["circuit_width"] + 0.5
            qubit = measurement["qubit"]
            qubit_y = layout["qubit_positions"][qubit] * self.config.qubit_spacing
            
            # Draw measurement symbol
            self._render_measurement(ax, measurement, x, [qubit], layout)
    
    def _draw_barriers(self, ax: plt.Axes, circuit_data: CircuitVisualizationData, layout: Dict[str, Any]):
        """Draw barriers in the circuit."""
        
        for barrier_pos in circuit_data.barriers:
            x = barrier_pos * self.config.gate_spacing
            self._render_barrier(ax, {}, x, circuit_data.qubits, layout)
    
    def _add_performance_overlay(self, ax: plt.Axes, circuit_data: CircuitVisualizationData, layout: Dict[str, Any]):
        """Add performance data overlay to circuit visualization."""
        
        if not circuit_data.gate_execution_times:
            return
            
        # Create colormap for execution times
        times = list(circuit_data.gate_execution_times.values())
        if not times:
            return
            
        min_time, max_time = min(times), max(times)
        
        # Add performance color bar
        sm = plt.cm.ScalarMappable(cmap=self.config.performance_colormap,
                                  norm=plt.Normalize(vmin=min_time, vmax=max_time))
        sm.set_array([])
        
        # Position colorbar
        cbar_ax = ax.figure.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = ax.figure.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Execution Time (s)', rotation=270, labelpad=20)
    
    def _get_performance_color(self, gate_id: int) -> str:
        """Get color based on gate performance data."""
        
        if not hasattr(self, 'performance_data') or gate_id not in self.performance_data:
            return self.colors["gate"]
            
        # Normalize performance metric to color
        perf_value = self.performance_data[gate_id]
        colormap = plt.cm.get_cmap(self.config.performance_colormap)
        
        # Assume performance_data is normalized between 0 and 1
        return colormap(perf_value)
    
    def _customize_circuit_plot(self, ax: plt.Axes, circuit_data: CircuitVisualizationData, title: str):
        """Customize circuit plot appearance."""
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Circuit Depth', fontsize=12)
        ax.set_ylabel('Qubits', fontsize=12)
        
        # Set axis limits
        ax.set_xlim(-1, max(4, len(circuit_data.gates)) * self.config.gate_spacing)
        ax.set_ylim(-0.5 * self.config.qubit_spacing, 
                   (len(circuit_data.qubits) - 0.5) * self.config.qubit_spacing)
        
        # Remove ticks and spines
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        # Set background color
        ax.set_facecolor(self.colors["background"])
        
        # Add grid if desired
        if self.config.enable_tooltips:
            ax.grid(True, alpha=0.3, color=self.colors["grid"])
    
    def _add_interactive_features(self, fig: plt.Figure, ax: plt.Axes, circuit_data: CircuitVisualizationData):
        """Add interactive features to circuit visualization."""
        
        if self.config.enable_tooltips:
            # Add hover tooltips for gates
            def on_hover(event):
                if event.inaxes == ax:
                    # Find which gate is being hovered
                    gate_info = self._find_gate_at_position(event.xdata, event.ydata, circuit_data)
                    if gate_info:
                        # Show tooltip with gate information
                        tooltip_text = self._format_gate_tooltip(gate_info)
                        ax.annotate(tooltip_text, xy=(event.xdata, event.ydata),
                                   xytext=(20, 20), textcoords="offset points",
                                   bbox=dict(boxstyle="round", fc="w"),
                                   arrowprops=dict(arrowstyle="->"))
                        fig.canvas.draw()
            
            fig.canvas.mpl_connect('motion_notify_event', on_hover)
    
    def _find_gate_at_position(self, x: float, y: float, circuit_data: CircuitVisualizationData) -> Optional[Dict[str, Any]]:
        """Find gate at given position."""
        
        # Implementation would check which gate is at the given coordinates
        # This is a simplified version
        for gate in circuit_data.gates:
            gate_x = gate.get("position", 0) * self.config.gate_spacing
            gate_qubits = gate["qubits"]
            
            # Check if position is within gate bounds
            if abs(x - gate_x) < self.config.gate_width / 2:
                for qubit in gate_qubits:
                    qubit_y = qubit * self.config.qubit_spacing
                    if abs(y - qubit_y) < self.config.gate_height / 2:
                        return gate
        return None
    
    def _format_gate_tooltip(self, gate: Dict[str, Any]) -> str:
        """Format tooltip text for a gate."""
        
        tooltip_lines = [
            f"Gate: {gate['type']}",
            f"Qubits: {gate['qubits']}"
        ]
        
        if gate.get("params"):
            tooltip_lines.append(f"Parameters: {gate['params']}")
            
        # Add performance data if available
        gate_id = gate.get("id")
        if hasattr(self, 'performance_data') and gate_id in self.performance_data:
            tooltip_lines.append(f"Execution Time: {self.performance_data[gate_id]:.4f}s")
            
        return "\n".join(tooltip_lines)


class StateVisualizer:
    """Advanced quantum state visualization with 3D capabilities."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.colors = self.config.get_color_palette()
        
    def visualize_state_vector(self, state_data: StateVisualizationData,
                             title: str = "Quantum State",
                             visualization_type: str = "amplitudes") -> plt.Figure:
        """Create comprehensive state vector visualization."""
        
        if len(state_data.state_vector) == 0:
            raise ValueError("State vector is empty")
            
        n_qubits = int(np.log2(len(state_data.state_vector)))
        
        if visualization_type == "amplitudes":
            return self._visualize_amplitudes(state_data, title)
        elif visualization_type == "probabilities":
            return self._visualize_probabilities(state_data, title)
        elif visualization_type == "phase":
            return self._visualize_phases(state_data, title)
        elif visualization_type == "bloch":
            return self._visualize_bloch_spheres(state_data, title)
        elif visualization_type == "density_matrix":
            return self._visualize_density_matrix(state_data, title)
        else:
            raise ValueError(f"Unknown visualization type: {visualization_type}")
    
    def create_bloch_sphere(self, qubit_index: int, state_data: StateVisualizationData,
                          title: Optional[str] = None) -> plt.Figure:
        """Create 3D Bloch sphere visualization for a specific qubit."""
        
        fig = plt.figure(figsize=self.config.figure_size)
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw Bloch sphere
        self._draw_bloch_sphere_surface(ax)
        self._draw_bloch_sphere_axes(ax)
        
        # Calculate and plot state vector
        x, y, z = state_data.calculate_bloch_coordinates(qubit_index)
        
        # Draw state vector
        ax.quiver(0, 0, 0, x, y, z, color='red', arrow_length_ratio=0.1, linewidth=3)
        
        # Add state point
        ax.scatter([x], [y], [z], color='red', s=100)
        
        # Customize plot
        if title is None:
            title = f"Bloch Sphere - Qubit {qubit_index}"
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Set axis properties
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_zlim(-1.1, 1.1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        return fig
    
    def create_state_evolution_animation(self, state_data: StateVisualizationData,
                                       qubit_index: int = 0,
                                       title: str = "State Evolution") -> animation.FuncAnimation:
        """Create animated visualization of state evolution on Bloch sphere."""
        
        if not state_data.time_evolution:
            raise ValueError("No time evolution data available")
            
        fig = plt.figure(figsize=self.config.figure_size)
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw static Bloch sphere
        self._draw_bloch_sphere_surface(ax)
        self._draw_bloch_sphere_axes(ax)
        
        # Initialize trajectory line
        trajectory_line, = ax.plot([], [], [], 'b-', alpha=0.5, linewidth=2)
        state_point, = ax.plot([], [], [], 'ro', markersize=8)
        state_vector = ax.quiver(0, 0, 0, 0, 0, 0, color='red', arrow_length_ratio=0.1)
        
        # Trajectory storage
        trajectory_x, trajectory_y, trajectory_z = [], [], []
        
        def animate(frame):
            # Get state at this time step
            time_step, state_vector_t = state_data.time_evolution[frame]
            
            # Calculate Bloch coordinates
            temp_state_data = StateVisualizationData(state_vector=state_vector_t)
            x, y, z = temp_state_data.calculate_bloch_coordinates(qubit_index)
            
            # Update trajectory
            trajectory_x.append(x)
            trajectory_y.append(y)
            trajectory_z.append(z)
            
            # Update plots
            trajectory_line.set_data_3d(trajectory_x, trajectory_y, trajectory_z)
            state_point.set_data_3d([x], [y], [z])
            
            # Update vector (this is simplified - would need to properly update quiver)
            ax.clear()
            self._draw_bloch_sphere_surface(ax)
            self._draw_bloch_sphere_axes(ax)
            ax.plot(trajectory_x, trajectory_y, trajectory_z, 'b-', alpha=0.5, linewidth=2)
            ax.scatter([x], [y], [z], color='red', s=100)
            ax.quiver(0, 0, 0, x, y, z, color='red', arrow_length_ratio=0.1, linewidth=3)
            
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_zlim(-1.1, 1.1)
            ax.set_title(f"{title} - t = {time_step:.3f}")
            
            return trajectory_line, state_point
        
        anim = animation.FuncAnimation(
            fig, animate, frames=len(state_data.time_evolution),
            interval=1000 / self.config.animation_speed,
            blit=False, repeat=True
        )
        
        return anim
    
    def _visualize_amplitudes(self, state_data: StateVisualizationData, title: str) -> plt.Figure:
        """Visualize state vector amplitudes."""
        
        state_vector = state_data.state_vector
        n_states = len(state_vector)
        n_qubits = int(np.log2(n_states))
        
        # Generate state labels
        state_labels = [format(i, f'0{n_qubits}b') for i in range(n_states)]
        
        # Extract real and imaginary parts
        real_parts = np.real(state_vector)
        imag_parts = np.imag(state_vector)
        
        # Create subplot for amplitudes
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.config.figure_size, 
                                      sharex=True)
        
        # Plot real parts
        bars1 = ax1.bar(range(n_states), real_parts, alpha=0.7, 
                       color=self.colors["qubit"], label='Real')
        ax1.set_ylabel('Real Amplitude')
        ax1.set_title(f'{title} - Real Parts')
        ax1.grid(True, alpha=0.3)
        
        # Plot imaginary parts
        bars2 = ax2.bar(range(n_states), imag_parts, alpha=0.7,
                       color=self.colors["entanglement"], label='Imaginary')
        ax2.set_ylabel('Imaginary Amplitude')
        ax2.set_xlabel('Quantum State')
        ax2.set_title(f'{title} - Imaginary Parts')
        ax2.grid(True, alpha=0.3)
        
        # Set x-axis labels
        ax2.set_xticks(range(n_states))
        ax2.set_xticklabels([f'|{label}⟩' for label in state_labels], rotation=45)
        
        # Add value labels on bars if not too many states
        if n_states <= 8:
            for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
                height1 = bar1.get_height()
                height2 = bar2.get_height()
                if abs(height1) > self.config.state_vector_threshold:
                    ax1.text(bar1.get_x() + bar1.get_width()/2., height1,
                           f'{height1:.3f}', ha='center', va='bottom')
                if abs(height2) > self.config.state_vector_threshold:
                    ax2.text(bar2.get_x() + bar2.get_width()/2., height2,
                           f'{height2:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def _visualize_probabilities(self, state_data: StateVisualizationData, title: str) -> plt.Figure:
        """Visualize measurement probabilities."""
        
        state_vector = state_data.state_vector
        probabilities = np.abs(state_vector) ** 2
        n_states = len(state_vector)
        n_qubits = int(np.log2(n_states))
        
        # Filter out very small probabilities
        significant_probs = [(i, p) for i, p in enumerate(probabilities) 
                           if p > self.config.probability_threshold]
        
        if len(significant_probs) > self.config.max_displayed_states:
            # Show only the largest probabilities
            significant_probs.sort(key=lambda x: x[1], reverse=True)
            significant_probs = significant_probs[:self.config.max_displayed_states]
            significant_probs.sort(key=lambda x: x[0])  # Sort by state index
        
        indices, probs = zip(*significant_probs) if significant_probs else ([], [])
        state_labels = [format(i, f'0{n_qubits}b') for i in indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Create probability bars
        bars = ax.bar(range(len(probs)), probs, alpha=0.8,
                     color=self.colors["measurement"])
        
        # Customize plot
        ax.set_ylabel('Probability')
        ax.set_xlabel('Quantum State')
        ax.set_title(f'{title} - Measurement Probabilities')
        ax.grid(True, alpha=0.3)
        
        # Set x-axis labels
        ax.set_xticks(range(len(probs)))
        ax.set_xticklabels([f'|{label}⟩' for label in state_labels], rotation=45)
        
        # Add probability values on bars
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{prob:.4f}', ha='center', va='bottom')
        
        # Add probability sum annotation
        total_prob = sum(probs)
        ax.text(0.02, 0.98, f'Displayed probability: {total_prob:.4f}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def _visualize_phases(self, state_data: StateVisualizationData, title: str) -> plt.Figure:
        """Visualize quantum state phases."""
        
        state_vector = state_data.state_vector
        amplitudes = np.abs(state_vector)
        phases = np.angle(state_vector)
        n_states = len(state_vector)
        n_qubits = int(np.log2(n_states))
        
        # Filter out states with negligible amplitudes
        significant_states = [(i, amp, phase) for i, (amp, phase) in enumerate(zip(amplitudes, phases))
                            if amp > self.config.state_vector_threshold]
        
        if not significant_states:
            # Create empty plot
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            ax.text(0.5, 0.5, 'No significant amplitudes to display', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{title} - Phases')
            return fig
        
        indices, amps, phases_sig = zip(*significant_states)
        state_labels = [format(i, f'0{n_qubits}b') for i in indices]
        
        # Create polar plot for phases
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.config.figure_size[0] * 1.5, self.config.figure_size[1]),
                                      subplot_kw={'projection': 'polar'})
        
        # Plot phases as polar coordinates
        for i, (amp, phase) in enumerate(zip(amps, phases_sig)):
            ax1.plot([0, phase], [0, amp], 'o-', linewidth=2, markersize=8,
                    label=f'|{state_labels[i]}⟩')
        
        ax1.set_title(f'{title} - Amplitude and Phase')
        ax1.set_ylim(0, max(amps) * 1.1)
        
        # Phase-only visualization
        colors = plt.cm.viridis(np.linspace(0, 1, len(phases_sig)))
        for i, (phase, color) in enumerate(zip(phases_sig, colors)):
            ax2.plot([phase, phase], [0, 1], 'o-', color=color, linewidth=3,
                    markersize=10, label=f'|{state_labels[i]}⟩')
        
        ax2.set_title(f'{title} - Phases Only')
        ax2.set_ylim(0, 1.1)
        
        # Add legends if not too many states
        if len(significant_states) <= 8:
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig
    
    def _visualize_bloch_spheres(self, state_data: StateVisualizationData, title: str) -> plt.Figure:
        """Visualize multiple Bloch spheres for multi-qubit states."""
        
        n_qubits = int(np.log2(len(state_data.state_vector)))
        
        # Create subplot grid for multiple Bloch spheres
        cols = min(n_qubits, 3)
        rows = (n_qubits + cols - 1) // cols
        
        fig = plt.figure(figsize=(cols * 5, rows * 5))
        
        for i in range(n_qubits):
            ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
            
            # Draw Bloch sphere
            self._draw_bloch_sphere_surface(ax)
            self._draw_bloch_sphere_axes(ax)
            
            # Calculate and plot state vector for this qubit
            x, y, z = state_data.calculate_bloch_coordinates(i)
            
            # Draw state vector
            ax.quiver(0, 0, 0, x, y, z, color='red', arrow_length_ratio=0.1, linewidth=3)
            ax.scatter([x], [y], [z], color='red', s=100)
            
            # Customize subplot
            ax.set_title(f'Qubit {i}', fontsize=12)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_zlim(-1.1, 1.1)
            
            # Add coordinate text
            ax.text2D(0.02, 0.98, f'({x:.3f}, {y:.3f}, {z:.3f})',
                     transform=ax.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def _visualize_density_matrix(self, state_data: StateVisualizationData, title: str) -> plt.Figure:
        """Visualize density matrix representation."""
        
        if state_data.density_matrix is not None:
            rho = state_data.density_matrix
        else:
            # Create density matrix from state vector
            state = state_data.state_vector
            rho = np.outer(state, np.conj(state))
        
        n_states = rho.shape[0]
        n_qubits = int(np.log2(n_states))
        
        # Create subplots for real and imaginary parts
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(self.config.figure_size[0] * 1.5, 
                                                                   self.config.figure_size[1] * 1.2))
        
        # Real part
        im1 = ax1.imshow(np.real(rho), cmap='RdBu', vmin=-1, vmax=1)
        ax1.set_title('Real Part')
        ax1.set_xlabel('State Index')
        ax1.set_ylabel('State Index')
        plt.colorbar(im1, ax=ax1)
        
        # Imaginary part
        im2 = ax2.imshow(np.imag(rho), cmap='RdBu', vmin=-1, vmax=1)
        ax2.set_title('Imaginary Part')
        ax2.set_xlabel('State Index')
        ax2.set_ylabel('State Index')
        plt.colorbar(im2, ax=ax2)
        
        # Absolute values
        im3 = ax3.imshow(np.abs(rho), cmap='Blues', vmin=0, vmax=1)
        ax3.set_title('Absolute Values')
        ax3.set_xlabel('State Index')
        ax3.set_ylabel('State Index')
        plt.colorbar(im3, ax=ax3)
        
        # Eigenvalues
        eigenvals, _ = np.linalg.eigh(rho)
        eigenvals = eigenvals[::-1]  # Sort in descending order
        ax4.bar(range(len(eigenvals)), eigenvals, alpha=0.7, color=self.colors["measurement"])
        ax4.set_title('Eigenvalues')
        ax4.set_xlabel('Eigenvalue Index')
        ax4.set_ylabel('Value')
        ax4.grid(True, alpha=0.3)
        
        # Add state labels if small enough
        if n_states <= 8:
            state_labels = [format(i, f'0{n_qubits}b') for i in range(n_states)]
            
            for ax in [ax1, ax2, ax3]:
                ax.set_xticks(range(n_states))
                ax.set_yticks(range(n_states))
                ax.set_xticklabels([f'|{label}⟩' for label in state_labels], rotation=45)
                ax.set_yticklabels([f'⟨{label}|' for label in state_labels])
        
        fig.suptitle(f'{title} - Density Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def _draw_bloch_sphere_surface(self, ax):
        """Draw the surface of a Bloch sphere."""
        
        # Create sphere surface
        u = np.linspace(0, 2 * np.pi, self.config.bloch_sphere_resolution)
        v = np.linspace(0, np.pi, self.config.bloch_sphere_resolution)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Plot surface with transparency
        ax.plot_surface(x, y, z, alpha=0.1, color='lightblue')
        
        # Draw equator and meridians
        theta = np.linspace(0, 2 * np.pi, 100)
        
        # Equator
        ax.plot(np.cos(theta), np.sin(theta), 0, 'k--', alpha=0.3)
        
        # Prime meridian
        phi = np.linspace(0, np.pi, 100)
        ax.plot(np.sin(phi), 0, np.cos(phi), 'k--', alpha=0.3)
        ax.plot(0, np.sin(phi), np.cos(phi), 'k--', alpha=0.3)
    
    def _draw_bloch_sphere_axes(self, ax):
        """Draw coordinate axes for Bloch sphere."""
        
        # Draw coordinate axes
        ax.quiver(0, 0, 0, 1.2, 0, 0, color='red', arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 1.2, 0, color='green', arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 0, 1.2, color='blue', arrow_length_ratio=0.1)
        
        # Add axis labels
        ax.text(1.3, 0, 0, 'X', fontsize=12, fontweight='bold')
        ax.text(0, 1.3, 0, 'Y', fontsize=12, fontweight='bold')
        ax.text(0, 0, 1.3, 'Z', fontsize=12, fontweight='bold')
        
        # Add pole labels
        ax.text(0, 0, 1.1, '|0⟩', fontsize=10, ha='center')
        ax.text(0, 0, -1.1, '|1⟩', fontsize=10, ha='center')


class PerformanceVisualizer:
    """Visualization for quantum algorithm performance analysis."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.colors = self.config.get_color_palette()
        
    def visualize_performance_metrics(self, metrics_data: List['PerformanceMetrics'],
                                    title: str = "Performance Analysis") -> plt.Figure:
        """Create comprehensive performance visualization."""
        
        if not metrics_data:
            raise ValueError("No performance metrics provided")
        
        # Create subplot grid
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Extract data
        execution_times = [m.execution_time for m in metrics_data]
        memory_usage = [m.memory_usage for m in metrics_data]
        gate_counts = [m.gate_count for m in metrics_data]
        circuit_depths = [m.circuit_depth for m in metrics_data]
        error_rates = [m.error_rate for m in metrics_data]
        fidelities = [m.fidelity for m in metrics_data]
        
        # Plot execution times
        axes[0, 0].plot(execution_times, 'o-', color=self.colors["qubit"])
        axes[0, 0].set_title('Execution Time')
        axes[0, 0].set_ylabel('Time (s)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot memory usage
        axes[0, 1].plot(memory_usage, 'o-', color=self.colors["gate"])
        axes[0, 1].set_title('Memory Usage')
        axes[0, 1].set_ylabel('Memory (MB)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot gate count vs circuit depth
        axes[0, 2].scatter(gate_counts, circuit_depths, c=execution_times, 
                          cmap=self.config.performance_colormap, alpha=0.7)
        axes[0, 2].set_title('Circuit Complexity')
        axes[0, 2].set_xlabel('Gate Count')
        axes[0, 2].set_ylabel('Circuit Depth')
        
        # Plot error rates
        axes[1, 0].plot(error_rates, 'o-', color=self.colors["classical"])
        axes[1, 0].set_title('Error Rate')
        axes[1, 0].set_ylabel('Error Rate')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot fidelities
        axes[1, 1].plot(fidelities, 'o-', color=self.colors["entanglement"])
        axes[1, 1].set_title('Fidelity')
        axes[1, 1].set_ylabel('Fidelity')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Performance correlation heatmap
        if len(metrics_data) > 1:
            # Create correlation matrix
            data_matrix = np.array([execution_times, memory_usage, gate_counts, 
                                  circuit_depths, error_rates, fidelities])
            correlation_matrix = np.corrcoef(data_matrix)
            
            im = axes[1, 2].imshow(correlation_matrix, cmap='RdBu', vmin=-1, vmax=1)
            axes[1, 2].set_title('Metric Correlations')
            
            # Add labels
            labels = ['Exec Time', 'Memory', 'Gates', 'Depth', 'Error', 'Fidelity']
            axes[1, 2].set_xticks(range(len(labels)))
            axes[1, 2].set_yticks(range(len(labels)))
            axes[1, 2].set_xticklabels(labels, rotation=45)
            axes[1, 2].set_yticklabels(labels)
            
            # Add correlation values
            for i in range(len(labels)):
                for j in range(len(labels)):
                    text = axes[1, 2].text(j, i, f'{correlation_matrix[i, j]:.2f}',
                                         ha="center", va="center", color="black")
            
            plt.colorbar(im, ax=axes[1, 2])
        else:
            axes[1, 2].text(0.5, 0.5, 'Insufficient data\nfor correlations',
                           ha='center', va='center', transform=axes[1, 2].transAxes)
        
        plt.tight_layout()
        return fig
    
    def create_performance_dashboard(self, metrics_data: List['PerformanceMetrics'],
                                   circuit_data: Optional[CircuitVisualizationData] = None) -> plt.Figure:
        """Create interactive performance dashboard."""
        
        # This would create a comprehensive dashboard with multiple visualizations
        # For now, return a placeholder
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        ax.text(0.5, 0.5, 'Performance Dashboard\n(Interactive features require web backend)',
               ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_title('Quantum Algorithm Performance Dashboard', fontsize=18, fontweight='bold')
        
        return fig


class QuantumAlgorithmVisualizer:
    """Main orchestrator for quantum algorithm visualization."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        
        # Initialize specialized visualizers
        self.circuit_visualizer = CircuitVisualizer(self.config)
        self.state_visualizer = StateVisualizer(self.config)
        self.performance_visualizer = PerformanceVisualizer(self.config)
        
        # Storage for visualization data
        self.circuit_data: Optional[CircuitVisualizationData] = None
        self.state_data: Optional[StateVisualizationData] = None
        self.performance_data: List['PerformanceMetrics'] = []
        
        # Integration with profiler
        if HAS_PROFILER:
            self.profiler: Optional['QuantumPerformanceProfiler'] = None
        
    def visualize_algorithm_execution(self, circuit, 
                                    include_state_evolution: bool = True,
                                    include_performance: bool = True,
                                    title: str = "Quantum Algorithm Execution") -> Dict[str, plt.Figure]:
        """Create comprehensive visualization of algorithm execution."""
        
        figures = {}
        
        # Extract circuit data
        self.circuit_data = self._extract_circuit_data(circuit)
        
        # Create circuit visualization
        figures['circuit'] = self.circuit_visualizer.visualize_circuit(
            self.circuit_data, title=f"{title} - Circuit Diagram"
        )
        
        # Create state evolution visualization if requested
        if include_state_evolution:
            # This would require running the circuit and tracking state evolution
            # For now, create placeholder
            self.state_data = self._extract_state_data(circuit)
            if self.state_data and len(self.state_data.state_vector) > 0:
                figures['state_amplitudes'] = self.state_visualizer.visualize_state_vector(
                    self.state_data, title=f"{title} - State Amplitudes", 
                    visualization_type="amplitudes"
                )
                figures['state_probabilities'] = self.state_visualizer.visualize_state_vector(
                    self.state_data, title=f"{title} - Probabilities",
                    visualization_type="probabilities"
                )
        
        # Create performance visualization if requested and available
        if include_performance and self.performance_data:
            figures['performance'] = self.performance_visualizer.visualize_performance_metrics(
                self.performance_data, title=f"{title} - Performance Analysis"
            )
        
        return figures
    
    def create_comparative_visualization(self, algorithms: List[Any],
                                       algorithm_names: List[str],
                                       title: str = "Algorithm Comparison") -> plt.Figure:
        """Create comparative visualization of multiple algorithms."""
        
        # Extract data for all algorithms
        all_circuit_data = []
        all_performance_data = []
        
        for algorithm in algorithms:
            circuit_data = self._extract_circuit_data(algorithm)
            all_circuit_data.append(circuit_data)
            
            # If performance data is available
            if hasattr(algorithm, 'performance_metrics'):
                all_performance_data.append(algorithm.performance_metrics)
        
        # Create comparison figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Compare circuit properties
        gate_counts = [data.total_gates for data in all_circuit_data]
        circuit_depths = [data.circuit_depth for data in all_circuit_data]
        entangling_gates = [data.entangling_gates for data in all_circuit_data]
        
        # Bar chart of gate counts
        x_pos = np.arange(len(algorithm_names))
        axes[0, 0].bar(x_pos, gate_counts, alpha=0.7, color=self.config.get_color_palette()["gate"])
        axes[0, 0].set_title('Gate Count Comparison')
        axes[0, 0].set_ylabel('Number of Gates')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(algorithm_names, rotation=45)
        
        # Bar chart of circuit depths
        axes[0, 1].bar(x_pos, circuit_depths, alpha=0.7, color=self.config.get_color_palette()["qubit"])
        axes[0, 1].set_title('Circuit Depth Comparison')
        axes[0, 1].set_ylabel('Circuit Depth')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(algorithm_names, rotation=45)
        
        # Scatter plot of complexity
        axes[1, 0].scatter(gate_counts, circuit_depths, s=100, alpha=0.7,
                          c=entangling_gates, cmap='viridis')
        axes[1, 0].set_title('Circuit Complexity')
        axes[1, 0].set_xlabel('Gate Count')
        axes[1, 0].set_ylabel('Circuit Depth')
        
        # Add algorithm labels to scatter plot
        for i, name in enumerate(algorithm_names):
            axes[1, 0].annotate(name, (gate_counts[i], circuit_depths[i]),
                               xytext=(5, 5), textcoords='offset points')
        
        # Performance comparison if available
        if all_performance_data:
            exec_times = [metrics.execution_time for metrics in all_performance_data]
            axes[1, 1].bar(x_pos, exec_times, alpha=0.7, 
                          color=self.config.get_color_palette()["measurement"])
            axes[1, 1].set_title('Execution Time Comparison')
            axes[1, 1].set_ylabel('Execution Time (s)')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(algorithm_names, rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, 'No performance data available',
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        return fig
    
    def export_visualization(self, figure: plt.Figure, filename: str, 
                           format: str = "png", **kwargs):
        """Export visualization to file."""
        
        if format not in self.config.export_formats:
            raise ValueError(f"Unsupported export format: {format}")
        
        # Set quality parameters based on config
        if self.config.export_quality == "high":
            dpi = 300
            bbox_inches = 'tight'
        elif self.config.export_quality == "medium":
            dpi = 150
            bbox_inches = 'tight'
        else:
            dpi = 100
            bbox_inches = None
        
        # Add metadata if requested
        metadata = {}
        if self.config.include_metadata:
            metadata = {
                'Title': 'QuantRS2 Visualization',
                'Author': 'QuantRS2 Visualization System',
                'Creator': 'QuantRS2',
                'CreationDate': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        
        # Export based on format
        if format in ['png', 'jpg', 'jpeg', 'pdf', 'svg']:
            figure.savefig(filename, format=format, dpi=dpi, 
                          bbox_inches=bbox_inches, metadata=metadata, **kwargs)
        elif format == 'html':
            # Convert matplotlib to HTML (simplified)
            import io
            import base64
            
            buffer = io.BytesIO()
            figure.savefig(buffer, format='png', dpi=dpi, bbox_inches=bbox_inches)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>QuantRS2 Visualization</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .visualization {{ text-align: center; }}
                    .metadata {{ margin-top: 20px; font-size: 12px; color: #666; }}
                </style>
            </head>
            <body>
                <div class="visualization">
                    <img src="data:image/png;base64,{image_base64}" alt="Quantum Visualization">
                </div>
                <div class="metadata">
                    Generated by QuantRS2 Visualization System on {time.strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </body>
            </html>
            """
            
            with open(filename, 'w') as f:
                f.write(html_content)
        
        logger.info(f"Visualization exported to {filename}")
    
    def _extract_circuit_data(self, circuit) -> CircuitVisualizationData:
        """Extract visualization data from a quantum circuit."""
        
        circuit_data = CircuitVisualizationData()
        
        # This would integrate with the actual QuantRS2 circuit structure
        # For now, create mock data
        try:
            # Try to extract from QuantRS2 circuit
            if hasattr(circuit, 'gates'):
                for i, gate in enumerate(circuit.gates):
                    gate_type = getattr(gate, 'name', 'UNKNOWN')
                    qubits = getattr(gate, 'qubits', [0])
                    params = getattr(gate, 'params', [])
                    
                    circuit_data.add_gate(gate_type, qubits, params)
            
            elif hasattr(circuit, 'num_qubits'):
                # Mock circuit data
                n_qubits = circuit.num_qubits
                circuit_data.qubits = list(range(n_qubits))
                
                # Add some example gates
                circuit_data.add_gate("H", [0])
                if n_qubits > 1:
                    circuit_data.add_gate("CNOT", [0, 1])
            
        except Exception as e:
            logger.warning(f"Failed to extract circuit data: {e}")
            # Create minimal circuit data
            circuit_data.qubits = [0, 1]
            circuit_data.add_gate("H", [0])
            circuit_data.add_gate("CNOT", [0, 1])
        
        return circuit_data
    
    def _extract_state_data(self, circuit) -> Optional[StateVisualizationData]:
        """Extract quantum state data from circuit execution."""
        
        try:
            # This would integrate with actual QuantRS2 simulation
            if hasattr(circuit, 'run'):
                result = circuit.run()
                if hasattr(result, 'state_vector'):
                    state_data = StateVisualizationData(state_vector=result.state_vector)
                    return state_data
            
            # Create mock state data for demonstration
            # Bell state as example
            state_vector = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
            state_data = StateVisualizationData(state_vector=state_vector)
            
            return state_data
            
        except Exception as e:
            logger.warning(f"Failed to extract state data: {e}")
            return None
    
    def integrate_with_profiler(self, profiler: 'QuantumPerformanceProfiler'):
        """Integrate with quantum performance profiler."""
        
        if not HAS_PROFILER:
            logger.warning("Performance profiler not available")
            return
        
        self.profiler = profiler
        self.performance_data = profiler.all_metrics
        
        # Update circuit visualizer with performance data
        if hasattr(self.circuit_visualizer, 'performance_data'):
            self.circuit_visualizer.performance_data = {
                i: metrics.execution_time 
                for i, metrics in enumerate(self.performance_data)
            }


# Convenience functions for easy usage
def visualize_quantum_circuit(circuit, title: str = "Quantum Circuit",
                            config: Optional[VisualizationConfig] = None) -> plt.Figure:
    """Convenience function to visualize a quantum circuit."""
    
    visualizer = QuantumAlgorithmVisualizer(config)
    circuit_data = visualizer._extract_circuit_data(circuit)
    return visualizer.circuit_visualizer.visualize_circuit(circuit_data, title)


def visualize_quantum_state(state_vector: np.ndarray, 
                           visualization_type: str = "amplitudes",
                           title: str = "Quantum State",
                           config: Optional[VisualizationConfig] = None) -> plt.Figure:
    """Convenience function to visualize a quantum state."""
    
    visualizer = QuantumAlgorithmVisualizer(config)
    state_data = StateVisualizationData(state_vector=state_vector)
    return visualizer.state_visualizer.visualize_state_vector(state_data, title, visualization_type)


def create_bloch_sphere_visualization(qubit_state: np.ndarray,
                                    title: str = "Bloch Sphere",
                                    config: Optional[VisualizationConfig] = None) -> plt.Figure:
    """Convenience function to create Bloch sphere visualization."""
    
    if len(qubit_state) != 2:
        raise ValueError("Bloch sphere visualization requires a 2-dimensional state vector")
    
    visualizer = QuantumAlgorithmVisualizer(config)
    state_data = StateVisualizationData(state_vector=qubit_state)
    return visualizer.state_visualizer.create_bloch_sphere(0, state_data, title)


def compare_quantum_algorithms(algorithms: List[Any], 
                             algorithm_names: List[str],
                             title: str = "Algorithm Comparison",
                             config: Optional[VisualizationConfig] = None) -> plt.Figure:
    """Convenience function to compare quantum algorithms."""
    
    visualizer = QuantumAlgorithmVisualizer(config)
    return visualizer.create_comparative_visualization(algorithms, algorithm_names, title)


# GUI and Interactive Components (if available)
if HAS_TKINTER:
    
    class VisualizationGUI:
        """Tkinter-based GUI for quantum algorithm visualization."""
        
        def __init__(self, config: Optional[VisualizationConfig] = None):
            self.config = config or VisualizationConfig()
            self.visualizer = QuantumAlgorithmVisualizer(self.config)
            
            self.root = tk.Tk()
            self.root.title("QuantRS2 Visualization Suite")
            self.root.geometry("800x600")
            
            self.create_interface()
            
        def create_interface(self):
            """Create the GUI interface."""
            
            # Main menu
            menubar = tk.Menu(self.root)
            self.root.config(menu=menubar)
            
            # File menu
            file_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="File", menu=file_menu)
            file_menu.add_command(label="Load Circuit", command=self.load_circuit)
            file_menu.add_command(label="Export Visualization", command=self.export_visualization)
            file_menu.add_separator()
            file_menu.add_command(label="Exit", command=self.root.quit)
            
            # Create main frames
            self.create_control_frame()
            self.create_visualization_frame()
            
        def create_control_frame(self):
            """Create control panel."""
            
            control_frame = ttk.Frame(self.root)
            control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
            
            # Visualization type selection
            ttk.Label(control_frame, text="Visualization Type:").pack(anchor=tk.W)
            self.viz_type = tk.StringVar(value="circuit")
            ttk.Radiobutton(control_frame, text="Circuit Diagram", 
                           variable=self.viz_type, value="circuit").pack(anchor=tk.W)
            ttk.Radiobutton(control_frame, text="State Amplitudes", 
                           variable=self.viz_type, value="amplitudes").pack(anchor=tk.W)
            ttk.Radiobutton(control_frame, text="State Probabilities", 
                           variable=self.viz_type, value="probabilities").pack(anchor=tk.W)
            ttk.Radiobutton(control_frame, text="Bloch Sphere", 
                           variable=self.viz_type, value="bloch").pack(anchor=tk.W)
            
            ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
            
            # Configuration options
            ttk.Label(control_frame, text="Configuration:").pack(anchor=tk.W)
            
            self.show_performance = tk.BooleanVar(value=True)
            ttk.Checkbutton(control_frame, text="Show Performance Data", 
                           variable=self.show_performance).pack(anchor=tk.W)
            
            self.enable_animation = tk.BooleanVar(value=False)
            ttk.Checkbutton(control_frame, text="Enable Animation", 
                           variable=self.enable_animation).pack(anchor=tk.W)
            
            ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
            
            # Action buttons
            ttk.Button(control_frame, text="Generate Visualization", 
                      command=self.generate_visualization).pack(fill=tk.X, pady=2)
            ttk.Button(control_frame, text="Refresh", 
                      command=self.refresh_visualization).pack(fill=tk.X, pady=2)
            ttk.Button(control_frame, text="Save Image", 
                      command=self.save_image).pack(fill=tk.X, pady=2)
            
        def create_visualization_frame(self):
            """Create visualization display area."""
            
            viz_frame = ttk.Frame(self.root)
            viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Matplotlib canvas
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
            self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            toolbar_frame = ttk.Frame(viz_frame)
            toolbar_frame.pack(fill=tk.X)
            
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
            toolbar.update()
        
        def load_circuit(self):
            """Load quantum circuit from file."""
            
            filename = filedialog.askopenfilename(
                title="Load Quantum Circuit",
                filetypes=[("Python files", "*.py"), ("All files", "*.*")]
            )
            
            if filename:
                try:
                    # This would load actual circuit data
                    messagebox.showinfo("Load Circuit", f"Circuit loaded from {filename}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load circuit: {e}")
        
        def generate_visualization(self):
            """Generate visualization based on current settings."""
            
            try:
                viz_type = self.viz_type.get()
                
                # Clear previous plot
                self.ax.clear()
                
                if viz_type == "circuit":
                    # Generate circuit visualization
                    circuit_data = CircuitVisualizationData()
                    circuit_data.qubits = [0, 1]
                    circuit_data.add_gate("H", [0])
                    circuit_data.add_gate("CNOT", [0, 1])
                    
                    self.visualizer.circuit_visualizer.visualize_circuit(circuit_data)
                
                elif viz_type in ["amplitudes", "probabilities"]:
                    # Generate state visualization
                    state_vector = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
                    state_data = StateVisualizationData(state_vector=state_vector)
                    self.visualizer.state_visualizer.visualize_state_vector(
                        state_data, visualization_type=viz_type
                    )
                
                elif viz_type == "bloch":
                    # Generate Bloch sphere
                    qubit_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
                    state_data = StateVisualizationData(state_vector=qubit_state)
                    self.visualizer.state_visualizer.create_bloch_sphere(0, state_data)
                
                self.canvas.draw()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate visualization: {e}")
        
        def refresh_visualization(self):
            """Refresh the current visualization."""
            self.generate_visualization()
        
        def save_image(self):
            """Save current visualization as image."""
            
            filename = filedialog.asksaveasfilename(
                title="Save Visualization",
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), 
                          ("SVG files", "*.svg"), ("All files", "*.*")]
            )
            
            if filename:
                try:
                    self.visualizer.export_visualization(self.fig, filename)
                    messagebox.showinfo("Save Image", f"Visualization saved to {filename}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save image: {e}")
        
        def export_visualization(self):
            """Export visualization with options."""
            # This would open an export dialog with format options
            self.save_image()
        
        def run(self):
            """Start the GUI application."""
            self.root.mainloop()


# Web-based Visualization (if Dash is available)
if HAS_DASH:
    
    def create_quantum_visualization_app(config: Optional[VisualizationConfig] = None):
        """Create a Dash web application for quantum visualization."""
        
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        visualizer = QuantumAlgorithmVisualizer(config)
        
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("QuantRS2 Visualization Suite", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Visualization Controls"),
                        dbc.CardBody([
                            html.Label("Visualization Type:"),
                            dcc.Dropdown(
                                id='viz-type-dropdown',
                                options=[
                                    {'label': 'Circuit Diagram', 'value': 'circuit'},
                                    {'label': 'State Amplitudes', 'value': 'amplitudes'},
                                    {'label': 'State Probabilities', 'value': 'probabilities'},
                                    {'label': 'Bloch Sphere', 'value': 'bloch'},
                                    {'label': 'Performance Analysis', 'value': 'performance'}
                                ],
                                value='circuit'
                            ),
                            html.Br(),
                            
                            dbc.Checklist(
                                id='viz-options',
                                options=[
                                    {'label': 'Show Performance Data', 'value': 'performance'},
                                    {'label': 'Enable Interactivity', 'value': 'interactive'},
                                    {'label': 'High Quality Export', 'value': 'high_quality'}
                                ],
                                value=['performance', 'interactive']
                            ),
                            html.Br(),
                            
                            dbc.Button("Generate Visualization", id="generate-btn", 
                                     color="primary", className="mb-2"),
                            dbc.Button("Export Image", id="export-btn", 
                                     color="secondary", className="mb-2")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Visualization Display"),
                        dbc.CardBody([
                            dcc.Graph(id='main-visualization')
                        ])
                    ])
                ], width=9)
            ])
        ], fluid=True)
        
        @app.callback(
            Output('main-visualization', 'figure'),
            [Input('generate-btn', 'n_clicks')],
            [State('viz-type-dropdown', 'value'),
             State('viz-options', 'value')]
        )
        def update_visualization(n_clicks, viz_type, options):
            if n_clicks is None:
                return {}
            
            # Generate visualization based on type
            if viz_type == 'circuit':
                return create_plotly_circuit_diagram()
            elif viz_type == 'amplitudes':
                return create_plotly_state_amplitudes()
            elif viz_type == 'probabilities':
                return create_plotly_state_probabilities()
            elif viz_type == 'bloch':
                return create_plotly_bloch_sphere()
            elif viz_type == 'performance':
                return create_plotly_performance_chart()
            
            return {}
        
        def create_plotly_circuit_diagram():
            """Create Plotly circuit diagram."""
            # Simplified circuit diagram using Plotly
            fig = go.Figure()
            
            # Add qubit lines
            fig.add_trace(go.Scatter(x=[0, 4], y=[0, 0], mode='lines', 
                                   name='Qubit 0', line=dict(color='blue', width=3)))
            fig.add_trace(go.Scatter(x=[0, 4], y=[1, 1], mode='lines',
                                   name='Qubit 1', line=dict(color='blue', width=3)))
            
            # Add gates (simplified as rectangles using shapes)
            fig.add_shape(type="rect", x0=0.8, y0=-0.2, x1=1.2, y1=0.2,
                         fillcolor="red", line=dict(color="black"))
            fig.add_shape(type="rect", x0=2.8, y0=-0.2, x1=3.2, y1=1.2,
                         fillcolor="green", line=dict(color="black"))
            
            # Add gate labels
            fig.add_annotation(x=1, y=0, text="H", showarrow=False, font=dict(color="white", size=14))
            fig.add_annotation(x=3, y=0.5, text="CNOT", showarrow=False, font=dict(color="white", size=12))
            
            fig.update_layout(
                title="Quantum Circuit Diagram",
                xaxis_title="Circuit Depth",
                yaxis_title="Qubits",
                showlegend=False,
                height=400
            )
            
            return fig
        
        def create_plotly_state_amplitudes():
            """Create Plotly state amplitudes visualization."""
            # Bell state amplitudes
            states = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
            real_parts = [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]
            imag_parts = [0, 0, 0, 0]
            
            fig = make_subplots(rows=2, cols=1, subplot_titles=('Real Parts', 'Imaginary Parts'))
            
            fig.add_trace(go.Bar(x=states, y=real_parts, name='Real', marker_color='blue'),
                         row=1, col=1)
            fig.add_trace(go.Bar(x=states, y=imag_parts, name='Imaginary', marker_color='red'),
                         row=2, col=1)
            
            fig.update_layout(title="State Vector Amplitudes", height=600)
            return fig
        
        def create_plotly_state_probabilities():
            """Create Plotly state probabilities visualization."""
            states = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
            probabilities = [0.5, 0.0, 0.0, 0.5]
            
            fig = go.Figure(data=[go.Bar(x=states, y=probabilities, marker_color='orange')])
            fig.update_layout(
                title="Measurement Probabilities",
                xaxis_title="Quantum State",
                yaxis_title="Probability",
                height=400
            )
            return fig
        
        def create_plotly_bloch_sphere():
            """Create Plotly 3D Bloch sphere visualization."""
            # Create sphere surface
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x_sphere = np.outer(np.cos(u), np.sin(v))
            y_sphere = np.outer(np.sin(u), np.sin(v))
            z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
            
            fig = go.Figure()
            
            # Add sphere surface
            fig.add_trace(go.Surface(x=x_sphere, y=y_sphere, z=z_sphere,
                                   opacity=0.3, colorscale='Blues', showscale=False))
            
            # Add state vector (example: |+⟩ state)
            fig.add_trace(go.Scatter3d(x=[0, 1], y=[0, 0], z=[0, 0],
                                     mode='lines+markers', line=dict(color='red', width=8),
                                     marker=dict(size=8, color='red'), name='State Vector'))
            
            # Add coordinate axes
            fig.add_trace(go.Scatter3d(x=[-1.2, 1.2], y=[0, 0], z=[0, 0],
                                     mode='lines', line=dict(color='black', width=4),
                                     showlegend=False))
            fig.add_trace(go.Scatter3d(x=[0, 0], y=[-1.2, 1.2], z=[0, 0],
                                     mode='lines', line=dict(color='black', width=4),
                                     showlegend=False))
            fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-1.2, 1.2],
                                     mode='lines', line=dict(color='black', width=4),
                                     showlegend=False))
            
            fig.update_layout(
                title="Bloch Sphere Visualization",
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    aspectmode='cube'
                ),
                height=600
            )
            return fig
        
        def create_plotly_performance_chart():
            """Create Plotly performance analysis chart."""
            # Mock performance data
            x = list(range(1, 11))
            execution_times = [0.1 + 0.02 * i + np.random.normal(0, 0.01) for i in x]
            memory_usage = [10 + 2 * i + np.random.normal(0, 1) for i in x]
            
            fig = make_subplots(rows=2, cols=1, subplot_titles=('Execution Time', 'Memory Usage'))
            
            fig.add_trace(go.Scatter(x=x, y=execution_times, mode='lines+markers',
                                   name='Execution Time', line=dict(color='blue')),
                         row=1, col=1)
            fig.add_trace(go.Scatter(x=x, y=memory_usage, mode='lines+markers',
                                   name='Memory Usage', line=dict(color='green')),
                         row=2, col=1)
            
            fig.update_layout(title="Performance Analysis", height=600)
            return fig
        
        return app


# CLI interface for visualization
def main():
    """CLI interface for quantum algorithm visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="QuantRS2 Quantum Algorithm Visualizer")
    parser.add_argument("--mode", choices=["gui", "web", "export"], default="gui",
                       help="Visualization mode")
    parser.add_argument("--circuit", help="Path to quantum circuit file")
    parser.add_argument("--output", help="Output file for export mode")
    parser.add_argument("--format", choices=["png", "pdf", "svg", "html"], default="png",
                       help="Export format")
    parser.add_argument("--type", choices=["circuit", "amplitudes", "probabilities", "bloch"],
                       default="circuit", help="Visualization type")
    parser.add_argument("--config", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = VisualizationConfig()
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config_data = json.load(f)
            # Update config with loaded data
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    if args.mode == "gui":
        if HAS_TKINTER:
            gui = VisualizationGUI(config)
            gui.run()
        else:
            print("GUI mode requires tkinter. Please install tkinter or use web mode.")
    
    elif args.mode == "web":
        if HAS_DASH:
            app = create_quantum_visualization_app(config)
            app.run_server(debug=True)
        else:
            print("Web mode requires Dash. Please install dash or use GUI mode.")
    
    elif args.mode == "export":
        if not args.output:
            print("Export mode requires --output argument")
            return
        
        # Create visualization and export
        visualizer = QuantumAlgorithmVisualizer(config)
        
        # Create mock circuit for demonstration
        circuit_data = CircuitVisualizationData()
        circuit_data.qubits = [0, 1]
        circuit_data.add_gate("H", [0])
        circuit_data.add_gate("CNOT", [0, 1])
        
        if args.type == "circuit":
            fig = visualizer.circuit_visualizer.visualize_circuit(circuit_data)
        elif args.type in ["amplitudes", "probabilities"]:
            state_vector = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
            state_data = StateVisualizationData(state_vector=state_vector)
            fig = visualizer.state_visualizer.visualize_state_vector(state_data, 
                                                                   visualization_type=args.type)
        elif args.type == "bloch":
            qubit_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
            state_data = StateVisualizationData(state_vector=qubit_state)
            fig = visualizer.state_visualizer.create_bloch_sphere(0, state_data)
        
        visualizer.export_visualization(fig, args.output, args.format)
        print(f"Visualization exported to {args.output}")


if __name__ == "__main__":
    main()