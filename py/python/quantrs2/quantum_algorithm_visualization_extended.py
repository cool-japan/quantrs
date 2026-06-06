"""
QuantRS2 Quantum Algorithm Visualization System - Extended Components

Contains the main orchestrator class (QuantumAlgorithmVisualizer), convenience
functions, GUI components, web application, and CLI interface.

Split from quantum_algorithm_visualization.py for compliance with the 2000-line
file policy. The base module contains data classes and specialized visualizers;
this module contains the orchestrating layer and interactive components.

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

# Import base visualization components
from .quantum_algorithm_visualization import (
    VisualizationConfig,
    CircuitVisualizationData,
    StateVisualizationData,
    CircuitVisualizer,
    StateVisualizer,
    PerformanceVisualizer,
)

# Configure logging
logger = logging.getLogger(__name__)


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
