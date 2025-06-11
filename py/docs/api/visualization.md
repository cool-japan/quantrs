# Visualization API Reference

The QuantRS2 visualization module provides comprehensive tools for visualizing quantum circuits, states, and algorithm performance.

## Quantum Algorithm Visualization

::: quantrs2.quantum_algorithm_visualization
    options:
      members:
        - QuantumAlgorithmVisualizer
        - CircuitVisualizer
        - StateVisualizer
        - PerformanceVisualizer
        - VisualizationConfig
        - visualize_quantum_circuit
        - visualize_quantum_state
        - create_bloch_sphere_visualization
        - compare_quantum_algorithms

### QuantumAlgorithmVisualizer

Main orchestrator for quantum algorithm visualization with comprehensive workflows.

#### Methods

- `visualize_circuit(circuit, config)`: Create circuit diagram visualization
- `visualize_state(state, config)`: Visualize quantum state
- `create_performance_dashboard(data)`: Build performance analytics dashboard
- `export_visualization(format, filename)`: Export in multiple formats
- `start_interactive_session()`: Launch interactive visualization

#### Usage Example

```python
from quantrs2.quantum_algorithm_visualization import (
    QuantumAlgorithmVisualizer, VisualizationConfig
)
from quantrs2 import Circuit

# Create a quantum circuit
circuit = Circuit(3)
circuit.h(0)
circuit.cnot(0, 1)
circuit.cnot(1, 2)

# Configure visualization
config = VisualizationConfig(
    style="modern",
    color_scheme="quantum",
    show_measurements=True,
    show_barriers=True,
    output_format="png"
)

# Create visualizer and generate circuit diagram
visualizer = QuantumAlgorithmVisualizer()
fig = visualizer.visualize_circuit(circuit, config)

# Export the visualization
visualizer.export_visualization("png", "bell_circuit.png")
```

### CircuitVisualizer

Specialized visualizer for quantum circuits with performance integration.

#### Methods

- `draw_circuit(circuit, ax)`: Draw circuit on matplotlib axes
- `add_performance_overlay(profiling_data)`: Add performance metrics
- `highlight_critical_path()`: Highlight performance bottlenecks
- `animate_execution()`: Create execution animation
- `export_svg()`: Export as scalable vector graphics

#### Configuration Options

- `gate_style`: Gate appearance ("box", "circle", "traditional")
- `wire_style`: Wire appearance ("solid", "dashed", "quantum")
- `color_mapping`: Custom color scheme for gates
- `font_size`: Text size for labels
- `spacing`: Gate and wire spacing

### StateVisualizer

3D quantum state visualization with Bloch sphere animations.

#### Methods

- `plot_bloch_sphere(state)`: Create 3D Bloch sphere visualization
- `animate_state_evolution(states)`: Animate state changes over time
- `plot_density_matrix(rho)`: Visualize density matrix
- `plot_amplitude_distribution(state)`: Show amplitude probabilities
- `create_entanglement_plot(state)`: Visualize entanglement structure

#### Usage Example

```python
from quantrs2.quantum_algorithm_visualization import StateVisualizer
import numpy as np

# Create superposition state
state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])  # |00⟩ + |11⟩

# Visualize state
state_viz = StateVisualizer()
fig = state_viz.plot_amplitude_distribution(state)
fig.show()

# Create Bloch sphere for single qubit
single_qubit_state = np.array([1/np.sqrt(2), 1j/np.sqrt(2)])
bloch_fig = state_viz.plot_bloch_sphere(single_qubit_state)
```

### PerformanceVisualizer

Analytics visualization for quantum algorithm performance.

#### Methods

- `create_performance_dashboard(metrics)`: Comprehensive analytics dashboard
- `plot_execution_timeline(trace_data)`: Show execution timeline
- `plot_resource_usage(memory_data)`: Memory and resource visualization
- `plot_convergence_curves(optimization_data)`: Show algorithm convergence
- `create_comparative_analysis(algorithms)`: Compare multiple algorithms

#### Dashboard Features

- Real-time performance metrics
- Interactive charts and graphs
- Resource utilization tracking
- Error rate analysis
- Scaling behavior visualization

### VisualizationConfig

Configuration dataclass for visualization settings.

#### Attributes

- `style`: Visual style ("modern", "classic", "minimal")
- `color_scheme`: Color palette ("quantum", "classic", "colorblind")
- `dpi`: Output resolution
- `figsize`: Figure dimensions
- `animation_speed`: Animation frame rate
- `export_format`: Output format ("png", "pdf", "svg", "html")

## Core Visualization Functions

### Circuit Visualization

```python
def visualize_quantum_circuit(
    circuit, 
    style="modern",
    show_measurements=True,
    output_format="png"
):
    """
    Create a visual representation of a quantum circuit.
    
    Args:
        circuit: Quantum circuit to visualize
        style: Visual style for the diagram
        show_measurements: Whether to show measurement operations
        output_format: Export format
    
    Returns:
        matplotlib Figure object
    """
```

### State Visualization

```python
def visualize_quantum_state(
    state,
    representation="bloch",
    interactive=False
):
    """
    Visualize a quantum state in various representations.
    
    Args:
        state: Quantum state vector or density matrix
        representation: Visualization type ("bloch", "bar", "phase")
        interactive: Enable interactive controls
    
    Returns:
        Visualization figure
    """
```

### Bloch Sphere Creation

```python
def create_bloch_sphere_visualization(
    states, 
    animate=False,
    show_trajectory=True
):
    """
    Create 3D Bloch sphere visualization.
    
    Args:
        states: Single state or list of states for animation
        animate: Whether to create animation
        show_trajectory: Show state evolution path
    
    Returns:
        3D Bloch sphere figure
    """
```

### Algorithm Comparison

```python
def compare_quantum_algorithms(
    algorithms,
    metrics=["execution_time", "fidelity", "resource_usage"],
    visualization_type="dashboard"
):
    """
    Compare performance of multiple quantum algorithms.
    
    Args:
        algorithms: List of algorithm results to compare
        metrics: Performance metrics to analyze
        visualization_type: Type of comparison visualization
    
    Returns:
        Comparative analysis dashboard
    """
```

## GUI and Web Interfaces

### Tkinter GUI Interface

For desktop applications with interactive controls:

```python
from quantrs2.quantum_algorithm_visualization import start_gui_interface

# Launch GUI application
gui = start_gui_interface()
gui.load_circuit(circuit)
gui.start_visualization()
```

### Web-based Dashboard

Using Dash for browser-based visualization:

```python
from quantrs2.quantum_algorithm_visualization import start_web_dashboard

# Start web server
dashboard = start_web_dashboard(port=8050)
dashboard.add_circuit(circuit)
dashboard.run(debug=True)
```

## Advanced Visualization Features

### Performance Integration

The visualization module integrates with the performance profiling system:

```python
# Combine circuit visualization with performance data
from quantrs2.profiler import CircuitProfiler

profiler = CircuitProfiler()
profile_data = profiler.profile_circuit(circuit)

visualizer.add_performance_overlay(profile_data)
```

### Real-time Monitoring

For live algorithm execution monitoring:

```python
# Create real-time dashboard
dashboard = visualizer.create_real_time_dashboard()
dashboard.monitor_algorithm(vqe_instance)
dashboard.start_streaming()
```

### Export Capabilities

Multiple export formats supported:

- **PNG/JPEG**: High-quality raster images
- **PDF**: Publication-ready vector graphics
- **SVG**: Scalable web graphics
- **HTML**: Interactive web visualizations
- **MP4**: Animation exports

## Customization Options

### Themes and Styling

```python
# Custom theme configuration
theme_config = {
    "gate_colors": {
        "H": "#FF6B6B",
        "CNOT": "#4ECDC4",
        "X": "#45B7D1"
    },
    "wire_color": "#2C3E50",
    "background": "#FFFFFF",
    "text_color": "#2C3E50"
}

visualizer.apply_theme(theme_config)
```

### Layout Customization

```python
# Circuit layout options
layout_config = {
    "gate_spacing": 1.5,
    "wire_spacing": 1.0,
    "margin": 0.5,
    "aspect_ratio": "auto"
}

circuit_viz.configure_layout(layout_config)
```

## Integration with Other Modules

### With Profiling System

```python
from quantrs2.profiler import CircuitProfiler

# Integrated profiling and visualization
profiler = CircuitProfiler()
profile_data = profiler.profile_circuit(circuit)
visualizer.visualize_with_profiling(circuit, profile_data)
```

### With Testing Framework

```python
from quantrs2.quantum_testing_tools import TestSuite

# Visualize test results
test_suite = TestSuite()
results = test_suite.run_tests()
visualizer.create_test_report_dashboard(results)
```

## Error Handling

Visualization-specific exceptions:

- `VisualizationError`: Base visualization exception
- `RenderingError`: Rendering and display errors
- `ExportError`: File export failures
- `ConfigurationError`: Invalid configuration settings

## Performance Considerations

- Large circuits may require simplified rendering
- Animations can be memory-intensive
- Web dashboards support lazy loading
- SVG exports are resolution-independent
- Use matplotlib backends appropriate for your environment

## Dependencies

Optional dependencies for enhanced functionality:

- `plotly`: Interactive web visualizations
- `dash`: Web dashboard framework
- `tkinter`: Desktop GUI (usually included with Python)
- `PIL/Pillow`: Enhanced image processing
- `ffmpeg`: Video export support

## See Also

- [Core Module](core.md) for basic circuit operations
- [Performance Profiling](../user-guide/performance.md) for optimization
- [Testing Tools](testing.md) for validation workflows
- [Development Tools](dev-tools.md) for IDE integration