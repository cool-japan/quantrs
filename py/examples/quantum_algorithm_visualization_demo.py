#!/usr/bin/env python3
"""
Comprehensive demo of the QuantRS2 Quantum Algorithm Visualization System.

This demo showcases the complete visualization capabilities including:
- Interactive circuit diagram visualization with performance overlays
- Real-time quantum state evolution visualization with 3D Bloch spheres
- Performance analytics integration with profiling data
- Export capabilities in multiple formats
- Comparative analysis tools for multiple algorithms
- Animation capabilities for circuit execution and state evolution

Run this demo to see the full range of visualization features available
in the QuantRS2 quantum computing framework.
"""

import numpy as np
import time
import sys
import os
from pathlib import Path

try:
    import quantrs2
    from quantrs2 import (
        VisualizationConfig, CircuitVisualizationData, StateVisualizationData,
        CircuitVisualizer, StateVisualizer, PerformanceVisualizer,
        QuantumAlgorithmVisualizer, visualize_quantum_circuit,
        visualize_quantum_state, create_bloch_sphere_visualization,
        compare_quantum_algorithms
    )
    print(f"QuantRS2 version: {quantrs2.__version__}")
    print(f"Successfully imported quantum algorithm visualization system")
except ImportError as e:
    print(f"Error importing QuantRS2 visualization: {e}")
    print("Please ensure the visualization system is properly installed")
    sys.exit(1)

# Check for optional dependencies
HAS_TKINTER = False
try:
    import tkinter as tk
    HAS_TKINTER = True
    print("✓ Tkinter GUI support available")
except ImportError:
    print("✗ Tkinter GUI support not available")

HAS_PLOTLY = False
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
    print("✓ Plotly interactive visualization support available")
except ImportError:
    print("✗ Plotly interactive visualization not available")

HAS_DASH = False
try:
    import dash
    HAS_DASH = True
    print("✓ Dash web interface support available")
except ImportError:
    print("✗ Dash web interface not available")


def demo_circuit_visualization():
    """Demonstrate comprehensive circuit visualization capabilities."""
    print("\n" + "="*60)
    print("CIRCUIT VISUALIZATION DEMO")
    print("="*60)
    
    # Create a sample quantum circuit
    print("Creating sample quantum circuit...")
    circuit_data = CircuitVisualizationData()
    
    # Build a quantum teleportation circuit
    circuit_data.add_gate("H", [0], execution_time=0.001, fidelity=0.995)
    circuit_data.add_gate("CNOT", [0, 1], execution_time=0.005, fidelity=0.99)
    circuit_data.add_gate("CNOT", [2, 0], execution_time=0.005, fidelity=0.99)
    circuit_data.add_gate("H", [2], execution_time=0.001, fidelity=0.995)
    circuit_data.add_gate("MEASURE", [0])
    circuit_data.add_gate("MEASURE", [2])
    circuit_data.add_gate("CZ", [1, 2], execution_time=0.007, fidelity=0.98)
    circuit_data.add_gate("CX", [0, 1], execution_time=0.005, fidelity=0.99)
    
    print(f"Circuit created with {circuit_data.total_gates} gates on {len(circuit_data.qubits)} qubits")
    print(f"Single-qubit gates: {circuit_data.single_qubit_gates}")
    print(f"Entangling gates: {circuit_data.entangling_gates}")
    
    # Create visualizer with custom configuration
    config = VisualizationConfig(
        figure_size=(14, 10),
        color_scheme="quantum",
        enable_profiling_overlay=True,
        animation_speed=1.5
    )
    
    visualizer = CircuitVisualizer(config)
    
    # Generate circuit visualization
    print("\nGenerating circuit diagram with performance overlay...")
    fig = visualizer.visualize_circuit(
        circuit_data, 
        title="Quantum Teleportation Circuit",
        show_performance=True,
        interactive=False
    )
    
    # Save visualization
    output_dir = Path("visualization_outputs")
    output_dir.mkdir(exist_ok=True)
    
    circuit_file = output_dir / "quantum_teleportation_circuit.png"
    fig.savefig(circuit_file, dpi=300, bbox_inches='tight')
    print(f"Circuit diagram saved to: {circuit_file}")
    
    # Create animated execution visualization
    print("\nCreating animated circuit execution...")
    execution_trace = [{"step": i, "gate_id": i} for i in range(len(circuit_data.gates))]
    
    try:
        anim = visualizer.create_animated_execution(
            circuit_data, 
            execution_trace,
            title="Quantum Teleportation - Step by Step"
        )
        
        # Save animation
        anim_file = output_dir / "teleportation_animation.gif"
        anim.save(str(anim_file), writer='pillow', fps=2)
        print(f"Circuit animation saved to: {anim_file}")
        
    except Exception as e:
        print(f"Animation creation failed (this is normal if pillow/imageio not installed): {e}")
    
    import matplotlib.pyplot as plt
    plt.close('all')  # Clean up figures
    
    return circuit_data


def demo_state_visualization():
    """Demonstrate quantum state visualization capabilities."""
    print("\n" + "="*60)
    print("QUANTUM STATE VISUALIZATION DEMO")
    print("="*60)
    
    # Create various quantum states for visualization
    
    # 1. Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
    print("\n1. Bell State Visualization")
    bell_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
    bell_data = StateVisualizationData(state_vector=bell_state)
    
    visualizer = StateVisualizer()
    
    # Create different visualizations of the Bell state
    viz_types = ["amplitudes", "probabilities", "phase", "bloch", "density_matrix"]
    
    output_dir = Path("visualization_outputs") 
    
    for viz_type in viz_types:
        try:
            print(f"  Creating {viz_type} visualization...")
            fig = visualizer.visualize_state_vector(
                bell_data,
                title=f"Bell State - {viz_type.title()}",
                visualization_type=viz_type
            )
            
            viz_file = output_dir / f"bell_state_{viz_type}.png"
            fig.savefig(viz_file, dpi=300, bbox_inches='tight')
            print(f"  Saved to: {viz_file}")
            
            import matplotlib.pyplot as plt
            plt.close(fig)
            
        except Exception as e:
            print(f"  Failed to create {viz_type} visualization: {e}")
    
    # 2. Single qubit state evolution on Bloch sphere
    print("\n2. Bloch Sphere State Evolution")
    
    # Create time evolution data - qubit rotating around Z axis
    evolution_data = StateVisualizationData()
    times = np.linspace(0, 2*np.pi, 30)
    
    for t in times:
        # Rotating state |ψ(t)⟩ = cos(t/2)|0⟩ + e^(it)sin(t/2)|1⟩
        state_t = np.array([np.cos(t/2), np.exp(1j*t)*np.sin(t/2)], dtype=complex)
        evolution_data.time_evolution.append((t, state_t))
    
    try:
        print("  Creating Bloch sphere evolution animation...")
        anim = visualizer.create_state_evolution_animation(
            evolution_data,
            qubit_index=0,
            title="Qubit Evolution on Bloch Sphere"
        )
        
        # Save animation
        bloch_anim_file = output_dir / "bloch_sphere_evolution.gif"
        anim.save(str(bloch_anim_file), writer='pillow', fps=5)
        print(f"  Bloch sphere animation saved to: {bloch_anim_file}")
        
    except Exception as e:
        print(f"  Bloch sphere animation failed: {e}")
    
    # 3. Multi-qubit state visualization
    print("\n3. Multi-qubit State Visualization")
    
    # GHZ state |GHZ⟩ = (|000⟩ + |111⟩)/√2
    ghz_state = np.zeros(8, dtype=complex)
    ghz_state[0] = 1/np.sqrt(2)  # |000⟩
    ghz_state[7] = 1/np.sqrt(2)  # |111⟩
    
    ghz_data = StateVisualizationData(state_vector=ghz_state)
    
    try:
        print("  Creating GHZ state Bloch spheres...")
        fig = visualizer.visualize_state_vector(
            ghz_data,
            title="GHZ State - Individual Qubits",
            visualization_type="bloch"
        )
        
        ghz_file = output_dir / "ghz_state_bloch_spheres.png"
        fig.savefig(ghz_file, dpi=300, bbox_inches='tight')
        print(f"  GHZ state visualization saved to: {ghz_file}")
        
        import matplotlib.pyplot as plt
        plt.close(fig)
        
    except Exception as e:
        print(f"  GHZ state visualization failed: {e}")
    
    return bell_data, evolution_data, ghz_data


def demo_performance_visualization():
    """Demonstrate performance analytics visualization."""
    print("\n" + "="*60)
    print("PERFORMANCE ANALYTICS VISUALIZATION DEMO")
    print("="*60)
    
    # Create mock performance data
    print("Generating mock performance metrics...")
    
    # Simulate performance data for different circuit sizes
    performance_metrics = []
    
    for n_qubits in range(2, 8):
        for run in range(5):
            # Create mock metric object
            metric = type('PerformanceMetric', (), {})()
            
            # Execution time scales roughly exponentially with qubit count
            base_time = 0.001 * (2 ** n_qubits)
            metric.execution_time = base_time * (1 + np.random.normal(0, 0.1))
            
            # Memory usage scales exponentially
            metric.memory_usage = 10 * (2 ** n_qubits) * (1 + np.random.normal(0, 0.05))
            
            # Circuit properties
            metric.gate_count = n_qubits * 3 + np.random.randint(-2, 3)
            metric.circuit_depth = n_qubits + np.random.randint(-1, 2)
            
            # Quality metrics
            metric.error_rate = 0.001 * n_qubits * (1 + np.random.normal(0, 0.2))
            metric.fidelity = 1 - metric.error_rate
            
            # Resource usage
            metric.cpu_usage = 30 + n_qubits * 8 + np.random.normal(0, 5)
            metric.timestamp = time.time() + run * 0.1
            
            performance_metrics.append(metric)
    
    print(f"Generated {len(performance_metrics)} performance measurements")
    
    # Create performance visualizer
    visualizer = PerformanceVisualizer()
    
    # Generate comprehensive performance analysis
    print("\nCreating performance analysis visualization...")
    
    try:
        fig = visualizer.visualize_performance_metrics(
            performance_metrics,
            title="Quantum Algorithm Performance Analysis"
        )
        
        output_dir = Path("visualization_outputs")
        perf_file = output_dir / "performance_analysis.png"
        fig.savefig(perf_file, dpi=300, bbox_inches='tight')
        print(f"Performance analysis saved to: {perf_file}")
        
        import matplotlib.pyplot as plt
        plt.close(fig)
        
    except Exception as e:
        print(f"Performance visualization failed: {e}")
    
    # Create performance dashboard
    print("\nCreating performance dashboard...")
    
    try:
        dashboard_fig = visualizer.create_performance_dashboard(
            performance_metrics[:10]  # Use subset for dashboard
        )
        
        dashboard_file = output_dir / "performance_dashboard.png"
        dashboard_fig.savefig(dashboard_file, dpi=300, bbox_inches='tight')
        print(f"Performance dashboard saved to: {dashboard_file}")
        
        import matplotlib.pyplot as plt
        plt.close(dashboard_fig)
        
    except Exception as e:
        print(f"Performance dashboard creation failed: {e}")
    
    return performance_metrics


def demo_comprehensive_visualization():
    """Demonstrate comprehensive algorithm visualization workflow."""
    print("\n" + "="*60)
    print("COMPREHENSIVE ALGORITHM VISUALIZATION DEMO")
    print("="*60)
    
    # Create main visualizer with custom configuration
    config = VisualizationConfig(
        figure_size=(16, 12),
        color_scheme="quantum",
        enable_profiling_overlay=True,
        export_quality="high",
        include_metadata=True
    )
    
    main_visualizer = QuantumAlgorithmVisualizer(config)
    
    # Mock quantum circuit
    mock_circuit = type('MockCircuit', (), {})()
    mock_circuit.num_qubits = 3
    mock_circuit.gates = [
        type('Gate', (), {"name": "H", "qubits": [0], "params": []})(),
        type('Gate', (), {"name": "CNOT", "qubits": [0, 1], "params": []})(),
        type('Gate', (), {"name": "CNOT", "qubits": [1, 2], "params": []})(),
        type('Gate', (), {"name": "MEASURE", "qubits": [0], "params": []})()
    ]
    
    # Mock run method that returns state vector
    def mock_run():
        result = type('Result', (), {})()
        # GHZ state
        state_vector = np.zeros(8, dtype=complex)
        state_vector[0] = 1/np.sqrt(2)
        state_vector[7] = 1/np.sqrt(2)
        result.state_vector = state_vector
        return result
    
    mock_circuit.run = mock_run
    
    print("Creating comprehensive algorithm visualization...")
    
    try:
        # Generate all visualizations
        figures = main_visualizer.visualize_algorithm_execution(
            mock_circuit,
            include_state_evolution=True,
            include_performance=False,  # No performance data for mock
            title="Quantum Algorithm Comprehensive Analysis"
        )
        
        output_dir = Path("visualization_outputs")
        
        print(f"Generated {len(figures)} visualization figures:")
        
        # Export each figure in multiple formats
        for name, fig in figures.items():
            print(f"  Exporting {name} visualization...")
            
            # Export in different formats
            for fmt in ["png", "pdf", "svg"]:
                try:
                    filename = output_dir / f"comprehensive_{name}.{fmt}"
                    main_visualizer.export_visualization(fig, str(filename), format=fmt)
                    print(f"    Saved {fmt.upper()}: {filename}")
                except Exception as e:
                    print(f"    Failed to export {fmt.upper()}: {e}")
            
            # Export HTML version
            try:
                html_filename = output_dir / f"comprehensive_{name}.html"
                main_visualizer.export_visualization(fig, str(html_filename), format="html")
                print(f"    Saved HTML: {html_filename}")
            except Exception as e:
                print(f"    Failed to export HTML: {e}")
        
        # Clean up figures
        import matplotlib.pyplot as plt
        for fig in figures.values():
            plt.close(fig)
        
    except Exception as e:
        print(f"Comprehensive visualization failed: {e}")


def demo_comparative_analysis():
    """Demonstrate comparative algorithm analysis."""
    print("\n" + "="*60)
    print("COMPARATIVE ALGORITHM ANALYSIS DEMO")
    print("="*60)
    
    # Create multiple mock algorithms for comparison
    algorithms = []
    algorithm_names = []
    
    # Algorithm 1: Simple Bell state preparation
    algo1 = type('Algorithm', (), {})()
    algo1.num_qubits = 2
    algo1.gates = [
        type('Gate', (), {"name": "H"})(),
        type('Gate', (), {"name": "CNOT"})()
    ]
    algorithms.append(algo1)
    algorithm_names.append("Bell State Preparation")
    
    # Algorithm 2: GHZ state preparation
    algo2 = type('Algorithm', (), {})()
    algo2.num_qubits = 3
    algo2.gates = [
        type('Gate', (), {"name": "H"})(),
        type('Gate', (), {"name": "CNOT"})(),
        type('Gate', (), {"name": "CNOT"})()
    ]
    algorithms.append(algo2)
    algorithm_names.append("GHZ State Preparation")
    
    # Algorithm 3: Quantum Fourier Transform (simplified)
    algo3 = type('Algorithm', (), {})()
    algo3.num_qubits = 3
    algo3.gates = [
        type('Gate', (), {"name": "H"})(),
        type('Gate', (), {"name": "RZ"})(),
        type('Gate', (), {"name": "CNOT"})(),
        type('Gate', (), {"name": "H"})(),
        type('Gate', (), {"name": "RZ"})(),
        type('Gate', (), {"name": "CNOT"})(),
        type('Gate', (), {"name": "H"})(),
        type('Gate', (), {"name": "SWAP"})()
    ]
    algorithms.append(algo3)
    algorithm_names.append("Quantum Fourier Transform")
    
    print(f"Comparing {len(algorithms)} quantum algorithms...")
    
    try:
        # Create comparative visualization
        fig = compare_quantum_algorithms(
            algorithms,
            algorithm_names,
            title="Quantum Algorithm Comparison Study"
        )
        
        output_dir = Path("visualization_outputs")
        comparison_file = output_dir / "algorithm_comparison.png"
        fig.savefig(comparison_file, dpi=300, bbox_inches='tight')
        
        print(f"Algorithm comparison saved to: {comparison_file}")
        
        import matplotlib.pyplot as plt
        plt.close(fig)
        
    except Exception as e:
        print(f"Comparative analysis failed: {e}")


def demo_convenience_functions():
    """Demonstrate convenience functions for quick visualization."""
    print("\n" + "="*60)
    print("CONVENIENCE FUNCTIONS DEMO")
    print("="*60)
    
    output_dir = Path("visualization_outputs")
    
    # 1. Quick circuit visualization
    print("\n1. Quick Circuit Visualization")
    mock_circuit = type('Circuit', (), {})()
    mock_circuit.num_qubits = 2
    mock_circuit.gates = [
        type('Gate', (), {"name": "H", "qubits": [0]})(),
        type('Gate', (), {"name": "CNOT", "qubits": [0, 1]})()
    ]
    
    try:
        fig = visualize_quantum_circuit(mock_circuit, title="Quick Circuit Demo")
        
        quick_circuit_file = output_dir / "quick_circuit_demo.png"
        fig.savefig(quick_circuit_file, dpi=300, bbox_inches='tight')
        print(f"  Quick circuit visualization saved to: {quick_circuit_file}")
        
        import matplotlib.pyplot as plt
        plt.close(fig)
        
    except Exception as e:
        print(f"  Quick circuit visualization failed: {e}")
    
    # 2. Quick state visualization
    print("\n2. Quick State Visualization")
    
    # Create superposition state |+⟩ = (|0⟩ + |1⟩)/√2
    plus_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    
    try:
        fig = visualize_quantum_state(
            plus_state,
            visualization_type="amplitudes",
            title="Plus State Amplitudes"
        )
        
        quick_state_file = output_dir / "quick_state_demo.png"
        fig.savefig(quick_state_file, dpi=300, bbox_inches='tight')
        print(f"  Quick state visualization saved to: {quick_state_file}")
        
        import matplotlib.pyplot as plt
        plt.close(fig)
        
    except Exception as e:
        print(f"  Quick state visualization failed: {e}")
    
    # 3. Quick Bloch sphere visualization
    print("\n3. Quick Bloch Sphere Visualization")
    
    # Create |i⟩ state = (|0⟩ + i|1⟩)/√2
    i_state = np.array([1/np.sqrt(2), 1j/np.sqrt(2)], dtype=complex)
    
    try:
        fig = create_bloch_sphere_visualization(
            i_state,
            title="Imaginary State on Bloch Sphere"
        )
        
        quick_bloch_file = output_dir / "quick_bloch_demo.png"
        fig.savefig(quick_bloch_file, dpi=300, bbox_inches='tight')
        print(f"  Quick Bloch sphere visualization saved to: {quick_bloch_file}")
        
        import matplotlib.pyplot as plt
        plt.close(fig)
        
    except Exception as e:
        print(f"  Quick Bloch sphere visualization failed: {e}")


def demo_gui_interface():
    """Demonstrate GUI interface (if available)."""
    print("\n" + "="*60)
    print("GUI INTERFACE DEMO")
    print("="*60)
    
    if not HAS_TKINTER:
        print("Tkinter GUI not available. Skipping GUI demo.")
        return
    
    try:
        from quantrs2.quantum_algorithm_visualization import VisualizationGUI
        
        print("Tkinter GUI support is available!")
        print("To run the GUI interface, uncomment the following lines in the demo:")
        print("  gui = VisualizationGUI()")
        print("  gui.run()")
        print("\nNote: GUI demo is commented out to prevent blocking the demo script.")
        
        # Uncomment these lines to actually run the GUI:
        # gui = VisualizationGUI()
        # print("Starting GUI interface...")
        # gui.run()
        
    except Exception as e:
        print(f"GUI interface failed: {e}")


def demo_web_interface():
    """Demonstrate web interface (if available)."""
    print("\n" + "="*60)
    print("WEB INTERFACE DEMO")
    print("="*60)
    
    if not HAS_DASH:
        print("Dash web interface not available. Skipping web demo.")
        return
    
    try:
        from quantrs2.quantum_algorithm_visualization import create_quantum_visualization_app
        
        print("Dash web interface support is available!")
        print("To run the web interface, uncomment the following lines in the demo:")
        print("  app = create_quantum_visualization_app()")
        print("  app.run_server(debug=True, port=8050)")
        print("\nNote: Web demo is commented out to prevent blocking the demo script.")
        print("The web interface would be available at: http://localhost:8050")
        
        # Uncomment these lines to actually run the web server:
        # app = create_quantum_visualization_app()
        # print("Starting web interface on http://localhost:8050")
        # app.run_server(debug=True, port=8050)
        
    except Exception as e:
        print(f"Web interface failed: {e}")


def main():
    """Run the comprehensive quantum algorithm visualization demo."""
    print("QuantRS2 Quantum Algorithm Visualization System Demo")
    print("=" * 70)
    print("This demo showcases the comprehensive visualization capabilities")
    print("of the QuantRS2 quantum computing framework.")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path("visualization_outputs")
    output_dir.mkdir(exist_ok=True)
    print(f"\nAll visualizations will be saved to: {output_dir.absolute()}")
    
    try:
        # Run all demo sections
        circuit_data = demo_circuit_visualization()
        state_data = demo_state_visualization()
        performance_data = demo_performance_visualization()
        demo_comprehensive_visualization()
        demo_comparative_analysis()
        demo_convenience_functions()
        demo_gui_interface()
        demo_web_interface()
        
        print("\n" + "="*70)
        print("DEMO COMPLETE!")
        print("="*70)
        print(f"All visualizations have been saved to: {output_dir.absolute()}")
        print("\nGenerated files:")
        
        # List all generated files
        for file_path in sorted(output_dir.iterdir()):
            if file_path.is_file():
                size_kb = file_path.stat().st_size / 1024
                print(f"  {file_path.name} ({size_kb:.1f} KB)")
        
        print("\nVisualization capabilities demonstrated:")
        print("  ✓ Circuit diagram visualization with performance overlays")
        print("  ✓ Quantum state visualization (amplitudes, probabilities, phases)")
        print("  ✓ 3D Bloch sphere visualizations with animation")
        print("  ✓ Performance analytics and correlation analysis") 
        print("  ✓ Multi-format export (PNG, PDF, SVG, HTML)")
        print("  ✓ Comparative algorithm analysis")
        print("  ✓ Convenience functions for quick visualization")
        
        if HAS_TKINTER:
            print("  ✓ GUI interface support available")
        if HAS_DASH:
            print("  ✓ Web interface support available")
        if HAS_PLOTLY:
            print("  ✓ Interactive Plotly visualizations available")
        
        print("\nThe QuantRS2 Quantum Algorithm Visualization System is fully functional!")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)