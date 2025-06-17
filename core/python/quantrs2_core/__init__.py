"""
QuantRS2-Core Python Bindings

This package provides Python bindings for the QuantRS2-Core quantum computing framework,
enabling seamless integration with Python ecosystem tools like NumPy, Jupyter, and 
scientific computing libraries.

The QuantRS2-Core framework offers:
- Comprehensive quantum gate implementations
- Advanced quantum decomposition algorithms  
- Variational quantum circuits with automatic differentiation
- Quantum sensor networks with distributed sensing capabilities
- Quantum internet simulation with global coverage
- Hardware-specific compilation and optimization
- Error correction and fault-tolerant computing
- Machine learning accelerated quantum algorithms
- Interactive Jupyter notebook visualization tools
- Comprehensive quantum algorithm complexity analysis

Example usage:
    >>> import quantrs2_core as qrs
    >>> 
    >>> # Create qubits
    >>> q0 = qrs.QubitId(0)
    >>> q1 = qrs.QubitId(1)
    >>> 
    >>> # Create quantum gates
    >>> h_gate = qrs.create_hadamard_gate(0)
    >>> cnot_gate = qrs.create_cnot_gate(0, 1)
    >>> 
    >>> # Get matrix representations
    >>> h_matrix = h_gate.matrix()
    >>> cnot_matrix = cnot_gate.matrix()
    >>> 
    >>> # Create variational circuits
    >>> circuit = qrs.VariationalCircuit(4)
    >>> circuit.add_rotation_layer("x")
    >>> circuit.add_entangling_layer()
    >>> 
    >>> # Quantum decomposition
    >>> import numpy as np
    >>> unitary = np.eye(2, dtype=complex)
    >>> decomp = qrs.decompose_single_qubit(unitary)
    >>> 
    >>> # Quantum sensor networks
    >>> network = qrs.QuantumSensorNetwork(12345)
    >>> sensor_id = network.add_sensor("magnetometer", 37.7749, -122.4194)
    >>> advantage = network.get_sensor_advantage()
    >>> 
    >>> # Quantum internet
    >>> internet = qrs.QuantumInternet()
    >>> node_id = internet.add_quantum_node(40.7128, -74.0060, "datacenter")
    >>> coverage = internet.get_coverage_percentage()
    >>> 
    >>> # Jupyter visualization tools
    >>> circuit_viz = qrs.QuantumCircuitVisualizer(3, "Bell State Circuit")
    >>> circuit_viz.add_gate("H", [0], None, 0.99)
    >>> circuit_viz.add_gate("CNOT", [0, 1], None, 0.95)
    >>> html_output = circuit_viz.to_html()  # For Jupyter notebook display
    >>> 
    >>> # Quantum state visualization
    >>> import numpy as np
    >>> bell_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
    >>> state_viz = qrs.QuantumStateVisualizer(bell_state)
    >>> probabilities = state_viz.measurement_probabilities()
    >>> 
    >>> # Quantum algorithm complexity analysis
    >>> analyzer = qrs.QuantumComplexityAnalyzer("Grover Search")
    >>> gates = [("H", [0], None), ("CNOT", [0, 1], None), ("X", [1], None)]
    >>> analyzer.analyze_circuit(gates, "Grover", 1000)
    >>> report = analyzer.get_analysis_report()
    >>> scaling = analyzer.predict_scaling([2000, 4000, 8000])
    >>> advantage = analyzer.quantum_advantage_analysis()
    >>> 
    >>> # Real-time quantum system monitoring
    >>> config = qrs.MonitoringConfig(monitoring_interval_secs=1.0, data_retention_hours=24.0)
    >>> config.set_alert_thresholds(max_gate_error_rate=0.01, max_readout_error_rate=0.05, min_coherence_time_us=50.0)
    >>> monitor = qrs.RealtimeMonitor(config)
    >>> monitor.start_monitoring()
    >>> 
    >>> # Get real-time metrics
    >>> current_metrics = monitor.get_current_metrics(["gate_error_rate", "qubit_coherence_time"])
    >>> stats = monitor.get_aggregated_stats("gate_error_rate")
    >>> alerts = monitor.get_active_alerts()
    >>> recommendations = monitor.get_optimization_recommendations()
    >>> status = monitor.get_monitoring_status()
    >>> 
    >>> # NumRS2 integration for high-performance arrays
    >>> import numpy as np
    >>> 
    >>> # Create NumRS2 arrays for quantum computations
    >>> quantum_state = qrs.numrs2_zeros([4])  # 2-qubit quantum state
    >>> gate_matrix = qrs.numrs2_ones([4, 4])  # 2-qubit gate matrix
    >>> 
    >>> # Convert between NumPy and NumRS2
    >>> numpy_array = np.array([[1+0j, 0], [0, 1+0j]], dtype=complex)
    >>> numrs2_array = qrs.numpy_to_numrs2(numpy_array)
    >>> back_to_numpy = numrs2_array.to_numpy()
    >>> 
    >>> # High-performance quantum operations with NumRS2
    >>> result = numrs2_array.matmul(numrs2_array)  # Matrix multiplication
    >>> transposed = numrs2_array.transpose()  # Transpose operation
    >>> reshaped = numrs2_array.reshape([2, 2])  # Reshape operation

For detailed documentation and examples, visit: https://docs.quantrs2.com
"""

# Import the core Rust module
from .quantrs2_core import *

# Package metadata
__version__ = "0.1.0-alpha.5"
__author__ = "QuantRS2 Team"
__description__ = "Python bindings for QuantRS2-Core quantum computing framework"
__license__ = "MIT OR Apache-2.0"
__homepage__ = "https://github.com/quantrs2/quantrs2-core"

# Re-export main classes and functions for convenience
__all__ = [
    # Core classes
    "QubitId",
    "QuantumGate", 
    "SingleQubitDecomposition",
    "CartanDecomposition",
    "VariationalCircuit",
    "QuantumSensorNetwork",
    "QuantumInternet",
    "NumRS2Array",
    
    # Jupyter visualization classes
    "QuantumCircuitVisualizer",
    "QuantumStateVisualizer", 
    "QuantumPerformanceMonitor",
    
    # Quantum complexity analysis classes
    "QuantumComplexityAnalyzer",
    
    # Real-time monitoring classes
    "RealtimeMonitor",
    "MonitoringConfig", 
    "MetricMeasurement",
    "AggregatedStats",
    "Alert",
    "OptimizationRecommendation",
    "MonitoringStatus",
    
    # Gate creation functions
    "create_hadamard_gate",
    "create_pauli_x_gate",
    "create_pauli_y_gate", 
    "create_pauli_z_gate",
    "create_rotation_x_gate",
    "create_rotation_y_gate",
    "create_rotation_z_gate",
    "create_cnot_gate",
    
    # Decomposition functions
    "decompose_single_qubit",
    "decompose_two_qubit_cartan",
    
    # NumRS2 integration functions
    "create_numrs2_array",
    "numrs2_zeros", 
    "numrs2_ones",
    "numpy_to_numrs2",
    "numrs2_from_vec",
    
    # Quantum complexity analysis functions
    "analyze_algorithm_complexity",
    "compare_quantum_classical_complexity",
    "calculate_theoretical_quantum_volume",
    
    # Metadata
    "__version__",
    "__author__",
    "__description__",
    "__license__",
    "__homepage__",
]

def info():
    """Print information about QuantRS2-Core"""
    print(f"QuantRS2-Core v{__version__}")
    print(f"Author: {__author__}")
    print(f"Description: {__description__}")
    print(f"License: {__license__}")
    print(f"Homepage: {__homepage__}")
    print("\nAvailable classes and functions:")
    for item in sorted(__all__):
        if not item.startswith("__"):
            print(f"  - {item}")

def get_version():
    """Get the version string"""
    return __version__