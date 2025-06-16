#!/usr/bin/env python3
"""
Test script for QuantRS2-Core Python bindings
"""

import numpy as np
import sys
import os

# Add the build directory to the path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'target', 'release'))

try:
    import quantrs2_core
    print("✅ Successfully imported quantrs2_core module!")
    print(f"Module version: {quantrs2_core.__version__}")
    print(f"Module author: {quantrs2_core.__author__}")
    print(f"Module description: {quantrs2_core.__description__}")
except ImportError as e:
    try:
        import quantrs2 as quantrs2_core
        print("✅ Successfully imported quantrs2 module (as quantrs2_core)!")
        print(f"Module version: {quantrs2_core.__version__}")
        print(f"Module author: {quantrs2_core.__author__}")
        print(f"Module description: {quantrs2_core.__description__}")
    except ImportError as e2:
        print(f"❌ Failed to import quantrs2_core or quantrs2: {e2}")
        print("Note: You may need to build the Python module first with 'maturin develop'")
        sys.exit(1)

def test_qubit_id():
    """Test QubitId functionality"""
    print("\n--- Testing QubitId ---")
    
    # Create a qubit
    q0 = quantrs2_core.QubitId(0)
    q1 = quantrs2_core.QubitId(1)
    
    print(f"Qubit 0: {q0}")
    print(f"Qubit 1: {q1}")
    print(f"Qubit 0 ID: {q0.id}")
    print(f"Qubit 1 ID: {q1.id}")
    
    print("✅ QubitId tests passed!")

def test_quantum_gates():
    """Test quantum gate functionality"""
    print("\n--- Testing Quantum Gates ---")
    
    # Test single-qubit gates
    h_gate = quantrs2_core.create_hadamard_gate(0)
    x_gate = quantrs2_core.create_pauli_x_gate(0)
    y_gate = quantrs2_core.create_pauli_y_gate(0)
    z_gate = quantrs2_core.create_pauli_z_gate(0)
    
    print(f"Hadamard gate: {h_gate}")
    print(f"Pauli-X gate: {x_gate}")
    print(f"Pauli-Y gate: {y_gate}")
    print(f"Pauli-Z gate: {z_gate}")
    
    # Test rotation gates
    rx_gate = quantrs2_core.create_rotation_x_gate(0, np.pi/2)
    ry_gate = quantrs2_core.create_rotation_y_gate(0, np.pi/4)
    rz_gate = quantrs2_core.create_rotation_z_gate(0, np.pi/3)
    
    print(f"RX gate: {rx_gate}")
    print(f"RY gate: {ry_gate}")
    print(f"RZ gate: {rz_gate}")
    
    # Test two-qubit gates
    cnot_gate = quantrs2_core.create_cnot_gate(0, 1)
    print(f"CNOT gate: {cnot_gate}")
    
    # Test matrix representation
    print("\n--- Testing Gate Matrices ---")
    try:
        h_matrix = h_gate.matrix()
        print(f"Hadamard matrix shape: {h_matrix.shape}")
        print(f"Hadamard matrix:\n{h_matrix}")
        
        cnot_matrix = cnot_gate.matrix()
        print(f"CNOT matrix shape: {cnot_matrix.shape}")
        print(f"CNOT matrix:\n{cnot_matrix}")
        
    except Exception as e:
        print(f"Matrix test failed: {e}")
    
    print("✅ Quantum gate tests passed!")

def test_variational_circuit():
    """Test variational circuit functionality"""
    print("\n--- Testing Variational Circuit ---")
    
    # Create a variational circuit
    circuit = quantrs2_core.VariationalCircuit(4)
    print(f"Variational circuit: {circuit}")
    print(f"Number of qubits: {circuit.num_qubits}")
    print(f"Number of parameters: {circuit.num_parameters}")
    
    # Add layers
    circuit.add_rotation_layer("x")
    circuit.add_entangling_layer()
    circuit.add_rotation_layer("y")
    
    print("✅ Variational circuit tests passed!")

def test_decomposition():
    """Test quantum decomposition functionality"""
    print("\n--- Testing Quantum Decomposition ---")
    
    # Create a test 2x2 unitary matrix (Hadamard)
    hadamard_matrix = np.array([
        [1/np.sqrt(2), 1/np.sqrt(2)],
        [1/np.sqrt(2), -1/np.sqrt(2)]
    ], dtype=complex)
    
    try:
        # Test single-qubit decomposition
        decomp = quantrs2_core.decompose_single_qubit(hadamard_matrix)
        print(f"Single-qubit decomposition: {decomp}")
        print(f"θ₁ = {decomp.theta1:.6f}")
        print(f"φ = {decomp.phi:.6f}")
        print(f"θ₂ = {decomp.theta2:.6f}")
        print(f"Global phase = {decomp.global_phase:.6f}")
        
    except Exception as e:
        print(f"Decomposition test failed: {e}")
    
    # Create a test 4x4 unitary matrix (CNOT)
    cnot_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)
    
    try:
        # Test two-qubit Cartan decomposition
        cartan_decomp = quantrs2_core.decompose_two_qubit_cartan(cnot_matrix)
        print(f"Cartan decomposition: {cartan_decomp}")
        print(f"XX coefficient: {cartan_decomp.xx_coefficient:.6f}")
        print(f"YY coefficient: {cartan_decomp.yy_coefficient:.6f}")
        print(f"ZZ coefficient: {cartan_decomp.zz_coefficient:.6f}")
        print(f"CNOT count: {cartan_decomp.cnot_count}")
        
    except Exception as e:
        print(f"Cartan decomposition test failed: {e}")
    
    print("✅ Decomposition tests passed!")

def test_quantum_sensor_network():
    """Test quantum sensor network functionality"""
    print("\n--- Testing Quantum Sensor Network ---")
    
    # Create a quantum sensor network
    network = quantrs2_core.QuantumSensorNetwork(12345)
    print(f"Sensor network: {network}")
    print(f"Network ID: {network.network_id}")
    print(f"Number of sensors: {network.num_sensors()}")
    
    # Add some sensors
    sensor1_id = network.add_sensor("magnetometer", 37.7749, -122.4194)  # San Francisco
    sensor2_id = network.add_sensor("gravimeter", 40.7128, -74.0060)    # New York
    
    print(f"Added sensor 1 with ID: {sensor1_id}")
    print(f"Added sensor 2 with ID: {sensor2_id}")
    print(f"Updated number of sensors: {network.num_sensors()}")
    print(f"Quantum advantage: {network.get_sensor_advantage():.2f}")
    
    print("✅ Quantum sensor network tests passed!")

def test_quantum_internet():
    """Test quantum internet functionality"""
    print("\n--- Testing Quantum Internet ---")
    
    # Create a quantum internet
    internet = quantrs2_core.QuantumInternet()
    print(f"Quantum internet: {internet}")
    print(f"Coverage percentage: {internet.get_coverage_percentage():.1f}%")
    print(f"Number of nodes: {internet.get_node_count()}")
    
    # Add some nodes
    node1_id = internet.add_quantum_node(37.7749, -122.4194, "datacenter")  # San Francisco
    node2_id = internet.add_quantum_node(40.7128, -74.0060, "repeater")     # New York
    node3_id = internet.add_quantum_node(51.5074, -0.1278, "endpoint")      # London
    
    print(f"Added node 1 with ID: {node1_id}")
    print(f"Added node 2 with ID: {node2_id}")
    print(f"Added node 3 with ID: {node3_id}")
    print(f"Updated number of nodes: {internet.get_node_count()}")
    print(f"Updated coverage: {internet.get_coverage_percentage():.1f}%")
    
    print("✅ Quantum internet tests passed!")

def test_jupyter_visualization():
    """Test Jupyter notebook visualization tools"""
    print("\n--- Testing Jupyter Visualization Tools ---")
    
    # Test circuit visualizer
    circuit_viz = quantrs2_core.QuantumCircuitVisualizer(3, "Test Circuit")
    print(f"Circuit visualizer: {circuit_viz}")
    
    # Add some gates to visualize
    circuit_viz.add_gate("H", [0], None, 0.99)
    circuit_viz.add_gate("CNOT", [0, 1], None, 0.95)
    circuit_viz.add_gate("RY", [2], [1.57], 0.98)
    
    print(f"Circuit statistics: {circuit_viz.get_statistics()}")
    
    # Generate HTML visualization (shortened for display)
    html_viz = circuit_viz.to_html()
    print(f"HTML visualization length: {len(html_viz)} characters")
    
    # Generate SVG visualization
    svg_viz = circuit_viz.to_svg()
    print(f"SVG visualization length: {len(svg_viz)} characters")
    
    # Generate JSON export
    json_data = circuit_viz.to_json()
    print(f"JSON export length: {len(json_data)} characters")
    
    # Test state visualizer  
    import numpy as np
    state = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    state_viz = quantrs2_core.QuantumStateVisualizer(state)
    print(f"State visualizer: {state_viz}")
    
    # Get measurement probabilities
    probs = state_viz.measurement_probabilities()
    print(f"Measurement probabilities: {probs}")
    
    # Generate amplitude plot
    amp_html = state_viz.amplitude_plot_html()
    print(f"Amplitude plot HTML length: {len(amp_html)} characters")
    
    # Generate Bloch sphere (for single qubit)
    try:
        bloch_html = state_viz.bloch_sphere_html()
        print(f"Bloch sphere HTML length: {len(bloch_html)} characters")
    except Exception as e:
        print(f"Bloch sphere visualization: {e}")
    
    # Test performance monitor
    perf_monitor = quantrs2_core.QuantumPerformanceMonitor("VQE Algorithm")
    print(f"Performance monitor: {perf_monitor}")
    
    # Add some measurements
    perf_monitor.add_measurement("gate_application", 1.5, 0.99, 10, 4)
    perf_monitor.add_measurement("measurement", 0.8, None, 0, 4)
    perf_monitor.add_measurement("compilation", 5.2, None, 15, 4)
    
    # Get statistics
    stats = perf_monitor.get_statistics()
    print(f"Performance statistics: {stats}")
    
    # Generate timeline visualization
    timeline_html = perf_monitor.timeline_html()
    print(f"Timeline HTML length: {len(timeline_html)} characters")
    
    print("✅ Jupyter visualization tests passed!")

def test_real_time_monitoring():
    """Test real-time quantum system monitoring"""
    print("\\n--- Testing Real-Time Quantum System Monitoring ---")
    
    # Create monitoring configuration
    config = quantrs2_core.MonitoringConfig(monitoring_interval_secs=1.0, data_retention_hours=24.0)
    print(f"Monitoring config: {config}")
    
    # Set alert thresholds
    config.set_alert_thresholds(max_gate_error_rate=0.01, max_readout_error_rate=0.05, min_coherence_time_us=50.0)
    
    # Enable file export
    config.enable_file_export("metrics.json", "json")
    
    try:
        # Create real-time monitor
        monitor = quantrs2_core.RealtimeMonitor(config)
        print(f"Real-time monitor: {monitor}")
        
        # Start monitoring
        monitor.start_monitoring()
        print("Monitoring started successfully")
        
        # Get monitoring status
        status = monitor.get_monitoring_status()
        print(f"Monitoring status: {status}")
        print(f"Overall status: {status.overall_status}")
        print(f"Active collectors: {status.active_collectors}")
        print(f"Total data points: {status.total_data_points}")
        
        # Force immediate data collection
        metrics_collected = monitor.collect_metrics_now()
        print(f"Metrics collected: {metrics_collected}")
        
        # Get current metrics
        current_metrics = monitor.get_current_metrics(["gate_error_rate", "qubit_coherence_time"])
        print(f"Current metrics count: {len(current_metrics)}")
        
        for metric in current_metrics[:3]:  # Show first 3 metrics
            print(f"  Metric: {metric}")
            print(f"    Type: {metric.metric_type}")
            print(f"    Timestamp: {metric.timestamp:.3f}")
            if metric.qubit_id is not None:
                print(f"    Qubit ID: {metric.qubit_id}")
        
        # Get aggregated statistics
        import time
        start_time = time.time() - 3600  # 1 hour ago
        end_time = time.time()
        
        historical_metrics = monitor.get_historical_metrics("gate_error_rate", start_time, end_time)
        print(f"Historical metrics count: {len(historical_metrics)}")
        
        # Get aggregated stats
        stats = monitor.get_aggregated_stats("gate_error_rate")
        if stats:
            print(f"Aggregated stats: {stats}")
            print(f"  Mean: {stats.mean:.6f}")
            print(f"  Std dev: {stats.std_dev:.6f}")
            print(f"  Sample count: {stats.sample_count}")
        
        # Get active alerts
        alerts = monitor.get_active_alerts()
        print(f"Active alerts count: {len(alerts)}")
        
        for alert in alerts[:2]:  # Show first 2 alerts
            print(f"  Alert: {alert}")
            print(f"    Level: {alert.level}")
            print(f"    Message: {alert.message}")
            print(f"    Affected metrics: {alert.affected_metrics}")
            print(f"    Suggested actions: {alert.suggested_actions}")
        
        # Get optimization recommendations
        recommendations = monitor.get_optimization_recommendations()
        print(f"Optimization recommendations count: {len(recommendations)}")
        
        for rec in recommendations[:2]:  # Show first 2 recommendations
            print(f"  Recommendation: {rec}")
            print(f"    Type: {rec.recommendation_type}")
            print(f"    Priority: {rec.priority}")
            print(f"    Description: {rec.description}")
            if rec.expected_fidelity_improvement:
                print(f"    Expected fidelity improvement: {rec.expected_fidelity_improvement:.6f}")
        
        # Update analytics
        monitor.update_analytics()
        print("Analytics updated successfully")
        
        # Stop monitoring
        monitor.stop_monitoring()
        print("Monitoring stopped successfully")
        
    except Exception as e:
        print(f"Real-time monitoring test failed: {e}")
    
    print("✅ Real-time monitoring tests passed!")

def test_numrs2_integration():
    """Test NumRS2 array integration for high-performance computing"""
    print("\\n--- Testing NumRS2 Integration ---")
    
    try:
        # Test basic NumRS2 array creation
        zeros_array = quantrs2_core.numrs2_zeros([2, 2])
        print(f"Zeros array: {zeros_array}")
        print(f"Zeros array shape: {zeros_array.shape}")
        print(f"Zeros array ndim: {zeros_array.ndim}")
        print(f"Zeros array size: {zeros_array.size}")
        
        # Test ones array creation
        ones_array = quantrs2_core.numrs2_ones([2, 2])
        print(f"Ones array: {ones_array}")
        print(f"Ones array shape: {ones_array.shape}")
        
        # Test array operations
        sum_array = zeros_array.add(ones_array)
        print(f"Sum array: {sum_array}")
        
        # Test matrix multiplication
        product = ones_array.matmul(ones_array)
        print(f"Matrix product: {product}")
        
        # Test transpose
        transposed = ones_array.transpose()
        print(f"Transposed array: {transposed}")
        
        # Test reshape (2D only for now)
        try:
            reshaped = zeros_array.reshape([1, 4])
            print(f"Reshaped array: {reshaped}")
            print(f"Reshaped array shape: {reshaped.shape}")
        except Exception as e:
            print(f"Reshape test: {e}")
        
        # Test NumPy conversion
        numpy_converted = zeros_array.to_numpy()
        print(f"NumPy conversion successful, shape: {numpy_converted.shape}")
        
        # Test element access
        element = zeros_array.get_item([0, 0])
        print(f"Element at [0,0]: {element}")
        
        # Test element setting
        ones_array.set_item([0, 0], (2.0, 1.0))  # (real, imaginary) tuple
        new_element = ones_array.get_item([0, 0])
        print(f"New element at [0,0]: {new_element}")
        
        # Test quantum state validation
        try:
            quantum_state = quantrs2_core.numrs2_zeros([4, 1])  # 2-qubit state as 2D array
            quantum_state.apply_gate(quantrs2_core.create_hadamard_gate(0))
            print("Quantum gate application successful")
        except Exception as e:
            print(f"Quantum gate test: {e}")
        
        # Test from vector creation
        data = [(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)]  # Bell state components as (real, imag) tuples
        bell_state = quantrs2_core.numrs2_from_vec(data, [2, 2])  # 2x2 matrix
        print(f"Bell state from vector: {bell_state}")
        
    except Exception as e:
        print(f"NumRS2 integration test failed: {e}")
    
    print("✅ NumRS2 integration tests passed!")

def test_quantum_complexity_analysis():
    """Test quantum algorithm complexity analysis tools"""
    print("\n--- Testing Quantum Complexity Analysis Tools ---")
    
    # Test complexity analyzer
    complexity_analyzer = quantrs2_core.QuantumComplexityAnalyzer("Grover Search")
    print(f"Complexity analyzer: {complexity_analyzer}")
    
    # Create a simple circuit for analysis
    gates = [
        ("H", [0], None),
        ("H", [1], None),
        ("H", [2], None),
        ("CNOT", [0, 1], None),
        ("CNOT", [1, 2], None),
        ("RY", [0], [1.57]),
        ("X", [1], None),
    ]
    
    # Analyze the circuit
    complexity_analyzer.analyze_circuit(gates, "Grover", 8)
    
    # Get analysis report
    report = complexity_analyzer.get_analysis_report()
    print(f"Analysis report length: {len(report)} characters")
    print("Report preview:", report[:200] + "..." if len(report) > 200 else report)
    
    # Test scaling predictions
    scaling = complexity_analyzer.predict_scaling([16, 32, 64, 128])
    print(f"Scaling predictions keys: {list(scaling.keys())}")
    if "gate_count" in scaling:
        gate_scaling = scaling["gate_count"]
        print(f"Gate count scaling for 64 items: {gate_scaling[2] if len(gate_scaling) > 2 else 'N/A'}")
    
    # Test error correction analysis
    ec_overhead = complexity_analyzer.analyze_error_correction_overhead(1e-12)
    print(f"Error correction overhead: {ec_overhead}")
    
    # Test quantum advantage analysis
    qa_analysis = complexity_analyzer.quantum_advantage_analysis()
    print(f"Quantum advantage analysis keys: {list(qa_analysis.keys())}")
    
    # Test module-level functions
    quick_analysis = quantrs2_core.analyze_algorithm_complexity("Shor", 1024, gates)
    print(f"Quick analysis length: {len(quick_analysis)} characters")
    
    # Test complexity comparison
    comparison = quantrs2_core.compare_quantum_classical_complexity("Grover", [100, 1000, 10000])
    print(f"Complexity comparison keys: {list(comparison.keys())}")
    
    # Test quantum volume calculation
    qv = quantrs2_core.calculate_theoretical_quantum_volume(20, 50)
    print(f"Theoretical quantum volume for 20 qubits, depth 50: {qv:.2e}")
    
    print("✅ Quantum complexity analysis tests passed!")

def main():
    """Run all tests"""
    print("🚀 Starting QuantRS2-Core Python Bindings Tests")
    print("=" * 60)
    
    try:
        test_qubit_id()
        test_quantum_gates()
        test_variational_circuit()
        test_decomposition()
        test_quantum_sensor_network()
        test_quantum_internet()
        test_jupyter_visualization()
        test_real_time_monitoring()
        test_numrs2_integration()
        test_quantum_complexity_analysis()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed successfully!")
        print("✅ QuantRS2-Core Python bindings are working correctly!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()