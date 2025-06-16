"""
QuantRS2 Core - Advanced quantum computing framework core functionality.

This module provides access to the core quantum computing framework 
functionality including real-time monitoring, hardware compilation,
NumRS2 integration, and advanced quantum operations.
"""

# Version information
__version__ = "0.1.0-alpha.5"
__author__ = "QuantRS2 Team"
__description__ = "Core quantum computing framework with real-time monitoring and hardware compilation"

try:
    # Import the core native module
    # This will be built by maturin as quantrs2.core
    from . import _core
    
    # Import core classes and functions
    from ._core import (
        # Core types
        QubitId,
        QuantumGate,
        VariationalCircuit,
        
        # Gate creation functions
        create_hadamard_gate,
        create_pauli_x_gate,
        create_pauli_y_gate,
        create_pauli_z_gate,
        create_rotation_x_gate,
        create_rotation_y_gate,
        create_rotation_z_gate,
        create_cnot_gate,
        create_phase_gate,
        create_s_gate,
        create_t_gate,
        create_identity_gate,
        
        # Decomposition functions
        decompose_single_qubit,
        decompose_two_qubit_cartan,
        SingleQubitDecomposition,
        CartanDecomposition,
        
        # Quantum sensor network
        QuantumSensorNetwork,
        
        # Quantum internet
        QuantumInternet,
        
        # Visualization tools
        QuantumCircuitVisualizer,
        QuantumStateVisualizer,
        QuantumPerformanceMonitor,
        
        # Real-time monitoring
        MonitoringConfig,
        RealtimeMonitor,
        MonitoringStatus,
        SystemMetric,
        AggregatedStats,
        OptimizationRecommendation,
        Alert,
        
        # NumRS2 integration
        NumRS2Array,
        numrs2_zeros,
        numrs2_ones,
        numrs2_from_vec,
        
        # Quantum complexity analysis
        QuantumComplexityAnalyzer,
        analyze_algorithm_complexity,
        compare_quantum_classical_complexity,
        calculate_theoretical_quantum_volume,
    )
    
    # Store reference to native module
    _native_core = _core
    
except ImportError as e:
    import warnings
    warnings.warn(f"Core native module not found: {e}. Using stub implementations.")
    
    # Stub implementations
    class QubitId:
        def __init__(self, id):
            self.id = id
        def __str__(self):
            return f"QubitId({self.id})"
    
    class QuantumGate:
        def __init__(self, name, qubits, params=None):
            self.name = name
            self.qubits = qubits
            self.params = params or []
        def __str__(self):
            return f"QuantumGate({self.name}, {self.qubits})"
    
    # Stub gate creation functions
    def create_hadamard_gate(qubit_id):
        return QuantumGate("H", [QubitId(qubit_id)])
    
    def create_pauli_x_gate(qubit_id):
        return QuantumGate("X", [QubitId(qubit_id)])
    
    def create_pauli_y_gate(qubit_id):
        return QuantumGate("Y", [QubitId(qubit_id)])
    
    def create_pauli_z_gate(qubit_id):
        return QuantumGate("Z", [QubitId(qubit_id)])
    
    def create_rotation_x_gate(qubit_id, angle):
        return QuantumGate("RX", [QubitId(qubit_id)], [angle])
    
    def create_rotation_y_gate(qubit_id, angle):
        return QuantumGate("RY", [QubitId(qubit_id)], [angle])
    
    def create_rotation_z_gate(qubit_id, angle):
        return QuantumGate("RZ", [QubitId(qubit_id)], [angle])
    
    def create_cnot_gate(control, target):
        return QuantumGate("CNOT", [QubitId(control), QubitId(target)])
    
    def create_phase_gate(qubit_id, angle):
        return QuantumGate("Phase", [QubitId(qubit_id)], [angle])
    
    def create_s_gate(qubit_id):
        return QuantumGate("S", [QubitId(qubit_id)])
    
    def create_t_gate(qubit_id):
        return QuantumGate("T", [QubitId(qubit_id)])
    
    def create_identity_gate(qubit_id):
        return QuantumGate("I", [QubitId(qubit_id)])
    
    # Stub other classes with basic functionality
    class VariationalCircuit:
        def __init__(self, num_qubits):
            self.num_qubits = num_qubits
            self.num_parameters = 0
        def add_rotation_layer(self, rotation_type):
            pass
        def add_entangling_layer(self):
            pass
    
    class SingleQubitDecomposition:
        def __init__(self):
            self.theta1 = 0.0
            self.phi = 0.0
            self.theta2 = 0.0
            self.global_phase = 0.0
    
    class CartanDecomposition:
        def __init__(self):
            self.xx_coefficient = 0.0
            self.yy_coefficient = 0.0
            self.zz_coefficient = 0.0
            self.cnot_count = 0
    
    def decompose_single_qubit(matrix):
        return SingleQubitDecomposition()
    
    def decompose_two_qubit_cartan(matrix):
        return CartanDecomposition()
    
    # Stub other functionality
    _native_core = None

# Export all public symbols
__all__ = [
    # Core types
    'QubitId',
    'QuantumGate', 
    'VariationalCircuit',
    
    # Gate creation functions
    'create_hadamard_gate',
    'create_pauli_x_gate',
    'create_pauli_y_gate',
    'create_pauli_z_gate',
    'create_rotation_x_gate',
    'create_rotation_y_gate',
    'create_rotation_z_gate',
    'create_cnot_gate',
    'create_phase_gate',
    'create_s_gate',
    'create_t_gate',
    'create_identity_gate',
    
    # Decomposition functions
    'decompose_single_qubit',
    'decompose_two_qubit_cartan',
    'SingleQubitDecomposition',
    'CartanDecomposition',
]

# Add conditional exports for native-only features
if _native_core is not None:
    __all__.extend([
        # Quantum sensor network
        'QuantumSensorNetwork',
        
        # Quantum internet
        'QuantumInternet',
        
        # Visualization tools
        'QuantumCircuitVisualizer',
        'QuantumStateVisualizer', 
        'QuantumPerformanceMonitor',
        
        # Real-time monitoring
        'MonitoringConfig',
        'RealtimeMonitor',
        'MonitoringStatus',
        'SystemMetric',
        'AggregatedStats',
        'OptimizationRecommendation',
        'Alert',
        
        # NumRS2 integration
        'NumRS2Array',
        'numrs2_zeros',
        'numrs2_ones',
        'numrs2_from_vec',
        
        # Quantum complexity analysis
        'QuantumComplexityAnalyzer',
        'analyze_algorithm_complexity',
        'compare_quantum_classical_complexity',
        'calculate_theoretical_quantum_volume',
    ])