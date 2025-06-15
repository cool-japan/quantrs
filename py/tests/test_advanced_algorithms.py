#!/usr/bin/env python3
"""
Tests for the advanced quantum algorithms module.
"""

import pytest
import numpy as np
import math
from unittest.mock import patch, MagicMock

# Safe import pattern
try:
    from quantrs2.advanced_algorithms import (
        AnsatzType,
        OptimizationMethod,
        AlgorithmResult,
        AdvancedVQE,
        AdvancedQAOA,
        QuantumWalkAlgorithms,
        QuantumErrorCorrection,
        QuantumTeleportation,
        ShorsAlgorithm,
        QuantumSimulatedAnnealing,
        create_advanced_vqe,
        create_advanced_qaoa,
        create_quantum_walk,
        create_error_correction_circuit,
        create_shors_circuit,
        create_teleportation_circuit,
        create_entangling_layer,
        create_rotation_layer
    )
    HAS_ADVANCED_ALGORITHMS = True
except ImportError:
    HAS_ADVANCED_ALGORITHMS = False
    
    # Stub implementations
    class AnsatzType:
        HARDWARE_EFFICIENT = "hardware_efficient"
        REAL_AMPLITUDES = "real_amplitudes"
        UCCSD = "uccsd"
    
    class OptimizationMethod:
        COBYLA = "cobyla"
        NELDER_MEAD = "nelder_mead"
    
    class AlgorithmResult:
        def __init__(self, success=False, optimal_value=None, optimal_parameters=None,
                     iteration_count=0, function_evaluations=0, convergence_data=None,
                     execution_time=None, metadata=None):
            self.success = success
            self.optimal_value = optimal_value
            self.optimal_parameters = optimal_parameters
            self.iteration_count = iteration_count
            self.function_evaluations = function_evaluations
            self.convergence_data = convergence_data
            self.execution_time = execution_time
            self.metadata = metadata
    
    class AdvancedVQE:
        def __init__(self, n_qubits, ansatz=None, optimizer=None, max_iterations=100):
            self.n_qubits = n_qubits
            self.ansatz = ansatz or AnsatzType.HARDWARE_EFFICIENT
            self.optimizer = optimizer or OptimizationMethod.COBYLA
            self.max_iterations = max_iterations
        
        def get_parameter_count(self, reps=1):
            if self.ansatz == AnsatzType.HARDWARE_EFFICIENT:
                return self.n_qubits * 2 * reps
            elif self.ansatz == AnsatzType.REAL_AMPLITUDES:
                return self.n_qubits * reps
            else:  # UCCSD
                return max(1, self.n_qubits * 2)
        
        def create_ansatz_circuit(self, parameters, reps=1):
            return MockCircuit(self.n_qubits)
    
    class AdvancedQAOA:
        def __init__(self, n_qubits, p_layers=1, problem_type="maxcut", mixer_type="x_mixer"):
            self.n_qubits = n_qubits
            self.p_layers = p_layers
            self.problem_type = problem_type
            self.mixer_type = mixer_type
        
        def create_qaoa_circuit(self, problem_instance, parameters):
            return MockCircuit(self.n_qubits)
    
    class QuantumWalkAlgorithms:
        @staticmethod
        def continuous_time_quantum_walk(n_qubits, adjacency_matrix, time, initial_state=0):
            return MockCircuit(n_qubits)
        
        @staticmethod
        def discrete_time_quantum_walk(n_position_qubits, n_coin_qubits, steps, coin_operator="hadamard"):
            return MockCircuit(n_position_qubits + n_coin_qubits)
    
    class QuantumErrorCorrection:
        @staticmethod
        def three_qubit_repetition_code(data_state):
            return MockCircuit(5)
        
        @staticmethod
        def steane_code():
            return MockCircuit(7)
        
        @staticmethod
        def surface_code_patch(distance):
            if distance % 2 == 0:
                raise ValueError("Distance must be odd")
            return MockCircuit(distance * distance)
    
    class QuantumTeleportation:
        @staticmethod
        def teleportation_circuit():
            return MockCircuit(3)
    
    class ShorsAlgorithm:
        def __init__(self, N):
            self.N = N
            self.n_qubits = max(8, int(np.log2(N)) * 2 + 4)
        
        def create_shor_circuit(self, a):
            return MockCircuit(self.n_qubits)
    
    class QuantumSimulatedAnnealing:
        def __init__(self, n_qubits, initial_temp=1.0, final_temp=0.01, n_steps=100):
            self.n_qubits = n_qubits
            self.initial_temp = initial_temp
            self.final_temp = final_temp
            self.n_steps = n_steps
        
        def create_annealing_circuit(self, problem_hamiltonian):
            return MockCircuit(self.n_qubits)
    
    def create_advanced_vqe(n_qubits, ansatz=None, **kwargs):
        return AdvancedVQE(n_qubits, ansatz, **kwargs)
    
    def create_advanced_qaoa(n_qubits, p_layers=1, problem_type="maxcut", **kwargs):
        return AdvancedQAOA(n_qubits, p_layers, problem_type, **kwargs)
    
    def create_quantum_walk(walk_type, **kwargs):
        if walk_type == "continuous":
            n_qubits = kwargs.get("n_qubits", 3)
            return QuantumWalkAlgorithms.continuous_time_quantum_walk(
                n_qubits, 
                kwargs.get("adjacency_matrix", np.eye(n_qubits)),
                kwargs.get("time", 1.0)
            )
        elif walk_type == "discrete":
            return QuantumWalkAlgorithms.discrete_time_quantum_walk(
                kwargs.get("n_position_qubits", 2),
                kwargs.get("n_coin_qubits", 1),
                kwargs.get("steps", 3)
            )
        else:
            raise ValueError("Unknown walk type")
    
    def create_error_correction_circuit(code_type, **kwargs):
        if code_type == "repetition":
            return QuantumErrorCorrection.three_qubit_repetition_code(
                kwargs.get("data_qubit_state", [1.0, 0.0])
            )
        elif code_type == "steane":
            return QuantumErrorCorrection.steane_code()
        elif code_type == "surface":
            return QuantumErrorCorrection.surface_code_patch(kwargs.get("distance", 3))
        else:
            raise ValueError("Unknown error correction code")
    
    def create_shors_circuit(N, a):
        shor = ShorsAlgorithm(N)
        return shor.create_shor_circuit(a)
    
    def create_teleportation_circuit():
        return QuantumTeleportation.teleportation_circuit()
    
    def create_entangling_layer(circuit, qubits, gate_type):
        if gate_type == "cnot":
            for i in range(len(qubits) - 1):
                circuit.cnot(qubits[i], qubits[i + 1])
        elif gate_type == "cz":
            for i in range(len(qubits) - 1):
                circuit.cz(qubits[i], qubits[i + 1])
        elif gate_type == "circular":
            for i in range(len(qubits)):
                circuit.cnot(qubits[i], qubits[(i + 1) % len(qubits)])
    
    def create_rotation_layer(circuit, qubits, parameters, gate_type):
        if len(parameters) < len(qubits):
            raise ValueError("Not enough parameters")
        
        for i, qubit in enumerate(qubits):
            if gate_type == "ry":
                circuit.ry(qubit, parameters[i])
            elif gate_type == "rz":
                circuit.rz(qubit, parameters[i])
            elif gate_type == "rx":
                circuit.rx(qubit, parameters[i])


# Mock Circuit class for testing
class MockCircuit:
    """Mock circuit class for testing."""
    
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.operations = []
    
    def h(self, qubit):
        self.operations.append(("h", qubit))
        return self
    
    def x(self, qubit):
        self.operations.append(("x", qubit))
        return self
    
    def y(self, qubit):
        self.operations.append(("y", qubit))
        return self
    
    def z(self, qubit):
        self.operations.append(("z", qubit))
        return self
    
    def cnot(self, control, target):
        self.operations.append(("cnot", control, target))
        return self
    
    def cz(self, control, target):
        self.operations.append(("cz", control, target))
        return self
    
    def crz(self, control, target, angle):
        self.operations.append(("crz", control, target, angle))
        return self
    
    def rx(self, qubit, angle):
        self.operations.append(("rx", qubit, angle))
        return self
    
    def ry(self, qubit, angle):
        self.operations.append(("ry", qubit, angle))
        return self
    
    def rz(self, qubit, angle):
        self.operations.append(("rz", qubit, angle))
        return self
    
    def swap(self, qubit1, qubit2):
        self.operations.append(("swap", qubit1, qubit2))
        return self
    
    def toffoli(self, control1, control2, target):
        self.operations.append(("toffoli", control1, control2, target))
        return self


@pytest.fixture
def mock_circuit():
    """Fixture providing mock circuit."""
    with patch('quantrs2.advanced_algorithms.Circuit', MockCircuit):
        yield MockCircuit


@pytest.mark.skipif(not HAS_ADVANCED_ALGORITHMS, reason="quantrs2.advanced_algorithms not available")
class TestAdvancedVQE:
    """Test cases for AdvancedVQE."""
    
    def test_initialization(self):
        """Test VQE initialization."""
        vqe = AdvancedVQE(
            n_qubits=4,
            ansatz=AnsatzType.HARDWARE_EFFICIENT,
            optimizer=OptimizationMethod.COBYLA,
            max_iterations=500
        )
        
        assert vqe.n_qubits == 4
        assert vqe.ansatz == AnsatzType.HARDWARE_EFFICIENT
        assert vqe.optimizer == OptimizationMethod.COBYLA
        assert vqe.max_iterations == 500
    
    def test_parameter_count_hardware_efficient(self):
        """Test parameter count for hardware efficient ansatz."""
        vqe = AdvancedVQE(4, ansatz=AnsatzType.HARDWARE_EFFICIENT)
        
        # Hardware efficient: 2 parameters per qubit per repetition
        assert vqe.get_parameter_count(reps=1) == 8
        assert vqe.get_parameter_count(reps=2) == 16
    
    def test_parameter_count_real_amplitudes(self):
        """Test parameter count for real amplitudes ansatz."""
        vqe = AdvancedVQE(4, ansatz=AnsatzType.REAL_AMPLITUDES)
        
        # Real amplitudes: 1 parameter per qubit per repetition
        assert vqe.get_parameter_count(reps=1) == 4
        assert vqe.get_parameter_count(reps=2) == 8
    
    def test_parameter_count_uccsd(self):
        """Test parameter count for UCCSD ansatz."""
        vqe = AdvancedVQE(4, ansatz=AnsatzType.UCCSD)
        
        # UCCSD parameter count depends on number of excitations
        param_count = vqe.get_parameter_count()
        assert param_count > 0
    
    def test_create_ansatz_circuit_hardware_efficient(self, mock_circuit):
        """Test creation of hardware efficient ansatz circuit."""
        vqe = AdvancedVQE(3, ansatz=AnsatzType.HARDWARE_EFFICIENT)
        parameters = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # 2 params per qubit
        
        circuit = vqe.create_ansatz_circuit(parameters, reps=1)
        
        assert isinstance(circuit, MockCircuit)
        assert circuit.n_qubits == 3
        
        # Check that we have rotation gates
        rotation_ops = [op for op in circuit.operations if op[0] in ['ry', 'rz']]
        assert len(rotation_ops) > 0
        
        # Check that we have entangling gates
        cnot_ops = [op for op in circuit.operations if op[0] == 'cnot']
        assert len(cnot_ops) > 0
    
    def test_create_ansatz_circuit_real_amplitudes(self, mock_circuit):
        """Test creation of real amplitudes ansatz circuit."""
        vqe = AdvancedVQE(3, ansatz=AnsatzType.REAL_AMPLITUDES)
        parameters = [0.1, 0.2, 0.3]  # 1 param per qubit
        
        circuit = vqe.create_ansatz_circuit(parameters, reps=1)
        
        # Should only have RY rotations (no RZ for real amplitudes)
        ry_ops = [op for op in circuit.operations if op[0] == 'ry']
        rz_ops = [op for op in circuit.operations if op[0] == 'rz']
        
        assert len(ry_ops) == 3
        assert len(rz_ops) == 0
    
    def test_create_ansatz_circuit_uccsd(self, mock_circuit):
        """Test creation of UCCSD ansatz circuit."""
        vqe = AdvancedVQE(4, ansatz=AnsatzType.UCCSD)
        param_count = vqe.get_parameter_count()
        parameters = [0.1] * param_count
        
        circuit = vqe.create_ansatz_circuit(parameters)
        
        assert isinstance(circuit, MockCircuit)
        assert circuit.n_qubits == 4
    
    def test_unsupported_ansatz(self, mock_circuit):
        """Test error for unsupported ansatz type."""
        vqe = AdvancedVQE(3)
        vqe.ansatz = "unsupported_ansatz"
        
        with pytest.raises(ValueError, match="Unsupported ansatz type"):
            vqe.create_ansatz_circuit([0.1, 0.2, 0.3])


@pytest.mark.skipif(not HAS_ADVANCED_ALGORITHMS, reason="quantrs2.advanced_algorithms not available")
class TestAdvancedQAOA:
    """Test cases for AdvancedQAOA."""
    
    def test_initialization(self):
        """Test QAOA initialization."""
        qaoa = AdvancedQAOA(
            n_qubits=5,
            p_layers=2,
            problem_type="maxcut",
            mixer_type="x_mixer"
        )
        
        assert qaoa.n_qubits == 5
        assert qaoa.p_layers == 2
        assert qaoa.problem_type == "maxcut"
        assert qaoa.mixer_type == "x_mixer"
    
    def test_create_qaoa_circuit_maxcut(self, mock_circuit):
        """Test QAOA circuit creation for MaxCut."""
        qaoa = AdvancedQAOA(4, p_layers=1, problem_type="maxcut")
        
        problem_instance = {
            "edges": [(0, 1), (1, 2), (2, 3)],
            "weights": [1.0, 1.0, 1.0]
        }
        parameters = [0.5, 0.3]  # [gamma, beta]
        
        circuit = qaoa.create_qaoa_circuit(problem_instance, parameters)
        
        assert isinstance(circuit, MockCircuit)
        assert circuit.n_qubits == 4
        
        # Check initial state preparation (Hadamards)
        h_ops = [op for op in circuit.operations if op[0] == 'h']
        assert len(h_ops) >= 4  # Initial Hadamards
        
        # Check cost Hamiltonian (CNOTs and RZ for edges)
        cnot_ops = [op for op in circuit.operations if op[0] == 'cnot']
        rz_ops = [op for op in circuit.operations if op[0] == 'rz']
        
        assert len(cnot_ops) >= 6  # 2 per edge
        assert len(rz_ops) >= 3  # 1 per edge
    
    def test_create_qaoa_circuit_max_k_sat(self, mock_circuit):
        """Test QAOA circuit creation for MAX-k-SAT."""
        qaoa = AdvancedQAOA(3, p_layers=1, problem_type="max_k_sat")
        
        problem_instance = {
            "clauses": [[1, -2, 3], [-1, 2, -3]]  # Two 3-SAT clauses
        }
        parameters = [0.4, 0.2]
        
        circuit = qaoa.create_qaoa_circuit(problem_instance, parameters)
        
        assert isinstance(circuit, MockCircuit)
        assert circuit.n_qubits == 3
    
    def test_create_qaoa_circuit_number_partitioning(self, mock_circuit):
        """Test QAOA circuit creation for number partitioning."""
        qaoa = AdvancedQAOA(4, p_layers=1, problem_type="number_partitioning")
        
        problem_instance = {
            "numbers": [3, 1, 1, 2]
        }
        parameters = [0.6, 0.1]
        
        circuit = qaoa.create_qaoa_circuit(problem_instance, parameters)
        
        assert isinstance(circuit, MockCircuit)
        assert circuit.n_qubits == 4
    
    def test_xy_mixer(self, mock_circuit):
        """Test XY mixer application."""
        qaoa = AdvancedQAOA(4, p_layers=1, mixer_type="xy_mixer")
        
        problem_instance = {"edges": [(0, 1)], "weights": [1.0]}
        parameters = [0.3, 0.4]
        
        circuit = qaoa.create_qaoa_circuit(problem_instance, parameters)
        
        # XY mixer should have RY gates
        ry_ops = [op for op in circuit.operations if op[0] == 'ry']
        assert len(ry_ops) > 0
    
    def test_unknown_problem_type(self, mock_circuit):
        """Test error for unknown problem type."""
        qaoa = AdvancedQAOA(3, problem_type="unknown_problem")
        
        with pytest.raises(ValueError, match="Unknown problem type"):
            qaoa.create_qaoa_circuit({}, [0.1, 0.2])


@pytest.mark.skipif(not HAS_ADVANCED_ALGORITHMS, reason="quantrs2.advanced_algorithms not available")
class TestQuantumWalkAlgorithms:
    """Test cases for quantum walk algorithms."""
    
    def test_continuous_time_quantum_walk(self, mock_circuit):
        """Test continuous-time quantum walk."""
        n_qubits = 3
        adjacency_matrix = np.array([
            [0, 1, 1],
            [1, 0, 1], 
            [1, 1, 0]
        ])
        time = 1.0
        
        circuit = QuantumWalkAlgorithms.continuous_time_quantum_walk(
            n_qubits, adjacency_matrix, time, initial_state=0
        )
        
        assert isinstance(circuit, MockCircuit)
        assert circuit.n_qubits == n_qubits
    
    def test_discrete_time_quantum_walk(self, mock_circuit):
        """Test discrete-time quantum walk."""
        circuit = QuantumWalkAlgorithms.discrete_time_quantum_walk(
            n_position_qubits=3,
            n_coin_qubits=1,
            steps=5,
            coin_operator="hadamard"
        )
        
        assert isinstance(circuit, MockCircuit)
        assert circuit.n_qubits == 4  # 3 position + 1 coin
        
        # Check for Hadamard gates (coin operations)
        h_ops = [op for op in circuit.operations if op[0] == 'h']
        assert len(h_ops) > 0


@pytest.mark.skipif(not HAS_ADVANCED_ALGORITHMS, reason="quantrs2.advanced_algorithms not available")
class TestQuantumErrorCorrection:
    """Test cases for quantum error correction."""
    
    def test_three_qubit_repetition_code(self, mock_circuit):
        """Test 3-qubit repetition code."""
        data_state = [np.sqrt(0.6), np.sqrt(0.4)]  # |ψ⟩ = √0.6|0⟩ + √0.4|1⟩
        
        circuit = QuantumErrorCorrection.three_qubit_repetition_code(data_state)
        
        assert isinstance(circuit, MockCircuit)
        assert circuit.n_qubits == 5  # 3 data + 2 syndrome
        
        # Check for encoding CNOTs
        cnot_ops = [op for op in circuit.operations if op[0] == 'cnot']
        assert len(cnot_ops) >= 4  # Encoding + syndrome measurement
    
    def test_steane_code(self, mock_circuit):
        """Test Steane code encoding."""
        circuit = QuantumErrorCorrection.steane_code()
        
        assert isinstance(circuit, MockCircuit)
        assert circuit.n_qubits == 7
        
        # Check for Hadamards and CNOTs
        h_ops = [op for op in circuit.operations if op[0] == 'h']
        cnot_ops = [op for op in circuit.operations if op[0] == 'cnot']
        
        assert len(h_ops) >= 3
        assert len(cnot_ops) >= 6
    
    def test_surface_code_patch(self, mock_circuit):
        """Test surface code patch."""
        distance = 3
        circuit = QuantumErrorCorrection.surface_code_patch(distance)
        
        assert isinstance(circuit, MockCircuit)
        assert circuit.n_qubits == distance * distance
    
    def test_surface_code_even_distance_error(self):
        """Test error for even distance in surface code."""
        with pytest.raises(ValueError, match="Distance must be odd"):
            QuantumErrorCorrection.surface_code_patch(4)


@pytest.mark.skipif(not HAS_ADVANCED_ALGORITHMS, reason="quantrs2.advanced_algorithms not available")
class TestQuantumTeleportation:
    """Test cases for quantum teleportation."""
    
    def test_teleportation_circuit(self, mock_circuit):
        """Test quantum teleportation circuit."""
        circuit = QuantumTeleportation.teleportation_circuit()
        
        assert isinstance(circuit, MockCircuit)
        assert circuit.n_qubits == 3
        
        # Check for entanglement preparation
        h_ops = [op for op in circuit.operations if op[0] == 'h']
        cnot_ops = [op for op in circuit.operations if op[0] == 'cnot']
        cz_ops = [op for op in circuit.operations if op[0] == 'cz']
        
        assert len(h_ops) >= 2  # At least one for entanglement prep + one for measurement
        assert len(cnot_ops) >= 2  # Entanglement + teleportation
        assert len(cz_ops) >= 1  # Conditional operation


@pytest.mark.skipif(not HAS_ADVANCED_ALGORITHMS, reason="quantrs2.advanced_algorithms not available")
class TestShorsAlgorithm:
    """Test cases for Shor's algorithm."""
    
    def test_initialization(self):
        """Test Shor's algorithm initialization."""
        N = 15
        shor = ShorsAlgorithm(N)
        
        assert shor.N == N
        assert shor.n_qubits >= 8
    
    def test_create_shor_circuit(self, mock_circuit):
        """Test Shor's algorithm circuit creation."""
        N = 15
        a = 7
        shor = ShorsAlgorithm(N)
        
        circuit = shor.create_shor_circuit(a)
        
        assert isinstance(circuit, MockCircuit)
        assert circuit.n_qubits == shor.n_qubits
        
        # Check for initial superposition
        h_ops = [op for op in circuit.operations if op[0] == 'h']
        assert len(h_ops) > 0
        
        # Check for initial state preparation
        x_ops = [op for op in circuit.operations if op[0] == 'x']
        assert len(x_ops) >= 1


@pytest.mark.skipif(not HAS_ADVANCED_ALGORITHMS, reason="quantrs2.advanced_algorithms not available")
class TestQuantumSimulatedAnnealing:
    """Test cases for quantum simulated annealing."""
    
    def test_initialization(self):
        """Test quantum simulated annealing initialization."""
        qsa = QuantumSimulatedAnnealing(
            n_qubits=4,
            initial_temp=2.0,
            final_temp=0.05,
            n_steps=50
        )
        
        assert qsa.n_qubits == 4
        assert qsa.initial_temp == 2.0
        assert qsa.final_temp == 0.05
        assert qsa.n_steps == 50
    
    def test_create_annealing_circuit(self, mock_circuit):
        """Test quantum annealing circuit creation."""
        qsa = QuantumSimulatedAnnealing(3, n_steps=10)
        
        problem_hamiltonian = {
            "edges": [(0, 1), (1, 2)],
            "fields": [0.5, -0.3, 0.1]
        }
        
        circuit = qsa.create_annealing_circuit(problem_hamiltonian)
        
        assert isinstance(circuit, MockCircuit)
        assert circuit.n_qubits == 3
        
        # Check for initial superposition
        h_ops = [op for op in circuit.operations if op[0] == 'h']
        assert len(h_ops) >= 3
        
        # Check for evolution operations
        rx_ops = [op for op in circuit.operations if op[0] == 'rx']
        rz_ops = [op for op in circuit.operations if op[0] == 'rz']
        
        assert len(rx_ops) > 0  # Transverse field
        assert len(rz_ops) > 0  # Problem Hamiltonian


@pytest.mark.skipif(not HAS_ADVANCED_ALGORITHMS, reason="quantrs2.advanced_algorithms not available")
class TestFactoryFunctions:
    """Test cases for factory functions."""
    
    def test_create_advanced_vqe(self):
        """Test VQE creation function."""
        vqe = create_advanced_vqe(4, ansatz=AnsatzType.REAL_AMPLITUDES)
        
        assert isinstance(vqe, AdvancedVQE)
        assert vqe.n_qubits == 4
        assert vqe.ansatz == AnsatzType.REAL_AMPLITUDES
    
    def test_create_advanced_qaoa(self):
        """Test QAOA creation function."""
        qaoa = create_advanced_qaoa(5, p_layers=3, problem_type="maxcut")
        
        assert isinstance(qaoa, AdvancedQAOA)
        assert qaoa.n_qubits == 5
        assert qaoa.p_layers == 3
        assert qaoa.problem_type == "maxcut"
    
    def test_create_quantum_walk_continuous(self, mock_circuit):
        """Test continuous quantum walk creation."""
        adj_matrix = np.eye(3)
        circuit = create_quantum_walk(
            "continuous",
            n_qubits=3,
            adjacency_matrix=adj_matrix,
            time=1.0
        )
        
        assert isinstance(circuit, MockCircuit)
    
    def test_create_quantum_walk_discrete(self, mock_circuit):
        """Test discrete quantum walk creation."""
        circuit = create_quantum_walk(
            "discrete",
            n_position_qubits=2,
            n_coin_qubits=1,
            steps=3
        )
        
        assert isinstance(circuit, MockCircuit)
    
    def test_create_quantum_walk_unknown_type(self):
        """Test error for unknown walk type."""
        with pytest.raises(ValueError, match="Unknown walk type"):
            create_quantum_walk("unknown_type")
    
    def test_create_error_correction_circuit_repetition(self, mock_circuit):
        """Test repetition code creation."""
        circuit = create_error_correction_circuit(
            "repetition",
            data_qubit_state=[1.0, 0.0]
        )
        
        assert isinstance(circuit, MockCircuit)
    
    def test_create_error_correction_circuit_steane(self, mock_circuit):
        """Test Steane code creation."""
        circuit = create_error_correction_circuit("steane")
        
        assert isinstance(circuit, MockCircuit)
    
    def test_create_error_correction_circuit_surface(self, mock_circuit):
        """Test surface code creation."""
        circuit = create_error_correction_circuit("surface", distance=3)
        
        assert isinstance(circuit, MockCircuit)
    
    def test_create_error_correction_circuit_unknown(self):
        """Test error for unknown error correction code."""
        with pytest.raises(ValueError, match="Unknown error correction code"):
            create_error_correction_circuit("unknown_code")
    
    def test_create_shors_circuit(self, mock_circuit):
        """Test Shor's circuit creation function."""
        circuit = create_shors_circuit(15, 7)
        
        assert isinstance(circuit, MockCircuit)
    
    def test_create_teleportation_circuit(self, mock_circuit):
        """Test teleportation circuit creation function."""
        circuit = create_teleportation_circuit()
        
        assert isinstance(circuit, MockCircuit)


@pytest.mark.skipif(not HAS_ADVANCED_ALGORITHMS, reason="quantrs2.advanced_algorithms not available")
class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_create_entangling_layer_cnot(self, mock_circuit):
        """Test entangling layer with CNOT gates."""
        circuit = MockCircuit(4)
        qubits = [0, 1, 2, 3]
        
        create_entangling_layer(circuit, qubits, "cnot")
        
        cnot_ops = [op for op in circuit.operations if op[0] == 'cnot']
        assert len(cnot_ops) == 3  # n-1 CNOTs
    
    def test_create_entangling_layer_cz(self, mock_circuit):
        """Test entangling layer with CZ gates."""
        circuit = MockCircuit(3)
        qubits = [0, 1, 2]
        
        create_entangling_layer(circuit, qubits, "cz")
        
        cz_ops = [op for op in circuit.operations if op[0] == 'cz']
        assert len(cz_ops) == 2
    
    def test_create_entangling_layer_circular(self, mock_circuit):
        """Test entangling layer with circular connectivity."""
        circuit = MockCircuit(4)
        qubits = [0, 1, 2, 3]
        
        create_entangling_layer(circuit, qubits, "circular")
        
        cnot_ops = [op for op in circuit.operations if op[0] == 'cnot']
        assert len(cnot_ops) == 4  # n CNOTs (including wrap-around)
    
    def test_create_rotation_layer_ry(self, mock_circuit):
        """Test rotation layer with RY gates."""
        circuit = MockCircuit(3)
        qubits = [0, 1, 2]
        parameters = [0.1, 0.2, 0.3]
        
        create_rotation_layer(circuit, qubits, parameters, "ry")
        
        ry_ops = [op for op in circuit.operations if op[0] == 'ry']
        assert len(ry_ops) == 3
        
        # Check parameters
        assert ry_ops[0] == ("ry", 0, 0.1)
        assert ry_ops[1] == ("ry", 1, 0.2)
        assert ry_ops[2] == ("ry", 2, 0.3)
    
    def test_create_rotation_layer_rz(self, mock_circuit):
        """Test rotation layer with RZ gates."""
        circuit = MockCircuit(2)
        qubits = [0, 1]
        parameters = [0.5, 0.6]
        
        create_rotation_layer(circuit, qubits, parameters, "rz")
        
        rz_ops = [op for op in circuit.operations if op[0] == 'rz']
        assert len(rz_ops) == 2
    
    def test_create_rotation_layer_rx(self, mock_circuit):
        """Test rotation layer with RX gates."""
        circuit = MockCircuit(2)
        qubits = [0, 1]
        parameters = [0.7, 0.8]
        
        create_rotation_layer(circuit, qubits, parameters, "rx")
        
        rx_ops = [op for op in circuit.operations if op[0] == 'rx']
        assert len(rx_ops) == 2
    
    def test_create_rotation_layer_insufficient_parameters(self, mock_circuit):
        """Test error for insufficient parameters."""
        circuit = MockCircuit(3)
        qubits = [0, 1, 2]
        parameters = [0.1, 0.2]  # Only 2 parameters for 3 qubits
        
        with pytest.raises(ValueError, match="Not enough parameters"):
            create_rotation_layer(circuit, qubits, parameters, "ry")


@pytest.mark.skipif(not HAS_ADVANCED_ALGORITHMS, reason="quantrs2.advanced_algorithms not available")
class TestAlgorithmResult:
    """Test cases for AlgorithmResult dataclass."""
    
    def test_algorithm_result_creation(self):
        """Test AlgorithmResult creation."""
        result = AlgorithmResult(
            success=True,
            optimal_value=1.5,
            optimal_parameters=[0.1, 0.2, 0.3],
            iteration_count=50,
            function_evaluations=200
        )
        
        assert result.success is True
        assert result.optimal_value == 1.5
        assert result.optimal_parameters == [0.1, 0.2, 0.3]
        assert result.iteration_count == 50
        assert result.function_evaluations == 200
    
    def test_algorithm_result_defaults(self):
        """Test AlgorithmResult with default values."""
        result = AlgorithmResult(success=False)
        
        assert result.success is False
        assert result.optimal_value is None
        assert result.optimal_parameters is None
        assert result.iteration_count == 0
        assert result.function_evaluations == 0
        assert result.convergence_data is None
        assert result.execution_time is None
        assert result.metadata is None


if __name__ == "__main__":
    pytest.main([__file__])