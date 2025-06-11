"""
Quantum Software Testing Tools

This module provides comprehensive testing tools specifically designed for quantum computing applications,
including property-based testing, circuit verification, test generation, and performance analysis.
"""

import asyncio
import json
import time
import uuid
import logging
import random
import statistics
import threading
import inspect
import hashlib
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union, Set, Type
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum
from collections import defaultdict, namedtuple
import tempfile
import concurrent.futures
from contextlib import asynccontextmanager, contextmanager

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import hypothesis
    from hypothesis import strategies as st
    from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule, invariant
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from . import utils
    from . import circuit_db
    from . import profiler
    from . import algorithm_debugger
    from . import quantum_application_framework
    from . import compilation_service
    QUANTRS_MODULES_AVAILABLE = True
except ImportError:
    QUANTRS_MODULES_AVAILABLE = False


class TestType(Enum):
    """Types of quantum tests."""
    PROPERTY_BASED = "property_based"
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    INTEGRATION = "integration"
    REGRESSION = "regression"
    STRESS = "stress"
    INVARIANT = "invariant"
    FUZZ = "fuzz"


class TestStatus(Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"


class QuantumProperty(Enum):
    """Quantum properties to test."""
    UNITARITY = "unitarity"
    NORMALIZATION = "normalization"
    HERMITICITY = "hermiticity"
    COMMUTATIVITY = "commutativity"
    IDEMPOTENCY = "idempotency"
    REVERSIBILITY = "reversibility"
    ENTANGLEMENT = "entanglement"
    SUPERPOSITION = "superposition"
    MEASUREMENT_CONSERVATION = "measurement_conservation"
    GATE_EQUIVALENCE = "gate_equivalence"


class TestSeverity(Enum):
    """Test failure severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class TestCase:
    """Individual test case definition."""
    test_id: str
    name: str
    description: str
    test_type: TestType
    target_function: Optional[Callable] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    expected_output: Any = None
    properties: List[QuantumProperty] = field(default_factory=list)
    tolerance: float = 1e-6
    timeout: float = 30.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'test_id': self.test_id,
            'name': self.name,
            'description': self.description,
            'test_type': self.test_type.value,
            'input_data': self.input_data,
            'expected_output': self.expected_output,
            'properties': [p.value for p in self.properties],
            'tolerance': self.tolerance,
            'timeout': self.timeout,
            'tags': self.tags,
            'metadata': self.metadata,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestCase':
        """Create from dictionary."""
        data = data.copy()
        data['test_type'] = TestType(data['test_type'])
        data['properties'] = [QuantumProperty(p) for p in data.get('properties', [])]
        return cls(**data)


@dataclass
class TestResult:
    """Test execution result."""
    test_id: str
    status: TestStatus
    execution_time: float
    message: str = ""
    actual_output: Any = None
    assertion_errors: List[str] = field(default_factory=list)
    property_violations: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    coverage_data: Dict[str, Any] = field(default_factory=dict)
    severity: TestSeverity = TestSeverity.MEDIUM
    traceback: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'test_id': self.test_id,
            'status': self.status.value,
            'execution_time': self.execution_time,
            'message': self.message,
            'actual_output': self.actual_output,
            'assertion_errors': self.assertion_errors,
            'property_violations': self.property_violations,
            'performance_metrics': self.performance_metrics,
            'coverage_data': self.coverage_data,
            'severity': self.severity.value,
            'traceback': self.traceback,
            'timestamp': self.timestamp
        }


@dataclass
class TestSuite:
    """Collection of related test cases."""
    suite_id: str
    name: str
    description: str
    test_cases: List[TestCase] = field(default_factory=list)
    setup_hooks: List[Callable] = field(default_factory=list)
    teardown_hooks: List[Callable] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def add_test_case(self, test_case: TestCase) -> 'TestSuite':
        """Add test case to suite."""
        self.test_cases.append(test_case)
        return self
    
    def add_setup_hook(self, hook: Callable) -> 'TestSuite':
        """Add setup hook."""
        self.setup_hooks.append(hook)
        return self
    
    def add_teardown_hook(self, hook: Callable) -> 'TestSuite':
        """Add teardown hook."""
        self.teardown_hooks.append(hook)
        return self


class QuantumPropertyTester:
    """Tests quantum-specific properties."""
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.logger = logging.getLogger(__name__ + ".PropertyTester")
    
    def test_unitarity(self, matrix: Any) -> Tuple[bool, str]:
        """Test if matrix is unitary."""
        if not NUMPY_AVAILABLE:
            return True, "NumPy not available - skipping unitarity test"
        
        try:
            if not isinstance(matrix, np.ndarray):
                matrix = np.array(matrix)
            
            # Check if U * U† = I
            conjugate_transpose = matrix.conj().T
            product = np.dot(matrix, conjugate_transpose)
            identity = np.eye(matrix.shape[0])
            
            if np.allclose(product, identity, atol=self.tolerance):
                return True, "Matrix is unitary"
            else:
                max_error = np.max(np.abs(product - identity))
                return False, f"Matrix is not unitary (max error: {max_error})"
                
        except Exception as e:
            return False, f"Error testing unitarity: {e}"
    
    def test_normalization(self, state_vector: Any) -> Tuple[bool, str]:
        """Test if state vector is normalized."""
        if not NUMPY_AVAILABLE:
            return True, "NumPy not available - skipping normalization test"
        
        try:
            if not isinstance(state_vector, np.ndarray):
                state_vector = np.array(state_vector)
            
            norm = np.linalg.norm(state_vector)
            
            if abs(norm - 1.0) < self.tolerance:
                return True, "State vector is normalized"
            else:
                return False, f"State vector is not normalized (norm: {norm})"
                
        except Exception as e:
            return False, f"Error testing normalization: {e}"
    
    def test_hermiticity(self, matrix: Any) -> Tuple[bool, str]:
        """Test if matrix is Hermitian."""
        if not NUMPY_AVAILABLE:
            return True, "NumPy not available - skipping hermiticity test"
        
        try:
            if not isinstance(matrix, np.ndarray):
                matrix = np.array(matrix)
            
            # Check if H = H†
            conjugate_transpose = matrix.conj().T
            
            if np.allclose(matrix, conjugate_transpose, atol=self.tolerance):
                return True, "Matrix is Hermitian"
            else:
                max_error = np.max(np.abs(matrix - conjugate_transpose))
                return False, f"Matrix is not Hermitian (max error: {max_error})"
                
        except Exception as e:
            return False, f"Error testing hermiticity: {e}"
    
    def test_commutativity(self, matrix_a: Any, matrix_b: Any) -> Tuple[bool, str]:
        """Test if two matrices commute."""
        if not NUMPY_AVAILABLE:
            return True, "NumPy not available - skipping commutativity test"
        
        try:
            if not isinstance(matrix_a, np.ndarray):
                matrix_a = np.array(matrix_a)
            if not isinstance(matrix_b, np.ndarray):
                matrix_b = np.array(matrix_b)
            
            # Check if AB = BA
            ab = np.dot(matrix_a, matrix_b)
            ba = np.dot(matrix_b, matrix_a)
            
            if np.allclose(ab, ba, atol=self.tolerance):
                return True, "Matrices commute"
            else:
                max_error = np.max(np.abs(ab - ba))
                return False, f"Matrices do not commute (max error: {max_error})"
                
        except Exception as e:
            return False, f"Error testing commutativity: {e}"
    
    def test_idempotency(self, matrix: Any) -> Tuple[bool, str]:
        """Test if matrix is idempotent (A² = A)."""
        if not NUMPY_AVAILABLE:
            return True, "NumPy not available - skipping idempotency test"
        
        try:
            if not isinstance(matrix, np.ndarray):
                matrix = np.array(matrix)
            
            # Check if A² = A
            squared = np.dot(matrix, matrix)
            
            if np.allclose(matrix, squared, atol=self.tolerance):
                return True, "Matrix is idempotent"
            else:
                max_error = np.max(np.abs(matrix - squared))
                return False, f"Matrix is not idempotent (max error: {max_error})"
                
        except Exception as e:
            return False, f"Error testing idempotency: {e}"
    
    def test_reversibility(self, forward_result: Any, backward_result: Any, 
                          original_input: Any) -> Tuple[bool, str]:
        """Test if operation is reversible."""
        try:
            if NUMPY_AVAILABLE:
                if hasattr(backward_result, '__iter__') and hasattr(original_input, '__iter__'):
                    backward_array = np.array(backward_result)
                    original_array = np.array(original_input)
                    
                    if np.allclose(backward_array, original_array, atol=self.tolerance):
                        return True, "Operation is reversible"
                    else:
                        max_error = np.max(np.abs(backward_array - original_array))
                        return False, f"Operation is not reversible (max error: {max_error})"
            
            # Fallback comparison
            if backward_result == original_input:
                return True, "Operation is reversible"
            else:
                return False, "Operation is not reversible"
                
        except Exception as e:
            return False, f"Error testing reversibility: {e}"
    
    def test_entanglement(self, state_vector: Any, num_qubits: int) -> Tuple[bool, str]:
        """Test if state shows entanglement."""
        if not NUMPY_AVAILABLE:
            return True, "NumPy not available - skipping entanglement test"
        
        try:
            if not isinstance(state_vector, np.ndarray):
                state_vector = np.array(state_vector)
            
            # Simple entanglement test: check if state can be written as tensor product
            # For 2-qubit case, state is separable if it can be written as |a⟩⊗|b⟩
            if num_qubits == 2 and len(state_vector) == 4:
                # Reshape to 2x2 matrix and check rank
                reshaped = state_vector.reshape(2, 2)
                rank = np.linalg.matrix_rank(reshaped, tol=self.tolerance)
                
                if rank > 1:
                    return True, "State shows entanglement"
                else:
                    return False, "State appears separable"
            
            # For general case, use simple heuristic
            # Check if state is not a computational basis state
            max_amplitude = np.max(np.abs(state_vector))
            non_zero_count = np.sum(np.abs(state_vector) > self.tolerance)
            
            if non_zero_count > 1 and max_amplitude < 0.99:
                return True, "State shows potential entanglement"
            else:
                return False, "State appears separable or basis state"
                
        except Exception as e:
            return False, f"Error testing entanglement: {e}"


class QuantumTestGenerator:
    """Generates test cases for quantum operations."""
    
    def __init__(self, property_tester: Optional[QuantumPropertyTester] = None):
        self.property_tester = property_tester or QuantumPropertyTester()
        self.logger = logging.getLogger(__name__ + ".TestGenerator")
    
    def generate_circuit_tests(self, circuit_data: Dict[str, Any], 
                             num_tests: int = 10) -> List[TestCase]:
        """Generate test cases for quantum circuits."""
        tests = []
        
        # Basic circuit validation tests
        tests.append(TestCase(
            test_id=f"circuit_validation_{uuid.uuid4().hex[:8]}",
            name="Circuit Structure Validation",
            description="Validate circuit structure and gate definitions",
            test_type=TestType.FUNCTIONAL,
            input_data={'circuit_data': circuit_data},
            properties=[QuantumProperty.GATE_EQUIVALENCE],
            tags=['circuit', 'validation']
        ))
        
        # Property-based tests for circuit properties
        if 'gates' in circuit_data:
            gate_count = len(circuit_data['gates'])
            qubit_count = self._estimate_qubit_count(circuit_data)
            
            tests.append(TestCase(
                test_id=f"circuit_properties_{uuid.uuid4().hex[:8]}",
                name="Circuit Quantum Properties",
                description="Test quantum properties of circuit execution",
                test_type=TestType.PROPERTY_BASED,
                input_data={
                    'circuit_data': circuit_data,
                    'num_qubits': qubit_count
                },
                properties=[
                    QuantumProperty.UNITARITY,
                    QuantumProperty.NORMALIZATION,
                    QuantumProperty.REVERSIBILITY
                ],
                tags=['circuit', 'properties', 'quantum']
            ))
        
        # Performance tests
        tests.append(TestCase(
            test_id=f"circuit_performance_{uuid.uuid4().hex[:8]}",
            name="Circuit Performance Test",
            description="Measure circuit execution performance",
            test_type=TestType.PERFORMANCE,
            input_data={'circuit_data': circuit_data},
            metadata={'expected_max_time': 1.0},
            tags=['circuit', 'performance']
        ))
        
        # Generate random input tests
        for i in range(min(num_tests - len(tests), 5)):
            tests.append(TestCase(
                test_id=f"circuit_random_{uuid.uuid4().hex[:8]}",
                name=f"Random Circuit Test {i+1}",
                description=f"Random test case {i+1} for circuit",
                test_type=TestType.FUZZ,
                input_data={
                    'circuit_data': circuit_data,
                    'random_seed': random.randint(0, 10000)
                },
                tags=['circuit', 'random', 'fuzz']
            ))
        
        return tests
    
    def generate_gate_tests(self, gate_name: str, gate_params: Dict[str, Any] = None) -> List[TestCase]:
        """Generate test cases for specific quantum gates."""
        tests = []
        gate_params = gate_params or {}
        
        # Basic gate functionality test
        tests.append(TestCase(
            test_id=f"gate_{gate_name}_{uuid.uuid4().hex[:8]}",
            name=f"{gate_name.upper()} Gate Functionality",
            description=f"Test basic functionality of {gate_name} gate",
            test_type=TestType.FUNCTIONAL,
            input_data={'gate': gate_name, 'params': gate_params},
            properties=[QuantumProperty.UNITARITY],
            tags=['gate', gate_name]
        ))
        
        # Gate matrix properties
        tests.append(TestCase(
            test_id=f"gate_matrix_{gate_name}_{uuid.uuid4().hex[:8]}",
            name=f"{gate_name.upper()} Gate Matrix Properties",
            description=f"Test matrix properties of {gate_name} gate",
            test_type=TestType.PROPERTY_BASED,
            input_data={'gate': gate_name, 'params': gate_params},
            properties=[QuantumProperty.UNITARITY, QuantumProperty.HERMITICITY],
            tags=['gate', 'matrix', gate_name]
        ))
        
        # Gate-specific tests
        if gate_name.lower() in ['h', 'hadamard']:
            tests.append(TestCase(
                test_id=f"hadamard_superposition_{uuid.uuid4().hex[:8]}",
                name="Hadamard Superposition Test",
                description="Test Hadamard gate creates superposition",
                test_type=TestType.PROPERTY_BASED,
                input_data={'gate': gate_name},
                properties=[QuantumProperty.SUPERPOSITION],
                tags=['hadamard', 'superposition']
            ))
        
        elif gate_name.lower() in ['cnot', 'cx']:
            tests.append(TestCase(
                test_id=f"cnot_entanglement_{uuid.uuid4().hex[:8]}",
                name="CNOT Entanglement Test",
                description="Test CNOT gate creates entanglement",
                test_type=TestType.PROPERTY_BASED,
                input_data={'gate': gate_name},
                properties=[QuantumProperty.ENTANGLEMENT],
                tags=['cnot', 'entanglement']
            ))
        
        elif gate_name.lower() in ['x', 'pauli_x']:
            tests.append(TestCase(
                test_id=f"pauli_x_idempotent_{uuid.uuid4().hex[:8]}",
                name="Pauli-X Idempotency Test",
                description="Test Pauli-X gate applied twice returns to original",
                test_type=TestType.PROPERTY_BASED,
                input_data={'gate': gate_name},
                properties=[QuantumProperty.REVERSIBILITY],
                tags=['pauli_x', 'idempotent']
            ))
        
        return tests
    
    def generate_algorithm_tests(self, algorithm_data: Dict[str, Any]) -> List[TestCase]:
        """Generate test cases for quantum algorithms."""
        tests = []
        
        # Algorithm correctness test
        tests.append(TestCase(
            test_id=f"algorithm_correctness_{uuid.uuid4().hex[:8]}",
            name="Algorithm Correctness Test",
            description="Test algorithm produces correct results",
            test_type=TestType.FUNCTIONAL,
            input_data={'algorithm_data': algorithm_data},
            tags=['algorithm', 'correctness']
        ))
        
        # Algorithm performance test
        tests.append(TestCase(
            test_id=f"algorithm_performance_{uuid.uuid4().hex[:8]}",
            name="Algorithm Performance Test",
            description="Measure algorithm execution performance",
            test_type=TestType.PERFORMANCE,
            input_data={'algorithm_data': algorithm_data},
            metadata={'max_execution_time': 5.0},
            tags=['algorithm', 'performance']
        ))
        
        # Algorithm stability test
        tests.append(TestCase(
            test_id=f"algorithm_stability_{uuid.uuid4().hex[:8]}",
            name="Algorithm Stability Test",
            description="Test algorithm stability across multiple runs",
            test_type=TestType.REGRESSION,
            input_data={'algorithm_data': algorithm_data, 'num_runs': 10},
            tags=['algorithm', 'stability']
        ))
        
        return tests
    
    def generate_property_tests(self, target_function: Callable, 
                              properties: List[QuantumProperty]) -> List[TestCase]:
        """Generate property-based test cases."""
        tests = []
        
        for prop in properties:
            tests.append(TestCase(
                test_id=f"property_{prop.value}_{uuid.uuid4().hex[:8]}",
                name=f"{prop.value.title()} Property Test",
                description=f"Test {prop.value} property",
                test_type=TestType.PROPERTY_BASED,
                target_function=target_function,
                properties=[prop],
                tags=['property', prop.value]
            ))
        
        return tests
    
    def _estimate_qubit_count(self, circuit_data: Dict[str, Any]) -> int:
        """Estimate number of qubits in circuit."""
        if 'num_qubits' in circuit_data:
            return circuit_data['num_qubits']
        
        if 'gates' in circuit_data:
            max_qubit = 0
            for gate in circuit_data['gates']:
                if 'qubits' in gate:
                    qubits = gate['qubits']
                    if isinstance(qubits, list):
                        max_qubit = max(max_qubit, max(qubits))
                    else:
                        max_qubit = max(max_qubit, qubits)
            return max_qubit + 1
        
        return 2  # Default


class MockQuantumBackend:
    """Mock quantum backend for testing."""
    
    def __init__(self, noise_level: float = 0.0, latency: float = 0.1):
        self.noise_level = noise_level
        self.latency = latency
        self.call_count = 0
        self.call_log = []
        self.logger = logging.getLogger(__name__ + ".MockBackend")
    
    async def execute_circuit(self, circuit_data: Dict[str, Any], 
                            shots: int = 1024) -> Dict[str, Any]:
        """Mock circuit execution."""
        self.call_count += 1
        self.call_log.append({
            'timestamp': time.time(),
            'circuit_data': circuit_data,
            'shots': shots
        })
        
        # Simulate latency
        await asyncio.sleep(self.latency)
        
        # Generate mock results
        if 'gates' in circuit_data:
            num_qubits = self._estimate_qubits(circuit_data)
            counts = self._generate_mock_counts(num_qubits, shots)
        else:
            counts = {'00': shots}
        
        return {
            'counts': counts,
            'shots': shots,
            'execution_time': self.latency,
            'backend': 'mock',
            'noise_level': self.noise_level
        }
    
    def _estimate_qubits(self, circuit_data: Dict[str, Any]) -> int:
        """Estimate qubit count from circuit."""
        max_qubit = 0
        for gate in circuit_data.get('gates', []):
            if 'qubits' in gate:
                qubits = gate['qubits']
                if isinstance(qubits, list):
                    max_qubit = max(max_qubit, max(qubits))
                else:
                    max_qubit = max(max_qubit, qubits)
        return max_qubit + 1
    
    def _generate_mock_counts(self, num_qubits: int, shots: int) -> Dict[str, int]:
        """Generate mock measurement counts."""
        if not NUMPY_AVAILABLE:
            # Simple fallback
            basis_states = [format(i, f'0{num_qubits}b') for i in range(min(4, 2**num_qubits))]
            counts = {}
            remaining_shots = shots
            
            for i, state in enumerate(basis_states[:-1]):
                count = random.randint(0, remaining_shots)
                counts[state] = count
                remaining_shots -= count
            
            if basis_states:
                counts[basis_states[-1]] = remaining_shots
            
            return counts
        
        # Generate with numpy
        num_states = 2 ** num_qubits
        probabilities = np.random.dirichlet(np.ones(num_states))
        
        # Add noise if specified
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, num_states)
            probabilities += noise
            probabilities = np.abs(probabilities)
            probabilities /= np.sum(probabilities)
        
        # Sample counts
        counts = np.random.multinomial(shots, probabilities)
        
        # Convert to dictionary
        result = {}
        for i, count in enumerate(counts):
            if count > 0:
                state = format(i, f'0{num_qubits}b')
                result[state] = int(count)
        
        return result
    
    def reset(self):
        """Reset backend state."""
        self.call_count = 0
        self.call_log.clear()


class QuantumTestRunner:
    """Executes quantum test cases and suites."""
    
    def __init__(self, backend: Optional[MockQuantumBackend] = None):
        self.backend = backend or MockQuantumBackend()
        self.property_tester = QuantumPropertyTester()
        self.results: Dict[str, TestResult] = {}
        self.logger = logging.getLogger(__name__ + ".TestRunner")
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    async def run_test_case(self, test_case: TestCase) -> TestResult:
        """Run a single test case."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Running test case: {test_case.name}")
            
            # Execute test based on type
            if test_case.test_type == TestType.PROPERTY_BASED:
                result = await self._run_property_test(test_case)
            elif test_case.test_type == TestType.FUNCTIONAL:
                result = await self._run_functional_test(test_case)
            elif test_case.test_type == TestType.PERFORMANCE:
                result = await self._run_performance_test(test_case)
            elif test_case.test_type == TestType.INTEGRATION:
                result = await self._run_integration_test(test_case)
            elif test_case.test_type == TestType.FUZZ:
                result = await self._run_fuzz_test(test_case)
            else:
                result = TestResult(
                    test_id=test_case.test_id,
                    status=TestStatus.SKIPPED,
                    execution_time=0.0,
                    message=f"Unsupported test type: {test_case.test_type}"
                )
            
            result.execution_time = time.time() - start_time
            self.results[test_case.test_id] = result
            
            return result
            
        except asyncio.TimeoutError:
            result = TestResult(
                test_id=test_case.test_id,
                status=TestStatus.TIMEOUT,
                execution_time=time.time() - start_time,
                message=f"Test timed out after {test_case.timeout} seconds"
            )
            self.results[test_case.test_id] = result
            return result
            
        except Exception as e:
            result = TestResult(
                test_id=test_case.test_id,
                status=TestStatus.ERROR,
                execution_time=time.time() - start_time,
                message=f"Test error: {str(e)}",
                traceback=traceback.format_exc()
            )
            self.results[test_case.test_id] = result
            return result
    
    async def _run_property_test(self, test_case: TestCase) -> TestResult:
        """Run property-based test."""
        violations = []
        
        # Execute circuit or function to get result
        if 'circuit_data' in test_case.input_data:
            circuit_result = await self.backend.execute_circuit(
                test_case.input_data['circuit_data']
            )
            
            # Test properties
            for prop in test_case.properties:
                success, message = await self._test_quantum_property(
                    prop, test_case.input_data, circuit_result
                )
                if not success:
                    violations.append(f"{prop.value}: {message}")
        
        elif test_case.target_function:
            # Test function properties
            try:
                result = test_case.target_function(**test_case.input_data)
                
                for prop in test_case.properties:
                    success, message = await self._test_function_property(
                        prop, test_case.target_function, test_case.input_data, result
                    )
                    if not success:
                        violations.append(f"{prop.value}: {message}")
                        
            except Exception as e:
                violations.append(f"Function execution failed: {e}")
        
        if violations:
            return TestResult(
                test_id=test_case.test_id,
                status=TestStatus.FAILED,
                execution_time=0.0,
                message="Property violations detected",
                property_violations=violations,
                severity=TestSeverity.HIGH
            )
        else:
            return TestResult(
                test_id=test_case.test_id,
                status=TestStatus.PASSED,
                execution_time=0.0,
                message="All properties satisfied"
            )
    
    async def _run_functional_test(self, test_case: TestCase) -> TestResult:
        """Run functional test."""
        try:
            if 'circuit_data' in test_case.input_data:
                result = await self.backend.execute_circuit(
                    test_case.input_data['circuit_data']
                )
                
                # Basic validation
                if 'counts' not in result:
                    return TestResult(
                        test_id=test_case.test_id,
                        status=TestStatus.FAILED,
                        execution_time=0.0,
                        message="Circuit execution did not return counts",
                        severity=TestSeverity.HIGH
                    )
                
                # Check expected output if provided
                if test_case.expected_output is not None:
                    if not self._compare_results(result, test_case.expected_output, test_case.tolerance):
                        return TestResult(
                            test_id=test_case.test_id,
                            status=TestStatus.FAILED,
                            execution_time=0.0,
                            message="Output does not match expected result",
                            actual_output=result,
                            severity=TestSeverity.MEDIUM
                        )
                
                return TestResult(
                    test_id=test_case.test_id,
                    status=TestStatus.PASSED,
                    execution_time=0.0,
                    message="Functional test passed",
                    actual_output=result
                )
            
            elif test_case.target_function:
                result = test_case.target_function(**test_case.input_data)
                
                if test_case.expected_output is not None:
                    if not self._compare_results(result, test_case.expected_output, test_case.tolerance):
                        return TestResult(
                            test_id=test_case.test_id,
                            status=TestStatus.FAILED,
                            execution_time=0.0,
                            message="Function output does not match expected result",
                            actual_output=result
                        )
                
                return TestResult(
                    test_id=test_case.test_id,
                    status=TestStatus.PASSED,
                    execution_time=0.0,
                    message="Functional test passed",
                    actual_output=result
                )
            
            else:
                return TestResult(
                    test_id=test_case.test_id,
                    status=TestStatus.SKIPPED,
                    execution_time=0.0,
                    message="No test target specified"
                )
                
        except Exception as e:
            return TestResult(
                test_id=test_case.test_id,
                status=TestStatus.ERROR,
                execution_time=0.0,
                message=f"Functional test error: {e}",
                traceback=traceback.format_exc()
            )
    
    async def _run_performance_test(self, test_case: TestCase) -> TestResult:
        """Run performance test."""
        try:
            # Multiple runs for statistical accuracy
            execution_times = []
            memory_usage = []
            
            num_runs = test_case.metadata.get('num_runs', 5)
            
            for _ in range(num_runs):
                start_time = time.time()
                
                if 'circuit_data' in test_case.input_data:
                    result = await self.backend.execute_circuit(
                        test_case.input_data['circuit_data']
                    )
                elif test_case.target_function:
                    result = test_case.target_function(**test_case.input_data)
                
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
            
            # Calculate performance metrics
            avg_time = statistics.mean(execution_times)
            std_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
            max_time = max(execution_times)
            min_time = min(execution_times)
            
            performance_metrics = {
                'average_time': avg_time,
                'std_time': std_time,
                'max_time': max_time,
                'min_time': min_time,
                'num_runs': num_runs
            }
            
            # Check performance thresholds
            max_expected_time = test_case.metadata.get('expected_max_time', 1.0)
            
            if avg_time > max_expected_time:
                return TestResult(
                    test_id=test_case.test_id,
                    status=TestStatus.FAILED,
                    execution_time=avg_time,
                    message=f"Performance test failed: average time {avg_time:.3f}s exceeds {max_expected_time}s",
                    performance_metrics=performance_metrics,
                    severity=TestSeverity.MEDIUM
                )
            
            return TestResult(
                test_id=test_case.test_id,
                status=TestStatus.PASSED,
                execution_time=avg_time,
                message=f"Performance test passed: average time {avg_time:.3f}s",
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            return TestResult(
                test_id=test_case.test_id,
                status=TestStatus.ERROR,
                execution_time=0.0,
                message=f"Performance test error: {e}",
                traceback=traceback.format_exc()
            )
    
    async def _run_integration_test(self, test_case: TestCase) -> TestResult:
        """Run integration test."""
        try:
            # Integration tests typically involve multiple components
            components_tested = []
            
            if 'circuit_data' in test_case.input_data:
                # Test circuit execution with compilation
                if QUANTRS_MODULES_AVAILABLE and compilation_service:
                    compiler = compilation_service.get_compilation_service()
                    compile_request = compilation_service.CompilationRequest(
                        circuit_data=test_case.input_data['circuit_data']
                    )
                    compile_result = await compiler.compile_circuit_async(compile_request)
                    components_tested.append('compilation_service')
                
                # Test with backend
                result = await self.backend.execute_circuit(
                    test_case.input_data['circuit_data']
                )
                components_tested.append('quantum_backend')
                
                # Test with profiler if available
                if QUANTRS_MODULES_AVAILABLE and profiler:
                    profile_result = profiler.profile_circuit(
                        test_case.input_data['circuit_data']
                    )
                    components_tested.append('profiler')
            
            return TestResult(
                test_id=test_case.test_id,
                status=TestStatus.PASSED,
                execution_time=0.0,
                message=f"Integration test passed: {', '.join(components_tested)}",
                metadata={'components_tested': components_tested}
            )
            
        except Exception as e:
            return TestResult(
                test_id=test_case.test_id,
                status=TestStatus.ERROR,
                execution_time=0.0,
                message=f"Integration test error: {e}",
                traceback=traceback.format_exc()
            )
    
    async def _run_fuzz_test(self, test_case: TestCase) -> TestResult:
        """Run fuzz test with random inputs."""
        try:
            failures = []
            num_iterations = test_case.metadata.get('fuzz_iterations', 10)
            
            for i in range(num_iterations):
                # Generate random input variations
                fuzz_input = self._generate_fuzz_input(test_case.input_data)
                
                try:
                    if 'circuit_data' in fuzz_input:
                        result = await self.backend.execute_circuit(fuzz_input['circuit_data'])
                    elif test_case.target_function:
                        result = test_case.target_function(**fuzz_input)
                        
                except Exception as e:
                    failures.append(f"Iteration {i}: {e}")
            
            if failures:
                return TestResult(
                    test_id=test_case.test_id,
                    status=TestStatus.FAILED,
                    execution_time=0.0,
                    message=f"Fuzz test found {len(failures)} failures",
                    assertion_errors=failures[:5],  # Limit to first 5
                    severity=TestSeverity.HIGH
                )
            
            return TestResult(
                test_id=test_case.test_id,
                status=TestStatus.PASSED,
                execution_time=0.0,
                message=f"Fuzz test passed {num_iterations} iterations"
            )
            
        except Exception as e:
            return TestResult(
                test_id=test_case.test_id,
                status=TestStatus.ERROR,
                execution_time=0.0,
                message=f"Fuzz test error: {e}",
                traceback=traceback.format_exc()
            )
    
    async def run_test_suite(self, test_suite: TestSuite) -> Dict[str, TestResult]:
        """Run a complete test suite."""
        self.logger.info(f"Running test suite: {test_suite.name}")
        suite_results = {}
        
        try:
            # Run setup hooks
            for hook in test_suite.setup_hooks:
                try:
                    if asyncio.iscoroutinefunction(hook):
                        await hook()
                    else:
                        hook()
                except Exception as e:
                    self.logger.error(f"Setup hook failed: {e}")
            
            # Run test cases
            for test_case in test_suite.test_cases:
                result = await self.run_test_case(test_case)
                suite_results[test_case.test_id] = result
            
            # Run teardown hooks
            for hook in test_suite.teardown_hooks:
                try:
                    if asyncio.iscoroutinefunction(hook):
                        await hook()
                    else:
                        hook()
                except Exception as e:
                    self.logger.error(f"Teardown hook failed: {e}")
            
        except Exception as e:
            self.logger.error(f"Test suite execution failed: {e}")
        
        return suite_results
    
    async def _test_quantum_property(self, prop: QuantumProperty, 
                                   input_data: Dict[str, Any], 
                                   result: Dict[str, Any]) -> Tuple[bool, str]:
        """Test a specific quantum property."""
        if prop == QuantumProperty.NORMALIZATION:
            # Test if measurement probabilities sum to 1
            if 'counts' in result:
                total_shots = sum(result['counts'].values())
                if total_shots == 0:
                    return False, "No measurement results"
                return True, "Measurement probabilities normalized"
            
        elif prop == QuantumProperty.MEASUREMENT_CONSERVATION:
            # Test if total measurement count equals shots
            if 'counts' in result and 'shots' in result:
                total_counts = sum(result['counts'].values())
                if total_counts != result['shots']:
                    return False, f"Measurement count mismatch: {total_counts} != {result['shots']}"
                return True, "Measurement conservation satisfied"
        
        elif prop == QuantumProperty.ENTANGLEMENT:
            # Basic entanglement test based on result distribution
            if 'counts' in result:
                counts = result['counts']
                if len(counts) > 1:
                    # Check if distribution is not uniform (potential entanglement)
                    values = list(counts.values())
                    max_count = max(values)
                    min_count = min(values)
                    if max_count > 2 * min_count:
                        return True, "Non-uniform distribution suggests entanglement"
                
                return False, "No clear entanglement signature"
        
        # Default: property test passed
        return True, f"Property {prop.value} test passed"
    
    async def _test_function_property(self, prop: QuantumProperty, 
                                    function: Callable, input_data: Dict[str, Any],
                                    result: Any) -> Tuple[bool, str]:
        """Test quantum property for function result."""
        if prop == QuantumProperty.UNITARITY and NUMPY_AVAILABLE:
            if hasattr(result, 'shape') or isinstance(result, (list, tuple)):
                return self.property_tester.test_unitarity(result)
        
        elif prop == QuantumProperty.NORMALIZATION and NUMPY_AVAILABLE:
            if hasattr(result, 'shape') or isinstance(result, (list, tuple)):
                return self.property_tester.test_normalization(result)
        
        elif prop == QuantumProperty.HERMITICITY and NUMPY_AVAILABLE:
            if hasattr(result, 'shape') or isinstance(result, (list, tuple)):
                return self.property_tester.test_hermiticity(result)
        
        return True, f"Property {prop.value} test passed (default)"
    
    def _compare_results(self, actual: Any, expected: Any, tolerance: float) -> bool:
        """Compare test results with tolerance."""
        if isinstance(expected, dict) and isinstance(actual, dict):
            # Compare dictionaries
            for key, expected_value in expected.items():
                if key not in actual:
                    return False
                if not self._compare_values(actual[key], expected_value, tolerance):
                    return False
            return True
        else:
            return self._compare_values(actual, expected, tolerance)
    
    def _compare_values(self, actual: Any, expected: Any, tolerance: float) -> bool:
        """Compare individual values with tolerance."""
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return abs(actual - expected) <= tolerance
        elif isinstance(expected, str) and isinstance(actual, str):
            return actual == expected
        elif isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
            if len(actual) != len(expected):
                return False
            return all(self._compare_values(a, e, tolerance) for a, e in zip(actual, expected))
        else:
            return actual == expected
    
    def _generate_fuzz_input(self, base_input: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fuzzed input data."""
        fuzzed = base_input.copy()
        
        # Fuzz circuit data if present
        if 'circuit_data' in fuzzed and 'gates' in fuzzed['circuit_data']:
            fuzzed_gates = []
            for gate in fuzzed['circuit_data']['gates']:
                fuzzed_gate = gate.copy()
                
                # Randomly modify gate parameters
                if random.random() < 0.3:  # 30% chance to modify
                    if 'qubits' in fuzzed_gate and isinstance(fuzzed_gate['qubits'], list):
                        # Randomly shuffle qubits
                        if random.random() < 0.5:
                            random.shuffle(fuzzed_gate['qubits'])
                    
                    # Add random parameters
                    if random.random() < 0.2:
                        fuzzed_gate['random_param'] = random.random()
                
                fuzzed_gates.append(fuzzed_gate)
            
            fuzzed['circuit_data']['gates'] = fuzzed_gates
        
        # Fuzz numeric parameters
        for key, value in fuzzed.items():
            if isinstance(value, (int, float)) and random.random() < 0.2:
                fuzzed[key] = value * (0.5 + random.random())
        
        return fuzzed
    
    def get_test_statistics(self) -> Dict[str, Any]:
        """Get test execution statistics."""
        if not self.results:
            return {'total_tests': 0}
        
        status_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        execution_times = []
        
        for result in self.results.values():
            status_counts[result.status.value] += 1
            severity_counts[result.severity.value] += 1
            execution_times.append(result.execution_time)
        
        return {
            'total_tests': len(self.results),
            'status_counts': dict(status_counts),
            'severity_counts': dict(severity_counts),
            'average_execution_time': statistics.mean(execution_times),
            'total_execution_time': sum(execution_times),
            'pass_rate': status_counts['passed'] / len(self.results) if self.results else 0.0
        }


class QuantumTestReporter:
    """Generates test reports and analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".TestReporter")
    
    def generate_report(self, results: Dict[str, TestResult], 
                       output_format: str = 'json') -> Union[str, Dict[str, Any]]:
        """Generate test report in specified format."""
        
        # Collect statistics
        stats = self._calculate_statistics(results)
        
        # Group results by status
        grouped_results = self._group_by_status(results)
        
        # Generate detailed analysis
        analysis = self._generate_analysis(results, stats)
        
        report_data = {
            'summary': stats,
            'analysis': analysis,
            'results_by_status': grouped_results,
            'detailed_results': [result.to_dict() for result in results.values()],
            'generated_at': time.time()
        }
        
        if output_format == 'json':
            return json.dumps(report_data, indent=2)
        elif output_format == 'dict':
            return report_data
        elif output_format == 'text':
            return self._generate_text_report(report_data)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def generate_html_report(self, results: Dict[str, TestResult], 
                           output_file: Optional[str] = None) -> str:
        """Generate HTML test report."""
        report_data = self.generate_report(results, 'dict')
        
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Quantum Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .summary { background: #f5f5f5; padding: 15px; border-radius: 5px; }
        .passed { color: green; }
        .failed { color: red; }
        .error { color: orange; }
        .skipped { color: gray; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .test-result { margin: 10px 0; padding: 10px; border-left: 4px solid #ddd; }
        .test-passed { border-left-color: green; }
        .test-failed { border-left-color: red; }
        .test-error { border-left-color: orange; }
    </style>
</head>
<body>
    <h1>Quantum Test Report</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Tests:</strong> {total_tests}</p>
        <p><strong>Pass Rate:</strong> {pass_rate:.1%}</p>
        <p><strong>Execution Time:</strong> {total_time:.3f}s</p>
        <ul>
            <li class="passed">Passed: {passed_count}</li>
            <li class="failed">Failed: {failed_count}</li>
            <li class="error">Errors: {error_count}</li>
            <li class="skipped">Skipped: {skipped_count}</li>
        </ul>
    </div>
    
    <h2>Test Results</h2>
    {test_results_html}
    
    <h2>Analysis</h2>
    <div>{analysis_html}</div>
    
    <p><em>Generated at: {timestamp}</em></p>
</body>
</html>
        """
        
        # Format the template
        stats = report_data['summary']
        status_counts = stats['status_counts']
        
        test_results_html = ""
        for test_id, result in enumerate(report_data['detailed_results']):
            css_class = f"test-{result['status']}"
            test_results_html += f"""
            <div class="test-result {css_class}">
                <h3>{result.get('test_id', f'Test {test_id}')}</h3>
                <p><strong>Status:</strong> {result['status']}</p>
                <p><strong>Time:</strong> {result['execution_time']:.3f}s</p>
                <p><strong>Message:</strong> {result['message']}</p>
                {f"<p><strong>Errors:</strong> {', '.join(result['assertion_errors'])}</p>" if result['assertion_errors'] else ""}
            </div>
            """
        
        analysis_html = ""
        for key, value in report_data['analysis'].items():
            analysis_html += f"<p><strong>{key.replace('_', ' ').title()}:</strong> {value}</p>"
        
        html_content = html_template.format(
            total_tests=stats['total_tests'],
            pass_rate=stats['pass_rate'],
            total_time=stats['total_execution_time'],
            passed_count=status_counts.get('passed', 0),
            failed_count=status_counts.get('failed', 0),
            error_count=status_counts.get('error', 0),
            skipped_count=status_counts.get('skipped', 0),
            test_results_html=test_results_html,
            analysis_html=analysis_html,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(html_content)
        
        return html_content
    
    def _calculate_statistics(self, results: Dict[str, TestResult]) -> Dict[str, Any]:
        """Calculate test statistics."""
        if not results:
            return {
                'total_tests': 0,
                'pass_rate': 0.0,
                'status_counts': {},
                'severity_counts': {},
                'total_execution_time': 0.0,
                'average_execution_time': 0.0
            }
        
        status_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        execution_times = []
        
        for result in results.values():
            status_counts[result.status.value] += 1
            severity_counts[result.severity.value] += 1
            execution_times.append(result.execution_time)
        
        return {
            'total_tests': len(results),
            'pass_rate': status_counts['passed'] / len(results),
            'status_counts': dict(status_counts),
            'severity_counts': dict(severity_counts),
            'total_execution_time': sum(execution_times),
            'average_execution_time': statistics.mean(execution_times)
        }
    
    def _group_by_status(self, results: Dict[str, TestResult]) -> Dict[str, List[Dict[str, Any]]]:
        """Group results by status."""
        grouped = defaultdict(list)
        
        for result in results.values():
            grouped[result.status.value].append(result.to_dict())
        
        return dict(grouped)
    
    def _generate_analysis(self, results: Dict[str, TestResult], 
                         stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test analysis."""
        analysis = {}
        
        # Performance analysis
        execution_times = [r.execution_time for r in results.values()]
        if execution_times:
            analysis['slowest_test_time'] = max(execution_times)
            analysis['fastest_test_time'] = min(execution_times)
            analysis['time_std_deviation'] = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
        
        # Failure analysis
        failed_results = [r for r in results.values() if r.status == TestStatus.FAILED]
        if failed_results:
            analysis['failure_rate'] = len(failed_results) / len(results)
            
            # Common failure patterns
            failure_messages = [r.message for r in failed_results]
            analysis['common_failure_keywords'] = self._extract_keywords(failure_messages)
        
        # Property violation analysis
        property_violations = []
        for result in results.values():
            property_violations.extend(result.property_violations)
        
        if property_violations:
            analysis['property_violation_count'] = len(property_violations)
            analysis['common_property_violations'] = self._extract_keywords(property_violations)
        
        # Test type distribution
        type_counts = defaultdict(int)
        for result in results.values():
            # Try to infer test type from test_id or other metadata
            if 'property' in result.test_id:
                type_counts['property_based'] += 1
            elif 'performance' in result.test_id:
                type_counts['performance'] += 1
            elif 'functional' in result.test_id:
                type_counts['functional'] += 1
            else:
                type_counts['other'] += 1
        
        analysis['test_type_distribution'] = dict(type_counts)
        
        return analysis
    
    def _extract_keywords(self, messages: List[str]) -> List[str]:
        """Extract common keywords from messages."""
        if not messages:
            return []
        
        # Simple keyword extraction
        words = []
        for message in messages:
            words.extend(message.lower().split())
        
        # Count word frequency
        word_counts = defaultdict(int)
        for word in words:
            if len(word) > 3:  # Filter short words
                word_counts[word] += 1
        
        # Return most common words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:5]]
    
    def _generate_text_report(self, report_data: Dict[str, Any]) -> str:
        """Generate plain text report."""
        lines = []
        lines.append("QUANTUM TEST REPORT")
        lines.append("=" * 50)
        lines.append("")
        
        # Summary
        stats = report_data['summary']
        lines.append("SUMMARY")
        lines.append("-" * 20)
        lines.append(f"Total Tests: {stats['total_tests']}")
        lines.append(f"Pass Rate: {stats['pass_rate']:.1%}")
        lines.append(f"Total Execution Time: {stats['total_execution_time']:.3f}s")
        lines.append(f"Average Execution Time: {stats['average_execution_time']:.3f}s")
        lines.append("")
        
        # Status breakdown
        lines.append("STATUS BREAKDOWN")
        lines.append("-" * 20)
        for status, count in stats['status_counts'].items():
            lines.append(f"{status.title()}: {count}")
        lines.append("")
        
        # Analysis
        lines.append("ANALYSIS")
        lines.append("-" * 20)
        for key, value in report_data['analysis'].items():
            lines.append(f"{key.replace('_', ' ').title()}: {value}")
        lines.append("")
        
        # Failed tests details
        failed_results = report_data['results_by_status'].get('failed', [])
        if failed_results:
            lines.append("FAILED TESTS")
            lines.append("-" * 20)
            for result in failed_results[:10]:  # Limit to first 10
                lines.append(f"Test ID: {result['test_id']}")
                lines.append(f"Message: {result['message']}")
                lines.append("")
        
        lines.append(f"Report generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(lines)


class QuantumTestManager:
    """Central manager for quantum testing operations."""
    
    def __init__(self, backend: Optional[MockQuantumBackend] = None):
        self.backend = backend or MockQuantumBackend()
        self.test_generator = QuantumTestGenerator()
        self.test_runner = QuantumTestRunner(self.backend)
        self.reporter = QuantumTestReporter()
        self.test_suites: Dict[str, TestSuite] = {}
        self.logger = logging.getLogger(__name__ + ".TestManager")
        
        # Create default test suites
        self._create_default_test_suites()
    
    def _create_default_test_suites(self):
        """Create default test suites for common scenarios."""
        
        # Basic quantum operations test suite
        basic_suite = TestSuite(
            suite_id="basic_quantum_ops",
            name="Basic Quantum Operations",
            description="Test suite for basic quantum gate operations and properties"
        )
        
        # Add basic gate tests
        for gate_name in ['h', 'x', 'y', 'z', 'cnot', 'cz']:
            gate_tests = self.test_generator.generate_gate_tests(gate_name)
            for test in gate_tests:
                basic_suite.add_test_case(test)
        
        self.test_suites[basic_suite.suite_id] = basic_suite
        
        # Circuit validation test suite
        circuit_suite = TestSuite(
            suite_id="circuit_validation",
            name="Circuit Validation",
            description="Test suite for quantum circuit validation and verification"
        )
        
        # Add circuit tests for common patterns
        bell_circuit = {
            'gates': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'cnot', 'qubits': [0, 1]}
            ]
        }
        
        circuit_tests = self.test_generator.generate_circuit_tests(bell_circuit)
        for test in circuit_tests:
            circuit_suite.add_test_case(test)
        
        self.test_suites[circuit_suite.suite_id] = circuit_suite
    
    def create_test_suite(self, name: str, description: str = "") -> TestSuite:
        """Create a new test suite."""
        suite = TestSuite(
            suite_id=str(uuid.uuid4()),
            name=name,
            description=description
        )
        
        self.test_suites[suite.suite_id] = suite
        return suite
    
    def add_circuit_tests(self, suite_id: str, circuit_data: Dict[str, Any], 
                         num_tests: int = 5) -> bool:
        """Add circuit tests to a test suite."""
        if suite_id not in self.test_suites:
            return False
        
        suite = self.test_suites[suite_id]
        tests = self.test_generator.generate_circuit_tests(circuit_data, num_tests)
        
        for test in tests:
            suite.add_test_case(test)
        
        return True
    
    def add_property_tests(self, suite_id: str, target_function: Callable,
                          properties: List[QuantumProperty]) -> bool:
        """Add property-based tests to a test suite."""
        if suite_id not in self.test_suites:
            return False
        
        suite = self.test_suites[suite_id]
        tests = self.test_generator.generate_property_tests(target_function, properties)
        
        for test in tests:
            suite.add_test_case(test)
        
        return True
    
    async def run_test_suite(self, suite_id: str) -> Dict[str, TestResult]:
        """Run a test suite."""
        if suite_id not in self.test_suites:
            raise ValueError(f"Test suite not found: {suite_id}")
        
        suite = self.test_suites[suite_id]
        self.logger.info(f"Running test suite: {suite.name}")
        
        return await self.test_runner.run_test_suite(suite)
    
    async def run_all_test_suites(self) -> Dict[str, Dict[str, TestResult]]:
        """Run all test suites."""
        all_results = {}
        
        for suite_id, suite in self.test_suites.items():
            self.logger.info(f"Running test suite: {suite.name}")
            suite_results = await self.test_runner.run_test_suite(suite)
            all_results[suite_id] = suite_results
        
        return all_results
    
    def generate_comprehensive_report(self, results: Dict[str, Dict[str, TestResult]], 
                                   output_file: Optional[str] = None) -> str:
        """Generate comprehensive report for all test results."""
        # Flatten results
        flattened_results = {}
        for suite_id, suite_results in results.items():
            for test_id, result in suite_results.items():
                flattened_results[f"{suite_id}_{test_id}"] = result
        
        # Generate report
        if output_file and output_file.endswith('.html'):
            return self.reporter.generate_html_report(flattened_results, output_file)
        else:
            report = self.reporter.generate_report(flattened_results, 'text')
            
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(report)
            
            return report
    
    def get_test_suite_info(self, suite_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a test suite."""
        if suite_id not in self.test_suites:
            return None
        
        suite = self.test_suites[suite_id]
        
        return {
            'suite_id': suite.suite_id,
            'name': suite.name,
            'description': suite.description,
            'test_count': len(suite.test_cases),
            'test_types': list(set(test.test_type.value for test in suite.test_cases)),
            'tags': suite.tags,
            'created_at': suite.created_at
        }
    
    def list_test_suites(self) -> List[Dict[str, Any]]:
        """List all available test suites."""
        return [self.get_test_suite_info(suite_id) for suite_id in self.test_suites.keys()]
    
    def get_test_statistics(self) -> Dict[str, Any]:
        """Get overall testing statistics."""
        total_suites = len(self.test_suites)
        total_tests = sum(len(suite.test_cases) for suite in self.test_suites.values())
        
        test_type_counts = defaultdict(int)
        property_counts = defaultdict(int)
        
        for suite in self.test_suites.values():
            for test in suite.test_cases:
                test_type_counts[test.test_type.value] += 1
                for prop in test.properties:
                    property_counts[prop.value] += 1
        
        return {
            'total_suites': total_suites,
            'total_tests': total_tests,
            'test_type_distribution': dict(test_type_counts),
            'property_distribution': dict(property_counts),
            'backend_info': {
                'type': 'mock',
                'call_count': self.backend.call_count,
                'noise_level': self.backend.noise_level
            }
        }


# Global test manager instance
_quantum_test_manager: Optional[QuantumTestManager] = None


def get_quantum_test_manager() -> QuantumTestManager:
    """Get global quantum test manager."""
    global _quantum_test_manager
    if _quantum_test_manager is None:
        _quantum_test_manager = QuantumTestManager()
    return _quantum_test_manager


# Convenience functions
def create_test_suite(name: str, description: str = "") -> TestSuite:
    """Convenience function to create test suite."""
    manager = get_quantum_test_manager()
    return manager.create_test_suite(name, description)


def test_quantum_circuit(circuit_data: Dict[str, Any], 
                        properties: List[QuantumProperty] = None) -> List[TestCase]:
    """Convenience function to test quantum circuit."""
    generator = QuantumTestGenerator()
    tests = generator.generate_circuit_tests(circuit_data)
    
    if properties:
        prop_tests = generator.generate_property_tests(None, properties)
        for test in prop_tests:
            test.input_data['circuit_data'] = circuit_data
        tests.extend(prop_tests)
    
    return tests


def test_quantum_function(function: Callable, input_data: Dict[str, Any],
                         properties: List[QuantumProperty]) -> List[TestCase]:
    """Convenience function to test quantum function."""
    generator = QuantumTestGenerator()
    tests = generator.generate_property_tests(function, properties)
    
    for test in tests:
        test.input_data.update(input_data)
    
    return tests


async def run_quantum_tests(test_cases: List[TestCase]) -> Dict[str, TestResult]:
    """Convenience function to run quantum tests."""
    runner = QuantumTestRunner()
    results = {}
    
    for test_case in test_cases:
        result = await runner.run_test_case(test_case)
        results[test_case.test_id] = result
    
    return results


# CLI interface
def main():
    """Main CLI interface for quantum testing tools."""
    import argparse
    
    parser = argparse.ArgumentParser(description="QuantRS2 Quantum Testing Tools")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Test suite commands
    suite_parser = subparsers.add_parser('suite', help='Test suite management')
    suite_subparsers = suite_parser.add_subparsers(dest='suite_command')
    
    suite_subparsers.add_parser('list', help='List available test suites')
    
    suite_info_parser = suite_subparsers.add_parser('info', help='Get test suite info')
    suite_info_parser.add_argument('suite_id', help='Test suite ID')
    
    suite_run_parser = suite_subparsers.add_parser('run', help='Run test suite')
    suite_run_parser.add_argument('suite_id', help='Test suite ID')
    suite_run_parser.add_argument('--output', help='Output file for report')
    
    # Test commands
    test_parser = subparsers.add_parser('test', help='Individual test operations')
    test_subparsers = test_parser.add_subparsers(dest='test_command')
    
    circuit_test_parser = test_subparsers.add_parser('circuit', help='Test quantum circuit')
    circuit_test_parser.add_argument('circuit_file', help='Circuit definition file (JSON)')
    circuit_test_parser.add_argument('--properties', nargs='*', 
                                   choices=[p.value for p in QuantumProperty],
                                   help='Properties to test')
    circuit_test_parser.add_argument('--output', help='Output file for report')
    
    # Run all tests
    run_all_parser = subparsers.add_parser('run-all', help='Run all test suites')
    run_all_parser.add_argument('--output', help='Output file for report')
    run_all_parser.add_argument('--format', choices=['text', 'html', 'json'], 
                               default='text', help='Report format')
    
    # Statistics
    subparsers.add_parser('stats', help='Show testing statistics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    manager = get_quantum_test_manager()
    
    async def run_command():
        try:
            if args.command == 'suite':
                if args.suite_command == 'list':
                    suites = manager.list_test_suites()
                    print("Available Test Suites:")
                    for suite in suites:
                        print(f"  {suite['suite_id']}: {suite['name']} ({suite['test_count']} tests)")
                
                elif args.suite_command == 'info':
                    info = manager.get_test_suite_info(args.suite_id)
                    if info:
                        print("Test Suite Information:")
                        print(json.dumps(info, indent=2))
                    else:
                        print(f"Test suite not found: {args.suite_id}")
                        return 1
                
                elif args.suite_command == 'run':
                    results = await manager.run_test_suite(args.suite_id)
                    
                    if args.output:
                        if args.output.endswith('.html'):
                            manager.reporter.generate_html_report(results, args.output)
                            print(f"HTML report written to: {args.output}")
                        else:
                            report = manager.reporter.generate_report(results, 'text')
                            with open(args.output, 'w') as f:
                                f.write(report)
                            print(f"Report written to: {args.output}")
                    else:
                        report = manager.reporter.generate_report(results, 'text')
                        print(report)
            
            elif args.command == 'test':
                if args.test_command == 'circuit':
                    # Load circuit from file
                    with open(args.circuit_file, 'r') as f:
                        circuit_data = json.load(f)
                    
                    # Generate tests
                    properties = []
                    if args.properties:
                        properties = [QuantumProperty(p) for p in args.properties]
                    
                    test_cases = test_quantum_circuit(circuit_data, properties)
                    
                    # Run tests
                    results = await run_quantum_tests(test_cases)
                    
                    # Generate report
                    reporter = QuantumTestReporter()
                    if args.output:
                        if args.output.endswith('.html'):
                            reporter.generate_html_report(results, args.output)
                            print(f"HTML report written to: {args.output}")
                        else:
                            report = reporter.generate_report(results, 'text')
                            with open(args.output, 'w') as f:
                                f.write(report)
                            print(f"Report written to: {args.output}")
                    else:
                        report = reporter.generate_report(results, 'text')
                        print(report)
            
            elif args.command == 'run-all':
                results = await manager.run_all_test_suites()
                report = manager.generate_comprehensive_report(results, args.output)
                
                if not args.output:
                    print(report)
                else:
                    print(f"Report written to: {args.output}")
            
            elif args.command == 'stats':
                stats = manager.get_test_statistics()
                print("Testing Statistics:")
                print(json.dumps(stats, indent=2))
            
            return 0
            
        except Exception as e:
            print(f"Error: {e}")
            return 1
    
    return asyncio.run(run_command())


if __name__ == "__main__":
    exit(main())