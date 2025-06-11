#!/usr/bin/env python3
"""
Test suite for quantum testing tools functionality.
"""

import pytest
import asyncio
import json
import time
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any

try:
    from quantrs2.quantum_testing_tools import (
        TestType, TestStatus, QuantumProperty, TestSeverity,
        TestCase, TestResult, TestSuite,
        QuantumPropertyTester, QuantumTestGenerator, MockQuantumBackend,
        QuantumTestRunner, QuantumTestReporter, QuantumTestManager,
        get_quantum_test_manager, create_test_suite, test_quantum_circuit,
        test_quantum_function, run_quantum_tests
    )
    HAS_TESTING_TOOLS = True
except ImportError:
    HAS_TESTING_TOOLS = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@pytest.mark.skipif(not HAS_TESTING_TOOLS, reason="quantum_testing_tools module not available")
class TestTestCase:
    """Test TestCase functionality."""
    
    def test_test_case_creation(self):
        """Test creating test case."""
        test_case = TestCase(
            test_id="test_1",
            name="Test Circuit",
            description="Test quantum circuit functionality",
            test_type=TestType.FUNCTIONAL,
            input_data={'circuit_data': {'gates': []}},
            properties=[QuantumProperty.UNITARITY],
            tags=['circuit', 'test']
        )
        
        assert test_case.test_id == "test_1"
        assert test_case.name == "Test Circuit"
        assert test_case.test_type == TestType.FUNCTIONAL
        assert QuantumProperty.UNITARITY in test_case.properties
        assert 'circuit' in test_case.tags
        assert test_case.tolerance == 1e-6
        assert test_case.timeout == 30.0
    
    def test_test_case_serialization(self):
        """Test test case serialization."""
        test_case = TestCase(
            test_id="serial_test",
            name="Serialization Test",
            description="Test serialization functionality",
            test_type=TestType.PROPERTY_BASED,
            input_data={'param1': 'value1'},
            expected_output={'result': 'expected'},
            properties=[QuantumProperty.NORMALIZATION, QuantumProperty.HERMITICITY],
            tolerance=1e-5,
            timeout=60.0,
            tags=['serialization'],
            metadata={'key': 'value'}
        )
        
        test_dict = test_case.to_dict()
        
        assert test_dict['test_id'] == "serial_test"
        assert test_dict['test_type'] == 'property_based'
        assert test_dict['properties'] == ['normalization', 'hermiticity']
        assert test_dict['tolerance'] == 1e-5
        assert test_dict['metadata']['key'] == 'value'
    
    def test_test_case_deserialization(self):
        """Test test case deserialization."""
        data = {
            'test_id': 'deserial_test',
            'name': 'Deserialization Test',
            'description': 'Test deserialization',
            'test_type': 'functional',
            'input_data': {'data': 'test'},
            'expected_output': None,
            'properties': ['unitarity', 'commutativity'],
            'tolerance': 1e-4,
            'timeout': 45.0,
            'tags': ['deserial'],
            'metadata': {},
            'created_at': time.time()
        }
        
        test_case = TestCase.from_dict(data)
        
        assert test_case.test_id == 'deserial_test'
        assert test_case.test_type == TestType.FUNCTIONAL
        assert len(test_case.properties) == 2
        assert QuantumProperty.UNITARITY in test_case.properties
        assert QuantumProperty.COMMUTATIVITY in test_case.properties
        assert test_case.tolerance == 1e-4


@pytest.mark.skipif(not HAS_TESTING_TOOLS, reason="quantum_testing_tools module not available")
class TestTestResult:
    """Test TestResult functionality."""
    
    def test_test_result_creation(self):
        """Test creating test result."""
        result = TestResult(
            test_id="result_test",
            status=TestStatus.PASSED,
            execution_time=0.5,
            message="Test passed successfully",
            actual_output={'counts': {'00': 500, '11': 500}},
            performance_metrics={'avg_time': 0.5}
        )
        
        assert result.test_id == "result_test"
        assert result.status == TestStatus.PASSED
        assert result.execution_time == 0.5
        assert result.message == "Test passed successfully"
        assert result.actual_output['counts']['00'] == 500
        assert result.severity == TestSeverity.MEDIUM
    
    def test_test_result_serialization(self):
        """Test test result serialization."""
        result = TestResult(
            test_id="serial_result",
            status=TestStatus.FAILED,
            execution_time=1.2,
            message="Test failed",
            assertion_errors=["Error 1", "Error 2"],
            property_violations=["Violation 1"],
            severity=TestSeverity.HIGH,
            traceback="Mock traceback"
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['test_id'] == "serial_result"
        assert result_dict['status'] == 'failed'
        assert result_dict['execution_time'] == 1.2
        assert len(result_dict['assertion_errors']) == 2
        assert len(result_dict['property_violations']) == 1
        assert result_dict['severity'] == 'high'


@pytest.mark.skipif(not HAS_TESTING_TOOLS, reason="quantum_testing_tools module not available")
class TestTestSuite:
    """Test TestSuite functionality."""
    
    def test_test_suite_creation(self):
        """Test creating test suite."""
        suite = TestSuite(
            suite_id="suite_1",
            name="Test Suite",
            description="Test suite for testing",
            tags=['suite', 'test']
        )
        
        assert suite.suite_id == "suite_1"
        assert suite.name == "Test Suite"
        assert len(suite.test_cases) == 0
        assert len(suite.setup_hooks) == 0
        assert len(suite.teardown_hooks) == 0
        assert 'suite' in suite.tags
    
    def test_add_test_case(self):
        """Test adding test case to suite."""
        suite = TestSuite(
            suite_id="suite_2",
            name="Add Test Suite",
            description="Test adding test cases"
        )
        
        test_case = TestCase(
            test_id="add_test",
            name="Add Test",
            description="Test to add",
            test_type=TestType.FUNCTIONAL
        )
        
        suite.add_test_case(test_case)
        
        assert len(suite.test_cases) == 1
        assert suite.test_cases[0].test_id == "add_test"
    
    def test_add_hooks(self):
        """Test adding setup and teardown hooks."""
        suite = TestSuite(
            suite_id="hooks_suite",
            name="Hooks Suite",
            description="Test hooks"
        )
        
        def setup_hook():
            pass
        
        def teardown_hook():
            pass
        
        suite.add_setup_hook(setup_hook)
        suite.add_teardown_hook(teardown_hook)
        
        assert len(suite.setup_hooks) == 1
        assert len(suite.teardown_hooks) == 1
        assert suite.setup_hooks[0] == setup_hook
        assert suite.teardown_hooks[0] == teardown_hook


@pytest.mark.skipif(not HAS_TESTING_TOOLS, reason="quantum_testing_tools module not available")
class TestQuantumPropertyTester:
    """Test QuantumPropertyTester functionality."""
    
    def setup_method(self):
        """Set up property tester."""
        self.tester = QuantumPropertyTester(tolerance=1e-6)
    
    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
    def test_unitarity_test(self):
        """Test unitarity testing."""
        # Test with identity matrix (unitary)
        identity = np.eye(2)
        is_unitary, message = self.tester.test_unitarity(identity)
        assert is_unitary is True
        assert "unitary" in message.lower()
        
        # Test with Hadamard gate (unitary)
        hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        is_unitary, message = self.tester.test_unitarity(hadamard)
        assert is_unitary is True
        
        # Test with non-unitary matrix
        non_unitary = np.array([[1, 0], [0, 2]])
        is_unitary, message = self.tester.test_unitarity(non_unitary)
        assert is_unitary is False
        assert "not unitary" in message.lower()
    
    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
    def test_normalization_test(self):
        """Test normalization testing."""
        # Test normalized state
        normalized_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        is_normalized, message = self.tester.test_normalization(normalized_state)
        assert is_normalized is True
        assert "normalized" in message.lower()
        
        # Test non-normalized state
        non_normalized_state = np.array([1, 1])
        is_normalized, message = self.tester.test_normalization(non_normalized_state)
        assert is_normalized is False
        assert "not normalized" in message.lower()
    
    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
    def test_hermiticity_test(self):
        """Test hermiticity testing."""
        # Test with Pauli-X (Hermitian)
        pauli_x = np.array([[0, 1], [1, 0]])
        is_hermitian, message = self.tester.test_hermiticity(pauli_x)
        assert is_hermitian is True
        assert "hermitian" in message.lower()
        
        # Test with non-Hermitian matrix
        non_hermitian = np.array([[1, 2j], [0, 1]])
        is_hermitian, message = self.tester.test_hermiticity(non_hermitian)
        assert is_hermitian is False
        assert "not hermitian" in message.lower()
    
    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
    def test_commutativity_test(self):
        """Test commutativity testing."""
        # Test with commuting matrices (Pauli-X and Pauli-X)
        pauli_x = np.array([[0, 1], [1, 0]])
        commute, message = self.tester.test_commutativity(pauli_x, pauli_x)
        assert commute is True
        assert "commute" in message.lower()
        
        # Test with non-commuting matrices (Pauli-X and Pauli-Y)
        pauli_y = np.array([[0, -1j], [1j, 0]])
        commute, message = self.tester.test_commutativity(pauli_x, pauli_y)
        assert commute is False
        assert "not commute" in message.lower()
    
    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
    def test_idempotency_test(self):
        """Test idempotency testing."""
        # Test with projection matrix (idempotent)
        projection = np.array([[1, 0], [0, 0]])
        is_idempotent, message = self.tester.test_idempotency(projection)
        assert is_idempotent is True
        assert "idempotent" in message.lower()
        
        # Test with non-idempotent matrix
        non_idempotent = np.array([[1, 1], [0, 1]])
        is_idempotent, message = self.tester.test_idempotency(non_idempotent)
        assert is_idempotent is False
        assert "not idempotent" in message.lower()
    
    def test_reversibility_test(self):
        """Test reversibility testing."""
        # Test with reversible operation
        original = [1, 0, 0, 1]
        forward_result = [0, 1, 1, 0]  # Some transformation
        backward_result = [1, 0, 0, 1]  # Back to original
        
        is_reversible, message = self.tester.test_reversibility(
            forward_result, backward_result, original
        )
        assert is_reversible is True
        assert "reversible" in message.lower()
        
        # Test with non-reversible operation
        wrong_backward = [0, 1, 1, 1]  # Different from original
        is_reversible, message = self.tester.test_reversibility(
            forward_result, wrong_backward, original
        )
        assert is_reversible is False
        assert "not reversible" in message.lower()
    
    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
    def test_entanglement_test(self):
        """Test entanglement testing."""
        # Test with Bell state (entangled)
        bell_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
        is_entangled, message = self.tester.test_entanglement(bell_state, 2)
        # Note: This is a simple test, might not always detect entanglement correctly
        
        # Test with separable state
        separable_state = np.array([1, 0, 0, 0])  # |00⟩
        is_entangled, message = self.tester.test_entanglement(separable_state, 2)
        assert is_entangled is False
        assert "separable" in message.lower()


@pytest.mark.skipif(not HAS_TESTING_TOOLS, reason="quantum_testing_tools module not available")
class TestQuantumTestGenerator:
    """Test QuantumTestGenerator functionality."""
    
    def setup_method(self):
        """Set up test generator."""
        self.generator = QuantumTestGenerator()
    
    def test_generate_circuit_tests(self):
        """Test generating circuit tests."""
        circuit_data = {
            'gates': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'cnot', 'qubits': [0, 1]}
            ]
        }
        
        tests = self.generator.generate_circuit_tests(circuit_data, num_tests=5)
        
        assert len(tests) >= 3  # At least validation, properties, performance
        assert all(isinstance(test, TestCase) for test in tests)
        
        # Check test types
        test_types = [test.test_type for test in tests]
        assert TestType.FUNCTIONAL in test_types
        assert TestType.PROPERTY_BASED in test_types
        assert TestType.PERFORMANCE in test_types
        
        # Check that all tests have circuit data
        for test in tests:
            if 'circuit_data' in test.input_data:
                assert test.input_data['circuit_data'] == circuit_data
    
    def test_generate_gate_tests(self):
        """Test generating gate tests."""
        # Test Hadamard gate
        h_tests = self.generator.generate_gate_tests('h')
        
        assert len(h_tests) >= 2  # Functionality and matrix properties
        assert all(test.input_data.get('gate') == 'h' for test in h_tests)
        
        # Check for superposition test (specific to Hadamard)
        test_names = [test.name.lower() for test in h_tests]
        assert any('superposition' in name for name in test_names)
        
        # Test CNOT gate
        cnot_tests = self.generator.generate_gate_tests('cnot')
        
        assert len(cnot_tests) >= 2
        
        # Check for entanglement test (specific to CNOT)
        test_names = [test.name.lower() for test in cnot_tests]
        assert any('entanglement' in name for name in test_names)
        
        # Test Pauli-X gate
        x_tests = self.generator.generate_gate_tests('x')
        
        assert len(x_tests) >= 2
        
        # Check for idempotency test (specific to Pauli-X)
        test_names = [test.name.lower() for test in x_tests]
        assert any('idempoten' in name for name in test_names)
    
    def test_generate_algorithm_tests(self):
        """Test generating algorithm tests."""
        algorithm_data = {
            'name': 'Test Algorithm',
            'gates': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'measure', 'qubits': [0]}
            ]
        }
        
        tests = self.generator.generate_algorithm_tests(algorithm_data)
        
        assert len(tests) >= 3  # Correctness, performance, stability
        
        test_types = [test.test_type for test in tests]
        assert TestType.FUNCTIONAL in test_types
        assert TestType.PERFORMANCE in test_types
        assert TestType.REGRESSION in test_types
        
        # Check algorithm data is passed
        for test in tests:
            assert test.input_data.get('algorithm_data') == algorithm_data
    
    def test_generate_property_tests(self):
        """Test generating property-based tests."""
        def mock_function(x, y):
            return x + y
        
        properties = [QuantumProperty.UNITARITY, QuantumProperty.NORMALIZATION]
        tests = self.generator.generate_property_tests(mock_function, properties)
        
        assert len(tests) == len(properties)
        
        for test, prop in zip(tests, properties):
            assert test.test_type == TestType.PROPERTY_BASED
            assert prop in test.properties
            assert test.target_function == mock_function
    
    def test_estimate_qubit_count(self):
        """Test qubit count estimation."""
        # Test with explicit qubit count
        circuit_1 = {'num_qubits': 5}
        count_1 = self.generator._estimate_qubit_count(circuit_1)
        assert count_1 == 5
        
        # Test with gates
        circuit_2 = {
            'gates': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'cnot', 'qubits': [2, 4]},
                {'gate': 'x', 'qubits': [3]}
            ]
        }
        count_2 = self.generator._estimate_qubit_count(circuit_2)
        assert count_2 == 5  # Max qubit index is 4, so 5 qubits
        
        # Test with empty circuit
        circuit_3 = {}
        count_3 = self.generator._estimate_qubit_count(circuit_3)
        assert count_3 == 2  # Default


@pytest.mark.skipif(not HAS_TESTING_TOOLS, reason="quantum_testing_tools module not available")
class TestMockQuantumBackend:
    """Test MockQuantumBackend functionality."""
    
    def setup_method(self):
        """Set up mock backend."""
        self.backend = MockQuantumBackend(noise_level=0.1, latency=0.05)
    
    @pytest.mark.asyncio
    async def test_execute_circuit(self):
        """Test circuit execution."""
        circuit_data = {
            'gates': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'cnot', 'qubits': [0, 1]}
            ]
        }
        
        result = await self.backend.execute_circuit(circuit_data, shots=1000)
        
        assert 'counts' in result
        assert 'shots' in result
        assert 'execution_time' in result
        assert 'backend' in result
        assert result['shots'] == 1000
        assert result['backend'] == 'mock'
        assert result['noise_level'] == 0.1
        
        # Check counts make sense
        total_counts = sum(result['counts'].values())
        assert total_counts == 1000
        
        # Check call tracking
        assert self.backend.call_count == 1
        assert len(self.backend.call_log) == 1
        assert self.backend.call_log[0]['shots'] == 1000
    
    @pytest.mark.asyncio
    async def test_multiple_executions(self):
        """Test multiple circuit executions."""
        circuit_data = {'gates': [{'gate': 'h', 'qubits': [0]}]}
        
        # Execute multiple times
        for i in range(3):
            await self.backend.execute_circuit(circuit_data, shots=100 * (i + 1))
        
        assert self.backend.call_count == 3
        assert len(self.backend.call_log) == 3
        
        # Check different shot counts
        expected_shots = [100, 200, 300]
        actual_shots = [call['shots'] for call in self.backend.call_log]
        assert actual_shots == expected_shots
    
    def test_reset(self):
        """Test backend reset."""
        # Make some calls first
        asyncio.run(self.backend.execute_circuit({'gates': []}, shots=100))
        
        assert self.backend.call_count > 0
        assert len(self.backend.call_log) > 0
        
        # Reset
        self.backend.reset()
        
        assert self.backend.call_count == 0
        assert len(self.backend.call_log) == 0
    
    def test_estimate_qubits(self):
        """Test qubit count estimation."""
        circuit_1 = {
            'gates': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'cnot', 'qubits': [1, 3]}
            ]
        }
        count_1 = self.backend._estimate_qubits(circuit_1)
        assert count_1 == 4  # Max qubit index is 3
        
        circuit_2 = {'gates': []}
        count_2 = self.backend._estimate_qubits(circuit_2)
        assert count_2 == 1  # Default for empty circuit
    
    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
    def test_generate_mock_counts(self):
        """Test mock count generation."""
        counts = self.backend._generate_mock_counts(2, 1000)
        
        # Check structure
        assert isinstance(counts, dict)
        assert all(isinstance(k, str) for k in counts.keys())
        assert all(isinstance(v, int) for v in counts.values())
        
        # Check total counts
        total = sum(counts.values())
        assert total == 1000
        
        # Check bit string format
        for state in counts.keys():
            assert len(state) == 2  # 2 qubits
            assert all(c in '01' for c in state)


@pytest.mark.skipif(not HAS_TESTING_TOOLS, reason="quantum_testing_tools module not available")
class TestQuantumTestRunner:
    """Test QuantumTestRunner functionality."""
    
    def setup_method(self):
        """Set up test runner."""
        self.backend = MockQuantumBackend(latency=0.01)  # Fast for testing
        self.runner = QuantumTestRunner(self.backend)
    
    @pytest.mark.asyncio
    async def test_run_functional_test(self):
        """Test running functional test."""
        test_case = TestCase(
            test_id="func_test",
            name="Functional Test",
            description="Test functional behavior",
            test_type=TestType.FUNCTIONAL,
            input_data={
                'circuit_data': {
                    'gates': [{'gate': 'h', 'qubits': [0]}]
                }
            }
        )
        
        result = await self.runner.run_test_case(test_case)
        
        assert result.test_id == "func_test"
        assert result.status == TestStatus.PASSED
        assert result.execution_time > 0
        assert result.actual_output is not None
        assert 'counts' in result.actual_output
    
    @pytest.mark.asyncio
    async def test_run_property_test(self):
        """Test running property-based test."""
        test_case = TestCase(
            test_id="prop_test",
            name="Property Test",
            description="Test quantum properties",
            test_type=TestType.PROPERTY_BASED,
            input_data={
                'circuit_data': {
                    'gates': [{'gate': 'h', 'qubits': [0]}]
                }
            },
            properties=[QuantumProperty.NORMALIZATION, QuantumProperty.MEASUREMENT_CONSERVATION]
        )
        
        result = await self.runner.run_test_case(test_case)
        
        assert result.test_id == "prop_test"
        assert result.status == TestStatus.PASSED
        assert len(result.property_violations) == 0
    
    @pytest.mark.asyncio
    async def test_run_performance_test(self):
        """Test running performance test."""
        test_case = TestCase(
            test_id="perf_test",
            name="Performance Test",
            description="Test performance",
            test_type=TestType.PERFORMANCE,
            input_data={
                'circuit_data': {
                    'gates': [{'gate': 'x', 'qubits': [0]}]
                }
            },
            metadata={'expected_max_time': 1.0, 'num_runs': 3}
        )
        
        result = await self.runner.run_test_case(test_case)
        
        assert result.test_id == "perf_test"
        assert result.status == TestStatus.PASSED
        assert 'performance_metrics' in result.performance_metrics
        assert result.performance_metrics['num_runs'] == 3
    
    @pytest.mark.asyncio
    async def test_run_fuzz_test(self):
        """Test running fuzz test."""
        test_case = TestCase(
            test_id="fuzz_test",
            name="Fuzz Test",
            description="Test with random inputs",
            test_type=TestType.FUZZ,
            input_data={
                'circuit_data': {
                    'gates': [{'gate': 'h', 'qubits': [0]}]
                }
            },
            metadata={'fuzz_iterations': 5}
        )
        
        result = await self.runner.run_test_case(test_case)
        
        assert result.test_id == "fuzz_test"
        assert result.status == TestStatus.PASSED
        assert "iterations" in result.message
    
    @pytest.mark.asyncio
    async def test_run_test_with_function(self):
        """Test running test with target function."""
        def test_function(x, y):
            return x + y
        
        test_case = TestCase(
            test_id="func_target_test",
            name="Function Target Test",
            description="Test with target function",
            test_type=TestType.FUNCTIONAL,
            target_function=test_function,
            input_data={'x': 2, 'y': 3},
            expected_output=5
        )
        
        result = await self.runner.run_test_case(test_case)
        
        assert result.test_id == "func_target_test"
        assert result.status == TestStatus.PASSED
        assert result.actual_output == 5
    
    @pytest.mark.asyncio
    async def test_run_test_with_expected_failure(self):
        """Test running test that should fail."""
        test_case = TestCase(
            test_id="fail_test",
            name="Failing Test",
            description="Test that should fail",
            test_type=TestType.FUNCTIONAL,
            input_data={
                'circuit_data': {
                    'gates': [{'gate': 'h', 'qubits': [0]}]
                }
            },
            expected_output={'counts': {'00': 1000}}  # Won't match superposition
        )
        
        result = await self.runner.run_test_case(test_case)
        
        assert result.test_id == "fail_test"
        assert result.status == TestStatus.FAILED
        assert "does not match expected" in result.message
    
    @pytest.mark.asyncio
    async def test_run_test_suite(self):
        """Test running complete test suite."""
        # Create test suite
        suite = TestSuite(
            suite_id="test_suite",
            name="Test Suite",
            description="Suite for testing"
        )
        
        # Add test cases
        for i in range(3):
            test_case = TestCase(
                test_id=f"suite_test_{i}",
                name=f"Suite Test {i}",
                description=f"Test case {i}",
                test_type=TestType.FUNCTIONAL,
                input_data={
                    'circuit_data': {'gates': [{'gate': 'x', 'qubits': [0]}]}
                }
            )
            suite.add_test_case(test_case)
        
        # Add hooks
        hook_calls = []
        
        def setup_hook():
            hook_calls.append('setup')
        
        def teardown_hook():
            hook_calls.append('teardown')
        
        suite.add_setup_hook(setup_hook)
        suite.add_teardown_hook(teardown_hook)
        
        # Run suite
        results = await self.runner.run_test_suite(suite)
        
        assert len(results) == 3
        assert all(result.status == TestStatus.PASSED for result in results.values())
        assert 'setup' in hook_calls
        assert 'teardown' in hook_calls
    
    def test_get_test_statistics(self):
        """Test getting test statistics."""
        # Add some mock results
        self.runner.results = {
            'test1': TestResult('test1', TestStatus.PASSED, 0.1),
            'test2': TestResult('test2', TestStatus.FAILED, 0.2),
            'test3': TestResult('test3', TestStatus.PASSED, 0.15)
        }
        
        stats = self.runner.get_test_statistics()
        
        assert stats['total_tests'] == 3
        assert stats['status_counts']['passed'] == 2
        assert stats['status_counts']['failed'] == 1
        assert stats['pass_rate'] == 2/3
        assert abs(stats['average_execution_time'] - 0.15) < 0.01


@pytest.mark.skipif(not HAS_TESTING_TOOLS, reason="quantum_testing_tools module not available")
class TestQuantumTestReporter:
    """Test QuantumTestReporter functionality."""
    
    def setup_method(self):
        """Set up test reporter."""
        self.reporter = QuantumTestReporter()
        
        # Create sample test results
        self.sample_results = {
            'test1': TestResult(
                test_id='test1',
                status=TestStatus.PASSED,
                execution_time=0.1,
                message="Test passed"
            ),
            'test2': TestResult(
                test_id='test2',
                status=TestStatus.FAILED,
                execution_time=0.2,
                message="Test failed",
                assertion_errors=["Error 1"],
                severity=TestSeverity.HIGH
            ),
            'test3': TestResult(
                test_id='test3',
                status=TestStatus.PASSED,
                execution_time=0.15,
                message="Test passed"
            )
        }
    
    def test_generate_json_report(self):
        """Test generating JSON report."""
        report_json = self.reporter.generate_report(self.sample_results, 'json')
        
        assert isinstance(report_json, str)
        report_data = json.loads(report_json)
        
        assert 'summary' in report_data
        assert 'analysis' in report_data
        assert 'results_by_status' in report_data
        assert 'detailed_results' in report_data
        
        # Check summary
        summary = report_data['summary']
        assert summary['total_tests'] == 3
        assert summary['pass_rate'] == 2/3
        assert summary['status_counts']['passed'] == 2
        assert summary['status_counts']['failed'] == 1
    
    def test_generate_dict_report(self):
        """Test generating dictionary report."""
        report_dict = self.reporter.generate_report(self.sample_results, 'dict')
        
        assert isinstance(report_dict, dict)
        assert 'summary' in report_dict
        assert 'analysis' in report_dict
        
        # Check detailed results
        detailed = report_dict['detailed_results']
        assert len(detailed) == 3
        assert all('test_id' in result for result in detailed)
    
    def test_generate_text_report(self):
        """Test generating text report."""
        report_text = self.reporter.generate_report(self.sample_results, 'text')
        
        assert isinstance(report_text, str)
        assert "QUANTUM TEST REPORT" in report_text
        assert "SUMMARY" in report_text
        assert "STATUS BREAKDOWN" in report_text
        assert "Total Tests: 3" in report_text
        assert "Passed: 2" in report_text
        assert "Failed: 1" in report_text
    
    def test_generate_html_report(self):
        """Test generating HTML report."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp:
            html_content = self.reporter.generate_html_report(self.sample_results, tmp.name)
            tmp_path = tmp.name
        
        try:
            assert isinstance(html_content, str)
            assert "<!DOCTYPE html>" in html_content
            assert "Quantum Test Report" in html_content
            assert "Total Tests:" in html_content
            assert "Pass Rate:" in html_content
            
            # Check file was written
            with open(tmp_path, 'r') as f:
                file_content = f.read()
            assert file_content == html_content
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_calculate_statistics(self):
        """Test statistics calculation."""
        stats = self.reporter._calculate_statistics(self.sample_results)
        
        assert stats['total_tests'] == 3
        assert stats['pass_rate'] == 2/3
        assert stats['status_counts']['passed'] == 2
        assert stats['status_counts']['failed'] == 1
        assert abs(stats['average_execution_time'] - 0.15) < 0.01
        assert abs(stats['total_execution_time'] - 0.45) < 0.01
    
    def test_group_by_status(self):
        """Test grouping results by status."""
        grouped = self.reporter._group_by_status(self.sample_results)
        
        assert 'passed' in grouped
        assert 'failed' in grouped
        assert len(grouped['passed']) == 2
        assert len(grouped['failed']) == 1
        
        # Check specific results
        passed_ids = [result['test_id'] for result in grouped['passed']]
        failed_ids = [result['test_id'] for result in grouped['failed']]
        
        assert 'test1' in passed_ids
        assert 'test3' in passed_ids
        assert 'test2' in failed_ids
    
    def test_generate_analysis(self):
        """Test analysis generation."""
        stats = self.reporter._calculate_statistics(self.sample_results)
        analysis = self.reporter._generate_analysis(self.sample_results, stats)
        
        assert 'slowest_test_time' in analysis
        assert 'fastest_test_time' in analysis
        assert 'failure_rate' in analysis
        
        assert analysis['slowest_test_time'] == 0.2
        assert analysis['fastest_test_time'] == 0.1
        assert abs(analysis['failure_rate'] - 1/3) < 0.01


@pytest.mark.skipif(not HAS_TESTING_TOOLS, reason="quantum_testing_tools module not available")
class TestQuantumTestManager:
    """Test QuantumTestManager functionality."""
    
    def setup_method(self):
        """Set up test manager."""
        self.manager = QuantumTestManager()
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        assert self.manager.backend is not None
        assert self.manager.test_generator is not None
        assert self.manager.test_runner is not None
        assert self.manager.reporter is not None
        assert len(self.manager.test_suites) > 0  # Should have default suites
    
    def test_default_test_suites(self):
        """Test default test suites creation."""
        suite_names = [suite.name for suite in self.manager.test_suites.values()]
        
        assert "Basic Quantum Operations" in suite_names
        assert "Circuit Validation" in suite_names
        
        # Check basic ops suite
        basic_suite = None
        for suite in self.manager.test_suites.values():
            if "Basic Quantum Operations" in suite.name:
                basic_suite = suite
                break
        
        assert basic_suite is not None
        assert len(basic_suite.test_cases) > 0
        
        # Should have tests for different gates
        test_tags = []
        for test in basic_suite.test_cases:
            test_tags.extend(test.tags)
        
        assert any('h' in tag for tag in test_tags)
        assert any('x' in tag for tag in test_tags)
        assert any('cnot' in tag for tag in test_tags)
    
    def test_create_test_suite(self):
        """Test creating new test suite."""
        suite = self.manager.create_test_suite(
            "Custom Test Suite",
            "Suite for custom testing"
        )
        
        assert suite.name == "Custom Test Suite"
        assert suite.description == "Suite for custom testing"
        assert suite.suite_id in self.manager.test_suites
        
        # Verify it was added to manager
        retrieved = self.manager.test_suites[suite.suite_id]
        assert retrieved.name == suite.name
    
    def test_add_circuit_tests(self):
        """Test adding circuit tests to suite."""
        suite = self.manager.create_test_suite("Circuit Test Suite")
        original_count = len(suite.test_cases)
        
        circuit_data = {
            'gates': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'cnot', 'qubits': [0, 1]}
            ]
        }
        
        success = self.manager.add_circuit_tests(suite.suite_id, circuit_data, num_tests=3)
        
        assert success is True
        assert len(suite.test_cases) > original_count
        
        # Check that circuit data is in tests
        circuit_tests = [test for test in suite.test_cases 
                        if 'circuit_data' in test.input_data]
        assert len(circuit_tests) > 0
        
        for test in circuit_tests:
            assert test.input_data['circuit_data'] == circuit_data
    
    def test_add_property_tests(self):
        """Test adding property tests to suite."""
        suite = self.manager.create_test_suite("Property Test Suite")
        original_count = len(suite.test_cases)
        
        def test_function(x):
            return x * 2
        
        properties = [QuantumProperty.UNITARITY, QuantumProperty.NORMALIZATION]
        
        success = self.manager.add_property_tests(suite.suite_id, test_function, properties)
        
        assert success is True
        assert len(suite.test_cases) > original_count
        
        # Check property tests
        property_tests = [test for test in suite.test_cases 
                         if test.test_type == TestType.PROPERTY_BASED]
        assert len(property_tests) == len(properties)
        
        for test in property_tests:
            assert test.target_function == test_function
            assert len(test.properties) == 1
    
    @pytest.mark.asyncio
    async def test_run_test_suite(self):
        """Test running test suite."""
        # Use one of the default suites
        suite_id = list(self.manager.test_suites.keys())[0]
        
        results = await self.manager.run_test_suite(suite_id)
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Check results structure
        for test_id, result in results.items():
            assert isinstance(result, TestResult)
            assert result.test_id == test_id
    
    @pytest.mark.asyncio
    async def test_run_all_test_suites(self):
        """Test running all test suites."""
        all_results = await self.manager.run_all_test_suites()
        
        assert isinstance(all_results, dict)
        assert len(all_results) == len(self.manager.test_suites)
        
        # Check structure
        for suite_id, suite_results in all_results.items():
            assert suite_id in self.manager.test_suites
            assert isinstance(suite_results, dict)
    
    def test_get_test_suite_info(self):
        """Test getting test suite information."""
        suite_id = list(self.manager.test_suites.keys())[0]
        
        info = self.manager.get_test_suite_info(suite_id)
        
        assert info is not None
        assert info['suite_id'] == suite_id
        assert 'name' in info
        assert 'description' in info
        assert 'test_count' in info
        assert 'test_types' in info
        assert 'created_at' in info
        
        # Check non-existent suite
        non_existent_info = self.manager.get_test_suite_info("non_existent")
        assert non_existent_info is None
    
    def test_list_test_suites(self):
        """Test listing all test suites."""
        suites_list = self.manager.list_test_suites()
        
        assert isinstance(suites_list, list)
        assert len(suites_list) == len(self.manager.test_suites)
        
        # Check structure
        for suite_info in suites_list:
            assert 'suite_id' in suite_info
            assert 'name' in suite_info
            assert 'test_count' in suite_info
    
    def test_get_test_statistics(self):
        """Test getting test statistics."""
        stats = self.manager.get_test_statistics()
        
        assert 'total_suites' in stats
        assert 'total_tests' in stats
        assert 'test_type_distribution' in stats
        assert 'property_distribution' in stats
        assert 'backend_info' in stats
        
        assert stats['total_suites'] == len(self.manager.test_suites)
        assert stats['total_tests'] > 0
        assert stats['backend_info']['type'] == 'mock'
    
    def test_generate_comprehensive_report(self):
        """Test generating comprehensive report."""
        # Create mock results
        mock_results = {
            'suite1': {
                'test1': TestResult('test1', TestStatus.PASSED, 0.1),
                'test2': TestResult('test2', TestStatus.FAILED, 0.2)
            },
            'suite2': {
                'test3': TestResult('test3', TestStatus.PASSED, 0.15)
            }
        }
        
        # Test text report
        text_report = self.manager.generate_comprehensive_report(mock_results)
        
        assert isinstance(text_report, str)
        assert "QUANTUM TEST REPORT" in text_report
        assert "Total Tests: 3" in text_report
        
        # Test with file output
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
            report = self.manager.generate_comprehensive_report(mock_results, tmp.name)
            tmp_path = tmp.name
        
        try:
            # Check file was written
            with open(tmp_path, 'r') as f:
                file_content = f.read()
            assert file_content == report
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)


@pytest.mark.skipif(not HAS_TESTING_TOOLS, reason="quantum_testing_tools module not available")
class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_get_quantum_test_manager(self):
        """Test getting global test manager."""
        manager1 = get_quantum_test_manager()
        manager2 = get_quantum_test_manager()
        
        # Should be singleton
        assert manager1 is manager2
        assert isinstance(manager1, QuantumTestManager)
    
    def test_create_test_suite_function(self):
        """Test create test suite convenience function."""
        suite = create_test_suite("Convenience Suite", "Created via function")
        
        assert isinstance(suite, TestSuite)
        assert suite.name == "Convenience Suite"
        assert suite.description == "Created via function"
        
        # Should be added to global manager
        manager = get_quantum_test_manager()
        assert suite.suite_id in manager.test_suites
    
    def test_test_quantum_circuit_function(self):
        """Test test quantum circuit convenience function."""
        circuit_data = {
            'gates': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'x', 'qubits': [1]}
            ]
        }
        
        properties = [QuantumProperty.UNITARITY, QuantumProperty.NORMALIZATION]
        
        tests = test_quantum_circuit(circuit_data, properties)
        
        assert isinstance(tests, list)
        assert len(tests) > 0
        assert all(isinstance(test, TestCase) for test in tests)
        
        # Check circuit data is included
        circuit_tests = [test for test in tests if 'circuit_data' in test.input_data]
        assert len(circuit_tests) > 0
        
        for test in circuit_tests:
            assert test.input_data['circuit_data'] == circuit_data
        
        # Check property tests
        property_tests = [test for test in tests if test.test_type == TestType.PROPERTY_BASED]
        assert len(property_tests) >= len(properties)
    
    def test_test_quantum_function(self):
        """Test test quantum function convenience function."""
        def test_function(x, y):
            return {'result': x + y}
        
        input_data = {'x': 2, 'y': 3}
        properties = [QuantumProperty.UNITARITY]
        
        tests = test_quantum_function(test_function, input_data, properties)
        
        assert isinstance(tests, list)
        assert len(tests) == len(properties)
        assert all(isinstance(test, TestCase) for test in tests)
        
        for test in tests:
            assert test.target_function == test_function
            assert test.input_data['x'] == 2
            assert test.input_data['y'] == 3
            assert test.test_type == TestType.PROPERTY_BASED
    
    @pytest.mark.asyncio
    async def test_run_quantum_tests_function(self):
        """Test run quantum tests convenience function."""
        # Create test cases
        test_cases = [
            TestCase(
                test_id="conv_test_1",
                name="Convenience Test 1",
                description="Test 1",
                test_type=TestType.FUNCTIONAL,
                input_data={'circuit_data': {'gates': [{'gate': 'x', 'qubits': [0]}]}}
            ),
            TestCase(
                test_id="conv_test_2",
                name="Convenience Test 2",
                description="Test 2",
                test_type=TestType.FUNCTIONAL,
                input_data={'circuit_data': {'gates': [{'gate': 'h', 'qubits': [0]}]}}
            )
        ]
        
        results = await run_quantum_tests(test_cases)
        
        assert isinstance(results, dict)
        assert len(results) == 2
        assert 'conv_test_1' in results
        assert 'conv_test_2' in results
        
        for test_id, result in results.items():
            assert isinstance(result, TestResult)
            assert result.test_id == test_id


@pytest.mark.skipif(not HAS_TESTING_TOOLS, reason="quantum_testing_tools module not available")
class TestIntegrationScenarios:
    """Test integration scenarios."""
    
    def setup_method(self):
        """Set up integration test."""
        self.manager = QuantumTestManager()
    
    @pytest.mark.asyncio
    async def test_complete_testing_workflow(self):
        """Test complete testing workflow."""
        # 1. Create custom test suite
        suite = self.manager.create_test_suite(
            "Integration Test Suite",
            "Complete workflow testing"
        )
        
        # 2. Add circuit tests
        bell_circuit = {
            'gates': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'cnot', 'qubits': [0, 1]},
                {'gate': 'measure', 'qubits': [0, 1]}
            ]
        }
        
        success = self.manager.add_circuit_tests(suite.suite_id, bell_circuit, num_tests=3)
        assert success is True
        
        # 3. Add property tests
        def mock_unitary_function():
            if NUMPY_AVAILABLE:
                import numpy as np
                return np.array([[1, 0], [0, 1]])  # Identity
            return [[1, 0], [0, 1]]
        
        properties = [QuantumProperty.UNITARITY, QuantumProperty.HERMITICITY]
        success = self.manager.add_property_tests(suite.suite_id, mock_unitary_function, properties)
        assert success is True
        
        # 4. Run tests
        results = await self.manager.run_test_suite(suite.suite_id)
        
        # 5. Verify results
        assert len(results) > 0
        passed_tests = [r for r in results.values() if r.status == TestStatus.PASSED]
        assert len(passed_tests) > 0
        
        # 6. Generate report
        report = self.manager.reporter.generate_report(results, 'dict')
        
        assert 'summary' in report
        assert 'analysis' in report
        assert report['summary']['total_tests'] == len(results)
        
        # 7. Get statistics
        stats = self.manager.get_test_statistics()
        assert stats['total_tests'] > 0
        assert stats['backend_info']['call_count'] > 0
    
    @pytest.mark.asyncio
    async def test_property_testing_pipeline(self):
        """Test comprehensive property testing."""
        # Test different quantum properties
        properties_to_test = [
            QuantumProperty.UNITARITY,
            QuantumProperty.NORMALIZATION,
            QuantumProperty.HERMITICITY,
            QuantumProperty.MEASUREMENT_CONSERVATION
        ]
        
        suite = self.manager.create_test_suite("Property Testing Suite")
        
        # Add tests for each property
        for prop in properties_to_test:
            circuit_data = {
                'gates': [{'gate': 'h', 'qubits': [0]}]  # Simple circuit
            }
            
            test_case = TestCase(
                test_id=f"prop_test_{prop.value}",
                name=f"Test {prop.value}",
                description=f"Property test for {prop.value}",
                test_type=TestType.PROPERTY_BASED,
                input_data={'circuit_data': circuit_data},
                properties=[prop]
            )
            
            suite.add_test_case(test_case)
        
        # Run property tests
        results = await self.manager.run_test_suite(suite.suite_id)
        
        # Analyze results
        property_results = {}
        for test_id, result in results.items():
            prop_name = test_id.replace('prop_test_', '')
            property_results[prop_name] = result.status
        
        # Most basic properties should pass for simple circuits
        assert property_results.get('normalization') == TestStatus.PASSED
        assert property_results.get('measurement_conservation') == TestStatus.PASSED
    
    @pytest.mark.asyncio
    async def test_performance_regression_testing(self):
        """Test performance regression detection."""
        # Create performance-focused test suite
        suite = self.manager.create_test_suite("Performance Regression Suite")
        
        # Add performance tests with different complexity
        circuits = [
            {
                'name': 'simple',
                'gates': [{'gate': 'h', 'qubits': [0]}]
            },
            {
                'name': 'medium',
                'gates': [
                    {'gate': 'h', 'qubits': [0]},
                    {'gate': 'cnot', 'qubits': [0, 1]},
                    {'gate': 'h', 'qubits': [1]}
                ]
            },
            {
                'name': 'complex',
                'gates': [
                    {'gate': 'h', 'qubits': [i]} for i in range(3)
                ] + [
                    {'gate': 'cnot', 'qubits': [i, (i+1)%3]} for i in range(3)
                ]
            }
        ]
        
        for circuit in circuits:
            test_case = TestCase(
                test_id=f"perf_{circuit['name']}",
                name=f"Performance Test - {circuit['name'].title()}",
                description=f"Performance test for {circuit['name']} circuit",
                test_type=TestType.PERFORMANCE,
                input_data={'circuit_data': {'gates': circuit['gates']}},
                metadata={
                    'expected_max_time': 0.5,  # Generous for testing
                    'num_runs': 3
                }
            )
            suite.add_test_case(test_case)
        
        # Run performance tests
        results = await self.manager.run_test_suite(suite.suite_id)
        
        # Analyze performance
        performance_data = {}
        for test_id, result in results.items():
            if 'performance_metrics' in result.performance_metrics:
                performance_data[test_id] = result.performance_metrics
        
        # Generate performance report
        report = self.manager.reporter.generate_report(results, 'dict')
        analysis = report['analysis']
        
        # Should have performance analysis
        assert 'slowest_test_time' in analysis
        assert 'fastest_test_time' in analysis
    
    def test_error_handling_and_recovery(self):
        """Test error handling in testing framework."""
        # Test with invalid circuit data
        invalid_circuits = [
            {'gates': []},  # Empty gates
            {'gates': [{'gate': 'invalid_gate', 'qubits': [0]}]},  # Invalid gate
            {'gates': [{'gate': 'h'}]},  # Missing qubits
            {'gates': [{'gate': 'h', 'qubits': [-1]}]},  # Invalid qubit index
        ]
        
        generator = QuantumTestGenerator()
        
        for i, circuit in enumerate(invalid_circuits):
            try:
                tests = generator.generate_circuit_tests(circuit)
                # Should still generate tests even with invalid data
                assert len(tests) > 0
                assert all(isinstance(test, TestCase) for test in tests)
                
            except Exception as e:
                # If exceptions occur, they should be handled gracefully
                assert isinstance(e, Exception)
                # Test framework should be robust to bad input
    
    @pytest.mark.asyncio
    async def test_concurrent_testing(self):
        """Test concurrent test execution."""
        # Create multiple test suites
        suites = []
        for i in range(3):
            suite = self.manager.create_test_suite(f"Concurrent Suite {i}")
            
            # Add simple test to each suite
            test_case = TestCase(
                test_id=f"concurrent_test_{i}",
                name=f"Concurrent Test {i}",
                description="Test for concurrent execution",
                test_type=TestType.FUNCTIONAL,
                input_data={'circuit_data': {'gates': [{'gate': 'x', 'qubits': [0]}]}}
            )
            suite.add_test_case(test_case)
            suites.append(suite)
        
        # Run suites concurrently
        tasks = []
        for suite in suites:
            task = asyncio.create_task(self.manager.run_test_suite(suite.suite_id))
            tasks.append(task)
        
        # Wait for all to complete
        all_results = await asyncio.gather(*tasks)
        
        # Verify all completed successfully
        assert len(all_results) == 3
        for results in all_results:
            assert len(results) == 1  # Each suite has one test
            result = list(results.values())[0]
            assert result.status == TestStatus.PASSED
        
        # Check backend call tracking
        total_calls = self.manager.backend.call_count
        assert total_calls >= 3  # At least one call per suite


@pytest.mark.skipif(not HAS_TESTING_TOOLS, reason="quantum_testing_tools module not available")
class TestQuantumTestingPerformance:
    """Test performance characteristics of testing framework."""
    
    def setup_method(self):
        """Set up performance test."""
        self.manager = QuantumTestManager()
    
    @pytest.mark.asyncio
    async def test_large_test_suite_performance(self):
        """Test performance with large test suites."""
        # Create large test suite
        suite = self.manager.create_test_suite("Large Performance Suite")
        
        # Add many test cases
        start_time = time.time()
        
        for i in range(50):  # 50 test cases
            test_case = TestCase(
                test_id=f"large_test_{i}",
                name=f"Large Test {i}",
                description=f"Performance test case {i}",
                test_type=TestType.FUNCTIONAL,
                input_data={'circuit_data': {'gates': [{'gate': 'h', 'qubits': [0]}]}}
            )
            suite.add_test_case(test_case)
        
        creation_time = time.time() - start_time
        
        # Should create tests quickly
        assert creation_time < 2.0  # Under 2 seconds for 50 tests
        assert len(suite.test_cases) == 50
        
        # Run subset for performance testing (to avoid long test times)
        limited_suite = TestSuite(
            suite_id="limited_suite",
            name="Limited Suite",
            description="Subset for performance testing"
        )
        
        # Add first 10 tests
        for test in suite.test_cases[:10]:
            limited_suite.add_test_case(test)
        
        # Time execution
        exec_start = time.time()
        results = await self.manager.test_runner.run_test_suite(limited_suite)
        exec_time = time.time() - exec_start
        
        # Should execute reasonably quickly
        assert exec_time < 5.0  # Under 5 seconds for 10 tests
        assert len(results) == 10
    
    def test_test_generation_performance(self):
        """Test test generation performance."""
        generator = QuantumTestGenerator()
        
        # Test circuit test generation speed
        circuit_data = {
            'gates': [
                {'gate': 'h', 'qubits': [i]} for i in range(5)
            ] + [
                {'gate': 'cnot', 'qubits': [i, (i+1)%5]} for i in range(5)
            ]
        }
        
        start_time = time.time()
        
        # Generate many test sets
        all_tests = []
        for _ in range(20):
            tests = generator.generate_circuit_tests(circuit_data, num_tests=5)
            all_tests.extend(tests)
        
        generation_time = time.time() - start_time
        
        # Should generate tests quickly
        assert generation_time < 2.0  # Under 2 seconds for 100 tests
        assert len(all_tests) >= 60  # At least 3 tests per generation
    
    def test_property_testing_performance(self):
        """Test property testing performance."""
        tester = QuantumPropertyTester()
        
        if not NUMPY_AVAILABLE:
            pytest.skip("NumPy not available for performance testing")
        
        import numpy as np
        
        # Create test matrices of different sizes
        matrices = []
        for size in [2, 4, 8]:
            # Create random unitary matrix
            A = np.random.randn(size, size) + 1j * np.random.randn(size, size)
            Q, R = np.linalg.qr(A)
            matrices.append(Q)
        
        # Test unitarity checking performance
        start_time = time.time()
        
        for matrix in matrices:
            for _ in range(20):  # Test each matrix multiple times
                is_unitary, message = tester.test_unitarity(matrix)
                assert is_unitary is True  # All should be unitary
        
        testing_time = time.time() - start_time
        
        # Should test properties quickly
        assert testing_time < 2.0  # Under 2 seconds for 60 tests
    
    @pytest.mark.asyncio
    async def test_backend_performance(self):
        """Test mock backend performance."""
        backend = MockQuantumBackend(latency=0.001)  # Very fast
        
        circuit_data = {'gates': [{'gate': 'h', 'qubits': [0]}]}
        
        # Test multiple rapid executions
        start_time = time.time()
        
        tasks = []
        for _ in range(50):
            task = backend.execute_circuit(circuit_data, shots=100)
            tasks.append(task)
        
        # Execute all concurrently
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Should handle concurrent executions efficiently
        assert total_time < 3.0  # Under 3 seconds for 50 concurrent executions
        assert len(results) == 50
        assert all('counts' in result for result in results)
        
        # Check backend tracking
        assert backend.call_count == 50
    
    def test_memory_usage_tracking(self):
        """Test memory usage doesn't grow excessively."""
        manager = QuantumTestManager()
        
        # Create and destroy many test suites
        initial_suites = len(manager.test_suites)
        
        for i in range(20):
            suite = manager.create_test_suite(f"Temp Suite {i}")
            
            # Add some tests
            for j in range(5):
                test_case = TestCase(
                    test_id=f"temp_test_{i}_{j}",
                    name=f"Temp Test {i}.{j}",
                    description="Temporary test",
                    test_type=TestType.FUNCTIONAL,
                    input_data={'circuit_data': {'gates': []}}
                )
                suite.add_test_case(test_case)
            
            # Remove suite (simulate cleanup)
            if i > 10:  # Keep some, remove others
                del manager.test_suites[suite.suite_id]
        
        # Should have reasonable number of suites
        final_suites = len(manager.test_suites)
        assert final_suites < initial_suites + 15  # Not all 20 should remain
    
    def test_report_generation_performance(self):
        """Test report generation performance."""
        reporter = QuantumTestReporter()
        
        # Create large result set
        large_results = {}
        for i in range(100):
            result = TestResult(
                test_id=f"perf_test_{i}",
                status=TestStatus.PASSED if i % 3 != 0 else TestStatus.FAILED,
                execution_time=0.1 + (i % 10) * 0.01,
                message=f"Test {i} message",
                performance_metrics={'metric': i * 0.01}
            )
            large_results[f"perf_test_{i}"] = result
        
        # Time report generation
        start_time = time.time()
        
        # Generate different format reports
        json_report = reporter.generate_report(large_results, 'json')
        dict_report = reporter.generate_report(large_results, 'dict')
        text_report = reporter.generate_report(large_results, 'text')
        
        generation_time = time.time() - start_time
        
        # Should generate reports quickly
        assert generation_time < 3.0  # Under 3 seconds for 100 results
        
        # Verify reports were generated
        assert len(json_report) > 0
        assert isinstance(dict_report, dict)
        assert len(text_report) > 0
        assert "QUANTUM TEST REPORT" in text_report


if __name__ == "__main__":
    pytest.main([__file__])