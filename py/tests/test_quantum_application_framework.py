#!/usr/bin/env python3
"""
Test suite for quantum application framework functionality.
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
    from quantrs2.quantum_application_framework import (
        ApplicationState, ApplicationType, ExecutionMode, ResourceType,
        ResourceRequirement, ApplicationConfig, ExecutionContext,
        QuantumApplication, AlgorithmApplication, OptimizationApplication,
        WorkflowStep, QuantumWorkflow, ResourceManager, ApplicationTemplate,
        QuantumApplicationRuntime, get_quantum_runtime,
        create_algorithm_application, create_optimization_application,
        run_quantum_algorithm, create_workflow
    )
    HAS_FRAMEWORK = True
except ImportError:
    HAS_FRAMEWORK = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@pytest.mark.skipif(not HAS_FRAMEWORK, reason="quantum_application_framework module not available")
class TestResourceRequirement:
    """Test ResourceRequirement functionality."""
    
    def test_requirement_creation(self):
        """Test creating resource requirement."""
        req = ResourceRequirement(
            resource_type=ResourceType.QUBITS,
            minimum=2,
            preferred=4,
            maximum=8,
            unit="qubits"
        )
        
        assert req.resource_type == ResourceType.QUBITS
        assert req.minimum == 2
        assert req.preferred == 4
        assert req.maximum == 8
        assert req.unit == "qubits"
    
    def test_requirement_serialization(self):
        """Test requirement serialization."""
        req = ResourceRequirement(
            resource_type=ResourceType.CLASSICAL_MEMORY,
            minimum=1024,
            preferred=2048,
            unit="MB"
        )
        
        req_dict = req.to_dict()
        
        assert req_dict['resource_type'] == 'classical_memory'
        assert req_dict['minimum'] == 1024
        assert req_dict['preferred'] == 2048
        assert req_dict['unit'] == "MB"
    
    def test_requirement_deserialization(self):
        """Test requirement deserialization."""
        data = {
            'resource_type': 'compute_power',
            'minimum': 2,
            'preferred': 4,
            'maximum': 8,
            'unit': 'cores',
            'constraints': {'cpu_type': 'x86_64'}
        }
        
        req = ResourceRequirement.from_dict(data)
        
        assert req.resource_type == ResourceType.COMPUTE_POWER
        assert req.minimum == 2
        assert req.preferred == 4
        assert req.constraints['cpu_type'] == 'x86_64'


@pytest.mark.skipif(not HAS_FRAMEWORK, reason="quantum_application_framework module not available")
class TestApplicationConfig:
    """Test ApplicationConfig functionality."""
    
    def test_config_creation(self):
        """Test creating application config."""
        config = ApplicationConfig(
            name="Test Application",
            version="1.0.0",
            description="Test quantum application",
            author="Test Author",
            application_type=ApplicationType.ALGORITHM,
            execution_mode=ExecutionMode.LOCAL
        )
        
        assert config.name == "Test Application"
        assert config.application_type == ApplicationType.ALGORITHM
        assert config.execution_mode == ExecutionMode.LOCAL
        assert isinstance(config.created_at, float)
    
    def test_config_with_requirements(self):
        """Test config with resource requirements."""
        requirements = [
            ResourceRequirement(ResourceType.QUBITS, 2, 4),
            ResourceRequirement(ResourceType.CLASSICAL_MEMORY, 1024, 2048, unit="MB")
        ]
        
        config = ApplicationConfig(
            name="Test App",
            version="1.0.0",
            description="Test",
            author="Author",
            application_type=ApplicationType.OPTIMIZATION,
            resource_requirements=requirements
        )
        
        assert len(config.resource_requirements) == 2
        assert config.resource_requirements[0].resource_type == ResourceType.QUBITS
        assert config.resource_requirements[1].unit == "MB"
    
    def test_config_serialization(self):
        """Test config serialization."""
        config = ApplicationConfig(
            name="Serialization Test",
            version="2.0.0",
            description="Test serialization",
            author="Test Author",
            application_type=ApplicationType.ML_HYBRID,
            execution_mode=ExecutionMode.CLOUD,
            parameters={'param1': 'value1'},
            metadata={'key1': 'value1'}
        )
        
        config_dict = config.to_dict()
        
        assert config_dict['name'] == "Serialization Test"
        assert config_dict['application_type'] == 'ml_hybrid'
        assert config_dict['execution_mode'] == 'cloud'
        assert config_dict['parameters']['param1'] == 'value1'
    
    def test_config_deserialization(self):
        """Test config deserialization."""
        data = {
            'name': 'Deserialization Test',
            'version': '1.5.0',
            'description': 'Test deserialization',
            'author': 'Test Author',
            'application_type': 'chemistry',
            'execution_mode': 'distributed',
            'resource_requirements': [
                {
                    'resource_type': 'qubits',
                    'minimum': 4,
                    'preferred': 8,
                    'maximum': None,
                    'unit': '',
                    'constraints': {}
                }
            ],
            'dependencies': ['numpy', 'scipy'],
            'parameters': {'shots': 1000},
            'metadata': {'category': 'chemistry'},
            'created_at': time.time()
        }
        
        config = ApplicationConfig.from_dict(data)
        
        assert config.name == 'Deserialization Test'
        assert config.application_type == ApplicationType.CHEMISTRY
        assert config.execution_mode == ExecutionMode.DISTRIBUTED
        assert len(config.resource_requirements) == 1
        assert config.resource_requirements[0].minimum == 4


@pytest.mark.skipif(not HAS_FRAMEWORK, reason="quantum_application_framework module not available")
class TestExecutionContext:
    """Test ExecutionContext functionality."""
    
    def test_context_creation(self):
        """Test creating execution context."""
        context = ExecutionContext(
            session_id="test_session",
            application_id="test_app"
        )
        
        assert context.session_id == "test_session"
        assert context.application_id == "test_app"
        assert context.state == ApplicationState.CREATED
        assert context.progress == 0.0
        assert len(context.checkpoints) == 0
    
    def test_context_serialization(self):
        """Test context serialization."""
        context = ExecutionContext(
            session_id="serialize_test",
            application_id="serialize_app",
            state=ApplicationState.RUNNING,
            start_time=time.time(),
            progress=0.5,
            allocated_resources={ResourceType.QUBITS: 4}
        )
        
        context_dict = context.to_dict()
        
        assert context_dict['session_id'] == "serialize_test"
        assert context_dict['state'] == 'running'
        assert context_dict['progress'] == 0.5
        assert context_dict['allocated_resources']['qubits'] == 4


@pytest.mark.skipif(not HAS_FRAMEWORK, reason="quantum_application_framework module not available")
class TestAlgorithmApplication:
    """Test AlgorithmApplication functionality."""
    
    def setup_method(self):
        """Set up test application."""
        config = ApplicationConfig(
            name="Test Algorithm",
            version="1.0.0",
            description="Test algorithm application",
            author="Test Author",
            application_type=ApplicationType.ALGORITHM,
            parameters={
                'circuit_data': {
                    'gates': [
                        {'gate': 'h', 'qubits': [0]},
                        {'gate': 'cnot', 'qubits': [0, 1]}
                    ]
                },
                'shots': 1024
            }
        )
        
        self.app = AlgorithmApplication(config)
        self.context = ExecutionContext(
            session_id="test_session",
            application_id=self.app.application_id
        )
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test application initialization."""
        success = await self.app.initialize(self.context)
        
        assert success is True
        assert self.context.state == ApplicationState.READY
        assert self.app.circuit_data is not None
        assert len(self.app.circuit_data['gates']) == 2
    
    @pytest.mark.asyncio
    async def test_local_execution(self):
        """Test local execution."""
        await self.app.initialize(self.context)
        result = await self.app.execute(self.context)
        
        assert result is not None
        assert 'counts' in result
        assert 'execution_time' in result
        assert self.context.state == ApplicationState.COMPLETED
        assert self.context.progress == 1.0
    
    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test application cleanup."""
        await self.app.initialize(self.context)
        success = await self.app.cleanup(self.context)
        
        assert success is True
    
    @pytest.mark.asyncio
    async def test_pause_resume(self):
        """Test pause and resume functionality."""
        await self.app.initialize(self.context)
        
        # Test pause
        success = await self.app.pause(self.context)
        assert success is True
        assert self.context.state == ApplicationState.PAUSED
        
        # Test resume
        success = await self.app.resume(self.context)
        assert success is True
        assert self.context.state == ApplicationState.RUNNING
    
    @pytest.mark.asyncio
    async def test_cancel(self):
        """Test application cancellation."""
        await self.app.initialize(self.context)
        
        success = await self.app.cancel(self.context)
        assert success is True
        assert self.context.state == ApplicationState.CANCELLED
    
    def test_checkpoint_creation(self):
        """Test checkpoint creation."""
        checkpoint_data = {'stage': 'initialization', 'progress': 0.1}
        self.app.create_checkpoint(self.context, checkpoint_data)
        
        assert len(self.context.checkpoints) == 1
        checkpoint = self.context.checkpoints[0]
        assert checkpoint['data']['stage'] == 'initialization'
        assert 'timestamp' in checkpoint
    
    def test_lifecycle_hooks(self):
        """Test lifecycle hooks."""
        hook_called = {'value': False}
        
        def test_hook(context):
            hook_called['value'] = True
        
        self.app.add_hook('initialize', test_hook)
        
        # Verify hook was added
        assert len(self.app._on_initialize_hooks) == 1


@pytest.mark.skipif(not HAS_FRAMEWORK, reason="quantum_application_framework module not available")
class TestOptimizationApplication:
    """Test OptimizationApplication functionality."""
    
    def setup_method(self):
        """Set up test optimization application."""
        config = ApplicationConfig(
            name="Test Optimization",
            version="1.0.0",
            description="Test optimization application",
            author="Test Author",
            application_type=ApplicationType.OPTIMIZATION,
            parameters={
                'problem_data': {
                    'hamiltonian': [[1, 0], [0, -1]],
                    'num_qubits': 2
                },
                'optimizer': {
                    'method': 'vqe',
                    'max_iterations': 10,
                    'tolerance': 1e-3
                }
            }
        )
        
        self.app = OptimizationApplication(config)
        self.context = ExecutionContext(
            session_id="opt_session",
            application_id=self.app.application_id
        )
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test optimization initialization."""
        success = await self.app.initialize(self.context)
        
        assert success is True
        assert self.context.state == ApplicationState.READY
        assert self.app.problem_data is not None
        assert self.app.optimizer_config['method'] == 'vqe'
    
    @pytest.mark.asyncio
    async def test_optimization_execution(self):
        """Test optimization execution."""
        await self.app.initialize(self.context)
        result = await self.app.execute(self.context)
        
        assert result is not None
        assert 'best_result' in result
        assert 'total_iterations' in result
        assert self.context.state == ApplicationState.COMPLETED
        assert self.context.progress == 1.0
        assert result['total_iterations'] <= 10  # Should respect max_iterations
    
    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test optimization cleanup."""
        await self.app.initialize(self.context)
        await self.app.execute(self.context)
        
        success = await self.app.cleanup(self.context)
        assert success is True
        assert self.app.current_iteration == 0
        assert self.app.best_result is None


@pytest.mark.skipif(not HAS_FRAMEWORK, reason="quantum_application_framework module not available")
class TestQuantumWorkflow:
    """Test QuantumWorkflow functionality."""
    
    def setup_method(self):
        """Set up test workflow."""
        self.workflow = QuantumWorkflow("Test Workflow")
        
        # Create test applications
        config1 = ApplicationConfig(
            name="Step 1",
            version="1.0.0",
            description="First step",
            author="Test",
            application_type=ApplicationType.ALGORITHM,
            parameters={'circuit_data': {'gates': [{'gate': 'h', 'qubits': [0]}]}}
        )
        
        config2 = ApplicationConfig(
            name="Step 2",
            version="1.0.0",
            description="Second step",
            author="Test",
            application_type=ApplicationType.ALGORITHM,
            parameters={'circuit_data': {'gates': [{'gate': 'x', 'qubits': [0]}]}}
        )
        
        app1 = AlgorithmApplication(config1)
        app2 = AlgorithmApplication(config2)
        
        # Create workflow steps
        self.step1 = WorkflowStep("step1", app1, [])
        self.step2 = WorkflowStep("step2", app2, ["step1"])
    
    def test_workflow_creation(self):
        """Test workflow creation."""
        assert self.workflow.name == "Test Workflow"
        assert len(self.workflow.steps) == 0
        assert len(self.workflow.execution_order) == 0
    
    def test_add_steps(self):
        """Test adding steps to workflow."""
        self.workflow.add_step(self.step1)
        self.workflow.add_step(self.step2)
        
        assert len(self.workflow.steps) == 2
        assert "step1" in self.workflow.steps
        assert "step2" in self.workflow.steps
        
        # Check execution order (step1 should come before step2)
        assert self.workflow.execution_order == ["step1", "step2"]
    
    def test_dependency_resolution(self):
        """Test dependency resolution."""
        # Add steps in reverse order to test sorting
        self.workflow.add_step(self.step2)
        self.workflow.add_step(self.step1)
        
        # Should still resolve to correct order
        assert self.workflow.execution_order == ["step1", "step2"]
    
    def test_circular_dependency_detection(self):
        """Test circular dependency detection."""
        # Create circular dependency
        step3_config = ApplicationConfig(
            name="Step 3",
            version="1.0.0",
            description="Third step",
            author="Test",
            application_type=ApplicationType.ALGORITHM,
            parameters={'circuit_data': {'gates': []}}
        )
        
        app3 = AlgorithmApplication(step3_config)
        step3 = WorkflowStep("step3", app3, ["step2"])
        
        # Modify step1 to depend on step3 (creating cycle)
        self.step1.dependencies = ["step3"]
        
        self.workflow.add_step(self.step1)
        self.workflow.add_step(self.step2)
        
        with pytest.raises(ValueError, match="Circular dependency"):
            self.workflow.add_step(step3)
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self):
        """Test workflow execution."""
        self.workflow.add_step(self.step1)
        self.workflow.add_step(self.step2)
        
        initial_inputs = {'global_param': 'test_value'}
        
        results = await self.workflow.execute(initial_inputs)
        
        assert len(results) == 2
        assert "step1" in results
        assert "step2" in results
        assert self.step1.status == ApplicationState.COMPLETED
        assert self.step2.status == ApplicationState.COMPLETED


@pytest.mark.skipif(not HAS_FRAMEWORK, reason="quantum_application_framework module not available")
class TestResourceManager:
    """Test ResourceManager functionality."""
    
    def setup_method(self):
        """Set up test resource manager."""
        self.manager = ResourceManager()
    
    def test_resource_initialization(self):
        """Test resource manager initialization."""
        assert ResourceType.QUBITS in self.manager.available_resources
        assert ResourceType.CLASSICAL_MEMORY in self.manager.available_resources
        assert self.manager.available_resources[ResourceType.QUBITS] == 20
    
    def test_availability_check(self):
        """Test resource availability checking."""
        requirements = [
            ResourceRequirement(ResourceType.QUBITS, 2, 4),
            ResourceRequirement(ResourceType.CLASSICAL_MEMORY, 1024, 2048)
        ]
        
        # Should be available initially
        assert self.manager.check_availability(requirements) is True
        
        # Check excessive requirements
        excessive_requirements = [
            ResourceRequirement(ResourceType.QUBITS, 100, 200)
        ]
        
        assert self.manager.check_availability(excessive_requirements) is False
    
    def test_resource_allocation(self):
        """Test resource allocation."""
        requirements = [
            ResourceRequirement(ResourceType.QUBITS, 2, 4),
            ResourceRequirement(ResourceType.CLASSICAL_MEMORY, 1024, 2048)
        ]
        
        allocation = self.manager.allocate_resources("session1", requirements)
        
        assert ResourceType.QUBITS in allocation
        assert ResourceType.CLASSICAL_MEMORY in allocation
        assert allocation[ResourceType.QUBITS] == 4  # Preferred amount
        assert allocation[ResourceType.CLASSICAL_MEMORY] == 2048
        
        # Check that resources are tracked
        assert "session1" in self.manager.allocated_resources
    
    def test_resource_release(self):
        """Test resource release."""
        requirements = [
            ResourceRequirement(ResourceType.QUBITS, 2, 4)
        ]
        
        self.manager.allocate_resources("session1", requirements)
        assert "session1" in self.manager.allocated_resources
        
        self.manager.release_resources("session1")
        assert "session1" not in self.manager.allocated_resources
    
    def test_resource_usage_stats(self):
        """Test resource usage statistics."""
        requirements = [
            ResourceRequirement(ResourceType.QUBITS, 2, 4)
        ]
        
        self.manager.allocate_resources("session1", requirements)
        
        usage = self.manager.get_resource_usage()
        
        assert 'qubits' in usage
        qubit_usage = usage['qubits']
        assert qubit_usage['total'] == 20
        assert qubit_usage['allocated'] == 4
        assert qubit_usage['free'] == 16
        assert qubit_usage['utilization'] == 0.2  # 4/20
    
    def test_concurrent_allocation(self):
        """Test concurrent resource allocation."""
        import threading
        
        results = []
        errors = []
        
        def allocate_session(session_id):
            try:
                requirements = [ResourceRequirement(ResourceType.QUBITS, 2, 4)]
                allocation = self.manager.allocate_resources(session_id, requirements)
                results.append((session_id, allocation))
            except Exception as e:
                errors.append((session_id, e))
        
        # Create multiple threads trying to allocate resources
        threads = []
        for i in range(5):
            thread = threading.Thread(target=allocate_session, args=[f"session_{i}"])
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should have successful allocations (up to capacity)
        assert len(results) > 0
        assert len(results) <= 5  # Limited by available resources


@pytest.mark.skipif(not HAS_FRAMEWORK, reason="quantum_application_framework module not available")
class TestApplicationTemplate:
    """Test ApplicationTemplate functionality."""
    
    def test_template_creation(self):
        """Test creating application template."""
        template = ApplicationTemplate("test_template", ApplicationType.ALGORITHM)
        
        assert template.name == "test_template"
        assert template.template_type == ApplicationType.ALGORITHM
        assert template.template_id is not None
    
    def test_template_configuration(self):
        """Test template configuration."""
        template = ApplicationTemplate("config_test", ApplicationType.OPTIMIZATION)
        
        config_template = {
            'name': 'Template Application',
            'version': '1.0.0',
            'description': 'Application from template',
            'author': 'Template System'
        }
        
        template.set_config_template(config_template)
        
        assert template.config_template == config_template
    
    def test_application_creation_from_template(self):
        """Test creating application from template."""
        template = ApplicationTemplate("creation_test", ApplicationType.ALGORITHM)
        
        template.set_config_template({
            'name': 'Template App',
            'version': '1.0.0',
            'description': 'App from template',
            'author': 'Template'
        }).set_application_class(AlgorithmApplication)
        
        app = template.create_application(
            parameters={'circuit_data': {'gates': []}},
            author='Custom Author'
        )
        
        assert isinstance(app, AlgorithmApplication)
        assert app.config.name == 'Template App'
        assert app.config.author == 'Custom Author'  # Override
        assert app.config.application_type == ApplicationType.ALGORITHM


@pytest.mark.skipif(not HAS_FRAMEWORK, reason="quantum_application_framework module not available")
class TestQuantumApplicationRuntime:
    """Test QuantumApplicationRuntime functionality."""
    
    def setup_method(self):
        """Set up test runtime."""
        self.runtime = QuantumApplicationRuntime()
    
    def test_runtime_initialization(self):
        """Test runtime initialization."""
        assert self.runtime.runtime_id is not None
        assert len(self.runtime.active_sessions) == 0
        assert isinstance(self.runtime.resource_manager, ResourceManager)
        assert len(self.runtime.templates) > 0  # Should have default templates
    
    def test_default_templates(self):
        """Test default templates."""
        assert 'basic_algorithm' in self.runtime.templates
        assert 'basic_optimization' in self.runtime.templates
        
        algo_template = self.runtime.templates['basic_algorithm']
        assert algo_template.template_type == ApplicationType.ALGORITHM
    
    def test_template_registration(self):
        """Test registering custom template."""
        custom_template = ApplicationTemplate("custom_test", ApplicationType.SIMULATION)
        custom_template.set_config_template({
            'name': 'Custom Template',
            'version': '1.0.0',
            'description': 'Custom template',
            'author': 'Test'
        }).set_application_class(AlgorithmApplication)
        
        self.runtime.register_template(custom_template)
        
        assert 'custom_test' in self.runtime.templates
        assert self.runtime.templates['custom_test'] == custom_template
    
    def test_application_creation(self):
        """Test creating application from runtime."""
        app = self.runtime.create_application(
            'basic_algorithm',
            name='Runtime Test App',
            parameters={'circuit_data': {'gates': []}}
        )
        
        assert isinstance(app, AlgorithmApplication)
        assert app.config.name == 'Runtime Test App'
    
    @pytest.mark.asyncio
    async def test_application_execution(self):
        """Test running application in runtime."""
        app = self.runtime.create_application(
            'basic_algorithm',
            parameters={'circuit_data': {'gates': [{'gate': 'h', 'qubits': [0]}]}}
        )
        
        context, result = await self.runtime.run_application(app)
        
        assert context.state == ApplicationState.COMPLETED
        assert result is not None
        assert 'counts' in result
        assert context.session_id not in self.runtime.active_sessions  # Should be cleaned up
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self):
        """Test running workflow in runtime."""
        workflow = QuantumWorkflow("Runtime Test Workflow")
        
        # Create simple workflow step
        app = self.runtime.create_application(
            'basic_algorithm',
            parameters={'circuit_data': {'gates': []}}
        )
        
        step = WorkflowStep("test_step", app, [])
        workflow.add_step(step)
        
        result = await self.runtime.run_workflow(workflow)
        
        assert 'test_step' in result
    
    def test_session_management(self):
        """Test session management."""
        # Initially no active sessions
        sessions = self.runtime.get_active_sessions()
        assert len(sessions) == 0
        
        # Test getting non-existent session
        session = self.runtime.get_session("non_existent")
        assert session is None
    
    @pytest.mark.asyncio
    async def test_session_control(self):
        """Test session pause/resume/cancel."""
        # Test with non-existent session
        assert await self.runtime.pause_session("non_existent") is False
        assert await self.runtime.resume_session("non_existent") is False
        assert await self.runtime.cancel_session("non_existent") is False
    
    def test_runtime_statistics(self):
        """Test runtime statistics."""
        stats = self.runtime.get_runtime_statistics()
        
        assert 'runtime_id' in stats
        assert 'active_sessions' in stats
        assert 'session_states' in stats
        assert 'resource_usage' in stats
        assert 'available_templates' in stats
        assert 'total_templates' in stats
        
        assert stats['active_sessions'] == 0
        assert stats['total_templates'] >= 2  # At least default templates


@pytest.mark.skipif(not HAS_FRAMEWORK, reason="quantum_application_framework module not available")
class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_get_quantum_runtime(self):
        """Test getting global runtime."""
        runtime1 = get_quantum_runtime()
        runtime2 = get_quantum_runtime()
        
        # Should be singleton
        assert runtime1 is runtime2
        assert isinstance(runtime1, QuantumApplicationRuntime)
    
    def test_create_algorithm_application(self):
        """Test creating algorithm application."""
        circuit_data = {'gates': [{'gate': 'h', 'qubits': [0]}]}
        
        app = create_algorithm_application(
            name="Convenience Test",
            algorithm_data=circuit_data,
            execution_mode='local',
            author='Test Author'
        )
        
        assert isinstance(app, AlgorithmApplication)
        assert app.config.name == "Convenience Test"
        assert app.config.execution_mode == ExecutionMode.LOCAL
        assert app.config.author == 'Test Author'
    
    def test_create_optimization_application(self):
        """Test creating optimization application."""
        problem_data = {'hamiltonian': [[1, 0], [0, -1]]}
        
        app = create_optimization_application(
            name="Optimization Test",
            problem_data=problem_data,
            version='2.0.0'
        )
        
        assert isinstance(app, OptimizationApplication)
        assert app.config.name == "Optimization Test"
        assert app.config.version == '2.0.0'
        assert app.config.parameters['problem_data'] == problem_data
    
    @pytest.mark.asyncio
    async def test_run_quantum_algorithm(self):
        """Test running quantum algorithm convenience function."""
        algorithm_data = {'gates': [{'gate': 'x', 'qubits': [0]}]}
        
        result = await run_quantum_algorithm(
            algorithm_data,
            execution_mode='local',
            name='Quick Test'
        )
        
        assert result is not None
        assert 'counts' in result
    
    def test_create_workflow(self):
        """Test creating workflow."""
        workflow = create_workflow("Convenience Workflow")
        
        assert isinstance(workflow, QuantumWorkflow)
        assert workflow.name == "Convenience Workflow"


@pytest.mark.skipif(not HAS_FRAMEWORK, reason="quantum_application_framework module not available")
class TestIntegrationScenarios:
    """Test integration scenarios."""
    
    def setup_method(self):
        """Set up integration test."""
        self.runtime = QuantumApplicationRuntime()
    
    @pytest.mark.asyncio
    async def test_complete_application_lifecycle(self):
        """Test complete application lifecycle."""
        # 1. Create application from template
        app = self.runtime.create_application(
            'basic_algorithm',
            name='Lifecycle Test',
            author='Integration Test',
            parameters={
                'circuit_data': {
                    'gates': [
                        {'gate': 'h', 'qubits': [0]},
                        {'gate': 'cnot', 'qubits': [0, 1]},
                        {'gate': 'measure', 'qubits': [0, 1]}
                    ]
                },
                'shots': 2048
            }
        )
        
        # 2. Run application
        context, result = await self.runtime.run_application(app)
        
        # 3. Verify execution
        assert context.state == ApplicationState.COMPLETED
        assert result is not None
        assert 'counts' in result
        assert 'execution_time' in result
        
        # 4. Check resource cleanup
        stats = self.runtime.get_runtime_statistics()
        assert stats['active_sessions'] == 0
    
    @pytest.mark.asyncio
    async def test_workflow_with_dependencies(self):
        """Test workflow with complex dependencies."""
        workflow = QuantumWorkflow("Complex Integration Workflow")
        
        # Create multiple applications
        apps = []
        for i in range(3):
            app = self.runtime.create_application(
                'basic_algorithm',
                name=f'Step {i+1}',
                parameters={'circuit_data': {'gates': [{'gate': 'h', 'qubits': [0]}]}}
            )
            apps.append(app)
        
        # Create workflow steps with dependencies
        step1 = WorkflowStep("preparation", apps[0], [])
        step2 = WorkflowStep("processing", apps[1], ["preparation"])
        step3 = WorkflowStep("analysis", apps[2], ["processing"])
        
        workflow.add_step(step1)
        workflow.add_step(step2)
        workflow.add_step(step3)
        
        # Execute workflow
        results = await self.runtime.run_workflow(workflow)
        
        # Verify all steps completed
        assert len(results) == 3
        assert "preparation" in results
        assert "processing" in results
        assert "analysis" in results
        
        # Verify execution order was respected
        assert workflow.execution_order == ["preparation", "processing", "analysis"]
    
    @pytest.mark.asyncio
    async def test_resource_constrained_execution(self):
        """Test execution with resource constraints."""
        # Create application with specific resource requirements
        requirements = [
            ResourceRequirement(ResourceType.QUBITS, 4, 8),
            ResourceRequirement(ResourceType.CLASSICAL_MEMORY, 2048, 4096, unit="MB")
        ]
        
        config = ApplicationConfig(
            name="Resource Test",
            version="1.0.0",
            description="Test with resource constraints",
            author="Test",
            application_type=ApplicationType.ALGORITHM,
            resource_requirements=requirements,
            parameters={'circuit_data': {'gates': [{'gate': 'h', 'qubits': [0]}]}}
        )
        
        app = AlgorithmApplication(config)
        
        # Run application
        context, result = await self.runtime.run_application(app)
        
        # Verify resource allocation
        assert ResourceType.QUBITS in context.allocated_resources
        assert ResourceType.CLASSICAL_MEMORY in context.allocated_resources
        assert context.allocated_resources[ResourceType.QUBITS] == 8  # Preferred amount
        
        # Verify execution completed
        assert context.state == ApplicationState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery."""
        # Create application that will fail during initialization
        class FailingApplication(AlgorithmApplication):
            async def initialize(self, context):
                raise ValueError("Intentional test failure")
        
        config = ApplicationConfig(
            name="Failing Test",
            version="1.0.0",
            description="Test error handling",
            author="Test",
            application_type=ApplicationType.ALGORITHM
        )
        
        app = FailingApplication(config)
        
        # Attempt to run application
        with pytest.raises(ValueError, match="Intentional test failure"):
            await self.runtime.run_application(app)
        
        # Verify runtime state is clean after failure
        stats = self.runtime.get_runtime_statistics()
        assert stats['active_sessions'] == 0
    
    @pytest.mark.asyncio
    async def test_concurrent_application_execution(self):
        """Test concurrent application execution."""
        import asyncio
        
        # Create multiple applications
        apps = []
        for i in range(5):
            app = self.runtime.create_application(
                'basic_algorithm',
                name=f'Concurrent App {i}',
                parameters={'circuit_data': {'gates': [{'gate': 'x', 'qubits': [0]}]}}
            )
            apps.append(app)
        
        # Run applications concurrently
        tasks = [self.runtime.run_application(app) for app in apps]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all succeeded
        successful_runs = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_runs) == 5
        
        # Verify each result
        for context, result in successful_runs:
            assert context.state == ApplicationState.COMPLETED
            assert result is not None
    
    def test_template_system_extensibility(self):
        """Test template system extensibility."""
        # Create custom application class
        class CustomApplication(QuantumApplication):
            async def initialize(self, context):
                context.state = ApplicationState.READY
                return True
            
            async def execute(self, context):
                context.state = ApplicationState.RUNNING
                await asyncio.sleep(0.01)  # Simulate work
                context.state = ApplicationState.COMPLETED
                return {'custom_result': 'success'}
            
            async def cleanup(self, context):
                return True
        
        # Create custom template
        custom_template = ApplicationTemplate("custom_app", ApplicationType.CUSTOM)
        custom_template.set_config_template({
            'name': 'Custom Application',
            'version': '1.0.0',
            'description': 'Custom application type',
            'author': 'Custom Developer'
        }).set_application_class(CustomApplication)
        
        # Register template
        self.runtime.register_template(custom_template)
        
        # Create and verify application
        app = self.runtime.create_application('custom_app')
        assert isinstance(app, CustomApplication)
        assert app.config.application_type == ApplicationType.CUSTOM


@pytest.mark.skipif(not HAS_FRAMEWORK, reason="quantum_application_framework module not available")
class TestPerformanceCharacteristics:
    """Test performance characteristics of the framework."""
    
    def setup_method(self):
        """Set up performance test."""
        self.runtime = QuantumApplicationRuntime()
    
    @pytest.mark.asyncio
    async def test_application_creation_performance(self):
        """Test application creation performance."""
        start_time = time.time()
        
        # Create many applications
        apps = []
        for i in range(100):
            app = self.runtime.create_application(
                'basic_algorithm',
                name=f'Perf Test {i}',
                parameters={'circuit_data': {'gates': []}}
            )
            apps.append(app)
        
        creation_time = time.time() - start_time
        
        # Should create applications quickly
        assert creation_time < 1.0  # 100 apps in under 1 second
        assert len(apps) == 100
    
    @pytest.mark.asyncio
    async def test_workflow_execution_performance(self):
        """Test workflow execution performance."""
        # Create workflow with many steps
        workflow = QuantumWorkflow("Performance Test Workflow")
        
        for i in range(20):
            app = self.runtime.create_application(
                'basic_algorithm',
                parameters={'circuit_data': {'gates': []}}
            )
            
            dependencies = [f"step_{i-1}"] if i > 0 else []
            step = WorkflowStep(f"step_{i}", app, dependencies)
            workflow.add_step(step)
        
        # Execute workflow
        start_time = time.time()
        results = await self.runtime.run_workflow(workflow)
        execution_time = time.time() - start_time
        
        # Should complete reasonably quickly
        assert execution_time < 5.0  # 20 steps in under 5 seconds
        assert len(results) == 20
    
    def test_resource_manager_performance(self):
        """Test resource manager performance."""
        manager = ResourceManager()
        
        # Test many allocation/deallocation cycles
        start_time = time.time()
        
        for i in range(1000):
            requirements = [ResourceRequirement(ResourceType.QUBITS, 1, 2)]
            session_id = f"perf_session_{i}"
            
            try:
                manager.allocate_resources(session_id, requirements)
                manager.release_resources(session_id)
            except RuntimeError:
                # Expected when resources are exhausted
                pass
        
        management_time = time.time() - start_time
        
        # Should handle operations quickly
        assert management_time < 2.0  # 1000 ops in under 2 seconds
    
    def test_concurrent_session_management(self):
        """Test concurrent session management performance."""
        import threading
        import concurrent.futures
        
        def create_and_run_session(session_id):
            try:
                app = self.runtime.create_application(
                    'basic_algorithm',
                    parameters={'circuit_data': {'gates': []}}
                )
                
                # Create minimal execution context
                context = ExecutionContext(
                    session_id=session_id,
                    application_id=app.application_id
                )
                
                return session_id, True
            except Exception as e:
                return session_id, False
        
        # Test concurrent session creation
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(create_and_run_session, f"concurrent_{i}")
                for i in range(50)
            ]
            
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        concurrent_time = time.time() - start_time
        
        # Should handle concurrent operations efficiently
        assert concurrent_time < 3.0  # 50 concurrent ops in under 3 seconds
        
        successful_sessions = [r for r in results if r[1]]
        assert len(successful_sessions) >= 40  # Most should succeed


if __name__ == "__main__":
    pytest.main([__file__])