"""
Quantum Application Framework

This module provides a comprehensive framework for building and deploying quantum applications
with high-level abstractions, workflow management, and seamless integration with the QuantRS2 ecosystem.
"""

import asyncio
import json
import time
import uuid
import logging
import threading
import inspect
import hashlib
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union, Set, Type
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum
from collections import defaultdict
import tempfile
import concurrent.futures
from contextlib import asynccontextmanager, contextmanager

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from . import quantum_cloud
    from . import compilation_service
    from . import algorithm_marketplace
    from . import distributed_simulation
    from . import quantum_networking
    from . import algorithm_debugger
    from . import circuit_db
    from . import profiler
    QUANTRS_MODULES_AVAILABLE = True
except ImportError:
    QUANTRS_MODULES_AVAILABLE = False


class ApplicationState(Enum):
    """Application execution state."""
    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ApplicationType(Enum):
    """Types of quantum applications."""
    ALGORITHM = "algorithm"
    OPTIMIZATION = "optimization"
    ML_HYBRID = "ml_hybrid"
    SIMULATION = "simulation"
    CRYPTOGRAPHY = "cryptography"
    FINANCE = "finance"
    CHEMISTRY = "chemistry"
    RESEARCH = "research"
    EDUCATIONAL = "educational"
    CUSTOM = "custom"


class ExecutionMode(Enum):
    """Application execution modes."""
    LOCAL = "local"
    DISTRIBUTED = "distributed"
    CLOUD = "cloud"
    HYBRID = "hybrid"


class ResourceType(Enum):
    """Types of quantum resources."""
    QUBITS = "qubits"
    CLASSICAL_MEMORY = "classical_memory"
    QUANTUM_MEMORY = "quantum_memory"
    COMPUTE_POWER = "compute_power"
    NETWORK_BANDWIDTH = "network_bandwidth"
    STORAGE = "storage"


@dataclass
class ResourceRequirement:
    """Resource requirement specification."""
    resource_type: ResourceType
    minimum: float
    preferred: float
    maximum: Optional[float] = None
    unit: str = ""
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResourceRequirement':
        """Create from dictionary."""
        data = data.copy()
        data['resource_type'] = ResourceType(data['resource_type'])
        return cls(**data)


@dataclass
class ApplicationConfig:
    """Application configuration."""
    name: str
    version: str
    description: str
    author: str
    application_type: ApplicationType
    execution_mode: ExecutionMode = ExecutionMode.LOCAL
    resource_requirements: List[ResourceRequirement] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'author': self.author,
            'application_type': self.application_type.value,
            'execution_mode': self.execution_mode.value,
            'resource_requirements': [req.to_dict() for req in self.resource_requirements],
            'dependencies': self.dependencies,
            'parameters': self.parameters,
            'metadata': self.metadata,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ApplicationConfig':
        """Create from dictionary."""
        data = data.copy()
        data['application_type'] = ApplicationType(data['application_type'])
        data['execution_mode'] = ExecutionMode(data['execution_mode'])
        data['resource_requirements'] = [
            ResourceRequirement.from_dict(req) for req in data.get('resource_requirements', [])
        ]
        return cls(**data)


@dataclass
class ExecutionContext:
    """Execution context for quantum applications."""
    session_id: str
    application_id: str
    state: ApplicationState = ApplicationState.CREATED
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    allocated_resources: Dict[ResourceType, float] = field(default_factory=dict)
    execution_metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    progress: float = 0.0
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'session_id': self.session_id,
            'application_id': self.application_id,
            'state': self.state.value,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'allocated_resources': {k.value: v for k, v in self.allocated_resources.items()},
            'execution_metadata': self.execution_metadata,
            'error_message': self.error_message,
            'progress': self.progress,
            'checkpoints': self.checkpoints
        }


class QuantumApplication(ABC):
    """Abstract base class for quantum applications."""
    
    def __init__(self, config: ApplicationConfig):
        self.config = config
        self.application_id = str(uuid.uuid4())
        self.context: Optional[ExecutionContext] = None
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Framework integration
        self._cloud_orchestrator = None
        self._compilation_service = None
        self._marketplace = None
        self._debugger = None
        self._profiler = None
        self._circuit_db = None
        
        # Lifecycle hooks
        self._setup_hooks()
    
    def _setup_hooks(self):
        """Set up lifecycle hooks."""
        self._on_initialize_hooks: List[Callable] = []
        self._on_start_hooks: List[Callable] = []
        self._on_pause_hooks: List[Callable] = []
        self._on_resume_hooks: List[Callable] = []
        self._on_complete_hooks: List[Callable] = []
        self._on_error_hooks: List[Callable] = []
    
    @abstractmethod
    async def initialize(self, context: ExecutionContext) -> bool:
        """Initialize the application."""
        pass
    
    @abstractmethod
    async def execute(self, context: ExecutionContext) -> Dict[str, Any]:
        """Execute the application."""
        pass
    
    @abstractmethod
    async def cleanup(self, context: ExecutionContext) -> bool:
        """Clean up resources."""
        pass
    
    async def pause(self, context: ExecutionContext) -> bool:
        """Pause application execution."""
        context.state = ApplicationState.PAUSED
        await self._call_hooks(self._on_pause_hooks, context)
        return True
    
    async def resume(self, context: ExecutionContext) -> bool:
        """Resume application execution."""
        context.state = ApplicationState.RUNNING
        await self._call_hooks(self._on_resume_hooks, context)
        return True
    
    async def cancel(self, context: ExecutionContext) -> bool:
        """Cancel application execution."""
        context.state = ApplicationState.CANCELLED
        await self.cleanup(context)
        return True
    
    def add_hook(self, event: str, hook: Callable):
        """Add lifecycle hook."""
        hook_list = getattr(self, f"_on_{event}_hooks", None)
        if hook_list is not None:
            hook_list.append(hook)
    
    async def _call_hooks(self, hooks: List[Callable], *args, **kwargs):
        """Call lifecycle hooks."""
        for hook in hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(*args, **kwargs)
                else:
                    hook(*args, **kwargs)
            except Exception as e:
                self._logger.error(f"Hook execution failed: {e}")
    
    def create_checkpoint(self, context: ExecutionContext, data: Dict[str, Any]):
        """Create execution checkpoint."""
        checkpoint = {
            'timestamp': time.time(),
            'progress': context.progress,
            'state': context.state.value,
            'data': data
        }
        context.checkpoints.append(checkpoint)
    
    def get_resource_usage(self, context: ExecutionContext) -> Dict[str, float]:
        """Get current resource usage."""
        return context.allocated_resources.copy()


class AlgorithmApplication(QuantumApplication):
    """Application for running quantum algorithms."""
    
    def __init__(self, config: ApplicationConfig, algorithm_entry=None):
        super().__init__(config)
        self.algorithm_entry = algorithm_entry
        self.circuit_data = None
        self.optimization_level = None
    
    async def initialize(self, context: ExecutionContext) -> bool:
        """Initialize algorithm application."""
        try:
            context.state = ApplicationState.INITIALIZING
            
            # Load algorithm if needed
            if self.algorithm_entry is None and 'algorithm_id' in self.config.parameters:
                if QUANTRS_MODULES_AVAILABLE and algorithm_marketplace:
                    marketplace = algorithm_marketplace.get_quantum_marketplace()
                    self.algorithm_entry = marketplace.get_algorithm(
                        self.config.parameters['algorithm_id']
                    )
            
            # Prepare circuit data
            if self.algorithm_entry:
                self.circuit_data = self.algorithm_entry.algorithm_data
            elif 'circuit_data' in self.config.parameters:
                self.circuit_data = self.config.parameters['circuit_data']
            else:
                raise ValueError("No algorithm or circuit data provided")
            
            # Set optimization level
            self.optimization_level = self.config.parameters.get('optimization_level', 'standard')
            
            context.state = ApplicationState.READY
            await self._call_hooks(self._on_initialize_hooks, context)
            return True
            
        except Exception as e:
            context.error_message = str(e)
            context.state = ApplicationState.FAILED
            await self._call_hooks(self._on_error_hooks, context, e)
            return False
    
    async def execute(self, context: ExecutionContext) -> Dict[str, Any]:
        """Execute algorithm."""
        try:
            context.state = ApplicationState.RUNNING
            context.start_time = time.time()
            await self._call_hooks(self._on_start_hooks, context)
            
            # Compile circuit if needed
            if QUANTRS_MODULES_AVAILABLE and compilation_service:
                compiler = compilation_service.get_compilation_service()
                compile_request = compilation_service.CompilationRequest(
                    circuit_data=self.circuit_data,
                    optimization_level=self.optimization_level,
                    target_backend=self.config.parameters.get('target_backend', 'generic')
                )
                
                compile_result = await compiler.compile_circuit_async(compile_request)
                if compile_result and compile_result.status == compilation_service.CompilationStatus.COMPLETED:
                    self.circuit_data = compile_result.optimized_circuit
                    context.execution_metadata['compilation_metrics'] = compile_result.metrics
            
            context.progress = 0.3
            self.create_checkpoint(context, {'stage': 'compilation_complete'})
            
            # Execute based on mode
            if self.config.execution_mode == ExecutionMode.CLOUD:
                result = await self._execute_cloud(context)
            elif self.config.execution_mode == ExecutionMode.DISTRIBUTED:
                result = await self._execute_distributed(context)
            else:
                result = await self._execute_local(context)
            
            context.progress = 1.0
            context.end_time = time.time()
            context.state = ApplicationState.COMPLETED
            await self._call_hooks(self._on_complete_hooks, context, result)
            
            return result
            
        except Exception as e:
            context.error_message = str(e)
            context.state = ApplicationState.FAILED
            context.end_time = time.time()
            await self._call_hooks(self._on_error_hooks, context, e)
            raise
    
    async def _execute_local(self, context: ExecutionContext) -> Dict[str, Any]:
        """Execute locally."""
        # Mock local execution
        await asyncio.sleep(0.1)  # Simulate processing
        return {
            'counts': {'00': 512, '11': 512},
            'execution_time': 0.1,
            'backend': 'local_simulator'
        }
    
    async def _execute_cloud(self, context: ExecutionContext) -> Dict[str, Any]:
        """Execute on cloud."""
        if not QUANTRS_MODULES_AVAILABLE or not quantum_cloud:
            raise RuntimeError("Cloud execution not available")
        
        orchestrator = quantum_cloud.get_quantum_cloud_orchestrator()
        
        # Prepare requirements
        requirements = {
            'min_qubits': self.config.parameters.get('min_qubits', 2),
            'device_type': quantum_cloud.DeviceType.SIMULATOR
        }
        
        for req in self.config.resource_requirements:
            if req.resource_type == ResourceType.QUBITS:
                requirements['min_qubits'] = int(req.minimum)
        
        # Submit job
        job = await orchestrator.submit_job_auto(
            self.circuit_data,
            requirements,
            self.config.parameters.get('shots', 1024)
        )
        
        if not job:
            raise RuntimeError("Failed to submit cloud job")
        
        context.execution_metadata['cloud_job_id'] = job.job_id
        context.progress = 0.7
        
        # Wait for completion (with polling)
        while True:
            status = await orchestrator.get_job_status(job.job_id)
            if status == quantum_cloud.JobStatus.COMPLETED:
                break
            elif status in [quantum_cloud.JobStatus.FAILED, quantum_cloud.JobStatus.CANCELLED]:
                raise RuntimeError(f"Cloud job failed with status: {status}")
            
            await asyncio.sleep(1.0)
        
        # Get result
        result = await orchestrator.get_job_result(job.job_id)
        if not result:
            raise RuntimeError("Failed to get cloud job result")
        
        result['backend'] = 'cloud'
        result['provider'] = job.provider.value
        return result
    
    async def _execute_distributed(self, context: ExecutionContext) -> Dict[str, Any]:
        """Execute in distributed mode."""
        if not QUANTRS_MODULES_AVAILABLE or not distributed_simulation:
            raise RuntimeError("Distributed execution not available")
        
        simulator = distributed_simulation.get_distributed_simulator()
        
        # Submit distributed job
        result = await simulator.simulate_circuit_async(
            circuit_data=self.circuit_data,
            shots=self.config.parameters.get('shots', 1024),
            strategy=distributed_simulation.DistributionStrategy.AMPLITUDE
        )
        
        result['backend'] = 'distributed'
        return result
    
    async def cleanup(self, context: ExecutionContext) -> bool:
        """Clean up algorithm resources."""
        # Clean up any allocated resources
        context.allocated_resources.clear()
        return True


class OptimizationApplication(QuantumApplication):
    """Application for quantum optimization problems."""
    
    def __init__(self, config: ApplicationConfig):
        super().__init__(config)
        self.problem_data = None
        self.optimizer_config = None
        self.current_iteration = 0
        self.best_result = None
    
    async def initialize(self, context: ExecutionContext) -> bool:
        """Initialize optimization application."""
        try:
            context.state = ApplicationState.INITIALIZING
            
            # Load problem data
            self.problem_data = self.config.parameters.get('problem_data')
            if not self.problem_data:
                raise ValueError("No problem data provided")
            
            # Set up optimizer
            self.optimizer_config = self.config.parameters.get('optimizer', {
                'method': 'vqe',
                'max_iterations': 100,
                'tolerance': 1e-6
            })
            
            context.state = ApplicationState.READY
            await self._call_hooks(self._on_initialize_hooks, context)
            return True
            
        except Exception as e:
            context.error_message = str(e)
            context.state = ApplicationState.FAILED
            return False
    
    async def execute(self, context: ExecutionContext) -> Dict[str, Any]:
        """Execute optimization."""
        try:
            context.state = ApplicationState.RUNNING
            context.start_time = time.time()
            
            max_iterations = self.optimizer_config.get('max_iterations', 100)
            tolerance = self.optimizer_config.get('tolerance', 1e-6)
            
            for iteration in range(max_iterations):
                self.current_iteration = iteration
                
                # Simulate optimization step
                await asyncio.sleep(0.01)  # Simulate computation
                
                # Mock optimization result
                current_energy = -1.0 + 0.1 * np.random.random() if NUMPY_AVAILABLE else -0.9
                
                if self.best_result is None or current_energy < self.best_result['energy']:
                    self.best_result = {
                        'energy': current_energy,
                        'iteration': iteration,
                        'parameters': [0.1 * i for i in range(5)]
                    }
                
                # Update progress
                context.progress = (iteration + 1) / max_iterations
                
                # Check convergence
                if iteration > 0 and abs(current_energy - self.best_result['energy']) < tolerance:
                    break
                
                # Create periodic checkpoints
                if iteration % 10 == 0:
                    self.create_checkpoint(context, {
                        'iteration': iteration,
                        'best_energy': self.best_result['energy']
                    })
            
            context.end_time = time.time()
            context.state = ApplicationState.COMPLETED
            
            result = {
                'best_result': self.best_result,
                'total_iterations': self.current_iteration + 1,
                'convergence_achieved': True,
                'execution_time': context.end_time - context.start_time
            }
            
            await self._call_hooks(self._on_complete_hooks, context, result)
            return result
            
        except Exception as e:
            context.error_message = str(e)
            context.state = ApplicationState.FAILED
            context.end_time = time.time()
            raise
    
    async def cleanup(self, context: ExecutionContext) -> bool:
        """Clean up optimization resources."""
        self.current_iteration = 0
        self.best_result = None
        return True


class WorkflowStep:
    """Individual step in a quantum workflow."""
    
    def __init__(self, name: str, application: QuantumApplication, 
                 dependencies: List[str] = None):
        self.name = name
        self.application = application
        self.dependencies = dependencies or []
        self.outputs: Dict[str, Any] = {}
        self.status = ApplicationState.CREATED
    
    async def execute(self, context: ExecutionContext, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow step."""
        # Update application parameters with inputs
        self.application.config.parameters.update(inputs)
        
        # Initialize and execute
        if await self.application.initialize(context):
            self.outputs = await self.application.execute(context)
            self.status = ApplicationState.COMPLETED
        else:
            self.status = ApplicationState.FAILED
            
        await self.application.cleanup(context)
        return self.outputs


class QuantumWorkflow:
    """Quantum application workflow manager."""
    
    def __init__(self, name: str):
        self.name = name
        self.workflow_id = str(uuid.uuid4())
        self.steps: Dict[str, WorkflowStep] = {}
        self.execution_order: List[str] = []
        self._logger = logging.getLogger(f"{__name__}.QuantumWorkflow")
    
    def add_step(self, step: WorkflowStep) -> 'QuantumWorkflow':
        """Add step to workflow."""
        self.steps[step.name] = step
        self._update_execution_order()
        return self
    
    def _update_execution_order(self):
        """Update execution order based on dependencies."""
        # Topological sort
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(step_name: str):
            if step_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {step_name}")
            if step_name not in visited:
                temp_visited.add(step_name)
                
                step = self.steps[step_name]
                for dep in step.dependencies:
                    if dep in self.steps:
                        visit(dep)
                
                temp_visited.remove(step_name)
                visited.add(step_name)
                order.append(step_name)
        
        for step_name in self.steps:
            if step_name not in visited:
                visit(step_name)
        
        self.execution_order = order
    
    async def execute(self, initial_inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute workflow."""
        initial_inputs = initial_inputs or {}
        step_outputs = {}
        
        for step_name in self.execution_order:
            step = self.steps[step_name]
            
            # Gather inputs from dependencies
            step_inputs = initial_inputs.copy()
            for dep_name in step.dependencies:
                if dep_name in step_outputs:
                    step_inputs.update(step_outputs[dep_name])
            
            # Create execution context for step
            context = ExecutionContext(
                session_id=str(uuid.uuid4()),
                application_id=step.application.application_id
            )
            
            # Execute step
            try:
                outputs = await step.execute(context, step_inputs)
                step_outputs[step_name] = outputs
                self._logger.info(f"Step {step_name} completed successfully")
                
            except Exception as e:
                self._logger.error(f"Step {step_name} failed: {e}")
                raise
        
        return step_outputs


class ResourceManager:
    """Manages quantum and classical resources."""
    
    def __init__(self):
        self.available_resources: Dict[ResourceType, float] = {
            ResourceType.QUBITS: 20,
            ResourceType.CLASSICAL_MEMORY: 8192,  # MB
            ResourceType.QUANTUM_MEMORY: 1024,   # MB
            ResourceType.COMPUTE_POWER: 8,       # cores
            ResourceType.NETWORK_BANDWIDTH: 1000,  # Mbps
            ResourceType.STORAGE: 10240          # MB
        }
        
        self.allocated_resources: Dict[str, Dict[ResourceType, float]] = {}
        self._lock = threading.Lock()
    
    def check_availability(self, requirements: List[ResourceRequirement]) -> bool:
        """Check if resources are available."""
        with self._lock:
            for req in requirements:
                available = self.available_resources.get(req.resource_type, 0)
                allocated = sum(
                    alloc.get(req.resource_type, 0) 
                    for alloc in self.allocated_resources.values()
                )
                
                if available - allocated < req.minimum:
                    return False
            
            return True
    
    def allocate_resources(self, session_id: str, 
                          requirements: List[ResourceRequirement]) -> Dict[ResourceType, float]:
        """Allocate resources for session."""
        with self._lock:
            if not self.check_availability(requirements):
                raise RuntimeError("Insufficient resources available")
            
            allocation = {}
            for req in requirements:
                available = self.available_resources.get(req.resource_type, 0)
                allocated = sum(
                    alloc.get(req.resource_type, 0) 
                    for alloc in self.allocated_resources.values()
                )
                
                # Allocate preferred amount if available, otherwise minimum
                free_amount = available - allocated
                allocation[req.resource_type] = min(req.preferred, free_amount)
            
            self.allocated_resources[session_id] = allocation
            return allocation
    
    def release_resources(self, session_id: str):
        """Release resources for session."""
        with self._lock:
            if session_id in self.allocated_resources:
                del self.allocated_resources[session_id]
    
    def get_resource_usage(self) -> Dict[str, Dict[str, float]]:
        """Get current resource usage."""
        with self._lock:
            usage = {}
            for resource_type in ResourceType:
                available = self.available_resources.get(resource_type, 0)
                allocated = sum(
                    alloc.get(resource_type, 0) 
                    for alloc in self.allocated_resources.values()
                )
                
                usage[resource_type.value] = {
                    'total': available,
                    'allocated': allocated,
                    'free': available - allocated,
                    'utilization': allocated / available if available > 0 else 0
                }
            
            return usage


class ApplicationTemplate:
    """Template for creating quantum applications."""
    
    def __init__(self, name: str, template_type: ApplicationType):
        self.name = name
        self.template_type = template_type
        self.template_id = str(uuid.uuid4())
        self.config_template: Dict[str, Any] = {}
        self.application_class: Optional[Type[QuantumApplication]] = None
        
    def set_config_template(self, template: Dict[str, Any]) -> 'ApplicationTemplate':
        """Set configuration template."""
        self.config_template = template
        return self
    
    def set_application_class(self, app_class: Type[QuantumApplication]) -> 'ApplicationTemplate':
        """Set application class."""
        self.application_class = app_class
        return self
    
    def create_application(self, **kwargs) -> QuantumApplication:
        """Create application from template."""
        if not self.application_class:
            raise ValueError("No application class defined for template")
        
        # Merge template with provided parameters
        config_data = self.config_template.copy()
        config_data.update(kwargs)
        
        # Ensure required fields
        config_data['application_type'] = self.template_type
        
        config = ApplicationConfig.from_dict(config_data)
        return self.application_class(config)


class QuantumApplicationRuntime:
    """Runtime environment for quantum applications."""
    
    def __init__(self):
        self.runtime_id = str(uuid.uuid4())
        self.active_sessions: Dict[str, ExecutionContext] = {}
        self.resource_manager = ResourceManager()
        self.templates: Dict[str, ApplicationTemplate] = {}
        self._setup_logging()
        self._setup_default_templates()
        
        # Integration components
        self._cloud_orchestrator = None
        self._marketplace = None
        self._debugger = None
        
        self._lock = threading.Lock()
    
    def _setup_logging(self):
        """Set up logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_default_templates(self):
        """Set up default application templates."""
        # Algorithm template
        algo_template = ApplicationTemplate("basic_algorithm", ApplicationType.ALGORITHM)
        algo_template.set_config_template({
            'name': 'Basic Algorithm Application',
            'version': '1.0.0',
            'description': 'Basic quantum algorithm application',
            'author': 'QuantRS2 Framework',
            'execution_mode': ExecutionMode.LOCAL.value,
            'parameters': {
                'shots': 1024,
                'optimization_level': 'standard'
            }
        }).set_application_class(AlgorithmApplication)
        
        self.templates['basic_algorithm'] = algo_template
        
        # Optimization template
        opt_template = ApplicationTemplate("basic_optimization", ApplicationType.OPTIMIZATION)
        opt_template.set_config_template({
            'name': 'Basic Optimization Application',
            'version': '1.0.0',
            'description': 'Basic quantum optimization application',
            'author': 'QuantRS2 Framework',
            'execution_mode': ExecutionMode.LOCAL.value,
            'parameters': {
                'optimizer': {
                    'method': 'vqe',
                    'max_iterations': 100,
                    'tolerance': 1e-6
                }
            }
        }).set_application_class(OptimizationApplication)
        
        self.templates['basic_optimization'] = opt_template
    
    def register_template(self, template: ApplicationTemplate):
        """Register application template."""
        self.templates[template.name] = template
        self.logger.info(f"Registered template: {template.name}")
    
    def create_application(self, template_name: str, **kwargs) -> QuantumApplication:
        """Create application from template."""
        if template_name not in self.templates:
            raise ValueError(f"Template not found: {template_name}")
        
        template = self.templates[template_name]
        return template.create_application(**kwargs)
    
    async def run_application(self, application: QuantumApplication, 
                            session_id: Optional[str] = None) -> Tuple[ExecutionContext, Dict[str, Any]]:
        """Run quantum application."""
        session_id = session_id or str(uuid.uuid4())
        
        # Create execution context
        context = ExecutionContext(
            session_id=session_id,
            application_id=application.application_id
        )
        
        with self._lock:
            self.active_sessions[session_id] = context
        
        try:
            # Check and allocate resources
            if application.config.resource_requirements:
                if not self.resource_manager.check_availability(application.config.resource_requirements):
                    raise RuntimeError("Insufficient resources to run application")
                
                context.allocated_resources = self.resource_manager.allocate_resources(
                    session_id, application.config.resource_requirements
                )
            
            # Initialize application
            if not await application.initialize(context):
                raise RuntimeError("Application initialization failed")
            
            # Execute application
            result = await application.execute(context)
            
            # Cleanup
            await application.cleanup(context)
            
            return context, result
            
        except Exception as e:
            self.logger.error(f"Application execution failed: {e}")
            context.error_message = str(e)
            context.state = ApplicationState.FAILED
            await application.cleanup(context)
            raise
            
        finally:
            # Release resources
            if session_id in self.active_sessions:
                self.resource_manager.release_resources(session_id)
                with self._lock:
                    del self.active_sessions[session_id]
    
    async def run_workflow(self, workflow: QuantumWorkflow, 
                          inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run quantum workflow."""
        self.logger.info(f"Starting workflow: {workflow.name}")
        
        try:
            result = await workflow.execute(inputs)
            self.logger.info(f"Workflow {workflow.name} completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Workflow {workflow.name} failed: {e}")
            raise
    
    def get_active_sessions(self) -> List[ExecutionContext]:
        """Get list of active sessions."""
        with self._lock:
            return list(self.active_sessions.values())
    
    def get_session(self, session_id: str) -> Optional[ExecutionContext]:
        """Get session by ID."""
        with self._lock:
            return self.active_sessions.get(session_id)
    
    async def pause_session(self, session_id: str) -> bool:
        """Pause session execution."""
        context = self.get_session(session_id)
        if context and context.state == ApplicationState.RUNNING:
            context.state = ApplicationState.PAUSED
            return True
        return False
    
    async def resume_session(self, session_id: str) -> bool:
        """Resume session execution."""
        context = self.get_session(session_id)
        if context and context.state == ApplicationState.PAUSED:
            context.state = ApplicationState.RUNNING
            return True
        return False
    
    async def cancel_session(self, session_id: str) -> bool:
        """Cancel session execution."""
        context = self.get_session(session_id)
        if context:
            context.state = ApplicationState.CANCELLED
            self.resource_manager.release_resources(session_id)
            with self._lock:
                if session_id in self.active_sessions:
                    del self.active_sessions[session_id]
            return True
        return False
    
    def get_runtime_statistics(self) -> Dict[str, Any]:
        """Get runtime statistics."""
        with self._lock:
            active_count = len(self.active_sessions)
            state_counts = defaultdict(int)
            for context in self.active_sessions.values():
                state_counts[context.state.value] += 1
        
        resource_usage = self.resource_manager.get_resource_usage()
        
        return {
            'runtime_id': self.runtime_id,
            'active_sessions': active_count,
            'session_states': dict(state_counts),
            'resource_usage': resource_usage,
            'available_templates': list(self.templates.keys()),
            'total_templates': len(self.templates)
        }


# Global runtime instance
_quantum_runtime: Optional[QuantumApplicationRuntime] = None


def get_quantum_runtime() -> QuantumApplicationRuntime:
    """Get global quantum application runtime."""
    global _quantum_runtime
    if _quantum_runtime is None:
        _quantum_runtime = QuantumApplicationRuntime()
    return _quantum_runtime


def create_algorithm_application(name: str, algorithm_data: Dict[str, Any], 
                               **kwargs) -> AlgorithmApplication:
    """Convenience function to create algorithm application."""
    config = ApplicationConfig(
        name=name,
        version=kwargs.get('version', '1.0.0'),
        description=kwargs.get('description', f'Algorithm application: {name}'),
        author=kwargs.get('author', 'User'),
        application_type=ApplicationType.ALGORITHM,
        execution_mode=ExecutionMode(kwargs.get('execution_mode', 'local')),
        parameters={
            'circuit_data': algorithm_data,
            **kwargs.get('parameters', {})
        }
    )
    
    return AlgorithmApplication(config)


def create_optimization_application(name: str, problem_data: Dict[str, Any], 
                                  **kwargs) -> OptimizationApplication:
    """Convenience function to create optimization application."""
    config = ApplicationConfig(
        name=name,
        version=kwargs.get('version', '1.0.0'),
        description=kwargs.get('description', f'Optimization application: {name}'),
        author=kwargs.get('author', 'User'),
        application_type=ApplicationType.OPTIMIZATION,
        execution_mode=ExecutionMode(kwargs.get('execution_mode', 'local')),
        parameters={
            'problem_data': problem_data,
            **kwargs.get('parameters', {})
        }
    )
    
    return OptimizationApplication(config)


async def run_quantum_algorithm(algorithm_data: Dict[str, Any], 
                               execution_mode: str = 'local',
                               **kwargs) -> Dict[str, Any]:
    """Convenience function to run quantum algorithm."""
    runtime = get_quantum_runtime()
    
    app = create_algorithm_application(
        name=kwargs.get('name', 'Quick Algorithm'),
        algorithm_data=algorithm_data,
        execution_mode=execution_mode,
        **kwargs
    )
    
    context, result = await runtime.run_application(app)
    return result


def create_workflow(name: str) -> QuantumWorkflow:
    """Convenience function to create workflow."""
    return QuantumWorkflow(name)


# CLI interface
def main():
    """Main CLI interface for quantum application framework."""
    import argparse
    
    parser = argparse.ArgumentParser(description="QuantRS2 Quantum Application Framework")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Runtime commands
    runtime_parser = subparsers.add_parser('runtime', help='Runtime management')
    runtime_subparsers = runtime_parser.add_subparsers(dest='runtime_command')
    
    runtime_subparsers.add_parser('status', help='Show runtime status')
    runtime_subparsers.add_parser('sessions', help='List active sessions')
    runtime_subparsers.add_parser('resources', help='Show resource usage')
    
    # Template commands
    template_parser = subparsers.add_parser('templates', help='Template management')
    template_subparsers = template_parser.add_subparsers(dest='template_command')
    
    template_subparsers.add_parser('list', help='List available templates')
    
    # Application commands
    app_parser = subparsers.add_parser('run', help='Run application')
    app_parser.add_argument('template', help='Application template name')
    app_parser.add_argument('--config', help='Configuration file path')
    app_parser.add_argument('--param', action='append', help='Parameter (key=value)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    runtime = get_quantum_runtime()
    
    if args.command == 'runtime':
        if args.runtime_command == 'status':
            stats = runtime.get_runtime_statistics()
            print("Runtime Status:")
            print(json.dumps(stats, indent=2))
        
        elif args.runtime_command == 'sessions':
            sessions = runtime.get_active_sessions()
            print(f"Active Sessions ({len(sessions)}):")
            for session in sessions:
                print(f"  {session.session_id} - {session.state.value}")
        
        elif args.runtime_command == 'resources':
            usage = runtime.resource_manager.get_resource_usage()
            print("Resource Usage:")
            for resource, info in usage.items():
                print(f"  {resource}: {info['allocated']}/{info['total']} ({info['utilization']:.1%})")
    
    elif args.command == 'templates':
        if args.template_command == 'list':
            print("Available Templates:")
            for name, template in runtime.templates.items():
                print(f"  {name} ({template.template_type.value})")
    
    elif args.command == 'run':
        async def run_app():
            try:
                # Parse parameters
                params = {}
                if args.param:
                    for param in args.param:
                        key, value = param.split('=', 1)
                        params[key] = value
                
                # Load config if provided
                if args.config:
                    with open(args.config, 'r') as f:
                        config_data = json.load(f) if args.config.endswith('.json') else yaml.safe_load(f)
                    params.update(config_data)
                
                # Create and run application
                app = runtime.create_application(args.template, **params)
                context, result = await runtime.run_application(app)
                
                print("Application Result:")
                print(json.dumps(result, indent=2))
                
                return 0
                
            except Exception as e:
                print(f"Error: {e}")
                return 1
        
        return asyncio.run(run_app())
    
    return 0


if __name__ == "__main__":
    exit(main())