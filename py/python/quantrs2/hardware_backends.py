#!/usr/bin/env python3
"""
Hardware Backend Integration for QuantRS2

This module provides integration with various quantum hardware backends,
including IBM Quantum, Google Quantum AI, AWS Braket, Rigetti, and IonQ.
"""

import asyncio
import json
import time
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
import warnings
from concurrent.futures import ThreadPoolExecutor, Future
import threading
import queue
import logging


class BackendType(Enum):
    """Types of quantum hardware backends."""
    IBM_QUANTUM = "ibm_quantum"
    GOOGLE_QUANTUM_AI = "google_quantum_ai"
    AWS_BRAKET = "aws_braket"
    RIGETTI = "rigetti"
    IONQ = "ionq"
    QUANTINUUM = "quantinuum"
    AZURE_QUANTUM = "azure_quantum"
    SIMULATOR = "simulator"
    LOCAL = "local"


class JobStatus(Enum):
    """Status of quantum jobs."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INITIALIZING = "initializing"


class DeviceStatus(Enum):
    """Status of quantum devices."""
    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    RETIRED = "retired"
    UNKNOWN = "unknown"


@dataclass
class DeviceCapabilities:
    """Capabilities and specifications of a quantum device."""
    max_qubits: int
    gate_set: List[str]
    connectivity: Optional[List[Tuple[int, int]]] = None
    gate_fidelities: Optional[Dict[str, float]] = None
    coherence_times: Optional[Dict[str, float]] = None  # T1, T2 times
    error_rates: Optional[Dict[str, float]] = None
    max_shots: int = 8192
    supports_mid_circuit_measurement: bool = False
    supports_reset: bool = False
    supports_conditional_gates: bool = False
    native_gates: Optional[List[str]] = None
    basis_gates: Optional[List[str]] = None


@dataclass
class DeviceInfo:
    """Information about a quantum device."""
    name: str
    backend_type: BackendType
    status: DeviceStatus
    capabilities: DeviceCapabilities
    queue_length: int = 0
    estimated_wait_time: Optional[float] = None  # in seconds
    cost_per_shot: Optional[float] = None
    location: Optional[str] = None
    provider: Optional[str] = None
    last_calibration: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JobRequest:
    """Request for submitting a quantum job."""
    circuit_data: Dict[str, Any]  # Serialized circuit
    device_name: str
    shots: int = 1024
    optimization_level: int = 1
    memory: bool = False
    max_experiments: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher numbers = higher priority


@dataclass
class JobResult:
    """Result from a quantum job execution."""
    job_id: str
    status: JobStatus
    device_name: str
    shots: int
    counts: Optional[Dict[str, int]] = None
    raw_counts: Optional[Dict[str, int]] = None
    memory: Optional[List[str]] = None
    execution_time: Optional[float] = None
    queue_time: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    backend_result: Optional[Any] = None  # Raw backend-specific result


class QuantumBackend(ABC):
    """Abstract base class for quantum hardware backends."""
    
    def __init__(self, 
                 name: str,
                 backend_type: BackendType,
                 credentials: Optional[Dict[str, Any]] = None):
        """
        Initialize quantum backend.
        
        Args:
            name: Backend name
            backend_type: Type of backend
            credentials: Authentication credentials
        """
        self.name = name
        self.backend_type = backend_type
        self.credentials = credentials
        self._authenticated = False
        self._devices_cache: Dict[str, DeviceInfo] = {}
        self._cache_timestamp = 0
        self._cache_ttl = 300  # 5 minutes
        self.logger = logging.getLogger(f"quantrs2.backend.{name}")
    
    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with the backend."""
        pass
    
    @abstractmethod
    async def get_devices(self, refresh: bool = False) -> List[DeviceInfo]:
        """Get available quantum devices."""
        pass
    
    @abstractmethod
    async def get_device_info(self, device_name: str) -> DeviceInfo:
        """Get detailed information about a specific device."""
        pass
    
    @abstractmethod
    async def submit_job(self, job_request: JobRequest) -> str:
        """Submit a quantum job and return job ID."""
        pass
    
    @abstractmethod
    async def get_job_status(self, job_id: str) -> JobStatus:
        """Get the status of a submitted job."""
        pass
    
    @abstractmethod
    async def get_job_result(self, job_id: str) -> JobResult:
        """Get the result of a completed job."""
        pass
    
    @abstractmethod
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a submitted job."""
        pass
    
    def _is_cache_valid(self) -> bool:
        """Check if the device cache is still valid."""
        return (time.time() - self._cache_timestamp) < self._cache_ttl
    
    def _update_cache(self, devices: List[DeviceInfo]):
        """Update the device cache."""
        self._devices_cache = {device.name: device for device in devices}
        self._cache_timestamp = time.time()


class IBMQuantumBackend(QuantumBackend):
    """IBM Quantum backend implementation."""
    
    def __init__(self, credentials: Optional[Dict[str, Any]] = None):
        super().__init__("IBM Quantum", BackendType.IBM_QUANTUM, credentials)
        self._service = None
        self._provider = None
    
    async def authenticate(self) -> bool:
        """Authenticate with IBM Quantum."""
        try:
            # Try to import qiskit-ibm-runtime or qiskit-ibm-provider
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService
                token = self.credentials.get("token") if self.credentials else None
                if token:
                    self._service = QiskitRuntimeService(token=token)
                else:
                    self._service = QiskitRuntimeService()  # Use saved credentials
                self._authenticated = True
                self.logger.info("Authenticated with IBM Quantum using Runtime Service")
                return True
            except ImportError:
                try:
                    from qiskit import IBMQ
                    token = self.credentials.get("token") if self.credentials else None
                    if token:
                        IBMQ.save_account(token, overwrite=True)
                    IBMQ.load_account()
                    self._provider = IBMQ.get_provider()
                    self._authenticated = True
                    self.logger.info("Authenticated with IBM Quantum using legacy provider")
                    return True
                except ImportError:
                    self.logger.error("IBM Quantum SDK not available")
                    return False
        except Exception as e:
            self.logger.error(f"IBM Quantum authentication failed: {e}")
            return False
    
    async def get_devices(self, refresh: bool = False) -> List[DeviceInfo]:
        """Get available IBM Quantum devices."""
        if not refresh and self._is_cache_valid():
            return list(self._devices_cache.values())
        
        if not self._authenticated:
            await self.authenticate()
        
        devices = []
        try:
            if self._service:
                backends = self._service.backends()
            elif self._provider:
                backends = self._provider.backends()
            else:
                return devices
            
            for backend in backends:
                device_info = await self._create_device_info_from_backend(backend)
                devices.append(device_info)
            
            self._update_cache(devices)
            
        except Exception as e:
            self.logger.error(f"Failed to get IBM Quantum devices: {e}")
        
        return devices
    
    async def _create_device_info_from_backend(self, backend) -> DeviceInfo:
        """Create DeviceInfo from IBM backend."""
        config = backend.configuration()
        status = backend.status()
        
        # Map IBM status to our DeviceStatus
        if status.operational and status.status_msg == "active":
            device_status = DeviceStatus.ONLINE
        elif "maintenance" in status.status_msg.lower():
            device_status = DeviceStatus.MAINTENANCE
        else:
            device_status = DeviceStatus.OFFLINE
        
        # Create capabilities
        capabilities = DeviceCapabilities(
            max_qubits=config.n_qubits,
            gate_set=config.gate_set if hasattr(config, 'gate_set') else config.basis_gates,
            connectivity=getattr(config, 'coupling_map', None),
            max_shots=getattr(config, 'max_shots', 8192),
            supports_mid_circuit_measurement=getattr(config, 'mid_circuit_measurement', False),
            supports_reset=getattr(config, 'qubit_reset', False),
            basis_gates=config.basis_gates
        )
        
        # Add error rates if available
        if hasattr(backend, 'properties') and backend.properties():
            props = backend.properties()
            capabilities.gate_fidelities = self._extract_gate_fidelities(props)
            capabilities.coherence_times = self._extract_coherence_times(props)
            capabilities.error_rates = self._extract_error_rates(props)
        
        return DeviceInfo(
            name=config.backend_name,
            backend_type=BackendType.IBM_QUANTUM,
            status=device_status,
            capabilities=capabilities,
            queue_length=status.pending_jobs,
            provider="IBM Quantum"
        )
    
    def _extract_gate_fidelities(self, properties) -> Dict[str, float]:
        """Extract gate fidelities from IBM backend properties."""
        fidelities = {}
        for gate in properties.gates:
            if hasattr(gate, 'parameters'):
                for param in gate.parameters:
                    if param.name == "gate_error":
                        fidelity = 1.0 - param.value
                        fidelities[gate.gate] = fidelity
        return fidelities
    
    def _extract_coherence_times(self, properties) -> Dict[str, float]:
        """Extract coherence times from IBM backend properties."""
        times = {}
        for qubit_idx, qubit_props in enumerate(properties.qubits):
            for prop in qubit_props:
                if prop.name == "T1":
                    times[f"T1_q{qubit_idx}"] = prop.value
                elif prop.name == "T2":
                    times[f"T2_q{qubit_idx}"] = prop.value
        return times
    
    def _extract_error_rates(self, properties) -> Dict[str, float]:
        """Extract error rates from IBM backend properties."""
        errors = {}
        
        # Readout errors
        for qubit_idx, qubit_props in enumerate(properties.qubits):
            for prop in qubit_props:
                if prop.name == "readout_error":
                    errors[f"readout_q{qubit_idx}"] = prop.value
        
        # Gate errors
        for gate in properties.gates:
            if hasattr(gate, 'parameters'):
                for param in gate.parameters:
                    if param.name == "gate_error":
                        errors[f"{gate.gate}_error"] = param.value
        
        return errors
    
    async def get_device_info(self, device_name: str) -> DeviceInfo:
        """Get detailed information about a specific IBM device."""
        devices = await self.get_devices()
        for device in devices:
            if device.name == device_name:
                return device
        raise ValueError(f"Device {device_name} not found")
    
    async def submit_job(self, job_request: JobRequest) -> str:
        """Submit a job to IBM Quantum."""
        if not self._authenticated:
            await self.authenticate()
        
        try:
            # Convert circuit to Qiskit format
            qiskit_circuit = self._convert_to_qiskit_circuit(job_request.circuit_data)
            
            # Get backend
            if self._service:
                backend = self._service.backend(job_request.device_name)
                # Use Sampler for shots-based execution
                job = backend.run(qiskit_circuit, shots=job_request.shots)
            elif self._provider:
                backend = self._provider.get_backend(job_request.device_name)
                job = backend.run(qiskit_circuit, shots=job_request.shots)
            else:
                raise RuntimeError("No IBM Quantum service available")
            
            return job.job_id()
            
        except Exception as e:
            self.logger.error(f"Failed to submit IBM job: {e}")
            raise
    
    def _convert_to_qiskit_circuit(self, circuit_data: Dict[str, Any]):
        """Convert QuantRS2 circuit data to Qiskit circuit."""
        try:
            from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
            
            n_qubits = circuit_data.get("n_qubits", 1)
            qreg = QuantumRegister(n_qubits)
            creg = ClassicalRegister(n_qubits)
            qc = QuantumCircuit(qreg, creg)
            
            # Add gates from circuit data
            operations = circuit_data.get("operations", [])
            for op in operations:
                gate_name = op.get("name", "")
                qubits = op.get("qubits", [])
                params = op.get("parameters", [])
                
                if gate_name == "h":
                    qc.h(qubits[0])
                elif gate_name == "x":
                    qc.x(qubits[0])
                elif gate_name == "y":
                    qc.y(qubits[0])
                elif gate_name == "z":
                    qc.z(qubits[0])
                elif gate_name == "cnot":
                    qc.cx(qubits[0], qubits[1])
                elif gate_name == "ry":
                    qc.ry(params[0], qubits[0])
                elif gate_name == "rz":
                    qc.rz(params[0], qubits[0])
                elif gate_name == "rx":
                    qc.rx(params[0], qubits[0])
                # Add more gate conversions as needed
            
            # Add measurements
            qc.measure_all()
            
            return qc
            
        except ImportError:
            raise ImportError("Qiskit not available for circuit conversion")
    
    async def get_job_status(self, job_id: str) -> JobStatus:
        """Get IBM Quantum job status."""
        try:
            if self._service:
                job = self._service.job(job_id)
            elif self._provider:
                job = self._provider.get_job(job_id)
            else:
                raise RuntimeError("No IBM Quantum service available")
            
            ibm_status = job.status()
            
            # Map IBM status to our JobStatus
            status_map = {
                "INITIALIZING": JobStatus.INITIALIZING,
                "QUEUED": JobStatus.QUEUED,
                "VALIDATING": JobStatus.QUEUED,
                "RUNNING": JobStatus.RUNNING,
                "CANCELLED": JobStatus.CANCELLED,
                "DONE": JobStatus.COMPLETED,
                "ERROR": JobStatus.FAILED
            }
            
            return status_map.get(ibm_status.name, JobStatus.UNKNOWN)
            
        except Exception as e:
            self.logger.error(f"Failed to get IBM job status: {e}")
            return JobStatus.FAILED
    
    async def get_job_result(self, job_id: str) -> JobResult:
        """Get IBM Quantum job result."""
        try:
            if self._service:
                job = self._service.job(job_id)
            elif self._provider:
                job = self._provider.get_job(job_id)
            else:
                raise RuntimeError("No IBM Quantum service available")
            
            if job.status().name != "DONE":
                return JobResult(
                    job_id=job_id,
                    status=await self.get_job_status(job_id),
                    device_name=job.backend().name,
                    shots=0
                )
            
            result = job.result()
            counts = result.get_counts()
            
            return JobResult(
                job_id=job_id,
                status=JobStatus.COMPLETED,
                device_name=job.backend().name,
                shots=sum(counts.values()),
                counts=counts,
                backend_result=result
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get IBM job result: {e}")
            return JobResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                device_name="unknown",
                shots=0,
                error_message=str(e)
            )
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel IBM Quantum job."""
        try:
            if self._service:
                job = self._service.job(job_id)
            elif self._provider:
                job = self._provider.get_job(job_id)
            else:
                return False
            
            job.cancel()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel IBM job: {e}")
            return False


class GoogleQuantumAIBackend(QuantumBackend):
    """Google Quantum AI backend implementation."""
    
    def __init__(self, credentials: Optional[Dict[str, Any]] = None):
        super().__init__("Google Quantum AI", BackendType.GOOGLE_QUANTUM_AI, credentials)
        self._service = None
    
    async def authenticate(self) -> bool:
        """Authenticate with Google Quantum AI."""
        try:
            import cirq_google
            
            project_id = self.credentials.get("project_id") if self.credentials else None
            if not project_id:
                self.logger.error("Google Cloud project ID required")
                return False
            
            # Use default credentials or service account
            self._service = cirq_google.Engine(project_id=project_id)
            self._authenticated = True
            self.logger.info("Authenticated with Google Quantum AI")
            return True
            
        except ImportError:
            self.logger.error("Cirq Google not available")
            return False
        except Exception as e:
            self.logger.error(f"Google Quantum AI authentication failed: {e}")
            return False
    
    async def get_devices(self, refresh: bool = False) -> List[DeviceInfo]:
        """Get available Google Quantum AI devices."""
        if not refresh and self._is_cache_valid():
            return list(self._devices_cache.values())
        
        if not self._authenticated:
            await self.authenticate()
        
        devices = []
        try:
            # Mock Google devices for demonstration
            # In real implementation, this would query actual Google processors
            mock_devices = [
                {
                    "name": "rainbow",
                    "qubits": 70,
                    "status": "online"
                },
                {
                    "name": "weber", 
                    "qubits": 53,
                    "status": "online"
                }
            ]
            
            for device_data in mock_devices:
                capabilities = DeviceCapabilities(
                    max_qubits=device_data["qubits"],
                    gate_set=["sqrt_x", "sqrt_y", "z", "fsim", "measurement"],
                    max_shots=1000000,
                    supports_mid_circuit_measurement=True
                )
                
                device_info = DeviceInfo(
                    name=device_data["name"],
                    backend_type=BackendType.GOOGLE_QUANTUM_AI,
                    status=DeviceStatus.ONLINE if device_data["status"] == "online" else DeviceStatus.OFFLINE,
                    capabilities=capabilities,
                    provider="Google Quantum AI"
                )
                devices.append(device_info)
            
            self._update_cache(devices)
            
        except Exception as e:
            self.logger.error(f"Failed to get Google Quantum AI devices: {e}")
        
        return devices
    
    async def get_device_info(self, device_name: str) -> DeviceInfo:
        """Get detailed information about a specific Google device."""
        devices = await self.get_devices()
        for device in devices:
            if device.name == device_name:
                return device
        raise ValueError(f"Device {device_name} not found")
    
    async def submit_job(self, job_request: JobRequest) -> str:
        """Submit a job to Google Quantum AI."""
        # Mock implementation
        import uuid
        job_id = str(uuid.uuid4())
        self.logger.info(f"Mock Google job submitted: {job_id}")
        return job_id
    
    async def get_job_status(self, job_id: str) -> JobStatus:
        """Get Google Quantum AI job status."""
        return JobStatus.COMPLETED  # Mock
    
    async def get_job_result(self, job_id: str) -> JobResult:
        """Get Google Quantum AI job result."""
        # Mock result
        return JobResult(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            device_name="rainbow",
            shots=1024,
            counts={"00": 512, "11": 512}
        )
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel Google Quantum AI job."""
        return True  # Mock


class AWSBraketBackend(QuantumBackend):
    """AWS Braket backend implementation."""
    
    def __init__(self, credentials: Optional[Dict[str, Any]] = None):
        super().__init__("AWS Braket", BackendType.AWS_BRAKET, credentials)
        self._braket = None
    
    async def authenticate(self) -> bool:
        """Authenticate with AWS Braket."""
        try:
            import boto3
            from braket.aws import AwsDevice
            
            # Use credentials if provided, otherwise use default AWS credentials
            if self.credentials:
                session = boto3.Session(
                    aws_access_key_id=self.credentials.get("access_key_id"),
                    aws_secret_access_key=self.credentials.get("secret_access_key"),
                    region_name=self.credentials.get("region", "us-east-1")
                )
            else:
                session = boto3.Session()
            
            # Test access by listing devices
            devices = AwsDevice.get_devices()
            self._authenticated = True
            self.logger.info("Authenticated with AWS Braket")
            return True
            
        except ImportError:
            self.logger.error("AWS Braket SDK not available")
            return False
        except Exception as e:
            self.logger.error(f"AWS Braket authentication failed: {e}")
            return False
    
    async def get_devices(self, refresh: bool = False) -> List[DeviceInfo]:
        """Get available AWS Braket devices."""
        if not refresh and self._is_cache_valid():
            return list(self._devices_cache.values())
        
        if not self._authenticated:
            await self.authenticate()
        
        devices = []
        try:
            from braket.aws import AwsDevice
            
            aws_devices = AwsDevice.get_devices()
            
            for device in aws_devices:
                capabilities = DeviceCapabilities(
                    max_qubits=device.properties.paradigm.qubitCount if hasattr(device.properties.paradigm, 'qubitCount') else 30,
                    gate_set=getattr(device.properties.paradigm, 'nativeGateSet', []),
                    max_shots=100000
                )
                
                device_info = DeviceInfo(
                    name=device.name,
                    backend_type=BackendType.AWS_BRAKET,
                    status=DeviceStatus.ONLINE if device.status == "ONLINE" else DeviceStatus.OFFLINE,
                    capabilities=capabilities,
                    provider=device.provider_name
                )
                devices.append(device_info)
            
            self._update_cache(devices)
            
        except Exception as e:
            self.logger.error(f"Failed to get AWS Braket devices: {e}")
        
        return devices
    
    async def get_device_info(self, device_name: str) -> DeviceInfo:
        """Get detailed information about a specific AWS Braket device."""
        devices = await self.get_devices()
        for device in devices:
            if device.name == device_name:
                return device
        raise ValueError(f"Device {device_name} not found")
    
    async def submit_job(self, job_request: JobRequest) -> str:
        """Submit a job to AWS Braket."""
        # Mock implementation
        import uuid
        job_id = str(uuid.uuid4())
        self.logger.info(f"Mock AWS Braket job submitted: {job_id}")
        return job_id
    
    async def get_job_status(self, job_id: str) -> JobStatus:
        """Get AWS Braket job status."""
        return JobStatus.COMPLETED  # Mock
    
    async def get_job_result(self, job_id: str) -> JobResult:
        """Get AWS Braket job result."""
        # Mock result
        return JobResult(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            device_name="sv1",
            shots=1024,
            counts={"00": 500, "01": 124, "10": 200, "11": 200}
        )
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel AWS Braket job."""
        return True  # Mock


class HardwareBackendManager:
    """Manager for multiple quantum hardware backends."""
    
    def __init__(self):
        """Initialize the backend manager."""
        self.backends: Dict[str, QuantumBackend] = {}
        self.active_jobs: Dict[str, Tuple[str, str]] = {}  # job_id -> (backend_name, device_name)
        self.job_queue = queue.PriorityQueue()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.logger = logging.getLogger("quantrs2.hardware_manager")
    
    def register_backend(self, backend: QuantumBackend):
        """Register a quantum backend."""
        self.backends[backend.name] = backend
        self.logger.info(f"Registered backend: {backend.name}")
    
    def get_backend(self, name: str) -> Optional[QuantumBackend]:
        """Get a backend by name."""
        return self.backends.get(name)
    
    async def authenticate_all(self) -> Dict[str, bool]:
        """Authenticate all registered backends."""
        results = {}
        for name, backend in self.backends.items():
            try:
                success = await backend.authenticate()
                results[name] = success
            except Exception as e:
                results[name] = False
                self.logger.error(f"Authentication failed for {name}: {e}")
        return results
    
    async def get_all_devices(self, refresh: bool = False) -> Dict[str, List[DeviceInfo]]:
        """Get devices from all backends."""
        results = {}
        for name, backend in self.backends.items():
            try:
                devices = await backend.get_devices(refresh)
                results[name] = devices
            except Exception as e:
                results[name] = []
                self.logger.error(f"Failed to get devices from {name}: {e}")
        return results
    
    async def find_best_device(self, 
                             requirements: Dict[str, Any],
                             exclude_backends: Optional[List[str]] = None) -> Optional[DeviceInfo]:
        """
        Find the best device matching requirements.
        
        Args:
            requirements: Device requirements (min_qubits, preferred_backend, etc.)
            exclude_backends: Backends to exclude from search
            
        Returns:
            Best matching device or None
        """
        exclude_backends = exclude_backends or []
        min_qubits = requirements.get("min_qubits", 1)
        preferred_backend = requirements.get("preferred_backend")
        max_queue_length = requirements.get("max_queue_length", float('inf'))
        
        all_devices = await self.get_all_devices()
        candidates = []
        
        # Collect all devices that meet requirements
        for backend_name, devices in all_devices.items():
            if backend_name in exclude_backends:
                continue
                
            for device in devices:
                if (device.capabilities.max_qubits >= min_qubits and
                    device.status == DeviceStatus.ONLINE and
                    device.queue_length <= max_queue_length):
                    candidates.append(device)
        
        if not candidates:
            return None
        
        # Score devices based on preferences
        def score_device(device: DeviceInfo) -> float:
            score = 0.0
            
            # Prefer specific backend
            if preferred_backend and device.backend_type.value == preferred_backend:
                score += 100
            
            # Prefer fewer qubits (more efficient for small circuits)
            score += 50 / device.capabilities.max_qubits
            
            # Prefer shorter queues
            score += 20 / (device.queue_length + 1)
            
            # Prefer devices with better capabilities
            if device.capabilities.supports_mid_circuit_measurement:
                score += 10
            if device.capabilities.supports_reset:
                score += 5
            
            return score
        
        # Return best device
        best_device = max(candidates, key=score_device)
        return best_device
    
    async def submit_job_to_best_device(self, 
                                      circuit_data: Dict[str, Any],
                                      requirements: Optional[Dict[str, Any]] = None,
                                      **job_kwargs) -> Tuple[str, DeviceInfo]:
        """
        Submit job to the best available device.
        
        Args:
            circuit_data: Circuit to execute
            requirements: Device requirements
            **job_kwargs: Additional job parameters
            
        Returns:
            Tuple of (job_id, device_info)
        """
        requirements = requirements or {}
        
        # Find best device
        device = await self.find_best_device(requirements)
        if not device:
            raise RuntimeError("No suitable device found")
        
        # Get backend
        backend = None
        for backend_obj in self.backends.values():
            if backend_obj.backend_type == device.backend_type:
                backend = backend_obj
                break
        
        if not backend:
            raise RuntimeError(f"No backend available for {device.backend_type}")
        
        # Create job request
        job_request = JobRequest(
            circuit_data=circuit_data,
            device_name=device.name,
            **job_kwargs
        )
        
        # Submit job
        job_id = await backend.submit_job(job_request)
        self.active_jobs[job_id] = (backend.name, device.name)
        
        self.logger.info(f"Job {job_id} submitted to {device.name} on {backend.name}")
        return job_id, device
    
    async def get_job_status(self, job_id: str) -> JobStatus:
        """Get status of any job."""
        if job_id not in self.active_jobs:
            return JobStatus.FAILED
        
        backend_name, _ = self.active_jobs[job_id]
        backend = self.backends.get(backend_name)
        
        if not backend:
            return JobStatus.FAILED
        
        return await backend.get_job_status(job_id)
    
    async def get_job_result(self, job_id: str) -> JobResult:
        """Get result of any job."""
        if job_id not in self.active_jobs:
            return JobResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                device_name="unknown",
                shots=0,
                error_message="Job not found"
            )
        
        backend_name, _ = self.active_jobs[job_id]
        backend = self.backends.get(backend_name)
        
        if not backend:
            return JobResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                device_name="unknown",
                shots=0,
                error_message="Backend not found"
            )
        
        result = await backend.get_job_result(job_id)
        
        # Clean up completed jobs
        if result.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            del self.active_jobs[job_id]
        
        return result
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel any job."""
        if job_id not in self.active_jobs:
            return False
        
        backend_name, _ = self.active_jobs[job_id]
        backend = self.backends.get(backend_name)
        
        if not backend:
            return False
        
        success = await backend.cancel_job(job_id)
        if success:
            del self.active_jobs[job_id]
        
        return success
    
    async def wait_for_job(self, 
                          job_id: str, 
                          timeout: Optional[float] = None,
                          poll_interval: float = 5.0) -> JobResult:
        """
        Wait for a job to complete.
        
        Args:
            job_id: Job ID to wait for
            timeout: Maximum time to wait in seconds
            poll_interval: How often to check job status
            
        Returns:
            Job result when completed
        """
        start_time = time.time()
        
        while True:
            result = await self.get_job_result(job_id)
            
            if result.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                return result
            
            if timeout and (time.time() - start_time) > timeout:
                return JobResult(
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    device_name="unknown",
                    shots=0,
                    error_message="Timeout waiting for job completion"
                )
            
            await asyncio.sleep(poll_interval)
    
    def get_active_jobs(self) -> Dict[str, Tuple[str, str]]:
        """Get all active jobs."""
        return self.active_jobs.copy()
    
    def get_backend_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all backends."""
        stats = {}
        for name, backend in self.backends.items():
            backend_jobs = [job_id for job_id, (backend_name, _) in self.active_jobs.items() 
                          if backend_name == name]
            stats[name] = {
                "authenticated": backend._authenticated,
                "active_jobs": len(backend_jobs),
                "backend_type": backend.backend_type.value
            }
        return stats


# Global hardware backend manager instance
_hardware_manager = None
_manager_lock = threading.Lock()


def get_hardware_manager() -> HardwareBackendManager:
    """Get the global hardware backend manager."""
    global _hardware_manager
    with _manager_lock:
        if _hardware_manager is None:
            _hardware_manager = HardwareBackendManager()
        return _hardware_manager


def register_ibm_backend(credentials: Optional[Dict[str, Any]] = None):
    """Register IBM Quantum backend."""
    backend = IBMQuantumBackend(credentials)
    get_hardware_manager().register_backend(backend)
    return backend


def register_google_backend(credentials: Optional[Dict[str, Any]] = None):
    """Register Google Quantum AI backend."""
    backend = GoogleQuantumAIBackend(credentials)
    get_hardware_manager().register_backend(backend)
    return backend


def register_aws_backend(credentials: Optional[Dict[str, Any]] = None):
    """Register AWS Braket backend."""
    backend = AWSBraketBackend(credentials)
    get_hardware_manager().register_backend(backend)
    return backend


async def submit_to_hardware(circuit_data: Dict[str, Any],
                           requirements: Optional[Dict[str, Any]] = None,
                           **kwargs) -> Tuple[str, DeviceInfo]:
    """
    Convenience function to submit a job to the best available hardware.
    
    Args:
        circuit_data: Circuit to execute
        requirements: Device requirements
        **kwargs: Additional job parameters
        
    Returns:
        Tuple of (job_id, device_info)
    """
    manager = get_hardware_manager()
    return await manager.submit_job_to_best_device(circuit_data, requirements, **kwargs)


async def get_hardware_devices(backend_names: Optional[List[str]] = None) -> Dict[str, List[DeviceInfo]]:
    """
    Get devices from specified backends.
    
    Args:
        backend_names: List of backend names to query (None for all)
        
    Returns:
        Dictionary mapping backend names to device lists
    """
    manager = get_hardware_manager()
    all_devices = await manager.get_all_devices()
    
    if backend_names:
        return {name: devices for name, devices in all_devices.items() 
                if name in backend_names}
    
    return all_devices


__all__ = [
    'BackendType',
    'JobStatus', 
    'DeviceStatus',
    'DeviceCapabilities',
    'DeviceInfo',
    'JobRequest',
    'JobResult',
    'QuantumBackend',
    'IBMQuantumBackend',
    'GoogleQuantumAIBackend',
    'AWSBraketBackend',
    'HardwareBackendManager',
    'get_hardware_manager',
    'register_ibm_backend',
    'register_google_backend',
    'register_aws_backend',
    'submit_to_hardware',
    'get_hardware_devices'
]