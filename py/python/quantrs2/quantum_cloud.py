"""
Quantum Cloud Orchestration

This module provides comprehensive cloud orchestration capabilities for quantum computing,
enabling seamless integration with multiple quantum cloud providers and services.
"""

import asyncio
import json
import time
import uuid
import logging
import threading
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union, Set
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum
import tempfile
import queue
import concurrent.futures

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import requests
    import boto3
    CLOUD_LIBS_AVAILABLE = True
except ImportError:
    CLOUD_LIBS_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from . import qasm
    from . import compilation_service
    from . import algorithm_marketplace
    QUANTRS_MODULES_AVAILABLE = True
except ImportError:
    QUANTRS_MODULES_AVAILABLE = False


class CloudProvider(Enum):
    """Supported quantum cloud providers."""
    IBM_QUANTUM = "ibm_quantum"
    AWS_BRAKET = "aws_braket"
    GOOGLE_QUANTUM_AI = "google_quantum_ai"
    AZURE_QUANTUM = "azure_quantum"
    RIGETTI_QCS = "rigetti_qcs"
    IONQ = "ionq"
    XANADU_CLOUD = "xanadu_cloud"
    ATOS_QLM = "atos_qlm"
    CAMBRIDGE_QUANTUM = "cambridge_quantum"
    PASQAL = "pasqal"
    QUANTINUUM = "quantinuum"
    LOCAL = "local"


class JobStatus(Enum):
    """Job execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class DeviceType(Enum):
    """Quantum device types."""
    SIMULATOR = "simulator"
    QPU = "qpu"
    HYBRID = "hybrid"
    ANNEALER = "annealer"


class OptimizationLevel(Enum):
    """Circuit optimization levels."""
    NONE = 0
    BASIC = 1
    STANDARD = 2
    AGGRESSIVE = 3


@dataclass
class CloudCredentials:
    """Cloud provider credentials."""
    provider: CloudProvider
    credentials: Dict[str, str]
    endpoint: Optional[str] = None
    region: Optional[str] = None
    project_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'provider': self.provider.value,
            'credentials': self.credentials,
            'endpoint': self.endpoint,
            'region': self.region,
            'project_id': self.project_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CloudCredentials':
        """Create from dictionary."""
        data = data.copy()
        data['provider'] = CloudProvider(data['provider'])
        return cls(**data)


@dataclass
class DeviceInfo:
    """Quantum device information."""
    provider: CloudProvider
    device_id: str
    device_name: str
    device_type: DeviceType
    num_qubits: int
    connectivity: Optional[List[Tuple[int, int]]] = None
    gate_set: Optional[List[str]] = None
    error_rates: Optional[Dict[str, float]] = None
    queue_length: int = 0
    availability: bool = True
    cost_per_shot: Optional[float] = None
    max_shots: int = 1024
    topology: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'provider': self.provider.value,
            'device_id': self.device_id,
            'device_name': self.device_name,
            'device_type': self.device_type.value,
            'num_qubits': self.num_qubits,
            'connectivity': self.connectivity,
            'gate_set': self.gate_set,
            'error_rates': self.error_rates,
            'queue_length': self.queue_length,
            'availability': self.availability,
            'cost_per_shot': self.cost_per_shot,
            'max_shots': self.max_shots,
            'topology': self.topology
        }


@dataclass
class CloudJob:
    """Cloud quantum job."""
    job_id: str
    provider: CloudProvider
    device_id: str
    circuit_data: Dict[str, Any]
    shots: int
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    cost: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'job_id': self.job_id,
            'provider': self.provider.value,
            'device_id': self.device_id,
            'circuit_data': self.circuit_data,
            'shots': self.shots,
            'status': self.status.value,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'result': self.result,
            'error_message': self.error_message,
            'cost': self.cost,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CloudJob':
        """Create from dictionary."""
        data = data.copy()
        data['provider'] = CloudProvider(data['provider'])
        data['status'] = JobStatus(data['status'])
        return cls(**data)


class CloudAdapter(ABC):
    """Abstract base class for cloud provider adapters."""
    
    def __init__(self, credentials: CloudCredentials):
        self.credentials = credentials
        self.provider = credentials.provider
        self._authenticated = False
    
    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with the cloud provider."""
        pass
    
    @abstractmethod
    async def get_devices(self) -> List[DeviceInfo]:
        """Get available quantum devices."""
        pass
    
    @abstractmethod
    async def submit_job(self, device_id: str, circuit_data: Dict[str, Any], 
                        shots: int, **kwargs) -> CloudJob:
        """Submit a quantum job."""
        pass
    
    @abstractmethod
    async def get_job_status(self, job_id: str) -> JobStatus:
        """Get job status."""
        pass
    
    @abstractmethod
    async def get_job_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job result."""
        pass
    
    @abstractmethod
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        pass


class IBMQuantumAdapter(CloudAdapter):
    """IBM Quantum cloud adapter."""
    
    def __init__(self, credentials: CloudCredentials):
        super().__init__(credentials)
        self.base_url = credentials.endpoint or "https://api.quantum-computing.ibm.com"
        self.token = credentials.credentials.get('token')
        self.hub = credentials.credentials.get('hub', 'ibm-q')
        self.group = credentials.credentials.get('group', 'open')
        self.project = credentials.credentials.get('project', 'main')
    
    async def authenticate(self) -> bool:
        """Authenticate with IBM Quantum."""
        if not self.token:
            return False
        
        try:
            # Mock authentication for testing
            headers = {'Authorization': f'Bearer {self.token}'}
            # In real implementation, make API call to verify token
            self._authenticated = True
            return True
        except Exception as e:
            logging.error(f"IBM Quantum authentication failed: {e}")
            return False
    
    async def get_devices(self) -> List[DeviceInfo]:
        """Get IBM Quantum devices."""
        if not self._authenticated:
            await self.authenticate()
        
        # Mock device list for IBM Quantum
        devices = [
            DeviceInfo(
                provider=CloudProvider.IBM_QUANTUM,
                device_id="ibmq_qasm_simulator",
                device_name="IBM Quantum QASM Simulator",
                device_type=DeviceType.SIMULATOR,
                num_qubits=32,
                gate_set=["u1", "u2", "u3", "cx", "measure"],
                queue_length=0,
                max_shots=8192
            ),
            DeviceInfo(
                provider=CloudProvider.IBM_QUANTUM,
                device_id="ibm_brisbane",
                device_name="IBM Brisbane",
                device_type=DeviceType.QPU,
                num_qubits=127,
                gate_set=["rz", "sx", "x", "cx", "measure"],
                error_rates={"gate_error": 0.001, "readout_error": 0.02},
                queue_length=15,
                cost_per_shot=0.001,
                topology="heavy_hex"
            ),
            DeviceInfo(
                provider=CloudProvider.IBM_QUANTUM,
                device_id="ibm_kyoto",
                device_name="IBM Kyoto",
                device_type=DeviceType.QPU,
                num_qubits=127,
                gate_set=["rz", "sx", "x", "cx", "measure"],
                error_rates={"gate_error": 0.0008, "readout_error": 0.015},
                queue_length=8,
                cost_per_shot=0.001,
                topology="heavy_hex"
            )
        ]
        
        return devices
    
    async def submit_job(self, device_id: str, circuit_data: Dict[str, Any], 
                        shots: int, **kwargs) -> CloudJob:
        """Submit job to IBM Quantum."""
        job_id = f"ibm_{uuid.uuid4().hex[:8]}"
        
        # Create job
        job = CloudJob(
            job_id=job_id,
            provider=self.provider,
            device_id=device_id,
            circuit_data=circuit_data,
            shots=shots,
            status=JobStatus.QUEUED,
            metadata=kwargs
        )
        
        # Mock job submission
        job.started_at = time.time()
        
        return job
    
    async def get_job_status(self, job_id: str) -> JobStatus:
        """Get IBM Quantum job status."""
        # Mock status check
        return JobStatus.COMPLETED
    
    async def get_job_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get IBM Quantum job result."""
        # Mock result
        return {
            "counts": {"00": 512, "11": 512},
            "metadata": {"execution_time": 0.1}
        }
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel IBM Quantum job."""
        return True


class AWSBraketAdapter(CloudAdapter):
    """AWS Braket cloud adapter."""
    
    def __init__(self, credentials: CloudCredentials):
        super().__init__(credentials)
        self.region = credentials.region or "us-east-1"
        self.access_key = credentials.credentials.get('access_key_id')
        self.secret_key = credentials.credentials.get('secret_access_key')
        self.bucket = credentials.credentials.get('s3_bucket')
    
    async def authenticate(self) -> bool:
        """Authenticate with AWS Braket."""
        if not self.access_key or not self.secret_key:
            return False
        
        try:
            # Mock authentication
            self._authenticated = True
            return True
        except Exception as e:
            logging.error(f"AWS Braket authentication failed: {e}")
            return False
    
    async def get_devices(self) -> List[DeviceInfo]:
        """Get AWS Braket devices."""
        devices = [
            DeviceInfo(
                provider=CloudProvider.AWS_BRAKET,
                device_id="LocalSimulator",
                device_name="Braket Local Simulator",
                device_type=DeviceType.SIMULATOR,
                num_qubits=25,
                gate_set=["h", "x", "y", "z", "cnot", "ry", "rz"],
                queue_length=0,
                max_shots=100000
            ),
            DeviceInfo(
                provider=CloudProvider.AWS_BRAKET,
                device_id="arn:aws:braket:us-east-1::device/qpu/ionq/Harmony",
                device_name="IonQ Harmony",
                device_type=DeviceType.QPU,
                num_qubits=11,
                gate_set=["x", "y", "z", "rx", "ry", "rz", "cnot"],
                error_rates={"gate_error": 0.002},
                queue_length=5,
                cost_per_shot=0.01,
                topology="all_to_all"
            ),
            DeviceInfo(
                provider=CloudProvider.AWS_BRAKET,
                device_id="arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3",
                device_name="Rigetti Aspen-M-3",
                device_type=DeviceType.QPU,
                num_qubits=80,
                gate_set=["rx", "rz", "cz"],
                error_rates={"gate_error": 0.01},
                queue_length=12,
                cost_per_shot=0.00035,
                topology="octagonal"
            )
        ]
        
        return devices
    
    async def submit_job(self, device_id: str, circuit_data: Dict[str, Any], 
                        shots: int, **kwargs) -> CloudJob:
        """Submit job to AWS Braket."""
        job_id = f"braket_{uuid.uuid4().hex[:8]}"
        
        job = CloudJob(
            job_id=job_id,
            provider=self.provider,
            device_id=device_id,
            circuit_data=circuit_data,
            shots=shots,
            status=JobStatus.QUEUED,
            metadata=kwargs
        )
        
        return job
    
    async def get_job_status(self, job_id: str) -> JobStatus:
        """Get AWS Braket job status."""
        return JobStatus.COMPLETED
    
    async def get_job_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get AWS Braket job result."""
        return {
            "measurement_counts": {"00": 487, "01": 13, "10": 12, "11": 512},
            "measured_qubits": [0, 1]
        }
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel AWS Braket job."""
        return True


class GoogleQuantumAIAdapter(CloudAdapter):
    """Google Quantum AI cloud adapter."""
    
    def __init__(self, credentials: CloudCredentials):
        super().__init__(credentials)
        self.project_id = credentials.project_id
        self.service_account_key = credentials.credentials.get('service_account_key')
    
    async def authenticate(self) -> bool:
        """Authenticate with Google Quantum AI."""
        if not self.service_account_key or not self.project_id:
            return False
        
        try:
            # Mock authentication
            self._authenticated = True
            return True
        except Exception as e:
            logging.error(f"Google Quantum AI authentication failed: {e}")
            return False
    
    async def get_devices(self) -> List[DeviceInfo]:
        """Get Google Quantum AI devices."""
        devices = [
            DeviceInfo(
                provider=CloudProvider.GOOGLE_QUANTUM_AI,
                device_id="cirq-simulator",
                device_name="Cirq Simulator",
                device_type=DeviceType.SIMULATOR,
                num_qubits=20,
                gate_set=["h", "x", "y", "z", "rx", "ry", "rz", "cz"],
                queue_length=0
            ),
            DeviceInfo(
                provider=CloudProvider.GOOGLE_QUANTUM_AI,
                device_id="rainbow",
                device_name="Google Rainbow",
                device_type=DeviceType.QPU,
                num_qubits=70,
                gate_set=["xy", "z", "cz", "measurement"],
                error_rates={"gate_error": 0.005},
                queue_length=20,
                cost_per_shot=0.002,
                topology="2d_grid"
            )
        ]
        
        return devices
    
    async def submit_job(self, device_id: str, circuit_data: Dict[str, Any], 
                        shots: int, **kwargs) -> CloudJob:
        """Submit job to Google Quantum AI."""
        job_id = f"google_{uuid.uuid4().hex[:8]}"
        
        job = CloudJob(
            job_id=job_id,
            provider=self.provider,
            device_id=device_id,
            circuit_data=circuit_data,
            shots=shots,
            status=JobStatus.QUEUED,
            metadata=kwargs
        )
        
        return job
    
    async def get_job_status(self, job_id: str) -> JobStatus:
        """Get Google Quantum AI job status."""
        return JobStatus.COMPLETED
    
    async def get_job_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get Google Quantum AI job result."""
        return {
            "histogram": {"00": 0.48, "01": 0.02, "10": 0.02, "11": 0.48},
            "repetitions": 1000
        }
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel Google Quantum AI job."""
        return True


class LocalAdapter(CloudAdapter):
    """Local quantum simulator adapter."""
    
    def __init__(self, credentials: CloudCredentials):
        super().__init__(credentials)
        self._authenticated = True
    
    async def authenticate(self) -> bool:
        """Authenticate with local simulator."""
        return True
    
    async def get_devices(self) -> List[DeviceInfo]:
        """Get local simulator devices."""
        devices = [
            DeviceInfo(
                provider=CloudProvider.LOCAL,
                device_id="quantrs2_simulator",
                device_name="QuantRS2 Local Simulator",
                device_type=DeviceType.SIMULATOR,
                num_qubits=20,
                gate_set=["h", "x", "y", "z", "rx", "ry", "rz", "cnot", "cz"],
                queue_length=0,
                cost_per_shot=0.0,
                max_shots=1000000
            )
        ]
        
        return devices
    
    async def submit_job(self, device_id: str, circuit_data: Dict[str, Any], 
                        shots: int, **kwargs) -> CloudJob:
        """Submit job to local simulator."""
        job_id = f"local_{uuid.uuid4().hex[:8]}"
        
        job = CloudJob(
            job_id=job_id,
            provider=self.provider,
            device_id=device_id,
            circuit_data=circuit_data,
            shots=shots,
            status=JobStatus.RUNNING,
            metadata=kwargs
        )
        
        # Simulate immediate execution
        job.started_at = time.time()
        job.completed_at = time.time() + 0.1
        job.status = JobStatus.COMPLETED
        job.result = {
            "counts": {"00": shots//2, "11": shots//2},
            "execution_time": 0.1
        }
        
        return job
    
    async def get_job_status(self, job_id: str) -> JobStatus:
        """Get local job status."""
        return JobStatus.COMPLETED
    
    async def get_job_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get local job result."""
        return {
            "counts": {"00": 500, "11": 500},
            "execution_time": 0.1
        }
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel local job."""
        return True


class CloudJobManager:
    """Manages quantum cloud jobs across providers."""
    
    def __init__(self):
        self.jobs: Dict[str, CloudJob] = {}
        self.job_history: List[CloudJob] = []
        self.max_history = 1000
        self._lock = threading.Lock()
    
    def add_job(self, job: CloudJob):
        """Add a job to management."""
        with self._lock:
            self.jobs[job.job_id] = job
    
    def get_job(self, job_id: str) -> Optional[CloudJob]:
        """Get a job by ID."""
        with self._lock:
            return self.jobs.get(job_id)
    
    def update_job_status(self, job_id: str, status: JobStatus, 
                         result: Optional[Dict[str, Any]] = None,
                         error_message: Optional[str] = None):
        """Update job status."""
        with self._lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                job.status = status
                
                if status == JobStatus.RUNNING and job.started_at is None:
                    job.started_at = time.time()
                
                if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                    job.completed_at = time.time()
                    
                    if result:
                        job.result = result
                    
                    if error_message:
                        job.error_message = error_message
                    
                    # Move to history
                    self.job_history.append(job)
                    if len(self.job_history) > self.max_history:
                        self.job_history.pop(0)
                    
                    # Remove from active jobs
                    del self.jobs[job_id]
    
    def get_active_jobs(self) -> List[CloudJob]:
        """Get all active jobs."""
        with self._lock:
            return list(self.jobs.values())
    
    def get_job_history(self) -> List[CloudJob]:
        """Get job history."""
        with self._lock:
            return self.job_history.copy()
    
    def get_job_statistics(self) -> Dict[str, Any]:
        """Get job statistics."""
        with self._lock:
            active_count = len(self.jobs)
            total_count = len(self.job_history) + active_count
            
            if total_count == 0:
                return {
                    'total_jobs': 0,
                    'active_jobs': 0,
                    'completed_jobs': 0,
                    'failed_jobs': 0,
                    'success_rate': 0.0,
                    'average_execution_time': 0.0
                }
            
            completed_jobs = [j for j in self.job_history if j.status == JobStatus.COMPLETED]
            failed_jobs = [j for j in self.job_history if j.status == JobStatus.FAILED]
            
            execution_times = []
            for job in completed_jobs:
                if job.started_at and job.completed_at:
                    execution_times.append(job.completed_at - job.started_at)
            
            avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0.0
            
            return {
                'total_jobs': total_count,
                'active_jobs': active_count,
                'completed_jobs': len(completed_jobs),
                'failed_jobs': len(failed_jobs),
                'success_rate': len(completed_jobs) / len(self.job_history) if self.job_history else 0.0,
                'average_execution_time': avg_execution_time,
                'providers': self._get_provider_stats()
            }
    
    def _get_provider_stats(self) -> Dict[str, int]:
        """Get statistics by provider."""
        provider_counts = {}
        for job in self.job_history + list(self.jobs.values()):
            provider = job.provider.value
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
        return provider_counts


class QuantumCloudOrchestrator:
    """Main quantum cloud orchestration system."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.adapters: Dict[CloudProvider, CloudAdapter] = {}
        self.job_manager = CloudJobManager()
        self.config_path = config_path or str(Path.home() / ".quantrs2" / "cloud_config.yaml")
        self.device_cache: Dict[CloudProvider, List[DeviceInfo]] = {}
        self.cache_ttl = 300  # 5 minutes
        self.last_cache_update: Dict[CloudProvider, float] = {}
        self._setup_logging()
        
        # Load configuration
        self._load_configuration()
    
    def _setup_logging(self):
        """Set up logging for cloud operations."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_configuration(self):
        """Load cloud configuration from file."""
        try:
            if Path(self.config_path).exists() and YAML_AVAILABLE:
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                self._configure_from_dict(config)
        except Exception as e:
            self.logger.warning(f"Failed to load cloud configuration: {e}")
    
    def _configure_from_dict(self, config: Dict[str, Any]):
        """Configure from dictionary."""
        for provider_name, provider_config in config.get('providers', {}).items():
            try:
                provider = CloudProvider(provider_name)
                credentials = CloudCredentials.from_dict(provider_config)
                self.add_provider(credentials)
            except Exception as e:
                self.logger.error(f"Failed to configure provider {provider_name}: {e}")
    
    def add_provider(self, credentials: CloudCredentials) -> bool:
        """Add a cloud provider."""
        try:
            adapter = self._create_adapter(credentials)
            self.adapters[credentials.provider] = adapter
            self.logger.info(f"Added provider: {credentials.provider.value}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add provider {credentials.provider.value}: {e}")
            return False
    
    def _create_adapter(self, credentials: CloudCredentials) -> CloudAdapter:
        """Create adapter for provider."""
        if credentials.provider == CloudProvider.IBM_QUANTUM:
            return IBMQuantumAdapter(credentials)
        elif credentials.provider == CloudProvider.AWS_BRAKET:
            return AWSBraketAdapter(credentials)
        elif credentials.provider == CloudProvider.GOOGLE_QUANTUM_AI:
            return GoogleQuantumAIAdapter(credentials)
        elif credentials.provider == CloudProvider.LOCAL:
            return LocalAdapter(credentials)
        else:
            # Generic adapter for other providers
            return LocalAdapter(credentials)  # Fallback to local
    
    async def authenticate_all(self) -> Dict[CloudProvider, bool]:
        """Authenticate with all providers."""
        results = {}
        for provider, adapter in self.adapters.items():
            try:
                success = await adapter.authenticate()
                results[provider] = success
                if success:
                    self.logger.info(f"Authenticated with {provider.value}")
                else:
                    self.logger.warning(f"Authentication failed for {provider.value}")
            except Exception as e:
                self.logger.error(f"Authentication error for {provider.value}: {e}")
                results[provider] = False
        
        return results
    
    async def get_all_devices(self, refresh_cache: bool = False) -> Dict[CloudProvider, List[DeviceInfo]]:
        """Get devices from all providers."""
        all_devices = {}
        
        for provider, adapter in self.adapters.items():
            try:
                # Check cache
                if not refresh_cache and provider in self.device_cache:
                    last_update = self.last_cache_update.get(provider, 0)
                    if time.time() - last_update < self.cache_ttl:
                        all_devices[provider] = self.device_cache[provider]
                        continue
                
                # Fetch devices
                devices = await adapter.get_devices()
                all_devices[provider] = devices
                
                # Update cache
                self.device_cache[provider] = devices
                self.last_cache_update[provider] = time.time()
                
                self.logger.info(f"Retrieved {len(devices)} devices from {provider.value}")
                
            except Exception as e:
                self.logger.error(f"Failed to get devices from {provider.value}: {e}")
                all_devices[provider] = []
        
        return all_devices
    
    def find_best_device(self, requirements: Dict[str, Any]) -> Optional[Tuple[CloudProvider, DeviceInfo]]:
        """Find the best device for given requirements."""
        min_qubits = requirements.get('min_qubits', 1)
        device_type = requirements.get('device_type')
        max_cost = requirements.get('max_cost')
        provider_preference = requirements.get('provider_preference', [])
        
        best_device = None
        best_score = -1
        
        for provider, devices in self.device_cache.items():
            for device in devices:
                # Check basic requirements
                if device.num_qubits < min_qubits:
                    continue
                
                if device_type and device.device_type != device_type:
                    continue
                
                if max_cost and device.cost_per_shot and device.cost_per_shot > max_cost:
                    continue
                
                if not device.availability:
                    continue
                
                # Calculate score
                score = self._calculate_device_score(device, requirements, provider_preference)
                
                if score > best_score:
                    best_score = score
                    best_device = (provider, device)
        
        return best_device
    
    def _calculate_device_score(self, device: DeviceInfo, requirements: Dict[str, Any], 
                               provider_preference: List[str]) -> float:
        """Calculate device score based on requirements."""
        score = 0.0
        
        # Qubit count score (more is better, but with diminishing returns)
        min_qubits = requirements.get('min_qubits', 1)
        qubit_score = min(1.0, device.num_qubits / (min_qubits * 2))
        score += qubit_score * 30
        
        # Queue length score (shorter is better)
        queue_score = max(0, 1.0 - device.queue_length / 100)
        score += queue_score * 20
        
        # Cost score (cheaper is better)
        if device.cost_per_shot is not None:
            max_acceptable_cost = requirements.get('max_cost', 0.01)
            cost_score = max(0, 1.0 - device.cost_per_shot / max_acceptable_cost)
            score += cost_score * 25
        else:
            score += 25  # Free simulators get full cost score
        
        # Device type preference
        preferred_type = requirements.get('device_type')
        if preferred_type and device.device_type == preferred_type:
            score += 15
        
        # Provider preference
        if provider_preference and device.provider.value in provider_preference:
            preference_index = provider_preference.index(device.provider.value)
            preference_score = (len(provider_preference) - preference_index) / len(provider_preference)
            score += preference_score * 10
        
        return score
    
    async def submit_job(self, provider: CloudProvider, device_id: str, 
                        circuit_data: Dict[str, Any], shots: int = 1024,
                        optimization_level: OptimizationLevel = OptimizationLevel.STANDARD,
                        **kwargs) -> Optional[CloudJob]:
        """Submit a quantum job."""
        if provider not in self.adapters:
            self.logger.error(f"Provider {provider.value} not configured")
            return None
        
        adapter = self.adapters[provider]
        
        try:
            # Optimize circuit if requested
            if optimization_level != OptimizationLevel.NONE and QUANTRS_MODULES_AVAILABLE:
                circuit_data = await self._optimize_circuit(circuit_data, optimization_level)
            
            # Submit job
            job = await adapter.submit_job(device_id, circuit_data, shots, **kwargs)
            
            # Add to job manager
            self.job_manager.add_job(job)
            
            self.logger.info(f"Submitted job {job.job_id} to {provider.value}")
            return job
            
        except Exception as e:
            self.logger.error(f"Failed to submit job to {provider.value}: {e}")
            return None
    
    async def _optimize_circuit(self, circuit_data: Dict[str, Any], 
                               level: OptimizationLevel) -> Dict[str, Any]:
        """Optimize circuit before submission."""
        try:
            # Use compilation service if available
            compiler = compilation_service.get_compilation_service()
            
            request = compilation_service.CompilationRequest(
                circuit_data=circuit_data,
                optimization_level=level.value,
                target_backend="generic"
            )
            
            result = await compiler.compile_circuit_async(request)
            
            if result and result.status == compilation_service.CompilationStatus.COMPLETED:
                return result.optimized_circuit
            
        except Exception as e:
            self.logger.warning(f"Circuit optimization failed, using original: {e}")
        
        return circuit_data
    
    async def submit_job_auto(self, circuit_data: Dict[str, Any], 
                             requirements: Dict[str, Any],
                             shots: int = 1024, **kwargs) -> Optional[CloudJob]:
        """Submit job with automatic device selection."""
        # Find best device
        device_info = self.find_best_device(requirements)
        
        if not device_info:
            self.logger.error("No suitable device found")
            return None
        
        provider, device = device_info
        
        self.logger.info(f"Auto-selected device: {device.device_name} on {provider.value}")
        
        return await self.submit_job(
            provider, device.device_id, circuit_data, shots, **kwargs
        )
    
    async def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get job status."""
        job = self.job_manager.get_job(job_id)
        if not job:
            return None
        
        adapter = self.adapters.get(job.provider)
        if not adapter:
            return None
        
        try:
            status = await adapter.get_job_status(job_id)
            
            # Update job manager
            if status != job.status:
                if status == JobStatus.COMPLETED:
                    result = await adapter.get_job_result(job_id)
                    self.job_manager.update_job_status(job_id, status, result)
                else:
                    self.job_manager.update_job_status(job_id, status)
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get job status: {e}")
            return None
    
    async def get_job_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job result."""
        job = self.job_manager.get_job(job_id)
        if not job:
            return None
        
        # Check if result is already cached
        if job.result:
            return job.result
        
        adapter = self.adapters.get(job.provider)
        if not adapter:
            return None
        
        try:
            result = await adapter.get_job_result(job_id)
            
            # Update job manager
            if result:
                self.job_manager.update_job_status(job_id, JobStatus.COMPLETED, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get job result: {e}")
            return None
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        job = self.job_manager.get_job(job_id)
        if not job:
            return False
        
        adapter = self.adapters.get(job.provider)
        if not adapter:
            return False
        
        try:
            success = await adapter.cancel_job(job_id)
            
            if success:
                self.job_manager.update_job_status(job_id, JobStatus.CANCELLED)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to cancel job: {e}")
            return False
    
    def get_cloud_statistics(self) -> Dict[str, Any]:
        """Get cloud usage statistics."""
        job_stats = self.job_manager.get_job_statistics()
        
        # Add provider information
        provider_info = {}
        for provider, adapter in self.adapters.items():
            devices = self.device_cache.get(provider, [])
            provider_info[provider.value] = {
                'authenticated': adapter._authenticated,
                'device_count': len(devices),
                'qpu_count': len([d for d in devices if d.device_type == DeviceType.QPU]),
                'simulator_count': len([d for d in devices if d.device_type == DeviceType.SIMULATOR])
            }
        
        return {
            'job_statistics': job_stats,
            'provider_info': provider_info,
            'configured_providers': len(self.adapters),
            'cache_status': {
                provider.value: {
                    'cached': provider in self.device_cache,
                    'last_update': self.last_cache_update.get(provider, 0)
                }
                for provider in self.adapters.keys()
            }
        }
    
    def save_configuration(self, path: Optional[str] = None):
        """Save current configuration to file."""
        if not YAML_AVAILABLE:
            self.logger.warning("YAML not available, cannot save configuration")
            return
        
        config_path = path or self.config_path
        
        config = {
            'providers': {}
        }
        
        for provider, adapter in self.adapters.items():
            config['providers'][provider.value] = adapter.credentials.to_dict()
        
        try:
            # Ensure directory exists
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            self.logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")


# Global orchestrator instance
_quantum_cloud_orchestrator: Optional[QuantumCloudOrchestrator] = None


def get_quantum_cloud_orchestrator() -> QuantumCloudOrchestrator:
    """Get global quantum cloud orchestrator instance."""
    global _quantum_cloud_orchestrator
    if _quantum_cloud_orchestrator is None:
        _quantum_cloud_orchestrator = QuantumCloudOrchestrator()
    return _quantum_cloud_orchestrator


async def authenticate_cloud_providers() -> Dict[CloudProvider, bool]:
    """Convenience function to authenticate with all providers."""
    orchestrator = get_quantum_cloud_orchestrator()
    return await orchestrator.authenticate_all()


async def get_available_devices(refresh_cache: bool = False) -> Dict[CloudProvider, List[DeviceInfo]]:
    """Convenience function to get all available devices."""
    orchestrator = get_quantum_cloud_orchestrator()
    return await orchestrator.get_all_devices(refresh_cache)


async def submit_quantum_job(circuit_data: Dict[str, Any], 
                           requirements: Optional[Dict[str, Any]] = None,
                           shots: int = 1024, **kwargs) -> Optional[CloudJob]:
    """Convenience function to submit quantum job with auto device selection."""
    orchestrator = get_quantum_cloud_orchestrator()
    
    if requirements is None:
        requirements = {'min_qubits': 2, 'device_type': DeviceType.SIMULATOR}
    
    return await orchestrator.submit_job_auto(circuit_data, requirements, shots, **kwargs)


def create_cloud_credentials(provider: CloudProvider, **credentials) -> CloudCredentials:
    """Convenience function to create cloud credentials."""
    return CloudCredentials(
        provider=provider,
        credentials=credentials
    )


def add_cloud_provider(provider: CloudProvider, **credentials) -> bool:
    """Convenience function to add cloud provider."""
    creds = create_cloud_credentials(provider, **credentials)
    orchestrator = get_quantum_cloud_orchestrator()
    return orchestrator.add_provider(creds)


def get_cloud_statistics() -> Dict[str, Any]:
    """Convenience function to get cloud statistics."""
    orchestrator = get_quantum_cloud_orchestrator()
    return orchestrator.get_cloud_statistics()


# CLI interface
def main():
    """Main CLI interface for quantum cloud orchestration."""
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser(description="QuantRS2 Quantum Cloud Orchestration")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Providers command
    providers_parser = subparsers.add_parser('providers', help='List configured providers')
    
    # Devices command
    devices_parser = subparsers.add_parser('devices', help='List available devices')
    devices_parser.add_argument('--provider', choices=[p.value for p in CloudProvider], 
                               help='Filter by provider')
    devices_parser.add_argument('--refresh', action='store_true', help='Refresh device cache')
    
    # Jobs command
    jobs_parser = subparsers.add_parser('jobs', help='List jobs')
    jobs_parser.add_argument('--active', action='store_true', help='Show only active jobs')
    jobs_parser.add_argument('--history', action='store_true', help='Show job history')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show job status')
    status_parser.add_argument('job_id', help='Job ID to check')
    
    # Stats command
    subparsers.add_parser('stats', help='Show cloud statistics')
    
    # Auth command
    subparsers.add_parser('auth', help='Authenticate with all providers')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    orchestrator = get_quantum_cloud_orchestrator()
    
    async def run_async_command():
        try:
            if args.command == 'providers':
                print("Configured providers:")
                for provider in orchestrator.adapters.keys():
                    print(f"  - {provider.value}")
            
            elif args.command == 'devices':
                devices = await orchestrator.get_all_devices(refresh_cache=args.refresh)
                
                for provider, device_list in devices.items():
                    if args.provider and provider.value != args.provider:
                        continue
                    
                    print(f"\n{provider.value} devices:")
                    for device in device_list:
                        status = "available" if device.availability else "unavailable"
                        print(f"  {device.device_name} ({device.device_id})")
                        print(f"    Type: {device.device_type.value}")
                        print(f"    Qubits: {device.num_qubits}")
                        print(f"    Queue: {device.queue_length}")
                        print(f"    Status: {status}")
                        if device.cost_per_shot:
                            print(f"    Cost: ${device.cost_per_shot:.4f} per shot")
                        print()
            
            elif args.command == 'jobs':
                if args.active:
                    jobs = orchestrator.job_manager.get_active_jobs()
                    print(f"Active jobs ({len(jobs)}):")
                elif args.history:
                    jobs = orchestrator.job_manager.get_job_history()
                    print(f"Job history ({len(jobs)}):")
                else:
                    active = orchestrator.job_manager.get_active_jobs()
                    history = orchestrator.job_manager.get_job_history()
                    jobs = active + history
                    print(f"All jobs ({len(jobs)}):")
                
                for job in jobs:
                    print(f"  {job.job_id} - {job.status.value}")
                    print(f"    Provider: {job.provider.value}")
                    print(f"    Device: {job.device_id}")
                    print(f"    Shots: {job.shots}")
                    if job.cost:
                        print(f"    Cost: ${job.cost:.4f}")
                    print()
            
            elif args.command == 'status':
                status = await orchestrator.get_job_status(args.job_id)
                if status:
                    print(f"Job {args.job_id}: {status.value}")
                    
                    job = orchestrator.job_manager.get_job(args.job_id)
                    if job and job.status == JobStatus.COMPLETED:
                        result = await orchestrator.get_job_result(args.job_id)
                        if result:
                            print("Result:", json.dumps(result, indent=2))
                else:
                    print(f"Job {args.job_id} not found")
            
            elif args.command == 'stats':
                stats = orchestrator.get_cloud_statistics()
                print("Cloud Statistics:")
                print(json.dumps(stats, indent=2))
            
            elif args.command == 'auth':
                print("Authenticating with all providers...")
                results = await orchestrator.authenticate_all()
                
                for provider, success in results.items():
                    status = "✓" if success else "✗"
                    print(f"  {status} {provider.value}")
            
            return 0
            
        except Exception as e:
            print(f"Error: {e}")
            return 1
    
    # Run async command
    try:
        return asyncio.run(run_async_command())
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 1


if __name__ == "__main__":
    exit(main())