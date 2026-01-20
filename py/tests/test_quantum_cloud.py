#!/usr/bin/env python3
"""
Test suite for quantum cloud orchestration functionality.
"""

import pytest
import asyncio
import json
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any

try:
    from quantrs2.quantum_cloud import (
        CloudProvider, JobStatus, DeviceType, OptimizationLevel,
        CloudCredentials, DeviceInfo, CloudJob,
        CloudAdapter, IBMQuantumAdapter, AWSBraketAdapter, 
        GoogleQuantumAIAdapter, LocalAdapter,
        CloudJobManager, QuantumCloudOrchestrator,
        get_quantum_cloud_orchestrator, authenticate_cloud_providers,
        get_available_devices, submit_quantum_job, create_cloud_credentials,
        add_cloud_provider, get_cloud_statistics
    )
    HAS_QUANTUM_CLOUD = True
except ImportError:
    HAS_QUANTUM_CLOUD = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@pytest.mark.skipif(not HAS_QUANTUM_CLOUD, reason="quantum_cloud module not available")
class TestCloudCredentials:
    """Test CloudCredentials functionality."""
    
    def test_credentials_creation(self):
        """Test creating cloud credentials."""
        creds = CloudCredentials(
            provider=CloudProvider.IBM_QUANTUM,
            credentials={'token': 'test_token'},
            endpoint='https://api.quantum-computing.ibm.com',
            region='us-east-1'
        )
        
        assert creds.provider == CloudProvider.IBM_QUANTUM
        assert creds.credentials['token'] == 'test_token'
        assert creds.endpoint == 'https://api.quantum-computing.ibm.com'
        assert creds.region == 'us-east-1'
    
    def test_credentials_serialization(self):
        """Test credentials serialization."""
        creds = CloudCredentials(
            provider=CloudProvider.AWS_BRAKET,
            credentials={
                'access_key_id': 'AKIATEST',
                'secret_access_key': 'secret123'
            },
            region='us-west-2'
        )
        
        creds_dict = creds.to_dict()
        
        assert creds_dict['provider'] == 'aws_braket'
        assert creds_dict['credentials']['access_key_id'] == 'AKIATEST'
        assert creds_dict['region'] == 'us-west-2'
    
    def test_credentials_deserialization(self):
        """Test credentials deserialization."""
        data = {
            'provider': 'google_quantum_ai',
            'credentials': {'service_account_key': 'key_data'},
            'project_id': 'test-project-123'
        }
        
        creds = CloudCredentials.from_dict(data)
        
        assert creds.provider == CloudProvider.GOOGLE_QUANTUM_AI
        assert creds.credentials['service_account_key'] == 'key_data'
        assert creds.project_id == 'test-project-123'


@pytest.mark.skipif(not HAS_QUANTUM_CLOUD, reason="quantum_cloud module not available")
class TestDeviceInfo:
    """Test DeviceInfo functionality."""
    
    def test_device_info_creation(self):
        """Test creating device info."""
        device = DeviceInfo(
            provider=CloudProvider.IBM_QUANTUM,
            device_id='ibm_brisbane',
            device_name='IBM Brisbane',
            device_type=DeviceType.QPU,
            num_qubits=127,
            gate_set=['rx', 'ry', 'rz', 'cx'],
            error_rates={'gate_error': 0.001},
            queue_length=5,
            cost_per_shot=0.001
        )
        
        assert device.provider == CloudProvider.IBM_QUANTUM
        assert device.num_qubits == 127
        assert device.device_type == DeviceType.QPU
        assert device.cost_per_shot == 0.001
        assert 'rx' in device.gate_set
    
    def test_device_info_serialization(self):
        """Test device info serialization."""
        device = DeviceInfo(
            provider=CloudProvider.LOCAL,
            device_id='local_sim',
            device_name='Local Simulator',
            device_type=DeviceType.SIMULATOR,
            num_qubits=20
        )
        
        device_dict = device.to_dict()
        
        assert device_dict['provider'] == 'local'
        assert device_dict['device_type'] == 'simulator'
        assert device_dict['num_qubits'] == 20


@pytest.mark.skipif(not HAS_QUANTUM_CLOUD, reason="quantum_cloud module not available")
class TestCloudJob:
    """Test CloudJob functionality."""
    
    def test_job_creation(self):
        """Test creating cloud job."""
        circuit_data = {
            'gates': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'cnot', 'qubits': [0, 1]}
            ]
        }
        
        job = CloudJob(
            job_id='test_job_123',
            provider=CloudProvider.IBM_QUANTUM,
            device_id='ibm_simulator',
            circuit_data=circuit_data,
            shots=1024
        )
        
        assert job.job_id == 'test_job_123'
        assert job.provider == CloudProvider.IBM_QUANTUM
        assert job.status == JobStatus.PENDING
        assert job.shots == 1024
        assert len(job.circuit_data['gates']) == 2
    
    def test_job_serialization(self):
        """Test job serialization."""
        job = CloudJob(
            job_id='job_456',
            provider=CloudProvider.AWS_BRAKET,
            device_id='LocalSimulator',
            circuit_data={'gates': []},
            shots=512,
            status=JobStatus.COMPLETED,
            result={'counts': {'00': 256, '11': 256}}
        )
        
        job_dict = job.to_dict()
        
        assert job_dict['job_id'] == 'job_456'
        assert job_dict['provider'] == 'aws_braket'
        assert job_dict['status'] == 'completed'
        assert job_dict['result']['counts']['00'] == 256
    
    def test_job_deserialization(self):
        """Test job deserialization."""
        data = {
            'job_id': 'restored_job',
            'provider': 'local',
            'device_id': 'local_sim',
            'circuit_data': {'gates': [{'gate': 'x', 'qubits': [0]}]},
            'shots': 100,
            'status': 'running',
            'created_at': time.time(),
            'started_at': None,
            'completed_at': None,
            'result': None,
            'error_message': None,
            'cost': None,
            'metadata': {}
        }
        
        job = CloudJob.from_dict(data)
        
        assert job.job_id == 'restored_job'
        assert job.provider == CloudProvider.LOCAL
        assert job.status == JobStatus.RUNNING
        assert job.shots == 100


@pytest.mark.skipif(not HAS_QUANTUM_CLOUD, reason="quantum_cloud module not available")
class TestLocalAdapter:
    """Test LocalAdapter functionality."""
    
    def setup_method(self):
        """Set up test adapter."""
        credentials = CloudCredentials(
            provider=CloudProvider.LOCAL,
            credentials={}
        )
        self.adapter = LocalAdapter(credentials)
    
    @pytest.mark.asyncio
    async def test_authentication(self):
        """Test local adapter authentication."""
        success = await self.adapter.authenticate()
        assert success is True
        assert self.adapter._authenticated is True
    
    @pytest.mark.asyncio
    async def test_get_devices(self):
        """Test getting local devices."""
        devices = await self.adapter.get_devices()
        
        assert len(devices) > 0
        device = devices[0]
        assert device.provider == CloudProvider.LOCAL
        assert device.device_type == DeviceType.SIMULATOR
        assert device.num_qubits > 0
        assert device.cost_per_shot == 0.0
    
    @pytest.mark.asyncio
    async def test_submit_job(self):
        """Test submitting job to local adapter."""
        circuit_data = {
            'gates': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'cnot', 'qubits': [0, 1]}
            ]
        }
        
        job = await self.adapter.submit_job('quantrs2_simulator', circuit_data, 1000)
        
        assert job is not None
        assert job.provider == CloudProvider.LOCAL
        assert job.status == JobStatus.COMPLETED  # Local jobs complete immediately
        assert job.shots == 1000
        assert job.result is not None
    
    @pytest.mark.asyncio
    async def test_get_job_status(self):
        """Test getting job status."""
        status = await self.adapter.get_job_status('test_job')
        assert status == JobStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_get_job_result(self):
        """Test getting job result."""
        result = await self.adapter.get_job_result('test_job')
        
        assert result is not None
        assert 'counts' in result
        assert 'execution_time' in result
    
    @pytest.mark.asyncio
    async def test_cancel_job(self):
        """Test canceling job."""
        success = await self.adapter.cancel_job('test_job')
        assert success is True


@pytest.mark.skipif(not HAS_QUANTUM_CLOUD, reason="quantum_cloud module not available")
class TestIBMQuantumAdapter:
    """Test IBMQuantumAdapter functionality."""
    
    def setup_method(self):
        """Set up test adapter."""
        credentials = CloudCredentials(
            provider=CloudProvider.IBM_QUANTUM,
            credentials={'token': 'test_token_123'},
            endpoint='https://api.quantum-computing.ibm.com'
        )
        self.adapter = IBMQuantumAdapter(credentials)
    
    def test_adapter_initialization(self):
        """Test adapter initialization."""
        assert self.adapter.provider == CloudProvider.IBM_QUANTUM
        assert self.adapter.token == 'test_token_123'
        assert self.adapter.hub == 'ibm-q'
        assert self.adapter.group == 'open'
        assert self.adapter.project == 'main'
    
    @pytest.mark.asyncio
    async def test_authentication(self):
        """Test IBM authentication."""
        success = await self.adapter.authenticate()
        assert success is True  # Mock always succeeds
    
    @pytest.mark.asyncio
    async def test_get_devices(self):
        """Test getting IBM devices."""
        devices = await self.adapter.get_devices()
        
        assert len(devices) > 0
        
        # Check for simulator
        simulator = next((d for d in devices if d.device_type == DeviceType.SIMULATOR), None)
        assert simulator is not None
        assert simulator.provider == CloudProvider.IBM_QUANTUM
        
        # Check for QPU
        qpu = next((d for d in devices if d.device_type == DeviceType.QPU), None)
        assert qpu is not None
        assert qpu.cost_per_shot is not None
        assert qpu.error_rates is not None
    
    @pytest.mark.asyncio
    async def test_submit_job(self):
        """Test submitting job to IBM."""
        circuit_data = {'gates': [{'gate': 'h', 'qubits': [0]}]}
        
        job = await self.adapter.submit_job('ibmq_qasm_simulator', circuit_data, 1024)
        
        assert job is not None
        assert job.provider == CloudProvider.IBM_QUANTUM
        assert job.device_id == 'ibmq_qasm_simulator'
        assert job.status == JobStatus.QUEUED
        assert 'ibm_' in job.job_id


@pytest.mark.skipif(not HAS_QUANTUM_CLOUD, reason="quantum_cloud module not available")
class TestAWSBraketAdapter:
    """Test AWSBraketAdapter functionality."""
    
    def setup_method(self):
        """Set up test adapter."""
        credentials = CloudCredentials(
            provider=CloudProvider.AWS_BRAKET,
            credentials={
                'access_key_id': 'AKIATEST',
                'secret_access_key': 'secret123'
            },
            region='us-east-1'
        )
        self.adapter = AWSBraketAdapter(credentials)
    
    def test_adapter_initialization(self):
        """Test adapter initialization."""
        assert self.adapter.provider == CloudProvider.AWS_BRAKET
        assert self.adapter.region == 'us-east-1'
        assert self.adapter.access_key == 'AKIATEST'
        assert self.adapter.secret_key == 'secret123'
    
    @pytest.mark.asyncio
    async def test_get_devices(self):
        """Test getting AWS Braket devices."""
        devices = await self.adapter.get_devices()
        
        assert len(devices) > 0
        
        # Check device variety
        device_types = set(d.device_type for d in devices)
        assert DeviceType.SIMULATOR in device_types
        assert DeviceType.QPU in device_types
        
        # Check for specific providers
        device_names = [d.device_name for d in devices]
        assert any('IonQ' in name for name in device_names)
        assert any('Rigetti' in name for name in device_names)
    
    @pytest.mark.asyncio
    async def test_submit_job(self):
        """Test submitting job to AWS Braket."""
        circuit_data = {'gates': [{'gate': 'x', 'qubits': [0]}]}
        
        job = await self.adapter.submit_job('LocalSimulator', circuit_data, 512)
        
        assert job is not None
        assert job.provider == CloudProvider.AWS_BRAKET
        assert 'braket_' in job.job_id


@pytest.mark.skipif(not HAS_QUANTUM_CLOUD, reason="quantum_cloud module not available")
class TestGoogleQuantumAIAdapter:
    """Test GoogleQuantumAIAdapter functionality."""
    
    def setup_method(self):
        """Set up test adapter."""
        credentials = CloudCredentials(
            provider=CloudProvider.GOOGLE_QUANTUM_AI,
            credentials={'service_account_key': 'key_data'},
            project_id='test-project-123'
        )
        self.adapter = GoogleQuantumAIAdapter(credentials)
    
    def test_adapter_initialization(self):
        """Test adapter initialization."""
        assert self.adapter.provider == CloudProvider.GOOGLE_QUANTUM_AI
        assert self.adapter.project_id == 'test-project-123'
        assert self.adapter.service_account_key == 'key_data'
    
    @pytest.mark.asyncio
    async def test_get_devices(self):
        """Test getting Google Quantum AI devices."""
        devices = await self.adapter.get_devices()
        
        assert len(devices) > 0
        
        # Check for simulator and QPU
        simulators = [d for d in devices if d.device_type == DeviceType.SIMULATOR]
        qpus = [d for d in devices if d.device_type == DeviceType.QPU]
        
        assert len(simulators) > 0
        assert len(qpus) > 0
        
        # Check Google-specific details
        assert any('Cirq' in d.device_name for d in simulators)
        assert any('Rainbow' in d.device_name for d in qpus)


@pytest.mark.skipif(not HAS_QUANTUM_CLOUD, reason="quantum_cloud module not available")
class TestCloudJobManager:
    """Test CloudJobManager functionality."""
    
    def setup_method(self):
        """Set up test job manager."""
        self.job_manager = CloudJobManager()
    
    def test_job_manager_initialization(self):
        """Test job manager initialization."""
        assert len(self.job_manager.jobs) == 0
        assert len(self.job_manager.job_history) == 0
        assert self.job_manager.max_history == 1000
    
    def test_add_and_get_job(self):
        """Test adding and getting jobs."""
        job = CloudJob(
            job_id='test_job',
            provider=CloudProvider.LOCAL,
            device_id='local_sim',
            circuit_data={'gates': []},
            shots=100
        )
        
        self.job_manager.add_job(job)
        
        retrieved_job = self.job_manager.get_job('test_job')
        assert retrieved_job is not None
        assert retrieved_job.job_id == 'test_job'
        
        # Test non-existent job
        missing_job = self.job_manager.get_job('missing_job')
        assert missing_job is None
    
    def test_update_job_status(self):
        """Test updating job status."""
        job = CloudJob(
            job_id='status_test',
            provider=CloudProvider.LOCAL,
            device_id='local_sim',
            circuit_data={'gates': []},
            shots=100
        )
        
        self.job_manager.add_job(job)
        
        # Update to running
        self.job_manager.update_job_status('status_test', JobStatus.RUNNING)
        
        updated_job = self.job_manager.get_job('status_test')
        assert updated_job.status == JobStatus.RUNNING
        assert updated_job.started_at is not None
        
        # Update to completed
        result = {'counts': {'0': 50, '1': 50}}
        self.job_manager.update_job_status('status_test', JobStatus.COMPLETED, result)
        
        # Job should be moved to history
        active_job = self.job_manager.get_job('status_test')
        assert active_job is None
        
        history = self.job_manager.get_job_history()
        assert len(history) == 1
        assert history[0].status == JobStatus.COMPLETED
        assert history[0].result == result
    
    def test_job_statistics(self):
        """Test job statistics calculation."""
        # Add some jobs to history
        for i in range(5):
            job = CloudJob(
                job_id=f'job_{i}',
                provider=CloudProvider.LOCAL,
                device_id='local_sim',
                circuit_data={'gates': []},
                shots=100,
                status=JobStatus.COMPLETED if i < 4 else JobStatus.FAILED
            )
            job.started_at = time.time()
            job.completed_at = time.time() + 1.0
            self.job_manager.job_history.append(job)
        
        # Add active job
        active_job = CloudJob(
            job_id='active_job',
            provider=CloudProvider.IBM_QUANTUM,
            device_id='ibm_sim',
            circuit_data={'gates': []},
            shots=100,
            status=JobStatus.RUNNING
        )
        self.job_manager.add_job(active_job)
        
        stats = self.job_manager.get_job_statistics()
        
        assert stats['total_jobs'] == 6
        assert stats['active_jobs'] == 1
        assert stats['completed_jobs'] == 4
        assert stats['failed_jobs'] == 1
        assert stats['success_rate'] == 0.8  # 4/5
        assert stats['average_execution_time'] == 1.0
        
        # Check provider stats
        assert 'local' in stats['providers']
        assert 'ibm_quantum' in stats['providers']
        assert stats['providers']['local'] == 5
        assert stats['providers']['ibm_quantum'] == 1


@pytest.mark.skipif(not HAS_QUANTUM_CLOUD, reason="quantum_cloud module not available")
class TestQuantumCloudOrchestrator:
    """Test QuantumCloudOrchestrator functionality."""
    
    def setup_method(self):
        """Set up test orchestrator."""
        # Use temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        self.temp_config.close()
        
        self.orchestrator = QuantumCloudOrchestrator(self.temp_config.name)
    
    def teardown_method(self):
        """Clean up test orchestrator."""
        Path(self.temp_config.name).unlink(missing_ok=True)
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        assert isinstance(self.orchestrator.job_manager, CloudJobManager)
        assert len(self.orchestrator.adapters) == 0
        assert self.orchestrator.cache_ttl == 300
    
    def test_add_provider(self):
        """Test adding cloud provider."""
        credentials = CloudCredentials(
            provider=CloudProvider.LOCAL,
            credentials={}
        )
        
        success = self.orchestrator.add_provider(credentials)
        
        assert success is True
        assert CloudProvider.LOCAL in self.orchestrator.adapters
        assert isinstance(self.orchestrator.adapters[CloudProvider.LOCAL], LocalAdapter)
    
    def test_add_multiple_providers(self):
        """Test adding multiple providers."""
        providers = [
            CloudCredentials(CloudProvider.LOCAL, {}),
            CloudCredentials(CloudProvider.IBM_QUANTUM, {'token': 'test'}),
            CloudCredentials(CloudProvider.AWS_BRAKET, {'access_key_id': 'test', 'secret_access_key': 'test'})
        ]
        
        for creds in providers:
            success = self.orchestrator.add_provider(creds)
            assert success is True
        
        assert len(self.orchestrator.adapters) == 3
        assert CloudProvider.LOCAL in self.orchestrator.adapters
        assert CloudProvider.IBM_QUANTUM in self.orchestrator.adapters
        assert CloudProvider.AWS_BRAKET in self.orchestrator.adapters
    
    @pytest.mark.asyncio
    async def test_authenticate_all(self):
        """Test authenticating with all providers."""
        # Add providers
        local_creds = CloudCredentials(CloudProvider.LOCAL, {})
        ibm_creds = CloudCredentials(CloudProvider.IBM_QUANTUM, {'token': 'test'})
        
        self.orchestrator.add_provider(local_creds)
        self.orchestrator.add_provider(ibm_creds)
        
        # Authenticate
        results = await self.orchestrator.authenticate_all()
        
        assert len(results) == 2
        assert results[CloudProvider.LOCAL] is True
        assert results[CloudProvider.IBM_QUANTUM] is True
    
    @pytest.mark.asyncio
    async def test_get_all_devices(self):
        """Test getting devices from all providers."""
        # Add local provider
        local_creds = CloudCredentials(CloudProvider.LOCAL, {})
        self.orchestrator.add_provider(local_creds)
        
        # Get devices
        devices = await self.orchestrator.get_all_devices()
        
        assert CloudProvider.LOCAL in devices
        assert len(devices[CloudProvider.LOCAL]) > 0
        
        # Check caching
        devices2 = await self.orchestrator.get_all_devices(refresh_cache=False)
        assert devices == devices2
    
    def test_find_best_device(self):
        """Test finding best device."""
        # Add mock devices to cache
        devices = [
            DeviceInfo(
                provider=CloudProvider.LOCAL,
                device_id='sim1',
                device_name='Simulator 1',
                device_type=DeviceType.SIMULATOR,
                num_qubits=10,
                queue_length=0,
                cost_per_shot=0.0
            ),
            DeviceInfo(
                provider=CloudProvider.IBM_QUANTUM,
                device_id='qpu1',
                device_name='QPU 1',
                device_type=DeviceType.QPU,
                num_qubits=20,
                queue_length=5,
                cost_per_shot=0.001
            )
        ]
        
        self.orchestrator.device_cache[CloudProvider.LOCAL] = [devices[0]]
        self.orchestrator.device_cache[CloudProvider.IBM_QUANTUM] = [devices[1]]
        
        # Test basic requirements
        requirements = {'min_qubits': 5}
        result = self.orchestrator.find_best_device(requirements)
        
        assert result is not None
        provider, device = result
        assert device.num_qubits >= 5
        
        # Test device type preference
        requirements = {'min_qubits': 5, 'device_type': DeviceType.QPU}
        result = self.orchestrator.find_best_device(requirements)
        
        assert result is not None
        provider, device = result
        assert device.device_type == DeviceType.QPU
        
        # Test cost constraint
        requirements = {'min_qubits': 5, 'max_cost': 0.0005}
        result = self.orchestrator.find_best_device(requirements)
        
        assert result is not None
        provider, device = result
        assert device.cost_per_shot == 0.0  # Should pick simulator
    
    @pytest.mark.asyncio
    async def test_submit_job(self):
        """Test submitting job."""
        # Add local provider
        local_creds = CloudCredentials(CloudProvider.LOCAL, {})
        self.orchestrator.add_provider(local_creds)
        
        circuit_data = {
            'gates': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'cnot', 'qubits': [0, 1]}
            ]
        }
        
        # Submit job
        job = await self.orchestrator.submit_job(
            CloudProvider.LOCAL,
            'quantrs2_simulator',
            circuit_data,
            shots=1000
        )
        
        assert job is not None
        assert job.provider == CloudProvider.LOCAL
        assert job.shots == 1000
        
        # Check job manager
        managed_job = self.orchestrator.job_manager.get_job(job.job_id)
        assert managed_job is not None
    
    @pytest.mark.asyncio
    async def test_submit_job_auto(self):
        """Test submitting job with auto device selection."""
        # Add local provider and populate cache
        local_creds = CloudCredentials(CloudProvider.LOCAL, {})
        self.orchestrator.add_provider(local_creds)
        await self.orchestrator.get_all_devices()
        
        circuit_data = {'gates': [{'gate': 'h', 'qubits': [0]}]}
        requirements = {'min_qubits': 1, 'device_type': DeviceType.SIMULATOR}
        
        job = await self.orchestrator.submit_job_auto(circuit_data, requirements, 500)
        
        assert job is not None
        assert job.shots == 500
    
    @pytest.mark.asyncio
    async def test_job_lifecycle(self):
        """Test complete job lifecycle."""
        # Add provider
        local_creds = CloudCredentials(CloudProvider.LOCAL, {})
        self.orchestrator.add_provider(local_creds)
        
        circuit_data = {'gates': [{'gate': 'x', 'qubits': [0]}]}
        
        # Submit job
        job = await self.orchestrator.submit_job(
            CloudProvider.LOCAL, 'quantrs2_simulator', circuit_data, 100
        )
        
        assert job is not None
        job_id = job.job_id
        
        # Check status
        status = await self.orchestrator.get_job_status(job_id)
        assert status is not None
        
        # Get result
        result = await self.orchestrator.get_job_result(job_id)
        assert result is not None
        assert 'counts' in result
        
        # Cancel job (should work even for completed jobs)
        success = await self.orchestrator.cancel_job(job_id)
        assert success is True
    
    def test_cloud_statistics(self):
        """Test cloud statistics."""
        # Add providers
        local_creds = CloudCredentials(CloudProvider.LOCAL, {})
        self.orchestrator.add_provider(local_creds)
        
        stats = self.orchestrator.get_cloud_statistics()
        
        assert 'job_statistics' in stats
        assert 'provider_info' in stats
        assert 'configured_providers' in stats
        assert 'cache_status' in stats
        
        assert stats['configured_providers'] == 1
        assert 'local' in stats['provider_info']
    
    @pytest.mark.skipif(not YAML_AVAILABLE, reason="YAML not available")
    def test_configuration_save_load(self):
        """Test saving and loading configuration."""
        # Add providers
        local_creds = CloudCredentials(CloudProvider.LOCAL, {})
        ibm_creds = CloudCredentials(
            CloudProvider.IBM_QUANTUM, 
            {'token': 'test_token'}
        )
        
        self.orchestrator.add_provider(local_creds)
        self.orchestrator.add_provider(ibm_creds)
        
        # Save configuration
        self.orchestrator.save_configuration()
        
        # Create new orchestrator with same config file
        new_orchestrator = QuantumCloudOrchestrator(self.temp_config.name)
        
        # Should have loaded providers
        assert len(new_orchestrator.adapters) == 2
        assert CloudProvider.LOCAL in new_orchestrator.adapters
        assert CloudProvider.IBM_QUANTUM in new_orchestrator.adapters


@pytest.mark.skipif(not HAS_QUANTUM_CLOUD, reason="quantum_cloud module not available")
class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_get_quantum_cloud_orchestrator(self):
        """Test getting global orchestrator instance."""
        orchestrator1 = get_quantum_cloud_orchestrator()
        orchestrator2 = get_quantum_cloud_orchestrator()
        
        # Should be singleton
        assert orchestrator1 is orchestrator2
        assert isinstance(orchestrator1, QuantumCloudOrchestrator)
    
    @pytest.mark.asyncio
    async def test_authenticate_cloud_providers(self):
        """Test authenticating cloud providers function."""
        # Add a provider first
        add_cloud_provider(CloudProvider.LOCAL)
        
        results = await authenticate_cloud_providers()
        
        assert isinstance(results, dict)
        assert CloudProvider.LOCAL in results
        assert results[CloudProvider.LOCAL] is True
    
    @pytest.mark.asyncio
    async def test_get_available_devices(self):
        """Test getting available devices function."""
        # Add a provider first
        add_cloud_provider(CloudProvider.LOCAL)
        
        devices = await get_available_devices()
        
        assert isinstance(devices, dict)
        assert CloudProvider.LOCAL in devices
        assert len(devices[CloudProvider.LOCAL]) > 0
    
    @pytest.mark.asyncio
    async def test_submit_quantum_job(self):
        """Test submitting quantum job function."""
        # Add a provider first
        add_cloud_provider(CloudProvider.LOCAL)
        
        # Get devices to populate cache
        await get_available_devices()
        
        circuit_data = {'gates': [{'gate': 'h', 'qubits': [0]}]}
        
        job = await submit_quantum_job(circuit_data, shots=100)
        
        assert job is not None
        assert job.shots == 100
    
    def test_create_cloud_credentials(self):
        """Test creating cloud credentials function."""
        creds = create_cloud_credentials(
            CloudProvider.IBM_QUANTUM,
            token='test_token',
            hub='test_hub'
        )
        
        assert creds.provider == CloudProvider.IBM_QUANTUM
        assert creds.credentials['token'] == 'test_token'
        assert creds.credentials['hub'] == 'test_hub'
    
    def test_add_cloud_provider(self):
        """Test adding cloud provider function."""
        success = add_cloud_provider(
            CloudProvider.LOCAL
        )
        
        assert success is True
        
        # Verify it was added
        orchestrator = get_quantum_cloud_orchestrator()
        assert CloudProvider.LOCAL in orchestrator.adapters
    
    def test_get_cloud_statistics(self):
        """Test getting cloud statistics function."""
        # Add a provider first
        add_cloud_provider(CloudProvider.LOCAL)
        
        stats = get_cloud_statistics()
        
        assert isinstance(stats, dict)
        assert 'job_statistics' in stats
        assert 'provider_info' in stats


@pytest.mark.skipif(not HAS_QUANTUM_CLOUD, reason="quantum_cloud module not available")
class TestCloudIntegration:
    """Test cloud integration scenarios."""
    
    def setup_method(self):
        """Set up integration test."""
        self.orchestrator = QuantumCloudOrchestrator()
    
    @pytest.mark.asyncio
    async def test_multi_provider_workflow(self):
        """Test workflow with multiple providers."""
        # Add multiple providers
        local_creds = CloudCredentials(CloudProvider.LOCAL, {})
        ibm_creds = CloudCredentials(CloudProvider.IBM_QUANTUM, {'token': 'test'})
        aws_creds = CloudCredentials(CloudProvider.AWS_BRAKET, {
            'access_key_id': 'test', 'secret_access_key': 'test'
        })
        
        self.orchestrator.add_provider(local_creds)
        self.orchestrator.add_provider(ibm_creds)
        self.orchestrator.add_provider(aws_creds)
        
        # Authenticate all
        auth_results = await self.orchestrator.authenticate_all()
        assert len(auth_results) == 3
        assert all(auth_results.values())
        
        # Get all devices
        devices = await self.orchestrator.get_all_devices()
        assert len(devices) == 3
        assert all(len(device_list) > 0 for device_list in devices.values())
        
        # Submit jobs to different providers
        circuit_data = {'gates': [{'gate': 'h', 'qubits': [0]}]}
        
        jobs = []
        for provider in [CloudProvider.LOCAL, CloudProvider.IBM_QUANTUM, CloudProvider.AWS_BRAKET]:
            device_list = devices[provider]
            if device_list:
                job = await self.orchestrator.submit_job(
                    provider, device_list[0].device_id, circuit_data, 100
                )
                if job:
                    jobs.append(job)
        
        assert len(jobs) > 0
        
        # Check job statistics
        stats = self.orchestrator.get_cloud_statistics()
        assert stats['job_statistics']['total_jobs'] >= len(jobs)
    
    @pytest.mark.asyncio
    async def test_device_selection_optimization(self):
        """Test device selection optimization."""
        # Add providers
        self.orchestrator.add_provider(CloudCredentials(CloudProvider.LOCAL, {}))
        self.orchestrator.add_provider(CloudCredentials(
            CloudProvider.IBM_QUANTUM, {'token': 'test'}
        ))
        
        # Get devices
        await self.orchestrator.get_all_devices()
        
        # Test different requirement scenarios
        test_cases = [
            {'min_qubits': 2, 'device_type': DeviceType.SIMULATOR},
            {'min_qubits': 50, 'device_type': DeviceType.QPU},
            {'min_qubits': 5, 'max_cost': 0.0},
            {'min_qubits': 10, 'provider_preference': ['local', 'ibm_quantum']}
        ]
        
        for requirements in test_cases:
            result = self.orchestrator.find_best_device(requirements)
            
            if result:
                provider, device = result
                assert device.num_qubits >= requirements.get('min_qubits', 1)
                
                if 'device_type' in requirements:
                    assert device.device_type == requirements['device_type']
                
                if 'max_cost' in requirements and device.cost_per_shot:
                    assert device.cost_per_shot <= requirements['max_cost']
    
    @pytest.mark.asyncio
    async def test_job_management_at_scale(self):
        """Test job management with multiple jobs."""
        # Add local provider
        self.orchestrator.add_provider(CloudCredentials(CloudProvider.LOCAL, {}))
        
        circuit_data = {'gates': [{'gate': 'x', 'qubits': [0]}]}
        
        # Submit multiple jobs
        jobs = []
        for i in range(10):
            job = await self.orchestrator.submit_job(
                CloudProvider.LOCAL, 'quantrs2_simulator', circuit_data, 10 * (i + 1)
            )
            if job:
                jobs.append(job)
        
        assert len(jobs) == 10
        
        # Check all jobs
        for job in jobs:
            status = await self.orchestrator.get_job_status(job.job_id)
            assert status is not None
            
            result = await self.orchestrator.get_job_result(job.job_id)
            assert result is not None
        
        # Check statistics
        stats = self.orchestrator.get_cloud_statistics()
        job_stats = stats['job_statistics']
        
        assert job_stats['total_jobs'] >= 10
        assert job_stats['completed_jobs'] >= 10
        assert job_stats['success_rate'] > 0.0
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery scenarios."""
        # Add local provider
        self.orchestrator.add_provider(CloudCredentials(CloudProvider.LOCAL, {}))
        
        # Test submitting to non-existent provider
        circuit_data = {'gates': [{'gate': 'h', 'qubits': [0]}]}
        
        job = await self.orchestrator.submit_job(
            CloudProvider.AWS_BRAKET,  # Not configured
            'fake_device',
            circuit_data,
            100
        )
        
        assert job is None  # Should fail gracefully
        
        # Test getting status for non-existent job
        status = await self.orchestrator.get_job_status('non_existent_job')
        assert status is None
        
        # Test getting result for non-existent job
        result = await self.orchestrator.get_job_result('non_existent_job')
        assert result is None
        
        # Test canceling non-existent job
        success = await self.orchestrator.cancel_job('non_existent_job')
        assert success is False
    
    def test_configuration_management(self):
        """Test configuration management."""
        # Test with missing configuration file
        orchestrator = QuantumCloudOrchestrator('/non/existent/path/config.yaml')
        assert len(orchestrator.adapters) == 0
        
        # Test adding providers programmatically
        local_creds = CloudCredentials(CloudProvider.LOCAL, {})
        success = orchestrator.add_provider(local_creds)
        assert success is True
        
        # Test statistics with configuration
        stats = orchestrator.get_cloud_statistics()
        assert stats['configured_providers'] == 1


@pytest.mark.skipif(not HAS_QUANTUM_CLOUD, reason="quantum_cloud module not available")
class TestCloudPerformance:
    """Test cloud orchestration performance characteristics."""
    
    def setup_method(self):
        """Set up performance test."""
        self.orchestrator = QuantumCloudOrchestrator()
        self.orchestrator.add_provider(CloudCredentials(CloudProvider.LOCAL, {}))
    
    @pytest.mark.asyncio
    async def test_device_caching_performance(self):
        """Test device caching performance."""
        # First call should populate cache
        start_time = time.time()
        devices1 = await self.orchestrator.get_all_devices()
        first_call_time = time.time() - start_time
        
        # Second call should use cache
        start_time = time.time()
        devices2 = await self.orchestrator.get_all_devices(refresh_cache=False)
        cached_call_time = time.time() - start_time
        
        # Cached call should be much faster
        assert cached_call_time < first_call_time
        assert devices1 == devices2
    
    @pytest.mark.asyncio
    async def test_concurrent_job_submission(self):
        """Test concurrent job submission."""
        circuit_data = {'gates': [{'gate': 'h', 'qubits': [0]}]}
        
        # Submit multiple jobs concurrently
        async def submit_job(i):
            return await self.orchestrator.submit_job(
                CloudProvider.LOCAL, 'quantrs2_simulator', circuit_data, 10
            )
        
        start_time = time.time()
        
        # Submit 20 jobs concurrently
        tasks = [submit_job(i) for i in range(20)]
        jobs = await asyncio.gather(*tasks)
        
        submission_time = time.time() - start_time
        
        # All jobs should succeed
        successful_jobs = [j for j in jobs if j is not None]
        assert len(successful_jobs) == 20
        
        # Should complete reasonably quickly
        assert submission_time < 5.0  # 5 seconds
    
    @pytest.mark.asyncio
    async def test_job_status_checking_performance(self):
        """Test job status checking performance."""
        # Submit some jobs
        circuit_data = {'gates': [{'gate': 'x', 'qubits': [0]}]}
        jobs = []
        
        for i in range(10):
            job = await self.orchestrator.submit_job(
                CloudProvider.LOCAL, 'quantrs2_simulator', circuit_data, 10
            )
            if job:
                jobs.append(job)
        
        # Check status of all jobs
        start_time = time.time()
        
        for job in jobs:
            status = await self.orchestrator.get_job_status(job.job_id)
            assert status is not None
        
        status_check_time = time.time() - start_time
        
        # Should be fast
        assert status_check_time < 2.0  # 2 seconds for 10 jobs
    
    def test_device_selection_performance(self):
        """Test device selection performance."""
        # Populate cache with many devices
        mock_devices = []
        for i in range(100):
            device = DeviceInfo(
                provider=CloudProvider.LOCAL,
                device_id=f'device_{i}',
                device_name=f'Device {i}',
                device_type=DeviceType.SIMULATOR if i % 2 == 0 else DeviceType.QPU,
                num_qubits=10 + i % 50,
                queue_length=i % 20,
                cost_per_shot=0.001 * (i % 10) if i % 2 == 1 else 0.0
            )
            mock_devices.append(device)
        
        self.orchestrator.device_cache[CloudProvider.LOCAL] = mock_devices
        
        # Test device selection performance
        requirements = {'min_qubits': 20, 'device_type': DeviceType.QPU}
        
        start_time = time.time()
        
        for _ in range(100):
            result = self.orchestrator.find_best_device(requirements)
            assert result is not None
        
        selection_time = time.time() - start_time
        
        # Should be fast even with many devices
        assert selection_time < 1.0  # 1 second for 100 selections
    
    def test_memory_usage(self):
        """Test memory usage doesn't grow excessively."""
        import gc
        
        # Add many jobs to history
        for i in range(100):
            job = CloudJob(
                job_id=f'memory_test_{i}',
                provider=CloudProvider.LOCAL,
                device_id='local_sim',
                circuit_data={'gates': []},
                shots=10,
                status=JobStatus.COMPLETED
            )
            self.orchestrator.job_manager.job_history.append(job)
        
        # Force garbage collection
        gc.collect()
        
        # Check history limit
        history = self.orchestrator.job_manager.get_job_history()
        assert len(history) <= self.orchestrator.job_manager.max_history
        
        # Should not crash or run out of memory
        stats = self.orchestrator.get_cloud_statistics()
        assert 'job_statistics' in stats


if __name__ == "__main__":
    pytest.main([__file__])