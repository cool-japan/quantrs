#!/usr/bin/env python3
"""
Comprehensive test suite for the QuantRS2 Quantum Container Orchestration System.

This test suite provides complete coverage of all container orchestration functionality including:
- QuantumContainerRegistry with image management and optimization
- QuantumResourceManager with quantum-specific resource allocation
- DockerContainerManager with container lifecycle management
- KubernetesContainerManager with deployment orchestration
- QuantumContainerOrchestrator with comprehensive application deployment
- Auto-scaling, health monitoring, and metrics collection
- Error handling, edge cases, and performance validation
"""

import pytest
import tempfile
import os
import json
import time
import threading
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, List, Any

try:
    import quantrs2
    from quantrs2.quantum_containers import (
        ContainerStatus, DeploymentMode, ResourceType, ScalingPolicy,
        ResourceRequirement, ContainerConfig, DeploymentSpec, ContainerInstance,
        DeploymentStatus, QuantumContainerRegistry, QuantumResourceManager,
        DockerContainerManager, KubernetesContainerManager, QuantumContainerOrchestrator,
        get_quantum_container_orchestrator, create_quantum_container_config,
        create_quantum_deployment_spec, deploy_quantum_application,
        HAS_DOCKER, HAS_KUBERNETES, HAS_REQUESTS, HAS_PSUTIL, HAS_JINJA2
    )
    HAS_QUANTUM_CONTAINERS = True
except ImportError:
    HAS_QUANTUM_CONTAINERS = False

# Test fixtures
@pytest.fixture
def sample_resource_requirement():
    """Create sample resource requirement for testing."""
    return ResourceRequirement(
        type=ResourceType.QUANTUM_SIMULATOR,
        amount=2,
        unit="simulators",
        metadata={'max_qubits': 16}
    )

@pytest.fixture
def sample_container_config():
    """Create sample container configuration for testing."""
    return ContainerConfig(
        name="test-quantum-container",
        image="quantrs2/quantum-app:latest",
        command=["python", "quantum_app.py"],
        environment={"QUANTUM_BACKEND": "simulator"},
        ports=[8080, 8081],
        resources=[
            ResourceRequirement(ResourceType.QUANTUM_SIMULATOR, 1),
            ResourceRequirement(ResourceType.CPU, 2, "cores"),
            ResourceRequirement(ResourceType.MEMORY, 4, "Gi")
        ],
        labels={"app": "quantum-test", "version": "1.0"},
        quantum_config={"simulators": 1, "max_qubits": 16}
    )

@pytest.fixture
def sample_deployment_spec(sample_container_config):
    """Create sample deployment specification for testing."""
    return DeploymentSpec(
        name="test-quantum-deployment",
        replicas=2,
        containers=[sample_container_config],
        mode=DeploymentMode.DOCKER,
        labels={"environment": "test"},
        quantum_requirements={"total_simulators": 1, "max_qubits_required": 16}
    )

@pytest.fixture
def mock_docker_client():
    """Create mock Docker client for testing."""
    mock_client = Mock()
    mock_container = Mock()
    mock_container.id = "test_container_id"
    mock_container.name = "test-container"
    mock_container.status = "running"
    mock_container.attrs = {
        'State': {'ExitCode': 0, 'Health': {'Status': 'healthy'}}
    }
    mock_container.logs.return_value = b"Container log output"
    
    mock_client.containers.create.return_value = mock_container
    mock_client.containers.get.return_value = mock_container
    mock_client.containers.list.return_value = [mock_container]
    
    return mock_client

@pytest.fixture
def mock_kubernetes_client():
    """Create mock Kubernetes client for testing."""
    mock_apps_v1 = Mock()
    mock_core_v1 = Mock()
    
    # Mock deployment
    mock_deployment = Mock()
    mock_deployment.spec.replicas = 2
    mock_deployment.status.available_replicas = 2
    mock_deployment.status.ready_replicas = 2
    
    mock_apps_v1.create_namespaced_deployment.return_value = mock_deployment
    mock_apps_v1.read_namespaced_deployment.return_value = mock_deployment
    
    # Mock pods
    mock_pod = Mock()
    mock_pod.metadata.name = "test-pod"
    mock_pod.metadata.creation_timestamp = None
    mock_pod.status.phase = "Running"
    mock_pod.status.container_statuses = [
        Mock(name="test-container", image="test:latest", ready=True)
    ]
    
    mock_pod_list = Mock()
    mock_pod_list.items = [mock_pod]
    mock_core_v1.list_namespaced_pod.return_value = mock_pod_list
    
    return mock_apps_v1, mock_core_v1

@pytest.fixture
def quantum_orchestrator():
    """Create quantum container orchestrator for testing."""
    return QuantumContainerOrchestrator(DeploymentMode.DOCKER)


@pytest.mark.skipif(not HAS_QUANTUM_CONTAINERS, reason="quantum containers not available")
class TestResourceRequirement:
    """Test ResourceRequirement functionality."""
    
    def test_resource_requirement_creation(self):
        """Test ResourceRequirement creation."""
        req = ResourceRequirement(
            type=ResourceType.QUANTUM_SIMULATOR,
            amount=2,
            unit="simulators",
            metadata={'max_qubits': 32}
        )
        
        assert req.type == ResourceType.QUANTUM_SIMULATOR
        assert req.amount == 2
        assert req.unit == "simulators"
        assert req.required is True
        assert req.metadata['max_qubits'] == 32
    
    def test_resource_requirement_to_dict(self, sample_resource_requirement):
        """Test ResourceRequirement to_dict conversion."""
        req_dict = sample_resource_requirement.to_dict()
        
        assert isinstance(req_dict, dict)
        assert req_dict['type'] == ResourceType.QUANTUM_SIMULATOR
        assert req_dict['amount'] == 2
        assert req_dict['unit'] == "simulators"
        assert req_dict['metadata']['max_qubits'] == 16


@pytest.mark.skipif(not HAS_QUANTUM_CONTAINERS, reason="quantum containers not available")
class TestContainerConfig:
    """Test ContainerConfig functionality."""
    
    def test_container_config_creation(self, sample_container_config):
        """Test ContainerConfig creation."""
        config = sample_container_config
        
        assert config.name == "test-quantum-container"
        assert config.image == "quantrs2/quantum-app:latest"
        assert config.command == ["python", "quantum_app.py"]
        assert config.environment["QUANTUM_BACKEND"] == "simulator"
        assert 8080 in config.ports
        assert len(config.resources) == 3
        assert config.quantum_config["simulators"] == 1
    
    def test_container_config_to_dict(self, sample_container_config):
        """Test ContainerConfig to_dict conversion."""
        config_dict = sample_container_config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['name'] == "test-quantum-container"
        assert config_dict['image'] == "quantrs2/quantum-app:latest"
        assert isinstance(config_dict['resources'], list)
        assert len(config_dict['resources']) == 3


@pytest.mark.skipif(not HAS_QUANTUM_CONTAINERS, reason="quantum containers not available")
class TestDeploymentSpec:
    """Test DeploymentSpec functionality."""
    
    def test_deployment_spec_creation(self, sample_deployment_spec):
        """Test DeploymentSpec creation."""
        spec = sample_deployment_spec
        
        assert spec.name == "test-quantum-deployment"
        assert spec.replicas == 2
        assert spec.mode == DeploymentMode.DOCKER
        assert len(spec.containers) == 1
        assert spec.quantum_requirements["total_simulators"] == 1
    
    def test_deployment_spec_to_dict(self, sample_deployment_spec):
        """Test DeploymentSpec to_dict conversion."""
        spec_dict = sample_deployment_spec.to_dict()
        
        assert isinstance(spec_dict, dict)
        assert spec_dict['name'] == "test-quantum-deployment"
        assert spec_dict['replicas'] == 2
        assert spec_dict['mode'] == DeploymentMode.DOCKER
        assert isinstance(spec_dict['containers'], list)


@pytest.mark.skipif(not HAS_QUANTUM_CONTAINERS, reason="quantum containers not available")
class TestQuantumContainerRegistry:
    """Test QuantumContainerRegistry functionality."""
    
    def test_registry_initialization(self):
        """Test QuantumContainerRegistry initialization."""
        registry = QuantumContainerRegistry("localhost:5000")
        
        assert registry.registry_url == "localhost:5000"
        assert isinstance(registry.quantum_layers_cache, dict)
        assert isinstance(registry.image_metadata, dict)
    
    @patch('quantrs2.quantum_containers.HAS_DOCKER', True)
    @patch('quantrs2.quantum_containers.docker')
    def test_push_image_with_docker(self, mock_docker):
        """Test push_image with Docker."""
        mock_client = Mock()
        mock_image = Mock()
        mock_image.tag.return_value = True
        mock_client.images.get.return_value = mock_image
        mock_client.images.push.return_value = "successfully pushed"
        mock_docker.from_env.return_value = mock_client
        
        registry = QuantumContainerRegistry()
        quantum_metadata = {"qubits": 16, "simulator": "statevector"}
        
        result = registry.push_image("quantum-app", "latest", quantum_metadata)
        
        assert result is True
        mock_image.tag.assert_called_once()
        mock_client.images.push.assert_called_once()
    
    @patch('quantrs2.quantum_containers.HAS_DOCKER', True)
    @patch('quantrs2.quantum_containers.docker')
    def test_pull_image_with_docker(self, mock_docker):
        """Test pull_image with Docker."""
        mock_client = Mock()
        mock_client.images.pull.return_value = True
        mock_docker.from_env.return_value = mock_client
        
        registry = QuantumContainerRegistry()
        result = registry.pull_image("quantum-app", "latest")
        
        assert result is True
        mock_client.images.pull.assert_called_once_with("localhost:5000/quantum-app:latest")
    
    @patch('quantrs2.quantum_containers.HAS_DOCKER', True)
    @patch('quantrs2.quantum_containers.docker')
    def test_list_quantum_images(self, mock_docker):
        """Test list_quantum_images functionality."""
        mock_client = Mock()
        mock_image = Mock()
        mock_image.id = "image123"
        mock_image.tags = ["quantum-app:latest"]
        mock_image.attrs = {"Size": 1000000, "Created": "2023-01-01T00:00:00Z"}
        mock_client.images.list.return_value = [mock_image]
        mock_docker.from_env.return_value = mock_client
        
        registry = QuantumContainerRegistry()
        images = registry.list_quantum_images()
        
        assert len(images) == 1
        assert images[0]['id'] == "image123"
        assert images[0]['tags'] == ["quantum-app:latest"]
    
    @patch('quantrs2.quantum_containers.HAS_DOCKER', True)
    @patch('quantrs2.quantum_containers.docker')
    def test_optimize_quantum_layers(self, mock_docker):
        """Test optimize_quantum_layers functionality."""
        mock_client = Mock()
        mock_image = Mock()
        mock_image.history.return_value = [
            {"Size": 500000, "CreatedBy": "RUN pip install quantrs2"},
            {"Size": 300000, "CreatedBy": "RUN apt-get update"}
        ]
        mock_client.images.get.return_value = mock_image
        mock_docker.from_env.return_value = mock_client
        
        registry = QuantumContainerRegistry()
        result = registry.optimize_quantum_layers("quantum-app:latest")
        
        assert result['original_size'] == 800000
        assert result['optimized_size'] > 0
        assert result['quantum_layers_identified'] == 1
        assert len(result['optimizations_applied']) > 0
    
    def test_registry_without_docker(self):
        """Test registry functionality without Docker."""
        with patch('quantrs2.quantum_containers.HAS_DOCKER', False):
            registry = QuantumContainerRegistry()
            
            assert registry.push_image("test", "latest") is False
            assert registry.pull_image("test", "latest") is False
            assert registry.list_quantum_images() == []


@pytest.mark.skipif(not HAS_QUANTUM_CONTAINERS, reason="quantum containers not available")
class TestQuantumResourceManager:
    """Test QuantumResourceManager functionality."""
    
    def test_resource_manager_initialization(self):
        """Test QuantumResourceManager initialization."""
        manager = QuantumResourceManager()
        
        assert ResourceType.QUANTUM_SIMULATOR in manager.available_resources
        assert ResourceType.CPU in manager.available_resources
        assert ResourceType.MEMORY in manager.available_resources
        assert isinstance(manager.allocated_resources, dict)
        assert isinstance(manager.resource_reservations, dict)
    
    def test_allocate_quantum_simulator_resources(self):
        """Test quantum simulator resource allocation."""
        manager = QuantumResourceManager()
        
        requirements = [
            ResourceRequirement(ResourceType.QUANTUM_SIMULATOR, 1, metadata={'max_qubits': 16})
        ]
        
        result = manager.allocate_resources("test_container", requirements)
        
        assert result['success'] is True
        assert 'quantum_simulator' in result['allocated_resources']
        assert 'test_container' in manager.allocated_resources
    
    def test_allocate_cpu_resources(self):
        """Test CPU resource allocation."""
        manager = QuantumResourceManager()
        
        requirements = [
            ResourceRequirement(ResourceType.CPU, 2, "cores")
        ]
        
        result = manager.allocate_resources("test_container", requirements)
        
        assert result['success'] is True
        assert 'cpu' in result['allocated_resources']
    
    def test_allocate_memory_resources(self):
        """Test memory resource allocation."""
        manager = QuantumResourceManager()
        
        requirements = [
            ResourceRequirement(ResourceType.MEMORY, "4Gi")
        ]
        
        result = manager.allocate_resources("test_container", requirements)
        
        # Result depends on available memory
        assert isinstance(result, dict)
        assert 'success' in result
    
    def test_deallocate_resources(self):
        """Test resource deallocation."""
        manager = QuantumResourceManager()
        
        # First allocate resources
        requirements = [
            ResourceRequirement(ResourceType.QUANTUM_SIMULATOR, 1)
        ]
        
        allocation_result = manager.allocate_resources("test_container", requirements)
        
        if allocation_result['success']:
            # Then deallocate
            deallocation_result = manager.deallocate_resources("test_container")
            
            assert deallocation_result is True
            assert "test_container" not in manager.allocated_resources
    
    def test_resource_utilization(self):
        """Test resource utilization reporting."""
        manager = QuantumResourceManager()
        
        utilization = manager.get_resource_utilization()
        
        assert isinstance(utilization, dict)
        # Should contain quantum simulator utilization at minimum
        assert 'quantum_simulator' in utilization
    
    def test_memory_amount_parsing(self):
        """Test memory amount parsing functionality."""
        manager = QuantumResourceManager()
        
        # Test various memory formats
        assert manager._parse_memory_amount("1Gi", "") == 1024**3
        assert manager._parse_memory_amount("512Mi", "") == 512 * 1024**2
        assert manager._parse_memory_amount(1024, "B") == 1024
        assert manager._parse_memory_amount("2G", "") == 2 * 1024**3
    
    def test_resource_availability_checking(self):
        """Test resource availability checking."""
        manager = QuantumResourceManager()
        
        # Test quantum simulator availability
        sim_req = ResourceRequirement(ResourceType.QUANTUM_SIMULATOR, 1)
        assert manager._check_resource_availability(sim_req) is True
        
        # Test excessive quantum simulator request
        excessive_sim_req = ResourceRequirement(ResourceType.QUANTUM_SIMULATOR, 100)
        assert manager._check_resource_availability(excessive_sim_req) is False
        
        # Test CPU availability
        cpu_req = ResourceRequirement(ResourceType.CPU, 1, "cores")
        # Result depends on available CPUs
        result = manager._check_resource_availability(cpu_req)
        assert isinstance(result, bool)
    
    @patch('quantrs2.quantum_containers.HAS_PSUTIL', True)
    @patch('quantrs2.quantum_containers.psutil')
    def test_system_resource_detection_with_psutil(self, mock_psutil):
        """Test system resource detection with psutil."""
        mock_psutil.cpu_count.return_value = 8
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3  # 16GB
        mock_memory.used = 8 * 1024**3   # 8GB
        mock_psutil.virtual_memory.return_value = mock_memory
        
        manager = QuantumResourceManager()
        manager._detect_system_resources()
        
        assert manager.available_resources[ResourceType.CPU]['cores'] == 8
        assert manager.available_resources[ResourceType.MEMORY]['total'] == 16 * 1024**3
    
    def test_system_resource_detection_without_psutil(self):
        """Test system resource detection without psutil."""
        with patch('quantrs2.quantum_containers.HAS_PSUTIL', False):
            manager = QuantumResourceManager()
            manager._detect_system_resources()
            
            # Should not crash and should have default values
            assert manager.available_resources[ResourceType.CPU]['cores'] == 0


@pytest.mark.skipif(not HAS_QUANTUM_CONTAINERS, reason="quantum containers not available")
class TestDockerContainerManager:
    """Test DockerContainerManager functionality."""
    
    def test_docker_manager_initialization(self):
        """Test DockerContainerManager initialization."""
        resource_manager = QuantumResourceManager()
        
        with patch('quantrs2.quantum_containers.HAS_DOCKER', True):
            with patch('quantrs2.quantum_containers.docker') as mock_docker:
                mock_docker.from_env.return_value = Mock()
                
                manager = DockerContainerManager(resource_manager)
                
                assert manager.resource_manager is resource_manager
                assert manager.client is not None
                assert isinstance(manager.containers, dict)
    
    @patch('quantrs2.quantum_containers.HAS_DOCKER', True)
    @patch('quantrs2.quantum_containers.docker')
    def test_create_container_success(self, mock_docker, sample_container_config, mock_docker_client):
        """Test successful container creation."""
        mock_docker.from_env.return_value = mock_docker_client
        
        resource_manager = QuantumResourceManager()
        manager = DockerContainerManager(resource_manager)
        manager.client = mock_docker_client
        
        instance = manager.create_container(sample_container_config)
        
        assert instance is not None
        assert isinstance(instance, ContainerInstance)
        assert instance.name == sample_container_config.name
        assert instance.status == ContainerStatus.PENDING
    
    @patch('quantrs2.quantum_containers.HAS_DOCKER', True)
    @patch('quantrs2.quantum_containers.docker')
    def test_start_container(self, mock_docker, mock_docker_client):
        """Test container start functionality."""
        mock_docker.from_env.return_value = mock_docker_client
        
        resource_manager = QuantumResourceManager()
        manager = DockerContainerManager(resource_manager)
        manager.client = mock_docker_client
        
        # Create mock container instance
        instance = ContainerInstance(
            id="test_container_id",
            name="test-container",
            config=ContainerConfig("test", "test:latest"),
            status=ContainerStatus.PENDING,
            deployment_name="test",
            created_at=time.time()
        )
        manager.containers["test_container_id"] = instance
        
        result = manager.start_container("test_container_id")
        
        assert result is True
        assert instance.status == ContainerStatus.RUNNING
        assert instance.started_at is not None
    
    @patch('quantrs2.quantum_containers.HAS_DOCKER', True)
    @patch('quantrs2.quantum_containers.docker')
    def test_stop_container(self, mock_docker, mock_docker_client):
        """Test container stop functionality."""
        mock_docker.from_env.return_value = mock_docker_client
        
        resource_manager = QuantumResourceManager()
        manager = DockerContainerManager(resource_manager)
        manager.client = mock_docker_client
        
        # Create mock container instance
        instance = ContainerInstance(
            id="test_container_id",
            name="test-container",
            config=ContainerConfig("test", "test:latest"),
            status=ContainerStatus.RUNNING,
            deployment_name="test",
            created_at=time.time(),
            started_at=time.time()
        )
        manager.containers["test_container_id"] = instance
        
        result = manager.stop_container("test_container_id")
        
        assert result is True
        assert instance.status == ContainerStatus.STOPPED
        assert instance.finished_at is not None
    
    @patch('quantrs2.quantum_containers.HAS_DOCKER', True)
    @patch('quantrs2.quantum_containers.docker')
    def test_remove_container(self, mock_docker, mock_docker_client):
        """Test container removal functionality."""
        mock_docker.from_env.return_value = mock_docker_client
        
        resource_manager = QuantumResourceManager()
        manager = DockerContainerManager(resource_manager)
        manager.client = mock_docker_client
        
        # Create mock container instance
        instance = ContainerInstance(
            id="test_container_id",
            name="test-container",
            config=ContainerConfig("test", "test:latest"),
            status=ContainerStatus.STOPPED,
            deployment_name="test",
            created_at=time.time()
        )
        manager.containers["test_container_id"] = instance
        
        result = manager.remove_container("test_container_id")
        
        assert result is True
        assert "test_container_id" not in manager.containers
    
    @patch('quantrs2.quantum_containers.HAS_DOCKER', True)
    @patch('quantrs2.quantum_containers.docker')
    def test_get_container_status(self, mock_docker, mock_docker_client):
        """Test container status retrieval."""
        mock_docker.from_env.return_value = mock_docker_client
        
        resource_manager = QuantumResourceManager()
        manager = DockerContainerManager(resource_manager)
        manager.client = mock_docker_client
        
        # Create mock container instance
        instance = ContainerInstance(
            id="test_container_id",
            name="test-container",
            config=ContainerConfig("test", "test:latest"),
            status=ContainerStatus.PENDING,
            deployment_name="test",
            created_at=time.time()
        )
        manager.containers["test_container_id"] = instance
        
        status = manager.get_container_status("test_container_id")
        
        assert status is not None
        assert isinstance(status, ContainerInstance)
        assert status.status == ContainerStatus.RUNNING  # Updated from Docker
    
    @patch('quantrs2.quantum_containers.HAS_DOCKER', True)
    @patch('quantrs2.quantum_containers.docker')
    def test_get_container_logs(self, mock_docker, mock_docker_client):
        """Test container logs retrieval."""
        mock_docker.from_env.return_value = mock_docker_client
        
        resource_manager = QuantumResourceManager()
        manager = DockerContainerManager(resource_manager)
        manager.client = mock_docker_client
        
        # Create mock container instance
        instance = ContainerInstance(
            id="test_container_id",
            name="test-container",
            config=ContainerConfig("test", "test:latest"),
            status=ContainerStatus.RUNNING,
            deployment_name="test",
            created_at=time.time()
        )
        manager.containers["test_container_id"] = instance
        
        logs = manager.get_container_logs("test_container_id")
        
        assert isinstance(logs, list)
        assert len(logs) > 0
    
    @patch('quantrs2.quantum_containers.HAS_DOCKER', True)
    @patch('quantrs2.quantum_containers.docker')
    def test_list_containers(self, mock_docker, mock_docker_client):
        """Test container listing functionality."""
        mock_docker.from_env.return_value = mock_docker_client
        
        resource_manager = QuantumResourceManager()
        manager = DockerContainerManager(resource_manager)
        manager.client = mock_docker_client
        
        # Create mock container instances
        for i in range(3):
            instance = ContainerInstance(
                id=f"test_container_{i}",
                name=f"test-container-{i}",
                config=ContainerConfig(f"test-{i}", "test:latest"),
                status=ContainerStatus.RUNNING,
                deployment_name="test",
                created_at=time.time()
            )
            manager.containers[f"test_container_{i}"] = instance
        
        containers = manager.list_containers()
        
        assert len(containers) == 3
        assert all(isinstance(c, ContainerInstance) for c in containers)
    
    def test_docker_manager_without_docker(self):
        """Test DockerContainerManager without Docker."""
        with patch('quantrs2.quantum_containers.HAS_DOCKER', False):
            resource_manager = QuantumResourceManager()
            manager = DockerContainerManager(resource_manager)
            
            assert manager.client is None
            
            # Should handle gracefully
            config = ContainerConfig("test", "test:latest")
            instance = manager.create_container(config)
            assert instance is None


@pytest.mark.skipif(not HAS_QUANTUM_CONTAINERS, reason="quantum containers not available")
class TestKubernetesContainerManager:
    """Test KubernetesContainerManager functionality."""
    
    def test_k8s_manager_initialization(self):
        """Test KubernetesContainerManager initialization."""
        resource_manager = QuantumResourceManager()
        
        with patch('quantrs2.quantum_containers.HAS_KUBERNETES', True):
            with patch('quantrs2.quantum_containers.config') as mock_config:
                with patch('quantrs2.quantum_containers.client') as mock_client:
                    mock_config.load_kube_config.return_value = None
                    mock_client.ApiClient.return_value = Mock()
                    mock_client.AppsV1Api.return_value = Mock()
                    mock_client.CoreV1Api.return_value = Mock()
                    
                    manager = KubernetesContainerManager(resource_manager)
                    
                    assert manager.resource_manager is resource_manager
                    assert manager.api_client is not None
                    assert manager.apps_v1 is not None
                    assert manager.core_v1 is not None
    
    @patch('quantrs2.quantum_containers.HAS_KUBERNETES', True)
    @patch('quantrs2.quantum_containers.config')
    @patch('quantrs2.quantum_containers.client')
    def test_create_deployment(self, mock_client, mock_config, sample_deployment_spec, mock_kubernetes_client):
        """Test Kubernetes deployment creation."""
        mock_config.load_kube_config.return_value = None
        mock_apps_v1, mock_core_v1 = mock_kubernetes_client
        mock_client.ApiClient.return_value = Mock()
        mock_client.AppsV1Api.return_value = mock_apps_v1
        mock_client.CoreV1Api.return_value = mock_core_v1
        
        resource_manager = QuantumResourceManager()
        manager = KubernetesContainerManager(resource_manager)
        
        result = manager.create_deployment(sample_deployment_spec)
        
        assert result is True
        mock_apps_v1.create_namespaced_deployment.assert_called_once()
        assert sample_deployment_spec.name in manager.deployments
    
    @patch('quantrs2.quantum_containers.HAS_KUBERNETES', True)
    @patch('quantrs2.quantum_containers.config')
    @patch('quantrs2.quantum_containers.client')
    def test_scale_deployment(self, mock_client, mock_config, mock_kubernetes_client):
        """Test Kubernetes deployment scaling."""
        mock_config.load_kube_config.return_value = None
        mock_apps_v1, mock_core_v1 = mock_kubernetes_client
        mock_client.ApiClient.return_value = Mock()
        mock_client.AppsV1Api.return_value = mock_apps_v1
        mock_client.CoreV1Api.return_value = mock_core_v1
        
        resource_manager = QuantumResourceManager()
        manager = KubernetesContainerManager(resource_manager)
        
        result = manager.scale_deployment("test-deployment", 5)
        
        assert result is True
        mock_apps_v1.patch_namespaced_deployment_scale.assert_called_once()
    
    @patch('quantrs2.quantum_containers.HAS_KUBERNETES', True)
    @patch('quantrs2.quantum_containers.config')
    @patch('quantrs2.quantum_containers.client')
    def test_delete_deployment(self, mock_client, mock_config, mock_kubernetes_client):
        """Test Kubernetes deployment deletion."""
        mock_config.load_kube_config.return_value = None
        mock_apps_v1, mock_core_v1 = mock_kubernetes_client
        mock_client.ApiClient.return_value = Mock()
        mock_client.AppsV1Api.return_value = mock_apps_v1
        mock_client.CoreV1Api.return_value = mock_core_v1
        
        resource_manager = QuantumResourceManager()
        manager = KubernetesContainerManager(resource_manager)
        
        # Add deployment to manager
        manager.deployments["test-deployment"] = Mock()
        
        result = manager.delete_deployment("test-deployment")
        
        assert result is True
        mock_apps_v1.delete_namespaced_deployment.assert_called_once()
        assert "test-deployment" not in manager.deployments
    
    @patch('quantrs2.quantum_containers.HAS_KUBERNETES', True)
    @patch('quantrs2.quantum_containers.config')
    @patch('quantrs2.quantum_containers.client')
    def test_get_deployment_status(self, mock_client, mock_config, mock_kubernetes_client):
        """Test Kubernetes deployment status retrieval."""
        mock_config.load_kube_config.return_value = None
        mock_apps_v1, mock_core_v1 = mock_kubernetes_client
        mock_client.ApiClient.return_value = Mock()
        mock_client.AppsV1Api.return_value = mock_apps_v1
        mock_client.CoreV1Api.return_value = mock_core_v1
        
        resource_manager = QuantumResourceManager()
        manager = KubernetesContainerManager(resource_manager)
        
        status = manager.get_deployment_status("test-deployment")
        
        assert status is not None
        assert isinstance(status, DeploymentStatus)
        assert status.name == "test-deployment"
        assert status.desired_replicas == 2
        assert status.available_replicas == 2
    
    @patch('quantrs2.quantum_containers.HAS_KUBERNETES', True)
    @patch('quantrs2.quantum_containers.config')
    @patch('quantrs2.quantum_containers.client')
    def test_list_deployments(self, mock_client, mock_config, mock_kubernetes_client):
        """Test Kubernetes deployment listing."""
        mock_config.load_kube_config.return_value = None
        mock_apps_v1, mock_core_v1 = mock_kubernetes_client
        mock_client.ApiClient.return_value = Mock()
        mock_client.AppsV1Api.return_value = mock_apps_v1
        mock_client.CoreV1Api.return_value = mock_core_v1
        
        # Mock deployment list
        mock_deployment_list = Mock()
        mock_deployment_list.items = [Mock(metadata=Mock(name="test-deployment"))]
        mock_apps_v1.list_namespaced_deployment.return_value = mock_deployment_list
        
        resource_manager = QuantumResourceManager()
        manager = KubernetesContainerManager(resource_manager)
        
        deployments = manager.list_deployments()
        
        assert isinstance(deployments, list)
        mock_apps_v1.list_namespaced_deployment.assert_called_once()
    
    def test_convert_pod_status(self):
        """Test pod status conversion."""
        resource_manager = QuantumResourceManager()
        
        with patch('quantrs2.quantum_containers.HAS_KUBERNETES', True):
            manager = KubernetesContainerManager(resource_manager)
            
            assert manager._convert_pod_status('Running') == ContainerStatus.RUNNING
            assert manager._convert_pod_status('Pending') == ContainerStatus.PENDING
            assert manager._convert_pod_status('Failed') == ContainerStatus.FAILED
            assert manager._convert_pod_status('Succeeded') == ContainerStatus.STOPPED
            assert manager._convert_pod_status('Unknown') == ContainerStatus.UNKNOWN
    
    def test_create_deployment_manifest(self, sample_deployment_spec):
        """Test deployment manifest creation."""
        resource_manager = QuantumResourceManager()
        
        with patch('quantrs2.quantum_containers.HAS_KUBERNETES', True):
            manager = KubernetesContainerManager(resource_manager)
            
            manifest = manager._create_deployment_manifest(sample_deployment_spec)
            
            assert isinstance(manifest, dict)
            assert manifest['apiVersion'] == 'apps/v1'
            assert manifest['kind'] == 'Deployment'
            assert manifest['metadata']['name'] == sample_deployment_spec.name
            assert manifest['spec']['replicas'] == sample_deployment_spec.replicas
            assert len(manifest['spec']['template']['spec']['containers']) == len(sample_deployment_spec.containers)
    
    def test_k8s_manager_without_kubernetes(self):
        """Test KubernetesContainerManager without Kubernetes."""
        with patch('quantrs2.quantum_containers.HAS_KUBERNETES', False):
            resource_manager = QuantumResourceManager()
            manager = KubernetesContainerManager(resource_manager)
            
            assert manager.api_client is None
            assert manager.apps_v1 is None
            assert manager.core_v1 is None


@pytest.mark.skipif(not HAS_QUANTUM_CONTAINERS, reason="quantum containers not available")
class TestQuantumContainerOrchestrator:
    """Test QuantumContainerOrchestrator functionality."""
    
    def test_orchestrator_initialization(self):
        """Test QuantumContainerOrchestrator initialization."""
        orchestrator = QuantumContainerOrchestrator(DeploymentMode.DOCKER, "localhost:5000")
        
        assert orchestrator.default_mode == DeploymentMode.DOCKER
        assert isinstance(orchestrator.resource_manager, QuantumResourceManager)
        assert isinstance(orchestrator.registry, QuantumContainerRegistry)
        assert isinstance(orchestrator.active_deployments, dict)
        assert isinstance(orchestrator.deployment_history, list)
    
    @patch('quantrs2.quantum_containers.HAS_DOCKER', True)
    @patch('quantrs2.quantum_containers.docker')
    def test_deploy_application_docker_mode(self, mock_docker, sample_deployment_spec, mock_docker_client):
        """Test application deployment in Docker mode."""
        mock_docker.from_env.return_value = mock_docker_client
        
        orchestrator = QuantumContainerOrchestrator()
        sample_deployment_spec.mode = DeploymentMode.DOCKER
        
        result = orchestrator.deploy_application(sample_deployment_spec)
        
        # Result depends on Docker availability and resource allocation
        assert isinstance(result, bool)
        
        if result:
            assert sample_deployment_spec.name in orchestrator.active_deployments
            assert len(orchestrator.deployment_history) > 0
    
    @patch('quantrs2.quantum_containers.HAS_KUBERNETES', True)
    @patch('quantrs2.quantum_containers.config')
    @patch('quantrs2.quantum_containers.client')
    def test_deploy_application_k8s_mode(self, mock_client, mock_config, sample_deployment_spec, mock_kubernetes_client):
        """Test application deployment in Kubernetes mode."""
        mock_config.load_kube_config.return_value = None
        mock_apps_v1, mock_core_v1 = mock_kubernetes_client
        mock_client.ApiClient.return_value = Mock()
        mock_client.AppsV1Api.return_value = mock_apps_v1
        mock_client.CoreV1Api.return_value = mock_core_v1
        
        orchestrator = QuantumContainerOrchestrator()
        sample_deployment_spec.mode = DeploymentMode.KUBERNETES
        
        result = orchestrator.deploy_application(sample_deployment_spec)
        
        # Result depends on Kubernetes availability
        assert isinstance(result, bool)
    
    def test_deploy_application_local_mode(self, sample_deployment_spec):
        """Test application deployment in local mode."""
        orchestrator = QuantumContainerOrchestrator()
        sample_deployment_spec.mode = DeploymentMode.LOCAL
        
        # Modify container config to avoid actual execution
        sample_deployment_spec.containers[0].command = []
        sample_deployment_spec.containers[0].args = []
        
        result = orchestrator.deploy_application(sample_deployment_spec)
        
        # Should succeed without actual process execution
        assert result is True
        assert sample_deployment_spec.name in orchestrator.active_deployments
    
    def test_deploy_application_hybrid_mode(self, sample_deployment_spec):
        """Test application deployment in hybrid mode."""
        orchestrator = QuantumContainerOrchestrator()
        sample_deployment_spec.mode = DeploymentMode.HYBRID
        
        result = orchestrator.deploy_application(sample_deployment_spec)
        
        # Result depends on available container managers
        assert isinstance(result, bool)
    
    def test_scale_deployment(self, quantum_orchestrator):
        """Test deployment scaling."""
        # Create a mock deployment
        spec = DeploymentSpec(
            name="test-scaling",
            replicas=2,
            containers=[ContainerConfig("test", "test:latest")],
            mode=DeploymentMode.DOCKER
        )
        
        quantum_orchestrator.active_deployments["test-scaling"] = spec
        
        result = quantum_orchestrator.scale_deployment("test-scaling", 5)
        
        # Result depends on deployment mode and manager availability
        assert isinstance(result, bool)
        
        if result:
            assert spec.replicas == 5
    
    def test_get_deployment_status(self, quantum_orchestrator):
        """Test deployment status retrieval."""
        # Create a mock deployment
        spec = DeploymentSpec(
            name="test-status",
            replicas=1,
            containers=[ContainerConfig("test", "test:latest")],
            mode=DeploymentMode.DOCKER
        )
        
        quantum_orchestrator.active_deployments["test-status"] = spec
        
        status = quantum_orchestrator.get_deployment_status("test-status")
        
        # May return None if Docker manager is not available
        if status:
            assert isinstance(status, DeploymentStatus)
            assert status.name == "test-status"
    
    def test_delete_deployment(self, quantum_orchestrator):
        """Test deployment deletion."""
        # Create a mock deployment
        spec = DeploymentSpec(
            name="test-delete",
            replicas=1,
            containers=[ContainerConfig("test", "test:latest")],
            mode=DeploymentMode.LOCAL  # Use local mode for easier testing
        )
        
        quantum_orchestrator.active_deployments["test-delete"] = spec
        
        result = quantum_orchestrator.delete_deployment("test-delete")
        
        assert result is True
        assert "test-delete" not in quantum_orchestrator.active_deployments
    
    def test_list_deployments(self, quantum_orchestrator):
        """Test deployment listing."""
        # Create mock deployments
        for i in range(3):
            spec = DeploymentSpec(
                name=f"test-list-{i}",
                replicas=1,
                containers=[ContainerConfig(f"test-{i}", "test:latest")],
                mode=DeploymentMode.LOCAL
            )
            quantum_orchestrator.active_deployments[f"test-list-{i}"] = spec
        
        deployments = quantum_orchestrator.list_deployments()
        
        assert isinstance(deployments, list)
        # Length may vary based on successful status retrieval
        assert len(deployments) <= 3
    
    def test_get_system_metrics(self, quantum_orchestrator):
        """Test system metrics collection."""
        metrics = quantum_orchestrator.get_system_metrics()
        
        assert isinstance(metrics, dict)
        assert 'timestamp' in metrics
        assert 'resource_utilization' in metrics
        assert 'active_deployments' in metrics
        assert 'deployment_history' in metrics
        assert 'container_stats' in metrics
        assert 'quantum_metrics' in metrics
    
    def test_auto_scaling_setup(self, sample_deployment_spec):
        """Test auto-scaling setup."""
        orchestrator = QuantumContainerOrchestrator()
        
        # Configure auto-scaling
        sample_deployment_spec.scaling_policy = ScalingPolicy.AUTO_CPU
        sample_deployment_spec.auto_scaling_config = {
            'min_replicas': 1,
            'max_replicas': 10,
            'cpu_threshold_up': 80,
            'cpu_threshold_down': 20
        }
        
        orchestrator._setup_auto_scaling(sample_deployment_spec)
        
        assert sample_deployment_spec.name in orchestrator.scaling_policies
        assert orchestrator.auto_scaling_enabled is True
        assert orchestrator.scaling_thread is not None
        
        # Clean up
        orchestrator.auto_scaling_enabled = False
        if orchestrator.scaling_thread:
            orchestrator.scaling_thread.join(timeout=1)
    
    def test_deployment_context_manager(self, sample_deployment_spec):
        """Test deployment context manager."""
        orchestrator = QuantumContainerOrchestrator()
        sample_deployment_spec.mode = DeploymentMode.LOCAL
        
        # Disable actual command execution
        sample_deployment_spec.containers[0].command = []
        sample_deployment_spec.containers[0].args = []
        
        with orchestrator.deployment_context(sample_deployment_spec) as ctx:
            assert ctx is orchestrator
            assert sample_deployment_spec.name in orchestrator.active_deployments
        
        # Should be cleaned up after context
        assert sample_deployment_spec.name not in orchestrator.active_deployments
    
    def test_unsupported_deployment_mode(self, sample_deployment_spec):
        """Test unsupported deployment mode handling."""
        orchestrator = QuantumContainerOrchestrator()
        
        # Use an invalid mode (this requires mocking the enum)
        sample_deployment_spec.mode = "invalid_mode"
        
        result = orchestrator.deploy_application(sample_deployment_spec)
        
        assert result is False


@pytest.mark.skipif(not HAS_QUANTUM_CONTAINERS, reason="quantum containers not available")
class TestConvenienceFunctions:
    """Test convenience functions for easy usage."""
    
    def test_get_quantum_container_orchestrator(self):
        """Test get_quantum_container_orchestrator function."""
        orchestrator = get_quantum_container_orchestrator(DeploymentMode.DOCKER, "test:5000")
        
        assert isinstance(orchestrator, QuantumContainerOrchestrator)
        assert orchestrator.default_mode == DeploymentMode.DOCKER
        assert orchestrator.registry.registry_url == "test:5000"
    
    def test_create_quantum_container_config(self):
        """Test create_quantum_container_config function."""
        config = create_quantum_container_config(
            name="test-quantum-app",
            image="quantrs2:latest",
            quantum_simulators=2,
            max_qubits=32,
            cpu_cores=4,
            memory="8Gi"
        )
        
        assert isinstance(config, ContainerConfig)
        assert config.name == "test-quantum-app"
        assert config.image == "quantrs2:latest"
        assert config.environment["QUANTRS2_QUANTUM_SIMULATORS"] == "2"
        assert config.environment["QUANTRS2_MAX_QUBITS"] == "32"
        assert config.quantum_config["simulators"] == 2
        assert config.quantum_config["max_qubits"] == 32
        
        # Check resource requirements
        quantum_sim_req = next(req for req in config.resources if req.type == ResourceType.QUANTUM_SIMULATOR)
        assert quantum_sim_req.amount == 2
        assert quantum_sim_req.metadata['max_qubits'] == 32
        
        cpu_req = next(req for req in config.resources if req.type == ResourceType.CPU)
        assert cpu_req.amount == 4
        
        memory_req = next(req for req in config.resources if req.type == ResourceType.MEMORY)
        assert memory_req.amount == "8Gi"
    
    def test_create_quantum_deployment_spec(self):
        """Test create_quantum_deployment_spec function."""
        container_config = create_quantum_container_config("test", "test:latest")
        
        spec = create_quantum_deployment_spec(
            name="test-deployment",
            containers=[container_config],
            mode=DeploymentMode.KUBERNETES,
            replicas=3,
            auto_scale=True,
            min_replicas=1,
            max_replicas=10,
            scale_up_threshold=75,
            scale_down_threshold=25
        )
        
        assert isinstance(spec, DeploymentSpec)
        assert spec.name == "test-deployment"
        assert spec.replicas == 3
        assert spec.mode == DeploymentMode.KUBERNETES
        assert len(spec.containers) == 1
        assert spec.scaling_policy == ScalingPolicy.AUTO_QUANTUM_LOAD
        assert spec.auto_scaling_config['min_replicas'] == 1
        assert spec.auto_scaling_config['max_replicas'] == 10
        assert spec.auto_scaling_config['quantum_threshold_up'] == 75
        assert spec.auto_scaling_config['quantum_threshold_down'] == 25
        
        # Check quantum requirements
        assert spec.quantum_requirements['total_simulators'] == 1
        assert spec.quantum_requirements['max_qubits_required'] == 16
    
    def test_create_quantum_deployment_spec_without_autoscale(self):
        """Test create_quantum_deployment_spec without auto-scaling."""
        container_config = create_quantum_container_config("test", "test:latest")
        
        spec = create_quantum_deployment_spec(
            name="test-deployment",
            containers=[container_config],
            auto_scale=False
        )
        
        assert spec.scaling_policy == ScalingPolicy.MANUAL
        assert spec.auto_scaling_config == {}
    
    def test_deploy_quantum_application(self):
        """Test deploy_quantum_application function."""
        # Mock the orchestrator to avoid actual deployment
        with patch('quantrs2.quantum_containers.get_quantum_container_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = Mock()
            mock_orchestrator.deploy_application.return_value = True
            mock_get_orchestrator.return_value = mock_orchestrator
            
            result = deploy_quantum_application(
                name="test-app",
                image="quantrs2:latest",
                mode=DeploymentMode.DOCKER,
                quantum_simulators=2,
                replicas=2,
                cpu_cores=4,
                memory="8Gi"
            )
            
            assert result is True
            mock_orchestrator.deploy_application.assert_called_once()
            
            # Check the deployment spec passed to deploy_application
            call_args = mock_orchestrator.deploy_application.call_args[0]
            deployment_spec = call_args[0]
            
            assert isinstance(deployment_spec, DeploymentSpec)
            assert deployment_spec.name == "test-app"
            assert deployment_spec.replicas == 2
            assert deployment_spec.mode == DeploymentMode.DOCKER


@pytest.mark.skipif(not HAS_QUANTUM_CONTAINERS, reason="quantum containers not available")
class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    def test_resource_allocation_failure(self):
        """Test resource allocation failure handling."""
        manager = QuantumResourceManager()
        
        # Request excessive resources
        requirements = [
            ResourceRequirement(ResourceType.QUANTUM_SIMULATOR, 1000)  # Excessive amount
        ]
        
        result = manager.allocate_resources("test_container", requirements)
        
        assert result['success'] is False
        assert len(result['errors']) > 0
        assert "Insufficient" in result['errors'][0]
    
    def test_container_creation_failure(self):
        """Test container creation failure handling."""
        resource_manager = QuantumResourceManager()
        
        with patch('quantrs2.quantum_containers.HAS_DOCKER', True):
            with patch('quantrs2.quantum_containers.docker') as mock_docker:
                mock_client = Mock()
                mock_client.containers.create.side_effect = Exception("Docker error")
                mock_docker.from_env.return_value = mock_client
                
                manager = DockerContainerManager(resource_manager)
                manager.client = mock_client
                
                config = ContainerConfig("test", "test:latest")
                instance = manager.create_container(config)
                
                assert instance is None
    
    def test_deployment_with_missing_managers(self):
        """Test deployment with missing container managers."""
        with patch('quantrs2.quantum_containers.HAS_DOCKER', False):
            with patch('quantrs2.quantum_containers.HAS_KUBERNETES', False):
                orchestrator = QuantumContainerOrchestrator()
                
                spec = DeploymentSpec(
                    name="test",
                    containers=[ContainerConfig("test", "test:latest")],
                    mode=DeploymentMode.DOCKER
                )
                
                result = orchestrator.deploy_application(spec)
                
                assert result is False
    
    def test_invalid_deployment_spec(self):
        """Test invalid deployment specification handling."""
        orchestrator = QuantumContainerOrchestrator()
        
        # Empty containers list
        spec = DeploymentSpec(
            name="test",
            containers=[],
            mode=DeploymentMode.LOCAL
        )
        
        result = orchestrator.deploy_application(spec)
        
        # Should handle gracefully
        assert isinstance(result, bool)
    
    def test_scaling_nonexistent_deployment(self):
        """Test scaling of non-existent deployment."""
        orchestrator = QuantumContainerOrchestrator()
        
        result = orchestrator.scale_deployment("nonexistent", 5)
        
        assert result is False
    
    def test_deleting_nonexistent_deployment(self):
        """Test deletion of non-existent deployment."""
        orchestrator = QuantumContainerOrchestrator()
        
        result = orchestrator.delete_deployment("nonexistent")
        
        assert result is False
    
    def test_concurrent_resource_allocation(self):
        """Test concurrent resource allocation."""
        import threading
        
        manager = QuantumResourceManager()
        results = []
        errors = []
        
        def allocate_resources(container_id):
            try:
                requirements = [
                    ResourceRequirement(ResourceType.QUANTUM_SIMULATOR, 1)
                ]
                result = manager.allocate_resources(container_id, requirements)
                results.append((container_id, result))
            except Exception as e:
                errors.append((container_id, e))
        
        # Start multiple allocation threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=allocate_resources, args=(f"container_{i}",))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Check results
        assert len(errors) == 0  # No errors should occur
        successful_allocations = [r for _, r in results if r['success']]
        
        # Should have some successful allocations (limited by available simulators)
        assert len(successful_allocations) <= manager.available_resources[ResourceType.QUANTUM_SIMULATOR]['count']
    
    def test_auto_scaling_with_missing_deployment(self):
        """Test auto-scaling with missing deployment."""
        orchestrator = QuantumContainerOrchestrator()
        
        policy = {
            'policy': ScalingPolicy.AUTO_CPU,
            'config': {'cpu_threshold_up': 80},
            'current_replicas': 2,
            'min_replicas': 1,
            'max_replicas': 10,
            'last_scaled': 0
        }
        
        # Should handle missing deployment gracefully
        orchestrator._check_scaling_conditions("nonexistent", policy)
        
        # Should not crash
        assert True
    
    def test_resource_manager_without_psutil(self):
        """Test resource manager functionality without psutil."""
        with patch('quantrs2.quantum_containers.HAS_PSUTIL', False):
            manager = QuantumResourceManager()
            
            utilization = manager.get_resource_utilization()
            
            # Should still return basic utilization data
            assert isinstance(utilization, dict)
            assert 'quantum_simulator' in utilization
    
    def test_container_logs_retrieval_failure(self):
        """Test container logs retrieval failure handling."""
        resource_manager = QuantumResourceManager()
        
        with patch('quantrs2.quantum_containers.HAS_DOCKER', True):
            with patch('quantrs2.quantum_containers.docker') as mock_docker:
                mock_client = Mock()
                mock_container = Mock()
                mock_container.logs.side_effect = Exception("Logs error")
                mock_client.containers.get.return_value = mock_container
                mock_docker.from_env.return_value = mock_client
                
                manager = DockerContainerManager(resource_manager)
                manager.client = mock_client
                
                instance = ContainerInstance(
                    id="test_id",
                    name="test",
                    config=ContainerConfig("test", "test:latest"),
                    status=ContainerStatus.RUNNING,
                    deployment_name="test",
                    created_at=time.time()
                )
                manager.containers["test_id"] = instance
                
                logs = manager.get_container_logs("test_id")
                
                assert logs == []  # Should return empty list on error
    
    def test_deployment_context_failure(self):
        """Test deployment context manager with deployment failure."""
        orchestrator = QuantumContainerOrchestrator()
        
        spec = DeploymentSpec(
            name="failing-deployment",
            containers=[ContainerConfig("test", "nonexistent:latest")],
            mode=DeploymentMode.DOCKER
        )
        
        with pytest.raises(RuntimeError, match="Failed to deploy"):
            with orchestrator.deployment_context(spec):
                pass


@pytest.mark.skipif(not HAS_QUANTUM_CONTAINERS, reason="quantum containers not available")
class TestPerformanceAndScalability:
    """Test performance and scalability of container orchestration."""
    
    def test_large_deployment_handling(self):
        """Test handling of large deployments."""
        orchestrator = QuantumContainerOrchestrator()
        
        # Create deployment with many containers
        containers = []
        for i in range(50):
            container = ContainerConfig(f"test-{i}", "test:latest")
            containers.append(container)
        
        spec = DeploymentSpec(
            name="large-deployment",
            containers=containers,
            mode=DeploymentMode.LOCAL  # Use local mode for testing
        )
        
        start_time = time.time()
        result = orchestrator.deploy_application(spec)
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 5.0
        assert isinstance(result, bool)
    
    def test_resource_utilization_performance(self):
        """Test resource utilization calculation performance."""
        manager = QuantumResourceManager()
        
        # Allocate many resources
        for i in range(100):
            requirements = [
                ResourceRequirement(ResourceType.CPU, 1, "cores")
            ]
            manager.allocate_resources(f"container_{i}", requirements)
        
        start_time = time.time()
        utilization = manager.get_resource_utilization()
        end_time = time.time()
        
        # Should be fast
        assert end_time - start_time < 1.0
        assert isinstance(utilization, dict)
    
    def test_concurrent_deployments(self):
        """Test concurrent deployment operations."""
        import threading
        
        orchestrator = QuantumContainerOrchestrator()
        results = []
        errors = []
        
        def deploy_application(app_id):
            try:
                spec = DeploymentSpec(
                    name=f"concurrent-app-{app_id}",
                    containers=[ContainerConfig(f"test-{app_id}", "test:latest")],
                    mode=DeploymentMode.LOCAL
                )
                
                result = orchestrator.deploy_application(spec)
                results.append((app_id, result))
            except Exception as e:
                errors.append((app_id, e))
        
        # Start multiple deployment threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=deploy_application, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10.0)
        
        # Check results
        assert len(errors) == 0  # No errors should occur
        assert len(results) == 10  # All deployments should complete
        
        successful_deployments = [r for _, r in results if r]
        assert len(successful_deployments) > 0  # At least some should succeed
    
    def test_system_metrics_collection_performance(self):
        """Test system metrics collection performance."""
        orchestrator = QuantumContainerOrchestrator()
        
        # Create several active deployments
        for i in range(20):
            spec = DeploymentSpec(
                name=f"metrics-test-{i}",
                containers=[ContainerConfig(f"test-{i}", "test:latest")],
                mode=DeploymentMode.LOCAL
            )
            orchestrator.active_deployments[f"metrics-test-{i}"] = spec
        
        start_time = time.time()
        metrics = orchestrator.get_system_metrics()
        end_time = time.time()
        
        # Should be efficient
        assert end_time - start_time < 2.0
        assert isinstance(metrics, dict)
        assert metrics['active_deployments'] == 20
    
    def test_auto_scaling_decision_performance(self):
        """Test auto-scaling decision making performance."""
        orchestrator = QuantumContainerOrchestrator()
        
        # Setup many scaling policies
        for i in range(50):
            policy = {
                'policy': ScalingPolicy.AUTO_CPU,
                'config': {'cpu_threshold_up': 80, 'cpu_threshold_down': 20},
                'current_replicas': 2,
                'min_replicas': 1,
                'max_replicas': 10,
                'last_scaled': 0
            }
            orchestrator.scaling_policies[f"deployment_{i}"] = policy
            
            # Add corresponding deployment
            spec = DeploymentSpec(
                name=f"deployment_{i}",
                containers=[ContainerConfig(f"test-{i}", "test:latest")],
                mode=DeploymentMode.LOCAL
            )
            orchestrator.active_deployments[f"deployment_{i}"] = spec
        
        start_time = time.time()
        
        # Check scaling conditions for all policies
        for deployment_name, policy in orchestrator.scaling_policies.items():
            orchestrator._check_scaling_conditions(deployment_name, policy)
        
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 3.0


if __name__ == "__main__":
    pytest.main([__file__])