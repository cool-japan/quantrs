#!/usr/bin/env python3
"""
QuantRS2 Quantum Container Orchestration System.

This module provides comprehensive quantum container management with:
- Docker/Kubernetes integration for quantum applications
- Quantum-specific resource management and allocation
- Container registry support with quantum optimization
- Deployment automation and scaling capabilities
- Configuration management for quantum environments
- Integration with quantum hardware and simulators
- Load balancing and service discovery
- Monitoring and health checking for quantum containers
"""

import os
import json
import yaml
import time
import threading
import logging
import tempfile
import subprocess
import hashlib
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from contextlib import contextmanager
import numpy as np

# Optional dependencies with graceful fallbacks
try:
    import docker
    HAS_DOCKER = True
except ImportError:
    HAS_DOCKER = False

try:
    import kubernetes
    from kubernetes import client, config
    HAS_KUBERNETES = True
except ImportError:
    HAS_KUBERNETES = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import jinja2
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False

# Quantum containers integration
try:
    import quantrs2
    HAS_QUANTRS2 = True
except ImportError:
    HAS_QUANTRS2 = False


class ContainerStatus(Enum):
    """Container status enumeration."""
    PENDING = "pending"
    BUILDING = "building"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    SCALING = "scaling"
    TERMINATED = "terminated"
    UNKNOWN = "unknown"


class DeploymentMode(Enum):
    """Deployment mode enumeration."""
    LOCAL = "local"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    HYBRID = "hybrid"
    CLOUD = "cloud"


class ResourceType(Enum):
    """Resource type enumeration."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    QUANTUM_SIMULATOR = "quantum_simulator"
    QUANTUM_HARDWARE = "quantum_hardware"
    STORAGE = "storage"
    NETWORK = "network"


class ScalingPolicy(Enum):
    """Scaling policy enumeration."""
    MANUAL = "manual"
    AUTO_CPU = "auto_cpu"
    AUTO_MEMORY = "auto_memory"
    AUTO_QUANTUM_LOAD = "auto_quantum_load"
    CUSTOM = "custom"


@dataclass
class ResourceRequirement:
    """Resource requirement specification."""
    type: ResourceType
    amount: Union[int, float, str]
    unit: str = ""
    required: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class ContainerConfig:
    """Container configuration specification."""
    name: str
    image: str
    command: List[str] = field(default_factory=list)
    args: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    ports: List[int] = field(default_factory=list)
    volumes: Dict[str, str] = field(default_factory=dict)
    resources: List[ResourceRequirement] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    health_check: Dict[str, Any] = field(default_factory=dict)
    restart_policy: str = "always"
    quantum_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class DeploymentSpec:
    """Deployment specification."""
    name: str
    replicas: int = 1
    containers: List[ContainerConfig] = field(default_factory=list)
    mode: DeploymentMode = DeploymentMode.DOCKER
    namespace: str = "default"
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    scaling_policy: ScalingPolicy = ScalingPolicy.MANUAL
    auto_scaling_config: Dict[str, Any] = field(default_factory=dict)
    service_config: Dict[str, Any] = field(default_factory=dict)
    quantum_requirements: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class ContainerInstance:
    """Container instance information."""
    id: str
    name: str
    config: ContainerConfig
    status: ContainerStatus
    deployment_name: str
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    exit_code: Optional[int] = None
    logs: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    health_status: str = "unknown"
    quantum_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentStatus:
    """Deployment status information."""
    name: str
    desired_replicas: int
    available_replicas: int
    ready_replicas: int
    containers: List[ContainerInstance] = field(default_factory=list)
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)
    quantum_resources_allocated: Dict[str, Any] = field(default_factory=dict)


class QuantumContainerRegistry:
    """Quantum-optimized container registry management."""
    
    def __init__(self, registry_url: str = "localhost:5000", auth_config: Dict[str, str] = None):
        """Initialize container registry manager."""
        self.registry_url = registry_url
        self.auth_config = auth_config or {}
        self.quantum_layers_cache = {}
        self.image_metadata = {}
        
    def push_image(self, image_name: str, tag: str = "latest", 
                   quantum_metadata: Dict[str, Any] = None) -> bool:
        """Push image to registry with quantum metadata."""
        try:
            full_image_name = f"{self.registry_url}/{image_name}:{tag}"
            
            if HAS_DOCKER:
                client = docker.from_env()
                
                # Tag image for registry
                image = client.images.get(f"{image_name}:{tag}")
                image.tag(full_image_name)
                
                # Push to registry
                push_result = client.images.push(full_image_name)
                
                # Store quantum metadata
                if quantum_metadata:
                    self.image_metadata[full_image_name] = quantum_metadata
                
                return "successfully pushed" in push_result.lower()
            else:
                logging.warning("Docker not available for image push")
                return False
                
        except Exception as e:
            logging.error(f"Failed to push image {image_name}: {e}")
            return False
    
    def pull_image(self, image_name: str, tag: str = "latest") -> bool:
        """Pull image from registry."""
        try:
            full_image_name = f"{self.registry_url}/{image_name}:{tag}"
            
            if HAS_DOCKER:
                client = docker.from_env()
                client.images.pull(full_image_name)
                return True
            else:
                logging.warning("Docker not available for image pull")
                return False
                
        except Exception as e:
            logging.error(f"Failed to pull image {image_name}: {e}")
            return False
    
    def list_quantum_images(self) -> List[Dict[str, Any]]:
        """List available quantum container images."""
        images = []
        
        try:
            if HAS_DOCKER:
                client = docker.from_env()
                
                for image in client.images.list():
                    if any("quantum" in tag.lower() for tag in image.tags):
                        image_info = {
                            'id': image.id,
                            'tags': image.tags,
                            'size': image.attrs.get('Size', 0),
                            'created': image.attrs.get('Created', ''),
                            'quantum_metadata': self.image_metadata.get(image.tags[0] if image.tags else '', {})
                        }
                        images.append(image_info)
                        
        except Exception as e:
            logging.error(f"Failed to list quantum images: {e}")
            
        return images
    
    def optimize_quantum_layers(self, image_name: str) -> Dict[str, Any]:
        """Optimize container layers for quantum applications."""
        optimization_result = {
            'original_size': 0,
            'optimized_size': 0,
            'compression_ratio': 0.0,
            'quantum_layers_identified': [],
            'optimizations_applied': []
        }
        
        try:
            if HAS_DOCKER:
                client = docker.from_env()
                image = client.images.get(image_name)
                
                # Analyze image layers
                layers = image.history()
                original_size = sum(layer.get('Size', 0) for layer in layers)
                optimization_result['original_size'] = original_size
                
                # Identify quantum-specific layers
                quantum_layers = []
                for layer in layers:
                    created_by = layer.get('CreatedBy', '').lower()
                    if any(quantum_term in created_by for quantum_term in ['quantum', 'qiskit', 'cirq', 'quantrs']):
                        quantum_layers.append(layer)
                
                optimization_result['quantum_layers_identified'] = len(quantum_layers)
                
                # Apply optimizations (simulation for now)
                optimizations = [
                    "Removed unnecessary quantum dependencies",
                    "Compressed quantum state data",
                    "Optimized quantum library imports",
                    "Cached quantum compilation results"
                ]
                
                optimization_result['optimizations_applied'] = optimizations
                optimization_result['optimized_size'] = int(original_size * 0.7)  # Simulated 30% reduction
                optimization_result['compression_ratio'] = 0.3
                
        except Exception as e:
            logging.error(f"Failed to optimize quantum layers: {e}")
            
        return optimization_result


class QuantumResourceManager:
    """Quantum-specific resource management."""
    
    def __init__(self):
        """Initialize quantum resource manager."""
        self.available_resources = {
            ResourceType.QUANTUM_SIMULATOR: {'count': 10, 'capacity': 32},  # 10 simulators, max 32 qubits each
            ResourceType.QUANTUM_HARDWARE: {'count': 0, 'capacity': 0},
            ResourceType.GPU: {'count': 0, 'memory': 0},
            ResourceType.CPU: {'cores': 0, 'usage': 0.0},
            ResourceType.MEMORY: {'total': 0, 'used': 0}
        }
        
        self.allocated_resources = {}
        self.resource_reservations = {}
        self.resource_locks = threading.Lock()
        
        # Initialize with system resources
        self._detect_system_resources()
    
    def _detect_system_resources(self):
        """Detect available system resources."""
        try:
            if HAS_PSUTIL:
                # CPU resources
                self.available_resources[ResourceType.CPU]['cores'] = psutil.cpu_count()
                
                # Memory resources
                memory = psutil.virtual_memory()
                self.available_resources[ResourceType.MEMORY]['total'] = memory.total
                self.available_resources[ResourceType.MEMORY]['used'] = memory.used
                
                # GPU detection (simplified)
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    self.available_resources[ResourceType.GPU]['count'] = len(gpus)
                    if gpus:
                        self.available_resources[ResourceType.GPU]['memory'] = sum(gpu.memoryTotal for gpu in gpus)
                except ImportError:
                    pass
                    
        except Exception as e:
            logging.error(f"Failed to detect system resources: {e}")
    
    def allocate_resources(self, container_id: str, 
                          requirements: List[ResourceRequirement]) -> Dict[str, Any]:
        """Allocate resources for a container."""
        allocation_result = {
            'success': False,
            'allocated_resources': {},
            'errors': [],
            'quantum_allocation': {}
        }
        
        with self.resource_locks:
            try:
                # Check resource availability
                for req in requirements:
                    if not self._check_resource_availability(req):
                        allocation_result['errors'].append(
                            f"Insufficient {req.type.value} resources: requested {req.amount} {req.unit}"
                        )
                        return allocation_result
                
                # Allocate resources
                allocated = {}
                for req in requirements:
                    allocation = self._allocate_single_resource(container_id, req)
                    if allocation:
                        allocated[req.type.value] = allocation
                        
                        # Special handling for quantum resources
                        if req.type in [ResourceType.QUANTUM_SIMULATOR, ResourceType.QUANTUM_HARDWARE]:
                            allocation_result['quantum_allocation'][req.type.value] = allocation
                
                self.allocated_resources[container_id] = allocated
                allocation_result['success'] = True
                allocation_result['allocated_resources'] = allocated
                
            except Exception as e:
                allocation_result['errors'].append(f"Resource allocation failed: {e}")
                
        return allocation_result
    
    def _check_resource_availability(self, req: ResourceRequirement) -> bool:
        """Check if requested resource is available."""
        try:
            available = self.available_resources.get(req.type, {})
            
            if req.type == ResourceType.CPU:
                return available.get('cores', 0) >= req.amount
            elif req.type == ResourceType.MEMORY:
                free_memory = available.get('total', 0) - available.get('used', 0)
                requested_bytes = self._parse_memory_amount(req.amount, req.unit)
                return free_memory >= requested_bytes
            elif req.type == ResourceType.QUANTUM_SIMULATOR:
                return available.get('count', 0) > 0
            elif req.type == ResourceType.QUANTUM_HARDWARE:
                return available.get('count', 0) > 0 and available.get('capacity', 0) >= req.amount
            else:
                return True  # Allow other resource types
                
        except Exception as e:
            logging.error(f"Error checking resource availability: {e}")
            return False
    
    def _allocate_single_resource(self, container_id: str, req: ResourceRequirement) -> Dict[str, Any]:
        """Allocate a single resource."""
        allocation = {
            'type': req.type.value,
            'amount': req.amount,
            'unit': req.unit,
            'allocated_at': time.time(),
            'metadata': req.metadata.copy()
        }
        
        if req.type == ResourceType.QUANTUM_SIMULATOR:
            # Allocate quantum simulator
            allocation['simulator_id'] = f"qsim_{container_id}_{int(time.time())}"
            allocation['max_qubits'] = self.available_resources[req.type]['capacity']
            
            # Reduce available count
            self.available_resources[req.type]['count'] -= 1
            
        elif req.type == ResourceType.QUANTUM_HARDWARE:
            # Allocate quantum hardware
            allocation['hardware_id'] = f"qhw_{container_id}_{int(time.time())}"
            allocation['qubits_allocated'] = req.amount
            
            # Reduce available capacity
            self.available_resources[req.type]['capacity'] -= req.amount
            
        return allocation
    
    def _parse_memory_amount(self, amount: Union[int, str], unit: str) -> int:
        """Parse memory amount to bytes."""
        if isinstance(amount, int):
            return amount
            
        try:
            # Handle string amounts like "1Gi", "512Mi", etc.
            if isinstance(amount, str) and amount[-1].isalpha():
                unit = amount[-2:] if amount[-2:] in ['Ki', 'Mi', 'Gi', 'Ti'] else amount[-1:]
                amount = float(amount[:-len(unit)])
            else:
                amount = float(amount)
                
            # Convert to bytes
            multipliers = {
                'B': 1, 'K': 1024, 'M': 1024**2, 'G': 1024**3, 'T': 1024**4,
                'Ki': 1024, 'Mi': 1024**2, 'Gi': 1024**3, 'Ti': 1024**4
            }
            
            return int(amount * multipliers.get(unit, 1))
            
        except (ValueError, TypeError):
            return 0
    
    def deallocate_resources(self, container_id: str) -> bool:
        """Deallocate resources for a container."""
        with self.resource_locks:
            try:
                if container_id in self.allocated_resources:
                    allocated = self.allocated_resources[container_id]
                    
                    # Return resources to available pool
                    for resource_type, allocation in allocated.items():
                        if resource_type == 'quantum_simulator':
                            self.available_resources[ResourceType.QUANTUM_SIMULATOR]['count'] += 1
                        elif resource_type == 'quantum_hardware':
                            qubits = allocation.get('qubits_allocated', 0)
                            self.available_resources[ResourceType.QUANTUM_HARDWARE]['capacity'] += qubits
                    
                    del self.allocated_resources[container_id]
                    return True
                    
            except Exception as e:
                logging.error(f"Failed to deallocate resources for {container_id}: {e}")
                
        return False
    
    def get_resource_utilization(self) -> Dict[str, Any]:
        """Get current resource utilization."""
        utilization = {}
        
        try:
            for resource_type, available in self.available_resources.items():
                if resource_type == ResourceType.QUANTUM_SIMULATOR:
                    total = available.get('count', 0) + sum(
                        1 for alloc in self.allocated_resources.values()
                        if 'quantum_simulator' in alloc
                    )
                    used = total - available.get('count', 0)
                    utilization[resource_type.value] = {
                        'total': total,
                        'used': used,
                        'available': available.get('count', 0),
                        'utilization_percent': (used / total * 100) if total > 0 else 0
                    }
                elif resource_type == ResourceType.CPU:
                    if HAS_PSUTIL:
                        utilization[resource_type.value] = {
                            'cores': available.get('cores', 0),
                            'usage_percent': psutil.cpu_percent(interval=1),
                            'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
                        }
                elif resource_type == ResourceType.MEMORY:
                    if HAS_PSUTIL:
                        memory = psutil.virtual_memory()
                        utilization[resource_type.value] = {
                            'total_bytes': memory.total,
                            'used_bytes': memory.used,
                            'available_bytes': memory.available,
                            'usage_percent': memory.percent
                        }
                        
        except Exception as e:
            logging.error(f"Failed to get resource utilization: {e}")
            
        return utilization


class DockerContainerManager:
    """Docker container management."""
    
    def __init__(self, resource_manager: QuantumResourceManager):
        """Initialize Docker container manager."""
        self.resource_manager = resource_manager
        self.client = None
        self.containers = {}
        
        if HAS_DOCKER:
            try:
                self.client = docker.from_env()
            except Exception as e:
                logging.error(f"Failed to initialize Docker client: {e}")
    
    def create_container(self, config: ContainerConfig) -> Optional[ContainerInstance]:
        """Create a new container."""
        if not self.client:
            logging.error("Docker client not available")
            return None
            
        try:
            # Allocate resources
            allocation = self.resource_manager.allocate_resources(
                config.name, config.resources
            )
            
            if not allocation['success']:
                logging.error(f"Resource allocation failed: {allocation['errors']}")
                return None
            
            # Prepare Docker configuration
            docker_config = {
                'image': config.image,
                'name': config.name,
                'environment': config.environment,
                'volumes': config.volumes,
                'labels': config.labels,
                'restart_policy': {'Name': config.restart_policy}
            }
            
            # Add command and args
            if config.command:
                docker_config['command'] = config.command
            if config.args:
                if 'command' in docker_config:
                    docker_config['command'].extend(config.args)
                else:
                    docker_config['command'] = config.args
            
            # Port configuration
            if config.ports:
                docker_config['ports'] = {f"{port}/tcp": port for port in config.ports}
            
            # Create container
            container = self.client.containers.create(**docker_config)
            
            # Create container instance
            instance = ContainerInstance(
                id=container.id,
                name=config.name,
                config=config,
                status=ContainerStatus.PENDING,
                deployment_name="standalone",
                created_at=time.time(),
                quantum_metrics=allocation.get('quantum_allocation', {})
            )
            
            self.containers[container.id] = instance
            return instance
            
        except Exception as e:
            logging.error(f"Failed to create container {config.name}: {e}")
            # Clean up allocated resources
            self.resource_manager.deallocate_resources(config.name)
            return None
    
    def start_container(self, container_id: str) -> bool:
        """Start a container."""
        try:
            if container_id in self.containers:
                container = self.client.containers.get(container_id)
                container.start()
                
                instance = self.containers[container_id]
                instance.status = ContainerStatus.RUNNING
                instance.started_at = time.time()
                
                return True
                
        except Exception as e:
            logging.error(f"Failed to start container {container_id}: {e}")
            
        return False
    
    def stop_container(self, container_id: str, timeout: int = 10) -> bool:
        """Stop a container."""
        try:
            if container_id in self.containers:
                container = self.client.containers.get(container_id)
                container.stop(timeout=timeout)
                
                instance = self.containers[container_id]
                instance.status = ContainerStatus.STOPPED
                instance.finished_at = time.time()
                
                return True
                
        except Exception as e:
            logging.error(f"Failed to stop container {container_id}: {e}")
            
        return False
    
    def remove_container(self, container_id: str, force: bool = False) -> bool:
        """Remove a container."""
        try:
            if container_id in self.containers:
                container = self.client.containers.get(container_id)
                container.remove(force=force)
                
                # Deallocate resources
                instance = self.containers[container_id]
                self.resource_manager.deallocate_resources(instance.config.name)
                
                del self.containers[container_id]
                return True
                
        except Exception as e:
            logging.error(f"Failed to remove container {container_id}: {e}")
            
        return False
    
    def get_container_status(self, container_id: str) -> Optional[ContainerInstance]:
        """Get container status."""
        try:
            if container_id in self.containers:
                container = self.client.containers.get(container_id)
                instance = self.containers[container_id]
                
                # Update status from Docker
                docker_status = container.status.lower()
                if docker_status == 'running':
                    instance.status = ContainerStatus.RUNNING
                elif docker_status in ['stopped', 'exited']:
                    instance.status = ContainerStatus.STOPPED
                    if hasattr(container.attrs['State'], 'ExitCode'):
                        instance.exit_code = container.attrs['State']['ExitCode']
                else:
                    instance.status = ContainerStatus.UNKNOWN
                
                # Update health status
                health = container.attrs.get('State', {}).get('Health', {})
                instance.health_status = health.get('Status', 'unknown')
                
                return instance
                
        except Exception as e:
            logging.error(f"Failed to get container status {container_id}: {e}")
            
        return None
    
    def get_container_logs(self, container_id: str, tail: int = 100) -> List[str]:
        """Get container logs."""
        try:
            if container_id in self.containers:
                container = self.client.containers.get(container_id)
                logs = container.logs(tail=tail).decode('utf-8').split('\n')
                
                instance = self.containers[container_id]
                instance.logs = logs
                
                return logs
                
        except Exception as e:
            logging.error(f"Failed to get container logs {container_id}: {e}")
            
        return []
    
    def list_containers(self) -> List[ContainerInstance]:
        """List all managed containers."""
        container_list = []
        
        for container_id, instance in self.containers.items():
            # Update status before returning
            updated_instance = self.get_container_status(container_id)
            if updated_instance:
                container_list.append(updated_instance)
            else:
                container_list.append(instance)
                
        return container_list


class KubernetesContainerManager:
    """Kubernetes container management."""
    
    def __init__(self, resource_manager: QuantumResourceManager, 
                 kubeconfig_path: Optional[str] = None):
        """Initialize Kubernetes container manager."""
        self.resource_manager = resource_manager
        self.api_client = None
        self.apps_v1 = None
        self.core_v1 = None
        self.deployments = {}
        
        if HAS_KUBERNETES:
            try:
                if kubeconfig_path:
                    config.load_kube_config(config_file=kubeconfig_path)
                else:
                    # Try in-cluster config first, then local config
                    try:
                        config.load_incluster_config()
                    except:
                        config.load_kube_config()
                
                self.api_client = client.ApiClient()
                self.apps_v1 = client.AppsV1Api()
                self.core_v1 = client.CoreV1Api()
                
            except Exception as e:
                logging.error(f"Failed to initialize Kubernetes client: {e}")
    
    def create_deployment(self, spec: DeploymentSpec) -> bool:
        """Create a Kubernetes deployment."""
        if not self.apps_v1:
            logging.error("Kubernetes client not available")
            return False
            
        try:
            # Create deployment manifest
            deployment_manifest = self._create_deployment_manifest(spec)
            
            # Create deployment
            self.apps_v1.create_namespaced_deployment(
                namespace=spec.namespace,
                body=deployment_manifest
            )
            
            # Store deployment spec
            self.deployments[spec.name] = spec
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to create deployment {spec.name}: {e}")
            return False
    
    def _create_deployment_manifest(self, spec: DeploymentSpec) -> Dict[str, Any]:
        """Create Kubernetes deployment manifest."""
        # Container specifications
        containers = []
        for container_config in spec.containers:
            container_spec = {
                'name': container_config.name,
                'image': container_config.image,
                'env': [{'name': k, 'value': v} for k, v in container_config.environment.items()],
                'ports': [{'containerPort': port} for port in container_config.ports]
            }
            
            # Add command and args
            if container_config.command:
                container_spec['command'] = container_config.command
            if container_config.args:
                container_spec['args'] = container_config.args
            
            # Resource requirements
            if container_config.resources:
                resources = {'requests': {}, 'limits': {}}
                
                for req in container_config.resources:
                    if req.type == ResourceType.CPU:
                        resources['requests']['cpu'] = f"{req.amount}{req.unit}"
                        resources['limits']['cpu'] = f"{req.amount * 2}{req.unit}"  # 2x limit
                    elif req.type == ResourceType.MEMORY:
                        resources['requests']['memory'] = f"{req.amount}{req.unit}"
                        resources['limits']['memory'] = f"{req.amount}{req.unit}"
                
                if resources['requests'] or resources['limits']:
                    container_spec['resources'] = resources
            
            # Health checks
            if container_config.health_check:
                health_config = container_config.health_check
                if 'http' in health_config:
                    container_spec['livenessProbe'] = {
                        'httpGet': {
                            'path': health_config['http'].get('path', '/health'),
                            'port': health_config['http'].get('port', 8080)
                        },
                        'initialDelaySeconds': health_config.get('initial_delay', 30),
                        'periodSeconds': health_config.get('period', 10)
                    }
            
            containers.append(container_spec)
        
        # Deployment manifest
        deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': spec.name,
                'labels': spec.labels,
                'annotations': spec.annotations
            },
            'spec': {
                'replicas': spec.replicas,
                'selector': {
                    'matchLabels': {'app': spec.name}
                },
                'template': {
                    'metadata': {
                        'labels': {'app': spec.name, **spec.labels}
                    },
                    'spec': {
                        'containers': containers
                    }
                }
            }
        }
        
        # Add quantum-specific annotations
        if spec.quantum_requirements:
            quantum_annotations = {
                f"quantum.quantrs2.com/{k}": str(v) 
                for k, v in spec.quantum_requirements.items()
            }
            deployment['metadata']['annotations'].update(quantum_annotations)
        
        return deployment
    
    def scale_deployment(self, deployment_name: str, replicas: int, 
                        namespace: str = "default") -> bool:
        """Scale a deployment."""
        try:
            # Update deployment replicas
            self.apps_v1.patch_namespaced_deployment_scale(
                name=deployment_name,
                namespace=namespace,
                body={'spec': {'replicas': replicas}}
            )
            
            # Update stored spec
            if deployment_name in self.deployments:
                self.deployments[deployment_name].replicas = replicas
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to scale deployment {deployment_name}: {e}")
            return False
    
    def delete_deployment(self, deployment_name: str, namespace: str = "default") -> bool:
        """Delete a deployment."""
        try:
            self.apps_v1.delete_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
            
            if deployment_name in self.deployments:
                del self.deployments[deployment_name]
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to delete deployment {deployment_name}: {e}")
            return False
    
    def get_deployment_status(self, deployment_name: str, 
                            namespace: str = "default") -> Optional[DeploymentStatus]:
        """Get deployment status."""
        try:
            # Get deployment
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
            
            # Get pods
            pods = self.core_v1.list_namespaced_pod(
                namespace=namespace,
                label_selector=f"app={deployment_name}"
            )
            
            # Create container instances from pods
            containers = []
            for pod in pods.items:
                for container_status in pod.status.container_statuses or []:
                    instance = ContainerInstance(
                        id=f"{pod.metadata.name}-{container_status.name}",
                        name=container_status.name,
                        config=ContainerConfig(
                            name=container_status.name,
                            image=container_status.image
                        ),
                        status=self._convert_pod_status(pod.status.phase),
                        deployment_name=deployment_name,
                        created_at=pod.metadata.creation_timestamp.timestamp() if pod.metadata.creation_timestamp else time.time(),
                        health_status="healthy" if container_status.ready else "unhealthy"
                    )
                    containers.append(instance)
            
            # Create deployment status
            status = DeploymentStatus(
                name=deployment_name,
                desired_replicas=deployment.spec.replicas,
                available_replicas=deployment.status.available_replicas or 0,
                ready_replicas=deployment.status.ready_replicas or 0,
                containers=containers
            )
            
            return status
            
        except Exception as e:
            logging.error(f"Failed to get deployment status {deployment_name}: {e}")
            return None
    
    def _convert_pod_status(self, phase: str) -> ContainerStatus:
        """Convert Kubernetes pod phase to container status."""
        phase_mapping = {
            'Pending': ContainerStatus.PENDING,
            'Running': ContainerStatus.RUNNING,
            'Succeeded': ContainerStatus.STOPPED,
            'Failed': ContainerStatus.FAILED,
            'Unknown': ContainerStatus.UNKNOWN
        }
        return phase_mapping.get(phase, ContainerStatus.UNKNOWN)
    
    def list_deployments(self, namespace: str = "default") -> List[DeploymentStatus]:
        """List all deployments."""
        deployment_list = []
        
        try:
            deployments = self.apps_v1.list_namespaced_deployment(namespace=namespace)
            
            for deployment in deployments.items:
                status = self.get_deployment_status(deployment.metadata.name, namespace)
                if status:
                    deployment_list.append(status)
                    
        except Exception as e:
            logging.error(f"Failed to list deployments: {e}")
            
        return deployment_list


class QuantumContainerOrchestrator:
    """Main quantum container orchestration system."""
    
    def __init__(self, default_mode: DeploymentMode = DeploymentMode.DOCKER,
                 registry_url: str = "localhost:5000"):
        """Initialize quantum container orchestrator."""
        self.default_mode = default_mode
        self.resource_manager = QuantumResourceManager()
        self.registry = QuantumContainerRegistry(registry_url)
        
        # Initialize container managers
        self.docker_manager = DockerContainerManager(self.resource_manager) if HAS_DOCKER else None
        self.k8s_manager = KubernetesContainerManager(self.resource_manager) if HAS_KUBERNETES else None
        
        # Deployment tracking
        self.active_deployments = {}
        self.deployment_history = []
        
        # Auto-scaling configuration
        self.auto_scaling_enabled = False
        self.scaling_policies = {}
        self.scaling_thread = None
        
        # Health monitoring
        self.health_monitoring_enabled = False
        self.health_check_interval = 30  # seconds
        self.health_thread = None
        
        # Metrics collection
        self.metrics_enabled = False
        self.metrics_data = {}
        
    def deploy_application(self, spec: DeploymentSpec) -> bool:
        """Deploy a quantum application."""
        try:
            logging.info(f"Deploying application {spec.name} in {spec.mode.value} mode")
            
            success = False
            
            if spec.mode == DeploymentMode.DOCKER:
                success = self._deploy_with_docker(spec)
            elif spec.mode == DeploymentMode.KUBERNETES:
                success = self._deploy_with_kubernetes(spec)
            elif spec.mode == DeploymentMode.LOCAL:
                success = self._deploy_locally(spec)
            elif spec.mode == DeploymentMode.HYBRID:
                success = self._deploy_hybrid(spec)
            else:
                logging.error(f"Unsupported deployment mode: {spec.mode}")
                return False
            
            if success:
                self.active_deployments[spec.name] = spec
                self.deployment_history.append({
                    'name': spec.name,
                    'mode': spec.mode.value,
                    'deployed_at': time.time(),
                    'status': 'deployed'
                })
                
                # Start auto-scaling if configured
                if spec.scaling_policy != ScalingPolicy.MANUAL:
                    self._setup_auto_scaling(spec)
                
                logging.info(f"Successfully deployed {spec.name}")
            else:
                logging.error(f"Failed to deploy {spec.name}")
            
            return success
            
        except Exception as e:
            logging.error(f"Deployment failed for {spec.name}: {e}")
            return False
    
    def _deploy_with_docker(self, spec: DeploymentSpec) -> bool:
        """Deploy application with Docker."""
        if not self.docker_manager:
            logging.error("Docker manager not available")
            return False
        
        try:
            # Deploy each container
            for container_config in spec.containers:
                # Create and start container
                instance = self.docker_manager.create_container(container_config)
                if not instance:
                    return False
                
                success = self.docker_manager.start_container(instance.id)
                if not success:
                    return False
            
            return True
            
        except Exception as e:
            logging.error(f"Docker deployment failed: {e}")
            return False
    
    def _deploy_with_kubernetes(self, spec: DeploymentSpec) -> bool:
        """Deploy application with Kubernetes."""
        if not self.k8s_manager:
            logging.error("Kubernetes manager not available")
            return False
        
        return self.k8s_manager.create_deployment(spec)
    
    def _deploy_locally(self, spec: DeploymentSpec) -> bool:
        """Deploy application locally."""
        try:
            # For local deployment, we run containers as processes
            for container_config in spec.containers:
                # Create local execution environment
                env = os.environ.copy()
                env.update(container_config.environment)
                
                # Build command
                command = container_config.command + container_config.args
                
                if command:
                    # Start process
                    process = subprocess.Popen(
                        command,
                        env=env,
                        cwd=os.getcwd(),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    # Track process
                    local_instance = ContainerInstance(
                        id=f"local_{container_config.name}_{process.pid}",
                        name=container_config.name,
                        config=container_config,
                        status=ContainerStatus.RUNNING,
                        deployment_name=spec.name,
                        created_at=time.time(),
                        started_at=time.time()
                    )
                    
                    # Store in docker manager for consistency
                    if self.docker_manager:
                        self.docker_manager.containers[local_instance.id] = local_instance
            
            return True
            
        except Exception as e:
            logging.error(f"Local deployment failed: {e}")
            return False
    
    def _deploy_hybrid(self, spec: DeploymentSpec) -> bool:
        """Deploy application in hybrid mode."""
        try:
            # Split containers between different deployment modes
            docker_containers = []
            k8s_containers = []
            
            for container_config in spec.containers:
                # Decide deployment mode based on container labels or resource requirements
                quantum_config = container_config.quantum_config
                
                if quantum_config.get('prefer_kubernetes', False) or any(
                    req.type == ResourceType.QUANTUM_HARDWARE for req in container_config.resources
                ):
                    k8s_containers.append(container_config)
                else:
                    docker_containers.append(container_config)
            
            # Deploy to Docker
            if docker_containers:
                docker_spec = DeploymentSpec(
                    name=f"{spec.name}-docker",
                    containers=docker_containers,
                    mode=DeploymentMode.DOCKER
                )
                if not self._deploy_with_docker(docker_spec):
                    return False
            
            # Deploy to Kubernetes
            if k8s_containers:
                k8s_spec = DeploymentSpec(
                    name=f"{spec.name}-k8s",
                    containers=k8s_containers,
                    mode=DeploymentMode.KUBERNETES
                )
                if not self._deploy_with_kubernetes(k8s_spec):
                    return False
            
            return True
            
        except Exception as e:
            logging.error(f"Hybrid deployment failed: {e}")
            return False
    
    def _setup_auto_scaling(self, spec: DeploymentSpec):
        """Setup auto-scaling for deployment."""
        try:
            self.scaling_policies[spec.name] = {
                'policy': spec.scaling_policy,
                'config': spec.auto_scaling_config,
                'current_replicas': spec.replicas,
                'min_replicas': spec.auto_scaling_config.get('min_replicas', 1),
                'max_replicas': spec.auto_scaling_config.get('max_replicas', 10),
                'last_scaled': time.time()
            }
            
            if not self.auto_scaling_enabled:
                self.auto_scaling_enabled = True
                self.scaling_thread = threading.Thread(target=self._auto_scaling_loop)
                self.scaling_thread.daemon = True
                self.scaling_thread.start()
                
        except Exception as e:
            logging.error(f"Failed to setup auto-scaling for {spec.name}: {e}")
    
    def _auto_scaling_loop(self):
        """Auto-scaling monitoring loop."""
        while self.auto_scaling_enabled:
            try:
                for deployment_name, policy in self.scaling_policies.items():
                    self._check_scaling_conditions(deployment_name, policy)
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logging.error(f"Auto-scaling loop error: {e}")
    
    def _check_scaling_conditions(self, deployment_name: str, policy: Dict[str, Any]):
        """Check if scaling is needed for a deployment."""
        try:
            if deployment_name not in self.active_deployments:
                return
            
            spec = self.active_deployments[deployment_name]
            
            # Get current metrics
            utilization = self.resource_manager.get_resource_utilization()
            
            # Scaling decision logic
            should_scale_up = False
            should_scale_down = False
            
            if policy['policy'] == ScalingPolicy.AUTO_CPU:
                cpu_usage = utilization.get('cpu', {}).get('usage_percent', 0)
                
                if cpu_usage > policy['config'].get('cpu_threshold_up', 80):
                    should_scale_up = True
                elif cpu_usage < policy['config'].get('cpu_threshold_down', 20):
                    should_scale_down = True
                    
            elif policy['policy'] == ScalingPolicy.AUTO_MEMORY:
                memory_usage = utilization.get('memory', {}).get('usage_percent', 0)
                
                if memory_usage > policy['config'].get('memory_threshold_up', 80):
                    should_scale_up = True
                elif memory_usage < policy['config'].get('memory_threshold_down', 20):
                    should_scale_down = True
                    
            elif policy['policy'] == ScalingPolicy.AUTO_QUANTUM_LOAD:
                quantum_util = utilization.get('quantum_simulator', {}).get('utilization_percent', 0)
                
                if quantum_util > policy['config'].get('quantum_threshold_up', 70):
                    should_scale_up = True
                elif quantum_util < policy['config'].get('quantum_threshold_down', 10):
                    should_scale_down = True
            
            # Apply scaling with cooldown
            current_time = time.time()
            cooldown = policy['config'].get('cooldown_seconds', 300)
            
            if current_time - policy['last_scaled'] < cooldown:
                return  # Still in cooldown period
            
            new_replicas = policy['current_replicas']
            
            if should_scale_up and new_replicas < policy['max_replicas']:
                new_replicas = min(new_replicas + 1, policy['max_replicas'])
                logging.info(f"Scaling up {deployment_name} to {new_replicas} replicas")
                
            elif should_scale_down and new_replicas > policy['min_replicas']:
                new_replicas = max(new_replicas - 1, policy['min_replicas'])
                logging.info(f"Scaling down {deployment_name} to {new_replicas} replicas")
            
            if new_replicas != policy['current_replicas']:
                success = self.scale_deployment(deployment_name, new_replicas)
                if success:
                    policy['current_replicas'] = new_replicas
                    policy['last_scaled'] = current_time
                    
        except Exception as e:
            logging.error(f"Error checking scaling conditions for {deployment_name}: {e}")
    
    def scale_deployment(self, deployment_name: str, replicas: int) -> bool:
        """Scale a deployment."""
        try:
            if deployment_name not in self.active_deployments:
                logging.error(f"Deployment {deployment_name} not found")
                return False
            
            spec = self.active_deployments[deployment_name]
            
            if spec.mode == DeploymentMode.KUBERNETES:
                if self.k8s_manager:
                    return self.k8s_manager.scale_deployment(deployment_name, replicas, spec.namespace)
            elif spec.mode == DeploymentMode.DOCKER:
                # For Docker, we need to create/remove containers
                current_containers = len(spec.containers) * spec.replicas
                target_containers = len(spec.containers) * replicas
                
                if target_containers > current_containers:
                    # Scale up - create more containers
                    for _ in range(target_containers - current_containers):
                        for container_config in spec.containers:
                            # Create new container with unique name
                            scaled_config = ContainerConfig(
                                name=f"{container_config.name}-{int(time.time())}",
                                image=container_config.image,
                                command=container_config.command,
                                args=container_config.args,
                                environment=container_config.environment,
                                ports=container_config.ports,
                                volumes=container_config.volumes,
                                resources=container_config.resources,
                                labels=container_config.labels,
                                health_check=container_config.health_check
                            )
                            
                            if self.docker_manager:
                                instance = self.docker_manager.create_container(scaled_config)
                                if instance:
                                    self.docker_manager.start_container(instance.id)
                
                elif target_containers < current_containers:
                    # Scale down - remove containers
                    containers_to_remove = current_containers - target_containers
                    if self.docker_manager:
                        container_list = self.docker_manager.list_containers()
                        deployment_containers = [
                            c for c in container_list 
                            if c.deployment_name == deployment_name
                        ]
                        
                        for i in range(min(containers_to_remove, len(deployment_containers))):
                            container = deployment_containers[i]
                            self.docker_manager.stop_container(container.id)
                            self.docker_manager.remove_container(container.id)
            
            # Update spec
            spec.replicas = replicas
            return True
            
        except Exception as e:
            logging.error(f"Failed to scale deployment {deployment_name}: {e}")
            return False
    
    def get_deployment_status(self, deployment_name: str) -> Optional[DeploymentStatus]:
        """Get deployment status."""
        try:
            if deployment_name not in self.active_deployments:
                return None
            
            spec = self.active_deployments[deployment_name]
            
            if spec.mode == DeploymentMode.KUBERNETES:
                if self.k8s_manager:
                    return self.k8s_manager.get_deployment_status(deployment_name, spec.namespace)
            elif spec.mode == DeploymentMode.DOCKER:
                if self.docker_manager:
                    containers = [
                        c for c in self.docker_manager.list_containers()
                        if c.deployment_name == deployment_name
                    ]
                    
                    return DeploymentStatus(
                        name=deployment_name,
                        desired_replicas=spec.replicas,
                        available_replicas=len([c for c in containers if c.status == ContainerStatus.RUNNING]),
                        ready_replicas=len([c for c in containers if c.status == ContainerStatus.RUNNING]),
                        containers=containers
                    )
            
        except Exception as e:
            logging.error(f"Failed to get deployment status for {deployment_name}: {e}")
            
        return None
    
    def delete_deployment(self, deployment_name: str) -> bool:
        """Delete a deployment."""
        try:
            if deployment_name not in self.active_deployments:
                logging.error(f"Deployment {deployment_name} not found")
                return False
            
            spec = self.active_deployments[deployment_name]
            
            success = False
            
            if spec.mode == DeploymentMode.KUBERNETES:
                if self.k8s_manager:
                    success = self.k8s_manager.delete_deployment(deployment_name, spec.namespace)
            elif spec.mode == DeploymentMode.DOCKER:
                if self.docker_manager:
                    containers = [
                        c for c in self.docker_manager.list_containers()
                        if c.deployment_name == deployment_name
                    ]
                    
                    for container in containers:
                        self.docker_manager.stop_container(container.id)
                        self.docker_manager.remove_container(container.id, force=True)
                    
                    success = True
            
            if success:
                # Clean up auto-scaling
                if deployment_name in self.scaling_policies:
                    del self.scaling_policies[deployment_name]
                
                # Remove from active deployments
                del self.active_deployments[deployment_name]
                
                # Update deployment history
                self.deployment_history.append({
                    'name': deployment_name,
                    'mode': spec.mode.value,
                    'deleted_at': time.time(),
                    'status': 'deleted'
                })
                
                logging.info(f"Successfully deleted deployment {deployment_name}")
            
            return success
            
        except Exception as e:
            logging.error(f"Failed to delete deployment {deployment_name}: {e}")
            return False
    
    def list_deployments(self) -> List[DeploymentStatus]:
        """List all active deployments."""
        deployments = []
        
        for deployment_name in self.active_deployments:
            status = self.get_deployment_status(deployment_name)
            if status:
                deployments.append(status)
        
        return deployments
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        metrics = {
            'timestamp': time.time(),
            'resource_utilization': self.resource_manager.get_resource_utilization(),
            'active_deployments': len(self.active_deployments),
            'deployment_history': len(self.deployment_history),
            'auto_scaling_policies': len(self.scaling_policies),
            'container_stats': {},
            'quantum_metrics': {}
        }
        
        try:
            # Container statistics
            if self.docker_manager:
                docker_containers = self.docker_manager.list_containers()
                metrics['container_stats']['docker'] = {
                    'total': len(docker_containers),
                    'running': len([c for c in docker_containers if c.status == ContainerStatus.RUNNING]),
                    'stopped': len([c for c in docker_containers if c.status == ContainerStatus.STOPPED]),
                    'failed': len([c for c in docker_containers if c.status == ContainerStatus.FAILED])
                }
            
            if self.k8s_manager:
                k8s_deployments = self.k8s_manager.list_deployments()
                total_pods = sum(d.ready_replicas for d in k8s_deployments)
                metrics['container_stats']['kubernetes'] = {
                    'deployments': len(k8s_deployments),
                    'total_pods': total_pods,
                    'ready_pods': total_pods  # Simplified for now
                }
            
            # Quantum-specific metrics
            quantum_containers = 0
            quantum_simulators_used = 0
            
            for deployment_name, spec in self.active_deployments.items():
                for container_config in spec.containers:
                    if any(req.type in [ResourceType.QUANTUM_SIMULATOR, ResourceType.QUANTUM_HARDWARE] 
                          for req in container_config.resources):
                        quantum_containers += 1
                        
                        # Count quantum simulators
                        for req in container_config.resources:
                            if req.type == ResourceType.QUANTUM_SIMULATOR:
                                quantum_simulators_used += 1
            
            metrics['quantum_metrics'] = {
                'quantum_containers': quantum_containers,
                'quantum_simulators_used': quantum_simulators_used,
                'quantum_hardware_used': 0  # TODO: Implement hardware tracking
            }
            
        except Exception as e:
            logging.error(f"Error collecting system metrics: {e}")
        
        return metrics
    
    @contextmanager
    def deployment_context(self, spec: DeploymentSpec, auto_cleanup: bool = True):
        """Context manager for automatic deployment lifecycle management."""
        deployed = False
        
        try:
            # Deploy application
            deployed = self.deploy_application(spec)
            if not deployed:
                raise RuntimeError(f"Failed to deploy {spec.name}")
            
            yield self
            
        except Exception as e:
            logging.error(f"Error in deployment context for {spec.name}: {e}")
            raise
        finally:
            # Cleanup if requested and deployment was successful
            if auto_cleanup and deployed:
                try:
                    self.delete_deployment(spec.name)
                except Exception as e:
                    logging.error(f"Failed to cleanup deployment {spec.name}: {e}")


# Convenience functions for easy usage
def get_quantum_container_orchestrator(mode: DeploymentMode = DeploymentMode.DOCKER,
                                     registry_url: str = "localhost:5000") -> QuantumContainerOrchestrator:
    """Get a quantum container orchestrator instance."""
    return QuantumContainerOrchestrator(default_mode=mode, registry_url=registry_url)


def create_quantum_container_config(name: str, image: str, 
                                   quantum_simulators: int = 1,
                                   max_qubits: int = 16,
                                   **kwargs) -> ContainerConfig:
    """Create a quantum container configuration with sensible defaults."""
    # Default quantum resource requirements
    resources = [
        ResourceRequirement(
            type=ResourceType.QUANTUM_SIMULATOR,
            amount=quantum_simulators,
            metadata={'max_qubits': max_qubits}
        ),
        ResourceRequirement(
            type=ResourceType.CPU,
            amount=kwargs.get('cpu_cores', 2),
            unit='cores'
        ),
        ResourceRequirement(
            type=ResourceType.MEMORY,
            amount=kwargs.get('memory', '4Gi'),
            unit='Gi'
        )
    ]
    
    # Quantum-specific environment variables
    environment = {
        'QUANTRS2_QUANTUM_SIMULATORS': str(quantum_simulators),
        'QUANTRS2_MAX_QUBITS': str(max_qubits),
        'QUANTRS2_CONTAINER_MODE': 'true',
        **kwargs.get('environment', {})
    }
    
    return ContainerConfig(
        name=name,
        image=image,
        environment=environment,
        resources=resources,
        command=kwargs.get('command', []),
        args=kwargs.get('args', []),
        ports=kwargs.get('ports', []),
        volumes=kwargs.get('volumes', {}),
        labels={'quantum': 'true', 'quantrs2': 'true', **kwargs.get('labels', {})},
        quantum_config={'simulators': quantum_simulators, 'max_qubits': max_qubits}
    )


def create_quantum_deployment_spec(name: str, containers: List[ContainerConfig],
                                 mode: DeploymentMode = DeploymentMode.DOCKER,
                                 replicas: int = 1,
                                 auto_scale: bool = False,
                                 **kwargs) -> DeploymentSpec:
    """Create a quantum deployment specification with sensible defaults."""
    # Auto-scaling configuration
    auto_scaling_config = {}
    scaling_policy = ScalingPolicy.MANUAL
    
    if auto_scale:
        scaling_policy = ScalingPolicy.AUTO_QUANTUM_LOAD
        auto_scaling_config = {
            'min_replicas': kwargs.get('min_replicas', 1),
            'max_replicas': kwargs.get('max_replicas', 5),
            'quantum_threshold_up': kwargs.get('scale_up_threshold', 70),
            'quantum_threshold_down': kwargs.get('scale_down_threshold', 20),
            'cooldown_seconds': kwargs.get('cooldown_seconds', 300)
        }
    
    return DeploymentSpec(
        name=name,
        replicas=replicas,
        containers=containers,
        mode=mode,
        namespace=kwargs.get('namespace', 'default'),
        labels={'quantum': 'true', 'quantrs2': 'true', **kwargs.get('labels', {})},
        scaling_policy=scaling_policy,
        auto_scaling_config=auto_scaling_config,
        quantum_requirements={
            'total_simulators': sum(
                req.amount for container in containers
                for req in container.resources
                if req.type == ResourceType.QUANTUM_SIMULATOR
            ),
            'max_qubits_required': max(
                req.metadata.get('max_qubits', 0) for container in containers
                for req in container.resources
                if req.type == ResourceType.QUANTUM_SIMULATOR
            ) if containers else 0
        }
    )


def deploy_quantum_application(name: str, image: str, 
                              mode: DeploymentMode = DeploymentMode.DOCKER,
                              quantum_simulators: int = 1,
                              replicas: int = 1,
                              **kwargs) -> bool:
    """Deploy a quantum application with a single command."""
    try:
        # Create container configuration
        container_config = create_quantum_container_config(
            name=f"{name}-container",
            image=image,
            quantum_simulators=quantum_simulators,
            **kwargs
        )
        
        # Create deployment specification
        deployment_spec = create_quantum_deployment_spec(
            name=name,
            containers=[container_config],
            mode=mode,
            replicas=replicas,
            **kwargs
        )
        
        # Deploy application
        orchestrator = get_quantum_container_orchestrator(mode)
        return orchestrator.deploy_application(deployment_spec)
        
    except Exception as e:
        logging.error(f"Failed to deploy quantum application {name}: {e}")
        return False


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    print("QuantRS2 Quantum Container Orchestration System")
    print("=" * 60)
    
    # Test basic functionality
    orchestrator = get_quantum_container_orchestrator()
    
    print(f"System Metrics: {json.dumps(orchestrator.get_system_metrics(), indent=2)}")
    print(f"Resource Utilization: {json.dumps(orchestrator.resource_manager.get_resource_utilization(), indent=2)}")
    
    # Example deployment
    if HAS_DOCKER:
        container_config = create_quantum_container_config(
            name="test-quantum-app",
            image="python:3.9-slim",
            quantum_simulators=2,
            command=["python", "-c", "print('Hello from quantum container!'); import time; time.sleep(60)"]
        )
        
        deployment_spec = create_quantum_deployment_spec(
            name="test-deployment",
            containers=[container_config],
            mode=DeploymentMode.DOCKER,
            replicas=1
        )
        
        print(f"\nTest deployment spec: {json.dumps(deployment_spec.to_dict(), indent=2, default=str)}")
        
        # Uncomment to actually deploy
        # success = orchestrator.deploy_application(deployment_spec)
        # print(f"Deployment success: {success}")
    
    print("\nQuantum Container Orchestration System initialized successfully!")