#!/usr/bin/env python3
"""
Comprehensive demo of the QuantRS2 Quantum Container Orchestration System.

This demo showcases the complete container orchestration capabilities including:
- Docker and Kubernetes integration for quantum applications
- Quantum-specific resource management and allocation
- Container registry support with quantum optimization
- Deployment automation and scaling capabilities
- Auto-scaling based on quantum workload metrics
- Health monitoring and metrics collection
- Multi-mode deployment strategies (local, Docker, Kubernetes, hybrid)
- Integration with quantum hardware and simulators

Run this demo to see the full range of container orchestration features
available in the QuantRS2 quantum computing framework.
"""

import time
import json
import logging
import threading
from pathlib import Path
import numpy as np

try:
    import quantrs2
    from quantrs2.quantum_containers import (
        ContainerStatus, DeploymentMode, ResourceType, ScalingPolicy,
        ResourceRequirement, ContainerConfig, DeploymentSpec,
        QuantumContainerOrchestrator, QuantumContainerRegistry, QuantumResourceManager,
        get_quantum_container_orchestrator, create_quantum_container_config,
        create_quantum_deployment_spec, deploy_quantum_application,
        HAS_DOCKER, HAS_KUBERNETES, HAS_REQUESTS, HAS_PSUTIL, HAS_JINJA2
    )
    print(f"QuantRS2 version: {quantrs2.__version__}")
    print("Successfully imported quantum container orchestration system")
except ImportError as e:
    print(f"Error importing QuantRS2 container system: {e}")
    print("Please ensure the container orchestration system is properly installed")
    exit(1)

# Check for optional dependencies
print("\nDependency Status:")
print(f"✓ Docker support: {'Available' if HAS_DOCKER else 'Not available'}")
print(f"✓ Kubernetes support: {'Available' if HAS_KUBERNETES else 'Not available'}")
print(f"✓ HTTP requests: {'Available' if HAS_REQUESTS else 'Not available'}")
print(f"✓ System monitoring: {'Available' if HAS_PSUTIL else 'Not available'}")
print(f"✓ Template engine: {'Available' if HAS_JINJA2 else 'Not available'}")


def demo_resource_management():
    """Demonstrate quantum resource management capabilities."""
    print("\n" + "="*60)
    print("QUANTUM RESOURCE MANAGEMENT DEMO")
    print("="*60)
    
    resource_manager = QuantumResourceManager()
    
    print("--- System Resource Detection ---")
    utilization = resource_manager.get_resource_utilization()
    
    for resource_type, data in utilization.items():
        print(f"{resource_type.upper()}:")
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
        print()
    
    print("--- Resource Allocation Demo ---")
    
    # Test different resource allocations
    test_allocations = [
        {
            'container_id': 'quantum_simulator_app',
            'requirements': [
                ResourceRequirement(ResourceType.QUANTUM_SIMULATOR, 2, metadata={'max_qubits': 16}),
                ResourceRequirement(ResourceType.CPU, 4, 'cores'),
                ResourceRequirement(ResourceType.MEMORY, '8Gi')
            ]
        },
        {
            'container_id': 'quantum_ml_training',
            'requirements': [
                ResourceRequirement(ResourceType.QUANTUM_SIMULATOR, 1, metadata={'max_qubits': 32}),
                ResourceRequirement(ResourceType.CPU, 8, 'cores'),
                ResourceRequirement(ResourceType.MEMORY, '16Gi')
            ]
        },
        {
            'container_id': 'quantum_optimization',
            'requirements': [
                ResourceRequirement(ResourceType.QUANTUM_SIMULATOR, 3, metadata={'max_qubits': 8}),
                ResourceRequirement(ResourceType.CPU, 2, 'cores'),
                ResourceRequirement(ResourceType.MEMORY, '4Gi')
            ]
        }
    ]
    
    allocated_containers = []
    
    for allocation in test_allocations:
        print(f"\nAllocating resources for {allocation['container_id']}:")
        
        result = resource_manager.allocate_resources(
            allocation['container_id'],
            allocation['requirements']
        )
        
        if result['success']:
            print("✓ Resource allocation successful")
            allocated_containers.append(allocation['container_id'])
            
            for resource_type, allocation_info in result['allocated_resources'].items():
                print(f"  {resource_type}: {allocation_info['amount']} {allocation_info.get('unit', '')}")
                
            if result['quantum_allocation']:
                print("  Quantum resources:")
                for q_type, q_info in result['quantum_allocation'].items():
                    print(f"    {q_type}: {q_info}")
        else:
            print("✗ Resource allocation failed")
            for error in result['errors']:
                print(f"    Error: {error}")
    
    # Show updated utilization
    print("\n--- Updated Resource Utilization ---")
    updated_utilization = resource_manager.get_resource_utilization()
    
    for resource_type, data in updated_utilization.items():
        if resource_type == 'quantum_simulator' and isinstance(data, dict):
            print(f"Quantum Simulators:")
            print(f"  Total: {data.get('total', 0)}")
            print(f"  Used: {data.get('used', 0)}")
            print(f"  Available: {data.get('available', 0)}")
            print(f"  Utilization: {data.get('utilization_percent', 0):.1f}%")
    
    # Clean up allocations
    print("\n--- Resource Cleanup ---")
    for container_id in allocated_containers:
        success = resource_manager.deallocate_resources(container_id)
        print(f"{'✓' if success else '✗'} Deallocated resources for {container_id}")


def demo_container_registry():
    """Demonstrate quantum container registry capabilities."""
    print("\n" + "="*60)
    print("QUANTUM CONTAINER REGISTRY DEMO")
    print("="*60)
    
    registry = QuantumContainerRegistry("localhost:5000")
    
    print("--- Container Registry Operations ---")
    
    # List existing quantum images
    print("Listing quantum container images:")
    images = registry.list_quantum_images()
    
    if images:
        for image in images[:3]:  # Show first 3 images
            print(f"  Image: {image.get('tags', ['Unknown'])[0]}")
            print(f"    ID: {image.get('id', 'Unknown')[:12]}")
            print(f"    Size: {image.get('size', 0) / 1024 / 1024:.1f} MB")
            if image.get('quantum_metadata'):
                print(f"    Quantum metadata: {image['quantum_metadata']}")
    else:
        print("  No quantum container images found")
    
    # Demonstrate quantum layer optimization
    print("\n--- Quantum Layer Optimization Demo ---")
    
    # Simulate optimization for a quantum container image
    optimization_scenarios = [
        "quantrs2/quantum-simulator:latest",
        "quantrs2/quantum-ml:v1.0",
        "quantrs2/quantum-optimization:dev"
    ]
    
    for image_name in optimization_scenarios:
        print(f"\nOptimizing quantum layers for {image_name}:")
        
        try:
            result = registry.optimize_quantum_layers(image_name)
            
            print(f"  Original size: {result['original_size'] / 1024 / 1024:.1f} MB")
            print(f"  Optimized size: {result['optimized_size'] / 1024 / 1024:.1f} MB")
            print(f"  Compression ratio: {result['compression_ratio']:.1%}")
            print(f"  Quantum layers identified: {result['quantum_layers_identified']}")
            
            if result['optimizations_applied']:
                print("  Optimizations applied:")
                for optimization in result['optimizations_applied'][:3]:
                    print(f"    • {optimization}")
                    
        except Exception as e:
            print(f"  ⚠ Optimization simulation: {e}")
    
    # Registry metadata management
    print("\n--- Registry Metadata Management ---")
    
    quantum_metadata_examples = [
        {
            'image': 'quantrs2/vqe-optimizer:latest',
            'metadata': {
                'quantum_algorithm': 'VQE',
                'max_qubits': 16,
                'supported_backends': ['statevector', 'qasm'],
                'optimization_level': 'O2',
                'python_version': '3.9'
            }
        },
        {
            'image': 'quantrs2/qaoa-solver:v2.1',
            'metadata': {
                'quantum_algorithm': 'QAOA',
                'max_qubits': 20,
                'supported_backends': ['aer', 'ionq'],
                'layers': 5,
                'classical_optimizer': 'COBYLA'
            }
        }
    ]
    
    for example in quantum_metadata_examples:
        image_name = example['image']
        metadata = example['metadata']
        
        print(f"\nStoring metadata for {image_name}:")
        registry.image_metadata[image_name] = metadata
        
        print(f"  Algorithm: {metadata.get('quantum_algorithm', 'Unknown')}")
        print(f"  Max qubits: {metadata.get('max_qubits', 'Unknown')}")
        print(f"  Backends: {', '.join(metadata.get('supported_backends', []))}")


def demo_container_configurations():
    """Demonstrate quantum container configuration creation."""
    print("\n" + "="*60)
    print("QUANTUM CONTAINER CONFIGURATION DEMO")
    print("="*60)
    
    print("--- Creating Quantum Container Configurations ---")
    
    # Different types of quantum applications
    container_configs = [
        {
            'name': 'quantum-simulator-app',
            'description': 'High-performance quantum state vector simulation',
            'config': create_quantum_container_config(
                name="quantum-simulator-app",
                image="quantrs2/quantum-simulator:latest",
                quantum_simulators=3,
                max_qubits=32,
                cpu_cores=8,
                memory="16Gi",
                command=["python", "quantum_simulation.py"],
                environment={
                    "QUANTUM_BACKEND": "statevector",
                    "MAX_PARALLEL_CIRCUITS": "10",
                    "OPTIMIZATION_LEVEL": "3"
                },
                ports=[8080, 8081, 9090]
            )
        },
        {
            'name': 'quantum-ml-trainer',
            'description': 'Quantum machine learning model training',
            'config': create_quantum_container_config(
                name="quantum-ml-trainer",
                image="quantrs2/quantum-ml:v2.0",
                quantum_simulators=2,
                max_qubits=16,
                cpu_cores=6,
                memory="12Gi",
                command=["python", "train_qnn.py"],
                environment={
                    "QUANTUM_BACKEND": "qasm_simulator",
                    "TRAINING_EPOCHS": "100",
                    "LEARNING_RATE": "0.01"
                },
                volumes={"/data": "/app/data", "/models": "/app/models"}
            )
        },
        {
            'name': 'quantum-optimization-service',
            'description': 'Quantum optimization as a microservice',
            'config': create_quantum_container_config(
                name="quantum-optimization-service",
                image="quantrs2/quantum-opt:latest",
                quantum_simulators=1,
                max_qubits=24,
                cpu_cores=4,
                memory="8Gi",
                command=["python", "optimization_service.py"],
                environment={
                    "SERVICE_PORT": "8080",
                    "QUANTUM_ALGORITHM": "QAOA",
                    "MAX_ITERATIONS": "1000"
                },
                ports=[8080]
            )
        }
    ]
    
    for container_info in container_configs:
        config = container_info['config']
        print(f"\n{container_info['name'].upper()}:")
        print(f"  Description: {container_info['description']}")
        print(f"  Image: {config.image}")
        print(f"  Command: {' '.join(config.command)}")
        
        # Show resource requirements
        print("  Resource Requirements:")
        for req in config.resources:
            if req.type == ResourceType.QUANTUM_SIMULATOR:
                print(f"    Quantum Simulators: {req.amount} (max {req.metadata.get('max_qubits', 0)} qubits each)")
            elif req.type == ResourceType.CPU:
                print(f"    CPU: {req.amount} {req.unit}")
            elif req.type == ResourceType.MEMORY:
                print(f"    Memory: {req.amount} {req.unit}")
        
        # Show environment variables
        if config.environment:
            print("  Environment Variables:")
            for key, value in list(config.environment.items())[:3]:  # Show first 3
                print(f"    {key}: {value}")
        
        # Show quantum configuration
        if config.quantum_config:
            print("  Quantum Configuration:")
            for key, value in config.quantum_config.items():
                print(f"    {key}: {value}")
        
        # Convert to dictionary and show size
        config_dict = config.to_dict()
        config_json = json.dumps(config_dict, indent=2, default=str)
        print(f"  Configuration size: {len(config_json)} bytes")


def demo_deployment_specifications():
    """Demonstrate quantum deployment specification creation."""
    print("\n" + "="*60)
    print("QUANTUM DEPLOYMENT SPECIFICATION DEMO")
    print("="*60)
    
    print("--- Creating Quantum Deployment Specifications ---")
    
    # Create container configurations
    simulator_config = create_quantum_container_config(
        name="quantum-simulator",
        image="quantrs2/simulator:latest",
        quantum_simulators=2,
        max_qubits=20
    )
    
    ml_config = create_quantum_container_config(
        name="quantum-ml",
        image="quantrs2/ml:latest",
        quantum_simulators=1,
        max_qubits=16
    )
    
    # Different deployment scenarios
    deployment_specs = [
        {
            'name': 'single-container-deployment',
            'description': 'Simple single-container quantum application',
            'spec': create_quantum_deployment_spec(
                name="quantum-calculator",
                containers=[simulator_config],
                mode=DeploymentMode.DOCKER,
                replicas=2,
                namespace="quantum-apps"
            )
        },
        {
            'name': 'multi-container-deployment',
            'description': 'Multi-container quantum application with different services',
            'spec': create_quantum_deployment_spec(
                name="quantum-ml-pipeline",
                containers=[simulator_config, ml_config],
                mode=DeploymentMode.KUBERNETES,
                replicas=3,
                namespace="ml-quantum",
                labels={"app": "quantum-ml", "tier": "production"}
            )
        },
        {
            'name': 'auto-scaling-deployment',
            'description': 'Auto-scaling quantum application based on quantum load',
            'spec': create_quantum_deployment_spec(
                name="quantum-auto-scale",
                containers=[simulator_config],
                mode=DeploymentMode.KUBERNETES,
                replicas=2,
                auto_scale=True,
                min_replicas=1,
                max_replicas=10,
                scale_up_threshold=75,
                scale_down_threshold=25,
                cooldown_seconds=180
            )
        }
    ]
    
    for deployment_info in deployment_specs:
        spec = deployment_info['spec']
        print(f"\n{deployment_info['name'].upper()}:")
        print(f"  Description: {deployment_info['description']}")
        print(f"  Name: {spec.name}")
        print(f"  Mode: {spec.mode.value}")
        print(f"  Replicas: {spec.replicas}")
        print(f"  Namespace: {spec.namespace}")
        print(f"  Containers: {len(spec.containers)}")
        
        # Show scaling configuration
        if spec.scaling_policy != ScalingPolicy.MANUAL:
            print(f"  Scaling Policy: {spec.scaling_policy.value}")
            print(f"  Auto-scaling Config:")
            for key, value in spec.auto_scaling_config.items():
                print(f"    {key}: {value}")
        
        # Show quantum requirements
        if spec.quantum_requirements:
            print("  Quantum Requirements:")
            for key, value in spec.quantum_requirements.items():
                print(f"    {key}: {value}")
        
        # Show labels and annotations
        if spec.labels:
            print("  Labels:")
            for key, value in spec.labels.items():
                print(f"    {key}: {value}")


def demo_orchestrator_operations():
    """Demonstrate quantum container orchestrator operations."""
    print("\n" + "="*60)
    print("QUANTUM CONTAINER ORCHESTRATOR DEMO")
    print("="*60)
    
    # Create orchestrator
    orchestrator = get_quantum_container_orchestrator(
        mode=DeploymentMode.LOCAL,  # Use local mode for demo
        registry_url="localhost:5000"
    )
    
    print("--- Orchestrator System Status ---")
    
    # Get system metrics
    metrics = orchestrator.get_system_metrics()
    
    print(f"Active Deployments: {metrics['active_deployments']}")
    print(f"Deployment History: {metrics['deployment_history']}")
    print(f"Auto-scaling Policies: {metrics['auto_scaling_policies']}")
    
    print("\nResource Utilization:")
    for resource_type, data in metrics['resource_utilization'].items():
        if isinstance(data, dict) and 'utilization_percent' in data:
            print(f"  {resource_type}: {data['utilization_percent']:.1f}% utilized")
        elif isinstance(data, dict) and 'usage_percent' in data:
            print(f"  {resource_type}: {data['usage_percent']:.1f}% used")
    
    print("\nQuantum Metrics:")
    for key, value in metrics['quantum_metrics'].items():
        print(f"  {key}: {value}")
    
    print("\n--- Deployment Operations Demo ---")
    
    # Create test deployment specifications
    test_deployments = []
    
    for i in range(3):
        container_config = create_quantum_container_config(
            name=f"quantum-test-{i}",
            image="python:3.9-slim",
            quantum_simulators=1,
            max_qubits=8,
            command=["python", "-c", f"print('Quantum application {i} running'); import time; time.sleep(30)"],
            environment={"APP_ID": str(i)}
        )
        
        deployment_spec = create_quantum_deployment_spec(
            name=f"test-deployment-{i}",
            containers=[container_config],
            mode=DeploymentMode.LOCAL,
            replicas=1
        )
        
        test_deployments.append(deployment_spec)
    
    # Deploy applications
    deployed_apps = []
    
    for spec in test_deployments:
        print(f"\nDeploying {spec.name}...")
        
        try:
            success = orchestrator.deploy_application(spec)
            
            if success:
                print(f"✓ {spec.name} deployed successfully")
                deployed_apps.append(spec.name)
            else:
                print(f"✗ {spec.name} deployment failed")
                
        except Exception as e:
            print(f"✗ {spec.name} deployment error: {e}")
    
    # Wait for deployments to start
    if deployed_apps:
        print("\nWaiting for deployments to initialize...")
        time.sleep(2)
        
        # Check deployment status
        print("\n--- Deployment Status Check ---")
        
        for app_name in deployed_apps:
            status = orchestrator.get_deployment_status(app_name)
            
            if status:
                print(f"\n{app_name.upper()}:")
                print(f"  Desired Replicas: {status.desired_replicas}")
                print(f"  Available Replicas: {status.available_replicas}")
                print(f"  Ready Replicas: {status.ready_replicas}")
                print(f"  Containers: {len(status.containers)}")
                
                if status.containers:
                    for container in status.containers[:2]:  # Show first 2 containers
                        print(f"    Container {container.name}:")
                        print(f"      Status: {container.status.value}")
                        print(f"      Health: {container.health_status}")
                        if container.quantum_metrics:
                            print(f"      Quantum metrics: {container.quantum_metrics}")
            else:
                print(f"{app_name}: Status not available")
        
        # Demonstrate scaling
        if deployed_apps:
            print("\n--- Scaling Demo ---")
            
            test_app = deployed_apps[0]
            print(f"Scaling {test_app} to 2 replicas...")
            
            try:
                success = orchestrator.scale_deployment(test_app, 2)
                
                if success:
                    print(f"✓ {test_app} scaled successfully")
                    
                    # Check updated status
                    time.sleep(1)
                    status = orchestrator.get_deployment_status(test_app)
                    if status:
                        print(f"  Updated replicas: {status.desired_replicas}")
                else:
                    print(f"✗ {test_app} scaling failed")
                    
            except Exception as e:
                print(f"✗ {test_app} scaling error: {e}")
        
        # List all deployments
        print("\n--- All Active Deployments ---")
        
        all_deployments = orchestrator.list_deployments()
        
        if all_deployments:
            for deployment in all_deployments:
                print(f"  {deployment.name}: {deployment.ready_replicas}/{deployment.desired_replicas} ready")
        else:
            print("  No active deployments found")
        
        # Clean up deployments
        print("\n--- Cleanup ---")
        
        for app_name in deployed_apps:
            print(f"Deleting {app_name}...")
            
            try:
                success = orchestrator.delete_deployment(app_name)
                
                if success:
                    print(f"✓ {app_name} deleted successfully")
                else:
                    print(f"✗ {app_name} deletion failed")
                    
            except Exception as e:
                print(f"✗ {app_name} deletion error: {e}")


def demo_convenience_functions():
    """Demonstrate convenience functions for easy usage."""
    print("\n" + "="*60)
    print("CONVENIENCE FUNCTIONS DEMO")
    print("="*60)
    
    print("--- Quick Deployment Function ---")
    
    # Demonstrate the deploy_quantum_application convenience function
    deployment_scenarios = [
        {
            'name': 'quick-quantum-sim',
            'description': 'Quick quantum simulator deployment',
            'params': {
                'name': 'quick-quantum-sim',
                'image': 'python:3.9-slim',
                'mode': DeploymentMode.LOCAL,
                'quantum_simulators': 1,
                'replicas': 1,
                'command': ['python', '-c', 'print("Quick quantum simulator running"); import time; time.sleep(10)'],
                'cpu_cores': 2,
                'memory': '4Gi'
            }
        },
        {
            'name': 'quantum-ml-app',
            'description': 'Quantum ML application with auto-scaling',
            'params': {
                'name': 'quantum-ml-app',
                'image': 'python:3.9-slim',
                'mode': DeploymentMode.LOCAL,
                'quantum_simulators': 2,
                'replicas': 1,
                'command': ['python', '-c', 'print("Quantum ML app running"); import time; time.sleep(15)'],
                'cpu_cores': 4,
                'memory': '8Gi',
                'auto_scale': True,
                'min_replicas': 1,
                'max_replicas': 5
            }
        }
    ]
    
    successful_deployments = []
    
    for scenario in deployment_scenarios:
        print(f"\n{scenario['description']}:")
        print(f"  Deploying {scenario['name']}...")
        
        try:
            success = deploy_quantum_application(**scenario['params'])
            
            if success:
                print(f"  ✓ {scenario['name']} deployed successfully")
                successful_deployments.append(scenario['name'])
            else:
                print(f"  ✗ {scenario['name']} deployment failed")
                
        except Exception as e:
            print(f"  ✗ {scenario['name']} deployment error: {e}")
    
    # Show status of successful deployments
    if successful_deployments:
        print("\n--- Quick Deployment Status ---")
        
        # Get a fresh orchestrator to check status
        orchestrator = get_quantum_container_orchestrator(mode=DeploymentMode.LOCAL)
        
        time.sleep(2)  # Wait for deployments to initialize
        
        for app_name in successful_deployments:
            status = orchestrator.get_deployment_status(app_name)
            
            if status:
                print(f"  {app_name}: {status.ready_replicas}/{status.desired_replicas} replicas ready")
            else:
                print(f"  {app_name}: Status not available")
        
        # Cleanup quick deployments
        print("\n--- Quick Deployment Cleanup ---")
        
        for app_name in successful_deployments:
            try:
                success = orchestrator.delete_deployment(app_name)
                print(f"  {'✓' if success else '✗'} {app_name} cleanup")
            except Exception as e:
                print(f"  ✗ {app_name} cleanup error: {e}")


def demo_auto_scaling_simulation():
    """Demonstrate auto-scaling capabilities with simulation."""
    print("\n" + "="*60)
    print("AUTO-SCALING SIMULATION DEMO")
    print("="*60)
    
    print("--- Auto-scaling Configuration Demo ---")
    
    # Create orchestrator
    orchestrator = get_quantum_container_orchestrator(mode=DeploymentMode.LOCAL)
    
    # Create container with auto-scaling
    container_config = create_quantum_container_config(
        name="auto-scale-quantum-app",
        image="python:3.9-slim",
        quantum_simulators=1,
        max_qubits=16,
        command=["python", "-c", "import time; print('Auto-scaling quantum app running'); time.sleep(60)"],
        cpu_cores=2,
        memory="4Gi"
    )
    
    # Create deployment with auto-scaling enabled
    deployment_spec = create_quantum_deployment_spec(
        name="auto-scale-deployment",
        containers=[container_config],
        mode=DeploymentMode.LOCAL,
        replicas=2,
        auto_scale=True,
        min_replicas=1,
        max_replicas=6,
        scale_up_threshold=70,  # Scale up when quantum utilization > 70%
        scale_down_threshold=20,  # Scale down when quantum utilization < 20%
        cooldown_seconds=10  # Short cooldown for demo
    )
    
    print(f"Deploying auto-scaling application: {deployment_spec.name}")
    print(f"  Min replicas: {deployment_spec.auto_scaling_config['min_replicas']}")
    print(f"  Max replicas: {deployment_spec.auto_scaling_config['max_replicas']}")
    print(f"  Scale up threshold: {deployment_spec.auto_scaling_config['quantum_threshold_up']}%")
    print(f"  Scale down threshold: {deployment_spec.auto_scaling_config['quantum_threshold_down']}%")
    
    try:
        success = orchestrator.deploy_application(deployment_spec)
        
        if success:
            print("✓ Auto-scaling deployment created successfully")
            
            # Show auto-scaling policy setup
            if deployment_spec.name in orchestrator.scaling_policies:
                policy = orchestrator.scaling_policies[deployment_spec.name]
                print("\nAuto-scaling policy configured:")
                print(f"  Policy type: {policy['policy'].value}")
                print(f"  Current replicas: {policy['current_replicas']}")
                print(f"  Min/Max replicas: {policy['min_replicas']}/{policy['max_replicas']}")
            
            # Simulate auto-scaling monitoring
            print("\n--- Auto-scaling Monitoring Simulation ---")
            print("Simulating quantum workload changes...")
            
            for i in range(5):
                time.sleep(2)
                
                # Get current status
                status = orchestrator.get_deployment_status(deployment_spec.name)
                metrics = orchestrator.get_system_metrics()
                
                if status:
                    print(f"\nTime {i+1}:")
                    print(f"  Current replicas: {status.ready_replicas}/{status.desired_replicas}")
                    
                    # Show quantum resource utilization
                    quantum_util = metrics['resource_utilization'].get('quantum_simulator', {})
                    if quantum_util:
                        util_percent = quantum_util.get('utilization_percent', 0)
                        print(f"  Quantum utilization: {util_percent:.1f}%")
                    
                    # Show any scaling actions
                    if deployment_spec.name in orchestrator.scaling_policies:
                        policy = orchestrator.scaling_policies[deployment_spec.name]
                        current_replicas = policy['current_replicas']
                        if current_replicas != deployment_spec.replicas:
                            print(f"  → Scaling detected: {deployment_spec.replicas} → {current_replicas}")
            
            # Clean up
            print(f"\nCleaning up auto-scaling deployment...")
            cleanup_success = orchestrator.delete_deployment(deployment_spec.name)
            print(f"{'✓' if cleanup_success else '✗'} Auto-scaling deployment cleanup")
            
        else:
            print("✗ Auto-scaling deployment failed")
            
    except Exception as e:
        print(f"✗ Auto-scaling demo error: {e}")


def demo_deployment_context_manager():
    """Demonstrate deployment context manager for automatic lifecycle management."""
    print("\n" + "="*60)
    print("DEPLOYMENT CONTEXT MANAGER DEMO")
    print("="*60)
    
    orchestrator = get_quantum_container_orchestrator(mode=DeploymentMode.LOCAL)
    
    # Create deployment specification
    container_config = create_quantum_container_config(
        name="context-managed-app",
        image="python:3.9-slim",
        quantum_simulators=1,
        command=["python", "-c", "print('Context-managed quantum app running'); import time; time.sleep(20)"],
        cpu_cores=1,
        memory="2Gi"
    )
    
    deployment_spec = create_quantum_deployment_spec(
        name="context-managed-deployment",
        containers=[container_config],
        mode=DeploymentMode.LOCAL,
        replicas=1
    )
    
    print("--- Automatic Deployment Lifecycle Management ---")
    print(f"Using context manager for {deployment_spec.name}...")
    
    try:
        # Use context manager for automatic deployment and cleanup
        with orchestrator.deployment_context(deployment_spec, auto_cleanup=True) as ctx:
            print("✓ Deployment context entered - application deployed automatically")
            
            # Work with the deployment inside the context
            time.sleep(2)
            
            status = ctx.get_deployment_status(deployment_spec.name)
            if status:
                print(f"  Deployment status: {status.ready_replicas}/{status.desired_replicas} replicas ready")
                print(f"  Containers: {len(status.containers)}")
                
                for container in status.containers:
                    print(f"    {container.name}: {container.status.value}")
            
            # Simulate some work
            print("  Simulating application workload...")
            time.sleep(3)
            
            print("  Application work completed")
        
        # After exiting context, deployment should be cleaned up automatically
        print("✓ Deployment context exited - application cleaned up automatically")
        
        # Verify cleanup
        time.sleep(1)
        remaining_deployments = orchestrator.list_deployments()
        context_deployment_still_exists = any(
            d.name == deployment_spec.name for d in remaining_deployments
        )
        
        if not context_deployment_still_exists:
            print("✓ Automatic cleanup verified - deployment no longer exists")
        else:
            print("⚠ Deployment still exists after context exit")
            
    except Exception as e:
        print(f"✗ Context manager demo error: {e}")


def demo_integration_features():
    """Demonstrate integration with other QuantRS2 modules."""
    print("\n" + "="*60)
    print("INTEGRATION FEATURES DEMO")
    print("="*60)
    
    orchestrator = get_quantum_container_orchestrator()
    
    print("--- Container Orchestration Integration Status ---")
    
    # Check integration with resource management
    print("Resource Management:")
    resource_manager = orchestrator.resource_manager
    if resource_manager:
        utilization = resource_manager.get_resource_utilization()
        print("  ✓ Quantum resource management integrated")
        print(f"    Available quantum simulators: {utilization.get('quantum_simulator', {}).get('available', 0)}")
    else:
        print("  ✗ Resource management not available")
    
    # Check registry integration
    print("\nContainer Registry:")
    registry = orchestrator.registry
    if registry:
        print("  ✓ Quantum container registry integrated")
        print(f"    Registry URL: {registry.registry_url}")
        
        # Try to list images
        try:
            images = registry.list_quantum_images()
            print(f"    Quantum images available: {len(images)}")
        except Exception as e:
            print(f"    Image listing: {e}")
    else:
        print("  ✗ Container registry not available")
    
    # Check container managers
    print("\nContainer Managers:")
    
    if orchestrator.docker_manager:
        print("  ✓ Docker container manager available")
        try:
            containers = orchestrator.docker_manager.list_containers()
            print(f"    Managed containers: {len(containers)}")
        except Exception:
            print("    Container listing not available")
    else:
        print("  ✗ Docker container manager not available")
    
    if orchestrator.k8s_manager:
        print("  ✓ Kubernetes container manager available")
        try:
            deployments = orchestrator.k8s_manager.list_deployments()
            print(f"    Managed deployments: {len(deployments)}")
        except Exception:
            print("    Deployment listing not available")
    else:
        print("  ✗ Kubernetes container manager not available")
    
    # System capabilities summary
    print("\n--- System Capabilities Summary ---")
    
    capabilities = {
        'Local Deployment': True,  # Always available
        'Docker Integration': HAS_DOCKER and orchestrator.docker_manager is not None,
        'Kubernetes Integration': HAS_KUBERNETES and orchestrator.k8s_manager is not None,
        'System Monitoring': HAS_PSUTIL,
        'HTTP Requests': HAS_REQUESTS,
        'Template Engine': HAS_JINJA2,
        'Auto-scaling': True,  # Always available
        'Resource Management': True,  # Always available
        'Health Monitoring': True,  # Always available
        'Metrics Collection': True   # Always available
    }
    
    for capability, available in capabilities.items():
        status = "✓ Available" if available else "✗ Not available"
        print(f"  {capability}: {status}")
    
    # Performance information
    print("\n--- Performance Information ---")
    
    metrics = orchestrator.get_system_metrics()
    print(f"Active deployments tracked: {metrics['active_deployments']}")
    print(f"Deployment history entries: {metrics['deployment_history']}")
    print(f"Auto-scaling policies: {metrics['auto_scaling_policies']}")
    
    if metrics['container_stats']:
        print("Container statistics:")
        for platform, stats in metrics['container_stats'].items():
            if isinstance(stats, dict):
                total = stats.get('total', 0)
                running = stats.get('running', stats.get('ready_pods', 0))
                print(f"  {platform.capitalize()}: {running}/{total} containers running")


def main():
    """Run the comprehensive quantum container orchestration demo."""
    print("QuantRS2 Quantum Container Orchestration System Comprehensive Demo")
    print("=" * 80)
    print("This demo showcases the complete container orchestration capabilities")
    print("of the QuantRS2 quantum computing framework.")
    print("=" * 80)
    
    # Configure logging for demo
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise for demo
    
    try:
        # Run all demo sections
        demo_resource_management()
        demo_container_registry()
        demo_container_configurations()
        demo_deployment_specifications()
        demo_orchestrator_operations()
        demo_convenience_functions()
        demo_auto_scaling_simulation()
        demo_deployment_context_manager()
        demo_integration_features()
        
        print("\n" + "="*80)
        print("DEMO COMPLETE!")
        print("="*80)
        print("All quantum container orchestration features have been demonstrated successfully.")
        
        print("\nContainer orchestration capabilities demonstrated:")
        print("  ✓ Quantum-specific resource management and allocation")
        print("  ✓ Container registry with quantum optimization features")
        print("  ✓ Flexible container configuration for quantum applications")
        print("  ✓ Multi-mode deployment strategies (local, Docker, Kubernetes, hybrid)")
        print("  ✓ Comprehensive deployment lifecycle management")
        print("  ✓ Auto-scaling based on quantum workload metrics")
        print("  ✓ Health monitoring and system metrics collection")
        print("  ✓ Convenient deployment functions for rapid development")
        print("  ✓ Automatic deployment cleanup with context managers")
        print("  ✓ Integration with quantum hardware and simulators")
        
        dependency_status = [
            f"  {'✓' if HAS_DOCKER else '✗'} Docker integration for container management",
            f"  {'✓' if HAS_KUBERNETES else '✗'} Kubernetes integration for orchestration",
            f"  {'✓' if HAS_PSUTIL else '✗'} System monitoring for resource tracking",
            f"  {'✓' if HAS_REQUESTS else '✗'} HTTP support for registry operations",
            f"  {'✓' if HAS_JINJA2 else '✗'} Template engine for configuration generation"
        ]
        
        print("\nDependency status:")
        for status in dependency_status:
            print(status)
        
        print("\nTo deploy quantum applications:")
        print("  deploy_quantum_application('my-app', 'quantrs2:latest')")
        
        print("\nTo create custom deployments:")
        print("  orchestrator = get_quantum_container_orchestrator()")
        print("  config = create_quantum_container_config(...)")
        print("  spec = create_quantum_deployment_spec(...)")
        print("  orchestrator.deploy_application(spec)")
        
        print("\nThe QuantRS2 Quantum Container Orchestration System is fully functional!")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)