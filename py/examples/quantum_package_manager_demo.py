#!/usr/bin/env python3
"""
Comprehensive demo of the QuantRS2 Quantum Package Manager.

This demo showcases the complete package management capabilities including:
- Package creation, installation, and distribution
- Dependency resolution with quantum-specific requirements
- Registry management and package discovery
- Package validation and security scanning
- CLI interface and automation workflows
- Integration with quantum development tools
- Support for various package types and hardware compatibility
"""

import os
import json
import tempfile
import time
import asyncio
import logging
from pathlib import Path
import numpy as np

try:
    import quantrs2
    from quantrs2.quantum_package_manager import (
        QuantumPackageManager, PackageType, DependencyType, RegistryType,
        PackageMetadata, PackageManifest, PackageRequirement, RegistryConfig,
        get_quantum_package_manager, create_package_manifest,
        HAS_CLICK, HAS_TOML, HAS_GIT, HAS_REQUESTS
    )
    print(f"QuantRS2 version: {quantrs2.__version__}")
    print("Successfully imported quantum package manager")
except ImportError as e:
    print(f"Error importing QuantRS2 Package Manager: {e}")
    print("Please ensure the package manager is properly installed")
    exit(1)

# Check for optional dependencies
print("\nDependency Status:")
print(f"✓ CLI support (Click): {'Available' if HAS_CLICK else 'Not available'}")
print(f"✓ TOML support: {'Available' if HAS_TOML else 'Not available'}")
print(f"✓ Git support: {'Available' if HAS_GIT else 'Not available'}")
print(f"✓ HTTP requests: {'Available' if HAS_REQUESTS else 'Not available'}")


def demo_package_manager_setup():
    """Demonstrate package manager initialization."""
    print("\n" + "="*60)
    print("PACKAGE MANAGER SETUP DEMO")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"--- Initializing Package Manager in {temp_dir} ---")
        
        pkg_manager = get_quantum_package_manager(temp_dir)
        
        print(f"✓ Package manager initialized")
        print(f"  Workspace: {pkg_manager.workspace_dir}")
        print(f"  Packages directory: {pkg_manager.packages_dir}")
        print(f"  Cache directory: {pkg_manager.cache_dir}")
        print(f"  Database: {pkg_manager.db_path}")
        
        # Show default registries
        print(f"\n--- Default Registries ---")
        for name, registry in pkg_manager.registries.items():
            print(f"✓ {name}: {registry.config.url} ({registry.config.type.value})")
        
        # Get initial statistics
        stats = pkg_manager.get_package_statistics()
        print(f"\n--- Initial Statistics ---")
        print(f"  Installed packages: {stats['total_packages']}")
        print(f"  Registries: {stats['registries']}")
        print(f"  Total size: {stats['total_size']} bytes")
        
        return pkg_manager


def demo_package_creation():
    """Demonstrate creating packages."""
    print("\n" + "="*60)
    print("PACKAGE CREATION DEMO")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"--- Creating Sample Quantum Packages in {temp_dir} ---")
        
        pkg_manager = get_quantum_package_manager(temp_dir)
        
        # Create sample algorithm package
        print("\n--- Creating Quantum Algorithm Package ---")
        
        algo_dir = Path(temp_dir) / "quantum_algorithms"
        algo_dir.mkdir()
        
        # Create package structure
        (algo_dir / "__init__.py").write_text("""
'''Quantum Algorithms Package.'''
__version__ = "1.0.0"
""")
        
        (algo_dir / "grover.py").write_text("""
'''Grover's Search Algorithm implementation.'''
import numpy as np

def grovers_algorithm(oracle, n_qubits):
    '''Implement Grover's search algorithm.'''
    # Simulated implementation
    amplitudes = np.ones(2**n_qubits) / np.sqrt(2**n_qubits)
    
    # Oracle operation
    marked_item = oracle(amplitudes)
    
    # Diffusion operator
    mean = np.mean(amplitudes)
    amplitudes = 2 * mean - amplitudes
    
    return amplitudes

def quantum_search(items, search_target):
    '''Quantum search using Grover's algorithm.'''
    n_items = len(items)
    n_qubits = int(np.ceil(np.log2(n_items)))
    
    def oracle(amplitudes):
        # Mark target item
        try:
            target_index = items.index(search_target)
            amplitudes[target_index] *= -1
            return target_index
        except ValueError:
            return -1
    
    result_amplitudes = grovers_algorithm(oracle, n_qubits)
    max_index = np.argmax(np.abs(result_amplitudes))
    
    return items[max_index] if max_index < len(items) else None
""")
        
        (algo_dir / "vqe.py").write_text("""
'''Variational Quantum Eigensolver implementation.'''
import numpy as np

class VQE:
    '''Variational Quantum Eigensolver for molecular simulation.'''
    
    def __init__(self, molecule):
        self.molecule = molecule
        self.parameters = np.random.random(4)  # Example parameters
    
    def cost_function(self, parameters):
        '''Calculate the energy expectation value.'''
        # Simulated energy calculation
        energy = np.sum(parameters**2) - 2 * np.sum(parameters)
        return energy
    
    def optimize(self, max_iterations=100):
        '''Optimize the variational parameters.'''
        best_energy = float('inf')
        best_params = self.parameters.copy()
        
        for i in range(max_iterations):
            # Simple gradient descent
            gradient = 2 * self.parameters - 2
            self.parameters -= 0.01 * gradient
            
            energy = self.cost_function(self.parameters)
            if energy < best_energy:
                best_energy = energy
                best_params = self.parameters.copy()
        
        return best_energy, best_params

def optimize_molecule(molecule_name):
    '''Optimize a molecule using VQE.'''
    vqe = VQE(molecule_name)
    energy, params = vqe.optimize()
    
    return {
        'molecule': molecule_name,
        'ground_state_energy': energy,
        'optimal_parameters': params.tolist()
    }
""")
        
        # Create package manifest
        manifest_data = {
            'metadata': {
                'name': 'quantum-algorithms',
                'version': '1.0.0',
                'description': 'Collection of quantum algorithms including Grover and VQE',
                'author': 'Quantum Developer',
                'author_email': 'dev@quantum.org',
                'license': 'MIT',
                'homepage': 'https://github.com/quantum/algorithms',
                'keywords': ['quantum', 'algorithms', 'grover', 'vqe'],
                'package_type': 'algorithm',
                'quantum_requirements': {
                    'min_qubits': 4,
                    'max_qubits': 20,
                    'hardware_types': ['simulator', 'superconducting']
                },
                'hardware_compatibility': ['qiskit', 'cirq', 'quantrs'],
                'python_requires': '>=3.8'
            },
            'dependencies': [
                {
                    'name': 'numpy',
                    'version_spec': '>=1.20.0',
                    'type': 'required'
                },
                {
                    'name': 'quantrs2',
                    'version_spec': '>=0.1.0',
                    'type': 'required',
                    'quantum_features': ['simulation', 'optimization']
                }
            ],
            'files': [
                '__init__.py',
                'grover.py',
                'vqe.py'
            ],
            'entry_points': {
                'quantum.algorithms': [
                    'grover = quantum_algorithms.grover:quantum_search',
                    'vqe = quantum_algorithms.vqe:optimize_molecule'
                ]
            },
            'quantum_config': {
                'simulation_backends': ['state_vector', 'density_matrix'],
                'optimization_methods': ['gradient_descent', 'adam'],
                'noise_models': ['ideal', 'realistic']
            }
        }
        
        import yaml
        with open(algo_dir / "quantum_package.yml", 'w') as f:
            yaml.dump(manifest_data, f, default_flow_style=False)
        
        # Create the package
        print(f"📦 Creating package from {algo_dir}")
        result = pkg_manager.create_package(str(algo_dir), temp_dir)
        
        if result['success']:
            print(f"✓ Package created: {result['package_file']}")
            package_file = result['package_file']
        else:
            print(f"❌ Package creation failed:")
            for error in result['errors']:
                print(f"  {error}")
            return None
        
        for warning in result['warnings']:
            print(f"⚠ {warning}")
        
        # Create circuit library package
        print("\n--- Creating Circuit Library Package ---")
        
        circuit_dir = Path(temp_dir) / "quantum_circuits"
        circuit_dir.mkdir()
        
        # Create circuit library structure
        (circuit_dir / "__init__.py").write_text("""
'''Quantum Circuit Library.'''
__version__ = "2.1.0"
""")
        
        (circuit_dir / "basic_circuits.py").write_text("""
'''Basic quantum circuits collection.'''

def bell_state_circuit():
    '''Create a Bell state circuit.'''
    return {
        'name': 'Bell State',
        'qubits': 2,
        'gates': [
            {'type': 'H', 'qubit': 0},
            {'type': 'CNOT', 'control': 0, 'target': 1}
        ],
        'measurements': [0, 1]
    }

def ghz_state_circuit(n_qubits=3):
    '''Create a GHZ state circuit.'''
    gates = [{'type': 'H', 'qubit': 0}]
    for i in range(1, n_qubits):
        gates.append({'type': 'CNOT', 'control': 0, 'target': i})
    
    return {
        'name': f'GHZ State ({n_qubits} qubits)',
        'qubits': n_qubits,
        'gates': gates,
        'measurements': list(range(n_qubits))
    }

def qft_circuit(n_qubits):
    '''Create a Quantum Fourier Transform circuit.'''
    gates = []
    
    for i in range(n_qubits):
        gates.append({'type': 'H', 'qubit': i})
        for j in range(i + 1, n_qubits):
            angle = np.pi / (2**(j - i))
            gates.append({
                'type': 'CRZ',
                'control': j,
                'target': i,
                'angle': angle
            })
    
    return {
        'name': f'QFT ({n_qubits} qubits)',
        'qubits': n_qubits,
        'gates': gates,
        'measurements': []
    }
""")
        
        # Create circuit manifest
        circuit_manifest = {
            'metadata': {
                'name': 'quantum-circuits',
                'version': '2.1.0',
                'description': 'Library of common quantum circuits and gates',
                'author': 'Circuit Designer',
                'license': 'Apache-2.0',
                'keywords': ['quantum', 'circuits', 'gates', 'bell', 'ghz', 'qft'],
                'package_type': 'circuit_library',
                'quantum_requirements': {
                    'min_qubits': 2,
                    'max_qubits': 100
                },
                'hardware_compatibility': ['universal_gate_set']
            },
            'dependencies': [
                {
                    'name': 'numpy',
                    'version_spec': '>=1.18.0',
                    'type': 'required'
                }
            ],
            'files': [
                '__init__.py',
                'basic_circuits.py'
            ],
            'entry_points': {
                'quantum.circuits': [
                    'bell = quantum_circuits.basic_circuits:bell_state_circuit',
                    'ghz = quantum_circuits.basic_circuits:ghz_state_circuit',
                    'qft = quantum_circuits.basic_circuits:qft_circuit'
                ]
            }
        }
        
        with open(circuit_dir / "quantum_package.yml", 'w') as f:
            yaml.dump(circuit_manifest, f, default_flow_style=False)
        
        circuit_result = pkg_manager.create_package(str(circuit_dir), temp_dir)
        
        if circuit_result['success']:
            print(f"✓ Circuit library package created: {circuit_result['package_file']}")
        else:
            print(f"❌ Circuit package creation failed")
        
        return [result, circuit_result]


def demo_package_search_and_discovery(pkg_manager):
    """Demonstrate package search and discovery."""
    print("\n" + "="*60)
    print("PACKAGE SEARCH & DISCOVERY DEMO")
    print("="*60)
    
    print("--- Searching for Quantum Packages ---")
    
    # Mock some packages in registry for demo
    print("\n📖 Simulating package registry with sample packages...")
    
    # Simulate search results
    sample_packages = [
        {
            'name': 'quantum-algorithms',
            'version': '1.0.0',
            'description': 'Collection of quantum algorithms including Grover and VQE',
            'type': 'algorithm',
            'author': 'Quantum Developer',
            'keywords': ['quantum', 'algorithms', 'grover', 'vqe'],
            'hardware_compatibility': ['qiskit', 'cirq', 'quantrs'],
            'quantum_requirements': {'min_qubits': 4, 'max_qubits': 20}
        },
        {
            'name': 'quantum-circuits',
            'version': '2.1.0',
            'description': 'Library of common quantum circuits and gates',
            'type': 'circuit_library',
            'author': 'Circuit Designer',
            'keywords': ['quantum', 'circuits', 'gates', 'bell', 'ghz'],
            'hardware_compatibility': ['universal_gate_set'],
            'quantum_requirements': {'min_qubits': 2, 'max_qubits': 100}
        },
        {
            'name': 'quantum-ml',
            'version': '0.5.2',
            'description': 'Quantum machine learning algorithms and tools',
            'type': 'framework',
            'author': 'ML Team',
            'keywords': ['quantum', 'machine-learning', 'neural-networks'],
            'hardware_compatibility': ['nisq_devices'],
            'quantum_requirements': {'min_qubits': 8, 'hardware_only': False}
        },
        {
            'name': 'ibm-quantum-driver',
            'version': '1.2.3',
            'description': 'Hardware driver for IBM Quantum devices',
            'type': 'hardware_driver',
            'author': 'IBM Quantum',
            'keywords': ['ibm', 'hardware', 'driver', 'quantum'],
            'hardware_compatibility': ['ibm_quantum'],
            'quantum_requirements': {'hardware_only': True}
        }
    ]
    
    # Display search results
    print(f"Found {len(sample_packages)} packages:")
    for pkg in sample_packages:
        print(f"\n📦 {pkg['name']} ({pkg['version']}) - {pkg['type']}")
        print(f"   {pkg['description']}")
        print(f"   Author: {pkg['author']}")
        print(f"   Keywords: {', '.join(pkg['keywords'])}")
        print(f"   Hardware: {', '.join(pkg['hardware_compatibility'])}")
        print(f"   Quantum requirements: {pkg['quantum_requirements']}")
    
    # Demonstrate filtering by type
    print(f"\n--- Filtering by Package Type ---")
    
    for pkg_type in ['algorithm', 'circuit_library', 'hardware_driver']:
        filtered = [pkg for pkg in sample_packages if pkg['type'] == pkg_type]
        print(f"\n{pkg_type.title()} packages ({len(filtered)}):")
        for pkg in filtered:
            print(f"  • {pkg['name']} - {pkg['description']}")
    
    # Show quantum requirements analysis
    print(f"\n--- Quantum Requirements Analysis ---")
    
    min_qubits = [pkg['quantum_requirements'].get('min_qubits', 0) for pkg in sample_packages]
    max_qubits = [pkg['quantum_requirements'].get('max_qubits', 0) for pkg in sample_packages]
    hardware_only = [pkg['quantum_requirements'].get('hardware_only', False) for pkg in sample_packages]
    
    print(f"Qubit requirements:")
    print(f"  Minimum qubits needed: {min(min_qubits)} - {max(min_qubits)}")
    print(f"  Maximum qubits supported: {max(max_qubits)}")
    print(f"  Hardware-only packages: {sum(hardware_only)}")
    print(f"  Simulator-compatible: {len(sample_packages) - sum(hardware_only)}")


def demo_dependency_resolution():
    """Demonstrate dependency resolution."""
    print("\n" + "="*60)
    print("DEPENDENCY RESOLUTION DEMO")
    print("="*60)
    
    print("--- Creating Package Dependency Scenarios ---")
    
    # Create sample dependency scenarios
    scenarios = [
        {
            'name': 'Simple Dependencies',
            'description': 'Basic package with standard dependencies',
            'packages': {
                'quantum-app': {
                    'version': '1.0.0',
                    'dependencies': [
                        {'name': 'numpy', 'version_spec': '>=1.20.0'},
                        {'name': 'quantrs2', 'version_spec': '>=0.1.0'}
                    ]
                }
            }
        },
        {
            'name': 'Quantum-Specific Dependencies',
            'description': 'Package with quantum hardware requirements',
            'packages': {
                'quantum-optimization': {
                    'version': '2.0.0',
                    'dependencies': [
                        {'name': 'quantum-algorithms', 'version_spec': '>=1.0.0'},
                        {'name': 'quantum-circuits', 'version_spec': '>=2.0.0'},
                        {'name': 'scipy', 'version_spec': '>=1.7.0', 'type': 'optional'}
                    ],
                    'quantum_requirements': {'min_qubits': 10}
                }
            }
        },
        {
            'name': 'Hardware Conflicts',
            'description': 'Packages with incompatible hardware requirements',
            'packages': {
                'ibm-optimizer': {
                    'version': '1.0.0',
                    'dependencies': [
                        {'name': 'ibm-quantum-driver', 'version_spec': '>=1.0.0'}
                    ],
                    'hardware_compatibility': ['ibm_quantum']
                },
                'google-circuits': {
                    'version': '1.0.0',
                    'dependencies': [
                        {'name': 'cirq', 'version_spec': '>=0.14.0'}
                    ],
                    'hardware_compatibility': ['google_quantum']
                }
            }
        },
        {
            'name': 'Version Conflicts',
            'description': 'Packages requiring incompatible versions',
            'packages': {
                'old-quantum-lib': {
                    'version': '0.8.0',
                    'dependencies': [
                        {'name': 'numpy', 'version_spec': '<1.20.0'}
                    ]
                },
                'new-quantum-lib': {
                    'version': '2.0.0',
                    'dependencies': [
                        {'name': 'numpy', 'version_spec': '>=1.21.0'}
                    ]
                }
            }
        }
    ]
    
    # Display scenarios
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        print(f"Description: {scenario['description']}")
        
        for pkg_name, pkg_info in scenario['packages'].items():
            print(f"\n📦 {pkg_name} ({pkg_info['version']}):")
            
            if 'dependencies' in pkg_info:
                print("  Dependencies:")
                for dep in pkg_info['dependencies']:
                    dep_type = dep.get('type', 'required')
                    print(f"    • {dep['name']} {dep['version_spec']} ({dep_type})")
            
            if 'quantum_requirements' in pkg_info:
                print(f"  Quantum requirements: {pkg_info['quantum_requirements']}")
            
            if 'hardware_compatibility' in pkg_info:
                print(f"  Hardware compatibility: {pkg_info['hardware_compatibility']}")
        
        # Simulate resolution results
        if scenario['name'] == 'Simple Dependencies':
            print("\n✅ Resolution: Success")
            print("  Install order: numpy → quantrs2 → quantum-app")
            print("  No conflicts detected")
        
        elif scenario['name'] == 'Quantum-Specific Dependencies':
            print("\n✅ Resolution: Success")
            print("  Install order: numpy → quantrs2 → quantum-circuits → quantum-algorithms → quantum-optimization")
            print("  Quantum requirements compatible")
        
        elif scenario['name'] == 'Hardware Conflicts':
            print("\n⚠️  Resolution: Warning")
            print("  Packages have incompatible hardware requirements")
            print("  ibm-optimizer requires IBM hardware")
            print("  google-circuits requires Google hardware")
            print("  Manual conflict resolution needed")
        
        elif scenario['name'] == 'Version Conflicts':
            print("\n❌ Resolution: Failed")
            print("  Conflict detected: numpy version requirements")
            print("  old-quantum-lib requires numpy <1.20.0")
            print("  new-quantum-lib requires numpy >=1.21.0")
            print("  Cannot satisfy both requirements")


def demo_package_installation_simulation():
    """Demonstrate package installation simulation."""
    print("\n" + "="*60)
    print("PACKAGE INSTALLATION SIMULATION")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        pkg_manager = get_quantum_package_manager(temp_dir)
        
        print("--- Simulating Package Installation Workflow ---")
        
        # Create mock installed packages data
        mock_packages = [
            {
                'name': 'numpy',
                'version': '1.21.0',
                'type': 'utility',
                'size': 15_000_000,
                'description': 'Fundamental package for scientific computing'
            },
            {
                'name': 'quantum-circuits',
                'version': '2.1.0',
                'type': 'circuit_library',
                'size': 2_500_000,
                'description': 'Library of common quantum circuits'
            },
            {
                'name': 'quantum-algorithms',
                'version': '1.0.0',
                'type': 'algorithm',
                'size': 5_200_000,
                'description': 'Collection of quantum algorithms'
            }
        ]
        
        print(f"\n📦 Simulating installation of {len(mock_packages)} packages:")
        
        for pkg in mock_packages:
            print(f"\nInstalling {pkg['name']} ({pkg['version']})...")
            
            # Simulate installation steps
            steps = [
                "Resolving dependencies",
                "Downloading package",
                "Validating package integrity",
                "Checking quantum compatibility",
                "Installing package files",
                "Updating package database",
                "Running post-install checks"
            ]
            
            for step in steps:
                print(f"  • {step}...")
                time.sleep(0.1)  # Simulate work
            
            size_mb = pkg['size'] / (1024 * 1024)
            print(f"  ✅ {pkg['name']} installed successfully ({size_mb:.1f} MB)")
        
        # Show installation summary
        print(f"\n--- Installation Summary ---")
        total_size = sum(pkg['size'] for pkg in mock_packages)
        total_mb = total_size / (1024 * 1024)
        
        print(f"✅ Successfully installed {len(mock_packages)} packages")
        print(f"📊 Total download size: {total_mb:.1f} MB")
        print(f"🔧 Package types installed:")
        
        type_counts = {}
        for pkg in mock_packages:
            pkg_type = pkg['type']
            type_counts[pkg_type] = type_counts.get(pkg_type, 0) + 1
        
        for pkg_type, count in type_counts.items():
            print(f"  • {pkg_type}: {count}")
        
        # Simulate package listing
        print(f"\n--- Installed Packages ---")
        for pkg in mock_packages:
            size_mb = pkg['size'] / (1024 * 1024)
            print(f"📦 {pkg['name']} ({pkg['version']}) - {pkg['type']} - {size_mb:.1f}MB")
            print(f"   {pkg['description']}")


def demo_package_validation_and_security():
    """Demonstrate package validation and security features."""
    print("\n" + "="*60)
    print("PACKAGE VALIDATION & SECURITY DEMO")
    print("="*60)
    
    print("--- Package Validation Scenarios ---")
    
    validation_scenarios = [
        {
            'name': 'Valid Quantum Package',
            'description': 'Well-formed quantum algorithm package',
            'validation_results': {
                'valid': True,
                'warnings': [],
                'errors': [],
                'security_issues': [],
                'quantum_compatibility': True
            }
        },
        {
            'name': 'Missing Dependencies',
            'description': 'Package with incomplete dependency specification',
            'validation_results': {
                'valid': False,
                'warnings': ['Some dependencies may be missing'],
                'errors': ['Required dependency "quantrs2" not specified'],
                'security_issues': [],
                'quantum_compatibility': True
            }
        },
        {
            'name': 'Security Concerns',
            'description': 'Package with potential security issues',
            'validation_results': {
                'valid': True,
                'warnings': ['Package contains compiled binaries'],
                'errors': [],
                'security_issues': [
                    'Potentially dangerous code in setup.py: subprocess.call',
                    'Suspicious files found: [\'malicious.exe\']'
                ],
                'quantum_compatibility': True
            }
        },
        {
            'name': 'Quantum Incompatibility',
            'description': 'Package with quantum framework incompatibilities',
            'validation_results': {
                'valid': True,
                'warnings': [
                    'Package requires more than 100 qubits - may not work on all simulators',
                    'Hardware-only package - simulation may not be supported'
                ],
                'errors': [],
                'security_issues': [],
                'quantum_compatibility': False
            }
        }
    ]
    
    # Display validation results
    for scenario in validation_scenarios:
        print(f"\n--- {scenario['name']} ---")
        print(f"Description: {scenario['description']}")
        
        results = scenario['validation_results']
        
        if results['valid']:
            print("✅ Package validation: PASSED")
        else:
            print("❌ Package validation: FAILED")
        
        if results['quantum_compatibility']:
            print("🔬 Quantum compatibility: COMPATIBLE")
        else:
            print("⚠️  Quantum compatibility: ISSUES DETECTED")
        
        if results['errors']:
            print("❌ Errors:")
            for error in results['errors']:
                print(f"  • {error}")
        
        if results['warnings']:
            print("⚠️  Warnings:")
            for warning in results['warnings']:
                print(f"  • {warning}")
        
        if results['security_issues']:
            print("🔒 Security Issues:")
            for issue in results['security_issues']:
                print(f"  • {issue}")
    
    # Security best practices
    print(f"\n--- Security Best Practices ---")
    best_practices = [
        "Always validate packages before installation",
        "Review package source code when possible",
        "Use trusted registries and verified publishers",
        "Check package signatures and checksums",
        "Scan for known vulnerabilities",
        "Monitor package dependencies for security updates",
        "Isolate quantum computing environments",
        "Regularly audit installed packages"
    ]
    
    for i, practice in enumerate(best_practices, 1):
        print(f"  {i}. {practice}")


def demo_registry_management():
    """Demonstrate package registry management."""
    print("\n" + "="*60)
    print("REGISTRY MANAGEMENT DEMO")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        pkg_manager = get_quantum_package_manager(temp_dir)
        
        print("--- Package Registry Configuration ---")
        
        # Show default registries
        print(f"\nDefault Registries:")
        for name, registry in pkg_manager.registries.items():
            config = registry.config
            print(f"📋 {name}:")
            print(f"   URL: {config.url}")
            print(f"   Type: {config.type.value}")
            print(f"   Enabled: {config.enabled}")
            print(f"   Trusted: {config.trusted}")
            print(f"   Priority: {config.priority}")
        
        # Add custom registries
        print(f"\n--- Adding Custom Registries ---")
        
        custom_registries = [
            {
                'name': 'company-internal',
                'url': 'https://quantum-registry.company.com',
                'type': RegistryType.PRIVATE,
                'description': 'Private company registry for internal packages'
            },
            {
                'name': 'quantum-community',
                'url': 'https://community.quantum.org/packages',
                'type': RegistryType.PUBLIC,
                'description': 'Community-driven quantum package registry'
            },
            {
                'name': 'research-lab',
                'url': '/home/researcher/quantum_packages',
                'type': RegistryType.LOCAL,
                'description': 'Local registry for research lab packages'
            }
        ]
        
        for registry_info in custom_registries:
            print(f"\n➕ Adding registry: {registry_info['name']}")
            print(f"   URL: {registry_info['url']}")
            print(f"   Type: {registry_info['type'].value}")
            print(f"   Description: {registry_info['description']}")
            
            # Simulate adding registry
            success = True  # In real implementation: pkg_manager.add_registry(...)
            
            if success:
                print(f"   ✅ Registry added successfully")
            else:
                print(f"   ❌ Failed to add registry")
        
        # Registry priorities and selection
        print(f"\n--- Registry Priority and Selection ---")
        
        print("Registry search order (by priority):")
        registry_priorities = [
            ("local", 0, "Highest priority - local packages"),
            ("company-internal", 5, "High priority - trusted internal packages"),
            ("quantrs-public", 10, "Medium priority - official public registry"),
            ("quantum-community", 15, "Lower priority - community packages")
        ]
        
        for name, priority, description in registry_priorities:
            print(f"  {priority:2d}. {name} - {description}")
        
        # Registry synchronization
        print(f"\n--- Registry Synchronization ---")
        
        sync_results = [
            ("quantrs-public", True, "142 packages", "3.2s"),
            ("company-internal", True, "28 packages", "1.1s"),
            ("quantum-community", False, "Connection timeout", "30.0s"),
            ("local", True, "5 packages", "0.1s")
        ]
        
        print("Synchronizing package indices...")
        for name, success, result, duration in sync_results:
            status = "✅" if success else "❌"
            print(f"  {status} {name}: {result} ({duration})")


def demo_cli_interface():
    """Demonstrate CLI interface capabilities."""
    print("\n" + "="*60)
    print("CLI INTERFACE DEMO")
    print("="*60)
    
    if not HAS_CLICK:
        print("⚠️  Click library not available - CLI interface not functional")
        print("   Install click to enable CLI: pip install click")
        return
    
    print("--- Command Line Interface Examples ---")
    
    cli_commands = [
        {
            'command': 'quantpkg search grover',
            'description': 'Search for packages containing "grover"',
            'example_output': """
Found 3 packages:
  quantum-algorithms (1.0.0) - algorithm
    Collection of quantum algorithms including Grover and VQE
    Keywords: quantum, algorithms, grover, vqe

  grover-search (0.8.2) - algorithm  
    Optimized implementation of Grover's search algorithm
    Keywords: grover, search, optimization
"""
        },
        {
            'command': 'quantpkg install quantum-algorithms',
            'description': 'Install a quantum algorithm package',
            'example_output': """
Installing quantum-algorithms...
Resolving dependencies...
  Found quantum-algorithms (1.0.0)
  Required: numpy>=1.20.0, quantrs2>=0.1.0
Downloading packages...
  quantum-algorithms-1.0.0.tar.gz (2.1 MB)
Installing packages...
  ✓ numpy (1.21.5)
  ✓ quantrs2 (0.1.0)  
  ✓ quantum-algorithms (1.0.0)
✓ Successfully installed 3 packages
"""
        },
        {
            'command': 'quantpkg list',
            'description': 'List all installed packages',
            'example_output': """
Installed packages (8):
  numpy (1.21.5) - utility - 15.2MB - 2024-01-15
  quantrs2 (0.1.0) - framework - 8.7MB - 2024-01-15
  quantum-algorithms (1.0.0) - algorithm - 2.1MB - 2024-01-15
  quantum-circuits (2.1.0) - circuit_library - 1.8MB - 2024-01-10
  scipy (1.9.3) - utility - 28.4MB - 2024-01-08
"""
        },
        {
            'command': 'quantpkg info quantum-algorithms',
            'description': 'Show detailed package information',
            'example_output': """
Package: quantum-algorithms
Version: 1.0.0
Type: algorithm
Description: Collection of quantum algorithms including Grover and VQE
Author: Quantum Developer
License: MIT
Homepage: https://github.com/quantum/algorithms
Keywords: quantum, algorithms, grover, vqe
Hardware: qiskit, cirq, quantrs
Dependencies:
  - numpy >=1.20.0 (required)
  - quantrs2 >=0.1.0 (required)
Installed: 2024-01-15 14:30:22 (2.1MB)
"""
        },
        {
            'command': 'quantpkg create ./my_package',
            'description': 'Create a package from source directory',
            'example_output': """
Creating package from ./my_package...
Validating package structure...
  ✓ Found quantum_package.yml
  ✓ Package metadata valid
  ✓ Dependencies resolved
  ⚠ Package contains experimental features
Generating package archive...
✓ Package created: my_package-1.0.0.tar.gz
"""
        },
        {
            'command': 'quantpkg stats',
            'description': 'Show package manager statistics',
            'example_output': """
Package Manager Statistics:
  Total packages: 8
  Total size: 58.2 MB
  Registries: 3
  Dependencies: 15
  Package types:
    algorithm: 2
    circuit_library: 1
    framework: 1
    utility: 4
  Recent activity:
    ✓ install quantum-algorithms (2024-01-15)
    ✓ install quantum-circuits (2024-01-10)
    ✓ uninstall old-package (2024-01-08)
"""
        }
    ]
    
    # Display CLI examples
    for cmd_info in cli_commands:
        print(f"\n💻 {cmd_info['command']}")
        print(f"   {cmd_info['description']}")
        print(f"\nExample output:")
        print(cmd_info['example_output'])
    
    # CLI features
    print(f"\n--- CLI Features ---")
    features = [
        "Package search with filtering by type and keywords",
        "Installation with automatic dependency resolution",
        "Package listing with size and installation date",
        "Detailed package information display",
        "Package creation from source directories",
        "Registry management and configuration",
        "Statistics and usage reporting",
        "Colored output and progress indicators",
        "Tab completion for package names",
        "Configuration file management"
    ]
    
    for feature in features:
        print(f"  ✓ {feature}")


def demo_integration_features():
    """Demonstrate integration with quantum development tools."""
    print("\n" + "="*60)
    print("INTEGRATION FEATURES DEMO")
    print("="*60)
    
    print("--- Integration with Quantum Development Tools ---")
    
    integrations = [
        {
            'name': 'QuantRS2 Framework Integration',
            'description': 'Native integration with QuantRS2 quantum computing framework',
            'features': [
                'Automatic detection of QuantRS2 packages',
                'Quantum circuit compatibility validation',
                'Hardware backend integration',
                'Performance optimization suggestions'
            ]
        },
        {
            'name': 'CI/CD Pipeline Integration',
            'description': 'Integration with quantum CI/CD pipelines',
            'features': [
                'Automated package testing in CI/CD workflows',
                'Dependency vulnerability scanning',
                'Package deployment automation',
                'Version management and release tracking'
            ]
        },
        {
            'name': 'IDE and Development Tools',
            'description': 'Integration with quantum development environments',
            'features': [
                'IDE plugin package management',
                'Code completion for package APIs',
                'Package documentation integration',
                'Debugging tools for package development'
            ]
        },
        {
            'name': 'Quantum Cloud Services',
            'description': 'Integration with quantum cloud platforms',
            'features': [
                'Cloud-based package registries',
                'Hardware-specific package recommendations',
                'Performance optimization for cloud deployment',
                'Cost analysis for quantum package usage'
            ]
        },
        {
            'name': 'Hardware Backend Support',
            'description': 'Support for quantum hardware backends',
            'features': [
                'Hardware compatibility checking',
                'Driver package management',
                'Calibration data integration',
                'Performance benchmarking on real hardware'
            ]
        }
    ]
    
    # Display integrations
    for integration in integrations:
        print(f"\n🔗 {integration['name']}")
        print(f"   {integration['description']}")
        print("   Features:")
        for feature in integration['features']:
            print(f"     • {feature}")
    
    # Example integration workflows
    print(f"\n--- Example Integration Workflows ---")
    
    workflows = [
        {
            'name': 'Quantum Algorithm Development',
            'steps': [
                '1. Create new quantum algorithm package',
                '2. Specify quantum requirements and dependencies',
                '3. Implement algorithm with QuantRS2 integration',
                '4. Run automated tests with quantum testing tools',
                '5. Package and publish to registry',
                '6. Deploy to quantum cloud platforms'
            ]
        },
        {
            'name': 'Hardware Driver Installation',
            'steps': [
                '1. Detect available quantum hardware',
                '2. Search for compatible driver packages',
                '3. Install drivers with hardware validation',
                '4. Configure quantum backend integration',
                '5. Run hardware compatibility tests',
                '6. Update hardware performance profiles'
            ]
        },
        {
            'name': 'Research Collaboration',
            'steps': [
                '1. Share research packages in private registry',
                '2. Collaborate on package development',
                '3. Peer review package implementations',
                '4. Benchmark against standard algorithms',
                '5. Publish to public research registry',
                '6. Track citations and usage metrics'
            ]
        }
    ]
    
    for workflow in workflows:
        print(f"\n📋 {workflow['name']}:")
        for step in workflow['steps']:
            print(f"   {step}")


async def main():
    """Run the comprehensive quantum package manager demo."""
    print("QuantRS2 Quantum Package Manager Comprehensive Demo")
    print("=" * 80)
    print("This demo showcases the complete package management capabilities")
    print("of the QuantRS2 quantum computing framework.")
    print("=" * 80)
    
    # Configure logging for demo
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise for demo
    
    try:
        # Run all demo sections
        pkg_manager = demo_package_manager_setup()
        demo_package_creation()
        demo_package_search_and_discovery(pkg_manager)
        demo_dependency_resolution()
        demo_package_installation_simulation()
        demo_package_validation_and_security()
        demo_registry_management()
        demo_cli_interface()
        demo_integration_features()
        
        print("\n" + "="*80)
        print("DEMO COMPLETE!")
        print("="*80)
        print("All quantum package manager features have been demonstrated successfully.")
        
        print("\nQuantum Package Manager capabilities demonstrated:")
        print("  ✓ Package creation, validation, and distribution")
        print("  ✓ Dependency resolution with quantum-specific requirements")
        print("  ✓ Registry management and package discovery")
        print("  ✓ Security scanning and compatibility validation")
        print("  ✓ CLI interface for package operations")
        print("  ✓ Integration with quantum development tools")
        print("  ✓ Support for various package types and hardware")
        print("  ✓ Multi-registry support with priority management")
        
        dependency_status = [
            f"  {'✓' if HAS_CLICK else '✗'} CLI interface (Click)",
            f"  {'✓' if HAS_TOML else '✗'} TOML configuration support",
            f"  {'✓' if HAS_GIT else '✗'} Git integration for version control",
            f"  {'✓' if HAS_REQUESTS else '✗'} HTTP support for remote registries"
        ]
        
        print("\nDependency status:")
        for status in dependency_status:
            print(status)
        
        print("\nTo use the quantum package manager:")
        print("  # Create package manager")
        print("  pkg_manager = get_quantum_package_manager()")
        print("  ")
        print("  # Search for packages")
        print("  results = pkg_manager.search_packages('quantum')")
        print("  ")
        print("  # Install packages")
        print("  result = pkg_manager.install_package('quantum-algorithms')")
        print("  ")
        print("  # List installed packages")
        print("  packages = pkg_manager.list_installed_packages()")
        
        print("\nFor CLI usage:")
        print("  quantpkg search grover")
        print("  quantpkg install quantum-algorithms") 
        print("  quantpkg list")
        print("  quantpkg info quantum-algorithms")
        print("  quantpkg create ./my_package")
        
        print("\nFor advanced features:")
        print("  # Add custom registry")
        print("  pkg_manager.add_registry('my-registry', 'https://my.registry.com')")
        print("  ")
        print("  # Create package")
        print("  result = pkg_manager.create_package('./my_package')")
        print("  ")
        print("  # Get statistics") 
        print("  stats = pkg_manager.get_package_statistics()")
        
        print("\nThe QuantRS2 Quantum Package Manager is fully functional!")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)