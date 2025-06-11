#!/usr/bin/env python3
"""
QuantRS2 Quantum Package Manager.

This module provides comprehensive package management for quantum software development:
- Package specification and metadata management with quantum-specific requirements
- Dependency resolution with version constraints and compatibility checking
- Package installation, distribution, and registry management
- Integration with quantum development tools and hardware backends
- Package validation, security scanning, and integrity verification
- CLI interface for package operations and automation
- Support for quantum algorithm libraries, circuit collections, and hardware drivers
- Integration with quantum cloud services and hardware providers
"""

import os
import json
# Optional dependencies with graceful fallbacks
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
import time
import hashlib
import shutil
import tarfile
import zipfile
import tempfile
import subprocess
import logging
import sqlite3
import threading
try:
    import semantic_version
    HAS_SEMANTIC_VERSION = True
except ImportError:
    HAS_SEMANTIC_VERSION = False
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from contextlib import contextmanager
import numpy as np

# Optional dependencies with graceful fallbacks
try:
    import toml
    HAS_TOML = True
except ImportError:
    HAS_TOML = False

try:
    import git
    HAS_GIT = True
except ImportError:
    HAS_GIT = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import click
    HAS_CLICK = True
except ImportError:
    HAS_CLICK = False

try:
    import setuptools
    HAS_SETUPTOOLS = True
except ImportError:
    HAS_SETUPTOOLS = False

try:
    import packaging.requirements
    import packaging.version
    HAS_PACKAGING = True
except ImportError:
    HAS_PACKAGING = False

# QuantRS2 integration
try:
    import quantrs2
    HAS_QUANTRS2 = True
except ImportError:
    HAS_QUANTRS2 = False


class PackageType(Enum):
    """Package type enumeration."""
    ALGORITHM = "algorithm"
    CIRCUIT_LIBRARY = "circuit_library"
    HARDWARE_DRIVER = "hardware_driver"
    OPTIMIZATION = "optimization"
    TESTING = "testing"
    VISUALIZATION = "visualization"
    SIMULATION = "simulation"
    APPLICATION = "application"
    UTILITY = "utility"
    FRAMEWORK = "framework"


class InstallationStatus(Enum):
    """Package installation status."""
    NOT_INSTALLED = "not_installed"
    INSTALLING = "installing"
    INSTALLED = "installed"
    FAILED = "failed"
    UPDATING = "updating"
    UNINSTALLING = "uninstalling"


class DependencyType(Enum):
    """Dependency type enumeration."""
    REQUIRED = "required"
    OPTIONAL = "optional"
    DEVELOPMENT = "development"
    BUILD = "build"
    QUANTUM_HARDWARE = "quantum_hardware"


class RegistryType(Enum):
    """Package registry type."""
    PUBLIC = "public"
    PRIVATE = "private"
    LOCAL = "local"
    ENTERPRISE = "enterprise"


@dataclass
class PackageRequirement:
    """Package requirement specification."""
    name: str
    version_spec: str = "*"
    type: DependencyType = DependencyType.REQUIRED
    quantum_features: List[str] = field(default_factory=list)
    hardware_requirements: Dict[str, Any] = field(default_factory=dict)
    optional: bool = False


@dataclass
class PackageMetadata:
    """Package metadata specification."""
    name: str
    version: str
    description: str = ""
    author: str = ""
    author_email: str = ""
    license: str = ""
    homepage: str = ""
    repository: str = ""
    keywords: List[str] = field(default_factory=list)
    classifiers: List[str] = field(default_factory=list)
    package_type: PackageType = PackageType.UTILITY
    quantum_requirements: Dict[str, Any] = field(default_factory=dict)
    hardware_compatibility: List[str] = field(default_factory=list)
    python_requires: str = ">=3.8"
    created_at: float = field(default_factory=time.time)


@dataclass
class PackageManifest:
    """Complete package manifest."""
    metadata: PackageMetadata
    dependencies: List[PackageRequirement] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    entry_points: Dict[str, str] = field(default_factory=dict)
    scripts: List[str] = field(default_factory=list)
    quantum_config: Dict[str, Any] = field(default_factory=dict)
    build_config: Dict[str, Any] = field(default_factory=dict)
    test_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InstalledPackage:
    """Installed package information."""
    manifest: PackageManifest
    installation_path: str
    installed_at: float
    status: InstallationStatus = InstallationStatus.INSTALLED
    checksum: str = ""
    size: int = 0


@dataclass
class RegistryConfig:
    """Package registry configuration."""
    name: str
    url: str
    type: RegistryType = RegistryType.PUBLIC
    credentials: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 0
    trusted: bool = False


class PackageValidator:
    """Package validation and security."""
    
    def __init__(self):
        """Initialize package validator."""
        self.validation_rules = []
        self.security_scanners = []
        
    def validate_package(self, package_path: str, manifest: PackageManifest) -> Dict[str, Any]:
        """Validate package contents and manifest."""
        results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'security_issues': [],
            'quantum_compatibility': True
        }
        
        try:
            # Validate manifest
            manifest_results = self._validate_manifest(manifest)
            results.update(manifest_results)
            
            # Validate package structure
            structure_results = self._validate_structure(package_path, manifest)
            results.update(structure_results)
            
            # Security scan
            security_results = self._security_scan(package_path)
            results['security_issues'].extend(security_results)
            
            # Quantum compatibility check
            quantum_results = self._check_quantum_compatibility(package_path, manifest)
            results['quantum_compatibility'] = quantum_results['compatible']
            results['warnings'].extend(quantum_results.get('warnings', []))
            
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Validation failed: {e}")
        
        return results
    
    def _validate_manifest(self, manifest: PackageManifest) -> Dict[str, Any]:
        """Validate package manifest."""
        results = {'warnings': [], 'errors': []}
        
        # Required fields
        if not manifest.metadata.name:
            results['errors'].append("Package name is required")
        
        if not manifest.metadata.version:
            results['errors'].append("Package version is required")
        
        # Version format validation
        try:
            semantic_version.Version(manifest.metadata.version)
        except ValueError:
            results['errors'].append(f"Invalid version format: {manifest.metadata.version}")
        
        # Dependency validation
        for dep in manifest.dependencies:
            if not dep.name:
                results['errors'].append("Dependency name is required")
            
            try:
                if HAS_PACKAGING:
                    packaging.requirements.Requirement(f"{dep.name}{dep.version_spec}")
            except Exception:
                results['warnings'].append(f"Invalid dependency version spec: {dep.name}{dep.version_spec}")
        
        return results
    
    def _validate_structure(self, package_path: str, manifest: PackageManifest) -> Dict[str, Any]:
        """Validate package file structure."""
        results = {'warnings': [], 'errors': []}
        
        package_dir = Path(package_path)
        
        # Check declared files exist
        for file_path in manifest.files:
            full_path = package_dir / file_path
            if not full_path.exists():
                results['errors'].append(f"Declared file not found: {file_path}")
        
        # Check for common quantum package patterns
        has_quantum_code = any(
            file_path.endswith(('.py', '.qasm', '.quil'))
            for file_path in manifest.files
        )
        
        if manifest.metadata.package_type in [PackageType.ALGORITHM, PackageType.CIRCUIT_LIBRARY] and not has_quantum_code:
            results['warnings'].append("Quantum package type declared but no quantum code files found")
        
        return results
    
    def _security_scan(self, package_path: str) -> List[str]:
        """Perform basic security scan."""
        security_issues = []
        
        try:
            package_dir = Path(package_path)
            
            # Check for suspicious files
            suspicious_patterns = [
                "*.exe", "*.dll", "*.so", "*.dylib", "*.sh", "*.bat", "*.cmd"
            ]
            
            for pattern in suspicious_patterns:
                suspicious_files = list(package_dir.rglob(pattern))
                if suspicious_files:
                    security_issues.append(f"Suspicious files found: {[str(f) for f in suspicious_files]}")
            
            # Check Python files for dangerous imports
            python_files = list(package_dir.rglob("*.py"))
            dangerous_imports = ['os.system', 'subprocess.call', 'eval', 'exec', '__import__']
            
            for py_file in python_files:
                try:
                    content = py_file.read_text(encoding='utf-8')
                    for dangerous in dangerous_imports:
                        if dangerous in content:
                            security_issues.append(f"Potentially dangerous code in {py_file}: {dangerous}")
                except Exception:
                    continue
                    
        except Exception as e:
            security_issues.append(f"Security scan failed: {e}")
        
        return security_issues
    
    def _check_quantum_compatibility(self, package_path: str, manifest: PackageManifest) -> Dict[str, Any]:
        """Check quantum framework compatibility."""
        results = {'compatible': True, 'warnings': []}
        
        try:
            # Check quantum requirements
            quantum_reqs = manifest.metadata.quantum_requirements
            
            if quantum_reqs.get('min_qubits', 0) > 100:
                results['warnings'].append("Package requires more than 100 qubits - may not work on all simulators")
            
            if quantum_reqs.get('hardware_only', False):
                results['warnings'].append("Package requires quantum hardware - simulation may not be supported")
            
            # Check for QuantRS2 compatibility
            if HAS_QUANTRS2:
                package_dir = Path(package_path)
                python_files = list(package_dir.rglob("*.py"))
                
                has_quantrs_import = False
                for py_file in python_files:
                    try:
                        content = py_file.read_text(encoding='utf-8')
                        if 'quantrs' in content.lower():
                            has_quantrs_import = True
                            break
                    except Exception:
                        continue
                
                if not has_quantrs_import and manifest.metadata.package_type in [PackageType.ALGORITHM, PackageType.CIRCUIT_LIBRARY]:
                    results['warnings'].append("Quantum package but no QuantRS integration detected")
            
        except Exception as e:
            results['warnings'].append(f"Compatibility check failed: {e}")
        
        return results


class DependencyResolver:
    """Package dependency resolution."""
    
    def __init__(self, installed_packages: Dict[str, InstalledPackage]):
        """Initialize dependency resolver."""
        self.installed_packages = installed_packages
        self.resolution_cache = {}
        
    def resolve_dependencies(self, requirements: List[PackageRequirement], 
                           available_packages: Dict[str, List[PackageManifest]]) -> Dict[str, Any]:
        """Resolve package dependencies."""
        resolution = {
            'install_order': [],
            'conflicts': [],
            'missing': [],
            'resolution_graph': {}
        }
        
        try:
            # Build dependency graph
            dep_graph = {}
            
            for req in requirements:
                self._build_dependency_graph(req, available_packages, dep_graph, set())
            
            # Topological sort for installation order
            install_order = self._topological_sort(dep_graph)
            resolution['install_order'] = install_order
            resolution['resolution_graph'] = dep_graph
            
            # Check for conflicts
            conflicts = self._check_conflicts(dep_graph, available_packages)
            resolution['conflicts'] = conflicts
            
            # Check for missing packages
            missing = self._check_missing_packages(dep_graph, available_packages)
            resolution['missing'] = missing
            
        except Exception as e:
            resolution['error'] = str(e)
            logging.error(f"Dependency resolution failed: {e}")
        
        return resolution
    
    def _build_dependency_graph(self, requirement: PackageRequirement, 
                               available_packages: Dict[str, List[PackageManifest]], 
                               graph: Dict[str, Set[str]], visited: Set[str]):
        """Build dependency graph recursively."""
        if requirement.name in visited:
            return  # Circular dependency handling
        
        visited.add(requirement.name)
        
        if requirement.name not in graph:
            graph[requirement.name] = set()
        
        # Find best matching version
        if requirement.name in available_packages:
            best_manifest = self._find_best_version(requirement, available_packages[requirement.name])
            
            if best_manifest:
                # Add dependencies
                for dep in best_manifest.dependencies:
                    if dep.type == DependencyType.REQUIRED or not dep.optional:
                        graph[requirement.name].add(dep.name)
                        self._build_dependency_graph(dep, available_packages, graph, visited.copy())
    
    def _find_best_version(self, requirement: PackageRequirement, 
                          available_versions: List[PackageManifest]) -> Optional[PackageManifest]:
        """Find the best matching version for a requirement."""
        compatible_versions = []
        
        for manifest in available_versions:
            if self._version_matches(manifest.metadata.version, requirement.version_spec):
                compatible_versions.append(manifest)
        
        if not compatible_versions:
            return None
        
        # Sort by version (newest first)
        compatible_versions.sort(
            key=lambda m: semantic_version.Version(m.metadata.version), 
            reverse=True
        )
        
        return compatible_versions[0]
    
    def _version_matches(self, version: str, spec: str) -> bool:
        """Check if version matches specification."""
        if spec == "*":
            return True
        
        try:
            if HAS_PACKAGING:
                req = packaging.requirements.Requirement(f"package{spec}")
                return packaging.version.Version(version) in req.specifier
            else:
                # Simple version matching fallback
                if spec.startswith(">="):
                    return semantic_version.Version(version) >= semantic_version.Version(spec[2:])
                elif spec.startswith("=="):
                    return version == spec[2:]
                else:
                    return True
        except Exception:
            return True
    
    def _topological_sort(self, graph: Dict[str, Set[str]]) -> List[str]:
        """Topological sort for installation order."""
        in_degree = {node: 0 for node in graph}
        
        for node in graph:
            for neighbor in graph[node]:
                if neighbor in in_degree:
                    in_degree[neighbor] += 1
        
        queue = [node for node in in_degree if in_degree[node] == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            for neighbor in graph.get(node, set()):
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
        
        return result
    
    def _check_conflicts(self, graph: Dict[str, Set[str]], 
                        available_packages: Dict[str, List[PackageManifest]]) -> List[Dict[str, Any]]:
        """Check for version conflicts."""
        conflicts = []
        
        # Check for packages with incompatible quantum requirements
        quantum_conflicts = self._check_quantum_conflicts(graph, available_packages)
        conflicts.extend(quantum_conflicts)
        
        return conflicts
    
    def _check_quantum_conflicts(self, graph: Dict[str, Set[str]], 
                                available_packages: Dict[str, List[PackageManifest]]) -> List[Dict[str, Any]]:
        """Check for quantum-specific conflicts."""
        conflicts = []
        
        # Check hardware compatibility
        hardware_requirements = {}
        
        for package_name in graph:
            if package_name in available_packages:
                manifest = available_packages[package_name][0]  # Latest version
                hw_compat = manifest.metadata.hardware_compatibility
                
                if hw_compat:
                    if package_name not in hardware_requirements:
                        hardware_requirements[package_name] = set(hw_compat)
                    else:
                        hardware_requirements[package_name] &= set(hw_compat)
        
        # Check for incompatible hardware requirements
        all_hardware = set()
        for hw_set in hardware_requirements.values():
            all_hardware.update(hw_set)
        
        for package_name, hw_reqs in hardware_requirements.items():
            if hw_reqs and not hw_reqs.intersection(all_hardware):
                conflicts.append({
                    'type': 'hardware_conflict',
                    'package': package_name,
                    'description': f"Hardware requirements {list(hw_reqs)} incompatible with other packages"
                })
        
        return conflicts
    
    def _check_missing_packages(self, graph: Dict[str, Set[str]], 
                               available_packages: Dict[str, List[PackageManifest]]) -> List[str]:
        """Check for missing packages."""
        missing = []
        
        for package_name in graph:
            if package_name not in available_packages:
                missing.append(package_name)
        
        return missing


class PackageRegistry:
    """Package registry management."""
    
    def __init__(self, config: RegistryConfig, cache_dir: str):
        """Initialize package registry."""
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_cache = {}
        self.last_update = 0
        
    def fetch_package_index(self, force_update: bool = False) -> Dict[str, List[PackageManifest]]:
        """Fetch package index from registry."""
        cache_file = self.cache_dir / f"{self.config.name}_index.json"
        cache_ttl = 3600  # 1 hour
        
        # Check cache
        if not force_update and cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < cache_ttl:
                try:
                    with open(cache_file, 'r') as f:
                        index_data = json.load(f)
                    return self._parse_index_data(index_data)
                except Exception:
                    pass
        
        # Fetch from registry
        try:
            if self.config.type == RegistryType.LOCAL:
                index = self._fetch_local_index()
            else:
                index = self._fetch_remote_index()
            
            # Cache the index
            with open(cache_file, 'w') as f:
                json.dump(self._serialize_index(index), f, indent=2)
            
            return index
            
        except Exception as e:
            logging.error(f"Failed to fetch package index from {self.config.name}: {e}")
            
            # Return cached version if available
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        index_data = json.load(f)
                    return self._parse_index_data(index_data)
                except Exception:
                    pass
            
            return {}
    
    def _fetch_local_index(self) -> Dict[str, List[PackageManifest]]:
        """Fetch index from local registry."""
        index = {}
        registry_path = Path(self.config.url)
        
        if not registry_path.exists():
            return index
        
        for package_dir in registry_path.iterdir():
            if package_dir.is_dir():
                manifest_file = package_dir / "quantum_package.yml"
                if manifest_file.exists():
                    try:
                        with open(manifest_file, 'r') as f:
                            manifest_data = yaml.safe_load(f)
                        manifest = self._parse_manifest(manifest_data)
                        
                        if manifest.metadata.name not in index:
                            index[manifest.metadata.name] = []
                        index[manifest.metadata.name].append(manifest)
                        
                    except Exception as e:
                        logging.warning(f"Failed to parse manifest in {package_dir}: {e}")
        
        return index
    
    def _fetch_remote_index(self) -> Dict[str, List[PackageManifest]]:
        """Fetch index from remote registry."""
        index = {}
        
        if not HAS_REQUESTS:
            logging.warning("Requests library not available for remote registry access")
            return index
        
        try:
            # Fetch index endpoint
            index_url = f"{self.config.url.rstrip('/')}/index.json"
            
            headers = {}
            if self.config.credentials.get('token'):
                headers['Authorization'] = f"Bearer {self.config.credentials['token']}"
            
            response = requests.get(index_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            index_data = response.json()
            return self._parse_index_data(index_data)
            
        except Exception as e:
            logging.error(f"Failed to fetch remote index: {e}")
            return index
    
    def _parse_index_data(self, index_data: Dict[str, Any]) -> Dict[str, List[PackageManifest]]:
        """Parse index data into manifests."""
        index = {}
        
        for package_name, versions_data in index_data.items():
            manifests = []
            for version_data in versions_data:
                try:
                    manifest = self._parse_manifest(version_data)
                    manifests.append(manifest)
                except Exception as e:
                    logging.warning(f"Failed to parse manifest for {package_name}: {e}")
            
            if manifests:
                index[package_name] = manifests
        
        return index
    
    def _parse_manifest(self, manifest_data: Dict[str, Any]) -> PackageManifest:
        """Parse manifest data."""
        metadata = PackageMetadata(**manifest_data['metadata'])
        
        dependencies = []
        for dep_data in manifest_data.get('dependencies', []):
            dependencies.append(PackageRequirement(**dep_data))
        
        return PackageManifest(
            metadata=metadata,
            dependencies=dependencies,
            files=manifest_data.get('files', []),
            entry_points=manifest_data.get('entry_points', {}),
            scripts=manifest_data.get('scripts', []),
            quantum_config=manifest_data.get('quantum_config', {}),
            build_config=manifest_data.get('build_config', {}),
            test_config=manifest_data.get('test_config', {})
        )
    
    def _serialize_index(self, index: Dict[str, List[PackageManifest]]) -> Dict[str, Any]:
        """Serialize index for caching."""
        serialized = {}
        
        for package_name, manifests in index.items():
            serialized[package_name] = [asdict(manifest) for manifest in manifests]
        
        return serialized
    
    def download_package(self, package_name: str, version: str, download_dir: str) -> str:
        """Download package from registry."""
        if self.config.type == RegistryType.LOCAL:
            return self._download_local_package(package_name, version, download_dir)
        else:
            return self._download_remote_package(package_name, version, download_dir)
    
    def _download_local_package(self, package_name: str, version: str, download_dir: str) -> str:
        """Download package from local registry."""
        registry_path = Path(self.config.url)
        package_path = registry_path / package_name / version
        
        if not package_path.exists():
            raise FileNotFoundError(f"Package {package_name}=={version} not found in local registry")
        
        # Copy package to download directory
        download_path = Path(download_dir) / f"{package_name}-{version}"
        shutil.copytree(package_path, download_path, dirs_exist_ok=True)
        
        return str(download_path)
    
    def _download_remote_package(self, package_name: str, version: str, download_dir: str) -> str:
        """Download package from remote registry."""
        if not HAS_REQUESTS:
            raise RuntimeError("Requests library not available for remote package download")
        
        # Download package archive
        download_url = f"{self.config.url.rstrip('/')}/packages/{package_name}/{version}/download"
        
        headers = {}
        if self.config.credentials.get('token'):
            headers['Authorization'] = f"Bearer {self.config.credentials['token']}"
        
        response = requests.get(download_url, headers=headers, timeout=300, stream=True)
        response.raise_for_status()
        
        # Save package archive
        download_path = Path(download_dir) / f"{package_name}-{version}.tar.gz"
        with open(download_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract package
        extract_path = Path(download_dir) / f"{package_name}-{version}"
        extract_path.mkdir(exist_ok=True)
        
        with tarfile.open(download_path, 'r:gz') as tar:
            tar.extractall(extract_path)
        
        # Remove archive
        download_path.unlink()
        
        return str(extract_path)


class QuantumPackageManager:
    """Main quantum package manager."""
    
    def __init__(self, workspace_dir: str = "./quantum_packages"):
        """Initialize quantum package manager."""
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Directories
        self.packages_dir = self.workspace_dir / "packages"
        self.cache_dir = self.workspace_dir / "cache"
        self.config_dir = self.workspace_dir / "config"
        
        for directory in [self.packages_dir, self.cache_dir, self.config_dir]:
            directory.mkdir(exist_ok=True)
        
        # Database for installed packages
        self.db_path = self.workspace_dir / "packages.db"
        self._init_database()
        
        # Components
        self.validator = PackageValidator()
        self.registries = {}
        self.installed_packages = {}
        
        # Load configuration
        self._load_configuration()
        self._load_installed_packages()
        
        # Default registries
        self._setup_default_registries()
    
    def _init_database(self):
        """Initialize package database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS installed_packages (
                    name TEXT PRIMARY KEY,
                    version TEXT NOT NULL,
                    installation_path TEXT NOT NULL,
                    installed_at REAL NOT NULL,
                    status TEXT NOT NULL,
                    checksum TEXT,
                    size INTEGER,
                    manifest_data TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS package_dependencies (
                    package_name TEXT,
                    dependency_name TEXT,
                    dependency_version TEXT,
                    dependency_type TEXT,
                    FOREIGN KEY (package_name) REFERENCES installed_packages (name)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS package_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    package_name TEXT,
                    action TEXT,
                    version TEXT,
                    timestamp REAL,
                    success INTEGER
                )
            """)
    
    def _load_configuration(self):
        """Load package manager configuration."""
        config_file = self.config_dir / "config.yml"
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Load registry configurations
                for registry_data in config_data.get('registries', []):
                    registry_config = RegistryConfig(**registry_data)
                    self.registries[registry_config.name] = PackageRegistry(
                        registry_config, str(self.cache_dir)
                    )
                    
            except Exception as e:
                logging.error(f"Failed to load configuration: {e}")
    
    def _load_installed_packages(self):
        """Load installed packages from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT name, version, installation_path, installed_at, status, 
                       checksum, size, manifest_data
                FROM installed_packages
            """)
            
            for row in cursor.fetchall():
                try:
                    manifest_data = json.loads(row[7])
                    manifest = self._parse_manifest_from_dict(manifest_data)
                    
                    installed_pkg = InstalledPackage(
                        manifest=manifest,
                        installation_path=row[2],
                        installed_at=row[3],
                        status=InstallationStatus(row[4]),
                        checksum=row[5] or "",
                        size=row[6] or 0
                    )
                    
                    self.installed_packages[row[0]] = installed_pkg
                    
                except Exception as e:
                    logging.warning(f"Failed to load installed package {row[0]}: {e}")
    
    def _setup_default_registries(self):
        """Setup default package registries."""
        if "quantrs-public" not in self.registries:
            public_registry = RegistryConfig(
                name="quantrs-public",
                url="https://registry.quantrs.org",
                type=RegistryType.PUBLIC,
                trusted=True,
                priority=10
            )
            self.registries["quantrs-public"] = PackageRegistry(
                public_registry, str(self.cache_dir)
            )
        
        if "local" not in self.registries:
            local_registry = RegistryConfig(
                name="local",
                url=str(self.workspace_dir / "local_registry"),
                type=RegistryType.LOCAL,
                trusted=True,
                priority=0
            )
            self.registries["local"] = PackageRegistry(
                local_registry, str(self.cache_dir)
            )
    
    def _parse_manifest_from_dict(self, manifest_data: Dict[str, Any]) -> PackageManifest:
        """Parse manifest from dictionary."""
        metadata = PackageMetadata(**manifest_data['metadata'])
        
        dependencies = []
        for dep_data in manifest_data.get('dependencies', []):
            dependencies.append(PackageRequirement(**dep_data))
        
        return PackageManifest(
            metadata=metadata,
            dependencies=dependencies,
            files=manifest_data.get('files', []),
            entry_points=manifest_data.get('entry_points', {}),
            scripts=manifest_data.get('scripts', []),
            quantum_config=manifest_data.get('quantum_config', {}),
            build_config=manifest_data.get('build_config', {}),
            test_config=manifest_data.get('test_config', {})
        )
    
    def search_packages(self, query: str, package_type: Optional[PackageType] = None,
                       update_index: bool = False) -> List[Dict[str, Any]]:
        """Search for packages in registries."""
        results = []
        
        # Fetch indices from all registries
        all_packages = {}
        for registry_name, registry in self.registries.items():
            if not registry.config.enabled:
                continue
                
            try:
                index = registry.fetch_package_index(force_update=update_index)
                for package_name, manifests in index.items():
                    if package_name not in all_packages:
                        all_packages[package_name] = []
                    all_packages[package_name].extend(manifests)
            except Exception as e:
                logging.error(f"Failed to fetch index from {registry_name}: {e}")
        
        # Search packages
        query_lower = query.lower()
        
        for package_name, manifests in all_packages.items():
            for manifest in manifests:
                # Check if package matches search criteria
                matches = False
                
                if query_lower in package_name.lower():
                    matches = True
                elif query_lower in manifest.metadata.description.lower():
                    matches = True
                elif any(query_lower in keyword.lower() for keyword in manifest.metadata.keywords):
                    matches = True
                
                if matches and (package_type is None or manifest.metadata.package_type == package_type):
                    results.append({
                        'name': manifest.metadata.name,
                        'version': manifest.metadata.version,
                        'description': manifest.metadata.description,
                        'type': manifest.metadata.package_type.value,
                        'author': manifest.metadata.author,
                        'keywords': manifest.metadata.keywords,
                        'hardware_compatibility': manifest.metadata.hardware_compatibility,
                        'quantum_requirements': manifest.metadata.quantum_requirements
                    })
        
        # Remove duplicates and sort by relevance
        unique_results = {}
        for result in results:
            key = (result['name'], result['version'])
            if key not in unique_results:
                unique_results[key] = result
        
        return sorted(unique_results.values(), key=lambda x: x['name'])
    
    def install_package(self, package_spec: str, upgrade: bool = False,
                       no_deps: bool = False, force: bool = False) -> Dict[str, Any]:
        """Install a quantum package."""
        install_result = {
            'success': False,
            'package': package_spec,
            'installed_packages': [],
            'errors': [],
            'warnings': []
        }
        
        try:
            # Parse package specification
            if "==" in package_spec:
                package_name, version = package_spec.split("==", 1)
            else:
                package_name = package_spec
                version = None
            
            # Check if already installed
            if package_name in self.installed_packages and not upgrade and not force:
                install_result['warnings'].append(f"Package {package_name} already installed")
                install_result['success'] = True
                return install_result
            
            # Find package in registries
            package_found = False
            for registry_name, registry in self.registries.items():
                if not registry.config.enabled:
                    continue
                
                try:
                    index = registry.fetch_package_index()
                    if package_name in index:
                        manifests = index[package_name]
                        
                        # Find best version
                        target_manifest = None
                        if version:
                            target_manifest = next(
                                (m for m in manifests if m.metadata.version == version), None
                            )
                        else:
                            # Get latest version
                            sorted_manifests = sorted(
                                manifests, 
                                key=lambda m: semantic_version.Version(m.metadata.version),
                                reverse=True
                            )
                            target_manifest = sorted_manifests[0] if sorted_manifests else None
                        
                        if target_manifest:
                            package_found = True
                            
                            # Resolve dependencies
                            install_list = [package_name]
                            if not no_deps:
                                dep_resolution = self._resolve_dependencies_for_install(
                                    target_manifest, index
                                )
                                if dep_resolution['conflicts']:
                                    install_result['errors'].extend([
                                        f"Dependency conflict: {conflict}"
                                        for conflict in dep_resolution['conflicts']
                                    ])
                                    return install_result
                                
                                install_list = dep_resolution['install_order']
                            
                            # Install packages in order
                            for pkg_name in install_list:
                                pkg_result = self._install_single_package(
                                    pkg_name, registry, index.get(pkg_name, [])
                                )
                                
                                if pkg_result['success']:
                                    install_result['installed_packages'].append(pkg_name)
                                else:
                                    install_result['errors'].extend(pkg_result['errors'])
                                    return install_result
                            
                            install_result['success'] = True
                            break
                            
                except Exception as e:
                    logging.error(f"Error searching in registry {registry_name}: {e}")
            
            if not package_found:
                install_result['errors'].append(f"Package {package_spec} not found in any registry")
            
            # Record installation history
            self._record_package_action(package_name, "install", 
                                      version or "latest", install_result['success'])
            
        except Exception as e:
            install_result['errors'].append(f"Installation failed: {e}")
            logging.error(f"Package installation failed: {e}")
        
        return install_result
    
    def _resolve_dependencies_for_install(self, manifest: PackageManifest, 
                                        available_packages: Dict[str, List[PackageManifest]]) -> Dict[str, Any]:
        """Resolve dependencies for installation."""
        resolver = DependencyResolver(self.installed_packages)
        return resolver.resolve_dependencies(manifest.dependencies, available_packages)
    
    def _install_single_package(self, package_name: str, registry: PackageRegistry, 
                               manifests: List[PackageManifest]) -> Dict[str, Any]:
        """Install a single package."""
        result = {'success': False, 'errors': []}
        
        try:
            if not manifests:
                result['errors'].append(f"No versions available for {package_name}")
                return result
            
            # Get latest version
            latest_manifest = sorted(
                manifests, 
                key=lambda m: semantic_version.Version(m.metadata.version),
                reverse=True
            )[0]
            
            # Skip if already installed at correct version
            if package_name in self.installed_packages:
                installed_version = self.installed_packages[package_name].manifest.metadata.version
                if installed_version == latest_manifest.metadata.version:
                    result['success'] = True
                    return result
            
            # Download package
            with tempfile.TemporaryDirectory() as temp_dir:
                download_path = registry.download_package(
                    package_name, latest_manifest.metadata.version, temp_dir
                )
                
                # Validate package
                validation_result = self.validator.validate_package(download_path, latest_manifest)
                if not validation_result['valid']:
                    result['errors'].extend(validation_result['errors'])
                    return result
                
                # Install package
                install_path = self.packages_dir / package_name / latest_manifest.metadata.version
                install_path.parent.mkdir(parents=True, exist_ok=True)
                
                if install_path.exists():
                    shutil.rmtree(install_path)
                
                shutil.copytree(download_path, install_path)
                
                # Calculate checksum and size
                checksum = self._calculate_directory_checksum(install_path)
                size = sum(f.stat().st_size for f in install_path.rglob('*') if f.is_file())
                
                # Create installed package record
                installed_package = InstalledPackage(
                    manifest=latest_manifest,
                    installation_path=str(install_path),
                    installed_at=time.time(),
                    status=InstallationStatus.INSTALLED,
                    checksum=checksum,
                    size=size
                )
                
                # Save to database
                self._save_installed_package(installed_package)
                
                # Update in-memory cache
                self.installed_packages[package_name] = installed_package
                
                result['success'] = True
                
        except Exception as e:
            result['errors'].append(str(e))
        
        return result
    
    def _calculate_directory_checksum(self, directory: Path) -> str:
        """Calculate checksum for directory contents."""
        sha256_hash = hashlib.sha256()
        
        for file_path in sorted(directory.rglob('*')):
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def _save_installed_package(self, installed_package: InstalledPackage):
        """Save installed package to database."""
        with sqlite3.connect(self.db_path) as conn:
            # Insert or replace package record
            conn.execute("""
                INSERT OR REPLACE INTO installed_packages 
                (name, version, installation_path, installed_at, status, checksum, size, manifest_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                installed_package.manifest.metadata.name,
                installed_package.manifest.metadata.version,
                installed_package.installation_path,
                installed_package.installed_at,
                installed_package.status.value,
                installed_package.checksum,
                installed_package.size,
                json.dumps(asdict(installed_package.manifest))
            ))
            
            # Clear old dependencies
            conn.execute("""
                DELETE FROM package_dependencies WHERE package_name = ?
            """, (installed_package.manifest.metadata.name,))
            
            # Insert dependencies
            for dep in installed_package.manifest.dependencies:
                conn.execute("""
                    INSERT INTO package_dependencies 
                    (package_name, dependency_name, dependency_version, dependency_type)
                    VALUES (?, ?, ?, ?)
                """, (
                    installed_package.manifest.metadata.name,
                    dep.name,
                    dep.version_spec,
                    dep.type.value
                ))
    
    def _record_package_action(self, package_name: str, action: str, 
                             version: str, success: bool):
        """Record package action in history."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO package_history (package_name, action, version, timestamp, success)
                VALUES (?, ?, ?, ?, ?)
            """, (package_name, action, version, time.time(), 1 if success else 0))
    
    def uninstall_package(self, package_name: str, force: bool = False) -> Dict[str, Any]:
        """Uninstall a quantum package."""
        result = {
            'success': False,
            'package': package_name,
            'removed_packages': [],
            'errors': [],
            'warnings': []
        }
        
        try:
            if package_name not in self.installed_packages:
                result['errors'].append(f"Package {package_name} is not installed")
                return result
            
            installed_pkg = self.installed_packages[package_name]
            
            # Check for packages that depend on this one
            dependents = self._find_dependents(package_name)
            if dependents and not force:
                result['errors'].append(
                    f"Cannot uninstall {package_name}: required by {', '.join(dependents)}"
                )
                return result
            
            # Remove package files
            install_path = Path(installed_pkg.installation_path)
            if install_path.exists():
                shutil.rmtree(install_path)
            
            # Remove from database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM installed_packages WHERE name = ?", (package_name,))
                conn.execute("DELETE FROM package_dependencies WHERE package_name = ?", (package_name,))
            
            # Remove from memory
            del self.installed_packages[package_name]
            
            result['removed_packages'].append(package_name)
            result['success'] = True
            
            # Record action
            self._record_package_action(package_name, "uninstall", 
                                      installed_pkg.manifest.metadata.version, True)
            
        except Exception as e:
            result['errors'].append(f"Uninstallation failed: {e}")
            self._record_package_action(package_name, "uninstall", "unknown", False)
        
        return result
    
    def _find_dependents(self, package_name: str) -> List[str]:
        """Find packages that depend on the given package."""
        dependents = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT DISTINCT package_name FROM package_dependencies 
                WHERE dependency_name = ?
            """, (package_name,))
            
            for row in cursor.fetchall():
                dependents.append(row[0])
        
        return dependents
    
    def list_installed_packages(self) -> List[Dict[str, Any]]:
        """List all installed packages."""
        packages = []
        
        for package_name, installed_pkg in self.installed_packages.items():
            packages.append({
                'name': package_name,
                'version': installed_pkg.manifest.metadata.version,
                'type': installed_pkg.manifest.metadata.package_type.value,
                'description': installed_pkg.manifest.metadata.description,
                'installed_at': installed_pkg.installed_at,
                'size': installed_pkg.size,
                'status': installed_pkg.status.value
            })
        
        return sorted(packages, key=lambda x: x['name'])
    
    def get_package_info(self, package_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a package."""
        if package_name not in self.installed_packages:
            return None
        
        installed_pkg = self.installed_packages[package_name]
        manifest = installed_pkg.manifest
        
        return {
            'name': manifest.metadata.name,
            'version': manifest.metadata.version,
            'description': manifest.metadata.description,
            'author': manifest.metadata.author,
            'license': manifest.metadata.license,
            'homepage': manifest.metadata.homepage,
            'repository': manifest.metadata.repository,
            'type': manifest.metadata.package_type.value,
            'keywords': manifest.metadata.keywords,
            'quantum_requirements': manifest.metadata.quantum_requirements,
            'hardware_compatibility': manifest.metadata.hardware_compatibility,
            'dependencies': [
                {
                    'name': dep.name,
                    'version_spec': dep.version_spec,
                    'type': dep.type.value,
                    'optional': dep.optional
                }
                for dep in manifest.dependencies
            ],
            'installed_at': installed_pkg.installed_at,
            'installation_path': installed_pkg.installation_path,
            'size': installed_pkg.size,
            'checksum': installed_pkg.checksum,
            'files': manifest.files[:20],  # Limit for display
            'entry_points': manifest.entry_points,
            'quantum_config': manifest.quantum_config
        }
    
    def create_package(self, package_dir: str, output_dir: str = None) -> Dict[str, Any]:
        """Create a package from source directory."""
        result = {
            'success': False,
            'package_file': '',
            'errors': [],
            'warnings': []
        }
        
        try:
            package_path = Path(package_dir)
            if not package_path.exists():
                result['errors'].append(f"Package directory not found: {package_dir}")
                return result
            
            # Look for package manifest
            manifest_file = package_path / "quantum_package.yml"
            if not manifest_file.exists():
                # Try to create from setup.py or pyproject.toml
                manifest = self._generate_manifest_from_setup(package_path)
                if not manifest:
                    result['errors'].append("No package manifest found and could not generate from setup files")
                    return result
            else:
                with open(manifest_file, 'r') as f:
                    manifest_data = yaml.safe_load(f)
                manifest = self._parse_manifest_from_dict(manifest_data)
            
            # Validate package
            validation_result = self.validator.validate_package(str(package_path), manifest)
            if not validation_result['valid']:
                result['errors'].extend(validation_result['errors'])
                if not any("warning" in error.lower() for error in validation_result['errors']):
                    return result
            
            result['warnings'].extend(validation_result['warnings'])
            
            # Create package archive
            output_path = Path(output_dir) if output_dir else package_path.parent
            package_filename = f"{manifest.metadata.name}-{manifest.metadata.version}.tar.gz"
            package_file = output_path / package_filename
            
            with tarfile.open(package_file, 'w:gz') as tar:
                for file_path in manifest.files:
                    full_path = package_path / file_path
                    if full_path.exists():
                        tar.add(full_path, arcname=file_path)
                
                # Add manifest
                manifest_info = tarfile.TarInfo("quantum_package.yml")
                manifest_content = yaml.dump(asdict(manifest), default_flow_style=False).encode()
                manifest_info.size = len(manifest_content)
                manifest_info.mtime = time.time()
                tar.addfile(manifest_info, io.BytesIO(manifest_content))
            
            result['success'] = True
            result['package_file'] = str(package_file)
            
        except Exception as e:
            result['errors'].append(f"Package creation failed: {e}")
        
        return result
    
    def _generate_manifest_from_setup(self, package_path: Path) -> Optional[PackageManifest]:
        """Generate manifest from setup.py or pyproject.toml."""
        try:
            # Try pyproject.toml first
            pyproject_file = package_path / "pyproject.toml"
            if pyproject_file.exists() and HAS_TOML:
                with open(pyproject_file, 'r') as f:
                    pyproject_data = toml.load(f)
                
                project_data = pyproject_data.get('project', {})
                if project_data:
                    metadata = PackageMetadata(
                        name=project_data.get('name', ''),
                        version=project_data.get('version', '0.1.0'),
                        description=project_data.get('description', ''),
                        author=project_data.get('authors', [{}])[0].get('name', ''),
                        license=project_data.get('license', {}).get('text', ''),
                        keywords=project_data.get('keywords', [])
                    )
                    
                    # Auto-detect files
                    files = []
                    for py_file in package_path.rglob("*.py"):
                        files.append(str(py_file.relative_to(package_path)))
                    
                    return PackageManifest(metadata=metadata, files=files)
            
            # Try setup.py
            setup_file = package_path / "setup.py"
            if setup_file.exists():
                # Basic parsing of setup.py (simplified)
                with open(setup_file, 'r') as f:
                    setup_content = f.read()
                
                # Extract basic info using simple regex
                import re
                
                name_match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', setup_content)
                version_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', setup_content)
                description_match = re.search(r'description\s*=\s*["\']([^"\']+)["\']', setup_content)
                
                if name_match:
                    metadata = PackageMetadata(
                        name=name_match.group(1),
                        version=version_match.group(1) if version_match else '0.1.0',
                        description=description_match.group(1) if description_match else ''
                    )
                    
                    # Auto-detect files
                    files = []
                    for py_file in package_path.rglob("*.py"):
                        files.append(str(py_file.relative_to(package_path)))
                    
                    return PackageManifest(metadata=metadata, files=files)
        
        except Exception as e:
            logging.error(f"Failed to generate manifest from setup files: {e}")
        
        return None
    
    def add_registry(self, name: str, url: str, registry_type: RegistryType = RegistryType.PUBLIC,
                    credentials: Dict[str, str] = None) -> bool:
        """Add a new package registry."""
        try:
            config = RegistryConfig(
                name=name,
                url=url,
                type=registry_type,
                credentials=credentials or {},
                enabled=True,
                priority=len(self.registries)
            )
            
            registry = PackageRegistry(config, str(self.cache_dir))
            
            # Test registry connectivity
            try:
                registry.fetch_package_index()
            except Exception as e:
                logging.warning(f"Could not connect to registry {name}: {e}")
            
            self.registries[name] = registry
            self._save_configuration()
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to add registry {name}: {e}")
            return False
    
    def _save_configuration(self):
        """Save configuration to file."""
        config_file = self.config_dir / "config.yml"
        
        config_data = {
            'registries': [
                {
                    'name': registry.config.name,
                    'url': registry.config.url,
                    'type': registry.config.type.value,
                    'credentials': registry.config.credentials,
                    'enabled': registry.config.enabled,
                    'priority': registry.config.priority,
                    'trusted': registry.config.trusted
                }
                for registry in self.registries.values()
            ]
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
    
    def get_package_statistics(self) -> Dict[str, Any]:
        """Get package manager statistics."""
        stats = {
            'total_packages': len(self.installed_packages),
            'package_types': {},
            'total_size': 0,
            'registries': len(self.registries),
            'recent_installations': [],
            'dependency_graph_size': 0
        }
        
        # Calculate package type distribution
        for installed_pkg in self.installed_packages.values():
            pkg_type = installed_pkg.manifest.metadata.package_type.value
            stats['package_types'][pkg_type] = stats['package_types'].get(pkg_type, 0) + 1
            stats['total_size'] += installed_pkg.size
        
        # Get recent installations
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT package_name, action, version, timestamp, success
                FROM package_history
                WHERE action IN ('install', 'uninstall')
                ORDER BY timestamp DESC
                LIMIT 10
            """)
            
            for row in cursor.fetchall():
                stats['recent_installations'].append({
                    'package': row[0],
                    'action': row[1],
                    'version': row[2],
                    'timestamp': row[3],
                    'success': bool(row[4])
                })
        
        # Calculate dependency graph size
        total_deps = sum(
            len(pkg.manifest.dependencies) 
            for pkg in self.installed_packages.values()
        )
        stats['dependency_graph_size'] = total_deps
        
        return stats


# CLI Interface
if HAS_CLICK:
    @click.group()
    @click.option('--workspace', default='./quantum_packages', help='Package manager workspace directory')
    @click.pass_context
    def cli(ctx, workspace):
        """QuantRS2 Quantum Package Manager CLI."""
        ctx.ensure_object(dict)
        ctx.obj['manager'] = QuantumPackageManager(workspace)
    
    @cli.command()
    @click.argument('query')
    @click.option('--type', 'package_type', type=click.Choice([t.value for t in PackageType]), 
                  help='Filter by package type')
    @click.option('--update', is_flag=True, help='Update package indices before searching')
    @click.pass_context
    def search(ctx, query, package_type, update):
        """Search for quantum packages."""
        manager = ctx.obj['manager']
        
        pkg_type = PackageType(package_type) if package_type else None
        results = manager.search_packages(query, pkg_type, update)
        
        if not results:
            click.echo(f"No packages found for '{query}'")
            return
        
        click.echo(f"Found {len(results)} packages:")
        for result in results:
            click.echo(f"  {result['name']} ({result['version']}) - {result['type']}")
            click.echo(f"    {result['description']}")
            if result['keywords']:
                click.echo(f"    Keywords: {', '.join(result['keywords'])}")
            click.echo()
    
    @cli.command()
    @click.argument('package_spec')
    @click.option('--upgrade', is_flag=True, help='Upgrade if already installed')
    @click.option('--no-deps', is_flag=True, help='Do not install dependencies')
    @click.option('--force', is_flag=True, help='Force installation')
    @click.pass_context
    def install(ctx, package_spec, upgrade, no_deps, force):
        """Install a quantum package."""
        manager = ctx.obj['manager']
        
        click.echo(f"Installing {package_spec}...")
        result = manager.install_package(package_spec, upgrade, no_deps, force)
        
        if result['success']:
            click.echo(f"✓ Successfully installed {len(result['installed_packages'])} packages:")
            for pkg in result['installed_packages']:
                click.echo(f"  - {pkg}")
        else:
            click.echo("✗ Installation failed:")
            for error in result['errors']:
                click.echo(f"  {error}")
        
        for warning in result['warnings']:
            click.echo(f"⚠ {warning}")
    
    @cli.command()
    @click.argument('package_name')
    @click.option('--force', is_flag=True, help='Force uninstall even if other packages depend on it')
    @click.pass_context
    def uninstall(ctx, package_name, force):
        """Uninstall a quantum package."""
        manager = ctx.obj['manager']
        
        click.echo(f"Uninstalling {package_name}...")
        result = manager.uninstall_package(package_name, force)
        
        if result['success']:
            click.echo(f"✓ Successfully uninstalled:")
            for pkg in result['removed_packages']:
                click.echo(f"  - {pkg}")
        else:
            click.echo("✗ Uninstallation failed:")
            for error in result['errors']:
                click.echo(f"  {error}")
    
    @cli.command()
    @click.pass_context
    def list(ctx):
        """List installed quantum packages."""
        manager = ctx.obj['manager']
        
        packages = manager.list_installed_packages()
        
        if not packages:
            click.echo("No packages installed")
            return
        
        click.echo(f"Installed packages ({len(packages)}):")
        for pkg in packages:
            size_mb = pkg['size'] / (1024 * 1024)
            installed_date = time.strftime('%Y-%m-%d', time.localtime(pkg['installed_at']))
            click.echo(f"  {pkg['name']} ({pkg['version']}) - {pkg['type']} - {size_mb:.1f}MB - {installed_date}")
    
    @cli.command()
    @click.argument('package_name')
    @click.pass_context
    def info(ctx, package_name):
        """Show detailed package information."""
        manager = ctx.obj['manager']
        
        info = manager.get_package_info(package_name)
        
        if not info:
            click.echo(f"Package '{package_name}' not found")
            return
        
        click.echo(f"Package: {info['name']}")
        click.echo(f"Version: {info['version']}")
        click.echo(f"Type: {info['type']}")
        click.echo(f"Description: {info['description']}")
        click.echo(f"Author: {info['author']}")
        click.echo(f"License: {info['license']}")
        
        if info['homepage']:
            click.echo(f"Homepage: {info['homepage']}")
        if info['repository']:
            click.echo(f"Repository: {info['repository']}")
        
        if info['keywords']:
            click.echo(f"Keywords: {', '.join(info['keywords'])}")
        
        if info['hardware_compatibility']:
            click.echo(f"Hardware: {', '.join(info['hardware_compatibility'])}")
        
        if info['dependencies']:
            click.echo("Dependencies:")
            for dep in info['dependencies']:
                click.echo(f"  - {dep['name']} {dep['version_spec']} ({dep['type']})")
        
        size_mb = info['size'] / (1024 * 1024)
        installed_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(info['installed_at']))
        click.echo(f"Installed: {installed_date} ({size_mb:.1f}MB)")
    
    @cli.command()
    @click.argument('package_dir')
    @click.option('--output', help='Output directory for package file')
    @click.pass_context
    def create(ctx, package_dir, output):
        """Create a package from source directory."""
        manager = ctx.obj['manager']
        
        click.echo(f"Creating package from {package_dir}...")
        result = manager.create_package(package_dir, output)
        
        if result['success']:
            click.echo(f"✓ Package created: {result['package_file']}")
        else:
            click.echo("✗ Package creation failed:")
            for error in result['errors']:
                click.echo(f"  {error}")
        
        for warning in result['warnings']:
            click.echo(f"⚠ {warning}")
    
    @cli.command()
    @click.pass_context
    def stats(ctx):
        """Show package manager statistics."""
        manager = ctx.obj['manager']
        
        stats = manager.get_package_statistics()
        
        click.echo("Package Manager Statistics:")
        click.echo(f"  Total packages: {stats['total_packages']}")
        click.echo(f"  Total size: {stats['total_size'] / (1024*1024):.1f} MB")
        click.echo(f"  Registries: {stats['registries']}")
        click.echo(f"  Dependencies: {stats['dependency_graph_size']}")
        
        if stats['package_types']:
            click.echo("  Package types:")
            for pkg_type, count in stats['package_types'].items():
                click.echo(f"    {pkg_type}: {count}")
        
        if stats['recent_installations']:
            click.echo("  Recent activity:")
            for activity in stats['recent_installations'][:5]:
                action = activity['action']
                status = "✓" if activity['success'] else "✗"
                date = time.strftime('%Y-%m-%d', time.localtime(activity['timestamp']))
                click.echo(f"    {status} {action} {activity['package']} ({date})")


# Convenience functions
def get_quantum_package_manager(workspace_dir: str = "./quantum_packages") -> QuantumPackageManager:
    """Get a quantum package manager instance."""
    return QuantumPackageManager(workspace_dir)


def create_package_manifest(name: str, version: str, **kwargs) -> PackageManifest:
    """Create a basic package manifest."""
    metadata = PackageMetadata(
        name=name,
        version=version,
        description=kwargs.get('description', ''),
        author=kwargs.get('author', ''),
        package_type=kwargs.get('package_type', PackageType.UTILITY),
        quantum_requirements=kwargs.get('quantum_requirements', {}),
        hardware_compatibility=kwargs.get('hardware_compatibility', [])
    )
    
    return PackageManifest(
        metadata=metadata,
        dependencies=kwargs.get('dependencies', []),
        files=kwargs.get('files', []),
        entry_points=kwargs.get('entry_points', {}),
        quantum_config=kwargs.get('quantum_config', {})
    )


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    print("QuantRS2 Quantum Package Manager")
    print("=" * 60)
    
    # Initialize package manager
    pkg_manager = get_quantum_package_manager()
    
    print("✅ Quantum Package Manager initialized successfully!")
    print(f"📦 Workspace: {pkg_manager.workspace_dir}")
    print(f"🔗 Registries: {list(pkg_manager.registries.keys())}")
    
    # Show statistics
    stats = pkg_manager.get_package_statistics()
    print(f"📊 Installed packages: {stats['total_packages']}")
    print(f"💾 Total size: {stats['total_size'] / (1024*1024):.1f} MB")
    
    print("\n🚀 Quantum Package Manager is ready!")
    print("   Use the CLI interface or Python API for package operations")
    
    if HAS_CLICK:
        print("   CLI available: python -m quantrs2.quantum_package_manager --help")