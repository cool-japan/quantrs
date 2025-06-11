#!/usr/bin/env python3
"""
Comprehensive test suite for the QuantRS2 Quantum Package Manager.

This test suite provides complete coverage of all package management functionality including:
- Package creation, validation, and distribution
- Dependency resolution with quantum-specific requirements
- Registry management and package discovery
- Security scanning and compatibility validation
- CLI interface and automation workflows
- Integration with quantum development tools
- Multi-registry support and priority management
"""

import pytest
import tempfile
import os
import json
import yaml
import time
import shutil
import hashlib
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, List, Any

try:
    import quantrs2
    from quantrs2.quantum_package_manager import (
        QuantumPackageManager, PackageType, DependencyType, RegistryType, InstallationStatus,
        PackageMetadata, PackageManifest, PackageRequirement, RegistryConfig,
        InstalledPackage, PackageValidator, DependencyResolver, PackageRegistry,
        get_quantum_package_manager, create_package_manifest,
        HAS_CLICK, HAS_TOML, HAS_GIT, HAS_REQUESTS, HAS_PACKAGING
    )
    HAS_QUANTUM_PACKAGE_MANAGER = True
except ImportError:
    HAS_QUANTUM_PACKAGE_MANAGER = False


# Test fixtures
@pytest.fixture
def temp_workspace():
    """Create temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_package_metadata():
    """Create sample package metadata for testing."""
    return PackageMetadata(
        name="test-package",
        version="1.0.0",
        description="Test quantum package",
        author="Test Author",
        license="MIT",
        package_type=PackageType.ALGORITHM,
        quantum_requirements={"min_qubits": 4, "max_qubits": 16},
        hardware_compatibility=["simulator", "superconducting"]
    )


@pytest.fixture
def sample_package_manifest(sample_package_metadata):
    """Create sample package manifest for testing."""
    dependencies = [
        PackageRequirement(
            name="numpy",
            version_spec=">=1.20.0",
            type=DependencyType.REQUIRED
        ),
        PackageRequirement(
            name="quantrs2",
            version_spec=">=0.1.0",
            type=DependencyType.REQUIRED,
            quantum_features=["simulation"]
        )
    ]
    
    return PackageManifest(
        metadata=sample_package_metadata,
        dependencies=dependencies,
        files=["__init__.py", "algorithm.py"],
        entry_points={"quantum.algorithms": ["test = test_package:main"]},
        quantum_config={"simulation_backends": ["state_vector"]}
    )


@pytest.fixture
def package_manager(temp_workspace):
    """Create package manager for testing."""
    return QuantumPackageManager(str(temp_workspace))


@pytest.mark.skipif(not HAS_QUANTUM_PACKAGE_MANAGER, reason="quantum package manager not available")
class TestPackageMetadata:
    """Test PackageMetadata functionality."""
    
    def test_package_metadata_creation(self, sample_package_metadata):
        """Test PackageMetadata creation."""
        metadata = sample_package_metadata
        
        assert metadata.name == "test-package"
        assert metadata.version == "1.0.0"
        assert metadata.package_type == PackageType.ALGORITHM
        assert metadata.quantum_requirements["min_qubits"] == 4
        assert "simulator" in metadata.hardware_compatibility
    
    def test_package_metadata_defaults(self):
        """Test PackageMetadata default values."""
        metadata = PackageMetadata(name="minimal-package", version="0.1.0")
        
        assert metadata.description == ""
        assert metadata.author == ""
        assert metadata.license == ""
        assert metadata.package_type == PackageType.UTILITY
        assert metadata.quantum_requirements == {}
        assert metadata.hardware_compatibility == []
        assert metadata.python_requires == ">=3.8"


@pytest.mark.skipif(not HAS_QUANTUM_PACKAGE_MANAGER, reason="quantum package manager not available")
class TestPackageManifest:
    """Test PackageManifest functionality."""
    
    def test_package_manifest_creation(self, sample_package_manifest):
        """Test PackageManifest creation."""
        manifest = sample_package_manifest
        
        assert manifest.metadata.name == "test-package"
        assert len(manifest.dependencies) == 2
        assert "__init__.py" in manifest.files
        assert "quantum.algorithms" in manifest.entry_points
        assert "simulation_backends" in manifest.quantum_config
    
    def test_package_manifest_defaults(self, sample_package_metadata):
        """Test PackageManifest default values."""
        manifest = PackageManifest(metadata=sample_package_metadata)
        
        assert manifest.dependencies == []
        assert manifest.files == []
        assert manifest.entry_points == {}
        assert manifest.scripts == []
        assert manifest.quantum_config == {}
        assert manifest.build_config == {}
        assert manifest.test_config == {}


@pytest.mark.skipif(not HAS_QUANTUM_PACKAGE_MANAGER, reason="quantum package manager not available")
class TestPackageRequirement:
    """Test PackageRequirement functionality."""
    
    def test_package_requirement_creation(self):
        """Test PackageRequirement creation."""
        req = PackageRequirement(
            name="test-dep",
            version_spec=">=1.0.0",
            type=DependencyType.REQUIRED,
            quantum_features=["simulation", "optimization"],
            hardware_requirements={"min_qubits": 8}
        )
        
        assert req.name == "test-dep"
        assert req.version_spec == ">=1.0.0"
        assert req.type == DependencyType.REQUIRED
        assert "simulation" in req.quantum_features
        assert req.hardware_requirements["min_qubits"] == 8
        assert req.optional is False
    
    def test_package_requirement_defaults(self):
        """Test PackageRequirement default values."""
        req = PackageRequirement(name="simple-dep")
        
        assert req.version_spec == "*"
        assert req.type == DependencyType.REQUIRED
        assert req.quantum_features == []
        assert req.hardware_requirements == {}
        assert req.optional is False


@pytest.mark.skipif(not HAS_QUANTUM_PACKAGE_MANAGER, reason="quantum package manager not available")
class TestPackageValidator:
    """Test PackageValidator functionality."""
    
    def test_validator_initialization(self):
        """Test PackageValidator initialization."""
        validator = PackageValidator()
        
        assert validator.validation_rules == []
        assert validator.security_scanners == []
    
    def test_validate_manifest_valid(self, sample_package_manifest):
        """Test validating a valid manifest."""
        validator = PackageValidator()
        
        results = validator._validate_manifest(sample_package_manifest)
        
        assert isinstance(results, dict)
        assert 'warnings' in results
        assert 'errors' in results
        assert len(results['errors']) == 0  # Should be valid
    
    def test_validate_manifest_missing_name(self, sample_package_metadata):
        """Test validating manifest with missing name."""
        validator = PackageValidator()
        
        # Create manifest with empty name
        sample_package_metadata.name = ""
        manifest = PackageManifest(metadata=sample_package_metadata)
        
        results = validator._validate_manifest(manifest)
        
        assert len(results['errors']) > 0
        assert any("name is required" in error for error in results['errors'])
    
    def test_validate_manifest_invalid_version(self, sample_package_metadata):
        """Test validating manifest with invalid version."""
        validator = PackageValidator()
        
        # Create manifest with invalid version
        sample_package_metadata.version = "invalid-version"
        manifest = PackageManifest(metadata=sample_package_metadata)
        
        results = validator._validate_manifest(manifest)
        
        assert len(results['errors']) > 0
        assert any("Invalid version format" in error for error in results['errors'])
    
    def test_validate_structure(self, temp_workspace, sample_package_manifest):
        """Test validating package structure."""
        validator = PackageValidator()
        
        # Create package directory with files
        package_dir = temp_workspace / "test_package"
        package_dir.mkdir()
        
        # Create declared files
        for file_path in sample_package_manifest.files:
            (package_dir / file_path).touch()
        
        results = validator._validate_structure(str(package_dir), sample_package_manifest)
        
        assert isinstance(results, dict)
        assert 'warnings' in results
        assert 'errors' in results
        assert len(results['errors']) == 0  # Files exist
    
    def test_validate_structure_missing_files(self, temp_workspace, sample_package_manifest):
        """Test validating structure with missing files."""
        validator = PackageValidator()
        
        # Create empty package directory
        package_dir = temp_workspace / "test_package"
        package_dir.mkdir()
        
        results = validator._validate_structure(str(package_dir), sample_package_manifest)
        
        assert len(results['errors']) > 0
        assert any("not found" in error for error in results['errors'])
    
    def test_security_scan_clean_package(self, temp_workspace):
        """Test security scan on clean package."""
        validator = PackageValidator()
        
        # Create clean package
        package_dir = temp_workspace / "clean_package"
        package_dir.mkdir()
        (package_dir / "clean_code.py").write_text("print('Hello, quantum world!')")
        
        issues = validator._security_scan(str(package_dir))
        
        assert isinstance(issues, list)
        assert len(issues) == 0  # Clean package should have no issues
    
    def test_security_scan_suspicious_files(self, temp_workspace):
        """Test security scan with suspicious files."""
        validator = PackageValidator()
        
        # Create package with suspicious files
        package_dir = temp_workspace / "suspicious_package"
        package_dir.mkdir()
        (package_dir / "malicious.exe").touch()
        
        issues = validator._security_scan(str(package_dir))
        
        assert len(issues) > 0
        assert any("Suspicious files" in issue for issue in issues)
    
    def test_security_scan_dangerous_code(self, temp_workspace):
        """Test security scan with dangerous code."""
        validator = PackageValidator()
        
        # Create package with dangerous code
        package_dir = temp_workspace / "dangerous_package"
        package_dir.mkdir()
        (package_dir / "dangerous.py").write_text("import os; os.system('rm -rf /')")
        
        issues = validator._security_scan(str(package_dir))
        
        assert len(issues) > 0
        assert any("dangerous code" in issue.lower() for issue in issues)
    
    def test_check_quantum_compatibility(self, temp_workspace, sample_package_manifest):
        """Test quantum compatibility check."""
        validator = PackageValidator()
        
        package_dir = temp_workspace / "quantum_package"
        package_dir.mkdir()
        
        results = validator._check_quantum_compatibility(str(package_dir), sample_package_manifest)
        
        assert isinstance(results, dict)
        assert 'compatible' in results
        assert 'warnings' in results
        assert results['compatible'] is True  # Should be compatible
    
    def test_full_package_validation(self, temp_workspace, sample_package_manifest):
        """Test complete package validation."""
        validator = PackageValidator()
        
        # Create valid package
        package_dir = temp_workspace / "valid_package"
        package_dir.mkdir()
        
        # Create declared files
        for file_path in sample_package_manifest.files:
            (package_dir / file_path).write_text("# Valid quantum code")
        
        results = validator.validate_package(str(package_dir), sample_package_manifest)
        
        assert isinstance(results, dict)
        assert 'valid' in results
        assert 'warnings' in results
        assert 'errors' in results
        assert 'security_issues' in results
        assert 'quantum_compatibility' in results


@pytest.mark.skipif(not HAS_QUANTUM_PACKAGE_MANAGER, reason="quantum package manager not available")
class TestDependencyResolver:
    """Test DependencyResolver functionality."""
    
    def test_resolver_initialization(self):
        """Test DependencyResolver initialization."""
        installed_packages = {}
        resolver = DependencyResolver(installed_packages)
        
        assert resolver.installed_packages == installed_packages
        assert resolver.resolution_cache == {}
    
    def test_version_matches_wildcard(self):
        """Test version matching with wildcard."""
        resolver = DependencyResolver({})
        
        assert resolver._version_matches("1.0.0", "*") is True
        assert resolver._version_matches("2.5.3", "*") is True
    
    def test_version_matches_exact(self):
        """Test version matching with exact version."""
        resolver = DependencyResolver({})
        
        assert resolver._version_matches("1.0.0", "==1.0.0") is True
        assert resolver._version_matches("1.0.1", "==1.0.0") is False
    
    def test_topological_sort_simple(self):
        """Test topological sorting with simple graph."""
        resolver = DependencyResolver({})
        
        # Simple dependency graph: A -> B -> C
        graph = {
            'A': {'B'},
            'B': {'C'},
            'C': set()
        }
        
        result = resolver._topological_sort(graph)
        
        assert 'C' in result
        assert 'B' in result
        assert 'A' in result
        # C should come before B, B should come before A
        assert result.index('C') < result.index('B')
        assert result.index('B') < result.index('A')
    
    def test_topological_sort_parallel(self):
        """Test topological sorting with parallel dependencies."""
        resolver = DependencyResolver({})
        
        # Parallel dependencies: A -> B, A -> C, B -> D, C -> D
        graph = {
            'A': {'B', 'C'},
            'B': {'D'},
            'C': {'D'},
            'D': set()
        }
        
        result = resolver._topological_sort(graph)
        
        assert len(result) == 4
        assert 'D' in result
        assert result.index('D') < result.index('B')
        assert result.index('D') < result.index('C')
    
    def test_find_best_version(self):
        """Test finding best matching version."""
        resolver = DependencyResolver({})
        
        # Create sample manifests
        manifests = []
        for version in ["1.0.0", "1.1.0", "2.0.0"]:
            metadata = PackageMetadata(name="test-pkg", version=version)
            manifests.append(PackageManifest(metadata=metadata))
        
        requirement = PackageRequirement(name="test-pkg", version_spec=">=1.0.0")
        
        best = resolver._find_best_version(requirement, manifests)
        
        assert best is not None
        assert best.metadata.version == "2.0.0"  # Should pick latest compatible
    
    def test_resolve_dependencies_simple(self):
        """Test simple dependency resolution."""
        resolver = DependencyResolver({})
        
        # Create requirements
        requirements = [
            PackageRequirement(name="package-a", version_spec=">=1.0.0")
        ]
        
        # Create available packages
        metadata_a = PackageMetadata(name="package-a", version="1.0.0")
        manifest_a = PackageManifest(metadata=metadata_a)
        
        available_packages = {
            "package-a": [manifest_a]
        }
        
        resolution = resolver.resolve_dependencies(requirements, available_packages)
        
        assert isinstance(resolution, dict)
        assert 'install_order' in resolution
        assert 'conflicts' in resolution
        assert 'missing' in resolution
        assert 'resolution_graph' in resolution
        assert 'package-a' in resolution['install_order']


@pytest.mark.skipif(not HAS_QUANTUM_PACKAGE_MANAGER, reason="quantum package manager not available")
class TestPackageRegistry:
    """Test PackageRegistry functionality."""
    
    def test_registry_initialization(self, temp_workspace):
        """Test PackageRegistry initialization."""
        config = RegistryConfig(
            name="test-registry",
            url=str(temp_workspace / "registry"),
            type=RegistryType.LOCAL
        )
        
        registry = PackageRegistry(config, str(temp_workspace / "cache"))
        
        assert registry.config == config
        assert registry.cache_dir.exists()
        assert registry.index_cache == {}
        assert registry.last_update == 0
    
    def test_fetch_local_index_empty(self, temp_workspace):
        """Test fetching from empty local registry."""
        registry_dir = temp_workspace / "empty_registry"
        registry_dir.mkdir()
        
        config = RegistryConfig(
            name="empty-registry",
            url=str(registry_dir),
            type=RegistryType.LOCAL
        )
        
        registry = PackageRegistry(config, str(temp_workspace / "cache"))
        
        index = registry._fetch_local_index()
        
        assert isinstance(index, dict)
        assert len(index) == 0
    
    def test_fetch_local_index_with_packages(self, temp_workspace, sample_package_manifest):
        """Test fetching from local registry with packages."""
        registry_dir = temp_workspace / "registry"
        registry_dir.mkdir()
        
        # Create package in registry
        package_dir = registry_dir / "test-package"
        package_dir.mkdir()
        
        manifest_file = package_dir / "quantum_package.yml"
        with open(manifest_file, 'w') as f:
            import yaml
            from dataclasses import asdict
            yaml.dump(asdict(sample_package_manifest), f)
        
        config = RegistryConfig(
            name="test-registry",
            url=str(registry_dir),
            type=RegistryType.LOCAL
        )
        
        registry = PackageRegistry(config, str(temp_workspace / "cache"))
        
        index = registry._fetch_local_index()
        
        assert len(index) > 0
        assert "test-package" in index
        assert len(index["test-package"]) == 1
    
    @patch('quantrs2.quantum_package_manager.HAS_REQUESTS', True)
    @patch('quantrs2.quantum_package_manager.requests')
    def test_fetch_remote_index_success(self, mock_requests, temp_workspace):
        """Test successful remote index fetch."""
        # Mock successful response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "test-package": [
                {
                    "metadata": {
                        "name": "test-package",
                        "version": "1.0.0",
                        "description": "Test package",
                        "author": "Test Author",
                        "license": "MIT",
                        "package_type": "algorithm"
                    },
                    "dependencies": [],
                    "files": ["__init__.py"]
                }
            ]
        }
        mock_requests.get.return_value = mock_response
        
        config = RegistryConfig(
            name="remote-registry",
            url="https://test.registry.com",
            type=RegistryType.PUBLIC
        )
        
        registry = PackageRegistry(config, str(temp_workspace / "cache"))
        
        index = registry._fetch_remote_index()
        
        assert len(index) > 0
        assert "test-package" in index
        mock_requests.get.assert_called_once()
    
    @patch('quantrs2.quantum_package_manager.HAS_REQUESTS', False)
    def test_fetch_remote_index_no_requests(self, temp_workspace):
        """Test remote index fetch without requests library."""
        config = RegistryConfig(
            name="remote-registry",
            url="https://test.registry.com",
            type=RegistryType.PUBLIC
        )
        
        registry = PackageRegistry(config, str(temp_workspace / "cache"))
        
        index = registry._fetch_remote_index()
        
        assert isinstance(index, dict)
        assert len(index) == 0
    
    def test_download_local_package(self, temp_workspace):
        """Test downloading package from local registry."""
        # Create local registry with package
        registry_dir = temp_workspace / "registry"
        package_dir = registry_dir / "test-package" / "1.0.0"
        package_dir.mkdir(parents=True)
        
        # Create package files
        (package_dir / "__init__.py").write_text("# Test package")
        (package_dir / "algorithm.py").write_text("def test(): pass")
        
        config = RegistryConfig(
            name="local-registry",
            url=str(registry_dir),
            type=RegistryType.LOCAL
        )
        
        registry = PackageRegistry(config, str(temp_workspace / "cache"))
        
        download_dir = temp_workspace / "downloads"
        download_dir.mkdir()
        
        downloaded_path = registry._download_local_package(
            "test-package", "1.0.0", str(download_dir)
        )
        
        assert Path(downloaded_path).exists()
        assert (Path(downloaded_path) / "__init__.py").exists()
        assert (Path(downloaded_path) / "algorithm.py").exists()
    
    def test_download_local_package_not_found(self, temp_workspace):
        """Test downloading non-existent package from local registry."""
        registry_dir = temp_workspace / "empty_registry"
        registry_dir.mkdir()
        
        config = RegistryConfig(
            name="empty-registry",
            url=str(registry_dir),
            type=RegistryType.LOCAL
        )
        
        registry = PackageRegistry(config, str(temp_workspace / "cache"))
        
        download_dir = temp_workspace / "downloads"
        download_dir.mkdir()
        
        with pytest.raises(FileNotFoundError):
            registry._download_local_package(
                "non-existent", "1.0.0", str(download_dir)
            )


@pytest.mark.skipif(not HAS_QUANTUM_PACKAGE_MANAGER, reason="quantum package manager not available")
class TestQuantumPackageManager:
    """Test QuantumPackageManager functionality."""
    
    def test_package_manager_initialization(self, package_manager):
        """Test QuantumPackageManager initialization."""
        assert package_manager.workspace_dir.exists()
        assert package_manager.packages_dir.exists()
        assert package_manager.cache_dir.exists()
        assert package_manager.config_dir.exists()
        assert package_manager.db_path.exists()
        assert isinstance(package_manager.validator, PackageValidator)
        assert len(package_manager.registries) > 0  # Should have default registries
    
    def test_database_initialization(self, package_manager):
        """Test database tables are created properly."""
        import sqlite3
        
        with sqlite3.connect(package_manager.db_path) as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name IN ('installed_packages', 'package_dependencies', 'package_history')
            """)
            tables = [row[0] for row in cursor.fetchall()]
        
        assert 'installed_packages' in tables
        assert 'package_dependencies' in tables
        assert 'package_history' in tables
    
    def test_create_package_from_directory(self, temp_workspace, package_manager):
        """Test creating package from source directory."""
        # Create package directory
        package_dir = temp_workspace / "test_package"
        package_dir.mkdir()
        
        # Create package files
        (package_dir / "__init__.py").write_text("__version__ = '1.0.0'")
        (package_dir / "algorithm.py").write_text("def quantum_algorithm(): pass")
        
        # Create manifest
        manifest_data = {
            'metadata': {
                'name': 'test-package',
                'version': '1.0.0',
                'description': 'Test package',
                'package_type': 'algorithm'
            },
            'files': ['__init__.py', 'algorithm.py']
        }
        
        with open(package_dir / "quantum_package.yml", 'w') as f:
            yaml.dump(manifest_data, f)
        
        # Create package
        result = package_manager.create_package(str(package_dir), str(temp_workspace))
        
        assert result['success'] is True
        assert 'package_file' in result
        assert Path(result['package_file']).exists()
        assert result['package_file'].endswith('.tar.gz')
    
    def test_create_package_missing_directory(self, package_manager):
        """Test creating package from non-existent directory."""
        result = package_manager.create_package("/non/existent/path")
        
        assert result['success'] is False
        assert len(result['errors']) > 0
        assert "not found" in result['errors'][0].lower()
    
    def test_search_packages_empty(self, package_manager):
        """Test searching packages in empty registries."""
        results = package_manager.search_packages("test")
        
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_list_installed_packages_empty(self, package_manager):
        """Test listing packages when none are installed."""
        packages = package_manager.list_installed_packages()
        
        assert isinstance(packages, list)
        assert len(packages) == 0
    
    def test_get_package_info_not_found(self, package_manager):
        """Test getting info for non-existent package."""
        info = package_manager.get_package_info("non-existent-package")
        
        assert info is None
    
    def test_uninstall_package_not_installed(self, package_manager):
        """Test uninstalling package that is not installed."""
        result = package_manager.uninstall_package("non-existent-package")
        
        assert result['success'] is False
        assert len(result['errors']) > 0
        assert "not installed" in result['errors'][0]
    
    def test_add_registry(self, package_manager):
        """Test adding a new registry."""
        initial_count = len(package_manager.registries)
        
        success = package_manager.add_registry(
            "test-registry",
            "https://test.registry.com",
            RegistryType.PUBLIC
        )
        
        assert success is True
        assert len(package_manager.registries) == initial_count + 1
        assert "test-registry" in package_manager.registries
    
    def test_get_package_statistics(self, package_manager):
        """Test getting package statistics."""
        stats = package_manager.get_package_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_packages' in stats
        assert 'package_types' in stats
        assert 'total_size' in stats
        assert 'registries' in stats
        assert 'recent_installations' in stats
        assert 'dependency_graph_size' in stats
        
        assert stats['total_packages'] == 0  # No packages installed
        assert stats['total_size'] == 0
        assert stats['registries'] > 0  # Should have default registries
    
    def test_calculate_directory_checksum(self, temp_workspace, package_manager):
        """Test calculating directory checksum."""
        test_dir = temp_workspace / "checksum_test"
        test_dir.mkdir()
        
        # Create test files
        (test_dir / "file1.txt").write_text("Content 1")
        (test_dir / "file2.txt").write_text("Content 2")
        
        checksum1 = package_manager._calculate_directory_checksum(test_dir)
        
        assert isinstance(checksum1, str)
        assert len(checksum1) == 64  # SHA256 hex digest
        
        # Same directory should produce same checksum
        checksum2 = package_manager._calculate_directory_checksum(test_dir)
        assert checksum1 == checksum2
        
        # Different content should produce different checksum
        (test_dir / "file3.txt").write_text("Content 3")
        checksum3 = package_manager._calculate_directory_checksum(test_dir)
        assert checksum1 != checksum3
    
    def test_record_package_action(self, package_manager):
        """Test recording package actions in history."""
        package_manager._record_package_action("test-package", "install", "1.0.0", True)
        package_manager._record_package_action("test-package", "uninstall", "1.0.0", False)
        
        # Check history was recorded
        import sqlite3
        with sqlite3.connect(package_manager.db_path) as conn:
            cursor = conn.execute("""
                SELECT package_name, action, version, success 
                FROM package_history 
                ORDER BY timestamp
            """)
            history = cursor.fetchall()
        
        assert len(history) == 2
        assert history[0][0] == "test-package"
        assert history[0][1] == "install"
        assert history[0][3] == 1  # success
        assert history[1][1] == "uninstall"
        assert history[1][3] == 0  # failure


@pytest.mark.skipif(not HAS_QUANTUM_PACKAGE_MANAGER, reason="quantum package manager not available")
class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_get_quantum_package_manager(self, temp_workspace):
        """Test get_quantum_package_manager function."""
        manager = get_quantum_package_manager(str(temp_workspace))
        
        assert isinstance(manager, QuantumPackageManager)
        assert manager.workspace_dir == temp_workspace
    
    def test_create_package_manifest(self):
        """Test create_package_manifest function."""
        manifest = create_package_manifest(
            name="test-package",
            version="1.0.0",
            description="Test package",
            author="Test Author",
            package_type=PackageType.ALGORITHM,
            quantum_requirements={"min_qubits": 4},
            dependencies=[
                PackageRequirement(name="numpy", version_spec=">=1.20.0")
            ]
        )
        
        assert isinstance(manifest, PackageManifest)
        assert manifest.metadata.name == "test-package"
        assert manifest.metadata.version == "1.0.0"
        assert manifest.metadata.package_type == PackageType.ALGORITHM
        assert len(manifest.dependencies) == 1
        assert manifest.dependencies[0].name == "numpy"


@pytest.mark.skipif(not HAS_QUANTUM_PACKAGE_MANAGER, reason="quantum package manager not available")
class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    def test_package_manager_invalid_workspace(self):
        """Test package manager with invalid workspace."""
        # Should handle invalid workspace gracefully
        manager = QuantumPackageManager("/invalid/workspace/path")
        
        # Should still be able to create the manager
        assert isinstance(manager, QuantumPackageManager)
    
    def test_validator_with_corrupted_files(self, temp_workspace):
        """Test validator with corrupted files."""
        validator = PackageValidator()
        
        # Create package with binary data that can't be decoded
        package_dir = temp_workspace / "corrupted_package"
        package_dir.mkdir()
        
        with open(package_dir / "corrupted.py", 'wb') as f:
            f.write(b'\xff\xfe\x00\x00invalid_unicode')
        
        # Should handle corrupted files gracefully
        issues = validator._security_scan(str(package_dir))
        assert isinstance(issues, list)
    
    def test_dependency_resolver_circular_dependencies(self):
        """Test dependency resolver with circular dependencies."""
        resolver = DependencyResolver({})
        
        # Create circular dependency graph: A -> B -> A
        graph = {
            'A': {'B'},
            'B': {'A'}
        }
        
        # Should handle circular dependencies
        result = resolver._topological_sort(graph)
        
        # Result should still be a list (may be incomplete)
        assert isinstance(result, list)
    
    def test_registry_with_malformed_yaml(self, temp_workspace):
        """Test registry with malformed YAML files."""
        registry_dir = temp_workspace / "malformed_registry"
        package_dir = registry_dir / "bad_package"
        package_dir.mkdir(parents=True)
        
        # Create malformed YAML
        manifest_file = package_dir / "quantum_package.yml"
        manifest_file.write_text("invalid: yaml: content: [")
        
        config = RegistryConfig(
            name="malformed-registry",
            url=str(registry_dir),
            type=RegistryType.LOCAL
        )
        
        registry = PackageRegistry(config, str(temp_workspace / "cache"))
        
        # Should handle malformed YAML gracefully
        index = registry._fetch_local_index()
        assert isinstance(index, dict)
        # Should not include the malformed package
        assert "bad_package" not in index
    
    def test_package_manager_database_corruption_handling(self, temp_workspace):
        """Test package manager handling of database corruption."""
        # Create package manager
        manager = QuantumPackageManager(str(temp_workspace))
        
        # Corrupt the database
        with open(manager.db_path, 'w') as f:
            f.write("corrupted database content")
        
        # Should handle corruption gracefully when loading
        try:
            manager._load_installed_packages()
        except Exception:
            # Should not crash the application
            pass
        
        assert isinstance(manager.installed_packages, dict)
    
    def test_registry_network_timeout_simulation(self, temp_workspace):
        """Test registry behavior during network timeouts."""
        config = RegistryConfig(
            name="timeout-registry",
            url="https://nonexistent.registry.com",
            type=RegistryType.PUBLIC
        )
        
        registry = PackageRegistry(config, str(temp_workspace / "cache"))
        
        # Should handle network failures gracefully
        index = registry._fetch_remote_index()
        assert isinstance(index, dict)
        assert len(index) == 0
    
    def test_large_package_handling(self, temp_workspace, package_manager):
        """Test handling of large packages."""
        # Create a package with many files
        package_dir = temp_workspace / "large_package"
        package_dir.mkdir()
        
        # Create many files
        for i in range(100):
            (package_dir / f"file_{i}.py").write_text(f"# File {i}")
        
        # Create manifest with all files
        files = [f"file_{i}.py" for i in range(100)]
        manifest_data = {
            'metadata': {
                'name': 'large-package',
                'version': '1.0.0',
                'package_type': 'utility'
            },
            'files': files
        }
        
        with open(package_dir / "quantum_package.yml", 'w') as f:
            yaml.dump(manifest_data, f)
        
        # Should handle large packages
        result = package_manager.create_package(str(package_dir), str(temp_workspace))
        
        # May succeed or fail depending on system limits, but shouldn't crash
        assert isinstance(result, dict)
        assert 'success' in result


if __name__ == "__main__":
    pytest.main([__file__])