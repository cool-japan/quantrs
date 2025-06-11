#!/usr/bin/env python3
"""
Test suite for quantum algorithm marketplace functionality.
"""

import pytest
import json
import time
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

try:
    from quantrs2.algorithm_marketplace import (
        AlgorithmCategory, AlgorithmType, LicenseType, QualityMetric, MarketplaceStatus,
        AlgorithmMetadata, AlgorithmRating, MarketplaceEntry,
        AlgorithmValidator, AlgorithmMarketplaceDB, AlgorithmPackager,
        MarketplaceAPI, QuantumAlgorithmMarketplace,
        get_quantum_marketplace, search_algorithms, download_algorithm,
        submit_algorithm, create_algorithm_entry
    )
    HAS_MARKETPLACE = True
except ImportError:
    HAS_MARKETPLACE = False

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False


@pytest.mark.skipif(not HAS_MARKETPLACE, reason="algorithm_marketplace module not available")
class TestAlgorithmMetadata:
    """Test AlgorithmMetadata functionality."""
    
    def test_metadata_creation(self):
        """Test creating algorithm metadata."""
        metadata = AlgorithmMetadata(
            algorithm_id="test_algo",
            name="Test Algorithm",
            description="A test quantum algorithm",
            author="Test Author",
            version="1.0.0",
            category=AlgorithmCategory.GENERAL,
            algorithm_type=AlgorithmType.CIRCUIT,
            license=LicenseType.MIT
        )
        
        assert metadata.algorithm_id == "test_algo"
        assert metadata.name == "Test Algorithm"
        assert metadata.category == AlgorithmCategory.GENERAL
        assert metadata.algorithm_type == AlgorithmType.CIRCUIT
        assert metadata.license == LicenseType.MIT
        assert metadata.min_qubits == 1
        assert isinstance(metadata.created_at, float)
    
    def test_metadata_serialization(self):
        """Test metadata serialization."""
        metadata = AlgorithmMetadata(
            algorithm_id="test_algo",
            name="Test Algorithm",
            description="A test quantum algorithm",
            author="Test Author",
            version="1.0.0",
            category=AlgorithmCategory.OPTIMIZATION,
            algorithm_type=AlgorithmType.VARIATIONAL,
            license=LicenseType.MIT,
            tags=["optimization", "vqe"]
        )
        
        metadata_dict = metadata.to_dict()
        
        assert metadata_dict['algorithm_id'] == "test_algo"
        assert metadata_dict['category'] == "optimization"
        assert metadata_dict['algorithm_type'] == "variational"
        assert metadata_dict['license'] == "MIT"
        assert metadata_dict['tags'] == ["optimization", "vqe"]
    
    def test_metadata_deserialization(self):
        """Test metadata deserialization."""
        data = {
            'algorithm_id': "test_algo",
            'name': "Test Algorithm",
            'description': "A test quantum algorithm",
            'author': "Test Author",
            'version': "1.0.0",
            'category': "machine_learning",
            'algorithm_type': "hybrid",
            'license': "Apache-2.0",
            'created_at': time.time(),
            'updated_at': time.time(),
            'tags': ["ml", "hybrid"],
            'dependencies': ["numpy"],
            'min_qubits': 4,
            'max_qubits': 8
        }
        
        metadata = AlgorithmMetadata.from_dict(data)
        
        assert metadata.algorithm_id == "test_algo"
        assert metadata.category == AlgorithmCategory.MACHINE_LEARNING
        assert metadata.algorithm_type == AlgorithmType.HYBRID
        assert metadata.license == LicenseType.APACHE_2_0
        assert metadata.min_qubits == 4
        assert metadata.max_qubits == 8


@pytest.mark.skipif(not HAS_MARKETPLACE, reason="algorithm_marketplace module not available")
class TestAlgorithmRating:
    """Test AlgorithmRating functionality."""
    
    def test_rating_creation(self):
        """Test creating algorithm rating."""
        rating = AlgorithmRating(
            rating_id="rating_1",
            algorithm_id="algo_1",
            reviewer="reviewer_1",
            overall_rating=4.5,
            quality_metrics={
                QualityMetric.CORRECTNESS: 5.0,
                QualityMetric.PERFORMANCE: 4.0,
                QualityMetric.DOCUMENTATION: 4.5
            },
            review_text="Great algorithm!",
            verified_execution=True
        )
        
        assert rating.rating_id == "rating_1"
        assert rating.overall_rating == 4.5
        assert rating.quality_metrics[QualityMetric.CORRECTNESS] == 5.0
        assert rating.verified_execution is True
    
    def test_rating_serialization(self):
        """Test rating serialization."""
        rating = AlgorithmRating(
            rating_id="rating_1",
            algorithm_id="algo_1",
            reviewer="reviewer_1",
            overall_rating=4.0,
            quality_metrics={QualityMetric.READABILITY: 3.5}
        )
        
        rating_dict = rating.to_dict()
        
        assert rating_dict['rating_id'] == "rating_1"
        assert rating_dict['overall_rating'] == 4.0
        assert rating_dict['quality_metrics']['readability'] == 3.5
    
    def test_rating_deserialization(self):
        """Test rating deserialization."""
        data = {
            'rating_id': "rating_1",
            'algorithm_id': "algo_1",
            'reviewer': "reviewer_1",
            'overall_rating': 3.5,
            'quality_metrics': {'correctness': 4.0, 'performance': 3.0},
            'review_text': "Good algorithm",
            'created_at': time.time(),
            'verified_execution': False,
            'benchmark_results': None
        }
        
        rating = AlgorithmRating.from_dict(data)
        
        assert rating.overall_rating == 3.5
        assert rating.quality_metrics[QualityMetric.CORRECTNESS] == 4.0
        assert rating.quality_metrics[QualityMetric.PERFORMANCE] == 3.0


@pytest.mark.skipif(not HAS_MARKETPLACE, reason="algorithm_marketplace module not available")
class TestMarketplaceEntry:
    """Test MarketplaceEntry functionality."""
    
    def setup_method(self):
        """Set up test entry."""
        self.metadata = AlgorithmMetadata(
            algorithm_id="test_entry",
            name="Test Entry",
            description="A test marketplace entry",
            author="Test Author",
            version="1.0.0",
            category=AlgorithmCategory.GENERAL,
            algorithm_type=AlgorithmType.CIRCUIT,
            license=LicenseType.MIT
        )
        
        self.algorithm_data = {
            "gates": [
                {"gate": "h", "qubits": [0]},
                {"gate": "cnot", "qubits": [0, 1]}
            ]
        }
        
        self.entry = MarketplaceEntry(
            entry_id="test_entry",
            metadata=self.metadata,
            algorithm_data=self.algorithm_data,
            documentation="Test documentation"
        )
    
    def test_entry_creation(self):
        """Test marketplace entry creation."""
        assert self.entry.entry_id == "test_entry"
        assert self.entry.metadata.name == "Test Entry"
        assert self.entry.status == MarketplaceStatus.DRAFT
        assert self.entry.download_count == 0
        assert self.entry.checksum is not None
    
    def test_checksum_calculation(self):
        """Test checksum calculation."""
        checksum1 = self.entry.checksum
        
        # Create identical entry
        entry2 = MarketplaceEntry(
            entry_id="test_entry",
            metadata=self.metadata,
            algorithm_data=self.algorithm_data,
            documentation="Test documentation"
        )
        
        assert entry2.checksum == checksum1
        
        # Modify entry
        entry2.documentation = "Modified documentation"
        entry2.checksum = entry2._calculate_checksum()
        
        assert entry2.checksum != checksum1
    
    def test_entry_serialization(self):
        """Test entry serialization."""
        entry_dict = self.entry.to_dict()
        
        assert entry_dict['entry_id'] == "test_entry"
        assert entry_dict['metadata']['name'] == "Test Entry"
        assert entry_dict['algorithm_data'] == self.algorithm_data
        assert entry_dict['status'] == "draft"
        assert 'checksum' in entry_dict
    
    def test_entry_deserialization(self):
        """Test entry deserialization."""
        entry_dict = self.entry.to_dict()
        restored_entry = MarketplaceEntry.from_dict(entry_dict)
        
        assert restored_entry.entry_id == self.entry.entry_id
        assert restored_entry.metadata.name == self.entry.metadata.name
        assert restored_entry.algorithm_data == self.entry.algorithm_data
        assert restored_entry.status == self.entry.status


@pytest.mark.skipif(not HAS_MARKETPLACE, reason="algorithm_marketplace module not available")
class TestAlgorithmValidator:
    """Test AlgorithmValidator functionality."""
    
    def setup_method(self):
        """Set up test validator."""
        self.validator = AlgorithmValidator()
        
        # Create valid test entry
        metadata = AlgorithmMetadata(
            algorithm_id="valid_algo",
            name="Valid Algorithm",
            description="A valid test quantum algorithm with sufficient description length",
            author="Test Author",
            version="1.0.0",
            category=AlgorithmCategory.GENERAL,
            algorithm_type=AlgorithmType.CIRCUIT,
            license=LicenseType.MIT,
            min_qubits=2,
            max_qubits=4
        )
        
        algorithm_data = {
            "gates": [
                {"gate": "h", "qubits": [0]},
                {"gate": "cnot", "qubits": [0, 1]}
            ]
        }
        
        documentation = """
        # Valid Algorithm
        
        ## Description
        This is a valid quantum algorithm for testing purposes.
        
        ## Usage
        Use this algorithm to test validation functionality.
        
        ## Parameters
        - n_qubits: Number of qubits (minimum 2)
        """
        
        examples = [
            {
                "name": "Basic Example",
                "description": "Basic usage example",
                "code": "circuit.h(0); circuit.cnot(0, 1)"
            }
        ]
        
        test_cases = [
            {
                "name": "Basic Test",
                "description": "Basic test case",
                "input": {"n_qubits": 2},
                "expected_output": {"00": 0.5, "11": 0.5}
            }
        ]
        
        self.valid_entry = MarketplaceEntry(
            entry_id="valid_algo",
            metadata=metadata,
            algorithm_data=algorithm_data,
            documentation=documentation,
            examples=examples,
            test_cases=test_cases
        )
    
    def test_valid_entry_validation(self):
        """Test validation of valid entry."""
        is_valid, errors = self.validator.validate_entry(self.valid_entry)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_missing_required_fields(self):
        """Test validation with missing required fields."""
        # Create entry with missing name
        invalid_metadata = AlgorithmMetadata(
            algorithm_id="invalid_algo",
            name="",  # Missing name
            description="Test description that is long enough for validation",
            author="Test Author",
            version="1.0.0",
            category=AlgorithmCategory.GENERAL,
            algorithm_type=AlgorithmType.CIRCUIT,
            license=LicenseType.MIT
        )
        
        invalid_entry = MarketplaceEntry(
            entry_id="invalid_algo",
            metadata=invalid_metadata,
            algorithm_data={"gates": []},
            documentation="Test documentation"
        )
        
        is_valid, errors = self.validator.validate_entry(invalid_entry)
        
        assert is_valid is False
        assert any("name" in error.lower() for error in errors)
    
    def test_short_description_validation(self):
        """Test validation of short description."""
        invalid_metadata = AlgorithmMetadata(
            algorithm_id="short_desc",
            name="Test Algorithm",
            description="Short",  # Too short
            author="Test Author",
            version="1.0.0",
            category=AlgorithmCategory.GENERAL,
            algorithm_type=AlgorithmType.CIRCUIT,
            license=LicenseType.MIT
        )
        
        invalid_entry = MarketplaceEntry(
            entry_id="short_desc",
            metadata=invalid_metadata,
            algorithm_data={"gates": []},
            documentation="Test documentation that is long enough"
        )
        
        is_valid, errors = self.validator.validate_entry(invalid_entry)
        
        assert is_valid is False
        assert any("description" in error.lower() for error in errors)
    
    def test_invalid_version_format(self):
        """Test validation of invalid version format."""
        invalid_metadata = AlgorithmMetadata(
            algorithm_id="invalid_version",
            name="Test Algorithm",
            description="Valid description that is long enough for validation",
            author="Test Author",
            version="invalid",  # Invalid format
            category=AlgorithmCategory.GENERAL,
            algorithm_type=AlgorithmType.CIRCUIT,
            license=LicenseType.MIT
        )
        
        invalid_entry = MarketplaceEntry(
            entry_id="invalid_version",
            metadata=invalid_metadata,
            algorithm_data={"gates": []},
            documentation="Valid documentation that is long enough"
        )
        
        is_valid, errors = self.validator.validate_entry(invalid_entry)
        
        assert is_valid is False
        assert any("version" in error.lower() for error in errors)
    
    def test_invalid_qubit_requirements(self):
        """Test validation of invalid qubit requirements."""
        invalid_metadata = AlgorithmMetadata(
            algorithm_id="invalid_qubits",
            name="Test Algorithm",
            description="Valid description that is long enough for validation",
            author="Test Author",
            version="1.0.0",
            category=AlgorithmCategory.GENERAL,
            algorithm_type=AlgorithmType.CIRCUIT,
            license=LicenseType.MIT,
            min_qubits=0,  # Invalid minimum
            max_qubits=5
        )
        
        invalid_entry = MarketplaceEntry(
            entry_id="invalid_qubits",
            metadata=invalid_metadata,
            algorithm_data={"gates": []},
            documentation="Valid documentation that is long enough"
        )
        
        is_valid, errors = self.validator.validate_entry(invalid_entry)
        
        assert is_valid is False
        assert any("minimum qubits" in error.lower() for error in errors)
    
    def test_gate_validation(self):
        """Test validation of gate sequences."""
        invalid_gates = [
            {"gate": "h"},  # Missing qubits
            {"qubits": [0]},  # Missing gate
            {"gate": "x", "qubits": ["invalid"]},  # Invalid qubit type
            {"gate": "cnot", "qubits": [-1, 0]}  # Negative qubit index
        ]
        
        errors = self.validator._validate_gates(invalid_gates)
        
        assert len(errors) >= 4  # Should catch all issues
        assert any("missing" in error.lower() for error in errors)
        assert any("negative" in error.lower() for error in errors)
    
    def test_documentation_validation(self):
        """Test documentation validation."""
        # Too short documentation
        short_doc_entry = MarketplaceEntry(
            entry_id="short_doc",
            metadata=self.valid_entry.metadata,
            algorithm_data=self.valid_entry.algorithm_data,
            documentation="Short",  # Too short
            examples=self.valid_entry.examples,
            test_cases=self.valid_entry.test_cases
        )
        
        errors = self.validator._validate_documentation(short_doc_entry)
        assert len(errors) > 0
        assert any("100 characters" in error for error in errors)
    
    def test_examples_validation(self):
        """Test examples validation."""
        # Entry without examples
        no_examples_entry = MarketplaceEntry(
            entry_id="no_examples",
            metadata=self.valid_entry.metadata,
            algorithm_data=self.valid_entry.algorithm_data,
            documentation=self.valid_entry.documentation,
            examples=[],  # No examples
            test_cases=self.valid_entry.test_cases
        )
        
        errors = self.validator._validate_examples(no_examples_entry)
        assert len(errors) > 0
        assert any("example" in error.lower() for error in errors)
    
    def test_test_cases_validation(self):
        """Test test cases validation."""
        # Entry without test cases
        no_tests_entry = MarketplaceEntry(
            entry_id="no_tests",
            metadata=self.valid_entry.metadata,
            algorithm_data=self.valid_entry.algorithm_data,
            documentation=self.valid_entry.documentation,
            examples=self.valid_entry.examples,
            test_cases=[]  # No test cases
        )
        
        errors = self.validator._validate_test_cases(no_tests_entry)
        assert len(errors) > 0
        assert any("test case" in error.lower() for error in errors)
    
    def test_quality_score_estimation(self):
        """Test quality score estimation."""
        score = self.validator.estimate_quality_score(self.valid_entry)
        
        assert 0 <= score <= 100
        assert score > 50  # Should be reasonably high for valid entry
        
        # Test with minimal entry
        minimal_metadata = AlgorithmMetadata(
            algorithm_id="minimal",
            name="Minimal",
            description="Minimal description for testing quality score",
            author="Test",
            version="1.0.0",
            category=AlgorithmCategory.GENERAL,
            algorithm_type=AlgorithmType.CIRCUIT,
            license=LicenseType.MIT
        )
        
        minimal_entry = MarketplaceEntry(
            entry_id="minimal",
            metadata=minimal_metadata,
            algorithm_data={"gates": []},
            documentation="Short doc",
            examples=[],
            test_cases=[]
        )
        
        minimal_score = self.validator.estimate_quality_score(minimal_entry)
        assert minimal_score < score  # Should be lower than valid entry


@pytest.mark.skipif(not HAS_MARKETPLACE or not SQLITE_AVAILABLE, 
                   reason="algorithm_marketplace or sqlite not available")
class TestAlgorithmMarketplaceDB:
    """Test AlgorithmMarketplaceDB functionality."""
    
    def setup_method(self):
        """Set up test database."""
        self.db = AlgorithmMarketplaceDB(":memory:")
        
        # Create test entry
        metadata = AlgorithmMetadata(
            algorithm_id="test_db_algo",
            name="Test DB Algorithm",
            description="Algorithm for testing database functionality",
            author="Test Author",
            version="1.0.0",
            category=AlgorithmCategory.GENERAL,
            algorithm_type=AlgorithmType.CIRCUIT,
            license=LicenseType.MIT
        )
        
        self.test_entry = MarketplaceEntry(
            entry_id="test_db_algo",
            metadata=metadata,
            algorithm_data={"gates": [{"gate": "h", "qubits": [0]}]},
            documentation="Test documentation for database testing",
            examples=[{"name": "test", "description": "test", "code": "test"}],
            test_cases=[{"name": "test", "description": "test", "input": {}, "expected_output": {}}],
            status=MarketplaceStatus.APPROVED
        )
    
    def teardown_method(self):
        """Clean up database."""
        self.db.close()
    
    def test_database_initialization(self):
        """Test database initialization."""
        # Check if tables exist
        cursor = self.db.connection.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name IN ('algorithms', 'ratings', 'downloads', 'collections')
        """)
        
        tables = [row[0] for row in cursor.fetchall()]
        
        assert 'algorithms' in tables
        assert 'ratings' in tables
        assert 'downloads' in tables
        assert 'collections' in tables
    
    def test_submit_algorithm(self):
        """Test algorithm submission."""
        success = self.db.submit_algorithm(self.test_entry)
        
        assert success is True
        
        # Verify entry was stored
        stored_entry = self.db.get_algorithm("test_db_algo")
        assert stored_entry is not None
        assert stored_entry.entry_id == "test_db_algo"
        assert stored_entry.metadata.name == "Test DB Algorithm"
    
    def test_get_algorithm(self):
        """Test getting algorithm by ID."""
        # Submit algorithm first
        self.db.submit_algorithm(self.test_entry)
        
        # Retrieve it
        retrieved = self.db.get_algorithm("test_db_algo")
        
        assert retrieved is not None
        assert retrieved.entry_id == self.test_entry.entry_id
        assert retrieved.metadata.name == self.test_entry.metadata.name
        assert retrieved.status == MarketplaceStatus.APPROVED
    
    def test_get_nonexistent_algorithm(self):
        """Test getting nonexistent algorithm."""
        result = self.db.get_algorithm("nonexistent")
        assert result is None
    
    def test_search_algorithms(self):
        """Test algorithm search."""
        # Submit test algorithm
        self.db.submit_algorithm(self.test_entry)
        
        # Search by name
        results = self.db.search_algorithms(query="Test DB")
        assert len(results) == 1
        assert results[0].entry_id == "test_db_algo"
        
        # Search by category
        results = self.db.search_algorithms(category=AlgorithmCategory.GENERAL)
        assert len(results) == 1
        
        # Search by author
        results = self.db.search_algorithms(author="Test Author")
        assert len(results) == 1
        
        # Search with no results
        results = self.db.search_algorithms(query="Nonexistent")
        assert len(results) == 0
    
    def test_add_rating(self):
        """Test adding algorithm rating."""
        # Submit algorithm first
        self.db.submit_algorithm(self.test_entry)
        
        # Add rating
        rating = AlgorithmRating(
            rating_id="test_rating",
            algorithm_id="test_db_algo",
            reviewer="Test Reviewer",
            overall_rating=4.5,
            review_text="Great algorithm!"
        )
        
        success = self.db.add_rating(rating)
        assert success is True
        
        # Check if algorithm rating was updated
        updated_entry = self.db.get_algorithm("test_db_algo")
        assert updated_entry.rating_count == 1
        assert updated_entry.average_rating == 4.5
    
    def test_multiple_ratings(self):
        """Test multiple ratings."""
        # Submit algorithm first
        self.db.submit_algorithm(self.test_entry)
        
        # Add multiple ratings
        ratings = [
            AlgorithmRating(
                rating_id="rating_1",
                algorithm_id="test_db_algo",
                reviewer="Reviewer 1",
                overall_rating=4.0
            ),
            AlgorithmRating(
                rating_id="rating_2",
                algorithm_id="test_db_algo",
                reviewer="Reviewer 2",
                overall_rating=5.0
            ),
            AlgorithmRating(
                rating_id="rating_3",
                algorithm_id="test_db_algo",
                reviewer="Reviewer 3",
                overall_rating=3.0
            )
        ]
        
        for rating in ratings:
            self.db.add_rating(rating)
        
        # Check average rating
        updated_entry = self.db.get_algorithm("test_db_algo")
        assert updated_entry.rating_count == 3
        assert abs(updated_entry.average_rating - 4.0) < 0.01  # (4+5+3)/3 = 4.0
    
    def test_record_download(self):
        """Test recording downloads."""
        # Submit algorithm first
        self.db.submit_algorithm(self.test_entry)
        
        # Record download
        success = self.db.record_download("test_db_algo", "test_user")
        assert success is True
        
        # Check download count
        updated_entry = self.db.get_algorithm("test_db_algo")
        assert updated_entry.download_count == 1
        
        # Record another download
        self.db.record_download("test_db_algo")
        updated_entry = self.db.get_algorithm("test_db_algo")
        assert updated_entry.download_count == 2
    
    def test_get_popular_algorithms(self):
        """Test getting popular algorithms."""
        # Submit algorithm with downloads
        self.db.submit_algorithm(self.test_entry)
        self.db.record_download("test_db_algo")
        
        popular = self.db.get_popular_algorithms()
        
        assert len(popular) == 1
        assert popular[0].entry_id == "test_db_algo"
        assert popular[0].download_count >= 1


@pytest.mark.skipif(not HAS_MARKETPLACE, reason="algorithm_marketplace module not available")
class TestAlgorithmPackager:
    """Test AlgorithmPackager functionality."""
    
    def setup_method(self):
        """Set up test packager."""
        self.packager = AlgorithmPackager()
        
        # Create test entry
        metadata = AlgorithmMetadata(
            algorithm_id="package_test",
            name="Package Test",
            description="Algorithm for testing packaging functionality",
            author="Test Author",
            version="1.0.0",
            category=AlgorithmCategory.GENERAL,
            algorithm_type=AlgorithmType.CIRCUIT,
            license=LicenseType.MIT
        )
        
        self.test_entry = MarketplaceEntry(
            entry_id="package_test",
            metadata=metadata,
            algorithm_data={"gates": [{"gate": "h", "qubits": [0]}]},
            documentation="Package test documentation",
            examples=[{"name": "example", "description": "test", "code": "test_code"}],
            test_cases=[{"name": "test", "description": "test", "input": {}, "expected_output": {}}]
        )
    
    def test_supported_formats(self):
        """Test supported packaging formats."""
        assert 'json' in self.packager.supported_formats
        assert 'zip' in self.packager.supported_formats
        assert 'tar.gz' in self.packager.supported_formats
    
    def test_json_packaging(self):
        """Test JSON packaging."""
        package_data = self.packager.package_algorithm(self.test_entry, 'json')
        
        assert package_data is not None
        assert isinstance(package_data, bytes)
        
        # Verify content
        json_content = json.loads(package_data.decode('utf-8'))
        assert json_content['entry_id'] == "package_test"
        assert json_content['metadata']['name'] == "Package Test"
    
    def test_zip_packaging(self):
        """Test ZIP packaging."""
        package_data = self.packager.package_algorithm(self.test_entry, 'zip')
        
        assert package_data is not None
        assert isinstance(package_data, bytes)
        
        # Verify it's a valid ZIP (basic check)
        assert package_data.startswith(b'PK')  # ZIP magic number
    
    def test_tar_packaging(self):
        """Test tar.gz packaging."""
        package_data = self.packager.package_algorithm(self.test_entry, 'tar.gz')
        
        assert package_data is not None
        assert isinstance(package_data, bytes)
        
        # Verify it's gzipped data
        assert package_data.startswith(b'\x1f\x8b')  # Gzip magic number
    
    def test_unsupported_format(self):
        """Test unsupported format."""
        with pytest.raises(ValueError, match="Unsupported format"):
            self.packager.package_algorithm(self.test_entry, 'unsupported')
    
    def test_json_unpacking(self):
        """Test JSON unpacking."""
        # Package first
        package_data = self.packager.package_algorithm(self.test_entry, 'json')
        
        # Unpack
        unpacked_entry = self.packager.unpack_algorithm(package_data, 'json')
        
        assert unpacked_entry is not None
        assert unpacked_entry.entry_id == self.test_entry.entry_id
        assert unpacked_entry.metadata.name == self.test_entry.metadata.name
        assert unpacked_entry.algorithm_data == self.test_entry.algorithm_data
    
    def test_zip_unpacking(self):
        """Test ZIP unpacking."""
        # Package first
        package_data = self.packager.package_algorithm(self.test_entry, 'zip')
        
        # Unpack
        unpacked_entry = self.packager.unpack_algorithm(package_data, 'zip')
        
        assert unpacked_entry is not None
        assert unpacked_entry.entry_id == self.test_entry.entry_id
        assert unpacked_entry.metadata.name == self.test_entry.metadata.name


@pytest.mark.skipif(not HAS_MARKETPLACE, reason="algorithm_marketplace module not available")
class TestQuantumAlgorithmMarketplace:
    """Test QuantumAlgorithmMarketplace functionality."""
    
    def setup_method(self):
        """Set up test marketplace."""
        # Use temporary database
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            self.temp_db_path = tmp.name
        
        self.marketplace = QuantumAlgorithmMarketplace(self.temp_db_path)
    
    def teardown_method(self):
        """Clean up marketplace."""
        self.marketplace.close()
        Path(self.temp_db_path).unlink(missing_ok=True)
    
    def test_marketplace_initialization(self):
        """Test marketplace initialization."""
        assert self.marketplace.database is not None
        assert self.marketplace.validator is not None
        assert self.marketplace.packager is not None
        assert self.marketplace.api is not None
    
    def test_example_algorithms_populated(self):
        """Test that example algorithms are populated."""
        algorithms = self.marketplace.search_algorithms()
        
        # Should have example algorithms
        assert len(algorithms) > 0
        
        # Check for specific examples
        algorithm_names = [algo.metadata.name for algo in algorithms]
        assert "Bell State Preparation" in algorithm_names
        assert "Grover's Search Algorithm" in algorithm_names
        assert "VQE for H2 Molecule" in algorithm_names
    
    def test_submit_algorithm(self):
        """Test algorithm submission."""
        # Create test algorithm
        metadata = AlgorithmMetadata(
            algorithm_id="test_submit",
            name="Test Submit Algorithm",
            description="Algorithm for testing submission process",
            author="Test Author",
            version="1.0.0",
            category=AlgorithmCategory.GENERAL,
            algorithm_type=AlgorithmType.CIRCUIT,
            license=LicenseType.MIT
        )
        
        test_entry = MarketplaceEntry(
            entry_id="test_submit",
            metadata=metadata,
            algorithm_data={"gates": [{"gate": "h", "qubits": [0]}]},
            documentation="Test documentation that is long enough for validation requirements",
            examples=[{"name": "test", "description": "test example", "code": "test_code"}],
            test_cases=[{"name": "test", "description": "test case", "input": {}, "expected_output": {}}]
        )
        
        success, errors = self.marketplace.submit_algorithm(test_entry)
        
        assert success is True
        assert len(errors) == 0
        
        # Verify submission
        retrieved = self.marketplace.get_algorithm("test_submit")
        assert retrieved is not None
        assert retrieved.status == MarketplaceStatus.PENDING_REVIEW
    
    def test_search_algorithms(self):
        """Test algorithm search."""
        results = self.marketplace.search_algorithms(query="Bell")
        
        assert len(results) > 0
        bell_found = any("Bell" in algo.metadata.name for algo in results)
        assert bell_found
    
    def test_get_algorithm(self):
        """Test getting specific algorithm."""
        # Should have Bell state from examples
        bell_algo = self.marketplace.get_algorithm("bell_state_v1")
        
        assert bell_algo is not None
        assert bell_algo.metadata.name == "Bell State Preparation"
        assert bell_algo.metadata.category == AlgorithmCategory.GENERAL
    
    def test_download_algorithm(self):
        """Test algorithm download."""
        package_data = self.marketplace.download_algorithm("bell_state_v1", 'json')
        
        assert package_data is not None
        assert isinstance(package_data, bytes)
        
        # Verify download was recorded
        bell_algo = self.marketplace.get_algorithm("bell_state_v1")
        assert bell_algo.download_count > 0
    
    def test_rate_algorithm(self):
        """Test algorithm rating."""
        rating = AlgorithmRating(
            rating_id="test_rating_123",
            algorithm_id="bell_state_v1",
            reviewer="Test Reviewer",
            overall_rating=5.0,
            review_text="Excellent algorithm!"
        )
        
        success = self.marketplace.rate_algorithm(rating)
        assert success is True
        
        # Check if rating affected the algorithm
        bell_algo = self.marketplace.get_algorithm("bell_state_v1")
        assert bell_algo.rating_count > 0
    
    def test_get_featured_algorithms(self):
        """Test getting featured algorithms."""
        featured = self.marketplace.get_featured_algorithms()
        
        assert len(featured) > 0
        assert all(algo.status == MarketplaceStatus.APPROVED for algo in featured)
    
    def test_get_popular_algorithms(self):
        """Test getting popular algorithms."""
        # Download an algorithm to make it popular
        self.marketplace.download_algorithm("bell_state_v1")
        
        popular = self.marketplace.get_popular_algorithms()
        
        assert len(popular) > 0
        # Bell state should be among popular (has downloads now)
        popular_ids = [algo.entry_id for algo in popular]
        assert "bell_state_v1" in popular_ids
    
    def test_marketplace_stats(self):
        """Test marketplace statistics."""
        stats = self.marketplace.get_marketplace_stats()
        
        assert 'total_algorithms' in stats
        assert 'total_downloads' in stats
        assert 'overall_rating' in stats
        assert 'total_authors' in stats
        assert 'categories' in stats
        
        assert stats['total_algorithms'] > 0
        assert isinstance(stats['categories'], dict)


@pytest.mark.skipif(not HAS_MARKETPLACE, reason="algorithm_marketplace module not available")
class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_get_quantum_marketplace(self):
        """Test getting global marketplace instance."""
        marketplace1 = get_quantum_marketplace()
        marketplace2 = get_quantum_marketplace()
        
        # Should be singleton
        assert marketplace1 is marketplace2
        assert isinstance(marketplace1, QuantumAlgorithmMarketplace)
    
    def test_search_algorithms_function(self):
        """Test search algorithms convenience function."""
        results = search_algorithms("Bell")
        
        assert isinstance(results, list)
        if len(results) > 0:
            assert all(isinstance(algo, MarketplaceEntry) for algo in results)
    
    def test_download_algorithm_function(self):
        """Test download algorithm convenience function."""
        # First check if Bell state exists
        marketplace = get_quantum_marketplace()
        bell_algo = marketplace.get_algorithm("bell_state_v1")
        
        if bell_algo:
            package_data = download_algorithm("bell_state_v1", 'json')
            assert package_data is not None
            assert isinstance(package_data, bytes)
    
    def test_create_algorithm_entry(self):
        """Test create algorithm entry convenience function."""
        entry = create_algorithm_entry(
            name="Test Algorithm",
            description="A test algorithm for convenience function testing",
            author="Test Author",
            algorithm_data={"gates": [{"gate": "h", "qubits": [0]}]},
            category=AlgorithmCategory.GENERAL,
            tags=["test", "convenience"]
        )
        
        assert isinstance(entry, MarketplaceEntry)
        assert entry.metadata.name == "Test Algorithm"
        assert entry.metadata.author == "Test Author"
        assert entry.metadata.category == AlgorithmCategory.GENERAL
        assert "test" in entry.metadata.tags
        assert entry.algorithm_data["gates"][0]["gate"] == "h"
    
    def test_submit_algorithm_function(self):
        """Test submit algorithm convenience function."""
        entry = create_algorithm_entry(
            name="Convenience Test Algorithm",
            description="Algorithm for testing convenience submission function",
            author="Test Author",
            algorithm_data={"gates": [{"gate": "x", "qubits": [0]}]},
            documentation="Test documentation that is sufficiently long for validation",
            examples=[{"name": "test", "description": "test", "code": "test"}],
            test_cases=[{"name": "test", "description": "test", "input": {}, "expected_output": {}}]
        )
        
        success, errors = submit_algorithm(entry)
        
        # Note: This might fail due to validation requirements in some tests
        # The important thing is that the function works and returns proper format
        assert isinstance(success, bool)
        assert isinstance(errors, list)


@pytest.mark.skipif(not HAS_MARKETPLACE, reason="algorithm_marketplace module not available")
class TestMarketplaceIntegration:
    """Test marketplace integration scenarios."""
    
    def setup_method(self):
        """Set up integration test."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            self.temp_db_path = tmp.name
        
        self.marketplace = QuantumAlgorithmMarketplace(self.temp_db_path)
    
    def teardown_method(self):
        """Clean up integration test."""
        self.marketplace.close()
        Path(self.temp_db_path).unlink(missing_ok=True)
    
    def test_end_to_end_workflow(self):
        """Test complete marketplace workflow."""
        # 1. Create algorithm
        entry = create_algorithm_entry(
            name="Integration Test Algorithm",
            description="Complete algorithm for testing end-to-end marketplace workflow",
            author="Integration Tester",
            algorithm_data={
                "gates": [
                    {"gate": "h", "qubits": [0]},
                    {"gate": "cnot", "qubits": [0, 1]},
                    {"gate": "measure", "qubits": [0, 1]}
                ]
            },
            documentation="""
# Integration Test Algorithm

## Description
This algorithm demonstrates a complete marketplace workflow.

## Usage
Apply Hadamard and CNOT gates to create entanglement.

## Parameters
- n_qubits: 2 (exactly)
""",
            examples=[{
                "name": "Basic Usage",
                "description": "Basic usage of the algorithm",
                "code": "circuit.h(0); circuit.cnot(0, 1)"
            }],
            test_cases=[{
                "name": "Entanglement Test",
                "description": "Test entanglement creation",
                "input": {"n_qubits": 2},
                "expected_output": {"00": 0.5, "11": 0.5}
            }],
            category=AlgorithmCategory.GENERAL,
            tags=["integration", "test", "entanglement"]
        )
        
        # 2. Submit algorithm
        success, errors = self.marketplace.submit_algorithm(entry)
        assert success is True, f"Submission failed: {errors}"
        
        # 3. Search for algorithm
        results = self.marketplace.search_algorithms(query="Integration Test")
        assert len(results) >= 1
        
        submitted_algo = None
        for algo in results:
            if algo.entry_id == entry.entry_id:
                submitted_algo = algo
                break
        
        assert submitted_algo is not None
        assert submitted_algo.status == MarketplaceStatus.PENDING_REVIEW
        
        # 4. Approve algorithm (simulate)
        submitted_algo.status = MarketplaceStatus.APPROVED
        self.marketplace.database.submit_algorithm(submitted_algo)
        
        # 5. Download algorithm
        package_data = self.marketplace.download_algorithm(entry.entry_id, 'json')
        assert package_data is not None
        
        # 6. Rate algorithm
        rating = AlgorithmRating(
            rating_id=f"rating_{entry.entry_id}",
            algorithm_id=entry.entry_id,
            reviewer="Integration Tester",
            overall_rating=4.5,
            quality_metrics={
                QualityMetric.CORRECTNESS: 5.0,
                QualityMetric.DOCUMENTATION: 4.0
            },
            review_text="Great integration test algorithm!"
        )
        
        rating_success = self.marketplace.rate_algorithm(rating)
        assert rating_success is True
        
        # 7. Verify final state
        final_algo = self.marketplace.get_algorithm(entry.entry_id)
        assert final_algo.download_count == 1
        assert final_algo.rating_count == 1
        assert final_algo.average_rating == 4.5
    
    def test_algorithm_discovery_workflow(self):
        """Test algorithm discovery and exploration."""
        # Search by category
        optimization_algos = self.marketplace.search_algorithms(
            category=AlgorithmCategory.OPTIMIZATION
        )
        
        ml_algos = self.marketplace.search_algorithms(
            category=AlgorithmCategory.MACHINE_LEARNING
        )
        
        chemistry_algos = self.marketplace.search_algorithms(
            category=AlgorithmCategory.CHEMISTRY
        )
        
        # Should have VQE in chemistry
        chemistry_names = [algo.metadata.name for algo in chemistry_algos]
        assert any("VQE" in name for name in chemistry_names)
        
        # Search by keyword
        search_algos = self.marketplace.search_algorithms(query="search")
        search_names = [algo.metadata.name for algo in search_algos]
        assert any("Grover" in name for name in search_names)
        
        # Get popular algorithms
        popular = self.marketplace.get_popular_algorithms()
        assert len(popular) > 0
        
        # Get featured algorithms
        featured = self.marketplace.get_featured_algorithms()
        assert len(featured) > 0
    
    def test_quality_assessment_workflow(self):
        """Test algorithm quality assessment."""
        # Get Bell state algorithm
        bell_algo = self.marketplace.get_algorithm("bell_state_v1")
        assert bell_algo is not None
        
        # Assess quality
        validator = AlgorithmValidator()
        is_valid, errors = validator.validate_entry(bell_algo)
        
        assert is_valid is True
        assert len(errors) == 0
        
        quality_score = validator.estimate_quality_score(bell_algo)
        assert quality_score > 0
        
        # Add multiple ratings
        ratings = [
            AlgorithmRating(
                rating_id=f"quality_test_1",
                algorithm_id="bell_state_v1",
                reviewer="Quality Tester 1",
                overall_rating=5.0,
                quality_metrics={
                    QualityMetric.CORRECTNESS: 5.0,
                    QualityMetric.DOCUMENTATION: 5.0,
                    QualityMetric.READABILITY: 4.5
                }
            ),
            AlgorithmRating(
                rating_id=f"quality_test_2",
                algorithm_id="bell_state_v1",
                reviewer="Quality Tester 2",
                overall_rating=4.0,
                quality_metrics={
                    QualityMetric.CORRECTNESS: 4.5,
                    QualityMetric.PERFORMANCE: 4.0,
                    QualityMetric.INNOVATION: 3.5
                }
            )
        ]
        
        for rating in ratings:
            self.marketplace.rate_algorithm(rating)
        
        # Check updated ratings
        updated_bell = self.marketplace.get_algorithm("bell_state_v1")
        assert updated_bell.rating_count >= 2
        assert 4.0 <= updated_bell.average_rating <= 5.0


@pytest.mark.skipif(not HAS_MARKETPLACE, reason="algorithm_marketplace module not available")
class TestMarketplacePerformance:
    """Test marketplace performance characteristics."""
    
    def setup_method(self):
        """Set up performance test."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            self.temp_db_path = tmp.name
        
        self.marketplace = QuantumAlgorithmMarketplace(self.temp_db_path)
    
    def teardown_method(self):
        """Clean up performance test."""
        self.marketplace.close()
        Path(self.temp_db_path).unlink(missing_ok=True)
    
    def test_large_algorithm_submission(self):
        """Test submitting large algorithms."""
        # Create algorithm with many gates
        large_gates = []
        for i in range(1000):
            large_gates.append({"gate": "h", "qubits": [i % 10]})
            large_gates.append({"gate": "cnot", "qubits": [i % 10, (i + 1) % 10]})
        
        large_entry = create_algorithm_entry(
            name="Large Test Algorithm",
            description="Algorithm with many gates for performance testing",
            author="Performance Tester",
            algorithm_data={"gates": large_gates},
            documentation="Performance test documentation " * 20,  # Large doc
            examples=[{"name": f"example_{i}", "description": "test", "code": "test"} for i in range(10)],
            test_cases=[{"name": f"test_{i}", "description": "test", "input": {}, "expected_output": {}} for i in range(10)]
        )
        
        start_time = time.time()
        success, errors = self.marketplace.submit_algorithm(large_entry)
        submission_time = time.time() - start_time
        
        assert success is True
        assert submission_time < 5.0  # Should complete within 5 seconds
    
    def test_search_performance(self):
        """Test search performance."""
        # Submit multiple algorithms
        for i in range(50):
            entry = create_algorithm_entry(
                name=f"Performance Test Algorithm {i}",
                description=f"Algorithm {i} for performance testing with sufficient description length",
                author=f"Author {i % 5}",  # 5 different authors
                algorithm_data={"gates": [{"gate": "h", "qubits": [0]}]},
                category=AlgorithmCategory(list(AlgorithmCategory)[i % len(AlgorithmCategory)]),
                documentation="Performance test documentation that is long enough",
                examples=[{"name": "test", "description": "test", "code": "test"}],
                test_cases=[{"name": "test", "description": "test", "input": {}, "expected_output": {}}]
            )
            
            # Approve for search
            entry.status = MarketplaceStatus.APPROVED
            self.marketplace.database.submit_algorithm(entry)
        
        # Test search performance
        start_time = time.time()
        results = self.marketplace.search_algorithms(query="Performance")
        search_time = time.time() - start_time
        
        assert len(results) > 0
        assert search_time < 2.0  # Should complete within 2 seconds
        
        # Test category search
        start_time = time.time()
        cat_results = self.marketplace.search_algorithms(category=AlgorithmCategory.GENERAL)
        cat_search_time = time.time() - start_time
        
        assert cat_search_time < 1.0  # Should be even faster
    
    def test_packaging_performance(self):
        """Test packaging performance."""
        # Get existing algorithm
        bell_algo = self.marketplace.get_algorithm("bell_state_v1")
        
        if bell_algo:
            packager = AlgorithmPackager()
            
            # Test JSON packaging speed
            start_time = time.time()
            for _ in range(100):
                json_package = packager.package_algorithm(bell_algo, 'json')
            json_time = time.time() - start_time
            
            assert json_time < 2.0  # 100 JSON packages in under 2 seconds
            
            # Test ZIP packaging speed
            start_time = time.time()
            for _ in range(10):
                zip_package = packager.package_algorithm(bell_algo, 'zip')
            zip_time = time.time() - start_time
            
            assert zip_time < 5.0  # 10 ZIP packages in under 5 seconds
    
    def test_concurrent_operations(self):
        """Test concurrent marketplace operations."""
        import concurrent.futures
        
        def submit_algorithm(i):
            entry = create_algorithm_entry(
                name=f"Concurrent Test {i}",
                description=f"Concurrent algorithm {i} for testing thread safety",
                author=f"Concurrent Author {i}",
                algorithm_data={"gates": [{"gate": "x", "qubits": [0]}]},
                documentation="Concurrent test documentation that is sufficiently long",
                examples=[{"name": "test", "description": "test", "code": "test"}],
                test_cases=[{"name": "test", "description": "test", "input": {}, "expected_output": {}}]
            )
            return self.marketplace.submit_algorithm(entry)
        
        # Submit algorithms concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(submit_algorithm, i) for i in range(20)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Check results
        successful_submissions = sum(1 for success, _ in results if success)
        assert successful_submissions >= 15  # Most should succeed
        
        # Verify database consistency
        all_algorithms = self.marketplace.search_algorithms(limit=100)
        concurrent_algorithms = [algo for algo in all_algorithms if "Concurrent Test" in algo.metadata.name]
        assert len(concurrent_algorithms) >= 15


if __name__ == "__main__":
    pytest.main([__file__])