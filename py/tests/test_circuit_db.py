#!/usr/bin/env python3
"""
Test suite for quantum circuit database functionality.
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime, timezone

try:
    from quantrs2.circuit_db import (
        CircuitMetadata, CircuitEntry, CircuitDatabase, CircuitTemplates,
        create_circuit_database, populate_template_circuits
    )
    HAS_CIRCUIT_DB = True
except ImportError:
    HAS_CIRCUIT_DB = False


# Mock circuit class for testing
class MockCircuit:
    """Mock quantum circuit for testing."""
    
    def __init__(self, n_qubits=2, gates=None):
        self.n_qubits = n_qubits
        self.gates = gates or []
        self._gate_count = len(self.gates)
        self._depth = len(self.gates)  # Simplified depth calculation
    
    def gate_count(self):
        return self._gate_count
    
    def depth(self):
        return self._depth
    
    def to_qasm(self):
        return f"OPENQASM 2.0;\nqreg q[{self.n_qubits}];\n"
    
    def __str__(self):
        return f"MockCircuit({self.n_qubits} qubits, {len(self.gates)} gates)"


@pytest.mark.skipif(not HAS_CIRCUIT_DB, reason="circuit_db module not available")
class TestCircuitMetadata:
    """Test CircuitMetadata functionality."""
    
    def test_metadata_creation(self):
        """Test creating circuit metadata."""
        metadata = CircuitMetadata(
            name="Test Circuit",
            description="A test quantum circuit",
            author="Test Author",
            tags=["test", "quantum"],
            n_qubits=2,
            category="test"
        )
        
        assert metadata.name == "Test Circuit"
        assert metadata.description == "A test quantum circuit"
        assert metadata.author == "Test Author"
        assert metadata.tags == ["test", "quantum"]
        assert metadata.n_qubits == 2
        assert metadata.category == "test"
        assert metadata.version == "1.0.0"  # Default
        assert isinstance(metadata.created_at, datetime)
    
    def test_metadata_with_custom_fields(self):
        """Test metadata with custom fields."""
        metadata = CircuitMetadata(
            name="Custom Circuit",
            algorithm_type="VQE",
            complexity="high",
            paper_reference="arXiv:1234.5678"
        )
        
        assert metadata.custom_fields["algorithm_type"] == "VQE"
        assert metadata.custom_fields["complexity"] == "high"
        assert metadata.custom_fields["paper_reference"] == "arXiv:1234.5678"
    
    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        created_at = datetime.now(timezone.utc)
        metadata = CircuitMetadata(
            name="Dict Test",
            description="Test description",
            author="Test Author",
            tags=["tag1", "tag2"],
            created_at=created_at,
            custom_field="custom_value"
        )
        
        data = metadata.to_dict()
        
        assert data["name"] == "Dict Test"
        assert data["description"] == "Test description"
        assert data["author"] == "Test Author"
        assert data["tags"] == ["tag1", "tag2"]
        assert data["created_at"] == created_at.isoformat()
        assert data["custom_field"] == "custom_value"
    
    def test_metadata_from_dict(self):
        """Test creating metadata from dictionary."""
        created_at = datetime.now(timezone.utc)
        data = {
            "name": "From Dict",
            "description": "Test description",
            "author": "Test Author",
            "tags": ["tag1", "tag2"],
            "n_qubits": 3,
            "created_at": created_at.isoformat(),
            "custom_field": "custom_value"
        }
        
        metadata = CircuitMetadata.from_dict(data)
        
        assert metadata.name == "From Dict"
        assert metadata.description == "Test description"
        assert metadata.author == "Test Author"
        assert metadata.tags == ["tag1", "tag2"]
        assert metadata.n_qubits == 3
        assert metadata.created_at == created_at
        assert metadata.custom_fields["custom_field"] == "custom_value"
    
    def test_metadata_roundtrip(self):
        """Test metadata dictionary roundtrip conversion."""
        original = CircuitMetadata(
            name="Roundtrip Test",
            description="Test roundtrip conversion",
            tags=["test"],
            n_qubits=4,
            gate_count=10,
            depth=5,
            custom_field="value"
        )
        
        data = original.to_dict()
        restored = CircuitMetadata.from_dict(data)
        
        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.tags == original.tags
        assert restored.n_qubits == original.n_qubits
        assert restored.gate_count == original.gate_count
        assert restored.depth == original.depth
        assert restored.custom_fields == original.custom_fields


@pytest.mark.skipif(not HAS_CIRCUIT_DB, reason="circuit_db module not available")
class TestCircuitEntry:
    """Test CircuitEntry functionality."""
    
    def test_entry_creation(self):
        """Test creating circuit entry."""
        circuit = MockCircuit(n_qubits=2)
        metadata = CircuitMetadata(name="Test Entry", n_qubits=2)
        
        entry = CircuitEntry("test_id", circuit, metadata)
        
        assert entry.circuit_id == "test_id"
        assert entry.circuit == circuit
        assert entry.metadata == metadata
        assert isinstance(entry.hash, str)
        assert len(entry.hash) == 64  # SHA256 hash length
    
    def test_entry_hash_consistency(self):
        """Test that entry hash is consistent."""
        circuit = MockCircuit(n_qubits=2)
        metadata = CircuitMetadata(name="Hash Test", n_qubits=2)
        
        entry1 = CircuitEntry("test_id", circuit, metadata)
        entry2 = CircuitEntry("test_id", circuit, metadata)
        
        assert entry1.hash == entry2.hash
    
    def test_entry_to_dict(self):
        """Test converting entry to dictionary."""
        circuit = MockCircuit(n_qubits=2)
        metadata = CircuitMetadata(name="Dict Entry", n_qubits=2)
        
        entry = CircuitEntry("dict_test", circuit, metadata)
        data = entry.to_dict()
        
        assert data["circuit_id"] == "dict_test"
        assert data["hash"] == entry.hash
        assert "metadata" in data
        assert data["metadata"]["name"] == "Dict Entry"


@pytest.mark.skipif(not HAS_CIRCUIT_DB, reason="circuit_db module not available")
class TestCircuitDatabase:
    """Test CircuitDatabase functionality."""
    
    def setup_method(self):
        """Set up test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_circuits.db"
        self.db = CircuitDatabase(self.db_path)
    
    def teardown_method(self):
        """Clean up test database."""
        shutil.rmtree(self.temp_dir)
    
    def test_database_creation(self):
        """Test database creation and initialization."""
        assert self.db_path.exists()
        assert self.db.db_path == self.db_path
    
    def test_add_circuit(self):
        """Test adding a circuit to the database."""
        circuit = MockCircuit(n_qubits=2)
        metadata = CircuitMetadata(
            name="Test Circuit",
            description="A test circuit",
            author="Test Author",
            tags=["test"]
        )
        
        circuit_id = self.db.add_circuit(circuit, metadata)
        
        assert isinstance(circuit_id, str)
        assert self.db.exists(circuit_id)
    
    def test_add_circuit_with_custom_id(self):
        """Test adding circuit with custom ID."""
        circuit = MockCircuit(n_qubits=3)
        metadata = CircuitMetadata(name="Custom ID Circuit")
        
        circuit_id = self.db.add_circuit(circuit, metadata, circuit_id="custom_test_id")
        
        assert circuit_id == "custom_test_id"
        assert self.db.exists("custom_test_id")
    
    def test_add_duplicate_circuit(self):
        """Test adding duplicate circuit without overwrite."""
        circuit = MockCircuit(n_qubits=2)
        metadata = CircuitMetadata(name="Duplicate Test")
        
        circuit_id = self.db.add_circuit(circuit, metadata, circuit_id="duplicate_id")
        
        # Should raise error without overwrite=True
        with pytest.raises(ValueError):
            self.db.add_circuit(circuit, metadata, circuit_id="duplicate_id")
        
        # Should work with overwrite=True
        new_id = self.db.add_circuit(circuit, metadata, circuit_id="duplicate_id", overwrite=True)
        assert new_id == "duplicate_id"
    
    def test_get_circuit(self):
        """Test retrieving a circuit."""
        circuit = MockCircuit(n_qubits=3, gates=[("h", [0]), ("cnot", [0, 1])])
        metadata = CircuitMetadata(
            name="Retrieve Test",
            description="Test retrieval",
            tags=["retrieve", "test"],
            n_qubits=3
        )
        
        circuit_id = self.db.add_circuit(circuit, metadata)
        retrieved = self.db.get_circuit(circuit_id)
        
        assert retrieved is not None
        assert retrieved.circuit_id == circuit_id
        assert retrieved.metadata.name == "Retrieve Test"
        assert retrieved.metadata.tags == ["retrieve", "test"]
        assert retrieved.metadata.n_qubits == 3
    
    def test_get_nonexistent_circuit(self):
        """Test retrieving nonexistent circuit."""
        result = self.db.get_circuit("nonexistent_id")
        assert result is None
    
    def test_delete_circuit(self):
        """Test deleting a circuit."""
        circuit = MockCircuit(n_qubits=2)
        metadata = CircuitMetadata(name="Delete Test")
        
        circuit_id = self.db.add_circuit(circuit, metadata)
        assert self.db.exists(circuit_id)
        
        deleted = self.db.delete_circuit(circuit_id)
        assert deleted is True
        assert not self.db.exists(circuit_id)
    
    def test_delete_nonexistent_circuit(self):
        """Test deleting nonexistent circuit."""
        deleted = self.db.delete_circuit("nonexistent_id")
        assert deleted is False
    
    def test_search_by_name(self):
        """Test searching circuits by name."""
        circuits = [
            (MockCircuit(2), CircuitMetadata(name="Bell State", category="entanglement")),
            (MockCircuit(3), CircuitMetadata(name="GHZ State", category="entanglement")),
            (MockCircuit(2), CircuitMetadata(name="Random Circuit", category="test"))
        ]
        
        for circuit, metadata in circuits:
            self.db.add_circuit(circuit, metadata)
        
        # Search with exact match
        results = self.db.search_circuits(name_pattern="Bell State")
        assert len(results) == 1
        assert results[0].metadata.name == "Bell State"
        
        # Search with pattern
        results = self.db.search_circuits(name_pattern="%State")
        assert len(results) == 2
        
        # Search with no matches
        results = self.db.search_circuits(name_pattern="Nonexistent")
        assert len(results) == 0
    
    def test_search_by_category(self):
        """Test searching circuits by category."""
        circuits = [
            (MockCircuit(2), CircuitMetadata(name="Circuit 1", category="entanglement")),
            (MockCircuit(3), CircuitMetadata(name="Circuit 2", category="entanglement")),
            (MockCircuit(2), CircuitMetadata(name="Circuit 3", category="algorithm"))
        ]
        
        for circuit, metadata in circuits:
            self.db.add_circuit(circuit, metadata)
        
        results = self.db.search_circuits(category="entanglement")
        assert len(results) == 2
        
        results = self.db.search_circuits(category="algorithm")
        assert len(results) == 1
    
    def test_search_by_tags(self):
        """Test searching circuits by tags."""
        circuits = [
            (MockCircuit(2), CircuitMetadata(name="Circuit 1", tags=["bell", "entanglement"])),
            (MockCircuit(3), CircuitMetadata(name="Circuit 2", tags=["ghz", "entanglement"])),
            (MockCircuit(2), CircuitMetadata(name="Circuit 3", tags=["test", "random"]))
        ]
        
        for circuit, metadata in circuits:
            self.db.add_circuit(circuit, metadata)
        
        # Search for single tag
        results = self.db.search_circuits(tags=["entanglement"])
        assert len(results) == 2
        
        # Search for multiple tags (AND operation)
        results = self.db.search_circuits(tags=["bell", "entanglement"])
        assert len(results) == 1
        assert results[0].metadata.name == "Circuit 1"
        
        # Search for non-matching combination
        results = self.db.search_circuits(tags=["bell", "ghz"])
        assert len(results) == 0
    
    def test_search_by_qubits(self):
        """Test searching circuits by number of qubits."""
        circuits = [
            (MockCircuit(2), CircuitMetadata(name="2-qubit", n_qubits=2)),
            (MockCircuit(3), CircuitMetadata(name="3-qubit", n_qubits=3)),
            (MockCircuit(4), CircuitMetadata(name="4-qubit", n_qubits=4)),
            (MockCircuit(5), CircuitMetadata(name="5-qubit", n_qubits=5))
        ]
        
        for circuit, metadata in circuits:
            self.db.add_circuit(circuit, metadata)
        
        # Search for exact qubit count
        results = self.db.search_circuits(n_qubits=3)
        assert len(results) == 1
        assert results[0].metadata.name == "3-qubit"
        
        # Search for qubit range
        results = self.db.search_circuits(n_qubits=(3, 4))
        assert len(results) == 2
        
        names = [r.metadata.name for r in results]
        assert "3-qubit" in names
        assert "4-qubit" in names
    
    def test_search_by_depth(self):
        """Test searching circuits by depth."""
        circuits = [
            (MockCircuit(2, [("h", [0])]), CircuitMetadata(name="Shallow", depth=1)),
            (MockCircuit(2, [("h", [0]), ("cnot", [0, 1])]), CircuitMetadata(name="Medium", depth=2)),
            (MockCircuit(2, [("h", [0]), ("cnot", [0, 1]), ("h", [1])]), CircuitMetadata(name="Deep", depth=3))
        ]
        
        for circuit, metadata in circuits:
            self.db.add_circuit(circuit, metadata)
        
        # Search by minimum depth
        results = self.db.search_circuits(min_depth=2)
        assert len(results) == 2
        
        # Search by maximum depth
        results = self.db.search_circuits(max_depth=2)
        assert len(results) == 2
        
        # Search by depth range
        results = self.db.search_circuits(min_depth=2, max_depth=2)
        assert len(results) == 1
        assert results[0].metadata.name == "Medium"
    
    def test_search_with_limit(self):
        """Test search result limiting."""
        for i in range(10):
            circuit = MockCircuit(2)
            metadata = CircuitMetadata(name=f"Circuit {i}", category="test")
            self.db.add_circuit(circuit, metadata)
        
        results = self.db.search_circuits(category="test", limit=5)
        assert len(results) == 5
    
    def test_list_circuits(self):
        """Test listing all circuits."""
        circuits = [
            (MockCircuit(2), CircuitMetadata(name="Circuit 1")),
            (MockCircuit(3), CircuitMetadata(name="Circuit 2")),
            (MockCircuit(4), CircuitMetadata(name="Circuit 3"))
        ]
        
        for circuit, metadata in circuits:
            self.db.add_circuit(circuit, metadata)
        
        all_circuits = self.db.list_circuits()
        assert len(all_circuits) == 3
        
        # Test with limit
        limited = self.db.list_circuits(limit=2)
        assert len(limited) == 2
        
        # Test with offset
        offset = self.db.list_circuits(limit=2, offset=1)
        assert len(offset) == 2
    
    def test_get_categories(self):
        """Test getting unique categories."""
        circuits = [
            (MockCircuit(2), CircuitMetadata(name="C1", category="entanglement")),
            (MockCircuit(3), CircuitMetadata(name="C2", category="algorithm")),
            (MockCircuit(2), CircuitMetadata(name="C3", category="entanglement"))
        ]
        
        for circuit, metadata in circuits:
            self.db.add_circuit(circuit, metadata)
        
        categories = self.db.get_categories()
        assert len(categories) == 2
        assert "entanglement" in categories
        assert "algorithm" in categories
    
    def test_get_tags(self):
        """Test getting unique tags."""
        circuits = [
            (MockCircuit(2), CircuitMetadata(name="C1", tags=["bell", "entanglement"])),
            (MockCircuit(3), CircuitMetadata(name="C2", tags=["ghz", "entanglement"])),
            (MockCircuit(2), CircuitMetadata(name="C3", tags=["bell", "test"]))
        ]
        
        for circuit, metadata in circuits:
            self.db.add_circuit(circuit, metadata)
        
        tags = self.db.get_tags()
        assert len(tags) == 4
        assert "bell" in tags
        assert "entanglement" in tags
        assert "ghz" in tags
        assert "test" in tags
    
    def test_get_authors(self):
        """Test getting unique authors."""
        circuits = [
            (MockCircuit(2), CircuitMetadata(name="C1", author="Alice")),
            (MockCircuit(3), CircuitMetadata(name="C2", author="Bob")),
            (MockCircuit(2), CircuitMetadata(name="C3", author="Alice"))
        ]
        
        for circuit, metadata in circuits:
            self.db.add_circuit(circuit, metadata)
        
        authors = self.db.get_authors()
        assert len(authors) == 2
        assert "Alice" in authors
        assert "Bob" in authors
    
    def test_get_statistics(self):
        """Test getting database statistics."""
        circuits = [
            (MockCircuit(2), CircuitMetadata(name="C1", author="Alice", category="test", n_qubits=2)),
            (MockCircuit(3), CircuitMetadata(name="C2", author="Bob", category="algorithm", n_qubits=3)),
            (MockCircuit(4), CircuitMetadata(name="C3", author="Alice", category="test", n_qubits=4))
        ]
        
        for circuit, metadata in circuits:
            self.db.add_circuit(circuit, metadata)
        
        stats = self.db.get_statistics()
        
        assert stats['total_circuits'] == 3
        assert stats['unique_authors'] == 2
        assert stats['unique_categories'] == 2
        assert stats['min_qubits'] == 2
        assert stats['max_qubits'] == 4
        assert stats['avg_qubits'] == 3.0
    
    def test_backup_and_restore(self):
        """Test database backup and restore."""
        # Add some circuits
        circuit = MockCircuit(2)
        metadata = CircuitMetadata(name="Backup Test")
        circuit_id = self.db.add_circuit(circuit, metadata)
        
        # Create backup
        backup_path = Path(self.temp_dir) / "backup.db"
        self.db.backup(backup_path)
        
        assert backup_path.exists()
        
        # Delete circuit from original
        self.db.delete_circuit(circuit_id)
        assert not self.db.exists(circuit_id)
        
        # Restore from backup
        self.db.restore(backup_path)
        assert self.db.exists(circuit_id)
    
    def test_export_import_json(self):
        """Test exporting and importing circuits in JSON format."""
        circuit = MockCircuit(2)
        metadata = CircuitMetadata(
            name="Export Test",
            description="Test export/import",
            tags=["export", "test"]
        )
        
        circuit_id = self.db.add_circuit(circuit, metadata)
        
        # Export to JSON
        export_path = Path(self.temp_dir) / "exported.json"
        self.db.export_circuit(circuit_id, export_path, format="json")
        
        assert export_path.exists()
        
        # Verify JSON content
        with open(export_path, 'r') as f:
            data = json.load(f)
        
        assert data['circuit_id'] == circuit_id
        assert data['metadata']['name'] == "Export Test"
        
        # Import circuit (with new name to avoid conflict)
        imported_id = self.db.import_circuit(export_path, format="json", name="Imported Test")
        
        imported_entry = self.db.get_circuit(imported_id)
        assert imported_entry is not None
        assert imported_entry.metadata.name == "Imported Test"
        assert imported_entry.metadata.description == "Test export/import"


@pytest.mark.skipif(not HAS_CIRCUIT_DB, reason="circuit_db module not available")
class TestCircuitTemplates:
    """Test CircuitTemplates functionality."""
    
    def test_bell_state_template(self):
        """Test Bell state template."""
        circuit, metadata = CircuitTemplates.bell_state()
        
        assert circuit is not None
        assert metadata.name == "Bell State"
        assert metadata.n_qubits == 2
        assert "bell" in metadata.tags
        assert "entanglement" in metadata.tags
    
    def test_ghz_state_template(self):
        """Test GHZ state template."""
        circuit, metadata = CircuitTemplates.ghz_state(4)
        
        assert circuit is not None
        assert metadata.n_qubits == 4
        assert "GHZ State (4 qubits)" in metadata.name
        assert "ghz" in metadata.tags
        assert "entanglement" in metadata.tags
    
    def test_qft_template(self):
        """Test Quantum Fourier Transform template."""
        circuit, metadata = CircuitTemplates.quantum_fourier_transform(3)
        
        assert circuit is not None
        assert metadata.n_qubits == 3
        assert "Quantum Fourier Transform (3 qubits)" in metadata.name
        assert "qft" in metadata.tags
        assert "algorithm" in metadata.tags


@pytest.mark.skipif(not HAS_CIRCUIT_DB, reason="circuit_db module not available")
class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_circuit_database(self):
        """Test creating database with convenience function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "convenience_test.db"
            db = create_circuit_database(db_path)
            
            assert isinstance(db, CircuitDatabase)
            assert db.db_path == db_path
    
    def test_populate_template_circuits(self):
        """Test populating database with templates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "templates_test.db"
            db = create_circuit_database(db_path)
            
            # Should start empty
            stats = db.get_statistics()
            assert stats['total_circuits'] == 0
            
            # Populate with templates
            populate_template_circuits(db)
            
            # Should now have template circuits
            stats = db.get_statistics()
            assert stats['total_circuits'] > 0
            
            # Check that we have expected categories
            categories = db.get_categories()
            assert "entanglement" in categories
            assert "algorithm" in categories


@pytest.mark.skipif(not HAS_CIRCUIT_DB, reason="circuit_db module not available")
class TestCircuitDatabaseIntegration:
    """Test circuit database integration scenarios."""
    
    def setup_method(self):
        """Set up test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "integration_test.db"
        self.db = CircuitDatabase(self.db_path)
    
    def teardown_method(self):
        """Clean up test database."""
        shutil.rmtree(self.temp_dir)
    
    def test_large_circuit_collection(self):
        """Test working with large collections of circuits."""
        # Add many circuits
        for i in range(100):
            circuit = MockCircuit(n_qubits=2 + (i % 5))
            metadata = CircuitMetadata(
                name=f"Circuit {i:03d}",
                author=f"Author {i % 10}",
                category=["entanglement", "algorithm", "test"][i % 3],
                tags=[f"tag{i % 5}", f"tag{i % 7}"],
                n_qubits=2 + (i % 5)
            )
            self.db.add_circuit(circuit, metadata)
        
        # Test statistics
        stats = self.db.get_statistics()
        assert stats['total_circuits'] == 100
        assert stats['unique_authors'] == 10
        assert stats['unique_categories'] == 3
        
        # Test search performance
        results = self.db.search_circuits(category="algorithm")
        assert len(results) > 0
        
        # Test pagination
        page1 = self.db.list_circuits(limit=20, offset=0)
        page2 = self.db.list_circuits(limit=20, offset=20)
        
        assert len(page1) == 20
        assert len(page2) == 20
        
        # Ensure no overlap
        page1_ids = {c['circuit_id'] for c in page1}
        page2_ids = {c['circuit_id'] for c in page2}
        assert len(page1_ids.intersection(page2_ids)) == 0
    
    def test_complex_search_queries(self):
        """Test complex search queries combining multiple criteria."""
        # Add circuits with various properties
        test_data = [
            (MockCircuit(2), CircuitMetadata(name="Bell", author="Alice", category="entanglement", tags=["bell", "basic"], n_qubits=2, depth=2)),
            (MockCircuit(3), CircuitMetadata(name="GHZ", author="Alice", category="entanglement", tags=["ghz", "advanced"], n_qubits=3, depth=3)),
            (MockCircuit(4), CircuitMetadata(name="QFT", author="Bob", category="algorithm", tags=["qft", "advanced"], n_qubits=4, depth=8)),
            (MockCircuit(2), CircuitMetadata(name="Random", author="Charlie", category="test", tags=["random", "basic"], n_qubits=2, depth=10)),
        ]
        
        for circuit, metadata in test_data:
            self.db.add_circuit(circuit, metadata)
        
        # Search by author and category
        results = self.db.search_circuits(author="Alice", category="entanglement")
        assert len(results) == 2
        
        # Search by tags and qubit range
        results = self.db.search_circuits(tags=["advanced"], n_qubits=(3, 4))
        assert len(results) == 2
        
        # Search by multiple criteria
        results = self.db.search_circuits(
            author="Alice",
            category="entanglement", 
            tags=["basic"],
            min_depth=1,
            max_depth=5
        )
        assert len(results) == 1
        assert results[0].metadata.name == "Bell"
    
    def test_database_persistence(self):
        """Test database persistence across instances."""
        # Add circuit in first instance
        circuit = MockCircuit(2)
        metadata = CircuitMetadata(name="Persistent Test")
        circuit_id = self.db.add_circuit(circuit, metadata)
        
        # Close database and create new instance
        del self.db
        new_db = CircuitDatabase(self.db_path)
        
        # Should still exist
        assert new_db.exists(circuit_id)
        entry = new_db.get_circuit(circuit_id)
        assert entry is not None
        assert entry.metadata.name == "Persistent Test"
    
    def test_concurrent_access_safety(self):
        """Test basic concurrent access patterns."""
        # This is a simplified test - real concurrent testing would require threading
        
        # Add circuit
        circuit = MockCircuit(2)
        metadata = CircuitMetadata(name="Concurrent Test")
        circuit_id = self.db.add_circuit(circuit, metadata)
        
        # Create second database instance
        db2 = CircuitDatabase(self.db_path)
        
        # Both instances should see the circuit
        assert self.db.exists(circuit_id)
        assert db2.exists(circuit_id)
        
        # Add circuit through second instance
        circuit2 = MockCircuit(3)
        metadata2 = CircuitMetadata(name="Second Instance")
        circuit_id2 = db2.add_circuit(circuit2, metadata2)
        
        # First instance should see new circuit after refresh
        assert self.db.exists(circuit_id2)


@pytest.mark.skipif(not HAS_CIRCUIT_DB, reason="circuit_db module not available")
class TestCircuitDatabaseErrorHandling:
    """Test error handling in circuit database."""
    
    def setup_method(self):
        """Set up test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "error_test.db"
        self.db = CircuitDatabase(self.db_path)
    
    def teardown_method(self):
        """Clean up test database."""
        shutil.rmtree(self.temp_dir)
    
    def test_invalid_export_format(self):
        """Test handling of invalid export format."""
        circuit = MockCircuit(2)
        metadata = CircuitMetadata(name="Export Error Test")
        circuit_id = self.db.add_circuit(circuit, metadata)
        
        export_path = Path(self.temp_dir) / "invalid.xyz"
        
        with pytest.raises(ValueError):
            self.db.export_circuit(circuit_id, export_path, format="invalid")
    
    def test_export_nonexistent_circuit(self):
        """Test exporting nonexistent circuit."""
        export_path = Path(self.temp_dir) / "nonexistent.json"
        
        with pytest.raises(ValueError):
            self.db.export_circuit("nonexistent_id", export_path)
    
    def test_import_nonexistent_file(self):
        """Test importing from nonexistent file."""
        nonexistent_path = Path(self.temp_dir) / "nonexistent.json"
        
        with pytest.raises(FileNotFoundError):
            self.db.import_circuit(nonexistent_path)
    
    def test_invalid_import_format(self):
        """Test handling of invalid import format."""
        # Create a text file
        test_file = Path(self.temp_dir) / "invalid.txt"
        with open(test_file, 'w') as f:
            f.write("This is not a valid format")
        
        with pytest.raises(ValueError):
            self.db.import_circuit(test_file, format="invalid")
    
    def test_restore_nonexistent_backup(self):
        """Test restoring from nonexistent backup."""
        nonexistent_backup = Path(self.temp_dir) / "nonexistent_backup.db"
        
        with pytest.raises(FileNotFoundError):
            self.db.restore(nonexistent_backup)
    
    def test_invalid_search_parameters(self):
        """Test search with invalid parameters."""
        # These should not raise errors but return empty results
        results = self.db.search_circuits(n_qubits=(-1, -2))  # Invalid range
        assert len(results) == 0
        
        results = self.db.search_circuits(min_depth=-5)  # Negative depth
        assert len(results) == 0


@pytest.mark.skipif(not HAS_CIRCUIT_DB, reason="circuit_db module not available")
class TestCircuitDatabasePerformance:
    """Test performance characteristics of circuit database."""
    
    def setup_method(self):
        """Set up test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "performance_test.db"
        self.db = CircuitDatabase(self.db_path)
    
    def teardown_method(self):
        """Clean up test database."""
        shutil.rmtree(self.temp_dir)
    
    def test_bulk_insert_performance(self):
        """Test performance of bulk circuit insertion."""
        import time
        
        start_time = time.time()
        
        # Insert many circuits
        for i in range(100):
            circuit = MockCircuit(n_qubits=2 + (i % 3))
            metadata = CircuitMetadata(
                name=f"Perf Test {i}",
                category=["test", "performance"][i % 2],
                n_qubits=2 + (i % 3)
            )
            self.db.add_circuit(circuit, metadata)
        
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 10.0  # 10 seconds for 100 circuits
        
        # Verify all circuits were added
        stats = self.db.get_statistics()
        assert stats['total_circuits'] == 100
    
    def test_search_performance(self):
        """Test search performance with many circuits."""
        import time
        
        # Insert test data
        for i in range(200):
            circuit = MockCircuit(n_qubits=2 + (i % 5))
            metadata = CircuitMetadata(
                name=f"Search Test {i}",
                author=f"Author {i % 20}",
                category=["entanglement", "algorithm", "test"][i % 3],
                tags=[f"tag{i % 10}"],
                n_qubits=2 + (i % 5)
            )
            self.db.add_circuit(circuit, metadata)
        
        start_time = time.time()
        
        # Perform various searches
        results1 = self.db.search_circuits(category="algorithm")
        results2 = self.db.search_circuits(n_qubits=4)
        results3 = self.db.search_circuits(tags=["tag5"])
        results4 = self.db.search_circuits(author="Author 5", category="test")
        
        end_time = time.time()
        
        # Should complete quickly
        assert end_time - start_time < 5.0  # 5 seconds for multiple searches
        
        # Verify results are reasonable
        assert len(results1) > 0
        assert len(results2) > 0
        assert len(results3) > 0


if __name__ == "__main__":
    pytest.main([__file__])