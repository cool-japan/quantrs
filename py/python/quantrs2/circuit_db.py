"""
Quantum Circuit Database Module

This module provides a comprehensive database system for storing, retrieving,
and sharing quantum circuits with metadata, versioning, and search capabilities.
"""

import os
import json
import sqlite3
import hashlib
import pickle
import gzip
from typing import Dict, List, Optional, Tuple, Any, Union, Iterator
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

try:
    from . import _quantrs2
    _NATIVE_AVAILABLE = True
except ImportError:
    _NATIVE_AVAILABLE = False


class CircuitMetadata:
    """Metadata for a quantum circuit."""
    
    def __init__(self, 
                 name: str,
                 description: str = "",
                 author: str = "",
                 tags: Optional[List[str]] = None,
                 n_qubits: Optional[int] = None,
                 gate_count: Optional[int] = None,
                 depth: Optional[int] = None,
                 category: str = "general",
                 version: str = "1.0.0",
                 created_at: Optional[datetime] = None,
                 **kwargs):
        """
        Initialize circuit metadata.
        
        Args:
            name: Circuit name
            description: Circuit description
            author: Circuit author
            tags: List of tags for categorization
            n_qubits: Number of qubits
            gate_count: Total number of gates
            depth: Circuit depth
            category: Circuit category
            version: Circuit version
            created_at: Creation timestamp
            **kwargs: Additional metadata fields
        """
        self.name = name
        self.description = description
        self.author = author
        self.tags = tags or []
        self.n_qubits = n_qubits
        self.gate_count = gate_count
        self.depth = depth
        self.category = category
        self.version = version
        self.created_at = created_at or datetime.now(timezone.utc)
        self.custom_fields = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'author': self.author,
            'tags': self.tags,
            'n_qubits': self.n_qubits,
            'gate_count': self.gate_count,
            'depth': self.depth,
            'category': self.category,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            **self.custom_fields
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CircuitMetadata':
        """Create metadata from dictionary."""
        created_at = datetime.fromisoformat(data.pop('created_at'))
        custom_fields = {k: v for k, v in data.items() 
                        if k not in ['name', 'description', 'author', 'tags', 
                                   'n_qubits', 'gate_count', 'depth', 'category', 'version']}
        
        return cls(
            name=data['name'],
            description=data.get('description', ''),
            author=data.get('author', ''),
            tags=data.get('tags', []),
            n_qubits=data.get('n_qubits'),
            gate_count=data.get('gate_count'),
            depth=data.get('depth'),
            category=data.get('category', 'general'),
            version=data.get('version', '1.0.0'),
            created_at=created_at,
            **custom_fields
        )


class CircuitEntry:
    """A circuit database entry containing circuit and metadata."""
    
    def __init__(self, circuit_id: str, circuit: Any, metadata: CircuitMetadata):
        """
        Initialize circuit entry.
        
        Args:
            circuit_id: Unique circuit identifier
            circuit: The quantum circuit object
            metadata: Circuit metadata
        """
        self.circuit_id = circuit_id
        self.circuit = circuit
        self.metadata = metadata
        self.hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute hash of circuit for integrity checking."""
        if hasattr(self.circuit, 'to_qasm'):
            content = self.circuit.to_qasm()
        elif hasattr(self.circuit, '__str__'):
            content = str(self.circuit)
        else:
            content = str(self.circuit)
        
        return hashlib.sha256(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary (without circuit data)."""
        return {
            'circuit_id': self.circuit_id,
            'hash': self.hash,
            'metadata': self.metadata.to_dict()
        }


class CircuitDatabase:
    """
    Quantum circuit database with SQLite backend.
    
    Provides storage, retrieval, search, and management of quantum circuits
    with comprehensive metadata support.
    """
    
    def __init__(self, db_path: Union[str, Path] = None, compression: bool = True):
        """
        Initialize circuit database.
        
        Args:
            db_path: Path to database file (default: ~/.quantrs2/circuits.db)
            compression: Whether to compress circuit data
        """
        if db_path is None:
            db_path = Path.home() / '.quantrs2' / 'circuits.db'
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS circuits (
                    circuit_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    author TEXT,
                    category TEXT,
                    version TEXT,
                    n_qubits INTEGER,
                    gate_count INTEGER,
                    depth INTEGER,
                    created_at TEXT,
                    hash TEXT,
                    circuit_data BLOB,
                    metadata_json TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS tags (
                    circuit_id TEXT,
                    tag TEXT,
                    FOREIGN KEY (circuit_id) REFERENCES circuits (circuit_id),
                    PRIMARY KEY (circuit_id, tag)
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_name ON circuits (name)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_author ON circuits (author)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_category ON circuits (category)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_n_qubits ON circuits (n_qubits)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_tags ON tags (tag)
            ''')
    
    def add_circuit(self, circuit: Any, metadata: CircuitMetadata, 
                   circuit_id: Optional[str] = None, overwrite: bool = False) -> str:
        """
        Add a circuit to the database.
        
        Args:
            circuit: Quantum circuit object
            metadata: Circuit metadata
            circuit_id: Optional circuit ID (auto-generated if None)
            overwrite: Whether to overwrite existing circuit
            
        Returns:
            Circuit ID
        """
        if circuit_id is None:
            circuit_id = self._generate_circuit_id(metadata.name)
        
        # Check if circuit already exists
        if not overwrite and self.exists(circuit_id):
            raise ValueError(f"Circuit {circuit_id} already exists. Use overwrite=True to replace.")
        
        # Extract circuit properties if available
        n_qubits = metadata.n_qubits
        gate_count = metadata.gate_count
        depth = metadata.depth
        
        if n_qubits is None and hasattr(circuit, 'n_qubits'):
            n_qubits = circuit.n_qubits
        elif n_qubits is None and hasattr(circuit, 'num_qubits'):
            n_qubits = circuit.num_qubits()
        
        if gate_count is None and hasattr(circuit, 'gate_count'):
            gate_count = circuit.gate_count()
        elif gate_count is None and hasattr(circuit, 'size'):
            gate_count = circuit.size()
        
        if depth is None and hasattr(circuit, 'depth'):
            depth = circuit.depth()
        
        # Update metadata with extracted properties
        metadata.n_qubits = n_qubits
        metadata.gate_count = gate_count
        metadata.depth = depth
        
        entry = CircuitEntry(circuit_id, circuit, metadata)
        
        # Serialize circuit
        circuit_data = self._serialize_circuit(circuit)
        
        with sqlite3.connect(self.db_path) as conn:
            # Insert or replace circuit
            conn.execute('''
                INSERT OR REPLACE INTO circuits (
                    circuit_id, name, description, author, category, version,
                    n_qubits, gate_count, depth, created_at, hash,
                    circuit_data, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                circuit_id, metadata.name, metadata.description, metadata.author,
                metadata.category, metadata.version, n_qubits, gate_count, depth,
                metadata.created_at.isoformat(), entry.hash,
                circuit_data, json.dumps(metadata.to_dict())
            ))
            
            # Remove existing tags
            conn.execute('DELETE FROM tags WHERE circuit_id = ?', (circuit_id,))
            
            # Insert tags
            for tag in metadata.tags:
                conn.execute('INSERT INTO tags (circuit_id, tag) VALUES (?, ?)', 
                           (circuit_id, tag))
        
        return circuit_id
    
    def get_circuit(self, circuit_id: str) -> Optional[CircuitEntry]:
        """
        Retrieve a circuit by ID.
        
        Args:
            circuit_id: Circuit identifier
            
        Returns:
            CircuitEntry or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT * FROM circuits WHERE circuit_id = ?
            ''', (circuit_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Get tags
            tag_cursor = conn.execute('''
                SELECT tag FROM tags WHERE circuit_id = ?
            ''', (circuit_id,))
            tags = [row[0] for row in tag_cursor.fetchall()]
            
            # Deserialize circuit
            circuit = self._deserialize_circuit(row['circuit_data'])
            
            # Create metadata
            metadata_dict = json.loads(row['metadata_json'])
            metadata_dict['tags'] = tags
            metadata = CircuitMetadata.from_dict(metadata_dict)
            
            return CircuitEntry(circuit_id, circuit, metadata)
    
    def exists(self, circuit_id: str) -> bool:
        """Check if a circuit exists in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT 1 FROM circuits WHERE circuit_id = ? LIMIT 1
            ''', (circuit_id,))
            return cursor.fetchone() is not None
    
    def delete_circuit(self, circuit_id: str) -> bool:
        """
        Delete a circuit from the database.
        
        Args:
            circuit_id: Circuit identifier
            
        Returns:
            True if circuit was deleted, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('DELETE FROM circuits WHERE circuit_id = ?', (circuit_id,))
            conn.execute('DELETE FROM tags WHERE circuit_id = ?', (circuit_id,))
            return cursor.rowcount > 0
    
    def search_circuits(self, 
                       name_pattern: Optional[str] = None,
                       author: Optional[str] = None,
                       category: Optional[str] = None,
                       tags: Optional[List[str]] = None,
                       n_qubits: Optional[Union[int, Tuple[int, int]]] = None,
                       min_depth: Optional[int] = None,
                       max_depth: Optional[int] = None,
                       limit: Optional[int] = None) -> List[CircuitEntry]:
        """
        Search circuits by various criteria.
        
        Args:
            name_pattern: Name pattern (supports SQL LIKE syntax)
            author: Author name
            category: Circuit category
            tags: List of required tags
            n_qubits: Number of qubits (int) or range (tuple)
            min_depth: Minimum circuit depth
            max_depth: Maximum circuit depth
            limit: Maximum number of results
            
        Returns:
            List of matching circuit entries
        """
        conditions = []
        params = []
        
        if name_pattern:
            conditions.append('name LIKE ?')
            params.append(name_pattern)
        
        if author:
            conditions.append('author = ?')
            params.append(author)
        
        if category:
            conditions.append('category = ?')
            params.append(category)
        
        if isinstance(n_qubits, int):
            conditions.append('n_qubits = ?')
            params.append(n_qubits)
        elif isinstance(n_qubits, tuple) and len(n_qubits) == 2:
            conditions.append('n_qubits BETWEEN ? AND ?')
            params.extend(n_qubits)
        
        if min_depth is not None:
            conditions.append('depth >= ?')
            params.append(min_depth)
        
        if max_depth is not None:
            conditions.append('depth <= ?')
            params.append(max_depth)
        
        query = 'SELECT DISTINCT circuit_id FROM circuits'
        
        if tags:
            # Join with tags table for tag filtering
            tag_placeholders = ','.join(['?'] * len(tags))
            query += f'''
                JOIN tags ON circuits.circuit_id = tags.circuit_id
                WHERE tags.tag IN ({tag_placeholders})
            '''
            params = list(tags) + params
            
            if conditions:
                query += ' AND ' + ' AND '.join(conditions)
            
            # Ensure all required tags are present
            query += f' GROUP BY circuits.circuit_id HAVING COUNT(DISTINCT tags.tag) = {len(tags)}'
        else:
            if conditions:
                query += ' WHERE ' + ' AND '.join(conditions)
        
        query += ' ORDER BY created_at DESC'
        
        if limit:
            query += ' LIMIT ?'
            params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            circuit_ids = [row[0] for row in cursor.fetchall()]
        
        # Retrieve full circuit entries
        results = []
        for circuit_id in circuit_ids:
            entry = self.get_circuit(circuit_id)
            if entry:
                results.append(entry)
        
        return results
    
    def list_circuits(self, limit: Optional[int] = None, 
                     offset: int = 0) -> List[Dict[str, Any]]:
        """
        List all circuits with basic information.
        
        Args:
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of circuit information dictionaries
        """
        query = '''
            SELECT circuit_id, name, author, category, n_qubits, 
                   gate_count, depth, created_at
            FROM circuits 
            ORDER BY created_at DESC
        '''
        
        params = []
        if limit:
            query += ' LIMIT ? OFFSET ?'
            params.extend([limit, offset])
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            
            results = []
            for row in cursor.fetchall():
                results.append(dict(row))
            
            return results
    
    def get_categories(self) -> List[str]:
        """Get all unique categories."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT DISTINCT category FROM circuits ORDER BY category')
            return [row[0] for row in cursor.fetchall() if row[0]]
    
    def get_tags(self) -> List[str]:
        """Get all unique tags."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT DISTINCT tag FROM tags ORDER BY tag')
            return [row[0] for row in cursor.fetchall()]
    
    def get_authors(self) -> List[str]:
        """Get all unique authors."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT DISTINCT author FROM circuits ORDER BY author')
            return [row[0] for row in cursor.fetchall() if row[0]]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT 
                    COUNT(*) as total_circuits,
                    COUNT(DISTINCT author) as unique_authors,
                    COUNT(DISTINCT category) as unique_categories,
                    AVG(n_qubits) as avg_qubits,
                    AVG(gate_count) as avg_gate_count,
                    AVG(depth) as avg_depth,
                    MIN(n_qubits) as min_qubits,
                    MAX(n_qubits) as max_qubits
                FROM circuits
            ''')
            
            stats = dict(cursor.fetchone())
            
            # Get tag count
            cursor = conn.execute('SELECT COUNT(DISTINCT tag) FROM tags')
            stats['unique_tags'] = cursor.fetchone()[0]
            
            return stats
    
    def export_circuit(self, circuit_id: str, file_path: Union[str, Path], 
                      format: str = 'qasm') -> None:
        """
        Export a circuit to a file.
        
        Args:
            circuit_id: Circuit identifier
            file_path: Output file path
            format: Export format ('qasm', 'json', 'pickle')
        """
        entry = self.get_circuit(circuit_id)
        if not entry:
            raise ValueError(f"Circuit {circuit_id} not found")
        
        file_path = Path(file_path)
        
        if format.lower() == 'qasm':
            if hasattr(entry.circuit, 'to_qasm'):
                content = entry.circuit.to_qasm()
            else:
                raise ValueError("Circuit does not support QASM export")
            
            with open(file_path, 'w') as f:
                f.write(content)
        
        elif format.lower() == 'json':
            data = {
                'circuit_id': entry.circuit_id,
                'metadata': entry.metadata.to_dict(),
                'qasm': entry.circuit.to_qasm() if hasattr(entry.circuit, 'to_qasm') else None
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format.lower() == 'pickle':
            with open(file_path, 'wb') as f:
                pickle.dump(entry, f)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def import_circuit(self, file_path: Union[str, Path], 
                      format: str = 'auto', **metadata_kwargs) -> str:
        """
        Import a circuit from a file.
        
        Args:
            file_path: Input file path
            format: Import format ('qasm', 'json', 'pickle', 'auto')
            **metadata_kwargs: Additional metadata fields
            
        Returns:
            Circuit ID
        """
        file_path = Path(file_path)
        
        if format == 'auto':
            # Auto-detect format from extension
            ext = file_path.suffix.lower()
            if ext in ['.qasm', '.qasm3']:
                format = 'qasm'
            elif ext == '.json':
                format = 'json'
            elif ext in ['.pkl', '.pickle']:
                format = 'pickle'
            else:
                raise ValueError(f"Cannot auto-detect format for extension: {ext}")
        
        if format.lower() == 'qasm':
            if not _NATIVE_AVAILABLE:
                raise ImportError("Native quantrs2 module required for QASM import")
            
            with open(file_path, 'r') as f:
                qasm_content = f.read()
            
            # Parse QASM (implementation depends on available parser)
            # For now, create a simple metadata entry
            metadata = CircuitMetadata(
                name=file_path.stem,
                description=f"Imported from {file_path}",
                **metadata_kwargs
            )
            
            # Create a placeholder circuit object
            # In a real implementation, this would parse the QASM
            circuit = {'qasm': qasm_content, 'type': 'qasm'}
            
            return self.add_circuit(circuit, metadata)
        
        elif format.lower() == 'json':
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            metadata = CircuitMetadata.from_dict(data['metadata'])
            # Update with any additional kwargs
            for key, value in metadata_kwargs.items():
                setattr(metadata, key, value)
            
            circuit = data.get('qasm', data)
            
            return self.add_circuit(circuit, metadata)
        
        elif format.lower() == 'pickle':
            with open(file_path, 'rb') as f:
                entry = pickle.load(f)
            
            if not isinstance(entry, CircuitEntry):
                raise ValueError("Pickled file does not contain a CircuitEntry")
            
            # Update metadata with kwargs
            for key, value in metadata_kwargs.items():
                setattr(entry.metadata, key, value)
            
            return self.add_circuit(entry.circuit, entry.metadata, entry.circuit_id)
        
        else:
            raise ValueError(f"Unsupported import format: {format}")
    
    def backup(self, backup_path: Union[str, Path]) -> None:
        """Create a backup of the database."""
        backup_path = Path(backup_path)
        
        # Copy database file
        import shutil
        shutil.copy2(self.db_path, backup_path)
    
    def restore(self, backup_path: Union[str, Path]) -> None:
        """Restore database from backup."""
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        # Copy backup over current database
        import shutil
        shutil.copy2(backup_path, self.db_path)
    
    def _generate_circuit_id(self, name: str) -> str:
        """Generate a unique circuit ID."""
        base_id = name.lower().replace(' ', '_').replace('-', '_')
        base_id = ''.join(c for c in base_id if c.isalnum() or c == '_')
        
        # Ensure uniqueness by appending timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{base_id}_{timestamp}"
    
    def _serialize_circuit(self, circuit: Any) -> bytes:
        """Serialize circuit for database storage."""
        data = pickle.dumps(circuit)
        
        if self.compression:
            data = gzip.compress(data)
        
        return data
    
    def _deserialize_circuit(self, data: bytes) -> Any:
        """Deserialize circuit from database storage."""
        if self.compression:
            data = gzip.decompress(data)
        
        return pickle.loads(data)


# Predefined circuit templates
class CircuitTemplates:
    """Collection of common quantum circuit templates."""
    
    @staticmethod
    def bell_state() -> Tuple[Any, CircuitMetadata]:
        """Create a Bell state circuit template."""
        if _NATIVE_AVAILABLE:
            circuit = _quantrs2.PyCircuit(2)
            circuit.h(0)
            circuit.cnot(0, 1)
        else:
            circuit = {'gates': [('h', [0]), ('cnot', [0, 1])], 'n_qubits': 2}
        
        metadata = CircuitMetadata(
            name="Bell State",
            description="Creates a maximally entangled Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2",
            category="entanglement",
            tags=["bell", "entanglement", "two-qubit"],
            n_qubits=2
        )
        
        return circuit, metadata
    
    @staticmethod
    def ghz_state(n_qubits: int = 3) -> Tuple[Any, CircuitMetadata]:
        """Create a GHZ state circuit template."""
        if _NATIVE_AVAILABLE:
            circuit = _quantrs2.PyCircuit(n_qubits)
            circuit.h(0)
            for i in range(1, n_qubits):
                circuit.cnot(0, i)
        else:
            gates = [('h', [0])]
            gates.extend([('cnot', [0, i]) for i in range(1, n_qubits)])
            circuit = {'gates': gates, 'n_qubits': n_qubits}
        
        metadata = CircuitMetadata(
            name=f"GHZ State ({n_qubits} qubits)",
            description=f"Creates a {n_qubits}-qubit GHZ state for multiparticle entanglement",
            category="entanglement",
            tags=["ghz", "entanglement", "multiparticle"],
            n_qubits=n_qubits
        )
        
        return circuit, metadata
    
    @staticmethod
    def quantum_fourier_transform(n_qubits: int) -> Tuple[Any, CircuitMetadata]:
        """Create a Quantum Fourier Transform circuit template."""
        if _NATIVE_AVAILABLE:
            circuit = _quantrs2.PyCircuit(n_qubits)
            # QFT implementation
            for i in range(n_qubits):
                circuit.h(i)
                for j in range(i + 1, n_qubits):
                    circuit.cp(i, j, np.pi / (2 ** (j - i)))
        else:
            gates = []
            for i in range(n_qubits):
                gates.append(('h', [i]))
                for j in range(i + 1, n_qubits):
                    gates.append(('cp', [i, j, np.pi / (2 ** (j - i))]))
            circuit = {'gates': gates, 'n_qubits': n_qubits}
        
        metadata = CircuitMetadata(
            name=f"Quantum Fourier Transform ({n_qubits} qubits)",
            description="Quantum Fourier Transform for quantum algorithm applications",
            category="algorithm",
            tags=["qft", "fourier", "algorithm"],
            n_qubits=n_qubits
        )
        
        return circuit, metadata


# Convenience functions
def create_circuit_database(db_path: Optional[Union[str, Path]] = None) -> CircuitDatabase:
    """Create a new circuit database instance."""
    return CircuitDatabase(db_path)


def populate_template_circuits(db: CircuitDatabase) -> None:
    """Populate database with common circuit templates."""
    templates = [
        CircuitTemplates.bell_state(),
        CircuitTemplates.ghz_state(3),
        CircuitTemplates.ghz_state(4),
        CircuitTemplates.quantum_fourier_transform(3),
        CircuitTemplates.quantum_fourier_transform(4),
    ]
    
    for circuit, metadata in templates:
        try:
            db.add_circuit(circuit, metadata)
        except ValueError:
            # Circuit already exists
            pass


__all__ = [
    'CircuitMetadata',
    'CircuitEntry', 
    'CircuitDatabase',
    'CircuitTemplates',
    'create_circuit_database',
    'populate_template_circuits',
]