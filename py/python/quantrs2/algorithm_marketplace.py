"""
Quantum Algorithm Marketplace

This module provides a comprehensive marketplace platform for sharing, discovering,
and collaborating on quantum algorithms, circuits, and applications.
"""

import json
import time
import hashlib
import sqlite3
import tempfile
import threading
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union, Set
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum
import logging
import socket
import http.server
import socketserver
import urllib.parse
import urllib.request
import base64
import uuid

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    import io
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from . import algorithm_debugger
    from . import profiler
    from . import qasm
    from . import circuit_builder
    QUANTRS_MODULES_AVAILABLE = True
except ImportError:
    QUANTRS_MODULES_AVAILABLE = False


class AlgorithmCategory(Enum):
    """Algorithm categories."""
    OPTIMIZATION = "optimization"
    MACHINE_LEARNING = "machine_learning"
    CRYPTOGRAPHY = "cryptography"
    SIMULATION = "simulation"
    FINANCE = "finance"
    CHEMISTRY = "chemistry"
    NETWORKING = "networking"
    ARITHMETIC = "arithmetic"
    SEARCH = "search"
    FACTORING = "factoring"
    ERROR_CORRECTION = "error_correction"
    BENCHMARKING = "benchmarking"
    GENERAL = "general"


class AlgorithmType(Enum):
    """Algorithm types."""
    CIRCUIT = "circuit"
    VARIATIONAL = "variational"
    HYBRID = "hybrid"
    TEMPLATE = "template"
    SUBROUTINE = "subroutine"
    APPLICATION = "application"


class LicenseType(Enum):
    """License types."""
    MIT = "MIT"
    APACHE_2_0 = "Apache-2.0"
    GPL_3_0 = "GPL-3.0"
    BSD_3_CLAUSE = "BSD-3-Clause"
    PROPRIETARY = "Proprietary"
    CC_BY_4_0 = "CC-BY-4.0"
    CC_BY_SA_4_0 = "CC-BY-SA-4.0"
    UNLICENSE = "Unlicense"


class QualityMetric(Enum):
    """Quality metrics."""
    CORRECTNESS = "correctness"
    PERFORMANCE = "performance"
    READABILITY = "readability"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    INNOVATION = "innovation"


class MarketplaceStatus(Enum):
    """Marketplace entry status."""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class AlgorithmMetadata:
    """Metadata for quantum algorithms."""
    algorithm_id: str
    name: str
    description: str
    author: str
    version: str
    category: AlgorithmCategory
    algorithm_type: AlgorithmType
    license: LicenseType
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    min_qubits: int = 1
    max_qubits: Optional[int] = None
    complexity_class: Optional[str] = None
    quantum_advantage: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['category'] = self.category.value
        data['algorithm_type'] = self.algorithm_type.value
        data['license'] = self.license.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlgorithmMetadata':
        """Create from dictionary."""
        data = data.copy()
        data['category'] = AlgorithmCategory(data['category'])
        data['algorithm_type'] = AlgorithmType(data['algorithm_type'])
        data['license'] = LicenseType(data['license'])
        return cls(**data)


@dataclass
class AlgorithmRating:
    """Rating and review for an algorithm."""
    rating_id: str
    algorithm_id: str
    reviewer: str
    overall_rating: float  # 1-5 stars
    quality_metrics: Dict[QualityMetric, float] = field(default_factory=dict)
    review_text: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    verified_execution: bool = False
    benchmark_results: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['quality_metrics'] = {k.value: v for k, v in self.quality_metrics.items()}
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlgorithmRating':
        """Create from dictionary."""
        data = data.copy()
        if 'quality_metrics' in data:
            data['quality_metrics'] = {QualityMetric(k): v for k, v in data['quality_metrics'].items()}
        return cls(**data)


@dataclass
class MarketplaceEntry:
    """Complete marketplace entry."""
    entry_id: str
    metadata: AlgorithmMetadata
    algorithm_data: Dict[str, Any]  # Circuit, code, parameters
    documentation: str
    examples: List[Dict[str, Any]] = field(default_factory=list)
    test_cases: List[Dict[str, Any]] = field(default_factory=list)
    status: MarketplaceStatus = MarketplaceStatus.DRAFT
    download_count: int = 0
    rating_count: int = 0
    average_rating: float = 0.0
    featured: bool = False
    checksum: Optional[str] = None
    
    def __post_init__(self):
        """Calculate checksum."""
        if self.checksum is None:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate algorithm checksum."""
        content = json.dumps({
            'metadata': self.metadata.to_dict(),
            'algorithm_data': self.algorithm_data,
            'documentation': self.documentation
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'entry_id': self.entry_id,
            'metadata': self.metadata.to_dict(),
            'algorithm_data': self.algorithm_data,
            'documentation': self.documentation,
            'examples': self.examples,
            'test_cases': self.test_cases,
            'status': self.status.value,
            'download_count': self.download_count,
            'rating_count': self.rating_count,
            'average_rating': self.average_rating,
            'featured': self.featured,
            'checksum': self.checksum
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketplaceEntry':
        """Create from dictionary."""
        data = data.copy()
        data['metadata'] = AlgorithmMetadata.from_dict(data['metadata'])
        data['status'] = MarketplaceStatus(data['status'])
        return cls(**data)


class AlgorithmValidator:
    """Validates quantum algorithms for marketplace submission."""
    
    def __init__(self):
        self.required_fields = [
            'name', 'description', 'author', 'version', 'category', 'algorithm_type'
        ]
        self.validation_rules = [
            self._validate_metadata,
            self._validate_algorithm_data,
            self._validate_documentation,
            self._validate_examples,
            self._validate_test_cases
        ]
    
    def validate_entry(self, entry: MarketplaceEntry) -> Tuple[bool, List[str]]:
        """Validate marketplace entry."""
        errors = []
        
        try:
            for rule in self.validation_rules:
                rule_errors = rule(entry)
                errors.extend(rule_errors)
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return False, errors
    
    def _validate_metadata(self, entry: MarketplaceEntry) -> List[str]:
        """Validate algorithm metadata."""
        errors = []
        metadata = entry.metadata
        
        # Check required fields
        for field in self.required_fields:
            if not hasattr(metadata, field) or not getattr(metadata, field):
                errors.append(f"Missing required field: {field}")
        
        # Validate name
        if hasattr(metadata, 'name') and len(metadata.name) < 3:
            errors.append("Algorithm name must be at least 3 characters")
        
        # Validate description
        if hasattr(metadata, 'description') and len(metadata.description) < 20:
            errors.append("Description must be at least 20 characters")
        
        # Validate version format
        if hasattr(metadata, 'version'):
            try:
                parts = metadata.version.split('.')
                if len(parts) < 2 or not all(part.isdigit() for part in parts):
                    errors.append("Version must be in format X.Y or X.Y.Z")
            except:
                errors.append("Invalid version format")
        
        # Validate qubit requirements
        if metadata.min_qubits < 1:
            errors.append("Minimum qubits must be at least 1")
        
        if metadata.max_qubits and metadata.max_qubits < metadata.min_qubits:
            errors.append("Maximum qubits cannot be less than minimum qubits")
        
        return errors
    
    def _validate_algorithm_data(self, entry: MarketplaceEntry) -> List[str]:
        """Validate algorithm data structure."""
        errors = []
        data = entry.algorithm_data
        
        if not data:
            errors.append("Algorithm data cannot be empty")
            return errors
        
        # Check for required algorithm components
        if entry.metadata.algorithm_type == AlgorithmType.CIRCUIT:
            if 'gates' not in data and 'qasm' not in data and 'circuit' not in data:
                errors.append("Circuit algorithms must include gates, QASM, or circuit data")
        
        elif entry.metadata.algorithm_type == AlgorithmType.VARIATIONAL:
            if 'parameters' not in data:
                errors.append("Variational algorithms must include parameters")
        
        # Validate circuit structure if present
        if 'gates' in data:
            gates_errors = self._validate_gates(data['gates'])
            errors.extend(gates_errors)
        
        # Validate QASM if present
        if 'qasm' in data and QUANTRS_MODULES_AVAILABLE:
            try:
                qasm.parse_qasm(data['qasm'])
            except Exception as e:
                errors.append(f"Invalid QASM: {str(e)}")
        
        return errors
    
    def _validate_gates(self, gates: List[Dict[str, Any]]) -> List[str]:
        """Validate gate sequence."""
        errors = []
        
        if not isinstance(gates, list):
            errors.append("Gates must be a list")
            return errors
        
        for i, gate in enumerate(gates):
            if not isinstance(gate, dict):
                errors.append(f"Gate {i} must be a dictionary")
                continue
            
            if 'gate' not in gate:
                errors.append(f"Gate {i} missing 'gate' field")
            
            if 'qubits' not in gate:
                errors.append(f"Gate {i} missing 'qubits' field")
            
            # Validate qubit indices
            if 'qubits' in gate:
                qubits = gate['qubits']
                if not isinstance(qubits, list):
                    errors.append(f"Gate {i} qubits must be a list")
                elif not all(isinstance(q, int) and q >= 0 for q in qubits):
                    errors.append(f"Gate {i} qubits must be non-negative integers")
        
        return errors
    
    def _validate_documentation(self, entry: MarketplaceEntry) -> List[str]:
        """Validate documentation quality."""
        errors = []
        doc = entry.documentation
        
        if not doc or len(doc.strip()) < 100:
            errors.append("Documentation must be at least 100 characters")
        
        # Check for common documentation sections
        doc_lower = doc.lower()
        required_sections = ['description', 'usage', 'parameters']
        missing_sections = []
        
        for section in required_sections:
            if section not in doc_lower:
                missing_sections.append(section)
        
        if missing_sections:
            errors.append(f"Documentation missing sections: {', '.join(missing_sections)}")
        
        return errors
    
    def _validate_examples(self, entry: MarketplaceEntry) -> List[str]:
        """Validate examples."""
        errors = []
        
        if not entry.examples:
            errors.append("At least one example is required")
            return errors
        
        for i, example in enumerate(entry.examples):
            if not isinstance(example, dict):
                errors.append(f"Example {i} must be a dictionary")
                continue
            
            if 'name' not in example:
                errors.append(f"Example {i} missing name")
            
            if 'code' not in example and 'circuit' not in example:
                errors.append(f"Example {i} missing code or circuit")
            
            if 'description' not in example:
                errors.append(f"Example {i} missing description")
        
        return errors
    
    def _validate_test_cases(self, entry: MarketplaceEntry) -> List[str]:
        """Validate test cases."""
        errors = []
        
        if not entry.test_cases:
            errors.append("At least one test case is required")
            return errors
        
        for i, test_case in enumerate(entry.test_cases):
            if not isinstance(test_case, dict):
                errors.append(f"Test case {i} must be a dictionary")
                continue
            
            if 'name' not in test_case:
                errors.append(f"Test case {i} missing name")
            
            if 'input' not in test_case:
                errors.append(f"Test case {i} missing input")
            
            if 'expected_output' not in test_case:
                errors.append(f"Test case {i} missing expected output")
        
        return errors
    
    def estimate_quality_score(self, entry: MarketplaceEntry) -> float:
        """Estimate algorithm quality score."""
        score = 0.0
        max_score = 100.0
        
        # Documentation quality (20 points)
        doc_length = len(entry.documentation)
        if doc_length >= 500:
            score += 20
        elif doc_length >= 200:
            score += 15
        elif doc_length >= 100:
            score += 10
        
        # Examples quality (20 points)
        example_count = len(entry.examples)
        score += min(example_count * 5, 20)
        
        # Test cases quality (20 points)
        test_count = len(entry.test_cases)
        score += min(test_count * 4, 20)
        
        # Metadata completeness (20 points)
        metadata_score = 0
        if entry.metadata.tags:
            metadata_score += 5
        if entry.metadata.dependencies:
            metadata_score += 5
        if entry.metadata.complexity_class:
            metadata_score += 5
        if entry.metadata.quantum_advantage:
            metadata_score += 5
        score += metadata_score
        
        # Algorithm complexity (20 points)
        if 'gates' in entry.algorithm_data:
            gate_count = len(entry.algorithm_data['gates'])
            if gate_count >= 10:
                score += 20
            elif gate_count >= 5:
                score += 15
            elif gate_count >= 2:
                score += 10
        else:
            score += 10  # Non-circuit algorithms get base points
        
        return min(score, max_score)


class AlgorithmMarketplaceDB:
    """Database management for the algorithm marketplace."""
    
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.connection = None
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        
        # Create tables
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS algorithms (
                entry_id TEXT PRIMARY KEY,
                metadata TEXT NOT NULL,
                algorithm_data TEXT NOT NULL,
                documentation TEXT NOT NULL,
                examples TEXT NOT NULL,
                test_cases TEXT NOT NULL,
                status TEXT NOT NULL,
                download_count INTEGER DEFAULT 0,
                rating_count INTEGER DEFAULT 0,
                average_rating REAL DEFAULT 0.0,
                featured BOOLEAN DEFAULT FALSE,
                checksum TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS ratings (
                rating_id TEXT PRIMARY KEY,
                algorithm_id TEXT NOT NULL,
                reviewer TEXT NOT NULL,
                overall_rating REAL NOT NULL,
                quality_metrics TEXT,
                review_text TEXT,
                created_at REAL NOT NULL,
                verified_execution BOOLEAN DEFAULT FALSE,
                benchmark_results TEXT,
                FOREIGN KEY (algorithm_id) REFERENCES algorithms (entry_id)
            );
            
            CREATE TABLE IF NOT EXISTS downloads (
                download_id TEXT PRIMARY KEY,
                algorithm_id TEXT NOT NULL,
                downloader TEXT,
                downloaded_at REAL NOT NULL,
                version TEXT,
                FOREIGN KEY (algorithm_id) REFERENCES algorithms (entry_id)
            );
            
            CREATE TABLE IF NOT EXISTS collections (
                collection_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                owner TEXT NOT NULL,
                algorithm_ids TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                public BOOLEAN DEFAULT TRUE
            );
            
            CREATE INDEX IF NOT EXISTS idx_algorithms_category ON algorithms(json_extract(metadata, '$.category'));
            CREATE INDEX IF NOT EXISTS idx_algorithms_author ON algorithms(json_extract(metadata, '$.author'));
            CREATE INDEX IF NOT EXISTS idx_algorithms_status ON algorithms(status);
            CREATE INDEX IF NOT EXISTS idx_ratings_algorithm ON ratings(algorithm_id);
            CREATE INDEX IF NOT EXISTS idx_downloads_algorithm ON downloads(algorithm_id);
        """)
        
        self.connection.commit()
    
    def submit_algorithm(self, entry: MarketplaceEntry) -> bool:
        """Submit algorithm to marketplace."""
        try:
            # Convert entry to database format
            entry_data = {
                'entry_id': entry.entry_id,
                'metadata': json.dumps(entry.metadata.to_dict()),
                'algorithm_data': json.dumps(entry.algorithm_data),
                'documentation': entry.documentation,
                'examples': json.dumps(entry.examples),
                'test_cases': json.dumps(entry.test_cases),
                'status': entry.status.value,
                'download_count': entry.download_count,
                'rating_count': entry.rating_count,
                'average_rating': entry.average_rating,
                'featured': entry.featured,
                'checksum': entry.checksum,
                'created_at': entry.metadata.created_at,
                'updated_at': entry.metadata.updated_at
            }
            
            # Insert into database
            self.connection.execute("""
                INSERT OR REPLACE INTO algorithms 
                (entry_id, metadata, algorithm_data, documentation, examples, test_cases, 
                 status, download_count, rating_count, average_rating, featured, checksum, 
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, tuple(entry_data.values()))
            
            self.connection.commit()
            return True
            
        except Exception as e:
            logging.error(f"Failed to submit algorithm: {e}")
            return False
    
    def get_algorithm(self, entry_id: str) -> Optional[MarketplaceEntry]:
        """Get algorithm by ID."""
        try:
            cursor = self.connection.execute(
                "SELECT * FROM algorithms WHERE entry_id = ?", (entry_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Convert back to MarketplaceEntry
            entry_data = {
                'entry_id': row['entry_id'],
                'metadata': json.loads(row['metadata']),
                'algorithm_data': json.loads(row['algorithm_data']),
                'documentation': row['documentation'],
                'examples': json.loads(row['examples']),
                'test_cases': json.loads(row['test_cases']),
                'status': row['status'],
                'download_count': row['download_count'],
                'rating_count': row['rating_count'],
                'average_rating': row['average_rating'],
                'featured': row['featured'],
                'checksum': row['checksum']
            }
            
            return MarketplaceEntry.from_dict(entry_data)
            
        except Exception as e:
            logging.error(f"Failed to get algorithm: {e}")
            return None
    
    def search_algorithms(self, query: str = "", category: Optional[AlgorithmCategory] = None,
                         algorithm_type: Optional[AlgorithmType] = None,
                         author: Optional[str] = None, tags: Optional[List[str]] = None,
                         min_rating: Optional[float] = None,
                         limit: int = 50, offset: int = 0) -> List[MarketplaceEntry]:
        """Search algorithms with filters."""
        try:
            conditions = ["status = 'approved'"]
            params = []
            
            if query:
                conditions.append("(json_extract(metadata, '$.name') LIKE ? OR json_extract(metadata, '$.description') LIKE ?)")
                params.extend([f"%{query}%", f"%{query}%"])
            
            if category:
                conditions.append("json_extract(metadata, '$.category') = ?")
                params.append(category.value)
            
            if algorithm_type:
                conditions.append("json_extract(metadata, '$.algorithm_type') = ?")
                params.append(algorithm_type.value)
            
            if author:
                conditions.append("json_extract(metadata, '$.author') LIKE ?")
                params.append(f"%{author}%")
            
            if min_rating:
                conditions.append("average_rating >= ?")
                params.append(min_rating)
            
            # Build query
            where_clause = " AND ".join(conditions)
            sql = f"""
                SELECT * FROM algorithms 
                WHERE {where_clause}
                ORDER BY average_rating DESC, download_count DESC
                LIMIT ? OFFSET ?
            """
            params.extend([limit, offset])
            
            cursor = self.connection.execute(sql, params)
            rows = cursor.fetchall()
            
            # Convert to MarketplaceEntry objects
            entries = []
            for row in rows:
                try:
                    entry_data = {
                        'entry_id': row['entry_id'],
                        'metadata': json.loads(row['metadata']),
                        'algorithm_data': json.loads(row['algorithm_data']),
                        'documentation': row['documentation'],
                        'examples': json.loads(row['examples']),
                        'test_cases': json.loads(row['test_cases']),
                        'status': row['status'],
                        'download_count': row['download_count'],
                        'rating_count': row['rating_count'],
                        'average_rating': row['average_rating'],
                        'featured': row['featured'],
                        'checksum': row['checksum']
                    }
                    
                    entry = MarketplaceEntry.from_dict(entry_data)
                    entries.append(entry)
                    
                except Exception as e:
                    logging.warning(f"Failed to parse entry {row['entry_id']}: {e}")
                    continue
            
            return entries
            
        except Exception as e:
            logging.error(f"Search failed: {e}")
            return []
    
    def add_rating(self, rating: AlgorithmRating) -> bool:
        """Add rating for an algorithm."""
        try:
            # Insert rating
            rating_data = {
                'rating_id': rating.rating_id,
                'algorithm_id': rating.algorithm_id,
                'reviewer': rating.reviewer,
                'overall_rating': rating.overall_rating,
                'quality_metrics': json.dumps({k.value: v for k, v in rating.quality_metrics.items()}),
                'review_text': rating.review_text,
                'created_at': rating.created_at,
                'verified_execution': rating.verified_execution,
                'benchmark_results': json.dumps(rating.benchmark_results) if rating.benchmark_results else None
            }
            
            self.connection.execute("""
                INSERT INTO ratings 
                (rating_id, algorithm_id, reviewer, overall_rating, quality_metrics, 
                 review_text, created_at, verified_execution, benchmark_results)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, tuple(rating_data.values()))
            
            # Update algorithm average rating
            self._update_algorithm_rating(rating.algorithm_id)
            
            self.connection.commit()
            return True
            
        except Exception as e:
            logging.error(f"Failed to add rating: {e}")
            return False
    
    def _update_algorithm_rating(self, algorithm_id: str):
        """Update algorithm's average rating."""
        cursor = self.connection.execute("""
            SELECT AVG(overall_rating) as avg_rating, COUNT(*) as count
            FROM ratings WHERE algorithm_id = ?
        """, (algorithm_id,))
        
        result = cursor.fetchone()
        avg_rating = result['avg_rating'] or 0.0
        count = result['count'] or 0
        
        self.connection.execute("""
            UPDATE algorithms 
            SET average_rating = ?, rating_count = ?
            WHERE entry_id = ?
        """, (avg_rating, count, algorithm_id))
    
    def record_download(self, algorithm_id: str, downloader: Optional[str] = None) -> bool:
        """Record algorithm download."""
        try:
            download_id = str(uuid.uuid4())
            
            self.connection.execute("""
                INSERT INTO downloads (download_id, algorithm_id, downloader, downloaded_at)
                VALUES (?, ?, ?, ?)
            """, (download_id, algorithm_id, downloader, time.time()))
            
            # Update download count
            self.connection.execute("""
                UPDATE algorithms 
                SET download_count = download_count + 1
                WHERE entry_id = ?
            """, (algorithm_id,))
            
            self.connection.commit()
            return True
            
        except Exception as e:
            logging.error(f"Failed to record download: {e}")
            return False
    
    def get_featured_algorithms(self, limit: int = 10) -> List[MarketplaceEntry]:
        """Get featured algorithms."""
        return self.search_algorithms(limit=limit)  # Could add featured filtering
    
    def get_popular_algorithms(self, limit: int = 10) -> List[MarketplaceEntry]:
        """Get popular algorithms by download count."""
        try:
            cursor = self.connection.execute("""
                SELECT * FROM algorithms 
                WHERE status = 'approved'
                ORDER BY download_count DESC, average_rating DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            
            entries = []
            for row in rows:
                try:
                    entry_data = {
                        'entry_id': row['entry_id'],
                        'metadata': json.loads(row['metadata']),
                        'algorithm_data': json.loads(row['algorithm_data']),
                        'documentation': row['documentation'],
                        'examples': json.loads(row['examples']),
                        'test_cases': json.loads(row['test_cases']),
                        'status': row['status'],
                        'download_count': row['download_count'],
                        'rating_count': row['rating_count'],
                        'average_rating': row['average_rating'],
                        'featured': row['featured'],
                        'checksum': row['checksum']
                    }
                    
                    entry = MarketplaceEntry.from_dict(entry_data)
                    entries.append(entry)
                    
                except Exception as e:
                    logging.warning(f"Failed to parse popular entry {row['entry_id']}: {e}")
                    continue
            
            return entries
            
        except Exception as e:
            logging.error(f"Failed to get popular algorithms: {e}")
            return []
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()


class AlgorithmPackager:
    """Packages algorithms for distribution."""
    
    def __init__(self):
        self.supported_formats = ['zip', 'tar.gz', 'json']
    
    def package_algorithm(self, entry: MarketplaceEntry, format: str = 'zip') -> Optional[bytes]:
        """Package algorithm for download."""
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format}")
        
        try:
            if format == 'json':
                return self._package_json(entry)
            elif format == 'zip':
                return self._package_zip(entry)
            elif format == 'tar.gz':
                return self._package_tar(entry)
            
        except Exception as e:
            logging.error(f"Packaging failed: {e}")
            return None
    
    def _package_json(self, entry: MarketplaceEntry) -> bytes:
        """Package as JSON."""
        data = entry.to_dict()
        return json.dumps(data, indent=2).encode('utf-8')
    
    def _package_zip(self, entry: MarketplaceEntry) -> bytes:
        """Package as ZIP file."""
        with tempfile.NamedTemporaryFile() as temp_file:
            with zipfile.ZipFile(temp_file, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Add main algorithm file
                zf.writestr('algorithm.json', json.dumps(entry.to_dict(), indent=2))
                
                # Add documentation
                zf.writestr('README.md', entry.documentation)
                
                # Add examples
                for i, example in enumerate(entry.examples):
                    example_name = example.get('name', f'example_{i}')
                    zf.writestr(f'examples/{example_name}.py', example.get('code', ''))
                
                # Add test cases
                for i, test_case in enumerate(entry.test_cases):
                    test_name = test_case.get('name', f'test_{i}')
                    zf.writestr(f'tests/{test_name}.py', test_case.get('code', ''))
                
                # Add metadata
                zf.writestr('metadata.json', json.dumps(entry.metadata.to_dict(), indent=2))
            
            temp_file.seek(0)
            return temp_file.read()
    
    def _package_tar(self, entry: MarketplaceEntry) -> bytes:
        """Package as tar.gz file."""
        with tempfile.NamedTemporaryFile() as temp_file:
            with tarfile.open(temp_file.name, 'w:gz') as tf:
                # Add files using StringIO
                import io
                
                # Algorithm data
                algo_data = json.dumps(entry.to_dict(), indent=2).encode('utf-8')
                algo_info = tarfile.TarInfo(name='algorithm.json')
                algo_info.size = len(algo_data)
                tf.addfile(algo_info, io.BytesIO(algo_data))
                
                # Documentation
                doc_data = entry.documentation.encode('utf-8')
                doc_info = tarfile.TarInfo(name='README.md')
                doc_info.size = len(doc_data)
                tf.addfile(doc_info, io.BytesIO(doc_data))
            
            with open(temp_file.name, 'rb') as f:
                return f.read()
    
    def unpack_algorithm(self, package_data: bytes, format: str = 'zip') -> Optional[MarketplaceEntry]:
        """Unpack algorithm from package."""
        try:
            if format == 'json':
                data = json.loads(package_data.decode('utf-8'))
                return MarketplaceEntry.from_dict(data)
            
            elif format == 'zip':
                with tempfile.NamedTemporaryFile() as temp_file:
                    temp_file.write(package_data)
                    temp_file.flush()
                    
                    with zipfile.ZipFile(temp_file.name, 'r') as zf:
                        if 'algorithm.json' in zf.namelist():
                            algo_data = json.loads(zf.read('algorithm.json').decode('utf-8'))
                            return MarketplaceEntry.from_dict(algo_data)
            
            return None
            
        except Exception as e:
            logging.error(f"Unpacking failed: {e}")
            return None


class MarketplaceAPI:
    """RESTful API for the quantum algorithm marketplace."""
    
    def __init__(self, database: AlgorithmMarketplaceDB, port: int = 8766):
        self.database = database
        self.port = port
        self.validator = AlgorithmValidator()
        self.packager = AlgorithmPackager()
        self.server = None
        self.server_thread = None
    
    def start_server(self) -> bool:
        """Start the marketplace API server."""
        try:
            handler = self._create_request_handler()
            self.server = socketserver.TCPServer(("localhost", self.port), handler)
            
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to start marketplace API: {e}")
            return False
    
    def stop_server(self):
        """Stop the marketplace API server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        
        if self.server_thread:
            self.server_thread.join(timeout=1.0)
            self.server_thread = None
    
    def _create_request_handler(self):
        """Create HTTP request handler class."""
        database = self.database
        validator = self.validator
        packager = self.packager
        
        class MarketplaceRequestHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                """Handle GET requests."""
                try:
                    parsed_url = urllib.parse.urlparse(self.path)
                    path = parsed_url.path
                    query_params = urllib.parse.parse_qs(parsed_url.query)
                    
                    if path == '/api/algorithms':
                        self._handle_search_algorithms(query_params)
                    elif path.startswith('/api/algorithms/'):
                        algorithm_id = path.split('/')[-1]
                        self._handle_get_algorithm(algorithm_id)
                    elif path == '/api/featured':
                        self._handle_get_featured()
                    elif path == '/api/popular':
                        self._handle_get_popular()
                    elif path.startswith('/api/download/'):
                        algorithm_id = path.split('/')[-1]
                        format_param = query_params.get('format', ['zip'])[0]
                        self._handle_download_algorithm(algorithm_id, format_param)
                    else:
                        self._send_error(404, "Not found")
                
                except Exception as e:
                    self._send_error(500, str(e))
            
            def do_POST(self):
                """Handle POST requests."""
                try:
                    content_length = int(self.headers['Content-Length'])
                    request_data = json.loads(self.rfile.read(content_length).decode('utf-8'))
                    
                    parsed_url = urllib.parse.urlparse(self.path)
                    path = parsed_url.path
                    
                    if path == '/api/submit':
                        self._handle_submit_algorithm(request_data)
                    elif path == '/api/rate':
                        self._handle_rate_algorithm(request_data)
                    elif path == '/api/validate':
                        self._handle_validate_algorithm(request_data)
                    else:
                        self._send_error(404, "Not found")
                
                except Exception as e:
                    self._send_error(500, str(e))
            
            def _handle_search_algorithms(self, query_params):
                """Handle algorithm search."""
                query = query_params.get('q', [''])[0]
                category_str = query_params.get('category', [None])[0]
                type_str = query_params.get('type', [None])[0]
                author = query_params.get('author', [None])[0]
                min_rating_str = query_params.get('min_rating', [None])[0]
                limit = int(query_params.get('limit', ['20'])[0])
                offset = int(query_params.get('offset', ['0'])[0])
                
                # Parse enum values
                category = AlgorithmCategory(category_str) if category_str else None
                algorithm_type = AlgorithmType(type_str) if type_str else None
                min_rating = float(min_rating_str) if min_rating_str else None
                
                # Search algorithms
                algorithms = database.search_algorithms(
                    query=query,
                    category=category,
                    algorithm_type=algorithm_type,
                    author=author,
                    min_rating=min_rating,
                    limit=limit,
                    offset=offset
                )
                
                result = {
                    'algorithms': [algo.to_dict() for algo in algorithms],
                    'count': len(algorithms),
                    'limit': limit,
                    'offset': offset
                }
                
                self._send_json_response(result)
            
            def _handle_get_algorithm(self, algorithm_id):
                """Handle get specific algorithm."""
                algorithm = database.get_algorithm(algorithm_id)
                
                if algorithm:
                    self._send_json_response(algorithm.to_dict())
                else:
                    self._send_error(404, "Algorithm not found")
            
            def _handle_get_featured(self):
                """Handle get featured algorithms."""
                algorithms = database.get_featured_algorithms()
                result = {
                    'algorithms': [algo.to_dict() for algo in algorithms]
                }
                self._send_json_response(result)
            
            def _handle_get_popular(self):
                """Handle get popular algorithms."""
                algorithms = database.get_popular_algorithms()
                result = {
                    'algorithms': [algo.to_dict() for algo in algorithms]
                }
                self._send_json_response(result)
            
            def _handle_download_algorithm(self, algorithm_id, format):
                """Handle algorithm download."""
                algorithm = database.get_algorithm(algorithm_id)
                
                if not algorithm:
                    self._send_error(404, "Algorithm not found")
                    return
                
                # Package algorithm
                package_data = packager.package_algorithm(algorithm, format)
                
                if package_data:
                    # Record download
                    database.record_download(algorithm_id)
                    
                    # Send package
                    self.send_response(200)
                    
                    if format == 'json':
                        self.send_header('Content-Type', 'application/json')
                        filename = f"{algorithm.metadata.name}.json"
                    elif format == 'zip':
                        self.send_header('Content-Type', 'application/zip')
                        filename = f"{algorithm.metadata.name}.zip"
                    elif format == 'tar.gz':
                        self.send_header('Content-Type', 'application/gzip')
                        filename = f"{algorithm.metadata.name}.tar.gz"
                    
                    self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
                    self.send_header('Content-Length', str(len(package_data)))
                    self.end_headers()
                    
                    self.wfile.write(package_data)
                else:
                    self._send_error(500, "Packaging failed")
            
            def _handle_submit_algorithm(self, request_data):
                """Handle algorithm submission."""
                try:
                    entry = MarketplaceEntry.from_dict(request_data)
                    
                    # Validate entry
                    is_valid, errors = validator.validate_entry(entry)
                    
                    if not is_valid:
                        self._send_json_response({
                            'success': False,
                            'errors': errors
                        }, status=400)
                        return
                    
                    # Set status to pending review
                    entry.status = MarketplaceStatus.PENDING_REVIEW
                    
                    # Submit to database
                    success = database.submit_algorithm(entry)
                    
                    if success:
                        quality_score = validator.estimate_quality_score(entry)
                        self._send_json_response({
                            'success': True,
                            'entry_id': entry.entry_id,
                            'status': entry.status.value,
                            'quality_score': quality_score
                        })
                    else:
                        self._send_error(500, "Submission failed")
                
                except Exception as e:
                    self._send_error(400, f"Invalid request: {str(e)}")
            
            def _handle_rate_algorithm(self, request_data):
                """Handle algorithm rating."""
                try:
                    rating = AlgorithmRating.from_dict(request_data)
                    
                    # Validate rating
                    if not (1 <= rating.overall_rating <= 5):
                        self._send_error(400, "Rating must be between 1 and 5")
                        return
                    
                    # Add to database
                    success = database.add_rating(rating)
                    
                    if success:
                        self._send_json_response({'success': True})
                    else:
                        self._send_error(500, "Rating submission failed")
                
                except Exception as e:
                    self._send_error(400, f"Invalid rating: {str(e)}")
            
            def _handle_validate_algorithm(self, request_data):
                """Handle algorithm validation."""
                try:
                    entry = MarketplaceEntry.from_dict(request_data)
                    
                    is_valid, errors = validator.validate_entry(entry)
                    quality_score = validator.estimate_quality_score(entry)
                    
                    self._send_json_response({
                        'valid': is_valid,
                        'errors': errors,
                        'quality_score': quality_score
                    })
                
                except Exception as e:
                    self._send_error(400, f"Invalid request: {str(e)}")
            
            def _send_json_response(self, data, status=200):
                """Send JSON response."""
                response_json = json.dumps(data, indent=2)
                
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(response_json)))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                self.wfile.write(response_json.encode('utf-8'))
            
            def _send_error(self, status, message):
                """Send error response."""
                error_data = {'error': message}
                self._send_json_response(error_data, status)
            
            def log_message(self, format, *args):
                """Suppress default logging."""
                pass
        
        return MarketplaceRequestHandler


class QuantumAlgorithmMarketplace:
    """Main marketplace interface."""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path.home() / ".quantrs2" / "marketplace.db")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.database = AlgorithmMarketplaceDB(db_path)
        self.validator = AlgorithmValidator()
        self.packager = AlgorithmPackager()
        self.api = MarketplaceAPI(self.database)
        
        # Populate with example algorithms
        self._populate_examples()
    
    def _populate_examples(self):
        """Populate marketplace with example algorithms."""
        examples = [
            self._create_bell_state_algorithm(),
            self._create_grover_search_algorithm(),
            self._create_vqe_algorithm()
        ]
        
        for example in examples:
            if not self.database.get_algorithm(example.entry_id):
                example.status = MarketplaceStatus.APPROVED
                self.database.submit_algorithm(example)
    
    def _create_bell_state_algorithm(self) -> MarketplaceEntry:
        """Create Bell state algorithm example."""
        metadata = AlgorithmMetadata(
            algorithm_id="bell_state_v1",
            name="Bell State Preparation",
            description="Prepares a maximally entangled Bell state between two qubits using Hadamard and CNOT gates.",
            author="QuantRS2 Team",
            version="1.0.0",
            category=AlgorithmCategory.GENERAL,
            algorithm_type=AlgorithmType.CIRCUIT,
            license=LicenseType.MIT,
            tags=["entanglement", "bell_state", "tutorial"],
            min_qubits=2,
            max_qubits=2,
            complexity_class="P",
            quantum_advantage="Demonstrates quantum entanglement"
        )
        
        algorithm_data = {
            "gates": [
                {"gate": "h", "qubits": [0]},
                {"gate": "cnot", "qubits": [0, 1]}
            ],
            "qasm": """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cnot q[0],q[1];
measure q -> c;
"""
        }
        
        documentation = """
# Bell State Preparation

## Description
This algorithm prepares a maximally entangled Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2 between two qubits.

## Usage
```python
import quantrs2
circuit = quantrs2.PyCircuit(2)
circuit.h(0)      # Apply Hadamard to qubit 0
circuit.cnot(0, 1) # Apply CNOT with qubit 0 as control
result = circuit.run()
```

## Parameters
- Qubits: 2 (exactly)
- Gates: Hadamard (H), Controlled-NOT (CNOT)

## Output
The resulting state will be in superposition with equal probability of measuring |00⟩ or |11⟩.
"""
        
        examples = [
            {
                "name": "Basic Bell State",
                "description": "Create and measure Bell state",
                "code": """
import quantrs2

# Create Bell state
circuit = quantrs2.PyCircuit(2)
circuit.h(0)
circuit.cnot(0, 1)

# Run simulation
result = circuit.run()
print("State probabilities:", result.state_probabilities())
""",
                "expected_output": {"00": 0.5, "11": 0.5}
            }
        ]
        
        test_cases = [
            {
                "name": "Bell State Verification",
                "description": "Verify Bell state preparation",
                "input": {"n_qubits": 2},
                "expected_output": {"00": 0.5, "11": 0.5},
                "tolerance": 0.01
            }
        ]
        
        return MarketplaceEntry(
            entry_id="bell_state_v1",
            metadata=metadata,
            algorithm_data=algorithm_data,
            documentation=documentation,
            examples=examples,
            test_cases=test_cases
        )
    
    def _create_grover_search_algorithm(self) -> MarketplaceEntry:
        """Create Grover's search algorithm example."""
        metadata = AlgorithmMetadata(
            algorithm_id="grover_search_v1",
            name="Grover's Search Algorithm",
            description="Quantum search algorithm that finds marked items in an unsorted database with quadratic speedup.",
            author="QuantRS2 Team",
            version="1.0.0",
            category=AlgorithmCategory.SEARCH,
            algorithm_type=AlgorithmType.CIRCUIT,
            license=LicenseType.MIT,
            tags=["grover", "search", "quantum_speedup"],
            min_qubits=2,
            max_qubits=10,
            complexity_class="BQP",
            quantum_advantage="Quadratic speedup over classical search"
        )
        
        algorithm_data = {
            "gates": [
                {"gate": "h", "qubits": [0]},
                {"gate": "h", "qubits": [1]},
                {"gate": "cz", "qubits": [0, 1]},  # Oracle
                {"gate": "h", "qubits": [0]},
                {"gate": "h", "qubits": [1]},
                {"gate": "x", "qubits": [0]},
                {"gate": "x", "qubits": [1]},
                {"gate": "cz", "qubits": [0, 1]},  # Diffuser
                {"gate": "x", "qubits": [0]},
                {"gate": "x", "qubits": [1]},
                {"gate": "h", "qubits": [0]},
                {"gate": "h", "qubits": [1]}
            ],
            "parameters": {
                "target_state": "11",
                "iterations": 1
            }
        }
        
        documentation = """
# Grover's Search Algorithm

## Description
Grover's algorithm provides a quadratic speedup for searching unsorted databases.
For N items, classical search requires O(N) operations, while Grover requires O(√N).

## Usage
```python
import quantrs2

# 2-qubit Grover search for state |11⟩
circuit = quantrs2.PyCircuit(2)

# Initialize superposition
circuit.h(0)
circuit.h(1)

# Grover iteration
circuit.cz(0, 1)  # Oracle marks |11⟩
circuit.h(0)
circuit.h(1)
circuit.x(0)
circuit.x(1)
circuit.cz(0, 1)  # Diffuser
circuit.x(0)
circuit.x(1)
circuit.h(0)
circuit.h(1)

result = circuit.run()
```

## Parameters
- target_state: The computational basis state to search for
- iterations: Number of Grover iterations (optimal is π√N/4)

## Output
High probability of measuring the target state.
"""
        
        examples = [
            {
                "name": "2-Qubit Grover Search",
                "description": "Search for |11⟩ state in 2-qubit space",
                "code": """
import quantrs2

circuit = quantrs2.PyCircuit(2)
# Grover implementation for 2 qubits
circuit.h(0)
circuit.h(1)
circuit.cz(0, 1)
circuit.h(0)
circuit.h(1)
circuit.x(0)
circuit.x(1)
circuit.cz(0, 1)
circuit.x(0)
circuit.x(1)
circuit.h(0)
circuit.h(1)

result = circuit.run()
print("Probabilities:", result.state_probabilities())
""",
                "expected_output": {"11": 1.0}
            }
        ]
        
        test_cases = [
            {
                "name": "Grover 2-Qubit Test",
                "description": "Verify Grover finds target state",
                "input": {"n_qubits": 2, "target": "11"},
                "expected_output": {"11": 1.0},
                "tolerance": 0.1
            }
        ]
        
        return MarketplaceEntry(
            entry_id="grover_search_v1",
            metadata=metadata,
            algorithm_data=algorithm_data,
            documentation=documentation,
            examples=examples,
            test_cases=test_cases
        )
    
    def _create_vqe_algorithm(self) -> MarketplaceEntry:
        """Create VQE algorithm example."""
        metadata = AlgorithmMetadata(
            algorithm_id="vqe_h2_v1",
            name="VQE for H2 Molecule",
            description="Variational Quantum Eigensolver for finding ground state energy of hydrogen molecule.",
            author="QuantRS2 Team",
            version="1.0.0",
            category=AlgorithmCategory.CHEMISTRY,
            algorithm_type=AlgorithmType.VARIATIONAL,
            license=LicenseType.MIT,
            tags=["vqe", "chemistry", "optimization", "variational"],
            min_qubits=4,
            max_qubits=4,
            complexity_class="BQP",
            quantum_advantage="Exponential advantage for quantum chemistry"
        )
        
        algorithm_data = {
            "ansatz": "UCCSD",
            "parameters": ["theta1", "theta2"],
            "hamiltonian": {
                "terms": [
                    {"pauli": "ZIII", "coeff": -1.0523732},
                    {"pauli": "IZII", "coeff": -1.0523732},
                    {"pauli": "IIZI", "coeff": -0.39793742},
                    {"pauli": "IIIZ", "coeff": -0.39793742}
                ]
            },
            "gates": [
                {"gate": "ry", "qubits": [0], "params": ["theta1"]},
                {"gate": "ry", "qubits": [1], "params": ["theta2"]},
                {"gate": "cnot", "qubits": [0, 1]},
                {"gate": "cnot", "qubits": [2, 3]}
            ]
        }
        
        documentation = """
# Variational Quantum Eigensolver (VQE) for H2

## Description
VQE is a hybrid quantum-classical algorithm for finding ground state energies of quantum systems.
This implementation targets the hydrogen molecule (H2).

## Usage
```python
import quantrs2

# VQE for H2 molecule
vqe = quantrs2.VQE(hamiltonian=h2_hamiltonian)
result = vqe.optimize(initial_params=[0.0, 0.0])
print(f"Ground state energy: {result.energy}")
```

## Parameters
- hamiltonian: Molecular Hamiltonian in Pauli operator form
- ansatz: Parameterized quantum circuit (UCCSD recommended)
- optimizer: Classical optimization method

## Output
Ground state energy and optimal parameters.
"""
        
        examples = [
            {
                "name": "H2 Ground State",
                "description": "Find H2 molecule ground state",
                "code": """
import quantrs2

# Create VQE instance
vqe = quantrs2.VQE(n_qubits=4)

# Define H2 Hamiltonian (simplified)
h2_terms = [
    ("ZIII", -1.0523732),
    ("IZII", -1.0523732), 
    ("IIZI", -0.39793742),
    ("IIIZ", -0.39793742)
]

# Run optimization
result = vqe.run_optimization(h2_terms)
print(f"Energy: {result['energy']}")
""",
                "expected_output": {"energy": -1.85}
            }
        ]
        
        test_cases = [
            {
                "name": "VQE H2 Test",
                "description": "Verify VQE finds correct energy",
                "input": {"molecule": "H2", "basis": "STO-3G"},
                "expected_output": {"energy": -1.8572750301938273},
                "tolerance": 0.1
            }
        ]
        
        return MarketplaceEntry(
            entry_id="vqe_h2_v1",
            metadata=metadata,
            algorithm_data=algorithm_data,
            documentation=documentation,
            examples=examples,
            test_cases=test_cases
        )
    
    def submit_algorithm(self, entry: MarketplaceEntry) -> Tuple[bool, List[str]]:
        """Submit algorithm to marketplace."""
        # Validate entry
        is_valid, errors = self.validator.validate_entry(entry)
        
        if not is_valid:
            return False, errors
        
        # Set metadata
        entry.metadata.updated_at = time.time()
        entry.status = MarketplaceStatus.PENDING_REVIEW
        
        # Submit to database
        success = self.database.submit_algorithm(entry)
        
        if success:
            return True, []
        else:
            return False, ["Database submission failed"]
    
    def search_algorithms(self, **kwargs) -> List[MarketplaceEntry]:
        """Search algorithms in marketplace."""
        return self.database.search_algorithms(**kwargs)
    
    def get_algorithm(self, algorithm_id: str) -> Optional[MarketplaceEntry]:
        """Get specific algorithm."""
        return self.database.get_algorithm(algorithm_id)
    
    def download_algorithm(self, algorithm_id: str, format: str = 'json') -> Optional[bytes]:
        """Download algorithm package."""
        algorithm = self.database.get_algorithm(algorithm_id)
        
        if not algorithm:
            return None
        
        # Record download
        self.database.record_download(algorithm_id)
        
        # Package and return
        return self.packager.package_algorithm(algorithm, format)
    
    def rate_algorithm(self, rating: AlgorithmRating) -> bool:
        """Rate an algorithm."""
        return self.database.add_rating(rating)
    
    def get_featured_algorithms(self) -> List[MarketplaceEntry]:
        """Get featured algorithms."""
        return self.database.get_featured_algorithms()
    
    def get_popular_algorithms(self) -> List[MarketplaceEntry]:
        """Get popular algorithms."""
        return self.database.get_popular_algorithms()
    
    def start_api_server(self) -> bool:
        """Start marketplace API server."""
        return self.api.start_server()
    
    def stop_api_server(self):
        """Stop marketplace API server."""
        self.api.stop_server()
    
    def get_marketplace_stats(self) -> Dict[str, Any]:
        """Get marketplace statistics."""
        try:
            cursor = self.database.connection.execute("""
                SELECT 
                    COUNT(*) as total_algorithms,
                    SUM(download_count) as total_downloads,
                    AVG(average_rating) as overall_rating,
                    COUNT(DISTINCT json_extract(metadata, '$.author')) as total_authors
                FROM algorithms 
                WHERE status = 'approved'
            """)
            
            stats = cursor.fetchone()
            
            # Category breakdown
            cursor = self.database.connection.execute("""
                SELECT 
                    json_extract(metadata, '$.category') as category,
                    COUNT(*) as count
                FROM algorithms 
                WHERE status = 'approved'
                GROUP BY category
            """)
            
            categories = {row['category']: row['count'] for row in cursor.fetchall()}
            
            return {
                'total_algorithms': stats['total_algorithms'] or 0,
                'total_downloads': stats['total_downloads'] or 0,
                'overall_rating': stats['overall_rating'] or 0.0,
                'total_authors': stats['total_authors'] or 0,
                'categories': categories
            }
            
        except Exception as e:
            logging.error(f"Failed to get stats: {e}")
            return {}
    
    def close(self):
        """Close marketplace."""
        self.api.stop_server()
        self.database.close()


# Global marketplace instance
_quantum_marketplace: Optional[QuantumAlgorithmMarketplace] = None


def get_quantum_marketplace() -> QuantumAlgorithmMarketplace:
    """Get global quantum algorithm marketplace instance."""
    global _quantum_marketplace
    if _quantum_marketplace is None:
        _quantum_marketplace = QuantumAlgorithmMarketplace()
    return _quantum_marketplace


def search_algorithms(query: str = "", **kwargs) -> List[MarketplaceEntry]:
    """Convenience function to search algorithms."""
    marketplace = get_quantum_marketplace()
    return marketplace.search_algorithms(query=query, **kwargs)


def download_algorithm(algorithm_id: str, format: str = 'json') -> Optional[bytes]:
    """Convenience function to download algorithm."""
    marketplace = get_quantum_marketplace()
    return marketplace.download_algorithm(algorithm_id, format)


def submit_algorithm(entry: MarketplaceEntry) -> Tuple[bool, List[str]]:
    """Convenience function to submit algorithm."""
    marketplace = get_quantum_marketplace()
    return marketplace.submit_algorithm(entry)


def create_algorithm_entry(name: str, description: str, author: str, 
                          algorithm_data: Dict[str, Any], **kwargs) -> MarketplaceEntry:
    """Convenience function to create algorithm entry."""
    algorithm_id = f"{name.lower().replace(' ', '_')}_{int(time.time())}"
    
    metadata = AlgorithmMetadata(
        algorithm_id=algorithm_id,
        name=name,
        description=description,
        author=author,
        version="1.0.0",
        category=kwargs.get('category', AlgorithmCategory.GENERAL),
        algorithm_type=kwargs.get('algorithm_type', AlgorithmType.CIRCUIT),
        license=kwargs.get('license', LicenseType.MIT),
        **{k: v for k, v in kwargs.items() if k in ['tags', 'min_qubits', 'max_qubits']}
    )
    
    return MarketplaceEntry(
        entry_id=algorithm_id,
        metadata=metadata,
        algorithm_data=algorithm_data,
        documentation=kwargs.get('documentation', description),
        examples=kwargs.get('examples', []),
        test_cases=kwargs.get('test_cases', [])
    )


# CLI interface
def main():
    """Main CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="QuantRS2 Algorithm Marketplace")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search algorithms')
    search_parser.add_argument('query', nargs='?', default='', help='Search query')
    search_parser.add_argument('--category', choices=[c.value for c in AlgorithmCategory], help='Filter by category')
    search_parser.add_argument('--author', help='Filter by author')
    search_parser.add_argument('--limit', type=int, default=10, help='Limit results')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download algorithm')
    download_parser.add_argument('algorithm_id', help='Algorithm ID to download')
    download_parser.add_argument('--format', choices=['json', 'zip', 'tar.gz'], default='json', help='Download format')
    download_parser.add_argument('--output', help='Output file path')
    
    # Stats command
    subparsers.add_parser('stats', help='Show marketplace statistics')
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Start marketplace API server')
    server_parser.add_argument('--port', type=int, default=8766, help='Server port')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    marketplace = get_quantum_marketplace()
    
    try:
        if args.command == 'search':
            category = AlgorithmCategory(args.category) if args.category else None
            algorithms = marketplace.search_algorithms(
                query=args.query,
                category=category,
                author=args.author,
                limit=args.limit
            )
            
            print(f"Found {len(algorithms)} algorithms:")
            for algo in algorithms:
                print(f"  {algo.metadata.name} ({algo.entry_id})")
                print(f"    Author: {algo.metadata.author}")
                print(f"    Category: {algo.metadata.category.value}")
                print(f"    Rating: {algo.average_rating:.1f} ({algo.rating_count} reviews)")
                print(f"    Downloads: {algo.download_count}")
                print()
        
        elif args.command == 'download':
            package_data = marketplace.download_algorithm(args.algorithm_id, args.format)
            
            if package_data:
                if args.output:
                    output_path = Path(args.output)
                else:
                    ext = 'json' if args.format == 'json' else args.format
                    output_path = Path(f"{args.algorithm_id}.{ext}")
                
                with open(output_path, 'wb') as f:
                    f.write(package_data)
                
                print(f"Downloaded to: {output_path}")
            else:
                print("Algorithm not found or download failed")
                return 1
        
        elif args.command == 'stats':
            stats = marketplace.get_marketplace_stats()
            
            print("Marketplace Statistics:")
            print(f"  Total algorithms: {stats.get('total_algorithms', 0)}")
            print(f"  Total downloads: {stats.get('total_downloads', 0)}")
            print(f"  Overall rating: {stats.get('overall_rating', 0):.1f}")
            print(f"  Total authors: {stats.get('total_authors', 0)}")
            
            print("\nCategories:")
            for category, count in stats.get('categories', {}).items():
                print(f"  {category}: {count}")
        
        elif args.command == 'server':
            marketplace.api.port = args.port
            success = marketplace.start_api_server()
            
            if success:
                print(f"Marketplace API server started on port {args.port}")
                print("Available endpoints:")
                print("  GET  /api/algorithms - Search algorithms")
                print("  GET  /api/algorithms/{id} - Get specific algorithm")
                print("  GET  /api/featured - Get featured algorithms")
                print("  GET  /api/popular - Get popular algorithms")
                print("  GET  /api/download/{id} - Download algorithm")
                print("  POST /api/submit - Submit algorithm")
                print("  POST /api/rate - Rate algorithm")
                print("  POST /api/validate - Validate algorithm")
                print("\nPress Ctrl+C to stop")
                
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\nStopping server...")
                    marketplace.stop_api_server()
                    return 0
            else:
                print("Failed to start server")
                return 1
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    finally:
        marketplace.close()


if __name__ == "__main__":
    exit(main())