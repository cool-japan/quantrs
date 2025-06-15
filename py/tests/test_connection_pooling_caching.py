"""
Tests for Connection Pooling and Caching System

This module tests the comprehensive connection pooling, caching strategies,
and performance optimization features.
"""

import pytest
import time
import threading
import tempfile
import sqlite3
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from quantrs2.connection_pooling import (
    DatabaseConnectionPool, QuantumResultCache, 
    ConnectionPoolConfig, CacheConfig, CacheBackend, CacheStrategy
)
from quantrs2.circuit_optimization_cache import (
    CircuitOptimizationCache, CircuitPatternDetector, 
    OptimizationLevel, CircuitPattern, CircuitSignature
)
from quantrs2.performance_manager import (
    PerformanceManager, PerformanceConfig, PerformanceProfile,
    ConnectionManager, CacheManager, get_performance_manager
)


class TestDatabaseConnectionPool:
    """Test database connection pooling functionality."""
    
    def test_pool_creation(self):
        """Test creating a connection pool."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            config = ConnectionPoolConfig(max_connections=5, min_connections=2)
            pool = DatabaseConnectionPool(tmp.name, config)
            
            try:
                assert pool.config.max_connections == 5
                assert pool.config.min_connections == 2
                
                # Check that minimum connections were created
                stats = pool.get_statistics()
                assert stats['total_connections'] >= 2
                
            finally:
                pool.close()
    
    def test_connection_borrowing_and_returning(self):
        """Test borrowing and returning connections."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            config = ConnectionPoolConfig(max_connections=3, min_connections=1)
            pool = DatabaseConnectionPool(tmp.name, config)
            
            try:
                # Test context manager
                with pool.get_connection() as conn:
                    assert isinstance(conn, sqlite3.Connection)
                    
                    # Execute a simple query
                    cursor = conn.execute("SELECT 1")
                    result = cursor.fetchone()
                    assert result[0] == 1
                
                # Check statistics
                stats = pool.get_statistics()
                assert stats['connections_borrowed'] >= 1
                assert stats['connections_returned'] >= 1
                
            finally:
                pool.close()
    
    def test_pool_exhaustion(self):
        """Test behavior when pool is exhausted."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            config = ConnectionPoolConfig(max_connections=2, min_connections=1)
            pool = DatabaseConnectionPool(tmp.name, config)
            
            try:
                connections = []
                
                # Borrow all available connections
                for _ in range(2):
                    conn = pool._borrow_connection()
                    connections.append(conn)
                
                # Next borrow should fail
                with pytest.raises(RuntimeError, match="exhausted"):
                    pool._borrow_connection()
                
                # Return connections
                for conn in connections:
                    pool._return_connection(conn)
                
            finally:
                pool.close()
    
    def test_connection_validation(self):
        """Test connection validation."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            config = ConnectionPoolConfig(validation_query="SELECT 42")
            pool = DatabaseConnectionPool(tmp.name, config)
            
            try:
                # Get a connection and validate it
                conn = pool._borrow_connection()
                assert pool._validate_connection(conn)
                
                # Close the connection to make it invalid
                conn.close()
                assert not pool._validate_connection(conn)
                
            finally:
                pool.close()


class TestQuantumResultCache:
    """Test quantum result caching functionality."""
    
    def test_memory_cache_basic_operations(self):
        """Test basic cache operations with memory backend."""
        config = CacheConfig(backend=CacheBackend.MEMORY, max_memory_mb=10)
        cache = QuantumResultCache(config)
        
        try:
            # Test put and get
            test_data = {"result": "bell_state", "probabilities": {"00": 0.5, "11": 0.5}}
            success = cache.put("test_key", test_data, ttl=60)
            assert success
            
            retrieved = cache.get("test_key")
            assert retrieved == test_data
            
            # Test cache miss
            missing = cache.get("nonexistent_key")
            assert missing is None
            
            # Test statistics
            stats = cache.get_statistics()
            assert stats['hits'] >= 1
            assert stats['misses'] >= 1
            
        finally:
            cache.close()
    
    def test_cache_expiration(self):
        """Test cache entry expiration."""
        config = CacheConfig(backend=CacheBackend.MEMORY, default_ttl=0.1)
        cache = QuantumResultCache(config)
        
        try:
            # Put with short TTL
            cache.put("expire_test", "data", ttl=0.1)
            
            # Should be available immediately
            assert cache.get("expire_test") == "data"
            
            # Wait for expiration
            time.sleep(0.2)
            
            # Should be expired now
            assert cache.get("expire_test") is None
            
        finally:
            cache.close()
    
    def test_cache_eviction(self):
        """Test cache eviction when limits are exceeded."""
        config = CacheConfig(
            backend=CacheBackend.MEMORY, 
            max_entries=3,
            strategy=CacheStrategy.LRU
        )
        cache = QuantumResultCache(config)
        
        try:
            # Fill cache to capacity
            for i in range(3):
                cache.put(f"key_{i}", f"data_{i}")
            
            # All should be retrievable
            for i in range(3):
                assert cache.get(f"key_{i}") == f"data_{i}"
            
            # Add one more to trigger eviction
            cache.put("key_3", "data_3")
            
            # The least recently used (key_0) should be evicted
            assert cache.get("key_0") is None
            assert cache.get("key_3") == "data_3"
            
        finally:
            cache.close()
    
    def test_cache_invalidation(self):
        """Test cache invalidation by key and tags."""
        config = CacheConfig(backend=CacheBackend.MEMORY)
        cache = QuantumResultCache(config)
        
        try:
            # Put entries with tags
            cache.put("key1", "data1", tags=["pattern:bell", "qubits:2"])
            cache.put("key2", "data2", tags=["pattern:ghz", "qubits:3"])
            cache.put("key3", "data3", tags=["pattern:bell", "qubits:4"])
            
            # Test single key invalidation
            assert cache.invalidate("key1")
            assert cache.get("key1") is None
            assert cache.get("key2") == "data2"
            
            # Test tag-based invalidation
            invalidated = cache.invalidate_by_tags(["pattern:bell"])
            assert invalidated >= 1  # Should invalidate key3
            assert cache.get("key3") is None
            assert cache.get("key2") == "data2"  # Should remain
            
        finally:
            cache.close()
    
    @pytest.mark.skipif(not hasattr(sqlite3, 'connect'), reason="SQLite not available")
    def test_sqlite_cache_backend(self):
        """Test SQLite cache backend."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            config = CacheConfig(
                backend=CacheBackend.SQLITE,
                sqlite_path=tmp.name
            )
            cache = QuantumResultCache(config)
            
            try:
                # Test operations
                cache.put("sqlite_test", {"data": "test"})
                retrieved = cache.get("sqlite_test")
                assert retrieved == {"data": "test"}
                
            finally:
                cache.close()


class TestCircuitPatternDetector:
    """Test circuit pattern detection."""
    
    def test_bell_state_detection(self):
        """Test Bell state pattern detection."""
        detector = CircuitPatternDetector()
        
        # Mock Bell state circuit
        circuit = Mock()
        circuit.data = [
            Mock(instruction=Mock(name='h')),
            Mock(instruction=Mock(name='cnot'))
        ]
        circuit.data[0][0] = Mock()
        circuit.data[0][0].name = 'h'
        circuit.data[1][0] = Mock() 
        circuit.data[1][0].name = 'cnot'
        
        pattern, confidence = detector.detect_pattern(circuit)
        
        # Should detect Bell state pattern with reasonable confidence
        assert pattern == CircuitPattern.BELL_STATE
        assert confidence > 0.5
    
    def test_custom_pattern_fallback(self):
        """Test fallback to custom pattern for unknown circuits."""
        detector = CircuitPatternDetector()
        
        # Mock unknown circuit
        circuit = Mock()
        circuit.data = [
            Mock(instruction=Mock(name='unknown_gate'))
        ]
        circuit.data[0][0] = Mock()
        circuit.data[0][0].name = 'unknown_gate'
        
        pattern, confidence = detector.detect_pattern(circuit)
        
        # Should fall back to custom pattern
        assert pattern == CircuitPattern.CUSTOM
        assert confidence >= 0.0


class TestCircuitOptimizationCache:
    """Test circuit optimization caching."""
    
    def test_circuit_signature_computation(self):
        """Test circuit signature computation."""
        config = CacheConfig(backend=CacheBackend.MEMORY)
        cache = CircuitOptimizationCache(config)
        
        try:
            # Mock circuit
            circuit = Mock()
            circuit.num_qubits = 2
            circuit.size = Mock(return_value=4)
            circuit.depth = Mock(return_value=2)
            
            signature = cache.compute_circuit_signature(circuit)
            
            assert signature.qubit_count == 2
            assert signature.gate_count == 4
            assert signature.depth == 2
            assert isinstance(signature.gate_sequence_hash, str)
            
        finally:
            cache.close()
    
    def test_execution_result_caching(self):
        """Test caching of execution results."""
        config = CacheConfig(backend=CacheBackend.MEMORY)
        cache = CircuitOptimizationCache(config)
        
        try:
            # Mock circuit and execution
            circuit = Mock()
            circuit.num_qubits = 2
            circuit.size = Mock(return_value=2)
            circuit.depth = Mock(return_value=1)
            
            execution_config = {"backend": "simulator", "shots": 1000}
            result = {"00": 0.5, "11": 0.5}
            
            # Cache result
            success = cache.cache_execution_result(
                circuit, execution_config, result, 
                execution_time=0.1, success=True
            )
            assert success
            
            # Retrieve cached result
            cached_result = cache.get_cached_result(circuit, execution_config)
            assert cached_result == result
            
        finally:
            cache.close()
    
    def test_optimization_result_caching(self):
        """Test caching of optimization results."""
        config = CacheConfig(backend=CacheBackend.MEMORY)
        cache = CircuitOptimizationCache(config)
        
        try:
            # Mock original and optimized circuits
            original_circuit = Mock()
            original_circuit.num_qubits = 3
            original_circuit.size = Mock(return_value=10)
            original_circuit.depth = Mock(return_value=5)
            
            optimized_circuit = Mock()
            optimized_circuit.num_qubits = 3
            optimized_circuit.size = Mock(return_value=8)
            optimized_circuit.depth = Mock(return_value=4)
            
            improvements = {
                "gate_count": {"original": 10, "optimized": 8, "ratio": 0.2},
                "depth": {"original": 5, "optimized": 4, "ratio": 0.2}
            }
            
            # Cache optimization
            success = cache.cache_optimization_result(
                original_circuit, optimized_circuit,
                OptimizationLevel.STANDARD, 0.5,
                ["dead_code_elimination", "gate_fusion"], improvements
            )
            assert success
            
            # Retrieve cached optimization
            cached_opt = cache.get_cached_optimization(original_circuit, OptimizationLevel.STANDARD)
            assert cached_opt is not None
            assert cached_opt.optimization_time == 0.5
            assert "dead_code_elimination" in cached_opt.applied_passes
            
        finally:
            cache.close()
    
    def test_execution_recommendations(self):
        """Test execution recommendations based on history."""
        config = CacheConfig(backend=CacheBackend.MEMORY)
        cache = CircuitOptimizationCache(config)
        
        try:
            # Mock circuit
            circuit = Mock()
            circuit.num_qubits = 2
            circuit.size = Mock(return_value=4)
            circuit.depth = Mock(return_value=2)
            
            # Get recommendations (should work even without history)
            recommendations = cache.get_execution_recommendations(circuit)
            
            assert 'detected_pattern' in recommendations
            assert 'pattern_confidence' in recommendations
            assert 'estimated_execution_time' in recommendations
            
        finally:
            cache.close()


class TestPerformanceManager:
    """Test unified performance management."""
    
    def test_performance_manager_creation(self):
        """Test creating performance manager with different profiles."""
        # Test development profile
        config = PerformanceConfig.for_profile(PerformanceProfile.DEVELOPMENT)
        manager = PerformanceManager(config)
        
        try:
            assert manager.config.profile == PerformanceProfile.DEVELOPMENT
            assert manager.config.max_db_connections == 5
            assert manager.config.cache_backend == CacheBackend.MEMORY
            
            # Test that managers are created
            assert manager.connection_manager is not None
            assert manager.cache_manager is not None
            
        finally:
            manager.close()
        
        # Test production profile
        config = PerformanceConfig.for_profile(PerformanceProfile.PRODUCTION)
        manager = PerformanceManager(config)
        
        try:
            assert manager.config.profile == PerformanceProfile.PRODUCTION
            assert manager.config.max_db_connections == 20
            assert manager.config.cache_backend == CacheBackend.HYBRID
            
        finally:
            manager.close()
    
    def test_database_connection_access(self):
        """Test database connection access through performance manager."""
        config = PerformanceConfig.for_profile(PerformanceProfile.TESTING)
        manager = PerformanceManager(config)
        
        try:
            # Test connection context manager
            with manager.database_connection('circuits') as conn:
                assert isinstance(conn, sqlite3.Connection)
                
                # Test basic query
                cursor = conn.execute("SELECT 1")
                result = cursor.fetchone()
                assert result[0] == 1
                
        finally:
            manager.close()
    
    def test_cache_access(self):
        """Test cache access through performance manager."""
        config = PerformanceConfig.for_profile(PerformanceProfile.TESTING)
        manager = PerformanceManager(config)
        
        try:
            # Test circuit cache
            circuit_cache = manager.get_circuit_cache()
            assert isinstance(circuit_cache, CircuitOptimizationCache)
            
            # Test general cache
            general_cache = manager.get_general_cache()
            assert isinstance(general_cache, QuantumResultCache)
            
            # Test basic cache operations
            general_cache.put("test", "data")
            assert general_cache.get("test") == "data"
            
        finally:
            manager.close()
    
    def test_performance_report(self):
        """Test performance reporting."""
        config = PerformanceConfig.for_profile(PerformanceProfile.TESTING)
        config.enable_performance_monitoring = False  # Disable for testing
        manager = PerformanceManager(config)
        
        try:
            report = manager.get_performance_report()
            
            assert 'config' in report
            assert 'connections' in report
            assert 'cache' in report
            assert 'monitoring_enabled' in report
            
            # Check config section
            assert report['config']['profile'] == PerformanceProfile.TESTING.value
            assert report['config']['cache_backend'] == CacheBackend.MEMORY.value
            
        finally:
            manager.close()
    
    def test_performance_optimization(self):
        """Test performance optimization procedures."""
        config = PerformanceConfig.for_profile(PerformanceProfile.TESTING)
        config.enable_performance_monitoring = False
        manager = PerformanceManager(config)
        
        try:
            # Should not raise exceptions
            manager.optimize_performance()
            
        finally:
            manager.close()


class TestGlobalPerformanceManager:
    """Test global performance manager functionality."""
    
    def test_global_manager_singleton(self):
        """Test global manager behaves as singleton."""
        from quantrs2.performance_manager import close_performance_manager
        
        # Clean up any existing manager
        close_performance_manager()
        
        try:
            # Get managers should return same instance
            manager1 = get_performance_manager()
            manager2 = get_performance_manager()
            
            assert manager1 is manager2
            
        finally:
            close_performance_manager()
    
    def test_global_manager_cleanup(self):
        """Test global manager cleanup."""
        from quantrs2.performance_manager import close_performance_manager
        
        # Clean up any existing manager
        close_performance_manager()
        
        try:
            # Get a manager
            manager = get_performance_manager()
            assert manager is not None
            
            # Close it
            close_performance_manager()
            
            # Getting a new one should create a new instance
            new_manager = get_performance_manager()
            assert new_manager is not manager
            
        finally:
            close_performance_manager()


class TestIntegrationScenarios:
    """Test integrated usage scenarios."""
    
    def test_circuit_execution_with_caching(self):
        """Test complete circuit execution with caching workflow."""
        config = PerformanceConfig.for_profile(PerformanceProfile.TESTING)
        config.enable_performance_monitoring = False
        manager = PerformanceManager(config)
        
        try:
            circuit_cache = manager.get_circuit_cache()
            
            # Mock circuit execution workflow
            circuit = Mock()
            circuit.num_qubits = 3
            circuit.size = Mock(return_value=6)
            circuit.depth = Mock(return_value=3)
            
            execution_config = {"backend": "simulator", "shots": 1024}
            
            # First execution (cache miss)
            cached_result = circuit_cache.get_cached_result(circuit, execution_config)
            assert cached_result is None
            
            # Simulate execution and cache result
            execution_result = {"000": 0.125, "111": 0.875}
            success = circuit_cache.cache_execution_result(
                circuit, execution_config, execution_result,
                execution_time=0.25, success=True
            )
            assert success
            
            # Second execution (cache hit)
            cached_result = circuit_cache.get_cached_result(circuit, execution_config)
            assert cached_result == execution_result
            
            # Get recommendations
            recommendations = circuit_cache.get_execution_recommendations(circuit)
            assert 'detected_pattern' in recommendations
            
        finally:
            manager.close()
    
    def test_multi_threaded_access(self):
        """Test thread safety of connection pools and caches."""
        config = PerformanceConfig.for_profile(PerformanceProfile.TESTING)
        config.enable_performance_monitoring = False
        manager = PerformanceManager(config)
        
        results = []
        errors = []
        
        def worker():
            try:
                # Test database connection
                with manager.database_connection('circuits') as conn:
                    cursor = conn.execute("SELECT 1")
                    result = cursor.fetchone()[0]
                    results.append(result)
                
                # Test cache operations
                cache = manager.get_general_cache()
                cache.put(f"thread_{threading.current_thread().ident}", "data")
                retrieved = cache.get(f"thread_{threading.current_thread().ident}")
                results.append(retrieved)
                
            except Exception as e:
                errors.append(e)
        
        try:
            # Run multiple threads
            threads = []
            for _ in range(5):
                thread = threading.Thread(target=worker)
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join(timeout=5)
            
            # Check results
            assert len(errors) == 0, f"Errors occurred: {errors}"
            assert len(results) >= 10  # Each thread should produce 2 results
            
        finally:
            manager.close()


if __name__ == "__main__":
    pytest.main([__file__])