#!/usr/bin/env python3
"""
Test suite for quantum compilation service functionality.
"""

import pytest
import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

try:
    from quantrs2.compilation_service import (
        CompilationBackend, OptimizationLevel, CompilationStatus,
        CompilationRequest, CompilationResult, CompilationBackendInterface,
        LocalCompilationBackend, RemoteCompilationBackend, CompilationCache,
        CompilationService, CompilationServiceAPI, get_compilation_service,
        compile_circuit, start_compilation_api
    )
    HAS_COMPILATION_SERVICE = True
except ImportError:
    HAS_COMPILATION_SERVICE = False

try:
    import flask
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False


@pytest.mark.skipif(not HAS_COMPILATION_SERVICE, reason="compilation_service module not available")
class TestCompilationRequest:
    """Test CompilationRequest functionality."""
    
    def test_compilation_request_creation(self):
        """Test creating compilation request."""
        circuit_data = {
            'n_qubits': 3,
            'gates': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'cnot', 'qubits': [0, 1]}
            ]
        }
        
        request = CompilationRequest(
            request_id="test_request",
            circuit_data=circuit_data,
            optimization_level=OptimizationLevel.STANDARD,
            target_backend="simulator"
        )
        
        assert request.request_id == "test_request"
        assert request.circuit_data == circuit_data
        assert request.optimization_level == OptimizationLevel.STANDARD
        assert request.target_backend == "simulator"
        assert request.custom_passes == []
        assert request.constraints == {}
        assert request.metadata == {}
        assert request.created_at > 0
    
    def test_compilation_request_with_custom_options(self):
        """Test request with custom options."""
        request = CompilationRequest(
            request_id="custom_request",
            circuit_data={'n_qubits': 2},
            optimization_level=OptimizationLevel.CUSTOM,
            custom_passes=['gate_fusion', 'depth_optimization'],
            constraints={'max_depth': 10},
            metadata={'user': 'test_user'}
        )
        
        assert request.optimization_level == OptimizationLevel.CUSTOM
        assert request.custom_passes == ['gate_fusion', 'depth_optimization']
        assert request.constraints == {'max_depth': 10}
        assert request.metadata == {'user': 'test_user'}
    
    def test_compilation_request_to_dict(self):
        """Test converting request to dictionary."""
        request = CompilationRequest(
            request_id="dict_test",
            circuit_data={'n_qubits': 2},
            optimization_level=OptimizationLevel.BASIC
        )
        
        request_dict = request.to_dict()
        
        assert request_dict['request_id'] == "dict_test"
        assert request_dict['circuit_data'] == {'n_qubits': 2}
        assert request_dict['optimization_level'] == OptimizationLevel.BASIC.value
        assert 'created_at' in request_dict


@pytest.mark.skipif(not HAS_COMPILATION_SERVICE, reason="compilation_service module not available")
class TestCompilationResult:
    """Test CompilationResult functionality."""
    
    def test_compilation_result_creation(self):
        """Test creating compilation result."""
        result = CompilationResult(
            request_id="test_result",
            status=CompilationStatus.COMPLETED,
            compiled_circuit={'n_qubits': 2, 'optimized': True},
            compilation_time=1.5
        )
        
        assert result.request_id == "test_result"
        assert result.status == CompilationStatus.COMPLETED
        assert result.compiled_circuit == {'n_qubits': 2, 'optimized': True}
        assert result.compilation_time == 1.5
        assert result.error_message is None
    
    def test_compilation_result_failed(self):
        """Test failed compilation result."""
        result = CompilationResult(
            request_id="failed_result",
            status=CompilationStatus.FAILED,
            error_message="Compilation failed",
            completed_at=time.time()
        )
        
        assert result.status == CompilationStatus.FAILED
        assert result.error_message == "Compilation failed"
        assert result.compiled_circuit is None
    
    def test_compilation_result_to_dict(self):
        """Test converting result to dictionary."""
        result = CompilationResult(
            request_id="dict_result",
            status=CompilationStatus.COMPLETED,
            metrics={'gate_count': 5}
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['request_id'] == "dict_result"
        assert result_dict['status'] == CompilationStatus.COMPLETED.value
        assert result_dict['metrics'] == {'gate_count': 5}


@pytest.mark.skipif(not HAS_COMPILATION_SERVICE, reason="compilation_service module not available")
class TestLocalCompilationBackend:
    """Test LocalCompilationBackend functionality."""
    
    def setup_method(self):
        """Set up test backend."""
        self.backend = LocalCompilationBackend()
    
    def test_backend_availability(self):
        """Test backend availability check."""
        # Should always be available (either native or mock)
        available = self.backend.is_available()
        assert isinstance(available, bool)
    
    def test_supported_optimizations(self):
        """Test getting supported optimizations."""
        optimizations = self.backend.get_supported_optimizations()
        
        assert isinstance(optimizations, list)
        assert len(optimizations) > 0
        assert 'gate_fusion' in optimizations
        assert 'circuit_simplification' in optimizations
    
    def test_compile_simple_circuit(self):
        """Test compiling a simple circuit."""
        circuit_data = {
            'n_qubits': 2,
            'gates': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'cnot', 'qubits': [0, 1]}
            ],
            'gate_count': 2,
            'depth': 2
        }
        
        request = CompilationRequest(
            request_id="simple_compile",
            circuit_data=circuit_data,
            optimization_level=OptimizationLevel.STANDARD
        )
        
        result = self.backend.compile_circuit(request)
        
        assert result.request_id == "simple_compile"
        assert result.status == CompilationStatus.COMPLETED
        assert result.compiled_circuit is not None
        assert result.compilation_time > 0
        assert result.optimization_report is not None
    
    def test_compile_with_optimization_levels(self):
        """Test compilation with different optimization levels."""
        circuit_data = {
            'n_qubits': 3,
            'gates': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'h', 'qubits': [1]},
                {'gate': 'cnot', 'qubits': [0, 2]}
            ],
            'gate_count': 3,
            'depth': 2
        }
        
        # Test different optimization levels
        for level in [OptimizationLevel.NONE, OptimizationLevel.BASIC, 
                     OptimizationLevel.STANDARD, OptimizationLevel.AGGRESSIVE]:
            request = CompilationRequest(
                request_id=f"opt_test_{level.value}",
                circuit_data=circuit_data,
                optimization_level=level
            )
            
            result = self.backend.compile_circuit(request)
            
            assert result.status == CompilationStatus.COMPLETED
            assert 'passes_applied' in result.optimization_report
            
            if level == OptimizationLevel.NONE:
                assert len(result.optimization_report['passes_applied']) == 0
            else:
                assert len(result.optimization_report['passes_applied']) > 0
    
    def test_compile_with_custom_passes(self):
        """Test compilation with custom optimization passes."""
        circuit_data = {'n_qubits': 2, 'gate_count': 5, 'depth': 3}
        
        request = CompilationRequest(
            request_id="custom_passes",
            circuit_data=circuit_data,
            optimization_level=OptimizationLevel.CUSTOM,
            custom_passes=['gate_fusion', 'noise_adaptive']
        )
        
        result = self.backend.compile_circuit(request)
        
        assert result.status == CompilationStatus.COMPLETED
        assert 'gate_fusion' in result.optimization_report['passes_applied']
        assert 'noise_adaptive' in result.optimization_report['passes_applied']
    
    def test_compile_invalid_circuit(self):
        """Test compilation with invalid circuit data."""
        invalid_data = {}  # Empty circuit data
        
        request = CompilationRequest(
            request_id="invalid_circuit",
            circuit_data=invalid_data
        )
        
        result = self.backend.compile_circuit(request)
        
        # Should handle gracefully, either success or controlled failure
        assert result.request_id == "invalid_circuit"
        assert result.status in [CompilationStatus.COMPLETED, CompilationStatus.FAILED]
    
    def test_circuit_metrics_calculation(self):
        """Test circuit metrics calculation."""
        circuit_data = {
            'n_qubits': 4,
            'gates': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'cnot', 'qubits': [0, 1]},
                {'gate': 'cnot', 'qubits': [1, 2]},
                {'gate': 'h', 'qubits': [3]}
            ],
            'gate_count': 4,
            'depth': 3
        }
        
        request = CompilationRequest(
            request_id="metrics_test",
            circuit_data=circuit_data,
            optimization_level=OptimizationLevel.STANDARD
        )
        
        result = self.backend.compile_circuit(request)
        
        assert result.status == CompilationStatus.COMPLETED
        assert 'original_gate_count' in result.metrics
        assert 'optimized_gate_count' in result.metrics
        assert 'compilation_time' in result.metrics
    
    def test_optimization_passes(self):
        """Test individual optimization passes."""
        test_circuit = {
            'n_qubits': 3,
            'gate_count': 10,
            'depth': 5
        }
        
        # Test gate fusion pass
        fused_circuit = self.backend._gate_fusion_pass(test_circuit)
        assert fused_circuit['gate_count'] <= test_circuit['gate_count']
        
        # Test circuit simplification pass
        simplified_circuit = self.backend._circuit_simplification_pass(test_circuit)
        assert simplified_circuit['depth'] <= test_circuit['depth']
        
        # Test depth optimization pass
        depth_optimized = self.backend._depth_optimization_pass(test_circuit)
        assert depth_optimized['depth'] <= test_circuit['depth']


@pytest.mark.skipif(not HAS_COMPILATION_SERVICE, reason="compilation_service module not available")
class TestRemoteCompilationBackend:
    """Test RemoteCompilationBackend functionality."""
    
    def setup_method(self):
        """Set up test backend."""
        self.backend = RemoteCompilationBackend(
            service_url="https://test-service.com",
            api_key="test_key"
        )
    
    def test_backend_availability(self):
        """Test remote backend availability."""
        available = self.backend.is_available()
        assert isinstance(available, bool)
    
    def test_supported_optimizations(self):
        """Test getting supported optimizations."""
        optimizations = self.backend.get_supported_optimizations()
        
        assert isinstance(optimizations, list)
        assert len(optimizations) > 0
        assert 'remote_optimization' in optimizations
    
    def test_remote_compilation(self):
        """Test remote compilation (mocked)."""
        circuit_data = {
            'n_qubits': 2,
            'gate_count': 3,
            'depth': 2
        }
        
        request = CompilationRequest(
            request_id="remote_test",
            circuit_data=circuit_data
        )
        
        result = self.backend.compile_circuit(request)
        
        assert result.request_id == "remote_test"
        assert result.status == CompilationStatus.COMPLETED
        assert 'remote_optimized' in result.compiled_circuit
        assert result.optimization_report['backend'] == 'remote'


@pytest.mark.skipif(not HAS_COMPILATION_SERVICE, reason="compilation_service module not available")
class TestCompilationCache:
    """Test CompilationCache functionality."""
    
    def setup_method(self):
        """Set up test cache."""
        self.cache = CompilationCache(max_size=5, ttl=1.0)  # Small cache for testing
    
    def test_cache_put_and_get(self):
        """Test basic cache operations."""
        result = CompilationResult(
            request_id="cache_test",
            status=CompilationStatus.COMPLETED,
            compiled_circuit={'optimized': True}
        )
        
        cache_key = "test_key"
        
        # Should be empty initially
        assert self.cache.get(cache_key) is None
        
        # Put and retrieve
        self.cache.put(cache_key, result)
        cached_result = self.cache.get(cache_key)
        
        assert cached_result is not None
        assert cached_result.status == CompilationStatus.CACHED
        assert cached_result.compiled_circuit == {'optimized': True}
    
    def test_cache_expiration(self):
        """Test cache TTL expiration."""
        result = CompilationResult(
            request_id="expire_test",
            status=CompilationStatus.COMPLETED
        )
        
        cache_key = "expire_key"
        self.cache.put(cache_key, result)
        
        # Should be available immediately
        assert self.cache.get(cache_key) is not None
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired
        assert self.cache.get(cache_key) is None
    
    def test_cache_size_limit(self):
        """Test cache size limit enforcement."""
        # Fill cache beyond limit
        for i in range(10):  # More than max_size (5)
            result = CompilationResult(
                request_id=f"size_test_{i}",
                status=CompilationStatus.COMPLETED
            )
            self.cache.put(f"key_{i}", result)
        
        # Should not exceed max size
        stats = self.cache.get_stats()
        assert stats['total_entries'] <= self.cache.max_size
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        request1 = CompilationRequest(
            request_id="key_test_1",
            circuit_data={'n_qubits': 2},
            optimization_level=OptimizationLevel.STANDARD
        )
        
        request2 = CompilationRequest(
            request_id="key_test_2",  # Different ID
            circuit_data={'n_qubits': 2},  # Same circuit data
            optimization_level=OptimizationLevel.STANDARD
        )
        
        key1 = self.cache.generate_cache_key(request1)
        key2 = self.cache.generate_cache_key(request2)
        
        # Should generate same key for same compilation parameters
        assert key1 == key2
        
        # Different optimization level should give different key
        request3 = CompilationRequest(
            request_id="key_test_3",
            circuit_data={'n_qubits': 2},
            optimization_level=OptimizationLevel.AGGRESSIVE
        )
        
        key3 = self.cache.generate_cache_key(request3)
        assert key1 != key3
    
    def test_cache_clear(self):
        """Test cache clearing."""
        # Add some entries
        for i in range(3):
            result = CompilationResult(f"clear_test_{i}", CompilationStatus.COMPLETED)
            self.cache.put(f"clear_key_{i}", result)
        
        assert self.cache.get_stats()['total_entries'] == 3
        
        # Clear cache
        self.cache.clear()
        
        assert self.cache.get_stats()['total_entries'] == 0
    
    def test_cache_stats(self):
        """Test cache statistics."""
        stats = self.cache.get_stats()
        
        assert 'total_entries' in stats
        assert 'valid_entries' in stats
        assert 'max_size' in stats
        assert 'ttl' in stats
        
        assert stats['max_size'] == 5
        assert stats['ttl'] == 1.0


@pytest.mark.skipif(not HAS_COMPILATION_SERVICE, reason="compilation_service module not available")
class TestCompilationService:
    """Test CompilationService functionality."""
    
    def setup_method(self):
        """Set up test service."""
        self.service = CompilationService()
    
    def teardown_method(self):
        """Clean up test service."""
        self.service.cleanup()
    
    def test_service_initialization(self):
        """Test service initialization."""
        assert self.service.backends is not None
        assert self.service.cache is not None
        assert self.service.is_running is True
        assert len(self.service.worker_threads) > 0
    
    def test_submit_compilation(self):
        """Test submitting compilation request."""
        circuit_data = {
            'n_qubits': 2,
            'gates': [{'gate': 'h', 'qubits': [0]}]
        }
        
        request_id = self.service.submit_compilation(circuit_data)
        
        assert request_id is not None
        assert request_id in self.service.active_requests
        
        # Wait for processing
        time.sleep(0.5)
        
        status = self.service.get_compilation_status(request_id)
        assert status in [CompilationStatus.PENDING, CompilationStatus.COMPLETED, CompilationStatus.CACHED]
    
    def test_synchronous_compilation(self):
        """Test synchronous compilation."""
        circuit_data = {
            'n_qubits': 3,
            'gates': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'cnot', 'qubits': [0, 1]}
            ]
        }
        
        result = self.service.compile_circuit_sync(
            circuit_data, OptimizationLevel.STANDARD, timeout=5.0
        )
        
        assert result is not None
        assert result.status in [CompilationStatus.COMPLETED, CompilationStatus.CACHED]
        assert result.compiled_circuit is not None
    
    def test_compilation_with_different_levels(self):
        """Test compilation with different optimization levels."""
        circuit_data = {'n_qubits': 2, 'gate_count': 5}
        
        for level in [OptimizationLevel.NONE, OptimizationLevel.BASIC, 
                     OptimizationLevel.STANDARD, OptimizationLevel.AGGRESSIVE]:
            result = self.service.compile_circuit_sync(circuit_data, level, timeout=5.0)
            assert result.status in [CompilationStatus.COMPLETED, CompilationStatus.CACHED]
    
    def test_get_compilation_result(self):
        """Test getting compilation results."""
        circuit_data = {'n_qubits': 2}
        request_id = self.service.submit_compilation(circuit_data)
        
        # Wait for completion
        start_time = time.time()
        while time.time() - start_time < 5.0:
            result = self.service.get_compilation_result(request_id)
            if result:
                break
            time.sleep(0.1)
        
        assert result is not None
        assert result.request_id == request_id
    
    def test_service_stats(self):
        """Test getting service statistics."""
        stats = self.service.get_service_stats()
        
        assert 'active_requests' in stats
        assert 'completed_requests' in stats
        assert 'available_backends' in stats
        assert 'worker_threads' in stats
        assert 'is_running' in stats
        assert 'cache_stats' in stats
        
        assert isinstance(stats['active_requests'], int)
        assert isinstance(stats['is_running'], bool)
    
    def test_list_available_optimizations(self):
        """Test listing available optimizations."""
        optimizations = self.service.list_available_optimizations()
        
        assert isinstance(optimizations, dict)
        assert len(optimizations) > 0
        
        for backend_name, opts in optimizations.items():
            assert isinstance(opts, list)
    
    def test_cache_integration(self):
        """Test cache integration with service."""
        circuit_data = {'n_qubits': 2, 'gate_count': 3}
        
        # First compilation
        result1 = self.service.compile_circuit_sync(circuit_data, OptimizationLevel.STANDARD)
        assert result1.status == CompilationStatus.COMPLETED
        
        # Second compilation (should be cached)
        result2 = self.service.compile_circuit_sync(circuit_data, OptimizationLevel.STANDARD)
        assert result2.status == CompilationStatus.CACHED
    
    def test_clear_cache(self):
        """Test clearing service cache."""
        circuit_data = {'n_qubits': 2}
        
        # Compile and cache
        self.service.compile_circuit_sync(circuit_data)
        
        cache_stats_before = self.service.get_service_stats()['cache_stats']
        
        # Clear cache
        self.service.clear_cache()
        
        cache_stats_after = self.service.get_service_stats()['cache_stats']
        assert cache_stats_after['total_entries'] == 0
    
    def test_compilation_timeout(self):
        """Test compilation timeout handling."""
        circuit_data = {'n_qubits': 2}
        
        # Very short timeout
        result = self.service.compile_circuit_sync(circuit_data, timeout=0.001)
        
        # Should either complete very quickly or timeout
        assert result.status in [CompilationStatus.COMPLETED, CompilationStatus.CACHED, 
                                CompilationStatus.FAILED]
    
    def test_worker_thread_management(self):
        """Test worker thread lifecycle."""
        initial_thread_count = len(self.service.worker_threads)
        assert initial_thread_count > 0
        
        # Stop workers
        self.service.stop_workers()
        assert self.service.is_running is False
        
        # Restart workers
        self.service.start_workers(num_workers=3)
        assert self.service.is_running is True
        assert len(self.service.worker_threads) == 3


@pytest.mark.skipif(not HAS_COMPILATION_SERVICE or not FLASK_AVAILABLE, 
                   reason="compilation_service or flask not available")
class TestCompilationServiceAPI:
    """Test CompilationServiceAPI functionality."""
    
    def setup_method(self):
        """Set up test API."""
        self.service = CompilationService()
        self.api = CompilationServiceAPI(self.service, host="localhost", port=5555)
        self.client = self.api.app.test_client()
    
    def teardown_method(self):
        """Clean up test API."""
        self.service.cleanup()
    
    def test_compile_endpoint(self):
        """Test compilation API endpoint."""
        circuit_data = {
            'n_qubits': 2,
            'gates': [{'gate': 'h', 'qubits': [0]}]
        }
        
        response = self.client.post('/api/compile', 
                                  json={
                                      'circuit_data': circuit_data,
                                      'optimization_level': 2
                                  })
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert 'request_id' in data
    
    def test_compile_sync_endpoint(self):
        """Test synchronous compilation API endpoint."""
        circuit_data = {
            'n_qubits': 2,
            'gates': [{'gate': 'h', 'qubits': [0]}]
        }
        
        response = self.client.post('/api/compile/sync',
                                  json={
                                      'circuit_data': circuit_data,
                                      'optimization_level': 1,
                                      'timeout': 10.0
                                  })
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert 'result' in data
        assert data['result']['status'] in ['completed', 'cached']
    
    def test_status_endpoint(self):
        """Test status API endpoint."""
        # Submit a request first
        circuit_data = {'n_qubits': 2}
        response = self.client.post('/api/compile', 
                                  json={'circuit_data': circuit_data})
        request_id = response.get_json()['request_id']
        
        # Check status
        response = self.client.get(f'/api/status/{request_id}')
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['request_id'] == request_id
        assert 'status' in data
    
    def test_result_endpoint(self):
        """Test result API endpoint."""
        # Submit and wait for completion
        circuit_data = {'n_qubits': 2}
        sync_response = self.client.post('/api/compile/sync',
                                       json={'circuit_data': circuit_data})
        request_id = sync_response.get_json()['result']['request_id']
        
        # Get result
        response = self.client.get(f'/api/result/{request_id}')
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['success'] is True
        assert 'result' in data
    
    def test_stats_endpoint(self):
        """Test statistics API endpoint."""
        response = self.client.get('/api/stats')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'active_requests' in data
        assert 'completed_requests' in data
        assert 'available_backends' in data
    
    def test_optimizations_endpoint(self):
        """Test optimizations API endpoint."""
        response = self.client.get('/api/optimizations')
        assert response.status_code == 200
        
        data = response.get_json()
        assert isinstance(data, dict)
        assert len(data) > 0
    
    def test_clear_cache_endpoint(self):
        """Test cache clearing API endpoint."""
        response = self.client.delete('/api/cache')
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['success'] is True
        assert data['message'] == 'Cache cleared'
    
    def test_nonexistent_request_endpoints(self):
        """Test endpoints with nonexistent requests."""
        # Status endpoint
        response = self.client.get('/api/status/nonexistent')
        assert response.status_code == 404
        
        # Result endpoint
        response = self.client.get('/api/result/nonexistent')
        assert response.status_code == 404
    
    def test_invalid_compilation_request(self):
        """Test API with invalid compilation request."""
        response = self.client.post('/api/compile', 
                                  json={'invalid': 'data'})
        
        # Should handle gracefully
        assert response.status_code in [200, 400]


@pytest.mark.skipif(not HAS_COMPILATION_SERVICE, reason="compilation_service module not available")
class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_get_compilation_service(self):
        """Test getting global compilation service."""
        service1 = get_compilation_service()
        service2 = get_compilation_service()
        
        # Should be singleton
        assert service1 is service2
        assert isinstance(service1, CompilationService)
    
    def test_compile_circuit_convenience(self):
        """Test compile_circuit convenience function."""
        circuit_data = {
            'n_qubits': 2,
            'gates': [{'gate': 'h', 'qubits': [0]}]
        }
        
        result = compile_circuit(circuit_data, OptimizationLevel.BASIC, timeout=5.0)
        
        assert isinstance(result, CompilationResult)
        assert result.status in [CompilationStatus.COMPLETED, CompilationStatus.CACHED]


@pytest.mark.skipif(not HAS_COMPILATION_SERVICE, reason="compilation_service module not available")
class TestErrorHandling:
    """Test error handling scenarios."""
    
    def setup_method(self):
        """Set up test environment."""
        self.service = CompilationService()
    
    def teardown_method(self):
        """Clean up test environment."""
        self.service.cleanup()
    
    def test_backend_failure_handling(self):
        """Test handling of backend failures."""
        # Create failing backend
        class FailingBackend(LocalCompilationBackend):
            def compile_circuit(self, request):
                raise Exception("Backend failure")
        
        # Replace backend
        self.service.backends[CompilationBackend.LOCAL] = FailingBackend()
        
        circuit_data = {'n_qubits': 2}
        result = self.service.compile_circuit_sync(circuit_data, timeout=5.0)
        
        assert result.status == CompilationStatus.FAILED
        assert "error" in result.error_message.lower()
    
    def test_invalid_optimization_level(self):
        """Test handling of invalid optimization parameters."""
        circuit_data = {'n_qubits': 2}
        
        # Should handle gracefully
        try:
            result = self.service.compile_circuit_sync(circuit_data, 
                                                     OptimizationLevel.CUSTOM,
                                                     timeout=5.0)
            assert result is not None
        except Exception:
            # Should not raise unhandled exceptions
            assert False, "Should handle invalid optimization gracefully"
    
    def test_empty_circuit_data(self):
        """Test handling of empty circuit data."""
        empty_data = {}
        
        result = self.service.compile_circuit_sync(empty_data, timeout=5.0)
        
        # Should complete (possibly with empty optimized circuit)
        assert result.status in [CompilationStatus.COMPLETED, CompilationStatus.FAILED]
    
    def test_worker_thread_exceptions(self):
        """Test worker thread exception handling."""
        # Submit multiple requests rapidly to test thread safety
        requests = []
        for i in range(10):
            circuit_data = {'n_qubits': 2, 'test_id': i}
            request_id = self.service.submit_compilation(circuit_data)
            requests.append(request_id)
        
        # Wait for all to complete
        time.sleep(2.0)
        
        # All should have completed or failed gracefully
        completed_count = 0
        for request_id in requests:
            status = self.service.get_compilation_status(request_id)
            if status in [CompilationStatus.COMPLETED, CompilationStatus.CACHED]:
                completed_count += 1
        
        # Should have completed most/all requests
        assert completed_count >= len(requests) * 0.8  # Allow some tolerance


@pytest.mark.skipif(not HAS_COMPILATION_SERVICE, reason="compilation_service module not available")
class TestCompilationServiceIntegration:
    """Test integration scenarios."""
    
    def test_end_to_end_compilation_workflow(self):
        """Test complete compilation workflow."""
        service = CompilationService()
        
        try:
            # Create a complex circuit
            circuit_data = {
                'n_qubits': 4,
                'gates': [
                    {'gate': 'h', 'qubits': [0]},
                    {'gate': 'h', 'qubits': [1]},
                    {'gate': 'cnot', 'qubits': [0, 2]},
                    {'gate': 'cnot', 'qubits': [1, 3]},
                    {'gate': 'rz', 'qubits': [2], 'params': [1.57]},
                    {'gate': 'ry', 'qubits': [3], 'params': [0.78]}
                ],
                'gate_count': 6,
                'depth': 3
            }
            
            # Test different optimization levels
            results = {}
            for level in [OptimizationLevel.NONE, OptimizationLevel.BASIC,
                         OptimizationLevel.STANDARD, OptimizationLevel.AGGRESSIVE]:
                result = service.compile_circuit_sync(circuit_data, level, timeout=10.0)
                results[level] = result
                
                assert result.status in [CompilationStatus.COMPLETED, CompilationStatus.CACHED]
                assert result.compiled_circuit is not None
                assert result.optimization_report is not None
                assert result.compilation_time >= 0
            
            # Check that optimization actually improves circuits
            none_result = results[OptimizationLevel.NONE]
            aggressive_result = results[OptimizationLevel.AGGRESSIVE]
            
            # Aggressive optimization should apply more passes
            assert (len(aggressive_result.optimization_report.get('passes_applied', [])) >=
                   len(none_result.optimization_report.get('passes_applied', [])))
            
            # Test caching - second compilation should be faster
            start_time = time.time()
            cached_result = service.compile_circuit_sync(circuit_data, 
                                                       OptimizationLevel.STANDARD)
            cache_time = time.time() - start_time
            
            assert cached_result.status == CompilationStatus.CACHED
            assert cache_time < 0.1  # Should be very fast
            
        finally:
            service.cleanup()
    
    def test_concurrent_compilation_requests(self):
        """Test handling concurrent compilation requests."""
        service = CompilationService()
        
        try:
            # Submit multiple requests concurrently
            import concurrent.futures
            
            def compile_circuit_task(circuit_id):
                circuit_data = {
                    'n_qubits': 3,
                    'circuit_id': circuit_id,
                    'gates': [
                        {'gate': 'h', 'qubits': [0]},
                        {'gate': 'cnot', 'qubits': [0, 1]},
                        {'gate': 'cnot', 'qubits': [1, 2]}
                    ]
                }
                return service.compile_circuit_sync(circuit_data, timeout=10.0)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(compile_circuit_task, i) for i in range(10)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            # All should complete successfully
            for result in results:
                assert result.status in [CompilationStatus.COMPLETED, CompilationStatus.CACHED]
            
            # Should have good mix of completed and cached results
            completed_count = sum(1 for r in results if r.status == CompilationStatus.COMPLETED)
            cached_count = sum(1 for r in results if r.status == CompilationStatus.CACHED)
            
            assert completed_count > 0  # At least some original compilations
            # May or may not have cache hits depending on timing
            
        finally:
            service.cleanup()
    
    def test_service_stats_accuracy(self):
        """Test accuracy of service statistics."""
        service = CompilationService()
        
        try:
            initial_stats = service.get_service_stats()
            
            # Submit some requests
            circuit_data = {'n_qubits': 2}
            request_ids = []
            
            for i in range(5):
                request_id = service.submit_compilation(circuit_data)
                request_ids.append(request_id)
            
            # Wait for completion
            time.sleep(1.0)
            
            final_stats = service.get_service_stats()
            
            # Check that stats reflect the activity
            assert final_stats['completed_requests'] >= initial_stats['completed_requests']
            assert final_stats['cache_stats']['total_entries'] >= 0
            
        finally:
            service.cleanup()


@pytest.mark.skipif(not HAS_COMPILATION_SERVICE, reason="compilation_service module not available")
class TestCompilationServicePerformance:
    """Test performance characteristics."""
    
    def test_compilation_performance(self):
        """Test compilation performance."""
        service = CompilationService()
        
        try:
            circuit_data = {
                'n_qubits': 5,
                'gates': [
                    {'gate': 'h', 'qubits': [i]} for i in range(5)
                ] + [
                    {'gate': 'cnot', 'qubits': [i, (i + 1) % 5]} for i in range(5)
                ],
                'gate_count': 10,
                'depth': 2
            }
            
            start_time = time.time()
            
            # Compile multiple circuits
            results = []
            for i in range(20):
                result = service.compile_circuit_sync(circuit_data, timeout=5.0)
                results.append(result)
            
            total_time = time.time() - start_time
            
            # Should complete within reasonable time
            assert total_time < 10.0  # 10 seconds for 20 compilations
            
            # All should succeed
            for result in results:
                assert result.status in [CompilationStatus.COMPLETED, CompilationStatus.CACHED]
            
            # Later requests should be cached and faster
            cached_count = sum(1 for r in results if r.status == CompilationStatus.CACHED)
            assert cached_count > 0  # Should have some cache hits
            
        finally:
            service.cleanup()
    
    def test_cache_performance(self):
        """Test cache performance under load."""
        service = CompilationService()
        
        try:
            # Generate many different circuits
            circuits = []
            for i in range(50):
                circuit_data = {
                    'n_qubits': 2,
                    'circuit_id': i,
                    'gates': [{'gate': 'h', 'qubits': [i % 2]}]
                }
                circuits.append(circuit_data)
            
            start_time = time.time()
            
            # Compile all circuits
            for circuit_data in circuits:
                service.compile_circuit_sync(circuit_data, timeout=5.0)
            
            compilation_time = time.time() - start_time
            
            # Now test cache retrieval performance
            cache_start_time = time.time()
            
            # Recompile same circuits (should hit cache)
            for circuit_data in circuits:
                result = service.compile_circuit_sync(circuit_data, timeout=5.0)
                assert result.status == CompilationStatus.CACHED
            
            cache_time = time.time() - cache_start_time
            
            # Cache retrieval should be much faster
            assert cache_time < compilation_time * 0.1  # At least 10x faster
            
        finally:
            service.cleanup()


if __name__ == "__main__":
    pytest.main([__file__])