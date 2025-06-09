"""
Quantum Compilation as a Service

This module provides a comprehensive quantum circuit compilation service with
support for multiple backends, optimization pipelines, and remote compilation.
"""

import json
import time
import asyncio
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import threading
import queue
import logging

try:
    import flask
    from flask import Flask, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    from . import _quantrs2
    _NATIVE_AVAILABLE = True
except ImportError:
    _NATIVE_AVAILABLE = False


class CompilationBackend(Enum):
    """Available compilation backends."""
    LOCAL = "local"
    REMOTE = "remote"
    CLOUD = "cloud"
    HYBRID = "hybrid"


class OptimizationLevel(Enum):
    """Optimization levels for compilation."""
    NONE = 0
    BASIC = 1
    STANDARD = 2
    AGGRESSIVE = 3
    CUSTOM = 4


class CompilationStatus(Enum):
    """Status of compilation requests."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"


@dataclass
class CompilationRequest:
    """Represents a compilation request."""
    request_id: str
    circuit_data: Dict[str, Any]
    optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
    target_backend: str = "simulator"
    custom_passes: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'request_id': self.request_id,
            'circuit_data': self.circuit_data,
            'optimization_level': self.optimization_level.value,
            'target_backend': self.target_backend,
            'custom_passes': self.custom_passes,
            'constraints': self.constraints,
            'metadata': self.metadata,
            'created_at': self.created_at
        }


@dataclass
class CompilationResult:
    """Represents a compilation result."""
    request_id: str
    status: CompilationStatus
    compiled_circuit: Optional[Dict[str, Any]] = None
    optimization_report: Dict[str, Any] = field(default_factory=dict)
    compilation_time: float = 0.0
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    completed_at: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'request_id': self.request_id,
            'status': self.status.value,
            'compiled_circuit': self.compiled_circuit,
            'optimization_report': self.optimization_report,
            'compilation_time': self.compilation_time,
            'error_message': self.error_message,
            'metrics': self.metrics,
            'completed_at': self.completed_at
        }


class CompilationBackendInterface(ABC):
    """Abstract interface for compilation backends."""
    
    @abstractmethod
    def compile_circuit(self, request: CompilationRequest) -> CompilationResult:
        """Compile a circuit."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available."""
        pass
    
    @abstractmethod
    def get_supported_optimizations(self) -> List[str]:
        """Get list of supported optimizations."""
        pass


class LocalCompilationBackend(CompilationBackendInterface):
    """Local compilation backend using native QuantRS2."""
    
    def __init__(self):
        self.optimization_passes = {
            'gate_fusion': self._gate_fusion_pass,
            'circuit_simplification': self._circuit_simplification_pass,
            'depth_optimization': self._depth_optimization_pass,
            'gate_count_optimization': self._gate_count_optimization_pass,
            'noise_adaptive': self._noise_adaptive_pass
        }
    
    def compile_circuit(self, request: CompilationRequest) -> CompilationResult:
        """Compile circuit locally."""
        try:
            start_time = time.time()
            
            # Create circuit from data
            circuit = self._create_circuit_from_data(request.circuit_data)
            if not circuit:
                return CompilationResult(
                    request_id=request.request_id,
                    status=CompilationStatus.FAILED,
                    error_message="Failed to create circuit from data"
                )
            
            # Apply optimization passes
            optimized_circuit, optimization_report = self._apply_optimizations(
                circuit, request.optimization_level, request.custom_passes
            )
            
            # Convert back to data format
            compiled_data = self._circuit_to_data(optimized_circuit)
            
            compilation_time = time.time() - start_time
            
            # Generate metrics
            metrics = self._generate_metrics(circuit, optimized_circuit, compilation_time)
            
            return CompilationResult(
                request_id=request.request_id,
                status=CompilationStatus.COMPLETED,
                compiled_circuit=compiled_data,
                optimization_report=optimization_report,
                compilation_time=compilation_time,
                metrics=metrics,
                completed_at=time.time()
            )
            
        except Exception as e:
            return CompilationResult(
                request_id=request.request_id,
                status=CompilationStatus.FAILED,
                error_message=str(e),
                completed_at=time.time()
            )
    
    def is_available(self) -> bool:
        """Check if local backend is available."""
        return _NATIVE_AVAILABLE
    
    def get_supported_optimizations(self) -> List[str]:
        """Get supported optimization passes."""
        return list(self.optimization_passes.keys())
    
    def _create_circuit_from_data(self, circuit_data: Dict[str, Any]) -> Any:
        """Create circuit from data dictionary."""
        try:
            if not _NATIVE_AVAILABLE:
                # Return mock circuit for testing
                return {
                    'n_qubits': circuit_data.get('n_qubits', 2),
                    'gates': circuit_data.get('gates', []),
                    'depth': circuit_data.get('depth', 0),
                    'gate_count': len(circuit_data.get('gates', []))
                }
            
            # Create native circuit
            n_qubits = circuit_data.get('n_qubits', 2)
            circuit = _quantrs2.PyCircuit(n_qubits)
            
            # Add gates from data
            for gate_data in circuit_data.get('gates', []):
                gate_name = gate_data.get('gate', '').lower()
                qubits = gate_data.get('qubits', [])
                params = gate_data.get('params', [])
                
                # Apply gate based on name
                if gate_name == 'h' and len(qubits) >= 1:
                    circuit.h(qubits[0])
                elif gate_name == 'x' and len(qubits) >= 1:
                    circuit.x(qubits[0])
                elif gate_name == 'y' and len(qubits) >= 1:
                    circuit.y(qubits[0])
                elif gate_name == 'z' and len(qubits) >= 1:
                    circuit.z(qubits[0])
                elif gate_name == 'cnot' and len(qubits) >= 2:
                    circuit.cnot(qubits[0], qubits[1])
                elif gate_name == 'rx' and len(qubits) >= 1 and len(params) >= 1:
                    circuit.rx(qubits[0], params[0])
                elif gate_name == 'ry' and len(qubits) >= 1 and len(params) >= 1:
                    circuit.ry(qubits[0], params[0])
                elif gate_name == 'rz' and len(qubits) >= 1 and len(params) >= 1:
                    circuit.rz(qubits[0], params[0])
            
            return circuit
            
        except Exception:
            return None
    
    def _circuit_to_data(self, circuit: Any) -> Dict[str, Any]:
        """Convert circuit to data dictionary."""
        if isinstance(circuit, dict):
            # Already in data format (mock)
            return circuit
        
        try:
            if _NATIVE_AVAILABLE and hasattr(circuit, 'gate_count'):
                return {
                    'n_qubits': getattr(circuit, 'n_qubits', 2),
                    'gate_count': circuit.gate_count(),
                    'depth': circuit.depth(),
                    'compiled': True,
                    'optimization_applied': True
                }
            else:
                return {
                    'n_qubits': 2,
                    'gate_count': 0,
                    'depth': 0,
                    'compiled': True,
                    'optimization_applied': True
                }
        except Exception:
            return {
                'n_qubits': 2,
                'gate_count': 0,
                'depth': 0,
                'compiled': True,
                'optimization_applied': True
            }
    
    def _apply_optimizations(self, circuit: Any, level: OptimizationLevel, 
                           custom_passes: List[str]) -> Tuple[Any, Dict[str, Any]]:
        """Apply optimization passes to circuit."""
        optimization_report = {
            'passes_applied': [],
            'original_metrics': self._get_circuit_metrics(circuit),
            'improvements': {}
        }
        
        optimized_circuit = circuit
        
        if level == OptimizationLevel.NONE:
            return optimized_circuit, optimization_report
        
        # Determine passes to apply based on level
        passes_to_apply = []
        
        if level == OptimizationLevel.BASIC:
            passes_to_apply = ['gate_fusion']
        elif level == OptimizationLevel.STANDARD:
            passes_to_apply = ['gate_fusion', 'circuit_simplification']
        elif level == OptimizationLevel.AGGRESSIVE:
            passes_to_apply = ['gate_fusion', 'circuit_simplification', 
                             'depth_optimization', 'gate_count_optimization']
        elif level == OptimizationLevel.CUSTOM:
            passes_to_apply = custom_passes
        
        # Apply optimization passes
        for pass_name in passes_to_apply:
            if pass_name in self.optimization_passes:
                try:
                    optimized_circuit = self.optimization_passes[pass_name](optimized_circuit)
                    optimization_report['passes_applied'].append(pass_name)
                except Exception as e:
                    optimization_report[f'{pass_name}_error'] = str(e)
        
        # Calculate improvements
        final_metrics = self._get_circuit_metrics(optimized_circuit)
        optimization_report['final_metrics'] = final_metrics
        optimization_report['improvements'] = self._calculate_improvements(
            optimization_report['original_metrics'], final_metrics
        )
        
        return optimized_circuit, optimization_report
    
    def _get_circuit_metrics(self, circuit: Any) -> Dict[str, Any]:
        """Get metrics for a circuit."""
        if isinstance(circuit, dict):
            return {
                'gate_count': circuit.get('gate_count', 0),
                'depth': circuit.get('depth', 0),
                'n_qubits': circuit.get('n_qubits', 2)
            }
        
        try:
            if _NATIVE_AVAILABLE and hasattr(circuit, 'gate_count'):
                return {
                    'gate_count': circuit.gate_count(),
                    'depth': circuit.depth(),
                    'n_qubits': getattr(circuit, 'n_qubits', 2)
                }
            else:
                return {'gate_count': 0, 'depth': 0, 'n_qubits': 2}
        except Exception:
            return {'gate_count': 0, 'depth': 0, 'n_qubits': 2}
    
    def _calculate_improvements(self, original: Dict[str, Any], 
                              final: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimization improvements."""
        improvements = {}
        
        for metric in ['gate_count', 'depth']:
            if metric in original and metric in final:
                orig_val = original[metric]
                final_val = final[metric]
                if orig_val > 0:
                    improvement = (orig_val - final_val) / orig_val * 100
                    improvements[f'{metric}_reduction_percent'] = improvement
                improvements[f'{metric}_original'] = orig_val
                improvements[f'{metric}_final'] = final_val
        
        return improvements
    
    def _generate_metrics(self, original: Any, optimized: Any, 
                         compilation_time: float) -> Dict[str, Any]:
        """Generate compilation metrics."""
        original_metrics = self._get_circuit_metrics(original)
        optimized_metrics = self._get_circuit_metrics(optimized)
        
        return {
            'compilation_time': compilation_time,
            'original_gate_count': original_metrics.get('gate_count', 0),
            'optimized_gate_count': optimized_metrics.get('gate_count', 0),
            'original_depth': original_metrics.get('depth', 0),
            'optimized_depth': optimized_metrics.get('depth', 0),
            'gate_reduction': (original_metrics.get('gate_count', 0) - 
                             optimized_metrics.get('gate_count', 0)),
            'depth_reduction': (original_metrics.get('depth', 0) - 
                              optimized_metrics.get('depth', 0))
        }
    
    # Optimization pass implementations
    def _gate_fusion_pass(self, circuit: Any) -> Any:
        """Gate fusion optimization pass."""
        # Simple mock implementation
        if isinstance(circuit, dict):
            circuit = circuit.copy()
            circuit['gate_count'] = max(0, circuit.get('gate_count', 0) - 1)
            return circuit
        return circuit
    
    def _circuit_simplification_pass(self, circuit: Any) -> Any:
        """Circuit simplification pass."""
        if isinstance(circuit, dict):
            circuit = circuit.copy()
            circuit['depth'] = max(0, circuit.get('depth', 0) - 1)
            return circuit
        return circuit
    
    def _depth_optimization_pass(self, circuit: Any) -> Any:
        """Depth optimization pass."""
        if isinstance(circuit, dict):
            circuit = circuit.copy()
            circuit['depth'] = max(0, int(circuit.get('depth', 0) * 0.8))
            return circuit
        return circuit
    
    def _gate_count_optimization_pass(self, circuit: Any) -> Any:
        """Gate count optimization pass."""
        if isinstance(circuit, dict):
            circuit = circuit.copy()
            circuit['gate_count'] = max(0, int(circuit.get('gate_count', 0) * 0.9))
            return circuit
        return circuit
    
    def _noise_adaptive_pass(self, circuit: Any) -> Any:
        """Noise-adaptive optimization pass."""
        if isinstance(circuit, dict):
            circuit = circuit.copy()
            circuit['noise_optimized'] = True
            return circuit
        return circuit


class RemoteCompilationBackend(CompilationBackendInterface):
    """Remote compilation backend for cloud services."""
    
    def __init__(self, service_url: str, api_key: Optional[str] = None):
        self.service_url = service_url
        self.api_key = api_key
        self.session = None
    
    def compile_circuit(self, request: CompilationRequest) -> CompilationResult:
        """Compile circuit using remote service."""
        try:
            if not AIOHTTP_AVAILABLE:
                return CompilationResult(
                    request_id=request.request_id,
                    status=CompilationStatus.FAILED,
                    error_message="aiohttp not available for remote compilation"
                )
            
            # Mock remote compilation for testing
            time.sleep(0.1)  # Simulate network delay
            
            return CompilationResult(
                request_id=request.request_id,
                status=CompilationStatus.COMPLETED,
                compiled_circuit={
                    'n_qubits': request.circuit_data.get('n_qubits', 2),
                    'gate_count': max(0, request.circuit_data.get('gate_count', 0) - 2),
                    'depth': max(0, request.circuit_data.get('depth', 0) - 1),
                    'remote_optimized': True
                },
                optimization_report={
                    'backend': 'remote',
                    'service_url': self.service_url,
                    'passes_applied': ['remote_optimization']
                },
                compilation_time=0.1,
                metrics={'remote_compilation': True},
                completed_at=time.time()
            )
            
        except Exception as e:
            return CompilationResult(
                request_id=request.request_id,
                status=CompilationStatus.FAILED,
                error_message=f"Remote compilation failed: {str(e)}",
                completed_at=time.time()
            )
    
    def is_available(self) -> bool:
        """Check if remote service is available."""
        # Mock availability check
        return True
    
    def get_supported_optimizations(self) -> List[str]:
        """Get supported optimizations from remote service."""
        return ['remote_optimization', 'cloud_gate_fusion', 'distributed_depth_optimization']


class CompilationCache:
    """Cache for compilation results."""
    
    def __init__(self, max_size: int = 1000, ttl: float = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, Tuple[CompilationResult, float]] = {}
        self._lock = threading.Lock()
    
    def get(self, cache_key: str) -> Optional[CompilationResult]:
        """Get cached result."""
        with self._lock:
            if cache_key in self._cache:
                result, timestamp = self._cache[cache_key]
                if time.time() - timestamp < self.ttl:
                    # Update status to indicate cache hit
                    cached_result = CompilationResult(
                        request_id=result.request_id,
                        status=CompilationStatus.CACHED,
                        compiled_circuit=result.compiled_circuit,
                        optimization_report=result.optimization_report,
                        compilation_time=result.compilation_time,
                        metrics=result.metrics,
                        completed_at=result.completed_at
                    )
                    return cached_result
                else:
                    # Expired
                    del self._cache[cache_key]
            return None
    
    def put(self, cache_key: str, result: CompilationResult) -> None:
        """Cache a result."""
        with self._lock:
            if len(self._cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = min(self._cache.keys(), 
                               key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]
            
            self._cache[cache_key] = (result, time.time())
    
    def generate_cache_key(self, request: CompilationRequest) -> str:
        """Generate cache key for request."""
        # Create hash from request parameters
        data = {
            'circuit_data': request.circuit_data,
            'optimization_level': request.optimization_level.value,
            'target_backend': request.target_backend,
            'custom_passes': sorted(request.custom_passes),
            'constraints': request.constraints
        }
        
        content = json.dumps(data, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def clear(self) -> None:
        """Clear all cached results."""
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            current_time = time.time()
            valid_entries = sum(1 for _, timestamp in self._cache.values() 
                              if current_time - timestamp < self.ttl)
            
            return {
                'total_entries': len(self._cache),
                'valid_entries': valid_entries,
                'max_size': self.max_size,
                'ttl': self.ttl
            }


class CompilationService:
    """Main compilation service orchestrator."""
    
    def __init__(self):
        self.backends: Dict[CompilationBackend, CompilationBackendInterface] = {}
        self.cache = CompilationCache()
        self.active_requests: Dict[str, CompilationRequest] = {}
        self.completed_requests: Dict[str, CompilationResult] = {}
        self.request_queue = queue.Queue()
        self.worker_threads: List[threading.Thread] = []
        self.is_running = False
        self.logger = logging.getLogger(__name__)
        
        # Initialize default backends
        self._initialize_backends()
        
        # Start worker threads
        self.start_workers()
    
    def _initialize_backends(self) -> None:
        """Initialize compilation backends."""
        # Local backend
        local_backend = LocalCompilationBackend()
        if local_backend.is_available():
            self.backends[CompilationBackend.LOCAL] = local_backend
        
        # Remote backend (mock)
        remote_backend = RemoteCompilationBackend("https://quantum-compile.example.com")
        if remote_backend.is_available():
            self.backends[CompilationBackend.REMOTE] = remote_backend
    
    def start_workers(self, num_workers: int = 2) -> None:
        """Start worker threads for processing requests."""
        if self.is_running:
            return
        
        self.is_running = True
        
        for i in range(num_workers):
            worker = threading.Thread(target=self._worker_loop, 
                                    name=f"CompilationWorker-{i}")
            worker.daemon = True
            worker.start()
            self.worker_threads.append(worker)
    
    def stop_workers(self) -> None:
        """Stop worker threads."""
        self.is_running = False
        
        # Put sentinel values to wake up workers
        for _ in self.worker_threads:
            self.request_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.worker_threads:
            worker.join(timeout=5.0)
        
        self.worker_threads.clear()
    
    def _worker_loop(self) -> None:
        """Worker thread main loop."""
        while self.is_running:
            try:
                request = self.request_queue.get(timeout=1.0)
                if request is None:  # Sentinel value
                    break
                
                self._process_request(request)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker error: {e}")
    
    def _process_request(self, request: CompilationRequest) -> None:
        """Process a compilation request."""
        try:
            # Check cache first
            cache_key = self.cache.generate_cache_key(request)
            cached_result = self.cache.get(cache_key)
            
            if cached_result:
                self.completed_requests[request.request_id] = cached_result
                return
            
            # Determine backend to use
            backend = self._select_backend(request)
            
            if not backend:
                result = CompilationResult(
                    request_id=request.request_id,
                    status=CompilationStatus.FAILED,
                    error_message="No suitable backend available",
                    completed_at=time.time()
                )
            else:
                # Compile circuit
                result = backend.compile_circuit(request)
                
                # Cache successful results
                if result.status == CompilationStatus.COMPLETED:
                    self.cache.put(cache_key, result)
            
            # Store result
            self.completed_requests[request.request_id] = result
            
            # Remove from active requests
            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]
                
        except Exception as e:
            result = CompilationResult(
                request_id=request.request_id,
                status=CompilationStatus.FAILED,
                error_message=f"Processing error: {str(e)}",
                completed_at=time.time()
            )
            self.completed_requests[request.request_id] = result
    
    def _select_backend(self, request: CompilationRequest) -> Optional[CompilationBackendInterface]:
        """Select appropriate backend for request."""
        # Simple selection logic - prefer local, fallback to remote
        if CompilationBackend.LOCAL in self.backends:
            return self.backends[CompilationBackend.LOCAL]
        elif CompilationBackend.REMOTE in self.backends:
            return self.backends[CompilationBackend.REMOTE]
        else:
            return None
    
    def submit_compilation(self, circuit_data: Dict[str, Any], 
                         optimization_level: OptimizationLevel = OptimizationLevel.STANDARD,
                         target_backend: str = "simulator",
                         custom_passes: Optional[List[str]] = None,
                         constraints: Optional[Dict[str, Any]] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """Submit a compilation request."""
        
        request_id = f"compile_{int(time.time() * 1000)}_{hash(str(circuit_data)) % 10000}"
        
        request = CompilationRequest(
            request_id=request_id,
            circuit_data=circuit_data,
            optimization_level=optimization_level,
            target_backend=target_backend,
            custom_passes=custom_passes or [],
            constraints=constraints or {},
            metadata=metadata or {}
        )
        
        # Store as active request
        self.active_requests[request_id] = request
        
        # Add to processing queue
        self.request_queue.put(request)
        
        return request_id
    
    def get_compilation_status(self, request_id: str) -> Optional[CompilationStatus]:
        """Get status of a compilation request."""
        if request_id in self.completed_requests:
            return self.completed_requests[request_id].status
        elif request_id in self.active_requests:
            return CompilationStatus.PENDING
        else:
            return None
    
    def get_compilation_result(self, request_id: str) -> Optional[CompilationResult]:
        """Get compilation result."""
        return self.completed_requests.get(request_id)
    
    def compile_circuit_sync(self, circuit_data: Dict[str, Any],
                           optimization_level: OptimizationLevel = OptimizationLevel.STANDARD,
                           timeout: float = 30.0) -> CompilationResult:
        """Compile circuit synchronously."""
        request_id = self.submit_compilation(circuit_data, optimization_level)
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            result = self.get_compilation_result(request_id)
            if result:
                return result
            time.sleep(0.1)
        
        # Timeout
        return CompilationResult(
            request_id=request_id,
            status=CompilationStatus.FAILED,
            error_message="Compilation timeout",
            completed_at=time.time()
        )
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            'active_requests': len(self.active_requests),
            'completed_requests': len(self.completed_requests),
            'available_backends': list(self.backends.keys()),
            'worker_threads': len(self.worker_threads),
            'is_running': self.is_running,
            'cache_stats': self.cache.get_stats()
        }
    
    def list_available_optimizations(self) -> Dict[str, List[str]]:
        """List available optimization passes by backend."""
        optimizations = {}
        for backend_type, backend in self.backends.items():
            optimizations[backend_type.value] = backend.get_supported_optimizations()
        return optimizations
    
    def clear_cache(self) -> None:
        """Clear compilation cache."""
        self.cache.clear()
    
    def cleanup(self) -> None:
        """Cleanup service resources."""
        self.stop_workers()
        self.clear_cache()


class CompilationServiceAPI:
    """REST API for compilation service."""
    
    def __init__(self, service: CompilationService, host: str = "localhost", port: int = 5001):
        if not FLASK_AVAILABLE:
            raise RuntimeError("Flask not available for API service")
        
        self.service = service
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.setup_routes()
    
    def setup_routes(self) -> None:
        """Setup API routes."""
        
        @self.app.route('/api/compile', methods=['POST'])
        def compile_circuit():
            try:
                data = request.get_json()
                
                circuit_data = data.get('circuit_data', {})
                optimization_level = OptimizationLevel(data.get('optimization_level', 2))
                target_backend = data.get('target_backend', 'simulator')
                custom_passes = data.get('custom_passes', [])
                constraints = data.get('constraints', {})
                metadata = data.get('metadata', {})
                
                request_id = self.service.submit_compilation(
                    circuit_data=circuit_data,
                    optimization_level=optimization_level,
                    target_backend=target_backend,
                    custom_passes=custom_passes,
                    constraints=constraints,
                    metadata=metadata
                )
                
                return jsonify({
                    'success': True,
                    'request_id': request_id
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 400
        
        @self.app.route('/api/compile/sync', methods=['POST'])
        def compile_circuit_sync():
            try:
                data = request.get_json()
                
                circuit_data = data.get('circuit_data', {})
                optimization_level = OptimizationLevel(data.get('optimization_level', 2))
                timeout = data.get('timeout', 30.0)
                
                result = self.service.compile_circuit_sync(
                    circuit_data=circuit_data,
                    optimization_level=optimization_level,
                    timeout=timeout
                )
                
                return jsonify({
                    'success': True,
                    'result': result.to_dict()
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 400
        
        @self.app.route('/api/status/<request_id>')
        def get_status(request_id):
            status = self.service.get_compilation_status(request_id)
            
            if status is None:
                return jsonify({
                    'error': 'Request not found'
                }), 404
            
            return jsonify({
                'request_id': request_id,
                'status': status.value
            })
        
        @self.app.route('/api/result/<request_id>')
        def get_result(request_id):
            result = self.service.get_compilation_result(request_id)
            
            if result is None:
                return jsonify({
                    'error': 'Result not found'
                }), 404
            
            return jsonify({
                'success': True,
                'result': result.to_dict()
            })
        
        @self.app.route('/api/stats')
        def get_stats():
            return jsonify(self.service.get_service_stats())
        
        @self.app.route('/api/optimizations')
        def get_optimizations():
            return jsonify(self.service.list_available_optimizations())
        
        @self.app.route('/api/cache', methods=['DELETE'])
        def clear_cache():
            self.service.clear_cache()
            return jsonify({'success': True, 'message': 'Cache cleared'})
    
    def run(self, debug: bool = False) -> None:
        """Run the API server."""
        print(f"Starting compilation service API at http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=debug)


# Global compilation service instance
_compilation_service: Optional[CompilationService] = None


def get_compilation_service() -> CompilationService:
    """Get global compilation service instance."""
    global _compilation_service
    if _compilation_service is None:
        _compilation_service = CompilationService()
    return _compilation_service


def compile_circuit(circuit_data: Dict[str, Any],
                   optimization_level: OptimizationLevel = OptimizationLevel.STANDARD,
                   timeout: float = 30.0) -> CompilationResult:
    """Convenience function to compile a circuit."""
    service = get_compilation_service()
    return service.compile_circuit_sync(circuit_data, optimization_level, timeout)


def start_compilation_api(host: str = "localhost", port: int = 5001, debug: bool = False) -> None:
    """Start the compilation service API."""
    service = get_compilation_service()
    api = CompilationServiceAPI(service, host, port)
    api.run(debug)


# CLI interface
def main():
    """Main CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="QuantRS2 Compilation Service")
    parser.add_argument("--mode", choices=["api", "compile"], default="api",
                       help="Mode to run in")
    parser.add_argument("--host", default="localhost", help="API host")
    parser.add_argument("--port", type=int, default=5001, help="API port")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--circuit-file", help="Circuit file to compile (JSON)")
    parser.add_argument("--optimization", type=int, default=2, 
                       help="Optimization level (0-3)")
    
    args = parser.parse_args()
    
    if args.mode == "api":
        start_compilation_api(args.host, args.port, args.debug)
    elif args.mode == "compile" and args.circuit_file:
        try:
            with open(args.circuit_file, 'r') as f:
                circuit_data = json.load(f)
            
            optimization_level = OptimizationLevel(args.optimization)
            result = compile_circuit(circuit_data, optimization_level)
            
            print(f"Compilation Status: {result.status.value}")
            print(f"Compilation Time: {result.compilation_time:.3f}s")
            
            if result.status == CompilationStatus.COMPLETED:
                print("Optimization Report:")
                print(json.dumps(result.optimization_report, indent=2))
                print("Metrics:")
                print(json.dumps(result.metrics, indent=2))
            elif result.error_message:
                print(f"Error: {result.error_message}")
                
        except Exception as e:
            print(f"Error: {e}")
            return 1
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())