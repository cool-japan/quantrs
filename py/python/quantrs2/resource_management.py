"""
Advanced Resource Management for QuantRS2

This module provides comprehensive resource monitoring, limiting, and management
for quantum circuit execution with production-grade resource controls.
"""

import os
import psutil
import threading
import time
import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError
from contextlib import contextmanager
import weakref
import gc
import tracemalloc
from collections import deque, defaultdict
import signal

logger = logging.getLogger(__name__)

class ResourceType(Enum):
    """Types of resources that can be monitored and limited."""
    MEMORY = "memory"
    CPU = "cpu"
    TIME = "time"
    DISK = "disk"
    NETWORK = "network"
    QUBITS = "qubits"
    GATES = "gates"
    DEPTH = "depth"

class ResourceStatus(Enum):
    """Status of resource usage."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EXCEEDED = "exceeded"

@dataclass
class ResourceLimit:
    """Resource limit configuration."""
    resource_type: ResourceType
    soft_limit: float  # Warning threshold
    hard_limit: float  # Hard blocking threshold
    unit: str = ""
    enabled: bool = True
    
    def check_usage(self, current_usage: float) -> ResourceStatus:
        """Check current usage against limits."""
        if not self.enabled:
            return ResourceStatus.NORMAL
            
        if current_usage >= self.hard_limit:
            return ResourceStatus.EXCEEDED
        elif current_usage >= self.soft_limit:
            return ResourceStatus.CRITICAL
        elif current_usage >= self.soft_limit * 0.8:
            return ResourceStatus.WARNING
        else:
            return ResourceStatus.NORMAL

@dataclass
class ResourceUsage:
    """Current resource usage information."""
    resource_type: ResourceType
    current_usage: float
    peak_usage: float
    average_usage: float
    samples_count: int
    unit: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def update(self, new_usage: float) -> None:
        """Update usage statistics."""
        self.current_usage = new_usage
        self.peak_usage = max(self.peak_usage, new_usage)
        self.average_usage = (self.average_usage * self.samples_count + new_usage) / (self.samples_count + 1)
        self.samples_count += 1
        self.timestamp = time.time()

@dataclass
class ResourceConfig:
    """Configuration for resource management."""
    
    # Memory limits (in MB)
    max_memory_mb: float = 4096.0
    memory_warning_threshold: float = 0.8
    enable_memory_monitoring: bool = True
    
    # CPU limits (percentage)
    max_cpu_percent: float = 80.0
    cpu_warning_threshold: float = 0.8
    enable_cpu_monitoring: bool = True
    
    # Execution time limits (seconds)
    max_execution_time: float = 300.0
    execution_warning_time: float = 240.0
    enable_time_monitoring: bool = True
    
    # Circuit complexity limits
    max_qubits: int = 50
    max_gates: int = 10000
    max_circuit_depth: int = 1000
    enable_circuit_monitoring: bool = True
    
    # Resource monitoring
    monitoring_interval: float = 1.0  # seconds
    resource_history_size: int = 1000
    enable_gc_monitoring: bool = True
    
    # Resource pool
    max_concurrent_executions: int = 4
    queue_timeout: float = 60.0
    enable_priority_scheduling: bool = True
    
    def to_limits(self) -> Dict[ResourceType, ResourceLimit]:
        """Convert config to resource limits."""
        return {
            ResourceType.MEMORY: ResourceLimit(
                ResourceType.MEMORY,
                self.max_memory_mb * self.memory_warning_threshold,
                self.max_memory_mb,
                "MB",
                self.enable_memory_monitoring
            ),
            ResourceType.CPU: ResourceLimit(
                ResourceType.CPU,
                self.max_cpu_percent * self.cpu_warning_threshold,
                self.max_cpu_percent,
                "%",
                self.enable_cpu_monitoring
            ),
            ResourceType.TIME: ResourceLimit(
                ResourceType.TIME,
                self.execution_warning_time,
                self.max_execution_time,
                "seconds",
                self.enable_time_monitoring
            ),
            ResourceType.QUBITS: ResourceLimit(
                ResourceType.QUBITS,
                self.max_qubits * 0.8,
                self.max_qubits,
                "count",
                self.enable_circuit_monitoring
            ),
            ResourceType.GATES: ResourceLimit(
                ResourceType.GATES,
                self.max_gates * 0.8,
                self.max_gates,
                "count",
                self.enable_circuit_monitoring
            ),
            ResourceType.DEPTH: ResourceLimit(
                ResourceType.DEPTH,
                self.max_circuit_depth * 0.8,
                self.max_circuit_depth,
                "layers",
                self.enable_circuit_monitoring
            )
        }

class ResourceException(Exception):
    """Exception raised when resource limits are exceeded."""
    
    def __init__(self, message: str, resource_type: ResourceType, 
                 current_usage: float, limit: float):
        super().__init__(message)
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limit = limit

class ResourceMonitor:
    """Real-time resource monitoring and enforcement."""
    
    def __init__(self, config: ResourceConfig):
        self.config = config
        self.limits = config.to_limits()
        self.usage_history: Dict[ResourceType, deque] = defaultdict(
            lambda: deque(maxlen=config.resource_history_size)
        )
        self.current_usage: Dict[ResourceType, ResourceUsage] = {}
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Process tracking
        self._process = psutil.Process()
        self._start_time = time.time()
        
        # Memory tracking
        if config.enable_memory_monitoring:
            tracemalloc.start()
        
        # Active execution tracking
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Resource monitor initialized")
    
    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="ResourceMonitor"
        )
        self._monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Resource monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                self._collect_metrics()
                time.sleep(self.config.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.config.monitoring_interval)
    
    def _collect_metrics(self) -> None:
        """Collect current resource metrics."""
        with self._lock:
            # Memory usage
            if self.config.enable_memory_monitoring:
                memory_mb = self._process.memory_info().rss / 1024 / 1024
                self._update_usage(ResourceType.MEMORY, memory_mb, "MB")
            
            # CPU usage
            if self.config.enable_cpu_monitoring:
                cpu_percent = self._process.cpu_percent()
                self._update_usage(ResourceType.CPU, cpu_percent, "%")
            
            # Update execution times for active executions
            current_time = time.time()
            for exec_id, exec_info in self.active_executions.items():
                elapsed = current_time - exec_info['start_time']
                self._update_usage(ResourceType.TIME, elapsed, "seconds")
    
    def _update_usage(self, resource_type: ResourceType, value: float, unit: str) -> None:
        """Update usage statistics for a resource type."""
        if resource_type not in self.current_usage:
            self.current_usage[resource_type] = ResourceUsage(
                resource_type=resource_type,
                current_usage=value,
                peak_usage=value,
                average_usage=value,
                samples_count=1,
                unit=unit
            )
        else:
            self.current_usage[resource_type].update(value)
        
        # Add to history
        self.usage_history[resource_type].append((time.time(), value))
    
    def check_resource_limits(self, resource_type: Optional[ResourceType] = None) -> Dict[ResourceType, ResourceStatus]:
        """Check current usage against limits."""
        with self._lock:
            results = {}
            
            resource_types = [resource_type] if resource_type else list(self.limits.keys())
            
            for rt in resource_types:
                if rt in self.current_usage and rt in self.limits:
                    limit = self.limits[rt]
                    usage = self.current_usage[rt]
                    status = limit.check_usage(usage.current_usage)
                    results[rt] = status
                    
                    # Log warnings and errors
                    if status in [ResourceStatus.CRITICAL, ResourceStatus.EXCEEDED]:
                        logger.warning(
                            f"Resource {rt.value} usage {usage.current_usage}{usage.unit} "
                            f"exceeded {'hard' if status == ResourceStatus.EXCEEDED else 'soft'} "
                            f"limit {limit.hard_limit if status == ResourceStatus.EXCEEDED else limit.soft_limit}"
                        )
            
            return results
    
    def enforce_limits(self, resource_type: Optional[ResourceType] = None) -> None:
        """Enforce resource limits, raising exceptions if exceeded."""
        status_map = self.check_resource_limits(resource_type)
        
        for rt, status in status_map.items():
            if status == ResourceStatus.EXCEEDED:
                usage = self.current_usage[rt]
                limit = self.limits[rt]
                raise ResourceException(
                    f"Resource limit exceeded for {rt.value}: "
                    f"{usage.current_usage}{usage.unit} > {limit.hard_limit}{usage.unit}",
                    rt,
                    usage.current_usage,
                    limit.hard_limit
                )
    
    def get_resource_usage(self, resource_type: ResourceType) -> Optional[ResourceUsage]:
        """Get current usage for a specific resource type."""
        with self._lock:
            return self.current_usage.get(resource_type)
    
    def get_all_usage(self) -> Dict[ResourceType, ResourceUsage]:
        """Get current usage for all monitored resources."""
        with self._lock:
            return self.current_usage.copy()
    
    def register_execution(self, execution_id: str, circuit_info: Dict[str, Any]) -> None:
        """Register a new execution for tracking."""
        with self._lock:
            self.active_executions[execution_id] = {
                'start_time': time.time(),
                'circuit_info': circuit_info
            }
            
            # Check circuit complexity limits
            if self.config.enable_circuit_monitoring:
                qubits = circuit_info.get('qubits', 0)
                gates = circuit_info.get('gates', 0)
                depth = circuit_info.get('depth', 0)
                
                if qubits > 0:
                    self._update_usage(ResourceType.QUBITS, qubits, "count")
                if gates > 0:
                    self._update_usage(ResourceType.GATES, gates, "count")
                if depth > 0:
                    self._update_usage(ResourceType.DEPTH, depth, "layers")
    
    def unregister_execution(self, execution_id: str) -> None:
        """Unregister a completed execution."""
        with self._lock:
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
    
    def get_resource_history(self, resource_type: ResourceType, 
                           duration_seconds: Optional[float] = None) -> List[Tuple[float, float]]:
        """Get resource usage history."""
        with self._lock:
            history = list(self.usage_history[resource_type])
            
            if duration_seconds:
                cutoff_time = time.time() - duration_seconds
                history = [(t, v) for t, v in history if t >= cutoff_time]
            
            return history
    
    def force_garbage_collection(self) -> Dict[str, Any]:
        """Force garbage collection and return memory stats."""
        if not self.config.enable_gc_monitoring:
            return {}
        
        # Get pre-GC stats
        pre_memory = self._process.memory_info().rss / 1024 / 1024
        
        # Force GC
        collected = gc.collect()
        
        # Get post-GC stats
        post_memory = self._process.memory_info().rss / 1024 / 1024
        memory_freed = pre_memory - post_memory
        
        stats = {
            'objects_collected': collected,
            'memory_before_mb': pre_memory,
            'memory_after_mb': post_memory,
            'memory_freed_mb': memory_freed,
            'gc_stats': gc.get_stats() if hasattr(gc, 'get_stats') else None
        }
        
        logger.info(f"Garbage collection freed {memory_freed:.2f}MB, collected {collected} objects")
        return stats
    
    def get_system_resources(self) -> Dict[str, Any]:
        """Get system-wide resource information."""
        try:
            return {
                'cpu_count': psutil.cpu_count(),
                'cpu_percent': psutil.cpu_percent(),
                'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
                'memory_available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage_gb': psutil.disk_usage('/').used / 1024 / 1024 / 1024,
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
            }
        except Exception as e:
            logger.error(f"Error getting system resources: {e}")
            return {}

class ResourcePool:
    """Resource pool for managing concurrent quantum circuit executions."""
    
    def __init__(self, config: ResourceConfig, monitor: ResourceMonitor):
        self.config = config
        self.monitor = monitor
        
        # Execution pool
        self.executor = ThreadPoolExecutor(
            max_workers=config.max_concurrent_executions,
            thread_name_prefix="QuantumExecution"
        )
        
        # Execution queue and tracking
        self.execution_queue: deque = deque()
        self.active_futures: Dict[str, Future] = {}
        self.completed_futures: Dict[str, Future] = {}
        
        # Priority queues
        self.high_priority_queue: deque = deque()
        self.normal_priority_queue: deque = deque()
        self.low_priority_queue: deque = deque()
        
        # Synchronization
        self._lock = threading.RLock()
        self._queue_condition = threading.Condition(self._lock)
        
        logger.info("Resource pool initialized")
    
    def submit_execution(self, execution_id: str, circuit_info: Dict[str, Any],
                        execution_func: Callable, priority: str = "normal") -> Future:
        """Submit a circuit execution to the resource pool."""
        
        # Check resource availability before queuing
        try:
            self.monitor.enforce_limits()
        except ResourceException as e:
            logger.error(f"Cannot submit execution {execution_id}: {e}")
            raise
        
        # Register execution for monitoring
        self.monitor.register_execution(execution_id, circuit_info)
        
        # Create wrapped execution function with resource monitoring
        def monitored_execution():
            try:
                return self._execute_with_monitoring(execution_id, execution_func)
            finally:
                self.monitor.unregister_execution(execution_id)
        
        # Submit to appropriate queue based on priority
        with self._lock:
            future = self.executor.submit(monitored_execution)
            self.active_futures[execution_id] = future
            
            # Add to priority queue
            if priority == "high":
                self.high_priority_queue.append((execution_id, future))
            elif priority == "low":
                self.low_priority_queue.append((execution_id, future))
            else:
                self.normal_priority_queue.append((execution_id, future))
            
            self._queue_condition.notify()
            
            logger.info(f"Execution {execution_id} submitted with {priority} priority")
            return future
    
    def _execute_with_monitoring(self, execution_id: str, execution_func: Callable) -> Any:
        """Execute function with continuous resource monitoring."""
        start_time = time.time()
        
        try:
            # Periodic resource checks during execution
            def check_resources():
                while execution_id in self.active_futures:
                    try:
                        self.monitor.enforce_limits()
                        time.sleep(self.config.monitoring_interval)
                    except ResourceException:
                        # Cancel execution if resources exceeded
                        if execution_id in self.active_futures:
                            self.active_futures[execution_id].cancel()
                        raise
                    except Exception as e:
                        logger.error(f"Error in resource monitoring for {execution_id}: {e}")
                        break
            
            # Start monitoring thread
            monitor_thread = threading.Thread(
                target=check_resources,
                daemon=True,
                name=f"ResourceMonitor-{execution_id}"
            )
            monitor_thread.start()
            
            # Execute the actual function
            result = execution_func()
            
            return result
            
        finally:
            # Clean up
            with self._lock:
                if execution_id in self.active_futures:
                    future = self.active_futures.pop(execution_id)
                    self.completed_futures[execution_id] = future
            
            execution_time = time.time() - start_time
            logger.info(f"Execution {execution_id} completed in {execution_time:.2f}s")
    
    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a pending or active execution."""
        with self._lock:
            if execution_id in self.active_futures:
                future = self.active_futures[execution_id]
                cancelled = future.cancel()
                
                if cancelled:
                    del self.active_futures[execution_id]
                    self.monitor.unregister_execution(execution_id)
                    logger.info(f"Execution {execution_id} cancelled")
                
                return cancelled
        
        return False
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        with self._lock:
            return {
                'active_executions': len(self.active_futures),
                'high_priority_queue': len(self.high_priority_queue),
                'normal_priority_queue': len(self.normal_priority_queue),
                'low_priority_queue': len(self.low_priority_queue),
                'completed_executions': len(self.completed_futures),
                'max_workers': self.executor._max_workers,
                'resource_usage': self.monitor.get_all_usage()
            }
    
    def wait_for_capacity(self, timeout: Optional[float] = None) -> bool:
        """Wait for execution capacity to become available."""
        with self._queue_condition:
            start_time = time.time()
            
            while len(self.active_futures) >= self.config.max_concurrent_executions:
                remaining_timeout = None
                if timeout:
                    elapsed = time.time() - start_time
                    remaining_timeout = max(0, timeout - elapsed)
                    if remaining_timeout <= 0:
                        return False
                
                if not self._queue_condition.wait(remaining_timeout):
                    return False
            
            return True
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the resource pool."""
        logger.info("Shutting down resource pool")
        
        # Cancel all pending executions
        with self._lock:
            for execution_id in list(self.active_futures.keys()):
                self.cancel_execution(execution_id)
        
        # Shutdown executor
        self.executor.shutdown(wait=wait)
        
        logger.info("Resource pool shutdown complete")

@contextmanager
def resource_context(config: ResourceConfig):
    """Context manager for resource monitoring."""
    monitor = ResourceMonitor(config)
    pool = ResourcePool(config, monitor)
    
    try:
        monitor.start_monitoring()
        yield monitor, pool
    finally:
        monitor.stop_monitoring()
        pool.shutdown()

# Utilities for circuit resource analysis
def analyze_circuit_resources(circuit: Any) -> Dict[str, Any]:
    """Analyze resource requirements of a quantum circuit."""
    try:
        # Basic circuit analysis
        analysis = {
            'qubits': 0,
            'gates': 0,
            'depth': 0,
            'estimated_memory_mb': 0,
            'estimated_time_seconds': 0
        }
        
        # Extract basic properties (implementation depends on circuit type)
        if hasattr(circuit, 'num_qubits'):
            analysis['qubits'] = circuit.num_qubits
        elif hasattr(circuit, 'n_qubits'):
            analysis['qubits'] = circuit.n_qubits
        
        if hasattr(circuit, 'size'):
            analysis['gates'] = circuit.size()
        elif hasattr(circuit, 'count_ops'):
            analysis['gates'] = sum(circuit.count_ops().values())
        
        if hasattr(circuit, 'depth'):
            analysis['depth'] = circuit.depth()
        
        # Estimate memory requirements (exponential in qubits for simulation)
        if analysis['qubits'] > 0:
            # State vector simulation: 2^n complex numbers * 16 bytes
            analysis['estimated_memory_mb'] = (2 ** analysis['qubits']) * 16 / 1024 / 1024
        
        # Estimate execution time (rough heuristic)
        analysis['estimated_time_seconds'] = analysis['gates'] * 0.001 + analysis['qubits'] * 0.1
        
        return analysis
        
    except Exception as e:
        logger.error(f"Circuit analysis failed: {e}")
        return {
            'qubits': 0,
            'gates': 0,
            'depth': 0,
            'estimated_memory_mb': 0,
            'estimated_time_seconds': 0,
            'analysis_error': str(e)
        }

# Export components
__all__ = [
    # Enums
    "ResourceType",
    "ResourceStatus",
    
    # Data classes
    "ResourceLimit",
    "ResourceUsage", 
    "ResourceConfig",
    
    # Exceptions
    "ResourceException",
    
    # Main classes
    "ResourceMonitor",
    "ResourcePool",
    
    # Utilities
    "resource_context",
    "analyze_circuit_resources",
]