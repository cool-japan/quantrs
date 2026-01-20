"""
Performance Manager for QuantRS2 - Unified Connection Pooling and Caching

This module provides a unified interface for managing database connections,
caching strategies, and performance optimization across the QuantRS2 framework.
"""

import time
import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .connection_pooling import (
    DatabaseConnectionPool, QuantumResultCache, 
    ConnectionPoolConfig, CacheConfig, CacheBackend
)
from .circuit_optimization_cache import (
    CircuitOptimizationCache, OptimizationLevel, CircuitPattern
)
from .config_management import get_config_manager

logger = logging.getLogger(__name__)


class PerformanceProfile(Enum):
    """Performance optimization profiles."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"
    HIGH_PERFORMANCE = "high_performance"
    LOW_MEMORY = "low_memory"


@dataclass
class PerformanceConfig:
    """Unified performance configuration."""
    profile: PerformanceProfile = PerformanceProfile.PRODUCTION
    
    # Connection pooling
    max_db_connections: int = 20
    min_db_connections: int = 5
    connection_timeout: float = 30.0
    
    # Caching
    cache_backend: CacheBackend = CacheBackend.HYBRID
    max_cache_memory_mb: float = 1024.0
    max_cache_entries: int = 10000
    cache_ttl: float = 3600.0
    enable_compression: bool = True
    
    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Performance optimization
    enable_circuit_optimization: bool = True
    default_optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
    enable_pattern_detection: bool = True
    enable_execution_profiling: bool = True
    
    # Monitoring
    enable_performance_monitoring: bool = True
    monitoring_interval: float = 60.0
    enable_metrics_collection: bool = True
    
    @classmethod
    def for_profile(cls, profile: PerformanceProfile) -> 'PerformanceConfig':
        """Create configuration for specific performance profile."""
        config = cls(profile=profile)
        
        if profile == PerformanceProfile.DEVELOPMENT:
            config.max_db_connections = 5
            config.max_cache_memory_mb = 256.0
            config.max_cache_entries = 1000
            config.cache_backend = CacheBackend.MEMORY
            config.enable_compression = False
            config.monitoring_interval = 300.0
            
        elif profile == PerformanceProfile.TESTING:
            config.max_db_connections = 10
            config.max_cache_memory_mb = 512.0
            config.max_cache_entries = 5000
            config.cache_backend = CacheBackend.MEMORY
            config.enable_compression = True
            config.monitoring_interval = 120.0
            
        elif profile == PerformanceProfile.PRODUCTION:
            config.max_db_connections = 20
            config.max_cache_memory_mb = 1024.0
            config.max_cache_entries = 10000
            config.cache_backend = CacheBackend.HYBRID
            config.enable_compression = True
            config.monitoring_interval = 60.0
            
        elif profile == PerformanceProfile.HIGH_PERFORMANCE:
            config.max_db_connections = 50
            config.max_cache_memory_mb = 4096.0
            config.max_cache_entries = 50000
            config.cache_backend = CacheBackend.REDIS
            config.enable_compression = False  # Trade memory for speed
            config.monitoring_interval = 30.0
            
        elif profile == PerformanceProfile.LOW_MEMORY:
            config.max_db_connections = 5
            config.max_cache_memory_mb = 128.0
            config.max_cache_entries = 1000
            config.cache_backend = CacheBackend.SQLITE
            config.enable_compression = True
            config.monitoring_interval = 300.0
        
        return config


class ConnectionManager:
    """Manages database connections across QuantRS2 components."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self._pools: Dict[str, DatabaseConnectionPool] = {}
        self._lock = threading.RLock()
        
        # Initialize default pools
        self._initialize_default_pools()
    
    def _initialize_default_pools(self):
        """Initialize default database connection pools."""
        try:
            # Circuit database pool
            circuit_db_path = Path.home() / '.quantrs2' / 'circuits.db'
            self.register_pool('circuits', str(circuit_db_path))
            
            # Analysis cache database pool
            analysis_db_path = Path.home() / '.quantrs2' / 'analysis_cache.db'
            self.register_pool('analysis', str(analysis_db_path))
            
            # Performance metrics database pool
            metrics_db_path = Path.home() / '.quantrs2' / 'metrics.db'
            self.register_pool('metrics', str(metrics_db_path))
            
            logger.info(f"Initialized {len(self._pools)} database connection pools")
            
        except Exception as e:
            logger.error(f"Failed to initialize default connection pools: {e}")
    
    def register_pool(self, name: str, db_path: str, 
                     pool_config: Optional[ConnectionPoolConfig] = None) -> bool:
        """Register a new database connection pool."""
        try:
            with self._lock:
                if name in self._pools:
                    logger.warning(f"Connection pool '{name}' already exists")
                    return False
                
                # Create pool configuration
                if pool_config is None:
                    pool_config = ConnectionPoolConfig(
                        max_connections=self.config.max_db_connections,
                        min_connections=self.config.min_db_connections,
                        connection_timeout=self.config.connection_timeout
                    )
                
                # Create and register pool
                pool = DatabaseConnectionPool(db_path, pool_config)
                self._pools[name] = pool
                
                logger.info(f"Registered connection pool '{name}' for {db_path}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to register connection pool '{name}': {e}")
            return False
    
    def get_pool(self, name: str) -> Optional[DatabaseConnectionPool]:
        """Get connection pool by name."""
        with self._lock:
            return self._pools.get(name)
    
    @contextmanager
    def get_connection(self, pool_name: str):
        """Get database connection from named pool."""
        pool = self.get_pool(pool_name)
        if pool is None:
            raise ValueError(f"Connection pool '{pool_name}' not found")
        
        with pool.get_connection() as conn:
            yield conn
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for all connection pools."""
        stats = {}
        with self._lock:
            for name, pool in self._pools.items():
                stats[name] = pool.get_statistics()
        return stats
    
    def close_all(self):
        """Close all connection pools."""
        with self._lock:
            for name, pool in self._pools.items():
                try:
                    pool.close()
                    logger.debug(f"Closed connection pool '{name}'")
                except Exception as e:
                    logger.error(f"Error closing pool '{name}': {e}")
            
            self._pools.clear()


class CacheManager:
    """Manages caching systems across QuantRS2 components."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        
        # Initialize cache configuration
        cache_config = CacheConfig(
            backend=config.cache_backend,
            max_memory_mb=config.max_cache_memory_mb,
            max_entries=config.max_cache_entries,
            default_ttl=config.cache_ttl,
            compression=config.enable_compression,
            redis_host=config.redis_host,
            redis_port=config.redis_port,
            redis_db=config.redis_db
        )
        
        # Initialize caches
        self.circuit_cache = CircuitOptimizationCache(cache_config)
        self.general_cache = QuantumResultCache(cache_config)
        
        logger.info("Cache manager initialized with circuit optimization and general caching")
    
    def get_circuit_cache(self) -> CircuitOptimizationCache:
        """Get circuit optimization cache."""
        return self.circuit_cache
    
    def get_general_cache(self) -> QuantumResultCache:
        """Get general purpose cache."""
        return self.general_cache
    
    def invalidate_all(self):
        """Invalidate all cache entries."""
        try:
            self.circuit_cache.result_cache.clear()
            self.circuit_cache.optimization_cache.clear()
            self.general_cache.clear()
            logger.info("All cache entries invalidated")
            
        except Exception as e:
            logger.error(f"Failed to invalidate all caches: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        return {
            'circuit_optimization': self.circuit_cache.get_statistics(),
            'general_cache': self.general_cache.get_statistics()
        }
    
    def close_all(self):
        """Close all cache systems."""
        try:
            self.circuit_cache.close()
            self.general_cache.close()
            logger.info("All cache systems closed")
            
        except Exception as e:
            logger.error(f"Error closing cache systems: {e}")


class PerformanceMonitor:
    """Monitors and reports performance metrics."""
    
    def __init__(self, config: PerformanceConfig, 
                 connection_manager: ConnectionManager,
                 cache_manager: CacheManager):
        self.config = config
        self.connection_manager = connection_manager
        self.cache_manager = cache_manager
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread = None
        self._shutdown = threading.Event()
        
        # Metrics storage
        self._metrics_history: List[Dict[str, Any]] = []
        self._metrics_lock = threading.RLock()
        
        if config.enable_performance_monitoring:
            self.start_monitoring()
    
    def start_monitoring(self):
        """Start performance monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        self._shutdown.set()
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self._shutdown.wait(self.config.monitoring_interval):
            try:
                metrics = self._collect_metrics()
                
                with self._metrics_lock:
                    self._metrics_history.append(metrics)
                    
                    # Keep only last 24 hours of metrics
                    max_entries = int(86400 / self.config.monitoring_interval)
                    if len(self._metrics_history) > max_entries:
                        self._metrics_history = self._metrics_history[-max_entries:]
                
                # Log warning if performance issues detected
                self._check_performance_issues(metrics)
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current performance metrics."""
        timestamp = time.time()
        
        # Connection pool metrics
        connection_stats = self.connection_manager.get_statistics()
        
        # Cache metrics
        cache_stats = self.cache_manager.get_statistics()
        
        # System metrics
        import psutil
        system_metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage_percent': psutil.disk_usage('/').percent
        }
        
        return {
            'timestamp': timestamp,
            'connections': connection_stats,
            'cache': cache_stats,
            'system': system_metrics
        }
    
    def _check_performance_issues(self, metrics: Dict[str, Any]):
        """Check for performance issues and log warnings."""
        # Check connection pool utilization
        for pool_name, pool_stats in metrics['connections'].items():
            utilization = pool_stats.get('pool_utilization', 0)
            if utilization > 0.9:
                logger.warning(f"Connection pool '{pool_name}' utilization high: {utilization:.1%}")
        
        # Check cache hit rates
        circuit_cache_stats = metrics['cache']['circuit_optimization']['result_cache']
        if circuit_cache_stats.get('hit_rate', 1.0) < 0.5:
            logger.warning(f"Circuit cache hit rate low: {circuit_cache_stats.get('hit_rate', 0):.1%}")
        
        # Check system resources
        system = metrics['system']
        if system['memory_percent'] > 90:
            logger.warning(f"System memory usage high: {system['memory_percent']:.1f}%")
        
        if system['cpu_percent'] > 90:
            logger.warning(f"System CPU usage high: {system['cpu_percent']:.1f}%")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self._collect_metrics()
    
    def get_metrics_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get metrics history for specified hours."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self._metrics_lock:
            return [
                m for m in self._metrics_history 
                if m['timestamp'] >= cutoff_time
            ]


class PerformanceManager:
    """Unified performance management for QuantRS2."""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        # Load configuration
        if config is None:
            config_manager = get_config_manager()
            current_config = config_manager.get_current_config()
            
            # Determine profile from environment
            env = current_config.environment
            if env.name == "development":
                profile = PerformanceProfile.DEVELOPMENT
            elif env.name == "testing":
                profile = PerformanceProfile.TESTING
            else:
                profile = PerformanceProfile.PRODUCTION
            
            config = PerformanceConfig.for_profile(profile)
        
        self.config = config
        
        # Initialize managers
        self.connection_manager = ConnectionManager(config)
        self.cache_manager = CacheManager(config)
        
        # Initialize monitoring
        self.monitor = None
        if config.enable_performance_monitoring:
            self.monitor = PerformanceMonitor(config, self.connection_manager, self.cache_manager)
        
        logger.info(f"Performance manager initialized with profile: {config.profile.value}")
    
    @contextmanager
    def database_connection(self, pool_name: str = 'circuits'):
        """Get database connection (context manager)."""
        with self.connection_manager.get_connection(pool_name) as conn:
            yield conn
    
    def get_circuit_cache(self) -> CircuitOptimizationCache:
        """Get circuit optimization cache."""
        return self.cache_manager.get_circuit_cache()
    
    def get_general_cache(self) -> QuantumResultCache:
        """Get general purpose cache."""
        return self.cache_manager.get_general_cache()
    
    def optimize_performance(self):
        """Run performance optimization procedures."""
        try:
            # Optimize circuit cache
            self.cache_manager.circuit_cache.optimize_cache_performance()
            
            # Force garbage collection in caches
            self.cache_manager.general_cache._cleanup_expired_entries()
            
            logger.info("Performance optimization completed")
            
        except Exception as e:
            logger.error(f"Performance optimization failed: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            'config': {
                'profile': self.config.profile.value,
                'cache_backend': self.config.cache_backend.value,
                'max_connections': self.config.max_db_connections,
                'max_cache_memory_mb': self.config.max_cache_memory_mb
            },
            'connections': self.connection_manager.get_statistics(),
            'cache': self.cache_manager.get_statistics(),
            'current_metrics': None,
            'monitoring_enabled': self.monitor is not None
        }
        
        if self.monitor:
            report['current_metrics'] = self.monitor.get_current_metrics()
        
        return report
    
    def close(self):
        """Close performance manager and all resources."""
        logger.info("Closing performance manager")
        
        # Stop monitoring
        if self.monitor:
            self.monitor.stop_monitoring()
        
        # Close caches
        self.cache_manager.close_all()
        
        # Close connection pools
        self.connection_manager.close_all()
        
        logger.info("Performance manager closed")


# Global performance manager instance
_performance_manager: Optional[PerformanceManager] = None
_manager_lock = threading.RLock()


def get_performance_manager(config: Optional[PerformanceConfig] = None) -> PerformanceManager:
    """Get global performance manager instance."""
    global _performance_manager
    
    with _manager_lock:
        if _performance_manager is None:
            _performance_manager = PerformanceManager(config)
        
        return _performance_manager


def close_performance_manager():
    """Close global performance manager."""
    global _performance_manager
    
    with _manager_lock:
        if _performance_manager:
            _performance_manager.close()
            _performance_manager = None


# Export main classes
__all__ = [
    'PerformanceProfile',
    'PerformanceConfig',
    'ConnectionManager',
    'CacheManager',
    'PerformanceMonitor',
    'PerformanceManager',
    'get_performance_manager',
    'close_performance_manager'
]