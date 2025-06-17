"""
Advanced Connection Pooling and Caching Strategy for QuantRS2

This module provides production-grade connection pooling, result caching,
and optimization strategies for quantum computing applications.
"""

import time
import json
import pickle
import gzip
import hashlib
import sqlite3
import threading
import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future
from collections import defaultdict, OrderedDict
import weakref

# Optional dependencies
try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

try:
    import memcached
    HAS_MEMCACHED = True
except ImportError:
    HAS_MEMCACHED = False

logger = logging.getLogger(__name__)


class CacheBackend(Enum):
    """Available cache backends."""
    MEMORY = "memory"
    SQLITE = "sqlite"
    REDIS = "redis"
    MEMCACHED = "memcached"
    HYBRID = "hybrid"


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    FIFO = "fifo"         # First In, First Out
    TTL = "ttl"           # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on usage patterns


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self) -> None:
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class ConnectionPoolConfig:
    """Configuration for connection pools."""
    max_connections: int = 20
    min_connections: int = 5
    connection_timeout: float = 30.0
    idle_timeout: float = 300.0
    validation_query: str = "SELECT 1"
    retry_attempts: int = 3
    retry_delay: float = 1.0
    enable_monitoring: bool = True


@dataclass
class CacheConfig:
    """Configuration for caching system."""
    backend: CacheBackend = CacheBackend.HYBRID
    max_memory_mb: float = 1024.0
    max_entries: int = 10000
    default_ttl: float = 3600.0  # 1 hour
    strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    compression: bool = True
    enable_persistence: bool = True
    persistence_interval: float = 300.0  # 5 minutes
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    sqlite_path: Optional[str] = None


class DatabaseConnectionPool:
    """Production-grade database connection pool."""
    
    def __init__(self, db_path: str, config: Optional[ConnectionPoolConfig] = None):
        self.db_path = db_path
        self.config = config or ConnectionPoolConfig()
        
        # Connection pool
        self._available_connections = []
        self._used_connections = set()
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            'connections_created': 0,
            'connections_destroyed': 0,
            'connections_borrowed': 0,
            'connections_returned': 0,
            'connection_errors': 0,
            'pool_exhausted_count': 0
        }
        
        # Monitoring
        self._monitor_thread = None
        self._shutdown = threading.Event()
        
        # Initialize pool
        self._initialize_pool()
        
        if self.config.enable_monitoring:
            self._start_monitoring()
    
    def _initialize_pool(self):
        """Initialize the connection pool with minimum connections."""
        for _ in range(self.config.min_connections):
            try:
                conn = self._create_connection()
                self._available_connections.append(conn)
            except Exception as e:
                logger.error(f"Failed to initialize connection: {e}")
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection."""
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=self.config.connection_timeout,
                check_same_thread=False
            )
            
            # Configure connection
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL") 
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            
            self._stats['connections_created'] += 1
            logger.debug(f"Created new database connection to {self.db_path}")
            
            return conn
            
        except Exception as e:
            self._stats['connection_errors'] += 1
            logger.error(f"Failed to create database connection: {e}")
            raise
    
    def _validate_connection(self, conn: sqlite3.Connection) -> bool:
        """Validate a connection is still usable."""
        try:
            cursor = conn.execute(self.config.validation_query)
            cursor.fetchone()
            return True
        except Exception:
            return False
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool (context manager)."""
        conn = None
        try:
            conn = self._borrow_connection()
            yield conn
        finally:
            if conn:
                self._return_connection(conn)
    
    def _borrow_connection(self) -> sqlite3.Connection:
        """Borrow a connection from the pool."""
        with self._lock:
            # Try to get an available connection
            while self._available_connections:
                conn = self._available_connections.pop()
                
                # Validate connection
                if self._validate_connection(conn):
                    self._used_connections.add(conn)
                    self._stats['connections_borrowed'] += 1
                    return conn
                else:
                    # Connection is stale, close it
                    try:
                        conn.close()
                        self._stats['connections_destroyed'] += 1
                    except Exception:
                        pass
            
            # No available connections, create new one if under limit
            total_connections = len(self._available_connections) + len(self._used_connections)
            
            if total_connections < self.config.max_connections:
                conn = self._create_connection()
                self._used_connections.add(conn)
                self._stats['connections_borrowed'] += 1
                return conn
            else:
                # Pool exhausted
                self._stats['pool_exhausted_count'] += 1
                raise RuntimeError("Connection pool exhausted")
    
    def _return_connection(self, conn: sqlite3.Connection):
        """Return a connection to the pool."""
        with self._lock:
            if conn in self._used_connections:
                self._used_connections.remove(conn)
                
                # Validate before returning
                if self._validate_connection(conn):
                    self._available_connections.append(conn)
                    self._stats['connections_returned'] += 1
                else:
                    # Connection is broken, close it
                    try:
                        conn.close()
                        self._stats['connections_destroyed'] += 1
                    except Exception:
                        pass
    
    def _start_monitoring(self):
        """Start connection pool monitoring."""
        def monitor():
            while not self._shutdown.wait(60):  # Check every minute
                with self._lock:
                    # Clean up idle connections
                    current_time = time.time()
                    connections_to_remove = []
                    
                    for conn in self._available_connections:
                        # In a real implementation, we'd track last use time
                        # For now, just validate connections
                        if not self._validate_connection(conn):
                            connections_to_remove.append(conn)
                    
                    for conn in connections_to_remove:
                        self._available_connections.remove(conn)
                        try:
                            conn.close()
                            self._stats['connections_destroyed'] += 1
                        except Exception:
                            pass
                    
                    # Log statistics
                    total_connections = len(self._available_connections) + len(self._used_connections)
                    logger.debug(f"Connection pool stats: {total_connections} total, "
                               f"{len(self._available_connections)} available, "
                               f"{len(self._used_connections)} in use")
        
        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        with self._lock:
            return {
                **self._stats,
                'total_connections': len(self._available_connections) + len(self._used_connections),
                'available_connections': len(self._available_connections),
                'used_connections': len(self._used_connections),
                'pool_utilization': len(self._used_connections) / self.config.max_connections
            }
    
    def close(self):
        """Close the connection pool."""
        logger.info("Closing database connection pool")
        
        # Stop monitoring
        if self._monitor_thread:
            self._shutdown.set()
            self._monitor_thread.join(timeout=5)
        
        # Close all connections
        with self._lock:
            all_connections = list(self._available_connections) + list(self._used_connections)
            
            for conn in all_connections:
                try:
                    conn.close()
                    self._stats['connections_destroyed'] += 1
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")
            
            self._available_connections.clear()
            self._used_connections.clear()


class QuantumResultCache:
    """Advanced caching system for quantum computation results."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        
        # Memory cache
        self._memory_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._cache_lock = threading.RLock()
        
        # Cache backends
        self._redis_client = None
        self._memcached_client = None
        self._sqlite_cache = None
        
        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_usage_bytes': 0,
            'average_access_time_ms': 0.0
        }
        
        # Background tasks
        self._persistence_thread = None
        self._cleanup_thread = None
        self._shutdown = threading.Event()
        
        # Initialize backends
        self._initialize_backends()
        self._start_background_tasks()
    
    def _initialize_backends(self):
        """Initialize cache backends."""
        if self.config.backend in [CacheBackend.REDIS, CacheBackend.HYBRID]:
            if HAS_REDIS:
                try:
                    import redis
                    self._redis_client = redis.Redis(
                        host=self.config.redis_host,
                        port=self.config.redis_port,
                        db=self.config.redis_db,
                        decode_responses=False
                    )
                    self._redis_client.ping()
                    logger.info("Redis cache backend initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize Redis: {e}")
                    self._redis_client = None
            else:
                logger.warning("Redis not available, using memory cache only")
        
        if self.config.backend in [CacheBackend.SQLITE, CacheBackend.HYBRID]:
            try:
                cache_path = self.config.sqlite_path or ":memory:"
                self._sqlite_cache = DatabaseConnectionPool(cache_path)
                
                # Initialize cache table
                with self._sqlite_cache.get_connection() as conn:
                    conn.execute('''
                        CREATE TABLE IF NOT EXISTS cache_entries (
                            key TEXT PRIMARY KEY,
                            value BLOB,
                            created_at REAL,
                            last_accessed REAL,
                            access_count INTEGER,
                            size_bytes INTEGER,
                            ttl REAL,
                            tags TEXT
                        )
                    ''')
                    conn.execute('CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_entries(last_accessed)')
                    conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON cache_entries(created_at)')
                    conn.commit()
                
                logger.info("SQLite cache backend initialized")
                
            except Exception as e:
                logger.error(f"Failed to initialize SQLite cache: {e}")
                self._sqlite_cache = None
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage."""
        data = pickle.dumps(value)
        if self.config.compression:
            data = gzip.compress(data)
        return data
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        if self.config.compression:
            data = gzip.decompress(data)
        return pickle.loads(data)
    
    def _generate_key(self, circuit_hash: str, parameters: Dict[str, Any]) -> str:
        """Generate cache key from circuit and parameters."""
        param_str = json.dumps(parameters, sort_keys=True)
        content = f"{circuit_hash}:{param_str}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        start_time = time.time()
        
        try:
            # Check memory cache first
            with self._cache_lock:
                if key in self._memory_cache:
                    entry = self._memory_cache[key]
                    
                    if not entry.is_expired():
                        entry.touch()
                        # Move to end for LRU
                        self._memory_cache.move_to_end(key)
                        self._stats['hits'] += 1
                        
                        access_time = (time.time() - start_time) * 1000
                        self._update_average_access_time(access_time)
                        
                        return entry.value
                    else:
                        # Expired entry
                        del self._memory_cache[key]
                        self._stats['evictions'] += 1
            
            # Check Redis
            if self._redis_client:
                try:
                    data = self._redis_client.get(key)
                    if data:
                        value = self._deserialize_value(data)
                        
                        # Cache in memory for faster access
                        self._cache_in_memory(key, value)
                        
                        self._stats['hits'] += 1
                        access_time = (time.time() - start_time) * 1000
                        self._update_average_access_time(access_time)
                        
                        return value
                except Exception as e:
                    logger.warning(f"Redis cache error: {e}")
            
            # Check SQLite
            if self._sqlite_cache:
                try:
                    with self._sqlite_cache.get_connection() as conn:
                        cursor = conn.execute(
                            'SELECT value, ttl, created_at FROM cache_entries WHERE key = ?',
                            (key,)
                        )
                        row = cursor.fetchone()
                        
                        if row:
                            value_data, ttl, created_at = row
                            
                            # Check expiration
                            if ttl and time.time() - created_at > ttl:
                                # Expired, remove it
                                conn.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
                                conn.commit()
                            else:
                                value = self._deserialize_value(value_data)
                                
                                # Update access statistics
                                conn.execute('''
                                    UPDATE cache_entries 
                                    SET last_accessed = ?, access_count = access_count + 1
                                    WHERE key = ?
                                ''', (time.time(), key))
                                conn.commit()
                                
                                # Cache in memory
                                self._cache_in_memory(key, value)
                                
                                self._stats['hits'] += 1
                                access_time = (time.time() - start_time) * 1000
                                self._update_average_access_time(access_time)
                                
                                return value
                except Exception as e:
                    logger.warning(f"SQLite cache error: {e}")
            
            # Cache miss
            self._stats['misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self._stats['misses'] += 1
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None, 
            tags: Optional[List[str]] = None) -> bool:
        """Put value in cache."""
        try:
            ttl = ttl or self.config.default_ttl
            tags = tags or []
            
            # Serialize value
            serialized_value = self._serialize_value(value)
            size_bytes = len(serialized_value)
            
            # Cache in memory
            self._cache_in_memory(key, value, ttl, tags, size_bytes)
            
            # Cache in Redis
            if self._redis_client:
                try:
                    self._redis_client.setex(key, int(ttl), serialized_value)
                except Exception as e:
                    logger.warning(f"Redis cache put error: {e}")
            
            # Cache in SQLite
            if self._sqlite_cache:
                try:
                    with self._sqlite_cache.get_connection() as conn:
                        conn.execute('''
                            INSERT OR REPLACE INTO cache_entries 
                            (key, value, created_at, last_accessed, access_count, size_bytes, ttl, tags)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            key, serialized_value, time.time(), time.time(),
                            1, size_bytes, ttl, json.dumps(tags)
                        ))
                        conn.commit()
                except Exception as e:
                    logger.warning(f"SQLite cache put error: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Cache put error: {e}")
            return False
    
    def _cache_in_memory(self, key: str, value: Any, ttl: Optional[float] = None,
                        tags: Optional[List[str]] = None, size_bytes: Optional[int] = None):
        """Cache value in memory."""
        with self._cache_lock:
            if size_bytes is None:
                size_bytes = len(self._serialize_value(value))
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                size_bytes=size_bytes,
                ttl=ttl,
                tags=tags or []
            )
            
            self._memory_cache[key] = entry
            self._memory_cache.move_to_end(key)
            
            # Update memory usage
            self._stats['memory_usage_bytes'] += size_bytes
            
            # Evict if necessary
            self._evict_if_necessary()
    
    def _evict_if_necessary(self):
        """Evict entries if cache limits are exceeded."""
        # Check memory limit
        memory_limit_bytes = self.config.max_memory_mb * 1024 * 1024
        
        while (self._stats['memory_usage_bytes'] > memory_limit_bytes or 
               len(self._memory_cache) > self.config.max_entries):
            
            if not self._memory_cache:
                break
            
            # Evict based on strategy
            if self.config.strategy == CacheStrategy.LRU:
                # Remove least recently used (first item)
                key, entry = self._memory_cache.popitem(last=False)
            elif self.config.strategy == CacheStrategy.LFU:
                # Remove least frequently used
                key = min(self._memory_cache.keys(), 
                         key=lambda k: self._memory_cache[k].access_count)
                entry = self._memory_cache.pop(key)
            elif self.config.strategy == CacheStrategy.FIFO:
                # Remove oldest entry
                key = min(self._memory_cache.keys(),
                         key=lambda k: self._memory_cache[k].created_at)
                entry = self._memory_cache.pop(key)
            elif self.config.strategy == CacheStrategy.TTL:
                # Remove expired entries first
                expired_keys = [
                    k for k, v in self._memory_cache.items() if v.is_expired()
                ]
                if expired_keys:
                    key = expired_keys[0]
                    entry = self._memory_cache.pop(key)
                else:
                    # No expired entries, use LRU
                    key, entry = self._memory_cache.popitem(last=False)
            else:  # ADAPTIVE
                # Adaptive strategy based on access patterns
                # For now, use LRU
                key, entry = self._memory_cache.popitem(last=False)
            
            self._stats['memory_usage_bytes'] -= entry.size_bytes
            self._stats['evictions'] += 1
    
    def _update_average_access_time(self, access_time_ms: float):
        """Update average access time statistics."""
        # Simple exponential moving average
        alpha = 0.1
        self._stats['average_access_time_ms'] = (
            alpha * access_time_ms + 
            (1 - alpha) * self._stats['average_access_time_ms']
        )
    
    def invalidate(self, key: str) -> bool:
        """Invalidate a cache entry."""
        success = False
        
        # Remove from memory
        with self._cache_lock:
            if key in self._memory_cache:
                entry = self._memory_cache.pop(key)
                self._stats['memory_usage_bytes'] -= entry.size_bytes
                success = True
        
        # Remove from Redis
        if self._redis_client:
            try:
                result = self._redis_client.delete(key)
                success = success or bool(result)
            except Exception as e:
                logger.warning(f"Redis invalidation error: {e}")
        
        # Remove from SQLite
        if self._sqlite_cache:
            try:
                with self._sqlite_cache.get_connection() as conn:
                    cursor = conn.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
                    success = success or bool(cursor.rowcount)
                    conn.commit()
            except Exception as e:
                logger.warning(f"SQLite invalidation error: {e}")
        
        return success
    
    def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate cache entries by tags."""
        invalidated_count = 0
        
        # Memory cache
        with self._cache_lock:
            keys_to_remove = []
            for key, entry in self._memory_cache.items():
                if any(tag in entry.tags for tag in tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                entry = self._memory_cache.pop(key)
                self._stats['memory_usage_bytes'] -= entry.size_bytes
                invalidated_count += 1
        
        # SQLite cache
        if self._sqlite_cache:
            try:
                with self._sqlite_cache.get_connection() as conn:
                    # This is a simplified implementation
                    # In a real system, you'd want proper tag indexing
                    cursor = conn.execute('SELECT key, tags FROM cache_entries')
                    keys_to_delete = []
                    
                    for key, tags_json in cursor.fetchall():
                        try:
                            entry_tags = json.loads(tags_json)
                            if any(tag in entry_tags for tag in tags):
                                keys_to_delete.append(key)
                        except (json.JSONDecodeError, TypeError):
                            continue
                    
                    for key in keys_to_delete:
                        conn.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
                        invalidated_count += 1
                    
                    conn.commit()
            except Exception as e:
                logger.warning(f"SQLite tag invalidation error: {e}")
        
        return invalidated_count
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            # Clear memory cache
            with self._cache_lock:
                self._memory_cache.clear()
                self._stats['memory_usage_bytes'] = 0
            
            # Clear Redis
            if self._redis_client:
                try:
                    self._redis_client.flushdb()
                except Exception as e:
                    logger.warning(f"Redis clear error: {e}")
            
            # Clear SQLite
            if self._sqlite_cache:
                try:
                    with self._sqlite_cache.get_connection() as conn:
                        conn.execute('DELETE FROM cache_entries')
                        conn.commit()
                except Exception as e:
                    logger.warning(f"SQLite clear error: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        if self.config.enable_persistence:
            def persistence_task():
                while not self._shutdown.wait(self.config.persistence_interval):
                    try:
                        self._persist_to_disk()
                    except Exception as e:
                        logger.error(f"Cache persistence error: {e}")
            
            self._persistence_thread = threading.Thread(target=persistence_task, daemon=True)
            self._persistence_thread.start()
        
        def cleanup_task():
            while not self._shutdown.wait(300):  # Run every 5 minutes
                try:
                    self._cleanup_expired_entries()
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        self._cleanup_thread.start()
    
    def _persist_to_disk(self):
        """Persist memory cache to disk."""
        if not self._sqlite_cache:
            return
        
        with self._cache_lock:
            entries_to_persist = list(self._memory_cache.items())
        
        try:
            with self._sqlite_cache.get_connection() as conn:
                for key, entry in entries_to_persist:
                    serialized_value = self._serialize_value(entry.value)
                    
                    conn.execute('''
                        INSERT OR REPLACE INTO cache_entries 
                        (key, value, created_at, last_accessed, access_count, size_bytes, ttl, tags)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        key, serialized_value, entry.created_at, entry.last_accessed,
                        entry.access_count, entry.size_bytes, entry.ttl, json.dumps(entry.tags)
                    ))
                
                conn.commit()
                logger.debug(f"Persisted {len(entries_to_persist)} cache entries to disk")
                
        except Exception as e:
            logger.error(f"Failed to persist cache to disk: {e}")
    
    def _cleanup_expired_entries(self):
        """Clean up expired cache entries."""
        current_time = time.time()
        
        # Clean memory cache
        with self._cache_lock:
            expired_keys = [
                key for key, entry in self._memory_cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                entry = self._memory_cache.pop(key)
                self._stats['memory_usage_bytes'] -= entry.size_bytes
                self._stats['evictions'] += 1
        
        # Clean SQLite cache
        if self._sqlite_cache:
            try:
                with self._sqlite_cache.get_connection() as conn:
                    cursor = conn.execute('''
                        DELETE FROM cache_entries 
                        WHERE ttl IS NOT NULL AND created_at + ttl < ?
                    ''', (current_time,))
                    
                    if cursor.rowcount > 0:
                        logger.debug(f"Cleaned up {cursor.rowcount} expired entries from SQLite cache")
                    
                    conn.commit()
            except Exception as e:
                logger.warning(f"SQLite cleanup error: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._stats['hits'] + self._stats['misses']
        hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0.0
        
        with self._cache_lock:
            memory_entries = len(self._memory_cache)
        
        return {
            **self._stats,
            'hit_rate': hit_rate,
            'memory_entries': memory_entries,
            'memory_usage_mb': self._stats['memory_usage_bytes'] / (1024 * 1024),
            'backends_active': {
                'memory': True,
                'redis': self._redis_client is not None,
                'sqlite': self._sqlite_cache is not None
            }
        }
    
    def close(self):
        """Close the cache system."""
        logger.info("Closing quantum result cache")
        
        # Stop background tasks
        self._shutdown.set()
        
        if self._persistence_thread:
            self._persistence_thread.join(timeout=5)
        
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
        
        # Final persistence
        if self.config.enable_persistence:
            try:
                self._persist_to_disk()
            except Exception as e:
                logger.error(f"Final cache persistence failed: {e}")
        
        # Close backends
        if self._redis_client:
            try:
                self._redis_client.close()
            except Exception:
                pass
        
        if self._sqlite_cache:
            self._sqlite_cache.close()


# Export main classes
__all__ = [
    'CacheBackend',
    'CacheStrategy', 
    'CacheConfig',
    'ConnectionPoolConfig',
    'DatabaseConnectionPool',
    'QuantumResultCache'
]