"""
Environment-Specific Configuration Management for QuantRS2

This module provides comprehensive configuration management for different
deployment environments with validation, encryption, and hot-reloading.
"""

import os
import json
import yaml
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import time

# Import security components with error handling
try:
    from .security import SecretsManager, encrypt_data, decrypt_data
except ImportError:
    # Provide stub implementations if security module is not available
    class SecretsManager:
        def __init__(self):
            pass
        def get_secret(self, name):
            return os.getenv(name)
        def set_secret(self, name, value):
            pass
    
    def encrypt_data(data):
        return data
    
    def decrypt_data(data):
        return data

logger = logging.getLogger(__name__)

class Environment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class ConfigFormat(Enum):
    """Configuration file formats."""
    JSON = "json"
    YAML = "yaml"
    ENV = "env"

@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "quantrs2"
    username: str = "quantrs2_user"
    password: Optional[str] = None
    max_connections: int = 10
    ssl_mode: str = "require"
    connection_timeout: int = 30

@dataclass
class QuantumBackendConfig:
    """Quantum backend configuration."""
    provider: str = "simulation"
    api_token: Optional[str] = None
    project_id: Optional[str] = None
    region: str = "us-east-1"
    max_jobs: int = 5
    timeout_seconds: int = 300
    retry_attempts: int = 3
    enable_caching: bool = True

@dataclass
class SecurityConfig:
    """Security configuration."""
    encryption_key: Optional[str] = None
    jwt_secret: Optional[str] = None
    session_timeout: int = 3600
    max_login_attempts: int = 5
    enable_2fa: bool = False
    allowed_origins: List[str] = field(default_factory=lambda: ["localhost"])
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

@dataclass
class PerformanceConfig:
    """Performance configuration."""
    max_circuit_qubits: int = 30
    simulation_memory_limit: int = 8192  # MB
    max_concurrent_jobs: int = 4
    circuit_cache_size: int = 1000
    result_cache_ttl: int = 3600
    enable_gpu: bool = False
    gpu_memory_fraction: float = 0.8

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5
    enable_structured_logging: bool = True
    log_to_console: bool = True

@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration."""
    enable_metrics: bool = True
    metrics_port: int = 9090
    metrics_endpoint: str = "/metrics"
    enable_health_checks: bool = True
    health_check_port: int = 8080
    alert_webhook_url: Optional[str] = None
    enable_tracing: bool = False
    tracing_endpoint: Optional[str] = None

@dataclass
class QuantRS2Config:
    """Main QuantRS2 configuration."""
    
    # Environment and basic settings
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    secret_key: str = "dev-secret-key"
    
    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    quantum_backends: Dict[str, QuantumBackendConfig] = field(default_factory=dict)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Custom settings
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuantRS2Config':
        """Create configuration from dictionary."""
        # Convert environment string to enum
        if 'environment' in data and isinstance(data['environment'], str):
            data['environment'] = Environment(data['environment'])
        
        # Create sub-configurations
        config = cls()
        
        if 'database' in data:
            config.database = DatabaseConfig(**data['database'])
        
        if 'quantum_backends' in data:
            config.quantum_backends = {
                name: QuantumBackendConfig(**backend_data)
                for name, backend_data in data['quantum_backends'].items()
            }
        
        if 'security' in data:
            config.security = SecurityConfig(**data['security'])
        
        if 'performance' in data:
            config.performance = PerformanceConfig(**data['performance'])
        
        if 'logging' in data:
            config.logging = LoggingConfig(**data['logging'])
        
        if 'monitoring' in data:
            config.monitoring = MonitoringConfig(**data['monitoring'])
        
        # Set other fields
        for key, value in data.items():
            if hasattr(config, key) and key not in ['database', 'quantum_backends', 'security', 'performance', 'logging', 'monitoring']:
                setattr(config, key, value)
        
        return config

class ConfigurationManager:
    """Manages configuration loading, validation, and hot-reloading."""
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        self.config_dir = Path(config_dir) if config_dir else Path.cwd() / "config"
        self.config: Optional[QuantRS2Config] = None
        self.secrets_manager = SecretsManager()
        
        # Hot-reloading
        self.file_watchers: Dict[str, float] = {}  # file -> last_modified
        self.reload_callbacks: List[Callable[[QuantRS2Config], None]] = []
        self.watch_thread: Optional[threading.Thread] = None
        self.watch_enabled = False
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        logger.info(f"Configuration manager initialized with config dir: {self.config_dir}")
    
    def load_config(self, environment: Environment = Environment.DEVELOPMENT,
                   config_file: Optional[str] = None) -> QuantRS2Config:
        """
        Load configuration for specified environment.
        
        Args:
            environment: Target environment
            config_file: Optional specific config file to load
            
        Returns:
            Loaded configuration
        """
        with self._lock:
            try:
                # Load base configuration
                base_config = self._load_base_config()
                
                # Load environment-specific configuration
                env_config = self._load_environment_config(environment)
                
                # Load specific config file if provided
                file_config = {}
                if config_file:
                    file_config = self._load_config_file(config_file)
                
                # Merge configurations (file > env > base)
                merged_config = self._merge_configs(base_config, env_config, file_config)
                
                # Load secrets and environment variables
                self._load_secrets_and_env_vars(merged_config)
                
                # Validate configuration
                self._validate_config(merged_config)
                
                # Create configuration object
                self.config = QuantRS2Config.from_dict(merged_config)
                self.config.environment = environment
                
                # Setup file watching for hot-reload
                self._setup_file_watching()
                
                logger.info(f"Configuration loaded for environment: {environment.value}")
                return self.config
                
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                raise ConfigurationError(f"Configuration loading failed: {e}")
    
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration."""
        base_file = self.config_dir / "base.yaml"
        if base_file.exists():
            return self._load_config_file(base_file)
        
        # Return default configuration
        return QuantRS2Config().to_dict()
    
    def _load_environment_config(self, environment: Environment) -> Dict[str, Any]:
        """Load environment-specific configuration."""
        env_files = [
            self.config_dir / f"{environment.value}.yaml",
            self.config_dir / f"{environment.value}.json",
        ]
        
        for env_file in env_files:
            if env_file.exists():
                return self._load_config_file(env_file)
        
        logger.warning(f"No environment config found for {environment.value}")
        return {}
    
    def _load_config_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ConfigurationError(f"Configuration file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Determine format and parse
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                config_data = yaml.safe_load(content)
            elif file_path.suffix.lower() == '.json':
                config_data = json.loads(content)
            else:
                raise ConfigurationError(f"Unsupported config file format: {file_path.suffix}")
            
            # Track file for hot-reloading
            self.file_watchers[str(file_path)] = file_path.stat().st_mtime
            
            logger.debug(f"Loaded config file: {file_path}")
            return config_data or {}
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load config file {file_path}: {e}")
    
    def _merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configuration dictionaries."""
        merged = {}
        
        for config in configs:
            if not config:
                continue
            
            for key, value in config.items():
                if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    # Recursively merge dictionaries
                    merged[key] = self._merge_configs(merged[key], value)
                else:
                    # Override with new value
                    merged[key] = value
        
        return merged
    
    def _load_secrets_and_env_vars(self, config: Dict[str, Any]) -> None:
        """Load secrets and environment variables."""
        # Load secrets for sensitive fields
        self._load_secrets(config)
        
        # Override with environment variables
        self._load_environment_variables(config)
    
    def _load_secrets(self, config: Dict[str, Any]) -> None:
        """Load encrypted secrets."""
        try:
            # Database password
            if 'database' in config and not config['database'].get('password'):
                db_password = self.secrets_manager.get_credential('database_password')
                if db_password:
                    config['database']['password'] = db_password
            
            # Quantum backend tokens
            if 'quantum_backends' in config:
                for backend_name, backend_config in config['quantum_backends'].items():
                    if not backend_config.get('api_token'):
                        token = self.secrets_manager.get_credential(f'{backend_name}_api_token')
                        if token:
                            backend_config['api_token'] = token
            
            # Security secrets
            if 'security' in config:
                security_config = config['security']
                
                if not security_config.get('encryption_key'):
                    encryption_key = self.secrets_manager.get_credential('encryption_key')
                    if encryption_key:
                        security_config['encryption_key'] = encryption_key
                
                if not security_config.get('jwt_secret'):
                    jwt_secret = self.secrets_manager.get_credential('jwt_secret')
                    if jwt_secret:
                        security_config['jwt_secret'] = jwt_secret
            
        except Exception as e:
            logger.warning(f"Failed to load some secrets: {e}")
    
    def _load_environment_variables(self, config: Dict[str, Any]) -> None:
        """Load configuration from environment variables."""
        env_mappings = {
            # Database
            'QUANTRS2_DB_HOST': ['database', 'host'],
            'QUANTRS2_DB_PORT': ['database', 'port'],
            'QUANTRS2_DB_NAME': ['database', 'database'],
            'QUANTRS2_DB_USER': ['database', 'username'],
            'QUANTRS2_DB_PASSWORD': ['database', 'password'],
            
            # Security
            'QUANTRS2_SECRET_KEY': ['secret_key'],
            'QUANTRS2_JWT_SECRET': ['security', 'jwt_secret'],
            'QUANTRS2_ENCRYPTION_KEY': ['security', 'encryption_key'],
            
            # Performance
            'QUANTRS2_MAX_QUBITS': ['performance', 'max_circuit_qubits'],
            'QUANTRS2_MEMORY_LIMIT': ['performance', 'simulation_memory_limit'],
            'QUANTRS2_MAX_JOBS': ['performance', 'max_concurrent_jobs'],
            
            # Logging
            'QUANTRS2_LOG_LEVEL': ['logging', 'level'],
            'QUANTRS2_LOG_FILE': ['logging', 'file_path'],
            
            # Monitoring
            'QUANTRS2_METRICS_PORT': ['monitoring', 'metrics_port'],
            'QUANTRS2_HEALTH_PORT': ['monitoring', 'health_check_port'],
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                self._set_nested_config_value(config, config_path, value)
    
    def _set_nested_config_value(self, config: Dict[str, Any], path: List[str], value: str) -> None:
        """Set nested configuration value from environment variable."""
        current = config
        
        # Navigate to parent
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set value with type conversion
        final_key = path[-1]
        
        # Attempt type conversion
        if value.lower() in ['true', 'false']:
            current[final_key] = value.lower() == 'true'
        elif value.isdigit():
            current[final_key] = int(value)
        elif value.replace('.', '').isdigit():
            current[final_key] = float(value)
        else:
            current[final_key] = value
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration."""
        errors = []
        
        # Validate database configuration
        if 'database' in config:
            db_config = config['database']
            if not db_config.get('host'):
                errors.append("Database host is required")
            if not isinstance(db_config.get('port', 0), int) or db_config.get('port', 0) <= 0:
                errors.append("Database port must be a positive integer")
        
        # Validate quantum backend configurations
        if 'quantum_backends' in config:
            for name, backend_config in config['quantum_backends'].items():
                if not backend_config.get('provider'):
                    errors.append(f"Quantum backend '{name}' missing provider")
        
        # Validate security configuration
        if 'security' in config:
            security_config = config['security']
            if config.get('environment') == Environment.PRODUCTION.value:
                if not security_config.get('encryption_key'):
                    errors.append("Encryption key is required in production")
                if not security_config.get('jwt_secret'):
                    errors.append("JWT secret is required in production")
        
        # Validate performance limits
        if 'performance' in config:
            perf_config = config['performance']
            max_qubits = perf_config.get('max_circuit_qubits', 0)
            if max_qubits > 50:
                logger.warning(f"Large maximum qubit count configured: {max_qubits}")
        
        if errors:
            raise ConfigurationError(f"Configuration validation failed: {', '.join(errors)}")
    
    def _setup_file_watching(self) -> None:
        """Setup file watching for hot-reload."""
        if not self.watch_enabled and self.file_watchers:
            self.watch_enabled = True
            self.watch_thread = threading.Thread(target=self._watch_files, daemon=True)
            self.watch_thread.start()
            logger.info("File watching enabled for hot-reload")
    
    def _watch_files(self) -> None:
        """Watch configuration files for changes."""
        while self.watch_enabled:
            try:
                time.sleep(1.0)  # Check every second
                
                with self._lock:
                    changed_files = []
                    
                    for file_path, last_modified in self.file_watchers.items():
                        try:
                            current_modified = Path(file_path).stat().st_mtime
                            if current_modified > last_modified:
                                changed_files.append(file_path)
                                self.file_watchers[file_path] = current_modified
                        except OSError:
                            # File might have been deleted
                            logger.warning(f"Config file no longer accessible: {file_path}")
                    
                    if changed_files:
                        logger.info(f"Configuration files changed: {changed_files}")
                        self._reload_config()
                        
            except Exception as e:
                logger.error(f"Error in file watching: {e}")
                time.sleep(5.0)  # Back off on error
    
    def _reload_config(self) -> None:
        """Reload configuration and notify callbacks."""
        try:
            if self.config:
                old_environment = self.config.environment
                new_config = self.load_config(old_environment)
                
                # Notify callbacks
                for callback in self.reload_callbacks:
                    try:
                        callback(new_config)
                    except Exception as e:
                        logger.error(f"Error in reload callback: {e}")
                
                logger.info("Configuration reloaded successfully")
                
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
    
    def register_reload_callback(self, callback: Callable[[QuantRS2Config], None]) -> None:
        """Register callback for configuration reload events."""
        self.reload_callbacks.append(callback)
    
    def get_config(self) -> Optional[QuantRS2Config]:
        """Get current configuration."""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration at runtime."""
        with self._lock:
            if self.config:
                # Apply updates to current config
                config_dict = self.config.to_dict()
                merged_config = self._merge_configs(config_dict, updates)
                
                # Validate updated configuration
                self._validate_config(merged_config)
                
                # Update configuration
                self.config = QuantRS2Config.from_dict(merged_config)
                
                logger.info("Configuration updated at runtime")
    
    def save_config(self, file_path: Union[str, Path], format: ConfigFormat = ConfigFormat.YAML) -> None:
        """Save current configuration to file."""
        if not self.config:
            raise ConfigurationError("No configuration loaded to save")
        
        file_path = Path(file_path)
        config_dict = self.config.to_dict()
        
        # Remove sensitive data before saving
        sanitized_config = self._sanitize_config_for_save(config_dict)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if format == ConfigFormat.YAML:
                    yaml.dump(sanitized_config, f, default_flow_style=False, indent=2)
                elif format == ConfigFormat.JSON:
                    json.dump(sanitized_config, f, indent=2)
                else:
                    raise ConfigurationError(f"Unsupported save format: {format}")
            
            logger.info(f"Configuration saved to: {file_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")
    
    def _sanitize_config_for_save(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive data from configuration before saving."""
        sanitized = config.copy()
        
        # Remove sensitive database fields
        if 'database' in sanitized and 'password' in sanitized['database']:
            sanitized['database']['password'] = None
        
        # Remove sensitive security fields
        if 'security' in sanitized:
            sanitized['security']['encryption_key'] = None
            sanitized['security']['jwt_secret'] = None
        
        # Remove API tokens from quantum backends
        if 'quantum_backends' in sanitized:
            for backend_config in sanitized['quantum_backends'].values():
                backend_config['api_token'] = None
        
        return sanitized
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.watch_enabled = False
        if self.watch_thread and self.watch_thread.is_alive():
            self.watch_thread.join(timeout=2.0)
        logger.info("Configuration manager cleanup completed")

class ConfigurationError(Exception):
    """Exception raised for configuration errors."""
    pass

# Global configuration manager
_config_manager = ConfigurationManager()

def get_config_manager(config_dir: Optional[Union[str, Path]] = None) -> ConfigurationManager:
    """Get the global configuration manager."""
    global _config_manager
    if config_dir:
        _config_manager = ConfigurationManager(config_dir)
    return _config_manager

def load_config(environment: Environment = Environment.DEVELOPMENT,
               config_file: Optional[str] = None,
               config_dir: Optional[Union[str, Path]] = None) -> QuantRS2Config:
    """Load configuration for specified environment."""
    manager = get_config_manager(config_dir)
    return manager.load_config(environment, config_file)

def get_current_config() -> Optional[QuantRS2Config]:
    """Get currently loaded configuration."""
    return get_config_manager().get_config()

def create_default_configs(config_dir: Union[str, Path]) -> None:
    """Create default configuration files for all environments."""
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Base configuration
    base_config = QuantRS2Config()
    base_dict = base_config.to_dict()
    
    # Environment-specific configurations
    env_configs = {
        Environment.DEVELOPMENT: {
            'debug': True,
            'database': {'host': 'localhost', 'database': 'quantrs2_dev'},
            'logging': {'level': 'DEBUG'},
            'performance': {'max_circuit_qubits': 20}
        },
        Environment.TESTING: {
            'debug': True,
            'database': {'host': 'localhost', 'database': 'quantrs2_test'},
            'logging': {'level': 'DEBUG'},
            'performance': {'max_circuit_qubits': 15}
        },
        Environment.STAGING: {
            'debug': False,
            'database': {'host': 'staging-db.example.com'},
            'logging': {'level': 'INFO'},
            'performance': {'max_circuit_qubits': 25}
        },
        Environment.PRODUCTION: {
            'debug': False,
            'database': {'host': 'prod-db.example.com'},
            'logging': {'level': 'WARNING', 'file_path': '/var/log/quantrs2.log'},
            'performance': {'max_circuit_qubits': 30},
            'monitoring': {'enable_metrics': True, 'enable_health_checks': True}
        }
    }
    
    # Save base configuration
    with open(config_dir / 'base.yaml', 'w') as f:
        yaml.dump(base_dict, f, default_flow_style=False, indent=2)
    
    # Save environment-specific configurations
    for env, env_config in env_configs.items():
        merged = ConfigurationManager()._merge_configs(base_dict, env_config)
        with open(config_dir / f'{env.value}.yaml', 'w') as f:
            yaml.dump(merged, f, default_flow_style=False, indent=2)
    
    logger.info(f"Default configuration files created in: {config_dir}")

# Export configuration management components
__all__ = [
    # Enums
    "Environment",
    "ConfigFormat",
    
    # Configuration classes
    "DatabaseConfig",
    "QuantumBackendConfig", 
    "SecurityConfig",
    "PerformanceConfig",
    "LoggingConfig",
    "MonitoringConfig",
    "QuantRS2Config",
    
    # Main classes
    "ConfigurationManager",
    "ConfigurationError",
    
    # Utilities
    "get_config_manager",
    "load_config",
    "get_current_config",
    "create_default_configs",
]