"""
Tests for QuantRS2 Configuration Management System
"""

import pytest
import tempfile
import yaml
import json
import os
import time
from pathlib import Path
from unittest.mock import Mock, patch

# Safe imports for config management module
try:
    from quantrs2.config_management import (
        Environment,
        ConfigFormat,
        DatabaseConfig,
        QuantumBackendConfig,
        SecurityConfig,
        PerformanceConfig,
        LoggingConfig,
        MonitoringConfig,
        QuantRS2Config,
        ConfigurationManager,
        ConfigurationError,
        get_config_manager,
        load_config,
        get_current_config,
        create_default_configs
    )
    HAS_CONFIG_MANAGEMENT = True
except ImportError:
    HAS_CONFIG_MANAGEMENT = False

@pytest.mark.skipif(not HAS_CONFIG_MANAGEMENT, reason="config_management module not available")
class TestConfigurationDataClasses:
    """Test configuration data classes."""
    
    def test_database_config_defaults(self):
        """Test database configuration defaults."""
        config = DatabaseConfig()
        
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "quantrs2"
        assert config.username == "quantrs2_user"
        assert config.password is None
        assert config.max_connections == 10
    
    def test_quantum_backend_config_defaults(self):
        """Test quantum backend configuration defaults."""
        config = QuantumBackendConfig()
        
        assert config.provider == "simulation"
        assert config.api_token is None
        assert config.region == "us-east-1"
        assert config.max_jobs == 5
        assert config.timeout_seconds == 300
    
    def test_security_config_defaults(self):
        """Test security configuration defaults."""
        config = SecurityConfig()
        
        assert config.encryption_key is None
        assert config.session_timeout == 3600
        assert config.max_login_attempts == 5
        assert not config.enable_2fa
        assert config.allowed_origins == ["localhost"]
    
    def test_quantrs2_config_creation(self):
        """Test main configuration creation."""
        config = QuantRS2Config()
        
        assert config.environment == Environment.DEVELOPMENT
        assert not config.debug
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.security, SecurityConfig)
        assert isinstance(config.performance, PerformanceConfig)
    
    def test_quantrs2_config_to_dict(self):
        """Test configuration serialization."""
        config = QuantRS2Config()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['environment'] == Environment.DEVELOPMENT
        assert 'database' in config_dict
        assert 'security' in config_dict
        assert 'performance' in config_dict
    
    def test_quantrs2_config_from_dict(self):
        """Test configuration deserialization."""
        config_data = {
            'environment': 'production',
            'debug': True,
            'database': {
                'host': 'test-host',
                'port': 5433,
                'database': 'test_db'
            },
            'security': {
                'session_timeout': 7200,
                'enable_2fa': True
            }
        }
        
        config = QuantRS2Config.from_dict(config_data)
        
        assert config.environment == Environment.PRODUCTION
        assert config.debug is True
        assert config.database.host == 'test-host'
        assert config.database.port == 5433
        assert config.security.session_timeout == 7200
        assert config.security.enable_2fa is True

@pytest.mark.skipif(not HAS_CONFIG_MANAGEMENT, reason="config_management module not available")
class TestConfigurationManager:
    """Test configuration manager functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "config"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.manager = ConfigurationManager(self.config_dir)
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_manager_initialization(self):
        """Test configuration manager initialization."""
        assert self.manager.config_dir == self.config_dir
        assert self.manager.config is None
        assert len(self.manager.file_watchers) == 0
        assert len(self.manager.reload_callbacks) == 0
    
    def test_load_config_file_yaml(self):
        """Test loading YAML configuration file."""
        config_data = {
            'debug': True,
            'database': {
                'host': 'yaml-host',
                'port': 5433
            }
        }
        
        config_file = self.config_dir / "test.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        loaded_data = self.manager._load_config_file(config_file)
        
        assert loaded_data['debug'] is True
        assert loaded_data['database']['host'] == 'yaml-host'
        assert loaded_data['database']['port'] == 5433
    
    def test_load_config_file_json(self):
        """Test loading JSON configuration file."""
        config_data = {
            'debug': False,
            'performance': {
                'max_circuit_qubits': 25
            }
        }
        
        config_file = self.config_dir / "test.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        loaded_data = self.manager._load_config_file(config_file)
        
        assert loaded_data['debug'] is False
        assert loaded_data['performance']['max_circuit_qubits'] == 25
    
    def test_load_config_file_not_found(self):
        """Test loading non-existent configuration file."""
        config_file = self.config_dir / "nonexistent.yaml"
        
        with pytest.raises(ConfigurationError):
            self.manager._load_config_file(config_file)
    
    def test_merge_configs(self):
        """Test configuration merging."""
        config1 = {
            'debug': True,
            'database': {
                'host': 'host1',
                'port': 5432
            },
            'custom': {'key1': 'value1'}
        }
        
        config2 = {
            'debug': False,
            'database': {
                'host': 'host2',
                'database': 'new_db'
            },
            'custom': {'key2': 'value2'}
        }
        
        merged = self.manager._merge_configs(config1, config2)
        
        assert merged['debug'] is False  # Overridden
        assert merged['database']['host'] == 'host2'  # Overridden
        assert merged['database']['port'] == 5432  # Preserved
        assert merged['database']['database'] == 'new_db'  # Added
        assert merged['custom']['key1'] == 'value1'  # Preserved
        assert merged['custom']['key2'] == 'value2'  # Added
    
    def test_set_nested_config_value(self):
        """Test setting nested configuration values."""
        config = {}
        
        self.manager._set_nested_config_value(config, ['database', 'host'], 'test-host')
        self.manager._set_nested_config_value(config, ['database', 'port'], '5433')
        self.manager._set_nested_config_value(config, ['debug'], 'true')
        
        assert config['database']['host'] == 'test-host'
        assert config['database']['port'] == 5433  # Converted to int
        assert config['debug'] is True  # Converted to bool
    
    def test_validate_config_success(self):
        """Test successful configuration validation."""
        valid_config = {
            'database': {
                'host': 'valid-host',
                'port': 5432
            },
            'quantum_backends': {
                'ibm': {
                    'provider': 'ibm_quantum'
                }
            }
        }
        
        # Should not raise an exception
        self.manager._validate_config(valid_config)
    
    def test_validate_config_failure(self):
        """Test configuration validation failure."""
        invalid_config = {
            'database': {
                'host': '',  # Empty host
                'port': -1   # Invalid port
            },
            'quantum_backends': {
                'ibm': {
                    # Missing provider
                }
            }
        }
        
        with pytest.raises(ConfigurationError):
            self.manager._validate_config(invalid_config)
    
    def test_load_environment_variables(self):
        """Test loading configuration from environment variables."""
        with patch.dict(os.environ, {
            'QUANTRS2_DB_HOST': 'env-host',
            'QUANTRS2_DB_PORT': '3306',
            'QUANTRS2_LOG_LEVEL': 'DEBUG',
            'QUANTRS2_MAX_QUBITS': '40'
        }):
            config = {}
            self.manager._load_environment_variables(config)
            
            assert config['database']['host'] == 'env-host'
            assert config['database']['port'] == 3306
            assert config['logging']['level'] == 'DEBUG'
            assert config['performance']['max_circuit_qubits'] == 40
    
    def test_full_config_loading(self):
        """Test complete configuration loading process."""
        # Create base config
        base_config = {
            'debug': False,
            'database': {
                'host': 'localhost',
                'port': 5432
            }
        }
        
        base_file = self.config_dir / "base.yaml"
        with open(base_file, 'w') as f:
            yaml.dump(base_config, f)
        
        # Create environment config
        env_config = {
            'debug': True,
            'database': {
                'database': 'dev_db'
            },
            'performance': {
                'max_circuit_qubits': 20
            }
        }
        
        env_file = self.config_dir / "development.yaml"
        with open(env_file, 'w') as f:
            yaml.dump(env_config, f)
        
        # Load configuration
        config = self.manager.load_config(Environment.DEVELOPMENT)
        
        assert isinstance(config, QuantRS2Config)
        assert config.environment == Environment.DEVELOPMENT
        assert config.debug is True  # From environment config
        assert config.database.host == 'localhost'  # From base config
        assert config.database.database == 'dev_db'  # From environment config
        assert config.performance.max_circuit_qubits == 20  # From environment config
    
    def test_config_update(self):
        """Test runtime configuration updates."""
        # Load initial config
        self.manager.config = QuantRS2Config()
        
        # Update configuration
        updates = {
            'debug': True,
            'database': {
                'host': 'updated-host'
            }
        }
        
        self.manager.update_config(updates)
        
        assert self.manager.config.debug is True
        assert self.manager.config.database.host == 'updated-host'
    
    def test_config_save_yaml(self):
        """Test saving configuration to YAML file."""
        config = QuantRS2Config()
        config.debug = True
        config.database.host = 'save-test-host'
        
        self.manager.config = config
        save_file = self.config_dir / "saved_config.yaml"
        
        self.manager.save_config(save_file, ConfigFormat.YAML)
        
        # Verify file was created and contains expected data
        assert save_file.exists()
        
        with open(save_file, 'r') as f:
            saved_data = yaml.safe_load(f)
        
        assert saved_data['debug'] is True
        assert saved_data['database']['host'] == 'save-test-host'
        # Sensitive data should be removed
        assert saved_data['database']['password'] is None
    
    def test_config_save_json(self):
        """Test saving configuration to JSON file."""
        config = QuantRS2Config()
        config.debug = False
        
        self.manager.config = config
        save_file = self.config_dir / "saved_config.json"
        
        self.manager.save_config(save_file, ConfigFormat.JSON)
        
        # Verify file was created and contains expected data
        assert save_file.exists()
        
        with open(save_file, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data['debug'] is False
    
    def test_reload_callback_registration(self):
        """Test reload callback registration."""
        callback_called = False
        callback_config = None
        
        def test_callback(config):
            nonlocal callback_called, callback_config
            callback_called = True
            callback_config = config
        
        self.manager.register_reload_callback(test_callback)
        
        # Simulate reload
        test_config = QuantRS2Config()
        for callback in self.manager.reload_callbacks:
            callback(test_config)
        
        assert callback_called
        assert callback_config == test_config

@pytest.mark.skipif(not HAS_CONFIG_MANAGEMENT, reason="config_management module not available")
class TestUtilityFunctions:
    """Test utility functions."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "config"
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_config_manager(self):
        """Test getting configuration manager."""
        manager = get_config_manager(self.config_dir)
        
        assert isinstance(manager, ConfigurationManager)
        assert manager.config_dir == self.config_dir
    
    def test_create_default_configs(self):
        """Test creating default configuration files."""
        create_default_configs(self.config_dir)
        
        # Check that all expected files were created
        expected_files = [
            'base.yaml',
            'development.yaml',
            'testing.yaml',
            'staging.yaml',
            'production.yaml'
        ]
        
        for filename in expected_files:
            config_file = self.config_dir / filename
            assert config_file.exists()
            
            # Verify file contains valid YAML
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            assert isinstance(config_data, dict)
    
    def test_load_config_function(self):
        """Test load_config utility function."""
        create_default_configs(self.config_dir)
        
        config = load_config(Environment.DEVELOPMENT, config_dir=self.config_dir)
        
        assert isinstance(config, QuantRS2Config)
        assert config.environment == Environment.DEVELOPMENT

@pytest.mark.skipif(not HAS_CONFIG_MANAGEMENT, reason="config_management module not available")
class TestHotReloading:
    """Test hot-reloading functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "config"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.manager = ConfigurationManager(self.config_dir)
    
    def teardown_method(self):
        """Cleanup test environment."""
        self.manager.cleanup()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_file_watching_setup(self):
        """Test file watching setup."""
        # Create a config file
        config_file = self.config_dir / "test.yaml"
        with open(config_file, 'w') as f:
            yaml.dump({'debug': False}, f)
        
        # Load config file (should setup watching)
        self.manager._load_config_file(config_file)
        self.manager._setup_file_watching()
        
        assert str(config_file) in self.manager.file_watchers
        assert self.manager.watch_enabled
        assert self.manager.watch_thread is not None
    
    def test_reload_callback_execution(self):
        """Test that reload callbacks are executed."""
        callback_executed = False
        
        def test_callback(config):
            nonlocal callback_executed
            callback_executed = True
        
        self.manager.register_reload_callback(test_callback)
        
        # Mock the reload process
        with patch.object(self.manager, 'load_config') as mock_load:
            mock_load.return_value = QuantRS2Config()
            self.manager.config = QuantRS2Config()
            
            self.manager._reload_config()
            
            assert callback_executed

@pytest.mark.skipif(not HAS_CONFIG_MANAGEMENT, reason="config_management module not available")
class TestErrorHandling:
    """Test error handling in configuration management."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "config"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.manager = ConfigurationManager(self.config_dir)
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_invalid_yaml_file(self):
        """Test handling of invalid YAML file."""
        config_file = self.config_dir / "invalid.yaml"
        with open(config_file, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with pytest.raises(ConfigurationError):
            self.manager._load_config_file(config_file)
    
    def test_invalid_json_file(self):
        """Test handling of invalid JSON file."""
        config_file = self.config_dir / "invalid.json"
        with open(config_file, 'w') as f:
            f.write('{"invalid": json"}')
        
        with pytest.raises(ConfigurationError):
            self.manager._load_config_file(config_file)
    
    def test_unsupported_file_format(self):
        """Test handling of unsupported file format."""
        config_file = self.config_dir / "config.txt"
        with open(config_file, 'w') as f:
            f.write("some content")
        
        with pytest.raises(ConfigurationError):
            self.manager._load_config_file(config_file)
    
    def test_config_update_validation_error(self):
        """Test that config updates are validated."""
        self.manager.config = QuantRS2Config()
        
        invalid_updates = {
            'database': {
                'port': 'invalid_port'  # Should be integer
            }
        }
        
        with pytest.raises(ConfigurationError):
            self.manager.update_config(invalid_updates)
    
    def test_save_config_without_loaded_config(self):
        """Test saving config when no config is loaded."""
        save_file = self.config_dir / "save_test.yaml"
        
        with pytest.raises(ConfigurationError):
            self.manager.save_config(save_file)

@pytest.mark.skipif(not HAS_CONFIG_MANAGEMENT, reason="config_management module not available")
class TestEnvironmentVariableOverrides:
    """Test environment variable override functionality."""
    
    def test_database_env_vars(self):
        """Test database environment variable overrides."""
        manager = ConfigurationManager()
        
        with patch.dict(os.environ, {
            'QUANTRS2_DB_HOST': 'env-db-host',
            'QUANTRS2_DB_PORT': '3306',
            'QUANTRS2_DB_NAME': 'env_database',
            'QUANTRS2_DB_USER': 'env_user',
            'QUANTRS2_DB_PASSWORD': 'env_password'
        }):
            config = {}
            manager._load_environment_variables(config)
            
            assert config['database']['host'] == 'env-db-host'
            assert config['database']['port'] == 3306
            assert config['database']['database'] == 'env_database'
            assert config['database']['username'] == 'env_user'
            assert config['database']['password'] == 'env_password'
    
    def test_security_env_vars(self):
        """Test security environment variable overrides."""
        manager = ConfigurationManager()
        
        with patch.dict(os.environ, {
            'QUANTRS2_SECRET_KEY': 'env-secret-key',
            'QUANTRS2_JWT_SECRET': 'env-jwt-secret',
            'QUANTRS2_ENCRYPTION_KEY': 'env-encryption-key'
        }):
            config = {}
            manager._load_environment_variables(config)
            
            assert config['secret_key'] == 'env-secret-key'
            assert config['security']['jwt_secret'] == 'env-jwt-secret'
            assert config['security']['encryption_key'] == 'env-encryption-key'
    
    def test_boolean_env_var_conversion(self):
        """Test boolean environment variable conversion."""
        manager = ConfigurationManager()
        
        with patch.dict(os.environ, {
            'QUANTRS2_DEBUG': 'true'
        }):
            config = {}
            manager._set_nested_config_value(config, ['debug'], 'true')
            
            assert config['debug'] is True
        
        with patch.dict(os.environ, {
            'QUANTRS2_DEBUG': 'false'
        }):
            config = {}
            manager._set_nested_config_value(config, ['debug'], 'false')
            
            assert config['debug'] is False

if __name__ == "__main__":
    pytest.main([__file__])