#!/usr/bin/env python3
"""
Test suite for quantum plugin system functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Type, Callable, Any, Optional

try:
    from quantrs2.plugins import (
        PluginType, PluginMetadata, PluginInterface, GatePlugin, AlgorithmPlugin,
        BackendPlugin, OptimizerPlugin, VisualizerPlugin, ConverterPlugin,
        MiddlewarePlugin, PluginInfo, PluginRegistry, PluginManager,
        get_plugin_manager, register_plugin, load_plugin,
        get_available_gates, get_available_algorithms, get_available_backends
    )
    HAS_PLUGINS = True
except ImportError:
    HAS_PLUGINS = False


# Mock plugin implementations for testing
class MockGatePlugin(GatePlugin):
    """Mock gate plugin for testing."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="MockGatePlugin",
            version="1.0.0",
            description="A mock gate plugin for testing",
            author="Test Author",
            plugin_type=PluginType.GATE,
            keywords=["test", "gate"]
        )
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        self._initialized = True
        return True
    
    def cleanup(self) -> None:
        self._initialized = False
    
    def get_gate_classes(self) -> Dict[str, Type]:
        return {
            "MockGate": object,  # Simplified for testing
            "TestGate": object
        }
    
    def get_gate_functions(self) -> Dict[str, Callable]:
        return {
            "mock_gate": lambda: "mock gate",
            "test_gate": lambda: "test gate"
        }


class MockAlgorithmPlugin(AlgorithmPlugin):
    """Mock algorithm plugin for testing."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="MockAlgorithmPlugin",
            version="2.0.0",
            description="A mock algorithm plugin for testing",
            author="Algorithm Author",
            plugin_type=PluginType.ALGORITHM,
            dependencies=["numpy"],
            keywords=["test", "algorithm"]
        )
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        self._config = config or {}
        return True
    
    def cleanup(self) -> None:
        self._config = {}
    
    def get_algorithms(self) -> Dict[str, Type]:
        return {
            "MockVQE": object,
            "TestQAOA": object
        }
    
    def get_algorithm_templates(self) -> Dict[str, Callable]:
        return {
            "mock_template": lambda: "mock template",
            "test_template": lambda: "test template"
        }


class MockBackendPlugin(BackendPlugin):
    """Mock backend plugin for testing."""
    
    def __init__(self, available: bool = True):
        self._available = available
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="MockBackendPlugin",
            version="1.5.0",
            description="A mock backend plugin for testing",
            author="Backend Author",
            plugin_type=PluginType.BACKEND,
            config_schema={
                "required": ["backend_url"],
                "properties": {
                    "backend_url": {"type": "string"},
                    "timeout": {"type": "integer", "default": 30}
                }
            }
        )
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        if config and not self.validate_config(config):
            return False
        self._config = config or {}
        return True
    
    def cleanup(self) -> None:
        self._config = {}
    
    def get_backend_class(self) -> Type:
        return object  # Simplified for testing
    
    def is_available(self) -> bool:
        return self._available


class MockMiddlewarePlugin(MiddlewarePlugin):
    """Mock middleware plugin for testing."""
    
    def __init__(self, priority: int = 10):
        self._priority = priority
        self._processed_circuits = []
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="MockMiddlewarePlugin",
            version="1.0.0",
            description="A mock middleware plugin for testing",
            author="Middleware Author",
            plugin_type=PluginType.MIDDLEWARE
        )
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        self._processed_circuits = []
        return True
    
    def cleanup(self) -> None:
        self._processed_circuits = []
    
    def process_circuit(self, circuit: Any, context: Dict[str, Any]) -> Any:
        self._processed_circuits.append(circuit)
        # Simple transformation for testing
        if isinstance(circuit, str):
            return f"processed_{circuit}"
        return circuit
    
    def get_middleware_priority(self) -> int:
        return self._priority


class FailingPlugin(PluginInterface):
    """Plugin that fails initialization for testing error handling."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="FailingPlugin",
            version="1.0.0",
            description="A plugin that fails for testing",
            author="Test Author",
            plugin_type=PluginType.EXTENSION
        )
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        return False  # Always fails
    
    def cleanup(self) -> None:
        pass


@pytest.mark.skipif(not HAS_PLUGINS, reason="plugins module not available")
class TestPluginMetadata:
    """Test PluginMetadata functionality."""
    
    def test_metadata_creation(self):
        """Test creating plugin metadata."""
        metadata = PluginMetadata(
            name="TestPlugin",
            version="1.0.0",
            description="A test plugin",
            author="Test Author",
            plugin_type=PluginType.GATE,
            dependencies=["numpy", "scipy"],
            keywords=["test", "quantum"]
        )
        
        assert metadata.name == "TestPlugin"
        assert metadata.version == "1.0.0"
        assert metadata.description == "A test plugin"
        assert metadata.author == "Test Author"
        assert metadata.plugin_type == PluginType.GATE
        assert metadata.dependencies == ["numpy", "scipy"]
        assert metadata.keywords == ["test", "quantum"]
        assert metadata.license == "MIT"  # Default
    
    def test_metadata_with_config_schema(self):
        """Test metadata with configuration schema."""
        config_schema = {
            "required": ["api_key"],
            "properties": {
                "api_key": {"type": "string"},
                "timeout": {"type": "integer", "default": 30}
            }
        }
        
        metadata = PluginMetadata(
            name="ConfigPlugin",
            version="1.0.0",
            description="Plugin with config schema",
            author="Test Author",
            plugin_type=PluginType.BACKEND,
            config_schema=config_schema
        )
        
        assert metadata.config_schema == config_schema


@pytest.mark.skipif(not HAS_PLUGINS, reason="plugins module not available")
class TestPluginRegistry:
    """Test PluginRegistry functionality."""
    
    def setup_method(self):
        """Set up test registry."""
        self.registry = PluginRegistry()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test registry."""
        # Cleanup all plugins
        for plugin_name in list(self.registry._plugins.keys()):
            self.registry.unregister_plugin(plugin_name)
        shutil.rmtree(self.temp_dir)
    
    def test_register_gate_plugin(self):
        """Test registering a gate plugin."""
        plugin = MockGatePlugin()
        
        success = self.registry.register_plugin(plugin)
        
        assert success is True
        assert "MockGatePlugin" in self.registry._plugins
        assert self.registry._plugins["MockGatePlugin"].is_active is True
        
        # Check plugin type categorization
        gate_plugins = self.registry.get_plugins_by_type(PluginType.GATE)
        assert len(gate_plugins) == 1
        assert gate_plugins[0] == plugin
    
    def test_register_algorithm_plugin(self):
        """Test registering an algorithm plugin."""
        plugin = MockAlgorithmPlugin()
        config = {"param1": "value1"}
        
        success = self.registry.register_plugin(plugin, config)
        
        assert success is True
        assert "MockAlgorithmPlugin" in self.registry._plugins
        
        plugin_info = self.registry._plugins["MockAlgorithmPlugin"]
        assert plugin_info.config == config
        assert plugin_info.metadata.plugin_type == PluginType.ALGORITHM
    
    def test_register_backend_plugin_with_config(self):
        """Test registering backend plugin with configuration."""
        plugin = MockBackendPlugin()
        config = {"backend_url": "http://test.com", "timeout": 60}
        
        success = self.registry.register_plugin(plugin, config)
        
        assert success is True
        assert "MockBackendPlugin" in self.registry._plugins
    
    def test_register_backend_plugin_invalid_config(self):
        """Test registering backend plugin with invalid configuration."""
        plugin = MockBackendPlugin()
        config = {"invalid_key": "value"}  # Missing required backend_url
        
        success = self.registry.register_plugin(plugin, config)
        
        assert success is False
        assert "MockBackendPlugin" not in self.registry._plugins
    
    def test_register_failing_plugin(self):
        """Test registering a plugin that fails initialization."""
        plugin = FailingPlugin()
        
        success = self.registry.register_plugin(plugin)
        
        assert success is False
        assert "FailingPlugin" not in self.registry._plugins
    
    def test_register_duplicate_plugin(self):
        """Test registering a plugin with duplicate name."""
        plugin1 = MockGatePlugin()
        plugin2 = MockGatePlugin()
        
        success1 = self.registry.register_plugin(plugin1)
        success2 = self.registry.register_plugin(plugin2)
        
        assert success1 is True
        assert success2 is True  # Should replace the first one
        assert len(self.registry._plugins) == 1
    
    def test_unregister_plugin(self):
        """Test unregistering a plugin."""
        plugin = MockGatePlugin()
        self.registry.register_plugin(plugin)
        
        assert "MockGatePlugin" in self.registry._plugins
        
        success = self.registry.unregister_plugin("MockGatePlugin")
        
        assert success is True
        assert "MockGatePlugin" not in self.registry._plugins
    
    def test_unregister_nonexistent_plugin(self):
        """Test unregistering a nonexistent plugin."""
        success = self.registry.unregister_plugin("NonexistentPlugin")
        assert success is False
    
    def test_get_plugin(self):
        """Test getting a plugin by name."""
        plugin = MockGatePlugin()
        self.registry.register_plugin(plugin)
        
        retrieved = self.registry.get_plugin("MockGatePlugin")
        
        assert retrieved == plugin
        
        nonexistent = self.registry.get_plugin("NonexistentPlugin")
        assert nonexistent is None
    
    def test_get_plugins_by_type(self):
        """Test getting plugins by type."""
        gate_plugin = MockGatePlugin()
        algorithm_plugin = MockAlgorithmPlugin()
        backend_plugin = MockBackendPlugin()
        
        self.registry.register_plugin(gate_plugin)
        self.registry.register_plugin(algorithm_plugin)
        self.registry.register_plugin(backend_plugin)
        
        gate_plugins = self.registry.get_plugins_by_type(PluginType.GATE)
        assert len(gate_plugins) == 1
        assert gate_plugins[0] == gate_plugin
        
        algorithm_plugins = self.registry.get_plugins_by_type(PluginType.ALGORITHM)
        assert len(algorithm_plugins) == 1
        assert algorithm_plugins[0] == algorithm_plugin
        
        backend_plugins = self.registry.get_plugins_by_type(PluginType.BACKEND)
        assert len(backend_plugins) == 1
        assert backend_plugins[0] == backend_plugin
        
        # Test empty type
        optimizer_plugins = self.registry.get_plugins_by_type(PluginType.OPTIMIZER)
        assert len(optimizer_plugins) == 0
    
    def test_activate_deactivate_plugin(self):
        """Test activating and deactivating plugins."""
        plugin = MockGatePlugin()
        self.registry.register_plugin(plugin)
        
        # Initially active
        active_plugins = self.registry.get_plugins_by_type(PluginType.GATE)
        assert len(active_plugins) == 1
        
        # Deactivate
        success = self.registry.deactivate_plugin("MockGatePlugin")
        assert success is True
        
        active_plugins = self.registry.get_plugins_by_type(PluginType.GATE)
        assert len(active_plugins) == 0
        
        # Reactivate
        success = self.registry.activate_plugin("MockGatePlugin")
        assert success is True
        
        active_plugins = self.registry.get_plugins_by_type(PluginType.GATE)
        assert len(active_plugins) == 1
    
    def test_list_plugins(self):
        """Test listing all plugins."""
        gate_plugin = MockGatePlugin()
        algorithm_plugin = MockAlgorithmPlugin()
        
        self.registry.register_plugin(gate_plugin)
        self.registry.register_plugin(algorithm_plugin)
        
        all_plugins = self.registry.list_plugins()
        
        assert len(all_plugins) == 2
        plugin_names = [p.name for p in all_plugins]
        assert "MockGatePlugin" in plugin_names
        assert "MockAlgorithmPlugin" in plugin_names
    
    def test_add_search_path(self):
        """Test adding search paths."""
        search_path = Path(self.temp_dir)
        
        self.registry.add_search_path(search_path)
        
        assert search_path in self.registry._search_paths
    
    def test_plugin_hooks(self):
        """Test plugin event hooks."""
        hook_calls = []
        
        def registration_hook(plugin_info):
            hook_calls.append(f"registered_{plugin_info.name}")
        
        def unregistration_hook(plugin_info):
            hook_calls.append(f"unregistered_{plugin_info.name}")
        
        self.registry.add_hook('plugin_registered', registration_hook)
        self.registry.add_hook('plugin_unregistered', unregistration_hook)
        
        # Register plugin
        plugin = MockGatePlugin()
        self.registry.register_plugin(plugin)
        
        assert "registered_MockGatePlugin" in hook_calls
        
        # Unregister plugin
        self.registry.unregister_plugin("MockGatePlugin")
        
        assert "unregistered_MockGatePlugin" in hook_calls
        
        # Remove hooks
        self.registry.remove_hook('plugin_registered', registration_hook)
        self.registry.remove_hook('plugin_unregistered', unregistration_hook)


@pytest.mark.skipif(not HAS_PLUGINS, reason="plugins module not available")
class TestPluginManager:
    """Test PluginManager functionality."""
    
    def setup_method(self):
        """Set up test plugin manager."""
        self.manager = PluginManager()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test plugin manager."""
        # Cleanup all plugins
        for plugin_name in list(self.manager.registry._plugins.keys()):
            self.manager.registry.unregister_plugin(plugin_name)
        shutil.rmtree(self.temp_dir)
    
    def test_get_gates(self):
        """Test getting gates from gate plugins."""
        gate_plugin = MockGatePlugin()
        self.manager.registry.register_plugin(gate_plugin)
        
        gates = self.manager.get_gates()
        
        assert "MockGate" in gates
        assert "TestGate" in gates
        assert len(gates) == 2
    
    def test_get_algorithms(self):
        """Test getting algorithms from algorithm plugins."""
        algorithm_plugin = MockAlgorithmPlugin()
        self.manager.registry.register_plugin(algorithm_plugin)
        
        algorithms = self.manager.get_algorithms()
        
        assert "MockVQE" in algorithms
        assert "TestQAOA" in algorithms
        assert len(algorithms) == 2
    
    def test_get_backends(self):
        """Test getting backends from backend plugins."""
        # Available backend
        available_backend = MockBackendPlugin(available=True)
        self.manager.registry.register_plugin(available_backend)
        
        backends = self.manager.get_backends()
        assert "MockBackendPlugin" in backends
        
        # Unavailable backend
        unavailable_backend = MockBackendPlugin(available=False)
        # Need to change name to avoid conflict
        unavailable_backend.metadata.name = "UnavailableBackend"
        self.manager.registry.register_plugin(unavailable_backend)
        
        backends = self.manager.get_backends()
        assert "UnavailableBackend" not in backends  # Should be filtered out
    
    def test_process_circuit_through_middleware(self):
        """Test processing circuit through middleware plugins."""
        # Register middleware plugins with different priorities
        middleware1 = MockMiddlewarePlugin(priority=5)
        middleware1.metadata.name = "Middleware1"
        
        middleware2 = MockMiddlewarePlugin(priority=10)
        middleware2.metadata.name = "Middleware2"
        
        middleware3 = MockMiddlewarePlugin(priority=1)
        middleware3.metadata.name = "Middleware3"
        
        self.manager.registry.register_plugin(middleware1)
        self.manager.registry.register_plugin(middleware2) 
        self.manager.registry.register_plugin(middleware3)
        
        # Process circuit
        input_circuit = "test_circuit"
        result = self.manager.process_circuit_through_middleware(input_circuit)
        
        # Should be processed by all middleware in priority order (1, 5, 10)
        assert result == "processed_processed_processed_test_circuit"
    
    def test_multiple_plugin_types(self):
        """Test working with multiple plugin types simultaneously."""
        # Register plugins of different types
        gate_plugin = MockGatePlugin()
        algorithm_plugin = MockAlgorithmPlugin()
        backend_plugin = MockBackendPlugin()
        
        self.manager.registry.register_plugin(gate_plugin)
        self.manager.registry.register_plugin(algorithm_plugin)
        self.manager.registry.register_plugin(backend_plugin)
        
        # Get components from each type
        gates = self.manager.get_gates()
        algorithms = self.manager.get_algorithms()
        backends = self.manager.get_backends()
        
        assert len(gates) == 2
        assert len(algorithms) == 2
        assert len(backends) == 1
        
        # Verify no cross-contamination
        assert "MockVQE" not in gates
        assert "MockGate" not in algorithms


@pytest.mark.skipif(not HAS_PLUGINS, reason="plugins module not available")
class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def setup_method(self):
        """Set up for tests."""
        # Reset global plugin manager
        import quantrs2.plugins as plugins_module
        plugins_module._plugin_manager = None
    
    def teardown_method(self):
        """Clean up after tests."""
        # Clean up global plugin manager
        manager = get_plugin_manager()
        for plugin_name in list(manager.registry._plugins.keys()):
            manager.registry.unregister_plugin(plugin_name)
    
    def test_get_plugin_manager(self):
        """Test getting global plugin manager."""
        manager1 = get_plugin_manager()
        manager2 = get_plugin_manager()
        
        # Should be singleton
        assert manager1 is manager2
        assert isinstance(manager1, PluginManager)
    
    def test_register_plugin_convenience(self):
        """Test registering plugin via convenience function."""
        plugin = MockGatePlugin()
        
        success = register_plugin(plugin)
        
        assert success is True
        
        # Should be registered in global manager
        manager = get_plugin_manager()
        assert "MockGatePlugin" in manager.registry._plugins
    
    def test_get_available_gates(self):
        """Test getting available gates via convenience function."""
        plugin = MockGatePlugin()
        register_plugin(plugin)
        
        gates = get_available_gates()
        
        assert "MockGate" in gates
        assert "TestGate" in gates
    
    def test_get_available_algorithms(self):
        """Test getting available algorithms via convenience function."""
        plugin = MockAlgorithmPlugin()
        register_plugin(plugin)
        
        algorithms = get_available_algorithms()
        
        assert "MockVQE" in algorithms
        assert "TestQAOA" in algorithms
    
    def test_get_available_backends(self):
        """Test getting available backends via convenience function."""
        plugin = MockBackendPlugin()
        register_plugin(plugin)
        
        backends = get_available_backends()
        
        assert "MockBackendPlugin" in backends


@pytest.mark.skipif(not HAS_PLUGINS, reason="plugins module not available")
class TestPluginValidation:
    """Test plugin validation functionality."""
    
    def test_validate_config_with_schema(self):
        """Test config validation with schema."""
        plugin = MockBackendPlugin()
        
        # Valid config
        valid_config = {"backend_url": "http://test.com"}
        assert plugin.validate_config(valid_config) is True
        
        # Invalid config (missing required field)
        invalid_config = {"timeout": 30}
        assert plugin.validate_config(invalid_config) is False
    
    def test_validate_config_without_schema(self):
        """Test config validation without schema."""
        plugin = MockGatePlugin()  # No config schema
        
        # Any config should be valid
        assert plugin.validate_config({}) is True
        assert plugin.validate_config({"any": "value"}) is True
    
    def test_plugin_initialization_with_valid_config(self):
        """Test plugin initialization with valid config."""
        plugin = MockBackendPlugin()
        config = {"backend_url": "http://test.com", "timeout": 60}
        
        success = plugin.initialize(config)
        assert success is True
    
    def test_plugin_initialization_with_invalid_config(self):
        """Test plugin initialization with invalid config."""
        plugin = MockBackendPlugin()
        invalid_config = {"invalid_key": "value"}
        
        success = plugin.initialize(invalid_config)
        assert success is False


@pytest.mark.skipif(not HAS_PLUGINS, reason="plugins module not available")
class TestPluginErrorHandling:
    """Test error handling in plugin system."""
    
    def setup_method(self):
        """Set up test registry."""
        self.registry = PluginRegistry()
    
    def teardown_method(self):
        """Clean up test registry."""
        for plugin_name in list(self.registry._plugins.keys()):
            self.registry.unregister_plugin(plugin_name)
    
    def test_error_in_plugin_initialization(self):
        """Test handling of errors during plugin initialization."""
        plugin = FailingPlugin()
        
        success = self.registry.register_plugin(plugin)
        
        assert success is False
        assert "FailingPlugin" not in self.registry._plugins
    
    def test_error_in_plugin_cleanup(self):
        """Test handling of errors during plugin cleanup."""
        class CleanupErrorPlugin(MockGatePlugin):
            def cleanup(self):
                raise Exception("Cleanup error")
        
        plugin = CleanupErrorPlugin()
        self.registry.register_plugin(plugin)
        
        # Should handle cleanup error gracefully
        success = self.registry.unregister_plugin("MockGatePlugin")
        assert success is True
    
    def test_error_in_hook_callback(self):
        """Test handling of errors in hook callbacks."""
        def failing_hook(plugin_info):
            raise Exception("Hook error")
        
        self.registry.add_hook('plugin_registered', failing_hook)
        
        plugin = MockGatePlugin()
        
        # Should not prevent plugin registration
        success = self.registry.register_plugin(plugin)
        assert success is True
        assert "MockGatePlugin" in self.registry._plugins
    
    def test_activate_nonexistent_plugin(self):
        """Test activating nonexistent plugin."""
        success = self.registry.activate_plugin("NonexistentPlugin")
        assert success is False
    
    def test_deactivate_nonexistent_plugin(self):
        """Test deactivating nonexistent plugin."""
        success = self.registry.deactivate_plugin("NonexistentPlugin")
        assert success is False


@pytest.mark.skipif(not HAS_PLUGINS, reason="plugins module not available")
class TestPluginIntegration:
    """Test plugin integration scenarios."""
    
    def setup_method(self):
        """Set up test environment."""
        self.manager = PluginManager()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test environment."""
        for plugin_name in list(self.manager.registry._plugins.keys()):
            self.manager.registry.unregister_plugin(plugin_name)
        shutil.rmtree(self.temp_dir)
    
    def test_plugin_workflow(self):
        """Test complete plugin workflow."""
        # Register multiple plugins
        gate_plugin = MockGatePlugin()
        algorithm_plugin = MockAlgorithmPlugin()
        middleware_plugin = MockMiddlewarePlugin()
        
        self.manager.registry.register_plugin(gate_plugin)
        self.manager.registry.register_plugin(algorithm_plugin)
        self.manager.registry.register_plugin(middleware_plugin)
        
        # Get available components
        gates = self.manager.get_gates()
        algorithms = self.manager.get_algorithms()
        
        assert len(gates) == 2
        assert len(algorithms) == 2
        
        # Process circuit through middleware
        circuit = "test"
        processed = self.manager.process_circuit_through_middleware(circuit)
        assert processed == "processed_test"
        
        # Deactivate a plugin
        self.manager.registry.deactivate_plugin("MockGatePlugin")
        
        # Gates should no longer be available
        gates = self.manager.get_gates()
        assert len(gates) == 0
        
        # But algorithms should still be available
        algorithms = self.manager.get_algorithms()
        assert len(algorithms) == 2
    
    def test_plugin_dependencies(self):
        """Test plugin with dependencies."""
        # Algorithm plugin has numpy dependency
        algorithm_plugin = MockAlgorithmPlugin()
        
        # Should still register (we don't check actual dependency availability in mock)
        success = self.manager.registry.register_plugin(algorithm_plugin)
        assert success is True
        
        plugin_info = self.manager.registry._plugins["MockAlgorithmPlugin"]
        assert "numpy" in plugin_info.metadata.dependencies
    
    def test_multiple_plugins_same_type(self):
        """Test multiple plugins of the same type."""
        # Create two different gate plugins
        gate_plugin1 = MockGatePlugin()
        
        class SecondGatePlugin(MockGatePlugin):
            @property
            def metadata(self):
                meta = super().metadata
                meta.name = "SecondGatePlugin"
                return meta
            
            def get_gate_classes(self):
                return {"SecondGate": object}
        
        gate_plugin2 = SecondGatePlugin()
        
        self.manager.registry.register_plugin(gate_plugin1)
        self.manager.registry.register_plugin(gate_plugin2)
        
        # Should have gates from both plugins
        gates = self.manager.get_gates()
        assert len(gates) == 3  # MockGate, TestGate, SecondGate
        assert "MockGate" in gates
        assert "TestGate" in gates
        assert "SecondGate" in gates


@pytest.mark.skipif(not HAS_PLUGINS, reason="plugins module not available")
class TestPluginPerformance:
    """Test plugin system performance characteristics."""
    
    def setup_method(self):
        """Set up test environment."""
        self.manager = PluginManager()
    
    def teardown_method(self):
        """Clean up test environment."""
        for plugin_name in list(self.manager.registry._plugins.keys()):
            self.manager.registry.unregister_plugin(plugin_name)
    
    def test_many_plugins_registration(self):
        """Test registering many plugins."""
        import time
        
        start_time = time.time()
        
        # Register many plugins
        for i in range(50):
            class TestPlugin(MockGatePlugin):
                @property
                def metadata(self):
                    meta = super().metadata
                    meta.name = f"TestPlugin{i}"
                    return meta
            
            plugin = TestPlugin()
            success = self.manager.registry.register_plugin(plugin)
            assert success is True
        
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 5.0  # 5 seconds for 50 plugins
        
        # Verify all plugins registered
        assert len(self.manager.registry._plugins) == 50
    
    def test_plugin_lookup_performance(self):
        """Test plugin lookup performance."""
        import time
        
        # Register some plugins
        for i in range(20):
            class TestPlugin(MockGatePlugin):
                @property
                def metadata(self):
                    meta = super().metadata
                    meta.name = f"TestPlugin{i}"
                    return meta
            
            plugin = TestPlugin()
            self.manager.registry.register_plugin(plugin)
        
        start_time = time.time()
        
        # Perform many lookups
        for i in range(100):
            plugin_name = f"TestPlugin{i % 20}"
            plugin = self.manager.registry.get_plugin(plugin_name)
            assert plugin is not None
        
        end_time = time.time()
        
        # Should be fast
        assert end_time - start_time < 1.0  # 1 second for 100 lookups
    
    def test_middleware_processing_performance(self):
        """Test middleware processing performance."""
        import time
        
        # Register multiple middleware plugins
        for i in range(10):
            class TestMiddleware(MockMiddlewarePlugin):
                @property
                def metadata(self):
                    meta = super().metadata
                    meta.name = f"TestMiddleware{i}"
                    return meta
            
            middleware = TestMiddleware(priority=i)
            self.manager.registry.register_plugin(middleware)
        
        start_time = time.time()
        
        # Process many circuits
        for i in range(100):
            circuit = f"circuit_{i}"
            result = self.manager.process_circuit_through_middleware(circuit)
            # Should be processed by all 10 middleware plugins
            assert result.count("processed_") == 10
        
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 5.0  # 5 seconds for 100 circuits × 10 middleware


if __name__ == "__main__":
    pytest.main([__file__])