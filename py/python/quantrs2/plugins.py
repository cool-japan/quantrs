"""
Plugin System for QuantRS2

This module provides a comprehensive plugin architecture for extending
QuantRS2 functionality with custom gates, algorithms, backends, and more.
"""

import os
import sys
import importlib
import inspect
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union, Callable, Protocol
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import logging

# Set up logging
logger = logging.getLogger(__name__)


class PluginType(Enum):
    """Types of plugins supported by QuantRS2."""
    GATE = "gate"
    ALGORITHM = "algorithm"
    BACKEND = "backend"
    OPTIMIZER = "optimizer"
    VISUALIZER = "visualizer"
    CONVERTER = "converter"
    MIDDLEWARE = "middleware"
    EXTENSION = "extension"


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    min_quantrs_version: str = "0.1.0"
    max_quantrs_version: Optional[str] = None
    license: str = "MIT"
    homepage: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    entry_points: Dict[str, str] = field(default_factory=dict)
    config_schema: Optional[Dict[str, Any]] = None


class PluginInterface(ABC):
    """Base interface for all plugins."""
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Plugin metadata."""
        pass
    
    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize the plugin.
        
        Args:
            config: Plugin configuration
            
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate plugin configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if configuration is valid
        """
        if self.metadata.config_schema is None:
            return True
        
        # Basic validation - in practice, use jsonschema
        try:
            required_keys = self.metadata.config_schema.get('required', [])
            for key in required_keys:
                if key not in config:
                    return False
            return True
        except Exception:
            return False


class GatePlugin(PluginInterface):
    """Base class for gate plugins."""
    
    @abstractmethod
    def get_gate_classes(self) -> Dict[str, Type]:
        """
        Get gate classes provided by this plugin.
        
        Returns:
            Dictionary mapping gate names to gate classes
        """
        pass
    
    @abstractmethod
    def get_gate_functions(self) -> Dict[str, Callable]:
        """
        Get gate convenience functions provided by this plugin.
        
        Returns:
            Dictionary mapping function names to functions
        """
        pass


class AlgorithmPlugin(PluginInterface):
    """Base class for algorithm plugins."""
    
    @abstractmethod
    def get_algorithms(self) -> Dict[str, Type]:
        """
        Get algorithm classes provided by this plugin.
        
        Returns:
            Dictionary mapping algorithm names to classes
        """
        pass
    
    @abstractmethod
    def get_algorithm_templates(self) -> Dict[str, Callable]:
        """
        Get algorithm template functions.
        
        Returns:
            Dictionary mapping template names to functions
        """
        pass


class BackendPlugin(PluginInterface):
    """Base class for backend plugins."""
    
    @abstractmethod
    def get_backend_class(self) -> Type:
        """
        Get the backend class.
        
        Returns:
            Backend class
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the backend is available.
        
        Returns:
            True if backend can be used
        """
        pass


class OptimizerPlugin(PluginInterface):
    """Base class for optimizer plugins."""
    
    @abstractmethod
    def get_optimizer_classes(self) -> Dict[str, Type]:
        """
        Get optimizer classes provided by this plugin.
        
        Returns:
            Dictionary mapping optimizer names to classes
        """
        pass


class VisualizerPlugin(PluginInterface):
    """Base class for visualizer plugins."""
    
    @abstractmethod
    def get_visualizer_classes(self) -> Dict[str, Type]:
        """
        Get visualizer classes provided by this plugin.
        
        Returns:
            Dictionary mapping visualizer names to classes
        """
        pass
    
    @abstractmethod
    def get_plot_functions(self) -> Dict[str, Callable]:
        """
        Get plotting functions provided by this plugin.
        
        Returns:
            Dictionary mapping function names to functions
        """
        pass


class ConverterPlugin(PluginInterface):
    """Base class for circuit converter plugins."""
    
    @abstractmethod
    def get_import_formats(self) -> List[str]:
        """
        Get supported import formats.
        
        Returns:
            List of supported file extensions
        """
        pass
    
    @abstractmethod
    def get_export_formats(self) -> List[str]:
        """
        Get supported export formats.
        
        Returns:
            List of supported file extensions
        """
        pass
    
    @abstractmethod
    def import_circuit(self, file_path: str, format: str) -> Any:
        """
        Import a circuit from file.
        
        Args:
            file_path: Path to input file
            format: File format
            
        Returns:
            Imported circuit object
        """
        pass
    
    @abstractmethod
    def export_circuit(self, circuit: Any, file_path: str, format: str) -> None:
        """
        Export a circuit to file.
        
        Args:
            circuit: Circuit to export
            file_path: Path to output file
            format: File format
        """
        pass


class MiddlewarePlugin(PluginInterface):
    """Base class for middleware plugins."""
    
    @abstractmethod
    def process_circuit(self, circuit: Any, context: Dict[str, Any]) -> Any:
        """
        Process a circuit through middleware.
        
        Args:
            circuit: Input circuit
            context: Processing context
            
        Returns:
            Processed circuit
        """
        pass
    
    @abstractmethod
    def get_middleware_priority(self) -> int:
        """
        Get middleware processing priority.
        
        Returns:
            Priority value (lower = higher priority)
        """
        pass


@dataclass
class PluginInfo:
    """Information about a loaded plugin."""
    name: str
    plugin: PluginInterface
    metadata: PluginMetadata
    config: Dict[str, Any]
    is_active: bool = True
    load_time: Optional[float] = None
    error: Optional[str] = None


class PluginRegistry:
    """Registry for managing plugins."""
    
    def __init__(self):
        self._plugins: Dict[str, PluginInfo] = {}
        self._plugins_by_type: Dict[PluginType, List[str]] = {
            plugin_type: [] for plugin_type in PluginType
        }
        self._search_paths: List[Path] = []
        self._hooks: Dict[str, List[Callable]] = {}
    
    def add_search_path(self, path: Union[str, Path]) -> None:
        """Add a directory to search for plugins."""
        path = Path(path)
        if path.exists() and path.is_dir():
            self._search_paths.append(path)
            logger.info(f"Added plugin search path: {path}")
    
    def register_plugin(self, plugin: PluginInterface, 
                       config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register a plugin instance.
        
        Args:
            plugin: Plugin instance
            config: Plugin configuration
            
        Returns:
            True if registration successful
        """
        try:
            metadata = plugin.metadata
            name = metadata.name
            
            if name in self._plugins:
                logger.warning(f"Plugin {name} already registered, replacing")
            
            # Validate configuration
            config = config or {}
            if not plugin.validate_config(config):
                logger.error(f"Invalid configuration for plugin {name}")
                return False
            
            # Initialize plugin
            import time
            start_time = time.time()
            
            if not plugin.initialize(config):
                logger.error(f"Failed to initialize plugin {name}")
                return False
            
            load_time = time.time() - start_time
            
            # Register plugin
            plugin_info = PluginInfo(
                name=name,
                plugin=plugin,
                metadata=metadata,
                config=config,
                load_time=load_time
            )
            
            self._plugins[name] = plugin_info
            self._plugins_by_type[metadata.plugin_type].append(name)
            
            logger.info(f"Registered plugin {name} (type: {metadata.plugin_type.value})")
            
            # Call registration hooks
            self._call_hooks('plugin_registered', plugin_info)
            
            return True
            
        except Exception as e:
            logger.error(f"Error registering plugin: {e}")
            return False
    
    def unregister_plugin(self, name: str) -> bool:
        """
        Unregister a plugin.
        
        Args:
            name: Plugin name
            
        Returns:
            True if unregistration successful
        """
        if name not in self._plugins:
            return False
        
        try:
            plugin_info = self._plugins[name]
            
            # Cleanup plugin
            plugin_info.plugin.cleanup()
            
            # Remove from registry
            del self._plugins[name]
            plugin_type = plugin_info.metadata.plugin_type
            if name in self._plugins_by_type[plugin_type]:
                self._plugins_by_type[plugin_type].remove(name)
            
            logger.info(f"Unregistered plugin {name}")
            
            # Call unregistration hooks
            self._call_hooks('plugin_unregistered', plugin_info)
            
            return True
            
        except Exception as e:
            logger.error(f"Error unregistering plugin {name}: {e}")
            return False
    
    def get_plugin(self, name: str) -> Optional[PluginInterface]:
        """Get a plugin by name."""
        plugin_info = self._plugins.get(name)
        return plugin_info.plugin if plugin_info else None
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginInterface]:
        """Get all plugins of a specific type."""
        plugin_names = self._plugins_by_type.get(plugin_type, [])
        return [self._plugins[name].plugin for name in plugin_names 
                if name in self._plugins and self._plugins[name].is_active]
    
    def list_plugins(self) -> List[PluginInfo]:
        """List all registered plugins."""
        return list(self._plugins.values())
    
    def activate_plugin(self, name: str) -> bool:
        """Activate a plugin."""
        if name in self._plugins:
            self._plugins[name].is_active = True
            logger.info(f"Activated plugin {name}")
            return True
        return False
    
    def deactivate_plugin(self, name: str) -> bool:
        """Deactivate a plugin."""
        if name in self._plugins:
            self._plugins[name].is_active = False
            logger.info(f"Deactivated plugin {name}")
            return True
        return False
    
    def discover_plugins(self) -> List[str]:
        """
        Discover plugins in search paths.
        
        Returns:
            List of discovered plugin module names
        """
        discovered = []
        
        for search_path in self._search_paths:
            try:
                for item in search_path.iterdir():
                    if item.is_file() and item.suffix == '.py':
                        # Single file plugin
                        module_name = item.stem
                        if self._is_plugin_module(item):
                            discovered.append(module_name)
                    
                    elif item.is_dir() and (item / '__init__.py').exists():
                        # Package plugin
                        if self._is_plugin_package(item):
                            discovered.append(item.name)
            
            except Exception as e:
                logger.error(f"Error discovering plugins in {search_path}: {e}")
        
        return discovered
    
    def load_plugin_from_module(self, module_name: str, 
                               config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Load a plugin from a module.
        
        Args:
            module_name: Module name to load
            config: Plugin configuration
            
        Returns:
            True if loading successful
        """
        try:
            # Add search paths to sys.path temporarily
            original_path = sys.path.copy()
            for search_path in self._search_paths:
                if str(search_path) not in sys.path:
                    sys.path.insert(0, str(search_path))
            
            try:
                # Import module
                module = importlib.import_module(module_name)
                
                # Find plugin classes
                plugin_classes = self._find_plugin_classes(module)
                
                if not plugin_classes:
                    logger.error(f"No plugin classes found in module {module_name}")
                    return False
                
                # Load first plugin class found
                plugin_class = plugin_classes[0]
                plugin_instance = plugin_class()
                
                return self.register_plugin(plugin_instance, config)
                
            finally:
                # Restore sys.path
                sys.path = original_path
                
        except Exception as e:
            logger.error(f"Error loading plugin from module {module_name}: {e}")
            return False
    
    def load_plugin_from_file(self, file_path: Union[str, Path], 
                             config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Load a plugin from a file.
        
        Args:
            file_path: Path to plugin file
            config: Plugin configuration
            
        Returns:
            True if loading successful
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"Plugin file not found: {file_path}")
            return False
        
        try:
            # Load module from file
            spec = importlib.util.spec_from_file_location("plugin_module", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin classes
            plugin_classes = self._find_plugin_classes(module)
            
            if not plugin_classes:
                logger.error(f"No plugin classes found in file {file_path}")
                return False
            
            # Load first plugin class found
            plugin_class = plugin_classes[0]
            plugin_instance = plugin_class()
            
            return self.register_plugin(plugin_instance, config)
            
        except Exception as e:
            logger.error(f"Error loading plugin from file {file_path}: {e}")
            return False
    
    def add_hook(self, event: str, callback: Callable) -> None:
        """Add a hook for plugin events."""
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)
    
    def remove_hook(self, event: str, callback: Callable) -> None:
        """Remove a hook for plugin events."""
        if event in self._hooks and callback in self._hooks[event]:
            self._hooks[event].remove(callback)
    
    def _call_hooks(self, event: str, *args, **kwargs) -> None:
        """Call hooks for an event."""
        for callback in self._hooks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in hook for event {event}: {e}")
    
    def _is_plugin_module(self, file_path: Path) -> bool:
        """Check if a file is a plugin module."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                return 'PluginInterface' in content or 'Plugin' in content
        except Exception:
            return False
    
    def _is_plugin_package(self, dir_path: Path) -> bool:
        """Check if a directory is a plugin package."""
        init_file = dir_path / '__init__.py'
        return self._is_plugin_module(init_file)
    
    def _find_plugin_classes(self, module) -> List[Type[PluginInterface]]:
        """Find plugin classes in a module."""
        plugin_classes = []
        
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (issubclass(obj, PluginInterface) and 
                obj is not PluginInterface and
                not inspect.isabstract(obj)):
                plugin_classes.append(obj)
        
        return plugin_classes


class PluginManager:
    """Main plugin manager for QuantRS2."""
    
    def __init__(self):
        self.registry = PluginRegistry()
        self._setup_default_search_paths()
    
    def _setup_default_search_paths(self):
        """Setup default plugin search paths."""
        # User plugins directory
        user_plugins = Path.home() / '.quantrs2' / 'plugins'
        if user_plugins.exists():
            self.registry.add_search_path(user_plugins)
        
        # System plugins directory
        try:
            import quantrs2
            quantrs2_path = Path(quantrs2.__file__).parent
            system_plugins = quantrs2_path / 'plugins'
            if system_plugins.exists():
                self.registry.add_search_path(system_plugins)
        except ImportError:
            pass
        
        # Environment variable
        env_path = os.environ.get('QUANTRS2_PLUGIN_PATH')
        if env_path:
            for path in env_path.split(os.pathsep):
                self.registry.add_search_path(Path(path))
    
    def load_all_plugins(self) -> Dict[str, bool]:
        """
        Load all discoverable plugins.
        
        Returns:
            Dictionary mapping plugin names to load success status
        """
        discovered = self.registry.discover_plugins()
        results = {}
        
        for module_name in discovered:
            results[module_name] = self.registry.load_plugin_from_module(module_name)
        
        return results
    
    def get_gates(self) -> Dict[str, Any]:
        """Get all gates from gate plugins."""
        gates = {}
        gate_plugins = self.registry.get_plugins_by_type(PluginType.GATE)
        
        for plugin in gate_plugins:
            try:
                plugin_gates = plugin.get_gate_classes()
                gates.update(plugin_gates)
            except Exception as e:
                logger.error(f"Error getting gates from plugin {plugin.metadata.name}: {e}")
        
        return gates
    
    def get_algorithms(self) -> Dict[str, Any]:
        """Get all algorithms from algorithm plugins."""
        algorithms = {}
        algorithm_plugins = self.registry.get_plugins_by_type(PluginType.ALGORITHM)
        
        for plugin in algorithm_plugins:
            try:
                plugin_algorithms = plugin.get_algorithms()
                algorithms.update(plugin_algorithms)
            except Exception as e:
                logger.error(f"Error getting algorithms from plugin {plugin.metadata.name}: {e}")
        
        return algorithms
    
    def get_backends(self) -> Dict[str, Any]:
        """Get all backends from backend plugins."""
        backends = {}
        backend_plugins = self.registry.get_plugins_by_type(PluginType.BACKEND)
        
        for plugin in backend_plugins:
            try:
                if plugin.is_available():
                    backend_class = plugin.get_backend_class()
                    backends[plugin.metadata.name] = backend_class
            except Exception as e:
                logger.error(f"Error getting backend from plugin {plugin.metadata.name}: {e}")
        
        return backends
    
    def get_visualizers(self) -> Dict[str, Any]:
        """Get all visualizers from visualizer plugins."""
        visualizers = {}
        viz_plugins = self.registry.get_plugins_by_type(PluginType.VISUALIZER)
        
        for plugin in viz_plugins:
            try:
                plugin_visualizers = plugin.get_visualizer_classes()
                visualizers.update(plugin_visualizers)
            except Exception as e:
                logger.error(f"Error getting visualizers from plugin {plugin.metadata.name}: {e}")
        
        return visualizers
    
    def process_circuit_through_middleware(self, circuit: Any, 
                                         context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Process a circuit through all active middleware plugins.
        
        Args:
            circuit: Input circuit
            context: Processing context
            
        Returns:
            Processed circuit
        """
        context = context or {}
        middleware_plugins = self.registry.get_plugins_by_type(PluginType.MIDDLEWARE)
        
        # Sort by priority
        middleware_plugins.sort(key=lambda p: p.get_middleware_priority())
        
        result_circuit = circuit
        for plugin in middleware_plugins:
            try:
                result_circuit = plugin.process_circuit(result_circuit, context)
            except Exception as e:
                logger.error(f"Error in middleware plugin {plugin.metadata.name}: {e}")
        
        return result_circuit


# Global plugin manager instance
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


# Convenience functions
def register_plugin(plugin: PluginInterface, 
                   config: Optional[Dict[str, Any]] = None) -> bool:
    """Register a plugin with the global plugin manager."""
    return get_plugin_manager().registry.register_plugin(plugin, config)


def load_plugin(module_name: str, 
               config: Optional[Dict[str, Any]] = None) -> bool:
    """Load a plugin by module name."""
    return get_plugin_manager().registry.load_plugin_from_module(module_name, config)


def get_available_gates() -> Dict[str, Any]:
    """Get all available gates from plugins."""
    return get_plugin_manager().get_gates()


def get_available_algorithms() -> Dict[str, Any]:
    """Get all available algorithms from plugins."""
    return get_plugin_manager().get_algorithms()


def get_available_backends() -> Dict[str, Any]:
    """Get all available backends from plugins."""
    return get_plugin_manager().get_backends()


__all__ = [
    'PluginType',
    'PluginMetadata',
    'PluginInterface',
    'GatePlugin',
    'AlgorithmPlugin', 
    'BackendPlugin',
    'OptimizerPlugin',
    'VisualizerPlugin',
    'ConverterPlugin',
    'MiddlewarePlugin',
    'PluginInfo',
    'PluginRegistry',
    'PluginManager',
    'get_plugin_manager',
    'register_plugin',
    'load_plugin',
    'get_available_gates',
    'get_available_algorithms',
    'get_available_backends',
]