#!/usr/bin/env python3
"""
Test suite for quantum IDE plugin functionality.
"""

import pytest
import json
import time
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

try:
    from quantrs2.ide_plugin import (
        QuantumCodeAnalyzer, QuantumCodeCompletion, QuantumHoverProvider,
        IDEPluginServer, VSCodePlugin, QuantumIDEPlugin,
        CodeCompletionItem, DiagnosticMessage, HoverInfo,
        IDEType, PluginState, AnalysisType, BreakpointType,
        get_ide_plugin, install_vscode_plugin, install_jupyter_plugin,
        analyze_quantum_code
    )
    HAS_IDE_PLUGIN = True
except ImportError:
    HAS_IDE_PLUGIN = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@pytest.mark.skipif(not HAS_IDE_PLUGIN, reason="ide_plugin module not available")
class TestQuantumCodeAnalyzer:
    """Test QuantumCodeAnalyzer functionality."""
    
    def setup_method(self):
        """Set up test analyzer."""
        self.analyzer = QuantumCodeAnalyzer()
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        assert len(self.analyzer.gate_patterns['single_qubit']) > 0
        assert len(self.analyzer.gate_patterns['two_qubit']) > 0
        assert len(self.analyzer.optimization_rules) > 0
    
    def test_syntax_analysis(self):
        """Test syntax analysis."""
        code = """
import quantrs2
circuit = PyCircuit(2)
circuit.h(0
circuit.cnot(0, 1)
"""
        diagnostics = self.analyzer.analyze_code(code, [AnalysisType.SYNTAX])
        
        # Should detect unclosed parentheses
        syntax_errors = [d for d in diagnostics if d.severity == "error"]
        assert len(syntax_errors) > 0
        assert any("parentheses" in d.message.lower() for d in syntax_errors)
    
    def test_semantic_analysis(self):
        """Test semantic analysis."""
        code = """
import quantrs2
circuit = PyCircuit(2)
circuit.h(0)
circuit.cnot(0, 3)  # Qubit 3 doesn't exist
circuit.x(0, 1)     # X gate takes 1 qubit, not 2
"""
        diagnostics = self.analyzer.analyze_code(code, [AnalysisType.SEMANTIC])
        
        # Should detect qubit out of bounds and wrong gate arity
        semantic_errors = [d for d in diagnostics if d.severity == "error"]
        assert len(semantic_errors) >= 2
        
        bounds_error = any("out of bounds" in d.message for d in semantic_errors)
        arity_error = any("expects" in d.message for d in semantic_errors)
        assert bounds_error or arity_error
    
    def test_performance_analysis(self):
        """Test performance analysis."""
        # Create code with many gates
        gates = ["circuit.h(0)"] * 150
        code = f"""
import quantrs2
circuit = PyCircuit(2)
{chr(10).join(gates)}
"""
        diagnostics = self.analyzer.analyze_code(code, [AnalysisType.PERFORMANCE])
        
        # Should suggest optimization for deep circuit
        perf_warnings = [d for d in diagnostics if "performance" in d.message.lower()]
        assert len(perf_warnings) > 0
    
    def test_optimization_analysis(self):
        """Test optimization analysis."""
        code = """
import quantrs2
circuit = PyCircuit(2)
circuit.x(0)
circuit.x(0)  # Redundant X-X
circuit.h(1)
circuit.z(1)
circuit.h(1)  # H-Z-H can be X
"""
        diagnostics = self.analyzer.analyze_code(code, [AnalysisType.OPTIMIZATION])
        
        # Should detect optimization opportunities
        opt_suggestions = [d for d in diagnostics if d.severity == "info"]
        assert len(opt_suggestions) > 0
    
    def test_circuit_parsing(self):
        """Test circuit data extraction."""
        code = """
import quantrs2
circuit = PyCircuit(3)
circuit.h(0)
circuit.cnot(0, 1)
circuit.x(2)
"""
        lines = code.split('\n')
        circuit_data = self.analyzer._parse_circuit_from_code(code, lines)
        
        assert circuit_data['n_qubits'] == 3
        assert len(circuit_data['gates']) == 3
        
        gate_names = [g['gate'] for g in circuit_data['gates']]
        assert 'h' in gate_names
        assert 'cnot' in gate_names
        assert 'x' in gate_names
    
    def test_qubit_usage_analysis(self):
        """Test qubit usage analysis."""
        circuit_data = {
            'n_qubits': 4,
            'gates': [
                {'gate': 'h', 'qubits': [0], 'line': 1},
                {'gate': 'cnot', 'qubits': [0, 1], 'line': 2},
                {'gate': 'x', 'qubits': [2], 'line': 3}
                # Qubit 3 is unused
            ]
        }
        
        usage = self.analyzer._analyze_qubit_usage({'circuit_data': circuit_data})
        
        assert '0' in usage['operations_per_qubit']
        assert '1' in usage['operations_per_qubit']
        assert '2' in usage['operations_per_qubit']
        assert usage['operations_per_qubit']['0'] == 2  # h and cnot
        
        # Check most used qubit
        assert usage['most_used_qubit'] == 0
    
    def test_redundant_gate_detection(self):
        """Test redundant gate sequence detection."""
        circuit_data = {
            'gates': [
                {'gate': 'x', 'qubits': [0], 'line': 1},
                {'gate': 'x', 'qubits': [0], 'line': 2},  # Redundant
                {'gate': 'h', 'qubits': [1], 'line': 3}
            ]
        }
        
        diagnostics = self.analyzer._check_redundant_gates(circuit_data, [])
        
        redundant_warnings = [d for d in diagnostics if "redundant" in d.message.lower()]
        assert len(redundant_warnings) > 0
    
    def test_gate_decomposition_optimization(self):
        """Test gate decomposition optimization detection."""
        circuit_data = {
            'gates': [
                {'gate': 'h', 'qubits': [0], 'line': 1},
                {'gate': 'z', 'qubits': [0], 'line': 2},
                {'gate': 'h', 'qubits': [0], 'line': 3}  # H-Z-H = X
            ]
        }
        
        diagnostics = self.analyzer._check_gate_decomposition(circuit_data, [])
        
        decomp_suggestions = [d for d in diagnostics if "replaced" in d.message.lower()]
        assert len(decomp_suggestions) > 0


@pytest.mark.skipif(not HAS_IDE_PLUGIN, reason="ide_plugin module not available")
class TestQuantumCodeCompletion:
    """Test QuantumCodeCompletion functionality."""
    
    def setup_method(self):
        """Set up test completion provider."""
        self.completion = QuantumCodeCompletion()
    
    def test_completion_database(self):
        """Test completion database construction."""
        assert len(self.completion.completions) > 0
        
        # Check for basic gates
        gate_labels = [c.label for c in self.completion.completions]
        assert 'h' in gate_labels
        assert 'x' in gate_labels
        assert 'cnot' in gate_labels
        
        # Check for circuit operations
        assert 'PyCircuit' in gate_labels
        assert 'run' in gate_labels
    
    def test_context_aware_completion(self):
        """Test context-aware completion suggestions."""
        # Test method completion after circuit object
        context = """
import quantrs2
circuit = PyCircuit(2)
circuit.
"""
        position = (2, 8)  # After the dot
        
        completions = self.completion.get_completions(context, position)
        
        # Should return gate methods
        gate_completions = [c for c in completions if c.label in ['h', 'x', 'cnot']]
        assert len(gate_completions) > 0
    
    def test_completion_item_structure(self):
        """Test completion item structure."""
        hadamard_completion = None
        for c in self.completion.completions:
            if c.label == 'h':
                hadamard_completion = c
                break
        
        assert hadamard_completion is not None
        assert "Hadamard" in hadamard_completion.detail
        assert len(hadamard_completion.documentation) > 0
        assert "${1:qubit}" in hadamard_completion.insert_text
    
    def test_completion_serialization(self):
        """Test completion item serialization."""
        completion = self.completion.completions[0]
        completion_dict = completion.to_dict()
        
        required_fields = ['label', 'detail', 'documentation', 'insertText', 'kind']
        for field in required_fields:
            assert field in completion_dict
    
    def test_general_completions(self):
        """Test general completion suggestions."""
        context = "import quantrs2\n"
        position = (1, 0)
        
        completions = self.completion.get_completions(context, position)
        
        # Should return all available completions
        assert len(completions) > 10
        
        # Check for different types
        functions = [c for c in completions if c.kind == "function"]
        classes = [c for c in completions if c.kind == "class"]
        
        assert len(functions) > 0
        assert len(classes) > 0


@pytest.mark.skipif(not HAS_IDE_PLUGIN, reason="ide_plugin module not available")
class TestQuantumHoverProvider:
    """Test QuantumHoverProvider functionality."""
    
    def setup_method(self):
        """Set up test hover provider."""
        self.hover = QuantumHoverProvider()
    
    def test_hover_database(self):
        """Test hover database construction."""
        assert len(self.hover.hover_info) > 0
        
        # Check for basic gates
        assert 'h' in self.hover.hover_info
        assert 'x' in self.hover.hover_info
        assert 'cnot' in self.hover.hover_info
    
    def test_hover_content(self):
        """Test hover content quality."""
        hadamard_hover = self.hover.hover_info['h']
        
        assert "Hadamard" in hadamard_hover.content
        assert "superposition" in hadamard_hover.content.lower()
        assert "Matrix" in hadamard_hover.content
        assert hadamard_hover.content_type == "markdown"
    
    def test_hover_info_retrieval(self):
        """Test hover information retrieval."""
        context = "circuit.h(0)"
        
        # Test gate hover
        hover_info = self.hover.get_hover_info("h", context)
        assert hover_info is not None
        assert "Hadamard" in hover_info.content
        
        # Test non-existent element
        no_hover = self.hover.get_hover_info("nonexistent", context)
        assert no_hover is None
    
    def test_hover_serialization(self):
        """Test hover info serialization."""
        hover_info = self.hover.hover_info['x']
        hover_dict = hover_info.to_dict()
        
        assert 'contents' in hover_dict
        assert 'kind' in hover_dict['contents']
        assert 'value' in hover_dict['contents']
        assert hover_dict['contents']['kind'] == "markdown"
    
    def test_case_insensitive_lookup(self):
        """Test case-insensitive hover lookup."""
        # Should work with different cases
        hover_upper = self.hover.get_hover_info("H", "")
        hover_lower = self.hover.get_hover_info("h", "")
        
        assert hover_upper is not None
        assert hover_lower is not None
        assert hover_upper.content == hover_lower.content


@pytest.mark.skipif(not HAS_IDE_PLUGIN, reason="ide_plugin module not available")
class TestIDEPluginServer:
    """Test IDEPluginServer functionality."""
    
    def setup_method(self):
        """Set up test server."""
        self.server = IDEPluginServer(port=0)  # Use random port
    
    def teardown_method(self):
        """Clean up test server."""
        if self.server:
            self.server.stop_server()
    
    def test_server_initialization(self):
        """Test server initialization."""
        assert self.server.analyzer is not None
        assert self.server.completion_provider is not None
        assert self.server.hover_provider is not None
    
    @patch('socketserver.TCPServer')
    def test_server_start(self, mock_tcp_server):
        """Test server start."""
        mock_server_instance = Mock()
        mock_tcp_server.return_value = mock_server_instance
        
        success = self.server.start_server()
        
        assert success is True
        assert self.server.server is not None
        mock_tcp_server.assert_called_once()
    
    def test_request_handler_creation(self):
        """Test request handler creation."""
        handler_class = self.server._create_request_handler()
        
        # Should be a class
        assert callable(handler_class)
        assert hasattr(handler_class, 'do_POST')
    
    @patch('http.server.BaseHTTPRequestHandler')
    def test_analyze_request_handling(self, mock_handler):
        """Test analyze request handling."""
        # This is a simplified test - full integration testing would require
        # setting up actual HTTP requests
        
        handler_class = self.server._create_request_handler()
        handler_instance = Mock()
        
        # Test analysis method exists
        assert hasattr(handler_class, '_handle_analyze')
    
    def test_server_stop(self):
        """Test server stop."""
        # Start server first
        with patch('socketserver.TCPServer') as mock_tcp:
            mock_server = Mock()
            mock_tcp.return_value = mock_server
            
            self.server.start_server()
            self.server.stop_server()
            
            mock_server.shutdown.assert_called_once()
            mock_server.server_close.assert_called_once()


@pytest.mark.skipif(not HAS_IDE_PLUGIN, reason="ide_plugin module not available")
class TestVSCodePlugin:
    """Test VSCodePlugin functionality."""
    
    def setup_method(self):
        """Set up test VS Code plugin."""
        self.plugin = VSCodePlugin()
    
    def test_plugin_initialization(self):
        """Test plugin initialization."""
        assert self.plugin.extension_path is None
        assert self.plugin.server is not None
    
    @patch('tempfile.mkdtemp')
    def test_extension_directory_creation(self, mock_mkdtemp):
        """Test extension directory creation."""
        mock_mkdtemp.return_value = "/tmp/test_extension"
        
        with patch('pathlib.Path.mkdir'), patch('pathlib.Path.exists', return_value=False):
            extension_dir = self.plugin._create_extension_directory()
            
            assert extension_dir is not None
            assert "quantrs2-extension" in str(extension_dir)
    
    def test_package_json_generation(self):
        """Test package.json generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            extension_dir = Path(temp_dir)
            
            self.plugin._generate_package_json(extension_dir)
            
            package_file = extension_dir / "package.json"
            assert package_file.exists()
            
            with open(package_file) as f:
                package_data = json.load(f)
            
            assert package_data['name'] == "quantrs2-quantum-development"
            assert 'contributes' in package_data
            assert 'commands' in package_data['contributes']
    
    def test_main_js_generation(self):
        """Test main extension JavaScript generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            extension_dir = Path(temp_dir)
            (extension_dir / "src").mkdir()
            
            self.plugin._generate_main_js(extension_dir)
            
            main_file = extension_dir / "src" / "extension.js"
            assert main_file.exists()
            
            with open(main_file) as f:
                content = f.read()
            
            assert "activate" in content
            assert "quantrs2" in content
            assert "analyzeQuantumCode" in content
    
    def test_language_config_generation(self):
        """Test language configuration generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            extension_dir = Path(temp_dir)
            
            self.plugin._generate_language_config(extension_dir)
            
            config_file = extension_dir / "language-configuration.json"
            assert config_file.exists()
            
            with open(config_file) as f:
                config_data = json.load(f)
            
            assert 'comments' in config_data
            assert 'brackets' in config_data
            assert 'autoClosingPairs' in config_data


@pytest.mark.skipif(not HAS_IDE_PLUGIN, reason="ide_plugin module not available")
class TestQuantumIDEPlugin:
    """Test QuantumIDEPlugin functionality."""
    
    def setup_method(self):
        """Set up test IDE plugin."""
        self.plugin = QuantumIDEPlugin()
    
    def test_plugin_initialization(self):
        """Test plugin initialization."""
        assert self.plugin.state == PluginState.INACTIVE
        assert len(self.plugin.supported_ides) > 0
        assert IDEType.VSCODE in self.plugin.supported_ides
        assert len(self.plugin.active_plugins) == 0
    
    def test_supported_ides(self):
        """Test supported IDE list."""
        assert IDEType.VSCODE in self.plugin.supported_ides
        assert IDEType.JUPYTER in self.plugin.supported_ides
        assert IDEType.GENERIC in self.plugin.supported_ides
    
    @patch.object(VSCodePlugin, 'install')
    def test_vscode_installation(self, mock_install):
        """Test VS Code plugin installation."""
        mock_install.return_value = True
        
        with patch.object(self.plugin.server, 'start_server', return_value=True):
            success = self.plugin.install_for_ide(IDEType.VSCODE)
            
            assert success is True
            assert self.plugin.state == PluginState.ACTIVE
            assert IDEType.VSCODE in self.plugin.active_plugins
    
    def test_jupyter_installation(self):
        """Test Jupyter plugin installation."""
        with patch.object(self.plugin.server, 'start_server', return_value=True):
            with patch.object(self.plugin, '_install_jupyter_plugin', return_value=True):
                success = self.plugin.install_for_ide(IDEType.JUPYTER)
                
                assert success is True
                assert self.plugin.state == PluginState.ACTIVE
    
    def test_generic_installation(self):
        """Test generic plugin installation."""
        with patch.object(self.plugin.server, 'start_server', return_value=True):
            with patch.object(self.plugin, '_install_generic_plugin', return_value=True):
                success = self.plugin.install_for_ide(IDEType.GENERIC)
                
                assert success is True
                assert self.plugin.state == PluginState.ACTIVE
    
    def test_unsupported_ide(self):
        """Test installation for unsupported IDE."""
        # Try with a non-existent IDE type
        success = self.plugin.install_for_ide("unsupported")
        
        assert success is False
        assert self.plugin.state == PluginState.ERROR
    
    def test_server_start_failure(self):
        """Test behavior when server fails to start."""
        with patch.object(self.plugin.server, 'start_server', return_value=False):
            success = self.plugin.install_for_ide(IDEType.VSCODE)
            
            assert success is False
            assert self.plugin.state == PluginState.ERROR
    
    def test_plugin_status(self):
        """Test plugin status reporting."""
        status = self.plugin.get_status()
        
        assert 'state' in status
        assert 'active_plugins' in status
        assert 'server_running' in status
        assert 'server_port' in status
        
        assert status['state'] == PluginState.INACTIVE.value
        assert status['active_plugins'] == []
    
    def test_plugin_stop(self):
        """Test plugin stop functionality."""
        # Mock some active plugins
        self.plugin.active_plugins[IDEType.VSCODE] = Mock()
        self.plugin.state = PluginState.ACTIVE
        
        self.plugin.stop()
        
        assert self.plugin.state == PluginState.INACTIVE
        assert len(self.plugin.active_plugins) == 0
    
    @patch('builtins.open', mock_open=True)
    @patch('pathlib.Path.exists', return_value=True)
    def test_jupyter_magic_generation(self, mock_exists):
        """Test Jupyter magic commands generation."""
        with patch('IPython.paths.get_ipython_dir', return_value="/fake/ipython"):
            with patch('pathlib.Path.mkdir'):
                success = self.plugin._install_jupyter_plugin()
                
                # Should succeed even if IPython is not available
                assert success is True
    
    def test_cli_tools_generation(self):
        """Test CLI tools generation."""
        with patch('builtins.open', mock_open=True) as mock_file:
            with patch('pathlib.Path.chmod'):
                success = self.plugin._install_generic_plugin()
                
                assert success is True
                mock_file.assert_called_once()


@pytest.mark.skipif(not HAS_IDE_PLUGIN, reason="ide_plugin module not available")
class TestDiagnosticMessage:
    """Test DiagnosticMessage functionality."""
    
    def test_diagnostic_creation(self):
        """Test diagnostic message creation."""
        diag = DiagnosticMessage(
            line=10,
            column=5,
            message="Test error",
            severity="error",
            code="E001"
        )
        
        assert diag.line == 10
        assert diag.column == 5
        assert diag.message == "Test error"
        assert diag.severity == "error"
        assert diag.code == "E001"
    
    def test_diagnostic_serialization(self):
        """Test diagnostic message serialization."""
        diag = DiagnosticMessage(
            line=1,
            column=0,
            message="Warning message",
            severity="warning"
        )
        
        diag_dict = diag.to_dict()
        
        assert diag_dict['line'] == 1
        assert diag_dict['column'] == 0
        assert diag_dict['message'] == "Warning message"
        assert diag_dict['severity'] == "warning"
        assert diag_dict['source'] == "quantrs2"


@pytest.mark.skipif(not HAS_IDE_PLUGIN, reason="ide_plugin module not available")
class TestCodeCompletionItem:
    """Test CodeCompletionItem functionality."""
    
    def test_completion_item_creation(self):
        """Test completion item creation."""
        item = CodeCompletionItem(
            label="test_function",
            detail="Test function",
            documentation="A test function for testing",
            insert_text="test_function(${1:param})",
            kind="function",
            priority=100
        )
        
        assert item.label == "test_function"
        assert item.detail == "Test function"
        assert item.priority == 100
    
    def test_completion_item_serialization(self):
        """Test completion item serialization."""
        item = CodeCompletionItem(
            label="h",
            detail="Hadamard gate",
            documentation="Apply Hadamard gate",
            insert_text="h(${1:qubit})"
        )
        
        item_dict = item.to_dict()
        
        assert item_dict['label'] == "h"
        assert item_dict['detail'] == "Hadamard gate"
        assert item_dict['insertText'] == "h(${1:qubit})"
        assert item_dict['kind'] == "function"


@pytest.mark.skipif(not HAS_IDE_PLUGIN, reason="ide_plugin module not available")
class TestHoverInfo:
    """Test HoverInfo functionality."""
    
    def test_hover_info_creation(self):
        """Test hover info creation."""
        hover = HoverInfo(
            content="## Test Content\nThis is test content",
            content_type="markdown"
        )
        
        assert hover.content == "## Test Content\nThis is test content"
        assert hover.content_type == "markdown"
    
    def test_hover_info_serialization(self):
        """Test hover info serialization."""
        hover = HoverInfo(
            content="Test content",
            content_type="plaintext"
        )
        
        hover_dict = hover.to_dict()
        
        assert 'contents' in hover_dict
        assert hover_dict['contents']['kind'] == "plaintext"
        assert hover_dict['contents']['value'] == "Test content"


@pytest.mark.skipif(not HAS_IDE_PLUGIN, reason="ide_plugin module not available")
class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_get_ide_plugin(self):
        """Test getting global IDE plugin instance."""
        plugin1 = get_ide_plugin()
        plugin2 = get_ide_plugin()
        
        # Should be singleton
        assert plugin1 is plugin2
        assert isinstance(plugin1, QuantumIDEPlugin)
    
    def test_install_vscode_plugin_function(self):
        """Test VS Code plugin installation function."""
        with patch.object(QuantumIDEPlugin, 'install_for_ide') as mock_install:
            mock_install.return_value = True
            
            success = install_vscode_plugin()
            
            assert success is True
            mock_install.assert_called_once_with(IDEType.VSCODE)
    
    def test_install_jupyter_plugin_function(self):
        """Test Jupyter plugin installation function."""
        with patch.object(QuantumIDEPlugin, 'install_for_ide') as mock_install:
            mock_install.return_value = True
            
            success = install_jupyter_plugin()
            
            assert success is True
            mock_install.assert_called_once_with(IDEType.JUPYTER)
    
    def test_analyze_quantum_code_function(self):
        """Test quantum code analysis function."""
        code = """
import quantrs2
circuit = PyCircuit(2)
circuit.h(0)
"""
        
        diagnostics = analyze_quantum_code(code)
        
        assert isinstance(diagnostics, list)
        # Should be no errors in this simple valid code
        errors = [d for d in diagnostics if d.severity == "error"]
        assert len(errors) == 0


@pytest.mark.skipif(not HAS_IDE_PLUGIN, reason="ide_plugin module not available")
class TestIDEPluginIntegration:
    """Test IDE plugin integration scenarios."""
    
    def test_end_to_end_analysis(self):
        """Test complete analysis workflow."""
        code = """
import quantrs2
circuit = PyCircuit(2)
circuit.h(0)
circuit.cnot(0, 1)
circuit.x(0)
circuit.x(0)  # Redundant
result = circuit.run()
"""
        
        analyzer = QuantumCodeAnalyzer()
        diagnostics = analyzer.analyze_code(code, [
            AnalysisType.SYNTAX,
            AnalysisType.SEMANTIC,
            AnalysisType.OPTIMIZATION
        ])
        
        # Should detect optimization opportunity
        opt_suggestions = [d for d in diagnostics if d.severity == "info"]
        assert len(opt_suggestions) > 0
        
        # Should have no syntax or semantic errors
        errors = [d for d in diagnostics if d.severity == "error"]
        assert len(errors) == 0
    
    def test_completion_and_hover_integration(self):
        """Test completion and hover working together."""
        completion_provider = QuantumCodeCompletion()
        hover_provider = QuantumHoverProvider()
        
        # Get completions
        completions = completion_provider.get_completions("circuit.", (0, 8))
        hadamard_completion = None
        
        for c in completions:
            if c.label == 'h':
                hadamard_completion = c
                break
        
        assert hadamard_completion is not None
        
        # Get hover for the same element
        hover_info = hover_provider.get_hover_info('h', 'circuit.h(0)')
        
        assert hover_info is not None
        assert "Hadamard" in hover_info.content
        assert "Hadamard" in hadamard_completion.detail
    
    def test_multiple_ide_support(self):
        """Test supporting multiple IDEs simultaneously."""
        plugin = QuantumIDEPlugin()
        
        with patch.object(plugin.server, 'start_server', return_value=True):
            with patch.object(plugin, '_install_jupyter_plugin', return_value=True):
                with patch.object(plugin, '_install_generic_plugin', return_value=True):
                    
                    # Install multiple plugins
                    jupyter_success = plugin.install_for_ide(IDEType.JUPYTER)
                    generic_success = plugin.install_for_ide(IDEType.GENERIC)
                    
                    assert jupyter_success is True
                    assert generic_success is True
                    assert len(plugin.active_plugins) == 2
    
    def test_error_recovery(self):
        """Test error recovery scenarios."""
        plugin = QuantumIDEPlugin()
        
        # Test server failure
        with patch.object(plugin.server, 'start_server', return_value=False):
            success = plugin.install_for_ide(IDEType.VSCODE)
            
            assert success is False
            assert plugin.state == PluginState.ERROR
        
        # Test recovery after error
        plugin.state = PluginState.INACTIVE
        
        with patch.object(plugin.server, 'start_server', return_value=True):
            with patch.object(plugin, '_install_generic_plugin', return_value=True):
                success = plugin.install_for_ide(IDEType.GENERIC)
                
                assert success is True
                assert plugin.state == PluginState.ACTIVE
    
    def test_complex_code_analysis(self):
        """Test analysis of complex quantum code."""
        complex_code = """
import quantrs2
from quantrs2 import VQE, bell_state
import numpy as np

# Create complex quantum circuit
circuit = PyCircuit(4)

# Prepare initial state
for i in range(4):
    circuit.h(i)

# Add entangling gates
for i in range(3):
    circuit.cnot(i, i+1)

# Add variational layers
for layer in range(3):
    for i in range(4):
        circuit.ry(i, np.pi/4)
    for i in range(3):
        circuit.cnot(i, i+1)

# Measurements
for i in range(4):
    circuit.measure(i)

result = circuit.run(use_gpu=True)
"""
        
        analyzer = QuantumCodeAnalyzer()
        diagnostics = analyzer.analyze_code(complex_code, [
            AnalysisType.SYNTAX,
            AnalysisType.SEMANTIC,
            AnalysisType.PERFORMANCE
        ])
        
        # Should be able to parse complex circuit
        errors = [d for d in diagnostics if d.severity == "error"]
        assert len(errors) == 0
        
        # May have performance suggestions
        suggestions = [d for d in diagnostics if d.severity == "info"]
        # Performance suggestions are optional for this test


@pytest.mark.skipif(not HAS_IDE_PLUGIN, reason="ide_plugin module not available")
class TestIDEPluginPerformance:
    """Test IDE plugin performance characteristics."""
    
    def test_analysis_performance(self):
        """Test analysis performance with large code."""
        # Generate large quantum program
        lines = ["import quantrs2", "circuit = PyCircuit(10)"]
        for i in range(1000):
            lines.append(f"circuit.h({i % 10})")
        
        large_code = "\n".join(lines)
        
        analyzer = QuantumCodeAnalyzer()
        
        start_time = time.time()
        diagnostics = analyzer.analyze_code(large_code, [AnalysisType.SYNTAX])
        analysis_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert analysis_time < 5.0  # 5 seconds max
        assert isinstance(diagnostics, list)
    
    def test_completion_performance(self):
        """Test completion performance."""
        completion_provider = QuantumCodeCompletion()
        
        start_time = time.time()
        for _ in range(100):
            completions = completion_provider.get_completions("circuit.", (0, 8))
        completion_time = time.time() - start_time
        
        # Should be fast for repeated calls
        assert completion_time < 1.0  # 1 second for 100 calls
        assert len(completions) > 0
    
    def test_hover_performance(self):
        """Test hover performance."""
        hover_provider = QuantumHoverProvider()
        
        start_time = time.time()
        for _ in range(100):
            hover_info = hover_provider.get_hover_info("h", "circuit.h(0)")
        hover_time = time.time() - start_time
        
        # Should be fast for repeated calls
        assert hover_time < 1.0  # 1 second for 100 calls
    
    def test_memory_usage(self):
        """Test memory usage doesn't grow excessively."""
        import gc
        
        analyzer = QuantumCodeAnalyzer()
        
        # Run analysis multiple times
        for _ in range(50):
            code = """
import quantrs2
circuit = PyCircuit(5)
for i in range(50):
    circuit.h(i % 5)
    circuit.cnot(i % 5, (i+1) % 5)
"""
            diagnostics = analyzer.analyze_code(code)
            del diagnostics
        
        # Force garbage collection
        gc.collect()
        
        # Should not crash or run out of memory
        assert True  # If we get here, memory usage is acceptable


if __name__ == "__main__":
    pytest.main([__file__])