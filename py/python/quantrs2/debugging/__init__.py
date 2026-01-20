"""
QuantRS2 Quantum Debugging Package

This package provides comprehensive quantum debugging capabilities including:
- Quantum state inspection and analysis
- Quantum error diagnosis and recovery
- Circuit validation and property checking
- Memory debugging and performance analysis
- Interactive debugging interfaces
"""

from .core import (
    DebugLevel, DebuggerState, ErrorType, InspectionMode, ValidationRule,
    DebugBreakpoint, DebugFrame, ErrorDiagnosis, ValidationResult,
    DebugSession, StateInspectionResult, MemoryDebugInfo
)
from .state_inspector import QuantumStateInspector
from .error_analyzer import QuantumErrorAnalyzer
from .circuit_validator import QuantumCircuitValidator
from .memory_debugger import QuantumMemoryDebugger
from .interactive_console import InteractiveQuantumDebugConsole
from .web_interface import QuantumDebuggingWebInterface
from .manager import QuantumDebuggingToolsManager

# Convenience function
def get_quantum_debugging_tools(debug_level: DebugLevel = DebugLevel.INFO) -> QuantumDebuggingToolsManager:
    """Get a configured quantum debugging tools manager."""
    return QuantumDebuggingToolsManager(debug_level=debug_level)

__all__ = [
    # Core types
    "DebugLevel", "DebuggerState", "ErrorType", "InspectionMode", "ValidationRule",
    "DebugBreakpoint", "DebugFrame", "ErrorDiagnosis", "ValidationResult",
    "DebugSession", "StateInspectionResult", "MemoryDebugInfo",
    
    # Main classes
    "QuantumStateInspector", "QuantumErrorAnalyzer", "QuantumCircuitValidator",
    "QuantumMemoryDebugger", "InteractiveQuantumDebugConsole", 
    "QuantumDebuggingWebInterface", "QuantumDebuggingToolsManager",
    
    # Utilities
    "get_quantum_debugging_tools",
]