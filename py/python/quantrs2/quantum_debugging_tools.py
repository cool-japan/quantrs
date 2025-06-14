"""
QuantRS2 Quantum Debugging Tools - Refactored Main Module

This module provides a backward-compatible interface to the refactored
quantum debugging tools package. The original large file has been split
into smaller, more manageable modules.
"""

import logging

# Import all components from the new debugging package
from .debugging import (
    # Core types
    DebugLevel, DebuggerState, ErrorType, InspectionMode, ValidationRule,
    DebugBreakpoint, DebugFrame, ErrorDiagnosis, ValidationResult,
    DebugSession, StateInspectionResult, MemoryDebugInfo,
    
    # Main classes
    QuantumStateInspector, QuantumErrorAnalyzer, QuantumCircuitValidator,
    QuantumMemoryDebugger, InteractiveQuantumDebugConsole, 
    QuantumDebuggingWebInterface, QuantumDebuggingToolsManager,
    
    # Convenience function
    get_quantum_debugging_tools
)

logger = logging.getLogger(__name__)

# Backward compatibility - expose all the classes and functions at module level
__all__ = [
    # Core types
    "DebugLevel", "DebuggerState", "ErrorType", "InspectionMode", "ValidationRule",
    "DebugBreakpoint", "DebugFrame", "ErrorDiagnosis", "ValidationResult",
    "DebugSession", "StateInspectionResult", "MemoryDebugInfo",
    
    # Main classes
    "QuantumStateInspector", "QuantumErrorAnalyzer", "QuantumCircuitValidator",
    "QuantumMemoryDebugger", "InteractiveQuantumDebugConsole", 
    "QuantumDebuggingWebInterface", "QuantumDebuggingToolsManager",
    
    # Convenience function
    "get_quantum_debugging_tools",
]

# Module information
__version__ = "1.0.0"
__author__ = "QuantRS2 Team"
__description__ = "Comprehensive quantum debugging tools for QuantRS2"

logger.info("QuantRS2 Quantum Debugging Tools loaded (refactored version)")

# Legacy compatibility functions for any code that might depend on the old structure
def create_quantum_debugger(debug_level: DebugLevel = DebugLevel.INFO) -> QuantumDebuggingToolsManager:
    """Create a quantum debugger instance (legacy compatibility)."""
    return get_quantum_debugging_tools(debug_level)

def start_debug_console():
    """Start interactive debug console (legacy compatibility)."""
    debugger = get_quantum_debugging_tools()
    debugger.start_interactive_console()

def start_debug_web_interface(host: str = '0.0.0.0', port: int = 5000):
    """Start web debug interface (legacy compatibility)."""
    debugger = get_quantum_debugging_tools()
    debugger.start_web_interface(host=host, port=port)