"""
Core debugging infrastructure for QuantRS2.

This module contains the fundamental data structures, enums, and base classes
used throughout the quantum debugging framework.
"""

import time
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)

class DebugLevel(Enum):
    """Debug level enumeration for controlling debug output verbosity."""
    TRACE = auto()
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

class DebuggerState(Enum):
    """Debugger state enumeration."""
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    STEPPING = auto()
    ANALYZING = auto()
    ERROR = auto()

class ErrorType(Enum):
    """Quantum error type enumeration."""
    GATE_ERROR = auto()
    MEASUREMENT_ERROR = auto()
    DECOHERENCE_ERROR = auto()
    CIRCUIT_ERROR = auto()
    STATE_ERROR = auto()
    COMPILATION_ERROR = auto()
    RUNTIME_ERROR = auto()
    VALIDATION_ERROR = auto()
    MEMORY_ERROR = auto()
    PERFORMANCE_ERROR = auto()

class InspectionMode(Enum):
    """State inspection mode enumeration."""
    AMPLITUDE_ANALYSIS = auto()
    PROBABILITY_ANALYSIS = auto()
    PHASE_ANALYSIS = auto()
    ENTANGLEMENT_ANALYSIS = auto()
    COHERENCE_ANALYSIS = auto()
    CORRELATION_ANALYSIS = auto()
    PURITY_ANALYSIS = auto()
    FIDELITY_ANALYSIS = auto()

class ValidationRule(Enum):
    """Circuit validation rule enumeration."""
    UNITARITY_CHECK = auto()
    NORMALIZATION_CHECK = auto()
    HERMITICITY_CHECK = auto()
    COMMUTATIVITY_CHECK = auto()
    CAUSALITY_CHECK = auto()
    RESOURCE_CHECK = auto()
    CONNECTIVITY_CHECK = auto()
    TIMING_CHECK = auto()

@dataclass
class DebugBreakpoint:
    """Represents a debugging breakpoint."""
    id: str
    location: str
    condition: Optional[str] = None
    hit_count: int = 0
    enabled: bool = True
    temporary: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DebugFrame:
    """Represents a debugging frame with execution context."""
    frame_id: str
    gate_index: int
    gate_type: str
    qubits: List[int]
    parameters: List[float]
    state_before: Optional[np.ndarray] = None
    state_after: Optional[np.ndarray] = None
    execution_time: float = 0.0
    memory_usage: int = 0
    error_probability: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ErrorDiagnosis:
    """Represents a quantum error diagnosis."""
    error_type: ErrorType
    severity: str
    message: str
    location: str
    suggestions: List[str]
    auto_fix_available: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class ValidationResult:
    """Represents a circuit validation result."""
    rule: ValidationRule
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

@dataclass
class DebugSession:
    """Represents a debugging session."""
    session_id: str
    circuit: Any
    start_time: float
    end_time: Optional[float] = None
    state: DebuggerState = DebuggerState.IDLE
    breakpoints: List[DebugBreakpoint] = field(default_factory=list)
    frames: List[DebugFrame] = field(default_factory=list)
    errors: List[ErrorDiagnosis] = field(default_factory=list)
    validations: List[ValidationResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StateInspectionResult:
    """Represents quantum state inspection results."""
    mode: InspectionMode
    analysis_data: Dict[str, Any]
    insights: List[str]
    anomalies: List[str]
    recommendations: List[str]
    timestamp: float = field(default_factory=time.time)

@dataclass
class MemoryDebugInfo:
    """Represents memory debugging information."""
    total_memory: int
    used_memory: int
    free_memory: int
    peak_memory: int
    allocation_count: int
    deallocation_count: int
    memory_leaks: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)