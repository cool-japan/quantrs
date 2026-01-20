"""Type stubs for telemetry module."""

from typing import Dict, Any, Optional, List
from enum import Enum
from pathlib import Path

class TelemetryLevel(Enum):
    DISABLED: str
    MINIMAL: str
    STANDARD: str
    DETAILED: str

class TelemetryEvent:
    event_type: str
    timestamp: str
    data: Dict[str, Any]
    session_id: Optional[str]

    def __init__(
        self,
        event_type: str,
        timestamp: str,
        data: Dict[str, Any] = ...,
        session_id: Optional[str] = ...
    ) -> None: ...

    def to_dict(self) -> Dict[str, Any]: ...

class SystemInfo:
    os_type: str
    os_version: str
    python_version: str
    quantrs2_version: str
    cpu_count: Optional[int]
    total_memory_gb: Optional[float]
    architecture: Optional[str]

    def __init__(
        self,
        os_type: str,
        os_version: str,
        python_version: str,
        quantrs2_version: str,
        cpu_count: Optional[int] = ...,
        total_memory_gb: Optional[float] = ...,
        architecture: Optional[str] = ...
    ) -> None: ...

    def to_dict(self) -> Dict[str, Any]: ...

class TelemetryConfig:
    enabled: bool
    level: TelemetryLevel
    endpoint: Optional[str]
    local_only: bool
    batch_size: int
    flush_interval_seconds: int
    collect_performance: bool
    collect_errors: bool

    def __init__(
        self,
        enabled: bool = ...,
        level: TelemetryLevel = ...,
        endpoint: Optional[str] = ...,
        local_only: bool = ...,
        batch_size: int = ...,
        flush_interval_seconds: int = ...,
        collect_performance: bool = ...,
        collect_errors: bool = ...
    ) -> None: ...

    def to_dict(self) -> Dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TelemetryConfig: ...

class TelemetryCollector:
    CONFIG_FILE: Path
    DATA_FILE: Path
    SESSION_FILE: Path
    config: TelemetryConfig

    def __init__(self, config: Optional[TelemetryConfig] = ...) -> None: ...
    def save_config(self) -> None: ...
    def record_event(self, event_type: str, data: Optional[Dict[str, Any]] = ...) -> None: ...
    def record_circuit_execution(
        self,
        n_qubits: int,
        n_gates: int,
        execution_time_ms: float,
        success: bool
    ) -> None: ...
    def record_error(
        self,
        error_type: str,
        error_category: str,
        context: Optional[str] = ...
    ) -> None: ...
    def record_performance_metric(
        self,
        metric_name: str,
        value: float,
        unit: str
    ) -> None: ...
    def record_feature_usage(self, feature_name: str) -> None: ...
    def flush(self) -> None: ...
    def get_statistics(self) -> Dict[str, Any]: ...
    def clear_data(self) -> None: ...
    def export_data(self, output_path: Path) -> None: ...

def get_collector() -> TelemetryCollector: ...
def enable_telemetry(level: TelemetryLevel = ..., **kwargs: Any) -> None: ...
def disable_telemetry() -> None: ...
def is_enabled() -> bool: ...
def record_event(event_type: str, data: Optional[Dict[str, Any]] = ...) -> None: ...
def flush_telemetry() -> None: ...
def get_statistics() -> Dict[str, Any]: ...
def clear_data() -> None: ...
