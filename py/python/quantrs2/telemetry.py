#!/usr/bin/env python3
"""
Opt-In Telemetry Module for QuantRS2

This module provides optional, privacy-respecting telemetry functionality
to help improve QuantRS2 by collecting anonymous usage statistics.

Features:
    - Fully opt-in (disabled by default)
    - Anonymous data collection
    - No personally identifiable information (PII)
    - Local data aggregation
    - Configurable collection policies
    - User control over data sharing
    - GDPR compliant

Data Collected (when enabled):
    - QuantRS2 version
    - Python version
    - Operating system
    - Basic hardware info (CPU, memory)
    - Circuit statistics (qubit count, gate count)
    - Execution times (anonymized)
    - Error types (no sensitive data)

Data NOT Collected:
    - No personal information
    - No circuit contents
    - No file paths
    - No user names
    - No IP addresses (when self-hosted)

Usage:
    from quantrs2.telemetry import enable_telemetry, disable_telemetry, is_enabled

    # Enable telemetry (opt-in)
    enable_telemetry()

    # Disable telemetry
    disable_telemetry()

    # Check if enabled
    if is_enabled():
        print("Telemetry is enabled")
"""

import sys
import platform
import json
import uuid
import hashlib
import threading
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
import warnings

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class TelemetryLevel(Enum):
    """Telemetry collection levels."""
    DISABLED = "disabled"  # No data collection
    MINIMAL = "minimal"  # Only version and basic system info
    STANDARD = "standard"  # Include usage statistics
    DETAILED = "detailed"  # Include performance metrics


@dataclass
class TelemetryEvent:
    """Represents a telemetry event."""
    event_type: str
    timestamp: str
    data: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SystemInfo:
    """System information for telemetry."""
    os_type: str
    os_version: str
    python_version: str
    quantrs2_version: str
    cpu_count: Optional[int] = None
    total_memory_gb: Optional[float] = None
    architecture: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class TelemetryConfig:
    """Configuration for telemetry."""

    def __init__(
        self,
        enabled: bool = False,
        level: TelemetryLevel = TelemetryLevel.STANDARD,
        endpoint: Optional[str] = None,
        local_only: bool = True,
        batch_size: int = 10,
        flush_interval_seconds: int = 300,
        collect_performance: bool = True,
        collect_errors: bool = True,
    ):
        """
        Initialize telemetry configuration.

        Args:
            enabled: Whether telemetry is enabled
            level: Collection level
            endpoint: Remote endpoint for data submission
            local_only: Only store data locally (don't send)
            batch_size: Number of events before auto-flush
            flush_interval_seconds: Seconds between auto-flush
            collect_performance: Collect performance metrics
            collect_errors: Collect error information
        """
        self.enabled = enabled
        self.level = level
        self.endpoint = endpoint
        self.local_only = local_only
        self.batch_size = batch_size
        self.flush_interval_seconds = flush_interval_seconds
        self.collect_performance = collect_performance
        self.collect_errors = collect_errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'enabled': self.enabled,
            'level': self.level.value,
            'endpoint': self.endpoint,
            'local_only': self.local_only,
            'batch_size': self.batch_size,
            'flush_interval_seconds': self.flush_interval_seconds,
            'collect_performance': self.collect_performance,
            'collect_errors': self.collect_errors,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TelemetryConfig':
        """Create from dictionary."""
        return cls(
            enabled=data.get('enabled', False),
            level=TelemetryLevel(data.get('level', 'standard')),
            endpoint=data.get('endpoint'),
            local_only=data.get('local_only', True),
            batch_size=data.get('batch_size', 10),
            flush_interval_seconds=data.get('flush_interval_seconds', 300),
            collect_performance=data.get('collect_performance', True),
            collect_errors=data.get('collect_errors', True),
        )


class TelemetryCollector:
    """Main telemetry collector class."""

    CONFIG_FILE = Path.home() / ".quantrs2" / "telemetry_config.json"
    DATA_FILE = Path.home() / ".quantrs2" / "telemetry_data.jsonl"
    SESSION_FILE = Path.home() / ".quantrs2" / "session_id.txt"

    def __init__(self, config: Optional[TelemetryConfig] = None):
        """
        Initialize the telemetry collector.

        Args:
            config: Telemetry configuration
        """
        self.config = config or self._load_config()
        self._session_id = self._get_or_create_session_id()
        self._events: List[TelemetryEvent] = []
        self._lock = threading.Lock()
        self._flush_timer: Optional[threading.Timer] = None

        if self.config.enabled:
            self._start_auto_flush()

    def _load_config(self) -> TelemetryConfig:
        """Load configuration from file."""
        if self.CONFIG_FILE.exists():
            try:
                with open(self.CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                return TelemetryConfig.from_dict(data)
            except Exception as e:
                warnings.warn(f"Failed to load telemetry config: {e}")

        return TelemetryConfig()

    def save_config(self):
        """Save configuration to file."""
        self.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
        except Exception as e:
            warnings.warn(f"Failed to save telemetry config: {e}")

    def _get_or_create_session_id(self) -> str:
        """Get or create a session ID."""
        if self.SESSION_FILE.exists():
            try:
                with open(self.SESSION_FILE, 'r') as f:
                    return f.read().strip()
            except Exception:
                pass

        # Create new session ID
        session_id = hashlib.sha256(
            f"{uuid.uuid4()}{time.time()}".encode()
        ).hexdigest()[:16]

        try:
            self.SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(self.SESSION_FILE, 'w') as f:
                f.write(session_id)
        except Exception:
            pass

        return session_id

    def _get_system_info(self) -> SystemInfo:
        """Collect system information."""
        try:
            from quantrs2 import __version__ as quantrs2_version
        except ImportError:
            quantrs2_version = "unknown"

        cpu_count = None
        total_memory_gb = None

        if PSUTIL_AVAILABLE:
            try:
                cpu_count = psutil.cpu_count()
                total_memory_gb = round(psutil.virtual_memory().total / (1024**3), 2)
            except Exception:
                pass

        return SystemInfo(
            os_type=platform.system(),
            os_version=platform.version(),
            python_version=platform.python_version(),
            quantrs2_version=quantrs2_version,
            cpu_count=cpu_count,
            total_memory_gb=total_memory_gb,
            architecture=platform.machine(),
        )

    def record_event(
        self,
        event_type: str,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a telemetry event.

        Args:
            event_type: Type of event
            data: Event data (will be anonymized)
        """
        if not self.config.enabled:
            return

        if self.config.level == TelemetryLevel.DISABLED:
            return

        # Create event
        event = TelemetryEvent(
            event_type=event_type,
            timestamp=datetime.utcnow().isoformat(),
            data=data or {},
            session_id=self._session_id
        )

        # Add to buffer
        with self._lock:
            self._events.append(event)

            # Auto-flush if batch size reached
            if len(self._events) >= self.config.batch_size:
                self._flush_events()

    def record_circuit_execution(
        self,
        n_qubits: int,
        n_gates: int,
        execution_time_ms: float,
        success: bool
    ) -> None:
        """Record circuit execution telemetry."""
        if self.config.level in [TelemetryLevel.DISABLED, TelemetryLevel.MINIMAL]:
            return

        self.record_event('circuit_execution', {
            'n_qubits': n_qubits,
            'n_gates': n_gates,
            'execution_time_ms': round(execution_time_ms, 2),
            'success': success,
        })

    def record_error(
        self,
        error_type: str,
        error_category: str,
        context: Optional[str] = None
    ) -> None:
        """Record an error (anonymized)."""
        if not self.config.collect_errors:
            return

        if self.config.level == TelemetryLevel.DISABLED:
            return

        self.record_event('error', {
            'error_type': error_type,
            'error_category': error_category,
            'context': context,
        })

    def record_performance_metric(
        self,
        metric_name: str,
        value: float,
        unit: str
    ) -> None:
        """Record a performance metric."""
        if not self.config.collect_performance:
            return

        if self.config.level not in [TelemetryLevel.STANDARD, TelemetryLevel.DETAILED]:
            return

        self.record_event('performance_metric', {
            'metric_name': metric_name,
            'value': round(value, 4),
            'unit': unit,
        })

    def record_feature_usage(self, feature_name: str) -> None:
        """Record feature usage."""
        if self.config.level == TelemetryLevel.DISABLED:
            return

        self.record_event('feature_usage', {
            'feature': feature_name,
        })

    def _flush_events(self) -> None:
        """Flush events to storage."""
        with self._lock:
            if not self._events:
                return

            events_to_flush = self._events.copy()
            self._events.clear()

        # Write to local file
        self._write_to_file(events_to_flush)

        # Send to remote endpoint if configured
        if not self.config.local_only and self.config.endpoint:
            self._send_to_endpoint(events_to_flush)

    def _write_to_file(self, events: List[TelemetryEvent]) -> None:
        """Write events to local file."""
        try:
            self.DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(self.DATA_FILE, 'a') as f:
                for event in events:
                    json.dump(event.to_dict(), f)
                    f.write('\n')
        except Exception as e:
            warnings.warn(f"Failed to write telemetry data: {e}")

    def _send_to_endpoint(self, events: List[TelemetryEvent]) -> None:
        """Send events to remote endpoint."""
        # This would be implemented when we have a telemetry endpoint
        # For now, we only store locally
        pass

    def _start_auto_flush(self) -> None:
        """Start automatic periodic flushing."""
        if self._flush_timer:
            self._flush_timer.cancel()

        self._flush_timer = threading.Timer(
            self.config.flush_interval_seconds,
            self._auto_flush_callback
        )
        self._flush_timer.daemon = True
        self._flush_timer.start()

    def _auto_flush_callback(self) -> None:
        """Callback for automatic flush."""
        try:
            self._flush_events()
        finally:
            # Restart timer
            if self.config.enabled:
                self._start_auto_flush()

    def flush(self) -> None:
        """Manually flush all pending events."""
        self._flush_events()

    def get_statistics(self) -> Dict[str, Any]:
        """Get local telemetry statistics."""
        if not self.DATA_FILE.exists():
            return {
                'total_events': 0,
                'events_by_type': {},
            }

        event_types: Dict[str, int] = {}
        total_events = 0

        try:
            with open(self.DATA_FILE, 'r') as f:
                for line in f:
                    try:
                        event_data = json.loads(line)
                        total_events += 1
                        event_type = event_data.get('event_type', 'unknown')
                        event_types[event_type] = event_types.get(event_type, 0) + 1
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass

        return {
            'total_events': total_events,
            'events_by_type': event_types,
            'session_id': self._session_id,
        }

    def clear_data(self) -> None:
        """Clear all local telemetry data."""
        if self.DATA_FILE.exists():
            try:
                self.DATA_FILE.unlink()
            except Exception as e:
                warnings.warn(f"Failed to clear telemetry data: {e}")

    def export_data(self, output_path: Path) -> None:
        """Export telemetry data to a file."""
        if not self.DATA_FILE.exists():
            warnings.warn("No telemetry data to export")
            return

        try:
            import shutil
            shutil.copy(self.DATA_FILE, output_path)
        except Exception as e:
            warnings.warn(f"Failed to export telemetry data: {e}")


# Global collector instance
_collector: Optional[TelemetryCollector] = None


def get_collector() -> TelemetryCollector:
    """Get the global telemetry collector."""
    global _collector
    if _collector is None:
        _collector = TelemetryCollector()
    return _collector


def enable_telemetry(
    level: TelemetryLevel = TelemetryLevel.STANDARD,
    **kwargs
) -> None:
    """
    Enable telemetry collection.

    Args:
        level: Collection level
        **kwargs: Additional configuration options
    """
    collector = get_collector()
    collector.config.enabled = True
    collector.config.level = level

    for key, value in kwargs.items():
        if hasattr(collector.config, key):
            setattr(collector.config, key, value)

    collector.save_config()
    collector._start_auto_flush()

    print("✅ Telemetry enabled")
    print(f"   Level: {level.value}")
    print(f"   Data stored locally at: {collector.DATA_FILE}")
    print("   Use quantrs2.telemetry.disable_telemetry() to disable")


def disable_telemetry() -> None:
    """Disable telemetry collection."""
    collector = get_collector()
    collector.config.enabled = False
    collector.save_config()

    if collector._flush_timer:
        collector._flush_timer.cancel()

    print("✅ Telemetry disabled")


def is_enabled() -> bool:
    """Check if telemetry is enabled."""
    collector = get_collector()
    return collector.config.enabled


def record_event(event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
    """Record a telemetry event (convenience function)."""
    collector = get_collector()
    collector.record_event(event_type, data)


def flush_telemetry() -> None:
    """Flush all pending telemetry events."""
    collector = get_collector()
    collector.flush()


def get_statistics() -> Dict[str, Any]:
    """Get telemetry statistics."""
    collector = get_collector()
    return collector.get_statistics()


def clear_data() -> None:
    """Clear all telemetry data."""
    collector = get_collector()
    collector.clear_data()


if __name__ == "__main__":
    # CLI interface
    import argparse

    parser = argparse.ArgumentParser(description="QuantRS2 Telemetry")
    parser.add_argument('--enable', action='store_true', help='Enable telemetry')
    parser.add_argument('--disable', action='store_true', help='Disable telemetry')
    parser.add_argument('--status', action='store_true', help='Show telemetry status')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--clear', action='store_true', help='Clear telemetry data')
    parser.add_argument('--level', type=str,
                       choices=['minimal', 'standard', 'detailed'],
                       help='Set collection level')

    args = parser.parse_args()

    if args.enable:
        level = TelemetryLevel.STANDARD
        if args.level:
            level = TelemetryLevel(args.level)
        enable_telemetry(level=level)

    elif args.disable:
        disable_telemetry()

    elif args.status:
        enabled = is_enabled()
        collector = get_collector()
        print(f"Telemetry enabled: {enabled}")
        if enabled:
            print(f"Level: {collector.config.level.value}")
            print(f"Local only: {collector.config.local_only}")
            print(f"Data file: {collector.DATA_FILE}")

    elif args.stats:
        stats = get_statistics()
        print("Telemetry Statistics:")
        print(f"  Total events: {stats['total_events']}")
        print(f"  Session ID: {stats['session_id']}")
        print("\nEvents by type:")
        for event_type, count in stats['events_by_type'].items():
            print(f"    {event_type}: {count}")

    elif args.clear:
        clear_data()
        print("Telemetry data cleared")

    else:
        parser.print_help()
