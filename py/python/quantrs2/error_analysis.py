"""
Error Analysis and Incident Management for QuantRS2

This module provides advanced error analysis, pattern detection, automated
incident management, and root cause analysis capabilities for production
QuantRS2 deployments.
"""

import time
import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import re
import hashlib
import statistics

from .structured_logging import ErrorInfo, ErrorCategory, get_global_logging_system

logger = get_global_logging_system().get_logger(__name__)


class IncidentSeverity(Enum):
    """Incident severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(Enum):
    """Incident status states."""
    OPEN = "open"
    INVESTIGATING = "investigating"
    IDENTIFIED = "identified"
    MONITORING = "monitoring"
    RESOLVED = "resolved"
    CLOSED = "closed"


class ErrorPattern(Enum):
    """Common error patterns."""
    FREQUENT_ERRORS = "frequent_errors"
    ERROR_BURST = "error_burst"
    CASCADING_FAILURES = "cascading_failures"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    TIMEOUT_CLUSTER = "timeout_cluster"
    PERMISSION_ISSUES = "permission_issues"
    CONFIGURATION_ERRORS = "configuration_errors"
    EXTERNAL_SERVICE_FAILURE = "external_service_failure"


@dataclass
class ErrorPatternMatch:
    """Represents a detected error pattern."""
    pattern_type: ErrorPattern
    confidence: float
    errors: List[str]  # Error IDs
    time_window: Tuple[float, float]  # Start and end timestamps
    description: str
    severity: IncidentSeverity
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Incident:
    """Represents a system incident."""
    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus
    
    # Timing
    created_at: float
    updated_at: float
    resolved_at: Optional[float] = None
    
    # Related data
    error_patterns: List[ErrorPatternMatch] = field(default_factory=list)
    related_errors: List[str] = field(default_factory=list)
    affected_components: Set[str] = field(default_factory=set)
    
    # Investigation
    root_cause: Optional[str] = None
    resolution: Optional[str] = None
    action_items: List[str] = field(default_factory=list)
    
    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    assignee: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert incident to dictionary."""
        return {
            'incident_id': self.incident_id,
            'title': self.title,
            'description': self.description,
            'severity': self.severity.value,
            'status': self.status.value,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'resolved_at': self.resolved_at,
            'error_patterns': [
                {
                    'pattern_type': p.pattern_type.value,
                    'confidence': p.confidence,
                    'errors': p.errors,
                    'time_window': p.time_window,
                    'description': p.description,
                    'severity': p.severity.value,
                    'metadata': p.metadata
                } for p in self.error_patterns
            ],
            'related_errors': self.related_errors,
            'affected_components': list(self.affected_components),
            'root_cause': self.root_cause,
            'resolution': self.resolution,
            'action_items': self.action_items,
            'tags': self.tags,
            'assignee': self.assignee
        }


class ErrorPatternDetector:
    """Detects patterns in error occurrences."""
    
    def __init__(self, detection_window: int = 3600):  # 1 hour window
        self.detection_window = detection_window
        self._error_history: List[Tuple[float, ErrorInfo]] = []
        self._lock = threading.RLock()
    
    def add_error(self, error_info: ErrorInfo):
        """Add error to pattern detection."""
        with self._lock:
            timestamp = error_info.occurred_at
            self._error_history.append((timestamp, error_info))
            
            # Clean up old errors outside detection window
            cutoff_time = time.time() - self.detection_window
            self._error_history = [
                (t, e) for t, e in self._error_history if t >= cutoff_time
            ]
    
    def detect_patterns(self) -> List[ErrorPatternMatch]:
        """Detect error patterns in recent history."""
        patterns = []
        current_time = time.time()
        
        with self._lock:
            if not self._error_history:
                return patterns
            
            # Detect various patterns
            patterns.extend(self._detect_frequent_errors())
            patterns.extend(self._detect_error_bursts())
            patterns.extend(self._detect_cascading_failures())
            patterns.extend(self._detect_resource_exhaustion())
            patterns.extend(self._detect_timeout_clusters())
            patterns.extend(self._detect_permission_issues())
            patterns.extend(self._detect_configuration_errors())
        
        return patterns
    
    def _detect_frequent_errors(self) -> List[ErrorPatternMatch]:
        """Detect frequently occurring errors."""
        patterns = []
        
        # Group errors by type and message similarity
        error_groups = defaultdict(list)
        
        for timestamp, error_info in self._error_history:
            # Create signature for error grouping
            signature = self._create_error_signature(error_info)
            error_groups[signature].append((timestamp, error_info))
        
        # Find groups with high frequency
        for signature, group in error_groups.items():
            if len(group) >= 5:  # Threshold for "frequent"
                error_ids = [e.error_id for _, e in group]
                time_window = (group[0][0], group[-1][0])
                
                # Calculate frequency
                duration = max(1, time_window[1] - time_window[0])
                frequency = len(group) / duration * 3600  # Errors per hour
                
                if frequency > 10:  # High frequency threshold
                    patterns.append(ErrorPatternMatch(
                        pattern_type=ErrorPattern.FREQUENT_ERRORS,
                        confidence=min(1.0, frequency / 100),
                        errors=error_ids,
                        time_window=time_window,
                        description=f"Frequent error pattern: {len(group)} occurrences in {duration:.1f}s",
                        severity=IncidentSeverity.MEDIUM if frequency < 50 else IncidentSeverity.HIGH,
                        metadata={
                            'frequency_per_hour': frequency,
                            'error_signature': signature,
                            'total_occurrences': len(group)
                        }
                    ))
        
        return patterns
    
    def _detect_error_bursts(self) -> List[ErrorPatternMatch]:
        """Detect sudden bursts of errors."""
        patterns = []
        
        # Analyze error rate over time windows
        window_size = 300  # 5 minutes
        current_time = time.time()
        
        # Create time windows
        windows = []
        for i in range(int(self.detection_window / window_size)):
            window_end = current_time - (i * window_size)
            window_start = window_end - window_size
            
            window_errors = [
                e for t, e in self._error_history
                if window_start <= t < window_end
            ]
            
            if window_errors:
                windows.append((window_start, window_end, window_errors))
        
        if len(windows) < 3:
            return patterns
        
        # Calculate error rates
        error_rates = [len(errors) for _, _, errors in windows]
        
        if len(error_rates) >= 3:
            avg_rate = statistics.mean(error_rates[1:])  # Exclude current window
            current_rate = error_rates[0]
            
            # Detect burst (current rate significantly higher than average)
            if avg_rate > 0 and current_rate > avg_rate * 3:  # 3x threshold
                window_start, window_end, burst_errors = windows[0]
                
                patterns.append(ErrorPatternMatch(
                    pattern_type=ErrorPattern.ERROR_BURST,
                    confidence=min(1.0, current_rate / (avg_rate * 5)),
                    errors=[e.error_id for e in burst_errors],
                    time_window=(window_start, window_end),
                    description=f"Error burst detected: {current_rate} errors vs {avg_rate:.1f} average",
                    severity=IncidentSeverity.HIGH,
                    metadata={
                        'current_rate': current_rate,
                        'average_rate': avg_rate,
                        'burst_ratio': current_rate / avg_rate
                    }
                ))
        
        return patterns
    
    def _detect_cascading_failures(self) -> List[ErrorPatternMatch]:
        """Detect cascading failure patterns."""
        patterns = []
        
        # Look for chains of related errors
        # Group errors by time proximity and check for dependency relationships
        time_threshold = 60  # 1 minute
        
        cascades = []
        for i, (timestamp, error_info) in enumerate(self._error_history):
            cascade = [(timestamp, error_info)]
            
            # Look for subsequent related errors
            for j in range(i + 1, len(self._error_history)):
                next_timestamp, next_error = self._error_history[j]
                
                if next_timestamp - timestamp > time_threshold:
                    break
                
                # Check if errors are related (simplified heuristic)
                if self._are_errors_related(error_info, next_error):
                    cascade.append((next_timestamp, next_error))
            
            if len(cascade) >= 3:  # At least 3 related errors
                cascades.append(cascade)
        
        # Convert cascades to patterns
        for cascade in cascades:
            if len(cascade) >= 3:
                error_ids = [e.error_id for _, e in cascade]
                time_window = (cascade[0][0], cascade[-1][0])
                
                patterns.append(ErrorPatternMatch(
                    pattern_type=ErrorPattern.CASCADING_FAILURES,
                    confidence=min(1.0, len(cascade) / 10),
                    errors=error_ids,
                    time_window=time_window,
                    description=f"Cascading failure detected: {len(cascade)} related errors",
                    severity=IncidentSeverity.HIGH,
                    metadata={
                        'cascade_length': len(cascade),
                        'duration_seconds': time_window[1] - time_window[0]
                    }
                ))
        
        return patterns
    
    def _detect_resource_exhaustion(self) -> List[ErrorPatternMatch]:
        """Detect resource exhaustion patterns."""
        patterns = []
        
        resource_keywords = [
            'memory', 'disk', 'cpu', 'connection', 'pool', 'limit',
            'quota', 'capacity', 'exhausted', 'full', 'timeout'
        ]
        
        resource_errors = []
        for timestamp, error_info in self._error_history:
            message_lower = error_info.message.lower()
            if any(keyword in message_lower for keyword in resource_keywords):
                resource_errors.append((timestamp, error_info))
        
        if len(resource_errors) >= 3:
            error_ids = [e.error_id for _, e in resource_errors]
            time_window = (resource_errors[0][0], resource_errors[-1][0])
            
            patterns.append(ErrorPatternMatch(
                pattern_type=ErrorPattern.RESOURCE_EXHAUSTION,
                confidence=min(1.0, len(resource_errors) / 10),
                errors=error_ids,
                time_window=time_window,
                description=f"Resource exhaustion pattern: {len(resource_errors)} resource-related errors",
                severity=IncidentSeverity.HIGH,
                metadata={'resource_error_count': len(resource_errors)}
            ))
        
        return patterns
    
    def _detect_timeout_clusters(self) -> List[ErrorPatternMatch]:
        """Detect clusters of timeout errors."""
        patterns = []
        
        timeout_errors = []
        for timestamp, error_info in self._error_history:
            if 'timeout' in error_info.message.lower() or 'timeout' in error_info.error_type.lower():
                timeout_errors.append((timestamp, error_info))
        
        if len(timeout_errors) >= 3:
            error_ids = [e.error_id for _, e in timeout_errors]
            time_window = (timeout_errors[0][0], timeout_errors[-1][0])
            
            patterns.append(ErrorPatternMatch(
                pattern_type=ErrorPattern.TIMEOUT_CLUSTER,
                confidence=min(1.0, len(timeout_errors) / 5),
                errors=error_ids,
                time_window=time_window,
                description=f"Timeout cluster detected: {len(timeout_errors)} timeout errors",
                severity=IncidentSeverity.MEDIUM,
                metadata={'timeout_error_count': len(timeout_errors)}
            ))
        
        return patterns
    
    def _detect_permission_issues(self) -> List[ErrorPatternMatch]:
        """Detect permission-related error patterns."""
        patterns = []
        
        permission_keywords = [
            'permission', 'denied', 'unauthorized', 'forbidden',
            'access', 'credential', 'authentication', 'authorization'
        ]
        
        permission_errors = []
        for timestamp, error_info in self._error_history:
            message_lower = error_info.message.lower()
            if any(keyword in message_lower for keyword in permission_keywords):
                permission_errors.append((timestamp, error_info))
        
        if len(permission_errors) >= 2:
            error_ids = [e.error_id for _, e in permission_errors]
            time_window = (permission_errors[0][0], permission_errors[-1][0])
            
            patterns.append(ErrorPatternMatch(
                pattern_type=ErrorPattern.PERMISSION_ISSUES,
                confidence=min(1.0, len(permission_errors) / 5),
                errors=error_ids,
                time_window=time_window,
                description=f"Permission issues detected: {len(permission_errors)} access-related errors",
                severity=IncidentSeverity.MEDIUM,
                metadata={'permission_error_count': len(permission_errors)}
            ))
        
        return patterns
    
    def _detect_configuration_errors(self) -> List[ErrorPatternMatch]:
        """Detect configuration-related error patterns."""
        patterns = []
        
        config_keywords = [
            'config', 'configuration', 'setting', 'parameter',
            'invalid', 'missing', 'not found', 'undefined'
        ]
        
        config_errors = []
        for timestamp, error_info in self._error_history:
            message_lower = error_info.message.lower()
            if any(keyword in message_lower for keyword in config_keywords):
                config_errors.append((timestamp, error_info))
        
        if len(config_errors) >= 2:
            error_ids = [e.error_id for _, e in config_errors]
            time_window = (config_errors[0][0], config_errors[-1][0])
            
            patterns.append(ErrorPatternMatch(
                pattern_type=ErrorPattern.CONFIGURATION_ERRORS,
                confidence=min(1.0, len(config_errors) / 3),
                errors=error_ids,
                time_window=time_window,
                description=f"Configuration issues detected: {len(config_errors)} config-related errors",
                severity=IncidentSeverity.MEDIUM,
                metadata={'config_error_count': len(config_errors)}
            ))
        
        return patterns
    
    def _create_error_signature(self, error_info: ErrorInfo) -> str:
        """Create a signature for error grouping."""
        # Normalize error message for grouping
        message = error_info.message.lower()
        
        # Remove variable parts (numbers, timestamps, IDs)
        message = re.sub(r'\d+', 'NUM', message)
        message = re.sub(r'[a-f0-9-]{8,}', 'ID', message)
        message = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', 'DATE', message)
        
        # Create signature from error type and normalized message
        signature_data = f"{error_info.error_type}:{error_info.error_category.value}:{message}"
        return hashlib.md5(signature_data.encode()).hexdigest()[:16]
    
    def _are_errors_related(self, error1: ErrorInfo, error2: ErrorInfo) -> bool:
        """Check if two errors are related (simple heuristic)."""
        # Same category suggests relation
        if error1.error_category == error2.error_category:
            return True
        
        # Similar error types
        if error1.error_type == error2.error_type:
            return True
        
        # Check for common keywords in messages
        words1 = set(error1.message.lower().split())
        words2 = set(error2.message.lower().split())
        common_words = words1.intersection(words2)
        
        # If they share significant words, they might be related
        return len(common_words) >= 2
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pattern detection statistics."""
        with self._lock:
            total_errors = len(self._error_history)
            
            # Group by category
            category_counts = defaultdict(int)
            for _, error_info in self._error_history:
                category_counts[error_info.error_category.value] += 1
            
            # Group by error type
            type_counts = defaultdict(int)
            for _, error_info in self._error_history:
                type_counts[error_info.error_type] += 1
            
            return {
                'total_errors_tracked': total_errors,
                'detection_window_hours': self.detection_window / 3600,
                'errors_by_category': dict(category_counts),
                'errors_by_type': dict(type_counts),
                'most_common_types': dict(Counter(type_counts).most_common(5))
            }


class IncidentManager:
    """Manages incidents and their lifecycle."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or str(Path.home() / '.quantrs2' / 'incidents.db')
        self._incidents: Dict[str, Incident] = {}
        self._lock = threading.RLock()
        
        # Initialize database
        self._init_database()
        
        # Load existing incidents
        self._load_incidents()
    
    def _init_database(self):
        """Initialize incidents database."""
        try:
            Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.storage_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS incidents (
                        incident_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        description TEXT,
                        severity TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        resolved_at REAL,
                        root_cause TEXT,
                        resolution TEXT,
                        assignee TEXT,
                        data TEXT  -- JSON data
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_incidents_status 
                    ON incidents(status)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_incidents_severity 
                    ON incidents(severity)
                """)
                
        except Exception as e:
            logger.error(f"Failed to initialize incidents database: {e}")
    
    def _load_incidents(self):
        """Load incidents from database."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.execute("""
                    SELECT incident_id, title, description, severity, status,
                           created_at, updated_at, resolved_at, root_cause,
                           resolution, assignee, data
                    FROM incidents
                    WHERE status NOT IN ('resolved', 'closed')
                """)
                
                for row in cursor:
                    incident_id, title, description, severity, status, \
                    created_at, updated_at, resolved_at, root_cause, \
                    resolution, assignee, data_json = row
                    
                    # Parse JSON data
                    data = json.loads(data_json) if data_json else {}
                    
                    # Reconstruct incident
                    incident = Incident(
                        incident_id=incident_id,
                        title=title,
                        description=description,
                        severity=IncidentSeverity(severity),
                        status=IncidentStatus(status),
                        created_at=created_at,
                        updated_at=updated_at,
                        resolved_at=resolved_at,
                        root_cause=root_cause,
                        resolution=resolution,
                        assignee=assignee,
                        related_errors=data.get('related_errors', []),
                        affected_components=set(data.get('affected_components', [])),
                        action_items=data.get('action_items', []),
                        tags=data.get('tags', {})
                    )
                    
                    # Reconstruct error patterns
                    for pattern_data in data.get('error_patterns', []):
                        pattern = ErrorPatternMatch(
                            pattern_type=ErrorPattern(pattern_data['pattern_type']),
                            confidence=pattern_data['confidence'],
                            errors=pattern_data['errors'],
                            time_window=tuple(pattern_data['time_window']),
                            description=pattern_data['description'],
                            severity=IncidentSeverity(pattern_data['severity']),
                            metadata=pattern_data.get('metadata', {})
                        )
                        incident.error_patterns.append(pattern)
                    
                    self._incidents[incident_id] = incident
                
                logger.info(f"Loaded {len(self._incidents)} active incidents")
                
        except Exception as e:
            logger.error(f"Failed to load incidents: {e}")
    
    def create_incident(self, title: str, description: str, 
                       severity: IncidentSeverity,
                       error_patterns: List[ErrorPatternMatch],
                       assignee: Optional[str] = None) -> Incident:
        """Create a new incident."""
        import uuid
        
        incident_id = str(uuid.uuid4())
        current_time = time.time()
        
        # Extract affected components and related errors
        affected_components = set()
        related_errors = []
        
        for pattern in error_patterns:
            related_errors.extend(pattern.errors)
            # Extract component info from error context if available
            # This would require integration with error tracking system
        
        incident = Incident(
            incident_id=incident_id,
            title=title,
            description=description,
            severity=severity,
            status=IncidentStatus.OPEN,
            created_at=current_time,
            updated_at=current_time,
            error_patterns=error_patterns,
            related_errors=related_errors,
            affected_components=affected_components,
            assignee=assignee
        )
        
        with self._lock:
            self._incidents[incident_id] = incident
            self._persist_incident(incident)
        
        logger.info(f"Created incident: {incident_id} - {title}")
        return incident
    
    def update_incident(self, incident_id: str, **updates) -> bool:
        """Update an incident."""
        with self._lock:
            if incident_id not in self._incidents:
                return False
            
            incident = self._incidents[incident_id]
            
            # Update fields
            for field, value in updates.items():
                if hasattr(incident, field):
                    setattr(incident, field, value)
            
            incident.updated_at = time.time()
            
            # Handle status changes
            if 'status' in updates and updates['status'] in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]:
                incident.resolved_at = time.time()
            
            self._persist_incident(incident)
            
        logger.info(f"Updated incident: {incident_id}")
        return True
    
    def resolve_incident(self, incident_id: str, resolution: str, 
                        root_cause: Optional[str] = None) -> bool:
        """Resolve an incident."""
        return self.update_incident(
            incident_id,
            status=IncidentStatus.RESOLVED,
            resolution=resolution,
            root_cause=root_cause
        )
    
    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Get incident by ID."""
        with self._lock:
            return self._incidents.get(incident_id)
    
    def get_active_incidents(self) -> List[Incident]:
        """Get all active incidents."""
        with self._lock:
            return [
                incident for incident in self._incidents.values()
                if incident.status not in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]
            ]
    
    def get_incidents_by_severity(self, severity: IncidentSeverity) -> List[Incident]:
        """Get incidents by severity."""
        with self._lock:
            return [
                incident for incident in self._incidents.values()
                if incident.severity == severity
            ]
    
    def _persist_incident(self, incident: Incident):
        """Persist incident to database."""
        try:
            # Prepare data for storage
            data = {
                'error_patterns': [
                    {
                        'pattern_type': p.pattern_type.value,
                        'confidence': p.confidence,
                        'errors': p.errors,
                        'time_window': p.time_window,
                        'description': p.description,
                        'severity': p.severity.value,
                        'metadata': p.metadata
                    } for p in incident.error_patterns
                ],
                'related_errors': incident.related_errors,
                'affected_components': list(incident.affected_components),
                'action_items': incident.action_items,
                'tags': incident.tags
            }
            
            with sqlite3.connect(self.storage_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO incidents (
                        incident_id, title, description, severity, status,
                        created_at, updated_at, resolved_at, root_cause,
                        resolution, assignee, data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    incident.incident_id,
                    incident.title,
                    incident.description,
                    incident.severity.value,
                    incident.status.value,
                    incident.created_at,
                    incident.updated_at,
                    incident.resolved_at,
                    incident.root_cause,
                    incident.resolution,
                    incident.assignee,
                    json.dumps(data)
                ))
                
        except Exception as e:
            logger.error(f"Failed to persist incident {incident.incident_id}: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get incident management statistics."""
        with self._lock:
            total_incidents = len(self._incidents)
            
            # Count by status
            status_counts = defaultdict(int)
            for incident in self._incidents.values():
                status_counts[incident.status.value] += 1
            
            # Count by severity
            severity_counts = defaultdict(int)
            for incident in self._incidents.values():
                severity_counts[incident.severity.value] += 1
            
            # Calculate resolution metrics
            resolved_incidents = [
                i for i in self._incidents.values()
                if i.resolved_at is not None
            ]
            
            avg_resolution_time = 0
            if resolved_incidents:
                resolution_times = [
                    i.resolved_at - i.created_at
                    for i in resolved_incidents
                ]
                avg_resolution_time = statistics.mean(resolution_times)
            
            return {
                'total_incidents': total_incidents,
                'by_status': dict(status_counts),
                'by_severity': dict(severity_counts),
                'average_resolution_time_hours': avg_resolution_time / 3600,
                'resolved_incidents': len(resolved_incidents)
            }


class ErrorAnalysisSystem:
    """Main error analysis and incident management system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.pattern_detector = ErrorPatternDetector(
            detection_window=self.config.get('pattern_detection_window', 3600)
        )
        self.incident_manager = IncidentManager(
            storage_path=self.config.get('incidents_db_path')
        )
        
        # Analysis settings
        self.auto_incident_creation = self.config.get('auto_incident_creation', True)
        self.incident_severity_thresholds = self.config.get('incident_severity_thresholds', {
            'critical': {'confidence': 0.8, 'pattern_count': 3},
            'high': {'confidence': 0.6, 'pattern_count': 2},
            'medium': {'confidence': 0.4, 'pattern_count': 1}
        })
        
        # Background processing
        self._analysis_thread = None
        self._shutdown = threading.Event()
        
        if self.config.get('enable_background_analysis', True):
            self._start_background_analysis()
        
        logger.info("Error analysis system initialized")
    
    def _start_background_analysis(self):
        """Start background error analysis."""
        self._analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self._analysis_thread.start()
    
    def _analysis_loop(self):
        """Background analysis loop."""
        analysis_interval = self.config.get('analysis_interval', 300)  # 5 minutes
        
        while not self._shutdown.wait(analysis_interval):
            try:
                self._perform_pattern_analysis()
            except Exception as e:
                logger.error(f"Background analysis error: {e}")
    
    def _perform_pattern_analysis(self):
        """Perform pattern analysis and create incidents if needed."""
        patterns = self.pattern_detector.detect_patterns()
        
        if not patterns:
            return
        
        logger.info(f"Detected {len(patterns)} error patterns")
        
        # Group patterns by severity and create incidents
        if self.auto_incident_creation:
            self._create_incidents_from_patterns(patterns)
    
    def _create_incidents_from_patterns(self, patterns: List[ErrorPatternMatch]):
        """Create incidents from detected patterns."""
        # Group patterns by severity
        critical_patterns = [p for p in patterns if p.severity == IncidentSeverity.CRITICAL]
        high_patterns = [p for p in patterns if p.severity == IncidentSeverity.HIGH]
        medium_patterns = [p for p in patterns if p.severity == IncidentSeverity.MEDIUM]
        
        # Create incidents based on thresholds
        if critical_patterns:
            self._create_incident_for_patterns(critical_patterns, IncidentSeverity.CRITICAL)
        
        if len(high_patterns) >= self.incident_severity_thresholds['high']['pattern_count']:
            self._create_incident_for_patterns(high_patterns, IncidentSeverity.HIGH)
        
        if len(medium_patterns) >= self.incident_severity_thresholds['medium']['pattern_count']:
            self._create_incident_for_patterns(medium_patterns, IncidentSeverity.MEDIUM)
    
    def _create_incident_for_patterns(self, patterns: List[ErrorPatternMatch], 
                                    severity: IncidentSeverity):
        """Create incident for a group of patterns."""
        # Generate incident title and description
        pattern_types = [p.pattern_type.value for p in patterns]
        title = f"Automated Incident: {', '.join(set(pattern_types))}"
        
        description = f"Automatically created incident based on detected error patterns:\n"
        for pattern in patterns:
            description += f"- {pattern.description} (confidence: {pattern.confidence:.2f})\n"
        
        # Check if similar incident already exists
        active_incidents = self.incident_manager.get_active_incidents()
        for incident in active_incidents:
            if self._are_patterns_similar(incident.error_patterns, patterns):
                logger.info(f"Similar incident already exists: {incident.incident_id}")
                return
        
        # Create new incident
        incident = self.incident_manager.create_incident(
            title=title,
            description=description,
            severity=severity,
            error_patterns=patterns
        )
        
        logger.info(f"Created automated incident: {incident.incident_id}")
    
    def _are_patterns_similar(self, existing_patterns: List[ErrorPatternMatch],
                            new_patterns: List[ErrorPatternMatch]) -> bool:
        """Check if pattern lists are similar."""
        existing_types = set(p.pattern_type for p in existing_patterns)
        new_types = set(p.pattern_type for p in new_patterns)
        
        # If they share more than half the pattern types, consider them similar
        overlap = len(existing_types.intersection(new_types))
        total_unique = len(existing_types.union(new_types))
        
        return overlap / total_unique > 0.5
    
    def analyze_error(self, error_info: ErrorInfo) -> Dict[str, Any]:
        """Analyze a single error and return insights."""
        # Add to pattern detection
        self.pattern_detector.add_error(error_info)
        
        # Immediate analysis
        analysis = {
            'error_id': error_info.error_id,
            'category': error_info.error_category.value,
            'severity_estimate': self._estimate_error_severity(error_info),
            'related_patterns': [],
            'recommendations': []
        }
        
        # Check for immediate patterns
        recent_patterns = self.pattern_detector.detect_patterns()
        
        # Find patterns that include this error
        for pattern in recent_patterns:
            if error_info.error_id in pattern.errors:
                analysis['related_patterns'].append({
                    'pattern_type': pattern.pattern_type.value,
                    'confidence': pattern.confidence,
                    'description': pattern.description
                })
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_error_recommendations(error_info)
        
        return analysis
    
    def _estimate_error_severity(self, error_info: ErrorInfo) -> str:
        """Estimate error severity based on various factors."""
        critical_keywords = ['critical', 'fatal', 'crash', 'corruption', 'security']
        high_keywords = ['error', 'failure', 'exception', 'timeout', 'connection']
        
        message_lower = error_info.message.lower()
        
        if any(keyword in message_lower for keyword in critical_keywords):
            return 'critical'
        elif any(keyword in message_lower for keyword in high_keywords):
            return 'high'
        elif error_info.error_category in [ErrorCategory.SECURITY, ErrorCategory.SYSTEM]:
            return 'high'
        else:
            return 'medium'
    
    def _generate_error_recommendations(self, error_info: ErrorInfo) -> List[str]:
        """Generate recommendations for error resolution."""
        recommendations = []
        
        if error_info.error_category == ErrorCategory.SECURITY:
            recommendations.append("Review security configurations and access controls")
            recommendations.append("Check for potential security breaches")
        
        elif error_info.error_category == ErrorCategory.CONFIGURATION:
            recommendations.append("Verify configuration files and settings")
            recommendations.append("Check for recent configuration changes")
        
        elif error_info.error_category == ErrorCategory.NETWORK:
            recommendations.append("Check network connectivity and firewall rules")
            recommendations.append("Verify external service availability")
        
        elif error_info.error_category == ErrorCategory.PERFORMANCE:
            recommendations.append("Monitor system resources (CPU, memory, disk)")
            recommendations.append("Consider scaling or optimization")
        
        # General recommendations
        if 'timeout' in error_info.message.lower():
            recommendations.append("Increase timeout values if appropriate")
            recommendations.append("Check for performance bottlenecks")
        
        if not recommendations:
            recommendations.append("Check application logs for additional context")
            recommendations.append("Verify system health and dependencies")
        
        return recommendations
    
    def get_analysis_report(self) -> Dict[str, Any]:
        """Get comprehensive error analysis report."""
        return {
            'pattern_detection': self.pattern_detector.get_statistics(),
            'incident_management': self.incident_manager.get_statistics(),
            'current_patterns': [
                {
                    'pattern_type': p.pattern_type.value,
                    'confidence': p.confidence,
                    'description': p.description,
                    'severity': p.severity.value,
                    'error_count': len(p.errors)
                } for p in self.pattern_detector.detect_patterns()
            ],
            'active_incidents': len(self.incident_manager.get_active_incidents()),
            'system_health': self._assess_system_health()
        }
    
    def _assess_system_health(self) -> str:
        """Assess overall system health based on errors and incidents."""
        active_incidents = self.incident_manager.get_active_incidents()
        critical_incidents = [i for i in active_incidents if i.severity == IncidentSeverity.CRITICAL]
        high_incidents = [i for i in active_incidents if i.severity == IncidentSeverity.HIGH]
        
        if critical_incidents:
            return "critical"
        elif len(high_incidents) > 2:
            return "degraded"
        elif active_incidents:
            return "warning"
        else:
            return "healthy"
    
    def close(self):
        """Close error analysis system."""
        self._shutdown.set()
        if self._analysis_thread:
            self._analysis_thread.join(timeout=5)


# Export main classes
__all__ = [
    'IncidentSeverity',
    'IncidentStatus',
    'ErrorPattern',
    'ErrorPatternMatch',
    'Incident',
    'ErrorPatternDetector',
    'IncidentManager',
    'ErrorAnalysisSystem'
]