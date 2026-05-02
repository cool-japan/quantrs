//! Structured JSON audit logging for quantum job operations.
//!
//! Every cloud API interaction (circuit submission, result fetch, job cancel)
//! can be recorded to an audit trail for compliance and debugging.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Type of quantum operation being audited
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub enum OperationType {
    /// Submit a quantum circuit for execution
    CircuitSubmit,
    /// Fetch results of a completed job
    ResultFetch,
    /// Query available backends
    BackendQuery,
    /// Authenticate with a cloud provider
    Authentication,
    /// Cancel a running or queued job
    JobCancel,
    /// Fetch device calibration data
    CalibrationFetch,
    /// Check status of a job
    StatusCheck,
}

impl std::fmt::Display for OperationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            OperationType::CircuitSubmit => "CircuitSubmit",
            OperationType::ResultFetch => "ResultFetch",
            OperationType::BackendQuery => "BackendQuery",
            OperationType::Authentication => "Authentication",
            OperationType::JobCancel => "JobCancel",
            OperationType::CalibrationFetch => "CalibrationFetch",
            OperationType::StatusCheck => "StatusCheck",
        };
        f.write_str(s)
    }
}

/// A single audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Unique event ID (timestamp + random suffix)
    pub id: String,
    /// Unix timestamp in milliseconds
    pub timestamp_ms: u64,
    /// Operation type
    pub operation: OperationType,
    /// Backend/provider name (e.g. "ibm_nairobi", "aws_sv1")
    pub backend_id: Option<String>,
    /// FNV-1a hex digest of the circuit string (for correlation without exposing circuit)
    pub circuit_hash: Option<String>,
    /// User or service identifier
    pub user_id: Option<String>,
    /// Whether the operation succeeded
    pub success: bool,
    /// Duration of the operation in milliseconds
    pub duration_ms: Option<u64>,
    /// Error description (if success == false)
    pub error: Option<String>,
    /// Additional key-value metadata
    pub metadata: HashMap<String, String>,
}

impl AuditEvent {
    /// Create a new audit event with current timestamp
    pub fn new(operation: OperationType, success: bool) -> Self {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;

        let id = format!("{}-{}", timestamp_ms, fastrand::u64(..));

        Self {
            id,
            timestamp_ms,
            operation,
            backend_id: None,
            circuit_hash: None,
            user_id: None,
            success,
            duration_ms: None,
            error: None,
            metadata: HashMap::new(),
        }
    }

    /// Set the backend identifier
    pub fn with_backend(mut self, backend: impl Into<String>) -> Self {
        self.backend_id = Some(backend.into());
        self
    }

    /// Set the circuit hash for correlation
    pub fn with_circuit_hash(mut self, hash: impl Into<String>) -> Self {
        self.circuit_hash = Some(hash.into());
        self
    }

    /// Set the user or service identifier
    pub fn with_user(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    /// Set the operation duration in milliseconds
    pub fn with_duration_ms(mut self, ms: u64) -> Self {
        self.duration_ms = Some(ms);
        self
    }

    /// Set an error description
    pub fn with_error(mut self, error: impl Into<String>) -> Self {
        self.error = Some(error.into());
        self
    }

    /// Add arbitrary metadata key-value pair
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Compute an FNV-1a hex string for circuit correlation.
///
/// Note: not cryptographically secure — suitable for correlation/deduplication only.
pub fn circuit_hash(circuit: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
    for byte in circuit.bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(0x100000001b3); // FNV prime
    }
    format!("{:016x}", h)
}

/// Errors from audit logging
#[derive(Debug)]
#[non_exhaustive]
pub enum AuditError {
    /// Underlying I/O failure
    IoError(std::io::Error),
    /// JSON serialization failure
    SerializationError(String),
}

impl std::fmt::Display for AuditError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuditError::IoError(e) => write!(f, "audit IO error: {}", e),
            AuditError::SerializationError(s) => write!(f, "audit serialization error: {}", s),
        }
    }
}

impl std::error::Error for AuditError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AuditError::IoError(e) => Some(e),
            AuditError::SerializationError(_) => None,
        }
    }
}

impl From<std::io::Error> for AuditError {
    fn from(e: std::io::Error) -> Self {
        AuditError::IoError(e)
    }
}

/// Trait for audit log sinks
pub trait AuditLogger: Send + Sync {
    /// Record an audit event
    fn log(&self, event: AuditEvent) -> Result<(), AuditError>;
    /// Flush any buffered events
    fn flush(&self) -> Result<(), AuditError>;
    /// Return all events — only `InMemoryAuditLogger` returns data; others return empty vec
    fn events(&self) -> Vec<AuditEvent> {
        vec![]
    }
}

/// Appends newline-delimited JSON audit events to a file
pub struct FileAuditLogger {
    path: PathBuf,
    file: Arc<Mutex<std::fs::File>>,
}

impl FileAuditLogger {
    /// Open (or create) the audit log file at the given path.
    pub fn new(path: impl Into<PathBuf>) -> Result<Self, AuditError> {
        let path = path.into();
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(AuditError::IoError)?;
        Ok(Self {
            path,
            file: Arc::new(Mutex::new(file)),
        })
    }

    /// Return the path of the underlying log file.
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }
}

impl AuditLogger for FileAuditLogger {
    fn log(&self, event: AuditEvent) -> Result<(), AuditError> {
        use std::io::Write;
        let json = serde_json::to_string(&event)
            .map_err(|e| AuditError::SerializationError(e.to_string()))?;
        let mut f = self.file.lock().unwrap_or_else(|e| e.into_inner());
        writeln!(f, "{}", json).map_err(AuditError::IoError)
    }

    fn flush(&self) -> Result<(), AuditError> {
        use std::io::Write;
        let mut f = self.file.lock().unwrap_or_else(|e| e.into_inner());
        f.flush().map_err(AuditError::IoError)
    }
}

/// In-memory audit logger for testing and in-process inspection
#[derive(Default)]
pub struct InMemoryAuditLogger {
    stored: Arc<Mutex<Vec<AuditEvent>>>,
}

impl InMemoryAuditLogger {
    /// Create a new empty in-memory logger
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the number of recorded events
    pub fn event_count(&self) -> usize {
        self.stored.lock().unwrap_or_else(|e| e.into_inner()).len()
    }
}

impl AuditLogger for InMemoryAuditLogger {
    fn log(&self, event: AuditEvent) -> Result<(), AuditError> {
        self.stored.lock().unwrap_or_else(|e| e.into_inner()).push(event);
        Ok(())
    }

    fn flush(&self) -> Result<(), AuditError> {
        Ok(())
    }

    fn events(&self) -> Vec<AuditEvent> {
        self.stored.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

/// Discards all audit events — for use when auditing is explicitly disabled
pub struct NullAuditLogger;

impl AuditLogger for NullAuditLogger {
    fn log(&self, _event: AuditEvent) -> Result<(), AuditError> {
        Ok(())
    }

    fn flush(&self) -> Result<(), AuditError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_in_memory_logger() {
        let logger = InMemoryAuditLogger::new();
        for _ in 0..100 {
            let event = AuditEvent::new(OperationType::CircuitSubmit, true)
                .with_backend("test_backend")
                .with_duration_ms(42);
            logger.log(event).expect("log should succeed");
        }
        assert_eq!(logger.event_count(), 100);
    }

    #[test]
    fn test_audit_event_json_roundtrip() {
        let event = AuditEvent::new(OperationType::ResultFetch, false)
            .with_backend("ibm_nairobi")
            .with_error("timeout");

        let json = serde_json::to_string(&event).expect("serialization should succeed");
        let parsed: AuditEvent = serde_json::from_str(&json).expect("deserialization should succeed");
        assert_eq!(parsed.backend_id, Some("ibm_nairobi".to_string()));
        assert!(!parsed.success);
        assert_eq!(parsed.error, Some("timeout".to_string()));
    }

    #[test]
    fn test_in_memory_logger_events() {
        let logger = InMemoryAuditLogger::new();
        let event = AuditEvent::new(OperationType::BackendQuery, true)
            .with_user("test-user")
            .with_metadata("region", "us-east-1");
        logger.log(event).expect("log should succeed");

        let events = logger.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].user_id, Some("test-user".to_string()));
        assert_eq!(
            events[0].metadata.get("region"),
            Some(&"us-east-1".to_string())
        );
    }

    #[test]
    fn test_file_audit_logger() {
        let dir = env::temp_dir();
        let path = dir.join(format!("quantrs_audit_test_{}.jsonl", fastrand::u64(..)));
        let logger = FileAuditLogger::new(&path).expect("file logger creation should succeed");

        let event = AuditEvent::new(OperationType::Authentication, true);
        logger.log(event).expect("log should succeed");
        logger.flush().expect("flush should succeed");

        assert_eq!(logger.path(), path.as_path());

        let content = std::fs::read_to_string(&path).expect("should read audit file");
        assert!(content.contains("Authentication"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_file_audit_logger_multiple_entries() {
        let dir = env::temp_dir();
        let path = dir.join(format!(
            "quantrs_audit_multi_{}.jsonl",
            fastrand::u64(..)
        ));
        let logger = FileAuditLogger::new(&path).expect("file logger creation should succeed");

        for op in [
            OperationType::CircuitSubmit,
            OperationType::ResultFetch,
            OperationType::JobCancel,
        ] {
            let event = AuditEvent::new(op, true);
            logger.log(event).expect("log should succeed");
        }
        logger.flush().expect("flush should succeed");

        let content = std::fs::read_to_string(&path).expect("should read audit file");
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 3);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_null_logger() {
        let logger = NullAuditLogger;
        let event = AuditEvent::new(OperationType::StatusCheck, true);
        logger.log(event).expect("null logger should always succeed");
        logger.flush().expect("null flush should always succeed");
        assert!(logger.events().is_empty());
    }

    #[test]
    fn test_circuit_hash_deterministic() {
        let circuit = "H q[0]; CNOT q[0],q[1];";
        let h1 = circuit_hash(circuit);
        let h2 = circuit_hash(circuit);
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 16);
    }

    #[test]
    fn test_circuit_hash_different_for_different_inputs() {
        let h1 = circuit_hash("H q[0];");
        let h2 = circuit_hash("X q[0];");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_operation_type_display() {
        assert_eq!(OperationType::CircuitSubmit.to_string(), "CircuitSubmit");
        assert_eq!(OperationType::Authentication.to_string(), "Authentication");
    }

    #[test]
    fn test_audit_event_builder_chain() {
        let event = AuditEvent::new(OperationType::CalibrationFetch, false)
            .with_backend("ibm_lagos")
            .with_circuit_hash("abcdef0123456789")
            .with_user("service-account")
            .with_duration_ms(250)
            .with_error("backend unavailable")
            .with_metadata("attempt", "3")
            .with_metadata("region", "eu-west");

        assert_eq!(event.backend_id, Some("ibm_lagos".to_string()));
        assert_eq!(event.circuit_hash, Some("abcdef0123456789".to_string()));
        assert_eq!(event.user_id, Some("service-account".to_string()));
        assert_eq!(event.duration_ms, Some(250));
        assert!(!event.success);
        assert_eq!(event.metadata.get("attempt"), Some(&"3".to_string()));
    }
}
