//! Real-Time Hardware Integration for Cloud Quantum Computers
//!
//! This module provides real-time integration capabilities with cloud quantum hardware,
//! including live job monitoring, streaming results, dynamic calibration tracking,
//! and real-time error rate monitoring. It enables responsive quantum-classical
//! hybrid algorithms with immediate feedback from quantum hardware.
//!
//! # Features
//! - Real-time job status monitoring with callbacks
//! - Streaming measurement results for iterative algorithms
//! - Dynamic hardware calibration tracking
//! - Live error rate monitoring and adaptation
//! - WebSocket-based event streaming (simulated)
//! - Circuit execution progress tracking
//! - Hardware availability notifications

use quantrs2_core::error::{QuantRS2Error, QuantRS2Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Real-time hardware integration manager
#[derive(Debug)]
pub struct RealtimeHardwareManager {
    /// Active hardware connections
    connections: Arc<RwLock<HashMap<String, HardwareConnection>>>,
    /// Job monitor for tracking execution
    job_monitor: JobMonitor,
    /// Calibration tracker
    calibration_tracker: CalibrationTracker,
    /// Event stream for real-time updates
    event_stream: EventStream,
    /// Configuration
    config: RealtimeConfig,
    /// Statistics
    stats: Arc<Mutex<RealtimeStats>>,
}

/// Configuration for real-time hardware integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeConfig {
    /// Polling interval for job status (milliseconds)
    pub polling_interval_ms: u64,
    /// Enable streaming results
    pub enable_streaming: bool,
    /// Maximum event buffer size
    pub max_event_buffer: usize,
    /// Calibration update interval (seconds)
    pub calibration_update_interval: u64,
    /// Enable adaptive error mitigation
    pub enable_adaptive_mitigation: bool,
    /// Timeout for real-time operations (seconds)
    pub operation_timeout: u64,
    /// Enable hardware availability notifications
    pub enable_availability_notifications: bool,
    /// Maximum concurrent jobs
    pub max_concurrent_jobs: usize,
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            polling_interval_ms: 500,
            enable_streaming: true,
            max_event_buffer: 1000,
            calibration_update_interval: 300,
            enable_adaptive_mitigation: true,
            operation_timeout: 3600,
            enable_availability_notifications: true,
            max_concurrent_jobs: 10,
        }
    }
}

/// Hardware connection state
#[derive(Debug, Clone)]
pub struct HardwareConnection {
    /// Connection ID
    pub id: String,
    /// Hardware provider
    pub provider: HardwareProvider,
    /// Connection status
    pub status: ConnectionStatus,
    /// Backend name
    pub backend: String,
    /// Connection timestamp
    pub connected_at: u64,
    /// Last heartbeat
    pub last_heartbeat: u64,
    /// Current calibration data
    pub calibration: Option<CalibrationData>,
}

/// Hardware provider types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HardwareProvider {
    IBMQuantum,
    GoogleQuantumAI,
    AmazonBraket,
    AzureQuantum,
    IonQ,
    Rigetti,
    Xanadu,
    Pasqal,
}

/// Connection status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionStatus {
    Connected,
    Connecting,
    Disconnected,
    Error,
    Maintenance,
}

/// Job monitor for tracking execution
pub struct JobMonitor {
    /// Active jobs being monitored
    active_jobs: Arc<RwLock<HashMap<String, JobState>>>,
    /// Job history
    job_history: Arc<Mutex<VecDeque<JobRecord>>>,
    /// Callback handlers for job events
    callbacks: Arc<Mutex<HashMap<String, Vec<Box<dyn Fn(&JobEvent) + Send + Sync>>>>>,
}

impl std::fmt::Debug for JobMonitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JobMonitor")
            .field("active_jobs", &"<jobs>")
            .field("job_history", &"<history>")
            .field("callbacks", &"<callbacks>")
            .finish()
    }
}

/// State of a monitored job
#[derive(Debug, Clone)]
pub struct JobState {
    /// Job ID
    pub job_id: String,
    /// Current status
    pub status: JobStatus,
    /// Progress (0.0 - 1.0)
    pub progress: f64,
    /// Start time
    pub start_time: Instant,
    /// Estimated completion time
    pub estimated_completion: Option<Duration>,
    /// Partial results (for streaming)
    pub partial_results: Vec<PartialResult>,
    /// Error information
    pub error_info: Option<String>,
    /// Queue position
    pub queue_position: Option<usize>,
}

/// Job status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
    TimedOut,
}

/// Partial result for streaming
#[derive(Debug, Clone)]
pub struct PartialResult {
    /// Result index
    pub index: usize,
    /// Measurement counts
    pub counts: HashMap<String, usize>,
    /// Timestamp
    pub timestamp: u64,
}

/// Job event for callbacks
#[derive(Debug, Clone)]
pub struct JobEvent {
    /// Event type
    pub event_type: JobEventType,
    /// Job ID
    pub job_id: String,
    /// Event data
    pub data: JobEventData,
    /// Timestamp
    pub timestamp: u64,
}

/// Types of job events
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobEventType {
    StatusChanged,
    ProgressUpdate,
    PartialResult,
    Completed,
    Failed,
    QueuePositionChanged,
}

/// Job event data
#[derive(Debug, Clone)]
pub enum JobEventData {
    Status(JobStatus),
    Progress(f64),
    Result(PartialResult),
    Error(String),
    QueuePosition(usize),
    None,
}

/// Job record for history
#[derive(Debug, Clone)]
pub struct JobRecord {
    /// Job ID
    pub job_id: String,
    /// Final status
    pub status: JobStatus,
    /// Start time
    pub start_time: u64,
    /// End time
    pub end_time: u64,
    /// Total shots
    pub total_shots: usize,
    /// Backend used
    pub backend: String,
}

/// Calibration tracker for hardware
#[derive(Debug)]
pub struct CalibrationTracker {
    /// Current calibration data by backend
    calibrations: Arc<RwLock<HashMap<String, CalibrationData>>>,
    /// Calibration history
    history: Arc<Mutex<HashMap<String, VecDeque<CalibrationSnapshot>>>>,
    /// Last update times
    last_updates: Arc<Mutex<HashMap<String, Instant>>>,
}

/// Hardware calibration data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationData {
    /// Backend name
    pub backend: String,
    /// Timestamp
    pub timestamp: u64,
    /// Single-qubit gate errors by qubit
    pub single_qubit_errors: HashMap<usize, f64>,
    /// Two-qubit gate errors by qubit pair
    pub two_qubit_errors: HashMap<(usize, usize), f64>,
    /// Readout errors by qubit
    pub readout_errors: HashMap<usize, f64>,
    /// T1 times by qubit (microseconds)
    pub t1_times: HashMap<usize, f64>,
    /// T2 times by qubit (microseconds)
    pub t2_times: HashMap<usize, f64>,
    /// Gate durations (nanoseconds)
    pub gate_durations: HashMap<String, f64>,
    /// Connectivity graph
    pub connectivity: Vec<(usize, usize)>,
}

/// Calibration snapshot for history
#[derive(Debug, Clone)]
pub struct CalibrationSnapshot {
    /// Timestamp
    pub timestamp: u64,
    /// Average single-qubit error
    pub avg_single_qubit_error: f64,
    /// Average two-qubit error
    pub avg_two_qubit_error: f64,
    /// Average readout error
    pub avg_readout_error: f64,
}

/// Event stream for real-time updates
pub struct EventStream {
    /// Event buffer
    buffer: Arc<Mutex<VecDeque<HardwareEvent>>>,
    /// Maximum buffer size
    max_size: usize,
    /// Event subscribers
    subscribers: Arc<Mutex<Vec<Box<dyn Fn(&HardwareEvent) + Send + Sync>>>>,
}

impl std::fmt::Debug for EventStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventStream")
            .field("buffer", &"<buffer>")
            .field("max_size", &self.max_size)
            .field("subscribers", &"<subscribers>")
            .finish()
    }
}

/// Hardware events
#[derive(Debug, Clone)]
pub struct HardwareEvent {
    /// Event type
    pub event_type: HardwareEventType,
    /// Source backend
    pub backend: String,
    /// Event data
    pub data: HardwareEventData,
    /// Timestamp
    pub timestamp: u64,
}

/// Types of hardware events
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareEventType {
    CalibrationUpdated,
    AvailabilityChanged,
    ErrorRateAlert,
    MaintenanceScheduled,
    JobQueued,
    JobStarted,
    JobCompleted,
}

/// Hardware event data
#[derive(Debug, Clone)]
pub enum HardwareEventData {
    Calibration(CalibrationData),
    Availability(bool),
    ErrorRate(f64),
    Maintenance { start: u64, end: u64 },
    JobInfo { job_id: String, shots: usize },
    None,
}

/// Statistics for real-time operations
#[derive(Debug, Clone, Default)]
pub struct RealtimeStats {
    /// Total jobs monitored
    pub jobs_monitored: u64,
    /// Jobs completed successfully
    pub jobs_completed: u64,
    /// Jobs failed
    pub jobs_failed: u64,
    /// Total events processed
    pub events_processed: u64,
    /// Calibration updates received
    pub calibration_updates: u64,
    /// Average job completion time
    pub avg_completion_time: Duration,
    /// Current active connections
    pub active_connections: usize,
}

impl RealtimeHardwareManager {
    /// Create a new real-time hardware manager
    #[must_use]
    pub fn new(config: RealtimeConfig) -> Self {
        Self {
            connections: Arc::new(RwLock::new(HashMap::new())),
            job_monitor: JobMonitor::new(),
            calibration_tracker: CalibrationTracker::new(),
            event_stream: EventStream::new(config.max_event_buffer),
            config,
            stats: Arc::new(Mutex::new(RealtimeStats::default())),
        }
    }

    /// Connect to hardware backend
    pub fn connect(&mut self, provider: HardwareProvider, backend: &str) -> QuantRS2Result<String> {
        let conn_id = format!("{provider:?}_{backend}");
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let connection = HardwareConnection {
            id: conn_id.clone(),
            provider,
            status: ConnectionStatus::Connected,
            backend: backend.to_string(),
            connected_at: now,
            last_heartbeat: now,
            calibration: None,
        };

        let mut connections = self.connections.write().map_err(|_| {
            QuantRS2Error::InvalidInput("Failed to acquire connections lock".to_string())
        })?;
        connections.insert(conn_id.clone(), connection);

        // Update stats
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| QuantRS2Error::InvalidInput("Failed to acquire stats lock".to_string()))?;
        stats.active_connections += 1;

        Ok(conn_id)
    }

    /// Disconnect from hardware backend
    pub fn disconnect(&mut self, connection_id: &str) -> QuantRS2Result<()> {
        let mut connections = self.connections.write().map_err(|_| {
            QuantRS2Error::InvalidInput("Failed to acquire connections lock".to_string())
        })?;

        if connections.remove(connection_id).is_some() {
            let mut stats = self.stats.lock().map_err(|_| {
                QuantRS2Error::InvalidInput("Failed to acquire stats lock".to_string())
            })?;
            if stats.active_connections > 0 {
                stats.active_connections -= 1;
            }
        }

        Ok(())
    }

    /// Submit job for real-time monitoring
    pub fn submit_job(&mut self, job_id: &str, connection_id: &str) -> QuantRS2Result<()> {
        let connections = self.connections.read().map_err(|_| {
            QuantRS2Error::InvalidInput("Failed to acquire connections lock".to_string())
        })?;

        if !connections.contains_key(connection_id) {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Connection {connection_id} not found"
            )));
        }

        // Create job state
        let job_state = JobState {
            job_id: job_id.to_string(),
            status: JobStatus::Queued,
            progress: 0.0,
            start_time: Instant::now(),
            estimated_completion: None,
            partial_results: Vec::new(),
            error_info: None,
            queue_position: Some(1),
        };

        self.job_monitor.add_job(job_state)?;

        // Update stats
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| QuantRS2Error::InvalidInput("Failed to acquire stats lock".to_string()))?;
        stats.jobs_monitored += 1;

        Ok(())
    }

    /// Get job status
    pub fn get_job_status(&self, job_id: &str) -> QuantRS2Result<JobStatus> {
        self.job_monitor.get_status(job_id)
    }

    /// Get job progress
    pub fn get_job_progress(&self, job_id: &str) -> QuantRS2Result<f64> {
        self.job_monitor.get_progress(job_id)
    }

    /// Update job status (simulates receiving update from hardware)
    pub fn update_job_status(
        &mut self,
        job_id: &str,
        status: JobStatus,
        progress: f64,
    ) -> QuantRS2Result<()> {
        self.job_monitor.update_status(job_id, status, progress)?;

        // Update stats if completed
        if status == JobStatus::Completed {
            let mut stats = self.stats.lock().map_err(|_| {
                QuantRS2Error::InvalidInput("Failed to acquire stats lock".to_string())
            })?;
            stats.jobs_completed += 1;
        } else if status == JobStatus::Failed {
            let mut stats = self.stats.lock().map_err(|_| {
                QuantRS2Error::InvalidInput("Failed to acquire stats lock".to_string())
            })?;
            stats.jobs_failed += 1;
        }

        Ok(())
    }

    /// Add partial result for streaming
    pub fn add_partial_result(
        &mut self,
        job_id: &str,
        counts: HashMap<String, usize>,
    ) -> QuantRS2Result<()> {
        self.job_monitor.add_partial_result(job_id, counts)
    }

    /// Get partial results for job
    pub fn get_partial_results(&self, job_id: &str) -> QuantRS2Result<Vec<PartialResult>> {
        self.job_monitor.get_partial_results(job_id)
    }

    /// Update calibration data for backend
    pub fn update_calibration(
        &mut self,
        backend: &str,
        calibration: CalibrationData,
    ) -> QuantRS2Result<()> {
        self.calibration_tracker
            .update_calibration(backend, calibration)?;

        // Update stats
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| QuantRS2Error::InvalidInput("Failed to acquire stats lock".to_string()))?;
        stats.calibration_updates += 1;

        Ok(())
    }

    /// Get current calibration for backend
    pub fn get_calibration(&self, backend: &str) -> QuantRS2Result<Option<CalibrationData>> {
        self.calibration_tracker.get_calibration(backend)
    }

    /// Get optimal qubits based on current calibration
    pub fn get_optimal_qubits(
        &self,
        backend: &str,
        num_qubits: usize,
    ) -> QuantRS2Result<Vec<usize>> {
        let calibration = self.get_calibration(backend)?;

        match calibration {
            Some(cal) => {
                // Sort qubits by error rate (lowest first)
                let mut qubits: Vec<(usize, f64)> = cal
                    .single_qubit_errors
                    .iter()
                    .map(|(&q, &e)| (q, e))
                    .collect();
                qubits.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                Ok(qubits
                    .into_iter()
                    .take(num_qubits)
                    .map(|(q, _)| q)
                    .collect())
            }
            None => {
                // No calibration data, return sequential qubits
                Ok((0..num_qubits).collect())
            }
        }
    }

    /// Get statistics
    pub fn get_stats(&self) -> QuantRS2Result<RealtimeStats> {
        let stats = self
            .stats
            .lock()
            .map_err(|_| QuantRS2Error::InvalidInput("Failed to acquire stats lock".to_string()))?;
        Ok(stats.clone())
    }

    /// Get active connections
    pub fn get_connections(&self) -> QuantRS2Result<Vec<HardwareConnection>> {
        let connections = self.connections.read().map_err(|_| {
            QuantRS2Error::InvalidInput("Failed to acquire connections lock".to_string())
        })?;
        Ok(connections.values().cloned().collect())
    }

    /// Register job event callback
    pub fn register_job_callback<F>(&mut self, job_id: &str, callback: F) -> QuantRS2Result<()>
    where
        F: Fn(&JobEvent) + Send + Sync + 'static,
    {
        self.job_monitor.register_callback(job_id, callback)
    }

    /// Check if backend is available
    pub fn is_backend_available(&self, connection_id: &str) -> QuantRS2Result<bool> {
        let connections = self.connections.read().map_err(|_| {
            QuantRS2Error::InvalidInput("Failed to acquire connections lock".to_string())
        })?;

        match connections.get(connection_id) {
            Some(conn) => Ok(conn.status == ConnectionStatus::Connected),
            None => Ok(false),
        }
    }
}

impl JobMonitor {
    /// Create new job monitor
    fn new() -> Self {
        Self {
            active_jobs: Arc::new(RwLock::new(HashMap::new())),
            job_history: Arc::new(Mutex::new(VecDeque::new())),
            callbacks: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Add job for monitoring
    fn add_job(&self, job_state: JobState) -> QuantRS2Result<()> {
        let mut jobs = self
            .active_jobs
            .write()
            .map_err(|_| QuantRS2Error::InvalidInput("Failed to acquire jobs lock".to_string()))?;
        jobs.insert(job_state.job_id.clone(), job_state);
        Ok(())
    }

    /// Get job status
    fn get_status(&self, job_id: &str) -> QuantRS2Result<JobStatus> {
        let jobs = self
            .active_jobs
            .read()
            .map_err(|_| QuantRS2Error::InvalidInput("Failed to acquire jobs lock".to_string()))?;

        jobs.get(job_id)
            .map(|j| j.status)
            .ok_or_else(|| QuantRS2Error::InvalidInput(format!("Job {job_id} not found")))
    }

    /// Get job progress
    fn get_progress(&self, job_id: &str) -> QuantRS2Result<f64> {
        let jobs = self
            .active_jobs
            .read()
            .map_err(|_| QuantRS2Error::InvalidInput("Failed to acquire jobs lock".to_string()))?;

        jobs.get(job_id)
            .map(|j| j.progress)
            .ok_or_else(|| QuantRS2Error::InvalidInput(format!("Job {job_id} not found")))
    }

    /// Update job status
    fn update_status(&self, job_id: &str, status: JobStatus, progress: f64) -> QuantRS2Result<()> {
        let mut jobs = self
            .active_jobs
            .write()
            .map_err(|_| QuantRS2Error::InvalidInput("Failed to acquire jobs lock".to_string()))?;

        if let Some(job) = jobs.get_mut(job_id) {
            job.status = status;
            job.progress = progress;

            // Trigger callbacks
            self.trigger_callback(
                job_id,
                JobEventType::StatusChanged,
                JobEventData::Status(status),
            )?;
            self.trigger_callback(
                job_id,
                JobEventType::ProgressUpdate,
                JobEventData::Progress(progress),
            )?;
        }

        Ok(())
    }

    /// Add partial result
    fn add_partial_result(
        &self,
        job_id: &str,
        counts: HashMap<String, usize>,
    ) -> QuantRS2Result<()> {
        let mut jobs = self
            .active_jobs
            .write()
            .map_err(|_| QuantRS2Error::InvalidInput("Failed to acquire jobs lock".to_string()))?;

        if let Some(job) = jobs.get_mut(job_id) {
            let result = PartialResult {
                index: job.partial_results.len(),
                counts,
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            };
            job.partial_results.push(result.clone());

            // Trigger callback
            drop(jobs);
            self.trigger_callback(
                job_id,
                JobEventType::PartialResult,
                JobEventData::Result(result),
            )?;
        }

        Ok(())
    }

    /// Get partial results
    fn get_partial_results(&self, job_id: &str) -> QuantRS2Result<Vec<PartialResult>> {
        let jobs = self
            .active_jobs
            .read()
            .map_err(|_| QuantRS2Error::InvalidInput("Failed to acquire jobs lock".to_string()))?;

        jobs.get(job_id)
            .map(|j| j.partial_results.clone())
            .ok_or_else(|| QuantRS2Error::InvalidInput(format!("Job {job_id} not found")))
    }

    /// Register callback for job events
    fn register_callback<F>(&self, job_id: &str, callback: F) -> QuantRS2Result<()>
    where
        F: Fn(&JobEvent) + Send + Sync + 'static,
    {
        let mut callbacks = self.callbacks.lock().map_err(|_| {
            QuantRS2Error::InvalidInput("Failed to acquire callbacks lock".to_string())
        })?;

        callbacks
            .entry(job_id.to_string())
            .or_insert_with(Vec::new)
            .push(Box::new(callback));

        Ok(())
    }

    /// Trigger callbacks for event
    fn trigger_callback(
        &self,
        job_id: &str,
        event_type: JobEventType,
        data: JobEventData,
    ) -> QuantRS2Result<()> {
        let callbacks = self.callbacks.lock().map_err(|_| {
            QuantRS2Error::InvalidInput("Failed to acquire callbacks lock".to_string())
        })?;

        if let Some(handlers) = callbacks.get(job_id) {
            let event = JobEvent {
                event_type,
                job_id: job_id.to_string(),
                data,
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            };

            for handler in handlers {
                handler(&event);
            }
        }

        Ok(())
    }
}

impl CalibrationTracker {
    /// Create new calibration tracker
    fn new() -> Self {
        Self {
            calibrations: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(Mutex::new(HashMap::new())),
            last_updates: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Update calibration data
    fn update_calibration(
        &self,
        backend: &str,
        calibration: CalibrationData,
    ) -> QuantRS2Result<()> {
        // Calculate snapshot
        let avg_single = if calibration.single_qubit_errors.is_empty() {
            0.0
        } else {
            calibration.single_qubit_errors.values().sum::<f64>()
                / calibration.single_qubit_errors.len() as f64
        };

        let avg_two = if calibration.two_qubit_errors.is_empty() {
            0.0
        } else {
            calibration.two_qubit_errors.values().sum::<f64>()
                / calibration.two_qubit_errors.len() as f64
        };

        let avg_readout = if calibration.readout_errors.is_empty() {
            0.0
        } else {
            calibration.readout_errors.values().sum::<f64>()
                / calibration.readout_errors.len() as f64
        };

        let snapshot = CalibrationSnapshot {
            timestamp: calibration.timestamp,
            avg_single_qubit_error: avg_single,
            avg_two_qubit_error: avg_two,
            avg_readout_error: avg_readout,
        };

        // Store calibration
        let mut calibrations = self.calibrations.write().map_err(|_| {
            QuantRS2Error::InvalidInput("Failed to acquire calibrations lock".to_string())
        })?;
        calibrations.insert(backend.to_string(), calibration);

        // Store snapshot in history
        let mut history = self.history.lock().map_err(|_| {
            QuantRS2Error::InvalidInput("Failed to acquire history lock".to_string())
        })?;
        history
            .entry(backend.to_string())
            .or_insert_with(VecDeque::new)
            .push_back(snapshot);

        // Update last update time
        let mut last_updates = self.last_updates.lock().map_err(|_| {
            QuantRS2Error::InvalidInput("Failed to acquire last_updates lock".to_string())
        })?;
        last_updates.insert(backend.to_string(), Instant::now());

        Ok(())
    }

    /// Get current calibration
    fn get_calibration(&self, backend: &str) -> QuantRS2Result<Option<CalibrationData>> {
        let calibrations = self.calibrations.read().map_err(|_| {
            QuantRS2Error::InvalidInput("Failed to acquire calibrations lock".to_string())
        })?;
        Ok(calibrations.get(backend).cloned())
    }
}

impl EventStream {
    /// Create new event stream
    fn new(max_size: usize) -> Self {
        Self {
            buffer: Arc::new(Mutex::new(VecDeque::new())),
            max_size,
            subscribers: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_realtime_manager_creation() {
        let config = RealtimeConfig::default();
        let manager = RealtimeHardwareManager::new(config);
        assert!(manager.get_stats().is_ok());
    }

    #[test]
    fn test_connect_disconnect() {
        let config = RealtimeConfig::default();
        let mut manager = RealtimeHardwareManager::new(config);

        let conn_id = manager
            .connect(HardwareProvider::IBMQuantum, "ibm_qasm_simulator")
            .expect("Connection should succeed");
        assert!(!conn_id.is_empty());

        let connections = manager
            .get_connections()
            .expect("Get connections should succeed");
        assert_eq!(connections.len(), 1);

        manager
            .disconnect(&conn_id)
            .expect("Disconnect should succeed");
        let connections = manager
            .get_connections()
            .expect("Get connections should succeed");
        assert_eq!(connections.len(), 0);
    }

    #[test]
    fn test_job_monitoring() {
        let config = RealtimeConfig::default();
        let mut manager = RealtimeHardwareManager::new(config);

        let conn_id = manager
            .connect(HardwareProvider::IBMQuantum, "ibm_qasm_simulator")
            .expect("Connection should succeed");

        manager
            .submit_job("job_123", &conn_id)
            .expect("Job submission should succeed");

        let status = manager
            .get_job_status("job_123")
            .expect("Get job status should succeed");
        assert_eq!(status, JobStatus::Queued);

        let progress = manager
            .get_job_progress("job_123")
            .expect("Get job progress should succeed");
        assert_eq!(progress, 0.0);
    }

    #[test]
    fn test_job_status_update() {
        let config = RealtimeConfig::default();
        let mut manager = RealtimeHardwareManager::new(config);

        let conn_id = manager
            .connect(HardwareProvider::IBMQuantum, "backend")
            .expect("Connection should succeed");

        manager
            .submit_job("job_456", &conn_id)
            .expect("Job submission should succeed");

        manager
            .update_job_status("job_456", JobStatus::Running, 0.5)
            .expect("Status update should succeed");
        let status = manager
            .get_job_status("job_456")
            .expect("Get job status should succeed");
        assert_eq!(status, JobStatus::Running);

        let progress = manager
            .get_job_progress("job_456")
            .expect("Get job progress should succeed");
        assert_eq!(progress, 0.5);
    }

    #[test]
    fn test_partial_results() {
        let config = RealtimeConfig::default();
        let mut manager = RealtimeHardwareManager::new(config);

        let conn_id = manager
            .connect(HardwareProvider::GoogleQuantumAI, "backend")
            .expect("Connection should succeed");

        manager
            .submit_job("job_789", &conn_id)
            .expect("Job submission should succeed");

        let mut counts = HashMap::new();
        counts.insert("00".to_string(), 450);
        counts.insert("11".to_string(), 550);

        manager
            .add_partial_result("job_789", counts)
            .expect("Add partial result should succeed");

        let results = manager
            .get_partial_results("job_789")
            .expect("Get partial results should succeed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].counts.get("00"), Some(&450));
    }

    #[test]
    fn test_calibration_tracking() {
        let config = RealtimeConfig::default();
        let mut manager = RealtimeHardwareManager::new(config);

        let mut single_qubit_errors = HashMap::new();
        single_qubit_errors.insert(0, 0.001);
        single_qubit_errors.insert(1, 0.002);
        single_qubit_errors.insert(2, 0.0015);

        let calibration = CalibrationData {
            backend: "test_backend".to_string(),
            timestamp: 12_345,
            single_qubit_errors,
            two_qubit_errors: HashMap::new(),
            readout_errors: HashMap::new(),
            t1_times: HashMap::new(),
            t2_times: HashMap::new(),
            gate_durations: HashMap::new(),
            connectivity: vec![(0, 1), (1, 2)],
        };

        manager
            .update_calibration("test_backend", calibration)
            .expect("Calibration update should succeed");

        let cal = manager
            .get_calibration("test_backend")
            .expect("Get calibration should succeed");
        assert!(cal.is_some());
        assert_eq!(
            cal.expect("Calibration data should exist")
                .single_qubit_errors
                .len(),
            3
        );
    }

    #[test]
    fn test_optimal_qubits() {
        let config = RealtimeConfig::default();
        let mut manager = RealtimeHardwareManager::new(config);

        let mut single_qubit_errors = HashMap::new();
        single_qubit_errors.insert(0, 0.005);
        single_qubit_errors.insert(1, 0.001);
        single_qubit_errors.insert(2, 0.003);
        single_qubit_errors.insert(3, 0.002);

        let calibration = CalibrationData {
            backend: "backend".to_string(),
            timestamp: 12_345,
            single_qubit_errors,
            two_qubit_errors: HashMap::new(),
            readout_errors: HashMap::new(),
            t1_times: HashMap::new(),
            t2_times: HashMap::new(),
            gate_durations: HashMap::new(),
            connectivity: vec![],
        };

        manager
            .update_calibration("backend", calibration)
            .expect("Calibration update should succeed");

        let optimal = manager
            .get_optimal_qubits("backend", 2)
            .expect("Get optimal qubits should succeed");
        assert_eq!(optimal.len(), 2);
        // Should return qubits with lowest error rates (1 and 3)
        assert!(optimal.contains(&1));
        assert!(optimal.contains(&3));
    }

    #[test]
    fn test_backend_availability() {
        let config = RealtimeConfig::default();
        let mut manager = RealtimeHardwareManager::new(config);

        let conn_id = manager
            .connect(HardwareProvider::AmazonBraket, "backend")
            .expect("Connection should succeed");

        assert!(manager
            .is_backend_available(&conn_id)
            .expect("Backend availability check should succeed"));
        assert!(!manager
            .is_backend_available("nonexistent")
            .expect("Backend availability check should succeed"));
    }

    #[test]
    fn test_statistics() {
        let config = RealtimeConfig::default();
        let mut manager = RealtimeHardwareManager::new(config);

        let conn_id = manager
            .connect(HardwareProvider::IonQ, "backend")
            .expect("Connection should succeed");

        manager
            .submit_job("job_a", &conn_id)
            .expect("Job submission should succeed");
        manager
            .submit_job("job_b", &conn_id)
            .expect("Job submission should succeed");

        manager
            .update_job_status("job_a", JobStatus::Completed, 1.0)
            .expect("Status update should succeed");
        manager
            .update_job_status("job_b", JobStatus::Failed, 0.5)
            .expect("Status update should succeed");

        let stats = manager.get_stats().expect("Get stats should succeed");
        assert_eq!(stats.jobs_monitored, 2);
        assert_eq!(stats.jobs_completed, 1);
        assert_eq!(stats.jobs_failed, 1);
    }

    #[test]
    fn test_config_defaults() {
        let config = RealtimeConfig::default();

        assert_eq!(config.polling_interval_ms, 500);
        assert!(config.enable_streaming);
        assert_eq!(config.max_event_buffer, 1000);
        assert!(config.enable_adaptive_mitigation);
        assert_eq!(config.max_concurrent_jobs, 10);
    }

    #[test]
    fn test_multiple_providers() {
        let config = RealtimeConfig::default();
        let mut manager = RealtimeHardwareManager::new(config);

        manager
            .connect(HardwareProvider::IBMQuantum, "ibm_backend")
            .expect("IBM connection should succeed");
        manager
            .connect(HardwareProvider::GoogleQuantumAI, "google_backend")
            .expect("Google connection should succeed");
        manager
            .connect(HardwareProvider::AzureQuantum, "azure_backend")
            .expect("Azure connection should succeed");

        let connections = manager
            .get_connections()
            .expect("Get connections should succeed");
        assert_eq!(connections.len(), 3);
    }

    #[test]
    fn test_job_completion() {
        let config = RealtimeConfig::default();
        let mut manager = RealtimeHardwareManager::new(config);

        let conn_id = manager
            .connect(HardwareProvider::Rigetti, "backend")
            .expect("Connection should succeed");

        manager
            .submit_job("job_complete", &conn_id)
            .expect("Job submission should succeed");

        // Simulate job progress
        manager
            .update_job_status("job_complete", JobStatus::Running, 0.0)
            .expect("Status update should succeed");
        manager
            .update_job_status("job_complete", JobStatus::Running, 0.5)
            .expect("Status update should succeed");
        manager
            .update_job_status("job_complete", JobStatus::Completed, 1.0)
            .expect("Status update should succeed");

        let status = manager
            .get_job_status("job_complete")
            .expect("Get job status should succeed");
        assert_eq!(status, JobStatus::Completed);
    }
}
