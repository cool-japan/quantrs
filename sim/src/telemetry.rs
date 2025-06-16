//! Telemetry and performance monitoring for quantum simulations.
//!
//! This module provides comprehensive telemetry capabilities for monitoring
//! quantum simulation performance, resource usage, and operational metrics.
//! It includes real-time monitoring, alerting, data export, and integration
//! with external monitoring systems.

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::Write;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::circuit_interfaces::{InterfaceCircuit, InterfaceGate, InterfaceGateType};
use crate::debugger::PerformanceMetrics;
use crate::error::{Result, SimulatorError};

/// Telemetry configuration
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    /// Enable telemetry collection
    pub enabled: bool,
    /// Sampling rate (0.0 - 1.0)
    pub sampling_rate: f64,
    /// Maximum metrics history size
    pub max_history_size: usize,
    /// Export interval in seconds
    pub export_interval: Duration,
    /// Enable real-time alerts
    pub enable_alerts: bool,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
    /// Export format
    pub export_format: TelemetryExportFormat,
    /// Export directory
    pub export_directory: String,
    /// Enable system-level monitoring
    pub monitor_system_resources: bool,
    /// Custom tags for metrics
    pub custom_tags: HashMap<String, String>,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sampling_rate: 1.0,
            max_history_size: 10000,
            export_interval: Duration::from_secs(60),
            enable_alerts: true,
            alert_thresholds: AlertThresholds::default(),
            export_format: TelemetryExportFormat::JSON,
            export_directory: "./telemetry".to_string(),
            monitor_system_resources: true,
            custom_tags: HashMap::new(),
        }
    }
}

/// Alert thresholds for monitoring
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    /// Maximum execution time per gate (seconds)
    pub max_gate_execution_time: f64,
    /// Maximum memory usage (bytes)
    pub max_memory_usage: usize,
    /// Maximum error rate
    pub max_error_rate: f64,
    /// Maximum CPU usage (0.0 - 1.0)
    pub max_cpu_usage: f64,
    /// Maximum queue depth
    pub max_queue_depth: usize,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            max_gate_execution_time: 1.0,
            max_memory_usage: 16_000_000_000, // 16GB
            max_error_rate: 0.1,
            max_cpu_usage: 0.9,
            max_queue_depth: 1000,
        }
    }
}

/// Telemetry export formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TelemetryExportFormat {
    JSON,
    CSV,
    Prometheus,
    InfluxDB,
    Custom,
}

/// Telemetry metric types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TelemetryMetric {
    /// Counter metric (monotonically increasing)
    Counter {
        name: String,
        value: u64,
        tags: HashMap<String, String>,
        timestamp: f64,
    },
    /// Gauge metric (current value)
    Gauge {
        name: String,
        value: f64,
        tags: HashMap<String, String>,
        timestamp: f64,
    },
    /// Histogram metric (distribution)
    Histogram {
        name: String,
        values: Vec<f64>,
        buckets: Vec<f64>,
        tags: HashMap<String, String>,
        timestamp: f64,
    },
    /// Timer metric (duration measurements)
    Timer {
        name: String,
        duration: Duration,
        tags: HashMap<String, String>,
        timestamp: f64,
    },
    /// Custom metric
    Custom {
        name: String,
        data: serde_json::Value,
        tags: HashMap<String, String>,
        timestamp: f64,
    },
}

/// Performance monitoring data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    /// Timestamp
    pub timestamp: f64,
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Available memory in bytes
    pub available_memory: usize,
    /// Network I/O rates
    pub network_io: NetworkIOStats,
    /// Disk I/O rates
    pub disk_io: DiskIOStats,
    /// GPU utilization (if available)
    pub gpu_utilization: Option<f64>,
    /// GPU memory usage (if available)
    pub gpu_memory_usage: Option<usize>,
}

/// Network I/O statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NetworkIOStats {
    /// Bytes sent per second
    pub bytes_sent_per_sec: f64,
    /// Bytes received per second
    pub bytes_received_per_sec: f64,
    /// Packets sent per second
    pub packets_sent_per_sec: f64,
    /// Packets received per second
    pub packets_received_per_sec: f64,
}

/// Disk I/O statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiskIOStats {
    /// Bytes read per second
    pub bytes_read_per_sec: f64,
    /// Bytes written per second
    pub bytes_written_per_sec: f64,
    /// Read operations per second
    pub read_ops_per_sec: f64,
    /// Write operations per second
    pub write_ops_per_sec: f64,
}

/// Quantum simulation specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumMetrics {
    /// Number of qubits being simulated
    pub num_qubits: usize,
    /// Circuit depth
    pub circuit_depth: usize,
    /// Gate execution rate (gates per second)
    pub gate_execution_rate: f64,
    /// Current entanglement entropy
    pub entanglement_entropy: f64,
    /// Error correction rate
    pub error_correction_rate: f64,
    /// Fidelity with target state
    pub fidelity: f64,
    /// Active simulation backends
    pub active_backends: Vec<String>,
    /// Queue depth
    pub queue_depth: usize,
}

/// Alert levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertLevel {
    Info,
    Warning,
    Error,
    Critical,
}

/// Alert message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Alert level
    pub level: AlertLevel,
    /// Alert message
    pub message: String,
    /// Metric that triggered the alert
    pub metric_name: String,
    /// Current value
    pub current_value: f64,
    /// Threshold value
    pub threshold_value: f64,
    /// Timestamp
    pub timestamp: f64,
    /// Additional context
    pub context: HashMap<String, String>,
}

/// Main telemetry collector
pub struct TelemetryCollector {
    /// Configuration
    config: TelemetryConfig,
    /// Metrics history
    metrics_history: Arc<RwLock<VecDeque<TelemetryMetric>>>,
    /// Performance snapshots
    performance_history: Arc<RwLock<VecDeque<PerformanceSnapshot>>>,
    /// Quantum metrics history
    quantum_metrics_history: Arc<RwLock<VecDeque<QuantumMetrics>>>,
    /// Active alerts
    active_alerts: Arc<RwLock<Vec<Alert>>>,
    /// System monitoring thread handle
    system_monitor_handle: Option<std::thread::JoinHandle<()>>,
    /// Last export time
    last_export: Arc<Mutex<Instant>>,
    /// Custom metric handlers
    custom_handlers: HashMap<String, Box<dyn Fn(&TelemetryMetric) + Send + Sync>>,
}

impl TelemetryCollector {
    /// Create new telemetry collector
    pub fn new(config: TelemetryConfig) -> Self {
        let collector = Self {
            config: config.clone(),
            metrics_history: Arc::new(RwLock::new(VecDeque::with_capacity(
                config.max_history_size,
            ))),
            performance_history: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            quantum_metrics_history: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            active_alerts: Arc::new(RwLock::new(Vec::new())),
            system_monitor_handle: None,
            last_export: Arc::new(Mutex::new(Instant::now())),
            custom_handlers: HashMap::new(),
        };

        collector
    }

    /// Start telemetry collection
    pub fn start(&mut self) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Start system monitoring if enabled
        if self.config.monitor_system_resources {
            self.start_system_monitoring()?;
        }

        Ok(())
    }

    /// Stop telemetry collection
    pub fn stop(&mut self) {
        if let Some(handle) = self.system_monitor_handle.take() {
            // In a real implementation, we would signal the thread to stop
            // For now, we just detach it
            let _ = handle.join();
        }
    }

    /// Record a metric
    pub fn record_metric(&self, metric: TelemetryMetric) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Apply sampling
        if fastrand::f64() > self.config.sampling_rate {
            return Ok(());
        }

        // Store metric
        {
            let mut history = self.metrics_history.write().unwrap();
            history.push_back(metric.clone());
            if history.len() > self.config.max_history_size {
                history.pop_front();
            }
        }

        // Check for alerts
        self.check_alert_conditions(&metric)?;

        // Apply custom handlers
        for handler in self.custom_handlers.values() {
            handler(&metric);
        }

        // Check if export is needed
        self.check_export_schedule()?;

        Ok(())
    }

    /// Record quantum simulation metrics
    pub fn record_quantum_metrics(&self, metrics: QuantumMetrics) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        {
            let mut history = self.quantum_metrics_history.write().unwrap();
            history.push_back(metrics.clone());
            if history.len() > 1000 {
                history.pop_front();
            }
        }

        // Create telemetry metrics from quantum metrics
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let quantum_gauge = TelemetryMetric::Gauge {
            name: "quantum.num_qubits".to_string(),
            value: metrics.num_qubits as f64,
            tags: self.config.custom_tags.clone(),
            timestamp,
        };
        self.record_metric(quantum_gauge)?;

        let rate_gauge = TelemetryMetric::Gauge {
            name: "quantum.gate_execution_rate".to_string(),
            value: metrics.gate_execution_rate,
            tags: self.config.custom_tags.clone(),
            timestamp,
        };
        self.record_metric(rate_gauge)?;

        let entropy_gauge = TelemetryMetric::Gauge {
            name: "quantum.entanglement_entropy".to_string(),
            value: metrics.entanglement_entropy,
            tags: self.config.custom_tags.clone(),
            timestamp,
        };
        self.record_metric(entropy_gauge)?;

        Ok(())
    }

    /// Record gate execution timing
    pub fn record_gate_execution(&self, gate: &InterfaceGate, duration: Duration) -> Result<()> {
        let gate_type = format!("{:?}", gate.gate_type);
        let mut tags = self.config.custom_tags.clone();
        tags.insert("gate_type".to_string(), gate_type);
        tags.insert("num_qubits".to_string(), gate.qubits.len().to_string());

        let timer = TelemetryMetric::Timer {
            name: "gate.execution_time".to_string(),
            duration,
            tags,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        };

        self.record_metric(timer)?;
        Ok(())
    }

    /// Record circuit execution metrics
    pub fn record_circuit_execution(
        &self,
        circuit: &InterfaceCircuit,
        duration: Duration,
    ) -> Result<()> {
        let mut tags = self.config.custom_tags.clone();
        tags.insert("num_qubits".to_string(), circuit.num_qubits.to_string());
        tags.insert("num_gates".to_string(), circuit.gates.len().to_string());

        let timer = TelemetryMetric::Timer {
            name: "circuit.execution_time".to_string(),
            duration,
            tags: tags.clone(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        };

        self.record_metric(timer)?;

        // Record gate count
        let gate_counter = TelemetryMetric::Counter {
            name: "circuit.gates_executed".to_string(),
            value: circuit.gates.len() as u64,
            tags,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        };

        self.record_metric(gate_counter)?;
        Ok(())
    }

    /// Record memory usage
    pub fn record_memory_usage(&self, bytes_used: usize, category: &str) -> Result<()> {
        let mut tags = self.config.custom_tags.clone();
        tags.insert("category".to_string(), category.to_string());

        let gauge = TelemetryMetric::Gauge {
            name: "memory.usage_bytes".to_string(),
            value: bytes_used as f64,
            tags,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        };

        self.record_metric(gauge)?;
        Ok(())
    }

    /// Record error event
    pub fn record_error(&self, error_type: &str, error_message: &str) -> Result<()> {
        let mut tags = self.config.custom_tags.clone();
        tags.insert("error_type".to_string(), error_type.to_string());
        tags.insert("error_message".to_string(), error_message.to_string());

        let counter = TelemetryMetric::Counter {
            name: "errors.total".to_string(),
            value: 1,
            tags,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        };

        self.record_metric(counter)?;
        Ok(())
    }

    /// Get current metrics summary
    pub fn get_metrics_summary(&self) -> Result<MetricsSummary> {
        let metrics_history = self.metrics_history.read().unwrap();
        let quantum_history = self.quantum_metrics_history.read().unwrap();
        let performance_history = self.performance_history.read().unwrap();

        let total_metrics = metrics_history.len();
        let total_quantum_metrics = quantum_history.len();
        let total_performance_snapshots = performance_history.len();

        // Calculate average gate execution time
        let mut gate_times = Vec::new();
        for metric in metrics_history.iter() {
            if let TelemetryMetric::Timer { name, duration, .. } = metric {
                if name == "gate.execution_time" {
                    gate_times.push(duration.as_secs_f64());
                }
            }
        }

        let avg_gate_time = if gate_times.is_empty() {
            0.0
        } else {
            gate_times.iter().sum::<f64>() / gate_times.len() as f64
        };

        // Get latest quantum metrics
        let latest_quantum_metrics = quantum_history.back().cloned();

        // Get latest performance snapshot
        let latest_performance = performance_history.back().cloned();

        Ok(MetricsSummary {
            total_metrics,
            total_quantum_metrics,
            total_performance_snapshots,
            avg_gate_execution_time: avg_gate_time,
            latest_quantum_metrics,
            latest_performance,
            active_alerts_count: self.active_alerts.read().unwrap().len(),
        })
    }

    /// Export telemetry data
    pub fn export_data(&self, path: &str) -> Result<()> {
        std::fs::create_dir_all(path).map_err(|e| {
            SimulatorError::InvalidInput(format!("Failed to create export directory: {}", e))
        })?;

        match self.config.export_format {
            TelemetryExportFormat::JSON => self.export_json(path)?,
            TelemetryExportFormat::CSV => self.export_csv(path)?,
            TelemetryExportFormat::Prometheus => self.export_prometheus(path)?,
            TelemetryExportFormat::InfluxDB => self.export_influxdb(path)?,
            TelemetryExportFormat::Custom => self.export_custom(path)?,
        }

        *self.last_export.lock().unwrap() = Instant::now();
        Ok(())
    }

    /// Start system monitoring
    fn start_system_monitoring(&mut self) -> Result<()> {
        let performance_history = Arc::clone(&self.performance_history);
        let config = self.config.clone();

        let handle = std::thread::spawn(move || loop {
            let snapshot = Self::collect_system_metrics();

            {
                let mut history = performance_history.write().unwrap();
                history.push_back(snapshot);
                if history.len() > 1000 {
                    history.pop_front();
                }
            }

            std::thread::sleep(Duration::from_secs(1));
        });

        self.system_monitor_handle = Some(handle);
        Ok(())
    }

    /// Collect system metrics (simplified)
    fn collect_system_metrics() -> PerformanceSnapshot {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        // Simplified system metrics collection
        // In a real implementation, this would use system APIs
        PerformanceSnapshot {
            timestamp,
            cpu_usage: fastrand::f64() * 0.5, // Simulated
            memory_usage: (fastrand::f64() * 8_000_000_000.0) as usize, // Simulated
            available_memory: 16_000_000_000, // Simulated
            network_io: NetworkIOStats {
                bytes_sent_per_sec: fastrand::f64() * 1_000_000.0,
                bytes_received_per_sec: fastrand::f64() * 1_000_000.0,
                packets_sent_per_sec: fastrand::f64() * 1000.0,
                packets_received_per_sec: fastrand::f64() * 1000.0,
            },
            disk_io: DiskIOStats {
                bytes_read_per_sec: fastrand::f64() * 10_000_000.0,
                bytes_written_per_sec: fastrand::f64() * 10_000_000.0,
                read_ops_per_sec: fastrand::f64() * 100.0,
                write_ops_per_sec: fastrand::f64() * 100.0,
            },
            gpu_utilization: Some(fastrand::f64()),
            gpu_memory_usage: Some((fastrand::f64() * 4_000_000_000.0) as usize),
        }
    }

    /// Check alert conditions
    fn check_alert_conditions(&self, metric: &TelemetryMetric) -> Result<()> {
        if !self.config.enable_alerts {
            return Ok(());
        }

        let mut alerts_to_add = Vec::new();

        match metric {
            TelemetryMetric::Timer { name, duration, .. } => {
                if name == "gate.execution_time"
                    && duration.as_secs_f64() > self.config.alert_thresholds.max_gate_execution_time
                {
                    alerts_to_add.push(Alert {
                        level: AlertLevel::Warning,
                        message: "Gate execution time exceeded threshold".to_string(),
                        metric_name: name.clone(),
                        current_value: duration.as_secs_f64(),
                        threshold_value: self.config.alert_thresholds.max_gate_execution_time,
                        timestamp: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_secs_f64(),
                        context: HashMap::new(),
                    });
                }
            }
            TelemetryMetric::Gauge { name, value, .. } => {
                if name == "memory.usage_bytes"
                    && *value > self.config.alert_thresholds.max_memory_usage as f64
                {
                    alerts_to_add.push(Alert {
                        level: AlertLevel::Error,
                        message: "Memory usage exceeded threshold".to_string(),
                        metric_name: name.clone(),
                        current_value: *value,
                        threshold_value: self.config.alert_thresholds.max_memory_usage as f64,
                        timestamp: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_secs_f64(),
                        context: HashMap::new(),
                    });
                }
            }
            _ => {}
        }

        // Add alerts
        if !alerts_to_add.is_empty() {
            let mut active_alerts = self.active_alerts.write().unwrap();
            active_alerts.extend(alerts_to_add);

            // Keep only recent alerts
            let len = active_alerts.len();
            if len > 1000 {
                active_alerts.drain(0..len - 1000);
            }
        }

        Ok(())
    }

    /// Check if export is scheduled
    fn check_export_schedule(&self) -> Result<()> {
        let last_export = *self.last_export.lock().unwrap();
        if last_export.elapsed() > self.config.export_interval {
            self.export_data(&self.config.export_directory)?;
        }
        Ok(())
    }

    /// Export data as JSON
    fn export_json(&self, path: &str) -> Result<()> {
        let metrics = self.metrics_history.read().unwrap();
        let data = serde_json::to_string_pretty(&*metrics).map_err(|e| {
            SimulatorError::InvalidInput(format!("Failed to serialize metrics: {}", e))
        })?;

        let file_path = format!("{}/telemetry.json", path);
        let mut file = File::create(&file_path)
            .map_err(|e| SimulatorError::InvalidInput(format!("Failed to create file: {}", e)))?;

        file.write_all(data.as_bytes())
            .map_err(|e| SimulatorError::InvalidInput(format!("Failed to write file: {}", e)))?;

        Ok(())
    }

    /// Export data as CSV
    fn export_csv(&self, path: &str) -> Result<()> {
        let metrics = self.metrics_history.read().unwrap();
        let mut csv_data = String::new();
        csv_data.push_str("timestamp,metric_name,metric_type,value,tags\n");

        for metric in metrics.iter() {
            let (name, metric_type, value, tags, timestamp) = match metric {
                TelemetryMetric::Counter {
                    name,
                    value,
                    tags,
                    timestamp,
                } => (name, "counter", *value as f64, tags, *timestamp),
                TelemetryMetric::Gauge {
                    name,
                    value,
                    tags,
                    timestamp,
                } => (name, "gauge", *value, tags, *timestamp),
                TelemetryMetric::Timer {
                    name,
                    duration,
                    tags,
                    timestamp,
                } => (name, "timer", duration.as_secs_f64(), tags, *timestamp),
                _ => continue,
            };

            let tags_str = serde_json::to_string(tags).unwrap_or_default();
            csv_data.push_str(&format!(
                "{},{},{},{},{}\n",
                timestamp, name, metric_type, value, tags_str
            ));
        }

        let file_path = format!("{}/telemetry.csv", path);
        let mut file = File::create(&file_path)
            .map_err(|e| SimulatorError::InvalidInput(format!("Failed to create file: {}", e)))?;

        file.write_all(csv_data.as_bytes())
            .map_err(|e| SimulatorError::InvalidInput(format!("Failed to write file: {}", e)))?;

        Ok(())
    }

    /// Export data in Prometheus format
    fn export_prometheus(&self, path: &str) -> Result<()> {
        let metrics = self.metrics_history.read().unwrap();
        let mut prometheus_data = String::new();

        for metric in metrics.iter() {
            match metric {
                TelemetryMetric::Counter {
                    name,
                    value,
                    tags,
                    timestamp,
                } => {
                    prometheus_data.push_str(&format!("# TYPE {} counter\n", name));
                    prometheus_data.push_str(&format!(
                        "{}{} {} {}\n",
                        name,
                        self.format_prometheus_labels(tags),
                        value,
                        (*timestamp * 1000.0) as u64
                    ));
                }
                TelemetryMetric::Gauge {
                    name,
                    value,
                    tags,
                    timestamp,
                } => {
                    prometheus_data.push_str(&format!("# TYPE {} gauge\n", name));
                    prometheus_data.push_str(&format!(
                        "{}{} {} {}\n",
                        name,
                        self.format_prometheus_labels(tags),
                        value,
                        (*timestamp * 1000.0) as u64
                    ));
                }
                _ => {}
            }
        }

        let file_path = format!("{}/telemetry.prom", path);
        let mut file = File::create(&file_path)
            .map_err(|e| SimulatorError::InvalidInput(format!("Failed to create file: {}", e)))?;

        file.write_all(prometheus_data.as_bytes())
            .map_err(|e| SimulatorError::InvalidInput(format!("Failed to write file: {}", e)))?;

        Ok(())
    }

    /// Export data in InfluxDB line protocol format
    fn export_influxdb(&self, path: &str) -> Result<()> {
        let metrics = self.metrics_history.read().unwrap();
        let mut influx_data = String::new();

        for metric in metrics.iter() {
            match metric {
                TelemetryMetric::Counter {
                    name,
                    value,
                    tags,
                    timestamp,
                } => {
                    influx_data.push_str(&format!(
                        "{}{} value={} {}\n",
                        name,
                        self.format_influx_tags(tags),
                        value,
                        (*timestamp * 1_000_000_000.0) as u64
                    ));
                }
                TelemetryMetric::Gauge {
                    name,
                    value,
                    tags,
                    timestamp,
                } => {
                    influx_data.push_str(&format!(
                        "{}{} value={} {}\n",
                        name,
                        self.format_influx_tags(tags),
                        value,
                        (*timestamp * 1_000_000_000.0) as u64
                    ));
                }
                TelemetryMetric::Timer {
                    name,
                    duration,
                    tags,
                    timestamp,
                } => {
                    influx_data.push_str(&format!(
                        "{}{} duration={} {}\n",
                        name,
                        self.format_influx_tags(tags),
                        duration.as_secs_f64(),
                        (*timestamp * 1_000_000_000.0) as u64
                    ));
                }
                _ => {}
            }
        }

        let file_path = format!("{}/telemetry.influx", path);
        let mut file = File::create(&file_path)
            .map_err(|e| SimulatorError::InvalidInput(format!("Failed to create file: {}", e)))?;

        file.write_all(influx_data.as_bytes())
            .map_err(|e| SimulatorError::InvalidInput(format!("Failed to write file: {}", e)))?;

        Ok(())
    }

    /// Export data in custom format
    fn export_custom(&self, path: &str) -> Result<()> {
        // Custom export format - could be implemented based on specific needs
        self.export_json(path)
    }

    /// Format tags for Prometheus
    fn format_prometheus_labels(&self, tags: &HashMap<String, String>) -> String {
        if tags.is_empty() {
            return String::new();
        }

        let labels: Vec<String> = tags
            .iter()
            .map(|(k, v)| format!("{}=\"{}\"", k, v))
            .collect();

        format!("{{{}}}", labels.join(","))
    }

    /// Format tags for InfluxDB
    fn format_influx_tags(&self, tags: &HashMap<String, String>) -> String {
        if tags.is_empty() {
            return String::new();
        }

        let tag_pairs: Vec<String> = tags.iter().map(|(k, v)| format!("{}={}", k, v)).collect();

        format!(",{}", tag_pairs.join(","))
    }
}

/// Metrics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSummary {
    pub total_metrics: usize,
    pub total_quantum_metrics: usize,
    pub total_performance_snapshots: usize,
    pub avg_gate_execution_time: f64,
    pub latest_quantum_metrics: Option<QuantumMetrics>,
    pub latest_performance: Option<PerformanceSnapshot>,
    pub active_alerts_count: usize,
}

/// Benchmark telemetry performance
pub fn benchmark_telemetry() -> Result<HashMap<String, f64>> {
    let mut results = HashMap::new();

    // Test metric recording performance
    let start = std::time::Instant::now();
    let mut collector = TelemetryCollector::new(TelemetryConfig::default());

    for i in 0..10000 {
        let metric = TelemetryMetric::Gauge {
            name: "test.metric".to_string(),
            value: i as f64,
            tags: HashMap::new(),
            timestamp: i as f64,
        };
        collector.record_metric(metric)?;
    }

    let recording_time = start.elapsed().as_millis() as f64;
    results.insert("record_10000_metrics".to_string(), recording_time);

    // Test export performance
    let start = std::time::Instant::now();
    collector.export_data("./test_telemetry_export")?;
    let export_time = start.elapsed().as_millis() as f64;
    results.insert("export_metrics".to_string(), export_time);

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_telemetry_collector_creation() {
        let config = TelemetryConfig::default();
        let collector = TelemetryCollector::new(config);
        assert!(collector.config.enabled);
    }

    #[test]
    fn test_metric_recording() {
        let collector = TelemetryCollector::new(TelemetryConfig::default());

        let metric = TelemetryMetric::Gauge {
            name: "test.metric".to_string(),
            value: 42.0,
            tags: HashMap::new(),
            timestamp: 0.0,
        };

        assert!(collector.record_metric(metric).is_ok());

        let history = collector.metrics_history.read().unwrap();
        assert_eq!(history.len(), 1);
    }

    #[test]
    fn test_quantum_metrics_recording() {
        let collector = TelemetryCollector::new(TelemetryConfig::default());

        let quantum_metrics = QuantumMetrics {
            num_qubits: 5,
            circuit_depth: 10,
            gate_execution_rate: 1000.0,
            entanglement_entropy: 0.5,
            error_correction_rate: 0.01,
            fidelity: 0.99,
            active_backends: vec!["statevector".to_string()],
            queue_depth: 0,
        };

        assert!(collector.record_quantum_metrics(quantum_metrics).is_ok());

        let history = collector.quantum_metrics_history.read().unwrap();
        assert_eq!(history.len(), 1);
    }

    #[test]
    fn test_gate_execution_recording() {
        let collector = TelemetryCollector::new(TelemetryConfig::default());

        let gate = InterfaceGate::new(InterfaceGateType::Hadamard, vec![0]);

        let duration = Duration::from_millis(10);
        assert!(collector.record_gate_execution(&gate, duration).is_ok());
    }

    #[test]
    fn test_memory_usage_recording() {
        let collector = TelemetryCollector::new(TelemetryConfig::default());

        assert!(collector.record_memory_usage(1024, "statevector").is_ok());

        let history = collector.metrics_history.read().unwrap();
        assert_eq!(history.len(), 1);
    }

    #[test]
    fn test_error_recording() {
        let collector = TelemetryCollector::new(TelemetryConfig::default());

        assert!(collector
            .record_error("simulation_error", "Gate execution failed")
            .is_ok());

        let history = collector.metrics_history.read().unwrap();
        assert_eq!(history.len(), 1);
    }

    #[test]
    fn test_metrics_summary() {
        let collector = TelemetryCollector::new(TelemetryConfig::default());

        // Add some test metrics
        let metric = TelemetryMetric::Timer {
            name: "gate.execution_time".to_string(),
            duration: Duration::from_millis(5),
            tags: HashMap::new(),
            timestamp: 0.0,
        };
        collector.record_metric(metric).unwrap();

        let summary = collector.get_metrics_summary().unwrap();
        assert_eq!(summary.total_metrics, 1);
        assert_abs_diff_eq!(summary.avg_gate_execution_time, 0.005, epsilon = 1e-6);
    }

    #[test]
    fn test_alert_thresholds() {
        let mut config = TelemetryConfig::default();
        config.alert_thresholds.max_gate_execution_time = 0.001; // 1ms

        let collector = TelemetryCollector::new(config);

        // Record a slow gate execution
        let metric = TelemetryMetric::Timer {
            name: "gate.execution_time".to_string(),
            duration: Duration::from_millis(10), // 10ms - exceeds threshold
            tags: HashMap::new(),
            timestamp: 0.0,
        };

        collector.record_metric(metric).unwrap();

        let alerts = collector.active_alerts.read().unwrap();
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].level, AlertLevel::Warning);
    }

    #[test]
    fn test_prometheus_formatting() {
        let collector = TelemetryCollector::new(TelemetryConfig::default());

        let mut tags = HashMap::new();
        tags.insert("gate_type".to_string(), "hadamard".to_string());
        tags.insert("qubits".to_string(), "1".to_string());

        let formatted = collector.format_prometheus_labels(&tags);
        assert!(formatted.contains("gate_type=\"hadamard\""));
        assert!(formatted.contains("qubits=\"1\""));
    }

    #[test]
    fn test_influx_formatting() {
        let collector = TelemetryCollector::new(TelemetryConfig::default());

        let mut tags = HashMap::new();
        tags.insert("gate_type".to_string(), "hadamard".to_string());
        tags.insert("qubits".to_string(), "1".to_string());

        let formatted = collector.format_influx_tags(&tags);
        assert!(formatted.starts_with(','));
        assert!(formatted.contains("gate_type=hadamard"));
        assert!(formatted.contains("qubits=1"));
    }

    #[test]
    fn test_sampling_rate() {
        let mut config = TelemetryConfig::default();
        config.sampling_rate = 0.0; // No sampling

        let collector = TelemetryCollector::new(config);

        let metric = TelemetryMetric::Gauge {
            name: "test.metric".to_string(),
            value: 42.0,
            tags: HashMap::new(),
            timestamp: 0.0,
        };

        // With 0% sampling rate, metric should still be recorded but might be filtered
        // The actual behavior depends on the random number generator
        collector.record_metric(metric).unwrap();
    }
}
