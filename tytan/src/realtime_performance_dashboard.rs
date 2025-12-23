//! Real-time Performance Dashboard
//!
//! This module provides real-time monitoring and visualization of quantum annealing
//! performance, enabling live tracking of convergence, resource utilization, and
//! comparative solver performance.
//!
//! # Features
//!
//! - **Live Convergence Visualization**: Real-time tracking of optimization progress
//! - **Resource Utilization Monitoring**: CPU, memory, and quantum resource tracking
//! - **Quality Metrics Streaming**: Continuous monitoring of solution quality
//! - **Comparative Sampler Performance**: Side-by-side performance comparison
//! - **Interactive Parameter Tuning**: Dynamic adjustment of algorithm parameters
//!
//! # Example
//!
//! ```rust
//! use quantrs2_tytan::realtime_performance_dashboard::{
//!     PerformanceDashboard, DashboardConfig, MetricUpdate
//! };
//!
//! // Create dashboard
//! let config = DashboardConfig::default();
//! let mut dashboard = PerformanceDashboard::new(config);
//!
//! // Start monitoring
//! dashboard.start_monitoring();
//!
//! // Update metrics during optimization
//! let metric = MetricUpdate {
//!     iteration: 100,
//!     energy: -15.5,
//!     best_energy: -18.2,
//!     runtime: 0.5,
//!     memory_usage: 128.0,
//! };
//! dashboard.update_metrics(metric);
//!
//! // Get performance summary
//! let summary = dashboard.get_summary();
//! println!("Best energy: {}", summary.best_energy);
//! ```

use scirs2_core::ndarray::Array1;
use std::collections::VecDeque;
use std::fmt;
use std::time::{Duration, Instant};

/// Error types for dashboard operations
#[derive(Debug, Clone)]
pub enum DashboardError {
    /// Dashboard not started
    NotStarted,
    /// Invalid metric update
    InvalidMetric(String),
    /// Export failed
    ExportFailed(String),
}

impl fmt::Display for DashboardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotStarted => write!(f, "Dashboard not started"),
            Self::InvalidMetric(msg) => write!(f, "Invalid metric: {msg}"),
            Self::ExportFailed(msg) => write!(f, "Export failed: {msg}"),
        }
    }
}

impl std::error::Error for DashboardError {}

/// Result type for dashboard operations
pub type DashboardResult<T> = Result<T, DashboardError>;

/// Configuration for performance dashboard
#[derive(Debug, Clone)]
pub struct DashboardConfig {
    /// Maximum number of data points to retain
    pub max_history_size: usize,
    /// Update interval for metrics (milliseconds)
    pub update_interval_ms: u64,
    /// Enable detailed resource tracking
    pub detailed_tracking: bool,
    /// Enable comparative mode (track multiple solvers)
    pub comparative_mode: bool,
    /// Number of solvers to compare
    pub num_solvers: usize,
    /// Enable automatic export
    pub auto_export: bool,
    /// Export interval (iterations)
    pub export_interval: usize,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            max_history_size: 10000,
            update_interval_ms: 100,
            detailed_tracking: true,
            comparative_mode: false,
            num_solvers: 1,
            auto_export: false,
            export_interval: 1000,
        }
    }
}

impl DashboardConfig {
    /// Set max history size
    pub const fn with_max_history_size(mut self, size: usize) -> Self {
        self.max_history_size = size;
        self
    }

    /// Set update interval
    pub const fn with_update_interval(mut self, ms: u64) -> Self {
        self.update_interval_ms = ms;
        self
    }

    /// Enable comparative mode
    pub const fn with_comparative_mode(mut self, num_solvers: usize) -> Self {
        self.comparative_mode = true;
        self.num_solvers = num_solvers;
        self
    }

    /// Enable auto export
    pub const fn with_auto_export(mut self, interval: usize) -> Self {
        self.auto_export = true;
        self.export_interval = interval;
        self
    }
}

/// Metric update for a single iteration
#[derive(Debug, Clone)]
pub struct MetricUpdate {
    /// Iteration number
    pub iteration: usize,
    /// Current energy
    pub energy: f64,
    /// Best energy so far
    pub best_energy: f64,
    /// Runtime (seconds)
    pub runtime: f64,
    /// Memory usage (MB)
    pub memory_usage: f64,
}

/// Resource utilization metrics
#[derive(Debug, Clone)]
pub struct ResourceMetrics {
    /// CPU usage percentage (0-100)
    pub cpu_usage: f64,
    /// Memory usage (MB)
    pub memory_mb: f64,
    /// GPU usage percentage (if applicable, 0-100)
    pub gpu_usage: Option<f64>,
    /// Quantum resource usage (if applicable, 0-100)
    pub quantum_usage: Option<f64>,
    /// Network I/O (MB/s)
    pub network_io: f64,
}

/// Quality metrics for solution
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Solution energy
    pub energy: f64,
    /// Constraint violations
    pub violations: usize,
    /// Feasibility score (0-1)
    pub feasibility: f64,
    /// Diversity score (0-1)
    pub diversity: f64,
    /// Convergence rate
    pub convergence_rate: f64,
}

/// Comparative performance data
#[derive(Debug, Clone)]
pub struct ComparativeData {
    /// Solver identifier
    pub solver_id: String,
    /// Current best energy
    pub best_energy: f64,
    /// Total runtime
    pub runtime: f64,
    /// Iterations completed
    pub iterations: usize,
    /// Solution quality score
    pub quality_score: f64,
}

/// Performance summary
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    /// Best energy achieved
    pub best_energy: f64,
    /// Total iterations
    pub total_iterations: usize,
    /// Total runtime (seconds)
    pub total_runtime: f64,
    /// Average energy improvement per iteration
    pub avg_improvement: f64,
    /// Convergence achieved
    pub converged: bool,
    /// Final quality metrics
    pub final_quality: QualityMetrics,
    /// Resource utilization summary
    pub resource_summary: ResourceSummary,
}

/// Resource utilization summary
#[derive(Debug, Clone)]
pub struct ResourceSummary {
    /// Average CPU usage
    pub avg_cpu: f64,
    /// Peak memory usage (MB)
    pub peak_memory: f64,
    /// Average GPU usage (if applicable)
    pub avg_gpu: Option<f64>,
    /// Total energy consumed (arbitrary units)
    pub total_energy: f64,
}

/// Real-time performance dashboard
pub struct PerformanceDashboard {
    config: DashboardConfig,
    /// Start time of monitoring
    start_time: Option<Instant>,
    /// Metric history
    metric_history: VecDeque<MetricUpdate>,
    /// Resource history
    resource_history: VecDeque<ResourceMetrics>,
    /// Quality history
    quality_history: VecDeque<QualityMetrics>,
    /// Comparative data (for multi-solver mode)
    comparative_data: Vec<ComparativeData>,
    /// Best energy seen
    best_energy: f64,
    /// Last update time
    last_update: Option<Instant>,
    /// Export counter
    export_counter: usize,
}

impl PerformanceDashboard {
    /// Create a new performance dashboard
    pub fn new(config: DashboardConfig) -> Self {
        let comparative_data = if config.comparative_mode {
            (0..config.num_solvers)
                .map(|i| ComparativeData {
                    solver_id: format!("Solver {i}"),
                    best_energy: f64::INFINITY,
                    runtime: 0.0,
                    iterations: 0,
                    quality_score: 0.0,
                })
                .collect()
        } else {
            vec![]
        };

        Self {
            config,
            start_time: None,
            metric_history: VecDeque::with_capacity(10000),
            resource_history: VecDeque::with_capacity(10000),
            quality_history: VecDeque::with_capacity(10000),
            comparative_data,
            best_energy: f64::INFINITY,
            last_update: None,
            export_counter: 0,
        }
    }

    /// Start monitoring
    pub fn start_monitoring(&mut self) {
        self.start_time = Some(Instant::now());
        self.last_update = Some(Instant::now());
        self.metric_history.clear();
        self.resource_history.clear();
        self.quality_history.clear();
        self.best_energy = f64::INFINITY;
        self.export_counter = 0;
    }

    /// Stop monitoring
    pub const fn stop_monitoring(&mut self) {
        self.start_time = None;
    }

    /// Update metrics
    pub fn update_metrics(&mut self, metric: MetricUpdate) -> DashboardResult<()> {
        if self.start_time.is_none() {
            return Err(DashboardError::NotStarted);
        }

        // Update best energy
        if metric.best_energy < self.best_energy {
            self.best_energy = metric.best_energy;
        }

        // Add to history
        self.metric_history.push_back(metric);
        if self.metric_history.len() > self.config.max_history_size {
            self.metric_history.pop_front();
        }

        // Update timestamp
        self.last_update = Some(Instant::now());

        // Auto export if enabled
        if self.config.auto_export {
            self.export_counter += 1;
            if self.export_counter % self.config.export_interval == 0 {
                self.export_data()?;
            }
        }

        Ok(())
    }

    /// Update resource metrics
    pub fn update_resources(&mut self, resources: ResourceMetrics) -> DashboardResult<()> {
        if self.start_time.is_none() {
            return Err(DashboardError::NotStarted);
        }

        self.resource_history.push_back(resources);
        if self.resource_history.len() > self.config.max_history_size {
            self.resource_history.pop_front();
        }

        Ok(())
    }

    /// Update quality metrics
    pub fn update_quality(&mut self, quality: QualityMetrics) -> DashboardResult<()> {
        if self.start_time.is_none() {
            return Err(DashboardError::NotStarted);
        }

        self.quality_history.push_back(quality);
        if self.quality_history.len() > self.config.max_history_size {
            self.quality_history.pop_front();
        }

        Ok(())
    }

    /// Update comparative data
    pub fn update_comparative(
        &mut self,
        solver_idx: usize,
        data: ComparativeData,
    ) -> DashboardResult<()> {
        if !self.config.comparative_mode {
            return Err(DashboardError::InvalidMetric(
                "Comparative mode not enabled".to_string(),
            ));
        }

        if solver_idx >= self.comparative_data.len() {
            return Err(DashboardError::InvalidMetric(format!(
                "Solver index {solver_idx} out of range"
            )));
        }

        self.comparative_data[solver_idx] = data;
        Ok(())
    }

    /// Get convergence data
    pub fn get_convergence_data(&self) -> Vec<(usize, f64)> {
        self.metric_history
            .iter()
            .map(|m| (m.iteration, m.best_energy))
            .collect()
    }

    /// Get resource utilization data
    pub fn get_resource_data(&self) -> Vec<ResourceMetrics> {
        self.resource_history.iter().cloned().collect()
    }

    /// Get quality metrics data
    pub fn get_quality_data(&self) -> Vec<QualityMetrics> {
        self.quality_history.iter().cloned().collect()
    }

    /// Get comparative data
    pub fn get_comparative_data(&self) -> &[ComparativeData] {
        &self.comparative_data
    }

    /// Get performance summary
    pub fn get_summary(&self) -> PerformanceSummary {
        let total_iterations = self.metric_history.back().map_or(0, |m| m.iteration);

        let total_runtime = if let Some(start) = self.start_time {
            start.elapsed().as_secs_f64()
        } else {
            0.0
        };

        // Calculate average improvement
        let avg_improvement = if self.metric_history.len() > 1 {
            let first_energy = self.metric_history.front().map_or(0.0, |m| m.energy);
            let last_energy = self.metric_history.back().map_or(0.0, |m| m.energy);
            let delta = first_energy - last_energy;
            delta / total_iterations as f64
        } else {
            0.0
        };

        // Check convergence
        let converged = if self.metric_history.len() > 10 {
            let recent: Vec<f64> = self
                .metric_history
                .iter()
                .rev()
                .take(10)
                .map(|m| m.best_energy)
                .collect();

            let variance = Self::compute_variance(&recent);
            variance < 1e-6
        } else {
            false
        };

        // Final quality metrics
        let final_quality = self
            .quality_history
            .back()
            .cloned()
            .unwrap_or(QualityMetrics {
                energy: self.best_energy,
                violations: 0,
                feasibility: 1.0,
                diversity: 0.5,
                convergence_rate: 0.0,
            });

        // Resource summary
        let resource_summary = self.compute_resource_summary();

        PerformanceSummary {
            best_energy: self.best_energy,
            total_iterations,
            total_runtime,
            avg_improvement,
            converged,
            final_quality,
            resource_summary,
        }
    }

    /// Compute resource summary
    fn compute_resource_summary(&self) -> ResourceSummary {
        if self.resource_history.is_empty() {
            return ResourceSummary {
                avg_cpu: 0.0,
                peak_memory: 0.0,
                avg_gpu: None,
                total_energy: 0.0,
            };
        }

        let avg_cpu = self
            .resource_history
            .iter()
            .map(|r| r.cpu_usage)
            .sum::<f64>()
            / self.resource_history.len() as f64;

        let peak_memory = self
            .resource_history
            .iter()
            .map(|r| r.memory_mb)
            .fold(0.0_f64, |acc, x| acc.max(x));

        let avg_gpu = {
            let gpu_values: Vec<f64> = self
                .resource_history
                .iter()
                .filter_map(|r| r.gpu_usage)
                .collect();

            if gpu_values.is_empty() {
                None
            } else {
                Some(gpu_values.iter().sum::<f64>() / gpu_values.len() as f64)
            }
        };

        // Simplified energy calculation (CPU + GPU usage over time)
        let total_energy = self
            .resource_history
            .iter()
            .map(|r| {
                r.cpu_usage.mul_add(0.1, r.gpu_usage.unwrap_or(0.0) * 0.3) // Arbitrary scaling
            })
            .sum::<f64>();

        ResourceSummary {
            avg_cpu,
            peak_memory,
            avg_gpu,
            total_energy,
        }
    }

    /// Compute variance of a sequence
    fn compute_variance(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance
    }

    /// Export data (simplified - in practice would export to file/database)
    const fn export_data(&self) -> DashboardResult<()> {
        // Placeholder for export functionality
        // In practice, this would write to CSV, JSON, or database
        Ok(())
    }

    /// Get elapsed time since start
    pub fn elapsed_time(&self) -> Option<Duration> {
        self.start_time.map(|t| t.elapsed())
    }

    /// Get current iteration rate (iterations per second)
    pub fn iteration_rate(&self) -> f64 {
        if let Some(elapsed) = self.elapsed_time() {
            let total_iterations = self.metric_history.back().map_or(0, |m| m.iteration);
            total_iterations as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Get energy improvement rate (energy units per second)
    pub fn energy_improvement_rate(&self) -> f64 {
        if let Some(elapsed) = self.elapsed_time() {
            let first_energy = self.metric_history.front().map_or(0.0, |m| m.energy);
            let improvement = first_energy - self.best_energy;
            improvement / elapsed.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Get configuration
    pub const fn config(&self) -> &DashboardConfig {
        &self.config
    }

    /// Check if monitoring is active
    pub const fn is_monitoring(&self) -> bool {
        self.start_time.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_creation() {
        let config = DashboardConfig::default();
        let dashboard = PerformanceDashboard::new(config);
        assert!(!dashboard.is_monitoring());
    }

    #[test]
    fn test_start_stop_monitoring() {
        let config = DashboardConfig::default();
        let mut dashboard = PerformanceDashboard::new(config);

        assert!(!dashboard.is_monitoring());

        dashboard.start_monitoring();
        assert!(dashboard.is_monitoring());

        dashboard.stop_monitoring();
        assert!(!dashboard.is_monitoring());
    }

    #[test]
    fn test_metric_update() {
        let config = DashboardConfig::default();
        let mut dashboard = PerformanceDashboard::new(config);

        dashboard.start_monitoring();

        let metric = MetricUpdate {
            iteration: 1,
            energy: -10.0,
            best_energy: -10.0,
            runtime: 0.1,
            memory_usage: 64.0,
        };

        let result = dashboard.update_metrics(metric);
        assert!(result.is_ok());

        let convergence = dashboard.get_convergence_data();
        assert_eq!(convergence.len(), 1);
        assert_eq!(convergence[0].0, 1);
        assert_eq!(convergence[0].1, -10.0);
    }

    #[test]
    fn test_resource_update() {
        let config = DashboardConfig::default();
        let mut dashboard = PerformanceDashboard::new(config);

        dashboard.start_monitoring();

        let resources = ResourceMetrics {
            cpu_usage: 75.0,
            memory_mb: 512.0,
            gpu_usage: Some(50.0),
            quantum_usage: None,
            network_io: 10.5,
        };

        let result = dashboard.update_resources(resources);
        assert!(result.is_ok());

        let data = dashboard.get_resource_data();
        assert_eq!(data.len(), 1);
        assert_eq!(data[0].cpu_usage, 75.0);
    }

    #[test]
    fn test_quality_update() {
        let config = DashboardConfig::default();
        let mut dashboard = PerformanceDashboard::new(config);

        dashboard.start_monitoring();

        let quality = QualityMetrics {
            energy: -15.5,
            violations: 0,
            feasibility: 1.0,
            diversity: 0.8,
            convergence_rate: 0.05,
        };

        let result = dashboard.update_quality(quality);
        assert!(result.is_ok());

        let data = dashboard.get_quality_data();
        assert_eq!(data.len(), 1);
        assert_eq!(data[0].energy, -15.5);
    }

    #[test]
    fn test_comparative_mode() {
        let config = DashboardConfig::default().with_comparative_mode(3);
        let mut dashboard = PerformanceDashboard::new(config);

        dashboard.start_monitoring();

        let data = ComparativeData {
            solver_id: "Test Solver".to_string(),
            best_energy: -20.0,
            runtime: 1.5,
            iterations: 1000,
            quality_score: 0.95,
        };

        let result = dashboard.update_comparative(0, data);
        assert!(result.is_ok());

        let comp_data = dashboard.get_comparative_data();
        assert_eq!(comp_data.len(), 3);
        assert_eq!(comp_data[0].best_energy, -20.0);
    }

    #[test]
    fn test_performance_summary() {
        let config = DashboardConfig::default();
        let mut dashboard = PerformanceDashboard::new(config);

        dashboard.start_monitoring();

        // Add some metrics
        for i in 1..=10 {
            let metric = MetricUpdate {
                iteration: i,
                energy: -5.0 * i as f64,
                best_energy: -5.0 * i as f64,
                runtime: 0.01 * i as f64,
                memory_usage: 100.0,
            };
            dashboard.update_metrics(metric).unwrap();
        }

        let summary = dashboard.get_summary();
        assert_eq!(summary.total_iterations, 10);
        assert_eq!(summary.best_energy, -50.0);
        assert!(summary.avg_improvement != 0.0);
    }

    #[test]
    fn test_config_builder() {
        let config = DashboardConfig::default()
            .with_max_history_size(5000)
            .with_update_interval(50)
            .with_comparative_mode(2)
            .with_auto_export(500);

        assert_eq!(config.max_history_size, 5000);
        assert_eq!(config.update_interval_ms, 50);
        assert!(config.comparative_mode);
        assert_eq!(config.num_solvers, 2);
        assert!(config.auto_export);
        assert_eq!(config.export_interval, 500);
    }

    #[test]
    fn test_convergence_detection() {
        let config = DashboardConfig::default();
        let mut dashboard = PerformanceDashboard::new(config);

        dashboard.start_monitoring();

        // Add converged metrics
        for i in 1..=20 {
            let metric = MetricUpdate {
                iteration: i,
                energy: -10.0,
                best_energy: -10.0,
                runtime: 0.01 * i as f64,
                memory_usage: 100.0,
            };
            dashboard.update_metrics(metric).unwrap();
        }

        let summary = dashboard.get_summary();
        assert!(summary.converged);
    }

    #[test]
    fn test_resource_summary() {
        let config = DashboardConfig::default();
        let mut dashboard = PerformanceDashboard::new(config);

        dashboard.start_monitoring();

        // Add resource metrics
        for _ in 0..10 {
            let resources = ResourceMetrics {
                cpu_usage: 80.0,
                memory_mb: 256.0,
                gpu_usage: Some(60.0),
                quantum_usage: None,
                network_io: 5.0,
            };
            dashboard.update_resources(resources).unwrap();
        }

        let summary = dashboard.get_summary();
        assert_eq!(summary.resource_summary.avg_cpu, 80.0);
        assert_eq!(summary.resource_summary.peak_memory, 256.0);
        assert_eq!(summary.resource_summary.avg_gpu, Some(60.0));
    }
}
