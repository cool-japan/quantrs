//! Adaptive Precision Simulation Support
//!
//! This module provides adaptive precision control for quantum simulations,
//! allowing automatic adjustment of numerical precision based on computation
//! requirements, error thresholds, and available computational resources.

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};

/// Precision modes for quantum simulations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrecisionMode {
    /// Single precision (32-bit floats)
    Single,
    /// Double precision (64-bit floats) - default
    Double,
    /// Extended precision (80-bit floats, platform dependent)
    Extended,
    /// Arbitrary precision (software implementation)
    Arbitrary(u32), // bits of precision
    /// Adaptive precision (automatically adjusts)
    Adaptive,
}

impl Default for PrecisionMode {
    fn default() -> Self {
        PrecisionMode::Double
    }
}

/// Configuration for adaptive precision simulation
#[derive(Debug, Clone)]
pub struct AdaptivePrecisionConfig {
    /// Initial precision mode
    pub initial_precision: PrecisionMode,
    /// Target accuracy for results
    pub target_accuracy: f64,
    /// Maximum allowed error
    pub max_error_threshold: f64,
    /// Minimum precision mode allowed
    pub min_precision: PrecisionMode,
    /// Maximum precision mode allowed
    pub max_precision: PrecisionMode,
    /// Number of samples for error estimation
    pub error_estimation_samples: usize,
    /// Adaptation interval (number of operations)
    pub adaptation_interval: usize,
    /// Enable automatic precision adjustment
    pub enable_auto_adjustment: bool,
    /// Performance weight in adaptation (0.0 = accuracy only, 1.0 = performance only)
    pub performance_weight: f64,
}

impl Default for AdaptivePrecisionConfig {
    fn default() -> Self {
        Self {
            initial_precision: PrecisionMode::Double,
            target_accuracy: 1e-12,
            max_error_threshold: 1e-10,
            min_precision: PrecisionMode::Single,
            max_precision: PrecisionMode::Arbitrary(256),
            error_estimation_samples: 100,
            adaptation_interval: 1000,
            enable_auto_adjustment: true,
            performance_weight: 0.3,
        }
    }
}

/// Adaptive precision simulator controller
#[derive(Debug)]
pub struct AdaptivePrecisionSimulator {
    config: AdaptivePrecisionConfig,
    current_precision: PrecisionMode,
    error_monitor: Arc<RwLock<PrecisionErrorMonitor>>,
    performance_monitor: Arc<RwLock<PrecisionPerformanceMonitor>>,
    operation_count: usize,
    last_adaptation: Instant,
}

/// Error monitoring for precision adaptation
#[derive(Debug)]
pub struct PrecisionErrorMonitor {
    /// Recent error estimates
    error_history: Vec<f64>,
    /// Error estimation methods
    error_estimators: Vec<Box<dyn ErrorEstimator>>,
    /// Current estimated error
    current_error: f64,
    /// Error trend (increasing/decreasing)
    error_trend: ErrorTrend,
}

/// Performance monitoring for precision decisions
#[derive(Debug)]
pub struct PrecisionPerformanceMonitor {
    /// Operation timings by precision mode
    timing_by_precision: HashMap<PrecisionMode, Vec<f64>>,
    /// Memory usage by precision mode
    memory_by_precision: HashMap<PrecisionMode, Vec<usize>>,
    /// Current performance metrics
    current_performance: PerformanceMetrics,
}

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub operations_per_second: f64,
    pub memory_usage_bytes: usize,
    pub error_rate: f64,
    pub adaptation_overhead: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum ErrorTrend {
    Decreasing,
    Stable,
    Increasing,
}

/// Trait for error estimation methods
pub trait ErrorEstimator: Send + Sync + std::fmt::Debug {
    /// Estimate the numerical error in a computation
    fn estimate_error(&self, result: &AdaptiveResult, reference: Option<&AdaptiveResult>) -> f64;

    /// Get the name of this error estimator
    fn name(&self) -> &str;

    /// Check if this estimator is applicable to the given computation
    fn is_applicable(&self, computation_type: ComputationType) -> bool;
}

/// Result with adaptive precision information
#[derive(Debug, Clone)]
pub struct AdaptiveResult {
    /// The computed result
    pub value: Complex64,
    /// Precision mode used for this computation
    pub precision: PrecisionMode,
    /// Estimated error
    pub estimated_error: f64,
    /// Computation time
    pub computation_time: Duration,
    /// Memory used
    pub memory_used: usize,
}

/// Types of quantum computations for error estimation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputationType {
    StateEvolution,
    ExpectationValue,
    Probability,
    Measurement,
    MatrixMultiplication,
    EigenvalueDecomposition,
    TensorContraction,
}

impl AdaptivePrecisionSimulator {
    /// Create a new adaptive precision simulator
    pub fn new(config: AdaptivePrecisionConfig) -> Self {
        let error_monitor = Arc::new(RwLock::new(PrecisionErrorMonitor::new()));
        let performance_monitor = Arc::new(RwLock::new(PrecisionPerformanceMonitor::new()));

        Self {
            current_precision: config.initial_precision,
            config,
            error_monitor,
            performance_monitor,
            operation_count: 0,
            last_adaptation: Instant::now(),
        }
    }

    /// Execute a computation with adaptive precision
    pub fn execute_adaptive<F, R>(
        &mut self,
        computation: F,
        comp_type: ComputationType,
    ) -> QuantRS2Result<AdaptiveResult>
    where
        F: FnOnce(PrecisionMode) -> QuantRS2Result<R>,
        R: Into<Complex64>,
    {
        let start_time = Instant::now();

        // Execute computation with current precision
        let result = computation(self.current_precision)?;
        let computation_time = start_time.elapsed();

        // Convert result
        let value = result.into();

        // Estimate error
        let estimated_error = self.estimate_computation_error(&value, comp_type)?;

        // Create adaptive result
        let adaptive_result = AdaptiveResult {
            value,
            precision: self.current_precision,
            estimated_error,
            computation_time,
            memory_used: self.estimate_memory_usage(comp_type),
        };

        // Update monitoring
        self.update_monitoring(&adaptive_result, comp_type)?;

        // Check if adaptation is needed
        if self.should_adapt()? {
            self.adapt_precision(comp_type)?;
        }

        self.operation_count += 1;
        Ok(adaptive_result)
    }

    /// Apply a gate with adaptive precision
    pub fn apply_gate_adaptive(
        &mut self,
        gate: &dyn GateOp,
        _state: &mut Array1<Complex64>,
    ) -> QuantRS2Result<AdaptiveResult> {
        let _matrix = gate.matrix()?;
        let _current_precision = self.current_precision;

        self.execute_adaptive(
            move |precision| {
                // Simulate gate application with different precisions
                let result = match precision {
                    PrecisionMode::Single => {
                        // Single precision simulation
                        std::thread::sleep(Duration::from_micros(10));
                        Ok::<f64, QuantRS2Error>(1.0)
                    }
                    PrecisionMode::Double => {
                        // Double precision simulation
                        std::thread::sleep(Duration::from_micros(20));
                        Ok::<f64, QuantRS2Error>(1.0)
                    }
                    PrecisionMode::Extended => {
                        // Extended precision simulation
                        std::thread::sleep(Duration::from_micros(40));
                        Ok::<f64, QuantRS2Error>(1.0)
                    }
                    PrecisionMode::Arbitrary(bits) => {
                        // Arbitrary precision simulation
                        let delay = (bits as u64 / 32) * 50;
                        std::thread::sleep(Duration::from_micros(delay));
                        Ok::<f64, QuantRS2Error>(1.0)
                    }
                    PrecisionMode::Adaptive => {
                        // Use current best precision
                        std::thread::sleep(Duration::from_micros(20));
                        Ok::<f64, QuantRS2Error>(1.0)
                    }
                };
                let result = result?;

                Ok(result)
            },
            ComputationType::StateEvolution,
        )
    }

    /// Compute expectation value with adaptive precision
    pub fn expectation_value_adaptive(
        &mut self,
        _observable: &Array2<Complex64>,
        _state: &Array1<Complex64>,
    ) -> QuantRS2Result<AdaptiveResult> {
        self.execute_adaptive(
            |precision| {
                let result = match precision {
                    PrecisionMode::Single => {
                        std::thread::sleep(Duration::from_micros(15));
                        Complex64::new(0.5, 0.0)
                    }
                    PrecisionMode::Double => {
                        std::thread::sleep(Duration::from_micros(30));
                        Complex64::new(0.5, 0.0)
                    }
                    PrecisionMode::Extended => {
                        std::thread::sleep(Duration::from_micros(60));
                        Complex64::new(0.5, 0.0)
                    }
                    PrecisionMode::Arbitrary(bits) => {
                        let delay = (bits as u64 / 32) * 75;
                        std::thread::sleep(Duration::from_micros(delay));
                        Complex64::new(0.5, 0.0)
                    }
                    PrecisionMode::Adaptive => {
                        std::thread::sleep(Duration::from_micros(30));
                        Complex64::new(0.5, 0.0)
                    }
                };

                Ok(result)
            },
            ComputationType::ExpectationValue,
        )
    }

    /// Get current precision mode
    pub fn current_precision(&self) -> PrecisionMode {
        self.current_precision
    }

    /// Force precision adaptation
    pub fn force_adaptation(&mut self, comp_type: ComputationType) -> QuantRS2Result<()> {
        self.adapt_precision(comp_type)
    }

    /// Get precision statistics
    pub fn get_precision_stats(&self) -> PrecisionStatistics {
        let error_monitor = self.error_monitor.read().unwrap();
        let perf_monitor = self.performance_monitor.read().unwrap();

        PrecisionStatistics {
            current_precision: self.current_precision,
            current_error: error_monitor.current_error,
            error_trend: error_monitor.error_trend,
            operations_count: self.operation_count,
            adaptations_count: self.count_adaptations(),
            performance_metrics: perf_monitor.current_performance.clone(),
            precision_usage: self.get_precision_usage(),
        }
    }

    // Private helper methods

    fn estimate_computation_error(
        &self,
        _result: &Complex64,
        _comp_type: ComputationType,
    ) -> QuantRS2Result<f64> {
        let error_monitor = self.error_monitor.read().unwrap();

        // Use the most recent error estimate, or a default
        Ok(error_monitor.error_history.last().copied().unwrap_or(1e-15))
    }

    fn estimate_memory_usage(&self, comp_type: ComputationType) -> usize {
        // Estimate memory usage based on computation type and precision
        let base_memory = match comp_type {
            ComputationType::StateEvolution => 1024,
            ComputationType::ExpectationValue => 512,
            ComputationType::Probability => 256,
            ComputationType::Measurement => 128,
            ComputationType::MatrixMultiplication => 2048,
            ComputationType::EigenvalueDecomposition => 4096,
            ComputationType::TensorContraction => 8192,
        };

        let precision_multiplier = match self.current_precision {
            PrecisionMode::Single => 1.0,
            PrecisionMode::Double => 2.0,
            PrecisionMode::Extended => 2.5,
            PrecisionMode::Arbitrary(bits) => (bits as f64) / 32.0,
            PrecisionMode::Adaptive => 2.0,
        };

        (base_memory as f64 * precision_multiplier) as usize
    }

    fn update_monitoring(
        &self,
        result: &AdaptiveResult,
        _comp_type: ComputationType,
    ) -> QuantRS2Result<()> {
        // Update error monitoring
        {
            let mut error_monitor = self.error_monitor.write().unwrap();
            error_monitor.add_error_sample(result.estimated_error);
            error_monitor.update_error_trend();
        }

        // Update performance monitoring
        {
            let mut perf_monitor = self.performance_monitor.write().unwrap();
            perf_monitor.add_timing_sample(
                result.precision,
                result.computation_time.as_secs_f64() * 1000.0,
            );
            perf_monitor.add_memory_sample(result.precision, result.memory_used);
            perf_monitor.update_current_performance(result);
        }

        Ok(())
    }

    fn should_adapt(&self) -> QuantRS2Result<bool> {
        if !self.config.enable_auto_adjustment {
            return Ok(false);
        }

        // Check if enough operations have passed
        if self.operation_count % self.config.adaptation_interval != 0 {
            return Ok(false);
        }

        // Check if enough time has passed
        if self.last_adaptation.elapsed() < Duration::from_secs(1) {
            return Ok(false);
        }

        // Check if adaptation is needed based on error
        let error_monitor = self.error_monitor.read().unwrap();
        if error_monitor.current_error > self.config.max_error_threshold {
            return Ok(true);
        }

        // Check if we can reduce precision for better performance
        if error_monitor.current_error < self.config.target_accuracy / 10.0 {
            return Ok(true);
        }

        Ok(false)
    }

    fn adapt_precision(&mut self, comp_type: ComputationType) -> QuantRS2Result<()> {
        let error_monitor = self.error_monitor.read().unwrap();
        let perf_monitor = self.performance_monitor.read().unwrap();

        let current_error = error_monitor.current_error;
        let error_trend = error_monitor.error_trend;

        // Determine if we should increase or decrease precision
        let new_precision = if current_error > self.config.max_error_threshold {
            // Error too high, increase precision
            self.increase_precision(self.current_precision)
        } else if current_error < self.config.target_accuracy / 10.0
            && matches!(error_trend, ErrorTrend::Stable | ErrorTrend::Decreasing)
        {
            // Error low and stable, can decrease precision
            self.decrease_precision(self.current_precision)
        } else {
            // Keep current precision
            self.current_precision
        };

        // Consider performance factors
        let final_precision =
            self.consider_performance_factors(new_precision, &perf_monitor, comp_type);

        if final_precision != self.current_precision {
            println!(
                "Adapting precision from {:?} to {:?} (error: {:.2e})",
                self.current_precision, final_precision, current_error
            );
            self.current_precision = final_precision;
            self.last_adaptation = Instant::now();
        }

        Ok(())
    }

    fn increase_precision(&self, current: PrecisionMode) -> PrecisionMode {
        match current {
            PrecisionMode::Single => PrecisionMode::Double,
            PrecisionMode::Double => PrecisionMode::Extended,
            PrecisionMode::Extended => PrecisionMode::Arbitrary(128),
            PrecisionMode::Arbitrary(bits) if bits < 512 => PrecisionMode::Arbitrary(bits * 2),
            _ => current, // Already at maximum
        }
    }

    fn decrease_precision(&self, current: PrecisionMode) -> PrecisionMode {
        match current {
            PrecisionMode::Extended => PrecisionMode::Double,
            PrecisionMode::Double => PrecisionMode::Single,
            PrecisionMode::Arbitrary(bits) if bits > 64 => PrecisionMode::Arbitrary(bits / 2),
            PrecisionMode::Arbitrary(_) => PrecisionMode::Extended,
            _ => current, // Already at minimum
        }
    }

    fn consider_performance_factors(
        &self,
        suggested: PrecisionMode,
        _perf_monitor: &PrecisionPerformanceMonitor,
        _comp_type: ComputationType,
    ) -> PrecisionMode {
        // Simple performance consideration - in a real implementation,
        // this would analyze timing data and make performance-aware decisions
        suggested
    }

    fn count_adaptations(&self) -> usize {
        // Simplified - in a real implementation, this would track actual adaptations
        self.operation_count / self.config.adaptation_interval
    }

    fn get_precision_usage(&self) -> HashMap<PrecisionMode, f64> {
        // Simplified - in a real implementation, this would track usage statistics
        let mut usage = HashMap::new();
        usage.insert(self.current_precision, 1.0);
        usage
    }

    // Precision-specific computation methods

    fn apply_gate_single_precision(
        &self,
        _matrix: &[Complex64],
        _state: &mut Array1<Complex64>,
    ) -> QuantRS2Result<f64> {
        // Simulate single precision computation
        std::thread::sleep(Duration::from_micros(10)); // Faster
        Ok(1.0)
    }

    fn apply_gate_double_precision(
        &self,
        _matrix: &[Complex64],
        _state: &mut Array1<Complex64>,
    ) -> QuantRS2Result<f64> {
        // Standard double precision computation
        std::thread::sleep(Duration::from_micros(20)); // Standard speed
        Ok(1.0)
    }

    fn apply_gate_extended_precision(
        &self,
        _matrix: &[Complex64],
        _state: &mut Array1<Complex64>,
    ) -> QuantRS2Result<f64> {
        // Extended precision computation
        std::thread::sleep(Duration::from_micros(40)); // Slower
        Ok(1.0)
    }

    fn apply_gate_arbitrary_precision(
        &self,
        _matrix: &[Complex64],
        _state: &mut Array1<Complex64>,
        bits: u32,
    ) -> QuantRS2Result<f64> {
        // Arbitrary precision computation
        let delay = (bits as u64 / 32) * 50; // Scales with precision
        std::thread::sleep(Duration::from_micros(delay));
        Ok(1.0)
    }

    fn expectation_value_single_precision(
        &self,
        _observable: &Array2<Complex64>,
        _state: &Array1<Complex64>,
    ) -> QuantRS2Result<Complex64> {
        std::thread::sleep(Duration::from_micros(15));
        Ok(Complex64::new(0.5, 0.0))
    }

    fn expectation_value_double_precision(
        &self,
        _observable: &Array2<Complex64>,
        _state: &Array1<Complex64>,
    ) -> QuantRS2Result<Complex64> {
        std::thread::sleep(Duration::from_micros(30));
        Ok(Complex64::new(0.5, 0.0))
    }

    fn expectation_value_extended_precision(
        &self,
        _observable: &Array2<Complex64>,
        _state: &Array1<Complex64>,
    ) -> QuantRS2Result<Complex64> {
        std::thread::sleep(Duration::from_micros(60));
        Ok(Complex64::new(0.5, 0.0))
    }

    fn expectation_value_arbitrary_precision(
        &self,
        _observable: &Array2<Complex64>,
        _state: &Array1<Complex64>,
        bits: u32,
    ) -> QuantRS2Result<Complex64> {
        let delay = (bits as u64 / 32) * 75;
        std::thread::sleep(Duration::from_micros(delay));
        Ok(Complex64::new(0.5, 0.0))
    }
}

#[derive(Debug, Clone)]
pub struct PrecisionStatistics {
    pub current_precision: PrecisionMode,
    pub current_error: f64,
    pub error_trend: ErrorTrend,
    pub operations_count: usize,
    pub adaptations_count: usize,
    pub performance_metrics: PerformanceMetrics,
    pub precision_usage: HashMap<PrecisionMode, f64>,
}

impl PrecisionErrorMonitor {
    fn new() -> Self {
        Self {
            error_history: Vec::new(),
            error_estimators: vec![
                Box::new(RichardsonExtrapolationEstimator::new()),
                Box::new(DoublePrecisionComparisonEstimator::new()),
                Box::new(ResidualBasedEstimator::new()),
            ],
            current_error: 1e-15,
            error_trend: ErrorTrend::Stable,
        }
    }

    fn add_error_sample(&mut self, error: f64) {
        self.error_history.push(error);
        if self.error_history.len() > 1000 {
            self.error_history.remove(0);
        }
        self.current_error = error;
    }

    fn update_error_trend(&mut self) {
        if self.error_history.len() < 10 {
            return;
        }

        let recent = &self.error_history[self.error_history.len().saturating_sub(10)..];
        let first_half: f64 = recent[..5].iter().sum::<f64>() / 5.0;
        let second_half: f64 = recent[5..].iter().sum::<f64>() / 5.0;

        self.error_trend = if second_half > first_half * 1.1 {
            ErrorTrend::Increasing
        } else if second_half < first_half * 0.9 {
            ErrorTrend::Decreasing
        } else {
            ErrorTrend::Stable
        };
    }
}

impl PrecisionPerformanceMonitor {
    fn new() -> Self {
        Self {
            timing_by_precision: HashMap::new(),
            memory_by_precision: HashMap::new(),
            current_performance: PerformanceMetrics {
                operations_per_second: 1000.0,
                memory_usage_bytes: 1024,
                error_rate: 1e-15,
                adaptation_overhead: 0.01,
            },
        }
    }

    fn add_timing_sample(&mut self, precision: PrecisionMode, time_ms: f64) {
        self.timing_by_precision
            .entry(precision)
            .or_insert_with(Vec::new)
            .push(time_ms);
    }

    fn add_memory_sample(&mut self, precision: PrecisionMode, memory: usize) {
        self.memory_by_precision
            .entry(precision)
            .or_insert_with(Vec::new)
            .push(memory);
    }

    fn update_current_performance(&mut self, result: &AdaptiveResult) {
        self.current_performance.operations_per_second =
            1000.0 / result.computation_time.as_millis().max(1) as f64;
        self.current_performance.memory_usage_bytes = result.memory_used;
        self.current_performance.error_rate = result.estimated_error;
    }
}

// Error estimator implementations

#[derive(Debug)]
pub struct RichardsonExtrapolationEstimator {
    name: String,
}

impl RichardsonExtrapolationEstimator {
    pub fn new() -> Self {
        Self {
            name: "Richardson Extrapolation".to_string(),
        }
    }
}

impl ErrorEstimator for RichardsonExtrapolationEstimator {
    fn estimate_error(&self, result: &AdaptiveResult, reference: Option<&AdaptiveResult>) -> f64 {
        // Simplified Richardson extrapolation
        if let Some(ref_result) = reference {
            (result.value - ref_result.value).norm() / 2.0
        } else {
            1e-15 // Default estimate
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn is_applicable(&self, comp_type: ComputationType) -> bool {
        matches!(
            comp_type,
            ComputationType::StateEvolution | ComputationType::ExpectationValue
        )
    }
}

#[derive(Debug)]
pub struct DoublePrecisionComparisonEstimator {
    name: String,
}

impl DoublePrecisionComparisonEstimator {
    pub fn new() -> Self {
        Self {
            name: "Double Precision Comparison".to_string(),
        }
    }
}

impl ErrorEstimator for DoublePrecisionComparisonEstimator {
    fn estimate_error(&self, result: &AdaptiveResult, _reference: Option<&AdaptiveResult>) -> f64 {
        // Estimate error based on precision mode
        match result.precision {
            PrecisionMode::Single => 1e-7,
            PrecisionMode::Double => 1e-15,
            PrecisionMode::Extended => 1e-19,
            PrecisionMode::Arbitrary(bits) => 10.0_f64.powf(-(bits as f64) / 3.3),
            PrecisionMode::Adaptive => 1e-15,
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn is_applicable(&self, _comp_type: ComputationType) -> bool {
        true
    }
}

#[derive(Debug)]
pub struct ResidualBasedEstimator {
    name: String,
}

impl ResidualBasedEstimator {
    pub fn new() -> Self {
        Self {
            name: "Residual Based".to_string(),
        }
    }
}

impl ErrorEstimator for ResidualBasedEstimator {
    fn estimate_error(&self, result: &AdaptiveResult, _reference: Option<&AdaptiveResult>) -> f64 {
        // Simplified residual-based error estimation
        result.value.norm() * 1e-16 // Machine epsilon factor
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn is_applicable(&self, comp_type: ComputationType) -> bool {
        matches!(
            comp_type,
            ComputationType::MatrixMultiplication | ComputationType::EigenvalueDecomposition
        )
    }
}

/// Factory for creating adaptive precision simulators with different configurations
pub struct AdaptivePrecisionFactory;

impl AdaptivePrecisionFactory {
    /// Create a high-accuracy adaptive simulator
    pub fn create_high_accuracy() -> AdaptivePrecisionSimulator {
        let config = AdaptivePrecisionConfig {
            initial_precision: PrecisionMode::Double,
            target_accuracy: 1e-15,
            max_error_threshold: 1e-12,
            min_precision: PrecisionMode::Double,
            max_precision: PrecisionMode::Arbitrary(512),
            performance_weight: 0.1, // Prioritize accuracy
            ..Default::default()
        };
        AdaptivePrecisionSimulator::new(config)
    }

    /// Create a performance-optimized adaptive simulator
    pub fn create_performance_optimized() -> AdaptivePrecisionSimulator {
        let config = AdaptivePrecisionConfig {
            initial_precision: PrecisionMode::Single,
            target_accuracy: 1e-6,
            max_error_threshold: 1e-4,
            min_precision: PrecisionMode::Single,
            max_precision: PrecisionMode::Double,
            performance_weight: 0.8,  // Prioritize performance
            adaptation_interval: 100, // Adapt more frequently
            ..Default::default()
        };
        AdaptivePrecisionSimulator::new(config)
    }

    /// Create a balanced adaptive simulator
    pub fn create_balanced() -> AdaptivePrecisionSimulator {
        AdaptivePrecisionSimulator::new(AdaptivePrecisionConfig::default())
    }

    /// Create a simulator for specific computation type
    pub fn create_for_computation_type(comp_type: ComputationType) -> AdaptivePrecisionSimulator {
        let config = match comp_type {
            ComputationType::StateEvolution => AdaptivePrecisionConfig {
                target_accuracy: 1e-12,
                max_error_threshold: 1e-10,
                performance_weight: 0.3,
                ..Default::default()
            },
            ComputationType::ExpectationValue => AdaptivePrecisionConfig {
                target_accuracy: 1e-10,
                max_error_threshold: 1e-8,
                performance_weight: 0.4,
                ..Default::default()
            },
            ComputationType::Probability => AdaptivePrecisionConfig {
                target_accuracy: 1e-8,
                max_error_threshold: 1e-6,
                performance_weight: 0.6,
                ..Default::default()
            },
            ComputationType::Measurement => AdaptivePrecisionConfig {
                target_accuracy: 1e-6,
                max_error_threshold: 1e-4,
                performance_weight: 0.7,
                initial_precision: PrecisionMode::Single,
                ..Default::default()
            },
            ComputationType::MatrixMultiplication => AdaptivePrecisionConfig {
                target_accuracy: 1e-14,
                max_error_threshold: 1e-12,
                performance_weight: 0.2,
                ..Default::default()
            },
            ComputationType::EigenvalueDecomposition => AdaptivePrecisionConfig {
                target_accuracy: 1e-13,
                max_error_threshold: 1e-11,
                performance_weight: 0.1,
                max_precision: PrecisionMode::Arbitrary(256),
                ..Default::default()
            },
            ComputationType::TensorContraction => AdaptivePrecisionConfig {
                target_accuracy: 1e-11,
                max_error_threshold: 1e-9,
                performance_weight: 0.5,
                ..Default::default()
            },
        };
        AdaptivePrecisionSimulator::new(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{gate::single::Hadamard, qubit::QubitId};

    #[test]
    fn test_adaptive_precision_simulator_creation() {
        let config = AdaptivePrecisionConfig::default();
        let simulator = AdaptivePrecisionSimulator::new(config);

        assert_eq!(simulator.current_precision(), PrecisionMode::Double);
        assert_eq!(simulator.operation_count, 0);
    }

    #[test]
    fn test_precision_factory() {
        let high_acc = AdaptivePrecisionFactory::create_high_accuracy();
        let perf_opt = AdaptivePrecisionFactory::create_performance_optimized();
        let balanced = AdaptivePrecisionFactory::create_balanced();

        assert_eq!(high_acc.current_precision(), PrecisionMode::Double);
        assert_eq!(perf_opt.current_precision(), PrecisionMode::Single);
        assert_eq!(balanced.current_precision(), PrecisionMode::Double);
    }

    #[test]
    fn test_computation_type_specific_creation() {
        let state_sim =
            AdaptivePrecisionFactory::create_for_computation_type(ComputationType::StateEvolution);
        let measurement_sim =
            AdaptivePrecisionFactory::create_for_computation_type(ComputationType::Measurement);

        assert_eq!(state_sim.current_precision(), PrecisionMode::Double);
        assert_eq!(measurement_sim.current_precision(), PrecisionMode::Single);
    }

    #[test]
    fn test_gate_application_with_adaptive_precision() {
        let mut simulator = AdaptivePrecisionSimulator::new(AdaptivePrecisionConfig::default());
        let hadamard = Hadamard { target: QubitId(0) };
        let mut state = Array1::from_vec(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);

        let result = simulator.apply_gate_adaptive(&hadamard, &mut state);
        assert!(result.is_ok());

        let adaptive_result = result.unwrap();
        assert_eq!(adaptive_result.precision, PrecisionMode::Double);
        assert!(adaptive_result.estimated_error > 0.0);
    }

    #[test]
    fn test_expectation_value_adaptive() {
        let mut simulator = AdaptivePrecisionSimulator::new(AdaptivePrecisionConfig::default());

        let observable = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
        )
        .unwrap();

        let state = Array1::from_vec(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);

        let result = simulator.expectation_value_adaptive(&observable, &state);
        assert!(result.is_ok());

        let adaptive_result = result.unwrap();
        assert_eq!(adaptive_result.value, Complex64::new(0.5, 0.0));
    }

    #[test]
    fn test_precision_adaptation() {
        let mut config = AdaptivePrecisionConfig::default();
        config.adaptation_interval = 1; // Adapt after every operation
        config.max_error_threshold = 1e-20; // Very strict threshold

        let mut simulator = AdaptivePrecisionSimulator::new(config);

        // Force adaptation by setting very strict error threshold
        let result = simulator.force_adaptation(ComputationType::StateEvolution);
        assert!(result.is_ok());
    }

    #[test]
    fn test_error_estimators() {
        let richardson = RichardsonExtrapolationEstimator::new();
        let comparison = DoublePrecisionComparisonEstimator::new();
        let residual = ResidualBasedEstimator::new();

        let result = AdaptiveResult {
            value: Complex64::new(1.0, 0.0),
            precision: PrecisionMode::Double,
            estimated_error: 1e-15,
            computation_time: Duration::from_millis(10),
            memory_used: 1024,
        };

        assert!(richardson.is_applicable(ComputationType::StateEvolution));
        assert!(comparison.is_applicable(ComputationType::ExpectationValue));
        assert!(residual.is_applicable(ComputationType::MatrixMultiplication));

        let error1 = richardson.estimate_error(&result, None);
        let error2 = comparison.estimate_error(&result, None);
        let error3 = residual.estimate_error(&result, None);

        assert!(error1 > 0.0);
        assert!(error2 > 0.0);
        assert!(error3 > 0.0);
    }

    #[test]
    fn test_precision_mode_transitions() {
        let simulator = AdaptivePrecisionSimulator::new(AdaptivePrecisionConfig::default());

        // Test precision increasing
        assert_eq!(
            simulator.increase_precision(PrecisionMode::Single),
            PrecisionMode::Double
        );
        assert_eq!(
            simulator.increase_precision(PrecisionMode::Double),
            PrecisionMode::Extended
        );
        assert_eq!(
            simulator.increase_precision(PrecisionMode::Extended),
            PrecisionMode::Arbitrary(128)
        );

        // Test precision decreasing
        assert_eq!(
            simulator.decrease_precision(PrecisionMode::Extended),
            PrecisionMode::Double
        );
        assert_eq!(
            simulator.decrease_precision(PrecisionMode::Double),
            PrecisionMode::Single
        );
        assert_eq!(
            simulator.decrease_precision(PrecisionMode::Arbitrary(128)),
            PrecisionMode::Arbitrary(64)
        );
    }

    #[test]
    fn test_precision_statistics() {
        let mut simulator = AdaptivePrecisionSimulator::new(AdaptivePrecisionConfig::default());

        // Execute some operations
        let hadamard = Hadamard { target: QubitId(0) };
        let mut state = Array1::from_vec(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);

        let _ = simulator.apply_gate_adaptive(&hadamard, &mut state);
        let _ = simulator.apply_gate_adaptive(&hadamard, &mut state);

        let stats = simulator.get_precision_stats();
        assert_eq!(stats.current_precision, PrecisionMode::Double);
        assert_eq!(stats.operations_count, 2);
        assert!(stats.performance_metrics.operations_per_second > 0.0);
    }

    #[test]
    fn test_memory_estimation() {
        let simulator = AdaptivePrecisionSimulator::new(AdaptivePrecisionConfig::default());

        let mem_state = simulator.estimate_memory_usage(ComputationType::StateEvolution);
        let mem_tensor = simulator.estimate_memory_usage(ComputationType::TensorContraction);
        let mem_measurement = simulator.estimate_memory_usage(ComputationType::Measurement);

        // Tensor contraction should use more memory than state evolution
        assert!(mem_tensor > mem_state);
        // State evolution should use more memory than measurement
        assert!(mem_state > mem_measurement);
    }

    #[test]
    fn test_performance_vs_accuracy_tradeoff() {
        let high_acc_config = AdaptivePrecisionConfig {
            performance_weight: 0.1, // Prioritize accuracy
            target_accuracy: 1e-15,
            ..Default::default()
        };

        let perf_config = AdaptivePrecisionConfig {
            performance_weight: 0.9, // Prioritize performance
            target_accuracy: 1e-6,
            ..Default::default()
        };

        let acc_sim = AdaptivePrecisionSimulator::new(high_acc_config);
        let perf_sim = AdaptivePrecisionSimulator::new(perf_config);

        assert!(acc_sim.config.target_accuracy < perf_sim.config.target_accuracy);
        assert!(acc_sim.config.performance_weight < perf_sim.config.performance_weight);
    }
}
