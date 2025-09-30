//! Advanced Quantum Circuit Profiler with Enhanced SciRS2 Performance Metrics
//!
//! This module provides state-of-the-art quantum circuit profiling capabilities
//! with comprehensive performance analysis, resource tracking, and optimization
//! recommendations using SciRS2's advanced performance metrics.

use crate::error::QuantRS2Error;
use crate::gate_translation::GateType;
use crate::scirs2_quantum_profiler::{
    CircuitProfilingResult, GateProfilingResult, MemoryAnalysis, OptimizationRecommendation,
    ProfilingPrecision, QuantumGate, SimdAnalysis,
};
use scirs2_core::Complex64;
// use scirs2_core::parallel_ops::*;
use crate::parallel_ops_stubs::*;
// use scirs2_core::memory::BufferPool;
use crate::buffer_pool::BufferPool;
use crate::platform::PlatformCapabilities;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::io::Write;
use std::sync::{
    atomic::{AtomicU64, AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};

/// Enhanced profiling configuration with SciRS2 metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedProfilingConfig {
    /// Base profiling precision
    pub precision: ProfilingPrecision,

    /// Enable deep performance analysis
    pub enable_deep_analysis: bool,

    /// Track memory allocation patterns
    pub track_memory_patterns: bool,

    /// Profile SIMD operations in detail
    pub profile_simd_operations: bool,

    /// Track parallel execution patterns
    pub track_parallel_patterns: bool,

    /// Enable cache analysis
    pub enable_cache_analysis: bool,

    /// Memory bandwidth tracking
    pub track_memory_bandwidth: bool,

    /// Instruction-level profiling
    pub enable_instruction_profiling: bool,

    /// Quantum resource estimation
    pub enable_resource_estimation: bool,

    /// Noise impact analysis
    pub analyze_noise_impact: bool,

    /// Circuit optimization suggestions
    pub generate_optimizations: bool,

    /// Bottleneck detection depth
    pub bottleneck_detection_depth: usize,

    /// Performance prediction model
    pub enable_performance_prediction: bool,

    /// Hardware-specific optimizations
    pub hardware_aware_profiling: bool,

    /// Export formats for reports
    pub export_formats: Vec<ExportFormat>,
}

impl Default for EnhancedProfilingConfig {
    fn default() -> Self {
        Self {
            precision: ProfilingPrecision::High,
            enable_deep_analysis: true,
            track_memory_patterns: true,
            profile_simd_operations: true,
            track_parallel_patterns: true,
            enable_cache_analysis: true,
            track_memory_bandwidth: true,
            enable_instruction_profiling: false,
            enable_resource_estimation: true,
            analyze_noise_impact: true,
            generate_optimizations: true,
            bottleneck_detection_depth: 5,
            enable_performance_prediction: true,
            hardware_aware_profiling: true,
            export_formats: vec![ExportFormat::JSON, ExportFormat::HTML, ExportFormat::CSV],
        }
    }
}

/// Export formats for profiling reports
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExportFormat {
    JSON,
    HTML,
    CSV,
    LaTeX,
    Markdown,
    Binary,
}

/// Performance metric types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricType {
    ExecutionTime,
    MemoryUsage,
    CacheHitRate,
    SimdUtilization,
    ParallelEfficiency,
    MemoryBandwidth,
    InstructionCount,
    BranchMisprediction,
    PowerConsumption,
    ThermalThrottle,
}

/// Advanced performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Raw metric values
    pub values: HashMap<MetricType, f64>,

    /// Time-series data for metrics
    pub time_series: HashMap<MetricType, Vec<(f64, f64)>>,

    /// Statistical analysis
    pub statistics: MetricStatistics,

    /// Correlations between metrics
    pub correlations: HashMap<(MetricType, MetricType), f64>,

    /// Anomaly detection results
    pub anomalies: Vec<AnomalyEvent>,
}

/// Statistical analysis of metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricStatistics {
    pub mean: HashMap<MetricType, f64>,
    pub std_dev: HashMap<MetricType, f64>,
    pub min: HashMap<MetricType, f64>,
    pub max: HashMap<MetricType, f64>,
    pub percentiles: HashMap<MetricType, BTreeMap<u8, f64>>,
}

/// Anomaly detection event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyEvent {
    pub timestamp: f64,
    pub metric: MetricType,
    pub severity: AnomalySeverity,
    pub description: String,
    pub impact: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Circuit bottleneck analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckAnalysis {
    /// Identified bottlenecks
    pub bottlenecks: Vec<Bottleneck>,

    /// Performance impact analysis
    pub impact_analysis: HashMap<String, f64>,

    /// Optimization opportunities
    pub opportunities: Vec<OptimizationOpportunity>,

    /// Resource utilization heatmap
    pub resource_heatmap: Array2<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bottleneck {
    pub location: CircuitLocation,
    pub bottleneck_type: BottleneckType,
    pub severity: f64,
    pub impact_percentage: f64,
    pub suggested_fixes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckType {
    MemoryBandwidth,
    ComputeIntensive,
    CacheMiss,
    ParallelizationIssue,
    SimdUnderutilization,
    DataDependency,
    ResourceContention,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitLocation {
    pub gate_index: usize,
    pub layer: usize,
    pub qubits: Vec<usize>,
    pub context: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationOpportunity {
    pub opportunity_type: OpportunityType,
    pub estimated_improvement: f64,
    pub difficulty: Difficulty,
    pub implementation: String,
    pub trade_offs: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpportunityType {
    GateFusion,
    Parallelization,
    SimdOptimization,
    MemoryReordering,
    CacheOptimization,
    AlgorithmicImprovement,
    HardwareSpecific,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Difficulty {
    Trivial,
    Easy,
    Medium,
    Hard,
    Expert,
}

/// Hardware performance model
#[derive(Serialize, Deserialize)]
pub struct HardwarePerformanceModel {
    /// Platform capabilities
    #[serde(skip, default = "PlatformCapabilities::detect")]
    pub platform: PlatformCapabilities,

    /// Performance characteristics
    pub characteristics: HardwareCharacteristics,

    /// Scaling models
    pub scaling_models: HashMap<String, ScalingModel>,

    /// Optimization strategies
    pub optimization_strategies: Vec<HardwareOptimizationStrategy>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareCharacteristics {
    pub cpu_frequency: f64,
    pub cache_sizes: Vec<usize>,
    pub memory_bandwidth: f64,
    pub simd_width: usize,
    pub num_cores: usize,
    pub gpu_available: bool,
    pub gpu_memory: Option<usize>,
    pub quantum_accelerator: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingModel {
    pub model_type: ScalingType,
    pub parameters: HashMap<String, f64>,
    pub confidence: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalingType {
    Linear,
    Logarithmic,
    Polynomial,
    Exponential,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareOptimizationStrategy {
    pub strategy_name: String,
    pub applicable_conditions: Vec<String>,
    pub expected_speedup: f64,
    pub implementation_cost: f64,
}

/// Enhanced quantum circuit profiler
pub struct EnhancedQuantumProfiler {
    config: EnhancedProfilingConfig,
    platform_caps: PlatformCapabilities,
    buffer_pool: Arc<BufferPool<Complex64>>,
    metrics_collector: Arc<MetricsCollector>,
    hardware_model: Option<HardwarePerformanceModel>,
    profiling_state: Arc<Mutex<ProfilingState>>,
}

/// Real-time metrics collector
struct MetricsCollector {
    execution_times: Mutex<HashMap<String, Vec<Duration>>>,
    memory_usage: AtomicUsize,
    simd_ops_count: AtomicU64,
    parallel_ops_count: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    bandwidth_bytes: AtomicU64,
    start_time: Instant,
}

impl MetricsCollector {
    fn new() -> Self {
        Self {
            execution_times: Mutex::new(HashMap::new()),
            memory_usage: AtomicUsize::new(0),
            simd_ops_count: AtomicU64::new(0),
            parallel_ops_count: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            bandwidth_bytes: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    fn record_execution(&self, operation: &str, duration: Duration) {
        let mut times = self.execution_times.lock().unwrap();
        times
            .entry(operation.to_string())
            .or_insert_with(Vec::new)
            .push(duration);
    }

    fn record_memory(&self, bytes: usize) {
        self.memory_usage.fetch_add(bytes, Ordering::Relaxed);
    }

    fn record_simd_op(&self) {
        self.simd_ops_count.fetch_add(1, Ordering::Relaxed);
    }

    fn record_parallel_op(&self) {
        self.parallel_ops_count.fetch_add(1, Ordering::Relaxed);
    }

    fn record_cache_access(&self, hit: bool) {
        if hit {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.cache_misses.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn record_bandwidth(&self, bytes: usize) {
        self.bandwidth_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
    }

    fn get_elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}

/// Profiling state management
struct ProfilingState {
    current_depth: usize,
    call_stack: Vec<String>,
    gate_timings: HashMap<usize, GateTimingInfo>,
    memory_snapshots: VecDeque<MemorySnapshot>,
    anomaly_detector: AnomalyDetector,
}

#[derive(Debug, Clone)]
struct GateTimingInfo {
    gate_type: GateType,
    start_time: Instant,
    end_time: Option<Instant>,
    memory_before: usize,
    memory_after: Option<usize>,
    simd_ops: u64,
    parallel_ops: u64,
}

#[derive(Debug, Clone)]
struct MemorySnapshot {
    timestamp: Instant,
    total_memory: usize,
    heap_memory: usize,
    stack_memory: usize,
    buffer_pool_memory: usize,
}

/// Anomaly detection system
struct AnomalyDetector {
    baseline_metrics: HashMap<MetricType, (f64, f64)>, // (mean, std_dev)
    detection_threshold: f64,
    history_window: usize,
    metric_history: HashMap<MetricType, VecDeque<f64>>,
}

impl AnomalyDetector {
    fn new(detection_threshold: f64, history_window: usize) -> Self {
        Self {
            baseline_metrics: HashMap::new(),
            detection_threshold,
            history_window,
            metric_history: HashMap::new(),
        }
    }

    fn update_metric(&mut self, metric: MetricType, value: f64) -> Option<AnomalyEvent> {
        let history = self
            .metric_history
            .entry(metric)
            .or_insert_with(VecDeque::new);
        history.push_back(value);

        if history.len() > self.history_window {
            history.pop_front();
        }

        // Calculate statistics
        if history.len() >= 10 {
            let mean: f64 = history.iter().sum::<f64>() / history.len() as f64;
            let variance: f64 =
                history.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / history.len() as f64;
            let std_dev = variance.sqrt();

            self.baseline_metrics.insert(metric, (mean, std_dev));

            // Check for anomaly
            if let Some(&(baseline_mean, baseline_std)) = self.baseline_metrics.get(&metric) {
                let z_score = (value - baseline_mean).abs() / baseline_std;

                if z_score > self.detection_threshold {
                    let severity = match z_score {
                        z if z < 3.0 => AnomalySeverity::Low,
                        z if z < 4.0 => AnomalySeverity::Medium,
                        z if z < 5.0 => AnomalySeverity::High,
                        _ => AnomalySeverity::Critical,
                    };

                    return Some(AnomalyEvent {
                        timestamp: history.len() as f64,
                        metric,
                        severity,
                        description: format!("Anomaly detected: z-score = {:.2}", z_score),
                        impact: z_score / 10.0, // Normalized impact
                    });
                }
            }
        }

        None
    }
}

impl EnhancedQuantumProfiler {
    /// Create a new enhanced profiler with default configuration
    pub fn new() -> Self {
        Self::with_config(EnhancedProfilingConfig::default())
    }

    /// Create a new enhanced profiler with custom configuration
    pub fn with_config(config: EnhancedProfilingConfig) -> Self {
        let platform_caps = PlatformCapabilities::detect();
        let buffer_pool = Arc::new(BufferPool::new());
        let metrics_collector = Arc::new(MetricsCollector::new());

        let hardware_model = if config.hardware_aware_profiling {
            Some(Self::build_hardware_model(&platform_caps))
        } else {
            None
        };

        let profiling_state = Arc::new(Mutex::new(ProfilingState {
            current_depth: 0,
            call_stack: Vec::new(),
            gate_timings: HashMap::new(),
            memory_snapshots: VecDeque::new(),
            anomaly_detector: AnomalyDetector::new(3.0, 100),
        }));

        Self {
            config,
            platform_caps,
            buffer_pool,
            metrics_collector,
            hardware_model,
            profiling_state,
        }
    }

    /// Build hardware performance model
    fn build_hardware_model(platform_caps: &PlatformCapabilities) -> HardwarePerformanceModel {
        let characteristics = HardwareCharacteristics {
            cpu_frequency: 3.0e9,                                      // 3 GHz estimate
            cache_sizes: vec![32 * 1024, 256 * 1024, 8 * 1024 * 1024], // L1, L2, L3
            memory_bandwidth: 50.0e9,                                  // 50 GB/s
            simd_width: if platform_caps.simd_available() {
                256
            } else {
                128
            },
            num_cores: platform_caps.cpu.logical_cores,
            gpu_available: platform_caps.gpu_available(),
            gpu_memory: if platform_caps.gpu_available() {
                Some(8 * 1024 * 1024 * 1024)
            } else {
                None
            },
            quantum_accelerator: None,
        };

        let mut scaling_models = HashMap::new();
        scaling_models.insert(
            "gate_execution".to_string(),
            ScalingModel {
                model_type: ScalingType::Linear,
                parameters: vec![("slope".to_string(), 1e-6), ("intercept".to_string(), 1e-7)]
                    .into_iter()
                    .collect(),
                confidence: 0.95,
            },
        );

        scaling_models.insert(
            "memory_access".to_string(),
            ScalingModel {
                model_type: ScalingType::Logarithmic,
                parameters: vec![("base".to_string(), 2.0), ("coefficient".to_string(), 1e-8)]
                    .into_iter()
                    .collect(),
                confidence: 0.90,
            },
        );

        let optimization_strategies = vec![
            HardwareOptimizationStrategy {
                strategy_name: "SIMD Vectorization".to_string(),
                applicable_conditions: vec!["vector_friendly_gates".to_string()],
                expected_speedup: 4.0,
                implementation_cost: 0.2,
            },
            HardwareOptimizationStrategy {
                strategy_name: "Parallel Execution".to_string(),
                applicable_conditions: vec!["independent_gates".to_string()],
                expected_speedup: characteristics.num_cores as f64 * 0.8,
                implementation_cost: 0.3,
            },
            HardwareOptimizationStrategy {
                strategy_name: "Cache Optimization".to_string(),
                applicable_conditions: vec!["repeated_access_patterns".to_string()],
                expected_speedup: 2.0,
                implementation_cost: 0.1,
            },
        ];

        HardwarePerformanceModel {
            platform: PlatformCapabilities::detect(),
            characteristics,
            scaling_models,
            optimization_strategies,
        }
    }

    /// Profile a quantum circuit with enhanced metrics
    pub fn profile_circuit(
        &self,
        circuit: &[QuantumGate],
        num_qubits: usize,
    ) -> Result<EnhancedProfilingReport, QuantRS2Error> {
        let start_time = Instant::now();

        // Initialize profiling
        self.initialize_profiling(num_qubits)?;

        // Profile each gate
        let mut gate_results = Vec::new();
        for (idx, gate) in circuit.iter().enumerate() {
            let gate_result = self.profile_gate(gate, idx, num_qubits)?;
            gate_results.push(gate_result);
        }

        // Collect overall metrics
        let performance_metrics = self.collect_performance_metrics()?;

        // Perform bottleneck analysis
        let bottleneck_analysis = if self.config.enable_deep_analysis {
            Some(self.analyze_bottlenecks(&gate_results, num_qubits)?)
        } else {
            None
        };

        // Generate optimization recommendations
        let optimizations = if self.config.generate_optimizations {
            self.generate_optimization_recommendations(&gate_results, &bottleneck_analysis)?
        } else {
            Vec::new()
        };

        // Predict performance on different hardware
        let performance_predictions = if self.config.enable_performance_prediction {
            Some(self.predict_performance(&gate_results, num_qubits)?)
        } else {
            None
        };

        // Create comprehensive report
        let total_time = start_time.elapsed();

        // Prepare export data before moving gate_results
        let export_data = self.prepare_export_data(&gate_results)?;

        Ok(EnhancedProfilingReport {
            summary: ProfilingSummary {
                total_execution_time: total_time,
                num_gates: circuit.len(),
                num_qubits,
                platform_info: PlatformCapabilities::detect(),
                profiling_config: self.config.clone(),
            },
            gate_results,
            performance_metrics,
            bottleneck_analysis,
            optimizations,
            performance_predictions,
            export_data,
        })
    }

    /// Initialize profiling state
    fn initialize_profiling(&self, num_qubits: usize) -> Result<(), QuantRS2Error> {
        let mut state = self.profiling_state.lock().unwrap();
        state.current_depth = 0;
        state.call_stack.clear();
        state.gate_timings.clear();
        state.memory_snapshots.clear();

        // Take initial memory snapshot
        let initial_snapshot = MemorySnapshot {
            timestamp: Instant::now(),
            total_memory: self.estimate_memory_usage(num_qubits),
            heap_memory: 0,
            stack_memory: 0,
            buffer_pool_memory: 0,
        };
        state.memory_snapshots.push_back(initial_snapshot);

        Ok(())
    }

    /// Profile individual gate
    fn profile_gate(
        &self,
        gate: &QuantumGate,
        gate_index: usize,
        num_qubits: usize,
    ) -> Result<EnhancedGateProfilingResult, QuantRS2Error> {
        let start_time = Instant::now();
        let memory_before = self.estimate_memory_usage(num_qubits);

        // Record gate start
        {
            let mut state = self.profiling_state.lock().unwrap();
            state.gate_timings.insert(
                gate_index,
                GateTimingInfo {
                    gate_type: gate.gate_type().clone(),
                    start_time,
                    end_time: None,
                    memory_before,
                    memory_after: None,
                    simd_ops: 0,
                    parallel_ops: 0,
                },
            );
        }

        // Simulate gate execution with metrics collection
        self.simulate_gate_execution(gate, num_qubits)?;

        let end_time = Instant::now();
        let memory_after = self.estimate_memory_usage(num_qubits);
        let execution_time = end_time - start_time;

        // Update gate timing info
        {
            let mut state = self.profiling_state.lock().unwrap();
            if let Some(timing_info) = state.gate_timings.get_mut(&gate_index) {
                timing_info.end_time = Some(end_time);
                timing_info.memory_after = Some(memory_after);
                timing_info.simd_ops = self
                    .metrics_collector
                    .simd_ops_count
                    .load(Ordering::Relaxed);
                timing_info.parallel_ops = self
                    .metrics_collector
                    .parallel_ops_count
                    .load(Ordering::Relaxed);
            }
        }

        // Record metrics
        self.metrics_collector
            .record_execution(&format!("{:?}", gate.gate_type()), execution_time);
        self.metrics_collector
            .record_memory(memory_after.saturating_sub(memory_before));

        // Check for anomalies
        let mut anomalies = Vec::new();
        {
            let mut state = self.profiling_state.lock().unwrap();
            if let Some(anomaly) = state
                .anomaly_detector
                .update_metric(MetricType::ExecutionTime, execution_time.as_secs_f64())
            {
                anomalies.push(anomaly);
            }
        }

        Ok(EnhancedGateProfilingResult {
            gate_index,
            gate_type: gate.gate_type().clone(),
            execution_time,
            memory_delta: memory_after as i64 - memory_before as i64,
            simd_operations: self
                .metrics_collector
                .simd_ops_count
                .load(Ordering::Relaxed),
            parallel_operations: self
                .metrics_collector
                .parallel_ops_count
                .load(Ordering::Relaxed),
            cache_efficiency: self.calculate_cache_efficiency(),
            bandwidth_usage: self.calculate_bandwidth_usage(execution_time),
            anomalies,
            detailed_metrics: self.collect_detailed_gate_metrics(gate, execution_time)?,
        })
    }

    /// Simulate gate execution for profiling
    fn simulate_gate_execution(
        &self,
        gate: &QuantumGate,
        num_qubits: usize,
    ) -> Result<(), QuantRS2Error> {
        // Simulate different aspects based on gate type
        match gate.gate_type() {
            GateType::H | GateType::X | GateType::Y | GateType::Z => {
                // Single-qubit gates
                if self.platform_caps.simd_available() {
                    self.metrics_collector.record_simd_op();
                }
                self.metrics_collector
                    .record_bandwidth(16 * (1 << num_qubits)); // Complex64 operations
            }
            GateType::CNOT | GateType::CZ => {
                // Two-qubit gates
                if num_qubits > 10 {
                    self.metrics_collector.record_parallel_op();
                }
                self.metrics_collector
                    .record_bandwidth(32 * (1 << num_qubits));
            }
            _ => {
                // Multi-qubit gates
                self.metrics_collector.record_parallel_op();
                self.metrics_collector
                    .record_bandwidth(64 * (1 << num_qubits));
            }
        }

        // Simulate cache behavior
        use scirs2_core::random::prelude::*;
        let cache_hit = thread_rng().gen::<f64>() > 0.2; // 80% hit rate simulation
        self.metrics_collector.record_cache_access(cache_hit);

        Ok(())
    }

    /// Estimate memory usage
    fn estimate_memory_usage(&self, num_qubits: usize) -> usize {
        let state_vector_size = (1 << num_qubits) * std::mem::size_of::<Complex64>();
        let overhead = state_vector_size / 10; // 10% overhead estimate
        state_vector_size + overhead
    }

    /// Calculate cache efficiency
    fn calculate_cache_efficiency(&self) -> f64 {
        let hits = self.metrics_collector.cache_hits.load(Ordering::Relaxed) as f64;
        let misses = self.metrics_collector.cache_misses.load(Ordering::Relaxed) as f64;
        let total = hits + misses;

        if total > 0.0 {
            hits / total
        } else {
            1.0 // Assume perfect efficiency if no data
        }
    }

    /// Calculate bandwidth usage
    fn calculate_bandwidth_usage(&self, duration: Duration) -> f64 {
        let bytes = self
            .metrics_collector
            .bandwidth_bytes
            .load(Ordering::Relaxed) as f64;
        let seconds = duration.as_secs_f64();

        if seconds > 0.0 {
            bytes / seconds
        } else {
            0.0
        }
    }

    /// Collect detailed gate metrics
    fn collect_detailed_gate_metrics(
        &self,
        gate: &QuantumGate,
        execution_time: Duration,
    ) -> Result<HashMap<String, f64>, QuantRS2Error> {
        let mut metrics = HashMap::new();

        metrics.insert(
            "execution_time_us".to_string(),
            execution_time.as_micros() as f64,
        );
        metrics.insert(
            "cache_efficiency".to_string(),
            self.calculate_cache_efficiency(),
        );
        metrics.insert(
            "bandwidth_mbps".to_string(),
            self.calculate_bandwidth_usage(execution_time) / 1e6,
        );

        if let Some(ref hw_model) = self.hardware_model {
            metrics.insert(
                "theoretical_flops".to_string(),
                self.estimate_flops(gate, &hw_model.characteristics),
            );
        }

        Ok(metrics)
    }

    /// Estimate FLOPS for a gate
    fn estimate_flops(&self, gate: &QuantumGate, hw_chars: &HardwareCharacteristics) -> f64 {
        let base_flops = match gate.gate_type() {
            GateType::H => 8.0, // 4 complex multiplications
            GateType::X | GateType::Y | GateType::Z => 4.0,
            GateType::CNOT | GateType::CZ => 16.0,
            _ => 32.0, // Conservative estimate for complex gates
        };

        base_flops * hw_chars.cpu_frequency
    }

    /// Collect overall performance metrics
    fn collect_performance_metrics(&self) -> Result<PerformanceMetrics, QuantRS2Error> {
        let mut values = HashMap::new();
        let elapsed = self.metrics_collector.get_elapsed();

        values.insert(MetricType::ExecutionTime, elapsed.as_secs_f64());
        values.insert(
            MetricType::MemoryUsage,
            self.metrics_collector.memory_usage.load(Ordering::Relaxed) as f64,
        );
        values.insert(
            MetricType::SimdUtilization,
            self.metrics_collector
                .simd_ops_count
                .load(Ordering::Relaxed) as f64,
        );
        values.insert(
            MetricType::ParallelEfficiency,
            self.metrics_collector
                .parallel_ops_count
                .load(Ordering::Relaxed) as f64,
        );
        values.insert(MetricType::CacheHitRate, self.calculate_cache_efficiency());
        values.insert(
            MetricType::MemoryBandwidth,
            self.calculate_bandwidth_usage(elapsed),
        );

        // Calculate statistics
        let statistics = self.calculate_metric_statistics(&values)?;

        // Time series data (simplified for this implementation)
        let mut time_series = HashMap::new();
        for (metric, value) in &values {
            time_series.insert(*metric, vec![(0.0, 0.0), (elapsed.as_secs_f64(), *value)]);
        }

        Ok(PerformanceMetrics {
            values,
            time_series,
            statistics,
            correlations: HashMap::new(), // Simplified
            anomalies: Vec::new(),        // Collected separately
        })
    }

    /// Calculate metric statistics
    fn calculate_metric_statistics(
        &self,
        values: &HashMap<MetricType, f64>,
    ) -> Result<MetricStatistics, QuantRS2Error> {
        let mut mean = HashMap::new();
        let mut std_dev = HashMap::new();
        let mut min = HashMap::new();
        let mut max = HashMap::new();
        let mut percentiles = HashMap::new();

        for (metric, value) in values {
            mean.insert(*metric, *value);
            std_dev.insert(*metric, 0.0); // Simplified
            min.insert(*metric, *value);
            max.insert(*metric, *value);

            let mut percs = BTreeMap::new();
            percs.insert(50, *value); // Median
            percs.insert(95, *value * 1.1); // 95th percentile estimate
            percs.insert(99, *value * 1.2); // 99th percentile estimate
            percentiles.insert(*metric, percs);
        }

        Ok(MetricStatistics {
            mean,
            std_dev,
            min,
            max,
            percentiles,
        })
    }

    /// Analyze bottlenecks in the circuit
    fn analyze_bottlenecks(
        &self,
        gate_results: &[EnhancedGateProfilingResult],
        num_qubits: usize,
    ) -> Result<BottleneckAnalysis, QuantRS2Error> {
        let mut bottlenecks = Vec::new();
        let mut impact_analysis = HashMap::new();
        let mut opportunities = Vec::new();

        // Find execution time bottlenecks
        let total_time: Duration = gate_results.iter().map(|r| r.execution_time).sum();
        let avg_time = total_time / gate_results.len() as u32;

        for (idx, result) in gate_results.iter().enumerate() {
            if result.execution_time > avg_time * 2 {
                let impact = result.execution_time.as_secs_f64() / total_time.as_secs_f64();

                bottlenecks.push(Bottleneck {
                    location: CircuitLocation {
                        gate_index: idx,
                        layer: idx / num_qubits, // Simplified layer calculation
                        qubits: vec![idx % num_qubits], // Simplified
                        context: format!("Gate {:?} at index {}", result.gate_type, idx),
                    },
                    bottleneck_type: BottleneckType::ComputeIntensive,
                    severity: impact * 100.0,
                    impact_percentage: impact * 100.0,
                    suggested_fixes: vec![
                        "Consider gate decomposition".to_string(),
                        "Explore parallel execution".to_string(),
                    ],
                });

                impact_analysis.insert(format!("gate_{}", idx), impact);
            }

            // Check for low cache efficiency
            if result.cache_efficiency < 0.5 {
                bottlenecks.push(Bottleneck {
                    location: CircuitLocation {
                        gate_index: idx,
                        layer: idx / num_qubits,
                        qubits: vec![idx % num_qubits],
                        context: format!("Poor cache efficiency at gate {}", idx),
                    },
                    bottleneck_type: BottleneckType::CacheMiss,
                    severity: (1.0 - result.cache_efficiency) * 50.0,
                    impact_percentage: 10.0, // Estimated impact
                    suggested_fixes: vec![
                        "Reorder operations for better locality".to_string(),
                        "Consider data prefetching".to_string(),
                    ],
                });
            }
        }

        // Identify optimization opportunities
        if self.platform_caps.simd_available() {
            let simd_utilization = gate_results
                .iter()
                .filter(|r| r.simd_operations > 0)
                .count() as f64
                / gate_results.len() as f64;

            if simd_utilization < 0.5 {
                opportunities.push(OptimizationOpportunity {
                    opportunity_type: OpportunityType::SimdOptimization,
                    estimated_improvement: (1.0 - simd_utilization) * 2.0,
                    difficulty: Difficulty::Medium,
                    implementation: "Vectorize gate operations using AVX2".to_string(),
                    trade_offs: vec!["Increased code complexity".to_string()],
                });
            }
        }

        // Create resource heatmap
        let resource_heatmap = Array2::zeros((gate_results.len(), 4)); // gates x resource types

        Ok(BottleneckAnalysis {
            bottlenecks,
            impact_analysis,
            opportunities,
            resource_heatmap,
        })
    }

    /// Generate optimization recommendations
    fn generate_optimization_recommendations(
        &self,
        gate_results: &[EnhancedGateProfilingResult],
        bottleneck_analysis: &Option<BottleneckAnalysis>,
    ) -> Result<Vec<EnhancedOptimizationRecommendation>, QuantRS2Error> {
        let mut recommendations = Vec::new();

        // Gate fusion opportunities
        for window in gate_results.windows(2) {
            if Self::can_fuse_gates(&window[0].gate_type, &window[1].gate_type) {
                recommendations.push(EnhancedOptimizationRecommendation {
                    recommendation_type: RecommendationType::GateFusion,
                    priority: Priority::High,
                    estimated_speedup: 1.5,
                    implementation_difficulty: Difficulty::Easy,
                    description: format!(
                        "Fuse {:?} and {:?} gates",
                        window[0].gate_type, window[1].gate_type
                    ),
                    code_example: Some(
                        self.generate_fusion_code(&window[0].gate_type, &window[1].gate_type),
                    ),
                    prerequisites: vec!["Adjacent gates must commute".to_string()],
                    risks: vec!["May increase numerical error".to_string()],
                });
            }
        }

        // Hardware-specific optimizations
        if let Some(ref hw_model) = self.hardware_model {
            for strategy in &hw_model.optimization_strategies {
                if strategy.expected_speedup > 1.5 {
                    recommendations.push(EnhancedOptimizationRecommendation {
                        recommendation_type: RecommendationType::HardwareSpecific,
                        priority: Priority::Medium,
                        estimated_speedup: strategy.expected_speedup,
                        implementation_difficulty: Difficulty::Hard,
                        description: strategy.strategy_name.clone(),
                        code_example: None,
                        prerequisites: strategy.applicable_conditions.clone(),
                        risks: vec!["Platform-specific code".to_string()],
                    });
                }
            }
        }

        // Bottleneck-based recommendations
        if let Some(bottleneck_analysis) = bottleneck_analysis {
            for opportunity in &bottleneck_analysis.opportunities {
                recommendations.push(EnhancedOptimizationRecommendation {
                    recommendation_type: match opportunity.opportunity_type {
                        OpportunityType::GateFusion => RecommendationType::GateFusion,
                        OpportunityType::Parallelization => RecommendationType::Parallelization,
                        OpportunityType::SimdOptimization => RecommendationType::SimdVectorization,
                        OpportunityType::MemoryReordering => RecommendationType::MemoryOptimization,
                        OpportunityType::CacheOptimization => RecommendationType::CacheOptimization,
                        OpportunityType::AlgorithmicImprovement => {
                            RecommendationType::AlgorithmicChange
                        }
                        OpportunityType::HardwareSpecific => RecommendationType::HardwareSpecific,
                    },
                    priority: match opportunity.difficulty {
                        Difficulty::Trivial | Difficulty::Easy => Priority::High,
                        Difficulty::Medium => Priority::Medium,
                        Difficulty::Hard | Difficulty::Expert => Priority::Low,
                    },
                    estimated_speedup: opportunity.estimated_improvement,
                    implementation_difficulty: opportunity.difficulty,
                    description: opportunity.implementation.clone(),
                    code_example: None,
                    prerequisites: Vec::new(),
                    risks: opportunity.trade_offs.clone(),
                });
            }
        }

        Ok(recommendations)
    }

    /// Check if two gates can be fused
    fn can_fuse_gates(gate1: &GateType, gate2: &GateType) -> bool {
        use GateType::*;
        matches!(
            (gate1, gate2),
            (H, H) | (X, X) | (Y, Y) | (Z, Z) | // Self-inverse gates
            (Rz(_), Rz(_)) | (Rx(_), Rx(_)) | (Ry(_), Ry(_)) // Rotation gates
        )
    }

    /// Generate fusion code example
    fn generate_fusion_code(&self, gate1: &GateType, gate2: &GateType) -> String {
        format!(
            "// Fused {:?} and {:?}\nlet fused_gate = FusedGate::new({:?}, {:?});\nfused_gate.apply(state);",
            gate1, gate2, gate1, gate2
        )
    }

    /// Predict performance on different hardware
    fn predict_performance(
        &self,
        gate_results: &[EnhancedGateProfilingResult],
        num_qubits: usize,
    ) -> Result<PerformancePredictions, QuantRS2Error> {
        let mut predictions = HashMap::new();

        // Current hardware baseline
        let current_time: Duration = gate_results.iter().map(|r| r.execution_time).sum();

        predictions.insert(
            "current".to_string(),
            PredictedPerformance {
                hardware_description: "Current Platform".to_string(),
                estimated_time: current_time,
                confidence: 1.0,
                limiting_factors: vec!["Actual measurement".to_string()],
            },
        );

        // GPU prediction
        if self.platform_caps.gpu_available() {
            let gpu_speedup = (num_qubits as f64).ln() * 2.0; // Logarithmic speedup model
            predictions.insert(
                "gpu".to_string(),
                PredictedPerformance {
                    hardware_description: "GPU Acceleration".to_string(),
                    estimated_time: current_time / gpu_speedup as u32,
                    confidence: 0.8,
                    limiting_factors: vec!["Memory transfer overhead".to_string()],
                },
            );
        }

        // Quantum hardware prediction
        predictions.insert(
            "quantum_hw".to_string(),
            PredictedPerformance {
                hardware_description: "Quantum Hardware (NISQ)".to_string(),
                estimated_time: Duration::from_millis(gate_results.len() as u64 * 10), // 10ms per gate
                confidence: 0.5,
                limiting_factors: vec![
                    "Gate fidelity".to_string(),
                    "Connectivity constraints".to_string(),
                    "Decoherence".to_string(),
                ],
            },
        );

        // Cloud QPU prediction
        predictions.insert(
            "cloud_qpu".to_string(),
            PredictedPerformance {
                hardware_description: "Cloud Quantum Processor".to_string(),
                estimated_time: Duration::from_secs(1)
                    + Duration::from_millis(gate_results.len() as u64),
                confidence: 0.6,
                limiting_factors: vec!["Network latency".to_string(), "Queue time".to_string()],
            },
        );

        // Generate hardware recommendations before moving predictions
        let hardware_recommendations =
            self.generate_hardware_recommendations(num_qubits, &predictions);

        Ok(PerformancePredictions {
            predictions,
            scaling_analysis: self.analyze_scaling(num_qubits)?,
            hardware_recommendations,
        })
    }

    /// Analyze scaling behavior
    fn analyze_scaling(&self, num_qubits: usize) -> Result<ScalingAnalysis, QuantRS2Error> {
        Ok(ScalingAnalysis {
            qubit_scaling: ScalingType::Exponential,
            gate_scaling: ScalingType::Linear,
            memory_scaling: ScalingType::Exponential,
            predicted_limits: HashMap::from([
                ("max_qubits_cpu".to_string(), 30.0),
                ("max_qubits_gpu".to_string(), 35.0),
                ("max_gates_per_second".to_string(), 1e6),
            ]),
        })
    }

    /// Generate hardware recommendations
    fn generate_hardware_recommendations(
        &self,
        num_qubits: usize,
        predictions: &HashMap<String, PredictedPerformance>,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if num_qubits > 20 {
            recommendations.push("Consider GPU acceleration for large circuits".to_string());
        }

        if num_qubits > 30 {
            recommendations.push("Tensor network methods recommended".to_string());
        }

        if let Some(gpu_pred) = predictions.get("gpu") {
            if gpu_pred.confidence > 0.7 {
                recommendations.push("GPU acceleration shows promising speedup".to_string());
            }
        }

        recommendations
    }

    /// Prepare export data
    fn prepare_export_data(
        &self,
        gate_results: &[EnhancedGateProfilingResult],
    ) -> Result<HashMap<ExportFormat, Vec<u8>>, QuantRS2Error> {
        let mut export_data = HashMap::new();

        for format in &self.config.export_formats {
            let data = match format {
                ExportFormat::JSON => self.export_to_json(gate_results)?,
                ExportFormat::CSV => self.export_to_csv(gate_results)?,
                ExportFormat::HTML => self.export_to_html(gate_results)?,
                _ => Vec::new(), // Other formats not implemented in this example
            };
            export_data.insert(*format, data);
        }

        Ok(export_data)
    }

    /// Export to JSON format
    fn export_to_json(
        &self,
        gate_results: &[EnhancedGateProfilingResult],
    ) -> Result<Vec<u8>, QuantRS2Error> {
        let json = serde_json::to_vec_pretty(gate_results).map_err(|e| {
            QuantRS2Error::ComputationError(format!("CSV generation failed: {}", e))
        })?;
        Ok(json)
    }

    /// Export to CSV format
    fn export_to_csv(
        &self,
        gate_results: &[EnhancedGateProfilingResult],
    ) -> Result<Vec<u8>, QuantRS2Error> {
        let mut csv = Vec::new();
        writeln!(csv, "gate_index,gate_type,execution_time_us,memory_delta,simd_ops,parallel_ops,cache_efficiency")
            .map_err(|e| QuantRS2Error::ComputationError(format!("IO error: {}", e)))?;

        for result in gate_results {
            writeln!(
                csv,
                "{},{:?},{},{},{},{},{:.2}",
                result.gate_index,
                result.gate_type,
                result.execution_time.as_micros(),
                result.memory_delta,
                result.simd_operations,
                result.parallel_operations,
                result.cache_efficiency
            )
            .map_err(|e| QuantRS2Error::ComputationError(format!("IO error: {}", e)))?;
        }

        Ok(csv)
    }

    /// Export to HTML format
    fn export_to_html(
        &self,
        gate_results: &[EnhancedGateProfilingResult],
    ) -> Result<Vec<u8>, QuantRS2Error> {
        let mut html = Vec::new();
        writeln!(
            html,
            "<html><head><title>Quantum Circuit Profiling Report</title>"
        )
        .map_err(|e| QuantRS2Error::ComputationError(format!("IO error: {}", e)))?;
        writeln!(html, "<style>table {{ border-collapse: collapse; }} th, td {{ border: 1px solid black; padding: 8px; }}</style>")
            .map_err(|e| QuantRS2Error::ComputationError(format!("IO error: {}", e)))?;
        writeln!(html, "</head><body><h1>Profiling Results</h1><table>")
            .map_err(|e| QuantRS2Error::ComputationError(format!("IO error: {}", e)))?;
        writeln!(html, "<tr><th>Gate</th><th>Type</th><th>Time (μs)</th><th>Memory</th><th>SIMD</th><th>Parallel</th><th>Cache</th></tr>")
            .map_err(|e| QuantRS2Error::ComputationError(format!("IO error: {}", e)))?;

        for result in gate_results {
            writeln!(html, "<tr><td>{}</td><td>{:?}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{:.1}%</td></tr>",
                result.gate_index,
                result.gate_type,
                result.execution_time.as_micros(),
                result.memory_delta,
                result.simd_operations,
                result.parallel_operations,
                result.cache_efficiency * 100.0
            ).map_err(|e| QuantRS2Error::ComputationError(format!("IO error: {}", e)))?;
        }

        writeln!(html, "</table></body></html>")
            .map_err(|e| QuantRS2Error::ComputationError(format!("IO error: {}", e)))?;

        Ok(html)
    }
}

/// Enhanced gate profiling result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedGateProfilingResult {
    pub gate_index: usize,
    pub gate_type: GateType,
    pub execution_time: Duration,
    pub memory_delta: i64,
    pub simd_operations: u64,
    pub parallel_operations: u64,
    pub cache_efficiency: f64,
    pub bandwidth_usage: f64,
    pub anomalies: Vec<AnomalyEvent>,
    pub detailed_metrics: HashMap<String, f64>,
}

/// Enhanced optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedOptimizationRecommendation {
    pub recommendation_type: RecommendationType,
    pub priority: Priority,
    pub estimated_speedup: f64,
    pub implementation_difficulty: Difficulty,
    pub description: String,
    pub code_example: Option<String>,
    pub prerequisites: Vec<String>,
    pub risks: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationType {
    GateFusion,
    Parallelization,
    SimdVectorization,
    MemoryOptimization,
    CacheOptimization,
    AlgorithmicChange,
    HardwareSpecific,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

/// Performance predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformancePredictions {
    pub predictions: HashMap<String, PredictedPerformance>,
    pub scaling_analysis: ScalingAnalysis,
    pub hardware_recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedPerformance {
    pub hardware_description: String,
    pub estimated_time: Duration,
    pub confidence: f64,
    pub limiting_factors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingAnalysis {
    pub qubit_scaling: ScalingType,
    pub gate_scaling: ScalingType,
    pub memory_scaling: ScalingType,
    pub predicted_limits: HashMap<String, f64>,
}

/// Enhanced profiling report
#[derive(Serialize, Deserialize)]
pub struct EnhancedProfilingReport {
    pub summary: ProfilingSummary,
    pub gate_results: Vec<EnhancedGateProfilingResult>,
    pub performance_metrics: PerformanceMetrics,
    pub bottleneck_analysis: Option<BottleneckAnalysis>,
    pub optimizations: Vec<EnhancedOptimizationRecommendation>,
    pub performance_predictions: Option<PerformancePredictions>,
    pub export_data: HashMap<ExportFormat, Vec<u8>>,
}

#[derive(Serialize, Deserialize)]
pub struct ProfilingSummary {
    pub total_execution_time: Duration,
    pub num_gates: usize,
    pub num_qubits: usize,
    #[serde(skip, default = "PlatformCapabilities::detect")]
    pub platform_info: PlatformCapabilities,
    pub profiling_config: EnhancedProfilingConfig,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_profiler_creation() {
        let profiler = EnhancedQuantumProfiler::new();
        assert!(profiler.platform_caps.simd_available());
    }

    #[test]
    fn test_basic_profiling() {
        let profiler = EnhancedQuantumProfiler::new();
        let gates = vec![
            QuantumGate::new(GateType::H, vec![0], None),
            QuantumGate::new(GateType::CNOT, vec![0, 1], None),
            QuantumGate::new(GateType::H, vec![1], None),
        ];

        let result = profiler.profile_circuit(&gates, 2).unwrap();
        assert_eq!(result.gate_results.len(), 3);
        assert!(result.summary.total_execution_time.as_nanos() > 0);
    }

    #[test]
    fn test_bottleneck_detection() {
        let config = EnhancedProfilingConfig {
            enable_deep_analysis: true,
            ..Default::default()
        };
        let profiler = EnhancedQuantumProfiler::with_config(config);

        let gates = vec![
            QuantumGate::new(GateType::H, vec![0], None),
            QuantumGate::new(GateType::T, vec![0], None),
            QuantumGate::new(GateType::H, vec![0], None),
        ];

        let result = profiler.profile_circuit(&gates, 1).unwrap();
        assert!(result.bottleneck_analysis.is_some());
    }

    #[test]
    fn test_optimization_recommendations() {
        let config = EnhancedProfilingConfig {
            generate_optimizations: true,
            ..Default::default()
        };
        let profiler = EnhancedQuantumProfiler::with_config(config);

        let gates = vec![
            QuantumGate::new(GateType::H, vec![0], None),
            QuantumGate::new(GateType::H, vec![0], None), // H^2 = I
        ];

        let result = profiler.profile_circuit(&gates, 1).unwrap();
        assert!(!result.optimizations.is_empty());
        assert!(result
            .optimizations
            .iter()
            .any(|opt| opt.recommendation_type == RecommendationType::GateFusion));
    }

    #[test]
    fn test_performance_prediction() {
        let config = EnhancedProfilingConfig {
            enable_performance_prediction: true,
            ..Default::default()
        };
        let profiler = EnhancedQuantumProfiler::with_config(config);

        let gates = vec![
            QuantumGate::new(GateType::X, vec![0], None),
            QuantumGate::new(GateType::Y, vec![1], None),
            QuantumGate::new(GateType::Z, vec![2], None),
        ];

        let result = profiler.profile_circuit(&gates, 3).unwrap();
        assert!(result.performance_predictions.is_some());

        let predictions = result.performance_predictions.unwrap();
        assert!(predictions.predictions.contains_key("current"));
        assert!(predictions.predictions.contains_key("quantum_hw"));
    }

    #[test]
    fn test_export_formats() {
        let config = EnhancedProfilingConfig {
            export_formats: vec![ExportFormat::JSON, ExportFormat::CSV, ExportFormat::HTML],
            ..Default::default()
        };
        let profiler = EnhancedQuantumProfiler::with_config(config);

        let gates = vec![QuantumGate::new(GateType::H, vec![0], None)];

        let result = profiler.profile_circuit(&gates, 1).unwrap();
        assert_eq!(result.export_data.len(), 3);
        assert!(result.export_data.contains_key(&ExportFormat::JSON));
        assert!(result.export_data.contains_key(&ExportFormat::CSV));
        assert!(result.export_data.contains_key(&ExportFormat::HTML));
    }
}
