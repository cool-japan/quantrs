//! Performance profiler for quantum optimization.
//!
//! This module provides comprehensive performance profiling tools
//! for analyzing QUBO generation, solving, and optimization performance.

#![allow(dead_code)]

#[cfg(feature = "dwave")]
use crate::compile::{Compile, CompiledModel};
#[cfg(feature = "plotters")]
use plotters::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Performance profiler
pub struct PerformanceProfiler {
    /// Configuration
    config: ProfilerConfig,
    /// Profile data
    profiles: Vec<Profile>,
    /// Current profile
    current_profile: Option<ProfileContext>,
    /// Metrics collectors
    collectors: Vec<Box<dyn MetricsCollector>>,
    /// Analysis engine
    analyzer: PerformanceAnalyzer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerConfig {
    /// Enable profiling
    pub enabled: bool,
    /// Sampling interval
    pub sampling_interval: Duration,
    /// Metrics to collect
    pub metrics: Vec<MetricType>,
    /// Memory profiling
    pub profile_memory: bool,
    /// CPU profiling
    pub profile_cpu: bool,
    /// GPU profiling
    pub profile_gpu: bool,
    /// Detailed timing
    pub detailed_timing: bool,
    /// Output format
    pub output_format: OutputFormat,
    /// Auto-save interval
    pub auto_save_interval: Option<Duration>,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sampling_interval: Duration::from_millis(100),
            metrics: vec![MetricType::Time, MetricType::Memory],
            profile_memory: true,
            profile_cpu: true,
            profile_gpu: false,
            detailed_timing: false,
            output_format: OutputFormat::Json,
            auto_save_interval: Some(Duration::from_secs(60)),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MetricType {
    /// Execution time
    Time,
    /// Memory usage
    Memory,
    /// CPU usage
    CPU,
    /// GPU usage
    GPU,
    /// Cache metrics
    Cache,
    /// I/O metrics
    IO,
    /// Network metrics
    Network,
    /// Custom metric
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    /// JSON format
    Json,
    /// Binary format
    Binary,
    /// CSV format
    Csv,
    /// Flame graph
    FlameGraph,
    /// Chrome tracing format
    ChromeTrace,
}

/// Profile data
#[derive(Debug, Clone)]
pub struct Profile {
    /// Profile ID
    pub id: String,
    /// Start time
    pub start_time: Instant,
    /// End time
    pub end_time: Option<Instant>,
    /// Events
    pub events: Vec<ProfileEvent>,
    /// Metrics
    pub metrics: MetricsData,
    /// Call graph
    pub call_graph: CallGraph,
    /// Resource usage
    pub resource_usage: ResourceUsage,
}

#[derive(Debug, Clone)]
pub struct ProfileEvent {
    /// Event timestamp
    pub timestamp: Instant,
    /// Event type
    pub event_type: EventType,
    /// Event name
    pub name: String,
    /// Duration (if applicable)
    pub duration: Option<Duration>,
    /// Associated data
    pub data: HashMap<String, String>,
    /// Thread ID
    pub thread_id: thread::ThreadId,
}

#[derive(Debug, Clone)]
pub enum EventType {
    /// Function call
    FunctionCall,
    /// Function return
    FunctionReturn,
    /// Memory allocation
    MemoryAlloc,
    /// Memory deallocation
    MemoryFree,
    /// I/O operation
    IOOperation,
    /// Synchronization
    Synchronization,
    /// Custom event
    Custom(String),
}

#[derive(Debug, Clone)]
pub struct MetricsData {
    /// Time metrics
    pub time_metrics: TimeMetrics,
    /// Memory metrics
    pub memory_metrics: MemoryMetrics,
    /// Computation metrics
    pub computation_metrics: ComputationMetrics,
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
}

#[derive(Debug, Clone)]
pub struct TimeMetrics {
    /// Total execution time
    pub total_time: Duration,
    /// QUBO generation time
    pub qubo_generation_time: Duration,
    /// Compilation time
    pub compilation_time: Duration,
    /// Solving time
    pub solving_time: Duration,
    /// Post-processing time
    pub post_processing_time: Duration,
    /// Time breakdown by function
    pub function_times: BTreeMap<String, Duration>,
    /// Time percentiles
    pub percentiles: Percentiles,
}

#[derive(Debug, Clone)]
pub struct Percentiles {
    pub p50: Duration,
    pub p90: Duration,
    pub p95: Duration,
    pub p99: Duration,
    pub p999: Duration,
}

#[derive(Debug, Clone)]
pub struct MemoryMetrics {
    /// Peak memory usage
    pub peak_memory: usize,
    /// Average memory usage
    pub avg_memory: usize,
    /// Memory allocations
    pub allocations: usize,
    /// Memory deallocations
    pub deallocations: usize,
    /// Largest allocation
    pub largest_allocation: usize,
    /// Memory timeline
    pub memory_timeline: Vec<(Instant, usize)>,
}

#[derive(Debug, Clone)]
pub struct ComputationMetrics {
    /// FLOPS (floating-point operations per second)
    pub flops: f64,
    /// Memory bandwidth
    pub memory_bandwidth: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Branch prediction accuracy
    pub branch_prediction_accuracy: f64,
    /// Vectorization efficiency
    pub vectorization_efficiency: f64,
}

#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Solution quality over time
    pub quality_timeline: Vec<(Duration, f64)>,
    /// Convergence rate
    pub convergence_rate: f64,
    /// Improvement per iteration
    pub improvement_per_iteration: f64,
    /// Time to first solution
    pub time_to_first_solution: Duration,
    /// Time to best solution
    pub time_to_best_solution: Duration,
}

#[derive(Debug, Clone)]
pub struct CallGraph {
    /// Nodes (functions)
    pub nodes: Vec<CallNode>,
    /// Edges (calls)
    pub edges: Vec<CallEdge>,
    /// Root nodes
    pub roots: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct CallNode {
    /// Node ID
    pub id: usize,
    /// Function name
    pub name: String,
    /// Total time
    pub total_time: Duration,
    /// Self time
    pub self_time: Duration,
    /// Call count
    pub call_count: usize,
    /// Average time per call
    pub avg_time: Duration,
}

#[derive(Debug, Clone)]
pub struct CallEdge {
    /// Source node
    pub from: usize,
    /// Target node
    pub to: usize,
    /// Number of calls
    pub call_count: usize,
    /// Total time
    pub total_time: Duration,
}

#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// CPU usage timeline
    pub cpu_usage: Vec<(Instant, f64)>,
    /// Memory usage timeline
    pub memory_usage: Vec<(Instant, usize)>,
    /// GPU usage timeline
    pub gpu_usage: Vec<(Instant, f64)>,
    /// I/O operations
    pub io_operations: Vec<IOOperation>,
    /// Network operations
    pub network_operations: Vec<NetworkOperation>,
}

#[derive(Debug, Clone)]
pub struct IOOperation {
    /// Timestamp
    pub timestamp: Instant,
    /// Operation type
    pub operation: IOOpType,
    /// Bytes transferred
    pub bytes: usize,
    /// Duration
    pub duration: Duration,
    /// File/device
    pub target: String,
}

#[derive(Debug, Clone)]
pub enum IOOpType {
    Read,
    Write,
    Seek,
    Flush,
}

#[derive(Debug, Clone)]
pub struct NetworkOperation {
    /// Timestamp
    pub timestamp: Instant,
    /// Operation type
    pub operation: NetworkOpType,
    /// Bytes transferred
    pub bytes: usize,
    /// Duration
    pub duration: Duration,
    /// Remote endpoint
    pub endpoint: String,
}

#[derive(Debug, Clone)]
pub enum NetworkOpType {
    Send,
    Receive,
    Connect,
    Disconnect,
}

/// Profile context for current profiling session
#[derive(Debug)]
struct ProfileContext {
    /// Profile being built
    profile: Profile,
    /// Stack of function calls
    call_stack: Vec<(String, Instant)>,
    /// Active timers
    timers: HashMap<String, Instant>,
    /// Metrics buffer
    metrics_buffer: MetricsBuffer,
}

#[derive(Debug, Default)]
struct MetricsBuffer {
    /// Time samples
    time_samples: Vec<(String, Duration)>,
    /// Memory samples
    memory_samples: Vec<(Instant, usize)>,
    /// CPU samples
    cpu_samples: Vec<(Instant, f64)>,
    /// Custom metrics
    custom_metrics: HashMap<String, Vec<f64>>,
}

/// Metrics collector trait
pub trait MetricsCollector: Send + Sync {
    /// Collect metrics
    fn collect(&self) -> Result<MetricsSample, String>;

    /// Collector name
    fn name(&self) -> &str;

    /// Supported metrics
    fn supported_metrics(&self) -> Vec<MetricType>;
}

#[derive(Debug, Clone)]
pub struct MetricsSample {
    /// Timestamp
    pub timestamp: Instant,
    /// Metric values
    pub values: HashMap<MetricType, f64>,
}

/// Performance analyzer
pub struct PerformanceAnalyzer {
    /// Analysis configuration
    config: AnalysisConfig,
    /// Bottleneck detector
    bottleneck_detector: BottleneckDetector,
    /// Optimization suggester
    optimization_suggester: OptimizationSuggester,
}

#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Enable bottleneck detection
    pub detect_bottlenecks: bool,
    /// Enable optimization suggestions
    pub suggest_optimizations: bool,
    /// Anomaly detection
    pub detect_anomalies: bool,
    /// Regression detection
    pub detect_regressions: bool,
    /// Baseline comparison
    pub baseline: Option<Profile>,
}

/// Bottleneck detector
pub struct BottleneckDetector {
    /// Threshold for hot functions
    hot_function_threshold: f64,
    /// Memory leak detection
    detect_memory_leaks: bool,
    /// Contention detection
    detect_contention: bool,
}

/// Optimization suggester
pub struct OptimizationSuggester {
    /// Suggestion rules
    rules: Vec<OptimizationRule>,
    /// Historical data
    history: Vec<Profile>,
}

#[derive(Debug, Clone)]
pub struct OptimizationRule {
    /// Rule name
    pub name: String,
    /// Condition
    pub condition: RuleCondition,
    /// Suggestion
    pub suggestion: String,
    /// Potential improvement
    pub improvement: f64,
}

#[derive(Debug, Clone)]
pub enum RuleCondition {
    /// High function time
    HighFunctionTime {
        function: String,
        threshold: Duration,
    },
    /// High memory usage
    HighMemoryUsage { threshold: usize },
    /// Low cache hit rate
    LowCacheHitRate { threshold: f64 },
    /// Custom condition
    Custom(String),
}

impl PerformanceProfiler {
    /// Start real-time monitoring
    pub fn start_real_time_monitoring(&mut self) -> Result<RealTimeMonitor, String> {
        if !self.config.enabled {
            return Err("Profiling not enabled".to_string());
        }

        let monitor = RealTimeMonitor::new(
            self.config.sampling_interval,
            self.collectors
                .iter()
                .map(|c| c.name().to_string())
                .collect(),
        )?;

        Ok(monitor)
    }

    /// Predict performance characteristics
    pub fn predict_performance(
        &self,
        problem_characteristics: &ProblemCharacteristics,
    ) -> PerformancePrediction {
        let predictor = PerformancePredictor::new(&self.profiles);
        predictor.predict(problem_characteristics)
    }

    /// Generate optimization recommendations
    pub fn generate_recommendations(&self, profile: &Profile) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Analyze hot functions
        let analysis = self.analyze_profile(profile);

        for bottleneck in &analysis.bottlenecks {
            match bottleneck.bottleneck_type {
                BottleneckType::CPU => {
                    if bottleneck.impact > 0.3 {
                        recommendations.push(OptimizationRecommendation {
                            title: format!("Optimize hot function: {}", bottleneck.location),
                            description:
                                "Consider algorithmic improvements, caching, or parallelization"
                                    .to_string(),
                            category: RecommendationCategory::Algorithm,
                            impact: RecommendationImpact::High,
                            effort: ImplementationEffort::Medium,
                            estimated_improvement: bottleneck.impact * 0.5,
                            code_suggestions: vec![
                                "Add memoization for expensive calculations".to_string(),
                                "Consider parallel processing".to_string(),
                                "Profile inner loops for micro-optimizations".to_string(),
                            ],
                        });
                    }
                }
                BottleneckType::Memory => {
                    recommendations.push(OptimizationRecommendation {
                        title: "Memory usage optimization".to_string(),
                        description: "Reduce memory allocations and improve data locality"
                            .to_string(),
                        category: RecommendationCategory::Memory,
                        impact: RecommendationImpact::Medium,
                        effort: ImplementationEffort::Low,
                        estimated_improvement: 0.2,
                        code_suggestions: vec![
                            "Use object pooling for frequently allocated objects".to_string(),
                            "Consider more compact data structures".to_string(),
                            "Implement streaming for large datasets".to_string(),
                        ],
                    });
                }
                _ => {}
            }
        }

        // Add general recommendations based on metrics
        if profile.metrics.computation_metrics.cache_hit_rate < 0.8 {
            recommendations.push(OptimizationRecommendation {
                title: "Improve cache locality".to_string(),
                description: "Restructure data access patterns for better cache performance"
                    .to_string(),
                category: RecommendationCategory::Memory,
                impact: RecommendationImpact::Medium,
                effort: ImplementationEffort::High,
                estimated_improvement: 0.15,
                code_suggestions: vec![
                    "Use structure-of-arrays instead of array-of-structures".to_string(),
                    "Implement cache-oblivious algorithms".to_string(),
                    "Add data prefetching hints".to_string(),
                ],
            });
        }

        // Sort by impact
        recommendations.sort_by(|a, b| {
            b.estimated_improvement
                .partial_cmp(&a.estimated_improvement)
                .expect("Failed to compare estimated improvements in recommendation sorting")
        });

        recommendations
    }

    /// Export profile for external analysis tools
    pub fn export_for_external_tool(
        &self,
        profile: &Profile,
        tool: ExternalTool,
    ) -> Result<String, String> {
        match tool {
            ExternalTool::Perf => self.export_perf_script(profile),
            ExternalTool::Valgrind => self.export_valgrind_format(profile),
            ExternalTool::FlameScope => self.export_flamescope_format(profile),
            ExternalTool::SpeedScope => self.export_speedscope_format(profile),
        }
    }

    /// Export in perf script format
    fn export_perf_script(&self, profile: &Profile) -> Result<String, String> {
        let mut output = String::new();

        for event in &profile.events {
            if matches!(event.event_type, EventType::FunctionCall) {
                output.push_str(&format!(
                    "{} {} [{}] {}: {}\n",
                    "comm",
                    std::process::id(),
                    format!("{:?}", event.thread_id),
                    event.timestamp.elapsed().as_micros(),
                    event.name
                ));
            }
        }

        Ok(output)
    }

    /// Export in valgrind callgrind format
    fn export_valgrind_format(&self, profile: &Profile) -> Result<String, String> {
        let mut output = String::new();

        output.push_str("events: Instructions\n");
        output.push_str("summary: 1000000\n\n");

        for node in &profile.call_graph.nodes {
            output.push_str(&format!(
                "fl={}\nfn={}\n1 {}\n\n",
                "unknown",
                node.name,
                node.total_time.as_micros()
            ));
        }

        Ok(output)
    }

    /// Export in FlameScope format
    fn export_flamescope_format(&self, profile: &Profile) -> Result<String, String> {
        // Simplified FlameScope JSON format
        let mut stacks = Vec::new();

        for event in &profile.events {
            if matches!(event.event_type, EventType::FunctionCall) {
                if let Some(duration) = event.duration {
                    stacks.push(serde_json::json!({
                        "name": event.name,
                        "value": duration.as_micros(),
                        "start": event.timestamp.elapsed().as_micros()
                    }));
                }
            }
        }

        serde_json::to_string(&stacks).map_err(|e| format!("JSON error: {e}"))
    }

    /// Export in SpeedScope format
    fn export_speedscope_format(&self, profile: &Profile) -> Result<String, String> {
        let speedscope_profile = serde_json::json!({
            "$schema": "https://www.speedscope.app/file-format-schema.json",
            "version": "0.0.1",
            "shared": {
                "frames": profile.call_graph.nodes.iter().map(|node| {
                    serde_json::json!({
                        "name": node.name,
                        "file": "unknown",
                        "line": 0,
                        "col": 0
                    })
                }).collect::<Vec<_>>()
            },
            "profiles": [{
                "type": "evented",
                "name": profile.id,
                "unit": "microseconds",
                "startValue": 0,
                "endValue": profile.metrics.time_metrics.total_time.as_micros(),
                "events": profile.events.iter().filter_map(|event| {
                    match event.event_type {
                        EventType::FunctionCall => Some(serde_json::json!({
                            "type": "O",
                            "at": event.timestamp.elapsed().as_micros(),
                            "frame": event.name
                        })),
                        EventType::FunctionReturn => Some(serde_json::json!({
                            "type": "C",
                            "at": event.timestamp.elapsed().as_micros(),
                            "frame": event.name
                        })),
                        _ => None
                    }
                }).collect::<Vec<_>>()
            }]
        });

        serde_json::to_string(&speedscope_profile).map_err(|e| format!("JSON error: {e}"))
    }

    /// Continuous profiling mode
    pub fn start_continuous_profiling(
        &mut self,
        duration: Duration,
    ) -> Result<ContinuousProfiler, String> {
        if !self.config.enabled {
            return Err("Profiling not enabled".to_string());
        }

        let profiler = ContinuousProfiler::new(duration, self.config.sampling_interval);
        Ok(profiler)
    }

    /// Benchmark comparison
    pub fn benchmark_compare(&self, profiles: &[Profile]) -> BenchmarkComparison {
        let mut comparison = BenchmarkComparison {
            profiles: profiles.iter().map(|p| p.id.clone()).collect(),
            metrics_comparison: Vec::new(),
            regression_analysis: Vec::new(),
            performance_trends: Vec::new(),
        };

        if profiles.len() < 2 {
            return comparison;
        }

        // Compare total times
        let times: Vec<f64> = profiles
            .iter()
            .map(|p| p.metrics.time_metrics.total_time.as_secs_f64())
            .collect();

        comparison.metrics_comparison.push(MetricComparison {
            metric_name: "total_time".to_string(),
            values: times.clone(),
            trend: if times.len() >= 2 {
                if times[times.len() - 1] < times[0] {
                    PerformanceTrend::Improving
                } else if times[times.len() - 1] > times[0] * 1.1 {
                    PerformanceTrend::Degrading
                } else {
                    PerformanceTrend::Stable
                }
            } else {
                PerformanceTrend::Unknown
            },
            variance: self.calculate_variance(&times),
        });

        // Compare memory usage
        let memory: Vec<f64> = profiles
            .iter()
            .map(|p| p.metrics.memory_metrics.peak_memory as f64)
            .collect();

        comparison.metrics_comparison.push(MetricComparison {
            metric_name: "peak_memory".to_string(),
            values: memory.clone(),
            trend: if memory.len() >= 2 {
                if memory[memory.len() - 1] < memory[0] {
                    PerformanceTrend::Improving
                } else if memory[memory.len() - 1] > memory[0] * 1.1 {
                    PerformanceTrend::Degrading
                } else {
                    PerformanceTrend::Stable
                }
            } else {
                PerformanceTrend::Unknown
            },
            variance: self.calculate_variance(&memory),
        });

        comparison
    }

    /// Calculate variance
    fn calculate_variance(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;

        variance.sqrt()
    }
    /// Create new profiler
    pub fn new(config: ProfilerConfig) -> Self {
        Self {
            config,
            profiles: Vec::new(),
            current_profile: None,
            collectors: Self::default_collectors(),
            analyzer: PerformanceAnalyzer::new(AnalysisConfig {
                detect_bottlenecks: true,
                suggest_optimizations: true,
                detect_anomalies: true,
                detect_regressions: false,
                baseline: None,
            }),
        }
    }

    /// Get default collectors
    fn default_collectors() -> Vec<Box<dyn MetricsCollector>> {
        vec![
            Box::new(TimeCollector),
            Box::new(MemoryCollector),
            Box::new(CPUCollector),
        ]
    }

    /// Start profiling
    pub fn start_profile(&mut self, name: &str) -> Result<(), String> {
        if !self.config.enabled {
            return Ok(());
        }

        if self.current_profile.is_some() {
            return Err("Profile already in progress".to_string());
        }

        let profile = Profile {
            id: format!(
                "{}_{}",
                name,
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("Failed to get current system time for profile ID generation")
                    .as_secs()
            ),
            start_time: Instant::now(),
            end_time: None,
            events: Vec::new(),
            metrics: MetricsData {
                time_metrics: TimeMetrics {
                    total_time: Duration::from_secs(0),
                    qubo_generation_time: Duration::from_secs(0),
                    compilation_time: Duration::from_secs(0),
                    solving_time: Duration::from_secs(0),
                    post_processing_time: Duration::from_secs(0),
                    function_times: BTreeMap::new(),
                    percentiles: Percentiles {
                        p50: Duration::from_secs(0),
                        p90: Duration::from_secs(0),
                        p95: Duration::from_secs(0),
                        p99: Duration::from_secs(0),
                        p999: Duration::from_secs(0),
                    },
                },
                memory_metrics: MemoryMetrics {
                    peak_memory: 0,
                    avg_memory: 0,
                    allocations: 0,
                    deallocations: 0,
                    largest_allocation: 0,
                    memory_timeline: Vec::new(),
                },
                computation_metrics: ComputationMetrics {
                    flops: 0.0,
                    memory_bandwidth: 0.0,
                    cache_hit_rate: 0.0,
                    branch_prediction_accuracy: 0.0,
                    vectorization_efficiency: 0.0,
                },
                quality_metrics: QualityMetrics {
                    quality_timeline: Vec::new(),
                    convergence_rate: 0.0,
                    improvement_per_iteration: 0.0,
                    time_to_first_solution: Duration::from_secs(0),
                    time_to_best_solution: Duration::from_secs(0),
                },
            },
            call_graph: CallGraph {
                nodes: Vec::new(),
                edges: Vec::new(),
                roots: Vec::new(),
            },
            resource_usage: ResourceUsage {
                cpu_usage: Vec::new(),
                memory_usage: Vec::new(),
                gpu_usage: Vec::new(),
                io_operations: Vec::new(),
                network_operations: Vec::new(),
            },
        };

        self.current_profile = Some(ProfileContext {
            profile,
            call_stack: Vec::new(),
            timers: HashMap::new(),
            metrics_buffer: MetricsBuffer::default(),
        });

        // Start metrics collection thread
        if self.config.sampling_interval > Duration::from_secs(0) {
            self.start_metrics_collection()?;
        }

        Ok(())
    }

    /// Stop profiling
    pub fn stop_profile(&mut self) -> Result<Profile, String> {
        if !self.config.enabled {
            return Err("Profiling not enabled".to_string());
        }

        let mut context = self
            .current_profile
            .take()
            .ok_or("No profile in progress")?;

        context.profile.end_time = Some(Instant::now());

        // Calculate total time
        context.profile.metrics.time_metrics.total_time = context
            .profile
            .end_time
            .expect("Profile end_time should be set before calculating total time")
            - context.profile.start_time;

        // Process metrics buffer
        self.process_metrics_buffer(&mut context)?;

        // Build call graph
        self.build_call_graph(&mut context.profile)?;

        // Calculate percentiles
        self.calculate_percentiles(&mut context.profile)?;

        // Store profile
        self.profiles.push(context.profile.clone());

        Ok(context.profile)
    }

    /// Record function entry
    pub fn enter_function(&mut self, name: &str) -> FunctionGuard {
        if !self.config.enabled {
            return FunctionGuard {
                profiler: None,
                name: String::new(),
            };
        }

        if let Some(ref mut context) = self.current_profile {
            let event = ProfileEvent {
                timestamp: Instant::now(),
                event_type: EventType::FunctionCall,
                name: name.to_string(),
                duration: None,
                data: HashMap::new(),
                thread_id: thread::current().id(),
            };

            context.profile.events.push(event);
            context.call_stack.push((name.to_string(), Instant::now()));
        }

        FunctionGuard {
            profiler: Some(std::ptr::from_mut::<Self>(self)),
            name: name.to_string(),
        }
    }

    /// Record function exit
    fn exit_function(&mut self, name: &str) {
        if let Some(ref mut context) = self.current_profile {
            if let Some((_, enter_time)) = context.call_stack.pop() {
                let duration = enter_time.elapsed();

                let event = ProfileEvent {
                    timestamp: Instant::now(),
                    event_type: EventType::FunctionReturn,
                    name: name.to_string(),
                    duration: Some(duration),
                    data: HashMap::new(),
                    thread_id: thread::current().id(),
                };

                context.profile.events.push(event);

                // Update function times
                *context
                    .profile
                    .metrics
                    .time_metrics
                    .function_times
                    .entry(name.to_string())
                    .or_insert(Duration::from_secs(0)) += duration;
            }
        }
    }

    /// Start timer
    pub fn start_timer(&mut self, name: &str) {
        if !self.config.enabled {
            return;
        }

        if let Some(ref mut context) = self.current_profile {
            context.timers.insert(name.to_string(), Instant::now());
        }
    }

    /// Stop timer
    pub fn stop_timer(&mut self, name: &str) -> Option<Duration> {
        if !self.config.enabled {
            return None;
        }

        if let Some(ref mut context) = self.current_profile {
            if let Some(start_time) = context.timers.remove(name) {
                let duration = start_time.elapsed();

                // Store in appropriate metric
                match name {
                    "qubo_generation" => {
                        context.profile.metrics.time_metrics.qubo_generation_time = duration;
                    }
                    "compilation" => {
                        context.profile.metrics.time_metrics.compilation_time = duration;
                    }
                    "solving" => {
                        context.profile.metrics.time_metrics.solving_time = duration;
                    }
                    "post_processing" => {
                        context.profile.metrics.time_metrics.post_processing_time = duration;
                    }
                    _ => {
                        context
                            .metrics_buffer
                            .time_samples
                            .push((name.to_string(), duration));
                    }
                }

                return Some(duration);
            }
        }

        None
    }

    /// Record memory allocation
    pub fn record_allocation(&mut self, size: usize) {
        if !self.config.enabled || !self.config.profile_memory {
            return;
        }

        if let Some(ref mut context) = self.current_profile {
            context.profile.metrics.memory_metrics.allocations += 1;
            context.profile.metrics.memory_metrics.largest_allocation = context
                .profile
                .metrics
                .memory_metrics
                .largest_allocation
                .max(size);

            let event = ProfileEvent {
                timestamp: Instant::now(),
                event_type: EventType::MemoryAlloc,
                name: "allocation".to_string(),
                duration: None,
                data: {
                    let mut data = HashMap::new();
                    data.insert("size".to_string(), size.to_string());
                    data
                },
                thread_id: thread::current().id(),
            };

            context.profile.events.push(event);
        }
    }

    /// Record memory deallocation
    pub fn record_deallocation(&mut self, size: usize) {
        if !self.config.enabled || !self.config.profile_memory {
            return;
        }

        if let Some(ref mut context) = self.current_profile {
            context.profile.metrics.memory_metrics.deallocations += 1;

            let event = ProfileEvent {
                timestamp: Instant::now(),
                event_type: EventType::MemoryFree,
                name: "deallocation".to_string(),
                duration: None,
                data: {
                    let mut data = HashMap::new();
                    data.insert("size".to_string(), size.to_string());
                    data
                },
                thread_id: thread::current().id(),
            };

            context.profile.events.push(event);
        }
    }

    /// Record solution quality
    pub fn record_solution_quality(&mut self, quality: f64) {
        if !self.config.enabled {
            return;
        }

        if let Some(ref mut context) = self.current_profile {
            let elapsed = context.profile.start_time.elapsed();
            context
                .profile
                .metrics
                .quality_metrics
                .quality_timeline
                .push((elapsed, quality));

            // Update time to first solution
            if context
                .profile
                .metrics
                .quality_metrics
                .quality_timeline
                .len()
                == 1
            {
                context
                    .profile
                    .metrics
                    .quality_metrics
                    .time_to_first_solution = elapsed;
            }
        }
    }

    /// Start metrics collection
    const fn start_metrics_collection(&self) -> Result<(), String> {
        // This would spawn a thread to collect metrics periodically
        // Simplified for now
        Ok(())
    }

    /// Process metrics buffer
    fn process_metrics_buffer(&self, context: &mut ProfileContext) -> Result<(), String> {
        // Calculate memory statistics
        if !context.metrics_buffer.memory_samples.is_empty() {
            let total_memory: usize = context
                .metrics_buffer
                .memory_samples
                .iter()
                .map(|(_, mem)| mem)
                .sum();
            context.profile.metrics.memory_metrics.avg_memory =
                total_memory / context.metrics_buffer.memory_samples.len();

            context.profile.metrics.memory_metrics.peak_memory = context
                .metrics_buffer
                .memory_samples
                .iter()
                .map(|(_, mem)| *mem)
                .max()
                .unwrap_or(0);

            context.profile.metrics.memory_metrics.memory_timeline =
                context.metrics_buffer.memory_samples.clone();
        }

        // Calculate CPU statistics
        if !context.metrics_buffer.cpu_samples.is_empty() {
            context.profile.resource_usage.cpu_usage = context.metrics_buffer.cpu_samples.clone();
        }

        Ok(())
    }

    /// Build call graph
    fn build_call_graph(&self, profile: &mut Profile) -> Result<(), String> {
        let mut node_map: HashMap<String, usize> = HashMap::new();
        let mut nodes = Vec::new();
        let mut edges: HashMap<(usize, usize), CallEdge> = HashMap::new();

        // Create nodes
        for (func_name, &total_time) in &profile.metrics.time_metrics.function_times {
            let node_id = nodes.len();
            node_map.insert(func_name.clone(), node_id);

            nodes.push(CallNode {
                id: node_id,
                name: func_name.clone(),
                total_time,
                self_time: total_time,            // Will be adjusted
                call_count: 0,                    // Will be counted
                avg_time: Duration::from_secs(0), // Will be calculated
            });
        }

        // Count calls and build edges
        let mut call_stack: Vec<usize> = Vec::new();

        for event in &profile.events {
            match event.event_type {
                EventType::FunctionCall => {
                    if let Some(&node_id) = node_map.get(&event.name) {
                        nodes[node_id].call_count += 1;

                        if let Some(&parent_id) = call_stack.last() {
                            let edge_key = (parent_id, node_id);
                            edges
                                .entry(edge_key)
                                .and_modify(|e| e.call_count += 1)
                                .or_insert(CallEdge {
                                    from: parent_id,
                                    to: node_id,
                                    call_count: 1,
                                    total_time: Duration::from_secs(0),
                                });
                        }

                        call_stack.push(node_id);
                    }
                }
                EventType::FunctionReturn => {
                    call_stack.pop();
                }
                _ => {}
            }
        }

        // Calculate average times
        for node in &mut nodes {
            if node.call_count > 0 {
                node.avg_time = node.total_time / node.call_count as u32;
            }
        }

        // Find root nodes
        let mut has_parent = vec![false; nodes.len()];
        for edge in edges.values() {
            has_parent[edge.to] = true;
        }

        let roots: Vec<usize> = (0..nodes.len()).filter(|&i| !has_parent[i]).collect();

        profile.call_graph = CallGraph {
            nodes,
            edges: edges.into_values().collect(),
            roots,
        };

        Ok(())
    }

    /// Calculate percentiles
    fn calculate_percentiles(&self, profile: &mut Profile) -> Result<(), String> {
        let mut durations: Vec<Duration> =
            profile.events.iter().filter_map(|e| e.duration).collect();

        if durations.is_empty() {
            return Ok(());
        }

        durations.sort();

        let len = durations.len();
        profile.metrics.time_metrics.percentiles = Percentiles {
            p50: durations[len * 50 / 100],
            p90: durations[len * 90 / 100],
            p95: durations[len * 95 / 100],
            p99: durations[len * 99 / 100],
            p999: durations[len.saturating_sub(1)],
        };

        Ok(())
    }

    /// Analyze profile
    pub fn analyze_profile(&self, profile: &Profile) -> AnalysisReport {
        self.analyzer.analyze(profile)
    }

    /// Compare profiles
    pub fn compare_profiles(&self, profile1: &Profile, profile2: &Profile) -> ComparisonReport {
        ComparisonReport {
            time_comparison: self.compare_time_metrics(
                &profile1.metrics.time_metrics,
                &profile2.metrics.time_metrics,
            ),
            memory_comparison: self.compare_memory_metrics(
                &profile1.metrics.memory_metrics,
                &profile2.metrics.memory_metrics,
            ),
            quality_comparison: self.compare_quality_metrics(
                &profile1.metrics.quality_metrics,
                &profile2.metrics.quality_metrics,
            ),
            regressions: Vec::new(),
            improvements: Vec::new(),
        }
    }

    /// Compare time metrics
    fn compare_time_metrics(&self, m1: &TimeMetrics, m2: &TimeMetrics) -> TimeComparison {
        TimeComparison {
            total_time_diff: m2.total_time.as_secs_f64() - m1.total_time.as_secs_f64(),
            total_time_ratio: m2.total_time.as_secs_f64() / m1.total_time.as_secs_f64(),
            qubo_time_diff: m2.qubo_generation_time.as_secs_f64()
                - m1.qubo_generation_time.as_secs_f64(),
            solving_time_diff: m2.solving_time.as_secs_f64() - m1.solving_time.as_secs_f64(),
            function_diffs: BTreeMap::new(), // TODO: implement
        }
    }

    /// Compare memory metrics
    fn compare_memory_metrics(&self, m1: &MemoryMetrics, m2: &MemoryMetrics) -> MemoryComparison {
        MemoryComparison {
            peak_memory_diff: m2.peak_memory as i64 - m1.peak_memory as i64,
            peak_memory_ratio: m2.peak_memory as f64 / m1.peak_memory as f64,
            avg_memory_diff: m2.avg_memory as i64 - m1.avg_memory as i64,
            allocations_diff: m2.allocations as i64 - m1.allocations as i64,
        }
    }

    /// Compare quality metrics
    fn compare_quality_metrics(
        &self,
        m1: &QualityMetrics,
        m2: &QualityMetrics,
    ) -> QualityComparison {
        QualityComparison {
            convergence_rate_diff: m2.convergence_rate - m1.convergence_rate,
            time_to_best_diff: m2.time_to_best_solution.as_secs_f64()
                - m1.time_to_best_solution.as_secs_f64(),
            final_quality_diff: 0.0, // TODO: implement
        }
    }

    /// Generate report
    pub fn generate_report(
        &self,
        profile: &Profile,
        format: &OutputFormat,
    ) -> Result<String, String> {
        match format {
            OutputFormat::Json => self.generate_json_report(profile),
            OutputFormat::Csv => self.generate_csv_report(profile),
            OutputFormat::FlameGraph => self.generate_flame_graph(profile),
            OutputFormat::ChromeTrace => self.generate_chrome_trace(profile),
            _ => Err("Format not implemented".to_string()),
        }
    }

    /// Generate JSON report
    fn generate_json_report(&self, profile: &Profile) -> Result<String, String> {
        use std::fmt::Write;

        let mut json = String::new();

        // Build comprehensive JSON report
        json.push_str("{\n");

        // Profile metadata
        json.push_str("  \"metadata\": {\n");
        writeln!(&mut json, "    \"id\": \"{}\",", profile.id)
            .expect("Failed to write profile ID to JSON report");
        writeln!(
            &mut json,
            "    \"start_time\": {},",
            profile.start_time.elapsed().as_millis()
        )
        .expect("Failed to write start_time to JSON report");
        if let Some(end_time) = profile.end_time {
            writeln!(
                &mut json,
                "    \"end_time\": {},",
                end_time.elapsed().as_millis()
            )
            .expect("Failed to write end_time to JSON report");
        }
        json.push_str("    \"duration_ms\": ");
        write!(
            &mut json,
            "{}",
            profile.metrics.time_metrics.total_time.as_millis()
        )
        .expect("Failed to write duration_ms to JSON report");
        json.push_str("\n  },\n");

        // Time metrics
        json.push_str("  \"time_metrics\": {\n");
        writeln!(
            &mut json,
            "    \"total_time_ms\": {},",
            profile.metrics.time_metrics.total_time.as_millis()
        )
        .expect("Failed to write total_time_ms to JSON report");
        writeln!(
            &mut json,
            "    \"qubo_generation_ms\": {},",
            profile
                .metrics
                .time_metrics
                .qubo_generation_time
                .as_millis()
        )
        .expect("Failed to write qubo_generation_ms to JSON report");
        writeln!(
            &mut json,
            "    \"compilation_ms\": {},",
            profile.metrics.time_metrics.compilation_time.as_millis()
        )
        .expect("Failed to write compilation_ms to JSON report");
        writeln!(
            &mut json,
            "    \"solving_ms\": {},",
            profile.metrics.time_metrics.solving_time.as_millis()
        )
        .expect("Failed to write solving_ms to JSON report");
        writeln!(
            &mut json,
            "    \"post_processing_ms\": {},",
            profile
                .metrics
                .time_metrics
                .post_processing_time
                .as_millis()
        )
        .expect("Failed to write post_processing_ms to JSON report");

        // Function times
        json.push_str("    \"function_times\": {\n");
        let func_entries: Vec<_> = profile.metrics.time_metrics.function_times.iter().collect();
        for (i, (func, time)) in func_entries.iter().enumerate() {
            write!(&mut json, "      \"{}\": {}", func, time.as_millis())
                .expect("Failed to write function time entry to JSON report");
            if i < func_entries.len() - 1 {
                json.push(',');
            }
            json.push('\n');
        }
        json.push_str("    },\n");

        // Percentiles
        json.push_str("    \"percentiles_ms\": {\n");
        writeln!(
            &mut json,
            "      \"p50\": {},",
            profile.metrics.time_metrics.percentiles.p50.as_millis()
        )
        .expect("Failed to write p50 percentile to JSON report");
        writeln!(
            &mut json,
            "      \"p90\": {},",
            profile.metrics.time_metrics.percentiles.p90.as_millis()
        )
        .expect("Failed to write p90 percentile to JSON report");
        writeln!(
            &mut json,
            "      \"p95\": {},",
            profile.metrics.time_metrics.percentiles.p95.as_millis()
        )
        .expect("Failed to write p95 percentile to JSON report");
        writeln!(
            &mut json,
            "      \"p99\": {},",
            profile.metrics.time_metrics.percentiles.p99.as_millis()
        )
        .expect("Failed to write p99 percentile to JSON report");
        writeln!(
            &mut json,
            "      \"p999\": {}",
            profile.metrics.time_metrics.percentiles.p999.as_millis()
        )
        .expect("Failed to write p999 percentile to JSON report");
        json.push_str("    }\n");
        json.push_str("  },\n");

        // Memory metrics
        json.push_str("  \"memory_metrics\": {\n");
        writeln!(
            &mut json,
            "    \"peak_memory_bytes\": {},",
            profile.metrics.memory_metrics.peak_memory
        )
        .expect("Failed to write peak_memory_bytes to JSON report");
        writeln!(
            &mut json,
            "    \"avg_memory_bytes\": {},",
            profile.metrics.memory_metrics.avg_memory
        )
        .expect("Failed to write avg_memory_bytes to JSON report");
        writeln!(
            &mut json,
            "    \"allocations\": {},",
            profile.metrics.memory_metrics.allocations
        )
        .expect("Failed to write allocations to JSON report");
        writeln!(
            &mut json,
            "    \"deallocations\": {},",
            profile.metrics.memory_metrics.deallocations
        )
        .expect("Failed to write deallocations to JSON report");
        writeln!(
            &mut json,
            "    \"largest_allocation_bytes\": {}",
            profile.metrics.memory_metrics.largest_allocation
        )
        .expect("Failed to write largest_allocation_bytes to JSON report");
        json.push_str("  },\n");

        // Computation metrics
        json.push_str("  \"computation_metrics\": {\n");
        writeln!(
            &mut json,
            "    \"flops\": {},",
            profile.metrics.computation_metrics.flops
        )
        .expect("Failed to write flops to JSON report");
        writeln!(
            &mut json,
            "    \"memory_bandwidth_gbps\": {},",
            profile.metrics.computation_metrics.memory_bandwidth
        )
        .expect("Failed to write memory_bandwidth_gbps to JSON report");
        writeln!(
            &mut json,
            "    \"cache_hit_rate\": {},",
            profile.metrics.computation_metrics.cache_hit_rate
        )
        .expect("Failed to write cache_hit_rate to JSON report");
        writeln!(
            &mut json,
            "    \"branch_prediction_accuracy\": {},",
            profile
                .metrics
                .computation_metrics
                .branch_prediction_accuracy
        )
        .expect("Failed to write branch_prediction_accuracy to JSON report");
        writeln!(
            &mut json,
            "    \"vectorization_efficiency\": {}",
            profile.metrics.computation_metrics.vectorization_efficiency
        )
        .expect("Failed to write vectorization_efficiency to JSON report");
        json.push_str("  },\n");

        // Quality metrics
        json.push_str("  \"quality_metrics\": {\n");
        writeln!(
            &mut json,
            "    \"convergence_rate\": {},",
            profile.metrics.quality_metrics.convergence_rate
        )
        .expect("Failed to write convergence_rate to JSON report");
        writeln!(
            &mut json,
            "    \"improvement_per_iteration\": {},",
            profile.metrics.quality_metrics.improvement_per_iteration
        )
        .expect("Failed to write improvement_per_iteration to JSON report");
        writeln!(
            &mut json,
            "    \"time_to_first_solution_ms\": {},",
            profile
                .metrics
                .quality_metrics
                .time_to_first_solution
                .as_millis()
        )
        .expect("Failed to write time_to_first_solution_ms to JSON report");
        writeln!(
            &mut json,
            "    \"time_to_best_solution_ms\": {},",
            profile
                .metrics
                .quality_metrics
                .time_to_best_solution
                .as_millis()
        )
        .expect("Failed to write time_to_best_solution_ms to JSON report");

        // Quality timeline
        json.push_str("    \"quality_timeline\": [\n");
        for (i, (time, quality)) in profile
            .metrics
            .quality_metrics
            .quality_timeline
            .iter()
            .enumerate()
        {
            json.push_str("      {\n");
            writeln!(&mut json, "        \"time_ms\": {},", time.as_millis())
                .expect("Failed to write quality timeline time_ms to JSON report");
            writeln!(&mut json, "        \"quality\": {quality}")
                .expect("Failed to write quality timeline quality value to JSON report");
            json.push_str("      }");
            if i < profile.metrics.quality_metrics.quality_timeline.len() - 1 {
                json.push(',');
            }
            json.push('\n');
        }
        json.push_str("    ]\n");
        json.push_str("  },\n");

        // Call graph
        json.push_str("  \"call_graph\": {\n");
        json.push_str("    \"nodes\": [\n");
        for (i, node) in profile.call_graph.nodes.iter().enumerate() {
            json.push_str("      {\n");
            writeln!(&mut json, "        \"id\": {},", node.id)
                .expect("Failed to write call graph node id to JSON report");
            writeln!(
                &mut json,
                "        \"name\": \"{}\",",
                node.name.replace('"', "\\\"")
            )
            .expect("Failed to write call graph node name to JSON report");
            writeln!(
                &mut json,
                "        \"total_time_ms\": {},",
                node.total_time.as_millis()
            )
            .expect("Failed to write call graph node total_time_ms to JSON report");
            writeln!(
                &mut json,
                "        \"self_time_ms\": {},",
                node.self_time.as_millis()
            )
            .expect("Failed to write call graph node self_time_ms to JSON report");
            writeln!(&mut json, "        \"call_count\": {},", node.call_count)
                .expect("Failed to write call graph node call_count to JSON report");
            writeln!(
                &mut json,
                "        \"avg_time_ms\": {}",
                node.avg_time.as_millis()
            )
            .expect("Failed to write call graph node avg_time_ms to JSON report");
            json.push_str("      }");
            if i < profile.call_graph.nodes.len() - 1 {
                json.push(',');
            }
            json.push('\n');
        }
        json.push_str("    ],\n");

        json.push_str("    \"edges\": [\n");
        for (i, edge) in profile.call_graph.edges.iter().enumerate() {
            json.push_str("      {\n");
            writeln!(&mut json, "        \"from\": {},", edge.from)
                .expect("Failed to write call graph edge from to JSON report");
            writeln!(&mut json, "        \"to\": {},", edge.to)
                .expect("Failed to write call graph edge to to JSON report");
            writeln!(&mut json, "        \"call_count\": {},", edge.call_count)
                .expect("Failed to write call graph edge call_count to JSON report");
            writeln!(
                &mut json,
                "        \"total_time_ms\": {}",
                edge.total_time.as_millis()
            )
            .expect("Failed to write call graph edge total_time_ms to JSON report");
            json.push_str("      }");
            if i < profile.call_graph.edges.len() - 1 {
                json.push(',');
            }
            json.push('\n');
        }
        json.push_str("    ]\n");
        json.push_str("  },\n");

        // Events summary
        json.push_str("  \"events_summary\": {\n");
        writeln!(&mut json, "    \"total_events\": {},", profile.events.len())
            .expect("Failed to write total_events to JSON report");

        // Count events by type
        let mut event_counts = std::collections::BTreeMap::new();
        for event in &profile.events {
            let type_name = match &event.event_type {
                EventType::FunctionCall => "function_call",
                EventType::FunctionReturn => "function_return",
                EventType::MemoryAlloc => "memory_alloc",
                EventType::MemoryFree => "memory_free",
                EventType::IOOperation => "io_operation",
                EventType::Synchronization => "synchronization",
                EventType::Custom(name) => name,
            };
            *event_counts.entry(type_name).or_insert(0) += 1;
        }

        json.push_str("    \"event_counts\": {\n");
        let count_entries: Vec<_> = event_counts.iter().collect();
        for (i, (event_type, count)) in count_entries.iter().enumerate() {
            write!(&mut json, "      \"{event_type}\": {count}")
                .expect("Failed to write event count entry to JSON report");
            if i < count_entries.len() - 1 {
                json.push(',');
            }
            json.push('\n');
        }
        json.push_str("    }\n");
        json.push_str("  }\n");

        json.push_str("}\n");

        Ok(json)
    }

    /// Generate CSV report
    fn generate_csv_report(&self, profile: &Profile) -> Result<String, String> {
        let mut csv = String::new();

        csv.push_str("function,total_time_ms,call_count,avg_time_ms\n");

        for node in &profile.call_graph.nodes {
            csv.push_str(&format!(
                "{},{},{},{}\n",
                node.name,
                node.total_time.as_millis(),
                node.call_count,
                node.avg_time.as_millis()
            ));
        }

        Ok(csv)
    }

    /// Generate flame graph
    fn generate_flame_graph(&self, profile: &Profile) -> Result<String, String> {
        // Simplified flame graph generation
        let mut stacks = Vec::new();

        for node in &profile.call_graph.nodes {
            let stack = vec![node.name.clone()];
            let value = node.self_time.as_micros() as usize;
            stacks.push((stack, value));
        }

        // Would use inferno crate for actual flame graph generation
        Ok(format!("Flame graph with {} stacks", stacks.len()))
    }

    /// Generate Chrome trace format
    fn generate_chrome_trace(&self, profile: &Profile) -> Result<String, String> {
        #[derive(Serialize)]
        struct TraceEvent {
            name: String,
            cat: String,
            ph: String,
            ts: u64,
            dur: Option<u64>,
            pid: u32,
            tid: String,
        }

        let mut events = Vec::new();

        for event in &profile.events {
            let trace_event = TraceEvent {
                name: event.name.clone(),
                cat: "function".to_string(),
                ph: match event.event_type {
                    EventType::FunctionCall => "B".to_string(),
                    EventType::FunctionReturn => "E".to_string(),
                    _ => "i".to_string(),
                },
                ts: event.timestamp.elapsed().as_micros() as u64,
                dur: event.duration.map(|d| d.as_micros() as u64),
                pid: std::process::id(),
                tid: format!("{:?}", event.thread_id),
            };

            events.push(trace_event);
        }

        serde_json::to_string(&events).map_err(|e| format!("JSON serialization error: {e}"))
    }

    /// Visualize profile
    #[cfg(feature = "plotters")]
    pub fn visualize_profile(&self, profile: &Profile, output_path: &str) -> Result<(), String> {
        let root = BitMapBackend::new(output_path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)
            .map_err(|e| format!("Drawing error: {e}"))?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Performance Profile", ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(
                0f64..profile.metrics.time_metrics.total_time.as_secs_f64(),
                0f64..100f64,
            )
            .map_err(|e| format!("Chart error: {e}"))?;

        chart
            .configure_mesh()
            .draw()
            .map_err(|e| format!("Mesh error: {e}"))?;

        // Plot CPU usage
        if !profile.resource_usage.cpu_usage.is_empty() {
            let cpu_data: Vec<(f64, f64)> = profile
                .resource_usage
                .cpu_usage
                .iter()
                .map(|(t, usage)| (t.duration_since(profile.start_time).as_secs_f64(), *usage))
                .collect();

            chart
                .draw_series(LineSeries::new(cpu_data, &RED))
                .map_err(|e| format!("Series error: {e}"))?
                .label("CPU Usage")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], RED));
        }

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()
            .map_err(|e| format!("Legend error: {e}"))?;

        root.present().map_err(|e| format!("Present error: {e}"))?;

        Ok(())
    }
}

/// RAII guard for function profiling
pub struct FunctionGuard {
    profiler: Option<*mut PerformanceProfiler>,
    name: String,
}

impl Drop for FunctionGuard {
    fn drop(&mut self) {
        if let Some(profiler_ptr) = self.profiler {
            unsafe {
                (*profiler_ptr).exit_function(&self.name);
            }
        }
    }
}

unsafe impl Send for FunctionGuard {}

impl PerformanceAnalyzer {
    /// Create new analyzer
    pub fn new(config: AnalysisConfig) -> Self {
        Self {
            config,
            bottleneck_detector: BottleneckDetector {
                hot_function_threshold: 0.1, // 10% of total time
                detect_memory_leaks: true,
                detect_contention: true,
            },
            optimization_suggester: OptimizationSuggester {
                rules: Self::default_optimization_rules(),
                history: Vec::new(),
            },
        }
    }

    /// Default optimization rules
    fn default_optimization_rules() -> Vec<OptimizationRule> {
        vec![
            OptimizationRule {
                name: "Hot function optimization".to_string(),
                condition: RuleCondition::HighFunctionTime {
                    function: "any".to_string(),
                    threshold: Duration::from_millis(100),
                },
                suggestion: "Consider optimizing this function or caching results".to_string(),
                improvement: 0.2,
            },
            OptimizationRule {
                name: "Memory optimization".to_string(),
                condition: RuleCondition::HighMemoryUsage {
                    threshold: 1024 * 1024 * 1024, // 1GB
                },
                suggestion: "Consider using more memory-efficient data structures".to_string(),
                improvement: 0.15,
            },
            OptimizationRule {
                name: "Cache optimization".to_string(),
                condition: RuleCondition::LowCacheHitRate { threshold: 0.8 },
                suggestion: "Consider improving data locality or cache-friendly algorithms"
                    .to_string(),
                improvement: 0.1,
            },
        ]
    }

    /// Analyze profile
    pub fn analyze(&self, profile: &Profile) -> AnalysisReport {
        let mut report = AnalysisReport {
            bottlenecks: Vec::new(),
            optimizations: Vec::new(),
            anomalies: Vec::new(),
            summary: AnalysisSummary {
                total_time: profile.metrics.time_metrics.total_time,
                peak_memory: profile.metrics.memory_metrics.peak_memory,
                hot_functions: Vec::new(),
                critical_path: Vec::new(),
            },
        };

        // Detect bottlenecks
        if self.config.detect_bottlenecks {
            report.bottlenecks = self.detect_bottlenecks(profile);
        }

        // Suggest optimizations
        if self.config.suggest_optimizations {
            report.optimizations = self.suggest_optimizations(profile);
        }

        // Detect anomalies
        if self.config.detect_anomalies {
            report.anomalies = self.detect_anomalies(profile);
        }

        // Find hot functions
        report.summary.hot_functions = self.find_hot_functions(profile);

        // Find critical path
        report.summary.critical_path = self.find_critical_path(profile);

        report
    }

    /// Detect bottlenecks
    fn detect_bottlenecks(&self, profile: &Profile) -> Vec<Bottleneck> {
        let mut bottlenecks = Vec::new();

        // CPU bottlenecks
        for node in &profile.call_graph.nodes {
            let time_percentage = node.total_time.as_secs_f64()
                / profile.metrics.time_metrics.total_time.as_secs_f64();

            if time_percentage > self.bottleneck_detector.hot_function_threshold {
                bottlenecks.push(Bottleneck {
                    bottleneck_type: BottleneckType::CPU,
                    location: node.name.clone(),
                    severity: if time_percentage > 0.5 {
                        Severity::High
                    } else if time_percentage > 0.3 {
                        Severity::Medium
                    } else {
                        Severity::Low
                    },
                    impact: time_percentage,
                    description: format!(
                        "Function uses {:.1}% of total time",
                        time_percentage * 100.0
                    ),
                });
            }
        }

        // Memory bottlenecks
        if self.bottleneck_detector.detect_memory_leaks {
            let alloc_dealloc_diff = profile.metrics.memory_metrics.allocations as i64
                - profile.metrics.memory_metrics.deallocations as i64;

            if alloc_dealloc_diff > 1000 {
                bottlenecks.push(Bottleneck {
                    bottleneck_type: BottleneckType::Memory,
                    location: "global".to_string(),
                    severity: Severity::High,
                    impact: alloc_dealloc_diff as f64
                        / profile.metrics.memory_metrics.allocations as f64,
                    description: format!(
                        "Potential memory leak: {alloc_dealloc_diff} unfreed allocations"
                    ),
                });
            }
        }

        bottlenecks
    }

    /// Suggest optimizations
    fn suggest_optimizations(&self, profile: &Profile) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();

        for rule in &self.optimization_suggester.rules {
            if self.check_rule_condition(&rule.condition, profile) {
                suggestions.push(OptimizationSuggestion {
                    title: rule.name.clone(),
                    description: rule.suggestion.clone(),
                    expected_improvement: rule.improvement,
                    implementation_effort: ImplementationEffort::Medium,
                    priority: Priority::High,
                });
            }
        }

        suggestions
    }

    /// Check rule condition
    fn check_rule_condition(&self, condition: &RuleCondition, profile: &Profile) -> bool {
        match condition {
            RuleCondition::HighFunctionTime {
                function,
                threshold,
            } => {
                if function == "any" {
                    profile
                        .call_graph
                        .nodes
                        .iter()
                        .any(|n| n.total_time > *threshold)
                } else {
                    profile
                        .call_graph
                        .nodes
                        .iter()
                        .any(|n| n.name == *function && n.total_time > *threshold)
                }
            }
            RuleCondition::HighMemoryUsage { threshold } => {
                profile.metrics.memory_metrics.peak_memory > *threshold
            }
            RuleCondition::LowCacheHitRate { threshold } => {
                profile.metrics.computation_metrics.cache_hit_rate < *threshold
            }
            RuleCondition::Custom(_) => false,
        }
    }

    /// Detect anomalies
    fn detect_anomalies(&self, profile: &Profile) -> Vec<Anomaly> {
        let mut anomalies = Vec::new();

        // Check for unusual time distributions
        for node in &profile.call_graph.nodes {
            if node.call_count > 10 {
                let avg_time = node.avg_time.as_secs_f64();
                let total_time = node.total_time.as_secs_f64();
                let expected_total = avg_time * node.call_count as f64;

                if (total_time - expected_total).abs() / expected_total > 0.5 {
                    anomalies.push(Anomaly {
                        anomaly_type: AnomalyType::Performance,
                        location: node.name.clone(),
                        description: "Unusual time distribution detected".to_string(),
                        confidence: 0.8,
                    });
                }
            }
        }

        anomalies
    }

    /// Find hot functions
    fn find_hot_functions(&self, profile: &Profile) -> Vec<(String, f64)> {
        let total_time = profile.metrics.time_metrics.total_time.as_secs_f64();

        let mut hot_functions: Vec<_> = profile
            .call_graph
            .nodes
            .iter()
            .map(|n| (n.name.clone(), n.total_time.as_secs_f64() / total_time))
            .collect();

        hot_functions.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .expect("Failed to compare time percentages in hot function sorting")
        });
        hot_functions.truncate(10);

        hot_functions
    }

    /// Find critical path
    fn find_critical_path(&self, profile: &Profile) -> Vec<String> {
        // Simplified critical path - longest execution path
        let mut path = Vec::new();

        if let Some(&root) = profile.call_graph.roots.first() {
            let mut current = root;
            path.push(profile.call_graph.nodes[current].name.clone());

            // Follow the most expensive child at each level
            while let Some(edge) = profile
                .call_graph
                .edges
                .iter()
                .filter(|e| e.from == current)
                .max_by_key(|e| profile.call_graph.nodes[e.to].total_time)
            {
                current = edge.to;
                path.push(profile.call_graph.nodes[current].name.clone());
            }
        }

        path
    }
}

/// Analysis report
#[derive(Debug, Clone)]
pub struct AnalysisReport {
    pub bottlenecks: Vec<Bottleneck>,
    pub optimizations: Vec<OptimizationSuggestion>,
    pub anomalies: Vec<Anomaly>,
    pub summary: AnalysisSummary,
}

#[derive(Debug, Clone)]
pub struct Bottleneck {
    pub bottleneck_type: BottleneckType,
    pub location: String,
    pub severity: Severity,
    pub impact: f64,
    pub description: String,
}

#[derive(Debug, Clone)]
pub enum BottleneckType {
    CPU,
    Memory,
    IO,
    Network,
    Contention,
}

#[derive(Debug, Clone)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone)]
pub struct OptimizationSuggestion {
    pub title: String,
    pub description: String,
    pub expected_improvement: f64,
    pub implementation_effort: ImplementationEffort,
    pub priority: Priority,
}

#[derive(Debug, Clone)]
pub enum ImplementationEffort {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone)]
pub struct Anomaly {
    pub anomaly_type: AnomalyType,
    pub location: String,
    pub description: String,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub enum AnomalyType {
    Performance,
    Memory,
    Behavior,
}

#[derive(Debug, Clone)]
pub struct AnalysisSummary {
    pub total_time: Duration,
    pub peak_memory: usize,
    pub hot_functions: Vec<(String, f64)>,
    pub critical_path: Vec<String>,
}

/// Comparison report
#[derive(Debug, Clone)]
pub struct ComparisonReport {
    pub time_comparison: TimeComparison,
    pub memory_comparison: MemoryComparison,
    pub quality_comparison: QualityComparison,
    pub regressions: Vec<Regression>,
    pub improvements: Vec<Improvement>,
}

#[derive(Debug, Clone)]
pub struct TimeComparison {
    pub total_time_diff: f64,
    pub total_time_ratio: f64,
    pub qubo_time_diff: f64,
    pub solving_time_diff: f64,
    pub function_diffs: BTreeMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct MemoryComparison {
    pub peak_memory_diff: i64,
    pub peak_memory_ratio: f64,
    pub avg_memory_diff: i64,
    pub allocations_diff: i64,
}

#[derive(Debug, Clone)]
pub struct QualityComparison {
    pub convergence_rate_diff: f64,
    pub time_to_best_diff: f64,
    pub final_quality_diff: f64,
}

#[derive(Debug, Clone)]
pub struct Regression {
    pub metric: String,
    pub old_value: f64,
    pub new_value: f64,
    pub change_percentage: f64,
    pub severity: Severity,
}

#[derive(Debug, Clone)]
pub struct Improvement {
    pub metric: String,
    pub old_value: f64,
    pub new_value: f64,
    pub change_percentage: f64,
}

/// Real-time performance monitor
pub struct RealTimeMonitor {
    /// Sampling interval
    sampling_interval: Duration,
    /// Collectors to use
    collector_names: Vec<String>,
    /// Live metrics
    live_metrics: Arc<Mutex<LiveMetrics>>,
    /// Monitor thread handle
    _monitor_handle: Option<thread::JoinHandle<()>>,
}

impl RealTimeMonitor {
    pub fn new(sampling_interval: Duration, collector_names: Vec<String>) -> Result<Self, String> {
        let live_metrics = Arc::new(Mutex::new(LiveMetrics::default()));

        Ok(Self {
            sampling_interval,
            collector_names,
            live_metrics,
            _monitor_handle: None,
        })
    }

    pub fn get_live_metrics(&self) -> LiveMetrics {
        self.live_metrics
            .lock()
            .expect("Failed to acquire lock on live_metrics for reading")
            .clone()
    }
}

#[derive(Debug, Clone, Default)]
pub struct LiveMetrics {
    pub current_cpu: f64,
    pub current_memory: usize,
    pub current_functions: Vec<(String, Duration)>,
    pub events_per_second: f64,
    pub last_update: Option<Instant>,
}

/// Performance prediction system
pub struct PerformancePredictor {
    /// Historical profiles
    history: Vec<Profile>,
    /// Prediction model
    model: PredictionModel,
}

impl PerformancePredictor {
    pub fn new(profiles: &[Profile]) -> Self {
        Self {
            history: profiles.to_vec(),
            model: PredictionModel::Linear,
        }
    }

    pub fn predict(&self, characteristics: &ProblemCharacteristics) -> PerformancePrediction {
        // Simplified prediction based on problem size
        let base_time = Duration::from_millis(100);
        let complexity_factor = match characteristics.complexity {
            ProblemComplexity::Linear => characteristics.size as f64,
            ProblemComplexity::Quadratic => (characteristics.size as f64).powi(2),
            ProblemComplexity::Exponential => 2.0_f64.powi(characteristics.size.min(30) as i32),
        };

        let estimated_time = base_time.mul_f64(complexity_factor / 1000.0);
        let estimated_memory = characteristics.size * 8; // 8 bytes per element

        PerformancePrediction {
            estimated_runtime: estimated_time,
            estimated_memory,
            confidence: if self.history.len() > 5 { 0.8 } else { 0.5 },
            bottleneck_predictions: vec![
                BottleneckPrediction {
                    location: "QUBO generation".to_string(),
                    probability: 0.3,
                    predicted_impact: 0.4,
                },
                BottleneckPrediction {
                    location: "Solving".to_string(),
                    probability: 0.7,
                    predicted_impact: 0.6,
                },
            ],
        }
    }
}

#[derive(Debug, Clone)]
pub enum PredictionModel {
    Linear,
    Polynomial,
    MachineLearning,
}

#[derive(Debug, Clone)]
pub struct ProblemCharacteristics {
    pub size: usize,
    pub complexity: ProblemComplexity,
    pub sparsity: f64,
    pub symmetry: bool,
    pub structure: ProblemStructure,
}

#[derive(Debug, Clone)]
pub enum ProblemComplexity {
    Linear,
    Quadratic,
    Exponential,
}

#[derive(Debug, Clone)]
pub enum ProblemStructure {
    Dense,
    Sparse,
    Structured,
    Random,
}

#[derive(Debug, Clone)]
pub struct PerformancePrediction {
    pub estimated_runtime: Duration,
    pub estimated_memory: usize,
    pub confidence: f64,
    pub bottleneck_predictions: Vec<BottleneckPrediction>,
}

#[derive(Debug, Clone)]
pub struct BottleneckPrediction {
    pub location: String,
    pub probability: f64,
    pub predicted_impact: f64,
}

/// Advanced optimization recommendations
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    pub title: String,
    pub description: String,
    pub category: RecommendationCategory,
    pub impact: RecommendationImpact,
    pub effort: ImplementationEffort,
    pub estimated_improvement: f64,
    pub code_suggestions: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum RecommendationCategory {
    Algorithm,
    Memory,
    IO,
    Parallelization,
    Caching,
    DataStructure,
}

#[derive(Debug, Clone)]
pub enum RecommendationImpact {
    Low,
    Medium,
    High,
    Critical,
}

/// External tool integration
#[derive(Debug, Clone)]
pub enum ExternalTool {
    Perf,
    Valgrind,
    FlameScope,
    SpeedScope,
}

/// Continuous profiling
pub struct ContinuousProfiler {
    duration: Duration,
    sampling_interval: Duration,
    profiles: Vec<Profile>,
}

impl ContinuousProfiler {
    pub const fn new(duration: Duration, sampling_interval: Duration) -> Self {
        Self {
            duration,
            sampling_interval,
            profiles: Vec::new(),
        }
    }

    pub fn get_profiles(&self) -> &[Profile] {
        &self.profiles
    }
}

/// Benchmark comparison
#[derive(Debug, Clone)]
pub struct BenchmarkComparison {
    pub profiles: Vec<String>,
    pub metrics_comparison: Vec<MetricComparison>,
    pub regression_analysis: Vec<RegressionAnalysis>,
    pub performance_trends: Vec<PerformanceTrendAnalysis>,
}

#[derive(Debug, Clone)]
pub struct MetricComparison {
    pub metric_name: String,
    pub values: Vec<f64>,
    pub trend: PerformanceTrend,
    pub variance: f64,
}

#[derive(Debug, Clone)]
pub enum PerformanceTrend {
    Improving,
    Stable,
    Degrading,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct RegressionAnalysis {
    pub metric: String,
    pub regression_coefficient: f64,
    pub correlation: f64,
    pub prediction_accuracy: f64,
}

#[derive(Debug, Clone)]
pub struct PerformanceTrendAnalysis {
    pub function_name: String,
    pub trend: PerformanceTrend,
    pub rate_of_change: f64,
    pub statistical_significance: f64,
}

/// Default collectors
struct TimeCollector;

impl MetricsCollector for TimeCollector {
    fn collect(&self) -> Result<MetricsSample, String> {
        Ok(MetricsSample {
            timestamp: Instant::now(),
            values: HashMap::new(),
        })
    }

    fn name(&self) -> &'static str {
        "TimeCollector"
    }

    fn supported_metrics(&self) -> Vec<MetricType> {
        vec![MetricType::Time]
    }
}

struct MemoryCollector;

impl MetricsCollector for MemoryCollector {
    fn collect(&self) -> Result<MetricsSample, String> {
        // Would use system APIs to get actual memory usage
        let mut values = HashMap::new();
        values.insert(MetricType::Memory, 0.0);

        Ok(MetricsSample {
            timestamp: Instant::now(),
            values,
        })
    }

    fn name(&self) -> &'static str {
        "MemoryCollector"
    }

    fn supported_metrics(&self) -> Vec<MetricType> {
        vec![MetricType::Memory]
    }
}

struct CPUCollector;

impl MetricsCollector for CPUCollector {
    fn collect(&self) -> Result<MetricsSample, String> {
        // Would use system APIs to get actual CPU usage
        let mut values = HashMap::new();
        values.insert(MetricType::CPU, 0.0);

        Ok(MetricsSample {
            timestamp: Instant::now(),
            values,
        })
    }

    fn name(&self) -> &'static str {
        "CPUCollector"
    }

    fn supported_metrics(&self) -> Vec<MetricType> {
        vec![MetricType::CPU]
    }
}

/// Profiling macros
#[macro_export]
macro_rules! profile {
    ($profiler:expr, $name:expr) => {
        $profiler.enter_function($name)
    };
}

#[macro_export]
macro_rules! time_it {
    ($profiler:expr, $name:expr, $code:block) => {{
        $profiler.start_timer($name);
        let mut result = $code;
        $profiler.stop_timer($name);
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_profiler() {
        let mut config = ProfilerConfig {
            enabled: true,
            sampling_interval: Duration::from_millis(10),
            metrics: vec![MetricType::Time, MetricType::Memory],
            profile_memory: true,
            profile_cpu: true,
            profile_gpu: false,
            detailed_timing: true,
            output_format: OutputFormat::Json,
            auto_save_interval: None,
        };

        let mut profiler = PerformanceProfiler::new(config);

        // Start profiling
        let mut result = profiler.start_profile("test_profile");
        assert!(result.is_ok());

        // Simulate some work
        {
            let _guard = profiler.enter_function("test_function");
            profiler.start_timer("computation");
            thread::sleep(Duration::from_millis(10));
            profiler.stop_timer("computation");

            profiler.record_allocation(1024);
            profiler.record_solution_quality(0.5);
            profiler.record_deallocation(1024);
        }

        // Stop profiling
        let profile = profiler.stop_profile();
        assert!(profile.is_ok());

        let profile = profile.expect("Failed to stop profiling in test_performance_profiler");
        assert!(!profile.events.is_empty());
        assert!(profile.metrics.time_metrics.total_time > Duration::from_secs(0));

        // Analyze profile
        let mut report = profiler.analyze_profile(&profile);
        assert!(report.summary.total_time > Duration::from_secs(0));

        // Generate report
        let json_report = profiler.generate_report(&profile, &OutputFormat::Json);
        assert!(json_report.is_ok());
    }
}
