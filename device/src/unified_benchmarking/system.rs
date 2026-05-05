//! Main unified benchmarking system implementation

use std::collections::{HashMap, VecDeque};
use std::sync::{mpsc, Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::config::{
    AlgorithmBenchmarkConfig, CircuitBenchmarkConfig, GateBenchmarkConfig, SystemBenchmarkConfig,
    UnifiedBenchmarkConfig,
};
use super::events::BenchmarkEvent;
use super::optimization::OptimizationEngine;
use super::reporting::ReportGenerator;
use super::results::{
    AlgorithmLevelResults, CircuitLevelResults, CoherenceTimes, ConnectivityInfo,
    CostAnalysisResult, CostMetrics, CrossPlatformAnalysis, DeviceInfo, DeviceSpecifications,
    DeviceStatus, ExecutionMetadata, GateLevelResults, HistoricalComparisonResult,
    OptimizationRecommendation, PlatformBenchmarkResult, PlatformPerformanceMetrics,
    QuantumTechnology, ReliabilityMetrics, ResourceAnalysisResult, SciRS2AnalysisResult,
    SystemLevelResults, TopologyType, UnifiedBenchmarkResult,
};
use super::types::{PerformanceBaseline, QuantumPlatform};

use crate::{
    advanced_benchmarking_suite::{AdvancedBenchmarkConfig, AdvancedHardwareBenchmarkSuite},
    calibration::CalibrationManager,
    cross_platform_benchmarking::{CrossPlatformBenchmarkConfig, CrossPlatformBenchmarker},
    topology::HardwareTopology,
    DeviceError, DeviceResult, QuantumDevice,
};
use quantrs2_core::error::{QuantRS2Error, QuantRS2Result};

use scirs2_core::ndarray::Array2;

/// Main unified benchmarking system
pub struct UnifiedQuantumBenchmarkSystem {
    /// Configuration
    config: Arc<RwLock<UnifiedBenchmarkConfig>>,
    /// Platform clients
    platform_clients: Arc<RwLock<HashMap<QuantumPlatform, Box<dyn QuantumDevice + Send + Sync>>>>,
    /// Cross-platform benchmarker
    cross_platform_benchmarker: Arc<Mutex<CrossPlatformBenchmarker>>,
    /// Advanced benchmarking suite
    advanced_suite: Arc<Mutex<AdvancedHardwareBenchmarkSuite>>,
    /// Calibration manager
    calibration_manager: Arc<Mutex<CalibrationManager>>,
    /// Historical data storage
    historical_data: Arc<RwLock<VecDeque<UnifiedBenchmarkResult>>>,
    /// Performance baselines
    baselines: Arc<RwLock<HashMap<String, PerformanceBaseline>>>,
    /// Real-time monitoring
    monitoring_handle: Arc<Mutex<Option<std::thread::JoinHandle<()>>>>,
    /// Event publisher
    event_publisher: mpsc::Sender<BenchmarkEvent>,
    /// Optimization engine
    optimization_engine: Arc<Mutex<OptimizationEngine>>,
    /// Report generator
    report_generator: Arc<Mutex<ReportGenerator>>,
}

impl UnifiedQuantumBenchmarkSystem {
    /// Create a new unified quantum benchmark system
    pub async fn new(
        config: UnifiedBenchmarkConfig,
        calibration_manager: CalibrationManager,
    ) -> DeviceResult<Self> {
        let (event_publisher, _) = mpsc::channel();
        let config = Arc::new(RwLock::new(config));

        // Initialize platform clients
        let platform_clients = Arc::new(RwLock::new(HashMap::new()));

        // Initialize cross-platform benchmarker
        let cross_platform_config = CrossPlatformBenchmarkConfig::default();
        let cross_platform_benchmarker = Arc::new(Mutex::new(CrossPlatformBenchmarker::new(
            cross_platform_config,
            calibration_manager.clone(),
        )));

        // Initialize advanced benchmarking suite
        let advanced_config = AdvancedBenchmarkConfig::default();
        let topology = HardwareTopology::linear_topology(8); // Default topology
        let advanced_suite = Arc::new(Mutex::new(
            AdvancedHardwareBenchmarkSuite::new(
                advanced_config,
                calibration_manager.clone(),
                topology,
            )
            .await?,
        ));

        let historical_data = Arc::new(RwLock::new(VecDeque::with_capacity(10000)));
        let baselines = Arc::new(RwLock::new(HashMap::new()));
        let monitoring_handle = Arc::new(Mutex::new(None));

        let optimization_engine = Arc::new(Mutex::new(OptimizationEngine::new()));
        let report_generator = Arc::new(Mutex::new(ReportGenerator::new()));

        Ok(Self {
            config,
            platform_clients,
            cross_platform_benchmarker,
            advanced_suite,
            calibration_manager: Arc::new(Mutex::new(calibration_manager)),
            historical_data,
            baselines,
            monitoring_handle,
            event_publisher,
            optimization_engine,
            report_generator,
        })
    }

    /// Register a quantum platform for benchmarking
    pub async fn register_platform(
        &self,
        platform: QuantumPlatform,
        device: Box<dyn QuantumDevice + Send + Sync>,
    ) -> DeviceResult<()> {
        let mut clients = self
            .platform_clients
            .write()
            .unwrap_or_else(|e| e.into_inner());
        clients.insert(platform, device);
        Ok(())
    }

    /// Run comprehensive unified benchmarks
    pub async fn run_comprehensive_benchmark(&self) -> DeviceResult<UnifiedBenchmarkResult> {
        let execution_id = self.generate_execution_id();
        let start_time = SystemTime::now();

        // Notify benchmark start
        let config = self
            .config
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let _ = self.event_publisher.send(BenchmarkEvent::BenchmarkStarted {
            execution_id: execution_id.clone(),
            platforms: config.target_platforms.clone(),
            timestamp: start_time,
        });

        // Execute benchmarks on all platforms
        let mut platform_results = HashMap::new();

        for platform in &config.target_platforms {
            match self.run_platform_benchmark(platform, &execution_id).await {
                Ok(result) => {
                    let _ = self
                        .event_publisher
                        .send(BenchmarkEvent::PlatformBenchmarkCompleted {
                            execution_id: execution_id.clone(),
                            platform: platform.clone(),
                            result: result.clone(),
                            timestamp: SystemTime::now(),
                        });
                    platform_results.insert(platform.clone(), result);
                }
                Err(e) => {
                    eprintln!("Platform benchmark failed for {platform:?}: {e}");
                    // Continue with other platforms
                }
            }
        }

        // Perform analysis
        let cross_platform_analysis = self
            .perform_cross_platform_analysis(&platform_results)
            .await?;
        let scirs2_analysis = self.perform_scirs2_analysis(&platform_results).await?;
        let resource_analysis = self.perform_resource_analysis(&platform_results).await?;
        let cost_analysis = self.perform_cost_analysis(&platform_results).await?;

        // Generate optimization recommendations
        let optimization_recommendations = self
            .generate_optimization_recommendations(
                &platform_results,
                &cross_platform_analysis,
                &scirs2_analysis,
            )
            .await?;

        // Perform historical comparison if available
        let historical_comparison = self
            .perform_historical_comparison(&platform_results)
            .await?;

        // Create execution metadata
        let execution_metadata = ExecutionMetadata {
            execution_start_time: start_time,
            execution_end_time: SystemTime::now(),
            total_duration: SystemTime::now()
                .duration_since(start_time)
                .unwrap_or(Duration::ZERO),
            platforms_tested: config.target_platforms.clone(),
            benchmarks_executed: platform_results.len(),
            system_info: self.get_system_info(),
        };

        let result = UnifiedBenchmarkResult {
            execution_id: execution_id.clone(),
            timestamp: start_time,
            config,
            platform_results,
            cross_platform_analysis,
            scirs2_analysis,
            resource_analysis,
            cost_analysis,
            optimization_recommendations,
            historical_comparison,
            execution_metadata,
        };

        // Store result in historical data
        self.store_historical_result(&result).await;

        // Update baselines if needed
        self.update_baselines(&result).await;

        // Trigger optimization if enabled
        if result
            .config
            .optimization_config
            .enable_intelligent_allocation
        {
            self.trigger_optimization(&result).await?;
        }

        // Generate automated reports if enabled
        if result
            .config
            .reporting_config
            .automated_reports
            .enable_automated
        {
            self.generate_automated_reports(&result).await?;
        }

        // Notify benchmark completion
        let _ = self
            .event_publisher
            .send(BenchmarkEvent::BenchmarkCompleted {
                execution_id: execution_id.clone(),
                result: result.clone(),
                timestamp: SystemTime::now(),
            });

        Ok(result)
    }

    /// Run benchmark on a specific platform
    async fn run_platform_benchmark(
        &self,
        platform: &QuantumPlatform,
        execution_id: &str,
    ) -> DeviceResult<PlatformBenchmarkResult> {
        let config = self
            .config
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();

        // Get device information
        let device_info = self.get_device_info(platform).await?;

        // Run benchmarks
        let gate_level_results = self
            .run_gate_level_benchmarks(platform, &config.benchmark_suite.gate_benchmarks)
            .await?;
        let circuit_level_results = self
            .run_circuit_level_benchmarks(platform, &config.benchmark_suite.circuit_benchmarks)
            .await?;
        let algorithm_level_results = self
            .run_algorithm_level_benchmarks(platform, &config.benchmark_suite.algorithm_benchmarks)
            .await?;
        let system_level_results = self
            .run_system_level_benchmarks(platform, &config.benchmark_suite.system_benchmarks)
            .await?;

        // Calculate metrics
        let performance_metrics = self
            .calculate_platform_performance_metrics(
                &gate_level_results,
                &circuit_level_results,
                &algorithm_level_results,
                &system_level_results,
            )
            .await?;

        let reliability_metrics = self
            .calculate_reliability_metrics(
                &gate_level_results,
                &circuit_level_results,
                &algorithm_level_results,
            )
            .await?;

        let cost_metrics = self
            .calculate_cost_metrics(
                &gate_level_results,
                &circuit_level_results,
                &algorithm_level_results,
            )
            .await?;

        Ok(PlatformBenchmarkResult {
            platform: platform.clone(),
            device_info,
            gate_level_results,
            circuit_level_results,
            algorithm_level_results,
            system_level_results,
            performance_metrics,
            reliability_metrics,
            cost_metrics,
        })
    }

    /// Generate unique execution ID
    fn generate_execution_id(&self) -> String {
        format!(
            "unified_benchmark_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_millis()
        )
    }

    /// Get device information for a platform
    async fn get_device_info(&self, platform: &QuantumPlatform) -> DeviceResult<DeviceInfo> {
        let (provider, technology) = match platform {
            QuantumPlatform::IBMQuantum { .. } => {
                ("IBM".to_string(), QuantumTechnology::Superconducting)
            }
            QuantumPlatform::AWSBraket { .. } => {
                ("AWS".to_string(), QuantumTechnology::Superconducting)
            }
            QuantumPlatform::AzureQuantum { .. } => {
                ("Microsoft".to_string(), QuantumTechnology::TrappedIon)
            }
            QuantumPlatform::IonQ { .. } => ("IonQ".to_string(), QuantumTechnology::TrappedIon),
            QuantumPlatform::Rigetti { .. } => {
                ("Rigetti".to_string(), QuantumTechnology::Superconducting)
            }
            QuantumPlatform::GoogleQuantumAI { .. } => {
                ("Google".to_string(), QuantumTechnology::Superconducting)
            }
            QuantumPlatform::Custom { .. } => (
                "Custom".to_string(),
                QuantumTechnology::Other("Custom".to_string()),
            ),
        };

        Ok(DeviceInfo {
            device_id: format!("{platform:?}"),
            provider,
            technology,
            specifications: DeviceSpecifications {
                num_qubits: 20,
                connectivity: ConnectivityInfo {
                    topology_type: TopologyType::Heavy,
                    coupling_map: vec![(0, 1), (1, 2), (2, 3)],
                    connectivity_matrix: Array2::eye(20),
                },
                gate_set: vec![
                    "X".to_string(),
                    "Y".to_string(),
                    "Z".to_string(),
                    "H".to_string(),
                    "CNOT".to_string(),
                ],
                coherence_times: CoherenceTimes {
                    t1: (0..20).map(|i| (i, Duration::from_micros(100))).collect(),
                    t2: (0..20).map(|i| (i, Duration::from_micros(50))).collect(),
                    t2_echo: (0..20).map(|i| (i, Duration::from_micros(80))).collect(),
                },
                gate_times: [
                    ("X".to_string(), Duration::from_nanos(20)),
                    ("CNOT".to_string(), Duration::from_nanos(100)),
                ]
                .iter()
                .cloned()
                .collect(),
                error_rates: [
                    ("single_qubit".to_string(), 0.001),
                    ("two_qubit".to_string(), 0.01),
                ]
                .iter()
                .cloned()
                .collect(),
            },
            current_status: DeviceStatus::Online,
            calibration_date: Some(SystemTime::now()),
        })
    }

    // Placeholder implementations for benchmark execution methods
    async fn run_gate_level_benchmarks(
        &self,
        _platform: &QuantumPlatform,
        _config: &GateBenchmarkConfig,
    ) -> DeviceResult<GateLevelResults> {
        // TODO: Implement actual gate-level benchmarks
        Err(DeviceError::NotImplemented(
            "Gate-level benchmarks not yet implemented".to_string(),
        ))
    }

    async fn run_circuit_level_benchmarks(
        &self,
        _platform: &QuantumPlatform,
        _config: &CircuitBenchmarkConfig,
    ) -> DeviceResult<CircuitLevelResults> {
        // TODO: Implement actual circuit-level benchmarks
        Err(DeviceError::NotImplemented(
            "Circuit-level benchmarks not yet implemented".to_string(),
        ))
    }

    async fn run_algorithm_level_benchmarks(
        &self,
        _platform: &QuantumPlatform,
        _config: &AlgorithmBenchmarkConfig,
    ) -> DeviceResult<AlgorithmLevelResults> {
        // TODO: Implement actual algorithm-level benchmarks
        Err(DeviceError::NotImplemented(
            "Algorithm-level benchmarks not yet implemented".to_string(),
        ))
    }

    async fn run_system_level_benchmarks(
        &self,
        _platform: &QuantumPlatform,
        _config: &SystemBenchmarkConfig,
    ) -> DeviceResult<SystemLevelResults> {
        // TODO: Implement actual system-level benchmarks
        Err(DeviceError::NotImplemented(
            "System-level benchmarks not yet implemented".to_string(),
        ))
    }

    // Analysis methods
    async fn perform_cross_platform_analysis(
        &self,
        _platform_results: &HashMap<QuantumPlatform, PlatformBenchmarkResult>,
    ) -> DeviceResult<CrossPlatformAnalysis> {
        // TODO: Implement cross-platform analysis
        Err(DeviceError::NotImplemented(
            "Cross-platform analysis not yet implemented".to_string(),
        ))
    }

    async fn perform_scirs2_analysis(
        &self,
        _platform_results: &HashMap<QuantumPlatform, PlatformBenchmarkResult>,
    ) -> DeviceResult<SciRS2AnalysisResult> {
        // TODO: Implement SciRS2 analysis
        Err(DeviceError::NotImplemented(
            "SciRS2 analysis not yet implemented".to_string(),
        ))
    }

    async fn perform_resource_analysis(
        &self,
        _platform_results: &HashMap<QuantumPlatform, PlatformBenchmarkResult>,
    ) -> DeviceResult<ResourceAnalysisResult> {
        // TODO: Implement resource analysis
        Err(DeviceError::NotImplemented(
            "Resource analysis not yet implemented".to_string(),
        ))
    }

    async fn perform_cost_analysis(
        &self,
        _platform_results: &HashMap<QuantumPlatform, PlatformBenchmarkResult>,
    ) -> DeviceResult<CostAnalysisResult> {
        // TODO: Implement cost analysis
        Err(DeviceError::NotImplemented(
            "Cost analysis not yet implemented".to_string(),
        ))
    }

    // Metrics calculation
    async fn calculate_platform_performance_metrics(
        &self,
        _gate_results: &GateLevelResults,
        _circuit_results: &CircuitLevelResults,
        _algorithm_results: &AlgorithmLevelResults,
        _system_results: &SystemLevelResults,
    ) -> DeviceResult<PlatformPerformanceMetrics> {
        // TODO: Implement performance metrics calculation
        Err(DeviceError::NotImplemented(
            "Performance metrics calculation not yet implemented".to_string(),
        ))
    }

    async fn calculate_reliability_metrics(
        &self,
        _gate_results: &GateLevelResults,
        _circuit_results: &CircuitLevelResults,
        _algorithm_results: &AlgorithmLevelResults,
    ) -> DeviceResult<ReliabilityMetrics> {
        // TODO: Implement reliability metrics calculation
        Err(DeviceError::NotImplemented(
            "Reliability metrics calculation not yet implemented".to_string(),
        ))
    }

    async fn calculate_cost_metrics(
        &self,
        _gate_results: &GateLevelResults,
        _circuit_results: &CircuitLevelResults,
        _algorithm_results: &AlgorithmLevelResults,
    ) -> DeviceResult<CostMetrics> {
        // TODO: Implement cost metrics calculation
        Err(DeviceError::NotImplemented(
            "Cost metrics calculation not yet implemented".to_string(),
        ))
    }

    // Utility methods
    async fn generate_optimization_recommendations(
        &self,
        platform_results: &HashMap<QuantumPlatform, PlatformBenchmarkResult>,
        cross_platform_analysis: &CrossPlatformAnalysis,
        _scirs2_analysis: &SciRS2AnalysisResult,
    ) -> DeviceResult<Vec<OptimizationRecommendation>> {
        Ok(generate_recommendations_from_metrics(
            platform_results,
            cross_platform_analysis,
        ))
    }

    async fn perform_historical_comparison(
        &self,
        platform_results: &HashMap<QuantumPlatform, PlatformBenchmarkResult>,
    ) -> DeviceResult<Option<HistoricalComparisonResult>> {
        // Compare current per-platform metrics with the most recent historical
        // entry stored in `historical_data`. Without history, no comparison
        // is possible — return `None`. Statistical significance here is a
        // placeholder p-value derived from relative change; it is meant for
        // dashboard ranking, not formal hypothesis testing (which would
        // require keeping individual measurements per metric).
        let historical_data = self
            .historical_data
            .read()
            .unwrap_or_else(|e| e.into_inner());

        // Skip the most recent entry if it's the run we just stored.
        let baseline = historical_data.iter().next_back();
        let baseline = match baseline {
            Some(b) => b,
            None => return Ok(None),
        };

        let mut baseline_comparison: Vec<super::results::MetricComparison> = Vec::new();
        let mut trend_analysis: HashMap<String, super::results::TrendAnalysisResult> =
            HashMap::new();

        for (platform, current) in platform_results {
            let prev = match baseline.platform_results.get(platform) {
                Some(p) => p,
                None => continue,
            };

            // Compare overall_fidelity, error_rate, throughput, availability.
            let pairs: [(&str, f64, f64); 4] = [
                (
                    "overall_fidelity",
                    current.performance_metrics.overall_fidelity,
                    prev.performance_metrics.overall_fidelity,
                ),
                (
                    "error_rate",
                    current.performance_metrics.error_rate,
                    prev.performance_metrics.error_rate,
                ),
                (
                    "throughput",
                    current.performance_metrics.throughput,
                    prev.performance_metrics.throughput,
                ),
                (
                    "availability",
                    current.performance_metrics.availability,
                    prev.performance_metrics.availability,
                ),
            ];

            for (name, cur_val, base_val) in pairs {
                let percentage_change = if base_val.abs() > f64::EPSILON {
                    (cur_val - base_val) / base_val * 100.0
                } else if cur_val.abs() < f64::EPSILON {
                    0.0
                } else {
                    100.0
                };
                // Map magnitude of relative change to a pseudo p-value:
                // larger changes → smaller p-value.
                let mag = percentage_change.abs();
                let significance = (-mag / 50.0).exp().clamp(0.0, 1.0);

                let metric_label = format!("{platform:?}.{name}");
                baseline_comparison.push(super::results::MetricComparison {
                    metric_name: metric_label.clone(),
                    current_value: cur_val,
                    baseline_value: base_val,
                    percentage_change,
                    statistical_significance: significance,
                });

                let direction = if cur_val > base_val {
                    "increasing"
                } else if cur_val < base_val {
                    "decreasing"
                } else {
                    "flat"
                };
                trend_analysis.insert(
                    metric_label,
                    super::results::TrendAnalysisResult {
                        trend_detected: mag > 1.0,
                        trend_direction: direction.to_string(),
                        trend_strength: (mag / 100.0).clamp(0.0, 1.0),
                        trend_coefficients: vec![cur_val - base_val],
                        change_points: Vec::new(),
                    },
                );
            }
        }

        // Build a single performance snapshot for the current run so the
        // evolution series is monotonically growing on each invocation.
        let mut metrics_map: HashMap<String, f64> = HashMap::new();
        for (platform, result) in platform_results {
            metrics_map.insert(
                format!("{platform:?}.overall_fidelity"),
                result.performance_metrics.overall_fidelity,
            );
            metrics_map.insert(
                format!("{platform:?}.error_rate"),
                result.performance_metrics.error_rate,
            );
            metrics_map.insert(
                format!("{platform:?}.throughput"),
                result.performance_metrics.throughput,
            );
            metrics_map.insert(
                format!("{platform:?}.availability"),
                result.performance_metrics.availability,
            );
        }

        let snapshot = super::results::PerformanceSnapshot {
            timestamp: SystemTime::now(),
            metrics: metrics_map,
            configuration: "current".to_string(),
        };

        if baseline_comparison.is_empty() {
            return Ok(None);
        }

        Ok(Some(HistoricalComparisonResult {
            baseline_comparison,
            trend_analysis,
            performance_evolution: vec![snapshot],
        }))
    }

    async fn store_historical_result(&self, result: &UnifiedBenchmarkResult) {
        let mut historical_data = self
            .historical_data
            .write()
            .unwrap_or_else(|e| e.into_inner());
        historical_data.push_back(result.clone());

        // Keep only the last 10000 results
        if historical_data.len() > 10000 {
            historical_data.pop_front();
        }
    }

    async fn update_baselines(&self, _result: &UnifiedBenchmarkResult) {
        // TODO: Implement baseline updates
    }

    async fn trigger_optimization(&self, result: &UnifiedBenchmarkResult) -> DeviceResult<()> {
        // Compute aggregated improvement signal from the recommendations and
        // publish an `OptimizationCompleted` event. The `expected_improvement`
        // values produced by `generate_optimization_recommendations` already
        // express the headroom each recommendation can recover; we forward
        // those to subscribers grouped by recommendation type.
        let recommendations = &result.optimization_recommendations;
        if recommendations.is_empty() {
            return Ok(());
        }

        let mut improvements: HashMap<String, f64> = HashMap::new();
        for r in recommendations {
            *improvements
                .entry(r.recommendation_type.clone())
                .or_insert(0.0) += r.expected_improvement;
        }

        // Also raise per-platform alerts when fidelity / error rate breach
        // configured thresholds. This makes the metrics visible to dashboards
        // even before any reporting runs.
        for (platform, platform_result) in &result.platform_results {
            let metrics = &platform_result.performance_metrics;
            if metrics.error_rate > 0.05 {
                let _ = self.event_publisher.send(BenchmarkEvent::PerformanceAlert {
                    metric: format!("{platform:?}.error_rate"),
                    current_value: metrics.error_rate,
                    threshold: 0.05,
                    timestamp: SystemTime::now(),
                });
            }
            if metrics.overall_fidelity < 0.90 {
                let _ = self.event_publisher.send(BenchmarkEvent::PerformanceAlert {
                    metric: format!("{platform:?}.overall_fidelity"),
                    current_value: metrics.overall_fidelity,
                    threshold: 0.90,
                    timestamp: SystemTime::now(),
                });
            }
        }

        let _ = self
            .event_publisher
            .send(BenchmarkEvent::OptimizationCompleted {
                execution_id: result.execution_id.clone(),
                improvements,
                timestamp: SystemTime::now(),
            });

        Ok(())
    }

    async fn generate_automated_reports(
        &self,
        result: &UnifiedBenchmarkResult,
    ) -> DeviceResult<()> {
        // Render the result to a JSON payload. We use `serde_json` (already a
        // dependency of this crate) so the payload is portable and consumable
        // by external tooling without any new dependency. Writing to a file
        // is gated on the `recipients` list being a filesystem-style URI
        // (`file://...`) — anything else is treated as out-of-scope here.
        let payload = serde_json::to_vec_pretty(result).map_err(|err| {
            DeviceError::APIError(format!("failed to serialize benchmark result: {err}"))
        })?;

        let config = self
            .config
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();

        for recipient in &config.reporting_config.automated_reports.recipients {
            if let Some(path) = recipient.strip_prefix("file://") {
                let dir = std::path::Path::new(path);
                if let Err(err) = std::fs::create_dir_all(dir) {
                    return Err(DeviceError::APIError(format!(
                        "failed to create report directory {dir:?}: {err}"
                    )));
                }
                let file = dir.join(format!("benchmark_{}.json", result.execution_id));
                if let Err(err) = std::fs::write(&file, &payload) {
                    return Err(DeviceError::APIError(format!(
                        "failed to write report {file:?}: {err}"
                    )));
                }
            }
        }

        // Emit a per-recipient performance alert summarizing the size of the
        // generated report — a concrete signal for downstream pipelines.
        let _ = self.event_publisher.send(BenchmarkEvent::PerformanceAlert {
            metric: "automated_report.bytes".to_string(),
            current_value: payload.len() as f64,
            threshold: 0.0,
            timestamp: SystemTime::now(),
        });

        Ok(())
    }

    fn get_system_info(&self) -> super::results::SystemInfo {
        super::results::SystemInfo {
            hostname: "localhost".to_string(),
            operating_system: std::env::consts::OS.to_string(),
            cpu_info: "Unknown".to_string(),
            memory_total: 0,
            disk_space: 0,
            network_info: "Unknown".to_string(),
        }
    }
}

/// Pure helper that maps per-platform performance metrics to a list of
/// concrete optimization recommendations. Extracted so it can be unit-tested
/// without instantiating `UnifiedQuantumBenchmarkSystem` (which requires a
/// fully wired calibration manager and async runtime).
pub(crate) fn generate_recommendations_from_metrics(
    platform_results: &HashMap<QuantumPlatform, PlatformBenchmarkResult>,
    cross_platform_analysis: &CrossPlatformAnalysis,
) -> Vec<OptimizationRecommendation> {
    let metrics_map: HashMap<QuantumPlatform, &PlatformPerformanceMetrics> = platform_results
        .iter()
        .map(|(p, r)| (p.clone(), &r.performance_metrics))
        .collect();
    generate_recommendations_from_perf_metrics(&metrics_map, cross_platform_analysis)
}

/// Inner helper that operates only on the lightweight
/// [`PlatformPerformanceMetrics`] type so unit tests do not need to fabricate
/// a full [`PlatformBenchmarkResult`].
pub(crate) fn generate_recommendations_from_perf_metrics(
    platform_metrics: &HashMap<QuantumPlatform, &PlatformPerformanceMetrics>,
    cross_platform_analysis: &CrossPlatformAnalysis,
) -> Vec<OptimizationRecommendation> {
    let mut recommendations: Vec<OptimizationRecommendation> = Vec::new();

    const ERROR_RATE_THRESHOLD: f64 = 0.05;
    const FIDELITY_THRESHOLD: f64 = 0.90;
    const AVAILABILITY_THRESHOLD: f64 = 0.95;
    const EXEC_TIME_SECS_THRESHOLD: f64 = 1.0;
    const THROUGHPUT_THRESHOLD: f64 = 1.0;

    for (platform, metrics) in platform_metrics {
        let platform_label = format!("{platform:?}");

        if metrics.error_rate > ERROR_RATE_THRESHOLD {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: "error_mitigation".to_string(),
                description: format!(
                    "Platform {platform_label} reports an error rate of {:.4} (threshold {:.2}). Apply zero-noise extrapolation, dynamical decoupling, or readout error mitigation to reduce systematic errors.",
                    metrics.error_rate, ERROR_RATE_THRESHOLD
                ),
                expected_improvement: (metrics.error_rate - ERROR_RATE_THRESHOLD).max(0.0),
                implementation_effort: "Medium".to_string(),
                priority: 2,
            });
        }

        if metrics.overall_fidelity < FIDELITY_THRESHOLD {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: "calibration".to_string(),
                description: format!(
                    "Platform {platform_label} fidelity {:.4} is below threshold {:.2}. Schedule recalibration of single- and two-qubit gates and refresh readout calibration.",
                    metrics.overall_fidelity, FIDELITY_THRESHOLD
                ),
                expected_improvement: (FIDELITY_THRESHOLD - metrics.overall_fidelity).max(0.0),
                implementation_effort: "Low".to_string(),
                priority: 1,
            });
        }

        let avg_secs = metrics.average_execution_time.as_secs_f64();
        if avg_secs > EXEC_TIME_SECS_THRESHOLD {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: "circuit_reduction".to_string(),
                description: format!(
                    "Platform {platform_label} average execution time {avg_secs:.3}s exceeds threshold {EXEC_TIME_SECS_THRESHOLD:.2}s. Apply circuit transpilation passes (gate fusion, commutation analysis, and depth-aware routing)."
                ),
                expected_improvement: ((avg_secs - EXEC_TIME_SECS_THRESHOLD)
                    / avg_secs.max(f64::EPSILON))
                .clamp(0.0, 1.0),
                implementation_effort: "Medium".to_string(),
                priority: 3,
            });
        }

        if metrics.availability < AVAILABILITY_THRESHOLD {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: "platform_redundancy".to_string(),
                description: format!(
                    "Platform {platform_label} availability {:.3} is below threshold {:.2}. Configure failover to secondary platforms and enable retry policies.",
                    metrics.availability, AVAILABILITY_THRESHOLD
                ),
                expected_improvement: (AVAILABILITY_THRESHOLD - metrics.availability).max(0.0),
                implementation_effort: "Medium".to_string(),
                priority: 2,
            });
        }

        if metrics.throughput < THROUGHPUT_THRESHOLD && metrics.throughput > 0.0 {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: "batching_optimization".to_string(),
                description: format!(
                    "Platform {platform_label} throughput {:.3} ops/s is below threshold {THROUGHPUT_THRESHOLD:.2}. Batch jobs and submit in larger groups to amortize queue overhead.",
                    metrics.throughput
                ),
                expected_improvement: ((THROUGHPUT_THRESHOLD - metrics.throughput)
                    / THROUGHPUT_THRESHOLD)
                    .clamp(0.0, 1.0),
                implementation_effort: "Low".to_string(),
                priority: 4,
            });
        }
    }

    if platform_metrics.len() > 1 {
        if let Some(best_for_fidelity) = cross_platform_analysis
            .best_platform_per_metric
            .get("fidelity")
        {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: "platform_routing".to_string(),
                description: format!(
                    "Cross-platform analysis identifies {best_for_fidelity:?} as the best platform for fidelity. Route fidelity-critical workloads there and use other platforms for high-throughput jobs."
                ),
                expected_improvement: 0.05,
                implementation_effort: "Low".to_string(),
                priority: 5,
            });
        }
    }

    if recommendations.is_empty() {
        recommendations.push(OptimizationRecommendation {
            recommendation_type: "monitoring".to_string(),
            description: "All platform metrics are within configured thresholds. Continue periodic benchmarking to detect drift early.".to_string(),
            expected_improvement: 0.0,
            implementation_effort: "Low".to_string(),
            priority: 9,
        });
    }

    recommendations.sort_by_key(|r| r.priority);
    recommendations
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn perf(
        fidelity: f64,
        error_rate: f64,
        availability: f64,
        avg_exec_secs: u64,
        throughput: f64,
    ) -> PlatformPerformanceMetrics {
        PlatformPerformanceMetrics {
            overall_fidelity: fidelity,
            average_execution_time: Duration::from_secs(avg_exec_secs),
            throughput,
            error_rate,
            availability,
        }
    }

    #[test]
    fn test_recommendations_within_thresholds() {
        // No metric breach → only the informational "monitoring" entry.
        let metrics = perf(0.99, 0.001, 0.99, 0, 100.0);
        let mut platforms: HashMap<QuantumPlatform, &PlatformPerformanceMetrics> = HashMap::new();
        platforms.insert(
            QuantumPlatform::IBMQuantum {
                device_name: "test".to_string(),
                hub: None,
            },
            &metrics,
        );
        let cpa = CrossPlatformAnalysis {
            platform_comparison: HashMap::new(),
            best_platform_per_metric: HashMap::new(),
            statistical_significance_tests: HashMap::new(),
        };
        let recs = generate_recommendations_from_perf_metrics(&platforms, &cpa);
        assert_eq!(recs.len(), 1, "expected single info entry, got {recs:?}");
        assert_eq!(recs[0].recommendation_type, "monitoring");
    }

    #[test]
    fn test_recommendations_breach_thresholds() {
        // Breach error_rate, fidelity, availability, exec time, throughput.
        let metrics = perf(0.50, 0.20, 0.50, 5, 0.1);
        let mut platforms: HashMap<QuantumPlatform, &PlatformPerformanceMetrics> = HashMap::new();
        platforms.insert(
            QuantumPlatform::IBMQuantum {
                device_name: "test".to_string(),
                hub: None,
            },
            &metrics,
        );
        let cpa = CrossPlatformAnalysis {
            platform_comparison: HashMap::new(),
            best_platform_per_metric: HashMap::new(),
            statistical_significance_tests: HashMap::new(),
        };
        let recs = generate_recommendations_from_perf_metrics(&platforms, &cpa);
        assert!(
            recs.len() >= 5,
            "expected >=5 recommendations, got {}: {recs:?}",
            recs.len()
        );
        assert!(recs.iter().all(|r| r.recommendation_type != "monitoring"));
        for w in recs.windows(2) {
            assert!(w[0].priority <= w[1].priority, "not sorted: {recs:?}");
        }
    }
}
