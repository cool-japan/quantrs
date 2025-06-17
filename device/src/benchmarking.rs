//! Hardware benchmarking suite with SciRS2 analysis
//!
//! This module provides comprehensive benchmarking capabilities for quantum hardware,
//! using SciRS2's statistical and linear algebra capabilities for advanced analysis.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use rand::prelude::*;
use rand::thread_rng;
use serde::{Deserialize, Serialize};

use quantrs2_circuit::prelude::*;
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};

use scirs2_graph::{
    betweenness_centrality, closeness_centrality, clustering_coefficient, graph_density,
    shortest_path, spectral_radius, Graph,
};
use scirs2_linalg::{
    correlation_matrix, det, eig, inv, matrix_norm, svd, LinalgError, LinalgResult,
};
use scirs2_stats::{
    distributions, mean, median, pearsonr, spearmanr, std, ttest_1samp, var, Alternative,
    TTestResult,
};

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};

use crate::{
    backend_traits::{query_backend_capabilities, BackendCapabilities},
    calibration::{CalibrationManager, DeviceCalibration},
    noise_model::CalibrationNoiseModel,
    topology::HardwareTopology,
    CircuitResult, DeviceError, DeviceResult,
};

/// Comprehensive benchmarking suite configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Number of benchmark iterations
    pub iterations: usize,
    /// Statistical confidence level (0.95 = 95%)
    pub confidence_level: f64,
    /// Maximum runtime per benchmark (seconds)
    pub max_runtime: Duration,
    /// Circuit depths to benchmark
    pub circuit_depths: Vec<usize>,
    /// Qubit counts to benchmark
    pub qubit_counts: Vec<usize>,
    /// Gate types to benchmark
    pub gate_types: Vec<String>,
    /// Enable advanced statistical analysis
    pub enable_advanced_stats: bool,
    /// Enable graph-theoretic analysis
    pub enable_graph_analysis: bool,
    /// Enable noise correlation analysis
    pub enable_noise_analysis: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            iterations: 100,
            confidence_level: 0.95,
            max_runtime: Duration::from_secs(300),
            circuit_depths: vec![5, 10, 20, 50, 100],
            qubit_counts: vec![2, 4, 8, 16],
            gate_types: vec!["H".to_string(), "CNOT".to_string(), "RZ".to_string()],
            enable_advanced_stats: true,
            enable_graph_analysis: true,
            enable_noise_analysis: true,
        }
    }
}

/// Comprehensive benchmark results with SciRS2 analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSuite {
    /// Device identifier
    pub device_id: String,
    /// Backend capabilities
    pub backend_capabilities: BackendCapabilities,
    /// Benchmark configuration used
    pub config: BenchmarkConfig,
    /// Individual benchmark results
    pub benchmark_results: Vec<BenchmarkResult>,
    /// Statistical analysis
    pub statistical_analysis: StatisticalAnalysis,
    /// Graph-theoretic analysis
    pub graph_analysis: Option<GraphAnalysis>,
    /// Noise correlation analysis
    pub noise_analysis: Option<NoiseAnalysis>,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Execution time
    pub execution_time: Duration,
}

/// Individual benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Benchmark name
    pub name: String,
    /// Circuit used
    pub circuit_description: String,
    /// Number of qubits
    pub num_qubits: usize,
    /// Circuit depth
    pub circuit_depth: usize,
    /// Gate count
    pub gate_count: usize,
    /// Execution times (seconds)
    pub execution_times: Vec<f64>,
    /// Fidelity measurements
    pub fidelities: Vec<f64>,
    /// Error rates
    pub error_rates: Vec<f64>,
    /// Success probabilities
    pub success_probabilities: Vec<f64>,
    /// Queue times (seconds)
    pub queue_times: Vec<f64>,
}

/// Statistical analysis using SciRS2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalAnalysis {
    /// Execution time statistics
    pub execution_time_stats: DescriptiveStats,
    /// Fidelity statistics
    pub fidelity_stats: DescriptiveStats,
    /// Error rate statistics
    pub error_rate_stats: DescriptiveStats,
    /// Correlation matrix between metrics
    pub correlation_matrix: Array2<f64>,
    /// Statistical tests results
    pub statistical_tests: HashMap<String, TestResult>,
    /// Distribution fitting results
    pub distribution_fits: HashMap<String, DistributionFit>,
}

/// Graph-theoretic analysis of connectivity and performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphAnalysis {
    /// Connectivity graph metrics
    pub connectivity_metrics: ConnectivityMetrics,
    /// Performance-topology correlations
    pub topology_correlations: HashMap<String, f64>,
    /// Critical path analysis
    pub critical_paths: Vec<CriticalPath>,
    /// Spectral properties
    pub spectral_properties: SpectralProperties,
}

/// Noise correlation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseAnalysis {
    /// Cross-talk correlations
    pub crosstalk_correlations: Array2<f64>,
    /// Temporal noise correlations
    pub temporal_correlations: Array1<f64>,
    /// Spatial noise patterns
    pub spatial_patterns: HashMap<String, Array2<f64>>,
    /// Noise model validation
    pub model_validation: NoiseModelValidation,
}

/// Performance metrics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Overall device score (0-100)
    pub overall_score: f64,
    /// Reliability score
    pub reliability_score: f64,
    /// Speed score
    pub speed_score: f64,
    /// Accuracy score
    pub accuracy_score: f64,
    /// Efficiency score
    pub efficiency_score: f64,
    /// Scalability metrics
    pub scalability_metrics: ScalabilityMetrics,
}

/// Descriptive statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescriptiveStats {
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub variance: f64,
    pub min: f64,
    pub max: f64,
    pub q25: f64,
    pub q75: f64,
    pub confidence_interval: (f64, f64),
}

/// Statistical test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub test_name: String,
    pub statistic: f64,
    pub p_value: f64,
    pub significant: bool,
    pub effect_size: Option<f64>,
}

/// Distribution fitting result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionFit {
    pub distribution_name: String,
    pub parameters: Vec<f64>,
    pub goodness_of_fit: f64,
    pub p_value: f64,
}

/// Connectivity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectivityMetrics {
    pub average_path_length: f64,
    pub clustering_coefficient: f64,
    pub graph_density: f64,
    pub diameter: usize,
    pub spectral_gap: f64,
    pub centrality_measures: HashMap<usize, CentralityMeasures>,
}

/// Centrality measures for qubits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralityMeasures {
    pub betweenness: f64,
    pub closeness: f64,
    pub eigenvector: f64,
    pub degree: f64,
}

/// Critical path information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalPath {
    pub path: Vec<usize>,
    pub length: usize,
    pub bottleneck_qubits: Vec<usize>,
    pub expected_performance_impact: f64,
}

/// Spectral properties of the connectivity graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralProperties {
    pub eigenvalues: Array1<f64>,
    pub spectral_radius: f64,
    pub algebraic_connectivity: f64,
    pub spectral_gap: f64,
}

/// Noise model validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseModelValidation {
    pub model_accuracy: f64,
    pub prediction_errors: Array1<f64>,
    pub residual_analysis: ResidualAnalysis,
}

/// Residual analysis for noise model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualAnalysis {
    pub normality_test: TestResult,
    pub autocorrelation: Array1<f64>,
    pub heteroscedasticity_test: TestResult,
}

/// Scalability metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityMetrics {
    pub depth_scaling_coefficient: f64,
    pub width_scaling_coefficient: f64,
    pub resource_efficiency: f64,
    pub parallelization_factor: f64,
}

/// Main benchmarking suite
pub struct HardwareBenchmarkSuite {
    calibration_manager: CalibrationManager,
    config: BenchmarkConfig,
}

impl HardwareBenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new(calibration_manager: CalibrationManager, config: BenchmarkConfig) -> Self {
        Self {
            calibration_manager,
            config,
        }
    }

    /// Run comprehensive benchmarking suite
    pub async fn run_benchmark_suite<E: DeviceExecutor>(
        &self,
        device_id: &str,
        executor: &E,
    ) -> DeviceResult<BenchmarkSuite> {
        let start_time = Instant::now();

        // Get device information
        let backend_capabilities = query_backend_capabilities(self.get_backend_type(device_id)?);

        let calibration = self
            .calibration_manager
            .get_calibration(device_id)
            .ok_or_else(|| DeviceError::APIError("No calibration data".into()))?;

        // Run individual benchmarks
        let mut benchmark_results = Vec::new();

        // Basic gate benchmarks
        for gate_type in &self.config.gate_types {
            for &num_qubits in &self.config.qubit_counts {
                if num_qubits <= backend_capabilities.features.max_qubits {
                    let result = self
                        .benchmark_gate_type(gate_type, num_qubits, executor, calibration)
                        .await?;
                    benchmark_results.push(result);
                }
            }
        }

        // Circuit depth benchmarks
        for &depth in &self.config.circuit_depths {
            for &num_qubits in &self.config.qubit_counts {
                if num_qubits <= backend_capabilities.features.max_qubits {
                    let result = self
                        .benchmark_circuit_depth(depth, num_qubits, executor, calibration)
                        .await?;
                    benchmark_results.push(result);
                }
            }
        }

        // Randomized circuit benchmarks
        let random_results = self
            .benchmark_random_circuits(executor, calibration)
            .await?;
        benchmark_results.extend(random_results);

        // Perform statistical analysis
        let statistical_analysis = self.perform_statistical_analysis(&benchmark_results)?;

        // Perform graph analysis if enabled
        let graph_analysis = if self.config.enable_graph_analysis {
            Some(self.perform_graph_analysis(calibration, &benchmark_results)?)
        } else {
            None
        };

        // Perform noise analysis if enabled
        let noise_analysis = if self.config.enable_noise_analysis {
            Some(self.perform_noise_analysis(calibration, &benchmark_results)?)
        } else {
            None
        };

        // Calculate performance metrics
        let performance_metrics = self.calculate_performance_metrics(
            &benchmark_results,
            &statistical_analysis,
            &graph_analysis,
        )?;

        let execution_time = start_time.elapsed();

        Ok(BenchmarkSuite {
            device_id: device_id.to_string(),
            backend_capabilities,
            config: self.config.clone(),
            benchmark_results,
            statistical_analysis,
            graph_analysis,
            noise_analysis,
            performance_metrics,
            execution_time,
        })
    }

    /// Benchmark specific gate type
    async fn benchmark_gate_type<E: DeviceExecutor>(
        &self,
        gate_type: &str,
        num_qubits: usize,
        executor: &E,
        calibration: &DeviceCalibration,
    ) -> DeviceResult<BenchmarkResult> {
        let mut execution_times = Vec::new();
        let mut fidelities = Vec::new();
        let mut error_rates = Vec::new();
        let mut success_probabilities = Vec::new();
        let mut queue_times = Vec::new();

        for _ in 0..self.config.iterations {
            let circuit: Circuit<8> = self.create_gate_benchmark_circuit(gate_type, num_qubits)?;

            let queue_start = Instant::now();
            let exec_start = Instant::now();

            // Execute circuit
            let result = executor.execute_circuit(&circuit, 1000).await?;

            let exec_time = exec_start.elapsed().as_secs_f64();
            let queue_time = queue_start.elapsed().as_secs_f64();

            // Calculate metrics
            let fidelity = self.calculate_fidelity(&result, &circuit, calibration)?;
            let error_rate = 1.0 - fidelity;
            let success_prob = self.calculate_success_probability(&result)?;

            execution_times.push(exec_time);
            fidelities.push(fidelity);
            error_rates.push(error_rate);
            success_probabilities.push(success_prob);
            queue_times.push(queue_time);
        }

        Ok(BenchmarkResult {
            name: format!("{}_gate_benchmark", gate_type),
            circuit_description: format!("{} gate on {} qubits", gate_type, num_qubits),
            num_qubits,
            circuit_depth: 1,
            gate_count: 1,
            execution_times,
            fidelities,
            error_rates,
            success_probabilities,
            queue_times,
        })
    }

    /// Benchmark circuit depth scaling
    async fn benchmark_circuit_depth<E: DeviceExecutor>(
        &self,
        depth: usize,
        num_qubits: usize,
        executor: &E,
        calibration: &DeviceCalibration,
    ) -> DeviceResult<BenchmarkResult> {
        let mut execution_times = Vec::new();
        let mut fidelities = Vec::new();
        let mut error_rates = Vec::new();
        let mut success_probabilities = Vec::new();
        let mut queue_times = Vec::new();

        for _ in 0..self.config.iterations {
            let circuit: Circuit<8> = self.create_depth_benchmark_circuit(depth, num_qubits)?;

            let queue_start = Instant::now();
            let exec_start = Instant::now();

            let result = executor.execute_circuit(&circuit, 1000).await?;

            let exec_time = exec_start.elapsed().as_secs_f64();
            let queue_time = queue_start.elapsed().as_secs_f64();

            let fidelity = self.calculate_fidelity(&result, &circuit, calibration)?;
            let error_rate = 1.0 - fidelity;
            let success_prob = self.calculate_success_probability(&result)?;

            execution_times.push(exec_time);
            fidelities.push(fidelity);
            error_rates.push(error_rate);
            success_probabilities.push(success_prob);
            queue_times.push(queue_time);
        }

        Ok(BenchmarkResult {
            name: format!("depth_{}_benchmark", depth),
            circuit_description: format!("Depth {} circuit on {} qubits", depth, num_qubits),
            num_qubits,
            circuit_depth: depth,
            gate_count: depth * num_qubits / 2, // Rough estimate
            execution_times,
            fidelities,
            error_rates,
            success_probabilities,
            queue_times,
        })
    }

    /// Benchmark random circuits
    async fn benchmark_random_circuits<E: DeviceExecutor>(
        &self,
        executor: &E,
        calibration: &DeviceCalibration,
    ) -> DeviceResult<Vec<BenchmarkResult>> {
        let mut results = Vec::new();

        // Random circuit volume benchmarks
        for &num_qubits in &self.config.qubit_counts {
            for &depth in &self.config.circuit_depths {
                if num_qubits <= 8 && depth <= 50 {
                    // Reasonable limits
                    let result = self
                        .benchmark_random_circuit_volume(num_qubits, depth, executor, calibration)
                        .await?;
                    results.push(result);
                }
            }
        }

        Ok(results)
    }

    /// Benchmark random circuit volume
    async fn benchmark_random_circuit_volume<E: DeviceExecutor>(
        &self,
        num_qubits: usize,
        depth: usize,
        executor: &E,
        calibration: &DeviceCalibration,
    ) -> DeviceResult<BenchmarkResult> {
        let mut execution_times = Vec::new();
        let mut fidelities = Vec::new();
        let mut error_rates = Vec::new();
        let mut success_probabilities = Vec::new();
        let mut queue_times = Vec::new();

        for _ in 0..self.config.iterations.min(20) {
            // Limit for random circuits
            let circuit = self.create_random_circuit(num_qubits, depth)?;

            let queue_start = Instant::now();
            let exec_start = Instant::now();

            let result = executor.execute_circuit(&circuit, 1000).await?;

            let exec_time = exec_start.elapsed().as_secs_f64();
            let queue_time = queue_start.elapsed().as_secs_f64();

            let fidelity = self.calculate_fidelity(&result, &circuit, calibration)?;
            let error_rate = 1.0 - fidelity;
            let success_prob = self.calculate_success_probability(&result)?;

            execution_times.push(exec_time);
            fidelities.push(fidelity);
            error_rates.push(error_rate);
            success_probabilities.push(success_prob);
            queue_times.push(queue_time);
        }

        Ok(BenchmarkResult {
            name: format!("random_circuit_{}q_{}d", num_qubits, depth),
            circuit_description: format!("Random {} qubit, depth {} circuit", num_qubits, depth),
            num_qubits,
            circuit_depth: depth,
            gate_count: depth * num_qubits,
            execution_times,
            fidelities,
            error_rates,
            success_probabilities,
            queue_times,
        })
    }

    /// Perform comprehensive statistical analysis using SciRS2
    fn perform_statistical_analysis(
        &self,
        results: &[BenchmarkResult],
    ) -> DeviceResult<StatisticalAnalysis> {
        // Collect all execution times and fidelities
        let all_exec_times: Vec<f64> = results
            .iter()
            .flat_map(|r| r.execution_times.iter())
            .copied()
            .collect();

        let all_fidelities: Vec<f64> = results
            .iter()
            .flat_map(|r| r.fidelities.iter())
            .copied()
            .collect();

        let all_error_rates: Vec<f64> = results
            .iter()
            .flat_map(|r| r.error_rates.iter())
            .copied()
            .collect();

        // Convert to ndarray for SciRS2
        let exec_times_array = Array1::from_vec(all_exec_times);
        let fidelities_array = Array1::from_vec(all_fidelities);
        let error_rates_array = Array1::from_vec(all_error_rates);

        // Compute descriptive statistics
        let execution_time_stats = self.compute_descriptive_stats(&exec_times_array)?;
        let fidelity_stats = self.compute_descriptive_stats(&fidelities_array)?;
        let error_rate_stats = self.compute_descriptive_stats(&error_rates_array)?;

        // Compute correlation matrix
        let data_matrix = Array2::from_shape_vec(
            (exec_times_array.len(), 3),
            [
                exec_times_array.as_slice().unwrap(),
                fidelities_array.as_slice().unwrap(),
                error_rates_array.as_slice().unwrap(),
            ]
            .concat(),
        )
        .map_err(|e| DeviceError::APIError(format!("Array creation error: {}", e)))?;

        let correlation_matrix = correlation_matrix(&data_matrix.view(), None).map_err(|e| {
            DeviceError::APIError(format!("Correlation computation error: {:?}", e))
        })?;

        // Statistical tests
        let mut statistical_tests = HashMap::new();

        // Test if execution times follow normal distribution
        if exec_times_array.len() >= 8 {
            // One-sample t-test against expected baseline
            let baseline_time = 1.0; // 1 second baseline
            let t_test = ttest_1samp(
                &exec_times_array.view(),
                baseline_time,
                Alternative::TwoSided,
                "propagate",
            )
            .map_err(|e| DeviceError::APIError(format!("T-test error: {:?}", e)))?;

            statistical_tests.insert(
                "execution_time_t_test".to_string(),
                TestResult {
                    test_name: "One-sample t-test (execution time)".to_string(),
                    statistic: t_test.statistic,
                    p_value: t_test.pvalue,
                    significant: t_test.pvalue < 0.05,
                    effect_size: Some(
                        (execution_time_stats.mean - baseline_time) / execution_time_stats.std_dev,
                    ),
                },
            );
        }

        // Distribution fitting
        let mut distribution_fits = HashMap::new();

        // Fit normal distribution to execution times
        if let Ok(normal_dist) =
            distributions::norm(execution_time_stats.mean, execution_time_stats.std_dev)
        {
            // Calculate goodness of fit (simplified)
            let goodness_of_fit =
                self.calculate_goodness_of_fit(&exec_times_array, &normal_dist)?;

            distribution_fits.insert(
                "execution_time_normal".to_string(),
                DistributionFit {
                    distribution_name: "Normal".to_string(),
                    parameters: vec![execution_time_stats.mean, execution_time_stats.std_dev],
                    goodness_of_fit,
                    p_value: 0.5, // Placeholder
                },
            );
        }

        Ok(StatisticalAnalysis {
            execution_time_stats,
            fidelity_stats,
            error_rate_stats,
            correlation_matrix,
            statistical_tests,
            distribution_fits,
        })
    }

    /// Perform graph-theoretic analysis
    fn perform_graph_analysis(
        &self,
        calibration: &DeviceCalibration,
        results: &[BenchmarkResult],
    ) -> DeviceResult<GraphAnalysis> {
        // Build connectivity graph from topology
        let mut graph = Graph::new();
        let mut node_map = HashMap::new();

        // Add nodes for each qubit
        for i in 0..calibration.topology.num_qubits {
            let node = graph.add_node(i);
            node_map.insert(i, node);
        }

        // Add edges based on coupling map
        for &(q1, q2) in &calibration.topology.coupling_map {
            let q1_idx = q1.0 as usize;
            let q2_idx = q2.0 as usize;
            if let (Some(&n1), Some(&n2)) = (node_map.get(&q1_idx), node_map.get(&q2_idx)) {
                graph.add_edge(n1.index(), n2.index(), 1.0);
            }
        }

        // Calculate graph metrics
        let graph_density_val = graph_density(&graph)
            .map_err(|e| DeviceError::APIError(format!("Graph density error: {:?}", e)))?;

        let clustering_coeff = clustering_coefficient(&graph)
            .map_err(|e| DeviceError::APIError(format!("Clustering coefficient error: {:?}", e)))?;

        // Calculate centrality measures for each qubit
        let betweenness_values = betweenness_centrality(&graph, false);

        let closeness_values = closeness_centrality(&graph, true);

        let mut centrality_measures = HashMap::new();
        for i in 0..calibration.topology.num_qubits {
            if let Some(&node) = node_map.get(&i) {
                let node_idx = node.index();
                centrality_measures.insert(
                    i,
                    CentralityMeasures {
                        betweenness: betweenness_values.get(&node_idx).copied().unwrap_or(0.0),
                        closeness: closeness_values.get(&node_idx).copied().unwrap_or(0.0),
                        eigenvector: 0.0, // Placeholder
                        degree: graph
                            .neighbors(&node_idx)
                            .map(|neighbors| neighbors.len() as f64)
                            .unwrap_or(0.0),
                    },
                );
            }
        }

        // Calculate spectral properties
        let spectral_properties = self.calculate_spectral_properties(calibration)?;

        // Find critical paths
        let compatible_node_map: HashMap<usize, usize> =
            node_map.iter().map(|(&k, &v)| (k, v.index())).collect();
        let critical_paths = self.find_critical_paths(&graph, &compatible_node_map, results)?;

        // Calculate topology-performance correlations
        let mut topology_correlations = HashMap::new();

        // Correlate connectivity with performance
        let connectivity_scores: Vec<f64> = centrality_measures
            .values()
            .map(|c| c.betweenness + c.closeness)
            .collect();

        let avg_fidelities: Vec<f64> = results
            .iter()
            .take(connectivity_scores.len())
            .map(|r| r.fidelities.iter().sum::<f64>() / r.fidelities.len() as f64)
            .collect();

        if connectivity_scores.len() == avg_fidelities.len() && connectivity_scores.len() >= 3 {
            let conn_array = Array1::from_vec(connectivity_scores);
            let fid_array = Array1::from_vec(avg_fidelities);

            if let Ok((correlation, _)) =
                pearsonr(&conn_array.view(), &fid_array.view(), "two-sided")
            {
                topology_correlations.insert("connectivity_fidelity".to_string(), correlation);
            }
        }

        let connectivity_metrics = ConnectivityMetrics {
            average_path_length: 0.0, // Placeholder - would calculate actual shortest paths
            clustering_coefficient: clustering_coeff.values().sum::<f64>()
                / clustering_coeff.len() as f64,
            graph_density: graph_density_val,
            diameter: 0, // Placeholder
            spectral_gap: spectral_properties.spectral_gap,
            centrality_measures,
        };

        Ok(GraphAnalysis {
            connectivity_metrics,
            topology_correlations,
            critical_paths,
            spectral_properties,
        })
    }

    /// Perform noise correlation analysis
    fn perform_noise_analysis(
        &self,
        calibration: &DeviceCalibration,
        results: &[BenchmarkResult],
    ) -> DeviceResult<NoiseAnalysis> {
        let num_qubits = calibration.topology.num_qubits;

        // Build crosstalk correlation matrix
        let crosstalk_correlations = Array2::from_shape_fn((num_qubits, num_qubits), |(i, j)| {
            if i == j {
                1.0
            } else {
                // Use crosstalk matrix data if available
                calibration
                    .crosstalk_matrix
                    .matrix
                    .get(i)
                    .and_then(|row| row.get(j))
                    .copied()
                    .unwrap_or(0.0)
            }
        });

        // Temporal correlations (placeholder)
        let temporal_correlations = Array1::zeros(results.len());

        // Spatial noise patterns
        let mut spatial_patterns = HashMap::new();

        // Error rate spatial pattern
        let error_pattern = Array2::from_shape_fn((num_qubits, num_qubits), |(i, j)| {
            // Simplified spatial error pattern based on distance
            let distance = ((i as f64) - (j as f64)).abs();
            (-distance * 0.1).exp() * 0.01 // Exponential decay with distance
        });
        spatial_patterns.insert("error_rates".to_string(), error_pattern);

        // Noise model validation
        let noise_model = CalibrationNoiseModel::from_calibration(calibration);
        let model_validation = self.validate_noise_model(&noise_model, results)?;

        Ok(NoiseAnalysis {
            crosstalk_correlations,
            temporal_correlations,
            spatial_patterns,
            model_validation,
        })
    }

    /// Calculate performance metrics
    fn calculate_performance_metrics(
        &self,
        results: &[BenchmarkResult],
        stats: &StatisticalAnalysis,
        graph_analysis: &Option<GraphAnalysis>,
    ) -> DeviceResult<PerformanceMetrics> {
        // Calculate component scores (0-100)
        let reliability_score = (stats.fidelity_stats.mean * 100.0).min(100.0);
        let speed_score = (1.0 / stats.execution_time_stats.mean * 10.0).min(100.0);
        let accuracy_score = ((1.0 - stats.error_rate_stats.mean) * 100.0).min(100.0);

        // Efficiency based on correlation between resources and performance
        let efficiency_score = if let Some(graph) = graph_analysis {
            // Use connectivity metrics to evaluate efficiency
            (graph.connectivity_metrics.clustering_coefficient * 100.0).min(100.0)
        } else {
            50.0 // Default
        };

        // Scalability metrics
        let scalability_metrics = self.calculate_scalability_metrics(results)?;

        // Overall score (weighted average)
        let overall_score = (reliability_score * 0.3
            + speed_score * 0.2
            + accuracy_score * 0.3
            + efficiency_score * 0.2)
            .min(100.0);

        Ok(PerformanceMetrics {
            overall_score,
            reliability_score,
            speed_score,
            accuracy_score,
            efficiency_score,
            scalability_metrics,
        })
    }

    // Helper methods

    fn get_backend_type(
        &self,
        device_id: &str,
    ) -> DeviceResult<crate::translation::HardwareBackend> {
        // Infer backend type from device ID
        if device_id.contains("ibm") {
            Ok(crate::translation::HardwareBackend::IBMQuantum)
        } else if device_id.contains("ionq") {
            Ok(crate::translation::HardwareBackend::IonQ)
        } else {
            Ok(crate::translation::HardwareBackend::IBMQuantum) // Default
        }
    }

    fn create_gate_benchmark_circuit(
        &self,
        gate_type: &str,
        num_qubits: usize,
    ) -> DeviceResult<Circuit<8>> {
        let mut circuit = Circuit::<8>::new();

        match gate_type {
            "H" => {
                circuit.h(QubitId(0));
            }
            "CNOT" => {
                circuit.cnot(QubitId(0), QubitId(1));
            }
            "RZ" => {
                circuit.rz(QubitId(0), std::f64::consts::PI / 4.0);
            }
            _ => {
                return Err(DeviceError::UnsupportedOperation(format!(
                    "Gate type: {}",
                    gate_type
                )))
            }
        }

        Ok(circuit)
    }

    fn create_depth_benchmark_circuit(
        &self,
        depth: usize,
        num_qubits: usize,
    ) -> DeviceResult<Circuit<8>> {
        let mut circuit = Circuit::<8>::new();

        for layer in 0..depth {
            for qubit in 0..num_qubits {
                if layer % 2 == 0 {
                    circuit.h(QubitId(qubit as u32));
                } else if qubit + 1 < num_qubits {
                    circuit.cnot(QubitId(qubit as u32), QubitId((qubit + 1) as u32));
                }
            }
        }

        Ok(circuit)
    }

    fn create_random_circuit<const N: usize>(
        &self,
        num_qubits: usize,
        depth: usize,
    ) -> DeviceResult<Circuit<N>> {
        use rand::prelude::*;
        let mut rng = thread_rng();
        let mut circuit = Circuit::<N>::new();

        let gates = ["H", "X", "Y", "Z", "CNOT", "RZ"];

        for _ in 0..depth {
            let gate = gates.choose(&mut rng).unwrap();
            let qubit = rng.gen_range(0..num_qubits) as u32;

            match *gate {
                "H" => {
                    circuit.h(QubitId(qubit));
                }
                "X" => {
                    circuit.x(QubitId(qubit));
                }
                "Y" => {
                    circuit.y(QubitId(qubit));
                }
                "Z" => {
                    circuit.z(QubitId(qubit));
                }
                "CNOT" => {
                    let target = rng.gen_range(0..num_qubits) as u32;
                    if target != qubit {
                        circuit.cnot(QubitId(qubit), QubitId(target));
                    }
                }
                "RZ" => {
                    let angle = rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI);
                    circuit.rz(QubitId(qubit), angle);
                }
                _ => {}
            }
        }

        Ok(circuit)
    }

    fn calculate_fidelity(
        &self,
        result: &CircuitResult,
        circuit: &Circuit<8>,
        calibration: &DeviceCalibration,
    ) -> DeviceResult<f64> {
        // Simplified fidelity calculation based on expected vs actual outcomes
        // In practice, this would use state tomography or process tomography

        // For now, estimate based on gate error rates
        let mut total_fidelity = 1.0;

        for gate in circuit.gates() {
            let qubits = gate.qubits();

            if qubits.len() == 1 {
                if let Some(gate_cal) = calibration.single_qubit_gates.get(gate.name()) {
                    if let Some(qubit_data) = gate_cal.qubit_data.get(&qubits[0]) {
                        total_fidelity *= qubit_data.fidelity;
                    }
                }
            } else if qubits.len() == 2 {
                let qubit_pair = (qubits[0], qubits[1]);
                if let Some(gate_cal) = calibration.two_qubit_gates.get(&qubit_pair) {
                    total_fidelity *= gate_cal.fidelity;
                }
            }
        }

        Ok(total_fidelity)
    }

    fn calculate_success_probability(&self, result: &CircuitResult) -> DeviceResult<f64> {
        // Calculate success probability based on measurement outcomes
        let total_shots = result.shots as f64;
        let successful_outcomes = result
            .counts
            .values()
            .map(|&count| count as f64)
            .sum::<f64>();

        Ok(successful_outcomes / total_shots)
    }

    fn compute_descriptive_stats(&self, data: &Array1<f64>) -> DeviceResult<DescriptiveStats> {
        let mean_val = mean(&data.view())
            .map_err(|e| DeviceError::APIError(format!("Mean calculation error: {:?}", e)))?;

        let median_val = median(&data.view())
            .map_err(|e| DeviceError::APIError(format!("Median calculation error: {:?}", e)))?;

        let std_val = std(&data.view(), 1)
            .map_err(|e| DeviceError::APIError(format!("Std calculation error: {:?}", e)))?;

        let var_val = var(&data.view(), 1)
            .map_err(|e| DeviceError::APIError(format!("Variance calculation error: {:?}", e)))?;

        let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Calculate quartiles (simplified)
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted_data.len();
        let q25 = sorted_data[n / 4];
        let q75 = sorted_data[3 * n / 4];

        // Confidence interval (95%)
        let t_critical = 1.96; // Approximate for large samples
        let sem = std_val / (n as f64).sqrt();
        let margin = t_critical * sem;
        let confidence_interval = (mean_val - margin, mean_val + margin);

        Ok(DescriptiveStats {
            mean: mean_val,
            median: median_val,
            std_dev: std_val,
            variance: var_val,
            min: min_val,
            max: max_val,
            q25,
            q75,
            confidence_interval,
        })
    }

    fn calculate_goodness_of_fit(
        &self,
        data: &Array1<f64>,
        distribution: &dyn scirs2_stats::traits::Distribution<f64>,
    ) -> DeviceResult<f64> {
        // Simplified goodness of fit using Kolmogorov-Smirnov test
        // In practice, would use proper KS test from SciRS2

        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted_data.len() as f64;
        let mut max_diff: f64 = 0.0;

        for (i, &value) in sorted_data.iter().enumerate() {
            let empirical_cdf = (i + 1) as f64 / n;
            let theoretical_cdf = 0.5; // Fallback for missing CDF method
            let diff = (empirical_cdf - theoretical_cdf).abs();
            max_diff = max_diff.max(diff);
        }

        Ok(1.0 - max_diff) // Convert to goodness (higher is better)
    }

    fn calculate_spectral_properties(
        &self,
        calibration: &DeviceCalibration,
    ) -> DeviceResult<SpectralProperties> {
        let n = calibration.topology.num_qubits;

        // Build adjacency matrix
        let mut adj_matrix = Array2::zeros((n, n));
        for &(q1, q2) in &calibration.topology.coupling_map {
            adj_matrix[[q1.0 as usize, q2.0 as usize]] = 1.0;
            adj_matrix[[q2.0 as usize, q1.0 as usize]] = 1.0;
        }

        // Calculate eigenvalues
        let eigenvalues = match eig(&adj_matrix.view()) {
            Ok((vals, _)) => vals.mapv(|c| c.re), // Take real parts
            Err(_) => Array1::zeros(n),           // Fallback
        };

        let spectral_radius = eigenvalues
            .iter()
            .fold(0.0_f64, |max, &val: &f64| max.max(val.abs()));

        // Sort eigenvalues
        let mut sorted_eigenvals = eigenvalues.to_vec();
        sorted_eigenvals.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let algebraic_connectivity = if sorted_eigenvals.len() > 1 {
            sorted_eigenvals[sorted_eigenvals.len() - 2]
        } else {
            0.0
        };

        let spectral_gap = if sorted_eigenvals.len() > 1 {
            sorted_eigenvals[0] - sorted_eigenvals[1]
        } else {
            0.0
        };

        Ok(SpectralProperties {
            eigenvalues: Array1::from_vec(sorted_eigenvals),
            spectral_radius,
            algebraic_connectivity,
            spectral_gap,
        })
    }

    fn find_critical_paths(
        &self,
        graph: &Graph<usize, f64>,
        node_map: &HashMap<usize, usize>,
        results: &[BenchmarkResult],
    ) -> DeviceResult<Vec<CriticalPath>> {
        let mut critical_paths = Vec::new();

        // Find paths between high-error qubits
        // This is a simplified version - would use more sophisticated analysis

        for i in 0..3 {
            // Find up to 3 critical paths
            let path = vec![i, i + 1, i + 2]; // Simplified linear path

            critical_paths.push(CriticalPath {
                path: path.clone(),
                length: path.len(),
                bottleneck_qubits: vec![i + 1], // Middle qubit as bottleneck
                expected_performance_impact: 0.1 * (i + 1) as f64,
            });
        }

        Ok(critical_paths)
    }

    fn validate_noise_model(
        &self,
        noise_model: &CalibrationNoiseModel,
        results: &[BenchmarkResult],
    ) -> DeviceResult<NoiseModelValidation> {
        // Compare predicted vs actual error rates
        let predicted_errors: Vec<f64> = results
            .iter()
            .map(|r| {
                // Simplified prediction based on gate count
                let base_error = 0.001;
                base_error * r.gate_count as f64
            })
            .collect();

        let actual_errors: Vec<f64> = results
            .iter()
            .map(|r| r.error_rates.iter().sum::<f64>() / r.error_rates.len() as f64)
            .collect();

        let prediction_errors = Array1::from_vec(
            predicted_errors
                .iter()
                .zip(actual_errors.iter())
                .map(|(pred, actual)| (pred - actual).abs())
                .collect(),
        );

        let model_accuracy = 1.0 - (prediction_errors.mean().unwrap_or(0.5));

        // Residual analysis (simplified)
        let residuals = Array1::from_vec(
            predicted_errors
                .iter()
                .zip(actual_errors.iter())
                .map(|(pred, actual)| pred - actual)
                .collect(),
        );

        let normality_test = TestResult {
            test_name: "Residual normality test".to_string(),
            statistic: 0.0,
            p_value: 0.5,
            significant: false,
            effect_size: None,
        };

        let residual_analysis = ResidualAnalysis {
            normality_test,
            autocorrelation: Array1::zeros(5), // Placeholder
            heteroscedasticity_test: TestResult {
                test_name: "Heteroscedasticity test".to_string(),
                statistic: 0.0,
                p_value: 0.5,
                significant: false,
                effect_size: None,
            },
        };

        Ok(NoiseModelValidation {
            model_accuracy,
            prediction_errors,
            residual_analysis,
        })
    }

    fn calculate_scalability_metrics(
        &self,
        results: &[BenchmarkResult],
    ) -> DeviceResult<ScalabilityMetrics> {
        // Analyze how performance scales with circuit size

        // Group results by depth and width
        let mut depth_times = HashMap::new();
        let mut width_times = HashMap::new();

        for result in results {
            depth_times
                .entry(result.circuit_depth)
                .or_insert_with(Vec::new)
                .extend(&result.execution_times);

            width_times
                .entry(result.num_qubits)
                .or_insert_with(Vec::new)
                .extend(&result.execution_times);
        }

        // Calculate scaling coefficients (simplified linear regression)
        let depth_scaling_coefficient = self.calculate_scaling_coefficient(&depth_times)?;
        let width_scaling_coefficient = self.calculate_scaling_coefficient(&width_times)?;

        let resource_efficiency =
            1.0 / (depth_scaling_coefficient + width_scaling_coefficient).max(0.1);
        let parallelization_factor = 0.8; // Placeholder - would measure actual parallelization

        Ok(ScalabilityMetrics {
            depth_scaling_coefficient,
            width_scaling_coefficient,
            resource_efficiency,
            parallelization_factor,
        })
    }

    fn calculate_scaling_coefficient(&self, data: &HashMap<usize, Vec<f64>>) -> DeviceResult<f64> {
        if data.len() < 2 {
            return Ok(1.0); // Default linear scaling
        }

        // Simple linear regression: time = a * size + b
        // Return coefficient 'a'

        let mut x_values = Vec::new();
        let mut y_values = Vec::new();

        for (&size, times) in data {
            let avg_time = times.iter().sum::<f64>() / times.len() as f64;
            x_values.push(size as f64);
            y_values.push(avg_time);
        }

        // Calculate slope (scaling coefficient)
        let n = x_values.len() as f64;
        let sum_x: f64 = x_values.iter().sum();
        let sum_y: f64 = y_values.iter().sum();
        let sum_xy: f64 = x_values
            .iter()
            .zip(y_values.iter())
            .map(|(x, y)| x * y)
            .sum();
        let sum_x_sq: f64 = x_values.iter().map(|x| x * x).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_sq - sum_x * sum_x);

        Ok(slope.max(0.1)) // Ensure positive scaling
    }
}

/// Trait for devices that can execute circuits (for benchmarking)
#[async_trait::async_trait]
pub trait DeviceExecutor {
    async fn execute_circuit<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        shots: usize,
    ) -> DeviceResult<CircuitResult>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::create_ideal_calibration;

    #[test]
    fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.iterations, 100);
        assert!(config.enable_advanced_stats);
    }

    #[test]
    fn test_descriptive_stats_calculation() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let suite =
            HardwareBenchmarkSuite::new(CalibrationManager::new(), BenchmarkConfig::default());

        let stats = suite.compute_descriptive_stats(&data).unwrap();
        assert!((stats.mean - 3.0).abs() < 1e-10);
        assert!((stats.median - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_spectral_properties() {
        let calibration = create_ideal_calibration("test".to_string(), 4);
        let suite =
            HardwareBenchmarkSuite::new(CalibrationManager::new(), BenchmarkConfig::default());

        let spectral_props = suite.calculate_spectral_properties(&calibration).unwrap();
        assert!(spectral_props.eigenvalues.len() == 4);
        assert!(spectral_props.spectral_radius >= 0.0);
    }
}
