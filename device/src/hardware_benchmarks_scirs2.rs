//! Hardware Benchmarking with SciRS2 Analysis
//!
//! This module provides comprehensive hardware performance benchmarking using SciRS2's
//! advanced statistical analysis and metrics capabilities for quantum device characterization.

use crate::{DeviceError, DeviceResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_stats::{mean, median, std};

/// Hardware benchmark configuration
#[derive(Debug, Clone)]
pub struct HardwareBenchmarkConfig {
    /// Number of benchmark iterations
    pub num_iterations: usize,
    /// Number of qubits to benchmark
    pub num_qubits: usize,
    /// Include gate fidelity benchmarks
    pub benchmark_gate_fidelity: bool,
    /// Include coherence time benchmarks
    pub benchmark_coherence: bool,
    /// Include readout fidelity benchmarks
    pub benchmark_readout: bool,
    /// Confidence level for statistical analysis
    pub confidence_level: f64,
}

impl Default for HardwareBenchmarkConfig {
    fn default() -> Self {
        Self {
            num_iterations: 1000,
            num_qubits: 2,
            benchmark_gate_fidelity: true,
            benchmark_coherence: true,
            benchmark_readout: true,
            confidence_level: 0.95,
        }
    }
}

/// Benchmark result with SciRS2 statistical analysis
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Benchmark name
    pub name: String,
    /// Mean performance metric
    pub mean_value: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Median value
    pub median_value: f64,
    /// Minimum value observed
    pub min_value: f64,
    /// Maximum value observed
    pub max_value: f64,
    /// 95% confidence interval
    pub confidence_interval: (f64, f64),
    /// Number of samples
    pub num_samples: usize,
    /// Statistical significance (p-value if comparing to baseline)
    pub p_value: Option<f64>,
}

/// Comprehensive hardware benchmark report
#[derive(Debug, Clone)]
pub struct HardwareBenchmarkReport {
    /// Device name or identifier
    pub device_name: String,
    /// Individual benchmark results
    pub benchmarks: Vec<BenchmarkResult>,
    /// Overall device score (0-100)
    pub overall_score: f64,
    /// Gate fidelity analysis
    pub gate_fidelity_analysis: Option<GateFidelityAnalysis>,
    /// Coherence time analysis
    pub coherence_analysis: Option<CoherenceAnalysis>,
    /// Readout fidelity analysis
    pub readout_analysis: Option<ReadoutFidelityAnalysis>,
    /// Timestamp of benchmark run
    pub timestamp: std::time::SystemTime,
}

/// Gate fidelity analysis using SciRS2 metrics
#[derive(Debug, Clone)]
pub struct GateFidelityAnalysis {
    /// Single-qubit gate fidelities
    pub single_qubit_fidelity: Array1<f64>,
    /// Two-qubit gate fidelities
    pub two_qubit_fidelity: Array1<f64>,
    /// Average single-qubit fidelity
    pub avg_single_qubit: f64,
    /// Average two-qubit fidelity
    pub avg_two_qubit: f64,
    /// Gate error rates
    pub error_rates: Array1<f64>,
}

/// Coherence time analysis
#[derive(Debug, Clone)]
pub struct CoherenceAnalysis {
    /// T1 relaxation times per qubit (microseconds)
    pub t1_times: Array1<f64>,
    /// T2 dephasing times per qubit (microseconds)
    pub t2_times: Array1<f64>,
    /// Average T1 time
    pub avg_t1: f64,
    /// Average T2 time
    pub avg_t2: f64,
    /// T1/T2 ratio analysis
    pub t1_t2_ratio: Array1<f64>,
}

/// Readout fidelity analysis
#[derive(Debug, Clone)]
pub struct ReadoutFidelityAnalysis {
    /// Readout fidelity per qubit
    pub readout_fidelity: Array1<f64>,
    /// Average readout fidelity
    pub avg_fidelity: f64,
    /// Assignment error matrix (confusion matrix)
    pub assignment_errors: Array2<f64>,
    /// SPAM (State Preparation And Measurement) error
    pub spam_error: f64,
}

/// Hardware benchmarking engine using SciRS2
pub struct HardwareBenchmarker {
    config: HardwareBenchmarkConfig,
    rng: StdRng,
}

impl HardwareBenchmarker {
    /// Create a new hardware benchmarker
    pub fn new(config: HardwareBenchmarkConfig) -> Self {
        Self {
            config,
            rng: StdRng::seed_from_u64(42),
        }
    }

    /// Create benchmarker with default configuration
    pub fn default() -> Self {
        Self::new(HardwareBenchmarkConfig::default())
    }

    /// Run comprehensive hardware benchmarks
    ///
    /// # Arguments
    /// * `device_name` - Name or identifier of the device being benchmarked
    /// * `measurement_data` - Raw measurement data from hardware
    ///
    /// # Returns
    /// Comprehensive benchmark report with SciRS2 statistical analysis
    pub fn run_benchmarks(
        &mut self,
        device_name: &str,
        measurement_data: &BenchmarkMeasurementData,
    ) -> DeviceResult<HardwareBenchmarkReport> {
        let mut benchmarks = Vec::new();

        // Benchmark gate execution times
        let gate_timing = self.benchmark_gate_timing(&measurement_data.gate_times)?;
        benchmarks.push(gate_timing);

        // Benchmark gate fidelities if enabled
        let gate_fidelity_analysis = if self.config.benchmark_gate_fidelity {
            Some(self.analyze_gate_fidelity(&measurement_data.fidelity_data)?)
        } else {
            None
        };

        // Benchmark coherence times if enabled
        let coherence_analysis = if self.config.benchmark_coherence {
            Some(self.analyze_coherence(&measurement_data.coherence_data)?)
        } else {
            None
        };

        // Benchmark readout fidelity if enabled
        let readout_analysis = if self.config.benchmark_readout {
            Some(self.analyze_readout_fidelity(&measurement_data.readout_data)?)
        } else {
            None
        };

        // Compute overall device score
        let overall_score = self.compute_overall_score(
            &gate_fidelity_analysis,
            &coherence_analysis,
            &readout_analysis,
        );

        Ok(HardwareBenchmarkReport {
            device_name: device_name.to_string(),
            benchmarks,
            overall_score,
            gate_fidelity_analysis,
            coherence_analysis,
            readout_analysis,
            timestamp: std::time::SystemTime::now(),
        })
    }

    /// Benchmark gate execution timing
    fn benchmark_gate_timing(&self, gate_times: &Array1<f64>) -> DeviceResult<BenchmarkResult> {
        if gate_times.is_empty() {
            return Err(DeviceError::InvalidInput(
                "Empty gate timing data".to_string(),
            ));
        }

        let mean_time = mean(&gate_times.view())?;
        let std_time = std(&gate_times.view(), 1, None)?;
        let median_time = median(&gate_times.view())?;

        let min_time = gate_times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_time = gate_times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Compute confidence interval (assuming normal distribution)
        let n = gate_times.len() as f64;
        let z_critical = 1.96; // 95% confidence
        let margin = z_critical * std_time / n.sqrt();

        Ok(BenchmarkResult {
            name: "Gate Execution Time".to_string(),
            mean_value: mean_time,
            std_dev: std_time,
            median_value: median_time,
            min_value: min_time,
            max_value: max_time,
            confidence_interval: (mean_time - margin, mean_time + margin),
            num_samples: gate_times.len(),
            p_value: None,
        })
    }

    /// Analyze gate fidelity using SciRS2 metrics
    fn analyze_gate_fidelity(
        &self,
        fidelity_data: &GateFidelityData,
    ) -> DeviceResult<GateFidelityAnalysis> {
        // Compute average fidelities
        let avg_single = mean(&fidelity_data.single_qubit_fidelities.view())?;
        let avg_two = mean(&fidelity_data.two_qubit_fidelities.view())?;

        // Compute error rates (1 - fidelity)
        let error_rates = fidelity_data.single_qubit_fidelities.mapv(|f| 1.0 - f);

        Ok(GateFidelityAnalysis {
            single_qubit_fidelity: fidelity_data.single_qubit_fidelities.clone(),
            two_qubit_fidelity: fidelity_data.two_qubit_fidelities.clone(),
            avg_single_qubit: avg_single,
            avg_two_qubit: avg_two,
            error_rates,
        })
    }

    /// Analyze coherence times
    fn analyze_coherence(&self, coherence_data: &CoherenceData) -> DeviceResult<CoherenceAnalysis> {
        let avg_t1 = mean(&coherence_data.t1_times.view())?;
        let avg_t2 = mean(&coherence_data.t2_times.view())?;

        // Compute T1/T2 ratio for each qubit
        let t1_t2_ratio = Array1::from_shape_fn(coherence_data.t1_times.len(), |i| {
            coherence_data.t1_times[i] / coherence_data.t2_times[i].max(0.001)
        });

        Ok(CoherenceAnalysis {
            t1_times: coherence_data.t1_times.clone(),
            t2_times: coherence_data.t2_times.clone(),
            avg_t1,
            avg_t2,
            t1_t2_ratio,
        })
    }

    /// Analyze readout fidelity
    fn analyze_readout_fidelity(
        &self,
        readout_data: &ReadoutData,
    ) -> DeviceResult<ReadoutFidelityAnalysis> {
        let avg_fidelity = mean(&readout_data.readout_fidelities.view())?;

        // Compute SPAM error (average of assignment errors)
        let spam_error = if readout_data.assignment_errors.nrows() > 0
            && readout_data.assignment_errors.ncols() > 0
        {
            let mut sum = 0.0;
            let mut count = 0;
            for i in 0..readout_data.assignment_errors.nrows() {
                for j in 0..readout_data.assignment_errors.ncols() {
                    if i != j {
                        // Off-diagonal elements are errors
                        sum += readout_data.assignment_errors[[i, j]];
                        count += 1;
                    }
                }
            }
            if count > 0 {
                sum / count as f64
            } else {
                0.0
            }
        } else {
            0.0
        };

        Ok(ReadoutFidelityAnalysis {
            readout_fidelity: readout_data.readout_fidelities.clone(),
            avg_fidelity,
            assignment_errors: readout_data.assignment_errors.clone(),
            spam_error,
        })
    }

    /// Compute overall device score (0-100)
    fn compute_overall_score(
        &self,
        gate_fidelity: &Option<GateFidelityAnalysis>,
        coherence: &Option<CoherenceAnalysis>,
        readout: &Option<ReadoutFidelityAnalysis>,
    ) -> f64 {
        let mut score = 0.0;
        let mut weight_sum = 0.0;

        // Gate fidelity contributes 40% to score
        if let Some(gf) = gate_fidelity {
            let gate_score = (gf.avg_single_qubit + gf.avg_two_qubit) / 2.0 * 100.0;
            score += gate_score * 0.4;
            weight_sum += 0.4;
        }

        // Coherence times contribute 30% to score (normalized to 100 μs)
        if let Some(coh) = coherence {
            let coherence_score =
                ((coh.avg_t1 / 100.0).min(1.0) + (coh.avg_t2 / 100.0).min(1.0)) / 2.0 * 100.0;
            score += coherence_score * 0.3;
            weight_sum += 0.3;
        }

        // Readout fidelity contributes 30% to score
        if let Some(ro) = readout {
            let readout_score = ro.avg_fidelity * 100.0;
            score += readout_score * 0.3;
            weight_sum += 0.3;
        }

        if weight_sum > 0.0 {
            score / weight_sum
        } else {
            0.0
        }
    }

    /// Generate synthetic benchmark data for testing
    pub fn generate_synthetic_data(&mut self, num_qubits: usize) -> BenchmarkMeasurementData {
        // Generate gate execution times (50-200 nanoseconds with noise)
        let gate_times = Array1::from_shape_fn(self.config.num_iterations, |_| {
            let base_time = 100.0;
            let noise = self.rng.gen::<f64>() * 50.0;
            base_time + noise
        });

        // Generate single-qubit gate fidelities (99.5% - 99.99%)
        let single_qubit_fidelities =
            Array1::from_shape_fn(num_qubits, |_| 0.995 + self.rng.gen::<f64>() * 0.0049);

        // Generate two-qubit gate fidelities (98% - 99.5%)
        let two_qubit_fidelities = Array1::from_shape_fn(num_qubits.saturating_sub(1), |_| {
            0.980 + self.rng.gen::<f64>() * 0.015
        });

        // Generate T1 times (30-100 microseconds)
        let t1_times = Array1::from_shape_fn(num_qubits, |_| 30.0 + self.rng.gen::<f64>() * 70.0);

        // Generate T2 times (20-80 microseconds, always <= T1)
        let t2_times = Array1::from_shape_fn(num_qubits, |i| {
            let max_t2 = t1_times[i] * 0.8;
            20.0 + self.rng.gen::<f64>() * (max_t2 - 20.0).max(0.0)
        });

        // Generate readout fidelities (95% - 99%)
        let readout_fidelities =
            Array1::from_shape_fn(num_qubits, |_| 0.95 + self.rng.gen::<f64>() * 0.04);

        // Generate assignment error matrix (simplified 2x2 for binary readout)
        let mut assignment_errors = Array2::zeros((2, 2));
        assignment_errors[[0, 0]] = 0.98; // P(measure 0 | state 0)
        assignment_errors[[0, 1]] = 0.02; // P(measure 1 | state 0)
        assignment_errors[[1, 0]] = 0.03; // P(measure 0 | state 1)
        assignment_errors[[1, 1]] = 0.97; // P(measure 1 | state 1)

        BenchmarkMeasurementData {
            gate_times,
            fidelity_data: GateFidelityData {
                single_qubit_fidelities,
                two_qubit_fidelities,
            },
            coherence_data: CoherenceData { t1_times, t2_times },
            readout_data: ReadoutData {
                readout_fidelities,
                assignment_errors,
            },
        }
    }
}

/// Input data for hardware benchmarking
#[derive(Debug, Clone)]
pub struct BenchmarkMeasurementData {
    /// Gate execution times (nanoseconds)
    pub gate_times: Array1<f64>,
    /// Gate fidelity measurements
    pub fidelity_data: GateFidelityData,
    /// Coherence time measurements
    pub coherence_data: CoherenceData,
    /// Readout fidelity measurements
    pub readout_data: ReadoutData,
}

/// Gate fidelity measurement data
#[derive(Debug, Clone)]
pub struct GateFidelityData {
    /// Single-qubit gate fidelities
    pub single_qubit_fidelities: Array1<f64>,
    /// Two-qubit gate fidelities
    pub two_qubit_fidelities: Array1<f64>,
}

/// Coherence time measurement data
#[derive(Debug, Clone)]
pub struct CoherenceData {
    /// T1 relaxation times (microseconds)
    pub t1_times: Array1<f64>,
    /// T2 dephasing times (microseconds)
    pub t2_times: Array1<f64>,
}

/// Readout fidelity measurement data
#[derive(Debug, Clone)]
pub struct ReadoutData {
    /// Readout fidelity per qubit
    pub readout_fidelities: Array1<f64>,
    /// Assignment error matrix
    pub assignment_errors: Array2<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmarker_creation() {
        let config = HardwareBenchmarkConfig::default();
        let benchmarker = HardwareBenchmarker::new(config);
        assert_eq!(benchmarker.config.num_iterations, 1000);
    }

    #[test]
    fn test_synthetic_data_generation() {
        let config = HardwareBenchmarkConfig::default();
        let mut benchmarker = HardwareBenchmarker::new(config);

        let data = benchmarker.generate_synthetic_data(3);

        assert_eq!(data.gate_times.len(), 1000);
        assert_eq!(data.fidelity_data.single_qubit_fidelities.len(), 3);
        assert_eq!(data.coherence_data.t1_times.len(), 3);
        assert_eq!(data.readout_data.readout_fidelities.len(), 3);
    }

    #[test]
    fn test_full_benchmark_run() {
        let config = HardwareBenchmarkConfig::default();
        let mut benchmarker = HardwareBenchmarker::new(config);

        let data = benchmarker.generate_synthetic_data(2);
        let report = benchmarker.run_benchmarks("TestDevice", &data);

        assert!(report.is_ok());
        let report = report.expect("Benchmark failed");

        assert_eq!(report.device_name, "TestDevice");
        assert!(!report.benchmarks.is_empty());
        assert!(report.overall_score > 0.0 && report.overall_score <= 100.0);
    }

    #[test]
    fn test_gate_timing_benchmark() {
        let config = HardwareBenchmarkConfig::default();
        let benchmarker = HardwareBenchmarker::new(config);

        let gate_times = Array1::from_vec(vec![100.0, 110.0, 105.0, 95.0, 100.0]);
        let result = benchmarker.benchmark_gate_timing(&gate_times);

        assert!(result.is_ok());
        let result = result.expect("Benchmark failed");

        assert_eq!(result.name, "Gate Execution Time");
        assert!((result.mean_value - 102.0).abs() < 1.0);
        assert!(result.std_dev > 0.0);
        assert_eq!(result.num_samples, 5);
    }

    #[test]
    fn test_gate_fidelity_analysis() {
        let config = HardwareBenchmarkConfig::default();
        let benchmarker = HardwareBenchmarker::new(config);

        let fidelity_data = GateFidelityData {
            single_qubit_fidelities: Array1::from_vec(vec![0.999, 0.998, 0.997]),
            two_qubit_fidelities: Array1::from_vec(vec![0.99, 0.985]),
        };

        let analysis = benchmarker.analyze_gate_fidelity(&fidelity_data);

        assert!(analysis.is_ok());
        let analysis = analysis.expect("Analysis failed");

        assert!((analysis.avg_single_qubit - 0.998).abs() < 0.001);
        assert!((analysis.avg_two_qubit - 0.9875).abs() < 0.001);
        assert_eq!(analysis.error_rates.len(), 3);
    }

    #[test]
    fn test_coherence_analysis() {
        let config = HardwareBenchmarkConfig::default();
        let benchmarker = HardwareBenchmarker::new(config);

        let coherence_data = CoherenceData {
            t1_times: Array1::from_vec(vec![50.0, 60.0, 55.0]),
            t2_times: Array1::from_vec(vec![40.0, 48.0, 44.0]),
        };

        let analysis = benchmarker.analyze_coherence(&coherence_data);

        assert!(analysis.is_ok());
        let analysis = analysis.expect("Analysis failed");

        assert!((analysis.avg_t1 - 55.0).abs() < 0.1);
        assert!((analysis.avg_t2 - 44.0).abs() < 0.1);
        assert_eq!(analysis.t1_t2_ratio.len(), 3);
    }

    #[test]
    fn test_readout_fidelity_analysis() {
        let config = HardwareBenchmarkConfig::default();
        let benchmarker = HardwareBenchmarker::new(config);

        let mut assignment_errors = Array2::zeros((2, 2));
        assignment_errors[[0, 0]] = 0.97;
        assignment_errors[[0, 1]] = 0.03;
        assignment_errors[[1, 0]] = 0.02;
        assignment_errors[[1, 1]] = 0.98;

        let readout_data = ReadoutData {
            readout_fidelities: Array1::from_vec(vec![0.97, 0.98, 0.96]),
            assignment_errors,
        };

        let analysis = benchmarker.analyze_readout_fidelity(&readout_data);

        assert!(analysis.is_ok());
        let analysis = analysis.expect("Analysis failed");

        assert!((analysis.avg_fidelity - 0.97).abs() < 0.01);
        assert!(analysis.spam_error > 0.0);
    }

    #[test]
    fn test_overall_score_computation() {
        let config = HardwareBenchmarkConfig::default();
        let benchmarker = HardwareBenchmarker::new(config);

        let gate_analysis = GateFidelityAnalysis {
            single_qubit_fidelity: Array1::from_vec(vec![0.999]),
            two_qubit_fidelity: Array1::from_vec(vec![0.99]),
            avg_single_qubit: 0.999,
            avg_two_qubit: 0.99,
            error_rates: Array1::from_vec(vec![0.001]),
        };

        let coherence_analysis = CoherenceAnalysis {
            t1_times: Array1::from_vec(vec![50.0]),
            t2_times: Array1::from_vec(vec![40.0]),
            avg_t1: 50.0,
            avg_t2: 40.0,
            t1_t2_ratio: Array1::from_vec(vec![1.25]),
        };

        let readout_analysis = ReadoutFidelityAnalysis {
            readout_fidelity: Array1::from_vec(vec![0.97]),
            avg_fidelity: 0.97,
            assignment_errors: Array2::zeros((2, 2)),
            spam_error: 0.025,
        };

        let score = benchmarker.compute_overall_score(
            &Some(gate_analysis),
            &Some(coherence_analysis),
            &Some(readout_analysis),
        );

        assert!(score > 50.0 && score < 100.0);
    }

    #[test]
    fn test_empty_gate_times_error() {
        let config = HardwareBenchmarkConfig::default();
        let benchmarker = HardwareBenchmarker::new(config);

        let empty_times = Array1::from_vec(vec![]);
        let result = benchmarker.benchmark_gate_timing(&empty_times);

        assert!(result.is_err());
    }
}
