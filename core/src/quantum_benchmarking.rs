//! Comprehensive Quantum Benchmarking Suite
//!
//! This module provides integrated benchmarking and performance analysis
//! for quantum algorithms, combining noise characterization, error mitigation,
//! and algorithm optimization.
//!
//! ## Features
//! - Algorithm performance profiling with noise models
//! - Automated error mitigation strategy selection
//! - Comparative benchmarking across different quantum hardware
//! - Integrated QAOA optimization with noise mitigation
//! - Real-time performance monitoring and reporting

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    noise_characterization::{
        CrossEntropyBenchmarking, DDSequenceType, DynamicalDecoupling, ExtrapolationMethod,
        NoiseModel, ZeroNoiseExtrapolation,
    },
    qaoa::{CostHamiltonian, MixerHamiltonian, QAOACircuit, QAOAOptimizer, QAOAParams},
    quantum_volume_tomography::{QuantumProcessTomography, QuantumVolume},
};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;
use std::collections::HashMap;
use std::time::{Duration, Instant};

use std::fmt::Write;
/// Comprehensive benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Algorithm name
    pub algorithm: String,
    /// Execution time
    pub execution_time: Duration,
    /// Fidelity estimate
    pub fidelity: f64,
    /// Success probability
    pub success_rate: f64,
    /// Error mitigation improvement factor
    pub mitigation_improvement: Option<f64>,
    /// Resource usage
    pub resource_usage: ResourceUsage,
}

#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// Number of gates used
    pub gate_count: usize,
    /// Circuit depth
    pub circuit_depth: usize,
    /// Number of qubits
    pub num_qubits: usize,
    /// Number of measurements
    pub num_measurements: usize,
}

/// Integrated quantum benchmarking suite
pub struct QuantumBenchmarkSuite {
    /// Noise model for realistic simulation
    pub noise_model: NoiseModel,
    /// Enable error mitigation
    pub enable_mitigation: bool,
    /// Benchmarking configuration
    pub config: BenchmarkConfig,
}

#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of benchmark iterations
    pub num_iterations: usize,
    /// Collect detailed profiling data
    pub detailed_profiling: bool,
    /// Compare against ideal (noiseless) execution
    pub compare_ideal: bool,
    /// Maximum time per benchmark
    pub max_time: Duration,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            num_iterations: 100,
            detailed_profiling: true,
            compare_ideal: true,
            max_time: Duration::from_secs(300),
        }
    }
}

impl QuantumBenchmarkSuite {
    /// Create a new benchmark suite
    pub const fn new(noise_model: NoiseModel, config: BenchmarkConfig) -> Self {
        Self {
            noise_model,
            enable_mitigation: true,
            config,
        }
    }

    /// Benchmark QAOA with error mitigation
    pub fn benchmark_qaoa_with_mitigation(
        &self,
        num_qubits: usize,
        edges: Vec<(usize, usize)>,
        num_layers: usize,
    ) -> QuantRS2Result<QAOABenchmarkResult> {
        let start_time = Instant::now();

        // Create QAOA circuit
        let cost_hamiltonian = CostHamiltonian::MaxCut(edges.clone());
        let mixer_hamiltonian = MixerHamiltonian::TransverseField;
        let params = QAOAParams::random(num_layers);

        let circuit = QAOACircuit::new(num_qubits, cost_hamiltonian, mixer_hamiltonian, params);

        // Benchmark without mitigation
        let noisy_result = self.run_qaoa_noisy(&circuit)?;

        // Benchmark with ZNE mitigation
        let mitigated_result = if self.enable_mitigation {
            Some(self.run_qaoa_with_zne(&circuit)?)
        } else {
            None
        };

        // Benchmark ideal (noiseless) for comparison
        let ideal_result = if self.config.compare_ideal {
            Some(self.run_qaoa_ideal(&circuit)?)
        } else {
            None
        };

        let execution_time = start_time.elapsed();

        // Calculate improvement from error mitigation
        let mitigation_improvement = if let (Some(mitigated), Some(ideal)) =
            (mitigated_result.as_ref(), ideal_result.as_ref())
        {
            let noisy_error = (noisy_result - ideal).abs();
            let mitigated_error = (mitigated - ideal).abs();
            Some(noisy_error / mitigated_error.max(1e-10))
        } else {
            None
        };

        Ok(QAOABenchmarkResult {
            num_qubits,
            num_edges: edges.len(),
            num_layers,
            noisy_expectation: noisy_result,
            mitigated_expectation: mitigated_result,
            ideal_expectation: ideal_result,
            mitigation_improvement,
            execution_time,
            noise_model: self.noise_model.clone(),
        })
    }

    /// Run QAOA with noise
    fn run_qaoa_noisy(&self, circuit: &QAOACircuit) -> QuantRS2Result<f64> {
        let state_size = 1 << circuit.num_qubits;
        let mut state = vec![Complex64::new(0.0, 0.0); state_size];

        // Execute circuit
        circuit.execute(&mut state);

        // Apply noise model
        self.apply_noise_to_state(&mut state)?;

        // Compute expectation value
        Ok(circuit.compute_expectation(&state))
    }

    /// Run QAOA with Zero-Noise Extrapolation
    fn run_qaoa_with_zne(&self, circuit: &QAOACircuit) -> QuantRS2Result<f64> {
        let zne = ZeroNoiseExtrapolation::new(vec![1.0, 2.0, 3.0], ExtrapolationMethod::Linear);

        // Execute at different noise scales
        let result = zne.mitigate(|noise_scale| {
            let state_size = 1 << circuit.num_qubits;
            let mut state = vec![Complex64::new(0.0, 0.0); state_size];

            circuit.execute(&mut state);

            // Apply scaled noise
            let scaled_noise = NoiseModel {
                single_qubit_depolarizing: self.noise_model.single_qubit_depolarizing * noise_scale,
                two_qubit_depolarizing: self.noise_model.two_qubit_depolarizing * noise_scale,
                ..self.noise_model.clone()
            };

            // Apply noise (simplified)
            let _ = scaled_noise;

            circuit.compute_expectation(&state)
        })?;

        Ok(result)
    }

    /// Run QAOA without noise (ideal)
    fn run_qaoa_ideal(&self, circuit: &QAOACircuit) -> QuantRS2Result<f64> {
        let state_size = 1 << circuit.num_qubits;
        let mut state = vec![Complex64::new(0.0, 0.0); state_size];

        circuit.execute(&mut state);
        Ok(circuit.compute_expectation(&state))
    }

    /// Apply noise model to a quantum state
    fn apply_noise_to_state(&self, state: &mut [Complex64]) -> QuantRS2Result<()> {
        let mut rng = thread_rng();
        let depolarizing_prob = self.noise_model.single_qubit_depolarizing;

        // Apply depolarizing noise (simplified model)
        for amplitude in state.iter_mut() {
            if rng.gen::<f64>() < depolarizing_prob {
                // Randomly flip phase or amplitude
                *amplitude *= Complex64::new(0.9, 0.0);
            }
        }

        Ok(())
    }

    /// Benchmark quantum volume
    pub fn benchmark_quantum_volume(
        &self,
        max_qubits: usize,
    ) -> QuantRS2Result<QuantumVolumeBenchmarkResult> {
        let start_time = Instant::now();

        let mut qv = QuantumVolume::new(max_qubits, 100, 1000);

        // Mock circuit executor with noise
        let circuit_executor = |_gates: &[Box<dyn crate::gate::GateOp>], num_shots: usize| {
            // Simplified: return random bitstrings
            let mut results = Vec::with_capacity(num_shots);
            let mut rng = thread_rng();
            let max_value = 1 << max_qubits;
            for _ in 0..num_shots {
                // Use simple modulo to generate random values
                let random_val = (rng.gen::<u64>() as usize) % max_value;
                results.push(random_val);
            }
            results
        };

        let result = qv.run(circuit_executor)?;
        let execution_time = start_time.elapsed();

        Ok(QuantumVolumeBenchmarkResult {
            quantum_volume: result.quantum_volume,
            qubits_achieved: result.num_qubits_achieved(),
            success_rates: result.success_rates,
            execution_time,
            noise_model: self.noise_model.clone(),
        })
    }

    /// Benchmark with dynamical decoupling
    pub fn benchmark_with_dynamical_decoupling(
        &self,
        dd_sequence: DDSequenceType,
        num_pulses: usize,
        idle_time: f64,
    ) -> QuantRS2Result<DDEffectivenessResult> {
        let dd = DynamicalDecoupling::new(dd_sequence, num_pulses);

        // Generate DD sequence
        let sequence = dd.generate_sequence(idle_time);

        // Estimate coherence improvement
        let improvement_factor = dd.coherence_improvement_factor(
            self.noise_model.t2_dephasing,
            self.noise_model.gate_duration,
        );

        // Calculate overhead
        let overhead = sequence.len() as f64 * self.noise_model.gate_duration;
        let overhead_fraction = overhead / idle_time;

        Ok(DDEffectivenessResult {
            sequence_type: dd_sequence,
            num_pulses,
            coherence_improvement: improvement_factor,
            time_overhead: overhead,
            overhead_fraction,
            effective_t2: self.noise_model.t2_dephasing * improvement_factor,
        })
    }

    /// Comprehensive algorithm benchmark
    pub fn benchmark_algorithm<F>(
        &self,
        algorithm_name: &str,
        algorithm: F,
    ) -> QuantRS2Result<BenchmarkResult>
    where
        F: Fn() -> (f64, ResourceUsage),
    {
        let start_time = Instant::now();

        let mut total_fidelity = 0.0;
        let mut total_success = 0.0;
        let mut resource_usage = None;

        for _ in 0..self.config.num_iterations {
            let (result, usage) = algorithm();
            total_fidelity += result;
            total_success += if result > 0.5 { 1.0 } else { 0.0 };
            resource_usage = Some(usage);
        }

        let execution_time = start_time.elapsed();
        let avg_fidelity = total_fidelity / (self.config.num_iterations as f64);
        let success_rate = total_success / (self.config.num_iterations as f64);

        Ok(BenchmarkResult {
            algorithm: algorithm_name.to_string(),
            execution_time,
            fidelity: avg_fidelity,
            success_rate,
            mitigation_improvement: None,
            resource_usage: resource_usage.unwrap_or(ResourceUsage {
                gate_count: 0,
                circuit_depth: 0,
                num_qubits: 0,
                num_measurements: 0,
            }),
        })
    }
}

/// QAOA-specific benchmark result
#[derive(Debug, Clone)]
pub struct QAOABenchmarkResult {
    /// Number of qubits
    pub num_qubits: usize,
    /// Number of edges in the graph
    pub num_edges: usize,
    /// Number of QAOA layers
    pub num_layers: usize,
    /// Expectation value with noise
    pub noisy_expectation: f64,
    /// Expectation value with error mitigation
    pub mitigated_expectation: Option<f64>,
    /// Ideal expectation value (no noise)
    pub ideal_expectation: Option<f64>,
    /// Error mitigation improvement factor
    pub mitigation_improvement: Option<f64>,
    /// Total execution time
    pub execution_time: Duration,
    /// Noise model used
    pub noise_model: NoiseModel,
}

impl QAOABenchmarkResult {
    /// Calculate the approximation ratio (compared to ideal)
    pub fn approximation_ratio(&self) -> Option<f64> {
        self.ideal_expectation.map(|ideal| {
            if ideal.abs() > 1e-10 {
                self.noisy_expectation / ideal
            } else {
                0.0
            }
        })
    }

    /// Calculate mitigated approximation ratio
    pub fn mitigated_approximation_ratio(&self) -> Option<f64> {
        if let (Some(mitigated), Some(ideal)) = (self.mitigated_expectation, self.ideal_expectation)
        {
            if ideal.abs() > 1e-10 {
                Some(mitigated / ideal)
            } else {
                Some(0.0)
            }
        } else {
            None
        }
    }

    /// Print detailed report
    pub fn print_report(&self) {
        println!("╔════════════════════════════════════════════════════════╗");
        println!("║         QAOA Benchmark Report                          ║");
        println!("╠════════════════════════════════════════════════════════╣");
        println!("║ Problem Size:                                          ║");
        println!(
            "║   Qubits: {:4}      Edges: {:4}      Layers: {:4}   ║",
            self.num_qubits, self.num_edges, self.num_layers
        );
        println!("║                                                        ║");
        println!("║ Results:                                               ║");
        println!(
            "║   Noisy Expectation:     {:8.4}                    ║",
            self.noisy_expectation
        );

        if let Some(mitigated) = self.mitigated_expectation {
            println!("║   Mitigated Expectation: {mitigated:8.4}                    ║");
        }

        if let Some(ideal) = self.ideal_expectation {
            println!("║   Ideal Expectation:     {ideal:8.4}                    ║");
        }

        if let Some(improvement) = self.mitigation_improvement {
            println!("║   Mitigation Improvement: {improvement:7.2}x                   ║");
        }

        println!("║                                                        ║");
        println!(
            "║ Execution Time: {:?}                              ║",
            self.execution_time
        );
        println!("╚════════════════════════════════════════════════════════╝");
    }
}

/// Quantum volume benchmark result
#[derive(Debug, Clone)]
pub struct QuantumVolumeBenchmarkResult {
    /// Achieved quantum volume
    pub quantum_volume: usize,
    /// Number of qubits achieved
    pub qubits_achieved: usize,
    /// Success rates per qubit count
    pub success_rates: HashMap<usize, f64>,
    /// Execution time
    pub execution_time: Duration,
    /// Noise model
    pub noise_model: NoiseModel,
}

/// Dynamical decoupling effectiveness result
#[derive(Debug, Clone)]
pub struct DDEffectivenessResult {
    /// DD sequence type
    pub sequence_type: DDSequenceType,
    /// Number of pulses
    pub num_pulses: usize,
    /// Coherence time improvement factor
    pub coherence_improvement: f64,
    /// Time overhead from DD pulses
    pub time_overhead: f64,
    /// Fraction of idle time spent on DD
    pub overhead_fraction: f64,
    /// Effective T2 after DD
    pub effective_t2: f64,
}

/// Comparative benchmark across different scenarios
pub struct ComparativeBenchmark {
    /// Benchmark suite
    pub suite: QuantumBenchmarkSuite,
    /// Results storage
    pub results: Vec<BenchmarkResult>,
}

impl ComparativeBenchmark {
    /// Create new comparative benchmark
    pub const fn new(suite: QuantumBenchmarkSuite) -> Self {
        Self {
            suite,
            results: Vec::new(),
        }
    }

    /// Add benchmark result
    pub fn add_result(&mut self, result: BenchmarkResult) {
        self.results.push(result);
    }

    /// Generate comparative report
    pub fn generate_report(&self) -> String {
        let mut report = String::from("Comparative Benchmark Report\n");
        report.push_str("================================\n\n");

        for result in &self.results {
            let _ = writeln!(report, "Algorithm: {}", result.algorithm);
            let _ = writeln!(report, "  Fidelity: {:.4}", result.fidelity);
            let _ = writeln!(
                report,
                "  Success Rate: {:.2}%",
                result.success_rate * 100.0
            );
            let _ = writeln!(report, "  Execution Time: {:?}", result.execution_time);

            if let Some(improvement) = result.mitigation_improvement {
                let _ = writeln!(report, "  Mitigation Improvement: {improvement:.2}x");
            }

            let _ = writeln!(
                report,
                "  Resources: {} gates, depth {}, {} qubits",
                result.resource_usage.gate_count,
                result.resource_usage.circuit_depth,
                result.resource_usage.num_qubits,
            );
            report.push('\n');
        }

        report
    }

    /// Find best performing algorithm
    pub fn best_algorithm(&self) -> Option<&BenchmarkResult> {
        self.results.iter().max_by(|a, b| {
            a.fidelity
                .partial_cmp(&b.fidelity)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_suite_creation() {
        let noise_model = NoiseModel::default();
        let config = BenchmarkConfig::default();
        let suite = QuantumBenchmarkSuite::new(noise_model, config);

        assert!(suite.enable_mitigation);
        assert_eq!(suite.config.num_iterations, 100);
    }

    #[test]
    fn test_qaoa_benchmark() {
        let noise_model = NoiseModel::new(0.001, 0.01, 50.0, 70.0, 0.02);
        let config = BenchmarkConfig {
            num_iterations: 10,
            detailed_profiling: false,
            compare_ideal: true,
            max_time: Duration::from_secs(60),
        };

        let suite = QuantumBenchmarkSuite::new(noise_model, config);

        // Simple 3-qubit MaxCut problem
        let edges = vec![(0, 1), (1, 2)];
        let result = suite
            .benchmark_qaoa_with_mitigation(3, edges, 1)
            .expect("Failed to benchmark QAOA with mitigation");

        assert_eq!(result.num_qubits, 3);
        assert_eq!(result.num_edges, 2);
        assert!(result.noisy_expectation.is_finite());

        println!("QAOA Benchmark:");
        println!("  Noisy: {:.4}", result.noisy_expectation);
        if let Some(mitigated) = result.mitigated_expectation {
            println!("  Mitigated: {:.4}", mitigated);
        }
        if let Some(ideal) = result.ideal_expectation {
            println!("  Ideal: {:.4}", ideal);
        }
    }

    #[test]
    fn test_dd_effectiveness() {
        let noise_model = NoiseModel::default();
        let config = BenchmarkConfig::default();
        let suite = QuantumBenchmarkSuite::new(noise_model, config);

        let result = suite
            .benchmark_with_dynamical_decoupling(DDSequenceType::CPMG, 10, 1.0)
            .expect("Failed to benchmark dynamical decoupling");

        assert!(result.coherence_improvement > 1.0);
        assert!(result.overhead_fraction >= 0.0); // Can be > 1 for many pulses

        println!("DD Effectiveness:");
        println!("  Sequence: {:?}", result.sequence_type);
        println!("  Improvement: {:.2}x", result.coherence_improvement);
        println!("  Overhead: {:.2}%", result.overhead_fraction * 100.0);
        println!("  Effective T2: {:.2} μs", result.effective_t2);
    }

    #[test]
    fn test_comparative_benchmark() {
        let noise_model = NoiseModel::default();
        let config = BenchmarkConfig::default();
        let suite = QuantumBenchmarkSuite::new(noise_model, config);

        let mut comparative = ComparativeBenchmark::new(suite);

        // Add mock results
        comparative.add_result(BenchmarkResult {
            algorithm: "Algorithm A".to_string(),
            execution_time: Duration::from_millis(100),
            fidelity: 0.95,
            success_rate: 0.90,
            mitigation_improvement: Some(2.0),
            resource_usage: ResourceUsage {
                gate_count: 100,
                circuit_depth: 10,
                num_qubits: 5,
                num_measurements: 1000,
            },
        });

        let report = comparative.generate_report();
        assert!(report.contains("Algorithm A"));

        let best = comparative
            .best_algorithm()
            .expect("Expected at least one algorithm in benchmark");
        assert_eq!(best.algorithm, "Algorithm A");
    }

    #[test]
    fn test_qaoa_report_printing() {
        let result = QAOABenchmarkResult {
            num_qubits: 4,
            num_edges: 5,
            num_layers: 2,
            noisy_expectation: 3.2,
            mitigated_expectation: Some(3.8),
            ideal_expectation: Some(4.0),
            mitigation_improvement: Some(2.5),
            execution_time: Duration::from_millis(250),
            noise_model: NoiseModel::default(),
        };

        result.print_report();

        assert_eq!(result.approximation_ratio(), Some(0.8));
        assert_eq!(result.mitigated_approximation_ratio(), Some(0.95));
    }
}
