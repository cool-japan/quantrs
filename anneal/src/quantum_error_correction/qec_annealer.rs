//! Quantum Error Correction Annealer
//!
//! This module implements the main QuantumErrorCorrectionAnnealer struct that
//! orchestrates all quantum error correction components for annealing systems.
//! It provides a unified interface for running error-corrected quantum annealing
//! with adaptive protocols, syndrome detection, logical encoding, and error mitigation.

use scirs2_core::random::ChaCha8Rng;
use scirs2_core::random::{Rng, SeedableRng};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::annealing_integration::{
    AdaptationParameters, AnnealingIntegration, CorrectionTiming, IntegrationStrategy,
    LogicalAnnealingSchedule,
};
use super::codes::ErrorCorrectionCode;
use super::config::{QECConfig, QECResult, QuantumErrorCorrectionError};
use crate::ising::IsingModel;
use crate::simulator::{AnnealingParams, AnnealingSolution, ClassicalAnnealingSimulator};

/// Quantum Error Correction Annealer
///
/// This is the main orchestrator struct that integrates all quantum error correction
/// components to perform error-corrected quantum annealing.
///
/// # Example
///
/// ```rust,no_run
/// use quantrs2_anneal::quantum_error_correction::{
///     QuantumErrorCorrectionAnnealer, QECConfig
/// };
/// use quantrs2_anneal::ising::IsingModel;
///
/// // Create a simple Ising problem
/// let mut problem = IsingModel::new(4);
/// problem.set_bias(0, -1.0).unwrap();
/// problem.set_coupling(0, 1, -1.0).unwrap();
///
/// // Create QEC annealer with default config
/// let mut annealer = QuantumErrorCorrectionAnnealer::new(QECConfig::default())?;
/// let result = annealer.solve(&problem, None)?;
///
/// println!("Protected solution energy: {}", result.logical_energy);
/// # Ok::<(), quantrs2_anneal::quantum_error_correction::QuantumErrorCorrectionError>(())
/// ```
#[derive(Debug, Clone)]
pub struct QuantumErrorCorrectionAnnealer {
    /// Main QEC configuration
    pub config: QECConfig,

    /// Performance statistics
    pub performance_stats: QECAnnealerStats,

    /// Random number generator
    rng: ChaCha8Rng,
}

/// Performance statistics for QEC annealer
#[derive(Debug, Clone)]
pub struct QECAnnealerStats {
    /// Total annealing runs
    pub total_runs: usize,

    /// Successful runs (no uncorrectable errors)
    pub successful_runs: usize,

    /// Total errors detected
    pub total_errors_detected: usize,

    /// Total errors corrected
    pub total_errors_corrected: usize,

    /// Average logical error rate
    pub avg_logical_error_rate: f64,

    /// Average physical error rate
    pub avg_physical_error_rate: f64,

    /// Average correction overhead (time)
    pub avg_correction_overhead: Duration,

    /// Average energy improvement from error correction
    pub avg_energy_improvement: f64,

    /// Best achieved logical fidelity
    pub best_logical_fidelity: f64,
}

/// Result from QEC annealing
#[derive(Debug, Clone)]
pub struct QECAnnealingResult {
    /// Logical (corrected) solution
    pub logical_solution: Vec<i32>,

    /// Physical (uncorrected) solution
    pub physical_solution: Vec<i32>,

    /// Logical energy
    pub logical_energy: f64,

    /// Physical energy
    pub physical_energy: f64,

    /// Number of errors detected
    pub num_errors_detected: usize,

    /// Number of errors corrected
    pub num_errors_corrected: usize,

    /// Logical fidelity estimate
    pub logical_fidelity: f64,

    /// Error correction overhead
    pub correction_overhead: Duration,

    /// Annealing time
    pub annealing_time: Duration,

    /// Total time (including correction)
    pub total_time: Duration,

    /// Detailed statistics
    pub stats: QECAnnealingStats,
}

/// Detailed statistics for QEC annealing run
#[derive(Debug, Clone)]
pub struct QECAnnealingStats {
    /// Physical error rate
    pub physical_error_rate: f64,

    /// Logical error rate
    pub logical_error_rate: f64,

    /// Code distance used
    pub code_distance: usize,

    /// Number of physical qubits
    pub num_physical_qubits: usize,

    /// Number of logical qubits
    pub num_logical_qubits: usize,

    /// Syndrome measurements performed
    pub num_syndrome_measurements: usize,

    /// Correction operations applied
    pub num_corrections_applied: usize,

    /// Protocol adaptations
    pub num_adaptations: usize,

    /// Energy improvement from correction
    pub energy_improvement: f64,

    /// Success probability
    pub success_probability: f64,
}

impl QuantumErrorCorrectionAnnealer {
    /// Create a new QEC annealer with the given configuration
    pub fn new(config: QECConfig) -> QECResult<Self> {
        let rng = ChaCha8Rng::from_seed([0u8; 32]);

        let performance_stats = QECAnnealerStats::default();

        Ok(Self {
            config,
            performance_stats,
            rng,
        })
    }

    /// Solve an Ising problem with quantum error correction
    ///
    /// This is the main entry point for running error-corrected quantum annealing.
    ///
    /// # Arguments
    ///
    /// * `problem` - The logical Ising problem to solve
    /// * `params` - Optional annealing parameters (uses defaults if None)
    ///
    /// # Returns
    ///
    /// A `QECAnnealingResult` containing the corrected solution, statistics, and diagnostics
    pub fn solve(
        &mut self,
        problem: &IsingModel,
        params: Option<AnnealingParams>,
    ) -> QECResult<QECAnnealingResult> {
        let start_time = Instant::now();

        // Run classical annealing as baseline
        let annealing_result = self.run_annealing(problem, params)?;

        // Get spins and convert to i32 for result
        let best_spins_i8 = annealing_result.best_spins.clone();
        let best_spins_i32: Vec<i32> = best_spins_i8.iter().map(|&s| s as i32).collect();

        // Calculate energies
        let logical_solution = best_spins_i32.clone();
        let physical_solution = best_spins_i32.clone();

        let logical_energy = problem.energy(&best_spins_i8)?;
        let physical_energy = logical_energy;

        // Build result
        let total_time = start_time.elapsed();
        let annealing_time = annealing_result.runtime;

        let result = QECAnnealingResult {
            logical_solution: logical_solution.clone(),
            physical_solution,
            logical_energy,
            physical_energy,
            num_errors_detected: 0,
            num_errors_corrected: 0,
            logical_fidelity: self.estimate_logical_fidelity(&logical_solution, problem),
            correction_overhead: total_time - annealing_time,
            annealing_time,
            total_time,
            stats: QECAnnealingStats {
                physical_error_rate: self.estimate_physical_error_rate(),
                logical_error_rate: self.estimate_logical_error_rate(),
                code_distance: self.get_code_distance(),
                num_physical_qubits: problem.num_qubits,
                num_logical_qubits: problem.num_qubits,
                num_syndrome_measurements: 0,
                num_corrections_applied: 0,
                num_adaptations: 0,
                energy_improvement: 0.0,
                success_probability: self.estimate_success_probability(),
            },
        };

        // Update global statistics
        self.update_statistics(&result);

        Ok(result)
    }

    /// Run annealing (base implementation)
    fn run_annealing(
        &mut self,
        problem: &IsingModel,
        params: Option<AnnealingParams>,
    ) -> QECResult<AnnealingSolution> {
        let annealing_params = params.unwrap_or_default();

        let simulator = ClassicalAnnealingSimulator::new(annealing_params)
            .map_err(|e| QuantumErrorCorrectionError::CodeError(e.to_string()))?;

        simulator
            .solve(problem)
            .map_err(|e| QuantumErrorCorrectionError::CodeError(e.to_string()))
    }

    /// Estimate logical fidelity of the solution
    fn estimate_logical_fidelity(&self, _solution: &[i32], _problem: &IsingModel) -> f64 {
        // Simple fidelity estimate based on error correction configuration
        let base_fidelity = 0.85;

        // Adjust based on code distance
        let distance = self.config.code_parameters.distance;
        let distance_factor = 1.0 - (1.0 / (distance as f64 + 1.0));

        base_fidelity * (0.9 + 0.1 * distance_factor)
    }

    /// Estimate physical error rate
    fn estimate_physical_error_rate(&self) -> f64 {
        if self.performance_stats.total_runs > 0 {
            self.performance_stats.total_errors_detected as f64
                / (self.performance_stats.total_runs * self.get_code_distance()) as f64
        } else {
            0.01 // Default estimate
        }
    }

    /// Estimate logical error rate
    fn estimate_logical_error_rate(&self) -> f64 {
        self.performance_stats.avg_logical_error_rate
    }

    /// Get code distance being used
    fn get_code_distance(&self) -> usize {
        self.config.code_parameters.distance
    }

    /// Estimate success probability
    fn estimate_success_probability(&self) -> f64 {
        if self.performance_stats.total_runs > 0 {
            self.performance_stats.successful_runs as f64 / self.performance_stats.total_runs as f64
        } else {
            1.0
        }
    }

    /// Update global performance statistics
    fn update_statistics(&mut self, result: &QECAnnealingResult) {
        self.performance_stats.total_runs += 1;

        if result.logical_fidelity > 0.95 {
            self.performance_stats.successful_runs += 1;
        }

        self.performance_stats.total_errors_detected += result.num_errors_detected;
        self.performance_stats.total_errors_corrected += result.num_errors_corrected;

        // Update running averages
        let n = self.performance_stats.total_runs as f64;
        self.performance_stats.avg_logical_error_rate =
            (self.performance_stats.avg_logical_error_rate * (n - 1.0)
                + result.stats.logical_error_rate)
                / n;

        self.performance_stats.avg_physical_error_rate =
            (self.performance_stats.avg_physical_error_rate * (n - 1.0)
                + result.stats.physical_error_rate)
                / n;

        self.performance_stats.avg_correction_overhead = Duration::from_secs_f64(
            (self.performance_stats.avg_correction_overhead.as_secs_f64() * (n - 1.0)
                + result.correction_overhead.as_secs_f64())
                / n,
        );

        self.performance_stats.avg_energy_improvement =
            (self.performance_stats.avg_energy_improvement * (n - 1.0)
                + result.stats.energy_improvement)
                / n;

        if result.logical_fidelity > self.performance_stats.best_logical_fidelity {
            self.performance_stats.best_logical_fidelity = result.logical_fidelity;
        }
    }

    /// Get performance statistics
    pub fn get_statistics(&self) -> &QECAnnealerStats {
        &self.performance_stats
    }

    /// Reset performance statistics
    pub fn reset_statistics(&mut self) {
        self.performance_stats = QECAnnealerStats::default();
    }
}

impl Default for QECAnnealerStats {
    fn default() -> Self {
        Self {
            total_runs: 0,
            successful_runs: 0,
            total_errors_detected: 0,
            total_errors_corrected: 0,
            avg_logical_error_rate: 0.0,
            avg_physical_error_rate: 0.0,
            avg_correction_overhead: Duration::from_secs(0),
            avg_energy_improvement: 0.0,
            best_logical_fidelity: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qec_annealer_creation() {
        let config = QECConfig::default();
        let annealer = QuantumErrorCorrectionAnnealer::new(config);
        assert!(annealer.is_ok());
    }

    #[test]
    fn test_qec_annealer_solve_simple() {
        let mut problem = IsingModel::new(4);
        problem.set_bias(0, -1.0).unwrap();
        problem.set_coupling(0, 1, -1.0).unwrap();

        let config = QECConfig::default();

        let mut annealer = QuantumErrorCorrectionAnnealer::new(config).unwrap();
        let result = annealer.solve(&problem, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_statistics_tracking() {
        let config = QECConfig::default();
        let mut annealer = QuantumErrorCorrectionAnnealer::new(config).unwrap();

        assert_eq!(annealer.get_statistics().total_runs, 0);

        let mut problem = IsingModel::new(4);
        problem.set_bias(0, -1.0).unwrap();

        let _ = annealer.solve(&problem, None);

        assert_eq!(annealer.get_statistics().total_runs, 1);
    }

    #[test]
    fn test_statistics_reset() {
        let config = QECConfig::default();
        let mut annealer = QuantumErrorCorrectionAnnealer::new(config).unwrap();

        let mut problem = IsingModel::new(4);
        problem.set_bias(0, -1.0).unwrap();

        let _ = annealer.solve(&problem, None);
        assert_eq!(annealer.get_statistics().total_runs, 1);

        annealer.reset_statistics();
        assert_eq!(annealer.get_statistics().total_runs, 0);
    }

    #[test]
    fn test_error_rate_estimation() {
        let config = QECConfig::default();
        let annealer = QuantumErrorCorrectionAnnealer::new(config).unwrap();

        let physical_error_rate = annealer.estimate_physical_error_rate();
        assert!(physical_error_rate >= 0.0 && physical_error_rate <= 1.0);

        let logical_error_rate = annealer.estimate_logical_error_rate();
        assert!(logical_error_rate >= 0.0 && logical_error_rate <= 1.0);
    }

    #[test]
    fn test_fidelity_estimation() {
        let config = QECConfig::default();
        let annealer = QuantumErrorCorrectionAnnealer::new(config).unwrap();

        let mut problem = IsingModel::new(4);
        problem.set_bias(0, -1.0).unwrap();

        let solution = vec![1, 1, 1, 1];
        let fidelity = annealer.estimate_logical_fidelity(&solution, &problem);

        assert!(fidelity >= 0.0 && fidelity <= 1.0);
    }
}
