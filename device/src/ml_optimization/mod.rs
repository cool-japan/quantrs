//! ML-Driven Circuit Optimization and Hardware Prediction with SciRS2
//!
//! This module provides comprehensive machine learning-driven circuit optimization
//! and hardware performance prediction using SciRS2's advanced ML capabilities,
//! statistical analysis, and optimization algorithms for intelligent quantum computing.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::f64::consts::PI;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use quantrs2_circuit::prelude::*;
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};

// SciRS2 dependencies are conditionally re-exported through fallback_scirs2

// Fallback implementations when SciRS2 is not available
pub mod fallback_scirs2;
pub use fallback_scirs2::*;

use scirs2_core::ndarray::{s, Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;
use tokio::sync::{broadcast, mpsc};

use crate::{
    adaptive_compilation::AdaptiveCompilationConfig,
    backend_traits::{query_backend_capabilities, BackendCapabilities},
    calibration::{CalibrationManager, DeviceCalibration},
    integrated_device_manager::IntegratedQuantumDeviceManager,
    noise_model::CalibrationNoiseModel,
    topology::HardwareTopology,
    CircuitResult, DeviceError, DeviceResult,
};

// Module declarations
pub mod config;
pub mod ensemble;
pub mod features;
pub mod hardware;
pub mod monitoring;
pub mod online_learning;
pub mod optimization;
pub mod training;
pub mod transfer_learning;
pub mod validation;

#[cfg(not(feature = "scirs2"))]
pub mod fallback;

// Re-exports for public API
pub use config::*;
pub use ensemble::*;
pub use features::*;
pub use hardware::*;
pub use monitoring::*;
pub use online_learning::*;
pub use optimization::*;
pub use training::*;
pub use transfer_learning::*;
pub use validation::*;

#[cfg(not(feature = "scirs2"))]
pub use fallback::*;

// ── MLCircuitOptimizer ────────────────────────────────────────────────────

/// Strategy used by `MLCircuitOptimizer` when choosing between candidate
/// circuit transformations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MLOptimizationStrategy {
    /// Prefer circuits that minimise total gate count.
    MinimizeGateCount,
    /// Prefer circuits with the smallest critical-path depth.
    MinimizeDepth,
    /// Balance gate count and depth equally.
    BalancedOptimization,
}

impl Default for MLOptimizationStrategy {
    fn default() -> Self {
        MLOptimizationStrategy::BalancedOptimization
    }
}

/// Score computed for a candidate circuit, used to rank transformations.
#[derive(Debug, Clone)]
pub struct CircuitScore {
    /// Raw two-qubit gate count (dominant cost on real hardware)
    pub two_qubit_gate_count: usize,
    /// Total gate count
    pub total_gate_count: usize,
    /// Estimated circuit depth (number of layers)
    pub circuit_depth: usize,
    /// Composite score (lower is better)
    pub composite: f64,
}

impl CircuitScore {
    /// Compute a score for `circuit` under the given strategy.
    pub fn compute<const N: usize>(
        circuit: &quantrs2_circuit::prelude::Circuit<N>,
        strategy: &MLOptimizationStrategy,
    ) -> Self {
        let gates = circuit.gates();
        let total = gates.len();
        let two_q = gates.iter().filter(|g| g.qubits().len() == 2).count();

        // Simple depth estimate: max over qubits of the number of gates touching it
        let mut qubit_depth = vec![0usize; N];
        for gate in gates {
            let max_d = gate
                .qubits()
                .iter()
                .map(|q| qubit_depth.get(q.id() as usize).copied().unwrap_or(0))
                .max()
                .unwrap_or(0);
            for q in gate.qubits() {
                let idx = q.id() as usize;
                if idx < N {
                    qubit_depth[idx] = max_d + 1;
                }
            }
        }
        let depth = qubit_depth.iter().copied().max().unwrap_or(0);

        let composite = match strategy {
            MLOptimizationStrategy::MinimizeGateCount => total as f64,
            MLOptimizationStrategy::MinimizeDepth => depth as f64,
            MLOptimizationStrategy::BalancedOptimization => {
                // Weighted combination: two-qubit gates are ~10× more expensive
                (two_q as f64) * 10.0 + (total as f64) + (depth as f64)
            }
        };

        Self {
            two_qubit_gate_count: two_q,
            total_gate_count: total,
            circuit_depth: depth,
            composite,
        }
    }
}

/// ML-driven circuit optimizer.
///
/// Uses a simple learned heuristic — prefer circuits with lower two-qubit gate
/// count weighted by estimated depth — to select among candidate transformed
/// circuits.  The optimisation proceeds in passes; each pass tries commuting
/// independent gates and removing adjacent inverse pairs.
///
/// # Strategy
/// | `MLOptimizationStrategy` | Scoring function |
/// |---|---|
/// | `MinimizeGateCount` | total gate count |
/// | `MinimizeDepth` | circuit depth (critical path) |
/// | `BalancedOptimization` | `10 * two_q + total + depth` |
pub struct MLCircuitOptimizer {
    /// Optimisation strategy
    pub strategy: MLOptimizationStrategy,
    /// Maximum number of optimisation passes
    pub max_passes: usize,
    /// Minimum score improvement to continue iterating (as a fraction)
    pub convergence_threshold: f64,
}

impl MLCircuitOptimizer {
    /// Create a new optimizer with the given strategy and default pass limit.
    pub fn new(strategy: MLOptimizationStrategy) -> Self {
        Self {
            strategy,
            max_passes: 5,
            convergence_threshold: 0.01,
        }
    }

    /// Create a balanced optimizer with a custom pass limit.
    pub fn with_passes(mut self, passes: usize) -> Self {
        self.max_passes = passes;
        self
    }

    /// Optimise `circuit` and return the best circuit found together with its
    /// score.
    ///
    /// The method uses graph-based topology analysis (`SciRS2GraphTranspiler`)
    /// to identify commuting gate pairs, then greedily reorders them to
    /// minimise the composite score under the chosen strategy.
    pub fn optimize<const N: usize>(
        &self,
        circuit: &quantrs2_circuit::prelude::Circuit<N>,
    ) -> QuantRS2Result<quantrs2_circuit::prelude::Circuit<N>> {
        use crate::transpiler_scirs2_graph::{SciRS2GraphTranspiler, SciRS2TranspilerConfig};

        let transpiler = SciRS2GraphTranspiler::new(SciRS2TranspilerConfig {
            enable_commutation: true,
            enable_critical_path_opt: true,
            enable_routing_opt: false,
            max_optimization_passes: self.max_passes,
            hardware_topology: None,
        });

        let mut best = circuit.clone();
        let mut best_score = CircuitScore::compute(&best, &self.strategy);

        for _pass in 0..self.max_passes {
            // Apply graph-based optimization
            let candidate = transpiler.optimize_circuit(&best).map_err(|e| {
                QuantRS2Error::InvalidInput(format!("ML optimizer pass failed: {}", e))
            })?;

            let candidate_score = CircuitScore::compute(&candidate, &self.strategy);

            // Accept if strictly better
            let improvement =
                (best_score.composite - candidate_score.composite) / best_score.composite.max(1.0);

            if candidate_score.composite < best_score.composite {
                best = candidate;
                best_score = candidate_score;
                if improvement < self.convergence_threshold {
                    break; // Converged
                }
            } else {
                break; // No improvement
            }
        }

        Ok(best)
    }

    /// Return the current score for a circuit without modifying it.
    pub fn score<const N: usize>(
        &self,
        circuit: &quantrs2_circuit::prelude::Circuit<N>,
    ) -> CircuitScore {
        CircuitScore::compute(circuit, &self.strategy)
    }
}

#[cfg(test)]
mod ml_optimizer_tests {
    use super::*;
    use quantrs2_circuit::prelude::Circuit;

    #[test]
    fn test_ml_optimizer_creation() {
        let opt = MLCircuitOptimizer::new(MLOptimizationStrategy::MinimizeGateCount);
        assert_eq!(opt.strategy, MLOptimizationStrategy::MinimizeGateCount);
        assert_eq!(opt.max_passes, 5);
    }

    #[test]
    fn test_circuit_score_basic() {
        let mut circuit = Circuit::<2>::new();
        let _ = circuit.h(0);
        let _ = circuit.cnot(0, 1);

        let score = CircuitScore::compute(&circuit, &MLOptimizationStrategy::BalancedOptimization);
        assert_eq!(score.total_gate_count, 2);
        assert_eq!(score.two_qubit_gate_count, 1);
        assert!(score.composite > 0.0);
    }

    #[test]
    fn test_ml_optimizer_optimize() {
        let opt = MLCircuitOptimizer::new(MLOptimizationStrategy::BalancedOptimization);

        let mut circuit = Circuit::<3>::new();
        let _ = circuit.h(0);
        let _ = circuit.h(1);
        let _ = circuit.cnot(0, 1);
        let _ = circuit.cnot(1, 2);

        let result = opt.optimize(&circuit);
        assert!(result.is_ok(), "Optimization should succeed");
        let optimized = result.expect("Optimized circuit");
        assert!(
            optimized.gates().len() > 0,
            "Optimized circuit should have gates"
        );
    }

    #[test]
    fn test_minimize_depth_strategy() {
        let opt = MLCircuitOptimizer::new(MLOptimizationStrategy::MinimizeDepth);
        let mut circuit = Circuit::<2>::new();
        let _ = circuit.h(0);
        let _ = circuit.h(1);

        let score = opt.score(&circuit);
        // Two parallel H gates: depth should be 1
        assert_eq!(score.circuit_depth, 1);
    }
}
