//! # QNNMetrics - Trait Implementations
//!
//! This module contains trait implementations for `QNNMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{
    ComputationalMetrics, QNNMetrics, QuantumMetrics, TrainingMetrics, ValidationMetrics,
};

impl Default for QNNMetrics {
    fn default() -> Self {
        Self {
            training_metrics: TrainingMetrics {
                final_training_loss: 0.0,
                convergence_rate: 0.0,
                epochs_to_convergence: 0,
                training_stability: 0.0,
                overfitting_measure: 0.0,
            },
            validation_metrics: ValidationMetrics {
                best_validation_loss: 0.0,
                validation_accuracy: 0.0,
                generalization_gap: 0.0,
                cv_scores: Vec::new(),
                confidence_intervals: Vec::new(),
            },
            quantum_metrics: QuantumMetrics {
                quantum_volume: 0.0,
                entanglement_measures: Vec::new(),
                quantum_advantage: 0.0,
                fidelity_measures: Vec::new(),
                coherence_utilization: 0.0,
            },
            computational_metrics: ComputationalMetrics {
                training_time_per_epoch: 0.0,
                inference_time: 0.0,
                memory_usage: 0.0,
                quantum_execution_time: 0.0,
                classical_computation_time: 0.0,
            },
        }
    }
}
