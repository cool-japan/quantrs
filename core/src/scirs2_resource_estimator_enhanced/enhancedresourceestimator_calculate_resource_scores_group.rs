//! # EnhancedResourceEstimator - calculate_resource_scores_group Methods
//!
//! This module contains method implementations for `EnhancedResourceEstimator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::parallel_ops_stubs::*;

use super::types::{BasicResourceAnalysis, MLPredictions, ReadinessLevel, ResourceScores};

use super::enhancedresourceestimator_type::EnhancedResourceEstimator;

impl EnhancedResourceEstimator {
    /// Calculate resource scores
    pub(super) fn calculate_resource_scores(
        &self,
        basic: &BasicResourceAnalysis,
        ml_predictions: &Option<MLPredictions>,
    ) -> ResourceScores {
        let efficiency_score = self.calculate_efficiency_score(basic);
        let scalability_score = self.calculate_scalability_score(basic);
        let feasibility_score = self.calculate_feasibility_score(basic, ml_predictions);
        let optimization_potential = self.calculate_optimization_potential(basic);
        ResourceScores {
            overall_score: (efficiency_score + scalability_score + feasibility_score) / 3.0,
            efficiency_score,
            scalability_score,
            feasibility_score,
            optimization_potential,
            readiness_level: self.determine_readiness_level(feasibility_score),
        }
    }
    /// Calculate efficiency score
    pub(super) fn calculate_efficiency_score(&self, basic: &BasicResourceAnalysis) -> f64 {
        let gate_efficiency = 1.0 / (1.0 + basic.gate_statistics.total_gates as f64 / 1000.0);
        let depth_efficiency = 1.0 / (1.0 + basic.complexity_metrics.t_depth as f64 / 100.0);
        let qubit_efficiency = 1.0 / (1.0 + basic.num_qubits as f64 / 50.0);
        (gate_efficiency + depth_efficiency + qubit_efficiency) / 3.0
    }
    /// Calculate scalability score
    pub(super) fn calculate_scalability_score(&self, basic: &BasicResourceAnalysis) -> f64 {
        let connectivity_score = 1.0 - basic.circuit_topology.connectivity_density.min(1.0);
        let volume_score = 1.0 / (1.0 + (basic.complexity_metrics.circuit_volume as f64).log10());
        f64::midpoint(connectivity_score, volume_score)
    }
    /// Calculate feasibility score
    pub(super) const fn calculate_feasibility_score(
        &self,
        basic: &BasicResourceAnalysis,
        ml_predictions: &Option<MLPredictions>,
    ) -> f64 {
        let base_score = if basic.resource_requirements.physical_qubits < 1000 {
            0.9
        } else if basic.resource_requirements.physical_qubits < 10000 {
            0.6
        } else {
            0.3
        };
        if let Some(predictions) = ml_predictions {
            f64::midpoint(base_score, predictions.feasibility_confidence)
        } else {
            base_score
        }
    }
    /// Calculate optimization potential
    pub(super) fn calculate_optimization_potential(&self, basic: &BasicResourceAnalysis) -> f64 {
        let pattern_potential = basic.gate_statistics.gate_patterns.len() as f64 * 0.1;
        let redundancy_potential = 0.2;
        (pattern_potential + redundancy_potential).min(1.0)
    }
    /// Determine readiness level
    pub(super) fn determine_readiness_level(&self, feasibility_score: f64) -> ReadinessLevel {
        if feasibility_score > 0.8 {
            ReadinessLevel::ProductionReady
        } else if feasibility_score > 0.6 {
            ReadinessLevel::Experimental
        } else if feasibility_score > 0.4 {
            ReadinessLevel::Research
        } else {
            ReadinessLevel::Theoretical
        }
    }
}
