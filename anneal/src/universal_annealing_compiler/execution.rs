//! Execution result types.
//!
//! This module contains types for representing execution results,
//! predictions, and metadata.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::compilation::CompilationResult;
use super::config::{OptimizationLevel, ResourceAllocationStrategy, SchedulingPriority};
use super::platform::QuantumPlatform;

/// Universal execution result
#[derive(Debug, Clone)]
pub struct UniversalExecutionResult {
    /// Problem identifier
    pub problem_id: String,
    /// Selected optimal platform
    pub optimal_platform: QuantumPlatform,
    /// Compilation results for all platforms
    pub compilation_results: HashMap<QuantumPlatform, CompilationResult>,
    /// Performance predictions
    pub performance_predictions: HashMap<QuantumPlatform, PlatformPerformancePrediction>,
    /// Execution result
    pub execution_result: PlatformExecutionResult,
    /// Total execution time
    pub total_time: Duration,
    /// Execution metadata
    pub metadata: UniversalExecutionMetadata,
}

/// Platform performance prediction
#[derive(Debug, Clone)]
pub struct PlatformPerformancePrediction {
    /// Target platform
    pub platform: QuantumPlatform,
    /// Predicted performance
    pub predicted_performance: PredictedPerformance,
    /// Confidence in prediction
    pub confidence_score: f64,
    /// Prediction metadata
    pub prediction_metadata: PredictionMetadata,
}

/// Predicted performance
#[derive(Debug, Clone)]
pub struct PredictedPerformance {
    /// Execution time
    pub execution_time: Duration,
    /// Solution quality
    pub solution_quality: f64,
    /// Success probability
    pub success_probability: f64,
    /// Cost
    pub cost: f64,
    /// Reliability score
    pub reliability_score: f64,
}

/// Prediction metadata
#[derive(Debug, Clone)]
pub struct PredictionMetadata {
    /// Model version
    pub model_version: String,
    /// Prediction timestamp
    pub prediction_timestamp: Instant,
    /// Features used
    pub features_used: Vec<String>,
    /// Model accuracy
    pub model_accuracy: f64,
}

/// Optimal platform selection
#[derive(Debug, Clone)]
pub struct OptimalPlatformSelection {
    /// Selected platform
    pub platform: QuantumPlatform,
    /// Selection score
    pub selection_score: f64,
    /// Selection rationale
    pub selection_rationale: String,
    /// Alternative platforms
    pub alternatives: Vec<QuantumPlatform>,
    /// Selection metadata
    pub selection_metadata: SelectionMetadata,
}

/// Selection metadata
#[derive(Debug, Clone)]
pub struct SelectionMetadata {
    /// Selection timestamp
    pub selection_timestamp: Instant,
    /// Strategy used
    pub strategy_used: ResourceAllocationStrategy,
    /// Confidence
    pub confidence: f64,
}

/// Execution plan
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Target platform
    pub platform: QuantumPlatform,
    /// Scheduled start time
    pub scheduled_start_time: Instant,
    /// Estimated duration
    pub estimated_duration: Duration,
    /// Resource allocation
    pub resource_allocation: PlatformResourceAllocation,
    /// Execution parameters
    pub execution_parameters: ExecutionParameters,
}

/// Platform resource allocation
#[derive(Debug, Clone)]
pub struct PlatformResourceAllocation {
    /// Allocated qubits
    pub qubits: Vec<usize>,
    /// Execution priority
    pub execution_priority: SchedulingPriority,
    /// Resource reservation
    pub resource_reservation: ResourceReservationInfo,
}

/// Resource reservation information
#[derive(Debug, Clone)]
pub struct ResourceReservationInfo {
    /// Reservation identifier
    pub reservation_id: String,
    /// Reserved until
    pub reserved_until: Instant,
}

/// Execution parameters
#[derive(Debug, Clone)]
pub struct ExecutionParameters {
    /// Number of shots
    pub shots: usize,
    /// Optimization level
    pub optimization_level: OptimizationLevel,
    /// Error mitigation enabled
    pub error_mitigation: bool,
}

/// Platform execution result
#[derive(Debug, Clone)]
pub struct PlatformExecutionResult {
    /// Platform used
    pub platform: QuantumPlatform,
    /// Execution identifier
    pub execution_id: String,
    /// Solution found
    pub solution: Vec<i32>,
    /// Objective value
    pub objective_value: f64,
    /// Execution time
    pub execution_time: Duration,
    /// Success indicator
    pub success: bool,
    /// Quality metrics
    pub quality_metrics: ExecutionQualityMetrics,
    /// Resource usage
    pub resource_usage: ExecutionResourceUsage,
    /// Execution metadata
    pub metadata: ExecutionMetadata,
}

/// Execution quality metrics
#[derive(Debug, Clone)]
pub struct ExecutionQualityMetrics {
    /// Solution quality
    pub solution_quality: f64,
    /// Fidelity
    pub fidelity: f64,
    /// Success probability
    pub success_probability: f64,
}

/// Execution resource usage
#[derive(Debug, Clone)]
pub struct ExecutionResourceUsage {
    /// Qubits used
    pub qubits_used: usize,
    /// Shots executed
    pub shots_executed: usize,
    /// Classical compute time
    pub classical_compute_time: Duration,
    /// Cost incurred
    pub cost_incurred: f64,
}

/// Execution metadata
#[derive(Debug, Clone)]
pub struct ExecutionMetadata {
    /// Execution timestamp
    pub execution_timestamp: Instant,
    /// Platform version
    pub platform_version: String,
    /// Execution environment
    pub execution_environment: String,
}

/// Universal execution metadata
#[derive(Debug, Clone)]
pub struct UniversalExecutionMetadata {
    /// Compiler version
    pub compiler_version: String,
    /// Platforms considered
    pub platforms_considered: usize,
    /// Optimization level used
    pub optimization_level: OptimizationLevel,
    /// Cost savings achieved
    pub cost_savings: f64,
    /// Performance improvement
    pub performance_improvement: f64,
}

/// Performance predictor backed by a real, growing history of observed
/// [`PlatformExecutionResult`]s per platform.
///
/// Rather than emitting fixed confidence/accuracy constants regardless of
/// input, [`Self::predict`]/[`Self::model_accuracy`]/[`Self::confidence_score`]
/// derive their outputs from whatever execution history has actually been
/// recorded via [`Self::record_result`] for that platform. With no history
/// for a platform, prediction honestly returns `None` rather than a
/// fabricated guess.
///
/// Note: this crate does not yet wire `record_result`/`predict` into
/// [`super::compiler::UniversalAnnealingCompiler`]'s `predict_performance` /
/// `update_performance_models`, which still construct
/// [`PlatformPerformancePrediction`] with fixed constants; see the crate's
/// TODOs for that remaining integration.
#[derive(Debug, Default)]
pub struct PerformancePredictor {
    /// Observed execution results, keyed by platform.
    history: HashMap<QuantumPlatform, Vec<PlatformExecutionResult>>,
}

impl PerformancePredictor {
    /// Create a new performance predictor with empty history.
    #[must_use]
    pub fn new() -> Self {
        Self {
            history: HashMap::new(),
        }
    }

    /// Record a real execution outcome, growing this platform's history.
    pub fn record_result(&mut self, result: &PlatformExecutionResult) {
        self.history
            .entry(result.platform.clone())
            .or_default()
            .push(result.clone());
    }

    /// Number of recorded results for `platform`.
    #[must_use]
    pub fn sample_count(&self, platform: &QuantumPlatform) -> usize {
        self.history.get(platform).map_or(0, Vec::len)
    }

    /// Predict performance for `platform` from its real recorded history.
    /// Returns `None` if nothing has been recorded for this platform yet.
    #[must_use]
    pub fn predict(&self, platform: &QuantumPlatform) -> Option<PredictedPerformance> {
        let results = self.history.get(platform)?;
        if results.is_empty() {
            return None;
        }
        let n = results.len() as f64;

        let mean_time_secs = results
            .iter()
            .map(|r| r.execution_time.as_secs_f64())
            .sum::<f64>()
            / n;
        let mean_quality = results
            .iter()
            .map(|r| r.quality_metrics.solution_quality)
            .sum::<f64>()
            / n;
        let mean_success_probability = results
            .iter()
            .map(|r| r.quality_metrics.success_probability)
            .sum::<f64>()
            / n;
        let mean_cost = results
            .iter()
            .map(|r| r.resource_usage.cost_incurred)
            .sum::<f64>()
            / n;
        let success_rate = results.iter().filter(|r| r.success).count() as f64 / n;

        Some(PredictedPerformance {
            execution_time: Duration::from_secs_f64(mean_time_secs.max(0.0)),
            solution_quality: mean_quality,
            success_probability: mean_success_probability,
            cost: mean_cost,
            reliability_score: success_rate,
        })
    }

    /// Real model accuracy for `platform`: `1 - coefficient_of_variation` of
    /// the recorded solution-quality samples, clamped to `[0, 1]`. A
    /// platform whose real outcomes are consistent scores near 1; one whose
    /// outcomes vary wildly scores low. Returns `0.0` (honestly "no
    /// evidence yet") when fewer than two samples have been recorded.
    #[must_use]
    pub fn model_accuracy(&self, platform: &QuantumPlatform) -> f64 {
        let Some(results) = self.history.get(platform) else {
            return 0.0;
        };
        if results.len() < 2 {
            return 0.0;
        }
        let n = results.len() as f64;
        let mean = results
            .iter()
            .map(|r| r.quality_metrics.solution_quality)
            .sum::<f64>()
            / n;
        if mean.abs() < 1e-12 {
            return 0.0;
        }
        let variance = results
            .iter()
            .map(|r| (r.quality_metrics.solution_quality - mean).powi(2))
            .sum::<f64>()
            / n;
        let coefficient_of_variation = variance.sqrt() / mean.abs();
        (1.0 - coefficient_of_variation).clamp(0.0, 1.0)
    }

    /// Real confidence score for `platform`: grows monotonically with the
    /// amount of real recorded evidence (`n / (n + 5)`), rather than a fixed
    /// constant regardless of how much (or how little) history exists.
    #[must_use]
    pub fn confidence_score(&self, platform: &QuantumPlatform) -> f64 {
        let n = self.sample_count(platform) as f64;
        n / (n + 5.0)
    }
}

/// Cost optimizer backed by a real, growing history of observed platform
/// costs, rather than emitting fabricated recommendations.
#[derive(Debug, Default)]
pub struct CostOptimizer {
    /// Observed incurred costs, keyed by platform.
    cost_history: HashMap<QuantumPlatform, Vec<f64>>,
}

impl CostOptimizer {
    /// Create a new cost optimizer with empty history.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cost_history: HashMap::new(),
        }
    }

    /// Record a real observed cost for `platform`.
    pub fn record_cost(&mut self, platform: QuantumPlatform, cost: f64) {
        self.cost_history.entry(platform).or_default().push(cost);
    }

    /// Mean of the real recorded costs for `platform`, or `None` if nothing
    /// has been recorded yet.
    #[must_use]
    pub fn estimate_cost(&self, platform: &QuantumPlatform) -> Option<f64> {
        let costs = self.cost_history.get(platform)?;
        if costs.is_empty() {
            return None;
        }
        Some(costs.iter().sum::<f64>() / costs.len() as f64)
    }

    /// Recommend the platform among `candidates` with the lowest real mean
    /// recorded cost. Candidates with no recorded history are skipped
    /// (rather than fabricating a cost for them); returns `None` if none of
    /// the candidates have any recorded history.
    #[must_use]
    pub fn recommend_cheapest<'a>(
        &self,
        candidates: &'a [QuantumPlatform],
    ) -> Option<&'a QuantumPlatform> {
        candidates
            .iter()
            .filter_map(|platform| self.estimate_cost(platform).map(|cost| (platform, cost)))
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(platform, _)| platform)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(
        platform: QuantumPlatform,
        quality: f64,
        cost: f64,
        success: bool,
    ) -> PlatformExecutionResult {
        PlatformExecutionResult {
            platform,
            execution_id: "test".to_string(),
            solution: vec![1, 0, 1],
            objective_value: -1.0,
            execution_time: Duration::from_millis(100),
            success,
            quality_metrics: ExecutionQualityMetrics {
                solution_quality: quality,
                fidelity: 0.95,
                success_probability: if success { 0.9 } else { 0.1 },
            },
            resource_usage: ExecutionResourceUsage {
                qubits_used: 4,
                shots_executed: 100,
                classical_compute_time: Duration::from_millis(10),
                cost_incurred: cost,
            },
            metadata: ExecutionMetadata {
                execution_timestamp: Instant::now(),
                platform_version: "1.0".to_string(),
                execution_environment: "test".to_string(),
            },
        }
    }

    #[test]
    fn performance_predictor_has_no_prediction_without_real_history() {
        let predictor = PerformancePredictor::new();
        assert!(predictor.predict(&QuantumPlatform::DWave).is_none());
        assert_eq!(predictor.model_accuracy(&QuantumPlatform::DWave), 0.0);
        assert_eq!(predictor.confidence_score(&QuantumPlatform::DWave), 0.0);
    }

    #[test]
    fn performance_predictor_derives_real_predictions_from_recorded_history() {
        let mut predictor = PerformancePredictor::new();
        predictor.record_result(&make_result(QuantumPlatform::DWave, 0.8, 1.0, true));
        predictor.record_result(&make_result(QuantumPlatform::DWave, 0.9, 2.0, true));
        predictor.record_result(&make_result(QuantumPlatform::DWave, 0.7, 3.0, false));

        let prediction = predictor
            .predict(&QuantumPlatform::DWave)
            .expect("prediction should exist once history has been recorded");

        assert!((prediction.solution_quality - 0.8).abs() < 1e-9);
        assert!((prediction.cost - 2.0).abs() < 1e-9);
        // 2 of 3 recorded runs succeeded -> real reliability, not a fixed 0.9.
        assert!((prediction.reliability_score - (2.0 / 3.0)).abs() < 1e-9);

        // Confidence must grow with recorded evidence rather than stay fixed.
        let confidence_after_3 = predictor.confidence_score(&QuantumPlatform::DWave);
        predictor.record_result(&make_result(QuantumPlatform::DWave, 0.85, 1.5, true));
        let confidence_after_4 = predictor.confidence_score(&QuantumPlatform::DWave);
        assert!(confidence_after_4 > confidence_after_3);
    }

    #[test]
    fn performance_predictor_accuracy_reflects_real_outcome_consistency() {
        let mut consistent = PerformancePredictor::new();
        consistent.record_result(&make_result(QuantumPlatform::IBM, 0.9, 1.0, true));
        consistent.record_result(&make_result(QuantumPlatform::IBM, 0.91, 1.0, true));
        consistent.record_result(&make_result(QuantumPlatform::IBM, 0.89, 1.0, true));

        let mut erratic = PerformancePredictor::new();
        erratic.record_result(&make_result(QuantumPlatform::IBM, 0.1, 1.0, true));
        erratic.record_result(&make_result(QuantumPlatform::IBM, 0.9, 1.0, true));
        erratic.record_result(&make_result(QuantumPlatform::IBM, 0.2, 1.0, false));

        let consistent_accuracy = consistent.model_accuracy(&QuantumPlatform::IBM);
        let erratic_accuracy = erratic.model_accuracy(&QuantumPlatform::IBM);

        assert!(
            consistent_accuracy > erratic_accuracy,
            "a platform with consistent real outcomes must score higher accuracy than an erratic one \
             (consistent={consistent_accuracy}, erratic={erratic_accuracy})"
        );
    }

    #[test]
    fn cost_optimizer_recommends_the_real_cheapest_platform() {
        let mut optimizer = CostOptimizer::new();
        optimizer.record_cost(QuantumPlatform::DWave, 5.0);
        optimizer.record_cost(QuantumPlatform::DWave, 7.0);
        optimizer.record_cost(QuantumPlatform::IBM, 1.0);
        optimizer.record_cost(QuantumPlatform::IBM, 2.0);

        assert!((optimizer.estimate_cost(&QuantumPlatform::DWave).unwrap() - 6.0).abs() < 1e-9);
        assert!((optimizer.estimate_cost(&QuantumPlatform::IBM).unwrap() - 1.5).abs() < 1e-9);

        let candidates = vec![QuantumPlatform::DWave, QuantumPlatform::IBM];
        let cheapest = optimizer
            .recommend_cheapest(&candidates)
            .expect("a cheapest platform should be found");
        assert_eq!(*cheapest, QuantumPlatform::IBM);
    }

    #[test]
    fn cost_optimizer_skips_platforms_with_no_recorded_history() {
        let mut optimizer = CostOptimizer::new();
        optimizer.record_cost(QuantumPlatform::IBM, 3.0);

        assert!(optimizer.estimate_cost(&QuantumPlatform::DWave).is_none());

        let candidates = vec![QuantumPlatform::DWave, QuantumPlatform::IBM];
        let cheapest = optimizer
            .recommend_cheapest(&candidates)
            .expect("should still find the one platform with real history");
        assert_eq!(*cheapest, QuantumPlatform::IBM);
    }
}
