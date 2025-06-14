//! Performance evaluation components

use std::collections::HashMap;
use std::time::Duration;

use super::*;

/// Performance evaluator
#[derive(Debug, Clone)]
pub struct PerformanceEvaluator {
    /// Evaluation metrics
    pub evaluation_metrics: Vec<EvaluationMetric>,
    /// Baseline comparisons
    pub baseline_comparisons: Vec<BaselineComparison>,
    /// Performance history
    pub performance_history: Vec<PerformanceRecord>,
    /// Evaluation cache
    pub evaluation_cache: HashMap<String, EvaluationResult>,
}

impl PerformanceEvaluator {
    pub fn new() -> Result<Self, String> {
        Ok(Self {
            evaluation_metrics: vec![
                EvaluationMetric::SolutionQuality,
                EvaluationMetric::ComputationTime,
                EvaluationMetric::MemoryUsage,
            ],
            baseline_comparisons: Vec::new(),
            performance_history: Vec::new(),
            evaluation_cache: HashMap::new(),
        })
    }
}

/// Baseline comparison
#[derive(Debug, Clone)]
pub struct BaselineComparison {
    /// Baseline name
    pub baseline_name: String,
    /// Baseline strategy
    pub baseline_strategy: DecompositionStrategy,
    /// Performance comparison
    pub performance_comparison: PerformanceComparison,
}

/// Performance comparison
#[derive(Debug, Clone)]
pub struct PerformanceComparison {
    /// Improvement factor
    pub improvement_factor: f64,
    /// Statistical significance
    pub statistical_significance: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
}

/// Evaluation result
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// Individual metric scores
    pub metric_scores: HashMap<EvaluationMetric, f64>,
    /// Overall evaluation score
    pub overall_score: f64,
    /// Performance improvement
    pub improvement_over_baseline: f64,
    /// Evaluation confidence
    pub confidence: f64,
}