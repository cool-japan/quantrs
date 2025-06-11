//! Statistical analysis and validation tools for VQA
//!
//! This module provides statistical methods for analyzing VQA convergence,
//! parameter distributions, and optimization landscapes.

use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Statistical analysis results for VQA optimization
#[derive(Debug, Clone)]
pub struct VQAStatistics {
    /// Convergence metrics
    pub convergence_rate: f64,
    /// Parameter variance across iterations
    pub parameter_variance: Vec<f64>,
    /// Objective function statistics
    pub objective_stats: ObjectiveStatistics,
}

/// Objective function statistical metrics
#[derive(Debug, Clone)]
pub struct ObjectiveStatistics {
    /// Mean objective value
    pub mean: f64,
    /// Standard deviation
    pub std: f64,
    /// Best achieved value
    pub best: f64,
    /// Worst value encountered
    pub worst: f64,
}

impl Default for VQAStatistics {
    fn default() -> Self {
        Self {
            convergence_rate: 0.0,
            parameter_variance: Vec::new(),
            objective_stats: ObjectiveStatistics::default(),
        }
    }
}

impl Default for ObjectiveStatistics {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std: 0.0,
            best: 0.0,
            worst: 0.0,
        }
    }
}

/// Analyze VQA convergence statistics
pub fn analyze_convergence(objective_history: &[f64]) -> VQAStatistics {
    if objective_history.is_empty() {
        return VQAStatistics::default();
    }

    let mean = objective_history.iter().sum::<f64>() / objective_history.len() as f64;
    let variance: f64 = objective_history
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>()
        / objective_history.len() as f64;
    let std = variance.sqrt();

    let best = objective_history
        .iter()
        .fold(f64::INFINITY, |a, &b| a.min(b));
    let worst = objective_history
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    VQAStatistics {
        convergence_rate: if objective_history.len() > 1 {
            (objective_history[0] - objective_history[objective_history.len() - 1]).abs()
                / objective_history.len() as f64
        } else {
            0.0
        },
        parameter_variance: vec![variance],
        objective_stats: ObjectiveStatistics {
            mean,
            std,
            best,
            worst,
        },
    }
}
