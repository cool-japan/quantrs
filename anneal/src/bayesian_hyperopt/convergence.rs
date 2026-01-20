//! Convergence Configuration Types

/// Convergence criteria configuration
#[derive(Debug, Clone)]
pub struct ConvergenceConfig {
    /// Tolerance for objective function improvement
    pub objective_tolerance: f64,
    /// Tolerance for parameter changes
    pub parameter_tolerance: f64,
    /// Maximum number of iterations without improvement
    pub max_stagnation: usize,
    /// Confidence level for convergence assessment
    pub confidence_level: f64,
    /// Early stopping criteria
    pub early_stopping: EarlyStoppingCriteria,
}

impl Default for ConvergenceConfig {
    fn default() -> Self {
        Self {
            objective_tolerance: 1e-6,
            parameter_tolerance: 1e-8,
            max_stagnation: 20,
            confidence_level: 0.95,
            early_stopping: EarlyStoppingCriteria::default(),
        }
    }
}

/// Early stopping criteria
#[derive(Debug, Clone)]
pub struct EarlyStoppingCriteria {
    /// Enable early stopping
    pub enabled: bool,
    /// Minimum number of iterations before stopping
    pub min_iterations: usize,
    /// Improvement threshold
    pub improvement_threshold: f64,
    /// Patience (iterations to wait for improvement)
    pub patience: usize,
}

impl Default for EarlyStoppingCriteria {
    fn default() -> Self {
        Self {
            enabled: true,
            min_iterations: 20,
            improvement_threshold: 1e-4,
            patience: 10,
        }
    }
}
