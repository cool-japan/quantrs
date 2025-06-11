//! Optimization Objectives
//!
//! This module defines optimization objectives and multi-objective optimization configurations.

/// Optimization objectives
#[derive(Debug, Clone)]
pub enum OptimizationObjective {
    /// Maximize accuracy/performance
    MaximizeAccuracy { weight: f64 },

    /// Minimize model complexity
    MinimizeComplexity { weight: f64 },

    /// Minimize quantum resource usage
    MinimizeQuantumResources { weight: f64 },

    /// Maximize quantum advantage
    MaximizeQuantumAdvantage { weight: f64 },

    /// Minimize inference time
    MinimizeInferenceTime { weight: f64 },

    /// Minimize training time
    MinimizeTrainingTime { weight: f64 },

    /// Maximize robustness
    MaximizeRobustness { weight: f64 },

    /// Maximize interpretability
    MaximizeInterpretability { weight: f64 },
}

impl OptimizationObjective {
    /// Get the weight of this objective
    pub fn weight(&self) -> f64 {
        match self {
            OptimizationObjective::MaximizeAccuracy { weight } => *weight,
            OptimizationObjective::MinimizeComplexity { weight } => *weight,
            OptimizationObjective::MinimizeQuantumResources { weight } => *weight,
            OptimizationObjective::MaximizeQuantumAdvantage { weight } => *weight,
            OptimizationObjective::MinimizeInferenceTime { weight } => *weight,
            OptimizationObjective::MinimizeTrainingTime { weight } => *weight,
            OptimizationObjective::MaximizeRobustness { weight } => *weight,
            OptimizationObjective::MaximizeInterpretability { weight } => *weight,
        }
    }

    /// Check if this is a maximization objective
    pub fn is_maximization(&self) -> bool {
        matches!(
            self,
            OptimizationObjective::MaximizeAccuracy { .. }
                | OptimizationObjective::MaximizeQuantumAdvantage { .. }
                | OptimizationObjective::MaximizeRobustness { .. }
                | OptimizationObjective::MaximizeInterpretability { .. }
        )
    }

    /// Get the objective name
    pub fn name(&self) -> &'static str {
        match self {
            OptimizationObjective::MaximizeAccuracy { .. } => "accuracy",
            OptimizationObjective::MinimizeComplexity { .. } => "complexity",
            OptimizationObjective::MinimizeQuantumResources { .. } => "quantum_resources",
            OptimizationObjective::MaximizeQuantumAdvantage { .. } => "quantum_advantage",
            OptimizationObjective::MinimizeInferenceTime { .. } => "inference_time",
            OptimizationObjective::MinimizeTrainingTime { .. } => "training_time",
            OptimizationObjective::MaximizeRobustness { .. } => "robustness",
            OptimizationObjective::MaximizeInterpretability { .. } => "interpretability",
        }
    }
}

/// Common objective configurations
impl OptimizationObjective {
    /// Single accuracy objective
    pub fn accuracy_only() -> Vec<Self> {
        vec![OptimizationObjective::MaximizeAccuracy { weight: 1.0 }]
    }

    /// Balanced accuracy and efficiency
    pub fn balanced() -> Vec<Self> {
        vec![
            OptimizationObjective::MaximizeAccuracy { weight: 0.5 },
            OptimizationObjective::MinimizeInferenceTime { weight: 0.3 },
            OptimizationObjective::MinimizeComplexity { weight: 0.2 },
        ]
    }

    /// Quantum-focused objectives
    pub fn quantum_focused() -> Vec<Self> {
        vec![
            OptimizationObjective::MaximizeQuantumAdvantage { weight: 0.5 },
            OptimizationObjective::MaximizeAccuracy { weight: 0.3 },
            OptimizationObjective::MinimizeQuantumResources { weight: 0.2 },
        ]
    }

    /// Production-ready objectives
    pub fn production() -> Vec<Self> {
        vec![
            OptimizationObjective::MaximizeAccuracy { weight: 0.4 },
            OptimizationObjective::MaximizeRobustness { weight: 0.3 },
            OptimizationObjective::MinimizeInferenceTime { weight: 0.2 },
            OptimizationObjective::MaximizeInterpretability { weight: 0.1 },
        ]
    }

    /// Research objectives
    pub fn research() -> Vec<Self> {
        vec![
            OptimizationObjective::MaximizeQuantumAdvantage { weight: 0.4 },
            OptimizationObjective::MaximizeAccuracy { weight: 0.3 },
            OptimizationObjective::MaximizeInterpretability { weight: 0.2 },
            OptimizationObjective::MinimizeComplexity { weight: 0.1 },
        ]
    }

    /// Fast inference objectives
    pub fn fast_inference() -> Vec<Self> {
        vec![
            OptimizationObjective::MinimizeInferenceTime { weight: 0.5 },
            OptimizationObjective::MaximizeAccuracy { weight: 0.3 },
            OptimizationObjective::MinimizeComplexity { weight: 0.2 },
        ]
    }

    /// Resource-constrained objectives
    pub fn resource_constrained() -> Vec<Self> {
        vec![
            OptimizationObjective::MinimizeQuantumResources { weight: 0.4 },
            OptimizationObjective::MaximizeAccuracy { weight: 0.3 },
            OptimizationObjective::MinimizeComplexity { weight: 0.2 },
            OptimizationObjective::MinimizeTrainingTime { weight: 0.1 },
        ]
    }
}
