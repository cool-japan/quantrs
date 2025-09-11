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
            Self::MaximizeAccuracy { weight }
            | Self::MinimizeComplexity { weight }
            | Self::MinimizeQuantumResources { weight }
            | Self::MaximizeQuantumAdvantage { weight }
            | Self::MinimizeInferenceTime { weight }
            | Self::MinimizeTrainingTime { weight }
            | Self::MaximizeRobustness { weight }
            | Self::MaximizeInterpretability { weight } => *weight,
        }
    }

    /// Check if this is a maximization objective
    pub fn is_maximization(&self) -> bool {
        matches!(
            self,
            Self::MaximizeAccuracy { .. }
                | Self::MaximizeQuantumAdvantage { .. }
                | Self::MaximizeRobustness { .. }
                | Self::MaximizeInterpretability { .. }
        )
    }

    /// Get the objective name
    pub fn name(&self) -> &'static str {
        match self {
            Self::MaximizeAccuracy { .. } => "accuracy",
            Self::MinimizeComplexity { .. } => "complexity",
            Self::MinimizeQuantumResources { .. } => "quantum_resources",
            Self::MaximizeQuantumAdvantage { .. } => "quantum_advantage",
            Self::MinimizeInferenceTime { .. } => "inference_time",
            Self::MinimizeTrainingTime { .. } => "training_time",
            Self::MaximizeRobustness { .. } => "robustness",
            Self::MaximizeInterpretability { .. } => "interpretability",
        }
    }
}

/// Common objective configurations
impl OptimizationObjective {
    /// Single accuracy objective
    pub fn accuracy_only() -> Vec<Self> {
        vec![Self::MaximizeAccuracy { weight: 1.0 }]
    }

    /// Balanced accuracy and efficiency
    pub fn balanced() -> Vec<Self> {
        vec![
            Self::MaximizeAccuracy { weight: 0.5 },
            Self::MinimizeInferenceTime { weight: 0.3 },
            OptimizationObjective::MinimizeComplexity { weight: 0.2 },
        ]
    }

    /// Quantum-focused objectives
    pub fn quantum_focused() -> Vec<Self> {
        vec![
            Self::MaximizeQuantumAdvantage { weight: 0.5 },
            Self::MaximizeAccuracy { weight: 0.3 },
            Self::MinimizeQuantumResources { weight: 0.2 },
        ]
    }

    /// Production-ready objectives
    pub fn production() -> Vec<Self> {
        vec![
            Self::MaximizeAccuracy { weight: 0.4 },
            Self::MaximizeRobustness { weight: 0.3 },
            Self::MinimizeInferenceTime { weight: 0.2 },
            Self::MaximizeInterpretability { weight: 0.1 },
        ]
    }

    /// Research objectives
    pub fn research() -> Vec<Self> {
        vec![
            Self::MaximizeQuantumAdvantage { weight: 0.4 },
            Self::MaximizeAccuracy { weight: 0.3 },
            Self::MaximizeInterpretability { weight: 0.2 },
            Self::MinimizeComplexity { weight: 0.1 },
        ]
    }

    /// Fast inference objectives
    pub fn fast_inference() -> Vec<Self> {
        vec![
            Self::MinimizeInferenceTime { weight: 0.5 },
            Self::MaximizeAccuracy { weight: 0.3 },
            Self::MinimizeComplexity { weight: 0.2 },
        ]
    }

    /// Resource-constrained objectives
    pub fn resource_constrained() -> Vec<Self> {
        vec![
            Self::MinimizeQuantumResources { weight: 0.4 },
            Self::MaximizeAccuracy { weight: 0.3 },
            Self::MinimizeComplexity { weight: 0.2 },
            Self::MinimizeTrainingTime { weight: 0.1 },
        ]
    }
}
