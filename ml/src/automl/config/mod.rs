//! AutoML Configuration Module
//!
//! This module contains all configuration-related structures for the Quantum AutoML framework.
//! It addresses the configuration explosion issue by organizing related configurations into
//! logical sub-modules.

pub mod evaluation;
pub mod objectives;
pub mod quantum_constraints;
pub mod search_budget;
pub mod search_space;

pub use evaluation::*;
pub use objectives::*;
pub use quantum_constraints::*;
pub use search_budget::*;
pub use search_space::*;

use std::collections::HashMap;

/// Main Quantum AutoML configuration
#[derive(Debug, Clone)]
pub struct QuantumAutoMLConfig {
    /// Task type (auto-detected if None)
    pub task_type: Option<MLTaskType>,

    /// Search budget configuration
    pub search_budget: SearchBudgetConfig,

    /// Optimization objectives
    pub objectives: Vec<OptimizationObjective>,

    /// Search space configuration
    pub search_space: SearchSpaceConfig,

    /// Quantum resource constraints
    pub quantum_constraints: QuantumConstraints,

    /// Evaluation configuration
    pub evaluation_config: EvaluationConfig,

    /// Advanced features
    pub advanced_features: AdvancedAutoMLFeatures,
}

/// Machine learning task types
#[derive(Debug, Clone, PartialEq)]
pub enum MLTaskType {
    /// Binary classification
    BinaryClassification,

    /// Multi-class classification
    MultiClassification { num_classes: usize },

    /// Multi-label classification
    MultiLabelClassification { num_labels: usize },

    /// Regression
    Regression,

    /// Time series forecasting
    TimeSeriesForecasting { horizon: usize },

    /// Clustering
    Clustering { num_clusters: Option<usize> },

    /// Anomaly detection
    AnomalyDetection,

    /// Dimensionality reduction
    DimensionalityReduction { target_dim: Option<usize> },

    /// Reinforcement learning
    ReinforcementLearning,

    /// Generative modeling
    GenerativeModeling,
}

/// Advanced AutoML features
#[derive(Debug, Clone)]
pub struct AdvancedAutoMLFeatures {
    /// Automated online learning
    pub online_learning: bool,

    /// Automated model interpretability
    pub interpretability: bool,

    /// Automated anomaly detection in pipelines
    pub pipeline_anomaly_detection: bool,

    /// Automated deployment optimization
    pub deployment_optimization: bool,

    /// Quantum error mitigation automation
    pub quantum_error_mitigation: bool,

    /// Automated warm-start from previous runs
    pub warm_start: bool,

    /// Multi-objective optimization
    pub multi_objective: bool,

    /// Automated fairness optimization
    pub fairness_optimization: bool,
}

impl Default for AdvancedAutoMLFeatures {
    fn default() -> Self {
        Self {
            online_learning: false,
            interpretability: true,
            pipeline_anomaly_detection: true,
            deployment_optimization: false,
            quantum_error_mitigation: true,
            warm_start: true,
            multi_objective: true,
            fairness_optimization: false,
        }
    }
}

impl QuantumAutoMLConfig {
    /// Create a basic configuration for rapid prototyping
    pub fn basic() -> Self {
        Self {
            task_type: None,
            search_budget: SearchBudgetConfig::quick(),
            objectives: vec![OptimizationObjective::MaximizeAccuracy { weight: 1.0 }],
            search_space: SearchSpaceConfig::default(),
            quantum_constraints: QuantumConstraints::default(),
            evaluation_config: EvaluationConfig::default(),
            advanced_features: AdvancedAutoMLFeatures::default(),
        }
    }

    /// Create a comprehensive configuration for full exploration
    pub fn comprehensive() -> Self {
        Self {
            task_type: None,
            search_budget: SearchBudgetConfig::extensive(),
            objectives: vec![
                OptimizationObjective::MaximizeAccuracy { weight: 0.6 },
                OptimizationObjective::MinimizeComplexity { weight: 0.2 },
                OptimizationObjective::MaximizeQuantumAdvantage { weight: 0.2 },
            ],
            search_space: SearchSpaceConfig::comprehensive(),
            quantum_constraints: QuantumConstraints::default(),
            evaluation_config: EvaluationConfig::rigorous(),
            advanced_features: AdvancedAutoMLFeatures {
                online_learning: true,
                interpretability: true,
                pipeline_anomaly_detection: true,
                deployment_optimization: true,
                quantum_error_mitigation: true,
                warm_start: true,
                multi_objective: true,
                fairness_optimization: true,
            },
        }
    }

    /// Create a production-ready configuration
    pub fn production() -> Self {
        Self {
            task_type: None,
            search_budget: SearchBudgetConfig::production(),
            objectives: vec![
                OptimizationObjective::MaximizeAccuracy { weight: 0.5 },
                OptimizationObjective::MinimizeInferenceTime { weight: 0.3 },
                OptimizationObjective::MaximizeRobustness { weight: 0.2 },
            ],
            search_space: SearchSpaceConfig::production(),
            quantum_constraints: QuantumConstraints::production(),
            evaluation_config: EvaluationConfig::production(),
            advanced_features: AdvancedAutoMLFeatures {
                online_learning: false,
                interpretability: true,
                pipeline_anomaly_detection: true,
                deployment_optimization: true,
                quantum_error_mitigation: true,
                warm_start: true,
                multi_objective: false,
                fairness_optimization: true,
            },
        }
    }
}
