//! Constraint Handling Configuration Types

/// Constraint handling configuration
#[derive(Debug, Clone)]
pub struct ConstraintConfig {
    /// Constraint handling method
    pub handling_method: ConstraintHandlingMethod,
    /// Constraint violation penalty
    pub violation_penalty: f64,
    /// Feasibility threshold
    pub feasibility_threshold: f64,
    /// Constraint approximation method
    pub approximation_method: ConstraintApproximationMethod,
}

impl Default for ConstraintConfig {
    fn default() -> Self {
        Self {
            handling_method: ConstraintHandlingMethod::ExpectedFeasibility,
            violation_penalty: 1000.0,
            feasibility_threshold: 1e-6,
            approximation_method: ConstraintApproximationMethod::GaussianProcess,
        }
    }
}

/// Constraint handling methods
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintHandlingMethod {
    /// Expected feasibility approach
    ExpectedFeasibility,
    /// Penalty method
    PenaltyMethod,
    /// Augmented Lagrangian
    AugmentedLagrangian,
    /// Constraint-dominated optimization
    ConstraintDominated,
    /// Feasibility rules
    FeasibilityRules,
}

/// Constraint approximation methods
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintApproximationMethod {
    /// Gaussian process model
    GaussianProcess,
    /// Support vector machine
    SupportVectorMachine,
    /// Random forest
    RandomForest,
    /// Neural network
    NeuralNetwork,
}
