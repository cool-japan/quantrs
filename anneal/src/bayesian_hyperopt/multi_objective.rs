//! Multi-Objective Optimization Configuration Types

/// Multi-objective optimization configuration
#[derive(Debug, Clone)]
pub struct MultiObjectiveConfig {
    /// Scalarization method
    pub scalarization_method: ScalarizationMethod,
    /// Reference point for hypervolume calculation
    pub reference_point: Option<Vec<f64>>,
    /// Objective weights
    pub objective_weights: Vec<f64>,
    /// Pareto front approximation method
    pub pareto_method: ParetoApproximationMethod,
    /// Number of objectives
    pub num_objectives: usize,
}

impl Default for MultiObjectiveConfig {
    fn default() -> Self {
        Self {
            scalarization_method: ScalarizationMethod::WeightedSum,
            reference_point: None,
            objective_weights: vec![1.0],
            pareto_method: ParetoApproximationMethod::NonDominatedSorting,
            num_objectives: 1,
        }
    }
}

/// Scalarization methods for multi-objective optimization
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScalarizationMethod {
    /// Weighted sum approach
    WeightedSum,
    /// Tchebycheff method
    Tchebycheff,
    /// Achievement scalarizing function
    Achievement,
    /// Hypervolume indicator
    Hypervolume,
    /// Expected hypervolume improvement
    ExpectedHypervolumeImprovement,
}

/// Pareto front approximation methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParetoApproximationMethod {
    /// Non-dominated sorting
    NonDominatedSorting,
    /// NSGA-II approach
    NSGAII,
    /// SPEA2 method
    SPEA2,
    /// Hypervolume-based selection
    HypervolumeSelection,
}
