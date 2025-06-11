//! Core types for the solution debugger.

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Problem information
#[derive(Debug, Clone, Serialize)]
pub struct ProblemInfo {
    /// Problem name
    pub name: String,
    /// Problem type
    pub problem_type: String,
    /// Number of variables
    pub num_variables: usize,
    /// Variable mapping
    pub var_map: HashMap<String, usize>,
    /// Reverse variable mapping
    pub reverse_var_map: HashMap<usize, String>,
    /// QUBO matrix
    pub qubo: Array2<f64>,
    /// Constraints
    pub constraints: Vec<ConstraintInfo>,
    /// Known optimal solution (if available)
    pub optimal_solution: Option<Solution>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Solution {
    /// Variable assignments
    pub assignments: HashMap<String, bool>,
    /// Solution energy
    pub energy: f64,
    /// Solution quality metrics
    pub quality_metrics: HashMap<String, f64>,
    /// Solution metadata
    pub metadata: HashMap<String, String>,
    /// Sampling statistics
    pub sampling_stats: Option<SamplingStats>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SamplingStats {
    /// Number of reads
    pub num_reads: usize,
    /// Annealing time
    pub annealing_time: f64,
    /// Chain break fraction
    pub chain_break_fraction: f64,
    /// Sampler used
    pub sampler: String,
}

/// Constraint information
#[derive(Debug, Clone, Serialize)]
pub struct ConstraintInfo {
    /// Constraint name
    pub name: Option<String>,
    /// Constraint type
    pub constraint_type: ConstraintType,
    /// Variables involved
    pub variables: Vec<String>,
    /// Constraint parameters
    pub parameters: HashMap<String, f64>,
    /// Penalty weight
    pub penalty: f64,
    /// Description
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub enum ConstraintType {
    /// Equality constraint
    Equality { target: f64 },
    /// Inequality constraint
    Inequality {
        bound: f64,
        direction: InequalityDirection,
    },
    /// All different constraint
    AllDifferent,
    /// Exactly one constraint
    ExactlyOne,
    /// At most one constraint
    AtMostOne,
    /// Custom constraint
    Custom { evaluator: String },
}

#[derive(Debug, Clone, Serialize)]
pub enum InequalityDirection {
    LessEqual,
    GreaterEqual,
}
