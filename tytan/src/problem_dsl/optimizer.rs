//! Optimization hints and strategies for the problem DSL.

use super::ast::Value;
use std::collections::HashMap;

/// Optimization hint
#[derive(Debug, Clone)]
pub enum OptimizationHint {
    /// Variable ordering hint
    VariableOrder(Vec<String>),
    /// Symmetry breaking
    Symmetry(SymmetryType),
    /// Decomposition hint
    Decomposition(DecompositionHint),
    /// Solver preference
    SolverPreference(String),
    /// Custom hint
    Custom { name: String, value: String },
}

#[derive(Debug, Clone)]
pub enum SymmetryType {
    /// Permutation symmetry
    Permutation(Vec<String>),
    /// Reflection symmetry
    Reflection { axis: String },
    /// Rotation symmetry
    Rotation { order: usize },
}

#[derive(Debug, Clone)]
pub struct DecompositionHint {
    pub method: String,
    pub parameters: HashMap<String, Value>,
}
