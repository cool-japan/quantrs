//! Constraint programming enhancements for optimization problems.
//!
//! This module provides various constraint types and propagation algorithms
//! to enhance the expressiveness and efficiency of optimization problems.

use ndarray::Array2;
use std::collections::{HashMap, HashSet};

#[cfg(feature = "dwave")]
use crate::symbol::Expression;

/// Global constraint types
#[derive(Debug, Clone)]
pub enum GlobalConstraint {
    /// All variables must have different values
    AllDifferent { variables: Vec<String> },
    /// Cumulative resource constraint
    Cumulative { tasks: Vec<Task>, capacity: i32 },
    /// Global cardinality constraint
    GlobalCardinality {
        variables: Vec<String>,
        values: Vec<i32>,
        min_occurrences: Vec<i32>,
        max_occurrences: Vec<i32>,
    },
    /// Regular constraint (finite automaton)
    Regular {
        variables: Vec<String>,
        automaton: FiniteAutomaton,
    },
    /// Element constraint
    Element {
        index_var: String,
        array: Vec<i32>,
        value_var: String,
    },
    /// Table constraint
    Table {
        variables: Vec<String>,
        tuples: Vec<Vec<i32>>,
        positive: bool,
    },
    /// Circuit constraint (Hamiltonian circuit)
    Circuit { variables: Vec<String> },
    /// Bin packing constraint
    BinPacking { items: Vec<Item>, bins: Vec<Bin> },
}

/// Task for cumulative constraint
#[derive(Debug, Clone)]
pub struct Task {
    pub start_var: String,
    pub duration: i32,
    pub resource_usage: i32,
}

/// Item for bin packing
#[derive(Debug, Clone)]
pub struct Item {
    pub size: i32,
    pub bin_var: String,
}

/// Bin for bin packing
#[derive(Debug, Clone)]
pub struct Bin {
    pub capacity: i32,
}

/// Finite automaton for regular constraint
#[derive(Debug, Clone)]
pub struct FiniteAutomaton {
    pub states: Vec<i32>,
    pub initial_state: i32,
    pub final_states: HashSet<i32>,
    pub transitions: HashMap<(i32, i32), i32>, // (state, value) -> next_state
}

/// Soft constraint with penalty function
#[derive(Debug, Clone)]
pub struct SoftConstraint {
    /// Constraint expression
    pub constraint: ConstraintExpression,
    /// Penalty function
    pub penalty: PenaltyFunction,
    /// Priority level
    pub priority: i32,
}

/// Constraint expression
#[derive(Debug, Clone)]
pub enum ConstraintExpression {
    /// Linear inequality: sum(ai * xi) <= b
    LinearInequality {
        coefficients: Vec<(String, f64)>,
        bound: f64,
    },
    /// Logical expression
    Logical(LogicalExpression),
    /// Custom expression
    #[cfg(feature = "dwave")]
    Custom(Expression),
}

/// Logical expression for constraints
#[derive(Debug, Clone)]
pub enum LogicalExpression {
    /// Variable reference
    Var(String),
    /// Negation
    Not(Box<LogicalExpression>),
    /// Conjunction
    And(Vec<LogicalExpression>),
    /// Disjunction
    Or(Vec<LogicalExpression>),
    /// Implication
    Implies(Box<LogicalExpression>, Box<LogicalExpression>),
    /// Equivalence
    Iff(Box<LogicalExpression>, Box<LogicalExpression>),
}

/// Penalty function types
#[derive(Debug, Clone)]
pub enum PenaltyFunction {
    /// Linear penalty: penalty = weight * violation
    Linear { weight: f64 },
    /// Quadratic penalty: penalty = weight * violation^2
    Quadratic { weight: f64 },
    /// Exponential penalty: penalty = weight * exp(violation) - weight
    Exponential { weight: f64 },
    /// Step function: penalty = weight if violated, 0 otherwise
    Step { weight: f64 },
    /// Piecewise linear
    PiecewiseLinear { points: Vec<(f64, f64)> },
}

/// Constraint propagation algorithm
pub trait ConstraintPropagator {
    /// Propagate constraints and reduce domains
    fn propagate(&mut self, domains: &mut HashMap<String, Domain>) -> Result<bool, String>;

    /// Check if constraint is satisfied
    fn is_satisfied(&self, assignment: &HashMap<String, i32>) -> bool;

    /// Get variables involved in constraint
    fn variables(&self) -> Vec<String>;
}

/// Variable domain
#[derive(Debug, Clone)]
pub struct Domain {
    /// Possible values
    pub values: HashSet<i32>,
    /// Min value
    pub min: i32,
    /// Max value
    pub max: i32,
}

impl Domain {
    /// Create new domain
    pub fn new(min: i32, max: i32) -> Self {
        let values: HashSet<i32> = (min..=max).collect();
        Self { values, min, max }
    }

    /// Create from specific values
    pub fn from_values(values: Vec<i32>) -> Self {
        let min = values.iter().min().cloned().unwrap_or(0);
        let max = values.iter().max().cloned().unwrap_or(0);
        Self {
            values: values.into_iter().collect(),
            min,
            max,
        }
    }

    /// Remove value from domain
    pub fn remove(&mut self, value: i32) -> bool {
        let removed = self.values.remove(&value);
        if removed {
            self.update_bounds();
        }
        removed
    }

    /// Keep only specified values
    pub fn intersect(&mut self, values: &HashSet<i32>) {
        self.values = self.values.intersection(values).cloned().collect();
        self.update_bounds();
    }

    /// Update min/max bounds
    fn update_bounds(&mut self) {
        self.min = self.values.iter().min().cloned().unwrap_or(self.min);
        self.max = self.values.iter().max().cloned().unwrap_or(self.max);
    }

    /// Check if domain is empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get size of domain
    pub fn size(&self) -> usize {
        self.values.len()
    }
}

/// AllDifferent constraint propagator
pub struct AllDifferentPropagator {
    variables: Vec<String>,
}

impl AllDifferentPropagator {
    pub fn new(variables: Vec<String>) -> Self {
        Self { variables }
    }
}

impl ConstraintPropagator for AllDifferentPropagator {
    fn propagate(&mut self, domains: &mut HashMap<String, Domain>) -> Result<bool, String> {
        let mut changed = false;

        // Find singleton domains
        let mut assigned_values = HashSet::new();
        for var in &self.variables {
            if let Some(domain) = domains.get(var) {
                if domain.size() == 1 {
                    assigned_values.insert(*domain.values.iter().next().unwrap());
                }
            }
        }

        // Remove assigned values from other domains
        for var in &self.variables {
            if let Some(domain) = domains.get_mut(var) {
                if domain.size() > 1 {
                    for &value in &assigned_values {
                        if domain.remove(value) {
                            changed = true;
                        }
                    }

                    if domain.is_empty() {
                        return Err(format!("Domain of {} became empty", var));
                    }
                }
            }
        }

        // Hall's theorem (matching theory) for stronger propagation
        if self.variables.len() <= 10 {
            changed |= self.hall_propagation(domains)?;
        }

        Ok(changed)
    }

    fn is_satisfied(&self, assignment: &HashMap<String, i32>) -> bool {
        let mut seen = HashSet::new();
        for var in &self.variables {
            if let Some(&value) = assignment.get(var) {
                if !seen.insert(value) {
                    return false;
                }
            }
        }
        true
    }

    fn variables(&self) -> Vec<String> {
        self.variables.clone()
    }
}

impl AllDifferentPropagator {
    /// Hall's theorem propagation
    fn hall_propagation(&self, domains: &mut HashMap<String, Domain>) -> Result<bool, String> {
        let n = self.variables.len();
        let mut changed = false;

        // Check all subsets (exponential, so limited to small constraints)
        for subset_bits in 1..(1 << n) {
            let mut subset_vars = Vec::new();
            let mut union_values: HashSet<i32> = HashSet::new();

            for (i, var) in self.variables.iter().enumerate() {
                if (subset_bits >> i) & 1 == 1 {
                    subset_vars.push(var);
                    if let Some(domain) = domains.get(var) {
                        union_values.extend(&domain.values);
                    }
                }
            }

            // Hall's condition: |vars| <= |possible values|
            if subset_vars.len() > union_values.len() {
                return Err("Unsatisfiable: Hall's condition violated".to_string());
            }

            // If |vars| == |values|, these values can't appear elsewhere
            if subset_vars.len() == union_values.len() {
                for var in &self.variables {
                    if !subset_vars.contains(&var) {
                        if let Some(domain) = domains.get_mut(var) {
                            let old_size = domain.size();
                            domain.values.retain(|v| !union_values.contains(v));
                            domain.update_bounds();

                            if domain.size() < old_size {
                                changed = true;
                            }

                            if domain.is_empty() {
                                return Err(format!("Domain of {} became empty", var));
                            }
                        }
                    }
                }
            }
        }

        Ok(changed)
    }
}

/// Cumulative constraint propagator
pub struct CumulativePropagator {
    tasks: Vec<Task>,
    capacity: i32,
}

impl CumulativePropagator {
    pub fn new(tasks: Vec<Task>, capacity: i32) -> Self {
        Self { tasks, capacity }
    }

    /// Time-tabling propagation
    fn time_tabling(&self, domains: &HashMap<String, Domain>) -> Result<(), String> {
        // Find time bounds
        let mut min_time = i32::MAX;
        let mut max_time = i32::MIN;

        for task in &self.tasks {
            if let Some(domain) = domains.get(&task.start_var) {
                min_time = min_time.min(domain.min);
                max_time = max_time.max(domain.max + task.duration);
            }
        }

        // Check resource usage at each time point
        for t in min_time..max_time {
            let mut min_usage = 0;

            for task in &self.tasks {
                if let Some(domain) = domains.get(&task.start_var) {
                    // Task must be running at time t
                    if domain.max < t && t < domain.min + task.duration {
                        min_usage += task.resource_usage;
                    }
                }
            }

            if min_usage > self.capacity {
                return Err(format!("Resource overload at time {}", t));
            }
        }

        Ok(())
    }
}

impl ConstraintPropagator for CumulativePropagator {
    fn propagate(&mut self, domains: &mut HashMap<String, Domain>) -> Result<bool, String> {
        // Time-tabling consistency check
        self.time_tabling(domains)?;

        // Edge-finding would go here for stronger propagation

        Ok(false) // Simplified - no domain reduction implemented
    }

    fn is_satisfied(&self, assignment: &HashMap<String, i32>) -> bool {
        // Build resource profile
        let mut events = Vec::new();

        for task in &self.tasks {
            if let Some(&start) = assignment.get(&task.start_var) {
                events.push((start, task.resource_usage));
                events.push((start + task.duration, -task.resource_usage));
            } else {
                return false; // Incomplete assignment
            }
        }

        events.sort_by_key(|&(time, _)| time);

        // Check capacity constraint
        let mut current_usage = 0;
        for (_, delta) in events {
            current_usage += delta;
            if current_usage > self.capacity {
                return false;
            }
        }

        true
    }

    fn variables(&self) -> Vec<String> {
        self.tasks.iter().map(|t| t.start_var.clone()).collect()
    }
}

/// Symmetry breaking constraints
#[derive(Debug, Clone)]
pub enum SymmetryBreaking {
    /// Lexicographic ordering
    LexOrdering { variable_groups: Vec<Vec<String>> },
    /// Value precedence
    ValuePrecedence {
        values: Vec<i32>,
        variables: Vec<String>,
    },
    /// Orbit fixing
    OrbitFixing {
        symmetry_group: SymmetryGroup,
        representative: Vec<(String, i32)>,
    },
}

#[derive(Debug, Clone)]
pub enum SymmetryGroup {
    /// Symmetric group (all permutations)
    Symmetric(usize),
    /// Cyclic group
    Cyclic(usize),
    /// Direct product
    Product(Box<SymmetryGroup>, Box<SymmetryGroup>),
}

/// Constraint library for common patterns
pub struct ConstraintLibrary;

impl ConstraintLibrary {
    /// N-Queens constraint
    pub fn n_queens(n: usize) -> Vec<GlobalConstraint> {
        let vars: Vec<String> = (0..n).map(|i| format!("queen_{}", i)).collect();

        let constraints = vec![
            // All queens in different columns
            GlobalConstraint::AllDifferent {
                variables: vars.clone(),
            },
        ];

        // No two queens on same diagonal
        // This would need custom constraints in practice

        constraints
    }

    /// Graph coloring constraint
    pub fn graph_coloring(edges: &[(usize, usize)], _num_colors: usize) -> Vec<GlobalConstraint> {
        let mut constraints = Vec::new();

        for &(i, j) in edges {
            // Adjacent vertices must have different colors
            constraints.push(GlobalConstraint::AllDifferent {
                variables: vec![format!("color_{}", i), format!("color_{}", j)],
            });
        }

        constraints
    }

    /// Sudoku constraints
    pub fn sudoku() -> Vec<GlobalConstraint> {
        let mut constraints = Vec::new();

        // Row constraints
        for row in 0..9 {
            let vars: Vec<String> = (0..9).map(|col| format!("cell_{}_{}", row, col)).collect();
            constraints.push(GlobalConstraint::AllDifferent { variables: vars });
        }

        // Column constraints
        for col in 0..9 {
            let vars: Vec<String> = (0..9).map(|row| format!("cell_{}_{}", row, col)).collect();
            constraints.push(GlobalConstraint::AllDifferent { variables: vars });
        }

        // Box constraints
        for box_row in 0..3 {
            for box_col in 0..3 {
                let mut vars = Vec::new();
                for r in 0..3 {
                    for c in 0..3 {
                        vars.push(format!("cell_{}_{}", box_row * 3 + r, box_col * 3 + c));
                    }
                }
                constraints.push(GlobalConstraint::AllDifferent { variables: vars });
            }
        }

        constraints
    }
}

/// Convert constraints to penalty terms
pub fn constraints_to_penalties(
    constraints: &[SoftConstraint],
    variables: &HashMap<String, usize>,
) -> Array2<f64> {
    let n = variables.len();
    let mut penalty_matrix = Array2::zeros((n, n));

    for constraint in constraints {
        match &constraint.constraint {
            ConstraintExpression::LinearInequality {
                coefficients,
                bound: _,
            } => {
                // Convert to quadratic penalty
                // (sum(ai * xi) - b)^2 if violated

                // This is a simplified version
                // Real implementation would handle inequality properly
                for (var1, coeff1) in coefficients {
                    if let Some(&idx1) = variables.get(var1) {
                        // Linear term
                        penalty_matrix[[idx1, idx1]] += coeff1 * coeff1;

                        // Quadratic terms
                        for (var2, coeff2) in coefficients {
                            if var1 != var2 {
                                if let Some(&idx2) = variables.get(var2) {
                                    penalty_matrix[[idx1, idx2]] += coeff1 * coeff2;
                                }
                            }
                        }
                    }
                }
            }
            _ => {
                // Other constraint types would need specific handling
            }
        }
    }

    penalty_matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_operations() {
        let mut domain = Domain::new(1, 5);
        assert_eq!(domain.size(), 5);

        domain.remove(3);
        assert_eq!(domain.size(), 4);
        assert!(!domain.values.contains(&3));

        let mut keep = vec![1, 2, 5].into_iter().collect();
        domain.intersect(&keep);
        assert_eq!(domain.size(), 3);
        assert_eq!(domain.min, 1);
        assert_eq!(domain.max, 5);
    }

    #[test]
    fn test_alldifferent_propagation() {
        let mut propagator =
            AllDifferentPropagator::new(vec!["x".to_string(), "y".to_string(), "z".to_string()]);

        let mut domains = HashMap::new();
        domains.insert("x".to_string(), Domain::from_values(vec![1]));
        domains.insert("y".to_string(), Domain::from_values(vec![1, 2, 3]));
        domains.insert("z".to_string(), Domain::from_values(vec![1, 2, 3]));

        let mut changed = propagator.propagate(&mut domains).unwrap();
        assert!(changed);

        // Value 1 should be removed from y and z
        assert!(!domains["y"].values.contains(&1));
        assert!(!domains["z"].values.contains(&1));
    }

    #[test]
    fn test_constraint_library() {
        let queens = ConstraintLibrary::n_queens(8);
        assert!(!queens.is_empty());

        let mut edges = vec![(0, 1), (1, 2), (2, 0)];
        let coloring = ConstraintLibrary::graph_coloring(&edges, 3);
        assert_eq!(coloring.len(), 3);
    }
}
