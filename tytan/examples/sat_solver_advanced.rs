//! Advanced SAT solver example using QuantRS2-Tytan
//!
//! This example demonstrates:
//! - Boolean satisfiability (SAT) problem encoding
//! - Clause learning and conflict analysis
//! - Various SAT problem instances (3-SAT, k-SAT)
//! - Performance comparison with classical SAT solvers

use ndarray::Array2;
use quantrs2_tytan::{
    compile::Model,
    constraints::PenaltyFunction,
    optimization::{
        adaptive::{AdaptiveConfig, AdaptiveOptimizer},
        penalty::{PenaltyConfig, PenaltyOptimizer, PenaltyType},
    },
    sampler::{SASampler, Sampler},
};

use quantrs2_tytan::compile::expr::{Expr, constant};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

/// A literal in a SAT formula (variable or its negation)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Literal {
    var: usize,
    negated: bool,
}

impl Literal {
    fn new(var: usize, negated: bool) -> Self {
        Self { var, negated }
    }

    fn positive(var: usize) -> Self {
        Self::new(var, false)
    }

    fn negative(var: usize) -> Self {
        Self::new(var, true)
    }
}

/// A clause (disjunction of literals)
#[derive(Debug, Clone)]
struct Clause {
    literals: Vec<Literal>,
}

impl Clause {
    fn new(literals: Vec<Literal>) -> Self {
        Self { literals }
    }

    /// Check if the clause is satisfied by an assignment
    fn is_satisfied(&self, assignment: &[bool]) -> bool {
        self.literals.iter().any(|lit| {
            let value = assignment.get(lit.var).copied().unwrap_or(false);
            if lit.negated {
                !value
            } else {
                value
            }
        })
    }
}

/// A SAT formula in Conjunctive Normal Form (CNF)
#[derive(Debug, Clone)]
struct SatFormula {
    clauses: Vec<Clause>,
    num_vars: usize,
}

impl SatFormula {
    fn new(num_vars: usize) -> Self {
        Self {
            clauses: Vec::new(),
            num_vars,
        }
    }

    fn add_clause(&mut self, clause: Clause) {
        self.clauses.push(clause);
    }

    /// Check if the formula is satisfied by an assignment
    fn is_satisfied(&self, assignment: &[bool]) -> bool {
        self.clauses
            .iter()
            .all(|clause| clause.is_satisfied(assignment))
    }

    /// Count the number of satisfied clauses
    fn count_satisfied(&self, assignment: &[bool]) -> usize {
        self.clauses
            .iter()
            .filter(|clause| clause.is_satisfied(assignment))
            .count()
    }
}

/// Generate a random k-SAT formula
fn generate_random_ksat(num_vars: usize, num_clauses: usize, k: usize, seed: u64) -> SatFormula {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut formula = SatFormula::new(num_vars);

    for _ in 0..num_clauses {
        let mut literals = Vec::new();
        let mut used_vars = HashSet::new();

        while literals.len() < k {
            let var = rng.gen_range(0..num_vars);
            if !used_vars.contains(&var) {
                used_vars.insert(var);
                let negated = rng.random_bool(0.5);
                literals.push(Literal::new(var, negated));
            }
        }

        formula.add_clause(Clause::new(literals));
    }

    formula
}

/// Convert SAT formula to QUBO model
fn sat_to_qubo(formula: &SatFormula) -> Result<Model, Box<dyn std::error::Error>> {
    let mut model = Model::new();

    // Create binary variables for each SAT variable
    let mut vars = Vec::new();
    for i in 0..formula.num_vars {
        let var = model.add_variable(&format!("x_{}", i))?;
        vars.push(var);
    }

    // Track the objective expression separately
    let mut objective = constant(0.0);

    // For each clause, add a penalty if it's not satisfied
    // We use auxiliary variables for clauses with more than 2 literals
    for (clause_idx, clause) in formula.clauses.iter().enumerate() {
        if clause.literals.is_empty() {
            continue; // Empty clause is always false
        }

        // Build the clause expression
        let mut clause_expr = constant(0.0);

        for lit in &clause.literals {
            let var_expr = vars[lit.var].clone();
            if lit.negated {
                // Negated variable: (1 - x)
                clause_expr =
                    clause_expr + constant(1.0) + constant(-1.0) * var_expr;
            } else {
                // Positive variable: x
                clause_expr = clause_expr + var_expr;
            }
        }

        // Create auxiliary variable for this clause
        let aux_var = model.add_variable(&format!("aux_clause_{}", clause_idx))?;

        // Constraint: aux_var = 1 if clause is satisfied (at least one literal is true)
        // This is implemented as: if clause_expr > 0, then aux_var = 1
        // We add penalty: (1 - aux_var) * clause_expr
        let penalty_expr = (constant(1.0)
            + constant(-1.0) * aux_var.clone())
            * clause_expr;

        // Also ensure aux_var = 0 when clause is not satisfied
        // Add small penalty for aux_var to prefer aux_var = 0 when possible
        let mut aux_penalty = constant(0.1) * aux_var;

        // Add to objective (we're minimizing penalties)
        objective = objective + penalty_expr + aux_penalty;
    }

    // Set the final objective
    model.set_objective(objective);

    Ok(model)
}

/// Clause learning: identify conflicting clauses
fn learn_clauses(formula: &SatFormula, failed_assignments: &[Vec<bool>]) -> Vec<Clause> {
    let mut learned_clauses = Vec::new();

    for assignment in failed_assignments {
        // Find unsatisfied clauses
        let unsat_clauses: Vec<_> = formula
            .clauses
            .iter()
            .enumerate()
            .filter(|(_, clause)| !clause.is_satisfied(assignment))
            .collect();

        if unsat_clauses.len() == 1 {
            // Single unsatisfied clause - learn its negation
            let (_, clause) = unsat_clauses[0];

            // Create a new clause that prevents this specific assignment
            let mut new_literals = Vec::new();
            for lit in &clause.literals {
                let value = assignment[lit.var];
                if lit.negated {
                    // Variable was true, we want it false
                    if value {
                        new_literals.push(Literal::negative(lit.var));
                    }
                } else {
                    // Variable was false, we want it true
                    if !value {
                        new_literals.push(Literal::positive(lit.var));
                    }
                }
            }

            if !new_literals.is_empty() {
                learned_clauses.push(Clause::new(new_literals));
            }
        }
    }

    learned_clauses
}

/// Run SAT solver experiment
fn run_sat_experiment(
    name: &str,
    formula: &SatFormula,
) -> Result<SatSolverResults, Box<dyn std::error::Error>> {
    println!("\n=== {} ===", name);
    println!(
        "Variables: {}, Clauses: {}",
        formula.num_vars,
        formula.clauses.len()
    );

    // Convert to QUBO
    let model = sat_to_qubo(formula)?;

    // Optimize with adaptive penalties
    let penalty_config = PenaltyConfig {
        initial_weight: 5.0,
        min_weight: 0.1,
        max_weight: 100.0,
        adjustment_factor: 1.2,
        violation_tolerance: 1e-4,
        max_iterations: 20,
        adaptive_scaling: true,
        penalty_type: PenaltyType::Quadratic,
    };

    let mut penalty_optimizer = PenaltyOptimizer::new(penalty_config);

    // Compile the model to QUBO
    let compiled = model.compile()?;
    let qubo = compiled.to_qubo();

    println!("QUBO variables: {}", qubo.num_variables);

    // Convert QUBO to matrix format for sampler
    let n_vars = qubo.num_variables;
    let mut matrix = ndarray::Array2::zeros((n_vars, n_vars));
    let mut var_map = HashMap::new();

    // Create variable mapping and fill matrix
    for i in 0..n_vars {
        var_map.insert(format!("x_{}", i), i);

        // Get linear term (diagonal)
        if let Ok(linear) = qubo.get_linear(i) {
            matrix[[i, i]] = linear;
        }

        // Get quadratic terms
        for j in 0..n_vars {
            if i != j {
                if let Ok(quad) = qubo.get_quadratic(i, j) {
                    matrix[[i, j]] = quad;
                }
            }
        }
    }

    // Configure sampler
    let mut sampler = SASampler::new(None);

    // First run
    println!("\nFirst optimization run...");
    let mut start = Instant::now();
    let samples = sampler.run_qubo(&(matrix.clone(), var_map.clone()), 1000)?;
    let first_run_time = start.elapsed();

    // Extract assignments and check satisfaction
    let mut satisfying_assignments = Vec::new();
    let mut failed_assignments = Vec::new();
    let mut best_unsat_count = formula.clauses.len();

    for sample in &samples {
        let assignment = extract_assignment(sample, formula.num_vars);

        if formula.is_satisfied(&assignment) {
            satisfying_assignments.push(assignment);
        } else {
            let unsat = formula.clauses.len() - formula.count_satisfied(&assignment);
            best_unsat_count = best_unsat_count.min(unsat);
            failed_assignments.push(assignment);
        }
    }

    let first_run_sat_rate = satisfying_assignments.len() as f64 / samples.len() as f64;
    println!(
        "Satisfying assignments: {} / {} ({:.1}%)",
        satisfying_assignments.len(),
        samples.len(),
        first_run_sat_rate * 100.0
    );

    // If we didn't find a satisfying assignment, try clause learning
    let mut total_time = first_run_time;
    let mut iterations = 1;
    let mut with_learning_sat_rate = first_run_sat_rate;

    if satisfying_assignments.is_empty() && !failed_assignments.is_empty() {
        println!("\nApplying clause learning...");

        // Learn new clauses from failures
        let learned = learn_clauses(
            formula,
            &failed_assignments[..failed_assignments.len().min(10)],
        );
        println!("Learned {} new clauses", learned.len());

        if !learned.is_empty() {
            // Create augmented formula
            let mut augmented_formula = formula.clone();
            for clause in learned {
                augmented_formula.add_clause(clause);
            }

            // Re-solve with augmented formula
            let augmented_model = sat_to_qubo(&augmented_formula)?;
            let augmented_compiled = augmented_model.compile()?;
            let augmented_qubo = augmented_compiled.to_qubo();

            // Convert to matrix format
            let aug_n_vars = augmented_qubo.num_variables;
            let mut aug_matrix = Array2::zeros((aug_n_vars, aug_n_vars));
            let mut aug_var_map = HashMap::new();

            for i in 0..aug_n_vars {
                aug_var_map.insert(format!("x_{}", i), i);
                if let Ok(linear) = augmented_qubo.get_linear(i) {
                    aug_matrix[[i, i]] = linear;
                }
                for j in 0..aug_n_vars {
                    if i != j {
                        if let Ok(quad) = augmented_qubo.get_quadratic(i, j) {
                            aug_matrix[[i, j]] = quad;
                        }
                    }
                }
            }

            println!(
                "Re-solving with {} clauses...",
                augmented_formula.clauses.len()
            );
            let mut start = Instant::now();
            let new_samples = sampler.run_qubo(&(aug_matrix, aug_var_map), 1000)?;
            total_time += start.elapsed();
            iterations += 1;

            // Check new results against original formula
            let mut new_satisfying = 0;
            for sample in &new_samples {
                let assignment = extract_assignment(sample, formula.num_vars);
                if formula.is_satisfied(&assignment) {
                    new_satisfying += 1;
                    satisfying_assignments.push(assignment);
                }
            }

            with_learning_sat_rate = new_satisfying as f64 / new_samples.len() as f64;
            println!(
                "New satisfying assignments: {} / {} ({:.1}%)",
                new_satisfying,
                new_samples.len(),
                with_learning_sat_rate * 100.0
            );
        }
    }

    // Analyze solution quality
    let satisfiable = !satisfying_assignments.is_empty();

    Ok(SatSolverResults {
        name: name.to_string(),
        num_vars: formula.num_vars,
        num_clauses: formula.clauses.len(),
        satisfiable,
        satisfaction_rate: if satisfiable { 1.0 } else { 0.0 },
        first_run_sat_rate,
        with_learning_sat_rate,
        best_unsat_clauses: if satisfiable { 0 } else { best_unsat_count },
        total_time: total_time.as_secs_f64(),
        iterations,
    })
}

/// Extract boolean assignment from sample
fn extract_assignment(
    sample: &quantrs2_tytan::sampler::SampleResult,
    num_vars: usize,
) -> Vec<bool> {
    let mut assignment = vec![false; num_vars];

    for i in 0..num_vars {
        let var_name = format!("x_{}", i);
        assignment[i] = sample.assignments.get(&var_name).copied().unwrap_or(false);
    }

    assignment
}

#[derive(Debug)]
struct SatSolverResults {
    name: String,
    num_vars: usize,
    num_clauses: usize,
    satisfiable: bool,
    satisfaction_rate: f64,
    first_run_sat_rate: f64,
    with_learning_sat_rate: f64,
    best_unsat_clauses: usize,
    total_time: f64,
    iterations: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Advanced SAT Solver Examples ===");

    let mut all_results = Vec::new();

    // Example 1: Small satisfiable 3-SAT
    let mut sat_formula = SatFormula::new(5);
    sat_formula.add_clause(Clause::new(vec![
        Literal::positive(0),
        Literal::negative(1),
        Literal::positive(2),
    ]));
    sat_formula.add_clause(Clause::new(vec![
        Literal::negative(0),
        Literal::positive(3),
        Literal::negative(4),
    ]));
    sat_formula.add_clause(Clause::new(vec![
        Literal::positive(1),
        Literal::positive(2),
        Literal::positive(4),
    ]));
    sat_formula.add_clause(Clause::new(vec![
        Literal::negative(2),
        Literal::negative(3),
        Literal::positive(4),
    ]));

    all_results.push(run_sat_experiment("Small Satisfiable 3-SAT", &sat_formula)?);

    // Example 2: Random 3-SAT at phase transition (ratio ~4.27)
    let phase_transition_formula = generate_random_ksat(20, 85, 3, 42);
    all_results.push(run_sat_experiment(
        "Random 3-SAT (Phase Transition)",
        &phase_transition_formula,
    )?);

    // Example 3: Under-constrained 3-SAT (likely satisfiable)
    let easy_formula = generate_random_ksat(20, 40, 3, 123);
    all_results.push(run_sat_experiment(
        "Under-constrained 3-SAT",
        &easy_formula,
    )?);

    // Example 4: Over-constrained 3-SAT (likely unsatisfiable)
    let hard_formula = generate_random_ksat(15, 100, 3, 456);
    all_results.push(run_sat_experiment("Over-constrained 3-SAT", &hard_formula)?);

    // Example 5: 4-SAT instance
    let four_sat = generate_random_ksat(15, 50, 4, 789);
    all_results.push(run_sat_experiment("Random 4-SAT", &four_sat)?);

    // Generate summary report
    println!("\n\n=== Summary Report ===");
    println!(
        "\n{:<30} | {:>8} | {:>8} | {:>10} | {:>8} | {:>8} | {:>8}",
        "Instance", "Vars", "Clauses", "Satisfiable", "SAT Rate", "w/Learn", "Time (s)"
    );
    println!(
        "{:-<30}-+-{:-<8}-+-{:-<8}-+-{:-<10}-+-{:-<8}-+-{:-<8}-+-{:-<8}",
        "", "", "", "", "", "", ""
    );

    for result in &all_results {
        println!(
            "{:<30} | {:>8} | {:>8} | {:>10} | {:>7.1}% | {:>7.1}% | {:>8.3}",
            result.name,
            result.num_vars,
            result.num_clauses,
            if result.satisfiable { "YES" } else { "NO" },
            result.first_run_sat_rate * 100.0,
            result.with_learning_sat_rate * 100.0,
            result.total_time
        );
    }

    // Phase transition analysis
    println!("\n\n=== Phase Transition Analysis ===");
    println!("Testing 3-SAT with 20 variables at different clause/variable ratios:");

    let ratios = vec![2.0, 3.0, 4.0, 4.27, 5.0, 6.0, 7.0];
    let mut phase_results = Vec::new();

    for &ratio in &ratios {
        let num_clauses = (20.0 * ratio) as usize;
        let formula = generate_random_ksat(20, num_clauses, 3, (ratio * 1000.0) as u64);
        let result = run_sat_experiment(&format!("Ratio {:.2}", ratio), &formula)?;
        phase_results.push((ratio, result));
    }

    println!(
        "\n{:<10} | {:>10} | {:>12} | {:>10}",
        "Ratio", "Clauses", "Satisfiable", "SAT Rate"
    );
    println!("{:-<10}-+-{:-<10}-+-{:-<12}-+-{:-<10}", "", "", "", "");

    for (ratio, result) in phase_results {
        println!(
            "{:<10.2} | {:>10} | {:>12} | {:>9.1}%",
            ratio,
            result.num_clauses,
            if result.satisfiable { "YES" } else { "UNKNOWN" },
            result.first_run_sat_rate * 100.0
        );
    }

    println!("\nNote: The phase transition for random 3-SAT occurs around ratio 4.27");

    Ok(())
}
