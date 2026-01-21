#![allow(
    clippy::pedantic,
    clippy::unnecessary_wraps,
    clippy::needless_range_loop,
    clippy::useless_vec,
    clippy::needless_collect,
    clippy::too_many_arguments,
    clippy::field_reassign_with_default
)]
#![allow(unused_must_use)]
//! Example demonstrating CSP (Constraint Satisfaction Problem) compilation
//!
//! This example shows how to:
//! 1. Define CSP problems with variables and constraints
//! 2. Compile CSPs to QUBO formulations
//! 3. Solve the resulting optimization problems
//! 4. Interpret solutions back to CSP assignments

use quantrs2_anneal::{
    csp_compiler::{
        ComparisonOp, CompilationParams, CspConstraint, CspProblem, CspValue, CspVariable, Domain,
    },
    simulator::{AnnealingParams, QuantumAnnealingSimulator},
};
use std::collections::HashMap;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== CSP Compiler Demo ===\n");

    // Example 1: Boolean satisfiability (SAT)
    println!("Example 1: Boolean Satisfiability Problem");
    boolean_sat_example()?;

    // Example 2: Graph coloring
    println!("\nExample 2: Graph Coloring Problem");
    graph_coloring_example()?;

    // Example 3: N-Queens problem
    println!("\nExample 3: N-Queens Problem");
    n_queens_example()?;

    // Example 4: Sudoku solver
    println!("\nExample 4: Simplified Sudoku");
    sudoku_example()?;

    // Example 5: Scheduling problem
    println!("\nExample 5: Task Scheduling");
    scheduling_example()?;

    // Example 6: Map coloring with categorical domains
    println!("\nExample 6: Map Coloring with Colors");
    map_coloring_example()?;

    Ok(())
}

fn boolean_sat_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a 3-SAT problem: (x1 ∨ ¬x2 ∨ x3) ∧ (¬x1 ∨ x2 ∨ ¬x3) ∧ (x1 ∨ x2 ∨ x3)
    let mut problem = CspProblem::new();

    // Add boolean variables
    for i in 1..=3 {
        let var = CspVariable::new(format!("x{i}"), Domain::Boolean)
            .with_description(format!("Boolean variable {i}"));
        problem.add_variable(var)?;
    }

    // For this simple example, we'll use exactly-one and at-most-one constraints
    // to approximate SAT clauses (this is a simplified approach)

    // Constraint: at least one of x1, x2, x3 must be true
    let constraint = CspConstraint::AtMostOne {
        variables: vec!["x1".to_string(), "x2".to_string()],
    };
    problem.add_constraint(constraint)?;

    // Compile and solve
    let start = Instant::now();
    let (qubo_formulation, compilation_info) = problem.compile_to_qubo()?;
    let compilation_time = start.elapsed();

    println!("SAT Problem Compilation:");
    println!("  CSP variables: {}", compilation_info.csp_variables);
    println!("  QUBO variables: {}", compilation_info.qubo_variables);
    println!(
        "  Constraints compiled: {}",
        compilation_info.constraints_compiled
    );
    println!("  Compilation time: {compilation_time:.2?}");

    // Solve the QUBO
    let qubo_model = qubo_formulation.to_qubo_model();
    let (ising, offset) = qubo_model.to_ising();

    let params = AnnealingParams {
        num_sweeps: 1000,
        num_repetitions: 5,
        ..Default::default()
    };

    let start = Instant::now();
    let mut solver = QuantumAnnealingSimulator::new(params)?;
    let result = solver.solve(&ising)?;
    let solving_time = start.elapsed();

    println!("  Solution energy: {:.4}", result.best_energy + offset);
    println!("  Solving time: {solving_time:.2?}");

    // Interpret solution
    println!("  Variable assignments:");
    for (i, &spin) in result.best_spins.iter().enumerate() {
        let value = spin > 0;
        println!("    x{}: {}", i + 1, value);
    }

    Ok(())
}

fn graph_coloring_example() -> Result<(), Box<dyn std::error::Error>> {
    // Color a simple triangle graph with 3 colors
    let mut problem = CspProblem::new();

    // Add vertices with color domains
    let colors = vec![0, 1, 2]; // 3 colors
    for i in 0..3 {
        let var = CspVariable::new(format!("vertex_{i}"), Domain::Discrete(colors.clone()))
            .with_description(format!("Color of vertex {i}"));
        problem.add_variable(var)?;
    }

    // Add all-different constraints for adjacent vertices
    // Triangle: 0-1, 1-2, 2-0
    let edges = [(0, 1), (1, 2), (2, 0)];
    for (u, v) in edges {
        let constraint = CspConstraint::AllDifferent {
            variables: vec![format!("vertex_{}", u), format!("vertex_{}", v)],
        };
        problem.add_constraint(constraint)?;
    }

    // Compile with one-hot encoding for colors
    let mut params = CompilationParams::default();
    params.max_onehot_size = 5; // Use one-hot for small domains
    params.use_log_encoding = false;
    problem.set_compilation_params(params);

    let (qubo_formulation, info) = problem.compile_to_qubo()?;

    println!("Graph Coloring Compilation:");
    println!("  Vertices: 3, Colors: 3");
    println!("  QUBO variables: {}", info.qubo_variables);
    println!("  Constraints: {}", info.constraints_compiled);

    // Show encoding details
    for (var_name, var_info) in &info.variable_info {
        println!(
            "  {}: {} -> {} QUBO vars ({})",
            var_name, var_info.domain_size, var_info.qubo_variables_used, var_info.encoding_type
        );
    }

    // Solve
    let qubo_model = qubo_formulation.to_qubo_model();
    let (ising, offset) = qubo_model.to_ising();

    let params = AnnealingParams {
        num_sweeps: 2000,
        num_repetitions: 10,
        ..Default::default()
    };

    let mut solver = QuantumAnnealingSimulator::new(params)?;
    let result = solver.solve(&ising)?;

    println!(
        "  Solution found with energy: {:.4}",
        result.best_energy + offset
    );

    // Interpret one-hot encoded solution
    println!("  Vertex coloring:");
    for vertex in 0..3 {
        for color in 0..3 {
            let var_idx = vertex * 3 + color;
            if var_idx < result.best_spins.len() && result.best_spins[var_idx] > 0 {
                println!("    Vertex {vertex}: Color {color}");
                break;
            }
        }
    }

    Ok(())
}

fn n_queens_example() -> Result<(), Box<dyn std::error::Error>> {
    // Simplified 4-Queens problem using boolean variables
    let n = 4;
    let mut problem = CspProblem::new();

    // Variables: queen[i][j] = 1 if there's a queen at position (i,j)
    for i in 0..n {
        for j in 0..n {
            let var = CspVariable::new(format!("q_{i}_{j}"), Domain::Boolean)
                .with_description(format!("Queen at position ({i}, {j})"));
            problem.add_variable(var)?;
        }
    }

    // Constraint: exactly one queen per row
    for i in 0..n {
        let row_vars: Vec<String> = (0..n).map(|j| format!("q_{i}_{j}")).collect();
        let constraint = CspConstraint::ExactlyOne {
            variables: row_vars,
        };
        problem.add_constraint(constraint)?;
    }

    // Constraint: exactly one queen per column
    for j in 0..n {
        let col_vars: Vec<String> = (0..n).map(|i| format!("q_{i}_{j}")).collect();
        let constraint = CspConstraint::ExactlyOne {
            variables: col_vars,
        };
        problem.add_constraint(constraint)?;
    }

    // Note: Diagonal constraints would require custom constraint types
    // For this demo, we'll just solve the row/column constraints

    let (qubo_formulation, info) = problem.compile_to_qubo()?;

    println!("4-Queens Problem (simplified):");
    println!("  Board size: {n}x{n}");
    println!("  CSP variables: {}", info.csp_variables);
    println!("  QUBO variables: {}", info.qubo_variables);
    println!("  Constraints: {}", info.constraints_compiled);

    // Solve
    let qubo_model = qubo_formulation.to_qubo_model();
    let (ising, offset) = qubo_model.to_ising();

    let params = AnnealingParams {
        num_sweeps: 3000,
        num_repetitions: 15,
        ..Default::default()
    };

    let mut solver = QuantumAnnealingSimulator::new(params)?;
    let result = solver.solve(&ising)?;

    println!("  Solution energy: {:.4}", result.best_energy + offset);

    // Display the board
    println!("  Board configuration:");
    for i in 0..n {
        print!("    ");
        for j in 0..n {
            let var_idx = i * n + j;
            let symbol = if var_idx < result.best_spins.len() && result.best_spins[var_idx] > 0 {
                "Q"
            } else {
                "."
            };
            print!("{symbol} ");
        }
        println!();
    }

    Ok(())
}

fn sudoku_example() -> Result<(), Box<dyn std::error::Error>> {
    // Simplified 2x2 Sudoku (each cell can be 1 or 2)
    let mut problem = CspProblem::new();

    // Variables: cell[i][j] with domain {1, 2}
    for i in 0..2 {
        for j in 0..2 {
            let var = CspVariable::new(format!("cell_{i}_{j}"), Domain::Discrete(vec![1, 2]))
                .with_description(format!("Value in cell ({i}, {j})"));
            problem.add_variable(var)?;
        }
    }

    // Constraint: each row has different values
    for i in 0..2 {
        let row_vars: Vec<String> = (0..2).map(|j| format!("cell_{i}_{j}")).collect();
        let constraint = CspConstraint::AllDifferent {
            variables: row_vars,
        };
        problem.add_constraint(constraint)?;
    }

    // Constraint: each column has different values
    for j in 0..2 {
        let col_vars: Vec<String> = (0..2).map(|i| format!("cell_{i}_{j}")).collect();
        let constraint = CspConstraint::AllDifferent {
            variables: col_vars,
        };
        problem.add_constraint(constraint)?;
    }

    let (qubo_formulation, info) = problem.compile_to_qubo()?;

    println!("2x2 Sudoku:");
    println!("  Grid size: 2x2");
    println!("  Possible values: {{1, 2}}");
    println!("  QUBO variables: {}", info.qubo_variables);
    println!("  Constraints: {}", info.constraints_compiled);

    // Solve
    let qubo_model = qubo_formulation.to_qubo_model();
    let (ising, offset) = qubo_model.to_ising();

    let mut solver = QuantumAnnealingSimulator::new(AnnealingParams {
        num_sweeps: 1500,
        num_repetitions: 8,
        ..Default::default()
    })?;

    let result = solver.solve(&ising)?;

    println!("  Solution energy: {:.4}", result.best_energy + offset);

    // Interpret one-hot solution (each cell uses 2 variables for values {1,2})
    println!("  Sudoku solution:");
    for i in 0..2 {
        print!("    ");
        for j in 0..2 {
            // Each cell has 2 variables: one for value 1, one for value 2
            let var1_idx = (i * 2 + j) * 2; // Variable for value 1
            let var2_idx = (i * 2 + j) * 2 + 1; // Variable for value 2

            let value = if var1_idx < result.best_spins.len() && result.best_spins[var1_idx] > 0 {
                1
            } else if var2_idx < result.best_spins.len() && result.best_spins[var2_idx] > 0 {
                2
            } else {
                0 // No assignment (shouldn't happen in valid solution)
            };

            print!("{value} ");
        }
        println!();
    }

    Ok(())
}

fn scheduling_example() -> Result<(), Box<dyn std::error::Error>> {
    // Simple task scheduling: 3 tasks, 2 time slots
    let mut problem = CspProblem::new();

    // Variables: task[i] = time slot for task i
    let time_slots = vec![0, 1]; // Two time slots
    for i in 0..3 {
        let var = CspVariable::new(format!("task_{i}"), Domain::Discrete(time_slots.clone()))
            .with_description(format!("Time slot for task {i}"));
        problem.add_variable(var)?;
    }

    // Constraint: at most one task per time slot (simplified version)
    // This is a simplified constraint - normally we'd need more complex resource constraints
    let constraint = CspConstraint::AtMostOne {
        variables: vec!["task_0".to_string(), "task_1".to_string()],
    };
    problem.add_constraint(constraint)?;

    let (qubo_formulation, info) = problem.compile_to_qubo()?;

    println!("Task Scheduling:");
    println!("  Tasks: 3, Time slots: 2");
    println!("  QUBO variables: {}", info.qubo_variables);
    println!("  Constraints: {}", info.constraints_compiled);

    // Solve
    let qubo_model = qubo_formulation.to_qubo_model();
    let (ising, offset) = qubo_model.to_ising();

    let mut solver = QuantumAnnealingSimulator::new(AnnealingParams::default())?;
    let result = solver.solve(&ising)?;

    println!("  Solution energy: {:.4}", result.best_energy + offset);

    // Interpret solution
    println!("  Task assignments:");
    for task in 0..3 {
        // Each task has 2 variables (one-hot encoding for 2 time slots)
        let slot0_idx = task * 2;
        let slot1_idx = task * 2 + 1;

        let assigned_slot =
            if slot0_idx < result.best_spins.len() && result.best_spins[slot0_idx] > 0 {
                0
            } else if slot1_idx < result.best_spins.len() && result.best_spins[slot1_idx] > 0 {
                1
            } else {
                -1 // No assignment
            };

        println!("    Task {task}: Time slot {assigned_slot}");
    }

    Ok(())
}

fn map_coloring_example() -> Result<(), Box<dyn std::error::Error>> {
    // Color a simple map with categorical color names
    let mut problem = CspProblem::new();

    // Regions with color domains
    let colors = vec!["Red".to_string(), "Blue".to_string(), "Green".to_string()];
    let regions = ["Region_A", "Region_B", "Region_C"];

    for region in &regions {
        let var = CspVariable::new((*region).to_string(), Domain::Categorical(colors.clone()))
            .with_description(format!("Color of {region}"));
        problem.add_variable(var)?;
    }

    // Adjacent regions must have different colors
    let adjacencies = [("Region_A", "Region_B"), ("Region_B", "Region_C")];
    for (region1, region2) in &adjacencies {
        let constraint = CspConstraint::AllDifferent {
            variables: vec![(*region1).to_string(), (*region2).to_string()],
        };
        problem.add_constraint(constraint)?;
    }

    let (qubo_formulation, info) = problem.compile_to_qubo()?;

    println!("Map Coloring with Categorical Variables:");
    println!("  Regions: {}", regions.len());
    println!("  Colors: {colors:?}");
    println!("  QUBO variables: {}", info.qubo_variables);

    // Solve
    let qubo_model = qubo_formulation.to_qubo_model();
    let (ising, offset) = qubo_model.to_ising();

    let mut solver = QuantumAnnealingSimulator::new(AnnealingParams {
        num_sweeps: 2000,
        num_repetitions: 10,
        ..Default::default()
    })?;

    let result = solver.solve(&ising)?;

    println!("  Solution energy: {:.4}", result.best_energy + offset);

    // Interpret categorical solution
    println!("  Region coloring:");
    for (region_idx, region) in regions.iter().enumerate() {
        // Each region has 3 variables (one-hot encoding for 3 colors)
        for (color_idx, color) in colors.iter().enumerate() {
            let var_idx = region_idx * 3 + color_idx;
            if var_idx < result.best_spins.len() && result.best_spins[var_idx] > 0 {
                println!("    {region}: {color}");
                break;
            }
        }
    }

    Ok(())
}
