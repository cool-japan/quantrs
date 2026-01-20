#![allow(clippy::pedantic, clippy::unnecessary_wraps)]
//! Example demonstrating problem-specific annealing schedules
//!
//! This example shows how to:
//! 1. Use automatic problem type detection
//! 2. Create optimized schedules for specific problem types
//! 3. Compare performance across different schedules
//! 4. Use adaptive schedule optimization

use quantrs2_anneal::{
    ising::IsingModel,
    problem_schedules::{AdaptiveScheduleOptimizer, ProblemSpecificScheduler, ProblemType},
    qubo::QuboBuilder,
    simulator::{ClassicalAnnealingSimulator, QuantumAnnealingSimulator},
};
use scirs2_core::random::prelude::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Problem-Specific Annealing Schedules Demo ===\n");

    // Create different types of problems
    let problems = vec![
        ("MaxCut", create_maxcut_problem(12)?),
        ("TSP", create_tsp_problem(8)?),
        (
            "Number Partitioning",
            create_number_partitioning_problem(10)?,
        ),
        ("Graph Coloring", create_graph_coloring_problem(9)?),
    ];

    let mut scheduler = ProblemSpecificScheduler::new();
    let mut adaptive_optimizer = AdaptiveScheduleOptimizer::new(0.2);

    for (name, model) in problems {
        println!("Problem: {name}");
        println!("Size: {} qubits", model.num_qubits);

        // Detect problem type automatically
        let detected_type = scheduler.detect_problem_type(&model)?;
        println!("Detected type: {detected_type:?}");

        // Get problem-specific schedule
        let specific_params = scheduler.create_schedule(&model, detected_type.clone())?;
        println!("Schedule parameters:");
        println!(
            "  Initial temperature: {:.2}",
            specific_params.initial_temperature
        );
        println!(
            "  Initial transverse field: {:.2}",
            specific_params.initial_transverse_field
        );
        println!("  Number of sweeps: {}", specific_params.num_sweeps);
        println!("  Repetitions: {}", specific_params.num_repetitions);

        // Run with problem-specific schedule
        let start = Instant::now();
        let mut specific_simulator = QuantumAnnealingSimulator::new(specific_params.clone())?;
        let specific_result = specific_simulator.solve(&model)?;
        let specific_time = start.elapsed();

        println!("\nProblem-specific schedule results:");
        println!("  Best energy: {:.4}", specific_result.best_energy);
        println!("  Time: {specific_time:.2?}");

        // Run with default schedule for comparison
        let default_params = Default::default();
        let start = Instant::now();
        let mut default_simulator = QuantumAnnealingSimulator::new(default_params)?;
        let default_result = default_simulator.solve(&model)?;
        let default_time = start.elapsed();

        println!("\nDefault schedule results:");
        println!("  Best energy: {:.4}", default_result.best_energy);
        println!("  Time: {default_time:.2?}");

        // Calculate improvement
        let energy_improvement = (default_result.best_energy - specific_result.best_energy)
            / default_result.best_energy.abs()
            * 100.0;
        let time_improvement = (default_time.as_secs_f64() - specific_time.as_secs_f64())
            / default_time.as_secs_f64()
            * 100.0;

        println!("\nImprovement:");
        println!("  Energy: {energy_improvement:.1}%");
        println!("  Time: {time_improvement:.1}%");

        // Record performance for adaptive optimization
        adaptive_optimizer.record_performance(
            detected_type.clone(),
            &specific_params,
            specific_result.best_energy,
            specific_time.as_secs_f64(),
            true,
        );

        println!("\n{}\n", "=".repeat(50));
    }

    // Demonstrate adaptive schedule optimization
    println!("=== Adaptive Schedule Optimization ===\n");

    // Create a new problem of known type
    let test_model = create_maxcut_problem(15)?;
    let problem_type = ProblemType::MaxCut;

    // Get base schedule
    let base_params = scheduler.create_schedule(&test_model, problem_type.clone())?;

    // Get adaptive suggestion based on history
    let adapted_params = adaptive_optimizer.suggest_schedule(problem_type, &base_params);

    println!("Adaptive optimization adjustments:");
    println!(
        "  Initial temperature: {:.2} -> {:.2}",
        base_params.initial_temperature, adapted_params.initial_temperature
    );
    println!(
        "  Initial field: {:.2} -> {:.2}",
        base_params.initial_transverse_field, adapted_params.initial_transverse_field
    );

    // Run with adapted schedule
    let mut adapted_simulator = QuantumAnnealingSimulator::new(adapted_params)?;
    let adapted_result = adapted_simulator.solve(&test_model)?;

    println!("\nAdapted schedule performance:");
    println!("  Best energy: {:.4}", adapted_result.best_energy);

    Ok(())
}

/// Create a `MaxCut` problem instance
fn create_maxcut_problem(n: usize) -> Result<IsingModel, Box<dyn std::error::Error>> {
    let mut model = IsingModel::new(n);

    // Create a random graph with negative couplings
    for i in 0..n {
        for j in (i + 1)..n {
            if thread_rng().gen::<f64>() < 0.3 {
                // 30% edge probability
                model.set_coupling(i, j, -1.0)?;
            }
        }
    }

    Ok(model)
}

/// Create a TSP problem instance
fn create_tsp_problem(n_cities: usize) -> Result<IsingModel, Box<dyn std::error::Error>> {
    // Create QUBO for small TSP
    let mut builder = QuboBuilder::new();

    // Distance matrix (simplified)
    let mut distances = vec![vec![0.0; n_cities]; n_cities];
    for i in 0..n_cities {
        for j in (i + 1)..n_cities {
            let dist = ((i as f64 - j as f64).abs() + 1.0) * thread_rng().gen::<f64>();
            distances[i][j] = dist;
            distances[j][i] = dist;
        }
    }

    // Variables: x[i][j] = 1 if city i is visited at position j
    let mut vars = vec![vec![None; n_cities]; n_cities];
    for i in 0..n_cities {
        for j in 0..n_cities {
            let var_name = format!("x_{i}_{j}");
            vars[i][j] = Some(builder.add_variable(var_name)?);
        }
    }

    // Constraint: each city visited exactly once
    builder.set_constraint_weight(10.0)?;
    for i in 0..n_cities {
        let city_vars: Vec<_> = vars[i]
            .iter()
            .filter_map(std::clone::Clone::clone)
            .collect();
        builder.constrain_one_hot(&city_vars)?;
    }

    // Constraint: each position has exactly one city
    for j in 0..n_cities {
        let position_vars: Vec<_> = (0..n_cities).filter_map(|i| vars[i][j].clone()).collect();
        builder.constrain_one_hot(&position_vars)?;
    }

    // Objective: minimize total distance
    for i in 0..n_cities {
        for j in 0..n_cities {
            for k in 0..n_cities {
                let next_j = (j + 1) % n_cities;
                if let (Some(var1), Some(var2)) = (&vars[i][j], &vars[k][next_j]) {
                    builder.set_quadratic_term(var1, var2, distances[i][k])?;
                }
            }
        }
    }

    let qubo = builder.build();
    Ok(qubo.to_ising().0)
}

/// Create a number partitioning problem
fn create_number_partitioning_problem(n: usize) -> Result<IsingModel, Box<dyn std::error::Error>> {
    let mut builder = QuboBuilder::new();

    // Generate random numbers to partition
    let numbers: Vec<f64> = (0..n).map(|_| thread_rng().gen::<f64>() * 100.0).collect();
    let target = numbers.iter().sum::<f64>() / 2.0;

    // Variables: x[i] = 1 if number i is in first partition
    let vars: Result<Vec<_>, _> = (0..n)
        .map(|i| builder.add_variable(format!("x_{i}")))
        .collect();
    let vars = vars?;

    // Objective: minimize (sum of partition - target)^2
    for i in 0..n {
        for j in 0..n {
            let coeff = if i == j {
                numbers[i] * 2.0f64.mul_add(-target, numbers[i])
            } else {
                2.0 * numbers[i] * numbers[j]
            };
            if i == j {
                builder.set_linear_term(&vars[i], coeff)?;
            } else {
                builder.set_quadratic_term(&vars[i], &vars[j], coeff)?;
            }
        }
    }

    let qubo = builder.build();
    Ok(qubo.to_ising().0)
}

/// Create a graph coloring problem
fn create_graph_coloring_problem(n: usize) -> Result<IsingModel, Box<dyn std::error::Error>> {
    let num_colors = 3;
    let mut builder = QuboBuilder::new();

    // Create a random graph
    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if thread_rng().gen::<f64>() < 0.4 {
                // 40% edge probability
                edges.push((i, j));
            }
        }
    }

    // Variables: x[i][c] = 1 if vertex i has color c
    let mut vars = vec![vec![None; num_colors]; n];
    for i in 0..n {
        for c in 0..num_colors {
            let var_name = format!("x_{i}_{c}");
            vars[i][c] = Some(builder.add_variable(var_name)?);
        }
    }

    // Constraint: each vertex has exactly one color
    builder.set_constraint_weight(10.0)?;
    for i in 0..n {
        let vertex_vars: Vec<_> = vars[i]
            .iter()
            .filter_map(std::clone::Clone::clone)
            .collect();
        builder.constrain_one_hot(&vertex_vars)?;
    }

    // Constraint: adjacent vertices have different colors
    for (i, j) in edges {
        for c in 0..num_colors {
            if let (Some(var1), Some(var2)) = (&vars[i][c], &vars[j][c]) {
                builder.set_quadratic_term(var1, var2, 10.0)?;
            }
        }
    }

    let qubo = builder.build();
    Ok(qubo.to_ising().0)
}
