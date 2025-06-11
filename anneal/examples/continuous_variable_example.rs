//! Example demonstrating continuous variable annealing
//!
//! This example shows how to:
//! 1. Define continuous optimization problems
//! 2. Set variable bounds and precision
//! 3. Add constraints to problems
//! 4. Solve using discretization and annealing
//! 5. Apply local search for refinement

use quantrs2_anneal::{
    continuous_variable::{
        create_quadratic_problem, ContinuousAnnealingConfig, ContinuousConstraint,
        ContinuousOptimizationProblem, ContinuousVariable, ContinuousVariableAnnealer,
    },
    simulator::AnnealingParams,
};
use std::collections::HashMap;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Continuous Variable Annealing Demo ===\n");

    // Example 1: Simple quadratic optimization
    println!("Example 1: Quadratic Programming");
    quadratic_programming_example()?;

    // Example 2: Constrained optimization
    println!("\nExample 2: Constrained Optimization");
    constrained_optimization_example()?;

    // Example 3: Portfolio optimization
    println!("\nExample 3: Portfolio Optimization");
    portfolio_optimization_example()?;

    // Example 4: Engineering design optimization
    println!("\nExample 4: Engineering Design Optimization");
    engineering_design_example()?;

    // Example 5: Multi-dimensional function optimization
    println!("\nExample 5: Multi-Dimensional Function Optimization");
    multidimensional_optimization_example()?;

    // Example 6: Parameter precision study
    println!("\nExample 6: Precision and Discretization Study");
    precision_study_example()?;

    Ok(())
}

fn quadratic_programming_example() -> Result<(), Box<dyn std::error::Error>> {
    // Minimize: x^2 + 2*y^2 - 4*x - 6*y + 10
    // Optimal solution: x = 2, y = 1.5, f(x,y) = 1.5

    let linear_coeffs = vec![-4.0, -6.0]; // Coefficients of x and y
    let quadratic_matrix = vec![
        vec![2.0, 0.0], // 2*x^2
        vec![0.0, 4.0], // 2*y^2 (coefficient is doubled for 0.5*x^T*Q*x form)
    ];
    let bounds = vec![(0.0, 5.0), (0.0, 3.0)]; // x ∈ [0,5], y ∈ [0,3]

    let problem = create_quadratic_problem(&linear_coeffs, &quadratic_matrix, &bounds, 8)?;

    let config = ContinuousAnnealingConfig {
        annealing_params: AnnealingParams {
            num_sweeps: 2000,
            num_repetitions: 10,
            ..Default::default()
        },
        adaptive_discretization: true,
        max_refinement_iterations: 3,
        local_search: true,
        local_search_iterations: 50,
        ..Default::default()
    };

    let start = Instant::now();
    let mut solver = ContinuousVariableAnnealer::new(config);
    let result = solver.solve(&problem)?;
    let runtime = start.elapsed();

    println!("Quadratic Programming Results:");
    println!("  Problem: minimize x² + 2y² - 4x - 6y + 10");
    println!("  Bounds: x ∈ [0,5], y ∈ [0,3]");
    println!("  Theoretical optimum: x=2, y=1.5, f=1.5");
    println!();
    println!("  Solution found:");
    for (var_name, &value) in &result.variable_values {
        println!("    {}: {:.4}", var_name, value);
    }
    println!("  Objective value: {:.4}", result.objective_value);
    println!("  Runtime: {:.2?}", runtime);

    // Compare with theoretical optimum
    let theoretical_optimum = 1.5;
    let gap = (result.objective_value - theoretical_optimum).abs();
    println!("  Gap to theoretical optimum: {:.4}", gap);

    // Show discretization statistics
    println!("\n  Discretization statistics:");
    println!(
        "    Refinement iterations: {}",
        result.stats.refinement_iterations
    );
    println!("    Converged: {}", result.stats.converged);
    for (var_name, resolution) in &result.stats.final_resolution {
        println!("    {} resolution: {:.6}", var_name, resolution);
    }

    Ok(())
}

fn constrained_optimization_example() -> Result<(), Box<dyn std::error::Error>> {
    // Minimize: (x - 3)² + (y - 2)²
    // Subject to: x + y ≤ 4, x ≥ 0, y ≥ 0

    // Create objective function
    let objective = Box::new(|vars: &HashMap<String, f64>| {
        let x = vars["x"];
        let y = vars["y"];
        (x - 3.0).powi(2) + (y - 2.0).powi(2)
    });

    let mut problem = ContinuousOptimizationProblem::new(objective);

    // Add variables
    let x_var = ContinuousVariable::new("x".to_string(), 0.0, 5.0, 8)?
        .with_description("X coordinate".to_string());
    let y_var = ContinuousVariable::new("y".to_string(), 0.0, 5.0, 8)?
        .with_description("Y coordinate".to_string());

    problem.add_variable(x_var)?;
    problem.add_variable(y_var)?;

    // Add constraint: x + y ≤ 4
    let constraint_fn = Box::new(|vars: &HashMap<String, f64>| {
        vars["x"] + vars["y"] - 4.0 // Returns ≤ 0 for feasible points
    });
    let constraint = ContinuousConstraint::new("sum_constraint".to_string(), constraint_fn, 100.0);
    problem.add_constraint(constraint);

    let config = ContinuousAnnealingConfig {
        annealing_params: AnnealingParams {
            num_sweeps: 3000,
            num_repetitions: 15,
            seed: Some(42),
            ..Default::default()
        },
        local_search: true,
        local_search_iterations: 100,
        ..Default::default()
    };

    let start = Instant::now();
    let mut solver = ContinuousVariableAnnealer::new(config);
    let result = solver.solve(&problem)?;
    let runtime = start.elapsed();

    println!("Constrained Optimization Results:");
    println!("  Problem: minimize (x-3)² + (y-2)²");
    println!("  Constraint: x + y ≤ 4");
    println!("  Bounds: x,y ∈ [0,5]");
    println!();
    println!("  Solution found:");
    for (var_name, &value) in &result.variable_values {
        println!("    {}: {:.4}", var_name, value);
    }
    println!("  Objective value: {:.4}", result.objective_value);
    println!("  Runtime: {:.2?}", runtime);

    // Check constraint satisfaction
    println!("\n  Constraint violations:");
    for (constraint_name, violation) in &result.constraint_violations {
        println!("    {}: {:.6}", constraint_name, violation);
    }

    // Theoretical solution: x + y = 4, minimize distance to (3,2)
    // Solution is on the line x + y = 4 closest to (3,2), which is (2.5, 1.5)
    let x_optimal = 2.5;
    let y_optimal = 1.5;
    let optimal_objective = (x_optimal - 3.0f64).powi(2) + (y_optimal - 2.0f64).powi(2);
    println!(
        "\n  Theoretical optimum: x={}, y={}, f={:.4}",
        x_optimal, y_optimal, optimal_objective
    );

    Ok(())
}

fn portfolio_optimization_example() -> Result<(), Box<dyn std::error::Error>> {
    // Portfolio optimization: minimize risk while achieving target return
    // Variables: weights of 4 assets, sum to 1

    let expected_returns = vec![0.08, 0.12, 0.15, 0.10];
    let target_return = 0.11;

    // Simplified covariance matrix (risk)
    let covariance = vec![
        vec![0.04, 0.01, 0.02, 0.01],
        vec![0.01, 0.09, 0.03, 0.02],
        vec![0.02, 0.03, 0.16, 0.04],
        vec![0.01, 0.02, 0.04, 0.06],
    ];

    // Objective: minimize portfolio variance
    let objective = Box::new(move |vars: &HashMap<String, f64>| {
        let weights: Vec<f64> = (0..4).map(|i| vars[&format!("w{}", i)]).collect();
        let mut variance = 0.0;

        for i in 0..4 {
            for j in 0..4 {
                variance += weights[i] * weights[j] * covariance[i][j];
            }
        }

        variance
    });

    let mut problem = ContinuousOptimizationProblem::new(objective);

    // Add weight variables (bounds: each weight between 0 and 1)
    for i in 0..4 {
        let var = ContinuousVariable::new(format!("w{}", i), 0.0, 1.0, 8)?
            .with_description(format!("Weight of asset {}", i));
        problem.add_variable(var)?;
    }

    // Constraint: weights sum to 1
    let sum_constraint_fn = Box::new(|vars: &HashMap<String, f64>| {
        let sum: f64 = (0..4).map(|i| vars[&format!("w{}", i)]).sum();
        (sum - 1.0).abs() - 0.01 // Allow small tolerance
    });
    let sum_constraint =
        ContinuousConstraint::new("weights_sum".to_string(), sum_constraint_fn, 1000.0);
    problem.add_constraint(sum_constraint);

    // Constraint: achieve target return
    let expected_returns_clone = expected_returns.clone();
    let return_constraint_fn = Box::new(move |vars: &HashMap<String, f64>| {
        let portfolio_return: f64 = (0..4)
            .map(|i| vars[&format!("w{}", i)] * expected_returns_clone[i])
            .sum();
        (target_return - portfolio_return).abs() - 0.005 // Allow small tolerance
    });
    let return_constraint =
        ContinuousConstraint::new("target_return".to_string(), return_constraint_fn, 1000.0);
    problem.add_constraint(return_constraint);

    let config = ContinuousAnnealingConfig {
        annealing_params: AnnealingParams {
            num_sweeps: 4000,
            num_repetitions: 20,
            seed: Some(123),
            ..Default::default()
        },
        max_refinement_iterations: 5,
        local_search: true,
        local_search_iterations: 200,
        ..Default::default()
    };

    let start = Instant::now();
    let mut solver = ContinuousVariableAnnealer::new(config);
    let result = solver.solve(&problem)?;
    let runtime = start.elapsed();

    println!("Portfolio Optimization Results:");
    println!("  Assets: 4");
    println!("  Expected returns: {:?}", expected_returns);
    println!("  Target return: {:.3}", target_return);
    println!();
    println!("  Optimal portfolio weights:");
    let mut total_weight = 0.0;
    let mut portfolio_return = 0.0;
    for i in 0..4 {
        let weight = result.variable_values[&format!("w{}", i)];
        total_weight += weight;
        portfolio_return += weight * expected_returns[i];
        println!("    Asset {}: {:.4} ({:.1}%)", i, weight, weight * 100.0);
    }
    println!();
    println!("  Portfolio statistics:");
    println!("    Total weight: {:.4}", total_weight);
    println!("    Portfolio return: {:.4}", portfolio_return);
    println!(
        "    Portfolio risk (variance): {:.6}",
        result.objective_value
    );
    println!(
        "    Portfolio volatility (std dev): {:.4}",
        result.objective_value.sqrt()
    );
    println!("    Runtime: {:.2?}", runtime);

    // Check constraints
    println!("\n  Constraint violations:");
    for (constraint_name, violation) in &result.constraint_violations {
        println!("    {}: {:.6}", constraint_name, violation);
    }

    Ok(())
}

fn engineering_design_example() -> Result<(), Box<dyn std::error::Error>> {
    // Pressure vessel design optimization
    // Minimize: cost = 0.6224*x1*x3*x4 + 1.7781*x2*x3² + 3.1661*x1²*x4 + 19.84*x1²*x3
    // Where: x1=thickness of shell, x2=thickness of head, x3=inner radius, x4=length

    let objective = Box::new(|vars: &HashMap<String, f64>| {
        let x1 = vars["shell_thickness"];
        let x2 = vars["head_thickness"];
        let x3 = vars["inner_radius"];
        let x4 = vars["length"];

        0.6224 * x1 * x3 * x4
            + 1.7781 * x2 * x3.powi(2)
            + 3.1661 * x1.powi(2) * x4
            + 19.84 * x1.powi(2) * x3
    });

    let mut problem = ContinuousOptimizationProblem::new(objective);

    // Add design variables with engineering bounds
    let shell_thickness =
        ContinuousVariable::new("shell_thickness".to_string(), 0.0625, 6.1875, 10)?
            .with_description("Thickness of cylindrical shell".to_string());
    let head_thickness = ContinuousVariable::new("head_thickness".to_string(), 0.0625, 6.1875, 10)?
        .with_description("Thickness of spherical head".to_string());
    let inner_radius = ContinuousVariable::new("inner_radius".to_string(), 10.0, 200.0, 10)?
        .with_description("Inner radius of cylinder".to_string());
    let length = ContinuousVariable::new("length".to_string(), 10.0, 200.0, 10)?
        .with_description("Length of cylinder".to_string());

    problem.add_variable(shell_thickness)?;
    problem.add_variable(head_thickness)?;
    problem.add_variable(inner_radius)?;
    problem.add_variable(length)?;

    // Add engineering constraints

    // Constraint 1: Pressure constraint
    let pressure_constraint = Box::new(|vars: &HashMap<String, f64>| {
        let x1 = vars["shell_thickness"];
        let x3 = vars["inner_radius"];
        -x1 + 0.0193 * x3 // Must be ≤ 0
    });
    problem.add_constraint(ContinuousConstraint::new(
        "pressure".to_string(),
        pressure_constraint,
        100.0,
    ));

    // Constraint 2: Stress constraint
    let stress_constraint = Box::new(|vars: &HashMap<String, f64>| {
        let x2 = vars["head_thickness"];
        let x3 = vars["inner_radius"];
        -x2 + 0.00954 * x3 // Must be ≤ 0
    });
    problem.add_constraint(ContinuousConstraint::new(
        "stress".to_string(),
        stress_constraint,
        100.0,
    ));

    // Constraint 3: Volume constraint (minimum volume requirement)
    let volume_constraint = Box::new(|vars: &HashMap<String, f64>| {
        let x3 = vars["inner_radius"];
        let x4 = vars["length"];
        let volume = std::f64::consts::PI * x3.powi(2) * x4
            + (4.0 / 3.0) * std::f64::consts::PI * x3.powi(3);
        1296000.0 - volume // Minimum volume requirement
    });
    problem.add_constraint(ContinuousConstraint::new(
        "volume".to_string(),
        volume_constraint,
        1.0,
    ));

    let config = ContinuousAnnealingConfig {
        annealing_params: AnnealingParams {
            num_sweeps: 5000,
            num_repetitions: 25,
            seed: Some(456),
            ..Default::default()
        },
        max_refinement_iterations: 4,
        local_search: true,
        local_search_iterations: 150,
        local_search_step_size: 0.005,
        ..Default::default()
    };

    let start = Instant::now();
    let mut solver = ContinuousVariableAnnealer::new(config);
    let result = solver.solve(&problem)?;
    let runtime = start.elapsed();

    println!("Engineering Design Optimization Results:");
    println!("  Problem: Pressure vessel design optimization");
    println!("  Objective: Minimize manufacturing cost");
    println!();
    println!("  Optimal design:");
    println!(
        "    Shell thickness: {:.4}",
        result.variable_values["shell_thickness"]
    );
    println!(
        "    Head thickness: {:.4}",
        result.variable_values["head_thickness"]
    );
    println!(
        "    Inner radius: {:.4}",
        result.variable_values["inner_radius"]
    );
    println!("    Length: {:.4}", result.variable_values["length"]);
    println!();
    println!("  Cost: {:.2}", result.objective_value);
    println!("  Runtime: {:.2?}", runtime);

    // Calculate final volume
    let x3 = result.variable_values["inner_radius"];
    let x4 = result.variable_values["length"];
    let volume =
        std::f64::consts::PI * x3.powi(2) * x4 + (4.0 / 3.0) * std::f64::consts::PI * x3.powi(3);
    println!("  Final volume: {:.2}", volume);

    // Check constraints
    println!("\n  Constraint violations:");
    for (constraint_name, violation) in &result.constraint_violations {
        println!("    {}: {:.6}", constraint_name, violation);
    }

    Ok(())
}

fn multidimensional_optimization_example() -> Result<(), Box<dyn std::error::Error>> {
    // Optimize Rosenbrock function: f(x,y) = 100*(y - x²)² + (1 - x)²
    // Global minimum at (1, 1) with f(1,1) = 0

    let objective = Box::new(|vars: &HashMap<String, f64>| {
        let x = vars["x"];
        let y = vars["y"];
        100.0 * (y - x.powi(2)).powi(2) + (1.0 - x).powi(2)
    });

    let mut problem = ContinuousOptimizationProblem::new(objective);

    // Add variables with bounds around the minimum
    let x_var = ContinuousVariable::new("x".to_string(), -2.0, 2.0, 10)?
        .with_description("X variable".to_string());
    let y_var = ContinuousVariable::new("y".to_string(), -1.0, 3.0, 10)?
        .with_description("Y variable".to_string());

    problem.add_variable(x_var)?;
    problem.add_variable(y_var)?;

    let config = ContinuousAnnealingConfig {
        annealing_params: AnnealingParams {
            num_sweeps: 6000,
            num_repetitions: 30,
            initial_temperature: 10.0, // Higher temperature for this difficult function
            seed: Some(789),
            ..Default::default()
        },
        max_refinement_iterations: 5,
        local_search: true,
        local_search_iterations: 300,
        local_search_step_size: 0.01,
        ..Default::default()
    };

    let start = Instant::now();
    let mut solver = ContinuousVariableAnnealer::new(config);
    let result = solver.solve(&problem)?;
    let runtime = start.elapsed();

    println!("Rosenbrock Function Optimization Results:");
    println!("  Problem: f(x,y) = 100*(y - x²)² + (1 - x)²");
    println!("  Global minimum: (1, 1) with f = 0");
    println!("  Search space: x ∈ [-2,2], y ∈ [-1,3]");
    println!();
    println!("  Solution found:");
    println!("    x: {:.6}", result.variable_values["x"]);
    println!("    y: {:.6}", result.variable_values["y"]);
    println!("  Objective value: {:.6}", result.objective_value);
    println!("  Runtime: {:.2?}", runtime);

    // Distance from global optimum
    let x_error = (result.variable_values["x"] - 1.0).abs();
    let y_error = (result.variable_values["y"] - 1.0).abs();
    let distance = (x_error.powi(2) + y_error.powi(2)).sqrt();

    println!("\n  Distance from global optimum: {:.6}", distance);
    println!("  Function value gap: {:.6}", result.objective_value);

    Ok(())
}

fn precision_study_example() -> Result<(), Box<dyn std::error::Error>> {
    // Study the effect of discretization precision on solution quality
    // Simple quadratic: f(x) = (x - π)², minimum at x = π ≈ 3.14159

    println!("Precision and Discretization Study:");
    println!("  Function: f(x) = (x - π)²");
    println!("  True minimum: x = π ≈ 3.14159, f = 0");
    println!("  Search range: x ∈ [0, 6]");
    println!();

    let precision_levels = vec![4, 6, 8, 10, 12];

    for &precision in &precision_levels {
        let objective = Box::new(|vars: &HashMap<String, f64>| {
            let x = vars["x"];
            (x - std::f64::consts::PI).powi(2)
        });

        let mut problem = ContinuousOptimizationProblem::new(objective);
        let var = ContinuousVariable::new("x".to_string(), 0.0, 6.0, precision)?;
        problem.add_variable(var.clone())?;

        let config = ContinuousAnnealingConfig {
            annealing_params: AnnealingParams {
                num_sweeps: 1000,
                num_repetitions: 5,
                seed: Some(42),
                ..Default::default()
            },
            adaptive_discretization: false, // Fixed precision for study
            local_search: false, // Don't use local search to see pure discretization effect
            ..Default::default()
        };

        let mut solver = ContinuousVariableAnnealer::new(config);
        let result = solver.solve(&problem)?;

        let error = (result.variable_values["x"] - std::f64::consts::PI).abs();
        let resolution = var.resolution();

        println!(
            "  {} bits: x = {:.6}, error = {:.6}, resolution = {:.6}, levels = {}",
            precision,
            result.variable_values["x"],
            error,
            resolution,
            var.num_levels()
        );
    }

    println!("\n  Observations:");
    println!("  - Higher precision bits give better approximation");
    println!("  - Resolution decreases exponentially with precision bits");
    println!("  - Trade-off between solution quality and computational cost");

    Ok(())
}
