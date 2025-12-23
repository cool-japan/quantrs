//! Example demonstrating the Domain-Specific Language (DSL) for Optimization
//!
//! This example shows how to:
//! 1. Define optimization models using the high-level DSL
//! 2. Use various variable types (binary, integer, categorical)
//! 3. Express constraints with natural syntax
//! 4. Use pre-built optimization patterns
//! 5. Compile models to QUBO/Ising formulations
//! 6. Solve optimization problems with the DSL

use quantrs2_anneal::{
    dsl::{patterns, Expression, OptimizationModel},
    simulator::{AnnealingParams, ClassicalAnnealingSimulator},
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Domain-Specific Language (DSL) for Optimization Demo ===\n");

    // Example 1: Portfolio Optimization
    println!("Example 1: Portfolio Optimization with DSL");
    portfolio_optimization_example()?;

    // Example 2: Sudoku Solver
    println!("\nExample 2: Sudoku Solver using DSL");
    sudoku_solver_example()?;

    // Example 3: Nurse Scheduling
    println!("\nExample 3: Nurse Scheduling Problem");
    nurse_scheduling_example()?;

    // Example 4: Pattern-based Knapsack
    println!("\nExample 4: Knapsack Problem using Pattern");
    knapsack_pattern_example()?;

    // Example 5: Graph Coloring Pattern
    println!("\nExample 5: Graph Coloring using Pattern");
    graph_coloring_pattern_example()?;

    // Example 6: Complex Expression Building
    println!("\nExample 6: Complex Expression Building");
    complex_expression_example()?;

    // Example 7: Multi-objective Optimization
    println!("\nExample 7: Multi-objective Optimization");
    multi_objective_example()?;

    Ok(())
}

fn portfolio_optimization_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = OptimizationModel::new("Portfolio Optimization");

    // Stock data
    let stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"];
    let expected_returns = vec![0.12, 0.15, 0.10, 0.18, 0.14];
    let risks = vec![0.20, 0.25, 0.15, 0.30, 0.28];
    let correlations = [
        vec![1.0, 0.3, 0.2, 0.1, 0.15],
        vec![0.3, 1.0, 0.25, 0.2, 0.35],
        vec![0.2, 0.25, 1.0, 0.15, 0.2],
        vec![0.1, 0.2, 0.15, 1.0, 0.4],
        vec![0.15, 0.35, 0.2, 0.4, 1.0],
    ];

    // Define binary variables for stock selection
    let selection = model.add_binary_vector("select", stocks.len())?;

    // Constraint 1: Select exactly 3 stocks
    model.add_constraint(selection.sum().equals(3))?;

    // Constraint 2: Risk diversification - no more than 40% in high-risk stocks
    let high_risk_stocks: Vec<f64> = risks
        .iter()
        .map(|&r| if r > 0.25 { 1.0 } else { 0.0 })
        .collect();

    model.add_constraint(
        selection
            .weighted_sum(&high_risk_stocks)
            .less_than_or_equal(1.0),
    )?;

    // Objective: Maximize expected return - risk penalty
    let return_expr = selection.weighted_sum(&expected_returns);

    // Simple risk penalty (for demonstration)
    let risk_penalty = selection.weighted_sum(&risks).scale(0.5);

    let objective = return_expr.add(risk_penalty.negate());
    model.maximize(objective)?;

    // Compile and solve
    println!("Model Summary:");
    println!("{}", model.summary());

    let ising = model.compile_to_ising()?;
    println!("Compiled to Ising model with {} qubits", ising.num_qubits);

    let start = Instant::now();
    let params = AnnealingParams::default();
    let simulator = ClassicalAnnealingSimulator::new(params)?;
    let result = simulator.solve(&ising)?;
    let runtime = start.elapsed();

    println!("\nOptimization Results:");
    println!("  Best energy: {:.6}", result.best_energy);
    println!("  Runtime: {runtime:.2?}");

    // Decode solution
    println!("\n  Selected stocks:");
    for (i, stock) in stocks.iter().enumerate() {
        if result.best_spins[i] == 1 {
            println!(
                "    - {} (return: {:.1}%, risk: {:.1}%)",
                stock,
                expected_returns[i] * 100.0,
                risks[i] * 100.0
            );
        }
    }

    let total_return: f64 = stocks
        .iter()
        .enumerate()
        .filter(|(i, _)| result.best_spins[*i] == 1)
        .map(|(i, _)| expected_returns[i])
        .sum();

    let total_risk: f64 = stocks
        .iter()
        .enumerate()
        .filter(|(i, _)| result.best_spins[*i] == 1)
        .map(|(i, _)| risks[i])
        .sum::<f64>()
        / 3.0;

    println!("\n  Portfolio metrics:");
    println!("    Expected return: {:.1}%", total_return * 100.0);
    println!("    Average risk: {:.1}%", total_risk * 100.0);

    Ok(())
}

fn sudoku_solver_example() -> Result<(), Box<dyn std::error::Error>> {
    // Simplified 4x4 Sudoku for demonstration
    let mut model = OptimizationModel::new("4x4 Sudoku Solver");

    // Binary variables x[row][col][value] = 1 if cell (row,col) contains value
    let mut x = Vec::new();
    for row in 0..4 {
        let mut row_vars = Vec::new();
        for col in 0..4 {
            let cell_vars = model.add_binary_vector(
                format!("cell_{row}_{col}"),
                4, // values 1-4
            )?;
            row_vars.push(cell_vars);
        }
        x.push(row_vars);
    }

    // Constraint 1: Each cell has exactly one value
    for row in 0..4 {
        for col in 0..4 {
            model.add_constraint(
                Expression::Sum(
                    (0..4)
                        .map(|v| Expression::Variable(x[row][col].variables[v].clone()))
                        .collect(),
                )
                .equals(1.0),
            )?;
        }
    }

    // Constraint 2: Each row contains each value exactly once
    for row in 0..4 {
        for value in 0..4 {
            let row_value_vars: Vec<_> = (0..4)
                .map(|col| x[row][col].variables[value].clone())
                .collect();

            model.add_constraint(
                Expression::Sum(
                    row_value_vars
                        .into_iter()
                        .map(Expression::Variable)
                        .collect(),
                )
                .equals(1.0),
            )?;
        }
    }

    // Constraint 3: Each column contains each value exactly once
    for col in 0..4 {
        for value in 0..4 {
            let col_value_vars: Vec<_> = (0..4)
                .map(|row| x[row][col].variables[value].clone())
                .collect();

            model.add_constraint(
                Expression::Sum(
                    col_value_vars
                        .into_iter()
                        .map(Expression::Variable)
                        .collect(),
                )
                .equals(1.0),
            )?;
        }
    }

    // Constraint 4: Each 2x2 box contains each value exactly once
    for box_row in 0..2 {
        for box_col in 0..2 {
            for value in 0..4 {
                let mut box_value_vars = Vec::new();
                for r in 0..2 {
                    for c in 0..2 {
                        let row = box_row * 2 + r;
                        let col = box_col * 2 + c;
                        box_value_vars.push(x[row][col].variables[value].clone());
                    }
                }

                model.add_constraint(
                    Expression::Sum(
                        box_value_vars
                            .into_iter()
                            .map(Expression::Variable)
                            .collect(),
                    )
                    .equals(1.0),
                )?;
            }
        }
    }

    // Add some known values (clues)
    let clues = vec![
        (0, 0, 0), // Cell (0,0) = 1 (value index 0)
        (1, 1, 2), // Cell (1,1) = 3 (value index 2)
        (2, 3, 1), // Cell (2,3) = 2 (value index 1)
    ];

    for (row, col, value) in clues {
        model.add_constraint(
            Expression::Variable(x[row][col].variables[value].clone()).equals(1.0),
        )?;
    }

    // No explicit objective - just find feasible solution
    model.minimize(Expression::constant(0.0))?;

    // Compile and solve
    println!(
        "Sudoku model created with {} qubits",
        model.summary().num_qubits
    );

    let ising = model.compile_to_ising()?;
    let params = AnnealingParams::default();
    let simulator = ClassicalAnnealingSimulator::new(params)?;
    let result = simulator.solve(&ising)?;

    // Decode and display solution
    println!("\nSudoku Solution:");
    let mut qubit_idx = 0;
    for row in 0..4 {
        print!("  ");
        for col in 0..4 {
            for value in 0..4 {
                if result.best_spins[qubit_idx] == 1 {
                    print!("{} ", value + 1);
                }
                qubit_idx += 1;
            }
        }
        println!();
    }

    Ok(())
}

fn nurse_scheduling_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = OptimizationModel::new("Nurse Scheduling");

    // Problem parameters
    let nurses = ["Alice", "Bob", "Carol", "David"];
    let shifts = ["Morning", "Afternoon", "Night"];
    let days = 3; // 3-day schedule

    // Nurse preferences (higher = more preferred)
    let preferences = vec![
        vec![0.8, 0.9, 0.2], // Alice prefers day shifts
        vec![0.5, 0.5, 0.9], // Bob prefers night shifts
        vec![0.9, 0.7, 0.3], // Carol prefers morning shifts
        vec![0.6, 0.8, 0.7], // David is flexible
    ];

    // Binary variables x[nurse][day][shift]
    let mut x = Vec::new();
    for n in 0..nurses.len() {
        let mut nurse_schedule = Vec::new();
        for d in 0..days {
            let day_shifts = model.add_binary_vector(format!("nurse_{n}_day_{d}"), shifts.len())?;
            nurse_schedule.push(day_shifts);
        }
        x.push(nurse_schedule);
    }

    // Constraint 1: Each nurse works at most one shift per day
    for n in 0..nurses.len() {
        for d in 0..days {
            let day_vars: Vec<_> = (0..shifts.len())
                .map(|s| x[n][d].variables[s].clone())
                .collect();

            model.add_constraint(
                Expression::Sum(day_vars.into_iter().map(Expression::Variable).collect())
                    .less_than_or_equal(1.0),
            )?;
        }
    }

    // Constraint 2: Each shift must have at least 1 nurse
    for d in 0..days {
        for s in 0..shifts.len() {
            let shift_vars: Vec<_> = (0..nurses.len())
                .map(|n| x[n][d].variables[s].clone())
                .collect();

            model.add_constraint(
                Expression::Sum(shift_vars.into_iter().map(Expression::Variable).collect())
                    .greater_than_or_equal(1.0),
            )?;
        }
    }

    // Constraint 3: No nurse works more than 2 shifts in the 3-day period
    for n in 0..nurses.len() {
        let mut nurse_total = Expression::Constant(0.0);
        for d in 0..days {
            nurse_total = nurse_total.add(x[n][d].sum());
        }
        model.add_constraint(nurse_total.less_than_or_equal(2.0))?;
    }

    // Objective: Maximize nurse satisfaction (preference scores)
    let mut satisfaction = Expression::Constant(0.0);

    for n in 0..nurses.len() {
        for d in 0..days {
            let shift_preferences = &preferences[n];
            satisfaction = satisfaction.add(x[n][d].weighted_sum(shift_preferences));
        }
    }

    model.maximize(satisfaction)?;

    // Compile and solve
    println!("Nurse scheduling model:");
    println!("  Variables: {}", model.summary().num_variables);
    println!("  Constraints: {}", model.summary().num_constraints);

    let ising = model.compile_to_ising()?;
    let params = AnnealingParams::default();
    let simulator = ClassicalAnnealingSimulator::new(params)?;
    let result = simulator.solve(&ising)?;

    // Decode and display schedule
    println!("\nNurse Schedule:");
    let mut qubit_idx = 0;
    for n in 0..nurses.len() {
        println!("  {}:", nurses[n]);
        for d in 0..days {
            print!("    Day {}: ", d + 1);
            let mut assigned = false;
            for s in 0..shifts.len() {
                if result.best_spins[qubit_idx] == 1 {
                    print!("{}", shifts[s]);
                    assigned = true;
                }
                qubit_idx += 1;
            }
            if !assigned {
                print!("Off");
            }
            println!();
        }
    }

    Ok(())
}

fn knapsack_pattern_example() -> Result<(), Box<dyn std::error::Error>> {
    let items = vec![
        "Laptop".to_string(),
        "Camera".to_string(),
        "Tablet".to_string(),
        "Phone".to_string(),
        "Headphones".to_string(),
    ];

    let values = vec![1000.0, 500.0, 300.0, 200.0, 150.0];
    let weights = vec![3.0, 2.0, 1.5, 0.5, 0.3];
    let capacity = 5.0;

    // Use the pre-built knapsack pattern
    let model = patterns::knapsack(&items, &values, &weights, capacity)?;

    println!("Knapsack problem model:");
    println!("{}", model.summary());

    // Compile and solve
    let ising = model.compile_to_ising()?;
    let params = AnnealingParams::default();
    let simulator = ClassicalAnnealingSimulator::new(params)?;
    let result = simulator.solve(&ising)?;

    // Decode solution
    println!("\nKnapsack Solution:");
    let mut total_value = 0.0;
    let mut total_weight = 0.0;

    println!("  Selected items:");
    for (i, item) in items.iter().enumerate() {
        if result.best_spins[i] == 1 {
            println!(
                "    - {} (value: ${:.0}, weight: {:.1}kg)",
                item, values[i], weights[i]
            );
            total_value += values[i];
            total_weight += weights[i];
        }
    }

    println!("\n  Total value: ${total_value:.0}");
    println!("  Total weight: {total_weight:.1}kg / {capacity:.1}kg");

    Ok(())
}

fn graph_coloring_pattern_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple graph (pentagon)
    let vertices = vec![
        "A".to_string(),
        "B".to_string(),
        "C".to_string(),
        "D".to_string(),
        "E".to_string(),
    ];

    let edges = vec![
        (0, 1), // A-B
        (1, 2), // B-C
        (2, 3), // C-D
        (3, 4), // D-E
        (4, 0), // E-A
        (0, 2), // A-C (diagonal)
        (0, 3), // A-D (diagonal)
    ];

    let num_colors = 4;

    // Use the pre-built graph coloring pattern
    let model = patterns::graph_coloring(&vertices, &edges, num_colors)?;

    println!("Graph coloring model:");
    println!("  Vertices: {}", vertices.len());
    println!("  Edges: {}", edges.len());
    println!("  Available colors: {num_colors}");
    println!("  Variables: {}", model.summary().num_variables);
    println!("  Constraints: {}", model.summary().num_constraints);

    // Compile and solve
    let ising = model.compile_to_ising()?;
    let params = AnnealingParams::default();
    let simulator = ClassicalAnnealingSimulator::new(params)?;
    let result = simulator.solve(&ising)?;

    // Decode solution
    println!("\nGraph Coloring Solution:");
    let color_names = ["Red", "Blue", "Green", "Yellow"];

    let mut qubit_idx = 0;
    for (v, vertex) in vertices.iter().enumerate() {
        print!("  {vertex}: ");
        for c in 0..num_colors {
            if result.best_spins[qubit_idx] == 1 {
                print!("{}", color_names[c]);
            }
            qubit_idx += 1;
        }
        println!();
    }

    // Verify solution
    println!("\n  Verification:");
    let mut valid = true;
    for &(u, v) in &edges {
        let u_color = (0..num_colors)
            .find(|&c| result.best_spins[u * num_colors + c] == 1)
            .unwrap_or(num_colors);
        let v_color = (0..num_colors)
            .find(|&c| result.best_spins[v * num_colors + c] == 1)
            .unwrap_or(num_colors + 1);

        if u_color == v_color {
            println!("    Edge {}-{}: CONFLICT!", vertices[u], vertices[v]);
            valid = false;
        }
    }

    if valid {
        println!("    All constraints satisfied ✓");
    }

    Ok(())
}

fn complex_expression_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = OptimizationModel::new("Complex Expression Example");

    // Create variables
    let x = model.add_binary("x")?;
    let y = model.add_binary("y")?;
    let z = model.add_binary("z")?;
    let w = model.add_integer("w", 0, 3)?;

    // Build complex expressions
    let expr1 = Expression::Variable(x.clone())
        .add(Expression::Variable(y.clone()).scale(2.0))
        .add(Expression::Variable(z.clone()).scale(3.0));

    let expr2 = Expression::Quadratic {
        var1: x.clone(),
        var2: y.clone(),
        coefficient: -1.5,
    };

    let expr3 = Expression::Variable(w.clone()).scale(0.5);

    // Combine expressions
    let total_expr = expr1.add(expr2).add(expr3);

    // Add constraints
    model.add_constraint(
        Expression::Variable(x)
            .add(Expression::Variable(y))
            .less_than_or_equal(1.0),
    )?;

    model.add_constraint(
        Expression::Variable(z)
            .add(Expression::Variable(w))
            .greater_than_or_equal(1.0),
    )?;

    // Set objective
    model.minimize(total_expr)?;

    // Compile and solve
    println!("Complex expression model:");
    println!("{}", model.summary());

    let ising = model.compile_to_ising()?;
    let params = AnnealingParams::default();
    let simulator = ClassicalAnnealingSimulator::new(params)?;
    let result = simulator.solve(&ising)?;

    // Decode solution
    println!("\nSolution:");
    println!("  x = {}", result.best_spins[0]);
    println!("  y = {}", result.best_spins[1]);
    println!("  z = {}", result.best_spins[2]);

    // Decode integer variable w (2 bits)
    let w_value = result.best_spins[3] + 2 * result.best_spins[4];
    println!("  w = {w_value}");

    // Calculate objective value
    let obj_value = 0.5f64.mul_add(
        f64::from(w_value),
        1.5f64.mul_add(
            -f64::from(result.best_spins[0] * result.best_spins[1]),
            3.0f64.mul_add(
                f64::from(result.best_spins[2]),
                2.0f64.mul_add(
                    f64::from(result.best_spins[1]),
                    f64::from(result.best_spins[0]),
                ),
            ),
        ),
    );

    println!("\n  Objective value: {obj_value:.2}");

    Ok(())
}

fn multi_objective_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = OptimizationModel::new("Multi-objective Optimization");

    // Production planning: maximize profit while minimizing environmental impact
    let products = ["A", "B", "C", "D"];
    let profits = vec![100.0, 150.0, 80.0, 120.0];
    let emissions = vec![5.0, 8.0, 3.0, 6.0]; // CO2 emissions
    let resources = vec![2.0, 3.0, 1.5, 2.5]; // Resource usage
    let resource_limit = 8.0;

    // Binary variables for production decisions
    let produce = model.add_binary_vector("produce", products.len())?;

    // Constraint: resource limit
    model.add_constraint(
        produce
            .weighted_sum(&resources)
            .less_than_or_equal(resource_limit),
    )?;

    // Multi-objective: maximize profit - emission penalty
    let profit_expr = produce.weighted_sum(&profits);
    let emission_expr = produce.weighted_sum(&emissions);

    // Weighted combination (scalarization)
    let emission_weight = 10.0; // Weight for environmental objective
    let combined_objective = profit_expr.add(emission_expr.scale(-emission_weight));

    model.maximize(combined_objective)?;

    // Compile and solve
    println!("Multi-objective model:");
    println!("{}", model.summary());

    let ising = model.compile_to_ising()?;
    let params = AnnealingParams::default();
    let simulator = ClassicalAnnealingSimulator::new(params)?;
    let result = simulator.solve(&ising)?;

    // Decode solution
    println!("\nProduction Plan:");
    let mut total_profit = 0.0;
    let mut total_emissions = 0.0;
    let mut total_resources = 0.0;

    for (i, product) in products.iter().enumerate() {
        if result.best_spins[i] == 1 {
            println!("  Produce product {product}");
            println!("    Profit: ${:.0}", profits[i]);
            println!("    Emissions: {:.1} units", emissions[i]);
            println!("    Resources: {:.1} units", resources[i]);

            total_profit += profits[i];
            total_emissions += emissions[i];
            total_resources += resources[i];
        }
    }

    println!("\n  Totals:");
    println!("    Total profit: ${total_profit:.0}");
    println!("    Total emissions: {total_emissions:.1} units");
    println!("    Total resources: {total_resources:.1}/{resource_limit:.1} units");

    println!("\n  Analysis:");
    println!(
        "    Profit per emission: ${:.2}/unit",
        total_profit / total_emissions
    );
    println!(
        "    Resource utilization: {:.1}%",
        total_resources / resource_limit * 100.0
    );

    Ok(())
}
