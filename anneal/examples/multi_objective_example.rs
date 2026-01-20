#![allow(clippy::pedantic, clippy::unnecessary_wraps)]
//! Example demonstrating multi-objective optimization with quantum annealing
//!
//! This example shows how to:
//! 1. Define multi-objective optimization problems
//! 2. Use different scalarization methods
//! 3. Generate Pareto-optimal solutions
//! 4. Calculate quality metrics for multi-objective results
//! 5. Analyze trade-offs between conflicting objectives

use quantrs2_anneal::{
    ising::IsingModel,
    multi_objective::{
        MultiObjectiveConfig, MultiObjectiveOptimizer, MultiObjectiveSolution, QualityMetrics,
        ScalarizationMethod,
    },
    simulator::AnnealingParams,
};
use std::collections::HashMap;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Multi-Objective Optimization Demo ===\n");

    // Example 1: Portfolio optimization (risk vs return)
    println!("Example 1: Portfolio Optimization");
    portfolio_optimization_example()?;

    // Example 2: Engineering design (cost vs performance)
    println!("\nExample 2: Engineering Design Optimization");
    engineering_design_example()?;

    // Example 3: Network design (latency vs reliability)
    println!("\nExample 3: Network Design Optimization");
    network_design_example()?;

    // Example 4: Scalarization method comparison
    println!("\nExample 4: Scalarization Method Comparison");
    scalarization_comparison_example()?;

    // Example 5: Quality metrics analysis
    println!("\nExample 5: Quality Metrics Analysis");
    quality_metrics_example()?;

    // Example 6: Three-objective optimization
    println!("\nExample 6: Three-Objective Optimization");
    three_objective_example()?;

    Ok(())
}

fn portfolio_optimization_example() -> Result<(), Box<dyn std::error::Error>> {
    // Portfolio optimization: minimize risk while maximizing return
    // Assets: [Conservative, Balanced, Aggressive, Speculative]

    let expected_returns = vec![0.05, 0.08, 0.12, 0.18];
    let risks = vec![0.02, 0.05, 0.10, 0.20];
    let correlations = vec![
        vec![1.0, 0.3, 0.1, -0.1],
        vec![0.3, 1.0, 0.5, 0.2],
        vec![0.1, 0.5, 1.0, 0.7],
        vec![-0.1, 0.2, 0.7, 1.0],
    ];

    // Create a 4-asset QUBO model
    let mut model = create_portfolio_model(&expected_returns, &risks, &correlations)?;

    // Objective 1: Minimize risk (portfolio variance)
    // Objective 2: Maximize return (negative to convert to minimization)
    let expected_returns_clone = expected_returns.clone();
    let risks_clone = risks.clone();
    let correlations_clone = correlations;
    let objective_function = Box::new(move |spins: &[i8]| -> Vec<f64> {
        // Convert spin solution to portfolio weights
        let weights = spins_to_portfolio_weights(spins);

        // Calculate portfolio risk (variance)
        let mut risk = 0.0;
        for i in 0..4 {
            for j in 0..4 {
                risk += weights[i]
                    * weights[j]
                    * risks_clone[i]
                    * risks_clone[j]
                    * correlations_clone[i][j];
            }
        }

        // Calculate portfolio return
        let portfolio_return: f64 = weights
            .iter()
            .zip(expected_returns_clone.iter())
            .map(|(w, r)| w * r)
            .sum();

        // Return both objectives (risk, negative return)
        vec![risk, -portfolio_return]
    });

    // Configure multi-objective optimization with weighted sum
    let config = MultiObjectiveConfig {
        annealing_params: AnnealingParams {
            num_sweeps: 2000,
            num_repetitions: 10,
            seed: Some(42),
            ..Default::default()
        },
        scalarization: ScalarizationMethod::WeightedSum {
            weights: vec![0.6, 0.4], // 60% weight on risk, 40% on return
        },
        num_pareto_runs: 15,
        population_size: 50,
        ..Default::default()
    };

    let start = Instant::now();
    let mut optimizer = MultiObjectiveOptimizer::new(config);
    let results = optimizer.solve(&model, objective_function, 2)?;
    let runtime = start.elapsed();

    println!("Portfolio Optimization Results:");
    println!("  Assets: Conservative, Balanced, Aggressive, Speculative");
    println!("  Expected returns: {expected_returns:?}");
    println!("  Risks (volatility): {risks:?}");
    println!();
    println!("  Solutions found: {}", results.all_solutions.len());
    println!("  Pareto front size: {}", results.pareto_front.len());
    println!("  Runtime: {runtime:.2?}");

    // Show best solutions for each objective
    if let Some(min_risk_sol) = results
        .all_solutions
        .iter()
        .min_by(|a, b| a.objectives[0].partial_cmp(&b.objectives[0]).unwrap())
    {
        let weights = spins_to_portfolio_weights(&min_risk_sol.variables);
        println!("\n  Minimum risk portfolio:");
        println!(
            "    Weights: {:?}",
            weights
                .iter()
                .map(|w| format!("{w:.3}"))
                .collect::<Vec<_>>()
        );
        println!("    Risk: {:.4}", min_risk_sol.objectives[0]);
        println!("    Return: {:.4}", -min_risk_sol.objectives[1]);
    }

    if let Some(max_return_sol) = results
        .all_solutions
        .iter()
        .min_by(|a, b| a.objectives[1].partial_cmp(&b.objectives[1]).unwrap())
    {
        let weights = spins_to_portfolio_weights(&max_return_sol.variables);
        println!("\n  Maximum return portfolio:");
        println!(
            "    Weights: {:?}",
            weights
                .iter()
                .map(|w| format!("{w:.3}"))
                .collect::<Vec<_>>()
        );
        println!("    Risk: {:.4}", max_return_sol.objectives[0]);
        println!("    Return: {:.4}", -max_return_sol.objectives[1]);
    }

    // Show Pareto front statistics
    println!("\n  Pareto front analysis:");
    let avg_crowding = results.stats.average_crowding_distance;
    println!("    Average crowding distance: {avg_crowding:.4}");
    println!("    Objective ranges:");
    for (i, (min_val, max_val)) in results.objective_bounds.iter().enumerate() {
        let obj_name = if i == 0 { "Risk" } else { "Return" };
        let display_max = if i == 1 { -min_val } else { *max_val };
        let display_min = if i == 1 { -max_val } else { *min_val };
        println!("      {obj_name}: [{display_min:.4}, {display_max:.4}]");
    }

    Ok(())
}

fn engineering_design_example() -> Result<(), Box<dyn std::error::Error>> {
    // Engineering design: minimize cost while maximizing performance
    // Design a system with multiple components

    let component_costs = vec![10.0, 25.0, 40.0, 80.0]; // Cost per component type
    let component_performance = vec![1.0, 2.5, 4.0, 7.5]; // Performance per component

    // Create model for component selection (4 components, each can be on/off)
    let mut model = IsingModel::new(4);

    // Add coupling to encourage balanced solutions
    for i in 0..4 {
        for j in (i + 1)..4 {
            model.set_coupling(i, j, -0.1)?;
        }
    }

    let component_costs_clone = component_costs.clone();
    let component_performance_clone = component_performance.clone();
    let objective_function = Box::new(move |spins: &[i8]| -> Vec<f64> {
        let mut total_cost = 0.0;
        let mut total_performance = 0.0;

        for i in 0..4 {
            if spins[i] > 0 {
                // Component is selected
                total_cost += component_costs_clone[i];
                total_performance += component_performance_clone[i];
            }
        }

        // Add penalty for no components selected
        if total_performance == 0.0 {
            total_cost += 1000.0;
        }

        // Return (cost, negative performance)
        vec![total_cost, -total_performance]
    });

    // Use Chebyshev scalarization for better Pareto front coverage
    let config = MultiObjectiveConfig {
        annealing_params: AnnealingParams {
            num_sweeps: 1500,
            num_repetitions: 8,
            seed: Some(123),
            ..Default::default()
        },
        scalarization: ScalarizationMethod::WeightedChebyshev {
            weights: vec![1.0, 1.0],
            reference_point: vec![200.0, -15.0], // Reference point beyond feasible region
        },
        num_pareto_runs: 12,
        ..Default::default()
    };

    let start = Instant::now();
    let mut optimizer = MultiObjectiveOptimizer::new(config);
    let results = optimizer.solve(&model, objective_function, 2)?;
    let runtime = start.elapsed();

    println!("Engineering Design Results:");
    println!("  Components: A (${}, {:.1} perf), B (${}, {:.1} perf), C (${}, {:.1} perf), D (${}, {:.1} perf)",
             component_costs[0], component_performance[0],
             component_costs[1], component_performance[1],
             component_costs[2], component_performance[2],
             component_costs[3], component_performance[3]);
    println!();
    println!("  Solutions found: {}", results.all_solutions.len());
    println!("  Pareto front size: {}", results.pareto_front.len());
    println!("  Runtime: {runtime:.2?}");

    // Show Pareto optimal designs
    println!("\n  Pareto optimal designs:");
    for (i, solution) in results.pareto_front.iter().take(5).enumerate() {
        let selected_components: Vec<char> = solution
            .variables
            .iter()
            .enumerate()
            .filter(|(_, &spin)| spin > 0)
            .map(|(idx, _)| (b'A' + idx as u8) as char)
            .collect();

        println!(
            "    Design {}: Components {:?}, Cost: ${:.0}, Performance: {:.1}",
            i + 1,
            selected_components,
            solution.objectives[0],
            -solution.objectives[1]
        );
    }

    Ok(())
}

fn network_design_example() -> Result<(), Box<dyn std::error::Error>> {
    // Network design: minimize latency while maximizing reliability
    // Design a communication network topology

    // Create a 6-node network design problem
    let mut model = IsingModel::new(15); // 6 choose 2 = 15 possible edges

    // Add constraints to encourage connected topologies
    let edges = generate_edge_list(6);
    for i in 0..edges.len() {
        model.set_bias(i, -0.5)?; // Slight preference for including edges
    }

    let objective_function = Box::new(move |spins: &[i8]| -> Vec<f64> {
        let selected_edges: Vec<(usize, usize)> = spins
            .iter()
            .enumerate()
            .filter(|(_, &spin)| spin > 0)
            .map(|(idx, _)| edges[idx])
            .collect();

        if selected_edges.is_empty() {
            return vec![1000.0, -0.0]; // Penalty for no edges
        }

        // Calculate network metrics
        let latency = calculate_average_path_length(&selected_edges, 6);
        let reliability = calculate_network_reliability(&selected_edges, 6);

        vec![latency, -reliability]
    });

    let config = MultiObjectiveConfig {
        annealing_params: AnnealingParams {
            num_sweeps: 3000,
            num_repetitions: 15,
            seed: Some(456),
            ..Default::default()
        },
        scalarization: ScalarizationMethod::EpsilonConstraint {
            primary_objective: 0,         // Minimize latency
            constraints: vec![0.0, -2.0], // Constraint on reliability
        },
        num_pareto_runs: 10,
        ..Default::default()
    };

    let start = Instant::now();
    let mut optimizer = MultiObjectiveOptimizer::new(config);
    let results = optimizer.solve(&model, objective_function, 2)?;
    let runtime = start.elapsed();

    println!("Network Design Results:");
    println!("  Nodes: 6");
    println!("  Possible edges: 15");
    println!("  Objectives: Minimize latency, Maximize reliability");
    println!();
    println!("  Solutions found: {}", results.all_solutions.len());
    println!("  Pareto front size: {}", results.pareto_front.len());
    println!("  Runtime: {runtime:.2?}");

    // Show best topologies
    println!("\n  Best network topologies:");
    for (i, solution) in results.pareto_front.iter().take(3).enumerate() {
        let num_edges = solution.variables.iter().filter(|&&spin| spin > 0).count();
        println!(
            "    Topology {}: {} edges, Latency: {:.2}, Reliability: {:.2}",
            i + 1,
            num_edges,
            solution.objectives[0],
            -solution.objectives[1]
        );
    }

    Ok(())
}

fn scalarization_comparison_example() -> Result<(), Box<dyn std::error::Error>> {
    // Compare different scalarization methods on the same problem

    let mut model = IsingModel::new(6);

    // Simple bi-objective problem
    for i in 0..6 {
        model.set_bias(i, (i as f64 - 2.5) * 0.5)?;
    }

    let objective_function = Box::new(|spins: &[i8]| -> Vec<f64> {
        let obj1 = spins
            .iter()
            .enumerate()
            .map(|(i, &s)| if s > 0 { i as f64 } else { 0.0 })
            .sum::<f64>();
        let obj2 = spins.iter().filter(|&&s| s > 0).count() as f64;
        vec![obj1, obj2]
    });

    let methods = vec![
        (
            "Weighted Sum",
            ScalarizationMethod::WeightedSum {
                weights: vec![0.7, 0.3],
            },
        ),
        (
            "Chebyshev",
            ScalarizationMethod::WeightedChebyshev {
                weights: vec![1.0, 1.0],
                reference_point: vec![20.0, 8.0],
            },
        ),
        (
            "ε-Constraint",
            ScalarizationMethod::EpsilonConstraint {
                primary_objective: 0,
                constraints: vec![0.0, 4.0],
            },
        ),
    ];

    println!("Scalarization Method Comparison:");

    for (method_name, scalarization) in methods {
        let config = MultiObjectiveConfig {
            annealing_params: AnnealingParams {
                num_sweeps: 1000,
                num_repetitions: 5,
                seed: Some(42),
                ..Default::default()
            },
            scalarization,
            num_pareto_runs: 8,
            ..Default::default()
        };

        let start = Instant::now();
        let mut optimizer = MultiObjectiveOptimizer::new(config);
        let results = optimizer.solve(&model, objective_function.clone(), 2)?;
        let runtime = start.elapsed();

        println!("\n  {method_name} Method:");
        println!("    Pareto front size: {}", results.pareto_front.len());
        println!(
            "    Avg crowding distance: {:.4}",
            results.stats.average_crowding_distance
        );
        println!("    Runtime: {runtime:.2?}");

        if !results.pareto_front.is_empty() {
            let best = &results.pareto_front[0];
            println!(
                "    Best solution: Obj1={:.2}, Obj2={:.2}",
                best.objectives[0], best.objectives[1]
            );
        }
    }

    Ok(())
}

fn quality_metrics_example() -> Result<(), Box<dyn std::error::Error>> {
    // Demonstrate quality metrics for multi-objective results

    // Create some example solutions
    let solutions = vec![
        MultiObjectiveSolution::new(vec![1, -1, 1], vec![2.0, 4.0]),
        MultiObjectiveSolution::new(vec![-1, 1, -1], vec![3.0, 3.0]),
        MultiObjectiveSolution::new(vec![1, 1, -1], vec![4.0, 2.0]),
        MultiObjectiveSolution::new(vec![-1, -1, 1], vec![1.5, 4.5]),
        MultiObjectiveSolution::new(vec![1, -1, -1], vec![3.5, 2.5]),
    ];

    println!("Quality Metrics Analysis:");
    println!("  Number of solutions: {}", solutions.len());

    // Calculate spacing metric
    let spacing = QualityMetrics::spacing(&solutions);
    println!("  Spacing metric: {spacing:.4}");

    // Calculate hypervolume with different reference points
    let reference_points = [vec![5.0, 5.0], vec![6.0, 6.0], vec![4.0, 6.0]];

    for (i, ref_point) in reference_points.iter().enumerate() {
        match QualityMetrics::hypervolume(&solutions, ref_point) {
            Ok(hv) => println!("  Hypervolume (ref {}): {:.4}", i + 1, hv),
            Err(e) => println!("  Hypervolume calculation failed: {e}"),
        }
    }

    // Show solution distribution
    println!("\n  Solution distribution:");
    for (i, sol) in solutions.iter().enumerate() {
        println!(
            "    Sol {}: ({:.2}, {:.2})",
            i + 1,
            sol.objectives[0],
            sol.objectives[1]
        );
    }

    // Calculate ranges
    let obj1_range = solutions
        .iter()
        .map(|s| s.objectives[0])
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), val| {
            (min.min(val), max.max(val))
        });
    let obj2_range = solutions
        .iter()
        .map(|s| s.objectives[1])
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), val| {
            (min.min(val), max.max(val))
        });

    println!("\n  Objective ranges:");
    println!(
        "    Objective 1: [{:.2}, {:.2}]",
        obj1_range.0, obj1_range.1
    );
    println!(
        "    Objective 2: [{:.2}, {:.2}]",
        obj2_range.0, obj2_range.1
    );

    Ok(())
}

fn three_objective_example() -> Result<(), Box<dyn std::error::Error>> {
    // Demonstrate three-objective optimization

    let mut model = IsingModel::new(8);

    // Add some structure to the problem
    for i in 0..8 {
        model.set_bias(i, (i as f64 - 3.5) * 0.2)?;
        if i < 7 {
            model.set_coupling(i, i + 1, -0.3)?;
        }
    }

    let objective_function = Box::new(|spins: &[i8]| -> Vec<f64> {
        let num_up = spins.iter().filter(|&&s| s > 0).count() as f64;
        let energy = spins
            .iter()
            .enumerate()
            .map(|(i, &s)| f64::from(s) * (i as f64 - 3.5) * 0.2)
            .sum::<f64>();
        let connectivity = spins.windows(2).filter(|pair| pair[0] == pair[1]).count() as f64;

        vec![energy, 8.0 - num_up, 7.0 - connectivity] // Three objectives to minimize
    });

    let config = MultiObjectiveConfig {
        annealing_params: AnnealingParams {
            num_sweeps: 2500,
            num_repetitions: 12,
            seed: Some(789),
            ..Default::default()
        },
        scalarization: ScalarizationMethod::WeightedSum {
            weights: vec![0.4, 0.3, 0.3],
        },
        num_pareto_runs: 20,
        max_pareto_solutions: 30,
        ..Default::default()
    };

    let start = Instant::now();
    let mut optimizer = MultiObjectiveOptimizer::new(config);
    let results = optimizer.solve(&model, objective_function, 3)?;
    let runtime = start.elapsed();

    println!("Three-Objective Optimization Results:");
    println!("  Problem: 8-spin system");
    println!("  Objectives: Energy, Sparsity (8-num_up), Fragmentation (7-connectivity)");
    println!();
    println!("  Solutions found: {}", results.all_solutions.len());
    println!("  Pareto front size: {}", results.pareto_front.len());
    println!("  Runtime: {runtime:.2?}");

    // Show diverse Pareto optimal solutions
    println!("\n  Pareto optimal solutions:");
    for (i, solution) in results.pareto_front.iter().take(8).enumerate() {
        let config_str = solution
            .variables
            .iter()
            .map(|&s| if s > 0 { "↑" } else { "↓" })
            .collect::<String>();
        println!(
            "    Sol {}: {} | Energy: {:.2}, Sparsity: {:.1}, Fragmentation: {:.1}",
            i + 1,
            config_str,
            solution.objectives[0],
            solution.objectives[1],
            solution.objectives[2]
        );
    }

    // Show objective bounds
    println!("\n  Objective bounds:");
    for (i, (min_val, max_val)) in results.objective_bounds.iter().enumerate() {
        let obj_name = match i {
            0 => "Energy",
            1 => "Sparsity",
            2 => "Fragmentation",
            _ => "Unknown",
        };
        println!("    {obj_name}: [{min_val:.2}, {max_val:.2}]");
    }

    Ok(())
}

// Helper functions

fn create_portfolio_model(
    _returns: &[f64],
    _risks: &[f64],
    _correlations: &[Vec<f64>],
) -> Result<IsingModel, Box<dyn std::error::Error>> {
    // Create a simplified portfolio model
    let mut model = IsingModel::new(8); // 2 bits per asset for weights

    // Add structure to encourage portfolio diversity
    for i in 0..8 {
        model.set_bias(i, -0.1)?;
    }

    // Couple bits within each asset
    for asset in 0..4 {
        let bit1 = asset * 2;
        let bit2 = asset * 2 + 1;
        model.set_coupling(bit1, bit2, 0.5)?;
    }

    Ok(model)
}

fn spins_to_portfolio_weights(spins: &[i8]) -> Vec<f64> {
    let mut weights = vec![0.0; 4];

    // Convert 2 bits per asset to weight (0, 0.33, 0.67, 1.0)
    for asset in 0..4 {
        let bit1 = i32::from(spins[asset * 2] > 0);
        let bit2 = i32::from(spins[asset * 2 + 1] > 0);
        let level = bit1 * 2 + bit2;
        weights[asset] = f64::from(level) / 3.0;
    }

    // Normalize weights to sum to 1
    let sum: f64 = weights.iter().sum();
    if sum > 0.0 {
        for weight in &mut weights {
            *weight /= sum;
        }
    } else {
        // Equal weights if all zero
        weights.fill(0.25);
    }

    weights
}

fn generate_edge_list(num_nodes: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    for i in 0..num_nodes {
        for j in (i + 1)..num_nodes {
            edges.push((i, j));
        }
    }
    edges
}

fn calculate_average_path_length(edges: &[(usize, usize)], num_nodes: usize) -> f64 {
    // Simplified path length calculation
    // In practice, would use Floyd-Warshall or BFS
    if edges.is_empty() {
        return 1000.0; // Penalty for disconnected graph
    }

    let num_edges = edges.len() as f64;
    let max_edges = (num_nodes * (num_nodes - 1) / 2) as f64;

    // Approximate: higher connectivity = lower average path length
    2.0f64.mul_add(-(num_edges / max_edges), 3.0)
}

fn calculate_network_reliability(edges: &[(usize, usize)], num_nodes: usize) -> f64 {
    // Simplified reliability calculation based on connectivity
    let num_edges = edges.len() as f64;
    let min_edges_connected = (num_nodes - 1) as f64; // Spanning tree

    if num_edges < min_edges_connected {
        return 0.0; // Disconnected
    }

    // Reliability increases with redundancy
    ((num_edges - min_edges_connected) / min_edges_connected).min(2.0)
}
