#![allow(clippy::pedantic, clippy::unnecessary_wraps)]
//! Example demonstrating large-scale QUBO decomposition
//!
//! This example shows how to:
//! 1. Create large QUBO problems
//! 2. Apply different decomposition strategies
//! 3. Solve sub-problems efficiently
//! 4. Compare decomposition vs. direct solving

use quantrs2_anneal::{
    qubo::QuboBuilder,
    qubo_decomposition::{DecompositionConfig, DecompositionStrategy, QuboDecomposer},
    simulator::{AnnealingParams, QuantumAnnealingSimulator},
};
use scirs2_core::random::{thread_rng, Rng};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Large-Scale QUBO Decomposition Demo ===\n");

    // Example 1: Block decomposition
    println!("Example 1: Block Decomposition Strategy");
    block_decomposition_example()?;

    // Example 2: Spectral decomposition
    println!("\nExample 2: Spectral Decomposition Strategy");
    spectral_decomposition_example()?;

    // Example 3: Hierarchical decomposition
    println!("\nExample 3: Hierarchical Decomposition Strategy");
    hierarchical_decomposition_example()?;

    // Example 4: Clustering decomposition
    println!("\nExample 4: Clustering Decomposition Strategy");
    clustering_decomposition_example()?;

    // Example 5: Comparison with direct solving
    println!("\nExample 5: Decomposition vs Direct Solving");
    comparison_example()?;

    // Example 6: Large-scale portfolio optimization
    println!("\nExample 6: Large-Scale Portfolio Optimization");
    portfolio_optimization_example()?;

    Ok(())
}

fn block_decomposition_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a medium-sized QUBO problem
    let qubo = create_chain_qubo(50)?;

    let config = DecompositionConfig {
        strategy: DecompositionStrategy::Block {
            block_size: 12,
            overlap_size: 3,
        },
        max_subproblem_size: 15,
        min_subproblem_size: 5,
        refinement_iterations: 3,
        ..Default::default()
    };

    let decomposer = QuboDecomposer::new(config);

    let start = Instant::now();
    let result = decomposer.solve(&qubo)?;
    let solving_time = start.elapsed();

    println!("Block decomposition results:");
    println!("  Problem size: {} variables", qubo.num_variables);
    println!("  Number of sub-problems: {}", result.stats.num_subproblems);
    println!(
        "  Average sub-problem size: {:.1}",
        result.stats.avg_subproblem_size
    );
    println!(
        "  Sub-problem size range: [{}, {}]",
        result.stats.min_subproblem_size, result.stats.max_subproblem_size
    );
    println!("  Best objective value: {:.4}", result.objective_value);
    println!("  Total solving time: {solving_time:.2?}");
    println!(
        "  Decomposition time: {:.2?}",
        result.stats.decomposition_time
    );
    println!(
        "  Sub-problem solving time: {:.2?}",
        result.stats.solving_time
    );
    println!(
        "  Refinement iterations: {}",
        result.stats.refinement_iterations
    );
    println!("  Converged: {}", result.stats.converged);

    Ok(())
}

fn spectral_decomposition_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a grid-like QUBO problem
    let qubo = create_grid_qubo(8, 8)?;

    let config = DecompositionConfig {
        strategy: DecompositionStrategy::Spectral {
            num_partitions: 4,
            overlap_size: 2,
        },
        max_subproblem_size: 20,
        refinement_iterations: 5,
        ..Default::default()
    };

    let decomposer = QuboDecomposer::new(config);

    let start = Instant::now();
    let result = decomposer.solve(&qubo)?;
    let solving_time = start.elapsed();

    println!("Spectral decomposition results:");
    println!("  Problem size: {} variables", qubo.num_variables);
    println!("  Number of partitions: {}", result.stats.num_subproblems);
    println!("  Best objective value: {:.4}", result.objective_value);
    println!("  Total solving time: {solving_time:.2?}");
    println!(
        "  Convergence: {}",
        if result.stats.converged { "Yes" } else { "No" }
    );

    // Analyze sub-problem balance
    let sizes: Vec<usize> = result
        .sub_solutions
        .iter()
        .map(|sol| sol.values.len())
        .collect();
    let size_variance = calculate_variance(&sizes);
    println!("  Sub-problem size variance: {size_variance:.2}");

    Ok(())
}

fn hierarchical_decomposition_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a hierarchical structure problem
    let qubo = create_hierarchical_qubo(32)?;

    let config = DecompositionConfig {
        strategy: DecompositionStrategy::Hierarchical {
            min_partition_size: 8,
            max_depth: 3,
            overlap_size: 2,
        },
        refinement_iterations: 4,
        ..Default::default()
    };

    let decomposer = QuboDecomposer::new(config);

    let start = Instant::now();
    let result = decomposer.solve(&qubo)?;
    let solving_time = start.elapsed();

    println!("Hierarchical decomposition results:");
    println!("  Problem size: {} variables", qubo.num_variables);
    println!(
        "  Number of leaf sub-problems: {}",
        result.stats.num_subproblems
    );
    println!(
        "  Average sub-problem size: {:.1}",
        result.stats.avg_subproblem_size
    );
    println!("  Best objective value: {:.4}", result.objective_value);
    println!("  Total solving time: {solving_time:.2?}");

    Ok(())
}

fn clustering_decomposition_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a clustered problem structure
    let qubo = create_clustered_qubo(40, 4)?;

    let config = DecompositionConfig {
        strategy: DecompositionStrategy::Clustering {
            num_clusters: 4,
            strength_threshold: 0.5,
            overlap_size: 3,
        },
        max_subproblem_size: 15,
        refinement_iterations: 3,
        ..Default::default()
    };

    let decomposer = QuboDecomposer::new(config);

    let start = Instant::now();
    let result = decomposer.solve(&qubo)?;
    let solving_time = start.elapsed();

    println!("Clustering decomposition results:");
    println!("  Problem size: {} variables", qubo.num_variables);
    println!(
        "  Number of clusters found: {}",
        result.stats.num_subproblems
    );
    println!("  Best objective value: {:.4}", result.objective_value);
    println!("  Total solving time: {solving_time:.2?}");

    // Show cluster sizes
    println!("  Cluster sizes:");
    for (i, sub_sol) in result.sub_solutions.iter().enumerate() {
        println!("    Cluster {}: {} variables", i, sub_sol.values.len());
    }

    Ok(())
}

fn comparison_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a problem that can be solved both ways
    let qubo = create_chain_qubo(30)?;

    // Solve with decomposition
    let decomp_config = DecompositionConfig {
        strategy: DecompositionStrategy::Block {
            block_size: 10,
            overlap_size: 2,
        },
        refinement_iterations: 3,
        ..Default::default()
    };

    let decomposer = QuboDecomposer::new(decomp_config);

    let start = Instant::now();
    let decomp_result = decomposer.solve(&qubo)?;
    let decomp_time = start.elapsed();

    // Solve directly
    let (ising, offset) = qubo.to_ising();
    let direct_params = AnnealingParams {
        num_sweeps: 5000,
        num_repetitions: 10,
        ..Default::default()
    };

    let start = Instant::now();
    let mut direct_solver = QuantumAnnealingSimulator::new(direct_params)?;
    let direct_result = direct_solver.solve(&ising)?;
    let direct_time = start.elapsed();

    let direct_objective = direct_result.best_energy + offset;

    println!("Comparison results:");
    println!("Decomposition approach:");
    println!("  Objective value: {:.4}", decomp_result.objective_value);
    println!("  Solving time: {decomp_time:.2?}");
    println!("  Sub-problems: {}", decomp_result.stats.num_subproblems);

    println!("Direct approach:");
    println!("  Objective value: {direct_objective:.4}");
    println!("  Solving time: {direct_time:.2?}");
    println!("  Repetitions: {}", direct_result.repetitions);

    let objective_diff = (decomp_result.objective_value - direct_objective).abs();
    let speedup = direct_time.as_secs_f64() / decomp_time.as_secs_f64();

    println!("Comparison:");
    println!("  Objective difference: {objective_diff:.6}");
    println!("  Speedup: {speedup:.2}x");

    Ok(())
}

fn portfolio_optimization_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a large portfolio optimization problem
    let qubo = create_portfolio_qubo(100, 20)?;

    let config = DecompositionConfig {
        strategy: DecompositionStrategy::Spectral {
            num_partitions: 8,
            overlap_size: 3,
        },
        max_subproblem_size: 25,
        refinement_iterations: 5,
        convergence_tolerance: 1e-4,
        ..Default::default()
    };

    let decomposer = QuboDecomposer::new(config);

    let start = Instant::now();
    let result = decomposer.solve(&qubo)?;
    let solving_time = start.elapsed();

    println!("Portfolio optimization results:");
    println!("  Assets: {}", qubo.num_variables);
    println!("  Portfolio objective: {:.4}", result.objective_value);
    println!(
        "  Assets selected: {}",
        result.variable_values.values().filter(|&&v| v).count()
    );
    println!("  Total solving time: {solving_time:.2?}");
    println!("  Sub-portfolios: {}", result.stats.num_subproblems);

    // Analyze the final portfolio
    let selected_assets: Vec<usize> = result
        .variable_values
        .iter()
        .filter(|(_, &value)| value)
        .map(|(&idx, _)| idx)
        .collect();

    println!(
        "  Selected assets: {:?}",
        &selected_assets[..std::cmp::min(10, selected_assets.len())]
    );
    if selected_assets.len() > 10 {
        println!("    ... and {} more", selected_assets.len() - 10);
    }

    Ok(())
}

/// Create a chain-structured QUBO problem
fn create_chain_qubo(
    n: usize,
) -> Result<quantrs2_anneal::ising::QuboModel, Box<dyn std::error::Error>> {
    let mut builder = QuboBuilder::new();

    // Create variables
    let vars: Vec<_> = (0..n)
        .map(|i| builder.add_variable(format!("x{i}")).unwrap())
        .collect();

    // Add linear terms (random costs)
    for i in 0..n {
        let cost = thread_rng().gen::<f64>().mul_add(2.0, -1.0); // Range [-1, 1]
        builder.set_linear_term(&vars[i], cost)?;
    }

    // Add chain interactions
    for i in 0..(n - 1) {
        let interaction = -thread_rng().gen::<f64>().mul_add(2.0, 1.0); // Range [-3, -1]
        builder.set_quadratic_term(&vars[i], &vars[i + 1], interaction)?;
    }

    // Add some long-range interactions
    for i in 0..n {
        for j in (i + 3)..std::cmp::min(i + 6, n) {
            if thread_rng().gen::<f64>() < 0.3 {
                let interaction = (thread_rng().gen::<f64>() - 0.5) * 0.5;
                builder.set_quadratic_term(&vars[i], &vars[j], interaction)?;
            }
        }
    }

    Ok(builder.build().to_qubo_model())
}

/// Create a grid-structured QUBO problem
fn create_grid_qubo(
    rows: usize,
    cols: usize,
) -> Result<quantrs2_anneal::ising::QuboModel, Box<dyn std::error::Error>> {
    let n = rows * cols;
    let mut builder = QuboBuilder::new();

    // Create variables
    let vars: Vec<_> = (0..n)
        .map(|i| builder.add_variable(format!("x{i}")).unwrap())
        .collect();

    // Add linear terms
    for i in 0..n {
        let cost = thread_rng().gen::<f64>() - 0.5;
        builder.set_linear_term(&vars[i], cost)?;
    }

    // Add grid interactions
    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;

            // Right neighbor
            if col + 1 < cols {
                let neighbor = row * cols + (col + 1);
                let interaction = -(thread_rng().gen::<f64>() + 0.5);
                builder.set_quadratic_term(&vars[idx], &vars[neighbor], interaction)?;
            }

            // Bottom neighbor
            if row + 1 < rows {
                let neighbor = (row + 1) * cols + col;
                let interaction = -(thread_rng().gen::<f64>() + 0.5);
                builder.set_quadratic_term(&vars[idx], &vars[neighbor], interaction)?;
            }
        }
    }

    Ok(builder.build().to_qubo_model())
}

/// Create a hierarchical QUBO problem
fn create_hierarchical_qubo(
    n: usize,
) -> Result<quantrs2_anneal::ising::QuboModel, Box<dyn std::error::Error>> {
    let mut builder = QuboBuilder::new();

    // Create variables
    let vars: Vec<_> = (0..n)
        .map(|i| builder.add_variable(format!("x{i}")).unwrap())
        .collect();

    // Add linear terms
    for i in 0..n {
        builder.set_linear_term(&vars[i], thread_rng().gen::<f64>() - 0.5)?;
    }

    // Add hierarchical structure
    let block_size = 4;
    let num_blocks = n / block_size;

    // Intra-block interactions (strong)
    for block in 0..num_blocks {
        let start = block * block_size;
        let end = std::cmp::min(start + block_size, n);

        for i in start..end {
            for j in (i + 1)..end {
                let interaction = -thread_rng().gen::<f64>().mul_add(2.0, 1.0);
                builder.set_quadratic_term(&vars[i], &vars[j], interaction)?;
            }
        }
    }

    // Inter-block interactions (weaker)
    for block1 in 0..num_blocks {
        for block2 in (block1 + 1)..num_blocks {
            if thread_rng().gen::<f64>() < 0.3 {
                let i = block1 * block_size + thread_rng().gen_range(0..block_size);
                let j = block2 * block_size + thread_rng().gen_range(0..block_size);
                if i < n && j < n {
                    let interaction = (thread_rng().gen::<f64>() - 0.5) * 0.5;
                    builder.set_quadratic_term(&vars[i], &vars[j], interaction)?;
                }
            }
        }
    }

    Ok(builder.build().to_qubo_model())
}

/// Create a clustered QUBO problem
fn create_clustered_qubo(
    n: usize,
    num_clusters: usize,
) -> Result<quantrs2_anneal::ising::QuboModel, Box<dyn std::error::Error>> {
    let mut builder = QuboBuilder::new();

    // Create variables
    let vars: Vec<_> = (0..n)
        .map(|i| builder.add_variable(format!("x{i}")).unwrap())
        .collect();

    // Add linear terms
    for i in 0..n {
        builder.set_linear_term(&vars[i], thread_rng().gen::<f64>() - 0.5)?;
    }

    let cluster_size = n / num_clusters;

    // Strong intra-cluster interactions
    for cluster in 0..num_clusters {
        let start = cluster * cluster_size;
        let end = std::cmp::min(start + cluster_size, n);

        for i in start..end {
            for j in (i + 1)..end {
                if thread_rng().gen::<f64>() < 0.7 {
                    let interaction = -thread_rng().gen::<f64>().mul_add(1.5, 0.5);
                    builder.set_quadratic_term(&vars[i], &vars[j], interaction)?;
                }
            }
        }
    }

    // Weak inter-cluster interactions
    for i in 0..n {
        for j in (i + 1)..n {
            let cluster_i = i / cluster_size;
            let cluster_j = j / cluster_size;

            if cluster_i != cluster_j && thread_rng().gen::<f64>() < 0.1 {
                let interaction = (thread_rng().gen::<f64>() - 0.5) * 0.3;
                builder.set_quadratic_term(&vars[i], &vars[j], interaction)?;
            }
        }
    }

    Ok(builder.build().to_qubo_model())
}

/// Create a portfolio optimization QUBO
fn create_portfolio_qubo(
    n_assets: usize,
    target_count: usize,
) -> Result<quantrs2_anneal::ising::QuboModel, Box<dyn std::error::Error>> {
    let mut builder = QuboBuilder::new();

    // Create asset variables
    let assets: Vec<_> = (0..n_assets)
        .map(|i| builder.add_variable(format!("asset_{i}")).unwrap())
        .collect();

    // Random returns and risks
    let returns: Vec<f64> = (0..n_assets)
        .map(|_| thread_rng().gen::<f64>() * 0.2)
        .collect();
    let risks: Vec<f64> = (0..n_assets)
        .map(|_| thread_rng().gen::<f64>() * 0.1)
        .collect();

    // Objective: maximize returns - risk penalty
    for i in 0..n_assets {
        let coefficient = risks[i].mul_add(2.0, -returns[i]); // Risk aversion
        builder.set_linear_term(&assets[i], coefficient)?;
    }

    // Correlation penalties
    for i in 0..n_assets {
        for j in (i + 1)..n_assets {
            if thread_rng().gen::<f64>() < 0.2 {
                // 20% of pairs are correlated
                let correlation = thread_rng().gen::<f64>() * 0.1;
                builder.set_quadratic_term(&assets[i], &assets[j], correlation)?;
            }
        }
    }

    // Constraint: select exactly target_count assets
    // Implement (sum(x_i) - target_count)^2 constraint manually
    let constraint_weight = 10.0;

    // Linear terms: add -2 * target_count * weight to existing terms
    for (i, asset) in assets.iter().enumerate() {
        // Get existing coefficient and add constraint term
        let existing_coeff = risks[i].mul_add(2.0, -returns[i]); // From above objective
        let constraint_term = -2.0 * target_count as f64 * constraint_weight;
        builder.set_linear_term(asset, existing_coeff + constraint_term)?;
    }

    // Quadratic terms: 2 * weight for each pair of variables
    for i in 0..assets.len() {
        for j in (i + 1)..assets.len() {
            let current_quad = 0.0; // Start fresh since we set them above
            builder.set_quadratic_term(
                &assets[i],
                &assets[j],
                2.0f64.mul_add(constraint_weight, current_quad),
            )?;
        }
    }

    // Constant term: target_count^2 * weight (this affects the offset but doesn't change the optimization)

    Ok(builder.build().to_qubo_model())
}

/// Calculate variance of a vector
fn calculate_variance(values: &[usize]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mean = values.iter().sum::<usize>() as f64 / values.len() as f64;
    let variance = values
        .iter()
        .map(|&x| (x as f64 - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;

    variance
}
