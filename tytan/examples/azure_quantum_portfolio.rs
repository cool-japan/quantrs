//! Azure Quantum Example: Portfolio Optimization
//!
//! This example demonstrates how to use the Azure Quantum sampler
//! to solve a portfolio optimization problem using various solvers.
//!
//! Portfolio Optimization: Select a subset of assets to maximize return
//! while minimizing risk, subject to budget constraints.

use quantrs2_tytan::sampler::hardware::{AzureQuantumConfig, AzureQuantumSampler, AzureSolver};
use quantrs2_tytan::sampler::Sampler;
use scirs2_core::ndarray::{Array, Array2};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Azure Quantum Portfolio Optimization Example ===\n");

    // Define portfolio parameters
    let num_assets = 8;
    let budget = 4; // Select 4 assets out of 8

    // Expected returns (higher is better)
    let returns = vec![0.12, 0.15, 0.10, 0.18, 0.14, 0.11, 0.16, 0.13];

    // Risk correlation matrix (lower is better)
    let risk_correlations = vec![
        vec![1.0, 0.3, 0.2, 0.1, 0.4, 0.2, 0.3, 0.1],
        vec![0.3, 1.0, 0.4, 0.2, 0.3, 0.1, 0.2, 0.3],
        vec![0.2, 0.4, 1.0, 0.3, 0.2, 0.4, 0.1, 0.2],
        vec![0.1, 0.2, 0.3, 1.0, 0.1, 0.3, 0.4, 0.5],
        vec![0.4, 0.3, 0.2, 0.1, 1.0, 0.2, 0.3, 0.2],
        vec![0.2, 0.1, 0.4, 0.3, 0.2, 1.0, 0.1, 0.3],
        vec![0.3, 0.2, 0.1, 0.4, 0.3, 0.1, 1.0, 0.2],
        vec![0.1, 0.3, 0.2, 0.5, 0.2, 0.3, 0.2, 1.0],
    ];

    println!("Portfolio Parameters:");
    println!("  Number of assets: {}", num_assets);
    println!("  Budget (assets to select): {}", budget);
    println!("  Expected returns: {:?}\n", returns);

    // Convert to QUBO formulation
    // Objective: Maximize return - risk_penalty * correlation
    // Constraint: Select exactly 'budget' assets
    let risk_weight = 0.5;
    let constraint_penalty = 10.0;

    let mut qubo_matrix = Array2::<f64>::zeros((num_assets, num_assets));

    // Add return terms (negative because we minimize in QUBO)
    for i in 0..num_assets {
        qubo_matrix[[i, i]] -= returns[i];
    }

    // Add risk correlation terms
    for i in 0..num_assets {
        for j in i..num_assets {
            if i == j {
                qubo_matrix[[i, i]] += risk_weight * risk_correlations[i][i];
            } else {
                qubo_matrix[[i, j]] += risk_weight * risk_correlations[i][j];
            }
        }
    }

    // Add budget constraint: (sum(x_i) - budget)^2
    for i in 0..num_assets {
        qubo_matrix[[i, i]] += constraint_penalty * (1.0 - 2.0 * budget as f64);
        for j in (i + 1)..num_assets {
            qubo_matrix[[i, j]] += 2.0 * constraint_penalty;
        }
    }

    // Create variable mapping
    let var_map: HashMap<String, usize> = (0..num_assets)
        .map(|i| (format!("asset_{}", i), i))
        .collect();

    println!("QUBO formulation complete.\n");

    // Example 1: Using Azure QIO Simulated Annealing
    println!("--- Example 1: Azure QIO Simulated Annealing ---");
    let sa_sampler = AzureQuantumSampler::with_workspace(
        "YOUR_SUBSCRIPTION_ID",
        "YOUR_RESOURCE_GROUP",
        "YOUR_WORKSPACE_NAME",
    )
    .with_solver(AzureSolver::SimulatedAnnealing)
    .with_timeout(60);

    println!("Running Azure QIO Simulated Annealing...");
    let sa_results = sa_sampler.run_qubo(&(qubo_matrix.clone(), var_map.clone()), 100)?;

    println!("Top 3 portfolios from Simulated Annealing:");
    for (idx, result) in sa_results.iter().take(3).enumerate() {
        let selected_assets = get_selected_assets(&result.assignments);
        let portfolio_return = calculate_return(&selected_assets, &returns);
        let portfolio_risk = calculate_risk(&selected_assets, &risk_correlations);

        println!("  {}. Energy: {:.4}", idx + 1, result.energy);
        println!("     Selected assets: {:?}", selected_assets);
        println!("     Expected return: {:.2}%", portfolio_return * 100.0);
        println!("     Portfolio risk: {:.4}", portfolio_risk);
    }
    println!();

    // Example 2: Using Azure QIO Parallel Tempering
    println!("--- Example 2: Azure QIO Parallel Tempering ---");
    let pt_sampler = AzureQuantumSampler::with_workspace(
        "YOUR_SUBSCRIPTION_ID",
        "YOUR_RESOURCE_GROUP",
        "YOUR_WORKSPACE_NAME",
    )
    .with_solver(AzureSolver::ParallelTempering)
    .with_timeout(120)
    .with_param("sweeps".to_string(), "1000".to_string());

    println!("Configuration:");
    println!("  - Solver: Parallel Tempering");
    println!("  - Timeout: 120 seconds");
    println!("  - Additional parameter: sweeps=1000");
    println!();

    // Example 3: Using Azure QIO Population Annealing
    println!("--- Example 3: Azure QIO Population Annealing ---");
    let pop_sampler = AzureQuantumSampler::with_workspace(
        "YOUR_SUBSCRIPTION_ID",
        "YOUR_RESOURCE_GROUP",
        "YOUR_WORKSPACE_NAME",
    )
    .with_solver(AzureSolver::PopulationAnnealing)
    .with_timeout(180);

    println!("Configuration:");
    println!("  - Solver: Population Annealing");
    println!("  - Timeout: 180 seconds");
    println!("  - Best for: Large-scale optimization problems");
    println!();

    // Example 4: Using IonQ quantum computer via Azure
    println!("--- Example 4: IonQ Quantum Computer (via Azure) ---");
    let ionq_sampler = AzureQuantumSampler::with_workspace(
        "YOUR_SUBSCRIPTION_ID",
        "YOUR_RESOURCE_GROUP",
        "YOUR_WORKSPACE_NAME",
    )
    .with_solver(AzureSolver::IonQ)
    .with_timeout(300);

    println!("Configuration:");
    println!("  - Solver: IonQ Quantum Computer");
    println!("  - Note: Requires quantum credits");
    println!("  - Best for: Small to medium QAOA problems");
    println!();

    // Compare solvers
    println!("--- Solver Comparison ---");
    println!("Solver Characteristics:");
    println!();
    println!("1. Simulated Annealing:");
    println!("   - Fast for small to medium problems");
    println!("   - Good quality solutions");
    println!("   - No quantum hardware required");
    println!();
    println!("2. Parallel Tempering:");
    println!("   - Better exploration of solution space");
    println!("   - Good for problems with many local minima");
    println!("   - Scales well to larger problems");
    println!();
    println!("3. Population Annealing:");
    println!("   - Best for very large problems");
    println!("   - Maintains population diversity");
    println!("   - Excellent for constraint satisfaction");
    println!();
    println!("4. Tabu Search:");
    println!("   - Memory-based local search");
    println!("   - Fast convergence");
    println!("   - Good for structured problems");
    println!();
    println!("5. Substrate Monte Carlo:");
    println!("   - Quantum-inspired algorithm");
    println!("   - Good for dense problems");
    println!("   - Efficient parallelization");
    println!();
    println!("6. IonQ/Quantinuum/Rigetti:");
    println!("   - True quantum hardware");
    println!("   - Quantum advantage for specific problems");
    println!("   - Requires quantum credits");

    println!("\n--- Setup Instructions ---");
    println!("1. Create Azure account: https://azure.microsoft.com/");
    println!("2. Create Quantum workspace: https://portal.azure.com/");
    println!("3. Add quantum providers (IonQ, Quantinuum, Rigetti, Microsoft QIO)");
    println!("4. Get subscription ID, resource group, and workspace name");
    println!("5. Configure Azure CLI authentication or use connection strings");

    Ok(())
}

/// Get list of selected assets from assignments
fn get_selected_assets(assignments: &HashMap<String, bool>) -> Vec<usize> {
    assignments
        .iter()
        .filter(|(_, &selected)| selected)
        .filter_map(|(name, _)| {
            name.strip_prefix("asset_")
                .and_then(|s| s.parse::<usize>().ok())
        })
        .collect()
}

/// Calculate portfolio return
fn calculate_return(selected_assets: &[usize], returns: &[f64]) -> f64 {
    selected_assets.iter().map(|&i| returns[i]).sum::<f64>() / selected_assets.len() as f64
}

/// Calculate portfolio risk
fn calculate_risk(selected_assets: &[usize], correlations: &[Vec<f64>]) -> f64 {
    let n = selected_assets.len();
    if n == 0 {
        return 0.0;
    }

    let mut total_risk = 0.0;
    for &i in selected_assets {
        for &j in selected_assets {
            total_risk += correlations[i][j];
        }
    }
    total_risk / (n * n) as f64
}
