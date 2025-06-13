//! Example demonstrating Fujitsu Digital Annealer usage
//!
//! This example shows how to:
//! 1. Connect to Fujitsu Digital Annealer hardware
//! 2. Submit optimization problems
//! 3. Use guidance mode for improved solutions
//! 4. Compare with local simulation

#[cfg(feature = "fujitsu")]
use quantrs2_anneal::{
    fujitsu::{FujitsuAnnealingParams, FujitsuClient, GuidanceConfig},
    ising::IsingModel,
    qubo::QuboBuilder,
    simulator::{AnnealingParams, QuantumAnnealingSimulator},
};
use std::time::Instant;

#[cfg(feature = "fujitsu")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Fujitsu Digital Annealer Demo ===\n");

    // Check if Fujitsu is available
    if !quantrs2_anneal::is_fujitsu_available() {
        println!("Fujitsu Digital Annealer not available.");
        println!("Please set FUJITSU_DAU_API_KEY and FUJITSU_DAU_ENDPOINT environment variables.");
        return Ok(());
    }

    // Create Fujitsu client
    let client = quantrs2_anneal::fujitsu::from_env()?;

    // Example 1: Solve a MaxCut problem
    println!("Example 1: MaxCut Problem");
    let model = create_maxcut_instance(20)?;

    // Configure Fujitsu parameters
    let mut params = FujitsuAnnealingParams::default();
    params.number_iterations = Some(20_000_000);
    params.number_runs = Some(32);
    params.time_limit_sec = Some(30);

    // Solve on Fujitsu hardware
    let start = Instant::now();
    let fujitsu_result = client.solve_ising(&model, params.clone()).await?;
    let fujitsu_time = start.elapsed();

    println!("Fujitsu Digital Annealer results:");
    println!("  Best energy: {:.4}", fujitsu_result.best_energy);
    println!("  Time: {:.2?}", fujitsu_time);
    println!("  Info: {}", fujitsu_result.info);

    // Compare with local simulation
    println!("\nLocal simulation for comparison:");
    let sim_params = AnnealingParams {
        num_sweeps: 10000,
        num_repetitions: 10,
        ..Default::default()
    };
    let mut simulator = QuantumAnnealingSimulator::new(sim_params)?;

    let start = Instant::now();
    let sim_result = simulator.solve(&model)?;
    let sim_time = start.elapsed();

    println!("  Best energy: {:.4}", sim_result.best_energy);
    println!("  Time: {:.2?}", sim_time);

    // Calculate speedup
    let speedup = sim_time.as_secs_f64() / fujitsu_time.as_secs_f64();
    println!("\nFujitsu speedup: {:.1}x", speedup);

    // Example 2: Use guidance mode
    println!(
        "\n{}Example 2: Guidance Mode{}",
        "=".repeat(10),
        "=".repeat(10)
    );

    // Start with a known good solution
    let initial_solution = sim_result.best_spins.clone();

    // Configure guidance
    params.guidance_config = Some(GuidanceConfig {
        enabled: true,
        initial_solution: Some(initial_solution),
        guidance_strength: 0.5,
    });

    let start = Instant::now();
    let guided_result = client.solve_ising(&model, params).await?;
    let guided_time = start.elapsed();

    println!("Guided optimization results:");
    println!("  Best energy: {:.4}", guided_result.best_energy);
    println!("  Time: {:.2?}", guided_time);
    println!(
        "  Improvement: {:.4}",
        sim_result.best_energy - guided_result.best_energy
    );

    // Example 3: Solve a QUBO problem directly
    println!(
        "\n{}Example 3: QUBO Problem{}",
        "=".repeat(10),
        "=".repeat(10)
    );

    let qubo = create_portfolio_optimization(15)?;

    let params = FujitsuAnnealingParams {
        number_iterations: Some(30_000_000),
        number_runs: Some(64),
        temperature_mode: Some("ADAPTIVE".to_string()),
        ..Default::default()
    };

    let start = Instant::now();
    let qubo_solution = client.solve_qubo(&qubo, params).await?;
    let qubo_time = start.elapsed();

    println!("Portfolio optimization results:");
    println!("  Solution: {:?}", qubo_solution);
    println!("  Time: {:.2?}", qubo_time);

    // Calculate portfolio statistics
    let selected_count = qubo_solution.iter().filter(|&&x| x).count();
    println!(
        "  Assets selected: {}/{}",
        selected_count,
        qubo_solution.len()
    );

    Ok(())
}

#[cfg(feature = "fujitsu")]
fn create_maxcut_instance(n: usize) -> Result<IsingModel, Box<dyn std::error::Error>> {
    let mut model = IsingModel::new(n);

    // Create a random graph with negative couplings for MaxCut
    for i in 0..n {
        for j in (i + 1)..n {
            if rand::random::<f64>() < 0.4 {
                // 40% edge density
                model.set_coupling(i, j, -1.0)?;
            }
        }
    }

    Ok(model)
}

#[cfg(feature = "fujitsu")]
fn create_portfolio_optimization(
    n_assets: usize,
) -> Result<Box<dyn quantrs2_anneal::qubo::QuboFormulation>, Box<dyn std::error::Error>> {
    let mut builder = QuboBuilder::new();

    // Generate random returns and risks
    let returns: Vec<f64> = (0..n_assets).map(|_| rand::random::<f64>() * 0.1).collect();
    let risks: Vec<f64> = (0..n_assets)
        .map(|_| rand::random::<f64>() * 0.05)
        .collect();

    // Decision variables: x[i] = 1 if asset i is selected
    let vars: Vec<_> = (0..n_assets)
        .map(|i| builder.add_variable(format!("asset_{}", i)).unwrap())
        .collect();

    // Objective: maximize returns - lambda * risks
    let lambda = 2.0; // Risk aversion parameter

    for i in 0..n_assets {
        builder.set_linear_term(&vars[i], -returns[i] + lambda * risks[i])?;
    }

    // Constraint: select exactly k assets
    let k = n_assets / 3;
    builder.set_constraint_weight(10.0)?;
    builder.constrain_equal(&vars, k as f64)?;

    // Add correlation penalties between assets
    for i in 0..n_assets {
        for j in (i + 1)..n_assets {
            if rand::random::<f64>() < 0.2 {
                // 20% correlation
                let correlation = rand::random::<f64>() * 0.1;
                builder.set_quadratic_term(&vars[i], &vars[j], lambda * correlation)?;
            }
        }
    }

    Ok(Box::new(builder.build()))
}

#[cfg(not(feature = "fujitsu"))]
fn main() {
    println!("This example requires the 'fujitsu' feature to be enabled.");
    println!("Run with: cargo run --example fujitsu_example --features fujitsu");
}
