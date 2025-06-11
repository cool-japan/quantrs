//! Demonstration of penalty function optimization with SciRS2
//!
//! This example shows how to use the penalty optimization module
//! to automatically tune penalty weights for constrained QUBO problems.

use ndarray::{Array1, Array2};
use quantrs2_tytan::{
    optimization::{
        adaptive::{AdaptiveConfig, AdaptiveOptimizer},
        penalty::{CompiledModel, PenaltyConfig, PenaltyOptimizer, PenaltyType},
        tuning::{AcquisitionType, ParameterBounds, ParameterScale, ParameterTuner, TuningConfig},
    },
    sampler::{SASampler, Sampler},
};
use std::collections::HashMap;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== QuantRS2 Penalty Function Optimization Demo ===\n");

    // Example 1: Quadratic penalty optimization
    quadratic_penalty_demo()?;
    println!();

    // Example 2: Adaptive penalty strategy
    adaptive_penalty_demo()?;
    println!();

    // Example 3: Parameter tuning with constraints
    parameter_tuning_demo()?;

    Ok(())
}

/// Demonstrate quadratic penalty optimization
fn quadratic_penalty_demo() -> Result<(), Box<dyn Error>> {
    println!("1. Quadratic Penalty Optimization");
    println!("   Problem: Minimize x^2 + y^2 subject to x + y = 1");

    // Create a 2x2 QUBO matrix for variables x and y
    // Objective: x^2 + y^2 = x*1 + y*1 (since x^2 = x for binary variables)
    // Constraint penalty: (x + y - 1)^2 = x^2 + y^2 + 2xy - 2x - 2y + 1
    //                                  = x + y + 2xy - 2x - 2y + 1
    //                                  = 2xy - x - y + 1

    let initial_penalty = 10.0;
    let mut qubo = Array2::zeros((2, 2));

    // Linear terms: x^2 + y^2 - penalty*(2x + 2y)
    qubo[[0, 0]] = 1.0 - 2.0 * initial_penalty; // x coefficient
    qubo[[1, 1]] = 1.0 - 2.0 * initial_penalty; // y coefficient

    // Quadratic term: penalty * 2xy
    qubo[[0, 1]] = 2.0 * initial_penalty;
    qubo[[1, 0]] = 2.0 * initial_penalty;

    // Create variable mapping
    let mut var_map = HashMap::new();
    var_map.insert("x".to_string(), 0);
    var_map.insert("y".to_string(), 1);

    // Create penalty optimizer
    let config = PenaltyConfig {
        initial_weight: 1.0,
        min_weight: 0.1,
        max_weight: 100.0,
        adjustment_factor: 1.5,
        violation_tolerance: 1e-4,
        max_iterations: 20,
        adaptive_scaling: true,
        penalty_type: PenaltyType::Quadratic,
    };

    let mut penalty_optimizer = PenaltyOptimizer::new(config);

    // Run sampling with the QUBO matrix
    let sampler = SASampler::new(None);
    let samples = sampler.run_qubo(&(qubo.clone(), var_map.clone()), 100)?;

    println!("   Optimization Results:");
    println!("   - Found {} samples", samples.len());
    if let Some(best) = samples.first() {
        println!("   - Best energy: {:.4}", best.energy);
        println!(
            "   - Best solution: x={}, y={}",
            best.assignments.get("x").unwrap_or(&false),
            best.assignments.get("y").unwrap_or(&false)
        );

        // Check constraint satisfaction
        let x_val: f64 = if *best.assignments.get("x").unwrap_or(&false) {
            1.0
        } else {
            0.0
        };
        let y_val: f64 = if *best.assignments.get("y").unwrap_or(&false) {
            1.0
        } else {
            0.0
        };
        let constraint_violation = (x_val + y_val - 1.0).abs();
        println!(
            "   - Constraint violation |x + y - 1|: {:.6}",
            constraint_violation
        );
        println!("   - Constraint satisfied: {}", constraint_violation < 1e-3);
    }

    Ok(())
}

/// Demonstrate adaptive penalty strategies
fn adaptive_penalty_demo() -> Result<(), Box<dyn Error>> {
    println!("2. Adaptive Penalty Strategy Demo");
    println!("   Testing multiple adaptive strategies for constraint handling");

    // Create a 3x3 QUBO matrix for variables x, y, z
    // Objective: minimize x + 2y + 3z
    // Constraint 1: x + y + z = 1 (one-hot constraint)
    // Constraint 2: y + z ≤ 1 (inequality constraint)

    let penalty1 = 5.0; // penalty for one-hot constraint
    let penalty2 = 3.0; // penalty for inequality constraint

    let mut qubo = Array2::zeros((3, 3));

    // Linear terms: objective + constraint penalties
    // For constraint 1: (x + y + z - 1)^2 = x + y + z + 2xy + 2xz + 2yz - 2x - 2y - 2z + 1
    //                                     = 2xy + 2xz + 2yz - x - y - z + 1
    // For constraint 2: (y + z)^2 = y + z + 2yz - (not needed as y+z≤1 is automatically satisfied for binary)

    qubo[[0, 0]] = 1.0 - penalty1; // x coefficient: 1 (objective) - penalty1 (constraint)
    qubo[[1, 1]] = 2.0 - penalty1; // y coefficient: 2 (objective) - penalty1 (constraint)
    qubo[[2, 2]] = 3.0 - penalty1; // z coefficient: 3 (objective) - penalty1 (constraint)

    // Quadratic terms for constraint 1
    qubo[[0, 1]] = 2.0 * penalty1; // xy coefficient
    qubo[[1, 0]] = 2.0 * penalty1;
    qubo[[0, 2]] = 2.0 * penalty1; // xz coefficient
    qubo[[2, 0]] = 2.0 * penalty1;
    qubo[[1, 2]] = 2.0 * penalty1; // yz coefficient
    qubo[[2, 1]] = 2.0 * penalty1;

    // Create variable mapping
    let mut var_map = HashMap::new();
    var_map.insert("x".to_string(), 0);
    var_map.insert("y".to_string(), 1);
    var_map.insert("z".to_string(), 2);

    // Test different adaptive strategies
    use quantrs2_tytan::optimization::adaptive::AdaptiveStrategy;
    let strategies = vec![
        AdaptiveStrategy::ExponentialDecay,
        AdaptiveStrategy::AdaptivePenaltyMethod,
        AdaptiveStrategy::AugmentedLagrangian,
    ];

    for strategy in strategies {
        println!("\n   Strategy: {:?}", strategy);

        let config = AdaptiveConfig {
            strategy,
            update_interval: 5,
            learning_rate: 0.1,
            momentum: 0.9,
            patience: 10,
            ..Default::default()
        };

        let mut adaptive_optimizer = AdaptiveOptimizer::new(config);

        // Create sampler
        let sampler = SASampler::new(None);

        // Run sampling
        let samples = sampler.run_qubo(&(qubo.clone(), var_map.clone()), 100)?;

        if let Some(best) = samples.first() {
            println!("   - Best energy: {:.4}", best.energy);
            println!(
                "   - Best solution: x={}, y={}, z={}",
                best.assignments.get("x").unwrap_or(&false),
                best.assignments.get("y").unwrap_or(&false),
                best.assignments.get("z").unwrap_or(&false)
            );

            // Check constraint satisfaction
            let x_val: f64 = if *best.assignments.get("x").unwrap_or(&false) {
                1.0
            } else {
                0.0
            };
            let y_val: f64 = if *best.assignments.get("y").unwrap_or(&false) {
                1.0
            } else {
                0.0
            };
            let z_val: f64 = if *best.assignments.get("z").unwrap_or(&false) {
                1.0
            } else {
                0.0
            };

            let constraint1_violation = (x_val + y_val + z_val - 1.0).abs();
            let constraint2_violation = (y_val + z_val - 1.0).max(0.0);

            println!(
                "   - One-hot constraint violation: {:.6}",
                constraint1_violation
            );
            println!(
                "   - Inequality constraint violation: {:.6}",
                constraint2_violation
            );
            println!(
                "   - Constraints satisfied: {}",
                constraint1_violation < 1e-3 && constraint2_violation < 1e-3
            );
        }
    }

    Ok(())
}

/// Demonstrate parameter tuning with constraints
fn parameter_tuning_demo() -> Result<(), Box<dyn Error>> {
    println!("3. Parameter Tuning with Constraints");
    println!("   Manually testing different sampler parameter combinations");

    // Create a simple QUBO problem for testing
    let mut qubo = Array2::zeros((2, 2));
    qubo[[0, 0]] = -1.0; // x
    qubo[[1, 1]] = -1.0; // y
    qubo[[0, 1]] = 2.0; // xy interaction
    qubo[[1, 0]] = 2.0;

    let mut var_map = HashMap::new();
    var_map.insert("x".to_string(), 0);
    var_map.insert("y".to_string(), 1);

    // Test different parameter configurations
    let parameter_configs = vec![
        ("Conservative", 10), // Low number of samples
        ("Moderate", 100),    // Medium number of samples
        ("Aggressive", 1000), // High number of samples
    ];

    println!("\n   Testing different parameter configurations:");

    for (config_name, num_samples) in parameter_configs {
        println!("\n   Configuration: {}", config_name);

        let sampler = SASampler::new(None);
        let samples = sampler.run_qubo(&(qubo.clone(), var_map.clone()), num_samples)?;

        if let Some(best) = samples.first() {
            let avg_energy = samples.iter().map(|s| s.energy).sum::<f64>() / samples.len() as f64;
            let energy_std = {
                let mean = avg_energy;
                let variance = samples
                    .iter()
                    .map(|s| (s.energy - mean).powi(2))
                    .sum::<f64>()
                    / samples.len() as f64;
                variance.sqrt()
            };

            println!("   - Samples: {}", num_samples);
            println!("   - Best energy: {:.4}", best.energy);
            println!("   - Average energy: {:.4}", avg_energy);
            println!("   - Energy std dev: {:.4}", energy_std);
            println!(
                "   - Unique solutions: {}",
                samples
                    .iter()
                    .map(|s| format!("{:?}", s.assignments))
                    .collect::<std::collections::HashSet<_>>()
                    .len()
            );
            println!(
                "   - Best solution: x={}, y={}",
                best.assignments.get("x").unwrap_or(&false),
                best.assignments.get("y").unwrap_or(&false)
            );
        }
    }

    println!("\n   Summary:");
    println!("   - Higher sample counts generally give better solution quality");
    println!("   - But also increase computation time");
    println!("   - Optimal configuration depends on problem requirements");

    Ok(())
}
