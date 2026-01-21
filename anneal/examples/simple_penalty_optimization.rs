#![allow(
    clippy::pedantic,
    clippy::unnecessary_wraps,
    clippy::needless_range_loop,
    clippy::useless_vec,
    clippy::needless_collect,
    clippy::too_many_arguments
)]
//! Simple Penalty Optimization Example
//!
//! This example demonstrates penalty optimization for quantum annealing
//! with a manually created embedding.

use quantrs2_anneal::{
    embedding::{Embedding, HardwareGraph},
    ising::IsingModel,
    penalty_optimization::{
        Constraint, ConstraintPenaltyOptimizer, ConstraintType, PenaltyConfig, PenaltyOptimizer,
    },
    simulator::{AnnealingParams, ClassicalAnnealingSimulator},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Penalty Optimization Demo ===\n");

    // Step 1: Create a simple hardware graph
    println!("1. Creating hardware graph...");
    let hardware = HardwareGraph::new_chimera(1, 2, 4)?;
    println!("   Hardware: {} qubits", hardware.num_qubits);

    // Step 2: Create a manual embedding for 2 logical variables
    println!("\n2. Creating manual embedding...");
    let mut embedding = Embedding::new();

    // Variable 0 uses chain [0, 1]
    embedding.add_chain(0, vec![0, 1])?;

    // Variable 1 uses chain [4, 5]
    embedding.add_chain(1, vec![4, 5])?;

    println!("   Embedding created:");
    for (var, chain) in &embedding.chains {
        println!("     Variable {var}: chain {chain:?}");
    }

    // Step 3: Create an Ising model
    println!("\n3. Creating Ising model...");
    let mut ising = IsingModel::new(hardware.num_qubits);

    // Set biases
    ising.set_bias(0, 0.5)?;
    ising.set_bias(1, 0.5)?;
    ising.set_bias(4, -0.5)?;
    ising.set_bias(5, -0.5)?;

    // Set coupling between chains
    ising.set_coupling(0, 4, -1.0)?;

    // Step 4: Solve without chain penalties
    println!("\n4. Solving without chain penalties...");
    let mut params = AnnealingParams::new();
    params.num_sweeps = 500;
    params.num_repetitions = 20;

    let simulator = ClassicalAnnealingSimulator::new(params)?;
    let initial_result = simulator.solve(&ising)?;

    println!("   Initial solution:");
    println!("   - Best energy: {:.4}", initial_result.best_energy);
    println!("   - Best spins: {:?}", &initial_result.best_spins[..8]);

    // Generate samples
    let mut samples = Vec::new();
    for _ in 0..50 {
        let result = simulator.solve(&ising)?;
        samples.push(result.best_spins);
    }

    // Step 5: Apply penalty optimization
    println!("\n5. Applying penalty optimization...");
    let penalty_config = PenaltyConfig {
        initial_chain_strength: 0.5,
        min_chain_strength: 0.1,
        max_chain_strength: 3.0,
        chain_strength_scale: 1.5,
        constraint_penalty: 1.0,
        adaptive: true,
        learning_rate: 0.1,
    };

    let mut penalty_optimizer = PenaltyOptimizer::new(penalty_config);
    let stats = penalty_optimizer.optimize_ising_penalties(&mut ising, &embedding, &samples)?;

    println!("   Optimization complete:");
    println!("   - Iterations: {}", stats.iterations);
    println!(
        "   - Chain break rate: {:.2}%",
        stats.chain_break_rate * 100.0
    );
    println!("   - Optimized chain strengths:");
    for (var, strength) in &stats.chain_strengths {
        println!("     Variable {var}: {strength:.3}");
    }

    // Step 6: Solve with optimized penalties
    println!("\n6. Solving with optimized penalties...");
    let optimized_result = simulator.solve(&ising)?;

    println!("   Optimized solution:");
    println!("   - Best energy: {:.4}", optimized_result.best_energy);
    println!("   - Best spins: {:?}", &optimized_result.best_spins[..8]);

    // Step 7: Demonstrate constraint penalty optimization
    println!("\n7. Testing constraint penalty optimization...");
    let mut constraint_optimizer = ConstraintPenaltyOptimizer::new(0.05);

    // Add a constraint that variables 0 and 1 should sum to 1
    constraint_optimizer.add_constraint(Constraint {
        name: "var_sum".to_string(),
        variables: vec![0, 1],
        constraint_type: ConstraintType::Equality,
        target: 1.0,
    });

    // Convert samples to binary (0/1) format for constraint checking
    let binary_samples: Vec<Vec<i8>> = samples
        .iter()
        .map(|s| s.iter().map(|&spin| i8::from(spin == 1)).collect())
        .collect();

    let optimized_penalties = constraint_optimizer.optimize_penalties(&binary_samples, 10);

    println!("   Constraint penalties:");
    for (name, weight) in &optimized_penalties {
        println!("     {name}: {weight:.3}");
    }

    println!("\n=== Demo Complete ===");

    Ok(())
}
