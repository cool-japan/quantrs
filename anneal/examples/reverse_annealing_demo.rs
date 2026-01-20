#![allow(clippy::pedantic, clippy::unnecessary_wraps)]
//! Example demonstrating reverse annealing for quantum optimization
//!
//! This example shows how to:
//! 1. Find an initial solution using forward annealing
//! 2. Use reverse annealing to improve the solution
//! 3. Compare different reverse annealing strategies

use quantrs2_anneal::{
    ising::IsingModel,
    reverse_annealing::{
        ReverseAnnealingParams, ReverseAnnealingScheduleBuilder, ReverseAnnealingSimulator,
    },
    simulator::{AnnealingParams, ClassicalAnnealingSimulator},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a frustrated Ising model (difficult to solve)
    let n = 10;
    let mut model = IsingModel::new(n);

    // Create a frustrated loop with random couplings
    for i in 0..n {
        // Add random biases
        model.set_bias(i, (i as f64 * 0.3).sin())?;

        // Add couplings in a ring
        let j = (i + 1) % n;
        let coupling = if i % 2 == 0 { -1.0 } else { 0.8 };
        model.set_coupling(i, j, coupling)?;
    }

    // Add some long-range couplings to increase frustration
    model.set_coupling(0, n / 2, 0.5)?;
    model.set_coupling(n / 4, 3 * n / 4, -0.7)?;

    println!("Created frustrated Ising model with {n} qubits");
    println!("Finding initial solution using forward annealing...\n");

    // Step 1: Find initial solution using classical forward annealing
    let forward_params = AnnealingParams {
        num_sweeps: 2000,
        num_repetitions: 20,
        ..Default::default()
    };

    let mut forward_simulator = ClassicalAnnealingSimulator::new(forward_params)?;
    let forward_result = forward_simulator.solve(&model)?;

    println!("Forward annealing results:");
    println!("  Best energy: {:.4}", forward_result.best_energy);
    println!("  Best solution: {:?}", forward_result.best_spins);
    println!("  Runtime: {:.2?}", forward_result.runtime);

    // Step 2: Use reverse annealing to improve the solution
    println!("\nApplying reverse annealing to improve solution...\n");

    // Try different reverse annealing strategies
    let strategies = vec![
        ("Conservative", 0.7, 0.1), // s_target, pause_duration
        ("Moderate", 0.45, 0.15),
        ("Aggressive", 0.2, 0.2),
    ];

    for (name, s_target, pause) in strategies {
        println!("Strategy: {name} (s_target={s_target}, pause={pause})");

        // Create reverse annealing schedule
        let schedule = ReverseAnnealingScheduleBuilder::new()
            .s_target(s_target)
            .pause_duration(pause)
            .build()?;

        // Configure reverse annealing
        let reverse_params = ReverseAnnealingParams {
            schedule,
            initial_state: forward_result.best_spins.clone(),
            num_sweeps: 1000,
            num_repetitions: 10,
            reinitialize_fraction: 0.0, // No random reinitialization
            local_search_radius: None,  // Global search
            seed: Some(42),             // Reproducible results
        };

        let mut reverse_simulator = ReverseAnnealingSimulator::new(reverse_params)?;
        let reverse_result = reverse_simulator.solve(&model)?;

        println!(
            "  Best energy: {:.4} (improvement: {:.4})",
            reverse_result.best_energy,
            forward_result.best_energy - reverse_result.best_energy
        );
        println!("  Runtime: {:.2?}", reverse_result.runtime);

        // Check if we found a better solution
        if reverse_result.best_energy < forward_result.best_energy {
            println!("  ✓ Found better solution!");
        }
        println!();
    }

    // Step 3: Demonstrate targeted reverse annealing with local search
    println!("Demonstrating targeted reverse annealing with local search...\n");

    let local_params = ReverseAnnealingParams::new(forward_result.best_spins.clone())
        .with_local_search(3)  // Only update spins within radius 3
        .with_reinitialization(0.1); // Reinitialize 10% of spins

    let mut local_simulator = ReverseAnnealingSimulator::new(local_params)?;
    let local_result = local_simulator.solve(&model)?;

    println!("Targeted reverse annealing results:");
    println!("  Best energy: {:.4}", local_result.best_energy);
    println!(
        "  Improvement over forward: {:.4}",
        forward_result.best_energy - local_result.best_energy
    );

    // Step 4: Analyze the reverse annealing schedule
    println!("\nAnalyzing reverse annealing schedule...\n");

    let schedule = ReverseAnnealingScheduleBuilder::new()
        .s_target(0.45)
        .pause_duration(0.1)
        .quench_rate(1.2)
        .build()?;

    println!("s(t) values during annealing:");
    for i in 0..=10 {
        let t = f64::from(i) / 10.0;
        let s = schedule.s_of_t(t);
        let a = schedule.transverse_field(s);
        let b = schedule.problem_strength(s);
        println!("  t={t:.1}: s={s:.3}, A(s)={a:.3}, B(s)={b:.3}");
    }

    // Demonstrate iterative reverse annealing
    println!("\nDemonstrating iterative reverse annealing...\n");

    let mut current_state = forward_result.best_spins.clone();
    let mut current_energy = forward_result.best_energy;

    for iteration in 1..=5 {
        // Use progressively more conservative reverse annealing
        let s_target = 0.1f64.mul_add(-(iteration as f64), 0.8);

        let iter_schedule = ReverseAnnealingScheduleBuilder::new()
            .s_target(s_target.max(0.3))
            .pause_duration(0.05)
            .build()?;

        let iter_params = ReverseAnnealingParams {
            schedule: iter_schedule,
            initial_state: current_state.clone(),
            num_sweeps: 500,
            num_repetitions: 5,
            reinitialize_fraction: 0.0,
            local_search_radius: None,
            seed: Some(42 + iteration),
        };

        let mut iter_simulator = ReverseAnnealingSimulator::new(iter_params)?;
        let iter_result = iter_simulator.solve(&model)?;

        if iter_result.best_energy < current_energy {
            current_state = iter_result.best_spins;
            current_energy = iter_result.best_energy;
            println!("  Iteration {iteration}: Improved to {current_energy:.4}");
        } else {
            println!(
                "  Iteration {}: No improvement (energy: {:.4})",
                iteration, iter_result.best_energy
            );
        }
    }

    println!("\nFinal results:");
    println!(
        "  Initial energy (forward annealing): {:.4}",
        forward_result.best_energy
    );
    println!("  Final energy (after reverse annealing): {current_energy:.4}");
    println!(
        "  Total improvement: {:.4}",
        forward_result.best_energy - current_energy
    );

    Ok(())
}
