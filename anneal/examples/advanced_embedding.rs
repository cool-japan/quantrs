#![allow(
    clippy::pedantic,
    clippy::unnecessary_wraps,
    clippy::needless_range_loop,
    clippy::useless_vec,
    clippy::needless_collect,
    clippy::too_many_arguments
)]
//! Advanced Graph Embedding Example
//!
//! This example demonstrates the use of advanced graph embedding algorithms
//! including layout-aware embedding and penalty optimization for quantum annealing.

use quantrs2_anneal::{
    embedding::{HardwareGraph, MinorMiner},
    ising::IsingModel,
    layout_embedding::{LayoutAwareEmbedder, LayoutConfig},
    penalty_optimization::{AdvancedPenaltyOptimizer, PenaltyConfig, PenaltyOptimizer},
    simulator::{AnnealingParams, ClassicalAnnealingSimulator},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Advanced Graph Embedding Demo ===\n");

    // Step 1: Create a logical problem graph (K3 - complete graph with 3 nodes)
    println!("1. Creating logical problem graph (K3)...");
    let num_vars = 3;
    let logical_edges = vec![(0, 1), (0, 2), (1, 2)];
    println!(
        "   Logical graph: {} variables, {} edges",
        num_vars,
        logical_edges.len()
    );

    // Step 2: Create hardware graph (Chimera 2x2x4)
    println!("\n2. Creating Chimera hardware graph...");
    let hardware = HardwareGraph::new_chimera(2, 2, 4)?;
    println!("   Hardware graph: {} qubits", hardware.num_qubits);

    // Step 3: Find embedding using standard MinorMiner
    println!("\n3. Finding embedding with MinorMiner...");
    let standard_embedder = MinorMiner::default();
    let standard_embedding =
        standard_embedder.find_embedding(&logical_edges, num_vars, &hardware)?;

    println!("   Standard embedding found:");
    for (var, chain) in &standard_embedding.chains {
        println!(
            "     Variable {}: chain {:?} (length {})",
            var,
            chain,
            chain.len()
        );
    }

    // Step 4: Find embedding using layout-aware approach
    println!("\n4. Finding layout-aware embedding...");
    let layout_config = LayoutConfig {
        distance_weight: 1.0,
        chain_length_weight: 2.0,
        chain_degree_weight: 0.5,
        max_chain_length: 4,
        use_spectral_placement: true,
        refinement_iterations: 5,
    };

    let mut layout_embedder = LayoutAwareEmbedder::new(layout_config);
    let (layout_embedding, layout_stats) =
        layout_embedder.find_embedding(&logical_edges, num_vars, &hardware)?;

    println!("   Layout-aware embedding found:");
    println!(
        "   - Average chain length: {:.2}",
        layout_stats.avg_chain_length
    );
    println!(
        "   - Maximum chain length: {}",
        layout_stats.max_chain_length
    );
    println!(
        "   - Long chains (>{} qubits): {}",
        4, layout_stats.long_chains
    );
    println!("   - Quality score: {:.2}", layout_stats.quality_score);

    // Step 5: Create an Ising model for the problem
    println!("\n5. Creating Ising model...");
    let mut ising = IsingModel::new(hardware.num_qubits);

    // Set some random biases for the logical problem
    for (var, chain) in &layout_embedding.chains {
        // Apply bias to all qubits in the chain
        for &qubit in chain {
            if qubit < hardware.num_qubits {
                ising.set_bias(qubit, 0.5 * (*var as f64 - 2.0))?;
            }
        }
    }

    // Set couplings between chains
    for &(i, j) in &logical_edges {
        if let (Some(chain_i), Some(chain_j)) = (
            layout_embedding.chains.get(&i),
            layout_embedding.chains.get(&j),
        ) {
            // Find a valid coupling between the chains
            'outer: for &qi in chain_i {
                for &qj in chain_j {
                    if qi < hardware.num_qubits
                        && qj < hardware.num_qubits
                        && hardware.are_connected(qi, qj)
                    {
                        ising.set_coupling(qi, qj, -1.0)?;
                        break 'outer;
                    }
                }
            }
        }
    }

    // Step 6: Solve without penalty optimization
    println!("\n6. Solving without penalty optimization...");
    let mut params = AnnealingParams::new();
    params.num_sweeps = 1000;
    params.num_repetitions = 100;
    params.initial_temperature = 10.0;
    params.temperature_schedule = quantrs2_anneal::simulator::TemperatureSchedule::Exponential(3.0);

    let simulator = ClassicalAnnealingSimulator::new(params)?;
    let initial_result = simulator.solve(&ising)?;

    println!("   Initial solution:");
    println!("   - Best energy: {:.4}", initial_result.best_energy);

    // Generate multiple samples by running the simulator multiple times
    let mut samples: Vec<Vec<i8>> = Vec::new();
    for _ in 0..10 {
        let result = simulator.solve(&ising)?;
        samples.push(result.best_spins);
    }

    // Step 7: Apply penalty optimization
    println!("\n7. Applying penalty optimization...");
    let penalty_config = PenaltyConfig {
        initial_chain_strength: 1.0,
        min_chain_strength: 0.5,
        max_chain_strength: 5.0,
        chain_strength_scale: 1.2,
        constraint_penalty: 1.0,
        adaptive: true,
        learning_rate: 0.1,
    };

    let mut penalty_optimizer = PenaltyOptimizer::new(penalty_config.clone());
    let penalty_stats =
        penalty_optimizer.optimize_ising_penalties(&mut ising, &layout_embedding, &samples)?;

    println!("   Penalty optimization complete:");
    println!("   - Iterations: {}", penalty_stats.iterations);
    println!(
        "   - Chain break rate: {:.2}%",
        penalty_stats.chain_break_rate * 100.0
    );

    // Step 8: Solve with optimized penalties
    println!("\n8. Solving with optimized penalties...");
    let optimized_result = simulator.solve(&ising)?;

    println!("   Optimized solution:");
    println!("   - Best energy: {:.4}", optimized_result.best_energy);
    println!(
        "   - Energy improvement: {:.4}",
        initial_result.best_energy - optimized_result.best_energy
    );

    // Step 9: Try advanced gradient-based optimization
    println!("\n9. Applying gradient-based penalty optimization...");
    let mut advanced_optimizer = AdvancedPenaltyOptimizer::new(penalty_config);
    let advanced_stats = advanced_optimizer.optimize_with_gradients(
        &mut ising,
        &layout_embedding,
        &samples,
        20, // max iterations
    )?;

    println!("   Advanced optimization complete:");
    println!("   - Final chain strengths:");
    for (var, strength) in &advanced_stats.chain_strengths {
        println!("     Variable {var}: {strength:.3}");
    }

    println!("\n=== Demo Complete ===");

    Ok(())
}
