//! Example demonstrating flux bias optimization for D-Wave quantum annealing
//!
//! This example shows how to:
//! 1. Create an Ising model with known calibration issues
//! 2. Optimize flux biases to compensate for hardware imperfections
//! 3. Submit the problem to D-Wave with optimized flux biases

use quantrs2_anneal::{
    ising::IsingModel,
    embedding::{Embedding, MinorMiner, HardwareTopology},
    flux_bias::{FluxBiasOptimizer, FluxBiasConfig, CalibrationData},
    simulator::ClassicalAnnealingSimulator,
    dwave::{DWaveClient, ProblemParams},
};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a sample Ising model
    let mut model = IsingModel::new(4);
    
    // Set biases
    model.set_bias(0, 0.5)?;
    model.set_bias(1, -0.3)?;
    model.set_bias(2, 0.2)?;
    model.set_bias(3, -0.4)?;
    
    // Set couplings
    model.set_coupling(0, 1, -1.0)?;
    model.set_coupling(1, 2, -0.5)?;
    model.set_coupling(2, 3, -0.8)?;
    model.set_coupling(0, 3, 0.3)?;
    
    // Create a hardware embedding (simplified for example)
    let mut embedding = Embedding::new();
    // Map logical qubits to hardware chains
    embedding.chains.insert(0, vec![0, 4]);  // Logical qubit 0 -> hardware qubits 0, 4
    embedding.chains.insert(1, vec![1, 5]);  // Logical qubit 1 -> hardware qubits 1, 5
    embedding.chains.insert(2, vec![2]);      // Logical qubit 2 -> hardware qubit 2
    embedding.chains.insert(3, vec![3]);      // Logical qubit 3 -> hardware qubit 3
    
    // Simulate calibration data (in practice, this would come from hardware)
    let mut calibration = CalibrationData {
        nominal_biases: HashMap::new(),
        bias_errors: HashMap::new(),
        coupling_errors: HashMap::new(),
        qubit_temperatures: HashMap::new(),
    };
    
    // Add some simulated calibration errors
    calibration.bias_errors.insert(0, 0.02);  // 2% bias error on qubit 0
    calibration.bias_errors.insert(1, -0.01); // -1% bias error on qubit 1
    calibration.qubit_temperatures.insert(0, 16.5); // Slightly warm qubit
    calibration.qubit_temperatures.insert(1, 14.8); // Slightly cool qubit
    
    // Generate some sample solutions using classical annealing
    println!("Generating sample solutions...");
    let simulator = ClassicalAnnealingSimulator::new(Default::default())?;
    let annealing_result = simulator.solve(&model)?;
    
    // Convert solutions to proper format (generate samples from best solution)
    let samples: Vec<Vec<i8>> = vec![annealing_result.best_spins.clone(); 10];
    
    // Configure flux bias optimization
    let config = FluxBiasConfig {
        max_flux_bias: 0.05,    // Conservative flux bias range
        use_gradients: true,    // Use gradient-based optimization
        learning_rate: 0.01,    // Small learning rate
        regularization: 0.001,  // Prevent large flux biases
        ..Default::default()
    };
    
    // Create and configure optimizer
    let mut optimizer = FluxBiasOptimizer::new(config);
    optimizer.set_calibration_data(calibration);
    
    // Optimize flux biases
    println!("Optimizing flux biases...");
    let flux_result = optimizer.optimize_ising(&model, &embedding, &samples)?;
    
    println!("Optimization complete!");
    println!("Energy improvement: {:.4}", flux_result.energy_improvement);
    println!("Solution quality: {:.2}%", flux_result.solution_quality * 100.0);
    println!("Iterations: {}", flux_result.iterations);
    
    // Display optimized flux biases
    println!("\nOptimized flux biases:");
    for (qubit, flux_bias) in &flux_result.flux_biases {
        println!("  Qubit {}: {:.4}", qubit, flux_bias);
    }
    
    // If D-Wave is available, submit with optimized flux biases
    #[cfg(feature = "dwave")]
    {
        if let Ok(token) = std::env::var("DWAVE_API_TOKEN") {
            println!("\nSubmitting to D-Wave with optimized flux biases...");
            
            let client = DWaveClient::new(token, None)?;
            
            // Configure submission parameters
            let params = ProblemParams {
                num_reads: 1000,
                annealing_time: 20,
                ..Default::default()
            };
            
            // Submit with flux biases
            let solution = client.submit_ising_with_flux_bias(
                &model,
                "DW_2000Q_6", // Example solver ID
                params,
                &flux_result.flux_biases,
            )?;
            
            println!("D-Wave solution received!");
            println!("Best energy: {}", solution.energies[0]);
            println!("Number of occurrences: {}", solution.occurrences[0]);
        } else {
            println!("\nSet DWAVE_API_TOKEN environment variable to submit to D-Wave");
        }
    }
    #[cfg(not(feature = "dwave"))]
    {
        println!("\nCompile with --features dwave to enable D-Wave submission");
    }
    
    // Demonstrate ML-enhanced flux bias optimization
    use quantrs2_anneal::flux_bias::MLFluxBiasOptimizer;
    
    let mut ml_optimizer = MLFluxBiasOptimizer::new(Default::default());
    
    // Learn pattern from this optimization
    ml_optimizer.learn_pattern("ferromagnetic_chain", &flux_result.flux_biases);
    
    // Apply learned pattern to similar problems
    if let Some(learned_biases) = ml_optimizer.apply_learned_patterns("ferromagnetic_chain", 6) {
        println!("\nLearned flux bias pattern can be applied to similar problems:");
        for (qubit, bias) in learned_biases {
            println!("  Qubit {}: {:.4}", qubit, bias);
        }
    }
    
    Ok(())
}