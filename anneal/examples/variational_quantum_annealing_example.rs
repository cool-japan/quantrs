//! Example demonstrating Variational Quantum Annealing (VQA)
//!
//! This example shows how to:
//! 1. Configure different ansatz types (hardware-efficient, QAOA-inspired, adiabatic)
//! 2. Use various classical optimizers (Adam, gradient descent, etc.)
//! 3. Solve optimization problems with VQA
//! 4. Analyze convergence and parameter evolution
//! 5. Compare VQA performance across different configurations
//! 6. Study parameter sensitivity and hyperparameter tuning

use quantrs2_anneal::{
    ising::IsingModel,
    simulator::AnnealingParams,
    variational_quantum_annealing::{
        create_adiabatic_vqa_config, create_hardware_efficient_vqa_config, create_qaoa_vqa_config,
        AnsatzType, ClassicalOptimizer, EntanglingGateType, MixerType, ParameterRef, QuantumGate,
        VariationalQuantumAnnealer, VqaConfig,
    },
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Variational Quantum Annealing Demo ===\n");

    // Example 1: QAOA-inspired VQA
    println!("Example 1: QAOA-Inspired Variational Quantum Annealing");
    qaoa_inspired_vqa_example()?;

    // Example 2: Hardware-efficient VQA
    println!("\nExample 2: Hardware-Efficient Variational Quantum Annealing");
    hardware_efficient_vqa_example()?;

    // Example 3: Adiabatic-inspired VQA
    println!("\nExample 3: Adiabatic-Inspired Variational Quantum Annealing");
    adiabatic_inspired_vqa_example()?;

    // Example 4: Optimizer comparison
    println!("\nExample 4: Classical Optimizer Comparison");
    optimizer_comparison_example()?;

    // Example 5: Ansatz depth study
    println!("\nExample 5: Ansatz Depth Sensitivity Study");
    ansatz_depth_study_example()?;

    // Example 6: Custom ansatz design
    println!("\nExample 6: Custom Ansatz Design");
    custom_ansatz_example()?;

    // Example 7: Large-scale optimization
    println!("\nExample 7: Large-Scale Optimization Problem");
    large_scale_optimization_example()?;

    Ok(())
}

fn qaoa_inspired_vqa_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a MaxCut problem on a triangle graph
    let mut problem = IsingModel::new(3);
    problem.set_coupling(0, 1, -1.0)?; // Negative couplings for MaxCut
    problem.set_coupling(1, 2, -1.0)?;
    problem.set_coupling(2, 0, -1.0)?;

    // Configure QAOA-inspired VQA with 3 layers
    let config = create_qaoa_vqa_config(3, 50);

    let start = Instant::now();
    let mut vqa = VariationalQuantumAnnealer::new(config)?;
    let results = vqa.optimize(&problem)?;
    let runtime = start.elapsed();

    println!("QAOA-Inspired VQA Results:");
    println!("  Problem: MaxCut on triangle graph (3 vertices)");
    println!("  Ansatz: QAOA-inspired with 3 layers");
    println!(
        "  Parameters: {} (2 per layer)",
        results.optimal_parameters.len()
    );
    println!("  Best energy: {:.6}", results.best_energy);
    println!("  Best solution: {:?}", results.best_solution);
    println!("  Iterations: {}", results.iterations_completed);
    println!("  Converged: {}", results.converged);
    println!("  Runtime: {:.2?}", runtime);

    // Analyze convergence
    println!("\n  Convergence analysis:");
    if results.energy_history.len() >= 10 {
        let initial_energy = results.energy_history[0];
        let final_energy = *results.energy_history.last().unwrap();
        let improvement = initial_energy - final_energy;
        println!("    Initial energy: {:.6}", initial_energy);
        println!("    Final energy: {:.6}", final_energy);
        println!("    Total improvement: {:.6}", improvement);

        // Show energy progression
        println!("    Energy progression (every 10 iterations):");
        for (i, &energy) in results.energy_history.iter().enumerate() {
            if i % 10 == 0 || i == results.energy_history.len() - 1 {
                println!("      Iter {}: {:.6}", i, energy);
            }
        }
    }

    // Parameter analysis
    println!("\n  Optimal parameters:");
    for (i, &param) in results.optimal_parameters.iter().enumerate() {
        let param_type = if i % 2 == 0 { "gamma" } else { "beta" };
        let layer = i / 2;
        println!("    Layer {} {}: {:.6}", layer, param_type, param);
    }

    // Performance metrics
    println!("\n  Performance metrics:");
    println!(
        "    Function evaluations: {}",
        results.statistics.function_evaluations
    );
    println!(
        "    Gradient evaluations: {}",
        results.statistics.gradient_evaluations
    );
    println!(
        "    Average energy: {:.6}",
        results.statistics.average_energy
    );
    println!(
        "    Energy variance: {:.6}",
        results.statistics.energy_variance
    );

    Ok(())
}

fn hardware_efficient_vqa_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a random Ising problem
    let mut problem = IsingModel::new(4);

    // Add random biases
    problem.set_bias(0, 0.5)?;
    problem.set_bias(1, -0.3)?;
    problem.set_bias(2, 0.8)?;
    problem.set_bias(3, -0.2)?;

    // Add random couplings
    problem.set_coupling(0, 1, 0.7)?;
    problem.set_coupling(1, 2, -0.4)?;
    problem.set_coupling(2, 3, 0.6)?;
    problem.set_coupling(0, 3, -0.5)?;

    // Configure hardware-efficient ansatz
    let mut config = create_hardware_efficient_vqa_config(2, 80);
    config.optimizer = ClassicalOptimizer::Adam {
        learning_rate: 0.05,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
    };
    config.convergence_tolerance = 1e-5;
    config.log_frequency = 20;

    let start = Instant::now();
    let mut vqa = VariationalQuantumAnnealer::new(config)?;
    let results = vqa.optimize(&problem)?;
    let runtime = start.elapsed();

    println!("Hardware-Efficient VQA Results:");
    println!("  Problem: Random 4-qubit Ising model");
    println!("  Ansatz: Hardware-efficient with depth 2");
    println!("  Entangling gates: CNOT");
    println!("  Optimizer: Adam (lr=0.05)");
    println!("  Parameters: {}", results.optimal_parameters.len());
    println!("  Best energy: {:.6}", results.best_energy);
    println!("  Best solution: {:?}", results.best_solution);
    println!("  Iterations: {}", results.iterations_completed);
    println!("  Converged: {}", results.converged);
    println!("  Runtime: {:.2?}", runtime);

    // Gradient analysis
    if !results.gradient_norms.is_empty() {
        println!("\n  Gradient analysis:");
        let initial_grad_norm = results.gradient_norms[0];
        let final_grad_norm = *results.gradient_norms.last().unwrap();
        println!("    Initial gradient norm: {:.6}", initial_grad_norm);
        println!("    Final gradient norm: {:.6}", final_grad_norm);

        // Show gradient progression
        println!("    Gradient norm progression:");
        for (i, &grad_norm) in results.gradient_norms.iter().enumerate() {
            if i % 20 == 0 || i == results.gradient_norms.len() - 1 {
                println!("      Iter {}: {:.6}", i, grad_norm);
            }
        }
    }

    // Energy landscape analysis
    println!("\n  Energy landscape features:");
    let energy_range = results
        .energy_history
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        - results
            .energy_history
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
    println!("    Energy range explored: {:.6}", energy_range);

    // Parameter statistics
    let param_magnitudes: Vec<f64> = results
        .optimal_parameters
        .iter()
        .map(|&p| p.abs())
        .collect();
    let avg_param_magnitude = param_magnitudes.iter().sum::<f64>() / param_magnitudes.len() as f64;
    let max_param_magnitude = param_magnitudes
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    println!(
        "    Average parameter magnitude: {:.6}",
        avg_param_magnitude
    );
    println!(
        "    Maximum parameter magnitude: {:.6}",
        max_param_magnitude
    );

    Ok(())
}

fn adiabatic_inspired_vqa_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a more complex problem (frustrated system)
    let mut problem = IsingModel::new(5);

    // Create a frustrated ring
    for i in 0..5 {
        let j = (i + 1) % 5;
        problem.set_coupling(i, j, 1.0)?; // Positive couplings for frustration
    }

    // Add some biases
    for i in 0..5 {
        problem.set_bias(i, (-1.0_f64).powi(i as i32) * 0.3)?;
    }

    // Configure adiabatic-inspired ansatz
    let mut config = create_adiabatic_vqa_config(10, 2.0, 100);
    config.optimizer = ClassicalOptimizer::GradientDescent {
        learning_rate: 0.02,
    };
    config.num_shots = 50;
    config.use_gradients = true;
    config.gradient_step = 0.005;

    let start = Instant::now();
    let mut vqa = VariationalQuantumAnnealer::new(config)?;
    let results = vqa.optimize(&problem)?;
    let runtime = start.elapsed();

    println!("Adiabatic-Inspired VQA Results:");
    println!("  Problem: Frustrated 5-qubit ring");
    println!("  Ansatz: Adiabatic-inspired (10 time steps, T=2.0)");
    println!("  Optimizer: Gradient Descent (lr=0.02)");
    println!(
        "  Parameters: {} (s-parameters)",
        results.optimal_parameters.len()
    );
    println!("  Best energy: {:.6}", results.best_energy);
    println!("  Best solution: {:?}", results.best_solution);
    println!("  Iterations: {}", results.iterations_completed);
    println!("  Converged: {}", results.converged);
    println!("  Runtime: {:.2?}", runtime);

    // Analyze s-parameter evolution
    println!("\n  Adiabatic schedule analysis:");
    println!("    Optimal s-parameters:");
    for (i, &s_param) in results.optimal_parameters.iter().enumerate() {
        let time_fraction = i as f64 / (results.optimal_parameters.len() - 1) as f64;
        println!("      t={:.2}: s={:.6}", time_fraction, s_param);
    }

    // Check if schedule is monotonic
    let mut is_monotonic = true;
    for i in 1..results.optimal_parameters.len() {
        if results.optimal_parameters[i] < results.optimal_parameters[i - 1] {
            is_monotonic = false;
            break;
        }
    }

    println!("    Schedule monotonic: {}", is_monotonic);
    println!("    Initial s: {:.6}", results.optimal_parameters[0]);
    println!(
        "    Final s: {:.6}",
        *results.optimal_parameters.last().unwrap()
    );

    // Frustration analysis
    let num_frustrated_bonds = count_frustrated_bonds(&results.best_solution, &problem);
    println!("\n  Frustration analysis:");
    println!("    Frustrated bonds: {}/5", num_frustrated_bonds);
    println!(
        "    Ground state quality: {:.1}%",
        (5 - num_frustrated_bonds) as f64 / 5.0 * 100.0
    );

    Ok(())
}

fn optimizer_comparison_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a test problem
    let mut problem = IsingModel::new(4);
    problem.set_coupling(0, 1, -1.0)?;
    problem.set_coupling(1, 2, 1.0)?;
    problem.set_coupling(2, 3, -1.0)?;
    problem.set_coupling(0, 3, 1.0)?;
    problem.set_coupling(0, 2, -0.5)?;
    problem.set_coupling(1, 3, 0.5)?;

    let optimizers = vec![
        (
            "Adam",
            ClassicalOptimizer::Adam {
                learning_rate: 0.01,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
        ),
        (
            "Gradient Descent",
            ClassicalOptimizer::GradientDescent {
                learning_rate: 0.05,
            },
        ),
        (
            "RMSprop",
            ClassicalOptimizer::RMSprop {
                learning_rate: 0.01,
                decay_rate: 0.9,
                epsilon: 1e-8,
            },
        ),
    ];

    println!("Classical Optimizer Comparison:");
    println!("  Problem: 4-qubit mixed coupling Ising model");
    println!("  Ansatz: QAOA-inspired (3 layers)");
    println!("  Max iterations: 60");

    for (optimizer_name, optimizer) in optimizers {
        let mut config = create_qaoa_vqa_config(3, 60);
        config.optimizer = optimizer;
        config.convergence_tolerance = 1e-6;
        config.log_frequency = 1000; // No intermediate logging
        config.seed = Some(42); // Same seed for fair comparison

        let start = Instant::now();
        let mut vqa = VariationalQuantumAnnealer::new(config)?;
        let results = vqa.optimize(&problem)?;
        let runtime = start.elapsed();

        println!("\n  {} Results:", optimizer_name);
        println!("    Best energy: {:.6}", results.best_energy);
        println!("    Iterations: {}", results.iterations_completed);
        println!("    Converged: {}", results.converged);
        println!("    Runtime: {:.2?}", runtime);
        println!(
            "    Function evals: {}",
            results.statistics.function_evaluations
        );
        println!(
            "    Gradient evals: {}",
            results.statistics.gradient_evaluations
        );

        if !results.energy_history.is_empty() {
            let initial_energy = results.energy_history[0];
            let final_energy = *results.energy_history.last().unwrap();
            let improvement = initial_energy - final_energy;
            println!("    Energy improvement: {:.6}", improvement);

            // Convergence rate analysis
            let convergence_rate = if results.iterations_completed > 1 {
                improvement / results.iterations_completed as f64
            } else {
                0.0
            };
            println!("    Avg improvement/iter: {:.8}", convergence_rate);
        }
    }

    Ok(())
}

fn ansatz_depth_study_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a test problem
    let mut problem = IsingModel::new(6);

    // Create a random problem
    for i in 0..6 {
        problem.set_bias(i, (i as f64 - 2.5) * 0.1)?;
        for j in (i + 1)..6 {
            if (i + j) % 3 == 0 {
                problem.set_coupling(i, j, if (i * j) % 2 == 0 { 0.5 } else { -0.5 })?;
            }
        }
    }

    let depths = vec![1, 2, 3, 4, 5];

    println!("Ansatz Depth Sensitivity Study:");
    println!("  Problem: 6-qubit structured Ising model");
    println!("  Ansatz: Hardware-efficient with varying depth");
    println!("  Optimizer: Adam (lr=0.02)");
    println!("  Max iterations: 40");

    for depth in depths {
        let mut config = create_hardware_efficient_vqa_config(depth, 40);
        config.optimizer = ClassicalOptimizer::Adam {
            learning_rate: 0.02,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        };
        config.convergence_tolerance = 1e-5;
        config.log_frequency = 1000;
        config.seed = Some(123);

        let start = Instant::now();
        let mut vqa = VariationalQuantumAnnealer::new(config)?;
        let results = vqa.optimize(&problem)?;
        let runtime = start.elapsed();

        println!("\n  Depth {} Results:", depth);
        println!("    Parameters: {}", results.optimal_parameters.len());
        println!("    Best energy: {:.6}", results.best_energy);
        println!("    Iterations: {}", results.iterations_completed);
        println!("    Converged: {}", results.converged);
        println!("    Runtime: {:.2?}", runtime);

        // Parameter utilization analysis
        let param_magnitudes: Vec<f64> = results
            .optimal_parameters
            .iter()
            .map(|&p| p.abs())
            .collect();
        let significant_params = param_magnitudes.iter().filter(|&&p| p > 0.1).count();
        let utilization = significant_params as f64 / param_magnitudes.len() as f64;

        println!(
            "    Significant parameters: {}/{}",
            significant_params,
            param_magnitudes.len()
        );
        println!("    Parameter utilization: {:.1}%", utilization * 100.0);

        // Expressivity analysis
        let energy_range = if results.energy_history.len() > 1 {
            results
                .energy_history
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
                - results
                    .energy_history
                    .iter()
                    .cloned()
                    .fold(f64::INFINITY, f64::min)
        } else {
            0.0
        };
        println!("    Energy range explored: {:.6}", energy_range);
    }

    println!("\n  Analysis:");
    println!("    - Deeper ansätze have more parameters but may require more iterations");
    println!("    - Optimal depth balances expressivity with trainability");
    println!("    - Parameter utilization indicates ansatz efficiency");

    Ok(())
}

fn custom_ansatz_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a small test problem
    let mut problem = IsingModel::new(3);
    problem.set_coupling(0, 1, -1.5)?;
    problem.set_coupling(1, 2, 1.0)?;
    problem.set_coupling(0, 2, -0.5)?;

    // Design a custom ansatz circuit
    let custom_structure = vec![
        // Initial layer of Y rotations
        QuantumGate::RY {
            qubit: 0,
            angle: ParameterRef::new(0),
        },
        QuantumGate::RY {
            qubit: 1,
            angle: ParameterRef::new(1),
        },
        QuantumGate::RY {
            qubit: 2,
            angle: ParameterRef::new(2),
        },
        // Entangling layer
        QuantumGate::CNOT {
            control: 0,
            target: 1,
        },
        QuantumGate::CNOT {
            control: 1,
            target: 2,
        },
        // Problem-specific rotations
        QuantumGate::RZ {
            qubit: 0,
            angle: ParameterRef::scaled(3, 0.5),
        },
        QuantumGate::RZ {
            qubit: 1,
            angle: ParameterRef::new(4),
        },
        QuantumGate::RZ {
            qubit: 2,
            angle: ParameterRef::scaled(5, 0.8),
        },
        // Final mixing layer
        QuantumGate::RX {
            qubit: 0,
            angle: ParameterRef::new(6),
        },
        QuantumGate::RX {
            qubit: 1,
            angle: ParameterRef::new(7),
        },
        QuantumGate::RX {
            qubit: 2,
            angle: ParameterRef::new(8),
        },
    ];

    let mut config = VqaConfig {
        ansatz: AnsatzType::Custom {
            structure: custom_structure,
        },
        optimizer: ClassicalOptimizer::Adam {
            learning_rate: 0.03,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        },
        max_iterations: 70,
        convergence_tolerance: 1e-6,
        num_shots: 80,
        parameter_init_range: (-0.3, 0.3),
        use_gradients: true,
        seed: Some(456),
        log_frequency: 15,
        ..Default::default()
    };

    let start = Instant::now();
    let mut vqa = VariationalQuantumAnnealer::new(config)?;
    let results = vqa.optimize(&problem)?;
    let runtime = start.elapsed();

    println!("Custom Ansatz VQA Results:");
    println!("  Problem: 3-qubit triangle with mixed couplings");
    println!("  Ansatz: Custom design with specific gate sequence");
    println!("  Circuit depth: 4 layers (Y→CNOT→Z→X)");
    println!(
        "  Parameters: {} with scaling factors",
        results.optimal_parameters.len()
    );
    println!("  Best energy: {:.6}", results.best_energy);
    println!("  Best solution: {:?}", results.best_solution);
    println!("  Iterations: {}", results.iterations_completed);
    println!("  Converged: {}", results.converged);
    println!("  Runtime: {:.2?}", runtime);

    // Analyze custom circuit structure
    println!("\n  Custom circuit analysis:");
    println!("    Gate sequence: RY → CNOT → RZ(scaled) → RX");
    println!("    Parameter roles:");
    for i in 0..results.optimal_parameters.len() {
        let role = match i {
            0..=2 => "Initial mixing",
            3..=5 => "Problem encoding",
            6..=8 => "Final mixing",
            _ => "Other",
        };
        println!(
            "      θ[{}] = {:.6} ({})",
            i, results.optimal_parameters[i], role
        );
    }

    // Circuit expressivity analysis
    let param_std = {
        let mean = results.optimal_parameters.iter().sum::<f64>()
            / results.optimal_parameters.len() as f64;
        let variance = results
            .optimal_parameters
            .iter()
            .map(|&p| (p - mean).powi(2))
            .sum::<f64>()
            / results.optimal_parameters.len() as f64;
        variance.sqrt()
    };

    println!("\n  Parameter statistics:");
    println!("    Parameter std deviation: {:.6}", param_std);
    println!(
        "    Parameter range: [{:.6}, {:.6}]",
        results
            .optimal_parameters
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min),
        results
            .optimal_parameters
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    );

    Ok(())
}

fn large_scale_optimization_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a larger optimization problem
    let problem_size = 8;
    let mut problem = IsingModel::new(problem_size);

    // Create a random graph with density ~30%
    for i in 0..problem_size {
        // Add random biases
        problem.set_bias(i, (i as f64 - 3.5) * 0.1)?;

        for j in (i + 1)..problem_size {
            // Add coupling with 30% probability
            if (i * 7 + j * 11) % 10 < 3 {
                let coupling = if (i + j) % 2 == 0 { 0.4 } else { -0.6 };
                problem.set_coupling(i, j, coupling)?;
            }
        }
    }

    // Use QAOA-inspired ansatz with moderate depth
    let mut config = create_qaoa_vqa_config(4, 150);
    config.optimizer = ClassicalOptimizer::Adam {
        learning_rate: 0.015,
        beta1: 0.95,
        beta2: 0.999,
        epsilon: 1e-8,
    };
    config.num_shots = 60;
    config.convergence_tolerance = 1e-5;
    config.log_frequency = 30;
    config.seed = Some(789);

    let start = Instant::now();
    let mut vqa = VariationalQuantumAnnealer::new(config)?;
    let results = vqa.optimize(&problem)?;
    let runtime = start.elapsed();

    println!("Large-Scale Optimization Results:");
    println!(
        "  Problem: {}-qubit random graph (~30% edge density)",
        problem_size
    );
    println!("  Ansatz: QAOA-inspired (4 layers)");
    println!(
        "  Parameters: {} (8 total: 4×2)",
        results.optimal_parameters.len()
    );
    println!("  Max iterations: 150");
    println!("  Best energy: {:.6}", results.best_energy);
    println!("  Best solution: {:?}", results.best_solution);
    println!("  Iterations: {}", results.iterations_completed);
    println!("  Converged: {}", results.converged);
    println!("  Runtime: {:.2?}", runtime);

    // Scaling analysis
    println!("\n  Scaling metrics:");
    println!("    Qubits: {}", problem_size);
    println!("    Parameters: {}", results.optimal_parameters.len());
    println!(
        "    Function evaluations: {}",
        results.statistics.function_evaluations
    );
    println!(
        "    Evaluations per qubit: {:.1}",
        results.statistics.function_evaluations as f64 / problem_size as f64
    );
    println!(
        "    Runtime per iteration: {:.2?}",
        runtime / results.iterations_completed.max(1) as u32
    );

    // Solution quality analysis
    let solution_energy = evaluate_ising_energy(&results.best_solution, &problem)?;
    println!("\n  Solution quality:");
    println!("    Computed energy: {:.6}", solution_energy);
    println!("    VQA best energy: {:.6}", results.best_energy);
    println!(
        "    Energy consistency: {}",
        if (solution_energy - results.best_energy).abs() < 1e-3 {
            "✓"
        } else {
            "✗"
        }
    );

    // Convergence behavior
    if results.energy_history.len() >= 20 {
        let early_progress = results.energy_history[0] - results.energy_history[19];
        let late_progress = results.energy_history[results.energy_history.len().saturating_sub(20)]
            - *results.energy_history.last().unwrap();

        println!("    Early progress (first 20 iters): {:.6}", early_progress);
        println!("    Late progress (last 20 iters): {:.6}", late_progress);
        println!(
            "    Convergence pattern: {}",
            if early_progress > late_progress * 5.0 {
                "Fast early"
            } else {
                "Steady"
            }
        );
    }

    Ok(())
}

/// Helper function to count frustrated bonds in a ring
fn count_frustrated_bonds(solution: &[i8], problem: &IsingModel) -> usize {
    let mut frustrated = 0;

    for i in 0..solution.len() {
        let j = (i + 1) % solution.len();
        if let Ok(coupling) = problem.get_coupling(i, j) {
            if coupling != 0.0 {
                // Bond is frustrated if spins are aligned but coupling is positive,
                // or spins are anti-aligned but coupling is negative
                let aligned = solution[i] == solution[j];
                let should_be_aligned = coupling < 0.0;

                if aligned != should_be_aligned {
                    frustrated += 1;
                }
            }
        }
    }

    frustrated
}

/// Helper function to evaluate Ising energy
fn evaluate_ising_energy(
    spins: &[i8],
    problem: &IsingModel,
) -> Result<f64, Box<dyn std::error::Error>> {
    let mut energy = 0.0;

    // Bias terms
    for i in 0..spins.len() {
        energy += problem.get_bias(i)? * spins[i] as f64;
    }

    // Coupling terms
    for i in 0..spins.len() {
        for j in (i + 1)..spins.len() {
            energy += problem.get_coupling(i, j)? * spins[i] as f64 * spins[j] as f64;
        }
    }

    Ok(energy)
}
