//! VQE Example with OptiRS Optimization
//!
//! This example demonstrates how to use the Variational Quantum Eigensolver (VQE)
//! algorithm with OptiRS's state-of-the-art optimizers (Adam, SGD, RMSprop, etc.)
//! for finding ground state energies of molecular systems.
//!
//! Run with: cargo run --example vqe_with_optirs --features optimize

use quantrs2_sim::autodiff_vqe::{
    AutoDiffContext, ConvergenceCriteria, GradientMethod, ParametricCircuit, ParametricRX,
    ParametricRY, ParametricRZ, VQEWithAutodiff,
};
use quantrs2_sim::error::Result;
use quantrs2_sim::optirs_integration::{OptiRSConfig, OptiRSOptimizerType};
use quantrs2_sim::pauli::{PauliOperatorSum, PauliString};
use scirs2_core::Complex64;
use std::time::Instant;

/// Helper function to build the ansatz circuit
fn build_ansatz(num_qubits: usize, num_layers: usize) -> ParametricCircuit {
    let mut ansatz = ParametricCircuit::new(num_qubits);
    let mut param_idx = 0;

    for _layer in 0..num_layers {
        // Rotation layer
        for qubit in 0..num_qubits {
            ansatz.add_gate(Box::new(ParametricRY { qubit, param_idx }));
            param_idx += 1;
        }

        // Z rotations
        for qubit in 0..num_qubits {
            ansatz.add_gate(Box::new(ParametricRZ { qubit, param_idx }));
            param_idx += 1;
        }
    }

    ansatz.num_parameters = param_idx;
    ansatz
}

/// Helper function to build H2 Hamiltonian
fn build_h2_hamiltonian() -> Result<PauliOperatorSum> {
    let mut hamiltonian = PauliOperatorSum::new(2); // 2 qubits
    hamiltonian.add_term(PauliString::from_string(
        "ZZ",
        Complex64::new(-1.0524, 0.0),
    )?)?;
    hamiltonian.add_term(PauliString::from_string("ZI", Complex64::new(0.3979, 0.0))?)?;
    hamiltonian.add_term(PauliString::from_string(
        "IZ",
        Complex64::new(-0.3979, 0.0),
    )?)?;
    hamiltonian.add_term(PauliString::from_string(
        "XX",
        Complex64::new(-0.0112, 0.0),
    )?)?;
    Ok(hamiltonian)
}

fn main() -> Result<()> {
    println!("=== VQE with OptiRS Optimization Demo ===\n");

    // Define H2 Hamiltonian
    println!("Hamiltonian:");
    println!("  H = -1.0524·ZZ + 0.3979·ZI - 0.3979·IZ - 0.0112·XX");
    println!("  Expected ground state energy: ~-1.857 Ha\n");

    // Circuit parameters
    let num_qubits = 2;
    let num_layers = 2;
    let num_parameters = num_qubits * num_layers * 2; // RY and RZ per qubit per layer

    println!(
        "Ansatz: Hardware-efficient with {} parameters",
        num_parameters
    );
    println!("  {} layers, {} qubits\n", num_layers, num_qubits);

    // Initialize parameters
    let initial_params = vec![0.0; num_parameters];

    // Create autodiff context
    let context = AutoDiffContext::new(
        initial_params.clone(),
        GradientMethod::FiniteDifference { step_size: 1e-4 },
    );

    // Set convergence criteria
    let convergence = ConvergenceCriteria {
        max_iterations: 100,
        energy_tolerance: 1e-6,
        gradient_tolerance: 1e-5,
        max_func_evals: 500,
    };

    // Test different OptiRS optimizers
    let optimizers = vec![
        (
            "SGD with Momentum",
            OptiRSOptimizerType::SGD { momentum: true },
        ),
        ("Adam", OptiRSOptimizerType::Adam),
        ("RMSprop", OptiRSOptimizerType::RMSprop),
        ("Adagrad", OptiRSOptimizerType::Adagrad),
    ];

    for (optimizer_name, optimizer_type) in optimizers {
        println!("\n=== Testing {} ===", optimizer_name);

        // Create fresh ansatz and hamiltonian for each run
        let ansatz = build_ansatz(num_qubits, num_layers);
        let hamiltonian = build_h2_hamiltonian()?;

        // Create VQE instance
        let mut vqe = VQEWithAutodiff {
            ansatz,
            hamiltonian,
            context: AutoDiffContext::new(
                initial_params.clone(),
                GradientMethod::FiniteDifference { step_size: 1e-4 },
            ),
            history: Vec::new(),
            convergence: ConvergenceCriteria {
                max_iterations: 100,
                energy_tolerance: 1e-6,
                gradient_tolerance: 1e-5,
                max_func_evals: 500,
            },
        };

        // Configure OptiRS optimizer
        let optirs_config = OptiRSConfig {
            optimizer_type,
            learning_rate: match optimizer_type {
                OptiRSOptimizerType::Adam => 0.01,
                OptiRSOptimizerType::SGD { .. } => 0.1,
                OptiRSOptimizerType::RMSprop => 0.01,
                OptiRSOptimizerType::Adagrad => 0.1,
            },
            convergence_tolerance: 1e-6,
            max_iterations: 100,
            gradient_clip_norm: Some(1.0),
            parameter_bounds: Some((-std::f64::consts::PI, std::f64::consts::PI)),
            ..Default::default()
        };

        // Run optimization
        let start = Instant::now();
        let result = vqe.optimize_with_optirs(optirs_config)?;
        let elapsed = start.elapsed();

        // Print results
        println!("  Converged: {}", result.converged);
        println!("  Final energy: {:.6} Ha", result.optimal_energy);
        println!("  Iterations: {}", result.iterations);
        println!("  Time: {:.2?}", elapsed);
        println!(
            "  Error from expected: {:.6} Ha",
            (result.optimal_energy + 1.857).abs()
        );

        // Print final parameters
        println!("  Final parameters:");
        for (i, param) in result.optimal_parameters.iter().enumerate() {
            println!("    θ{}: {:.4}", i, param);
        }

        // Print optimization history (last 5 iterations)
        if result.history.len() > 5 {
            println!("\n  Last 5 iterations:");
            for iter in result.history.iter().rev().take(5).rev() {
                println!(
                    "    Iter {}: E = {:.6} Ha, |∇| = {:.6}",
                    iter.iteration, iter.energy, iter.gradient_norm
                );
            }
        }
    }

    println!("\n=== Comparison with basic gradient descent ===");

    // Run with basic gradient descent for comparison
    let ansatz_gd = build_ansatz(num_qubits, num_layers);
    let hamiltonian_gd = build_h2_hamiltonian()?;

    let mut vqe_gd = VQEWithAutodiff {
        ansatz: ansatz_gd,
        hamiltonian: hamiltonian_gd,
        context: AutoDiffContext::new(
            initial_params.clone(),
            GradientMethod::FiniteDifference { step_size: 1e-4 },
        ),
        history: Vec::new(),
        convergence: ConvergenceCriteria {
            max_iterations: 100,
            energy_tolerance: 1e-6,
            gradient_tolerance: 1e-5,
            max_func_evals: 500,
        },
    };

    let start = Instant::now();
    let result_gd = vqe_gd.optimize(0.1)?;
    let elapsed_gd = start.elapsed();

    println!("  Converged: {}", result_gd.converged);
    println!("  Final energy: {:.6} Ha", result_gd.optimal_energy);
    println!("  Iterations: {}", result_gd.iterations);
    println!("  Time: {:.2?}", elapsed_gd);
    println!(
        "  Error from expected: {:.6} Ha",
        (result_gd.optimal_energy + 1.857).abs()
    );

    println!("\n=== Summary ===");
    println!("OptiRS optimizers typically converge faster and more reliably");
    println!("than basic gradient descent for VQE problems, especially Adam.");
    println!("The adaptive learning rate and momentum help escape local minima.");

    Ok(())
}
