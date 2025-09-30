//! Demonstration of advanced variational parameter optimization using SciRS2

use quantrs2_core::{
    prelude::*,
    variational::{VariationalGate, VariationalCircuit},
    variational_optimization::{
        VariationalQuantumOptimizer, OptimizationMethod, OptimizationConfig,
        ConstrainedVariationalOptimizer, HyperparameterOptimizer,
        create_vqe_optimizer, create_qaoa_optimizer, create_natural_gradient_optimizer,
    },
    gate::single::{Hadamard, PauliX},
};
use scirs2_core::Complex64;
use scirs2_core::ndarray::Array2;
use rustc_hash::FxHashMap;
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== QuantRS2 Variational Optimization Demo ===\n");

    // Demo 1: Basic gradient descent optimization
    demo_gradient_descent()?;

    // Demo 2: Advanced optimizers (Adam, RMSprop)
    demo_advanced_optimizers()?;

    // Demo 3: SciRS2 optimizers (BFGS, L-BFGS)
    demo_scirs2_optimizers()?;

    // Demo 4: Natural gradient descent
    demo_natural_gradient()?;

    // Demo 5: SPSA for noisy quantum devices
    demo_spsa_optimization()?;

    // Demo 6: Constrained optimization
    demo_constrained_optimization()?;

    // Demo 7: VQE optimization
    demo_vqe_optimization()?;

    // Demo 8: Hyperparameter optimization
    demo_hyperparameter_optimization()?;

    Ok(())
}

fn demo_gradient_descent() -> Result<(), Box<dyn std::error::Error>> {
    println!("Demo 1: Basic Gradient Descent Optimization");
    println!("==========================================");

    // Create a simple variational circuit
    let mut circuit = VariationalCircuit::new(2);
    circuit.add_gate(VariationalGate::rx(QubitId(0), "theta1".to_string(), 0.1));
    circuit.add_gate(VariationalGate::ry(QubitId(1), "theta2".to_string(), 0.2));
    circuit.add_gate(VariationalGate::cry(QubitId(0), QubitId(1), "theta3".to_string(), 0.3));

    // Define cost function: minimize <ψ|Z₀Z₁|ψ>
    let cost_fn = |circuit: &VariationalCircuit| -> QuantRS2Result<f64> {
        // Simplified cost function for demonstration
        let params = circuit.get_parameters();
        let theta1 = params.get("theta1").copied().unwrap_or(0.0);
        let theta2 = params.get("theta2").copied().unwrap_or(0.0);
        let theta3 = params.get("theta3").copied().unwrap_or(0.0);

        // Simulate expectation value
        let cost = theta1.cos() * theta2.cos() - 0.5 * theta3.sin();
        Ok(cost)
    };

    // Create optimizer
    let config = OptimizationConfig {
        max_iterations: 50,
        f_tol: 1e-6,
        g_tol: 1e-6,
        parallel_gradients: true,
        ..Default::default()
    };

    let mut optimizer = VariationalQuantumOptimizer::new(
        OptimizationMethod::GradientDescent { learning_rate: 0.1 },
        config,
    );

    // Run optimization
    let result = optimizer.optimize(&mut circuit, cost_fn)?;

    println!("Initial parameters:");
    println!("  theta1: 0.1");
    println!("  theta2: 0.2");
    println!("  theta3: 0.3");
    println!("\nOptimized parameters:");
    for (name, value) in &result.optimal_parameters {
        println!("  {}: {:.6}", name, value);
    }
    println!("Final loss: {:.6}", result.final_loss);
    println!("Iterations: {}", result.iterations);
    println!("Converged: {}\n", result.converged);

    Ok(())
}

fn demo_advanced_optimizers() -> Result<(), Box<dyn std::error::Error>> {
    println!("Demo 2: Advanced Optimizers (Adam, RMSprop)");
    println!("==========================================");

    // Create circuit with more parameters
    let mut circuit = VariationalCircuit::new(3);
    for i in 0..3 {
        circuit.add_gate(VariationalGate::rx(
            QubitId(i as u32),
            format!("rx_{}", i),
            0.5,
        ));
        circuit.add_gate(VariationalGate::ry(
            QubitId(i as u32),
            format!("ry_{}", i),
            0.5,
        ));
    }

    // Cost function with multiple local minima
    let cost_fn = |circuit: &VariationalCircuit| -> QuantRS2Result<f64> {
        let params = circuit.get_parameters();
        let mut cost = 0.0;

        for i in 0..3 {
            let rx = params.get(&format!("rx_{}", i)).copied().unwrap_or(0.0);
            let ry = params.get(&format!("ry_{}", i)).copied().unwrap_or(0.0);
            cost += (rx - PI/4.0).powi(2) + (ry - PI/3.0).powi(2);
            cost += 0.1 * (2.0 * rx).sin() * (3.0 * ry).cos();
        }

        Ok(cost)
    };

    // Test Adam optimizer
    println!("Testing Adam optimizer:");
    let config = OptimizationConfig {
        max_iterations: 100,
        ..Default::default()
    };

    let mut adam_optimizer = VariationalQuantumOptimizer::new(
        OptimizationMethod::Adam {
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        },
        config.clone(),
    );

    let adam_result = adam_optimizer.optimize(&mut circuit.clone(), cost_fn.clone())?;
    println!("  Final loss: {:.6}", adam_result.final_loss);
    println!("  Iterations: {}", adam_result.iterations);

    // Test RMSprop optimizer
    println!("\nTesting RMSprop optimizer:");
    let mut rmsprop_optimizer = VariationalQuantumOptimizer::new(
        OptimizationMethod::RMSprop {
            learning_rate: 0.01,
            decay_rate: 0.9,
            epsilon: 1e-8,
        },
        config,
    );

    let rmsprop_result = rmsprop_optimizer.optimize(&mut circuit, cost_fn)?;
    println!("  Final loss: {:.6}", rmsprop_result.final_loss);
    println!("  Iterations: {}\n", rmsprop_result.iterations);

    Ok(())
}

fn demo_scirs2_optimizers() -> Result<(), Box<dyn std::error::Error>> {
    println!("Demo 3: SciRS2 Optimizers (BFGS, L-BFGS)");
    println!("=========================================");

    let mut circuit = VariationalCircuit::new(2);
    circuit.add_gate(VariationalGate::rz(QubitId(0), "alpha".to_string(), 0.0));
    circuit.add_gate(VariationalGate::ry(QubitId(0), "beta".to_string(), 0.0));
    circuit.add_gate(VariationalGate::rz(QubitId(0), "gamma".to_string(), 0.0));

    // Rosenbrock-like cost function
    let cost_fn = |circuit: &VariationalCircuit| -> QuantRS2Result<f64> {
        let params = circuit.get_parameters();
        let a = params.get("alpha").copied().unwrap_or(0.0);
        let b = params.get("beta").copied().unwrap_or(0.0);
        let c = params.get("gamma").copied().unwrap_or(0.0);

        let cost = 100.0 * (b - a.powi(2)).powi(2) + (1.0 - a).powi(2) + 50.0 * (c - b).powi(2);
        Ok(cost)
    };

    // Test BFGS
    println!("Testing BFGS optimizer:");
    let mut bfgs_optimizer = VariationalQuantumOptimizer::new(
        OptimizationMethod::BFGS,
        Default::default(),
    );

    let bfgs_result = bfgs_optimizer.optimize(&mut circuit.clone(), cost_fn.clone())?;
    println!("  Final loss: {:.6}", bfgs_result.final_loss);
    println!("  Iterations: {}", bfgs_result.iterations);

    // Test L-BFGS
    println!("\nTesting L-BFGS optimizer:");
    let mut lbfgs_optimizer = VariationalQuantumOptimizer::new(
        OptimizationMethod::LBFGS { memory_size: 10 },
        Default::default(),
    );

    let lbfgs_result = lbfgs_optimizer.optimize(&mut circuit, cost_fn)?;
    println!("  Final loss: {:.6}", lbfgs_result.final_loss);
    println!("  Iterations: {}\n", lbfgs_result.iterations);

    Ok(())
}

fn demo_natural_gradient() -> Result<(), Box<dyn std::error::Error>> {
    println!("Demo 4: Natural Gradient Descent");
    println!("================================");

    let mut circuit = VariationalCircuit::new(2);
    circuit.add_gate(VariationalGate::ry(QubitId(0), "theta1".to_string(), 0.1));
    circuit.add_gate(VariationalGate::ry(QubitId(1), "theta2".to_string(), 0.2));

    // Cost function with ill-conditioned landscape
    let cost_fn = |circuit: &VariationalCircuit| -> QuantRS2Result<f64> {
        let params = circuit.get_parameters();
        let t1 = params.get("theta1").copied().unwrap_or(0.0);
        let t2 = params.get("theta2").copied().unwrap_or(0.0);

        // Elongated valley
        Ok(10.0 * (t1 - 1.0).powi(2) + 0.1 * (t2 - 1.0).powi(2))
    };

    // Compare standard gradient descent with natural gradient
    println!("Standard gradient descent:");
    let mut gd_optimizer = VariationalQuantumOptimizer::new(
        OptimizationMethod::GradientDescent { learning_rate: 0.01 },
        OptimizationConfig {
            max_iterations: 100,
            ..Default::default()
        },
    );

    let gd_result = gd_optimizer.optimize(&mut circuit.clone(), cost_fn.clone())?;
    println!("  Iterations: {}", gd_result.iterations);
    println!("  Final loss: {:.6}", gd_result.final_loss);

    println!("\nNatural gradient descent:");
    let mut natural_optimizer = create_natural_gradient_optimizer(0.1);
    let natural_result = natural_optimizer.optimize(&mut circuit, cost_fn)?;
    println!("  Iterations: {}", natural_result.iterations);
    println!("  Final loss: {:.6}\n", natural_result.final_loss);

    Ok(())
}

fn demo_spsa_optimization() -> Result<(), Box<dyn std::error::Error>> {
    println!("Demo 5: SPSA for Noisy Quantum Devices");
    println!("======================================");

    let mut circuit = VariationalCircuit::new(2);
    circuit.add_gate(VariationalGate::rx(QubitId(0), "x".to_string(), 0.0));
    circuit.add_gate(VariationalGate::ry(QubitId(1), "y".to_string(), 0.0));

    // Noisy cost function
    let cost_fn = |circuit: &VariationalCircuit| -> QuantRS2Result<f64> {
        use rand::{Rng, thread_rng};
        let mut rng = thread_rng();

        let params = circuit.get_parameters();
        let x = params.get("x").copied().unwrap_or(0.0);
        let y = params.get("y").copied().unwrap_or(0.0);

        // True cost with added noise
        let true_cost = (x - PI/4.0).powi(2) + (y - PI/3.0).powi(2);
        let noise = rng.gen_range(-0.1..0.1);

        Ok(true_cost + noise)
    };

    let mut spsa_optimizer = create_spsa_optimizer();
    let result = spsa_optimizer.optimize(&mut circuit, cost_fn)?;

    println!("Optimized parameters (target: x=π/4, y=π/3):");
    println!("  x: {:.6} (target: {:.6})", result.optimal_parameters["x"], PI/4.0);
    println!("  y: {:.6} (target: {:.6})", result.optimal_parameters["y"], PI/3.0);
    println!("  Iterations: {}\n", result.iterations);

    Ok(())
}

fn demo_constrained_optimization() -> Result<(), Box<dyn std::error::Error>> {
    println!("Demo 6: Constrained Optimization");
    println!("================================");

    let mut circuit = VariationalCircuit::new(2);
    circuit.add_gate(VariationalGate::rx(QubitId(0), "theta1".to_string(), 0.0));
    circuit.add_gate(VariationalGate::ry(QubitId(1), "theta2".to_string(), 0.0));

    // Minimize theta1^2 + theta2^2 subject to theta1 + theta2 = 1
    let cost_fn = |circuit: &VariationalCircuit| -> QuantRS2Result<f64> {
        let params = circuit.get_parameters();
        let t1 = params.get("theta1").copied().unwrap_or(0.0);
        let t2 = params.get("theta2").copied().unwrap_or(0.0);
        Ok(t1.powi(2) + t2.powi(2))
    };

    let base_optimizer = VariationalQuantumOptimizer::new(
        OptimizationMethod::BFGS,
        Default::default(),
    );

    let mut constrained_opt = ConstrainedVariationalOptimizer::new(base_optimizer);

    // Add equality constraint: theta1 + theta2 = 1
    constrained_opt.add_equality_constraint(
        |params| {
            let t1 = params.get("theta1").copied().unwrap_or(0.0);
            let t2 = params.get("theta2").copied().unwrap_or(0.0);
            t1 + t2
        },
        1.0,
    );

    let result = constrained_opt.optimize(&mut circuit, cost_fn)?;

    println!("Optimized parameters (should satisfy theta1 + theta2 = 1):");
    let t1 = result.optimal_parameters["theta1"];
    let t2 = result.optimal_parameters["theta2"];
    println!("  theta1: {:.6}", t1);
    println!("  theta2: {:.6}", t2);
    println!("  Sum: {:.6} (target: 1.0)", t1 + t2);
    println!("  Final loss: {:.6}\n", result.final_loss);

    Ok(())
}

fn demo_vqe_optimization() -> Result<(), Box<dyn std::error::Error>> {
    println!("Demo 7: VQE Optimization");
    println!("========================");

    // Create a simple ansatz for H2 molecule
    let mut circuit = VariationalCircuit::new(4);

    // Initial state preparation
    circuit.add_gate(VariationalGate::ry(QubitId(0), "theta1".to_string(), 0.1));
    circuit.add_gate(VariationalGate::ry(QubitId(1), "theta2".to_string(), 0.1));
    circuit.add_gate(VariationalGate::ry(QubitId(2), "theta3".to_string(), 0.1));
    circuit.add_gate(VariationalGate::ry(QubitId(3), "theta4".to_string(), 0.1));

    // Entangling layer
    circuit.add_gate(VariationalGate::cry(QubitId(0), QubitId(1), "phi1".to_string(), 0.1));
    circuit.add_gate(VariationalGate::cry(QubitId(2), QubitId(3), "phi2".to_string(), 0.1));

    // Simplified H2 Hamiltonian energy function
    let cost_fn = |circuit: &VariationalCircuit| -> QuantRS2Result<f64> {
        let params = circuit.get_parameters();

        // Simulate energy expectation value
        let mut energy = -1.0;  // Ground state offset

        for (name, &value) in params {
            if name.starts_with("theta") {
                energy += 0.1 * value.cos();
            } else if name.starts_with("phi") {
                energy += 0.05 * value.sin();
            }
        }

        Ok(energy)
    };

    let mut vqe_optimizer = create_vqe_optimizer();
    let result = vqe_optimizer.optimize(&mut circuit, cost_fn)?;

    println!("VQE Results:");
    println!("  Ground state energy: {:.6}", result.final_loss);
    println!("  Iterations: {}", result.iterations);
    println!("  Converged: {}", result.converged);
    println!("  Optimal parameters:");
    for (name, value) in result.optimal_parameters.iter().take(4) {
        println!("    {}: {:.6}", name, value);
    }
    println!();

    Ok(())
}

fn demo_hyperparameter_optimization() -> Result<(), Box<dyn std::error::Error>> {
    println!("Demo 8: Hyperparameter Optimization");
    println!("===================================");

    // Define circuit builder with hyperparameters
    let circuit_builder = |hyperparams: &FxHashMap<String, f64>| -> VariationalCircuit {
        let depth = hyperparams.get("depth").copied().unwrap_or(1.0) as usize;
        let rotation_scale = hyperparams.get("rotation_scale").copied().unwrap_or(1.0);

        let mut circuit = VariationalCircuit::new(2);

        for d in 0..depth {
            circuit.add_gate(VariationalGate::rx(
                QubitId(0),
                format!("rx_{}_{}", 0, d),
                0.1 * rotation_scale,
            ));
            circuit.add_gate(VariationalGate::ry(
                QubitId(1),
                format!("ry_{}_{}", 1, d),
                0.1 * rotation_scale,
            ));
            circuit.add_gate(VariationalGate::cry(
                QubitId(0),
                QubitId(1),
                format!("cry_{}", d),
                0.1,
            ));
        }

        circuit
    };

    // Cost function
    let cost_fn = |circuit: &VariationalCircuit| -> QuantRS2Result<f64> {
        let params = circuit.get_parameters();
        let mut cost = 0.0;

        for (_, &value) in params {
            cost += (value - PI/4.0).powi(2);
        }

        Ok(cost)
    };

    let mut hyperparam_opt = HyperparameterOptimizer::new(10);
    hyperparam_opt.add_hyperparameter("depth".to_string(), 1.0, 5.0);
    hyperparam_opt.add_hyperparameter("rotation_scale".to_string(), 0.5, 2.0);

    let result = hyperparam_opt.optimize(circuit_builder, cost_fn)?;

    println!("Best hyperparameters:");
    for (name, value) in &result.best_hyperparameters {
        println!("  {}: {:.6}", name, value);
    }
    println!("Best loss achieved: {:.6}", result.best_loss);
    println!("Total trials: {}", result.all_trials.len());

    Ok(())
}