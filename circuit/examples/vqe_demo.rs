//! Variational Quantum Eigensolver (VQE) demonstration
//!
//! This example shows how to use the VQE module to find ground state energies
//! of quantum systems using different ansätze and optimization methods.

use quantrs2_circuit::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧮 Variational Quantum Eigensolver (VQE) Demo");
    println!("============================================\n");

    // Example 1: Hardware-efficient ansatz for small molecules
    println!("1. Hardware-Efficient Ansatz for 4-qubit system");
    println!("-----------------------------------------------");

    let mut vqe_circuit = VQECircuit::<4>::new(VQEAnsatz::HardwareEfficient { layers: 2 })?;
    println!(
        "Created VQE circuit with {} parameters",
        vqe_circuit.num_parameters()
    );

    // Set some example parameters
    let param_names = vqe_circuit.parameter_names.clone();
    for (i, param_name) in param_names.iter().enumerate() {
        vqe_circuit.set_parameter(param_name, 0.1 * i as f64)?;
    }

    println!("Set parameters: {:?}", vqe_circuit.parameters);
    println!("Parameter names: {:?}\n", vqe_circuit.parameter_names);

    // Example 2: UCCSD ansatz for quantum chemistry
    println!("2. UCCSD Ansatz for quantum chemistry");
    println!("--------------------------------------");

    let uccsd_circuit = VQECircuit::<6>::new(VQEAnsatz::UCCSD {
        occupied_orbitals: 2,
        virtual_orbitals: 4,
    })?;
    println!(
        "Created UCCSD circuit with {} parameters",
        uccsd_circuit.num_parameters()
    );
    println!("Circuit has {} gates", uccsd_circuit.circuit.num_gates());

    // Example 3: Observables for different Hamiltonians
    println!("\n3. Creating Observables");
    println!("----------------------");

    // Heisenberg model
    let heisenberg = VQEObservable::heisenberg_model(4, 1.0);
    println!("Heisenberg model with {} terms", heisenberg.terms.len());

    // Transverse Field Ising Model
    let tfim = VQEObservable::tfim(4, 1.0, 0.5);
    println!("TFIM model with {} terms", tfim.terms.len());

    // Custom observable
    let mut custom_obs = VQEObservable::new();
    custom_obs.add_pauli_term(1.0, vec![(0, PauliOperator::Z), (1, PauliOperator::Z)]);
    custom_obs.add_pauli_term(0.5, vec![(0, PauliOperator::X)]);
    custom_obs.add_pauli_term(0.5, vec![(1, PauliOperator::X)]);
    println!("Custom observable with {} terms", custom_obs.terms.len());

    // Example 4: Custom ansatz construction
    println!("\n4. Custom Ansatz Construction");
    println!("-----------------------------");

    let mut custom_vqe = VQECircuit::<3>::new(VQEAnsatz::Custom)?;

    // Add custom parameterized gates
    custom_vqe.add_parameterized_ry(Qubit(0), "theta_0")?;
    custom_vqe.add_parameterized_rz(Qubit(1), "phi_1")?;
    custom_vqe.add_parameterized_ry(Qubit(2), "theta_2")?;

    // Add fixed gates to the underlying circuit
    custom_vqe.circuit.cnot(Qubit(0), Qubit(1))?;
    custom_vqe.circuit.cnot(Qubit(1), Qubit(2))?;

    println!(
        "Custom VQE circuit with {} parameters",
        custom_vqe.num_parameters()
    );

    // Set parameter values
    custom_vqe.set_parameter("theta_0", std::f64::consts::PI / 4.0)?;
    custom_vqe.set_parameter("phi_1", std::f64::consts::PI / 2.0)?;
    custom_vqe.set_parameter("theta_2", std::f64::consts::PI / 6.0)?;

    println!(
        "theta_0 = {:.4}",
        custom_vqe.get_parameter("theta_0").unwrap()
    );
    println!("phi_1 = {:.4}", custom_vqe.get_parameter("phi_1").unwrap());
    println!(
        "theta_2 = {:.4}",
        custom_vqe.get_parameter("theta_2").unwrap()
    );

    // Example 5: VQE Optimization (mock demonstration)
    println!("\n5. VQE Optimization");
    println!("-------------------");

    let optimizer = VQEOptimizer::new(VQEOptimizerType::GradientDescent);
    println!(
        "Created optimizer with max iterations: {}",
        optimizer.max_iterations
    );
    println!("Tolerance: {:.2e}", optimizer.tolerance);
    println!("Learning rate: {}", optimizer.learning_rate);

    // In a real implementation, this would actually optimize the parameters
    let result = optimizer.optimize(&mut custom_vqe, &custom_obs)?;
    println!("Optimization result:");
    println!("  Ground state energy: {:.6}", result.ground_state_energy);
    println!("  Converged: {}", result.converged);
    println!("  Iterations: {}", result.iterations);
    println!("  Gradient norm: {:.2e}", result.gradient_norm);

    // Example 6: Different optimizer types
    println!("\n6. Different Optimizer Types");
    println!("----------------------------");

    let optimizers = vec![
        ("Gradient Descent", VQEOptimizerType::GradientDescent),
        (
            "Adam",
            VQEOptimizerType::Adam {
                beta1: 0.9,
                beta2: 0.999,
            },
        ),
        ("BFGS", VQEOptimizerType::BFGS),
        ("Nelder-Mead", VQEOptimizerType::NelderMead),
        (
            "SPSA",
            VQEOptimizerType::SPSA {
                alpha: 0.602,
                gamma: 0.101,
            },
        ),
    ];

    for (name, opt_type) in optimizers {
        let opt = VQEOptimizer::new(opt_type);
        println!("  {}: {:?}", name, opt.optimizer_type);
    }

    // Example 7: Real-space ansatz
    println!("\n7. Real-Space Ansatz");
    println!("--------------------");

    let geometry = vec![
        (0.0, 0.0, 0.0), // Site 0
        (1.0, 0.0, 0.0), // Site 1
        (0.0, 1.0, 0.0), // Site 2
        (1.0, 1.0, 0.0), // Site 3
    ];

    let real_space_circuit = VQECircuit::<4>::new(VQEAnsatz::RealSpace { geometry })?;
    println!(
        "Real-space circuit with {} parameters",
        real_space_circuit.num_parameters()
    );
    println!("Based on 2x2 square lattice geometry");

    // Example 8: Molecular Hamiltonian
    println!("\n8. Molecular Hamiltonian");
    println!("------------------------");

    // Example one-body and two-body integrals (simplified)
    let one_body = vec![
        (0, 0, -1.0), // h_00
        (1, 1, -1.0), // h_11
        (0, 1, -0.5), // h_01
    ];

    let two_body = vec![
        (0, 1, 0, 1, 0.5), // (00|11)
        (0, 0, 1, 1, 0.3), // (01|01)
    ];

    let molecular_ham = VQEObservable::molecular_hamiltonian(&one_body, &two_body);
    println!(
        "Molecular Hamiltonian with {} terms",
        molecular_ham.terms.len()
    );

    println!("\n✅ VQE Demo completed successfully!");
    println!("\nNote: This demo shows the VQE framework structure.");
    println!("Real VQE optimization requires quantum simulation or hardware execution.");

    Ok(())
}
