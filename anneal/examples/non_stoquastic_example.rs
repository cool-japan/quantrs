#![allow(clippy::pedantic, clippy::unnecessary_wraps)]
//! Example demonstrating Non-Stoquastic Hamiltonian Support
//!
//! This example shows how to:
//! 1. Create various non-stoquastic Hamiltonians (XY, XYZ, complex Ising)
//! 2. Analyze the sign problem and its severity
//! 3. Run quantum Monte Carlo simulations with sign problem mitigation
//! 4. Compare non-stoquastic vs stoquastic approximations
//! 5. Use population annealing for complex systems
//! 6. Examine quantum advantages in non-stoquastic systems
//! 7. Convert between different Hamiltonian representations

use quantrs2_anneal::{
    ising::IsingModel,
    non_stoquastic::{
        create_frustrated_xy_triangle, create_tfxy_model, create_xy_chain,
        xy_to_ising_approximation, ComplexCoupling, HamiltonianType, InteractionType,
        NonStoquasticHamiltonian, NonStoquasticQMCConfig, NonStoquasticSimulator,
        SignMitigationStrategy,
    },
    simulator::{AnnealingParams, ClassicalAnnealingSimulator},
};
use scirs2_core::Complex as NComplex;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Non-Stoquastic Hamiltonian Support Demo ===\n");

    // Example 1: XY Model
    println!("Example 1: XY Model Analysis");
    xy_model_example()?;

    // Example 2: XYZ (Heisenberg) Model
    println!("\nExample 2: XYZ (Heisenberg) Model");
    xyz_model_example()?;

    // Example 3: Sign Problem Analysis
    println!("\nExample 3: Sign Problem Analysis");
    sign_problem_analysis_example()?;

    // Example 4: Quantum Monte Carlo Simulation
    println!("\nExample 4: Quantum Monte Carlo with Sign Problem");
    quantum_monte_carlo_example()?;

    // Example 5: Population Annealing
    println!("\nExample 5: Population Annealing for Non-Stoquastic Systems");
    population_annealing_example()?;

    // Example 6: Stoquastic vs Non-Stoquastic Comparison
    println!("\nExample 6: Stoquastic vs Non-Stoquastic Comparison");
    stoquastic_comparison_example()?;

    // Example 7: Complex Weighted Hamiltonians
    println!("\nExample 7: Complex Weighted Hamiltonians");
    complex_hamiltonian_example()?;

    // Example 8: Frustrated Systems
    println!("\nExample 8: Frustrated XY Systems");
    frustrated_systems_example()?;

    Ok(())
}

fn xy_model_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Creating and analyzing XY spin chain...");

    // Create XY model with different coupling strengths
    let num_qubits = 6;
    let j_x = 1.0;
    let j_y = 0.8;

    let xy_hamiltonian = NonStoquasticHamiltonian::xy_model(num_qubits, j_x, j_y)?;

    println!("    System size: {} qubits", xy_hamiltonian.num_qubits);
    println!(
        "    Hamiltonian type: {:?}",
        xy_hamiltonian.hamiltonian_type
    );
    println!("    Has sign problem: {}", xy_hamiltonian.has_sign_problem);
    println!(
        "    Number of coupling terms: {}",
        xy_hamiltonian.complex_couplings.len()
    );

    // Analyze coupling structure
    println!("\n    Coupling structure:");
    for (i, coupling) in xy_hamiltonian.complex_couplings.iter().enumerate() {
        println!(
            "      Coupling {}: sites ({}, {}), type {:?}, strength {:.3}",
            i + 1,
            coupling.sites.0,
            coupling.sites.1,
            coupling.interaction_type,
            coupling.strength.norm()
        );
    }

    // Check sign problem severity
    let sign_severity = xy_hamiltonian.sign_problem_severity();
    println!("\n    Sign problem severity: {sign_severity:.2}");

    if sign_severity > 0.5 {
        println!("      ⚠️  Severe sign problem - advanced methods required");
    } else if sign_severity > 0.1 {
        println!("      ⚡ Moderate sign problem - mitigation strategies recommended");
    } else {
        println!("      ✅ Mild sign problem - standard methods should work");
    }

    // Add transverse fields
    let mut tfxy = xy_hamiltonian;
    for site in 0..num_qubits {
        let field_strength = 0.5 * (site as f64 / num_qubits as f64 - 0.5);
        tfxy.set_local_field(site, field_strength)?;
    }

    println!("\n    Added transverse fields:");
    for (site, &field) in tfxy.local_fields.iter().enumerate() {
        if field.abs() > 1e-10 {
            println!("      Site {site}: field = {field:.3}");
        }
    }

    // Matrix representation for small system
    if num_qubits <= 4 {
        println!("\n    Computing exact matrix representation...");
        let start = Instant::now();
        let matrix = tfxy.to_matrix()?;
        let matrix_time = start.elapsed();

        println!("      Matrix size: {}×{}", matrix.len(), matrix[0].len());
        println!("      Computation time: {matrix_time:.2?}");

        // Check Hermiticity
        let mut max_hermiticity_error: f64 = 0.0;
        for i in 0..matrix.len() {
            for j in 0..matrix[i].len() {
                let error = (matrix[i][j] - matrix[j][i].conj()).norm();
                max_hermiticity_error = max_hermiticity_error.max(error);
            }
        }
        println!("      Hermiticity check: max error = {max_hermiticity_error:.2e}");

        // Sample matrix elements
        println!("      Sample matrix elements:");
        for i in 0..matrix.len().min(4) {
            let elements: Vec<String> = matrix[i]
                .iter()
                .take(4)
                .map(|z| format!("{:.2}+{:.2}i", z.re, z.im))
                .collect();
            println!(
                "        Row {}: [{}{}]",
                i,
                elements.join(", "),
                if matrix[i].len() > 4 { ", ..." } else { "" }
            );
        }
    }

    Ok(())
}

fn xyz_model_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Creating and analyzing XYZ (Heisenberg) model...");

    // Create XYZ model (anisotropic Heisenberg)
    let num_qubits = 4;
    let j_x = 1.0;
    let j_y = 1.0;
    let j_z = 0.5; // Anisotropic

    let xyz_hamiltonian = NonStoquasticHamiltonian::xyz_model(num_qubits, j_x, j_y, j_z)?;

    println!("    XYZ Model Parameters:");
    println!("      J_x = {j_x:.2}, J_y = {j_y:.2}, J_z = {j_z:.2}");
    println!("      Anisotropy ratio (J_z/J_x): {:.2}", j_z / j_x);

    // Classify the model
    if (j_x - j_y).abs() < 1e-10 && j_z.abs() < 1e-10 {
        println!("      Classification: Pure XY model");
    } else if (j_x - j_y).abs() < 1e-10 && (j_x - j_z).abs() < 1e-10 {
        println!("      Classification: Isotropic Heisenberg model");
    } else if j_z.abs() < 1e-10 {
        println!("      Classification: Anisotropic XY model");
    } else {
        println!("      Classification: Anisotropic XYZ (Heisenberg) model");
    }

    // Count interaction types
    let mut interaction_counts = std::collections::HashMap::new();
    for coupling in &xyz_hamiltonian.complex_couplings {
        *interaction_counts
            .entry(format!("{:?}", coupling.interaction_type))
            .or_insert(0) += 1;
    }

    println!("\n    Interaction breakdown:");
    for (interaction_type, count) in interaction_counts {
        println!("      {interaction_type}: {count} terms");
    }

    // Analyze different parameter regimes
    println!("\n    Testing different parameter regimes:");

    let test_cases = vec![
        ("Ferromagnetic XY", 1.0, 1.0, 0.0),
        ("Antiferromagnetic XY", -1.0, -1.0, 0.0),
        ("Ising limit", 0.0, 0.0, 1.0),
        ("Isotropic Heisenberg", 1.0, 1.0, 1.0),
        ("XXZ model", 1.0, 1.0, 2.0),
    ];

    for (name, jx, jy, jz) in test_cases {
        let test_hamiltonian = NonStoquasticHamiltonian::xyz_model(num_qubits, jx, jy, jz)?;
        let sign_severity = test_hamiltonian.sign_problem_severity();
        let is_stoquastic = test_hamiltonian.is_stoquastic();

        println!("      {name}: sign problem = {sign_severity:.2}, stoquastic = {is_stoquastic}");
    }

    Ok(())
}

fn sign_problem_analysis_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Analyzing sign problem across different models...");

    // Test various models and analyze their sign problems
    let models = vec![
        (
            "Pure Ising (Z-Z)",
            HamiltonianType::XYZModel {
                j_x: 0.0,
                j_y: 0.0,
                j_z: 1.0,
            },
        ),
        (
            "XY Ferromagnetic",
            HamiltonianType::XYModel { j_x: 1.0, j_y: 1.0 },
        ),
        (
            "XY Antiferromagnetic",
            HamiltonianType::XYModel {
                j_x: -1.0,
                j_y: -1.0,
            },
        ),
        (
            "Isotropic Heisenberg",
            HamiltonianType::XYZModel {
                j_x: 1.0,
                j_y: 1.0,
                j_z: 1.0,
            },
        ),
        (
            "XXZ Easy-axis",
            HamiltonianType::XYZModel {
                j_x: 1.0,
                j_y: 1.0,
                j_z: 2.0,
            },
        ),
        (
            "XXZ Easy-plane",
            HamiltonianType::XYZModel {
                j_x: 1.0,
                j_y: 1.0,
                j_z: 0.5,
            },
        ),
        ("Complex Ising", HamiltonianType::ComplexIsingModel),
    ];

    println!("\n    Sign Problem Analysis:");
    println!("    ┌─────────────────────┬────────────┬─────────────┬─────────────────┐");
    println!("    │ Model Type          │ Stoquastic │ Sign Prob.  │ Mitigation      │");
    println!("    ├─────────────────────┼────────────┼─────────────┼─────────────────┤");

    for (name, hamiltonian_type) in models {
        let hamiltonian = NonStoquasticHamiltonian::new(4, hamiltonian_type);
        let is_stoquastic = hamiltonian.is_stoquastic();
        let sign_severity = hamiltonian.sign_problem_severity();

        let mitigation = if sign_severity == 0.0 {
            "None needed"
        } else if sign_severity < 0.3 {
            "Reweighting"
        } else if sign_severity < 0.7 {
            "Pop. Annealing"
        } else {
            "Advanced methods"
        };

        println!(
            "    │ {:19} │ {:10} │ {:11.2} │ {:15} │",
            name,
            if is_stoquastic { "Yes" } else { "No" },
            sign_severity,
            mitigation
        );
    }
    println!("    └─────────────────────┴────────────┴─────────────┴─────────────────┘");

    // Detailed analysis of XY model with varying anisotropy
    println!("\n    XY Model Anisotropy Analysis:");
    println!("    (Analyzing J_x = 1.0, J_y = δ, δ ∈ [0, 2])");

    let delta_values = vec![0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0];

    for &delta in &delta_values {
        let xy_hamiltonian = NonStoquasticHamiltonian::xy_model(4, 1.0, delta)?;
        let sign_severity = xy_hamiltonian.sign_problem_severity();

        println!("      δ = {delta:.1}: sign problem severity = {sign_severity:.3}");
    }

    // System size scaling
    println!("\n    System Size Scaling (XY model, J_x = J_y = 1.0):");

    for &size in &[3, 4, 6, 8, 10] {
        let xy_hamiltonian = NonStoquasticHamiltonian::xy_model(size, 1.0, 1.0)?;
        let coupling_count = xy_hamiltonian.complex_couplings.len();
        let sign_severity = xy_hamiltonian.sign_problem_severity();

        println!(
            "      N = {size:2}: {coupling_count} couplings, sign severity = {sign_severity:.3}"
        );
    }

    Ok(())
}

fn quantum_monte_carlo_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Running Quantum Monte Carlo simulations...");

    // Create a moderately sized XY system
    let num_qubits = 6;
    let xy_hamiltonian = NonStoquasticHamiltonian::xy_model(num_qubits, 1.0, 0.8)?;

    println!("    System: XY chain with {num_qubits} qubits");
    println!("    Couplings: J_x = 1.0, J_y = 0.8");

    // Test different simulation strategies
    let strategies = vec![
        ("Basic QMC", SignMitigationStrategy::ReweightingMethod),
        (
            "Population Annealing",
            SignMitigationStrategy::PopulationAnnealing,
        ),
        ("Constrained Path", SignMitigationStrategy::ConstrainedPath),
    ];

    println!("\n    Comparing simulation strategies:");
    println!("    ┌──────────────────┬─────────────┬──────────────┬─────────────┬─────────────┐");
    println!("    │ Strategy         │ Ground E    │ Sign Avg     │ Runtime     │ Converged   │");
    println!("    ├──────────────────┼─────────────┼──────────────┼─────────────┼─────────────┤");

    for (name, strategy) in strategies {
        // Configure simulation
        let config = NonStoquasticQMCConfig {
            num_steps: 5000,
            thermalization_steps: 500,
            temperature: 0.1,
            tau: 0.05,
            num_time_slices: 8,
            population_size: 200,
            sign_mitigation: strategy,
            seed: Some(42),
            measurement_interval: 10,
            convergence_threshold: 1e-4,
        };

        let start = Instant::now();
        let mut simulator = NonStoquasticSimulator::new(xy_hamiltonian.clone(), config)?;
        let result = simulator.simulate()?;
        let runtime = start.elapsed();

        println!(
            "    │ {:16} │ {:11.4} │ {:12.3} │ {:11.2?} │ {:11} │",
            name,
            result.ground_state_energy.re,
            result.average_sign.norm(),
            runtime,
            if result.convergence_info.converged {
                "Yes"
            } else {
                "No"
            }
        );
    }
    println!("    └──────────────────┴─────────────┴──────────────┴─────────────┴─────────────┘");

    // Detailed analysis of one simulation
    println!("\n    Detailed analysis (Population Annealing):");

    let detailed_config = NonStoquasticQMCConfig {
        num_steps: 3000,
        thermalization_steps: 300,
        temperature: 0.2,
        population_size: 500,
        sign_mitigation: SignMitigationStrategy::PopulationAnnealing,
        seed: Some(123),
        ..Default::default()
    };

    let start = Instant::now();
    let mut detailed_simulator = NonStoquasticSimulator::new(xy_hamiltonian, detailed_config)?;
    let detailed_result = detailed_simulator.simulate()?;
    let detailed_runtime = start.elapsed();

    println!(
        "      Ground state energy: {:.6} + {:.6}i",
        detailed_result.ground_state_energy.re, detailed_result.ground_state_energy.im
    );
    println!(
        "      Energy variance: {:.6}",
        detailed_result.energy_variance
    );
    println!(
        "      Average sign magnitude: {:.4}",
        detailed_result.average_sign.norm()
    );
    println!(
        "      Sign problem severity: {:.3}",
        detailed_result.sign_problem_severity
    );
    println!("      Total runtime: {detailed_runtime:.2?}");

    // QMC Statistics
    let stats = &detailed_result.qmc_statistics;
    println!("\n      QMC Statistics:");
    println!(
        "        Acceptance rate: {:.1}%",
        stats.acceptance_rate * 100.0
    );
    println!(
        "        Autocorrelation time: {:.2}",
        stats.autocorrelation_time
    );
    println!(
        "        Effective sample size: {}",
        stats.effective_sample_size
    );

    if !stats.population_evolution.is_empty() {
        println!(
            "        Population evolution: {} → {} → {} → ...",
            stats.population_evolution.first().unwrap_or(&0),
            stats.population_evolution.get(1).unwrap_or(&0),
            stats.population_evolution.get(2).unwrap_or(&0)
        );
    }

    // Convergence analysis
    let conv = &detailed_result.convergence_info;
    println!("\n      Convergence Analysis:");
    println!("        Converged: {}", conv.converged);
    if let Some(step) = conv.convergence_step {
        println!("        Convergence step: {step}");
    }

    if conv.energy_history.len() > 5 {
        println!("        Energy history (last 5): ");
        for (i, energy) in conv.energy_history.iter().rev().take(5).enumerate() {
            println!("          -{}: {:.4}", i, energy.re);
        }
    }

    Ok(())
}

fn population_annealing_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Demonstrating Population Annealing for complex systems...");

    // Create a frustrated XY triangle
    let triangle_hamiltonian = create_frustrated_xy_triangle(1.0)?;

    println!("    System: Frustrated XY triangle");
    println!("    All couplings: J_xy = 1.0 (frustrated geometry)");
    println!("    Expected: Non-trivial ground state due to frustration");

    // Population annealing with different population sizes
    let population_sizes = vec![50, 100, 200, 500];

    println!("\n    Population size scaling:");
    println!("    ┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐");
    println!("    │ Pop. Size   │ Ground E    │ Energy Var  │ Runtime     │ Eff. Size   │");
    println!("    ├─────────────┼─────────────┼─────────────┼─────────────┼─────────────┤");

    for &pop_size in &population_sizes {
        let config = NonStoquasticQMCConfig {
            num_steps: 2000,
            population_size: pop_size,
            sign_mitigation: SignMitigationStrategy::PopulationAnnealing,
            temperature: 0.1,
            seed: Some(42),
            ..Default::default()
        };

        let start = Instant::now();
        let mut simulator = NonStoquasticSimulator::new(triangle_hamiltonian.clone(), config)?;
        let result = simulator.simulate()?;
        let runtime = start.elapsed();

        println!(
            "    │ {:11} │ {:11.4} │ {:11.6} │ {:11.2?} │ {:11} │",
            pop_size,
            result.ground_state_energy.re,
            result.energy_variance,
            runtime,
            result.qmc_statistics.effective_sample_size
        );
    }
    println!("    └─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘");

    // Temperature scaling analysis
    println!("\n    Temperature dependence (Pop. size = 200):");

    let temperatures = vec![0.01, 0.05, 0.1, 0.2, 0.5, 1.0];

    for &temp in &temperatures {
        let config = NonStoquasticQMCConfig {
            num_steps: 1500,
            population_size: 200,
            temperature: temp,
            sign_mitigation: SignMitigationStrategy::PopulationAnnealing,
            seed: Some(42),
            ..Default::default()
        };

        let mut simulator = NonStoquasticSimulator::new(triangle_hamiltonian.clone(), config)?;
        let result = simulator.simulate()?;

        println!(
            "      T = {:.2}: E = {:.4}, |<Ψ|Ψ>| = {:.3}",
            temp,
            result.ground_state_energy.re,
            result.average_sign.norm()
        );
    }

    // Compare with different geometries
    println!("\n    Geometry comparison:");

    let geometries = vec![
        ("Linear XY chain", create_xy_chain(3, 1.0, 1.0)?),
        ("Frustrated triangle", create_frustrated_xy_triangle(1.0)?),
        ("TFXY (h=0.5)", create_tfxy_model(3, 1.0, 1.0, 0.5)?),
    ];

    for (name, hamiltonian) in geometries {
        let config = NonStoquasticQMCConfig {
            num_steps: 1000,
            population_size: 150,
            temperature: 0.1,
            sign_mitigation: SignMitigationStrategy::PopulationAnnealing,
            seed: Some(42),
            ..Default::default()
        };

        let mut simulator = NonStoquasticSimulator::new(hamiltonian, config)?;
        let result = simulator.simulate()?;

        println!(
            "      {}: E = {:.4}, sign = {:.3}",
            name,
            result.ground_state_energy.re,
            result.average_sign.norm()
        );
    }

    Ok(())
}

fn stoquastic_comparison_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Comparing stoquastic approximations with exact non-stoquastic...");

    // Create XY model
    let num_qubits = 5;
    let xy_hamiltonian = NonStoquasticHamiltonian::xy_model(num_qubits, 1.0, 0.8)?;

    println!("    Original system: XY model with {num_qubits} qubits");

    // Convert to stoquastic approximation
    let ising_approx = xy_to_ising_approximation(&xy_hamiltonian)?;

    println!("    Generated stoquastic (Ising) approximation");

    // Simulate both
    println!("\n    Simulation comparison:");

    // Non-stoquastic simulation
    let ns_config = NonStoquasticQMCConfig {
        num_steps: 3000,
        population_size: 200,
        temperature: 0.1,
        sign_mitigation: SignMitigationStrategy::ReweightingMethod,
        seed: Some(42),
        ..Default::default()
    };

    let start = Instant::now();
    let mut ns_simulator = NonStoquasticSimulator::new(xy_hamiltonian, ns_config)?;
    let ns_result = ns_simulator.simulate()?;
    let ns_runtime = start.elapsed();

    // Classical annealing for stoquastic approximation
    let annealing_params = AnnealingParams {
        num_sweeps: 3000,
        num_repetitions: 10,
        initial_temperature: 2.0,
        timeout: Some(5.0),
        ..Default::default()
    };

    let start = Instant::now();
    let classical_simulator = ClassicalAnnealingSimulator::new(annealing_params)?;
    let classical_result = classical_simulator.solve(&ising_approx)?;
    let classical_runtime = start.elapsed();

    println!("    ┌─────────────────────┬─────────────────┬─────────────────┐");
    println!("    │ Method              │ Ground Energy   │ Runtime         │");
    println!("    ├─────────────────────┼─────────────────┼─────────────────┤");
    println!(
        "    │ Non-stoquastic QMC  │ {:15.4} │ {:15.2?} │",
        ns_result.ground_state_energy.re, ns_runtime
    );
    println!(
        "    │ Stoquastic approx   │ {:15.4} │ {:15.2?} │",
        classical_result.best_energy, classical_runtime
    );
    println!("    └─────────────────────┴─────────────────┴─────────────────┘");

    let energy_diff = (ns_result.ground_state_energy.re - classical_result.best_energy).abs();
    println!("\n    Analysis:");
    println!("      Energy difference: {energy_diff:.6}");

    if energy_diff < 0.01 {
        println!("      ✅ Stoquastic approximation is very accurate");
    } else if energy_diff < 0.1 {
        println!("      ⚡ Stoquastic approximation is reasonably good");
    } else {
        println!("      ⚠️  Significant difference - quantum effects important");
    }

    println!(
        "      Sign problem severity: {:.3}",
        ns_result.sign_problem_severity
    );
    println!("      Average sign: {:.3}", ns_result.average_sign.norm());

    if ns_result.average_sign.norm() < 0.1 {
        println!("      ⚠️  Severe sign problem detected");
    }

    // Compare ground state configurations
    println!("\n    Ground state analysis:");

    if let Some(ref ns_ground_state) = ns_result.ground_state {
        println!("      Non-stoquastic ground state: {ns_ground_state:?}");
    }

    println!(
        "      Stoquastic ground state:     {:?}",
        &classical_result.best_spins[..ns_result
            .ground_state
            .as_ref()
            .map_or(classical_result.best_spins.len(), |gs| gs
                .len()
                .min(classical_result.best_spins.len()))]
    );

    // Calculate overlap if possible
    if let Some(ref ns_gs) = ns_result.ground_state {
        let overlap = ns_gs
            .iter()
            .zip(classical_result.best_spins.iter())
            .map(|(&a, &b)| if a == b { 1.0 } else { 0.0 })
            .sum::<f64>()
            / ns_gs.len() as f64;

        println!("      Ground state overlap: {overlap:.2}");

        if overlap > 0.9 {
            println!("      ✅ Ground states are very similar");
        } else if overlap > 0.7 {
            println!("      ⚡ Ground states are somewhat similar");
        } else {
            println!("      ⚠️  Ground states differ significantly");
        }
    }

    Ok(())
}

fn complex_hamiltonian_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Exploring complex-weighted Hamiltonians...");

    let num_qubits = 4;
    let mut complex_hamiltonian = NonStoquasticHamiltonian::complex_ising_model(num_qubits);

    println!("    Creating complex Ising model with {num_qubits} qubits");

    // Add complex couplings
    let complex_couplings = vec![
        ((0, 1), NComplex::new(1.0, 0.5)),   // Real + imaginary
        ((1, 2), NComplex::new(0.8, -0.3)),  // Different complex weight
        ((2, 3), NComplex::new(-0.5, 0.7)),  // Negative real part
        ((0, 3), NComplex::new(0.0, 1.0)),   // Pure imaginary
        ((0, 2), NComplex::new(1.5, 0.0)),   // Pure real
        ((1, 3), NComplex::new(-0.3, -0.4)), // Complex with negative parts
    ];

    println!("\n    Complex coupling structure:");
    for (i, ((site1, site2), strength)) in complex_couplings.iter().enumerate() {
        complex_hamiltonian.add_complex_coupling(ComplexCoupling {
            sites: (*site1, *site2),
            strength: *strength,
            interaction_type: InteractionType::ZZ, // Start with stoquastic base
        })?;

        println!(
            "      Coupling {}: ({}, {}) → {:.2} + {:.2}i (|J| = {:.3})",
            i + 1,
            site1,
            site2,
            strength.re,
            strength.im,
            strength.norm()
        );
    }

    // Add non-stoquastic terms
    complex_hamiltonian.add_complex_coupling(ComplexCoupling {
        sites: (0, 1),
        strength: NComplex::new(0.3, 0.2),
        interaction_type: InteractionType::XX,
    })?;

    complex_hamiltonian.add_complex_coupling(ComplexCoupling {
        sites: (2, 3),
        strength: NComplex::new(0.2, -0.1),
        interaction_type: InteractionType::YY,
    })?;

    println!("\n    Added non-stoquastic XX and YY terms");

    // Analyze the complexity
    let sign_severity = complex_hamiltonian.sign_problem_severity();
    println!("    Sign problem severity: {sign_severity:.3}");

    // Add complex local fields
    for site in 0..num_qubits {
        let field = 0.1 * (site as f64 - 1.5);
        complex_hamiltonian.set_local_field(site, field)?;
    }

    println!("\n    Local magnetic fields:");
    for (site, &field) in complex_hamiltonian.local_fields.iter().enumerate() {
        if field.abs() > 1e-10 {
            println!("      Site {site}: h = {field:.2}");
        }
    }

    // Matrix representation analysis
    println!("\n    Matrix representation analysis:");
    let matrix = complex_hamiltonian.to_matrix()?;
    let dim = matrix.len();

    println!("      Matrix dimension: {dim}×{dim}");

    // Analyze matrix properties
    let mut max_real: f64 = 0.0;
    let mut max_imag: f64 = 0.0;
    let mut nnz_count = 0;
    let mut hermiticity_error: f64 = 0.0;

    for i in 0..dim {
        for j in 0..dim {
            let element = matrix[i][j];
            max_real = max_real.max(element.re.abs());
            max_imag = max_imag.max(element.im.abs());

            if element.norm() > 1e-12 {
                nnz_count += 1;
            }

            let hermitian_error = (element - matrix[j][i].conj()).norm();
            hermiticity_error = hermiticity_error.max(hermitian_error);
        }
    }

    println!("      Matrix statistics:");
    println!("        Max real part: {max_real:.4}");
    println!("        Max imaginary part: {max_imag:.4}");
    println!(
        "        Non-zero elements: {} ({:.1}%)",
        nnz_count,
        100.0 * f64::from(nnz_count) / (dim * dim) as f64
    );
    println!("        Hermiticity error: {hermiticity_error:.2e}");

    if hermiticity_error < 1e-10 {
        println!("        ✅ Matrix is Hermitian (as expected)");
    } else {
        println!("        ⚠️  Matrix is not perfectly Hermitian");
    }

    // Show sample matrix elements
    println!("\n      Sample matrix elements (top-left 4×4):");
    for i in 0..4.min(dim) {
        let row_elements: Vec<String> = matrix[i]
            .iter()
            .take(4)
            .map(|z| {
                if z.norm() < 1e-10 {
                    "    0    ".to_string()
                } else if z.im.abs() < 1e-10 {
                    format!("{:8.3}", z.re)
                } else {
                    format!("{:.2}+{:.2}i", z.re, z.im)
                }
            })
            .collect();
        println!("        [{}]", row_elements.join(" "));
    }

    Ok(())
}

fn frustrated_systems_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Analyzing frustrated quantum spin systems...");

    // Create different frustrated geometries
    println!("    Comparing frustrated geometries:");

    // 1. Frustrated triangle
    let triangle = create_frustrated_xy_triangle(1.0)?;

    // 2. Frustrated chain with competing interactions
    let mut frustrated_chain = NonStoquasticHamiltonian::xy_model(4, 1.0, 1.0)?;
    // Add next-nearest neighbor interactions (frustrating)
    frustrated_chain.add_complex_coupling(ComplexCoupling {
        sites: (0, 2),
        strength: NComplex::new(-0.5, 0.0), // Competing interaction
        interaction_type: InteractionType::XX,
    })?;
    frustrated_chain.add_complex_coupling(ComplexCoupling {
        sites: (1, 3),
        strength: NComplex::new(-0.5, 0.0),
        interaction_type: InteractionType::XX,
    })?;

    // 3. XY model with disorder
    let mut disordered_xy =
        NonStoquasticHamiltonian::new(4, HamiltonianType::XYModel { j_x: 1.0, j_y: 1.0 });
    let disorder_strengths = [1.2, 0.8, 1.5, 0.6]; // Random disorder
    for i in 0..3 {
        let j_random = disorder_strengths[i];
        disordered_xy.add_complex_coupling(ComplexCoupling {
            sites: (i, i + 1),
            strength: NComplex::new(j_random, 0.0),
            interaction_type: InteractionType::XX,
        })?;
        disordered_xy.add_complex_coupling(ComplexCoupling {
            sites: (i, i + 1),
            strength: NComplex::new(j_random * 0.8, 0.0),
            interaction_type: InteractionType::YY,
        })?;
    }

    let systems = vec![
        ("Frustrated Triangle", triangle),
        ("Frustrated Chain", frustrated_chain),
        ("Disordered XY", disordered_xy),
    ];

    println!("\n    System analysis:");
    println!("    ┌─────────────────────┬─────────────┬─────────────┬─────────────┐");
    println!("    │ System              │ Sign Prob.  │ Ground E    │ Convergence │");
    println!("    ├─────────────────────┼─────────────┼─────────────┼─────────────┤");

    for (name, hamiltonian) in systems {
        let sign_severity = hamiltonian.sign_problem_severity();

        // Simulate
        let config = NonStoquasticQMCConfig {
            num_steps: 2000,
            population_size: 150,
            temperature: 0.05,
            sign_mitigation: SignMitigationStrategy::PopulationAnnealing,
            seed: Some(42),
            ..Default::default()
        };

        let mut simulator = NonStoquasticSimulator::new(hamiltonian, config)?;
        let result = simulator.simulate()?;

        println!(
            "    │ {:19} │ {:11.3} │ {:11.4} │ {:11} │",
            name,
            sign_severity,
            result.ground_state_energy.re,
            if result.convergence_info.converged {
                "Yes"
            } else {
                "No"
            }
        );
    }
    println!("    └─────────────────────┴─────────────┴─────────────┴─────────────┘");

    // Detailed frustration analysis
    println!("\n    Frustration parameter sweep (XY triangle):");
    println!("    (Varying coupling strength while maintaining frustration)");

    let j_values = vec![0.5, 0.8, 1.0, 1.2, 1.5, 2.0];

    for &j in &j_values {
        let frustrated_triangle = create_frustrated_xy_triangle(j)?;

        let config = NonStoquasticQMCConfig {
            num_steps: 1500,
            population_size: 100,
            temperature: 0.1,
            sign_mitigation: SignMitigationStrategy::ReweightingMethod,
            seed: Some(42),
            ..Default::default()
        };

        let mut simulator = NonStoquasticSimulator::new(frustrated_triangle, config)?;
        let result = simulator.simulate()?;

        println!(
            "      J = {:.1}: E = {:8.4}, |<sign>| = {:.3}, severity = {:.3}",
            j,
            result.ground_state_energy.re,
            result.average_sign.norm(),
            result.sign_problem_severity
        );
    }

    // Temperature dependence in frustrated systems
    println!("\n    Temperature scaling in frustrated triangle:");

    let temperatures = vec![0.01, 0.05, 0.1, 0.2, 0.5];
    let frustrated_triangle = create_frustrated_xy_triangle(1.0)?;

    for &temp in &temperatures {
        let config = NonStoquasticQMCConfig {
            num_steps: 1000,
            population_size: 100,
            temperature: temp,
            sign_mitigation: SignMitigationStrategy::PopulationAnnealing,
            seed: Some(42),
            ..Default::default()
        };

        let mut simulator = NonStoquasticSimulator::new(frustrated_triangle.clone(), config)?;
        let result = simulator.simulate()?;

        println!(
            "      T = {:.2}: E = {:7.4}, sign = {:.3}",
            temp,
            result.ground_state_energy.re,
            result.average_sign.norm()
        );
    }

    Ok(())
}
