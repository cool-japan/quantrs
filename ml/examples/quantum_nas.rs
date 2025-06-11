//! Quantum Neural Architecture Search Example
//!
//! This example demonstrates various quantum neural architecture search algorithms
//! including evolutionary search, reinforcement learning, random search,
//! Bayesian optimization, and DARTS.

use ndarray::{Array1, Array2};
use quantrs2_ml::prelude::*;
use quantrs2_ml::qnn::QNNLayerType;

fn main() -> Result<()> {
    println!("=== Quantum Neural Architecture Search Demo ===\n");

    // Step 1: Evolutionary algorithm search
    println!("1. Evolutionary Algorithm Search...");
    evolutionary_search_demo()?;

    // Step 2: Random search baseline
    println!("\n2. Random Search Baseline...");
    random_search_demo()?;

    // Step 3: Reinforcement learning search
    println!("\n3. Reinforcement Learning Search...");
    rl_search_demo()?;

    // Step 4: Bayesian optimization
    println!("\n4. Bayesian Optimization Search...");
    bayesian_search_demo()?;

    // Step 5: DARTS (Differentiable Architecture Search)
    println!("\n5. DARTS (Differentiable Architecture Search)...");
    darts_demo()?;

    // Step 6: Multi-objective optimization
    println!("\n6. Multi-Objective Optimization...");
    multi_objective_demo()?;

    // Step 7: Architecture analysis
    println!("\n7. Architecture Analysis...");
    architecture_analysis_demo()?;

    println!("\n=== Quantum NAS Demo Complete ===");

    Ok(())
}

/// Evolutionary algorithm search demonstration
fn evolutionary_search_demo() -> Result<()> {
    // Create search space
    let search_space = create_default_search_space();

    // Configure evolutionary strategy
    let strategy = SearchStrategy::Evolutionary {
        population_size: 20,
        mutation_rate: 0.2,
        crossover_rate: 0.7,
        elitism_ratio: 0.1,
    };

    let mut nas = QuantumNAS::new(strategy, search_space);

    println!("   Created evolutionary NAS:");
    println!("   - Population size: 20");
    println!("   - Mutation rate: 0.2");
    println!("   - Crossover rate: 0.7");
    println!("   - Elitism ratio: 0.1");

    // Set evaluation data (synthetic for demo)
    let eval_data = Array2::from_shape_fn((100, 4), |(i, j)| (i as f64 + j as f64) / 50.0);
    let eval_labels = Array1::from_shape_fn(100, |i| i % 2);
    nas.set_evaluation_data(eval_data, eval_labels);

    // Run search
    println!("\n   Running evolutionary search for 10 generations...");
    let best_architectures = nas.search(10)?;

    println!("   Search complete!");
    println!(
        "   - Best architectures found: {}",
        best_architectures.len()
    );

    if let Some(best) = best_architectures.first() {
        println!("   - Best architecture: {}", best);
        println!("   - Circuit depth: {}", best.metrics.circuit_depth);
        println!("   - Parameter count: {}", best.metrics.parameter_count);

        if let Some(expressivity) = best.properties.expressivity {
            println!("   - Expressivity: {:.3}", expressivity);
        }
    }

    // Show search summary
    let summary = nas.get_search_summary();
    println!(
        "   - Total architectures evaluated: {}",
        summary.total_architectures_evaluated
    );
    println!("   - Pareto front size: {}", summary.pareto_front_size);

    Ok(())
}

/// Random search baseline demonstration
fn random_search_demo() -> Result<()> {
    let search_space = create_default_search_space();
    let strategy = SearchStrategy::Random { num_samples: 50 };

    let mut nas = QuantumNAS::new(strategy, search_space);

    println!("   Created random search NAS:");
    println!("   - Number of samples: 50");

    // Generate synthetic evaluation data
    let eval_data = Array2::from_shape_fn((80, 4), |(i, j)| {
        0.5 * (i as f64).sin() + 0.3 * (j as f64).cos()
    });
    let eval_labels = Array1::from_shape_fn(80, |i| if i % 3 == 0 { 0 } else { 1 });
    nas.set_evaluation_data(eval_data, eval_labels);

    println!("\n   Running random search...");
    let best_architectures = nas.search(50)?;

    println!("   Random search complete!");
    if let Some(best) = best_architectures.first() {
        println!("   - Best random architecture: {}", best);
        if let Some(accuracy) = best.metrics.accuracy {
            println!("   - Accuracy: {:.3}", accuracy);
        }
    }

    Ok(())
}

/// Reinforcement learning search demonstration
fn rl_search_demo() -> Result<()> {
    let search_space = create_custom_search_space();

    let strategy = SearchStrategy::ReinforcementLearning {
        agent_type: RLAgentType::PolicyGradient,
        exploration_rate: 0.3,
        learning_rate: 0.01,
    };

    let mut nas = QuantumNAS::new(strategy, search_space);

    println!("   Created RL-based NAS:");
    println!("   - Agent type: Policy Gradient");
    println!("   - Exploration rate: 0.3");
    println!("   - Learning rate: 0.01");

    println!("\n   Running RL search for 100 episodes...");
    let best_architectures = nas.search(100)?;

    println!("   RL search complete!");
    println!("   - Architectures found: {}", best_architectures.len());

    if let Some(best) = best_architectures.first() {
        println!("   - Best RL architecture: {}", best);
        if let Some(entanglement) = best.properties.entanglement_capability {
            println!("   - Entanglement capability: {:.3}", entanglement);
        }
    }

    Ok(())
}

/// Bayesian optimization search demonstration
fn bayesian_search_demo() -> Result<()> {
    let search_space = create_default_search_space();

    let strategy = SearchStrategy::BayesianOptimization {
        acquisition_function: AcquisitionFunction::ExpectedImprovement,
        num_initial_points: 10,
    };

    let mut nas = QuantumNAS::new(strategy, search_space);

    println!("   Created Bayesian optimization NAS:");
    println!("   - Acquisition function: Expected Improvement");
    println!("   - Initial random points: 10");

    // Set up evaluation data
    let eval_data = generate_quantum_data(60, 4);
    let eval_labels = Array1::from_shape_fn(60, |i| i % 3);
    nas.set_evaluation_data(eval_data, eval_labels);

    println!("\n   Running Bayesian optimization for 30 iterations...");
    let best_architectures = nas.search(30)?;

    println!("   Bayesian optimization complete!");
    if let Some(best) = best_architectures.first() {
        println!("   - Best Bayesian architecture: {}", best);
        if let Some(hardware_eff) = best.metrics.hardware_efficiency {
            println!("   - Hardware efficiency: {:.3}", hardware_eff);
        }
    }

    Ok(())
}

/// DARTS demonstration
fn darts_demo() -> Result<()> {
    let search_space = create_darts_search_space();

    let strategy = SearchStrategy::DARTS {
        learning_rate: 0.01,
        weight_decay: 1e-4,
    };

    let mut nas = QuantumNAS::new(strategy, search_space);

    println!("   Created DARTS NAS:");
    println!("   - Learning rate: 0.01");
    println!("   - Weight decay: 1e-4");
    println!("   - Differentiable architecture search");

    println!("\n   Running DARTS for 200 epochs...");
    let best_architectures = nas.search(200)?;

    println!("   DARTS search complete!");
    if let Some(best) = best_architectures.first() {
        println!("   - DARTS architecture: {}", best);
        println!("   - Learned through gradient-based optimization");

        if let Some(gradient_var) = best.properties.gradient_variance {
            println!("   - Gradient variance: {:.3}", gradient_var);
        }
    }

    Ok(())
}

/// Multi-objective optimization demonstration
fn multi_objective_demo() -> Result<()> {
    let search_space = create_default_search_space();

    let strategy = SearchStrategy::Evolutionary {
        population_size: 30,
        mutation_rate: 0.15,
        crossover_rate: 0.8,
        elitism_ratio: 0.2,
    };

    let mut nas = QuantumNAS::new(strategy, search_space);

    println!("   Multi-objective optimization:");
    println!("   - Optimizing accuracy vs. complexity");
    println!("   - Finding Pareto-optimal architectures");

    // Run search
    nas.search(15)?;

    // Analyze Pareto front
    let pareto_front = nas.get_pareto_front();
    println!("   Pareto front analysis:");
    println!("   - Pareto-optimal architectures: {}", pareto_front.len());

    for (i, arch) in pareto_front.iter().take(3).enumerate() {
        println!(
            "   Architecture {}: {} params, {:.3} accuracy",
            i + 1,
            arch.metrics.parameter_count,
            arch.metrics.accuracy.unwrap_or(0.0)
        );
    }

    Ok(())
}

/// Architecture analysis demonstration
fn architecture_analysis_demo() -> Result<()> {
    println!("   Analyzing quantum circuit architectures...");

    // Create sample architectures with different properties
    let architectures = create_sample_architectures();

    println!("\n   Architecture comparison:");
    for (i, arch) in architectures.iter().enumerate() {
        println!("   Architecture {}:", i + 1);
        println!("     - Layers: {}", arch.layers.len());
        println!("     - Qubits: {}", arch.num_qubits);
        println!("     - Circuit depth: {}", arch.metrics.circuit_depth);

        if let Some(expressivity) = arch.properties.expressivity {
            println!("     - Expressivity: {:.3}", expressivity);
        }

        if let Some(entanglement) = arch.properties.entanglement_capability {
            println!("     - Entanglement: {:.3}", entanglement);
        }

        if let Some(barren_plateau) = arch.properties.barren_plateau_score {
            println!("     - Barren plateau risk: {:.3}", barren_plateau);
        }

        println!();
    }

    // Performance trade-offs analysis
    println!("   Performance trade-offs:");
    println!("   - Deeper circuits: higher expressivity, more barren plateaus");
    println!("   - More entanglement: better feature mixing, higher noise sensitivity");
    println!("   - More parameters: greater capacity, overfitting risk");

    Ok(())
}

/// Generate quantum-inspired synthetic data
fn generate_quantum_data(samples: usize, features: usize) -> Array2<f64> {
    Array2::from_shape_fn((samples, features), |(i, j)| {
        let phase = (i as f64 * 0.1 + j as f64 * 0.2).sin();
        let amplitude = (i as f64 / samples as f64).exp() * 0.5;
        amplitude * phase + 0.1 * fastrand::f64()
    })
}

/// Create custom search space for RL demo
fn create_custom_search_space() -> SearchSpace {
    SearchSpace {
        layer_types: vec![
            QNNLayerType::VariationalLayer { num_params: 4 },
            QNNLayerType::VariationalLayer { num_params: 8 },
            QNNLayerType::EntanglementLayer {
                connectivity: "circular".to_string(),
            },
            QNNLayerType::EntanglementLayer {
                connectivity: "linear".to_string(),
            },
        ],
        depth_range: (1, 5),
        qubit_constraints: QubitConstraints {
            min_qubits: 3,
            max_qubits: 6,
            topology: Some(QuantumTopology::Ring),
        },
        param_ranges: vec![("variational_params".to_string(), (3, 12))]
            .into_iter()
            .collect(),
        connectivity_patterns: vec!["linear".to_string(), "circular".to_string()],
        measurement_bases: vec!["computational".to_string(), "Pauli-Z".to_string()],
    }
}

/// Create search space optimized for DARTS
fn create_darts_search_space() -> SearchSpace {
    SearchSpace {
        layer_types: vec![
            QNNLayerType::VariationalLayer { num_params: 6 },
            QNNLayerType::VariationalLayer { num_params: 9 },
            QNNLayerType::EntanglementLayer {
                connectivity: "full".to_string(),
            },
        ],
        depth_range: (3, 6),
        qubit_constraints: QubitConstraints {
            min_qubits: 4,
            max_qubits: 4, // Fixed for DARTS
            topology: Some(QuantumTopology::Complete),
        },
        param_ranges: vec![("variational_params".to_string(), (6, 9))]
            .into_iter()
            .collect(),
        connectivity_patterns: vec!["full".to_string()],
        measurement_bases: vec!["computational".to_string()],
    }
}

/// Create sample architectures for analysis
fn create_sample_architectures() -> Vec<ArchitectureCandidate> {
    vec![
        // Simple architecture
        ArchitectureCandidate {
            id: "simple".to_string(),
            layers: vec![
                QNNLayerType::EncodingLayer { num_features: 4 },
                QNNLayerType::VariationalLayer { num_params: 6 },
                QNNLayerType::MeasurementLayer {
                    measurement_basis: "computational".to_string(),
                },
            ],
            num_qubits: 3,
            metrics: ArchitectureMetrics {
                accuracy: Some(0.65),
                loss: Some(0.4),
                circuit_depth: 3,
                parameter_count: 6,
                training_time: Some(10.0),
                memory_usage: Some(512),
                hardware_efficiency: Some(0.8),
            },
            properties: ArchitectureProperties {
                expressivity: Some(0.3),
                entanglement_capability: Some(0.2),
                gradient_variance: Some(0.1),
                barren_plateau_score: Some(0.2),
                noise_resilience: Some(0.7),
            },
        },
        // Complex architecture
        ArchitectureCandidate {
            id: "complex".to_string(),
            layers: vec![
                QNNLayerType::EncodingLayer { num_features: 6 },
                QNNLayerType::VariationalLayer { num_params: 12 },
                QNNLayerType::EntanglementLayer {
                    connectivity: "full".to_string(),
                },
                QNNLayerType::VariationalLayer { num_params: 12 },
                QNNLayerType::EntanglementLayer {
                    connectivity: "circular".to_string(),
                },
                QNNLayerType::MeasurementLayer {
                    measurement_basis: "Pauli-Z".to_string(),
                },
            ],
            num_qubits: 6,
            metrics: ArchitectureMetrics {
                accuracy: Some(0.85),
                loss: Some(0.2),
                circuit_depth: 8,
                parameter_count: 24,
                training_time: Some(45.0),
                memory_usage: Some(2048),
                hardware_efficiency: Some(0.4),
            },
            properties: ArchitectureProperties {
                expressivity: Some(0.8),
                entanglement_capability: Some(0.9),
                gradient_variance: Some(0.3),
                barren_plateau_score: Some(0.7),
                noise_resilience: Some(0.3),
            },
        },
        // Balanced architecture
        ArchitectureCandidate {
            id: "balanced".to_string(),
            layers: vec![
                QNNLayerType::EncodingLayer { num_features: 4 },
                QNNLayerType::VariationalLayer { num_params: 8 },
                QNNLayerType::EntanglementLayer {
                    connectivity: "circular".to_string(),
                },
                QNNLayerType::VariationalLayer { num_params: 8 },
                QNNLayerType::MeasurementLayer {
                    measurement_basis: "computational".to_string(),
                },
            ],
            num_qubits: 4,
            metrics: ArchitectureMetrics {
                accuracy: Some(0.78),
                loss: Some(0.28),
                circuit_depth: 5,
                parameter_count: 16,
                training_time: Some(25.0),
                memory_usage: Some(1024),
                hardware_efficiency: Some(0.65),
            },
            properties: ArchitectureProperties {
                expressivity: Some(0.6),
                entanglement_capability: Some(0.5),
                gradient_variance: Some(0.15),
                barren_plateau_score: Some(0.4),
                noise_resilience: Some(0.6),
            },
        },
    ]
}
