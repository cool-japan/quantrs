//! Machine Learning and Reinforcement Learning for Quantum Optimization
//!
//! This example demonstrates advanced quantum circuit optimization using:
//! 1. Machine Learning-based Error Mitigation
//! 2. Reinforcement Learning for Circuit Optimization
//!
//! These techniques represent the cutting edge of quantum computing, combining
//! classical ML/RL with quantum circuit analysis to achieve superior results.

use quantrs2_core::error::QuantRS2Result;
use quantrs2_core::ml_error_mitigation::{
    AdaptiveErrorMitigation, CircuitFeatures, NeuralErrorPredictor,
};
use quantrs2_core::rl_circuit_optimization::{
    CircuitState, OptimizationAction, OptimizationEpisode, QLearningOptimizer,
};

fn main() -> QuantRS2Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║   Machine Learning & Reinforcement Learning Quantum Optimization ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    // Part 1: ML-Based Error Mitigation
    demonstrate_ml_error_mitigation()?;

    println!("\n{}\n", "=".repeat(70));

    // Part 2: Reinforcement Learning Circuit Optimization
    demonstrate_rl_circuit_optimization()?;

    println!("\n{}\n", "=".repeat(70));

    // Part 3: Combined Approach
    demonstrate_combined_approach()?;

    Ok(())
}

/// Demonstrate neural network-based error prediction
fn demonstrate_ml_error_mitigation() -> QuantRS2Result<()> {
    println!("📊 Part 1: Machine Learning-Based Error Mitigation");
    println!("{}\n", "-".repeat(70));

    println!("Creating neural error predictor with 8 input features and 16 hidden neurons...");
    let mut predictor = NeuralErrorPredictor::new(8, 16, 0.01);

    println!("\n🎯 Training Phase:");
    println!("Simulating 100 training examples from quantum hardware...\n");

    // Simulate training data (in reality, this would come from actual quantum hardware)
    let training_data = vec![
        // (depth, single_gates, two_gates, connectivity, fidelity, measurements, width, entanglement) -> error_rate
        (vec![10.0, 20.0, 5.0, 1.25, 0.99, 4.0, 4.0, 1.25], 0.02),
        (vec![15.0, 30.0, 8.0, 1.27, 0.98, 6.0, 5.0, 1.6], 0.04),
        (vec![20.0, 40.0, 12.0, 1.3, 0.97, 8.0, 6.0, 2.0], 0.06),
        (vec![8.0, 16.0, 3.0, 1.2, 0.995, 3.0, 3.0, 1.0], 0.01),
        (vec![25.0, 50.0, 15.0, 1.35, 0.96, 10.0, 7.0, 2.14], 0.08),
        (vec![5.0, 10.0, 2.0, 1.2, 0.998, 2.0, 2.0, 1.0], 0.005),
        (vec![30.0, 60.0, 20.0, 1.33, 0.95, 12.0, 8.0, 2.5], 0.10),
        (vec![12.0, 24.0, 6.0, 1.25, 0.985, 5.0, 4.0, 1.5], 0.03),
    ];

    // Train the predictor
    for (epoch, (features, error_rate)) in training_data.iter().cycle().take(100).enumerate() {
        predictor.train(features, *error_rate)?;

        if (epoch + 1) % 25 == 0 {
            let accuracy = predictor.calculate_accuracy();
            println!("  Epoch {}: Accuracy = {:.2}%", epoch + 1, accuracy * 100.0);
        }
    }

    println!("\n✅ Training Complete!");
    println!(
        "Final Accuracy: {:.2}%",
        predictor.calculate_accuracy() * 100.0
    );

    println!("\n🔍 Testing Predictions:");

    // Test predictions
    let test_cases = vec![
        (
            vec![12.0, 25.0, 7.0, 1.26, 0.987, 5.0, 4.0, 1.75],
            "Medium complexity circuit",
        ),
        (
            vec![5.0, 10.0, 2.0, 1.18, 0.999, 2.0, 2.0, 0.67],
            "Simple, high-fidelity circuit",
        ),
        (
            vec![35.0, 70.0, 25.0, 1.36, 0.94, 14.0, 9.0, 2.78],
            "Complex, challenging circuit",
        ),
    ];

    for (features, description) in test_cases {
        let predicted_error = predictor.predict(&features)?;
        println!(
            "  {}: {:.2}% predicted error rate",
            description,
            predicted_error * 100.0
        );
    }

    println!("\n📈 Adaptive Error Mitigation:");
    let mut adaptive_mitigation = AdaptiveErrorMitigation::new();

    let circuit_features = CircuitFeatures {
        depth: 15,
        single_qubit_gates: 30,
        two_qubit_gates: 10,
        connectivity: 1.33,
        average_gate_fidelity: 0.985,
        measurement_count: 5,
        width: 5,
        entanglement_entropy: 2.0,
    };

    let (recommended_shots, mitigation_strength) =
        adaptive_mitigation.recommend_mitigation(&circuit_features)?;

    println!("  Circuit Analysis:");
    println!("    - Depth: {} layers", circuit_features.depth);
    println!(
        "    - Gates: {} (single: {}, two-qubit: {})",
        circuit_features.single_qubit_gates + circuit_features.two_qubit_gates,
        circuit_features.single_qubit_gates,
        circuit_features.two_qubit_gates
    );
    println!(
        "    - Estimated Fidelity: {:.4}",
        circuit_features.average_gate_fidelity
    );
    println!("\n  Recommendations:");
    println!("    - Shots: {recommended_shots} (adaptive allocation)");
    println!("    - Mitigation Strength: {mitigation_strength:.2}x");

    // Simulate updating with results
    adaptive_mitigation.update_from_results(&circuit_features, 0.03)?;

    let metrics = adaptive_mitigation.get_metrics();
    println!("\n  System Metrics:");
    println!("    - Circuits Analyzed: {}", metrics.total_circuits);
    println!(
        "    - Prediction Accuracy: {:.2}%",
        metrics.prediction_accuracy * 100.0
    );

    Ok(())
}

/// Demonstrate Q-learning for circuit optimization
fn demonstrate_rl_circuit_optimization() -> QuantRS2Result<()> {
    println!("🤖 Part 2: Reinforcement Learning Circuit Optimization");
    println!("{}\n", "-".repeat(70));

    println!("Creating Q-Learning optimizer (α=0.1, γ=0.95, ε=0.3)...");
    let mut optimizer = QLearningOptimizer::new(0.1, 0.95, 0.3);

    println!("\n🎓 Training Phase:");
    println!("Running 50 optimization episodes...\n");

    // Simulate optimization episodes
    for episode_num in 1..=50 {
        // Initial circuit state
        let initial_state = CircuitState {
            depth: 20 + (episode_num % 10) as usize,
            gate_count: 80 + (episode_num % 20) as usize,
            two_qubit_count: 25 + (episode_num % 8) as usize,
            fidelity: ((episode_num % 5) as f64).mul_add(-0.01, 0.94),
            qubit_count: 6,
            connectivity_density: 0.5,
            entanglement_measure: 0.7,
        };

        let mut current_state = initial_state.clone();
        let mut steps = 0;
        let mut total_reward = 0.0;

        // Simulate optimization steps
        for _ in 0..10 {
            let available_actions = vec![
                OptimizationAction::MergeSingleQubitGates { gate_index: 0 },
                OptimizationAction::CancelInversePairs { gate_index: 1 },
                OptimizationAction::OptimizeTwoQubitGate { gate_index: 2 },
                OptimizationAction::CommuteGates {
                    gate1_index: 0,
                    gate2_index: 1,
                },
            ];

            let action = optimizer.choose_action(&current_state, &available_actions);

            // Simulate state transition (in reality, would apply actual transformations)
            let new_state = CircuitState {
                depth: current_state.depth.saturating_sub(1),
                gate_count: current_state.gate_count.saturating_sub(2),
                two_qubit_count: if matches!(
                    action,
                    OptimizationAction::OptimizeTwoQubitGate { .. }
                ) {
                    current_state.two_qubit_count.saturating_sub(1)
                } else {
                    current_state.two_qubit_count
                },
                fidelity: (current_state.fidelity + 0.001).min(1.0),
                qubit_count: current_state.qubit_count,
                connectivity_density: current_state.connectivity_density,
                entanglement_measure: current_state.entanglement_measure,
            };

            let reward = optimizer.calculate_reward(&current_state, &new_state);
            total_reward += reward;

            optimizer.update_q_value(
                &current_state,
                action,
                reward,
                &new_state,
                &available_actions,
            );

            current_state = new_state;
            steps += 1;

            if current_state.depth <= initial_state.depth / 2 {
                break; // Good enough optimization
            }
        }

        let episode = OptimizationEpisode {
            initial_depth: initial_state.depth,
            final_depth: current_state.depth,
            initial_gate_count: initial_state.gate_count,
            final_gate_count: current_state.gate_count,
            reward: total_reward,
            steps_taken: steps,
        };

        optimizer.finish_episode(episode.clone());

        if episode_num % 10 == 0 {
            println!(
                "  Episode {}: Depth {} → {} (reduced by {}), Reward: {:.2}",
                episode_num,
                episode.initial_depth,
                episode.final_depth,
                episode.initial_depth - episode.final_depth,
                episode.reward
            );
        }
    }

    println!("\n✅ Training Complete!");

    let stats = optimizer.get_statistics();
    println!("\n📊 Optimization Statistics:");
    println!("  - Total Episodes: {}", stats.total_episodes);
    println!(
        "  - Avg Depth Improvement: {:.2} layers",
        stats.average_depth_improvement
    );
    println!(
        "  - Avg Gate Reduction: {:.2} gates",
        stats.average_gate_reduction
    );
    println!("  - Avg Reward per Episode: {:.2}", stats.average_reward);
    println!(
        "  - Current Exploration Rate (ε): {:.3}",
        stats.current_epsilon
    );
    println!("  - Q-Table Size: {} entries", stats.q_table_size);

    println!("\n🎯 Applying Learned Policy:");

    // Test the trained optimizer
    let test_state = CircuitState {
        depth: 25,
        gate_count: 100,
        two_qubit_count: 30,
        fidelity: 0.93,
        qubit_count: 6,
        connectivity_density: 0.5,
        entanglement_measure: 0.7,
    };

    let test_actions = vec![
        OptimizationAction::MergeSingleQubitGates { gate_index: 0 },
        OptimizationAction::CancelInversePairs { gate_index: 1 },
        OptimizationAction::OptimizeTwoQubitGate { gate_index: 2 },
    ];

    let recommended_action = optimizer.choose_action(&test_state, &test_actions);
    println!(
        "  Test Circuit: Depth={}, Gates={}, Two-Qubit={}",
        test_state.depth, test_state.gate_count, test_state.two_qubit_count
    );
    println!("  Recommended Action: {recommended_action:?}");

    Ok(())
}

/// Demonstrate combined ML and RL approach
fn demonstrate_combined_approach() -> QuantRS2Result<()> {
    println!("🔬 Part 3: Combined ML + RL Approach");
    println!("{}\n", "-".repeat(70));

    println!("Integrating ML error mitigation with RL circuit optimization...\n");

    // Create both systems
    let mut error_mitigator = AdaptiveErrorMitigation::new();
    let mut circuit_optimizer = QLearningOptimizer::new(0.15, 0.95, 0.2);

    println!("🎯 Workflow:");
    println!("  1. ML predicts error rates for circuit");
    println!("  2. RL optimizes circuit to reduce predicted errors");
    println!("  3. ML recommends mitigation strategies for optimized circuit");
    println!("  4. System learns from actual execution results\n");

    // Example circuit
    let initial_features = CircuitFeatures {
        depth: 22,
        single_qubit_gates: 44,
        two_qubit_gates: 16,
        connectivity: 1.36,
        average_gate_fidelity: 0.975,
        measurement_count: 6,
        width: 6,
        entanglement_entropy: 2.67,
    };

    println!("📋 Initial Circuit:");
    println!("  - Depth: {} layers", initial_features.depth);
    println!(
        "  - Total Gates: {}",
        initial_features.single_qubit_gates + initial_features.two_qubit_gates
    );
    println!(
        "  - Two-Qubit Gates: {} (expensive)",
        initial_features.two_qubit_gates
    );

    // Step 1: ML predicts error
    let (initial_shots, initial_strength) =
        error_mitigator.recommend_mitigation(&initial_features)?;
    println!("\n🔮 ML Prediction (Before Optimization):");
    println!("  - Recommended Shots: {initial_shots}");
    println!("  - Mitigation Strength: {initial_strength:.2}x");

    // Step 2: RL optimizes circuit
    println!("\n🤖 RL Optimization:");
    let initial_state = CircuitState::from_circuit(&[], initial_features.width);
    let mut current_state = initial_state;

    for step in 1..=5 {
        let actions = vec![
            OptimizationAction::MergeSingleQubitGates { gate_index: 0 },
            OptimizationAction::OptimizeTwoQubitGate { gate_index: 1 },
        ];

        let action = circuit_optimizer.choose_action(&current_state, &actions);

        let new_state = CircuitState {
            depth: current_state.depth.saturating_sub(1),
            gate_count: current_state.gate_count.saturating_sub(2),
            two_qubit_count: current_state.two_qubit_count.saturating_sub(1),
            fidelity: (current_state.fidelity * 1.005).min(1.0),
            ..current_state
        };

        let reward = circuit_optimizer.calculate_reward(&current_state, &new_state);
        circuit_optimizer.update_q_value(&current_state, action, reward, &new_state, &actions);

        current_state = new_state;

        if step == 1 || step == 5 {
            println!(
                "  Step {}: Depth={}, Gates={}, Fidelity={:.4}",
                step, current_state.depth, current_state.gate_count, current_state.fidelity
            );
        }
    }

    // Step 3: ML re-evaluates optimized circuit
    let optimized_features = CircuitFeatures {
        depth: current_state.depth,
        single_qubit_gates: ((current_state.gate_count as f64 * 0.7) as usize),
        two_qubit_gates: current_state.two_qubit_count,
        connectivity: initial_features.connectivity,
        average_gate_fidelity: current_state.fidelity,
        measurement_count: initial_features.measurement_count,
        width: initial_features.width,
        entanglement_entropy: initial_features.entanglement_entropy * 0.9,
    };

    let (final_shots, final_strength) =
        error_mitigator.recommend_mitigation(&optimized_features)?;

    println!("\n🔮 ML Prediction (After Optimization):");
    println!("  - Recommended Shots: {final_shots} (vs {initial_shots} before)");
    println!("  - Mitigation Strength: {final_strength:.2}x (vs {initial_strength:.2}x before)");

    // Calculate improvements
    let shot_change = if final_shots <= initial_shots {
        ((initial_shots - final_shots) as f64 / initial_shots as f64) * 100.0
    } else {
        -((final_shots - initial_shots) as f64 / initial_shots as f64) * 100.0
    };
    let depth_reduction = if optimized_features.depth <= initial_features.depth {
        ((initial_features.depth - optimized_features.depth) as f64 / initial_features.depth as f64)
            * 100.0
    } else {
        -((optimized_features.depth - initial_features.depth) as f64
            / initial_features.depth as f64)
            * 100.0
    };

    println!("\n📈 Overall Improvements:");
    println!(
        "  ✓ Circuit Depth: {}{:.1}%",
        if depth_reduction >= 0.0 {
            "Reduced by "
        } else {
            "Increased by "
        },
        depth_reduction.abs()
    );
    println!(
        "  ✓ Required Shots: {}{:.1}%",
        if shot_change >= 0.0 {
            "Reduced by "
        } else {
            "Increased by "
        },
        shot_change.abs()
    );
    println!(
        "  ✓ Estimated Fidelity: Improved by {:.2}%",
        (current_state.fidelity - 0.93) * 100.0
    );

    println!("\n💡 Benefits of Combined Approach:");
    println!("  • ML provides fast error predictions without execution");
    println!("  • RL learns optimal transformations from experience");
    println!("  • Together they minimize both errors AND resource usage");
    println!("  • Adaptive to specific hardware characteristics");

    Ok(())
}
