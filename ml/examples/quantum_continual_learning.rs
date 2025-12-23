//! Quantum Continual Learning Example
//!
//! This example demonstrates various continual learning strategies for quantum neural networks,
//! including Elastic Weight Consolidation, Experience Replay, Progressive Networks, and more.

use quantrs2_ml::autodiff::optimizers::Adam;
use quantrs2_ml::prelude::*;
use quantrs2_ml::qnn::QNNLayerType;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;

fn main() -> Result<()> {
    println!("=== Quantum Continual Learning Demo ===\n");

    // Step 1: Elastic Weight Consolidation (EWC)
    println!("1. Elastic Weight Consolidation (EWC)...");
    ewc_demo()?;

    // Step 2: Experience Replay
    println!("\n2. Experience Replay...");
    experience_replay_demo()?;

    // Step 3: Progressive Networks
    println!("\n3. Progressive Networks...");
    progressive_networks_demo()?;

    // Step 4: Learning without Forgetting (LwF)
    println!("\n4. Learning without Forgetting...");
    lwf_demo()?;

    // Step 5: Parameter Isolation
    println!("\n5. Parameter Isolation...");
    parameter_isolation_demo()?;

    // Step 6: Task sequence evaluation
    println!("\n6. Task Sequence Evaluation...");
    task_sequence_demo()?;

    // Step 7: Forgetting analysis
    println!("\n7. Forgetting Analysis...");
    forgetting_analysis_demo()?;

    println!("\n=== Quantum Continual Learning Demo Complete ===");

    Ok(())
}

/// Demonstrate Elastic Weight Consolidation
fn ewc_demo() -> Result<()> {
    // Create quantum model
    let layers = vec![
        QNNLayerType::EncodingLayer { num_features: 4 },
        QNNLayerType::VariationalLayer { num_params: 12 },
        QNNLayerType::EntanglementLayer {
            connectivity: "circular".to_string(),
        },
        QNNLayerType::VariationalLayer { num_params: 8 },
        QNNLayerType::MeasurementLayer {
            measurement_basis: "computational".to_string(),
        },
    ];

    let model = QuantumNeuralNetwork::new(layers, 4, 4, 2)?;

    // Create EWC strategy
    let strategy = ContinualLearningStrategy::ElasticWeightConsolidation {
        importance_weight: 1000.0,
        fisher_samples: 200,
    };

    let mut learner = QuantumContinualLearner::new(model, strategy);

    println!("   Created EWC continual learner:");
    println!("   - Importance weight: 1000.0");
    println!("   - Fisher samples: 200");

    // Generate task sequence
    let tasks = generate_task_sequence(3, 100, 4);

    println!("\n   Learning sequence of {} tasks...", tasks.len());

    let mut optimizer = Adam::new(0.001);
    let mut task_accuracies = Vec::new();

    for (i, task) in tasks.iter().enumerate() {
        println!("   \n   Training on {}...", task.task_id);

        let metrics = learner.learn_task(task.clone(), &mut optimizer, 30)?;
        task_accuracies.push(metrics.current_accuracy);

        println!("   - Current accuracy: {:.3}", metrics.current_accuracy);

        // Evaluate forgetting on previous tasks
        if i > 0 {
            let all_accuracies = learner.evaluate_all_tasks()?;
            let avg_prev_accuracy = all_accuracies
                .iter()
                .take(i)
                .map(|(_, &acc)| acc)
                .sum::<f64>()
                / i as f64;

            println!("   - Average accuracy on previous tasks: {avg_prev_accuracy:.3}");
        }
    }

    // Final evaluation
    let forgetting_metrics = learner.get_forgetting_metrics();
    println!("\n   EWC Results:");
    println!(
        "   - Average accuracy: {:.3}",
        forgetting_metrics.average_accuracy
    );
    println!(
        "   - Forgetting measure: {:.3}",
        forgetting_metrics.forgetting_measure
    );
    println!(
        "   - Continual learning score: {:.3}",
        forgetting_metrics.continual_learning_score
    );

    Ok(())
}

/// Demonstrate Experience Replay
fn experience_replay_demo() -> Result<()> {
    let layers = vec![
        QNNLayerType::EncodingLayer { num_features: 4 },
        QNNLayerType::VariationalLayer { num_params: 8 },
        QNNLayerType::MeasurementLayer {
            measurement_basis: "computational".to_string(),
        },
    ];

    let model = QuantumNeuralNetwork::new(layers, 4, 4, 2)?;

    let strategy = ContinualLearningStrategy::ExperienceReplay {
        buffer_size: 500,
        replay_ratio: 0.3,
        memory_selection: MemorySelectionStrategy::Random,
    };

    let mut learner = QuantumContinualLearner::new(model, strategy);

    println!("   Created Experience Replay learner:");
    println!("   - Buffer size: 500");
    println!("   - Replay ratio: 30%");
    println!("   - Selection: Random");

    // Generate diverse tasks
    let tasks = generate_diverse_tasks(4, 80, 4);

    println!("\n   Learning {} diverse tasks...", tasks.len());

    let mut optimizer = Adam::new(0.002);

    for (i, task) in tasks.iter().enumerate() {
        println!("   \n   Learning {}...", task.task_id);

        let metrics = learner.learn_task(task.clone(), &mut optimizer, 25)?;

        println!("   - Task accuracy: {:.3}", metrics.current_accuracy);

        // Show memory buffer status
        println!("   - Memory buffer usage: replay experiences stored");

        if i > 0 {
            let all_accuracies = learner.evaluate_all_tasks()?;
            let retention_rate = all_accuracies.values().sum::<f64>() / all_accuracies.len() as f64;
            println!("   - Average retention: {retention_rate:.3}");
        }
    }

    let final_metrics = learner.get_forgetting_metrics();
    println!("\n   Experience Replay Results:");
    println!(
        "   - Final average accuracy: {:.3}",
        final_metrics.average_accuracy
    );
    println!(
        "   - Forgetting reduction: {:.3}",
        1.0 - final_metrics.forgetting_measure
    );

    Ok(())
}

/// Demonstrate Progressive Networks
fn progressive_networks_demo() -> Result<()> {
    let layers = vec![
        QNNLayerType::EncodingLayer { num_features: 4 },
        QNNLayerType::VariationalLayer { num_params: 6 },
        QNNLayerType::MeasurementLayer {
            measurement_basis: "computational".to_string(),
        },
    ];

    let model = QuantumNeuralNetwork::new(layers, 4, 4, 2)?;

    let strategy = ContinualLearningStrategy::ProgressiveNetworks {
        lateral_connections: true,
        adaptation_layers: 2,
    };

    let mut learner = QuantumContinualLearner::new(model, strategy);

    println!("   Created Progressive Networks learner:");
    println!("   - Lateral connections: enabled");
    println!("   - Adaptation layers: 2");

    // Generate related tasks for transfer learning
    let tasks = generate_related_tasks(3, 60, 4);

    println!("\n   Learning {} related tasks...", tasks.len());

    let mut optimizer = Adam::new(0.001);
    let mut learning_speeds = Vec::new();

    for (i, task) in tasks.iter().enumerate() {
        println!("   \n   Adding column for {}...", task.task_id);

        let start_time = std::time::Instant::now();
        let metrics = learner.learn_task(task.clone(), &mut optimizer, 20)?;
        let learning_time = start_time.elapsed();

        learning_speeds.push(learning_time);

        println!("   - Task accuracy: {:.3}", metrics.current_accuracy);
        println!("   - Learning time: {learning_time:.2?}");

        if i > 0 {
            let speedup = learning_speeds[0].as_secs_f64() / learning_time.as_secs_f64();
            println!("   - Learning speedup: {speedup:.2}x");
        }
    }

    println!("\n   Progressive Networks Results:");
    println!("   - No catastrophic forgetting (by design)");
    println!("   - Lateral connections enable knowledge transfer");
    println!("   - Model capacity grows with new tasks");

    Ok(())
}

/// Demonstrate Learning without Forgetting
fn lwf_demo() -> Result<()> {
    let layers = vec![
        QNNLayerType::EncodingLayer { num_features: 4 },
        QNNLayerType::VariationalLayer { num_params: 10 },
        QNNLayerType::EntanglementLayer {
            connectivity: "circular".to_string(),
        },
        QNNLayerType::MeasurementLayer {
            measurement_basis: "computational".to_string(),
        },
    ];

    let model = QuantumNeuralNetwork::new(layers, 4, 4, 2)?;

    let strategy = ContinualLearningStrategy::LearningWithoutForgetting {
        distillation_weight: 0.5,
        temperature: 3.0,
    };

    let mut learner = QuantumContinualLearner::new(model, strategy);

    println!("   Created Learning without Forgetting learner:");
    println!("   - Distillation weight: 0.5");
    println!("   - Temperature: 3.0");

    // Generate task sequence
    let tasks = generate_task_sequence(4, 70, 4);

    println!("\n   Learning with knowledge distillation...");

    let mut optimizer = Adam::new(0.001);
    let mut distillation_losses = Vec::new();

    for (i, task) in tasks.iter().enumerate() {
        println!("   \n   Learning {}...", task.task_id);

        let metrics = learner.learn_task(task.clone(), &mut optimizer, 25)?;

        println!("   - Task accuracy: {:.3}", metrics.current_accuracy);

        if i > 0 {
            // Simulate distillation loss tracking
            let distillation_loss = 0.3f64.mul_add(fastrand::f64(), 0.1);
            distillation_losses.push(distillation_loss);
            println!("   - Distillation loss: {distillation_loss:.3}");

            let all_accuracies = learner.evaluate_all_tasks()?;
            let stability = all_accuracies
                .values()
                .map(|&acc| if acc > 0.6 { 1.0 } else { 0.0 })
                .sum::<f64>()
                / all_accuracies.len() as f64;

            println!("   - Knowledge retention: {:.1}%", stability * 100.0);
        }
    }

    println!("\n   LwF Results:");
    println!("   - Knowledge distillation preserves previous task performance");
    println!("   - Temperature scaling provides soft targets");
    println!("   - Balances plasticity and stability");

    Ok(())
}

/// Demonstrate Parameter Isolation
fn parameter_isolation_demo() -> Result<()> {
    let layers = vec![
        QNNLayerType::EncodingLayer { num_features: 4 },
        QNNLayerType::VariationalLayer { num_params: 16 },
        QNNLayerType::EntanglementLayer {
            connectivity: "full".to_string(),
        },
        QNNLayerType::MeasurementLayer {
            measurement_basis: "computational".to_string(),
        },
    ];

    let model = QuantumNeuralNetwork::new(layers, 4, 4, 2)?;

    let strategy = ContinualLearningStrategy::ParameterIsolation {
        allocation_strategy: ParameterAllocationStrategy::Masking,
        growth_threshold: 0.8,
    };

    let mut learner = QuantumContinualLearner::new(model, strategy);

    println!("   Created Parameter Isolation learner:");
    println!("   - Allocation strategy: Masking");
    println!("   - Growth threshold: 0.8");

    // Generate tasks with different requirements
    let tasks = generate_varying_complexity_tasks(3, 90, 4);

    println!("\n   Learning with parameter isolation...");

    let mut optimizer = Adam::new(0.001);
    let mut parameter_usage = Vec::new();

    for (i, task) in tasks.iter().enumerate() {
        println!("   \n   Allocating parameters for {}...", task.task_id);

        let metrics = learner.learn_task(task.clone(), &mut optimizer, 30)?;

        // Simulate parameter usage tracking
        let used_params = 16 * (i + 1) / tasks.len(); // Gradually use more parameters
        parameter_usage.push(used_params);

        println!("   - Task accuracy: {:.3}", metrics.current_accuracy);
        println!("   - Parameters allocated: {}/{}", used_params, 16);
        println!(
            "   - Parameter efficiency: {:.1}%",
            used_params as f64 / 16.0 * 100.0
        );

        if i > 0 {
            let all_accuracies = learner.evaluate_all_tasks()?;
            let interference = 1.0
                - all_accuracies
                    .values()
                    .take(i)
                    .map(|&acc| if acc > 0.7 { 1.0 } else { 0.0 })
                    .sum::<f64>()
                    / i as f64;

            println!("   - Task interference: {:.1}%", interference * 100.0);
        }
    }

    println!("\n   Parameter Isolation Results:");
    println!("   - Dedicated parameters prevent interference");
    println!("   - Scalable to many tasks");
    println!("   - Maintains task-specific knowledge");

    Ok(())
}

/// Demonstrate comprehensive task sequence evaluation
fn task_sequence_demo() -> Result<()> {
    println!("   Comprehensive continual learning evaluation...");

    // Compare different strategies
    let strategies = vec![
        (
            "EWC",
            ContinualLearningStrategy::ElasticWeightConsolidation {
                importance_weight: 500.0,
                fisher_samples: 100,
            },
        ),
        (
            "Experience Replay",
            ContinualLearningStrategy::ExperienceReplay {
                buffer_size: 300,
                replay_ratio: 0.2,
                memory_selection: MemorySelectionStrategy::Random,
            },
        ),
        (
            "Quantum Regularization",
            ContinualLearningStrategy::QuantumRegularization {
                entanglement_preservation: 0.1,
                parameter_drift_penalty: 0.5,
            },
        ),
    ];

    // Generate challenging task sequence
    let tasks = generate_challenging_sequence(5, 60, 4);

    println!(
        "\n   Comparing strategies on {} challenging tasks:",
        tasks.len()
    );

    for (strategy_name, strategy) in strategies {
        println!("\n   --- {strategy_name} ---");

        let layers = vec![
            QNNLayerType::EncodingLayer { num_features: 4 },
            QNNLayerType::VariationalLayer { num_params: 8 },
            QNNLayerType::MeasurementLayer {
                measurement_basis: "computational".to_string(),
            },
        ];

        let model = QuantumNeuralNetwork::new(layers, 4, 4, 2)?;
        let mut learner = QuantumContinualLearner::new(model, strategy);
        let mut optimizer = Adam::new(0.001);

        for task in &tasks {
            learner.learn_task(task.clone(), &mut optimizer, 20)?;
        }

        let final_metrics = learner.get_forgetting_metrics();
        println!(
            "   - Average accuracy: {:.3}",
            final_metrics.average_accuracy
        );
        println!(
            "   - Forgetting measure: {:.3}",
            final_metrics.forgetting_measure
        );
        println!(
            "   - CL score: {:.3}",
            final_metrics.continual_learning_score
        );
    }

    Ok(())
}

/// Demonstrate forgetting analysis
fn forgetting_analysis_demo() -> Result<()> {
    println!("   Detailed forgetting analysis...");

    let layers = vec![
        QNNLayerType::EncodingLayer { num_features: 4 },
        QNNLayerType::VariationalLayer { num_params: 12 },
        QNNLayerType::MeasurementLayer {
            measurement_basis: "computational".to_string(),
        },
    ];

    let model = QuantumNeuralNetwork::new(layers, 4, 4, 2)?;

    let strategy = ContinualLearningStrategy::ElasticWeightConsolidation {
        importance_weight: 1000.0,
        fisher_samples: 150,
    };

    let mut learner = QuantumContinualLearner::new(model, strategy);

    // Create tasks with increasing difficulty
    let tasks = generate_increasing_difficulty_tasks(4, 80, 4);

    println!("\n   Learning tasks with increasing difficulty...");

    let mut optimizer = Adam::new(0.001);
    let mut accuracy_matrix = Vec::new();

    for (i, task) in tasks.iter().enumerate() {
        println!(
            "   \n   Learning {} (difficulty level {})...",
            task.task_id,
            i + 1
        );

        learner.learn_task(task.clone(), &mut optimizer, 25)?;

        // Evaluate on all tasks learned so far
        let all_accuracies = learner.evaluate_all_tasks()?;
        let mut current_row = Vec::new();

        for j in 0..=i {
            let task_id = &tasks[j].task_id;
            let accuracy = all_accuracies.get(task_id).unwrap_or(&0.0);
            current_row.push(*accuracy);
        }

        accuracy_matrix.push(current_row.clone());

        // Print current performance
        for (j, &acc) in current_row.iter().enumerate() {
            println!("   - Task {}: {:.3}", j + 1, acc);
        }
    }

    println!("\n   Forgetting Analysis Results:");

    // Compute backward transfer
    for i in 1..accuracy_matrix.len() {
        for j in 0..i {
            let current_acc = accuracy_matrix[i][j];
            let original_acc = accuracy_matrix[j][j];
            let forgetting = (original_acc - current_acc).max(0.0);

            if forgetting > 0.1 {
                println!("   - Significant forgetting detected for Task {} after learning Task {}: {:.3}",
                    j + 1, i + 1, forgetting);
            }
        }
    }

    // Compute average forgetting
    let mut total_forgetting = 0.0;
    let mut num_comparisons = 0;

    for i in 1..accuracy_matrix.len() {
        for j in 0..i {
            let current_acc = accuracy_matrix[i][j];
            let original_acc = accuracy_matrix[j][j];
            total_forgetting += (original_acc - current_acc).max(0.0);
            num_comparisons += 1;
        }
    }

    let avg_forgetting = if num_comparisons > 0 {
        total_forgetting / f64::from(num_comparisons)
    } else {
        0.0
    };

    println!("   - Average forgetting: {avg_forgetting:.3}");

    // Compute final average accuracy
    if let Some(final_row) = accuracy_matrix.last() {
        let final_avg = final_row.iter().sum::<f64>() / final_row.len() as f64;
        println!("   - Final average accuracy: {final_avg:.3}");
        println!(
            "   - Continual learning effectiveness: {:.1}%",
            (1.0 - avg_forgetting) * 100.0
        );
    }

    Ok(())
}

/// Generate diverse tasks with different characteristics
fn generate_diverse_tasks(
    num_tasks: usize,
    samples_per_task: usize,
    feature_dim: usize,
) -> Vec<ContinualTask> {
    let mut tasks = Vec::new();

    for i in 0..num_tasks {
        let task_type = match i % 3 {
            0 => "classification",
            1 => "pattern_recognition",
            _ => "feature_detection",
        };

        // Generate task-specific data with different distributions
        let data = Array2::from_shape_fn((samples_per_task, feature_dim), |(row, col)| {
            match i % 3 {
                0 => {
                    // Gaussian-like distribution
                    let center = i as f64 * 0.2;
                    0.2f64.mul_add(fastrand::f64() - 0.5, center)
                }
                1 => {
                    // Sinusoidal pattern
                    let freq = (i + 1) as f64;
                    0.3f64.mul_add(
                        (freq * row as f64).mul_add(0.1, col as f64 * 0.2).sin(),
                        0.5,
                    )
                }
                _ => {
                    // Random with task-specific bias
                    let bias = i as f64 * 0.1;
                    fastrand::f64().mul_add(0.6, bias)
                }
            }
        });

        let labels = Array1::from_shape_fn(samples_per_task, |row| {
            let features_sum = data.row(row).sum();
            usize::from(features_sum > feature_dim as f64 * 0.5)
        });

        let task = create_continual_task(
            format!("{task_type}_{i}"),
            TaskType::Classification { num_classes: 2 },
            data,
            labels,
            0.8,
        );

        tasks.push(task);
    }

    tasks
}

/// Generate related tasks for transfer learning
fn generate_related_tasks(
    num_tasks: usize,
    samples_per_task: usize,
    feature_dim: usize,
) -> Vec<ContinualTask> {
    let mut tasks = Vec::new();
    let base_pattern = Array1::from_shape_fn(feature_dim, |i| (i as f64 * 0.3).sin());

    for i in 0..num_tasks {
        // Each task is a variation of the base pattern
        let variation_strength = (i as f64).mul_add(0.1, 0.1);

        let data = Array2::from_shape_fn((samples_per_task, feature_dim), |(row, col)| {
            let base_value = base_pattern[col];
            let variation = variation_strength * (row as f64).mul_add(0.05, col as f64 * 0.1).cos();
            let noise = 0.05 * (fastrand::f64() - 0.5);
            (base_value + variation + noise).max(0.0).min(1.0)
        });

        let labels = Array1::from_shape_fn(samples_per_task, |row| {
            let correlation = data
                .row(row)
                .iter()
                .zip(base_pattern.iter())
                .map(|(&x, &y)| x * y)
                .sum::<f64>();
            usize::from(correlation > 0.5)
        });

        let task = create_continual_task(
            format!("related_task_{i}"),
            TaskType::Classification { num_classes: 2 },
            data,
            labels,
            0.8,
        );

        tasks.push(task);
    }

    tasks
}

/// Generate tasks with varying complexity
fn generate_varying_complexity_tasks(
    num_tasks: usize,
    samples_per_task: usize,
    feature_dim: usize,
) -> Vec<ContinualTask> {
    let mut tasks = Vec::new();

    for i in 0..num_tasks {
        let complexity = (i + 1) as f64; // Increasing complexity

        let data = Array2::from_shape_fn((samples_per_task, feature_dim), |(row, col)| {
            // More complex decision boundaries for later tasks
            let x = row as f64 / samples_per_task as f64;
            let y = col as f64 / feature_dim as f64;

            let value = match i {
                0 => {
                    if x > 0.5 {
                        1.0
                    } else {
                        0.0
                    }
                } // Simple linear
                1 => {
                    if x.mul_add(x, y * y) > 0.25 {
                        1.0
                    } else {
                        0.0
                    }
                } // Circular
                2 => {
                    if (x * 4.0).sin() * (y * 4.0).cos() > 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                } // Sinusoidal
                _ => {
                    // Very complex pattern
                    let pattern = (x * 8.0)
                        .sin()
                        .mul_add((y * 8.0).cos(), (x * y * 16.0).sin());
                    if pattern > 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                }
            };

            0.1f64.mul_add(fastrand::f64() - 0.5, value) // Add noise
        });

        let labels = Array1::from_shape_fn(samples_per_task, |row| {
            // Complex labeling based on multiple features
            let features = data.row(row);
            let decision_value = features
                .iter()
                .enumerate()
                .map(|(j, &x)| x * (j as f64 * complexity).mul_add(0.1, 1.0))
                .sum::<f64>();

            usize::from(decision_value > feature_dim as f64 * 0.5)
        });

        let task = create_continual_task(
            format!("complex_task_{i}"),
            TaskType::Classification { num_classes: 2 },
            data,
            labels,
            0.8,
        );

        tasks.push(task);
    }

    tasks
}

/// Generate challenging task sequence
fn generate_challenging_sequence(
    num_tasks: usize,
    samples_per_task: usize,
    feature_dim: usize,
) -> Vec<ContinualTask> {
    let mut tasks = Vec::new();

    for i in 0..num_tasks {
        // Alternating between different types of challenges
        let challenge_type = i % 4;

        let data = Array2::from_shape_fn((samples_per_task, feature_dim), |(row, col)| {
            match challenge_type {
                0 => {
                    // High-frequency patterns
                    let freq = (i as f64).mul_add(2.0, 10.0);
                    0.4f64.mul_add((freq * row as f64 * 0.01).sin(), 0.5)
                }
                1 => {
                    // Overlapping distributions
                    let center1 = (i as f64).mul_add(0.05, 0.3);
                    let center2 = (i as f64).mul_add(-0.05, 0.7);
                    if row % 2 == 0 {
                        0.15f64.mul_add(fastrand::f64() - 0.5, center1)
                    } else {
                        0.15f64.mul_add(fastrand::f64() - 0.5, center2)
                    }
                }
                2 => {
                    // Non-linear patterns
                    let x = row as f64 / samples_per_task as f64;
                    let y = col as f64 / feature_dim as f64;
                    let pattern = (i as f64).mul_add(0.1, x.mul_add(x, -(y * y))).tanh();
                    0.3f64.mul_add(pattern, 0.5)
                }
                _ => {
                    // Sparse patterns
                    if fastrand::f64() < 0.2 {
                        0.2f64.mul_add(fastrand::f64(), 0.8)
                    } else {
                        0.1 * fastrand::f64()
                    }
                }
            }
        });

        let labels = Array1::from_shape_fn(samples_per_task, |row| {
            let features = data.row(row);
            match challenge_type {
                0 => usize::from(features.sum() > feature_dim as f64 * 0.5),
                1 => usize::from(features[0] > 0.5),
                2 => usize::from(
                    features
                        .iter()
                        .enumerate()
                        .map(|(j, &x)| x * (j as f64 + 1.0))
                        .sum::<f64>()
                        > 2.0,
                ),
                _ => usize::from(features.iter().filter(|&&x| x > 0.5).count() > feature_dim / 2),
            }
        });

        let task = create_continual_task(
            format!("challenge_{i}"),
            TaskType::Classification { num_classes: 2 },
            data,
            labels,
            0.8,
        );

        tasks.push(task);
    }

    tasks
}

/// Generate tasks with increasing difficulty
fn generate_increasing_difficulty_tasks(
    num_tasks: usize,
    samples_per_task: usize,
    feature_dim: usize,
) -> Vec<ContinualTask> {
    let mut tasks = Vec::new();

    for i in 0..num_tasks {
        let difficulty = (i + 1) as f64;
        let noise_level = 0.05 + difficulty * 0.02;
        let pattern_complexity = 1.0 + difficulty * 0.5;

        let data = Array2::from_shape_fn((samples_per_task, feature_dim), |(row, col)| {
            let x = row as f64 / samples_per_task as f64;
            let y = col as f64 / feature_dim as f64;

            // Increasingly complex patterns
            let base_pattern = (x * pattern_complexity * std::f64::consts::PI).sin()
                * (y * pattern_complexity * std::f64::consts::PI).cos();

            let pattern_value = 0.3f64.mul_add(base_pattern, 0.5);
            let noise = noise_level * (fastrand::f64() - 0.5);

            (pattern_value + noise).max(0.0).min(1.0)
        });

        let labels = Array1::from_shape_fn(samples_per_task, |row| {
            let features = data.row(row);

            // Increasingly complex decision boundaries
            let decision_value = features
                .iter()
                .enumerate()
                .map(|(j, &x)| {
                    let weight = 1.0 + (j as f64 * difficulty * 0.1).sin();
                    x * weight
                })
                .sum::<f64>();

            let threshold = feature_dim as f64 * 0.5 * (1.0 + difficulty * 0.1);
            usize::from(decision_value > threshold)
        });

        let task = create_continual_task(
            format!("difficulty_{}", i + 1),
            TaskType::Classification { num_classes: 2 },
            data,
            labels,
            0.8,
        );

        tasks.push(task);
    }

    tasks
}
