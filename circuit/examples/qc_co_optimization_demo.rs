//! Quantum-Classical Co-optimization demonstration
//!
//! This example shows how to set up and optimize hybrid quantum-classical algorithms
//! where quantum circuits and classical processing are interleaved and co-optimized.

use quantrs2_circuit::prelude::*;
use quantrs2_circuit::qc_co_optimization::{
    LinearAlgebraOp, LoadBalancingStrategy, MLModelType as QCMLModelType, ScheduleType, UpdateRule,
};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Quantum-Classical Co-optimization Demo");
    println!("=========================================\n");

    // Example 1: Basic hybrid problem setup
    println!("1. Setting up a Hybrid Optimization Problem");
    println!("-------------------------------------------");

    let mut problem = HybridOptimizationProblem::<4>::new();

    // Set global parameters for the optimization
    problem.set_global_parameters(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    println!("Set {} global parameters", problem.global_parameters.len());

    // Create a quantum circuit component
    let mut quantum_circuit = Circuit::<4>::new();
    quantum_circuit.h(Qubit(0))?;
    quantum_circuit.cnot(Qubit(0), Qubit(1))?;
    quantum_circuit.ry(Qubit(2), 0.0)?; // Will be parameterized
    quantum_circuit.cnot(Qubit(2), Qubit(3))?;

    // Add quantum component to the problem
    problem.add_quantum_component(
        "quantum_processor".to_string(),
        quantum_circuit,
        vec![0, 1, 2], // Use parameters 0, 1, 2 from global parameter vector
    )?;

    println!(
        "Added quantum circuit component with {} gates",
        problem.quantum_circuits[0].circuit.num_gates()
    );

    // Example 2: Adding classical processing steps
    println!("\n2. Adding Classical Processing Steps");
    println!("-----------------------------------");

    // Add a classical linear algebra step
    problem.add_classical_step(
        "matrix_operations".to_string(),
        ClassicalStepType::LinearAlgebra(LinearAlgebraOp::MatrixMultiplication),
        vec!["quantum_processor".to_string()],
        vec!["processed_measurements".to_string()],
    )?;

    // Add a machine learning inference step
    problem.add_classical_step(
        "ml_inference".to_string(),
        ClassicalStepType::MachineLearning(QCMLModelType::NeuralNetwork),
        vec!["processed_measurements".to_string()],
        vec!["ml_output".to_string()],
    )?;

    // Add a parameter update step
    problem.add_classical_step(
        "parameter_update".to_string(),
        ClassicalStepType::ParameterUpdate(UpdateRule::AdamOptimizer),
        vec!["ml_output".to_string()],
        vec!["updated_parameters".to_string()],
    )?;

    println!(
        "Added {} classical processing steps",
        problem.classical_steps.len()
    );

    // Print details of each classical step
    for (i, step) in problem.classical_steps.iter().enumerate() {
        println!("  Step {}: {} ({:?})", i + 1, step.id, step.step_type);
    }

    // Example 3: Setting up data flow
    println!("\n3. Setting up Data Flow");
    println!("-----------------------");

    // Connect quantum processor to matrix operations
    problem.add_data_flow(
        "quantum_processor".to_string(),
        "matrix_operations".to_string(),
        DataType::Measurements(vec![0.5, 0.3, 0.8, 0.1]),
    )?;

    // Connect matrix operations to ML inference
    problem.add_data_flow(
        "matrix_operations".to_string(),
        "ml_inference".to_string(),
        DataType::Matrix(vec![vec![1.0, 0.5], vec![0.3, 0.8]]),
    )?;

    // Connect ML inference to parameter update
    problem.add_data_flow(
        "ml_inference".to_string(),
        "parameter_update".to_string(),
        DataType::Scalar(0.75),
    )?;

    println!(
        "Set up {} data flow connections",
        problem.data_flow.edges.len()
    );

    // Print data flow details
    for (i, (source, target, data_type)) in problem.data_flow.edges.iter().enumerate() {
        println!(
            "  Flow {}: {} -> {} ({:?})",
            i + 1,
            source,
            target,
            data_type
        );
    }

    // Example 4: Adding regularization
    println!("\n4. Adding Regularization Terms");
    println!("------------------------------");

    // Add L2 regularization on parameters 0-2
    problem.add_regularization(RegularizationType::L2, 0.01, vec![0, 1, 2])?;

    // Add sparsity regularization on parameters 3-5
    problem.add_regularization(RegularizationType::Sparsity, 0.005, vec![3, 4, 5])?;

    println!(
        "Added {} regularization terms",
        problem.objective.regularization.len()
    );

    for (i, reg) in problem.objective.regularization.iter().enumerate() {
        println!(
            "  Regularization {}: {:?} (strength: {:.3})",
            i + 1,
            reg.reg_type,
            reg.strength
        );
    }

    // Example 5: Validating the problem
    println!("\n5. Problem Validation");
    println!("---------------------");

    match problem.validate() {
        Ok(()) => println!("✅ Problem validation passed"),
        Err(e) => println!("❌ Problem validation failed: {e}"),
    }

    println!("Total components: {}", problem.data_flow.nodes.len());
    println!("Total connections: {}", problem.data_flow.edges.len());

    // Example 6: Different optimization algorithms
    println!("\n6. Optimization Algorithms");
    println!("--------------------------");

    let algorithms = vec![
        (
            "Coordinate Descent",
            HybridOptimizationAlgorithm::CoordinateDescent,
        ),
        (
            "Simultaneous Optimization",
            HybridOptimizationAlgorithm::SimultaneousOptimization,
        ),
        (
            "Hierarchical Optimization",
            HybridOptimizationAlgorithm::HierarchicalOptimization,
        ),
        (
            "Adaptive Optimization",
            HybridOptimizationAlgorithm::AdaptiveOptimization,
        ),
    ];

    for (name, algorithm) in algorithms {
        let optimizer = HybridOptimizer::new(algorithm.clone());
        println!("  {}: {:?}", name, optimizer.algorithm);
        println!("    Max iterations: {}", optimizer.max_iterations);
        println!("    Tolerance: {:.2e}", optimizer.tolerance);
        println!(
            "    Initial learning rate: {}",
            optimizer.learning_rate_schedule.initial_rate
        );
    }

    // Example 7: Learning rate schedules
    println!("\n7. Learning Rate Schedules");
    println!("--------------------------");

    let mut optimizer = HybridOptimizer::new(HybridOptimizationAlgorithm::SimultaneousOptimization);

    // Set up different schedules
    let schedules = vec![
        ("Constant", ScheduleType::Constant),
        ("Linear Decay", ScheduleType::LinearDecay),
        ("Exponential Decay", ScheduleType::ExponentialDecay),
        ("Step Decay", ScheduleType::StepDecay),
        ("Cosine Annealing", ScheduleType::CosineAnnealing),
    ];

    for (name, schedule_type) in schedules {
        optimizer.learning_rate_schedule.schedule_type = schedule_type.clone();
        println!("  {name}: {schedule_type:?}");
    }

    // Example 8: Parallelization configuration
    println!("\n8. Parallelization Configuration");
    println!("--------------------------------");

    optimizer.parallelization.quantum_parallelism = 4;
    optimizer.parallelization.classical_parallelism = 8;
    optimizer.parallelization.asynchronous = true;
    optimizer.parallelization.load_balancing = LoadBalancingStrategy::WorkStealing;

    println!(
        "Quantum parallelism: {} circuits",
        optimizer.parallelization.quantum_parallelism
    );
    println!(
        "Classical parallelism: {} threads",
        optimizer.parallelization.classical_parallelism
    );
    println!(
        "Asynchronous execution: {}",
        optimizer.parallelization.asynchronous
    );
    println!(
        "Load balancing: {:?}",
        optimizer.parallelization.load_balancing
    );

    // Example 9: Running optimization (mock)
    println!("\n9. Running Optimization");
    println!("-----------------------");

    println!("Starting hybrid optimization...");
    let result = optimizer.optimize(&mut problem)?;

    println!("Optimization completed:");
    println!("  Optimal value: {:.6}", result.optimal_value);
    println!("  Converged: {}", result.converged);
    println!("  Iterations: {}", result.iterations);
    println!("  Final parameters: {:?}", result.optimal_parameters);

    // Print some history information
    if !result.history.objective_values.is_empty() {
        println!("  Objective value history (first 5):");
        for (i, &value) in result.history.objective_values.iter().take(5).enumerate() {
            println!("    Iteration {i}: {value:.6}");
        }
    }

    if !result.history.gradient_norms.is_empty() {
        let final_gradient = result.history.gradient_norms.last().unwrap();
        println!("  Final gradient norm: {final_gradient:.2e}");
    }

    // Example 10: Specific hybrid algorithm patterns
    println!("\n10. Common Hybrid Algorithm Patterns");
    println!("------------------------------------");

    // QAOA-like pattern
    println!("QAOA-like pattern:");
    println!("  1. Prepare initial state (quantum)");
    println!("  2. Apply parameterized unitaries (quantum)");
    println!("  3. Measure expectation values (quantum)");
    println!("  4. Classical optimization step (classical)");
    println!("  5. Repeat until convergence");

    // VQE with neural network pattern
    println!("\nVQE with Neural Network pattern:");
    println!("  1. Prepare ansatz circuit (quantum)");
    println!("  2. Measure Pauli expectations (quantum)");
    println!("  3. Neural network post-processing (classical)");
    println!("  4. Gradient-based parameter update (classical)");
    println!("  5. Repeat until convergence");

    // Quantum-enhanced ML pattern
    println!("\nQuantum-enhanced ML pattern:");
    println!("  1. Classical feature encoding (classical)");
    println!("  2. Quantum feature mapping (quantum)");
    println!("  3. Quantum measurements (quantum)");
    println!("  4. Classical ML inference (classical)");
    println!("  5. End-to-end gradient updates (hybrid)");

    // Example 11: Data types and flow
    println!("\n11. Data Types in Hybrid Algorithms");
    println!("-----------------------------------");

    let data_types = vec![
        (
            "Quantum measurements",
            "Raw measurement outcomes from quantum circuits",
        ),
        (
            "Probability distributions",
            "Estimated probability distributions from measurements",
        ),
        (
            "Classical matrices",
            "Processed data in matrix form for linear algebra",
        ),
        (
            "Scalar values",
            "Single numerical values like energies or costs",
        ),
        (
            "Parameter vectors",
            "Optimization parameters for both quantum and classical components",
        ),
        (
            "Control signals",
            "Boolean flags for adaptive algorithm control",
        ),
    ];

    for (dtype, description) in data_types {
        println!("  {dtype}: {description}");
    }

    println!("\n✅ Quantum-Classical Co-optimization Demo completed!");
    println!("\nNote: This demo shows the framework structure for hybrid optimization.");
    println!("Real co-optimization requires actual quantum and classical execution engines.");

    Ok(())
}
