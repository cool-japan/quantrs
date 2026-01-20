#![allow(clippy::pedantic, clippy::unnecessary_wraps)]
//! Example demonstrating the QAOA circuit bridge with the circuit module
//!
//! This example shows how to use the QAOA circuit bridge to:
//! 1. Convert QAOA circuits to the circuit module format
//! 2. Apply circuit optimizations to QAOA circuits
//! 3. Estimate the benefits of circuit optimization for QAOA

use std::f64::consts::PI;

use quantrs2_anneal::{
    ising::IsingModel,
    qaoa::{
        MixerType, ParameterInitialization, ProblemEncoding, QaoaClassicalOptimizer, QaoaConfig,
        QaoaOptimizer, QaoaVariant,
    },
    qaoa_circuit_bridge::{
        create_qaoa_bridge_for_problem, validate_circuit_compatibility, EnhancedQaoaOptimizer,
        OptimizationLevel, QaoaCircuitBridge,
    },
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("QAOA Circuit Bridge Example");
    println!("===========================\n");

    // Create a simple test problem: Max-Cut on a triangle
    let mut problem = IsingModel::new(3);

    // Add couplings for edges of triangle (all with weight -1 for Max-Cut)
    problem.set_coupling(0, 1, -1.0)?;
    problem.set_coupling(1, 2, -1.0)?;
    problem.set_coupling(2, 0, -1.0)?;

    println!("Created Max-Cut problem on triangle:");
    println!("  - 3 qubits");
    println!("  - 3 edges with coupling strength -1.0");

    // Example 1: Basic bridge usage
    println!("\n1. Basic QAOA Circuit Bridge");
    println!("----------------------------");

    let bridge = create_qaoa_bridge_for_problem(&problem);
    println!("Created bridge for {} qubits", bridge.num_qubits);

    // Create QAOA parameters
    let parameters = vec![PI / 4.0, PI / 3.0]; // gamma, beta for 1 layer

    // Build optimizable QAOA circuit
    let circuit = bridge.build_optimizable_qaoa_circuit(&problem, &parameters, 1)?;
    println!(
        "Built QAOA circuit with {} gates and {} parameters",
        circuit.gates.len(),
        circuit.num_parameters
    );

    // Validate circuit compatibility
    validate_circuit_compatibility(&circuit)?;
    println!("Circuit validation passed");

    // Example 2: Enhanced QAOA optimizer with circuit optimization
    println!("\n2. Enhanced QAOA Optimizer");
    println!("--------------------------");

    let enhanced_optimizer = EnhancedQaoaOptimizer::new(3, OptimizationLevel::Basic);

    // Build optimized circuit
    let optimized_circuit = enhanced_optimizer.build_optimized_circuit(&problem, &parameters, 1)?;
    println!(
        "Built optimized QAOA circuit with {} gates",
        optimized_circuit.gates.len()
    );

    // Estimate circuit cost
    let cost_estimate = enhanced_optimizer.estimate_circuit_cost(&optimized_circuit);
    println!("Circuit cost estimate:");
    println!("  - Total gates: {}", cost_estimate.total_gates);
    println!(
        "  - Single-qubit gates: {}",
        cost_estimate.single_qubit_gates
    );
    println!("  - Two-qubit gates: {}", cost_estimate.two_qubit_gates);
    println!(
        "  - Estimated execution time: {:.3} ms",
        cost_estimate.estimated_execution_time_ms
    );

    // Example 3: Compare different optimization levels
    println!("\n3. Optimization Level Comparison");
    println!("--------------------------------");

    let optimization_levels = vec![
        OptimizationLevel::None,
        OptimizationLevel::Basic,
        OptimizationLevel::Advanced,
    ];

    for level in optimization_levels {
        let optimizer = EnhancedQaoaOptimizer::new(3, level.clone());
        let circuit = optimizer.build_optimized_circuit(&problem, &parameters, 2)?; // 2 layers
        let cost = optimizer.estimate_circuit_cost(&circuit);

        println!("Optimization level: {level:?}");
        println!(
            "  - Gates: {}, Depth: {}, Time: {:.3} ms",
            cost.total_gates, cost.estimated_depth, cost.estimated_execution_time_ms
        );
    }

    // Example 4: Gate conversion demonstration
    println!("\n4. Gate Conversion Examples");
    println!("---------------------------");

    use quantrs2_anneal::qaoa::QuantumGate as QaoaGate;

    let bridge = QaoaCircuitBridge::new(3);

    // Convert various QAOA gates
    let qaoa_gates = [
        QaoaGate::RX {
            qubit: 0,
            angle: PI / 2.0,
        },
        QaoaGate::RY {
            qubit: 1,
            angle: PI / 4.0,
        },
        QaoaGate::RZ {
            qubit: 2,
            angle: PI / 6.0,
        },
        QaoaGate::CNOT {
            control: 0,
            target: 1,
        },
        QaoaGate::ZZ {
            qubit1: 1,
            qubit2: 2,
            angle: PI / 8.0,
        },
        QaoaGate::H { qubit: 0 },
    ];

    for (i, qaoa_gate) in qaoa_gates.iter().enumerate() {
        match bridge.convert_qaoa_gate_to_circuit_gates(qaoa_gate) {
            Ok(circuit_gates) => {
                println!(
                    "Gate {}: {:?} -> {} circuit gates",
                    i + 1,
                    qaoa_gate,
                    circuit_gates.len()
                );
                for (j, gate) in circuit_gates.iter().enumerate() {
                    println!(
                        "  Gate {}.{}: {} on qubits {:?}",
                        i + 1,
                        j + 1,
                        gate.name(),
                        gate.qubits()
                    );
                }
            }
            Err(e) => {
                println!("Gate {}: {:?} -> Error: {}", i + 1, qaoa_gate, e);
            }
        }
    }

    // Example 5: Multi-layer QAOA circuit
    println!("\n5. Multi-layer QAOA Circuit");
    println!("---------------------------");

    let layers = 3;
    let multi_params = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]; // 3 layers × 2 params per layer

    let multi_circuit = bridge.build_optimizable_qaoa_circuit(&problem, &multi_params, layers)?;
    println!("Multi-layer QAOA circuit ({layers} layers):");
    println!("  - Total gates: {}", multi_circuit.gates.len());
    println!(
        "  - Parameter references: {}",
        multi_circuit.parameter_map.len()
    );

    // Count parameter types
    let mut gamma_count = 0;
    let mut beta_count = 0;
    for param_ref in &multi_circuit.parameter_map {
        match param_ref.parameter_type {
            quantrs2_anneal::qaoa_circuit_bridge::ParameterType::Gamma => gamma_count += 1,
            quantrs2_anneal::qaoa_circuit_bridge::ParameterType::Beta => beta_count += 1,
        }
    }
    println!("  - Gamma parameters: {gamma_count}");
    println!("  - Beta parameters: {beta_count}");

    // Example 6: Problem representation for circuit optimization
    println!("\n6. Problem Representation");
    println!("-------------------------");

    let problem_repr = bridge.prepare_problem_for_circuit_optimization(&problem)?;
    println!("Problem representation:");
    println!("  - Number of qubits: {}", problem_repr.num_qubits);
    println!("  - Linear terms: {}", problem_repr.linear_terms.len());
    println!(
        "  - Quadratic terms: {}",
        problem_repr.quadratic_terms.len()
    );

    for (i, quad_term) in problem_repr.quadratic_terms.iter().enumerate() {
        println!(
            "    Quadratic term {}: J_{{{},{}}} = {}",
            i + 1,
            quad_term.qubit1,
            quad_term.qubit2,
            quad_term.coefficient
        );
    }

    // Example 7: Traditional QAOA vs Enhanced QAOA
    println!("\n7. Traditional vs Enhanced QAOA");
    println!("-------------------------------");

    // Traditional QAOA
    let qaoa_config = QaoaConfig {
        variant: QaoaVariant::Standard { layers: 2 },
        mixer_type: MixerType::XMixer,
        problem_encoding: ProblemEncoding::Ising,
        optimizer: QaoaClassicalOptimizer::NelderMead {
            initial_size: 0.1,
            tolerance: 1e-6,
            max_iterations: 100,
        },
        num_shots: 1000,
        parameter_init: ParameterInitialization::Random { range: (-PI, PI) },
        convergence_tolerance: 1e-6,
        max_optimization_time: Some(std::time::Duration::from_secs(30)),
        seed: Some(42),
        detailed_logging: false,
        track_optimization_history: true,
        max_circuit_depth: None,
        use_symmetry_reduction: false,
    };

    // Note: For demonstration purposes, we're just showing the setup
    // In practice, you would run the optimization
    println!("Traditional QAOA configuration created");
    println!("  - Variant: Standard with 2 layers");
    println!("  - Mixer: X-mixer");
    println!("  - Optimizer: Nelder-Mead");

    // Enhanced QAOA
    let enhanced = EnhancedQaoaOptimizer::new(3, OptimizationLevel::Advanced);
    println!("Enhanced QAOA optimizer created");
    println!("  - Circuit optimization: enabled");
    println!("  - Optimization level: Advanced");

    println!("\nQAOA Circuit Bridge Example completed successfully!");

    Ok(())
}

/// Helper function to demonstrate parameter extraction
fn demonstrate_parameter_extraction() -> Result<(), Box<dyn std::error::Error>> {
    println!("\nParameter Extraction Demo");
    println!("------------------------");

    let problem = IsingModel::new(2);
    let bridge = QaoaCircuitBridge::new(2);
    let parameters = vec![0.5, 1.0]; // gamma, beta

    let circuit = bridge.build_optimizable_qaoa_circuit(&problem, &parameters, 1)?;
    let extracted = bridge.extract_qaoa_parameters(&circuit);

    println!("Original parameters: {parameters:?}");
    println!("Extracted parameters: {extracted:?}");

    Ok(())
}

/// Helper function to show optimization metrics
fn demonstrate_optimization_metrics() -> Result<(), Box<dyn std::error::Error>> {
    println!("\nOptimization Metrics Demo");
    println!("------------------------");

    let problem = IsingModel::new(4);
    let bridge = QaoaCircuitBridge::new(4);
    let parameters = vec![0.1, 0.2, 0.3, 0.4]; // 2 layers

    let original = bridge.build_optimizable_qaoa_circuit(&problem, &parameters, 2)?;
    let optimized = bridge.optimize_qaoa_circuit(&original)?;

    let metrics = bridge.estimate_optimization_benefit(&original, &optimized);

    println!("Optimization metrics:");
    println!("  - Original depth: {}", metrics.original_depth);
    println!("  - Optimized depth: {}", metrics.optimized_depth);
    println!("  - Gate reduction: {}", metrics.gate_count_reduction);
    println!("  - Estimated speedup: {:.2}x", metrics.estimated_speedup);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_basic_functionality() {
        let problem = IsingModel::new(3);
        let bridge = create_qaoa_bridge_for_problem(&problem);
        assert_eq!(bridge.num_qubits, 3);
    }

    #[test]
    fn test_enhanced_optimizer_creation() {
        let optimizer = EnhancedQaoaOptimizer::new(4, OptimizationLevel::Basic);
        assert_eq!(optimizer.bridge.num_qubits, 4);
        assert!(optimizer.enable_circuit_optimization);
    }

    #[test]
    fn test_circuit_building() {
        let mut problem = IsingModel::new(2);
        problem.set_coupling(0, 1, -1.0).unwrap();

        let bridge = QaoaCircuitBridge::new(2);
        let parameters = vec![0.5, 1.0];

        let circuit = bridge
            .build_optimizable_qaoa_circuit(&problem, &parameters, 1)
            .unwrap();
        assert!(circuit.gates.len() > 0);
        assert_eq!(circuit.num_qubits, 2);
        assert_eq!(circuit.num_parameters, 2);
    }
}
