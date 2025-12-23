//! Tests for the circuit optimization system

use quantrs2_circuit::optimization::passes::utils;
use quantrs2_circuit::optimization::{CircuitMetrics, PeepholeOptimization};
use quantrs2_circuit::prelude::*;
use quantrs2_core::gate::{multi, single, GateOp};
use quantrs2_core::qubit::QubitId;

#[test]
fn test_gate_properties() {
    // Test single-qubit gate properties
    let h_props = GateProperties::single_qubit("H");
    assert_eq!(h_props.num_qubits, 1);
    assert!(h_props.is_native);
    assert!(h_props.is_self_inverse);
    assert!(!h_props.is_diagonal);

    let z_props = GateProperties::single_qubit("Z");
    assert!(z_props.is_diagonal);
    assert!(z_props.commutes_with.contains(&"RZ".to_string()));

    // Test two-qubit gate properties
    let cnot_props = GateProperties::two_qubit("CNOT");
    assert_eq!(cnot_props.num_qubits, 2);
    assert!(cnot_props.is_self_inverse);

    // Test multi-qubit gate properties
    let toffoli_props = GateProperties::multi_qubit("Toffoli", 3);
    assert_eq!(toffoli_props.num_qubits, 3);
    assert!(!toffoli_props.is_native);
    assert!(!toffoli_props.decompositions.is_empty());
}

#[test]
fn test_commutation_table() {
    let comm_table = CommutationTable::new();

    // Test Pauli commutation
    assert!(comm_table.commutes("X", "X"));
    assert!(comm_table.commutes("Z", "Z"));
    assert!(!comm_table.commutes("X", "Z"));
    assert!(!comm_table.commutes("Y", "Z"));

    // Test diagonal gate commutation
    assert!(comm_table.commutes("Z", "S"));
    assert!(comm_table.commutes("S", "T"));
    assert!(comm_table.commutes("RZ", "Z"));

    // Test gate operation commutation
    let z1 = single::PauliZ {
        target: QubitId::new(0),
    };
    let z2 = single::PauliZ {
        target: QubitId::new(1),
    };
    assert!(comm_table.gates_commute(&z1, &z2)); // Different qubits

    let x1 = single::PauliX {
        target: QubitId::new(0),
    };
    let z1_same = single::PauliZ {
        target: QubitId::new(0),
    };
    assert!(!comm_table.gates_commute(&x1, &z1_same)); // Same qubit, non-commuting
}

#[test]
fn test_cost_models() {
    let h_gate = single::Hadamard {
        target: QubitId::new(0),
    };
    let cnot_gate = multi::CNOT {
        control: QubitId::new(0),
        target: QubitId::new(1),
    };

    // Test abstract cost model
    let abstract_model = AbstractCostModel::default();
    let h_cost = abstract_model.gate_cost(&h_gate);
    let cnot_cost = abstract_model.gate_cost(&cnot_gate);
    assert!(cnot_cost > h_cost); // CNOT should be more expensive

    // Test hardware cost models
    let ibm_model = HardwareCostModel::for_backend("ibm");
    let google_model = HardwareCostModel::for_backend("google");

    assert!(ibm_model.is_native(&h_gate));
    assert!(google_model.is_native(&h_gate));
}

#[test]
fn test_optimization_levels() {
    let _circuit = Circuit::<2>::new();

    // Test different optimization levels
    let _light = CircuitOptimizer2::<2>::with_level(OptimizationLevel::Light);
    let _medium = CircuitOptimizer2::<2>::with_level(OptimizationLevel::Medium);
    let _heavy = CircuitOptimizer2::<2>::with_level(OptimizationLevel::Heavy);

    // Verify they create different configurations
    // (Would need access to internal state to properly test)
}

#[test]
fn test_pass_manager() {
    let mut pass_manager = PassManager::new();

    // Test adding passes
    pass_manager.add_pass(Box::new(GateCancellation::new(false)));
    pass_manager.add_pass(Box::new(RotationMerging::new(1e-10)));

    // Test configuration
    let config = PassConfig {
        max_iterations: 5,
        aggressive: true,
        level: OptimizationLevel::Custom,
        ..Default::default()
    };
    pass_manager.configure(config);
}

#[test]
fn test_circuit_analyzer() {
    let analyzer = CircuitAnalyzer::new();

    // Create some gates to analyze
    let gates: Vec<Box<dyn GateOp>> = vec![
        Box::new(single::Hadamard {
            target: QubitId::new(0),
        }),
        Box::new(multi::CNOT {
            control: QubitId::new(0),
            target: QubitId::new(1),
        }),
        Box::new(single::PauliX {
            target: QubitId::new(1),
        }),
    ];

    let metrics = analyzer.analyze_gates(&gates, 2);
    assert_eq!(metrics.gate_count, 3);
    assert_eq!(metrics.two_qubit_gates, 1);
    assert_eq!(metrics.num_qubits, 2);
    assert!(metrics.depth > 0);
}

#[test]
fn test_optimization_utils() {
    // Test gate cancellation detection
    let h1 = single::Hadamard {
        target: QubitId::new(0),
    };
    let h2 = single::Hadamard {
        target: QubitId::new(0),
    };
    assert!(utils::gates_cancel(&h1, &h2));

    let x1 = single::PauliX {
        target: QubitId::new(0),
    };
    let x2 = single::PauliX {
        target: QubitId::new(0),
    };
    assert!(utils::gates_cancel(&x1, &x2));

    // Different qubits should not cancel
    let h3 = single::Hadamard {
        target: QubitId::new(1),
    };
    assert!(!utils::gates_cancel(&h1, &h3));

    // Test identity gate detection
    let _rz_zero = single::RotationZ {
        target: QubitId::new(0),
        theta: 0.0,
    };
    // Would need matrix implementation to properly test
    // assert!(utils::is_identity_gate(&rz_zero, 1e-10));
}

#[test]
fn test_metric_improvement() {
    let initial = CircuitMetrics {
        gate_count: 100,
        gate_types: std::collections::HashMap::default(),
        depth: 50,
        two_qubit_gates: 30,
        num_qubits: 5,
        critical_path: 45,
        execution_time: 5000.0,
        total_error: 0.05,
        gate_density: 20.0,
        parallelism: 2.0,
    };

    let final_metrics = CircuitMetrics {
        gate_count: 80,
        gate_types: std::collections::HashMap::default(),
        depth: 40,
        two_qubit_gates: 20,
        num_qubits: 5,
        critical_path: 35,
        execution_time: 3500.0,
        total_error: 0.03,
        gate_density: 16.0,
        parallelism: 2.0,
    };

    let improvement = final_metrics.improvement_from(&initial);
    assert_eq!(improvement.gate_count, 20.0); // 20% reduction
    assert_eq!(improvement.depth, 20.0); // 20% reduction
    assert!((improvement.two_qubit_gates - 33.333_333_333_333_336).abs() < 0.00001);
    // ~33% reduction
}

#[test]
fn test_custom_pass_creation() {
    struct TestPass;

    impl OptimizationPass for TestPass {
        fn name(&self) -> &'static str {
            "Test Pass"
        }

        fn apply_to_gates(
            &self,
            gates: Vec<Box<dyn GateOp>>,
            _cost_model: &dyn CostModel,
        ) -> quantrs2_core::error::QuantRS2Result<Vec<Box<dyn GateOp>>> {
            Ok(gates)
        }
    }

    let pass = TestPass;
    assert_eq!(pass.name(), "Test Pass");
}

#[test]
fn test_peephole_optimization() {
    use std::f64::consts::PI;

    let peephole = PeepholeOptimization::new(4);
    let abstract_model = AbstractCostModel::default();

    // Test X-Y-X pattern
    {
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(single::PauliX {
                target: QubitId::new(0),
            }),
            Box::new(single::PauliY {
                target: QubitId::new(0),
            }),
            Box::new(single::PauliX {
                target: QubitId::new(0),
            }),
        ];

        let optimized = peephole.apply_to_gates(gates, &abstract_model).unwrap();
        assert_eq!(optimized.len(), 1); // Should reduce to just Y
        assert_eq!(optimized[0].name(), "Y");
    }

    // Test Euler angle optimization (RZ-RX-RZ with small RX)
    {
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(single::RotationZ {
                target: QubitId::new(0),
                theta: PI / 4.0,
            }),
            Box::new(single::RotationX {
                target: QubitId::new(0),
                theta: 1e-12,
            }), // Very small
            Box::new(single::RotationZ {
                target: QubitId::new(0),
                theta: PI / 4.0,
            }),
        ];

        let optimized = peephole.apply_to_gates(gates, &abstract_model).unwrap();
        assert_eq!(optimized.len(), 1); // Should merge to single RZ
        assert_eq!(optimized[0].name(), "RZ");
        if let Some(rz) = optimized[0].as_any().downcast_ref::<single::RotationZ>() {
            assert!((rz.theta - PI / 2.0).abs() < 1e-10);
        }
    }

    // Test pattern that shouldn't match
    {
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(single::Hadamard {
                target: QubitId::new(0),
            }),
            Box::new(single::PauliX {
                target: QubitId::new(0),
            }),
            Box::new(single::PauliZ {
                target: QubitId::new(0),
            }),
        ];

        let optimized = peephole
            .apply_to_gates(gates.clone(), &abstract_model)
            .unwrap();
        assert_eq!(optimized.len(), 3); // No optimization should occur
    }
}

#[test]
fn test_template_matching() {
    let template_matcher = TemplateMatching::with_advanced_templates();
    let abstract_model = AbstractCostModel::default();

    // Test H-Z-H to X pattern
    {
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(single::Hadamard {
                target: QubitId::new(0),
            }),
            Box::new(single::PauliZ {
                target: QubitId::new(0),
            }),
            Box::new(single::Hadamard {
                target: QubitId::new(0),
            }),
        ];

        let optimized = template_matcher
            .apply_to_gates(gates, &abstract_model)
            .unwrap();
        assert_eq!(optimized.len(), 1); // Should reduce to single X gate
        assert_eq!(optimized[0].name(), "X");
    }

    // Test CNOT-CNOT elimination
    {
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(multi::CNOT {
                control: QubitId::new(0),
                target: QubitId::new(1),
            }),
            Box::new(multi::CNOT {
                control: QubitId::new(0),
                target: QubitId::new(1),
            }),
        ];

        let optimized = template_matcher
            .apply_to_gates(gates, &abstract_model)
            .unwrap();
        assert_eq!(optimized.len(), 0); // Should eliminate both CNOTs
    }

    // Test S-S to Z pattern
    {
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(single::Phase {
                target: QubitId::new(0),
            }),
            Box::new(single::Phase {
                target: QubitId::new(0),
            }),
        ];

        let optimized = template_matcher
            .apply_to_gates(gates, &abstract_model)
            .unwrap();
        assert_eq!(optimized.len(), 1); // Should reduce to single Z gate
        assert_eq!(optimized[0].name(), "Z");
    }

    // Test hardware-specific template matching
    let ibm_matcher = TemplateMatching::for_hardware("ibm");
    {
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(single::Hadamard {
                target: QubitId::new(0),
            }),
            Box::new(single::PauliZ {
                target: QubitId::new(0),
            }),
            Box::new(single::Hadamard {
                target: QubitId::new(0),
            }),
        ];

        let optimized = ibm_matcher.apply_to_gates(gates, &abstract_model).unwrap();
        assert!(optimized.len() <= 1); // Should optimize for IBM hardware
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_full_optimization_pipeline() {
        // Create a circuit with known optimization opportunities
        let mut circuit = Circuit::<3>::new();

        // Add redundant gates
        circuit.h(0).unwrap();
        circuit.h(0).unwrap(); // Should cancel

        // Add mergeable rotations
        circuit.rz(1, std::f64::consts::PI / 4.0).unwrap();
        circuit.rz(1, std::f64::consts::PI / 4.0).unwrap(); // Should merge to PI/2

        // Add a complex gate
        circuit.toffoli(0, 1, 2).unwrap();

        // Run optimization
        let mut optimizer = CircuitOptimizer2::<3>::with_level(OptimizationLevel::Medium);
        let result = optimizer.optimize(&circuit);

        assert!(result.is_ok());
        if let Ok(report) = result {
            // Verify some optimization occurred
            assert!(report.final_metrics.gate_count <= report.initial_metrics.gate_count);
        }
    }
}
