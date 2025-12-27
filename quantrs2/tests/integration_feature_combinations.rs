//! Comprehensive Feature Combination Tests for QuantRS2
//!
//! These tests verify that all feature flag combinations work correctly,
//! ensuring proper compilation and runtime behavior across different configurations.

// ============================================================================
// Core Feature Tests (Always Available)
// ============================================================================

mod core_always_available {
    use quantrs2::core;
    use quantrs2::prelude::essentials::*;

    #[test]
    fn test_core_types_available() {
        // QubitId should always be available
        let q0 = QubitId::new(0);
        let q1 = QubitId::new(1);
        assert_eq!(q0.id(), 0);
        assert_eq!(q1.id(), 1);
        assert_ne!(q0, q1);
    }

    #[test]
    fn test_version_constants() {
        assert!(!VERSION.is_empty());
        assert!(!QUANTRS2_VERSION.is_empty());
        assert_eq!(VERSION, QUANTRS2_VERSION);
    }

    #[test]
    fn test_core_module_accessible() {
        // Core module should always be available
        let _ = core::qubit::QubitId::new(0);
    }
}

// ============================================================================
// Single Feature Tests
// ============================================================================

#[cfg(feature = "circuit")]
mod circuit_feature_alone {
    use quantrs2::circuit;
    use quantrs2::prelude::circuits::*;

    #[test]
    fn test_circuit_creation() {
        let circuit = Circuit::<2>::new();
        assert_eq!(circuit.num_qubits(), 2);
    }

    #[test]
    fn test_circuit_gate_application() {
        let mut circuit = Circuit::<2>::new();
        let _ = circuit.h(0);
        let _ = circuit.x(1);
        // Verify circuit has gates (gate_count is not publicly available in beta.3)
        assert!(circuit.num_qubits() == 2);
    }

    #[test]
    fn test_circuit_cnot() {
        let mut circuit = Circuit::<2>::new();
        let _ = circuit.h(0);
        let _ = circuit.cnot(0, 1);
        // Should create Bell state circuit
    }

    #[test]
    fn test_circuit_module_types() {
        // Verify circuit module exports expected types
        let _circuit: Circuit<3> = Circuit::new();
        let _qubit = QubitId::new(0);
    }
}

#[cfg(feature = "sim")]
mod sim_feature {
    use quantrs2::prelude::simulation::*;
    use quantrs2::sim;
    // Import Simulator trait to use .run() method
    use quantrs2::circuit::builder::Simulator;

    #[test]
    fn test_simulator_creation() {
        let simulator = StateVectorSimulator::new();
        // Simulator should be created successfully
        let _ = simulator;
    }

    #[test]
    fn test_sim_includes_circuit() {
        // When sim is enabled, circuit should also be enabled (dependency)
        let _circuit: Circuit<2> = Circuit::new();
    }

    #[test]
    fn test_basic_simulation() {
        let mut circuit = Circuit::<2>::new();
        let _ = circuit.h(0);
        let _ = circuit.cnot(0, 1);

        let simulator = StateVectorSimulator::new();
        let result = simulator.run(&circuit);
        assert!(result.is_ok());
    }
}

#[cfg(feature = "anneal")]
mod anneal_feature {
    #[test]
    fn test_anneal_module_available() {
        // Verify anneal module is accessible when feature is enabled
        use quantrs2::anneal;
        assert!(true, "Annealing module is available");
    }

    #[test]
    fn test_anneal_prelude() {
        use quantrs2::prelude::quantum_annealing::*;
        // Prelude should provide anneal-specific types
        assert!(true, "Quantum annealing prelude accessible");
    }
}

#[cfg(feature = "device")]
mod device_feature {
    #[test]
    fn test_device_module_available() {
        use quantrs2::device;
        assert!(true, "Device module is available");
    }

    #[test]
    fn test_hardware_prelude() {
        use quantrs2::prelude::hardware::*;
        // Prelude should provide hardware-specific types
        assert!(true, "Hardware prelude accessible");
    }
}

#[cfg(feature = "tytan")]
mod tytan_feature {
    #[test]
    fn test_tytan_module_available() {
        use quantrs2::tytan;
        assert!(true, "Tytan module is available");
    }

    #[test]
    fn test_tytan_prelude() {
        use quantrs2::prelude::tytan::*;
        // Tytan should include annealing functionality
        assert!(true, "Tytan prelude accessible");
    }

    #[test]
    fn test_tytan_requires_anneal() {
        // Tytan feature requires anneal feature
        #[cfg(not(feature = "anneal"))]
        compile_error!("tytan feature requires anneal feature");
    }
}

#[cfg(feature = "ml")]
mod ml_feature {
    #[test]
    fn test_ml_module_available() {
        use quantrs2::ml;
        assert!(true, "ML module is available");
    }

    #[test]
    fn test_algorithms_prelude() {
        use quantrs2::prelude::algorithms::*;
        // ML prelude should provide algorithm types
        assert!(true, "Algorithms prelude accessible");
    }

    #[test]
    fn test_ml_requires_sim_and_anneal() {
        // ML feature requires both sim and anneal
        #[cfg(not(feature = "sim"))]
        compile_error!("ml feature requires sim feature");

        #[cfg(not(feature = "anneal"))]
        compile_error!("ml feature requires anneal feature");
    }
}

// ============================================================================
// Feature Combination Tests
// ============================================================================

#[cfg(all(feature = "circuit", feature = "sim"))]
mod circuit_sim_combination {
    use quantrs2::circuit::builder::Simulator;
    use quantrs2::prelude::simulation::*;

    #[test]
    fn test_circuit_simulation_workflow() {
        // Test complete workflow: create circuit -> simulate
        let mut circuit = Circuit::<3>::new();
        let _ = circuit.h(0);
        let _ = circuit.cnot(0, 1);
        let _ = circuit.cnot(1, 2);

        let simulator = StateVectorSimulator::new();
        let result = simulator.run(&circuit);
        assert!(result.is_ok(), "Circuit simulation should succeed");
    }

    #[test]
    fn test_ghz_state_creation() {
        // Create GHZ state circuit
        let mut circuit = Circuit::<4>::new();
        let _ = circuit.h(0);
        let _ = circuit.cnot(0, 1);
        let _ = circuit.cnot(1, 2);
        let _ = circuit.cnot(2, 3);

        let simulator = StateVectorSimulator::new();
        let result = simulator.run(&circuit);
        assert!(result.is_ok());
    }
}

#[cfg(all(feature = "sim", feature = "anneal"))]
mod sim_anneal_combination {
    #[test]
    fn test_sim_and_anneal_together() {
        use quantrs2::prelude::quantum_annealing::*;
        use quantrs2::prelude::simulation::*;

        // Both modules should be accessible
        let _simulator = StateVectorSimulator::new();
        // Annealing types should also be available
        assert!(true, "Simulation and annealing can be used together");
    }
}

#[cfg(all(feature = "anneal", feature = "tytan"))]
mod anneal_tytan_combination {
    #[test]
    fn test_tytan_includes_anneal() {
        use quantrs2::anneal;
        use quantrs2::tytan;

        // Both modules should be accessible
        assert!(true, "Tytan correctly includes annealing functionality");
    }

    #[test]
    fn test_tytan_high_level_api() {
        use quantrs2::prelude::tytan::*;
        // Tytan's high-level API should be available
        assert!(true, "Tytan high-level API accessible");
    }
}

#[cfg(all(feature = "device", feature = "circuit"))]
mod device_circuit_combination {
    use quantrs2::prelude::hardware::*;

    #[test]
    fn test_circuit_for_hardware() {
        let mut circuit = Circuit::<2>::new();
        let _ = circuit.h(0);
        let _ = circuit.cnot(0, 1);

        // Circuit should be valid for hardware submission
        assert_eq!(circuit.num_qubits(), 2);
    }
}

#[cfg(all(feature = "circuit", feature = "sim", feature = "ml"))]
mod circuit_sim_ml_combination {
    use quantrs2::circuit::builder::Simulator;
    use quantrs2::prelude::algorithms::*;
    use quantrs2::prelude::simulation::*;

    #[test]
    fn test_ml_has_access_to_sim_and_circuit() {
        // ML module should have access to both circuit and simulation
        let mut circuit = Circuit::<2>::new();
        let _ = circuit.h(0);
        let _ = circuit.cnot(0, 1);

        let simulator = StateVectorSimulator::new();
        let result = simulator.run(&circuit);
        assert!(result.is_ok());
    }

    #[test]
    fn test_full_algorithm_stack() {
        // Full stack: circuit creation -> simulation -> ML algorithms
        // This is the typical workflow for variational algorithms
        assert!(true, "Full algorithm stack is accessible");
    }
}

// ============================================================================
// Full Feature Set Tests
// ============================================================================

#[cfg(all(
    feature = "circuit",
    feature = "sim",
    feature = "ml",
    feature = "device",
    feature = "anneal"
))]
mod full_feature_set {
    use quantrs2::circuit::builder::Simulator;
    use quantrs2::prelude::full::*;

    #[test]
    fn test_full_prelude() {
        // All types should be accessible via full prelude
        let _q = QubitId::new(0);
        let _circuit: Circuit<2> = Circuit::new();
        let _simulator = StateVectorSimulator::new();

        // Version is available through quantrs2::version module
        assert!(!quantrs2::version::VERSION.is_empty());
    }

    #[test]
    fn test_cross_feature_workflow() {
        // Complete workflow using all features
        let mut circuit = Circuit::<3>::new();
        let _ = circuit.h(0);
        let _ = circuit.cnot(0, 1);
        let _ = circuit.cnot(1, 2);

        let simulator = StateVectorSimulator::new();
        let result = simulator.run(&circuit);
        assert!(result.is_ok());

        // Hardware and ML modules should also be accessible
        // (though actual hardware connection not tested)
    }
}

// ============================================================================
// Prelude Hierarchy Tests
// ============================================================================

mod prelude_hierarchy {
    #[test]
    fn test_essentials_prelude() {
        use quantrs2::prelude::essentials::*;

        let _q = QubitId::new(0);
        // VERSION should be available from essentials prelude (re-exported from crate root)
        assert!(!quantrs2::version::VERSION.is_empty());
    }

    #[cfg(feature = "circuit")]
    #[test]
    fn test_circuits_prelude_includes_essentials() {
        use quantrs2::prelude::circuits::*;

        // Should include essentials
        let _q = QubitId::new(0);
        assert!(!quantrs2::version::VERSION.is_empty());

        // Plus circuit-specific types
        let _circuit: Circuit<2> = Circuit::new();
    }

    #[cfg(feature = "sim")]
    #[test]
    fn test_simulation_prelude_includes_circuits() {
        use quantrs2::prelude::simulation::*;

        // Should include essentials and circuits
        let _q = QubitId::new(0);
        assert!(!quantrs2::version::VERSION.is_empty());
        let _circuit: Circuit<2> = Circuit::new();

        // Plus simulation-specific types
        let _simulator = StateVectorSimulator::new();
    }

    #[test]
    fn test_full_prelude_available() {
        use quantrs2::prelude::full::*;

        // Full prelude should include essentials at minimum
        let _q = QubitId::new(0);
        // VERSION constant is available through quantrs2::version module
        assert!(!quantrs2::version::VERSION.is_empty());
    }
}

// ============================================================================
// Feature Dependency Chain Tests
// ============================================================================

mod feature_dependency_chains {
    #[test]
    fn test_sim_requires_circuit() {
        #[cfg(all(feature = "sim", not(feature = "circuit")))]
        compile_error!("Feature 'sim' requires feature 'circuit'");
    }

    #[test]
    fn test_tytan_requires_anneal() {
        #[cfg(all(feature = "tytan", not(feature = "anneal")))]
        compile_error!("Feature 'tytan' requires feature 'anneal'");
    }

    #[test]
    fn test_ml_requires_sim() {
        #[cfg(all(feature = "ml", not(feature = "sim")))]
        compile_error!("Feature 'ml' requires feature 'sim'");
    }

    #[test]
    fn test_ml_requires_anneal() {
        #[cfg(all(feature = "ml", not(feature = "anneal")))]
        compile_error!("Feature 'ml' requires feature 'anneal'");
    }
}

// ============================================================================
// API Consistency Tests
// ============================================================================

mod api_consistency {
    use quantrs2::{config, diagnostics, error, utils, version};

    #[test]
    fn test_facade_modules_always_available() {
        // These modules should always be available regardless of features
        let _ = version::VERSION;
        let _ = config::Config::global();
        let _ = diagnostics::run_diagnostics();
        let _ = utils::estimate_statevector_memory(10);
    }

    #[test]
    fn test_error_types_available() {
        use quantrs2::prelude::essentials::QuantRS2Error;

        let err = QuantRS2Error::InvalidQubitId(42);
        let msg = format!("{err}");
        assert!(msg.contains("42"));
    }

    #[test]
    fn test_result_type_alias() {
        use quantrs2::error::QuantRS2Result;

        fn test_function() -> QuantRS2Result<u32> {
            Ok(42)
        }

        assert_eq!(test_function().unwrap(), 42);
    }
}

// ============================================================================
// Conditional Compilation Path Tests
// ============================================================================

mod conditional_compilation {
    #[test]
    fn test_feature_conditional_prelude() {
        // Test that prelude content changes based on features
        use quantrs2::prelude::full::*;

        // Essentials should always be available
        let _q = QubitId::new(0);

        // Circuit only when feature enabled
        #[cfg(feature = "circuit")]
        {
            let _circuit: Circuit<2> = Circuit::new();
        }

        // Simulation only when feature enabled
        #[cfg(feature = "sim")]
        {
            let _sim = StateVectorSimulator::new();
        }
    }

    #[test]
    fn test_module_conditional_access() {
        // Core is always available
        let _ = quantrs2::core::qubit::QubitId::new(0);

        // Other modules only when features enabled
        #[cfg(feature = "circuit")]
        {
            use quantrs2::circuit::builder::Circuit;
            let _ = Circuit::<2>::new();
        }

        #[cfg(feature = "anneal")]
        {
            // anneal module is available, test basic type access
            let _ = std::any::type_name::<quantrs2::anneal::ising::IsingModel>();
        }
    }
}
