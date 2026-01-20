#![allow(clippy::pedantic, clippy::assertions_on_constants)]
//! Feature flag integration tests for the QuantRS2 facade crate

// Test that core is always available
#[test]
fn test_core_always_available() {
    use quantrs2::core;

    // QubitId should always be available from core
    let _q = core::qubit::QubitId::new(0);
}

// Test circuit feature availability
#[cfg(feature = "circuit")]
#[test]
fn test_circuit_feature() {
    use quantrs2::circuit;
    use quantrs2::prelude::circuits::*;

    // Circuit should be available
    let _circuit = Circuit::<2>::new();

    // QubitId should be available through prelude
    let _q = QubitId::new(0);
}

// Test that circuit prelude is not available without the feature
#[cfg(not(feature = "circuit"))]
#[test]
fn test_circuit_not_available() {
    // This should compile - we're just checking that the module doesn't exist
    // The actual check is that the code below would NOT compile:
    // use quantrs2::circuit::Circuit; // Would fail without feature
}

// Test simulation feature availability
#[cfg(feature = "sim")]
#[test]
fn test_sim_feature() {
    use quantrs2::prelude::simulation::*;
    use quantrs2::sim;

    // StateVectorSimulator should be available
    let _simulator = StateVectorSimulator::new();

    // Circuit should also be available (sim depends on circuit)
    let _circuit = Circuit::<2>::new();
}

// Test ML feature availability
#[cfg(feature = "ml")]
#[test]
fn test_ml_feature() {
    use quantrs2::ml;
    use quantrs2::prelude::algorithms::*;

    // ML module should be available
    // Note: We're just checking module accessibility, not running algorithms
}

// Test annealing feature availability
#[cfg(feature = "anneal")]
#[test]
fn test_anneal_feature() {
    use quantrs2::anneal;
    use quantrs2::prelude::quantum_annealing::*;

    // Annealing module should be available
}

// Test Tytan feature availability
#[cfg(feature = "tytan")]
#[test]
fn test_tytan_feature() {
    use quantrs2::prelude::tytan::*;
    use quantrs2::tytan;

    // Tytan module should be available
}

// Test device feature availability
#[cfg(feature = "device")]
#[test]
fn test_device_feature() {
    use quantrs2::device;
    use quantrs2::prelude::hardware::*;

    // Device module should be available
}

// Test symengine feature availability
#[cfg(feature = "symengine")]
#[test]
fn test_symengine_feature() {
    use quantrs2::symengine;

    // SymEngine module should be available
}

// Test that preludes work correctly with different feature combinations
#[test]
fn test_essentials_prelude_always_works() {
    use quantrs2::prelude::essentials::*;

    // Essentials should always work regardless of features
    let _q = QubitId::new(0);
    assert!(!VERSION.is_empty());
}

#[test]
fn test_full_prelude_includes_available_features() {
    use quantrs2::prelude::full::*;

    // Full prelude should include all enabled features
    let _q = QubitId::new(0);

    // Circuit should be available if feature is enabled
    #[cfg(feature = "circuit")]
    {
        let _circuit = Circuit::<2>::new();
    }

    // Simulator should be available if feature is enabled
    #[cfg(feature = "sim")]
    {
        let _simulator = StateVectorSimulator::new();
    }
}

#[cfg(all(feature = "circuit", feature = "sim"))]
#[test]
fn test_circuit_sim_integration() {
    use quantrs2::prelude::simulation::*;

    // When both circuit and sim are enabled, we should be able to use both
    let mut circuit = Circuit::<2>::new();
    circuit.h(QubitId::new(0)).unwrap();
    circuit.cnot(QubitId::new(0), QubitId::new(1)).unwrap();

    let simulator = StateVectorSimulator::new();
    // Note: Just testing that types are available, not running full simulation
}

#[cfg(all(feature = "sim", feature = "ml"))]
#[test]
fn test_sim_ml_integration() {
    use quantrs2::prelude::algorithms::*;

    // When both sim and ml are enabled, algorithms should have access to both
    // Note: Just testing module accessibility
}

#[cfg(all(feature = "anneal", feature = "tytan"))]
#[test]
fn test_anneal_tytan_integration() {
    use quantrs2::prelude::tytan::*;

    // Tytan should include annealing functionality
    // Note: Just testing module accessibility
}

// Test feature dependency validation
#[test]
fn test_feature_dependencies() {
    // If sim is enabled, circuit should also be enabled (dependency)
    #[cfg(feature = "sim")]
    {
        #[cfg(not(feature = "circuit"))]
        compile_error!("sim feature requires circuit feature");
    }

    // If tytan is enabled, anneal should also be enabled (dependency)
    #[cfg(feature = "tytan")]
    {
        #[cfg(not(feature = "anneal"))]
        compile_error!("tytan feature requires anneal feature");
    }

    // If ml is enabled, sim and anneal should also be enabled (dependencies)
    #[cfg(feature = "ml")]
    {
        #[cfg(not(feature = "sim"))]
        compile_error!("ml feature requires sim feature");

        #[cfg(not(feature = "anneal"))]
        compile_error!("ml feature requires anneal feature");
    }
}
