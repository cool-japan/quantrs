//! Tests for enhanced transpiler

use super::*;

#[test]
fn test_enhanced_transpiler_creation() {
    let config = EnhancedTranspilerConfig::default();
    let transpiler = EnhancedTranspiler::<100>::new(config);
    assert!(transpiler.ml_router.is_some());
}

#[test]
fn test_hardware_spec_default() {
    let spec = HardwareSpec::default();
    assert_eq!(spec.max_qubits, 27);
    assert_eq!(spec.backend_type, HardwareBackend::Superconducting);
}

#[test]
fn test_optimization_levels() {
    assert_eq!(
        EnhancedTranspilerConfig::default().optimization_level,
        OptimizationLevel::Aggressive
    );
}
