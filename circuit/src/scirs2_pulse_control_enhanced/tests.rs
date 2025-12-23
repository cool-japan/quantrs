//! Tests for enhanced pulse control

use super::*;

#[test]
fn test_enhanced_pulse_controller_creation() {
    let config = EnhancedPulseConfig::default();
    let controller = EnhancedPulseController::new(config);
    assert!(controller.ml_optimizer.is_some());
}

#[test]
fn test_hardware_constraints_default() {
    let constraints = HardwareConstraints::default();
    assert_eq!(constraints.bandwidth, 500e6);
    assert_eq!(constraints.rise_time, 2e-9);
}

#[test]
fn test_pulse_library_default() {
    let library = PulseLibrary::default();
    assert_eq!(library.gaussian.sigma, 10e-9);
    assert_eq!(library.drag.beta, 0.1);
}

#[test]
fn test_signal_processor_creation() {
    let processor = SignalProcessor::new();
    assert_eq!(processor.config.window_size, 1024);
    assert!(processor.config.enable_simd);
}

#[test]
fn test_filter_state_creation() {
    let state = FilterState::new(4);
    assert_eq!(state.delay_line.capacity(), 4);
    assert_eq!(state.history.capacity(), 4);
}

#[test]
fn test_predistortion_models() {
    let linear = PredistortionModel::Linear;
    let poly = PredistortionModel::Polynomial;
    let memory = PredistortionModel::MemoryPolynomial;

    assert!(matches!(linear, PredistortionModel::Linear));
    assert!(matches!(poly, PredistortionModel::Polynomial));
    assert!(matches!(memory, PredistortionModel::MemoryPolynomial));
}
