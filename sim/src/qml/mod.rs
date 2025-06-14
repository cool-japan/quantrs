//! Quantum Machine Learning (QML) module.
//!
//! This module provides comprehensive quantum machine learning algorithms
//! with hardware-aware optimization, adaptive training strategies, and
//! support for various quantum computing architectures.

pub mod config;
pub mod circuit;
pub mod trainer;
pub mod benchmarks;

// Re-export commonly used types and structs
pub use config::{
    QMLConfig, HardwareArchitecture, QMLAlgorithmType, GradientMethod, OptimizerType,
};
pub use circuit::{ParameterizedQuantumCircuit, HardwareOptimizations};
pub use trainer::{
    QuantumMLTrainer, OptimizerState, TrainingHistory, HardwareMetrics,
    HardwareAwareCompiler, CompilationStats, TrainingResult,
};
pub use benchmarks::{
    benchmark_quantum_ml_algorithms, benchmark_gradient_methods, benchmark_optimizers,
    run_comprehensive_benchmarks,
};

use crate::error::Result;

/// Initialize the QML subsystem
pub fn initialize() -> Result<()> {
    // Perform any necessary initialization
    Ok(())
}

/// Check if hardware-aware optimization is available
pub fn is_hardware_optimization_available() -> bool {
    // In practice, this would check for hardware-specific libraries
    true
}

/// Get supported hardware architectures
pub fn get_supported_architectures() -> Vec<HardwareArchitecture> {
    vec![
        HardwareArchitecture::NISQ,
        HardwareArchitecture::FaultTolerant,
        HardwareArchitecture::Superconducting,
        HardwareArchitecture::TrappedIon,
        HardwareArchitecture::Photonic,
        HardwareArchitecture::NeutralAtom,
        HardwareArchitecture::ClassicalSimulation,
    ]
}

/// Get supported QML algorithms
pub fn get_supported_algorithms() -> Vec<QMLAlgorithmType> {
    vec![
        QMLAlgorithmType::VQE,
        QMLAlgorithmType::QAOA,
        QMLAlgorithmType::QCNN,
        QMLAlgorithmType::QSVM,
        QMLAlgorithmType::QRL,
        QMLAlgorithmType::QGAN,
        QMLAlgorithmType::QBM,
    ]
}

/// Get supported gradient methods
pub fn get_supported_gradient_methods() -> Vec<GradientMethod> {
    vec![
        GradientMethod::ParameterShift,
        GradientMethod::FiniteDifferences,
        GradientMethod::AutomaticDifferentiation,
        GradientMethod::NaturalGradients,
        GradientMethod::StochasticParameterShift,
    ]
}

/// Get supported optimizers
pub fn get_supported_optimizers() -> Vec<OptimizerType> {
    vec![
        OptimizerType::Adam,
        OptimizerType::SGD,
        OptimizerType::RMSprop,
        OptimizerType::LBFGS,
        OptimizerType::QuantumNaturalGradient,
        OptimizerType::SPSA,
    ]
}

/// Create a default configuration for a specific algorithm
pub fn create_default_config(algorithm: QMLAlgorithmType) -> QMLConfig {
    QMLConfig::for_algorithm(algorithm)
}

/// Create a configuration optimized for specific hardware
pub fn create_hardware_config(hardware: HardwareArchitecture) -> QMLConfig {
    QMLConfig::for_hardware(hardware)
}

/// Validate QML configuration
pub fn validate_config(config: &QMLConfig) -> Result<()> {
    config.validate().map_err(|e| crate::error::SimulatorError::InvalidInput(e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_qml_initialization() {
        let result = initialize();
        assert!(result.is_ok());
    }

    #[test]
    fn test_supported_architectures() {
        let archs = get_supported_architectures();
        assert!(!archs.is_empty());
        assert!(archs.contains(&HardwareArchitecture::NISQ));
    }

    #[test]
    fn test_supported_algorithms() {
        let algos = get_supported_algorithms();
        assert!(!algos.is_empty());
        assert!(algos.contains(&QMLAlgorithmType::VQE));
    }

    #[test]
    fn test_default_config_creation() {
        let config = create_default_config(QMLAlgorithmType::VQE);
        assert_eq!(config.algorithm_type, QMLAlgorithmType::VQE);
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn test_hardware_config_creation() {
        let config = create_hardware_config(HardwareArchitecture::Superconducting);
        assert_eq!(config.hardware_architecture, HardwareArchitecture::Superconducting);
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn test_config_validation() {
        let mut config = QMLConfig::default();
        assert!(validate_config(&config).is_ok());
        
        // Test invalid configuration
        config.num_qubits = 0;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_parameterized_circuit_creation() {
        use crate::circuit_interfaces::InterfaceCircuit;
        
        let circuit = InterfaceCircuit::new(4, 0);
        let parameters = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let parameter_names = vec!["p0".to_string(), "p1".to_string(), "p2".to_string(), "p3".to_string()];
        
        let pqc = ParameterizedQuantumCircuit::new(
            circuit,
            parameters,
            parameter_names,
            HardwareArchitecture::NISQ,
        );
        
        assert_eq!(pqc.num_parameters(), 4);
        assert_eq!(pqc.num_qubits(), 4);
    }

    #[test]
    fn test_hardware_optimizations() {
        let opts = HardwareOptimizations::for_hardware(HardwareArchitecture::Superconducting, 4);
        
        // Test connectivity for superconducting (linear)
        assert!(opts.connectivity_graph[[0, 1]]);
        assert!(opts.connectivity_graph[[1, 2]]);
        assert!(!opts.connectivity_graph[[0, 2]]);
        
        // Test gate fidelities
        assert!(opts.gate_fidelities.contains_key("X"));
        assert!(opts.gate_fidelities.contains_key("CNOT"));
    }

    #[test]
    fn test_trainer_creation() {
        use crate::circuit_interfaces::InterfaceCircuit;
        
        let config = QMLConfig::default();
        let circuit = InterfaceCircuit::new(config.num_qubits, 0);
        let parameters = Array1::zeros(config.num_parameters);
        let parameter_names = (0..config.num_parameters)
            .map(|i| format!("param_{}", i))
            .collect();
        
        let pqc = ParameterizedQuantumCircuit::new(
            circuit,
            parameters,
            parameter_names,
            config.hardware_architecture,
        );
        
        let trainer = QuantumMLTrainer::new(config, pqc, None);
        assert!(trainer.is_ok());
    }

    #[test]
    fn test_optimizer_state() {
        let state = OptimizerState::new(4, 0.01);
        assert_eq!(state.parameters.len(), 4);
        assert_eq!(state.learning_rate, 0.01);
        assert_eq!(state.iteration, 0);
    }

    #[test]
    fn test_training_history() {
        let mut history = TrainingHistory::default();
        history.loss_history.push(1.0);
        history.loss_history.push(0.5);
        history.loss_history.push(0.2);
        
        assert_eq!(history.latest_loss(), Some(0.2));
        assert_eq!(history.best_loss(), Some(0.2));
    }
}