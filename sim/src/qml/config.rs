//! Configuration structures and enums for quantum machine learning algorithms.
//!
//! This module provides configuration types for hardware architectures,
//! algorithm types, optimization methods, and training parameters.

use serde::{Deserialize, Serialize};

/// Hardware architecture types for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HardwareArchitecture {
    /// Noisy Intermediate-Scale Quantum devices
    NISQ,
    /// Fault-tolerant quantum computers
    FaultTolerant,
    /// Superconducting quantum processors
    Superconducting,
    /// Trapped ion systems
    TrappedIon,
    /// Photonic quantum computers
    Photonic,
    /// Neutral atom systems
    NeutralAtom,
    /// Classical simulation
    ClassicalSimulation,
}

/// Quantum machine learning algorithm types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QMLAlgorithmType {
    /// Variational Quantum Eigensolver
    VQE,
    /// Quantum Approximate Optimization Algorithm
    QAOA,
    /// Quantum Convolutional Neural Network
    QCNN,
    /// Quantum Support Vector Machine
    QSVM,
    /// Quantum Reinforcement Learning
    QRL,
    /// Quantum Generative Adversarial Network
    QGAN,
    /// Quantum Boltzmann Machine
    QBM,
}

/// Gradient estimation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GradientMethod {
    /// Parameter shift rule
    ParameterShift,
    /// Finite differences
    FiniteDifferences,
    /// Automatic differentiation
    AutomaticDifferentiation,
    /// Natural gradients
    NaturalGradients,
    /// Stochastic parameter shift
    StochasticParameterShift,
}

/// Optimization algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizerType {
    /// Adam optimizer
    Adam,
    /// Stochastic gradient descent
    SGD,
    /// RMSprop
    RMSprop,
    /// L-BFGS
    LBFGS,
    /// Quantum natural gradient
    QuantumNaturalGradient,
    /// SPSA (Simultaneous Perturbation Stochastic Approximation)
    SPSA,
}

/// QML configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QMLConfig {
    /// Target hardware architecture
    pub hardware_architecture: HardwareArchitecture,
    /// Algorithm type
    pub algorithm_type: QMLAlgorithmType,
    /// Number of qubits
    pub num_qubits: usize,
    /// Circuit depth
    pub circuit_depth: usize,
    /// Number of parameters
    pub num_parameters: usize,
    /// Gradient estimation method
    pub gradient_method: GradientMethod,
    /// Optimizer type
    pub optimizer_type: OptimizerType,
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size
    pub batch_size: usize,
    /// Maximum epochs
    pub max_epochs: usize,
    /// Convergence tolerance
    pub convergence_tolerance: f64,
    /// Enable hardware-aware optimization
    pub hardware_aware_optimization: bool,
    /// Enable noise adaptation
    pub noise_adaptive_training: bool,
    /// Shot budget for expectation value estimation
    pub shot_budget: usize,
}

impl Default for QMLConfig {
    fn default() -> Self {
        Self {
            hardware_architecture: HardwareArchitecture::NISQ,
            algorithm_type: QMLAlgorithmType::VQE,
            num_qubits: 4,
            circuit_depth: 3,
            num_parameters: 12,
            gradient_method: GradientMethod::ParameterShift,
            optimizer_type: OptimizerType::Adam,
            learning_rate: 0.01,
            batch_size: 32,
            max_epochs: 100,
            convergence_tolerance: 1e-6,
            hardware_aware_optimization: true,
            noise_adaptive_training: true,
            shot_budget: 8192,
        }
    }
}

impl QMLConfig {
    /// Create a new QML configuration for a specific algorithm type
    pub fn for_algorithm(algorithm_type: QMLAlgorithmType) -> Self {
        let mut config = Self::default();
        config.algorithm_type = algorithm_type;
        
        // Adjust default parameters based on algorithm
        match algorithm_type {
            QMLAlgorithmType::VQE => {
                config.num_qubits = 4;
                config.circuit_depth = 3;
                config.num_parameters = 12;
                config.gradient_method = GradientMethod::ParameterShift;
            }
            QMLAlgorithmType::QAOA => {
                config.num_qubits = 6;
                config.circuit_depth = 2;
                config.num_parameters = 4;
                config.gradient_method = GradientMethod::ParameterShift;
            }
            QMLAlgorithmType::QCNN => {
                config.num_qubits = 8;
                config.circuit_depth = 4;
                config.num_parameters = 24;
                config.gradient_method = GradientMethod::AutomaticDifferentiation;
            }
            QMLAlgorithmType::QSVM => {
                config.num_qubits = 6;
                config.circuit_depth = 2;
                config.num_parameters = 8;
                config.gradient_method = GradientMethod::FiniteDifferences;
            }
            QMLAlgorithmType::QRL => {
                config.num_qubits = 4;
                config.circuit_depth = 5;
                config.num_parameters = 20;
                config.gradient_method = GradientMethod::NaturalGradients;
            }
            QMLAlgorithmType::QGAN => {
                config.num_qubits = 8;
                config.circuit_depth = 6;
                config.num_parameters = 32;
                config.gradient_method = GradientMethod::AutomaticDifferentiation;
            }
            QMLAlgorithmType::QBM => {
                config.num_qubits = 10;
                config.circuit_depth = 3;
                config.num_parameters = 15;
                config.gradient_method = GradientMethod::StochasticParameterShift;
            }
        }
        
        config
    }

    /// Create a configuration optimized for a specific hardware architecture
    pub fn for_hardware(hardware: HardwareArchitecture) -> Self {
        let mut config = Self::default();
        config.hardware_architecture = hardware;
        
        // Adjust parameters based on hardware constraints
        match hardware {
            HardwareArchitecture::NISQ => {
                config.circuit_depth = 3;
                config.shot_budget = 8192;
                config.noise_adaptive_training = true;
            }
            HardwareArchitecture::FaultTolerant => {
                config.circuit_depth = 10;
                config.shot_budget = 1_000_000;
                config.noise_adaptive_training = false;
            }
            HardwareArchitecture::Superconducting => {
                config.circuit_depth = 5;
                config.shot_budget = 16384;
                config.noise_adaptive_training = true;
            }
            HardwareArchitecture::TrappedIon => {
                config.circuit_depth = 8;
                config.shot_budget = 32768;
                config.noise_adaptive_training = true;
            }
            HardwareArchitecture::Photonic => {
                config.circuit_depth = 4;
                config.shot_budget = 4096;
                config.noise_adaptive_training = true;
            }
            HardwareArchitecture::NeutralAtom => {
                config.circuit_depth = 6;
                config.shot_budget = 16384;
                config.noise_adaptive_training = true;
            }
            HardwareArchitecture::ClassicalSimulation => {
                config.circuit_depth = 15;
                config.shot_budget = 1_000_000;
                config.noise_adaptive_training = false;
            }
        }
        
        config
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.num_qubits == 0 {
            return Err("Number of qubits must be positive".to_string());
        }
        if self.circuit_depth == 0 {
            return Err("Circuit depth must be positive".to_string());
        }
        if self.num_parameters == 0 {
            return Err("Number of parameters must be positive".to_string());
        }
        if self.learning_rate <= 0.0 {
            return Err("Learning rate must be positive".to_string());
        }
        if self.batch_size == 0 {
            return Err("Batch size must be positive".to_string());
        }
        if self.max_epochs == 0 {
            return Err("Maximum epochs must be positive".to_string());
        }
        if self.convergence_tolerance <= 0.0 {
            return Err("Convergence tolerance must be positive".to_string());
        }
        if self.shot_budget == 0 {
            return Err("Shot budget must be positive".to_string());
        }
        
        Ok(())
    }
}