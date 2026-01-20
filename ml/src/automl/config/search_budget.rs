//! Search Budget Configuration
//!
//! This module contains all search budget and resource allocation configurations.

/// Search budget configuration
#[derive(Debug, Clone)]
pub struct SearchBudgetConfig {
    /// Maximum time budget in seconds
    pub max_time_seconds: f64,

    /// Maximum number of trials
    pub max_trials: usize,

    /// Maximum quantum circuit evaluations
    pub max_quantum_evaluations: usize,

    /// Early stopping configuration
    pub early_stopping: EarlyStoppingConfig,

    /// Resource budget per trial
    pub per_trial_budget: PerTrialBudget,
}

/// Early stopping configuration
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    /// Enable early stopping
    pub enabled: bool,

    /// Patience (trials without improvement)
    pub patience: usize,

    /// Minimum improvement threshold
    pub min_improvement: f64,

    /// Validation metric for early stopping
    pub validation_metric: String,
}

/// Per-trial resource budget
#[derive(Debug, Clone)]
pub struct PerTrialBudget {
    /// Maximum training time per trial
    pub max_training_time: f64,

    /// Maximum memory usage (MB)
    pub max_memory_mb: f64,

    /// Maximum quantum resources
    pub max_quantum_resources: QuantumResourceBudget,
}

/// Quantum resource budget
#[derive(Debug, Clone)]
pub struct QuantumResourceBudget {
    /// Maximum number of qubits
    pub max_qubits: usize,

    /// Maximum circuit depth
    pub max_circuit_depth: usize,

    /// Maximum number of gates
    pub max_gates: usize,

    /// Maximum coherence time usage
    pub max_coherence_time: f64,
}

impl Default for SearchBudgetConfig {
    fn default() -> Self {
        Self {
            max_time_seconds: 3600.0, // 1 hour
            max_trials: 100,
            max_quantum_evaluations: 1000,
            early_stopping: EarlyStoppingConfig::default(),
            per_trial_budget: PerTrialBudget::default(),
        }
    }
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            patience: 10,
            min_improvement: 0.01,
            validation_metric: "accuracy".to_string(),
        }
    }
}

impl Default for PerTrialBudget {
    fn default() -> Self {
        Self {
            max_training_time: 300.0, // 5 minutes
            max_memory_mb: 1024.0,    // 1 GB
            max_quantum_resources: QuantumResourceBudget::default(),
        }
    }
}

impl Default for QuantumResourceBudget {
    fn default() -> Self {
        Self {
            max_qubits: 16,
            max_circuit_depth: 20,
            max_gates: 1000,
            max_coherence_time: 100.0, // microseconds
        }
    }
}

impl SearchBudgetConfig {
    /// Quick search configuration for rapid prototyping
    pub fn quick() -> Self {
        Self {
            max_time_seconds: 300.0, // 5 minutes
            max_trials: 20,
            max_quantum_evaluations: 100,
            early_stopping: EarlyStoppingConfig {
                enabled: true,
                patience: 5,
                min_improvement: 0.05,
                validation_metric: "accuracy".to_string(),
            },
            per_trial_budget: PerTrialBudget {
                max_training_time: 60.0, // 1 minute
                max_memory_mb: 512.0,    // 512 MB
                max_quantum_resources: QuantumResourceBudget {
                    max_qubits: 8,
                    max_circuit_depth: 10,
                    max_gates: 100,
                    max_coherence_time: 50.0,
                },
            },
        }
    }

    /// Extensive search configuration for thorough exploration
    pub fn extensive() -> Self {
        Self {
            max_time_seconds: 14400.0, // 4 hours
            max_trials: 500,
            max_quantum_evaluations: 5000,
            early_stopping: EarlyStoppingConfig {
                enabled: true,
                patience: 50,
                min_improvement: 0.001,
                validation_metric: "f1_score".to_string(),
            },
            per_trial_budget: PerTrialBudget {
                max_training_time: 1800.0, // 30 minutes
                max_memory_mb: 4096.0,     // 4 GB
                max_quantum_resources: QuantumResourceBudget {
                    max_qubits: 32,
                    max_circuit_depth: 50,
                    max_gates: 5000,
                    max_coherence_time: 200.0,
                },
            },
        }
    }

    /// Production search configuration with balanced performance
    pub fn production() -> Self {
        Self {
            max_time_seconds: 7200.0, // 2 hours
            max_trials: 200,
            max_quantum_evaluations: 2000,
            early_stopping: EarlyStoppingConfig {
                enabled: true,
                patience: 20,
                min_improvement: 0.005,
                validation_metric: "f1_score".to_string(),
            },
            per_trial_budget: PerTrialBudget {
                max_training_time: 600.0, // 10 minutes
                max_memory_mb: 2048.0,    // 2 GB
                max_quantum_resources: QuantumResourceBudget {
                    max_qubits: 20,
                    max_circuit_depth: 30,
                    max_gates: 2000,
                    max_coherence_time: 150.0,
                },
            },
        }
    }

    /// Research configuration for exploring new algorithms
    pub fn research() -> Self {
        Self {
            max_time_seconds: 28800.0, // 8 hours
            max_trials: 1000,
            max_quantum_evaluations: 10000,
            early_stopping: EarlyStoppingConfig {
                enabled: false, // No early stopping for research
                patience: 100,
                min_improvement: 0.0001,
                validation_metric: "quantum_advantage".to_string(),
            },
            per_trial_budget: PerTrialBudget {
                max_training_time: 3600.0, // 1 hour
                max_memory_mb: 8192.0,     // 8 GB
                max_quantum_resources: QuantumResourceBudget {
                    max_qubits: 64,
                    max_circuit_depth: 100,
                    max_gates: 10000,
                    max_coherence_time: 500.0,
                },
            },
        }
    }
}
