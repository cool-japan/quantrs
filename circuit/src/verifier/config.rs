//! Configuration types for the quantum circuit verifier

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Verification configuration options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierConfig {
    /// Enable property verification
    pub enable_property_verification: bool,
    /// Enable invariant checking
    pub enable_invariant_checking: bool,
    /// Enable theorem proving
    pub enable_theorem_proving: bool,
    /// Enable model checking
    pub enable_model_checking: bool,
    /// Enable symbolic execution
    pub enable_symbolic_execution: bool,
    /// Maximum verification depth
    pub max_verification_depth: usize,
    /// Timeout for verification tasks
    pub verification_timeout: Duration,
    /// Precision level for numerical verification
    pub numerical_precision: f64,
    /// Enable statistical verification
    pub enable_statistical_verification: bool,
    /// Confidence level for statistical tests
    pub confidence_level: f64,
    /// Maximum number of samples for statistical verification
    pub max_samples: usize,
    /// Enable parallel verification
    pub enable_parallel_verification: bool,
}

impl Default for VerifierConfig {
    fn default() -> Self {
        Self {
            enable_property_verification: true,
            enable_invariant_checking: true,
            enable_theorem_proving: true,
            enable_model_checking: true,
            enable_symbolic_execution: true,
            max_verification_depth: 1000,
            verification_timeout: Duration::from_secs(300),
            numerical_precision: 1e-12,
            enable_statistical_verification: true,
            confidence_level: 0.99,
            max_samples: 10000,
            enable_parallel_verification: true,
        }
    }
}
