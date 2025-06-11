//! Bayesian Optimization for Hyperparameter Tuning
//!
//! This module implements advanced Bayesian optimization techniques for automatically
//! tuning hyperparameters in quantum annealing systems. It uses Gaussian processes
//! as surrogate models and sophisticated acquisition functions to efficiently explore
//! the hyperparameter space.
//!
//! Key features:
//! - Multi-objective Bayesian optimization
//! - Mixed parameter types (continuous, discrete, categorical)
//! - Advanced acquisition functions (EI, UCB, PI, Entropy Search)
//! - Gaussian process surrogate models with different kernels
//! - Constraint handling and feasibility modeling
//! - Transfer learning across related optimization problems
//! - Parallel and batch optimization
//! - Uncertainty quantification and confidence intervals

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::embedding::{Embedding, HardwareTopology};
use crate::hardware_compilation::{CompilerConfig, HardwareCompiler};
use crate::ising::IsingModel;
use crate::simulator::{AnnealingParams, AnnealingResult, ClassicalAnnealingSimulator};

// Module declarations
pub mod acquisition;
pub mod config;
pub mod constraints;
pub mod convergence;
pub mod gaussian_process;
pub mod multi_objective;
pub mod parallel;
pub mod transfer;

// Re-export main types for backward compatibility
pub use acquisition::*;
pub use config::*;
pub use constraints::*;
pub use convergence::*;
pub use gaussian_process::*;
pub use multi_objective::*;
pub use parallel::*;
pub use transfer::*;

// Placeholder functions to maintain API compatibility
pub fn create_annealing_parameter_space() -> ParameterSpace {
    ParameterSpace::default()
}

pub fn create_bayesian_optimizer() -> BayesianHyperoptimizer {
    BayesianHyperoptimizer::default()
}

pub fn create_custom_bayesian_optimizer() -> BayesianHyperoptimizer {
    BayesianHyperoptimizer::default()
}
