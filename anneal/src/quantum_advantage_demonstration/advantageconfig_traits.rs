//! # AdvantageConfig - Trait Implementations
//!
//! This module contains trait implementations for `AdvantageConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;
use std::time::{Duration, Instant};

use super::types::*;

impl Default for AdvantageConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            num_repetitions: 100,
            problem_size_range: (10, 5000),
            time_limit: Duration::from_secs(3600),
            classical_algorithms: vec![
                ClassicalAlgorithm::SimulatedAnnealing,
                ClassicalAlgorithm::TabuSearch,
                ClassicalAlgorithm::GeneticAlgorithm,
                ClassicalAlgorithm::ParticleSwarmOptimization,
                ClassicalAlgorithm::BranchAndBound,
            ],
            quantum_devices: vec![
                QuantumDevice::DWaveAdvantage,
                QuantumDevice::AWSBraket,
                QuantumDevice::Simulator,
            ],
            advantage_metrics: vec![
                AdvantageMetric::TimeToSolution,
                AdvantageMetric::SolutionQuality,
                AdvantageMetric::EnergyConsumption,
                AdvantageMetric::CostEfficiency,
                AdvantageMetric::Scalability,
            ],
            problem_categories: vec![
                ProblemCategory::Optimization,
                ProblemCategory::Sampling,
                ProblemCategory::ConstraintSatisfaction,
                ProblemCategory::MachineLearning,
                ProblemCategory::ScientificComputing,
            ],
        }
    }
}
