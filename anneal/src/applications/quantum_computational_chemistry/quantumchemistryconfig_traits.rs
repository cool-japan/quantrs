//! # QuantumChemistryConfig - Trait Implementations
//!
//! This module contains trait implementations for `QuantumChemistryConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::advanced_quantum_algorithms::{
    AdiabaticShortcutsOptimizer, AdvancedAlgorithmConfig, AdvancedQuantumAlgorithms,
    AlgorithmSelectionStrategy, InfiniteDepthQAOA, InfiniteQAOAConfig, QuantumZenoAnnealer,
    ShortcutsConfig, ZenoConfig,
};
use crate::quantum_error_correction::{
    ErrorCorrectionCode, ErrorMitigationConfig, ErrorMitigationManager, LogicalAnnealingEncoder,
    NoiseResilientAnnealingProtocol, SyndromeDetector,
};

use super::types::{
    BasisSet, ConvergenceCriteria, CorrelationMethod, ElectronicStructureMethod,
    QuantumChemistryConfig,
};

impl Default for QuantumChemistryConfig {
    fn default() -> Self {
        Self {
            method: ElectronicStructureMethod::HartreeFock,
            basis_set: BasisSet::STO3G,
            correlation: CorrelationMethod::CCSD,
            convergence: ConvergenceCriteria::default(),
            error_correction: ErrorMitigationConfig::default(),
            advanced_algorithms: AdvancedAlgorithmConfig::default(),
            monitoring_enabled: true,
        }
    }
}
