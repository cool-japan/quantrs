//! # LintFindingType - Trait Implementations
//!
//! This module contains trait implementations for `LintFindingType`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::parallel_ops_stubs::*;
use std::fmt;

use super::types::LintFindingType;

impl fmt::Display for LintFindingType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RedundantGates => write!(f, "Redundant Gates"),
            Self::InefficientDecomposition => write!(f, "Inefficient Decomposition"),
            Self::MissedFusionOpportunity => write!(f, "Missed Fusion Opportunity"),
            Self::SuboptimalGateOrder => write!(f, "Suboptimal Gate Order"),
            Self::ExcessiveCircuitDepth => write!(f, "Excessive Circuit Depth"),
            Self::QuantumAntiPattern => write!(f, "Quantum Anti-Pattern"),
            Self::UnnecessaryMeasurement => write!(f, "Unnecessary Measurement"),
            Self::EntanglementLeak => write!(f, "Entanglement Leak"),
            Self::CoherenceViolation => write!(f, "Coherence Violation"),
            Self::MissingErrorMitigation => write!(f, "Missing Error Mitigation"),
            Self::PoorQubitAllocation => write!(f, "Poor Qubit Allocation"),
            Self::InadequateParameterization => write!(f, "Inadequate Parameterization"),
            Self::LackOfModularity => write!(f, "Lack of Modularity"),
            Self::UnsupportedGateSet => write!(f, "Unsupported Gate Set"),
            Self::ConnectivityViolation => write!(f, "Connectivity Violation"),
            Self::ExceedsCoherenceTime => write!(f, "Exceeds Coherence Time"),
            Self::CalibrationMismatch => write!(f, "Calibration Mismatch"),
            Self::IncorrectAlgorithmImplementation => {
                write!(f, "Incorrect Algorithm Implementation")
            }
            Self::SuboptimalAlgorithmChoice => write!(f, "Suboptimal Algorithm Choice"),
            Self::MissingAncillaQubits => write!(f, "Missing Ancilla Qubits"),
            Self::ExcessiveQubitUsage => write!(f, "Excessive Qubit Usage"),
            Self::MemoryInefficiency => write!(f, "Memory Inefficiency"),
            Self::ParallelizationOpportunity => write!(f, "Parallelization Opportunity"),
            Self::NumericalInstability => write!(f, "Numerical Instability"),
            Self::PrecisionLoss => write!(f, "Precision Loss"),
            Self::PhaseAccumulation => write!(f, "Phase Accumulation"),
            Self::CustomRule(name) => write!(f, "Custom Rule: {name}"),
        }
    }
}
