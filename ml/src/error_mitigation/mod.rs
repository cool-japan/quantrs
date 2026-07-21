//! Advanced Error Mitigation for Quantum Machine Learning
//!
//! This module provides comprehensive error mitigation techniques specifically designed
//! for quantum machine learning applications, including noise-aware training,
//! error correction protocols, and adaptive mitigation strategies.
//!
//! Split into submodules to stay under the workspace's per-file line limit:
//! - [`types`]: data/configuration types (re-exported at this module's root)
//! - `mitigator`: [`QuantumMLErrorMitigator`]'s real mitigation logic

mod mitigator;
mod types;

pub use types::*;
