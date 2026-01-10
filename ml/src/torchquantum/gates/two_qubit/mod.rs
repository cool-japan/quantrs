//! Two-Qubit Gates Module
//!
//! This module provides two-qubit quantum gates for TorchQuantum.
//!
//! ## Submodules
//!
//! - `standard`: Standard two-qubit gates (CNOT, CZ, SWAP)
//! - `rotation`: Parameterized rotation gates (RXX, RYY, RZZ, RZX)
//! - `controlled`: Controlled rotation gates (CRX, CRY, CRZ)
//! - `special`: Special two-qubit gates (iSWAP, ECR, CY, etc.)

mod controlled;
mod rotation;
mod special;
mod standard;

pub use controlled::*;
pub use rotation::*;
pub use special::*;
pub use standard::*;
