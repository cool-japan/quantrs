//! Core types and traits for the QuantRS2 quantum computing framework.
//!
//! This crate provides the foundational types and traits used throughout
//! the QuantRS2 ecosystem, including qubit identifiers, quantum gates,
//! and register representations.

pub mod decomposition;
pub mod error;
pub mod gate;
pub mod parametric;
pub mod qubit;
pub mod register;

/// Re-exports of commonly used types and traits
pub mod prelude {
    pub use crate::decomposition::*;
    pub use crate::error::*;
    pub use crate::gate::*;
    pub use crate::parametric::*;
    pub use crate::qubit::*;
    pub use crate::register::*;
}
