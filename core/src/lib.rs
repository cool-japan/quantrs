//! Core types and traits for the quantrs quantum computing framework.
//!
//! This crate provides the foundational types and traits used throughout
//! the quantrs ecosystem, including qubit identifiers, quantum gates,
//! and register representations.

pub mod error;
pub mod gate;
pub mod qubit;
pub mod register;

/// Re-exports of commonly used types and traits
pub mod prelude {
    pub use crate::error::*;
    pub use crate::gate::*;
    pub use crate::qubit::*;
    pub use crate::register::*;
}
