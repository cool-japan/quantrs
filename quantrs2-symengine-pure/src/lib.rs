// Allow HashMap without generic hasher for simpler API
#![allow(clippy::implicit_hasher)]

//! # QuantRS2 SymEngine Pure
//!
//! A pure Rust symbolic mathematics library for quantum computing.
//!
//! This crate provides symbolic computation capabilities without any C/C++ dependencies,
//! using [egg](https://egraphs-good.github.io/) for e-graph based simplification and
//! optimization.
//!
//! ## Features
//!
//! - **Pure Rust**: No C/C++ dependencies, fully portable
//! - **Symbolic Expressions**: Create and manipulate symbolic mathematical expressions
//! - **Automatic Differentiation**: Compute symbolic gradients and Hessians
//! - **E-Graph Optimization**: Advanced expression simplification via equality saturation
//! - **Quantum Computing**: Specialized support for quantum gates, operators, and states
//! - **SciRS2 Integration**: Seamless integration with the SciRS2 scientific computing ecosystem
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use quantrs2_symengine_pure::Expression;
//!
//! // Create symbolic expressions
//! let x = Expression::symbol("x");
//! let y = Expression::symbol("y");
//!
//! // Perform operations
//! let expr = x.clone() * x.clone() + x.clone() * 2.0 * y.clone() + y.clone() * y.clone();
//! let expanded = expr.expand();
//!
//! // Compute derivatives
//! let dx = expr.diff(&x);
//!
//! println!("Expression: {}", expr);
//! println!("Derivative wrt x: {}", dx);
//! ```
//!
//! ## Policy Compliance
//!
//! This crate follows the QuantRS2 policies:
//! - **Pure Rust Policy**: No C/C++/Fortran dependencies
//! - **SciRS2 Policy**: Uses `scirs2_core` for complex numbers, arrays, and random generation
//! - **COOLJAPAN Policy**: Uses `oxicode` for serialization (not bincode)
//! - **No unwrap Policy**: All fallible operations return Result types

pub mod cache;
pub mod diff;
pub mod error;
pub mod eval;
pub mod expr;
pub mod matrix;
pub mod ops;
pub mod optimization;
pub mod parser;
pub mod pattern;
pub mod quantum;
pub mod scirs2_bridge;
pub mod serialize;
pub mod simplify;

// Re-export main types
pub use error::{SymEngineError, SymEngineResult};
pub use expr::Expression;
pub use matrix::SymbolicMatrix;

// Re-export SciRS2 types for convenience (following SciRS2 POLICY)
pub use scirs2_core::Complex64;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Check if the library is available (always true for pure Rust implementation)
#[must_use]
pub const fn is_available() -> bool {
    true
}

/// Check if this is the pure Rust implementation
#[must_use]
pub const fn is_pure_rust() -> bool {
    true
}
