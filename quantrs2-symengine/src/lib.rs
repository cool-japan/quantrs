//! # `SymEngine` for Rust
//!
//! Symbolic computation in Rust, powered by [SymEngine](https://github.com/symengine/symengine).
//!
//! This crate provides safe, idiomatic Rust bindings to `SymEngine`, a fast symbolic manipulation
//! library written in C++. It allows you to perform symbolic mathematics operations such as
//! algebraic manipulation, calculus, equation solving, and more.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use quantrs2_symengine::Expression;
//!
//! // Create symbolic expressions
//! let x = Expression::symbol("x");
//! let y = Expression::symbol("y");
//!
//! // Perform operations
//! let expr = x.clone() * x.clone() + x.clone() * 2 * y.clone() + y.clone() * y.clone();
//! let expanded = expr.expand();
//!
//! println!("Expression: {}", expr);
//! println!("Expanded: {}", expanded);
//! ```
//!
//! ## Features
//!
//! - **Fast**: Built on `SymEngine`'s optimized C++ core
//! - **Safe**: Memory-safe Rust interface with proper error handling
//! - **Feature-rich**: Supports algebraic operations, calculus, equation solving
//! - **Serializable**: Optional serde support for persistence
//! - **Cross-platform**: Works on Linux, macOS, and Windows
//!
//! ## Optional Features
//!
//! - `serde-serialize`: Enable serialization/deserialization support
//! - `static`: Link `SymEngine` statically
//! - `system-deps`: Use system-installed `SymEngine` via pkg-config

pub mod cache;
pub mod error;
pub mod expr;
// pub mod map;  // Disabled: Missing C wrapper functions (mapbasicbasic_free, etc.) in symengine-sys
// TODO: Add C wrappers to quantrs2-symengine-sys or wait for upstream SymEngine support
pub mod ndarray_integration;
pub mod ops;
pub mod polynomial;
pub mod quantum;
pub mod scirs2_integration;
pub mod simd_eval;
pub mod solving;

pub use error::{SymEngineError, SymEngineResult};
pub use expr::Expression;
// pub use map::ExprMap;  // Disabled: See map module comment above

// Reexport symengine_sys for advanced users
pub use quantrs2_symengine_sys as symengine_sys;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Check if `SymEngine` is properly initialized and available
#[must_use]
pub const fn is_available() -> bool {
    // For now, assume it's available if we can link
    true
}
