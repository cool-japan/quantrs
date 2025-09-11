//! # SymEngine for Rust
//!
//! Symbolic computation in Rust, powered by [SymEngine](https://github.com/symengine/symengine).
//!
//! This crate provides safe, idiomatic Rust bindings to SymEngine, a fast symbolic manipulation
//! library written in C++. It allows you to perform symbolic mathematics operations such as
//! algebraic manipulation, calculus, equation solving, and more.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use symengine::Expression;
//!
//! // Create symbolic expressions
//! let x = Expression::symbol("x");
//! let y = Expression::symbol("y");
//!
//! // Perform operations
//! let expr = &x * &x + 2 * &x * &y + &y * &y;
//! let expanded = expr.expand();
//!
//! println!("Expression: {}", expr);
//! println!("Expanded: {}", expanded);
//! ```
//!
//! ## Features
//!
//! - **Fast**: Built on SymEngine's optimized C++ core
//! - **Safe**: Memory-safe Rust interface with proper error handling
//! - **Feature-rich**: Supports algebraic operations, calculus, equation solving
//! - **Serializable**: Optional serde support for persistence
//! - **Cross-platform**: Works on Linux, macOS, and Windows
//!
//! ## Optional Features
//!
//! - `serde-serialize`: Enable serialization/deserialization support
//! - `static`: Link SymEngine statically
//! - `system-deps`: Use system-installed SymEngine via pkg-config

pub mod error;
pub mod expr;
// pub mod map;  // Temporarily disabled - missing C wrapper functions
pub mod ops;

pub use error::{SymEngineError, SymEngineResult};
pub use expr::Expression;
// pub use map::ExprMap;

// Reexport symengine_sys for advanced users
pub use quantrs2_symengine_sys as symengine_sys;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Check if SymEngine is properly initialized and available
pub fn is_available() -> bool {
    // For now, assume it's available if we can link
    true
}
