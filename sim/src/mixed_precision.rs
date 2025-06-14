//! Mixed-precision quantum simulation with automatic precision selection.
//!
//! This module has been refactored into a modular structure under `crate::precision`.
//! This file now serves as a compatibility layer.

// Re-export the new precision module for backwards compatibility
pub use crate::precision::*;