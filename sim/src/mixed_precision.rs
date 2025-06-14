//! Mixed-precision quantum simulation with automatic precision selection.
//!
//! This module has been refactored into a modular structure under `crate::mixed_precision_impl`.
//! This file now serves as a compatibility layer.

// Re-export the new mixed precision implementation module for backwards compatibility
pub use crate::mixed_precision_impl::*;