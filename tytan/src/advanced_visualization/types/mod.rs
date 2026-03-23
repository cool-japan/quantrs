//! Types module for advanced visualization.
//!
//! Split from types.rs (1938 lines) into sub-modules for size compliance.
//! - `definitions`: All struct and enum type definitions
//! - `implementations`: All impl blocks for those types

pub mod definitions;
pub mod implementations;

// Re-export everything so existing `use super::types::*` still works transparently
pub use definitions::*;
pub use implementations::*;
