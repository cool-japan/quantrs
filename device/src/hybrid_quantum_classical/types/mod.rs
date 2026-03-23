//! Types module for hybrid quantum-classical execution.
//!
//! Split from types.rs (1932 lines) into sub-modules for size compliance.
//! - `definitions`: All struct and enum type definitions (~1281 lines)
//! - `implementations`: All impl blocks (~653 lines)

pub mod definitions;
pub mod implementations;

// Re-export everything so existing `use super::types::*` still works transparently
pub use definitions::*;
pub use implementations::*;
