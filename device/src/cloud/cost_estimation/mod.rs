//! Cost estimation module for quantum cloud services.
//!
//! Split from cost_estimation.rs (1903 lines) into sub-modules for size compliance.
//! - `definitions`: All struct and enum type definitions (~1593 lines)
//! - `implementations`: All impl blocks (~310 lines)

pub mod definitions;
pub mod implementations;

// Re-export everything so existing `use super::cost_estimation::*` still works transparently
pub use definitions::*;
pub use implementations::*;
