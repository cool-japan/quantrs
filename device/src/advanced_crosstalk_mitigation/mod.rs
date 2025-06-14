//! Advanced Cross-talk Characterization and Mitigation with ML and Real-time Adaptation
//!
//! This module extends the basic crosstalk analysis with machine learning-driven
//! predictive modeling, real-time adaptive mitigation, and advanced SciRS2 signal
//! processing for comprehensive crosstalk management in quantum systems.

pub mod config;
pub mod types;
pub mod core;
pub mod ml;
pub mod prediction;
pub mod signal_processing;
pub mod adaptive_compensation;
pub mod monitoring;
pub mod multilevel;
pub mod utils;

// Re-export all public types
pub use config::*;
pub use types::*;
pub use core::*;
pub use ml::*;
pub use prediction::*;
pub use signal_processing::*;
pub use adaptive_compensation::*;
pub use monitoring::*;
pub use multilevel::*;
pub use utils::*;