//! Quantum Continual Learning Module
//!
//! This module implements quantum-enhanced continual learning algorithms that can
//! learn new tasks while retaining knowledge from previous tasks, leveraging quantum
//! computing principles for enhanced memory and learning capabilities.

pub mod core;
pub mod config;
pub mod memory;
pub mod strategies;
pub mod tasks;
pub mod replay;
pub mod evaluation;

// Re-export main types for backward compatibility
pub use core::*;
pub use config::*;
pub use memory::*;
pub use strategies::*;
pub use tasks::*;
pub use replay::*;
pub use evaluation::*;