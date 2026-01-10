//! Quantum Machine Learning Layers Framework
//!
//! This module provides a comprehensive implementation of quantum machine learning layers,
//! including parameterized quantum circuits, quantum convolutional layers, quantum recurrent
//! networks, and hybrid classical-quantum training algorithms.

mod config;
mod framework;
mod layers;
mod types;
mod utils;

pub use config::*;
pub use framework::*;
pub use layers::*;
pub use types::*;
pub use utils::*;
