//! Search Module
//!
//! This module contains search algorithms, hyperparameter optimization,
//! and model selection functionality.

pub mod hyperparameter_optimizer;
pub mod model_selector;
pub mod search_history;

pub use hyperparameter_optimizer::QuantumHyperparameterOptimizer;
pub use model_selector::QuantumModelSelector;
pub use search_history::SearchHistory;

use crate::error::Result;
use std::collections::HashMap;
