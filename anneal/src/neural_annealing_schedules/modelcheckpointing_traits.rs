//! # ModelCheckpointing - Trait Implementations
//!
//! This module contains trait implementations for `ModelCheckpointing`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::ModelCheckpointing;

impl Default for ModelCheckpointing {
    fn default() -> Self {
        Self {
            save_frequency: 10,
            best_model_path: "best_model.pt".to_string(),
            checkpoint_history: Vec::new(),
        }
    }
}
