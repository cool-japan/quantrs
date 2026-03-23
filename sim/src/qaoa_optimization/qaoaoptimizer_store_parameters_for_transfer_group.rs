//! # QAOAOptimizer - store_parameters_for_transfer_group Methods
//!
//! This module contains method implementations for `QAOAOptimizer`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use scirs2_core::random::prelude::*;

use super::qaoaoptimizer_type::QAOAOptimizer;

impl QAOAOptimizer {
    pub(super) fn store_parameters_for_transfer(&self) -> Result<()> {
        let characteristics = self.extract_problem_characteristics()?;
        let mut database = self.parameter_database.lock().map_err(|e| {
            crate::error::SimulatorError::InvalidInput(format!(
                "Failed to lock parameter database: {}",
                e
            ))
        })?;
        let entry = database.parameters.entry(characteristics).or_default();
        entry.push((
            self.best_gammas.clone(),
            self.best_betas.clone(),
            self.best_cost,
        ));
        entry.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        entry.truncate(5);
        Ok(())
    }
}
