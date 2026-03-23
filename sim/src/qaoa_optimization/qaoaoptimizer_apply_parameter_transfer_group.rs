//! # QAOAOptimizer - apply_parameter_transfer_group Methods
//!
//! This module contains method implementations for `QAOAOptimizer`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use scirs2_core::random::prelude::*;

use super::types::ProblemCharacteristics;

use super::qaoaoptimizer_type::QAOAOptimizer;

impl QAOAOptimizer {
    /// Helper methods
    pub(super) fn apply_parameter_transfer(&mut self) -> Result<()> {
        let characteristics = self.extract_problem_characteristics()?;
        let database = self.parameter_database.lock().map_err(|e| {
            crate::error::SimulatorError::InvalidInput(format!(
                "Failed to lock parameter database: {}",
                e
            ))
        })?;
        if let Some(similar_params) = database.parameters.get(&characteristics) {
            if let Some((gammas, betas, _cost)) = similar_params.first() {
                self.gammas = gammas.clone();
                self.betas = betas.clone();
            }
        }
        Ok(())
    }
    pub(super) fn extract_problem_characteristics(&self) -> Result<ProblemCharacteristics> {
        let num_edges = self
            .graph
            .adjacency_matrix
            .iter()
            .map(|&x| usize::from(x.abs() > 1e-10))
            .sum::<usize>();
        let max_edges = self.graph.num_vertices * (self.graph.num_vertices - 1) / 2;
        let density = if max_edges > 0 {
            (100.0 * num_edges as f64 / max_edges as f64) as u32
        } else {
            0
        };
        Ok(ProblemCharacteristics {
            problem_type: self.problem_type,
            num_vertices: self.graph.num_vertices,
            density,
            regularity: 50,
        })
    }
}
