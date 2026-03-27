//! # TopologicalQuantumSimulator - fuse_anyons_group Methods
//!
//! This module contains method implementations for `TopologicalQuantumSimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};

use super::types::AnyonType;

use super::topologicalquantumsimulator_type::TopologicalQuantumSimulator;

impl TopologicalQuantumSimulator {
    /// Fuse two anyons
    pub fn fuse_anyons(&mut self, anyon_a: usize, anyon_b: usize) -> Result<Vec<AnyonType>> {
        if anyon_a >= self.state.anyon_config.anyons.len()
            || anyon_b >= self.state.anyon_config.anyons.len()
        {
            return Err(SimulatorError::InvalidInput(
                "Invalid anyon indices".to_string(),
            ));
        }
        let type_a = &self.state.anyon_config.anyons[anyon_a].1;
        let type_b = &self.state.anyon_config.anyons[anyon_b].1;
        let fusion_outcomes = if let Some(outcomes) = type_a.fusion_rules.get(&type_b.label) {
            outcomes.clone()
        } else {
            vec!["vacuum".to_string()]
        };
        let outcome_types: Vec<AnyonType> = fusion_outcomes
            .iter()
            .map(|label| self.create_anyon_from_label(label))
            .collect::<Result<Vec<_>>>()?;
        let mut indices_to_remove = vec![anyon_a, anyon_b];
        indices_to_remove.sort_by(|a, b| b.cmp(a));
        for &index in &indices_to_remove {
            if index < self.state.anyon_config.anyons.len() {
                self.state.anyon_config.anyons.remove(index);
            }
            if index < self.state.anyon_config.worldlines.len() {
                self.state.anyon_config.worldlines.remove(index);
            }
        }
        Ok(outcome_types)
    }
}
