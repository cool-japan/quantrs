//! # TopologicalQuantumSimulator - move_anyon_group Methods
//!
//! This module contains method implementations for `TopologicalQuantumSimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};

use super::topologicalquantumsimulator_type::TopologicalQuantumSimulator;

impl TopologicalQuantumSimulator {
    /// Move anyon to new position
    pub fn move_anyon(&mut self, anyon_id: usize, new_position: Vec<usize>) -> Result<()> {
        if anyon_id >= self.state.anyon_config.anyons.len() {
            return Err(SimulatorError::InvalidInput("Invalid anyon ID".to_string()));
        }
        for (i, &pos) in new_position.iter().enumerate() {
            if pos >= self.config.dimensions[i] {
                return Err(SimulatorError::InvalidInput(
                    "New position out of bounds".to_string(),
                ));
            }
        }
        self.state.anyon_config.anyons[anyon_id].0 = new_position.clone();
        if anyon_id < self.state.anyon_config.worldlines.len() {
            self.state.anyon_config.worldlines[anyon_id]
                .path
                .push(new_position);
            let current_time = self.braiding_history.len() as f64;
            self.state.anyon_config.worldlines[anyon_id]
                .time_stamps
                .push(current_time);
        }
        Ok(())
    }
}
