//! # QuantumGravitySimulator - simulate_group Methods
//!
//! This module contains method implementations for `QuantumGravitySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;

use super::types::{GravityApproach, GravitySimulationResult};

use super::quantumgravitysimulator_type::QuantumGravitySimulator;

impl QuantumGravitySimulator {
    /// Run quantum gravity simulation
    pub fn simulate(&mut self) -> Result<GravitySimulationResult> {
        let start_time = std::time::Instant::now();
        match self.config.gravity_approach {
            GravityApproach::LoopQuantumGravity => {
                self.initialize_spacetime()?;
                self.initialize_lqg_spin_network()?;
                self.simulate_lqg()?;
            }
            GravityApproach::CausalDynamicalTriangulation => {
                self.initialize_spacetime()?;
                self.initialize_cdt()?;
                self.simulate_cdt()?;
            }
            GravityApproach::AsymptoticSafety => {
                self.initialize_asymptotic_safety()?;
                self.simulate_asymptotic_safety()?;
            }
            GravityApproach::HolographicGravity => {
                self.initialize_ads_cft()?;
                self.simulate_ads_cft()?;
            }
            _ => {
                return Err(SimulatorError::InvalidConfiguration(format!(
                    "Gravity approach {:?} not yet implemented",
                    self.config.gravity_approach
                )));
            }
        }
        let computation_time = start_time.elapsed().as_secs_f64();
        self.stats.total_time += computation_time;
        self.stats.calculations_performed += 1;
        self.stats.avg_time_per_step =
            self.stats.total_time / self.stats.calculations_performed as f64;
        self.simulation_history.last().cloned().ok_or_else(|| {
            SimulatorError::InvalidConfiguration("No simulation results available".to_string())
        })
    }
}
