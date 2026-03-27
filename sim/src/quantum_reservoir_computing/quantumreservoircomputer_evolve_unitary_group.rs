//! # QuantumReservoirComputer - evolve_unitary_group Methods
//!
//! This module contains method implementations for `QuantumReservoirComputer`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use scirs2_core::random::prelude::*;

use super::quantumreservoircomputer_type::QuantumReservoirComputer;

impl QuantumReservoirComputer {
    /// Unitary evolution
    pub(super) fn evolve_unitary(&mut self) -> Result<()> {
        self.simulator
            .apply_interface_circuit(&self.reservoir_circuit)?;
        Ok(())
    }
    /// Open system evolution with noise
    pub(super) fn evolve_open_system(&mut self) -> Result<()> {
        self.evolve_unitary()?;
        self.apply_decoherence()?;
        Ok(())
    }
    /// NISQ evolution with realistic noise
    pub(super) fn evolve_nisq(&mut self) -> Result<()> {
        self.evolve_unitary()?;
        self.apply_gate_errors()?;
        self.apply_measurement_errors()?;
        Ok(())
    }
    /// Adiabatic evolution
    pub(super) fn evolve_adiabatic(&mut self) -> Result<()> {
        self.evolve_unitary()?;
        Ok(())
    }
}
