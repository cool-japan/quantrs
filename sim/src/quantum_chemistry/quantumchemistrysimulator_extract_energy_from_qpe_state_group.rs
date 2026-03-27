//! # QuantumChemistrySimulator - extract_energy_from_qpe_state_group Methods
//!
//! This module contains method implementations for `QuantumChemistrySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::ndarray::{Array1, Array2, Array4};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;
use std::f64::consts::PI;

use super::quantumchemistrysimulator_type::QuantumChemistrySimulator;

impl QuantumChemistrySimulator {
    /// Extract energy from quantum phase estimation measurement
    pub(super) fn extract_energy_from_qpe_state(
        &self,
        state: &Array1<Complex64>,
        ancilla_qubits: usize,
    ) -> Result<f64> {
        let ancilla_states = 1 << ancilla_qubits;
        let system_size = state.len() / ancilla_states;
        let mut max_prob = 0.0;
        let mut most_likely_phase = 0;
        for phase_int in 0..ancilla_states {
            let mut prob = 0.0;
            for sys_state in 0..system_size {
                let idx = phase_int * system_size + sys_state;
                if idx < state.len() {
                    prob += state[idx].norm_sqr();
                }
            }
            if prob > max_prob {
                max_prob = prob;
                most_likely_phase = phase_int;
            }
        }
        let phase = most_likely_phase as f64 / ancilla_states as f64;
        let energy = phase * 2.0 * PI;
        Ok(energy)
    }
}
