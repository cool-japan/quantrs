//! # QuantumContinuousFlow - integrate_quantum_ode_group Methods
//!
//! This module contains method implementations for `QuantumContinuousFlow`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;
use scirs2_core::{Complex32, Complex64};

use super::types::{ClassicalDynamics, QuantumDynamics, QuantumFlowState, QuantumODEFunction};

use super::quantumcontinuousflow_type::QuantumContinuousFlow;

impl QuantumContinuousFlow {
    /// Integrate quantum ODE
    pub(super) fn integrate_quantum_ode(
        &self,
        initial_state: &QuantumFlowState,
        ode_func: &QuantumODEFunction,
        integration_time: f64,
    ) -> Result<QuantumFlowState> {
        let num_steps = 100;
        let dt = integration_time / num_steps as f64;
        let mut state = initial_state.clone();
        for step in 0..num_steps {
            let current_time = step as f64 * dt;
            state = self.apply_quantum_dynamics(&ode_func.quantum_dynamics, &state, dt)?;
            let classical_contribution =
                self.apply_classical_dynamics(&ode_func.classical_dynamics, &state, dt)?;
            state = self.apply_hybrid_coupling(
                &ode_func.hybrid_coupling,
                &state,
                &classical_contribution,
                dt,
            )?;
            state.coherence_time *=
                (-dt / ode_func.quantum_dynamics.decoherence_model.t2_time).exp();
            state.fidelity *=
                (1.0 - ode_func.quantum_dynamics.decoherence_model.gate_error_rate * dt);
        }
        Ok(state)
    }
    /// Apply quantum dynamics
    fn apply_quantum_dynamics(
        &self,
        dynamics: &QuantumDynamics,
        state: &QuantumFlowState,
        dt: f64,
    ) -> Result<QuantumFlowState> {
        let mut new_state = state.clone();
        for i in 0..new_state.amplitudes.len() {
            let energy = dynamics.hamiltonian[[
                i % dynamics.hamiltonian.nrows(),
                i % dynamics.hamiltonian.ncols(),
            ]];
            let time_evolution = Complex64::from_polar(1.0, -energy.re * dt);
            new_state.amplitudes[i] *= time_evolution;
            new_state.phases[i] *= time_evolution;
        }
        new_state.entanglement_measure = (new_state.entanglement_measure * 1.01).min(1.0);
        Ok(new_state)
    }
    /// Apply classical dynamics
    fn apply_classical_dynamics(
        &self,
        dynamics: &ClassicalDynamics,
        state: &QuantumFlowState,
        dt: f64,
    ) -> Result<Array1<f64>> {
        let classical_data = state.amplitudes.mapv(|amp| amp.re);
        let mut output = classical_data;
        for layer in &dynamics.dynamics_network {
            output = self.apply_classical_flow_layer(layer, &output)?;
        }
        Ok(output * dt)
    }
}
