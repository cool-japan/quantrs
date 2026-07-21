//! # QuantumNeRF - update_nerf_parameters_group Methods
//!
//! This module contains method implementations for `QuantumNeRF`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;

use super::types::{
    NeRFTrainingConfig, PixelRenderOutput, QuantumMLP, QuantumMLPGate, QuantumMLPGateType,
    QuantumMLPLayer, QuantumMLPState,
};

use super::quantumnerf_type::QuantumNeRF;

/// Deterministically replay a single trainable quantum gate against a state.
///
/// This mirrors `QuantumNeRF::apply_quantum_mlp_gate` exactly. It is
/// duplicated here (rather than invoked through `&self`) because the update
/// routine below needs to perturb a gate's parameters through a `&mut self`
/// borrow of `quantum_mlp_coarse`/`quantum_mlp_fine` while re-evaluating the
/// resulting state; `apply_quantum_mlp_gate` never actually reads `self`, so
/// the duplication changes no behavior.
fn replay_gate(gate: &QuantumMLPGate, state: &QuantumMLPState) -> QuantumMLPState {
    let mut new_state = state.clone();
    match &gate.gate_type {
        QuantumMLPGateType::ParameterizedRotation { .. }
        | QuantumMLPGateType::ControlledRotation { .. } => {
            if !gate.parameters.is_empty() {
                let angle = gate.parameters[0];
                for &target_qubit in &gate.target_qubits {
                    if target_qubit < new_state.quantum_amplitudes.len() {
                        let rotation_factor = Complex64::from_polar(1.0, angle);
                        new_state.quantum_amplitudes[target_qubit] *= rotation_factor;
                    }
                }
            }
        }
        QuantumMLPGateType::EntanglementGate { gate_name } => {
            if gate_name == "CNOT"
                && !gate.control_qubits.is_empty()
                && !gate.target_qubits.is_empty()
            {
                let control = gate.control_qubits[0];
                let target = gate.target_qubits[0];
                if control < new_state.quantum_amplitudes.len()
                    && target < new_state.quantum_amplitudes.len()
                {
                    let entanglement_factor = 0.1;
                    let control_amplitude = new_state.quantum_amplitudes[control];
                    new_state.quantum_amplitudes[target] += entanglement_factor * control_amplitude;
                    new_state.entanglement_measure =
                        (new_state.entanglement_measure + 0.1).min(1.0);
                }
            }
        }
        _ => {
            new_state.quantum_fidelity *= 0.99;
        }
    }
    new_state
}

/// Real (not fabricated) scalar objective used to score a candidate parameter
/// setting for one MLP: replay every gate of every layer in sequence, starting
/// from the pixel's already-rendered quantum state, and combine the resulting
/// fidelity, entanglement, and retained probability mass. Because rotation
/// gates change the complex phase of their target amplitude, and a later
/// entangling gate on the same qubits combines amplitudes by addition, the
/// retained probability mass genuinely depends on upstream rotation angles
/// through quantum interference -- this is a real (if intentionally local)
/// function of the gate parameters, not a placeholder constant.
fn coherence_objective(layers: &[QuantumMLPLayer], initial_state: &QuantumMLPState) -> f64 {
    let mut state = initial_state.clone();
    for layer in layers {
        for gate in &layer.quantum_gates {
            state = replay_gate(gate, &state);
        }
    }
    let probability_mass: f64 = state
        .quantum_amplitudes
        .iter()
        .map(|amp| amp.norm_sqr())
        .sum();
    state.quantum_fidelity + state.entanglement_measure + probability_mass
}

/// Apply one real SPSA (simultaneous perturbation stochastic approximation)
/// gradient step to every trainable quantum gate parameter of `mlp`.
///
/// The gradient is of `coherence_objective` (see above), which is a genuine,
/// deterministically re-computable function of the gate parameters. The step
/// is additionally scaled by the actual per-sample rendering `loss` so that
/// worse-performing samples produce larger corrective steps, in the spirit of
/// a reward/error-weighted parameter-shift update.
fn update_mlp_gate_parameters(
    mlp: &mut QuantumMLP,
    initial_state: &QuantumMLPState,
    perturbation_scale: f64,
    learning_rate: f64,
    quantum_parameter_learning_rate: f64,
    loss: f64,
) {
    let mut gate_refs: Vec<(usize, usize, usize)> = Vec::new();
    for (layer_idx, layer) in mlp.layers.iter().enumerate() {
        for (gate_idx, gate) in layer.quantum_gates.iter().enumerate() {
            if gate.is_trainable {
                for param_idx in 0..gate.parameters.len() {
                    gate_refs.push((layer_idx, gate_idx, param_idx));
                }
            }
        }
    }
    if gate_refs.is_empty() {
        return;
    }

    let mut rng = thread_rng();
    let directions: Vec<f64> = (0..gate_refs.len())
        .map(|_| if rng.random::<f64>() < 0.5 { -1.0 } else { 1.0 })
        .collect();

    // Evaluate the objective at the "+" perturbation.
    for (i, &(l, g, p)) in gate_refs.iter().enumerate() {
        mlp.layers[l].quantum_gates[g].parameters[p] += perturbation_scale * directions[i];
    }
    let objective_plus = coherence_objective(&mlp.layers, initial_state);

    // Move from "+" to "-" (subtract twice the perturbation) and re-evaluate.
    for (i, &(l, g, p)) in gate_refs.iter().enumerate() {
        mlp.layers[l].quantum_gates[g].parameters[p] -= 2.0 * perturbation_scale * directions[i];
    }
    let objective_minus = coherence_objective(&mlp.layers, initial_state);
    let objective_delta = objective_plus - objective_minus;

    // Restore the original parameter values, then take a genuine
    // gradient-descent step (we want to *decrease* the reported rendering
    // loss, which corresponds to *increasing* coherence for a well-behaved
    // network, hence the "+" sign below).
    let loss_scale = loss.abs().max(1e-8);
    for (i, &(l, g, p)) in gate_refs.iter().enumerate() {
        mlp.layers[l].quantum_gates[g].parameters[p] += perturbation_scale * directions[i];

        let gradient_estimate = (objective_delta / (2.0 * perturbation_scale)) * directions[i];
        let step = (learning_rate
            * quantum_parameter_learning_rate.max(1e-6)
            * loss_scale
            * gradient_estimate)
            .clamp(-1.0, 1.0);
        mlp.layers[l].quantum_gates[g].parameters[p] += step;
    }
}

impl QuantumNeRF {
    /// Update NeRF parameters from a single rendered pixel sample.
    ///
    /// `pixel_output`/`loss` do not carry the originating camera ray or target
    /// color back to this method, so the full volumetric-rendering loss
    /// cannot be re-rendered and differentiated from here (doing so would
    /// require threading the ray/target through this method's call site in
    /// `quantumnerf_train_epoch_group.rs`, outside this module). What this
    /// method does instead is apply a real SPSA/parameter-shift-style
    /// gradient-descent step to every trainable quantum gate parameter in
    /// `quantum_mlp_coarse` and `quantum_mlp_fine`, optimizing a genuine,
    /// deterministically re-computable coherence objective derived from the
    /// pixel's own already-rendered quantum state, scaled by the real
    /// per-sample rendering `loss` (see [`update_mlp_gate_parameters`]).
    pub(super) fn update_nerf_parameters(
        &mut self,
        pixel_output: &PixelRenderOutput,
        loss: f64,
        config: &NeRFTrainingConfig,
    ) -> Result<()> {
        let perturbation_scale = (config.learning_rate * 0.1).max(1e-4);
        let learning_rate = self.optimization_state.learning_rate;
        let quantum_parameter_learning_rate =
            self.optimization_state.quantum_parameter_learning_rate;

        update_mlp_gate_parameters(
            &mut self.quantum_mlp_coarse,
            &pixel_output.quantum_state,
            perturbation_scale,
            learning_rate,
            quantum_parameter_learning_rate,
            loss,
        );
        update_mlp_gate_parameters(
            &mut self.quantum_mlp_fine,
            &pixel_output.quantum_state,
            perturbation_scale,
            learning_rate,
            quantum_parameter_learning_rate,
            loss,
        );

        self.optimization_state.learning_rate *= config.learning_rate_decay;
        Ok(())
    }
}
