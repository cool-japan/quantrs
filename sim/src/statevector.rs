use num_complex::Complex64;
use rayon::prelude::*;

use quantrs_circuit::builder::{Circuit, Simulator};
use quantrs_core::{
    error::{QuantrsError, QuantrsResult},
    gate::{multi, single, GateOp},
    qubit::QubitId,
    register::Register,
};

use crate::utils::{flip_bit, gate_vec_to_array2};

/// A state vector simulator for quantum circuits
///
/// This simulator implements the state vector approach, where the full quantum
/// state is represented as a complex vector of dimension 2^N for N qubits.
#[derive(Debug, Clone)]
pub struct StateVectorSimulator {
    /// Use parallel execution
    pub parallel: bool,

    /// Noise model (if any)
    pub noise_model: Option<crate::noise::NoiseModel>,
}

impl StateVectorSimulator {
    /// Create a new state vector simulator with default settings
    pub fn new() -> Self {
        Self {
            parallel: true,
            noise_model: None,
        }
    }

    /// Create a new state vector simulator with parallel execution disabled
    pub fn sequential() -> Self {
        Self {
            parallel: false,
            noise_model: None,
        }
    }

    /// Create a new state vector simulator with a noise model
    pub fn with_noise(noise_model: crate::noise::NoiseModel) -> Self {
        Self {
            parallel: true,
            noise_model: Some(noise_model),
        }
    }

    /// Set the noise model
    pub fn set_noise_model(&mut self, noise_model: crate::noise::NoiseModel) -> &mut Self {
        self.noise_model = Some(noise_model);
        self
    }

    /// Remove the noise model
    pub fn remove_noise_model(&mut self) -> &mut Self {
        self.noise_model = None;
        self
    }

    /// Apply a single-qubit gate to a state vector
    fn apply_single_qubit_gate<const N: usize>(
        &self,
        state: &mut [Complex64],
        gate_matrix: &[Complex64],
        target: QubitId,
    ) -> QuantrsResult<()> {
        let target_idx = target.id() as usize;
        if target_idx >= N {
            return Err(QuantrsError::InvalidQubitId(target.id()));
        }

        // Convert the gate matrix to a 2x2 ndarray
        let matrix = gate_vec_to_array2(gate_matrix, 2);

        // Apply the gate to each amplitude
        if self.parallel {
            // Create a copy of the state to read from while we modify the original
            let state_copy = state.to_vec();

            state.par_iter_mut().enumerate().for_each(|(idx, amp)| {
                let bit_val = (idx >> target_idx) & 1;
                let paired_idx = if bit_val == 0 {
                    idx | (1 << target_idx)
                } else {
                    idx & !(1 << target_idx)
                };

                let idx0 = if bit_val == 0 { idx } else { paired_idx };
                let idx1 = if bit_val == 0 { paired_idx } else { idx };

                let val0 = state_copy[idx0];
                let val1 = state_copy[idx1];

                *amp = if idx == idx0 {
                    matrix[[0, 0]] * val0 + matrix[[0, 1]] * val1
                } else {
                    matrix[[1, 0]] * val0 + matrix[[1, 1]] * val1
                };
            });
        } else {
            // Sequential implementation
            let dim = state.len();
            let mut new_state = vec![Complex64::new(0.0, 0.0); dim];

            for i in 0..dim {
                let bit_val = (i >> target_idx) & 1;
                let paired_idx = flip_bit(i, target_idx);

                if bit_val == 0 {
                    new_state[i] = matrix[[0, 0]] * state[i] + matrix[[0, 1]] * state[paired_idx];
                    new_state[paired_idx] =
                        matrix[[1, 0]] * state[i] + matrix[[1, 1]] * state[paired_idx];
                }
            }

            state.copy_from_slice(&new_state);
        }

        Ok(())
    }

    /// Apply a two-qubit gate to a state vector
    fn apply_two_qubit_gate<const N: usize>(
        &self,
        state: &mut [Complex64],
        gate_matrix: &[Complex64],
        control: QubitId,
        target: QubitId,
    ) -> QuantrsResult<()> {
        let control_idx = control.id() as usize;
        let target_idx = target.id() as usize;

        if control_idx >= N || target_idx >= N {
            return Err(QuantrsError::InvalidQubitId(if control_idx >= N {
                control.id()
            } else {
                target.id()
            }));
        }

        if control_idx == target_idx {
            return Err(QuantrsError::CircuitValidationFailed(
                "Control and target qubits must be different".into(),
            ));
        }

        // Convert the gate matrix to a 4x4 ndarray
        let matrix = gate_vec_to_array2(gate_matrix, 4);

        // Apply the gate to each amplitude
        if self.parallel {
            // Create a copy of the state for reading
            let state_copy = state.to_vec();

            state.par_iter_mut().enumerate().for_each(|(idx, amp)| {
                let idx00 = idx & !(1 << control_idx) & !(1 << target_idx);
                let idx01 = idx00 | (1 << target_idx);
                let idx10 = idx00 | (1 << control_idx);
                let idx11 = idx00 | (1 << control_idx) | (1 << target_idx);

                let val00 = state_copy[idx00];
                let val01 = state_copy[idx01];
                let val10 = state_copy[idx10];
                let val11 = state_copy[idx11];

                *amp = match idx {
                    i if i == idx00 => {
                        matrix[[0, 0]] * val00
                            + matrix[[0, 1]] * val01
                            + matrix[[0, 2]] * val10
                            + matrix[[0, 3]] * val11
                    }
                    i if i == idx01 => {
                        matrix[[1, 0]] * val00
                            + matrix[[1, 1]] * val01
                            + matrix[[1, 2]] * val10
                            + matrix[[1, 3]] * val11
                    }
                    i if i == idx10 => {
                        matrix[[2, 0]] * val00
                            + matrix[[2, 1]] * val01
                            + matrix[[2, 2]] * val10
                            + matrix[[2, 3]] * val11
                    }
                    i if i == idx11 => {
                        matrix[[3, 0]] * val00
                            + matrix[[3, 1]] * val01
                            + matrix[[3, 2]] * val10
                            + matrix[[3, 3]] * val11
                    }
                    _ => unreachable!(),
                };
            });
        } else {
            // Sequential implementation
            let dim = state.len();
            let mut new_state = vec![Complex64::new(0.0, 0.0); dim];

            #[allow(clippy::needless_range_loop)]
            for i in 0..dim {
                let control_bit = (i >> control_idx) & 1;
                let target_bit = (i >> target_idx) & 1;

                // Calculate the four basis states in the 2-qubit subspace
                let i00 = i & !(1 << control_idx) & !(1 << target_idx);
                let i01 = i00 | (1 << target_idx);
                let i10 = i00 | (1 << control_idx);
                let i11 = i10 | (1 << target_idx);

                let basis_idx = (control_bit << 1) | target_bit;

                // Calculate the new amplitude for this state
                new_state[i] = matrix[[basis_idx, 0]] * state[i00]
                    + matrix[[basis_idx, 1]] * state[i01]
                    + matrix[[basis_idx, 2]] * state[i10]
                    + matrix[[basis_idx, 3]] * state[i11];
            }

            state.copy_from_slice(&new_state);
        }

        Ok(())
    }

    /// Apply CNOT gate efficiently (special case)
    fn apply_cnot<const N: usize>(
        &self,
        state: &mut [Complex64],
        control: QubitId,
        target: QubitId,
    ) -> QuantrsResult<()> {
        let control_idx = control.id() as usize;
        let target_idx = target.id() as usize;

        if control_idx >= N || target_idx >= N {
            return Err(QuantrsError::InvalidQubitId(if control_idx >= N {
                control.id()
            } else {
                target.id()
            }));
        }

        if control_idx == target_idx {
            return Err(QuantrsError::CircuitValidationFailed(
                "Control and target qubits must be different".into(),
            ));
        }

        // Apply the CNOT gate - only swap amplitudes where control is 1
        let state_copy = state.to_vec();

        if self.parallel {
            state.par_iter_mut().enumerate().for_each(|(i, amp)| {
                if (i >> control_idx) & 1 == 1 {
                    let flipped = flip_bit(i, target_idx);
                    *amp = state_copy[flipped];
                }
            });
        } else {
            let dim = state.len();
            let mut new_state = vec![Complex64::new(0.0, 0.0); dim];

            for i in 0..dim {
                if (i >> control_idx) & 1 == 1 {
                    let flipped = flip_bit(i, target_idx);
                    new_state[flipped] = state[i];
                    new_state[i] = state[flipped];
                } else {
                    new_state[i] = state[i];
                }
            }

            state.copy_from_slice(&new_state);
        }

        Ok(())
    }

    /// Apply SWAP gate efficiently (special case)
    fn apply_swap<const N: usize>(
        &self,
        state: &mut [Complex64],
        qubit1: QubitId,
        qubit2: QubitId,
    ) -> QuantrsResult<()> {
        let q1_idx = qubit1.id() as usize;
        let q2_idx = qubit2.id() as usize;

        if q1_idx >= N || q2_idx >= N {
            return Err(QuantrsError::InvalidQubitId(if q1_idx >= N {
                qubit1.id()
            } else {
                qubit2.id()
            }));
        }

        if q1_idx == q2_idx {
            return Err(QuantrsError::CircuitValidationFailed(
                "Qubits must be different for SWAP gate".into(),
            ));
        }

        // Apply the SWAP gate - swap amplitudes where qubits have different values
        let state_copy = state.to_vec();

        if self.parallel {
            state.par_iter_mut().enumerate().for_each(|(i, amp)| {
                let bit1 = (i >> q1_idx) & 1;
                let bit2 = (i >> q2_idx) & 1;

                if bit1 != bit2 {
                    let swapped = flip_bit(flip_bit(i, q1_idx), q2_idx);
                    *amp = state_copy[swapped];
                }
            });
        } else {
            let dim = state.len();
            let mut new_state = vec![Complex64::new(0.0, 0.0); dim];

            for i in 0..dim {
                let bit1 = (i >> q1_idx) & 1;
                let bit2 = (i >> q2_idx) & 1;

                if bit1 != bit2 {
                    let swapped = flip_bit(flip_bit(i, q1_idx), q2_idx);
                    new_state[swapped] = state[i];
                    new_state[i] = state[swapped];
                } else {
                    new_state[i] = state[i];
                }
            }

            state.copy_from_slice(&new_state);
        }

        Ok(())
    }
}

impl Default for StateVectorSimulator {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> Simulator<N> for StateVectorSimulator {
    fn run(&self, circuit: &Circuit<N>) -> QuantrsResult<Register<N>> {
        // Initialize state vector to |0...0⟩
        let dim = 1 << N;
        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        state[0] = Complex64::new(1.0, 0.0);

        // Apply each gate in the circuit
        for gate in circuit.gates() {
            match gate.name() {
                // Single-qubit gates
                "H" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::Hadamard>() {
                        let matrix = g.matrix()?;
                        self.apply_single_qubit_gate::<N>(&mut state, &matrix, g.target)?;
                    }
                }
                "X" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::PauliX>() {
                        let matrix = g.matrix()?;
                        self.apply_single_qubit_gate::<N>(&mut state, &matrix, g.target)?;
                    }
                }
                "Y" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::PauliY>() {
                        let matrix = g.matrix()?;
                        self.apply_single_qubit_gate::<N>(&mut state, &matrix, g.target)?;
                    }
                }
                "Z" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::PauliZ>() {
                        let matrix = g.matrix()?;
                        self.apply_single_qubit_gate::<N>(&mut state, &matrix, g.target)?;
                    }
                }
                "RX" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::RotationX>() {
                        let matrix = g.matrix()?;
                        self.apply_single_qubit_gate::<N>(&mut state, &matrix, g.target)?;
                    }
                }
                "RY" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::RotationY>() {
                        let matrix = g.matrix()?;
                        self.apply_single_qubit_gate::<N>(&mut state, &matrix, g.target)?;
                    }
                }
                "RZ" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::RotationZ>() {
                        let matrix = g.matrix()?;
                        self.apply_single_qubit_gate::<N>(&mut state, &matrix, g.target)?;
                    }
                }
                "S" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::Phase>() {
                        let matrix = g.matrix()?;
                        self.apply_single_qubit_gate::<N>(&mut state, &matrix, g.target)?;
                    }
                }
                "T" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::T>() {
                        let matrix = g.matrix()?;
                        self.apply_single_qubit_gate::<N>(&mut state, &matrix, g.target)?;
                    }
                }
                "S†" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::PhaseDagger>() {
                        let matrix = g.matrix()?;
                        self.apply_single_qubit_gate::<N>(&mut state, &matrix, g.target)?;
                    }
                }
                "T†" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::TDagger>() {
                        let matrix = g.matrix()?;
                        self.apply_single_qubit_gate::<N>(&mut state, &matrix, g.target)?;
                    }
                }
                "√X" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::SqrtX>() {
                        let matrix = g.matrix()?;
                        self.apply_single_qubit_gate::<N>(&mut state, &matrix, g.target)?;
                    }
                }
                "√X†" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::SqrtXDagger>() {
                        let matrix = g.matrix()?;
                        self.apply_single_qubit_gate::<N>(&mut state, &matrix, g.target)?;
                    }
                }

                // Two-qubit gates
                "CNOT" => {
                    if let Some(g) = gate.as_any().downcast_ref::<multi::CNOT>() {
                        // Use optimized implementation for CNOT
                        self.apply_cnot::<N>(&mut state, g.control, g.target)?;
                    }
                }
                "CZ" => {
                    if let Some(g) = gate.as_any().downcast_ref::<multi::CZ>() {
                        let matrix = g.matrix()?;
                        self.apply_two_qubit_gate::<N>(&mut state, &matrix, g.control, g.target)?;
                    }
                }
                "SWAP" => {
                    if let Some(g) = gate.as_any().downcast_ref::<multi::SWAP>() {
                        // Use optimized implementation for SWAP
                        self.apply_swap::<N>(&mut state, g.qubit1, g.qubit2)?;
                    }
                }
                "CY" => {
                    if let Some(g) = gate.as_any().downcast_ref::<multi::CY>() {
                        let matrix = g.matrix()?;
                        self.apply_two_qubit_gate::<N>(&mut state, &matrix, g.control, g.target)?;
                    }
                }
                "CH" => {
                    if let Some(g) = gate.as_any().downcast_ref::<multi::CH>() {
                        let matrix = g.matrix()?;
                        self.apply_two_qubit_gate::<N>(&mut state, &matrix, g.control, g.target)?;
                    }
                }
                "CS" => {
                    if let Some(g) = gate.as_any().downcast_ref::<multi::CS>() {
                        let matrix = g.matrix()?;
                        self.apply_two_qubit_gate::<N>(&mut state, &matrix, g.control, g.target)?;
                    }
                }
                "CRX" => {
                    if let Some(g) = gate.as_any().downcast_ref::<multi::CRX>() {
                        let matrix = g.matrix()?;
                        self.apply_two_qubit_gate::<N>(&mut state, &matrix, g.control, g.target)?;
                    }
                }
                "CRY" => {
                    if let Some(g) = gate.as_any().downcast_ref::<multi::CRY>() {
                        let matrix = g.matrix()?;
                        self.apply_two_qubit_gate::<N>(&mut state, &matrix, g.control, g.target)?;
                    }
                }
                "CRZ" => {
                    if let Some(g) = gate.as_any().downcast_ref::<multi::CRZ>() {
                        let matrix = g.matrix()?;
                        self.apply_two_qubit_gate::<N>(&mut state, &matrix, g.control, g.target)?;
                    }
                }

                // Three-qubit gates
                "Toffoli" => {
                    if gate.as_any().downcast_ref::<multi::Toffoli>().is_some() {
                        // Implement Toffoli as a sequence of simpler gates
                        // (This is a placeholder for a more efficient implementation)
                        return Err(QuantrsError::UnsupportedOperation(
                            "Direct Toffoli gate not yet implemented. Use gate decomposition."
                                .into(),
                        ));
                    }
                }
                "Fredkin" => {
                    if gate.as_any().downcast_ref::<multi::Fredkin>().is_some() {
                        // Implement Fredkin as a sequence of simpler gates
                        // (This is a placeholder for a more efficient implementation)
                        return Err(QuantrsError::UnsupportedOperation(
                            "Direct Fredkin gate not yet implemented. Use gate decomposition."
                                .into(),
                        ));
                    }
                }

                _ => {
                    return Err(QuantrsError::UnsupportedOperation(format!(
                        "Gate {} not supported",
                        gate.name()
                    )));
                }
            }

            // Apply per-gate noise if configured
            if let Some(ref noise_model) = self.noise_model {
                if noise_model.per_gate {
                    noise_model.apply_to_statevector(&mut state)?;
                }
            }
        }

        // Apply final noise if not per-gate
        if let Some(ref noise_model) = self.noise_model {
            if !noise_model.per_gate {
                noise_model.apply_to_statevector(&mut state)?;
            }
        }

        // Create register from final state
        Register::<N>::with_amplitudes(state)
    }
}
