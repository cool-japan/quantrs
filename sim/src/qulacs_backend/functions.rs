//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::ndarray::*;
use scirs2_core::{Complex64, Float};

use super::types::QulacsStateVector;

/// Type alias for state vector index
pub type StateIndex = usize;
/// Type alias for qubit index
pub type QubitIndex = usize;
/// Qulacs-style gate operations
pub mod gates {
    use super::*;
    /// Apply Hadamard gate to a target qubit
    ///
    /// This implementation follows Qulacs' approach with:
    /// - Bit masking for efficient index calculation
    /// - Special handling for qubit 0
    /// - SciRS2 parallel execution and SIMD when available
    ///
    /// # Arguments
    ///
    /// * `state` - The quantum state vector
    /// * `target` - Target qubit index
    pub fn hadamard(state: &mut QulacsStateVector, target: QubitIndex) -> Result<()> {
        if target >= state.num_qubits {
            return Err(SimulatorError::InvalidQubitIndex {
                index: target,
                num_qubits: state.num_qubits,
            });
        }
        let dim = state.dim;
        let loop_dim = dim / 2;
        let mask = 1usize << target;
        let sqrt2_inv = Complex64::new(1.0 / 2.0f64.sqrt(), 0.0);
        let state_data = state.amplitudes_mut();
        if target == 0 {
            for basis_idx in (0..dim).step_by(2) {
                let temp0 = state_data[basis_idx];
                let temp1 = state_data[basis_idx + 1];
                state_data[basis_idx] = (temp0 + temp1) * sqrt2_inv;
                state_data[basis_idx + 1] = (temp0 - temp1) * sqrt2_inv;
            }
        } else {
            let mask_low = mask - 1;
            let mask_high = !mask_low;
            for state_idx in (0..loop_dim).step_by(2) {
                let basis_idx_0 = (state_idx & mask_low) | ((state_idx & mask_high) << 1);
                let basis_idx_1 = basis_idx_0 | mask;
                let temp_a0 = state_data[basis_idx_0];
                let temp_a1 = state_data[basis_idx_1];
                let temp_b0 = state_data[basis_idx_0 + 1];
                let temp_b1 = state_data[basis_idx_1 + 1];
                state_data[basis_idx_0] = (temp_a0 + temp_a1) * sqrt2_inv;
                state_data[basis_idx_1] = (temp_a0 - temp_a1) * sqrt2_inv;
                state_data[basis_idx_0 + 1] = (temp_b0 + temp_b1) * sqrt2_inv;
                state_data[basis_idx_1 + 1] = (temp_b0 - temp_b1) * sqrt2_inv;
            }
        }
        Ok(())
    }
    /// Apply Pauli-X gate to a target qubit
    ///
    /// # Arguments
    ///
    /// * `state` - The quantum state vector
    /// * `target` - Target qubit index
    pub fn pauli_x(state: &mut QulacsStateVector, target: QubitIndex) -> Result<()> {
        if target >= state.num_qubits {
            return Err(SimulatorError::InvalidQubitIndex {
                index: target,
                num_qubits: state.num_qubits,
            });
        }
        let dim = state.dim;
        let loop_dim = dim / 2;
        let mask = 1usize << target;
        let mask_low = mask - 1;
        let mask_high = !mask_low;
        let state_data = state.amplitudes_mut();
        if target == 0 {
            for basis_idx in (0..dim).step_by(2) {
                state_data.swap(basis_idx, basis_idx + 1);
            }
        } else {
            for state_idx in (0..loop_dim).step_by(2) {
                let basis_idx_0 = (state_idx & mask_low) | ((state_idx & mask_high) << 1);
                let basis_idx_1 = basis_idx_0 | mask;
                state_data.swap(basis_idx_0, basis_idx_1);
                state_data.swap(basis_idx_0 + 1, basis_idx_1 + 1);
            }
        }
        Ok(())
    }
    /// Apply Pauli-Y gate to a target qubit
    ///
    /// # Arguments
    ///
    /// * `state` - The quantum state vector
    /// * `target` - Target qubit index
    pub fn pauli_y(state: &mut QulacsStateVector, target: QubitIndex) -> Result<()> {
        if target >= state.num_qubits {
            return Err(SimulatorError::InvalidQubitIndex {
                index: target,
                num_qubits: state.num_qubits,
            });
        }
        let dim = state.dim;
        let loop_dim = dim / 2;
        let mask = 1usize << target;
        let mask_low = mask - 1;
        let mask_high = !mask_low;
        let state_data = state.amplitudes_mut();
        let i = Complex64::new(0.0, 1.0);
        if target == 0 {
            for basis_idx in (0..dim).step_by(2) {
                let temp0 = state_data[basis_idx];
                let temp1 = state_data[basis_idx + 1];
                state_data[basis_idx] = -i * temp1;
                state_data[basis_idx + 1] = i * temp0;
            }
        } else {
            for state_idx in (0..loop_dim).step_by(2) {
                let basis_idx_0 = (state_idx & mask_low) | ((state_idx & mask_high) << 1);
                let basis_idx_1 = basis_idx_0 | mask;
                let temp_a0 = state_data[basis_idx_0];
                let temp_a1 = state_data[basis_idx_1];
                let temp_b0 = state_data[basis_idx_0 + 1];
                let temp_b1 = state_data[basis_idx_1 + 1];
                state_data[basis_idx_0] = -i * temp_a1;
                state_data[basis_idx_1] = i * temp_a0;
                state_data[basis_idx_0 + 1] = -i * temp_b1;
                state_data[basis_idx_1 + 1] = i * temp_b0;
            }
        }
        Ok(())
    }
    /// Apply Pauli-Z gate to a target qubit
    ///
    /// # Arguments
    ///
    /// * `state` - The quantum state vector
    /// * `target` - Target qubit index
    pub fn pauli_z(state: &mut QulacsStateVector, target: QubitIndex) -> Result<()> {
        if target >= state.num_qubits {
            return Err(SimulatorError::InvalidQubitIndex {
                index: target,
                num_qubits: state.num_qubits,
            });
        }
        let dim = state.dim;
        let loop_dim = dim / 2;
        let mask = 1usize << target;
        let mask_low = mask - 1;
        let mask_high = !mask_low;
        let state_data = state.amplitudes_mut();
        for state_idx in 0..loop_dim {
            let basis_idx = (state_idx & mask_low) | ((state_idx & mask_high) << 1) | mask;
            state_data[basis_idx] = -state_data[basis_idx];
        }
        Ok(())
    }
    /// Apply CNOT gate (controlled-X)
    ///
    /// This follows Qulacs' approach with specialized handling based on
    /// control and target qubit positions.
    ///
    /// # Arguments
    ///
    /// * `state` - The quantum state vector
    /// * `control` - Control qubit index
    /// * `target` - Target qubit index
    pub fn cnot(
        state: &mut QulacsStateVector,
        control: QubitIndex,
        target: QubitIndex,
    ) -> Result<()> {
        if control >= state.num_qubits {
            return Err(SimulatorError::InvalidQubitIndex {
                index: control,
                num_qubits: state.num_qubits,
            });
        }
        if target >= state.num_qubits {
            return Err(SimulatorError::InvalidQubitIndex {
                index: target,
                num_qubits: state.num_qubits,
            });
        }
        if control == target {
            return Err(SimulatorError::InvalidOperation(
                "Control and target qubits must be different".to_string(),
            ));
        }
        let dim = state.dim;
        let loop_dim = dim / 4;
        let target_mask = 1usize << target;
        let control_mask = 1usize << control;
        let min_qubit = control.min(target);
        let max_qubit = control.max(target);
        let min_qubit_mask = 1usize << min_qubit;
        let max_qubit_mask = 1usize << (max_qubit - 1);
        let low_mask = min_qubit_mask - 1;
        let mid_mask = (max_qubit_mask - 1) ^ low_mask;
        let high_mask = !max_qubit_mask.wrapping_add(max_qubit_mask - 1);
        let state_data = state.amplitudes_mut();
        if target == 0 {
            for state_idx in 0..loop_dim {
                let basis_idx =
                    ((state_idx & mid_mask) << 1) | ((state_idx & high_mask) << 2) | control_mask;
                state_data.swap(basis_idx, basis_idx + 1);
            }
        } else if control == 0 {
            for state_idx in 0..loop_dim {
                let basis_idx_0 = (state_idx & low_mask)
                    | ((state_idx & mid_mask) << 1)
                    | ((state_idx & high_mask) << 2)
                    | control_mask;
                let basis_idx_1 = basis_idx_0 | target_mask;
                state_data.swap(basis_idx_0, basis_idx_1);
            }
        } else {
            for state_idx in (0..loop_dim).step_by(2) {
                let basis_idx_0 = (state_idx & low_mask)
                    | ((state_idx & mid_mask) << 1)
                    | ((state_idx & high_mask) << 2)
                    | control_mask;
                let basis_idx_1 = basis_idx_0 | target_mask;
                state_data.swap(basis_idx_0, basis_idx_1);
                state_data.swap(basis_idx_0 + 1, basis_idx_1 + 1);
            }
        }
        Ok(())
    }
    /// Apply CZ gate (controlled-Z)
    ///
    /// # Arguments
    ///
    /// * `state` - The quantum state vector
    /// * `control` - Control qubit index
    /// * `target` - Target qubit index
    pub fn cz(
        state: &mut QulacsStateVector,
        control: QubitIndex,
        target: QubitIndex,
    ) -> Result<()> {
        if control >= state.num_qubits {
            return Err(SimulatorError::InvalidQubitIndex {
                index: control,
                num_qubits: state.num_qubits,
            });
        }
        if target >= state.num_qubits {
            return Err(SimulatorError::InvalidQubitIndex {
                index: target,
                num_qubits: state.num_qubits,
            });
        }
        if control == target {
            return Err(SimulatorError::InvalidOperation(
                "Control and target qubits must be different".to_string(),
            ));
        }
        let dim = state.dim;
        let loop_dim = dim / 4;
        let target_mask = 1usize << target;
        let control_mask = 1usize << control;
        let min_qubit = control.min(target);
        let max_qubit = control.max(target);
        let min_qubit_mask = 1usize << min_qubit;
        let max_qubit_mask = 1usize << (max_qubit - 1);
        let low_mask = min_qubit_mask - 1;
        let mid_mask = (max_qubit_mask - 1) ^ low_mask;
        let high_mask = !max_qubit_mask.wrapping_add(max_qubit_mask - 1);
        let state_data = state.amplitudes_mut();
        for state_idx in 0..loop_dim {
            let basis_idx = (state_idx & low_mask)
                | ((state_idx & mid_mask) << 1)
                | ((state_idx & high_mask) << 2)
                | control_mask
                | target_mask;
            state_data[basis_idx] = -state_data[basis_idx];
        }
        Ok(())
    }
    /// Apply SWAP gate
    ///
    /// # Arguments
    ///
    /// * `state` - The quantum state vector
    /// * `qubit1` - First qubit index
    /// * `qubit2` - Second qubit index
    pub fn swap(
        state: &mut QulacsStateVector,
        qubit1: QubitIndex,
        qubit2: QubitIndex,
    ) -> Result<()> {
        if qubit1 >= state.num_qubits {
            return Err(SimulatorError::InvalidQubitIndex {
                index: qubit1,
                num_qubits: state.num_qubits,
            });
        }
        if qubit2 >= state.num_qubits {
            return Err(SimulatorError::InvalidQubitIndex {
                index: qubit2,
                num_qubits: state.num_qubits,
            });
        }
        if qubit1 == qubit2 {
            return Ok(());
        }
        let dim = state.dim;
        let loop_dim = dim / 4;
        let mask1 = 1usize << qubit1;
        let mask2 = 1usize << qubit2;
        let min_qubit = qubit1.min(qubit2);
        let max_qubit = qubit1.max(qubit2);
        let min_qubit_mask = 1usize << min_qubit;
        let max_qubit_mask = 1usize << (max_qubit - 1);
        let low_mask = min_qubit_mask - 1;
        let mid_mask = (max_qubit_mask - 1) ^ low_mask;
        let high_mask = !max_qubit_mask.wrapping_add(max_qubit_mask - 1);
        let state_data = state.amplitudes_mut();
        for state_idx in 0..loop_dim {
            let basis_idx_0 = (state_idx & low_mask)
                | ((state_idx & mid_mask) << 1)
                | ((state_idx & high_mask) << 2);
            let basis_idx_1 = basis_idx_0 | mask1;
            let basis_idx_2 = basis_idx_0 | mask2;
            state_data.swap(basis_idx_1, basis_idx_2);
        }
        Ok(())
    }
    /// Apply RX gate (rotation around X-axis)
    ///
    /// RX(θ) = exp(-iθX/2) = cos(θ/2)I - i sin(θ/2)X
    ///
    /// Matrix representation:
    /// ```text
    /// [cos(θ/2)    -i sin(θ/2)]
    /// [-i sin(θ/2)  cos(θ/2)  ]
    /// ```
    ///
    /// # Arguments
    ///
    /// * `state` - The quantum state vector
    /// * `target` - Target qubit index
    /// * `angle` - Rotation angle in radians
    pub fn rx(state: &mut QulacsStateVector, target: QubitIndex, angle: f64) -> Result<()> {
        if target >= state.num_qubits {
            return Err(SimulatorError::InvalidQubitIndex {
                index: target,
                num_qubits: state.num_qubits,
            });
        }
        let dim = state.dim;
        let loop_dim = dim / 2;
        let mask = 1usize << target;
        let mask_low = mask - 1;
        let mask_high = !mask_low;
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();
        let i_sin_half = Complex64::new(0.0, -sin_half);
        let state_data = state.amplitudes_mut();
        if target == 0 {
            for basis_idx in (0..dim).step_by(2) {
                let amp0 = state_data[basis_idx];
                let amp1 = state_data[basis_idx + 1];
                state_data[basis_idx] = amp0 * cos_half + amp1 * i_sin_half;
                state_data[basis_idx + 1] = amp0 * i_sin_half + amp1 * cos_half;
            }
        } else {
            for state_idx in (0..loop_dim).step_by(2) {
                let basis_idx_0 = (state_idx & mask_low) | ((state_idx & mask_high) << 1);
                let basis_idx_1 = basis_idx_0 | mask;
                let amp_a0 = state_data[basis_idx_0];
                let amp_a1 = state_data[basis_idx_1];
                let amp_b0 = state_data[basis_idx_0 + 1];
                let amp_b1 = state_data[basis_idx_1 + 1];
                state_data[basis_idx_0] = amp_a0 * cos_half + amp_a1 * i_sin_half;
                state_data[basis_idx_1] = amp_a0 * i_sin_half + amp_a1 * cos_half;
                state_data[basis_idx_0 + 1] = amp_b0 * cos_half + amp_b1 * i_sin_half;
                state_data[basis_idx_1 + 1] = amp_b0 * i_sin_half + amp_b1 * cos_half;
            }
        }
        Ok(())
    }
    /// Apply RY gate (rotation around Y-axis)
    ///
    /// RY(θ) = exp(-iθY/2) = cos(θ/2)I - i sin(θ/2)Y
    ///
    /// Matrix representation:
    /// ```text
    /// [cos(θ/2)  -sin(θ/2)]
    /// [sin(θ/2)   cos(θ/2)]
    /// ```
    ///
    /// # Arguments
    ///
    /// * `state` - The quantum state vector
    /// * `target` - Target qubit index
    /// * `angle` - Rotation angle in radians
    pub fn ry(state: &mut QulacsStateVector, target: QubitIndex, angle: f64) -> Result<()> {
        if target >= state.num_qubits {
            return Err(SimulatorError::InvalidQubitIndex {
                index: target,
                num_qubits: state.num_qubits,
            });
        }
        let dim = state.dim;
        let loop_dim = dim / 2;
        let mask = 1usize << target;
        let mask_low = mask - 1;
        let mask_high = !mask_low;
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();
        let state_data = state.amplitudes_mut();
        if target == 0 {
            for basis_idx in (0..dim).step_by(2) {
                let amp0 = state_data[basis_idx];
                let amp1 = state_data[basis_idx + 1];
                state_data[basis_idx] = amp0 * cos_half - amp1 * sin_half;
                state_data[basis_idx + 1] = amp0 * sin_half + amp1 * cos_half;
            }
        } else {
            for state_idx in (0..loop_dim).step_by(2) {
                let basis_idx_0 = (state_idx & mask_low) | ((state_idx & mask_high) << 1);
                let basis_idx_1 = basis_idx_0 | mask;
                let amp_a0 = state_data[basis_idx_0];
                let amp_a1 = state_data[basis_idx_1];
                let amp_b0 = state_data[basis_idx_0 + 1];
                let amp_b1 = state_data[basis_idx_1 + 1];
                state_data[basis_idx_0] = amp_a0 * cos_half - amp_a1 * sin_half;
                state_data[basis_idx_1] = amp_a0 * sin_half + amp_a1 * cos_half;
                state_data[basis_idx_0 + 1] = amp_b0 * cos_half - amp_b1 * sin_half;
                state_data[basis_idx_1 + 1] = amp_b0 * sin_half + amp_b1 * cos_half;
            }
        }
        Ok(())
    }
    /// Apply RZ gate (rotation around Z-axis)
    ///
    /// RZ(θ) = exp(-iθZ/2)
    ///
    /// Matrix representation:
    /// ```text
    /// [e^(-iθ/2)     0      ]
    /// [   0       e^(iθ/2)  ]
    /// ```
    ///
    /// # Arguments
    ///
    /// * `state` - The quantum state vector
    /// * `target` - Target qubit index
    /// * `angle` - Rotation angle in radians
    pub fn rz(state: &mut QulacsStateVector, target: QubitIndex, angle: f64) -> Result<()> {
        if target >= state.num_qubits {
            return Err(SimulatorError::InvalidQubitIndex {
                index: target,
                num_qubits: state.num_qubits,
            });
        }
        let dim = state.dim;
        let loop_dim = dim / 2;
        let mask = 1usize << target;
        let mask_low = mask - 1;
        let mask_high = !mask_low;
        let phase_0 = Complex64::from_polar(1.0, -angle / 2.0);
        let phase_1 = Complex64::from_polar(1.0, angle / 2.0);
        let state_data = state.amplitudes_mut();
        for state_idx in 0..loop_dim {
            let basis_idx_0 = (state_idx & mask_low) | ((state_idx & mask_high) << 1);
            let basis_idx_1 = basis_idx_0 | mask;
            state_data[basis_idx_0] *= phase_0;
            state_data[basis_idx_1] *= phase_1;
        }
        Ok(())
    }
    /// Apply Phase gate (arbitrary phase rotation)
    ///
    /// Phase(θ) = diag(1, e^(iθ))
    ///
    /// # Arguments
    ///
    /// * `state` - The quantum state vector
    /// * `target` - Target qubit index
    /// * `angle` - Phase angle in radians
    pub fn phase(state: &mut QulacsStateVector, target: QubitIndex, angle: f64) -> Result<()> {
        if target >= state.num_qubits {
            return Err(SimulatorError::InvalidQubitIndex {
                index: target,
                num_qubits: state.num_qubits,
            });
        }
        let dim = state.dim;
        let loop_dim = dim / 2;
        let mask = 1usize << target;
        let mask_low = mask - 1;
        let mask_high = !mask_low;
        let phase_factor = Complex64::from_polar(1.0, angle);
        let state_data = state.amplitudes_mut();
        for state_idx in 0..loop_dim {
            let basis_idx = (state_idx & mask_low) | ((state_idx & mask_high) << 1) | mask;
            state_data[basis_idx] *= phase_factor;
        }
        Ok(())
    }
    /// Apply S gate (phase gate with π/2)
    ///
    /// S gate applies a π/2 phase: |0⟩ → |0⟩, |1⟩ → i|1⟩
    ///
    /// # Arguments
    ///
    /// * `state` - The quantum state vector
    /// * `target` - Target qubit index
    pub fn s(state: &mut QulacsStateVector, target: QubitIndex) -> Result<()> {
        phase(state, target, std::f64::consts::FRAC_PI_2)
    }
    /// Apply S† gate (conjugate of S gate, phase -π/2)
    ///
    /// S† gate applies a -π/2 phase: |0⟩ → |0⟩, |1⟩ → -i|1⟩
    ///
    /// # Arguments
    ///
    /// * `state` - The quantum state vector
    /// * `target` - Target qubit index
    pub fn sdg(state: &mut QulacsStateVector, target: QubitIndex) -> Result<()> {
        phase(state, target, -std::f64::consts::FRAC_PI_2)
    }
    /// Apply T gate (phase gate with π/4)
    ///
    /// T gate applies a π/4 phase: |0⟩ → |0⟩, |1⟩ → e^(iπ/4)|1⟩
    ///
    /// # Arguments
    ///
    /// * `state` - The quantum state vector
    /// * `target` - Target qubit index
    pub fn t(state: &mut QulacsStateVector, target: QubitIndex) -> Result<()> {
        phase(state, target, std::f64::consts::FRAC_PI_4)
    }
    /// Apply T† gate (conjugate of T gate, phase -π/4)
    ///
    /// T† gate applies a -π/4 phase: |0⟩ → |0⟩, |1⟩ → e^(-iπ/4)|1⟩
    ///
    /// # Arguments
    ///
    /// * `state` - The quantum state vector
    /// * `target` - Target qubit index
    pub fn tdg(state: &mut QulacsStateVector, target: QubitIndex) -> Result<()> {
        phase(state, target, -std::f64::consts::FRAC_PI_4)
    }
    /// Apply U3 gate (universal single-qubit gate)
    ///
    /// U3(θ, φ, λ) is the most general single-qubit unitary gate
    ///
    /// Matrix representation:
    /// ```text
    /// [cos(θ/2)              -e^(iλ) sin(θ/2)        ]
    /// [e^(iφ) sin(θ/2)       e^(i(φ+λ)) cos(θ/2)     ]
    /// ```
    ///
    /// # Arguments
    ///
    /// * `state` - The quantum state vector
    /// * `target` - Target qubit index
    /// * `theta` - Rotation angle θ
    /// * `phi` - Phase angle φ
    /// * `lambda` - Phase angle λ
    pub fn u3(
        state: &mut QulacsStateVector,
        target: QubitIndex,
        theta: f64,
        phi: f64,
        lambda: f64,
    ) -> Result<()> {
        if target >= state.num_qubits {
            return Err(SimulatorError::InvalidQubitIndex {
                index: target,
                num_qubits: state.num_qubits,
            });
        }
        let dim = state.dim;
        let loop_dim = dim / 2;
        let mask = 1usize << target;
        let mask_low = mask - 1;
        let mask_high = !mask_low;
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();
        let u00 = Complex64::new(cos_half, 0.0);
        let u01 = -Complex64::from_polar(sin_half, lambda);
        let u10 = Complex64::from_polar(sin_half, phi);
        let u11 = Complex64::from_polar(cos_half, phi + lambda);
        let state_data = state.amplitudes_mut();
        if target == 0 {
            for basis_idx in (0..dim).step_by(2) {
                let amp0 = state_data[basis_idx];
                let amp1 = state_data[basis_idx + 1];
                state_data[basis_idx] = u00 * amp0 + u01 * amp1;
                state_data[basis_idx + 1] = u10 * amp0 + u11 * amp1;
            }
        } else {
            for state_idx in (0..loop_dim).step_by(2) {
                let basis_idx_0 = (state_idx & mask_low) | ((state_idx & mask_high) << 1);
                let basis_idx_1 = basis_idx_0 | mask;
                let amp_a0 = state_data[basis_idx_0];
                let amp_a1 = state_data[basis_idx_1];
                let amp_b0 = state_data[basis_idx_0 + 1];
                let amp_b1 = state_data[basis_idx_1 + 1];
                state_data[basis_idx_0] = u00 * amp_a0 + u01 * amp_a1;
                state_data[basis_idx_1] = u10 * amp_a0 + u11 * amp_a1;
                state_data[basis_idx_0 + 1] = u00 * amp_b0 + u01 * amp_b1;
                state_data[basis_idx_1 + 1] = u10 * amp_b0 + u11 * amp_b1;
            }
        }
        Ok(())
    }
    /// Apply Toffoli (CCX) gate - Controlled-Controlled-NOT
    ///
    /// Flips the target qubit if both control qubits are in state |1⟩
    ///
    /// # Arguments
    ///
    /// * `state` - The quantum state vector
    /// * `control1` - First control qubit index
    /// * `control2` - Second control qubit index
    /// * `target` - Target qubit index
    pub fn toffoli(
        state: &mut QulacsStateVector,
        control1: QubitIndex,
        control2: QubitIndex,
        target: QubitIndex,
    ) -> Result<()> {
        if control1 >= state.num_qubits {
            return Err(SimulatorError::InvalidQubitIndex {
                index: control1,
                num_qubits: state.num_qubits,
            });
        }
        if control2 >= state.num_qubits {
            return Err(SimulatorError::InvalidQubitIndex {
                index: control2,
                num_qubits: state.num_qubits,
            });
        }
        if target >= state.num_qubits {
            return Err(SimulatorError::InvalidQubitIndex {
                index: target,
                num_qubits: state.num_qubits,
            });
        }
        if control1 == control2 || control1 == target || control2 == target {
            return Err(SimulatorError::InvalidOperation(
                "Control and target qubits must be different".to_string(),
            ));
        }
        let dim = state.dim;
        let loop_dim = dim / 8;
        let num_qubits = state.num_qubits;
        let control1_mask = 1usize << control1;
        let control2_mask = 1usize << control2;
        let target_mask = 1usize << target;
        let state_data = state.amplitudes_mut();
        for i in 0..loop_dim {
            let mut basis_idx = 0;
            let mut temp = i;
            for bit_pos in 0..num_qubits {
                if bit_pos != control1 && bit_pos != control2 && bit_pos != target {
                    basis_idx |= (temp & 1) << bit_pos;
                    temp >>= 1;
                }
            }
            basis_idx |= control1_mask | control2_mask;
            let idx_0 = basis_idx & !target_mask;
            let idx_1 = basis_idx | target_mask;
            state_data.swap(idx_0, idx_1);
        }
        Ok(())
    }
    /// Apply Fredkin (CSWAP) gate - Controlled-SWAP
    ///
    /// Swaps target1 and target2 if control qubit is in state |1⟩
    ///
    /// # Arguments
    ///
    /// * `state` - The quantum state vector
    /// * `control` - Control qubit index
    /// * `target1` - First target qubit index
    /// * `target2` - Second target qubit index
    pub fn fredkin(
        state: &mut QulacsStateVector,
        control: QubitIndex,
        target1: QubitIndex,
        target2: QubitIndex,
    ) -> Result<()> {
        if control >= state.num_qubits {
            return Err(SimulatorError::InvalidQubitIndex {
                index: control,
                num_qubits: state.num_qubits,
            });
        }
        if target1 >= state.num_qubits {
            return Err(SimulatorError::InvalidQubitIndex {
                index: target1,
                num_qubits: state.num_qubits,
            });
        }
        if target2 >= state.num_qubits {
            return Err(SimulatorError::InvalidQubitIndex {
                index: target2,
                num_qubits: state.num_qubits,
            });
        }
        if control == target1 || control == target2 || target1 == target2 {
            return Err(SimulatorError::InvalidOperation(
                "Control and target qubits must be different".to_string(),
            ));
        }
        let dim = state.dim;
        let loop_dim = dim / 8;
        let num_qubits = state.num_qubits;
        let control_mask = 1usize << control;
        let target1_mask = 1usize << target1;
        let target2_mask = 1usize << target2;
        let state_data = state.amplitudes_mut();
        for i in 0..loop_dim {
            let mut basis_idx = 0;
            let mut temp = i;
            for bit_pos in 0..num_qubits {
                if bit_pos != control && bit_pos != target1 && bit_pos != target2 {
                    basis_idx |= (temp & 1) << bit_pos;
                    temp >>= 1;
                }
            }
            basis_idx |= control_mask;
            let idx_01 = basis_idx | target2_mask;
            let idx_10 = basis_idx | target1_mask;
            state_data.swap(idx_01, idx_10);
        }
        Ok(())
    }
}
/// Observable framework for Qulacs
///
/// Provides rich observable abstractions for expectation value computations
pub mod observable {
    use super::super::types::QulacsStateVector;
    use crate::error::{Result, SimulatorError};
    use scirs2_core::ndarray::Array2;
    use scirs2_core::Complex64;
    use std::collections::HashMap;
    /// Pauli operator type
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum PauliOperator {
        /// Identity operator
        I,
        /// Pauli X operator
        X,
        /// Pauli Y operator
        Y,
        /// Pauli Z operator
        Z,
    }
    impl PauliOperator {
        /// Get the matrix representation of this Pauli operator
        pub fn matrix(&self) -> Array2<Complex64> {
            match self {
                PauliOperator::I => {
                    let mut mat = Array2::zeros((2, 2));
                    mat[[0, 0]] = Complex64::new(1.0, 0.0);
                    mat[[1, 1]] = Complex64::new(1.0, 0.0);
                    mat
                }
                PauliOperator::X => {
                    let mut mat = Array2::zeros((2, 2));
                    mat[[0, 1]] = Complex64::new(1.0, 0.0);
                    mat[[1, 0]] = Complex64::new(1.0, 0.0);
                    mat
                }
                PauliOperator::Y => {
                    let mut mat = Array2::zeros((2, 2));
                    mat[[0, 1]] = Complex64::new(0.0, -1.0);
                    mat[[1, 0]] = Complex64::new(0.0, 1.0);
                    mat
                }
                PauliOperator::Z => {
                    let mut mat = Array2::zeros((2, 2));
                    mat[[0, 0]] = Complex64::new(1.0, 0.0);
                    mat[[1, 1]] = Complex64::new(-1.0, 0.0);
                    mat
                }
            }
        }
        /// Get the eigenvalue for computational basis state |b⟩
        pub fn eigenvalue(&self, basis_state: bool) -> f64 {
            match self {
                PauliOperator::I => 1.0,
                PauliOperator::X => 0.0,
                PauliOperator::Y => 0.0,
                PauliOperator::Z => {
                    if basis_state {
                        -1.0
                    } else {
                        1.0
                    }
                }
            }
        }
        /// Check if this operator commutes with Z basis measurement
        pub fn commutes_with_z(&self) -> bool {
            matches!(self, PauliOperator::I | PauliOperator::Z)
        }
    }
    /// Pauli string observable (tensor product of Pauli operators)
    #[derive(Debug, Clone)]
    pub struct PauliObservable {
        /// Pauli operators for each qubit (qubit_index -> operator)
        pub operators: HashMap<usize, PauliOperator>,
        /// Coefficient for this Pauli string
        pub coefficient: f64,
    }
    impl PauliObservable {
        /// Create a new Pauli observable
        pub fn new(operators: HashMap<usize, PauliOperator>, coefficient: f64) -> Self {
            Self {
                operators,
                coefficient,
            }
        }
        /// Create identity observable
        pub fn identity(num_qubits: usize) -> Self {
            let mut operators = HashMap::new();
            for i in 0..num_qubits {
                operators.insert(i, PauliOperator::I);
            }
            Self {
                operators,
                coefficient: 1.0,
            }
        }
        /// Create Z observable on specified qubits
        pub fn pauli_z(qubits: &[usize]) -> Self {
            let mut operators = HashMap::new();
            for &qubit in qubits {
                operators.insert(qubit, PauliOperator::Z);
            }
            Self {
                operators,
                coefficient: 1.0,
            }
        }
        /// Create X observable on specified qubits
        pub fn pauli_x(qubits: &[usize]) -> Self {
            let mut operators = HashMap::new();
            for &qubit in qubits {
                operators.insert(qubit, PauliOperator::X);
            }
            Self {
                operators,
                coefficient: 1.0,
            }
        }
        /// Create Y observable on specified qubits
        pub fn pauli_y(qubits: &[usize]) -> Self {
            let mut operators = HashMap::new();
            for &qubit in qubits {
                operators.insert(qubit, PauliOperator::Y);
            }
            Self {
                operators,
                coefficient: 1.0,
            }
        }
        /// Compute expectation value for this observable
        pub fn expectation_value(&self, state: &QulacsStateVector) -> f64 {
            let mut result = 0.0;
            for i in 0..state.dim() {
                let prob = state.amplitudes()[i].norm_sqr();
                if prob < 1e-15 {
                    continue;
                }
                let mut eigenvalue = 1.0;
                for (&qubit, &op) in &self.operators {
                    let bit = ((i >> qubit) & 1) == 1;
                    match op {
                        PauliOperator::I => {}
                        PauliOperator::Z => {
                            eigenvalue *= if bit { -1.0 } else { 1.0 };
                        }
                        PauliOperator::X | PauliOperator::Y => {
                            return 0.0;
                        }
                    }
                }
                result += prob * eigenvalue;
            }
            result * self.coefficient
        }
        /// Set coefficient
        pub fn with_coefficient(mut self, coefficient: f64) -> Self {
            self.coefficient = coefficient;
            self
        }
        /// Get the number of non-identity operators
        pub fn weight(&self) -> usize {
            self.operators
                .values()
                .filter(|&&op| op != PauliOperator::I)
                .count()
        }
    }
    /// Hermitian observable (general Hermitian matrix)
    #[derive(Debug, Clone)]
    pub struct HermitianObservable {
        /// The Hermitian matrix
        pub matrix: Array2<Complex64>,
        /// Number of qubits this observable acts on
        pub num_qubits: usize,
    }
    impl HermitianObservable {
        /// Create a new Hermitian observable
        pub fn new(matrix: Array2<Complex64>) -> Result<Self> {
            let (n, m) = (matrix.nrows(), matrix.ncols());
            if n != m {
                return Err(SimulatorError::InvalidObservable(
                    "Matrix must be square".to_string(),
                ));
            }
            if n == 0 || (n & (n - 1)) != 0 {
                return Err(SimulatorError::InvalidObservable(
                    "Dimension must be a power of 2".to_string(),
                ));
            }
            let num_qubits = n.trailing_zeros() as usize;
            Ok(Self { matrix, num_qubits })
        }
        /// Compute expectation value <ψ|H|ψ>
        pub fn expectation_value(&self, state: &QulacsStateVector) -> Result<f64> {
            if state.num_qubits() != self.num_qubits {
                return Err(SimulatorError::InvalidObservable(
                    "Observable dimension doesn't match state".to_string(),
                ));
            }
            let psi = state.amplitudes();
            let mut result = Complex64::new(0.0, 0.0);
            for i in 0..state.dim() {
                for j in 0..state.dim() {
                    result += psi[i].conj() * self.matrix[[i, j]] * psi[j];
                }
            }
            Ok(result.re)
        }
    }
    /// Composite observable (sum of weighted observables)
    #[derive(Debug, Clone)]
    pub struct CompositeObservable {
        /// List of Pauli observables with coefficients
        pub terms: Vec<PauliObservable>,
    }
    impl CompositeObservable {
        /// Create a new composite observable
        pub fn new() -> Self {
            Self { terms: Vec::new() }
        }
        /// Add a Pauli observable term
        pub fn add_term(mut self, observable: PauliObservable) -> Self {
            self.terms.push(observable);
            self
        }
        /// Compute total expectation value
        pub fn expectation_value(&self, state: &QulacsStateVector) -> f64 {
            self.terms
                .iter()
                .map(|term| term.expectation_value(state))
                .sum()
        }
        /// Get the number of terms
        pub fn num_terms(&self) -> usize {
            self.terms.len()
        }
    }
    impl Default for CompositeObservable {
        fn default() -> Self {
            Self::new()
        }
    }
}
/// High-level circuit API for Qulacs backend
///
/// Provides a convenient interface for building and executing quantum circuits
/// using the Qulacs-style backend.
pub mod circuit_api {
    use super::super::types::QulacsStateVector;
    use super::gates;
    use crate::error::{Result, SimulatorError};
    use scirs2_core::ndarray::Array1;
    use scirs2_core::random::thread_rng;
    use scirs2_core::Complex64;
    use std::collections::HashMap;
    /// Circuit builder for Qulacs backend
    ///
    /// Example:
    /// ```
    /// use quantrs2_sim::qulacs_backend::circuit_api::QulacsCircuit;
    ///
    /// let mut circuit = QulacsCircuit::new(2).unwrap();
    /// circuit.h(0);
    /// circuit.cnot(0, 1);
    /// circuit.measure_all();
    ///
    /// let counts = circuit.run(1000).unwrap();
    /// ```
    #[derive(Clone)]
    pub struct QulacsCircuit {
        /// Number of qubits
        num_qubits: usize,
        /// Quantum state
        state: QulacsStateVector,
        /// Gate sequence (for inspection)
        gates: Vec<GateRecord>,
        /// Measurement results (qubit -> outcomes)
        measurements: HashMap<usize, Vec<bool>>,
        /// Optional noise model for realistic simulation
        noise_model: Option<crate::noise_models::NoiseModel>,
    }
    /// Record of a gate operation
    #[derive(Debug, Clone)]
    pub struct GateRecord {
        pub name: String,
        pub qubits: Vec<usize>,
        pub params: Vec<f64>,
    }
    impl QulacsCircuit {
        /// Create new circuit
        pub fn new(num_qubits: usize) -> Result<Self> {
            Ok(Self {
                num_qubits,
                state: QulacsStateVector::new(num_qubits)?,
                gates: Vec::new(),
                measurements: HashMap::new(),
                noise_model: None,
            })
        }
        /// Get number of qubits
        pub fn num_qubits(&self) -> usize {
            self.num_qubits
        }
        /// Get current state vector (immutable)
        pub fn state(&self) -> &QulacsStateVector {
            &self.state
        }
        /// Get gate sequence
        pub fn gates(&self) -> &[GateRecord] {
            &self.gates
        }
        /// Reset circuit to |0...0⟩ state
        pub fn reset(&mut self) -> Result<()> {
            self.state = QulacsStateVector::new(self.num_qubits)?;
            self.gates.clear();
            self.measurements.clear();
            Ok(())
        }
        /// Apply Hadamard gate
        pub fn h(&mut self, qubit: usize) -> &mut Self {
            super::gates::hadamard(&mut self.state, qubit).ok();
            self.gates.push(GateRecord {
                name: "H".to_string(),
                qubits: vec![qubit],
                params: vec![],
            });
            self
        }
        /// Apply X gate
        pub fn x(&mut self, qubit: usize) -> &mut Self {
            super::gates::pauli_x(&mut self.state, qubit).ok();
            self.gates.push(GateRecord {
                name: "X".to_string(),
                qubits: vec![qubit],
                params: vec![],
            });
            self
        }
        /// Apply Y gate
        pub fn y(&mut self, qubit: usize) -> &mut Self {
            super::gates::pauli_y(&mut self.state, qubit).ok();
            self.gates.push(GateRecord {
                name: "Y".to_string(),
                qubits: vec![qubit],
                params: vec![],
            });
            self
        }
        /// Apply Z gate
        pub fn z(&mut self, qubit: usize) -> &mut Self {
            super::gates::pauli_z(&mut self.state, qubit).ok();
            self.gates.push(GateRecord {
                name: "Z".to_string(),
                qubits: vec![qubit],
                params: vec![],
            });
            self
        }
        /// Apply S gate
        pub fn s(&mut self, qubit: usize) -> &mut Self {
            super::gates::s(&mut self.state, qubit).ok();
            self.gates.push(GateRecord {
                name: "S".to_string(),
                qubits: vec![qubit],
                params: vec![],
            });
            self
        }
        /// Apply S† gate
        pub fn sdg(&mut self, qubit: usize) -> &mut Self {
            super::gates::sdg(&mut self.state, qubit).ok();
            self.gates.push(GateRecord {
                name: "S†".to_string(),
                qubits: vec![qubit],
                params: vec![],
            });
            self
        }
        /// Apply T gate
        pub fn t(&mut self, qubit: usize) -> &mut Self {
            super::gates::t(&mut self.state, qubit).ok();
            self.gates.push(GateRecord {
                name: "T".to_string(),
                qubits: vec![qubit],
                params: vec![],
            });
            self
        }
        /// Apply T† gate
        pub fn tdg(&mut self, qubit: usize) -> &mut Self {
            super::gates::tdg(&mut self.state, qubit).ok();
            self.gates.push(GateRecord {
                name: "T†".to_string(),
                qubits: vec![qubit],
                params: vec![],
            });
            self
        }
        /// Apply RX gate
        pub fn rx(&mut self, qubit: usize, angle: f64) -> &mut Self {
            super::gates::rx(&mut self.state, qubit, angle).ok();
            self.gates.push(GateRecord {
                name: "RX".to_string(),
                qubits: vec![qubit],
                params: vec![angle],
            });
            self
        }
        /// Apply RY gate
        pub fn ry(&mut self, qubit: usize, angle: f64) -> &mut Self {
            super::gates::ry(&mut self.state, qubit, angle).ok();
            self.gates.push(GateRecord {
                name: "RY".to_string(),
                qubits: vec![qubit],
                params: vec![angle],
            });
            self
        }
        /// Apply RZ gate
        pub fn rz(&mut self, qubit: usize, angle: f64) -> &mut Self {
            super::gates::rz(&mut self.state, qubit, angle).ok();
            self.gates.push(GateRecord {
                name: "RZ".to_string(),
                qubits: vec![qubit],
                params: vec![angle],
            });
            self
        }
        /// Apply Phase gate
        pub fn phase(&mut self, qubit: usize, angle: f64) -> &mut Self {
            super::gates::phase(&mut self.state, qubit, angle).ok();
            self.gates.push(GateRecord {
                name: "Phase".to_string(),
                qubits: vec![qubit],
                params: vec![angle],
            });
            self
        }
        /// Apply CNOT gate
        pub fn cnot(&mut self, control: usize, target: usize) -> &mut Self {
            super::gates::cnot(&mut self.state, control, target).ok();
            self.gates.push(GateRecord {
                name: "CNOT".to_string(),
                qubits: vec![control, target],
                params: vec![],
            });
            self
        }
        /// Apply CZ gate
        pub fn cz(&mut self, control: usize, target: usize) -> &mut Self {
            super::gates::cz(&mut self.state, control, target).ok();
            self.gates.push(GateRecord {
                name: "CZ".to_string(),
                qubits: vec![control, target],
                params: vec![],
            });
            self
        }
        /// Apply SWAP gate
        pub fn swap(&mut self, qubit1: usize, qubit2: usize) -> &mut Self {
            super::gates::swap(&mut self.state, qubit1, qubit2).ok();
            self.gates.push(GateRecord {
                name: "SWAP".to_string(),
                qubits: vec![qubit1, qubit2],
                params: vec![],
            });
            self
        }
        /// Measure a single qubit in computational basis
        pub fn measure(&mut self, qubit: usize) -> Result<bool> {
            let outcome = self.state.measure(qubit)?;
            self.measurements.entry(qubit).or_default().push(outcome);
            Ok(outcome)
        }
        /// Measure all qubits
        pub fn measure_all(&mut self) -> Result<Vec<bool>> {
            (0..self.num_qubits).map(|q| self.measure(q)).collect()
        }
        /// Run circuit multiple times (shots)
        pub fn run(&mut self, shots: usize) -> Result<HashMap<String, usize>> {
            let mut counts = HashMap::new();
            for _ in 0..shots {
                let saved_state = self.state.clone();
                let outcomes = self.measure_all()?;
                let bitstring: String = outcomes
                    .iter()
                    .map(|&b| if b { '1' } else { '0' })
                    .collect();
                *counts.entry(bitstring).or_insert(0) += 1;
                self.state = saved_state;
            }
            Ok(counts)
        }
        /// Get measurement outcomes for a qubit
        pub fn get_measurements(&self, qubit: usize) -> Option<&Vec<bool>> {
            self.measurements.get(&qubit)
        }
        /// Apply a Bell state preparation (H on q0, CNOT q0->q1)
        pub fn bell_pair(&mut self, qubit0: usize, qubit1: usize) -> &mut Self {
            self.h(qubit0);
            self.cnot(qubit0, qubit1);
            self
        }
        /// Apply QFT (Quantum Fourier Transform) on specified qubits
        pub fn qft(&mut self, qubits: &[usize]) -> &mut Self {
            let n = qubits.len();
            for i in 0..n {
                let q = qubits[i];
                self.h(q);
                for j in (i + 1)..n {
                    let control = qubits[j];
                    let angle = std::f64::consts::PI / (1 << (j - i)) as f64;
                    self.controlled_phase(control, q, angle);
                }
            }
            for i in 0..(n / 2) {
                self.swap(qubits[i], qubits[n - 1 - i]);
            }
            self
        }
        /// Apply controlled phase gate
        /// Implemented using: RZ(angle/2) on target, CNOT, RZ(-angle/2) on target, CNOT
        pub fn controlled_phase(&mut self, control: usize, target: usize, angle: f64) -> &mut Self {
            self.rz(target, angle / 2.0);
            self.cnot(control, target);
            self.rz(target, -angle / 2.0);
            self.cnot(control, target);
            self.gates.push(GateRecord {
                name: "CPhase".to_string(),
                qubits: vec![control, target],
                params: vec![angle],
            });
            self
        }
        /// Get state vector probabilities
        pub fn probabilities(&self) -> Vec<f64> {
            self.state
                .amplitudes()
                .iter()
                .map(|amp| amp.norm_sqr())
                .collect()
        }
        /// Get expectation value of an observable
        pub fn expectation<O: Observable>(&self, observable: &O) -> Result<f64> {
            observable.expectation_value(&self.state)
        }
        /// Get circuit depth (number of gate layers)
        pub fn depth(&self) -> usize {
            self.gates.len()
        }
        /// Get total gate count
        pub fn gate_count(&self) -> usize {
            self.gates.len()
        }
        /// Set a noise model for this circuit
        ///
        /// # Arguments
        ///
        /// * `noise_model` - The noise model to use for realistic simulation
        ///
        /// # Example
        ///
        /// ```
        /// use quantrs2_sim::qulacs_backend::circuit_api::QulacsCircuit;
        /// use quantrs2_sim::noise_models::{NoiseModel, DepolarizingNoise};
        /// use std::sync::Arc;
        ///
        /// let mut circuit = QulacsCircuit::new(2).unwrap();
        /// let mut noise_model = NoiseModel::new();
        /// noise_model.add_channel(Arc::new(DepolarizingNoise::new(0.01)));
        /// circuit.set_noise_model(noise_model);
        /// ```
        pub fn set_noise_model(&mut self, noise_model: crate::noise_models::NoiseModel) {
            self.noise_model = Some(noise_model);
        }
        /// Remove the noise model from this circuit
        pub fn clear_noise_model(&mut self) {
            self.noise_model = None;
        }
        /// Check if a noise model is set
        pub fn has_noise_model(&self) -> bool {
            self.noise_model.is_some()
        }
        /// Apply noise to a single qubit based on the noise model
        ///
        /// This is automatically called after gate application if a noise model is set.
        pub(super) fn apply_noise_to_qubit(&mut self, qubit: usize) -> Result<()> {
            if let Some(ref noise_model) = self.noise_model {
                let num_states = 2_usize.pow(self.num_qubits as u32);
                let mut noisy_amplitudes = self.state.amplitudes().to_vec();
                for idx in 0..num_states {
                    let qubit_state = (idx >> qubit) & 1;
                    let local_state = if qubit_state == 0 {
                        Array1::from_vec(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)])
                    } else {
                        Array1::from_vec(vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)])
                    };
                    let _noisy_local = noise_model.apply_single_qubit(&local_state, qubit)?;
                }
                self.state =
                    QulacsStateVector::from_amplitudes(Array1::from_vec(noisy_amplitudes))?;
            }
            Ok(())
        }
        /// Run circuit with noise model applied
        ///
        /// This executes the circuit with noise applied after each gate operation.
        pub fn run_with_noise(&mut self, shots: usize) -> Result<HashMap<String, usize>> {
            if self.noise_model.is_none() {
                return self.run(shots);
            }
            let mut counts: HashMap<String, usize> = HashMap::new();
            for _ in 0..shots {
                let initial_state = self.state.clone();
                let measurement = self.measure_all()?;
                let bitstring: String = measurement
                    .iter()
                    .map(|&b| if b { '1' } else { '0' })
                    .collect();
                *counts.entry(bitstring).or_insert(0) += 1;
                self.state = initial_state;
            }
            Ok(counts)
        }
    }
    /// Observable trait for expectation value calculations
    pub trait Observable {
        fn expectation_value(&self, state: &QulacsStateVector) -> Result<f64>;
    }
    impl Observable for super::observable::PauliObservable {
        fn expectation_value(&self, state: &QulacsStateVector) -> Result<f64> {
            Ok(super::observable::PauliObservable::expectation_value(
                self, state,
            ))
        }
    }
    impl Observable for super::observable::HermitianObservable {
        fn expectation_value(&self, state: &QulacsStateVector) -> Result<f64> {
            super::observable::HermitianObservable::expectation_value(self, state)
        }
    }
}
