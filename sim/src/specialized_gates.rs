//! Specialized gate implementations for simulation
//!
//! This module provides optimized implementations of common quantum gates
//! that take advantage of their specific structure for improved performance.
//! These implementations avoid general matrix multiplication and directly
//! manipulate state vector amplitudes.

use num_complex::Complex64;
use scirs2_core::parallel_ops::*;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};

/// Trait for specialized gate implementations
pub trait SpecializedGate: GateOp {
    /// Apply the gate directly to a state vector (optimized implementation)
    fn apply_specialized(
        &self,
        state: &mut [Complex64],
        n_qubits: usize,
        parallel: bool,
    ) -> QuantRS2Result<()>;

    /// Check if this gate can be fused with another gate
    fn can_fuse_with(&self, other: &dyn SpecializedGate) -> bool {
        false
    }

    /// Fuse this gate with another gate if possible
    fn fuse_with(&self, other: &dyn SpecializedGate) -> Option<Box<dyn SpecializedGate>> {
        None
    }
}

// ============= Single-Qubit Gates =============

/// Specialized Hadamard gate
#[derive(Debug, Clone, Copy)]
pub struct HadamardSpecialized {
    pub target: QubitId,
}

impl SpecializedGate for HadamardSpecialized {
    fn apply_specialized(
        &self,
        state: &mut [Complex64],
        n_qubits: usize,
        parallel: bool,
    ) -> QuantRS2Result<()> {
        let target_idx = self.target.id() as usize;
        if target_idx >= n_qubits {
            return Err(QuantRS2Error::InvalidQubitId(self.target.id()));
        }

        let sqrt2_inv = 1.0 / std::f64::consts::SQRT_2;

        if parallel {
            let state_copy = state.to_vec();
            state.par_iter_mut().enumerate().for_each(|(idx, amp)| {
                let bit_val = (idx >> target_idx) & 1;
                let paired_idx = idx ^ (1 << target_idx);

                let val0 = if bit_val == 0 {
                    state_copy[idx]
                } else {
                    state_copy[paired_idx]
                };
                let val1 = if bit_val == 0 {
                    state_copy[paired_idx]
                } else {
                    state_copy[idx]
                };

                *amp = sqrt2_inv
                    * if bit_val == 0 {
                        val0 + val1
                    } else {
                        val0 - val1
                    };
            });
        } else {
            for i in 0..(1 << n_qubits) {
                if (i >> target_idx) & 1 == 0 {
                    let j = i | (1 << target_idx);
                    let temp0 = state[i];
                    let temp1 = state[j];
                    state[i] = sqrt2_inv * (temp0 + temp1);
                    state[j] = sqrt2_inv * (temp0 - temp1);
                }
            }
        }

        Ok(())
    }
}

/// Specialized Pauli-X gate
#[derive(Debug, Clone, Copy)]
pub struct PauliXSpecialized {
    pub target: QubitId,
}

impl SpecializedGate for PauliXSpecialized {
    fn apply_specialized(
        &self,
        state: &mut [Complex64],
        n_qubits: usize,
        parallel: bool,
    ) -> QuantRS2Result<()> {
        let target_idx = self.target.id() as usize;
        if target_idx >= n_qubits {
            return Err(QuantRS2Error::InvalidQubitId(self.target.id()));
        }

        if parallel {
            let state_copy = state.to_vec();
            state.par_iter_mut().enumerate().for_each(|(idx, amp)| {
                let flipped_idx = idx ^ (1 << target_idx);
                *amp = state_copy[flipped_idx];
            });
        } else {
            for i in 0..(1 << n_qubits) {
                if (i >> target_idx) & 1 == 0 {
                    let j = i | (1 << target_idx);
                    state.swap(i, j);
                }
            }
        }

        Ok(())
    }
}

/// Specialized Pauli-Y gate
#[derive(Debug, Clone, Copy)]
pub struct PauliYSpecialized {
    pub target: QubitId,
}

impl SpecializedGate for PauliYSpecialized {
    fn apply_specialized(
        &self,
        state: &mut [Complex64],
        n_qubits: usize,
        parallel: bool,
    ) -> QuantRS2Result<()> {
        let target_idx = self.target.id() as usize;
        if target_idx >= n_qubits {
            return Err(QuantRS2Error::InvalidQubitId(self.target.id()));
        }

        let i_unit = Complex64::new(0.0, 1.0);

        if parallel {
            let state_copy = state.to_vec();
            state.par_iter_mut().enumerate().for_each(|(idx, amp)| {
                let bit_val = (idx >> target_idx) & 1;
                let flipped_idx = idx ^ (1 << target_idx);
                *amp = if bit_val == 0 {
                    i_unit * state_copy[flipped_idx]
                } else {
                    -i_unit * state_copy[flipped_idx]
                };
            });
        } else {
            for i in 0..(1 << n_qubits) {
                if (i >> target_idx) & 1 == 0 {
                    let j = i | (1 << target_idx);
                    let temp0 = state[i];
                    let temp1 = state[j];
                    state[i] = i_unit * temp1;
                    state[j] = -i_unit * temp0;
                }
            }
        }

        Ok(())
    }
}

/// Specialized Pauli-Z gate
#[derive(Debug, Clone, Copy)]
pub struct PauliZSpecialized {
    pub target: QubitId,
}

impl SpecializedGate for PauliZSpecialized {
    fn apply_specialized(
        &self,
        state: &mut [Complex64],
        n_qubits: usize,
        parallel: bool,
    ) -> QuantRS2Result<()> {
        let target_idx = self.target.id() as usize;
        if target_idx >= n_qubits {
            return Err(QuantRS2Error::InvalidQubitId(self.target.id()));
        }

        if parallel {
            state.par_iter_mut().enumerate().for_each(|(idx, amp)| {
                if (idx >> target_idx) & 1 == 1 {
                    *amp = -*amp;
                }
            });
        } else {
            for i in 0..(1 << n_qubits) {
                if (i >> target_idx) & 1 == 1 {
                    state[i] = -state[i];
                }
            }
        }

        Ok(())
    }
}

/// Specialized phase gate
#[derive(Debug, Clone, Copy)]
pub struct PhaseSpecialized {
    pub target: QubitId,
    pub phase: f64,
}

impl SpecializedGate for PhaseSpecialized {
    fn apply_specialized(
        &self,
        state: &mut [Complex64],
        n_qubits: usize,
        parallel: bool,
    ) -> QuantRS2Result<()> {
        let target_idx = self.target.id() as usize;
        if target_idx >= n_qubits {
            return Err(QuantRS2Error::InvalidQubitId(self.target.id()));
        }

        let phase_factor = Complex64::from_polar(1.0, self.phase);

        if parallel {
            state.par_iter_mut().enumerate().for_each(|(idx, amp)| {
                if (idx >> target_idx) & 1 == 1 {
                    *amp *= phase_factor;
                }
            });
        } else {
            for i in 0..(1 << n_qubits) {
                if (i >> target_idx) & 1 == 1 {
                    state[i] *= phase_factor;
                }
            }
        }

        Ok(())
    }
}

/// Specialized S gate (√Z)
#[derive(Debug, Clone, Copy)]
pub struct SGateSpecialized {
    pub target: QubitId,
}

impl SpecializedGate for SGateSpecialized {
    fn apply_specialized(
        &self,
        state: &mut [Complex64],
        n_qubits: usize,
        parallel: bool,
    ) -> QuantRS2Result<()> {
        let phase_gate = PhaseSpecialized {
            target: self.target,
            phase: FRAC_PI_2,
        };
        phase_gate.apply_specialized(state, n_qubits, parallel)
    }
}

/// Specialized T gate (4th root of Z)
#[derive(Debug, Clone, Copy)]
pub struct TGateSpecialized {
    pub target: QubitId,
}

impl SpecializedGate for TGateSpecialized {
    fn apply_specialized(
        &self,
        state: &mut [Complex64],
        n_qubits: usize,
        parallel: bool,
    ) -> QuantRS2Result<()> {
        let phase_gate = PhaseSpecialized {
            target: self.target,
            phase: FRAC_PI_4,
        };
        phase_gate.apply_specialized(state, n_qubits, parallel)
    }
}

/// Specialized RX rotation gate
#[derive(Debug, Clone, Copy)]
pub struct RXSpecialized {
    pub target: QubitId,
    pub theta: f64,
}

impl SpecializedGate for RXSpecialized {
    fn apply_specialized(
        &self,
        state: &mut [Complex64],
        n_qubits: usize,
        parallel: bool,
    ) -> QuantRS2Result<()> {
        let target_idx = self.target.id() as usize;
        if target_idx >= n_qubits {
            return Err(QuantRS2Error::InvalidQubitId(self.target.id()));
        }

        let cos_half = (self.theta / 2.0).cos();
        let sin_half = (self.theta / 2.0).sin();
        let i_sin = Complex64::new(0.0, -sin_half);

        if parallel {
            let state_copy = state.to_vec();
            state.par_iter_mut().enumerate().for_each(|(idx, amp)| {
                let bit_val = (idx >> target_idx) & 1;
                let paired_idx = idx ^ (1 << target_idx);

                let val0 = if bit_val == 0 {
                    state_copy[idx]
                } else {
                    state_copy[paired_idx]
                };
                let val1 = if bit_val == 0 {
                    state_copy[paired_idx]
                } else {
                    state_copy[idx]
                };

                *amp = if bit_val == 0 {
                    cos_half * val0 + i_sin * val1
                } else {
                    i_sin * val0 + cos_half * val1
                };
            });
        } else {
            for i in 0..(1 << n_qubits) {
                if (i >> target_idx) & 1 == 0 {
                    let j = i | (1 << target_idx);
                    let temp0 = state[i];
                    let temp1 = state[j];
                    state[i] = cos_half * temp0 + i_sin * temp1;
                    state[j] = i_sin * temp0 + cos_half * temp1;
                }
            }
        }

        Ok(())
    }
}

/// Specialized RY rotation gate
#[derive(Debug, Clone, Copy)]
pub struct RYSpecialized {
    pub target: QubitId,
    pub theta: f64,
}

impl SpecializedGate for RYSpecialized {
    fn apply_specialized(
        &self,
        state: &mut [Complex64],
        n_qubits: usize,
        parallel: bool,
    ) -> QuantRS2Result<()> {
        let target_idx = self.target.id() as usize;
        if target_idx >= n_qubits {
            return Err(QuantRS2Error::InvalidQubitId(self.target.id()));
        }

        let cos_half = (self.theta / 2.0).cos();
        let sin_half = (self.theta / 2.0).sin();

        if parallel {
            let state_copy = state.to_vec();
            state.par_iter_mut().enumerate().for_each(|(idx, amp)| {
                let bit_val = (idx >> target_idx) & 1;
                let paired_idx = idx ^ (1 << target_idx);

                let val0 = if bit_val == 0 {
                    state_copy[idx]
                } else {
                    state_copy[paired_idx]
                };
                let val1 = if bit_val == 0 {
                    state_copy[paired_idx]
                } else {
                    state_copy[idx]
                };

                *amp = if bit_val == 0 {
                    cos_half * val0 - sin_half * val1
                } else {
                    sin_half * val0 + cos_half * val1
                };
            });
        } else {
            for i in 0..(1 << n_qubits) {
                if (i >> target_idx) & 1 == 0 {
                    let j = i | (1 << target_idx);
                    let temp0 = state[i];
                    let temp1 = state[j];
                    state[i] = cos_half * temp0 - sin_half * temp1;
                    state[j] = sin_half * temp0 + cos_half * temp1;
                }
            }
        }

        Ok(())
    }
}

/// Specialized RZ rotation gate
#[derive(Debug, Clone, Copy)]
pub struct RZSpecialized {
    pub target: QubitId,
    pub theta: f64,
}

impl SpecializedGate for RZSpecialized {
    fn apply_specialized(
        &self,
        state: &mut [Complex64],
        n_qubits: usize,
        parallel: bool,
    ) -> QuantRS2Result<()> {
        let target_idx = self.target.id() as usize;
        if target_idx >= n_qubits {
            return Err(QuantRS2Error::InvalidQubitId(self.target.id()));
        }

        let phase_0 = Complex64::from_polar(1.0, -self.theta / 2.0);
        let phase_1 = Complex64::from_polar(1.0, self.theta / 2.0);

        if parallel {
            state.par_iter_mut().enumerate().for_each(|(idx, amp)| {
                if (idx >> target_idx) & 1 == 0 {
                    *amp *= phase_0;
                } else {
                    *amp *= phase_1;
                }
            });
        } else {
            for i in 0..(1 << n_qubits) {
                if (i >> target_idx) & 1 == 0 {
                    state[i] *= phase_0;
                } else {
                    state[i] *= phase_1;
                }
            }
        }

        Ok(())
    }
}

// ============= Two-Qubit Gates =============

/// Specialized CNOT gate
#[derive(Debug, Clone, Copy)]
pub struct CNOTSpecialized {
    pub control: QubitId,
    pub target: QubitId,
}

impl SpecializedGate for CNOTSpecialized {
    fn apply_specialized(
        &self,
        state: &mut [Complex64],
        n_qubits: usize,
        parallel: bool,
    ) -> QuantRS2Result<()> {
        let control_idx = self.control.id() as usize;
        let target_idx = self.target.id() as usize;

        if control_idx >= n_qubits || target_idx >= n_qubits {
            return Err(QuantRS2Error::InvalidQubitId(if control_idx >= n_qubits {
                self.control.id()
            } else {
                self.target.id()
            }));
        }

        if control_idx == target_idx {
            return Err(QuantRS2Error::CircuitValidationFailed(
                "Control and target qubits must be different".into(),
            ));
        }

        if parallel {
            let state_copy = state.to_vec();
            state.par_iter_mut().enumerate().for_each(|(idx, amp)| {
                if (idx >> control_idx) & 1 == 1 {
                    let flipped_idx = idx ^ (1 << target_idx);
                    *amp = state_copy[flipped_idx];
                }
            });
        } else {
            for i in 0..(1 << n_qubits) {
                if (i >> control_idx) & 1 == 1 && (i >> target_idx) & 1 == 0 {
                    let j = i | (1 << target_idx);
                    state.swap(i, j);
                }
            }
        }

        Ok(())
    }

    fn can_fuse_with(&self, other: &dyn SpecializedGate) -> bool {
        // Two CNOTs with same control and target cancel out
        if let Some(other_cnot) = other.as_any().downcast_ref::<CNOTSpecialized>() {
            self.control == other_cnot.control && self.target == other_cnot.target
        } else {
            false
        }
    }
}

/// Specialized CZ gate
#[derive(Debug, Clone, Copy)]
pub struct CZSpecialized {
    pub control: QubitId,
    pub target: QubitId,
}

impl SpecializedGate for CZSpecialized {
    fn apply_specialized(
        &self,
        state: &mut [Complex64],
        n_qubits: usize,
        parallel: bool,
    ) -> QuantRS2Result<()> {
        let control_idx = self.control.id() as usize;
        let target_idx = self.target.id() as usize;

        if control_idx >= n_qubits || target_idx >= n_qubits {
            return Err(QuantRS2Error::InvalidQubitId(if control_idx >= n_qubits {
                self.control.id()
            } else {
                self.target.id()
            }));
        }

        if control_idx == target_idx {
            return Err(QuantRS2Error::CircuitValidationFailed(
                "Control and target qubits must be different".into(),
            ));
        }

        if parallel {
            state.par_iter_mut().enumerate().for_each(|(idx, amp)| {
                if (idx >> control_idx) & 1 == 1 && (idx >> target_idx) & 1 == 1 {
                    *amp = -*amp;
                }
            });
        } else {
            for i in 0..(1 << n_qubits) {
                if (i >> control_idx) & 1 == 1 && (i >> target_idx) & 1 == 1 {
                    state[i] = -state[i];
                }
            }
        }

        Ok(())
    }
}

/// Specialized SWAP gate
#[derive(Debug, Clone, Copy)]
pub struct SWAPSpecialized {
    pub qubit1: QubitId,
    pub qubit2: QubitId,
}

impl SpecializedGate for SWAPSpecialized {
    fn apply_specialized(
        &self,
        state: &mut [Complex64],
        n_qubits: usize,
        parallel: bool,
    ) -> QuantRS2Result<()> {
        let idx1 = self.qubit1.id() as usize;
        let idx2 = self.qubit2.id() as usize;

        if idx1 >= n_qubits || idx2 >= n_qubits {
            return Err(QuantRS2Error::InvalidQubitId(if idx1 >= n_qubits {
                self.qubit1.id()
            } else {
                self.qubit2.id()
            }));
        }

        if idx1 == idx2 {
            return Ok(()); // SWAP with itself is identity
        }

        if parallel {
            let state_copy = state.to_vec();
            state.par_iter_mut().enumerate().for_each(|(idx, amp)| {
                let bit1 = (idx >> idx1) & 1;
                let bit2 = (idx >> idx2) & 1;

                if bit1 != bit2 {
                    let swapped_idx =
                        (idx & !(1 << idx1) & !(1 << idx2)) | (bit2 << idx1) | (bit1 << idx2);
                    *amp = state_copy[swapped_idx];
                }
            });
        } else {
            for i in 0..(1 << n_qubits) {
                let bit1 = (i >> idx1) & 1;
                let bit2 = (i >> idx2) & 1;

                if bit1 == 0 && bit2 == 1 {
                    let j = (i | (1 << idx1)) & !(1 << idx2);
                    state.swap(i, j);
                }
            }
        }

        Ok(())
    }
}

/// Specialized controlled phase gate
#[derive(Debug, Clone, Copy)]
pub struct CPhaseSpecialized {
    pub control: QubitId,
    pub target: QubitId,
    pub phase: f64,
}

impl SpecializedGate for CPhaseSpecialized {
    fn apply_specialized(
        &self,
        state: &mut [Complex64],
        n_qubits: usize,
        parallel: bool,
    ) -> QuantRS2Result<()> {
        let control_idx = self.control.id() as usize;
        let target_idx = self.target.id() as usize;

        if control_idx >= n_qubits || target_idx >= n_qubits {
            return Err(QuantRS2Error::InvalidQubitId(if control_idx >= n_qubits {
                self.control.id()
            } else {
                self.target.id()
            }));
        }

        if control_idx == target_idx {
            return Err(QuantRS2Error::CircuitValidationFailed(
                "Control and target qubits must be different".into(),
            ));
        }

        let phase_factor = Complex64::from_polar(1.0, self.phase);

        if parallel {
            state.par_iter_mut().enumerate().for_each(|(idx, amp)| {
                if (idx >> control_idx) & 1 == 1 && (idx >> target_idx) & 1 == 1 {
                    *amp *= phase_factor;
                }
            });
        } else {
            for i in 0..(1 << n_qubits) {
                if (i >> control_idx) & 1 == 1 && (i >> target_idx) & 1 == 1 {
                    state[i] *= phase_factor;
                }
            }
        }

        Ok(())
    }
}

// ============= Multi-Qubit Gates =============

/// Specialized Toffoli (CCX) gate
#[derive(Debug, Clone, Copy)]
pub struct ToffoliSpecialized {
    pub control1: QubitId,
    pub control2: QubitId,
    pub target: QubitId,
}

impl SpecializedGate for ToffoliSpecialized {
    fn apply_specialized(
        &self,
        state: &mut [Complex64],
        n_qubits: usize,
        parallel: bool,
    ) -> QuantRS2Result<()> {
        let ctrl1_idx = self.control1.id() as usize;
        let ctrl2_idx = self.control2.id() as usize;
        let target_idx = self.target.id() as usize;

        if ctrl1_idx >= n_qubits || ctrl2_idx >= n_qubits || target_idx >= n_qubits {
            return Err(QuantRS2Error::InvalidQubitId(if ctrl1_idx >= n_qubits {
                self.control1.id()
            } else if ctrl2_idx >= n_qubits {
                self.control2.id()
            } else {
                self.target.id()
            }));
        }

        if ctrl1_idx == ctrl2_idx || ctrl1_idx == target_idx || ctrl2_idx == target_idx {
            return Err(QuantRS2Error::CircuitValidationFailed(
                "All qubits must be different".into(),
            ));
        }

        if parallel {
            let state_copy = state.to_vec();
            state.par_iter_mut().enumerate().for_each(|(idx, amp)| {
                if (idx >> ctrl1_idx) & 1 == 1 && (idx >> ctrl2_idx) & 1 == 1 {
                    let flipped_idx = idx ^ (1 << target_idx);
                    *amp = state_copy[flipped_idx];
                }
            });
        } else {
            for i in 0..(1 << n_qubits) {
                if (i >> ctrl1_idx) & 1 == 1
                    && (i >> ctrl2_idx) & 1 == 1
                    && (i >> target_idx) & 1 == 0
                {
                    let j = i | (1 << target_idx);
                    state.swap(i, j);
                }
            }
        }

        Ok(())
    }
}

/// Specialized Fredkin (CSWAP) gate
#[derive(Debug, Clone, Copy)]
pub struct FredkinSpecialized {
    pub control: QubitId,
    pub target1: QubitId,
    pub target2: QubitId,
}

impl SpecializedGate for FredkinSpecialized {
    fn apply_specialized(
        &self,
        state: &mut [Complex64],
        n_qubits: usize,
        parallel: bool,
    ) -> QuantRS2Result<()> {
        let ctrl_idx = self.control.id() as usize;
        let tgt1_idx = self.target1.id() as usize;
        let tgt2_idx = self.target2.id() as usize;

        if ctrl_idx >= n_qubits || tgt1_idx >= n_qubits || tgt2_idx >= n_qubits {
            return Err(QuantRS2Error::InvalidQubitId(if ctrl_idx >= n_qubits {
                self.control.id()
            } else if tgt1_idx >= n_qubits {
                self.target1.id()
            } else {
                self.target2.id()
            }));
        }

        if ctrl_idx == tgt1_idx || ctrl_idx == tgt2_idx || tgt1_idx == tgt2_idx {
            return Err(QuantRS2Error::CircuitValidationFailed(
                "All qubits must be different".into(),
            ));
        }

        if parallel {
            let state_copy = state.to_vec();
            state.par_iter_mut().enumerate().for_each(|(idx, amp)| {
                if (idx >> ctrl_idx) & 1 == 1 {
                    let bit1 = (idx >> tgt1_idx) & 1;
                    let bit2 = (idx >> tgt2_idx) & 1;

                    if bit1 != bit2 {
                        let swapped_idx = (idx & !(1 << tgt1_idx) & !(1 << tgt2_idx))
                            | (bit2 << tgt1_idx)
                            | (bit1 << tgt2_idx);
                        *amp = state_copy[swapped_idx];
                    }
                }
            });
        } else {
            for i in 0..(1 << n_qubits) {
                if (i >> ctrl_idx) & 1 == 1 {
                    let bit1 = (i >> tgt1_idx) & 1;
                    let bit2 = (i >> tgt2_idx) & 1;

                    if bit1 == 0 && bit2 == 1 {
                        let j = (i | (1 << tgt1_idx)) & !(1 << tgt2_idx);
                        state.swap(i, j);
                    }
                }
            }
        }

        Ok(())
    }
}

// ============= Helper Functions =============

/// Convert a general gate to its specialized implementation if available
pub fn specialize_gate(gate: &dyn GateOp) -> Option<Box<dyn SpecializedGate>> {
    use quantrs2_core::gate::{multi::*, single::*};
    use std::any::Any;

    // Try single-qubit gates
    if let Some(h) = gate.as_any().downcast_ref::<Hadamard>() {
        return Some(Box::new(HadamardSpecialized { target: h.target }));
    }
    if let Some(x) = gate.as_any().downcast_ref::<PauliX>() {
        return Some(Box::new(PauliXSpecialized { target: x.target }));
    }
    if let Some(y) = gate.as_any().downcast_ref::<PauliY>() {
        return Some(Box::new(PauliYSpecialized { target: y.target }));
    }
    if let Some(z) = gate.as_any().downcast_ref::<PauliZ>() {
        return Some(Box::new(PauliZSpecialized { target: z.target }));
    }
    if let Some(rx) = gate.as_any().downcast_ref::<RotationX>() {
        return Some(Box::new(RXSpecialized {
            target: rx.target,
            theta: rx.theta,
        }));
    }
    if let Some(ry) = gate.as_any().downcast_ref::<RotationY>() {
        return Some(Box::new(RYSpecialized {
            target: ry.target,
            theta: ry.theta,
        }));
    }
    if let Some(rz) = gate.as_any().downcast_ref::<RotationZ>() {
        return Some(Box::new(RZSpecialized {
            target: rz.target,
            theta: rz.theta,
        }));
    }
    if let Some(s) = gate.as_any().downcast_ref::<Phase>() {
        return Some(Box::new(SGateSpecialized { target: s.target }));
    }
    if let Some(t) = gate.as_any().downcast_ref::<T>() {
        return Some(Box::new(TGateSpecialized { target: t.target }));
    }

    // Try two-qubit gates
    if let Some(cnot) = gate.as_any().downcast_ref::<CNOT>() {
        return Some(Box::new(CNOTSpecialized {
            control: cnot.control,
            target: cnot.target,
        }));
    }
    if let Some(cz) = gate.as_any().downcast_ref::<CZ>() {
        return Some(Box::new(CZSpecialized {
            control: cz.control,
            target: cz.target,
        }));
    }
    if let Some(swap) = gate.as_any().downcast_ref::<SWAP>() {
        return Some(Box::new(SWAPSpecialized {
            qubit1: swap.qubit1,
            qubit2: swap.qubit2,
        }));
    }

    None
}

// Implement GateOp trait for all specialized gates

macro_rules! impl_gate_op_for_specialized {
    ($gate_type:ty, $name:expr, $qubits:expr, $matrix:expr) => {
        impl GateOp for $gate_type {
            fn name(&self) -> &'static str {
                $name
            }

            fn qubits(&self) -> Vec<QubitId> {
                $qubits(self)
            }

            fn matrix(&self) -> QuantRS2Result<Vec<Complex64>> {
                $matrix(self)
            }

            fn as_any(&self) -> &dyn Any {
                self
            }

            fn clone_gate(&self) -> Box<dyn GateOp> {
                Box::new(self.clone())
            }
        }
    };
}

// Implement GateOp for single-qubit specialized gates
impl_gate_op_for_specialized!(
    HadamardSpecialized,
    "H",
    |g: &HadamardSpecialized| vec![g.target],
    |_: &HadamardSpecialized| {
        let sqrt2_inv = 1.0 / std::f64::consts::SQRT_2;
        Ok(vec![
            Complex64::new(sqrt2_inv, 0.0),
            Complex64::new(sqrt2_inv, 0.0),
            Complex64::new(sqrt2_inv, 0.0),
            Complex64::new(-sqrt2_inv, 0.0),
        ])
    }
);

impl_gate_op_for_specialized!(
    PauliXSpecialized,
    "X",
    |g: &PauliXSpecialized| vec![g.target],
    |_: &PauliXSpecialized| Ok(vec![
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    ])
);

impl_gate_op_for_specialized!(
    PauliYSpecialized,
    "Y",
    |g: &PauliYSpecialized| vec![g.target],
    |_: &PauliYSpecialized| Ok(vec![
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, -1.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(0.0, 0.0),
    ])
);

impl_gate_op_for_specialized!(
    PauliZSpecialized,
    "Z",
    |g: &PauliZSpecialized| vec![g.target],
    |_: &PauliZSpecialized| Ok(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(-1.0, 0.0),
    ])
);

// Implement GateOp for two-qubit specialized gates
impl_gate_op_for_specialized!(
    CNOTSpecialized,
    "CNOT",
    |g: &CNOTSpecialized| vec![g.control, g.target],
    |_: &CNOTSpecialized| Ok(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    ])
);

// Implement GateOp for phase-related specialized gates
impl_gate_op_for_specialized!(
    PhaseSpecialized,
    "Phase",
    |g: &PhaseSpecialized| vec![g.target],
    |g: &PhaseSpecialized| Ok(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::from_polar(1.0, g.phase),
    ])
);

impl_gate_op_for_specialized!(
    SGateSpecialized,
    "S",
    |g: &SGateSpecialized| vec![g.target],
    |_: &SGateSpecialized| Ok(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 1.0),
    ])
);

impl_gate_op_for_specialized!(
    TGateSpecialized,
    "T",
    |g: &TGateSpecialized| vec![g.target],
    |_: &TGateSpecialized| {
        let phase = Complex64::from_polar(1.0, PI / 4.0);
        Ok(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            phase,
        ])
    }
);

impl_gate_op_for_specialized!(
    RXSpecialized,
    "RX",
    |g: &RXSpecialized| vec![g.target],
    |g: &RXSpecialized| {
        let cos = (g.theta / 2.0).cos();
        let sin = (g.theta / 2.0).sin();
        Ok(vec![
            Complex64::new(cos, 0.0),
            Complex64::new(0.0, -sin),
            Complex64::new(0.0, -sin),
            Complex64::new(cos, 0.0),
        ])
    }
);

impl_gate_op_for_specialized!(
    RYSpecialized,
    "RY",
    |g: &RYSpecialized| vec![g.target],
    |g: &RYSpecialized| {
        let cos = (g.theta / 2.0).cos();
        let sin = (g.theta / 2.0).sin();
        Ok(vec![
            Complex64::new(cos, 0.0),
            Complex64::new(-sin, 0.0),
            Complex64::new(sin, 0.0),
            Complex64::new(cos, 0.0),
        ])
    }
);

impl_gate_op_for_specialized!(
    RZSpecialized,
    "RZ",
    |g: &RZSpecialized| vec![g.target],
    |g: &RZSpecialized| {
        let phase_pos = Complex64::from_polar(1.0, g.theta / 2.0);
        let phase_neg = Complex64::from_polar(1.0, -g.theta / 2.0);
        Ok(vec![
            phase_neg,
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            phase_pos,
        ])
    }
);

impl_gate_op_for_specialized!(
    CZSpecialized,
    "CZ",
    |g: &CZSpecialized| vec![g.control, g.target],
    |_: &CZSpecialized| Ok(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(-1.0, 0.0),
    ])
);

impl_gate_op_for_specialized!(
    SWAPSpecialized,
    "SWAP",
    |g: &SWAPSpecialized| vec![g.qubit1, g.qubit2],
    |_: &SWAPSpecialized| Ok(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ])
);

// Implement GateOp for multi-qubit specialized gates
impl_gate_op_for_specialized!(
    CPhaseSpecialized,
    "CPhase",
    |g: &CPhaseSpecialized| vec![g.control, g.target],
    |g: &CPhaseSpecialized| {
        let phase = Complex64::from_polar(1.0, g.phase);
        Ok(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            phase,
        ])
    }
);

impl_gate_op_for_specialized!(
    ToffoliSpecialized,
    "Toffoli",
    |g: &ToffoliSpecialized| vec![g.control1, g.control2, g.target],
    |_: &ToffoliSpecialized| Ok(vec![
        // 8x8 Toffoli matrix
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    ])
);

impl_gate_op_for_specialized!(
    FredkinSpecialized,
    "Fredkin",
    |g: &FredkinSpecialized| vec![g.control, g.target1, g.target2],
    |_: &FredkinSpecialized| Ok(vec![
        // 8x8 Fredkin (controlled-SWAP) matrix
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ])
);

use std::any::Any;

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn test_hadamard_specialized() {
        let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let gate = HadamardSpecialized { target: QubitId(0) };

        gate.apply_specialized(&mut state, 1, false).unwrap();

        let sqrt2_inv = 1.0 / std::f64::consts::SQRT_2;
        assert!((state[0] - Complex64::new(sqrt2_inv, 0.0)).norm() < 1e-10);
        assert!((state[1] - Complex64::new(sqrt2_inv, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn test_cnot_specialized() {
        let mut state = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let gate = CNOTSpecialized {
            control: QubitId(0),
            target: QubitId(1),
        };

        gate.apply_specialized(&mut state, 2, false).unwrap();

        assert!((state[0] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
        assert!((state[1] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
        assert!((state[2] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
        assert!((state[3] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    }
}
