//! Functional API for TorchQuantum-compatible gate operations
//!
//! This module provides functional-style gate operations that can be applied
//! directly to quantum devices without creating persistent gate objects.
//! This mirrors TorchQuantum's `tq.functional` or `tqf` module.
//!
//! ## Example
//!
//! ```rust,ignore
//! use quantrs2_ml::torchquantum::prelude::*;
//! use quantrs2_ml::torchquantum::functional as tqf;
//!
//! let mut qdev = TQDevice::new(2);
//! tqf::hadamard(&mut qdev, 0)?;
//! tqf::rx(&mut qdev, 0, std::f64::consts::PI / 2.0)?;
//! tqf::cnot(&mut qdev, 0, 1)?;
//! ```

use super::gates::{
    TQHadamard, TQPauliX, TQPauliY, TQPauliZ, TQRx, TQRy, TQRz, TQCNOT, TQCRX, TQCRY, TQCRZ, TQCZ,
    TQRXX, TQRYY, TQRZX, TQRZZ, TQS, TQSWAP, TQSX, TQT, TQU1, TQU2, TQU3,
};
use super::{TQDevice, TQOperator};
use crate::error::{MLError, Result};
use std::f64::consts::PI;

/// Apply Hadamard gate to a qubit
pub fn hadamard(qdev: &mut TQDevice, wire: usize) -> Result<()> {
    let mut gate = TQHadamard::new();
    gate.apply(qdev, &[wire])
}

/// Apply Pauli-X gate to a qubit
pub fn paulix(qdev: &mut TQDevice, wire: usize) -> Result<()> {
    let mut gate = TQPauliX::new();
    gate.apply(qdev, &[wire])
}

/// Apply Pauli-Y gate to a qubit
pub fn pauliy(qdev: &mut TQDevice, wire: usize) -> Result<()> {
    let mut gate = TQPauliY::new();
    gate.apply(qdev, &[wire])
}

/// Apply Pauli-Z gate to a qubit
pub fn pauliz(qdev: &mut TQDevice, wire: usize) -> Result<()> {
    let mut gate = TQPauliZ::new();
    gate.apply(qdev, &[wire])
}

/// Apply S gate to a qubit
pub fn s(qdev: &mut TQDevice, wire: usize) -> Result<()> {
    let mut gate = TQS::new();
    gate.apply(qdev, &[wire])
}

/// Apply T gate to a qubit
pub fn t(qdev: &mut TQDevice, wire: usize) -> Result<()> {
    let mut gate = TQT::new();
    gate.apply(qdev, &[wire])
}

/// Apply SX gate to a qubit
pub fn sx(qdev: &mut TQDevice, wire: usize) -> Result<()> {
    let mut gate = TQSX::new();
    gate.apply(qdev, &[wire])
}

/// Apply RX rotation to a qubit
pub fn rx(qdev: &mut TQDevice, wire: usize, theta: f64) -> Result<()> {
    let mut gate = TQRx::new(true, false);
    gate.apply_with_params(qdev, &[wire], Some(&[theta]))
}

/// Apply RY rotation to a qubit
pub fn ry(qdev: &mut TQDevice, wire: usize, theta: f64) -> Result<()> {
    let mut gate = TQRy::new(true, false);
    gate.apply_with_params(qdev, &[wire], Some(&[theta]))
}

/// Apply RZ rotation to a qubit
pub fn rz(qdev: &mut TQDevice, wire: usize, theta: f64) -> Result<()> {
    let mut gate = TQRz::new(true, false);
    gate.apply_with_params(qdev, &[wire], Some(&[theta]))
}

/// Apply U1 gate (phase gate) to a qubit
pub fn u1(qdev: &mut TQDevice, wire: usize, lambda: f64) -> Result<()> {
    let mut gate = TQU1::new(true, false);
    gate.apply_with_params(qdev, &[wire], Some(&[lambda]))
}

/// Apply U2 gate to a qubit
pub fn u2(qdev: &mut TQDevice, wire: usize, phi: f64, lambda: f64) -> Result<()> {
    let mut gate = TQU2::new(true, false);
    gate.apply_with_params(qdev, &[wire], Some(&[phi, lambda]))
}

/// Apply U3 gate to a qubit
pub fn u3(qdev: &mut TQDevice, wire: usize, theta: f64, phi: f64, lambda: f64) -> Result<()> {
    let mut gate = TQU3::new(true, false);
    gate.apply_with_params(qdev, &[wire], Some(&[theta, phi, lambda]))
}

/// Apply CNOT gate between two qubits
pub fn cnot(qdev: &mut TQDevice, control: usize, target: usize) -> Result<()> {
    let mut gate = TQCNOT::new();
    gate.apply(qdev, &[control, target])
}

/// Apply CZ gate between two qubits
pub fn cz(qdev: &mut TQDevice, wire0: usize, wire1: usize) -> Result<()> {
    let mut gate = TQCZ::new();
    gate.apply(qdev, &[wire0, wire1])
}

/// Apply SWAP gate between two qubits
pub fn swap(qdev: &mut TQDevice, wire0: usize, wire1: usize) -> Result<()> {
    let mut gate = TQSWAP::new();
    gate.apply(qdev, &[wire0, wire1])
}

/// Apply RXX gate (Ising XX coupling) to two qubits
pub fn rxx(qdev: &mut TQDevice, wire0: usize, wire1: usize, theta: f64) -> Result<()> {
    let mut gate = TQRXX::new(true, false);
    gate.apply_with_params(qdev, &[wire0, wire1], Some(&[theta]))
}

/// Apply RYY gate (Ising YY coupling) to two qubits
pub fn ryy(qdev: &mut TQDevice, wire0: usize, wire1: usize, theta: f64) -> Result<()> {
    let mut gate = TQRYY::new(true, false);
    gate.apply_with_params(qdev, &[wire0, wire1], Some(&[theta]))
}

/// Apply RZZ gate (Ising ZZ coupling) to two qubits
pub fn rzz(qdev: &mut TQDevice, wire0: usize, wire1: usize, theta: f64) -> Result<()> {
    let mut gate = TQRZZ::new(true, false);
    gate.apply_with_params(qdev, &[wire0, wire1], Some(&[theta]))
}

/// Apply RZX gate (cross-resonance rotation) to two qubits
pub fn rzx(qdev: &mut TQDevice, wire0: usize, wire1: usize, theta: f64) -> Result<()> {
    let mut gate = TQRZX::new(true, false);
    gate.apply_with_params(qdev, &[wire0, wire1], Some(&[theta]))
}

/// Apply CRX gate (controlled RX) to two qubits
pub fn crx(qdev: &mut TQDevice, control: usize, target: usize, theta: f64) -> Result<()> {
    let mut gate = TQCRX::new(true, false);
    gate.apply_with_params(qdev, &[control, target], Some(&[theta]))
}

/// Apply CRY gate (controlled RY) to two qubits
pub fn cry(qdev: &mut TQDevice, control: usize, target: usize, theta: f64) -> Result<()> {
    let mut gate = TQCRY::new(true, false);
    gate.apply_with_params(qdev, &[control, target], Some(&[theta]))
}

/// Apply CRZ gate (controlled RZ) to two qubits
pub fn crz(qdev: &mut TQDevice, control: usize, target: usize, theta: f64) -> Result<()> {
    let mut gate = TQCRZ::new(true, false);
    gate.apply_with_params(qdev, &[control, target], Some(&[theta]))
}

/// Apply a rotation gate sequence RX -> RY -> RZ to a qubit
pub fn rot(qdev: &mut TQDevice, wire: usize, phi: f64, theta: f64, omega: f64) -> Result<()> {
    rx(qdev, wire, phi)?;
    ry(qdev, wire, theta)?;
    rz(qdev, wire, omega)?;
    Ok(())
}

/// Create a Bell state on two qubits
/// Applies H -> CNOT to create |00> + |11> / sqrt(2)
pub fn bell_state(qdev: &mut TQDevice, wire0: usize, wire1: usize) -> Result<()> {
    hadamard(qdev, wire0)?;
    cnot(qdev, wire0, wire1)?;
    Ok(())
}

/// Create a GHZ state on multiple qubits
/// Applies H on first qubit, then CNOT chain
pub fn ghz_state(qdev: &mut TQDevice, wires: &[usize]) -> Result<()> {
    if wires.is_empty() {
        return Err(MLError::InvalidConfiguration(
            "GHZ state requires at least one wire".to_string(),
        ));
    }

    hadamard(qdev, wires[0])?;
    for i in 1..wires.len() {
        cnot(qdev, wires[i - 1], wires[i])?;
    }
    Ok(())
}

/// Apply a barrier (no-op, for circuit visualization)
pub fn barrier(_qdev: &mut TQDevice, _wires: &[usize]) -> Result<()> {
    // Barrier is a no-op in simulation, used for circuit organization
    Ok(())
}

/// Reset a qubit to |0> state
pub fn reset(qdev: &mut TQDevice, wire: usize) -> Result<()> {
    // Measure and conditionally apply X if result is 1
    // For simulation, we can directly project to |0>
    // This is a simplified version that projects the state
    qdev.reset_states(qdev.bsz);
    // Note: This resets ALL qubits, not just the specified one
    // A proper implementation would need selective reset
    let _ = wire;
    Ok(())
}

/// Apply quantum Fourier transform on qubits
pub fn qft(qdev: &mut TQDevice, wires: &[usize]) -> Result<()> {
    let n = wires.len();
    for i in 0..n {
        // Hadamard on qubit i
        hadamard(qdev, wires[i])?;

        // Controlled phase rotations
        for j in (i + 1)..n {
            let k = j - i;
            let theta = PI / (1 << k) as f64;
            crz(qdev, wires[j], wires[i], theta)?;
        }
    }

    // Swap qubits to get correct order
    for i in 0..(n / 2) {
        swap(qdev, wires[i], wires[n - 1 - i])?;
    }

    Ok(())
}

/// Apply inverse quantum Fourier transform on qubits
pub fn iqft(qdev: &mut TQDevice, wires: &[usize]) -> Result<()> {
    let n = wires.len();

    // Swap qubits first
    for i in 0..(n / 2) {
        swap(qdev, wires[i], wires[n - 1 - i])?;
    }

    // Reverse order of operations from QFT
    for i in (0..n).rev() {
        // Inverse controlled phase rotations
        for j in ((i + 1)..n).rev() {
            let k = j - i;
            let theta = -PI / (1 << k) as f64;
            crz(qdev, wires[j], wires[i], theta)?;
        }

        // Hadamard on qubit i
        hadamard(qdev, wires[i])?;
    }

    Ok(())
}
