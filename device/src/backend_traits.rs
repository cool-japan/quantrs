//! Common traits and utilities for hardware backend translation
//!
//! This module provides shared functionality for working with different
//! quantum hardware backends and their gate sets.

use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};

use crate::translation::{DecomposedGate, HardwareBackend, NativeGateSet};

/// Trait for hardware-specific gate implementations
pub trait HardwareGate: GateOp {
    /// Get the hardware backend this gate is native to
    fn backend(&self) -> HardwareBackend;

    /// Get hardware-specific metadata
    fn metadata(&self) -> HashMap<String, String> {
        HashMap::new()
    }

    /// Check if this gate requires calibration
    fn requires_calibration(&self) -> bool {
        true
    }

    /// Get calibration parameters
    fn calibration_params(&self) -> Vec<String> {
        vec![]
    }
}

/// IBM-specific gate implementations
pub mod ibm_gates {
    use super::*;
    use scirs2_core::Complex64;
    use std::any::Any;

    /// IBM's SX gate (√X gate)
    #[derive(Debug, Clone, Copy)]
    pub struct SXGate {
        pub target: QubitId,
    }

    impl GateOp for SXGate {
        fn name(&self) -> &'static str {
            "sx"
        }

        fn qubits(&self) -> Vec<QubitId> {
            vec![self.target]
        }

        fn matrix(&self) -> QuantRS2Result<Vec<Complex64>> {
            let half = 0.5;
            let i_half = Complex64::new(0.0, 0.5);
            Ok(vec![
                Complex64::new(half, 0.0) + i_half,
                Complex64::new(half, 0.0) - i_half,
                Complex64::new(half, 0.0) - i_half,
                Complex64::new(half, 0.0) + i_half,
            ])
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn clone_gate(&self) -> Box<dyn GateOp> {
            Box::new(*self)
        }
    }

    impl HardwareGate for SXGate {
        fn backend(&self) -> HardwareBackend {
            HardwareBackend::IBMQuantum
        }

        fn metadata(&self) -> HashMap<String, String> {
            let mut meta = HashMap::new();
            meta.insert("gate_type".to_string(), "basis".to_string());
            meta.insert("duration_ns".to_string(), "35.5".to_string());
            meta
        }
    }
}

/// Google-specific gate implementations
pub mod google_gates {
    use super::*;
    use scirs2_core::Complex64;
    use std::any::Any;
    use std::f64::consts::PI;

    /// Google's Sycamore gate
    #[derive(Debug, Clone, Copy)]
    pub struct SycamoreGate {
        pub qubit1: QubitId,
        pub qubit2: QubitId,
    }

    impl GateOp for SycamoreGate {
        fn name(&self) -> &'static str {
            "syc"
        }

        fn qubits(&self) -> Vec<QubitId> {
            vec![self.qubit1, self.qubit2]
        }

        fn matrix(&self) -> QuantRS2Result<Vec<Complex64>> {
            // Sycamore gate matrix
            // This is a simplified version - actual gate is more complex
            let fsim_theta = PI / 2.0;
            let fsim_phi = PI / 6.0;

            // Create fSIM gate matrix
            let c = fsim_theta.cos();
            let s = Complex64::new(0.0, -fsim_theta.sin());
            let phase = Complex64::from_polar(1.0, -fsim_phi);

            Ok(vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(c, 0.0),
                s,
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                s,
                Complex64::new(c, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                phase,
            ])
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn clone_gate(&self) -> Box<dyn GateOp> {
            Box::new(*self)
        }
    }

    impl HardwareGate for SycamoreGate {
        fn backend(&self) -> HardwareBackend {
            HardwareBackend::GoogleSycamore
        }

        fn metadata(&self) -> HashMap<String, String> {
            let mut meta = HashMap::new();
            meta.insert("gate_type".to_string(), "entangling".to_string());
            meta.insert("duration_ns".to_string(), "12".to_string());
            meta.insert("fidelity".to_string(), "0.995".to_string());
            meta
        }
    }

    /// Google's powered gates (X^t, Y^t, Z^t)
    #[derive(Debug, Clone, Copy)]
    pub struct PoweredGate {
        pub target: QubitId,
        pub axis: char, // 'X', 'Y', or 'Z'
        pub power: f64,
    }

    impl GateOp for PoweredGate {
        fn name(&self) -> &'static str {
            match self.axis {
                'X' => "x_pow",
                'Y' => "y_pow",
                'Z' => "z_pow",
                _ => "pow",
            }
        }

        fn qubits(&self) -> Vec<QubitId> {
            vec![self.target]
        }

        fn matrix(&self) -> QuantRS2Result<Vec<Complex64>> {
            let angle = PI * self.power;
            let cos_half = (angle / 2.0).cos();
            let sin_half = (angle / 2.0).sin();

            match self.axis {
                'X' => Ok(vec![
                    Complex64::new(cos_half, 0.0),
                    Complex64::new(0.0, -sin_half),
                    Complex64::new(0.0, -sin_half),
                    Complex64::new(cos_half, 0.0),
                ]),
                'Y' => Ok(vec![
                    Complex64::new(cos_half, 0.0),
                    Complex64::new(-sin_half, 0.0),
                    Complex64::new(sin_half, 0.0),
                    Complex64::new(cos_half, 0.0),
                ]),
                'Z' => Ok(vec![
                    Complex64::from_polar(1.0, -angle / 2.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::from_polar(1.0, angle / 2.0),
                ]),
                _ => Err(QuantRS2Error::InvalidInput("Invalid axis".to_string())),
            }
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn clone_gate(&self) -> Box<dyn GateOp> {
            Box::new(*self)
        }
    }
}

/// IonQ-specific gate implementations
pub mod ionq_gates {
    use super::*;
    use scirs2_core::Complex64;
    use std::any::Any;

    /// IonQ's XX gate (Mølmer-Sørensen gate)
    #[derive(Debug, Clone, Copy)]
    pub struct XXGate {
        pub qubit1: QubitId,
        pub qubit2: QubitId,
        pub angle: f64,
    }

    impl GateOp for XXGate {
        fn name(&self) -> &'static str {
            "xx"
        }

        fn qubits(&self) -> Vec<QubitId> {
            vec![self.qubit1, self.qubit2]
        }

        fn matrix(&self) -> QuantRS2Result<Vec<Complex64>> {
            let c = self.angle.cos();
            let s = Complex64::new(0.0, -self.angle.sin());

            Ok(vec![
                Complex64::new(c, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                s,
                Complex64::new(0.0, 0.0),
                Complex64::new(c, 0.0),
                s,
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                s,
                Complex64::new(c, 0.0),
                Complex64::new(0.0, 0.0),
                s,
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(c, 0.0),
            ])
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn clone_gate(&self) -> Box<dyn GateOp> {
            Box::new(*self)
        }
    }

    impl HardwareGate for XXGate {
        fn backend(&self) -> HardwareBackend {
            HardwareBackend::IonQ
        }

        fn metadata(&self) -> HashMap<String, String> {
            let mut meta = HashMap::new();
            meta.insert("gate_type".to_string(), "ms".to_string());
            meta.insert("interaction".to_string(), "all-to-all".to_string());
            meta
        }

        fn calibration_params(&self) -> Vec<String> {
            vec!["ms_amplitude".to_string(), "ms_phase".to_string()]
        }
    }
}

/// Rigetti-specific gate implementations
pub mod rigetti_gates {
    use super::*;
    use scirs2_core::Complex64;
    use std::any::Any;

    /// Rigetti's parametrized XY gate
    #[derive(Debug, Clone, Copy)]
    pub struct XYGate {
        pub qubit1: QubitId,
        pub qubit2: QubitId,
        pub angle: f64,
    }

    impl GateOp for XYGate {
        fn name(&self) -> &'static str {
            "xy"
        }

        fn qubits(&self) -> Vec<QubitId> {
            vec![self.qubit1, self.qubit2]
        }

        fn matrix(&self) -> QuantRS2Result<Vec<Complex64>> {
            let c = (self.angle / 2.0).cos();
            let s = Complex64::new(0.0, (self.angle / 2.0).sin());

            Ok(vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(c, 0.0),
                s,
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                s,
                Complex64::new(c, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ])
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn clone_gate(&self) -> Box<dyn GateOp> {
            Box::new(*self)
        }
    }
}

/// Honeywell-specific gate implementations
pub mod honeywell_gates {
    use super::*;
    use scirs2_core::Complex64;
    use std::any::Any;

    /// Honeywell's native ZZ interaction
    #[derive(Debug, Clone, Copy)]
    pub struct ZZGate {
        pub qubit1: QubitId,
        pub qubit2: QubitId,
        pub angle: f64,
    }

    impl GateOp for ZZGate {
        fn name(&self) -> &'static str {
            "zz"
        }

        fn qubits(&self) -> Vec<QubitId> {
            vec![self.qubit1, self.qubit2]
        }

        fn matrix(&self) -> QuantRS2Result<Vec<Complex64>> {
            let phase_p = Complex64::from_polar(1.0, self.angle / 2.0);
            let phase_m = Complex64::from_polar(1.0, -self.angle / 2.0);

            Ok(vec![
                phase_m,
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                phase_p,
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                phase_p,
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                phase_m,
            ])
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn clone_gate(&self) -> Box<dyn GateOp> {
            Box::new(*self)
        }
    }

    impl HardwareGate for ZZGate {
        fn backend(&self) -> HardwareBackend {
            HardwareBackend::Honeywell
        }

        fn metadata(&self) -> HashMap<String, String> {
            let mut meta = HashMap::new();
            meta.insert("gate_type".to_string(), "native".to_string());
            meta.insert("fidelity".to_string(), "0.999".to_string());
            meta
        }
    }

    /// Honeywell's U3 gate (general single-qubit rotation)
    #[derive(Debug, Clone, Copy)]
    pub struct U3Gate {
        pub target: QubitId,
        pub theta: f64,
        pub phi: f64,
        pub lambda: f64,
    }

    impl GateOp for U3Gate {
        fn name(&self) -> &'static str {
            "u3"
        }

        fn qubits(&self) -> Vec<QubitId> {
            vec![self.target]
        }

        fn matrix(&self) -> QuantRS2Result<Vec<Complex64>> {
            let cos_half = (self.theta / 2.0).cos();
            let sin_half = (self.theta / 2.0).sin();

            Ok(vec![
                Complex64::new(cos_half, 0.0),
                -Complex64::from_polar(sin_half, self.lambda),
                Complex64::from_polar(sin_half, self.phi),
                Complex64::from_polar(cos_half, self.phi + self.lambda),
            ])
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn clone_gate(&self) -> Box<dyn GateOp> {
            Box::new(*self)
        }
    }
}

/// Decomposition validator
pub struct DecompositionValidator {
    /// Tolerance for matrix comparison
    tolerance: f64,
}

impl DecompositionValidator {
    /// Create a new validator
    pub const fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }

    /// Validate that a decomposition is equivalent to original gate.
    ///
    /// Reconstructs the product unitary of the decomposed native gates and
    /// compares it to the original gate's matrix (up to a global phase). Returns
    /// `true` iff the two unitaries agree within `self.tolerance`.
    pub fn validate(
        &self,
        original: &dyn GateOp,
        decomposed: &[DecomposedGate],
    ) -> QuantRS2Result<bool> {
        let fidelity = self.calculate_fidelity(original, decomposed)?;
        Ok((1.0 - fidelity).abs() <= self.tolerance)
    }

    /// Calculate the (global-phase-invariant) gate fidelity between the original
    /// gate and the product of its decomposition.
    ///
    /// For a `d`-dimensional Hilbert space the average-gate-fidelity-equivalent
    /// quantity used here is `|Tr(U_orig† · U_decomp)|² / d²`, which equals 1
    /// exactly when the two unitaries are identical up to a global phase.
    ///
    /// Returns an honest [`QuantRS2Error::UnsupportedOperation`] when the
    /// decomposition uses a native gate this validator cannot reconstruct a
    /// matrix for (rather than fabricating a high fidelity).
    pub fn calculate_fidelity(
        &self,
        original: &dyn GateOp,
        decomposed: &[DecomposedGate],
    ) -> QuantRS2Result<f64> {
        let original_qubits = original.qubits();
        let num_qubits = original_qubits.len();
        if num_qubits == 0 {
            return Err(QuantRS2Error::InvalidInput(
                "Original gate acts on zero qubits".to_string(),
            ));
        }
        // Build a canonical ordering of the qubits the decomposition acts on.
        let mut qubit_order: Vec<QubitId> = original_qubits.clone();
        for gate in decomposed {
            for q in &gate.qubits {
                if !qubit_order.contains(q) {
                    qubit_order.push(*q);
                }
            }
        }
        let dim = 1usize << qubit_order.len();

        // Original unitary embedded into the (possibly larger) qubit space.
        let original_matrix = flat_to_square(&original.matrix()?)?;
        let original_embedded = embed_unitary(&original_matrix, &original_qubits, &qubit_order)?;
        // Start from identity and left-multiply the decomposed gates in order.
        let mut product = identity_matrix(dim);
        for gate in decomposed {
            let gate_matrix = native_gate_matrix(&gate.native_gate, &gate.parameters)?;
            let embedded = embed_unitary(&gate_matrix, &gate.qubits, &qubit_order)?;
            product = matmul(&embedded, &product);
        }
        if original_embedded.len() != product.len() {
            return Err(QuantRS2Error::InvalidInput(
                "Dimension mismatch between original and decomposed unitaries".to_string(),
            ));
        }

        // Fidelity = |Tr(U_orig† U_decomp)|² / d².
        let mut trace = Complex64::new(0.0, 0.0);
        for row in 0..dim {
            for col in 0..dim {
                // (U_orig†)[row][col] = conj(U_orig[col][row]).
                trace += original_embedded[col * dim + row].conj() * product[row * dim + col];
            }
        }
        let d = dim as f64;
        let fidelity = trace.norm_sqr() / (d * d);
        // Numerical guard: clamp into [0, 1].
        Ok(fidelity.clamp(0.0, 1.0))
    }
}

/// Convert a row-major flat unitary into a square `Vec<Complex64>` of the same
/// row-major layout, validating that the length is a perfect square.
fn flat_to_square(flat: &[Complex64]) -> QuantRS2Result<Vec<Complex64>> {
    let dim = (flat.len() as f64).sqrt().round() as usize;
    if dim * dim != flat.len() {
        return Err(QuantRS2Error::InvalidInput(format!(
            "Gate matrix length {} is not a perfect square",
            flat.len()
        )));
    }
    Ok(flat.to_vec())
}

/// Identity matrix of dimension `dim` in row-major layout.
fn identity_matrix(dim: usize) -> Vec<Complex64> {
    let mut m = vec![Complex64::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        m[i * dim + i] = Complex64::new(1.0, 0.0);
    }
    m
}

/// Row-major dense matrix multiply `a · b` (both `dim×dim`).
fn matmul(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    let dim = (a.len() as f64).sqrt().round() as usize;
    let mut out = vec![Complex64::new(0.0, 0.0); dim * dim];
    for row in 0..dim {
        for k in 0..dim {
            let a_rk = a[row * dim + k];
            if a_rk == Complex64::new(0.0, 0.0) {
                continue;
            }
            for col in 0..dim {
                out[row * dim + col] += a_rk * b[k * dim + col];
            }
        }
    }
    out
}

/// Embed a unitary acting on `gate_qubits` into the full Hilbert space spanned
/// by `qubit_order`, using the framework's little-endian convention (qubit at
/// position `p` in `qubit_order` is bit `p` of the basis index).
fn embed_unitary(
    gate_matrix: &[Complex64],
    gate_qubits: &[QubitId],
    qubit_order: &[QubitId],
) -> QuantRS2Result<Vec<Complex64>> {
    let total = qubit_order.len();
    let full_dim = 1usize << total;
    let sub = gate_qubits.len();
    let sub_dim = 1usize << sub;
    if gate_matrix.len() != sub_dim * sub_dim {
        return Err(QuantRS2Error::InvalidInput(format!(
            "Gate on {} qubits has matrix of length {} (expected {})",
            sub,
            gate_matrix.len(),
            sub_dim * sub_dim
        )));
    }
    // Map each gate qubit to its bit position in the full index.
    let mut positions = Vec::with_capacity(sub);
    for q in gate_qubits {
        let pos = qubit_order.iter().position(|p| p == q).ok_or_else(|| {
            QuantRS2Error::InvalidInput("Gate qubit not in qubit ordering".to_string())
        })?;
        positions.push(pos);
    }

    let mut out = vec![Complex64::new(0.0, 0.0); full_dim * full_dim];
    for full_col in 0..full_dim {
        // Decode the sub-index of the columns spanned by the gate qubits.
        let mut sub_col = 0usize;
        for (i, &pos) in positions.iter().enumerate() {
            if full_col & (1usize << pos) != 0 {
                sub_col |= 1usize << i;
            }
        }
        // The "rest" bits (qubits the gate does not touch) must be preserved.
        for sub_row in 0..sub_dim {
            let amp = gate_matrix[sub_row * sub_dim + sub_col];
            if amp == Complex64::new(0.0, 0.0) {
                continue;
            }
            // Build the full row index: copy untouched bits from full_col, set
            // gate-qubit bits from sub_row.
            let mut full_row = full_col;
            for (i, &pos) in positions.iter().enumerate() {
                let bit = 1usize << pos;
                if sub_row & (1usize << i) != 0 {
                    full_row |= bit;
                } else {
                    full_row &= !bit;
                }
            }
            out[full_row * full_dim + full_col] = amp;
        }
    }
    Ok(out)
}

/// Reconstruct the 2x2 / 4x4 unitary of a native gate from its name and
/// parameters. Returns an honest error for names this function does not yet
/// know, so callers never silently assume a fabricated identity.
fn native_gate_matrix(name: &str, params: &[f64]) -> QuantRS2Result<Vec<Complex64>> {
    let frac = std::f64::consts::FRAC_1_SQRT_2;
    let c = Complex64::new;
    let upper = name.to_ascii_uppercase();
    let param = |i: usize| -> QuantRS2Result<f64> {
        params.get(i).copied().ok_or_else(|| {
            QuantRS2Error::InvalidInput(format!(
                "Native gate {upper} requires parameter index {i} but only {} provided",
                params.len()
            ))
        })
    };
    match upper.as_str() {
        "I" | "ID" => Ok(vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)]),
        "X" | "NOT" => Ok(vec![c(0.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)]),
        "Y" => Ok(vec![c(0.0, 0.0), c(0.0, -1.0), c(0.0, 1.0), c(0.0, 0.0)]),
        "Z" => Ok(vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(-1.0, 0.0)]),
        "H" => Ok(vec![
            c(frac, 0.0),
            c(frac, 0.0),
            c(frac, 0.0),
            c(-frac, 0.0),
        ]),
        "S" => Ok(vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 1.0)]),
        "SDG" | "SDAGGER" => Ok(vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, -1.0)]),
        "T" => Ok(vec![
            c(1.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4),
        ]),
        "TDG" | "TDAGGER" => Ok(vec![
            c(1.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            Complex64::from_polar(1.0, -std::f64::consts::FRAC_PI_4),
        ]),
        "SX" => {
            // sqrt(X) = 1/2 [[1+i, 1-i],[1-i, 1+i]].
            let half = Complex64::new(0.5, 0.5);
            let half_conj = Complex64::new(0.5, -0.5);
            Ok(vec![half, half_conj, half_conj, half])
        }
        "RX" => {
            let theta = param(0)?;
            let cos = (theta / 2.0).cos();
            let sin = (theta / 2.0).sin();
            Ok(vec![c(cos, 0.0), c(0.0, -sin), c(0.0, -sin), c(cos, 0.0)])
        }
        "RY" => {
            let theta = param(0)?;
            let cos = (theta / 2.0).cos();
            let sin = (theta / 2.0).sin();
            Ok(vec![c(cos, 0.0), c(-sin, 0.0), c(sin, 0.0), c(cos, 0.0)])
        }
        "RZ" => {
            let theta = param(0)?;
            Ok(vec![
                Complex64::from_polar(1.0, -theta / 2.0),
                c(0.0, 0.0),
                c(0.0, 0.0),
                Complex64::from_polar(1.0, theta / 2.0),
            ])
        }
        "P" | "PHASE" | "U1" => {
            let lambda = param(0)?;
            Ok(vec![
                c(1.0, 0.0),
                c(0.0, 0.0),
                c(0.0, 0.0),
                Complex64::from_polar(1.0, lambda),
            ])
        }
        "U" | "U3" => {
            let theta = param(0)?;
            let phi = param(1)?;
            let lambda = param(2)?;
            let cos = (theta / 2.0).cos();
            let sin = (theta / 2.0).sin();
            Ok(vec![
                c(cos, 0.0),
                -Complex64::from_polar(sin, lambda),
                Complex64::from_polar(sin, phi),
                Complex64::from_polar(cos, phi + lambda),
            ])
        }
        "CNOT" | "CX" => Ok(vec![
            c(1.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(1.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(1.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(1.0, 0.0),
            c(0.0, 0.0),
        ]),
        "CZ" => Ok(vec![
            c(1.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(1.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(1.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(-1.0, 0.0),
        ]),
        other => Err(QuantRS2Error::UnsupportedOperation(format!(
            "DecompositionValidator cannot reconstruct a matrix for native gate '{other}': \
             add it to native_gate_matrix to validate decompositions that use it"
        ))),
    }
}

/// Backend capabilities query
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendCapabilities {
    /// Backend identifier
    pub backend: HardwareBackend,
    /// Native gate set
    pub native_gates: NativeGateSet,
    /// Supported features
    pub features: BackendFeatures,
    /// Performance characteristics
    pub performance: BackendPerformance,
}

impl Default for BackendCapabilities {
    fn default() -> Self {
        Self {
            backend: HardwareBackend::Custom(0),
            native_gates: NativeGateSet::default(),
            features: BackendFeatures::default(),
            performance: BackendPerformance::default(),
        }
    }
}

/// Backend feature support
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendFeatures {
    /// Supports mid-circuit measurements
    pub mid_circuit_measurement: bool,
    /// Supports conditional gates
    pub conditional_gates: bool,
    /// Supports parametric compilation
    pub parametric_compilation: bool,
    /// Supports pulse-level control
    pub pulse_control: bool,
    /// Maximum circuit width
    pub max_qubits: usize,
    /// Maximum circuit depth
    pub max_depth: Option<usize>,
    /// Maximum number of mid-circuit measurements
    pub max_mid_circuit_measurements: Option<usize>,
    /// Classical register size (bits)
    pub classical_register_size: usize,
    /// Supports real-time feedback
    pub supports_real_time_feedback: bool,
    /// Supports parallel execution
    pub supports_parallel_execution: bool,
    /// Supports reset operations
    pub supports_reset: bool,
    /// Supports barrier operations
    pub supports_barriers: bool,
    /// Measurement types supported (Z, X, Y, Pauli, etc.)
    pub supported_measurement_bases: Vec<String>,
}

impl Default for BackendFeatures {
    fn default() -> Self {
        Self {
            mid_circuit_measurement: false,
            conditional_gates: false,
            parametric_compilation: true,
            pulse_control: false,
            max_qubits: 64,
            max_depth: None,
            max_mid_circuit_measurements: None,
            classical_register_size: 64,
            supports_real_time_feedback: false,
            supports_parallel_execution: false,
            supports_reset: true,
            supports_barriers: true,
            supported_measurement_bases: vec!["Z".to_string()],
        }
    }
}

/// Backend performance characteristics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendPerformance {
    /// Single-qubit gate time (ns)
    pub single_qubit_gate_time: f64,
    /// Two-qubit gate time (ns)
    pub two_qubit_gate_time: f64,
    /// Measurement time (ns)
    pub measurement_time: f64,
    /// Typical T1 time (μs)
    pub t1_time: f64,
    /// Typical T2 time (μs)
    pub t2_time: f64,
    /// Single-qubit gate fidelity
    pub single_qubit_fidelity: f64,
    /// Two-qubit gate fidelity
    pub two_qubit_fidelity: f64,
}

impl Default for BackendPerformance {
    fn default() -> Self {
        Self {
            single_qubit_gate_time: 50.0, // ns
            two_qubit_gate_time: 500.0,   // ns
            measurement_time: 1000.0,     // ns
            t1_time: 100.0,               // μs
            t2_time: 50.0,                // μs
            single_qubit_fidelity: 0.999,
            two_qubit_fidelity: 0.99,
        }
    }
}

/// Query backend capabilities
pub fn query_backend_capabilities(backend: HardwareBackend) -> BackendCapabilities {
    match backend {
        HardwareBackend::IBMQuantum => BackendCapabilities {
            backend,
            native_gates: NativeGateSet {
                backend,
                single_qubit_gates: ["id", "rz", "sx", "x"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                two_qubit_gates: vec!["cx".to_string()],
                multi_qubit_gates: vec![],
                arbitrary_single_qubit: false,
                rotation_axes: vec![crate::translation::RotationAxis::Z],
                constraints: crate::translation::BackendConstraints {
                    max_depth: None,
                    discrete_angles: None,
                    virtual_z: true,
                    coupling_map: None,
                    timing_constraints: None,
                },
            },
            features: BackendFeatures {
                mid_circuit_measurement: true,
                conditional_gates: true,
                parametric_compilation: true,
                pulse_control: true,
                max_qubits: 127,
                max_depth: Some(10000),
                max_mid_circuit_measurements: Some(127), // One per qubit
                classical_register_size: 128,
                supports_real_time_feedback: true,
                supports_parallel_execution: false, // IBM executes serially
                supports_reset: true,
                supports_barriers: true,
                supported_measurement_bases: vec![
                    "Z".to_string(),
                    "X".to_string(),
                    "Y".to_string(),
                ],
            },
            performance: BackendPerformance {
                single_qubit_gate_time: 35.0,
                two_qubit_gate_time: 300.0,
                measurement_time: 3000.0,
                t1_time: 100.0,
                t2_time: 100.0,
                single_qubit_fidelity: 0.9999,
                two_qubit_fidelity: 0.99,
            },
        },
        HardwareBackend::IonQ => BackendCapabilities {
            backend,
            native_gates: NativeGateSet {
                backend,
                single_qubit_gates: ["rx", "ry", "rz"].iter().map(|s| s.to_string()).collect(),
                two_qubit_gates: vec!["xx".to_string()],
                multi_qubit_gates: vec![],
                arbitrary_single_qubit: true,
                rotation_axes: vec![
                    crate::translation::RotationAxis::X,
                    crate::translation::RotationAxis::Y,
                    crate::translation::RotationAxis::Z,
                ],
                constraints: crate::translation::BackendConstraints {
                    max_depth: None,
                    discrete_angles: None,
                    virtual_z: false,
                    coupling_map: None, // All-to-all
                    timing_constraints: None,
                },
            },
            features: BackendFeatures {
                mid_circuit_measurement: false,
                conditional_gates: false,
                parametric_compilation: true,
                pulse_control: false,
                max_qubits: 32,
                max_depth: None,
                max_mid_circuit_measurements: None, // Not supported
                classical_register_size: 0,         // No classical registers
                supports_real_time_feedback: false,
                supports_parallel_execution: true, // All-to-all connectivity allows parallelism
                supports_reset: false,
                supports_barriers: false,
                supported_measurement_bases: vec!["Z".to_string()],
            },
            performance: BackendPerformance {
                single_qubit_gate_time: 135.0,
                two_qubit_gate_time: 600.0,
                measurement_time: 100.0,
                t1_time: 10000.0, // 10 ms
                t2_time: 1000.0,  // 1 ms
                single_qubit_fidelity: 0.9995,
                two_qubit_fidelity: 0.97,
            },
        },
        _ => {
            // Default capabilities
            BackendCapabilities {
                backend,
                native_gates: NativeGateSet {
                    backend,
                    single_qubit_gates: vec![],
                    two_qubit_gates: vec![],
                    multi_qubit_gates: vec![],
                    arbitrary_single_qubit: true,
                    rotation_axes: vec![],
                    constraints: crate::translation::BackendConstraints {
                        max_depth: None,
                        discrete_angles: None,
                        virtual_z: false,
                        coupling_map: None,
                        timing_constraints: None,
                    },
                },
                features: BackendFeatures {
                    mid_circuit_measurement: false,
                    conditional_gates: false,
                    parametric_compilation: false,
                    pulse_control: false,
                    max_qubits: 20,
                    max_depth: None,
                    max_mid_circuit_measurements: None,
                    classical_register_size: 0,
                    supports_real_time_feedback: false,
                    supports_parallel_execution: false,
                    supports_reset: false,
                    supports_barriers: false,
                    supported_measurement_bases: vec!["Z".to_string()],
                },
                performance: BackendPerformance {
                    single_qubit_gate_time: 50.0,
                    two_qubit_gate_time: 500.0,
                    measurement_time: 1000.0,
                    t1_time: 50.0,
                    t2_time: 50.0,
                    single_qubit_fidelity: 0.999,
                    two_qubit_fidelity: 0.99,
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_gate_implementations() {
        // Test IBM SX gate
        let sx = ibm_gates::SXGate { target: QubitId(0) };
        assert_eq!(sx.name(), "sx");
        assert_eq!(sx.backend(), HardwareBackend::IBMQuantum);

        // Test Google Sycamore gate
        let syc = google_gates::SycamoreGate {
            qubit1: QubitId(0),
            qubit2: QubitId(1),
        };
        assert_eq!(syc.name(), "syc");
        assert_eq!(syc.backend(), HardwareBackend::GoogleSycamore);

        // Test IonQ XX gate
        let xx = ionq_gates::XXGate {
            qubit1: QubitId(0),
            qubit2: QubitId(1),
            angle: std::f64::consts::PI / 2.0,
        };
        assert_eq!(xx.name(), "xx");
        assert_eq!(xx.backend(), HardwareBackend::IonQ);
    }

    #[test]
    fn test_backend_capabilities() {
        let ibm_caps = query_backend_capabilities(HardwareBackend::IBMQuantum);
        assert!(ibm_caps.features.pulse_control);
        assert!(ibm_caps.features.mid_circuit_measurement);
        assert_eq!(ibm_caps.performance.single_qubit_gate_time, 35.0);

        let ionq_caps = query_backend_capabilities(HardwareBackend::IonQ);
        assert!(!ionq_caps.features.pulse_control);
        assert!(ionq_caps.performance.t1_time > ibm_caps.performance.t1_time);
    }
}
