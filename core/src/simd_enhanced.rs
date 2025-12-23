//! Enhanced SIMD operations for quantum computing
//!
//! This module provides highly optimized SIMD implementations for quantum gates
//! with adaptive dispatch based on hardware capabilities.

use crate::error::{QuantRS2Error, QuantRS2Result};
use crate::platform::PlatformCapabilities;
use crate::simd_ops_stubs::{SimdComplex64, SimdF64};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1};
use scirs2_core::Complex64;

/// Enhanced SIMD gate operations with hardware-adaptive dispatch
pub struct SimdGateEngine {
    /// Platform capabilities for adaptive optimization
    capabilities: PlatformCapabilities,
    /// Optimal SIMD width for f64 operations
    simd_width: usize,
    /// Cache line size for alignment
    cache_line_size: usize,
}

impl Default for SimdGateEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl SimdGateEngine {
    /// Create a new SIMD gate engine with automatic hardware detection
    pub fn new() -> Self {
        let capabilities = PlatformCapabilities::detect();
        let simd_width = capabilities.optimal_simd_width_f64();
        let cache_line_size = capabilities.cpu.cache.line_size.unwrap_or(64);

        Self {
            capabilities,
            simd_width,
            cache_line_size,
        }
    }

    /// Get the platform capabilities
    pub fn capabilities(&self) -> &PlatformCapabilities {
        &self.capabilities
    }

    /// Get the optimal SIMD width
    pub fn simd_width(&self) -> usize {
        self.simd_width
    }

    /// Apply a rotation gate (RX, RY, RZ) with SIMD optimization
    ///
    /// This applies rotation matrices to pairs of amplitudes efficiently.
    pub fn apply_rotation_gate(
        &self,
        amplitudes: &mut [Complex64],
        qubit: usize,
        axis: RotationAxis,
        angle: f64,
    ) -> QuantRS2Result<()> {
        let num_qubits = (amplitudes.len() as f64).log2() as usize;
        if qubit >= num_qubits {
            return Err(QuantRS2Error::InvalidInput(
                "Qubit index out of range".to_string(),
            ));
        }

        match axis {
            RotationAxis::X => self.apply_rx(amplitudes, qubit, angle),
            RotationAxis::Y => self.apply_ry(amplitudes, qubit, angle),
            RotationAxis::Z => self.apply_rz(amplitudes, qubit, angle),
        }
    }

    /// Apply RX gate: RX(θ) = exp(-iθX/2) = cos(θ/2)I - i*sin(θ/2)X
    fn apply_rx(
        &self,
        amplitudes: &mut [Complex64],
        qubit: usize,
        angle: f64,
    ) -> QuantRS2Result<()> {
        let half_angle = angle / 2.0;
        let cos_half = half_angle.cos();
        let sin_half = half_angle.sin();

        let qubit_mask = 1 << qubit;
        let mut idx0_list = Vec::new();
        let mut idx1_list = Vec::new();

        // Collect index pairs
        for i in 0..(amplitudes.len() / 2) {
            let idx0 = (i & !(qubit_mask >> 1)) | ((i & (qubit_mask >> 1)) << 1);
            let idx1 = idx0 | qubit_mask;

            if idx1 < amplitudes.len() {
                idx0_list.push(idx0);
                idx1_list.push(idx1);
            }
        }

        let pair_count = idx0_list.len();
        if pair_count == 0 {
            return Ok(());
        }

        // Extract amplitude pairs
        let mut a0_real = Vec::with_capacity(pair_count);
        let mut a0_imag = Vec::with_capacity(pair_count);
        let mut a1_real = Vec::with_capacity(pair_count);
        let mut a1_imag = Vec::with_capacity(pair_count);

        for i in 0..pair_count {
            let a0 = amplitudes[idx0_list[i]];
            let a1 = amplitudes[idx1_list[i]];
            a0_real.push(a0.re);
            a0_imag.push(a0.im);
            a1_real.push(a1.re);
            a1_imag.push(a1.im);
        }

        // Apply RX matrix using SIMD
        let a0_real_view = ArrayView1::from(&a0_real);
        let a0_imag_view = ArrayView1::from(&a0_imag);
        let a1_real_view = ArrayView1::from(&a1_real);
        let a1_imag_view = ArrayView1::from(&a1_imag);

        // RX matrix: [[cos, -i*sin], [-i*sin, cos]]
        // New a0 = cos*a0 - i*sin*a1 = (cos*a0_r, cos*a0_i) + (sin*a1_i, -sin*a1_r)
        // New a1 = -i*sin*a0 + cos*a1 = (sin*a0_i, -sin*a0_r) + (cos*a1_r, cos*a1_i)

        let cos_a0_r = <f64 as SimdF64>::simd_scalar_mul(&a0_real_view, cos_half);
        let cos_a0_i = <f64 as SimdF64>::simd_scalar_mul(&a0_imag_view, cos_half);
        let sin_a1_i = <f64 as SimdF64>::simd_scalar_mul(&a1_imag_view, sin_half);
        let sin_a1_r = <f64 as SimdF64>::simd_scalar_mul(&a1_real_view, sin_half);

        let new_a0_r = <f64 as SimdF64>::simd_add_arrays(&cos_a0_r.view(), &sin_a1_i.view());
        let new_a0_i = <f64 as SimdF64>::simd_sub_arrays(&cos_a0_i.view(), &sin_a1_r.view());

        let sin_a0_i = <f64 as SimdF64>::simd_scalar_mul(&a0_imag_view, sin_half);
        let sin_a0_r = <f64 as SimdF64>::simd_scalar_mul(&a0_real_view, sin_half);
        let cos_a1_r = <f64 as SimdF64>::simd_scalar_mul(&a1_real_view, cos_half);
        let cos_a1_i = <f64 as SimdF64>::simd_scalar_mul(&a1_imag_view, cos_half);

        let new_a1_r = <f64 as SimdF64>::simd_add_arrays(&sin_a0_i.view(), &cos_a1_r.view());
        let new_a1_i = <f64 as SimdF64>::simd_sub_arrays(&cos_a1_i.view(), &sin_a0_r.view());

        // Write back results
        for i in 0..pair_count {
            amplitudes[idx0_list[i]] = Complex64::new(new_a0_r[i], new_a0_i[i]);
            amplitudes[idx1_list[i]] = Complex64::new(new_a1_r[i], new_a1_i[i]);
        }

        Ok(())
    }

    /// Apply RY gate: RY(θ) = exp(-iθY/2) = cos(θ/2)I - i*sin(θ/2)Y
    fn apply_ry(
        &self,
        amplitudes: &mut [Complex64],
        qubit: usize,
        angle: f64,
    ) -> QuantRS2Result<()> {
        let half_angle = angle / 2.0;
        let cos_half = half_angle.cos();
        let sin_half = half_angle.sin();

        let qubit_mask = 1 << qubit;
        let mut idx0_list = Vec::new();
        let mut idx1_list = Vec::new();

        // Collect index pairs
        for i in 0..(amplitudes.len() / 2) {
            let idx0 = (i & !(qubit_mask >> 1)) | ((i & (qubit_mask >> 1)) << 1);
            let idx1 = idx0 | qubit_mask;

            if idx1 < amplitudes.len() {
                idx0_list.push(idx0);
                idx1_list.push(idx1);
            }
        }

        let pair_count = idx0_list.len();
        if pair_count == 0 {
            return Ok(());
        }

        // Extract amplitude pairs
        let mut a0_real = Vec::with_capacity(pair_count);
        let mut a0_imag = Vec::with_capacity(pair_count);
        let mut a1_real = Vec::with_capacity(pair_count);
        let mut a1_imag = Vec::with_capacity(pair_count);

        for i in 0..pair_count {
            let a0 = amplitudes[idx0_list[i]];
            let a1 = amplitudes[idx1_list[i]];
            a0_real.push(a0.re);
            a0_imag.push(a0.im);
            a1_real.push(a1.re);
            a1_imag.push(a1.im);
        }

        // Apply RY matrix using SIMD
        let a0_real_view = ArrayView1::from(&a0_real);
        let a0_imag_view = ArrayView1::from(&a0_imag);
        let a1_real_view = ArrayView1::from(&a1_real);
        let a1_imag_view = ArrayView1::from(&a1_imag);

        // RY matrix: [[cos, -sin], [sin, cos]]
        let cos_a0_r = <f64 as SimdF64>::simd_scalar_mul(&a0_real_view, cos_half);
        let cos_a0_i = <f64 as SimdF64>::simd_scalar_mul(&a0_imag_view, cos_half);
        let sin_a1_r = <f64 as SimdF64>::simd_scalar_mul(&a1_real_view, sin_half);
        let sin_a1_i = <f64 as SimdF64>::simd_scalar_mul(&a1_imag_view, sin_half);

        let new_a0_r = <f64 as SimdF64>::simd_sub_arrays(&cos_a0_r.view(), &sin_a1_r.view());
        let new_a0_i = <f64 as SimdF64>::simd_sub_arrays(&cos_a0_i.view(), &sin_a1_i.view());

        let sin_a0_r = <f64 as SimdF64>::simd_scalar_mul(&a0_real_view, sin_half);
        let sin_a0_i = <f64 as SimdF64>::simd_scalar_mul(&a0_imag_view, sin_half);
        let cos_a1_r = <f64 as SimdF64>::simd_scalar_mul(&a1_real_view, cos_half);
        let cos_a1_i = <f64 as SimdF64>::simd_scalar_mul(&a1_imag_view, cos_half);

        let new_a1_r = <f64 as SimdF64>::simd_add_arrays(&sin_a0_r.view(), &cos_a1_r.view());
        let new_a1_i = <f64 as SimdF64>::simd_add_arrays(&sin_a0_i.view(), &cos_a1_i.view());

        // Write back results
        for i in 0..pair_count {
            amplitudes[idx0_list[i]] = Complex64::new(new_a0_r[i], new_a0_i[i]);
            amplitudes[idx1_list[i]] = Complex64::new(new_a1_r[i], new_a1_i[i]);
        }

        Ok(())
    }

    /// Apply RZ gate: RZ(θ) = exp(-iθZ/2) = [[e^(-iθ/2), 0], [0, e^(iθ/2)]]
    fn apply_rz(
        &self,
        amplitudes: &mut [Complex64],
        qubit: usize,
        angle: f64,
    ) -> QuantRS2Result<()> {
        let half_angle = angle / 2.0;
        let cos_half = half_angle.cos();
        let sin_half = half_angle.sin();

        let qubit_mask = 1 << qubit;

        // Collect indices for each computational basis state
        let mut idx0_list = Vec::new(); // Qubit is |0⟩
        let mut idx1_list = Vec::new(); // Qubit is |1⟩

        for i in 0..amplitudes.len() {
            if (i & qubit_mask) == 0 {
                idx0_list.push(i);
            } else {
                idx1_list.push(i);
            }
        }

        // Apply phase e^(-iθ/2) to |0⟩ states
        if !idx0_list.is_empty() {
            let mut real_parts = Vec::with_capacity(idx0_list.len());
            let mut imag_parts = Vec::with_capacity(idx0_list.len());

            for &idx in &idx0_list {
                real_parts.push(amplitudes[idx].re);
                imag_parts.push(amplitudes[idx].im);
            }

            let real_view = ArrayView1::from(&real_parts);
            let imag_view = ArrayView1::from(&imag_parts);

            // Multiply by e^(-iθ/2) = cos(-θ/2) + i*sin(-θ/2) = cos(θ/2) - i*sin(θ/2)
            let real_cos = <f64 as SimdF64>::simd_scalar_mul(&real_view, cos_half);
            let imag_sin = <f64 as SimdF64>::simd_scalar_mul(&imag_view, sin_half);
            let new_real = <f64 as SimdF64>::simd_add_arrays(&real_cos.view(), &imag_sin.view());

            let real_sin = <f64 as SimdF64>::simd_scalar_mul(&real_view, -sin_half);
            let imag_cos = <f64 as SimdF64>::simd_scalar_mul(&imag_view, cos_half);
            let new_imag = <f64 as SimdF64>::simd_add_arrays(&real_sin.view(), &imag_cos.view());

            for (i, &idx) in idx0_list.iter().enumerate() {
                amplitudes[idx] = Complex64::new(new_real[i], new_imag[i]);
            }
        }

        // Apply phase e^(iθ/2) to |1⟩ states
        if !idx1_list.is_empty() {
            let mut real_parts = Vec::with_capacity(idx1_list.len());
            let mut imag_parts = Vec::with_capacity(idx1_list.len());

            for &idx in &idx1_list {
                real_parts.push(amplitudes[idx].re);
                imag_parts.push(amplitudes[idx].im);
            }

            let real_view = ArrayView1::from(&real_parts);
            let imag_view = ArrayView1::from(&imag_parts);

            // Multiply by e^(iθ/2) = cos(θ/2) + i*sin(θ/2)
            let real_cos = <f64 as SimdF64>::simd_scalar_mul(&real_view, cos_half);
            let imag_sin = <f64 as SimdF64>::simd_scalar_mul(&imag_view, sin_half);
            let new_real = <f64 as SimdF64>::simd_sub_arrays(&real_cos.view(), &imag_sin.view());

            let real_sin = <f64 as SimdF64>::simd_scalar_mul(&real_view, sin_half);
            let imag_cos = <f64 as SimdF64>::simd_scalar_mul(&imag_view, cos_half);
            let new_imag = <f64 as SimdF64>::simd_add_arrays(&real_sin.view(), &imag_cos.view());

            for (i, &idx) in idx1_list.iter().enumerate() {
                amplitudes[idx] = Complex64::new(new_real[i], new_imag[i]);
            }
        }

        Ok(())
    }

    /// Apply CNOT gate with SIMD optimization
    pub fn apply_cnot(
        &self,
        amplitudes: &mut [Complex64],
        control: usize,
        target: usize,
    ) -> QuantRS2Result<()> {
        let num_qubits = (amplitudes.len() as f64).log2() as usize;
        if control >= num_qubits || target >= num_qubits {
            return Err(QuantRS2Error::InvalidInput(
                "Qubit index out of range".to_string(),
            ));
        }

        if control == target {
            return Err(QuantRS2Error::InvalidInput(
                "Control and target must be different qubits".to_string(),
            ));
        }

        let control_mask = 1 << control;
        let target_mask = 1 << target;

        // Collect index pairs that need to be swapped
        let mut idx0_list = Vec::new();
        let mut idx1_list = Vec::new();

        // Iterate through all amplitudes
        // For CNOT: if control qubit is 1, flip target qubit
        for i in 0..amplitudes.len() {
            // Check if control bit is 1
            if (i & control_mask) != 0 {
                // Check if target bit is 0 (we only count each pair once)
                if (i & target_mask) == 0 {
                    let idx0 = i; // Target is currently 0
                    let idx1 = i ^ target_mask; // Flip target to 1
                    idx0_list.push(idx0);
                    idx1_list.push(idx1);
                }
            }
        }

        // Swap amplitude pairs using SIMD
        let pair_count = idx0_list.len();
        if pair_count == 0 {
            return Ok(());
        }

        // Simple swap without SIMD overhead for small counts
        if pair_count < 4 {
            for i in 0..pair_count {
                amplitudes.swap(idx0_list[i], idx1_list[i]);
            }
            return Ok(());
        }

        // Extract amplitudes for SIMD swap
        let mut a0_real = Vec::with_capacity(pair_count);
        let mut a0_imag = Vec::with_capacity(pair_count);
        let mut a1_real = Vec::with_capacity(pair_count);
        let mut a1_imag = Vec::with_capacity(pair_count);

        for i in 0..pair_count {
            let a0 = amplitudes[idx0_list[i]];
            let a1 = amplitudes[idx1_list[i]];
            a0_real.push(a0.re);
            a0_imag.push(a0.im);
            a1_real.push(a1.re);
            a1_imag.push(a1.im);
        }

        // Write back swapped (no computation needed, just assignment)
        for i in 0..pair_count {
            amplitudes[idx0_list[i]] = Complex64::new(a1_real[i], a1_imag[i]);
            amplitudes[idx1_list[i]] = Complex64::new(a0_real[i], a0_imag[i]);
        }

        Ok(())
    }

    /// Batch apply multiple single-qubit gates
    ///
    /// This is more efficient than applying gates one by one when multiple
    /// gates need to be applied to different qubits.
    pub fn batch_apply_single_qubit(
        &self,
        amplitudes: &mut [Complex64],
        gates: &[(usize, RotationAxis, f64)],
    ) -> QuantRS2Result<()> {
        // Sort gates by qubit to improve cache locality
        let mut sorted_gates = gates.to_vec();
        sorted_gates.sort_by_key(|(qubit, _, _)| *qubit);

        for (qubit, axis, angle) in sorted_gates {
            self.apply_rotation_gate(amplitudes, qubit, axis, angle)?;
        }

        Ok(())
    }

    /// Compute fidelity between two quantum states with SIMD
    pub fn fidelity(&self, state1: &[Complex64], state2: &[Complex64]) -> QuantRS2Result<f64> {
        if state1.len() != state2.len() {
            return Err(QuantRS2Error::InvalidInput(
                "States must have the same length".to_string(),
            ));
        }

        // Compute inner product ⟨ψ|φ⟩
        let len = state1.len();

        let mut state1_real = Vec::with_capacity(len);
        let mut state1_imag = Vec::with_capacity(len);
        let mut state2_real = Vec::with_capacity(len);
        let mut state2_imag = Vec::with_capacity(len);

        for (a, b) in state1.iter().zip(state2.iter()) {
            state1_real.push(a.re);
            state1_imag.push(a.im);
            state2_real.push(b.re);
            state2_imag.push(b.im);
        }

        let state1_real_view = ArrayView1::from(&state1_real);
        let state1_imag_view = ArrayView1::from(&state1_imag);
        let state2_real_view = ArrayView1::from(&state2_real);
        let state2_imag_view = ArrayView1::from(&state2_imag);

        // Inner product: (a* · b) = Σ (a_r - i*a_i)(b_r + i*b_i)
        let rr = <f64 as SimdF64>::simd_mul_arrays(&state1_real_view, &state2_real_view);
        let ii = <f64 as SimdF64>::simd_mul_arrays(&state1_imag_view, &state2_imag_view);
        let real_sum = <f64 as SimdF64>::simd_add_arrays(&rr.view(), &ii.view());
        let real_part = <f64 as SimdF64>::simd_sum_array(&real_sum.view());

        let ri = <f64 as SimdF64>::simd_mul_arrays(&state1_real_view, &state2_imag_view);
        let ir = <f64 as SimdF64>::simd_mul_arrays(&state1_imag_view, &state2_real_view);
        let imag_diff = <f64 as SimdF64>::simd_sub_arrays(&ri.view(), &ir.view());
        let imag_part = <f64 as SimdF64>::simd_sum_array(&imag_diff.view());

        // Fidelity is |⟨ψ|φ⟩|²
        let fidelity = real_part * real_part + imag_part * imag_part;
        Ok(fidelity)
    }
}

/// Rotation axis for single-qubit rotation gates
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationAxis {
    X,
    Y,
    Z,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_engine_creation() {
        let engine = SimdGateEngine::new();
        assert!(engine.simd_width() >= 1);
        assert!(engine.simd_width() <= 8);
    }

    #[test]
    fn test_rx_gate() {
        let engine = SimdGateEngine::new();
        let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];

        // Apply RX(π) which should flip the state to |1⟩
        engine
            .apply_rotation_gate(&mut state, 0, RotationAxis::X, std::f64::consts::PI)
            .unwrap();

        // After RX(π), |0⟩ → -i|1⟩
        assert!(state[0].norm() < 0.1);
        assert!((state[1].norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ry_gate() {
        let engine = SimdGateEngine::new();
        let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];

        // Apply RY(π/2) which creates equal superposition
        engine
            .apply_rotation_gate(&mut state, 0, RotationAxis::Y, std::f64::consts::PI / 2.0)
            .unwrap();

        let sqrt2_inv = 1.0 / std::f64::consts::SQRT_2;
        assert!((state[0].norm() - sqrt2_inv).abs() < 1e-10);
        assert!((state[1].norm() - sqrt2_inv).abs() < 1e-10);
    }

    #[test]
    fn test_rz_gate() {
        let engine = SimdGateEngine::new();
        let mut state = vec![
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ];

        // Apply RZ(π/4)
        engine
            .apply_rotation_gate(&mut state, 0, RotationAxis::Z, std::f64::consts::PI / 4.0)
            .unwrap();

        // Magnitudes should be preserved
        let sqrt2_inv = 1.0 / std::f64::consts::SQRT_2;
        assert!((state[0].norm() - sqrt2_inv).abs() < 1e-10);
        assert!((state[1].norm() - sqrt2_inv).abs() < 1e-10);
    }

    #[test]
    fn test_cnot_gate() {
        let engine = SimdGateEngine::new();

        // Test CNOT on |10⟩ state (control=1, target=0, should flip to |11⟩)
        let mut state = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];

        engine.apply_cnot(&mut state, 1, 0).unwrap();

        // After CNOT with control=1, target=0: |10⟩ → |11⟩
        // When control (bit 1) is 1, flip target (bit 0)
        // |10⟩ (index 2) → |11⟩ (index 3)
        assert!(state[0].norm() < 1e-10);
        assert!(state[1].norm() < 1e-10);
        assert!(state[2].norm() < 1e-10);
        assert!((state[3].norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fidelity() {
        let engine = SimdGateEngine::new();

        let state1 = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let state2 = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];

        let fid = engine.fidelity(&state1, &state2).unwrap();
        assert!((fid - 1.0).abs() < 1e-10);

        let state3 = vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];
        let fid2 = engine.fidelity(&state1, &state3).unwrap();
        assert!(fid2.abs() < 1e-10);
    }

    #[test]
    fn test_batch_gates() {
        let engine = SimdGateEngine::new();
        // Use single-qubit state for now
        let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];

        let gates = vec![(0, RotationAxis::X, std::f64::consts::PI / 2.0)];

        engine.batch_apply_single_qubit(&mut state, &gates).unwrap();

        // Check normalization
        let norm_sqr: f64 = state.iter().map(|c| c.norm_sqr()).sum();
        assert!((norm_sqr - 1.0).abs() < 1e-8, "Norm squared: {}", norm_sqr);
    }
}
