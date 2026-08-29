//! CPU backend implementation for GPU abstraction
//!
//! This provides a CPU-based fallback implementation of the GPU backend
//! interface, useful for testing and systems without GPU support.

use super::{GpuBackend, GpuBuffer, GpuKernel};
use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    qubit::QubitId,
};
use scirs2_core::ndarray::Array2;
use scirs2_core::Complex64;
use std::sync::{Arc, Mutex};

/// CPU-based buffer implementation
pub struct CpuBuffer {
    data: Arc<Mutex<Vec<Complex64>>>,
}

impl CpuBuffer {
    /// Create a new CPU buffer
    pub fn new(size: usize) -> Self {
        Self {
            data: Arc::new(Mutex::new(vec![Complex64::new(0.0, 0.0); size])),
        }
    }

    /// Get a reference to the data
    pub fn data(&self) -> std::sync::MutexGuard<'_, Vec<Complex64>> {
        self.data.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl GpuBuffer for CpuBuffer {
    fn size(&self) -> usize {
        self.data.lock().unwrap_or_else(|e| e.into_inner()).len() * std::mem::size_of::<Complex64>()
    }

    fn upload(&mut self, data: &[Complex64]) -> QuantRS2Result<()> {
        let mut buffer = self.data.lock().unwrap_or_else(|e| e.into_inner());
        if buffer.len() != data.len() {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Buffer size mismatch: {} != {}",
                buffer.len(),
                data.len()
            )));
        }
        buffer.copy_from_slice(data);
        Ok(())
    }

    fn download(&self, data: &mut [Complex64]) -> QuantRS2Result<()> {
        let buffer = self.data.lock().unwrap_or_else(|e| e.into_inner());
        if buffer.len() != data.len() {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Buffer size mismatch: {} != {}",
                buffer.len(),
                data.len()
            )));
        }
        data.copy_from_slice(&buffer);
        Ok(())
    }

    fn sync(&self) -> QuantRS2Result<()> {
        // No-op for CPU backend
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// CPU-based kernel implementation
pub struct CpuKernel;

impl CpuKernel {
    /// Apply a gate matrix to specific qubit indices
    fn apply_gate_to_indices(state: &mut [Complex64], gate: &[Complex64], indices: &[usize]) {
        let gate_size = indices.len();
        let mut temp = vec![Complex64::new(0.0, 0.0); gate_size];

        // Read values
        for (i, &idx) in indices.iter().enumerate() {
            temp[i] = state[idx];
        }

        // Apply gate
        for (i, &idx) in indices.iter().enumerate() {
            let mut sum = Complex64::new(0.0, 0.0);
            for j in 0..gate_size {
                sum += gate[i * gate_size + j] * temp[j];
            }
            state[idx] = sum;
        }
    }
}

impl GpuKernel for CpuKernel {
    fn apply_single_qubit_gate(
        &self,
        state: &mut dyn GpuBuffer,
        gate_matrix: &[Complex64; 4],
        qubit: QubitId,
        n_qubits: usize,
    ) -> QuantRS2Result<()> {
        let cpu_buffer = state
            .as_any_mut()
            .downcast_mut::<CpuBuffer>()
            .ok_or_else(|| QuantRS2Error::InvalidInput("Expected CpuBuffer".to_string()))?;

        let mut data = cpu_buffer.data();
        let qubit_idx = qubit.0 as usize;
        let stride = 1 << qubit_idx;
        let pairs = 1 << (n_qubits - 1);

        // Apply gate using bit manipulation
        for i in 0..pairs {
            let i0 = ((i >> qubit_idx) << (qubit_idx + 1)) | (i & ((1 << qubit_idx) - 1));
            let i1 = i0 | stride;

            let a = data[i0];
            let b = data[i1];

            data[i0] = gate_matrix[0] * a + gate_matrix[1] * b;
            data[i1] = gate_matrix[2] * a + gate_matrix[3] * b;
        }

        Ok(())
    }

    fn apply_two_qubit_gate(
        &self,
        state: &mut dyn GpuBuffer,
        gate_matrix: &[Complex64; 16],
        control: QubitId,
        target: QubitId,
        n_qubits: usize,
    ) -> QuantRS2Result<()> {
        let cpu_buffer = state
            .as_any_mut()
            .downcast_mut::<CpuBuffer>()
            .ok_or_else(|| QuantRS2Error::InvalidInput("Expected CpuBuffer".to_string()))?;

        let mut data = cpu_buffer.data();
        let control_idx = control.0 as usize;
        let target_idx = target.0 as usize;

        if control_idx >= n_qubits || target_idx >= n_qubits {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Qubit index out of range: control={control_idx}, target={target_idx}, \
                 n_qubits={n_qubits}"
            )));
        }
        if control_idx == target_idx {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Control and target must differ, both are qubit {control_idx}"
            )));
        }

        // Basis convention: qubit `q` occupies bit `q` of the state-vector index,
        // while the 4x4 gate matrix is ordered |control target> with the *control*
        // as the high bit of the local index (row/col 2 and 3 have control = 1).
        // The two orderings are independent -- mapping the local index by numeric
        // qubit order instead of by the control/target roles silently exchanges the
        // two operands whenever `control < target`.
        let control_stride = 1usize << control_idx;
        let target_stride = 1usize << target_idx;

        let (high_idx, low_idx) = if control_idx > target_idx {
            (control_idx, target_idx)
        } else {
            (target_idx, control_idx)
        };

        // Expand a dense counter over the `n_qubits - 2` spectator qubits into a
        // state-vector index with bit `low_idx` and bit `high_idx` cleared.
        let low_mask = (1usize << low_idx) - 1;
        let mid_mask = ((1usize << (high_idx - 1)) - 1) ^ low_mask;
        let groups = 1usize << (n_qubits - 2);

        for group in 0..groups {
            let base = (group & low_mask)
                | ((group & mid_mask) << 1)
                | ((group >> (high_idx - 1)) << (high_idx + 1));

            let indices = [
                base,
                base | target_stride,
                base | control_stride,
                base | control_stride | target_stride,
            ];

            Self::apply_gate_to_indices(&mut data, gate_matrix, &indices);
        }

        Ok(())
    }

    fn apply_multi_qubit_gate(
        &self,
        state: &mut dyn GpuBuffer,
        gate_matrix: &Array2<Complex64>,
        qubits: &[QubitId],
        n_qubits: usize,
    ) -> QuantRS2Result<()> {
        let cpu_buffer = state
            .as_any_mut()
            .downcast_mut::<CpuBuffer>()
            .ok_or_else(|| QuantRS2Error::InvalidInput("Expected CpuBuffer".to_string()))?;

        let mut data = cpu_buffer.data();
        let gate_qubits = qubits.len();
        let gate_dim = 1 << gate_qubits;

        if gate_matrix.dim() != (gate_dim, gate_dim) {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Gate matrix dimension mismatch: {:?} != ({}, {})",
                gate_matrix.dim(),
                gate_dim,
                gate_dim
            )));
        }

        // Operand order carries meaning: `qubits[0]` is the most significant bit of
        // the gate's local basis index, `qubits[gate_qubits - 1]` the least. Sorting
        // the list before mapping local index bits onto state-vector bits would
        // permute the gate's operands, so the caller-supplied order is kept here and
        // a sorted copy is used only to work out which bits are spectators.
        let operand_indices: Vec<usize> = qubits.iter().map(|q| q.0 as usize).collect();

        if let Some(&bad) = operand_indices.iter().find(|&&q| q >= n_qubits) {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Qubit index {bad} out of range for a {n_qubits}-qubit state"
            )));
        }

        let mut sorted_indices = operand_indices.clone();
        sorted_indices.sort_unstable();
        if sorted_indices.windows(2).any(|w| w[0] == w[1]) {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Gate operands must be distinct qubits, got {operand_indices:?}"
            )));
        }

        // Convert gate matrix to flat array for easier indexing
        let gate_flat: Vec<Complex64> = gate_matrix.iter().copied().collect();

        // Calculate indices for all affected basis states
        let affected_states = 1usize << gate_qubits;
        let unaffected_qubits = n_qubits - gate_qubits;
        let iterations = 1usize << unaffected_qubits;

        let mut indices = vec![0usize; affected_states];

        // Apply gate to each group of affected states
        for i in 0..iterations {
            // Scatter the spectator counter across the bit positions the gate
            // does not touch, leaving every operand bit clear.
            let mut base = 0usize;
            let mut remaining = i;
            let mut qubit_pos = 0;

            for bit in 0..n_qubits {
                if qubit_pos < gate_qubits && bit == sorted_indices[qubit_pos] {
                    qubit_pos += 1;
                } else {
                    if remaining & 1 == 1 {
                        base |= 1 << bit;
                    }
                    remaining >>= 1;
                }
            }

            // Generate all indices for this gate application, MSB-first over the
            // operand list so local basis state |q0 q1 ... qk> lines up with the
            // row/column ordering of `gate_matrix`.
            for (j, slot) in indices.iter_mut().enumerate() {
                let mut idx = base;
                for (position, &qubit_idx) in operand_indices.iter().enumerate() {
                    if (j >> (gate_qubits - 1 - position)) & 1 == 1 {
                        idx |= 1 << qubit_idx;
                    }
                }
                *slot = idx;
            }

            Self::apply_gate_to_indices(&mut data, &gate_flat, &indices);
        }

        Ok(())
    }

    fn measure_qubit(
        &self,
        state: &dyn GpuBuffer,
        qubit: QubitId,
        n_qubits: usize,
    ) -> QuantRS2Result<(bool, f64)> {
        let cpu_buffer = state
            .as_any()
            .downcast_ref::<CpuBuffer>()
            .ok_or_else(|| QuantRS2Error::InvalidInput("Expected CpuBuffer".to_string()))?;

        let data = cpu_buffer.data();
        let qubit_idx = qubit.0 as usize;
        // let _stride = 1 << qubit_idx;

        // Calculate probability of measuring |1⟩
        let mut prob_one = 0.0;
        for i in 0..(1 << n_qubits) {
            if (i >> qubit_idx) & 1 == 1 {
                prob_one += data[i].norm_sqr();
            }
        }

        // Simulate measurement
        use scirs2_core::random::prelude::*;
        let outcome = thread_rng().random::<f64>() < prob_one;

        Ok((outcome, if outcome { prob_one } else { 1.0 - prob_one }))
    }

    fn expectation_value(
        &self,
        state: &dyn GpuBuffer,
        observable: &Array2<Complex64>,
        qubits: &[QubitId],
        n_qubits: usize,
    ) -> QuantRS2Result<f64> {
        let cpu_buffer = state
            .as_any()
            .downcast_ref::<CpuBuffer>()
            .ok_or_else(|| QuantRS2Error::InvalidInput("Expected CpuBuffer".to_string()))?;

        let data = cpu_buffer.data();

        // For now, implement expectation value for single-qubit observables
        if qubits.len() != 1 || observable.dim() != (2, 2) {
            return Err(QuantRS2Error::UnsupportedOperation(
                "Only single-qubit observables supported currently".to_string(),
            ));
        }

        let qubit_idx = qubits[0].0 as usize;
        let stride = 1 << qubit_idx;
        let pairs = 1 << (n_qubits - 1);

        let mut expectation = Complex64::new(0.0, 0.0);

        for i in 0..pairs {
            let i0 = ((i >> qubit_idx) << (qubit_idx + 1)) | (i & ((1 << qubit_idx) - 1));
            let i1 = i0 | stride;

            let a = data[i0];
            let b = data[i1];

            expectation += a.conj() * (observable[(0, 0)] * a + observable[(0, 1)] * b);
            expectation += b.conj() * (observable[(1, 0)] * a + observable[(1, 1)] * b);
        }

        if expectation.im.abs() > 1e-10 {
            return Err(QuantRS2Error::InvalidInput(
                "Observable expectation value is not real".to_string(),
            ));
        }

        Ok(expectation.re)
    }
}

/// CPU backend implementation
pub struct CpuBackend {
    kernel: CpuKernel,
}

impl CpuBackend {
    /// Create a new CPU backend
    pub const fn new() -> Self {
        Self { kernel: CpuKernel }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuBackend for CpuBackend {
    fn is_available() -> bool {
        true // CPU is always available
    }

    fn name(&self) -> &'static str {
        "CPU"
    }

    fn device_info(&self) -> String {
        // Use scirs2_core::parallel_ops (SciRS2 POLICY compliant)
        use scirs2_core::parallel_ops::current_num_threads;
        format!("CPU backend with {} threads", current_num_threads())
    }

    fn allocate_state_vector(&self, n_qubits: usize) -> QuantRS2Result<Box<dyn GpuBuffer>> {
        let size = 1 << n_qubits;
        Ok(Box::new(CpuBuffer::new(size)))
    }

    fn allocate_density_matrix(&self, n_qubits: usize) -> QuantRS2Result<Box<dyn GpuBuffer>> {
        let size = 1 << (2 * n_qubits);
        Ok(Box::new(CpuBuffer::new(size)))
    }

    fn kernel(&self) -> &dyn GpuKernel {
        &self.kernel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_buffer() {
        let mut buffer = CpuBuffer::new(4);
        let data = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(0.0, -1.0),
        ];

        buffer
            .upload(&data)
            .expect("Failed to upload data to buffer");

        let mut downloaded = vec![Complex64::new(0.0, 0.0); 4];
        buffer
            .download(&mut downloaded)
            .expect("Failed to download data from buffer");

        assert_eq!(data, downloaded);
    }

    #[test]
    fn test_cpu_backend() {
        let backend = CpuBackend::new();
        assert!(CpuBackend::is_available());
        assert_eq!(backend.name(), "CPU");

        // Test state vector allocation
        let buffer = backend
            .allocate_state_vector(3)
            .expect("Failed to allocate state vector");
        assert_eq!(buffer.size(), 8 * std::mem::size_of::<Complex64>());
    }

    /// Flat, row-major CNOT in the |control target> basis, i.e. the same layout
    /// `gate::multi::CNOT::matrix()` produces.
    fn cnot_matrix() -> [Complex64; 16] {
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        [
            one, zero, zero, zero, //
            zero, one, zero, zero, //
            zero, zero, zero, one, //
            zero, zero, one, zero,
        ]
    }

    fn state_after_cnot(
        n_qubits: usize,
        control: u32,
        target: u32,
        initial: &[Complex64],
    ) -> Vec<Complex64> {
        let backend = CpuBackend::new();
        let mut buffer = backend
            .allocate_state_vector(n_qubits)
            .expect("Failed to allocate state vector");
        buffer.upload(initial).expect("Failed to upload state");

        backend
            .kernel()
            .apply_two_qubit_gate(
                buffer.as_mut(),
                &cnot_matrix(),
                QubitId(control),
                QubitId(target),
                n_qubits,
            )
            .expect("Failed to apply CNOT");

        let mut out = vec![Complex64::new(0.0, 0.0); initial.len()];
        buffer.download(&mut out).expect("Failed to download state");
        out
    }

    /// A two-qubit gate's operands are identified by role, not by numeric qubit
    /// index. Deriving the local basis ordering from `min`/`max` silently swaps
    /// control and target whenever `control < target`, which turns the Bell
    /// circuit H(q0)·CNOT(q0→q1) into an unentangled state.
    #[test]
    fn test_cnot_respects_control_target_order() {
        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);

        // |01> in bit order (q0 = 1, q1 = 0) is index 1.
        let mut basis_q0_set = vec![zero; 4];
        basis_q0_set[1] = one;

        // CNOT(control = q0, target = q1) must flip q1 -> index 3.
        let flipped = state_after_cnot(2, 0, 1, &basis_q0_set);
        assert_eq!(flipped[3], one, "CNOT(q0->q1) must flip the target qubit");
        assert_eq!(flipped[1], zero);

        // The reversed CNOT sees control q1 = 0, so it must leave the state alone.
        let unchanged = state_after_cnot(2, 1, 0, &basis_q0_set);
        assert_eq!(
            unchanged[1], one,
            "CNOT(q1->q0) must be the identity when q1 = 0"
        );
    }

    #[test]
    fn test_cnot_builds_bell_state() {
        let backend = CpuBackend::new();
        let mut buffer = backend
            .allocate_state_vector(2)
            .expect("Failed to allocate state vector");

        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let h = [
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(-inv_sqrt2, 0.0),
        ];

        let mut initial = vec![Complex64::new(0.0, 0.0); 4];
        initial[0] = Complex64::new(1.0, 0.0);
        buffer.upload(&initial).expect("Failed to upload state");

        backend
            .kernel()
            .apply_single_qubit_gate(buffer.as_mut(), &h, QubitId(0), 2)
            .expect("Failed to apply H");
        backend
            .kernel()
            .apply_two_qubit_gate(buffer.as_mut(), &cnot_matrix(), QubitId(0), QubitId(1), 2)
            .expect("Failed to apply CNOT");

        let mut out = vec![Complex64::new(0.0, 0.0); 4];
        buffer.download(&mut out).expect("Failed to download state");

        let probs: Vec<f64> = out.iter().map(|c| c.norm_sqr()).collect();
        assert!((probs[0] - 0.5).abs() < 1e-12, "probs = {probs:?}");
        assert!(probs[1].abs() < 1e-12, "probs = {probs:?}");
        assert!(probs[2].abs() < 1e-12, "probs = {probs:?}");
        assert!((probs[3] - 0.5).abs() < 1e-12, "probs = {probs:?}");
    }

    /// Spectator qubits must be preserved exactly, including when the operands
    /// straddle them and are not adjacent.
    #[test]
    fn test_cnot_on_non_adjacent_qubits_preserves_spectators() {
        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);

        // 5 qubits, control = q1, target = q4, spectators q0/q2/q3 all set.
        let n_qubits = 5;
        let spectators = (1 << 0) | (1 << 2) | (1 << 3);
        let start_index = spectators | (1 << 1); // control q1 = 1, target q4 = 0

        let mut initial = vec![zero; 1 << n_qubits];
        initial[start_index] = one;

        let out = state_after_cnot(n_qubits, 1, 4, &initial);

        let expected_index = start_index | (1 << 4);
        assert_eq!(out[expected_index], one, "target q4 should have flipped");
        assert_eq!(out[start_index], zero);
        assert_eq!(
            out.iter().filter(|c| c.norm_sqr() > 1e-24).count(),
            1,
            "exactly one basis state should carry amplitude"
        );
    }

    /// The multi-qubit path shares the same convention: `qubits[0]` is the most
    /// significant bit of the gate's local basis index.
    #[test]
    fn test_multi_qubit_gate_respects_operand_order() {
        use scirs2_core::ndarray::Array2;

        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);

        // Toffoli: flips the last operand when the first two are both |1>.
        let mut toffoli = Array2::from_elem((8, 8), zero);
        for i in 0..6 {
            toffoli[(i, i)] = one;
        }
        toffoli[(6, 7)] = one;
        toffoli[(7, 6)] = one;

        let backend = CpuBackend::new();

        // Controls q2 and q0 set, target q1 clear: index = 1 + 4 = 5.
        let mut initial = vec![zero; 8];
        initial[5] = one;

        let mut buffer = backend
            .allocate_state_vector(3)
            .expect("Failed to allocate state vector");
        buffer.upload(&initial).expect("Failed to upload state");

        backend
            .kernel()
            .apply_multi_qubit_gate(
                buffer.as_mut(),
                &toffoli,
                &[QubitId(2), QubitId(0), QubitId(1)],
                3,
            )
            .expect("Failed to apply Toffoli");

        let mut out = vec![zero; 8];
        buffer.download(&mut out).expect("Failed to download state");

        assert_eq!(out[7], one, "target q1 should have flipped, got {out:?}");
        assert_eq!(out[5], zero);
    }

    #[test]
    fn test_two_qubit_gate_rejects_invalid_operands() {
        let backend = CpuBackend::new();
        let mut buffer = backend
            .allocate_state_vector(2)
            .expect("Failed to allocate state vector");

        assert!(backend
            .kernel()
            .apply_two_qubit_gate(buffer.as_mut(), &cnot_matrix(), QubitId(0), QubitId(0), 2)
            .is_err());
        assert!(backend
            .kernel()
            .apply_two_qubit_gate(buffer.as_mut(), &cnot_matrix(), QubitId(0), QubitId(5), 2)
            .is_err());
    }
}
