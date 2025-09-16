//! CPU backend implementation for GPU abstraction
//!
//! This provides a CPU-based fallback implementation of the GPU backend
//! interface, useful for testing and systems without GPU support.

#![allow(clippy::elided_lifetimes_in_paths)]

use super::{GpuBackend, GpuBuffer, GpuKernel};
use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    qubit::QubitId,
};
use ndarray::Array2;
use num_complex::Complex64;
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
        self.data.lock().unwrap()
    }
}

impl GpuBuffer for CpuBuffer {
    fn size(&self) -> usize {
        self.data.lock().unwrap().len() * std::mem::size_of::<Complex64>()
    }

    fn upload(&mut self, data: &[Complex64]) -> QuantRS2Result<()> {
        let mut buffer = self.data.lock().unwrap();
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
        let buffer = self.data.lock().unwrap();
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

        // Determine bit positions
        let (high_idx, low_idx) = if control_idx > target_idx {
            (control_idx, target_idx)
        } else {
            (target_idx, control_idx)
        };

        let high_stride = 1 << high_idx;
        let low_stride = 1 << low_idx;

        let state_size = 1 << n_qubits;
        let block_size = 1 << (high_idx + 1);
        let num_blocks = state_size / block_size;

        // Apply gate to each block
        for block in 0..num_blocks {
            let block_start = block * block_size;

            for i in 0..(block_size / 4) {
                // Calculate indices for the 4 basis states
                let base = block_start
                    + (i & ((1 << low_idx) - 1))
                    + ((i >> low_idx) << (low_idx + 1))
                    + ((i >> (high_idx - 1)) << (high_idx + 1));

                let indices = [
                    base,
                    base + low_stride,
                    base + high_stride,
                    base + low_stride + high_stride,
                ];

                Self::apply_gate_to_indices(&mut data, gate_matrix, &indices);
            }
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

        // Convert gate matrix to flat array for easier indexing
        let gate_flat: Vec<Complex64> = gate_matrix.iter().cloned().collect();

        // Calculate indices for all affected basis states
        let _total_states = 1 << n_qubits;
        let affected_states = 1 << gate_qubits;
        let unaffected_qubits = n_qubits - gate_qubits;
        let iterations = 1 << unaffected_qubits;

        // Sort qubit indices for consistent ordering
        let mut qubit_indices: Vec<usize> = qubits.iter().map(|q| q.0 as usize).collect();
        qubit_indices.sort_unstable();

        // Apply gate to each group of affected states
        for i in 0..iterations {
            let mut indices = vec![0; affected_states];

            // Calculate base index
            let mut base = 0;
            let mut remaining = i;
            let mut qubit_pos = 0;

            for bit in 0..n_qubits {
                if qubit_pos < gate_qubits && bit == qubit_indices[qubit_pos] {
                    qubit_pos += 1;
                } else {
                    if remaining & 1 == 1 {
                        base |= 1 << bit;
                    }
                    remaining >>= 1;
                }
            }

            // Generate all indices for this gate application
            for j in 0..affected_states {
                indices[j] = base;
                for (k, &qubit_idx) in qubit_indices.iter().enumerate() {
                    if (j >> k) & 1 == 1 {
                        indices[j] |= 1 << qubit_idx;
                    }
                }
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
        let _stride = 1 << qubit_idx;

        // Calculate probability of measuring |1⟩
        let mut prob_one = 0.0;
        for i in 0..(1 << n_qubits) {
            if (i >> qubit_idx) & 1 == 1 {
                prob_one += data[i].norm_sqr();
            }
        }

        // Simulate measurement
        let outcome = rand::random::<f64>() < prob_one;

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
    pub fn new() -> Self {
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

    fn name(&self) -> &str {
        "CPU"
    }

    fn device_info(&self) -> String {
        format!("CPU backend with {} threads", rayon::current_num_threads())
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

        buffer.upload(&data).unwrap();

        let mut downloaded = vec![Complex64::new(0.0, 0.0); 4];
        buffer.download(&mut downloaded).unwrap();

        assert_eq!(data, downloaded);
    }

    #[test]
    fn test_cpu_backend() {
        let backend = CpuBackend::new();
        assert!(CpuBackend::is_available());
        assert_eq!(backend.name(), "CPU");

        // Test state vector allocation
        let buffer = backend.allocate_state_vector(3).unwrap();
        assert_eq!(buffer.size(), 8 * std::mem::size_of::<Complex64>());
    }
}
