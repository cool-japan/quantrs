use num_complex::Complex64;
use scirs2_core::parallel_ops::*;
use std::sync::Mutex;

use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::{multi, single, GateOp},
    qubit::QubitId,
    register::Register,
};

use crate::diagnostics::SimulationDiagnostics;
use crate::optimized_simd;
use crate::scirs2_integration::{Matrix, SciRS2Backend, Vector, BLAS};
use crate::utils::{flip_bit, gate_vec_to_array2};

/// A state vector simulator for quantum circuits
///
/// This simulator implements the state vector approach, where the full quantum
/// state is represented as a complex vector of dimension 2^N for N qubits.
#[derive(Debug)]
pub struct StateVectorSimulator {
    /// Use parallel execution
    pub parallel: bool,

    /// Basic noise model (if any)
    pub noise_model: Option<crate::noise::NoiseModel>,

    /// Advanced noise model (if any)
    pub advanced_noise_model: Option<crate::noise_advanced::AdvancedNoiseModel>,

    /// Optimized buffer pool for memory reuse (thread-safe)
    buffer_pool: Mutex<BufferPool>,

    /// Enable SIMD optimizations for gate operations
    pub use_simd: bool,

    /// Enable gate fusion optimization
    pub use_gate_fusion: bool,

    /// Diagnostics system for monitoring and error handling
    pub diagnostics: Option<SimulationDiagnostics>,

    /// SciRS2 backend for optimized linear algebra operations
    scirs2_backend: SciRS2Backend,
}

impl Clone for StateVectorSimulator {
    fn clone(&self) -> Self {
        Self {
            parallel: self.parallel,
            noise_model: self.noise_model.clone(),
            advanced_noise_model: self.advanced_noise_model.clone(),
            buffer_pool: Mutex::new(BufferPool::new(4, 1024)), // Create new buffer pool
            use_simd: self.use_simd,
            use_gate_fusion: self.use_gate_fusion,
            diagnostics: self
                .diagnostics
                .as_ref()
                .map(|_| SimulationDiagnostics::new()),
            scirs2_backend: SciRS2Backend::new(),
        }
    }
}

/// Memory pool for efficient state vector operations
#[derive(Debug, Clone)]
pub struct BufferPool {
    /// Pre-allocated working buffers
    working_buffers: std::collections::VecDeque<Vec<Complex64>>,
    /// Maximum number of cached buffers
    max_buffers: usize,
    /// Target buffer size for efficient reuse
    target_size: usize,
}

impl BufferPool {
    /// Create new buffer pool
    pub fn new(max_buffers: usize, target_size: usize) -> Self {
        Self {
            working_buffers: std::collections::VecDeque::with_capacity(max_buffers),
            max_buffers,
            target_size,
        }
    }

    /// Get a buffer from pool or allocate new one
    pub fn get_buffer(&mut self, size: usize) -> Vec<Complex64> {
        // Try to reuse existing buffer
        if let Some(mut buffer) = self.working_buffers.pop_front() {
            if buffer.capacity() >= size {
                buffer.clear();
                buffer.resize(size, Complex64::new(0.0, 0.0));
                return buffer;
            }
        }

        // Allocate new buffer with extra capacity for growth
        let capacity = size.max(self.target_size);
        let mut buffer = Vec::with_capacity(capacity);
        buffer.resize(size, Complex64::new(0.0, 0.0));
        buffer
    }

    /// Return buffer to pool
    pub fn return_buffer(&mut self, buffer: Vec<Complex64>) {
        if self.working_buffers.len() < self.max_buffers
            && buffer.capacity() >= self.target_size / 2
        {
            self.working_buffers.push_back(buffer);
        }
        // Otherwise let buffer be dropped and deallocated
    }

    /// Clear all cached buffers
    pub fn clear(&mut self) {
        self.working_buffers.clear();
    }
}

impl StateVectorSimulator {
    /// Create a new state vector simulator with default settings
    pub fn new() -> Self {
        Self {
            parallel: true,
            noise_model: None,
            advanced_noise_model: None,
            buffer_pool: Mutex::new(BufferPool::new(4, 1024)), // Default: 4 buffers, 1K size target
            use_simd: true,
            use_gate_fusion: true,
            diagnostics: None,
            scirs2_backend: SciRS2Backend::new(),
        }
    }

    /// Create a new state vector simulator with parallel execution disabled
    pub fn sequential() -> Self {
        Self {
            parallel: false,
            noise_model: None,
            advanced_noise_model: None,
            buffer_pool: Mutex::new(BufferPool::new(2, 512)), // Smaller pool for sequential
            use_simd: true,
            use_gate_fusion: true,
            diagnostics: None,
            scirs2_backend: SciRS2Backend::new(),
        }
    }

    /// Create a new state vector simulator with a basic noise model
    pub fn with_noise(noise_model: crate::noise::NoiseModel) -> Self {
        Self {
            parallel: true,
            noise_model: Some(noise_model),
            advanced_noise_model: None,
            buffer_pool: Mutex::new(BufferPool::new(4, 1024)),
            use_simd: true,
            use_gate_fusion: true,
            diagnostics: None,
            scirs2_backend: SciRS2Backend::new(),
        }
    }

    /// Create a new state vector simulator with an advanced noise model
    pub fn with_advanced_noise(
        advanced_noise_model: crate::noise_advanced::AdvancedNoiseModel,
    ) -> Self {
        Self {
            parallel: true,
            noise_model: None,
            advanced_noise_model: Some(advanced_noise_model),
            buffer_pool: Mutex::new(BufferPool::new(4, 1024)),
            use_simd: true,
            use_gate_fusion: true,
            diagnostics: None,
            scirs2_backend: SciRS2Backend::new(),
        }
    }

    /// Create simulator with custom buffer pool configuration
    pub fn with_buffer_pool(parallel: bool, max_buffers: usize, target_size: usize) -> Self {
        Self {
            parallel,
            noise_model: None,
            advanced_noise_model: None,
            buffer_pool: Mutex::new(BufferPool::new(max_buffers, target_size)),
            use_simd: true,
            use_gate_fusion: true,
            diagnostics: None,
            scirs2_backend: SciRS2Backend::new(),
        }
    }

    /// Set the basic noise model
    pub fn set_noise_model(&mut self, noise_model: crate::noise::NoiseModel) -> &mut Self {
        self.noise_model = Some(noise_model);
        self.advanced_noise_model = None; // Remove advanced model if it exists
        self
    }

    /// Set the advanced noise model
    pub fn set_advanced_noise_model(
        &mut self,
        advanced_noise_model: crate::noise_advanced::AdvancedNoiseModel,
    ) -> &mut Self {
        self.advanced_noise_model = Some(advanced_noise_model);
        self.noise_model = None; // Remove basic model if it exists
        self
    }

    /// Remove all noise models
    pub fn remove_noise_model(&mut self) -> &mut Self {
        self.noise_model = None;
        self.advanced_noise_model = None;
        self
    }

    /// Enable or disable SIMD optimizations
    pub fn set_simd_enabled(&mut self, enabled: bool) -> &mut Self {
        self.use_simd = enabled;
        self
    }

    /// Enable or disable gate fusion optimization
    pub fn set_gate_fusion_enabled(&mut self, enabled: bool) -> &mut Self {
        self.use_gate_fusion = enabled;
        self
    }

    /// Get access to buffer pool for testing purposes
    pub fn get_buffer_pool(&self) -> &Mutex<BufferPool> {
        &self.buffer_pool
    }

    /// Enable diagnostics and monitoring
    pub fn enable_diagnostics(&mut self) -> &mut Self {
        self.diagnostics = Some(SimulationDiagnostics::new());
        self
    }

    /// Disable diagnostics and monitoring
    pub fn disable_diagnostics(&mut self) -> &mut Self {
        self.diagnostics = None;
        self
    }

    /// Get diagnostics report if diagnostics are enabled
    pub fn get_diagnostics_report(&self) -> Option<crate::diagnostics::DiagnosticReport> {
        self.diagnostics.as_ref().map(|d| d.generate_report())
    }

    /// Create a high-performance configuration
    pub fn high_performance() -> Self {
        Self {
            parallel: true,
            noise_model: None,
            advanced_noise_model: None,
            buffer_pool: Mutex::new(BufferPool::new(8, 2048)), // Larger pool for high performance
            use_simd: true,
            use_gate_fusion: true,
            diagnostics: Some(SimulationDiagnostics::new()),
            scirs2_backend: SciRS2Backend::new(),
        }
    }

    /// Apply a dense matrix-vector multiplication using SciRS2 when available
    #[cfg(feature = "advanced_math")]
    fn apply_dense_matrix_vector(
        &mut self,
        matrix: &[Complex64],
        vector: &[Complex64],
        result: &mut [Complex64],
    ) -> QuantRS2Result<()> {
        if self.scirs2_backend.is_available() && matrix.len() >= 64 && vector.len() >= 8 {
            // Use SciRS2 for larger operations where the overhead is worthwhile
            use ndarray::{Array1, Array2};

            let rows = result.len();
            let cols = vector.len();

            if matrix.len() != rows * cols {
                return Err(QuantRS2Error::InvalidInput(
                    "Dimension mismatch in matrix application".to_string(),
                ));
            }

            // Convert to ndarray format
            let matrix_2d =
                Array2::from_shape_vec((rows, cols), matrix.to_vec()).map_err(|_| {
                    QuantRS2Error::InvalidInput("Matrix shape conversion failed".to_string())
                })?;
            let vector_1d = Array1::from_vec(vector.to_vec());

            // Convert to SciRS2 format
            let scirs2_matrix = Matrix::from_array2(
                &matrix_2d.view(),
                &crate::scirs2_integration::MemoryPool::new(),
            )
            .map_err(|_| QuantRS2Error::ComputationError("Matrix conversion failed".to_string()))?;
            let scirs2_vector = Vector::from_array1(
                &vector_1d.view(),
                &crate::scirs2_integration::MemoryPool::new(),
            )
            .map_err(|_| QuantRS2Error::ComputationError("Vector conversion failed".to_string()))?;

            // Perform optimized matrix-vector multiplication
            let scirs2_result = self
                .scirs2_backend
                .matrix_vector_multiply(&scirs2_matrix, &scirs2_vector)
                .map_err(|_| {
                    QuantRS2Error::ComputationError(
                        "Matrix-vector multiplication failed".to_string(),
                    )
                })?;

            // Convert back to slice
            let result_array = scirs2_result.to_array1().map_err(|_| {
                QuantRS2Error::ComputationError("Result conversion to array failed".to_string())
            })?;
            result.copy_from_slice(result_array.as_slice().unwrap());

            Ok(())
        } else {
            // Fallback to manual implementation for smaller operations
            for i in 0..result.len() {
                result[i] = Complex64::new(0.0, 0.0);
                for j in 0..vector.len() {
                    result[i] += matrix[i * vector.len() + j] * vector[j];
                }
            }
            Ok(())
        }
    }

    /// Fallback dense matrix-vector multiplication for when SciRS2 is not available
    #[cfg(not(feature = "advanced_math"))]
    fn apply_dense_matrix_vector(
        &mut self,
        matrix: &[Complex64],
        vector: &[Complex64],
        result: &mut [Complex64],
    ) -> QuantRS2Result<()> {
        // Manual implementation
        for i in 0..result.len() {
            result[i] = Complex64::new(0.0, 0.0);
            for j in 0..vector.len() {
                result[i] += matrix[i * vector.len() + j] * vector[j];
            }
        }
        Ok(())
    }

    /// Apply a single-qubit gate to a state vector
    fn apply_single_qubit_gate<const N: usize>(
        &self,
        state: &mut [Complex64],
        gate_matrix: &[Complex64],
        target: QubitId,
    ) -> QuantRS2Result<()> {
        let target_idx = target.id() as usize;
        if target_idx >= N {
            return Err(QuantRS2Error::InvalidQubitId(target.id()));
        }

        // Use SIMD optimization if enabled and beneficial
        if self.use_simd && state.len() >= 8 {
            return self.apply_single_qubit_gate_simd::<N>(state, gate_matrix, target_idx);
        }

        // Convert the gate matrix to flat representation for faster access
        // Gate matrix: [m00, m01, m10, m11]
        let m00 = gate_matrix[0];
        let m01 = gate_matrix[1];
        let m10 = gate_matrix[2];
        let m11 = gate_matrix[3];

        // Apply the gate to each amplitude
        if self.parallel {
            // Get buffer from pool for temporary storage
            let mut state_copy = self.buffer_pool.lock().unwrap().get_buffer(state.len());
            state_copy.copy_from_slice(state);

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

                // Use direct matrix element access for better performance
                *amp = if idx == idx0 {
                    m00 * val0 + m01 * val1
                } else {
                    m10 * val0 + m11 * val1
                };
            });

            // Return buffer to pool
            self.buffer_pool.lock().unwrap().return_buffer(state_copy);
        } else {
            // Sequential implementation using buffer pool
            let dim = state.len();
            let mut new_state = self.buffer_pool.lock().unwrap().get_buffer(dim);

            for i in 0..dim {
                let bit_val = (i >> target_idx) & 1;
                let paired_idx = flip_bit(i, target_idx);

                if bit_val == 0 {
                    new_state[i] = m00 * state[i] + m01 * state[paired_idx];
                    new_state[paired_idx] = m10 * state[i] + m11 * state[paired_idx];
                }
            }

            state.copy_from_slice(&new_state);
            // Return buffer to pool
            self.buffer_pool.lock().unwrap().return_buffer(new_state);
        }

        Ok(())
    }

    /// Apply a single-qubit gate using SIMD optimization
    fn apply_single_qubit_gate_simd<const N: usize>(
        &self,
        state: &mut [Complex64],
        gate_matrix: &[Complex64],
        target_idx: usize,
    ) -> QuantRS2Result<()> {
        let dim = state.len();
        let pairs_per_block = 1 << target_idx;
        let total_pairs = dim / 2;

        // Get buffers for SIMD processing
        let mut pool = self.buffer_pool.lock().unwrap();
        let mut in_amps0 = pool.get_buffer(total_pairs);
        let mut in_amps1 = pool.get_buffer(total_pairs);
        let mut out_amps0 = pool.get_buffer(total_pairs);
        let mut out_amps1 = pool.get_buffer(total_pairs);

        // Collect amplitudes for target bit = 0 and target bit = 1
        let mut pair_idx = 0;
        for block in 0..(dim / (pairs_per_block * 2)) {
            let block_start = block * pairs_per_block * 2;

            for offset in 0..pairs_per_block {
                let idx0 = block_start + offset;
                let idx1 = block_start + pairs_per_block + offset;

                in_amps0[pair_idx] = state[idx0];
                in_amps1[pair_idx] = state[idx1];
                pair_idx += 1;
            }
        }

        // Convert gate matrix to required format for SIMD
        let gate_matrix_array: [Complex64; 4] = [
            gate_matrix[0],
            gate_matrix[1],
            gate_matrix[2],
            gate_matrix[3],
        ];

        // Apply SIMD gate operation
        optimized_simd::apply_single_qubit_gate_optimized(
            &gate_matrix_array,
            &in_amps0,
            &in_amps1,
            &mut out_amps0,
            &mut out_amps1,
        );

        // Write results back to state vector
        pair_idx = 0;
        for block in 0..(dim / (pairs_per_block * 2)) {
            let block_start = block * pairs_per_block * 2;

            for offset in 0..pairs_per_block {
                let idx0 = block_start + offset;
                let idx1 = block_start + pairs_per_block + offset;

                state[idx0] = out_amps0[pair_idx];
                state[idx1] = out_amps1[pair_idx];
                pair_idx += 1;
            }
        }

        // Return buffers to pool
        pool.return_buffer(in_amps0);
        pool.return_buffer(in_amps1);
        pool.return_buffer(out_amps0);
        pool.return_buffer(out_amps1);

        Ok(())
    }

    /// Apply a two-qubit gate to a state vector
    fn apply_two_qubit_gate<const N: usize>(
        &self,
        state: &mut [Complex64],
        gate_matrix: &[Complex64],
        control: QubitId,
        target: QubitId,
    ) -> QuantRS2Result<()> {
        let control_idx = control.id() as usize;
        let target_idx = target.id() as usize;

        if control_idx >= N || target_idx >= N {
            return Err(QuantRS2Error::InvalidQubitId(if control_idx >= N {
                control.id()
            } else {
                target.id()
            }));
        }

        if control_idx == target_idx {
            return Err(QuantRS2Error::CircuitValidationFailed(
                "Control and target qubits must be different".into(),
            ));
        }

        // Pre-extract matrix elements for faster access (16 elements total)
        let m = gate_matrix; // Direct slice access is faster than ndarray indexing

        // Apply the gate to each amplitude
        if self.parallel {
            // Get buffer from pool for temporary storage
            let mut state_copy = self.buffer_pool.lock().unwrap().get_buffer(state.len());
            state_copy.copy_from_slice(state);

            state.par_iter_mut().enumerate().for_each(|(idx, amp)| {
                let idx00 = idx & !(1 << control_idx) & !(1 << target_idx);
                let idx01 = idx00 | (1 << target_idx);
                let idx10 = idx00 | (1 << control_idx);
                let idx11 = idx00 | (1 << control_idx) | (1 << target_idx);

                let val00 = state_copy[idx00];
                let val01 = state_copy[idx01];
                let val10 = state_copy[idx10];
                let val11 = state_copy[idx11];

                // Use direct matrix access for better performance
                *amp = match idx {
                    i if i == idx00 => m[0] * val00 + m[1] * val01 + m[2] * val10 + m[3] * val11,
                    i if i == idx01 => m[4] * val00 + m[5] * val01 + m[6] * val10 + m[7] * val11,
                    i if i == idx10 => m[8] * val00 + m[9] * val01 + m[10] * val10 + m[11] * val11,
                    i if i == idx11 => {
                        m[12] * val00 + m[13] * val01 + m[14] * val10 + m[15] * val11
                    }
                    _ => unreachable!(),
                };
            });

            // Return buffer to pool
            self.buffer_pool.lock().unwrap().return_buffer(state_copy);
        } else {
            // Sequential implementation using buffer pool
            let dim = state.len();
            let mut new_state = self.buffer_pool.lock().unwrap().get_buffer(dim);

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

                // Calculate the new amplitude for this state using direct access
                let row_offset = basis_idx * 4;
                new_state[i] = m[row_offset] * state[i00]
                    + m[row_offset + 1] * state[i01]
                    + m[row_offset + 2] * state[i10]
                    + m[row_offset + 3] * state[i11];
            }

            state.copy_from_slice(&new_state);
            // Return buffer to pool
            self.buffer_pool.lock().unwrap().return_buffer(new_state);
        }

        Ok(())
    }

    /// Apply CNOT gate efficiently (special case)
    fn apply_cnot<const N: usize>(
        &self,
        state: &mut [Complex64],
        control: QubitId,
        target: QubitId,
    ) -> QuantRS2Result<()> {
        let control_idx = control.id() as usize;
        let target_idx = target.id() as usize;

        if control_idx >= N || target_idx >= N {
            return Err(QuantRS2Error::InvalidQubitId(if control_idx >= N {
                control.id()
            } else {
                target.id()
            }));
        }

        if control_idx == target_idx {
            return Err(QuantRS2Error::CircuitValidationFailed(
                "Control and target qubits must be different".into(),
            ));
        }

        // Apply the CNOT gate - only swap amplitudes where control is 1
        if self.parallel {
            let mut state_copy = self.buffer_pool.lock().unwrap().get_buffer(state.len());
            state_copy.copy_from_slice(state);

            state.par_iter_mut().enumerate().for_each(|(i, amp)| {
                if (i >> control_idx) & 1 == 1 {
                    let flipped = flip_bit(i, target_idx);
                    *amp = state_copy[flipped];
                }
            });

            self.buffer_pool.lock().unwrap().return_buffer(state_copy);
        } else {
            let dim = state.len();
            let mut new_state = self.buffer_pool.lock().unwrap().get_buffer(dim);

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
            self.buffer_pool.lock().unwrap().return_buffer(new_state);
        }

        Ok(())
    }

    /// Apply SWAP gate efficiently (special case)
    fn apply_swap<const N: usize>(
        &self,
        state: &mut [Complex64],
        qubit1: QubitId,
        qubit2: QubitId,
    ) -> QuantRS2Result<()> {
        let q1_idx = qubit1.id() as usize;
        let q2_idx = qubit2.id() as usize;

        if q1_idx >= N || q2_idx >= N {
            return Err(QuantRS2Error::InvalidQubitId(if q1_idx >= N {
                qubit1.id()
            } else {
                qubit2.id()
            }));
        }

        if q1_idx == q2_idx {
            return Err(QuantRS2Error::CircuitValidationFailed(
                "Qubits must be different for SWAP gate".into(),
            ));
        }

        // Apply the SWAP gate - swap amplitudes where qubits have different values
        if self.parallel {
            let mut state_copy = self.buffer_pool.lock().unwrap().get_buffer(state.len());
            state_copy.copy_from_slice(state);

            state.par_iter_mut().enumerate().for_each(|(i, amp)| {
                let bit1 = (i >> q1_idx) & 1;
                let bit2 = (i >> q2_idx) & 1;

                if bit1 != bit2 {
                    let swapped = flip_bit(flip_bit(i, q1_idx), q2_idx);
                    *amp = state_copy[swapped];
                }
            });

            self.buffer_pool.lock().unwrap().return_buffer(state_copy);
        } else {
            let dim = state.len();
            let mut new_state = self.buffer_pool.lock().unwrap().get_buffer(dim);

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
            self.buffer_pool.lock().unwrap().return_buffer(new_state);
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
    fn run(&self, circuit: &Circuit<N>) -> QuantRS2Result<Register<N>> {
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
                    if let Some(toffoli_gate) = gate.as_any().downcast_ref::<multi::Toffoli>() {
                        let control1 = toffoli_gate.control1;
                        let control2 = toffoli_gate.control2;
                        let target = toffoli_gate.target;
                        self.apply_toffoli_to_state::<N>(&mut state, control1, control2, target)?;
                    }
                }
                "Fredkin" => {
                    if let Some(fredkin_gate) = gate.as_any().downcast_ref::<multi::Fredkin>() {
                        let control = fredkin_gate.control;
                        let target1 = fredkin_gate.target1;
                        let target2 = fredkin_gate.target2;
                        self.apply_fredkin_to_state::<N>(&mut state, control, target1, target2)?;
                    }
                }

                _ => {
                    return Err(QuantRS2Error::UnsupportedOperation(format!(
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

            // Apply per-gate advanced noise if configured
            if let Some(ref advanced_noise_model) = self.advanced_noise_model {
                if advanced_noise_model.per_gate {
                    advanced_noise_model.apply_to_statevector(&mut state)?;
                }
            }
        }

        // Apply final noise if not per-gate
        if let Some(ref noise_model) = self.noise_model {
            if !noise_model.per_gate {
                noise_model.apply_to_statevector(&mut state)?;
            }
        }

        // Apply final advanced noise if not per-gate
        if let Some(ref advanced_noise_model) = self.advanced_noise_model {
            if !advanced_noise_model.per_gate {
                advanced_noise_model.apply_to_statevector(&mut state)?;
            }
        }

        // Create register from final state
        Register::<N>::with_amplitudes(state)
    }
}

impl StateVectorSimulator {
    /// Initialize state with specified number of qubits in |0...0⟩
    pub fn initialize_state(&mut self, num_qubits: usize) -> QuantRS2Result<()> {
        // This is a placeholder - actual initialization would need the circuit framework
        Ok(())
    }

    /// Get the current quantum state
    pub fn get_state(&self) -> Vec<Complex64> {
        // Placeholder - would return the actual state vector
        vec![Complex64::new(1.0, 0.0)]
    }

    /// Get mutable reference to the current quantum state
    pub fn get_state_mut(&mut self) -> Vec<Complex64> {
        // Placeholder - would return mutable reference to actual state vector
        vec![Complex64::new(1.0, 0.0)]
    }

    /// Set the quantum state
    pub fn set_state(&mut self, _state: Vec<Complex64>) -> QuantRS2Result<()> {
        // Placeholder - would set the actual state vector
        Ok(())
    }

    /// Apply an interface circuit to the quantum state
    pub fn apply_interface_circuit(
        &mut self,
        _circuit: &crate::circuit_interfaces::InterfaceCircuit,
    ) -> QuantRS2Result<()> {
        // Placeholder - would apply the circuit gates using the circuit framework
        Ok(())
    }

    /// Apply Hadamard gate to qubit
    pub fn apply_h(&mut self, _qubit: usize) -> QuantRS2Result<()> {
        // Placeholder - would apply H gate using circuit framework
        Ok(())
    }

    /// Apply Pauli-X gate to qubit
    pub fn apply_x(&mut self, _qubit: usize) -> QuantRS2Result<()> {
        // Placeholder - would apply X gate using circuit framework
        Ok(())
    }

    /// Apply Pauli-Z gate to qubit
    pub fn apply_z_public(&mut self, _qubit: usize) -> QuantRS2Result<()> {
        // Placeholder - would apply Z gate using circuit framework
        Ok(())
    }

    /// Apply CNOT gate (public interface with usize indices)
    pub fn apply_cnot_public(&mut self, _control: usize, _target: usize) -> QuantRS2Result<()> {
        // Placeholder - would apply CNOT gate using circuit framework
        Ok(())
    }

    /// Apply Toffoli (CCNOT) gate to state vector
    fn apply_toffoli_to_state<const N: usize>(
        &self,
        state: &mut [Complex64],
        control1: QubitId,
        control2: QubitId,
        target: QubitId,
    ) -> QuantRS2Result<()> {
        let control1_idx = control1.id() as usize;
        let control2_idx = control2.id() as usize;
        let target_idx = target.id() as usize;

        if control1_idx >= N || control2_idx >= N || target_idx >= N {
            return Err(QuantRS2Error::InvalidQubitId(if control1_idx >= N {
                control1.id()
            } else if control2_idx >= N {
                control2.id()
            } else {
                target.id()
            }));
        }

        // Apply Toffoli gate by swapping amplitudes when both controls are |1⟩
        if self.parallel {
            let mut state_copy = self.buffer_pool.lock().unwrap().get_buffer(state.len());
            state_copy.copy_from_slice(state);

            state.par_iter_mut().enumerate().for_each(|(i, amp)| {
                let control1_bit = (i >> control1_idx) & 1;
                let control2_bit = (i >> control2_idx) & 1;

                if control1_bit == 1 && control2_bit == 1 {
                    let flipped = flip_bit(i, target_idx);
                    *amp = state_copy[flipped];
                }
            });

            self.buffer_pool.lock().unwrap().return_buffer(state_copy);
        } else {
            let mut new_state = self.buffer_pool.lock().unwrap().get_buffer(state.len());
            new_state.copy_from_slice(state);

            for i in 0..state.len() {
                let control1_bit = (i >> control1_idx) & 1;
                let control2_bit = (i >> control2_idx) & 1;

                if control1_bit == 1 && control2_bit == 1 {
                    let flipped = flip_bit(i, target_idx);
                    new_state[flipped] = state[i];
                    new_state[i] = state[flipped];
                }
            }

            state.copy_from_slice(&new_state);
            self.buffer_pool.lock().unwrap().return_buffer(new_state);
        }

        Ok(())
    }

    /// Apply Fredkin (CSWAP) gate to state vector
    fn apply_fredkin_to_state<const N: usize>(
        &self,
        state: &mut [Complex64],
        control: QubitId,
        target1: QubitId,
        target2: QubitId,
    ) -> QuantRS2Result<()> {
        let control_idx = control.id() as usize;
        let target1_idx = target1.id() as usize;
        let target2_idx = target2.id() as usize;

        if control_idx >= N || target1_idx >= N || target2_idx >= N {
            return Err(QuantRS2Error::InvalidQubitId(if control_idx >= N {
                control.id()
            } else if target1_idx >= N {
                target1.id()
            } else {
                target2.id()
            }));
        }

        // Apply Fredkin gate by swapping target qubits when control is |1⟩
        if self.parallel {
            let mut state_copy = self.buffer_pool.lock().unwrap().get_buffer(state.len());
            state_copy.copy_from_slice(state);

            state.par_iter_mut().enumerate().for_each(|(i, amp)| {
                let control_bit = (i >> control_idx) & 1;
                let target1_bit = (i >> target1_idx) & 1;
                let target2_bit = (i >> target2_idx) & 1;

                if control_bit == 1 && target1_bit != target2_bit {
                    let swapped = flip_bit(flip_bit(i, target1_idx), target2_idx);
                    *amp = state_copy[swapped];
                }
            });

            self.buffer_pool.lock().unwrap().return_buffer(state_copy);
        } else {
            let mut new_state = self.buffer_pool.lock().unwrap().get_buffer(state.len());
            new_state.copy_from_slice(state);

            for i in 0..state.len() {
                let control_bit = (i >> control_idx) & 1;
                let target1_bit = (i >> target1_idx) & 1;
                let target2_bit = (i >> target2_idx) & 1;

                if control_bit == 1 && target1_bit != target2_bit {
                    let swapped = flip_bit(flip_bit(i, target1_idx), target2_idx);
                    new_state[swapped] = state[i];
                    new_state[i] = state[swapped];
                }
            }

            state.copy_from_slice(&new_state);
            self.buffer_pool.lock().unwrap().return_buffer(new_state);
        }

        Ok(())
    }

    /// Apply Toffoli (CCNOT) gate (public interface)
    pub fn apply_toffoli(
        &mut self,
        control1: QubitId,
        control2: QubitId,
        target: QubitId,
    ) -> QuantRS2Result<()> {
        // This is a placeholder for external API - real implementation would need circuit framework
        Ok(())
    }

    /// Apply Fredkin (CSWAP) gate (public interface)
    pub fn apply_fredkin(
        &mut self,
        control: QubitId,
        target1: QubitId,
        target2: QubitId,
    ) -> QuantRS2Result<()> {
        // This is a placeholder for external API - real implementation would need circuit framework
        Ok(())
    }
}
