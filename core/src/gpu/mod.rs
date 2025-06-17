//! GPU acceleration backend for quantum operations
//!
//! This module provides an abstraction layer for GPU-accelerated quantum
//! computations, supporting multiple backends (CUDA, Metal, Vulkan, etc.)

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use std::sync::Arc;

pub mod cpu_backend;
#[cfg(feature = "cuda")]
pub mod cuda_backend;
#[cfg(feature = "metal")]
pub mod metal_backend;
#[cfg(feature = "vulkan")]
pub mod vulkan_backend;

// Enhanced GPU optimization modules
pub mod adaptive_simd;
pub mod large_scale_simulation;
pub mod specialized_kernels;

// Re-export key optimization components
pub use adaptive_simd::{
    apply_batch_gates_adaptive, apply_single_qubit_adaptive, apply_two_qubit_adaptive,
    get_adaptive_performance_report, initialize_adaptive_simd, AdaptiveSimdDispatcher, CpuFeatures,
    SimdVariant,
};
pub use large_scale_simulation::{
    LargeScaleGateType, LargeScaleObservable, LargeScalePerformanceStats, LargeScaleSimAccelerator,
    LargeScaleSimConfig, LargeScaleStateVectorSim, LargeScaleTensorContractor, SimulationTaskType,
    TensorDecompositionType,
};
pub use specialized_kernels::{
    FusionType, OptimizationConfig, PerformanceReport, PostQuantumCompressionType,
    SpecializedGpuKernels,
};

/// GPU memory buffer abstraction
pub trait GpuBuffer: Send + Sync {
    /// Get the size of the buffer in bytes
    fn size(&self) -> usize;

    /// Copy data from host to device
    fn upload(&mut self, data: &[Complex64]) -> QuantRS2Result<()>;

    /// Copy data from device to host
    fn download(&self, data: &mut [Complex64]) -> QuantRS2Result<()>;

    /// Synchronize GPU operations
    fn sync(&self) -> QuantRS2Result<()>;

    /// Enable downcasting to concrete types
    fn as_any(&self) -> &dyn std::any::Any;

    /// Enable mutable downcasting to concrete types
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// Enhanced GPU kernel for specialized quantum operations
pub trait SpecializedGpuKernel: Send + Sync {
    /// Apply a holonomic gate with optimized GPU execution
    fn apply_holonomic_gate(
        &self,
        state: &mut dyn GpuBuffer,
        holonomy_matrix: &[Complex64],
        target_qubits: &[QubitId],
    ) -> QuantRS2Result<()>;

    /// Apply post-quantum cryptographic hash gate
    fn apply_post_quantum_hash_gate(
        &self,
        state: &mut dyn GpuBuffer,
        hash_circuit: &[Complex64],
        compression_type: PostQuantumCompressionType,
    ) -> QuantRS2Result<()>;

    /// Apply quantum ML attention mechanism
    fn apply_quantum_ml_attention(
        &self,
        state: &mut dyn GpuBuffer,
        query_params: &[Complex64],
        key_params: &[Complex64],
        value_params: &[Complex64],
        num_heads: usize,
    ) -> QuantRS2Result<()>;

    /// Apply fused gate sequences for optimal performance
    fn apply_fused_gate_sequence(
        &self,
        state: &mut dyn GpuBuffer,
        gates: &[Box<dyn GateOp>],
    ) -> QuantRS2Result<()>;

    /// Apply tensor network contraction
    fn apply_tensor_contraction(
        &self,
        tensor_data: &mut dyn GpuBuffer,
        contraction_indices: &[usize],
        bond_dimension: usize,
    ) -> QuantRS2Result<()>;
}

/// GPU kernel for quantum operations
pub trait GpuKernel: Send + Sync {
    /// Apply a single-qubit gate
    fn apply_single_qubit_gate(
        &self,
        state: &mut dyn GpuBuffer,
        gate_matrix: &[Complex64; 4],
        qubit: QubitId,
        n_qubits: usize,
    ) -> QuantRS2Result<()>;

    /// Apply a two-qubit gate
    fn apply_two_qubit_gate(
        &self,
        state: &mut dyn GpuBuffer,
        gate_matrix: &[Complex64; 16],
        control: QubitId,
        target: QubitId,
        n_qubits: usize,
    ) -> QuantRS2Result<()>;

    /// Apply a multi-qubit gate
    fn apply_multi_qubit_gate(
        &self,
        state: &mut dyn GpuBuffer,
        gate_matrix: &Array2<Complex64>,
        qubits: &[QubitId],
        n_qubits: usize,
    ) -> QuantRS2Result<()>;

    /// Measure a qubit
    fn measure_qubit(
        &self,
        state: &dyn GpuBuffer,
        qubit: QubitId,
        n_qubits: usize,
    ) -> QuantRS2Result<(bool, f64)>;

    /// Calculate expectation value
    fn expectation_value(
        &self,
        state: &dyn GpuBuffer,
        observable: &Array2<Complex64>,
        qubits: &[QubitId],
        n_qubits: usize,
    ) -> QuantRS2Result<f64>;
}

/// Enhanced GPU backend trait for specialized quantum computations
pub trait EnhancedGpuBackend: GpuBackend {
    /// Get the specialized kernel implementation
    fn specialized_kernel(&self) -> Option<&dyn SpecializedGpuKernel>;

    /// Apply a holonomic gate with GPU optimization
    fn apply_holonomic_gate(
        &self,
        state: &mut dyn GpuBuffer,
        holonomy_matrix: &[Complex64],
        target_qubits: &[QubitId],
    ) -> QuantRS2Result<()> {
        if let Some(kernel) = self.specialized_kernel() {
            kernel.apply_holonomic_gate(state, holonomy_matrix, target_qubits)
        } else {
            Err(QuantRS2Error::UnsupportedOperation(
                "Holonomic gates not supported by this backend".to_string(),
            ))
        }
    }

    /// Apply post-quantum cryptographic operations
    fn apply_post_quantum_crypto(
        &self,
        state: &mut dyn GpuBuffer,
        hash_circuit: &[Complex64],
        compression_type: PostQuantumCompressionType,
    ) -> QuantRS2Result<()> {
        if let Some(kernel) = self.specialized_kernel() {
            kernel.apply_post_quantum_hash_gate(state, hash_circuit, compression_type)
        } else {
            Err(QuantRS2Error::UnsupportedOperation(
                "Post-quantum crypto gates not supported by this backend".to_string(),
            ))
        }
    }

    /// Apply quantum ML operations
    fn apply_quantum_ml_attention(
        &self,
        state: &mut dyn GpuBuffer,
        query_params: &[Complex64],
        key_params: &[Complex64],
        value_params: &[Complex64],
        num_heads: usize,
    ) -> QuantRS2Result<()> {
        if let Some(kernel) = self.specialized_kernel() {
            kernel.apply_quantum_ml_attention(
                state,
                query_params,
                key_params,
                value_params,
                num_heads,
            )
        } else {
            Err(QuantRS2Error::UnsupportedOperation(
                "Quantum ML attention not supported by this backend".to_string(),
            ))
        }
    }

    /// Apply optimized gate fusion
    fn apply_fused_gates(
        &self,
        state: &mut dyn GpuBuffer,
        gates: &[Box<dyn GateOp>],
    ) -> QuantRS2Result<()> {
        if let Some(kernel) = self.specialized_kernel() {
            kernel.apply_fused_gate_sequence(state, gates)
        } else {
            // Fallback to applying gates individually
            for gate in gates {
                let qubits = gate.qubits();
                self.apply_gate(state, gate.as_ref(), &qubits, qubits.len())?;
            }
            Ok(())
        }
    }

    /// Get optimization configuration
    fn optimization_config(&self) -> OptimizationConfig {
        OptimizationConfig::default()
    }

    /// Get performance statistics
    fn performance_stats(&self) -> PerformanceReport {
        PerformanceReport {
            average_kernel_times: std::collections::HashMap::new(),
            cache_hit_rate: 0.0,
            tensor_core_utilization: 0.0,
            memory_bandwidth_utilization: 0.0,
        }
    }
}

/// GPU backend trait for quantum computations
pub trait GpuBackend: Send + Sync {
    /// Check if this backend is available on the current system
    fn is_available() -> bool
    where
        Self: Sized;

    /// Get the name of this backend
    fn name(&self) -> &str;

    /// Get device information
    fn device_info(&self) -> String;

    /// Allocate a GPU buffer for a state vector
    fn allocate_state_vector(&self, n_qubits: usize) -> QuantRS2Result<Box<dyn GpuBuffer>>;

    /// Allocate a GPU buffer for a density matrix
    fn allocate_density_matrix(&self, n_qubits: usize) -> QuantRS2Result<Box<dyn GpuBuffer>>;

    /// Get the kernel implementation
    fn kernel(&self) -> &dyn GpuKernel;

    /// Apply a quantum gate
    fn apply_gate(
        &self,
        state: &mut dyn GpuBuffer,
        gate: &dyn GateOp,
        qubits: &[QubitId],
        n_qubits: usize,
    ) -> QuantRS2Result<()> {
        match qubits.len() {
            1 => {
                let matrix = gate.matrix()?;
                let gate_array: [Complex64; 4] = [matrix[0], matrix[1], matrix[2], matrix[3]];
                self.kernel()
                    .apply_single_qubit_gate(state, &gate_array, qubits[0], n_qubits)
            }
            2 => {
                let matrix = gate.matrix()?;
                let mut gate_array = [Complex64::new(0.0, 0.0); 16];
                for (i, &val) in matrix.iter().enumerate() {
                    gate_array[i] = val;
                }
                self.kernel().apply_two_qubit_gate(
                    state,
                    &gate_array,
                    qubits[0],
                    qubits[1],
                    n_qubits,
                )
            }
            _ => {
                let matrix_vec = gate.matrix()?;
                let size = (1 << qubits.len(), 1 << qubits.len());
                let matrix = Array2::from_shape_vec(size, matrix_vec)?;
                self.kernel()
                    .apply_multi_qubit_gate(state, &matrix, qubits, n_qubits)
            }
        }
    }

    /// Measure a qubit and collapse the state
    fn measure(
        &self,
        state: &mut dyn GpuBuffer,
        qubit: QubitId,
        n_qubits: usize,
    ) -> QuantRS2Result<bool> {
        let (outcome, _prob) = self.kernel().measure_qubit(state, qubit, n_qubits)?;
        Ok(outcome)
    }

    /// Get measurement probability without collapsing
    fn get_probability(
        &self,
        state: &dyn GpuBuffer,
        qubit: QubitId,
        n_qubits: usize,
    ) -> QuantRS2Result<f64> {
        let (_outcome, prob) = self.kernel().measure_qubit(state, qubit, n_qubits)?;
        Ok(prob)
    }
}

/// GPU-accelerated state vector
pub struct GpuStateVector {
    /// The GPU backend
    backend: Arc<dyn GpuBackend>,
    /// The GPU buffer holding the state
    buffer: Box<dyn GpuBuffer>,
    /// Number of qubits
    n_qubits: usize,
}

impl GpuStateVector {
    /// Create a new GPU state vector
    pub fn new(backend: Arc<dyn GpuBackend>, n_qubits: usize) -> QuantRS2Result<Self> {
        let buffer = backend.allocate_state_vector(n_qubits)?;
        Ok(Self {
            backend,
            buffer,
            n_qubits,
        })
    }

    /// Initialize to |00...0⟩ state
    pub fn initialize_zero_state(&mut self) -> QuantRS2Result<()> {
        let size = 1 << self.n_qubits;
        let mut data = vec![Complex64::new(0.0, 0.0); size];
        data[0] = Complex64::new(1.0, 0.0);
        self.buffer.upload(&data)
    }

    /// Apply a gate
    pub fn apply_gate(&mut self, gate: &dyn GateOp, qubits: &[QubitId]) -> QuantRS2Result<()> {
        self.backend
            .apply_gate(self.buffer.as_mut(), gate, qubits, self.n_qubits)
    }

    /// Measure a qubit
    pub fn measure(&mut self, qubit: QubitId) -> QuantRS2Result<bool> {
        self.backend
            .measure(self.buffer.as_mut(), qubit, self.n_qubits)
    }

    /// Get the state vector as a host array
    pub fn to_array(&self) -> QuantRS2Result<Array1<Complex64>> {
        let size = 1 << self.n_qubits;
        let mut data = vec![Complex64::new(0.0, 0.0); size];
        self.buffer.download(&mut data)?;
        Ok(Array1::from_vec(data))
    }

    /// Get measurement probabilities for all basis states
    pub fn get_probabilities(&self) -> QuantRS2Result<Vec<f64>> {
        let state = self.to_array()?;
        Ok(state.iter().map(|c| c.norm_sqr()).collect())
    }
}

/// GPU backend factory
pub struct GpuBackendFactory;

impl GpuBackendFactory {
    /// Create the best available GPU backend
    pub fn create_best_available() -> QuantRS2Result<Arc<dyn GpuBackend>> {
        // Try backends in order of preference
        #[cfg(feature = "cuda")]
        if cuda_backend::CudaBackend::is_available() {
            return Ok(Arc::new(cuda_backend::CudaBackend::new()?));
        }

        #[cfg(feature = "metal")]
        if metal_backend::MetalBackend::is_available() {
            return Ok(Arc::new(metal_backend::MetalBackend::new()?));
        }

        #[cfg(feature = "vulkan")]
        if vulkan_backend::VulkanBackend::is_available() {
            return Ok(Arc::new(vulkan_backend::VulkanBackend::new()?));
        }

        // Fallback to CPU backend
        Ok(Arc::new(cpu_backend::CpuBackend::new()))
    }

    /// Create a specific backend
    pub fn create_backend(backend_type: &str) -> QuantRS2Result<Arc<dyn GpuBackend>> {
        match backend_type.to_lowercase().as_str() {
            #[cfg(feature = "cuda")]
            "cuda" => Ok(Arc::new(cuda_backend::CudaBackend::new()?)),

            #[cfg(feature = "metal")]
            "metal" => Ok(Arc::new(metal_backend::MetalBackend::new()?)),

            #[cfg(feature = "vulkan")]
            "vulkan" => Ok(Arc::new(vulkan_backend::VulkanBackend::new()?)),

            "cpu" => Ok(Arc::new(cpu_backend::CpuBackend::new())),

            _ => Err(QuantRS2Error::InvalidInput(format!(
                "Unknown backend type: {}",
                backend_type
            ))),
        }
    }

    /// List available backends
    pub fn available_backends() -> Vec<&'static str> {
        #[allow(unused_mut)]
        let mut backends = vec!["cpu"];

        #[cfg(feature = "cuda")]
        if cuda_backend::CudaBackend::is_available() {
            backends.push("cuda");
        }

        #[cfg(feature = "metal")]
        if metal_backend::MetalBackend::is_available() {
            backends.push("metal");
        }

        #[cfg(feature = "vulkan")]
        if vulkan_backend::VulkanBackend::is_available() {
            backends.push("vulkan");
        }

        backends
    }
}

/// Configuration for GPU operations
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Preferred backend (None for auto-selection)
    pub backend: Option<String>,
    /// Maximum GPU memory to use (in bytes)
    pub max_memory: Option<usize>,
    /// Number of GPU threads/work items
    pub num_threads: Option<usize>,
    /// Enable profiling
    pub enable_profiling: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            backend: None,
            max_memory: None,
            num_threads: None,
            enable_profiling: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::single::Hadamard;

    #[test]
    fn test_gpu_backend_factory() {
        let backends = GpuBackendFactory::available_backends();
        assert!(backends.contains(&"cpu"));

        // Should always be able to create CPU backend
        let backend = GpuBackendFactory::create_backend("cpu").unwrap();
        assert_eq!(backend.name(), "CPU");
    }

    #[test]
    fn test_gpu_state_vector() {
        let backend = GpuBackendFactory::create_best_available().unwrap();
        let mut state = GpuStateVector::new(backend, 2).unwrap();

        // Initialize to |00⟩
        state.initialize_zero_state().unwrap();

        // Apply Hadamard to first qubit
        let h_gate = Hadamard { target: QubitId(0) };
        state.apply_gate(&h_gate, &[QubitId(0)]).unwrap();

        // Get probabilities
        let probs = state.get_probabilities().unwrap();
        assert_eq!(probs.len(), 4);

        // Should be in equal superposition on first qubit
        // With our bit ordering (LSB), |00⟩ and |01⟩ should have probability 0.5 each
        assert!((probs[0] - 0.5).abs() < 1e-10); // |00⟩
        assert!((probs[1] - 0.5).abs() < 1e-10); // |01⟩
        assert!((probs[2] - 0.0).abs() < 1e-10); // |10⟩
        assert!((probs[3] - 0.0).abs() < 1e-10); // |11⟩
    }
}
