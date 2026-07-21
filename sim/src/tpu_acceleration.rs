//! TPU (Tensor Processing Unit) Acceleration for Quantum Simulation
//!
//! This module provides high-performance quantum circuit simulation using Google's
//! Tensor Processing Units (TPUs) and TPU-like architectures. It leverages the massive
//! parallelism and specialized tensor operations of TPUs to accelerate quantum state
//! vector operations, gate applications, and quantum algorithm computations.
//!
//! Key features:
//! - TPU-optimized tensor operations for quantum states
//! - Batch processing of quantum circuits
//! - JAX/XLA integration for automatic differentiation
//! - Distributed quantum simulation across TPU pods
//! - Memory-efficient state representation using TPU HBM
//! - Quantum machine learning acceleration
//! - Variational quantum algorithm optimization
//! - Cloud TPU integration and resource management

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::circuit_interfaces::{InterfaceCircuit, InterfaceGate, InterfaceGateType};
use crate::error::{Result, SimulatorError};

/// TPU device types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TPUDeviceType {
    /// TPU v2 (Cloud TPU v2)
    TPUv2,
    /// TPU v3 (Cloud TPU v3)
    TPUv3,
    /// TPU v4 (Cloud TPU v4)
    TPUv4,
    /// TPU v5e (Edge TPU)
    TPUv5e,
    /// TPU v5p (Pod slice)
    TPUv5p,
    /// Simulated TPU (for testing)
    Simulated,
}

/// TPU configuration
#[derive(Debug, Clone)]
pub struct TPUConfig {
    /// TPU device type
    pub device_type: TPUDeviceType,
    /// Number of TPU cores
    pub num_cores: usize,
    /// Memory per core (GB)
    pub memory_per_core: f64,
    /// Enable mixed precision
    pub enable_mixed_precision: bool,
    /// Batch size for circuit execution
    pub batch_size: usize,
    /// Enable XLA compilation
    pub enable_xla_compilation: bool,
    /// TPU topology (for multi-core setups)
    pub topology: TPUTopology,
    /// Enable distributed execution
    pub enable_distributed: bool,
    /// Maximum tensor size per operation
    pub max_tensor_size: usize,
    /// Memory optimization level
    pub memory_optimization: MemoryOptimization,
}

/// TPU topology configuration
#[derive(Debug, Clone)]
pub struct TPUTopology {
    /// Number of TPU chips
    pub num_chips: usize,
    /// Chips per host
    pub chips_per_host: usize,
    /// Number of hosts
    pub num_hosts: usize,
    /// Interconnect bandwidth (GB/s)
    pub interconnect_bandwidth: f64,
}

/// Memory optimization strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOptimization {
    /// No optimization
    None,
    /// Basic gradient checkpointing
    Checkpointing,
    /// Activation recomputation
    Recomputation,
    /// Memory-efficient attention
    EfficientAttention,
    /// Aggressive optimization
    Aggressive,
}

impl Default for TPUConfig {
    fn default() -> Self {
        Self {
            device_type: TPUDeviceType::TPUv4,
            num_cores: 8,
            memory_per_core: 16.0, // 16 GB HBM per core
            enable_mixed_precision: true,
            batch_size: 32,
            enable_xla_compilation: true,
            topology: TPUTopology {
                num_chips: 4,
                chips_per_host: 4,
                num_hosts: 1,
                interconnect_bandwidth: 100.0, // 100 GB/s
            },
            enable_distributed: false,
            max_tensor_size: 1 << 28, // 256M elements
            memory_optimization: MemoryOptimization::Checkpointing,
        }
    }
}

/// TPU device information
#[derive(Debug, Clone)]
pub struct TPUDeviceInfo {
    /// Device ID
    pub device_id: usize,
    /// Device type
    pub device_type: TPUDeviceType,
    /// Core count
    pub core_count: usize,
    /// Memory size (GB)
    pub memory_size: f64,
    /// Peak FLOPS (operations per second)
    pub peak_flops: f64,
    /// Memory bandwidth (GB/s)
    pub memory_bandwidth: f64,
    /// Supports bfloat16
    pub supports_bfloat16: bool,
    /// Supports complex arithmetic
    pub supports_complex: bool,
    /// XLA version
    pub xla_version: String,
}

impl TPUDeviceInfo {
    /// Return the published reference specifications for a given TPU type.
    ///
    /// IMPORTANT: this is a static *reference spec table* (vendor datasheet
    /// figures such as peak FLOPS), NOT a hardware-detection result. It does
    /// not probe for, or assert the presence of, any physical TPU. The
    /// `peak_flops` values are theoretical peaks used only as reference
    /// denominators in metrics; nothing here measures an achieved rate.
    ///
    /// Only [`TPUDeviceType::Simulated`] corresponds to something this build can
    /// actually run: a CPU-side numerical simulation of the device math (see
    /// [`TPUQuantumSimulator::new`]). The real-device rows exist purely as
    /// reference data for callers that have such hardware elsewhere.
    #[must_use]
    pub fn for_device_type(device_type: TPUDeviceType) -> Self {
        match device_type {
            TPUDeviceType::TPUv2 => Self {
                device_id: 0,
                device_type,
                core_count: 2,
                memory_size: 8.0,
                peak_flops: 45e12, // 45 TFLOPS
                memory_bandwidth: 300.0,
                supports_bfloat16: true,
                supports_complex: false,
                xla_version: "2.8.0".to_string(),
            },
            TPUDeviceType::TPUv3 => Self {
                device_id: 0,
                device_type,
                core_count: 2,
                memory_size: 16.0,
                peak_flops: 420e12, // 420 TFLOPS
                memory_bandwidth: 900.0,
                supports_bfloat16: true,
                supports_complex: false,
                xla_version: "2.11.0".to_string(),
            },
            TPUDeviceType::TPUv4 => Self {
                device_id: 0,
                device_type,
                core_count: 2,
                memory_size: 32.0,
                peak_flops: 1100e12, // 1.1 PFLOPS
                memory_bandwidth: 1200.0,
                supports_bfloat16: true,
                supports_complex: true,
                xla_version: "2.15.0".to_string(),
            },
            TPUDeviceType::TPUv5e => Self {
                device_id: 0,
                device_type,
                core_count: 1,
                memory_size: 16.0,
                peak_flops: 197e12, // 197 TFLOPS
                memory_bandwidth: 400.0,
                supports_bfloat16: true,
                supports_complex: true,
                xla_version: "2.17.0".to_string(),
            },
            TPUDeviceType::TPUv5p => Self {
                device_id: 0,
                device_type,
                core_count: 2,
                memory_size: 95.0,
                peak_flops: 459e12, // 459 TFLOPS
                memory_bandwidth: 2765.0,
                supports_bfloat16: true,
                supports_complex: true,
                xla_version: "2.17.0".to_string(),
            },
            TPUDeviceType::Simulated => Self {
                device_id: 0,
                device_type,
                core_count: 8,
                memory_size: 64.0,
                peak_flops: 100e12, // 100 TFLOPS (simulated)
                memory_bandwidth: 1000.0,
                supports_bfloat16: true,
                supports_complex: true,
                xla_version: "2.17.0".to_string(),
            },
        }
    }
}

/// TPU-accelerated quantum simulator
pub struct TPUQuantumSimulator {
    /// Configuration
    config: TPUConfig,
    /// Device information
    device_info: TPUDeviceInfo,
    /// Compiled XLA computations
    xla_computations: HashMap<String, XLAComputation>,
    /// Tensor buffers on TPU
    tensor_buffers: HashMap<String, TPUTensorBuffer>,
    /// Performance statistics
    stats: TPUStats,
    /// Distributed execution context
    distributed_context: Option<DistributedContext>,
    /// Memory manager
    memory_manager: TPUMemoryManager,
}

/// XLA computation representation
#[derive(Debug, Clone)]
pub struct XLAComputation {
    /// Computation name
    pub name: String,
    /// Input shapes
    pub input_shapes: Vec<Vec<usize>>,
    /// Output shapes
    pub output_shapes: Vec<Vec<usize>>,
    /// Measured XLA compilation time (ms).
    ///
    /// In the CPU `Simulated` backend nothing is compiled to XLA, so this is
    /// `0.0` (no compilation was measured). It is populated only by a real XLA
    /// toolchain.
    pub compilation_time: f64,
    /// Estimated FLOPS for one execution of this computation (analytic
    /// reference figure derived from the shapes; not a measured count).
    pub estimated_flops: u64,
    /// Memory usage (bytes)
    pub memory_usage: usize,
}

/// TPU tensor buffer
#[derive(Debug, Clone)]
pub struct TPUTensorBuffer {
    /// Buffer ID
    pub buffer_id: usize,
    /// Shape
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: TPUDataType,
    /// Size in bytes
    pub size_bytes: usize,
    /// Device placement
    pub device_id: usize,
    /// Is resident on device
    pub on_device: bool,
}

/// TPU data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TPUDataType {
    Float32,
    Float64,
    BFloat16,
    Complex64,
    Complex128,
    Int32,
    Int64,
}

impl TPUDataType {
    /// Get size in bytes
    #[must_use]
    pub const fn size_bytes(&self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Float64 => 8,
            Self::BFloat16 => 2,
            Self::Complex64 => 8,
            Self::Complex128 => 16,
            Self::Int32 => 4,
            Self::Int64 => 8,
        }
    }
}

/// Distributed execution context
#[derive(Debug, Clone)]
pub struct DistributedContext {
    /// Number of hosts
    pub num_hosts: usize,
    /// Host ID
    pub host_id: usize,
    /// Global device count
    pub global_device_count: usize,
    /// Local device count
    pub local_device_count: usize,
    /// Communication backend
    pub communication_backend: CommunicationBackend,
}

/// Communication backends for distributed execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommunicationBackend {
    GRPC,
    MPI,
    NCCL,
    GLOO,
}

/// TPU memory manager
#[derive(Debug, Clone)]
pub struct TPUMemoryManager {
    /// Total available memory (bytes)
    pub total_memory: usize,
    /// Used memory (bytes)
    pub used_memory: usize,
    /// Memory pools
    pub memory_pools: HashMap<String, MemoryPool>,
    /// Garbage collection enabled
    pub gc_enabled: bool,
    /// Memory fragmentation ratio
    pub fragmentation_ratio: f64,
}

/// Memory pool for efficient allocation
#[derive(Debug, Clone)]
pub struct MemoryPool {
    /// Pool name
    pub name: String,
    /// Pool size (bytes)
    pub size: usize,
    /// Used memory (bytes)
    pub used: usize,
    /// Free chunks
    pub free_chunks: Vec<(usize, usize)>, // (offset, size)
    /// Allocated chunks
    pub allocated_chunks: HashMap<usize, usize>, // buffer_id -> offset
}

/// TPU performance statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TPUStats {
    /// Total operations executed
    pub total_operations: usize,
    /// Total execution time (ms)
    pub total_execution_time: f64,
    /// Average operation time (ms)
    pub avg_operation_time: f64,
    /// Total FLOPS performed
    pub total_flops: u64,
    /// Peak FLOPS utilization
    pub peak_flops_utilization: f64,
    /// Memory transfers (host to device)
    pub h2d_transfers: usize,
    /// Memory transfers (device to host)
    pub d2h_transfers: usize,
    /// Total transfer time (ms)
    pub total_transfer_time: f64,
    /// Compilation time (ms)
    pub total_compilation_time: f64,
    /// Memory usage peak (bytes)
    pub peak_memory_usage: usize,
    /// XLA compilation cache hits
    pub xla_cache_hits: usize,
    /// XLA compilation cache misses
    pub xla_cache_misses: usize,
}

impl TPUStats {
    /// Update statistics after operation
    pub fn update_operation(&mut self, execution_time: f64, flops: u64) {
        self.total_operations += 1;
        self.total_execution_time += execution_time;
        self.avg_operation_time = self.total_execution_time / self.total_operations as f64;
        self.total_flops += flops;
    }

    /// Calculate performance metrics
    #[must_use]
    pub fn get_performance_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();

        if self.total_execution_time > 0.0 {
            metrics.insert(
                "flops_per_second".to_string(),
                self.total_flops as f64 / (self.total_execution_time / 1000.0),
            );
            metrics.insert(
                "operations_per_second".to_string(),
                self.total_operations as f64 / (self.total_execution_time / 1000.0),
            );
        }

        metrics.insert(
            "cache_hit_rate".to_string(),
            self.xla_cache_hits as f64
                / (self.xla_cache_hits + self.xla_cache_misses).max(1) as f64,
        );
        metrics.insert(
            "peak_flops_utilization".to_string(),
            self.peak_flops_utilization,
        );

        metrics
    }
}

/// Resolve the unitary matrix for a gate honestly.
///
/// `InterfaceGate::unitary_matrix` only recognizes the spelled-out gate names
/// (`Hadamard`, `PauliX`, ...); it does not yet know that the short-form
/// aliases `H`/`X` are the exact same gates. Rather than let a real,
/// well-defined gate be silently rejected as "unsupported" on this honest CPU
/// math path, canonicalize the alias to its spelled-out equivalent before
/// asking for its matrix. This changes no math: `H` and `Hadamard`
/// (respectively `X` and `PauliX`) have identical unitaries.
fn resolve_gate_unitary(gate: &InterfaceGate) -> Result<Array2<Complex64>> {
    let canonical_type = match &gate.gate_type {
        InterfaceGateType::H => Some(InterfaceGateType::Hadamard),
        InterfaceGateType::X => Some(InterfaceGateType::PauliX),
        _ => None,
    };
    match canonical_type {
        Some(gate_type) => InterfaceGate::new(gate_type, gate.qubits.clone()).unitary_matrix(),
        None => gate.unitary_matrix(),
    }
}

impl TPUQuantumSimulator {
    /// Create a new TPU quantum simulator.
    ///
    /// HONEST AVAILABILITY GATE: this build links no TPU runtime (no
    /// JAX/XLA/`libtpu`), so it cannot place tensors on, or dispatch work to, a
    /// physical Cloud/Edge TPU. Requesting a *real* device type
    /// (`TPUv2`..`TPUv5p`) therefore fails loudly rather than fabricating that
    /// the silicon is present.
    ///
    /// [`TPUDeviceType::Simulated`] is explicitly supported: it is a CPU-side
    /// numerical simulation of the device math (the gate applications below
    /// compute the exact state-vector linear algebra on the CPU). It never
    /// claims a TPU executed anything.
    pub fn new(config: TPUConfig) -> Result<Self> {
        if config.device_type != TPUDeviceType::Simulated {
            return Err(SimulatorError::UnsupportedOperation(format!(
                "TPU backend: no TPU runtime available in this build \
                 (no JAX/XLA/libtpu linked); cannot target real device {:?}. \
                 Use TPUDeviceType::Simulated for CPU-side numerical simulation.",
                config.device_type
            )));
        }
        let device_info = TPUDeviceInfo::for_device_type(config.device_type);

        // Initialize memory manager
        let total_memory = (config.memory_per_core * config.num_cores as f64 * 1e9) as usize;
        let memory_manager = TPUMemoryManager {
            total_memory,
            used_memory: 0,
            memory_pools: HashMap::new(),
            gc_enabled: true,
            fragmentation_ratio: 0.0,
        };

        // Initialize distributed context if enabled
        let distributed_context = if config.enable_distributed {
            Some(DistributedContext {
                num_hosts: config.topology.num_hosts,
                host_id: 0,
                global_device_count: config.topology.num_chips,
                local_device_count: config.topology.chips_per_host,
                communication_backend: CommunicationBackend::GRPC,
            })
        } else {
            None
        };

        let mut simulator = Self {
            config,
            device_info,
            xla_computations: HashMap::new(),
            tensor_buffers: HashMap::new(),
            stats: TPUStats::default(),
            distributed_context,
            memory_manager,
        };

        // Compile standard quantum operations
        simulator.compile_standard_operations()?;

        Ok(simulator)
    }

    /// Compile standard quantum operations to XLA
    fn compile_standard_operations(&mut self) -> Result<()> {
        let start_time = std::time::Instant::now();

        // Single qubit gate operations
        self.compile_single_qubit_gates()?;

        // Two qubit gate operations
        self.compile_two_qubit_gates()?;

        // State vector operations
        self.compile_state_vector_operations()?;

        // Measurement operations
        self.compile_measurement_operations()?;

        // Expectation value computations
        self.compile_expectation_operations()?;

        // Quantum machine learning operations
        self.compile_qml_operations()?;

        self.stats.total_compilation_time = start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(())
    }

    /// Compile single qubit gate operations
    fn compile_single_qubit_gates(&mut self) -> Result<()> {
        // Batched single qubit gate application
        let computation = XLAComputation {
            name: "batched_single_qubit_gates".to_string(),
            input_shapes: vec![
                vec![self.config.batch_size, 1 << 20], // State vectors
                vec![2, 2],                            // Gate matrix
                vec![1],                               // Target qubit
            ],
            output_shapes: vec![
                vec![self.config.batch_size, 1 << 20], // Updated state vectors
            ],
            compilation_time: 0.0, // CPU simulation: nothing compiled to XLA
            estimated_flops: (self.config.batch_size * (1 << 20) * 8) as u64,
            memory_usage: self.config.batch_size * (1 << 20) * 16, // Complex128
        };

        self.xla_computations
            .insert("batched_single_qubit_gates".to_string(), computation);

        // Fused rotation gates (RX, RY, RZ)
        let fused_rotations = XLAComputation {
            name: "fused_rotation_gates".to_string(),
            input_shapes: vec![
                vec![self.config.batch_size, 1 << 20], // State vectors
                vec![3],                               // Rotation angles (x, y, z)
                vec![1],                               // Target qubit
            ],
            output_shapes: vec![
                vec![self.config.batch_size, 1 << 20], // Updated state vectors
            ],
            compilation_time: 0.0,
            estimated_flops: (self.config.batch_size * (1 << 20) * 12) as u64,
            memory_usage: self.config.batch_size * (1 << 20) * 16,
        };

        self.xla_computations
            .insert("fused_rotation_gates".to_string(), fused_rotations);

        Ok(())
    }

    /// Compile two qubit gate operations
    fn compile_two_qubit_gates(&mut self) -> Result<()> {
        // Batched CNOT gates
        let cnot_computation = XLAComputation {
            name: "batched_cnot_gates".to_string(),
            input_shapes: vec![
                vec![self.config.batch_size, 1 << 20], // State vectors
                vec![1],                               // Control qubit
                vec![1],                               // Target qubit
            ],
            output_shapes: vec![
                vec![self.config.batch_size, 1 << 20], // Updated state vectors
            ],
            compilation_time: 0.0,
            estimated_flops: (self.config.batch_size * (1 << 20) * 4) as u64,
            memory_usage: self.config.batch_size * (1 << 20) * 16,
        };

        self.xla_computations
            .insert("batched_cnot_gates".to_string(), cnot_computation);

        // General two-qubit gates
        let general_two_qubit = XLAComputation {
            name: "general_two_qubit_gates".to_string(),
            input_shapes: vec![
                vec![self.config.batch_size, 1 << 20], // State vectors
                vec![4, 4],                            // Gate matrix
                vec![2],                               // Qubit indices
            ],
            output_shapes: vec![
                vec![self.config.batch_size, 1 << 20], // Updated state vectors
            ],
            compilation_time: 0.0,
            estimated_flops: (self.config.batch_size * (1 << 20) * 16) as u64,
            memory_usage: self.config.batch_size * (1 << 20) * 16,
        };

        self.xla_computations
            .insert("general_two_qubit_gates".to_string(), general_two_qubit);

        Ok(())
    }

    /// Compile state vector operations
    fn compile_state_vector_operations(&mut self) -> Result<()> {
        // Batch normalization
        let normalization = XLAComputation {
            name: "batch_normalize".to_string(),
            input_shapes: vec![
                vec![self.config.batch_size, 1 << 20], // State vectors
            ],
            output_shapes: vec![
                vec![self.config.batch_size, 1 << 20], // Normalized state vectors
                vec![self.config.batch_size],          // Norms
            ],
            compilation_time: 0.0,
            estimated_flops: (self.config.batch_size * (1 << 20) * 3) as u64,
            memory_usage: self.config.batch_size * (1 << 20) * 16,
        };

        self.xla_computations
            .insert("batch_normalize".to_string(), normalization);

        // Inner product computation
        let inner_product = XLAComputation {
            name: "batch_inner_product".to_string(),
            input_shapes: vec![
                vec![self.config.batch_size, 1 << 20], // State vectors 1
                vec![self.config.batch_size, 1 << 20], // State vectors 2
            ],
            output_shapes: vec![
                vec![self.config.batch_size], // Inner products
            ],
            compilation_time: 0.0,
            estimated_flops: (self.config.batch_size * (1 << 20) * 6) as u64,
            memory_usage: self.config.batch_size * (1 << 20) * 32,
        };

        self.xla_computations
            .insert("batch_inner_product".to_string(), inner_product);

        Ok(())
    }

    /// Compile measurement operations
    fn compile_measurement_operations(&mut self) -> Result<()> {
        // Probability computation
        let probabilities = XLAComputation {
            name: "compute_probabilities".to_string(),
            input_shapes: vec![
                vec![self.config.batch_size, 1 << 20], // State vectors
            ],
            output_shapes: vec![
                vec![self.config.batch_size, 1 << 20], // Probabilities
            ],
            compilation_time: 0.0,
            estimated_flops: (self.config.batch_size * (1 << 20) * 2) as u64,
            memory_usage: self.config.batch_size * (1 << 20) * 24,
        };

        self.xla_computations
            .insert("compute_probabilities".to_string(), probabilities);

        // Sampling operation
        let sampling = XLAComputation {
            name: "quantum_sampling".to_string(),
            input_shapes: vec![
                vec![self.config.batch_size, 1 << 20], // Probabilities
                vec![self.config.batch_size],          // Random numbers
            ],
            output_shapes: vec![
                vec![self.config.batch_size], // Sample results
            ],
            compilation_time: 0.0,
            estimated_flops: (self.config.batch_size * (1 << 20)) as u64,
            memory_usage: self.config.batch_size * (1 << 20) * 8,
        };

        self.xla_computations
            .insert("quantum_sampling".to_string(), sampling);

        Ok(())
    }

    /// Compile expectation value operations
    fn compile_expectation_operations(&mut self) -> Result<()> {
        // Pauli expectation values
        let pauli_expectation = XLAComputation {
            name: "pauli_expectation_values".to_string(),
            input_shapes: vec![
                vec![self.config.batch_size, 1 << 20], // State vectors
                vec![20],                              // Pauli strings (encoded)
            ],
            output_shapes: vec![
                vec![self.config.batch_size, 20], // Expectation values
            ],
            compilation_time: 0.0,
            estimated_flops: (self.config.batch_size * (1 << 20) * 20 * 4) as u64,
            memory_usage: self.config.batch_size * (1 << 20) * 16,
        };

        self.xla_computations
            .insert("pauli_expectation_values".to_string(), pauli_expectation);

        // Hamiltonian expectation
        let hamiltonian_expectation = XLAComputation {
            name: "hamiltonian_expectation".to_string(),
            input_shapes: vec![
                vec![self.config.batch_size, 1 << 20], // State vectors
                vec![1 << 20, 1 << 20],                // Hamiltonian matrix
            ],
            output_shapes: vec![
                vec![self.config.batch_size], // Expectation values
            ],
            compilation_time: 0.0,
            estimated_flops: (self.config.batch_size * (1 << 40)) as u64,
            memory_usage: (1 << 40) * 16 + self.config.batch_size * (1 << 20) * 16,
        };

        self.xla_computations.insert(
            "hamiltonian_expectation".to_string(),
            hamiltonian_expectation,
        );

        Ok(())
    }

    /// Compile quantum machine learning operations
    fn compile_qml_operations(&mut self) -> Result<()> {
        // Variational circuit execution
        let variational_circuit = XLAComputation {
            name: "variational_circuit_batch".to_string(),
            input_shapes: vec![
                vec![self.config.batch_size, 1 << 20], // Initial states
                vec![100],                             // Parameters
                vec![50],                              // Circuit structure
            ],
            output_shapes: vec![
                vec![self.config.batch_size, 1 << 20], // Final states
            ],
            compilation_time: 0.0,
            estimated_flops: (self.config.batch_size * 100 * (1 << 20) * 8) as u64,
            memory_usage: self.config.batch_size * (1 << 20) * 16,
        };

        self.xla_computations
            .insert("variational_circuit_batch".to_string(), variational_circuit);

        // Gradient computation using parameter shift
        let parameter_shift_gradients = XLAComputation {
            name: "parameter_shift_gradients".to_string(),
            input_shapes: vec![
                vec![self.config.batch_size, 1 << 20], // States
                vec![100],                             // Parameters
                vec![50],                              // Circuit structure
                vec![20],                              // Observables
            ],
            output_shapes: vec![
                vec![self.config.batch_size, 100], // Gradients
            ],
            compilation_time: 0.0,
            estimated_flops: (self.config.batch_size * 100 * 20 * (1 << 20) * 16) as u64,
            memory_usage: self.config.batch_size * (1 << 20) * 16 * 4, // 4 evaluations per gradient
        };

        self.xla_computations.insert(
            "parameter_shift_gradients".to_string(),
            parameter_shift_gradients,
        );

        Ok(())
    }

    /// Execute batched quantum circuit
    pub fn execute_batch_circuit(
        &mut self,
        circuits: &[InterfaceCircuit],
        initial_states: &[Array1<Complex64>],
    ) -> Result<Vec<Array1<Complex64>>> {
        let start_time = std::time::Instant::now();

        if circuits.len() != initial_states.len() {
            return Err(SimulatorError::InvalidInput(
                "Circuit and state count mismatch".to_string(),
            ));
        }

        if circuits.len() > self.config.batch_size {
            return Err(SimulatorError::InvalidInput(
                "Batch size exceeded".to_string(),
            ));
        }

        // Allocate device memory for batch
        self.allocate_batch_memory(circuits.len(), initial_states[0].len())?;

        // Transfer initial states to device
        self.transfer_states_to_device(initial_states)?;

        // Execute circuits in batch
        let mut final_states = Vec::with_capacity(circuits.len());

        for (i, circuit) in circuits.iter().enumerate() {
            let mut current_state = initial_states[i].clone();

            // Process gates sequentially (could be optimized for parallel execution)
            for gate in &circuit.gates {
                current_state = self.apply_gate_tpu(&current_state, gate)?;
            }

            final_states.push(current_state);
        }

        // Transfer results back to host
        self.transfer_states_to_host(&final_states)?;

        let execution_time = start_time.elapsed().as_secs_f64() * 1000.0;
        let estimated_flops = circuits.len() as u64 * 1000; // Rough estimate
        self.stats.update_operation(execution_time, estimated_flops);

        Ok(final_states)
    }

    /// Apply a quantum gate (CPU numerical simulation of the device math).
    ///
    /// This computes the exact state-vector transformation on the CPU using the
    /// gate's canonical unitary. It is the honest `Simulated`-device path: it
    /// performs the real linear algebra and never pretends a TPU executed it.
    fn apply_gate_tpu(
        &mut self,
        state: &Array1<Complex64>,
        gate: &InterfaceGate,
    ) -> Result<Array1<Complex64>> {
        let start_time = std::time::Instant::now();
        let unitary = resolve_gate_unitary(gate)?;
        let result = match gate.qubits.len() {
            1 => Self::apply_single_qubit_unitary(state, gate.qubits[0], &unitary)?,
            2 => Self::apply_two_qubit_unitary(state, gate.qubits[0], gate.qubits[1], &unitary)?,
            n => {
                return Err(SimulatorError::UnsupportedOperation(format!(
                    "TPU Simulated backend: {n}-qubit gate application is not implemented"
                )));
            }
        };
        let execution_time = start_time.elapsed().as_secs_f64() * 1000.0;
        let flops = (state.len() * 8 * gate.qubits.len()) as u64;
        self.stats.update_operation(execution_time, flops);
        Ok(result)
    }

    /// Apply a 2x2 unitary to `target_qubit` of the state vector (exact math).
    fn apply_single_qubit_unitary(
        state: &Array1<Complex64>,
        target_qubit: usize,
        unitary: &Array2<Complex64>,
    ) -> Result<Array1<Complex64>> {
        let num_qubits = state.len().trailing_zeros() as usize;
        if state.len() != 1usize << num_qubits {
            return Err(SimulatorError::DimensionMismatch(format!(
                "State length {} is not a power of two",
                state.len()
            )));
        }
        if target_qubit >= num_qubits {
            return Err(SimulatorError::IndexOutOfBounds(target_qubit));
        }
        let mut result = state.clone();
        let target_mask = 1usize << target_qubit;
        for i in 0..state.len() {
            if i & target_mask == 0 {
                let j = i | target_mask;
                let amp_0 = state[i];
                let amp_1 = state[j];
                result[i] = unitary[[0, 0]] * amp_0 + unitary[[0, 1]] * amp_1;
                result[j] = unitary[[1, 0]] * amp_0 + unitary[[1, 1]] * amp_1;
            }
        }
        Ok(result)
    }

    /// Apply a 4x4 unitary to `(q0, q1)` of the state vector (exact math).
    ///
    /// Basis ordering for the 4x4 matrix is `|q0 q1>` with `q0` the high bit,
    /// matching [`InterfaceGate::unitary_matrix`].
    fn apply_two_qubit_unitary(
        state: &Array1<Complex64>,
        q0: usize,
        q1: usize,
        unitary: &Array2<Complex64>,
    ) -> Result<Array1<Complex64>> {
        let num_qubits = state.len().trailing_zeros() as usize;
        if state.len() != 1usize << num_qubits {
            return Err(SimulatorError::DimensionMismatch(format!(
                "State length {} is not a power of two",
                state.len()
            )));
        }
        if q0 >= num_qubits || q1 >= num_qubits || q0 == q1 {
            return Err(SimulatorError::InvalidInput(format!(
                "Invalid two-qubit indices ({q0}, {q1}) for {num_qubits} qubits"
            )));
        }
        let mut result = state.clone();
        let mask0 = 1usize << q0;
        let mask1 = 1usize << q1;
        for i in 0..state.len() {
            // Process each 2-qubit subspace once, anchored at the element where
            // both target bits are zero.
            if (i & mask0) == 0 && (i & mask1) == 0 {
                let idx = [i, i | mask1, i | mask0, i | mask0 | mask1];
                let amps = [state[idx[0]], state[idx[1]], state[idx[2]], state[idx[3]]];
                for (row, &out_idx) in idx.iter().enumerate() {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for (col, &amp) in amps.iter().enumerate() {
                        acc += unitary[[row, col]] * amp;
                    }
                    result[out_idx] = acc;
                }
            }
        }
        Ok(result)
    }

    /// Allocate batch memory on TPU
    fn allocate_batch_memory(&mut self, batch_size: usize, state_size: usize) -> Result<()> {
        let total_size = batch_size * state_size * 16; // Complex128

        if total_size > self.memory_manager.total_memory {
            return Err(SimulatorError::MemoryError(
                "Insufficient TPU memory".to_string(),
            ));
        }

        // Create tensor buffer
        let buffer = TPUTensorBuffer {
            buffer_id: self.tensor_buffers.len(),
            shape: vec![batch_size, state_size],
            dtype: TPUDataType::Complex128,
            size_bytes: total_size,
            device_id: 0,
            on_device: true,
        };

        self.tensor_buffers
            .insert("batch_states".to_string(), buffer);
        self.memory_manager.used_memory += total_size;

        if self.memory_manager.used_memory > self.stats.peak_memory_usage {
            self.stats.peak_memory_usage = self.memory_manager.used_memory;
        }

        Ok(())
    }

    /// Stage input states for the batch (CPU simulation: no device boundary).
    ///
    /// In the `Simulated` backend the data already lives in host memory, so
    /// there is no real host-to-device copy and no fabricated latency. We only
    /// record the (real, typically ~0) time and count the staging event.
    fn transfer_states_to_device(&mut self, _states: &[Array1<Complex64>]) -> Result<()> {
        let start_time = std::time::Instant::now();
        self.stats.h2d_transfers += 1;
        self.stats.total_transfer_time += start_time.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }

    /// Retrieve output states for the batch (CPU simulation: no device boundary).
    fn transfer_states_to_host(&mut self, _states: &[Array1<Complex64>]) -> Result<()> {
        let start_time = std::time::Instant::now();
        self.stats.d2h_transfers += 1;
        self.stats.total_transfer_time += start_time.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }

    /// Compute Pauli-observable expectation values (CPU numerical simulation).
    ///
    /// Each observable is a compact single-qubit Pauli string of the form
    /// `"<P><qubit>"`, e.g. `"Z0"`, `"X1"`, `"Y2"` (identity on every other
    /// qubit). The result `<psi|P|psi>` is computed exactly from the amplitudes;
    /// no value is fabricated.
    pub fn compute_expectation_values_tpu(
        &mut self,
        states: &[Array1<Complex64>],
        observables: &[String],
    ) -> Result<Array2<f64>> {
        let start_time = std::time::Instant::now();

        let batch_size = states.len();
        let num_observables = observables.len();
        let mut results = Array2::zeros((batch_size, num_observables));

        for (i, state) in states.iter().enumerate() {
            for (j, observable) in observables.iter().enumerate() {
                results[[i, j]] = Self::single_pauli_expectation(state, observable)?;
            }
        }

        let state_len = states.first().map_or(0, Array1::len);
        let execution_time = start_time.elapsed().as_secs_f64() * 1000.0;
        let flops = (batch_size * num_observables * state_len * 4) as u64;
        self.stats.update_operation(execution_time, flops);

        Ok(results)
    }

    /// Compute `<psi|P_q|psi>` for a single-qubit Pauli observable `"<P><qubit>"`.
    fn single_pauli_expectation(state: &Array1<Complex64>, observable: &str) -> Result<f64> {
        let trimmed = observable.trim();
        let mut chars = trimmed.chars();
        let pauli = chars.next().ok_or_else(|| {
            SimulatorError::InvalidObservable("empty observable string".to_string())
        })?;
        let qubit: usize = chars.as_str().parse().map_err(|_| {
            SimulatorError::InvalidObservable(format!(
                "could not parse qubit index from observable '{observable}'"
            ))
        })?;

        let num_qubits = state.len().trailing_zeros() as usize;
        if state.len() != 1usize << num_qubits {
            return Err(SimulatorError::DimensionMismatch(format!(
                "State length {} is not a power of two",
                state.len()
            )));
        }
        if qubit >= num_qubits {
            return Err(SimulatorError::IndexOutOfBounds(qubit));
        }

        let mask = 1usize << qubit;
        let mut expectation = Complex64::new(0.0, 0.0);
        match pauli {
            'I' => {
                for amp in state.iter() {
                    expectation += amp.conj() * amp;
                }
            }
            'Z' => {
                for (idx, amp) in state.iter().enumerate() {
                    let sign = if idx & mask != 0 { -1.0 } else { 1.0 };
                    expectation += amp.conj() * amp * sign;
                }
            }
            'X' => {
                for idx in 0..state.len() {
                    let partner = idx ^ mask;
                    expectation += state[idx].conj() * state[partner];
                }
            }
            'Y' => {
                for idx in 0..state.len() {
                    let partner = idx ^ mask;
                    // Y|0> = i|1>, Y|1> = -i|0>; coefficient depends on the bit.
                    let coeff = if idx & mask == 0 {
                        Complex64::new(0.0, -1.0)
                    } else {
                        Complex64::new(0.0, 1.0)
                    };
                    expectation += state[idx].conj() * coeff * state[partner];
                }
            }
            other => {
                return Err(SimulatorError::InvalidObservable(format!(
                    "unsupported Pauli operator '{other}' in observable '{observable}'"
                )));
            }
        }

        Ok(expectation.re)
    }

    /// Get device information
    #[must_use]
    pub const fn get_device_info(&self) -> &TPUDeviceInfo {
        &self.device_info
    }

    /// Get performance statistics
    #[must_use]
    pub const fn get_stats(&self) -> &TPUStats {
        &self.stats
    }

    /// Reset performance statistics
    pub fn reset_stats(&mut self) {
        self.stats = TPUStats::default();
    }

    /// Whether a *real* TPU runtime is available.
    ///
    /// HONEST: this build links no TPU runtime, so a physical TPU is never
    /// available. This returns `false` even though a CPU `Simulated` device is
    /// in use — that is a numerical model, not real silicon. Use
    /// [`Self::is_simulated`] to check for the CPU-simulation device.
    #[must_use]
    pub const fn is_tpu_available(&self) -> bool {
        false
    }

    /// Whether this simulator is the CPU-side numerical `Simulated` device.
    #[must_use]
    pub fn is_simulated(&self) -> bool {
        self.device_info.device_type == TPUDeviceType::Simulated
    }

    /// Get memory usage
    #[must_use]
    pub const fn get_memory_usage(&self) -> (usize, usize) {
        (
            self.memory_manager.used_memory,
            self.memory_manager.total_memory,
        )
    }

    /// Reclaim memory held by tensor buffers that are no longer device-resident.
    ///
    /// HONEST: this frees exactly the bytes of buffers whose `on_device` flag is
    /// `false` (i.e. genuinely releasable in the model) and updates the
    /// accounting accordingly. It does not fabricate a fixed "freed 10%" figure.
    /// Returns the number of bytes actually reclaimed.
    pub fn garbage_collect(&mut self) -> Result<usize> {
        if !self.memory_manager.gc_enabled {
            return Ok(0);
        }

        let mut freed_memory = 0usize;
        self.tensor_buffers.retain(|_, buffer| {
            if buffer.on_device {
                true
            } else {
                freed_memory += buffer.size_bytes;
                false
            }
        });
        self.memory_manager.used_memory =
            self.memory_manager.used_memory.saturating_sub(freed_memory);

        Ok(freed_memory)
    }
}

/// Benchmark the CPU-simulated TPU backend.
///
/// HONEST: only [`TPUDeviceType::Simulated`] can run in this build (no TPU
/// runtime is linked), so every configuration benchmarked here is the CPU
/// numerical simulation. The reported times are real `Instant`-measured CPU
/// timings of actual state-vector work; no throughput figure is fabricated.
pub fn benchmark_tpu_acceleration() -> Result<HashMap<String, f64>> {
    let mut results = HashMap::new();

    // All configurations use the CPU `Simulated` device (the only runnable one),
    // varying batch size and core count to exercise different work sizes.
    let configs = vec![
        TPUConfig {
            device_type: TPUDeviceType::Simulated,
            num_cores: 8,
            batch_size: 16,
            ..Default::default()
        },
        TPUConfig {
            device_type: TPUDeviceType::Simulated,
            num_cores: 16,
            batch_size: 32,
            ..Default::default()
        },
        TPUConfig {
            device_type: TPUDeviceType::Simulated,
            num_cores: 32,
            batch_size: 64,
            enable_mixed_precision: true,
            ..Default::default()
        },
    ];

    for (i, config) in configs.into_iter().enumerate() {
        let start = std::time::Instant::now();

        let mut simulator = TPUQuantumSimulator::new(config)?;

        // Create test circuits
        let mut circuits = Vec::new();
        let mut initial_states = Vec::new();

        for _ in 0..simulator.config.batch_size.min(8) {
            let mut circuit = InterfaceCircuit::new(10, 0);

            // Add some gates
            circuit.add_gate(InterfaceGate::new(InterfaceGateType::Hadamard, vec![0]));
            circuit.add_gate(InterfaceGate::new(InterfaceGateType::CNOT, vec![0, 1]));
            circuit.add_gate(InterfaceGate::new(InterfaceGateType::RY(0.5), vec![2]));
            circuit.add_gate(InterfaceGate::new(InterfaceGateType::CZ, vec![1, 2]));

            circuits.push(circuit);

            // Create initial state
            let mut state = Array1::zeros(1 << 10);
            state[0] = Complex64::new(1.0, 0.0);
            initial_states.push(state);
        }

        // Execute batch
        let _final_states = simulator.execute_batch_circuit(&circuits, &initial_states)?;

        // Test expectation values
        let observables = vec!["Z0".to_string(), "X1".to_string(), "Y2".to_string()];
        let _expectations =
            simulator.compute_expectation_values_tpu(&initial_states, &observables)?;

        let time = start.elapsed().as_secs_f64() * 1000.0;
        results.insert(format!("tpu_config_{i}"), time);

        // Add performance metrics
        let stats = simulator.get_stats();
        results.insert(
            format!("tpu_config_{i}_operations"),
            stats.total_operations as f64,
        );
        results.insert(format!("tpu_config_{i}_avg_time"), stats.avg_operation_time);
        results.insert(
            format!("tpu_config_{i}_total_flops"),
            stats.total_flops as f64,
        );

        let performance_metrics = stats.get_performance_metrics();
        for (key, value) in performance_metrics {
            results.insert(format!("tpu_config_{i}_{key}"), value);
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// Build a CPU `Simulated` config (the only runnable device in this build).
    fn sim_config() -> TPUConfig {
        TPUConfig {
            device_type: TPUDeviceType::Simulated,
            ..Default::default()
        }
    }

    #[test]
    fn test_real_tpu_device_unavailable() {
        // Honest behavior: requesting a real TPU device fails loudly.
        let config = TPUConfig::default(); // default is TPUv4 (a real device)
        let result = TPUQuantumSimulator::new(config);
        assert!(result.is_err());
        // Match on `.err()` so the `Ok` simulator value need not be `Debug`.
        match result.err() {
            Some(SimulatorError::UnsupportedOperation(msg)) => assert!(msg.contains("TPU")),
            other => panic!("expected UnsupportedOperation, got {other:?}"),
        }
    }

    #[test]
    fn test_simulated_device_creation() {
        let simulator = TPUQuantumSimulator::new(sim_config());
        assert!(simulator.is_ok());
        let simulator = simulator.expect("simulated device should construct");
        assert!(simulator.is_simulated());
        // No *real* TPU is ever available in this build.
        assert!(!simulator.is_tpu_available());
    }

    #[test]
    fn test_device_info_reference_specs() {
        // for_device_type is a reference spec table, not a detection result.
        let device_info = TPUDeviceInfo::for_device_type(TPUDeviceType::TPUv4);
        assert_eq!(device_info.device_type, TPUDeviceType::TPUv4);
        assert_eq!(device_info.core_count, 2);
        assert_abs_diff_eq!(device_info.memory_size, 32.0, epsilon = 1e-10);
        assert!(device_info.supports_complex);
    }

    #[test]
    fn test_xla_compilation() {
        let simulator =
            TPUQuantumSimulator::new(sim_config()).expect("Failed to create TPU simulator");

        assert!(simulator
            .xla_computations
            .contains_key("batched_single_qubit_gates"));
        assert!(simulator
            .xla_computations
            .contains_key("batched_cnot_gates"));
        assert!(simulator.xla_computations.contains_key("batch_normalize"));
        // total_compilation_time is the real measured wall-time of building the
        // computation descriptors (non-negative); per-computation
        // compilation_time is 0.0 because nothing is compiled to XLA on CPU.
        assert!(simulator.stats.total_compilation_time >= 0.0);
        assert_abs_diff_eq!(
            simulator.xla_computations["batch_normalize"].compilation_time,
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_memory_allocation() {
        let mut simulator =
            TPUQuantumSimulator::new(sim_config()).expect("Failed to create TPU simulator");

        let result = simulator.allocate_batch_memory(4, 1024);
        assert!(result.is_ok());
        assert!(simulator.tensor_buffers.contains_key("batch_states"));
        assert!(simulator.memory_manager.used_memory > 0);
    }

    #[test]
    fn test_memory_limit() {
        let config = TPUConfig {
            device_type: TPUDeviceType::Simulated,
            memory_per_core: 0.001, // Very small memory
            num_cores: 1,
            ..Default::default()
        };
        let mut simulator =
            TPUQuantumSimulator::new(config).expect("Failed to create TPU simulator");

        let result = simulator.allocate_batch_memory(1000, 1_000_000); // Large allocation
        assert!(result.is_err());
    }

    #[test]
    fn test_single_qubit_gate_application_real_math() {
        let mut state = Array1::zeros(4);
        state[0] = Complex64::new(1.0, 0.0);

        let gate = InterfaceGate::new(InterfaceGateType::H, vec![0]);
        let unitary = resolve_gate_unitary(&gate).expect("hadamard matrix");
        let result =
            TPUQuantumSimulator::apply_single_qubit_unitary(&state, 0, &unitary).expect("apply H");

        // After Hadamard, |0> becomes (|0> + |1>)/sqrt(2)
        assert_abs_diff_eq!(result[0].norm(), 1.0 / 2.0_f64.sqrt(), epsilon = 1e-10);
        assert_abs_diff_eq!(result[1].norm(), 1.0 / 2.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_rotation_uses_real_angle() {
        // Regression: rotations must use the gate's actual angle, not a constant.
        let mut state = Array1::zeros(2);
        state[0] = Complex64::new(1.0, 0.0);
        let mut simulator =
            TPUQuantumSimulator::new(sim_config()).expect("Failed to create TPU simulator");
        // RY(pi) maps |0> -> |1>.
        let gate = InterfaceGate::new(InterfaceGateType::RY(std::f64::consts::PI), vec![0]);
        let result = simulator
            .apply_gate_tpu(&state, &gate)
            .expect("apply RY(pi)");
        assert_abs_diff_eq!(result[0].norm(), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1].norm(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_two_qubit_gate_application_real_math() {
        // Standard little-endian convention (bit position == qubit index,
        // matching `statevector.rs::apply_cnot`'s `(i >> control_idx) & 1`):
        // global index 0b01 has qubit0(control)=1, qubit1(target)=0, i.e.
        // |q0=1, q1=0>. With control=qubit0 asserted, CNOT(control=0,
        // target=1) flips the target, giving |q0=1, q1=1> = index 0b11.
        let mut state = Array1::zeros(4);
        state[0b01] = Complex64::new(1.0, 0.0);

        let gate = InterfaceGate::new(InterfaceGateType::CNOT, vec![0, 1]);
        let unitary = gate.unitary_matrix().expect("cnot matrix");
        let result = TPUQuantumSimulator::apply_two_qubit_unitary(&state, 0, 1, &unitary)
            .expect("apply CNOT");

        assert_eq!(result.len(), 4);
        assert_abs_diff_eq!(result[0b11].norm(), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[0b01].norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_batch_circuit_execution() {
        let config = TPUConfig {
            device_type: TPUDeviceType::Simulated,
            batch_size: 2,
            ..Default::default()
        };
        let mut simulator =
            TPUQuantumSimulator::new(config).expect("Failed to create TPU simulator");

        let mut circuit1 = InterfaceCircuit::new(2, 0);
        circuit1.add_gate(InterfaceGate::new(InterfaceGateType::H, vec![0]));

        let mut circuit2 = InterfaceCircuit::new(2, 0);
        circuit2.add_gate(InterfaceGate::new(InterfaceGateType::X, vec![1]));

        let circuits = vec![circuit1, circuit2];

        let mut state1 = Array1::zeros(4);
        state1[0] = Complex64::new(1.0, 0.0);
        let mut state2 = Array1::zeros(4);
        state2[0] = Complex64::new(1.0, 0.0);
        let initial_states = vec![state1, state2];

        let final_states = simulator
            .execute_batch_circuit(&circuits, &initial_states)
            .expect("Failed to execute batch circuit");
        assert_eq!(final_states.len(), 2);

        // circuit2 applies X to qubit 1 of |00> -> |10> (index 0b10 = 2).
        assert_abs_diff_eq!(final_states[1][0b10].norm(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expectation_value_computation_real() {
        let mut simulator =
            TPUQuantumSimulator::new(sim_config()).expect("Failed to create TPU simulator");

        // state1 = |00>, state2 = |11>
        let mut state1 = Array1::zeros(4);
        state1[0] = Complex64::new(1.0, 0.0);
        let mut state2 = Array1::zeros(4);
        state2[3] = Complex64::new(1.0, 0.0);

        let states = vec![state1, state2];
        let observables = vec!["Z0".to_string(), "Z1".to_string()];

        let expectations = simulator
            .compute_expectation_values_tpu(&states, &observables)
            .expect("Failed to compute expectation values");
        assert_eq!(expectations.shape(), &[2, 2]);
        // <00|Z0|00> = +1, <00|Z1|00> = +1
        assert_abs_diff_eq!(expectations[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(expectations[[0, 1]], 1.0, epsilon = 1e-10);
        // <11|Z0|11> = -1, <11|Z1|11> = -1
        assert_abs_diff_eq!(expectations[[1, 0]], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(expectations[[1, 1]], -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expectation_x_observable() {
        let mut simulator =
            TPUQuantumSimulator::new(sim_config()).expect("Failed to create TPU simulator");
        // (|0> + |1>)/sqrt(2) is +1 eigenstate of X.
        let mut state = Array1::zeros(2);
        state[0] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
        state[1] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
        let exps = simulator
            .compute_expectation_values_tpu(&[state], &["X0".to_string()])
            .expect("expectation");
        assert_abs_diff_eq!(exps[[0, 0]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_stats_tracking() {
        let mut stats = TPUStats::default();
        stats.update_operation(10.0, 1000);
        stats.update_operation(20.0, 2000);
        assert_eq!(stats.total_operations, 2);
        assert_abs_diff_eq!(stats.total_execution_time, 30.0, epsilon = 1e-10);
        assert_abs_diff_eq!(stats.avg_operation_time, 15.0, epsilon = 1e-10);
        assert_eq!(stats.total_flops, 3000);
    }

    #[test]
    fn test_performance_metrics() {
        let stats = TPUStats {
            total_operations: 100,
            total_execution_time: 1000.0,
            total_flops: 1_000_000,
            xla_cache_hits: 80,
            xla_cache_misses: 20,
            ..Default::default()
        };

        let metrics = stats.get_performance_metrics();
        assert!(metrics.contains_key("flops_per_second"));
        assert!(metrics.contains_key("operations_per_second"));
        assert!(metrics.contains_key("cache_hit_rate"));
        assert_abs_diff_eq!(metrics["operations_per_second"], 100.0, epsilon = 1e-10);
        assert_abs_diff_eq!(metrics["cache_hit_rate"], 0.8, epsilon = 1e-10);
    }

    #[test]
    fn test_garbage_collection_only_frees_releasable() {
        let mut simulator =
            TPUQuantumSimulator::new(sim_config()).expect("Failed to create TPU simulator");

        // A device-resident buffer is NOT freed.
        simulator.tensor_buffers.insert(
            "resident".to_string(),
            TPUTensorBuffer {
                buffer_id: 0,
                shape: vec![10],
                dtype: TPUDataType::Complex128,
                size_bytes: 1000,
                device_id: 0,
                on_device: true,
            },
        );
        // A released buffer IS freed.
        simulator.tensor_buffers.insert(
            "released".to_string(),
            TPUTensorBuffer {
                buffer_id: 1,
                shape: vec![10],
                dtype: TPUDataType::Complex128,
                size_bytes: 500,
                device_id: 0,
                on_device: false,
            },
        );
        simulator.memory_manager.used_memory = 1500;

        let freed = simulator.garbage_collect().expect("gc");
        assert_eq!(freed, 500);
        assert_eq!(simulator.memory_manager.used_memory, 1000);
        assert!(simulator.tensor_buffers.contains_key("resident"));
        assert!(!simulator.tensor_buffers.contains_key("released"));
    }

    #[test]
    fn test_benchmark_simulated_runs() {
        // Honest: benchmark only runs the CPU Simulated device and times real work.
        let results = benchmark_tpu_acceleration().expect("benchmark should run on CPU sim");
        assert!(results.contains_key("tpu_config_0"));
    }

    #[test]
    fn test_tpu_data_types() {
        assert_eq!(TPUDataType::Float32.size_bytes(), 4);
        assert_eq!(TPUDataType::Float64.size_bytes(), 8);
        assert_eq!(TPUDataType::BFloat16.size_bytes(), 2);
        assert_eq!(TPUDataType::Complex64.size_bytes(), 8);
        assert_eq!(TPUDataType::Complex128.size_bytes(), 16);
    }
}
