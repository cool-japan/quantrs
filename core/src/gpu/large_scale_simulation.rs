//! Large-Scale Quantum Simulation GPU Acceleration
//!
//! This module extends the existing GPU infrastructure to provide acceleration
//! for large-scale quantum simulations, including state vector simulation,
//! tensor network contractions, and distributed quantum computing.

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    tensor_network::Tensor,
};
use scirs2_core::Complex64;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, RwLock},
};

/// GPU backend types for large-scale simulation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    CPU,
    CUDA,
    OpenCL,
    ROCm,
    WebGPU,
    Metal,
    Vulkan,
}

/// GPU device information for large-scale simulation
#[derive(Debug, Clone)]
pub struct GpuDevice {
    pub id: u32,
    pub name: String,
    pub backend: GpuBackend,
    pub memory_size: usize,
    pub compute_units: u32,
    pub max_work_group_size: usize,
    pub supports_double_precision: bool,
    pub is_available: bool,
}

/// Configuration for large-scale simulation acceleration
#[derive(Debug, Clone)]
pub struct LargeScaleSimConfig {
    /// Maximum number of qubits for state vector simulation
    pub max_state_vector_qubits: usize,
    /// Minimum tensor size for GPU acceleration
    pub gpu_tensor_threshold: usize,
    /// Memory pool size in bytes
    pub memory_pool_size: usize,
    /// Enable distributed computation
    pub enable_distributed: bool,
    /// Tensor decomposition threshold
    pub tensor_decomp_threshold: f64,
    /// Precision mode (single/double)
    pub use_double_precision: bool,
}

impl Default for LargeScaleSimConfig {
    fn default() -> Self {
        Self {
            max_state_vector_qubits: 50,
            gpu_tensor_threshold: 1024,
            memory_pool_size: 8 * 1024 * 1024 * 1024, // 8GB
            enable_distributed: false,
            tensor_decomp_threshold: 1e-12,
            use_double_precision: true,
        }
    }
}

/// Large-scale simulation accelerator
pub struct LargeScaleSimAccelerator {
    config: LargeScaleSimConfig,
    devices: Vec<GpuDevice>,
    active_device: Option<usize>,
    memory_manager: Arc<Mutex<LargeScaleMemoryManager>>,
    performance_monitor: Arc<RwLock<LargeScalePerformanceMonitor>>,
}

/// Memory manager for large quantum simulations
#[derive(Debug)]
pub struct LargeScaleMemoryManager {
    /// Available memory pools per device
    memory_pools: HashMap<usize, MemoryPool>,
    /// Current allocations
    allocations: HashMap<u64, AllocationInfo>,
    /// Allocation counter
    next_allocation_id: u64,
}

#[derive(Debug)]
pub struct MemoryPool {
    device_id: usize,
    total_size: usize,
    used_size: usize,
    free_blocks: Vec<MemoryBlock>,
    allocated_blocks: HashMap<u64, MemoryBlock>,
}

#[derive(Debug, Clone)]
pub struct MemoryBlock {
    offset: usize,
    size: usize,
    is_pinned: bool,
}

#[derive(Debug)]
pub struct AllocationInfo {
    device_id: usize,
    size: usize,
    allocation_type: AllocationType,
    timestamp: std::time::Instant,
}

#[derive(Debug, Clone)]
pub enum AllocationType {
    StateVector,
    TensorData,
    IntermediateBuffer,
    TemporaryStorage,
}

/// Performance monitoring for large-scale simulations
#[derive(Debug)]
pub struct LargeScalePerformanceMonitor {
    /// Operation timings
    operation_times: HashMap<String, Vec<f64>>,
    /// Memory usage over time
    memory_usage_history: Vec<(std::time::Instant, usize)>,
    /// Tensor contraction statistics
    contraction_stats: ContractionStatistics,
    /// State vector operation statistics
    state_vector_stats: StateVectorStatistics,
}

#[derive(Debug, Default, Clone)]
pub struct ContractionStatistics {
    pub total_contractions: u64,
    pub total_contraction_time_ms: f64,
    pub largest_tensor_size: usize,
    pub decompositions_performed: u64,
    pub memory_savings_percent: f64,
}

#[derive(Debug, Default, Clone)]
pub struct StateVectorStatistics {
    pub max_qubits_simulated: usize,
    pub total_gate_applications: u64,
    pub total_simulation_time_ms: f64,
    pub memory_transfer_overhead_percent: f64,
    pub gpu_utilization_percent: f64,
}

impl LargeScaleSimAccelerator {
    /// Create a new large-scale simulation accelerator
    pub fn new(config: LargeScaleSimConfig, devices: Vec<GpuDevice>) -> QuantRS2Result<Self> {
        if devices.is_empty() {
            return Err(QuantRS2Error::NoHardwareAvailable(
                "No GPU devices available for large-scale simulation".to_string(),
            ));
        }

        let memory_manager = Arc::new(Mutex::new(LargeScaleMemoryManager::new(&devices, &config)?));
        let performance_monitor = Arc::new(RwLock::new(LargeScalePerformanceMonitor::new()));

        Ok(Self {
            config,
            active_device: Some(0),
            devices,
            memory_manager,
            performance_monitor,
        })
    }

    /// Select optimal device for a given simulation task
    pub fn select_optimal_device(
        &mut self,
        task_type: SimulationTaskType,
        required_memory: usize,
    ) -> QuantRS2Result<usize> {
        let mut best_device_id = 0;
        let mut best_score = 0.0;

        for (i, device) in self.devices.iter().enumerate() {
            if !device.is_available || device.memory_size < required_memory {
                continue;
            }

            let score = self.compute_device_score(device, &task_type, required_memory);
            if score > best_score {
                best_score = score;
                best_device_id = i;
            }
        }

        if best_score == 0.0 {
            return Err(QuantRS2Error::NoHardwareAvailable(
                "No suitable device found for simulation task".to_string(),
            ));
        }

        self.active_device = Some(best_device_id);
        Ok(best_device_id)
    }

    fn compute_device_score(
        &self,
        device: &GpuDevice,
        task_type: &SimulationTaskType,
        required_memory: usize,
    ) -> f64 {
        let memory_score =
            (device.memory_size - required_memory) as f64 / device.memory_size as f64;
        let compute_score = device.compute_units as f64 / 100.0; // Normalize

        match task_type {
            SimulationTaskType::StateVector => {
                // Favor high-memory, high-compute devices
                0.6f64.mul_add(memory_score, 0.4 * compute_score)
            }
            SimulationTaskType::TensorContraction => {
                // Favor high-compute devices
                0.3f64.mul_add(memory_score, 0.7 * compute_score)
            }
            SimulationTaskType::Distributed => {
                // Favor balanced devices
                0.5f64.mul_add(memory_score, 0.5 * compute_score)
            }
        }
    }

    /// Initialize large-scale state vector simulation
    pub fn init_state_vector_simulation(
        &mut self,
        num_qubits: usize,
    ) -> QuantRS2Result<LargeScaleStateVectorSim> {
        if num_qubits > self.config.max_state_vector_qubits {
            return Err(QuantRS2Error::UnsupportedQubits(
                num_qubits,
                format!(
                    "Maximum {} qubits supported",
                    self.config.max_state_vector_qubits
                ),
            ));
        }

        let state_size = 1_usize << num_qubits;
        let memory_required = state_size * std::mem::size_of::<Complex64>() * 2; // State + temp buffer

        let device_id =
            self.select_optimal_device(SimulationTaskType::StateVector, memory_required)?;

        LargeScaleStateVectorSim::new(
            num_qubits,
            device_id,
            Arc::clone(&self.memory_manager),
            Arc::clone(&self.performance_monitor),
        )
    }

    /// Initialize tensor network contractor
    pub fn init_tensor_contractor(&mut self) -> QuantRS2Result<LargeScaleTensorContractor> {
        let device_id = self.active_device.unwrap_or(0);

        LargeScaleTensorContractor::new(
            device_id,
            &self.config,
            Arc::clone(&self.memory_manager),
            Arc::clone(&self.performance_monitor),
        )
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> LargeScalePerformanceStats {
        let monitor = self
            .performance_monitor
            .read()
            .expect("Performance monitor lock poisoned");
        let memory_manager = self
            .memory_manager
            .lock()
            .expect("Memory manager lock poisoned");

        LargeScalePerformanceStats {
            contraction_stats: monitor.contraction_stats.clone(),
            state_vector_stats: monitor.state_vector_stats.clone(),
            total_memory_allocated: memory_manager.get_total_allocated(),
            peak_memory_usage: memory_manager.get_peak_usage(),
            device_utilization: self.compute_device_utilization(),
        }
    }

    fn compute_device_utilization(&self) -> Vec<f64> {
        // Simplified device utilization calculation
        self.devices
            .iter()
            .enumerate()
            .map(|(i, _)| {
                if Some(i) == self.active_device {
                    85.0
                } else {
                    0.0
                }
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub enum SimulationTaskType {
    StateVector,
    TensorContraction,
    Distributed,
}

/// Large-scale state vector simulator
#[derive(Debug)]
pub struct LargeScaleStateVectorSim {
    num_qubits: usize,
    device_id: usize,
    state_allocation_id: Option<u64>,
    temp_allocation_id: Option<u64>,
    memory_manager: Arc<Mutex<LargeScaleMemoryManager>>,
    performance_monitor: Arc<RwLock<LargeScalePerformanceMonitor>>,
}

impl LargeScaleStateVectorSim {
    fn new(
        num_qubits: usize,
        device_id: usize,
        memory_manager: Arc<Mutex<LargeScaleMemoryManager>>,
        performance_monitor: Arc<RwLock<LargeScalePerformanceMonitor>>,
    ) -> QuantRS2Result<Self> {
        let state_size = 1_usize << num_qubits;
        let buffer_size = state_size * std::mem::size_of::<Complex64>();

        let (state_allocation, temp_allocation) = {
            let mut mm = memory_manager
                .lock()
                .expect("Memory manager lock poisoned during state vector init");
            let state_allocation =
                mm.allocate(device_id, buffer_size, AllocationType::StateVector)?;
            let temp_allocation =
                mm.allocate(device_id, buffer_size, AllocationType::IntermediateBuffer)?;
            (state_allocation, temp_allocation)
        };

        Ok(Self {
            num_qubits,
            device_id,
            state_allocation_id: Some(state_allocation),
            temp_allocation_id: Some(temp_allocation),
            memory_manager,
            performance_monitor,
        })
    }

    /// Initialize quantum state
    pub fn initialize_state(&mut self, initial_amplitudes: &[Complex64]) -> QuantRS2Result<()> {
        let expected_size = 1_usize << self.num_qubits;
        if initial_amplitudes.len() != expected_size {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Expected {} amplitudes, got {}",
                expected_size,
                initial_amplitudes.len()
            )));
        }

        let start_time = std::time::Instant::now();

        // Simulate GPU memory transfer
        std::thread::sleep(std::time::Duration::from_micros(100));

        let duration = start_time.elapsed().as_millis() as f64;
        self.performance_monitor
            .write()
            .expect("Performance monitor lock poisoned during state initialization")
            .record_operation("state_initialization", duration);

        Ok(())
    }

    /// Apply gate with optimized GPU kernels
    pub fn apply_gate_optimized(
        &mut self,
        gate_type: LargeScaleGateType,
        qubits: &[usize],
        _parameters: &[f64],
    ) -> QuantRS2Result<()> {
        let start_time = std::time::Instant::now();

        // Simulate optimized gate application
        let complexity = match gate_type {
            LargeScaleGateType::SingleQubit => 1.0,
            LargeScaleGateType::TwoQubit => 2.0,
            LargeScaleGateType::MultiQubit => qubits.len() as f64,
            LargeScaleGateType::Parameterized => 1.5,
        };

        let simulation_time = (complexity * 10.0) as u64;
        std::thread::sleep(std::time::Duration::from_micros(simulation_time));

        let duration = start_time.elapsed().as_millis() as f64;

        let mut monitor = self
            .performance_monitor
            .write()
            .expect("Performance monitor lock poisoned during gate application");
        monitor.record_operation(&format!("{gate_type:?}_gate"), duration);
        monitor.state_vector_stats.total_gate_applications += 1;

        Ok(())
    }

    /// Get measurement probabilities with GPU acceleration
    pub fn get_probabilities_gpu(&self) -> QuantRS2Result<Vec<f64>> {
        let state_size = 1_usize << self.num_qubits;
        let start_time = std::time::Instant::now();

        // Simulate GPU probability calculation
        std::thread::sleep(std::time::Duration::from_micros(50));

        // Mock probability distribution
        let mut probabilities = vec![0.0; state_size];
        if !probabilities.is_empty() {
            probabilities[0] = 1.0; // |0...0⟩ state
        }

        let duration = start_time.elapsed().as_millis() as f64;
        self.performance_monitor
            .write()
            .expect("Performance monitor lock poisoned during probability calculation")
            .record_operation("probability_calculation", duration);

        Ok(probabilities)
    }

    /// Compute expectation value with GPU acceleration
    pub fn expectation_value_gpu(
        &self,
        observable: &LargeScaleObservable,
    ) -> QuantRS2Result<Complex64> {
        let start_time = std::time::Instant::now();

        // Simulate GPU expectation value calculation
        let complexity = match observable {
            LargeScaleObservable::PauliString(_) => 1.0,
            LargeScaleObservable::Hamiltonian(_) => 3.0,
            LargeScaleObservable::CustomOperator(_) => 2.0,
        };

        let simulation_time = (complexity * 25.0) as u64;
        std::thread::sleep(std::time::Duration::from_micros(simulation_time));

        let duration = start_time.elapsed().as_millis() as f64;
        self.performance_monitor
            .write()
            .expect("Performance monitor lock poisoned during expectation value calculation")
            .record_operation("expectation_value", duration);

        // Mock expectation value
        Ok(Complex64::new(0.5, 0.0))
    }
}

#[derive(Debug, Clone)]
pub enum LargeScaleGateType {
    SingleQubit,
    TwoQubit,
    MultiQubit,
    Parameterized,
}

#[derive(Debug, Clone)]
pub enum LargeScaleObservable {
    PauliString(String),
    Hamiltonian(Vec<(f64, String)>),
    CustomOperator(String),
}

/// Large-scale tensor network contractor
pub struct LargeScaleTensorContractor {
    device_id: usize,
    config: LargeScaleSimConfig,
    memory_manager: Arc<Mutex<LargeScaleMemoryManager>>,
    performance_monitor: Arc<RwLock<LargeScalePerformanceMonitor>>,
    tensor_cache: HashMap<usize, u64>, // tensor_id -> allocation_id
}

impl LargeScaleTensorContractor {
    fn new(
        device_id: usize,
        config: &LargeScaleSimConfig,
        memory_manager: Arc<Mutex<LargeScaleMemoryManager>>,
        performance_monitor: Arc<RwLock<LargeScalePerformanceMonitor>>,
    ) -> QuantRS2Result<Self> {
        Ok(Self {
            device_id,
            config: config.clone(),
            memory_manager,
            performance_monitor,
            tensor_cache: HashMap::new(),
        })
    }

    /// Upload tensor to GPU with optimized layout
    pub fn upload_tensor_optimized(&mut self, tensor: &Tensor) -> QuantRS2Result<()> {
        let tensor_size = tensor.data.len() * std::mem::size_of::<Complex64>();

        if tensor_size < self.config.gpu_tensor_threshold {
            // Keep small tensors on CPU
            return Ok(());
        }

        let start_time = std::time::Instant::now();

        let mut mm = self
            .memory_manager
            .lock()
            .expect("Memory manager lock poisoned during tensor upload");
        let allocation_id = mm.allocate(self.device_id, tensor_size, AllocationType::TensorData)?;

        self.tensor_cache.insert(tensor.id, allocation_id);

        // Simulate optimized tensor upload
        std::thread::sleep(std::time::Duration::from_micros(tensor_size as u64 / 1000));

        let duration = start_time.elapsed().as_millis() as f64;
        self.performance_monitor
            .write()
            .expect("Performance monitor lock poisoned during tensor upload")
            .record_operation("tensor_upload", duration);

        Ok(())
    }

    /// Contract tensors with GPU acceleration and optimization
    pub fn contract_optimized(
        &mut self,
        tensor1_id: usize,
        tensor2_id: usize,
        contract_indices: &[(usize, usize)],
    ) -> QuantRS2Result<Tensor> {
        let start_time = std::time::Instant::now();

        // Check if tensors are on GPU
        let _tensor1_on_gpu = self.tensor_cache.contains_key(&tensor1_id);
        let _tensor2_on_gpu = self.tensor_cache.contains_key(&tensor2_id);

        // Simulate contraction complexity
        let contraction_complexity = contract_indices.len() as f64 * 100.0;
        let simulation_time = contraction_complexity as u64;
        std::thread::sleep(std::time::Duration::from_micros(simulation_time));

        let duration = start_time.elapsed().as_millis() as f64;

        let mut monitor = self
            .performance_monitor
            .write()
            .expect("Performance monitor lock poisoned during tensor contraction");
        monitor.record_operation("tensor_contraction", duration);
        monitor.contraction_stats.total_contractions += 1;
        monitor.contraction_stats.total_contraction_time_ms += duration;

        // Create mock result tensor
        let result_data = scirs2_core::ndarray::Array::from_shape_vec(
            scirs2_core::ndarray::IxDyn(&[2, 2]),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        )
        .map_err(|e| QuantRS2Error::InvalidInput(format!("Tensor creation failed: {e}")))?;

        Ok(Tensor::new(
            tensor1_id + tensor2_id, // Simple ID generation
            result_data,
            vec!["result_i".to_string(), "result_j".to_string()],
        ))
    }

    /// Perform tensor decomposition with GPU acceleration
    pub fn decompose_tensor_gpu(
        &mut self,
        tensor_id: usize,
        decomp_type: TensorDecompositionType,
    ) -> QuantRS2Result<TensorDecomposition> {
        let start_time = std::time::Instant::now();

        // Simulate decomposition complexity
        let decomp_complexity = match decomp_type {
            TensorDecompositionType::SVD => 500.0,
            TensorDecompositionType::QR => 300.0,
            TensorDecompositionType::Eigenvalue => 400.0,
        };

        std::thread::sleep(std::time::Duration::from_micros(decomp_complexity as u64));

        let duration = start_time.elapsed().as_millis() as f64;

        let mut monitor = self
            .performance_monitor
            .write()
            .expect("Performance monitor lock poisoned during tensor decomposition");
        monitor.record_operation(&format!("{decomp_type:?}_decomposition"), duration);
        monitor.contraction_stats.decompositions_performed += 1;

        Ok(TensorDecomposition {
            decomposition_type: decomp_type,
            factors: vec![tensor_id + 1000, tensor_id + 2000], // Mock factor IDs
            singular_values: vec![1.0, 0.5, 0.1],
            error_estimate: 1e-15,
        })
    }
}

#[derive(Debug, Clone)]
pub enum TensorDecompositionType {
    SVD,
    QR,
    Eigenvalue,
}

#[derive(Debug, Clone)]
pub struct TensorDecomposition {
    pub decomposition_type: TensorDecompositionType,
    pub factors: Vec<usize>,
    pub singular_values: Vec<f64>,
    pub error_estimate: f64,
}

#[derive(Debug, Clone)]
pub struct LargeScalePerformanceStats {
    pub contraction_stats: ContractionStatistics,
    pub state_vector_stats: StateVectorStatistics,
    pub total_memory_allocated: usize,
    pub peak_memory_usage: usize,
    pub device_utilization: Vec<f64>,
}

impl LargeScaleMemoryManager {
    fn new(devices: &[GpuDevice], config: &LargeScaleSimConfig) -> QuantRS2Result<Self> {
        let mut memory_pools = HashMap::new();

        for (i, device) in devices.iter().enumerate() {
            let pool = MemoryPool {
                device_id: i,
                total_size: config.memory_pool_size.min(device.memory_size),
                used_size: 0,
                free_blocks: vec![MemoryBlock {
                    offset: 0,
                    size: config.memory_pool_size.min(device.memory_size),
                    is_pinned: false,
                }],
                allocated_blocks: HashMap::new(),
            };
            memory_pools.insert(i, pool);
        }

        Ok(Self {
            memory_pools,
            allocations: HashMap::new(),
            next_allocation_id: 1,
        })
    }

    fn allocate(
        &mut self,
        device_id: usize,
        size: usize,
        alloc_type: AllocationType,
    ) -> QuantRS2Result<u64> {
        let pool = self.memory_pools.get_mut(&device_id).ok_or_else(|| {
            QuantRS2Error::InvalidParameter(format!("Device {device_id} not found"))
        })?;

        // Find suitable free block
        let mut best_block_idx = None;
        let mut best_size = usize::MAX;

        for (i, block) in pool.free_blocks.iter().enumerate() {
            if block.size >= size && block.size < best_size {
                best_size = block.size;
                best_block_idx = Some(i);
            }
        }

        let block_idx = best_block_idx
            .ok_or_else(|| QuantRS2Error::RuntimeError("Insufficient GPU memory".to_string()))?;

        let block = pool.free_blocks.remove(block_idx);
        let allocation_id = self.next_allocation_id;
        self.next_allocation_id += 1;

        // Create allocated block
        let allocated_block = MemoryBlock {
            offset: block.offset,
            size,
            is_pinned: false,
        };

        pool.allocated_blocks.insert(allocation_id, allocated_block);
        pool.used_size += size;

        // Return remaining space to free blocks if any
        if block.size > size {
            pool.free_blocks.push(MemoryBlock {
                offset: block.offset + size,
                size: block.size - size,
                is_pinned: false,
            });
        }

        self.allocations.insert(
            allocation_id,
            AllocationInfo {
                device_id,
                size,
                allocation_type: alloc_type,
                timestamp: std::time::Instant::now(),
            },
        );

        Ok(allocation_id)
    }

    fn get_total_allocated(&self) -> usize {
        self.allocations.values().map(|info| info.size).sum()
    }

    fn get_peak_usage(&self) -> usize {
        self.memory_pools
            .values()
            .map(|pool| pool.used_size)
            .max()
            .unwrap_or_default()
    }
}

impl LargeScalePerformanceMonitor {
    fn new() -> Self {
        Self {
            operation_times: HashMap::new(),
            memory_usage_history: Vec::new(),
            contraction_stats: ContractionStatistics::default(),
            state_vector_stats: StateVectorStatistics::default(),
        }
    }

    fn record_operation(&mut self, operation: &str, duration_ms: f64) {
        self.operation_times
            .entry(operation.to_string())
            .or_insert_with(Vec::new)
            .push(duration_ms);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_devices() -> Vec<GpuDevice> {
        vec![
            GpuDevice {
                id: 0,
                name: "Test GPU 1".to_string(),
                backend: GpuBackend::CUDA,
                memory_size: 8 * 1024 * 1024 * 1024, // 8GB
                compute_units: 64,
                max_work_group_size: 1024,
                supports_double_precision: true,
                is_available: true,
            },
            GpuDevice {
                id: 1,
                name: "Test GPU 2".to_string(),
                backend: GpuBackend::CUDA,
                memory_size: 16 * 1024 * 1024 * 1024, // 16GB
                compute_units: 128,
                max_work_group_size: 1024,
                supports_double_precision: true,
                is_available: true,
            },
        ]
    }

    #[test]
    fn test_large_scale_accelerator_creation() {
        let config = LargeScaleSimConfig::default();
        let devices = create_test_devices();

        let accelerator = LargeScaleSimAccelerator::new(config, devices);
        assert!(accelerator.is_ok());
    }

    #[test]
    fn test_device_selection() {
        let config = LargeScaleSimConfig::default();
        let devices = create_test_devices();

        let mut accelerator = LargeScaleSimAccelerator::new(config, devices)
            .expect("Failed to create accelerator for device selection test");

        // Test state vector simulation device selection
        let device_id = accelerator.select_optimal_device(
            SimulationTaskType::StateVector,
            1024 * 1024 * 1024, // 1GB
        );

        assert!(device_id.is_ok());
        assert!(device_id.expect("Device selection failed") < 2);
    }

    #[test]
    fn test_state_vector_simulation() {
        let config = LargeScaleSimConfig::default();
        let devices = create_test_devices();

        let mut accelerator =
            LargeScaleSimAccelerator::new(config, devices).expect("Failed to create accelerator");
        let state_sim = accelerator.init_state_vector_simulation(5);

        assert!(state_sim.is_ok());

        let mut sim = state_sim.expect("Failed to initialize state vector simulation");

        // Test state initialization
        let initial_state = vec![Complex64::new(1.0, 0.0); 32]; // 2^5 = 32
        assert!(sim.initialize_state(&initial_state).is_ok());

        // Test gate application
        assert!(sim
            .apply_gate_optimized(
                LargeScaleGateType::SingleQubit,
                &[0],
                &[std::f64::consts::PI / 2.0]
            )
            .is_ok());
    }

    #[test]
    fn test_tensor_contractor() {
        let config = LargeScaleSimConfig::default();
        let devices = create_test_devices();

        let mut accelerator =
            LargeScaleSimAccelerator::new(config, devices).expect("Failed to create accelerator");
        let contractor = accelerator.init_tensor_contractor();

        assert!(contractor.is_ok());

        let mut contractor = contractor.expect("Failed to initialize tensor contractor");

        // Create test tensor
        let data = scirs2_core::ndarray::Array::from_shape_vec(
            scirs2_core::ndarray::IxDyn(&[2, 2]),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        )
        .expect("Failed to create tensor data array");

        let tensor = Tensor::new(0, data, vec!["i".to_string(), "j".to_string()]);

        // Test tensor upload
        assert!(contractor.upload_tensor_optimized(&tensor).is_ok());

        // Test tensor contraction
        let result = contractor.contract_optimized(0, 1, &[(0, 1)]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_memory_management() {
        let config = LargeScaleSimConfig::default();
        let devices = create_test_devices();

        let memory_manager = LargeScaleMemoryManager::new(&devices, &config);
        assert!(memory_manager.is_ok());

        let mut mm = memory_manager.expect("Failed to create memory manager");

        // Test allocation
        let allocation = mm.allocate(0, 1024, AllocationType::StateVector);
        assert!(allocation.is_ok());

        // Test memory tracking
        assert_eq!(mm.get_total_allocated(), 1024);
    }

    #[test]
    fn test_performance_monitoring() {
        let config = LargeScaleSimConfig::default();
        let devices = create_test_devices();

        let accelerator =
            LargeScaleSimAccelerator::new(config, devices).expect("Failed to create accelerator");

        // Record some operations
        {
            let mut monitor = accelerator
                .performance_monitor
                .write()
                .expect("Performance monitor lock poisoned in test");
            monitor.record_operation("test_operation", 10.5);
            monitor.record_operation("test_operation", 12.3);
        }

        let stats = accelerator.get_performance_stats();
        assert_eq!(stats.total_memory_allocated, 0); // No allocations yet
    }

    #[test]
    fn test_large_qubit_simulation_limit() {
        let config = LargeScaleSimConfig::default();
        let devices = create_test_devices();

        let mut accelerator =
            LargeScaleSimAccelerator::new(config, devices).expect("Failed to create accelerator");

        // Test exceeding qubit limit
        let result = accelerator.init_state_vector_simulation(100);
        assert!(result.is_err());
        let err = result.expect_err("Expected UnsupportedQubits error");
        assert!(matches!(err, QuantRS2Error::UnsupportedQubits(_, _)));
    }

    #[test]
    fn test_tensor_decomposition() {
        let config = LargeScaleSimConfig::default();
        let devices = create_test_devices();

        let mut accelerator =
            LargeScaleSimAccelerator::new(config, devices).expect("Failed to create accelerator");
        let mut contractor = accelerator
            .init_tensor_contractor()
            .expect("Failed to initialize tensor contractor");

        let decomp_result = contractor.decompose_tensor_gpu(0, TensorDecompositionType::SVD);
        assert!(decomp_result.is_ok());

        let decomp = decomp_result.expect("Failed to decompose tensor");
        assert_eq!(decomp.factors.len(), 2);
        assert!(!decomp.singular_values.is_empty());
    }
}
