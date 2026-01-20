//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use quantrs2_circuit::prelude::Circuit;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;
use std::collections::HashMap;
use thiserror::Error;

/// cuQuantum simulation configuration
#[derive(Debug, Clone)]
pub struct CuQuantumConfig {
    /// Device ID to use (-1 for auto-select)
    pub device_id: i32,
    /// Enable multi-GPU execution
    pub multi_gpu: bool,
    /// Number of GPUs to use (0 for all available)
    pub num_gpus: usize,
    /// Memory pool size in bytes (0 for auto)
    pub memory_pool_size: usize,
    /// Enable asynchronous execution
    pub async_execution: bool,
    /// Enable memory optimization (may reduce peak memory)
    pub memory_optimization: bool,
    /// Computation precision
    pub precision: ComputePrecision,
    /// Gate fusion level
    pub gate_fusion_level: GateFusionLevel,
    /// Enable profiling
    pub enable_profiling: bool,
    /// Maximum number of qubits for state vector simulation
    pub max_statevec_qubits: usize,
    /// Tensor network contraction algorithm
    pub tensor_contraction: TensorContractionAlgorithm,
    /// Enable TF32 tensor core mode (NVIDIA Ampere and newer)
    /// When enabled, FP32 matrix operations use 19-bit TensorFloat-32 format
    /// providing near-FP32 accuracy with ~8x speedup on tensor cores
    /// Only effective when device has tensor cores (compute capability ≥ 8.0)
    pub enable_tf32: bool,
}
impl CuQuantumConfig {
    /// Create configuration optimized for large circuits
    pub fn large_circuit() -> Self {
        Self {
            memory_optimization: true,
            gate_fusion_level: GateFusionLevel::Aggressive,
            tensor_contraction: TensorContractionAlgorithm::OptimalWithSlicing,
            enable_tf32: true, // Enable TF32 for performance
            ..Default::default()
        }
    }
    /// Create configuration optimized for variational algorithms (VQE/QAOA)
    pub fn variational() -> Self {
        Self {
            async_execution: true,
            gate_fusion_level: GateFusionLevel::Moderate,
            enable_profiling: false,
            enable_tf32: true, // Enable TF32 for VQE/QAOA speedup
            ..Default::default()
        }
    }
    /// Create configuration for multi-GPU execution
    pub fn multi_gpu(num_gpus: usize) -> Self {
        Self {
            multi_gpu: true,
            num_gpus,
            memory_optimization: true,
            enable_tf32: true, // Enable TF32 on all GPUs
            ..Default::default()
        }
    }

    /// Create configuration with TF32 explicitly enabled/disabled
    pub fn with_tf32(mut self, enable: bool) -> Self {
        self.enable_tf32 = enable;
        self
    }

    /// Check if TF32 should be used based on device capabilities
    pub fn should_use_tf32(&self, device_info: &CudaDeviceInfo) -> bool {
        self.enable_tf32
            && device_info.has_tensor_cores
            && device_info.compute_capability >= (8, 0) // Ampere and newer
            && matches!(
                self.precision,
                ComputePrecision::Single | ComputePrecision::Mixed
            )
    }
}
/// CUDA device information
#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    /// Device ID
    pub device_id: i32,
    /// Device name
    pub name: String,
    /// Total global memory in bytes
    pub total_memory: usize,
    /// Free memory in bytes
    pub free_memory: usize,
    /// Compute capability (major, minor)
    pub compute_capability: (i32, i32),
    /// Number of streaming multiprocessors
    pub sm_count: i32,
    /// Maximum threads per block
    pub max_threads_per_block: i32,
    /// Warp size
    pub warp_size: i32,
    /// Whether tensor cores are available
    pub has_tensor_cores: bool,
}
impl CudaDeviceInfo {
    /// Get maximum qubits supportable for state vector simulation
    pub fn max_statevec_qubits(&self) -> usize {
        let available_memory = (self.free_memory as f64 * 0.8) as usize;
        let bytes_per_amplitude = 16;
        let max_amplitudes = available_memory / bytes_per_amplitude;
        (max_amplitudes as f64).log2().floor() as usize
    }
}
/// Recommended simulation backend
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendedBackend {
    /// Use state vector simulation (smaller circuits)
    StateVector,
    /// Use tensor network simulation (larger circuits)
    TensorNetwork,
    /// Hybrid approach
    Hybrid,
    /// Cannot simulate (too large)
    NotFeasible,
}
/// Tensor network state representation
#[derive(Debug, Clone)]
pub struct TensorNetworkState {
    /// Tensors in the network
    tensors: Vec<Tensor>,
    /// Connections between tensors
    edges: Vec<TensorEdge>,
    /// Open indices (not contracted)
    open_indices: Vec<usize>,
}
impl TensorNetworkState {
    /// Create from a quantum circuit
    pub fn from_circuit<const N: usize>(circuit: &Circuit<N>) -> Result<Self> {
        let mut tensors = Vec::new();
        let mut edges = Vec::new();
        for qubit in 0..N {
            tensors.push(Tensor::initial_state(qubit));
        }
        for (gate_idx, gate) in circuit.gates().iter().enumerate() {
            let qubits: Vec<usize> = gate.qubits().iter().map(|q| q.id() as usize).collect();
            tensors.push(Tensor::from_gate(gate_idx, &qubits));
            for &qubit in &qubits {
                edges.push(TensorEdge {
                    tensor_a: qubit,
                    tensor_b: N + gate_idx,
                    index: qubit,
                });
            }
        }
        Ok(Self {
            tensors,
            edges,
            open_indices: (0..N).collect(),
        })
    }
    /// Get number of tensors
    pub fn num_tensors(&self) -> usize {
        self.tensors.len()
    }
    /// Get number of edges
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }
}
/// cuQuantum simulation result
#[derive(Debug, Clone)]
pub struct CuQuantumResult {
    /// State vector (if computed)
    pub state_vector: Option<Array1<Complex64>>,
    /// Measurement counts
    pub counts: HashMap<String, usize>,
    /// Individual measurement outcomes
    pub measurement_outcomes: Vec<u64>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// Number of qubits
    pub num_qubits: usize,
}
impl CuQuantumResult {
    /// Create a new result with state vector
    pub fn from_state_vector(state: Array1<Complex64>, num_qubits: usize) -> Self {
        Self {
            state_vector: Some(state),
            counts: HashMap::new(),
            measurement_outcomes: Vec::new(),
            metadata: HashMap::new(),
            num_qubits,
        }
    }
    /// Create a new result with measurement counts
    pub fn from_counts(counts: HashMap<String, usize>, num_qubits: usize) -> Self {
        Self {
            state_vector: None,
            counts,
            measurement_outcomes: Vec::new(),
            metadata: HashMap::new(),
            num_qubits,
        }
    }
    /// Get probabilities from state vector
    pub fn probabilities(&self) -> Option<Vec<f64>> {
        self.state_vector
            .as_ref()
            .map(|sv| sv.iter().map(|c| c.norm_sqr()).collect())
    }
    /// Get expectation value of computational basis measurement
    pub fn expectation_z(&self, qubit: usize) -> Option<f64> {
        self.probabilities().map(|probs| {
            let mut exp = 0.0;
            for (i, &p) in probs.iter().enumerate() {
                let bit = (i >> qubit) & 1;
                exp += if bit == 0 { p } else { -p };
            }
            exp
        })
    }
}
/// Single tensor in the network
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Tensor ID
    id: usize,
    /// Shape of the tensor
    shape: Vec<usize>,
    /// Data (only stored for leaf tensors)
    data: Option<Array2<Complex64>>,
}
impl Tensor {
    /// Create initial state tensor |0⟩
    fn initial_state(qubit: usize) -> Self {
        let mut data = Array2::zeros((2, 1));
        data[[0, 0]] = Complex64::new(1.0, 0.0);
        Self {
            id: qubit,
            shape: vec![2],
            data: Some(data),
        }
    }
    /// Create tensor from gate
    fn from_gate(gate_idx: usize, _qubits: &[usize]) -> Self {
        Self {
            id: gate_idx,
            shape: vec![2; _qubits.len() * 2],
            data: None,
        }
    }
}
/// Edge connecting two tensors
#[derive(Debug, Clone)]
pub struct TensorEdge {
    /// First tensor index
    tensor_a: usize,
    /// Second tensor index
    tensor_b: usize,
    /// Index being contracted
    index: usize,
}
/// cuStateVec-based state vector simulator
///
/// This simulator uses NVIDIA's cuStateVec library for GPU-accelerated
/// state vector simulation of quantum circuits.
pub struct CuStateVecSimulator {
    /// Configuration
    pub config: CuQuantumConfig,
    /// Device information
    pub device_info: Option<CudaDeviceInfo>,
    /// Simulation statistics
    pub stats: SimulationStats,
    /// Whether the simulator is initialized
    pub initialized: bool,
    #[cfg(feature = "cuquantum")]
    pub handle: Option<CuStateVecHandle>,
    #[cfg(feature = "cuquantum")]
    pub state_buffer: Option<GpuBuffer>,
}
impl CuStateVecSimulator {
    /// Create a new cuStateVec simulator
    pub fn new(config: CuQuantumConfig) -> Result<Self> {
        let device_info = Self::get_device_info(config.device_id)?;
        Ok(Self {
            config,
            device_info: Some(device_info),
            stats: SimulationStats::default(),
            initialized: false,
            #[cfg(feature = "cuquantum")]
            handle: None,
            #[cfg(feature = "cuquantum")]
            state_buffer: None,
        })
    }
    /// Create with default configuration
    pub fn default_config() -> Result<Self> {
        Self::new(CuQuantumConfig::default())
    }
    /// Check if cuQuantum is available
    pub fn is_available() -> bool {
        #[cfg(feature = "cuquantum")]
        {
            Self::check_cuquantum_available()
        }
        #[cfg(not(feature = "cuquantum"))]
        {
            false
        }
    }
    /// Get device information
    pub fn get_device_info(device_id: i32) -> Result<CudaDeviceInfo> {
        #[cfg(feature = "cuquantum")]
        {
            Self::get_cuda_device_info(device_id)
        }
        #[cfg(not(feature = "cuquantum"))]
        {
            Ok(CudaDeviceInfo {
                device_id: if device_id < 0 { 0 } else { device_id },
                name: "Mock CUDA Device (cuQuantum not available)".to_string(),
                total_memory: 16 * 1024 * 1024 * 1024,
                free_memory: 12 * 1024 * 1024 * 1024,
                compute_capability: (8, 6),
                sm_count: 84,
                max_threads_per_block: 1024,
                warp_size: 32,
                has_tensor_cores: true,
            })
        }
    }
    /// Initialize the simulator for a specific number of qubits
    pub fn initialize(&mut self, num_qubits: usize) -> Result<()> {
        if num_qubits > self.config.max_statevec_qubits {
            return Err(SimulatorError::InvalidParameter(format!(
                "Number of qubits ({}) exceeds maximum ({})",
                num_qubits, self.config.max_statevec_qubits
            )));
        }
        #[cfg(feature = "cuquantum")]
        {
            self.initialize_custatevec(num_qubits)?;
        }
        self.initialized = true;
        Ok(())
    }
    /// Simulate a quantum circuit
    pub fn simulate<const N: usize>(&mut self, circuit: &Circuit<N>) -> Result<CuQuantumResult> {
        if !self.initialized {
            self.initialize(N)?;
        }
        let start_time = std::time::Instant::now();
        #[cfg(target_os = "macos")]
        {
            self.simulate_mock(circuit, start_time)
        }
        #[cfg(all(feature = "cuquantum", not(target_os = "macos")))]
        {
            self.simulate_with_custatevec(circuit)
        }
        #[cfg(all(not(feature = "cuquantum"), not(target_os = "macos")))]
        {
            self.simulate_mock(circuit, start_time)
        }
    }
    /// Mock simulation for non-CUDA platforms
    /// Available on macOS (always) and when cuquantum feature is disabled
    #[cfg(any(target_os = "macos", not(feature = "cuquantum")))]
    fn simulate_mock<const N: usize>(
        &mut self,
        circuit: &Circuit<N>,
        start_time: std::time::Instant,
    ) -> Result<CuQuantumResult> {
        let state_size = 1 << N;
        let mut state = Array1::zeros(state_size);
        state[0] = Complex64::new(1.0, 0.0);
        self.stats.total_simulations += 1;
        self.stats.total_gates += circuit.gates().len();
        self.stats.total_time_ms += start_time.elapsed().as_millis() as f64;
        Ok(CuQuantumResult::from_state_vector(state, N))
    }
    /// Get simulation statistics
    pub fn stats(&self) -> &SimulationStats {
        &self.stats
    }
    /// Reset simulation statistics
    pub fn reset_stats(&mut self) {
        self.stats = SimulationStats::default();
    }
    /// Get device information
    pub fn device_info(&self) -> Option<&CudaDeviceInfo> {
        self.device_info.as_ref()
    }
    #[cfg(feature = "cuquantum")]
    fn check_cuquantum_available() -> bool {
        false
    }
    #[cfg(feature = "cuquantum")]
    fn get_cuda_device_info(device_id: i32) -> Result<CudaDeviceInfo> {
        #[cfg(target_os = "macos")]
        {
            Ok(CudaDeviceInfo {
                device_id: if device_id < 0 { 0 } else { device_id },
                name: "Mock CUDA Device (macOS - no CUDA)".to_string(),
                total_memory: 24 * 1024 * 1024 * 1024,
                free_memory: 20 * 1024 * 1024 * 1024,
                compute_capability: (8, 9),
                sm_count: 128,
                max_threads_per_block: 1024,
                warp_size: 32,
                has_tensor_cores: true,
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            Ok(CudaDeviceInfo {
                device_id: if device_id < 0 { 0 } else { device_id },
                name: "Mock CUDA Device (cuQuantum stub)".to_string(),
                total_memory: 24 * 1024 * 1024 * 1024,
                free_memory: 20 * 1024 * 1024 * 1024,
                compute_capability: (8, 9),
                sm_count: 128,
                max_threads_per_block: 1024,
                warp_size: 32,
                has_tensor_cores: true,
            })
        }
    }
    #[cfg(feature = "cuquantum")]
    fn initialize_custatevec(&mut self, num_qubits: usize) -> Result<()> {
        Ok(())
    }
    #[cfg(feature = "cuquantum")]
    fn simulate_with_custatevec<const N: usize>(
        &mut self,
        circuit: &Circuit<N>,
    ) -> Result<CuQuantumResult> {
        Err(SimulatorError::GpuError(
            "cuStateVec simulation not yet implemented".to_string(),
        ))
    }
}
/// Simulation statistics
#[derive(Debug, Clone, Default)]
pub struct SimulationStats {
    /// Total number of simulations run
    pub total_simulations: usize,
    /// Total gates applied
    pub total_gates: usize,
    /// Total simulation time in milliseconds
    pub total_time_ms: f64,
    /// Peak GPU memory usage in bytes
    pub peak_memory_bytes: usize,
    /// Number of tensor contractions (for cuTensorNet)
    pub tensor_contractions: usize,
    /// Total FLOP count
    pub total_flops: f64,
}
impl SimulationStats {
    /// Get average gates per simulation
    pub fn avg_gates_per_sim(&self) -> f64 {
        if self.total_simulations > 0 {
            self.total_gates as f64 / self.total_simulations as f64
        } else {
            0.0
        }
    }
    /// Get average time per simulation in milliseconds
    pub fn avg_time_per_sim(&self) -> f64 {
        if self.total_simulations > 0 {
            self.total_time_ms / self.total_simulations as f64
        } else {
            0.0
        }
    }
    /// Get throughput in GFLOP/s
    pub fn throughput_gflops(&self) -> f64 {
        if self.total_time_ms > 0.0 {
            (self.total_flops / 1e9) / (self.total_time_ms / 1000.0)
        } else {
            0.0
        }
    }
}
/// Computation precision
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputePrecision {
    /// Half precision (float16) - reduced memory, faster on tensor cores
    /// Suitable for approximate calculations where high precision isn't critical
    Half,
    /// Single precision (float32) - balanced precision and performance
    /// Recommended for most quantum simulations
    Single,
    /// Double precision (float64) - highest precision
    /// Required for high-fidelity simulations and error-sensitive algorithms
    Double,
    /// Mixed precision (automatic FP16/FP32 switching)
    /// Uses FP16 for matrix operations (tensor cores) and FP32 for accumulation
    /// Provides near-FP32 accuracy with FP16 speed
    Mixed,
}

impl ComputePrecision {
    /// Get bytes per complex amplitude for this precision
    pub fn bytes_per_amplitude(self) -> usize {
        match self {
            ComputePrecision::Half => 4,    // FP16: 2 bytes × 2 (complex)
            ComputePrecision::Single => 8,  // FP32: 4 bytes × 2 (complex)
            ComputePrecision::Double => 16, // FP64: 8 bytes × 2 (complex)
            ComputePrecision::Mixed => 8,   // Mixed: FP32 for state vector storage
        }
    }

    /// Get relative speed multiplier (approximate)
    /// Higher values = faster computation
    pub fn speed_factor(self) -> f64 {
        match self {
            ComputePrecision::Half => 2.0, // ~2x faster than FP32 on tensor cores
            ComputePrecision::Single => 1.0, // Baseline
            ComputePrecision::Double => 0.5, // ~2x slower than FP32
            ComputePrecision::Mixed => 1.7, // ~1.7x faster than FP32 (with tensor cores)
        }
    }

    /// Get relative accuracy (approximate)
    /// Higher values = more accurate
    pub fn accuracy_factor(self) -> f64 {
        match self {
            ComputePrecision::Half => 0.3,   // ~3 decimal digits precision
            ComputePrecision::Single => 1.0, // ~7 decimal digits precision (baseline)
            ComputePrecision::Double => 2.2, // ~15 decimal digits precision
            ComputePrecision::Mixed => 0.95, // Near-FP32 accuracy
        }
    }

    /// Check if precision uses tensor cores (if available)
    pub fn uses_tensor_cores(self) -> bool {
        matches!(self, ComputePrecision::Half | ComputePrecision::Mixed)
    }

    /// Get human-readable description
    pub fn description(self) -> &'static str {
        match self {
            ComputePrecision::Half => {
                "Half precision (FP16): Fastest, lowest memory, reduced accuracy"
            }
            ComputePrecision::Single => {
                "Single precision (FP32): Balanced speed and accuracy, recommended"
            }
            ComputePrecision::Double => {
                "Double precision (FP64): Highest accuracy, slower, more memory"
            }
            ComputePrecision::Mixed => {
                "Mixed precision (FP16/FP32): Near-FP32 accuracy with FP16 speed on tensor cores"
            }
        }
    }
}
/// cuQuantum-specific errors
#[derive(Debug, Error)]
pub enum CuQuantumError {
    #[error("cuQuantum not available: {0}")]
    NotAvailable(String),
    #[error("CUDA error: {0}")]
    CudaError(String),
    #[error("cuStateVec error: {0}")]
    CuStateVecError(String),
    #[error("cuTensorNet error: {0}")]
    CuTensorNetError(String),
    #[error("Memory allocation error: {0}")]
    MemoryError(String),
    #[error("Invalid configuration: {0}")]
    ConfigError(String),
    #[error("Device error: {0}")]
    DeviceError(String),
    #[error("Simulation error: {0}")]
    SimulationError(String),
}
/// cuTensorNet-based tensor network simulator
///
/// This simulator uses NVIDIA's cuTensorNet library for GPU-accelerated
/// tensor network contraction, enabling simulation of circuits beyond
/// the state vector memory limit.
pub struct CuTensorNetSimulator {
    /// Configuration
    pub config: CuQuantumConfig,
    /// Device information
    pub device_info: Option<CudaDeviceInfo>,
    /// Simulation statistics
    pub stats: SimulationStats,
    /// Tensor network representation of the circuit
    pub tensor_network: Option<TensorNetworkState>,
}
impl CuTensorNetSimulator {
    /// Create a new cuTensorNet simulator
    pub fn new(config: CuQuantumConfig) -> Result<Self> {
        let device_info = CuStateVecSimulator::get_device_info(config.device_id)?;
        Ok(Self {
            config,
            device_info: Some(device_info),
            stats: SimulationStats::default(),
            tensor_network: None,
        })
    }
    /// Create with default configuration
    pub fn default_config() -> Result<Self> {
        Self::new(CuQuantumConfig::default())
    }
    /// Check if cuTensorNet is available
    pub fn is_available() -> bool {
        #[cfg(feature = "cuquantum")]
        {
            Self::check_cutensornet_available()
        }
        #[cfg(not(feature = "cuquantum"))]
        {
            false
        }
    }
    /// Build tensor network from circuit
    pub fn build_network<const N: usize>(&mut self, circuit: &Circuit<N>) -> Result<()> {
        self.tensor_network = Some(TensorNetworkState::from_circuit(circuit)?);
        Ok(())
    }
    /// Contract the tensor network to compute amplitudes
    pub fn contract(&mut self, output_indices: &[usize]) -> Result<Array1<Complex64>> {
        let network = self
            .tensor_network
            .as_ref()
            .ok_or_else(|| SimulatorError::InvalidParameter("Network not built".to_string()))?;
        #[cfg(target_os = "macos")]
        {
            self.contract_mock(network, output_indices)
        }
        #[cfg(all(feature = "cuquantum", not(target_os = "macos")))]
        {
            self.contract_with_cutensornet(network, output_indices)
        }
        #[cfg(all(not(feature = "cuquantum"), not(target_os = "macos")))]
        {
            self.contract_mock(network, output_indices)
        }
    }
    /// Compute expectation value of an observable
    pub fn expectation_value(&mut self, observable: &Observable) -> Result<f64> {
        let _network = self
            .tensor_network
            .as_ref()
            .ok_or_else(|| SimulatorError::InvalidParameter("Network not built".to_string()))?;
        #[cfg(target_os = "macos")]
        {
            let _ = observable;
            Ok(0.5)
        }
        #[cfg(all(feature = "cuquantum", not(target_os = "macos")))]
        {
            self.expectation_with_cutensornet(_network, observable)
        }
        #[cfg(all(not(feature = "cuquantum"), not(target_os = "macos")))]
        {
            let _ = observable;
            Ok(0.5)
        }
    }
    /// Get optimal contraction order
    pub fn find_contraction_order(&self) -> Result<ContractionPath> {
        let network = self
            .tensor_network
            .as_ref()
            .ok_or_else(|| SimulatorError::InvalidParameter("Network not built".to_string()))?;
        match self.config.tensor_contraction {
            TensorContractionAlgorithm::Auto => self.auto_contraction_order(network),
            TensorContractionAlgorithm::Greedy => self.greedy_contraction_order(network),
            TensorContractionAlgorithm::Optimal => self.optimal_contraction_order(network),
            TensorContractionAlgorithm::OptimalWithSlicing => {
                self.optimal_sliced_contraction_order(network)
            }
            TensorContractionAlgorithm::RandomGreedy => {
                self.random_greedy_contraction_order(network)
            }
        }
    }
    /// Mock contraction for non-CUDA platforms
    /// Available on macOS (always) and when cuquantum feature is disabled
    #[cfg(any(target_os = "macos", not(feature = "cuquantum")))]
    fn contract_mock(
        &self,
        _network: &TensorNetworkState,
        output_indices: &[usize],
    ) -> Result<Array1<Complex64>> {
        let size = 1 << output_indices.len();
        let mut result = Array1::zeros(size);
        result[0] = Complex64::new(1.0, 0.0);
        Ok(result)
    }
    fn auto_contraction_order(&self, network: &TensorNetworkState) -> Result<ContractionPath> {
        if network.num_tensors() < 20 {
            self.optimal_contraction_order(network)
        } else {
            self.greedy_contraction_order(network)
        }
    }
    fn greedy_contraction_order(&self, network: &TensorNetworkState) -> Result<ContractionPath> {
        let mut path = ContractionPath::new();
        let mut remaining: Vec<usize> = (0..network.num_tensors()).collect();
        while remaining.len() > 1 {
            let mut best_cost = f64::MAX;
            let mut best_pair = (0, 1);
            for i in 0..remaining.len() {
                for j in (i + 1)..remaining.len() {
                    let cost = self.estimate_contraction_cost(remaining[i], remaining[j]);
                    if cost < best_cost {
                        best_cost = cost;
                        best_pair = (i, j);
                    }
                }
            }
            path.add_contraction(remaining[best_pair.0], remaining[best_pair.1]);
            remaining.remove(best_pair.1);
        }
        Ok(path)
    }
    fn optimal_contraction_order(&self, network: &TensorNetworkState) -> Result<ContractionPath> {
        if network.num_tensors() > 15 {
            return self.greedy_contraction_order(network);
        }
        self.greedy_contraction_order(network)
    }
    fn optimal_sliced_contraction_order(
        &self,
        network: &TensorNetworkState,
    ) -> Result<ContractionPath> {
        let mut path = self.optimal_contraction_order(network)?;
        path.enable_slicing(self.config.memory_pool_size);
        Ok(path)
    }
    fn random_greedy_contraction_order(
        &self,
        network: &TensorNetworkState,
    ) -> Result<ContractionPath> {
        use scirs2_core::random::{thread_rng, Rng};
        let mut rng = thread_rng();
        let mut best_path = self.greedy_contraction_order(network)?;
        let mut best_cost = best_path.total_cost();
        for _ in 0..10 {
            let path = self.randomized_greedy_order(network, &mut rng)?;
            let cost = path.total_cost();
            if cost < best_cost {
                best_cost = cost;
                best_path = path;
            }
        }
        Ok(best_path)
    }
    fn randomized_greedy_order<R: scirs2_core::random::Rng>(
        &self,
        network: &TensorNetworkState,
        rng: &mut R,
    ) -> Result<ContractionPath> {
        let mut path = ContractionPath::new();
        let mut remaining: Vec<usize> = (0..network.num_tensors()).collect();
        while remaining.len() > 1 {
            let mut candidates: Vec<((usize, usize), f64)> = Vec::new();
            for i in 0..remaining.len() {
                for j in (i + 1)..remaining.len() {
                    let cost = self.estimate_contraction_cost(remaining[i], remaining[j]);
                    candidates.push(((i, j), cost));
                }
            }
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let pick_range = (candidates.len() / 3).max(1);
            let pick_idx = rng.gen_range(0..pick_range);
            let (best_pair, _) = candidates[pick_idx];
            path.add_contraction(remaining[best_pair.0], remaining[best_pair.1]);
            remaining.remove(best_pair.1);
        }
        Ok(path)
    }
    fn estimate_contraction_cost(&self, _tensor_a: usize, _tensor_b: usize) -> f64 {
        1.0
    }
    #[cfg(feature = "cuquantum")]
    fn check_cutensornet_available() -> bool {
        false
    }
    #[cfg(feature = "cuquantum")]
    fn contract_with_cutensornet(
        &self,
        _network: &TensorNetworkState,
        _output_indices: &[usize],
    ) -> Result<Array1<Complex64>> {
        Err(SimulatorError::GpuError(
            "cuTensorNet contraction not yet implemented".to_string(),
        ))
    }
    #[cfg(feature = "cuquantum")]
    fn expectation_with_cutensornet(
        &self,
        _network: &TensorNetworkState,
        _observable: &Observable,
    ) -> Result<f64> {
        Err(SimulatorError::GpuError(
            "cuTensorNet expectation not yet implemented".to_string(),
        ))
    }
}
/// Observable for expectation value computation
#[derive(Debug, Clone)]
pub enum Observable {
    /// Pauli Z on specified qubits
    PauliZ(Vec<usize>),
    /// Pauli X on specified qubits
    PauliX(Vec<usize>),
    /// Pauli Y on specified qubits
    PauliY(Vec<usize>),
    /// General Hermitian matrix
    Hermitian(Array2<Complex64>),
    /// Sum of observables
    Sum(Vec<Observable>),
    /// Product of observables
    Product(Vec<Observable>),
}
#[cfg(feature = "cuquantum")]
pub struct GpuBuffer {
    _ptr: *mut std::ffi::c_void,
    _size: usize,
}
#[cfg(feature = "cuquantum")]
pub struct CuStateVecHandle {
    _handle: *mut std::ffi::c_void,
}
/// Performance estimation results for a quantum circuit
#[derive(Debug, Clone)]
pub struct PerformanceEstimate {
    /// Estimated simulation time in milliseconds
    pub estimated_time_ms: f64,
    /// Estimated peak memory usage in bytes
    pub estimated_memory_bytes: usize,
    /// Estimated FLOPS required
    pub estimated_flops: f64,
    /// Recommended backend (state vector or tensor network)
    pub recommended_backend: RecommendedBackend,
    /// Whether the simulation will fit in GPU memory
    pub fits_in_memory: bool,
    /// Estimated GPU utilization (0.0 to 1.0)
    pub estimated_gpu_utilization: f64,
    /// Warnings or suggestions
    pub suggestions: Vec<String>,
}
/// Contraction path for tensor network
#[derive(Debug, Clone)]
pub struct ContractionPath {
    /// Sequence of contractions (pairs of tensor indices)
    pub contractions: Vec<(usize, usize)>,
    /// Estimated cost of each contraction
    pub costs: Vec<f64>,
    /// Slicing configuration
    pub slicing: Option<SlicingConfig>,
}
impl ContractionPath {
    /// Create empty path
    pub fn new() -> Self {
        Self {
            contractions: Vec::new(),
            costs: Vec::new(),
            slicing: None,
        }
    }
    /// Add a contraction step
    pub fn add_contraction(&mut self, tensor_a: usize, tensor_b: usize) {
        self.contractions.push((tensor_a, tensor_b));
        self.costs.push(1.0);
    }
    /// Get total cost
    pub fn total_cost(&self) -> f64 {
        self.costs.iter().sum()
    }
    /// Enable slicing for memory reduction
    pub fn enable_slicing(&mut self, memory_limit: usize) {
        self.slicing = Some(SlicingConfig {
            memory_limit,
            slice_indices: Vec::new(),
        });
    }
}
/// GPU performance estimator for quantum circuit simulation
#[derive(Debug)]
pub struct PerformanceEstimator {
    /// Device information
    device_info: CudaDeviceInfo,
    /// Configuration
    config: CuQuantumConfig,
}
impl PerformanceEstimator {
    /// Create a new performance estimator
    pub fn new(device_info: CudaDeviceInfo, config: CuQuantumConfig) -> Self {
        Self {
            device_info,
            config,
        }
    }
    /// Create with default device (mock on macOS)
    pub fn with_default_device(config: CuQuantumConfig) -> Result<Self> {
        let device_info = CuStateVecSimulator::get_device_info(config.device_id)?;
        Ok(Self::new(device_info, config))
    }
    /// Estimate performance for a quantum circuit
    pub fn estimate<const N: usize>(&self, circuit: &Circuit<N>) -> PerformanceEstimate {
        let num_qubits = N;
        let num_gates = circuit.gates().len();
        let state_vector_bytes = self.calculate_state_vector_memory(num_qubits);
        let estimated_flops = self.calculate_flops(num_qubits, num_gates);
        let fits_in_memory =
            state_vector_bytes <= (self.device_info.free_memory as f64 * 0.8) as usize;
        let recommended_backend = self.recommend_backend(num_qubits, num_gates, fits_in_memory);
        let estimated_time_ms = self.estimate_time(num_qubits, num_gates, &recommended_backend);
        let estimated_gpu_utilization =
            self.estimate_gpu_utilization(num_qubits, num_gates, &recommended_backend);
        let suggestions = self.generate_suggestions(num_qubits, num_gates, fits_in_memory);
        PerformanceEstimate {
            estimated_time_ms,
            estimated_memory_bytes: state_vector_bytes,
            estimated_flops,
            recommended_backend,
            fits_in_memory,
            estimated_gpu_utilization,
            suggestions,
        }
    }
    /// Calculate state vector memory requirements
    fn calculate_state_vector_memory(&self, num_qubits: usize) -> usize {
        let num_amplitudes: usize = 1 << num_qubits;
        num_amplitudes * self.config.precision.bytes_per_amplitude()
    }
    /// Calculate estimated FLOPS for simulation
    fn calculate_flops(&self, num_qubits: usize, num_gates: usize) -> f64 {
        let state_size = 1u64 << num_qubits;
        let flops_per_gate = state_size as f64 * 8.0;
        num_gates as f64 * flops_per_gate
    }
    /// Recommend the best backend for simulation
    fn recommend_backend(
        &self,
        num_qubits: usize,
        num_gates: usize,
        fits_in_memory: bool,
    ) -> RecommendedBackend {
        if !fits_in_memory {
            if num_qubits > 50 {
                RecommendedBackend::NotFeasible
            } else {
                RecommendedBackend::TensorNetwork
            }
        } else if num_qubits <= self.config.max_statevec_qubits {
            let circuit_depth = (num_gates as f64 / num_qubits as f64).ceil() as usize;
            if circuit_depth > num_qubits * 10 {
                RecommendedBackend::Hybrid
            } else {
                RecommendedBackend::StateVector
            }
        } else {
            RecommendedBackend::TensorNetwork
        }
    }
    /// Estimate simulation time
    fn estimate_time(
        &self,
        num_qubits: usize,
        num_gates: usize,
        backend: &RecommendedBackend,
    ) -> f64 {
        let base_flops = self.calculate_flops(num_qubits, num_gates);
        let gpu_throughput_gflops = match self.device_info.compute_capability {
            (9, _) => 150.0,
            (8, 9) => 83.0,
            (8, 6) => 35.0,
            (8, 0) => 19.5,
            (7, _) => 16.0,
            _ => 10.0,
        } * 1000.0;
        let raw_time_ms = base_flops / (gpu_throughput_gflops * 1e6);
        let overhead = match backend {
            RecommendedBackend::StateVector => 1.2,
            RecommendedBackend::TensorNetwork => 2.5,
            RecommendedBackend::Hybrid => 1.8,
            RecommendedBackend::NotFeasible => f64::MAX,
        };
        raw_time_ms * overhead
    }
    /// Estimate GPU utilization
    fn estimate_gpu_utilization(
        &self,
        num_qubits: usize,
        num_gates: usize,
        backend: &RecommendedBackend,
    ) -> f64 {
        match backend {
            RecommendedBackend::NotFeasible => 0.0,
            _ => {
                let size_factor = (num_qubits as f64 / 30.0).min(1.0);
                let gate_factor = (num_gates as f64 / 1000.0).min(1.0);
                (size_factor * 0.6 + gate_factor * 0.4).clamp(0.1, 0.95)
            }
        }
    }
    /// Generate performance suggestions
    fn generate_suggestions(
        &self,
        num_qubits: usize,
        num_gates: usize,
        fits_in_memory: bool,
    ) -> Vec<String> {
        let mut suggestions = Vec::new();
        if !fits_in_memory {
            suggestions
                .push(
                    format!(
                        "Circuit requires {} qubits, which exceeds available GPU memory. Consider using tensor network simulation.",
                        num_qubits
                    ),
                );
        }
        if num_qubits > 25 && self.config.gate_fusion_level != GateFusionLevel::Aggressive {
            suggestions.push(
                "Enable aggressive gate fusion for better performance on large circuits."
                    .to_string(),
            );
        }
        if num_gates > 10000 && !self.config.async_execution {
            suggestions.push("Enable async execution for circuits with many gates.".to_string());
        }
        if num_qubits > 28 && self.config.precision == ComputePrecision::Double {
            suggestions.push(
                "Consider using single precision for very large circuits to reduce memory usage."
                    .to_string(),
            );
        }
        if self.config.multi_gpu && num_qubits < 26 {
            suggestions
                .push(
                    "Multi-GPU mode is overkill for small circuits. Consider single GPU for better efficiency."
                        .to_string(),
                );
        }
        suggestions
    }
    /// Get device information
    pub fn device_info(&self) -> &CudaDeviceInfo {
        &self.device_info
    }
}
/// Slicing configuration for memory-efficient contraction
#[derive(Debug, Clone)]
pub struct SlicingConfig {
    /// Memory limit in bytes
    memory_limit: usize,
    /// Indices to slice over
    slice_indices: Vec<usize>,
}
/// Unified cuQuantum simulator that automatically selects the best backend
pub struct CuQuantumSimulator {
    /// cuStateVec simulator for state vector simulation
    pub statevec: Option<CuStateVecSimulator>,
    /// cuTensorNet simulator for tensor network simulation
    pub tensornet: Option<CuTensorNetSimulator>,
    /// Configuration
    pub config: CuQuantumConfig,
    /// Threshold for switching to tensor network (number of qubits)
    pub tensornet_threshold: usize,
}
impl CuQuantumSimulator {
    /// Create a new unified cuQuantum simulator
    pub fn new(config: CuQuantumConfig) -> Result<Self> {
        let tensornet_threshold = config.max_statevec_qubits;
        let statevec = CuStateVecSimulator::new(config.clone()).ok();
        let tensornet = CuTensorNetSimulator::new(config.clone()).ok();
        Ok(Self {
            statevec,
            tensornet,
            config,
            tensornet_threshold,
        })
    }
    /// Check if any cuQuantum backend is available
    pub fn is_available() -> bool {
        CuStateVecSimulator::is_available() || CuTensorNetSimulator::is_available()
    }
    /// Simulate a circuit, automatically selecting the best backend
    pub fn simulate<const N: usize>(&mut self, circuit: &Circuit<N>) -> Result<CuQuantumResult> {
        if N <= self.tensornet_threshold {
            if let Some(ref mut sv) = self.statevec {
                return sv.simulate(circuit);
            }
        }
        if let Some(ref mut tn) = self.tensornet {
            tn.build_network(circuit)?;
            let amplitudes = tn.contract(&(0..N).collect::<Vec<_>>())?;
            return Ok(CuQuantumResult::from_state_vector(amplitudes, N));
        }
        Err(SimulatorError::GpuError(
            "No cuQuantum backend available".to_string(),
        ))
    }
    /// Get combined statistics
    pub fn stats(&self) -> SimulationStats {
        let mut stats = SimulationStats::default();
        if let Some(ref sv) = self.statevec {
            let sv_stats = sv.stats();
            stats.total_simulations += sv_stats.total_simulations;
            stats.total_gates += sv_stats.total_gates;
            stats.total_time_ms += sv_stats.total_time_ms;
            stats.peak_memory_bytes = stats.peak_memory_bytes.max(sv_stats.peak_memory_bytes);
        }
        if let Some(ref tn) = self.tensornet {
            stats.tensor_contractions += tn.stats.tensor_contractions;
        }
        stats
    }
}
/// Gate fusion optimization level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateFusionLevel {
    /// No fusion
    None,
    /// Conservative fusion (adjacent single-qubit gates)
    Conservative,
    /// Moderate fusion (single-qubit + some two-qubit)
    Moderate,
    /// Aggressive fusion (maximize fusion opportunities)
    Aggressive,
}
/// GPU resource planner for multi-circuit simulation
#[derive(Debug)]
pub struct GpuResourcePlanner {
    /// Available devices
    devices: Vec<CudaDeviceInfo>,
    /// Configuration
    config: CuQuantumConfig,
}
impl GpuResourcePlanner {
    /// Create a new resource planner
    pub fn new(devices: Vec<CudaDeviceInfo>, config: CuQuantumConfig) -> Self {
        Self { devices, config }
    }
    /// Plan resource allocation for batch simulation
    pub fn plan_batch<const N: usize>(&self, circuits: &[Circuit<N>]) -> Vec<(usize, usize)> {
        if self.devices.is_empty() || circuits.is_empty() {
            return Vec::new();
        }
        let mut assignments = Vec::new();
        for (idx, _circuit) in circuits.iter().enumerate() {
            let device_idx = idx % self.devices.len();
            assignments.push((self.devices[device_idx].device_id as usize, idx));
        }
        assignments
    }
    /// Estimate total memory required for batch simulation
    pub fn estimate_batch_memory<const N: usize>(&self, circuits: &[Circuit<N>]) -> usize {
        let state_size: usize = 1 << N;
        state_size * self.config.precision.bytes_per_amplitude() * circuits.len()
    }
}
/// Circuit complexity analyzer
#[derive(Debug, Clone)]
pub struct CircuitComplexity {
    /// Number of qubits
    pub num_qubits: usize,
    /// Total number of gates
    pub num_gates: usize,
    /// Number of single-qubit gates
    pub single_qubit_gates: usize,
    /// Number of two-qubit gates
    pub two_qubit_gates: usize,
    /// Number of multi-qubit gates (3+)
    pub multi_qubit_gates: usize,
    /// Circuit depth
    pub depth: usize,
    /// Estimated entanglement degree (0.0 to 1.0)
    pub entanglement_degree: f64,
    /// Gate types used
    pub gate_types: Vec<String>,
}
impl CircuitComplexity {
    /// Analyze a quantum circuit
    pub fn analyze<const N: usize>(circuit: &Circuit<N>) -> Self {
        let mut single_qubit_gates = 0;
        let mut two_qubit_gates = 0;
        let mut multi_qubit_gates = 0;
        let mut gate_types = std::collections::HashSet::new();
        for gate in circuit.gates() {
            let num_qubits_affected = gate.qubits().len();
            match num_qubits_affected {
                1 => single_qubit_gates += 1,
                2 => two_qubit_gates += 1,
                _ => multi_qubit_gates += 1,
            }
            gate_types.insert(gate.name().to_string());
        }
        let depth = if N > 0 {
            (circuit.gates().len() as f64 / N as f64).ceil() as usize
        } else {
            0
        };
        let total_gates = circuit.gates().len();
        let entanglement_degree = if total_gates > 0 {
            (two_qubit_gates + multi_qubit_gates * 2) as f64 / total_gates as f64
        } else {
            0.0
        };
        Self {
            num_qubits: N,
            num_gates: total_gates,
            single_qubit_gates,
            two_qubit_gates,
            multi_qubit_gates,
            depth,
            entanglement_degree,
            gate_types: gate_types.into_iter().collect(),
        }
    }
}
/// Tensor network contraction algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorContractionAlgorithm {
    /// Automatic selection based on circuit structure
    Auto,
    /// Greedy contraction order
    Greedy,
    /// Optimal contraction order (may be expensive for large circuits)
    Optimal,
    /// Optimal with index slicing for memory reduction
    OptimalWithSlicing,
    /// Random greedy trials
    RandomGreedy,
}
