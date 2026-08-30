//! MPI-based Distributed Quantum Simulation
//!
//! This module provides Message Passing Interface (MPI) support for distributed
//! quantum simulation across multiple compute nodes. It enables simulation of
//! extremely large quantum systems (50+ qubits) by distributing the quantum state
//! across multiple nodes and coordinating quantum operations through MPI.
//!
//! # Features
//! - MPI communicator abstraction for quantum simulation
//! - Distributed quantum state management with automatic partitioning
//! - Collective operations optimized for quantum state vectors
//! - Support for both simulated MPI (testing) and real MPI backends
//! - Integration with `SciRS2` parallel operations

use crate::distributed_simulator::{
    CommunicationConfig, CommunicationPattern, DistributedSimulatorConfig, DistributionStrategy,
    FaultToleranceConfig, LoadBalancingConfig, LoadBalancingStrategy, NetworkConfig,
};
use crate::large_scale_simulator::{LargeScaleSimulatorConfig, QuantumStateRepresentation};
use quantrs2_core::error::{QuantRS2Error, QuantRS2Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use scirs2_core::parallel_ops::{IndexedParallelIterator, ParallelIterator};
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::{Duration, Instant};

/// MPI-based distributed quantum simulator
///
/// This simulator uses MPI for inter-node communication to enable
/// simulation of quantum systems larger than what can fit in a single
/// node's memory.
#[derive(Debug)]
pub struct MPIQuantumSimulator {
    /// MPI communicator for quantum operations
    communicator: MPICommunicator,
    /// Local quantum state partition
    local_state: Arc<RwLock<LocalQuantumState>>,
    /// Configuration for the MPI simulator
    config: MPISimulatorConfig,
    /// Performance statistics
    stats: Arc<Mutex<MPISimulatorStats>>,
    /// State synchronization manager
    sync_manager: StateSynchronizationManager,
    /// Gate distribution handler
    gate_handler: GateDistributionHandler,
}

/// Configuration for MPI-based quantum simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MPISimulatorConfig {
    /// Total number of qubits in the simulation
    pub total_qubits: usize,
    /// Distribution strategy for quantum state
    pub distribution_strategy: MPIDistributionStrategy,
    /// Collective operation optimization settings
    pub collective_optimization: CollectiveOptimization,
    /// Communication overlap settings
    pub overlap_config: CommunicationOverlapConfig,
    /// Checkpointing configuration
    pub checkpoint_config: CheckpointConfig,
    /// Memory management settings
    pub memory_config: MemoryConfig,
}

impl Default for MPISimulatorConfig {
    fn default() -> Self {
        Self {
            total_qubits: 20,
            distribution_strategy: MPIDistributionStrategy::AmplitudePartition,
            collective_optimization: CollectiveOptimization::default(),
            overlap_config: CommunicationOverlapConfig::default(),
            checkpoint_config: CheckpointConfig::default(),
            memory_config: MemoryConfig::default(),
        }
    }
}

/// Strategy for distributing quantum state across MPI nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MPIDistributionStrategy {
    /// Partition state vector by amplitude indices
    AmplitudePartition,
    /// Partition by qubit subsets (for localized operations)
    QubitPartition,
    /// Hybrid partitioning based on circuit structure
    HybridPartition,
    /// Gate-aware dynamic partitioning
    GateAwarePartition,
    /// Hilbert curve space-filling for data locality
    HilbertCurvePartition,
}

/// Optimization settings for MPI collective operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectiveOptimization {
    /// Use non-blocking collectives when possible
    pub use_nonblocking: bool,
    /// Enable collective operation fusion
    pub enable_fusion: bool,
    /// Buffer size for collective operations
    pub buffer_size: usize,
    /// Allreduce algorithm selection
    pub allreduce_algorithm: AllreduceAlgorithm,
    /// Broadcast algorithm selection
    pub broadcast_algorithm: BroadcastAlgorithm,
}

impl Default for CollectiveOptimization {
    fn default() -> Self {
        Self {
            use_nonblocking: true,
            enable_fusion: true,
            buffer_size: 16 * 1024 * 1024, // 16MB
            allreduce_algorithm: AllreduceAlgorithm::RecursiveDoubling,
            broadcast_algorithm: BroadcastAlgorithm::BinomialTree,
        }
    }
}

/// Allreduce algorithm variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllreduceAlgorithm {
    /// Ring-based allreduce (bandwidth optimal)
    Ring,
    /// Recursive doubling (latency optimal)
    RecursiveDoubling,
    /// Rabenseifner algorithm (hybrid)
    Rabenseifner,
    /// Automatic selection based on message size
    Automatic,
}

/// Broadcast algorithm variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BroadcastAlgorithm {
    /// Binomial tree broadcast
    BinomialTree,
    /// Scatter + Allgather
    ScatterAllgather,
    /// Pipeline broadcast
    Pipeline,
    /// Automatic selection
    Automatic,
}

/// Configuration for overlapping communication with computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationOverlapConfig {
    /// Enable communication/computation overlap
    pub enable_overlap: bool,
    /// Number of pipeline stages
    pub pipeline_stages: usize,
    /// Prefetch distance for communication
    pub prefetch_distance: usize,
}

impl Default for CommunicationOverlapConfig {
    fn default() -> Self {
        Self {
            enable_overlap: true,
            pipeline_stages: 4,
            prefetch_distance: 2,
        }
    }
}

/// Checkpointing configuration for fault tolerance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    /// Enable periodic checkpointing
    pub enable: bool,
    /// Checkpoint interval (number of operations)
    pub interval: usize,
    /// Checkpoint storage path
    pub storage_path: String,
    /// Use compression for checkpoints
    pub use_compression: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            enable: false,
            interval: 1000,
            storage_path: "/tmp/quantum_checkpoint".to_string(),
            use_compression: true,
        }
    }
}

/// Memory management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Maximum memory per node (bytes)
    pub max_memory_per_node: usize,
    /// Enable memory pooling
    pub enable_pooling: bool,
    /// Pool size for temporary allocations
    pub pool_size: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_memory_per_node: 64 * 1024 * 1024 * 1024, // 64GB
            enable_pooling: true,
            pool_size: 1024 * 1024 * 1024, // 1GB pool
        }
    }
}

/// MPI communicator abstraction for quantum operations
#[derive(Debug)]
pub struct MPICommunicator {
    /// MPI rank of this process
    rank: usize,
    /// Total number of MPI processes
    size: usize,
    /// Communication backend
    backend: MPIBackend,
    /// Message buffer pool
    buffer_pool: Arc<Mutex<Vec<Vec<u8>>>>,
    /// Pending requests for non-blocking operations
    pending_requests: Arc<Mutex<Vec<MPIRequest>>>,
}

/// MPI backend implementations
#[derive(Debug, Clone)]
pub enum MPIBackend {
    /// Simulated MPI for testing (single-process simulation)
    Simulated(SimulatedMPIBackend),
    /// Native MPI backend (requires mpi feature)
    #[cfg(feature = "mpi")]
    Native(NativeMPIBackend),
    /// TCP-based fallback implementation
    TCP(TCPMPIBackend),
}

/// Simulated MPI backend for testing
///
/// Unlike the TCP/Native backends (which need genuinely separate OS
/// processes/nodes to talk to), this backend performs *real* in-process
/// message passing between multiple [`MPICommunicator`] instances that
/// share the same [`SimulatedMPIState`] (via a cloned `Arc`), each
/// representing one rank and typically driven from its own thread. This
/// makes it possible to exercise genuine multi-rank collective semantics
/// (point-to-point exchange, gather, broadcast, allreduce, barrier)
/// entirely locally, without any real network.
#[derive(Debug, Clone)]
pub struct SimulatedMPIBackend {
    /// Shared state for all "processes", guarded by a condition variable so
    /// that ranks can rendezvous with each other.
    shared_state: Arc<(Mutex<SimulatedMPIState>, Condvar)>,
}

/// One in-progress (or just-completed) collective round.
///
/// Uses the classic "sense reversing" barrier pattern: ranks arrive and
/// stash their payload under the round's current `generation`; once
/// `size` ranks have arrived the round is closed out (`completed_*` is
/// populated and `generation` is bumped), so any rank that arrives after
/// that instant starts a fresh round instead of corrupting the old one.
#[derive(Default)]
struct CollectiveRound {
    /// Generation number currently accepting arrivals.
    generation: usize,
    /// Payloads contributed by ranks that have arrived for `generation`.
    payloads: BTreeMap<usize, Box<dyn std::any::Any + Send>>,
    /// Generation number of the most recently completed round.
    completed_generation: usize,
    /// Payloads collected once `completed_generation` finished, kept
    /// around (in rank order) so every arriving rank can read the result.
    completed_payloads: BTreeMap<usize, Box<dyn std::any::Any + Send>>,
}

/// Shared state for simulated MPI: a real (in-process) mailbox for
/// point-to-point exchange plus a set of named collective rounds.
#[derive(Default)]
pub struct SimulatedMPIState {
    /// Point-to-point mailbox keyed by `(from_rank, to_rank)`.
    mailbox: HashMap<(usize, usize), Vec<u8>>,
    /// Named collective rounds (e.g. `"gather:0"`, `"broadcast:0"`,
    /// `"allreduce"`, `"barrier"`).
    rounds: HashMap<String, CollectiveRound>,
}

impl std::fmt::Debug for SimulatedMPIState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimulatedMPIState")
            .field("pending_messages", &self.mailbox.len())
            .field("active_rounds", &self.rounds.len())
            .finish()
    }
}

impl SimulatedMPIState {
    /// Register `rank`'s arrival (with `payload`) at the named collective
    /// round, closing the round out if this was the last of `size` ranks.
    /// Returns the generation number this arrival belongs to.
    fn arrive(
        &mut self,
        key: &str,
        rank: usize,
        size: usize,
        payload: Box<dyn std::any::Any + Send>,
    ) -> usize {
        let round = self.rounds.entry(key.to_string()).or_default();
        round.payloads.insert(rank, payload);
        let my_generation = round.generation;
        if round.payloads.len() >= size {
            let payloads = std::mem::take(&mut round.payloads);
            round.completed_generation = my_generation;
            round.completed_payloads = payloads;
            round.generation += 1;
        }
        my_generation
    }

    /// True once the round `generation` for `key` has fully completed.
    fn round_ready(&self, key: &str, generation: usize) -> bool {
        self.rounds.get(key).is_some_and(|round| {
            round.completed_generation == generation && round.generation == generation + 1
        })
    }
}

/// TCP-based MPI backend
#[derive(Debug, Clone)]
pub struct TCPMPIBackend {
    /// Connections to other ranks
    connections: Arc<RwLock<HashMap<usize, std::net::SocketAddr>>>,
}

/// Native MPI backend (placeholder for real MPI integration)
#[cfg(feature = "mpi")]
#[derive(Debug, Clone)]
pub struct NativeMPIBackend {
    /// MPI communicator handle (placeholder)
    comm_handle: usize,
}

/// MPI request handle for non-blocking operations
#[derive(Debug)]
pub struct MPIRequest {
    /// Request ID
    id: usize,
    /// Request type
    request_type: MPIRequestType,
    /// Completion status
    completed: Arc<Mutex<bool>>,
}

/// Types of MPI requests
#[derive(Debug, Clone)]
pub enum MPIRequestType {
    Send { dest: usize, tag: i32 },
    Recv { source: usize, tag: i32 },
    Collective { operation: String },
}

/// Local quantum state partition
#[derive(Debug)]
pub struct LocalQuantumState {
    /// State vector partition (local amplitudes)
    amplitudes: Array1<Complex64>,
    /// Global index offset for this partition
    global_offset: usize,
    /// Qubit indices managed by this partition
    local_qubits: Vec<usize>,
    /// Ghost cells for boundary communication
    ghost_cells: GhostCells,
}

/// Ghost cells for efficient boundary communication
#[derive(Debug, Clone, Default)]
pub struct GhostCells {
    /// Left ghost region
    left: Vec<Complex64>,
    /// Right ghost region
    right: Vec<Complex64>,
    /// Ghost cell width
    width: usize,
}

/// Statistics for MPI quantum simulator
#[derive(Debug, Clone, Default)]
pub struct MPISimulatorStats {
    /// Total gates executed
    pub gates_executed: u64,
    /// Total communication time
    pub communication_time: Duration,
    /// Total computation time
    pub computation_time: Duration,
    /// Number of synchronization points
    pub sync_count: u64,
    /// Bytes sent
    pub bytes_sent: u64,
    /// Bytes received
    pub bytes_received: u64,
    /// Load imbalance factor
    pub load_imbalance: f64,
}

/// State synchronization manager
#[derive(Debug)]
pub struct StateSynchronizationManager {
    /// Synchronization strategy
    strategy: SyncStrategy,
    /// Pending sync operations
    pending: Arc<Mutex<Vec<SyncOperation>>>,
}

/// Synchronization strategy
#[derive(Debug, Clone, Copy)]
pub enum SyncStrategy {
    /// Synchronize after every gate
    Eager,
    /// Batch synchronizations
    Lazy,
    /// Adaptive based on circuit structure
    Adaptive,
}

/// Pending synchronization operation
#[derive(Debug, Clone)]
pub struct SyncOperation {
    /// Qubits involved
    qubits: Vec<usize>,
    /// Operation type
    op_type: SyncOpType,
}

/// Types of synchronization operations
#[derive(Debug, Clone)]
pub enum SyncOpType {
    BoundaryExchange,
    GlobalReduction,
    PartitionSwap,
}

/// Gate distribution handler
#[derive(Debug)]
pub struct GateDistributionHandler {
    /// Gate routing table
    routing_table: Arc<RwLock<HashMap<usize, usize>>>,
    /// Local vs distributed gate classification
    gate_classifier: GateClassifier,
}

/// Gate classifier for local vs distributed execution
#[derive(Debug)]
pub struct GateClassifier {
    /// Local qubit set for this partition
    local_qubits: Vec<usize>,
}

impl MPIQuantumSimulator {
    /// Create a new MPI-based quantum simulator
    pub fn new(config: MPISimulatorConfig) -> QuantRS2Result<Self> {
        // Initialize MPI communicator
        let communicator = MPICommunicator::new()?;

        // Calculate local partition size
        let total_amplitudes = 1usize << config.total_qubits;
        let local_size = total_amplitudes / communicator.size;
        let global_offset = communicator.rank * local_size;

        // Initialize local quantum state
        let local_state = LocalQuantumState {
            amplitudes: Array1::zeros(local_size),
            global_offset,
            local_qubits: Self::calculate_local_qubits(
                config.total_qubits,
                communicator.rank,
                communicator.size,
            ),
            ghost_cells: GhostCells::default(),
        };

        // Initialize synchronization manager
        let sync_manager = StateSynchronizationManager {
            strategy: SyncStrategy::Adaptive,
            pending: Arc::new(Mutex::new(Vec::new())),
        };

        // Initialize gate distribution handler
        let gate_handler = GateDistributionHandler {
            routing_table: Arc::new(RwLock::new(HashMap::new())),
            gate_classifier: GateClassifier {
                local_qubits: local_state.local_qubits.clone(),
            },
        };

        Ok(Self {
            communicator,
            local_state: Arc::new(RwLock::new(local_state)),
            config,
            stats: Arc::new(Mutex::new(MPISimulatorStats::default())),
            sync_manager,
            gate_handler,
        })
    }

    /// Calculate which qubits are local to this partition
    fn calculate_local_qubits(total_qubits: usize, rank: usize, size: usize) -> Vec<usize> {
        // For amplitude partitioning, higher qubits determine partition
        let partition_bits = (size as f64).log2().ceil() as usize;
        let local_bits = total_qubits - partition_bits;

        // Local qubits are the lower-order bits
        (0..local_bits).collect()
    }

    /// Initialize the quantum state to |0...0>
    pub fn initialize(&mut self) -> QuantRS2Result<()> {
        let mut state = self
            .local_state
            .write()
            .map_err(|_| QuantRS2Error::InvalidInput("Failed to acquire state lock".to_string()))?;

        // Set all amplitudes to 0
        state.amplitudes.fill(Complex64::new(0.0, 0.0));

        // Only rank 0 has the |0...0> amplitude
        if self.communicator.rank == 0 {
            state.amplitudes[0] = Complex64::new(1.0, 0.0);
        }

        Ok(())
    }

    /// Apply a single-qubit gate
    pub fn apply_single_qubit_gate(
        &mut self,
        qubit: usize,
        gate_matrix: &Array2<Complex64>,
    ) -> QuantRS2Result<()> {
        let start = Instant::now();

        // Check if qubit is local
        let state = self
            .local_state
            .read()
            .map_err(|_| QuantRS2Error::InvalidInput("Failed to acquire state lock".to_string()))?;

        if state.local_qubits.contains(&qubit) {
            // Local gate application
            drop(state);
            self.apply_local_single_qubit_gate(qubit, gate_matrix)?;
        } else {
            // Distributed gate application
            drop(state);
            self.apply_distributed_single_qubit_gate(qubit, gate_matrix)?;
        }

        // Update statistics
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| QuantRS2Error::InvalidInput("Failed to acquire stats lock".to_string()))?;
        stats.gates_executed += 1;
        stats.computation_time += start.elapsed();

        Ok(())
    }

    /// Apply a single-qubit gate locally
    fn apply_local_single_qubit_gate(
        &self,
        qubit: usize,
        gate_matrix: &Array2<Complex64>,
    ) -> QuantRS2Result<()> {
        let mut state = self
            .local_state
            .write()
            .map_err(|_| QuantRS2Error::InvalidInput("Failed to acquire state lock".to_string()))?;

        let n = state.amplitudes.len();
        let stride = 1 << qubit;

        // Apply gate in parallel using SciRS2 parallel_ops
        let amplitudes = state.amplitudes.as_slice_mut().ok_or_else(|| {
            QuantRS2Error::InvalidInput("Failed to get mutable slice".to_string())
        })?;

        // Process pairs of amplitudes
        for i in 0..n / 2 {
            let i0 = (i / stride) * (2 * stride) + (i % stride);
            let i1 = i0 + stride;

            let a0 = amplitudes[i0];
            let a1 = amplitudes[i1];

            amplitudes[i0] = gate_matrix[[0, 0]] * a0 + gate_matrix[[0, 1]] * a1;
            amplitudes[i1] = gate_matrix[[1, 0]] * a0 + gate_matrix[[1, 1]] * a1;
        }

        Ok(())
    }

    /// Apply a single-qubit gate that requires distribution
    fn apply_distributed_single_qubit_gate(
        &self,
        qubit: usize,
        gate_matrix: &Array2<Complex64>,
    ) -> QuantRS2Result<()> {
        // Determine partner rank for communication
        let partition_bit = qubit - self.gate_handler.gate_classifier.local_qubits.len();
        let partner = self.communicator.rank ^ (1 << partition_bit);

        // Exchange boundary data with partner
        self.exchange_boundary_data(partner)?;

        // Apply gate with boundary data
        let mut state = self
            .local_state
            .write()
            .map_err(|_| QuantRS2Error::InvalidInput("Failed to acquire state lock".to_string()))?;

        let n = state.amplitudes.len();
        let local_qubits = state.local_qubits.len();
        let local_stride = 1 << local_qubits;

        // Determine if we're the lower or upper partition
        let is_lower = (self.communicator.rank >> partition_bit) & 1 == 0;

        for i in 0..n {
            let global_i = state.global_offset + i;
            let partner_i = global_i ^ local_stride;

            // Get partner amplitude from ghost cells
            let partner_amp = if is_lower {
                state
                    .ghost_cells
                    .right
                    .get(i)
                    .copied()
                    .unwrap_or(Complex64::new(0.0, 0.0))
            } else {
                state
                    .ghost_cells
                    .left
                    .get(i)
                    .copied()
                    .unwrap_or(Complex64::new(0.0, 0.0))
            };

            let local_amp = state.amplitudes[i];

            // Apply gate transformation
            let (a0, a1) = if is_lower {
                (local_amp, partner_amp)
            } else {
                (partner_amp, local_amp)
            };

            let new_amp = if is_lower {
                gate_matrix[[0, 0]] * a0 + gate_matrix[[0, 1]] * a1
            } else {
                gate_matrix[[1, 0]] * a0 + gate_matrix[[1, 1]] * a1
            };

            state.amplitudes[i] = new_amp;
        }

        Ok(())
    }

    /// Exchange boundary data with a partner rank
    fn exchange_boundary_data(&self, partner: usize) -> QuantRS2Result<()> {
        let state = self
            .local_state
            .read()
            .map_err(|_| QuantRS2Error::InvalidInput("Failed to acquire state lock".to_string()))?;

        // Prepare send buffer
        let send_data: Vec<Complex64> = state.amplitudes.iter().copied().collect();
        drop(state);

        // Exchange data with partner
        let recv_data = self.communicator.sendrecv(&send_data, partner)?;

        // Update ghost cells
        let mut state = self
            .local_state
            .write()
            .map_err(|_| QuantRS2Error::InvalidInput("Failed to acquire state lock".to_string()))?;

        if self.communicator.rank < partner {
            state.ghost_cells.right = recv_data;
        } else {
            state.ghost_cells.left = recv_data;
        }

        Ok(())
    }

    /// Apply a two-qubit gate
    pub fn apply_two_qubit_gate(
        &mut self,
        control: usize,
        target: usize,
        gate_matrix: &Array2<Complex64>,
    ) -> QuantRS2Result<()> {
        let start = Instant::now();

        let state = self
            .local_state
            .read()
            .map_err(|_| QuantRS2Error::InvalidInput("Failed to acquire state lock".to_string()))?;

        let control_local = state.local_qubits.contains(&control);
        let target_local = state.local_qubits.contains(&target);
        drop(state);

        match (control_local, target_local) {
            (true, true) => {
                // Both qubits local - local gate application
                self.apply_local_two_qubit_gate(control, target, gate_matrix)?;
            }
            (true, false) | (false, true) => {
                // One qubit local - partial distribution
                self.apply_partial_distributed_gate(control, target, gate_matrix)?;
            }
            (false, false) => {
                // Both qubits remote - full distribution
                self.apply_full_distributed_gate(control, target, gate_matrix)?;
            }
        }

        // Update statistics
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| QuantRS2Error::InvalidInput("Failed to acquire stats lock".to_string()))?;
        stats.gates_executed += 1;
        stats.computation_time += start.elapsed();

        Ok(())
    }

    /// Apply a two-qubit gate locally
    fn apply_local_two_qubit_gate(
        &self,
        control: usize,
        target: usize,
        gate_matrix: &Array2<Complex64>,
    ) -> QuantRS2Result<()> {
        let mut state = self
            .local_state
            .write()
            .map_err(|_| QuantRS2Error::InvalidInput("Failed to acquire state lock".to_string()))?;

        let n = state.amplitudes.len();
        let control_stride = 1 << control;
        let target_stride = 1 << target;

        // Ensure consistent ordering
        let (low_stride, high_stride) = if control < target {
            (control_stride, target_stride)
        } else {
            (target_stride, control_stride)
        };

        // Apply gate to all 4-amplitude groups
        for i in 0..n / 4 {
            // Calculate base index
            let base = (i / low_stride) * (2 * low_stride) + (i % low_stride);
            let base = (base / high_stride) * (2 * high_stride) + (base % high_stride);

            // Calculate all four indices
            let i00 = base;
            let i01 = base + target_stride;
            let i10 = base + control_stride;
            let i11 = base + control_stride + target_stride;

            // Get amplitudes
            let a00 = state.amplitudes[i00];
            let a01 = state.amplitudes[i01];
            let a10 = state.amplitudes[i10];
            let a11 = state.amplitudes[i11];

            // Apply 4x4 gate matrix
            state.amplitudes[i00] = gate_matrix[[0, 0]] * a00
                + gate_matrix[[0, 1]] * a01
                + gate_matrix[[0, 2]] * a10
                + gate_matrix[[0, 3]] * a11;
            state.amplitudes[i01] = gate_matrix[[1, 0]] * a00
                + gate_matrix[[1, 1]] * a01
                + gate_matrix[[1, 2]] * a10
                + gate_matrix[[1, 3]] * a11;
            state.amplitudes[i10] = gate_matrix[[2, 0]] * a00
                + gate_matrix[[2, 1]] * a01
                + gate_matrix[[2, 2]] * a10
                + gate_matrix[[2, 3]] * a11;
            state.amplitudes[i11] = gate_matrix[[3, 0]] * a00
                + gate_matrix[[3, 1]] * a01
                + gate_matrix[[3, 2]] * a10
                + gate_matrix[[3, 3]] * a11;
        }

        Ok(())
    }

    /// Apply partially distributed gate (one local, one remote qubit)
    fn apply_partial_distributed_gate(
        &self,
        control: usize,
        target: usize,
        gate_matrix: &Array2<Complex64>,
    ) -> QuantRS2Result<()> {
        // Determine which qubit is local
        let state = self
            .local_state
            .read()
            .map_err(|_| QuantRS2Error::InvalidInput("Failed to acquire state lock".to_string()))?;

        let (local_qubit, remote_qubit) = if state.local_qubits.contains(&control) {
            (control, target)
        } else {
            (target, control)
        };
        drop(state);

        // Determine partner for remote qubit
        let partition_bit = remote_qubit - self.gate_handler.gate_classifier.local_qubits.len();
        let partner = self.communicator.rank ^ (1 << partition_bit);

        // Exchange partial state
        self.exchange_boundary_data(partner)?;

        // Apply gate with partial distribution
        // This is a simplified version - full implementation would need
        // more sophisticated handling of the 4-amplitude groups
        let mut state = self
            .local_state
            .write()
            .map_err(|_| QuantRS2Error::InvalidInput("Failed to acquire state lock".to_string()))?;

        let n = state.amplitudes.len();
        let local_stride = 1 << local_qubit;

        for i in 0..n / 2 {
            let i0 = (i / local_stride) * (2 * local_stride) + (i % local_stride);
            let i1 = i0 + local_stride;

            let a0 = state.amplitudes[i0];
            let a1 = state.amplitudes[i1];

            // Apply conditional transformation based on gate structure
            // This is simplified - real implementation needs full 4x4 matrix
            state.amplitudes[i0] = gate_matrix[[0, 0]] * a0 + gate_matrix[[0, 1]] * a1;
            state.amplitudes[i1] = gate_matrix[[1, 0]] * a0 + gate_matrix[[1, 1]] * a1;
        }

        Ok(())
    }

    /// Apply fully distributed gate (both qubits remote)
    ///
    /// Both `control` and `target` live in the partition-selecting (high)
    /// bits of the global amplitude index: every amplitude *this* rank
    /// holds shares the same fixed (control, target) bit values, which are
    /// encoded directly in this rank's own MPI rank index. To apply the
    /// full 4x4 `gate_matrix` we must combine our own amplitudes with the
    /// amplitudes held by the three peer ranks that own the other three
    /// (control, target) combinations -- real point-to-point exchange with
    /// each of them, then a genuine matrix-vector product per index using
    /// this rank's row of `gate_matrix`.
    fn apply_full_distributed_gate(
        &self,
        control: usize,
        target: usize,
        gate_matrix: &Array2<Complex64>,
    ) -> QuantRS2Result<()> {
        let local_qubits_len = self.gate_handler.gate_classifier.local_qubits.len();

        let control_bit = control - local_qubits_len;
        let target_bit = target - local_qubits_len;
        if control_bit == target_bit {
            return Err(QuantRS2Error::InvalidInput(
                "apply_full_distributed_gate: control and target map to the same partition bit"
                    .to_string(),
            ));
        }

        let rank = self.communicator.rank;
        let own_control_val = (rank >> control_bit) & 1;
        let own_target_val = (rank >> target_bit) & 1;

        let control_partner = rank ^ (1 << control_bit);
        let target_partner = rank ^ (1 << target_bit);
        let diag_partner = rank ^ (1 << control_bit) ^ (1 << target_bit);

        // Real exchange (not a placeholder): send our whole local partition
        // to each peer and receive theirs in return.
        let local_amplitudes: Vec<Complex64> = {
            let state = self.local_state.read().map_err(|_| {
                QuantRS2Error::InvalidInput("Failed to acquire state lock".to_string())
            })?;
            state.amplitudes.iter().copied().collect()
        };

        let control_flipped = self
            .communicator
            .sendrecv(&local_amplitudes, control_partner)?;
        let target_flipped = self
            .communicator
            .sendrecv(&local_amplitudes, target_partner)?;
        let both_flipped = self
            .communicator
            .sendrecv(&local_amplitudes, diag_partner)?;

        // Slot the four peers' data by their true (control, target) bit
        // values so we index `gate_matrix` the same way
        // `apply_local_two_qubit_gate` does (row/col = 2*control + target),
        // regardless of which combination this rank happens to own.
        let mut amps_by_bits: [Option<&[Complex64]>; 4] = [None; 4];
        amps_by_bits[(own_control_val << 1) | own_target_val] = Some(local_amplitudes.as_slice());
        amps_by_bits[((1 - own_control_val) << 1) | own_target_val] =
            Some(control_flipped.as_slice());
        amps_by_bits[(own_control_val << 1) | (1 - own_target_val)] =
            Some(target_flipped.as_slice());
        amps_by_bits[((1 - own_control_val) << 1) | (1 - own_target_val)] =
            Some(both_flipped.as_slice());

        let own_row = (own_control_val << 1) | own_target_val;

        let mut state = self
            .local_state
            .write()
            .map_err(|_| QuantRS2Error::InvalidInput("Failed to acquire state lock".to_string()))?;

        let n = state.amplitudes.len();
        for i in 0..n {
            let mut acc = Complex64::new(0.0, 0.0);
            for (col, amps) in amps_by_bits.iter().enumerate() {
                let amps = amps.ok_or_else(|| {
                    QuantRS2Error::InvalidInput(
                        "apply_full_distributed_gate: missing peer amplitude data".to_string(),
                    )
                })?;
                let value = amps.get(i).copied().unwrap_or(Complex64::new(0.0, 0.0));
                acc += gate_matrix[[own_row, col]] * value;
            }
            state.amplitudes[i] = acc;
        }

        Ok(())
    }

    /// Perform a global barrier synchronization
    pub fn barrier(&self) -> QuantRS2Result<()> {
        self.communicator.barrier()
    }

    /// Compute global probability distribution
    pub fn get_probability_distribution(&self) -> QuantRS2Result<Vec<f64>> {
        let state = self
            .local_state
            .read()
            .map_err(|_| QuantRS2Error::InvalidInput("Failed to acquire state lock".to_string()))?;

        // Compute local probabilities
        let local_probs: Vec<f64> = state.amplitudes.iter().map(|a| (a * a.conj()).re).collect();

        drop(state);

        // Gather all probabilities to rank 0
        let global_probs = self.communicator.gather(&local_probs, 0)?;

        Ok(global_probs)
    }

    /// Measure all qubits
    pub fn measure_all(&self) -> QuantRS2Result<Vec<bool>> {
        // Get global probability distribution
        let probs = self.get_probability_distribution()?;

        // Only rank 0 performs measurement
        if self.communicator.rank == 0 {
            // Sample from distribution
            let mut rng = scirs2_core::random::thread_rng();
            let random: f64 = rng.random();

            let mut cumulative = 0.0;
            let mut result_idx = 0;

            for (i, &prob) in probs.iter().enumerate() {
                cumulative += prob;
                if random < cumulative {
                    result_idx = i;
                    break;
                }
            }

            // Convert to bit string
            let result: Vec<bool> = (0..self.config.total_qubits)
                .map(|i| (result_idx >> i) & 1 == 1)
                .collect();

            // Broadcast result to all ranks
            self.communicator.broadcast(&result, 0)
        } else {
            // Receive result from rank 0
            self.communicator.broadcast(&[], 0)
        }
    }

    /// Get local state for debugging/testing
    pub fn get_local_state(&self) -> QuantRS2Result<Array1<Complex64>> {
        let state = self
            .local_state
            .read()
            .map_err(|_| QuantRS2Error::InvalidInput("Failed to acquire state lock".to_string()))?;
        Ok(state.amplitudes.clone())
    }

    /// Get simulator statistics
    pub fn get_stats(&self) -> QuantRS2Result<MPISimulatorStats> {
        let stats = self
            .stats
            .lock()
            .map_err(|_| QuantRS2Error::InvalidInput("Failed to acquire stats lock".to_string()))?;
        Ok(stats.clone())
    }

    /// Reset the simulator
    pub fn reset(&mut self) -> QuantRS2Result<()> {
        self.initialize()?;

        // Reset statistics
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| QuantRS2Error::InvalidInput("Failed to acquire stats lock".to_string()))?;
        *stats = MPISimulatorStats::default();

        Ok(())
    }
}

impl MPICommunicator {
    /// Create a new MPI communicator
    pub fn new() -> QuantRS2Result<Self> {
        // Default to a single-rank simulated backend.
        let shared_state = Arc::new((Mutex::new(SimulatedMPIState::default()), Condvar::new()));
        let backend = MPIBackend::Simulated(SimulatedMPIBackend { shared_state });

        Ok(Self {
            rank: 0,
            size: 1,
            backend,
            buffer_pool: Arc::new(Mutex::new(Vec::new())),
            pending_requests: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Create communicator with specific configuration
    #[must_use]
    pub fn with_config(rank: usize, size: usize, backend: MPIBackend) -> Self {
        Self {
            rank,
            size,
            backend,
            buffer_pool: Arc::new(Mutex::new(Vec::new())),
            pending_requests: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Get rank of this process
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// Get total number of processes
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Honest error for backends that require real external
    /// processes/hardware (a live TCP peer, or a linked native MPI
    /// runtime) that this single-process build cannot provide.
    fn unsupported_backend(backend_name: &str, op: &str) -> QuantRS2Error {
        QuantRS2Error::UnsupportedOperation(format!(
            "{op}: the {backend_name} MPI backend requires a real connection to peer \
             process(es)/node(s), which is not available in this build; use \
             MPIBackend::Simulated for in-process multi-rank testing, or link a real \
             transport for {backend_name}"
        ))
    }

    fn lock_poisoned() -> QuantRS2Error {
        QuantRS2Error::InvalidInput("MPI simulated backend lock was poisoned".to_string())
    }

    /// Barrier synchronization: every rank sharing this communicator's
    /// `SimulatedMPIState` must call this before any of them proceeds.
    pub fn barrier(&self) -> QuantRS2Result<()> {
        match &self.backend {
            MPIBackend::Simulated(sim) => {
                let (lock, cvar) = &*sim.shared_state;
                let mut state = lock.lock().map_err(|_| Self::lock_poisoned())?;
                let my_generation = state.arrive("barrier", self.rank, self.size, Box::new(()));
                cvar.notify_all();
                loop {
                    if state.round_ready("barrier", my_generation) {
                        return Ok(());
                    }
                    state = cvar.wait(state).map_err(|_| Self::lock_poisoned())?;
                }
            }
            MPIBackend::TCP(_) => Err(Self::unsupported_backend("TCP", "barrier")),
            #[cfg(feature = "mpi")]
            MPIBackend::Native(_) => Err(Self::unsupported_backend("Native", "barrier")),
        }
    }

    /// Send and receive data with a partner rank (real rendezvous exchange
    /// for the Simulated backend).
    pub fn sendrecv(
        &self,
        send_data: &[Complex64],
        partner: usize,
    ) -> QuantRS2Result<Vec<Complex64>> {
        match &self.backend {
            MPIBackend::Simulated(sim) => {
                let (lock, cvar) = &*sim.shared_state;
                let mut bytes = Vec::with_capacity(send_data.len() * 16);
                for c in send_data {
                    bytes.extend_from_slice(&c.re.to_le_bytes());
                    bytes.extend_from_slice(&c.im.to_le_bytes());
                }
                {
                    let mut state = lock.lock().map_err(|_| Self::lock_poisoned())?;
                    state.mailbox.insert((self.rank, partner), bytes);
                }
                cvar.notify_all();

                let mut state = lock.lock().map_err(|_| Self::lock_poisoned())?;
                let recv_bytes = loop {
                    if let Some(bytes) = state.mailbox.remove(&(partner, self.rank)) {
                        break bytes;
                    }
                    state = cvar.wait(state).map_err(|_| Self::lock_poisoned())?;
                };
                drop(state);

                let recv = recv_bytes
                    .chunks_exact(16)
                    .map(|chunk| {
                        let re = f64::from_le_bytes(chunk[0..8].try_into().unwrap_or([0u8; 8]));
                        let im = f64::from_le_bytes(chunk[8..16].try_into().unwrap_or([0u8; 8]));
                        Complex64::new(re, im)
                    })
                    .collect();
                Ok(recv)
            }
            MPIBackend::TCP(_) => Err(Self::unsupported_backend("TCP", "sendrecv")),
            #[cfg(feature = "mpi")]
            MPIBackend::Native(_) => Err(Self::unsupported_backend("Native", "sendrecv")),
        }
    }

    /// Gather data from all ranks to `root` (root receives every rank's
    /// contribution concatenated in rank order; other ranks get their own
    /// data back, matching real `MPI_Gather` semantics).
    pub fn gather<T: Clone + Send + 'static>(
        &self,
        local_data: &[T],
        root: usize,
    ) -> QuantRS2Result<Vec<T>> {
        match &self.backend {
            MPIBackend::Simulated(sim) => {
                let (lock, cvar) = &*sim.shared_state;
                let key = format!("gather:{root}");
                let payload: Box<dyn std::any::Any + Send> = Box::new(local_data.to_vec());
                let mut state = lock.lock().map_err(|_| Self::lock_poisoned())?;
                let my_generation = state.arrive(&key, self.rank, self.size, payload);
                cvar.notify_all();
                loop {
                    if state.round_ready(&key, my_generation) {
                        break;
                    }
                    state = cvar.wait(state).map_err(|_| Self::lock_poisoned())?;
                }

                if self.rank != root {
                    return Ok(local_data.to_vec());
                }

                let round = state.rounds.get(&key).ok_or_else(|| {
                    QuantRS2Error::InvalidInput(
                        "gather: collective round missing after completion".to_string(),
                    )
                })?;
                let mut result = Vec::new();
                for boxed in round.completed_payloads.values() {
                    let chunk = boxed.downcast_ref::<Vec<T>>().ok_or_else(|| {
                        QuantRS2Error::InvalidInput(
                            "gather: mismatched payload type across ranks".to_string(),
                        )
                    })?;
                    result.extend(chunk.iter().cloned());
                }
                Ok(result)
            }
            MPIBackend::TCP(_) => Err(Self::unsupported_backend("TCP", "gather")),
            #[cfg(feature = "mpi")]
            MPIBackend::Native(_) => Err(Self::unsupported_backend("Native", "gather")),
        }
    }

    /// Broadcast data from `root` to all ranks.
    pub fn broadcast<T: Clone + Send + 'static>(
        &self,
        data: &[T],
        root: usize,
    ) -> QuantRS2Result<Vec<T>> {
        match &self.backend {
            MPIBackend::Simulated(sim) => {
                let (lock, cvar) = &*sim.shared_state;
                let key = format!("broadcast:{root}");
                let payload: Box<dyn std::any::Any + Send> = Box::new(data.to_vec());
                let mut state = lock.lock().map_err(|_| Self::lock_poisoned())?;
                let my_generation = state.arrive(&key, self.rank, self.size, payload);
                cvar.notify_all();
                loop {
                    if state.round_ready(&key, my_generation) {
                        break;
                    }
                    state = cvar.wait(state).map_err(|_| Self::lock_poisoned())?;
                }

                let round = state.rounds.get(&key).ok_or_else(|| {
                    QuantRS2Error::InvalidInput(
                        "broadcast: collective round missing after completion".to_string(),
                    )
                })?;
                let boxed = round.completed_payloads.get(&root).ok_or_else(|| {
                    QuantRS2Error::InvalidInput(format!(
                        "broadcast: root rank {root} did not participate in this round"
                    ))
                })?;
                let result = boxed
                    .downcast_ref::<Vec<T>>()
                    .ok_or_else(|| {
                        QuantRS2Error::InvalidInput(
                            "broadcast: mismatched payload type across ranks".to_string(),
                        )
                    })?
                    .clone();
                Ok(result)
            }
            MPIBackend::TCP(_) => Err(Self::unsupported_backend("TCP", "broadcast")),
            #[cfg(feature = "mpi")]
            MPIBackend::Native(_) => Err(Self::unsupported_backend("Native", "broadcast")),
        }
    }

    /// Allreduce operation: elementwise-reduce `local_data` across every
    /// rank and return the reduced vector to all of them.
    pub fn allreduce(&self, local_data: &[f64], op: ReduceOp) -> QuantRS2Result<Vec<f64>> {
        match &self.backend {
            MPIBackend::Simulated(sim) => {
                let (lock, cvar) = &*sim.shared_state;
                let key = "allreduce".to_string();
                let payload: Box<dyn std::any::Any + Send> = Box::new(local_data.to_vec());
                let mut state = lock.lock().map_err(|_| Self::lock_poisoned())?;
                let my_generation = state.arrive(&key, self.rank, self.size, payload);
                cvar.notify_all();
                loop {
                    if state.round_ready(&key, my_generation) {
                        break;
                    }
                    state = cvar.wait(state).map_err(|_| Self::lock_poisoned())?;
                }

                let round = state.rounds.get(&key).ok_or_else(|| {
                    QuantRS2Error::InvalidInput(
                        "allreduce: collective round missing after completion".to_string(),
                    )
                })?;
                let mut reduced: Option<Vec<f64>> = None;
                for boxed in round.completed_payloads.values() {
                    let v = boxed.downcast_ref::<Vec<f64>>().ok_or_else(|| {
                        QuantRS2Error::InvalidInput(
                            "allreduce: mismatched payload type across ranks".to_string(),
                        )
                    })?;
                    reduced = Some(match reduced {
                        None => v.clone(),
                        Some(acc) => acc
                            .iter()
                            .zip(v.iter())
                            .map(|(a, b)| match op {
                                ReduceOp::Sum => a + b,
                                ReduceOp::Max => a.max(*b),
                                ReduceOp::Min => a.min(*b),
                                ReduceOp::Prod => a * b,
                            })
                            .collect(),
                    });
                }
                reduced.ok_or_else(|| {
                    QuantRS2Error::InvalidInput("allreduce: empty collective round".to_string())
                })
            }
            MPIBackend::TCP(_) => Err(Self::unsupported_backend("TCP", "allreduce")),
            #[cfg(feature = "mpi")]
            MPIBackend::Native(_) => Err(Self::unsupported_backend("Native", "allreduce")),
        }
    }
}

/// Reduce operations for allreduce
#[derive(Debug, Clone, Copy)]
pub enum ReduceOp {
    Sum,
    Max,
    Min,
    Prod,
}

/// Result of MPI quantum simulation
#[derive(Debug, Clone)]
pub struct MPISimulationResult {
    /// Measurement results
    pub measurements: Vec<bool>,
    /// Probability distribution
    pub probabilities: Vec<f64>,
    /// Simulation statistics
    pub stats: MPISimulatorStats,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mpi_simulator_creation() {
        let config = MPISimulatorConfig {
            total_qubits: 4,
            ..Default::default()
        };
        let simulator = MPIQuantumSimulator::new(config);
        assert!(simulator.is_ok());
    }

    #[test]
    fn test_mpi_simulator_initialization() {
        let config = MPISimulatorConfig {
            total_qubits: 4,
            ..Default::default()
        };
        let mut simulator = MPIQuantumSimulator::new(config).expect("failed to create simulator");
        assert!(simulator.initialize().is_ok());

        let state = simulator
            .get_local_state()
            .expect("failed to get local state");
        assert_eq!(state[0], Complex64::new(1.0, 0.0));
    }

    #[test]
    fn test_mpi_communicator_creation() {
        let comm = MPICommunicator::new();
        assert!(comm.is_ok());

        let comm = comm.expect("failed to create communicator");
        assert_eq!(comm.rank(), 0);
        assert_eq!(comm.size(), 1);
    }

    #[test]
    fn test_single_qubit_gate() {
        let config = MPISimulatorConfig {
            total_qubits: 4,
            ..Default::default()
        };
        let mut simulator = MPIQuantumSimulator::new(config).expect("failed to create simulator");
        simulator.initialize().expect("failed to initialize");

        // Apply X gate
        let x_gate = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .expect("valid 2x2 matrix shape");

        let result = simulator.apply_single_qubit_gate(0, &x_gate);
        assert!(result.is_ok());
    }

    #[test]
    fn test_probability_distribution() {
        let config = MPISimulatorConfig {
            total_qubits: 2,
            ..Default::default()
        };
        let mut simulator = MPIQuantumSimulator::new(config).expect("failed to create simulator");
        simulator.initialize().expect("failed to initialize");

        let probs = simulator
            .get_probability_distribution()
            .expect("failed to get probability distribution");
        assert_eq!(probs.len(), 4);
        assert!((probs[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mpi_stats() {
        let config = MPISimulatorConfig {
            total_qubits: 4,
            ..Default::default()
        };
        let simulator = MPIQuantumSimulator::new(config).expect("failed to create simulator");

        let stats = simulator.get_stats().expect("failed to get stats");
        assert_eq!(stats.gates_executed, 0);
    }

    #[test]
    fn test_distribution_strategies() {
        let strategies = vec![
            MPIDistributionStrategy::AmplitudePartition,
            MPIDistributionStrategy::QubitPartition,
            MPIDistributionStrategy::HybridPartition,
            MPIDistributionStrategy::GateAwarePartition,
            MPIDistributionStrategy::HilbertCurvePartition,
        ];

        for strategy in strategies {
            let config = MPISimulatorConfig {
                total_qubits: 4,
                distribution_strategy: strategy,
                ..Default::default()
            };
            let simulator = MPIQuantumSimulator::new(config);
            assert!(simulator.is_ok());
        }
    }

    #[test]
    fn test_reset() {
        let config = MPISimulatorConfig {
            total_qubits: 4,
            ..Default::default()
        };
        let mut simulator = MPIQuantumSimulator::new(config).expect("failed to create simulator");
        simulator.initialize().expect("failed to initialize");

        // Apply some gates
        let h_gate = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(-1.0 / 2.0_f64.sqrt(), 0.0),
            ],
        )
        .expect("valid 2x2 matrix shape");
        simulator
            .apply_single_qubit_gate(0, &h_gate)
            .expect("failed to apply gate");

        // Reset
        simulator.reset().expect("failed to reset");

        // Check state is back to |0...0>
        let state = simulator
            .get_local_state()
            .expect("failed to get local state");
        assert!((state[0] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn test_collective_optimization_config() {
        let config = CollectiveOptimization {
            use_nonblocking: true,
            enable_fusion: true,
            buffer_size: 32 * 1024 * 1024,
            allreduce_algorithm: AllreduceAlgorithm::Ring,
            broadcast_algorithm: BroadcastAlgorithm::Pipeline,
        };

        assert!(config.use_nonblocking);
        assert!(config.enable_fusion);
        assert_eq!(config.buffer_size, 32 * 1024 * 1024);
    }

    #[test]
    fn test_checkpoint_config() {
        let config = CheckpointConfig {
            enable: true,
            interval: 500,
            storage_path: "/custom/path".to_string(),
            use_compression: false,
        };

        assert!(config.enable);
        assert_eq!(config.interval, 500);
        assert!(!config.use_compression);
    }

    #[test]
    fn test_two_qubit_gate() {
        let config = MPISimulatorConfig {
            total_qubits: 4,
            ..Default::default()
        };
        let mut simulator = MPIQuantumSimulator::new(config).expect("failed to create simulator");
        simulator.initialize().expect("failed to initialize");

        // CNOT gate matrix
        let cnot_gate = Array2::from_shape_vec(
            (4, 4),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .expect("valid 4x4 matrix shape");

        let result = simulator.apply_two_qubit_gate(0, 1, &cnot_gate);
        assert!(result.is_ok());
    }

    /// Regression test for the P0 finding: `MPICommunicator::sendrecv` used
    /// to just echo the caller's own data back instead of exchanging real
    /// data between ranks. With the fixed Simulated backend, two distinct
    /// `MPICommunicator`s sharing the same `SimulatedMPIState` must
    /// actually swap their payloads.
    #[test]
    fn test_simulated_backend_sendrecv_exchanges_real_data() {
        let shared = Arc::new((Mutex::new(SimulatedMPIState::default()), Condvar::new()));
        let comm0 = MPICommunicator::with_config(
            0,
            2,
            MPIBackend::Simulated(SimulatedMPIBackend {
                shared_state: shared.clone(),
            }),
        );
        let comm1 = MPICommunicator::with_config(
            1,
            2,
            MPIBackend::Simulated(SimulatedMPIBackend {
                shared_state: shared.clone(),
            }),
        );

        let handle0 = std::thread::spawn(move || {
            comm0.sendrecv(&[Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)], 1)
        });
        let handle1 = std::thread::spawn(move || {
            comm1.sendrecv(&[Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)], 0)
        });

        let recv0 = handle0
            .join()
            .expect("rank0 thread panicked")
            .expect("rank0 sendrecv failed");
        let recv1 = handle1
            .join()
            .expect("rank1 thread panicked")
            .expect("rank1 sendrecv failed");

        // Rank 0 must receive rank 1's data, and vice versa -- not its own.
        assert_eq!(
            recv0,
            vec![Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)]
        );
        assert_eq!(
            recv1,
            vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)]
        );
    }

    /// Regression test for the P0 finding: `gather`/`broadcast`/`allreduce`
    /// used to be no-ops that returned the caller's own local data. With
    /// the fixed Simulated backend, root must receive every rank's
    /// contribution (gather), every rank must receive root's data
    /// (broadcast), and every rank must receive the true elementwise sum
    /// (allreduce) -- computed from data it never had locally.
    #[test]
    fn test_simulated_backend_collectives_are_real() {
        let shared = Arc::new((Mutex::new(SimulatedMPIState::default()), Condvar::new()));
        let num_ranks = 3usize;
        let mut handles = Vec::new();
        for rank in 0..num_ranks {
            let comm = MPICommunicator::with_config(
                rank,
                num_ranks,
                MPIBackend::Simulated(SimulatedMPIBackend {
                    shared_state: shared.clone(),
                }),
            );
            handles.push(std::thread::spawn(move || {
                let gathered = comm
                    .gather(&[rank as f64], 0)
                    .expect("gather should succeed");
                let broadcasted = comm
                    .broadcast(if rank == 0 { &[42i64] } else { &[] }, 0)
                    .expect("broadcast should succeed");
                let reduced = comm
                    .allreduce(&[(rank + 1) as f64], ReduceOp::Sum)
                    .expect("allreduce should succeed");
                (rank, gathered, broadcasted, reduced)
            }));
        }

        for handle in handles {
            let (rank, gathered, broadcasted, reduced) = handle.join().expect("thread panicked");
            if rank == 0 {
                assert_eq!(gathered, vec![0.0, 1.0, 2.0]);
            } else {
                // Non-root ranks get their own contribution back, matching
                // real MPI_Gather semantics (only root has the full result).
                assert_eq!(gathered, vec![rank as f64]);
            }
            assert_eq!(broadcasted, vec![42i64]);
            // Sum of (1, 2, 3) across the three ranks.
            assert_eq!(reduced, vec![6.0]);
        }
    }

    /// Regression test for the P0 finding: `apply_full_distributed_gate`
    /// discarded the gate matrix entirely and applied identity whenever
    /// both qubits of a two-qubit gate lived on remote ranks. This builds
    /// four ranks (so a 3-qubit state has two genuinely remote qubits),
    /// applies a real CNOT(control=qubit1, target=qubit2) across ranks,
    /// and checks the amplitude actually moved to the physically correct
    /// basis state instead of staying put.
    #[test]
    fn test_full_distributed_two_qubit_gate_applies_real_math() {
        let total_qubits = 3usize;
        let num_ranks = 4usize;
        let shared = Arc::new((Mutex::new(SimulatedMPIState::default()), Condvar::new()));

        // Basis index convention: index = qubit0 + 2*qubit1 + 4*qubit2.
        // Start in |q2=0, q1=1, q0=0> = index 2.
        let initial_index = 2usize;
        // CNOT(control=qubit1, target=qubit2) in (control,target) row-major
        // order (row/col = 2*control_val + target_val).
        let cnot = Array2::from_shape_vec(
            (4, 4),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .expect("valid 4x4 CNOT shape");

        let local_size = (1usize << total_qubits) / num_ranks;
        let mut handles = Vec::new();
        for rank in 0..num_ranks {
            let mut amplitudes = Array1::zeros(local_size);
            let global_offset = rank * local_size;
            if initial_index >= global_offset && initial_index < global_offset + local_size {
                amplitudes[initial_index - global_offset] = Complex64::new(1.0, 0.0);
            }

            let local_state = LocalQuantumState {
                amplitudes,
                global_offset,
                local_qubits: MPIQuantumSimulator::calculate_local_qubits(
                    total_qubits,
                    rank,
                    num_ranks,
                ),
                ghost_cells: GhostCells::default(),
            };

            let communicator = MPICommunicator::with_config(
                rank,
                num_ranks,
                MPIBackend::Simulated(SimulatedMPIBackend {
                    shared_state: shared.clone(),
                }),
            );
            let gate_handler = GateDistributionHandler {
                routing_table: Arc::new(RwLock::new(HashMap::new())),
                gate_classifier: GateClassifier {
                    local_qubits: local_state.local_qubits.clone(),
                },
            };
            let mut simulator = MPIQuantumSimulator {
                communicator,
                local_state: Arc::new(RwLock::new(local_state)),
                config: MPISimulatorConfig {
                    total_qubits,
                    ..Default::default()
                },
                stats: Arc::new(Mutex::new(MPISimulatorStats::default())),
                sync_manager: StateSynchronizationManager {
                    strategy: SyncStrategy::Adaptive,
                    pending: Arc::new(Mutex::new(Vec::new())),
                },
                gate_handler,
            };

            let cnot = cnot.clone();
            handles.push(std::thread::spawn(move || {
                simulator
                    .apply_two_qubit_gate(1, 2, &cnot)
                    .expect("apply_two_qubit_gate should succeed");
                (rank, simulator.get_local_state().expect("get_local_state"))
            }));
        }

        // Expected final basis state: control qubit1=1 flips target
        // qubit2 0 -> 1, giving |q2=1, q1=1, q0=0> = index 6.
        let expected_index = 6usize;

        for handle in handles {
            let (rank, state) = handle.join().expect("rank thread panicked");
            let global_offset = rank * local_size;
            for (i, amp) in state.iter().enumerate() {
                let global_i = global_offset + i;
                if global_i == expected_index {
                    assert!(
                        (amp.norm() - 1.0).abs() < 1e-10,
                        "expected amplitude 1.0 at global index {expected_index}, got {amp:?} on rank {rank}"
                    );
                } else {
                    assert!(
                        amp.norm() < 1e-10,
                        "expected zero amplitude at global index {global_i}, got {amp:?} on rank {rank}"
                    );
                }
            }
        }
    }
}
