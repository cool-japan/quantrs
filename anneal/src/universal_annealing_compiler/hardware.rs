//! Hardware specification types for quantum platforms.
//!
//! This module contains types for describing hardware requirements,
//! connectivity, error characteristics, and operating conditions.

use std::time::Duration;

/// Hardware requirements
#[derive(Debug, Clone)]
pub struct HardwareRequirements {
    /// Connectivity requirements
    pub connectivity: ConnectivityRequirements,
    /// Embedding requirements
    pub embedding: EmbeddingRequirements,
    /// Coherence requirements
    pub coherence: CoherenceRequirements,
    /// Error rate requirements
    pub error_rates: ErrorRateRequirements,
}

impl Default for HardwareRequirements {
    fn default() -> Self {
        Self {
            connectivity: ConnectivityRequirements::default(),
            embedding: EmbeddingRequirements::default(),
            coherence: CoherenceRequirements::default(),
            error_rates: ErrorRateRequirements::default(),
        }
    }
}

/// Connectivity requirements
#[derive(Debug, Clone)]
pub struct ConnectivityRequirements {
    /// Minimum average degree
    pub min_average_degree: f64,
    /// Required topology type
    pub required_topology: Option<TopologyType>,
    /// Maximum embedding overhead
    pub max_embedding_overhead: f64,
}

impl Default for ConnectivityRequirements {
    fn default() -> Self {
        Self {
            min_average_degree: 2.0,
            required_topology: None,
            max_embedding_overhead: 3.0,
        }
    }
}

/// Topology types
#[derive(Debug, Clone, PartialEq)]
pub enum TopologyType {
    /// Pegasus topology (D-Wave)
    Pegasus,
    /// Chimera topology (D-Wave)
    Chimera,
    /// Zephyr topology (D-Wave)
    Zephyr,
    /// Heavy-hex (IBM)
    HeavyHex,
    /// All-to-all connectivity
    AllToAll,
    /// Linear chain
    Linear,
    /// Grid
    Grid,
    /// Custom topology
    Custom(String),
}

/// Embedding requirements
#[derive(Debug, Clone)]
pub struct EmbeddingRequirements {
    /// Maximum chain length
    pub max_chain_length: usize,
    /// Preferred embedding strategy
    pub embedding_strategy: EmbeddingStrategy,
    /// Allow embedding failures
    pub allow_failures: bool,
}

impl Default for EmbeddingRequirements {
    fn default() -> Self {
        Self {
            max_chain_length: 10,
            embedding_strategy: EmbeddingStrategy::MinorMinimization,
            allow_failures: false,
        }
    }
}

/// Embedding strategies
#[derive(Debug, Clone, PartialEq)]
pub enum EmbeddingStrategy {
    /// Minimize minor embedding
    MinorMinimization,
    /// Minimize chain length
    ChainLengthMinimization,
    /// Balance load
    LoadBalanced,
    /// Error-aware embedding
    ErrorAware,
    /// Fast embedding
    Fast,
}

/// Coherence requirements
#[derive(Debug, Clone)]
pub struct CoherenceRequirements {
    /// Minimum T1 relaxation time
    pub min_t1_time: Option<Duration>,
    /// Minimum T2 dephasing time
    pub min_t2_time: Option<Duration>,
    /// Minimum coherence fidelity
    pub min_coherence_fidelity: f64,
}

impl Default for CoherenceRequirements {
    fn default() -> Self {
        Self {
            min_t1_time: Some(Duration::from_micros(100)),
            min_t2_time: Some(Duration::from_micros(50)),
            min_coherence_fidelity: 0.95,
        }
    }
}

/// Error rate requirements
#[derive(Debug, Clone)]
pub struct ErrorRateRequirements {
    /// Maximum single-qubit error rate
    pub max_single_qubit_error_rate: f64,
    /// Maximum two-qubit error rate
    pub max_two_qubit_error_rate: f64,
    /// Maximum readout error rate
    pub max_readout_error_rate: f64,
}

impl Default for ErrorRateRequirements {
    fn default() -> Self {
        Self {
            max_single_qubit_error_rate: 0.001,
            max_two_qubit_error_rate: 0.01,
            max_readout_error_rate: 0.01,
        }
    }
}

/// Hardware specification
#[derive(Debug, Clone)]
pub struct HardwareSpecification {
    /// Device name
    pub device_name: String,
    /// Number of qubits
    pub num_qubits: usize,
    /// Connectivity graph
    pub connectivity: ConnectivityGraph,
    /// Error characteristics
    pub error_characteristics: ErrorCharacteristics,
    /// Operating conditions
    pub operating_conditions: OperatingConditions,
}

/// Connectivity graph representation
#[derive(Debug, Clone)]
pub struct ConnectivityGraph {
    /// Adjacency matrix
    pub adjacency_matrix: Vec<Vec<bool>>,
    /// Topology type
    pub topology_type: TopologyType,
    /// Graph properties
    pub properties: GraphProperties,
}

/// Graph properties
#[derive(Debug, Clone)]
pub struct GraphProperties {
    /// Average degree
    pub average_degree: f64,
    /// Diameter
    pub diameter: usize,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
    /// Spectral gap
    pub spectral_gap: f64,
}

/// Error characteristics
#[derive(Debug, Clone)]
pub struct ErrorCharacteristics {
    /// Single-qubit error rates
    pub single_qubit_errors: Vec<f64>,
    /// Two-qubit error rates
    pub two_qubit_errors: Vec<Vec<f64>>,
    /// Readout errors
    pub readout_errors: Vec<f64>,
    /// Coherence times
    pub coherence_times: CoherenceTimes,
}

/// Coherence times
#[derive(Debug, Clone)]
pub struct CoherenceTimes {
    /// T1 relaxation times
    pub t1_times: Vec<Duration>,
    /// T2 dephasing times
    pub t2_times: Vec<Duration>,
    /// T2* times
    pub t2_star_times: Vec<Duration>,
}

/// Operating conditions
#[derive(Debug, Clone)]
pub struct OperatingConditions {
    /// Temperature
    pub temperature: f64,
    /// Magnetic field
    pub magnetic_field: f64,
    /// Pressure
    pub pressure: f64,
    /// Environmental noise
    pub environmental_noise: f64,
}

/// Platform performance characteristics
#[derive(Debug, Clone)]
pub struct PlatformPerformanceCharacteristics {
    /// Typical execution time
    pub typical_execution_time: Duration,
    /// Queue wait time
    pub typical_queue_time: Duration,
    /// Success rate
    pub success_rate: f64,
    /// Fidelity
    pub fidelity: f64,
    /// Throughput
    pub throughput: f64,
}
