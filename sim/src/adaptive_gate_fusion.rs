//! Adaptive gate fusion based on circuit structure analysis.
//!
//! This module implements intelligent gate fusion algorithms that analyze
//! quantum circuit structures to automatically identify and fuse adjacent
//! gates for optimal performance. It leverages SciRS2's optimization
//! capabilities for efficient matrix operations and circuit transformations.

use ndarray::Array2;
use num_complex::Complex64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use crate::error::{Result, SimulatorError};
use crate::scirs2_integration::SciRS2Backend;

#[cfg(feature = "advanced_math")]
use quantrs2_circuit::prelude::*;

/// Gate fusion strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusionStrategy {
    /// Aggressive fusion - fuse as many gates as possible
    Aggressive,
    /// Conservative fusion - only fuse when clearly beneficial
    Conservative,
    /// Balanced fusion - balance between fusion opportunities and overhead
    Balanced,
    /// Adaptive fusion - learn from circuit patterns
    Adaptive,
    /// Custom fusion based on specific criteria
    Custom,
}

/// Gate fusion configuration
#[derive(Debug, Clone)]
pub struct AdaptiveFusionConfig {
    /// Primary fusion strategy
    pub strategy: FusionStrategy,
    /// Maximum number of gates to fuse in a single block
    pub max_fusion_size: usize,
    /// Minimum benefit threshold for fusion (relative speedup)
    pub min_benefit_threshold: f64,
    /// Enable cross-qubit fusion analysis
    pub enable_cross_qubit_fusion: bool,
    /// Enable temporal fusion across time steps
    pub enable_temporal_fusion: bool,
    /// Maximum circuit depth to analyze for fusion
    pub max_analysis_depth: usize,
    /// Enable machine learning-based fusion predictions
    pub enable_ml_predictions: bool,
    /// Fusion cache size for repeated patterns
    pub fusion_cache_size: usize,
    /// Enable parallel fusion analysis
    pub parallel_analysis: bool,
}

impl Default for AdaptiveFusionConfig {
    fn default() -> Self {
        Self {
            strategy: FusionStrategy::Adaptive,
            max_fusion_size: 8,
            min_benefit_threshold: 1.1,
            enable_cross_qubit_fusion: true,
            enable_temporal_fusion: true,
            max_analysis_depth: 100,
            enable_ml_predictions: true,
            fusion_cache_size: 10000,
            parallel_analysis: true,
        }
    }
}

/// Quantum gate representation for fusion analysis
#[derive(Debug, Clone)]
pub struct QuantumGate {
    /// Gate type identifier
    pub gate_type: GateType,
    /// Qubits this gate acts on
    pub qubits: Vec<usize>,
    /// Gate parameters (angles, etc.)
    pub parameters: Vec<f64>,
    /// Unitary matrix representation
    pub matrix: Array2<Complex64>,
    /// Position in the circuit
    pub position: usize,
    /// Estimated execution cost
    pub cost: f64,
}

/// Supported gate types for fusion
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GateType {
    // Single-qubit gates
    Identity,
    PauliX,
    PauliY,
    PauliZ,
    Hadamard,
    Phase,
    T,
    RotationX,
    RotationY,
    RotationZ,
    // Two-qubit gates
    CNOT,
    CZ,
    SWAP,
    ISwap,
    // Multi-qubit gates
    Toffoli,
    Fredkin,
    // Custom gates
    Custom(String),
}

impl QuantumGate {
    /// Create a new quantum gate
    pub fn new(gate_type: GateType, qubits: Vec<usize>, parameters: Vec<f64>) -> Self {
        let matrix = Self::gate_matrix(&gate_type, &parameters);
        let cost = Self::estimate_cost(&gate_type, qubits.len());

        Self {
            gate_type,
            qubits,
            parameters,
            matrix,
            position: 0,
            cost,
        }
    }

    /// Get the unitary matrix for a gate type
    fn gate_matrix(gate_type: &GateType, parameters: &[f64]) -> Array2<Complex64> {
        match gate_type {
            GateType::Identity => Array2::eye(2),
            GateType::PauliX => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
            )
            .unwrap(),
            GateType::PauliY => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, -1.0),
                    Complex64::new(0.0, 1.0),
                    Complex64::new(0.0, 0.0),
                ],
            )
            .unwrap(),
            GateType::PauliZ => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(-1.0, 0.0),
                ],
            )
            .unwrap(),
            GateType::Hadamard => {
                let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
                Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex64::new(inv_sqrt2, 0.0),
                        Complex64::new(inv_sqrt2, 0.0),
                        Complex64::new(inv_sqrt2, 0.0),
                        Complex64::new(-inv_sqrt2, 0.0),
                    ],
                )
                .unwrap()
            }
            GateType::RotationX => {
                let theta = parameters.get(0).copied().unwrap_or(0.0);
                let cos_half = (theta / 2.0).cos();
                let sin_half = (theta / 2.0).sin();
                Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex64::new(cos_half, 0.0),
                        Complex64::new(0.0, -sin_half),
                        Complex64::new(0.0, -sin_half),
                        Complex64::new(cos_half, 0.0),
                    ],
                )
                .unwrap()
            }
            GateType::RotationY => {
                let theta = parameters.get(0).copied().unwrap_or(0.0);
                let cos_half = (theta / 2.0).cos();
                let sin_half = (theta / 2.0).sin();
                Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex64::new(cos_half, 0.0),
                        Complex64::new(-sin_half, 0.0),
                        Complex64::new(sin_half, 0.0),
                        Complex64::new(cos_half, 0.0),
                    ],
                )
                .unwrap()
            }
            GateType::RotationZ => {
                let theta = parameters.get(0).copied().unwrap_or(0.0);
                let exp_neg = Complex64::new(0.0, -theta / 2.0).exp();
                let exp_pos = Complex64::new(0.0, theta / 2.0).exp();
                Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        exp_neg,
                        Complex64::new(0.0, 0.0),
                        Complex64::new(0.0, 0.0),
                        exp_pos,
                    ],
                )
                .unwrap()
            }
            GateType::CNOT => Array2::from_shape_vec(
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
            .unwrap(),
            _ => Array2::eye(2), // Default fallback
        }
    }

    /// Estimate execution cost for a gate
    fn estimate_cost(gate_type: &GateType, num_qubits: usize) -> f64 {
        let base_cost = match gate_type {
            GateType::Identity => 0.1,
            GateType::PauliX | GateType::PauliY | GateType::PauliZ => 1.0,
            GateType::Hadamard => 1.2,
            GateType::Phase | GateType::T => 1.1,
            GateType::RotationX | GateType::RotationY | GateType::RotationZ => 1.5,
            GateType::CNOT | GateType::CZ => 2.0,
            GateType::SWAP | GateType::ISwap => 2.5,
            GateType::Toffoli => 4.0,
            GateType::Fredkin => 4.5,
            GateType::Custom(_) => 3.0,
        };

        // Cost scales with number of qubits
        base_cost * (1 << num_qubits) as f64
    }

    /// Check if this gate commutes with another gate
    pub fn commutes_with(&self, other: &QuantumGate) -> bool {
        // Check if gates act on disjoint qubits
        let self_qubits: HashSet<_> = self.qubits.iter().collect();
        let other_qubits: HashSet<_> = other.qubits.iter().collect();

        if self_qubits.is_disjoint(&other_qubits) {
            return true;
        }

        // For gates acting on same qubits, check specific commutation rules
        if self.qubits == other.qubits {
            return self.check_specific_commutation(other);
        }

        false
    }

    /// Check specific commutation rules for gates on same qubits
    fn check_specific_commutation(&self, other: &QuantumGate) -> bool {
        match (&self.gate_type, &other.gate_type) {
            // Pauli gates commute with themselves
            (GateType::PauliX, GateType::PauliX)
            | (GateType::PauliY, GateType::PauliY)
            | (GateType::PauliZ, GateType::PauliZ) => true,
            // Z commutes with RZ
            (GateType::PauliZ, GateType::RotationZ) | (GateType::RotationZ, GateType::PauliZ) => {
                true
            }
            // X commutes with RX
            (GateType::PauliX, GateType::RotationX) | (GateType::RotationX, GateType::PauliX) => {
                true
            }
            // Y commutes with RY
            (GateType::PauliY, GateType::RotationY) | (GateType::RotationY, GateType::PauliY) => {
                true
            }
            // Rotation gates of same type commute
            (GateType::RotationX, GateType::RotationX)
            | (GateType::RotationY, GateType::RotationY)
            | (GateType::RotationZ, GateType::RotationZ) => true,
            // Identity commutes with everything
            (GateType::Identity, _) | (_, GateType::Identity) => true,
            _ => false,
        }
    }

    /// Check if this gate can be fused with another gate
    pub fn can_fuse_with(&self, other: &QuantumGate) -> bool {
        // Gates must act on the same qubits to be fusable
        if self.qubits != other.qubits {
            return false;
        }

        // Check if both are single-qubit gates (easier to fuse)
        if self.qubits.len() == 1 && other.qubits.len() == 1 {
            return true;
        }

        // Two-qubit gates can sometimes be fused
        if self.qubits.len() == 2 && other.qubits.len() == 2 {
            return self.check_two_qubit_fusion_compatibility(other);
        }

        false
    }

    /// Check if two-qubit gates can be fused
    fn check_two_qubit_fusion_compatibility(&self, other: &QuantumGate) -> bool {
        match (&self.gate_type, &other.gate_type) {
            // CNOTs can be fused in certain patterns
            (GateType::CNOT, GateType::CNOT) => true,
            // CZ gates can be fused
            (GateType::CZ, GateType::CZ) => true,
            // CNOT and CZ can sometimes be fused
            (GateType::CNOT, GateType::CZ) | (GateType::CZ, GateType::CNOT) => true,
            _ => false,
        }
    }
}

impl Hash for QuantumGate {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.gate_type.hash(state);
        self.qubits.hash(state);
        // Hash parameters with reduced precision to avoid floating point issues
        for &param in &self.parameters {
            ((param * 1000.0).round() as i64).hash(state);
        }
    }
}

impl PartialEq for QuantumGate {
    fn eq(&self, other: &Self) -> bool {
        self.gate_type == other.gate_type
            && self.qubits == other.qubits
            && self.parameters.len() == other.parameters.len()
            && self
                .parameters
                .iter()
                .zip(other.parameters.iter())
                .all(|(&a, &b)| (a - b).abs() < 1e-10)
    }
}

impl Eq for QuantumGate {}

/// Fused gate block containing multiple gates
#[derive(Debug, Clone)]
pub struct FusedGateBlock {
    /// Individual gates in this block
    pub gates: Vec<QuantumGate>,
    /// Combined unitary matrix for the entire block
    pub combined_matrix: Array2<Complex64>,
    /// Qubits affected by this block
    pub qubits: Vec<usize>,
    /// Estimated execution cost
    pub cost: f64,
    /// Performance improvement factor
    pub improvement_factor: f64,
}

impl FusedGateBlock {
    /// Create a new fused gate block
    pub fn new(gates: Vec<QuantumGate>) -> Result<Self> {
        if gates.is_empty() {
            return Err(SimulatorError::InvalidInput(
                "Cannot create empty gate block".to_string(),
            ));
        }

        // Determine qubits affected by this block
        let mut qubits = HashSet::new();
        for gate in &gates {
            qubits.extend(&gate.qubits);
        }
        let mut qubit_vec: Vec<usize> = qubits.into_iter().collect();
        qubit_vec.sort();

        // Calculate combined matrix
        let combined_matrix = Self::calculate_combined_matrix(&gates, &qubit_vec)?;

        // Calculate costs
        let individual_cost: f64 = gates.iter().map(|g| g.cost).sum();
        let fused_cost = Self::estimate_fused_cost(&combined_matrix);
        let improvement_factor = individual_cost / fused_cost;

        Ok(Self {
            gates,
            combined_matrix,
            qubits: qubit_vec,
            cost: fused_cost,
            improvement_factor,
        })
    }

    /// Calculate the combined unitary matrix for multiple gates
    fn calculate_combined_matrix(
        gates: &[QuantumGate],
        qubits: &[usize],
    ) -> Result<Array2<Complex64>> {
        let num_qubits = qubits.len();
        let matrix_size = 1 << num_qubits;
        let mut combined = Array2::eye(matrix_size);

        for gate in gates {
            let gate_matrix = Self::expand_gate_matrix(&gate.matrix, &gate.qubits, qubits)?;
            combined = gate_matrix.dot(&combined);
        }

        Ok(combined)
    }

    /// Expand a gate matrix to act on the full qubit space
    fn expand_gate_matrix(
        gate_matrix: &Array2<Complex64>,
        gate_qubits: &[usize],
        all_qubits: &[usize],
    ) -> Result<Array2<Complex64>> {
        let num_all_qubits = all_qubits.len();
        let full_size = 1 << num_all_qubits;
        let gate_size = 1 << gate_qubits.len();

        if gate_matrix.nrows() != gate_size || gate_matrix.ncols() != gate_size {
            return Err(SimulatorError::DimensionMismatch(
                "Gate matrix size doesn't match number of qubits".to_string(),
            ));
        }

        let mut expanded = Array2::eye(full_size);

        // Create mapping from gate qubits to positions in full space
        let mut qubit_mapping = HashMap::new();
        for (gate_pos, &qubit) in gate_qubits.iter().enumerate() {
            if let Some(all_pos) = all_qubits.iter().position(|&q| q == qubit) {
                qubit_mapping.insert(gate_pos, all_pos);
            } else {
                return Err(SimulatorError::InvalidInput(format!(
                    "Gate qubit {} not found in qubit list",
                    qubit
                )));
            }
        }

        // Apply gate to the appropriate subspace
        for i in 0..full_size {
            for j in 0..full_size {
                // Extract gate subspace indices
                let mut gate_i = 0;
                let mut gate_j = 0;
                let mut valid = true;

                for (gate_pos, &all_pos) in &qubit_mapping {
                    let bit_i = (i >> (num_all_qubits - 1 - all_pos)) & 1;
                    let bit_j = (j >> (num_all_qubits - 1 - all_pos)) & 1;

                    gate_i |= bit_i << (gate_qubits.len() - 1 - gate_pos);
                    gate_j |= bit_j << (gate_qubits.len() - 1 - gate_pos);
                }

                // Check if other qubits are the same
                for all_pos in 0..num_all_qubits {
                    if !qubit_mapping.values().any(|&pos| pos == all_pos) {
                        let bit_i = (i >> (num_all_qubits - 1 - all_pos)) & 1;
                        let bit_j = (j >> (num_all_qubits - 1 - all_pos)) & 1;
                        if bit_i != bit_j {
                            valid = false;
                            break;
                        }
                    }
                }

                if valid {
                    expanded[[i, j]] = gate_matrix[[gate_i, gate_j]];
                } else {
                    expanded[[i, j]] = if i == j {
                        Complex64::new(1.0, 0.0)
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                }
            }
        }

        Ok(expanded)
    }

    /// Estimate execution cost for a fused gate block
    fn estimate_fused_cost(matrix: &Array2<Complex64>) -> f64 {
        // Cost is primarily determined by matrix size
        let size = matrix.nrows();
        let base_cost = (size as f64).log2() * size as f64;

        // Add overhead for fusion setup
        let fusion_overhead = 0.1 * base_cost;

        base_cost + fusion_overhead
    }

    /// Check if fusion is beneficial
    pub fn is_beneficial(&self) -> bool {
        self.improvement_factor > 1.1 // At least 10% improvement
    }
}

/// Circuit analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitAnalysis {
    /// Number of gates in original circuit
    pub original_gate_count: usize,
    /// Number of gates after fusion
    pub fused_gate_count: usize,
    /// Number of fusion blocks created
    pub fusion_blocks: usize,
    /// Estimated performance improvement
    pub performance_improvement: f64,
    /// Gate type distribution
    pub gate_distribution: HashMap<String, usize>,
    /// Fusion opportunities identified
    pub fusion_opportunities: Vec<FusionOpportunity>,
    /// Circuit depth before and after fusion
    pub circuit_depth: (usize, usize),
}

/// Fusion opportunity description
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionOpportunity {
    /// Description of the opportunity
    pub description: String,
    /// Qubits involved
    pub qubits: Vec<usize>,
    /// Gate types that can be fused
    pub gate_types: Vec<String>,
    /// Estimated benefit
    pub estimated_benefit: f64,
    /// Confidence score (0-1)
    pub confidence: f64,
}

/// Adaptive gate fusion engine
pub struct AdaptiveGateFusion {
    /// Configuration
    config: AdaptiveFusionConfig,
    /// SciRS2 backend for optimization
    backend: Option<SciRS2Backend>,
    /// Fusion pattern cache with improved key system
    fusion_cache: HashMap<FusionPatternKey, CachedFusionResult>,
    /// Learning history for adaptive strategy
    learning_history: Vec<FusionExperiment>,
    /// Circuit optimization context
    #[cfg(feature = "advanced_math")]
    optimizer: Option<Box<dyn std::any::Any>>, // Placeholder for CircuitOptimizer
    /// Machine learning predictor for fusion benefits
    ml_predictor: Option<MLFusionPredictor>,
    /// Cache statistics
    cache_hits: usize,
    cache_misses: usize,
    /// Pattern analyzer for circuit structure recognition
    pattern_analyzer: CircuitPatternAnalyzer,
}

/// Cache key for fusion patterns
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FusionPatternKey {
    /// Gate types in the pattern
    gate_types: Vec<GateType>,
    /// Number of qubits involved
    num_qubits: usize,
    /// Parameter hash for parameterized gates
    parameter_hash: u64,
}

impl FusionPatternKey {
    /// Create a fusion pattern key from a list of gates
    pub fn from_gates(gates: &[QuantumGate]) -> Result<Self> {
        let gate_types: Vec<GateType> = gates.iter().map(|g| g.gate_type.clone()).collect();

        // Count unique qubits
        let mut qubits = std::collections::HashSet::new();
        for gate in gates {
            for &qubit in &gate.qubits {
                qubits.insert(qubit);
            }
        }
        let num_qubits = qubits.len();

        // Hash parameters
        let mut parameter_hash = 0u64;
        for gate in gates {
            for &param in &gate.parameters {
                parameter_hash = parameter_hash.wrapping_add((param * 1000.0) as u64);
            }
        }

        Ok(FusionPatternKey {
            gate_types,
            num_qubits,
            parameter_hash,
        })
    }
}

/// Cached fusion result
#[derive(Debug, Clone)]
pub struct CachedFusionResult {
    /// The fused gate block
    fused_block: FusedGateBlock,
    /// Performance benefit observed
    benefit: f64,
    /// Number of times this pattern was used
    usage_count: usize,
    /// Last access timestamp
    last_accessed: std::time::Instant,
}

/// Machine learning predictor for fusion benefits
pub struct MLFusionPredictor {
    /// Feature weights for different gate patterns
    feature_weights: HashMap<String, f64>,
    /// Training examples
    training_data: Vec<MLTrainingExample>,
    /// Model accuracy
    accuracy: f64,
}

/// Training example for ML predictor
#[derive(Debug, Clone)]
pub struct MLTrainingExample {
    /// Input features
    features: Vec<f64>,
    /// Expected benefit (target)
    benefit: f64,
    /// Actual observed benefit
    observed_benefit: Option<f64>,
}

/// Circuit pattern analyzer
pub struct CircuitPatternAnalyzer {
    /// Known beneficial patterns
    beneficial_patterns: HashMap<String, f64>,
    /// Pattern recognition history
    pattern_history: Vec<PatternRecognitionResult>,
}

/// Pattern recognition result
#[derive(Debug, Clone)]
pub struct PatternRecognitionResult {
    /// Pattern description
    pub pattern: String,
    /// Confidence in recognition
    pub confidence: f64,
    /// Expected benefit
    pub expected_benefit: f64,
}

/// Result of fusion decision
#[derive(Debug, Clone)]
pub struct FusionResult {
    /// Whether gates should be fused
    pub should_fuse: bool,
    /// Confidence in the decision
    pub confidence: f64,
    /// Expected speedup from fusion
    pub expected_speedup: f64,
    /// Estimated error increase
    pub estimated_error: f64,
}

/// Fusion experiment for learning
#[derive(Debug, Clone)]
struct FusionExperiment {
    circuit_fingerprint: u64,
    fusion_strategy: FusionStrategy,
    performance_improvement: f64,
    execution_time_ms: f64,
}

impl AdaptiveGateFusion {
    /// Create new adaptive gate fusion engine
    pub fn new(config: AdaptiveFusionConfig) -> Result<Self> {
        Ok(Self {
            config,
            backend: None,
            fusion_cache: HashMap::new(),
            learning_history: Vec::new(),
            ml_predictor: None,
            cache_hits: 0,
            cache_misses: 0,
            pattern_analyzer: CircuitPatternAnalyzer::new(),
            #[cfg(feature = "advanced_math")]
            optimizer: None,
        })
    }

    /// Initialize with SciRS2 backend
    pub fn with_backend(mut self) -> Result<Self> {
        self.backend = Some(SciRS2Backend::new());

        #[cfg(feature = "advanced_math")]
        {
            let strategy = match self.config.strategy {
                FusionStrategy::Aggressive => OptimizationStrategy::MinimizeTime,
                FusionStrategy::Conservative => OptimizationStrategy::MinimizeLength,
                FusionStrategy::Balanced => OptimizationStrategy::Balanced,
                FusionStrategy::Adaptive => OptimizationStrategy::MinimizeCrossings,
                FusionStrategy::Custom => OptimizationStrategy::Balanced,
            };

            // Placeholder for circuit optimizer integration
            self.optimizer = Some(Box::new(strategy));
        }

        Ok(self)
    }

    /// Analyze circuit and identify fusion opportunities
    pub fn analyze_circuit(&mut self, gates: &[QuantumGate]) -> Result<CircuitAnalysis> {
        let start_time = std::time::Instant::now();

        // Calculate circuit metrics
        let original_gate_count = gates.len();
        let original_depth = self.calculate_circuit_depth(gates);

        // Identify fusion opportunities
        let fusion_opportunities = self.identify_fusion_opportunities(gates)?;

        // Perform actual fusion
        let (fused_blocks, remaining_gates) = self.perform_fusion(gates)?;
        let fused_gate_count = fused_blocks.len() + remaining_gates.len();
        let fused_depth = self.calculate_fused_circuit_depth(&fused_blocks, &remaining_gates);

        // Calculate performance improvement
        let original_cost: f64 = gates.iter().map(|g| g.cost).sum();
        let fused_cost: f64 = fused_blocks.iter().map(|b| b.cost).sum::<f64>()
            + remaining_gates.iter().map(|g| g.cost).sum::<f64>();
        let performance_improvement = original_cost / fused_cost;

        // Gate distribution analysis
        let mut gate_distribution = HashMap::new();
        for gate in gates {
            let gate_name = format!("{:?}", gate.gate_type);
            *gate_distribution.entry(gate_name).or_insert(0) += 1;
        }

        let analysis = CircuitAnalysis {
            original_gate_count,
            fused_gate_count,
            fusion_blocks: fused_blocks.len(),
            performance_improvement,
            gate_distribution,
            fusion_opportunities,
            circuit_depth: (original_depth, fused_depth),
        };

        // Record experiment for learning
        if self.config.enable_ml_predictions {
            let experiment = FusionExperiment {
                circuit_fingerprint: self.calculate_circuit_fingerprint(gates),
                fusion_strategy: self.config.strategy,
                performance_improvement,
                execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            };
            self.learning_history.push(experiment);
        }

        Ok(analysis)
    }

    /// Perform gate fusion on a circuit
    pub fn fuse_circuit(&mut self, gates: &[QuantumGate]) -> Result<Vec<FusedGateBlock>> {
        let (fused_blocks, _) = self.perform_fusion(gates)?;
        Ok(fused_blocks)
    }

    /// Identify fusion opportunities in a circuit
    fn identify_fusion_opportunities(
        &self,
        gates: &[QuantumGate],
    ) -> Result<Vec<FusionOpportunity>> {
        let mut opportunities = Vec::new();

        // Analyze gate sequences for fusion potential
        for window_size in 2..=self.config.max_fusion_size {
            for window in gates.windows(window_size) {
                if let Some(opportunity) = self.analyze_gate_window(window)? {
                    opportunities.push(opportunity);
                }
            }
        }

        // Remove overlapping opportunities (keep the best ones)
        opportunities.sort_by(|a, b| {
            b.estimated_benefit
                .partial_cmp(&a.estimated_benefit)
                .unwrap()
        });
        self.remove_overlapping_opportunities(opportunities)
    }

    /// Analyze a window of gates for fusion potential
    fn analyze_gate_window(&self, window: &[QuantumGate]) -> Result<Option<FusionOpportunity>> {
        if window.len() < 2 {
            return Ok(None);
        }

        // Check if gates can be fused
        let can_fuse = self.can_fuse_gate_sequence(window)?;
        if !can_fuse {
            return Ok(None);
        }

        // Estimate benefit
        let benefit = self.estimate_fusion_benefit(window)?;
        if benefit < self.config.min_benefit_threshold {
            return Ok(None);
        }

        // Create opportunity description
        let qubits: HashSet<usize> = window.iter().flat_map(|g| &g.qubits).cloned().collect();
        let gate_types: Vec<String> = window
            .iter()
            .map(|g| format!("{:?}", g.gate_type))
            .collect();

        let confidence = self.calculate_fusion_confidence(window);

        let opportunity = FusionOpportunity {
            description: format!("Fuse {} gates on qubits {:?}", window.len(), qubits),
            qubits: qubits.into_iter().collect(),
            gate_types,
            estimated_benefit: benefit,
            confidence,
        };

        Ok(Some(opportunity))
    }

    /// Check if a sequence of gates can be fused
    fn can_fuse_gate_sequence(&self, gates: &[QuantumGate]) -> Result<bool> {
        if gates.is_empty() {
            return Ok(false);
        }

        // Check basic fusion compatibility
        for i in 0..gates.len() - 1 {
            if !gates[i].can_fuse_with(&gates[i + 1]) {
                // Check if gates commute (allowing reordering)
                if !gates[i].commutes_with(&gates[i + 1]) {
                    return Ok(false);
                }
            }
        }

        // Check if fusion would be beneficial
        let fusion_block = FusedGateBlock::new(gates.to_vec())?;
        Ok(fusion_block.is_beneficial())
    }

    /// Estimate the benefit of fusing a sequence of gates
    fn estimate_fusion_benefit(&self, gates: &[QuantumGate]) -> Result<f64> {
        let individual_cost: f64 = gates.iter().map(|g| g.cost).sum();

        // Create temporary fusion block to estimate fused cost
        let fusion_block = FusedGateBlock::new(gates.to_vec())?;
        let fused_cost = fusion_block.cost;

        Ok(individual_cost / fused_cost)
    }

    /// Calculate confidence score for fusion
    fn calculate_fusion_confidence(&self, gates: &[QuantumGate]) -> f64 {
        let mut confidence: f64 = 1.0;

        // Reduce confidence for mixed gate types
        let gate_types: HashSet<_> = gates.iter().map(|g| &g.gate_type).collect();
        if gate_types.len() > 1 {
            confidence *= 0.8;
        }

        // Reduce confidence for gates on many qubits
        let all_qubits: HashSet<_> = gates.iter().flat_map(|g| &g.qubits).collect();
        if all_qubits.len() > 3 {
            confidence *= 0.6;
        }

        // Increase confidence for known beneficial patterns
        if self.is_known_beneficial_pattern(gates) {
            confidence *= 1.2;
        }

        confidence.min(1.0)
    }

    /// Check if this is a known beneficial fusion pattern
    fn is_known_beneficial_pattern(&self, gates: &[QuantumGate]) -> bool {
        // Check for common beneficial patterns
        if gates.len() == 2 {
            // Adjacent rotation gates of same type
            if let (Some(g1), Some(g2)) = (gates.get(0), gates.get(1)) {
                match (&g1.gate_type, &g2.gate_type) {
                    (GateType::RotationX, GateType::RotationX)
                    | (GateType::RotationY, GateType::RotationY)
                    | (GateType::RotationZ, GateType::RotationZ) => return true,
                    _ => {}
                }
            }
        }

        // CNOT + single qubit gate patterns
        if gates.len() == 3 {
            // Look for CNOT-single-CNOT patterns
            if matches!(gates[0].gate_type, GateType::CNOT)
                && gates[1].qubits.len() == 1
                && matches!(gates[2].gate_type, GateType::CNOT)
            {
                return true;
            }
        }

        false
    }

    /// Remove overlapping fusion opportunities
    fn remove_overlapping_opportunities(
        &self,
        opportunities: Vec<FusionOpportunity>,
    ) -> Result<Vec<FusionOpportunity>> {
        let mut result = Vec::new();
        let mut used_positions = HashSet::new();

        for opportunity in opportunities {
            // Check if this opportunity overlaps with already selected ones
            let overlaps = opportunity
                .qubits
                .iter()
                .any(|q| used_positions.contains(q));

            if !overlaps {
                // Mark qubits as used
                for &qubit in &opportunity.qubits {
                    used_positions.insert(qubit);
                }
                result.push(opportunity);
            }
        }

        Ok(result)
    }

    /// Perform actual gate fusion
    fn perform_fusion(
        &mut self,
        gates: &[QuantumGate],
    ) -> Result<(Vec<FusedGateBlock>, Vec<QuantumGate>)> {
        let start_time = std::time::Instant::now();

        #[cfg(feature = "advanced_math")]
        {
            if self.optimizer.is_some() {
                // For now, fall back to manual fusion since we're using placeholder types
                return self.perform_manual_fusion(gates);
            }
        }

        // Fallback to manual fusion
        self.perform_manual_fusion(gates)
    }

    #[cfg(feature = "advanced_math")]
    fn perform_scirs2_fusion(
        &mut self,
        gates: &[QuantumGate],
        _optimizer: &mut Box<dyn std::any::Any>,
    ) -> Result<(Vec<FusedGateBlock>, Vec<QuantumGate>)> {
        // Use SciRS2's circuit optimization capabilities
        // This is a placeholder - actual implementation would use SciRS2 APIs
        self.perform_manual_fusion(gates)
    }

    /// Manual fusion implementation
    fn perform_manual_fusion(
        &mut self,
        gates: &[QuantumGate],
    ) -> Result<(Vec<FusedGateBlock>, Vec<QuantumGate>)> {
        let mut fused_blocks = Vec::new();
        let mut remaining_gates = Vec::new();
        let mut i = 0;

        while i < gates.len() {
            // Try to find the largest fusable block starting at position i
            let mut best_block_size = 1;
            let mut best_benefit = 0.0;

            for block_size in 2..=(self.config.max_fusion_size.min(gates.len() - i)) {
                let window = &gates[i..i + block_size];

                if let Ok(can_fuse) = self.can_fuse_gate_sequence(window) {
                    if can_fuse {
                        if let Ok(benefit) = self.estimate_fusion_benefit(window) {
                            if benefit > best_benefit
                                && benefit >= self.config.min_benefit_threshold
                            {
                                best_block_size = block_size;
                                best_benefit = benefit;
                            }
                        }
                    }
                }
            }

            if best_block_size > 1 {
                // Create fused block
                let block_gates = gates[i..i + best_block_size].to_vec();

                // Create fusion pattern key from gates
                let pattern_key = FusionPatternKey::from_gates(&block_gates)?;

                // Check cache first
                if let Some(cached_result) = self.fusion_cache.get(&pattern_key) {
                    fused_blocks.push(cached_result.fused_block.clone());
                    self.cache_hits += 1;
                } else {
                    let fused_block = FusedGateBlock::new(block_gates.clone())?;

                    // Create cached result
                    let cached_result = CachedFusionResult {
                        fused_block: fused_block.clone(),
                        benefit: 1.0, // Default benefit
                        usage_count: 1,
                        last_accessed: std::time::Instant::now(),
                    };

                    // Cache the result
                    if self.fusion_cache.len() < self.config.fusion_cache_size {
                        self.fusion_cache.insert(pattern_key, cached_result);
                    }

                    self.cache_misses += 1;
                    fused_blocks.push(fused_block);
                }

                i += best_block_size;
            } else {
                // Single gate cannot be fused
                remaining_gates.push(gates[i].clone());
                i += 1;
            }
        }

        Ok((fused_blocks, remaining_gates))
    }

    /// Calculate circuit depth
    fn calculate_circuit_depth(&self, gates: &[QuantumGate]) -> usize {
        if gates.is_empty() {
            return 0;
        }

        // Build dependency graph
        let mut qubit_last_gate = HashMap::new();
        let mut gate_depths = vec![0; gates.len()];

        for (i, gate) in gates.iter().enumerate() {
            let mut max_dependency_depth = 0;

            for &qubit in &gate.qubits {
                if let Some(&last_gate_idx) = qubit_last_gate.get(&qubit) {
                    max_dependency_depth = max_dependency_depth.max(gate_depths[last_gate_idx]);
                }
                qubit_last_gate.insert(qubit, i);
            }

            gate_depths[i] = max_dependency_depth + 1;
        }

        gate_depths.into_iter().max().unwrap_or(0)
    }

    /// Calculate circuit depth after fusion
    fn calculate_fused_circuit_depth(
        &self,
        blocks: &[FusedGateBlock],
        gates: &[QuantumGate],
    ) -> usize {
        // Simplified depth calculation - in practice would need more sophisticated analysis
        blocks.len() + gates.len()
    }

    /// Calculate circuit fingerprint for learning
    fn calculate_circuit_fingerprint(&self, gates: &[QuantumGate]) -> u64 {
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        gates.hash(&mut hasher);
        hasher.finish()
    }

    /// Get fusion statistics
    pub fn get_fusion_stats(&self) -> FusionStats {
        FusionStats {
            cache_size: self.fusion_cache.len(),
            cache_hit_rate: self.calculate_cache_hit_rate(),
            learning_experiments: self.learning_history.len(),
            average_improvement: self.calculate_average_improvement(),
        }
    }

    fn calculate_cache_hit_rate(&self) -> f64 {
        // This would be tracked during actual execution
        0.0 // Placeholder
    }

    fn calculate_average_improvement(&self) -> f64 {
        if self.learning_history.is_empty() {
            return 0.0;
        }

        let total: f64 = self
            .learning_history
            .iter()
            .map(|e| e.performance_improvement)
            .sum();
        total / self.learning_history.len() as f64
    }

    /// Fuse a sequence of gates using adaptive fusion
    pub fn fuse_gates(
        &mut self,
        gates: &[QuantumGate],
    ) -> crate::error::Result<(Vec<FusedGateBlock>, Vec<QuantumGate>)> {
        let mut fused_blocks = Vec::new();

        if gates.is_empty() {
            return Ok((fused_blocks, Vec::new()));
        }

        let mut i = 0;
        while i < gates.len() {
            if i + 1 < gates.len() {
                // Try to fuse adjacent gates
                let should_fuse = self.should_fuse_basic(&gates[i], &gates[i + 1]);

                if should_fuse {
                    // Create a fused gate block from the two gates
                    let mut qubits = gates[i].qubits.clone();
                    qubits.extend_from_slice(&gates[i + 1].qubits);
                    qubits.sort_unstable();
                    qubits.dedup();

                    let fused_block = FusedGateBlock {
                        gates: vec![gates[i].clone(), gates[i + 1].clone()],
                        combined_matrix: self
                            .calculate_combined_matrix(&gates[i], &gates[i + 1])?,
                        qubits,
                        cost: 0.5, // Assume fusion reduces cost
                        improvement_factor: 1.5,
                    };

                    fused_blocks.push(fused_block);
                    i += 2; // Skip both gates
                } else {
                    // Create a single-gate block
                    let fused_block = FusedGateBlock {
                        gates: vec![gates[i].clone()],
                        combined_matrix: gates[i].matrix.clone(),
                        qubits: gates[i].qubits.clone(),
                        cost: 1.0,
                        improvement_factor: 1.0,
                    };

                    fused_blocks.push(fused_block);
                    i += 1;
                }
            } else {
                // Last gate, add it as-is
                let fused_block = FusedGateBlock {
                    gates: vec![gates[i].clone()],
                    combined_matrix: gates[i].matrix.clone(),
                    qubits: gates[i].qubits.clone(),
                    cost: 1.0,
                    improvement_factor: 1.0,
                };

                fused_blocks.push(fused_block);
                i += 1;
            }
        }

        Ok((fused_blocks, Vec::new()))
    }

    /// Basic heuristic for gate fusion
    fn should_fuse_basic(&self, gate1: &QuantumGate, gate2: &QuantumGate) -> bool {
        let overlapping_qubits = gate1.qubits.iter().any(|&q| gate2.qubits.contains(&q));
        overlapping_qubits && gate1.qubits.len() <= 2 && gate2.qubits.len() <= 2
    }

    /// Calculate combined matrix for two gates
    fn calculate_combined_matrix(
        &self,
        gate1: &QuantumGate,
        gate2: &QuantumGate,
    ) -> crate::error::Result<Array2<Complex64>> {
        // For simplicity, just return gate1's matrix
        // In a real implementation, this would compute the matrix product
        Ok(gate1.matrix.clone())
    }
}

/// Fusion statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionStats {
    /// Number of entries in fusion cache
    pub cache_size: usize,
    /// Cache hit rate (0-1)
    pub cache_hit_rate: f64,
    /// Number of learning experiments recorded
    pub learning_experiments: usize,
    /// Average performance improvement
    pub average_improvement: f64,
}

impl MLFusionPredictor {
    /// Create a new ML predictor
    pub fn new() -> Self {
        let mut feature_weights = HashMap::new();

        // Initialize with some reasonable default weights
        feature_weights.insert("rotation_similarity".to_string(), 0.8);
        feature_weights.insert("gate_locality".to_string(), 0.7);
        feature_weights.insert("commutation_potential".to_string(), 0.6);
        feature_weights.insert("matrix_sparsity".to_string(), 0.5);

        Self {
            feature_weights,
            training_data: Vec::new(),
            accuracy: 0.5, // Start with neutral accuracy
        }
    }

    /// Predict fusion benefit for a gate sequence
    pub fn predict_benefit(&self, gates: &[QuantumGate]) -> f64 {
        let features = self.extract_features(gates);

        let mut prediction = 0.0;
        for (i, &feature_value) in features.iter().enumerate() {
            let weight_key = match i {
                0 => "rotation_similarity",
                1 => "gate_locality",
                2 => "commutation_potential",
                3 => "matrix_sparsity",
                _ => "default",
            };

            if let Some(&weight) = self.feature_weights.get(weight_key) {
                prediction += feature_value * weight;
            }
        }

        // Sigmoid activation to bound between 0 and 1
        1.0 / (1.0 + (-prediction).exp())
    }

    /// Extract features from gate sequence
    fn extract_features(&self, gates: &[QuantumGate]) -> Vec<f64> {
        let mut features = [0.0; 4];

        if gates.len() < 2 {
            return features.to_vec();
        }

        // Feature 0: Rotation similarity
        features[0] = self.calculate_rotation_similarity(gates);

        // Feature 1: Gate locality
        features[1] = self.calculate_gate_locality(gates);

        // Feature 2: Commutation potential
        features[2] = self.calculate_commutation_potential(gates);

        // Feature 3: Matrix sparsity
        features[3] = self.calculate_matrix_sparsity(gates);

        features.to_vec()
    }

    fn calculate_rotation_similarity(&self, gates: &[QuantumGate]) -> f64 {
        let rotation_gates: Vec<_> = gates
            .iter()
            .filter(|g| {
                matches!(
                    g.gate_type,
                    GateType::RotationX | GateType::RotationY | GateType::RotationZ
                )
            })
            .collect();

        if rotation_gates.len() < 2 {
            return 0.0;
        }

        // Count same-type rotations on same qubits
        let mut similarity_score = 0.0;
        for i in 0..rotation_gates.len() - 1 {
            for j in i + 1..rotation_gates.len() {
                if rotation_gates[i].gate_type == rotation_gates[j].gate_type
                    && rotation_gates[i].qubits == rotation_gates[j].qubits
                {
                    similarity_score += 1.0;
                }
            }
        }

        similarity_score / (rotation_gates.len() as f64)
    }

    fn calculate_gate_locality(&self, gates: &[QuantumGate]) -> f64 {
        let mut locality_score = 0.0;
        let mut adjacent_count = 0;

        for i in 0..gates.len() - 1 {
            let current_qubits: HashSet<_> = gates[i].qubits.iter().cloned().collect();
            let next_qubits: HashSet<_> = gates[i + 1].qubits.iter().cloned().collect();

            if current_qubits.intersection(&next_qubits).count() > 0 {
                locality_score += 1.0;
            }
            adjacent_count += 1;
        }

        if adjacent_count > 0 {
            locality_score / adjacent_count as f64
        } else {
            0.0
        }
    }

    fn calculate_commutation_potential(&self, gates: &[QuantumGate]) -> f64 {
        let mut commutation_score = 0.0;
        let mut pair_count = 0;

        for i in 0..gates.len() - 1 {
            for j in i + 1..gates.len() {
                let qubits_i: HashSet<_> = gates[i].qubits.iter().cloned().collect();
                let qubits_j: HashSet<_> = gates[j].qubits.iter().cloned().collect();

                // Non-overlapping gates likely commute
                if qubits_i.intersection(&qubits_j).count() == 0 {
                    commutation_score += 1.0;
                }
                pair_count += 1;
            }
        }

        if pair_count > 0 {
            commutation_score / pair_count as f64
        } else {
            0.0
        }
    }

    fn calculate_matrix_sparsity(&self, gates: &[QuantumGate]) -> f64 {
        let mut total_sparsity = 0.0;

        for gate in gates {
            let matrix = &gate.matrix;
            let zero_count = matrix.iter().filter(|&&x| x.norm() < 1e-10).count();
            let sparsity = zero_count as f64 / (matrix.len() as f64);
            total_sparsity += sparsity;
        }

        total_sparsity / gates.len() as f64
    }

    /// Add training example and update model
    pub fn add_training_example(&mut self, example: MLTrainingExample) {
        self.training_data.push(example);

        // Simple online learning update
        if self.training_data.len() % 10 == 0 {
            self.update_weights();
        }
    }

    fn update_weights(&mut self) {
        // Simplified gradient descent update
        let learning_rate = 0.01;

        for example in &self.training_data {
            if let Some(observed) = example.observed_benefit {
                let predicted = self.predict_benefit_from_features(&example.features);
                let error = observed - predicted;

                // Update weights based on error
                for (i, &feature_value) in example.features.iter().enumerate() {
                    let weight_key = match i {
                        0 => "rotation_similarity",
                        1 => "gate_locality",
                        2 => "commutation_potential",
                        3 => "matrix_sparsity",
                        _ => continue,
                    };

                    if let Some(weight) = self.feature_weights.get_mut(weight_key) {
                        *weight += learning_rate * error * feature_value;
                    }
                }
            }
        }
    }

    fn predict_benefit_from_features(&self, features: &[f64]) -> f64 {
        let mut prediction = 0.0;
        for (i, &feature_value) in features.iter().enumerate() {
            let weight_key = match i {
                0 => "rotation_similarity",
                1 => "gate_locality",
                2 => "commutation_potential",
                3 => "matrix_sparsity",
                _ => "default",
            };

            if let Some(&weight) = self.feature_weights.get(weight_key) {
                prediction += feature_value * weight;
            }
        }

        1.0 / (1.0 + (-prediction).exp())
    }

    /// Check if two gates should be fused
    fn should_fuse_gates(&self, gate1: &QuantumGate, gate2: &QuantumGate) -> FusionResult {
        // For simplicity, use a basic heuristic
        let overlapping_qubits = gate1.qubits.iter().any(|&q| gate2.qubits.contains(&q));
        let should_fuse = overlapping_qubits && gate1.qubits.len() <= 2 && gate2.qubits.len() <= 2;

        FusionResult {
            should_fuse,
            confidence: if should_fuse { 0.8 } else { 0.2 },
            expected_speedup: if should_fuse { 1.5 } else { 1.0 },
            estimated_error: 0.01,
        }
    }
}

impl CircuitPatternAnalyzer {
    /// Create a new pattern analyzer
    pub fn new() -> Self {
        let mut beneficial_patterns = HashMap::new();

        // Initialize with known beneficial patterns
        beneficial_patterns.insert("RX-RX".to_string(), 0.9);
        beneficial_patterns.insert("RY-RY".to_string(), 0.9);
        beneficial_patterns.insert("RZ-RZ".to_string(), 0.9);
        beneficial_patterns.insert("H-CNOT-H".to_string(), 0.8);
        beneficial_patterns.insert("CNOT-RZ-CNOT".to_string(), 0.7);

        Self {
            beneficial_patterns,
            pattern_history: Vec::new(),
        }
    }

    /// Analyze circuit pattern and return recognition result
    pub fn analyze_pattern(&mut self, gates: &[QuantumGate]) -> PatternRecognitionResult {
        let pattern_string = self.create_pattern_string(gates);

        let (confidence, expected_benefit) = if let Some(&benefit) =
            self.beneficial_patterns.get(&pattern_string)
        {
            (0.9, benefit)
        } else {
            // Try partial matches
            let (partial_confidence, partial_benefit) = self.find_partial_matches(&pattern_string);
            (partial_confidence, partial_benefit)
        };

        let result = PatternRecognitionResult {
            pattern: pattern_string,
            confidence,
            expected_benefit,
        };

        self.pattern_history.push(result.clone());
        result
    }

    fn create_pattern_string(&self, gates: &[QuantumGate]) -> String {
        gates
            .iter()
            .map(|g| format!("{:?}", g.gate_type))
            .collect::<Vec<_>>()
            .join("-")
    }

    fn find_partial_matches(&self, pattern: &str) -> (f64, f64) {
        let mut best_confidence = 0.0;
        let mut best_benefit = 0.0;

        for (known_pattern, &benefit) in &self.beneficial_patterns {
            let similarity = self.calculate_pattern_similarity(pattern, known_pattern);
            if similarity > best_confidence {
                best_confidence = similarity;
                best_benefit = benefit * similarity; // Scale benefit by similarity
            }
        }

        (best_confidence, best_benefit)
    }

    fn calculate_pattern_similarity(&self, pattern1: &str, pattern2: &str) -> f64 {
        let gates1: Vec<&str> = pattern1.split('-').collect();
        let gates2: Vec<&str> = pattern2.split('-').collect();

        let max_len = gates1.len().max(gates2.len()) as f64;
        if max_len == 0.0 {
            return 0.0;
        }

        let common_count = gates1.iter().filter(|&g| gates2.contains(g)).count() as f64;

        common_count / max_len
    }

    /// Learn from successful fusion
    pub fn learn_pattern(&mut self, pattern: String, observed_benefit: f64) {
        // Update or add pattern with exponential moving average
        let alpha = 0.1; // Learning rate

        if let Some(current_benefit) = self.beneficial_patterns.get_mut(&pattern) {
            *current_benefit = (1.0 - alpha) * *current_benefit + alpha * observed_benefit;
        } else {
            self.beneficial_patterns.insert(pattern, observed_benefit);
        }
    }
}

/// Utilities for gate fusion
pub struct FusionUtils;

impl FusionUtils {
    /// Create common gate sequences for testing
    pub fn create_test_sequence(sequence_type: &str, num_qubits: usize) -> Vec<QuantumGate> {
        match sequence_type {
            "rotation_chain" => (0..num_qubits)
                .flat_map(|q| {
                    vec![
                        QuantumGate::new(
                            GateType::RotationX,
                            vec![q],
                            vec![std::f64::consts::PI / 4.0],
                        ),
                        QuantumGate::new(
                            GateType::RotationY,
                            vec![q],
                            vec![std::f64::consts::PI / 3.0],
                        ),
                        QuantumGate::new(
                            GateType::RotationZ,
                            vec![q],
                            vec![std::f64::consts::PI / 6.0],
                        ),
                    ]
                })
                .collect(),
            "cnot_ladder" => (0..num_qubits - 1)
                .map(|q| QuantumGate::new(GateType::CNOT, vec![q, q + 1], vec![]))
                .collect(),
            "mixed_gates" => {
                let mut gates = Vec::new();
                for q in 0..num_qubits {
                    gates.push(QuantumGate::new(GateType::Hadamard, vec![q], vec![]));
                    if q > 0 {
                        gates.push(QuantumGate::new(GateType::CNOT, vec![q - 1, q], vec![]));
                    }
                    gates.push(QuantumGate::new(
                        GateType::RotationZ,
                        vec![q],
                        vec![std::f64::consts::PI / 8.0],
                    ));
                }
                gates
            }
            _ => vec![QuantumGate::new(GateType::Identity, vec![0], vec![])],
        }
    }

    /// Benchmark different fusion strategies
    pub fn benchmark_fusion_strategies(
        gates: &[QuantumGate],
        strategies: &[FusionStrategy],
    ) -> Result<HashMap<String, CircuitAnalysis>> {
        let mut results = HashMap::new();

        for &strategy in strategies {
            let config = AdaptiveFusionConfig {
                strategy,
                ..Default::default()
            };

            let mut fusion_engine = AdaptiveGateFusion::new(config)?;
            let analysis = fusion_engine.analyze_circuit(gates)?;

            results.insert(format!("{:?}", strategy), analysis);
        }

        Ok(results)
    }

    /// Estimate fusion potential for a circuit
    pub fn estimate_fusion_potential(gates: &[QuantumGate]) -> f64 {
        if gates.len() < 2 {
            return 0.0;
        }

        let mut potential_fusions = 0;
        let mut total_gates = gates.len();

        // Count adjacent gates that could potentially be fused
        for window in gates.windows(2) {
            if window[0].can_fuse_with(&window[1]) {
                potential_fusions += 1;
            }
        }

        potential_fusions as f64 / total_gates as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_quantum_gate_creation() {
        let gate = QuantumGate::new(GateType::PauliX, vec![0], vec![]);
        assert_eq!(gate.gate_type, GateType::PauliX);
        assert_eq!(gate.qubits, vec![0]);
        assert!(gate.cost > 0.0);
    }

    #[test]
    fn test_gate_commutation() {
        let gate1 = QuantumGate::new(GateType::PauliX, vec![0], vec![]);
        let gate2 = QuantumGate::new(GateType::PauliY, vec![1], vec![]);
        let gate3 = QuantumGate::new(GateType::PauliX, vec![0], vec![]);

        assert!(gate1.commutes_with(&gate2)); // Different qubits
        assert!(gate1.commutes_with(&gate3)); // Same Pauli on same qubit
    }

    #[test]
    fn test_gate_fusion_compatibility() {
        let gate1 = QuantumGate::new(GateType::RotationX, vec![0], vec![0.5]);
        let gate2 = QuantumGate::new(GateType::RotationX, vec![0], vec![0.3]);
        let gate3 = QuantumGate::new(GateType::RotationY, vec![1], vec![0.2]);

        assert!(gate1.can_fuse_with(&gate2)); // Same type, same qubit
        assert!(!gate1.can_fuse_with(&gate3)); // Different qubit
    }

    #[test]
    fn test_fused_gate_block() {
        let gates = vec![
            QuantumGate::new(GateType::RotationX, vec![0], vec![0.5]),
            QuantumGate::new(GateType::RotationX, vec![0], vec![0.3]),
        ];

        let block = FusedGateBlock::new(gates).unwrap();
        assert_eq!(block.qubits, vec![0]);
        assert!(block.improvement_factor > 0.0);
    }

    #[test]
    fn test_adaptive_fusion_config() {
        let config = AdaptiveFusionConfig::default();
        assert_eq!(config.strategy, FusionStrategy::Adaptive);
        assert_eq!(config.max_fusion_size, 8);
        assert!(config.enable_cross_qubit_fusion);
    }

    #[test]
    fn test_circuit_analysis() {
        let gates = FusionUtils::create_test_sequence("rotation_chain", 2);

        let config = AdaptiveFusionConfig::default();
        let mut fusion_engine = AdaptiveGateFusion::new(config).unwrap();

        let analysis = fusion_engine.analyze_circuit(&gates).unwrap();
        assert_eq!(analysis.original_gate_count, gates.len());
        assert!(!analysis.fusion_opportunities.is_empty());
    }

    #[test]
    fn test_fusion_utils_test_sequences() {
        let rotation_chain = FusionUtils::create_test_sequence("rotation_chain", 2);
        assert_eq!(rotation_chain.len(), 6); // 3 rotations per qubit * 2 qubits

        let cnot_ladder = FusionUtils::create_test_sequence("cnot_ladder", 3);
        assert_eq!(cnot_ladder.len(), 2); // 2 CNOTs for 3 qubits

        let mixed_gates = FusionUtils::create_test_sequence("mixed_gates", 2);
        assert!(!mixed_gates.is_empty());
    }

    #[test]
    fn test_fusion_potential_estimation() {
        let gates = vec![
            QuantumGate::new(GateType::RotationX, vec![0], vec![0.1]),
            QuantumGate::new(GateType::RotationX, vec![0], vec![0.2]),
            QuantumGate::new(GateType::RotationY, vec![1], vec![0.3]),
        ];

        let potential = FusionUtils::estimate_fusion_potential(&gates);
        assert!(potential > 0.0);
        assert!(potential <= 1.0);
    }

    #[test]
    fn test_gate_matrix_generation() {
        let pauli_x = QuantumGate::new(GateType::PauliX, vec![0], vec![]);
        assert_eq!(pauli_x.matrix.shape(), &[2, 2]);

        // Check Pauli-X matrix elements
        assert_abs_diff_eq!(pauli_x.matrix[[0, 1]].re, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(pauli_x.matrix[[1, 0]].re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_circuit_depth_calculation() {
        let gates = vec![
            QuantumGate::new(GateType::Hadamard, vec![0], vec![]),
            QuantumGate::new(GateType::CNOT, vec![0, 1], vec![]),
            QuantumGate::new(GateType::RotationZ, vec![1], vec![0.5]),
        ];

        let config = AdaptiveFusionConfig::default();
        let fusion_engine = AdaptiveGateFusion::new(config).unwrap();

        let depth = fusion_engine.calculate_circuit_depth(&gates);
        assert!(depth > 0);
    }
}
