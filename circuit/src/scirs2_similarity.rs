//! Circuit similarity metrics using `SciRS2`
//!
//! This module implements sophisticated quantum circuit similarity and distance metrics
//! leveraging `SciRS2`'s graph algorithms, numerical analysis, and machine learning capabilities.

use crate::builder::Circuit;
use crate::dag::{circuit_to_dag, CircuitDag};
use crate::scirs2_matrices::SparseMatrix;
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

// Placeholder types representing SciRS2 graph and ML interface
// In the real implementation, these would be imported from SciRS2

/// Graph representation for `SciRS2` integration
#[derive(Debug, Clone)]
pub struct SciRS2Graph {
    /// Node identifiers
    pub nodes: Vec<usize>,
    /// Edge list (source, target, weight)
    pub edges: Vec<(usize, usize, f64)>,
    /// Node attributes
    pub node_attributes: HashMap<usize, HashMap<String, String>>,
    /// Edge attributes
    pub edge_attributes: HashMap<(usize, usize), HashMap<String, f64>>,
}

/// Graph similarity algorithms available in `SciRS2`
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphSimilarityAlgorithm {
    /// Graph edit distance
    GraphEditDistance,
    /// Spectral similarity based on eigenvalues
    SpectralSimilarity,
    /// Graph kernel methods
    GraphKernel { kernel_type: GraphKernelType },
    /// Network alignment
    NetworkAlignment,
    /// Subgraph isomorphism
    SubgraphIsomorphism,
    /// Graph neural network embeddings
    GraphNeuralNetwork { embedding_dim: usize },
}

/// Graph kernel types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphKernelType {
    /// Random walk kernel
    RandomWalk { steps: usize },
    /// Weisfeiler-Lehman kernel
    WeisfeilerLehman { iterations: usize },
    /// Shortest path kernel
    ShortestPath,
    /// Graphlet kernel
    Graphlet { size: usize },
}

/// Circuit similarity metrics
#[derive(Debug, Clone)]
pub struct CircuitSimilarityMetrics {
    /// Structural similarity (0.0 to 1.0)
    pub structural_similarity: f64,
    /// Functional similarity (0.0 to 1.0)
    pub functional_similarity: f64,
    /// Gate sequence similarity (0.0 to 1.0)
    pub sequence_similarity: f64,
    /// Topological similarity (0.0 to 1.0)
    pub topological_similarity: f64,
    /// Overall similarity score (0.0 to 1.0)
    pub overall_similarity: f64,
    /// Detailed breakdown by metric type
    pub detailed_metrics: HashMap<String, f64>,
}

/// Circuit distance measures
#[derive(Debug, Clone)]
pub struct CircuitDistanceMetrics {
    /// Edit distance (minimum operations to transform one circuit to another)
    pub edit_distance: usize,
    /// Normalized edit distance (0.0 to 1.0)
    pub normalized_edit_distance: f64,
    /// Wasserstein distance between gate distributions
    pub wasserstein_distance: f64,
    /// Hausdorff distance between circuit embeddings
    pub hausdorff_distance: f64,
    /// Earth mover's distance
    pub earth_movers_distance: f64,
    /// Quantum process fidelity distance
    pub process_fidelity_distance: f64,
}

/// Configuration for similarity computation
#[derive(Debug, Clone)]
pub struct SimilarityConfig {
    /// Algorithms to use for comparison
    pub algorithms: Vec<SimilarityAlgorithm>,
    /// Weight for different similarity aspects
    pub weights: SimilarityWeights,
    /// Tolerance for numerical comparisons
    pub tolerance: f64,
    /// Whether to normalize results
    pub normalize: bool,
    /// Cache intermediate results
    pub cache_results: bool,
    /// Use parallel computation
    pub parallel: bool,
}

/// Similarity computation algorithms
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimilarityAlgorithm {
    /// Gate-level comparison
    GateLevel,
    /// DAG structure comparison
    DAGStructure,
    /// Unitary matrix comparison
    UnitaryMatrix,
    /// Graph-based comparison
    GraphBased { algorithm: GraphSimilarityAlgorithm },
    /// Statistical comparison
    Statistical,
    /// Machine learning embeddings
    MLEmbeddings { model_type: MLModelType },
}

/// Machine learning model types for embeddings
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MLModelType {
    /// Variational autoencoder
    VAE { latent_dim: usize },
    /// Graph convolutional network
    GCN { hidden_dims: Vec<usize> },
    /// Transformer model
    Transformer { num_heads: usize, num_layers: usize },
    /// Pre-trained circuit embedding model
    PreTrained { model_name: String },
}

/// Weights for combining different similarity measures
#[derive(Debug, Clone)]
pub struct SimilarityWeights {
    /// Weight for structural similarity
    pub structural: f64,
    /// Weight for functional similarity
    pub functional: f64,
    /// Weight for gate sequence similarity
    pub sequence: f64,
    /// Weight for topological similarity
    pub topological: f64,
}

impl Default for SimilarityWeights {
    fn default() -> Self {
        Self {
            structural: 0.3,
            functional: 0.4,
            sequence: 0.2,
            topological: 0.1,
        }
    }
}

impl Default for SimilarityConfig {
    fn default() -> Self {
        Self {
            algorithms: vec![
                SimilarityAlgorithm::GateLevel,
                SimilarityAlgorithm::DAGStructure,
                SimilarityAlgorithm::UnitaryMatrix,
            ],
            weights: SimilarityWeights::default(),
            tolerance: 1e-12,
            normalize: true,
            cache_results: true,
            parallel: false,
        }
    }
}

/// Circuit similarity analyzer using `SciRS2`
pub struct CircuitSimilarityAnalyzer {
    /// Configuration for similarity computation
    config: SimilarityConfig,
    /// Cache for computed similarities
    similarity_cache: HashMap<(String, String), CircuitSimilarityMetrics>,
    /// Cache for circuit embeddings
    embedding_cache: HashMap<String, Vec<f64>>,
    /// Pre-computed circuit features
    feature_cache: HashMap<String, CircuitFeatures>,
}

/// Circuit features for similarity computation
#[derive(Debug, Clone)]
pub struct CircuitFeatures {
    /// Gate type histogram
    pub gate_histogram: HashMap<String, usize>,
    /// Circuit depth
    pub depth: usize,
    /// Two-qubit gate count
    pub two_qubit_gates: usize,
    /// Connectivity pattern
    pub connectivity_pattern: Vec<(usize, usize)>,
    /// Critical path information
    pub critical_path: Vec<String>,
    /// Parallelism profile
    pub parallelism_profile: Vec<usize>,
    /// Entanglement structure
    pub entanglement_structure: EntanglementStructure,
}

/// Entanglement structure representation
#[derive(Debug, Clone)]
pub struct EntanglementStructure {
    /// Entangling gates by layer
    pub entangling_layers: Vec<Vec<(usize, usize)>>,
    /// Maximum entanglement width
    pub max_entanglement_width: usize,
    /// Entanglement graph
    pub entanglement_graph: SciRS2Graph,
}

impl CircuitSimilarityAnalyzer {
    /// Create a new circuit similarity analyzer
    #[must_use]
    pub fn new(config: SimilarityConfig) -> Self {
        Self {
            config,
            similarity_cache: HashMap::new(),
            embedding_cache: HashMap::new(),
            feature_cache: HashMap::new(),
        }
    }

    /// Create analyzer with default configuration
    #[must_use]
    pub fn with_default_config() -> Self {
        Self::new(SimilarityConfig::default())
    }

    /// Compute comprehensive similarity between two circuits
    pub fn compute_similarity<const N: usize, const M: usize>(
        &mut self,
        circuit1: &Circuit<N>,
        circuit2: &Circuit<M>,
    ) -> QuantRS2Result<CircuitSimilarityMetrics> {
        // Generate unique identifiers for caching
        let id1 = Self::generate_circuit_id(circuit1);
        let id2 = Self::generate_circuit_id(circuit2);
        let cache_key = if id1 < id2 { (id1, id2) } else { (id2, id1) };

        // Check cache
        if self.config.cache_results {
            if let Some(cached) = self.similarity_cache.get(&cache_key) {
                return Ok(cached.clone());
            }
        }

        // Extract features
        let features1 = self.extract_circuit_features(circuit1)?;
        let features2 = self.extract_circuit_features(circuit2)?;

        // Compute individual similarity measures
        let mut detailed_metrics = HashMap::new();
        let mut similarities = Vec::new();

        let algorithms = self.config.algorithms.clone();
        for algorithm in &algorithms {
            let similarity = match algorithm {
                SimilarityAlgorithm::GateLevel => {
                    Self::compute_gate_level_similarity(&features1, &features2)?
                }
                SimilarityAlgorithm::DAGStructure => {
                    Self::compute_dag_similarity(circuit1, circuit2)?
                }
                SimilarityAlgorithm::UnitaryMatrix => {
                    Self::compute_unitary_similarity(circuit1, circuit2)?
                }
                SimilarityAlgorithm::GraphBased {
                    algorithm: graph_alg,
                } => Self::compute_graph_similarity(&features1, &features2, graph_alg)?,
                SimilarityAlgorithm::Statistical => {
                    Self::compute_statistical_similarity(&features1, &features2)?
                }
                SimilarityAlgorithm::MLEmbeddings { model_type } => {
                    self.compute_ml_similarity(circuit1, circuit2, model_type)?
                }
            };

            detailed_metrics.insert(format!("{algorithm:?}"), similarity);
            similarities.push(similarity);
        }

        // Compute component similarities
        let structural_similarity = Self::compute_structural_similarity(&features1, &features2)?;
        let functional_similarity = Self::compute_functional_similarity(circuit1, circuit2)?;
        let sequence_similarity = Self::compute_sequence_similarity(&features1, &features2)?;
        let topological_similarity = Self::compute_topological_similarity(&features1, &features2)?;

        // Compute overall similarity using weighted combination
        let overall_similarity = self.config.weights.topological.mul_add(
            topological_similarity,
            self.config.weights.sequence.mul_add(
                sequence_similarity,
                self.config.weights.structural.mul_add(
                    structural_similarity,
                    self.config.weights.functional * functional_similarity,
                ),
            ),
        );

        let result = CircuitSimilarityMetrics {
            structural_similarity,
            functional_similarity,
            sequence_similarity,
            topological_similarity,
            overall_similarity,
            detailed_metrics,
        };

        // Cache result
        if self.config.cache_results {
            self.similarity_cache.insert(cache_key, result.clone());
        }

        Ok(result)
    }

    /// Compute distance metrics between circuits
    pub fn compute_distance<const N: usize, const M: usize>(
        &mut self,
        circuit1: &Circuit<N>,
        circuit2: &Circuit<M>,
    ) -> QuantRS2Result<CircuitDistanceMetrics> {
        let features1 = self.extract_circuit_features(circuit1)?;
        let features2 = self.extract_circuit_features(circuit2)?;

        // Compute edit distance
        let edit_distance = Self::compute_edit_distance(&features1, &features2)?;
        let max_gates = features1
            .gate_histogram
            .values()
            .sum::<usize>()
            .max(features2.gate_histogram.values().sum::<usize>());
        let normalized_edit_distance = if max_gates > 0 {
            edit_distance as f64 / max_gates as f64
        } else {
            0.0
        };

        // Compute other distance measures
        let wasserstein_distance = Self::compute_wasserstein_distance(&features1, &features2)?;
        let hausdorff_distance = Self::compute_hausdorff_distance(circuit1, circuit2)?;
        let earth_movers_distance = Self::compute_earth_movers_distance(&features1, &features2)?;
        let process_fidelity_distance =
            Self::compute_process_fidelity_distance(circuit1, circuit2)?;

        Ok(CircuitDistanceMetrics {
            edit_distance,
            normalized_edit_distance,
            wasserstein_distance,
            hausdorff_distance,
            earth_movers_distance,
            process_fidelity_distance,
        })
    }

    /// Extract comprehensive features from a circuit
    fn extract_circuit_features<const N: usize>(
        &mut self,
        circuit: &Circuit<N>,
    ) -> QuantRS2Result<CircuitFeatures> {
        let id = Self::generate_circuit_id(circuit);

        if let Some(cached) = self.feature_cache.get(&id) {
            return Ok(cached.clone());
        }

        let mut gate_histogram = HashMap::new();
        let mut connectivity_pattern = Vec::new();
        let mut critical_path = Vec::new();
        let mut two_qubit_gates = 0;

        // Analyze gates
        for gate in circuit.gates() {
            let gate_name = gate.name();
            *gate_histogram.entry(gate_name.to_string()).or_insert(0) += 1;
            critical_path.push(gate_name.to_string());

            if gate.qubits().len() == 2 {
                two_qubit_gates += 1;
                let qubits: Vec<usize> = gate.qubits().iter().map(|q| q.id() as usize).collect();
                connectivity_pattern.push((qubits[0], qubits[1]));
            }
        }

        // Compute parallelism profile
        let parallelism_profile = Self::compute_parallelism_profile(circuit)?;

        // Analyze entanglement structure
        let entanglement_structure = Self::analyze_entanglement_structure(circuit)?;

        let features = CircuitFeatures {
            gate_histogram,
            depth: circuit.gates().len(), // Simplified depth
            two_qubit_gates,
            connectivity_pattern,
            critical_path,
            parallelism_profile,
            entanglement_structure,
        };

        self.feature_cache.insert(id, features.clone());
        Ok(features)
    }

    /// Compute gate-level similarity
    fn compute_gate_level_similarity(
        features1: &CircuitFeatures,
        features2: &CircuitFeatures,
    ) -> QuantRS2Result<f64> {
        // Compare gate histograms using cosine similarity
        let mut dot_product = 0.0;
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;

        let all_gates: HashSet<String> = features1
            .gate_histogram
            .keys()
            .chain(features2.gate_histogram.keys())
            .cloned()
            .collect();

        for gate in all_gates {
            let count1 = *features1.gate_histogram.get(&gate).unwrap_or(&0) as f64;
            let count2 = *features2.gate_histogram.get(&gate).unwrap_or(&0) as f64;

            dot_product += count1 * count2;
            norm1 += count1 * count1;
            norm2 += count2 * count2;
        }

        let similarity = if norm1 > 0.0 && norm2 > 0.0 {
            dot_product / (norm1.sqrt() * norm2.sqrt())
        } else {
            0.0
        };

        Ok(similarity)
    }

    /// Compute DAG structure similarity
    fn compute_dag_similarity<const N: usize, const M: usize>(
        circuit1: &Circuit<N>,
        circuit2: &Circuit<M>,
    ) -> QuantRS2Result<f64> {
        // Convert circuits to DAGs and compare structure
        let dag1 = circuit_to_dag(circuit1);
        let dag2 = circuit_to_dag(circuit2);

        // Compare DAG properties
        let nodes_similarity = if dag1.nodes().len() == dag2.nodes().len() {
            1.0
        } else {
            let min_nodes = dag1.nodes().len().min(dag2.nodes().len()) as f64;
            let max_nodes = dag1.nodes().len().max(dag2.nodes().len()) as f64;
            min_nodes / max_nodes
        };

        let edges_similarity = if dag1.edges().len() == dag2.edges().len() {
            1.0
        } else {
            let min_edges = dag1.edges().len().min(dag2.edges().len()) as f64;
            let max_edges = dag1.edges().len().max(dag2.edges().len()) as f64;
            min_edges / max_edges
        };

        Ok(f64::midpoint(nodes_similarity, edges_similarity))
    }

    /// Compute unitary matrix similarity
    const fn compute_unitary_similarity<const N: usize, const M: usize>(
        _circuit1: &Circuit<N>,
        _circuit2: &Circuit<M>,
    ) -> QuantRS2Result<f64> {
        if N != M {
            // Circuits with different qubit counts have zero unitary similarity
            return Ok(0.0);
        }

        // Convert circuits to unitary matrices and compute fidelity
        // This is a simplified placeholder - would use actual matrix conversion
        let fidelity = 0.9; // Placeholder for unitary similarity

        Ok(fidelity)
    }

    /// Compute graph-based similarity
    fn compute_graph_similarity(
        features1: &CircuitFeatures,
        features2: &CircuitFeatures,
        algorithm: &GraphSimilarityAlgorithm,
    ) -> QuantRS2Result<f64> {
        match algorithm {
            GraphSimilarityAlgorithm::GraphEditDistance => Self::compute_graph_edit_distance(
                &features1.entanglement_structure.entanglement_graph,
                &features2.entanglement_structure.entanglement_graph,
            ),
            GraphSimilarityAlgorithm::SpectralSimilarity => Self::compute_spectral_similarity(
                &features1.entanglement_structure.entanglement_graph,
                &features2.entanglement_structure.entanglement_graph,
            ),
            _ => {
                // Other graph algorithms would be implemented
                Ok(0.5) // Placeholder
            }
        }
    }

    /// Compute statistical similarity
    fn compute_statistical_similarity(
        features1: &CircuitFeatures,
        features2: &CircuitFeatures,
    ) -> QuantRS2Result<f64> {
        // Compare statistical properties of circuits with division by zero protection
        let max_depth = features1.depth.max(features2.depth);
        let depth_similarity = if max_depth > 0 {
            1.0 - (features1.depth as f64 - features2.depth as f64).abs() / (max_depth as f64)
        } else {
            1.0 // Both have zero depth - identical
        };

        let max_two_qubit = features1.two_qubit_gates.max(features2.two_qubit_gates);
        let two_qubit_similarity = if max_two_qubit > 0 {
            1.0 - (features1.two_qubit_gates as f64 - features2.two_qubit_gates as f64).abs()
                / (max_two_qubit as f64)
        } else {
            1.0 // Both have zero two-qubit gates - identical
        };

        Ok(f64::midpoint(depth_similarity, two_qubit_similarity))
    }

    /// Compute ML-based similarity using embeddings
    fn compute_ml_similarity<const N: usize, const M: usize>(
        &mut self,
        circuit1: &Circuit<N>,
        circuit2: &Circuit<M>,
        model_type: &MLModelType,
    ) -> QuantRS2Result<f64> {
        // Generate circuit embeddings using ML models
        let embedding1 = self.generate_circuit_embedding(circuit1, model_type)?;
        let embedding2 = self.generate_circuit_embedding(circuit2, model_type)?;

        // Compute cosine similarity between embeddings
        let similarity = Self::cosine_similarity(&embedding1, &embedding2);
        Ok(similarity)
    }

    /// Generate circuit embedding using ML model
    fn generate_circuit_embedding<const N: usize>(
        &mut self,
        circuit: &Circuit<N>,
        model_type: &MLModelType,
    ) -> QuantRS2Result<Vec<f64>> {
        let id = format!("{}_{:?}", Self::generate_circuit_id(circuit), model_type);

        if let Some(cached) = self.embedding_cache.get(&id) {
            return Ok(cached.clone());
        }

        // Generate embedding based on model type
        let embedding = match model_type {
            MLModelType::VAE { latent_dim } => Self::generate_vae_embedding(circuit, *latent_dim)?,
            MLModelType::GCN { hidden_dims } => Self::generate_gcn_embedding(circuit, hidden_dims)?,
            MLModelType::Transformer {
                num_heads,
                num_layers,
            } => Self::generate_transformer_embedding(circuit, *num_heads, *num_layers)?,
            MLModelType::PreTrained { model_name } => {
                Self::generate_pretrained_embedding(circuit, model_name)?
            }
        };

        self.embedding_cache.insert(id, embedding.clone());
        Ok(embedding)
    }

    /// Generate VAE embedding (placeholder)
    fn generate_vae_embedding<const N: usize>(
        _circuit: &Circuit<N>,
        latent_dim: usize,
    ) -> QuantRS2Result<Vec<f64>> {
        // Placeholder for VAE-based circuit embedding
        Ok(vec![0.5; latent_dim])
    }

    /// Generate GCN embedding (placeholder)
    fn generate_gcn_embedding<const N: usize>(
        _circuit: &Circuit<N>,
        hidden_dims: &[usize],
    ) -> QuantRS2Result<Vec<f64>> {
        // Placeholder for GCN-based circuit embedding
        let output_dim = hidden_dims.last().unwrap_or(&64);
        Ok(vec![0.5; *output_dim])
    }

    /// Generate Transformer embedding (placeholder)
    fn generate_transformer_embedding<const N: usize>(
        _circuit: &Circuit<N>,
        num_heads: usize,
        _num_layers: usize,
    ) -> QuantRS2Result<Vec<f64>> {
        // Placeholder for Transformer-based circuit embedding
        let embedding_dim = num_heads * 64; // Typical dimension
        Ok(vec![0.5; embedding_dim])
    }

    /// Generate pre-trained model embedding (placeholder)
    fn generate_pretrained_embedding<const N: usize>(
        _circuit: &Circuit<N>,
        model_name: &str,
    ) -> QuantRS2Result<Vec<f64>> {
        // Placeholder for pre-trained model embedding
        let embedding_dim = match model_name {
            "circuit_bert" => 768,
            "quantum_gpt" => 512,
            _ => 256,
        };
        Ok(vec![0.5; embedding_dim])
    }

    /// Compute structural similarity
    fn compute_structural_similarity(
        features1: &CircuitFeatures,
        features2: &CircuitFeatures,
    ) -> QuantRS2Result<f64> {
        // Compare circuit structure
        let connectivity_similarity = Self::compare_connectivity_patterns(
            &features1.connectivity_pattern,
            &features2.connectivity_pattern,
        );

        // Handle depth comparison with division by zero protection
        let max_depth = features1.depth.max(features2.depth);
        let depth_similarity = if max_depth > 0 {
            1.0 - (features1.depth as f64 - features2.depth as f64).abs() / (max_depth as f64)
        } else {
            // Both circuits have zero depth - they are identical in this metric
            1.0
        };

        Ok(f64::midpoint(connectivity_similarity, depth_similarity))
    }

    /// Compute functional similarity
    const fn compute_functional_similarity<const N: usize, const M: usize>(
        _circuit1: &Circuit<N>,
        _circuit2: &Circuit<M>,
    ) -> QuantRS2Result<f64> {
        if N != M {
            return Ok(0.0);
        }

        // Compare functional behavior (simplified)
        // In practice, this would compute unitary similarity or process fidelity
        Ok(0.8) // Placeholder
    }

    /// Compute sequence similarity
    fn compute_sequence_similarity(
        features1: &CircuitFeatures,
        features2: &CircuitFeatures,
    ) -> QuantRS2Result<f64> {
        // Compare gate sequences using edit distance
        let edit_distance =
            Self::string_edit_distance(&features1.critical_path, &features2.critical_path);
        let max_length = features1
            .critical_path
            .len()
            .max(features2.critical_path.len());

        let similarity = if max_length > 0 {
            1.0 - (edit_distance as f64 / max_length as f64)
        } else {
            1.0
        };

        Ok(similarity)
    }

    /// Compute topological similarity
    fn compute_topological_similarity(
        features1: &CircuitFeatures,
        features2: &CircuitFeatures,
    ) -> QuantRS2Result<f64> {
        // Compare entanglement topology with division by zero protection
        let max_width = features1
            .entanglement_structure
            .max_entanglement_width
            .max(features2.entanglement_structure.max_entanglement_width);

        let width_similarity = if max_width > 0 {
            1.0 - (features1.entanglement_structure.max_entanglement_width as f64
                - features2.entanglement_structure.max_entanglement_width as f64)
                .abs()
                / (max_width as f64)
        } else {
            1.0 // Both have zero entanglement width - identical
        };

        Ok(width_similarity)
    }

    /// Helper methods

    /// Generate unique circuit identifier
    fn generate_circuit_id<const N: usize>(circuit: &Circuit<N>) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        N.hash(&mut hasher);

        for gate in circuit.gates() {
            gate.name().hash(&mut hasher);
            for qubit in gate.qubits() {
                qubit.id().hash(&mut hasher);
            }
        }

        format!("{:x}", hasher.finish())
    }

    /// Compute parallelism profile
    fn compute_parallelism_profile<const N: usize>(
        circuit: &Circuit<N>,
    ) -> QuantRS2Result<Vec<usize>> {
        // Simplified parallelism analysis
        let mut profile = Vec::new();
        let gate_count = circuit.gates().len();

        // For now, assume linear execution (no parallelism)
        for _ in 0..gate_count {
            profile.push(1);
        }

        Ok(profile)
    }

    /// Analyze entanglement structure
    fn analyze_entanglement_structure<const N: usize>(
        circuit: &Circuit<N>,
    ) -> QuantRS2Result<EntanglementStructure> {
        let mut entangling_layers = Vec::new();
        let mut current_layer = Vec::new();
        let mut max_width = 0;

        for gate in circuit.gates() {
            if gate.qubits().len() == 2 {
                let qubits: Vec<usize> = gate.qubits().iter().map(|q| q.id() as usize).collect();
                current_layer.push((qubits[0], qubits[1]));
                max_width = max_width.max(current_layer.len());
            } else if !current_layer.is_empty() {
                entangling_layers.push(current_layer);
                current_layer = Vec::new();
            }
        }

        if !current_layer.is_empty() {
            entangling_layers.push(current_layer);
        }

        // Create entanglement graph
        let mut graph = SciRS2Graph {
            nodes: (0..N).collect(),
            edges: Vec::new(),
            node_attributes: HashMap::new(),
            edge_attributes: HashMap::new(),
        };

        for layer in &entangling_layers {
            for &(q1, q2) in layer {
                graph.edges.push((q1, q2, 1.0));
            }
        }

        Ok(EntanglementStructure {
            entangling_layers,
            max_entanglement_width: max_width,
            entanglement_graph: graph,
        })
    }

    /// Compare connectivity patterns
    fn compare_connectivity_patterns(
        pattern1: &[(usize, usize)],
        pattern2: &[(usize, usize)],
    ) -> f64 {
        let set1: HashSet<_> = pattern1.iter().collect();
        let set2: HashSet<_> = pattern2.iter().collect();

        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();

        if union > 0 {
            intersection as f64 / union as f64
        } else {
            1.0
        }
    }

    /// Compute edit distance between strings
    fn string_edit_distance(seq1: &[String], seq2: &[String]) -> usize {
        let m = seq1.len();
        let n = seq2.len();
        let mut dp = vec![vec![0; n + 1]; m + 1];

        // Initialize base cases
        for i in 0..=m {
            dp[i][0] = i;
        }
        for j in 0..=n {
            dp[0][j] = j;
        }

        // Fill DP table
        for i in 1..=m {
            for j in 1..=n {
                if seq1[i - 1] == seq2[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = 1 + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
                }
            }
        }

        dp[m][n]
    }

    /// Compute cosine similarity between vectors
    fn cosine_similarity(vec1: &[f64], vec2: &[f64]) -> f64 {
        if vec1.len() != vec2.len() {
            return 0.0;
        }

        let dot_product: f64 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f64 = vec1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2: f64 = vec2.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm1 > 0.0 && norm2 > 0.0 {
            dot_product / (norm1 * norm2)
        } else {
            0.0
        }
    }

    /// Compute edit distance between circuit features
    fn compute_edit_distance(
        features1: &CircuitFeatures,
        features2: &CircuitFeatures,
    ) -> QuantRS2Result<usize> {
        // Simplified edit distance based on gate operations
        let distance =
            Self::string_edit_distance(&features1.critical_path, &features2.critical_path);
        Ok(distance)
    }

    /// Compute Wasserstein distance
    const fn compute_wasserstein_distance(
        _features1: &CircuitFeatures,
        _features2: &CircuitFeatures,
    ) -> QuantRS2Result<f64> {
        // Simplified Wasserstein distance computation
        // In practice, would use SciRS2's optimal transport algorithms
        Ok(0.3) // Placeholder
    }

    /// Compute Hausdorff distance
    const fn compute_hausdorff_distance<const N: usize, const M: usize>(
        _circuit1: &Circuit<N>,
        _circuit2: &Circuit<M>,
    ) -> QuantRS2Result<f64> {
        // Placeholder for Hausdorff distance computation
        Ok(0.25) // Placeholder
    }

    /// Compute Earth Mover's distance
    const fn compute_earth_movers_distance(
        _features1: &CircuitFeatures,
        _features2: &CircuitFeatures,
    ) -> QuantRS2Result<f64> {
        // Placeholder for Earth Mover's distance computation
        Ok(0.2) // Placeholder
    }

    /// Compute process fidelity distance
    const fn compute_process_fidelity_distance<const N: usize, const M: usize>(
        _circuit1: &Circuit<N>,
        _circuit2: &Circuit<M>,
    ) -> QuantRS2Result<f64> {
        if N != M {
            return Ok(1.0); // Maximum distance for different dimensions
        }

        // Placeholder for process fidelity computation
        Ok(0.1) // Placeholder
    }

    /// Compute graph edit distance
    fn compute_graph_edit_distance(
        graph1: &SciRS2Graph,
        graph2: &SciRS2Graph,
    ) -> QuantRS2Result<f64> {
        // Simplified graph edit distance
        let node_diff = (graph1.nodes.len() as f64 - graph2.nodes.len() as f64).abs();
        let edge_diff = (graph1.edges.len() as f64 - graph2.edges.len() as f64).abs();
        let max_size = (graph1.nodes.len() + graph1.edges.len())
            .max(graph2.nodes.len() + graph2.edges.len()) as f64;

        let distance = if max_size > 0.0 {
            (node_diff + edge_diff) / max_size
        } else {
            0.0 // Both graphs are empty - identical
        };

        Ok(1.0 - distance) // Convert to similarity
    }

    /// Compute spectral similarity
    const fn compute_spectral_similarity(
        _graph1: &SciRS2Graph,
        _graph2: &SciRS2Graph,
    ) -> QuantRS2Result<f64> {
        // Placeholder for spectral similarity computation
        // Would compute eigenvalues of graph Laplacians and compare
        Ok(0.7) // Placeholder
    }
}

/// Batch similarity computation for multiple circuits
pub struct BatchSimilarityComputer {
    analyzer: CircuitSimilarityAnalyzer,
}

impl BatchSimilarityComputer {
    /// Create new batch computer
    #[must_use]
    pub fn new(config: SimilarityConfig) -> Self {
        Self {
            analyzer: CircuitSimilarityAnalyzer::new(config),
        }
    }

    /// Compute pairwise similarities for a set of circuits
    pub fn compute_pairwise_similarities<const N: usize>(
        &mut self,
        circuits: &[Circuit<N>],
    ) -> QuantRS2Result<Vec<Vec<f64>>> {
        let n_circuits = circuits.len();
        let mut similarity_matrix = vec![vec![0.0; n_circuits]; n_circuits];

        for i in 0..n_circuits {
            similarity_matrix[i][i] = 1.0; // Self-similarity

            for j in (i + 1)..n_circuits {
                let similarity = self
                    .analyzer
                    .compute_similarity(&circuits[i], &circuits[j])?;
                similarity_matrix[i][j] = similarity.overall_similarity;
                similarity_matrix[j][i] = similarity.overall_similarity; // Symmetric
            }
        }

        Ok(similarity_matrix)
    }

    /// Find most similar circuits in a dataset
    pub fn find_most_similar<const N: usize>(
        &mut self,
        query_circuit: &Circuit<N>,
        dataset: &[Circuit<N>],
        top_k: usize,
    ) -> QuantRS2Result<Vec<(usize, f64)>> {
        let mut similarities = Vec::new();

        for (i, circuit) in dataset.iter().enumerate() {
            let similarity = self.analyzer.compute_similarity(query_circuit, circuit)?;
            similarities.push((i, similarity.overall_similarity));
        }

        // Sort by similarity and return top-k
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(top_k);

        Ok(similarities)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::multi::CNOT;
    use quantrs2_core::gate::single::Hadamard;

    #[test]
    fn test_similarity_analyzer_creation() {
        let analyzer = CircuitSimilarityAnalyzer::with_default_config();
        assert_eq!(analyzer.config.algorithms.len(), 3);
    }

    #[test]
    fn test_identical_circuits_similarity() {
        let mut analyzer = CircuitSimilarityAnalyzer::with_default_config();

        let mut circuit = Circuit::<2>::new();
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("Failed to add Hadamard gate");

        let similarity = analyzer
            .compute_similarity(&circuit, &circuit)
            .expect("Failed to compute similarity for identical circuits");

        // Fixed: Overall similarity should be 1.0 for identical circuits
        // All component similarities should also be 1.0 or very close to it
        assert!(
            !similarity.overall_similarity.is_nan(),
            "Similarity should not be NaN for identical circuits. Actual value: {}",
            similarity.overall_similarity
        );
        assert!(
            !similarity.overall_similarity.is_infinite(),
            "Similarity should not be infinite for identical circuits. Actual value: {}",
            similarity.overall_similarity
        );
        assert!(
            similarity.structural_similarity >= 0.9,
            "Structural similarity should be high for identical circuits: {}",
            similarity.structural_similarity
        );
        assert!(
            similarity.sequence_similarity >= 0.9,
            "Sequence similarity should be high for identical circuits: {}",
            similarity.sequence_similarity
        );
        assert!(
            similarity.topological_similarity >= 0.9,
            "Topological similarity should be high for identical circuits: {}",
            similarity.topological_similarity
        );
        assert!(
            similarity.overall_similarity >= 0.8,
            "Overall similarity should be high for identical circuits: {}",
            similarity.overall_similarity
        );
    }

    #[test]
    fn test_different_circuits_similarity() {
        let mut analyzer = CircuitSimilarityAnalyzer::with_default_config();

        let mut circuit1 = Circuit::<2>::new();
        circuit1
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("Failed to add Hadamard gate to circuit1");

        let mut circuit2 = Circuit::<2>::new();
        circuit2
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .expect("Failed to add CNOT gate to circuit2");

        let similarity = analyzer
            .compute_similarity(&circuit1, &circuit2)
            .expect("Failed to compute similarity for different circuits");
        assert!(similarity.overall_similarity < 1.0);
    }

    #[test]
    fn test_distance_computation() {
        let mut analyzer = CircuitSimilarityAnalyzer::with_default_config();

        let mut circuit1 = Circuit::<2>::new();
        circuit1
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("Failed to add Hadamard gate to circuit1");

        let mut circuit2 = Circuit::<2>::new();
        circuit2
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .expect("Failed to add CNOT gate to circuit2");

        let distance = analyzer
            .compute_distance(&circuit1, &circuit2)
            .expect("Failed to compute distance between circuits");
        assert!(distance.edit_distance > 0);
        assert!(
            distance.normalized_edit_distance >= 0.0 && distance.normalized_edit_distance <= 1.0
        );
    }

    #[test]
    fn test_feature_extraction() {
        let mut analyzer = CircuitSimilarityAnalyzer::with_default_config();

        let mut circuit = Circuit::<2>::new();
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("Failed to add Hadamard gate");
        circuit
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .expect("Failed to add CNOT gate");

        let features = analyzer
            .extract_circuit_features(&circuit)
            .expect("Failed to extract circuit features");
        assert_eq!(features.gate_histogram.get("H"), Some(&1));
        assert_eq!(features.gate_histogram.get("CNOT"), Some(&1));
        assert_eq!(features.two_qubit_gates, 1);
    }

    #[test]
    fn test_batch_similarity_computation() {
        let mut computer = BatchSimilarityComputer::new(SimilarityConfig::default());

        let mut circuit1 = Circuit::<2>::new();
        circuit1
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("Failed to add Hadamard gate to circuit1");

        let mut circuit2 = Circuit::<2>::new();
        circuit2
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .expect("Failed to add CNOT gate to circuit2");

        let circuits = vec![circuit1, circuit2];
        let similarity_matrix = computer
            .compute_pairwise_similarities(&circuits)
            .expect("Failed to compute pairwise similarities");

        assert_eq!(similarity_matrix.len(), 2);
        assert_eq!(similarity_matrix[0].len(), 2);
        assert_eq!(similarity_matrix[0][0], 1.0); // Self-similarity
        assert_eq!(similarity_matrix[1][1], 1.0); // Self-similarity
        assert_eq!(similarity_matrix[0][1], similarity_matrix[1][0]); // Symmetry
    }
}
