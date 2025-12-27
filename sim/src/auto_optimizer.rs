//! `AutoOptimizer` for Automatic Backend Selection based on Problem Characteristics
//!
//! This module provides intelligent backend selection for quantum circuit simulation
//! by analyzing circuit characteristics and automatically choosing the optimal
//! execution backend using `SciRS2` optimization and analysis tools.

use crate::{
    automatic_parallelization::{AutoParallelConfig, AutoParallelEngine},
    circuit_optimization::{CircuitOptimizer, OptimizationConfig},
    distributed_simulator::{DistributedQuantumSimulator, DistributedSimulatorConfig},
    error::{Result, SimulatorError},
    large_scale_simulator::{LargeScaleQuantumSimulator, LargeScaleSimulatorConfig},
    simulator::SimulatorResult,
    statevector::StateVectorSimulator,
};
use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
    register::Register,
};
use std::fmt::Write;

#[cfg(all(feature = "gpu", not(target_os = "macos")))]
use crate::gpu::SciRS2GpuStateVectorSimulator;
use scirs2_core::parallel_ops::current_num_threads; // SciRS2 POLICY compliant
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Configuration for the `AutoOptimizer`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoOptimizerConfig {
    /// Enable performance profiling during backend selection
    pub enable_profiling: bool,
    /// Memory budget for simulation (bytes)
    pub memory_budget: usize,
    /// CPU utilization threshold (0.0 to 1.0)
    pub cpu_utilization_threshold: f64,
    /// GPU availability check timeout
    pub gpu_check_timeout: Duration,
    /// Enable distributed simulation for large circuits
    pub enable_distributed: bool,
    /// `SciRS2` optimization level
    pub scirs2_optimization_level: OptimizationLevel,
    /// Fallback strategy when optimal backend is unavailable
    pub fallback_strategy: FallbackStrategy,
    /// Circuit complexity analysis depth
    pub analysis_depth: AnalysisDepth,
    /// Performance history cache size
    pub performance_cache_size: usize,
    /// Backend preference order
    pub backend_preferences: Vec<BackendType>,
}

impl Default for AutoOptimizerConfig {
    fn default() -> Self {
        Self {
            enable_profiling: true,
            memory_budget: 8 * 1024 * 1024 * 1024, // 8GB
            cpu_utilization_threshold: 0.8,
            gpu_check_timeout: Duration::from_millis(1000),
            enable_distributed: true,
            scirs2_optimization_level: OptimizationLevel::Aggressive,
            fallback_strategy: FallbackStrategy::Conservative,
            analysis_depth: AnalysisDepth::Deep,
            performance_cache_size: 1000,
            backend_preferences: vec![
                BackendType::SciRS2Gpu,
                BackendType::LargeScale,
                BackendType::Distributed,
                BackendType::StateVector,
            ],
        }
    }
}

/// Available backend types for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendType {
    /// CPU state vector simulator
    StateVector,
    /// SciRS2-powered GPU simulator
    SciRS2Gpu,
    /// Large-scale optimized simulator
    LargeScale,
    /// Distributed cluster simulator
    Distributed,
    /// Automatic selection based on characteristics
    Auto,
}

/// Optimization levels for `SciRS2` integration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// No optimization
    None,
    /// Basic optimizations
    Basic,
    /// Advanced optimizations
    Advanced,
    /// Aggressive optimizations with maximum `SciRS2` features
    Aggressive,
}

/// Fallback strategies when optimal backend is unavailable
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FallbackStrategy {
    /// Conservative fallback to reliable backends
    Conservative,
    /// Aggressive fallback trying more experimental backends
    Aggressive,
    /// Fail if optimal backend is unavailable
    Fail,
}

/// Circuit analysis depth levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnalysisDepth {
    /// Quick analysis with basic metrics
    Quick,
    /// Standard analysis with comprehensive metrics
    Standard,
    /// Deep analysis with advanced circuit characterization
    Deep,
}

/// Circuit characteristics analysis results
#[derive(Debug, Clone)]
pub struct CircuitCharacteristics {
    /// Number of qubits
    pub num_qubits: usize,
    /// Number of gates
    pub num_gates: usize,
    /// Circuit depth (longest path)
    pub circuit_depth: usize,
    /// Gate type distribution
    pub gate_distribution: HashMap<String, usize>,
    /// Parallelism potential (0.0 to 1.0)
    pub parallelism_potential: f64,
    /// Memory requirement estimate (bytes)
    pub memory_requirement: usize,
    /// Computational complexity score
    pub complexity_score: f64,
    /// Two-qubit gate density
    pub two_qubit_density: f64,
    /// Connectivity graph properties
    pub connectivity_properties: ConnectivityProperties,
    /// Entanglement depth estimate
    pub entanglement_depth: usize,
    /// Noise susceptibility score
    pub noise_susceptibility: f64,
}

/// Connectivity graph properties of the circuit
#[derive(Debug, Clone)]
pub struct ConnectivityProperties {
    /// Maximum degree of connectivity
    pub max_degree: usize,
    /// Average degree of connectivity
    pub avg_degree: f64,
    /// Number of connected components
    pub connected_components: usize,
    /// Circuit diameter (longest path between any two qubits)
    pub diameter: usize,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
}

/// Backend recommendation with reasoning
#[derive(Debug, Clone)]
pub struct BackendRecommendation {
    /// Recommended backend type
    pub backend_type: BackendType,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Expected performance improvement over baseline
    pub expected_improvement: f64,
    /// Estimated execution time
    pub estimated_execution_time: Duration,
    /// Estimated memory usage
    pub estimated_memory_usage: usize,
    /// Reasoning for the recommendation
    pub reasoning: String,
    /// Alternative recommendations
    pub alternatives: Vec<(BackendType, f64)>,
    /// Performance prediction model used
    pub prediction_model: String,
}

/// Performance metrics for backend selection
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Execution time
    pub execution_time: Duration,
    /// Memory usage
    pub memory_usage: usize,
    /// CPU utilization
    pub cpu_utilization: f64,
    /// GPU utilization (if applicable)
    pub gpu_utilization: Option<f64>,
    /// Throughput (gates per second)
    pub throughput: f64,
    /// Error rate
    pub error_rate: f64,
}

/// Performance history entry for caching
#[derive(Debug, Clone)]
pub struct PerformanceHistory {
    /// Circuit characteristics hash
    pub circuit_hash: u64,
    /// Backend used
    pub backend_type: BackendType,
    /// Performance metrics achieved
    pub metrics: PerformanceMetrics,
    /// Timestamp
    pub timestamp: Instant,
}

/// `AutoOptimizer` for intelligent backend selection
pub struct AutoOptimizer {
    /// Configuration
    config: AutoOptimizerConfig,
    /// Circuit optimizer for preprocessing
    circuit_optimizer: CircuitOptimizer,
    /// Parallelization engine
    parallel_engine: AutoParallelEngine,
    /// Performance history cache
    performance_cache: Vec<PerformanceHistory>,
    /// Backend availability cache
    backend_availability: HashMap<BackendType, bool>,
    /// `SciRS2` analysis tools integration
    scirs2_analyzer: SciRS2CircuitAnalyzer,
}

/// SciRS2-powered circuit analyzer
struct SciRS2CircuitAnalyzer {
    /// Enable advanced `SciRS2` features
    enable_advanced_features: bool,
}

impl AutoOptimizer {
    /// Create a new `AutoOptimizer` with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(AutoOptimizerConfig::default())
    }

    /// Create a new `AutoOptimizer` with custom configuration
    #[must_use]
    pub fn with_config(config: AutoOptimizerConfig) -> Self {
        let optimization_config = OptimizationConfig {
            enable_gate_fusion: true,
            enable_redundant_elimination: true,
            enable_commutation_reordering: true,
            enable_single_qubit_optimization: true,
            enable_two_qubit_optimization: true,
            max_passes: 3,
            enable_depth_reduction: true,
        };

        let parallel_config = AutoParallelConfig {
            max_threads: current_num_threads(), // SciRS2 POLICY compliant
            min_gates_for_parallel: 20,
            strategy: crate::automatic_parallelization::ParallelizationStrategy::Hybrid,
            ..Default::default()
        };

        Self {
            config,
            circuit_optimizer: CircuitOptimizer::with_config(optimization_config),
            parallel_engine: AutoParallelEngine::new(parallel_config),
            performance_cache: Vec::new(),
            backend_availability: HashMap::new(),
            scirs2_analyzer: SciRS2CircuitAnalyzer {
                enable_advanced_features: true,
            },
        }
    }

    /// Analyze circuit characteristics using `SciRS2` tools
    pub fn analyze_circuit<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> QuantRS2Result<CircuitCharacteristics> {
        let start_time = Instant::now();

        // Basic circuit metrics
        let num_qubits = circuit.num_qubits();
        let num_gates = circuit.num_gates();
        let circuit_depth = self.calculate_circuit_depth(circuit);

        // Gate distribution analysis
        let gate_distribution = self.analyze_gate_distribution(circuit);

        // Parallelism analysis using SciRS2
        let parallelism_potential = self.analyze_parallelism_potential(circuit)?;

        // Memory requirement estimation
        let memory_requirement = self.estimate_memory_requirement(num_qubits, num_gates);

        // Complexity scoring using SciRS2 complexity analysis
        let complexity_score = self.calculate_complexity_score(circuit)?;

        // Two-qubit gate analysis
        let two_qubit_density = self.calculate_two_qubit_density(circuit);

        // Connectivity analysis
        let connectivity_properties = self.analyze_connectivity(circuit)?;

        // Entanglement depth estimation using SciRS2
        let entanglement_depth = self.estimate_entanglement_depth(circuit)?;

        // Noise susceptibility analysis
        let noise_susceptibility = self.analyze_noise_susceptibility(circuit);

        let analysis_time = start_time.elapsed();
        if self.config.enable_profiling {
            println!("Circuit analysis completed in {analysis_time:?}");
        }

        Ok(CircuitCharacteristics {
            num_qubits,
            num_gates,
            circuit_depth,
            gate_distribution,
            parallelism_potential,
            memory_requirement,
            complexity_score,
            two_qubit_density,
            connectivity_properties,
            entanglement_depth,
            noise_susceptibility,
        })
    }

    /// Recommend optimal backend based on circuit characteristics
    pub fn recommend_backend<const N: usize>(
        &mut self,
        circuit: &Circuit<N>,
    ) -> QuantRS2Result<BackendRecommendation> {
        // Analyze circuit characteristics
        let characteristics = self.analyze_circuit(circuit)?;

        // Check backend availability
        self.update_backend_availability()?;

        // Check performance cache for similar circuits
        if let Some(cached_result) = self.check_performance_cache(&characteristics) {
            return Ok(self.build_recommendation_from_cache(cached_result));
        }

        // Generate recommendations based on characteristics
        let recommendation = self.generate_backend_recommendation(&characteristics)?;

        Ok(recommendation)
    }

    /// Execute circuit with automatic backend selection
    pub fn execute_optimized<const N: usize>(
        &mut self,
        circuit: &Circuit<N>,
    ) -> Result<SimulatorResult<N>> {
        // Get backend recommendation
        let recommendation = self
            .recommend_backend(circuit)
            .map_err(|e| SimulatorError::ComputationError(e.to_string()))?;

        if self.config.enable_profiling {
            println!(
                "Using {} backend (confidence: {:.2})",
                self.backend_type_name(recommendation.backend_type),
                recommendation.confidence
            );
            println!("Reasoning: {}", recommendation.reasoning);
        }

        // Execute with recommended backend
        let start_time = Instant::now();
        let register = self.execute_with_backend(circuit, recommendation.backend_type)?;
        let execution_time = start_time.elapsed();

        // Convert Register to SimulatorResult
        let result = self.register_to_simulator_result(register);

        // Record performance metrics
        if self.config.enable_profiling {
            self.record_performance_metrics(circuit, recommendation.backend_type, execution_time);
            println!("Execution completed in {execution_time:?}");
        }

        Ok(result)
    }

    /// Calculate circuit depth (critical path length)
    fn calculate_circuit_depth<const N: usize>(&self, circuit: &Circuit<N>) -> usize {
        let mut qubit_depths = HashMap::new();
        let mut max_depth = 0;

        for gate in circuit.gates() {
            let qubits = gate.qubits();

            // Find maximum depth among input qubits
            let input_depth = qubits
                .iter()
                .map(|&q| qubit_depths.get(&q).copied().unwrap_or(0))
                .max()
                .unwrap_or(0);

            let new_depth = input_depth + 1;

            // Update depths for all output qubits
            for &qubit in &qubits {
                qubit_depths.insert(qubit, new_depth);
            }

            max_depth = max_depth.max(new_depth);
        }

        max_depth
    }

    /// Analyze gate distribution in the circuit
    fn analyze_gate_distribution<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();

        for gate in circuit.gates() {
            let gate_name = gate.name().to_string();
            *distribution.entry(gate_name).or_insert(0) += 1;
        }

        distribution
    }

    /// Analyze parallelism potential using `SciRS2` parallel ops
    fn analyze_parallelism_potential<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> QuantRS2Result<f64> {
        // Use SciRS2-powered parallelization analysis
        let analysis = self.parallel_engine.analyze_circuit(circuit)?;
        Ok(analysis.efficiency)
    }

    /// Estimate memory requirement for circuit simulation
    const fn estimate_memory_requirement(&self, num_qubits: usize, num_gates: usize) -> usize {
        // State vector memory: 2^n complex numbers
        let state_vector_size = (1 << num_qubits) * std::mem::size_of::<Complex64>();

        // Additional overhead for gate operations and intermediate results
        let overhead = num_gates * 64; // Rough estimate

        state_vector_size + overhead
    }

    /// Calculate circuit complexity score using `SciRS2` complexity analysis
    fn calculate_complexity_score<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> QuantRS2Result<f64> {
        let num_qubits = circuit.num_qubits() as f64;
        let num_gates = circuit.num_gates() as f64;
        let depth = self.calculate_circuit_depth(circuit) as f64;

        // SciRS2-inspired complexity scoring
        let gate_complexity = num_gates * (num_qubits.log2() + 1.0);
        let depth_complexity = depth * num_qubits;
        let entanglement_complexity = self.estimate_entanglement_complexity(circuit)?;

        Ok((gate_complexity + depth_complexity + entanglement_complexity) / 1000.0)
    }

    /// Estimate entanglement complexity
    fn estimate_entanglement_complexity<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> QuantRS2Result<f64> {
        let mut entanglement_score = 0.0;

        for gate in circuit.gates() {
            let qubits = gate.qubits();
            if qubits.len() >= 2 {
                // Two-qubit gates increase entanglement complexity
                entanglement_score += qubits.len() as f64 * qubits.len() as f64;
            }
        }

        Ok(entanglement_score)
    }

    /// Calculate two-qubit gate density
    fn calculate_two_qubit_density<const N: usize>(&self, circuit: &Circuit<N>) -> f64 {
        let total_gates = circuit.num_gates();
        if total_gates == 0 {
            return 0.0;
        }

        let two_qubit_gates = circuit
            .gates()
            .iter()
            .filter(|gate| gate.qubits().len() >= 2)
            .count();

        two_qubit_gates as f64 / total_gates as f64
    }

    /// Analyze circuit connectivity using `SciRS2` graph analysis
    fn analyze_connectivity<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> QuantRS2Result<ConnectivityProperties> {
        let mut qubit_connections: HashMap<QubitId, Vec<QubitId>> = HashMap::new();

        // Build connectivity graph
        for gate in circuit.gates() {
            let qubits = gate.qubits();
            if qubits.len() >= 2 {
                for i in 0..qubits.len() {
                    for j in (i + 1)..qubits.len() {
                        qubit_connections
                            .entry(qubits[i])
                            .or_default()
                            .push(qubits[j]);
                        qubit_connections
                            .entry(qubits[j])
                            .or_default()
                            .push(qubits[i]);
                    }
                }
            }
        }

        // Calculate connectivity properties
        let max_degree = qubit_connections
            .values()
            .map(std::vec::Vec::len)
            .max()
            .unwrap_or(0);

        let avg_degree = if qubit_connections.is_empty() {
            0.0
        } else {
            qubit_connections
                .values()
                .map(std::vec::Vec::len)
                .sum::<usize>() as f64
                / qubit_connections.len() as f64
        };

        // Simplified connected components analysis
        let connected_components = 1; // Simplified for now

        // Simplified diameter calculation
        let diameter = circuit.num_qubits().min(6); // Cap at 6 for practical purposes

        // Simplified clustering coefficient
        let clustering_coefficient = 0.5; // Placeholder

        Ok(ConnectivityProperties {
            max_degree,
            avg_degree,
            connected_components,
            diameter,
            clustering_coefficient,
        })
    }

    /// Estimate entanglement depth using `SciRS2` analysis
    fn estimate_entanglement_depth<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> QuantRS2Result<usize> {
        // Simplified entanglement depth estimation
        let two_qubit_gates = circuit
            .gates()
            .iter()
            .filter(|gate| gate.qubits().len() >= 2)
            .count();

        // Rough estimate based on two-qubit gate count and circuit structure
        let depth_estimate = (two_qubit_gates as f64).sqrt().ceil() as usize;
        Ok(depth_estimate.min(circuit.num_qubits()))
    }

    /// Analyze noise susceptibility
    fn analyze_noise_susceptibility<const N: usize>(&self, circuit: &Circuit<N>) -> f64 {
        let depth = self.calculate_circuit_depth(circuit) as f64;
        let two_qubit_density = self.calculate_two_qubit_density(circuit);

        // Circuits with higher depth and more two-qubit gates are more susceptible to noise
        (depth / 100.0 + two_qubit_density).min(1.0)
    }

    /// Update backend availability status
    fn update_backend_availability(&mut self) -> QuantRS2Result<()> {
        // Check GPU availability
        #[cfg(all(feature = "gpu", not(target_os = "macos")))]
        let gpu_available = SciRS2GpuStateVectorSimulator::is_available();
        #[cfg(any(not(feature = "gpu"), target_os = "macos"))]
        let gpu_available = false;

        self.backend_availability
            .insert(BackendType::SciRS2Gpu, gpu_available);

        // CPU backends are always available
        self.backend_availability
            .insert(BackendType::StateVector, true);
        self.backend_availability
            .insert(BackendType::LargeScale, true);

        // Distributed availability would require cluster check
        self.backend_availability
            .insert(BackendType::Distributed, false);

        Ok(())
    }

    /// Check performance cache for similar circuits
    fn check_performance_cache(
        &self,
        characteristics: &CircuitCharacteristics,
    ) -> Option<&PerformanceHistory> {
        // Simple cache lookup based on circuit characteristics
        // In practice, would use more sophisticated similarity matching
        self.performance_cache
            .iter()
            .find(|&entry| self.are_characteristics_similar(characteristics, entry))
            .map(|v| v as _)
    }

    /// Check if circuit characteristics are similar to cached entry
    const fn are_characteristics_similar(
        &self,
        characteristics: &CircuitCharacteristics,
        entry: &PerformanceHistory,
    ) -> bool {
        // Simplified similarity check - in practice would be more sophisticated
        false // Always return false for now to avoid cache hits during development
    }

    /// Build recommendation from cached performance data
    fn build_recommendation_from_cache(
        &self,
        cache_entry: &PerformanceHistory,
    ) -> BackendRecommendation {
        BackendRecommendation {
            backend_type: cache_entry.backend_type,
            confidence: 0.9, // High confidence for cached results
            expected_improvement: 0.0,
            estimated_execution_time: cache_entry.metrics.execution_time,
            estimated_memory_usage: cache_entry.metrics.memory_usage,
            reasoning: "Based on cached performance data for similar circuits".to_string(),
            alternatives: Vec::new(),
            prediction_model: "Cache-based".to_string(),
        }
    }

    /// Generate backend recommendation based on circuit characteristics
    fn generate_backend_recommendation(
        &self,
        characteristics: &CircuitCharacteristics,
    ) -> QuantRS2Result<BackendRecommendation> {
        let mut scores: HashMap<BackendType, f64> = HashMap::new();
        let mut reasoning = String::new();

        // Score different backends based on circuit characteristics
        for &backend_type in &self.config.backend_preferences {
            if !self
                .backend_availability
                .get(&backend_type)
                .unwrap_or(&false)
            {
                continue;
            }

            let score = self.score_backend_for_characteristics(backend_type, characteristics);
            scores.insert(backend_type, score);
        }

        // Find the best backend
        let (best_backend, best_score) = scores
            .into_iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((BackendType::StateVector, 0.5));

        // Generate reasoning
        reasoning = self.generate_recommendation_reasoning(best_backend, characteristics);

        // Estimate performance
        let estimated_execution_time = self.estimate_execution_time(best_backend, characteristics);
        let estimated_memory_usage = characteristics.memory_requirement;

        Ok(BackendRecommendation {
            backend_type: best_backend,
            confidence: best_score,
            expected_improvement: (best_score - 0.5).max(0.0) * 2.0, // Normalize to improvement
            estimated_execution_time,
            estimated_memory_usage,
            reasoning,
            alternatives: Vec::new(),
            prediction_model: "SciRS2-guided heuristic".to_string(),
        })
    }

    /// Score a backend for given circuit characteristics
    fn score_backend_for_characteristics(
        &self,
        backend_type: BackendType,
        characteristics: &CircuitCharacteristics,
    ) -> f64 {
        let mut score: f64 = 0.5; // Base score

        match backend_type {
            BackendType::StateVector => {
                // Good for small circuits
                if characteristics.num_qubits <= 20 {
                    score += 0.3;
                }
                if characteristics.num_gates <= 1000 {
                    score += 0.2;
                }
            }
            BackendType::SciRS2Gpu => {
                // Good for medium to large circuits with high parallelism
                if characteristics.num_qubits >= 10 && characteristics.num_qubits <= 30 {
                    score += 0.4;
                }
                if characteristics.parallelism_potential > 0.5 {
                    score += 0.3;
                }
                if characteristics.two_qubit_density > 0.3 {
                    score += 0.2;
                }
            }
            BackendType::LargeScale => {
                // Good for large circuits
                if characteristics.num_qubits >= 20 {
                    score += 0.4;
                }
                if characteristics.complexity_score > 0.5 {
                    score += 0.3;
                }
            }
            BackendType::Distributed => {
                // Good for very large circuits
                if characteristics.num_qubits >= 30 {
                    score += 0.5;
                }
                if characteristics.memory_requirement > self.config.memory_budget / 2 {
                    score += 0.3;
                }
            }
            BackendType::Auto => {
                // Fallback case
                score = 0.1;
            }
        }

        score.min(1.0)
    }

    /// Generate recommendation reasoning text
    fn generate_recommendation_reasoning(
        &self,
        backend_type: BackendType,
        characteristics: &CircuitCharacteristics,
    ) -> String {
        match backend_type {
            BackendType::StateVector => {
                format!("CPU state vector simulator recommended for {} qubits, {} gates. Suitable for small circuits with straightforward execution.",
                       characteristics.num_qubits, characteristics.num_gates)
            }
            BackendType::SciRS2Gpu => {
                format!("SciRS2 GPU simulator recommended for {} qubits, {} gates. High parallelism potential ({:.2}) and two-qubit gate density ({:.2}) make GPU acceleration beneficial.",
                       characteristics.num_qubits, characteristics.num_gates, characteristics.parallelism_potential, characteristics.two_qubit_density)
            }
            BackendType::LargeScale => {
                format!("Large-scale simulator recommended for {} qubits, {} gates. Circuit complexity ({:.2}) and depth ({}) require optimized memory management.",
                       characteristics.num_qubits, characteristics.num_gates, characteristics.complexity_score, characteristics.circuit_depth)
            }
            BackendType::Distributed => {
                format!("Distributed simulator recommended for {} qubits, {} gates. Memory requirement ({:.1} MB) exceeds single-node capacity.",
                       characteristics.num_qubits, characteristics.num_gates, characteristics.memory_requirement as f64 / (1024.0 * 1024.0))
            }
            BackendType::Auto => "Automatic backend selection".to_string(),
        }
    }

    /// Estimate execution time for backend and characteristics
    fn estimate_execution_time(
        &self,
        backend_type: BackendType,
        characteristics: &CircuitCharacteristics,
    ) -> Duration {
        let base_time_ms = match backend_type {
            BackendType::StateVector => characteristics.num_gates as u64 * 10,
            BackendType::SciRS2Gpu => characteristics.num_gates as u64 * 2,
            BackendType::LargeScale => characteristics.num_gates as u64 * 5,
            BackendType::Distributed => characteristics.num_gates as u64 * 15,
            BackendType::Auto => characteristics.num_gates as u64 * 10,
        };

        // Apply complexity factor
        let complexity_factor = characteristics.complexity_score.mul_add(2.0, 1.0) as u64;
        Duration::from_millis(base_time_ms * complexity_factor)
    }

    /// Execute circuit with specified backend
    fn execute_with_backend<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        backend_type: BackendType,
    ) -> Result<Register<N>> {
        match backend_type {
            BackendType::StateVector => {
                let simulator = StateVectorSimulator::new();
                simulator
                    .run(circuit)
                    .map_err(|e| SimulatorError::ComputationError(e.to_string()))
                    .and_then(|result| {
                        Register::with_amplitudes(result.amplitudes().to_vec())
                            .map_err(|e| SimulatorError::ComputationError(e.to_string()))
                    })
            }
            BackendType::SciRS2Gpu => {
                #[cfg(all(feature = "gpu", not(target_os = "macos")))]
                {
                    let mut simulator = SciRS2GpuStateVectorSimulator::new()
                        .map_err(|e| SimulatorError::ComputationError(e.to_string()))?;
                    use crate::simulator::Simulator;
                    simulator
                        .run(circuit)
                        .map_err(|e| SimulatorError::ComputationError(e.to_string()))
                        .and_then(|result| {
                            Register::with_amplitudes(result.amplitudes().to_vec())
                                .map_err(|e| SimulatorError::ComputationError(e.to_string()))
                        })
                }
                #[cfg(any(not(feature = "gpu"), target_os = "macos"))]
                {
                    // Fallback to state vector if GPU not available
                    let simulator = StateVectorSimulator::new();
                    simulator
                        .run(circuit)
                        .map_err(|e| SimulatorError::ComputationError(e.to_string()))
                        .and_then(|result| {
                            Register::with_amplitudes(result.amplitudes().to_vec())
                                .map_err(|e| SimulatorError::ComputationError(e.to_string()))
                        })
                }
            }
            BackendType::LargeScale => {
                // Create large-scale simulator with optimized configuration
                let config = LargeScaleSimulatorConfig::default();
                let simulator = LargeScaleQuantumSimulator::new(config)
                    .map_err(|e| SimulatorError::ComputationError(e.to_string()))?;
                simulator
                    .run(circuit)
                    .map_err(|e| SimulatorError::ComputationError(e.to_string()))
            }
            BackendType::Distributed => {
                // Fallback to large-scale for now
                let config = LargeScaleSimulatorConfig::default();
                let simulator = LargeScaleQuantumSimulator::new(config)
                    .map_err(|e| SimulatorError::ComputationError(e.to_string()))?;
                simulator
                    .run(circuit)
                    .map_err(|e| SimulatorError::ComputationError(e.to_string()))
            }
            BackendType::Auto => {
                // This should not happen, but fallback to state vector
                let simulator = StateVectorSimulator::new();
                simulator
                    .run(circuit)
                    .map_err(|e| SimulatorError::ComputationError(e.to_string()))
            }
        }
    }

    /// Convert Register to `SimulatorResult`
    fn register_to_simulator_result<const N: usize>(
        &self,
        register: Register<N>,
    ) -> SimulatorResult<N> {
        // Extract amplitudes from register
        let amplitudes = register.amplitudes().to_vec();

        SimulatorResult {
            amplitudes,
            num_qubits: N,
        }
    }

    /// Record performance metrics for future optimization
    fn record_performance_metrics<const N: usize>(
        &mut self,
        circuit: &Circuit<N>,
        backend_type: BackendType,
        execution_time: Duration,
    ) {
        let metrics = PerformanceMetrics {
            execution_time,
            memory_usage: 0,      // Would be measured in practice
            cpu_utilization: 0.0, // Would be measured in practice
            gpu_utilization: None,
            throughput: circuit.num_gates() as f64 / execution_time.as_secs_f64(),
            error_rate: 0.0,
        };

        let history_entry = PerformanceHistory {
            circuit_hash: self.compute_circuit_hash(circuit),
            backend_type,
            metrics,
            timestamp: Instant::now(),
        };

        self.performance_cache.push(history_entry);

        // Maintain cache size limit
        if self.performance_cache.len() > self.config.performance_cache_size {
            self.performance_cache.remove(0);
        }
    }

    /// Compute hash for circuit caching
    fn compute_circuit_hash<const N: usize>(&self, circuit: &Circuit<N>) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        circuit.num_gates().hash(&mut hasher);
        circuit.num_qubits().hash(&mut hasher);

        for gate in circuit.gates() {
            gate.name().hash(&mut hasher);
            gate.qubits().len().hash(&mut hasher);
        }

        hasher.finish()
    }

    /// Get human-readable backend type name
    const fn backend_type_name(&self, backend_type: BackendType) -> &'static str {
        match backend_type {
            BackendType::StateVector => "CPU StateVector",
            BackendType::SciRS2Gpu => "SciRS2 GPU",
            BackendType::LargeScale => "Large-Scale",
            BackendType::Distributed => "Distributed",
            BackendType::Auto => "Auto",
        }
    }

    /// Get optimization statistics
    #[must_use]
    pub fn get_performance_summary(&self) -> String {
        let total_circuits = self.performance_cache.len();
        if total_circuits == 0 {
            return "No performance data available".to_string();
        }

        let avg_execution_time = self
            .performance_cache
            .iter()
            .map(|entry| entry.metrics.execution_time.as_millis())
            .sum::<u128>()
            / total_circuits as u128;

        let backend_usage: HashMap<BackendType, usize> =
            self.performance_cache
                .iter()
                .fold(HashMap::new(), |mut acc, entry| {
                    *acc.entry(entry.backend_type).or_insert(0) += 1;
                    acc
                });

        let mut summary = "AutoOptimizer Performance Summary\n".to_string();
        writeln!(summary, "Total circuits processed: {total_circuits}")
            .expect("Writing to String should never fail");
        writeln!(summary, "Average execution time: {avg_execution_time}ms")
            .expect("Writing to String should never fail");
        summary.push_str("Backend usage:\n");

        for (backend, count) in backend_usage {
            let percentage = (count as f64 / total_circuits as f64) * 100.0;
            writeln!(
                summary,
                "  {}: {} ({:.1}%)",
                self.backend_type_name(backend),
                count,
                percentage
            )
            .expect("Writing to String should never fail");
        }

        summary
    }
}

impl Default for AutoOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl SciRS2CircuitAnalyzer {
    /// Analyze circuit using `SciRS2` tools (placeholder for future `SciRS2` integration)
    const fn analyze_circuit_with_scirs2<const N: usize>(
        &self,
        _circuit: &Circuit<N>,
    ) -> QuantRS2Result<f64> {
        // Placeholder for SciRS2-specific circuit analysis
        // Would use scirs2_core analysis tools when available
        Ok(0.7) // Mock analysis result
    }
}

/// Convenience function to execute a circuit with automatic optimization
pub fn execute_with_auto_optimization<const N: usize>(
    circuit: &Circuit<N>,
) -> Result<SimulatorResult<N>> {
    let mut optimizer = AutoOptimizer::new();
    optimizer.execute_optimized(circuit)
}

/// Convenience function to get backend recommendation for a circuit
pub fn recommend_backend_for_circuit<const N: usize>(
    circuit: &Circuit<N>,
) -> QuantRS2Result<BackendRecommendation> {
    let mut optimizer = AutoOptimizer::new();
    optimizer.recommend_backend(circuit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_circuit::builder::CircuitBuilder;

    #[test]
    fn test_auto_optimizer_creation() {
        let optimizer = AutoOptimizer::new();
        assert!(optimizer.config.enable_profiling);
    }

    #[test]
    fn test_circuit_characteristics_analysis() {
        let optimizer = AutoOptimizer::new();

        // Create a simple test circuit
        let mut builder = CircuitBuilder::<4>::new();
        let _ = builder.h(0);
        let _ = builder.cnot(0, 1);
        let _ = builder.h(2);
        let _ = builder.cnot(2, 3);
        let circuit = builder.build();

        let characteristics = optimizer
            .analyze_circuit(&circuit)
            .expect("Failed to analyze circuit characteristics");

        assert_eq!(characteristics.num_qubits, 4);
        assert_eq!(characteristics.num_gates, 4);
        assert!(characteristics.circuit_depth > 0);
        assert!(characteristics.two_qubit_density > 0.0);
    }

    #[test]
    fn test_backend_recommendation() {
        let mut optimizer = AutoOptimizer::new();

        // Create a small circuit
        let mut builder = CircuitBuilder::<2>::new();
        let _ = builder.h(0);
        let _ = builder.cnot(0, 1);
        let circuit = builder.build();

        let recommendation = optimizer
            .recommend_backend(&circuit)
            .expect("Failed to get backend recommendation");

        assert!(recommendation.confidence > 0.0);
        assert!(!recommendation.reasoning.is_empty());
    }

    #[test]
    fn test_execute_with_optimization() {
        let mut optimizer = AutoOptimizer::new();

        // Create a simple circuit
        let mut builder = CircuitBuilder::<2>::new();
        let _ = builder.h(0);
        let _ = builder.cnot(0, 1);
        let circuit = builder.build();

        let result = optimizer.execute_optimized(&circuit);
        assert!(result.is_ok());

        if let Ok(sim_result) = result {
            assert_eq!(sim_result.num_qubits, 2);
            assert_eq!(sim_result.amplitudes.len(), 4);
        }
    }

    #[test]
    fn test_convenience_functions() {
        // Create a simple circuit
        let mut builder = CircuitBuilder::<2>::new();
        let _ = builder.h(0);
        let _ = builder.cnot(0, 1);
        let circuit = builder.build();

        // Test recommendation function
        let recommendation = recommend_backend_for_circuit(&circuit);
        assert!(recommendation.is_ok());

        // Test execution function
        let result = execute_with_auto_optimization(&circuit);
        assert!(result.is_ok());
    }
}
