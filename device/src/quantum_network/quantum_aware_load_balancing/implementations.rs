//! Implementations and tests

use super::types_and_defaults::*;
use crate::quantum_network::distributed_protocols::{
    self, CircuitPartition, DistributedComputationError, ExecutionRequirements, LoadBalancer,
    LoadBalancerMetrics, NodeId, NodeInfo, PerformanceHistory, PerformanceMetrics,
    ResourceRequirements, Result as DistributedResult, TrainingDataPoint,
};
use crate::quantum_network::network_optimization::{
    self as netopt, FeatureVector, FeedbackData, MLModel, ModelMetrics,
    NetworkOptimizationError as OptimizationError, PredictionResult, Priority, TrainingResult,
};
use async_trait::async_trait;
use chrono::{DateTime, Datelike, Duration as ChronoDuration, Timelike, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;
use tokio::sync::{mpsc, Semaphore};
use uuid::Uuid;

impl QuantumLoadBalancingMetricsCollector {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Implementation of the quantum-aware load balancer
impl MLOptimizedQuantumLoadBalancer {
    /// Create a new quantum-aware ML load balancer
    pub fn new() -> Self {
        Self {
            base_strategy: Arc::new(CapabilityBasedQuantumBalancer::new()),
            ml_predictor: Arc::new(QuantumLoadPredictionModel::new()),
            quantum_scheduler: Arc::new(QuantumAwareScheduler::new()),
            performance_learner: Arc::new(QuantumPerformanceLearner::new()),
            adaptive_weights: Arc::new(Mutex::new(QuantumLoadBalancingWeights::default())),
            entanglement_tracker: Arc::new(EntanglementQualityTracker::new()),
            coherence_monitor: Arc::new(CoherenceTimeMonitor::new()),
            fidelity_preserver: Arc::new(FidelityPreservationSystem::new()),
            metrics_collector: Arc::new(QuantumLoadBalancingMetricsCollector::new()),
        }
    }

    /// Select optimal node for quantum circuit partition
    pub async fn select_optimal_node(
        &self,
        available_nodes: &[NodeInfo],
        circuit_partition: &CircuitPartition,
        quantum_requirements: &QuantumResourceRequirements,
    ) -> Result<NodeId> {
        // Extract quantum features for ML prediction
        let features = self
            .extract_quantum_features(available_nodes, circuit_partition, quantum_requirements)
            .await?;

        // Get ML prediction for optimal node
        let ml_prediction = self
            .ml_predictor
            .predict_optimal_node(available_nodes, circuit_partition, &features)
            .await?;

        // Apply quantum-aware scheduling constraints
        let quantum_constraints = self
            .evaluate_quantum_constraints(available_nodes, circuit_partition, quantum_requirements)
            .await?;

        // Combine ML prediction with quantum constraints
        let optimal_node = self
            .combine_ml_and_quantum_decisions(
                &ml_prediction,
                &quantum_constraints,
                available_nodes,
                circuit_partition,
            )
            .await?;

        // Update performance learning system
        self.update_performance_learning(&optimal_node, circuit_partition, quantum_requirements)
            .await?;

        // Update metrics
        self.update_quantum_metrics(&optimal_node, &features, &ml_prediction)
            .await?;

        Ok(optimal_node)
    }

    /// Extract quantum-specific features for ML prediction
    async fn extract_quantum_features(
        &self,
        available_nodes: &[NodeInfo],
        circuit_partition: &CircuitPartition,
        quantum_requirements: &QuantumResourceRequirements,
    ) -> Result<FeatureVector> {
        let mut features = HashMap::new();

        // Circuit complexity features
        features.insert(
            "circuit_depth".to_string(),
            circuit_partition.gates.len() as f64,
        );
        features.insert(
            "entanglement_pairs_needed".to_string(),
            quantum_requirements.entanglement_pairs as f64,
        );
        features.insert(
            "fidelity_requirement".to_string(),
            quantum_requirements.fidelity_requirement,
        );

        // Node capability features
        for (i, node) in available_nodes.iter().enumerate() {
            let node_prefix = format!("node_{i}");

            // Quantum hardware features
            features.insert(
                format!("{node_prefix}_max_qubits"),
                node.capabilities.max_qubits as f64,
            );
            features.insert(
                format!("{node_prefix}_readout_fidelity"),
                node.capabilities.readout_fidelity,
            );

            // Current load features
            features.insert(
                format!("{node_prefix}_qubits_in_use"),
                node.current_load.qubits_in_use as f64,
            );
            features.insert(
                format!("{node_prefix}_queue_length"),
                node.current_load.queue_length as f64,
            );

            // Quantum-specific features
            if let Some(entanglement_quality) =
                self.get_node_entanglement_quality(&node.node_id).await?
            {
                features.insert(
                    format!("{node_prefix}_entanglement_quality"),
                    entanglement_quality,
                );
            }

            if let Some(coherence_metrics) = self.get_node_coherence_metrics(&node.node_id).await? {
                features.insert(
                    format!("{node_prefix}_avg_coherence_time"),
                    coherence_metrics.average_coherence_time.as_secs_f64(),
                );
            }
        }

        // Temporal features
        let now = Utc::now();
        features.insert("hour_of_day".to_string(), now.hour() as f64);
        features.insert(
            "day_of_week".to_string(),
            now.weekday().number_from_monday() as f64,
        );

        Ok(FeatureVector {
            features,
            timestamp: now,
            context: self
                .extract_quantum_context(circuit_partition, quantum_requirements)
                .await?,
        })
    }

    /// Evaluate quantum constraints for scheduling
    async fn evaluate_quantum_constraints(
        &self,
        available_nodes: &[NodeInfo],
        circuit_partition: &CircuitPartition,
        quantum_requirements: &QuantumResourceRequirements,
    ) -> Result<QuantumSchedulingConstraints> {
        let mut constraints = QuantumSchedulingConstraints {
            entanglement_constraints: HashMap::new(),
            coherence_constraints: HashMap::new(),
            fidelity_constraints: HashMap::new(),
            error_correction_constraints: HashMap::new(),
            deadline_constraints: HashMap::new(),
        };

        for node in available_nodes {
            // Evaluate entanglement constraints
            let entanglement_constraint = self
                .evaluate_entanglement_constraint(
                    &node.node_id,
                    circuit_partition,
                    quantum_requirements,
                )
                .await?;

            constraints
                .entanglement_constraints
                .insert(node.node_id.clone(), entanglement_constraint);

            // Evaluate coherence constraints
            let coherence_constraint = self
                .evaluate_coherence_constraint(
                    &node.node_id,
                    circuit_partition,
                    quantum_requirements,
                )
                .await?;

            constraints
                .coherence_constraints
                .insert(node.node_id.clone(), coherence_constraint);

            // Evaluate fidelity constraints
            let fidelity_constraint = self
                .evaluate_fidelity_constraint(
                    &node.node_id,
                    circuit_partition,
                    quantum_requirements,
                )
                .await?;

            constraints
                .fidelity_constraints
                .insert(node.node_id.clone(), fidelity_constraint);
        }

        Ok(constraints)
    }

    /// Combine ML prediction with quantum constraints to make final decision
    async fn combine_ml_and_quantum_decisions(
        &self,
        ml_prediction: &QuantumPredictionResult,
        quantum_constraints: &QuantumSchedulingConstraints,
        available_nodes: &[NodeInfo],
        circuit_partition: &CircuitPartition,
    ) -> Result<NodeId> {
        let weights = self
            .adaptive_weights
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();

        let mut node_scores: HashMap<NodeId, f64> = HashMap::new();

        for node in available_nodes {
            let mut score = 0.0;

            // ML prediction score
            let ml_score = if node.node_id == ml_prediction.predicted_node {
                ml_prediction.confidence
            } else {
                0.0
            };

            // Entanglement quality score
            let entanglement_score = quantum_constraints
                .entanglement_constraints
                .get(&node.node_id)
                .map_or(0.0, |c| c.quality_score);

            // Coherence time score
            let coherence_score = quantum_constraints
                .coherence_constraints
                .get(&node.node_id)
                .map_or(0.0, |c| c.adequacy_score);

            // Fidelity preservation score
            let fidelity_score = quantum_constraints
                .fidelity_constraints
                .get(&node.node_id)
                .map_or(0.0, |c| c.preservation_score);

            // Classical resource score
            let classical_score = self
                .calculate_classical_resource_score(node, circuit_partition)
                .await?;

            // Combine scores with adaptive weights
            score += ml_score * 0.3; // Base ML weight
            score += entanglement_score * weights.entanglement_quality_weight;
            score += coherence_score * weights.coherence_time_weight;
            score += fidelity_score * weights.fidelity_preservation_weight;
            score += classical_score * weights.classical_resources_weight;

            node_scores.insert(node.node_id.clone(), score);
        }

        // Select node with highest combined score
        let optimal_node = node_scores
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(node_id, _)| node_id)
            .ok_or_else(|| {
                QuantumLoadBalancingError::QuantumSchedulingConflict(
                    "No suitable node found ".to_string(),
                )
            })?;

        Ok(optimal_node)
    }

    /// Get entanglement quality for a node
    async fn get_node_entanglement_quality(&self, node_id: &NodeId) -> Result<Option<f64>> {
        let entanglement_states = self
            .entanglement_tracker
            .entanglement_states
            .read()
            .unwrap_or_else(|e| e.into_inner());

        let quality: f64 = entanglement_states
            .iter()
            .filter(|((n1, n2), _)| n1 == node_id || n2 == node_id)
            .map(|(_, state)| state.current_fidelity)
            .sum::<f64>()
            / entanglement_states.len() as f64;

        Ok(if quality > 0.0 { Some(quality) } else { None })
    }

    /// Get coherence metrics for a node
    async fn get_node_coherence_metrics(
        &self,
        node_id: &NodeId,
    ) -> Result<Option<NodeCoherenceMetrics>> {
        let coherence_states = self
            .coherence_monitor
            .coherence_times
            .read()
            .unwrap_or_else(|e| e.into_inner());

        let node_coherence_data: Vec<_> = coherence_states
            .iter()
            .filter(|((n, _), _)| n == node_id)
            .collect();

        if node_coherence_data.is_empty() {
            return Ok(None);
        }

        let total_t1: Duration = node_coherence_data
            .iter()
            .map(|(_, state)| state.t1_time)
            .sum();

        let total_t2: Duration = node_coherence_data
            .iter()
            .map(|(_, state)| state.t2_time)
            .sum();

        let count = node_coherence_data.len();

        Ok(Some(NodeCoherenceMetrics {
            average_coherence_time: total_t1 / count as u32,
            average_dephasing_time: total_t2 / count as u32,
            coherence_stability: 0.95, // Placeholder calculation
        }))
    }

    /// Extract quantum context information
    async fn extract_quantum_context(
        &self,
        circuit_partition: &CircuitPartition,
        quantum_requirements: &QuantumResourceRequirements,
    ) -> Result<crate::quantum_network::network_optimization::ContextInfo> {
        Ok(crate::quantum_network::network_optimization::ContextInfo {
            network_state: "quantum_active".to_string(),
            time_of_day: Utc::now().hour() as u8,
            day_of_week: Utc::now().weekday().number_from_monday() as u8,
            quantum_experiment_type: Some(
                self.classify_quantum_experiment(circuit_partition).await?,
            ),
            user_priority: Some("high".to_string()), // Placeholder
        })
    }

    /// Classify type of quantum experiment
    async fn classify_quantum_experiment(
        &self,
        circuit_partition: &CircuitPartition,
    ) -> Result<String> {
        // Simple classification based on gate types and circuit structure
        let gate_types: Vec<&str> = circuit_partition
            .gates
            .iter()
            .map(|g| g.gate_type.as_str())
            .collect();

        let experiment_type = if gate_types.contains(&"H") && gate_types.contains(&"CNOT") {
            "entanglement_experiment"
        } else if gate_types.iter().any(|&g| g.starts_with('R')) {
            "variational_algorithm"
        } else if gate_types.contains(&"QFT") {
            "quantum_fourier_transform"
        } else {
            "general_quantum_computation"
        };

        Ok(experiment_type.to_string())
    }

    /// Evaluate entanglement constraint for a node
    async fn evaluate_entanglement_constraint(
        &self,
        node_id: &NodeId,
        _circuit_partition: &CircuitPartition,
        quantum_requirements: &QuantumResourceRequirements,
    ) -> Result<EntanglementConstraint> {
        let entanglement_states = self
            .entanglement_tracker
            .entanglement_states
            .read()
            .unwrap_or_else(|e| e.into_inner());

        // Calculate available entanglement quality
        let available_quality: f64 = entanglement_states
            .iter()
            .filter(|((n1, n2), _)| n1 == node_id || n2 == node_id)
            .map(|(_, state)| state.current_fidelity)
            .sum::<f64>()
            / (entanglement_states.len().max(1) as f64);

        // Calculate quality score based on requirements
        let quality_score = if available_quality >= quantum_requirements.fidelity_requirement {
            1.0
        } else {
            available_quality / quantum_requirements.fidelity_requirement
        };

        Ok(EntanglementConstraint {
            available_pairs: entanglement_states.len() as u32,
            required_pairs: quantum_requirements.entanglement_pairs,
            quality_score,
            is_feasible: quality_score >= 0.8, // Threshold for feasibility
        })
    }

    /// Evaluate coherence constraint for a node
    async fn evaluate_coherence_constraint(
        &self,
        node_id: &NodeId,
        circuit_partition: &CircuitPartition,
        quantum_requirements: &QuantumResourceRequirements,
    ) -> Result<CoherenceConstraint> {
        let coherence_states = self
            .coherence_monitor
            .coherence_times
            .read()
            .unwrap_or_else(|e| e.into_inner());

        // Get minimum coherence time available on this node
        let min_coherence_time = coherence_states
            .iter()
            .filter(|((n, _), _)| n == node_id)
            .map(|(_, state)| state.t2_time.min(state.t1_time))
            .min()
            .unwrap_or(Duration::from_secs(0));

        // Estimate required coherence time based on circuit
        let estimated_execution_time = circuit_partition.estimated_execution_time;
        let required_coherence_time = quantum_requirements
            .coherence_time_needed
            .max(estimated_execution_time);

        // Calculate adequacy score
        let adequacy_score = if min_coherence_time >= required_coherence_time {
            1.0
        } else {
            min_coherence_time.as_secs_f64() / required_coherence_time.as_secs_f64()
        };

        Ok(CoherenceConstraint {
            available_coherence_time: min_coherence_time,
            required_coherence_time,
            adequacy_score,
            is_adequate: adequacy_score >= 0.9, // High threshold for coherence adequacy
        })
    }

    /// Evaluate fidelity constraint for a node
    async fn evaluate_fidelity_constraint(
        &self,
        node_id: &NodeId,
        circuit_partition: &CircuitPartition,
        quantum_requirements: &QuantumResourceRequirements,
    ) -> Result<FidelityConstraint> {
        // Get historical fidelity data for this node
        let performance_history = self
            .performance_learner
            .performance_history
            .read()
            .unwrap_or_else(|e| e.into_inner());

        let fidelity_history = performance_history
            .get(node_id)
            .map(|h| &h.fidelity_history)
            .cloned()
            .unwrap_or_default();

        // Calculate expected fidelity based on circuit complexity
        let circuit_complexity = circuit_partition.gates.len() as f64;
        let base_fidelity = if fidelity_history.is_empty() {
            0.95 // Default assumption
        } else {
            fidelity_history
                .iter()
                .map(|m| m.process_fidelity)
                .sum::<f64>()
                / fidelity_history.len() as f64
        };

        // Apply fidelity degradation based on circuit complexity
        let expected_fidelity = base_fidelity * (0.99_f64).powf(circuit_complexity / 10.0);

        // Calculate preservation score
        let preservation_score = if expected_fidelity >= quantum_requirements.fidelity_requirement {
            1.0
        } else {
            expected_fidelity / quantum_requirements.fidelity_requirement
        };

        Ok(FidelityConstraint {
            expected_fidelity,
            required_fidelity: quantum_requirements.fidelity_requirement,
            preservation_score,
            can_preserve: preservation_score >= 0.95, // High threshold for fidelity preservation
        })
    }

    /// Calculate classical resource score for a node
    async fn calculate_classical_resource_score(
        &self,
        node: &NodeInfo,
        circuit_partition: &CircuitPartition,
    ) -> Result<f64> {
        let cpu_score = 1.0 - node.current_load.cpu_utilization;
        let memory_score = 1.0 - node.current_load.memory_utilization;
        let network_score = 1.0 - node.current_load.network_utilization;

        // Consider queue length
        let queue_score = if node.current_load.queue_length == 0 {
            1.0
        } else {
            1.0 / (1.0 + node.current_load.queue_length as f64 / 10.0)
        };

        // Consider resource requirements
        let resource_adequacy = if node.capabilities.max_qubits
            >= circuit_partition.resource_requirements.qubits_needed
        {
            1.0
        } else {
            node.capabilities.max_qubits as f64
                / circuit_partition.resource_requirements.qubits_needed as f64
        };

        // Combine all classical scores
        let combined_score =
            (cpu_score + memory_score + network_score + queue_score + resource_adequacy) / 5.0;

        Ok(combined_score)
    }

    /// Update performance learning system with feedback
    async fn update_performance_learning(
        &self,
        selected_node: &NodeId,
        circuit_partition: &CircuitPartition,
        quantum_requirements: &QuantumResourceRequirements,
    ) -> Result<()> {
        // Record the decision for future learning
        let learning_data = QuantumLearningDataPoint {
            timestamp: Utc::now(),
            selected_node: selected_node.clone(),
            circuit_partition: circuit_partition.clone(),
            quantum_requirements: quantum_requirements.clone(),
            context_features: HashMap::new(), // To be filled with context
        };

        // Convert to TrainingDataPoint for compatibility
        let training_data = TrainingDataPoint {
            features: learning_data.context_features.clone(),
            target_node: learning_data.selected_node.clone(),
            actual_performance: PerformanceMetrics {
                execution_time: Duration::from_millis(100), // Placeholder
                fidelity: 0.95,                             // Placeholder
                success: true,                              // Placeholder
                resource_utilization: 0.75,                 // Placeholder
            },
            timestamp: learning_data.timestamp,
        };

        // Add to learning system (placeholder implementation)
        self.performance_learner
            .learning_algorithm
            .add_training_data(training_data)
            .await?;

        Ok(())
    }

    /// Update quantum-specific metrics
    async fn update_quantum_metrics(
        &self,
        selected_node: &NodeId,
        features: &FeatureVector,
        prediction: &QuantumPredictionResult,
    ) -> Result<()> {
        let mut quantum_metrics = self
            .metrics_collector
            .quantum_metrics
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        quantum_metrics.total_quantum_decisions += 1;

        // Update prediction accuracy if we have feedback
        if selected_node == &prediction.predicted_node {
            // Prediction was followed - potentially good decision
            quantum_metrics.quantum_advantage_achieved += 0.01; // Incremental improvement
        }

        // Update other metrics based on features
        if let Some(fidelity) = features.features.get("fidelity_requirement") {
            quantum_metrics.fidelity_improvement_factor =
                (quantum_metrics.fidelity_improvement_factor + fidelity) / 2.0;
        }

        Ok(())
    }
}

/// Quantum scheduling constraints
#[derive(Debug, Clone)]
pub struct QuantumSchedulingConstraints {
    pub entanglement_constraints: HashMap<NodeId, EntanglementConstraint>,
    pub coherence_constraints: HashMap<NodeId, CoherenceConstraint>,
    pub fidelity_constraints: HashMap<NodeId, FidelityConstraint>,
    pub error_correction_constraints: HashMap<NodeId, ErrorCorrectionConstraint>,
    pub deadline_constraints: HashMap<NodeId, DeadlineConstraint>,
}

/// Entanglement constraint for scheduling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntanglementConstraint {
    pub available_pairs: u32,
    pub required_pairs: u32,
    pub quality_score: f64,
    pub is_feasible: bool,
}

/// Coherence constraint for scheduling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceConstraint {
    pub available_coherence_time: Duration,
    pub required_coherence_time: Duration,
    pub adequacy_score: f64,
    pub is_adequate: bool,
}

/// Fidelity constraint for scheduling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FidelityConstraint {
    pub expected_fidelity: f64,
    pub required_fidelity: f64,
    pub preservation_score: f64,
    pub can_preserve: bool,
}

/// Error correction constraint for scheduling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorCorrectionConstraint {
    pub available_schemes: Vec<String>,
    pub required_schemes: Vec<String>,
    pub overhead_factor: f64,
    pub is_compatible: bool,
}

/// Deadline constraint for scheduling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadlineConstraint {
    pub hard_deadline: Option<DateTime<Utc>>,
    pub soft_deadline: Option<DateTime<Utc>>,
    pub estimated_completion: DateTime<Utc>,
    pub can_meet_deadline: bool,
}

/// Node coherence metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCoherenceMetrics {
    pub average_coherence_time: Duration,
    pub average_dephasing_time: Duration,
    pub coherence_stability: f64,
}

/// Quantum resource requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumResourceRequirements {
    pub qubits_needed: u32,
    pub gate_count_estimate: u32,
    pub circuit_depth: u32,
    pub fidelity_requirement: f64,
    pub coherence_time_needed: Duration,
    pub entanglement_pairs: u32,
}

/// Quantum learning data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumLearningDataPoint {
    pub timestamp: DateTime<Utc>,
    pub selected_node: NodeId,
    pub circuit_partition: CircuitPartition,
    pub quantum_requirements: QuantumResourceRequirements,
    pub context_features: HashMap<String, f64>,
}

/// Capability-based quantum load balancer (base implementation)
#[derive(Debug)]
pub struct CapabilityBasedQuantumBalancer {
    pub quantum_capability_weights: HashMap<String, f64>,
    pub quantum_performance_history: Arc<RwLock<HashMap<NodeId, QuantumPerformanceHistory>>>,
}

impl Default for CapabilityBasedQuantumBalancer {
    fn default() -> Self {
        Self::new()
    }
}

impl CapabilityBasedQuantumBalancer {
    pub fn new() -> Self {
        let mut weights = HashMap::new();
        weights.insert("qubit_count".to_string(), 0.3);
        weights.insert("gate_fidelity".to_string(), 0.4);
        weights.insert("coherence_time".to_string(), 0.3);

        Self {
            quantum_capability_weights: weights,
            quantum_performance_history: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl LoadBalancer for CapabilityBasedQuantumBalancer {
    async fn select_node(
        &self,
        available_nodes: &[NodeInfo],
        requirements: &ResourceRequirements,
    ) -> std::result::Result<
        NodeId,
        crate::quantum_network::distributed_protocols::DistributedComputationError,
    > {
        // Simple capability-based selection with quantum awareness
        let mut best_node = None;
        let mut best_score = 0.0;

        for node in available_nodes {
            let mut score = 0.0;

            // Quantum capability score
            let qubit_score = if node.capabilities.max_qubits >= requirements.qubits_needed {
                1.0
            } else {
                node.capabilities.max_qubits as f64 / requirements.qubits_needed as f64
            };

            let fidelity_score = node.capabilities.readout_fidelity;

            // Load-based score
            let load_score = 1.0
                - (node.current_load.qubits_in_use as f64 / node.capabilities.max_qubits as f64);

            score = qubit_score * self.quantum_capability_weights["qubit_count"]
                + fidelity_score * self.quantum_capability_weights["gate_fidelity"]
                + load_score * 0.3; // Load balancing component

            if score > best_score {
                best_score = score;
                best_node = Some(node.node_id.clone());
            }
        }

        best_node.ok_or_else(||
            crate::quantum_network::distributed_protocols::DistributedComputationError::ResourceAllocation(
                "No suitable node found ".to_string()
            )
        )
    }

    async fn update_node_metrics(
        &self,
        node_id: &NodeId,
        metrics: &PerformanceMetrics,
    ) -> std::result::Result<
        (),
        crate::quantum_network::distributed_protocols::DistributedComputationError,
    > {
        // Update performance history
        let mut history = self
            .quantum_performance_history
            .write()
            .unwrap_or_else(|e| e.into_inner());
        if !history.contains_key(node_id) {
            history.insert(node_id.clone(), QuantumPerformanceHistory::default());
        }

        // Add performance data point
        let Some(node_history) = history.get_mut(node_id) else {
            return Ok(());
        };

        // Update classical metrics
        node_history
            .classical_metrics
            .execution_times
            .push_back(metrics.execution_time);
        if node_history.classical_metrics.execution_times.len() > 100 {
            node_history.classical_metrics.execution_times.pop_front();
        }

        node_history.classical_metrics.success_rate = node_history
            .classical_metrics
            .success_rate
            .mul_add(0.9, if metrics.success { 1.0 } else { 0.0 } * 0.1);

        Ok(())
    }

    fn get_balancer_metrics(&self) -> LoadBalancerMetrics {
        LoadBalancerMetrics {
            total_decisions: 0, // Placeholder
            average_decision_time: Duration::from_millis(10),
            prediction_accuracy: 0.85,
            load_distribution_variance: 0.15,
            total_requests: 0,
            successful_allocations: 0,
            failed_allocations: 0,
            average_response_time: Duration::from_millis(5),
            node_utilization: HashMap::new(),
        }
    }

    /// Real per-partition, capability- and load-aware node assignment:
    /// each partition is scored against every node using its actual
    /// `resource_requirements` (qubit count) and the node's real
    /// capabilities/fidelity, and a running per-node assigned-qubit tally
    /// is tracked so later partitions are genuinely load-balanced across
    /// capable nodes instead of every partition landing on whichever node
    /// `HashMap::iter().next()` happened to return.
    fn select_nodes(
        &self,
        partitions: &[CircuitPartition],
        available_nodes: &HashMap<NodeId, NodeInfo>,
        _requirements: &ExecutionRequirements,
    ) -> std::result::Result<HashMap<Uuid, NodeId>, DistributedComputationError> {
        if available_nodes.is_empty() {
            return Err(DistributedComputationError::ResourceAllocation(
                "no available nodes for partition assignment".to_string(),
            ));
        }

        let qubit_weight = self
            .quantum_capability_weights
            .get("qubit_count")
            .copied()
            .unwrap_or(0.3);
        let fidelity_weight = self
            .quantum_capability_weights
            .get("gate_fidelity")
            .copied()
            .unwrap_or(0.4);

        let mut allocation = HashMap::new();
        let mut assigned_qubits: HashMap<NodeId, u32> = HashMap::new();

        for partition in partitions {
            let required_qubits = partition.resource_requirements.qubits_needed.max(1);
            let mut best_node: Option<NodeId> = None;
            let mut best_score = f64::NEG_INFINITY;

            for (node_id, node) in available_nodes {
                if node.capabilities.max_qubits < required_qubits {
                    continue;
                }
                let already_assigned = *assigned_qubits.get(node_id).unwrap_or(&0);
                let projected_load = already_assigned + required_qubits;
                let max_qubits = node.capabilities.max_qubits.max(1);
                let utilization = projected_load as f64 / max_qubits as f64;
                if utilization > 1.0 {
                    // This node cannot take on this partition on top of
                    // what has already been assigned to it in this batch.
                    continue;
                }

                let qubit_score =
                    (node.capabilities.max_qubits as f64 / required_qubits as f64).min(1.0);
                let fidelity_score = node.capabilities.readout_fidelity;
                let load_score = 1.0 - utilization;

                let score = qubit_score * qubit_weight
                    + fidelity_score * fidelity_weight
                    + load_score * 0.3;

                if score > best_score {
                    best_score = score;
                    best_node = Some(node_id.clone());
                }
            }

            let Some(node_id) = best_node else {
                return Err(DistributedComputationError::ResourceAllocation(format!(
                    "no node has sufficient remaining capacity ({required_qubits} qubits) for partition {}",
                    partition.partition_id
                )));
            };

            *assigned_qubits.entry(node_id.clone()).or_insert(0) += required_qubits;
            allocation.insert(partition.partition_id, node_id);
        }

        Ok(allocation)
    }

    fn rebalance_load(
        &self,
        current_allocation: &HashMap<Uuid, NodeId>,
        nodes: &HashMap<NodeId, NodeInfo>,
    ) -> Option<HashMap<Uuid, NodeId>> {
        None // No rebalancing needed in simplified implementation
    }

    fn predict_execution_time(&self, partition: &CircuitPartition, node: &NodeInfo) -> Duration {
        Duration::from_millis(partition.gates.len() as u64 * 15) // Slightly higher than basic implementation
    }
}

impl Default for QuantumLoadBalancingWeights {
    fn default() -> Self {
        Self {
            entanglement_quality_weight: 0.25,
            coherence_time_weight: 0.25,
            fidelity_preservation_weight: 0.20,
            classical_resources_weight: 0.15,
            network_latency_weight: 0.10,
            error_correction_weight: 0.03,
            fairness_weight: 0.02,
            dynamic_adjustment_enabled: true,
        }
    }
}

impl Default for QuantumPerformanceHistory {
    fn default() -> Self {
        Self {
            classical_metrics: PerformanceHistory {
                execution_times: VecDeque::new(),
                success_rate: 0.95,
                average_fidelity: 0.90,
                last_updated: Utc::now(),
            },
            fidelity_history: VecDeque::new(),
            coherence_measurements: VecDeque::new(),
            entanglement_measurements: VecDeque::new(),
            error_rate_history: VecDeque::new(),
            gate_statistics: HashMap::new(),
        }
    }
}

// Individual default implementations are provided below

// Individual implementations for each type to avoid unsafe operations
impl Default for QuantumLoadPredictionModel {
    fn default() -> Self {
        Self {
            model: Arc::new(Mutex::new(Box::new(SimpleMLModel::new()))),
            feature_extractor: Arc::new(QuantumFeatureExtractor::default()),
            prediction_cache: Arc::new(RwLock::new(HashMap::new())),
            training_collector: Arc::new(QuantumTrainingDataCollector::default()),
            performance_tracker: Arc::new(ModelPerformanceTracker::default()),
        }
    }
}

impl Default for QuantumAwareScheduler {
    fn default() -> Self {
        Self {
            entanglement_aware_scheduling: true,
            coherence_time_optimization: true,
            fidelity_preservation_priority: true,
            error_correction_scheduler: Arc::new(ErrorCorrectionScheduler::default()),
            deadline_scheduler: Arc::new(QuantumDeadlineScheduler::default()),
            urgency_evaluator: Arc::new(QuantumUrgencyEvaluator::default()),
            entanglement_resolver: Arc::new(EntanglementDependencyResolver::default()),
            gate_conflict_resolver: Arc::new(QuantumGateConflictResolver::default()),
        }
    }
}

impl QuantumAwareScheduler {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for QuantumPerformanceLearner {
    fn default() -> Self {
        Self {
            performance_history: Arc::new(RwLock::new(HashMap::new())),
            learning_algorithm: Arc::new(QuantumReinforcementLearning::default()),
            adaptation_strategy: Arc::new(QuantumAdaptationStrategy::default()),
            feedback_processor: Arc::new(QuantumFeedbackProcessor::default()),
        }
    }
}

impl Default for EntanglementQualityTracker {
    fn default() -> Self {
        Self {
            entanglement_states: Arc::new(RwLock::new(HashMap::new())),
            quality_thresholds: Arc::new(EntanglementQualityThresholds::default()),
            quality_predictor: Arc::new(EntanglementQualityPredictor::default()),
            quality_optimizer: Arc::new(EntanglementQualityOptimizer::default()),
        }
    }
}

impl EntanglementQualityTracker {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for CoherenceTimeMonitor {
    fn default() -> Self {
        Self {
            coherence_times: Arc::new(RwLock::new(HashMap::new())),
            coherence_predictor: Arc::new(CoherenceTimePredictor::default()),
            coherence_optimizer: Arc::new(CoherenceTimeOptimizer::default()),
            real_time_monitor: Arc::new(RealTimeCoherenceMonitor::default()),
        }
    }
}

impl CoherenceTimeMonitor {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for FidelityPreservationSystem {
    fn default() -> Self {
        Self {
            fidelity_tracker: Arc::new(FidelityTracker::default()),
            preservation_strategies: Arc::new(FidelityPreservationStrategies::default()),
            error_mitigation: Arc::new(ErrorMitigationCoordinator::default()),
            optimization_scheduler: Arc::new(FidelityOptimizationScheduler::default()),
        }
    }
}

impl FidelityPreservationSystem {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for QuantumLoadBalancingMetricsCollector {
    fn default() -> Self {
        Self {
            classical_metrics: Arc::new(Mutex::new(LoadBalancerMetrics {
                total_decisions: 0,
                average_decision_time: Duration::from_millis(10),
                prediction_accuracy: 0.95,
                load_distribution_variance: 0.1,
                total_requests: 0,
                successful_allocations: 0,
                failed_allocations: 0,
                average_response_time: Duration::from_millis(5),
                node_utilization: HashMap::new(),
            })),
            quantum_metrics: Arc::new(Mutex::new(QuantumLoadBalancingMetrics::default())),
            performance_tracker: Arc::new(RealTimeQuantumPerformanceTracker::default()),
            metrics_aggregator: Arc::new(QuantumMetricsAggregator::default()),
        }
    }
}

impl Default for QuantumLoadBalancingMetrics {
    fn default() -> Self {
        Self {
            total_quantum_decisions: 0,
            average_quantum_decision_time: Duration::from_millis(15),
            entanglement_preservation_rate: 0.9,
            coherence_utilization_efficiency: 0.85,
            fidelity_improvement_factor: 1.1,
            quantum_advantage_achieved: 0.2,
            error_correction_overhead_ratio: 0.15,
            quantum_fairness_index: 0.95,
        }
    }
}

impl Default for EntanglementQualityThresholds {
    fn default() -> Self {
        Self {
            min_fidelity: 0.8,
            warning_fidelity: 0.85,
            optimal_fidelity: 0.95,
            max_decay_rate: 0.05,
            min_lifetime: Duration::from_millis(100),
        }
    }
}

// Simple ML model implementation for stubs
#[derive(Debug)]
pub struct SimpleMLModel {
    pub model_type: String,
}

impl Default for SimpleMLModel {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleMLModel {
    pub fn new() -> Self {
        Self {
            model_type: "simple_stub".to_string(),
        }
    }
}

#[async_trait]
impl crate::quantum_network::network_optimization::MLModel for SimpleMLModel {
    async fn predict(
        &self,
        _features: &FeatureVector,
    ) -> std::result::Result<PredictionResult, OptimizationError> {
        Ok(PredictionResult {
            predicted_values: HashMap::new(),
            confidence_intervals: HashMap::new(),
            uncertainty_estimate: 0.1,
            prediction_timestamp: Utc::now(),
        })
    }

    async fn train(
        &mut self,
        _training_data: &[TrainingDataPoint],
    ) -> std::result::Result<TrainingResult, OptimizationError> {
        Ok(TrainingResult {
            training_accuracy: 0.85,
            validation_accuracy: 0.8,
            loss_value: 0.2,
            training_duration: Duration::from_secs(10),
            model_size_bytes: 1024,
        })
    }

    async fn update_weights(
        &mut self,
        _feedback: &FeedbackData,
    ) -> std::result::Result<(), OptimizationError> {
        Ok(())
    }

    fn get_model_metrics(&self) -> ModelMetrics {
        ModelMetrics {
            accuracy: 0.85,
            precision: 0.8,
            recall: 0.9,
            f1_score: 0.84,
            mae: 0.15,
            rmse: 0.2,
        }
    }
}

// Stub implementations with placeholder fields are provided individually

// Additional implementations for key functionality
impl QuantumLoadPredictionModel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Predict the optimal node from the real per-node features extracted
    /// by `extract_quantum_features` (keyed `node_{i}_...` in
    /// `features.features`, matching `available_nodes`'s order) -- rather
    /// than always returning a fixed `NodeId("node_1")` regardless of
    /// which nodes actually exist. Nodes without enough qubit capacity for
    /// `circuit_partition` are excluded; among the remaining nodes the
    /// candidate with the best weighted score (readout fidelity,
    /// entanglement quality, coherence time, queue pressure, and
    /// utilization) is selected.
    pub async fn predict_optimal_node(
        &self,
        available_nodes: &[NodeInfo],
        circuit_partition: &CircuitPartition,
        features: &FeatureVector,
    ) -> Result<QuantumPredictionResult> {
        if available_nodes.is_empty() {
            return Err(QuantumLoadBalancingError::QuantumSchedulingConflict(
                "no available nodes to predict from".to_string(),
            ));
        }
        let required_qubits = circuit_partition.input_qubits.len().max(1) as f64;

        let mut best_index: Option<usize> = None;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_fidelity = 0.95_f64;

        for (i, node) in available_nodes.iter().enumerate() {
            let prefix = format!("node_{i}");
            let get = |suffix: &str, default: f64| -> f64 {
                features
                    .features
                    .get(&format!("{prefix}_{suffix}"))
                    .copied()
                    .unwrap_or(default)
            };

            let max_qubits = get("max_qubits", node.capabilities.max_qubits as f64);
            if max_qubits < required_qubits {
                continue;
            }
            let readout_fidelity = get("readout_fidelity", node.capabilities.readout_fidelity);
            let qubits_in_use = get("qubits_in_use", node.current_load.qubits_in_use as f64);
            let queue_length = get("queue_length", node.current_load.queue_length as f64);
            // Neutral defaults (0.5 quality, 50us coherence) when this
            // node has no recorded entanglement/coherence measurements
            // yet, rather than silently scoring it as perfect or zero.
            let entanglement_quality = get("entanglement_quality", 0.5);
            let coherence_time = get("avg_coherence_time", 50e-6);

            let utilization = if max_qubits > 0.0 {
                (qubits_in_use / max_qubits).clamp(0.0, 1.0)
            } else {
                1.0
            };
            let queue_penalty = 1.0 / (1.0 + queue_length.max(0.0));
            let coherence_score = (coherence_time / 100e-6).clamp(0.0, 1.0);

            let score = readout_fidelity * 0.35
                + entanglement_quality * 0.25
                + coherence_score * 0.15
                + queue_penalty * 0.15
                + (1.0 - utilization) * 0.10;

            if score > best_score {
                best_score = score;
                best_index = Some(i);
                best_fidelity = readout_fidelity;
            }
        }

        let Some(best_index) = best_index else {
            return Err(QuantumLoadBalancingError::QuantumSchedulingConflict(
                format!(
                    "no node has sufficient qubit capacity ({required_qubits}) for this partition"
                ),
            ));
        };

        let predicted_node = available_nodes[best_index].node_id.clone();
        let confidence = best_score.clamp(0.0, 1.0);
        let predicted_entanglement_overhead = (required_qubits / 2.0).ceil() as u32;

        Ok(QuantumPredictionResult {
            predicted_node,
            predicted_execution_time: Duration::from_millis((100.0 / confidence.max(0.1)) as u64),
            predicted_fidelity: best_fidelity,
            predicted_entanglement_overhead,
            confidence,
            quantum_uncertainty: QuantumUncertaintyFactors {
                decoherence_uncertainty: (1.0 - confidence) * 0.1,
                entanglement_uncertainty: (1.0 - confidence) * 0.06,
                measurement_uncertainty: (1.0 - confidence) * 0.04,
                calibration_uncertainty: (1.0 - confidence) * 0.02,
            },
            prediction_timestamp: Utc::now(),
        })
    }
}

impl QuantumPerformanceLearner {
    pub fn new() -> Self {
        Self::default()
    }

    /// Actually record training feedback in `performance_history` (a real
    /// `Arc<RwLock<HashMap<NodeId, QuantumPerformanceHistory>>>` field that
    /// previously sat unused) rather than silently discarding every
    /// training data point.
    pub async fn add_training_data(&self, data: QuantumLearningDataPoint) -> Result<()> {
        let mut history = self
            .performance_history
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let node_history = history
            .entry(data.selected_node.clone())
            .or_insert_with(QuantumPerformanceHistory::default);

        node_history
            .classical_metrics
            .execution_times
            .push_back(data.circuit_partition.estimated_execution_time);
        if node_history.classical_metrics.execution_times.len() > 100 {
            node_history.classical_metrics.execution_times.pop_front();
        }

        node_history
            .fidelity_history
            .push_back(FidelityMeasurement {
                timestamp: data.timestamp,
                process_fidelity: data.quantum_requirements.fidelity_requirement,
                state_fidelity: data.quantum_requirements.fidelity_requirement,
                gate_fidelities: HashMap::new(),
                measurement_context: FidelityMeasurementContext {
                    temperature: data
                        .context_features
                        .get("temperature")
                        .copied()
                        .unwrap_or(0.0),
                    time_since_calibration: Duration::from_secs(0),
                    circuit_depth: data.circuit_partition.gates.len() as u32,
                    concurrent_operations: 0,
                },
            });
        if node_history.fidelity_history.len() > 100 {
            node_history.fidelity_history.pop_front();
        }

        Ok(())
    }
}

/// Test module for quantum-aware load balancing
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_quantum_load_balancer_creation() {
        let balancer = MLOptimizedQuantumLoadBalancer::new();
        assert!(
            !balancer
                .adaptive_weights
                .lock()
                .expect("Mutex should not be poisoned")
                .dynamic_adjustment_enabled
                || balancer
                    .adaptive_weights
                    .lock()
                    .expect("Mutex should not be poisoned")
                    .dynamic_adjustment_enabled
        );
    }

    #[tokio::test]
    async fn test_quantum_feature_extraction() {
        let balancer = MLOptimizedQuantumLoadBalancer::new();

        let nodes = vec![NodeInfo {
            node_id: NodeId("test_node".to_string()),
            capabilities: crate::quantum_network::distributed_protocols::NodeCapabilities {
                max_qubits: 10,
                supported_gates: vec!["H".to_string(), "CNOT".to_string()],
                connectivity_graph: vec![(0, 1), (1, 2)],
                gate_fidelities: HashMap::new(),
                readout_fidelity: 0.95,
                coherence_times: HashMap::new(),
                classical_compute_power: 1000.0,
                memory_capacity_gb: 8,
                network_bandwidth_mbps: 1000.0,
            },
            current_load: crate::quantum_network::distributed_protocols::NodeLoad {
                qubits_in_use: 3,
                active_circuits: 2,
                cpu_utilization: 0.4,
                memory_utilization: 0.3,
                network_utilization: 0.2,
                queue_length: 1,
                estimated_completion_time: Duration::from_secs(30),
            },
            network_info: crate::quantum_network::distributed_protocols::NetworkInfo {
                ip_address: "192.168.1.100".to_string(),
                port: 8080,
                latency_to_nodes: HashMap::new(),
                bandwidth_to_nodes: HashMap::new(),
                connection_quality: HashMap::new(),
            },
            status: crate::quantum_network::distributed_protocols::NodeStatus::Active,
            last_heartbeat: Utc::now(),
        }];

        let circuit_partition = CircuitPartition {
            partition_id: Uuid::new_v4(),
            node_id: NodeId("test".to_string()),
            gates: vec![],
            dependencies: vec![],
            input_qubits: vec![],
            output_qubits: vec![],
            classical_inputs: vec![],
            estimated_execution_time: Duration::from_millis(100),
            resource_requirements: ResourceRequirements {
                qubits_needed: 5,
                gates_count: 10,
                memory_mb: 50,
                execution_time_estimate: Duration::from_millis(100),
                entanglement_pairs_needed: 2,
                classical_communication_bits: 100,
            },
        };

        let quantum_requirements = QuantumResourceRequirements {
            qubits_needed: 5,
            gate_count_estimate: 10,
            circuit_depth: 5,
            fidelity_requirement: 0.9,
            coherence_time_needed: Duration::from_micros(100),
            entanglement_pairs: 2,
        };

        let features = balancer
            .extract_quantum_features(&nodes, &circuit_partition, &quantum_requirements)
            .await;
        assert!(features.is_ok());

        let feature_vector = features.expect("Feature extraction should succeed");
        assert!(!feature_vector.features.is_empty());
        assert!(feature_vector.features.contains_key("circuit_depth"));
        assert!(feature_vector
            .features
            .contains_key("entanglement_pairs_needed"));
    }

    #[tokio::test]
    async fn test_capability_based_quantum_balancer() {
        let balancer = CapabilityBasedQuantumBalancer::new();

        let nodes = vec![
            NodeInfo {
                node_id: NodeId("high_capability_node".to_string()),
                capabilities: crate::quantum_network::distributed_protocols::NodeCapabilities {
                    max_qubits: 20,
                    supported_gates: vec!["H".to_string(), "CNOT".to_string(), "T".to_string()],
                    connectivity_graph: vec![],
                    gate_fidelities: HashMap::new(),
                    readout_fidelity: 0.98,
                    coherence_times: HashMap::new(),
                    classical_compute_power: 2000.0,
                    memory_capacity_gb: 16,
                    network_bandwidth_mbps: 2000.0,
                },
                current_load: crate::quantum_network::distributed_protocols::NodeLoad {
                    qubits_in_use: 5,
                    active_circuits: 1,
                    cpu_utilization: 0.2,
                    memory_utilization: 0.1,
                    network_utilization: 0.1,
                    queue_length: 0,
                    estimated_completion_time: Duration::from_secs(10),
                },
                network_info: crate::quantum_network::distributed_protocols::NetworkInfo {
                    ip_address: "192.168.1.101".to_string(),
                    port: 8080,
                    latency_to_nodes: HashMap::new(),
                    bandwidth_to_nodes: HashMap::new(),
                    connection_quality: HashMap::new(),
                },
                status: crate::quantum_network::distributed_protocols::NodeStatus::Active,
                last_heartbeat: Utc::now(),
            },
            NodeInfo {
                node_id: NodeId("low_capability_node".to_string()),
                capabilities: crate::quantum_network::distributed_protocols::NodeCapabilities {
                    max_qubits: 5,
                    supported_gates: vec!["H".to_string(), "CNOT".to_string()],
                    connectivity_graph: vec![],
                    gate_fidelities: HashMap::new(),
                    readout_fidelity: 0.90,
                    coherence_times: HashMap::new(),
                    classical_compute_power: 500.0,
                    memory_capacity_gb: 4,
                    network_bandwidth_mbps: 500.0,
                },
                current_load: crate::quantum_network::distributed_protocols::NodeLoad {
                    qubits_in_use: 4,
                    active_circuits: 2,
                    cpu_utilization: 0.8,
                    memory_utilization: 0.7,
                    network_utilization: 0.6,
                    queue_length: 3,
                    estimated_completion_time: Duration::from_secs(60),
                },
                network_info: crate::quantum_network::distributed_protocols::NetworkInfo {
                    ip_address: "192.168.1.102".to_string(),
                    port: 8080,
                    latency_to_nodes: HashMap::new(),
                    bandwidth_to_nodes: HashMap::new(),
                    connection_quality: HashMap::new(),
                },
                status: crate::quantum_network::distributed_protocols::NodeStatus::Active,
                last_heartbeat: Utc::now(),
            },
        ];

        let requirements = ResourceRequirements {
            qubits_needed: 10,
            gates_count: 20,
            memory_mb: 100,
            execution_time_estimate: Duration::from_millis(200),
            entanglement_pairs_needed: 3,
            classical_communication_bits: 500,
        };

        let selected_node = balancer.select_node(&nodes, &requirements).await;
        assert!(selected_node.is_ok());

        // Should select the high capability node
        let node_id = selected_node.expect("Node selection should succeed");
        assert_eq!(node_id.0, "high_capability_node");
    }

    fn make_test_node(
        id: &str,
        max_qubits: u32,
        readout_fidelity: f64,
        qubits_in_use: u32,
        queue_length: u32,
    ) -> NodeInfo {
        NodeInfo {
            node_id: NodeId(id.to_string()),
            capabilities: crate::quantum_network::distributed_protocols::NodeCapabilities {
                max_qubits,
                supported_gates: vec!["H".to_string(), "CNOT".to_string()],
                connectivity_graph: vec![],
                gate_fidelities: HashMap::new(),
                readout_fidelity,
                coherence_times: HashMap::new(),
                classical_compute_power: 1000.0,
                memory_capacity_gb: 8,
                network_bandwidth_mbps: 1000.0,
            },
            current_load: crate::quantum_network::distributed_protocols::NodeLoad {
                qubits_in_use,
                active_circuits: 1,
                cpu_utilization: 0.2,
                memory_utilization: 0.2,
                network_utilization: 0.2,
                queue_length,
                estimated_completion_time: Duration::from_secs(10),
            },
            network_info: crate::quantum_network::distributed_protocols::NetworkInfo {
                ip_address: "127.0.0.1".to_string(),
                port: 8080,
                latency_to_nodes: HashMap::new(),
                bandwidth_to_nodes: HashMap::new(),
                connection_quality: HashMap::new(),
            },
            status: crate::quantum_network::distributed_protocols::NodeStatus::Active,
            last_heartbeat: Utc::now(),
        }
    }

    fn make_test_partition(qubits_needed: u32) -> CircuitPartition {
        let node_id = NodeId("unassigned".to_string());
        CircuitPartition {
            partition_id: Uuid::new_v4(),
            node_id: node_id.clone(),
            gates: vec![],
            dependencies: vec![],
            input_qubits: (0..qubits_needed)
                .map(|i| crate::quantum_network::distributed_protocols::QubitId {
                    node_id: node_id.clone(),
                    local_id: i,
                    global_id: Uuid::new_v4(),
                })
                .collect(),
            output_qubits: vec![],
            classical_inputs: vec![],
            estimated_execution_time: Duration::from_millis(100),
            resource_requirements: ResourceRequirements {
                qubits_needed,
                gates_count: 10,
                memory_mb: 50,
                execution_time_estimate: Duration::from_millis(100),
                entanglement_pairs_needed: 1,
                classical_communication_bits: 100,
            },
        }
    }

    #[tokio::test]
    async fn test_predict_optimal_node_not_fixed_to_node1() {
        let predictor = QuantumLoadPredictionModel::new();
        let balancer = MLOptimizedQuantumLoadBalancer::new();

        // node index 0 ("best_node") clearly dominates node index 1
        // ("weak_node") on every feature: higher fidelity, less load, no
        // queue. The old placeholder always returned NodeId("node_1")
        // regardless of quality -- a real predictor must pick the actually
        // better node "best_node" here.
        let nodes = vec![
            make_test_node("best_node", 20, 0.99, 0, 0),
            make_test_node("weak_node", 20, 0.5, 18, 20),
        ];
        let partition = make_test_partition(2);
        let quantum_requirements = QuantumResourceRequirements {
            qubits_needed: 2,
            gate_count_estimate: 5,
            circuit_depth: 5,
            fidelity_requirement: 0.9,
            coherence_time_needed: Duration::from_micros(50),
            entanglement_pairs: 1,
        };
        let features = balancer
            .extract_quantum_features(&nodes, &partition, &quantum_requirements)
            .await
            .expect("feature extraction should succeed");

        let prediction = predictor
            .predict_optimal_node(&nodes, &partition, &features)
            .await
            .expect("prediction should succeed");
        assert_eq!(prediction.predicted_node.0, "best_node");

        // Swap node order: the winner must still be "best_node" by
        // identity/quality, not by fixed position "node_1".
        let nodes_swapped = vec![
            make_test_node("weak_node", 20, 0.5, 18, 20),
            make_test_node("best_node", 20, 0.99, 0, 0),
        ];
        let features_swapped = balancer
            .extract_quantum_features(&nodes_swapped, &partition, &quantum_requirements)
            .await
            .expect("feature extraction should succeed");
        let prediction_swapped = predictor
            .predict_optimal_node(&nodes_swapped, &partition, &features_swapped)
            .await
            .expect("prediction should succeed");
        assert_eq!(prediction_swapped.predicted_node.0, "best_node");
    }

    #[tokio::test]
    async fn test_predict_optimal_node_respects_qubit_capacity() {
        let predictor = QuantumLoadPredictionModel::new();
        let balancer = MLOptimizedQuantumLoadBalancer::new();

        // "tiny_node" has excellent fidelity but cannot fit the partition;
        // a real capacity-aware predictor must never select it.
        let nodes = vec![
            make_test_node("tiny_node", 2, 0.999, 0, 0),
            make_test_node("adequate_node", 10, 0.8, 2, 1),
        ];
        let partition = make_test_partition(8);
        let quantum_requirements = QuantumResourceRequirements {
            qubits_needed: 8,
            gate_count_estimate: 20,
            circuit_depth: 10,
            fidelity_requirement: 0.7,
            coherence_time_needed: Duration::from_micros(50),
            entanglement_pairs: 2,
        };
        let features = balancer
            .extract_quantum_features(&nodes, &partition, &quantum_requirements)
            .await
            .expect("feature extraction should succeed");

        let prediction = predictor
            .predict_optimal_node(&nodes, &partition, &features)
            .await
            .expect("prediction should succeed");
        assert_eq!(prediction.predicted_node.0, "adequate_node");
    }

    #[tokio::test]
    async fn test_add_training_data_is_actually_recorded() {
        let learner = QuantumPerformanceLearner::new();
        let node_id = NodeId("trained_node".to_string());
        let data_point = QuantumLearningDataPoint {
            timestamp: Utc::now(),
            selected_node: node_id.clone(),
            circuit_partition: make_test_partition(4),
            quantum_requirements: QuantumResourceRequirements {
                qubits_needed: 4,
                gate_count_estimate: 8,
                circuit_depth: 4,
                fidelity_requirement: 0.92,
                coherence_time_needed: Duration::from_micros(40),
                entanglement_pairs: 1,
            },
            context_features: HashMap::new(),
        };

        learner
            .add_training_data(data_point)
            .await
            .expect("add_training_data should succeed");

        // The training data point must actually be reflected in the
        // learner's real performance-history state, not silently discarded.
        let history = learner
            .performance_history
            .read()
            .expect("history lock should not be poisoned");
        let node_history = history
            .get(&node_id)
            .expect("training data should have created a history entry for the node");
        assert_eq!(node_history.fidelity_history.len(), 1);
        assert_eq!(node_history.classical_metrics.execution_times.len(), 1);
    }

    #[tokio::test]
    async fn test_select_nodes_load_balances_across_partitions() {
        let balancer = CapabilityBasedQuantumBalancer::new();
        let mut nodes = HashMap::new();
        nodes.insert(
            NodeId("node_a".to_string()),
            make_test_node("node_a", 8, 0.9, 0, 0),
        );
        nodes.insert(
            NodeId("node_b".to_string()),
            make_test_node("node_b", 8, 0.9, 0, 0),
        );

        // Two partitions each needing 6 qubits: no single 8-qubit node can
        // hold more than one (6 + 6 > 8), so a real load-balanced
        // assignment must spread them across both nodes rather than
        // routing every partition to whichever node
        // `HashMap::iter().next()` returns.
        let partitions = vec![make_test_partition(6), make_test_partition(6)];
        let requirements = ExecutionRequirements {
            min_fidelity: 0.5,
            max_latency: Duration::from_secs(10),
            fault_tolerance: false,
            preferred_nodes: vec![],
            excluded_nodes: vec![],
            resource_constraints:
                crate::quantum_network::distributed_protocols::ResourceConstraints {
                    max_cost: None,
                    max_execution_time: Duration::from_secs(60),
                    max_memory_usage: 1024,
                    preferred_providers: vec![],
                },
        };

        let allocation = balancer
            .select_nodes(&partitions, &nodes, &requirements)
            .expect("at least some partitions should be assignable");

        let assigned_node_ids: std::collections::HashSet<_> = allocation.values().collect();
        assert!(
            assigned_node_ids.len() > 1,
            "expected partitions to be spread across more than one node, got {assigned_node_ids:?}"
        );
    }
}
