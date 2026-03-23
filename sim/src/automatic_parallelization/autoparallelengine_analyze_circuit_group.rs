//! # AutoParallelEngine - analyze_circuit_group Methods
//!
//! This module contains method implementations for `AutoParallelEngine`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::distributed_simulator::{DistributedQuantumSimulator, DistributedSimulatorConfig};
use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use scirs2_core::Complex64;
use std::time::{Duration, Instant};

use super::types::{
    DependencyGraph, HardwareStrategy, MLPredictedStrategy, ParallelTask, ParallelizationAnalysis,
    ParallelizationStrategy, ResourceUtilization,
};

use super::autoparallelengine_type::AutoParallelEngine;

impl AutoParallelEngine {
    /// Analyze a circuit for parallelization opportunities
    pub fn analyze_circuit<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> QuantRS2Result<ParallelizationAnalysis> {
        let start_time = Instant::now();
        if self.config.enable_analysis_caching {
            let circuit_hash = Self::compute_circuit_hash(circuit);
            if let Some(cached_analysis) = self
                .analysis_cache
                .read()
                .expect("analysis cache read lock should not be poisoned")
                .get(&circuit_hash)
            {
                return Ok(cached_analysis.clone());
            }
        }
        let dependency_graph = self.build_dependency_graph(circuit)?;
        let tasks = match self.config.strategy {
            ParallelizationStrategy::DependencyAnalysis => {
                self.dependency_based_parallelization(&dependency_graph)?
            }
            ParallelizationStrategy::LayerBased => {
                self.layer_based_parallelization(&dependency_graph)?
            }
            ParallelizationStrategy::QubitPartitioning => {
                self.qubit_partitioning_parallelization(circuit, &dependency_graph)?
            }
            ParallelizationStrategy::Hybrid => {
                self.hybrid_parallelization(circuit, &dependency_graph)?
            }
            ParallelizationStrategy::MLGuided => {
                self.ml_guided_parallelization(circuit, &dependency_graph)?
            }
            ParallelizationStrategy::HardwareAware => {
                self.hardware_aware_parallelization(circuit, &dependency_graph)?
            }
        };
        let analysis = self.calculate_parallelization_metrics(circuit, &dependency_graph, tasks)?;
        if self.config.enable_analysis_caching {
            let circuit_hash = Self::compute_circuit_hash(circuit);
            self.analysis_cache
                .write()
                .expect("analysis cache write lock should not be poisoned")
                .insert(circuit_hash, analysis.clone());
        }
        self.update_performance_stats(start_time.elapsed(), &analysis);
        Ok(analysis)
    }
    /// Execute circuit with distributed parallelization
    pub fn execute_distributed<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        distributed_sim: &mut DistributedQuantumSimulator,
    ) -> QuantRS2Result<Vec<Complex64>> {
        let analysis = self.analyze_circuit(circuit)?;
        let distributed_tasks =
            self.distribute_tasks_across_nodes(&analysis.tasks, distributed_sim)?;
        let results = self.execute_distributed_tasks(&distributed_tasks, distributed_sim)?;
        let final_result = Self::aggregate_distributed_results(results)?;
        Ok(final_result)
    }
    /// ML-guided parallelization strategy
    pub(super) fn ml_guided_parallelization<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        graph: &DependencyGraph,
    ) -> QuantRS2Result<Vec<ParallelTask>> {
        let features = self.extract_ml_features(circuit, graph);
        let predicted_strategy = Self::predict_parallelization_strategy(&features);
        let task_groups = match predicted_strategy {
            MLPredictedStrategy::HighParallelism => self.aggressive_parallelization(graph)?,
            MLPredictedStrategy::BalancedParallelism => {
                self.hybrid_parallelization(circuit, graph)?
            }
            MLPredictedStrategy::ConservativeParallelism => {
                self.dependency_based_parallelization(graph)?
            }
            MLPredictedStrategy::LayerOptimized => self.layer_based_parallelization(graph)?,
        };
        let optimized_tasks = self.ml_optimize_tasks(task_groups, &features)?;
        Ok(optimized_tasks)
    }
    /// Hardware-aware parallelization strategy
    pub(super) fn hardware_aware_parallelization<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        graph: &DependencyGraph,
    ) -> QuantRS2Result<Vec<ParallelTask>> {
        let hw_char = Self::detect_hardware_characteristics();
        let tasks = match self.select_hardware_strategy(&hw_char, circuit, graph)? {
            HardwareStrategy::CacheOptimized => {
                self.cache_optimized_parallelization(graph, &hw_char)?
            }
            HardwareStrategy::SIMDOptimized => {
                self.simd_optimized_parallelization(graph, &hw_char)?
            }
            HardwareStrategy::NUMAAware => self.numa_aware_parallelization(graph, &hw_char)?,
            HardwareStrategy::GPUOffload => self.dependency_based_parallelization(graph)?,
            HardwareStrategy::Hybrid => self.hybrid_hardware_parallelization(graph, &hw_char)?,
        };
        let optimized_tasks = Self::optimize_hardware_affinity(tasks, &hw_char)?;
        Ok(optimized_tasks)
    }
    /// Calculate parallelization metrics
    pub(super) fn calculate_parallelization_metrics<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        graph: &DependencyGraph,
        tasks: Vec<ParallelTask>,
    ) -> QuantRS2Result<ParallelizationAnalysis> {
        let num_layers = graph.layers.len();
        let max_parallelism = graph
            .layers
            .iter()
            .map(std::vec::Vec::len)
            .max()
            .unwrap_or(1);
        let critical_path_length = graph.layers.len();
        let efficiency = if circuit.num_gates() > 0 {
            max_parallelism as f64 / circuit.num_gates() as f64
        } else {
            0.0
        };
        let resource_utilization = ResourceUtilization {
            cpu_utilization: vec![0.8; self.config.max_threads],
            memory_usage: vec![
                self.config.memory_budget / self.config.max_threads;
                self.config.max_threads
            ],
            load_balance_score: 0.85,
            communication_overhead: 0.1,
        };
        let recommendations = self.generate_optimization_recommendations(circuit, graph, &tasks);
        Ok(ParallelizationAnalysis {
            tasks,
            num_layers,
            efficiency,
            max_parallelism,
            critical_path_length,
            resource_utilization,
            recommendations,
        })
    }
    /// Update performance statistics
    pub(super) fn update_performance_stats(
        &self,
        execution_time: Duration,
        analysis: &ParallelizationAnalysis,
    ) {
        let mut stats = self
            .performance_stats
            .lock()
            .expect("performance stats mutex should not be poisoned");
        stats.circuits_processed += 1;
        stats.total_execution_time += execution_time;
        stats.average_efficiency = stats
            .average_efficiency
            .mul_add((stats.circuits_processed - 1) as f64, analysis.efficiency)
            / stats.circuits_processed as f64;
    }
}
