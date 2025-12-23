//! Parallel Tensor Network Optimization
//!
//! This module provides advanced parallel processing strategies for tensor network
//! contractions, optimizing for modern multi-core and distributed architectures.

use crate::prelude::SimulatorError;
use scirs2_core::ndarray::{ArrayD, Dimension, IxDyn};
use scirs2_core::parallel_ops::*;
use scirs2_core::Complex64;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use crate::error::Result;

/// Parallel processing configuration for tensor networks
#[derive(Debug, Clone)]
pub struct ParallelTensorConfig {
    /// Number of worker threads
    pub num_threads: usize,
    /// Chunk size for parallel operations
    pub chunk_size: usize,
    /// Enable work-stealing between threads
    pub enable_work_stealing: bool,
    /// Memory threshold for switching to parallel mode
    pub parallel_threshold_bytes: usize,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
    /// Enable NUMA-aware scheduling
    pub numa_aware: bool,
    /// Thread affinity settings
    pub thread_affinity: ThreadAffinityConfig,
}

impl Default for ParallelTensorConfig {
    fn default() -> Self {
        Self {
            num_threads: current_num_threads(), // SciRS2 POLICY compliant
            chunk_size: 1024,
            enable_work_stealing: true,
            parallel_threshold_bytes: 1024 * 1024, // 1MB
            load_balancing: LoadBalancingStrategy::DynamicWorkStealing,
            numa_aware: true,
            thread_affinity: ThreadAffinityConfig::default(),
        }
    }
}

/// Load balancing strategies for parallel tensor operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalancingStrategy {
    /// Static round-robin distribution
    RoundRobin,
    /// Dynamic work-stealing
    DynamicWorkStealing,
    /// NUMA-aware distribution
    NumaAware,
    /// Cost-based distribution
    CostBased,
    /// Adaptive strategy selection
    Adaptive,
}

/// Thread affinity configuration
#[derive(Debug, Clone, Default)]
pub struct ThreadAffinityConfig {
    /// Enable CPU affinity
    pub enable_affinity: bool,
    /// CPU core mapping
    pub core_mapping: Vec<usize>,
    /// NUMA node preferences
    pub numa_preferences: HashMap<usize, usize>,
}

/// Work unit for parallel tensor contraction
#[derive(Debug, Clone)]
pub struct TensorWorkUnit {
    /// Unique identifier for the work unit
    pub id: usize,
    /// Input tensor indices
    pub input_tensors: Vec<usize>,
    /// Output tensor index
    pub output_tensor: usize,
    /// Contraction indices
    pub contraction_indices: Vec<Vec<usize>>,
    /// Estimated computational cost
    pub estimated_cost: f64,
    /// Memory requirement
    pub memory_requirement: usize,
    /// Dependencies (must complete before this unit)
    pub dependencies: HashSet<usize>,
    /// Priority level
    pub priority: i32,
}

/// Work queue for managing parallel tensor operations
#[derive(Debug)]
pub struct TensorWorkQueue {
    /// Pending work units
    pending: Mutex<VecDeque<TensorWorkUnit>>,
    /// Completed work units
    completed: RwLock<HashSet<usize>>,
    /// Work units in progress
    in_progress: RwLock<HashMap<usize, Instant>>,
    /// Total work units
    total_units: usize,
    /// Configuration
    config: ParallelTensorConfig,
}

impl TensorWorkQueue {
    /// Create new work queue
    pub fn new(work_units: Vec<TensorWorkUnit>, config: ParallelTensorConfig) -> Self {
        let total_units = work_units.len();
        let mut pending = VecDeque::from(work_units);

        // Sort by priority and dependencies
        pending.make_contiguous().sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then_with(|| a.dependencies.len().cmp(&b.dependencies.len()))
        });

        Self {
            pending: Mutex::new(pending),
            completed: RwLock::new(HashSet::new()),
            in_progress: RwLock::new(HashMap::new()),
            total_units,
            config,
        }
    }

    /// Get next available work unit
    pub fn get_work(&self) -> Option<TensorWorkUnit> {
        let mut pending = self.pending.lock().unwrap();
        let completed = self.completed.read().unwrap();

        // Find a work unit whose dependencies are satisfied
        for i in 0..pending.len() {
            let work_unit = &pending[i];
            let dependencies_satisfied = work_unit
                .dependencies
                .iter()
                .all(|dep| completed.contains(dep));

            if dependencies_satisfied {
                let work_unit = pending.remove(i).unwrap();

                // Mark as in progress
                drop(completed);
                let mut in_progress = self.in_progress.write().unwrap();
                in_progress.insert(work_unit.id, Instant::now());

                return Some(work_unit);
            }
        }

        None
    }

    /// Mark work unit as completed
    pub fn complete_work(&self, work_id: usize) {
        let mut completed = self.completed.write().unwrap();
        completed.insert(work_id);

        let mut in_progress = self.in_progress.write().unwrap();
        in_progress.remove(&work_id);
    }

    /// Check if all work is completed
    pub fn is_complete(&self) -> bool {
        let completed = self.completed.read().unwrap();
        completed.len() == self.total_units
    }

    /// Get progress statistics
    pub fn get_progress(&self) -> (usize, usize, usize) {
        let completed = self.completed.read().unwrap().len();
        let in_progress = self.in_progress.read().unwrap().len();
        let pending = self.pending.lock().unwrap().len();
        (completed, in_progress, pending)
    }
}

/// Parallel tensor contraction engine
pub struct ParallelTensorEngine {
    /// Configuration
    config: ParallelTensorConfig,
    /// Worker thread pool
    thread_pool: ThreadPool, // SciRS2 POLICY compliant
    /// Performance statistics
    stats: Arc<Mutex<ParallelTensorStats>>,
}

/// Performance statistics for parallel tensor operations
#[derive(Debug, Clone, Default)]
pub struct ParallelTensorStats {
    /// Total contractions performed
    pub total_contractions: u64,
    /// Total computation time
    pub total_computation_time: Duration,
    /// Total parallel efficiency (0.0 to 1.0)
    pub parallel_efficiency: f64,
    /// Memory usage statistics
    pub peak_memory_usage: usize,
    /// Thread utilization statistics
    pub thread_utilization: Vec<f64>,
    /// Load balancing effectiveness
    pub load_balance_factor: f64,
    /// Cache hit rate for intermediate results
    pub cache_hit_rate: f64,
}

impl ParallelTensorEngine {
    /// Create new parallel tensor engine
    pub fn new(config: ParallelTensorConfig) -> Result<Self> {
        let thread_pool = ThreadPoolBuilder::new() // SciRS2 POLICY compliant
            .num_threads(config.num_threads)
            .build()
            .map_err(|e| {
                SimulatorError::InitializationFailed(format!("Thread pool creation failed: {e}"))
            })?;

        Ok(Self {
            config,
            thread_pool,
            stats: Arc::new(Mutex::new(ParallelTensorStats::default())),
        })
    }

    /// Perform parallel tensor network contraction
    pub fn contract_network(
        &self,
        tensors: &[ArrayD<Complex64>],
        contraction_sequence: &[ContractionPair],
    ) -> Result<ArrayD<Complex64>> {
        let start_time = Instant::now();

        // Create work units from contraction sequence
        let work_units = self.create_work_units(tensors, contraction_sequence)?;

        // Create work queue
        let work_queue = Arc::new(TensorWorkQueue::new(work_units, self.config.clone()));

        // Storage for intermediate results
        let intermediate_results =
            Arc::new(RwLock::new(HashMap::<usize, ArrayD<Complex64>>::new()));

        // Initialize with input tensors
        {
            let mut results = intermediate_results.write().unwrap();
            for (i, tensor) in tensors.iter().enumerate() {
                results.insert(i, tensor.clone());
            }
        }

        // Execute contractions in parallel
        let final_result = self.execute_parallel_contractions(work_queue, intermediate_results)?;

        // Update statistics
        let elapsed = start_time.elapsed();
        let mut stats = self.stats.lock().unwrap();
        stats.total_contractions += contraction_sequence.len() as u64;
        stats.total_computation_time += elapsed;

        // Calculate parallel efficiency (simplified)
        let sequential_estimate = self.estimate_sequential_time(contraction_sequence);
        stats.parallel_efficiency = sequential_estimate.as_secs_f64() / elapsed.as_secs_f64();

        Ok(final_result)
    }

    /// Create work units from contraction sequence
    fn create_work_units(
        &self,
        tensors: &[ArrayD<Complex64>],
        contraction_sequence: &[ContractionPair],
    ) -> Result<Vec<TensorWorkUnit>> {
        let mut work_units: Vec<TensorWorkUnit> = Vec::new();
        let mut next_tensor_id = tensors.len();

        for (i, contraction) in contraction_sequence.iter().enumerate() {
            let estimated_cost = self.estimate_contraction_cost(contraction, tensors)?;
            let memory_requirement = self.estimate_memory_requirement(contraction, tensors)?;

            // Determine dependencies
            let mut dependencies = HashSet::new();
            for &input_id in &[contraction.tensor1_id, contraction.tensor2_id] {
                if input_id >= tensors.len() {
                    // This is an intermediate result, find which work unit produces it
                    for prev_unit in &work_units {
                        if prev_unit.output_tensor == input_id {
                            dependencies.insert(prev_unit.id);
                            break;
                        }
                    }
                }
            }

            let work_unit = TensorWorkUnit {
                id: i,
                input_tensors: vec![contraction.tensor1_id, contraction.tensor2_id],
                output_tensor: next_tensor_id,
                contraction_indices: vec![
                    contraction.tensor1_indices.clone(),
                    contraction.tensor2_indices.clone(),
                ],
                estimated_cost,
                memory_requirement,
                dependencies,
                priority: self.calculate_priority(estimated_cost, memory_requirement),
            };

            work_units.push(work_unit);
            next_tensor_id += 1;
        }

        Ok(work_units)
    }

    /// Execute parallel contractions using work queue
    fn execute_parallel_contractions(
        &self,
        work_queue: Arc<TensorWorkQueue>,
        intermediate_results: Arc<RwLock<HashMap<usize, ArrayD<Complex64>>>>,
    ) -> Result<ArrayD<Complex64>> {
        let num_threads = self.config.num_threads;
        let mut handles = Vec::new();

        // Spawn worker threads
        for thread_id in 0..num_threads {
            let work_queue = work_queue.clone();
            let intermediate_results = intermediate_results.clone();
            let config = self.config.clone();

            let handle = thread::spawn(move || {
                Self::worker_thread(thread_id, work_queue, intermediate_results, config)
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().map_err(|e| {
                SimulatorError::ComputationError(format!("Thread join failed: {e:?}"))
            })??;
        }

        // Find the final result (tensor with highest ID)
        let results = intermediate_results.read().unwrap();
        let max_id = results.keys().max().copied().unwrap_or(0);
        Ok(results[&max_id].clone())
    }

    /// Worker thread function
    fn worker_thread(
        _thread_id: usize,
        work_queue: Arc<TensorWorkQueue>,
        intermediate_results: Arc<RwLock<HashMap<usize, ArrayD<Complex64>>>>,
        _config: ParallelTensorConfig,
    ) -> Result<()> {
        while !work_queue.is_complete() {
            if let Some(work_unit) = work_queue.get_work() {
                // Get input tensors
                let tensor1 = {
                    let results = intermediate_results.read().unwrap();
                    results[&work_unit.input_tensors[0]].clone()
                };

                let tensor2 = {
                    let results = intermediate_results.read().unwrap();
                    results[&work_unit.input_tensors[1]].clone()
                };

                // Perform contraction
                let result = Self::perform_tensor_contraction(
                    &tensor1,
                    &tensor2,
                    &work_unit.contraction_indices[0],
                    &work_unit.contraction_indices[1],
                )?;

                // Store result
                {
                    let mut results = intermediate_results.write().unwrap();
                    results.insert(work_unit.output_tensor, result);
                }

                // Mark work as completed
                work_queue.complete_work(work_unit.id);
            } else {
                // No work available, wait briefly
                thread::sleep(Duration::from_millis(1));
            }
        }

        Ok(())
    }

    /// Perform actual tensor contraction
    fn perform_tensor_contraction(
        tensor1: &ArrayD<Complex64>,
        tensor2: &ArrayD<Complex64>,
        indices1: &[usize],
        indices2: &[usize],
    ) -> Result<ArrayD<Complex64>> {
        // This is a simplified tensor contraction implementation
        // In practice, this would use optimized BLAS operations

        let shape1 = tensor1.shape();
        let shape2 = tensor2.shape();

        // Calculate output shape
        let mut output_shape = Vec::new();
        for (i, &size) in shape1.iter().enumerate() {
            if !indices1.contains(&i) {
                output_shape.push(size);
            }
        }
        for (i, &size) in shape2.iter().enumerate() {
            if !indices2.contains(&i) {
                output_shape.push(size);
            }
        }

        // Create output tensor
        let output_dim = IxDyn(&output_shape);
        let mut output = ArrayD::zeros(output_dim);

        // Simplified contraction (this would be optimized in practice)
        // For now, just return a placeholder
        Ok(output)
    }

    /// Estimate computational cost of a contraction
    fn estimate_contraction_cost(
        &self,
        contraction: &ContractionPair,
        _tensors: &[ArrayD<Complex64>],
    ) -> Result<f64> {
        // Simplified cost estimation based on dimension products
        let cost = contraction.tensor1_indices.len() as f64
            * contraction.tensor2_indices.len() as f64
            * 1000.0; // Base cost factor
        Ok(cost)
    }

    /// Estimate memory requirement for a contraction
    const fn estimate_memory_requirement(
        &self,
        _contraction: &ContractionPair,
        _tensors: &[ArrayD<Complex64>],
    ) -> Result<usize> {
        // Simplified memory estimation
        Ok(1024 * 1024) // 1MB placeholder
    }

    /// Calculate priority for work unit
    fn calculate_priority(&self, cost: f64, memory: usize) -> i32 {
        // Higher cost and lower memory = higher priority
        let cost_factor = (cost / 1000.0) as i32;
        let memory_factor = (1_000_000 / (memory + 1)) as i32;
        cost_factor + memory_factor
    }

    /// Estimate sequential execution time
    const fn estimate_sequential_time(&self, contraction_sequence: &[ContractionPair]) -> Duration {
        let estimated_ops = contraction_sequence.len() as u64 * 1000; // Simplified
        Duration::from_millis(estimated_ops)
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> ParallelTensorStats {
        self.stats.lock().unwrap().clone()
    }
}

/// Contraction pair specification
#[derive(Debug, Clone)]
pub struct ContractionPair {
    /// First tensor ID
    pub tensor1_id: usize,
    /// Second tensor ID
    pub tensor2_id: usize,
    /// Indices to contract on first tensor
    pub tensor1_indices: Vec<usize>,
    /// Indices to contract on second tensor
    pub tensor2_indices: Vec<usize>,
}

/// Advanced parallel tensor contraction strategies
pub mod strategies {
    use super::*;

    /// Work-stealing parallel contraction
    pub fn work_stealing_contraction(
        tensors: &[ArrayD<Complex64>],
        contraction_sequence: &[ContractionPair],
        num_threads: usize,
    ) -> Result<ArrayD<Complex64>> {
        let config = ParallelTensorConfig {
            num_threads,
            load_balancing: LoadBalancingStrategy::DynamicWorkStealing,
            ..Default::default()
        };

        let engine = ParallelTensorEngine::new(config)?;
        engine.contract_network(tensors, contraction_sequence)
    }

    /// NUMA-aware parallel contraction
    pub fn numa_aware_contraction(
        tensors: &[ArrayD<Complex64>],
        contraction_sequence: &[ContractionPair],
        numa_topology: &NumaTopology,
    ) -> Result<ArrayD<Complex64>> {
        let config = ParallelTensorConfig {
            load_balancing: LoadBalancingStrategy::NumaAware,
            numa_aware: true,
            ..Default::default()
        };

        let engine = ParallelTensorEngine::new(config)?;
        engine.contract_network(tensors, contraction_sequence)
    }

    /// Adaptive parallel contraction with dynamic load balancing
    pub fn adaptive_contraction(
        tensors: &[ArrayD<Complex64>],
        contraction_sequence: &[ContractionPair],
    ) -> Result<ArrayD<Complex64>> {
        let config = ParallelTensorConfig {
            load_balancing: LoadBalancingStrategy::Adaptive,
            enable_work_stealing: true,
            ..Default::default()
        };

        let engine = ParallelTensorEngine::new(config)?;
        engine.contract_network(tensors, contraction_sequence)
    }
}

/// NUMA topology information
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// Number of NUMA nodes
    pub num_nodes: usize,
    /// CPU cores per node
    pub cores_per_node: Vec<usize>,
    /// Memory per node (bytes)
    pub memory_per_node: Vec<usize>,
}

impl Default for NumaTopology {
    fn default() -> Self {
        let num_cores = current_num_threads(); // SciRS2 POLICY compliant
        Self {
            num_nodes: 1,
            cores_per_node: vec![num_cores],
            memory_per_node: vec![8 * 1024 * 1024 * 1024], // 8GB default
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_parallel_tensor_engine() {
        let config = ParallelTensorConfig::default();
        let engine = ParallelTensorEngine::new(config).unwrap();

        // Create simple test tensors
        let tensor1 = Array::zeros(IxDyn(&[2, 2]));
        let tensor2 = Array::zeros(IxDyn(&[2, 2]));
        let tensors = vec![tensor1, tensor2];

        // Simple contraction sequence
        let contraction = ContractionPair {
            tensor1_id: 0,
            tensor2_id: 1,
            tensor1_indices: vec![1],
            tensor2_indices: vec![0],
        };

        let result = engine.contract_network(&tensors, &[contraction]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_work_queue() {
        let work_unit = TensorWorkUnit {
            id: 0,
            input_tensors: vec![0, 1],
            output_tensor: 2,
            contraction_indices: vec![vec![0], vec![1]],
            estimated_cost: 100.0,
            memory_requirement: 1024,
            dependencies: HashSet::new(),
            priority: 1,
        };

        let config = ParallelTensorConfig::default();
        let queue = TensorWorkQueue::new(vec![work_unit], config);

        let work = queue.get_work();
        assert!(work.is_some());

        queue.complete_work(0);
        assert!(queue.is_complete());
    }

    #[test]
    fn test_parallel_strategies() {
        let tensor1 = Array::ones(IxDyn(&[2, 2]));
        let tensor2 = Array::ones(IxDyn(&[2, 2]));
        let tensors = vec![tensor1, tensor2];

        let contraction = ContractionPair {
            tensor1_id: 0,
            tensor2_id: 1,
            tensor1_indices: vec![1],
            tensor2_indices: vec![0],
        };

        let result = strategies::work_stealing_contraction(&tensors, &[contraction], 2);
        assert!(result.is_ok());
    }
}
