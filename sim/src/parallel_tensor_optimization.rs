//! Parallel Tensor Network Optimization
//!
//! This module provides advanced parallel processing strategies for tensor network
//! contractions, optimizing for modern multi-core and distributed architectures.

use crate::prelude::SimulatorError;
use scirs2_core::ndarray::{ArrayD, Dimension, IxDyn};
use scirs2_core::parallel_ops::{
    current_num_threads, IndexedParallelIterator, ParallelIterator, ThreadPool, ThreadPoolBuilder,
};
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
    #[must_use]
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
        // Use expect() for lock poisoning - if locks are poisoned, we have a bigger problem
        let mut pending = self
            .pending
            .lock()
            .expect("pending lock should not be poisoned");
        let completed = self
            .completed
            .read()
            .expect("completed lock should not be poisoned");

        // Find a work unit whose dependencies are satisfied
        for i in 0..pending.len() {
            let work_unit = &pending[i];
            let dependencies_satisfied = work_unit
                .dependencies
                .iter()
                .all(|dep| completed.contains(dep));

            if dependencies_satisfied {
                // Safety: i is always within bounds (from for loop condition)
                let work_unit = pending
                    .remove(i)
                    .expect("index i is guaranteed to be within bounds");

                // Mark as in progress
                drop(completed);
                let mut in_progress = self
                    .in_progress
                    .write()
                    .expect("in_progress lock should not be poisoned");
                in_progress.insert(work_unit.id, Instant::now());

                return Some(work_unit);
            }
        }

        None
    }

    /// Mark work unit as completed
    pub fn complete_work(&self, work_id: usize) {
        let mut completed = self
            .completed
            .write()
            .expect("completed lock should not be poisoned");
        completed.insert(work_id);

        let mut in_progress = self
            .in_progress
            .write()
            .expect("in_progress lock should not be poisoned");
        in_progress.remove(&work_id);
    }

    /// Check if all work is completed
    pub fn is_complete(&self) -> bool {
        let completed = self
            .completed
            .read()
            .expect("completed lock should not be poisoned");
        completed.len() == self.total_units
    }

    /// Get progress statistics
    pub fn get_progress(&self) -> (usize, usize, usize) {
        let completed = self
            .completed
            .read()
            .expect("completed lock should not be poisoned")
            .len();
        let in_progress = self
            .in_progress
            .read()
            .expect("in_progress lock should not be poisoned")
            .len();
        let pending = self
            .pending
            .lock()
            .expect("pending lock should not be poisoned")
            .len();
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
            let mut results = intermediate_results
                .write()
                .expect("intermediate_results lock should not be poisoned");
            for (i, tensor) in tensors.iter().enumerate() {
                results.insert(i, tensor.clone());
            }
        }

        // Execute contractions in parallel
        let final_result = self.execute_parallel_contractions(work_queue, intermediate_results)?;

        // Update statistics
        let elapsed = start_time.elapsed();
        let mut stats = self
            .stats
            .lock()
            .expect("stats lock should not be poisoned");
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
        let results = intermediate_results
            .read()
            .expect("intermediate_results lock should not be poisoned");
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
                    let results = intermediate_results
                        .read()
                        .expect("intermediate_results lock should not be poisoned");
                    results[&work_unit.input_tensors[0]].clone()
                };

                let tensor2 = {
                    let results = intermediate_results
                        .read()
                        .expect("intermediate_results lock should not be poisoned");
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
                    let mut results = intermediate_results
                        .write()
                        .expect("intermediate_results lock should not be poisoned");
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

    /// Perform an einsum-style pairwise tensor contraction.
    ///
    /// Contracts axes `indices1` of `tensor1` against axes `indices2` of
    /// `tensor2` (paired positionally, so `indices1[k]` is summed with
    /// `indices2[k]` and their dimensions must match). The output carries the
    /// free (uncontracted) axes of `tensor1` followed by those of `tensor2`:
    ///
    /// `out[f1, f2] = Σ_c tensor1[f1, c] * tensor2[f2, c]`
    ///
    /// where `c` ranges over all combinations of the contracted axes.
    fn perform_tensor_contraction(
        tensor1: &ArrayD<Complex64>,
        tensor2: &ArrayD<Complex64>,
        indices1: &[usize],
        indices2: &[usize],
    ) -> Result<ArrayD<Complex64>> {
        let shape1 = tensor1.shape().to_vec();
        let shape2 = tensor2.shape().to_vec();

        if indices1.len() != indices2.len() {
            return Err(SimulatorError::InvalidInput(format!(
                "Contraction requires equal numbers of contracted axes, got {} and {}",
                indices1.len(),
                indices2.len()
            )));
        }
        for (&a, &b) in indices1.iter().zip(indices2.iter()) {
            if a >= shape1.len() || b >= shape2.len() {
                return Err(SimulatorError::InvalidInput(format!(
                    "Contracted axis out of range: tensor1 axis {a} (rank {}), tensor2 axis {b} (rank {})",
                    shape1.len(),
                    shape2.len()
                )));
            }
            if shape1[a] != shape2[b] {
                return Err(SimulatorError::InvalidInput(format!(
                    "Contracted dimension mismatch: tensor1 axis {a} has size {}, tensor2 axis {b} has size {}",
                    shape1[a], shape2[b]
                )));
            }
        }

        // Partition axes into free (kept) and contracted (summed).
        let free1: Vec<usize> = (0..shape1.len())
            .filter(|i| !indices1.contains(i))
            .collect();
        let free2: Vec<usize> = (0..shape2.len())
            .filter(|i| !indices2.contains(i))
            .collect();

        let free1_dims: Vec<usize> = free1.iter().map(|&i| shape1[i]).collect();
        let free2_dims: Vec<usize> = free2.iter().map(|&i| shape2[i]).collect();
        let contracted_dims: Vec<usize> = indices1.iter().map(|&i| shape1[i]).collect();

        let output_shape: Vec<usize> = free1_dims
            .iter()
            .copied()
            .chain(free2_dims.iter().copied())
            .collect();
        let mut output = ArrayD::zeros(IxDyn(&output_shape));

        let free1_size: usize = free1_dims.iter().product::<usize>().max(1);
        let free2_size: usize = free2_dims.iter().product::<usize>().max(1);
        let contracted_size: usize = contracted_dims.iter().product::<usize>().max(1);

        // Reusable multi-index scratch buffers.
        let mut idx1 = vec![0usize; shape1.len()];
        let mut idx2 = vec![0usize; shape2.len()];
        let mut out_idx = vec![0usize; output_shape.len()];

        for f1 in 0..free1_size {
            let f1_idx = unravel_index(f1, &free1_dims);
            for (pos, &axis) in free1.iter().enumerate() {
                idx1[axis] = f1_idx[pos];
                out_idx[pos] = f1_idx[pos];
            }
            for f2 in 0..free2_size {
                let f2_idx = unravel_index(f2, &free2_dims);
                for (pos, &axis) in free2.iter().enumerate() {
                    idx2[axis] = f2_idx[pos];
                    out_idx[free1.len() + pos] = f2_idx[pos];
                }

                let mut acc = Complex64::new(0.0, 0.0);
                for c in 0..contracted_size {
                    let c_idx = unravel_index(c, &contracted_dims);
                    for (pos, (&a, &b)) in indices1.iter().zip(indices2.iter()).enumerate() {
                        idx1[a] = c_idx[pos];
                        idx2[b] = c_idx[pos];
                    }
                    acc += tensor1[IxDyn(&idx1)] * tensor2[IxDyn(&idx2)];
                }
                output[IxDyn(&out_idx)] = acc;
            }
        }

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
    #[must_use]
    pub fn get_stats(&self) -> ParallelTensorStats {
        self.stats
            .lock()
            .expect("stats lock should not be poisoned")
            .clone()
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
    use super::{
        ArrayD, Complex64, ContractionPair, LoadBalancingStrategy, NumaTopology,
        ParallelTensorConfig, ParallelTensorEngine, Result,
    };

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

/// Convert a row-major linear index into its multi-index for the given `dims`.
///
/// An empty `dims` slice yields an empty multi-index (the single element of a
/// zero-dimensional / scalar axis set).
fn unravel_index(mut linear: usize, dims: &[usize]) -> Vec<usize> {
    let mut multi = vec![0usize; dims.len()];
    for i in (0..dims.len()).rev() {
        let d = dims[i];
        multi[i] = linear % d;
        linear /= d;
    }
    multi
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_parallel_tensor_engine() {
        let config = ParallelTensorConfig::default();
        let engine =
            ParallelTensorEngine::new(config).expect("should create parallel tensor engine");

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

    /// Regression: `perform_tensor_contraction` previously returned an all-zero
    /// placeholder. It must now compute the real contraction; a matrix product
    /// is the canonical single-index contraction.
    #[test]
    fn test_perform_tensor_contraction_matrix_product() {
        // A (2x3) . B (3x2) via contracting A axis 1 with B axis 0.
        let a = Array::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(6.0, 0.0),
            ],
        )
        .expect("A shape");
        let b = Array::from_shape_vec(
            IxDyn(&[3, 2]),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        )
        .expect("B shape");

        let out = ParallelTensorEngine::perform_tensor_contraction(&a, &b, &[1], &[0])
            .expect("contraction should succeed");
        assert_eq!(out.shape(), &[2, 2]);
        // Expected A.B = [[4,5],[10,11]].
        assert!((out[IxDyn(&[0, 0])] - Complex64::new(4.0, 0.0)).norm() < 1e-12);
        assert!((out[IxDyn(&[0, 1])] - Complex64::new(5.0, 0.0)).norm() < 1e-12);
        assert!((out[IxDyn(&[1, 0])] - Complex64::new(10.0, 0.0)).norm() < 1e-12);
        assert!((out[IxDyn(&[1, 1])] - Complex64::new(11.0, 0.0)).norm() < 1e-12);
    }

    /// A contraction with no shared axes is an outer product; verify that too.
    #[test]
    fn test_perform_tensor_contraction_outer_product() {
        let a = Array::from_shape_vec(
            IxDyn(&[2]),
            vec![Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)],
        )
        .expect("A shape");
        let b = Array::from_shape_vec(
            IxDyn(&[2]),
            vec![Complex64::new(5.0, 0.0), Complex64::new(7.0, 0.0)],
        )
        .expect("B shape");

        let out = ParallelTensorEngine::perform_tensor_contraction(&a, &b, &[], &[])
            .expect("outer product should succeed");
        assert_eq!(out.shape(), &[2, 2]);
        assert!((out[IxDyn(&[0, 0])] - Complex64::new(10.0, 0.0)).norm() < 1e-12);
        assert!((out[IxDyn(&[0, 1])] - Complex64::new(14.0, 0.0)).norm() < 1e-12);
        assert!((out[IxDyn(&[1, 0])] - Complex64::new(15.0, 0.0)).norm() < 1e-12);
        assert!((out[IxDyn(&[1, 1])] - Complex64::new(21.0, 0.0)).norm() < 1e-12);
    }
}
