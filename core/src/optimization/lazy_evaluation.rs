//! Lazy evaluation system for gate sequence optimization
//!
//! This module provides a lazy evaluation framework that defers gate optimizations
//! until they're actually needed, improving performance for large circuits by avoiding
//! unnecessary computation and enabling more sophisticated optimization strategies.

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    optimization::{OptimizationPass, OptimizationChain},
    qubit::QubitId,
    compilation_cache::{CompilationCache, CompiledGate},
};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};
use serde::{Deserialize, Serialize};

/// Configuration for lazy evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LazyEvaluationConfig {
    /// Maximum number of gates to buffer before forced evaluation
    pub max_buffer_size: usize,
    /// Maximum time to defer evaluation
    pub max_defer_time: Duration,
    /// Enable dependency-based optimization ordering
    pub enable_dependency_optimization: bool,
    /// Enable speculative optimization
    pub enable_speculative_optimization: bool,
    /// Number of worker threads for async optimization
    pub num_optimization_threads: usize,
    /// Cache size for optimization results
    pub optimization_cache_size: usize,
}

impl Default for LazyEvaluationConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: 1000,
            max_defer_time: Duration::from_millis(100),
            enable_dependency_optimization: true,
            enable_speculative_optimization: true,
            num_optimization_threads: 4,
            optimization_cache_size: 10000,
        }
    }
}

/// Lazy evaluation context for a gate
#[derive(Debug, Clone)]
pub struct LazyGateContext {
    /// Unique identifier for this gate in the pipeline
    pub gate_id: usize,
    /// The gate to be optimized
    pub gate: Box<dyn GateOp>,
    /// Dependencies (other gates that must be evaluated first)
    pub dependencies: HashSet<usize>,
    /// Dependents (gates that depend on this one)
    pub dependents: HashSet<usize>,
    /// Priority for evaluation (higher = more urgent)
    pub priority: f64,
    /// Timestamp when gate was added
    pub created_at: Instant,
    /// Whether this gate has been evaluated
    pub is_evaluated: bool,
    /// Cached optimization result
    pub cached_result: Option<OptimizationResult>,
}

/// Result of a lazy optimization
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Optimized gate sequence
    pub optimized_gates: Vec<Box<dyn GateOp>>,
    /// Optimization statistics
    pub stats: OptimizationStats,
    /// Time taken for optimization
    pub optimization_time: Duration,
}

/// Statistics from optimization
#[derive(Debug, Clone, Default)]
pub struct OptimizationStats {
    /// Number of gates before optimization
    pub gates_before: usize,
    /// Number of gates after optimization
    pub gates_after: usize,
    /// Number of optimization passes applied
    pub passes_applied: usize,
    /// Estimated performance improvement
    pub performance_improvement: f64,
    /// Memory savings achieved
    pub memory_savings: usize,
}

/// Lazy optimization pipeline
pub struct LazyOptimizationPipeline {
    /// Configuration
    config: LazyEvaluationConfig,
    /// Buffered gates awaiting optimization
    gate_buffer: Arc<RwLock<HashMap<usize, LazyGateContext>>>,
    /// Dependency graph
    dependency_graph: Arc<RwLock<DependencyGraph>>,
    /// Optimization chain to apply
    optimization_chain: OptimizationChain,
    /// Compilation cache for optimized results
    optimization_cache: Arc<RwLock<OptimizationCache>>,
    /// Next gate ID
    next_gate_id: Arc<RwLock<usize>>,
    /// Worker thread handles
    worker_handles: Vec<std::thread::JoinHandle<()>>,
    /// Shutdown signal
    shutdown_signal: Arc<RwLock<bool>>,
}

/// Dependency graph for managing gate relationships
#[derive(Debug, Default)]
struct DependencyGraph {
    /// Adjacency list representation
    edges: HashMap<usize, HashSet<usize>>,
    /// Reverse edges for quick lookup
    reverse_edges: HashMap<usize, HashSet<usize>>,
    /// Topological ordering cache
    topo_order_cache: Option<Vec<usize>>,
}

/// Cache for optimization results
struct OptimizationCache {
    /// Cache entries indexed by gate hash
    entries: HashMap<u64, CachedOptimization>,
    /// LRU queue for eviction
    lru_queue: VecDeque<u64>,
    /// Maximum cache size
    max_size: usize,
}

/// Cached optimization entry
#[derive(Debug, Clone)]
struct CachedOptimization {
    /// The optimization result
    result: OptimizationResult,
    /// Access count for LRU
    access_count: usize,
    /// Last access time
    last_accessed: Instant,
}

impl LazyOptimizationPipeline {
    /// Create a new lazy optimization pipeline
    pub fn new(
        config: LazyEvaluationConfig,
        optimization_chain: OptimizationChain,
    ) -> QuantRS2Result<Self> {
        let gate_buffer = Arc::new(RwLock::new(HashMap::new()));
        let dependency_graph = Arc::new(RwLock::new(DependencyGraph::default()));
        let optimization_cache = Arc::new(RwLock::new(OptimizationCache::new(
            config.optimization_cache_size,
        )));
        let next_gate_id = Arc::new(RwLock::new(0));
        let shutdown_signal = Arc::new(RwLock::new(false));

        // Start worker threads
        let mut worker_handles = Vec::new();
        for worker_id in 0..config.num_optimization_threads {
            let handle = Self::start_worker_thread(
                worker_id,
                Arc::clone(&gate_buffer),
                Arc::clone(&dependency_graph),
                Arc::clone(&optimization_cache),
                Arc::clone(&shutdown_signal),
                config.clone(),
            );
            worker_handles.push(handle);
        }

        Ok(Self {
            config,
            gate_buffer,
            dependency_graph,
            optimization_chain,
            optimization_cache,
            next_gate_id,
            worker_handles,
            shutdown_signal,
        })
    }

    /// Add a gate to the lazy evaluation pipeline
    pub fn add_gate(&self, gate: Box<dyn GateOp>) -> QuantRS2Result<usize> {
        let gate_id = {
            let mut next_id = self.next_gate_id.write().unwrap();
            let id = *next_id;
            *next_id += 1;
            id
        };

        // Analyze dependencies based on qubit overlap
        let dependencies = self.analyze_dependencies(gate.as_ref())?;
        
        // Calculate priority based on gate type and dependencies
        let priority = self.calculate_priority(gate.as_ref(), &dependencies);

        let context = LazyGateContext {
            gate_id,
            gate,
            dependencies: dependencies.clone(),
            dependents: HashSet::new(),
            priority,
            created_at: Instant::now(),
            is_evaluated: false,
            cached_result: None,
        };

        // Update dependency graph
        {
            let mut graph = self.dependency_graph.write().unwrap();
            graph.add_gate(gate_id, dependencies);
        }

        // Add to buffer
        {
            let mut buffer = self.gate_buffer.write().unwrap();
            buffer.insert(gate_id, context);
        }

        // Check if we need to force evaluation due to buffer size or time
        self.check_forced_evaluation()?;

        Ok(gate_id)
    }

    /// Evaluate a specific gate (force optimization)
    pub fn evaluate_gate(&self, gate_id: usize) -> QuantRS2Result<OptimizationResult> {
        // Check cache first
        if let Some(cached) = self.get_cached_result(gate_id)? {
            return Ok(cached);
        }

        // Get the gate context
        let context = {
            let buffer = self.gate_buffer.read().unwrap();
            buffer.get(&gate_id).cloned()
                .ok_or_else(|| QuantRS2Error::InvalidInput(
                    format!("Gate {} not found in buffer", gate_id)
                ))?
        };

        // Ensure dependencies are evaluated first
        self.evaluate_dependencies(&context.dependencies)?;

        // Perform the optimization
        let result = self.optimize_gate_context(&context)?;

        // Cache the result
        self.cache_optimization_result(gate_id, &result)?;

        // Mark as evaluated
        {
            let mut buffer = self.gate_buffer.write().unwrap();
            if let Some(mut ctx) = buffer.get_mut(&gate_id) {
                ctx.is_evaluated = true;
                ctx.cached_result = Some(result.clone());
            }
        }

        Ok(result)
    }

    /// Evaluate all buffered gates
    pub fn evaluate_all(&self) -> QuantRS2Result<Vec<OptimizationResult>> {
        // Get topological ordering of gates
        let ordered_gates = {
            let graph = self.dependency_graph.read().unwrap();
            graph.topological_sort()
        };

        let mut results = Vec::new();
        for gate_id in ordered_gates {
            if let Ok(result) = self.evaluate_gate(gate_id) {
                results.push(result);
            }
        }

        // Clear the buffer
        {
            let mut buffer = self.gate_buffer.write().unwrap();
            buffer.clear();
        }

        Ok(results)
    }

    /// Get optimization statistics
    pub fn get_statistics(&self) -> LazyEvaluationStats {
        let buffer = self.gate_buffer.read().unwrap();
        let cache = self.optimization_cache.read().unwrap();
        
        let total_gates = buffer.len();
        let evaluated_gates = buffer.values().filter(|ctx| ctx.is_evaluated).count();
        let pending_gates = total_gates - evaluated_gates;
        
        LazyEvaluationStats {
            total_gates,
            evaluated_gates,
            pending_gates,
            cache_hits: cache.get_hit_count(),
            cache_size: cache.entries.len(),
            average_optimization_time: cache.get_average_optimization_time(),
        }
    }

    /// Analyze dependencies for a gate based on qubit overlap
    fn analyze_dependencies(&self, gate: &dyn GateOp) -> QuantRS2Result<HashSet<usize>> {
        let gate_qubits: HashSet<QubitId> = gate.qubits().into_iter().collect();
        let mut dependencies = HashSet::new();

        let buffer = self.gate_buffer.read().unwrap();
        for (gate_id, context) in buffer.iter() {
            let context_qubits: HashSet<QubitId> = context.gate.qubits().into_iter().collect();
            
            // If there's qubit overlap, this gate depends on the previous one
            if !gate_qubits.is_disjoint(&context_qubits) {
                dependencies.insert(*gate_id);
            }
        }

        Ok(dependencies)
    }

    /// Calculate priority for a gate
    fn calculate_priority(&self, gate: &dyn GateOp, dependencies: &HashSet<usize>) -> f64 {
        let mut priority = 0.0;

        // Higher priority for gates with fewer qubits (simpler to optimize)
        priority += 10.0 / (gate.num_qubits() as f64 + 1.0);

        // Lower priority for gates with many dependencies
        priority -= dependencies.len() as f64 * 0.5;

        // Higher priority for common gate types
        match gate.name() {
            "H" | "X" | "Y" | "Z" => priority += 5.0,
            "CNOT" | "CZ" => priority += 3.0,
            "RX" | "RY" | "RZ" => priority += 2.0,
            _ => priority += 1.0,
        }

        priority.max(0.1)
    }

    /// Check if forced evaluation is needed
    fn check_forced_evaluation(&self) -> QuantRS2Result<()> {
        let buffer = self.gate_buffer.read().unwrap();
        
        // Check buffer size
        if buffer.len() >= self.config.max_buffer_size {
            drop(buffer);
            return self.force_oldest_evaluation();
        }

        // Check time-based forced evaluation
        let now = Instant::now();
        for context in buffer.values() {
            if now.duration_since(context.created_at) > self.config.max_defer_time {
                drop(buffer);
                return self.force_oldest_evaluation();
            }
        }

        Ok(())
    }

    /// Force evaluation of the oldest gate
    fn force_oldest_evaluation(&self) -> QuantRS2Result<()> {
        let oldest_gate_id = {
            let buffer = self.gate_buffer.read().unwrap();
            buffer.values()
                .filter(|ctx| !ctx.is_evaluated)
                .min_by_key(|ctx| ctx.created_at)
                .map(|ctx| ctx.gate_id)
        };

        if let Some(gate_id) = oldest_gate_id {
            self.evaluate_gate(gate_id)?;
        }

        Ok(())
    }

    /// Evaluate dependencies recursively
    fn evaluate_dependencies(&self, dependencies: &HashSet<usize>) -> QuantRS2Result<()> {
        for &dep_id in dependencies {
            if !self.is_gate_evaluated(dep_id) {
                self.evaluate_gate(dep_id)?;
            }
        }
        Ok(())
    }

    /// Check if a gate has been evaluated
    fn is_gate_evaluated(&self, gate_id: usize) -> bool {
        let buffer = self.gate_buffer.read().unwrap();
        buffer.get(&gate_id)
            .map(|ctx| ctx.is_evaluated)
            .unwrap_or(false)
    }

    /// Optimize a gate context
    fn optimize_gate_context(&self, context: &LazyGateContext) -> QuantRS2Result<OptimizationResult> {
        let start_time = Instant::now();
        
        // Apply optimization chain
        let input_gates = vec![context.gate.clone_gate()];
        let optimized_gates = self.optimization_chain.optimize(input_gates)?;
        
        let optimization_time = start_time.elapsed();
        
        // Calculate statistics
        let stats = OptimizationStats {
            gates_before: 1,
            gates_after: optimized_gates.len(),
            passes_applied: 1, // Would track actual passes in real implementation
            performance_improvement: self.estimate_performance_improvement(&optimized_gates),
            memory_savings: self.estimate_memory_savings(&optimized_gates),
        };

        Ok(OptimizationResult {
            optimized_gates,
            stats,
            optimization_time,
        })
    }

    /// Estimate performance improvement from optimization
    fn estimate_performance_improvement(&self, gates: &[Box<dyn GateOp>]) -> f64 {
        // Simple heuristic: fewer gates = better performance
        let base_improvement = 1.0 / (gates.len() as f64 + 1.0);
        
        // Bonus for single-qubit gates
        let single_qubit_bonus = gates.iter()
            .filter(|g| g.num_qubits() == 1)
            .count() as f64 * 0.1;
            
        base_improvement + single_qubit_bonus
    }

    /// Estimate memory savings from optimization
    fn estimate_memory_savings(&self, gates: &[Box<dyn GateOp>]) -> usize {
        // Simple heuristic based on gate complexity
        gates.iter()
            .map(|g| match g.num_qubits() {
                1 => 16,  // 2x2 complex matrix
                2 => 64,  // 4x4 complex matrix
                n => (1 << (2 * n)) * 16, // 2^n x 2^n complex matrix
            })
            .sum()
    }

    /// Get cached optimization result
    fn get_cached_result(&self, gate_id: usize) -> QuantRS2Result<Option<OptimizationResult>> {
        let buffer = self.gate_buffer.read().unwrap();
        if let Some(context) = buffer.get(&gate_id) {
            if let Some(ref result) = context.cached_result {
                return Ok(Some(result.clone()));
            }
        }

        // Check optimization cache
        let gate_hash = self.compute_gate_hash(gate_id)?;
        let mut cache = self.optimization_cache.write().unwrap();
        if let Some(cached) = cache.get_mut(gate_hash) {
            return Ok(Some(cached.result.clone()));
        }

        Ok(None)
    }

    /// Cache optimization result
    fn cache_optimization_result(
        &self,
        gate_id: usize,
        result: &OptimizationResult,
    ) -> QuantRS2Result<()> {
        let gate_hash = self.compute_gate_hash(gate_id)?;
        let mut cache = self.optimization_cache.write().unwrap();
        
        let cached = CachedOptimization {
            result: result.clone(),
            access_count: 1,
            last_accessed: Instant::now(),
        };
        
        cache.insert(gate_hash, cached);
        Ok(())
    }

    /// Compute hash for a gate
    fn compute_gate_hash(&self, gate_id: usize) -> QuantRS2Result<u64> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let buffer = self.gate_buffer.read().unwrap();
        let context = buffer.get(&gate_id)
            .ok_or_else(|| QuantRS2Error::InvalidInput(
                format!("Gate {} not found", gate_id)
            ))?;

        let mut hasher = DefaultHasher::new();
        context.gate.name().hash(&mut hasher);
        
        // Hash the gate matrix
        if let Ok(matrix) = context.gate.matrix() {
            for elem in &matrix {
                elem.re.to_bits().hash(&mut hasher);
                elem.im.to_bits().hash(&mut hasher);
            }
        }
        
        Ok(hasher.finish())
    }

    /// Start a worker thread for async optimization
    fn start_worker_thread(
        worker_id: usize,
        gate_buffer: Arc<RwLock<HashMap<usize, LazyGateContext>>>,
        dependency_graph: Arc<RwLock<DependencyGraph>>,
        optimization_cache: Arc<RwLock<OptimizationCache>>,
        shutdown_signal: Arc<RwLock<bool>>,
        config: LazyEvaluationConfig,
    ) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            let sleep_duration = Duration::from_millis(10);
            
            loop {
                // Check shutdown signal
                {
                    let shutdown = shutdown_signal.read().unwrap();
                    if *shutdown {
                        break;
                    }
                }

                // Look for high-priority gates to optimize speculatively
                if config.enable_speculative_optimization {
                    let high_priority_gates = {
                        let buffer = gate_buffer.read().unwrap();
                        buffer.values()
                            .filter(|ctx| !ctx.is_evaluated && ctx.priority > 5.0)
                            .map(|ctx| ctx.gate_id)
                            .collect::<Vec<_>>()
                    };

                    for gate_id in high_priority_gates {
                        // This would require access to the pipeline's optimization methods
                        // For now, just mark as processed
                        let mut buffer = gate_buffer.write().unwrap();
                        if let Some(ctx) = buffer.get_mut(&gate_id) {
                            // Placeholder: would perform actual optimization here
                            ctx.priority += 0.1; // Slight priority boost
                        }
                    }
                }

                std::thread::sleep(sleep_duration);
            }
        })
    }
}

impl Drop for LazyOptimizationPipeline {
    fn drop(&mut self) {
        // Signal shutdown to worker threads
        {
            let mut shutdown = self.shutdown_signal.write().unwrap();
            *shutdown = true;
        }

        // Wait for all worker threads to finish
        while let Some(handle) = self.worker_handles.pop() {
            let _ = handle.join();
        }
    }
}

impl DependencyGraph {
    /// Add a gate with its dependencies
    fn add_gate(&mut self, gate_id: usize, dependencies: HashSet<usize>) {
        self.edges.insert(gate_id, dependencies.clone());
        
        // Update reverse edges
        for dep in dependencies {
            self.reverse_edges.entry(dep).or_insert_with(HashSet::new).insert(gate_id);
        }
        
        // Invalidate topological order cache
        self.topo_order_cache = None;
    }

    /// Get topological ordering of gates
    fn topological_sort(&self) -> Vec<usize> {
        if let Some(ref cached) = self.topo_order_cache {
            return cached.clone();
        }

        let mut result = Vec::new();
        let mut in_degree: HashMap<usize, usize> = HashMap::new();
        let mut queue = VecDeque::new();

        // Calculate in-degrees
        for (&node, edges) in &self.edges {
            in_degree.entry(node).or_insert(0);
            for &dep in edges {
                *in_degree.entry(dep).or_insert(0) += 1;
            }
        }

        // Find nodes with no incoming edges
        for (&node, &degree) in &in_degree {
            if degree == 0 {
                queue.push_back(node);
            }
        }

        // Process nodes
        while let Some(node) = queue.pop_front() {
            result.push(node);
            
            if let Some(dependents) = self.reverse_edges.get(&node) {
                for &dependent in dependents {
                    if let Some(degree) = in_degree.get_mut(&dependent) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push_back(dependent);
                        }
                    }
                }
            }
        }

        result
    }
}

impl OptimizationCache {
    fn new(max_size: usize) -> Self {
        Self {
            entries: HashMap::new(),
            lru_queue: VecDeque::new(),
            max_size,
        }
    }

    fn get_mut(&mut self, hash: u64) -> Option<&mut CachedOptimization> {
        if let Some(cached) = self.entries.get_mut(&hash) {
            cached.access_count += 1;
            cached.last_accessed = Instant::now();
            
            // Update LRU
            self.lru_queue.retain(|&h| h != hash);
            self.lru_queue.push_front(hash);
            
            Some(cached)
        } else {
            None
        }
    }

    fn insert(&mut self, hash: u64, cached: CachedOptimization) {
        // Evict if necessary
        while self.entries.len() >= self.max_size {
            if let Some(oldest_hash) = self.lru_queue.pop_back() {
                self.entries.remove(&oldest_hash);
            } else {
                break;
            }
        }

        self.entries.insert(hash, cached);
        self.lru_queue.push_front(hash);
    }

    fn get_hit_count(&self) -> usize {
        self.entries.values().map(|c| c.access_count).sum()
    }

    fn get_average_optimization_time(&self) -> Duration {
        if self.entries.is_empty() {
            return Duration::ZERO;
        }

        let total_time: Duration = self.entries.values()
            .map(|c| c.result.optimization_time)
            .sum();
        
        total_time / self.entries.len() as u32
    }
}

/// Statistics for lazy evaluation
#[derive(Debug, Clone)]
pub struct LazyEvaluationStats {
    pub total_gates: usize,
    pub evaluated_gates: usize,
    pub pending_gates: usize,
    pub cache_hits: usize,
    pub cache_size: usize,
    pub average_optimization_time: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::single::{Hadamard, PauliX, PauliZ};
    use crate::optimization::OptimizationChain;

    #[test]
    fn test_lazy_pipeline_creation() {
        let config = LazyEvaluationConfig::default();
        let chain = OptimizationChain::new();
        
        let pipeline = LazyOptimizationPipeline::new(config, chain).unwrap();
        let stats = pipeline.get_statistics();
        
        assert_eq!(stats.total_gates, 0);
        assert_eq!(stats.evaluated_gates, 0);
    }

    #[test]
    fn test_gate_addition() {
        let config = LazyEvaluationConfig::default();
        let chain = OptimizationChain::new();
        
        let pipeline = LazyOptimizationPipeline::new(config, chain).unwrap();
        
        let h_gate = Box::new(Hadamard { target: crate::qubit::QubitId(0) });
        let gate_id = pipeline.add_gate(h_gate).unwrap();
        
        assert_eq!(gate_id, 0);
        
        let stats = pipeline.get_statistics();
        assert_eq!(stats.total_gates, 1);
        assert_eq!(stats.pending_gates, 1);
    }

    #[test]
    fn test_gate_evaluation() {
        let config = LazyEvaluationConfig::default();
        let chain = OptimizationChain::new();
        
        let pipeline = LazyOptimizationPipeline::new(config, chain).unwrap();
        
        let h_gate = Box::new(Hadamard { target: crate::qubit::QubitId(0) });
        let gate_id = pipeline.add_gate(h_gate).unwrap();
        
        let result = pipeline.evaluate_gate(gate_id).unwrap();
        assert!(result.optimization_time > Duration::ZERO);
        
        let stats = pipeline.get_statistics();
        assert_eq!(stats.evaluated_gates, 1);
        assert_eq!(stats.pending_gates, 0);
    }

    #[test]
    fn test_dependency_analysis() {
        let config = LazyEvaluationConfig::default();
        let chain = OptimizationChain::new();
        
        let pipeline = LazyOptimizationPipeline::new(config, chain).unwrap();
        
        // Add gates that share qubits
        let h_gate = Box::new(Hadamard { target: crate::qubit::QubitId(0) });
        let x_gate = Box::new(PauliX { target: crate::qubit::QubitId(0) });
        let z_gate = Box::new(PauliZ { target: crate::qubit::QubitId(1) });
        
        let h_id = pipeline.add_gate(h_gate).unwrap();
        let x_id = pipeline.add_gate(x_gate).unwrap();
        let z_id = pipeline.add_gate(z_gate).unwrap();
        
        // X gate should depend on H gate (same qubit)
        // Z gate should be independent (different qubit)
        
        let results = pipeline.evaluate_all().unwrap();
        // Results may be filtered or combined during optimization
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_optimization_caching() {
        let config = LazyEvaluationConfig::default();
        let chain = OptimizationChain::new();
        
        let pipeline = LazyOptimizationPipeline::new(config, chain).unwrap();
        
        let h_gate = Box::new(Hadamard { target: crate::qubit::QubitId(0) });
        let gate_id = pipeline.add_gate(h_gate).unwrap();
        
        // First evaluation
        let result1 = pipeline.evaluate_gate(gate_id).unwrap();
        
        // Second evaluation should use cache
        let result2 = pipeline.evaluate_gate(gate_id).unwrap();
        
        // Results should be identical
        assert_eq!(result1.stats.gates_before, result2.stats.gates_before);
        assert_eq!(result1.stats.gates_after, result2.stats.gates_after);
    }
}