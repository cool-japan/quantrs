//! Algorithm optimization types for scientific performance optimization.
//!
//! This module contains problem decomposition, result caching,
//! approximation engines, and streaming processors.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use crate::applications::{
    drug_discovery::DrugDiscoveryProblem, materials_science::MaterialsOptimizationProblem,
    protein_folding::ProteinFoldingProblem,
};
use crate::ising::{IsingModel, QuboModel};

use super::config::{
    AlgorithmOptimizationConfig, ApproximationConfig, ApproximationStrategy, CachingConfig,
    DecompositionStrategy, StreamingConfig,
};
use super::memory::CacheStatistics;

/// Algorithm optimizer for improving computational efficiency
pub struct AlgorithmOptimizer {
    /// Configuration
    pub config: AlgorithmOptimizationConfig,
    /// Problem decomposer
    pub decomposer: ProblemDecomposer,
    /// Result cache
    pub result_cache: ResultCache,
    /// Approximation engine
    pub approximation_engine: ApproximationEngine,
    /// Streaming processor
    pub streaming_processor: StreamingProcessor,
}

impl AlgorithmOptimizer {
    /// Create a new algorithm optimizer
    #[must_use]
    pub fn new(config: AlgorithmOptimizationConfig) -> Self {
        Self {
            config,
            decomposer: ProblemDecomposer::new(),
            result_cache: ResultCache::new(),
            approximation_engine: ApproximationEngine::new(),
            streaming_processor: StreamingProcessor::new(),
        }
    }
}

/// Problem decomposer for hierarchical problem solving
#[derive(Debug)]
pub struct ProblemDecomposer {
    /// Decomposition strategy
    pub strategy: DecompositionStrategy,
    /// Subproblem registry
    pub subproblems: HashMap<String, Subproblem>,
    /// Decomposition statistics
    pub statistics: DecompositionStatistics,
}

impl ProblemDecomposer {
    /// Create a new problem decomposer
    #[must_use]
    pub fn new() -> Self {
        Self {
            strategy: DecompositionStrategy::Adaptive,
            subproblems: HashMap::new(),
            statistics: DecompositionStatistics::default(),
        }
    }

    /// Decompose a problem into subproblems
    pub fn decompose(&mut self, problem_id: &str, problem_data: ProblemData) -> Vec<String> {
        let subproblem_ids = self.create_subproblems(problem_id, &problem_data);
        self.statistics.decompositions += 1;
        subproblem_ids
    }

    /// Create subproblems based on strategy
    fn create_subproblems(&mut self, parent_id: &str, _problem_data: &ProblemData) -> Vec<String> {
        let mut subproblem_ids = Vec::new();

        // Create subproblems based on strategy
        let num_subproblems = match self.strategy {
            DecompositionStrategy::Uniform => 4,
            DecompositionStrategy::Adaptive => 4,
            DecompositionStrategy::GraphBased => 4,
            DecompositionStrategy::Hierarchical => 2,
        };

        for i in 0..num_subproblems {
            let id = format!("{parent_id}_sub_{i}");
            let subproblem = Subproblem {
                id: id.clone(),
                parent_id: Some(parent_id.to_string()),
                problem_data: ProblemData::Generic(Vec::new()),
                status: SubproblemStatus::Pending,
                dependencies: Vec::new(),
            };
            self.subproblems.insert(id.clone(), subproblem);
            subproblem_ids.push(id);
        }

        subproblem_ids
    }

    /// Get subproblem status
    pub fn get_status(&self, subproblem_id: &str) -> Option<&SubproblemStatus> {
        self.subproblems.get(subproblem_id).map(|s| &s.status)
    }

    /// Update subproblem status
    pub fn update_status(&mut self, subproblem_id: &str, status: SubproblemStatus) {
        if let Some(subproblem) = self.subproblems.get_mut(subproblem_id) {
            subproblem.status = status;
        }
    }
}

impl Default for ProblemDecomposer {
    fn default() -> Self {
        Self::new()
    }
}

/// Subproblem representation
#[derive(Debug)]
pub struct Subproblem {
    /// Subproblem identifier
    pub id: String,
    /// Parent problem
    pub parent_id: Option<String>,
    /// Problem data
    pub problem_data: ProblemData,
    /// Solution status
    pub status: SubproblemStatus,
    /// Dependencies
    pub dependencies: Vec<String>,
}

/// Problem data types
#[derive(Debug)]
pub enum ProblemData {
    /// Ising model
    Ising(IsingModel),
    /// QUBO model
    QUBO(QuboModel),
    /// Protein folding problem
    ProteinFolding(ProteinFoldingProblem),
    /// Materials science problem
    MaterialsScience(MaterialsOptimizationProblem),
    /// Drug discovery problem
    DrugDiscovery(DrugDiscoveryProblem),
    /// Generic data
    Generic(Vec<u8>),
}

/// Subproblem status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubproblemStatus {
    /// Not started
    Pending,
    /// Currently solving
    InProgress,
    /// Completed successfully
    Completed,
    /// Failed to solve
    Failed,
    /// Cancelled
    Cancelled,
}

/// Result cache for memoization
#[derive(Debug)]
pub struct ResultCache {
    /// Cache configuration
    pub config: CachingConfig,
    /// Cached results
    pub cache: HashMap<String, CachedResult>,
    /// Cache access order
    pub access_order: VecDeque<String>,
    /// Cache statistics
    pub statistics: CacheStatistics,
}

impl ResultCache {
    /// Create a new result cache
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: CachingConfig::default(),
            cache: HashMap::new(),
            access_order: VecDeque::new(),
            statistics: CacheStatistics::default(),
        }
    }

    /// Get cached result
    pub fn get(&mut self, key: &str) -> Option<&CachedResult> {
        if self.cache.contains_key(key) {
            self.statistics.record_hit();
            // Move to front
            self.access_order.retain(|k| k != key);
            self.access_order.push_front(key.to_string());

            // Update access count
            if let Some(result) = self.cache.get_mut(key) {
                result.access_count += 1;
            }

            self.cache.get(key)
        } else {
            self.statistics.record_miss();
            None
        }
    }

    /// Cache a result
    pub fn put(&mut self, key: String, result_data: Vec<u8>, quality_score: f64) {
        // Evict if necessary
        while self.cache.len() >= self.config.cache_size_limit {
            if let Some(lru_key) = self.access_order.pop_back() {
                self.cache.remove(&lru_key);
            }
        }

        let cached = CachedResult {
            result_data,
            timestamp: Instant::now(),
            access_count: 1,
            quality_score,
        };

        self.cache.insert(key.clone(), cached);
        self.access_order.push_front(key);
    }

    /// Check if key exists
    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        self.cache.contains_key(key)
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.access_order.clear();
    }
}

impl Default for ResultCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Cached result representation
#[derive(Debug, Clone)]
pub struct CachedResult {
    /// Result data
    pub result_data: Vec<u8>,
    /// Cache timestamp
    pub timestamp: Instant,
    /// Access count
    pub access_count: u64,
    /// Result quality
    pub quality_score: f64,
}

/// Approximation engine for fast approximate solutions
#[derive(Debug)]
pub struct ApproximationEngine {
    /// Configuration
    pub config: ApproximationConfig,
    /// Available strategies
    pub strategies: Vec<ApproximationStrategy>,
    /// Strategy performance
    pub strategy_performance: HashMap<ApproximationStrategy, StrategyPerformance>,
}

impl ApproximationEngine {
    /// Create a new approximation engine
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: ApproximationConfig::default(),
            strategies: vec![
                ApproximationStrategy::Sampling,
                ApproximationStrategy::Clustering,
                ApproximationStrategy::DimensionalityReduction,
            ],
            strategy_performance: HashMap::new(),
        }
    }

    /// Select best strategy based on problem characteristics
    #[must_use]
    pub fn select_strategy(&self) -> ApproximationStrategy {
        // Find strategy with best average quality
        self.strategy_performance
            .iter()
            .max_by(|a, b| {
                a.1.average_quality
                    .partial_cmp(&b.1.average_quality)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(s, _)| s.clone())
            .unwrap_or(ApproximationStrategy::Sampling)
    }

    /// Record strategy performance
    pub fn record_performance(
        &mut self,
        strategy: ApproximationStrategy,
        quality: f64,
        speedup: f64,
        success: bool,
    ) {
        let perf = self
            .strategy_performance
            .entry(strategy.clone())
            .or_insert_with(|| StrategyPerformance::new(strategy));

        perf.usage_count += 1;
        if success {
            // Update rolling averages
            let n = perf.usage_count as f64;
            perf.average_quality = (perf.average_quality * (n - 1.0) + quality) / n;
            perf.average_speedup = (perf.average_speedup * (n - 1.0) + speedup) / n;
            perf.success_rate = (perf.success_rate * (n - 1.0) + 1.0) / n;
        } else {
            perf.success_rate =
                (perf.success_rate * (perf.usage_count - 1) as f64) / perf.usage_count as f64;
        }
    }
}

impl Default for ApproximationEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Strategy performance tracking
#[derive(Debug, Clone)]
pub struct StrategyPerformance {
    /// Strategy
    pub strategy: ApproximationStrategy,
    /// Success rate
    pub success_rate: f64,
    /// Average quality
    pub average_quality: f64,
    /// Average speedup
    pub average_speedup: f64,
    /// Usage count
    pub usage_count: u64,
}

impl StrategyPerformance {
    /// Create new strategy performance tracker
    #[must_use]
    pub fn new(strategy: ApproximationStrategy) -> Self {
        Self {
            strategy,
            success_rate: 0.0,
            average_quality: 0.0,
            average_speedup: 0.0,
            usage_count: 0,
        }
    }
}

/// Streaming processor for continuous data processing
#[derive(Debug)]
pub struct StreamingProcessor {
    /// Configuration
    pub config: StreamingConfig,
    /// Processing windows
    pub windows: Vec<ProcessingWindow>,
    /// Stream statistics
    pub statistics: StreamingStatistics,
}

impl StreamingProcessor {
    /// Create a new streaming processor
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: StreamingConfig::default(),
            windows: Vec::new(),
            statistics: StreamingStatistics::default(),
        }
    }

    /// Add element to stream
    pub fn add_element(&mut self, data: Vec<u8>, metadata: HashMap<String, String>) {
        let element = StreamElement {
            data,
            timestamp: Instant::now(),
            metadata,
        };

        // Add to current window or create new window
        if let Some(window) = self.windows.last_mut() {
            if window.data.len() < self.config.window_size {
                window.data.push_back(element);
                return;
            }
        }

        // Create new window
        let mut new_window = ProcessingWindow {
            id: format!("window_{}", self.windows.len()),
            data: VecDeque::new(),
            start_time: Instant::now(),
            duration: Duration::from_secs(60),
        };
        new_window.data.push_back(element);
        self.windows.push(new_window);
        self.statistics.windows_created += 1;
    }

    /// Process current windows
    pub fn process(&mut self) -> Vec<StreamElement> {
        let mut processed = Vec::new();

        // Process completed windows
        let now = Instant::now();
        for window in &mut self.windows {
            if now.duration_since(window.start_time) >= window.duration {
                while let Some(element) = window.data.pop_front() {
                    processed.push(element);
                    self.statistics.elements_processed += 1;
                }
            }
        }

        // Remove empty windows
        self.windows.retain(|w| !w.data.is_empty());

        processed
    }
}

impl Default for StreamingProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Processing window for streaming
#[derive(Debug)]
pub struct ProcessingWindow {
    /// Window identifier
    pub id: String,
    /// Window data
    pub data: VecDeque<StreamElement>,
    /// Window start time
    pub start_time: Instant,
    /// Window duration
    pub duration: Duration,
}

/// Stream element
#[derive(Debug, Clone)]
pub struct StreamElement {
    /// Element data
    pub data: Vec<u8>,
    /// Timestamp
    pub timestamp: Instant,
    /// Element metadata
    pub metadata: HashMap<String, String>,
}

/// Decomposition statistics
#[derive(Debug, Clone, Default)]
pub struct DecompositionStatistics {
    /// Total decompositions
    pub decompositions: u64,
    /// Subproblems created
    pub subproblems_created: u64,
    /// Subproblems solved
    pub subproblems_solved: u64,
    /// Average subproblem size
    pub avg_subproblem_size: f64,
}

/// Streaming statistics
#[derive(Debug, Clone, Default)]
pub struct StreamingStatistics {
    /// Elements processed
    pub elements_processed: u64,
    /// Windows created
    pub windows_created: u64,
    /// Average window size
    pub avg_window_size: f64,
    /// Processing rate
    pub processing_rate: f64,
}
