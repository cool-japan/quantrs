//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Barrier, Mutex, RwLock};

use super::types::{
    AutoParallelConfig, LoadBalancer, ParallelPerformanceStats, ParallelizationAnalysis,
};

/// Automatic parallelization engine for quantum circuits
pub struct AutoParallelEngine {
    /// Configuration
    pub(super) config: AutoParallelConfig,
    /// Analysis cache for circuits
    pub(super) analysis_cache: Arc<RwLock<HashMap<u64, ParallelizationAnalysis>>>,
    /// Performance statistics
    pub(super) performance_stats: Arc<Mutex<ParallelPerformanceStats>>,
    /// `SciRS2` integration components
    /// Load balancer
    pub(super) load_balancer: Arc<Mutex<LoadBalancer>>,
}
