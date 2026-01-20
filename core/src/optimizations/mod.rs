//! Quantum Computing Optimizations using SciRS2 Beta.3 Features
//!
//! This module contains optimizations that leverage the advanced features
//! available in scirs2-core beta.3, including:
//!
//! - Advanced caching and memoization
//! - Memory pool management
//! - Performance profiling and monitoring
//! - SIMD-accelerated computations
//! - Intelligent load balancing
//!

pub mod gate_cache;
pub mod memory_optimization;
pub mod profiling_integration;

pub use gate_cache::{
    global_gate_cache, CachedGateMatrix, GateKey, QuantumGateCache, QuantumGateCacheStats,
};
pub use memory_optimization::{
    optimized_state_vector_allocation, QuantumBufferPool, StateVectorManager,
};
pub use profiling_integration::{
    enable_quantum_profiling, QuantumOperationProfile, QuantumProfiler,
};

/// Initialize all optimization systems
pub fn initialize_optimizations() -> crate::error::QuantRS2Result<()> {
    // Initialize gate cache with common gates
    global_gate_cache().prewarm_common_gates()?;

    // Initialize memory pools
    memory_optimization::initialize_buffer_pools();

    // Enable profiling
    profiling_integration::enable_quantum_profiling();

    Ok(())
}

/// Get optimization statistics
pub fn get_optimization_stats() -> OptimizationStats {
    OptimizationStats {
        gate_cache_stats: global_gate_cache().get_performance_stats(),
        memory_stats: memory_optimization::get_memory_usage_stats(),
        profiling_active: profiling_integration::is_profiling_active(),
    }
}

/// Combined optimization statistics
#[derive(Debug, Clone)]
pub struct OptimizationStats {
    pub gate_cache_stats: QuantumGateCacheStats,
    pub memory_stats: memory_optimization::MemoryUsageStats,
    pub profiling_active: bool,
}
