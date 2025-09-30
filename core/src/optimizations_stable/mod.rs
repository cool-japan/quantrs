//! Stable Quantum Computing Optimizations
//!
//! This module provides quantum computing optimizations that work with stable
//! dependencies and don't rely on experimental scirs2-core features.
//! These optimizations provide immediate performance benefits.

pub mod adaptive_precision;
pub mod circuit_optimization;
pub mod gate_fusion;
pub mod quantum_cache;
pub mod simd_gates;

pub use adaptive_precision::{
    adapt_precision_for_circuit, AdaptivePrecisionManager, PrecisionLevel,
};
pub use circuit_optimization::{
    optimize_circuit, CircuitMetrics, CircuitOptimizer, OptimizationLevel,
};
pub use gate_fusion::{apply_gate_fusion, FusedGateSequence, FusionRule, GateFusionEngine};
pub use quantum_cache::{get_global_cache, CacheKey, CachedResult, StableQuantumCache};
pub use simd_gates::{process_gates_simd, SIMDGateProcessor, VectorizedOperation};

/// Initialize all stable optimization systems
pub fn initialize_stable_optimizations() -> crate::error::QuantRS2Result<()> {
    // Initialize quantum cache
    let _cache = get_global_cache();

    // Initialize SIMD processor
    let _simd = SIMDGateProcessor::new();

    // Precompute common gate matrices
    precompute_common_gates()?;

    Ok(())
}

/// Precompute and cache common quantum gate matrices
fn precompute_common_gates() -> crate::error::QuantRS2Result<()> {
    use scirs2_core::Complex64;
    use std::f64::consts::{FRAC_1_SQRT_2, PI};

    let cache = get_global_cache();

    // Pauli matrices
    cache.insert(
        CacheKey::new("pauli_x", vec![], 1),
        CachedResult::Matrix(vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]),
    );

    cache.insert(
        CacheKey::new("pauli_y", vec![], 1),
        CachedResult::Matrix(vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 0.0),
        ]),
    );

    cache.insert(
        CacheKey::new("pauli_z", vec![], 1),
        CachedResult::Matrix(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ]),
    );

    // Hadamard matrix
    cache.insert(
        CacheKey::new("hadamard", vec![], 1),
        CachedResult::Matrix(vec![
            Complex64::new(FRAC_1_SQRT_2, 0.0),
            Complex64::new(FRAC_1_SQRT_2, 0.0),
            Complex64::new(FRAC_1_SQRT_2, 0.0),
            Complex64::new(-FRAC_1_SQRT_2, 0.0),
        ]),
    );

    // CNOT matrix
    cache.insert(
        CacheKey::new("cnot", vec![], 2),
        CachedResult::Matrix(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]),
    );

    // Common rotation angles
    for &angle in &[PI / 4.0, PI / 2.0, 3.0 * PI / 4.0, PI] {
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();

        // RX rotation
        cache.insert(
            CacheKey::new("rx", vec![angle], 1),
            CachedResult::Matrix(vec![
                Complex64::new(cos_half, 0.0),
                Complex64::new(0.0, -sin_half),
                Complex64::new(0.0, -sin_half),
                Complex64::new(cos_half, 0.0),
            ]),
        );

        // RY rotation
        cache.insert(
            CacheKey::new("ry", vec![angle], 1),
            CachedResult::Matrix(vec![
                Complex64::new(cos_half, 0.0),
                Complex64::new(-sin_half, 0.0),
                Complex64::new(sin_half, 0.0),
                Complex64::new(cos_half, 0.0),
            ]),
        );

        // RZ rotation
        cache.insert(
            CacheKey::new("rz", vec![angle], 1),
            CachedResult::Matrix(vec![
                Complex64::new(cos_half, -sin_half),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(cos_half, sin_half),
            ]),
        );
    }

    Ok(())
}

/// Get comprehensive optimization statistics
pub fn get_optimization_statistics() -> OptimizationStatistics {
    OptimizationStatistics {
        cache_stats: get_global_cache().get_statistics(),
        fusion_stats: GateFusionEngine::get_global_statistics(),
        simd_stats: SIMDGateProcessor::get_global_statistics(),
    }
}

/// Combined optimization statistics
#[derive(Debug, Clone)]
pub struct OptimizationStatistics {
    pub cache_stats: quantum_cache::CacheStatistics,
    pub fusion_stats: gate_fusion::FusionStatistics,
    pub simd_stats: simd_gates::SIMDStatistics,
}
