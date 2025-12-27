//! Optimized quantum circuit simulator with automatic selection of best implementation
//!
//! This module provides a high-performance simulator implementation that automatically
//! selects the most appropriate optimization strategy based on qubit count and hardware support.

use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_core::{error::QuantRS2Result, register::Register};

/// An optimized simulator for quantum circuits that automatically selects the best
/// implementation based on circuit size and hardware capabilities
#[derive(Debug, Clone)]
pub struct OptimizedSimulator {
    /// Use SIMD acceleration when available
    _use_simd: bool,
    /// Use memory-efficient algorithms for large qubit counts
    memory_efficient: bool,
    /// Qubit count threshold for switching to memory-efficient implementation
    memory_efficient_threshold: usize,
}

impl OptimizedSimulator {
    /// Create a new optimized simulator with default settings
    #[must_use]
    pub const fn new() -> Self {
        Self {
            _use_simd: cfg!(feature = "simd"),
            memory_efficient: cfg!(feature = "memory_efficient"),
            memory_efficient_threshold: 25, // Switch to memory-efficient at 25+ qubits
        }
    }

    /// Create a new optimized simulator with custom settings
    #[must_use]
    pub const fn with_options(
        use_simd: bool,
        memory_efficient: bool,
        memory_efficient_threshold: usize,
    ) -> Self {
        Self {
            _use_simd: use_simd,
            memory_efficient,
            memory_efficient_threshold,
        }
    }

    /// Create a new simulator optimized for maximum performance
    #[must_use]
    pub const fn high_performance() -> Self {
        Self {
            _use_simd: true,
            memory_efficient: true,
            memory_efficient_threshold: 28, // Higher threshold favors performance over memory usage
        }
    }

    /// Create a new simulator optimized for memory efficiency
    #[must_use]
    pub const fn memory_efficient() -> Self {
        Self {
            _use_simd: true,
            memory_efficient: true,
            memory_efficient_threshold: 20, // Lower threshold favors memory usage over performance
        }
    }

    /// Check if SIMD is available on this system
    #[must_use]
    pub fn is_simd_available() -> bool {
        use quantrs2_core::platform::PlatformCapabilities;
        let platform = PlatformCapabilities::detect();
        platform.cpu.simd.avx2 || platform.cpu.simd.avx512 || platform.cpu.simd.sse4_1
    }
}

impl Default for OptimizedSimulator {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> Simulator<N> for OptimizedSimulator {
    fn run(&self, circuit: &Circuit<N>) -> QuantRS2Result<Register<N>> {
        // For extremely large circuits, memory efficiency is critical
        if N >= 30 && self.memory_efficient {
            // Defer to chunked implementation
            let chunked_simulator =
                crate::optimized_simulator_chunked::OptimizedSimulatorChunked::new();
            return chunked_simulator.run(circuit);
        }

        // For large circuits, use memory-efficient implementation if enabled
        if N >= self.memory_efficient_threshold && self.memory_efficient {
            // Defer to chunked implementation
            let chunked_simulator =
                crate::optimized_simulator_chunked::OptimizedSimulatorChunked::new();
            return chunked_simulator.run(circuit);
        }

        // For smaller circuits, use the simple optimized implementation
        let standard_simulator = crate::optimized_simulator_simple::OptimizedSimulatorSimple::new();
        standard_simulator.run(circuit)
    }
}
