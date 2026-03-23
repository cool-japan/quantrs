//! Advanced Job Priority and Scheduling Optimization for Quantum Hardware
//!
//! This module provides comprehensive job scheduling, prioritization, and optimization
//! capabilities for quantum hardware backends, including:
//! - Multi-level priority queue management
//! - Intelligent resource allocation and load balancing
//! - Cross-provider job coordination
//! - SciRS2-powered scheduling optimization algorithms
//! - Queue analytics and prediction
//! - Job persistence and recovery mechanisms
//! - Dynamic backend selection based on performance metrics

mod impls_core;
mod impls_extended;
mod impls_util;
pub mod types;

// Shared fallback module for when scirs2 is not available
#[cfg(not(feature = "scirs2"))]
pub(crate) mod fallback_scirs2 {
    pub fn mean(_data: &[f64]) -> f64 {
        0.0
    }
    pub fn std(_data: &[f64]) -> f64 {
        1.0
    }
    pub fn correlation(_x: &[f64], _y: &[f64]) -> f64 {
        0.0
    }
    pub struct OptimizeResult {
        pub x: Vec<f64>,
        pub success: bool,
    }
    pub fn minimize<F>(_func: F, _x0: Vec<f64>, _bounds: Option<Vec<(f64, f64)>>) -> OptimizeResult
    where
        F: Fn(&[f64]) -> f64,
    {
        OptimizeResult {
            x: vec![0.0],
            success: false,
        }
    }
}

// Re-export all public types for backward compatibility
pub use impls_core::*;
pub use impls_extended::*;
pub use impls_util::*;
pub use types::*;
