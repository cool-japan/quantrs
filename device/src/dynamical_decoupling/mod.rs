//! Dynamical Decoupling Sequences with SciRS2 optimization
//!
//! This module provides comprehensive dynamical decoupling (DD) sequence generation,
//! optimization, and analysis using SciRS2's advanced optimization and statistical capabilities
//! for robust coherence preservation on quantum hardware.

pub mod config;
pub mod sequences;

pub use config::*;
pub use sequences::*;

use std::collections::HashMap;
use std::time::{Duration, Instant};

use quantrs2_circuit::prelude::*;
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};

use crate::{
    backend_traits::{query_backend_capabilities, BackendCapabilities},
    calibration::{CalibrationManager, DeviceCalibration},
    noise_model::CalibrationNoiseModel,
    topology::HardwareTopology,
    translation::HardwareBackend,
    CircuitResult, DeviceError, DeviceResult,
};

/// Main result type for dynamical decoupling operations
#[derive(Debug, Clone)]
pub struct DynamicalDecouplingResult {
    /// Optimized DD sequence
    pub optimized_sequence: DDSequence,
    /// Execution timing
    pub execution_time: Duration,
    /// Success status
    pub success: bool,
    /// Quality metrics
    pub quality_score: f64,
}

/// Circuit executor trait for DD operations
pub trait DDCircuitExecutor {
    /// Execute a circuit and return results
    fn execute_circuit<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> Result<CircuitExecutionResults, DeviceError>;

    /// Get backend capabilities
    fn get_capabilities(&self) -> BackendCapabilities;

    /// Estimate execution time
    fn estimate_execution_time<const N: usize>(&self, circuit: &Circuit<N>) -> Duration;
}

/// Circuit execution results
#[derive(Debug, Clone)]
pub struct CircuitExecutionResults {
    /// Measurement results
    pub measurements: HashMap<String, Vec<i32>>,
    /// Execution fidelity
    pub fidelity: f64,
    /// Execution time
    pub execution_time: Duration,
    /// Error rates
    pub error_rates: HashMap<String, f64>,
}
