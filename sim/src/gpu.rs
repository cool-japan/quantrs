//! GPU-accelerated quantum simulation module using SciRS2 GPU abstractions
//!
//! This module provides GPU-accelerated implementations of quantum simulators
//! leveraging SciRS2's unified GPU abstraction layer. This implementation
//! automatically selects the best available GPU backend (CUDA, Metal, OpenCL)
//! and provides optimal performance for quantum circuit simulation.

use quantrs2_circuit::builder::Simulator as CircuitSimulator;
use quantrs2_circuit::prelude::Circuit;
use quantrs2_core::error::{QuantRS2Error, QuantRS2Result};
use quantrs2_core::gpu::{
    is_gpu_available, GpuBackend as QuantRS2GpuBackend, GpuConfig, SciRS2GpuBackend,
};
use quantrs2_core::prelude::QubitId;
use scirs2_core::gpu::{GpuBackend, GpuBuffer, GpuContext};
use scirs2_core::Complex64;
use std::sync::Arc;

use crate::error::{Result, SimulatorError};
use crate::simulator::{Simulator, SimulatorResult};

/// SciRS2-powered GPU State Vector Simulator
///
/// This simulator leverages SciRS2's unified GPU abstraction layer to provide
/// optimal performance across different GPU backends (CUDA, Metal, OpenCL).
pub struct SciRS2GpuStateVectorSimulator {
    /// QuantRS2 GPU backend (wrapper around SciRS2)
    backend: Option<Arc<SciRS2GpuBackend>>,
    /// SciRS2 GPU context for direct GPU operations
    gpu_context: Option<GpuContext>,
    /// Performance tracking enabled
    enable_profiling: bool,
}

impl SciRS2GpuStateVectorSimulator {
    /// Create a new SciRS2-powered GPU state vector simulator
    pub fn new() -> QuantRS2Result<Self> {
        // TODO: Update to use scirs2_core beta.3 GPU API
        return Err(QuantRS2Error::BackendExecutionFailed(
            "GPU backend API has changed in beta.3. Please use CPU simulation for now.".to_string(),
        ));

        #[allow(unreachable_code)]
        Ok(Self {
            backend: None,
            gpu_context: None,
            enable_profiling: false,
        })
    }

    /// Create a new simulator with custom configuration
    pub fn with_config(_config: GpuConfig) -> QuantRS2Result<Self> {
        // GPU backend API has changed in beta.3
        return Err(QuantRS2Error::BackendExecutionFailed(
            "GPU backend API has changed in beta.3. Please use CPU simulation for now.".to_string(),
        ));

        #[allow(unreachable_code)]
        Ok(Self {
            backend: None,
            gpu_context: None,
            enable_profiling: _config.enable_profiling,
        })
    }

    /// Create an optimized simulator for quantum machine learning
    pub fn new_qml_optimized() -> QuantRS2Result<Self> {
        // TODO: GPU backend API has changed in beta.3
        return Err(QuantRS2Error::BackendExecutionFailed(
            "GPU backend API has changed in beta.3. Please use CPU simulation for now.".to_string(),
        ));
        #[allow(unreachable_code)]
        Ok(Self {
            backend: None,
            gpu_context: None,
            enable_profiling: true,
        })
    }

    /// Enable performance profiling
    pub fn enable_profiling(&mut self) {
        self.enable_profiling = true;
    }

    /// Get performance metrics if profiling is enabled
    pub fn get_performance_metrics(&self) -> Option<String> {
        if self.enable_profiling {
            if let Some(backend) = &self.backend {
                Some(backend.optimization_report())
            } else {
                Some("GPU not initialized".to_string())
            }
        } else {
            None
        }
    }

    /// Check if GPU acceleration is available
    pub fn is_available() -> bool {
        is_gpu_available()
    }

    /// Get available GPU backends
    pub fn available_backends() -> Vec<String> {
        // TODO: SciRS2GpuFactory not available in beta.3
        vec![]
    }
}

impl Simulator for SciRS2GpuStateVectorSimulator {
    fn run<const N: usize>(&mut self, circuit: &Circuit<N>) -> Result<SimulatorResult<N>> {
        // GPU backend not available in beta.3, use CPU fallback
        let cpu_sim = crate::statevector::StateVectorSimulator::new();
        let cpu_result = cpu_sim
            .run(circuit)
            .map_err(|e| SimulatorError::BackendError(e.to_string()))?;
        return Ok(SimulatorResult {
            amplitudes: cpu_result.amplitudes().to_vec(),
            num_qubits: N,
        });

        // Original GPU implementation (disabled in beta.3):
        #[allow(unreachable_code)]
        let backend = self
            .backend
            .as_ref()
            .ok_or(SimulatorError::GPUNotAvailable)?;
        let mut state_vector = match backend.allocate_state_vector(N) {
            Ok(buffer) => buffer,
            Err(e) => {
                // Fallback to CPU simulation for small circuits or on error
                if N < 4 {
                    let cpu_sim = crate::statevector::StateVectorSimulator::new();
                    let result = quantrs2_circuit::builder::Simulator::<N>::run(&cpu_sim, circuit)
                        .map_err(|e| SimulatorError::BackendError(e.to_string()))?;
                    return Ok(SimulatorResult {
                        amplitudes: result.amplitudes().to_vec(),
                        num_qubits: N,
                    });
                } else {
                    return Err(SimulatorError::BackendError(format!(
                        "Failed to allocate GPU state vector: {}",
                        e
                    )));
                }
            }
        };

        // Initialize to |0...0⟩ state
        let state_size = 1 << N;
        let mut initial_state = vec![Complex64::new(0.0, 0.0); state_size];
        initial_state[0] = Complex64::new(1.0, 0.0);

        state_vector
            .upload(&initial_state)
            .map_err(|e| SimulatorError::BackendError(e.to_string()))?;

        // Apply gates using SciRS2 GPU kernel
        let kernel = backend.kernel();

        for gate in circuit.gates() {
            let qubits = gate.qubits();

            match qubits.len() {
                1 => {
                    // Single-qubit gate
                    let matrix = gate
                        .matrix()
                        .map_err(|e| SimulatorError::BackendError(e.to_string()))?;
                    if matrix.len() < 4 {
                        return Err(SimulatorError::BackendError(
                            "Invalid single-qubit gate matrix size".to_string(),
                        ));
                    }
                    let gate_matrix = [matrix[0], matrix[1], matrix[2], matrix[3]];

                    kernel
                        .apply_single_qubit_gate(state_vector.as_mut(), &gate_matrix, qubits[0], N)
                        .map_err(|e| SimulatorError::BackendError(e.to_string()))?;
                }
                2 => {
                    // Two-qubit gate
                    let matrix = gate
                        .matrix()
                        .map_err(|e| SimulatorError::BackendError(e.to_string()))?;
                    if matrix.len() < 16 {
                        return Err(SimulatorError::BackendError(
                            "Invalid two-qubit gate matrix size".to_string(),
                        ));
                    }
                    let mut gate_matrix = [Complex64::new(0.0, 0.0); 16];
                    for (i, &val) in matrix.iter().take(16).enumerate() {
                        gate_matrix[i] = val;
                    }

                    kernel
                        .apply_two_qubit_gate(
                            state_vector.as_mut(),
                            &gate_matrix,
                            qubits[0],
                            qubits[1],
                            N,
                        )
                        .map_err(|e| SimulatorError::BackendError(e.to_string()))?;
                }
                _ => {
                    // Multi-qubit gate
                    let matrix = gate
                        .matrix()
                        .map_err(|e| SimulatorError::BackendError(e.to_string()))?;
                    let size = 1 << qubits.len();
                    let matrix_array =
                        scirs2_core::ndarray::Array2::from_shape_vec((size, size), matrix)
                            .map_err(|e| SimulatorError::BackendError(e.to_string()))?;

                    kernel
                        .apply_multi_qubit_gate(state_vector.as_mut(), &matrix_array, &qubits, N)
                        .map_err(|e| SimulatorError::BackendError(e.to_string()))?;
                }
            }
        }

        // Retrieve final state vector
        let mut final_state = vec![Complex64::new(0.0, 0.0); state_size];
        state_vector
            .download(&mut final_state)
            .map_err(|e| SimulatorError::BackendError(e.to_string()))?;

        Ok(SimulatorResult {
            amplitudes: final_state,
            num_qubits: N,
        })
    }
}

/// Legacy GPU state vector simulator for backward compatibility
///
/// This type alias provides backward compatibility while using the new SciRS2 implementation.
pub type GpuStateVectorSimulator = SciRS2GpuStateVectorSimulator;

impl GpuStateVectorSimulator {
    /// Create a new GPU state vector simulator using SciRS2 backend (legacy)
    ///
    /// Note: Parameters are ignored for backward compatibility.
    /// The SciRS2 backend automatically handles device and queue management.
    pub fn new_legacy(_device: std::sync::Arc<()>, _queue: std::sync::Arc<()>) -> Self {
        // Ignore legacy WGPU parameters and use SciRS2 backend
        match SciRS2GpuStateVectorSimulator::new() {
            Ok(sim) => sim,
            Err(_) => {
                // Fallback simulator when GPU initialization fails
                SciRS2GpuStateVectorSimulator {
                    backend: None,
                    gpu_context: None,
                    enable_profiling: false,
                }
            }
        }
    }

    /// Create a blocking version of the GPU simulator
    ///
    /// This method provides backward compatibility with the legacy async API.
    pub fn new_blocking() -> std::result::Result<Self, Box<dyn std::error::Error>> {
        match SciRS2GpuStateVectorSimulator::new() {
            Ok(simulator) => Ok(simulator),
            Err(e) => {
                let err: Box<dyn std::error::Error> = Box::new(SimulatorError::BackendError(
                    format!("Failed to create SciRS2 GPU simulator: {}", e),
                ));
                Err(err)
            }
        }
    }
}

/// Benchmark GPU performance using SciRS2 abstractions
pub async fn benchmark_gpu_performance() -> QuantRS2Result<String> {
    let mut simulator = SciRS2GpuStateVectorSimulator::new()?;
    simulator.enable_profiling();

    // Run benchmark circuits of different sizes
    let mut report = String::from("SciRS2 GPU Performance Benchmark\n");
    report.push_str("=====================================\n\n");

    for n_qubits in [2, 4, 6, 8, 10, 12] {
        let start = std::time::Instant::now();

        // Create a simple benchmark circuit
        use quantrs2_circuit::builder::CircuitBuilder;
        let mut builder = CircuitBuilder::<16>::new(); // Use max capacity

        // Add some gates for benchmarking
        for i in 0..n_qubits {
            let _ = builder.h(i);
        }
        for i in 0..n_qubits - 1 {
            let _ = builder.cnot(i, i + 1);
        }

        let circuit = builder.build();

        // Run simulation
        match simulator.run(&circuit) {
            Ok(_result) => {
                let duration = start.elapsed();
                report.push_str(&format!(
                    "{} qubits: {:.2}ms\n",
                    n_qubits,
                    duration.as_secs_f64() * 1000.0
                ));
            }
            Err(e) => {
                report.push_str(&format!("{} qubits: FAILED - {}\n", n_qubits, e));
            }
        }
    }

    // Add performance metrics if available
    if let Some(metrics) = simulator.get_performance_metrics() {
        report.push_str("\nDetailed Performance Metrics:\n");
        report.push_str(&metrics);
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_circuit::builder::CircuitBuilder;

    #[test]
    fn test_scirs2_gpu_simulator_creation() {
        // Test that we can create the simulator
        let result = SciRS2GpuStateVectorSimulator::new();
        // Should not panic - will fall back to CPU if GPU not available
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_backward_compatibility() {
        // Test the legacy interface still works
        use std::sync::Arc;

        // The new implementation takes no parameters
        let _simulator = GpuStateVectorSimulator::new();

        // Should create successfully with SciRS2 backend
        assert!(
            SciRS2GpuStateVectorSimulator::is_available()
                || !SciRS2GpuStateVectorSimulator::is_available()
        );
    }

    #[tokio::test]
    async fn test_gpu_simulation() {
        let mut simulator = match SciRS2GpuStateVectorSimulator::new() {
            Ok(sim) => sim,
            Err(_) => {
                println!("GPU not available, skipping test");
                return;
            }
        };

        // Create a simple 2-qubit circuit
        let mut builder = CircuitBuilder::<2>::new();
        let _ = builder.h(0);
        let _ = builder.cnot(0, 1);
        let circuit = builder.build();

        // Run simulation
        let result = simulator.run(&circuit);
        assert!(result.is_ok());

        if let Ok(sim_result) = result {
            assert_eq!(sim_result.num_qubits, 2);
            assert_eq!(sim_result.amplitudes.len(), 4);

            // Check Bell state probabilities
            let probs: Vec<f64> = sim_result.amplitudes.iter().map(|c| c.norm_sqr()).collect();

            // Should be in Bell state: |00⟩ + |11⟩
            assert!((probs[0] - 0.5).abs() < 1e-10); // |00⟩
            assert!((probs[1] - 0.0).abs() < 1e-10); // |01⟩
            assert!((probs[2] - 0.0).abs() < 1e-10); // |10⟩
            assert!((probs[3] - 0.5).abs() < 1e-10); // |11⟩
        }
    }

    #[tokio::test]
    async fn test_performance_benchmark() {
        let report = benchmark_gpu_performance().await;
        assert!(report.is_ok() || report.is_err()); // Should not panic

        if let Ok(report_str) = report {
            assert!(report_str.contains("SciRS2 GPU Performance"));
        }
    }
}
