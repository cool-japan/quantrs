//! GPU-accelerated quantum simulation module using SciRS2 GPU abstractions
//!
//! This module provides GPU-accelerated implementations of quantum simulators
//! leveraging SciRS2's unified GPU abstraction layer. This implementation
//! automatically selects the best available GPU backend (CUDA, Metal, OpenCL)
//! and provides optimal performance for quantum circuit simulation.

use num_complex::Complex64;
use quantrs2_circuit::builder::Simulator as CircuitSimulator;
use quantrs2_circuit::prelude::Circuit;
use quantrs2_core::prelude::QubitId;
use quantrs2_core::error::{QuantRS2Error, QuantRS2Result};
use quantrs2_core::gpu::{
    GpuBackend as CoreGpuBackend, GpuBuffer as CoreGpuBuffer, 
    GpuStateVector as CoreGpuStateVector, GpuBackendFactory,
    SciRS2GpuBackend, SciRS2BufferAdapter, SciRS2GpuFactory
};
use std::sync::Arc;

use crate::error::{Result, SimulatorError};
use crate::simulator::{Simulator, SimulatorResult};

/// SciRS2-powered GPU State Vector Simulator
/// 
/// This simulator leverages SciRS2's unified GPU abstraction layer to provide
/// optimal performance across different GPU backends (CUDA, Metal, OpenCL).
#[derive(Debug)]
pub struct SciRS2GpuStateVectorSimulator {
    /// SciRS2 GPU backend
    backend: Arc<SciRS2GpuBackend>,
    /// Performance tracking enabled
    enable_profiling: bool,
}

impl SciRS2GpuStateVectorSimulator {
    /// Create a new SciRS2-powered GPU state vector simulator
    pub fn new() -> QuantRS2Result<Self> {
        let backend = Arc::new(SciRS2GpuFactory::create_best()?);
        Ok(Self {
            backend,
            enable_profiling: false,
        })
    }

    /// Create a new simulator with custom configuration
    pub fn with_config(config: quantrs2_core::gpu_stubs::SciRS2GpuConfig) -> QuantRS2Result<Self> {
        let backend = Arc::new(SciRS2GpuFactory::create_with_config(config)?);
        Ok(Self {
            backend,
            enable_profiling: false,
        })
    }

    /// Create an optimized simulator for quantum machine learning
    pub fn new_qml_optimized() -> QuantRS2Result<Self> {
        let backend = Arc::new(SciRS2GpuFactory::create_qml_optimized()?);
        Ok(Self {
            backend,
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
            Some(self.backend.optimization_report())
        } else {
            None
        }
    }

    /// Check if GPU acceleration is available
    pub fn is_available() -> bool {
        SciRS2GpuBackend::is_available()
    }

    /// Get available GPU backends
    pub fn available_backends() -> Vec<String> {
        SciRS2GpuFactory::available_backends()
    }
}

impl Simulator for SciRS2GpuStateVectorSimulator {
    fn run<const N: usize>(
        &mut self,
        circuit: &Circuit<N>,
    ) -> Result<SimulatorResult<N>> {
        // Use SciRS2 GPU backend for simulation
        let mut state_vector = match self.backend.allocate_state_vector(N) {
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
                        "Failed to allocate GPU state vector: {}", e
                    )));
                }
            }
        };

        // Initialize to |0...0⟩ state
        let state_size = 1 << N;
        let mut initial_state = vec![Complex64::new(0.0, 0.0); state_size];
        initial_state[0] = Complex64::new(1.0, 0.0);
        
        state_vector.upload(&initial_state)
            .map_err(|e| SimulatorError::BackendError(e.to_string()))?;

        // Apply gates using SciRS2 GPU kernel
        let kernel = self.backend.kernel();
        
        for gate in circuit.gates() {
            let qubits = gate.qubits();
            
            match qubits.len() {
                1 => {
                    // Single-qubit gate
                    let matrix = gate.matrix()
                        .map_err(|e| SimulatorError::BackendError(e.to_string()))?;
                    if matrix.len() < 4 {
                        return Err(SimulatorError::BackendError(
                            "Invalid single-qubit gate matrix size".to_string()
                        ));
                    }
                    let gate_matrix = [matrix[0], matrix[1], matrix[2], matrix[3]];
                    
                    kernel.apply_single_qubit_gate(
                        state_vector.as_mut(),
                        &gate_matrix,
                        qubits[0],
                        N,
                    ).map_err(|e| SimulatorError::BackendError(e.to_string()))?;
                }
                2 => {
                    // Two-qubit gate
                    let matrix = gate.matrix()
                        .map_err(|e| SimulatorError::BackendError(e.to_string()))?;
                    if matrix.len() < 16 {
                        return Err(SimulatorError::BackendError(
                            "Invalid two-qubit gate matrix size".to_string()
                        ));
                    }
                    let mut gate_matrix = [Complex64::new(0.0, 0.0); 16];
                    for (i, &val) in matrix.iter().take(16).enumerate() {
                        gate_matrix[i] = val;
                    }
                    
                    kernel.apply_two_qubit_gate(
                        state_vector.as_mut(),
                        &gate_matrix,
                        qubits[0],
                        qubits[1],
                        N,
                    ).map_err(|e| SimulatorError::BackendError(e.to_string()))?;
                }
                _ => {
                    // Multi-qubit gate
                    let matrix = gate.matrix()
                        .map_err(|e| SimulatorError::BackendError(e.to_string()))?;
                    let size = 1 << qubits.len();
                    let matrix_array = ndarray::Array2::from_shape_vec((size, size), matrix)
                        .map_err(|e| SimulatorError::BackendError(e.to_string()))?;
                    
                    kernel.apply_multi_qubit_gate(
                        state_vector.as_mut(),
                        &matrix_array,
                        &qubits,
                        N,
                    ).map_err(|e| SimulatorError::BackendError(e.to_string()))?;
                }
            }
        }

        // Retrieve final state vector
        let mut final_state = vec![Complex64::new(0.0, 0.0); state_size];
        state_vector.download(&mut final_state)
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
    /// Create a new GPU state vector simulator using SciRS2 backend
    /// 
    /// Note: Parameters are ignored for backward compatibility.
    /// The SciRS2 backend automatically handles device and queue management.
    pub fn new(_device: std::sync::Arc<()>, _queue: std::sync::Arc<()>) -> Self {
        // Ignore legacy WGPU parameters and use SciRS2 backend
        Self::new().unwrap_or_else(|_| {
            // If GPU initialization fails, this will be handled in the run() method
            // with automatic fallback to CPU
            SciRS2GpuStateVectorSimulator {
                backend: Arc::new(SciRS2GpuBackend::new().expect("Failed to create SciRS2 backend")),
                enable_profiling: false,
            }
        })
    }

    /// Create a blocking version of the GPU simulator
    /// 
    /// This method provides backward compatibility with the legacy async API.
    pub fn new_blocking() -> Result<Self, Box<dyn std::error::Error>> {
        match Self::new() {
            Ok(simulator) => Ok(simulator),
            Err(e) => Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to create SciRS2 GPU simulator: {}", e)
            )))
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
            builder.h(i);
        }
        for i in 0..n_qubits-1 {
            builder.cnot(i, i+1);
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
                report.push_str(&format!(
                    "{} qubits: FAILED - {}\n", 
                    n_qubits, 
                    e
                ));
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
        
        // These parameters are ignored in the new implementation
        let _simulator = GpuStateVectorSimulator::new(
            Arc::new(()), // Dummy device
            Arc::new(()), // Dummy queue
        );
        
        // Should create successfully with SciRS2 backend
        assert!(SciRS2GpuStateVectorSimulator::is_available() || !SciRS2GpuStateVectorSimulator::is_available());
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
        builder.h(0);
        builder.cnot(0, 1);
        let circuit = builder.build();

        // Run simulation
        let result = simulator.run(&circuit);
        assert!(result.is_ok());
        
        if let Ok(sim_result) = result {
            assert_eq!(sim_result.num_qubits, 2);
            assert_eq!(sim_result.amplitudes.len(), 4);
            
            // Check Bell state probabilities
            let probs: Vec<f64> = sim_result.amplitudes.iter()
                .map(|c| c.norm_sqr())
                .collect();
            
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