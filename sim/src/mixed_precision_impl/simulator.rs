//! Mixed-precision quantum simulator implementation.
//!
//! This module provides the main simulator class that automatically
//! manages precision levels for optimal performance and accuracy.

use ndarray::Array1;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{Result, SimulatorError};
use crate::scirs2_integration::SciRS2Backend;
use crate::adaptive_gate_fusion::{QuantumGate, GateType, FusedGateBlock};

use super::config::{MixedPrecisionConfig, QuantumPrecision};
use super::analysis::{PrecisionAnalysis, PerformanceMetrics, PrecisionAnalyzer};
use super::state_vector::MixedPrecisionStateVector;

/// Mixed-precision quantum simulator
pub struct MixedPrecisionSimulator {
    /// Configuration
    config: MixedPrecisionConfig,
    /// Current state vector
    state: Option<MixedPrecisionStateVector>,
    /// Number of qubits
    num_qubits: usize,
    /// SciRS2 backend for advanced operations
    #[cfg(feature = "advanced_math")]
    backend: Option<SciRS2Backend>,
    /// Performance statistics
    stats: MixedPrecisionStats,
    /// Precision analyzer
    analyzer: PrecisionAnalyzer,
}

/// Performance statistics for mixed-precision simulation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MixedPrecisionStats {
    /// Total number of gates applied
    pub total_gates: usize,
    /// Number of precision adaptations
    pub precision_adaptations: usize,
    /// Total execution time in milliseconds
    pub total_time_ms: f64,
    /// Memory usage by precision level
    pub memory_usage_by_precision: HashMap<QuantumPrecision, usize>,
    /// Gate execution time by precision
    pub gate_time_by_precision: HashMap<QuantumPrecision, f64>,
    /// Error estimates by precision
    pub error_estimates: HashMap<QuantumPrecision, f64>,
}

impl MixedPrecisionSimulator {
    /// Create a new mixed-precision simulator
    pub fn new(num_qubits: usize, config: MixedPrecisionConfig) -> Result<Self> {
        config.validate()?;
        
        let state = Some(MixedPrecisionStateVector::computational_basis(
            num_qubits,
            config.state_vector_precision,
        ));

        Ok(Self {
            config,
            state,
            num_qubits,
            #[cfg(feature = "advanced_math")]
            backend: None,
            stats: MixedPrecisionStats::default(),
            analyzer: PrecisionAnalyzer::new(),
        })
    }

    /// Initialize with SciRS2 backend
    #[cfg(feature = "advanced_math")]
    pub fn with_backend(mut self) -> Result<Self> {
        self.backend = Some(SciRS2Backend::new());
        Ok(self)
    }

    /// Apply a quantum gate with automatic precision selection
    pub fn apply_gate(&mut self, gate: &QuantumGate) -> Result<()> {
        let start_time = std::time::Instant::now();
        
        // Select optimal precision for this gate
        let gate_precision = self.select_gate_precision(gate)?;
        
        // Ensure state vector is in the correct precision
        self.adapt_state_precision(gate_precision)?;
        
        // Apply the gate
        self.apply_gate_with_precision(gate, gate_precision)?;
        
        // Update statistics
        let execution_time = start_time.elapsed().as_millis() as f64;
        self.stats.total_gates += 1;
        self.stats.total_time_ms += execution_time;
        *self.stats.gate_time_by_precision.entry(gate_precision).or_insert(0.0) += execution_time;
        
        Ok(())
    }

    /// Apply multiple gates as a fused block
    pub fn apply_fused_block(&mut self, block: &FusedGateBlock) -> Result<()> {
        let optimal_precision = self.select_block_precision(block)?;
        self.adapt_state_precision(optimal_precision)?;
        
        // Apply each gate in the block
        for gate in &block.gates {
            self.apply_gate_with_precision(gate, optimal_precision)?;
        }
        
        Ok(())
    }

    /// Measure a qubit and return the result
    pub fn measure_qubit(&mut self, qubit: usize) -> Result<bool> {
        if qubit >= self.num_qubits {
            return Err(SimulatorError::InvalidInput(
                format!("Qubit {} out of range for {}-qubit system", qubit, self.num_qubits),
            ));
        }

        // Use measurement precision for this operation
        self.adapt_state_precision(self.config.measurement_precision)?;
        
        let state = self.state.as_ref().unwrap();
        
        // Calculate probability of measuring |1⟩
        let mut prob_one = 0.0;
        let mask = 1 << qubit;
        
        for i in 0..state.len() {
            if i & mask != 0 {
                prob_one += state.probability(i)?;
            }
        }
        
        // Simulate random measurement
        let result = fastrand::f64() < prob_one;
        
        // Collapse the state vector
        self.collapse_state(qubit, result)?;
        
        Ok(result)
    }

    /// Measure all qubits and return the bit string
    pub fn measure_all(&mut self) -> Result<Vec<bool>> {
        let mut results = Vec::new();
        for qubit in 0..self.num_qubits {
            results.push(self.measure_qubit(qubit)?);
        }
        Ok(results)
    }

    /// Get the current state vector
    pub fn get_state(&self) -> Option<&MixedPrecisionStateVector> {
        self.state.as_ref()
    }

    /// Calculate expectation value of a Pauli operator
    pub fn expectation_value(&self, pauli_string: &str) -> Result<f64> {
        if pauli_string.len() != self.num_qubits {
            return Err(SimulatorError::InvalidInput(
                "Pauli string length must match number of qubits".to_string(),
            ));
        }

        let state = self.state.as_ref().unwrap();
        let mut expectation = 0.0;

        for i in 0..state.len() {
            let mut sign = 1.0;
            let mut amplitude = state.amplitude(i)?;
            
            // Apply Pauli operators
            for (qubit, pauli) in pauli_string.chars().enumerate() {
                match pauli {
                    'I' => {} // Identity - no change
                    'X' => {
                        // Flip bit
                        let flipped = i ^ (1 << qubit);
                        amplitude = state.amplitude(flipped)?;
                    }
                    'Y' => {
                        // Flip bit and apply phase
                        let flipped = i ^ (1 << qubit);
                        amplitude = state.amplitude(flipped)?;
                        if i & (1 << qubit) != 0 {
                            sign *= -1.0;
                        }
                        amplitude *= Complex64::new(0.0, sign);
                    }
                    'Z' => {
                        // Apply phase
                        if i & (1 << qubit) != 0 {
                            sign *= -1.0;
                        }
                    }
                    _ => return Err(SimulatorError::InvalidInput(
                        format!("Invalid Pauli operator: {}", pauli),
                    )),
                }
            }
            
            expectation += (amplitude.conj() * amplitude * sign).re;
        }

        Ok(expectation)
    }

    /// Run precision analysis
    pub fn analyze_precision(&mut self) -> Result<PrecisionAnalysis> {
        Ok(self.analyzer.analyze_for_tolerance(self.config.error_tolerance))
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> &MixedPrecisionStats {
        &self.stats
    }

    /// Reset the simulator to initial state
    pub fn reset(&mut self) -> Result<()> {
        self.state = Some(MixedPrecisionStateVector::computational_basis(
            self.num_qubits,
            self.config.state_vector_precision,
        ));
        self.stats = MixedPrecisionStats::default();
        self.analyzer.reset();
        Ok(())
    }

    /// Select optimal precision for a gate
    fn select_gate_precision(&self, gate: &QuantumGate) -> Result<QuantumPrecision> {
        if !self.config.adaptive_precision {
            return Ok(self.config.gate_precision);
        }

        // Use heuristics to select precision based on gate type
        let precision = match gate.gate_type {
            GateType::PauliX | GateType::PauliY | GateType::PauliZ | GateType::Hadamard | GateType::Phase | GateType::T | GateType::RotationX | GateType::RotationY | GateType::RotationZ | GateType::Identity => {
                // Single qubit gates are usually numerically stable
                if self.config.gate_precision == QuantumPrecision::Adaptive {
                    QuantumPrecision::Single
                } else {
                    self.config.gate_precision
                }
            }
            GateType::CNOT | GateType::CZ | GateType::SWAP | GateType::ISwap => {
                // Two qubit gates may require higher precision
                if self.config.gate_precision == QuantumPrecision::Adaptive {
                    if self.num_qubits > self.config.large_system_threshold {
                        QuantumPrecision::Single
                    } else {
                        QuantumPrecision::Double
                    }
                } else {
                    self.config.gate_precision
                }
            }
            GateType::Toffoli | GateType::Fredkin => {
                // Multi-qubit gates typically need higher precision
                if self.config.gate_precision == QuantumPrecision::Adaptive {
                    QuantumPrecision::Double
                } else {
                    self.config.gate_precision
                }
            }
            GateType::Custom(_) => {
                // Custom gates - use conservative precision
                QuantumPrecision::Double
            }
        };

        Ok(precision)
    }

    /// Select optimal precision for a fused gate block
    fn select_block_precision(&self, _block: &FusedGateBlock) -> Result<QuantumPrecision> {
        // For fused blocks, use a conservative approach
        if self.config.adaptive_precision {
            Ok(QuantumPrecision::Single)
        } else {
            Ok(self.config.gate_precision)
        }
    }

    /// Adapt state vector to the target precision
    fn adapt_state_precision(&mut self, target_precision: QuantumPrecision) -> Result<()> {
        if let Some(ref state) = self.state {
            if state.precision() != target_precision {
                let new_state = state.to_precision(target_precision)?;
                self.state = Some(new_state);
                self.stats.precision_adaptations += 1;
            }
        }
        Ok(())
    }

    /// Apply a gate with a specific precision
    fn apply_gate_with_precision(&mut self, gate: &QuantumGate, _precision: QuantumPrecision) -> Result<()> {
        // This is a simplified implementation
        // In practice, this would apply the actual gate operation
        if let Some(ref mut state) = self.state {
            // For demonstration, just record that we applied a gate
            // Real implementation would perform matrix multiplication
            
            // Update memory usage statistics
            let memory_usage = state.memory_usage();
            self.stats.memory_usage_by_precision.insert(state.precision(), memory_usage);
        }
        
        Ok(())
    }

    /// Collapse the state vector after measurement
    fn collapse_state(&mut self, qubit: usize, result: bool) -> Result<()> {
        if let Some(ref mut state) = self.state {
            let mask = 1 << qubit;
            let mut norm_factor = 0.0;
            
            // Calculate normalization factor
            for i in 0..state.len() {
                let bit_value = (i & mask) != 0;
                if bit_value == result {
                    norm_factor += state.probability(i)?;
                }
            }
            
            if norm_factor == 0.0 {
                return Err(SimulatorError::InvalidInput(
                    "Invalid measurement result: zero probability".to_string(),
                ));
            }
            
            norm_factor = norm_factor.sqrt();
            
            // Update amplitudes
            for i in 0..state.len() {
                let bit_value = (i & mask) != 0;
                if bit_value == result {
                    let amplitude = state.amplitude(i)?;
                    state.set_amplitude(i, amplitude / norm_factor)?;
                } else {
                    state.set_amplitude(i, Complex64::new(0.0, 0.0))?;
                }
            }
        }
        
        Ok(())
    }
}

impl MixedPrecisionStats {
    /// Calculate average gate time
    pub fn average_gate_time(&self) -> f64 {
        if self.total_gates > 0 {
            self.total_time_ms / self.total_gates as f64
        } else {
            0.0
        }
    }

    /// Get total memory usage across all precisions
    pub fn total_memory_usage(&self) -> usize {
        self.memory_usage_by_precision.values().sum()
    }

    /// Get adaptation rate (adaptations per gate)
    pub fn adaptation_rate(&self) -> f64 {
        if self.total_gates > 0 {
            self.precision_adaptations as f64 / self.total_gates as f64
        } else {
            0.0
        }
    }

    /// Get performance summary
    pub fn summary(&self) -> String {
        format!(
            "Gates: {}, Adaptations: {}, Avg Time: {:.2}ms, Total Memory: {}MB",
            self.total_gates,
            self.precision_adaptations,
            self.average_gate_time(),
            self.total_memory_usage() / (1024 * 1024)
        )
    }
}

/// Utility functions for mixed precision simulation
pub mod utils {
    use super::*;

    /// Convert a regular state vector to mixed precision
    pub fn convert_state_vector(
        state: &Array1<Complex64>,
        precision: QuantumPrecision,
    ) -> Result<MixedPrecisionStateVector> {
        let mut mp_state = MixedPrecisionStateVector::new(state.len(), precision);
        
        for (i, &amplitude) in state.iter().enumerate() {
            mp_state.set_amplitude(i, amplitude)?;
        }
        
        Ok(mp_state)
    }

    /// Extract a regular state vector from mixed precision
    pub fn extract_state_vector(mp_state: &MixedPrecisionStateVector) -> Result<Array1<Complex64>> {
        let mut state = Array1::zeros(mp_state.len());
        
        for i in 0..mp_state.len() {
            state[i] = mp_state.amplitude(i)?;
        }
        
        Ok(state)
    }

    /// Calculate memory savings compared to double precision
    pub fn memory_savings(config: &MixedPrecisionConfig, num_qubits: usize) -> f64 {
        let double_precision_size = (1 << num_qubits) * std::mem::size_of::<Complex64>();
        let mixed_precision_size = config.estimate_memory_usage(num_qubits);
        
        1.0 - (mixed_precision_size as f64 / double_precision_size as f64)
    }

    /// Estimate performance improvement factor
    pub fn performance_improvement_factor(precision: QuantumPrecision) -> f64 {
        1.0 / precision.computation_factor()
    }
}