//! Stim circuit sampler for efficient batch sampling
//!
//! This module provides optimized sampling capabilities:
//! - `compile_sampler()` - Compile a circuit for repeated sampling
//! - `sample()` / `sample_batch()` - Efficient sampling methods
//! - Bit-packed output formats
//!
//! ## Example
//!
//! ```ignore
//! let circuit = StimCircuit::from_str("H 0\nCNOT 0 1\nM 0 1").unwrap();
//! let sampler = DetectorSampler::compile(&circuit);
//! let samples = sampler.sample_batch(1000);
//! ```

use crate::error::{Result, SimulatorError};
use crate::stim_dem::DetectorErrorModel;
use crate::stim_executor::{ExecutionResult, StimExecutor};
use crate::stim_parser::StimCircuit;
use scirs2_core::random::prelude::*;

/// Compiled circuit for efficient sampling
#[derive(Debug, Clone)]
pub struct CompiledStimCircuit {
    /// Original circuit
    circuit: StimCircuit,
    /// Number of qubits
    pub num_qubits: usize,
    /// Number of measurements
    pub num_measurements: usize,
    /// Number of detectors
    pub num_detectors: usize,
    /// Number of observables
    pub num_observables: usize,
    /// Pre-computed DEM for fast sampling (optional)
    dem: Option<DetectorErrorModel>,
}

impl CompiledStimCircuit {
    /// Compile a Stim circuit for sampling
    pub fn compile(circuit: &StimCircuit) -> Result<Self> {
        // Run once to determine counts
        let mut executor = StimExecutor::from_circuit(circuit);
        let result = executor.execute(circuit)?;

        Ok(Self {
            circuit: circuit.clone(),
            num_qubits: circuit.num_qubits,
            num_measurements: result.num_measurements,
            num_detectors: result.num_detectors,
            num_observables: result.num_observables,
            dem: None,
        })
    }

    /// Compile with DEM for faster error-only sampling
    pub fn compile_with_dem(circuit: &StimCircuit) -> Result<Self> {
        let mut compiled = Self::compile(circuit)?;
        compiled.dem = Some(DetectorErrorModel::from_circuit(circuit)?);
        Ok(compiled)
    }

    /// Get the underlying circuit
    #[must_use]
    pub fn circuit(&self) -> &StimCircuit {
        &self.circuit
    }

    /// Check if DEM is available
    #[must_use]
    pub fn has_dem(&self) -> bool {
        self.dem.is_some()
    }
}

/// Detector sampler for efficient batch sampling
#[derive(Debug)]
pub struct DetectorSampler {
    /// Compiled circuit
    compiled: CompiledStimCircuit,
}

impl DetectorSampler {
    /// Create a new detector sampler from a compiled circuit
    #[must_use]
    pub fn new(compiled: CompiledStimCircuit) -> Self {
        Self { compiled }
    }

    /// Compile and create a sampler from a circuit
    pub fn compile(circuit: &StimCircuit) -> Result<Self> {
        Ok(Self::new(CompiledStimCircuit::compile(circuit)?))
    }

    /// Compile with DEM for faster sampling
    pub fn compile_with_dem(circuit: &StimCircuit) -> Result<Self> {
        Ok(Self::new(CompiledStimCircuit::compile_with_dem(circuit)?))
    }

    /// Sample once, returning full execution result
    pub fn sample(&self) -> Result<ExecutionResult> {
        let mut executor = StimExecutor::from_circuit(&self.compiled.circuit);
        executor.execute(&self.compiled.circuit)
    }

    /// Sample once, returning only detector values
    pub fn sample_detectors(&self) -> Result<Vec<bool>> {
        let result = self.sample()?;
        Ok(result.detector_values)
    }

    /// Sample once, returning only measurement values
    pub fn sample_measurements(&self) -> Result<Vec<bool>> {
        let result = self.sample()?;
        Ok(result.measurement_record)
    }

    /// Sample batch with full results
    pub fn sample_batch(&self, num_shots: usize) -> Result<Vec<ExecutionResult>> {
        (0..num_shots).map(|_| self.sample()).collect()
    }

    /// Sample batch, returning only detector values
    pub fn sample_batch_detectors(&self, num_shots: usize) -> Result<Vec<Vec<bool>>> {
        (0..num_shots).map(|_| self.sample_detectors()).collect()
    }

    /// Sample batch, returning bit-packed detector values
    pub fn sample_batch_detectors_packed(&self, num_shots: usize) -> Result<Vec<Vec<u8>>> {
        let samples = self.sample_batch_detectors(num_shots)?;
        Ok(samples.into_iter().map(|s| pack_bits(&s)).collect())
    }

    /// Sample batch, returning bit-packed measurement values
    pub fn sample_batch_measurements_packed(&self, num_shots: usize) -> Result<Vec<Vec<u8>>> {
        let samples: Vec<Vec<bool>> = (0..num_shots)
            .map(|_| self.sample_measurements())
            .collect::<Result<Vec<_>>>()?;
        Ok(samples.into_iter().map(|s| pack_bits(&s)).collect())
    }

    /// Get statistics from samples
    pub fn sample_statistics(&self, num_shots: usize) -> Result<SampleStatistics> {
        let samples = self.sample_batch(num_shots)?;

        let mut detector_fire_counts = vec![0usize; self.compiled.num_detectors];
        let mut measurement_one_counts = vec![0usize; self.compiled.num_measurements];
        let mut total_detector_fires = 0;

        for result in &samples {
            for (i, &val) in result.detector_values.iter().enumerate() {
                if val {
                    detector_fire_counts[i] += 1;
                    total_detector_fires += 1;
                }
            }
            for (i, &val) in result.measurement_record.iter().enumerate() {
                if val {
                    measurement_one_counts[i] += 1;
                }
            }
        }

        Ok(SampleStatistics {
            num_shots,
            num_detectors: self.compiled.num_detectors,
            num_measurements: self.compiled.num_measurements,
            detector_fire_counts,
            measurement_one_counts,
            total_detector_fires,
            logical_error_rate: 0.0, // Would need observable tracking
        })
    }

    /// Get number of detectors
    #[must_use]
    pub fn num_detectors(&self) -> usize {
        self.compiled.num_detectors
    }

    /// Get number of measurements
    #[must_use]
    pub fn num_measurements(&self) -> usize {
        self.compiled.num_measurements
    }

    /// Get number of qubits
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.compiled.num_qubits
    }
}

/// Statistics from batch sampling
#[derive(Debug, Clone)]
pub struct SampleStatistics {
    /// Number of shots taken
    pub num_shots: usize,
    /// Number of detectors
    pub num_detectors: usize,
    /// Number of measurements per shot
    pub num_measurements: usize,
    /// Number of times each detector fired
    pub detector_fire_counts: Vec<usize>,
    /// Number of times each measurement was 1
    pub measurement_one_counts: Vec<usize>,
    /// Total number of detector fires across all shots
    pub total_detector_fires: usize,
    /// Estimated logical error rate (if observables tracked)
    pub logical_error_rate: f64,
}

impl SampleStatistics {
    /// Get the fire rate for a specific detector
    #[must_use]
    pub fn detector_fire_rate(&self, detector_idx: usize) -> f64 {
        if detector_idx < self.detector_fire_counts.len() && self.num_shots > 0 {
            self.detector_fire_counts[detector_idx] as f64 / self.num_shots as f64
        } else {
            0.0
        }
    }

    /// Get the average number of detector fires per shot
    #[must_use]
    pub fn average_detector_fires(&self) -> f64 {
        if self.num_shots > 0 {
            self.total_detector_fires as f64 / self.num_shots as f64
        } else {
            0.0
        }
    }

    /// Get the probability of any detector firing
    #[must_use]
    pub fn any_detector_fire_rate(&self) -> f64 {
        let shots_with_fire = self.detector_fire_counts.iter().filter(|&&c| c > 0).count();
        if self.num_shots > 0 {
            shots_with_fire as f64 / self.num_shots as f64
        } else {
            0.0
        }
    }
}

/// Pack boolean values into bytes (LSB first)
fn pack_bits(bits: &[bool]) -> Vec<u8> {
    bits.chunks(8)
        .map(|chunk| {
            let mut byte = 0u8;
            for (i, &bit) in chunk.iter().enumerate() {
                if bit {
                    byte |= 1 << i;
                }
            }
            byte
        })
        .collect()
}

/// Unpack bytes into boolean values
fn unpack_bits(bytes: &[u8], num_bits: usize) -> Vec<bool> {
    let mut bits = Vec::with_capacity(num_bits);
    for (byte_idx, &byte) in bytes.iter().enumerate() {
        for bit_idx in 0..8 {
            if byte_idx * 8 + bit_idx >= num_bits {
                break;
            }
            bits.push((byte >> bit_idx) & 1 == 1);
        }
    }
    bits
}

/// Compile a circuit for sampling (convenience function)
pub fn compile_sampler(circuit: &StimCircuit) -> Result<DetectorSampler> {
    DetectorSampler::compile(circuit)
}

/// Compile a circuit with DEM for faster sampling (convenience function)
pub fn compile_sampler_with_dem(circuit: &StimCircuit) -> Result<DetectorSampler> {
    DetectorSampler::compile_with_dem(circuit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_sampler() {
        let circuit_str = r#"
            H 0
            CNOT 0 1
            M 0 1
            DETECTOR rec[-1] rec[-2]
        "#;

        let circuit = StimCircuit::from_str(circuit_str).unwrap();
        let sampler = compile_sampler(&circuit).unwrap();

        assert_eq!(sampler.num_qubits(), 2);
        assert_eq!(sampler.num_measurements(), 2);
        assert_eq!(sampler.num_detectors(), 1);
    }

    #[test]
    fn test_sample_basic() {
        let circuit_str = r#"
            H 0
            CNOT 0 1
            M 0 1
        "#;

        let circuit = StimCircuit::from_str(circuit_str).unwrap();
        let sampler = compile_sampler(&circuit).unwrap();

        let result = sampler.sample().unwrap();
        assert_eq!(result.measurement_record.len(), 2);
        // Bell state: measurements should be correlated
        assert_eq!(result.measurement_record[0], result.measurement_record[1]);
    }

    #[test]
    fn test_sample_batch() {
        let circuit_str = r#"
            M 0
        "#;

        let circuit = StimCircuit::from_str(circuit_str).unwrap();
        let sampler = compile_sampler(&circuit).unwrap();

        let results = sampler.sample_batch(10).unwrap();
        assert_eq!(results.len(), 10);
        // |0⟩ state should always give 0
        for result in &results {
            assert!(!result.measurement_record[0]);
        }
    }

    #[test]
    fn test_sample_detectors() {
        let circuit_str = r#"
            M 0 1
            DETECTOR rec[-1] rec[-2]
        "#;

        let circuit = StimCircuit::from_str(circuit_str).unwrap();
        let sampler = compile_sampler(&circuit).unwrap();

        let detectors = sampler.sample_detectors().unwrap();
        assert_eq!(detectors.len(), 1);
        assert!(!detectors[0]); // |00⟩: XOR = 0, no fire
    }

    #[test]
    fn test_sample_batch_packed() {
        let circuit_str = r#"
            M 0 1 2 3 4 5 6 7 8
            DETECTOR rec[-1] rec[-2]
        "#;

        let circuit = StimCircuit::from_str(circuit_str).unwrap();
        let sampler = compile_sampler(&circuit).unwrap();

        let packed = sampler.sample_batch_measurements_packed(5).unwrap();
        assert_eq!(packed.len(), 5);
        // 9 measurements = 2 bytes per shot
        assert_eq!(packed[0].len(), 2);
    }

    #[test]
    fn test_sample_statistics() {
        let circuit_str = r#"
            M 0
            DETECTOR rec[-1]
        "#;

        let circuit = StimCircuit::from_str(circuit_str).unwrap();
        let sampler = compile_sampler(&circuit).unwrap();

        let stats = sampler.sample_statistics(100).unwrap();
        assert_eq!(stats.num_shots, 100);
        assert_eq!(stats.num_detectors, 1);
        assert_eq!(stats.num_measurements, 1);
    }

    #[test]
    fn test_pack_unpack_bits() {
        let bits = vec![true, false, true, true, false, false, true, false, true];
        let packed = pack_bits(&bits);
        let unpacked = unpack_bits(&packed, bits.len());
        assert_eq!(bits, unpacked);
    }

    #[test]
    fn test_compile_with_dem() {
        let circuit_str = r#"
            H 0
            CNOT 0 1
            M 0 1
            DETECTOR rec[-1] rec[-2]
        "#;

        let circuit = StimCircuit::from_str(circuit_str).unwrap();
        let sampler = compile_sampler_with_dem(&circuit).unwrap();

        assert!(sampler.compiled.has_dem());
    }
}
