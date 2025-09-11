//! Adaptive precision control for quantum state vectors.
//!
//! This module provides mechanisms to dynamically adjust numerical precision
//! based on the requirements of the simulation, enabling efficient memory usage
//! and computation for large quantum systems.

use crate::prelude::SimulatorError;
use half::f16;
use ndarray::Array1;
use num_complex::{Complex32, Complex64};
use std::fmt;

use crate::error::Result;

/// Precision level for state vector representation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// Half precision (16-bit)
    Half,
    /// Single precision (32-bit)
    Single,
    /// Double precision (64-bit)
    Double,
    /// Extended precision (128-bit) - future support
    Extended,
}

impl Precision {
    /// Get bytes per complex number
    pub fn bytes_per_complex(&self) -> usize {
        match self {
            Self::Half => 4,      // 2 * f16
            Self::Single => 8,    // 2 * f32
            Self::Double => 16,   // 2 * f64
            Self::Extended => 32, // 2 * f128 (future)
        }
    }

    /// Get relative epsilon for this precision
    pub fn epsilon(&self) -> f64 {
        match self {
            Self::Half => 0.001,     // ~2^-10
            Self::Single => 1e-7,    // ~2^-23
            Self::Double => 1e-15,   // ~2^-52
            Self::Extended => 1e-30, // ~2^-100
        }
    }

    /// Determine minimum precision needed for a given error tolerance
    pub fn from_tolerance(tolerance: f64) -> Self {
        if tolerance >= 0.001 {
            Self::Half
        } else if tolerance >= 1e-7 {
            Self::Single
        } else if tolerance >= 1e-15 {
            Self::Double
        } else {
            Self::Extended
        }
    }
}

/// Trait for types that can represent complex amplitudes
pub trait ComplexAmplitude: Clone + Send + Sync {
    /// Convert to Complex64 for computation
    fn to_complex64(&self) -> Complex64;

    /// Create from Complex64
    fn from_complex64(c: Complex64) -> Self;

    /// Get norm squared
    fn norm_sqr(&self) -> f64;

    /// Multiply by scalar
    fn scale(&mut self, factor: f64);
}

impl ComplexAmplitude for Complex64 {
    fn to_complex64(&self) -> Complex64 {
        *self
    }

    fn from_complex64(c: Complex64) -> Self {
        c
    }

    fn norm_sqr(&self) -> f64 {
        self.norm_sqr()
    }

    fn scale(&mut self, factor: f64) {
        *self *= factor;
    }
}

impl ComplexAmplitude for Complex32 {
    fn to_complex64(&self) -> Complex64 {
        Complex64::new(self.re as f64, self.im as f64)
    }

    fn from_complex64(c: Complex64) -> Self {
        Complex32::new(c.re as f32, c.im as f32)
    }

    fn norm_sqr(&self) -> f64 {
        (self.re * self.re + self.im * self.im) as f64
    }

    fn scale(&mut self, factor: f64) {
        *self *= factor as f32;
    }
}

/// Half-precision complex number
#[derive(Debug, Clone, Copy)]
pub struct ComplexF16 {
    pub re: f16,
    pub im: f16,
}

impl ComplexAmplitude for ComplexF16 {
    fn to_complex64(&self) -> Complex64 {
        Complex64::new(self.re.to_f64(), self.im.to_f64())
    }

    fn from_complex64(c: Complex64) -> Self {
        ComplexF16 {
            re: f16::from_f64(c.re),
            im: f16::from_f64(c.im),
        }
    }

    fn norm_sqr(&self) -> f64 {
        let r = self.re.to_f64();
        let i = self.im.to_f64();
        r * r + i * i
    }

    fn scale(&mut self, factor: f64) {
        self.re = f16::from_f64(self.re.to_f64() * factor);
        self.im = f16::from_f64(self.im.to_f64() * factor);
    }
}

/// Adaptive precision state vector
pub enum AdaptiveStateVector {
    Half(Array1<ComplexF16>),
    Single(Array1<Complex32>),
    Double(Array1<Complex64>),
}

impl AdaptiveStateVector {
    /// Create a new state vector with specified precision
    pub fn new(num_qubits: usize, precision: Precision) -> Result<Self> {
        let size = 1 << num_qubits;

        if num_qubits > 30 {
            return Err(SimulatorError::InvalidQubits(num_qubits));
        }

        match precision {
            Precision::Half => {
                let mut state = Array1::from_elem(
                    size,
                    ComplexF16 {
                        re: f16::from_f64(0.0),
                        im: f16::from_f64(0.0),
                    },
                );
                state[0] = ComplexF16 {
                    re: f16::from_f64(1.0),
                    im: f16::from_f64(0.0),
                };
                Ok(AdaptiveStateVector::Half(state))
            }
            Precision::Single => {
                let mut state = Array1::zeros(size);
                state[0] = Complex32::new(1.0, 0.0);
                Ok(AdaptiveStateVector::Single(state))
            }
            Precision::Double => {
                let mut state = Array1::zeros(size);
                state[0] = Complex64::new(1.0, 0.0);
                Ok(AdaptiveStateVector::Double(state))
            }
            Precision::Extended => Err(SimulatorError::InvalidConfiguration(
                "Extended precision not yet supported".to_string(),
            )),
        }
    }

    /// Get current precision
    pub fn precision(&self) -> Precision {
        match self {
            Self::Half(_) => Precision::Half,
            Self::Single(_) => Precision::Single,
            Self::Double(_) => Precision::Double,
        }
    }

    /// Get number of qubits
    pub fn num_qubits(&self) -> usize {
        let size = match self {
            Self::Half(v) => v.len(),
            Self::Single(v) => v.len(),
            Self::Double(v) => v.len(),
        };
        (size as f64).log2() as usize
    }

    /// Convert to double precision for computation
    pub fn to_complex64(&self) -> Array1<Complex64> {
        match self {
            Self::Half(v) => v.map(|c| c.to_complex64()),
            Self::Single(v) => v.map(|c| c.to_complex64()),
            Self::Double(v) => v.clone(),
        }
    }

    /// Update from double precision
    pub fn from_complex64(&mut self, data: &Array1<Complex64>) -> Result<()> {
        match self {
            Self::Half(v) => {
                if v.len() != data.len() {
                    return Err(SimulatorError::DimensionMismatch(format!(
                        "Size mismatch: {} vs {}",
                        v.len(),
                        data.len()
                    )));
                }
                for (i, &c) in data.iter().enumerate() {
                    v[i] = ComplexF16::from_complex64(c);
                }
            }
            Self::Single(v) => {
                if v.len() != data.len() {
                    return Err(SimulatorError::DimensionMismatch(format!(
                        "Size mismatch: {} vs {}",
                        v.len(),
                        data.len()
                    )));
                }
                for (i, &c) in data.iter().enumerate() {
                    v[i] = Complex32::from_complex64(c);
                }
            }
            Self::Double(v) => {
                v.assign(data);
            }
        }
        Ok(())
    }

    /// Check if precision upgrade is needed
    pub fn needs_precision_upgrade(&self, threshold: f64) -> bool {
        // Check if small amplitudes might be lost
        let min_amplitude = match self {
            Self::Half(v) => v
                .iter()
                .map(|c| c.norm_sqr())
                .filter(|&n| n > 0.0)
                .fold(None, |acc, x| match acc {
                    None => Some(x),
                    Some(y) => Some(if x < y { x } else { y }),
                }),
            Self::Single(v) => v
                .iter()
                .map(|c| c.norm_sqr() as f64)
                .filter(|&n| n > 0.0)
                .fold(None, |acc, x| match acc {
                    None => Some(x),
                    Some(y) => Some(if x < y { x } else { y }),
                }),
            Self::Double(v) => v
                .iter()
                .map(|c| c.norm_sqr())
                .filter(|&n| n > 0.0)
                .fold(None, |acc, x| match acc {
                    None => Some(x),
                    Some(y) => Some(if x < y { x } else { y }),
                }),
        };

        if let Some(min_amp) = min_amplitude {
            min_amp < threshold * self.precision().epsilon()
        } else {
            false
        }
    }

    /// Upgrade precision if necessary
    pub fn upgrade_precision(&mut self) -> Result<()> {
        let new_precision = match self.precision() {
            Precision::Half => Precision::Single,
            Precision::Single => Precision::Double,
            Precision::Double => return Ok(()), // Already at max
            Precision::Extended => unreachable!(),
        };

        let data = self.to_complex64();
        *self = Self::new(self.num_qubits(), new_precision)?;
        self.from_complex64(&data)?;

        Ok(())
    }

    /// Downgrade precision if possible
    pub fn downgrade_precision(&mut self, tolerance: f64) -> Result<()> {
        let new_precision = match self.precision() {
            Precision::Half => return Ok(()), // Already at min
            Precision::Single => Precision::Half,
            Precision::Double => Precision::Single,
            Precision::Extended => Precision::Double,
        };

        // Check if downgrade would lose too much precision
        let data = self.to_complex64();
        let test_vec = Self::new(self.num_qubits(), new_precision)?;

        // Compute error from downgrade
        let mut max_error: f64 = 0.0;
        match &test_vec {
            Self::Half(_) => {
                for &c in data.iter() {
                    let converted = ComplexF16::from_complex64(c).to_complex64();
                    let error = (c - converted).norm();
                    max_error = max_error.max(error);
                }
            }
            Self::Single(_) => {
                for &c in data.iter() {
                    let converted = Complex32::from_complex64(c).to_complex64();
                    let error = (c - converted).norm();
                    max_error = max_error.max(error);
                }
            }
            _ => unreachable!(),
        }

        if max_error < tolerance {
            *self = test_vec;
            self.from_complex64(&data)?;
        }

        Ok(())
    }

    /// Memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let elements = match self {
            Self::Half(v) => v.len(),
            Self::Single(v) => v.len(),
            Self::Double(v) => v.len(),
        };
        elements * self.precision().bytes_per_complex()
    }
}

/// Adaptive precision simulator config
#[derive(Debug, Clone)]
pub struct AdaptivePrecisionConfig {
    /// Initial precision
    pub initial_precision: Precision,
    /// Error tolerance for automatic precision adjustment
    pub error_tolerance: f64,
    /// Check precision every N gates
    pub check_interval: usize,
    /// Enable automatic precision upgrade
    pub auto_upgrade: bool,
    /// Enable automatic precision downgrade
    pub auto_downgrade: bool,
    /// Minimum amplitude threshold
    pub min_amplitude: f64,
}

impl Default for AdaptivePrecisionConfig {
    fn default() -> Self {
        Self {
            initial_precision: Precision::Single,
            error_tolerance: 1e-10,
            check_interval: 100,
            auto_upgrade: true,
            auto_downgrade: true,
            min_amplitude: 1e-12,
        }
    }
}

/// Track precision changes during simulation
#[derive(Debug)]
pub struct PrecisionTracker {
    /// History of precision changes
    changes: Vec<(usize, Precision, Precision)>, // (gate_count, from, to)
    /// Current gate count
    gate_count: usize,
    /// Config
    config: AdaptivePrecisionConfig,
}

impl PrecisionTracker {
    /// Create a new tracker
    pub fn new(config: AdaptivePrecisionConfig) -> Self {
        Self {
            changes: Vec::new(),
            gate_count: 0,
            config,
        }
    }

    /// Record a gate application
    pub fn record_gate(&mut self) {
        self.gate_count += 1;
    }

    /// Check if precision adjustment is needed
    pub fn should_check_precision(&self) -> bool {
        self.gate_count % self.config.check_interval == 0
    }

    /// Record precision change
    pub fn record_change(&mut self, from: Precision, to: Precision) {
        self.changes.push((self.gate_count, from, to));
    }

    /// Get precision history
    pub fn history(&self) -> &[(usize, Precision, Precision)] {
        &self.changes
    }

    /// Get statistics
    pub fn stats(&self) -> PrecisionStats {
        let mut upgrades = 0;
        let mut downgrades = 0;

        for (_, from, to) in &self.changes {
            match (from, to) {
                (Precision::Half, Precision::Single)
                | (Precision::Single, Precision::Double)
                | (Precision::Double, Precision::Extended) => upgrades += 1,
                _ => downgrades += 1,
            }
        }

        PrecisionStats {
            total_gates: self.gate_count,
            precision_changes: self.changes.len(),
            upgrades,
            downgrades,
        }
    }
}

/// Precision statistics
#[derive(Debug)]
pub struct PrecisionStats {
    pub total_gates: usize,
    pub precision_changes: usize,
    pub upgrades: usize,
    pub downgrades: usize,
}

impl fmt::Display for PrecisionStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Precision Stats: {} gates, {} changes ({} upgrades, {} downgrades)",
            self.total_gates, self.precision_changes, self.upgrades, self.downgrades
        )
    }
}

/// Benchmark different precisions
pub fn benchmark_precisions(num_qubits: usize) -> Result<()> {
    println!("\nPrecision Benchmark for {} qubits:", num_qubits);
    println!("{:-<60}", "");

    for precision in [Precision::Half, Precision::Single, Precision::Double] {
        let state = AdaptiveStateVector::new(num_qubits, precision)?;
        let memory = state.memory_usage();
        let memory_mb = memory as f64 / (1024.0 * 1024.0);

        println!(
            "{:?} precision: {:.2} MB ({} bytes per amplitude)",
            precision,
            memory_mb,
            precision.bytes_per_complex()
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_levels() {
        assert_eq!(Precision::Half.bytes_per_complex(), 4);
        assert_eq!(Precision::Single.bytes_per_complex(), 8);
        assert_eq!(Precision::Double.bytes_per_complex(), 16);
    }

    #[test]
    fn test_precision_from_tolerance() {
        assert_eq!(Precision::from_tolerance(0.01), Precision::Half);
        assert_eq!(Precision::from_tolerance(1e-8), Precision::Double); // 1e-8 < 1e-7, so Double
        assert_eq!(Precision::from_tolerance(1e-16), Precision::Extended); // 1e-16 < 1e-15, so Extended
    }

    #[test]
    fn test_complex_f16() {
        let c = ComplexF16 {
            re: f16::from_f64(0.5),
            im: f16::from_f64(0.5),
        };

        let c64 = c.to_complex64();
        assert!((c64.re - 0.5).abs() < 0.01);
        assert!((c64.im - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_adaptive_state_vector() {
        let mut state = AdaptiveStateVector::new(2, Precision::Single).unwrap();
        assert_eq!(state.precision(), Precision::Single);
        assert_eq!(state.num_qubits(), 2);

        // Test conversion
        let c64 = state.to_complex64();
        assert_eq!(c64.len(), 4);
        assert_eq!(c64[0], Complex64::new(1.0, 0.0));
    }

    #[test]
    fn test_precision_upgrade() {
        let mut state = AdaptiveStateVector::new(2, Precision::Half).unwrap();
        state.upgrade_precision().unwrap();
        assert_eq!(state.precision(), Precision::Single);
    }

    #[test]
    fn test_precision_tracker() {
        let config = AdaptivePrecisionConfig::default();
        let mut tracker = PrecisionTracker::new(config);

        // Record exactly 100 gates so gate_count % check_interval == 0
        for _ in 0..100 {
            tracker.record_gate();
        }

        assert!(tracker.should_check_precision());

        tracker.record_change(Precision::Single, Precision::Double);
        let stats = tracker.stats();
        assert_eq!(stats.upgrades, 1);
        assert_eq!(stats.downgrades, 0);
    }

    #[test]
    fn test_memory_usage() {
        let state = AdaptiveStateVector::new(10, Precision::Half).unwrap();
        let memory = state.memory_usage();
        assert_eq!(memory, 1024 * 4); // 2^10 * 4 bytes
    }
}
