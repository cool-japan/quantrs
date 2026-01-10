//! Mixed-precision state vector implementations for quantum simulation.
//!
//! This module provides state vector representations that can dynamically
//! switch between different numerical precisions based on accuracy requirements
//! and performance constraints.

use scirs2_core::ndarray::Array1;
use scirs2_core::{Complex32, Complex64};

use super::config::QuantumPrecision;
use crate::error::{Result, SimulatorError};

/// Mixed-precision state vector
pub enum MixedPrecisionStateVector {
    /// Half precision state vector (using Complex32 as approximation)
    Half(Array1<Complex32>),
    /// BFloat16 precision state vector (using Complex32 as storage)
    BFloat16(Array1<Complex32>),
    /// TF32 precision state vector (stored as Complex32)
    TF32(Array1<Complex32>),
    /// Single precision state vector
    Single(Array1<Complex32>),
    /// Double precision state vector
    Double(Array1<Complex64>),
    /// Adaptive precision with multiple representations
    Adaptive {
        primary: Box<Self>,
        secondary: Option<Box<Self>>,
        precision_map: Vec<QuantumPrecision>,
    },
}

impl MixedPrecisionStateVector {
    /// Create a new state vector with the specified precision
    #[must_use]
    pub fn new(size: usize, precision: QuantumPrecision) -> Self {
        match precision {
            QuantumPrecision::Half => Self::Half(Array1::zeros(size)),
            QuantumPrecision::BFloat16 => Self::BFloat16(Array1::zeros(size)),
            QuantumPrecision::TF32 => Self::TF32(Array1::zeros(size)),
            QuantumPrecision::Single => Self::Single(Array1::zeros(size)),
            QuantumPrecision::Double => Self::Double(Array1::zeros(size)),
            QuantumPrecision::Adaptive => {
                // Start with single precision for adaptive
                let primary = Box::new(Self::Single(Array1::zeros(size)));
                Self::Adaptive {
                    primary,
                    secondary: None,
                    precision_map: vec![QuantumPrecision::Single; size],
                }
            }
        }
    }

    /// Create a computational basis state |0...0>
    #[must_use]
    pub fn computational_basis(num_qubits: usize, precision: QuantumPrecision) -> Self {
        let size = 1 << num_qubits;
        let mut state = Self::new(size, precision);

        // Set |0...0> state
        match &mut state {
            Self::Half(ref mut arr)
            | Self::BFloat16(ref mut arr)
            | Self::TF32(ref mut arr)
            | Self::Single(ref mut arr) => arr[0] = Complex32::new(1.0, 0.0),
            Self::Double(ref mut arr) => arr[0] = Complex64::new(1.0, 0.0),
            Self::Adaptive {
                ref mut primary, ..
            } => {
                **primary = Self::computational_basis(num_qubits, QuantumPrecision::Single);
            }
        }

        state
    }

    /// Get the length of the state vector
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Half(arr) | Self::BFloat16(arr) | Self::TF32(arr) | Self::Single(arr) => {
                arr.len()
            }
            Self::Double(arr) => arr.len(),
            Self::Adaptive { primary, .. } => primary.len(),
        }
    }

    /// Check if the state vector is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the current precision of the state vector
    #[must_use]
    pub const fn precision(&self) -> QuantumPrecision {
        match self {
            Self::Half(_) => QuantumPrecision::Half,
            Self::BFloat16(_) => QuantumPrecision::BFloat16,
            Self::TF32(_) => QuantumPrecision::TF32,
            Self::Single(_) => QuantumPrecision::Single,
            Self::Double(_) => QuantumPrecision::Double,
            Self::Adaptive { .. } => QuantumPrecision::Adaptive,
        }
    }

    /// Convert to a specific precision
    pub fn to_precision(&self, target_precision: QuantumPrecision) -> Result<Self> {
        if self.precision() == target_precision {
            return Ok(self.clone());
        }

        let size = self.len();
        let mut result = Self::new(size, target_precision);

        match (self, &mut result) {
            (Self::Single(src), Self::Double(dst)) => {
                for (i, &val) in src.iter().enumerate() {
                    dst[i] = Complex64::new(f64::from(val.re), f64::from(val.im));
                }
            }
            (Self::Double(src), Self::Single(dst)) => {
                for (i, &val) in src.iter().enumerate() {
                    dst[i] = Complex32::new(val.re as f32, val.im as f32);
                }
            }
            (Self::Half(src), Self::Single(dst)) => {
                dst.clone_from(src);
            }
            (Self::Single(src), Self::Half(dst)) => {
                dst.clone_from(src);
            }
            (Self::Half(src), Self::Double(dst)) => {
                for (i, &val) in src.iter().enumerate() {
                    dst[i] = Complex64::new(f64::from(val.re), f64::from(val.im));
                }
            }
            (Self::Double(src), Self::Half(dst)) => {
                for (i, &val) in src.iter().enumerate() {
                    dst[i] = Complex32::new(val.re as f32, val.im as f32);
                }
            }
            _ => {
                return Err(SimulatorError::UnsupportedOperation(
                    "Complex precision conversion not supported".to_string(),
                ));
            }
        }

        Ok(result)
    }

    /// Normalize the state vector
    pub fn normalize(&mut self) -> Result<()> {
        let norm = self.norm();
        if norm == 0.0 {
            return Err(SimulatorError::InvalidInput(
                "Cannot normalize zero vector".to_string(),
            ));
        }

        match self {
            Self::Half(arr) | Self::BFloat16(arr) | Self::TF32(arr) | Self::Single(arr) => {
                let norm_f32 = norm as f32;
                for val in arr.iter_mut() {
                    *val /= norm_f32;
                }
            }
            Self::Double(arr) => {
                for val in arr.iter_mut() {
                    *val /= norm;
                }
            }
            Self::Adaptive {
                ref mut primary, ..
            } => {
                primary.normalize()?;
            }
        }

        Ok(())
    }

    /// Calculate the L2 norm of the state vector
    #[must_use]
    pub fn norm(&self) -> f64 {
        match self {
            Self::Half(arr) | Self::BFloat16(arr) | Self::TF32(arr) | Self::Single(arr) => arr
                .iter()
                .map(|x| f64::from(x.norm_sqr()))
                .sum::<f64>()
                .sqrt(),
            Self::Double(arr) => arr
                .iter()
                .map(scirs2_core::Complex::norm_sqr)
                .sum::<f64>()
                .sqrt(),
            Self::Adaptive { primary, .. } => primary.norm(),
        }
    }

    /// Calculate probability of measuring a specific state
    pub fn probability(&self, index: usize) -> Result<f64> {
        if index >= self.len() {
            return Err(SimulatorError::InvalidInput(format!(
                "Index {} out of bounds for state vector of length {}",
                index,
                self.len()
            )));
        }

        let prob = match self {
            Self::Half(arr) | Self::BFloat16(arr) | Self::TF32(arr) | Self::Single(arr) => {
                f64::from(arr[index].norm_sqr())
            }
            Self::Double(arr) => arr[index].norm_sqr(),
            Self::Adaptive { primary, .. } => primary.probability(index)?,
        };

        Ok(prob)
    }

    /// Get amplitude at a specific index as Complex64
    pub fn amplitude(&self, index: usize) -> Result<Complex64> {
        if index >= self.len() {
            return Err(SimulatorError::InvalidInput(format!(
                "Index {} out of bounds for state vector of length {}",
                index,
                self.len()
            )));
        }

        let amplitude = match self {
            Self::Half(arr) | Self::BFloat16(arr) | Self::TF32(arr) | Self::Single(arr) => {
                let val = arr[index];
                Complex64::new(f64::from(val.re), f64::from(val.im))
            }
            Self::Double(arr) => arr[index],
            Self::Adaptive { primary, .. } => primary.amplitude(index)?,
        };

        Ok(amplitude)
    }

    /// Set amplitude at a specific index
    pub fn set_amplitude(&mut self, index: usize, amplitude: Complex64) -> Result<()> {
        if index >= self.len() {
            return Err(SimulatorError::InvalidInput(format!(
                "Index {} out of bounds for state vector of length {}",
                index,
                self.len()
            )));
        }

        match self {
            Self::Half(arr) | Self::BFloat16(arr) | Self::TF32(arr) | Self::Single(arr) => {
                arr[index] = Complex32::new(amplitude.re as f32, amplitude.im as f32);
            }
            Self::Double(arr) => {
                arr[index] = amplitude;
            }
            Self::Adaptive {
                ref mut primary, ..
            } => {
                primary.set_amplitude(index, amplitude)?;
            }
        }

        Ok(())
    }

    /// Calculate fidelity with another state vector
    pub fn fidelity(&self, other: &Self) -> Result<f64> {
        if self.len() != other.len() {
            return Err(SimulatorError::InvalidInput(
                "State vectors must have the same length for fidelity calculation".to_string(),
            ));
        }

        let mut inner_product = Complex64::new(0.0, 0.0);

        for i in 0..self.len() {
            let amp1 = self.amplitude(i)?;
            let amp2 = other.amplitude(i)?;
            inner_product += amp1.conj() * amp2;
        }

        Ok(inner_product.norm_sqr())
    }

    /// Clone the state vector to a specific precision
    pub fn clone_to_precision(&self, precision: QuantumPrecision) -> Result<Self> {
        self.to_precision(precision)
    }

    /// Estimate memory usage in bytes
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        match self {
            Self::Half(arr) | Self::BFloat16(arr) | Self::TF32(arr) | Self::Single(arr) => {
                arr.len() * std::mem::size_of::<Complex32>()
            }
            Self::Double(arr) => arr.len() * std::mem::size_of::<Complex64>(),
            Self::Adaptive {
                primary, secondary, ..
            } => {
                let mut usage = primary.memory_usage();
                if let Some(sec) = secondary {
                    usage += sec.memory_usage();
                }
                usage += std::mem::size_of::<QuantumPrecision>() * primary.len(); // precision_map
                usage
            }
        }
    }

    /// Check if the state vector is normalized (within tolerance)
    #[must_use]
    pub fn is_normalized(&self, tolerance: f64) -> bool {
        (self.norm() - 1.0).abs() < tolerance
    }

    /// Get the number of qubits this state vector represents
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        (self.len() as f64).log2() as usize
    }
}

impl Clone for MixedPrecisionStateVector {
    fn clone(&self) -> Self {
        match self {
            Self::Half(arr) => Self::Half(arr.clone()),
            Self::BFloat16(arr) => Self::BFloat16(arr.clone()),
            Self::TF32(arr) => Self::TF32(arr.clone()),
            Self::Single(arr) => Self::Single(arr.clone()),
            Self::Double(arr) => Self::Double(arr.clone()),
            Self::Adaptive {
                primary,
                secondary,
                precision_map,
            } => Self::Adaptive {
                primary: primary.clone(),
                secondary: secondary.clone(),
                precision_map: precision_map.clone(),
            },
        }
    }
}

impl std::fmt::Debug for MixedPrecisionStateVector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Half(arr) => write!(f, "Half({} elements)", arr.len()),
            Self::BFloat16(arr) => write!(f, "BFloat16({} elements)", arr.len()),
            Self::TF32(arr) => write!(f, "TF32({} elements)", arr.len()),
            Self::Single(arr) => write!(f, "Single({} elements)", arr.len()),
            Self::Double(arr) => write!(f, "Double({} elements)", arr.len()),
            Self::Adaptive {
                primary, secondary, ..
            } => {
                write!(
                    f,
                    "Adaptive(primary: {primary:?}, secondary: {secondary:?})"
                )
            }
        }
    }
}
