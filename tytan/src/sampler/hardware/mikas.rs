//! MIKAS Sampler Implementation

use scirs2_core::ndarray::{Array, Ix2};
use std::collections::HashMap;

use super::super::gpu::ArminSampler;
use super::super::{SampleResult, Sampler, SamplerError, SamplerResult};

/// MIKAS Sampler (Advanced GPU-accelerated version)
///
/// This sampler is an enhanced version of the ArminSampler with additional
/// optimization strategies and advanced features for quantum annealing.
#[cfg(feature = "gpu")]
pub struct MIKASAmpler {
    /// Underlying Armin sampler
    armin_sampler: ArminSampler,
    /// Enhanced optimization mode
    optimization_mode: String,
}

#[cfg(feature = "gpu")]
impl MIKASAmpler {
    /// Create a new MIKAS sampler
    ///
    /// # Arguments
    ///
    /// * `seed` - An optional random seed for reproducibility
    #[must_use]
    pub fn new(seed: Option<u64>) -> Self {
        Self {
            armin_sampler: ArminSampler::new(seed),
            optimization_mode: "advanced".to_string(),
        }
    }

    /// Create a new MIKAS sampler with custom parameters
    ///
    /// # Arguments
    ///
    /// * `seed` - An optional random seed for reproducibility
    /// * `mode` - Whether to use GPU ("GPU") or CPU ("CPU")
    /// * `device` - Device to use (e.g., "cuda:0")
    /// * `verbose` - Whether to show verbose output
    #[must_use]
    pub fn with_params(seed: Option<u64>, mode: &str, device: &str, verbose: bool) -> Self {
        Self {
            armin_sampler: ArminSampler::with_params(seed, mode, device, verbose),
            optimization_mode: "advanced".to_string(),
        }
    }

    /// Set optimization mode
    pub fn with_optimization_mode(mut self, mode: &str) -> Self {
        self.optimization_mode = mode.to_string();
        self
    }
}

#[cfg(feature = "gpu")]
impl Sampler for MIKASAmpler {
    fn run_qubo(
        &self,
        qubo: &(Array<f64, Ix2>, HashMap<String, usize>),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        // For now, delegate to the underlying ArminSampler
        // In a full implementation, this would include additional
        // optimization strategies specific to MIKAS
        self.armin_sampler.run_qubo(qubo, shots)
    }

    fn run_hobo(
        &self,
        hobo: &(
            Array<f64, scirs2_core::ndarray::IxDyn>,
            HashMap<String, usize>,
        ),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        // For now, delegate to the underlying ArminSampler
        // In a full implementation, this would include specialized
        // HOBO handling and optimization strategies
        self.armin_sampler.run_hobo(hobo, shots)
    }
}

// Fallback implementation when GPU feature is not enabled
#[cfg(not(feature = "gpu"))]
pub struct MIKASAmpler {
    /// Fallback sampler
    fallback_sampler: ArminSampler,
}

#[cfg(not(feature = "gpu"))]
impl MIKASAmpler {
    #[must_use]
    pub fn new(_seed: Option<u64>) -> Self {
        Self {
            fallback_sampler: ArminSampler::new(None),
        }
    }

    #[must_use]
    pub fn with_params(_seed: Option<u64>, _mode: &str, _device: &str, _verbose: bool) -> Self {
        Self {
            fallback_sampler: ArminSampler::new(None),
        }
    }
}

#[cfg(not(feature = "gpu"))]
impl Sampler for MIKASAmpler {
    fn run_qubo(
        &self,
        _qubo: &(Array<f64, Ix2>, HashMap<String, usize>),
        _shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        Err(SamplerError::GpuError(
            "GPU support not enabled".to_string(),
        ))
    }

    fn run_hobo(
        &self,
        _hobo: &(
            Array<f64, scirs2_core::ndarray::IxDyn>,
            HashMap<String, usize>,
        ),
        _shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        Err(SamplerError::GpuError(
            "GPU support not enabled".to_string(),
        ))
    }
}
