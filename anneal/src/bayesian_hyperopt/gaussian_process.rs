//! Gaussian Process Surrogate Models

/// Gaussian process configuration (alias for backward compatibility)
pub type GaussianProcessConfig = GaussianProcessSurrogate;

/// Gaussian process surrogate model
#[derive(Debug, Clone)]
pub struct GaussianProcessSurrogate {
    pub kernel: KernelFunction,
    pub noise_variance: f64,
    pub mean_function: MeanFunction,
}

impl Default for GaussianProcessSurrogate {
    fn default() -> Self {
        Self {
            kernel: KernelFunction::RBF,
            noise_variance: 1e-6,
            mean_function: MeanFunction::Zero,
        }
    }
}

/// Kernel functions for Gaussian processes
#[derive(Debug, Clone, PartialEq)]
pub enum KernelFunction {
    /// Radial Basis Function (RBF) kernel
    RBF,
    /// Matern kernel
    Matern,
    /// Linear kernel
    Linear,
    /// Polynomial kernel
    Polynomial,
    /// Spectral mixture kernel
    SpectralMixture,
}

/// Mean functions for Gaussian processes
#[derive(Debug, Clone, PartialEq)]
pub enum MeanFunction {
    /// Zero mean function
    Zero,
    /// Constant mean function
    Constant(f64),
    /// Linear mean function
    Linear,
    /// Polynomial mean function
    Polynomial { degree: usize },
}

/// Gaussian process hyperparameters
#[derive(Debug, Clone)]
pub struct GPHyperparameters {
    pub length_scales: Vec<f64>,
    pub signal_variance: f64,
    pub noise_variance: f64,
    pub mean_parameters: Vec<f64>,
}

impl Default for GPHyperparameters {
    fn default() -> Self {
        Self {
            length_scales: vec![1.0],
            signal_variance: 1.0,
            noise_variance: 1e-6,
            mean_parameters: vec![0.0],
        }
    }
}
