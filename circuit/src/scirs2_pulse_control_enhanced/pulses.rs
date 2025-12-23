//! Pulse library and shape definitions

use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Pulse library with predefined shapes
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PulseLibrary {
    pub gaussian: GaussianPulse,
    pub drag: DRAGPulse,
    pub cosine: CosinePulse,
    pub erf: ErfPulse,
    pub sech: SechPulse,
    pub custom_shapes: HashMap<String, CustomPulseShape>,
}

/// Gaussian pulse parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianPulse {
    pub sigma: f64,
    pub truncation: f64,
}

impl Default for GaussianPulse {
    fn default() -> Self {
        Self {
            sigma: 10e-9,
            truncation: 4.0,
        }
    }
}

/// DRAG pulse parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DRAGPulse {
    pub gaussian_params: GaussianPulse,
    pub beta: f64,
    pub anharmonicity: f64,
}

impl Default for DRAGPulse {
    fn default() -> Self {
        Self {
            gaussian_params: GaussianPulse::default(),
            beta: 0.1,
            anharmonicity: -300e6,
        }
    }
}

/// Cosine pulse parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CosinePulse {
    pub rise_time_fraction: f64,
}

impl Default for CosinePulse {
    fn default() -> Self {
        Self {
            rise_time_fraction: 0.1,
        }
    }
}

/// Error function pulse parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErfPulse {
    pub rise_time: f64,
    pub fall_time: f64,
}

impl Default for ErfPulse {
    fn default() -> Self {
        Self {
            rise_time: 2e-9,
            fall_time: 2e-9,
        }
    }
}

/// Hyperbolic secant pulse parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SechPulse {
    pub bandwidth: f64,
    pub truncation: f64,
}

impl Default for SechPulse {
    fn default() -> Self {
        Self {
            bandwidth: 100e6,
            truncation: 4.0,
        }
    }
}

/// Custom pulse shape
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomPulseShape {
    pub name: String,
    pub samples: Vec<Complex64>,
    pub parametric_form: Option<String>,
}
