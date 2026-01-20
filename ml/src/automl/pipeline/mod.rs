//! Pipeline Module
//!
//! This module contains pipeline construction and management functionality.

pub mod constructor;
pub mod quantum_ml_pipeline;

pub use constructor::AutomatedPipelineConstructor;
pub use quantum_ml_pipeline::QuantumMLPipeline;

use crate::error::Result;
