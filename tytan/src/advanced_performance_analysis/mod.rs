//! Advanced Performance Analysis and Benchmarking
//!
//! This module provides comprehensive performance analysis tools for quantum
//! annealing systems, including real-time monitoring, comparative analysis,
//! and performance prediction capabilities.
//!
//! Key Features:
//! - Real-time performance monitoring
//! - Comprehensive benchmarking suite
//! - Statistical analysis and trend detection
//! - Performance prediction models
//! - Bottleneck identification and optimization recommendations
//! - Comparative analysis and regression testing
//! - Report generation and visualization

// Re-export all public types
pub use analysis::*;
pub use benchmarking::*;
pub use config::*;
pub use core::*;
pub use monitoring::*;
pub use prediction::*;
pub use reporting::*;
pub use types::*;
pub use utils::*;

// Module declarations
pub mod analysis;
pub mod benchmarking;
pub mod config;
pub mod core;
pub mod monitoring;
pub mod prediction;
pub mod reporting;
pub mod types;
pub mod utils;

// Common imports for all submodules
pub use crate::sampler::{SampleResult, Sampler, SamplerError, SamplerResult};
pub use ndarray::{Array1, Array2, Array3, ArrayD};
pub use rand::prelude::*;
pub use serde::{Deserialize, Serialize};
pub use std::collections::HashMap;
pub use std::sync::{Arc, Mutex};
pub use std::time::{Duration, Instant};
