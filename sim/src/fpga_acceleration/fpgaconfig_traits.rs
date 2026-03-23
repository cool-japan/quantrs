//! # FPGAConfig - Trait Implementations
//!
//! This module contains trait implementations for `FPGAConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{FPGAConfig, FPGAPlatform, HDLTarget};

impl Default for FPGAConfig {
    fn default() -> Self {
        Self {
            platform: FPGAPlatform::IntelStratix10,
            clock_frequency: 300.0,
            num_processing_units: 16,
            memory_bandwidth: 50.0,
            enable_pipelining: true,
            pipeline_depth: 8,
            data_path_width: 512,
            enable_dsp_optimization: true,
            enable_bram_optimization: true,
            max_state_size: 1 << 22,
            enable_realtime: false,
            hdl_target: HDLTarget::SystemVerilog,
        }
    }
}
