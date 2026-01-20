//! # NetworkResourceAllocation - Trait Implementations
//!
//! This module contains trait implementations for `NetworkResourceAllocation`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::traits::ProviderOptimizer;
use super::types::*;
use crate::prelude::CloudProvider;
use crate::DeviceResult;
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

impl Default for NetworkResourceAllocation {
    fn default() -> Self {
        Self {
            bandwidth_requirements: BandwidthRequirements {
                min_bandwidth_mbps: 100.0,
                burst_bandwidth_mbps: None,
                data_transfer_gb: 10.0,
            },
            latency_requirements: NetworkLatencyRequirements {
                max_latency_ms: 100.0,
                jitter_tolerance_ms: 10.0,
                packet_loss_tolerance: 0.01,
            },
            security_requirements: NetworkSecurityRequirements {
                vpn_required: false,
                private_network: false,
                traffic_encryption: true,
                firewall_requirements: Vec::new(),
            },
        }
    }
}
