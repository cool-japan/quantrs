//! # TrappedIonCollector - Trait Implementations
//!
//! This module contains trait implementations for `TrappedIonCollector`.
//!
//! ## Implemented Traits
//!
//! - `MetricCollector`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    hardware_compilation::{HardwarePlatform, NativeGateType},
    qubit::QubitId,
};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt,
    sync::{Arc, RwLock},
    thread,
    time::{Duration, SystemTime},
};

use super::functions::MetricCollector;
use super::types::{MetricMeasurement, MetricType, TrappedIonCollector};

impl MetricCollector for TrappedIonCollector {
    fn collect_metrics(&self) -> QuantRS2Result<Vec<MetricMeasurement>> {
        Ok(vec![])
    }
    fn supported_metrics(&self) -> HashSet<MetricType> {
        HashSet::new()
    }
    fn platform(&self) -> HardwarePlatform {
        HardwarePlatform::TrappedIon
    }
    fn initialize(&mut self) -> QuantRS2Result<()> {
        self.connected = true;
        Ok(())
    }
    fn is_connected(&self) -> bool {
        self.connected
    }
    fn disconnect(&mut self) -> QuantRS2Result<()> {
        self.connected = false;
        Ok(())
    }
}
