//! # SuperconductingCollector - Trait Implementations
//!
//! This module contains trait implementations for `SuperconductingCollector`.
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
use super::types::{MetricMeasurement, MetricType, MetricValue, SuperconductingCollector};

impl MetricCollector for SuperconductingCollector {
    fn collect_metrics(&self) -> QuantRS2Result<Vec<MetricMeasurement>> {
        let mut metrics = Vec::new();
        metrics.push(MetricMeasurement {
            metric_type: MetricType::GateErrorRate,
            value: MetricValue::Float(0.001),
            timestamp: SystemTime::now(),
            qubit: Some(QubitId::new(0)),
            gate_type: Some(NativeGateType::CNOT),
            metadata: HashMap::new(),
            uncertainty: Some(0.0001),
        });
        metrics.push(MetricMeasurement {
            metric_type: MetricType::QubitCoherenceTime,
            value: MetricValue::Duration(Duration::from_micros(100)),
            timestamp: SystemTime::now(),
            qubit: Some(QubitId::new(0)),
            gate_type: None,
            metadata: HashMap::new(),
            uncertainty: Some(0.01),
        });
        Ok(metrics)
    }
    fn supported_metrics(&self) -> HashSet<MetricType> {
        let mut metrics = HashSet::new();
        metrics.insert(MetricType::GateErrorRate);
        metrics.insert(MetricType::QubitCoherenceTime);
        metrics.insert(MetricType::QubitReadoutError);
        metrics.insert(MetricType::QubitTemperature);
        metrics
    }
    fn platform(&self) -> HardwarePlatform {
        HardwarePlatform::Superconducting
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
