//! # MetricType - Trait Implementations
//!
//! This module contains trait implementations for `MetricType`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::MetricType;
use std::fmt;

impl fmt::Display for MetricType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GateErrorRate => write!(f, "Gate Error Rate"),
            Self::QubitCoherenceTime => write!(f, "Qubit Coherence Time"),
            Self::SystemUptime => write!(f, "System Uptime"),
            Self::Custom(name) => write!(f, "Custom: {name}"),
            _ => write!(f, "{self:?}"),
        }
    }
}
