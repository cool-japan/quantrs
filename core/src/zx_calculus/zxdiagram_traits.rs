//! # ZXDiagram - Trait Implementations
//!
//! This module contains trait implementations for `ZXDiagram`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::{single::*, GateOp},
    qubit::QubitId,
};

use super::types::ZXDiagram;

impl Default for ZXDiagram {
    fn default() -> Self {
        Self::new()
    }
}
