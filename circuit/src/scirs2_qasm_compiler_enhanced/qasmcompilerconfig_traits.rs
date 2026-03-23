//! # QASMCompilerConfig - Trait Implementations
//!
//! This module contains trait implementations for `QASMCompilerConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use super::types::{QASMCompilerConfig, QASMVersion};

impl Default for QASMCompilerConfig {
    fn default() -> Self {
        Self {
            qasm_version: QASMVersion::QASM3,
            strict_mode: false,
            include_gate_definitions: true,
            default_includes: vec!["qelib1.inc".to_string()],
            custom_gates: HashMap::new(),
            hardware_constraints: None,
        }
    }
}
