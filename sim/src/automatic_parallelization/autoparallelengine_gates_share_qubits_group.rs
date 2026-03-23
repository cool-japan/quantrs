//! # AutoParallelEngine - gates_share_qubits_group Methods
//!
//! This module contains method implementations for `AutoParallelEngine`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use quantrs2_core::qubit::QubitId;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use super::autoparallelengine_type::AutoParallelEngine;

impl AutoParallelEngine {
    /// Check if gates share qubits
    pub(super) fn gates_share_qubits(
        qubits1: &HashSet<QubitId>,
        qubits2: &HashSet<QubitId>,
    ) -> bool {
        !qubits1.is_disjoint(qubits2)
    }
}
