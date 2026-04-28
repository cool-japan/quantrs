//! Wire/qubit mapping types for the PennyLane JSON protocol.
//!
//! PennyLane represents qubits as "wires" (integers or strings).
//! This module provides the mapping between PennyLane wires and
//! QuantRS2 `QubitId` values.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A PennyLane wire identifier (integer wire index in the device JSON).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WireId(pub usize);

impl WireId {
    /// The wire index as a `usize`.
    pub const fn index(self) -> usize {
        self.0
    }
}

impl std::fmt::Display for WireId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Bidirectional mapping between PennyLane wire labels and qubit indices.
///
/// PennyLane uses zero-based integer wires by default.  When a circuit
/// uses wires `[0, 1, 2, ...]` (dense mapping) the `WireMap` is trivial.
/// Non-contiguous or string-labelled wires must be remapped to a contiguous
/// qubit range before simulation.
#[derive(Debug, Clone, Default)]
pub struct WireMap {
    wire_to_qubit: HashMap<usize, u32>,
    qubit_to_wire: HashMap<u32, usize>,
}

impl WireMap {
    /// Build a wire map from a sorted list of unique wire indices.
    ///
    /// Wire `wires[i]` maps to qubit `i`.
    pub fn from_wires(wires: &[usize]) -> Self {
        let mut wire_to_qubit = HashMap::with_capacity(wires.len());
        let mut qubit_to_wire = HashMap::with_capacity(wires.len());
        for (qubit_idx, &wire) in wires.iter().enumerate() {
            wire_to_qubit.insert(wire, qubit_idx as u32);
            qubit_to_wire.insert(qubit_idx as u32, wire);
        }
        Self {
            wire_to_qubit,
            qubit_to_wire,
        }
    }

    /// Map a PennyLane wire index to a QuantRS2 `QubitId`.
    pub fn wire_to_qubit(&self, wire: usize) -> Option<quantrs2_core::qubit::QubitId> {
        self.wire_to_qubit
            .get(&wire)
            .copied()
            .map(quantrs2_core::qubit::QubitId)
    }

    /// Map a QuantRS2 qubit index back to a PennyLane wire index.
    pub fn qubit_to_wire(&self, qubit: u32) -> Option<usize> {
        self.qubit_to_wire.get(&qubit).copied()
    }

    /// Total number of wires (= total number of qubits).
    pub fn num_wires(&self) -> usize {
        self.wire_to_qubit.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_wire_map() {
        let map = WireMap::from_wires(&[0, 1, 2]);
        assert_eq!(map.num_wires(), 3);
        assert_eq!(map.wire_to_qubit(0).map(|q| q.0), Some(0));
        assert_eq!(map.wire_to_qubit(2).map(|q| q.0), Some(2));
        assert_eq!(map.qubit_to_wire(1), Some(1));
    }

    #[test]
    fn test_sparse_wire_map() {
        let map = WireMap::from_wires(&[3, 7, 12]);
        // wire 3 → qubit 0, wire 7 → qubit 1, wire 12 → qubit 2
        assert_eq!(map.wire_to_qubit(3).map(|q| q.0), Some(0));
        assert_eq!(map.wire_to_qubit(7).map(|q| q.0), Some(1));
        assert_eq!(map.wire_to_qubit(12).map(|q| q.0), Some(2));
        assert_eq!(map.qubit_to_wire(0), Some(3));
    }

    #[test]
    fn test_missing_wire() {
        let map = WireMap::from_wires(&[0, 1]);
        assert!(map.wire_to_qubit(99).is_none());
    }
}
