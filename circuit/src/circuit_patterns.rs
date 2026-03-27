//! Composite circuit preparation patterns for [`Circuit`].
//!
//! This module provides higher-level circuit-building methods that compose
//! multiple primitive gates into well-known quantum state preparation patterns
//! and entanglement topologies, such as Bell states, GHZ states, W states,
//! CNOT ladders/rings, and SWAP/CZ ladder patterns.

use quantrs2_core::{error::QuantRS2Result, qubit::QubitId};

use crate::builder::Circuit;

impl<const N: usize> Circuit<N> {
    // ============ Quantum State Preparation Patterns ============

    /// Prepare a Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2 on two qubits
    ///
    /// # Example
    /// ```ignore
    /// let mut circuit = Circuit::<2>::new();
    /// circuit.bell_state(0, 1)?; // Prepare Bell state on qubits 0 and 1
    /// ```
    pub fn bell_state(&mut self, qubit1: u32, qubit2: u32) -> QuantRS2Result<&mut Self> {
        self.h(QubitId::new(qubit1))?;
        self.cnot(QubitId::new(qubit1), QubitId::new(qubit2))?;
        Ok(self)
    }

    /// Prepare a GHZ state (|000...⟩ + |111...⟩)/√2 on specified qubits
    ///
    /// # Example
    /// ```ignore
    /// let mut circuit = Circuit::<3>::new();
    /// circuit.ghz_state(&[0, 1, 2])?; // Prepare GHZ state on qubits 0, 1, and 2
    /// ```
    pub fn ghz_state(&mut self, qubits: &[u32]) -> QuantRS2Result<&mut Self> {
        if qubits.is_empty() {
            return Ok(self);
        }

        // Apply Hadamard to first qubit
        self.h(QubitId::new(qubits[0]))?;

        // Apply CNOT gates to entangle all qubits
        for i in 1..qubits.len() {
            self.cnot(QubitId::new(qubits[0]), QubitId::new(qubits[i]))?;
        }

        Ok(self)
    }

    /// Prepare a W state on specified qubits
    ///
    /// W state: (|100...⟩ + |010...⟩ + |001...⟩ + ...)/√n
    ///
    /// This is an approximation using rotation gates.
    pub fn w_state(&mut self, qubits: &[u32]) -> QuantRS2Result<&mut Self> {
        if qubits.is_empty() {
            return Ok(self);
        }

        let n = qubits.len() as f64;

        // For n qubits, prepare W state using controlled rotations
        self.ry(QubitId::new(qubits[0]), 2.0 * (1.0 / n.sqrt()).acos())?;

        for i in 1..qubits.len() {
            let angle = 2.0 * (1.0 / (n - i as f64).sqrt()).acos();
            self.cry(QubitId::new(qubits[i - 1]), QubitId::new(qubits[i]), angle)?;
        }

        // Apply X gates to ensure proper state preparation
        for i in 0..qubits.len() - 1 {
            self.cnot(QubitId::new(qubits[i + 1]), QubitId::new(qubits[i]))?;
        }

        Ok(self)
    }

    /// Prepare a product state |++++...⟩ by applying Hadamard to all qubits
    ///
    /// # Example
    /// ```ignore
    /// let mut circuit = Circuit::<4>::new();
    /// circuit.plus_state_all()?; // Prepare |+⟩ on all 4 qubits
    /// ```
    pub fn plus_state_all(&mut self) -> QuantRS2Result<&mut Self> {
        for i in 0..N {
            self.h(QubitId::new(i as u32))?;
        }
        Ok(self)
    }

    // ============ Entanglement Topology Patterns ============

    /// Create a ladder of CNOT gates connecting adjacent qubits
    ///
    /// # Example
    /// ```ignore
    /// let mut circuit = Circuit::<4>::new();
    /// circuit.cnot_ladder(&[0, 1, 2, 3])?; // Creates: CNOT(0,1), CNOT(1,2), CNOT(2,3)
    /// ```
    pub fn cnot_ladder(&mut self, qubits: &[u32]) -> QuantRS2Result<&mut Self> {
        if qubits.len() < 2 {
            return Ok(self);
        }

        for i in 0..qubits.len() - 1 {
            self.cnot(QubitId::new(qubits[i]), QubitId::new(qubits[i + 1]))?;
        }

        Ok(self)
    }

    /// Create a ring of CNOT gates connecting qubits in a cycle
    ///
    /// Like CNOT ladder but also connects last to first qubit.
    pub fn cnot_ring(&mut self, qubits: &[u32]) -> QuantRS2Result<&mut Self> {
        if qubits.len() < 2 {
            return Ok(self);
        }

        // Add ladder
        self.cnot_ladder(qubits)?;

        // Close the ring by connecting last to first
        let last_idx = qubits.len() - 1;
        self.cnot(QubitId::new(qubits[last_idx]), QubitId::new(qubits[0]))?;

        Ok(self)
    }

    /// Create a ladder of SWAP gates connecting adjacent qubits
    ///
    /// # Example
    /// ```ignore
    /// let mut circuit = Circuit::<4>::new();
    /// circuit.swap_ladder(&[0, 1, 2, 3])?; // Creates: SWAP(0,1), SWAP(1,2), SWAP(2,3)
    /// ```
    pub fn swap_ladder(&mut self, qubits: &[u32]) -> QuantRS2Result<&mut Self> {
        if qubits.len() < 2 {
            return Ok(self);
        }

        for i in 0..qubits.len() - 1 {
            self.swap(QubitId::new(qubits[i]), QubitId::new(qubits[i + 1]))?;
        }

        Ok(self)
    }

    /// Create a ladder of CZ gates connecting adjacent qubits
    ///
    /// # Example
    /// ```ignore
    /// let mut circuit = Circuit::<4>::new();
    /// circuit.cz_ladder(&[0, 1, 2, 3])?; // Creates: CZ(0,1), CZ(1,2), CZ(2,3)
    /// ```
    pub fn cz_ladder(&mut self, qubits: &[u32]) -> QuantRS2Result<&mut Self> {
        if qubits.len() < 2 {
            return Ok(self);
        }

        for i in 0..qubits.len() - 1 {
            self.cz(QubitId::new(qubits[i]), QubitId::new(qubits[i + 1]))?;
        }

        Ok(self)
    }
}
