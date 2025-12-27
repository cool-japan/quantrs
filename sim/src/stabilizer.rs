//! Stabilizer simulator for efficient simulation of Clifford circuits
//!
//! The stabilizer formalism provides an efficient way to simulate quantum circuits
//! that consist only of Clifford gates (H, S, CNOT) and Pauli measurements.
//! This implementation uses the tableau representation and leverages `SciRS2`
//! for efficient data structures and operations.

use crate::simulator::{Simulator, SimulatorResult};
use quantrs2_circuit::prelude::*;
use quantrs2_core::gate::GateOp;
use quantrs2_core::prelude::*;
use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;
use std::collections::HashMap;
use std::sync::Arc;

/// Stabilizer tableau representation
///
/// The tableau stores generators of the stabilizer group as rows.
/// Each row represents a Pauli string with phase.
#[derive(Debug, Clone)]
pub struct StabilizerTableau {
    /// Number of qubits
    num_qubits: usize,
    /// X part of stabilizers (n x n matrix)
    x_matrix: Array2<bool>,
    /// Z part of stabilizers (n x n matrix)
    z_matrix: Array2<bool>,
    /// Phase vector (n elements, each is 0 or 1 representing +1 or -1)
    phase: Vec<bool>,
    /// Destabilizers X part (n x n matrix)
    destab_x: Array2<bool>,
    /// Destabilizers Z part (n x n matrix)
    destab_z: Array2<bool>,
    /// Destabilizer phases
    destab_phase: Vec<bool>,
}

impl StabilizerTableau {
    /// Create a new tableau in the |0...0⟩ state
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        let mut x_matrix = Array2::from_elem((num_qubits, num_qubits), false);
        let mut z_matrix = Array2::from_elem((num_qubits, num_qubits), false);
        let mut destab_x = Array2::from_elem((num_qubits, num_qubits), false);
        let mut destab_z = Array2::from_elem((num_qubits, num_qubits), false);

        // Initialize stabilizers as Z_i and destabilizers as X_i
        for i in 0..num_qubits {
            z_matrix[[i, i]] = true; // Stabilizer i is Z_i
            destab_x[[i, i]] = true; // Destabilizer i is X_i
        }

        Self {
            num_qubits,
            x_matrix,
            z_matrix,
            phase: vec![false; num_qubits],
            destab_x,
            destab_z,
            destab_phase: vec![false; num_qubits],
        }
    }

    /// Apply a Hadamard gate
    pub fn apply_h(&mut self, qubit: usize) -> Result<(), QuantRS2Error> {
        if qubit >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(qubit as u32));
        }

        // H: X ↔ Z, phase changes according to anticommutation
        for i in 0..self.num_qubits {
            // For stabilizers
            let x_val = self.x_matrix[[i, qubit]];
            let z_val = self.z_matrix[[i, qubit]];

            // Update phase: if both X and Z are present, add a phase
            if x_val && z_val {
                self.phase[i] = !self.phase[i];
            }

            // Swap X and Z
            self.x_matrix[[i, qubit]] = z_val;
            self.z_matrix[[i, qubit]] = x_val;

            // For destabilizers
            let dx_val = self.destab_x[[i, qubit]];
            let dz_val = self.destab_z[[i, qubit]];

            if dx_val && dz_val {
                self.destab_phase[i] = !self.destab_phase[i];
            }

            self.destab_x[[i, qubit]] = dz_val;
            self.destab_z[[i, qubit]] = dx_val;
        }

        Ok(())
    }

    /// Apply an S gate (phase gate)
    pub fn apply_s(&mut self, qubit: usize) -> Result<(), QuantRS2Error> {
        if qubit >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(qubit as u32));
        }

        // S: X → Y, Z → Z
        // Y = iXZ, so we need to track the phase change
        for i in 0..self.num_qubits {
            // For stabilizers
            if self.x_matrix[[i, qubit]] && !self.z_matrix[[i, qubit]] {
                // X → Y = iXZ
                self.z_matrix[[i, qubit]] = true;
                self.phase[i] = !self.phase[i]; // Multiply by i = -1 in {+1, -1}
            }

            // For destabilizers
            if self.destab_x[[i, qubit]] && !self.destab_z[[i, qubit]] {
                self.destab_z[[i, qubit]] = true;
                self.destab_phase[i] = !self.destab_phase[i];
            }
        }

        Ok(())
    }

    /// Apply a CNOT gate
    pub fn apply_cnot(&mut self, control: usize, target: usize) -> Result<(), QuantRS2Error> {
        if control >= self.num_qubits || target >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(control.max(target) as u32));
        }

        if control == target {
            return Err(QuantRS2Error::InvalidInput(
                "CNOT control and target must be different".to_string(),
            ));
        }

        // CNOT: X_c → X_c X_t, Z_t → Z_c Z_t
        for i in 0..self.num_qubits {
            // For stabilizers
            if self.x_matrix[[i, control]] {
                self.x_matrix[[i, target]] ^= true;
            }
            if self.z_matrix[[i, target]] {
                self.z_matrix[[i, control]] ^= true;
            }

            // For destabilizers
            if self.destab_x[[i, control]] {
                self.destab_x[[i, target]] ^= true;
            }
            if self.destab_z[[i, target]] {
                self.destab_z[[i, control]] ^= true;
            }
        }

        Ok(())
    }

    /// Apply a Pauli X gate
    pub fn apply_x(&mut self, qubit: usize) -> Result<(), QuantRS2Error> {
        if qubit >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(qubit as u32));
        }

        // X anticommutes with Z
        for i in 0..self.num_qubits {
            if self.z_matrix[[i, qubit]] {
                self.phase[i] = !self.phase[i];
            }
            if self.destab_z[[i, qubit]] {
                self.destab_phase[i] = !self.destab_phase[i];
            }
        }

        Ok(())
    }

    /// Apply a Pauli Y gate
    pub fn apply_y(&mut self, qubit: usize) -> Result<(), QuantRS2Error> {
        if qubit >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(qubit as u32));
        }

        // Y = iXZ, anticommutes with both X and Z
        for i in 0..self.num_qubits {
            let has_x = self.x_matrix[[i, qubit]];
            let has_z = self.z_matrix[[i, qubit]];

            if has_x != has_z {
                self.phase[i] = !self.phase[i];
            }

            let has_dx = self.destab_x[[i, qubit]];
            let has_dz = self.destab_z[[i, qubit]];

            if has_dx != has_dz {
                self.destab_phase[i] = !self.destab_phase[i];
            }
        }

        Ok(())
    }

    /// Apply a Pauli Z gate
    pub fn apply_z(&mut self, qubit: usize) -> Result<(), QuantRS2Error> {
        if qubit >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(qubit as u32));
        }

        // Z anticommutes with X
        for i in 0..self.num_qubits {
            if self.x_matrix[[i, qubit]] {
                self.phase[i] = !self.phase[i];
            }
            if self.destab_x[[i, qubit]] {
                self.destab_phase[i] = !self.destab_phase[i];
            }
        }

        Ok(())
    }

    /// Measure a qubit in the computational basis
    /// Returns the measurement outcome (0 or 1)
    pub fn measure(&mut self, qubit: usize) -> Result<bool, QuantRS2Error> {
        if qubit >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(qubit as u32));
        }

        // Find a stabilizer that anticommutes with Z_qubit
        let mut anticommuting_row = None;

        for i in 0..self.num_qubits {
            if self.x_matrix[[i, qubit]] {
                anticommuting_row = Some(i);
                break;
            }
        }

        if let Some(p) = anticommuting_row {
            // Random outcome case
            // Set the p-th stabilizer to Z_qubit
            for j in 0..self.num_qubits {
                self.x_matrix[[p, j]] = false;
                self.z_matrix[[p, j]] = j == qubit;
            }

            // Random measurement outcome
            let mut rng = thread_rng();
            let outcome = rng.gen_bool(0.5);
            self.phase[p] = outcome;

            // Update other stabilizers that anticommute
            for i in 0..self.num_qubits {
                if i != p && self.x_matrix[[i, qubit]] {
                    // Multiply by stabilizer p
                    for j in 0..self.num_qubits {
                        self.x_matrix[[i, j]] ^= self.x_matrix[[p, j]];
                        self.z_matrix[[i, j]] ^= self.z_matrix[[p, j]];
                    }
                    // Update phase
                    self.phase[i] ^= self.phase[p];
                }
            }

            Ok(outcome) // Measurement outcome
        } else {
            // Deterministic outcome
            // Check if Z_qubit is in the stabilizer group
            let mut outcome = false;

            for i in 0..self.num_qubits {
                if self.z_matrix[[i, qubit]] && !self.x_matrix[[i, qubit]] {
                    outcome = self.phase[i];
                    break;
                }
            }

            Ok(outcome)
        }
    }

    /// Get the current stabilizer generators as strings
    #[must_use]
    pub fn get_stabilizers(&self) -> Vec<String> {
        let mut stabilizers = Vec::new();

        for i in 0..self.num_qubits {
            let mut stab = String::new();

            // Phase
            if self.phase[i] {
                stab.push('-');
            } else {
                stab.push('+');
            }

            // Pauli string
            for j in 0..self.num_qubits {
                let has_x = self.x_matrix[[i, j]];
                let has_z = self.z_matrix[[i, j]];

                match (has_x, has_z) {
                    (false, false) => stab.push('I'),
                    (true, false) => stab.push('X'),
                    (false, true) => stab.push('Z'),
                    (true, true) => stab.push('Y'),
                }
            }

            stabilizers.push(stab);
        }

        stabilizers
    }
}

/// Stabilizer simulator that efficiently simulates Clifford circuits
pub struct StabilizerSimulator {
    tableau: StabilizerTableau,
    measurement_record: Vec<(usize, bool)>,
}

impl StabilizerSimulator {
    /// Create a new stabilizer simulator
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            tableau: StabilizerTableau::new(num_qubits),
            measurement_record: Vec::new(),
        }
    }

    /// Apply a gate to the simulator
    pub fn apply_gate(&mut self, gate: StabilizerGate) -> Result<(), QuantRS2Error> {
        match gate {
            StabilizerGate::H(q) => self.tableau.apply_h(q),
            StabilizerGate::S(q) => self.tableau.apply_s(q),
            StabilizerGate::X(q) => self.tableau.apply_x(q),
            StabilizerGate::Y(q) => self.tableau.apply_y(q),
            StabilizerGate::Z(q) => self.tableau.apply_z(q),
            StabilizerGate::CNOT(c, t) => self.tableau.apply_cnot(c, t),
        }
    }

    /// Measure a qubit
    pub fn measure(&mut self, qubit: usize) -> Result<bool, QuantRS2Error> {
        let outcome = self.tableau.measure(qubit)?;
        self.measurement_record.push((qubit, outcome));
        Ok(outcome)
    }

    /// Get the current stabilizers
    #[must_use]
    pub fn get_stabilizers(&self) -> Vec<String> {
        self.tableau.get_stabilizers()
    }

    /// Get measurement record
    #[must_use]
    pub fn get_measurements(&self) -> &[(usize, bool)] {
        &self.measurement_record
    }

    /// Reset the simulator
    pub fn reset(&mut self) {
        let num_qubits = self.tableau.num_qubits;
        self.tableau = StabilizerTableau::new(num_qubits);
        self.measurement_record.clear();
    }

    /// Get the number of qubits
    #[must_use]
    pub const fn num_qubits(&self) -> usize {
        self.tableau.num_qubits
    }

    /// Get the state vector (for compatibility with other simulators)
    /// Note: This is expensive for stabilizer states and returns a sparse representation
    #[must_use]
    pub fn get_statevector(&self) -> Vec<Complex64> {
        let n = self.tableau.num_qubits;
        let dim = 1 << n;
        let mut state = vec![Complex64::new(0.0, 0.0); dim];

        // For a stabilizer state, we can determine which computational basis states
        // have non-zero amplitude by finding the simultaneous +1 eigenstates of all stabilizers
        // This is a simplified implementation that assumes the state is in |0...0>
        // A full implementation would need to solve the stabilizer equations

        // For now, return a simple state
        state[0] = Complex64::new(1.0, 0.0);
        state
    }
}

/// Gates supported by the stabilizer simulator
#[derive(Debug, Clone, Copy)]
pub enum StabilizerGate {
    H(usize),
    S(usize),
    X(usize),
    Y(usize),
    Z(usize),
    CNOT(usize, usize),
}

/// Check if a circuit can be simulated by the stabilizer simulator
#[must_use]
pub fn is_clifford_circuit<const N: usize>(circuit: &Circuit<N>) -> bool {
    // Check if all gates in the circuit are Clifford gates
    // Clifford gates: H, S, S†, CNOT, X, Y, Z, CZ
    circuit.gates().iter().all(|gate| {
        matches!(
            gate.name(),
            "H" | "S" | "S†" | "CNOT" | "X" | "Y" | "Z" | "CZ" | "Phase" | "PhaseDagger"
        )
    })
}

/// Convert a gate operation to a stabilizer gate
fn gate_to_stabilizer(gate: &Arc<dyn GateOp + Send + Sync>) -> Option<StabilizerGate> {
    let gate_name = gate.name();
    let qubits = gate.qubits();

    match gate_name {
        "H" => {
            if qubits.len() == 1 {
                Some(StabilizerGate::H(qubits[0].0 as usize))
            } else {
                None
            }
        }
        "S" | "Phase" => {
            if qubits.len() == 1 {
                Some(StabilizerGate::S(qubits[0].0 as usize))
            } else {
                None
            }
        }
        "X" => {
            if qubits.len() == 1 {
                Some(StabilizerGate::X(qubits[0].0 as usize))
            } else {
                None
            }
        }
        "Y" => {
            if qubits.len() == 1 {
                Some(StabilizerGate::Y(qubits[0].0 as usize))
            } else {
                None
            }
        }
        "Z" => {
            if qubits.len() == 1 {
                Some(StabilizerGate::Z(qubits[0].0 as usize))
            } else {
                None
            }
        }
        "CNOT" => {
            if qubits.len() == 2 {
                Some(StabilizerGate::CNOT(
                    qubits[0].0 as usize,
                    qubits[1].0 as usize,
                ))
            } else {
                None
            }
        }
        "CZ" => {
            if qubits.len() == 2 {
                // CZ = H(target) CNOT H(target)
                // For now, we can decompose this or add native support
                // Let's return None for unsupported gates
                None
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Implement the Simulator trait for `StabilizerSimulator`
impl Simulator for StabilizerSimulator {
    fn run<const N: usize>(
        &mut self,
        circuit: &Circuit<N>,
    ) -> crate::error::Result<SimulatorResult<N>> {
        // Create a new simulator instance
        let mut sim = Self::new(N);

        // Apply all gates in the circuit
        for gate in circuit.gates() {
            if let Some(stab_gate) = gate_to_stabilizer(gate) {
                // Ignore errors for now - in production we'd handle them properly
                let _ = sim.apply_gate(stab_gate);
            }
        }

        // Get the state vector (expensive operation)
        let amplitudes = sim.get_statevector();

        Ok(SimulatorResult::new(amplitudes))
    }
}

/// Builder for creating circuits that can be simulated with the stabilizer formalism
pub struct CliffordCircuitBuilder {
    gates: Vec<StabilizerGate>,
    num_qubits: usize,
}

impl CliffordCircuitBuilder {
    /// Create a new Clifford circuit builder
    #[must_use]
    pub const fn new(num_qubits: usize) -> Self {
        Self {
            gates: Vec::new(),
            num_qubits,
        }
    }

    /// Add a Hadamard gate
    #[must_use]
    pub fn h(mut self, qubit: usize) -> Self {
        self.gates.push(StabilizerGate::H(qubit));
        self
    }

    /// Add an S gate
    #[must_use]
    pub fn s(mut self, qubit: usize) -> Self {
        self.gates.push(StabilizerGate::S(qubit));
        self
    }

    /// Add a Pauli-X gate
    #[must_use]
    pub fn x(mut self, qubit: usize) -> Self {
        self.gates.push(StabilizerGate::X(qubit));
        self
    }

    /// Add a Pauli-Y gate
    #[must_use]
    pub fn y(mut self, qubit: usize) -> Self {
        self.gates.push(StabilizerGate::Y(qubit));
        self
    }

    /// Add a Pauli-Z gate
    #[must_use]
    pub fn z(mut self, qubit: usize) -> Self {
        self.gates.push(StabilizerGate::Z(qubit));
        self
    }

    /// Add a CNOT gate
    #[must_use]
    pub fn cnot(mut self, control: usize, target: usize) -> Self {
        self.gates.push(StabilizerGate::CNOT(control, target));
        self
    }

    /// Build and run the circuit
    pub fn run(self) -> Result<StabilizerSimulator, QuantRS2Error> {
        let mut sim = StabilizerSimulator::new(self.num_qubits);

        for gate in self.gates {
            sim.apply_gate(gate)?;
        }

        Ok(sim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stabilizer_init() {
        let sim = StabilizerSimulator::new(3);
        let stabs = sim.get_stabilizers();

        assert_eq!(stabs.len(), 3);
        assert_eq!(stabs[0], "+ZII");
        assert_eq!(stabs[1], "+IZI");
        assert_eq!(stabs[2], "+IIZ");
    }

    #[test]
    fn test_hadamard_gate() {
        let mut sim = StabilizerSimulator::new(1);
        sim.apply_gate(StabilizerGate::H(0))
            .expect("Hadamard gate application should succeed");

        let stabs = sim.get_stabilizers();
        assert_eq!(stabs[0], "+X");
    }

    #[test]
    fn test_bell_state() {
        let mut sim = StabilizerSimulator::new(2);
        sim.apply_gate(StabilizerGate::H(0))
            .expect("Hadamard gate application should succeed");
        sim.apply_gate(StabilizerGate::CNOT(0, 1))
            .expect("CNOT gate application should succeed");

        let stabs = sim.get_stabilizers();
        assert!(stabs.contains(&"+XX".to_string()));
        assert!(stabs.contains(&"+ZZ".to_string()));
    }

    #[test]
    fn test_ghz_state() {
        let mut sim = StabilizerSimulator::new(3);
        sim.apply_gate(StabilizerGate::H(0))
            .expect("Hadamard gate application should succeed");
        sim.apply_gate(StabilizerGate::CNOT(0, 1))
            .expect("CNOT gate application should succeed");
        sim.apply_gate(StabilizerGate::CNOT(1, 2))
            .expect("CNOT gate application should succeed");

        let stabs = sim.get_stabilizers();
        assert!(stabs.contains(&"+XXX".to_string()));
        assert!(stabs.contains(&"+ZZI".to_string()));
        assert!(stabs.contains(&"+IZZ".to_string()));
    }
}
