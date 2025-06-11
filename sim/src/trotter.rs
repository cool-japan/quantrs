//! Trotter-Suzuki decomposition for time evolution of quantum systems.
//!
//! This module implements various Trotter-Suzuki formulas for approximating
//! the time evolution operator exp(-iHt) where H = H1 + H2 + ... + Hn.

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use std::f64::consts::PI;

use crate::error::{Result, SimulatorError};
use crate::sparse::{CSRMatrix, SparseMatrixBuilder};
use quantrs2_core::gate::{multi::*, single::*, GateOp};
use quantrs2_core::qubit::QubitId;

/// Dynamic circuit that doesn't require compile-time qubit count
#[derive(Debug)]
pub struct DynamicCircuit {
    /// Gates in the circuit
    gates: Vec<Box<dyn GateOp>>,
    /// Number of qubits
    num_qubits: usize,
}

// Manual Clone implementation since Box<dyn GateOp> doesn't implement Clone
impl Clone for DynamicCircuit {
    fn clone(&self) -> Self {
        // We can't clone Box<dyn GateOp> directly, so we create an empty circuit
        // In practice, we'd need a method to deep clone gates
        Self {
            gates: Vec::new(),
            num_qubits: self.num_qubits,
        }
    }
}

impl DynamicCircuit {
    /// Create a new dynamic circuit
    pub fn new(num_qubits: usize) -> Self {
        Self {
            gates: Vec::new(),
            num_qubits,
        }
    }

    /// Add a gate to the circuit
    pub fn add_gate(&mut self, gate: Box<dyn GateOp>) -> Result<()> {
        // Validate qubits are in range
        for qubit in gate.qubits() {
            if qubit.id() as usize >= self.num_qubits {
                return Err(SimulatorError::IndexOutOfBounds(qubit.id() as usize));
            }
        }
        self.gates.push(gate);
        Ok(())
    }

    /// Get the gates
    pub fn gates(&self) -> &[Box<dyn GateOp>] {
        &self.gates
    }

    /// Get gate count
    pub fn gate_count(&self) -> usize {
        self.gates.len()
    }
}

/// Hamiltonian term type
#[derive(Debug, Clone)]
pub enum HamiltonianTerm {
    /// Single-qubit Pauli operator
    SinglePauli {
        qubit: usize,
        pauli: String,
        coefficient: f64,
    },
    /// Two-qubit Pauli operator
    TwoPauli {
        qubit1: usize,
        qubit2: usize,
        pauli1: String,
        pauli2: String,
        coefficient: f64,
    },
    /// Multi-qubit Pauli string
    PauliString {
        qubits: Vec<usize>,
        paulis: Vec<String>,
        coefficient: f64,
    },
    /// Custom unitary operator
    Custom {
        qubits: Vec<usize>,
        matrix: CSRMatrix,
        coefficient: f64,
    },
}

/// Hamiltonian as sum of terms
#[derive(Debug, Clone)]
pub struct Hamiltonian {
    /// Individual terms in the Hamiltonian
    pub terms: Vec<HamiltonianTerm>,
    /// Total number of qubits
    pub num_qubits: usize,
}

impl Hamiltonian {
    /// Create a new Hamiltonian
    pub fn new(num_qubits: usize) -> Self {
        Self {
            terms: Vec::new(),
            num_qubits,
        }
    }

    /// Add a single-qubit Pauli term
    pub fn add_single_pauli(&mut self, qubit: usize, pauli: &str, coefficient: f64) -> Result<()> {
        if qubit >= self.num_qubits {
            return Err(SimulatorError::IndexOutOfBounds(qubit));
        }

        self.terms.push(HamiltonianTerm::SinglePauli {
            qubit,
            pauli: pauli.to_uppercase(),
            coefficient,
        });
        Ok(())
    }

    /// Add a two-qubit Pauli term
    pub fn add_two_pauli(
        &mut self,
        qubit1: usize,
        qubit2: usize,
        pauli1: &str,
        pauli2: &str,
        coefficient: f64,
    ) -> Result<()> {
        if qubit1 >= self.num_qubits || qubit2 >= self.num_qubits {
            return Err(SimulatorError::InvalidInput(
                "Qubit index out of bounds".to_string(),
            ));
        }

        self.terms.push(HamiltonianTerm::TwoPauli {
            qubit1,
            qubit2,
            pauli1: pauli1.to_uppercase(),
            pauli2: pauli2.to_uppercase(),
            coefficient,
        });
        Ok(())
    }

    /// Add a Pauli string term
    pub fn add_pauli_string(
        &mut self,
        qubits: Vec<usize>,
        paulis: Vec<String>,
        coefficient: f64,
    ) -> Result<()> {
        if qubits.len() != paulis.len() {
            return Err(SimulatorError::InvalidInput(
                "Qubits and Paulis must have the same length".to_string(),
            ));
        }

        for &q in &qubits {
            if q >= self.num_qubits {
                return Err(SimulatorError::IndexOutOfBounds(q));
            }
        }

        self.terms.push(HamiltonianTerm::PauliString {
            qubits,
            paulis: paulis.iter().map(|p| p.to_uppercase()).collect(),
            coefficient,
        });
        Ok(())
    }

    /// Get the number of qubits
    pub fn get_num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Add a term to the Hamiltonian
    pub fn add_term(&mut self, term: HamiltonianTerm) {
        self.terms.push(term);
    }

    /// Add a Pauli term with the signature expected by adiabatic module
    pub fn add_pauli_term(&mut self, coefficient: f64, paulis: &[(usize, char)]) -> Result<()> {
        if paulis.is_empty() {
            return Ok(());
        }

        if paulis.len() == 1 {
            let (qubit, pauli) = paulis[0];
            self.add_single_pauli(qubit, &pauli.to_string(), coefficient)
        } else if paulis.len() == 2 {
            let (qubit1, pauli1) = paulis[0];
            let (qubit2, pauli2) = paulis[1];
            self.add_two_pauli(
                qubit1,
                qubit2,
                &pauli1.to_string(),
                &pauli2.to_string(),
                coefficient,
            )
        } else {
            let qubits: Vec<usize> = paulis.iter().map(|(q, _)| *q).collect();
            let pauli_strings: Vec<String> = paulis.iter().map(|(_, p)| p.to_string()).collect();
            self.add_pauli_string(qubits, pauli_strings, coefficient)
        }
    }

    /// Convert a term to a circuit with time evolution
    fn term_to_circuit(&self, term: &HamiltonianTerm, time: f64) -> Result<DynamicCircuit> {
        let mut circuit = DynamicCircuit::new(self.num_qubits);

        match term {
            HamiltonianTerm::SinglePauli {
                qubit,
                pauli,
                coefficient,
            } => {
                let angle = -2.0 * coefficient * time;
                match pauli.as_str() {
                    "X" => circuit.add_gate(Box::new(RotationX {
                        target: QubitId::new(*qubit as u32),
                        theta: angle,
                    }))?,
                    "Y" => circuit.add_gate(Box::new(RotationY {
                        target: QubitId::new(*qubit as u32),
                        theta: angle,
                    }))?,
                    "Z" => circuit.add_gate(Box::new(RotationZ {
                        target: QubitId::new(*qubit as u32),
                        theta: angle,
                    }))?,
                    _ => {
                        return Err(SimulatorError::InvalidInput(format!(
                            "Unknown Pauli operator: {}",
                            pauli
                        )))
                    }
                }
            }
            HamiltonianTerm::TwoPauli {
                qubit1,
                qubit2,
                pauli1,
                pauli2,
                coefficient,
            } => {
                // exp(-i*coeff*t*P1⊗P2) implementation
                let angle = -2.0 * coefficient * time;

                // Basis transformation
                self.apply_pauli_basis_change(&mut circuit, *qubit1, pauli1, false)?;
                self.apply_pauli_basis_change(&mut circuit, *qubit2, pauli2, false)?;

                // ZZ rotation
                circuit.add_gate(Box::new(CNOT {
                    control: QubitId::new(*qubit1 as u32),
                    target: QubitId::new(*qubit2 as u32),
                }))?;
                circuit.add_gate(Box::new(RotationZ {
                    target: QubitId::new(*qubit2 as u32),
                    theta: angle,
                }))?;
                circuit.add_gate(Box::new(CNOT {
                    control: QubitId::new(*qubit1 as u32),
                    target: QubitId::new(*qubit2 as u32),
                }))?;

                // Inverse basis transformation
                self.apply_pauli_basis_change(&mut circuit, *qubit1, pauli1, true)?;
                self.apply_pauli_basis_change(&mut circuit, *qubit2, pauli2, true)?;
            }
            HamiltonianTerm::PauliString {
                qubits,
                paulis,
                coefficient,
            } => {
                let angle = -2.0 * coefficient * time;

                // Basis transformations
                for (q, p) in qubits.iter().zip(paulis.iter()) {
                    self.apply_pauli_basis_change(&mut circuit, *q, p, false)?;
                }

                // Multi-qubit Z rotation using CNOT ladder
                for i in 0..qubits.len() - 1 {
                    circuit.add_gate(Box::new(CNOT {
                        control: QubitId::new(qubits[i] as u32),
                        target: QubitId::new(qubits[i + 1] as u32),
                    }))?;
                }

                circuit.add_gate(Box::new(RotationZ {
                    target: QubitId::new(qubits[qubits.len() - 1] as u32),
                    theta: angle,
                }))?;

                // Reverse CNOT ladder
                for i in (0..qubits.len() - 1).rev() {
                    circuit.add_gate(Box::new(CNOT {
                        control: QubitId::new(qubits[i] as u32),
                        target: QubitId::new(qubits[i + 1] as u32),
                    }))?;
                }

                // Inverse basis transformations
                for (q, p) in qubits.iter().zip(paulis.iter()) {
                    self.apply_pauli_basis_change(&mut circuit, *q, p, true)?;
                }
            }
            HamiltonianTerm::Custom { .. } => {
                return Err(SimulatorError::InvalidOperation(
                    "Custom terms not yet supported in Trotter decomposition".to_string(),
                ));
            }
        }

        Ok(circuit)
    }

    /// Apply Pauli basis change
    fn apply_pauli_basis_change(
        &self,
        circuit: &mut DynamicCircuit,
        qubit: usize,
        pauli: &str,
        inverse: bool,
    ) -> Result<()> {
        match pauli {
            "X" => {
                circuit.add_gate(Box::new(Hadamard {
                    target: QubitId::new(qubit as u32),
                }))?;
            }
            "Y" => {
                if inverse {
                    circuit.add_gate(Box::new(PhaseDagger {
                        target: QubitId::new(qubit as u32),
                    }))?;
                    circuit.add_gate(Box::new(Hadamard {
                        target: QubitId::new(qubit as u32),
                    }))?;
                } else {
                    circuit.add_gate(Box::new(Hadamard {
                        target: QubitId::new(qubit as u32),
                    }))?;
                    circuit.add_gate(Box::new(Phase {
                        target: QubitId::new(qubit as u32),
                    }))?;
                }
            }
            "Z" => {
                // No basis change needed for Z
            }
            _ => {
                return Err(SimulatorError::InvalidInput(format!(
                    "Unknown Pauli operator: {}",
                    pauli
                )))
            }
        }
        Ok(())
    }
}

/// Trotter-Suzuki decomposition methods
#[derive(Debug, Clone, Copy)]
pub enum TrotterMethod {
    /// First-order Trotter formula: exp(-iHt) ≈ ∏_j exp(-iH_j*t)
    FirstOrder,
    /// Second-order Suzuki formula: S2(t) = ∏_j exp(-iH_j*t/2) * ∏_{j'} exp(-iH_{j'}*t/2)
    SecondOrder,
    /// Fourth-order Suzuki formula
    FourthOrder,
    /// Sixth-order Suzuki formula
    SixthOrder,
    /// Randomized Trotter formula
    Randomized,
}

/// Trotter-Suzuki decomposer
pub struct TrotterDecomposer {
    /// Method to use
    method: TrotterMethod,
    /// Number of Trotter steps
    num_steps: usize,
}

impl TrotterDecomposer {
    /// Create a new decomposer
    pub fn new(method: TrotterMethod, num_steps: usize) -> Self {
        Self { method, num_steps }
    }

    /// Decompose time evolution into a circuit
    pub fn decompose(&self, hamiltonian: &Hamiltonian, total_time: f64) -> Result<DynamicCircuit> {
        let dt = total_time / self.num_steps as f64;
        let mut full_circuit = DynamicCircuit::new(hamiltonian.num_qubits);

        for _ in 0..self.num_steps {
            let step_circuit = match self.method {
                TrotterMethod::FirstOrder => self.first_order_step(hamiltonian, dt)?,
                TrotterMethod::SecondOrder => self.second_order_step(hamiltonian, dt)?,
                TrotterMethod::FourthOrder => self.fourth_order_step(hamiltonian, dt)?,
                TrotterMethod::SixthOrder => self.sixth_order_step(hamiltonian, dt)?,
                TrotterMethod::Randomized => self.randomized_step(hamiltonian, dt)?,
            };

            // Append step circuit
            for gate in step_circuit.gates() {
                full_circuit.add_gate(gate.clone())?;
            }
        }

        Ok(full_circuit)
    }

    /// First-order Trotter step
    fn first_order_step(&self, hamiltonian: &Hamiltonian, dt: f64) -> Result<DynamicCircuit> {
        let mut circuit = DynamicCircuit::new(hamiltonian.num_qubits);

        for term in &hamiltonian.terms {
            let term_circuit = hamiltonian.term_to_circuit(term, dt)?;
            for gate in term_circuit.gates() {
                circuit.add_gate(gate.clone())?;
            }
        }

        Ok(circuit)
    }

    /// Second-order Suzuki step
    fn second_order_step(&self, hamiltonian: &Hamiltonian, dt: f64) -> Result<DynamicCircuit> {
        let mut circuit = DynamicCircuit::new(hamiltonian.num_qubits);

        // Forward pass with dt/2
        for term in &hamiltonian.terms {
            let term_circuit = hamiltonian.term_to_circuit(term, dt / 2.0)?;
            for gate in term_circuit.gates() {
                circuit.add_gate(gate.clone())?;
            }
        }

        // Backward pass with dt/2
        for term in hamiltonian.terms.iter().rev() {
            let term_circuit = hamiltonian.term_to_circuit(term, dt / 2.0)?;
            for gate in term_circuit.gates() {
                circuit.add_gate(gate.clone())?;
            }
        }

        Ok(circuit)
    }

    /// Fourth-order Suzuki step
    fn fourth_order_step(&self, hamiltonian: &Hamiltonian, dt: f64) -> Result<DynamicCircuit> {
        let mut circuit = DynamicCircuit::new(hamiltonian.num_qubits);

        // S4(t) = S2(s*t)*S2(s*t)*S2((1-4s)*t)*S2(s*t)*S2(s*t)
        // where s = 1/(4 - 4^(1/3))
        let s = 1.0 / (4.0 - 4.0_f64.powf(1.0 / 3.0));

        // Apply S2(s*dt) twice
        for _ in 0..2 {
            let step = self.second_order_step(hamiltonian, s * dt)?;
            for gate in step.gates() {
                circuit.add_gate(gate.clone())?;
            }
        }

        // Apply S2((1-4s)*dt)
        let middle_step = self.second_order_step(hamiltonian, (1.0 - 4.0 * s) * dt)?;
        for gate in middle_step.gates() {
            circuit.add_gate(gate.clone())?;
        }

        // Apply S2(s*dt) twice more
        for _ in 0..2 {
            let step = self.second_order_step(hamiltonian, s * dt)?;
            for gate in step.gates() {
                circuit.add_gate(gate.clone())?;
            }
        }

        Ok(circuit)
    }

    /// Sixth-order Suzuki step
    fn sixth_order_step(&self, hamiltonian: &Hamiltonian, dt: f64) -> Result<DynamicCircuit> {
        let mut circuit = DynamicCircuit::new(hamiltonian.num_qubits);

        // Sixth-order coefficients
        let w1 = 1.0 / (2.0 - 2.0_f64.powf(1.0 / 5.0));
        let w0 = 1.0 - 2.0 * w1;

        // Apply S4(w1*dt) twice
        for _ in 0..2 {
            let step = self.fourth_order_step(hamiltonian, w1 * dt)?;
            for gate in step.gates() {
                circuit.add_gate(gate.clone())?;
            }
        }

        // Apply S4(w0*dt)
        let middle_step = self.fourth_order_step(hamiltonian, w0 * dt)?;
        for gate in middle_step.gates() {
            circuit.add_gate(gate.clone())?;
        }

        // Apply S4(w1*dt) twice more
        for _ in 0..2 {
            let step = self.fourth_order_step(hamiltonian, w1 * dt)?;
            for gate in step.gates() {
                circuit.add_gate(gate.clone())?;
            }
        }

        Ok(circuit)
    }

    /// Randomized Trotter step
    fn randomized_step(&self, hamiltonian: &Hamiltonian, dt: f64) -> Result<DynamicCircuit> {
        let mut circuit = DynamicCircuit::new(hamiltonian.num_qubits);

        // Randomly permute the terms
        let mut indices: Vec<usize> = (0..hamiltonian.terms.len()).collect();
        fastrand::shuffle(&mut indices);

        // Apply terms in random order
        for &idx in &indices {
            let term_circuit = hamiltonian.term_to_circuit(&hamiltonian.terms[idx], dt)?;
            for gate in term_circuit.gates() {
                circuit.add_gate(gate.clone())?;
            }
        }

        Ok(circuit)
    }

    /// Estimate error bound for the decomposition
    pub fn error_bound(&self, hamiltonian: &Hamiltonian, total_time: f64) -> f64 {
        let dt = total_time / self.num_steps as f64;
        let num_terms = hamiltonian.terms.len();

        // Compute sum of coefficient magnitudes
        let coeff_sum: f64 = hamiltonian
            .terms
            .iter()
            .map(|term| match term {
                HamiltonianTerm::SinglePauli { coefficient, .. }
                | HamiltonianTerm::TwoPauli { coefficient, .. }
                | HamiltonianTerm::PauliString { coefficient, .. }
                | HamiltonianTerm::Custom { coefficient, .. } => coefficient.abs(),
            })
            .sum();

        // Error bounds for different orders
        match self.method {
            TrotterMethod::FirstOrder => {
                // O(dt^2) error
                0.5 * coeff_sum.powi(2) * dt.powi(2) * self.num_steps as f64
            }
            TrotterMethod::SecondOrder => {
                // O(dt^3) error
                coeff_sum.powi(3) * dt.powi(3) * self.num_steps as f64 / 12.0
            }
            TrotterMethod::FourthOrder => {
                // O(dt^5) error
                coeff_sum.powi(5) * dt.powi(5) * self.num_steps as f64 / 360.0
            }
            TrotterMethod::SixthOrder => {
                // O(dt^7) error
                coeff_sum.powi(7) * dt.powi(7) * self.num_steps as f64 / 20160.0
            }
            TrotterMethod::Randomized => {
                // Empirical bound for randomized
                0.5 * coeff_sum * (total_time / (self.num_steps as f64).sqrt())
            }
        }
    }
}

/// Create common Hamiltonians
pub struct HamiltonianLibrary;

impl HamiltonianLibrary {
    /// Transverse field Ising model: H = -J∑_<ij> Z_i Z_j - h∑_i X_i
    pub fn transverse_ising_1d(
        num_qubits: usize,
        j: f64,
        h: f64,
        periodic: bool,
    ) -> Result<Hamiltonian> {
        let mut ham = Hamiltonian::new(num_qubits);

        // ZZ interactions
        for i in 0..num_qubits - 1 {
            ham.add_two_pauli(i, i + 1, "Z", "Z", -j)?;
        }

        // Periodic boundary condition
        if periodic && num_qubits > 2 {
            ham.add_two_pauli(num_qubits - 1, 0, "Z", "Z", -j)?;
        }

        // Transverse field
        for i in 0..num_qubits {
            ham.add_single_pauli(i, "X", -h)?;
        }

        Ok(ham)
    }

    /// Heisenberg model: H = J∑_<ij> (X_i X_j + Y_i Y_j + Δ Z_i Z_j)
    pub fn heisenberg_1d(
        num_qubits: usize,
        j: f64,
        delta: f64,
        periodic: bool,
    ) -> Result<Hamiltonian> {
        let mut ham = Hamiltonian::new(num_qubits);

        for i in 0..num_qubits - 1 {
            ham.add_two_pauli(i, i + 1, "X", "X", j)?;
            ham.add_two_pauli(i, i + 1, "Y", "Y", j)?;
            ham.add_two_pauli(i, i + 1, "Z", "Z", j * delta)?;
        }

        if periodic && num_qubits > 2 {
            ham.add_two_pauli(num_qubits - 1, 0, "X", "X", j)?;
            ham.add_two_pauli(num_qubits - 1, 0, "Y", "Y", j)?;
            ham.add_two_pauli(num_qubits - 1, 0, "Z", "Z", j * delta)?;
        }

        Ok(ham)
    }

    /// XY model: H = J∑_<ij> (X_i X_j + Y_i Y_j)
    pub fn xy_model(num_qubits: usize, j: f64, periodic: bool) -> Result<Hamiltonian> {
        Self::heisenberg_1d(num_qubits, j, 0.0, periodic)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamiltonian_construction() {
        let mut ham = Hamiltonian::new(3);

        ham.add_single_pauli(0, "X", 0.5).unwrap();
        ham.add_two_pauli(0, 1, "Z", "Z", -1.0).unwrap();
        ham.add_pauli_string(
            vec![0, 1, 2],
            vec!["X".to_string(), "Y".to_string(), "Z".to_string()],
            0.25,
        )
        .unwrap();

        assert_eq!(ham.terms.len(), 3);
    }

    #[test]
    fn test_ising_model() {
        let ham = HamiltonianLibrary::transverse_ising_1d(4, 1.0, 0.5, false).unwrap();

        // 3 ZZ terms + 4 X terms
        assert_eq!(ham.terms.len(), 7);
    }

    #[test]
    fn test_trotter_decomposition() {
        let ham = HamiltonianLibrary::transverse_ising_1d(3, 1.0, 0.5, false).unwrap();
        let decomposer = TrotterDecomposer::new(TrotterMethod::SecondOrder, 10);

        let circuit = decomposer.decompose(&ham, 1.0).unwrap();
        assert!(circuit.gate_count() > 0);
    }

    #[test]
    fn test_error_bounds() {
        let ham = HamiltonianLibrary::xy_model(4, 1.0, true).unwrap();
        let decomposer = TrotterDecomposer::new(TrotterMethod::FourthOrder, 100);

        let error = decomposer.error_bound(&ham, 1.0);
        assert!(error < 1e-6); // Fourth order with 100 steps should be very accurate
    }

    #[test]
    fn test_heisenberg_model() {
        let ham = HamiltonianLibrary::heisenberg_1d(3, 1.0, 1.0, false).unwrap();

        // 2 * 3 interactions (XX, YY, ZZ)
        assert_eq!(ham.terms.len(), 6);
    }
}
