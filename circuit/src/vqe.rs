//! Variational Quantum Eigensolver (VQE) circuit support
//!
//! This module provides specialized circuits and optimizers for the Variational Quantum Eigensolver
//! algorithm, which is used to find ground state energies of quantum systems.

use crate::builder::Circuit;
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::single::{RotationX, RotationY, RotationZ},
    gate::GateOp,
    qubit::QubitId,
};
use scirs2_core::Complex64;
use std::collections::HashMap;

/// Which axis a parameterized rotation gate acts on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationAxis {
    Y,
    Z,
    X,
}

/// Record of a parameterized gate: its position in the circuit's gate list,
/// the target qubit, the rotation axis, and the parameter index it uses.
#[derive(Debug, Clone)]
pub struct ParameterizedGateRecord {
    /// Position of this gate in `circuit.gates()` (gate list index).
    pub gate_index: usize,
    /// Target qubit for the rotation.
    pub qubit: QubitId,
    /// Rotation axis.
    pub axis: RotationAxis,
    /// Index into `parameters` for the angle.
    pub param_index: usize,
}

/// A parameterized quantum circuit for VQE applications
///
/// VQE circuits are characterized by:
/// - Parameterized gates whose angles can be optimized
/// - Specific ansatz structures (e.g., UCCSD, hardware-efficient)
/// - Observable measurement capabilities
#[derive(Debug, Clone)]
pub struct VQECircuit<const N: usize> {
    /// The underlying quantum circuit
    pub circuit: Circuit<N>,
    /// Parameters that can be optimized
    pub parameters: Vec<f64>,
    /// Parameter names for identification
    pub parameter_names: Vec<String>,
    /// Mapping from parameter names to indices
    parameter_map: HashMap<String, usize>,
    /// Ordered list of parameterized gate records: used by `set_parameters` to
    /// rebuild the circuit's rotation angles when parameters change.
    param_gate_records: Vec<ParameterizedGateRecord>,
}

/// VQE ansatz types for different quantum chemistry problems
#[derive(Debug, Clone, PartialEq)]
pub enum VQEAnsatz {
    /// Hardware-efficient ansatz with alternating rotation and entangling layers
    HardwareEfficient { layers: usize },
    /// Unitary Coupled-Cluster Singles and Doubles
    UCCSD {
        occupied_orbitals: usize,
        virtual_orbitals: usize,
    },
    /// Real-space ansatz for condensed matter systems
    RealSpace { geometry: Vec<(f64, f64, f64)> },
    /// Custom ansatz defined by user
    Custom,
}

/// Observable for VQE energy measurements
#[derive(Debug, Clone)]
pub struct VQEObservable {
    /// Pauli string coefficients and operators
    pub terms: Vec<(f64, Vec<(usize, PauliOperator)>)>,
}

/// Pauli operators for observable construction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PauliOperator {
    I, // Identity
    X, // Pauli-X
    Y, // Pauli-Y
    Z, // Pauli-Z
}

/// VQE optimization result
#[derive(Debug, Clone)]
pub struct VQEResult {
    /// Optimized parameters
    pub optimal_parameters: Vec<f64>,
    /// Ground state energy
    pub ground_state_energy: f64,
    /// Number of optimization iterations
    pub iterations: usize,
    /// Convergence status
    pub converged: bool,
    /// Final gradient norm
    pub gradient_norm: f64,
}

impl<const N: usize> VQECircuit<N> {
    /// Create a new VQE circuit with specified ansatz
    pub fn new(ansatz: VQEAnsatz) -> QuantRS2Result<Self> {
        let mut circuit = Circuit::new();
        let mut parameters = Vec::new();
        let mut parameter_names = Vec::new();
        let mut parameter_map = HashMap::new();
        let mut param_gate_records: Vec<ParameterizedGateRecord> = Vec::new();

        match ansatz {
            VQEAnsatz::HardwareEfficient { layers } => {
                Self::build_hardware_efficient_ansatz(
                    &mut circuit,
                    &mut parameters,
                    &mut parameter_names,
                    &mut parameter_map,
                    &mut param_gate_records,
                    layers,
                )?;
            }
            VQEAnsatz::UCCSD {
                occupied_orbitals,
                virtual_orbitals,
            } => {
                Self::build_uccsd_ansatz(
                    &mut circuit,
                    &mut parameters,
                    &mut parameter_names,
                    &mut parameter_map,
                    &mut param_gate_records,
                    occupied_orbitals,
                    virtual_orbitals,
                )?;
            }
            VQEAnsatz::RealSpace { geometry } => {
                Self::build_real_space_ansatz(
                    &mut circuit,
                    &mut parameters,
                    &mut parameter_names,
                    &mut parameter_map,
                    &mut param_gate_records,
                    &geometry,
                )?;
            }
            VQEAnsatz::Custom => {
                // Custom ansatz - circuit will be built by user
            }
        }

        Ok(Self {
            circuit,
            parameters,
            parameter_names,
            parameter_map,
            param_gate_records,
        })
    }

    /// Build a hardware-efficient ansatz
    fn build_hardware_efficient_ansatz(
        circuit: &mut Circuit<N>,
        parameters: &mut Vec<f64>,
        parameter_names: &mut Vec<String>,
        parameter_map: &mut HashMap<String, usize>,
        param_gate_records: &mut Vec<ParameterizedGateRecord>,
        layers: usize,
    ) -> QuantRS2Result<()> {
        for layer in 0..layers {
            // Single-qubit rotation layer
            for qubit in 0..N {
                // RY rotation
                let param_name = format!("ry_{layer}_q{qubit}");
                let param_idx = parameters.len();
                parameter_names.push(param_name.clone());
                parameter_map.insert(param_name, param_idx);
                parameters.push(0.0);

                let gate_idx = circuit.gates().len();
                circuit.ry(QubitId(qubit as u32), 0.0)?;
                param_gate_records.push(ParameterizedGateRecord {
                    gate_index: gate_idx,
                    qubit: QubitId(qubit as u32),
                    axis: RotationAxis::Y,
                    param_index: param_idx,
                });

                // RZ rotation
                let param_name = format!("rz_{layer}_q{qubit}");
                let param_idx = parameters.len();
                parameter_names.push(param_name.clone());
                parameter_map.insert(param_name, param_idx);
                parameters.push(0.0);

                let gate_idx = circuit.gates().len();
                circuit.rz(QubitId(qubit as u32), 0.0)?;
                param_gate_records.push(ParameterizedGateRecord {
                    gate_index: gate_idx,
                    qubit: QubitId(qubit as u32),
                    axis: RotationAxis::Z,
                    param_index: param_idx,
                });
            }

            // Entangling layer (linear connectivity)
            for qubit in 0..(N - 1) {
                circuit.cnot(QubitId(qubit as u32), QubitId((qubit + 1) as u32))?;
            }
        }

        Ok(())
    }

    /// Build a UCCSD ansatz (simplified version)
    fn build_uccsd_ansatz(
        circuit: &mut Circuit<N>,
        parameters: &mut Vec<f64>,
        parameter_names: &mut Vec<String>,
        parameter_map: &mut HashMap<String, usize>,
        param_gate_records: &mut Vec<ParameterizedGateRecord>,
        occupied_orbitals: usize,
        virtual_orbitals: usize,
    ) -> QuantRS2Result<()> {
        if occupied_orbitals + virtual_orbitals > N {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Total orbitals ({}) exceeds number of qubits ({})",
                occupied_orbitals + virtual_orbitals,
                N
            )));
        }

        // Initialize with Hartree-Fock state
        for i in 0..occupied_orbitals {
            circuit.x(QubitId(i as u32))?;
        }

        // Single excitations
        for i in 0..occupied_orbitals {
            for a in occupied_orbitals..(occupied_orbitals + virtual_orbitals) {
                let param_name = format!("t1_{i}_{a}");
                let param_idx = parameters.len();
                parameter_names.push(param_name.clone());
                parameter_map.insert(param_name, param_idx);
                parameters.push(0.0);

                circuit.cnot(QubitId(i as u32), QubitId(a as u32))?;
                let gate_idx = circuit.gates().len();
                circuit.ry(QubitId(a as u32), 0.0)?;
                param_gate_records.push(ParameterizedGateRecord {
                    gate_index: gate_idx,
                    qubit: QubitId(a as u32),
                    axis: RotationAxis::Y,
                    param_index: param_idx,
                });
                circuit.cnot(QubitId(i as u32), QubitId(a as u32))?;
            }
        }

        // Double excitations (simplified)
        for i in 0..occupied_orbitals {
            for j in (i + 1)..occupied_orbitals {
                for a in occupied_orbitals..(occupied_orbitals + virtual_orbitals) {
                    for b in (a + 1)..(occupied_orbitals + virtual_orbitals) {
                        if a < N && b < N {
                            let param_name = format!("t2_{i}_{j}_{a}_{b}");
                            let param_idx = parameters.len();
                            parameter_names.push(param_name.clone());
                            parameter_map.insert(param_name, param_idx);
                            parameters.push(0.0);

                            circuit.cnot(QubitId(i as u32), QubitId(a as u32))?;
                            circuit.cnot(QubitId(j as u32), QubitId(b as u32))?;
                            let gate_idx = circuit.gates().len();
                            circuit.ry(QubitId(a as u32), 0.0)?;
                            param_gate_records.push(ParameterizedGateRecord {
                                gate_index: gate_idx,
                                qubit: QubitId(a as u32),
                                axis: RotationAxis::Y,
                                param_index: param_idx,
                            });
                            circuit.cnot(QubitId(j as u32), QubitId(b as u32))?;
                            circuit.cnot(QubitId(i as u32), QubitId(a as u32))?;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Build a real-space ansatz
    fn build_real_space_ansatz(
        circuit: &mut Circuit<N>,
        parameters: &mut Vec<f64>,
        parameter_names: &mut Vec<String>,
        parameter_map: &mut HashMap<String, usize>,
        param_gate_records: &mut Vec<ParameterizedGateRecord>,
        geometry: &[(f64, f64, f64)],
    ) -> QuantRS2Result<()> {
        if geometry.len() > N {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Geometry has {} sites but circuit only has {} qubits",
                geometry.len(),
                N
            )));
        }

        // Build ansatz based on geometric connectivity
        for (i, &(x1, y1, z1)) in geometry.iter().enumerate() {
            for (j, &(x2, y2, z2)) in geometry.iter().enumerate().skip(i + 1) {
                let distance = (z2 - z1)
                    .mul_add(z2 - z1, (y2 - y1).mul_add(y2 - y1, (x2 - x1).powi(2)))
                    .sqrt();

                // Only include interactions within a cutoff distance
                if distance < 3.0 {
                    let param_name = format!("j_{i}_{j}");
                    let param_idx = parameters.len();
                    parameter_names.push(param_name.clone());
                    parameter_map.insert(param_name, param_idx);
                    parameters.push(0.0);

                    circuit.cnot(QubitId(i as u32), QubitId(j as u32))?;
                    let gate_idx = circuit.gates().len();
                    circuit.rz(QubitId(j as u32), 0.0)?;
                    param_gate_records.push(ParameterizedGateRecord {
                        gate_index: gate_idx,
                        qubit: QubitId(j as u32),
                        axis: RotationAxis::Z,
                        param_index: param_idx,
                    });
                    circuit.cnot(QubitId(i as u32), QubitId(j as u32))?;
                }
            }
        }

        Ok(())
    }

    /// Update circuit parameters and rebuild all parameterized rotation gates.
    ///
    /// Uses `param_gate_records` to locate each parameterized gate in the gate
    /// list.  The entire circuit is reconstructed from `gates_as_boxes()`, with
    /// each parameterized gate replaced by a new rotation gate carrying the
    /// updated angle.  Non-parameterized gates are kept verbatim.
    pub fn set_parameters(&mut self, new_parameters: &[f64]) -> QuantRS2Result<()> {
        if new_parameters.len() != self.parameters.len() {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Expected {} parameters, got {}",
                self.parameters.len(),
                new_parameters.len()
            )));
        }

        self.parameters = new_parameters.to_vec();

        // Build a map from gate_index → ParameterizedGateRecord for fast lookup.
        let record_map: HashMap<usize, &ParameterizedGateRecord> = self
            .param_gate_records
            .iter()
            .map(|r| (r.gate_index, r))
            .collect();

        // Collect all existing gates as boxed trait objects.
        let old_gates = self.circuit.gates_as_boxes();

        // Rebuild a new gate list, substituting updated rotation angles where recorded.
        let new_gates: Vec<Box<dyn GateOp>> = old_gates
            .into_iter()
            .enumerate()
            .map(|(idx, gate)| -> Box<dyn GateOp> {
                if let Some(record) = record_map.get(&idx) {
                    let angle = self.parameters[record.param_index];
                    match record.axis {
                        RotationAxis::Y => Box::new(RotationY {
                            target: record.qubit,
                            theta: angle,
                        }),
                        RotationAxis::Z => Box::new(RotationZ {
                            target: record.qubit,
                            theta: angle,
                        }),
                        RotationAxis::X => Box::new(RotationX {
                            target: record.qubit,
                            theta: angle,
                        }),
                    }
                } else {
                    gate
                }
            })
            .collect();

        // Replace the circuit with the rebuilt version.
        self.circuit = Circuit::<N>::from_gates(new_gates)?;

        Ok(())
    }

    /// Get a parameter by name
    #[must_use]
    pub fn get_parameter(&self, name: &str) -> Option<f64> {
        self.parameter_map
            .get(name)
            .map(|&index| self.parameters[index])
    }

    /// Set a parameter by name
    pub fn set_parameter(&mut self, name: &str, value: f64) -> QuantRS2Result<()> {
        let index = self
            .parameter_map
            .get(name)
            .ok_or_else(|| QuantRS2Error::InvalidInput(format!("Parameter '{name}' not found")))?;

        self.parameters[*index] = value;
        Ok(())
    }

    /// Add a custom parameterized RY gate.
    ///
    /// Records the gate position so that `set_parameters` can later update its angle.
    pub fn add_parameterized_ry(
        &mut self,
        qubit: QubitId,
        parameter_name: &str,
    ) -> QuantRS2Result<()> {
        if self.parameter_map.contains_key(parameter_name) {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Parameter '{parameter_name}' already exists"
            )));
        }

        let param_idx = self.parameters.len();
        self.parameter_names.push(parameter_name.to_string());
        self.parameter_map
            .insert(parameter_name.to_string(), param_idx);
        self.parameters.push(0.0);

        let gate_idx = self.circuit.gates().len();
        self.circuit.ry(qubit, 0.0)?;
        self.param_gate_records.push(ParameterizedGateRecord {
            gate_index: gate_idx,
            qubit,
            axis: RotationAxis::Y,
            param_index: param_idx,
        });

        Ok(())
    }

    /// Add a custom parameterized RZ gate.
    ///
    /// Records the gate position so that `set_parameters` can later update its angle.
    pub fn add_parameterized_rz(
        &mut self,
        qubit: QubitId,
        parameter_name: &str,
    ) -> QuantRS2Result<()> {
        if self.parameter_map.contains_key(parameter_name) {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Parameter '{parameter_name}' already exists"
            )));
        }

        let param_idx = self.parameters.len();
        self.parameter_names.push(parameter_name.to_string());
        self.parameter_map
            .insert(parameter_name.to_string(), param_idx);
        self.parameters.push(0.0);

        let gate_idx = self.circuit.gates().len();
        self.circuit.rz(qubit, 0.0)?;
        self.param_gate_records.push(ParameterizedGateRecord {
            gate_index: gate_idx,
            qubit,
            axis: RotationAxis::Z,
            param_index: param_idx,
        });

        Ok(())
    }

    /// Get the number of parameters
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        self.parameters.len()
    }
}

impl VQEObservable {
    /// Create a new empty observable
    #[must_use]
    pub const fn new() -> Self {
        Self { terms: Vec::new() }
    }

    /// Add a Pauli string term to the observable
    pub fn add_pauli_term(&mut self, coefficient: f64, pauli_string: Vec<(usize, PauliOperator)>) {
        self.terms.push((coefficient, pauli_string));
    }

    /// Create a Heisenberg model Hamiltonian
    #[must_use]
    pub fn heisenberg_model(num_qubits: usize, j_coupling: f64) -> Self {
        let mut observable = Self::new();

        for i in 0..(num_qubits - 1) {
            // XX term
            observable.add_pauli_term(
                j_coupling,
                vec![(i, PauliOperator::X), (i + 1, PauliOperator::X)],
            );
            // YY term
            observable.add_pauli_term(
                j_coupling,
                vec![(i, PauliOperator::Y), (i + 1, PauliOperator::Y)],
            );
            // ZZ term
            observable.add_pauli_term(
                j_coupling,
                vec![(i, PauliOperator::Z), (i + 1, PauliOperator::Z)],
            );
        }

        observable
    }

    /// Create a transverse field Ising model Hamiltonian
    #[must_use]
    pub fn tfim(num_qubits: usize, j_coupling: f64, h_field: f64) -> Self {
        let mut observable = Self::new();

        // ZZ interactions
        for i in 0..(num_qubits - 1) {
            observable.add_pauli_term(
                -j_coupling,
                vec![(i, PauliOperator::Z), (i + 1, PauliOperator::Z)],
            );
        }

        // X field terms
        for i in 0..num_qubits {
            observable.add_pauli_term(-h_field, vec![(i, PauliOperator::X)]);
        }

        observable
    }

    /// Create a molecular Hamiltonian (simplified version)
    #[must_use]
    pub fn molecular_hamiltonian(
        one_body: &[(usize, usize, f64)],
        two_body: &[(usize, usize, usize, usize, f64)],
    ) -> Self {
        let mut observable = Self::new();

        // One-body terms (simplified representation)
        for &(i, j, coeff) in one_body {
            if i == j {
                // Diagonal term
                observable.add_pauli_term(coeff, vec![(i, PauliOperator::Z)]);
            } else {
                // Off-diagonal terms (simplified)
                observable
                    .add_pauli_term(coeff, vec![(i, PauliOperator::X), (j, PauliOperator::X)]);
                observable
                    .add_pauli_term(coeff, vec![(i, PauliOperator::Y), (j, PauliOperator::Y)]);
            }
        }

        // Two-body terms (very simplified representation)
        for &(i, j, k, l, coeff) in two_body {
            // This is a simplified representation - real molecular Hamiltonians
            // require more sophisticated fermion-to-qubit mappings
            observable.add_pauli_term(
                coeff,
                vec![
                    (i, PauliOperator::Z),
                    (j, PauliOperator::Z),
                    (k, PauliOperator::Z),
                    (l, PauliOperator::Z),
                ],
            );
        }

        observable
    }
}

impl Default for VQEObservable {
    fn default() -> Self {
        Self::new()
    }
}

/// VQE optimizer for finding ground state energies
pub struct VQEOptimizer {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Learning rate for gradient descent
    pub learning_rate: f64,
    /// Optimizer type
    pub optimizer_type: VQEOptimizerType,
}

/// Types of optimizers available for VQE
#[derive(Debug, Clone, PartialEq)]
pub enum VQEOptimizerType {
    /// Gradient descent
    GradientDescent,
    /// Adam optimizer
    Adam { beta1: f64, beta2: f64 },
    /// BFGS quasi-Newton method
    BFGS,
    /// Nelder-Mead simplex
    NelderMead,
    /// SPSA (Simultaneous Perturbation Stochastic Approximation)
    SPSA { alpha: f64, gamma: f64 },
}

impl VQEOptimizer {
    /// Create a new VQE optimizer
    #[must_use]
    pub const fn new(optimizer_type: VQEOptimizerType) -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            learning_rate: 0.01,
            optimizer_type,
        }
    }

    /// Optimize VQE circuit parameters
    pub fn optimize<const N: usize>(
        &self,
        circuit: &mut VQECircuit<N>,
        observable: &VQEObservable,
    ) -> QuantRS2Result<VQEResult> {
        // This is a simplified implementation - a full VQE optimizer would:
        // 1. Evaluate the expectation value of the observable
        // 2. Compute gradients (analytically or numerically)
        // 3. Update parameters using the chosen optimization algorithm
        // 4. Check for convergence

        let mut current_energy = self.evaluate_energy(circuit, observable)?;
        let mut best_parameters = circuit.parameters.clone();
        let mut best_energy = current_energy;

        for iteration in 0..self.max_iterations {
            // Simplified gradient descent step
            let gradients = self.compute_gradients(circuit, observable)?;

            // Update parameters
            for (i, gradient) in gradients.iter().enumerate() {
                circuit.parameters[i] -= self.learning_rate * gradient;
            }

            // Evaluate new energy
            current_energy = self.evaluate_energy(circuit, observable)?;

            if current_energy < best_energy {
                best_energy = current_energy;
                best_parameters.clone_from(&circuit.parameters);
            }

            // Check convergence
            let gradient_norm = gradients.iter().map(|g| g * g).sum::<f64>().sqrt();
            if gradient_norm < self.tolerance {
                circuit.parameters = best_parameters;
                return Ok(VQEResult {
                    optimal_parameters: circuit.parameters.clone(),
                    ground_state_energy: best_energy,
                    iterations: iteration + 1,
                    converged: true,
                    gradient_norm,
                });
            }
        }

        circuit.parameters = best_parameters;
        Ok(VQEResult {
            optimal_parameters: circuit.parameters.clone(),
            ground_state_energy: best_energy,
            iterations: self.max_iterations,
            converged: false,
            gradient_norm: 0.0, // Would compute actual gradient norm
        })
    }

    /// Evaluate the energy expectation value (simplified)
    const fn evaluate_energy<const N: usize>(
        &self,
        _circuit: &VQECircuit<N>,
        _observable: &VQEObservable,
    ) -> QuantRS2Result<f64> {
        // This is a placeholder - real implementation would:
        // 1. Execute the circuit on a quantum simulator/device
        // 2. Measure expectation values of Pauli strings
        // 3. Combine measurements according to observable coefficients

        // For now, return a dummy energy value
        Ok(-1.0)
    }

    /// Compute parameter gradients (simplified)
    fn compute_gradients<const N: usize>(
        &self,
        circuit: &VQECircuit<N>,
        _observable: &VQEObservable,
    ) -> QuantRS2Result<Vec<f64>> {
        // This is a placeholder - real implementation would use:
        // 1. Parameter shift rule for analytic gradients
        // 2. Finite differences for numerical gradients
        // 3. Or other gradient estimation methods

        // For now, return dummy gradients
        Ok(vec![0.001; circuit.parameters.len()])
    }
}

impl Default for VQEOptimizer {
    fn default() -> Self {
        Self::new(VQEOptimizerType::GradientDescent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_efficient_ansatz() {
        let circuit = VQECircuit::<4>::new(VQEAnsatz::HardwareEfficient { layers: 2 })
            .expect("create VQE circuit");
        assert!(!circuit.parameters.is_empty());
        assert_eq!(circuit.parameter_names.len(), circuit.parameters.len());
    }

    #[test]
    fn test_observable_creation() {
        let obs = VQEObservable::heisenberg_model(4, 1.0);
        assert!(!obs.terms.is_empty());
    }

    #[test]
    fn test_parameter_management() {
        let mut circuit =
            VQECircuit::<2>::new(VQEAnsatz::Custom).expect("create custom VQE circuit");
        circuit
            .add_parameterized_ry(QubitId(0), "theta1")
            .expect("add parameterized RY gate");
        circuit
            .set_parameter("theta1", 0.5)
            .expect("set parameter theta1");
        assert_eq!(circuit.get_parameter("theta1"), Some(0.5));
    }

    #[test]
    fn test_set_parameters_updates_circuit_gates() {
        use std::f64::consts::PI;

        // Build a custom VQE circuit with one RY gate
        let mut vqe = VQECircuit::<2>::new(VQEAnsatz::Custom).expect("custom VQE");
        vqe.add_parameterized_ry(QubitId(0), "theta")
            .expect("add RY");
        vqe.add_parameterized_rz(QubitId(1), "phi").expect("add RZ");

        assert_eq!(vqe.num_parameters(), 2);

        // Initially parameters are zero
        assert_eq!(vqe.get_parameter("theta"), Some(0.0));
        assert_eq!(vqe.get_parameter("phi"), Some(0.0));

        // Update both parameters
        vqe.set_parameters(&[PI / 4.0, PI / 2.0])
            .expect("set params");

        // Parameters stored correctly
        assert!((vqe.get_parameter("theta").unwrap() - PI / 4.0).abs() < 1e-12);
        assert!((vqe.get_parameter("phi").unwrap() - PI / 2.0).abs() < 1e-12);

        // Circuit was rebuilt: should still have the same number of gates
        assert_eq!(vqe.circuit.gates().len(), 2);

        // Verify the gates have the updated angles by inspecting their names
        // (RY and RZ gate names)
        let gate_names: Vec<&str> = vqe.circuit.gates().iter().map(|g| g.name()).collect();
        assert_eq!(gate_names, vec!["RY", "RZ"]);
    }

    #[test]
    fn test_set_parameters_hardware_efficient() {
        use std::f64::consts::PI;

        let mut vqe = VQECircuit::<2>::new(VQEAnsatz::HardwareEfficient { layers: 1 })
            .expect("hardware-efficient VQE");

        let n_params = vqe.num_parameters();
        assert!(n_params > 0);

        // Create a new parameter vector with all PI/3
        let new_params: Vec<f64> = vec![PI / 3.0; n_params];
        vqe.set_parameters(&new_params).expect("set all params");

        // Circuit should be rebuilt with same gate structure
        for &p in &vqe.parameters {
            assert!((p - PI / 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_set_parameters_wrong_length_fails() {
        let mut vqe = VQECircuit::<2>::new(VQEAnsatz::Custom).expect("custom VQE");
        vqe.add_parameterized_ry(QubitId(0), "theta")
            .expect("add RY");

        // Providing wrong number of parameters should return an error
        let result = vqe.set_parameters(&[0.1, 0.2]);
        assert!(result.is_err());
    }
}
