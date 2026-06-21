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

        let mut best_energy = self.evaluate_energy(circuit, observable)?;
        let mut best_parameters = circuit.parameters.clone();

        for iteration in 0..self.max_iterations {
            // Gradient at the current parameter point (analytic parameter-shift).
            let gradients = self.compute_gradients(circuit, observable)?;
            let gradient_norm = gradients.iter().map(|g| g * g).sum::<f64>().sqrt();

            // Converged: the gradient is (numerically) zero, so we are at a
            // stationary point.  Check this *before* taking another step.
            if gradient_norm < self.tolerance {
                // Make sure the circuit holds the best parameters found.
                circuit.set_parameters(&best_parameters)?;
                return Ok(VQEResult {
                    optimal_parameters: best_parameters.clone(),
                    ground_state_energy: best_energy,
                    iterations: iteration + 1,
                    converged: true,
                    gradient_norm,
                });
            }

            // Gradient-descent update.  Crucially we rebuild the underlying
            // circuit via `set_parameters` so that the next energy/gradient
            // evaluation simulates the *updated* state (a previous version
            // mutated `parameters` directly, leaving the circuit gates stale).
            let mut next_parameters = circuit.parameters.clone();
            for (param, gradient) in next_parameters.iter_mut().zip(gradients.iter()) {
                *param -= self.learning_rate * gradient;
            }
            circuit.set_parameters(&next_parameters)?;

            // Evaluate new energy and track the best point seen.
            let current_energy = self.evaluate_energy(circuit, observable)?;
            if current_energy < best_energy {
                best_energy = current_energy;
                best_parameters.clone_from(&circuit.parameters);
            }
        }

        // Restore the best parameters and report the gradient norm there.
        circuit.set_parameters(&best_parameters)?;
        let final_gradient = self.compute_gradients(circuit, observable)?;
        let final_gradient_norm = final_gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        Ok(VQEResult {
            optimal_parameters: best_parameters.clone(),
            ground_state_energy: best_energy,
            iterations: self.max_iterations,
            converged: false,
            gradient_norm: final_gradient_norm,
        })
    }

    /// Evaluate the energy expectation value `⟨ψ(θ)|H|ψ(θ)⟩`.
    ///
    /// The ansatz state `|ψ(θ)⟩` is obtained by simulating the parameterized
    /// circuit on a dense state vector starting from `|0…0⟩` (see
    /// [`statevector::simulate`]).  The observable energy is the sum of each
    /// Pauli-string term's coefficient times its expectation value
    /// `⟨ψ|P|ψ⟩`, computed exactly via [`statevector::pauli_string_expectation`].
    fn evaluate_energy<const N: usize>(
        &self,
        circuit: &VQECircuit<N>,
        observable: &VQEObservable,
    ) -> QuantRS2Result<f64> {
        let state = statevector::simulate(&circuit.circuit)?;

        let mut energy = 0.0;
        for (coefficient, pauli_string) in &observable.terms {
            let expectation = statevector::pauli_string_expectation(&state, N, pauli_string)?;
            // For a physical Hamiltonian every Pauli expectation is real; we take
            // the real part and surface any spurious imaginary component as an
            // error rather than silently discarding it.
            if expectation.im.abs() > 1e-9 {
                return Err(QuantRS2Error::ComputationError(format!(
                    "Pauli-string expectation has non-negligible imaginary part ({:.3e}); \
                     observable is not Hermitian",
                    expectation.im
                )));
            }
            energy += coefficient * expectation.re;
        }

        Ok(energy)
    }

    /// Compute parameter gradients using the analytic parameter-shift rule.
    ///
    /// For a gate generated by a Pauli operator `P` (so that `U(θ) =
    /// exp(-i θ P / 2)`, which holds for the `RX`/`RY`/`RZ` rotations used by all
    /// the VQE ansätze), the energy gradient with respect to that parameter is
    /// `∂E/∂θ = ½[E(θ + π/2) − E(θ − π/2)]`.  This is exact (not a finite-
    /// difference approximation) for such gates.
    fn compute_gradients<const N: usize>(
        &self,
        circuit: &VQECircuit<N>,
        observable: &VQEObservable,
    ) -> QuantRS2Result<Vec<f64>> {
        let num_params = circuit.parameters.len();
        let mut gradients = Vec::with_capacity(num_params);

        // Work on a clone so the caller's circuit/parameters are untouched.
        let mut shifted = circuit.clone();
        let base_parameters = circuit.parameters.clone();
        let shift = std::f64::consts::FRAC_PI_2;

        for i in 0..num_params {
            let mut plus = base_parameters.clone();
            plus[i] += shift;
            shifted.set_parameters(&plus)?;
            let energy_plus = self.evaluate_energy(&shifted, observable)?;

            let mut minus = base_parameters.clone();
            minus[i] -= shift;
            shifted.set_parameters(&minus)?;
            let energy_minus = self.evaluate_energy(&shifted, observable)?;

            gradients.push(0.5 * (energy_plus - energy_minus));
        }

        // Restore the original parameters on the working copy (defensive; the
        // clone is dropped anyway, but keeps the helper side-effect-free).
        shifted.set_parameters(&base_parameters)?;

        Ok(gradients)
    }
}

/// Dense state-vector simulation utilities used by the VQE energy evaluator.
///
/// `quantrs2-circuit` is a dependency of `quantrs2-sim`, so it cannot depend on
/// the simulator crate (that would be a dependency cycle).  These helpers
/// therefore provide a small, self-contained exact state-vector engine driven
/// purely by the generic [`GateOp::matrix`] / [`GateOp::qubits`] interface, so
/// they correctly handle *every* gate type a VQE ansatz can contain, not just a
/// hard-coded subset.
mod statevector {
    use super::{Circuit, GateOp};
    use quantrs2_core::error::{QuantRS2Error, QuantRS2Result};
    use scirs2_core::Complex64;

    use super::PauliOperator;

    /// Simulate `circuit` on `2^N` amplitudes starting from `|0…0⟩`.
    pub fn simulate<const N: usize>(circuit: &Circuit<N>) -> QuantRS2Result<Vec<Complex64>> {
        let dim = 1usize << N;
        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        state[0] = Complex64::new(1.0, 0.0);

        for gate in circuit.gates() {
            apply_gate(&mut state, N, gate.as_ref())?;
        }

        Ok(state)
    }

    /// Apply a single (possibly multi-qubit) gate to the state vector in place.
    ///
    /// The gate's `2^k × 2^k` unitary (row-major, `k = gate.num_qubits()`) is
    /// applied to the subspace spanned by the gate's qubits.  Qubit `q` is the
    /// bit at position `q` of the basis index (little-endian), matching the rest
    /// of the framework's `QubitId` convention.
    pub fn apply_gate(
        state: &mut [Complex64],
        num_qubits: usize,
        gate: &dyn GateOp,
    ) -> QuantRS2Result<()> {
        let targets: Vec<usize> = gate.qubits().iter().map(|q| q.id() as usize).collect();
        let k = targets.len();
        if k == 0 {
            // Gates with no qubits (e.g. a global barrier) act as the identity.
            return Ok(());
        }
        for &t in &targets {
            if t >= num_qubits {
                return Err(QuantRS2Error::InvalidInput(format!(
                    "Gate '{}' acts on qubit {} but circuit only has {} qubits",
                    gate.name(),
                    t,
                    num_qubits
                )));
            }
        }

        let matrix = gate.matrix()?;
        let side = 1usize << k;
        if matrix.len() != side * side {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Gate '{}' returned a {}-element matrix but {} qubits require {}",
                gate.name(),
                matrix.len(),
                k,
                side * side
            )));
        }

        // Map each *local* bit position of the gate-block index to a state
        // qubit.  The framework's gate matrices are row-major in the basis where
        // the FIRST qubit of `qubits()` is the most-significant bit of the block
        // index (e.g. CNOT with `qubits()=[control,target]` swaps block indices
        // 2↔3 = |10⟩↔|11⟩, flipping the target when the control is set).  So
        // local bit `p` (p=0 is the LSB of the block index) corresponds to
        // `targets[k - 1 - p]`.
        let bit_masks: Vec<usize> = (0..k).map(|p| 1usize << targets[k - 1 - p]).collect();
        // Mask of all qubit bits touched by the gate; we iterate over every
        // assignment of the remaining bits and apply the dense block to the 2^k
        // amplitudes selected by the target bits.
        let mut fixed_mask = 0usize;
        for &m in &bit_masks {
            fixed_mask |= m;
        }
        let dim = state.len();

        let mut visited = vec![false; dim];
        let mut amplitudes = vec![Complex64::new(0.0, 0.0); side];
        let mut indices = vec![0usize; side];

        for base in 0..dim {
            if visited[base] || (base & fixed_mask) != 0 {
                // Only start from indices whose target bits are all zero; that
                // base seeds exactly one 2^k block.
                continue;
            }

            // Gather the amplitudes of the block.
            for (local, slot) in indices.iter_mut().enumerate() {
                let mut idx = base;
                for (bit, &mask) in bit_masks.iter().enumerate() {
                    if (local >> bit) & 1 == 1 {
                        idx |= mask;
                    }
                }
                *slot = idx;
                amplitudes[local] = state[idx];
                visited[idx] = true;
            }

            // Apply the dense unitary block: out[r] = Σ_c M[r,c] · in[c].
            for r in 0..side {
                let mut acc = Complex64::new(0.0, 0.0);
                let row = r * side;
                for (c, amp) in amplitudes.iter().enumerate() {
                    acc += matrix[row + c] * amp;
                }
                state[indices[r]] = acc;
            }
        }

        Ok(())
    }

    /// Compute `⟨ψ|P|ψ⟩` for a Pauli string `P = ⊗_q P_q`.
    ///
    /// Qubits absent from `pauli_string` carry an implicit identity.  The Pauli
    /// operators are applied directly to a working copy of the state (no dense
    /// matrix is formed), then the overlap with the original state is returned.
    pub fn pauli_string_expectation(
        state: &[Complex64],
        num_qubits: usize,
        pauli_string: &[(usize, PauliOperator)],
    ) -> QuantRS2Result<Complex64> {
        let mut transformed = state.to_vec();

        for &(qubit, op) in pauli_string {
            if qubit >= num_qubits {
                return Err(QuantRS2Error::InvalidInput(format!(
                    "Pauli term targets qubit {qubit} but state has {num_qubits} qubits"
                )));
            }
            apply_pauli(&mut transformed, qubit, op);
        }

        // ⟨ψ|P|ψ⟩ = Σ_i conj(ψ_i) · (Pψ)_i
        let mut acc = Complex64::new(0.0, 0.0);
        for (psi, pphi) in state.iter().zip(transformed.iter()) {
            acc += psi.conj() * pphi;
        }
        Ok(acc)
    }

    /// Apply a single-qubit Pauli operator to `state` in place.
    fn apply_pauli(state: &mut [Complex64], qubit: usize, op: PauliOperator) {
        let mask = 1usize << qubit;
        match op {
            PauliOperator::I => {}
            PauliOperator::X => {
                for idx in 0..state.len() {
                    if idx & mask == 0 {
                        state.swap(idx, idx | mask);
                    }
                }
            }
            PauliOperator::Y => {
                // Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩.
                let i = Complex64::new(0.0, 1.0);
                for idx in 0..state.len() {
                    if idx & mask == 0 {
                        let partner = idx | mask;
                        let a = state[idx];
                        let b = state[partner];
                        state[idx] = -i * b;
                        state[partner] = i * a;
                    }
                }
            }
            PauliOperator::Z => {
                for (idx, amp) in state.iter_mut().enumerate() {
                    if idx & mask != 0 {
                        *amp = -*amp;
                    }
                }
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::super::{Circuit, QubitId};
        use super::{apply_gate, simulate};
        use quantrs2_core::gate::multi::CNOT;
        use quantrs2_core::gate::single::{Hadamard, PauliX};
        use scirs2_core::Complex64;

        /// CNOT must map |10⟩ → |11⟩ (control = qubit 0, the MSB of the gate
        /// block).  This pins down the multi-qubit endianness: a wrong mapping
        /// would instead flip qubit 0 when qubit 1 is set.
        #[test]
        fn test_cnot_endianness() {
            // 2-qubit state |10⟩: qubit 0 = 1.  Little-endian basis index = 1<<0 = 1.
            let mut state = vec![Complex64::new(0.0, 0.0); 4];
            state[1] = Complex64::new(1.0, 0.0); // |q1 q0⟩ index: q0=1 → idx 1 = |10⟩
            let cnot = CNOT {
                control: QubitId(0),
                target: QubitId(1),
            };
            apply_gate(&mut state, 2, &cnot).expect("apply cnot");
            // Expect |11⟩: q0=1, q1=1 → idx = 0b11 = 3.
            assert!((state[3] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
            for (i, amp) in state.iter().enumerate() {
                if i != 3 {
                    assert!(amp.norm() < 1e-12, "unexpected amplitude at {i}: {amp}");
                }
            }
        }

        /// CNOT must leave |01⟩ unchanged (control qubit 0 = 0).
        #[test]
        fn test_cnot_control_zero_is_identity() {
            let mut state = vec![Complex64::new(0.0, 0.0); 4];
            state[2] = Complex64::new(1.0, 0.0); // q1=1, q0=0 → idx 0b10 = 2 = |01⟩
            let cnot = CNOT {
                control: QubitId(0),
                target: QubitId(1),
            };
            apply_gate(&mut state, 2, &cnot).expect("apply cnot");
            assert!((state[2] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
        }

        /// A Bell circuit H(0); CNOT(0,1) produces (|00⟩ + |11⟩)/√2.
        #[test]
        fn test_bell_state() {
            let mut circuit = Circuit::<2>::new();
            circuit
                .add_gate(Hadamard { target: QubitId(0) })
                .expect("h");
            circuit
                .add_gate(CNOT {
                    control: QubitId(0),
                    target: QubitId(1),
                })
                .expect("cnot");

            let state = simulate(&circuit).expect("simulate");
            let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
            assert!(
                (state[0].re - inv_sqrt2).abs() < 1e-12,
                "|00>: {}",
                state[0]
            );
            assert!(state[1].norm() < 1e-12, "|10>: {}", state[1]);
            assert!(state[2].norm() < 1e-12, "|01>: {}", state[2]);
            assert!(
                (state[3].re - inv_sqrt2).abs() < 1e-12,
                "|11>: {}",
                state[3]
            );
        }

        /// Applying X to qubit 1 of |00⟩ sets exactly qubit 1 (the high bit).
        #[test]
        fn test_single_qubit_targets_correct_bit() {
            let mut state = vec![Complex64::new(0.0, 0.0); 4];
            state[0] = Complex64::new(1.0, 0.0);
            let x = PauliX { target: QubitId(1) };
            apply_gate(&mut state, 2, &x).expect("apply x");
            // qubit 1 set → idx 0b10 = 2.
            assert!((state[2] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
        }
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

    /// `⟨0|RY(θ)† Z RY(θ)|0⟩ = cos θ` is a textbook identity.  This pins the
    /// real expectation-value engine to an analytic value and would fail for the
    /// former hard-coded `-1.0`.
    #[test]
    fn test_evaluate_energy_matches_analytic_cos() {
        use std::f64::consts::PI;

        let optimizer = VQEOptimizer::default();
        let mut z_observable = VQEObservable::new();
        z_observable.add_pauli_term(1.0, vec![(0, PauliOperator::Z)]);

        for &theta in &[0.0, PI / 6.0, PI / 3.0, PI / 2.0, 2.0 * PI / 3.0, PI] {
            let mut vqe = VQECircuit::<1>::new(VQEAnsatz::Custom).expect("custom VQE");
            vqe.add_parameterized_ry(QubitId(0), "theta").expect("RY");
            vqe.set_parameters(&[theta]).expect("set theta");

            let energy = optimizer
                .evaluate_energy(&vqe, &z_observable)
                .expect("evaluate energy");
            assert!(
                (energy - theta.cos()).abs() < 1e-9,
                "⟨Z⟩ for RY({theta}) was {energy}, expected {}",
                theta.cos()
            );
        }
    }

    /// The energy must depend on the parameters: a constant-`-1.0` fabrication
    /// would make every value identical.
    #[test]
    fn test_evaluate_energy_is_not_constant() {
        use std::f64::consts::PI;

        let optimizer = VQEOptimizer::default();
        let mut obs = VQEObservable::new();
        obs.add_pauli_term(1.0, vec![(0, PauliOperator::Z)]);

        let mut vqe = VQECircuit::<1>::new(VQEAnsatz::Custom).expect("custom VQE");
        vqe.add_parameterized_ry(QubitId(0), "theta").expect("RY");

        vqe.set_parameters(&[0.0]).expect("set");
        let e0 = optimizer.evaluate_energy(&vqe, &obs).expect("e0");
        vqe.set_parameters(&[PI]).expect("set");
        let e_pi = optimizer.evaluate_energy(&vqe, &obs).expect("e_pi");

        assert!((e0 - 1.0).abs() < 1e-9, "⟨Z⟩ at θ=0 should be +1, got {e0}");
        assert!(
            (e_pi + 1.0).abs() < 1e-9,
            "⟨Z⟩ at θ=π should be -1, got {e_pi}"
        );
        assert!((e0 - e_pi).abs() > 1.0, "energy must vary with parameters");
    }

    /// X expectation of `RY(θ)|0⟩` is `sin θ` — exercises a non-diagonal Pauli.
    #[test]
    fn test_evaluate_energy_pauli_x() {
        use std::f64::consts::PI;

        let optimizer = VQEOptimizer::default();
        let mut obs = VQEObservable::new();
        obs.add_pauli_term(1.0, vec![(0, PauliOperator::X)]);

        let mut vqe = VQECircuit::<1>::new(VQEAnsatz::Custom).expect("custom VQE");
        vqe.add_parameterized_ry(QubitId(0), "theta").expect("RY");
        vqe.set_parameters(&[PI / 2.0]).expect("set");

        let energy = optimizer.evaluate_energy(&vqe, &obs).expect("energy");
        assert!(
            (energy - 1.0).abs() < 1e-9,
            "⟨X⟩ for RY(π/2)|0⟩ should be 1, got {energy}"
        );
    }

    /// The analytic parameter-shift gradient must agree with a central finite
    /// difference of the (real) energy.
    #[test]
    fn test_parameter_shift_gradient_matches_finite_difference() {
        use std::f64::consts::PI;

        let optimizer = VQEOptimizer::default();
        // A non-trivial 2-qubit Hamiltonian with several Pauli terms.
        let mut obs = VQEObservable::new();
        obs.add_pauli_term(0.7, vec![(0, PauliOperator::Z)]);
        obs.add_pauli_term(-0.4, vec![(1, PauliOperator::X)]);
        obs.add_pauli_term(0.55, vec![(0, PauliOperator::Z), (1, PauliOperator::Z)]);

        let mut vqe = VQECircuit::<2>::new(VQEAnsatz::HardwareEfficient { layers: 1 })
            .expect("hardware-efficient VQE");
        let n = vqe.num_parameters();

        // Use a generic, non-symmetric parameter point.
        let base: Vec<f64> = (0..n)
            .map(|i| 0.13 + 0.21 * (i as f64) - PI / 5.0)
            .collect();
        vqe.set_parameters(&base).expect("set base params");

        let analytic = optimizer
            .compute_gradients(&vqe, &obs)
            .expect("analytic gradient");

        let eps = 1e-6;
        for i in 0..n {
            let mut plus = base.clone();
            plus[i] += eps;
            vqe.set_parameters(&plus).expect("set+");
            let ep = optimizer.evaluate_energy(&vqe, &obs).expect("e+");

            let mut minus = base.clone();
            minus[i] -= eps;
            vqe.set_parameters(&minus).expect("set-");
            let em = optimizer.evaluate_energy(&vqe, &obs).expect("e-");

            let numeric = (ep - em) / (2.0 * eps);
            assert!(
                (analytic[i] - numeric).abs() < 1e-5,
                "param {i}: analytic {} vs finite-difference {}",
                analytic[i],
                numeric
            );
        }
    }

    /// End-to-end: optimizing the single-qubit Hamiltonian `H = Z` with an RY
    /// ansatz must drive the energy toward the true ground-state energy `-1`.
    #[test]
    fn test_optimize_reaches_z_ground_state() {
        let optimizer = VQEOptimizer {
            learning_rate: 0.3,
            max_iterations: 500,
            ..VQEOptimizer::default()
        };

        let mut obs = VQEObservable::new();
        obs.add_pauli_term(1.0, vec![(0, PauliOperator::Z)]);

        let mut vqe = VQECircuit::<1>::new(VQEAnsatz::Custom).expect("custom VQE");
        vqe.add_parameterized_ry(QubitId(0), "theta").expect("RY");
        // Start away from both the minimum (θ=π) and the maximum (θ=0).
        vqe.set_parameters(&[0.6]).expect("init");

        let result = optimizer.optimize(&mut vqe, &obs).expect("optimize");
        assert!(
            (result.ground_state_energy + 1.0).abs() < 1e-3,
            "optimized energy {} should approach -1",
            result.ground_state_energy
        );
    }
}
