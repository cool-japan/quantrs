//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use crate::fermionic_simulation::{FermionicHamiltonian, FermionicOperator, FermionicString};
use crate::pauli::{PauliOperator, PauliOperatorSum, PauliString};
use scirs2_core::ndarray::{Array1, Array2, Array4};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;
use std::collections::HashMap;
use std::f64::consts::PI;

/// Molecular Hamiltonian in second quantization
#[derive(Debug, Clone)]
pub struct MolecularHamiltonian {
    /// One-electron integrals (kinetic + nuclear attraction)
    pub one_electron_integrals: Array2<f64>,
    /// Two-electron integrals (electron-electron repulsion)
    pub two_electron_integrals: Array4<f64>,
    /// Nuclear repulsion energy
    pub nuclear_repulsion: f64,
    /// Number of molecular orbitals
    pub num_orbitals: usize,
    /// Number of electrons
    pub num_electrons: usize,
    /// Fermionic Hamiltonian representation
    pub fermionic_hamiltonian: FermionicHamiltonian,
    /// Pauli representation (after fermion-to-spin mapping)
    pub pauli_hamiltonian: Option<PauliOperatorSum>,
}
/// Molecular orbital representation
#[derive(Debug, Clone)]
pub struct MolecularOrbitals {
    /// Orbital coefficients
    pub coefficients: Array2<f64>,
    /// Orbital energies
    pub energies: Array1<f64>,
    /// Occupation numbers
    pub occupations: Array1<f64>,
    /// Number of basis functions
    pub num_basis: usize,
    /// Number of molecular orbitals
    pub num_orbitals: usize,
}
/// Fermion-to-spin mapping methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FermionMapping {
    /// Jordan-Wigner transformation
    JordanWigner,
    /// Parity mapping
    Parity,
    /// Bravyi-Kitaev transformation
    BravyiKitaev,
    /// Symmetry-conserving Bravyi-Kitaev
    SymmetryConservingBK,
    /// Fenwick tree mapping
    FenwickTree,
}
/// Statistics for quantum chemistry calculations
#[derive(Debug, Clone, Default)]
pub struct ChemistryStats {
    /// Total computation time
    pub total_time_ms: f64,
    /// Hamiltonian construction time
    pub hamiltonian_time_ms: f64,
    /// VQE optimization time
    pub vqe_time_ms: f64,
    /// Number of quantum circuit evaluations
    pub circuit_evaluations: usize,
    /// Number of parameter updates
    pub parameter_updates: usize,
    /// Memory usage for matrices
    pub memory_usage_mb: f64,
    /// Hamiltonian terms count
    pub hamiltonian_terms: usize,
}
/// Fermion-to-spin mapping utilities
#[derive(Debug, Clone)]
pub struct FermionMapper {
    /// Mapping method
    pub(super) method: FermionMapping,
    /// Number of spin orbitals
    pub(crate) num_spin_orbitals: usize,
    /// Cached mappings
    mapping_cache: HashMap<String, PauliString>,
}
impl FermionMapper {
    #[must_use]
    pub fn new(method: FermionMapping, num_spin_orbitals: usize) -> Self {
        Self {
            method,
            num_spin_orbitals,
            mapping_cache: HashMap::new(),
        }
    }
    pub(super) fn map_fermionic_string(
        &self,
        fermionic_string: &FermionicString,
    ) -> Result<PauliString> {
        let mut paulis = HashMap::new();
        for (i, operator) in fermionic_string.operators.iter().enumerate() {
            match operator {
                FermionicOperator::Creation(site) => {
                    paulis.insert(*site, PauliOperator::X);
                }
                FermionicOperator::Annihilation(site) => {
                    paulis.insert(*site, PauliOperator::X);
                }
                _ => {
                    paulis.insert(i, PauliOperator::Z);
                }
            }
        }
        let mut operators_vec = vec![PauliOperator::I; self.num_spin_orbitals];
        for (qubit, op) in paulis {
            if qubit < operators_vec.len() {
                operators_vec[qubit] = op;
            }
        }
        let num_qubits = operators_vec.len();
        Ok(PauliString {
            operators: operators_vec,
            coefficient: fermionic_string.coefficient,
            num_qubits,
        })
    }
    /// Calculate molecular dipole moment from density matrix
    pub(super) fn calculate_dipole_moment(
        &self,
        density_matrix: &Array2<f64>,
    ) -> Result<Array1<f64>> {
        let mut dipole = Array1::zeros(3);
        let num_orbitals = density_matrix.nrows();
        for i in 0..num_orbitals {
            for j in 0..num_orbitals {
                let density_element = density_matrix[[i, j]];
                if i == j {
                    let orbital_pos = i as f64 / num_orbitals as f64;
                    dipole[0] -= density_element * orbital_pos;
                    dipole[1] -= density_element * orbital_pos * 0.5;
                    dipole[2] -= density_element * orbital_pos * 0.3;
                }
            }
        }
        Ok(dipole)
    }
    /// Get method reference
    #[must_use]
    pub const fn get_method(&self) -> &FermionMapping {
        &self.method
    }
    /// Get number of spin orbitals
    #[must_use]
    pub const fn get_num_spin_orbitals(&self) -> usize {
        self.num_spin_orbitals
    }
}
/// Optimizers for chemistry VQE
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChemistryOptimizer {
    /// Constrained Optimization BY Linear Approximation
    COBYLA,
    /// Sequential Least Squares Programming
    SLSQP,
    /// Powell's method
    Powell,
    /// Gradient descent
    GradientDescent,
    /// Adam optimizer
    Adam,
    /// Quantum Natural Gradient
    QuantumNaturalGradient,
}
/// VQE optimizer for chemistry problems
#[derive(Debug, Clone)]
pub struct VQEOptimizer {
    /// Optimization method
    method: ChemistryOptimizer,
    /// Current parameters
    pub(super) parameters: Array1<f64>,
    /// Parameter bounds
    pub(crate) bounds: Vec<(f64, f64)>,
    /// Optimization history
    pub(super) history: Vec<f64>,
    /// Gradient estimates
    gradients: Array1<f64>,
    /// Learning rate (for gradient-based methods)
    pub(super) learning_rate: f64,
}
impl VQEOptimizer {
    #[must_use]
    pub fn new(method: ChemistryOptimizer) -> Self {
        Self {
            method,
            parameters: Array1::zeros(0),
            bounds: Vec::new(),
            history: Vec::new(),
            gradients: Array1::zeros(0),
            learning_rate: 0.01,
        }
    }
    pub(super) fn initialize_parameters(&mut self, num_parameters: usize) {
        self.parameters = Array1::from_vec(
            (0..num_parameters)
                .map(|_| (thread_rng().random::<f64>() - 0.5) * 0.1)
                .collect(),
        );
        self.bounds = vec![(-PI, PI); num_parameters];
        self.gradients = Array1::zeros(num_parameters);
    }
    /// Initialize parameters (public version)
    pub fn initialize_parameters_public(&mut self, num_parameters: usize) {
        self.initialize_parameters(num_parameters);
    }
    /// Get parameters reference
    #[must_use]
    pub const fn get_parameters(&self) -> &Array1<f64> {
        &self.parameters
    }
    /// Get bounds reference
    #[must_use]
    pub fn get_bounds(&self) -> &[(f64, f64)] {
        &self.bounds
    }
    /// Get method reference
    #[must_use]
    pub const fn get_method(&self) -> &ChemistryOptimizer {
        &self.method
    }
}
/// Electronic structure methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElectronicStructureMethod {
    /// Hartree-Fock method
    HartreeFock,
    /// Variational Quantum Eigensolver
    VQE,
    /// Quantum Configuration Interaction
    QuantumCI,
    /// Quantum Coupled Cluster
    QuantumCC,
    /// Quantum Phase Estimation
    QPE,
}
/// Hartree-Fock calculation result
#[derive(Debug, Clone)]
pub struct HartreeFockResult {
    /// SCF energy
    pub scf_energy: f64,
    /// Molecular orbitals
    pub molecular_orbitals: MolecularOrbitals,
    /// Density matrix
    pub density_matrix: Array2<f64>,
    /// Fock matrix
    pub fock_matrix: Array2<f64>,
    /// Convergence achieved
    pub converged: bool,
    /// SCF iterations
    pub scf_iterations: usize,
}
/// VQE configuration for chemistry calculations
#[derive(Debug, Clone)]
pub struct VQEConfig {
    /// Ansatz type for VQE
    pub ansatz: ChemistryAnsatz,
    /// Optimizer for VQE
    pub optimizer: ChemistryOptimizer,
    /// Maximum VQE iterations
    pub max_iterations: usize,
    /// Convergence threshold for energy
    pub energy_threshold: f64,
    /// Gradient threshold for convergence
    pub gradient_threshold: f64,
    /// Shot noise for measurements
    pub shots: usize,
    /// Enable noise mitigation
    pub enable_noise_mitigation: bool,
}
/// Electronic structure result
#[derive(Debug, Clone)]
pub struct ElectronicStructureResult {
    /// Ground state energy
    pub ground_state_energy: f64,
    /// Molecular orbitals
    pub molecular_orbitals: MolecularOrbitals,
    /// Electronic density matrix
    pub density_matrix: Array2<f64>,
    /// Dipole moment
    pub dipole_moment: Array1<f64>,
    /// Convergence achieved
    pub converged: bool,
    /// Number of iterations performed
    pub iterations: usize,
    /// Final quantum state
    pub quantum_state: Array1<Complex64>,
    /// VQE optimization history
    pub vqe_history: Vec<f64>,
    /// Computational statistics
    pub stats: ChemistryStats,
}
/// Electronic structure configuration
#[derive(Debug, Clone)]
pub struct ElectronicStructureConfig {
    /// Method for electronic structure calculation
    pub method: ElectronicStructureMethod,
    /// Convergence criteria for SCF
    pub convergence_threshold: f64,
    /// Maximum SCF iterations
    pub max_scf_iterations: usize,
    /// Active space specification
    pub active_space: Option<ActiveSpace>,
    /// Enable second quantization optimization
    pub enable_second_quantization_optimization: bool,
    /// Fermion-to-spin mapping method
    pub fermion_mapping: FermionMapping,
    /// Enable orbital optimization
    pub enable_orbital_optimization: bool,
    /// VQE optimizer settings
    pub vqe_config: VQEConfig,
}
/// Active space specification for reduced basis calculations
#[derive(Debug, Clone)]
pub struct ActiveSpace {
    /// Number of active electrons
    pub num_electrons: usize,
    /// Number of active orbitals
    pub num_orbitals: usize,
    /// Orbital indices to include in active space
    pub orbital_indices: Vec<usize>,
    /// Frozen core orbitals
    pub frozen_core: Vec<usize>,
    /// Virtual orbitals to exclude
    pub frozen_virtual: Vec<usize>,
}
/// Molecular structure representation
#[derive(Debug, Clone)]
pub struct Molecule {
    /// Atomic numbers
    pub atomic_numbers: Vec<u32>,
    /// Atomic positions (x, y, z coordinates)
    pub positions: Array2<f64>,
    /// Molecular charge
    pub charge: i32,
    /// Spin multiplicity (2S + 1)
    pub multiplicity: u32,
    /// Basis set name
    pub basis_set: String,
}
/// Chemistry-specific ansätze for VQE
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChemistryAnsatz {
    /// Unitary Coupled Cluster Singles and Doubles
    UCCSD,
    /// Hardware Efficient Ansatz
    HardwareEfficient,
    /// Symmetry-Preserving Ansatz
    SymmetryPreserving,
    /// Low-Depth Circuit Ansatz
    LowDepth,
    /// Adaptive VQE ansatz
    Adaptive,
}
