//! # QuantumChemistrySimulator - run_calculation_group Methods
//!
//! This module contains method implementations for `QuantumChemistrySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::ndarray::{Array1, Array2, Array4};
use scirs2_core::random::prelude::*;

use super::types::{
    ChemistryAnsatz, ChemistryOptimizer, ElectronicStructureMethod, ElectronicStructureResult,
    HartreeFockResult, MolecularOrbitals, Molecule,
};

use super::quantumchemistrysimulator_type::QuantumChemistrySimulator;

impl QuantumChemistrySimulator {
    /// Run complete electronic structure calculation
    pub fn run_calculation(&mut self) -> Result<ElectronicStructureResult> {
        let start_time = std::time::Instant::now();
        if self.molecule.is_none() {
            return Err(SimulatorError::InvalidConfiguration(
                "Molecule not set".to_string(),
            ));
        }
        let hamiltonian_start = std::time::Instant::now();
        let molecule_clone = self
            .molecule
            .clone()
            .ok_or_else(|| SimulatorError::InvalidConfiguration("Molecule not set".to_string()))?;
        self.construct_molecular_hamiltonian(&molecule_clone)?;
        self.stats.hamiltonian_time_ms = hamiltonian_start.elapsed().as_millis() as f64;
        self.run_hartree_fock()?;
        let result = match self.config.method {
            ElectronicStructureMethod::HartreeFock => self.run_hartree_fock_only(),
            ElectronicStructureMethod::VQE => self.run_vqe(),
            ElectronicStructureMethod::QuantumCI => self.run_quantum_ci(),
            ElectronicStructureMethod::QuantumCC => self.run_quantum_coupled_cluster(),
            ElectronicStructureMethod::QPE => self.run_quantum_phase_estimation(),
        }?;
        self.stats.total_time_ms = start_time.elapsed().as_millis() as f64;
        Ok(result)
    }
    /// Run Hartree-Fock calculation
    pub(super) fn run_hartree_fock(&mut self) -> Result<()> {
        let hamiltonian = self.hamiltonian.as_ref().ok_or_else(|| {
            SimulatorError::InvalidConfiguration("Hamiltonian not constructed".to_string())
        })?;
        let num_orbitals = hamiltonian.num_orbitals;
        let num_electrons = hamiltonian.num_electrons;
        let mut density_matrix = Array2::zeros((num_orbitals, num_orbitals));
        let mut fock_matrix = hamiltonian.one_electron_integrals.clone();
        let mut scf_energy = 0.0;
        let mut converged = false;
        let mut iteration = 0;
        while iteration < self.config.max_scf_iterations && !converged {
            self.build_fock_matrix(&mut fock_matrix, &density_matrix, hamiltonian)?;
            let (_energies, orbitals) = self.diagonalize_fock_matrix(&fock_matrix)?;
            let new_density = self.build_density_matrix(&orbitals, num_electrons)?;
            let new_energy = self.calculate_scf_energy(
                &new_density,
                &hamiltonian.one_electron_integrals,
                &fock_matrix,
            )?;
            let energy_change = (new_energy - scf_energy).abs();
            let density_change = (&new_density - &density_matrix).map(|x| x.abs()).sum();
            if energy_change < self.config.convergence_threshold
                && density_change < self.config.convergence_threshold
            {
                converged = true;
            }
            density_matrix = new_density;
            scf_energy = new_energy;
            iteration += 1;
        }
        let (energies, orbitals) = self.diagonalize_fock_matrix(&fock_matrix)?;
        let occupations = self.determine_occupations(&energies, num_electrons);
        let molecular_orbitals = MolecularOrbitals {
            coefficients: orbitals,
            energies,
            occupations,
            num_basis: num_orbitals,
            num_orbitals,
        };
        self.hartree_fock = Some(HartreeFockResult {
            scf_energy: scf_energy + hamiltonian.nuclear_repulsion,
            molecular_orbitals,
            density_matrix,
            fock_matrix,
            converged,
            scf_iterations: iteration,
        });
        Ok(())
    }
    /// Run VQE calculation
    pub(super) fn run_vqe(&mut self) -> Result<ElectronicStructureResult> {
        let vqe_start = std::time::Instant::now();
        let hamiltonian = self.hamiltonian.as_ref().ok_or_else(|| {
            SimulatorError::InvalidConfiguration("Hamiltonian not constructed".to_string())
        })?;
        let nuclear_repulsion = hamiltonian.nuclear_repulsion;
        let hf = self.hartree_fock.as_ref().ok_or_else(|| {
            SimulatorError::InvalidConfiguration("Hartree-Fock not converged".to_string())
        })?;
        let hf_molecular_orbitals = hf.molecular_orbitals.clone();
        let hf_density_matrix = hf.density_matrix.clone();
        let hf_result = self.hartree_fock.as_ref().ok_or_else(|| {
            SimulatorError::InvalidConfiguration("Hartree-Fock not converged".to_string())
        })?;
        let initial_state = self.prepare_hartree_fock_state(hf_result)?;
        let ansatz_circuit = self.create_ansatz_circuit(&initial_state)?;
        let num_parameters = self.get_ansatz_parameter_count(&ansatz_circuit);
        self.vqe_optimizer.initialize_parameters(num_parameters);
        let mut best_energy = std::f64::INFINITY;
        let mut best_state = initial_state;
        let mut iteration = 0;
        while iteration < self.config.vqe_config.max_iterations {
            let parameterized_circuit =
                self.apply_ansatz_parameters(&ansatz_circuit, &self.vqe_optimizer.parameters)?;
            let energy = {
                let hamiltonian = self.hamiltonian.as_ref().ok_or_else(|| {
                    SimulatorError::InvalidConfiguration("Hamiltonian not available".to_string())
                })?;
                self.evaluate_energy_expectation(&parameterized_circuit, hamiltonian)?
            };
            self.vqe_optimizer.history.push(energy);
            if energy < best_energy {
                best_energy = energy;
                best_state = self.get_circuit_final_state(&parameterized_circuit)?;
            }
            if iteration > 0 {
                let energy_change = (energy - self.vqe_optimizer.history[iteration - 1]).abs();
                if energy_change < self.config.vqe_config.energy_threshold {
                    break;
                }
            }
            let hamiltonian_clone = self.hamiltonian.clone().ok_or_else(|| {
                SimulatorError::InvalidConfiguration("Hamiltonian not available".to_string())
            })?;
            self.update_vqe_parameters(&parameterized_circuit, &hamiltonian_clone)?;
            iteration += 1;
        }
        self.stats.vqe_time_ms = vqe_start.elapsed().as_millis() as f64;
        self.stats.circuit_evaluations = iteration;
        Ok(ElectronicStructureResult {
            ground_state_energy: best_energy + nuclear_repulsion,
            molecular_orbitals: hf_molecular_orbitals,
            density_matrix: hf_density_matrix.clone(),
            dipole_moment: self.calculate_dipole_moment(&hf_density_matrix),
            converged: iteration < self.config.vqe_config.max_iterations,
            iterations: iteration,
            quantum_state: best_state,
            vqe_history: self.vqe_optimizer.history.clone(),
            stats: self.stats.clone(),
        })
    }
    pub(super) fn run_quantum_ci(&mut self) -> Result<ElectronicStructureResult> {
        let original_ansatz = self.config.vqe_config.ansatz;
        self.config.vqe_config.ansatz = ChemistryAnsatz::Adaptive;
        let original_threshold = self.config.vqe_config.energy_threshold;
        self.config.vqe_config.energy_threshold = original_threshold * 0.1;
        let result = self.run_vqe();
        self.config.vqe_config.ansatz = original_ansatz;
        self.config.vqe_config.energy_threshold = original_threshold;
        result
    }
    pub(super) fn run_quantum_coupled_cluster(&mut self) -> Result<ElectronicStructureResult> {
        let original_ansatz = self.config.vqe_config.ansatz;
        let original_optimizer = self.config.vqe_config.optimizer;
        self.config.vqe_config.ansatz = ChemistryAnsatz::UCCSD;
        self.config.vqe_config.optimizer = ChemistryOptimizer::Adam;
        let num_orbitals = if let Some(hf) = &self.hartree_fock {
            hf.molecular_orbitals.num_orbitals
        } else {
            4
        };
        let num_singles = num_orbitals * num_orbitals;
        let num_doubles = (num_orbitals * (num_orbitals - 1) / 2).pow(2);
        let total_params = num_singles + num_doubles;
        self.vqe_optimizer.initialize_parameters(total_params);
        let result = self.run_vqe();
        self.config.vqe_config.ansatz = original_ansatz;
        self.config.vqe_config.optimizer = original_optimizer;
        result
    }
}
