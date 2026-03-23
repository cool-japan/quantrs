//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::scirs2_integration::SciRS2Backend;
use scirs2_core::random::prelude::*;

use super::types::{
    ChemistryStats, ElectronicStructureConfig, FermionMapper, HartreeFockResult,
    MolecularHamiltonian, Molecule, VQEOptimizer,
};

/// Quantum chemistry simulator
pub struct QuantumChemistrySimulator {
    /// Configuration
    pub(super) config: ElectronicStructureConfig,
    /// `SciRS2` backend for linear algebra
    pub(super) backend: Option<SciRS2Backend>,
    /// Current molecule
    pub(super) molecule: Option<Molecule>,
    /// Molecular Hamiltonian
    pub(super) hamiltonian: Option<MolecularHamiltonian>,
    /// Hartree-Fock solution
    pub(super) hartree_fock: Option<HartreeFockResult>,
    /// Fermion-to-spin mapper
    pub(super) fermion_mapper: FermionMapper,
    /// VQE optimizer
    pub(super) vqe_optimizer: VQEOptimizer,
    /// Computation statistics
    pub(super) stats: ChemistryStats,
}
