//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::circuit_interfaces::{InterfaceCircuit, InterfaceGate, InterfaceGateType};
use crate::error::{Result, SimulatorError};
use scirs2_core::ndarray::{Array1, Array2, Array4};
use scirs2_core::random::prelude::*;

use super::quantumchemistrysimulator_type::QuantumChemistrySimulator;
use super::types::{
    ChemistryOptimizer, ElectronicStructureConfig, FermionMapper, FermionMapping, Molecule,
    VQEOptimizer,
};

/// Benchmark function for quantum chemistry simulation
pub fn benchmark_quantum_chemistry() -> Result<()> {
    println!("Benchmarking Quantum Chemistry Simulation...");
    let h2_molecule = Molecule {
        atomic_numbers: vec![1, 1],
        positions: Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.4])?,
        charge: 0,
        multiplicity: 1,
        basis_set: "STO-3G".to_string(),
    };
    let config = ElectronicStructureConfig::default();
    let mut simulator = QuantumChemistrySimulator::new(config)?;
    simulator.set_molecule(h2_molecule)?;
    let start_time = std::time::Instant::now();
    let result = simulator.run_calculation()?;
    let duration = start_time.elapsed();
    println!("✅ Quantum Chemistry Results:");
    println!(
        "   Ground State Energy: {:.6} Hartree",
        result.ground_state_energy
    );
    println!("   Converged: {}", result.converged);
    println!("   Iterations: {}", result.iterations);
    println!("   Hamiltonian Terms: {}", result.stats.hamiltonian_terms);
    println!(
        "   Circuit Evaluations: {}",
        result.stats.circuit_evaluations
    );
    println!("   Total Time: {:.2}ms", duration.as_millis());
    println!("   VQE Time: {:.2}ms", result.stats.vqe_time_ms);
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_quantum_chemistry_simulator_creation() {
        let config = ElectronicStructureConfig::default();
        let simulator = QuantumChemistrySimulator::new(config);
        assert!(simulator.is_ok());
    }
    #[test]
    fn test_h2_molecule_creation() {
        let h2 = Molecule {
            atomic_numbers: vec![1, 1],
            positions: Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.4])
                .expect("Failed to create H2 molecule positions array"),
            charge: 0,
            multiplicity: 1,
            basis_set: "STO-3G".to_string(),
        };
        assert_eq!(h2.atomic_numbers, vec![1, 1]);
        assert_eq!(h2.charge, 0);
        assert_eq!(h2.multiplicity, 1);
    }
    #[test]
    fn test_molecular_hamiltonian_construction() {
        let config = ElectronicStructureConfig::default();
        let mut simulator = QuantumChemistrySimulator::new(config)
            .expect("Failed to create quantum chemistry simulator");
        let h2 = Molecule {
            atomic_numbers: vec![1, 1],
            positions: Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.4])
                .expect("Failed to create H2 molecule positions array"),
            charge: 0,
            multiplicity: 1,
            basis_set: "STO-3G".to_string(),
        };
        simulator.set_molecule(h2).expect("Failed to set molecule");
        let molecule_clone = simulator.molecule.clone().expect("Molecule should be set");
        let result = simulator.construct_molecular_hamiltonian(&molecule_clone);
        assert!(result.is_ok());
    }
    #[test]
    fn test_fermion_mapper_creation() {
        let mapper = FermionMapper::new(FermionMapping::JordanWigner, 4);
        assert_eq!(mapper.method, FermionMapping::JordanWigner);
        assert_eq!(mapper.num_spin_orbitals, 4);
    }
    #[test]
    fn test_vqe_optimizer_initialization() {
        let mut optimizer = VQEOptimizer::new(ChemistryOptimizer::GradientDescent);
        optimizer.initialize_parameters(10);
        assert_eq!(optimizer.parameters.len(), 10);
        assert_eq!(optimizer.bounds.len(), 10);
    }
    #[test]
    fn test_ansatz_parameter_counting() {
        let config = ElectronicStructureConfig::default();
        let simulator = QuantumChemistrySimulator::new(config)
            .expect("Failed to create quantum chemistry simulator");
        let mut circuit = InterfaceCircuit::new(4, 0);
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::RY(0.0), vec![0]));
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::RZ(0.0), vec![1]));
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::CNOT, vec![0, 1]));
        let param_count = simulator.get_ansatz_parameter_count(&circuit);
        assert_eq!(param_count, 2);
    }
}
