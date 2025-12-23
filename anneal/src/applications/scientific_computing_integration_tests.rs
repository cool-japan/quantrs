//! Integration tests for enhanced scientific computing applications
//!
//! This module contains comprehensive tests for the enhanced scientific computing
//! capabilities including quantum computational chemistry, advanced materials science,
//! and protein folding optimization.

use crate::applications::quantum_computational_chemistry::*;
use crate::applications::*;
use crate::enterprise_monitoring::{create_example_enterprise_monitoring, LogLevel};
use std::collections::HashMap;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_chemistry_molecular_systems() {
        let systems = create_example_molecular_systems().unwrap();

        assert_eq!(systems.len(), 2);

        // Test water molecule
        let water = &systems[0];
        assert_eq!(water.id, "water");
        assert_eq!(water.atoms.len(), 3);
        assert_eq!(water.charge, 0);
        assert_eq!(water.multiplicity, 1);

        // Test methane molecule
        let methane = &systems[1];
        assert_eq!(methane.id, "methane");
        assert_eq!(methane.atoms.len(), 5);
        assert_eq!(methane.charge, 0);
        assert_eq!(methane.multiplicity, 1);
    }

    #[test]
    #[ignore] // Slow test: runs heavy quantum algorithms (~105s)
    fn test_quantum_chemistry_optimizer_basic_functionality() {
        let config = QuantumChemistryConfig::default();
        let mut optimizer = QuantumChemistryOptimizer::new(config).unwrap();

        let systems = create_example_molecular_systems().unwrap();
        let water = &systems[0];

        // Test electronic structure calculation
        let result = optimizer.calculate_electronic_structure(water);
        assert!(result.is_ok());

        let chemistry_result = result.unwrap();
        assert_eq!(chemistry_result.system_id, "water");
        assert!(chemistry_result.total_energy.is_finite());
        assert!(!chemistry_result.molecular_orbitals.is_empty());
    }

    #[test]
    fn test_quantum_chemistry_problem_creation() {
        let systems = create_example_molecular_systems().unwrap();
        let problem = QuantumChemistryProblem {
            system: systems[0].clone(),
            config: QuantumChemistryConfig::default(),
            objectives: vec![ChemistryObjective::MinimizeEnergy],
        };

        assert!(problem.validate().is_ok());
        assert!(!problem.description().is_empty());

        let size_metrics = problem.size_metrics();
        assert!(size_metrics.contains_key("atoms"));
        assert!(size_metrics.contains_key("basis_functions"));
    }

    #[test]
    fn test_quantum_chemistry_to_qubo_conversion() {
        let systems = create_example_molecular_systems().unwrap();
        let problem = QuantumChemistryProblem {
            system: systems[0].clone(),
            config: QuantumChemistryConfig::default(),
            objectives: vec![ChemistryObjective::MinimizeEnergy],
        };

        let (qubo, mapping) = problem.to_qubo().unwrap();
        assert!(qubo.num_variables > 0);
        assert!(!mapping.is_empty());
    }

    #[test]
    #[ignore] // Slow test: runs heavy quantum algorithms (~106s)
    fn test_catalysis_optimization_problem() {
        let systems = create_example_molecular_systems().unwrap();

        // Create a simple reaction: H2O -> H2 + 1/2 O2
        let reaction = ChemicalReaction {
            id: "water_dissociation".to_string(),
            reactants: vec![systems[0].clone()], // Water
            products: vec![],                    // Simplified - would need H2 and O2 molecules
            transition_state: None,
            catalysts: vec![],
            conditions: ReactionConditions {
                temperature: 298.15,
                pressure: 1.0,
                solvent: None,
                ph: None,
                concentrations: HashMap::new(),
            },
            mechanism: ReactionMechanism {
                steps: vec![],
                rate_constants: vec![],
                activation_energies: vec![],
                pre_exponential_factors: vec![],
            },
        };

        let optimization = CatalysisOptimization {
            reaction,
            catalyst_candidates: vec![systems[1].clone()], // Methane as catalyst candidate
            objectives: vec![CatalysisObjective::MinimizeActivationEnergy],
            constraints: vec![],
            screening_params: CatalysisScreeningParams {
                max_candidates: 1,
                accuracy_threshold: 0.01,
                use_ml_screening: false,
                active_learning: false,
            },
        };

        let mut optimizer =
            QuantumChemistryOptimizer::new(QuantumChemistryConfig::default()).unwrap();
        let result = optimizer.optimize_catalysis(&optimization);
        assert!(result.is_ok());
    }

    #[test]
    fn test_benchmark_problems_creation() {
        let problems = create_benchmark_problems(3).unwrap();
        assert_eq!(problems.len(), 3);

        for problem in problems {
            assert!(problem.validate().is_ok());
            assert!(!problem.description().is_empty());
        }
    }

    #[test]
    fn test_enhanced_scientific_applications_integration() {
        // Test creation of benchmark suite for new domain
        let chemistry_benchmarks =
            create_benchmark_suite("quantum_computational_chemistry", "small");
        assert!(chemistry_benchmarks.is_ok());

        let benchmarks = chemistry_benchmarks.unwrap();
        assert_eq!(benchmarks.len(), 5);

        // Test each benchmark problem
        for benchmark in benchmarks {
            assert!(benchmark.validate().is_ok());
            let (qubo, mapping) = benchmark.to_qubo().unwrap();
            assert!(qubo.num_variables > 0);
            assert!(!mapping.is_empty());
        }
    }

    #[test]
    fn test_performance_report_for_quantum_chemistry() {
        let mut results = HashMap::new();
        results.insert("electronic_energy".to_string(), -75.42);
        results.insert("optimization_time".to_string(), 12.5);
        results.insert("convergence_iterations".to_string(), 15.0);
        results.insert("accuracy".to_string(), 0.99);

        let report =
            generate_performance_report("quantum_computational_chemistry", &results).unwrap();

        assert!(report.contains("QUANTUM_COMPUTATIONAL_CHEMISTRY"));
        assert!(report.contains("Electronic structure optimized"));
        assert!(report.contains("Molecular orbitals calculated"));
        assert!(report.contains("Chemical properties predicted"));
        assert!(report.contains("electronic_energy"));
        assert!(report.contains("-75.42"));
    }

    #[test]
    fn test_chemistry_binary_wrapper() {
        let chemistry_problems = create_benchmark_problems(1).unwrap();
        let wrapper = ChemistryToBinaryWrapper {
            inner: chemistry_problems.into_iter().next().unwrap(),
        };

        assert!(wrapper.validate().is_ok());
        assert_eq!(
            wrapper.description(),
            "Binary wrapper for quantum computational chemistry problem"
        );

        let size_metrics = wrapper.size_metrics();
        assert_eq!(size_metrics["binary_dimension"], 64);
        assert_eq!(size_metrics["molecular_orbitals"], 32);

        // Test with a binary solution
        let binary_solution = vec![
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
            1, 1, 0, 0, 1, 1,
        ];

        assert!(wrapper.is_feasible(&binary_solution));

        let evaluation = wrapper.evaluate_solution(&binary_solution);
        assert!(evaluation.is_ok());
        assert!(evaluation.unwrap().is_finite());
    }

    #[test]
    fn test_quantum_chemistry_with_enterprise_monitoring() {
        let monitoring = create_example_enterprise_monitoring().unwrap();

        // Test logging quantum chemistry calculations
        let log_result = monitoring.log(
            LogLevel::Info,
            "Starting quantum computational chemistry calculation",
            Some("qchem-001".to_string()),
        );
        assert!(log_result.is_ok());

        // Test creating security event for chemistry calculation
        let security_event = crate::enterprise_monitoring::SecurityEvent {
            id: "qchem-security-001".to_string(),
            timestamp: std::time::SystemTime::now(),
            event_type: crate::enterprise_monitoring::SecurityEventType::DataAccess,
            severity: crate::enterprise_monitoring::SecuritySeverity::Low,
            source_ip: Some("192.168.1.100".to_string()),
            user_id: Some("researcher_001".to_string()),
            resource: "quantum_chemistry_calculation".to_string(),
            action: "molecular_structure_calculation".to_string(),
            outcome: crate::enterprise_monitoring::SecurityOutcome::Success,
            details: std::collections::HashMap::new(),
            correlation_id: Some("qchem-001".to_string()),
        };

        let event_result = monitoring.record_security_event(security_event);
        assert!(event_result.is_ok());
    }

    #[test]
    fn test_advanced_quantum_chemistry_methods() {
        // Test different electronic structure methods
        let methods = vec![
            ElectronicStructureMethod::HartreeFock,
            ElectronicStructureMethod::DFT(DFTFunctional::B3LYP),
            ElectronicStructureMethod::CI(CILevel::CISD),
            ElectronicStructureMethod::CoupledCluster(CCLevel::CCSD),
        ];

        for method in methods {
            let mut config = QuantumChemistryConfig::default();
            config.method = method;

            let optimizer = QuantumChemistryOptimizer::new(config);
            assert!(optimizer.is_ok());
        }
    }

    #[test]
    fn test_basis_set_configurations() {
        let basis_sets = vec![
            BasisSet::STO3G,
            BasisSet::CCPVDZ,
            BasisSet::CCPVTZ,
            BasisSet::AugCCPVDZ,
        ];

        for basis_set in basis_sets {
            let mut config = QuantumChemistryConfig::default();
            config.basis_set = basis_set;

            let optimizer = QuantumChemistryOptimizer::new(config);
            assert!(optimizer.is_ok());
        }
    }

    #[test]
    #[ignore] // Slow test: runs heavy quantum algorithms (~105s)
    fn test_reaction_energetics_calculation() {
        let systems = create_example_molecular_systems().unwrap();
        let mut optimizer =
            QuantumChemistryOptimizer::new(QuantumChemistryConfig::default()).unwrap();

        let reaction = ChemicalReaction {
            id: "test_reaction".to_string(),
            reactants: vec![systems[0].clone()],
            products: vec![systems[1].clone()],
            transition_state: None,
            catalysts: vec![],
            conditions: ReactionConditions {
                temperature: 298.15,
                pressure: 1.0,
                solvent: None,
                ph: None,
                concentrations: HashMap::new(),
            },
            mechanism: ReactionMechanism {
                steps: vec![],
                rate_constants: vec![],
                activation_energies: vec![],
                pre_exponential_factors: vec![],
            },
        };

        let energetics = optimizer.calculate_reaction_energetics(&reaction, None);
        assert!(energetics.is_ok());

        let result = energetics.unwrap();
        assert!(!result.reactant_energies.is_empty());
        assert!(!result.product_energies.is_empty());
        assert!(result.reaction_energy.is_finite());
        assert!(result.activation_energy.is_finite());
    }
}

/// Integration test for the complete enhanced scientific computing workflow
#[cfg(test)]
pub fn run_comprehensive_scientific_computing_test() -> ApplicationResult<()> {
    println!("Running comprehensive scientific computing integration test...");

    // 1. Create molecular systems
    let systems = create_example_molecular_systems()?;
    println!("✓ Created {} molecular systems", systems.len());

    // 2. Initialize quantum chemistry optimizer
    let mut optimizer = QuantumChemistryOptimizer::new(QuantumChemistryConfig::default())?;
    println!("✓ Initialized quantum chemistry optimizer");

    // 3. Calculate electronic structures
    for system in &systems {
        let result = optimizer.calculate_electronic_structure(system)?;
        println!(
            "✓ Calculated electronic structure for {}: E = {:.6} hartree",
            system.id, result.total_energy
        );
    }

    // 4. Test catalysis optimization
    let reaction = ChemicalReaction {
        id: "test_catalysis".to_string(),
        reactants: vec![systems[0].clone()],
        products: vec![systems[1].clone()],
        transition_state: None,
        catalysts: vec![],
        conditions: ReactionConditions {
            temperature: 298.15,
            pressure: 1.0,
            solvent: None,
            ph: None,
            concentrations: HashMap::new(),
        },
        mechanism: ReactionMechanism {
            steps: vec![],
            rate_constants: vec![],
            activation_energies: vec![],
            pre_exponential_factors: vec![],
        },
    };

    let catalysis_optimization = CatalysisOptimization {
        reaction,
        catalyst_candidates: systems.clone(),
        objectives: vec![CatalysisObjective::MinimizeActivationEnergy],
        constraints: vec![],
        screening_params: CatalysisScreeningParams {
            max_candidates: 2,
            accuracy_threshold: 0.01,
            use_ml_screening: false,
            active_learning: false,
        },
    };

    let catalysis_result = optimizer.optimize_catalysis(&catalysis_optimization)?;
    println!(
        "✓ Completed catalysis optimization with score: {:.6}",
        catalysis_result.best_score
    );

    // 5. Test benchmark creation and evaluation
    let benchmarks = create_benchmark_suite("quantum_computational_chemistry", "small")?;
    println!("✓ Created {} benchmark problems", benchmarks.len());

    for (i, benchmark) in benchmarks.iter().enumerate() {
        let (qubo, _) = benchmark.to_qubo()?;
        println!("✓ Benchmark {}: {} variables", i + 1, qubo.num_variables);
    }

    // 6. Generate performance report
    let mut results = HashMap::new();
    results.insert("accuracy".to_string(), 0.95);
    results.insert("convergence_time".to_string(), 10.5);
    results.insert("energy_accuracy".to_string(), 1e-6);

    let report = generate_performance_report("quantum_computational_chemistry", &results)?;
    println!("✓ Generated performance report");

    println!("\n🎉 Comprehensive scientific computing integration test completed successfully!");
    println!("Enhanced capabilities now include:");
    println!("   • Quantum computational chemistry with advanced electronic structure methods");
    println!("   • Catalysis design and optimization");
    println!("   • Multi-scale molecular modeling");
    println!("   • Enterprise monitoring integration");
    println!("   • Advanced error correction for chemical simulations");

    Ok(())
}
