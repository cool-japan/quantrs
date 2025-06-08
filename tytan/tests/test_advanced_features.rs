//! Comprehensive tests for advanced features.

use quantrs2_tytan::*;
use quantrs2_tytan::coherent_ising_machine::*;
use quantrs2_tytan::problem_decomposition::*;
use quantrs2_tytan::solution_debugger::*;
use quantrs2_tytan::performance_profiler::*;
use quantrs2_tytan::testing_framework::*;
use quantrs2_tytan::problem_dsl::*;
use quantrs2_tytan::applications::finance::*;
use quantrs2_tytan::applications::logistics::*;
use std::collections::HashMap;
use ndarray::{Array1, Array2, array};

#[cfg(test)]
mod cim_tests {
    use super::*;

    #[test]
    fn test_basic_cim() {
        let cim = CIMSimulator::new(3)
            .with_pump_parameter(1.5)
            .with_evolution_time(5.0)
            .with_seed(42);
        
        let mut qubo = Array2::zeros((3, 3));
        qubo[[0, 1]] = -1.0;
        qubo[[1, 0]] = -1.0;
        
        let mut var_map = HashMap::new();
        for i in 0..3 {
            var_map.insert(format!("x{}", i), i);
        }
        
        let results = cim.run_qubo(&(qubo, var_map), 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].assignments.len(), 3);
    }

    #[test]
    fn test_advanced_cim_pulse_shaping() {
        let cim = AdvancedCIM::new(4)
            .with_pulse_shape(PulseShape::Gaussian { 
                width: 1.0, 
                amplitude: 2.0 
            })
            .with_num_rounds(2);
        
        assert_eq!(cim.num_rounds, 2);
    }

    #[test]
    fn test_cim_error_correction() {
        let n = 4;
        let mut check_matrix = Array2::from_elem((2, n), false);
        check_matrix[[0, 0]] = true;
        check_matrix[[0, 1]] = true;
        check_matrix[[1, 2]] = true;
        check_matrix[[1, 3]] = true;
        
        let cim = AdvancedCIM::new(n)
            .with_error_correction(ErrorCorrectionScheme::ParityCheck { 
                check_matrix 
            });
        
        // Test that it can be created
        assert_eq!(cim.base_cim.n_spins, n);
    }

    #[test]
    fn test_networked_cim() {
        let net_cim = NetworkedCIM::new(4, 3, NetworkTopology::Ring);
        
        assert_eq!(net_cim.modules.len(), 4);
        
        // Test neighbor calculation
        let neighbors_0 = net_cim.get_neighbors(0);
        assert_eq!(neighbors_0, vec![3, 1]);
        
        let neighbors_2 = net_cim.get_neighbors(2);
        assert_eq!(neighbors_2, vec![1, 3]);
    }

    #[test]
    fn test_bifurcation_control() {
        let control = BifurcationControl {
            initial_param: 0.0,
            final_param: 2.0,
            ramp_time: 10.0,
            ramp_type: RampType::Linear,
        };
        
        let cim = AdvancedCIM::new(3)
            .with_bifurcation_control(control);
        
        // Test parameter computation
        assert_eq!(cim.bifurcation_control.initial_param, 0.0);
        assert_eq!(cim.bifurcation_control.final_param, 2.0);
    }
}

#[cfg(test)]
mod decomposition_tests {
    use super::*;

    #[test]
    fn test_graph_partitioner() {
        let size = 8;
        let mut qubo = Array2::zeros((size, size));
        
        // Create chain structure
        for i in 0..size-1 {
            qubo[[i, i+1]] = -1.0;
            qubo[[i+1, i]] = -1.0;
        }
        
        let partitioner = GraphPartitioner::new()
            .with_num_partitions(2)
            .with_algorithm(PartitioningAlgorithm::KernighanLin);
        
        let partitions = partitioner.partition(&qubo).unwrap();
        assert_eq!(partitions.len(), 2);
        
        // Check that all variables are assigned
        let total_vars: usize = partitions.iter()
            .map(|p| p.variables.len())
            .sum();
        assert_eq!(total_vars, size);
    }

    #[test]
    fn test_hierarchical_solver() {
        let size = 16;
        let mut qubo = Array2::zeros((size, size));
        
        // Add structure
        for i in 0..size {
            qubo[[i, i]] = -1.0;
            if i < size-1 {
                qubo[[i, i+1]] = -0.5;
            }
        }
        
        let solver = HierarchicalSolver::new()
            .with_max_levels(3)
            .with_min_coarse_size(4);
        
        let hierarchy = solver.create_hierarchy(&qubo).unwrap();
        assert!(!hierarchy.is_empty());
        assert!(hierarchy.len() <= 3);
        
        // Check coarsening
        for i in 1..hierarchy.len() {
            assert!(hierarchy[i].size < hierarchy[i-1].size);
        }
    }

    #[test]
    fn test_domain_decomposer() {
        let size = 12;
        let qubo = Array2::random((size, size), ndarray_rand::rand_distr::Uniform::new(-1.0, 1.0));
        
        let decomposer = DomainDecomposer::new()
            .with_method(DecompositionMethod::ADMM)
            .with_num_domains(3)
            .with_overlap(1);
        
        let domains = decomposer.decompose(&qubo).unwrap();
        assert_eq!(domains.len(), 3);
        
        // Check overlap
        for domain in &domains {
            assert!(domain.variables.len() >= size / 3);
        }
    }

    #[test]
    fn test_parallel_coordinator() {
        let coordinator = ParallelCoordinator::new()
            .with_num_threads(4)
            .with_coordination_method(CoordinationMethod::MasterWorker);
        
        // Create test subproblems
        let subproblems = vec![
            Array2::eye(3),
            Array2::eye(3),
            Array2::eye(3),
        ];
        
        // Test coordinator can handle subproblems
        assert_eq!(coordinator.num_threads, 4);
    }
}

#[cfg(test)]
mod debugger_tests {
    use super::*;

    #[test]
    fn test_solution_debugger() {
        let problem_info = create_test_problem_info();
        let config = DebuggerConfig {
            detailed_analysis: true,
            check_constraints: true,
            analyze_energy: true,
            compare_solutions: false,
            generate_visuals: false,
            output_format: DebugOutputFormat::Console,
            verbosity: VerbosityLevel::Normal,
        };
        
        let mut debugger = SolutionDebugger::new(problem_info, config);
        
        let solution = Solution {
            assignments: {
                let mut map = HashMap::new();
                map.insert("x".to_string(), true);
                map.insert("y".to_string(), false);
                map.insert("z".to_string(), true);
                map
            },
            objective_value: -2.0,
            timestamp: None,
            solver: Some("Test".to_string()),
        };
        
        let report = debugger.debug_solution(&solution);
        
        assert!(report.energy_analysis.is_some());
        assert!(report.constraint_analysis.is_some());
        assert_eq!(report.solution.objective_value, -2.0);
    }

    #[test]
    fn test_interactive_debugger() {
        let problem_info = create_test_problem_info();
        let mut debugger = InteractiveDebugger::new(problem_info);
        
        // Test loading solution
        let solution = Solution {
            assignments: {
                let mut map = HashMap::new();
                map.insert("x".to_string(), true);
                map.insert("y".to_string(), true);
                map.insert("z".to_string(), false);
                map
            },
            objective_value: -1.0,
            timestamp: None,
            solver: Some("Test".to_string()),
        };
        
        debugger.load_solution(solution);
        assert!(debugger.current_solution.is_some());
        
        // Test commands
        let output = debugger.execute_command("help");
        assert!(output.contains("Available commands"));
        
        // Test watch variables
        debugger.add_watch("x".to_string());
        assert_eq!(debugger.watch_variables.len(), 1);
        
        // Test breakpoints
        debugger.add_breakpoint(Breakpoint::EnergyThreshold { threshold: -5.0 });
        assert_eq!(debugger.breakpoints.len(), 1);
    }

    #[test]
    fn test_constraint_analyzer() {
        let analyzer = ConstraintAnalyzer::new(1e-6);
        
        let constraint = ConstraintInfo {
            name: "test_one_hot".to_string(),
            constraint_type: ConstraintType::OneHot,
            variables: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            parameters: HashMap::new(),
            penalty: 10.0,
            is_hard: true,
        };
        
        let mut solution = HashMap::new();
        solution.insert("a".to_string(), true);
        solution.insert("b".to_string(), false);
        solution.insert("c".to_string(), false);
        
        let violations = analyzer.analyze(&[constraint.clone()], &solution);
        assert_eq!(violations.len(), 0); // Should satisfy one-hot
        
        // Violate constraint
        solution.insert("b".to_string(), true);
        let violations = analyzer.analyze(&[constraint], &solution);
        assert_eq!(violations.len(), 1);
        assert!(!violations[0].suggested_fixes.is_empty());
    }

    fn create_test_problem_info() -> ProblemInfo {
        let mut qubo = Array2::zeros((3, 3));
        qubo[[0, 0]] = -1.0;
        qubo[[1, 1]] = -1.0;
        qubo[[0, 1]] = 2.0;
        qubo[[1, 0]] = 2.0;
        
        let mut var_map = HashMap::new();
        var_map.insert("x".to_string(), 0);
        var_map.insert("y".to_string(), 1);
        var_map.insert("z".to_string(), 2);
        
        ProblemInfo {
            name: "Test Problem".to_string(),
            problem_type: "QUBO".to_string(),
            num_variables: 3,
            var_map: var_map.clone(),
            reverse_var_map: {
                let mut rev = HashMap::new();
                for (k, v) in &var_map {
                    rev.insert(*v, k.clone());
                }
                rev
            },
            qubo,
            constraints: vec![
                ConstraintInfo {
                    name: "test_constraint".to_string(),
                    constraint_type: ConstraintType::OneHot,
                    variables: vec!["x".to_string(), "y".to_string()],
                    parameters: HashMap::new(),
                    penalty: 10.0,
                    is_hard: true,
                },
            ],
            optimal_solution: None,
            metadata: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod profiler_tests {
    use super::*;

    #[test]
    fn test_performance_profiler() {
        let config = ProfilerConfig {
            enabled: true,
            profile_memory: true,
            profile_cpu: true,
            profile_gpu: false,
            sampling_interval: 10,
            stack_depth: 5,
            output_dir: None,
        };
        
        let mut profiler = PerformanceProfiler::new(config);
        
        profiler.start_profile("test_profile").unwrap();
        
        // Simulate some work
        {
            let _guard = profiler.enter_function("test_function");
            profiler.start_timer("computation");
            std::thread::sleep(std::time::Duration::from_millis(10));
            profiler.stop_timer("computation");
        }
        
        let profile = profiler.stop_profile().unwrap();
        
        assert_eq!(profile.name, "test_profile");
        assert!(!profile.functions.is_empty());
        assert!(profile.total_time > 0.0);
        
        let analysis = profiler.analyze_profile(&profile);
        assert!(analysis.total_time > 0.0);
        assert!(!analysis.hot_paths.is_empty());
    }

    #[test]
    fn test_profiler_macros() {
        let mut profiler = PerformanceProfiler::new(ProfilerConfig::default());
        
        profiler.start_profile("macro_test").unwrap();
        
        profile!(profiler, "test_macro");
        
        time_it!(profiler, "timed_operation", {
            std::thread::sleep(std::time::Duration::from_millis(5));
        });
        
        let profile = profiler.stop_profile().unwrap();
        assert!(profile.functions.len() >= 2);
    }
}

#[cfg(test)]
mod dsl_tests {
    use super::*;

    #[test]
    fn test_dsl_parsing() {
        let mut dsl = ProblemDSL::new();
        
        let code = r#"
            var x[3] binary;
            minimize sum(i in 0..3: x[i]);
            subject to
                x[0] + x[1] >= 1;
        "#;
        
        let ast = dsl.parse(code).unwrap();
        
        match ast {
            AST::Program { declarations, objective, constraints } => {
                assert_eq!(declarations.len(), 1);
                assert!(!constraints.is_empty());
            }
        }
    }

    #[test]
    fn test_dsl_macros() {
        let mut dsl = ProblemDSL::new();
        
        // Define a macro
        dsl.define_macro(
            "one_hot".to_string(),
            vec!["vars".to_string()],
            MacroBody::Text("sum(vars) == 1".to_string()),
        );
        
        assert_eq!(dsl.macros.len(), 1);
    }

    #[test]
    fn test_optimization_hints() {
        let mut dsl = ProblemDSL::new();
        
        // Add hints
        dsl.add_hint(OptimizationHint::VariableOrder(vec![
            "x".to_string(),
            "y".to_string(),
            "z".to_string(),
        ]));
        
        dsl.add_hint(OptimizationHint::Symmetry(SymmetryType::Permutation(vec![
            "a".to_string(),
            "b".to_string(),
        ])));
        
        assert_eq!(dsl.optimization_hints.len(), 2);
    }
}

#[cfg(test)]
mod application_tests {
    use super::*;

    #[test]
    fn test_portfolio_optimizer() {
        let returns = array![0.05, 0.10, 0.15];
        let covariance = array![
            [0.01, 0.002, 0.001],
            [0.002, 0.02, 0.003],
            [0.001, 0.003, 0.03]
        ];
        
        let optimizer = PortfolioOptimizer::new(returns, covariance)
            .with_risk_aversion(2.0)
            .with_constraints(PortfolioConstraints {
                min_investment: 0.0,
                max_investment: 1.0,
                target_return: Some(0.08),
                max_assets: Some(2),
                sector_limits: HashMap::new(),
            });
        
        let (qubo, mapping) = optimizer.to_qubo().unwrap();
        
        assert!(!mapping.is_empty());
        assert_eq!(qubo.shape()[0], qubo.shape()[1]);
    }

    #[test]
    fn test_vrp_problem() {
        let vrp = VehicleRoutingProblem::new()
            .with_depot(Location { x: 0.0, y: 0.0 })
            .with_customers(vec![
                Customer { 
                    location: Location { x: 1.0, y: 0.0 }, 
                    demand: 10.0 
                },
                Customer { 
                    location: Location { x: 0.0, y: 1.0 }, 
                    demand: 15.0 
                },
            ])
            .with_vehicles(1)
            .with_capacity(30.0);
        
        let (qubo, mapping) = vrp.to_qubo().unwrap();
        
        assert!(!mapping.is_empty());
        // Should have variables for routes
        assert!(mapping.len() >= 4); // At least 2 customers × 2 positions
    }
}

#[cfg(test)]
mod testing_framework_tests {
    use super::*;

    #[test]
    fn test_test_generator() {
        let generator = MaxCutGenerator::new(42);
        let test_case = generator.generate(10, Difficulty::Easy).unwrap();
        
        assert_eq!(test_case.size, 10);
        assert_eq!(test_case.difficulty, Difficulty::Easy);
        assert!(!test_case.data.is_empty());
    }

    #[test]
    fn test_testing_framework() {
        let config = TestConfig {
            test_sizes: vec![5, 10],
            difficulties: vec![Difficulty::Easy],
            timeout: 10,
            parallel: false,
        };
        
        let mut framework = TestingFramework::new(config);
        
        framework.add_category(TestCategory {
            name: "Test Category".to_string(),
            problem_types: vec![ProblemType::MaxCut],
            difficulties: vec![Difficulty::Easy],
            tags: vec!["test".to_string()],
        });
        
        framework.generate_suite().unwrap();
        
        let suite = &framework.suite;
        assert!(!suite.test_cases.is_empty());
        assert_eq!(suite.categories.len(), 1);
    }

    #[test]
    fn test_solution_validator() {
        let validator = ConstraintValidator::new();
        
        let mut solution = HashMap::new();
        solution.insert("x".to_string(), true);
        solution.insert("y".to_string(), false);
        
        let constraints = vec![
            ("x".to_string(), "y".to_string(), ConstraintType::AtMostK { k: 1 }),
        ];
        
        let violations = validator.validate(&solution, &constraints);
        assert_eq!(violations.len(), 0); // Should satisfy at-most-1
    }
}