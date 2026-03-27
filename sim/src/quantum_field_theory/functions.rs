//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array4};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;
use std::collections::{HashMap, VecDeque};
use std::f64::consts::PI;

use super::types::{
    ActionEvaluator, ActionType, FieldOperator, FieldOperatorType, FieldTheoryType, FixedPointType,
    GaugeFieldConfig, GaugeFixing, GaugeGroup, LatticeParameters, MonteCarloAlgorithm,
    ParticleState, PathIntegralConfig, QFTConfig, QuantumFieldTheorySimulator, ScatteringProcess,
    TimeOrdering,
};

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    #[test]
    fn test_qft_simulator_creation() {
        let config = QFTConfig::default();
        let simulator = QuantumFieldTheorySimulator::new(config);
        assert!(simulator.is_ok());
    }
    #[test]
    fn test_scalar_phi4_configuration() {
        let mut config = QFTConfig::default();
        config.field_theory = FieldTheoryType::ScalarPhi4;
        config.lattice_size = vec![8, 8, 8, 16];
        let simulator = QuantumFieldTheorySimulator::new(config);
        assert!(simulator.is_ok());
        let sim = simulator.expect("ScalarPhi4 simulator should be created successfully");
        assert!(sim.field_configs.contains_key("phi"));
    }
    #[test]
    fn test_qed_configuration() {
        let mut config = QFTConfig::default();
        config.field_theory = FieldTheoryType::QED;
        config.spacetime_dimensions = 4;
        let simulator = QuantumFieldTheorySimulator::new(config);
        assert!(simulator.is_ok());
        let sim = simulator.expect("QED simulator should be created successfully");
        assert!(sim.field_configs.contains_key("psi"));
        assert!(sim.field_configs.contains_key("A_0"));
        assert!(sim.field_configs.contains_key("A_1"));
        assert!(sim.field_configs.contains_key("A_2"));
        assert!(sim.field_configs.contains_key("A_3"));
    }
    #[test]
    fn test_path_integral_initialization() {
        let config = QFTConfig::default();
        let mut simulator = QuantumFieldTheorySimulator::new(config)
            .expect("QFT simulator should be created successfully");
        let pi_config = PathIntegralConfig {
            time_slices: 32,
            time_step: 0.1,
            action_type: ActionType::Euclidean,
            mc_algorithm: MonteCarloAlgorithm::Metropolis,
            importance_sampling: true,
            target_acceptance_rate: 0.5,
        };
        let result = simulator.initialize_path_integral(pi_config);
        assert!(result.is_ok());
        assert!(simulator.path_integral.is_some());
    }
    #[test]
    fn test_gauge_field_setup_u1() {
        let config = QFTConfig::default();
        let mut simulator = QuantumFieldTheorySimulator::new(config)
            .expect("QFT simulator should be created successfully");
        let gauge_config = GaugeFieldConfig {
            gauge_group: GaugeGroup::U1,
            num_colors: 1,
            gauge_coupling: 0.1,
            gauge_fixing: GaugeFixing::Landau,
            wilson_loops: Vec::new(),
        };
        let result = simulator.setup_gauge_fields(gauge_config);
        assert!(result.is_ok());
        assert!(simulator.gauge_configs.is_some());
    }
    #[test]
    fn test_gauge_field_setup_su3() {
        let config = QFTConfig::default();
        let mut simulator = QuantumFieldTheorySimulator::new(config)
            .expect("QFT simulator should be created successfully");
        let gauge_config = GaugeFieldConfig {
            gauge_group: GaugeGroup::SU(3),
            num_colors: 3,
            gauge_coupling: 0.2,
            gauge_fixing: GaugeFixing::Coulomb,
            wilson_loops: Vec::new(),
        };
        let result = simulator.setup_gauge_fields(gauge_config);
        assert!(result.is_ok());
        assert!(simulator.gauge_configs.is_some());
        assert!(simulator.field_configs.contains_key("U_0_00"));
        assert!(simulator.field_configs.contains_key("U_0_11"));
        assert!(simulator.field_configs.contains_key("U_0_22"));
    }
    #[test]
    fn test_wilson_loop_generation() {
        let config = QFTConfig::default();
        let simulator = QuantumFieldTheorySimulator::new(config)
            .expect("QFT simulator should be created successfully");
        let path = simulator
            .generate_wilson_loop_path(2, 3)
            .expect("Wilson loop path generation should succeed");
        assert_eq!(path.len(), 10);
        assert_eq!(path[0], path[path.len() - 1]);
    }
    #[test]
    fn test_action_evaluation_phi4() {
        let config = QFTConfig::default();
        let _simulator = QuantumFieldTheorySimulator::new(config)
            .expect("QFT simulator should be created successfully");
        let lattice_params = LatticeParameters {
            spacing: 1.0,
            bare_mass: 1.0,
            hopping_parameter: 0.125,
            plaquette_size: 1,
        };
        let mut couplings = HashMap::new();
        couplings.insert("lambda".to_string(), 0.1);
        let evaluator = ActionEvaluator {
            theory_type: FieldTheoryType::ScalarPhi4,
            couplings,
            lattice_params,
        };
        let field_config = Array4::zeros((4, 4, 4, 4));
        let action = evaluator.evaluate_action(&field_config);
        assert!(action.is_ok());
        assert_eq!(
            action.expect("Action evaluation should succeed for zero field"),
            0.0
        );
    }
    #[test]
    fn test_beta_function_phi4() {
        let config = QFTConfig::default();
        let simulator = QuantumFieldTheorySimulator::new(config)
            .expect("QFT simulator should be created successfully");
        let mut couplings = HashMap::new();
        couplings.insert("lambda".to_string(), 0.1);
        let beta_lambda = simulator.calculate_beta_function("lambda", 1.0, &couplings);
        assert!(beta_lambda.is_ok());
        let beta_val = beta_lambda.expect("Beta function calculation should succeed");
        assert!(beta_val > 0.0);
    }
    #[test]
    fn test_beta_function_qed() {
        let mut config = QFTConfig::default();
        config.field_theory = FieldTheoryType::QED;
        let simulator = QuantumFieldTheorySimulator::new(config)
            .expect("QED simulator should be created successfully");
        let mut couplings = HashMap::new();
        couplings.insert("e".to_string(), 0.3);
        let beta_e = simulator.calculate_beta_function("e", 1.0, &couplings);
        assert!(beta_e.is_ok());
        let beta_val = beta_e.expect("QED beta function calculation should succeed");
        assert!(beta_val > 0.0);
    }
    #[test]
    fn test_beta_function_yang_mills() {
        let mut config = QFTConfig::default();
        config.field_theory = FieldTheoryType::YangMills;
        let simulator = QuantumFieldTheorySimulator::new(config)
            .expect("Yang-Mills simulator should be created successfully");
        let mut couplings = HashMap::new();
        couplings.insert("g".to_string(), 0.5);
        let beta_g = simulator.calculate_beta_function("g", 1.0, &couplings);
        assert!(beta_g.is_ok());
        let beta_val = beta_g.expect("Yang-Mills beta function calculation should succeed");
        assert!(beta_val < 0.0);
    }
    #[test]
    fn test_rg_flow_analysis() {
        let config = QFTConfig::default();
        let mut simulator = QuantumFieldTheorySimulator::new(config)
            .expect("QFT simulator should be created successfully");
        let energy_scales = [0.1, 1.0, 10.0, 100.0];
        let rg_flow = simulator.analyze_rg_flow(&energy_scales);
        assert!(rg_flow.is_ok());
        let flow = rg_flow.expect("RG flow analysis should succeed");
        assert!(flow.beta_functions.contains_key("lambda"));
        assert!(!flow.fixed_points.is_empty());
        assert!(flow
            .fixed_points
            .iter()
            .any(|fp| fp.fp_type == FixedPointType::Gaussian));
    }
    #[test]
    fn test_scattering_cross_section_phi4() {
        let config = QFTConfig::default();
        let mut simulator = QuantumFieldTheorySimulator::new(config)
            .expect("QFT simulator should be created successfully");
        let process = ScatteringProcess {
            initial_state: vec![
                ParticleState {
                    particle_type: "phi".to_string(),
                    momentum: [1.0, 0.0, 0.0, 0.0],
                    spin_state: vec![Complex64::new(1.0, 0.0)],
                    quantum_numbers: HashMap::new(),
                },
                ParticleState {
                    particle_type: "phi".to_string(),
                    momentum: [-1.0, 0.0, 0.0, 0.0],
                    spin_state: vec![Complex64::new(1.0, 0.0)],
                    quantum_numbers: HashMap::new(),
                },
            ],
            final_state: vec![
                ParticleState {
                    particle_type: "phi".to_string(),
                    momentum: [0.0, 1.0, 0.0, 0.0],
                    spin_state: vec![Complex64::new(1.0, 0.0)],
                    quantum_numbers: HashMap::new(),
                },
                ParticleState {
                    particle_type: "phi".to_string(),
                    momentum: [0.0, -1.0, 0.0, 0.0],
                    spin_state: vec![Complex64::new(1.0, 0.0)],
                    quantum_numbers: HashMap::new(),
                },
            ],
            cms_energy: 2.0,
            scattering_angle: PI / 2.0,
            cross_section: None,
            s_matrix_element: None,
        };
        let cross_section = simulator.calculate_scattering_cross_section(&process);
        assert!(cross_section.is_ok());
        let sigma = cross_section.expect("Cross section calculation should succeed");
        assert!(sigma > 0.0);
        assert!(sigma.is_finite());
    }
    #[test]
    fn test_field_operator_creation() {
        let field_op = FieldOperator {
            field_type: FieldOperatorType::Scalar,
            position: vec![0.0, 0.0, 0.0, 0.0],
            momentum: None,
            component: 0,
            time_ordering: TimeOrdering::TimeOrdered,
            normal_ordering: true,
        };
        assert_eq!(field_op.field_type, FieldOperatorType::Scalar);
        assert_eq!(field_op.position.len(), 4);
        assert!(field_op.normal_ordering);
    }
    #[test]
    #[ignore]
    fn test_path_integral_monte_carlo_short() {
        let config = QFTConfig::default();
        let mut simulator = QuantumFieldTheorySimulator::new(config)
            .expect("QFT simulator should be created successfully");
        let pi_config = PathIntegralConfig {
            time_slices: 8,
            time_step: 0.1,
            action_type: ActionType::Euclidean,
            mc_algorithm: MonteCarloAlgorithm::Metropolis,
            importance_sampling: true,
            target_acceptance_rate: 0.5,
        };
        simulator
            .initialize_path_integral(pi_config)
            .expect("Path integral initialization should succeed");
        let result = simulator.run_path_integral_mc(100);
        assert!(result.is_ok());
        let action_history = result.expect("Path integral MC should complete successfully");
        assert_eq!(action_history.len(), 100);
        assert!(action_history.iter().all(|&a| a.is_finite()));
    }
    #[test]
    #[ignore]
    fn test_correlation_function_calculation() {
        let config = QFTConfig::default();
        let mut simulator = QuantumFieldTheorySimulator::new(config)
            .expect("QFT simulator should be created successfully");
        let pi_config = PathIntegralConfig {
            time_slices: 8,
            time_step: 0.1,
            action_type: ActionType::Euclidean,
            mc_algorithm: MonteCarloAlgorithm::Metropolis,
            importance_sampling: true,
            target_acceptance_rate: 0.5,
        };
        simulator
            .initialize_path_integral(pi_config)
            .expect("Path integral initialization should succeed");
        simulator
            .run_path_integral_mc(50)
            .expect("Path integral MC should complete successfully");
        let operators = vec![
            FieldOperator {
                field_type: FieldOperatorType::Scalar,
                position: vec![0.0, 0.0, 0.0, 0.0],
                momentum: None,
                component: 0,
                time_ordering: TimeOrdering::TimeOrdered,
                normal_ordering: true,
            },
            FieldOperator {
                field_type: FieldOperatorType::Scalar,
                position: vec![1.0, 0.0, 0.0, 0.0],
                momentum: None,
                component: 0,
                time_ordering: TimeOrdering::TimeOrdered,
                normal_ordering: true,
            },
        ];
        let correlation = simulator.calculate_correlation_function(&operators, 4);
        assert!(correlation.is_ok());
        let corr_fn = correlation.expect("Correlation function calculation should succeed");
        assert_eq!(corr_fn.separations.len(), 5);
        assert_eq!(corr_fn.values.len(), 5);
        assert_eq!(corr_fn.errors.len(), 5);
    }
    #[test]
    fn test_export_field_configuration() {
        let config = QFTConfig::default();
        let simulator = QuantumFieldTheorySimulator::new(config)
            .expect("QFT simulator should be created successfully");
        let field_config = simulator.export_field_configuration("phi");
        assert!(field_config.is_ok());
        let config_array = field_config.expect("Field configuration export should succeed");
        assert_eq!(config_array.ndim(), 4);
        let invalid_field = simulator.export_field_configuration("nonexistent");
        assert!(invalid_field.is_err());
    }
    #[test]
    fn test_statistics_tracking() {
        let config = QFTConfig::default();
        let simulator = QuantumFieldTheorySimulator::new(config)
            .expect("QFT simulator should be created successfully");
        let stats = simulator.get_statistics();
        assert_eq!(stats.field_evaluations, 0);
        assert_eq!(stats.pi_samples, 0);
        assert_eq!(stats.correlation_calculations, 0);
        assert_eq!(stats.rg_steps, 0);
    }
}
