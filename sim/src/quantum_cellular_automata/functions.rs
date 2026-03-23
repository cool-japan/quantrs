//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;

use super::types::{
    BoundaryConditions, NeighborhoodType, QCAConfig, QCARuleType, QCAUtils,
    QuantumCellularAutomaton,
};

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    #[test]
    fn test_qca_creation() {
        let config = QCAConfig::default();
        let qca = QuantumCellularAutomaton::new(config);
        assert!(qca.is_ok());
    }
    #[test]
    fn test_qca_1d_evolution() {
        let config = QCAConfig {
            dimensions: vec![4],
            evolution_steps: 5,
            ..Default::default()
        };
        let mut qca =
            QuantumCellularAutomaton::new(config).expect("QCA creation should succeed in test");
        let result = qca.evolve(5);
        assert!(result.is_ok());
        let evolution_result = result.expect("Evolution should succeed in test");
        assert_eq!(evolution_result.total_steps, 5);
        assert!(evolution_result.total_time_ms > 0.0);
    }
    #[test]
    fn test_qca_2d_evolution() {
        let config = QCAConfig {
            dimensions: vec![3, 3],
            evolution_steps: 3,
            rule_type: QCARuleType::Partitioned,
            ..Default::default()
        };
        let mut qca =
            QuantumCellularAutomaton::new(config).expect("QCA creation should succeed in test");
        let result = qca.evolve(3);
        assert!(result.is_ok());
    }
    #[test]
    fn test_qca_measurement() {
        let config = QCAConfig {
            dimensions: vec![3],
            ..Default::default()
        };
        let mut qca =
            QuantumCellularAutomaton::new(config).expect("QCA creation should succeed in test");
        let results = qca.measure_cells(&[0, 1, 2]);
        assert!(results.is_ok());
        assert_eq!(
            results.expect("Measurement should succeed in test").len(),
            3
        );
    }
    #[test]
    fn test_boundary_conditions() {
        let configs = vec![
            BoundaryConditions::Periodic,
            BoundaryConditions::Fixed,
            BoundaryConditions::Open,
            BoundaryConditions::Reflective,
        ];
        for boundary in configs {
            let config = QCAConfig {
                dimensions: vec![4],
                boundary_conditions: boundary,
                evolution_steps: 2,
                ..Default::default()
            };
            let mut qca =
                QuantumCellularAutomaton::new(config).expect("QCA creation should succeed in test");
            let result = qca.evolve(2);
            assert!(result.is_ok());
        }
    }
    #[test]
    fn test_neighborhood_types() {
        let neighborhoods = vec![NeighborhoodType::VonNeumann, NeighborhoodType::Moore];
        for neighborhood in neighborhoods {
            let config = QCAConfig {
                dimensions: vec![3, 3],
                neighborhood,
                evolution_steps: 2,
                ..Default::default()
            };
            let mut qca =
                QuantumCellularAutomaton::new(config).expect("QCA creation should succeed in test");
            let result = qca.evolve(2);
            assert!(result.is_ok());
        }
    }
    #[test]
    fn test_predefined_configs() {
        let config_types = vec!["game_of_life", "elementary_ca", "margolus_ca"];
        for config_type in config_types {
            let config = QCAUtils::create_predefined_config(config_type, 4);
            let qca = QuantumCellularAutomaton::new(config);
            assert!(qca.is_ok());
        }
    }
    #[test]
    fn test_initial_patterns() {
        let dimensions = vec![4, 4];
        let patterns = vec!["random", "glider", "uniform"];
        for pattern in patterns {
            let state = QCAUtils::create_initial_pattern(pattern, &dimensions);
            assert_eq!(state.len(), 1 << 16);
            let norm: f64 = state.iter().map(|x| x.norm_sqr()).sum();
            assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-10);
        }
    }
    #[test]
    fn test_entanglement_entropy_calculation() {
        let config = QCAConfig {
            dimensions: vec![4],
            ..Default::default()
        };
        let mut qca =
            QuantumCellularAutomaton::new(config).expect("QCA creation should succeed in test");
        let state_size = qca.state.len();
        qca.state = Array1::zeros(state_size);
        qca.state[0] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
        qca.state[15] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
        let entropy = qca
            .calculate_entanglement_entropy()
            .expect("Entropy calculation should succeed in test");
        assert!(entropy >= 0.0);
    }
    #[test]
    fn test_local_observables() {
        let config = QCAConfig {
            dimensions: vec![3],
            ..Default::default()
        };
        let qca =
            QuantumCellularAutomaton::new(config).expect("QCA creation should succeed in test");
        let observables = qca
            .calculate_local_observables()
            .expect("Observable calculation should succeed in test");
        assert!(observables.contains_key("magnetization_0"));
        assert!(observables.contains_key("magnetization_1"));
        assert!(observables.contains_key("magnetization_2"));
    }
    #[test]
    fn test_unitary_check() {
        let cnot = QuantumCellularAutomaton::create_cnot_unitary();
        assert!(QuantumCellularAutomaton::is_unitary(&cnot));
        let rotation =
            QuantumCellularAutomaton::create_rotation_unitary(std::f64::consts::PI / 4.0);
        assert!(QuantumCellularAutomaton::is_unitary(&rotation));
    }
}
