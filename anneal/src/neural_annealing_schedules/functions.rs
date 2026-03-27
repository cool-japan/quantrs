//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::ising::{IsingModel, QuboModel};
use scirs2_core::ndarray::{Array1, Array2, Array3};
use scirs2_core::random::prelude::*;
use std::collections::HashMap;

use super::types::{
    ActivationFunction, AnnealingSchedule, NeuralAnnealingScheduler, NeuralSchedulerConfig,
    ScheduleConstraints,
};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_neural_scheduler_creation() {
        let config = NeuralSchedulerConfig::default();
        let scheduler = NeuralAnnealingScheduler::new(config);
        assert!(scheduler.is_ok());
    }
    #[test]
    fn test_problem_encoding() {
        let config = NeuralSchedulerConfig::default();
        let mut scheduler =
            NeuralAnnealingScheduler::new(config).expect("Failed to create scheduler");
        let problem = IsingModel::new(4);
        let features = scheduler.encode_problem(&problem);
        assert!(features.is_ok());
        let feature_vec = features.expect("Failed to encode problem");
        assert_eq!(feature_vec.len(), 128);
    }
    #[test]
    fn test_schedule_generation() {
        let config = NeuralSchedulerConfig::default();
        let mut scheduler =
            NeuralAnnealingScheduler::new(config).expect("Failed to create scheduler");
        let mut problem = IsingModel::new(2);
        problem.set_bias(0, 1.0).expect("Failed to set bias");
        problem
            .set_coupling(0, 1, -0.5)
            .expect("Failed to set coupling");
        let schedule = scheduler.generate_schedule(&problem, None);
        if let Err(e) = &schedule {
            eprintln!("Schedule generation failed with error: {:?}", e);
        }
        assert!(schedule.is_ok());
        let schedule = schedule.expect("Failed to generate schedule");
        assert_eq!(schedule.time_points.len(), 100);
        assert_eq!(schedule.transverse_field.len(), 100);
        assert_eq!(schedule.problem_hamiltonian.len(), 100);
    }
    #[test]
    fn test_activation_functions() {
        let config = NeuralSchedulerConfig::default();
        let scheduler = NeuralAnnealingScheduler::new(config).expect("Failed to create scheduler");
        let input = Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let relu_output = scheduler.apply_activation(&input, &ActivationFunction::ReLU);
        assert_eq!(relu_output, Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 2.0]));
        let sigmoid_output = scheduler.apply_activation(&input, &ActivationFunction::Sigmoid);
        for &val in sigmoid_output.iter() {
            assert!(val > 0.0 && val < 1.0);
        }
    }
    #[test]
    fn test_feature_extraction() {
        let config = NeuralSchedulerConfig::default();
        let scheduler = NeuralAnnealingScheduler::new(config).expect("Failed to create scheduler");
        let mut problem = IsingModel::new(4);
        problem.set_bias(0, 1.0).expect("Failed to set bias");
        problem
            .set_coupling(0, 1, -0.5)
            .expect("Failed to set coupling");
        problem
            .set_coupling(1, 2, 0.3)
            .expect("Failed to set coupling");
        let graph_features = scheduler
            .extract_graph_features(&problem, 20)
            .expect("Failed to extract graph features");
        assert_eq!(graph_features.len(), 20);
        assert_eq!(graph_features[0], 4.0);
        assert_eq!(graph_features[1], 2.0);
        let stat_features = scheduler
            .extract_statistical_features(&problem, 15)
            .expect("Failed to extract statistical features");
        assert_eq!(stat_features.len(), 15);
    }
    #[test]
    fn test_schedule_validation() {
        let config = NeuralSchedulerConfig::default();
        let scheduler = NeuralAnnealingScheduler::new(config).expect("Failed to create scheduler");
        let problem = IsingModel::new(2);
        let valid_schedule = AnnealingSchedule {
            time_points: Array1::linspace(0.0, 1000.0, 10),
            transverse_field: Array1::linspace(1.0, 0.1, 10),
            problem_hamiltonian: Array1::linspace(0.1, 1.0, 10),
            additional_controls: HashMap::new(),
            constraints: ScheduleConstraints::default(),
        };
        assert!(scheduler
            .validate_schedule(&valid_schedule, &problem)
            .is_ok());
        let invalid_schedule = AnnealingSchedule {
            time_points: Array1::linspace(0.0, 1000.0, 3),
            transverse_field: Array1::from_vec(vec![1.0, 0.5, 0.8]),
            problem_hamiltonian: Array1::linspace(0.1, 1.0, 3),
            additional_controls: HashMap::new(),
            constraints: ScheduleConstraints::default(),
        };
        assert!(scheduler
            .validate_schedule(&invalid_schedule, &problem)
            .is_err());
    }
}
