//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::prelude::*;

use super::types::{
    BinaryVehicleRoutingProblem, Customer, DistributionCenter, Supplier, SupplyChainNetwork,
    SupplyChainOptimizer, TSPOptimizer, TimeWindow, VehicleRoutingOptimizer, Warehouse,
};

/// Simplified optimization problem trait for binary VRP
pub trait OptimizationProblem {
    type Solution;
    /// Evaluate the objective function
    fn evaluate(&self, solution: &Self::Solution) -> f64;
}
/// Create benchmark problems for testing
pub fn create_benchmark_problems() -> Vec<BinaryVehicleRoutingProblem> {
    let mut problems = Vec::new();
    let small_distances = Array2::from_shape_vec(
        (4, 4),
        vec![
            0.0, 10.0, 15.0, 20.0, 10.0, 0.0, 25.0, 30.0, 15.0, 25.0, 0.0, 35.0, 20.0, 30.0, 35.0,
            0.0,
        ],
    )
    .expect("Small benchmark distance matrix has valid shape");
    let small_demands = Array1::from_vec(vec![0.0, 10.0, 15.0, 20.0]);
    let small_optimizer =
        VehicleRoutingOptimizer::new(small_distances.clone(), 50.0, small_demands.clone(), 2)
            .expect("Small benchmark VRP has valid configuration");
    problems.push(BinaryVehicleRoutingProblem::new(small_optimizer));
    let medium_distances = Array2::from_shape_vec(
        (6, 6),
        vec![
            0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 10.0, 0.0, 25.0, 30.0, 35.0, 40.0, 15.0, 25.0, 0.0,
            35.0, 40.0, 45.0, 20.0, 30.0, 35.0, 0.0, 45.0, 50.0, 25.0, 35.0, 40.0, 45.0, 0.0, 55.0,
            30.0, 40.0, 45.0, 50.0, 55.0, 0.0,
        ],
    )
    .expect("Medium benchmark distance matrix has valid shape");
    let medium_demands = Array1::from_vec(vec![0.0, 12.0, 18.0, 22.0, 16.0, 14.0]);
    let medium_optimizer = VehicleRoutingOptimizer::new(medium_distances, 60.0, medium_demands, 3)
        .expect("Medium benchmark VRP has valid configuration");
    problems.push(BinaryVehicleRoutingProblem::new(medium_optimizer));
    let cvrptw_optimizer = VehicleRoutingOptimizer::new(small_distances, 40.0, small_demands, 2)
        .expect("CVRPTW benchmark VRP has valid configuration")
        .with_time_windows(vec![
            TimeWindow {
                start: 0.0,
                end: 100.0,
                service_time: 5.0,
            },
            TimeWindow {
                start: 10.0,
                end: 50.0,
                service_time: 10.0,
            },
            TimeWindow {
                start: 20.0,
                end: 60.0,
                service_time: 8.0,
            },
            TimeWindow {
                start: 30.0,
                end: 80.0,
                service_time: 12.0,
            },
        ]);
    problems.push(BinaryVehicleRoutingProblem::new(cvrptw_optimizer));
    problems
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_vrp_optimizer() {
        let distances = Array2::from_shape_vec(
            (4, 4),
            vec![
                0.0, 10.0, 15.0, 20.0, 10.0, 0.0, 25.0, 30.0, 15.0, 25.0, 0.0, 35.0, 20.0, 30.0,
                35.0, 0.0,
            ],
        )
        .expect("Test distance matrix has valid shape");
        let demands = Array1::from_vec(vec![0.0, 10.0, 15.0, 20.0]);
        let optimizer = VehicleRoutingOptimizer::new(distances, 50.0, demands, 2)
            .expect("Test VRP optimizer should be created with valid inputs");
        let (_qubo, var_map) = optimizer
            .build_qubo()
            .expect("VRP QUBO should build successfully");
        assert!(!var_map.is_empty());
    }
    #[test]
    fn test_tsp_optimizer() {
        let distances = Array2::from_shape_vec(
            (4, 4),
            vec![
                0.0, 10.0, 15.0, 20.0, 10.0, 0.0, 25.0, 30.0, 15.0, 25.0, 0.0, 35.0, 20.0, 30.0,
                35.0, 0.0,
            ],
        )
        .expect("Test distance matrix has valid shape");
        let optimizer =
            TSPOptimizer::new(distances).expect("TSP optimizer should be created with valid input");
        let (_qubo, var_map) = optimizer
            .build_qubo()
            .expect("TSP QUBO should build successfully");
        assert_eq!(var_map.len(), 16);
    }
    #[test]
    fn test_supply_chain() {
        let network = SupplyChainNetwork {
            suppliers: vec![Supplier {
                id: 0,
                capacity: 100.0,
                cost_per_unit: 10.0,
                lead_time: 2,
                reliability: 0.95,
            }],
            warehouses: vec![Warehouse {
                id: 0,
                capacity: 200.0,
                holding_cost: 1.0,
                fixed_cost: 1000.0,
                location: (0.0, 0.0),
            }],
            distribution_centers: vec![DistributionCenter {
                id: 0,
                capacity: 150.0,
                processing_cost: 2.0,
                location: (10.0, 10.0),
            }],
            customers: vec![Customer {
                id: 0,
                demand: Array1::from_vec(vec![20.0, 25.0, 30.0]),
                priority: 1.0,
                location: (20.0, 20.0),
            }],
            links: vec![],
        };
        let optimizer = SupplyChainOptimizer::new(network, 3);
        let (_qubo, var_map) = optimizer
            .build_qubo()
            .expect("Supply chain QUBO should build successfully");
        assert!(!var_map.is_empty());
    }
    #[test]
    fn test_binary_vrp_wrapper() {
        let distances = Array2::from_shape_vec(
            (4, 4),
            vec![
                0.0, 10.0, 15.0, 20.0, 10.0, 0.0, 25.0, 30.0, 15.0, 25.0, 0.0, 35.0, 20.0, 30.0,
                35.0, 0.0,
            ],
        )
        .expect("Test distance matrix has valid shape");
        let demands = Array1::from_vec(vec![0.0, 10.0, 15.0, 20.0]);
        let optimizer = VehicleRoutingOptimizer::new(distances, 50.0, demands, 2)
            .expect("Test VRP optimizer should be created with valid inputs");
        let binary_vrp = BinaryVehicleRoutingProblem::new(optimizer);
        assert_eq!(binary_vrp.num_variables(), 32);
        let solution = binary_vrp.random_solution();
        assert_eq!(solution.len(), 32);
        let energy = binary_vrp.evaluate_binary(&solution);
        assert!(energy.is_finite());
        let routes = binary_vrp.decode_binary_solution(&solution);
        assert!(routes.len() <= 2);
    }
    #[test]
    fn test_create_benchmark_problems() {
        let problems = create_benchmark_problems();
        assert_eq!(problems.len(), 3);
        for (i, problem) in problems.iter().enumerate() {
            let solution = problem.random_solution();
            let energy = problem.evaluate_binary(&solution);
            assert!(energy.is_finite(), "Problem {i} should have finite energy");
            let routes = problem.decode_binary_solution(&solution);
            assert!(
                routes.len() <= 3,
                "Problem {i} should have at most 3 routes"
            );
        }
    }
}
