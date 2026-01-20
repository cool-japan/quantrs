//! Test generators for various optimization problems.
//!
//! This module provides test case generators for different problem types
//! including Max-Cut, TSP, Graph Coloring, Knapsack, and industry-specific problems.

use scirs2_core::ndarray::Array2;
use scirs2_core::random::prelude::*;
use scirs2_core::random::SeedableRng;
use std::collections::HashMap;
use std::time::Duration;

use super::types::{
    Constraint, ConstraintType, Difficulty, GeneratorConfig, ProblemType, TestCase, TestGenerator,
    TestMetadata,
};

/// Max-cut problem generator
pub struct MaxCutGenerator;

impl TestGenerator for MaxCutGenerator {
    fn generate(&self, config: &GeneratorConfig) -> Result<Vec<TestCase>, String> {
        let mut rng = if let Some(seed) = config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            let mut thread_rng = thread_rng();
            StdRng::from_rng(&mut thread_rng)
        };

        let mut test_cases = Vec::new();

        // Generate random graph
        let n = config.size;
        let edge_probability = match config.difficulty {
            Difficulty::Easy => 0.3,
            Difficulty::Medium => 0.5,
            Difficulty::Hard => 0.7,
            Difficulty::VeryHard => 0.9,
            Difficulty::Extreme => 0.95,
        };

        let mut qubo = Array2::zeros((n, n));
        let mut var_map = HashMap::new();

        for i in 0..n {
            var_map.insert(format!("x_{i}"), i);
        }

        // Generate edges
        for i in 0..n {
            for j in i + 1..n {
                if rng.random::<f64>() < edge_probability {
                    let weight = rng.random_range(1.0..10.0);
                    // Max-cut: minimize -w_ij * (x_i + x_j - 2*x_i*x_j)
                    qubo[[i, i]] -= weight;
                    qubo[[j, j]] -= weight;
                    qubo[[i, j]] += 2.0 * weight;
                    qubo[[j, i]] += 2.0 * weight;
                }
            }
        }

        test_cases.push(TestCase {
            id: format!("maxcut_{}_{:?}", n, config.difficulty),
            problem_type: ProblemType::MaxCut,
            size: n,
            qubo,
            var_map,
            optimal_solution: None,
            optimal_value: None,
            constraints: Vec::new(),
            metadata: TestMetadata {
                generation_method: "Random graph".to_string(),
                difficulty: config.difficulty.clone(),
                expected_runtime: Duration::from_millis(100),
                notes: format!("Edge probability: {edge_probability}"),
                tags: vec!["graph".to_string(), "maxcut".to_string()],
            },
        });

        Ok(test_cases)
    }

    fn name(&self) -> &'static str {
        "MaxCutGenerator"
    }

    fn supported_types(&self) -> Vec<ProblemType> {
        vec![ProblemType::MaxCut]
    }
}

/// TSP generator
pub struct TSPGenerator;

impl TestGenerator for TSPGenerator {
    fn generate(&self, config: &GeneratorConfig) -> Result<Vec<TestCase>, String> {
        let mut rng = if let Some(seed) = config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            let mut thread_rng = thread_rng();
            StdRng::from_rng(&mut thread_rng)
        };

        let n_cities = config.size;
        let mut test_cases = Vec::new();

        // Generate random city locations
        let mut cities = Vec::new();
        for _ in 0..n_cities {
            cities.push((rng.random_range(0.0..100.0), rng.random_range(0.0..100.0)));
        }

        // Calculate distances
        let mut distances = Array2::zeros((n_cities, n_cities));
        for i in 0..n_cities {
            for j in 0..n_cities {
                if i != j {
                    let dx: f64 = cities[i].0 - cities[j].0;
                    let dy: f64 = cities[i].1 - cities[j].1;
                    distances[[i, j]] = dx.hypot(dy);
                }
            }
        }

        // Create QUBO
        let n_vars = n_cities * n_cities;
        let mut qubo = Array2::zeros((n_vars, n_vars));
        let mut var_map = HashMap::new();

        // Variable mapping: x[i,j] = city i at position j
        for i in 0..n_cities {
            for j in 0..n_cities {
                let idx = i * n_cities + j;
                var_map.insert(format!("x_{i}_{j}"), idx);
            }
        }

        // Objective: minimize total distance
        for i in 0..n_cities {
            for j in 0..n_cities {
                for k in 0..n_cities {
                    let next_j = (j + 1) % n_cities;
                    let idx1 = i * n_cities + j;
                    let idx2 = k * n_cities + next_j;
                    qubo[[idx1, idx2]] += distances[[i, k]];
                }
            }
        }

        // Constraints
        let mut constraints = Vec::new();
        let penalty = 1000.0;

        // Each city visited exactly once
        for i in 0..n_cities {
            let vars: Vec<_> = (0..n_cities).map(|j| format!("x_{i}_{j}")).collect();

            constraints.push(Constraint {
                constraint_type: ConstraintType::ExactlyK { k: 1 },
                variables: vars,
                parameters: HashMap::new(),
                penalty,
            });
        }

        // Each position has exactly one city
        for j in 0..n_cities {
            let vars: Vec<_> = (0..n_cities).map(|i| format!("x_{i}_{j}")).collect();

            constraints.push(Constraint {
                constraint_type: ConstraintType::ExactlyK { k: 1 },
                variables: vars,
                parameters: HashMap::new(),
                penalty,
            });
        }

        // Add constraint penalties to QUBO
        self.add_constraint_penalties(&mut qubo, &var_map, &constraints)?;

        test_cases.push(TestCase {
            id: format!("tsp_{}_{:?}", n_cities, config.difficulty),
            problem_type: ProblemType::TSP,
            size: n_cities,
            qubo,
            var_map,
            optimal_solution: None,
            optimal_value: None,
            constraints,
            metadata: TestMetadata {
                generation_method: "Random cities".to_string(),
                difficulty: config.difficulty.clone(),
                expected_runtime: Duration::from_millis(500),
                notes: format!("{n_cities} cities"),
                tags: vec!["routing".to_string(), "tsp".to_string()],
            },
        });

        Ok(test_cases)
    }

    fn name(&self) -> &'static str {
        "TSPGenerator"
    }

    fn supported_types(&self) -> Vec<ProblemType> {
        vec![ProblemType::TSP]
    }
}

impl TSPGenerator {
    fn add_constraint_penalties(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
        constraints: &[Constraint],
    ) -> Result<(), String> {
        for constraint in constraints {
            if let ConstraintType::ExactlyK { k } = &constraint.constraint_type {
                // (sum x_i - k)^2
                for v1 in &constraint.variables {
                    if let Some(&idx1) = var_map.get(v1) {
                        // Linear term: -2k
                        qubo[[idx1, idx1]] +=
                            constraint.penalty * 2.0f64.mul_add(-(*k as f64), 1.0);

                        // Quadratic terms
                        for v2 in &constraint.variables {
                            if v1 != v2 {
                                if let Some(&idx2) = var_map.get(v2) {
                                    qubo[[idx1, idx2]] += constraint.penalty * 2.0;
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Graph coloring generator
pub struct GraphColoringGenerator;

impl TestGenerator for GraphColoringGenerator {
    fn generate(&self, config: &GeneratorConfig) -> Result<Vec<TestCase>, String> {
        let mut rng = if let Some(seed) = config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            let mut thread_rng = thread_rng();
            StdRng::from_rng(&mut thread_rng)
        };

        let n_vertices = config.size;
        let n_colors = match config.difficulty {
            Difficulty::Easy => 4,
            Difficulty::Medium => 3,
            _ => 3,
        };

        let mut test_cases = Vec::new();

        // Generate random graph
        let edge_prob = 0.3;
        let mut edges = Vec::new();

        for i in 0..n_vertices {
            for j in i + 1..n_vertices {
                if rng.random::<f64>() < edge_prob {
                    edges.push((i, j));
                }
            }
        }

        // Create QUBO
        let n_vars = n_vertices * n_colors;
        let mut qubo = Array2::zeros((n_vars, n_vars));
        let mut var_map = HashMap::new();

        // Variable mapping: x[v,c] = vertex v has color c
        for v in 0..n_vertices {
            for c in 0..n_colors {
                let idx = v * n_colors + c;
                var_map.insert(format!("x_{v}_{c}"), idx);
            }
        }

        // Objective: minimize number of colors used (simplified)
        for v in 0..n_vertices {
            for c in 0..n_colors {
                let idx = v * n_colors + c;
                qubo[[idx, idx]] -= c as f64; // Prefer lower colors
            }
        }

        // Constraints
        let mut constraints = Vec::new();
        let penalty = 100.0;

        // Each vertex has exactly one color
        for v in 0..n_vertices {
            let vars: Vec<_> = (0..n_colors).map(|c| format!("x_{v}_{c}")).collect();

            constraints.push(Constraint {
                constraint_type: ConstraintType::ExactlyK { k: 1 },
                variables: vars,
                parameters: HashMap::new(),
                penalty,
            });
        }

        // Adjacent vertices have different colors
        for (u, v) in &edges {
            for c in 0..n_colors {
                let idx_u = u * n_colors + c;
                let idx_v = v * n_colors + c;
                qubo[[idx_u, idx_v]] += penalty;
                qubo[[idx_v, idx_u]] += penalty;
            }
        }

        test_cases.push(TestCase {
            id: format!(
                "coloring_{}_{}_{}_{:?}",
                n_vertices,
                n_colors,
                edges.len(),
                config.difficulty
            ),
            problem_type: ProblemType::GraphColoring,
            size: n_vertices,
            qubo,
            var_map,
            optimal_solution: None,
            optimal_value: None,
            constraints,
            metadata: TestMetadata {
                generation_method: "Random graph".to_string(),
                difficulty: config.difficulty.clone(),
                expected_runtime: Duration::from_millis(200),
                notes: format!(
                    "{} vertices, {} colors, {} edges",
                    n_vertices,
                    n_colors,
                    edges.len()
                ),
                tags: vec!["graph".to_string(), "coloring".to_string()],
            },
        });

        Ok(test_cases)
    }

    fn name(&self) -> &'static str {
        "GraphColoringGenerator"
    }

    fn supported_types(&self) -> Vec<ProblemType> {
        vec![ProblemType::GraphColoring]
    }
}

/// Knapsack generator
pub struct KnapsackGenerator;

impl TestGenerator for KnapsackGenerator {
    fn generate(&self, config: &GeneratorConfig) -> Result<Vec<TestCase>, String> {
        let mut rng = if let Some(seed) = config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            let mut thread_rng = thread_rng();
            StdRng::from_rng(&mut thread_rng)
        };

        let n_items = config.size;
        let mut test_cases = Vec::new();

        // Generate items
        let mut values = Vec::new();
        let mut weights = Vec::new();

        for _ in 0..n_items {
            values.push(rng.random_range(1.0..100.0));
            weights.push(rng.random_range(1.0..50.0));
        }

        let capacity = weights.iter().sum::<f64>() * 0.5; // 50% of total weight

        // Create QUBO
        let mut qubo = Array2::zeros((n_items, n_items));
        let mut var_map = HashMap::new();

        for i in 0..n_items {
            var_map.insert(format!("x_{i}"), i);
            // Maximize value (negative in minimization)
            qubo[[i, i]] -= values[i];
        }

        // Weight constraint penalty
        let _penalty = values.iter().sum::<f64>() * 2.0;

        // Add soft constraint for capacity
        // Penalty for exceeding capacity: (sum w_i x_i - W)^2 if sum > W
        // This is simplified - proper implementation would use slack variables

        test_cases.push(TestCase {
            id: format!("knapsack_{}_{:?}", n_items, config.difficulty),
            problem_type: ProblemType::Knapsack,
            size: n_items,
            qubo,
            var_map,
            optimal_solution: None,
            optimal_value: None,
            constraints: vec![],
            metadata: TestMetadata {
                generation_method: "Random items".to_string(),
                difficulty: config.difficulty.clone(),
                expected_runtime: Duration::from_millis(100),
                notes: format!("{n_items} items, capacity: {capacity:.1}"),
                tags: vec!["optimization".to_string(), "knapsack".to_string()],
            },
        });

        Ok(test_cases)
    }

    fn name(&self) -> &'static str {
        "KnapsackGenerator"
    }

    fn supported_types(&self) -> Vec<ProblemType> {
        vec![ProblemType::Knapsack]
    }
}

/// Random QUBO generator
pub struct RandomQuboGenerator;

impl TestGenerator for RandomQuboGenerator {
    fn generate(&self, config: &GeneratorConfig) -> Result<Vec<TestCase>, String> {
        let mut rng = if let Some(seed) = config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            let mut thread_rng = thread_rng();
            StdRng::from_rng(&mut thread_rng)
        };

        let n = config.size;
        let mut test_cases = Vec::new();

        // Generate random QUBO
        let mut qubo = Array2::zeros((n, n));
        let density = match config.difficulty {
            Difficulty::Easy => 0.3,
            Difficulty::Medium => 0.5,
            Difficulty::Hard => 0.7,
            Difficulty::VeryHard => 0.9,
            Difficulty::Extreme => 1.0,
        };

        for i in 0..n {
            for j in i..n {
                if rng.random::<f64>() < density {
                    let value = rng.random_range(-10.0..10.0);
                    qubo[[i, j]] = value;
                    if i != j {
                        qubo[[j, i]] = value;
                    }
                }
            }
        }

        let mut var_map = HashMap::new();
        for i in 0..n {
            var_map.insert(format!("x_{i}"), i);
        }

        test_cases.push(TestCase {
            id: format!("random_{}_{:?}", n, config.difficulty),
            problem_type: ProblemType::Custom {
                name: "Random QUBO".to_string(),
            },
            size: n,
            qubo,
            var_map,
            optimal_solution: None,
            optimal_value: None,
            constraints: vec![],
            metadata: TestMetadata {
                generation_method: "Random generation".to_string(),
                difficulty: config.difficulty.clone(),
                expected_runtime: Duration::from_millis(50),
                notes: format!("Density: {density}"),
                tags: vec!["random".to_string(), "qubo".to_string()],
            },
        });

        Ok(test_cases)
    }

    fn name(&self) -> &'static str {
        "RandomQuboGenerator"
    }

    fn supported_types(&self) -> Vec<ProblemType> {
        vec![
            ProblemType::Custom {
                name: "Random".to_string(),
            },
            ProblemType::Ising,
        ]
    }
}

/// Finance industry test generator
pub struct FinanceTestGenerator;

impl TestGenerator for FinanceTestGenerator {
    fn generate(&self, config: &GeneratorConfig) -> Result<Vec<TestCase>, String> {
        let mut rng = if let Some(seed) = config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            let mut thread_rng = thread_rng();
            StdRng::from_rng(&mut thread_rng)
        };

        let n_assets = config.size;
        let mut test_cases = Vec::new();

        // Generate portfolio optimization test case
        let mut qubo = Array2::zeros((n_assets, n_assets));
        let mut var_map = HashMap::new();

        for i in 0..n_assets {
            var_map.insert(format!("asset_{i}"), i);

            // Expected return (negative for minimization)
            let expected_return = rng.random_range(0.05..0.15);
            qubo[[i, i]] -= expected_return;
        }

        // Risk covariance terms
        for i in 0..n_assets {
            for j in 0..n_assets {
                let covariance = if i == j {
                    rng.random_range(0.01..0.04) // Variance
                } else {
                    rng.random_range(-0.01..0.01) // Covariance
                };
                qubo[[i, j]] += covariance;
            }
        }

        test_cases.push(TestCase {
            id: format!("portfolio_{}_{:?}", n_assets, config.difficulty),
            problem_type: ProblemType::Portfolio,
            size: n_assets,
            qubo,
            var_map,
            optimal_solution: None,
            optimal_value: None,
            constraints: vec![Constraint {
                constraint_type: ConstraintType::LinearEquality { target: 1.0 },
                variables: (0..n_assets).map(|i| format!("asset_{i}")).collect(),
                parameters: HashMap::new(),
                penalty: 1000.0,
            }],
            metadata: TestMetadata {
                generation_method: "Random portfolio".to_string(),
                difficulty: config.difficulty.clone(),
                expected_runtime: Duration::from_millis(200),
                notes: format!("{n_assets} assets"),
                tags: vec!["finance".to_string(), "portfolio".to_string()],
            },
        });

        Ok(test_cases)
    }

    fn name(&self) -> &'static str {
        "FinanceTestGenerator"
    }

    fn supported_types(&self) -> Vec<ProblemType> {
        vec![ProblemType::Portfolio]
    }
}

/// Logistics industry test generator
pub struct LogisticsTestGenerator;

impl TestGenerator for LogisticsTestGenerator {
    fn generate(&self, config: &GeneratorConfig) -> Result<Vec<TestCase>, String> {
        let mut rng = if let Some(seed) = config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            let mut thread_rng = thread_rng();
            StdRng::from_rng(&mut thread_rng)
        };

        let n_vehicles = 2;
        let n_locations = config.size;
        let mut test_cases = Vec::new();

        // Generate vehicle routing problem
        let n_vars = n_vehicles * n_locations * n_locations;
        let mut qubo = Array2::zeros((n_vars, n_vars));
        let mut var_map = HashMap::new();

        // Variable mapping: x[v][i][j] = vehicle v goes from i to j
        for v in 0..n_vehicles {
            for i in 0..n_locations {
                for j in 0..n_locations {
                    let idx = v * n_locations * n_locations + i * n_locations + j;
                    var_map.insert(format!("x_{v}_{i}_{j}"), idx);
                }
            }
        }

        // Add distance objective
        for v in 0..n_vehicles {
            for i in 0..n_locations {
                for j in 0..n_locations {
                    if i != j {
                        let idx = v * n_locations * n_locations + i * n_locations + j;
                        let distance = rng.random_range(1.0..20.0);
                        qubo[[idx, idx]] += distance;
                    }
                }
            }
        }

        test_cases.push(TestCase {
            id: format!("vrp_{}_{}_{:?}", n_vehicles, n_locations, config.difficulty),
            problem_type: ProblemType::VRP,
            size: n_locations,
            qubo,
            var_map,
            optimal_solution: None,
            optimal_value: None,
            constraints: vec![],
            metadata: TestMetadata {
                generation_method: "Random VRP".to_string(),
                difficulty: config.difficulty.clone(),
                expected_runtime: Duration::from_millis(500),
                notes: format!("{n_vehicles} vehicles, {n_locations} locations"),
                tags: vec!["logistics".to_string(), "vrp".to_string()],
            },
        });

        Ok(test_cases)
    }

    fn name(&self) -> &'static str {
        "LogisticsTestGenerator"
    }

    fn supported_types(&self) -> Vec<ProblemType> {
        vec![ProblemType::VRP]
    }
}

/// Manufacturing industry test generator
pub struct ManufacturingTestGenerator;

impl TestGenerator for ManufacturingTestGenerator {
    fn generate(&self, config: &GeneratorConfig) -> Result<Vec<TestCase>, String> {
        let mut rng = if let Some(seed) = config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            let mut thread_rng = thread_rng();
            StdRng::from_rng(&mut thread_rng)
        };

        let n_jobs = config.size;
        let n_machines = 3;
        let mut test_cases = Vec::new();

        // Generate job scheduling problem
        let n_vars = n_jobs * n_machines;
        let mut qubo = Array2::zeros((n_vars, n_vars));
        let mut var_map = HashMap::new();

        // Variable mapping: x[j][m] = job j on machine m
        for j in 0..n_jobs {
            for m in 0..n_machines {
                let idx = j * n_machines + m;
                var_map.insert(format!("job_{j}_machine_{m}"), idx);
            }
        }

        // Add processing time objective
        for j in 0..n_jobs {
            for m in 0..n_machines {
                let idx = j * n_machines + m;
                let processing_time = rng.random_range(1.0..10.0);
                qubo[[idx, idx]] += processing_time;
            }
        }

        // Add constraints: each job assigned to exactly one machine
        let mut constraints = Vec::new();
        for j in 0..n_jobs {
            let vars: Vec<_> = (0..n_machines)
                .map(|m| format!("job_{j}_machine_{m}"))
                .collect();
            constraints.push(Constraint {
                constraint_type: ConstraintType::ExactlyK { k: 1 },
                variables: vars,
                parameters: HashMap::new(),
                penalty: 100.0,
            });
        }

        test_cases.push(TestCase {
            id: format!(
                "scheduling_{}_{}_{:?}",
                n_jobs, n_machines, config.difficulty
            ),
            problem_type: ProblemType::JobScheduling,
            size: n_jobs,
            qubo,
            var_map,
            optimal_solution: None,
            optimal_value: None,
            constraints,
            metadata: TestMetadata {
                generation_method: "Random job scheduling".to_string(),
                difficulty: config.difficulty.clone(),
                expected_runtime: Duration::from_millis(300),
                notes: format!("{n_jobs} jobs, {n_machines} machines"),
                tags: vec!["manufacturing".to_string(), "scheduling".to_string()],
            },
        });

        Ok(test_cases)
    }

    fn name(&self) -> &'static str {
        "ManufacturingTestGenerator"
    }

    fn supported_types(&self) -> Vec<ProblemType> {
        vec![ProblemType::JobScheduling]
    }
}

/// Create default generators
pub fn default_generators() -> Vec<Box<dyn TestGenerator>> {
    vec![
        Box::new(MaxCutGenerator),
        Box::new(TSPGenerator),
        Box::new(GraphColoringGenerator),
        Box::new(KnapsackGenerator),
        Box::new(RandomQuboGenerator),
    ]
}
