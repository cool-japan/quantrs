//! Comprehensive tests for enhanced industry applications.

use ndarray::{Array1, Array2};
use quantrs2_tytan::applications::drug_discovery::*;
use quantrs2_tytan::applications::materials::*;
use quantrs2_tytan::applications::finance::*;
use quantrs2_tytan::applications::logistics::*;
use quantrs2_tytan::applications::ml_tools::*;
use quantrs2_tytan::sampler::{SASampler, Sampler};
use std::collections::HashMap;

#[test]
fn test_drug_discovery_molecular_design() {
    let config = MolecularDesignConfig {
        target_properties: TargetProperties {
            molecular_weight: Some((200.0, 500.0)),
            logp: Some((1.0, 4.0)),
            tpsa: Some((20.0, 140.0)),
            num_rotatable_bonds: Some((0, 10)),
            num_hbd: Some((0, 5)),
            num_hba: Some((0, 10)),
            custom_properties: HashMap::new(),
        },
        fragment_library: FragmentLibrary {
            fragments: vec![
                Fragment {
                    id: "frag1".to_string(),
                    smiles: "CCO".to_string(),
                    molecular_weight: 46.07,
                    properties: FragmentProperties {
                        logp: -0.31,
                        tpsa: 20.23,
                        num_rotatable_bonds: 1,
                        num_hbd: 1,
                        num_hba: 1,
                        reactivity: FragmentReactivity::Moderate,
                        toxicity_score: 0.1,
                        synthesis_complexity: 1.0,
                    },
                    connection_points: vec![ConnectionPoint {
                        atom_index: 0,
                        connection_type: ConnectionType::Single,
                        available: true,
                    }],
                },
                Fragment {
                    id: "frag2".to_string(),
                    smiles: "C1=CC=CC=C1".to_string(),
                    molecular_weight: 78.11,
                    properties: FragmentProperties {
                        logp: 2.13,
                        tpsa: 0.0,
                        num_rotatable_bonds: 0,
                        num_hbd: 0,
                        num_hba: 0,
                        reactivity: FragmentReactivity::Low,
                        toxicity_score: 0.2,
                        synthesis_complexity: 2.0,
                    },
                    connection_points: vec![ConnectionPoint {
                        atom_index: 0,
                        connection_type: ConnectionType::Single,
                        available: true,
                    }],
                },
            ],
            compatibility_matrix: Array2::from_shape_vec((2, 2), vec![1.0, 0.8, 0.8, 1.0]).unwrap(),
        },
        optimization_config: OptimizationConfig {
            max_fragments: 5,
            target_molecular_weight: 300.0,
            property_weights: PropertyWeights {
                molecular_weight: 1.0,
                logp: 1.0,
                tpsa: 1.0,
                drug_likeness: 2.0,
                synthesis_complexity: 0.5,
                novelty: 1.0,
            },
            constraints: OptimizationConstraints {
                lipinski_rule: true,
                veber_rule: true,
                pains_filter: true,
                custom_constraints: Vec::new(),
            },
        },
        sampler_config: SamplerConfig {
            sampler_type: SamplerType::SimulatedAnnealing,
            num_samples: 100,
            temperature_schedule: TemperatureSchedule::Exponential { start: 10.0, decay: 0.95 },
        },
    };

    let designer = MolecularDesigner::new(config);
    
    let result = designer.design_molecules(10);
    assert!(result.is_ok());
    
    let molecules = result.unwrap();
    assert!(!molecules.is_empty());
    
    // Verify molecules meet basic constraints
    for molecule in &molecules {
        assert!(molecule.fragments.len() <= 5);
        assert!(molecule.molecular_weight >= 200.0 && molecule.molecular_weight <= 500.0);
        assert!(molecule.drug_likeness_score >= 0.0 && molecule.drug_likeness_score <= 1.0);
    }
    
    println!("Generated {} drug candidate molecules", molecules.len());
}

#[test]
fn test_materials_crystal_structure_prediction() {
    let config = CrystalPredictionConfig {
        composition: ChemicalComposition {
            elements: vec![
                ("Si".to_string(), 1),
                ("O".to_string(), 2),
            ],
            charge_neutrality: true,
            max_formula_units: 4,
        },
        space_group_constraints: SpaceGroupConstraints {
            allowed_space_groups: vec![225, 227, 229], // Common cubic space groups
            crystal_system: Some(CrystalSystem::Cubic),
            point_group: None,
        },
        structure_constraints: StructureConstraints {
            min_bond_length: 1.0,
            max_bond_length: 3.0,
            coordination_constraints: HashMap::new(),
            symmetry_tolerance: 0.1,
        },
        optimization_config: StructureOptimizationConfig {
            energy_model: EnergyModel::ClassicalPotential {
                potential_type: PotentialType::LennardJones,
                parameters: HashMap::new(),
            },
            lattice_optimization: true,
            atomic_relaxation: true,
            max_iterations: 500,
            convergence_threshold: 1e-6,
        },
        search_config: StructureSearchConfig {
            num_generations: 50,
            population_size: 20,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            diversity_threshold: 0.05,
        },
    };

    let predictor = CrystalStructurePredictor::new(config);
    
    let result = predictor.predict_structures(5);
    assert!(result.is_ok());
    
    let structures = result.unwrap();
    assert!(!structures.is_empty());
    
    // Verify structures have reasonable properties
    for structure in &structures {
        assert!(structure.lattice_parameters.a > 0.0);
        assert!(structure.lattice_parameters.b > 0.0);
        assert!(structure.lattice_parameters.c > 0.0);
        assert!(structure.space_group >= 1 && structure.space_group <= 230);
        assert!(!structure.atomic_positions.is_empty());
        assert!(structure.energy < 0.0); // Should be negative for stable structures
    }
    
    println!("Predicted {} crystal structures", structures.len());
}

#[test]
fn test_finance_portfolio_optimization() {
    let config = PortfolioConfig {
        assets: vec![
            AssetInfo {
                symbol: "AAPL".to_string(),
                name: "Apple Inc.".to_string(),
                sector: "Technology".to_string(),
                expected_return: 0.12,
                volatility: 0.25,
                liquidity_score: 0.95,
                esg_score: Some(0.85),
                market_cap: 2500e9,
            },
            AssetInfo {
                symbol: "GOOGL".to_string(),
                name: "Alphabet Inc.".to_string(),
                sector: "Technology".to_string(),
                expected_return: 0.15,
                volatility: 0.30,
                liquidity_score: 0.90,
                esg_score: Some(0.80),
                market_cap: 1800e9,
            },
            AssetInfo {
                symbol: "JPM".to_string(),
                name: "JPMorgan Chase".to_string(),
                sector: "Finance".to_string(),
                expected_return: 0.10,
                volatility: 0.20,
                liquidity_score: 0.88,
                esg_score: Some(0.70),
                market_cap: 450e9,
            },
        ],
        correlation_matrix: Array2::from_shape_vec(
            (3, 3),
            vec![
                1.0, 0.6, 0.3,
                0.6, 1.0, 0.4,
                0.3, 0.4, 1.0,
            ],
        ).unwrap(),
        constraints: PortfolioConstraints {
            max_weight_per_asset: 0.4,
            min_weight_per_asset: 0.05,
            sector_limits: {
                let mut limits = HashMap::new();
                limits.insert("Technology".to_string(), 0.6);
                limits.insert("Finance".to_string(), 0.3);
                limits
            },
            turnover_limit: Some(0.1),
            tracking_error_limit: None,
            liquidity_requirement: 0.8,
        },
        optimization_objective: OptimizationObjective::MaxSharpeRatio {
            risk_free_rate: 0.02,
            target_return: None,
        },
        risk_model: RiskModel::MeanVariance {
            lookback_period: 252,
            shrinkage_target: Some(ShrinkageTarget::Identity),
        },
    };

    let optimizer = PortfolioOptimizer::new(config);
    
    let result = optimizer.optimize_portfolio(1000000.0); // $1M portfolio
    assert!(result.is_ok());
    
    let portfolio = result.unwrap();
    
    // Verify portfolio constraints
    let total_weight: f64 = portfolio.weights.iter().sum();
    assert!((total_weight - 1.0).abs() < 1e-6);
    
    for &weight in &portfolio.weights {
        assert!(weight >= 0.05 - 1e-6); // Min weight constraint
        assert!(weight <= 0.4 + 1e-6);  // Max weight constraint
    }
    
    assert!(portfolio.expected_return > 0.0);
    assert!(portfolio.volatility > 0.0);
    assert!(portfolio.sharpe_ratio > 0.0);
    
    println!("Optimized portfolio: expected return = {:.2}%, volatility = {:.2}%, Sharpe = {:.2}",
             portfolio.expected_return * 100.0,
             portfolio.volatility * 100.0,
             portfolio.sharpe_ratio);
}

#[test]
fn test_logistics_vehicle_routing() {
    let config = VRPConfig {
        vehicles: vec![
            Vehicle {
                id: "truck1".to_string(),
                capacity: 1000.0,
                max_distance: 500.0,
                cost_per_km: 2.0,
                depot_location: (0.0, 0.0),
                available_time_window: (0, 480), // 8 hours in minutes
                vehicle_type: VehicleType::Truck,
                special_requirements: Vec::new(),
            },
            Vehicle {
                id: "truck2".to_string(),
                capacity: 800.0,
                max_distance: 400.0,
                cost_per_km: 1.8,
                depot_location: (0.0, 0.0),
                available_time_window: (0, 480),
                vehicle_type: VehicleType::Truck,
                special_requirements: Vec::new(),
            },
        ],
        customers: vec![
            Customer {
                id: "cust1".to_string(),
                location: (10.0, 5.0),
                demand: 100.0,
                time_window: (60, 120), // 1-2 hours
                service_time: 15,
                priority: Priority::High,
                customer_type: CustomerType::Commercial,
                special_requirements: Vec::new(),
            },
            Customer {
                id: "cust2".to_string(),
                location: (15.0, 10.0),
                demand: 150.0,
                time_window: (90, 180),
                service_time: 20,
                priority: Priority::Medium,
                customer_type: CustomerType::Residential,
                special_requirements: Vec::new(),
            },
            Customer {
                id: "cust3".to_string(),
                location: (5.0, 15.0),
                demand: 80.0,
                time_window: (120, 240),
                service_time: 10,
                priority: Priority::Low,
                customer_type: CustomerType::Commercial,
                special_requirements: Vec::new(),
            },
        ],
        distance_matrix: Array2::from_shape_vec(
            (4, 4), // 1 depot + 3 customers
            vec![
                0.0, 11.2, 18.0, 15.8,
                11.2, 0.0, 7.1, 14.1,
                18.0, 7.1, 0.0, 11.2,
                15.8, 14.1, 11.2, 0.0,
            ],
        ).unwrap(),
        constraints: VRPConstraints {
            max_route_duration: 240, // 4 hours
            max_waiting_time: 30,
            enforce_time_windows: true,
            capacity_constraints: true,
            driver_break_requirements: Vec::new(),
        },
        optimization_objective: VRPObjective::MinimizeTotalCost {
            distance_weight: 1.0,
            time_weight: 0.5,
            vehicle_cost_weight: 0.3,
        },
    };

    let solver = VRPSolver::new(config);
    
    let result = solver.solve();
    assert!(result.is_ok());
    
    let solution = result.unwrap();
    
    // Verify solution constraints
    assert_eq!(solution.routes.len(), 2); // Two vehicles
    
    for route in &solution.routes {
        assert!(!route.customers.is_empty());
        
        // Check capacity constraint
        let total_demand: f64 = route.customers.iter()
            .map(|c| solver.get_customer_demand(c))
            .sum();
        let vehicle_capacity = solver.get_vehicle_capacity(&route.vehicle_id);
        assert!(total_demand <= vehicle_capacity);
        
        // Check time constraint
        assert!(route.total_duration <= 240);
    }
    
    assert!(solution.total_cost > 0.0);
    assert!(solution.total_distance > 0.0);
    
    println!("VRP solution: {} routes, total cost = {:.2}, total distance = {:.2}",
             solution.routes.len(),
             solution.total_cost,
             solution.total_distance);
}

#[test]
fn test_ml_quantum_feature_selection() {
    let config = QuantumFeatureSelectionConfig {
        feature_interaction_depth: 2,
        entanglement_strength: 0.5,
        measurement_shots: 1000,
        optimization_method: QuantumOptimizationMethod::QAOA {
            num_layers: 3,
            mixer_angles: vec![0.1, 0.2, 0.3],
            cost_angles: vec![0.4, 0.5, 0.6],
        },
        classical_preprocessing: ClassicalPreprocessing {
            scaling: Some(ScalingMethod::StandardScaler),
            dimensionality_reduction: Some(DimensionalityReduction::PCA { n_components: 50 }),
            correlation_threshold: 0.95,
        },
        selection_criteria: SelectionCriteria {
            max_features: 20,
            min_importance_score: 0.1,
            stability_threshold: 0.8,
            diversity_penalty: 0.1,
        },
    };

    let selector = QuantumFeatureSelector::new(config);
    
    // Create synthetic feature data
    let num_samples = 100;
    let num_features = 30;
    
    let mut features = Array2::zeros((num_samples, num_features));
    let mut target = Array1::zeros(num_samples);
    
    // Fill with synthetic data
    for i in 0..num_samples {
        for j in 0..num_features {
            features[[i, j]] = (i as f64 * 0.1 + j as f64 * 0.01) % 1.0;
        }
        target[i] = if i % 2 == 0 { 1.0 } else { 0.0 };
    }
    
    let result = selector.select_features(&features, &target);
    assert!(result.is_ok());
    
    let selection_result = result.unwrap();
    
    // Verify selection results
    assert!(!selection_result.selected_features.is_empty());
    assert!(selection_result.selected_features.len() <= 20);
    
    for &feature_idx in &selection_result.selected_features {
        assert!(feature_idx < num_features);
    }
    
    assert!(selection_result.importance_scores.len() == num_features);
    assert!(selection_result.stability_score >= 0.0 && selection_result.stability_score <= 1.0);
    
    println!("Selected {} features out of {} with stability score {:.3}",
             selection_result.selected_features.len(),
             num_features,
             selection_result.stability_score);
}

#[test]
fn test_ml_quantum_hyperparameter_optimization() {
    let config = QuantumHyperparameterConfig {
        parameter_space: vec![
            HyperparameterDimension {
                name: "learning_rate".to_string(),
                param_type: ParameterType::Continuous { min: 0.001, max: 0.1 },
                encoding: QuantumEncoding::Amplitude { num_qubits: 4 },
                optimization_priority: 1.0,
            },
            HyperparameterDimension {
                name: "batch_size".to_string(),
                param_type: ParameterType::Categorical { 
                    options: vec![16, 32, 64, 128, 256].into_iter().map(|x| x.into()).collect()
                },
                encoding: QuantumEncoding::Basis { num_qubits: 3 },
                optimization_priority: 0.8,
            },
            HyperparameterDimension {
                name: "num_layers".to_string(),
                param_type: ParameterType::Integer { min: 1, max: 10 },
                encoding: QuantumEncoding::Basis { num_qubits: 4 },
                optimization_priority: 0.9,
            },
        ],
        quantum_circuit_config: QuantumCircuitConfig {
            ansatz_type: AnsatzType::RealAmplitudes { num_layers: 3 },
            entanglement_pattern: EntanglementPattern::Linear,
            parameter_initialization: ParameterInitialization::Random { seed: Some(42) },
        },
        optimization_strategy: OptimizationStrategy::VariationalQuantumEigensolver {
            max_iterations: 100,
            convergence_threshold: 1e-6,
            optimizer: ClassicalOptimizer::COBYLA,
        },
        evaluation_config: EvaluationConfig {
            cross_validation_folds: 5,
            scoring_metric: ScoringMetric::Accuracy,
            early_stopping: Some(EarlyStoppingConfig {
                patience: 10,
                min_delta: 1e-4,
            }),
        },
    };

    let optimizer = QuantumHyperparameterOptimizer::new(config);
    
    // Mock objective function
    let objective = |params: &HashMap<String, f64>| -> Result<f64, String> {
        let lr = params.get("learning_rate").unwrap_or(&0.01);
        let batch_size = params.get("batch_size").unwrap_or(&32.0);
        let num_layers = params.get("num_layers").unwrap_or(&3.0);
        
        // Simple mock scoring function
        let score = 0.8 + 0.1 * (1.0 - (lr - 0.01).abs()) + 
                   0.05 * (1.0 - (batch_size - 64.0).abs() / 100.0) +
                   0.05 * (1.0 - (num_layers - 5.0).abs() / 5.0);
        
        Ok(score.min(1.0))
    };
    
    let result = optimizer.optimize(objective, 20); // 20 evaluations
    assert!(result.is_ok());
    
    let optimization_result = result.unwrap();
    
    // Verify optimization results
    assert!(optimization_result.best_parameters.contains_key("learning_rate"));
    assert!(optimization_result.best_parameters.contains_key("batch_size"));
    assert!(optimization_result.best_parameters.contains_key("num_layers"));
    
    assert!(optimization_result.best_score >= 0.0 && optimization_result.best_score <= 1.0);
    assert!(optimization_result.convergence_history.len() <= 20);
    
    println!("Best hyperparameters found with score {:.3}: {:?}",
             optimization_result.best_score,
             optimization_result.best_parameters);
}