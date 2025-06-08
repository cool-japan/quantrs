//! Demonstration of penalty function optimization with SciRS2
//!
//! This example shows how to use the penalty optimization module
//! to automatically tune penalty weights for constrained QUBO problems.

use quantrs2_tytan::optimization::prelude::*;
use quantrs2_tytan::sampler::{SASampler, Sampler};
use quantrs2_tytan::compile::Compile;
use quantrs2_tytan::symbol::{symbols, Expression};
use std::collections::HashMap;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== QuantRS2 Penalty Function Optimization Demo ===\n");
    
    // Example 1: Quadratic penalty optimization
    quadratic_penalty_demo()?;
    println!();
    
    // Example 2: Adaptive penalty strategy
    adaptive_penalty_demo()?;
    println!();
    
    // Example 3: Parameter tuning with constraints
    parameter_tuning_demo()?;
    
    Ok(())
}

/// Demonstrate quadratic penalty optimization
fn quadratic_penalty_demo() -> Result<(), Box<dyn Error>> {
    println!("1. Quadratic Penalty Optimization");
    println!("   Problem: Minimize x^2 + y^2 subject to x + y = 1");
    
    // Define symbolic variables
    let x = symbols("x");
    let y = symbols("y");
    
    // Objective function: x^2 + y^2
    let objective = x.pow(2) + y.pow(2);
    
    // Constraint: x + y = 1
    let constraint = x + y - 1;
    
    // Initial penalty weight
    let initial_penalty = 10.0;
    
    // Combined expression with penalty
    let h = objective + initial_penalty * constraint.pow(2);
    
    // Compile to QUBO
    let model = Compile::new(&h);
    let (qubo, _offset) = model.get_qubo()?;
    
    // Create penalty optimizer
    let config = PenaltyConfig {
        initial_weight: 1.0,
        min_weight: 0.1,
        max_weight: 100.0,
        adjustment_factor: 1.5,
        violation_tolerance: 1e-4,
        max_iterations: 20,
        adaptive_scaling: true,
        penalty_type: PenaltyType::Quadratic,
    };
    
    let mut penalty_optimizer = PenaltyOptimizer::new(config);
    penalty_optimizer.initialize_weights(&["equality_constraint".to_string()]);
    
    // Run initial sampling
    let mut sampler = SASampler::new(None);
    let samples = sampler.run_qubo(&qubo, 100)?;
    
    // Convert to format for penalty optimization
    let sample_results: Vec<(Vec<bool>, f64)> = samples.iter()
        .map(|s| {
            let assignment: Vec<bool> = (0..2)
                .map(|i| *s.assignments.get(&format!("x_{}", i)).unwrap_or(&false))
                .collect();
            (assignment, s.energy)
        })
        .collect();
    
    // Mock compiled model for demonstration
    let compiled_model = MockCompiledModel::new();
    
    // Optimize penalties
    let result = penalty_optimizer.optimize_penalties(
        &compiled_model,
        &sample_results,
    )?;
    
    println!("   Optimization Results:");
    println!("   - Converged: {}", result.converged);
    println!("   - Iterations: {}", result.iterations);
    println!("   - Optimal penalty weight: {:.3}", 
        result.optimal_weights.get("equality_constraint").unwrap_or(&0.0));
    println!("   - Final constraint violation: {:.6}", 
        result.final_violations.get("equality_constraint").unwrap_or(&0.0));
    println!("   - Constraint satisfaction: {:.1}%", result.constraint_satisfaction * 100.0);
    
    Ok(())
}

/// Demonstrate adaptive penalty strategies
fn adaptive_penalty_demo() -> Result<(), Box<dyn Error>> {
    println!("2. Adaptive Penalty Strategy Demo");
    println!("   Testing multiple adaptive strategies for constraint handling");
    
    // Define a more complex problem
    let x = symbols("x");
    let y = symbols("y");
    let z = symbols("z");
    
    // Objective: minimize x + 2y + 3z
    let objective = x + 2.0 * y + 3.0 * z;
    
    // Constraints:
    // 1. x + y + z = 1 (one-hot)
    // 2. y + z <= 1
    let constraint1 = (x + y + z - 1).pow(2);
    let constraint2 = (y + z).pow(2); // Simplified inequality
    
    // Initial parameters
    let initial_params = HashMap::from([
        ("T_0".to_string(), 10.0),
        ("T_f".to_string(), 0.01),
        ("steps".to_string(), 1000.0),
    ]);
    
    let initial_penalties = HashMap::from([
        ("one_hot".to_string(), 5.0),
        ("inequality".to_string(), 3.0),
    ]);
    
    // Test different adaptive strategies
    let strategies = vec![
        AdaptiveStrategy::ExponentialDecay,
        AdaptiveStrategy::AdaptivePenaltyMethod,
        AdaptiveStrategy::AugmentedLagrangian,
    ];
    
    for strategy in strategies {
        println!("\n   Strategy: {:?}", strategy);
        
        let config = AdaptiveConfig {
            strategy,
            update_interval: 5,
            learning_rate: 0.1,
            momentum: 0.9,
            patience: 10,
            ..Default::default()
        };
        
        let mut adaptive_optimizer = AdaptiveOptimizer::new(config);
        
        // Create sampler
        let sampler = SASampler::new(Some(initial_params.clone()));
        
        // Mock model for demo
        let model = MockCompiledModel::new();
        
        // Run adaptive optimization
        let result = adaptive_optimizer.optimize(
            sampler,
            &model,
            initial_params.clone(),
            initial_penalties.clone(),
            50,
        )?;
        
        println!("   - Total iterations: {}", result.total_iterations);
        println!("   - Best energy: {:.4}", result.best_solution.energy);
        println!("   - Final penalty weights: {:?}", result.final_penalty_weights);
    }
    
    Ok(())
}

/// Demonstrate parameter tuning with constraints
fn parameter_tuning_demo() -> Result<(), Box<dyn Error>> {
    println!("3. Parameter Tuning with Constraints");
    println!("   Automatically tuning sampler parameters for best performance");
    
    // Define parameter bounds for tuning
    let parameter_bounds = vec![
        ParameterBounds {
            name: "T_0".to_string(),
            min: 1.0,
            max: 100.0,
            scale: ParameterScale::Logarithmic,
            integer: false,
        },
        ParameterBounds {
            name: "T_f".to_string(),
            min: 0.001,
            max: 1.0,
            scale: ParameterScale::Logarithmic,
            integer: false,
        },
        ParameterBounds {
            name: "steps".to_string(),
            min: 100.0,
            max: 10000.0,
            scale: ParameterScale::Linear,
            integer: true,
        },
    ];
    
    // Tuning configuration
    let tuning_config = TuningConfig {
        max_evaluations: 30,
        initial_samples: 10,
        acquisition: AcquisitionType::ExpectedImprovement,
        exploration_factor: 0.15,
        ..Default::default()
    };
    
    let mut tuner = ParameterTuner::new(tuning_config);
    tuner.add_parameters(parameter_bounds);
    
    // Define sampler factory
    let sampler_factory = |params: HashMap<String, f64>| {
        SASampler::new(Some(params))
    };
    
    // Mock model
    let model = MockCompiledModel::new();
    
    // Define objective function (minimize average energy)
    let objective = |samples: &[quantrs2_tytan::sampler::SampleResult]| {
        samples.iter().map(|s| s.energy).sum::<f64>() / samples.len() as f64
    };
    
    // Run parameter tuning
    println!("   Running Bayesian optimization...");
    let tuning_result = tuner.tune_sampler(sampler_factory, &model, objective)?;
    
    println!("\n   Tuning Results:");
    println!("   - Best parameters:");
    for (param, value) in &tuning_result.best_parameters {
        println!("     {}: {:.3}", param, value);
    }
    println!("   - Best objective: {:.4}", tuning_result.best_objective);
    println!("   - Improvement: {:.1}%", tuning_result.improvement_over_default * 100.0);
    println!("   - Total evaluations: {}", tuning_result.total_evaluations);
    println!("   - Converged: {}", tuning_result.converged);
    
    println!("\n   Parameter importance:");
    for (param, importance) in &tuning_result.parameter_importance {
        println!("   - {}: {:.2}", param, importance);
    }
    
    Ok(())
}

/// Demonstrate constraint handling
#[cfg(feature = "scirs")]
fn constraint_handling_demo() -> Result<(), Box<dyn Error>> {
    use quantrs2_tytan::optimization::constraints::{ConstraintHandler, EncodingType};
    
    println!("4. Advanced Constraint Handling");
    println!("   Demonstrating various constraint types and encodings");
    
    let mut handler = ConstraintHandler::new();
    
    // Add one-hot constraint
    handler.add_one_hot(
        "selection".to_string(),
        vec!["opt_a".to_string(), "opt_b".to_string(), "opt_c".to_string()],
    )?;
    
    // Add cardinality constraint
    handler.add_cardinality(
        "at_most_two".to_string(),
        vec!["x1".to_string(), "x2".to_string(), "x3".to_string(), "x4".to_string()],
        2,
    )?;
    
    // Add integer encoding
    let int_vars = handler.add_integer_encoding(
        "int_value".to_string(),
        "v".to_string(),
        0,
        7,
        EncodingType::Binary,
    )?;
    
    println!("   Integer encoding variables: {:?}", int_vars);
    
    // Analyze constraint structure
    let analysis = handler.analyze_constraints();
    println!("\n   Constraint Analysis:");
    println!("   - Total constraints: {}", analysis.total_constraints);
    println!("   - Total variables: {}", analysis.total_variables);
    println!("   - Slack variables: {}", analysis.slack_variables);
    println!("   - Average variables per constraint: {:.1}", 
        analysis.avg_variables_per_constraint);
    
    Ok(())
}

// Mock implementation for demonstration
struct MockCompiledModel;

impl MockCompiledModel {
    fn new() -> Self {
        Self
    }
    
    fn get_constraints(&self) -> HashMap<String, quantrs2_tytan::symbol::Term> {
        HashMap::new()
    }
    
    fn get_variable_map(&self) -> HashMap<String, usize> {
        HashMap::from([
            ("x".to_string(), 0),
            ("y".to_string(), 1),
            ("z".to_string(), 2),
        ])
    }
    
    fn to_qubo(&self) -> (ndarray::Array2<f64>, HashMap<String, usize>) {
        let mut matrix = ndarray::Array2::zeros((3, 3));
        matrix[[0, 0]] = -1.0;
        matrix[[1, 1]] = -1.0;
        matrix[[2, 2]] = -1.0;
        
        (matrix, self.get_variable_map())
    }
}

impl Clone for MockCompiledModel {
    fn clone(&self) -> Self {
        Self::new()
    }
}