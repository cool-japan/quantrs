//! Tests for the sampler module.

use quantrs_tytan::*;
use std::collections::HashMap;

#[test]
fn test_sa_sampler_simple() {
    // Test SASampler on a simple QUBO problem
    // Create a simple QUBO matrix for testing
    let mut matrix = ndarray::Array::<f64, _>::zeros((2, 2));
    matrix[[0, 0]] = -1.0;  // Minimize x
    matrix[[1, 1]] = -1.0;  // Minimize y
    matrix[[0, 1]] = 2.0;   // Penalty for x and y both being 1
    matrix[[1, 0]] = 2.0;   // (symmetric)
    
    // Create variable map
    let mut var_map = HashMap::new();
    var_map.insert("x".to_string(), 0);
    var_map.insert("y".to_string(), 1);
    
    let qubo = (matrix, var_map);
    
    // Create sampler with fixed seed for reproducibility
    let sampler = SASampler::new(Some(42));
    
    // Run sampler with a few shots
    let results = sampler.run_qubo(&qubo, 10).unwrap();
    
    // Check that we got at least one result
    assert!(!results.is_empty());
    
    // Check that the best solution makes sense
    // For this problem, the optimal solution should be x=1, y=0 or x=0, y=1
    let best = &results[0];
    
    // Either x=1, y=0 or x=0, y=1 should be optimal
    let x = best.assignments.get("x").unwrap();
    let y = best.assignments.get("y").unwrap();
    
    assert!(
        (*x && !*y) || (!*x && *y),
        "Expected either x=1,y=0 or x=0,y=1, got x={},y={}", x, y
    );
    
    // Energy should be -1.0
    assert!(
        (best.energy - (-1.0)).abs() < 1e-6,
        "Expected energy -1.0, got {}", best.energy
    );
}

#[test]
fn test_ga_sampler_simple() {
    // Test GASampler on a simple QUBO problem
    // Create a simple QUBO matrix for testing
    let mut matrix = ndarray::Array::<f64, _>::zeros((2, 2));
    matrix[[0, 0]] = -1.0;  // Minimize x
    matrix[[1, 1]] = -1.0;  // Minimize y
    matrix[[0, 1]] = 2.0;   // Penalty for x and y both being 1
    matrix[[1, 0]] = 2.0;   // (symmetric)
    
    // Create variable map
    let mut var_map = HashMap::new();
    var_map.insert("x".to_string(), 0);
    var_map.insert("y".to_string(), 1);
    
    let qubo = (matrix, var_map);
    
    // Create sampler with fixed seed for reproducibility
    let sampler = GASampler::new(Some(42));
    
    // Run sampler with a few shots
    let results = sampler.run_qubo(&qubo, 10).unwrap();
    
    // Check that we got at least one result
    assert!(!results.is_empty());
    
    // Check that the best solution makes sense
    // For this problem, the optimal solution should be x=1, y=0 or x=0, y=1
    let best = &results[0];
    
    // Either x=1, y=0 or x=0, y=1 should be optimal
    let x = best.assignments.get("x").unwrap();
    let y = best.assignments.get("y").unwrap();
    
    assert!(
        (*x && !*y) || (!*x && *y),
        "Expected either x=1,y=0 or x=0,y=1, got x={},y={}", x, y
    );
    
    // Energy should be -1.0
    assert!(
        (best.energy - (-1.0)).abs() < 1e-6,
        "Expected energy -1.0, got {}", best.energy
    );
}

#[test]
fn test_optimize_qubo() {
    // Test optimize_qubo function
    // Create a simple QUBO matrix for testing
    let mut matrix = ndarray::Array::<f64, _>::zeros((2, 2));
    matrix[[0, 0]] = -1.0;  // Minimize x
    matrix[[1, 1]] = -1.0;  // Minimize y
    matrix[[0, 1]] = 2.0;   // Penalty for x and y both being 1
    matrix[[1, 0]] = 2.0;   // (symmetric)
    
    // Create variable map
    let mut var_map = HashMap::new();
    var_map.insert("x".to_string(), 0);
    var_map.insert("y".to_string(), 1);
    
    // Run optimization
    let results = optimize_qubo(&matrix, &var_map, None, 100);
    
    // Check that we got at least one result
    assert!(!results.is_empty());
    
    // Check that the best solution makes sense
    // For this problem, the optimal solution should be x=1, y=0 or x=0, y=1
    let best = &results[0];
    
    // Either x=1, y=0 or x=0, y=1 should be optimal
    let x = best.assignments.get("x").unwrap();
    let y = best.assignments.get("y").unwrap();
    
    assert!(
        (*x && !*y) || (!*x && *y),
        "Expected either x=1,y=0 or x=0,y=1, got x={},y={}", x, y
    );
    
    // Energy should be -1.0
    assert!(
        (best.energy - (-1.0)).abs() < 1e-6,
        "Expected energy -1.0, got {}", best.energy
    );
}

#[test]
fn test_sampler_one_hot_constraint() {
    // Test a one-hot constraint problem (exactly one variable is 1)
    let x = symbols("x");
    let y = symbols("y");
    let z = symbols("z");
    
    // Constraint: (x + y + z - 1)^2
    let expr = (x.clone() + y.clone() + z.clone() - 1).pow(2);
    
    // Compile to QUBO
    let (qubo, _) = Compile::new(&expr).get_qubo().unwrap();
    
    // Create sampler with fixed seed for reproducibility
    let sampler = SASampler::new(Some(42));
    
    // Run sampler with a reasonable number of shots
    let results = sampler.run_qubo(&qubo, 100).unwrap();
    
    // Check that the best solution satisfies the one-hot constraint
    let best = &results[0];
    
    // Extract assignments
    let x_val = best.assignments.get("x").unwrap();
    let y_val = best.assignments.get("y").unwrap();
    let z_val = best.assignments.get("z").unwrap();
    
    // Verify exactly one variable is 1
    let sum = (*x_val as i32) + (*y_val as i32) + (*z_val as i32);
    assert_eq!(sum, 1, "Expected exactly one variable to be 1, got {}", sum);
    
    // Best energy should be 0 (no constraint violation)
    assert!(
        best.energy.abs() < 1e-6,
        "Expected energy 0.0, got {}", best.energy
    );
}