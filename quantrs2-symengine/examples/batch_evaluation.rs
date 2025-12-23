//! Example: Batch evaluation of symbolic expressions
#![allow(clippy::redundant_clone)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::approx_constant)]
#![allow(clippy::mutable_key_type)]
#![allow(clippy::uninlined_format_args)]
//!
//! Demonstrates efficient evaluation of symbolic expressions at multiple parameter points
//! using SIMD acceleration and parallel processing.

use quantrs2_symengine::simd_eval::{BatchEvalConfig, BatchEvaluator};
use quantrs2_symengine::Expression;
use scirs2_core::ndarray::array;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Batch Evaluation Example ===\n");

    // Create a symbolic expression for quantum gate rotation angle
    let theta = Expression::symbol("theta");
    let phi = Expression::symbol("phi");

    // Example 1: Single parameter evaluation
    println!("1. Single parameter batch evaluation:");
    println!("   Expression: cos(theta)^2");

    let expr = Expression::new("cos(theta)").pow(&Expression::from(2));

    let evaluator = BatchEvaluator::new();
    let theta_values = array![0.0, 0.785, 1.571, 2.356, 3.142]; // 0, π/4, π/2, 3π/4, π

    match evaluator.eval_scalar(&expr, "theta", theta_values.view()) {
        Ok(results) => {
            println!("   Results:");
            for (i, &val) in results.iter().enumerate() {
                println!(
                    "     theta = {:.3} rad -> cos²(θ) = {:.6}",
                    theta_values[i], val
                );
            }
        }
        Err(e) => eprintln!("   Error: {}", e),
    }

    // Example 2: Vector of expressions
    println!("\n2. Multiple expressions batch evaluation:");
    println!("   Expressions: [theta, theta^2, theta^3]");

    let theta_sym = Expression::symbol("theta");
    let expressions = vec![
        theta_sym.clone(),
        theta_sym.pow(&Expression::from(2)),
        theta_sym.pow(&Expression::from(3)),
    ];

    let theta_values_vec = array![1.0, 2.0, 3.0, 4.0];

    match evaluator.eval_vector(&expressions, "theta", theta_values_vec.view()) {
        Ok(results) => {
            println!("   Results (rows = parameter points, columns = expressions):");
            for i in 0..results.nrows() {
                print!("     theta = {} -> [", theta_values_vec[i]);
                for j in 0..results.ncols() {
                    print!("{:.2}", results[[i, j]]);
                    if j < results.ncols() - 1 {
                        print!(", ");
                    }
                }
                println!("]");
            }
        }
        Err(e) => eprintln!("   Error: {}", e),
    }

    // Example 3: Multi-parameter evaluation (quantum gate parameters)
    println!("\n3. Multi-parameter evaluation (U3 gate parameters):");
    println!("   Expression: sin(theta) * cos(phi)");

    let expr_multi = Expression::new("sin(theta)") * Expression::new("cos(phi)");

    let param_sets = vec![
        {
            let mut map = HashMap::new();
            map.insert("theta".to_string(), 0.0);
            map.insert("phi".to_string(), 0.0);
            map
        },
        {
            let mut map = HashMap::new();
            map.insert("theta".to_string(), std::f64::consts::FRAC_PI_4); // π/4
            map.insert("phi".to_string(), 0.0);
            map
        },
        {
            let mut map = HashMap::new();
            map.insert("theta".to_string(), std::f64::consts::FRAC_PI_2); // π/2
            map.insert("phi".to_string(), std::f64::consts::FRAC_PI_4);
            map
        },
        {
            let mut map = HashMap::new();
            map.insert("theta".to_string(), std::f64::consts::PI); // π
            map.insert("phi".to_string(), std::f64::consts::FRAC_PI_2);
            map
        },
    ];

    match evaluator.eval_multi_param(&expr_multi, &param_sets) {
        Ok(results) => {
            println!("   Results:");
            for (i, params) in param_sets.iter().enumerate() {
                let theta_val = params.get("theta").unwrap();
                let phi_val = params.get("phi").unwrap();
                println!(
                    "     (θ={:.3}, φ={:.3}) -> sin(θ)cos(φ) = {:.6}",
                    theta_val, phi_val, results[i]
                );
            }
        }
        Err(e) => eprintln!("   Error: {}", e),
    }

    // Example 4: Performance configuration
    println!("\n4. Custom batch evaluation configuration:");

    let config = BatchEvalConfig::new()
        .with_parallel_threshold(50)
        .without_simd();

    let custom_evaluator = BatchEvaluator::with_config(config);

    println!("   Created evaluator with:");
    println!("     - Parallel threshold: 50 points");
    println!("     - SIMD: disabled");
    println!("   This configuration is useful for debugging or specific hardware.");

    // Example 5: Gradient evaluation at multiple points
    println!("\n5. Gradient evaluation (for VQE/QAOA optimization):");
    println!("   Function: f(x, y) = x² + y²");
    println!("   Gradient: [2x, 2y]");

    let x = Expression::symbol("x");
    let y = Expression::symbol("y");
    let f = x.pow(&Expression::from(2)) + y.pow(&Expression::from(2));

    let symbols = vec![x.clone(), y.clone()];
    let gradient_points = vec![
        {
            let mut map = HashMap::new();
            map.insert("x".to_string(), 0.0);
            map.insert("y".to_string(), 0.0);
            map
        },
        {
            let mut map = HashMap::new();
            map.insert("x".to_string(), 1.0);
            map.insert("y".to_string(), 1.0);
            map
        },
        {
            let mut map = HashMap::new();
            map.insert("x".to_string(), 2.0);
            map.insert("y".to_string(), 3.0);
            map
        },
    ];

    match quantrs2_symengine::simd_eval::batch_gradient(&f, &symbols, &gradient_points) {
        Ok(gradients) => {
            println!("   Results:");
            for (i, params) in gradient_points.iter().enumerate() {
                let x_val = params.get("x").unwrap();
                let y_val = params.get("y").unwrap();
                println!(
                    "     (x={}, y={}) -> ∇f = [{:.2}, {:.2}]",
                    x_val,
                    y_val,
                    gradients[[i, 0]],
                    gradients[[i, 1]]
                );
            }
        }
        Err(e) => eprintln!("   Error: {}", e),
    }

    println!("\n=== Batch Evaluation Complete ===");

    Ok(())
}
