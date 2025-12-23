//! Example: Numeric equation solving
#![allow(clippy::redundant_clone)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::approx_constant)]
#![allow(clippy::mutable_key_type)]
#![allow(clippy::uninlined_format_args)]
//!
//! Demonstrates various numeric methods for solving equations including:
//! - Newton-Raphson method
//! - Bisection method
//! - Secant method
//! - Brent's method

use quantrs2_symengine::solving::{bisection, brent, newton_raphson, secant, SolverConfig};
use quantrs2_symengine::Expression;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Numeric Equation Solving Example ===\n");

    // Example 1: Newton-Raphson method
    println!("1. Newton-Raphson Method");
    println!("   Finding roots of x² - 2 = 0 (computing √2)");

    let x = Expression::symbol("x");
    let expr1 = x.pow(&Expression::from(2)) - Expression::from(2);

    let config = SolverConfig::default();

    match newton_raphson(&expr1, &x, 1.0, &config) {
        Ok(result) => {
            println!("   Result:");
            println!("     Root: {:.10}", result.root);
            println!("     Iterations: {}", result.iterations);
            println!("     Error: {:.2e}", result.error);
            println!("     Converged: {}", result.converged);
            println!("     Actual √2: {:.10}", 2.0_f64.sqrt());
            println!(
                "     Difference: {:.2e}",
                (result.root - 2.0_f64.sqrt()).abs()
            );
        }
        Err(e) => eprintln!("   Error: {}", e),
    }

    // Example 2: Bisection method
    println!("\n2. Bisection Method");
    println!("   Finding roots of x³ - x - 2 = 0");

    let expr2 = x.pow(&Expression::from(3)) - x.clone() - Expression::from(2);

    match bisection(&expr2, &x, 1.0, 2.0, &config) {
        Ok(result) => {
            println!("   Result:");
            println!("     Root: {:.10}", result.root);
            println!("     Iterations: {}", result.iterations);
            println!("     Error: {:.2e}", result.error);
            println!("     Converged: {}", result.converged);
            println!(
                "   Verification: f({:.6}) ≈ {:.2e}",
                result.root, result.error
            );
        }
        Err(e) => eprintln!("   Error: {}", e),
    }

    // Example 3: Secant method
    println!("\n3. Secant Method");
    println!("   Finding roots of cos(x) = x");

    let expr3 = Expression::new("cos(x)") - x.clone();

    match secant(&expr3, &x, 0.5, 1.0, &config) {
        Ok(result) => {
            println!("   Result:");
            println!("     Root: {:.10}", result.root);
            println!("     Iterations: {}", result.iterations);
            println!("     Error: {:.2e}", result.error);
            println!("     Converged: {}", result.converged);
            println!("   This root is known as the Dottie number ≈ 0.7390851332");
        }
        Err(e) => eprintln!("   Error: {}", e),
    }

    // Example 4: Brent's method (hybrid method)
    println!("\n4. Brent's Method");
    println!("   Finding roots of e^x - 3x = 0");

    let expr4 = Expression::new("exp(x)") - (Expression::from(3) * x.clone());

    match brent(&expr4, &x, 0.5, 2.0, &config) {
        Ok(result) => {
            println!("   Result:");
            println!("     Root: {:.10}", result.root);
            println!("     Iterations: {}", result.iterations);
            println!("     Error: {:.2e}", result.error);
            println!("     Converged: {}", result.converged);
            println!("   Brent's method combines bisection, secant, and inverse");
            println!("   quadratic interpolation for robustness and speed.");
        }
        Err(e) => eprintln!("   Error: {}", e),
    }

    // Example 5: Custom solver configuration
    println!("\n5. Custom Solver Configuration");
    println!("   High-precision Newton-Raphson for √2");

    let high_precision_config = SolverConfig::new()
        .with_max_iterations(100)
        .with_abs_tolerance(1e-15)
        .with_rel_tolerance(1e-15);

    match newton_raphson(&expr1, &x, 1.5, &high_precision_config) {
        Ok(result) => {
            println!("   Result with high precision:");
            println!("     Root: {:.16}", result.root);
            println!("     Iterations: {}", result.iterations);
            println!("     Error: {:.2e}", result.error);
            println!("     Converged: {}", result.converged);
            println!(
                "     Difference from √2: {:.2e}",
                (result.root - 2.0_f64.sqrt()).abs()
            );
        }
        Err(e) => eprintln!("   Error: {}", e),
    }

    // Example 6: Quantum eigenvalue problem
    println!("\n6. Quantum Application: Finding Energy Eigenvalue");
    println!("   Solving E - 1/2 * (1 + cos(θ)) = 0 for θ");

    let theta = Expression::symbol("theta");
    let energy = 0.75; // Target energy
    let hamiltonian_eq = Expression::from(energy)
        - (Expression::from(0.5) * (Expression::from(1) + Expression::new("cos(theta)")));

    match newton_raphson(&hamiltonian_eq, &theta, 1.0, &config) {
        Ok(result) => {
            println!("   Result:");
            println!("     Angle θ: {:.6} rad", result.root);
            println!("     Angle θ: {:.2}°", result.root.to_degrees());
            println!("     Iterations: {}", result.iterations);
            println!("     Converged: {}", result.converged);
        }
        Err(e) => eprintln!("   Error: {}", e),
    }

    // Example 7: Comparison of methods
    println!("\n7. Method Comparison");
    println!("   Finding the positive root of x² - 2 = 0");

    let methods = vec![
        ("Newton-Raphson", {
            newton_raphson(&expr1, &x, 1.5, &config)
        }),
        ("Bisection", { bisection(&expr1, &x, 1.0, 2.0, &config) }),
        ("Secant", { secant(&expr1, &x, 1.0, 2.0, &config) }),
        ("Brent", { brent(&expr1, &x, 1.0, 2.0, &config) }),
    ];

    println!("   Method            | Root         | Iterations | Error");
    println!("   ------------------|--------------|------------|------------");
    for (method_name, result) in methods {
        match result {
            Ok(r) => {
                println!(
                    "   {:<17} | {:.10} | {:>10} | {:.2e}",
                    method_name, r.root, r.iterations, r.error
                );
            }
            Err(e) => {
                println!("   {:<17} | Error: {}", method_name, e);
            }
        }
    }

    println!("\n=== Key Observations ===");
    println!("• Newton-Raphson: Fastest convergence when derivative is available");
    println!("• Bisection: Most robust but slower, guaranteed to converge");
    println!("• Secant: Good balance, doesn't require derivatives");
    println!("• Brent: Best overall performance, combines multiple methods");

    println!("\n=== Numeric Solving Complete ===");

    Ok(())
}
