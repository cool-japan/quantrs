//! Example: Polynomial operations and manipulation
#![allow(clippy::redundant_clone)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::approx_constant)]
#![allow(clippy::mutable_key_type)]
#![allow(clippy::uninlined_format_args)]
//!
//! Demonstrates polynomial arithmetic, differentiation, integration, and more.

use quantrs2_symengine::polynomial::Polynomial;
use quantrs2_symengine::Expression;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Polynomial Operations Example ===\n");

    // Example 1: Creating polynomials
    println!("1. Creating Polynomials");

    let x = Expression::symbol("x");

    // Polynomial: 1 + 2x + 3x²
    let coeffs1 = vec![
        Expression::from(1),
        Expression::from(2),
        Expression::from(3),
    ];
    let poly1 = Polynomial::new(coeffs1, x.clone());

    println!("   Polynomial P(x) = 1 + 2x + 3x²");
    println!("     Degree: {}", poly1.degree());
    println!(
        "     Leading coefficient: {}",
        poly1.leading_coefficient().unwrap()
    );
    println!("     Coefficient of x: {}", poly1.coefficient(1).unwrap());
    println!("     Expression form: {}", poly1.to_expression());

    // Example 2: Polynomial evaluation
    println!("\n2. Polynomial Evaluation (Horner's method)");

    let eval_points = vec![0.0, 1.0, 2.0, -1.0];

    println!("   Evaluating P(x) at various points:");
    for point in eval_points {
        let result = poly1.evaluate(&Expression::from(point));
        println!("     P({}) = {}", point, result);
    }

    // Example 3: Polynomial addition
    println!("\n3. Polynomial Addition");

    // Polynomial: 4 + 5x
    let coeffs2 = vec![Expression::from(4), Expression::from(5)];
    let poly2 = Polynomial::new(coeffs2, x.clone());

    println!("   P₁(x) = {}", poly1.to_expression());
    println!("   P₂(x) = {}", poly2.to_expression());

    let sum = poly1.add(&poly2);
    println!("   P₁(x) + P₂(x) = {}", sum.to_expression());
    println!("   Expected: 5 + 7x + 3x²");

    // Example 4: Polynomial multiplication
    println!("\n4. Polynomial Multiplication");

    // (1 + 2x) * (3 + 4x) = 3 + 10x + 8x²
    let poly_a = Polynomial::new(vec![Expression::from(1), Expression::from(2)], x.clone());
    let poly_b = Polynomial::new(vec![Expression::from(3), Expression::from(4)], x.clone());

    println!("   P_a(x) = {}", poly_a.to_expression());
    println!("   P_b(x) = {}", poly_b.to_expression());

    let product = poly_a.multiply(&poly_b);
    println!("   P_a(x) * P_b(x) = {}", product.to_expression());
    println!("   Expected: 3 + 10x + 8x²");

    // Example 5: Polynomial differentiation
    println!("\n5. Polynomial Differentiation");

    // P(x) = 1 + 2x + 3x² + 4x³
    let poly_diff = Polynomial::new(
        vec![
            Expression::from(1),
            Expression::from(2),
            Expression::from(3),
            Expression::from(4),
        ],
        x.clone(),
    );

    println!("   P(x) = {}", poly_diff.to_expression());

    let derivative = poly_diff.differentiate();
    println!("   P'(x) = {}", derivative.to_expression());
    println!("   Expected: 2 + 6x + 12x²");

    let second_derivative = derivative.differentiate();
    println!("   P''(x) = {}", second_derivative.to_expression());
    println!("   Expected: 6 + 24x");

    // Example 6: Polynomial integration
    println!("\n6. Polynomial Integration");

    // Integrate 2 + 6x + 12x²
    println!("   ∫(2 + 6x + 12x²) dx");

    let integrated = derivative.integrate();
    println!("   Result: {}", integrated.to_expression());
    println!("   Expected: 0 + 2x + 3x² + 4x³ (constant = 0)");

    // Example 7: Quantum application - Characteristic polynomial
    println!("\n7. Quantum Application: Characteristic Polynomial");
    println!("   For a 2×2 matrix, det(A - λI) = 0");
    println!("   Example: Pauli-X eigenvalues");

    let lambda = Expression::symbol("lambda");

    // For Pauli-X: det([[−λ, 1], [1, −λ]]) = λ² − 1
    let char_poly = Polynomial::new(
        vec![
            Expression::from(-1),
            Expression::from(0),
            Expression::from(1),
        ],
        lambda.clone(),
    );

    println!(
        "   Characteristic polynomial: {}",
        char_poly.to_expression()
    );
    println!("   This factors as (λ-1)(λ+1), giving eigenvalues ±1");

    // Example 8: Quantum gate parameterization
    println!("\n8. Quantum Gate Parameterization");
    println!("   Rotating gate: R(θ) = I*cos(θ) + X*sin(θ)");
    println!("   Taylor expansion of cos(θ) around θ=0");

    let theta = Expression::symbol("theta");

    // cos(θ) ≈ 1 - θ²/2 + θ⁴/24
    let cos_taylor = Polynomial::new(
        vec![
            Expression::from(1),          // constant
            Expression::from(0),          // θ
            Expression::from(-0.5),       // θ²/2!
            Expression::from(0),          // θ³
            Expression::from(1.0 / 24.0), // θ⁴/4!
        ],
        theta.clone(),
    );

    println!("   cos(θ) Taylor expansion (up to θ⁴):");
    println!("     {}", cos_taylor.to_expression());

    println!("   Evaluating at θ = π/4:");
    let theta_val = std::f64::consts::FRAC_PI_4;
    let result = cos_taylor.evaluate(&Expression::from(theta_val));
    println!("     Taylor: {}", result);
    println!("     Exact:  {:.10}", theta_val.cos());

    // Example 9: Building complex polynomials
    println!("\n9. Building Polynomials from Expressions");

    let y = Expression::symbol("y");

    // Create a polynomial symbolically
    let symbolic_poly = y.pow(&Expression::from(3))
        + Expression::from(2) * y.pow(&Expression::from(2))
        + Expression::from(3) * y.clone()
        + Expression::from(4);

    println!("   Symbolic expression: {}", symbolic_poly);

    // Extract and work with coefficients
    println!("   This represents: 4 + 3y + 2y² + y³");

    // Example 10: Legendre polynomial (quantum angular momentum)
    println!("\n10. Special Polynomials: Legendre P₂(x)");
    println!("    Used in quantum angular momentum calculations");

    // P₂(x) = (3x² - 1)/2
    let legendre_p2_unnormalized = Polynomial::new(
        vec![
            Expression::from(-0.5),
            Expression::from(0),
            Expression::from(1.5),
        ],
        x.clone(),
    );

    println!("    P₂(x) = {}", legendre_p2_unnormalized.to_expression());

    let eval_x = vec![-1.0, 0.0, 1.0];
    println!("    Evaluations:");
    for &val in &eval_x {
        let result = legendre_p2_unnormalized.evaluate(&Expression::from(val));
        println!("      P₂({}) = {}", val, result);
    }

    println!("\n=== Polynomial Operations Complete ===");
    println!("\nKey takeaways:");
    println!("• Polynomials support standard arithmetic operations");
    println!("• Horner's method provides efficient evaluation");
    println!("• Symbolic differentiation and integration are exact");
    println!("• Useful for quantum gate parameterizations and eigenvalue problems");

    Ok(())
}
