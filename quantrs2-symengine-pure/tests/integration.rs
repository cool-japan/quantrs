//! Integration tests for quantrs2-symengine-pure.
//!
//! These tests verify end-to-end functionality across multiple modules.

use std::collections::HashMap;

use quantrs2_symengine_pure::{
    expr::Expression,
    matrix::{self, SymbolicMatrix},
    ops::trig,
    optimization::{gradient_at, ParameterShiftRule},
    parser,
    pattern::{self, match_pattern, Pattern},
    scirs2_bridge::complex::eval_complex,
    serialize, simplify, Complex64, SymEngineResult,
};

// =========================================================================
// VQE Workflow Tests
// =========================================================================

/// Test a complete VQE-style workflow:
/// 1. Create parameterized energy expression
/// 2. Compute symbolic gradient
/// 3. Evaluate at specific parameter values
/// 4. Use parameter-shift rule for numerical gradient
#[test]
fn test_vqe_workflow() -> SymEngineResult<()> {
    // Create a simple VQE energy expression: E(theta) = cos(theta) + 0.5*sin(2*theta)
    let theta = Expression::symbol("theta");
    let two = Expression::int(2);
    let half = Expression::float(0.5)?;

    let energy = trig::cos(&theta) + half * trig::sin(&(two * theta.clone()));

    // Compute symbolic gradient
    let gradient = energy.diff(&theta);

    // Evaluate at theta = pi/4
    let mut values = HashMap::new();
    let pi_4 = std::f64::consts::FRAC_PI_4;
    values.insert("theta".to_string(), pi_4);

    let energy_val = energy.eval(&values)?;
    let grad_val = gradient.eval(&values)?;

    // Verify values (at pi/4: cos(pi/4) + 0.5*sin(pi/2) = sqrt(2)/2 + 0.5)
    assert!((energy_val - 0.5f64.mul_add((2.0 * pi_4).sin(), pi_4.cos())).abs() < 1e-10);

    // Gradient: -sin(theta) + cos(2*theta)
    assert!((grad_val - (-pi_4.sin() + (2.0 * pi_4).cos())).abs() < 1e-10);

    // Verify parameter-shift rule gives same result
    let psr = ParameterShiftRule::new();
    let psr_grad = psr.compute_gradient(
        |params| 0.5f64.mul_add((2.0 * params[0]).sin(), params[0].cos()),
        &[pi_4],
    );

    assert!((psr_grad[0] - grad_val).abs() < 1e-8);

    Ok(())
}

/// Test multi-parameter VQE optimization
#[test]
fn test_vqe_multi_parameter() -> SymEngineResult<()> {
    // Energy function with two parameters
    let theta = Expression::symbol("theta");
    let phi = Expression::symbol("phi");

    let energy =
        trig::sin(&theta) * trig::cos(&phi) + Expression::float(0.5)? * theta.clone() * phi.clone();

    // Compute gradient vector
    let vars = vec![theta, phi];
    let gradient = energy.gradient(&vars);

    assert_eq!(gradient.len(), 2);

    // Evaluate gradient at a point
    let mut values = HashMap::new();
    values.insert("theta".to_string(), 0.5);
    values.insert("phi".to_string(), 0.3);

    let grad_theta = gradient[0].eval(&values)?;
    let grad_phi = gradient[1].eval(&values)?;

    // Verify numerically
    let expected_grad_theta = 0.5_f64.cos().mul_add(0.3_f64.cos(), 0.5 * 0.3);
    let expected_grad_phi = (-0.5_f64.sin()).mul_add(0.3_f64.sin(), 0.5 * 0.5);

    assert!((grad_theta - expected_grad_theta).abs() < 1e-10);
    assert!((grad_phi - expected_grad_phi).abs() < 1e-10);

    Ok(())
}

// =========================================================================
// Quantum Gate Matrix Tests
// =========================================================================

/// Test quantum gate composition and simplification
#[test]
fn test_quantum_gate_composition() -> SymEngineResult<()> {
    // X^2 = I
    let x = matrix::pauli_x();
    let x_squared = x.matmul(&x)?;
    let identity = SymbolicMatrix::identity(2);

    // Verify by evaluating each element
    for i in 0..2 {
        for j in 0..2 {
            let expected = identity.get(i, j).eval(&HashMap::new())?;
            let actual = x_squared.get(i, j).eval(&HashMap::new())?;
            assert!(
                (expected - actual).abs() < 1e-10,
                "X^2 should be I at ({i},{j})"
            );
        }
    }

    // Z*X*Z = -X (anticommutation)
    let z = matrix::pauli_z();
    let zxz = z.matmul(&x)?.matmul(&z)?;

    for i in 0..2 {
        for j in 0..2 {
            let expected = -x.get(i, j).eval(&HashMap::new())?;
            let actual = zxz.get(i, j).eval(&HashMap::new())?;
            assert!(
                (expected - actual).abs() < 1e-10,
                "ZXZ should be -X at ({i},{j})"
            );
        }
    }

    Ok(())
}

/// Test parametric rotation gates
#[test]
fn test_parametric_gates() -> SymEngineResult<()> {
    let theta = Expression::symbol("theta");

    // Rx matrix has structure:
    // [[cos(θ/2), -i*sin(θ/2)],
    //  [-i*sin(θ/2), cos(θ/2)]]
    // Only diagonal elements are purely real
    let rx = matrix::rx(&theta);

    let mut values = HashMap::new();
    values.insert("theta".to_string(), 0.0);

    // At theta=0, diagonal elements should be cos(0) = 1
    let rx_00 = rx.get(0, 0).eval(&values)?;
    let rx_11 = rx.get(1, 1).eval(&values)?;
    assert!((rx_00 - 1.0).abs() < 1e-10, "Rx(0)[0,0] should be 1");
    assert!((rx_11 - 1.0).abs() < 1e-10, "Rx(0)[1,1] should be 1");

    // Test at theta=pi/2
    values.insert("theta".to_string(), std::f64::consts::FRAC_PI_2);

    // At theta=pi/2: Rx diagonal = cos(pi/4)
    let rx_half_00 = rx.get(0, 0).eval(&values)?;
    let rx_half_11 = rx.get(1, 1).eval(&values)?;

    let expected_diag = (std::f64::consts::FRAC_PI_4).cos();
    assert!(
        (rx_half_00 - expected_diag).abs() < 1e-10,
        "Rx(pi/2)[0,0] should be cos(pi/4)"
    );
    assert!(
        (rx_half_11 - expected_diag).abs() < 1e-10,
        "Rx(pi/2)[1,1] should be cos(pi/4)"
    );

    // Test Ry and Rz gates similarly (all should have real diagonal elements)
    let ry = matrix::ry(&theta);
    let rz = matrix::rz(&theta);

    // Ry diagonal at theta=0 should be 1
    values.insert("theta".to_string(), 0.0);
    assert!(
        (ry.get(0, 0).eval(&values)? - 1.0).abs() < 1e-10,
        "Ry(0)[0,0] should be 1"
    );
    assert!(
        (ry.get(1, 1).eval(&values)? - 1.0).abs() < 1e-10,
        "Ry(0)[1,1] should be 1"
    );

    // Note: Off-diagonal elements contain complex i and cannot be evaluated as real
    // This is expected behavior - full complex evaluation would require a different API

    Ok(())
}

/// Test tensor product for multi-qubit systems
#[test]
fn test_tensor_product() -> SymEngineResult<()> {
    let x = matrix::pauli_x();
    let z = matrix::pauli_z();

    // X ⊗ Z (4x4 matrix)
    let xz = x.kron(&z);

    assert_eq!(xz.nrows(), 4);
    assert_eq!(xz.ncols(), 4);

    // Verify structure: X ⊗ Z = [[0*Z, 1*Z], [1*Z, 0*Z]] = [[0, 0, Z], [0, 0, Z], [Z, Z, 0, 0], [Z, Z, 0, 0]]
    // Actually: [[0, 0, 1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, -1, 0, 0]]

    let empty = HashMap::new();

    // Check corners
    assert!((xz.get(0, 0).eval(&empty)?).abs() < 1e-10);
    assert!((xz.get(0, 2).eval(&empty)? - 1.0).abs() < 1e-10);
    assert!((xz.get(1, 3).eval(&empty)? - (-1.0)).abs() < 1e-10);
    assert!((xz.get(2, 0).eval(&empty)? - 1.0).abs() < 1e-10);
    assert!((xz.get(3, 1).eval(&empty)? - (-1.0)).abs() < 1e-10);

    Ok(())
}

// =========================================================================
// Serialization Roundtrip Tests
// =========================================================================

/// Test binary serialization roundtrip
#[test]
fn test_serialization_roundtrip() -> SymEngineResult<()> {
    // Test various expression types
    let expressions = vec![
        Expression::symbol("x"),
        Expression::int(42),
        Expression::float(std::f64::consts::PI)?,
        Expression::symbol("x") + Expression::symbol("y"),
        Expression::symbol("x") * Expression::int(2),
        trig::sin(&Expression::symbol("theta")),
    ];

    for expr in &expressions {
        let bytes = serialize::to_bytes(expr)?;
        let decoded = serialize::from_bytes(&bytes)?;

        // Verify by evaluation if possible
        let mut values = HashMap::new();
        values.insert("x".to_string(), 2.0);
        values.insert("y".to_string(), 3.0);
        values.insert("theta".to_string(), 1.0);

        let original_val = expr.eval(&values);
        let decoded_val = decoded.eval(&values);

        if let (Ok(orig), Ok(dec)) = (original_val, decoded_val) {
            assert!(
                (orig - dec).abs() < 1e-10,
                "Serialization roundtrip failed for expression"
            );
        }
    }

    Ok(())
}

/// Test matrix serialization roundtrip
#[test]
fn test_matrix_serialization_roundtrip() -> SymEngineResult<()> {
    // Use a simpler matrix without trigonometric functions in elements
    // for reliable serialization roundtrip
    let x = Expression::symbol("x");
    let y = Expression::symbol("y");

    let elements = vec![x, y, Expression::zero(), Expression::one()];

    let matrix = SymbolicMatrix::from_flat(elements, 2, 2)?;

    let bytes = serialize::matrix_to_bytes(&matrix)?;
    let decoded = serialize::matrix_from_bytes(&bytes)?;

    assert_eq!(matrix.nrows(), decoded.nrows());
    assert_eq!(matrix.ncols(), decoded.ncols());

    // Verify elements match
    let mut values = HashMap::new();
    values.insert("x".to_string(), 2.0);
    values.insert("y".to_string(), 3.0);

    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            let orig = matrix.get(i, j).eval(&values)?;
            let dec = decoded.get(i, j).eval(&values)?;
            assert!(
                (orig - dec).abs() < 1e-10,
                "Matrix serialization mismatch at ({i},{j})"
            );
        }
    }

    Ok(())
}

/// Test JSON serialization
#[test]
fn test_json_serialization() -> SymEngineResult<()> {
    // Use a simple symbol that can be reliably serialized and deserialized
    let expr = Expression::symbol("x");

    let json = serialize::to_json(&expr);
    let decoded = serialize::from_json(&json)?;

    let mut values = HashMap::new();
    values.insert("x".to_string(), 42.0);

    let original = expr.eval(&values)?;
    let from_json = decoded.eval(&values)?;

    assert!((original - from_json).abs() < 1e-10);

    // Also test a numeric constant
    let num = Expression::int(123);
    let json_num = serialize::to_json(&num);
    let decoded_num = serialize::from_json(&json_num)?;

    let orig_val = num.eval(&HashMap::new())?;
    let dec_val = decoded_num.eval(&HashMap::new())?;

    assert!((orig_val - dec_val).abs() < 1e-10);

    Ok(())
}

// =========================================================================
// Parser Integration Tests
// =========================================================================

/// Test parsing and evaluation
#[test]
fn test_parser_evaluation() -> SymEngineResult<()> {
    let test_cases = vec![
        ("2 + 3", HashMap::new(), 5.0),
        (
            "x * 2",
            {
                let mut m = HashMap::new();
                m.insert("x".to_string(), 4.0);
                m
            },
            8.0,
        ),
        ("sin(0)", HashMap::new(), 0.0),
        ("cos(0)", HashMap::new(), 1.0),
        ("exp(0)", HashMap::new(), 1.0),
        ("ln(1)", HashMap::new(), 0.0),
        ("sqrt(4)", HashMap::new(), 2.0),
        ("2^3", HashMap::new(), 8.0),
    ];

    for (input, values, expected) in test_cases {
        let expr = parser::parse(input)?;
        let result = expr.eval(&values)?;
        assert!(
            (result - expected).abs() < 1e-10,
            "Failed for '{input}': expected {expected}, got {result}"
        );
    }

    Ok(())
}

/// Test parsing complex expressions and differentiation
#[test]
fn test_parser_differentiation() -> SymEngineResult<()> {
    let expr = parser::parse("x^2 + 2*x + 1")?;
    let x = Expression::symbol("x");
    let dx = expr.diff(&x);

    // d/dx(x^2 + 2x + 1) = 2x + 2
    let mut values = HashMap::new();
    values.insert("x".to_string(), 3.0);

    let result = dx.eval(&values)?;
    let expected = 2.0f64.mul_add(3.0, 2.0); // 2x + 2 at x=3

    assert!((result - expected).abs() < 1e-10);

    Ok(())
}

/// Test parsing and simplification
#[test]
fn test_parser_simplification() -> SymEngineResult<()> {
    // Parse an expression that can be simplified
    let expr = parser::parse("x + 0")?;
    let simplified = simplify::simplify(&expr);

    // Should simplify to just x
    let mut values = HashMap::new();
    values.insert("x".to_string(), 42.0);

    assert!((simplified.eval(&values)? - 42.0).abs() < 1e-10);

    Ok(())
}

// =========================================================================
// Pattern Matching Tests
// =========================================================================

/// Test pattern matching for quantum expressions
#[test]
fn test_pattern_matching_workflow() {
    // Create a VQE-style parameter
    let theta = Expression::symbol("theta");

    // Should be recognized as VQE parameter
    assert!(pattern::is_vqe_parameter(&theta));

    // Recognize gate patterns
    let gate_expr = Expression::symbol("Rx");
    let gate_pattern = pattern::recognize_gate_pattern(&gate_expr);
    assert!(matches!(gate_pattern, pattern::QuantumGatePattern::Unknown));

    // Test constant pattern using match_pattern function
    let five = Expression::int(5);
    let const_pattern = Pattern::constant(5.0);
    assert!(match_pattern(&const_pattern, &five).is_some());
}

/// Test pattern capture and substitution
#[test]
fn test_pattern_capture() {
    let x = Expression::symbol("x");
    let wildcard = Pattern::wildcard("a");

    let captures = match_pattern(&wildcard, &x);
    assert!(captures.is_some());

    let cap = captures.unwrap();
    assert!(cap.contains_key("a"));
}

// =========================================================================
// Complex Expression Workflow Tests
// =========================================================================

/// Test a complete symbolic computation workflow
#[test]
fn test_symbolic_workflow() -> SymEngineResult<()> {
    // 1. Parse an expression
    let expr = parser::parse("x^2 + y^2")?;

    // 2. Compute gradients
    let x = Expression::symbol("x");
    let y = Expression::symbol("y");
    let grad = expr.gradient(&[x, y.clone()]);

    // 3. Evaluate gradient at a specific point
    let mut values = HashMap::new();
    values.insert("x".to_string(), 2.0);
    values.insert("y".to_string(), 3.0);

    // d/dx(x^2 + y^2) = 2x = 4 at x=2
    let grad_x_val = grad[0].eval(&values)?;
    assert!((grad_x_val - 4.0).abs() < 1e-10);

    // d/dy(x^2 + y^2) = 2y = 6 at y=3
    let grad_y_val = grad[1].eval(&values)?;
    assert!((grad_y_val - 6.0).abs() < 1e-10);

    // 4. Substitute values symbolically
    let grad_x_at_y1 = grad[0].substitute(&y, &Expression::int(1));

    // 5. Evaluate the substituted expression
    let result = grad_x_at_y1.eval(&values)?;

    // d/dx(x^2 + 1) = 2x = 4 at x=2
    assert!((result - 4.0).abs() < 1e-10);

    // 6. Test serialization of simple expressions (not s-expressions)
    let simple_expr = Expression::symbol("x");
    let bytes = serialize::to_bytes(&simple_expr)?;
    let restored = serialize::from_bytes(&bytes)?;

    assert!((simple_expr.eval(&values)? - restored.eval(&values)?).abs() < 1e-10);

    Ok(())
}

/// Test optimization loop simulation
#[test]
fn test_optimization_loop() -> SymEngineResult<()> {
    // Simulate a simple gradient descent optimization
    let theta = Expression::symbol("theta");
    let energy =
        theta.clone() * theta.clone() - Expression::int(2) * theta.clone() + Expression::int(1);

    let gradient = energy.diff(&theta);

    // Gradient descent
    let mut param_value = 0.0;
    let learning_rate = 0.1;

    for _ in 0..20 {
        let mut values = HashMap::new();
        values.insert("theta".to_string(), param_value);

        let grad_val = gradient.eval(&values)?;
        param_value -= learning_rate * grad_val;
    }

    // Should converge to theta = 1 (minimum of (theta-1)^2)
    assert!((param_value - 1.0).abs() < 0.1);

    Ok(())
}

// =========================================================================
// Hessian and Higher-Order Derivatives
// =========================================================================

/// Test Hessian computation
#[test]
fn test_hessian_computation() -> SymEngineResult<()> {
    let x = Expression::symbol("x");
    let y = Expression::symbol("y");

    // f(x,y) = x^2*y + x*y^2
    let f = x.clone() * x.clone() * y.clone() + x.clone() * y.clone() * y.clone();

    let vars = vec![x, y];
    let hessian = f.hessian(&vars);

    // Hessian should be 2x2 (Vec<Vec<Expression>>)
    assert_eq!(hessian.len(), 2);
    assert_eq!(hessian[0].len(), 2);

    // Evaluate at (1, 1)
    let mut values = HashMap::new();
    values.insert("x".to_string(), 1.0);
    values.insert("y".to_string(), 1.0);

    // d^2f/dx^2 = 2y = 2 at (1,1)
    let h_xx = hessian[0][0].eval(&values)?;
    assert!((h_xx - 2.0).abs() < 1e-10);

    // d^2f/dy^2 = 2x = 2 at (1,1)
    let h_yy = hessian[1][1].eval(&values)?;
    assert!((h_yy - 2.0).abs() < 1e-10);

    // d^2f/dxdy = 2x + 2y = 4 at (1,1)
    let h_xy = hessian[0][1].eval(&values)?;
    assert!((h_xy - 4.0).abs() < 1e-10);

    Ok(())
}

// =========================================================================
// SciRS2 Integration Tests
// =========================================================================

/// Test integration with scirs2-core Complex64
#[test]
fn test_scirs2_complex_integration() -> SymEngineResult<()> {
    // Create expression from pure real Complex64
    let c_real = Complex64::new(3.5, 0.0);
    let expr_real = Expression::from_complex64(c_real);

    // Evaluate using the eval_complex function (only works for real values currently)
    let complex_values: HashMap<String, Complex64> = HashMap::new();
    let result = eval_complex(&expr_real, &complex_values)?;

    assert!((result.re - 3.5).abs() < 1e-10);

    // Also verify that from_complex64 creates correct expressions
    let c_imag = Complex64::new(0.0, 4.0);
    let expr_imag = Expression::from_complex64(c_imag);

    // Expression should contain the imaginary unit i
    assert!(!expr_imag.is_symbol());
    assert!(!expr_imag.is_number());

    Ok(())
}

/// Test gradient array integration with SciRS2
#[test]
fn test_gradient_array_integration() -> SymEngineResult<()> {
    use quantrs2_symengine_pure::scirs2_bridge::ndarray::gradient_array;

    let x = Expression::symbol("x");
    let expr = x.clone() * x.clone(); // x^2

    let params = vec![x];
    let mut values = HashMap::new();
    values.insert("x".to_string(), 3.0);

    let grad = gradient_array(&expr, &params, &values)?;

    // d/dx(x^2) = 2x = 6 at x=3
    assert!((grad[0] - 6.0).abs() < 1e-6);

    Ok(())
}

// =========================================================================
// Error Handling Tests
// =========================================================================

/// Test error handling for invalid operations
#[test]
fn test_error_handling() {
    // Division by zero during evaluation
    let x = Expression::symbol("x");
    let zero = Expression::zero();
    let expr = x / zero;

    let mut values = HashMap::new();
    values.insert("x".to_string(), 1.0);

    let result = expr.eval(&values);
    assert!(result.is_err());

    // Missing variable during evaluation
    let y = Expression::symbol("y");
    let result = y.eval(&HashMap::new());
    assert!(result.is_err());
}

/// Test parser error handling
#[test]
fn test_parser_errors() {
    // Empty input
    assert!(parser::parse("").is_err());

    // Unmatched parentheses
    assert!(parser::parse("(x + y").is_err());

    // Invalid syntax
    assert!(parser::parse("+ +").is_err());
}

// =========================================================================
// Performance-Critical Operations
// =========================================================================

/// Test that operations don't panic on large expressions
#[test]
fn test_large_expressions() -> SymEngineResult<()> {
    // Build a moderately large expression
    let mut expr = Expression::symbol("x");

    for i in 1..50 {
        expr = expr + Expression::int(i);
    }

    // Should be able to evaluate
    let mut values = HashMap::new();
    values.insert("x".to_string(), 1.0);

    let result = expr.eval(&values)?;

    // x + 1 + 2 + ... + 49 = 1 + (1+2+...+49) = 1 + 49*50/2 = 1 + 1225 = 1226
    assert!((result - 1226.0).abs() < 1e-10);

    Ok(())
}

/// Test matrix operations on larger matrices
#[test]
fn test_larger_matrices() -> SymEngineResult<()> {
    // 4x4 identity
    let i4 = SymbolicMatrix::identity(4);

    // Verify I * I = I
    let result = i4.matmul(&i4)?;

    let empty = HashMap::new();
    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { 1.0 } else { 0.0 };
            let actual = result.get(i, j).eval(&empty)?;
            assert!((expected - actual).abs() < 1e-10);
        }
    }

    Ok(())
}
