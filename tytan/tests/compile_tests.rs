#![allow(clippy::pedantic, clippy::unnecessary_wraps)]
//! Tests for the compile module.

use quantrs2_tytan::*;

#[cfg(feature = "dwave")]
use quantrs2_tytan::compile::Compile;
#[cfg(feature = "dwave")]
use quantrs2_tytan::symbol::symbols;

#[test]
#[cfg(feature = "dwave")]
fn test_compile_simple_expression() {
    // Test compiling a simple expression
    let x = symbols("x");
    let y = symbols("y");

    // Simple linear expression: x + 2*y
    let two = quantrs2_symengine_pure::Expression::from(2);
    let expr = x + y * two;

    // Compile to QUBO
    let (qubo, offset) = Compile::new(expr).get_qubo().unwrap();
    let (matrix, var_map) = qubo;

    // Check offset
    assert_eq!(offset, 0.0);

    // Check variable map
    assert_eq!(var_map.len(), 2);
    assert!(var_map.contains_key("x"));
    assert!(var_map.contains_key("y"));

    // Check matrix dimensions
    assert_eq!(matrix.shape(), &[2, 2]);

    // Check matrix values
    let x_idx = var_map["x"];
    let y_idx = var_map["y"];

    assert_eq!(matrix[[x_idx, x_idx]], 1.0); // Coefficient of x
    assert_eq!(matrix[[y_idx, y_idx]], 2.0); // Coefficient of y
}

#[test]
#[cfg(feature = "dwave")]
fn test_compile_quadratic_expression() {
    // Test compiling a quadratic expression
    let x = symbols("x");
    let y = symbols("y");

    // Quadratic expression: x*y + x^2 (which is just x for binary variables)
    let expr = x.clone() * y + x.pow(&quantrs2_symengine_pure::Expression::from(2));

    // Compile to QUBO
    let (qubo, offset) = Compile::new(expr).get_qubo().unwrap();
    let (matrix, var_map) = qubo;

    // Check offset
    assert_eq!(offset, 0.0);

    // Check variable map
    assert_eq!(var_map.len(), 2);

    // Check matrix dimensions
    assert_eq!(matrix.shape(), &[2, 2]);

    // Check matrix values
    let x_idx = var_map["x"];
    let y_idx = var_map["y"];

    assert_eq!(matrix[[x_idx, x_idx]], 1.0); // From x^2 which becomes x

    // The quadratic term should be divided between the two locations
    assert!(matrix[[x_idx, y_idx]] == 1.0 || matrix[[y_idx, x_idx]] == 1.0);
}

#[test]
#[cfg(feature = "dwave")]
fn test_compile_constraint_expression() {
    // Test compiling a constraint expression
    // For example, exactly one of x, y, z must be 1
    let x = symbols("x");
    let y = symbols("y");
    let z = symbols("z");

    // Constraint: (x + y + z - 1)^2
    let one = quantrs2_symengine_pure::Expression::from(1);
    let two = quantrs2_symengine_pure::Expression::from(2);
    let expr = (x + y + z - one).pow(&two);

    // Compile to QUBO
    let (qubo, offset) = Compile::new(expr).get_qubo().unwrap();
    let (matrix, var_map) = qubo;

    // Check offset
    assert_eq!(offset, 1.0); // From the constant term in the expansion

    // Check variable map
    assert_eq!(var_map.len(), 3);

    // Check matrix dimensions
    assert_eq!(matrix.shape(), &[3, 3]);

    // Check matrix values - specific values depend on variable ordering
    // but we can check some properties

    // Linear terms: from (x+y+z-1)^2 = x^2+y^2+z^2+2xy+2xz+2yz-2x-2y-2z+1
    // With x^2=x for binary: x+y+z+2xy+2xz+2yz-2x-2y-2z+1 = -x-y-z+2xy+2xz+2yz+1
    let x_idx = var_map["x"];
    let y_idx = var_map["y"];
    let z_idx = var_map["z"];

    assert_eq!(matrix[[x_idx, x_idx]], -1.0);
    assert_eq!(matrix[[y_idx, y_idx]], -1.0);
    assert_eq!(matrix[[z_idx, z_idx]], -1.0);

    // Quadratic terms should all be 2.0 (coefficient of x*y, x*z, y*z in the expansion)
    // Depending on how the matrix is stored, check both locations for symmetry
    assert_eq!(matrix[[x_idx, y_idx]], 2.0);
    assert_eq!(matrix[[x_idx, z_idx]], 2.0);
    assert_eq!(matrix[[y_idx, z_idx]], 2.0);
}

#[test]
#[cfg(feature = "dwave")]
fn test_compile_cubic_expression() {
    // Test compiling a cubic expression
    let x = symbols("x");
    let y = symbols("y");
    let z = symbols("z");

    // Cubic expression: x*y*z
    let expr = x * y * z;

    // Compile to HOBO
    let (hobo, offset) = Compile::new(expr).get_hobo().unwrap();
    let (tensor, var_map) = hobo;

    // Check offset
    assert_eq!(offset, 0.0);

    // Check variable map
    assert_eq!(var_map.len(), 3);

    // Check tensor dimensions
    assert_eq!(tensor.ndim(), 3);
    assert_eq!(tensor.shape(), &[3, 3, 3]);

    // Check tensor values - the cubic term should be 1.0
    let x_idx = var_map["x"];
    let y_idx = var_map["y"];
    let z_idx = var_map["z"];

    // This assumes the tensor is stored in a canonical form where indices are ordered
    let indices = [x_idx, y_idx, z_idx];
    let mut sorted_indices = indices;
    sorted_indices.sort_unstable();

    // Check that the tensor has a 1.0 at the expected position
    assert_eq!(tensor[scirs2_core::ndarray::IxDyn(&sorted_indices)], 1.0);
}

#[test]
#[cfg(feature = "dwave")]
fn test_compile_matrix_input() {
    // `Compile` only accepts symbolic expressions (there is no raw-matrix
    // constructor), so this test rebuilds the same 3x3 QUBO the original
    // (disabled) raw-matrix version constructed, via the expression /
    // SymbolBuilder API:
    //
    //   diag(0,0) = diag(1,1) = diag(2,2) = -3.0   (linear terms)
    //   off-diag  (0,1) = (0,2) = (1,2)    =  2.0   (quadratic terms, symmetric)
    //
    // As an expression: -3*x0 - 3*x1 - 3*x2 + 2*x0*x1 + 2*x0*x2 + 2*x1*x2
    let x0 = symbols("x0");
    let x1 = symbols("x1");
    let x2 = symbols("x2");

    let neg_three = quantrs2_symengine_pure::Expression::from(-3);
    let two = quantrs2_symengine_pure::Expression::from(2);

    let expr = neg_three.clone() * x0.clone()
        + neg_three.clone() * x1.clone()
        + neg_three * x2.clone()
        + two.clone() * x0.clone() * x1.clone()
        + two.clone() * x0.clone() * x2.clone()
        + two * x1.clone() * x2.clone();

    // Compile to QUBO
    let (qubo, offset) = Compile::new(expr).get_qubo().unwrap();
    let (matrix, var_map) = qubo;

    // Check offset
    assert_eq!(offset, 0.0);

    // Check variable map
    assert_eq!(var_map.len(), 3);
    assert!(var_map.contains_key("x0"));
    assert!(var_map.contains_key("x1"));
    assert!(var_map.contains_key("x2"));

    // Check matrix dimensions
    assert_eq!(matrix.shape(), &[3, 3]);

    let x0_idx = var_map["x0"];
    let x1_idx = var_map["x1"];
    let x2_idx = var_map["x2"];

    // Linear (diagonal) terms
    assert_eq!(matrix[[x0_idx, x0_idx]], -3.0);
    assert_eq!(matrix[[x1_idx, x1_idx]], -3.0);
    assert_eq!(matrix[[x2_idx, x2_idx]], -3.0);

    // Quadratic terms are stored once, in whichever half is upper
    // triangular for the assigned variable ordering (see
    // `test_compile_constraint_expression` for the same convention).
    let pair = |a: usize, b: usize| -> f64 {
        if a <= b {
            matrix[[a, b]]
        } else {
            matrix[[b, a]]
        }
    };
    assert_eq!(pair(x0_idx, x1_idx), 2.0);
    assert_eq!(pair(x0_idx, x2_idx), 2.0);
    assert_eq!(pair(x1_idx, x2_idx), 2.0);
}
