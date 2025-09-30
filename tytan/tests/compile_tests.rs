//! Tests for the compile module.

use scirs2_core::ndarray::Array;
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
    let expr = x.clone() + y.clone() * 2;

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
    let expr = x.clone() * y.clone() + x.clone().pow(quantrs2_symengine::Expression::from(2));

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
    let expr = (x.clone() + y.clone() + z.clone() - 1).pow(quantrs2_symengine::Expression::from(2));

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
#[ignore] // Enable when HOBO support is fully implemented
fn test_compile_cubic_expression() {
    // Test compiling a cubic expression
    let x = symbols("x");
    let y = symbols("y");
    let z = symbols("z");

    // Cubic expression: x*y*z
    let expr = x.clone() * y.clone() * z.clone();

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
    let mut sorted_indices = indices.clone();
    sorted_indices.sort();

    // Check that the tensor has a 1.0 at the expected position
    assert_eq!(tensor[scirs2_core::ndarray::IxDyn(&sorted_indices)], 1.0);
}

// TODO: This test needs to be rewritten to use expressions instead of raw matrix
// #[test]
// #[cfg(feature = "dwave")]
// fn test_compile_matrix_input() {
//     // Test compiling from a matrix input
//     // Create a 3x3 QUBO matrix directly
//     let mut matrix = Array::zeros((3, 3));
//
//     // Set some values
//     matrix[[0, 0]] = -3.0; // Linear term for variable 0
//     matrix[[1, 1]] = -3.0; // Linear term for variable 1
//     matrix[[2, 2]] = -3.0; // Linear term for variable 2
//     matrix[[0, 1]] = 2.0; // Quadratic term between variables 0 and 1
//     matrix[[0, 2]] = 2.0; // Quadratic term between variables 0 and 2
//     matrix[[1, 2]] = 2.0; // Quadratic term between variables 1 and 2
//
//     // Make matrix symmetric
//     matrix[[1, 0]] = matrix[[0, 1]];
//     matrix[[2, 0]] = matrix[[0, 2]];
//     matrix[[2, 1]] = matrix[[1, 2]];
//
//     // Compile
//     // let (qubo, offset) = Compile::new(matrix).get_qubo().unwrap();
//     let (result_matrix, var_map) = qubo;
//
//     // Check offset
//     assert_eq!(offset, 0.0);
//
//     // Check variable map
//     assert_eq!(var_map.len(), 3);
//
//     // Check matrix dimensions
//     assert_eq!(result_matrix.shape(), &[3, 3]);
//
//     // Check that the matrices are the same
//     for i in 0..3 {
//         for j in 0..3 {
//             assert_eq!(matrix[[i, j]], result_matrix[[i, j]]);
//         }
//     }
// }
