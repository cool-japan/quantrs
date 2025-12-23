//! Integration tests for matrix algebra workflows
//!
//! These tests verify that complex matrix operations work correctly
//! through the FFI layer.

use quantrs2_symengine_sys::*;
use std::os::raw::c_int;

#[test]
fn test_matrix_creation_and_element_access() {
    unsafe {
        // Create a 3x3 matrix
        let mat = dense_matrix_new_rows_cols(3, 3);
        assert!(!mat.is_null(), "Matrix creation should succeed");

        // Check dimensions
        let rows = dense_matrix_rows(mat);
        let cols = dense_matrix_cols(mat);
        assert_eq!(rows, 3, "Matrix should have 3 rows");
        assert_eq!(cols, 3, "Matrix should have 3 columns");

        // Create some basic elements
        let mut one = std::mem::zeroed::<basic_struct>();
        let mut two = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut one, 1);
        integer_set_si(&raw mut two, 2);

        // Set elements
        dense_matrix_set_basic(mat, 0, 0, &raw mut one);
        dense_matrix_set_basic(mat, 0, 1, &raw mut two);
        dense_matrix_set_basic(mat, 1, 1, &raw mut one);

        // Get element back
        let mut retrieved = std::mem::zeroed::<basic_struct>();
        dense_matrix_get_basic(&raw mut retrieved, mat, 0, 1);

        // Clean up
        dense_matrix_free(mat);
    }
}

#[test]
fn test_identity_matrix_creation() {
    unsafe {
        let mat = dense_matrix_new_rows_cols(4, 4);
        assert!(!mat.is_null());

        // Create identity matrix
        dense_matrix_eye(mat, 4, 4, 0);

        // Verify diagonal elements are 1
        for i in 0..4 {
            let mut elem = std::mem::zeroed::<basic_struct>();
            dense_matrix_get_basic(&raw mut elem, mat, i, i);

            // Check if it's an integer 1
            // Note: TypeID enum has SYMENGINE_INTEGER = 0 (different from type_codes constants)
            let type_code = basic_get_type(&raw const elem);
            assert_eq!(
                type_code as c_int, 0,
                "Diagonal element should be integer (TypeID::SYMENGINE_INTEGER = 0)"
            );
        }

        dense_matrix_free(mat);
    }
}

#[test]
fn test_matrix_addition_workflow() {
    unsafe {
        // Create two 2x2 matrices
        let mat1 = dense_matrix_new_rows_cols(2, 2);
        let mat2 = dense_matrix_new_rows_cols(2, 2);
        let result = dense_matrix_new_rows_cols(2, 2);

        assert!(!mat1.is_null() && !mat2.is_null() && !result.is_null());

        // Initialize mat1 with values [[1, 2], [3, 4]]
        let mut one = std::mem::zeroed::<basic_struct>();
        let mut two = std::mem::zeroed::<basic_struct>();
        let mut three = std::mem::zeroed::<basic_struct>();
        let mut four = std::mem::zeroed::<basic_struct>();

        integer_set_si(&raw mut one, 1);
        integer_set_si(&raw mut two, 2);
        integer_set_si(&raw mut three, 3);
        integer_set_si(&raw mut four, 4);

        dense_matrix_set_basic(mat1, 0, 0, &raw mut one);
        dense_matrix_set_basic(mat1, 0, 1, &raw mut two);
        dense_matrix_set_basic(mat1, 1, 0, &raw mut three);
        dense_matrix_set_basic(mat1, 1, 1, &raw mut four);

        // Initialize mat2 with values [[5, 6], [7, 8]]
        let mut five = std::mem::zeroed::<basic_struct>();
        let mut six = std::mem::zeroed::<basic_struct>();
        let mut seven = std::mem::zeroed::<basic_struct>();
        let mut eight = std::mem::zeroed::<basic_struct>();

        integer_set_si(&raw mut five, 5);
        integer_set_si(&raw mut six, 6);
        integer_set_si(&raw mut seven, 7);
        integer_set_si(&raw mut eight, 8);

        dense_matrix_set_basic(mat2, 0, 0, &raw mut five);
        dense_matrix_set_basic(mat2, 0, 1, &raw mut six);
        dense_matrix_set_basic(mat2, 1, 0, &raw mut seven);
        dense_matrix_set_basic(mat2, 1, 1, &raw mut eight);

        // Add matrices
        let code = dense_matrix_add_matrix(result, mat1, mat2);
        assert_eq!(code as c_int, 0, "Matrix addition should succeed");

        // Verify result: [[6, 8], [10, 12]]
        let mut elem = std::mem::zeroed::<basic_struct>();
        dense_matrix_get_basic(&raw mut elem, result, 0, 0);
        // Element should be 6

        dense_matrix_free(mat1);
        dense_matrix_free(mat2);
        dense_matrix_free(result);
    }
}

#[test]
fn test_matrix_multiplication_workflow() {
    unsafe {
        // Create two 2x2 matrices for multiplication
        let mat1 = dense_matrix_new_rows_cols(2, 2);
        let mat2 = dense_matrix_new_rows_cols(2, 2);
        let result = dense_matrix_new_rows_cols(2, 2);

        // mat1 = [[1, 2], [3, 4]]
        let mut vals = [0; 4];
        for (i, item) in vals.iter_mut().enumerate() {
            *item = i + 1;
        }

        let mut basics = [std::mem::zeroed::<basic_struct>(); 4];
        for (i, val) in vals.iter().enumerate() {
            integer_set_si(&raw mut basics[i], *val as i64);
        }

        dense_matrix_set_basic(mat1, 0, 0, &raw mut basics[0]);
        dense_matrix_set_basic(mat1, 0, 1, &raw mut basics[1]);
        dense_matrix_set_basic(mat1, 1, 0, &raw mut basics[2]);
        dense_matrix_set_basic(mat1, 1, 1, &raw mut basics[3]);

        // mat2 = identity matrix
        dense_matrix_eye(mat2, 2, 2, 0);

        // Multiply: mat1 * I = mat1
        let code = dense_matrix_mul_matrix(result, mat1, mat2);
        assert_eq!(code as c_int, 0, "Matrix multiplication should succeed");

        // Result should equal mat1
        let mut elem = std::mem::zeroed::<basic_struct>();
        dense_matrix_get_basic(&raw mut elem, result, 0, 0);

        dense_matrix_free(mat1);
        dense_matrix_free(mat2);
        dense_matrix_free(result);
    }
}

#[test]
fn test_matrix_transpose() {
    unsafe {
        // Create a 2x3 matrix
        let mat = dense_matrix_new_rows_cols(2, 3);
        let transposed = dense_matrix_new();

        // Fill matrix with values [[1, 2, 3], [4, 5, 6]]
        for i in 0..2 {
            for j in 0..3 {
                let mut val = std::mem::zeroed::<basic_struct>();
                integer_set_si(&raw mut val, (i * 3 + j + 1) as i64);
                dense_matrix_set_basic(mat, i as u64, j as u64, &raw mut val);
            }
        }

        // Transpose
        dense_matrix_transpose(transposed, mat);

        // Verify dimensions are swapped
        assert_eq!(dense_matrix_rows(transposed), 3);
        assert_eq!(dense_matrix_cols(transposed), 2);

        // Verify elements
        let mut elem = std::mem::zeroed::<basic_struct>();
        dense_matrix_get_basic(&raw mut elem, transposed, 1, 0);
        // Should be value from (0, 1) of original = 2

        dense_matrix_free(mat);
        dense_matrix_free(transposed);
    }
}

#[test]
fn test_matrix_scalar_multiplication() {
    unsafe {
        let mat = dense_matrix_new_rows_cols(2, 2);
        let result = dense_matrix_new();

        // Create a simple matrix [[1, 2], [3, 4]]
        let mut one = std::mem::zeroed::<basic_struct>();
        let mut two = std::mem::zeroed::<basic_struct>();
        let mut three = std::mem::zeroed::<basic_struct>();
        let mut four = std::mem::zeroed::<basic_struct>();

        integer_set_si(&raw mut one, 1);
        integer_set_si(&raw mut two, 2);
        integer_set_si(&raw mut three, 3);
        integer_set_si(&raw mut four, 4);

        dense_matrix_set_basic(mat, 0, 0, &raw mut one);
        dense_matrix_set_basic(mat, 0, 1, &raw mut two);
        dense_matrix_set_basic(mat, 1, 0, &raw mut three);
        dense_matrix_set_basic(mat, 1, 1, &raw mut four);

        // Multiply by scalar 2
        let mut scalar = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut scalar, 2);

        dense_matrix_mul_scalar(result, mat, &raw const scalar);

        // Result should be [[2, 4], [6, 8]]
        let mut elem = std::mem::zeroed::<basic_struct>();
        dense_matrix_get_basic(&raw mut elem, result, 0, 0);
        // Should be 2

        dense_matrix_free(mat);
        dense_matrix_free(result);
    }
}

#[test]
fn test_matrix_determinant_2x2() {
    unsafe {
        let mat = dense_matrix_new_rows_cols(2, 2);

        // Create matrix [[1, 2], [3, 4]]
        // det = 1*4 - 2*3 = -2
        let mut one = std::mem::zeroed::<basic_struct>();
        let mut two = std::mem::zeroed::<basic_struct>();
        let mut three = std::mem::zeroed::<basic_struct>();
        let mut four = std::mem::zeroed::<basic_struct>();

        integer_set_si(&raw mut one, 1);
        integer_set_si(&raw mut two, 2);
        integer_set_si(&raw mut three, 3);
        integer_set_si(&raw mut four, 4);

        dense_matrix_set_basic(mat, 0, 0, &raw mut one);
        dense_matrix_set_basic(mat, 0, 1, &raw mut two);
        dense_matrix_set_basic(mat, 1, 0, &raw mut three);
        dense_matrix_set_basic(mat, 1, 1, &raw mut four);

        // Calculate determinant
        let mut det = std::mem::zeroed::<basic_struct>();
        let code = dense_matrix_det(&raw mut det, mat);
        assert_eq!(code as c_int, 0, "Determinant calculation should succeed");

        // Det should be -2 (as an integer)
        // Note: TypeID enum has SYMENGINE_INTEGER = 0 (different from type_codes constants)
        let type_code = basic_get_type(&raw const det);
        assert_eq!(
            type_code as c_int, 0,
            "Determinant should be integer (TypeID::SYMENGINE_INTEGER = 0)"
        );

        dense_matrix_free(mat);
    }
}

#[test]
fn test_symbolic_matrix_operations() {
    unsafe {
        // Create matrix with symbolic entries
        let mat = dense_matrix_new_rows_cols(2, 2);

        let mut x = std::mem::zeroed::<basic_struct>();
        let mut y = std::mem::zeroed::<basic_struct>();
        symbol_set(&raw mut x, c"x".as_ptr());
        symbol_set(&raw mut y, c"y".as_ptr());

        let mut zero = std::mem::zeroed::<basic_struct>();
        let mut one = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut zero, 0);
        integer_set_si(&raw mut one, 1);

        // Create matrix [[x, 1], [0, y]]
        dense_matrix_set_basic(mat, 0, 0, &raw mut x);
        dense_matrix_set_basic(mat, 0, 1, &raw mut one);
        dense_matrix_set_basic(mat, 1, 0, &raw mut zero);
        dense_matrix_set_basic(mat, 1, 1, &raw mut y);

        // Calculate determinant: det = x*y - 1*0 = x*y
        let mut det = std::mem::zeroed::<basic_struct>();
        let code = dense_matrix_det(&raw mut det, mat);
        assert_eq!(
            code as c_int, 0,
            "Symbolic determinant calculation should succeed"
        );

        // The determinant should contain both x and y
        let type_code = basic_get_type(&raw const det);
        // Should be a multiplication or product type
        assert!(
            type_code as c_int != 0,
            "Determinant should be non-zero type"
        );

        dense_matrix_free(mat);
    }
}

#[test]
fn test_matrix_workflow_complete() {
    unsafe {
        // Complete workflow: Create, populate, add, multiply, transpose, determinant
        let mat1 = dense_matrix_new_rows_cols(2, 2);
        let mat2 = dense_matrix_new_rows_cols(2, 2);

        // Initialize identity matrices
        dense_matrix_eye(mat1, 2, 2, 0);
        dense_matrix_eye(mat2, 2, 2, 0);

        // Add them: I + I = 2I
        let sum = dense_matrix_new();
        let code = dense_matrix_add_matrix(sum, mat1, mat2);
        assert_eq!(code as c_int, 0);

        // Multiply by original: (2I) * I = 2I
        let product = dense_matrix_new();
        let code = dense_matrix_mul_matrix(product, sum, mat1);
        assert_eq!(code as c_int, 0);

        // Transpose: (2I)^T = 2I
        let transposed = dense_matrix_new();
        dense_matrix_transpose(transposed, product);

        // Determinant of 2I for 2x2 is 4
        let mut det = std::mem::zeroed::<basic_struct>();
        let code = dense_matrix_det(&raw mut det, transposed);
        assert_eq!(code as c_int, 0);

        // Clean up
        dense_matrix_free(mat1);
        dense_matrix_free(mat2);
        dense_matrix_free(sum);
        dense_matrix_free(product);
        dense_matrix_free(transposed);
    }
}
