//! Matrix operations example
//!
//! This example demonstrates how to use the SymEngine matrix operations
//! for symbolic linear algebra.

use quantrs2_symengine_sys::*;
use std::ffi::CStr;
use std::os::raw::c_int;

fn main() {
    unsafe {
        println!("SymEngine Matrix Operations Example");
        println!("====================================\n");

        // Create symbols for matrix elements
        let mut a = std::mem::zeroed::<basic_struct>();
        let mut b = std::mem::zeroed::<basic_struct>();
        let mut c = std::mem::zeroed::<basic_struct>();
        let mut d = std::mem::zeroed::<basic_struct>();

        symbol_set(&raw mut a, c"a".as_ptr());
        symbol_set(&raw mut b, c"b".as_ptr());
        symbol_set(&raw mut c, c"c".as_ptr());
        symbol_set(&raw mut d, c"d".as_ptr());

        // Create a 2x2 matrix: [[a, b], [c, d]]
        let mat = dense_matrix_new_rows_cols(2, 2);
        dense_matrix_set_basic(mat, 0, 0, &raw mut a);
        dense_matrix_set_basic(mat, 0, 1, &raw mut b);
        dense_matrix_set_basic(mat, 1, 0, &raw mut c);
        dense_matrix_set_basic(mat, 1, 1, &raw mut d);

        // Print the matrix
        let str_ptr = dense_matrix_str(mat);
        let c_str = CStr::from_ptr(str_ptr);
        println!("Matrix M =");
        println!("{}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Compute determinant: ad - bc
        let mut det = std::mem::zeroed::<basic_struct>();
        let code = dense_matrix_det(&raw mut det, mat);
        check_result(code as c_int).expect("Failed to compute determinant");

        let str_ptr = basic_str(&raw const det);
        let c_str = CStr::from_ptr(str_ptr);
        println!("\ndet(M) = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Create a transpose
        let mat_t = dense_matrix_new();
        let code = dense_matrix_transpose(mat_t, mat);
        check_result(code as c_int).expect("Failed to transpose");

        let str_ptr = dense_matrix_str(mat_t);
        let c_str = CStr::from_ptr(str_ptr);
        println!("\nTranspose(M) =");
        println!("{}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Create identity matrix
        let identity = dense_matrix_new();
        let code = dense_matrix_eye(identity, 2, 2, 0);
        check_result(code as c_int).expect("Failed to create identity");

        let str_ptr = dense_matrix_str(identity);
        let c_str = CStr::from_ptr(str_ptr);
        println!("\nIdentity matrix =");
        println!("{}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Matrix addition: M + I
        let mat_plus_i = dense_matrix_new();
        let code = dense_matrix_add_matrix(mat_plus_i, mat, identity);
        check_result(code as c_int).expect("Failed to add matrices");

        let str_ptr = dense_matrix_str(mat_plus_i);
        let c_str = CStr::from_ptr(str_ptr);
        println!("\nM + I =");
        println!("{}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Scalar multiplication: 2*M
        let mut two = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut two, 2);

        let mat_times_2 = dense_matrix_new();
        let code = dense_matrix_mul_scalar(mat_times_2, mat, &raw const two);
        check_result(code as c_int).expect("Failed to multiply by scalar");

        let str_ptr = dense_matrix_str(mat_times_2);
        let c_str = CStr::from_ptr(str_ptr);
        println!("\n2*M =");
        println!("{}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Differentiate matrix with respect to a
        let mat_diff = dense_matrix_new();
        let code = dense_matrix_diff(mat_diff, mat, &raw const a);
        check_result(code as c_int).expect("Failed to differentiate matrix");

        let str_ptr = dense_matrix_str(mat_diff);
        let c_str = CStr::from_ptr(str_ptr);
        println!("\nd/da(M) =");
        println!("{}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Clean up
        dense_matrix_free(mat);
        dense_matrix_free(mat_t);
        dense_matrix_free(identity);
        dense_matrix_free(mat_plus_i);
        dense_matrix_free(mat_times_2);
        dense_matrix_free(mat_diff);

        println!("\n✓ All matrix operations completed successfully!");
    }
}
