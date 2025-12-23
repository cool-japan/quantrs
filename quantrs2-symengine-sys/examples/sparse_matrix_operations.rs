//! Sparse matrix operations example
//!
//! This example demonstrates how to use the quantrs2-symengine-sys FFI bindings
//! for sparse matrix operations. Sparse matrices are efficient for storing
//! large matrices with mostly zero elements.

use quantrs2_symengine_sys::*;
use std::ffi::CStr;

fn main() {
    unsafe {
        println!("SymEngine Sparse Matrix Operations Example");
        println!("============================================\n");

        // Create a new sparse matrix
        let sparse = sparse_matrix_new();
        assert!(!sparse.is_null(), "Sparse matrix creation should succeed");

        // Initialize the sparse matrix
        sparse_matrix_init(sparse);

        println!("✓ Created and initialized sparse matrix");

        // Create symbolic elements
        let mut x = std::mem::zeroed::<basic_struct>();
        let mut y = std::mem::zeroed::<basic_struct>();
        let mut zero = std::mem::zeroed::<basic_struct>();
        let mut one = std::mem::zeroed::<basic_struct>();
        let mut two = std::mem::zeroed::<basic_struct>();

        symbol_set(&raw mut x, c"x".as_ptr());
        symbol_set(&raw mut y, c"y".as_ptr());
        integer_set_si(&raw mut zero, 0);
        integer_set_si(&raw mut one, 1);
        integer_set_si(&raw mut two, 2);

        println!("\n--- Setting Sparse Matrix Elements ---");

        // Set some elements in the sparse matrix
        // Sparse matrices are useful when most elements are zero
        // Let's create a 5x5 matrix with only a few non-zero elements

        // Set element at (0, 0) = x
        sparse_matrix_set_basic(sparse, 0, 0, &raw mut x);
        println!("Set (0, 0) = x");

        // Set element at (1, 1) = y
        sparse_matrix_set_basic(sparse, 1, 1, &raw mut y);
        println!("Set (1, 1) = y");

        // Set element at (2, 2) = 2
        sparse_matrix_set_basic(sparse, 2, 2, &raw mut two);
        println!("Set (2, 2) = 2");

        // Set element at (0, 4) = 1 (off-diagonal element)
        sparse_matrix_set_basic(sparse, 0, 4, &raw mut one);
        println!("Set (0, 4) = 1");

        // Set element at (4, 0) = 1 (symmetric element)
        sparse_matrix_set_basic(sparse, 4, 0, &raw mut one);
        println!("Set (4, 0) = 1");

        println!("\n--- Getting Sparse Matrix Elements ---");

        // Get and display the non-zero elements we set
        let mut elem_00 = std::mem::zeroed::<basic_struct>();
        sparse_matrix_get_basic(&raw mut elem_00, sparse, 0, 0);
        let str_ptr = basic_str(&raw const elem_00);
        let c_str = CStr::from_ptr(str_ptr);
        println!("Element (0, 0) = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        let mut elem_11 = std::mem::zeroed::<basic_struct>();
        sparse_matrix_get_basic(&raw mut elem_11, sparse, 1, 1);
        let str_ptr = basic_str(&raw const elem_11);
        let c_str = CStr::from_ptr(str_ptr);
        println!("Element (1, 1) = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        let mut elem_22 = std::mem::zeroed::<basic_struct>();
        sparse_matrix_get_basic(&raw mut elem_22, sparse, 2, 2);
        let str_ptr = basic_str(&raw const elem_22);
        let c_str = CStr::from_ptr(str_ptr);
        println!("Element (2, 2) = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Get a zero element
        let mut elem_01 = std::mem::zeroed::<basic_struct>();
        sparse_matrix_get_basic(&raw mut elem_01, sparse, 0, 1);
        let str_ptr = basic_str(&raw const elem_01);
        let c_str = CStr::from_ptr(str_ptr);
        println!("Element (0, 1) = {} (should be 0)", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        println!("\n--- Sparse Matrix String Representation ---");

        // Get string representation of the sparse matrix
        let sparse_str_ptr = sparse_matrix_str(sparse);
        if !sparse_str_ptr.is_null() {
            let sparse_c_str = CStr::from_ptr(sparse_str_ptr);
            println!("Sparse matrix:\n{}", sparse_c_str.to_str().unwrap());
            basic_str_free(sparse_str_ptr);
        }

        println!("\n--- Creating Another Sparse Matrix for Comparison ---");

        // Create another sparse matrix for equality testing
        let sparse2 = sparse_matrix_new();
        sparse_matrix_init(sparse2);

        // Set the same elements
        sparse_matrix_set_basic(sparse2, 0, 0, &raw mut x);
        sparse_matrix_set_basic(sparse2, 1, 1, &raw mut y);
        sparse_matrix_set_basic(sparse2, 2, 2, &raw mut two);
        sparse_matrix_set_basic(sparse2, 0, 4, &raw mut one);
        sparse_matrix_set_basic(sparse2, 4, 0, &raw mut one);

        // Test equality (returns 0 for false, non-zero for true)
        let are_equal = sparse_matrix_eq(sparse, sparse2);
        println!(
            "Are the two sparse matrices equal? {}",
            if are_equal != 0 { "Yes" } else { "No" }
        );

        println!("\n--- Creating a Different Sparse Matrix ---");

        let sparse3 = sparse_matrix_new();
        sparse_matrix_init(sparse3);

        // Set different elements
        sparse_matrix_set_basic(sparse3, 0, 0, &raw mut two);
        sparse_matrix_set_basic(sparse3, 1, 1, &raw mut x);

        let are_equal2 = sparse_matrix_eq(sparse, sparse3);
        println!(
            "Are the first and third matrices equal? {}",
            if are_equal2 != 0 { "Yes" } else { "No" }
        );

        println!("\n--- Sparse Matrix Use Cases ---");
        println!("Sparse matrices are ideal for:");
        println!("  • Large matrices with few non-zero elements");
        println!("  • Diagonal or block-diagonal matrices");
        println!("  • Adjacency matrices in graph theory");
        println!("  • Finite element analysis");
        println!("  • Quantum chemistry calculations");
        println!("  • Network analysis");

        println!("\n--- Memory Efficiency ---");
        println!("Sparse matrices only store non-zero elements,");
        println!("making them much more memory-efficient than dense");
        println!("matrices for large, sparse systems.");

        // Clean up
        sparse_matrix_free(sparse);
        sparse_matrix_free(sparse2);
        sparse_matrix_free(sparse3);

        println!("\n✓ All sparse matrix operations completed successfully!");
    }
}
