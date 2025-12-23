//! Basic symbolic arithmetic operations example
//!
//! This example demonstrates how to use the quantrs2-symengine-sys FFI bindings
//! to perform basic symbolic arithmetic operations.

use quantrs2_symengine_sys::*;
use std::ffi::CStr;
use std::os::raw::c_int;

fn main() {
    unsafe {
        println!("SymEngine Basic Arithmetic Example");
        println!("===================================\n");

        // Create symbols x and y
        let mut x = std::mem::zeroed::<basic_struct>();
        let mut y = std::mem::zeroed::<basic_struct>();

        symbol_set(&raw mut x, c"x".as_ptr());
        symbol_set(&raw mut y, c"y".as_ptr());

        println!("Created symbols: x and y");

        // Create integer 2
        let mut two = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut two, 2);

        // Compute x + y
        let mut sum = std::mem::zeroed::<basic_struct>();
        let code = basic_add(&raw mut sum, &raw const x, &raw const y);
        check_result(code as c_int).expect("Failed to add");

        // Print the result
        let str_ptr = basic_str(&raw const sum);
        let c_str = CStr::from_ptr(str_ptr);
        println!("x + y = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Compute x * y
        let mut product = std::mem::zeroed::<basic_struct>();
        let code = basic_mul(&raw mut product, &raw const x, &raw const y);
        check_result(code as c_int).expect("Failed to multiply");

        let str_ptr = basic_str(&raw const product);
        let c_str = CStr::from_ptr(str_ptr);
        println!("x * y = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Compute x^2
        let mut x_squared = std::mem::zeroed::<basic_struct>();
        let code = basic_pow(&raw mut x_squared, &raw const x, &raw const two);
        check_result(code as c_int).expect("Failed to compute power");

        let str_ptr = basic_str(&raw const x_squared);
        let c_str = CStr::from_ptr(str_ptr);
        println!("x^2 = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Compute (x + y)^2
        let mut sum_squared = std::mem::zeroed::<basic_struct>();
        let code = basic_pow(&raw mut sum_squared, &raw const sum, &raw const two);
        check_result(code as c_int).expect("Failed to compute power");

        let str_ptr = basic_str(&raw const sum_squared);
        let c_str = CStr::from_ptr(str_ptr);
        println!("(x + y)^2 = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Expand (x + y)^2
        let mut expanded = std::mem::zeroed::<basic_struct>();
        let code = basic_expand(&raw mut expanded, &raw const sum_squared);
        check_result(code as c_int).expect("Failed to expand");

        let str_ptr = basic_str(&raw const expanded);
        let c_str = CStr::from_ptr(str_ptr);
        println!("expand((x + y)^2) = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Differentiate with respect to x
        let mut derivative = std::mem::zeroed::<basic_struct>();
        let code = basic_diff(&raw mut derivative, &raw const expanded, &raw const x);
        check_result(code as c_int).expect("Failed to differentiate");

        let str_ptr = basic_str(&raw const derivative);
        let c_str = CStr::from_ptr(str_ptr);
        println!("d/dx(expand((x + y)^2)) = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        println!("\n✓ All operations completed successfully!");
    }
}
