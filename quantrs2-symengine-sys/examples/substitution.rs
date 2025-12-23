//! Variable substitution example
//!
//! This example demonstrates how to use CMapBasicBasic for variable substitution
//! in symbolic expressions.

use quantrs2_symengine_sys::*;
use std::ffi::CStr;
use std::os::raw::c_int;

fn main() {
    unsafe {
        println!("SymEngine Substitution Example");
        println!("===============================\n");

        // Create symbols x and y
        let mut x = std::mem::zeroed::<basic_struct>();
        let mut y = std::mem::zeroed::<basic_struct>();

        symbol_set(&raw mut x, c"x".as_ptr());
        symbol_set(&raw mut y, c"y".as_ptr());

        // Create expression: x^2 + 2*x*y + y^2
        let mut two = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut two, 2);

        let mut x_squared = std::mem::zeroed::<basic_struct>();
        basic_pow(&raw mut x_squared, &raw const x, &raw const two);

        let mut y_squared = std::mem::zeroed::<basic_struct>();
        basic_pow(&raw mut y_squared, &raw const y, &raw const two);

        let mut xy = std::mem::zeroed::<basic_struct>();
        basic_mul(&raw mut xy, &raw const x, &raw const y);

        let mut two_xy = std::mem::zeroed::<basic_struct>();
        basic_mul(&raw mut two_xy, &raw const two, &raw const xy);

        let mut temp = std::mem::zeroed::<basic_struct>();
        basic_add(&raw mut temp, &raw const x_squared, &raw const two_xy);

        let mut expr = std::mem::zeroed::<basic_struct>();
        basic_add(&raw mut expr, &raw const temp, &raw const y_squared);

        // Print original expression
        let str_ptr = basic_str(&raw const expr);
        let c_str = CStr::from_ptr(str_ptr);
        println!("Original expression:");
        println!("f(x, y) = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Method 1: Simple substitution using basic_subs2
        // Substitute x = 3
        let mut three = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut three, 3);

        let mut result1 = std::mem::zeroed::<basic_struct>();
        let code = basic_subs2(
            &raw mut result1,
            &raw const expr,
            &raw const x,
            &raw const three,
        );
        check_result(code as c_int).expect("Failed to substitute");

        let str_ptr = basic_str(&raw const result1);
        let c_str = CStr::from_ptr(str_ptr);
        println!("\nAfter substituting x = 3:");
        println!("f(3, y) = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Method 2: Multiple substitutions using CMapBasicBasic
        // Substitute x = 2, y = 5
        let map = mapbasicbasic_new();

        let mut two_val = std::mem::zeroed::<basic_struct>();
        let mut five = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut two_val, 2);
        integer_set_si(&raw mut five, 5);

        // Insert x -> 2 and y -> 5 into the map
        mapbasicbasic_insert(map, &raw const x, &raw const two_val);
        mapbasicbasic_insert(map, &raw const y, &raw const five);

        println!("\nSubstitution map:");
        println!("  x -> 2");
        println!("  y -> 5");

        // Perform substitution
        let mut result2 = std::mem::zeroed::<basic_struct>();
        let code = basic_subs(&raw mut result2, &raw const expr, map);
        check_result(code as c_int).expect("Failed to substitute with map");

        let str_ptr = basic_str(&raw const result2);
        let c_str = CStr::from_ptr(str_ptr);
        println!("\nAfter substituting x = 2, y = 5:");
        println!("f(2, 5) = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Clean up
        mapbasicbasic_free(map);

        // Method 3: Substitution with symbolic expressions
        // Create a new symbol z
        let mut z = std::mem::zeroed::<basic_struct>();
        symbol_set(&raw mut z, c"z".as_ptr());

        // Substitute y = z^2
        let mut z_squared = std::mem::zeroed::<basic_struct>();
        basic_pow(&raw mut z_squared, &raw const z, &raw const two);

        let mut result3 = std::mem::zeroed::<basic_struct>();
        let code = basic_subs2(
            &raw mut result3,
            &raw const expr,
            &raw const y,
            &raw const z_squared,
        );
        check_result(code as c_int).expect("Failed to substitute");

        let str_ptr = basic_str(&raw const result3);
        let c_str = CStr::from_ptr(str_ptr);
        println!("\nAfter substituting y = z^2:");
        println!("f(x, z^2) = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Expand the result
        let mut expanded = std::mem::zeroed::<basic_struct>();
        basic_expand(&raw mut expanded, &raw const result3);

        let str_ptr = basic_str(&raw const expanded);
        let c_str = CStr::from_ptr(str_ptr);
        println!("Expanded: {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        println!("\n✓ All substitution operations completed successfully!");
    }
}
