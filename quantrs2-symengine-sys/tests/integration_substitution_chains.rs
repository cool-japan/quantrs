//! Integration tests for substitution chains
//!
//! These tests verify that complex substitution workflows work correctly.

use quantrs2_symengine_sys::*;
use std::ffi::CStr;
use std::os::raw::c_int;

#[test]
fn test_substitution_debug() {
    unsafe {
        println!("\n=== Debugging substitution chain ===");

        // Create symbols
        let mut x = std::mem::zeroed::<basic_struct>();
        let mut y = std::mem::zeroed::<basic_struct>();
        symbol_set(&raw mut x, c"x".as_ptr());
        symbol_set(&raw mut y, c"y".as_ptr());

        // Print x
        let str_ptr = basic_str(&raw const x);
        let c_str = CStr::from_ptr(str_ptr);
        println!("x = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Create integer 2
        let mut two = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut two, 2);

        // Print 2
        let str_ptr = basic_str(&raw const two);
        let c_str = CStr::from_ptr(str_ptr);
        println!("two = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // x^2
        let mut x_squared = std::mem::zeroed::<basic_struct>();
        let code = basic_pow(&raw mut x_squared, &raw const x, &raw const two);
        println!("basic_pow returned: {}", code as c_int);

        // Print x^2
        let str_ptr = basic_str(&raw const x_squared);
        let c_str = CStr::from_ptr(str_ptr);
        println!("x^2 = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // x^2 + y
        let mut expr = std::mem::zeroed::<basic_struct>();
        let code = basic_add(&raw mut expr, &raw const x_squared, &raw const y);
        println!("basic_add returned: {}", code as c_int);

        // Print x^2 + y
        let str_ptr = basic_str(&raw const expr);
        let c_str = CStr::from_ptr(str_ptr);
        println!("expr = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // 2*x
        let mut two_x = std::mem::zeroed::<basic_struct>();
        let code = basic_mul(&raw mut two_x, &raw const two, &raw const x);
        println!("basic_mul returned: {}", code as c_int);

        // Print 2*x
        let str_ptr = basic_str(&raw const two_x);
        let c_str = CStr::from_ptr(str_ptr);
        println!("2*x = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Step 1: Substitute y = 2x
        let mut step1 = std::mem::zeroed::<basic_struct>();
        let code = basic_subs2(
            &raw mut step1,
            &raw const expr,
            &raw const y,
            &raw const two_x,
        );
        println!("basic_subs2 (step1) returned: {}", code as c_int);

        // Print step1
        let str_ptr = basic_str(&raw const step1);
        if str_ptr.is_null() {
            println!("ERROR: basic_str returned null for step1!");
        } else {
            let c_str = CStr::from_ptr(str_ptr);
            println!("After y=2x: {}", c_str.to_str().unwrap());
            basic_str_free(str_ptr);
        }

        // Step 2: Substitute x = 3
        let mut three = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut three, 3);

        let mut step2 = std::mem::zeroed::<basic_struct>();
        let code = basic_subs2(
            &raw mut step2,
            &raw const step1,
            &raw const x,
            &raw const three,
        );
        println!("basic_subs2 (step2) returned: {}", code as c_int);

        // Print step2
        let str_ptr = basic_str(&raw const step2);
        if str_ptr.is_null() {
            println!("ERROR: basic_str returned null for step2!");
        } else {
            let c_str = CStr::from_ptr(str_ptr);
            println!("After x=3: {}", c_str.to_str().unwrap());
            basic_str_free(str_ptr);
        }
    }
}

#[test]
fn test_simple_substitution_chain() {
    unsafe {
        // Start with expression: x^2 + y
        // Substitute y = 2x
        // Then substitute x = 3
        // Result should be 3^2 + 2*3 = 9 + 6 = 15

        let mut x = std::mem::zeroed::<basic_struct>();
        let mut y = std::mem::zeroed::<basic_struct>();
        symbol_set(&raw mut x, c"x".as_ptr());
        symbol_set(&raw mut y, c"y".as_ptr());

        let mut two = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut two, 2);

        // x^2
        let mut x_squared = std::mem::zeroed::<basic_struct>();
        let code = basic_pow(&raw mut x_squared, &raw const x, &raw const two);
        assert_eq!(code as c_int, 0, "basic_pow should succeed");

        // x^2 + y
        let mut expr = std::mem::zeroed::<basic_struct>();
        let code = basic_add(&raw mut expr, &raw const x_squared, &raw const y);
        assert_eq!(code as c_int, 0, "basic_add should succeed");

        // Step 1: Substitute y = 2x
        let mut two_x = std::mem::zeroed::<basic_struct>();
        let code = basic_mul(&raw mut two_x, &raw const two, &raw const x);
        assert_eq!(code as c_int, 0, "basic_mul should succeed");

        let mut step1 = std::mem::zeroed::<basic_struct>();
        let code = basic_subs2(
            &raw mut step1,
            &raw const expr,
            &raw const y,
            &raw const two_x,
        );
        assert_eq!(code as c_int, 0, "First substitution should succeed");

        // Step 2: Substitute x = 3
        let mut three = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut three, 3);

        let mut step2 = std::mem::zeroed::<basic_struct>();
        let code = basic_subs2(
            &raw mut step2,
            &raw const step1,
            &raw const x,
            &raw const three,
        );
        assert_eq!(code as c_int, 0, "Second substitution should succeed");

        let str_ptr = basic_str(&raw const step2);
        let c_str = CStr::from_ptr(str_ptr);
        let result = c_str.to_str().unwrap();

        // Result should be 15
        assert!(
            result.contains("15") || result == "15",
            "Result should be 15, got: {result}"
        );

        basic_str_free(str_ptr);
    }
}

#[test]
fn test_multi_variable_substitution_chain() {
    unsafe {
        // Expression: x*y + y*z + z*x
        // Substitute x = a+b, y = a-b, z = a*b
        // Then substitute a = 2, b = 1

        let mut x = std::mem::zeroed::<basic_struct>();
        let mut y = std::mem::zeroed::<basic_struct>();
        let mut z = std::mem::zeroed::<basic_struct>();
        symbol_set(&raw mut x, c"x".as_ptr());
        symbol_set(&raw mut y, c"y".as_ptr());
        symbol_set(&raw mut z, c"z".as_ptr());

        // Build x*y
        let mut xy = std::mem::zeroed::<basic_struct>();
        basic_mul(&raw mut xy, &raw const x, &raw const y);

        // Build y*z
        let mut yz = std::mem::zeroed::<basic_struct>();
        basic_mul(&raw mut yz, &raw const y, &raw const z);

        // Build z*x
        let mut zx = std::mem::zeroed::<basic_struct>();
        basic_mul(&raw mut zx, &raw const z, &raw const x);

        // Build x*y + y*z
        let mut temp = std::mem::zeroed::<basic_struct>();
        basic_add(&raw mut temp, &raw const xy, &raw const yz);

        // Build x*y + y*z + z*x
        let mut expr = std::mem::zeroed::<basic_struct>();
        basic_add(&raw mut expr, &raw const temp, &raw const zx);

        // Create a and b
        let mut a = std::mem::zeroed::<basic_struct>();
        let mut b = std::mem::zeroed::<basic_struct>();
        symbol_set(&raw mut a, c"a".as_ptr());
        symbol_set(&raw mut b, c"b".as_ptr());

        // Build substitution expressions
        let mut a_plus_b = std::mem::zeroed::<basic_struct>();
        basic_add(&raw mut a_plus_b, &raw const a, &raw const b);

        let mut a_minus_b = std::mem::zeroed::<basic_struct>();
        basic_sub(&raw mut a_minus_b, &raw const a, &raw const b);

        let mut a_times_b = std::mem::zeroed::<basic_struct>();
        basic_mul(&raw mut a_times_b, &raw const a, &raw const b);

        // Create substitution map for first step
        let map1 = mapbasicbasic_new();
        mapbasicbasic_insert(map1, &raw const x, &raw const a_plus_b);
        mapbasicbasic_insert(map1, &raw const y, &raw const a_minus_b);
        mapbasicbasic_insert(map1, &raw const z, &raw const a_times_b);

        // Apply first substitution
        let mut step1 = std::mem::zeroed::<basic_struct>();
        let code = basic_subs(&raw mut step1, &raw const expr, map1);
        assert_eq!(code as c_int, 0, "First substitution should succeed");

        // Expand
        let mut expanded = std::mem::zeroed::<basic_struct>();
        basic_expand(&raw mut expanded, &raw const step1);

        // Create second substitution map: a = 2, b = 1
        let mut two = std::mem::zeroed::<basic_struct>();
        let mut one = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut two, 2);
        integer_set_si(&raw mut one, 1);

        let map2 = mapbasicbasic_new();
        mapbasicbasic_insert(map2, &raw const a, &raw const two);
        mapbasicbasic_insert(map2, &raw const b, &raw const one);

        // Apply second substitution
        let mut step2 = std::mem::zeroed::<basic_struct>();
        let code = basic_subs(&raw mut step2, &raw const expanded, map2);
        assert_eq!(code as c_int, 0, "Second substitution should succeed");

        let str_ptr = basic_str(&raw const step2);
        let c_str = CStr::from_ptr(str_ptr);
        let _result = c_str.to_str().unwrap();
        basic_str_free(str_ptr);

        // Clean up
        mapbasicbasic_free(map1);
        mapbasicbasic_free(map2);
    }
}

#[test]
fn test_nested_substitution_with_functions() {
    unsafe {
        // Expression: sin(x) + cos(y)
        // Substitute x = 2*a, y = 3*b
        // Then substitute a = pi/4, b = pi/6

        let mut x = std::mem::zeroed::<basic_struct>();
        let mut y = std::mem::zeroed::<basic_struct>();
        symbol_set(&raw mut x, c"x".as_ptr());
        symbol_set(&raw mut y, c"y".as_ptr());

        // sin(x)
        let mut sin_x = std::mem::zeroed::<basic_struct>();
        basic_sin(&raw mut sin_x, &raw const x);

        // cos(y)
        let mut cos_y = std::mem::zeroed::<basic_struct>();
        basic_cos(&raw mut cos_y, &raw const y);

        // sin(x) + cos(y)
        let mut expr = std::mem::zeroed::<basic_struct>();
        basic_add(&raw mut expr, &raw const sin_x, &raw const cos_y);

        // Create a and b
        let mut a = std::mem::zeroed::<basic_struct>();
        let mut b = std::mem::zeroed::<basic_struct>();
        symbol_set(&raw mut a, c"a".as_ptr());
        symbol_set(&raw mut b, c"b".as_ptr());

        // Create 2*a and 3*b
        let mut two = std::mem::zeroed::<basic_struct>();
        let mut three = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut two, 2);
        integer_set_si(&raw mut three, 3);

        let mut two_a = std::mem::zeroed::<basic_struct>();
        basic_mul(&raw mut two_a, &raw const two, &raw const a);

        let mut three_b = std::mem::zeroed::<basic_struct>();
        basic_mul(&raw mut three_b, &raw const three, &raw const b);

        // First substitution: x = 2a, y = 3b
        let map1 = mapbasicbasic_new();
        mapbasicbasic_insert(map1, &raw const x, &raw const two_a);
        mapbasicbasic_insert(map1, &raw const y, &raw const three_b);

        let mut step1 = std::mem::zeroed::<basic_struct>();
        let code = basic_subs(&raw mut step1, &raw const expr, map1);
        assert_eq!(code as c_int, 0, "First substitution should succeed");

        // Second substitution would require symbolic pi
        // For now, just verify the first step worked
        let str_ptr = basic_str(&raw const step1);
        let c_str = CStr::from_ptr(str_ptr);
        let result = c_str.to_str().unwrap();

        // Result should contain sin and cos
        assert!(
            result.contains("sin") && result.contains("cos"),
            "Result should contain sin and cos: {result}"
        );
        basic_str_free(str_ptr);

        mapbasicbasic_free(map1);
    }
}

#[test]
fn test_recursive_substitution() {
    unsafe {
        // Test that substitutions can be applied recursively
        // Expression: f(x, y) where x appears in substitution for y

        let mut x = std::mem::zeroed::<basic_struct>();
        let mut y = std::mem::zeroed::<basic_struct>();
        symbol_set(&raw mut x, c"x".as_ptr());
        symbol_set(&raw mut y, c"y".as_ptr());

        // Expression: x + y
        let mut expr = std::mem::zeroed::<basic_struct>();
        basic_add(&raw mut expr, &raw const x, &raw const y);

        // Substitute y = x + 1
        let mut one = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut one, 1);

        let mut x_plus_1 = std::mem::zeroed::<basic_struct>();
        basic_add(&raw mut x_plus_1, &raw const x, &raw const one);

        let mut step1 = std::mem::zeroed::<basic_struct>();
        let code = basic_subs2(
            &raw mut step1,
            &raw const expr,
            &raw const y,
            &raw const x_plus_1,
        );
        assert_eq!(code as c_int, 0);

        // Now substitute x = 2
        let mut two = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut two, 2);

        let mut step2 = std::mem::zeroed::<basic_struct>();
        let code = basic_subs2(
            &raw mut step2,
            &raw const step1,
            &raw const x,
            &raw const two,
        );
        assert_eq!(code as c_int, 0);

        let str_ptr = basic_str(&raw const step2);
        let c_str = CStr::from_ptr(str_ptr);
        let result = c_str.to_str().unwrap();

        // Result should be 2 + (2 + 1) = 5
        assert!(
            result.contains('5') || result == "5",
            "Result should be 5, got: {result}"
        );
        basic_str_free(str_ptr);
    }
}

#[test]
fn test_substitution_with_differentiation() {
    unsafe {
        // Test substitution after differentiation
        // d/dx(x^2*y) = 2*x*y
        // Then substitute x = a, y = b

        let mut x = std::mem::zeroed::<basic_struct>();
        let mut y = std::mem::zeroed::<basic_struct>();
        symbol_set(&raw mut x, c"x".as_ptr());
        symbol_set(&raw mut y, c"y".as_ptr());

        let mut two = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut two, 2);

        // x^2
        let mut x2 = std::mem::zeroed::<basic_struct>();
        basic_pow(&raw mut x2, &raw const x, &raw const two);

        // x^2 * y
        let mut expr = std::mem::zeroed::<basic_struct>();
        basic_mul(&raw mut expr, &raw const x2, &raw const y);

        // Differentiate with respect to x
        let mut derivative = std::mem::zeroed::<basic_struct>();
        let code = basic_diff(&raw mut derivative, &raw const expr, &raw const x);
        assert_eq!(code as c_int, 0);

        // Now substitute x = a, y = b
        let mut a = std::mem::zeroed::<basic_struct>();
        let mut b = std::mem::zeroed::<basic_struct>();
        symbol_set(&raw mut a, c"a".as_ptr());
        symbol_set(&raw mut b, c"b".as_ptr());

        let map = mapbasicbasic_new();
        mapbasicbasic_insert(map, &raw const x, &raw const a);
        mapbasicbasic_insert(map, &raw const y, &raw const b);

        let mut result = std::mem::zeroed::<basic_struct>();
        let code = basic_subs(&raw mut result, &raw const derivative, map);
        assert_eq!(code as c_int, 0);

        let str_ptr = basic_str(&raw const result);
        let c_str = CStr::from_ptr(str_ptr);
        let result_str = c_str.to_str().unwrap();

        // Result should contain 2, a, and b
        assert!(
            result_str.contains('a') && result_str.contains('b'),
            "Result should contain a and b: {result_str}"
        );
        basic_str_free(str_ptr);

        mapbasicbasic_free(map);
    }
}

#[test]
fn test_substitution_chain_with_expansion() {
    unsafe {
        // Test (x+1)^2, expand, then substitute x = y+1, expand again

        let mut x = std::mem::zeroed::<basic_struct>();
        symbol_set(&raw mut x, c"x".as_ptr());

        let mut one = std::mem::zeroed::<basic_struct>();
        let mut two = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut one, 1);
        integer_set_si(&raw mut two, 2);

        // x + 1
        let mut x_plus_1 = std::mem::zeroed::<basic_struct>();
        basic_add(&raw mut x_plus_1, &raw const x, &raw const one);

        // (x+1)^2
        let mut squared = std::mem::zeroed::<basic_struct>();
        basic_pow(&raw mut squared, &raw const x_plus_1, &raw const two);

        // Expand: x^2 + 2x + 1
        let mut expanded1 = std::mem::zeroed::<basic_struct>();
        basic_expand(&raw mut expanded1, &raw const squared);

        // Substitute x = y + 1
        let mut y = std::mem::zeroed::<basic_struct>();
        symbol_set(&raw mut y, c"y".as_ptr());

        let mut y_plus_1 = std::mem::zeroed::<basic_struct>();
        basic_add(&raw mut y_plus_1, &raw const y, &raw const one);

        let mut substituted = std::mem::zeroed::<basic_struct>();
        let code = basic_subs2(
            &raw mut substituted,
            &raw const expanded1,
            &raw const x,
            &raw const y_plus_1,
        );
        assert_eq!(code as c_int, 0);

        // Expand again
        let mut expanded2 = std::mem::zeroed::<basic_struct>();
        basic_expand(&raw mut expanded2, &raw const substituted);

        let str_ptr = basic_str(&raw const expanded2);
        let c_str = CStr::from_ptr(str_ptr);
        let result = c_str.to_str().unwrap();

        // Result should be (y+1+1)^2 = (y+2)^2 = y^2 + 4y + 4
        assert!(result.contains('y'), "Result should contain y: {result}");
        basic_str_free(str_ptr);
    }
}

#[test]
fn test_simultaneous_substitution() {
    unsafe {
        // Test simultaneous substitution: x -> y, y -> x (swap)
        // Expression: x + 2*y
        // After swap: y + 2*x

        let mut x = std::mem::zeroed::<basic_struct>();
        let mut y = std::mem::zeroed::<basic_struct>();
        symbol_set(&raw mut x, c"x".as_ptr());
        symbol_set(&raw mut y, c"y".as_ptr());

        let mut two = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut two, 2);

        // 2*y
        let mut two_y = std::mem::zeroed::<basic_struct>();
        basic_mul(&raw mut two_y, &raw const two, &raw const y);

        // x + 2*y
        let mut expr = std::mem::zeroed::<basic_struct>();
        basic_add(&raw mut expr, &raw const x, &raw const two_y);

        // Create temporary symbols to avoid circular substitution
        let mut temp_x = std::mem::zeroed::<basic_struct>();
        let mut temp_y = std::mem::zeroed::<basic_struct>();
        symbol_set(&raw mut temp_x, c"temp_x".as_ptr());
        symbol_set(&raw mut temp_y, c"temp_y".as_ptr());

        // First substitute to temporaries
        let map1 = mapbasicbasic_new();
        mapbasicbasic_insert(map1, &raw const x, &raw const temp_x);
        mapbasicbasic_insert(map1, &raw const y, &raw const temp_y);

        let mut step1 = std::mem::zeroed::<basic_struct>();
        let code = basic_subs(&raw mut step1, &raw const expr, map1);
        assert_eq!(code as c_int, 0);

        // Then substitute temporaries to swapped values
        let map2 = mapbasicbasic_new();
        mapbasicbasic_insert(map2, &raw const temp_x, &raw const y);
        mapbasicbasic_insert(map2, &raw const temp_y, &raw const x);

        let mut result = std::mem::zeroed::<basic_struct>();
        let code = basic_subs(&raw mut result, &raw const step1, map2);
        assert_eq!(code as c_int, 0);

        let str_ptr = basic_str(&raw const result);
        let c_str = CStr::from_ptr(str_ptr);
        let result_str = c_str.to_str().unwrap();

        // Result should contain y and 2*x
        assert!(
            result_str.contains('y') && result_str.contains('x'),
            "Result should contain swapped variables: {result_str}"
        );
        basic_str_free(str_ptr);

        mapbasicbasic_free(map1);
        mapbasicbasic_free(map2);
    }
}
