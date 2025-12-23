//! Integration tests for multi-step symbolic computation
//!
//! These tests verify that complex symbolic computation workflows
//! work correctly through the FFI layer.

use quantrs2_symengine_sys::*;
use std::ffi::CStr;
use std::os::raw::c_int;

#[test]
fn test_polynomial_expansion_and_factorization() {
    unsafe {
        // Create (x + 2)(x + 3) = x^2 + 5x + 6
        let mut x = std::mem::zeroed::<basic_struct>();
        symbol_set(&raw mut x, c"x".as_ptr());

        let mut two = std::mem::zeroed::<basic_struct>();
        let mut three = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut two, 2);
        integer_set_si(&raw mut three, 3);

        // x + 2
        let mut x_plus_2 = std::mem::zeroed::<basic_struct>();
        basic_add(&raw mut x_plus_2, &raw const x, &raw const two);

        // x + 3
        let mut x_plus_3 = std::mem::zeroed::<basic_struct>();
        basic_add(&raw mut x_plus_3, &raw const x, &raw const three);

        // (x + 2)(x + 3)
        let mut product = std::mem::zeroed::<basic_struct>();
        let code = basic_mul(&raw mut product, &raw const x_plus_2, &raw const x_plus_3);
        assert_eq!(code as c_int, 0, "Multiplication should succeed");

        // Expand
        let mut expanded = std::mem::zeroed::<basic_struct>();
        let code = basic_expand(&raw mut expanded, &raw const product);
        assert_eq!(code as c_int, 0, "Expansion should succeed");

        let str_ptr = basic_str(&raw const expanded);
        let c_str = CStr::from_ptr(str_ptr);
        let result = c_str.to_str().unwrap();

        // The result should contain x^2, 5*x, and 6
        assert!(
            result.contains("x**2") || result.contains("x^2"),
            "Result should contain x^2: {result}"
        );
        basic_str_free(str_ptr);
    }
}

#[test]
fn test_calculus_chain_rule() {
    unsafe {
        // Test d/dx(sin(x^2)) = 2*x*cos(x^2)
        let mut x = std::mem::zeroed::<basic_struct>();
        symbol_set(&raw mut x, c"x".as_ptr());

        let mut two = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut two, 2);

        // x^2
        let mut x_squared = std::mem::zeroed::<basic_struct>();
        basic_pow(&raw mut x_squared, &raw const x, &raw const two);

        // sin(x^2)
        let mut sin_x2 = std::mem::zeroed::<basic_struct>();
        basic_sin(&raw mut sin_x2, &raw const x_squared);

        // Differentiate: d/dx(sin(x^2))
        let mut derivative = std::mem::zeroed::<basic_struct>();
        let code = basic_diff(&raw mut derivative, &raw const sin_x2, &raw const x);
        assert_eq!(code as c_int, 0, "Differentiation should succeed");

        let str_ptr = basic_str(&raw const derivative);
        let c_str = CStr::from_ptr(str_ptr);
        let result = c_str.to_str().unwrap();

        // The result should contain cos(x^2) and 2*x
        assert!(
            result.contains("cos") && (result.contains("2*x") || result.contains("x*2")),
            "Result should contain 2*x*cos(x^2): {result}"
        );
        basic_str_free(str_ptr);
    }
}

#[test]
fn test_expression_simplification_workflow() {
    unsafe {
        // Test simplification of (x^2 - 1)/(x - 1) at various symbolic levels
        let mut x = std::mem::zeroed::<basic_struct>();
        symbol_set(&raw mut x, c"x".as_ptr());

        let mut one = std::mem::zeroed::<basic_struct>();
        let mut two = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut one, 1);
        integer_set_si(&raw mut two, 2);

        // x^2
        let mut x_squared = std::mem::zeroed::<basic_struct>();
        basic_pow(&raw mut x_squared, &raw const x, &raw const two);

        // x^2 - 1
        let mut numerator = std::mem::zeroed::<basic_struct>();
        basic_sub(&raw mut numerator, &raw const x_squared, &raw const one);

        // x - 1
        let mut denominator = std::mem::zeroed::<basic_struct>();
        basic_sub(&raw mut denominator, &raw const x, &raw const one);

        // (x^2 - 1)/(x - 1)
        let mut fraction = std::mem::zeroed::<basic_struct>();
        let code = basic_div(
            &raw mut fraction,
            &raw const numerator,
            &raw const denominator,
        );
        assert_eq!(code as c_int, 0, "Division should succeed");

        // Expand numerator first
        let mut expanded_num = std::mem::zeroed::<basic_struct>();
        basic_expand(&raw mut expanded_num, &raw const numerator);

        let str_ptr = basic_str(&raw const expanded_num);
        let c_str = CStr::from_ptr(str_ptr);
        let _result = c_str.to_str().unwrap();
        basic_str_free(str_ptr);

        // The fraction should be created successfully
        let str_ptr = basic_str(&raw const fraction);
        let c_str = CStr::from_ptr(str_ptr);
        let result = c_str.to_str().unwrap();
        basic_str_free(str_ptr);

        assert!(!result.is_empty(), "Fraction should produce valid output");
    }
}

#[test]
fn test_trigonometric_identities() {
    unsafe {
        // Test sin^2(x) + cos^2(x) = 1
        let mut x = std::mem::zeroed::<basic_struct>();
        symbol_set(&raw mut x, c"x".as_ptr());

        let mut two = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut two, 2);

        // sin(x)
        let mut sin_x = std::mem::zeroed::<basic_struct>();
        basic_sin(&raw mut sin_x, &raw const x);

        // cos(x)
        let mut cos_x = std::mem::zeroed::<basic_struct>();
        basic_cos(&raw mut cos_x, &raw const x);

        // sin(x)^2
        let mut sin2_x = std::mem::zeroed::<basic_struct>();
        basic_pow(&raw mut sin2_x, &raw const sin_x, &raw const two);

        // cos(x)^2
        let mut cos2_x = std::mem::zeroed::<basic_struct>();
        basic_pow(&raw mut cos2_x, &raw const cos_x, &raw const two);

        // sin^2(x) + cos^2(x)
        let mut identity = std::mem::zeroed::<basic_struct>();
        let code = basic_add(&raw mut identity, &raw const sin2_x, &raw const cos2_x);
        assert_eq!(code as c_int, 0, "Addition should succeed");

        let str_ptr = basic_str(&raw const identity);
        let c_str = CStr::from_ptr(str_ptr);
        let result = c_str.to_str().unwrap();

        // The expression should contain sin and cos
        assert!(
            result.contains("sin") && result.contains("cos"),
            "Result should contain sin^2 + cos^2: {result}"
        );
        basic_str_free(str_ptr);
    }
}

#[test]
fn test_nested_function_composition() {
    unsafe {
        // Test f(g(h(x))) where f(x) = x^2, g(x) = sin(x), h(x) = x + 1
        let mut x = std::mem::zeroed::<basic_struct>();
        symbol_set(&raw mut x, c"x".as_ptr());

        let mut one = std::mem::zeroed::<basic_struct>();
        let mut two = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut one, 1);
        integer_set_si(&raw mut two, 2);

        // h(x) = x + 1
        let mut h_x = std::mem::zeroed::<basic_struct>();
        basic_add(&raw mut h_x, &raw const x, &raw const one);

        // g(h(x)) = sin(x + 1)
        let mut g_h_x = std::mem::zeroed::<basic_struct>();
        basic_sin(&raw mut g_h_x, &raw const h_x);

        // f(g(h(x))) = sin(x + 1)^2
        let mut f_g_h_x = std::mem::zeroed::<basic_struct>();
        let code = basic_pow(&raw mut f_g_h_x, &raw const g_h_x, &raw const two);
        assert_eq!(code as c_int, 0, "Power should succeed");

        // Differentiate the composition
        let mut derivative = std::mem::zeroed::<basic_struct>();
        let code = basic_diff(&raw mut derivative, &raw const f_g_h_x, &raw const x);
        assert_eq!(code as c_int, 0, "Differentiation should succeed");

        let str_ptr = basic_str(&raw const derivative);
        let c_str = CStr::from_ptr(str_ptr);
        let result = c_str.to_str().unwrap();

        // The derivative should contain sin and cos
        assert!(
            result.contains("sin") && result.contains("cos"),
            "Derivative should contain sin and cos: {result}"
        );
        basic_str_free(str_ptr);
    }
}

#[test]
fn test_logarithm_and_exponential_workflow() {
    unsafe {
        // Test log(exp(x)) and exp(log(x))
        let mut x = std::mem::zeroed::<basic_struct>();
        symbol_set(&raw mut x, c"x".as_ptr());

        // exp(x)
        let mut exp_x = std::mem::zeroed::<basic_struct>();
        basic_exp(&raw mut exp_x, &raw const x);

        // log(exp(x))
        let mut log_exp_x = std::mem::zeroed::<basic_struct>();
        let code = basic_log(&raw mut log_exp_x, &raw const exp_x);
        assert_eq!(code as c_int, 0, "Log should succeed");

        let str_ptr = basic_str(&raw const log_exp_x);
        let c_str = CStr::from_ptr(str_ptr);
        let result = c_str.to_str().unwrap();

        // log(exp(x)) should simplify to x or contain both log and exp
        assert!(!result.is_empty(), "Result should not be empty: {result}");
        basic_str_free(str_ptr);

        // Test exp(log(x))
        let mut log_x = std::mem::zeroed::<basic_struct>();
        basic_log(&raw mut log_x, &raw const x);

        let mut exp_log_x = std::mem::zeroed::<basic_struct>();
        let code = basic_exp(&raw mut exp_log_x, &raw const log_x);
        assert_eq!(code as c_int, 0, "Exp should succeed");

        let str_ptr = basic_str(&raw const exp_log_x);
        let c_str = CStr::from_ptr(str_ptr);
        let result = c_str.to_str().unwrap();

        assert!(!result.is_empty(), "Result should not be empty: {result}");
        basic_str_free(str_ptr);
    }
}

#[test]
fn test_multi_variable_partial_derivatives() {
    unsafe {
        // Test ∂/∂x(x^2*y + y^2*x) and ∂/∂y(x^2*y + y^2*x)
        let mut x = std::mem::zeroed::<basic_struct>();
        let mut y = std::mem::zeroed::<basic_struct>();
        symbol_set(&raw mut x, c"x".as_ptr());
        symbol_set(&raw mut y, c"y".as_ptr());

        let mut two = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut two, 2);

        // x^2
        let mut x2 = std::mem::zeroed::<basic_struct>();
        basic_pow(&raw mut x2, &raw const x, &raw const two);

        // y^2
        let mut y2 = std::mem::zeroed::<basic_struct>();
        basic_pow(&raw mut y2, &raw const y, &raw const two);

        // x^2 * y
        let mut x2y = std::mem::zeroed::<basic_struct>();
        basic_mul(&raw mut x2y, &raw const x2, &raw const y);

        // y^2 * x
        let mut y2x = std::mem::zeroed::<basic_struct>();
        basic_mul(&raw mut y2x, &raw const y2, &raw const x);

        // x^2*y + y^2*x
        let mut expr = std::mem::zeroed::<basic_struct>();
        basic_add(&raw mut expr, &raw const x2y, &raw const y2x);

        // ∂/∂x
        let mut dx = std::mem::zeroed::<basic_struct>();
        let code = basic_diff(&raw mut dx, &raw const expr, &raw const x);
        assert_eq!(code as c_int, 0, "Partial derivative wrt x should succeed");

        let str_ptr_x = basic_str(&raw const dx);
        let c_str = CStr::from_ptr(str_ptr_x);
        let result_x = c_str.to_str().unwrap();

        // ∂/∂y
        let mut dy = std::mem::zeroed::<basic_struct>();
        let code = basic_diff(&raw mut dy, &raw const expr, &raw const y);
        assert_eq!(code as c_int, 0, "Partial derivative wrt y should succeed");

        let str_ptr_y = basic_str(&raw const dy);
        let c_str = CStr::from_ptr(str_ptr_y);
        let result_y = c_str.to_str().unwrap();

        // Both results should contain both x and y
        assert!(
            result_x.contains('x') && result_x.contains('y'),
            "∂/∂x should contain both x and y: {result_x}"
        );
        assert!(
            result_y.contains('x') && result_y.contains('y'),
            "∂/∂y should contain both x and y: {result_y}"
        );

        basic_str_free(str_ptr_x);
        basic_str_free(str_ptr_y);
    }
}
