//! Expression parsing example
//!
//! This example demonstrates how to use the quantrs2-symengine-sys FFI bindings
//! for parsing mathematical expressions from strings.

use quantrs2_symengine_sys::*;
use std::ffi::{CStr, CString};
use std::os::raw::c_int;

fn main() {
    unsafe {
        println!("SymEngine Expression Parsing Example");
        println!("=====================================\n");

        println!("--- Basic Expression Parsing ---");

        // Parse a simple expression: x + y
        let mut expr1 = std::mem::zeroed::<basic_struct>();
        let expr_str1 = CString::new("x + y").unwrap();
        let code = basic_parse(&raw mut expr1, expr_str1.as_ptr());

        if check_result(code as c_int).is_ok() {
            let str_ptr = basic_str(&raw const expr1);
            let c_str = CStr::from_ptr(str_ptr);
            println!("Parsed 'x + y' = {}", c_str.to_str().unwrap());
            basic_str_free(str_ptr);
        } else {
            println!("Failed to parse 'x + y'");
        }

        println!("\n--- Parsing Complex Expressions ---");

        // Parse a more complex expression: (x^2 + 2*x + 1)
        let mut expr2 = std::mem::zeroed::<basic_struct>();
        let expr_str2 = CString::new("x**2 + 2*x + 1").unwrap();
        let code = basic_parse(&raw mut expr2, expr_str2.as_ptr());

        if check_result(code as c_int).is_ok() {
            let str_ptr = basic_str(&raw const expr2);
            let c_str = CStr::from_ptr(str_ptr);
            println!("Parsed 'x**2 + 2*x + 1' = {}", c_str.to_str().unwrap());
            basic_str_free(str_ptr);
        } else {
            println!("Failed to parse expression");
        }

        println!("\n--- Parsing Trigonometric Functions ---");

        // Parse trigonometric expression: sin(x) + cos(y)
        let mut expr3 = std::mem::zeroed::<basic_struct>();
        let expr_str3 = CString::new("sin(x) + cos(y)").unwrap();
        let code = basic_parse(&raw mut expr3, expr_str3.as_ptr());

        if check_result(code as c_int).is_ok() {
            let str_ptr = basic_str(&raw const expr3);
            let c_str = CStr::from_ptr(str_ptr);
            println!("Parsed 'sin(x) + cos(y)' = {}", c_str.to_str().unwrap());
            basic_str_free(str_ptr);
        } else {
            println!("Failed to parse trigonometric expression");
        }

        println!("\n--- Parsing Exponential and Logarithmic Functions ---");

        // Parse: exp(x) * log(y)
        let mut expr4 = std::mem::zeroed::<basic_struct>();
        let expr_str4 = CString::new("exp(x) * log(y)").unwrap();
        let code = basic_parse(&raw mut expr4, expr_str4.as_ptr());

        if check_result(code as c_int).is_ok() {
            let str_ptr = basic_str(&raw const expr4);
            let c_str = CStr::from_ptr(str_ptr);
            println!("Parsed 'exp(x) * log(y)' = {}", c_str.to_str().unwrap());
            basic_str_free(str_ptr);
        } else {
            println!("Failed to parse exp/log expression");
        }

        println!("\n--- Parsing Fractions ---");

        // Parse: (x + 1) / (x - 1)
        let mut expr5 = std::mem::zeroed::<basic_struct>();
        let expr_str5 = CString::new("(x + 1) / (x - 1)").unwrap();
        let code = basic_parse(&raw mut expr5, expr_str5.as_ptr());

        if check_result(code as c_int).is_ok() {
            let str_ptr = basic_str(&raw const expr5);
            let c_str = CStr::from_ptr(str_ptr);
            println!("Parsed '(x + 1) / (x - 1)' = {}", c_str.to_str().unwrap());
            basic_str_free(str_ptr);
        } else {
            println!("Failed to parse fraction");
        }

        println!("\n--- Parsing and Expanding ---");

        // Parse and then expand: (x + 1)^2
        let mut expr6 = std::mem::zeroed::<basic_struct>();
        let expr_str6 = CString::new("(x + 1)**2").unwrap();
        let code = basic_parse(&raw mut expr6, expr_str6.as_ptr());

        if check_result(code as c_int).is_ok() {
            let str_ptr = basic_str(&raw const expr6);
            let c_str = CStr::from_ptr(str_ptr);
            println!("Parsed '(x + 1)**2' = {}", c_str.to_str().unwrap());
            basic_str_free(str_ptr);

            // Expand the expression
            let mut expanded = std::mem::zeroed::<basic_struct>();
            basic_expand(&raw mut expanded, &raw const expr6);

            let str_ptr = basic_str(&raw const expanded);
            let c_str = CStr::from_ptr(str_ptr);
            println!("Expanded = {}", c_str.to_str().unwrap());
            basic_str_free(str_ptr);
        } else {
            println!("Failed to parse expression");
        }

        println!("\n--- Parsing and Differentiation ---");

        // Parse and differentiate: x^3 + 2*x^2 + x
        let mut expr7 = std::mem::zeroed::<basic_struct>();
        let expr_str7 = CString::new("x**3 + 2*x**2 + x").unwrap();
        let code = basic_parse(&raw mut expr7, expr_str7.as_ptr());

        if check_result(code as c_int).is_ok() {
            let str_ptr = basic_str(&raw const expr7);
            let c_str = CStr::from_ptr(str_ptr);
            println!("Parsed 'x**3 + 2*x**2 + x' = {}", c_str.to_str().unwrap());
            basic_str_free(str_ptr);

            // Differentiate with respect to x
            let mut x = std::mem::zeroed::<basic_struct>();
            symbol_set(&raw mut x, c"x".as_ptr());

            let mut derivative = std::mem::zeroed::<basic_struct>();
            basic_diff(&raw mut derivative, &raw const expr7, &raw const x);

            let str_ptr = basic_str(&raw const derivative);
            let c_str = CStr::from_ptr(str_ptr);
            println!("d/dx = {}", c_str.to_str().unwrap());
            basic_str_free(str_ptr);
        } else {
            println!("Failed to parse expression");
        }

        println!("\n--- Parsing with Multiple Variables ---");

        // Parse: x*y + y*z + z*x
        let mut expr8 = std::mem::zeroed::<basic_struct>();
        let expr_str8 = CString::new("x*y + y*z + z*x").unwrap();
        let code = basic_parse(&raw mut expr8, expr_str8.as_ptr());

        if check_result(code as c_int).is_ok() {
            let str_ptr = basic_str(&raw const expr8);
            let c_str = CStr::from_ptr(str_ptr);
            println!("Parsed 'x*y + y*z + z*x' = {}", c_str.to_str().unwrap());
            basic_str_free(str_ptr);
        } else {
            println!("Failed to parse multi-variable expression");
        }

        println!("\n--- Parsing Numeric Constants ---");

        // Parse: 3.14159 * x + 2.71828
        let mut expr9 = std::mem::zeroed::<basic_struct>();
        let expr_str9 = CString::new("3.14159 * x + 2.71828").unwrap();
        let code = basic_parse(&raw mut expr9, expr_str9.as_ptr());

        if check_result(code as c_int).is_ok() {
            let str_ptr = basic_str(&raw const expr9);
            let c_str = CStr::from_ptr(str_ptr);
            println!(
                "Parsed '3.14159 * x + 2.71828' = {}",
                c_str.to_str().unwrap()
            );
            basic_str_free(str_ptr);
        } else {
            println!("Failed to parse numeric expression");
        }

        println!("\n--- Error Handling: Invalid Expression ---");

        // Try to parse an invalid expression
        let mut expr_invalid = std::mem::zeroed::<basic_struct>();
        let expr_str_invalid = CString::new("x + + y").unwrap(); // Invalid syntax
        let code = basic_parse(&raw mut expr_invalid, expr_str_invalid.as_ptr());

        match check_result(code as c_int) {
            Ok(()) => {
                let str_ptr = basic_str(&raw const expr_invalid);
                let c_str = CStr::from_ptr(str_ptr);
                println!(
                    "Unexpectedly parsed 'x + + y' = {}",
                    c_str.to_str().unwrap()
                );
                basic_str_free(str_ptr);
            }
            Err(e) => {
                println!("Expected error parsing 'x + + y': {e:?}");
            }
        }

        println!("\n--- Parsing Nested Functions ---");

        // Parse: sin(cos(x))
        let mut expr10 = std::mem::zeroed::<basic_struct>();
        let expr_str10 = CString::new("sin(cos(x))").unwrap();
        let code = basic_parse(&raw mut expr10, expr_str10.as_ptr());

        if check_result(code as c_int).is_ok() {
            let str_ptr = basic_str(&raw const expr10);
            let c_str = CStr::from_ptr(str_ptr);
            println!("Parsed 'sin(cos(x))' = {}", c_str.to_str().unwrap());
            basic_str_free(str_ptr);
        } else {
            println!("Failed to parse nested functions");
        }

        println!("\n--- Practical Applications ---");
        println!("Expression parsing is useful for:");
        println!("  • User input in symbolic calculators");
        println!("  • Configuration files with formulas");
        println!("  • Dynamic equation evaluation");
        println!("  • Mathematical software interfaces");
        println!("  • Data analysis with custom formulas");
        println!("  • Quantum circuit parameter expressions");

        println!("\n✓ All parsing operations completed!");
    }
}
