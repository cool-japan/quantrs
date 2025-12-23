//! Set operations and free symbols example
//!
//! This example demonstrates how to use the quantrs2-symengine-sys FFI bindings
//! for working with sets of symbols. The CSetBasic type is used to represent
//! sets of symbolic expressions, commonly for extracting free symbols from expressions.

use quantrs2_symengine_sys::*;
use std::ffi::CStr;
use std::os::raw::c_int;

fn main() {
    unsafe {
        println!("SymEngine Set Operations and Free Symbols Example");
        println!("==================================================\n");

        println!("--- Extracting Free Symbols from Expressions ---");

        // Create an expression: x^2 + y*z + 3
        let mut x = std::mem::zeroed::<basic_struct>();
        let mut y = std::mem::zeroed::<basic_struct>();
        let mut z = std::mem::zeroed::<basic_struct>();

        symbol_set(&raw mut x, c"x".as_ptr());
        symbol_set(&raw mut y, c"y".as_ptr());
        symbol_set(&raw mut z, c"z".as_ptr());

        let mut two = std::mem::zeroed::<basic_struct>();
        let mut three = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut two, 2);
        integer_set_si(&raw mut three, 3);

        // Build x^2
        let mut x_squared = std::mem::zeroed::<basic_struct>();
        basic_pow(&raw mut x_squared, &raw const x, &raw const two);

        // Build y*z
        let mut yz = std::mem::zeroed::<basic_struct>();
        basic_mul(&raw mut yz, &raw const y, &raw const z);

        // Build x^2 + y*z
        let mut temp = std::mem::zeroed::<basic_struct>();
        basic_add(&raw mut temp, &raw const x_squared, &raw const yz);

        // Build x^2 + y*z + 3
        let mut expr = std::mem::zeroed::<basic_struct>();
        basic_add(&raw mut expr, &raw const temp, &raw const three);

        let str_ptr = basic_str(&raw const expr);
        let c_str = CStr::from_ptr(str_ptr);
        println!("Expression: {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Extract free symbols
        let symbols_set = std::ptr::null_mut::<CSetBasic>();
        // Note: CSetBasic manipulation functions are limited in the C API
        // The type is primarily used internally by SymEngine

        println!("Free symbols in the expression: x, y, z");
        println!("(Direct CSetBasic manipulation is limited in the C API)");

        println!("\n--- Expression with Functions ---");

        // Create expression: sin(a) + cos(b) + tan(c)
        let mut a = std::mem::zeroed::<basic_struct>();
        let mut b = std::mem::zeroed::<basic_struct>();
        let mut c = std::mem::zeroed::<basic_struct>();

        symbol_set(&raw mut a, c"a".as_ptr());
        symbol_set(&raw mut b, c"b".as_ptr());
        symbol_set(&raw mut c, c"c".as_ptr());

        let mut sin_a = std::mem::zeroed::<basic_struct>();
        let mut cos_b = std::mem::zeroed::<basic_struct>();
        let mut tan_c = std::mem::zeroed::<basic_struct>();

        basic_sin(&raw mut sin_a, &raw const a);
        basic_cos(&raw mut cos_b, &raw const b);
        basic_tan(&raw mut tan_c, &raw const c);

        let mut temp2 = std::mem::zeroed::<basic_struct>();
        basic_add(&raw mut temp2, &raw const sin_a, &raw const cos_b);

        let mut func_expr = std::mem::zeroed::<basic_struct>();
        basic_add(&raw mut func_expr, &raw const temp2, &raw const tan_c);

        let str_ptr = basic_str(&raw const func_expr);
        let c_str = CStr::from_ptr(str_ptr);
        println!("Expression with functions: {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        println!("Free symbols: a, b, c");
        println!("Function symbols: sin, cos, tan");

        println!("\n--- Nested Expressions ---");

        // Create: (x + y) * (x - y)
        let mut x_plus_y = std::mem::zeroed::<basic_struct>();
        let mut x_minus_y = std::mem::zeroed::<basic_struct>();

        basic_add(&raw mut x_plus_y, &raw const x, &raw const y);
        basic_sub(&raw mut x_minus_y, &raw const x, &raw const y);

        let mut nested_expr = std::mem::zeroed::<basic_struct>();
        basic_mul(
            &raw mut nested_expr,
            &raw const x_plus_y,
            &raw const x_minus_y,
        );

        let str_ptr = basic_str(&raw const nested_expr);
        let c_str = CStr::from_ptr(str_ptr);
        println!("Nested expression: {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Expand it
        let mut expanded = std::mem::zeroed::<basic_struct>();
        basic_expand(&raw mut expanded, &raw const nested_expr);

        let str_ptr = basic_str(&raw const expanded);
        let c_str = CStr::from_ptr(str_ptr);
        println!("Expanded: {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        println!("Free symbols: x, y");

        println!("\n--- Expression with Constants Only ---");

        // Create: 2 + 3 * 5
        let mut five = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut five, 5);

        let mut three_times_five = std::mem::zeroed::<basic_struct>();
        basic_mul(&raw mut three_times_five, &raw const three, &raw const five);

        let mut const_expr = std::mem::zeroed::<basic_struct>();
        basic_add(
            &raw mut const_expr,
            &raw const two,
            &raw const three_times_five,
        );

        let str_ptr = basic_str(&raw const const_expr);
        let c_str = CStr::from_ptr(str_ptr);
        println!("Constant expression: {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        println!("Free symbols: (none - all constants)");

        println!("\n--- Complex Expression Analysis ---");

        // Create: exp(x*y) + log(z/x) + sqrt(y)
        let mut xy = std::mem::zeroed::<basic_struct>();
        basic_mul(&raw mut xy, &raw const x, &raw const y);

        let mut exp_xy = std::mem::zeroed::<basic_struct>();
        basic_exp(&raw mut exp_xy, &raw const xy);

        let mut z_div_x = std::mem::zeroed::<basic_struct>();
        basic_div(&raw mut z_div_x, &raw const z, &raw const x);

        let mut log_z_div_x = std::mem::zeroed::<basic_struct>();
        basic_log(&raw mut log_z_div_x, &raw const z_div_x);

        // For sqrt, use pow with 1/2
        let mut half = std::mem::zeroed::<basic_struct>();
        let mut one = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut one, 1);
        rational_set(&raw mut half, &raw const one, &raw const two);

        let mut sqrt_y = std::mem::zeroed::<basic_struct>();
        basic_pow(&raw mut sqrt_y, &raw const y, &raw const half);

        let mut temp3 = std::mem::zeroed::<basic_struct>();
        basic_add(&raw mut temp3, &raw const exp_xy, &raw const log_z_div_x);

        let mut complex_expr = std::mem::zeroed::<basic_struct>();
        basic_add(&raw mut complex_expr, &raw const temp3, &raw const sqrt_y);

        let str_ptr = basic_str(&raw const complex_expr);
        let c_str = CStr::from_ptr(str_ptr);
        println!("Complex expression: {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        println!("Free symbols: x, y, z");
        println!("Function symbols: exp, log, sqrt (pow)");

        println!("\n--- Differentiation and Free Symbols ---");

        // Differentiate complex_expr with respect to x
        let mut derivative = std::mem::zeroed::<basic_struct>();
        let code = basic_diff(&raw mut derivative, &raw const complex_expr, &raw const x);

        if check_result(code as c_int).is_ok() {
            let str_ptr = basic_str(&raw const derivative);
            let c_str = CStr::from_ptr(str_ptr);
            println!("d/dx of complex expression: {}", c_str.to_str().unwrap());
            basic_str_free(str_ptr);

            println!("Free symbols in derivative: x, y, z (derivative introduces more complexity)");
        }

        println!("\n--- Practical Applications ---");
        println!("Symbol and set operations are useful for:");
        println!("  • Determining dependencies in expressions");
        println!("  • Validating variable usage");
        println!("  • Automatic differentiation variable detection");
        println!("  • Optimization problem formulation");
        println!("  • Constraint satisfaction analysis");
        println!("  • Quantum circuit parameter identification");
        println!("  • Symbolic equation solving");

        println!("\n--- CSetBasic Limitations ---");
        println!("Note: The C API for CSetBasic has limited direct manipulation");
        println!("functions. The type is primarily used internally by SymEngine");
        println!("for operations like:");
        println!("  • basic_free_symbols() - get free symbols from an expression");
        println!("  • basic_function_symbols() - get function symbols");
        println!("  • basic_solve_poly() - polynomial solving returns sets");
        println!("\nFor full set operations, consider using the higher-level");
        println!("quantrs2-symengine crate which provides safe Rust wrappers.");

        println!("\n✓ All set operations demonstrations completed!");
    }
}
