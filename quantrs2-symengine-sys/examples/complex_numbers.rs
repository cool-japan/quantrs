//! Complex number operations example
//!
//! This example demonstrates how to use the quantrs2-symengine-sys FFI bindings
//! to perform complex number operations including creation, arithmetic,
//! real/imaginary part extraction, and conjugation.

use quantrs2_symengine_sys::*;
use std::ffi::CStr;
use std::os::raw::c_int;

fn main() {
    unsafe {
        println!("SymEngine Complex Number Operations Example");
        println!("============================================\n");

        // Create complex numbers using complex_set with real_double parts
        // First create real and imaginary parts as real_double
        let mut re1 = std::mem::zeroed::<basic_struct>();
        let mut im1 = std::mem::zeroed::<basic_struct>();
        real_double_set_d(&raw mut re1, 3.0);
        real_double_set_d(&raw mut im1, 4.0);

        // Complex number: 3 + 4i
        let mut z1 = std::mem::zeroed::<basic_struct>();
        complex_set(&raw mut z1, &raw const re1, &raw const im1);

        let str_ptr = basic_str(&raw const z1);
        let c_str = CStr::from_ptr(str_ptr);
        println!("z1 = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Complex number: 1 + 2i
        let mut re2 = std::mem::zeroed::<basic_struct>();
        let mut im2 = std::mem::zeroed::<basic_struct>();
        real_double_set_d(&raw mut re2, 1.0);
        real_double_set_d(&raw mut im2, 2.0);

        let mut z2 = std::mem::zeroed::<basic_struct>();
        complex_set(&raw mut z2, &raw const re2, &raw const im2);

        let str_ptr = basic_str(&raw const z2);
        let c_str = CStr::from_ptr(str_ptr);
        println!("z2 = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Complex addition: z1 + z2
        println!("\n--- Complex Addition ---");
        let mut sum = std::mem::zeroed::<basic_struct>();
        let code = basic_add(&raw mut sum, &raw const z1, &raw const z2);
        check_result(code as c_int).expect("Failed to add complex numbers");

        let str_ptr = basic_str(&raw const sum);
        let c_str = CStr::from_ptr(str_ptr);
        println!("z1 + z2 = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Complex multiplication: z1 * z2
        println!("\n--- Complex Multiplication ---");
        let mut product = std::mem::zeroed::<basic_struct>();
        let code = basic_mul(&raw mut product, &raw const z1, &raw const z2);
        check_result(code as c_int).expect("Failed to multiply complex numbers");

        let str_ptr = basic_str(&raw const product);
        let c_str = CStr::from_ptr(str_ptr);
        println!("z1 * z2 = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Complex division: z1 / z2
        println!("\n--- Complex Division ---");
        let mut quotient = std::mem::zeroed::<basic_struct>();
        let code = basic_div(&raw mut quotient, &raw const z1, &raw const z2);
        check_result(code as c_int).expect("Failed to divide complex numbers");

        let str_ptr = basic_str(&raw const quotient);
        let c_str = CStr::from_ptr(str_ptr);
        println!("z1 / z2 = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Extract real and imaginary parts
        println!("\n--- Extract Real and Imaginary Parts ---");
        let mut real_part = std::mem::zeroed::<basic_struct>();
        let mut imag_part = std::mem::zeroed::<basic_struct>();

        complex_base_real_part(&raw mut real_part, &raw const z1);
        complex_base_imaginary_part(&raw mut imag_part, &raw const z1);

        let str_ptr = basic_str(&raw const real_part);
        let c_str = CStr::from_ptr(str_ptr);
        println!("Re(z1) = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        let str_ptr = basic_str(&raw const imag_part);
        let c_str = CStr::from_ptr(str_ptr);
        println!("Im(z1) = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Compute conjugate manually (conjugate(a + bi) = a - bi)
        println!("\n--- Complex Conjugate ---");

        // Get real and imaginary parts
        let mut real_z1 = std::mem::zeroed::<basic_struct>();
        let mut imag_z1 = std::mem::zeroed::<basic_struct>();
        complex_base_real_part(&raw mut real_z1, &raw const z1);
        complex_base_imaginary_part(&raw mut imag_z1, &raw const z1);

        // Negate imaginary part
        let mut neg_imag = std::mem::zeroed::<basic_struct>();
        basic_neg(&raw mut neg_imag, &raw const imag_z1);

        // Create conjugate
        let mut conj_z1 = std::mem::zeroed::<basic_struct>();
        complex_set(&raw mut conj_z1, &raw const real_z1, &raw const neg_imag);

        let str_ptr = basic_str(&raw const conj_z1);
        let c_str = CStr::from_ptr(str_ptr);
        println!("conj(z1) = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Verify conjugate property: z * conj(z) = |z|^2 (real number)
        println!("\n--- Verify Conjugate Property: z * conj(z) = |z|^2 ---");
        let mut z_times_conj = std::mem::zeroed::<basic_struct>();
        let code = basic_mul(&raw mut z_times_conj, &raw const z1, &raw const conj_z1);
        check_result(code as c_int).expect("Failed to multiply z1 * conj(z1)");

        let str_ptr = basic_str(&raw const z_times_conj);
        let c_str = CStr::from_ptr(str_ptr);
        println!("z1 * conj(z1) = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Complex power: z1^2
        println!("\n--- Complex Power ---");
        let mut two = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut two, 2);

        let mut z1_squared = std::mem::zeroed::<basic_struct>();
        let code = basic_pow(&raw mut z1_squared, &raw const z1, &raw const two);
        check_result(code as c_int).expect("Failed to compute z1^2");

        let str_ptr = basic_str(&raw const z1_squared);
        let c_str = CStr::from_ptr(str_ptr);
        println!("z1^2 = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Create complex number with symbolic parts
        println!("\n--- Complex Numbers with Symbolic Parts ---");
        let mut x = std::mem::zeroed::<basic_struct>();
        let mut y = std::mem::zeroed::<basic_struct>();
        symbol_set(&raw mut x, c"x".as_ptr());
        symbol_set(&raw mut y, c"y".as_ptr());

        // Create imaginary unit i (0 + 1i)
        let mut zero_re = std::mem::zeroed::<basic_struct>();
        let mut one_im = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut zero_re, 0);
        integer_set_si(&raw mut one_im, 1);

        let mut i = std::mem::zeroed::<basic_struct>();
        complex_set(&raw mut i, &raw const zero_re, &raw const one_im);

        // Create x + yi
        let mut yi = std::mem::zeroed::<basic_struct>();
        basic_mul(&raw mut yi, &raw const y, &raw const i);

        let mut symbolic_complex = std::mem::zeroed::<basic_struct>();
        basic_add(&raw mut symbolic_complex, &raw const x, &raw const yi);

        let str_ptr = basic_str(&raw const symbolic_complex);
        let c_str = CStr::from_ptr(str_ptr);
        println!("Symbolic complex: {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Conjugate of symbolic complex (manually: x - yi)
        let mut neg_yi = std::mem::zeroed::<basic_struct>();
        basic_neg(&raw mut neg_yi, &raw const yi);

        let mut conj_symbolic = std::mem::zeroed::<basic_struct>();
        basic_add(&raw mut conj_symbolic, &raw const x, &raw const neg_yi);

        let str_ptr = basic_str(&raw const conj_symbolic);
        let c_str = CStr::from_ptr(str_ptr);
        println!("conj(x + yi) = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Square of symbolic complex number: (x + yi)^2
        let mut symbolic_squared = std::mem::zeroed::<basic_struct>();
        let code = basic_pow(
            &raw mut symbolic_squared,
            &raw const symbolic_complex,
            &raw const two,
        );
        check_result(code as c_int).expect("Failed to compute (x + yi)^2");

        let str_ptr = basic_str(&raw const symbolic_squared);
        let c_str = CStr::from_ptr(str_ptr);
        println!("(x + yi)^2 = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Expand the result
        let mut expanded = std::mem::zeroed::<basic_struct>();
        basic_expand(&raw mut expanded, &raw const symbolic_squared);

        let str_ptr = basic_str(&raw const expanded);
        let c_str = CStr::from_ptr(str_ptr);
        println!("Expanded: {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Complex exponential: e^(i*pi) = -1 (Euler's formula)
        println!("\n--- Euler's Formula: e^(i*pi) = -1 ---");
        let mut pi = std::mem::zeroed::<basic_struct>();
        symbol_set(&raw mut pi, c"pi".as_ptr());

        let mut i_pi = std::mem::zeroed::<basic_struct>();
        basic_mul(&raw mut i_pi, &raw const i, &raw const pi);

        let mut e_to_i_pi = std::mem::zeroed::<basic_struct>();
        basic_exp(&raw mut e_to_i_pi, &raw const i_pi);

        let str_ptr = basic_str(&raw const e_to_i_pi);
        let c_str = CStr::from_ptr(str_ptr);
        println!("e^(i*pi) = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        println!("\n✓ All complex number operations completed successfully!");
    }
}
