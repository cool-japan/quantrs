//! Number theory operations example
//!
//! This example demonstrates number theory functions available in SymEngine.

use quantrs2_symengine_sys::*;
use std::ffi::CStr;
use std::os::raw::c_int;

fn main() {
    unsafe {
        println!("SymEngine Number Theory Example");
        println!("================================\n");

        // GCD of 48 and 18
        let mut a = std::mem::zeroed::<basic_struct>();
        let mut b = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut a, 48);
        integer_set_si(&raw mut b, 18);

        let mut gcd_result = std::mem::zeroed::<basic_struct>();
        let code = ntheory_gcd(&raw mut gcd_result, &raw const a, &raw const b);
        check_result(code as c_int).expect("Failed to compute GCD");

        let str_ptr = basic_str(&raw const gcd_result);
        let c_str = CStr::from_ptr(str_ptr);
        println!("GCD(48, 18) = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // LCM of 12 and 15
        let mut c = std::mem::zeroed::<basic_struct>();
        let mut d = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut c, 12);
        integer_set_si(&raw mut d, 15);

        let mut lcm_result = std::mem::zeroed::<basic_struct>();
        let code = ntheory_lcm(&raw mut lcm_result, &raw const c, &raw const d);
        check_result(code as c_int).expect("Failed to compute LCM");

        let str_ptr = basic_str(&raw const lcm_result);
        let c_str = CStr::from_ptr(str_ptr);
        println!("LCM(12, 15) = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Factorial: 10!
        let mut factorial = std::mem::zeroed::<basic_struct>();
        let code = ntheory_factorial(&raw mut factorial, 10);
        check_result(code as c_int).expect("Failed to compute factorial");

        let str_ptr = basic_str(&raw const factorial);
        let c_str = CStr::from_ptr(str_ptr);
        println!("\n10! = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Fibonacci numbers
        println!("\nFibonacci sequence (first 10 numbers):");
        for i in 0..10 {
            let mut fib = std::mem::zeroed::<basic_struct>();
            let code = ntheory_fibonacci(&raw mut fib, i);
            check_result(code as c_int).expect("Failed to compute Fibonacci");

            let str_ptr = basic_str(&raw const fib);
            let c_str = CStr::from_ptr(str_ptr);
            print!("{} ", c_str.to_str().unwrap());
            basic_str_free(str_ptr);
        }
        println!();

        // Lucas numbers
        println!("\nLucas sequence (first 10 numbers):");
        for i in 0..10 {
            let mut lucas = std::mem::zeroed::<basic_struct>();
            let code = ntheory_lucas(&raw mut lucas, i);
            check_result(code as c_int).expect("Failed to compute Lucas");

            let str_ptr = basic_str(&raw const lucas);
            let c_str = CStr::from_ptr(str_ptr);
            print!("{} ", c_str.to_str().unwrap());
            basic_str_free(str_ptr);
        }
        println!();

        // Binomial coefficient: C(10, 3)
        let mut n = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut n, 10);

        let mut binomial = std::mem::zeroed::<basic_struct>();
        let code = ntheory_binomial(&raw mut binomial, &raw const n, 3);
        check_result(code as c_int).expect("Failed to compute binomial");

        let str_ptr = basic_str(&raw const binomial);
        let c_str = CStr::from_ptr(str_ptr);
        println!("\nC(10, 3) = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Modular arithmetic: 17 mod 5
        let mut mod_val = std::mem::zeroed::<basic_struct>();
        let mut val = std::mem::zeroed::<basic_struct>();
        let mut modulus = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut val, 17);
        integer_set_si(&raw mut modulus, 5);

        let code = ntheory_mod(&raw mut mod_val, &raw const val, &raw const modulus);
        check_result(code as c_int).expect("Failed to compute mod");

        let str_ptr = basic_str(&raw const mod_val);
        let c_str = CStr::from_ptr(str_ptr);
        println!("\n17 mod 5 = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Quotient: 17 // 5
        let mut quot = std::mem::zeroed::<basic_struct>();
        let code = ntheory_quotient(&raw mut quot, &raw const val, &raw const modulus);
        check_result(code as c_int).expect("Failed to compute quotient");

        let str_ptr = basic_str(&raw const quot);
        let c_str = CStr::from_ptr(str_ptr);
        println!("17 // 5 = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Modular inverse: inverse of 3 mod 7
        let mut inv_base = std::mem::zeroed::<basic_struct>();
        let mut inv_mod = std::mem::zeroed::<basic_struct>();
        integer_set_si(&raw mut inv_base, 3);
        integer_set_si(&raw mut inv_mod, 7);

        let mut inv_result = std::mem::zeroed::<basic_struct>();
        let code =
            ntheory_mod_inverse(&raw mut inv_result, &raw const inv_base, &raw const inv_mod);
        check_result(code as c_int).expect("Failed to compute modular inverse");

        let str_ptr = basic_str(&raw const inv_result);
        let c_str = CStr::from_ptr(str_ptr);
        println!("\ninverse of 3 mod 7 = {}", c_str.to_str().unwrap());
        basic_str_free(str_ptr);

        // Verify: 3 * inv ≡ 1 (mod 7)
        let mut verify = std::mem::zeroed::<basic_struct>();
        basic_mul(&raw mut verify, &raw const inv_base, &raw const inv_result);

        let mut verify_mod = std::mem::zeroed::<basic_struct>();
        ntheory_mod(&raw mut verify_mod, &raw const verify, &raw const inv_mod);

        let str_ptr = basic_str(&raw const verify_mod);
        let c_str = CStr::from_ptr(str_ptr);
        println!(
            "Verification: 3 * {} mod 7 = {}",
            CStr::from_ptr(basic_str(&raw const inv_result))
                .to_str()
                .unwrap(),
            c_str.to_str().unwrap()
        );
        basic_str_free(str_ptr);

        println!("\n✓ All number theory operations completed successfully!");
    }
}
