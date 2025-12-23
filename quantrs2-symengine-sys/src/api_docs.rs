//! # SymEngine C API Documentation
//!
//! This module provides comprehensive documentation for the auto-generated
//! FFI bindings to SymEngine. All functions here are automatically generated
//! by `bindgen` from the SymEngine C wrapper API (cwrapper.h).
//!
//! ## Safety
//!
//! **ALL FUNCTIONS IN THIS CRATE ARE UNSAFE** and require careful handling:
//!
//! - Memory management is manual - you must free allocated resources
//! - Pointers must be valid and properly initialized
//! - No bounds checking is performed
//! - Thread safety depends on SymEngine's configuration
//! - Error codes must be checked after each operation
//!
//! ## Function Groups
//!
//! The bindings are organized into the following functional groups:
//!
//! ### 1. Basic Operations (`basic_*`)
//!
//! Core symbolic arithmetic and manipulation functions.
//!
//! #### Memory Management
//! - No explicit `basic_new/basic_free` - use `std::mem::zeroed()` for initialization
//! - SymEngine manages internal memory
//!
//! #### Key Functions
//! - `basic_str` - Convert expression to string (must call `basic_str_free` on result!)
//! - `basic_str_free` - Free string returned by `basic_str`
//! - `basic_add`, `basic_sub`, `basic_mul`, `basic_div` - Arithmetic operations
//! - `basic_pow` - Power operation
//! - `basic_neg` - Negation
//! - `basic_expand` - Expand expression
//! - `basic_diff` - Differentiation
//! - `basic_subs`, `basic_subs2` - Variable substitution
//! - `basic_parse` - Parse string to expression
//! - `basic_get_type` - Get type code of expression
//!
//! #### Example
//! ```no_run
//! use quantrs2_symengine_sys::*;
//! unsafe {
//!     let mut x = std::mem::zeroed::<basic_struct>();
//!     symbol_set(&raw mut x, c"x".as_ptr());
//!
//!     let str_ptr = basic_str(&raw const x);
//!     // ... use string ...
//!     basic_str_free(str_ptr); // MUST free!
//! }
//! ```
//!
//! ### 2. Symbol Operations (`symbol_*`)
//!
//! Create and manipulate symbolic variables.
//!
//! #### Key Functions
//! - `symbol_set` - Create a symbol with a given name
//!
//! #### Safety Notes
//! - Name must be a valid null-terminated C string
//! - Symbol names should be unique within your context
//!
//! ### 3. Numeric Types
//!
//! #### Integer Operations (`integer_*`)
//! - `integer_set_si` - Set from signed integer
//! - `integer_set_ui` - Set from unsigned integer
//! - `integer_set_str` - Set from string
//! - `integer_get_si` - Get as signed integer
//!
//! #### Rational Operations (`rational_*`)
//! - `rational_set` - Set from numerator and denominator
//! - `rational_set_si` - Set from signed integers
//!
//! #### Real Double Operations (`real_double_*`)
//! - `real_double_set_d` - Set from double
//! - `real_double_get_d` - Get as double
//!
//! #### Complex Operations (`complex_*`)
//! - `complex_set` - Set from real and imaginary parts
//! - `complex_double_get` - Get as dcomplex struct
//! - `complex_base_real_part` - Extract real part
//! - `complex_base_imaginary_part` - Extract imaginary part
//!
//! ### 4. Container Types
//!
//! #### CVecBasic - Vector of Expressions
//!
//! Dynamic array of symbolic expressions.
//!
//! **Memory Management**: Must call `vecbasic_free` when done!
//!
//! ##### Key Functions
//! - `vecbasic_new` - Create new vector
//! - `vecbasic_free` - Free vector (**REQUIRED**)
//! - `vecbasic_size` - Get size
//! - `vecbasic_push_back` - Add element
//! - `vecbasic_get` - Get element at index
//! - `vecbasic_set` - Set element at index
//!
//! #### CMapBasicBasic - Map/Dictionary
//!
//! Hash map for variable substitution and key-value storage.
//!
//! **Memory Management**: Must call `mapbasicbasic_free` when done!
//!
//! ##### Key Functions
//! - `mapbasicbasic_new` - Create new map
//! - `mapbasicbasic_free` - Free map (**REQUIRED**)
//! - `mapbasicbasic_insert` - Insert key-value pair
//! - `mapbasicbasic_get` - Get value by key
//! - `mapbasicbasic_size` - Get size
//!
//! #### CSetBasic - Set of Expressions
//!
//! Set type used for free symbols and solution sets.
//!
//! **Note**: Direct manipulation functions are limited. Primarily used
//! internally by functions like `basic_free_symbols`.
//!
//! ### 5. Matrix Operations
//!
//! #### Dense Matrices (`dense_matrix_*`)
//!
//! Full matrix representation.
//!
//! **Memory Management**: Must call `dense_matrix_free` when done!
//!
//! ##### Creation and Management
//! - `dense_matrix_new` - Create empty matrix
//! - `dense_matrix_new_rows_cols` - Create with dimensions
//! - `dense_matrix_free` - Free matrix (**REQUIRED**)
//! - `dense_matrix_rows`, `dense_matrix_cols` - Get dimensions
//!
//! ##### Element Access
//! - `dense_matrix_set_basic` - Set element
//! - `dense_matrix_get_basic` - Get element
//!
//! ##### Operations
//! - `dense_matrix_add_matrix` - Matrix addition
//! - `dense_matrix_mul_matrix` - Matrix multiplication
//! - `dense_matrix_mul_scalar` - Scalar multiplication
//! - `dense_matrix_transpose` - Transpose
//! - `dense_matrix_det` - Determinant
//! - `dense_matrix_inv` - Inverse
//! - `dense_matrix_jacobian` - Jacobian matrix
//!
//! ##### Utilities
//! - `dense_matrix_eye` - Identity matrix
//! - `dense_matrix_zeros` - Zero matrix
//! - `dense_matrix_ones` - Matrix of ones
//! - `dense_matrix_diag` - Diagonal matrix
//!
//! #### Sparse Matrices (`sparse_matrix_*`)
//!
//! Memory-efficient representation for matrices with many zeros.
//!
//! **Memory Management**: Must call `sparse_matrix_free` when done!
//!
//! ##### Key Functions
//! - `sparse_matrix_new` - Create new sparse matrix
//! - `sparse_matrix_free` - Free sparse matrix (**REQUIRED**)
//! - `sparse_matrix_init` - Initialize
//! - `sparse_matrix_set_basic` - Set element
//! - `sparse_matrix_get_basic` - Get element
//! - `sparse_matrix_eq` - Test equality (returns c_int, 0 = false)
//! - `sparse_matrix_str` - String representation
//!
//! ### 6. Trigonometric Functions
//!
//! All return result in first parameter (out parameter).
//!
//! - `basic_sin`, `basic_cos`, `basic_tan` - Basic trig functions
//! - `basic_asin`, `basic_acos`, `basic_atan` - Inverse trig functions
//! - `basic_csc`, `basic_sec`, `basic_cot` - Reciprocal trig functions
//! - `basic_sinh`, `basic_cosh`, `basic_tanh` - Hyperbolic functions
//! - `basic_asinh`, `basic_acosh`, `basic_atanh` - Inverse hyperbolic
//!
//! ### 7. Special Functions
//!
//! - `basic_exp`, `basic_log` - Exponential and logarithm
//! - `basic_sqrt` - Square root
//! - `basic_abs` - Absolute value
//! - `basic_atan2` - Two-argument arctangent
//! - `basic_erf`, `basic_erfc` - Error functions
//! - `basic_gamma`, `basic_loggamma` - Gamma function
//! - `basic_lambertw` - Lambert W function
//!
//! ### 8. Number Theory (`ntheory_*`)
//!
//! - `ntheory_gcd` - Greatest common divisor
//! - `ntheory_lcm` - Least common multiple
//! - `ntheory_mod` - Modular arithmetic
//! - `ntheory_quotient` - Integer quotient
//! - `ntheory_factorial` - Factorial
//! - `ntheory_fibonacci` - Fibonacci numbers
//! - `ntheory_lucas` - Lucas numbers
//! - `ntheory_binomial` - Binomial coefficients
//!
//! ### 9. Parsing
//!
//! - `basic_parse` - Parse string to expression
//! - `basic_parse2` - Parse with additional options
//!
//! **Returns**: Error code (use `check_result`)
//!
//! ### 10. Constants
//!
//! - `basic_const_zero`, `basic_const_one` - Numeric constants
//! - `basic_const_minus_one` - -1
//! - `basic_const_I` - Imaginary unit
//! - `basic_const_pi` - π
//! - `basic_const_E` - Euler's number e
//! - `basic_const_infinity` - Infinity
//! - `basic_const_complex_infinity` - Complex infinity
//!
//! ## Error Handling
//!
//! Most functions return `CWRAPPER_OUTPUT_TYPE` (c_int):
//! - `0` (SYMENGINE_NO_EXCEPTION) = Success
//! - Non-zero = Error (see `error_codes` module)
//!
//! ### Error Codes
//! - `SYMENGINE_NO_EXCEPTION` (0) - Success
//! - `SYMENGINE_RUNTIME_ERROR` (1) - Runtime error
//! - `SYMENGINE_DIV_BY_ZERO` (2) - Division by zero
//! - `SYMENGINE_NOT_IMPLEMENTED` (3) - Feature not implemented
//! - `SYMENGINE_DOMAIN_ERROR` (4) - Domain error
//! - `SYMENGINE_PARSE_ERROR` (5) - Parse error
//!
//! ### Checking Results
//!
//! ```no_run
//! use quantrs2_symengine_sys::*;
//! use std::os::raw::c_int;
//!
//! unsafe {
//!     let mut result = std::mem::zeroed::<basic_struct>();
//!     let code = basic_parse(&raw mut result, c"x + y".as_ptr());
//!
//!     match check_result(code as c_int) {
//!         Ok(_) => { /* success */ },
//!         Err(e) => { /* handle error */ }
//!     }
//! }
//! ```
//!
//! ## Type Codes
//!
//! Use `basic_get_type` to determine expression type:
//! - `SYMENGINE_SYMBOL` (1) - Variable
//! - `SYMENGINE_ADD` (2) - Addition
//! - `SYMENGINE_MUL` (3) - Multiplication
//! - `SYMENGINE_POW` (4) - Power
//! - `SYMENGINE_INTEGER` (5) - Integer
//! - `SYMENGINE_RATIONAL` (6) - Rational number
//! - `SYMENGINE_REAL_DOUBLE` (7) - Double precision real
//! - `SYMENGINE_COMPLEX_DOUBLE` (8) - Double precision complex
//!
//! ## Memory Management Guidelines
//!
//! ### Stack-Allocated Structures
//! - `basic_struct` - Use `std::mem::zeroed()`, no free needed
//!
//! ### Heap-Allocated Containers (MUST FREE!)
//! - `CVecBasic` - Call `vecbasic_free`
//! - `CMapBasicBasic` - Call `mapbasicbasic_free`
//! - `CDenseMatrix` - Call `dense_matrix_free`
//! - `CSparseMatrix` - Call `sparse_matrix_free`
//!
//! ### Strings from SymEngine (MUST FREE!)
//! - `basic_str` result - Call `basic_str_free`
//! - `dense_matrix_str` result - Call `basic_str_free`
//! - `sparse_matrix_str` result - Call `basic_str_free`
//!
//! ### Memory Leak Prevention
//!
//! ```no_run
//! use quantrs2_symengine_sys::*;
//!
//! unsafe {
//!     let vec = vecbasic_new();
//!     // ... use vec ...
//!     vecbasic_free(vec); // MUST call!
//!
//!     let mut expr = std::mem::zeroed::<basic_struct>();
//!     symbol_set(&raw mut expr, c"x".as_ptr());
//!     let s = basic_str(&raw const expr);
//!     // ... use string ...
//!     basic_str_free(s); // MUST call!
//! }
//! ```
//!
//! ## Thread Safety
//!
//! - SymEngine uses thread-local storage by default
//! - Each thread should maintain its own symbol tables
//! - Sharing `basic_struct` across threads is generally safe
//! - Container types (CVecBasic, etc.) should not be shared without synchronization
//! - Check SymEngine's compilation flags for thread-safety guarantees
//!
//! ## Performance Tips
//!
//! 1. **Reuse structures**: Don't create/destroy frequently
//! 2. **Batch operations**: Group related calculations
//! 3. **Use appropriate types**:
//!    - Sparse matrices for large, mostly-zero matrices
//!    - Integer arithmetic when possible (faster than symbolic)
//! 4. **Minimize string conversions**: `basic_str` is expensive
//! 5. **Pre-allocate containers**: Set initial capacity when known
//!
//! ## Common Patterns
//!
//! ### Creating a Polynomial
//!
//! ```no_run
//! use quantrs2_symengine_sys::*;
//!
//! unsafe {
//!     // Create x^2 + 2*x + 1
//!     let mut x = std::mem::zeroed::<basic_struct>();
//!     let mut two = std::mem::zeroed::<basic_struct>();
//!     let mut one = std::mem::zeroed::<basic_struct>();
//!
//!     symbol_set(&raw mut x, c"x".as_ptr());
//!     integer_set_si(&raw mut two, 2);
//!     integer_set_si(&raw mut one, 1);
//!
//!     let mut x2 = std::mem::zeroed::<basic_struct>();
//!     basic_pow(&raw mut x2, &raw const x, &raw const two);
//!
//!     let mut two_x = std::mem::zeroed::<basic_struct>();
//!     basic_mul(&raw mut two_x, &raw const two, &raw const x);
//!
//!     let mut temp = std::mem::zeroed::<basic_struct>();
//!     basic_add(&raw mut temp, &raw const x2, &raw const two_x);
//!
//!     let mut result = std::mem::zeroed::<basic_struct>();
//!     basic_add(&raw mut result, &raw const temp, &raw const one);
//! }
//! ```
//!
//! ### Substitution Workflow
//!
//! ```no_run
//! use quantrs2_symengine_sys::*;
//!
//! unsafe {
//!     let expr = /* expression with x */;
//!     let mut x = std::mem::zeroed::<basic_struct>();
//!     let mut val = std::mem::zeroed::<basic_struct>();
//!
//!     symbol_set(&raw mut x, c"x".as_ptr());
//!     integer_set_si(&raw mut val, 42);
//!
//!     let mut result = std::mem::zeroed::<basic_struct>();
//!     basic_subs2(&raw mut result, &raw const expr, &raw const x, &raw const val);
//! }
//! ```
//!
//! ### Matrix Computation
//!
//! ```no_run
//! use quantrs2_symengine_sys::*;
//!
//! unsafe {
//!     let mat = dense_matrix_new_rows_cols(3, 3);
//!
//!     // Set elements
//!     let mut one = std::mem::zeroed::<basic_struct>();
//!     integer_set_si(&raw mut one, 1);
//!     dense_matrix_set_basic(mat, 0, 0, &raw const one);
//!
//!     // Compute determinant
//!     let mut det = std::mem::zeroed::<basic_struct>();
//!     dense_matrix_det(&raw mut det, mat);
//!
//!     dense_matrix_free(mat); // Don't forget!
//! }
//! ```
//!
//! ## Integration with Higher-Level Crates
//!
//! This crate provides raw FFI bindings. For a safer, more ergonomic interface:
//! - Use `quantrs2-symengine` for safe Rust wrappers
//! - Use `quantrs2-circuit` for symbolic quantum circuit operations
//! - Use `quantrs2-ml` for symbolic gradients in quantum ML
//!
//! ## See Also
//!
//! - [SymEngine C API Documentation](https://github.com/symengine/symengine/wiki/C-API)
//! - [Examples](../examples) - Complete working examples
//! - [Integration Tests](../tests) - Complex workflow tests
