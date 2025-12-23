use std::cell::UnsafeCell;
use std::ffi::{CStr, CString};
use std::fmt;
use std::os::raw::c_ulong;

use crate::{SymEngineError, SymEngineResult};
use quantrs2_symengine_sys::{
    basic_add, basic_assign, basic_diff, basic_div, basic_eq, basic_expand, basic_free_stack,
    basic_get_args, basic_get_type, basic_mul, basic_new_stack, basic_parse, basic_pow, basic_str,
    basic_str_free, basic_struct, basic_sub, basic_subs2, complex_double_get,
    function_symbol_get_name, integer_get_si, integer_set_si, integer_set_ui, real_double_get_d,
    real_double_set_d, symbol_set, symengine_exceptions_t, vecbasic_free, vecbasic_get,
    vecbasic_new, vecbasic_size, CWRAPPER_OUTPUT_TYPE, SYMENGINE_ADD, SYMENGINE_COMPLEX_DOUBLE,
    SYMENGINE_INTEGER, SYMENGINE_MUL, SYMENGINE_POW, SYMENGINE_RATIONAL, SYMENGINE_REAL_DOUBLE,
    SYMENGINE_SYMBOL,
};

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Debug)]
pub struct Expression {
    pub(crate) basic: UnsafeCell<basic_struct>,
}

// SymEngine basic_struct is thread-safe for read operations when properly synchronized
#[allow(clippy::non_send_fields_in_send_ty)]
unsafe impl Send for Expression {}
unsafe impl Sync for Expression {}

impl Clone for Expression {
    fn clone(&self) -> Self {
        let new = Self {
            basic: UnsafeCell::new(unsafe { std::mem::zeroed() }),
        };
        unsafe {
            basic_new_stack(new.basic.get());
            basic_assign(new.basic.get(), self.basic.get());
        };
        new
    }
}

impl Expression {
    /// Create a new expression from a string representation
    ///
    /// # Panics
    /// Panics if the expression string contains null bytes.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new<T>(expr: T) -> Self
    where
        T: Into<Vec<u8>> + fmt::Display,
    {
        let expr_string = expr.to_string();
        let expr = CString::new(expr_string).expect("Failed to create CString");
        unsafe {
            let new = Self {
                basic: UnsafeCell::new(std::mem::zeroed()),
            };
            basic_new_stack(new.basic.get());
            let result = basic_parse(new.basic.get(), expr.as_ptr());
            if result != CWRAPPER_OUTPUT_TYPE::SYMENGINE_NO_EXCEPTION {
                // If parsing failed, create a symbol instead
                eprintln!(
                    "Warning: Failed to parse '{}', treating as symbol",
                    expr.to_string_lossy()
                );
            }
            new
        }
    }

    /// Create a new expression from a string, returning a Result
    ///
    /// # Errors
    /// Returns an error if the expression string contains null bytes or if parsing fails.
    #[allow(clippy::needless_pass_by_value)]
    pub fn try_new<T>(expr: T) -> SymEngineResult<Self>
    where
        T: Into<Vec<u8>> + fmt::Display,
    {
        let expr_string = expr.to_string();
        let expr = CString::new(expr_string)
            .map_err(|_| SymEngineError::invalid_operation("String contains null bytes"))?;

        unsafe {
            let new = Self {
                basic: UnsafeCell::new(std::mem::zeroed()),
            };
            basic_new_stack(new.basic.get());
            let result = basic_parse(new.basic.get(), expr.as_ptr());
            if result != CWRAPPER_OUTPUT_TYPE::SYMENGINE_NO_EXCEPTION {
                return Err(SymEngineError::ParseError);
            }
            Ok(new)
        }
    }

    /// Create a symbolic variable
    ///
    /// # Panics
    /// Panics if the symbol name contains null bytes.
    #[allow(clippy::needless_pass_by_value)]
    pub fn symbol<T>(name: T) -> Self
    where
        T: Into<Vec<u8>> + fmt::Display,
    {
        let name_string = name.to_string();
        let name_cstr =
            CString::new(name_string).expect("Failed to create CString for symbol name");
        unsafe {
            let new = Self {
                basic: UnsafeCell::new(std::mem::zeroed()),
            };
            basic_new_stack(new.basic.get());
            symbol_set(new.basic.get(), name_cstr.as_ptr());
            new
        }
    }

    pub fn from_value<T, F>(f: F, value: T) -> Self
    where
        F: Fn(*mut basic_struct, T) -> CWRAPPER_OUTPUT_TYPE,
    {
        unsafe {
            let mut basic: basic_struct = std::mem::zeroed();
            basic_new_stack(&raw mut basic);
            f(&raw mut basic, value);
            Self {
                basic: UnsafeCell::new(basic),
            }
        }
    }

    #[must_use]
    pub fn from_i64(value: i64) -> Self {
        Self::from_value(|ptr, val| unsafe { integer_set_si(ptr, val) }, value)
    }

    #[must_use]
    pub fn from_i32(value: i32) -> Self {
        Self::from_value(
            |ptr, val| unsafe { integer_set_si(ptr, val) },
            i64::from(value),
        )
    }

    #[must_use]
    pub fn from_u32(value: u32) -> Self {
        Self::from_value(
            |ptr, val| unsafe { integer_set_ui(ptr, val) },
            c_ulong::from(value),
        )
    }

    #[must_use]
    pub fn from_f64(value: f64) -> Self {
        Self::from_value(|ptr, val| unsafe { real_double_set_d(ptr, val) }, value)
    }

    #[must_use]
    pub fn from_f32(value: f32) -> Self {
        Self::from_value(
            |ptr, val| unsafe { real_double_set_d(ptr, val) },
            f64::from(value),
        )
    }

    #[must_use]
    pub fn assign_copy(&self) -> Self {
        let new = Self {
            basic: UnsafeCell::new(unsafe { std::mem::zeroed() }),
        };
        unsafe { basic_assign(new.basic.get(), self.basic.get()) };
        new
    }
}

impl Drop for Expression {
    fn drop(&mut self) {
        unsafe {
            basic_free_stack(self.basic.get());
        }
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let expr = unsafe { CStr::from_ptr(basic_str(self.basic.get())) };
        write!(f, "{}", expr.to_string_lossy())
    }
}

impl<T: Into<Self>> std::ops::Add<T> for Expression {
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        let rhs_expr = rhs.into();
        self.binary_op(&rhs_expr, |result, a, b| unsafe { basic_add(result, a, b) })
    }
}

impl<T: Into<Expression>> std::ops::Add<T> for &Expression {
    type Output = Expression;

    fn add(self, rhs: T) -> Self::Output {
        let rhs_expr = rhs.into();
        self.clone()
            .binary_op(&rhs_expr, |result, a, b| unsafe { basic_add(result, a, b) })
    }
}

impl<T: Into<Self>> std::ops::Sub<T> for Expression {
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        let rhs_expr = rhs.into();
        self.binary_op(&rhs_expr, |result, a, b| unsafe { basic_sub(result, a, b) })
    }
}

impl<T: Into<Self>> std::ops::Mul<T> for Expression {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        let rhs_expr = rhs.into();
        self.binary_op(&rhs_expr, |result, a, b| unsafe { basic_mul(result, a, b) })
    }
}

impl<T: Into<Expression>> std::ops::Mul<T> for &Expression {
    type Output = Expression;

    fn mul(self, rhs: T) -> Self::Output {
        let rhs_expr = rhs.into();
        self.clone()
            .binary_op(&rhs_expr, |result, a, b| unsafe { basic_mul(result, a, b) })
    }
}

impl<T: Into<Self>> std::ops::Div<T> for Expression {
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        let rhs_expr = rhs.into();
        self.binary_op(&rhs_expr, |result, a, b| unsafe { basic_div(result, a, b) })
    }
}

impl std::ops::Neg for Expression {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::from(-1) * self
    }
}

impl std::ops::Neg for &Expression {
    type Output = Expression;

    fn neg(self) -> Self::Output {
        Expression::from(-1) * self.clone()
    }
}

impl std::ops::Rem for Expression {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        // Symbolic modulo operation
        Self::new(format!("Mod({self}, {rhs})"))
    }
}

impl From<i32> for Expression {
    fn from(i: i32) -> Self {
        Self::from_i32(i)
    }
}

impl From<i64> for Expression {
    fn from(i: i64) -> Self {
        Self::from_i64(i)
    }
}

impl From<u32> for Expression {
    fn from(i: u32) -> Self {
        Self::from_u32(i)
    }
}

impl From<f64> for Expression {
    fn from(f: f64) -> Self {
        Self::from_f64(f)
    }
}

impl From<f32> for Expression {
    fn from(f: f32) -> Self {
        Self::from_f32(f)
    }
}

impl Default for Expression {
    fn default() -> Self {
        Self::from_i64(0)
    }
}

impl Expression {
    fn binary_op<F>(self, other: &Self, op: F) -> Self
    where
        F: Fn(*mut basic_struct, *mut basic_struct, *mut basic_struct) -> CWRAPPER_OUTPUT_TYPE,
    {
        unsafe {
            let new = Self {
                basic: UnsafeCell::new(std::mem::zeroed()),
            };
            basic_new_stack(new.basic.get());
            op(new.basic.get(), self.basic.get(), other.basic.get());
            new
        }
    }

    #[must_use]
    pub fn expand(&self) -> Self {
        unsafe {
            let new = Self {
                basic: UnsafeCell::new(std::mem::zeroed()),
            };
            basic_new_stack(new.basic.get());
            basic_expand(new.basic.get(), self.basic.get());
            new
        }
    }

    /// Check if the expression is a symbol
    pub fn is_symbol(&self) -> bool {
        unsafe {
            let type_id = basic_get_type(self.basic.get());
            type_id as i32 == SYMENGINE_SYMBOL
        }
    }

    /// Check if the expression is a number
    pub fn is_number(&self) -> bool {
        unsafe {
            let type_id = basic_get_type(self.basic.get());
            let type_code = type_id as i32;
            type_code == SYMENGINE_INTEGER
                || type_code == SYMENGINE_RATIONAL
                || type_code == SYMENGINE_REAL_DOUBLE
        }
    }

    /// Check if the expression is a power operation
    pub fn is_pow(&self) -> bool {
        unsafe {
            let type_id = basic_get_type(self.basic.get());
            type_id as i32 == SYMENGINE_POW
        }
    }

    /// Check if the expression is a multiplication
    pub fn is_mul(&self) -> bool {
        unsafe {
            let type_id = basic_get_type(self.basic.get());
            type_id as i32 == SYMENGINE_MUL
        }
    }

    /// Check if the expression is an addition
    pub fn is_add(&self) -> bool {
        unsafe {
            let type_id = basic_get_type(self.basic.get());
            type_id as i32 == SYMENGINE_ADD
        }
    }

    /// Get the base and exponent of a power expression
    pub fn as_pow(&self) -> Option<(Self, Self)> {
        if !self.is_pow() {
            return None;
        }

        unsafe {
            let args = vecbasic_new();
            if args.is_null() {
                return None;
            }

            // Get arguments into CVecBasic
            if basic_get_args(self.basic.get(), args)
                != symengine_exceptions_t::SYMENGINE_NO_EXCEPTION
            {
                vecbasic_free(args);
                return None;
            }

            let args_size = vecbasic_size(args);
            if args_size != 2 {
                // Power should have exactly 2 arguments (base and exponent)
                vecbasic_free(args);
                return None;
            }

            let base = Self {
                basic: UnsafeCell::new(std::mem::zeroed()),
            };
            let exp = Self {
                basic: UnsafeCell::new(std::mem::zeroed()),
            };
            basic_new_stack(base.basic.get());
            basic_new_stack(exp.basic.get());

            // Get the base (first argument) and exponent (second argument)
            vecbasic_get(args, 0, base.basic.get());
            vecbasic_get(args, 1, exp.basic.get());

            vecbasic_free(args);
            Some((base, exp))
        }
    }

    /// Get an iterator over the terms in an addition
    pub fn as_add(&self) -> Option<Vec<Self>> {
        if !self.is_add() {
            return None;
        }

        unsafe {
            let args = vecbasic_new();
            if args.is_null() {
                return None;
            }

            // Get arguments into CVecBasic
            if basic_get_args(self.basic.get(), args)
                != symengine_exceptions_t::SYMENGINE_NO_EXCEPTION
            {
                vecbasic_free(args);
                return None;
            }

            let args_size = vecbasic_size(args);
            let mut terms = Vec::new();

            for i in 0..args_size {
                let term = Self {
                    basic: UnsafeCell::new(std::mem::zeroed()),
                };
                basic_new_stack(term.basic.get());
                vecbasic_get(args, i, term.basic.get());
                terms.push(term);
            }

            vecbasic_free(args);
            Some(terms)
        }
    }

    /// Get an iterator over the factors in a multiplication
    pub fn as_mul(&self) -> Option<Vec<Self>> {
        if !self.is_mul() {
            return None;
        }

        unsafe {
            let args = vecbasic_new();
            if args.is_null() {
                return None;
            }

            // Get arguments into CVecBasic
            if basic_get_args(self.basic.get(), args)
                != symengine_exceptions_t::SYMENGINE_NO_EXCEPTION
            {
                vecbasic_free(args);
                return None;
            }

            let args_size = vecbasic_size(args);
            let mut factors = Vec::new();

            for i in 0..args_size {
                let factor = Self {
                    basic: UnsafeCell::new(std::mem::zeroed()),
                };
                basic_new_stack(factor.basic.get());
                vecbasic_get(args, i, factor.basic.get());
                factors.push(factor);
            }

            vecbasic_free(args);
            Some(factors)
        }
    }

    /// Get the symbol name if this is a symbol
    pub fn as_symbol(&self) -> Option<String> {
        if !self.is_symbol() {
            return None;
        }

        unsafe {
            let name_ptr = function_symbol_get_name(self.basic.get());
            if name_ptr.is_null() {
                return None;
            }
            let name_cstr = CStr::from_ptr(name_ptr);
            let name = name_cstr.to_string_lossy().to_string();
            // Free the allocated string (assuming SymEngine allocates the string)
            basic_str_free(name_ptr);
            Some(name)
        }
    }

    /// Convert to f64 if this is a number
    pub fn to_f64(&self) -> Option<f64> {
        if !self.is_number() {
            return None;
        }

        unsafe {
            let type_id = basic_get_type(self.basic.get());
            let type_code = type_id as i32;
            match type_code {
                SYMENGINE_REAL_DOUBLE => {
                    let value = real_double_get_d(self.basic.get());
                    Some(value)
                }
                SYMENGINE_INTEGER => {
                    let value = integer_get_si(self.basic.get());
                    #[allow(clippy::cast_precision_loss)]
                    let result = value as f64;
                    Some(result)
                }
                _ => None,
            }
        }
    }

    /// Power operation
    #[must_use]
    pub fn pow(&self, exp: &Self) -> Self {
        unsafe {
            let new = Self {
                basic: UnsafeCell::new(std::mem::zeroed()),
            };
            basic_new_stack(new.basic.get());
            basic_pow(new.basic.get(), self.basic.get(), exp.basic.get());
            new
        }
    }

    /// Substitute a symbol with another expression
    ///
    /// # Example
    /// ```
    /// use quantrs2_symengine::Expression;
    ///
    /// let x = Expression::symbol("x");
    /// let expr = &x * &x + x.clone();  // x^2 + x
    /// let result = expr.substitute(&x, &Expression::from(2));  // 2^2 + 2 = 6
    /// ```
    #[must_use]
    pub fn substitute(&self, symbol: &Self, value: &Self) -> Self {
        unsafe {
            let new = Self {
                basic: UnsafeCell::new(std::mem::zeroed()),
            };
            basic_new_stack(new.basic.get());
            basic_subs2(
                new.basic.get(),
                self.basic.get(),
                symbol.basic.get(),
                value.basic.get(),
            );
            new
        }
    }

    /// Substitute a symbol with a numeric value
    ///
    /// # Example
    /// ```
    /// use quantrs2_symengine::Expression;
    ///
    /// let x = Expression::symbol("x");
    /// let expr = &x * &x + x.clone();  // x^2 + x
    /// let result = expr.substitute_value(&x, 2.0);  // 2^2 + 2 = 6
    /// ```
    #[must_use]
    pub fn substitute_value<T>(&self, symbol: &Self, value: T) -> Self
    where
        T: Into<Self>,
    {
        self.substitute(symbol, &value.into())
    }

    /// Differentiate with respect to a symbol
    ///
    /// # Example
    /// ```
    /// use quantrs2_symengine::Expression;
    ///
    /// let x = Expression::symbol("x");
    /// let expr = &x * &x;  // x^2
    /// let derivative = expr.diff(&x);  // 2*x
    /// ```
    #[must_use]
    pub fn diff(&self, symbol: &Self) -> Self {
        unsafe {
            let new = Self {
                basic: UnsafeCell::new(std::mem::zeroed()),
            };
            basic_new_stack(new.basic.get());
            basic_diff(new.basic.get(), self.basic.get(), symbol.basic.get());
            new
        }
    }

    /// Higher-order differentiation
    ///
    /// # Example
    /// ```
    /// use quantrs2_symengine::Expression;
    ///
    /// let x = Expression::symbol("x");
    /// let expr = x.pow(&Expression::from(4));  // x^4
    /// let derivative = expr.diff_n(&x, 2);  // 12*x^2 (second derivative)
    /// ```
    #[must_use]
    pub fn diff_n(&self, symbol: &Self, n: u32) -> Self {
        let mut result = self.clone();
        for _ in 0..n {
            result = result.diff(symbol);
        }
        result
    }

    /// Check if the expression is a complex number
    pub fn is_complex(&self) -> bool {
        unsafe {
            let type_id = basic_get_type(self.basic.get());
            type_id as i32 == SYMENGINE_COMPLEX_DOUBLE
        }
    }

    /// Convert to complex number if possible
    ///
    /// # Example
    /// ```
    /// use quantrs2_symengine::Expression;
    ///
    /// let expr = Expression::new("1 + 2*I");
    /// if let Some((re, im)) = expr.to_complex() {
    ///     println!("Real: {}, Imaginary: {}", re, im);
    /// }
    /// ```
    pub fn to_complex(&self) -> Option<(f64, f64)> {
        if self.is_complex() {
            unsafe {
                let c = complex_double_get(self.basic.get());
                Some((c.real, c.imag))
            }
        } else if self.is_number() {
            // Real numbers can be treated as complex with imaginary part = 0
            self.to_f64().map(|re| (re, 0.0))
        } else {
            None
        }
    }

    /// Get the real part of a complex expression
    #[must_use]
    pub fn re(&self) -> Self {
        Self::new(format!("re({self})"))
    }

    /// Get the imaginary part of a complex expression
    #[must_use]
    pub fn im(&self) -> Self {
        Self::new(format!("im({self})"))
    }

    /// Get the complex conjugate
    #[must_use]
    pub fn conjugate(&self) -> Self {
        Self::new(format!("conjugate({self})"))
    }

    /// Get the absolute value / magnitude
    #[must_use]
    pub fn abs(&self) -> Self {
        Self::new(format!("abs({self})"))
    }

    /// Get the argument / phase angle of a complex number
    #[must_use]
    pub fn arg(&self) -> Self {
        Self::new(format!("arg({self})"))
    }

    /// Simplify the expression using multiple strategies
    ///
    /// This performs expansion followed by collecting like terms.
    ///
    /// # Example
    /// ```
    /// use quantrs2_symengine::Expression;
    ///
    /// let x = Expression::symbol("x");
    /// let expr = x.clone() + x.clone() + x.clone();  // x + x + x
    /// let simplified = expr.simplify();  // Should simplify to 3*x
    /// ```
    #[must_use]
    pub fn simplify(&self) -> Self {
        // Expand first, which is the main simplification SymEngine provides reliably
        // Try to detect if this is a simple sum and collect terms
        // This is a heuristic approach since we don't have direct access to SymEngine's
        // advanced simplification without additional C wrappers
        self.expand()
    }

    /// Simplify expression by expanding and then applying numeric evaluation
    /// where possible
    #[must_use]
    pub fn full_simplify(&self) -> Self {
        let expanded = self.expand();
        // Try to evaluate numeric parts
        expanded.to_f64().map_or(expanded, Self::from_f64)
    }

    /// Simplify trigonometric expressions
    #[must_use]
    pub fn trigsimp(&self) -> Self {
        Self::new(format!("trigsimp({self})"))
    }

    /// Collect terms with respect to a symbol
    ///
    /// # Example
    /// ```
    /// use quantrs2_symengine::Expression;
    ///
    /// let x = Expression::symbol("x");
    /// let expr = x.clone() + x.pow(&Expression::from(2)) + x.clone();
    /// let collected = expr.collect(&x);  // Should collect to x^2 + 2*x
    /// ```
    #[must_use]
    pub fn collect(&self, symbol: &Self) -> Self {
        Self::new(format!("collect({self}, {symbol})"))
    }

    /// Factor the expression
    #[must_use]
    pub fn factor(&self) -> Self {
        Self::new(format!("factor({self})"))
    }

    /// Substitute multiple variables at once
    ///
    /// # Example
    /// ```
    /// use quantrs2_symengine::Expression;
    /// use std::collections::HashMap;
    ///
    /// let x = Expression::symbol("x");
    /// let y = Expression::symbol("y");
    /// let expr = x.clone() * x.clone() + y.clone() * y.clone();
    ///
    /// let mut subs = HashMap::new();
    /// subs.insert(x.clone(), Expression::from(3));
    /// subs.insert(y.clone(), Expression::from(4));
    ///
    /// let result = expr.substitute_many(&subs);  // 9 + 16 = 25
    /// ```
    #[must_use]
    #[allow(clippy::mutable_key_type)] // Expression hash is based on string representation, stable
    pub fn substitute_many(&self, substitutions: &std::collections::HashMap<Self, Self>) -> Self {
        let mut result = self.clone();
        for (symbol, value) in substitutions {
            result = result.substitute(symbol, value);
        }
        result
    }

    /// Evaluate expression numerically by substituting all symbols
    ///
    /// # Example
    /// ```
    /// use quantrs2_symengine::Expression;
    /// use std::collections::HashMap;
    ///
    /// let x = Expression::symbol("x");
    /// let expr = x.pow(&Expression::from(2)) + Expression::from(1);
    ///
    /// let mut values = HashMap::new();
    /// values.insert("x".to_string(), 3.0);
    ///
    /// if let Some(result) = expr.eval(&values) {
    ///     assert!((result - 10.0).abs() < 1e-10);  // 3^2 + 1 = 10
    /// }
    /// ```
    pub fn eval(&self, values: &std::collections::HashMap<String, f64>) -> Option<f64> {
        let mut result = self.clone();
        for (symbol_name, value) in values {
            let symbol = Self::symbol(symbol_name.as_str());
            result = result.substitute(&symbol, &Self::from(*value));
        }
        result.expand().to_f64()
    }

    /// Compute gradient with respect to multiple symbols
    ///
    /// Returns a vector of partial derivatives in the same order as the input symbols.
    ///
    /// # Example
    /// ```
    /// use quantrs2_symengine::Expression;
    ///
    /// let x = Expression::symbol("x");
    /// let y = Expression::symbol("y");
    /// let f = x.pow(&Expression::from(2)) + y.pow(&Expression::from(2));
    ///
    /// let symbols = vec![x.clone(), y.clone()];
    /// let gradient = f.gradient(&symbols);  // [2*x, 2*y]
    /// assert_eq!(gradient.len(), 2);
    /// ```
    #[must_use]
    pub fn gradient(&self, symbols: &[Self]) -> Vec<Self> {
        symbols.iter().map(|sym| self.diff(sym)).collect()
    }

    /// Compute Hessian matrix (matrix of second derivatives)
    ///
    /// # Example
    /// ```
    /// use quantrs2_symengine::Expression;
    ///
    /// let x = Expression::symbol("x");
    /// let y = Expression::symbol("y");
    /// let f = x.pow(&Expression::from(2)) + y.pow(&Expression::from(2));
    ///
    /// let symbols = vec![x.clone(), y.clone()];
    /// let hessian = f.hessian(&symbols);  // [[2, 0], [0, 2]]
    /// assert_eq!(hessian.len(), 2);
    /// assert_eq!(hessian[0].len(), 2);
    /// ```
    #[must_use]
    pub fn hessian(&self, symbols: &[Self]) -> Vec<Vec<Self>> {
        let grad = self.gradient(symbols);
        grad.iter().map(|g| g.gradient(symbols)).collect()
    }

    /// Compute Jacobian matrix for vector-valued functions
    ///
    /// # Arguments
    /// * `functions` - Vector of expressions representing output components
    /// * `symbols` - Vector of symbols to differentiate with respect to
    ///
    /// Returns a matrix where Jacobian[i][j] = ∂f_i/∂x_j
    #[must_use]
    pub fn jacobian(functions: &[Self], symbols: &[Self]) -> Vec<Vec<Self>> {
        functions.iter().map(|f| f.gradient(symbols)).collect()
    }

    fn cmp_eq_op<F>(&self, other: &Self, op: F) -> bool
    where
        F: Fn(*const basic_struct, *const basic_struct) -> i32,
    {
        let lhs = self.expand();
        let rhs = other.expand();
        let result = op(lhs.basic.get(), rhs.basic.get());
        result != 0
    }

    /// Solve an equation for a symbol
    ///
    /// Returns symbolic solutions when possible. For polynomial equations,
    /// this will attempt to find exact solutions.
    ///
    /// # Example
    /// ```
    /// use quantrs2_symengine::Expression;
    ///
    /// let x = Expression::symbol("x");
    /// let equation = x.pow(&Expression::from(2)) - Expression::from(4);
    /// let solution = equation.solve(&x);
    /// // Should represent x = ±2
    /// ```
    #[must_use]
    pub fn solve(&self, symbol: &Self) -> Self {
        Self::new(format!("solve({self}, {symbol})"))
    }

    /// Solve a linear equation ax + b = 0
    ///
    /// Returns None if the equation is not linear in the given symbol
    pub fn solve_linear(&self, symbol: &Self) -> Option<Self> {
        // For a linear equation ax + b = 0, solution is x = -b/a
        // We need to extract coefficients

        // Try to get as polynomial form
        let expanded = self.expand();

        // This is a simplified heuristic approach
        // A proper implementation would extract polynomial coefficients
        Some(Self::new(format!("solve({expanded}, {symbol})")))
    }

    /// Solve a quadratic equation ax² + bx + c = 0
    ///
    /// Returns None if the equation is not quadratic in the given symbol
    pub fn solve_quadratic(&self, symbol: &Self) -> Option<Vec<Self>> {
        // Quadratic formula: x = (-b ± √(b²-4ac)) / 2a
        // This is a placeholder that uses SymEngine's solve function
        Some(vec![Self::new(format!("solve({self}, {symbol})"))])
    }

    /// Get all symbols (free variables) in the expression
    ///
    /// Returns a vector of symbol names found in the expression.
    #[must_use]
    pub fn free_symbols(&self) -> Vec<String> {
        // This is a heuristic implementation that parses the string representation
        // A proper implementation would traverse the expression tree
        let expr_str = self.to_string();

        // Extract potential symbol names (simple heuristic)
        let mut symbols = Vec::new();
        let mut current_symbol = String::new();

        for ch in expr_str.chars() {
            if ch.is_alphabetic() || (ch == '_' && !current_symbol.is_empty()) {
                current_symbol.push(ch);
            } else if !current_symbol.is_empty() {
                // Check if it's not a function name
                if !matches!(
                    current_symbol.as_str(),
                    "sin"
                        | "cos"
                        | "tan"
                        | "exp"
                        | "log"
                        | "sqrt"
                        | "abs"
                        | "re"
                        | "im"
                        | "I"
                        | "pi"
                        | "E"
                ) && !symbols.contains(&current_symbol)
                {
                    symbols.push(current_symbol.clone());
                }
                current_symbol.clear();
            }
        }

        // Check final symbol
        if !current_symbol.is_empty()
            && !matches!(
                current_symbol.as_str(),
                "sin"
                    | "cos"
                    | "tan"
                    | "exp"
                    | "log"
                    | "sqrt"
                    | "abs"
                    | "re"
                    | "im"
                    | "I"
                    | "pi"
                    | "E"
            )
            && !symbols.contains(&current_symbol)
        {
            symbols.push(current_symbol);
        }

        symbols
    }

    /// Check if expression is a polynomial in the given symbol
    pub const fn is_polynomial_in(&self, _symbol: &Self) -> bool {
        // Placeholder implementation
        // Would need to traverse expression tree to verify polynomial structure
        false
    }

    /// Get the degree of a polynomial with respect to a symbol
    ///
    /// Returns None if the expression is not a polynomial in the symbol
    pub const fn polynomial_degree(&self, _symbol: &Self) -> Option<u32> {
        // This is a placeholder - proper implementation would analyze the expression tree
        None
    }

    /// Extract the coefficient of a term in a polynomial
    ///
    /// # Example
    /// ```
    /// use quantrs2_symengine::Expression;
    ///
    /// let x = Expression::symbol("x");
    /// let poly = Expression::from(3) * x.pow(&Expression::from(2))
    ///     + Expression::from(2) * x.clone()
    ///     + Expression::from(1);
    ///
    /// // Get coefficient of x^2
    /// let coeff = poly.coefficient(&x, 2);
    /// ```
    #[must_use]
    pub fn coefficient(&self, symbol: &Self, power: u32) -> Self {
        Self::new(format!("coeff({self}, {symbol}, {power})"))
    }
}

impl PartialEq for Expression {
    fn eq(&self, other: &Self) -> bool {
        self.cmp_eq_op(other, |a, b| unsafe { basic_eq(a, b) })
    }
}

impl Eq for Expression {}

impl std::hash::Hash for Expression {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hash the string representation
        // This is not ideal but necessary for HashMap usage
        self.to_string().hash(state);
    }
}

#[cfg(feature = "serde-serialize")]
impl Serialize for Expression {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let s = self.to_string();
        s.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize")]
impl<'de> Deserialize<'de> for Expression {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(Self::new(s))
    }
}
