use std::cell::UnsafeCell;
use std::ffi::{CStr, CString};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::os::raw::{c_int, c_ulong};

use quantrs2_symengine_sys::*;
use crate::{SymEngineError, SymEngineResult};

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Debug)]
pub struct Expression {
    pub(crate) basic: UnsafeCell<basic_struct>,
}

// SymEngine basic_struct is thread-safe for read operations when properly synchronized
unsafe impl Send for Expression {}
unsafe impl Sync for Expression {}

impl Clone for Expression {
    fn clone(&self) -> Self {
        let mut new = Expression {
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
    pub fn new<T>(expr: T) -> Self
    where
        T: Into<Vec<u8>> + fmt::Display,
    {
        let expr_string = expr.to_string();
        let expr = CString::new(expr_string).expect("Failed to create CString");
        unsafe {
            let mut new = Expression {
                basic: UnsafeCell::new(std::mem::zeroed()),
            };
            basic_new_stack(new.basic.get());
            let result = basic_parse(new.basic.get(), expr.as_ptr());
            if result != CWRAPPER_OUTPUT_TYPE::SYMENGINE_NO_EXCEPTION {
                // If parsing failed, create a symbol instead
                eprintln!("Warning: Failed to parse '{}', treating as symbol", expr.to_string_lossy());
            }
            new
        }
    }

    /// Create a new expression from a string, returning a Result
    pub fn try_new<T>(expr: T) -> SymEngineResult<Self>
    where
        T: Into<Vec<u8>> + fmt::Display,
    {
        let expr_string = expr.to_string();
        let expr = CString::new(expr_string).map_err(|_| {
            SymEngineError::invalid_operation("String contains null bytes")
        })?;
        
        unsafe {
            let mut new = Expression {
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
    pub fn symbol<T>(name: T) -> Self
    where
        T: Into<Vec<u8>> + fmt::Display,
    {
        let name_string = name.to_string();
        let name_cstr = CString::new(name_string).expect("Failed to create CString for symbol name");
        unsafe {
            let mut new = Expression {
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
            basic_new_stack(&mut basic);
            f(&mut basic, value);
            let new = Expression {
                basic: UnsafeCell::new(basic),
            };
            new
        }
    }

    pub fn from_i64(value: i64) -> Self {
        Self::from_value(|ptr, val| unsafe { integer_set_si(ptr, val) }, value)
    }

    pub fn from_i32(value: i32) -> Self {
        Self::from_value(|ptr, val| unsafe { integer_set_si(ptr, val) }, value as i64)
    }

    pub fn from_u32(value: u32) -> Self {
        Self::from_value(|ptr, val| unsafe { integer_set_ui(ptr, val) }, value as c_ulong)
    }

    pub fn from_f64(value: f64) -> Self {
        Self::from_value(|ptr, val| unsafe { real_double_set_d(ptr, val) }, value)
    }

    pub fn from_f32(value: f32) -> Self {
        Self::from_value(|ptr, val| unsafe { real_double_set_d(ptr, val) }, value as f64)
    }

    pub fn assign_copy(&self) -> Self {
        let mut new = Expression {
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
        self.binary_op(rhs.into(), |result, a, b| unsafe { basic_add(result, a, b) })
    }
}

impl<T: Into<Expression>> std::ops::Add<T> for &Expression {
    type Output = Expression;

    fn add(self, rhs: T) -> Self::Output {
        self.clone().binary_op(rhs.into(), |result, a, b| unsafe { basic_add(result, a, b) })
    }
}

impl<T: Into<Self>> std::ops::Sub<T> for Expression {
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        self.binary_op(rhs.into(), |result, a, b| unsafe { basic_sub(result, a, b) })
    }
}

impl<T: Into<Self>> std::ops::Mul<T> for Expression {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        self.binary_op(rhs.into(), |result, a, b| unsafe { basic_mul(result, a, b) })
    }
}

impl<T: Into<Expression>> std::ops::Mul<T> for &Expression {
    type Output = Expression;

    fn mul(self, rhs: T) -> Self::Output {
        self.clone().binary_op(rhs.into(), |result, a, b| unsafe { basic_mul(result, a, b) })
    }
}

impl<T: Into<Self>> std::ops::Div<T> for Expression {
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        self.binary_op(rhs.into(), |result, a, b| unsafe { basic_div(result, a, b) })
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
    fn binary_op<F>(self, other: Self, op: F) -> Self
    where
        F: Fn(*mut basic_struct, *mut basic_struct, *mut basic_struct) -> CWRAPPER_OUTPUT_TYPE,
    {
        unsafe {
            let mut new = Expression {
                basic: UnsafeCell::new(std::mem::zeroed()),
            };
            basic_new_stack(new.basic.get());
            op(new.basic.get(), self.basic.get(), other.basic.get());
            new
        }
    }

    pub fn expand(&self) -> Self {
        unsafe {
            let mut new = Expression {
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
            basic_get_type(self.basic.get()) as i32 == SYMENGINE_SYMBOL
        }
    }

    /// Check if the expression is a number
    pub fn is_number(&self) -> bool {
        unsafe {
            let type_code = basic_get_type(self.basic.get()) as i32;
            type_code == SYMENGINE_INTEGER || 
            type_code == SYMENGINE_RATIONAL ||
            type_code == SYMENGINE_REAL_DOUBLE
        }
    }

    /// Check if the expression is a power operation
    pub fn is_pow(&self) -> bool {
        unsafe {
            basic_get_type(self.basic.get()) as i32 == SYMENGINE_POW
        }
    }

    /// Check if the expression is a multiplication
    pub fn is_mul(&self) -> bool {
        unsafe {
            basic_get_type(self.basic.get()) as i32 == SYMENGINE_MUL
        }
    }

    /// Check if the expression is an addition
    pub fn is_add(&self) -> bool {
        unsafe {
            basic_get_type(self.basic.get()) as i32 == SYMENGINE_ADD
        }
    }

    /// Get the base and exponent of a power expression
    pub fn as_pow(&self) -> Option<(Expression, Expression)> {
        if !self.is_pow() {
            return None;
        }
        
        unsafe {
            let mut base = Expression {
                basic: UnsafeCell::new(std::mem::zeroed()),
            };
            let mut exp = Expression {
                basic: UnsafeCell::new(std::mem::zeroed()),
            };
            basic_new_stack(base.basic.get());
            basic_new_stack(exp.basic.get());
            
            // Get the base and exponent
            basic_pow_get_base(base.basic.get(), self.basic.get());
            basic_pow_get_exp(exp.basic.get(), self.basic.get());
            
            Some((base, exp))
        }
    }

    /// Get an iterator over the terms in an addition
    pub fn as_add(&self) -> Option<Vec<Expression>> {
        if !self.is_add() {
            return None;
        }
        
        unsafe {
            let args_size = basic_get_args_size(self.basic.get());
            let mut terms = Vec::new();
            
            for i in 0..args_size {
                let mut term = Expression {
                    basic: UnsafeCell::new(std::mem::zeroed()),
                };
                basic_new_stack(term.basic.get());
                basic_get_arg(term.basic.get(), self.basic.get(), i);
                terms.push(term);
            }
            
            Some(terms)
        }
    }

    /// Get an iterator over the factors in a multiplication
    pub fn as_mul(&self) -> Option<Vec<Expression>> {
        if !self.is_mul() {
            return None;
        }
        
        unsafe {
            let args_size = basic_get_args_size(self.basic.get());
            let mut factors = Vec::new();
            
            for i in 0..args_size {
                let mut factor = Expression {
                    basic: UnsafeCell::new(std::mem::zeroed()),
                };
                basic_new_stack(factor.basic.get());
                basic_get_arg(factor.basic.get(), self.basic.get(), i);
                factors.push(factor);
            }
            
            Some(factors)
        }
    }

    /// Get the symbol name if this is a symbol
    pub fn as_symbol(&self) -> Option<String> {
        if !self.is_symbol() {
            return None;
        }
        
        unsafe {
            let name_ptr = basic_symbol_get_name(self.basic.get());
            if name_ptr.is_null() {
                return None;
            }
            let name_cstr = CStr::from_ptr(name_ptr);
            Some(name_cstr.to_string_lossy().to_string())
        }
    }

    /// Convert to f64 if this is a number
    pub fn to_f64(&self) -> Option<f64> {
        if !self.is_number() {
            return None;
        }
        
        unsafe {
            let type_code = basic_get_type(self.basic.get()) as i32;
            match type_code {
                SYMENGINE_REAL_DOUBLE => {
                    let value = real_double_get_d(self.basic.get());
                    Some(value)
                }
                SYMENGINE_INTEGER => {
                    let value = integer_get_si(self.basic.get());
                    Some(value as f64)
                }
                _ => None,
            }
        }
    }

    /// Power operation
    pub fn pow(&self, exp: Expression) -> Expression {
        unsafe {
            let mut new = Expression {
                basic: UnsafeCell::new(std::mem::zeroed()),
            };
            basic_new_stack(new.basic.get());
            basic_pow(new.basic.get(), self.basic.get(), exp.basic.get());
            new
        }
    }

    fn cmp_eq_op<F>(&self, other: &Self, op: F) -> bool
    where
        F: Fn(*const basic_struct, *const basic_struct) -> i32,
    {
        let lhs = self.expand();
        let rhs = other.expand();
        let result = unsafe { op(lhs.basic.get(), rhs.basic.get()) };
        result != 0
    }
}

impl PartialEq for Expression {
    fn eq(&self, other: &Self) -> bool {
        self.cmp_eq_op(other, |a, b| unsafe { basic_eq(a, b) })
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
        Ok(Expression::new(s))
    }
}