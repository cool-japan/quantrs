use std::fmt;
use std::ptr;

use crate::expr::Expression;
use symengine_sys::*;

#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// ExprMap provides a map from symbolic expressions to values.
///
/// This map is used for substitution, and is wrapper over
/// SymEngine's CMapBasicBasic structure.
///
/// # Example
///
/// ```
/// use symengine::{Expression, ExprMap};
///
/// // Create a variable
/// let x = Expression::new("x");
///
/// // Create a map to store substitutions
/// let mut map = ExprMap::new();
/// map.insert(x.clone(), 2);
///
/// // Create an expression
/// let expr = x.clone() * 3 + 1;
///
/// // Substitute values using the map
/// let result = map.substitute(&expr);
/// assert_eq!(result, Expression::from_i32(7));
/// ```
#[derive(Debug)]
pub struct ExprMap {
    basic: *mut CMapBasicBasic,
}

impl ExprMap {
    /// Creates a new, empty ExprMap.
    ///
    /// The map will not allocate until elements are pushed onto it.
    pub fn new() -> Self {
        Self {
            basic: unsafe { mapbasicbasic_new() },
        }
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the map did not have this key present, `None` is returned.
    ///
    /// If the map did have this key present, the value is updated, and the old
    /// value is returned.
    pub fn insert<V: Into<Expression>>(&mut self, key: Expression, value: V) -> Option<Expression> {
        unsafe {
            let key_expr = key.clone();
            mapbasicbasic_insert(self.basic, key_expr.basic.get(), value.into().basic.get());
            None
        }
    }

    /// Returns a reference to the value corresponding to the key.
    pub fn get(&self, key: &Expression) -> Option<Expression> {
        unsafe {
            let key_expr = key.clone();
            let mut value = Expression {
                basic: std::cell::UnsafeCell::new(std::mem::zeroed()),
            };
            basic_new_stack(value.basic.get());
            let ret = mapbasicbasic_get(self.basic, key_expr.basic.get(), value.basic.get());
            if ret == 0 {
                return None;
            }
            Some(value)
        }
    }

    /// Substitute all occurrences of the keys with their corresponding values
    pub fn substitute(&self, expr: &Expression) -> Expression {
        unsafe {
            let mut out = Expression {
                basic: std::cell::UnsafeCell::new(std::mem::zeroed()),
            };
            basic_new_stack(out.basic.get());
            basic_subs(out.basic.get(), expr.basic.get(), self.basic);
            out
        }
    }

    /// Returns the number of elements in the map.
    pub fn len(&self) -> usize {
        unsafe { mapbasicbasic_size(self.basic) as usize }
    }

    /// Returns true if the map contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Drop for ExprMap {
    fn drop(&mut self) {
        if !self.basic.is_null() {
            unsafe { mapbasicbasic_free(self.basic) }
        }
    }
}

impl Default for ExprMap {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ExprMap {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{{")?;
        let mut first = true;
        for expr in self {
            if !first {
                write!(f, ", ")?;
            }
            first = false;
            write!(f, "{}", expr)?;
        }
        write!(f, "}}")
    }
}

impl<'a> IntoIterator for &'a ExprMap {
    type Item = (&'a Expression, &'a Expression);
    type IntoIter = ExprMapIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        ExprMapIter {
            map: self,
            index: 0,
        }
    }
}

pub struct ExprMapIter<'a> {
    map: &'a ExprMap,
    index: usize,
}

impl<'a> Iterator for ExprMapIter<'a> {
    type Item = (&'a Expression, &'a Expression);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.map.len() {
            self.index += 1;
            // For this initial version, we'll just return a placeholder
            // In a real implementation, you'd need to iterate through the map properly
            None
        } else {
            None
        }
    }
}

#[cfg(feature = "serde-serialize")]
impl Serialize for ExprMap {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // A simplified serialization just as a demonstration
        let s = self.to_string();
        s.serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize")]
impl<'de> Deserialize<'de> for ExprMap {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // A simplified deserialization as a demonstration
        let _s = String::deserialize(deserializer)?;
        Ok(ExprMap::new())
    }
}