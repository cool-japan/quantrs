//! Symbolic representation of variables for quantum annealing problems.
//!
//! This module provides utilities for creating and manipulating symbolic
//! variables for QUBO/HOBO problem formulation.

#[cfg(feature = "dwave")]
use ndarray::Array;
use std::fmt::Write;
#[cfg(feature = "dwave")]
use symengine::Expression as SymEngineExpression;
use thiserror::Error;

/// Errors that can occur when working with symbols
#[derive(Error, Debug)]
pub enum SymbolError {
    /// Error when the format string doesn't match required dimensions
    #[error("Format string must have same number of placeholders as dimensions")]
    FormatMismatch,

    /// Error when placeholder format is invalid
    #[error("Format string must separate placeholders, got {0}")]
    InvalidPlaceholders(String),

    /// Error when dimensions exceed supported limits
    #[error("Currently only up to 5 dimensions are supported")]
    TooManyDimensions,

    /// Error when creating a symbol
    #[error("Failed to create symbol: {0}")]
    SymbolCreationError(String),
}

/// Result type for symbol operations
pub type SymbolResult<T> = Result<T, SymbolError>;

/// Create symbols from a string
///
/// This is a wrapper around Symengine's symbol creation function.
/// It allows creating single or multiple variables at once.
///
/// # Examples
///
/// ```
/// use quantrs_tytan::symbols;
///
/// // Create a single symbol
/// let x = symbols("x");
///
/// // Create multiple symbols
/// let (x, y, z) = symbols("x y z");
/// ```
#[cfg(feature = "dwave")]
pub fn symbols<T: AsRef<str>>(name: T) -> SymEngineExpression {
    SymEngineExpression::symbol(name.as_ref())
}

/// Placeholder for when dwave feature is not enabled
#[cfg(not(feature = "dwave"))]
#[doc(hidden)]
/// # Panics
///
/// This function always panics when the dwave feature is not enabled.
pub fn symbols<T: AsRef<str>>(_name: T) {
    panic!("The dwave feature is required to use symbolic functionality")
}

/// Create a multi-dimensional array of symbols
///
/// This function creates an n-dimensional array of symbols with
/// placeholders in the name that correspond to indices.
///
/// # Arguments
///
/// * `shape` - The shape of the array (can be a single integer or a slice)
/// * `format_txt` - The format string with {} placeholders for indices
///
/// # Examples
///
/// ```
/// use quantrs_tytan::symbols_list;
///
/// // Create a 3x3 grid of symbols
/// let q = symbols_list([3, 3], "q{}_{}");
/// ```
#[cfg(feature = "dwave")]
pub fn symbols_list<T>(
    shape: T,
    format_txt: &str,
) -> SymbolResult<Array<SymEngineExpression, ndarray::IxDyn>>
where
    T: Into<Vec<usize>>,
{
    let shape = shape.into();

    // Validate shape and format
    let dim = shape.len();
    if dim != format_txt.matches("{}").count() {
        return Err(SymbolError::FormatMismatch);
    }

    if format_txt.contains("}{") {
        return Err(SymbolError::InvalidPlaceholders(format_txt.to_string()));
    }

    if dim > 5 {
        return Err(SymbolError::TooManyDimensions);
    }

    // Create the array
    let shape_dim = ndarray::IxDyn(&shape);
    let mut array = Array::from_elem(shape_dim, SymEngineExpression::from_i64(0));

    // Fill the array with symbols
    let mut indices = vec![0; dim];
    fill_symbol_array(&mut array, &mut indices, 0, dim, format_txt)?;

    Ok(array)
}

// Helper function to recursively fill the symbol array
#[cfg(feature = "dwave")]
fn fill_symbol_array(
    array: &mut Array<SymEngineExpression, ndarray::IxDyn>,
    indices: &mut Vec<usize>,
    level: usize,
    max_level: usize,
    format_txt: &str,
) -> SymbolResult<()> {
    if level == max_level {
        // Create the format arguments
        let format_args: Vec<String> = indices.iter().map(ToString::to_string).collect();

        // Format the symbol name
        let mut symbol_name = format_txt.to_string();
        for arg in &format_args {
            symbol_name = symbol_name.replacen("{}", arg, 1);
        }

        // Create the symbol and store it in the array
        let sym = SymEngineExpression::symbol(symbol_name.clone());
        let idx = ndarray::IxDyn(indices);
        array[idx] = sym;

        Ok(())
    } else {
        let dim = array.shape()[level];
        for i in 0..dim {
            indices[level] = i;
            fill_symbol_array(array, indices, level + 1, max_level, format_txt)?;
        }
        Ok(())
    }
}

/// Generate symbol definition commands for multi-dimensional symbols
///
/// This function generates Rust code as a string that defines
/// individual symbol variables for each element in a multi-dimensional grid.
///
/// # Arguments
///
/// * `shape` - The shape of the array (can be a single integer or a slice)
/// * `format_txt` - The format string with {} placeholders for indices
///
/// # Examples
///
/// ```
/// use quantrs_tytan::symbols_define;
///
/// // Generate commands for a 2x2 grid
/// let commands = symbols_define([2, 2], "q{}_{}");
/// println!("{}", commands);
/// // Output:
/// // q0_0 = symbols("q0_0")
/// // q0_1 = symbols("q0_1")
/// // q1_0 = symbols("q1_0")
/// // q1_1 = symbols("q1_1")
/// ```
#[cfg(feature = "dwave")]
pub fn symbols_define<T>(shape: T, format_txt: &str) -> SymbolResult<String>
where
    T: Into<Vec<usize>>,
{
    let shape = shape.into();

    // Validate shape and format
    let dim = shape.len();
    if dim != format_txt.matches("{}").count() {
        return Err(SymbolError::FormatMismatch);
    }

    if format_txt.contains("}{") {
        return Err(SymbolError::InvalidPlaceholders(format_txt.to_string()));
    }

    if dim > 5 {
        return Err(SymbolError::TooManyDimensions);
    }

    // Generate the commands
    let mut commands = String::new();
    let mut indices = vec![0; dim];
    generate_symbol_commands(&mut commands, &mut indices, 0, &shape, format_txt)?;

    Ok(commands)
}

// Helper function to recursively generate symbol commands
#[allow(dead_code)]
fn generate_symbol_commands(
    commands: &mut String,
    indices: &mut Vec<usize>,
    level: usize,
    shape: &[usize],
    format_txt: &str,
) -> SymbolResult<()> {
    if level == shape.len() {
        // Create the format arguments
        let format_args: Vec<String> = indices.iter().map(ToString::to_string).collect();

        // Format the symbol name
        let mut symbol_name = format_txt.to_string();
        for arg in &format_args {
            symbol_name = symbol_name.replacen("{}", arg, 1);
        }

        // Add the command
        writeln!(commands, "{symbol_name} = symbols(\"{symbol_name}\");")
            .expect("Writing to string should not fail");

        Ok(())
    } else {
        for i in 0..shape[level] {
            indices[level] = i;
            generate_symbol_commands(commands, indices, level + 1, shape, format_txt)?;
        }
        Ok(())
    }
}

/// Create a binary-encoded integer variable using n bits
///
/// This function creates a symbolic expression representing
/// an integer variable encoded using n binary variables.
///
/// # Arguments
///
/// * `start` - The minimum value of the range
/// * `stop` - The maximum value of the range
/// * `format_txt` - The format string with a {} placeholder for bit index
/// * `num` - The number of bits to use
///
/// # Examples
///
/// ```
/// use quantrs_tytan::symbols_nbit;
///
/// // Create an 8-bit variable representing values from 0 to 255
/// let x = symbols_nbit(0, 256, "x{}", 8);
/// ```
#[cfg(feature = "dwave")]
pub fn symbols_nbit(
    start: u64,
    stop: u64,
    format_txt: &str,
    num: usize,
) -> SymbolResult<SymEngineExpression> {
    // Validate format
    if format_txt.matches("{}").count() != 1 {
        return Err(SymbolError::FormatMismatch);
    }

    // Create bit variables
    let mut result = SymEngineExpression::from_i64(0);
    let range = (stop - start) as f64;

    for n in 0..num {
        // Format the bit variable name
        let bit_name = format_txt.replacen("{}", &n.to_string(), 1);

        // Create the symbolic weight * bit
        let bit = SymEngineExpression::symbol(bit_name);
        let weight = range * 2.0_f64.powi((num as i32) - 1 - (n as i32)) / 2.0_f64.powi(num as i32);

        // Add to the result
        result = result + (SymEngineExpression::from_f64(weight) * bit);
    }

    // Add the start offset
    if start > 0 {
        result = result + SymEngineExpression::from_i64(start as i64);
    }

    Ok(result)
}

/// Type alias for SymEngine Expression when dwave feature is enabled
#[cfg(feature = "dwave")]
pub type Symbol = SymEngineExpression;

/// Type alias for SymEngine Expression when dwave feature is enabled
#[cfg(feature = "dwave")]
pub type Expression = SymEngineExpression;

/// Placeholder Symbol type when dwave feature is not enabled
#[cfg(not(feature = "dwave"))]
#[derive(Debug, Clone)]
pub struct Symbol {
    name: String,
}

#[cfg(not(feature = "dwave"))]
impl Symbol {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }
}

/// Placeholder Expression type when dwave feature is not enabled
#[cfg(not(feature = "dwave"))]
#[derive(Debug, Clone)]
pub struct Expression {
    value: String,
}

#[cfg(not(feature = "dwave"))]
impl Expression {
    pub fn new(value: &str) -> Self {
        Self {
            value: value.to_string(),
        }
    }
}
