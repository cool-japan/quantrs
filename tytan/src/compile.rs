//! Compilation of symbolic expressions to QUBO/HOBO models.
//!
//! This module provides utilities for compiling symbolic expressions
//! into QUBO (Quadratic Unconstrained Binary Optimization) and
//! HOBO (Higher-Order Binary Optimization) models.

use std::collections::{HashMap, HashSet};
use ndarray::Array;
use symengine::{self, Symbol as SymengineSymbol, Expr};
use thiserror::Error;

use quantrs_anneal::{QuboModel, QuboError, QuboResult};

/// Errors that can occur during compilation
#[derive(Error, Debug)]
pub enum CompileError {
    /// Error when the expression is invalid
    #[error("Invalid expression: {0}")]
    InvalidExpression(String),
    
    /// Error when a term has too high a degree
    #[error("Term has degree {0}, but maximum supported is {1}")]
    DegreeTooHigh(usize, usize),
    
    /// Error in the underlying QUBO model
    #[error("QUBO error: {0}")]
    QuboError(#[from] QuboError),
    
    /// Error in Symengine operations
    #[error("Symengine error: {0}")]
    SymengineError(String),
}

/// Result type for compilation operations
pub type CompileResult<T> = Result<T, CompileError>;

/// Compiler for converting symbolic expressions to QUBO models
///
/// This struct provides methods for converting symbolic expressions
/// to QUBO models, which can then be solved using quantum annealing.
pub struct Compile {
    /// The symbolic expression to compile
    expr: Expr,
}

impl Compile {
    /// Create a new compiler with the given expression
    pub fn new<T: Into<Expr>>(expr: T) -> Self {
        Self {
            expr: expr.into(),
        }
    }
    
    /// Compile the expression to a QUBO model
    ///
    /// This method compiles the symbolic expression to a QUBO model,
    /// which can then be passed to a sampler for solving.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - A tuple with the QUBO matrix and a mapping of variable names to indices
    /// - An offset value that should be added to all energy values
    pub fn get_qubo(&self) -> CompileResult<((Array<f64, ndarray::Ix2>, HashMap<String, usize>), f64)> {
        // Expand the expression to simplify
        let expr = symengine::expand(&self.expr);
        
        // Check the degree of each term
        let max_degree = calc_highest_degree(&expr)?;
        if max_degree > 2 {
            return Err(CompileError::DegreeTooHigh(max_degree, 2));
        }
        
        // Replace all second-degree terms (x^2) with x, since x^2 = x for binary variables
        let expr = replace_squared_terms(&expr)?;
        
        // Expand again to collect like terms
        let expr = symengine::expand(&expr);
        
        // Extract the coefficients and variables
        let (coeffs, offset) = extract_coefficients(&expr)?;
        
        // Convert to a QUBO matrix
        let (matrix, var_map) = build_qubo_matrix(&coeffs)?;
        
        Ok(((matrix, var_map), offset))
    }
    
    /// Compile the expression to a HOBO model
    ///
    /// This method compiles the symbolic expression to a Higher-Order Binary Optimization model,
    /// which can handle terms of degree higher than 2.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - A tuple with the HOBO tensor and a mapping of variable names to indices
    /// - An offset value that should be added to all energy values
    pub fn get_hobo(&self) -> CompileResult<((Array<f64, ndarray::IxDyn>, HashMap<String, usize>), f64)> {
        // Expand the expression to simplify
        let expr = symengine::expand(&self.expr);
        
        // Calculate highest degree (dimension of the tensor)
        let max_degree = calc_highest_degree(&expr)?;
        
        // Replace all squared terms (x^2) with x, since x^2 = x for binary variables
        let expr = replace_squared_terms(&expr)?;
        
        // Expand again to collect like terms
        let expr = symengine::expand(&expr);
        
        // Extract the coefficients and variables
        let (coeffs, offset) = extract_coefficients(&expr)?;
        
        // Build the HOBO tensor
        let (tensor, var_map) = build_hobo_tensor(&coeffs, max_degree)?;
        
        Ok(((tensor, var_map), offset))
    }
}

// Helper function to calculate the highest degree in the expression
fn calc_highest_degree(expr: &Expr) -> CompileResult<usize> {
    // If the expression is a single variable, it's degree 1
    if expr.is_symbol() {
        return Ok(1);
    }

    // If it's a number constant, degree is 0
    if expr.is_number() {
        return Ok(0);
    }

    // If it's a power operation (like x^2)
    if expr.is_pow() {
        let (base, exp) = expr.as_pow().unwrap();

        // If the base is a symbol and exponent is a number
        if base.is_symbol() && exp.is_number() {
            let exp_val = match exp.to_f64() {
                Some(n) => n,
                None => return Err(CompileError::InvalidExpression("Invalid exponent".to_string())),
            };

            // Check if exponent is a positive integer
            if exp_val.is_sign_positive() && exp_val.fract() == 0.0 {
                return Ok(exp_val as usize);
            }
        }

        // For other power expressions, recursively calculate the degree
        let base_degree = calc_highest_degree(&base)?;
        let exp_degree = if exp.is_number() {
            match exp.to_f64() {
                Some(n) => {
                    if n.is_sign_positive() && n.fract() == 0.0 {
                        n as usize
                    } else {
                        0 // Non-integer or negative exponents don't contribute to degree
                    }
                },
                None => 0,
            }
        } else {
            0 // Non-constant exponents don't contribute to degree
        };

        return Ok(base_degree * exp_degree);
    }

    // If it's a product (like x*y or x*x)
    if expr.is_mul() {
        let mut total_degree = 0;
        for factor in expr.as_mul().unwrap() {
            total_degree += calc_highest_degree(&factor)?;
        }
        return Ok(total_degree);
    }

    // If it's a sum (like x + y)
    if expr.is_add() {
        let mut max_degree = 0;
        for term in expr.as_add().unwrap() {
            let term_degree = calc_highest_degree(&term)?;
            max_degree = std::cmp::max(max_degree, term_degree);
        }
        return Ok(max_degree);
    }

    // Default case - for simplicity, we'll say degree is 0
    // but for a complete implementation, we'd need to handle all cases
    Err(CompileError::InvalidExpression(format!("Can't determine degree of: {}", expr)))
}

// Helper function to replace squared terms with linear terms
fn replace_squared_terms(expr: &Expr) -> CompileResult<Expr> {
    // For binary variables, x^2 = x since x ∈ {0,1}

    // If the expression is a symbol or number, just return it
    if expr.is_symbol() || expr.is_number() {
        return Ok(expr.clone());
    }

    // If it's a power operation (like x^2)
    if expr.is_pow() {
        let (base, exp) = expr.as_pow().unwrap();

        // If the base is a symbol and exponent is 2, replace with base
        if base.is_symbol() && exp.is_number() {
            let exp_val = match exp.to_f64() {
                Some(n) => n,
                None => return Err(CompileError::InvalidExpression("Invalid exponent".to_string())),
            };

            // Check if exponent is 2 (for higher exponents we'd need to recurse)
            if exp_val == 2.0 {
                return Ok(base);
            }
        }

        // For other power expressions, recursively replace
        let new_base = replace_squared_terms(&base)?;
        return Ok(new_base.pow(exp));
    }

    // If it's a product (like x*y or x*x)
    if expr.is_mul() {
        let mut new_terms = Vec::new();
        for factor in expr.as_mul().unwrap() {
            new_terms.push(replace_squared_terms(&factor)?);
        }

        // Combine the terms back into a product
        let mut result = Expr::from(1);
        for term in new_terms {
            result = result * term;
        }
        return Ok(result);
    }

    // If it's a sum (like x + y)
    if expr.is_add() {
        let mut new_terms = Vec::new();
        for term in expr.as_add().unwrap() {
            new_terms.push(replace_squared_terms(&term)?);
        }

        // Combine the terms back into a sum
        let mut result = Expr::from(0);
        for term in new_terms {
            result = result + term;
        }
        return Ok(result);
    }

    // For any other type of expression, just return it unchanged
    Ok(expr.clone())
}

// Helper function to extract coefficients and variables from the expression
fn extract_coefficients(expr: &Expr) -> CompileResult<(HashMap<Vec<String>, f64>, f64)> {
    let mut coeffs = HashMap::new();
    let mut offset = 0.0;

    // Process expression as a sum of terms
    if expr.is_add() {
        for term in expr.as_add().unwrap() {
            let (term_coeffs, term_offset) = extract_term_coefficients(&term)?;

            // Merge coefficients
            for (vars, coeff) in term_coeffs {
                *coeffs.entry(vars).or_insert(0.0) += coeff;
            }

            // Add constant terms to offset
            offset += term_offset;
        }
    } else {
        // Process a single term
        let (term_coeffs, term_offset) = extract_term_coefficients(expr)?;

        // Merge coefficients
        for (vars, coeff) in term_coeffs {
            *coeffs.entry(vars).or_insert(0.0) += coeff;
        }

        // Add constant terms to offset
        offset += term_offset;
    }

    Ok((coeffs, offset))
}

// Helper function to extract coefficient and variables from a single term
fn extract_term_coefficients(term: &Expr) -> CompileResult<(HashMap<Vec<String>, f64>, f64)> {
    let mut coeffs = HashMap::new();

    // If it's a number constant, it's an offset
    if term.is_number() {
        let value = match term.to_f64() {
            Some(n) => n,
            None => return Err(CompileError::InvalidExpression("Invalid number".to_string())),
        };
        return Ok((coeffs, value));
    }

    // If it's a symbol, it's a linear term with coefficient 1
    if term.is_symbol() {
        let var_name = term.as_symbol().unwrap().to_string();
        let vars = vec![var_name];
        coeffs.insert(vars, 1.0);
        return Ok((coeffs, 0.0));
    }

    // If it's a product of terms
    if term.is_mul() {
        let mut coeff = 1.0;
        let mut vars = Vec::new();

        for factor in term.as_mul().unwrap() {
            if factor.is_number() {
                // Numerical factor is a coefficient
                let value = match factor.to_f64() {
                    Some(n) => n,
                    None => return Err(CompileError::InvalidExpression("Invalid number in product".to_string())),
                };
                coeff *= value;
            } else if factor.is_symbol() {
                // Symbol is a variable
                let var_name = factor.as_symbol().unwrap().to_string();
                vars.push(var_name);
            } else {
                // More complex factors not supported in this example
                return Err(CompileError::InvalidExpression(
                    format!("Unsupported term in product: {}", factor)
                ));
            }
        }

        // Sort variables for consistent ordering
        vars.sort();

        if !vars.is_empty() {
            coeffs.insert(vars, coeff);
        } else {
            // If there are no variables, it's a constant term
            return Ok((coeffs, coeff));
        }

        return Ok((coeffs, 0.0));
    }

    // If it's a power operation (like x^2), should have been simplified earlier
    if term.is_pow() {
        return Err(CompileError::InvalidExpression(
            format!("Unexpected power term after simplification: {}", term)
        ));
    }

    // Unsupported term type
    Err(CompileError::InvalidExpression(format!("Unsupported term: {}", term)))
}

// Helper function to build the QUBO matrix
fn build_qubo_matrix(coeffs: &HashMap<Vec<String>, f64>) -> CompileResult<(Array<f64, ndarray::Ix2>, HashMap<String, usize>)> {
    // Collect all unique variable names
    let mut all_vars = HashSet::new();
    for vars in coeffs.keys() {
        for var in vars {
            all_vars.insert(var.clone());
        }
    }

    // Convert to a sorted vector
    let mut sorted_vars: Vec<String> = all_vars.into_iter().collect();
    sorted_vars.sort();

    // Create the variable-to-index mapping
    let var_map: HashMap<String, usize> = sorted_vars
        .iter()
        .enumerate()
        .map(|(i, var)| (var.clone(), i))
        .collect();

    // Size of the matrix
    let n = var_map.len();

    // Create an empty matrix
    let mut matrix = Array::zeros((n, n));

    // Fill the matrix with coefficients
    for (vars, &coeff) in coeffs {
        match vars.len() {
            0 => {
                // Should never happen since constants are handled in offset
                continue;
            },
            1 => {
                // Linear term: var * coeff
                let i = *var_map.get(&vars[0]).unwrap();
                matrix[[i, i]] += coeff;
            },
            2 => {
                // Quadratic term: var1 * var2 * coeff
                let i = *var_map.get(&vars[0]).unwrap();
                let j = *var_map.get(&vars[1]).unwrap();

                // QUBO format requires i <= j
                if i == j {
                    // Diagonal term
                    matrix[[i, i]] += coeff;
                } else {
                    // Off-diagonal term
                    matrix[[i, j]] += coeff;
                    matrix[[j, i]] += coeff; // Make sure matrix is symmetric
                }
            },
            _ => {
                // Higher-order terms are not supported in QUBO
                return Err(CompileError::DegreeTooHigh(vars.len(), 2));
            }
        }
    }

    Ok((matrix, var_map))
}

// Helper function to build the HOBO tensor
fn build_hobo_tensor(coeffs: &HashMap<Vec<String>, f64>, max_degree: usize) -> CompileResult<(Array<f64, ndarray::IxDyn>, HashMap<String, usize>)> {
    // Collect all unique variable names
    let mut all_vars = HashSet::new();
    for vars in coeffs.keys() {
        for var in vars {
            all_vars.insert(var.clone());
        }
    }

    // Convert to a sorted vector
    let mut sorted_vars: Vec<String> = all_vars.into_iter().collect();
    sorted_vars.sort();

    // Create the variable-to-index mapping
    let var_map: HashMap<String, usize> = sorted_vars
        .iter()
        .enumerate()
        .map(|(i, var)| (var.clone(), i))
        .collect();

    // Size of each dimension
    let n = var_map.len();

    // Create shape vector for the tensor
    let shape: Vec<usize> = vec![n; max_degree];

    // Create an empty tensor
    let mut tensor = Array::zeros(ndarray::IxDyn(&shape));

    // Fill the tensor with coefficients
    for (vars, &coeff) in coeffs {
        let degree = vars.len();

        if degree == 0 {
            // Should never happen since constants are handled in offset
            continue;
        }

        if degree > max_degree {
            return Err(CompileError::DegreeTooHigh(degree, max_degree));
        }

        // Convert variable names to indices
        let mut indices: Vec<usize> = vars.iter()
            .map(|var| *var_map.get(var).unwrap())
            .collect();

        // Sort indices (canonical ordering)
        indices.sort();

        // Pad indices to match tensor order if necessary
        while indices.len() < max_degree {
            indices.insert(0, indices[0]); // Padding with first index
        }

        // Set the coefficient in the tensor
        let idx = ndarray::IxDyn(&indices);
        tensor[idx] += coeff;
    }

    Ok((tensor, var_map))
}

/// Special compiler for problems with one-hot constraints
///
/// This is a specialized compiler that is optimized for problems
/// with one-hot constraints, common in many optimization problems.
pub struct PieckCompile {
    /// The symbolic expression to compile
    expr: Expr,
    /// Whether to show verbose output
    verbose: bool,
}

impl PieckCompile {
    /// Create a new Pieck compiler with the given expression
    pub fn new<T: Into<Expr>>(expr: T, verbose: bool) -> Self {
        Self {
            expr: expr.into(),
            verbose,
        }
    }
    
    /// Compile the expression to a QUBO model optimized for one-hot constraints
    pub fn get_qubo(&self) -> CompileResult<((Array<f64, ndarray::Ix2>, HashMap<String, usize>), f64)> {
        // Implementation will compile the expression using specialized techniques
        // For now, call the regular compiler
        Compile::new(&self.expr).get_qubo()
    }
}