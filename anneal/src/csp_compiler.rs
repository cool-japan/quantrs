//! Constraint Satisfaction Problem (CSP) compiler
//!
//! This module provides a compiler that translates constraint satisfaction problems
//! into QUBO formulations that can be solved using quantum annealing.

use std::collections::{HashMap, HashSet};
use std::fmt;
use thiserror::Error;

use crate::ising::QuboModel;
use crate::qubo::{QuboBuilder, QuboFormulation};

/// Errors that can occur during CSP compilation
#[derive(Error, Debug)]
pub enum CspError {
    /// Invalid variable definition
    #[error("Invalid variable: {0}")]
    InvalidVariable(String),

    /// Invalid constraint definition
    #[error("Invalid constraint: {0}")]
    InvalidConstraint(String),

    /// Compilation failed
    #[error("Compilation failed: {0}")]
    CompilationFailed(String),

    /// Unsupported constraint type
    #[error("Unsupported constraint type: {0}")]
    UnsupportedConstraint(String),

    /// Domain error
    #[error("Domain error: {0}")]
    DomainError(String),
}

/// Result type for CSP operations
pub type CspResult<T> = Result<T, CspError>;

/// Variable domain specification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Domain {
    /// Boolean domain {0, 1}
    Boolean,

    /// Integer range [min, max]
    IntegerRange { min: i32, max: i32 },

    /// Discrete set of values
    Discrete(Vec<i32>),

    /// Finite set of strings/labels
    Categorical(Vec<String>),
}

impl Domain {
    /// Get the size of the domain
    #[must_use]
    pub fn size(&self) -> usize {
        match self {
            Self::Boolean => 2,
            Self::IntegerRange { min, max } => ((max - min + 1) as usize).max(0),
            Self::Discrete(values) => values.len(),
            Self::Categorical(labels) => labels.len(),
        }
    }

    /// Check if a value is in the domain
    #[must_use]
    pub fn contains(&self, value: &CspValue) -> bool {
        match (self, value) {
            (Self::Boolean, CspValue::Boolean(_)) => true,
            (Self::IntegerRange { min, max }, CspValue::Integer(v)) => v >= min && v <= max,
            (Self::Discrete(values), CspValue::Integer(v)) => values.contains(v),
            (Self::Categorical(labels), CspValue::String(s)) => labels.contains(s),
            _ => false,
        }
    }

    /// Convert to list of values
    pub fn values(&self) -> Vec<CspValue> {
        match self {
            Self::Boolean => vec![CspValue::Boolean(false), CspValue::Boolean(true)],
            Self::IntegerRange { min, max } => (*min..=*max).map(CspValue::Integer).collect(),
            Self::Discrete(values) => values.iter().map(|&v| CspValue::Integer(v)).collect(),
            Self::Categorical(labels) => {
                labels.iter().map(|s| CspValue::String(s.clone())).collect()
            }
        }
    }
}

/// CSP variable value
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CspValue {
    Boolean(bool),
    Integer(i32),
    String(String),
}

impl fmt::Display for CspValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Boolean(b) => write!(f, "{b}"),
            Self::Integer(i) => write!(f, "{i}"),
            Self::String(s) => write!(f, "{s}"),
        }
    }
}

/// CSP variable definition
#[derive(Debug, Clone)]
pub struct CspVariable {
    /// Variable name
    pub name: String,

    /// Variable domain
    pub domain: Domain,

    /// Variable description
    pub description: Option<String>,
}

impl CspVariable {
    /// Create a new CSP variable
    #[must_use]
    pub const fn new(name: String, domain: Domain) -> Self {
        Self {
            name,
            domain,
            description: None,
        }
    }

    /// Add description to the variable
    #[must_use]
    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }
}

/// CSP constraint types
pub enum CspConstraint {
    /// All different constraint: all variables must have different values
    AllDifferent { variables: Vec<String> },

    /// Linear constraint: sum of weighted variables with comparison
    Linear {
        terms: Vec<(String, f64)>, // (variable, coefficient)
        comparison: ComparisonOp,
        rhs: f64,
    },

    /// Exactly one constraint: exactly one variable is true/active
    ExactlyOne { variables: Vec<String> },

    /// At most one constraint: at most one variable is true/active
    AtMostOne { variables: Vec<String> },

    /// Element constraint: var\\[index\\] = value
    Element {
        array_var: String,
        index_var: String,
        value_var: String,
    },

    /// Global cardinality constraint
    GlobalCardinality {
        variables: Vec<String>,
        values: Vec<CspValue>,
        min_counts: Vec<usize>,
        max_counts: Vec<usize>,
    },

    /// Table constraint: allowed/forbidden tuples
    Table {
        variables: Vec<String>,
        tuples: Vec<Vec<CspValue>>,
        allowed: bool, // true for allowed, false for forbidden
    },

    /// Custom constraint with penalty function
    Custom {
        name: String,
        variables: Vec<String>,
        penalty_function: Box<dyn Fn(&HashMap<String, CspValue>) -> f64 + Send + Sync>,
    },
}

impl fmt::Debug for CspConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AllDifferent { variables } => f
                .debug_struct("AllDifferent")
                .field("variables", variables)
                .finish(),
            Self::Linear {
                terms,
                comparison,
                rhs,
            } => f
                .debug_struct("Linear")
                .field("terms", terms)
                .field("comparison", comparison)
                .field("rhs", rhs)
                .finish(),
            Self::ExactlyOne { variables } => f
                .debug_struct("ExactlyOne")
                .field("variables", variables)
                .finish(),
            Self::AtMostOne { variables } => f
                .debug_struct("AtMostOne")
                .field("variables", variables)
                .finish(),
            Self::Element {
                array_var,
                index_var,
                value_var,
            } => f
                .debug_struct("Element")
                .field("array_var", array_var)
                .field("index_var", index_var)
                .field("value_var", value_var)
                .finish(),
            Self::GlobalCardinality {
                variables,
                values,
                min_counts,
                max_counts,
            } => f
                .debug_struct("GlobalCardinality")
                .field("variables", variables)
                .field("values", values)
                .field("min_counts", min_counts)
                .field("max_counts", max_counts)
                .finish(),
            Self::Table {
                variables,
                tuples,
                allowed,
            } => f
                .debug_struct("Table")
                .field("variables", variables)
                .field("tuples", tuples)
                .field("allowed", allowed)
                .finish(),
            Self::Custom {
                name, variables, ..
            } => f
                .debug_struct("Custom")
                .field("name", name)
                .field("variables", variables)
                .field("penalty_function", &"<function>")
                .finish(),
        }
    }
}

/// Comparison operators for linear constraints
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComparisonOp {
    Equal,
    LessEqual,
    GreaterEqual,
    Less,
    Greater,
}

/// CSP problem definition
#[derive(Debug)]
pub struct CspProblem {
    /// Variables in the problem
    variables: HashMap<String, CspVariable>,

    /// Constraints in the problem
    constraints: Vec<CspConstraint>,

    /// Objective function (optional)
    objective: Option<CspObjective>,

    /// Compilation parameters
    compilation_params: CompilationParams,
}

/// CSP objective function
pub enum CspObjective {
    /// Minimize/maximize linear combination of variables
    Linear {
        terms: Vec<(String, f64)>,
        minimize: bool,
    },

    /// Minimize/maximize custom function
    Custom {
        function: Box<dyn Fn(&HashMap<String, CspValue>) -> f64 + Send + Sync>,
        minimize: bool,
    },
}

impl fmt::Debug for CspObjective {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Linear { terms, minimize } => f
                .debug_struct("Linear")
                .field("terms", terms)
                .field("minimize", minimize)
                .finish(),
            Self::Custom { minimize, .. } => f
                .debug_struct("Custom")
                .field("function", &"<function>")
                .field("minimize", minimize)
                .finish(),
        }
    }
}

/// Parameters for CSP compilation
#[derive(Debug, Clone)]
pub struct CompilationParams {
    /// Penalty weight for constraint violations
    pub constraint_penalty: f64,

    /// Use logarithmic encoding for large domains
    pub use_log_encoding: bool,

    /// Maximum domain size for one-hot encoding
    pub max_onehot_size: usize,

    /// Slack variable penalty for inequality constraints
    pub slack_penalty: f64,
}

impl Default for CompilationParams {
    fn default() -> Self {
        Self {
            constraint_penalty: 10.0,
            use_log_encoding: true,
            max_onehot_size: 16,
            slack_penalty: 1.0,
        }
    }
}

/// CSP solution
#[derive(Debug, Clone)]
pub struct CspSolution {
    /// Variable assignments
    pub assignments: HashMap<String, CspValue>,

    /// Objective value (if any)
    pub objective_value: Option<f64>,

    /// Constraint violations
    pub violations: Vec<ConstraintViolation>,

    /// QUBO objective value
    pub qubo_objective: f64,
}

/// Constraint violation information
#[derive(Debug, Clone)]
pub struct ConstraintViolation {
    /// Constraint index
    pub constraint_index: usize,

    /// Violation description
    pub description: String,

    /// Penalty incurred
    pub penalty: f64,
}

impl CspProblem {
    /// Create a new CSP problem
    #[must_use]
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            constraints: Vec::new(),
            objective: None,
            compilation_params: CompilationParams::default(),
        }
    }

    /// Add a variable to the problem
    pub fn add_variable(&mut self, variable: CspVariable) -> CspResult<()> {
        if self.variables.contains_key(&variable.name) {
            return Err(CspError::InvalidVariable(format!(
                "Variable '{}' already exists",
                variable.name
            )));
        }

        self.variables.insert(variable.name.clone(), variable);
        Ok(())
    }

    /// Add a constraint to the problem
    pub fn add_constraint(&mut self, constraint: CspConstraint) -> CspResult<()> {
        // Validate constraint variables exist
        let var_names = self.get_constraint_variables(&constraint);
        for var_name in &var_names {
            if !self.variables.contains_key(var_name) {
                return Err(CspError::InvalidConstraint(format!(
                    "Unknown variable '{var_name}' in constraint"
                )));
            }
        }

        self.constraints.push(constraint);
        Ok(())
    }

    /// Set the objective function
    pub fn set_objective(&mut self, objective: CspObjective) -> CspResult<()> {
        // Validate objective variables exist
        let var_names = self.get_objective_variables(&objective);
        for var_name in &var_names {
            if !self.variables.contains_key(var_name) {
                return Err(CspError::InvalidConstraint(format!(
                    "Unknown variable '{var_name}' in objective"
                )));
            }
        }

        self.objective = Some(objective);
        Ok(())
    }

    /// Set compilation parameters
    pub const fn set_compilation_params(&mut self, params: CompilationParams) {
        self.compilation_params = params;
    }

    /// Compile the CSP to a QUBO formulation
    pub fn compile_to_qubo(&self) -> CspResult<(QuboModel, CspCompilationInfo)> {
        let mut builder = QuboBuilder::new();
        let mut info = CspCompilationInfo::new();

        // Step 1: Create binary variables for CSP variables
        let var_encoding = self.create_variable_encoding(&mut builder, &mut info)?;

        // Step 2: Add constraint penalties
        self.add_constraint_penalties(&mut builder, &var_encoding, &mut info)?;

        // Step 3: Add objective function
        if let Some(ref objective) = self.objective {
            self.add_objective_function(&mut builder, &var_encoding, objective, &mut info)?;
        }

        // Step 4: Set constraint penalty weight
        builder
            .set_constraint_weight(self.compilation_params.constraint_penalty)
            .map_err(|e| CspError::CompilationFailed(e.to_string()))?;

        Ok((builder.build(), info))
    }

    /// Create binary variable encoding for CSP variables
    fn create_variable_encoding(
        &self,
        builder: &mut QuboBuilder,
        info: &mut CspCompilationInfo,
    ) -> CspResult<HashMap<String, VariableEncoding>> {
        let mut encodings = HashMap::new();

        for (var_name, csp_var) in &self.variables {
            let encoding = if csp_var.domain == Domain::Boolean {
                let qubo_var = builder
                    .add_variable(var_name.clone())
                    .map_err(|e| CspError::CompilationFailed(e.to_string()))?;
                VariableEncoding::Direct(qubo_var)
            } else {
                let domain_size = csp_var.domain.size();

                if domain_size <= self.compilation_params.max_onehot_size
                    && !self.compilation_params.use_log_encoding
                {
                    // One-hot encoding
                    let mut qubo_vars = Vec::new();
                    for (i, value) in csp_var.domain.values().iter().enumerate() {
                        let var_name_encoded = format!("{var_name}_{i}");
                        let qubo_var = builder
                            .add_variable(var_name_encoded)
                            .map_err(|e| CspError::CompilationFailed(e.to_string()))?;
                        qubo_vars.push((qubo_var, value.clone()));
                    }

                    // Add exactly-one constraint for one-hot encoding
                    let vars_only: Vec<_> = qubo_vars.iter().map(|(var, _)| var.clone()).collect();
                    builder
                        .constrain_one_hot(&vars_only)
                        .map_err(|e| CspError::CompilationFailed(e.to_string()))?;

                    VariableEncoding::OneHot(qubo_vars)
                } else {
                    // Binary/logarithmic encoding
                    let num_bits = (domain_size as f64).log2().ceil() as usize;
                    let mut qubo_vars = Vec::new();

                    for i in 0..num_bits {
                        let var_name_bit = format!("{var_name}_bit_{i}");
                        let qubo_var = builder
                            .add_variable(var_name_bit)
                            .map_err(|e| CspError::CompilationFailed(e.to_string()))?;
                        qubo_vars.push(qubo_var);
                    }

                    VariableEncoding::Binary {
                        bits: qubo_vars,
                        domain_values: csp_var.domain.values(),
                    }
                }
            };

            encodings.insert(var_name.clone(), encoding);
            info.add_variable_info(var_name.clone(), csp_var.domain.size());
        }

        Ok(encodings)
    }

    /// Add constraint penalties to the QUBO
    fn add_constraint_penalties(
        &self,
        builder: &mut QuboBuilder,
        var_encoding: &HashMap<String, VariableEncoding>,
        info: &mut CspCompilationInfo,
    ) -> CspResult<()> {
        for (i, constraint) in self.constraints.iter().enumerate() {
            match constraint {
                CspConstraint::AllDifferent { variables } => {
                    self.add_all_different_constraint(builder, var_encoding, variables, info)?;
                }

                CspConstraint::Linear {
                    terms,
                    comparison,
                    rhs,
                } => {
                    self.add_linear_constraint(
                        builder,
                        var_encoding,
                        terms,
                        comparison,
                        *rhs,
                        info,
                    )?;
                }

                CspConstraint::ExactlyOne { variables } => {
                    self.add_exactly_one_constraint(builder, var_encoding, variables, info)?;
                }

                CspConstraint::AtMostOne { variables } => {
                    self.add_at_most_one_constraint(builder, var_encoding, variables, info)?;
                }

                _ => {
                    return Err(CspError::UnsupportedConstraint(format!(
                        "Constraint type not yet implemented: constraint {i}"
                    )));
                }
            }

            info.constraints_compiled += 1;
        }

        Ok(())
    }

    /// Add all-different constraint
    fn add_all_different_constraint(
        &self,
        builder: &mut QuboBuilder,
        var_encoding: &HashMap<String, VariableEncoding>,
        variables: &[String],
        _info: &mut CspCompilationInfo,
    ) -> CspResult<()> {
        // For each pair of variables, penalize if they have the same value
        for i in 0..variables.len() {
            for j in (i + 1)..variables.len() {
                let var1 = &variables[i];
                let var2 = &variables[j];

                self.add_not_equal_penalty(builder, var_encoding, var1, var2)?;
            }
        }

        Ok(())
    }

    /// Add penalty for two variables being equal
    fn add_not_equal_penalty(
        &self,
        builder: &mut QuboBuilder,
        var_encoding: &HashMap<String, VariableEncoding>,
        var1: &str,
        var2: &str,
    ) -> CspResult<()> {
        let encoding1 = &var_encoding[var1];
        let encoding2 = &var_encoding[var2];

        match (encoding1, encoding2) {
            (VariableEncoding::Direct(q1), VariableEncoding::Direct(q2)) => {
                // Penalty for both being 1: x1 * x2
                builder
                    .set_quadratic_term(q1, q2, self.compilation_params.constraint_penalty)
                    .map_err(|e| CspError::CompilationFailed(e.to_string()))?;
                // Penalty for both being 0: (1-x1) * (1-x2) = 1 - x1 - x2 + x1*x2
                // This becomes: 1 - x1 - x2 + 2*x1*x2 (since we already have x1*x2)
                builder
                    .set_linear_term(q1, -self.compilation_params.constraint_penalty)
                    .map_err(|e| CspError::CompilationFailed(e.to_string()))?;
                builder
                    .set_linear_term(q2, -self.compilation_params.constraint_penalty)
                    .map_err(|e| CspError::CompilationFailed(e.to_string()))?;
                builder
                    .set_quadratic_term(q1, q2, self.compilation_params.constraint_penalty)
                    .map_err(|e| CspError::CompilationFailed(e.to_string()))?;
            }

            (VariableEncoding::OneHot(vars1), VariableEncoding::OneHot(vars2)) => {
                // For each matching value, penalize if both variables have that value
                let values1: HashMap<_, _> = vars1.iter().map(|(var, val)| (val, var)).collect();
                let values2: HashMap<_, _> = vars2.iter().map(|(var, val)| (val, var)).collect();

                for (value, qubo_var1) in &values1 {
                    if let Some(qubo_var2) = values2.get(value) {
                        builder
                            .set_quadratic_term(
                                qubo_var1,
                                qubo_var2,
                                self.compilation_params.constraint_penalty,
                            )
                            .map_err(|e| CspError::CompilationFailed(e.to_string()))?;
                    }
                }
            }

            _ => {
                return Err(CspError::UnsupportedConstraint(
                    "All-different constraint with mixed encoding types not supported".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Add linear constraint
    fn add_linear_constraint(
        &self,
        _builder: &mut QuboBuilder,
        _var_encoding: &HashMap<String, VariableEncoding>,
        _terms: &[(String, f64)],
        _comparison: &ComparisonOp,
        _rhs: f64,
        _info: &mut CspCompilationInfo,
    ) -> CspResult<()> {
        // This would require implementing slack variables and penalty methods
        // For now, return unsupported
        Err(CspError::UnsupportedConstraint(
            "Linear constraints not yet implemented".to_string(),
        ))
    }

    /// Add exactly-one constraint
    fn add_exactly_one_constraint(
        &self,
        builder: &mut QuboBuilder,
        var_encoding: &HashMap<String, VariableEncoding>,
        variables: &[String],
        _info: &mut CspCompilationInfo,
    ) -> CspResult<()> {
        let mut qubo_vars = Vec::new();

        for var_name in variables {
            match &var_encoding[var_name] {
                VariableEncoding::Direct(qubo_var) => {
                    qubo_vars.push(qubo_var.clone());
                }
                _ => {
                    return Err(CspError::UnsupportedConstraint(
                        "Exactly-one constraint only supports boolean variables".to_string(),
                    ));
                }
            }
        }

        builder
            .constrain_one_hot(&qubo_vars)
            .map_err(|e| CspError::CompilationFailed(e.to_string()))?;

        Ok(())
    }

    /// Add at-most-one constraint
    fn add_at_most_one_constraint(
        &self,
        builder: &mut QuboBuilder,
        var_encoding: &HashMap<String, VariableEncoding>,
        variables: &[String],
        _info: &mut CspCompilationInfo,
    ) -> CspResult<()> {
        // At most one: penalize any pair being true simultaneously
        let mut qubo_vars = Vec::new();

        for var_name in variables {
            match &var_encoding[var_name] {
                VariableEncoding::Direct(qubo_var) => {
                    qubo_vars.push(qubo_var.clone());
                }
                _ => {
                    return Err(CspError::UnsupportedConstraint(
                        "At-most-one constraint only supports boolean variables".to_string(),
                    ));
                }
            }
        }

        // Penalize all pairs
        for i in 0..qubo_vars.len() {
            for j in (i + 1)..qubo_vars.len() {
                builder
                    .set_quadratic_term(
                        &qubo_vars[i],
                        &qubo_vars[j],
                        self.compilation_params.constraint_penalty,
                    )
                    .map_err(|e| CspError::CompilationFailed(e.to_string()))?;
            }
        }

        Ok(())
    }

    /// Add objective function
    fn add_objective_function(
        &self,
        _builder: &mut QuboBuilder,
        _var_encoding: &HashMap<String, VariableEncoding>,
        _objective: &CspObjective,
        _info: &mut CspCompilationInfo,
    ) -> CspResult<()> {
        // Implementation would depend on objective type and variable encodings
        // For now, return unsupported
        Err(CspError::UnsupportedConstraint(
            "Objective functions not yet implemented".to_string(),
        ))
    }

    /// Get variables involved in a constraint
    fn get_constraint_variables(&self, constraint: &CspConstraint) -> Vec<String> {
        match constraint {
            CspConstraint::AllDifferent { variables } => variables.clone(),
            CspConstraint::Linear { terms, .. } => {
                terms.iter().map(|(var, _)| var.clone()).collect()
            }
            CspConstraint::ExactlyOne { variables } => variables.clone(),
            CspConstraint::AtMostOne { variables } => variables.clone(),
            CspConstraint::Element {
                array_var,
                index_var,
                value_var,
            } => {
                vec![array_var.clone(), index_var.clone(), value_var.clone()]
            }
            CspConstraint::GlobalCardinality { variables, .. } => variables.clone(),
            CspConstraint::Table { variables, .. } => variables.clone(),
            CspConstraint::Custom { variables, .. } => variables.clone(),
        }
    }

    /// Get variables involved in an objective
    fn get_objective_variables(&self, objective: &CspObjective) -> Vec<String> {
        match objective {
            CspObjective::Linear { terms, .. } => {
                terms.iter().map(|(var, _)| var.clone()).collect()
            }
            CspObjective::Custom { .. } => Vec::new(), // Cannot determine statically
        }
    }
}

impl Default for CspProblem {
    fn default() -> Self {
        Self::new()
    }
}

/// Variable encoding in QUBO
#[derive(Debug, Clone)]
enum VariableEncoding {
    /// Direct binary variable
    Direct(crate::qubo::Variable),

    /// One-hot encoding
    OneHot(Vec<(crate::qubo::Variable, CspValue)>),

    /// Binary/logarithmic encoding
    Binary {
        bits: Vec<crate::qubo::Variable>,
        domain_values: Vec<CspValue>,
    },
}

/// Information about CSP compilation
#[derive(Debug, Clone)]
pub struct CspCompilationInfo {
    /// Number of CSP variables
    pub csp_variables: usize,

    /// Number of QUBO variables created
    pub qubo_variables: usize,

    /// Number of constraints compiled
    pub constraints_compiled: usize,

    /// Variable encoding information
    pub variable_info: HashMap<String, VariableInfo>,
}

/// Information about how a CSP variable was encoded
#[derive(Debug, Clone)]
pub struct VariableInfo {
    /// Original domain size
    pub domain_size: usize,

    /// Number of QUBO variables used
    pub qubo_variables_used: usize,

    /// Encoding type used
    pub encoding_type: String,
}

impl CspCompilationInfo {
    fn new() -> Self {
        Self {
            csp_variables: 0,
            qubo_variables: 0,
            constraints_compiled: 0,
            variable_info: HashMap::new(),
        }
    }

    fn add_variable_info(&mut self, name: String, domain_size: usize) {
        let qubo_vars_used = if domain_size == 2 { 1 } else { domain_size };
        let encoding_type = if domain_size == 2 {
            "Direct".to_string()
        } else if domain_size <= 16 {
            "OneHot".to_string()
        } else {
            "Binary".to_string()
        };

        self.variable_info.insert(
            name,
            VariableInfo {
                domain_size,
                qubo_variables_used: qubo_vars_used,
                encoding_type,
            },
        );

        self.csp_variables += 1;
        self.qubo_variables += qubo_vars_used;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_creation() {
        let bool_domain = Domain::Boolean;
        assert_eq!(bool_domain.size(), 2);

        let int_domain = Domain::IntegerRange { min: 1, max: 5 };
        assert_eq!(int_domain.size(), 5);

        let discrete_domain = Domain::Discrete(vec![1, 3, 7, 9]);
        assert_eq!(discrete_domain.size(), 4);
    }

    #[test]
    fn test_csp_variable() {
        let var = CspVariable::new("x".to_string(), Domain::Boolean)
            .with_description("Test variable".to_string());

        assert_eq!(var.name, "x");
        assert_eq!(var.domain, Domain::Boolean);
        assert_eq!(var.description, Some("Test variable".to_string()));
    }

    #[test]
    fn test_simple_csp_problem() {
        let mut problem = CspProblem::new();

        // Add variables
        let x = CspVariable::new("x".to_string(), Domain::Boolean);
        let y = CspVariable::new("y".to_string(), Domain::Boolean);

        problem.add_variable(x).expect("should add variable x");
        problem.add_variable(y).expect("should add variable y");

        // Add exactly-one constraint
        let constraint = CspConstraint::ExactlyOne {
            variables: vec!["x".to_string(), "y".to_string()],
        };
        problem
            .add_constraint(constraint)
            .expect("should add exactly-one constraint");

        // Compile to QUBO
        let (qubo_formulation, info) = problem.compile_to_qubo().expect("should compile to QUBO");

        assert_eq!(info.csp_variables, 2);
        assert_eq!(info.qubo_variables, 2);
        assert_eq!(info.constraints_compiled, 1);

        let qubo_model = qubo_formulation.to_qubo_model();
        assert_eq!(qubo_model.num_variables, 2);
    }

    #[test]
    fn test_all_different_constraint() {
        let mut problem = CspProblem::new();

        // Add three boolean variables
        for i in 0..3 {
            let var = CspVariable::new(format!("x{}", i), Domain::Boolean);
            problem
                .add_variable(var)
                .expect("should add boolean variable");
        }

        // Add all-different constraint (only one can be true for boolean vars)
        let constraint = CspConstraint::AllDifferent {
            variables: vec!["x0".to_string(), "x1".to_string(), "x2".to_string()],
        };
        problem
            .add_constraint(constraint)
            .expect("should add all-different constraint");

        let (_, info) = problem.compile_to_qubo().expect("should compile to QUBO");
        assert_eq!(info.constraints_compiled, 1);
    }

    #[test]
    fn test_one_hot_encoding() {
        let mut problem = CspProblem::new();

        // Add variable with small discrete domain
        let var = CspVariable::new("color".to_string(), Domain::Discrete(vec![1, 2, 3]));
        problem
            .add_variable(var)
            .expect("should add discrete variable");

        let (_, info) = problem.compile_to_qubo().expect("should compile to QUBO");

        // Should use one-hot encoding: 3 QUBO variables for 3 domain values
        assert_eq!(info.variable_info["color"].qubo_variables_used, 3);
        assert_eq!(info.variable_info["color"].encoding_type, "OneHot");
    }
}
