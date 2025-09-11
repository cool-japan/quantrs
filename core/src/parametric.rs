use num_complex::Complex64;
use std::collections::HashMap;
use std::ops::{Add, Div, Mul, Sub};

use crate::error::QuantRS2Result;
use crate::gate::GateOp;
use crate::qubit::QubitId;
use crate::symbolic::SymbolicExpression;

/// Parameter value representation for parametric gates
#[derive(Debug, Clone)]
pub enum Parameter {
    /// Constant floating-point value
    Constant(f64),

    /// Complex constant value
    ComplexConstant(Complex64),

    /// Symbolic parameter with a name and optional value (legacy)
    Symbol(SymbolicParameter),

    /// Full symbolic expression using SymEngine
    Symbolic(SymbolicExpression),
}

impl Parameter {
    /// Create a new constant parameter
    pub fn constant(value: f64) -> Self {
        Self::Constant(value)
    }

    /// Create a new complex constant parameter
    pub fn complex_constant(value: Complex64) -> Self {
        Self::ComplexConstant(value)
    }

    /// Create a new symbolic parameter (legacy)
    pub fn symbol(name: &str) -> Self {
        Self::Symbol(SymbolicParameter::new(name))
    }

    /// Create a new symbolic parameter with a value (legacy)
    pub fn symbol_with_value(name: &str, value: f64) -> Self {
        Self::Symbol(SymbolicParameter::with_value(name, value))
    }

    /// Create a symbolic expression parameter
    pub fn symbolic(expr: SymbolicExpression) -> Self {
        Self::Symbolic(expr)
    }

    /// Create a symbolic variable
    pub fn variable(name: &str) -> Self {
        Self::Symbolic(SymbolicExpression::variable(name))
    }

    /// Parse a parameter from a string expression
    pub fn parse(expr: &str) -> QuantRS2Result<Self> {
        // Try to parse as a number first
        if let Ok(value) = expr.parse::<f64>() {
            return Ok(Self::Constant(value));
        }

        // Otherwise parse as symbolic expression
        let symbolic_expr = SymbolicExpression::parse(expr)?;
        Ok(Self::Symbolic(symbolic_expr))
    }

    /// Get the value of the parameter, if available
    pub fn value(&self) -> Option<f64> {
        match self {
            Self::Constant(val) => Some(*val),
            Self::ComplexConstant(val) => {
                if val.im.abs() < 1e-12 {
                    Some(val.re)
                } else {
                    None // Cannot convert complex to real
                }
            }
            Self::Symbol(sym) => sym.value,
            Self::Symbolic(expr) => {
                // Try to evaluate with empty variables
                expr.evaluate(&HashMap::new()).ok()
            }
        }
    }

    /// Get the complex value of the parameter, if available
    pub fn complex_value(&self) -> Option<Complex64> {
        match self {
            Self::Constant(val) => Some(Complex64::new(*val, 0.0)),
            Self::ComplexConstant(val) => Some(*val),
            Self::Symbol(sym) => sym.value.map(|v| Complex64::new(v, 0.0)),
            Self::Symbolic(expr) => {
                // Try to evaluate with empty variables
                expr.evaluate_complex(&HashMap::new()).ok()
            }
        }
    }

    /// Check if the parameter has a concrete value
    pub fn has_value(&self) -> bool {
        match self {
            Self::Constant(_) | Self::ComplexConstant(_) => true,
            Self::Symbol(sym) => sym.value.is_some(),
            Self::Symbolic(expr) => expr.is_constant(),
        }
    }

    /// Evaluate the parameter with given variable values
    pub fn evaluate(&self, variables: &HashMap<String, f64>) -> QuantRS2Result<f64> {
        match self {
            Self::Constant(val) => Ok(*val),
            Self::ComplexConstant(val) => {
                if val.im.abs() < 1e-12 {
                    Ok(val.re)
                } else {
                    Err(crate::error::QuantRS2Error::InvalidInput(
                        "Cannot evaluate complex parameter to real number".to_string(),
                    ))
                }
            }
            Self::Symbol(sym) => {
                if let Some(value) = sym.value {
                    Ok(value)
                } else {
                    variables.get(&sym.name).copied().ok_or_else(|| {
                        crate::error::QuantRS2Error::InvalidInput(format!(
                            "Variable '{}' not found",
                            sym.name
                        ))
                    })
                }
            }
            Self::Symbolic(expr) => expr.evaluate(variables),
        }
    }

    /// Evaluate the parameter with given variable values (complex)
    pub fn evaluate_complex(
        &self,
        variables: &HashMap<String, Complex64>,
    ) -> QuantRS2Result<Complex64> {
        match self {
            Self::Constant(val) => Ok(Complex64::new(*val, 0.0)),
            Self::ComplexConstant(val) => Ok(*val),
            Self::Symbol(sym) => {
                if let Some(value) = sym.value {
                    Ok(Complex64::new(value, 0.0))
                } else {
                    variables.get(&sym.name).copied().ok_or_else(|| {
                        crate::error::QuantRS2Error::InvalidInput(format!(
                            "Variable '{}' not found",
                            sym.name
                        ))
                    })
                }
            }
            Self::Symbolic(expr) => expr.evaluate_complex(variables),
        }
    }

    /// Get all variable names in the parameter
    pub fn variables(&self) -> Vec<String> {
        match self {
            Self::Constant(_) | Self::ComplexConstant(_) => Vec::new(),
            Self::Symbol(sym) => {
                if sym.value.is_some() {
                    Vec::new()
                } else {
                    vec![sym.name.clone()]
                }
            }
            Self::Symbolic(expr) => expr.variables(),
        }
    }

    /// Substitute variables with expressions
    pub fn substitute(&self, substitutions: &HashMap<String, Self>) -> QuantRS2Result<Self> {
        match self {
            Self::Constant(_) | Self::ComplexConstant(_) => Ok(self.clone()),
            Self::Symbol(sym) => {
                if let Some(replacement) = substitutions.get(&sym.name) {
                    Ok(replacement.clone())
                } else {
                    Ok(self.clone())
                }
            }
            Self::Symbolic(expr) => {
                // Convert Parameter substitutions to SymbolicExpression substitutions
                let symbolic_subs: HashMap<String, SymbolicExpression> = substitutions
                    .iter()
                    .map(|(k, v)| (k.clone(), v.to_symbolic_expression()))
                    .collect();

                let new_expr = expr.substitute(&symbolic_subs)?;
                Ok(Self::Symbolic(new_expr))
            }
        }
    }

    /// Convert parameter to SymbolicExpression
    pub fn to_symbolic_expression(&self) -> SymbolicExpression {
        match self {
            Self::Constant(val) => SymbolicExpression::constant(*val),
            Self::ComplexConstant(val) => SymbolicExpression::complex_constant(*val),
            Self::Symbol(sym) => {
                if let Some(value) = sym.value {
                    SymbolicExpression::constant(value)
                } else {
                    SymbolicExpression::variable(&sym.name)
                }
            }
            Self::Symbolic(expr) => expr.clone(),
        }
    }

    /// Differentiate the parameter with respect to a variable
    #[cfg(feature = "symbolic")]
    pub fn diff(&self, var: &str) -> QuantRS2Result<Self> {
        use crate::symbolic::calculus;

        let expr = self.to_symbolic_expression();
        let diff_expr = calculus::diff(&expr, var)?;
        Ok(Parameter::Symbolic(diff_expr))
    }

    /// Integrate the parameter with respect to a variable
    #[cfg(feature = "symbolic")]
    pub fn integrate(&self, var: &str) -> QuantRS2Result<Self> {
        use crate::symbolic::calculus;

        let expr = self.to_symbolic_expression();
        let int_expr = calculus::integrate(&expr, var)?;
        Ok(Parameter::Symbolic(int_expr))
    }
}

impl From<f64> for Parameter {
    fn from(value: f64) -> Self {
        Self::Constant(value)
    }
}

impl From<Complex64> for Parameter {
    fn from(value: Complex64) -> Self {
        if value.im == 0.0 {
            Self::Constant(value.re)
        } else {
            Self::ComplexConstant(value)
        }
    }
}

impl From<SymbolicExpression> for Parameter {
    fn from(expr: SymbolicExpression) -> Self {
        Self::Symbolic(expr)
    }
}

impl From<&str> for Parameter {
    fn from(name: &str) -> Self {
        Self::variable(name)
    }
}

impl Add for Parameter {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Constant(a), Self::Constant(b)) => Self::Constant(a + b),
            (Self::ComplexConstant(a), Self::ComplexConstant(b)) => {
                Self::ComplexConstant(a + b)
            }
            (Self::Constant(a), Self::ComplexConstant(b)) => {
                Self::ComplexConstant(Complex64::new(a, 0.0) + b)
            }
            (Self::ComplexConstant(a), Self::Constant(b)) => {
                Self::ComplexConstant(a + Complex64::new(b, 0.0))
            }
            (a, b) => {
                // Convert to symbolic expressions and add
                let a_expr = a.to_symbolic_expression();
                let b_expr = b.to_symbolic_expression();
                Self::Symbolic(a_expr + b_expr)
            }
        }
    }
}

impl Sub for Parameter {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Constant(a), Self::Constant(b)) => Self::Constant(a - b),
            (Self::ComplexConstant(a), Self::ComplexConstant(b)) => {
                Self::ComplexConstant(a - b)
            }
            (Self::Constant(a), Self::ComplexConstant(b)) => {
                Self::ComplexConstant(Complex64::new(a, 0.0) - b)
            }
            (Self::ComplexConstant(a), Self::Constant(b)) => {
                Self::ComplexConstant(a - Complex64::new(b, 0.0))
            }
            (a, b) => {
                // Convert to symbolic expressions and subtract
                let a_expr = a.to_symbolic_expression();
                let b_expr = b.to_symbolic_expression();
                Self::Symbolic(a_expr - b_expr)
            }
        }
    }
}

impl Mul for Parameter {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Constant(a), Self::Constant(b)) => Self::Constant(a * b),
            (Self::ComplexConstant(a), Self::ComplexConstant(b)) => {
                Self::ComplexConstant(a * b)
            }
            (Self::Constant(a), Self::ComplexConstant(b)) => {
                Self::ComplexConstant(Complex64::new(a, 0.0) * b)
            }
            (Self::ComplexConstant(a), Self::Constant(b)) => {
                Self::ComplexConstant(a * Complex64::new(b, 0.0))
            }
            (a, b) => {
                // Convert to symbolic expressions and multiply
                let a_expr = a.to_symbolic_expression();
                let b_expr = b.to_symbolic_expression();
                Self::Symbolic(a_expr * b_expr)
            }
        }
    }
}

impl Div for Parameter {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Constant(a), Self::Constant(b)) => Self::Constant(a / b),
            (Self::ComplexConstant(a), Self::ComplexConstant(b)) => {
                Self::ComplexConstant(a / b)
            }
            (Self::Constant(a), Self::ComplexConstant(b)) => {
                Self::ComplexConstant(Complex64::new(a, 0.0) / b)
            }
            (Self::ComplexConstant(a), Self::Constant(b)) => {
                Self::ComplexConstant(a / Complex64::new(b, 0.0))
            }
            (a, b) => {
                // Convert to symbolic expressions and divide
                let a_expr = a.to_symbolic_expression();
                let b_expr = b.to_symbolic_expression();
                Self::Symbolic(a_expr / b_expr)
            }
        }
    }
}

/// Symbolic parameter that can be used in parametric gates
#[derive(Debug, Clone)]
pub struct SymbolicParameter {
    /// Name of the parameter
    pub name: String,

    /// Optional value of the parameter
    pub value: Option<f64>,
}

impl SymbolicParameter {
    /// Create a new symbolic parameter without a value
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            value: None,
        }
    }

    /// Create a new symbolic parameter with a value
    pub fn with_value(name: &str, value: f64) -> Self {
        Self {
            name: name.to_string(),
            value: Some(value),
        }
    }

    /// Set the value of the parameter
    pub fn set_value(&mut self, value: f64) {
        self.value = Some(value);
    }

    /// Clear the value of the parameter
    pub fn clear_value(&mut self) {
        self.value = None;
    }
}

// Note: Cannot implement Copy because String doesn't implement Copy
// Use Clone instead (already implemented above)

/// Trait for parametric gates that extends GateOp with parameter-related functionality
pub trait ParametricGate: GateOp {
    /// Returns the parameters of the gate
    fn parameters(&self) -> Vec<Parameter>;

    /// Returns the names of the parameters
    fn parameter_names(&self) -> Vec<String>;

    /// Returns a new gate with updated parameters
    fn with_parameters(&self, params: &[Parameter]) -> QuantRS2Result<Box<dyn ParametricGate>>;

    /// Returns a new gate with updated parameter at the specified index
    fn with_parameter_at(
        &self,
        index: usize,
        param: Parameter,
    ) -> QuantRS2Result<Box<dyn ParametricGate>>;

    /// Assigns values to symbolic parameters
    fn assign(&self, values: &[(String, f64)]) -> QuantRS2Result<Box<dyn ParametricGate>>;

    /// Returns the gate with all parameters set to the specified values
    fn bind(&self, values: &[(String, f64)]) -> QuantRS2Result<Box<dyn ParametricGate>>;
}

/// Specialized implementation of rotation gates around the X-axis
#[derive(Debug, Clone)]
pub struct ParametricRotationX {
    /// Target qubit
    pub target: QubitId,

    /// Rotation angle parameter
    pub theta: Parameter,
}

impl ParametricRotationX {
    /// Create a new X-rotation gate with a constant angle
    pub fn new(target: QubitId, theta: f64) -> Self {
        Self {
            target,
            theta: Parameter::constant(theta),
        }
    }

    /// Create a new X-rotation gate with a symbolic angle
    pub fn new_symbolic(target: QubitId, name: &str) -> Self {
        Self {
            target,
            theta: Parameter::symbol(name),
        }
    }
}

impl GateOp for ParametricRotationX {
    fn name(&self) -> &'static str {
        "RX"
    }

    fn qubits(&self) -> Vec<QubitId> {
        vec![self.target]
    }

    fn is_parameterized(&self) -> bool {
        true
    }

    fn matrix(&self) -> QuantRS2Result<Vec<Complex64>> {
        if let Some(theta) = self.theta.value() {
            let cos = (theta / 2.0).cos();
            let sin = (theta / 2.0).sin();
            Ok(vec![
                Complex64::new(cos, 0.0),
                Complex64::new(0.0, -sin),
                Complex64::new(0.0, -sin),
                Complex64::new(cos, 0.0),
            ])
        } else {
            Err(crate::error::QuantRS2Error::UnsupportedOperation(
                "Cannot generate matrix for RX gate with unbound symbolic parameter".into(),
            ))
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_gate(&self) -> Box<dyn GateOp> {
        Box::new(self.clone())
    }
}

impl ParametricGate for ParametricRotationX {
    fn parameters(&self) -> Vec<Parameter> {
        vec![self.theta.clone()]
    }

    fn parameter_names(&self) -> Vec<String> {
        match self.theta {
            Parameter::Symbol(ref sym) => vec![sym.name.clone()],
            _ => Vec::new(),
        }
    }

    fn with_parameters(&self, params: &[Parameter]) -> QuantRS2Result<Box<dyn ParametricGate>> {
        if params.len() != 1 {
            return Err(crate::error::QuantRS2Error::InvalidInput(format!(
                "RotationX expects 1 parameter, got {}",
                params.len()
            )));
        }
        Ok(Box::new(Self {
            target: self.target,
            theta: params[0].clone(),
        }))
    }

    fn with_parameter_at(
        &self,
        index: usize,
        param: Parameter,
    ) -> QuantRS2Result<Box<dyn ParametricGate>> {
        if index != 0 {
            return Err(crate::error::QuantRS2Error::InvalidInput(format!(
                "RotationX has only 1 parameter, got index {}",
                index
            )));
        }
        Ok(Box::new(Self {
            target: self.target,
            theta: param.clone(),
        }))
    }

    fn assign(&self, values: &[(String, f64)]) -> QuantRS2Result<Box<dyn ParametricGate>> {
        match self.theta {
            Parameter::Symbol(ref sym) => {
                for (name, value) in values {
                    if sym.name == *name {
                        return Ok(Box::new(Self {
                            target: self.target,
                            theta: Parameter::Symbol(SymbolicParameter::with_value(
                                &sym.name, *value,
                            )),
                        }));
                    }
                }
                // Parameter not found in values, return a clone of the original gate
                Ok(Box::new(self.clone()))
            }
            _ => Ok(Box::new(self.clone())), // Not a symbolic parameter, no change needed
        }
    }

    fn bind(&self, values: &[(String, f64)]) -> QuantRS2Result<Box<dyn ParametricGate>> {
        self.assign(values)
    }
}

/// Specialized implementation of rotation gates around the Y-axis
#[derive(Debug, Clone)]
pub struct ParametricRotationY {
    /// Target qubit
    pub target: QubitId,

    /// Rotation angle parameter
    pub theta: Parameter,
}

impl ParametricRotationY {
    /// Create a new Y-rotation gate with a constant angle
    pub fn new(target: QubitId, theta: f64) -> Self {
        Self {
            target,
            theta: Parameter::constant(theta),
        }
    }

    /// Create a new Y-rotation gate with a symbolic angle
    pub fn new_symbolic(target: QubitId, name: &str) -> Self {
        Self {
            target,
            theta: Parameter::symbol(name),
        }
    }
}

impl GateOp for ParametricRotationY {
    fn name(&self) -> &'static str {
        "RY"
    }

    fn qubits(&self) -> Vec<QubitId> {
        vec![self.target]
    }

    fn is_parameterized(&self) -> bool {
        true
    }

    fn matrix(&self) -> QuantRS2Result<Vec<Complex64>> {
        if let Some(theta) = self.theta.value() {
            let cos = (theta / 2.0).cos();
            let sin = (theta / 2.0).sin();
            Ok(vec![
                Complex64::new(cos, 0.0),
                Complex64::new(-sin, 0.0),
                Complex64::new(sin, 0.0),
                Complex64::new(cos, 0.0),
            ])
        } else {
            Err(crate::error::QuantRS2Error::UnsupportedOperation(
                "Cannot generate matrix for RY gate with unbound symbolic parameter".into(),
            ))
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_gate(&self) -> Box<dyn GateOp> {
        Box::new(self.clone())
    }
}

impl ParametricGate for ParametricRotationY {
    fn parameters(&self) -> Vec<Parameter> {
        vec![self.theta.clone()]
    }

    fn parameter_names(&self) -> Vec<String> {
        match self.theta {
            Parameter::Symbol(ref sym) => vec![sym.name.clone()],
            _ => Vec::new(),
        }
    }

    fn with_parameters(&self, params: &[Parameter]) -> QuantRS2Result<Box<dyn ParametricGate>> {
        if params.len() != 1 {
            return Err(crate::error::QuantRS2Error::InvalidInput(format!(
                "RotationY expects 1 parameter, got {}",
                params.len()
            )));
        }
        Ok(Box::new(Self {
            target: self.target,
            theta: params[0].clone(),
        }))
    }

    fn with_parameter_at(
        &self,
        index: usize,
        param: Parameter,
    ) -> QuantRS2Result<Box<dyn ParametricGate>> {
        if index != 0 {
            return Err(crate::error::QuantRS2Error::InvalidInput(format!(
                "RotationY has only 1 parameter, got index {}",
                index
            )));
        }
        Ok(Box::new(Self {
            target: self.target,
            theta: param.clone(),
        }))
    }

    fn assign(&self, values: &[(String, f64)]) -> QuantRS2Result<Box<dyn ParametricGate>> {
        match self.theta {
            Parameter::Symbol(ref sym) => {
                for (name, value) in values {
                    if sym.name == *name {
                        return Ok(Box::new(Self {
                            target: self.target,
                            theta: Parameter::Symbol(SymbolicParameter::with_value(
                                &sym.name, *value,
                            )),
                        }));
                    }
                }
                // Parameter not found in values, return a clone of the original gate
                Ok(Box::new(self.clone()))
            }
            _ => Ok(Box::new(self.clone())), // Not a symbolic parameter, no change needed
        }
    }

    fn bind(&self, values: &[(String, f64)]) -> QuantRS2Result<Box<dyn ParametricGate>> {
        self.assign(values)
    }
}

/// Specialized implementation of rotation gates around the Z-axis
#[derive(Debug, Clone)]
pub struct ParametricRotationZ {
    /// Target qubit
    pub target: QubitId,

    /// Rotation angle parameter
    pub theta: Parameter,
}

impl ParametricRotationZ {
    /// Create a new Z-rotation gate with a constant angle
    pub fn new(target: QubitId, theta: f64) -> Self {
        Self {
            target,
            theta: Parameter::constant(theta),
        }
    }

    /// Create a new Z-rotation gate with a symbolic angle
    pub fn new_symbolic(target: QubitId, name: &str) -> Self {
        Self {
            target,
            theta: Parameter::symbol(name),
        }
    }
}

impl GateOp for ParametricRotationZ {
    fn name(&self) -> &'static str {
        "RZ"
    }

    fn qubits(&self) -> Vec<QubitId> {
        vec![self.target]
    }

    fn is_parameterized(&self) -> bool {
        true
    }

    fn matrix(&self) -> QuantRS2Result<Vec<Complex64>> {
        if let Some(theta) = self.theta.value() {
            let phase = Complex64::new(0.0, -theta / 2.0).exp();
            let phase_conj = Complex64::new(0.0, theta / 2.0).exp();
            Ok(vec![
                phase_conj,
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                phase,
            ])
        } else {
            Err(crate::error::QuantRS2Error::UnsupportedOperation(
                "Cannot generate matrix for RZ gate with unbound symbolic parameter".into(),
            ))
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_gate(&self) -> Box<dyn GateOp> {
        Box::new(self.clone())
    }
}

impl ParametricGate for ParametricRotationZ {
    fn parameters(&self) -> Vec<Parameter> {
        vec![self.theta.clone()]
    }

    fn parameter_names(&self) -> Vec<String> {
        match self.theta {
            Parameter::Symbol(ref sym) => vec![sym.name.clone()],
            _ => Vec::new(),
        }
    }

    fn with_parameters(&self, params: &[Parameter]) -> QuantRS2Result<Box<dyn ParametricGate>> {
        if params.len() != 1 {
            return Err(crate::error::QuantRS2Error::InvalidInput(format!(
                "RotationZ expects 1 parameter, got {}",
                params.len()
            )));
        }
        Ok(Box::new(Self {
            target: self.target,
            theta: params[0].clone(),
        }))
    }

    fn with_parameter_at(
        &self,
        index: usize,
        param: Parameter,
    ) -> QuantRS2Result<Box<dyn ParametricGate>> {
        if index != 0 {
            return Err(crate::error::QuantRS2Error::InvalidInput(format!(
                "RotationZ has only 1 parameter, got index {}",
                index
            )));
        }
        Ok(Box::new(Self {
            target: self.target,
            theta: param.clone(),
        }))
    }

    fn assign(&self, values: &[(String, f64)]) -> QuantRS2Result<Box<dyn ParametricGate>> {
        match self.theta {
            Parameter::Symbol(ref sym) => {
                for (name, value) in values {
                    if sym.name == *name {
                        return Ok(Box::new(Self {
                            target: self.target,
                            theta: Parameter::Symbol(SymbolicParameter::with_value(
                                &sym.name, *value,
                            )),
                        }));
                    }
                }
                // Parameter not found in values, return a clone of the original gate
                Ok(Box::new(self.clone()))
            }
            _ => Ok(Box::new(self.clone())), // Not a symbolic parameter, no change needed
        }
    }

    fn bind(&self, values: &[(String, f64)]) -> QuantRS2Result<Box<dyn ParametricGate>> {
        self.assign(values)
    }
}

/// Specialized implementation of a general U-gate (parameterized single-qubit gate)
#[derive(Debug, Clone)]
pub struct ParametricU {
    /// Target qubit
    pub target: QubitId,

    /// Theta parameter (rotation angle)
    pub theta: Parameter,

    /// Phi parameter (phase angle)
    pub phi: Parameter,

    /// Lambda parameter (phase angle)
    pub lambda: Parameter,
}

impl ParametricU {
    /// Create a new U-gate with constant angles
    pub fn new(target: QubitId, theta: f64, phi: f64, lambda: f64) -> Self {
        Self {
            target,
            theta: Parameter::constant(theta),
            phi: Parameter::constant(phi),
            lambda: Parameter::constant(lambda),
        }
    }

    /// Create a new U-gate with symbolic angles
    pub fn new_symbolic(
        target: QubitId,
        theta_name: &str,
        phi_name: &str,
        lambda_name: &str,
    ) -> Self {
        Self {
            target,
            theta: Parameter::symbol(theta_name),
            phi: Parameter::symbol(phi_name),
            lambda: Parameter::symbol(lambda_name),
        }
    }
}

impl GateOp for ParametricU {
    fn name(&self) -> &'static str {
        "U"
    }

    fn qubits(&self) -> Vec<QubitId> {
        vec![self.target]
    }

    fn is_parameterized(&self) -> bool {
        true
    }

    fn matrix(&self) -> QuantRS2Result<Vec<Complex64>> {
        if let (Some(theta), Some(phi), Some(lambda)) =
            (self.theta.value(), self.phi.value(), self.lambda.value())
        {
            let cos = (theta / 2.0).cos();
            let sin = (theta / 2.0).sin();

            let e_phi = Complex64::new(0.0, phi).exp();
            let e_lambda = Complex64::new(0.0, lambda).exp();

            Ok(vec![
                Complex64::new(cos, 0.0),
                -sin * e_lambda,
                sin * e_phi,
                cos * e_phi * e_lambda,
            ])
        } else {
            Err(crate::error::QuantRS2Error::UnsupportedOperation(
                "Cannot generate matrix for U gate with unbound symbolic parameters".into(),
            ))
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_gate(&self) -> Box<dyn GateOp> {
        Box::new(self.clone())
    }
}

impl ParametricGate for ParametricU {
    fn parameters(&self) -> Vec<Parameter> {
        vec![self.theta.clone(), self.phi.clone(), self.lambda.clone()]
    }

    fn parameter_names(&self) -> Vec<String> {
        let mut names = Vec::new();

        if let Parameter::Symbol(ref sym) = self.theta {
            names.push(sym.name.clone());
        }

        if let Parameter::Symbol(ref sym) = self.phi {
            names.push(sym.name.clone());
        }

        if let Parameter::Symbol(ref sym) = self.lambda {
            names.push(sym.name.clone());
        }

        names
    }

    fn with_parameters(&self, params: &[Parameter]) -> QuantRS2Result<Box<dyn ParametricGate>> {
        if params.len() != 3 {
            return Err(crate::error::QuantRS2Error::InvalidInput(format!(
                "U gate expects 3 parameters, got {}",
                params.len()
            )));
        }
        Ok(Box::new(Self {
            target: self.target,
            theta: params[0].clone(),
            phi: params[1].clone(),
            lambda: params[2].clone(),
        }))
    }

    fn with_parameter_at(
        &self,
        index: usize,
        param: Parameter,
    ) -> QuantRS2Result<Box<dyn ParametricGate>> {
        match index {
            0 => Ok(Box::new(Self {
                target: self.target,
                theta: param.clone(),
                phi: self.phi.clone(),
                lambda: self.lambda.clone(),
            })),
            1 => Ok(Box::new(Self {
                target: self.target,
                theta: self.theta.clone(),
                phi: param.clone(),
                lambda: self.lambda.clone(),
            })),
            2 => Ok(Box::new(Self {
                target: self.target,
                theta: self.theta.clone(),
                phi: self.phi.clone(),
                lambda: param.clone(),
            })),
            _ => Err(crate::error::QuantRS2Error::InvalidInput(format!(
                "U gate has only 3 parameters, got index {}",
                index
            ))),
        }
    }

    fn assign(&self, values: &[(String, f64)]) -> QuantRS2Result<Box<dyn ParametricGate>> {
        let mut result = self.clone();

        // Update theta if it's a symbolic parameter
        if let Parameter::Symbol(ref sym) = self.theta {
            for (name, value) in values {
                if sym.name == *name {
                    result.theta =
                        Parameter::Symbol(SymbolicParameter::with_value(&sym.name, *value));
                    break;
                }
            }
        }

        // Update phi if it's a symbolic parameter
        if let Parameter::Symbol(ref sym) = self.phi {
            for (name, value) in values {
                if sym.name == *name {
                    result.phi =
                        Parameter::Symbol(SymbolicParameter::with_value(&sym.name, *value));
                    break;
                }
            }
        }

        // Update lambda if it's a symbolic parameter
        if let Parameter::Symbol(ref sym) = self.lambda {
            for (name, value) in values {
                if sym.name == *name {
                    result.lambda =
                        Parameter::Symbol(SymbolicParameter::with_value(&sym.name, *value));
                    break;
                }
            }
        }

        Ok(Box::new(result))
    }

    fn bind(&self, values: &[(String, f64)]) -> QuantRS2Result<Box<dyn ParametricGate>> {
        self.assign(values)
    }
}

/// Specialized implementation of controlled parametric rotation around X-axis
#[derive(Debug, Clone)]
pub struct ParametricCRX {
    /// Control qubit
    pub control: QubitId,

    /// Target qubit
    pub target: QubitId,

    /// Rotation angle parameter
    pub theta: Parameter,
}

impl ParametricCRX {
    /// Create a new CRX gate with a constant angle
    pub fn new(control: QubitId, target: QubitId, theta: f64) -> Self {
        Self {
            control,
            target,
            theta: Parameter::constant(theta),
        }
    }

    /// Create a new CRX gate with a symbolic angle
    pub fn new_symbolic(control: QubitId, target: QubitId, name: &str) -> Self {
        Self {
            control,
            target,
            theta: Parameter::symbol(name),
        }
    }
}

impl GateOp for ParametricCRX {
    fn name(&self) -> &'static str {
        "CRX"
    }

    fn qubits(&self) -> Vec<QubitId> {
        vec![self.control, self.target]
    }

    fn is_parameterized(&self) -> bool {
        true
    }

    fn matrix(&self) -> QuantRS2Result<Vec<Complex64>> {
        if let Some(theta) = self.theta.value() {
            let cos = (theta / 2.0).cos();
            let sin = (theta / 2.0).sin();

            Ok(vec![
                Complex64::new(1.0, 0.0),  // (0,0)
                Complex64::new(0.0, 0.0),  // (0,1)
                Complex64::new(0.0, 0.0),  // (0,2)
                Complex64::new(0.0, 0.0),  // (0,3)
                Complex64::new(0.0, 0.0),  // (1,0)
                Complex64::new(1.0, 0.0),  // (1,1)
                Complex64::new(0.0, 0.0),  // (1,2)
                Complex64::new(0.0, 0.0),  // (1,3)
                Complex64::new(0.0, 0.0),  // (2,0)
                Complex64::new(0.0, 0.0),  // (2,1)
                Complex64::new(cos, 0.0),  // (2,2)
                Complex64::new(0.0, -sin), // (2,3)
                Complex64::new(0.0, 0.0),  // (3,0)
                Complex64::new(0.0, 0.0),  // (3,1)
                Complex64::new(0.0, -sin), // (3,2)
                Complex64::new(cos, 0.0),  // (3,3)
            ])
        } else {
            Err(crate::error::QuantRS2Error::UnsupportedOperation(
                "Cannot generate matrix for CRX gate with unbound symbolic parameter".into(),
            ))
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_gate(&self) -> Box<dyn GateOp> {
        Box::new(self.clone())
    }
}

impl ParametricGate for ParametricCRX {
    fn parameters(&self) -> Vec<Parameter> {
        vec![self.theta.clone()]
    }

    fn parameter_names(&self) -> Vec<String> {
        match self.theta {
            Parameter::Symbol(ref sym) => vec![sym.name.clone()],
            _ => Vec::new(),
        }
    }

    fn with_parameters(&self, params: &[Parameter]) -> QuantRS2Result<Box<dyn ParametricGate>> {
        if params.len() != 1 {
            return Err(crate::error::QuantRS2Error::InvalidInput(format!(
                "CRX expects 1 parameter, got {}",
                params.len()
            )));
        }
        Ok(Box::new(Self {
            control: self.control,
            target: self.target,
            theta: params[0].clone(),
        }))
    }

    fn with_parameter_at(
        &self,
        index: usize,
        param: Parameter,
    ) -> QuantRS2Result<Box<dyn ParametricGate>> {
        if index != 0 {
            return Err(crate::error::QuantRS2Error::InvalidInput(format!(
                "CRX has only 1 parameter, got index {}",
                index
            )));
        }
        Ok(Box::new(Self {
            control: self.control,
            target: self.target,
            theta: param.clone(),
        }))
    }

    fn assign(&self, values: &[(String, f64)]) -> QuantRS2Result<Box<dyn ParametricGate>> {
        match self.theta {
            Parameter::Symbol(ref sym) => {
                for (name, value) in values {
                    if sym.name == *name {
                        return Ok(Box::new(Self {
                            control: self.control,
                            target: self.target,
                            theta: Parameter::Symbol(SymbolicParameter::with_value(
                                &sym.name, *value,
                            )),
                        }));
                    }
                }
                // Parameter not found in values, return a clone of the original gate
                Ok(Box::new(self.clone()))
            }
            _ => Ok(Box::new(self.clone())), // Not a symbolic parameter, no change needed
        }
    }

    fn bind(&self, values: &[(String, f64)]) -> QuantRS2Result<Box<dyn ParametricGate>> {
        self.assign(values)
    }
}

/// Phase shift gate with a parameterized phase
#[derive(Debug, Clone)]
pub struct ParametricPhaseShift {
    /// Target qubit
    pub target: QubitId,

    /// Phase parameter
    pub phi: Parameter,
}

impl ParametricPhaseShift {
    /// Create a new phase shift gate with a constant phase
    pub fn new(target: QubitId, phi: f64) -> Self {
        Self {
            target,
            phi: Parameter::constant(phi),
        }
    }

    /// Create a new phase shift gate with a symbolic phase
    pub fn new_symbolic(target: QubitId, name: &str) -> Self {
        Self {
            target,
            phi: Parameter::symbol(name),
        }
    }
}

impl GateOp for ParametricPhaseShift {
    fn name(&self) -> &'static str {
        "P"
    }

    fn qubits(&self) -> Vec<QubitId> {
        vec![self.target]
    }

    fn is_parameterized(&self) -> bool {
        true
    }

    fn matrix(&self) -> QuantRS2Result<Vec<Complex64>> {
        if let Some(phi) = self.phi.value() {
            let phase = Complex64::new(phi.cos(), phi.sin());
            Ok(vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                phase,
            ])
        } else {
            Err(crate::error::QuantRS2Error::UnsupportedOperation(
                "Cannot generate matrix for phase shift gate with unbound symbolic parameter"
                    .into(),
            ))
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_gate(&self) -> Box<dyn GateOp> {
        Box::new(self.clone())
    }
}

impl ParametricGate for ParametricPhaseShift {
    fn parameters(&self) -> Vec<Parameter> {
        vec![self.phi.clone()]
    }

    fn parameter_names(&self) -> Vec<String> {
        match self.phi {
            Parameter::Symbol(ref sym) => vec![sym.name.clone()],
            _ => Vec::new(),
        }
    }

    fn with_parameters(&self, params: &[Parameter]) -> QuantRS2Result<Box<dyn ParametricGate>> {
        if params.len() != 1 {
            return Err(crate::error::QuantRS2Error::InvalidInput(format!(
                "Phase shift gate expects 1 parameter, got {}",
                params.len()
            )));
        }
        Ok(Box::new(Self {
            target: self.target,
            phi: params[0].clone(),
        }))
    }

    fn with_parameter_at(
        &self,
        index: usize,
        param: Parameter,
    ) -> QuantRS2Result<Box<dyn ParametricGate>> {
        if index != 0 {
            return Err(crate::error::QuantRS2Error::InvalidInput(format!(
                "Phase shift gate has only 1 parameter, got index {}",
                index
            )));
        }
        Ok(Box::new(Self {
            target: self.target,
            phi: param.clone(),
        }))
    }

    fn assign(&self, values: &[(String, f64)]) -> QuantRS2Result<Box<dyn ParametricGate>> {
        match self.phi {
            Parameter::Symbol(ref sym) => {
                for (name, value) in values {
                    if sym.name == *name {
                        return Ok(Box::new(Self {
                            target: self.target,
                            phi: Parameter::Symbol(SymbolicParameter::with_value(
                                &sym.name, *value,
                            )),
                        }));
                    }
                }
                // Parameter not found in values, return a clone of the original gate
                Ok(Box::new(self.clone()))
            }
            _ => Ok(Box::new(self.clone())), // Not a symbolic parameter, no change needed
        }
    }

    fn bind(&self, values: &[(String, f64)]) -> QuantRS2Result<Box<dyn ParametricGate>> {
        self.assign(values)
    }
}

/// Module for utilities related to parametric gates
pub mod utils {
    use super::*;
    use crate::gate::{multi, single};

    /// Helper function to create a parametric gate from a standard gate
    pub fn parametrize_rotation_gate(gate: &dyn GateOp) -> Option<Box<dyn ParametricGate>> {
        if !gate.is_parameterized() {
            return None;
        }

        if let Some(rx) = gate.as_any().downcast_ref::<single::RotationX>() {
            Some(Box::new(ParametricRotationX::new(rx.target, rx.theta)))
        } else if let Some(ry) = gate.as_any().downcast_ref::<single::RotationY>() {
            Some(Box::new(ParametricRotationY::new(ry.target, ry.theta)))
        } else if let Some(rz) = gate.as_any().downcast_ref::<single::RotationZ>() {
            Some(Box::new(ParametricRotationZ::new(rz.target, rz.theta)))
        } else if let Some(crx) = gate.as_any().downcast_ref::<multi::CRX>() {
            Some(Box::new(ParametricCRX::new(
                crx.control,
                crx.target,
                crx.theta,
            )))
        } else {
            None
        }
    }

    /// Convert a parameter value to a symbolic parameter
    pub fn symbolize_parameter(param: f64, name: &str) -> Parameter {
        Parameter::symbol_with_value(name, param)
    }

    /// Check if two parameters are approximately equal
    pub fn parameters_approx_eq(p1: &Parameter, p2: &Parameter, epsilon: f64) -> bool {
        match (p1.value(), p2.value()) {
            (Some(v1), Some(v2)) => (v1 - v2).abs() < epsilon,
            _ => false,
        }
    }
}
