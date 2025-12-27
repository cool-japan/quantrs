//! Compiler for the problem DSL.

use super::ast::{
    AggregationOp, BinaryOperator, ComparisonOp, Constraint, ConstraintExpression, Declaration,
    Expression, Objective, ObjectiveType, Value, AST,
};
use super::error::CompileError;
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;

/// Compiler options
#[derive(Debug, Clone)]
pub struct CompilerOptions {
    /// Optimization level
    pub optimization_level: OptimizationLevel,
    /// Target backend
    pub target: TargetBackend,
    /// Debug information
    pub debug_info: bool,
    /// Warnings as errors
    pub warnings_as_errors: bool,
}

#[derive(Debug, Clone)]
pub enum OptimizationLevel {
    None,
    Basic,
    Full,
}

#[derive(Debug, Clone)]
pub enum TargetBackend {
    QUBO,
    Ising,
    HigherOrder,
}

impl Default for CompilerOptions {
    fn default() -> Self {
        Self {
            optimization_level: OptimizationLevel::Basic,
            target: TargetBackend::QUBO,
            debug_info: false,
            warnings_as_errors: false,
        }
    }
}

/// Variable registry for tracking variables during compilation
#[derive(Debug, Clone)]
struct VariableRegistry {
    /// Maps variable names to their indices in the QUBO matrix
    var_indices: HashMap<String, usize>,
    /// Maps indexed variables (e.g., x[i,j]) to their indices
    indexed_var_indices: HashMap<String, HashMap<Vec<usize>, usize>>,
    /// Total number of variables
    num_vars: usize,
    /// Variable domains
    domains: HashMap<String, VariableDomain>,
}

#[derive(Debug, Clone)]
enum VariableDomain {
    Binary,
    Integer { min: i32, max: i32 },
    Continuous { min: f64, max: f64 },
}

impl VariableRegistry {
    fn new() -> Self {
        Self {
            var_indices: HashMap::new(),
            indexed_var_indices: HashMap::new(),
            num_vars: 0,
            domains: HashMap::new(),
        }
    }

    fn register_variable(&mut self, name: &str, domain: VariableDomain) -> usize {
        if let Some(&idx) = self.var_indices.get(name) {
            return idx;
        }
        let idx = self.num_vars;
        self.var_indices.insert(name.to_string(), idx);
        self.domains.insert(name.to_string(), domain);
        self.num_vars += 1;
        idx
    }

    fn register_indexed_variable(
        &mut self,
        base_name: &str,
        indices: Vec<usize>,
        domain: VariableDomain,
    ) -> usize {
        let indexed_map = self
            .indexed_var_indices
            .entry(base_name.to_string())
            .or_default();
        if let Some(&idx) = indexed_map.get(&indices) {
            return idx;
        }
        let idx = self.num_vars;
        indexed_map.insert(indices, idx);
        let full_name = format!("{}_{}", base_name, self.num_vars);
        self.domains.insert(full_name, domain);
        self.num_vars += 1;
        idx
    }
}

/// Compile AST to QUBO matrix
pub fn compile_to_qubo(ast: &AST, options: &CompilerOptions) -> Result<Array2<f64>, CompileError> {
    match ast {
        AST::Program {
            declarations,
            objective,
            constraints,
        } => {
            let mut compiler = Compiler::new(options.clone());

            // Process declarations
            for decl in declarations {
                compiler.process_declaration(decl)?;
            }

            // Build QUBO from objective
            let mut qubo = compiler.build_objective_qubo(objective)?;

            // Add constraint penalties
            for constraint in constraints {
                compiler.add_constraint_penalty(&mut qubo, constraint)?;
            }

            Ok(qubo)
        }
        _ => Err(CompileError {
            message: "Can only compile program AST nodes".to_string(),
            context: "compile_to_qubo".to_string(),
        }),
    }
}

/// Internal compiler state
#[derive(Clone)]
struct Compiler {
    options: CompilerOptions,
    registry: VariableRegistry,
    parameters: HashMap<String, Value>,
    penalty_weight: f64,
}

impl Compiler {
    fn new(options: CompilerOptions) -> Self {
        Self {
            options,
            registry: VariableRegistry::new(),
            parameters: HashMap::new(),
            penalty_weight: 1000.0, // Default penalty weight for constraints
        }
    }

    fn process_declaration(&mut self, decl: &Declaration) -> Result<(), CompileError> {
        match decl {
            Declaration::Variable {
                name,
                var_type: _,
                domain: _,
                attributes: _,
            } => {
                // For now, assume all variables are binary
                self.registry
                    .register_variable(name, VariableDomain::Binary);
                Ok(())
            }
            Declaration::Parameter { name, value, .. } => {
                self.parameters.insert(name.clone(), value.clone());
                Ok(())
            }
            Declaration::Set { name, elements } => {
                // Register set as parameter for later use in aggregations
                self.parameters
                    .insert(name.clone(), Value::Array(elements.clone()));
                Ok(())
            }
            Declaration::Function {
                name,
                params: _,
                body: _,
            } => {
                // Store function definition for later expansion
                // For now, treat as a complex parameter
                self.parameters.insert(
                    format!("func_{name}"),
                    Value::String(format!("function_{name}")),
                );
                Ok(())
            }
        }
    }

    fn build_objective_qubo(&mut self, objective: &Objective) -> Result<Array2<f64>, CompileError> {
        let num_vars = self.registry.num_vars;
        let mut qubo = Array2::zeros((num_vars, num_vars));

        match objective {
            Objective::Minimize(expr) => {
                self.add_expression_to_qubo(&mut qubo, expr, 1.0)?;
            }
            Objective::Maximize(expr) => {
                self.add_expression_to_qubo(&mut qubo, expr, -1.0)?;
            }
            Objective::MultiObjective { objectives } => {
                for (obj_type, expr, weight) in objectives {
                    let sign = match obj_type {
                        ObjectiveType::Minimize => 1.0,
                        ObjectiveType::Maximize => -1.0,
                    };
                    self.add_expression_to_qubo(&mut qubo, expr, sign * weight)?;
                }
            }
        }

        Ok(qubo)
    }

    fn add_expression_to_qubo(
        &mut self,
        qubo: &mut Array2<f64>,
        expr: &Expression,
        coefficient: f64,
    ) -> Result<(), CompileError> {
        match expr {
            Expression::Variable(name) => {
                if let Some(&idx) = self.registry.var_indices.get(name) {
                    qubo[[idx, idx]] += coefficient;
                } else {
                    return Err(CompileError {
                        message: format!("Unknown variable: {name}"),
                        context: "add_expression_to_qubo".to_string(),
                    });
                }
            }
            Expression::BinaryOp { op, left, right } => {
                match op {
                    BinaryOperator::Add => {
                        self.add_expression_to_qubo(qubo, left, coefficient)?;
                        self.add_expression_to_qubo(qubo, right, coefficient)?;
                    }
                    BinaryOperator::Subtract => {
                        self.add_expression_to_qubo(qubo, left, coefficient)?;
                        self.add_expression_to_qubo(qubo, right, -coefficient)?;
                    }
                    BinaryOperator::Multiply => {
                        // Handle multiplication of two variables (creates quadratic term)
                        if let (Expression::Variable(v1), Expression::Variable(v2)) =
                            (left.as_ref(), right.as_ref())
                        {
                            if let (Some(&idx1), Some(&idx2)) = (
                                self.registry.var_indices.get(v1),
                                self.registry.var_indices.get(v2),
                            ) {
                                if idx1 == idx2 {
                                    // x*x = x for binary variables
                                    qubo[[idx1, idx1]] += coefficient;
                                } else {
                                    // Quadratic term
                                    qubo[[idx1, idx2]] += coefficient / 2.0;
                                    qubo[[idx2, idx1]] += coefficient / 2.0;
                                }
                            }
                        } else {
                            return Err(CompileError {
                                message: "Complex multiplication not yet supported".to_string(),
                                context: "add_expression_to_qubo".to_string(),
                            });
                        }
                    }
                    _ => {
                        return Err(CompileError {
                            message: format!("Unsupported binary operator: {op:?}"),
                            context: "add_expression_to_qubo".to_string(),
                        });
                    }
                }
            }
            Expression::Literal(Value::Number(_)) => {
                // Constants don't affect the optimization, but we could track them
                // for the objective value offset
            }
            Expression::Aggregation {
                op,
                variables,
                expression,
            } => {
                match op {
                    AggregationOp::Sum => {
                        // Expand sum over index sets
                        for (var_name, set_name) in variables {
                            // Clone the elements to avoid borrowing conflicts
                            let elements = if let Some(Value::Array(elements)) =
                                self.parameters.get(set_name)
                            {
                                elements.clone()
                            } else {
                                return Err(CompileError {
                                    message: format!("Unknown set for aggregation: {set_name}"),
                                    context: "add_expression_to_qubo".to_string(),
                                });
                            };

                            // For each element in the set, substitute and add expression
                            for (i, element) in elements.iter().enumerate() {
                                // Create substituted expression
                                let substituted_expr = {
                                    let mut compiler = self.clone();
                                    compiler.substitute_variable_in_expression(
                                        expression, var_name, element, i,
                                    )?
                                };
                                let mut qubo_mut = qubo.clone();
                                let mut compiler = self.clone();
                                compiler.add_expression_to_qubo(
                                    &mut qubo_mut,
                                    &substituted_expr,
                                    coefficient,
                                )?;
                                *qubo = qubo_mut;
                            }
                        }
                    }
                    AggregationOp::Product => {
                        // Product expansion (multiply all terms)
                        let mut product_expr = Expression::Literal(Value::Number(1.0));
                        for (var_name, set_name) in variables {
                            // Clone the elements to avoid borrowing conflicts
                            let elements = if let Some(Value::Array(elements)) =
                                self.parameters.get(set_name)
                            {
                                elements.clone()
                            } else {
                                continue; // Skip if set doesn't exist
                            };

                            for (i, element) in elements.iter().enumerate() {
                                let substituted_expr = {
                                    let mut compiler = self.clone();
                                    compiler.substitute_variable_in_expression(
                                        expression, var_name, element, i,
                                    )?
                                };
                                product_expr = Expression::BinaryOp {
                                    op: BinaryOperator::Multiply,
                                    left: Box::new(product_expr),
                                    right: Box::new(substituted_expr),
                                };
                            }
                        }
                        let mut qubo_mut = qubo.clone();
                        let mut compiler = self.clone();
                        compiler.add_expression_to_qubo(
                            &mut qubo_mut,
                            &product_expr,
                            coefficient,
                        )?;
                        *qubo = qubo_mut;
                    }
                    _ => {
                        return Err(CompileError {
                            message: format!("Unsupported aggregation operator: {op:?}"),
                            context: "add_expression_to_qubo".to_string(),
                        });
                    }
                }
            }
            _ => {
                return Err(CompileError {
                    message: "Expression type not yet supported".to_string(),
                    context: "add_expression_to_qubo".to_string(),
                });
            }
        }
        Ok(())
    }

    fn substitute_variable_in_expression(
        &mut self,
        expr: &Expression,
        var_name: &str,
        value: &Value,
        index: usize,
    ) -> Result<Expression, CompileError> {
        match expr {
            Expression::Variable(name) if name == var_name => {
                // Replace with indexed variable or direct substitution
                match value {
                    Value::Number(_n) => {
                        let indexed_name = format!("{var_name}_{index}");
                        self.registry
                            .register_variable(&indexed_name, VariableDomain::Binary);
                        Ok(Expression::Variable(indexed_name))
                    }
                    _ => Ok(Expression::Literal(value.clone())),
                }
            }
            Expression::Variable(name) => Ok(Expression::Variable(name.clone())),
            Expression::BinaryOp { op, left, right } => {
                let new_left =
                    self.substitute_variable_in_expression(left, var_name, value, index)?;
                let new_right =
                    self.substitute_variable_in_expression(right, var_name, value, index)?;
                Ok(Expression::BinaryOp {
                    op: op.clone(),
                    left: Box::new(new_left),
                    right: Box::new(new_right),
                })
            }
            Expression::IndexedVar { name, indices } => {
                let new_indices = indices
                    .iter()
                    .map(|idx| self.substitute_variable_in_expression(idx, var_name, value, index))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Expression::IndexedVar {
                    name: name.clone(),
                    indices: new_indices,
                })
            }
            _ => Ok(expr.clone()),
        }
    }

    fn add_constraint_penalty(
        &mut self,
        qubo: &mut Array2<f64>,
        constraint: &Constraint,
    ) -> Result<(), CompileError> {
        match &constraint.expression {
            ConstraintExpression::Comparison { left, op, right } => {
                match op {
                    ComparisonOp::Equal => {
                        // For equality constraint: (left - right)^2
                        // Expand: left^2 - 2*left*right + right^2
                        self.add_expression_to_qubo(qubo, left, self.penalty_weight)?;
                        self.add_expression_to_qubo(qubo, right, self.penalty_weight)?;

                        // Cross term: -2*left*right
                        if let (Expression::Variable(v1), Expression::Variable(v2)) = (left, right)
                        {
                            if let (Some(&idx1), Some(&idx2)) = (
                                self.registry.var_indices.get(v1),
                                self.registry.var_indices.get(v2),
                            ) {
                                qubo[[idx1, idx2]] -= self.penalty_weight;
                                qubo[[idx2, idx1]] -= self.penalty_weight;
                            }
                        }
                    }
                    ComparisonOp::LessEqual => {
                        // For a <= b, add slack variable: a + s = b, where s >= 0
                        let slack_name = format!("slack_{}", self.registry.num_vars);
                        let _slack_idx = self
                            .registry
                            .register_variable(&slack_name, VariableDomain::Binary);

                        // Convert to equality: (a + s - b)^2
                        let penalty_expr = Expression::BinaryOp {
                            op: BinaryOperator::Subtract,
                            left: Box::new(Expression::BinaryOp {
                                op: BinaryOperator::Add,
                                left: Box::new(left.clone()),
                                right: Box::new(Expression::Variable(slack_name)),
                            }),
                            right: Box::new(right.clone()),
                        };

                        // Add squared penalty
                        self.add_squared_penalty_to_qubo(qubo, &penalty_expr)?;
                    }
                    ComparisonOp::GreaterEqual => {
                        // For a >= b, equivalent to b <= a
                        let slack_name = format!("slack_{}", self.registry.num_vars);
                        let _slack_idx = self
                            .registry
                            .register_variable(&slack_name, VariableDomain::Binary);

                        // Convert to equality: (b + s - a)^2
                        let penalty_expr = Expression::BinaryOp {
                            op: BinaryOperator::Subtract,
                            left: Box::new(Expression::BinaryOp {
                                op: BinaryOperator::Add,
                                left: Box::new(right.clone()),
                                right: Box::new(Expression::Variable(slack_name)),
                            }),
                            right: Box::new(left.clone()),
                        };

                        // Add squared penalty
                        self.add_squared_penalty_to_qubo(qubo, &penalty_expr)?;
                    }
                    _ => {
                        return Err(CompileError {
                            message: format!("Unsupported comparison operator: {op:?}"),
                            context: "add_constraint_penalty".to_string(),
                        });
                    }
                }
            }
            _ => {
                return Err(CompileError {
                    message: "Complex constraints not yet supported".to_string(),
                    context: "add_constraint_penalty".to_string(),
                });
            }
        }
        Ok(())
    }

    fn add_squared_penalty_to_qubo(
        &mut self,
        qubo: &mut Array2<f64>,
        expr: &Expression,
    ) -> Result<(), CompileError> {
        // For a squared penalty (expr)^2, we expand it and add to QUBO
        // This is a simplified implementation for basic expressions
        match expr {
            Expression::Variable(name) => {
                if let Some(&idx) = self.registry.var_indices.get(name) {
                    qubo[[idx, idx]] += self.penalty_weight;
                }
            }
            Expression::BinaryOp { op, left, right } => {
                match op {
                    BinaryOperator::Add => {
                        // (a + b)^2 = a^2 + 2ab + b^2
                        self.add_squared_penalty_to_qubo(qubo, left)?;
                        self.add_squared_penalty_to_qubo(qubo, right)?;
                        self.add_cross_term_penalty(qubo, left, right, 2.0)?;
                    }
                    BinaryOperator::Subtract => {
                        // (a - b)^2 = a^2 - 2ab + b^2
                        self.add_squared_penalty_to_qubo(qubo, left)?;
                        self.add_squared_penalty_to_qubo(qubo, right)?;
                        self.add_cross_term_penalty(qubo, left, right, -2.0)?;
                    }
                    _ => {
                        return Err(CompileError {
                            message: "Complex penalty expressions not yet supported".to_string(),
                            context: "add_squared_penalty_to_qubo".to_string(),
                        });
                    }
                }
            }
            _ => {
                return Err(CompileError {
                    message: "Unsupported penalty expression type".to_string(),
                    context: "add_squared_penalty_to_qubo".to_string(),
                });
            }
        }
        Ok(())
    }

    fn add_cross_term_penalty(
        &mut self,
        qubo: &mut Array2<f64>,
        left: &Expression,
        right: &Expression,
        coefficient: f64,
    ) -> Result<(), CompileError> {
        if let (Expression::Variable(v1), Expression::Variable(v2)) = (left, right) {
            if let (Some(&idx1), Some(&idx2)) = (
                self.registry.var_indices.get(v1),
                self.registry.var_indices.get(v2),
            ) {
                let penalty = self.penalty_weight * coefficient;
                qubo[[idx1, idx2]] += penalty / 2.0;
                qubo[[idx2, idx1]] += penalty / 2.0;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem_dsl::types::VarType;

    #[test]
    fn test_simple_binary_compilation() {
        // Create a simple AST manually
        let ast = AST::Program {
            declarations: vec![
                Declaration::Variable {
                    name: "x".to_string(),
                    var_type: VarType::Binary,
                    domain: None,
                    attributes: HashMap::new(),
                },
                Declaration::Variable {
                    name: "y".to_string(),
                    var_type: VarType::Binary,
                    domain: None,
                    attributes: HashMap::new(),
                },
            ],
            objective: Objective::Minimize(Expression::BinaryOp {
                op: BinaryOperator::Add,
                left: Box::new(Expression::Variable("x".to_string())),
                right: Box::new(Expression::Variable("y".to_string())),
            }),
            constraints: vec![],
        };

        let options = CompilerOptions::default();
        let result = compile_to_qubo(&ast, &options);

        assert!(result.is_ok());
        let qubo = result.expect("compilation should succeed for valid binary program");
        assert_eq!(qubo.shape(), &[2, 2]);
        assert_eq!(qubo[[0, 0]], 1.0); // x coefficient
        assert_eq!(qubo[[1, 1]], 1.0); // y coefficient
    }

    #[test]
    fn test_quadratic_term_compilation() {
        // Test x*y term
        let ast = AST::Program {
            declarations: vec![
                Declaration::Variable {
                    name: "x".to_string(),
                    var_type: VarType::Binary,
                    domain: None,
                    attributes: HashMap::new(),
                },
                Declaration::Variable {
                    name: "y".to_string(),
                    var_type: VarType::Binary,
                    domain: None,
                    attributes: HashMap::new(),
                },
            ],
            objective: Objective::Minimize(Expression::BinaryOp {
                op: BinaryOperator::Multiply,
                left: Box::new(Expression::Variable("x".to_string())),
                right: Box::new(Expression::Variable("y".to_string())),
            }),
            constraints: vec![],
        };

        let options = CompilerOptions::default();
        let result = compile_to_qubo(&ast, &options);

        assert!(result.is_ok());
        let qubo = result.expect("compilation should succeed for quadratic term");
        assert_eq!(qubo.shape(), &[2, 2]);
        assert_eq!(qubo[[0, 1]], 0.5); // x*y coefficient (split)
        assert_eq!(qubo[[1, 0]], 0.5); // y*x coefficient (split)
    }

    #[test]
    fn test_equality_constraint() {
        // Test x == y constraint
        let ast = AST::Program {
            declarations: vec![
                Declaration::Variable {
                    name: "x".to_string(),
                    var_type: VarType::Binary,
                    domain: None,
                    attributes: HashMap::new(),
                },
                Declaration::Variable {
                    name: "y".to_string(),
                    var_type: VarType::Binary,
                    domain: None,
                    attributes: HashMap::new(),
                },
            ],
            objective: Objective::Minimize(Expression::Literal(Value::Number(0.0))),
            constraints: vec![Constraint {
                name: None,
                expression: ConstraintExpression::Comparison {
                    left: Expression::Variable("x".to_string()),
                    op: ComparisonOp::Equal,
                    right: Expression::Variable("y".to_string()),
                },
                tags: vec![],
            }],
        };

        let options = CompilerOptions::default();
        let result = compile_to_qubo(&ast, &options);

        assert!(result.is_ok());
        let qubo = result.expect("compilation should succeed for equality constraint");
        assert_eq!(qubo.shape(), &[2, 2]);
        // For x == y, penalty is (x - y)^2 = x^2 - 2xy + y^2
        assert_eq!(qubo[[0, 0]], 1000.0); // x^2 term with penalty weight
        assert_eq!(qubo[[1, 1]], 1000.0); // y^2 term with penalty weight
        assert_eq!(qubo[[0, 1]], -1000.0); // -xy term
        assert_eq!(qubo[[1, 0]], -1000.0); // -yx term
    }
}
