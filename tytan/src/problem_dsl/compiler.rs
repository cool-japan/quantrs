//! Compiler for the problem DSL.

use super::ast::{AST, Declaration, Expression, Constraint, ConstraintExpression, Objective, ObjectiveType, ComparisonOp, BinaryOperator, UnaryOperator, AggregationOp, Value};
use super::error::CompileError;
use ndarray::Array2;
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

    fn register_indexed_variable(&mut self, base_name: &str, indices: Vec<usize>, domain: VariableDomain) -> usize {
        let indexed_map = self.indexed_var_indices.entry(base_name.to_string()).or_insert_with(HashMap::new);
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
            Declaration::Variable { name, var_type, domain, attributes } => {
                // For now, assume all variables are binary
                self.registry.register_variable(name, VariableDomain::Binary);
                Ok(())
            }
            Declaration::Parameter { name, value, .. } => {
                self.parameters.insert(name.clone(), value.clone());
                Ok(())
            }
            _ => Ok(()), // TODO: Handle other declaration types
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

    fn add_expression_to_qubo(&mut self, qubo: &mut Array2<f64>, expr: &Expression, coefficient: f64) -> Result<(), CompileError> {
        match expr {
            Expression::Variable(name) => {
                if let Some(&idx) = self.registry.var_indices.get(name) {
                    qubo[[idx, idx]] += coefficient;
                } else {
                    return Err(CompileError {
                        message: format!("Unknown variable: {}", name),
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
                        if let (Expression::Variable(v1), Expression::Variable(v2)) = (left.as_ref(), right.as_ref()) {
                            if let (Some(&idx1), Some(&idx2)) = (self.registry.var_indices.get(v1), self.registry.var_indices.get(v2)) {
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
                            message: format!("Unsupported binary operator: {:?}", op),
                            context: "add_expression_to_qubo".to_string(),
                        });
                    }
                }
            }
            Expression::Literal(Value::Number(n)) => {
                // Constants don't affect the optimization, but we could track them
                // for the objective value offset
            }
            Expression::Aggregation { op, variables, expression } => {
                match op {
                    AggregationOp::Sum => {
                        // TODO: Expand sum over index sets
                        // For now, just a placeholder
                        return Err(CompileError {
                            message: "Aggregation not yet fully implemented".to_string(),
                            context: "add_expression_to_qubo".to_string(),
                        });
                    }
                    _ => {
                        return Err(CompileError {
                            message: format!("Unsupported aggregation operator: {:?}", op),
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

    fn add_constraint_penalty(&mut self, qubo: &mut Array2<f64>, constraint: &Constraint) -> Result<(), CompileError> {
        match &constraint.expression {
            ConstraintExpression::Comparison { left, op, right } => {
                match op {
                    ComparisonOp::Equal => {
                        // For equality constraint: (left - right)^2
                        // Expand: left^2 - 2*left*right + right^2
                        self.add_expression_to_qubo(qubo, left, self.penalty_weight)?;
                        self.add_expression_to_qubo(qubo, right, self.penalty_weight)?;
                        
                        // Cross term: -2*left*right
                        if let (Expression::Variable(v1), Expression::Variable(v2)) = (left, right) {
                            if let (Some(&idx1), Some(&idx2)) = (self.registry.var_indices.get(v1), self.registry.var_indices.get(v2)) {
                                qubo[[idx1, idx2]] -= self.penalty_weight;
                                qubo[[idx2, idx1]] -= self.penalty_weight;
                            }
                        }
                    }
                    ComparisonOp::LessEqual | ComparisonOp::GreaterEqual => {
                        // Inequality constraints require slack variables
                        // TODO: Implement slack variable handling
                        return Err(CompileError {
                            message: "Inequality constraints not yet fully implemented".to_string(),
                            context: "add_constraint_penalty".to_string(),
                        });
                    }
                    _ => {
                        return Err(CompileError {
                            message: format!("Unsupported comparison operator: {:?}", op),
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem_dsl::parser::Parser;
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
        let qubo = result.unwrap();
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
        let qubo = result.unwrap();
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
            constraints: vec![
                Constraint {
                    name: None,
                    expression: ConstraintExpression::Comparison {
                        left: Expression::Variable("x".to_string()),
                        op: ComparisonOp::Equal,
                        right: Expression::Variable("y".to_string()),
                    },
                    tags: vec![],
                },
            ],
        };

        let options = CompilerOptions::default();
        let result = compile_to_qubo(&ast, &options);
        
        assert!(result.is_ok());
        let qubo = result.unwrap();
        assert_eq!(qubo.shape(), &[2, 2]);
        // For x == y, penalty is (x - y)^2 = x^2 - 2xy + y^2
        assert_eq!(qubo[[0, 0]], 1000.0); // x^2 term with penalty weight
        assert_eq!(qubo[[1, 1]], 1000.0); // y^2 term with penalty weight
        assert_eq!(qubo[[0, 1]], -1000.0); // -xy term
        assert_eq!(qubo[[1, 0]], -1000.0); // -yx term
    }
}
