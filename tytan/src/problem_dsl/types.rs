//! Type system for the problem DSL.

use super::ast::AST;
use super::error::TypeError;
use std::collections::HashMap;

/// Variable types
#[derive(Debug, Clone, PartialEq)]
pub enum VarType {
    Binary,
    Integer,
    Continuous,
    Spin,
    Array {
        element_type: Box<Self>,
        dimensions: Vec<usize>,
    },
    Matrix {
        element_type: Box<Self>,
        rows: usize,
        cols: usize,
    },
}

/// Type checker
#[derive(Debug, Clone)]
pub struct TypeChecker {
    /// Variable types
    var_types: HashMap<String, VarType>,
    /// Function signatures
    func_signatures: HashMap<String, FunctionSignature>,
    /// Type errors
    errors: Vec<TypeError>,
}

/// Function signature
#[derive(Debug, Clone)]
pub struct FunctionSignature {
    pub param_types: Vec<VarType>,
    pub return_type: VarType,
}

impl TypeChecker {
    /// Create a new type checker
    pub fn new() -> Self {
        let mut checker = Self {
            var_types: HashMap::new(),
            func_signatures: HashMap::new(),
            errors: Vec::new(),
        };

        // Register built-in functions
        checker.register_builtin_functions();
        checker
    }

    /// Register built-in function signatures
    fn register_builtin_functions(&mut self) {
        // Mathematical functions
        self.func_signatures.insert(
            "abs".to_string(),
            FunctionSignature {
                param_types: vec![VarType::Continuous],
                return_type: VarType::Continuous,
            },
        );

        self.func_signatures.insert(
            "sqrt".to_string(),
            FunctionSignature {
                param_types: vec![VarType::Continuous],
                return_type: VarType::Continuous,
            },
        );

        self.func_signatures.insert(
            "exp".to_string(),
            FunctionSignature {
                param_types: vec![VarType::Continuous],
                return_type: VarType::Continuous,
            },
        );

        self.func_signatures.insert(
            "log".to_string(),
            FunctionSignature {
                param_types: vec![VarType::Continuous],
                return_type: VarType::Continuous,
            },
        );

        // Aggregation functions
        self.func_signatures.insert(
            "sum".to_string(),
            FunctionSignature {
                param_types: vec![VarType::Array {
                    element_type: Box::new(VarType::Continuous),
                    dimensions: vec![0],
                }],
                return_type: VarType::Continuous,
            },
        );

        self.func_signatures.insert(
            "product".to_string(),
            FunctionSignature {
                param_types: vec![VarType::Array {
                    element_type: Box::new(VarType::Continuous),
                    dimensions: vec![0],
                }],
                return_type: VarType::Continuous,
            },
        );

        self.func_signatures.insert(
            "min".to_string(),
            FunctionSignature {
                param_types: vec![VarType::Array {
                    element_type: Box::new(VarType::Continuous),
                    dimensions: vec![0],
                }],
                return_type: VarType::Continuous,
            },
        );

        self.func_signatures.insert(
            "max".to_string(),
            FunctionSignature {
                param_types: vec![VarType::Array {
                    element_type: Box::new(VarType::Continuous),
                    dimensions: vec![0],
                }],
                return_type: VarType::Continuous,
            },
        );
    }

    /// Type check an AST
    pub fn check(&mut self, ast: &AST) -> Result<(), TypeError> {
        self.errors.clear();
        self.check_ast(ast);

        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(self.errors[0].clone())
        }
    }

    /// Check AST node
    fn check_ast(&mut self, ast: &AST) {
        match ast {
            AST::Program {
                declarations,
                objective,
                constraints,
            } => {
                // Check declarations first to build symbol table
                for decl in declarations {
                    self.check_declaration(decl);
                }

                // Check objective
                self.check_objective(objective);

                // Check constraints
                for constraint in constraints {
                    self.check_constraint(constraint);
                }
            }
            AST::VarDecl { name, var_type, .. } => {
                self.var_types.insert(name.clone(), var_type.clone());
            }
            AST::Expr(expr) => {
                self.check_expression(expr);
            }
            AST::Stmt(stmt) => {
                self.check_statement(stmt);
            }
        }
    }

    /// Check declaration
    fn check_declaration(&mut self, decl: &super::ast::Declaration) {
        match decl {
            super::ast::Declaration::Variable { name, var_type, .. } => {
                self.var_types.insert(name.clone(), var_type.clone());
            }
            super::ast::Declaration::Parameter { name, value, .. } => {
                let value_type = self.infer_value_type(value);
                self.var_types.insert(name.clone(), value_type);
            }
            super::ast::Declaration::Set { name, elements } => {
                if !elements.is_empty() {
                    let element_type = self.infer_value_type(&elements[0]);
                    let array_type = VarType::Array {
                        element_type: Box::new(element_type),
                        dimensions: vec![elements.len()],
                    };
                    self.var_types.insert(name.clone(), array_type);
                }
            }
            super::ast::Declaration::Function { name, params, body } => {
                // For now, assume functions return continuous values
                let param_types = params.iter().map(|_| VarType::Continuous).collect();
                let signature = FunctionSignature {
                    param_types,
                    return_type: VarType::Continuous,
                };
                self.func_signatures.insert(name.clone(), signature);

                // Check function body
                self.check_expression(body);
            }
        }
    }

    /// Check objective
    fn check_objective(&mut self, obj: &super::ast::Objective) {
        match obj {
            super::ast::Objective::Minimize(expr) | super::ast::Objective::Maximize(expr) => {
                self.check_expression(expr);
            }
            super::ast::Objective::MultiObjective { objectives } => {
                for (_, expr, _) in objectives {
                    self.check_expression(expr);
                }
            }
        }
    }

    /// Check constraint
    fn check_constraint(&mut self, constraint: &super::ast::Constraint) {
        self.check_constraint_expression(&constraint.expression);
    }

    /// Check constraint expression
    fn check_constraint_expression(&mut self, expr: &super::ast::ConstraintExpression) {
        match expr {
            super::ast::ConstraintExpression::Comparison { left, right, .. } => {
                self.check_expression(left);
                self.check_expression(right);
            }
            super::ast::ConstraintExpression::Logical { operands, .. } => {
                for operand in operands {
                    self.check_constraint_expression(operand);
                }
            }
            super::ast::ConstraintExpression::Quantified { constraint, .. } => {
                self.check_constraint_expression(constraint);
            }
            super::ast::ConstraintExpression::Implication {
                condition,
                consequence,
            } => {
                self.check_constraint_expression(condition);
                self.check_constraint_expression(consequence);
            }
            super::ast::ConstraintExpression::Counting { count, .. } => {
                self.check_expression(count);
            }
        }
    }

    /// Check expression
    fn check_expression(&mut self, expr: &super::ast::Expression) {
        match expr {
            super::ast::Expression::Literal(_) => {
                // Literals are always valid
            }
            super::ast::Expression::Variable(name) => {
                if !self.var_types.contains_key(name) {
                    self.errors.push(TypeError {
                        message: format!("Undefined variable: {name}"),
                        location: name.clone(),
                    });
                }
            }
            super::ast::Expression::IndexedVar { name, indices } => {
                if !self.var_types.contains_key(name) {
                    self.errors.push(TypeError {
                        message: format!("Undefined variable: {name}"),
                        location: name.clone(),
                    });
                }
                for index in indices {
                    self.check_expression(index);
                }
            }
            super::ast::Expression::BinaryOp { left, right, .. } => {
                self.check_expression(left);
                self.check_expression(right);
            }
            super::ast::Expression::UnaryOp { operand, .. } => {
                self.check_expression(operand);
            }
            super::ast::Expression::FunctionCall { name, args } => {
                if let Some(signature) = self.func_signatures.get(name) {
                    if args.len() != signature.param_types.len() {
                        self.errors.push(TypeError {
                            message: format!(
                                "Function {} expects {} arguments, got {}",
                                name,
                                signature.param_types.len(),
                                args.len()
                            ),
                            location: name.clone(),
                        });
                    }
                } else {
                    self.errors.push(TypeError {
                        message: format!("Undefined function: {name}"),
                        location: name.clone(),
                    });
                }

                for arg in args {
                    self.check_expression(arg);
                }
            }
            super::ast::Expression::Aggregation { expression, .. } => {
                self.check_expression(expression);
            }
            super::ast::Expression::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                self.check_constraint_expression(condition);
                self.check_expression(then_expr);
                self.check_expression(else_expr);
            }
        }
    }

    /// Check statement
    fn check_statement(&mut self, stmt: &super::ast::Statement) {
        match stmt {
            super::ast::Statement::Assignment { target, value } => {
                if !self.var_types.contains_key(target) {
                    self.errors.push(TypeError {
                        message: format!("Undefined variable: {target}"),
                        location: target.clone(),
                    });
                }
                self.check_expression(value);
            }
            super::ast::Statement::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.check_constraint_expression(condition);
                for stmt in then_branch {
                    self.check_statement(stmt);
                }
                if let Some(else_stmts) = else_branch {
                    for stmt in else_stmts {
                        self.check_statement(stmt);
                    }
                }
            }
            super::ast::Statement::For { body, .. } => {
                for stmt in body {
                    self.check_statement(stmt);
                }
            }
        }
    }

    /// Infer type from value
    fn infer_value_type(&self, value: &super::ast::Value) -> VarType {
        match value {
            super::ast::Value::Number(_) => VarType::Continuous,
            super::ast::Value::Boolean(_) => VarType::Binary,
            super::ast::Value::String(_) => VarType::Continuous, // Treat as parameter
            super::ast::Value::Array(elements) => {
                if elements.is_empty() {
                    VarType::Array {
                        element_type: Box::new(VarType::Continuous),
                        dimensions: vec![0],
                    }
                } else {
                    let element_type = self.infer_value_type(&elements[0]);
                    VarType::Array {
                        element_type: Box::new(element_type),
                        dimensions: vec![elements.len()],
                    }
                }
            }
            super::ast::Value::Tuple(elements) => {
                if elements.is_empty() {
                    VarType::Array {
                        element_type: Box::new(VarType::Continuous),
                        dimensions: vec![0],
                    }
                } else {
                    let element_type = self.infer_value_type(&elements[0]);
                    VarType::Array {
                        element_type: Box::new(element_type),
                        dimensions: vec![elements.len()],
                    }
                }
            }
        }
    }

    /// Get variable type
    pub fn get_var_type(&self, name: &str) -> Option<&VarType> {
        self.var_types.get(name)
    }

    /// Get function signature
    pub fn get_function_signature(&self, name: &str) -> Option<&FunctionSignature> {
        self.func_signatures.get(name)
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}
