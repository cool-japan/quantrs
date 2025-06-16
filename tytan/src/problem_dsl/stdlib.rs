//! Standard library for the problem DSL.

use super::ast::{Expression, Value};
use super::types::{FunctionSignature, VarType};
use std::collections::HashMap;

/// Standard library
#[derive(Debug, Clone)]
pub struct StandardLibrary {
    /// Built-in functions
    functions: HashMap<String, BuiltinFunction>,
    /// Common patterns
    patterns: HashMap<String, Pattern>,
    /// Problem templates
    templates: HashMap<String, Template>,
}

/// Built-in function
#[derive(Debug, Clone)]
pub struct BuiltinFunction {
    pub name: String,
    pub signature: FunctionSignature,
    pub description: String,
    pub implementation: FunctionImpl,
}

/// Function implementation
#[derive(Debug, Clone)]
pub enum FunctionImpl {
    /// Native Rust implementation
    Native,
    /// DSL implementation
    DSL { body: Expression },
}

/// Common pattern
#[derive(Debug, Clone)]
pub struct Pattern {
    pub name: String,
    pub description: String,
    pub parameters: Vec<String>,
    pub expansion: super::ast::AST,
}

/// Problem template
#[derive(Debug, Clone)]
pub struct Template {
    pub name: String,
    pub description: String,
    pub parameters: Vec<TemplateParam>,
    pub body: String,
}

#[derive(Debug, Clone)]
pub struct TemplateParam {
    pub name: String,
    pub param_type: String,
    pub default: Option<Value>,
}

impl StandardLibrary {
    /// Create a new standard library
    pub fn new() -> Self {
        let mut stdlib = Self {
            functions: HashMap::new(),
            patterns: HashMap::new(),
            templates: HashMap::new(),
        };

        stdlib.register_builtin_functions();
        stdlib.register_common_patterns();
        stdlib.register_templates();

        stdlib
    }

    /// Register built-in functions
    fn register_builtin_functions(&mut self) {
        // Mathematical functions
        self.functions.insert(
            "abs".to_string(),
            BuiltinFunction {
                name: "abs".to_string(),
                signature: FunctionSignature {
                    param_types: vec![VarType::Continuous],
                    return_type: VarType::Continuous,
                },
                description: "Absolute value function".to_string(),
                implementation: FunctionImpl::Native,
            },
        );

        self.functions.insert(
            "sqrt".to_string(),
            BuiltinFunction {
                name: "sqrt".to_string(),
                signature: FunctionSignature {
                    param_types: vec![VarType::Continuous],
                    return_type: VarType::Continuous,
                },
                description: "Square root function".to_string(),
                implementation: FunctionImpl::Native,
            },
        );

        // Aggregation functions
        self.functions.insert(
            "sum".to_string(),
            BuiltinFunction {
                name: "sum".to_string(),
                signature: FunctionSignature {
                    param_types: vec![VarType::Array {
                        element_type: Box::new(VarType::Continuous),
                        dimensions: vec![0],
                    }],
                    return_type: VarType::Continuous,
                },
                description: "Sum aggregation function".to_string(),
                implementation: FunctionImpl::Native,
            },
        );
    }

    /// Register common patterns
    fn register_common_patterns(&mut self) {
        // All-different constraint pattern
        // Pattern for ensuring variables take different values
        self.patterns.insert(
            "all_different".to_string(),
            Pattern {
                name: "all_different".to_string(),
                description: "Ensures all variables in a set take different values".to_string(),
                parameters: vec!["variables".to_string()],
                expansion: super::ast::AST::Program {
                    declarations: vec![],
                    objective: super::ast::Objective::Minimize(super::ast::Expression::Literal(
                        super::ast::Value::Number(0.0),
                    )),
                    constraints: vec![], // Would be filled with actual constraints during expansion
                },
            },
        );

        // Cardinality constraint pattern
        self.patterns.insert(
            "cardinality".to_string(),
            Pattern {
                name: "cardinality".to_string(),
                description: "Constrains the number of true variables in a set".to_string(),
                parameters: vec![
                    "variables".to_string(),
                    "min_count".to_string(),
                    "max_count".to_string(),
                ],
                expansion: super::ast::AST::Program {
                    declarations: vec![],
                    objective: super::ast::Objective::Minimize(super::ast::Expression::Literal(
                        super::ast::Value::Number(0.0),
                    )),
                    constraints: vec![],
                },
            },
        );

        // At-most-one constraint pattern
        self.patterns.insert(
            "at_most_one".to_string(),
            Pattern {
                name: "at_most_one".to_string(),
                description: "Ensures at most one variable in a set is true".to_string(),
                parameters: vec!["variables".to_string()],
                expansion: super::ast::AST::Program {
                    declarations: vec![],
                    objective: super::ast::Objective::Minimize(super::ast::Expression::Literal(
                        super::ast::Value::Number(0.0),
                    )),
                    constraints: vec![],
                },
            },
        );

        // Exactly-one constraint pattern
        self.patterns.insert(
            "exactly_one".to_string(),
            Pattern {
                name: "exactly_one".to_string(),
                description: "Ensures exactly one variable in a set is true".to_string(),
                parameters: vec!["variables".to_string()],
                expansion: super::ast::AST::Program {
                    declarations: vec![],
                    objective: super::ast::Objective::Minimize(super::ast::Expression::Literal(
                        super::ast::Value::Number(0.0),
                    )),
                    constraints: vec![],
                },
            },
        );
    }

    /// Register problem templates
    fn register_templates(&mut self) {
        // Traveling Salesman Problem template
        self.templates.insert(
            "tsp".to_string(),
            Template {
                name: "tsp".to_string(),
                description: "Traveling Salesman Problem template".to_string(),
                parameters: vec![
                    TemplateParam {
                        name: "n_cities".to_string(),
                        param_type: "integer".to_string(),
                        default: Some(Value::Number(4.0)),
                    },
                    TemplateParam {
                        name: "distance_matrix".to_string(),
                        param_type: "matrix".to_string(),
                        default: None,
                    },
                ],
                body: r#"
                    param n = {n_cities};
                    param distances = {distance_matrix};
                    
                    var x[n, n] binary;
                    
                    minimize sum(i in 0..n, j in 0..n: distances[i][j] * x[i,j]);
                    
                    subject to
                        forall(i in 0..n): sum(j in 0..n: x[i,j]) == 1;
                        forall(j in 0..n): sum(i in 0..n: x[i,j]) == 1;
                "#
                .to_string(),
            },
        );

        // Graph Coloring template
        self.templates.insert(
            "graph_coloring".to_string(),
            Template {
                name: "graph_coloring".to_string(),
                description: "Graph coloring problem template".to_string(),
                parameters: vec![
                    TemplateParam {
                        name: "n_vertices".to_string(),
                        param_type: "integer".to_string(),
                        default: Some(Value::Number(5.0)),
                    },
                    TemplateParam {
                        name: "n_colors".to_string(),
                        param_type: "integer".to_string(),
                        default: Some(Value::Number(3.0)),
                    },
                    TemplateParam {
                        name: "edges".to_string(),
                        param_type: "array".to_string(),
                        default: None,
                    },
                ],
                body: r#"
                    param n_vertices = {n_vertices};
                    param n_colors = {n_colors};
                    param edges = {edges};
                    
                    var color[n_vertices, n_colors] binary;
                    
                    minimize sum(v in 0..n_vertices, c in 0..n_colors: c * color[v,c]);
                    
                    subject to
                        forall(v in 0..n_vertices): sum(c in 0..n_colors: color[v,c]) == 1;
                        forall((u,v) in edges, c in 0..n_colors): color[u,c] + color[v,c] <= 1;
                "#
                .to_string(),
            },
        );

        // Knapsack Problem template
        self.templates.insert(
            "knapsack".to_string(),
            Template {
                name: "knapsack".to_string(),
                description: "0-1 Knapsack problem template".to_string(),
                parameters: vec![
                    TemplateParam {
                        name: "n_items".to_string(),
                        param_type: "integer".to_string(),
                        default: Some(Value::Number(10.0)),
                    },
                    TemplateParam {
                        name: "weights".to_string(),
                        param_type: "array".to_string(),
                        default: None,
                    },
                    TemplateParam {
                        name: "values".to_string(),
                        param_type: "array".to_string(),
                        default: None,
                    },
                    TemplateParam {
                        name: "capacity".to_string(),
                        param_type: "number".to_string(),
                        default: Some(Value::Number(100.0)),
                    },
                ],
                body: r#"
                    param n = {n_items};
                    param weights = {weights};
                    param values = {values};
                    param capacity = {capacity};
                    
                    var x[n] binary;
                    
                    maximize sum(i in 0..n: values[i] * x[i]);
                    
                    subject to
                        sum(i in 0..n: weights[i] * x[i]) <= capacity;
                "#
                .to_string(),
            },
        );

        // Maximum Cut template
        self.templates.insert(
            "max_cut".to_string(),
            Template {
                name: "max_cut".to_string(),
                description: "Maximum cut problem template".to_string(),
                parameters: vec![
                    TemplateParam {
                        name: "n_vertices".to_string(),
                        param_type: "integer".to_string(),
                        default: Some(Value::Number(6.0)),
                    },
                    TemplateParam {
                        name: "edges".to_string(),
                        param_type: "array".to_string(),
                        default: None,
                    },
                    TemplateParam {
                        name: "weights".to_string(),
                        param_type: "array".to_string(),
                        default: None,
                    },
                ],
                body: r#"
                    param n = {n_vertices};
                    param edges = {edges};
                    param weights = {weights};
                    
                    var x[n] binary;
                    
                    maximize sum((i,j,w) in zip(edges, weights): w * (x[i] + x[j] - 2*x[i]*x[j]));
                "#
                .to_string(),
            },
        );
    }

    /// Get function by name
    pub fn get_function(&self, name: &str) -> Option<&BuiltinFunction> {
        self.functions.get(name)
    }

    /// Get pattern by name
    pub fn get_pattern(&self, name: &str) -> Option<&Pattern> {
        self.patterns.get(name)
    }

    /// Get template by name
    pub fn get_template(&self, name: &str) -> Option<&Template> {
        self.templates.get(name)
    }
}

impl Default for StandardLibrary {
    fn default() -> Self {
        Self::new()
    }
}
