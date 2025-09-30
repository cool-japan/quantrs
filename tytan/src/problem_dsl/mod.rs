//! Problem modeling DSL for quantum optimization.
//!
//! This module provides a domain-specific language for defining
//! optimization problems in a high-level, intuitive way.

pub mod ast;
pub mod compiler;
pub mod error;
pub mod examples;
pub mod imports;
pub mod lexer;
pub mod macros;
pub mod optimizer;
pub mod parser;
pub mod stdlib;
pub mod types;

#[cfg(feature = "dwave")]
use crate::compile::{Compile, CompiledModel};
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;

// Re-export main types
pub use ast::*;
pub use compiler::*;
pub use error::*;
pub use imports::*;
pub use lexer::*;
pub use macros::*;
pub use optimizer::*;
pub use parser::*;
pub use stdlib::*;
pub use types::*;

/// DSL parser for optimization problems
pub struct ProblemDSL {
    /// Parser state
    parser: parser::Parser,
    /// Type checker
    type_checker: types::TypeChecker,
    /// Standard library
    stdlib: stdlib::StandardLibrary,
    /// Compiler options
    options: compiler::CompilerOptions,
    /// Macro definitions
    macros: HashMap<String, macros::Macro>,
    /// Import resolver
    import_resolver: imports::ImportResolver,
    /// Optimization hints
    optimization_hints: Vec<optimizer::OptimizationHint>,
}

impl ProblemDSL {
    /// Create a new DSL instance
    pub fn new() -> Self {
        Self {
            parser: parser::Parser::new(),
            type_checker: types::TypeChecker::new(),
            stdlib: stdlib::StandardLibrary::new(),
            options: compiler::CompilerOptions::default(),
            macros: HashMap::new(),
            import_resolver: imports::ImportResolver::new(),
            optimization_hints: Vec::new(),
        }
    }

    /// Set compiler options
    pub fn with_options(mut self, options: compiler::CompilerOptions) -> Self {
        self.options = options;
        self
    }

    /// Add optimization hints
    pub fn with_hints(mut self, hints: Vec<optimizer::OptimizationHint>) -> Self {
        self.optimization_hints = hints;
        self
    }

    /// Parse DSL source code
    pub fn parse(&mut self, source: &str) -> Result<ast::AST, error::ParseError> {
        // Tokenize
        let tokens = self.tokenize(source)?;

        // Parse
        self.parser.set_tokens(tokens);
        let ast = self.parser.parse()?;

        // Type check
        self.type_checker.check(&ast)?;

        Ok(ast)
    }

    /// Tokenize source code
    pub fn tokenize(&self, source: &str) -> Result<Vec<lexer::Token>, error::ParseError> {
        lexer::tokenize(source)
    }

    /// Compile AST to QUBO
    pub fn compile_to_qubo(&self, ast: &ast::AST) -> Result<Array2<f64>, error::CompileError> {
        compiler::compile_to_qubo(ast, &self.options)
    }

    /// Generate example problems
    pub fn example<'a>(&self, name: &'a str) -> Option<&'a str> {
        examples::get_example(name)
    }
}

impl Default for ProblemDSL {
    fn default() -> Self {
        Self::new()
    }
}
