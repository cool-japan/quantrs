//! Problem modeling DSL for quantum optimization.
//!
//! This module provides a domain-specific language for defining
//! optimization problems in a high-level, intuitive way.
//!
//! The module has been refactored into submodules for better organization:
//! - `lexer`: Tokenization and lexical analysis
//! - `parser`: Parsing tokens into AST
//! - `ast`: Abstract syntax tree definitions
//! - `types`: Type system and type checking
//! - `compiler`: Compilation to QUBO/Ising models
//! - `stdlib`: Standard library functions
//! - `macros`: Macro system
//! - `imports`: Import resolution
//! - `optimizer`: Optimization hints
//! - `examples`: Example problems

// Re-export everything from the new modular structure for backward compatibility
mod problem_dsl;
pub use problem_dsl::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer() {
        let mut dsl = ProblemDSL::new();
        let source = "var x binary;";
        let tokens = dsl.tokenize(source);

        assert!(tokens.is_ok());
        let tokens = tokens.expect("Tokenization should succeed for valid source");
        assert_eq!(tokens.len(), 5); // var, x, binary, ;, EOF
    }

    #[test]
    fn test_parser() {
        let mut dsl = ProblemDSL::new();
        let source = examples::get_example("simple_binary")
            .expect("simple_binary example should exist");
        let ast = dsl.parse(source);

        assert!(ast.is_ok());
    }

    #[test]
    fn test_examples() {
        assert!(examples::get_example("simple_binary").is_some());
        assert!(examples::get_example("tsp").is_some());
        assert!(examples::get_example("graph_coloring").is_some());
        assert!(examples::get_example("nonexistent").is_none());
    }
}