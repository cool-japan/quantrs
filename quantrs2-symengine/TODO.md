# quantrs2-symengine TODO

## High Priority

### Core Features
- [ ] **Substitution System** - Implement proper substitution without ExprMap dependency
  - [ ] Add `substitute` method to Expression
  - [ ] Support multiple simultaneous substitutions
  - [ ] Add substitution with numeric values
  - [ ] Add substitution with expressions

- [ ] **Expression Simplification**
  - [ ] Add `simplify()` method using SymEngine's simplification
  - [ ] Add `trigsimp()` for trigonometric simplification
  - [ ] Add `ratsimp()` for rational expression simplification
  - [ ] Add `powsimp()` for power simplification

- [ ] **Polynomial Operations**
  - [ ] Add `collect()` to collect terms by symbol
  - [ ] Add `expand_trig()` for trigonometric expansion
  - [ ] Add `expand_complex()` for complex number expansion
  - [ ] Add `factor()` using SymEngine's factorization
  - [ ] Add polynomial degree calculation
  - [ ] Add coefficient extraction

- [ ] **Equation Solving**
  - [ ] Add `solve()` for algebraic equations
  - [ ] Add `solve_linear_system()` for linear systems
  - [ ] Add `solve_poly()` for polynomial equations
  - [ ] Add numeric root finding

### SciRS2 Integration

- [ ] **Complex Number Support**
  - [ ] Convert Expression to/from `scirs2_core::Complex64`
  - [ ] Support complex arithmetic in symbolic expressions
  - [ ] Add `re()` and `im()` for real/imaginary parts
  - [ ] Add `conjugate()` operation

- [ ] **Array Integration**
  - [ ] Convert symbolic matrices to `scirs2_core::ndarray::Array2`
  - [ ] Evaluate symbolic arrays numerically
  - [ ] Support broadcasting operations

- [ ] **Numeric Traits**
  - [ ] Implement more `num_traits` traits for Expression
  - [ ] Add `NumCast` for type conversions
  - [ ] Add `Float` trait partial implementation
  - [ ] Add `Signed` trait implementation

### Quantum Computing Support

- [ ] **Quantum-Specific Functions**
  - [ ] Add Pauli matrix symbolic representations (σx, σy, σz)
  - [ ] Add quantum gate symbolic operations (H, CNOT, etc.)
  - [ ] Add commutator and anticommutator operations
  - [ ] Add unitary matrix verification
  - [ ] Add Hermitian matrix verification
  - [ ] Add trace operations for density matrices

- [ ] **Quantum Operator Algebra**
  - [ ] Support for creation/annihilation operators
  - [ ] Ladder operator algebra
  - [ ] Spin operator algebra
  - [ ] Angular momentum operators

### Performance & Optimization

- [ ] **Caching & Memoization**
  - [ ] Cache expanded/simplified expressions
  - [ ] Memoize expensive operations
  - [ ] Add lazy evaluation support

- [ ] **SIMD Support**
  - [ ] Vectorized evaluation of symbolic expressions
  - [ ] Batch evaluation for arrays of values
  - [ ] Integration with scirs2_core::simd_ops

- [ ] **Parallel Evaluation**
  - [ ] Parallel symbolic differentiation
  - [ ] Parallel matrix operations
  - [ ] Integration with scirs2_core::parallel_ops

## Medium Priority

### Expression Manipulation

- [ ] **Pattern Matching**
  - [ ] Add pattern matching for expression trees
  - [ ] Support wildcard patterns
  - [ ] Add replacement rules

- [ ] **Expression Analysis**
  - [ ] Add `free_symbols()` to get all symbols
  - [ ] Add `is_polynomial()` check
  - [ ] Add `is_rational()` check
  - [ ] Add expression complexity measures

- [ ] **Rewriting**
  - [ ] Add `rewrite()` for expression transformation
  - [ ] Support different basis functions (exp, trig, etc.)
  - [ ] Add normal forms (CNF, DNF for logic)

### Calculus Enhancements

- [ ] **Advanced Differentiation**
  - [ ] Gradient computation for multivariate functions
  - [ ] Hessian matrix computation
  - [ ] Jacobian matrix computation
  - [ ] Total derivatives

- [ ] **Integration Enhancements**
  - [ ] Multiple integration
  - [ ] Line integrals
  - [ ] Surface integrals
  - [ ] Contour integration

- [ ] **Differential Equations**
  - [ ] ODE solving (symbolic)
  - [ ] PDE support
  - [ ] Initial value problems
  - [ ] Boundary value problems

### Linear Algebra Enhancements

- [ ] **Matrix Operations**
  - [ ] Eigenvalues/eigenvectors (symbolic)
  - [ ] Matrix diagonalization
  - [ ] SVD (symbolic)
  - [ ] QR decomposition
  - [ ] LU decomposition
  - [ ] Cholesky decomposition

- [ ] **Vector Operations**
  - [ ] Dot product
  - [ ] Cross product
  - [ ] Gram-Schmidt orthogonalization
  - [ ] Vector projection

### Testing & Quality

- [ ] **Unit Tests**
  - [ ] Comprehensive Expression tests
  - [ ] All mathematical operations tests
  - [ ] Edge case coverage
  - [ ] Error handling tests

- [ ] **Integration Tests**
  - [ ] SciRS2 integration tests
  - [ ] Quantum computing workflow tests
  - [ ] Performance regression tests

- [ ] **Benchmarks**
  - [ ] Expression creation benchmarks
  - [ ] Arithmetic operation benchmarks
  - [ ] Simplification benchmarks
  - [ ] Comparison with other symbolic libraries

- [ ] **Property-Based Tests**
  - [ ] Algebraic properties (associativity, commutativity, etc.)
  - [ ] Calculus identities
  - [ ] Trigonometric identities

## Low Priority

### Documentation

- [ ] **Examples**
  - [ ] Basic symbolic computation examples
  - [ ] Quantum computing examples
  - [ ] Calculus examples
  - [ ] Linear algebra examples
  - [ ] Integration with QuantRS2 examples

- [ ] **API Documentation**
  - [ ] Complete docstrings for all public items
  - [ ] Mathematical notation in docs
  - [ ] Performance notes
  - [ ] Usage patterns

- [ ] **Tutorials**
  - [ ] Getting started guide
  - [ ] Advanced features guide
  - [ ] Performance optimization guide
  - [ ] Quantum computing integration guide

### Additional Features

- [ ] **Serialization**
  - [ ] Optimize serde implementation
  - [ ] Custom binary format for performance
  - [ ] MathML export
  - [ ] LaTeX export

- [ ] **Printing & Formatting**
  - [ ] Pretty printing with Unicode
  - [ ] LaTeX rendering
  - [ ] MathML rendering
  - [ ] Custom formatting options

- [ ] **Logic & Boolean Algebra**
  - [ ] Boolean expressions
  - [ ] Logic operators (AND, OR, NOT, etc.)
  - [ ] Truth tables
  - [ ] SAT solving integration

- [ ] **Set Theory**
  - [ ] Set operations
  - [ ] Interval arithmetic
  - [ ] Relation operations

### Infrastructure

- [ ] **Build System**
  - [ ] Optimize build configuration
  - [ ] Feature flags for optional components
  - [ ] Cross-platform testing

- [ ] **CI/CD**
  - [ ] Automated testing
  - [ ] Documentation generation
  - [ ] Benchmark tracking

## Notes

### Design Principles
1. **SciRS2 First**: Always use SciRS2 abstractions (scirs2_core::num_traits, etc.)
2. **Zero-Cost**: Leverage Rust's zero-cost abstractions
3. **Type Safety**: Use Rust's type system for mathematical correctness
4. **Performance**: Optimize for QuantRS2's quantum computing workloads
5. **Ergonomics**: Provide intuitive, Rust-idiomatic APIs

### Current Limitations
- ExprMap is disabled due to missing C wrapper functions in symengine-sys
- Some advanced SymEngine features not yet exposed
- Limited complex number support in current implementation

### Integration Points
- **quantrs2-core**: Quantum types and traits
- **quantrs2-circuit**: Symbolic circuit optimization
- **quantrs2-ml**: Symbolic gradient computation for QML
- **scirs2-core**: Numeric types and array operations
- **scirs2-linalg**: Matrix operations
- **scirs2-autograd**: Automatic differentiation
