# quantrs2-symengine-sys TODO

## Completed ✅

### v0.1.0-beta.3 Enhancements (2025-11-23)

#### Core FFI Bindings
- [x] CMapBasicBasic (ExprMap) FFI bindings for substitution
- [x] Dense matrix operations (30+ functions)
- [x] Complex number operations (real/imag parts, conjugate)
- [x] Special mathematical functions (atan2, beta, gamma, etc.)
- [x] Number theory functions (GCD, LCM, factorial, Fibonacci, etc.)
- [x] Enhanced bindgen configuration for comprehensive API coverage

#### Error Handling
- [x] Enhanced SymEngineError with better messages
- [x] Helper methods: is_ok(), is_err(), code(), runtime_error()
- [x] Full std::error::Error trait implementation

#### Testing & Documentation
- [x] 13 comprehensive unit tests
- [x] Container operation tests (CVecBasic, CMapBasicBasic, CDenseMatrix)
- [x] Error handling tests
- [x] Complete README rewrite with examples
- [x] Safety guidelines documentation

#### Build System
- [x] macOS support (Intel & Apple Silicon)
- [x] Linux support (Ubuntu, Fedora, Arch)
- [x] Windows support (experimental via vcpkg)
- [x] Improved library path discovery
- [x] Platform-specific clang args

## High Priority

### Examples & Integration Tests

- [x] **Example Programs**
  - [x] Basic symbolic arithmetic example
  - [x] Matrix operations example
  - [x] Substitution and evaluation example
  - [x] Complex number operations example
  - [x] Number theory operations example

- [x] **Integration Tests** (Partial - 3 of 5)
  - [x] Multi-step symbolic computation tests
  - [x] Matrix algebra workflow tests
  - [x] Substitution chains tests
  - [ ] Memory leak tests (with valgrind/instruments)
  - [ ] Thread safety tests

### Additional Bindings

- [x] **Sparse Matrix Support** ✅
  - [x] sparse_matrix_new/free operations
  - [x] sparse_matrix_set/get_basic
  - [x] Sparse matrix equality testing
  - [x] Example program created

- [x] **Set Operations** (Partial) ✅
  - [x] CSetBasic type available
  - [x] basic_free_symbols usage demonstrated
  - [x] Example program created
  - [ ] Note: Direct set manipulation functions are limited in C API

- [x] **Parser Functions** ✅
  - [x] basic_parse with error handling
  - [x] basic_parse2 available
  - [x] Example program created
  - [ ] Expression serialization (if available in C API)

- [ ] **Series Expansion**
  - [ ] basic_series for Taylor/Laurent series (if available)
  - [ ] Series manipulation functions

### Documentation Improvements

- [ ] **API Documentation**
  - [ ] Document all auto-generated functions
  - [ ] Add safety notes for each function group
  - [ ] Memory management guidelines per API section
  - [ ] Thread safety notes

- [ ] **Developer Guide**
  - [ ] How to add new bindings
  - [ ] How to test new bindings
  - [ ] Bindgen configuration guide
  - [ ] Platform-specific setup guide

## Medium Priority

### Helper Utilities

- [ ] **Safe Wrappers (Optional)**
  - [ ] RAII wrapper for basic_struct (BasicHandle)
  - [ ] RAII wrapper for CVecBasic (VecBasicHandle)
  - [ ] RAII wrapper for CMapBasicBasic (MapBasicHandle)
  - [ ] RAII wrapper for CDenseMatrix (DenseMatrixHandle)

- [ ] **Conversion Helpers**
  - [ ] String to basic_struct helper
  - [ ] basic_struct to String helper
  - [ ] Numeric types to basic_struct helpers
  - [ ] Array conversion helpers

- [ ] **Macros**
  - [ ] macro for safe basic_struct creation
  - [ ] macro for error checking
  - [ ] macro for type code matching

### Performance & Optimization

- [ ] **Benchmarks**
  - [ ] Basic arithmetic operations benchmarks
  - [ ] Matrix operations benchmarks
  - [ ] Substitution performance benchmarks
  - [ ] Comparison with direct SymEngine C++ usage

- [ ] **Profiling**
  - [ ] Memory usage profiling
  - [ ] CPU usage profiling
  - [ ] Identify bottlenecks in FFI layer

### Testing Enhancements

- [ ] **Property-Based Tests**
  - [ ] Arithmetic properties (associativity, commutativity)
  - [ ] Matrix operation properties
  - [ ] Round-trip serialization tests

- [ ] **Fuzzing**
  - [ ] Fuzz test parser
  - [ ] Fuzz test arithmetic operations
  - [ ] Fuzz test matrix operations

## Low Priority

### Advanced Features

- [ ] **LLVM Backend Support**
  - [ ] llvm_double_* function bindings
  - [ ] Code generation support

- [ ] **Lambda Functions**
  - [ ] lambda_double_* function bindings
  - [ ] Function compilation and execution

- [ ] **Evaluation Backends**
  - [ ] eval_mpfr bindings
  - [ ] eval_mpc bindings
  - [ ] eval_arb bindings (if available)

### Build System Improvements

- [ ] **Feature Flags**
  - [ ] Feature for MPFR support
  - [ ] Feature for MPC support
  - [ ] Feature for LLVM backend
  - [ ] Feature for thread-local storage

- [ ] **Cross-Compilation**
  - [ ] Test cross-compilation to ARM
  - [ ] Test cross-compilation to WebAssembly (if possible)
  - [ ] CI/CD for multiple platforms

### Infrastructure

- [ ] **CI/CD Pipeline**
  - [ ] Automated testing on macOS
  - [ ] Automated testing on Linux (multiple distros)
  - [ ] Automated testing on Windows
  - [ ] Coverage reporting
  - [ ] Performance regression tracking

- [ ] **Documentation Automation**
  - [ ] Auto-generate API docs from SymEngine headers
  - [ ] Keep binding coverage table up-to-date
  - [ ] Version compatibility matrix

## Future Considerations

### Breaking Changes (Next Major Version)

- [ ] Consider removing deprecated functions
- [ ] Update to newer SymEngine API if available
- [ ] Improve error type ergonomics
- [ ] Consider zero-copy string APIs

### Research & Investigation

- [ ] Investigate SymEngine 1.0 API changes
- [ ] Research integration with symbolic differentiation frameworks
- [ ] Explore automatic binding generation improvements
- [ ] Investigate direct Rust-C++ bindings (cxx crate)

## Notes

### Design Principles

1. **Minimal FFI Layer**: Keep this crate as thin as possible - just FFI bindings
2. **Safety Through Documentation**: Document unsafe operations clearly
3. **Complete Coverage**: Expose all useful SymEngine C API functions
4. **Cross-Platform**: Support macOS, Linux, Windows equally
5. **Zero Runtime Cost**: FFI should have no overhead

### Current Limitations

- Requires SymEngine 0.11.0+ installed on system
- Some advanced SymEngine features may not be exposed
- Thread safety depends on SymEngine's configuration
- No async/await support (SymEngine is synchronous)

### Integration Points

- **quantrs2-symengine**: Higher-level safe Rust API
- **quantrs2-core**: Quantum computing primitives
- **quantrs2-circuit**: Symbolic circuit optimization
- **quantrs2-ml**: Symbolic gradients for quantum ML

### Maintenance

- Review SymEngine releases for new API additions
- Keep bindgen configuration up-to-date
- Test with new Rust versions
- Monitor for security advisories in dependencies

## Statistics (Current)

- **Functions Bound**: 50+ (via bindgen)
- **Tests**: 14 (13 unit + 1 doc)
- **Lines of Code**: ~380 (src/lib.rs)
- **Build Time**: ~1-2 minutes
- **Platforms Supported**: macOS, Linux, Windows (experimental)
- **SymEngine Version**: 0.11.0+
- **Rust Version**: 1.70+

## Version History

### v0.1.0-beta.3 (2025-11-23)
- Added MapBasicBasic bindings
- Added comprehensive matrix operations
- Added complex number operations
- Enhanced error handling
- Improved documentation
- Added 13 unit tests

### v0.1.0-beta.2 (Earlier)
- Initial FFI bindings
- Basic error handling
- macOS support fixes

## Contributing

When adding new bindings:

1. Update `build.rs` allowlist if needed
2. Add tests for new functionality
3. Document safety requirements
4. Update README.md with new features
5. Run `cargo test` and `cargo clippy`
6. Update this TODO.md

## License

MIT
