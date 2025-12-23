# quantrs2-symengine-sys

Low-level Rust FFI bindings to the [SymEngine](https://github.com/symengine/symengine) C API.

This crate provides comprehensive, safe FFI bindings to SymEngine's C wrapper API, with enhancements for macOS compatibility and modern bindgen versions.

## Features

### Core Bindings
- **Basic Operations**: All symbolic arithmetic operations (add, mul, pow, etc.)
- **Complex Numbers**: Real/imaginary part extraction, conjugation
- **Substitution**: Variable substitution with `CMapBasicBasic` support
- **Calculus**: Differentiation, integration, series expansion
- **Special Functions**: Gamma, Beta, Polygamma, Kronecker delta, etc.
- **Number Theory**: GCD, LCM, modular arithmetic, factorials, Fibonacci, etc.

### Matrix Operations
- **Dense Matrices**: Full support for `CDenseMatrix` operations
- **Matrix Algebra**: Addition, multiplication, scalar operations
- **Linear Algebra**: Determinant, inverse, transpose, LU/LDL/FFLU factorization
- **Matrix Calculus**: Differentiation, Jacobian computation
- **Utility Functions**: Identity, zeros, ones, diagonal matrices

### Container Types
- **CVecBasic**: Dynamic vector of symbolic expressions
- **CMapBasicBasic**: Hash map for variable substitution and replacement
- **CSetBasic**: Set operations on symbolic expressions

### Error Handling
- **Type-Safe Errors**: Rust enum-based error handling
- **Detailed Messages**: Comprehensive error descriptions
- **Error Context**: Helper methods for error inspection (`is_ok()`, `is_err()`, `code()`)

## Requirements

You need SymEngine and its dependencies installed on your system.

### macOS

```bash
brew install symengine gmp mpfr
```

When building, set the following environment variables:

```bash
export SYMENGINE_DIR=$(brew --prefix symengine)
export GMP_DIR=$(brew --prefix gmp)
export MPFR_DIR=$(brew --prefix mpfr)
export BINDGEN_EXTRA_CLANG_ARGS="-I$(brew --prefix symengine)/include -I$(brew --prefix gmp)/include -I$(brew --prefix mpfr)/include"
```

### Linux

```bash
# Ubuntu/Debian
sudo apt-get install libsymengine-dev libgmp-dev libmpfr-dev

# Fedora/RHEL
sudo dnf install symengine-devel gmp-devel mpfr-devel

# Arch Linux
sudo pacman -S symengine gmp mpfr
```

### Windows (via vcpkg)

```powershell
vcpkg install symengine gmp mpfr
```

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
quantrs2-symengine-sys = { version = "0.1.0-beta.3" }
```

### Features

- `static`: Enable static linking to SymEngine
- `system-deps`: Use pkg-config to find system dependencies
- `serde`: Enable serde serialization support (when available)

Example with features:

```toml
[dependencies]
quantrs2-symengine-sys = { version = "0.1.0-beta.3", features = ["static", "serde"] }
```

## Safety

All functions in this crate are `unsafe` as they directly interface with C/C++ code. Users must ensure:

1. **Memory Management**: Proper allocation and deallocation using SymEngine's functions
2. **Pointer Validity**: All pointers passed to SymEngine must be valid
3. **Thread Safety**: Follow SymEngine's thread-safety guidelines
4. **Error Handling**: Always check return codes using `check_result()`

## Example

```rust,no_run
use quantrs2_symengine_sys::*;
use std::os::raw::c_int;

unsafe {
    // Create a symbol
    let mut x = std::mem::zeroed::<basic_struct>();
    symbol_set(&mut x as *mut _, "x\0".as_ptr() as *const i8);

    // Create an integer
    let mut two = std::mem::zeroed::<basic_struct>();
    integer_set_si(&mut two as *mut _, 2);

    // Compute x^2 and check for errors
    let mut result = std::mem::zeroed::<basic_struct>();
    let code = basic_pow(&mut result as *mut _, &x as *const _, &two as *const _);
    check_result(code as c_int).expect("Failed to compute power");
}
```

For a safer, more idiomatic Rust interface, use the [`quantrs2-symengine`](../quantrs2-symengine) crate which builds on these bindings.

## API Coverage

This crate provides bindings to the following SymEngine C API groups:

### Basic Operations
- `basic_*`: Core symbolic operations
- `symbol_*`: Symbol creation and manipulation
- `integer_*`, `rational_*`, `real_double_*`, `complex_double_*`: Numeric types

### Advanced Features
- `vecbasic_*`: Vector operations
- `mapbasicbasic_*`: Map/dictionary operations
- `dense_matrix_*`: Dense matrix operations
- `sparse_matrix_*`: Sparse matrix operations (when available)
- `ntheory_*`: Number theory functions

### Calculus & Analysis
- `basic_diff`: Differentiation
- `basic_expand`: Expression expansion
- `dense_matrix_jacobian`: Jacobian matrix computation

## Platform Support

- **macOS**: Full support (both Intel and Apple Silicon)
- **Linux**: Full support (tested on Ubuntu, Fedora, Arch)
- **Windows**: Experimental support via vcpkg
- **BSD**: Should work but untested

## Integration with QuantRS2

This crate is part of the QuantRS2 quantum computing framework and follows the **SciRS2 Policy**:

- Uses SymEngine for symbolic computation in quantum algorithms
- Supports parameterized quantum circuits
- Enables symbolic optimization of quantum gates
- Facilitates quantum machine learning with symbolic gradients

## License

MIT

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass: `cargo test`
2. Code is formatted: `cargo fmt`
3. No clippy warnings: `cargo clippy`
4. Documentation is updated

## Acknowledgments

- Original `symengine-sys` crate authors
- [SymEngine](https://github.com/symengine/symengine) project
- QuantRS2 development team