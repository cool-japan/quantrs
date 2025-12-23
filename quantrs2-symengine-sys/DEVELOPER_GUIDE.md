# Developer Guide: quantrs2-symengine-sys

This guide explains how to develop, maintain, and extend the `quantrs2-symengine-sys` crate.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Build System](#build-system)
3. [Adding New Bindings](#adding-new-bindings)
4. [Testing Strategy](#testing-strategy)
5. [Platform-Specific Development](#platform-specific-development)
6. [Troubleshooting](#troubleshooting)
7. [Release Process](#release-process)

## Architecture Overview

### Crate Structure

```
quantrs2-symengine-sys/
├── src/
│   ├── lib.rs           # Main library with error types and utilities
│   └── api_docs.rs      # Comprehensive API documentation
├── examples/            # Usage examples
│   ├── basic_arithmetic.rs
│   ├── matrix_operations.rs
│   ├── substitution.rs
│   ├── complex_numbers.rs
│   ├── number_theory.rs
│   ├── sparse_matrix_operations.rs
│   ├── parser_operations.rs
│   └── set_operations.rs
├── tests/              # Integration tests
│   ├── integration_symbolic_computation.rs
│   ├── integration_matrix_operations.rs
│   └── integration_substitution_chains.rs
├── build.rs            # Build script for bindgen
└── wrapper.h           # C header wrapper
```

### Key Components

1. **build.rs**: Configures `bindgen` to generate FFI bindings
2. **lib.rs**: Exports bindings and provides Rust-friendly utilities
3. **wrapper.h**: Simple include of SymEngine's cwrapper.h
4. **Bindings (generated)**: Auto-generated in OUT_DIR by bindgen

## Build System

### Build Script (build.rs)

The build script performs three main tasks:

1. **Link Configuration**: Sets up library paths and linking flags
2. **Platform Detection**: Configures platform-specific paths
3. **Binding Generation**: Runs bindgen to create FFI bindings

#### Key Functions

```rust
fn setup_manual_linking()  // Configures library linking
fn setup_platform_specific()  // Platform-specific paths
fn generate_bindings()  // Runs bindgen
```

### Bindgen Configuration

The bindgen configuration uses allowlists to control what gets bound:

```rust
.allowlist_function("basic_.*")
.allowlist_function("integer_.*")
.allowlist_function("symbol_.*")
.allowlist_function("vecbasic_.*")
.allowlist_function("mapbasicbasic_.*")
.allowlist_function("dense_matrix_.*")
.allowlist_function("sparse_matrix_.*")
.allowlist_function("ntheory_.*")
// ... and more
```

### Environment Variables

- `SYMENGINE_DIR`: Path to SymEngine installation
- `GMP_DIR`: Path to GMP installation
- `MPFR_DIR`: Path to MPFR installation
- `BINDGEN_EXTRA_CLANG_ARGS`: Additional clang arguments

## Adding New Bindings

### Step-by-Step Process

#### 1. Identify the SymEngine Functions

Check SymEngine's `cwrapper.h` for available functions:

```bash
# On macOS with Homebrew
cat /opt/homebrew/opt/symengine/include/symengine/cwrapper.h | grep "CWRAPPER_OUTPUT_TYPE"

# On Linux
cat /usr/include/symengine/cwrapper.h | grep "CWRAPPER_OUTPUT_TYPE"
```

#### 2. Update build.rs Allowlist

Add pattern to match your new functions in `generate_bindings()`:

```rust
.allowlist_function("your_function_prefix_.*")
```

For example, to bind all `basic_series_*` functions:

```rust
.allowlist_function("basic_series_.*")
```

#### 3. Add Types if Needed

If your functions use new types:

```rust
.allowlist_type("YourNewType")
```

#### 4. Test the Bindings

```bash
cargo clean
cargo build
```

Check that bindings are generated:

```bash
grep "your_function_name" target/debug/build/quantrs2-symengine-sys-*/out/bindings.rs
```

#### 5. Create Example

Create an example in `examples/your_feature.rs`:

```rust
//! Your feature example
//!
//! Demonstrates usage of your new bindings

use quantrs2_symengine_sys::*;
use std::ffi::CStr;

fn main() {
    unsafe {
        // Your example code
    }
}
```

#### 6. Write Tests

Add integration tests in `tests/integration_your_feature.rs`:

```rust
use quantrs2_symengine_sys::*;

#[test]
fn test_your_feature() {
    unsafe {
        // Test code
    }
}
```

#### 7. Document

Update `src/api_docs.rs` with documentation for your new functions.

### Example: Adding Series Expansion Functions

```rust
// 1. In build.rs, add to generate_bindings():
.allowlist_function("basic_series.*")

// 2. Create examples/series_operations.rs:
//! Series expansion example
use quantrs2_symengine_sys::*;

fn main() {
    unsafe {
        // Example using basic_series if available
    }
}

// 3. Create tests/integration_series.rs:
#[test]
fn test_series_expansion() {
    // Test series operations
}

// 4. Update src/api_docs.rs:
//! ### Series Operations
//! - `basic_series` - Taylor/Laurent series expansion
```

## Testing Strategy

### Test Levels

1. **Unit Tests** (in `src/lib.rs`)
   - Error type conversions
   - Utility function correctness
   - No SymEngine functions (may not be available in all CI environments)

2. **Integration Tests** (in `tests/`)
   - Multi-step symbolic workflows
   - Complex operations
   - Memory safety verification

3. **Examples** (in `examples/`)
   - User-facing documentation
   - Complete, runnable programs
   - Cover all major feature areas

### Writing Tests

#### Basic Integration Test Template

```rust
use quantrs2_symengine_sys::*;
use std::os::raw::c_int;

#[test]
fn test_feature_name() {
    unsafe {
        // Setup
        let mut x = std::mem::zeroed::<basic_struct>();
        symbol_set(&raw mut x, c"x".as_ptr());

        // Operation
        let mut result = std::mem::zeroed::<basic_struct>();
        let code = some_operation(&raw mut result, &raw const x);

        // Verification
        assert_eq!(code as c_int, 0, "Operation should succeed");

        // Cleanup (if needed)
    }
}
```

#### Memory Safety Tests

For containers that require explicit freeing:

```rust
#[test]
fn test_container_lifecycle() {
    unsafe {
        let container = container_new();
        assert!(!container.is_null());

        // Use container

        container_free(container);
        // Don't use container after this!
    }
}
```

### Running Tests

```bash
# All tests
cargo test

# Specific test
cargo test test_name

# With output
cargo test -- --nocapture

# Integration tests only
cargo test --test '*'

# Examples (compile check)
cargo check --examples

# Run specific example
cargo run --example basic_arithmetic
```

## Platform-Specific Development

### macOS

#### Development Setup

```bash
# Install dependencies
brew install symengine gmp mpfr

# Set environment variables
export SYMENGINE_DIR=$(brew --prefix symengine)
export GMP_DIR=$(brew --prefix gmp)
export MPFR_DIR=$(brew --prefix mpfr)
export BINDGEN_EXTRA_CLANG_ARGS="-I$(brew --prefix symengine)/include"
```

#### Common Issues

1. **Apple Silicon vs Intel**
   - Homebrew paths differ: `/opt/homebrew` (ARM) vs `/usr/local` (Intel)
   - Build script handles both automatically

2. **Multiple Clang Versions**
   - Use Xcode's clang: `xcode-select --install`
   - Or specify in BINDGEN_EXTRA_CLANG_ARGS

### Linux

#### Development Setup

```bash
# Ubuntu/Debian
sudo apt-get install libsymengine-dev libgmp-dev libmpfr-dev clang libclang-dev

# Fedora/RHEL
sudo dnf install symengine-devel gmp-devel mpfr-devel clang-devel

# Arch Linux
sudo pacman -S symengine gmp mpfr clang
```

#### Distribution-Specific Notes

- **Ubuntu**: SymEngine may be old; consider building from source
- **Fedora**: Usually has recent SymEngine versions
- **Arch**: Rolling release, very current

### Windows

#### Development Setup (vcpkg)

```powershell
# Install vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# Install dependencies
.\vcpkg install symengine:x64-windows gmp:x64-windows mpfr:x64-windows

# Set environment
$env:VCPKG_ROOT = "C:\path\to\vcpkg"
```

#### Known Limitations

- Windows support is experimental
- Some SymEngine features may not be available
- MSVC and MinGW have different linking requirements

## Troubleshooting

### Bindgen Errors

#### "Cannot find cwrapper.h"

**Problem**: SymEngine headers not found

**Solution**:
```bash
# Set BINDGEN_EXTRA_CLANG_ARGS
export BINDGEN_EXTRA_CLANG_ARGS="-I/path/to/symengine/include"

# Or set SYMENGINE_DIR
export SYMENGINE_DIR="/path/to/symengine"
```

#### "Undefined symbols when linking"

**Problem**: SymEngine library not found

**Solution**:
```bash
# Check library exists
ls /usr/local/lib/libsymengine.*

# Add to linker path
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH  # Linux
export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH  # macOS
```

### Runtime Errors

#### "Symbol not found" or "Library not loaded"

**Problem**: SymEngine dynamic library not in path

**Solution**:
```bash
# macOS
install_name_tool -add_rpath /opt/homebrew/lib target/debug/your_binary

# Linux
ldd target/debug/your_binary  # Check dependencies
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

#### Segmentation Faults

**Problem**: Memory management error

**Common Causes**:
1. Using freed pointer
2. Dangling reference
3. Null pointer dereference
4. Type mismatch

**Debugging**:
```bash
# With gdb
rust-gdb target/debug/your_binary
(gdb) run
(gdb) backtrace

# With lldb (macOS)
rust-lldb target/debug/your_binary
(lldb) run
(lldb) bt

# With valgrind (Linux)
valgrind --leak-check=full target/debug/your_binary
```

### Build Failures

#### "Failed to run custom build command for `quantrs2-symengine-sys`"

Check:
1. SymEngine installed: `pkg-config --modversion symengine`
2. Clang/LLVM installed: `clang --version`
3. Environment variables set correctly

#### Bindgen Version Mismatch

Update bindgen in Cargo.toml to match system clang version.

## Release Process

### Pre-Release Checklist

- [ ] All tests passing: `cargo test`
- [ ] All examples compile: `cargo check --examples`
- [ ] Clippy clean: `cargo clippy -- -D warnings`
- [ ] Format code: `cargo fmt`
- [ ] Update CHANGELOG.md
- [ ] Update version in Cargo.toml
- [ ] Update documentation
- [ ] Test on multiple platforms

### Version Numbering

Follow Semantic Versioning (SemVer):
- **Major** (X.0.0): Breaking API changes
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, backward compatible

### Release Steps

1. **Update Version**
   ```bash
   # In Cargo.toml
   version = "0.1.0-beta.4"
   ```

2. **Update CHANGELOG**
   ```markdown
   ## [0.1.0-beta.4] - 2025-12-04
   ### Added
   - New example programs for parser operations
   - Comprehensive API documentation
   ### Fixed
   - Sparse matrix equality testing
   ```

3. **Tag Release**
   ```bash
   git tag -a v0.1.0-beta.4 -m "Release v0.1.0-beta.4"
   git push origin v0.1.0-beta.4
   ```

4. **Publish** (when ready)
   ```bash
   cargo publish --dry-run  # Test first
   cargo publish
   ```

## Best Practices

### Code Style

1. **Use `raw` pointers consistently** for FFI calls
2. **Check all error codes** immediately after FFI calls
3. **Free all allocated memory** explicitly
4. **Document safety requirements** for each example
5. **Add comments** explaining complex FFI interactions

### Documentation

1. **Every example should be runnable** (`cargo run --example name`)
2. **Include safety notes** in all examples
3. **Explain memory management** for each container type
4. **Provide context** for why certain patterns are used

### Testing

1. **Test error paths** not just success cases
2. **Verify memory cleanup** with valgrind/instruments
3. **Include edge cases** (empty containers, null pointers, etc.)
4. **Document test assumptions** (SymEngine version, platform, etc.)

## Contributing

### Workflow

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Make changes
4. Add tests
5. Update documentation
6. Run checks: `cargo test && cargo clippy && cargo fmt --check`
7. Commit: `git commit -m "feat: your feature"`
8. Push: `git push origin feature/your-feature`
9. Create Pull Request

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**: feat, fix, docs, test, refactor, chore

**Example**:
```
feat(bindings): add series expansion functions

Added bindgen configuration for basic_series_* functions
and created example demonstrating Taylor series expansion.

Closes #123
```

## Resources

### SymEngine Documentation

- [SymEngine GitHub](https://github.com/symengine/symengine)
- [SymEngine C API](https://github.com/symengine/symengine/wiki/C-API)
- [SymEngine Wiki](https://github.com/symengine/symengine/wiki)

### Rust FFI

- [Rust Nomicon - FFI](https://doc.rust-lang.org/nomicon/ffi.html)
- [Bindgen User Guide](https://rust-lang.github.io/rust-bindgen/)
- [Rust FFI Omnibus](http://jakegoulding.com/rust-ffi-omnibus/)

### Tools

- [cargo-outdated](https://github.com/kbknapp/cargo-outdated) - Check for outdated dependencies
- [cargo-audit](https://github.com/RustSec/cargo-audit) - Security vulnerability checks
- [cargo-expand](https://github.com/dtolnay/cargo-expand) - Expand macros

## Maintenance

### Regular Tasks

- **Monthly**: Check for SymEngine updates
- **Quarterly**: Review and update examples
- **Annually**: Major version planning

### Monitoring

- Watch SymEngine releases for API changes
- Monitor Rust toolchain updates
- Track bindgen compatibility

## Support

For questions or issues:
- GitHub Issues: https://github.com/cool-japan/quantrs/issues
- Check existing issues first
- Provide minimal reproducible examples
- Include platform and version information
