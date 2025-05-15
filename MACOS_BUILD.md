# Building Quantrs on macOS

Quantrs uses several complex dependencies, including linear algebra libraries that can sometimes cause build issues on macOS, particularly on Apple Silicon (ARM) Macs. This document provides troubleshooting steps for common build problems.

## OpenBLAS Build Issues

The primary issue you might encounter is with the OpenBLAS dependency used by linear algebra libraries. You'll see errors like:

```
error: failed to run custom build command for `openblas-src v0.10.11`
```

With a long series of compiler warnings and eventually a build failure.

## Solution 1: Use System BLAS (Recommended)

macOS includes Apple's high-performance Accelerate framework, which includes BLAS/LAPACK implementations optimized for Apple Silicon. The easiest solution is to use this system library instead of building OpenBLAS:

```bash
# Build using the system BLAS
OPENBLAS_SYSTEM=1 OPENBLAS64_SYSTEM=1 cargo build
```

Or add these settings to your `.cargo/config.toml` file:

```toml
[env]
OPENBLAS_SYSTEM = "1"
OPENBLAS64_SYSTEM = "1"
```

## Solution 2: Build Components Incrementally

If you're still experiencing issues, try building the components incrementally:

```bash
# First build the core components
cargo build -p quantrs-core -p quantrs-circuit

# Then build sim components without default features
cargo build -p quantrs-sim --no-default-features
```

## Solution 3: Disable Advanced Math Features

For basic quantum simulation, you can build without the advanced math features:

```bash
# Build without advanced math features (disables tensor network simulation)
cargo build -p quantrs-sim --no-default-features
```

Then, if you need specific features, you can enable them selectively:

```bash
# Add optimizations (but still no advanced math)
cargo build -p quantrs-sim --no-default-features --features="optimize"

# Add GPU support (but still no advanced math)
cargo build -p quantrs-sim --no-default-features --features="optimize,gpu"
```

## Solution 4: Install BLAS/LAPACK via Homebrew

If you need the full feature set and Solutions 1-3 don't work, you can install OpenBLAS via Homebrew:

```bash
brew install openblas
```

Then build with:

```bash
RUSTFLAGS="-L/usr/local/opt/openblas/lib" OPENBLAS_SYSTEM=1 cargo build
```

## Testing Your Build

Once you've successfully built the package, you can run the tests:

```bash
# Run tests for core components
cargo test -p quantrs-core

# Run minimal simulator tests
cargo test -p quantrs-sim --no-default-features
```

## Reporting Build Issues

If you continue to experience build issues, please open an issue on the GitHub repository with:

1. Your macOS version
2. Architecture (Intel/ARM)
3. Rust version (`rustc --version`)
4. Complete build output