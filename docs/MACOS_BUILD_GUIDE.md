# macOS Build and Testing Guide

This document provides platform-specific information for building and testing QuantRS2 on macOS systems.

## Build Configuration

QuantRS2 includes macOS-specific build configurations to handle C++ library linking and dependencies:

### C++ Standard Library

On macOS, the project automatically configures the build system to use `libc++` instead of `libstdc++` through `build.rs` files. This resolves linking issues commonly encountered with dependencies that require C++ standard library support (such as SymEngine).

### Accelerate Framework

The simulator components automatically link against Apple's Accelerate framework on macOS for optimized BLAS/LAPACK operations.

## Testing Strategy

Due to platform-specific dependencies and optional features, the recommended testing approach varies:

### Recommended Test Commands

#### Basic Functionality (Core Features)
```bash
cargo test --no-default-features --features "parallel"
```

#### Standard Development Testing
```bash
cargo test --features "parallel,scirs,plotters"
```

#### Extended Feature Testing (without problematic dependencies)
```bash
cargo test --features "parallel,scirs,plotters,clustering,advanced_optimization"
```

### Features to Avoid in Testing

#### --all-features Flag
**❌ NOT RECOMMENDED:**
```bash
cargo test --all-features  # Avoid this on macOS
```

**Reason:** The `--all-features` flag enables the `dwave` feature, which depends on SymEngine with C++ linking requirements that may cause build failures on some macOS configurations.

#### Individual Problematic Features

- **`dwave`**: Requires SymEngine with C++ standard library linking
- **`gpu`**: May fail tests on systems without compatible GPU hardware

## Troubleshooting

### Common Issues

1. **SymEngine/C++ Linking Errors**
   - **Symptom:** `ld: library 'stdc++' not found`
   - **Solution:** Use recommended test commands that exclude the `dwave` feature

2. **GPU Test Failures**
   - **Symptom:** GPU-related tests fail with hardware errors
   - **Solution:** This is expected on systems without compatible GPU hardware

3. **Build Warnings**
   - All build warnings have been eliminated from the standard configuration
   - If you encounter warnings, ensure you're using a clean build: `cargo clean && cargo test`

### Platform-Specific Dependencies

The following system dependencies may be required:

- **OpenBLAS**: Automatically handled through Homebrew paths
- **Accelerate Framework**: Automatically linked on macOS
- **symengine**: Only required for `dwave` feature (optional)

## CI/CD Considerations

For continuous integration on macOS:

1. Use the recommended test commands
2. Avoid `--all-features` flag
3. Consider testing feature combinations separately:
   ```bash
   cargo test --features "parallel"
   cargo test --features "parallel,scirs"
   cargo test --features "parallel,plotters"
   ```

## Performance Notes

- The Accelerate framework provides optimized linear algebra operations
- GPU features require compatible hardware and drivers
- SciRS2 integration provides additional performance optimizations when enabled

## Getting Help

If you encounter platform-specific build issues:

1. Ensure you're using the recommended test commands
2. Check that system dependencies are properly installed
3. Verify that Xcode command line tools are installed: `xcode-select --install`
4. For SymEngine-related issues, consider using features that don't require the `dwave` dependency