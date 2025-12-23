# Changelog

All notable changes to the `quantrs2` facade crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0-beta.3] - 2025-12-04

### Added

#### Core Features
- **Hierarchical Prelude System**: Seven-level prelude hierarchy for granular control over imports
  - `essentials`: Core types and error handling
  - `circuits`: Circuit construction (includes essentials)
  - `simulation`: Quantum simulators (includes circuits)
  - `algorithms`: ML and optimization algorithms (includes simulation)
  - `hardware`: Real quantum device integration (includes circuits)
  - `quantum_annealing`: QUBO and Ising models (includes essentials)
  - `tytan`: High-level Tytan DSL (includes quantum_annealing)
  - `full`: All features combined

#### Developer Utilities
- **Configuration Management** (`quantrs2::config`):
  - Global singleton configuration with builder pattern
  - Environment variable support (`QUANTRS2_NUM_THREADS`, etc.)
  - Thread pool, memory limits, logging levels, backend selection

- **System Diagnostics** (`quantrs2::diagnostics`):
  - Hardware capability detection (CPU, memory, GPU, SIMD)
  - Feature availability checking
  - SciRS2 integration validation
  - Comprehensive diagnostic reports with issue detection

- **Utility Functions** (`quantrs2::utils`):
  - Memory estimation for quantum circuits
  - Quantum mathematical functions (fidelity, entropy, Hilbert space)
  - Formatting utilities for memory and duration
  - Qubit capacity planning

- **Testing Helpers** (`quantrs2::testing`):
  - Floating-point assertion functions
  - Vector comparison utilities
  - Measurement statistics validation
  - Reproducible test data generation

- **Benchmarking Tools** (`quantrs2::bench`):
  - High-precision timing utilities
  - Statistical aggregation (mean, median, std dev)
  - Throughput measurement
  - Percentile calculations

#### Error Handling
- **Unified Error System** (`quantrs2::error`):
  - `QuantRS2Error` enum with 8 error categories
  - `QuantRS2Result<T>` type alias
  - Error categorization (Core, Circuit, Simulation, Hardware, etc.)
  - Recovery and user-friendly error messages
  - Context addition with `with_context()`

#### Version Management
- **Version Information** (`quantrs2::version`):
  - Build-time constants (VERSION, SCIRS2_VERSION, BUILD_TIMESTAMP, GIT_COMMIT_HASH)
  - Platform information (TARGET_TRIPLE, BUILD_PROFILE, RUST_VERSION)
  - Compatibility checking functions
  - Detailed version info structures

#### Deprecation Framework
- **Deprecation Tracking** (`quantrs2::deprecation`):
  - Deprecation status tracking (Stable, PendingDeprecation, Deprecated, Removed)
  - Stability levels (Experimental, Unstable, Stable)
  - Migration path documentation
  - Module stability registry
  - Comprehensive deprecation reports

### Testing
- **Comprehensive Test Suite**: 274+ tests across multiple categories
  - 77 unit tests (lib)
  - 43 feature combination tests
  - 23 performance tests
  - 89 documentation tests (37 active, 52 feature-specific)
  - Cross-subcrate integration tests
  - API consistency verification

### Documentation
- **Enhanced README**:
  - Quick start guide with Bell state example
  - 5 detailed use case scenarios
  - Facade vs individual crates comparison
  - Performance optimization tips
  - Compilation time benchmarks
  - 8 runnable examples

- **Build-Time Validation**:
  - Feature combination checking
  - SciRS2 version validation
  - Deprecated dependency warnings
  - Optimization suggestions

### Changed
- Improved SciRS2 integration patterns across all modules
- Enhanced compilation time optimization through feature gates
- Refined error handling with better context preservation
- Updated workspace dependencies to SciRS2 v0.1.0-rc.2

### Fixed
- Test compatibility issues with Simulator trait imports
- VERSION constant exports in prelude modules
- Integration test failures with correct module paths
- Clippy warnings in test suites

### Performance
- Zero-cost facade abstraction verified through performance tests
- Minimal overhead for error handling (~10-50ns)
- Efficient diagnostics with caching (sub-microsecond access)
- Optimized feature detection

### Compatibility
- **Rust Version**: 1.86.0+
- **SciRS2**: v0.1.0-rc.2
- **OptiRS**: v0.1.0-beta.2
- **NumRS2**: v0.1.0-beta.3

### Developer Experience
- **Feature Selection**: Granular control via Cargo features
- **Import Simplicity**: Hierarchical preludes reduce boilerplate
- **Error Handling**: Unified error types across all subcrates
- **Configuration**: Single global config for all components
- **Diagnostics**: Comprehensive system readiness checking

## [0.1.0-beta.2] - 2025-11-19

### Added
- Initial facade crate structure
- Basic feature-based re-exports
- Core prelude module
- Version information module

### Changed
- Migrated from individual crate usage pattern
- Consolidated error handling

## Migration Guide

### From Alpha to Beta.3

#### Import Changes
```rust
// Before (alpha)
use quantrs2::prelude::*;  // Imported everything

// After (beta.3)
use quantrs2::prelude::essentials::*;  // Minimal imports
use quantrs2::prelude::simulation::*;  // Add as needed
```

#### Configuration Changes
```rust
// Before (alpha)
// No unified configuration

// After (beta.3)
use quantrs2::config::Config;
Config::builder()
    .num_threads(8)
    .memory_limit_gb(16)
    .apply();
```

#### Error Handling
```rust
// Before (alpha)
use quantrs2_core::error::QuantRS2Error;

// After (beta.3)
use quantrs2::prelude::essentials::*;  // Includes QuantRS2Error
// Or
use quantrs2::error::{QuantRS2Error, QuantRS2Result};
```

### From Individual Crates to Facade

#### Dependency Changes
```toml
# Before (individual crates)
[dependencies]
quantrs2-core = "0.1.0-beta.3"
quantrs2-circuit = "0.1.0-beta.3"
quantrs2-sim = "0.1.0-beta.3"

# After (facade)
[dependencies]
quantrs2 = { version = "0.1.0-beta.3", features = ["sim"] }
```

#### Import Changes
```rust
// Before (individual crates)
use quantrs2_circuit::builder::Circuit;
use quantrs2_sim::StateVectorSimulator;

// After (facade)
use quantrs2::prelude::simulation::*;
// Circuit and StateVectorSimulator are now available
```

## Contributing

Please see the main [CONTRIBUTING.md](https://github.com/cool-japan/quantrs/blob/master/CONTRIBUTING.md) for guidelines.

## License

Licensed under either of

 * Apache License, Version 2.0 ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

[Unreleased]: https://github.com/cool-japan/quantrs/compare/v0.1.0-beta.3...HEAD
[0.1.0-beta.3]: https://github.com/cool-japan/quantrs/compare/v0.1.0-beta.2...v0.1.0-beta.3
[0.1.0-beta.2]: https://github.com/cool-japan/quantrs/releases/tag/v0.1.0-beta.2
