# Contributing to QuantRS2

Thank you for your interest in contributing to QuantRS2. This document describes
the development setup, policies, and workflow for contributing to the project.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Branching Strategy](#branching-strategy)
3. [COOLJAPAN Policies](#cooljapan-policies)
4. [No-unwrap Policy](#no-unwrap-policy)
5. [File-Size Policy](#file-size-policy)
6. [Naming Conventions](#naming-conventions)
7. [Commit Message Conventions](#commit-message-conventions)
8. [Workspace Policy](#workspace-policy)
9. [No-Warnings Policy](#no-warnings-policy)
10. [Pull Request Guidelines](#pull-request-guidelines)
11. [Code of Conduct](#code-of-conduct)
12. [Security](#security)

---

## Getting Started

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Rust toolchain | stable 1.78+ | `rustup update stable` |
| cargo | bundled with Rust | |
| cargo-nextest | latest (recommended) | `cargo install cargo-nextest` |

### Clone and Build

```bash
git clone https://github.com/cool-japan/quantrs.git
cd quantrs
cargo build --all-features
```

### Running Tests

With cargo-nextest (recommended for parallel execution and better output):

```bash
cargo nextest run --all-features
```

Without cargo-nextest (fallback):

```bash
cargo test --all-features
```

### Running Clippy

```bash
cargo clippy --all-features --all-targets -- -D warnings
```

All clippy diagnostics must pass before a PR is merged.

---

## Branching Strategy

Branch names drive version numbers in QuantRS2. The workspace `Cargo.toml`
version is bumped exactly when the active branch name changes.

| Branch | Purpose |
|--------|---------|
| `master` | Stable, released state — no direct version bumps |
| `0.x.y` | Release branch for version 0.x.y |
| `feature/*` | Feature development; targets the current release branch |
| `fix/*` | Bug fixes; targets the current release branch or `master` |

Rules:

- Never bump versions directly on `master`.
- The branch name (e.g., `0.2.0`) defines what goes into `version` in the
  root `Cargo.toml` and all subcrate `Cargo.toml` files.
- PRs for breaking changes target the next release branch.
- PRs for non-breaking bug fixes or documentation may target `master`.

---

## COOLJAPAN Policies

These policies are **mandatory**. They exist to keep the entire QuantRS2
ecosystem 100% Pure Rust (no C, C++, or Fortran at runtime under default
features) and to maintain a consistent dependency set across all COOLJAPAN
projects.

### Forbidden Dependencies and Their Replacements

| Forbidden | Replacement | Reason |
|-----------|-------------|--------|
| `openblas` / `netlib-blas` | `oxiblas` | Pure Rust BLAS |
| `bincode` | `oxicode` | Pure Rust serialization |
| `rustfft` | `OxiFFT` (`oxifft`) | Pure Rust FFT |
| `z3` | `OxiZ` (`oxiz`) | Pure Rust SMT |
| `zip` | `oxiarc-archive` | Pure Rust archive |
| `flate2` | `oxiarc-compress` | Pure Rust compression |
| `zstd` | `oxiarc-compress` | Pure Rust compression |
| `bzip2` | `oxiarc-compress` | Pure Rust compression |
| `lz4` | `oxiarc-compress` | Pure Rust compression |
| `tar` | `oxiarc-archive` | Pure Rust archive |
| `snap` | `oxiarc-compress` | Pure Rust compression |
| `brotli` | `oxiarc-compress` | Pure Rust compression |
| `miniz_oxide` | `oxiarc-compress` | Pure Rust compression |

All compression and decompression must use the `oxiarc-*` crate family.

### Array and Random Number Usage

Do not import raw `ndarray` or `rand` in library code. Use the re-exports from
SciRS2 instead:

```rust
// Correct
use scirs2_core::ndarray::Array2;
use scirs2_core::random::Rng;

// Wrong (do not use directly in library crates)
use ndarray::Array2;
use rand::Rng;
```

This ensures the entire ecosystem converges on a single ndarray/rand version
managed by SciRS2.

### Pure Rust Default Features

Default features must compile with **zero** C, C++, or Fortran dependencies.
Any feature that requires a C/Fortran library must be placed behind a
non-default feature gate. Verify with:

```bash
cargo build --no-default-features
cargo build  # must be pure Rust
```

### Dependency Versions

Always use the **latest version** available on crates.io. Do not pin to
outdated versions without an explicit reason documented in a comment in
`Cargo.toml`.

---

## No-unwrap Policy

Production code must not call `.unwrap()` or `.expect()` without a compelling
reason. Instead, use proper error propagation:

```rust
// Wrong — panics on None/Err in production
let value = some_option.unwrap();
let result = fallible_call().expect("should not fail");

// Correct — propagates errors
let value = some_option.ok_or_else(|| Error::Missing("field"))?;
let result = fallible_call().map_err(|e| Error::Internal(e.to_string()))?;
```

Permitted exceptions:

- Tests may use `.unwrap()` on paths that are genuinely infallible (e.g.,
  `std::env::temp_dir()`, constructing a known-valid string).
- Benchmark code may use `.unwrap()` where panicking on failure is acceptable.
- If a path is truly unreachable, use `unreachable!()` instead of `.unwrap()`.

When reviewing PRs, any `.unwrap()` or `.expect()` in a non-test file is a
mandatory comment point. The author must explain why it is safe or replace it.

---

## File-Size Policy

Single Rust source files must not exceed **2000 lines**. This limit keeps
modules reviewable and promotes focused responsibilities.

### Detecting Oversized Files

```bash
# List all .rs files with their line counts, sorted descending
find . -name '*.rs' | xargs wc -l | sort -rn | head -100

# Or use rslines (if installed)
rslines 50
```

### Splitting Files

Use `splitrs` (installed at `~/work/splitrs/`) to split files that exceed the
limit:

```bash
splitrs --help
splitrs path/to/large_file.rs
```

After splitting, verify that `cargo build --all-features` and
`cargo nextest run --all-features` still pass.

---

## Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Variables | `snake_case` | `qubit_count`, `gate_matrix` |
| Functions | `snake_case` | `apply_gate`, `build_circuit` |
| Types / Structs / Enums | `CamelCase` | `QuantumCircuit`, `GateKind` |
| Traits | `CamelCase` | `Simulator`, `GateApplicator` |
| Constants | `SCREAMING_SNAKE_CASE` | `MAX_QUBITS`, `DEFAULT_SHOTS` |
| Modules | `snake_case` | `state_vector`, `error_correction` |
| Crates | `kebab-case` (manifest) / `snake_case` (Rust identifier) | `quantrs2-core` / `quantrs2_core` |

These conventions follow the standard Rust API Guidelines
(https://rust-lang.github.io/api-guidelines/). Clippy enforces most of them
automatically.

---

## Commit Message Conventions

QuantRS2 uses **plain descriptive sentences** for commit messages — no
conventional-commits prefixes (`feat:`, `fix:`, `chore:`, etc.) unless the
project explicitly adopts them in the future.

Examples drawn from the project history:

```
Update SciRS2 dependencies to version 0.4.1 for improved features and stability
Availability of 0.1.3
PyO3 0.28 compatibility fixes applied
```

Guidelines:

- Use the imperative mood or a descriptive noun phrase in the subject line.
- Subject line should be 72 characters or fewer.
- If the change requires explanation, add a body separated by a blank line.
- Reference issue numbers when relevant: `Closes #123`.
- Co-authoring with Claude Code is acceptable. Add the footer:

  ```
  Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
  ```

---

## Workspace Policy

QuantRS2 is a Cargo workspace. All per-crate `Cargo.toml` files must use the
workspace inheritance mechanism.

### Correct Pattern

```toml
# In a subcrate Cargo.toml
[package]
name = "quantrs2-core"
version.workspace = true
edition.workspace = true
authors.workspace = true
license.workspace = true
repository.workspace = true

[dependencies]
scirs2-core.workspace = true
```

### Rules

- Never pin a specific version number in a subcrate `Cargo.toml`.
- All shared metadata (`version`, `edition`, `authors`, `license`,
  `repository`, `rust-version`) must use `*.workspace = true`.
- The `keywords` and `categories` sections may differ per crate because each
  crate serves a distinct audience.
- All version upgrades happen only in the root `Cargo.toml` `[workspace]`
  section or its `[workspace.dependencies]` table.

---

## No-Warnings Policy

Before opening a PR, verify that the entire workspace compiles with zero
clippy diagnostics:

```bash
cargo clippy --all-features --all-targets -- -D warnings
```

Do not suppress warnings with `#[allow(...)]` unless there is a documented
reason in a comment on the same line or the line above. Examples that are
acceptable:

```rust
// This field is part of the public API surface; removal is a breaking change.
#[allow(dead_code)]
pub reserved: u64,
```

Blanket `#![allow(unused_imports)]` at the crate root is not acceptable.

---

## Pull Request Guidelines

### PR Template

When opening a pull request, include the following sections in the description:

```
## Summary
Short description of what this PR does and why.

## Affected Crates
List of crates that have changed (e.g., `quantrs2-core`, `quantrs2-sim`).

## Test Plan
- [ ] `cargo nextest run --all-features` passes
- [ ] `cargo clippy --all-features --all-targets -- -D warnings` passes
- [ ] New or updated tests cover the change
- [ ] No new `.unwrap()` / `.expect()` in production code
- [ ] All modified files are under 2000 lines

## Breaking Changes
Yes / No. If yes, describe the impact and migration path.
```

### Review Checklist

Reviewers will verify:

1. COOLJAPAN policy compliance (no forbidden dependencies).
2. No-unwrap policy compliance.
3. All files under 2000 lines.
4. Naming conventions followed.
5. Workspace policy followed (no direct version pins in subcrate `Cargo.toml`).
6. Zero clippy warnings.
7. Tests cover the changed behaviour.

---

## Code of Conduct

All contributors are expected to follow the project's [Code of Conduct](CODE_OF_CONDUCT.md).
Violations may be reported to `kitahata@gmail.com`.

---

## Security

If you discover a security vulnerability, **do not open a public GitHub issue**.
See [SECURITY.md](SECURITY.md) for the responsible disclosure process.
