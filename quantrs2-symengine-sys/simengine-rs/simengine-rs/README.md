# symengine-rs

Rust bindings to [SymEngine](https://github.com/symengine/symengine).

This is a fork of the original symengine-rs with fixes for macOS and updated symengine-sys bindings.

## Requirements

You need to have SymEngine installed on your system.

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

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
symengine = { git = "https://github.com/cool-japan/symengine-rs.git", branch = "fixed-macos" }
```

## License

MIT