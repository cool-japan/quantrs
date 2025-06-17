# symengine-sys

Rust FFI bindings to the [symengine](https://github.com/symengine/symengine) C API.

This is a fork of the original symengine-sys with fixes for macOS and modern bindgen versions.

## Requirements

You need to have symengine installed on your system.

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
symengine-sys = { git = "https://github.com/cool-japan/symengine-sys.git" }
```

## License

MIT