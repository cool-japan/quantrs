# Build Notes: quantrs2-symengine-sys on Linux (CUDA)

This document describes how to build and link `quantrs2-symengine-sys` on a typical Linux system with CUDA support.

## 1. System dependencies

Install core packages:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake ninja-build pkg-config git \
  clang-14 libclang-14-dev libclang1-14 llvm-14-dev \
  libgmp-dev libmpfr-dev
```

Notes:

* We use LLVM/Clang 14 for compatibility (`bindgen` requires the C API `libclang.so`).
* Other versions (e.g. LLVM 16/18 on newer distros) also work, but adjust `LIBCLANG_PATH` accordingly.

CUDA toolkit is optional but recommended if GPU acceleration is needed:

```bash
sudo apt-get install -y nvidia-cuda-toolkit
```

## 2. Build SymEngine from source

If `libsymengine-dev` is not available in your distribution:

```bash
git clone --depth=1 https://github.com/symengine/symengine.git
cd symengine
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DWITH_GMP=ON \
  -DWITH_MPFR=ON \
  -DWITH_OPENMP=ON
ninja -C build
sudo ninja -C build install
sudo ldconfig
cd ..
```

This installs `libsymengine.so` under `/usr/local/lib` and headers under `/usr/local/include/symengine`.

Verify installation:

```bash
test -f /usr/local/include/symengine/cwrapper.h && echo "SymEngine headers OK"
ldconfig -p | grep symengine
```

## 3. Environment variables for Rust build

Ensure `bindgen` can locate libclang and headers:

```bash
# Point to LLVM 14's libclang
export LIBCLANG_PATH=/usr/lib/llvm-14/lib
export LD_LIBRARY_PATH=$LIBCLANG_PATH:/usr/local/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Ensure SymEngine headers are visible
export BINDGEN_EXTRA_CLANG_ARGS="-I/usr/local/include -I/usr/include"

# Optional explicit hints
export SYMENGINE_DIR=/usr/local
export GMP_DIR=/usr
export MPFR_DIR=/usr
```

Check:

```bash
ls $LIBCLANG_PATH/libclang*.so*
ls /usr/local/include/symengine/cwrapper.h
```

## 4. Build the Rust crate

```bash
cd quantrs/quantrs2-symengine-sys
cargo clean
RUST_BACKTRACE=1 cargo build -vv
```

## 5. Common pitfalls

* **Error: `Unable to find libclang`**
  → `libclang.so` is missing. Install `libclang-XX-dev` and set `LIBCLANG_PATH=/usr/lib/llvm-XX/lib`.

* **Error: `fatal error: 'symengine/cwrapper.h' file not found`**
  → SymEngine headers not installed. Build and install from source.

* **Error: `fatal error: 'stddef.h' file not found`**
  → Python’s bundled `libclang` lacks builtin headers. Always use system LLVM (`apt install clang-XX libclang-XX-dev`).

---

With these steps, `quantrs2-symengine-sys` should compile cleanly on Linux with CUDA-enabled environments.

