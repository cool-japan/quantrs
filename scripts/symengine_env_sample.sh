# SymEngine environment variables for quantrs2-symengine-sys

export SYMENGINE_DIR=/tmp/symengine_install
export GMP_DIR=/usr
export MPFR_DIR=/usr
export LD_LIBRARY_PATH=/tmp/symengine_install/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export BINDGEN_EXTRA_CLANG_ARGS="-I/tmp/symengine_install/include -I/usr/include"

echo "SymEngine environment configured:"
echo "  SYMENGINE_DIR=$SYMENGINE_DIR"
echo "  GMP_DIR=$GMP_DIR"
echo "  MPFR_DIR=$MPFR_DIR"
echo "  LD_LIBRARY_PATH includes SymEngine libs"
echo "  BINDGEN_EXTRA_CLANG_ARGS configured"
