#!/bin/bash

# SymEngine Installation Script for quantrs2-symengine-sys
# This script installs SymEngine and its dependencies in /tmp
# and configures the environment for building quantrs2-symengine-sys

set -e  # Exit on error

echo "==========================================="
echo " SymEngine Installation Script"
echo "==========================================="

# Installation directory
INSTALL_DIR="/tmp/symengine_install"
SYMENGINE_SRC="/tmp/symengine"

echo ""
echo "Step 1: Installing system dependencies..."
echo "------------------------------------------"

# Install required system packages if not already installed
if ! dpkg -l | grep -q "libgmp-dev"; then
    echo "Installing GMP and MPFR development packages..."
    apt-get update
    apt-get install -y libgmp-dev libmpfr-dev
else
    echo "GMP and MPFR already installed."
fi

# Install build tools if not available
if ! command -v cmake &> /dev/null; then
    echo "Installing CMake..."
    apt-get install -y cmake
fi

if ! command -v ninja &> /dev/null; then
    echo "Installing Ninja..."
    apt-get install -y ninja-build
fi

echo ""
echo "Step 2: Cloning SymEngine repository..."
echo "------------------------------------------"

# Remove old source if exists
if [ -d "$SYMENGINE_SRC" ]; then
    echo "Removing existing SymEngine source..."
    rm -rf "$SYMENGINE_SRC"
fi

# Clone SymEngine
cd /tmp
git clone --depth=1 https://github.com/symengine/symengine.git

echo ""
echo "Step 3: Building SymEngine..."
echo "------------------------------------------"

cd "$SYMENGINE_SRC"

# Configure with CMake
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DWITH_GMP=ON \
  -DWITH_MPFR=ON \
  -DWITH_OPENMP=ON \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"

# Build SymEngine
echo "Building SymEngine (this may take a few minutes)..."
ninja -C build -j$(nproc)

echo ""
echo "Step 4: Installing SymEngine..."
echo "------------------------------------------"

# Remove old installation if exists
if [ -d "$INSTALL_DIR" ]; then
    echo "Removing existing installation..."
    rm -rf "$INSTALL_DIR"
fi

# Install
ninja -C build install

echo ""
echo "Step 5: Verifying installation..."
echo "------------------------------------------"

# Verify installation
if [ -f "$INSTALL_DIR/include/symengine/cwrapper.h" ]; then
    echo "✓ SymEngine headers installed successfully"
else
    echo "✗ SymEngine headers not found!"
    exit 1
fi

if [ -f "$INSTALL_DIR/lib/libsymengine.so" ]; then
    echo "✓ SymEngine library installed successfully"
else
    echo "✗ SymEngine library not found!"
    exit 1
fi

echo ""
echo "Step 6: Setting up environment variables..."
echo "------------------------------------------"

# Create environment script
ENV_SCRIPT="/tmp/symengine_env.sh"
cat > "$ENV_SCRIPT" << 'EOF'
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
EOF

chmod +x "$ENV_SCRIPT"

echo "Environment script created at: $ENV_SCRIPT"
echo "To use: source $ENV_SCRIPT"

echo ""
echo "Step 7: Testing the build..."
echo "------------------------------------------"

# Test the build
cd /notebooks/quantrs

# Export environment variables for test
export SYMENGINE_DIR="$INSTALL_DIR"
export GMP_DIR=/usr
export MPFR_DIR=/usr
export LD_LIBRARY_PATH="$INSTALL_DIR/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export BINDGEN_EXTRA_CLANG_ARGS="-I$INSTALL_DIR/include -I/usr/include"

echo "Running cargo build for quantrs2-symengine-sys..."
if cargo build --package quantrs2-symengine-sys; then
    echo "✓ Build successful!"
else
    echo "✗ Build failed!"
    exit 1
fi

echo ""
echo "==========================================="
echo " Installation Complete!"
echo "==========================================="
echo ""
echo "SymEngine has been successfully installed to: $INSTALL_DIR"
echo ""
echo "To use in your current shell session:"
echo "  source /tmp/symengine_env.sh"
echo ""
echo "To rebuild quantrs2-symengine-sys:"
echo "  source /tmp/symengine_env.sh"
echo "  cargo build --package quantrs2-symengine-sys"
echo ""
echo "This script can be run again anytime with:"
echo "  bash /tmp/sym.sh"
echo ""