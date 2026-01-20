#!/bin/bash
# Build QuantRS2 Python bindings with GPU support

set -e  # Exit on error

echo "Building QuantRS2 Python bindings with GPU support..."

# Change to the script directory
cd "$(dirname "$0")"

# Check if maturin is installed
if ! command -v maturin &> /dev/null; then
    echo "❌ Error: maturin is not installed"
    echo "Please install it with: pip install maturin"
    exit 1
fi

# Build with maturin
echo "Running maturin develop with GPU feature..."

# For MacOS with Apple Silicon, use the accelerate framework
if [[ "$(uname)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    echo "Building on Apple Silicon macOS, setting environment variables for Accelerate framework..."
    export OPENBLAS_SYSTEM=1 
    export OPENBLAS64_SYSTEM=1
fi

# Use debug build first to get better error messages
echo "Building in debug mode first to check for errors..."
maturin develop --features gpu

# If that succeeds, build in release mode
echo "Now building in release mode..."
maturin develop --release --features gpu

echo "✅ Build completed successfully!"
echo ""
echo "To test GPU support, run:"
echo "python gpu_test.py"