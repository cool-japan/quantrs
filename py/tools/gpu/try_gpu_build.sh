#!/bin/bash
# Attempt building QuantRS2 Python bindings with GPU support - limited version

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

# For MacOS with Apple Silicon, use the accelerate framework
if [[ "$(uname)" == "Darwin" ]]; then
    echo "Building on macOS, setting environment variables for Accelerate framework..."
    export OPENBLAS_SYSTEM=1 
    export OPENBLAS64_SYSTEM=1
fi

# Use debug build first to check for errors
echo "Building in debug mode without GPU for basic sanity check..."
RUST_BACKTRACE=1 maturin develop

echo "Basic build successful! Now trying with GPU feature..."
RUST_BACKTRACE=1 maturin develop --features gpu