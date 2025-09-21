#!/bin/bash
# Build QuantRS2 Python bindings with GPU support (stub version)

set -e  # Exit on error

echo "Building QuantRS2 Python bindings with GPU support (stub version)..."

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

# Create a temporary feature flag
echo "Creating temporary GPU stub feature flag..."
cat > /tmp/quantrs_gpu_feature.rs << 'EOF'
// This creates a stub GPU feature flag that will be included in the build
#[cfg(feature = "gpu")]
pub mod gpu {
    use std::error::Error;
    
    /// GPU-accelerated state vector simulator (stub implementation)
    #[derive(Debug)]
    pub struct GpuStateVectorSimulator {}
    
    impl GpuStateVectorSimulator {
        /// Create a new GPU-accelerated state vector simulator
        pub async fn new() -> Result<Self, Box<dyn Error>> {
            Err("GPU support is stubbed".into())
        }
        
        /// Create a new GPU-accelerated state vector simulator synchronously
        pub fn new_blocking() -> Result<Self, Box<dyn Error>> {
            Err("GPU support is stubbed".into())
        }
        
        /// Check if GPU acceleration is available on this system
        pub fn is_available() -> bool {
            false
        }
    }
}
EOF

# Temporarily move the real GPU implementation and use our stub
echo "Temporarily disabling real GPU implementation..."
ORIGINAL_GPU_FILE="$quantrs/sim/src/gpu.rs"
BACKUP_GPU_FILE="$quantrs/sim/src/gpu.rs.bak"

if [ -f "$ORIGINAL_GPU_FILE" ]; then
    mv "$ORIGINAL_GPU_FILE" "$BACKUP_GPU_FILE"
fi

cp /tmp/quantrs_gpu_feature.rs "$ORIGINAL_GPU_FILE"

# Build with maturin
echo "Building with GPU stub feature..."
RUST_BACKTRACE=1 maturin develop --release --features gpu

# Restore the original GPU implementation
echo "Restoring original GPU implementation..."
if [ -f "$BACKUP_GPU_FILE" ]; then
    mv "$BACKUP_GPU_FILE" "$ORIGINAL_GPU_FILE"
fi

echo "✅ Build completed successfully with GPU stub!"
echo ""
echo "To test if the GPU code path works, run:"
echo "python minimal_gpu.py"
echo ""
echo "NOTE: This is a STUB implementation. The actual GPU acceleration is disabled,"
echo "but the code path is working, so your Python code can use the use_gpu=True flag."