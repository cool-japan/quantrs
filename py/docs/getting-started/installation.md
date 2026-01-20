# Installation Guide

Get QuantRS2 up and running on your system in just a few minutes.

## Requirements

### System Requirements
- **Python**: 3.8 or higher (3.9+ recommended)
- **Operating Systems**: Linux, macOS, Windows
- **Memory**: 4GB RAM minimum (8GB+ recommended for large simulations)
- **Storage**: 500MB free disk space

### Optional Requirements
- **GPU**: NVIDIA GPU with CUDA 11.0+ for GPU acceleration
- **Rust**: For building from source (1.70+ required)

## Quick Installation

### Install from PyPI (Recommended)

The easiest way to install QuantRS2 is using pip:

```bash
pip install quantrs2
```

For the latest development version:

```bash
pip install quantrs2[dev]
```

### Install with Optional Dependencies

Install with all optional features:

```bash
# Full installation with all features
pip install quantrs2[full]

# Install with GPU support
pip install quantrs2[gpu]

# Install with visualization tools
pip install quantrs2[viz]

# Install with machine learning extras
pip install quantrs2[ml]

# Install development tools
pip install quantrs2[dev,test]
```

## Verify Installation

Test your installation:

```python
import quantrs2

# Check version
print(f"QuantRS2 version: {quantrs2.__version__}")

# Create a simple circuit
circuit = quantrs2.Circuit(2)
circuit.h(0)
circuit.cx(0, 1)

# Run simulation
result = circuit.run()
print("Installation successful!")
print(f"Bell state probabilities: {result.state_probabilities()}")
```

Expected output:
```
QuantRS2 version: 0.1.0a3
Installation successful!
Bell state probabilities: {'00': 0.5, '11': 0.5}
```

## Installation Methods

### 1. PyPI Installation (Stable)

For most users, installing from PyPI is recommended:

```bash
# Basic installation
pip install quantrs2

# Upgrade to latest version
pip install --upgrade quantrs2

# Install in virtual environment (recommended)
python -m venv quantrs2-env
source quantrs2-env/bin/activate  # On Windows: quantrs2-env\Scripts\activate
pip install quantrs2
```

### 2. Development Installation

For contributing to QuantRS2 or accessing the latest features:

```bash
# Clone the repository
git clone https://github.com/quantrs/quantrs2.git
cd quantrs2

# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e .[dev,test]
```

### 3. Conda Installation

QuantRS2 is also available through conda-forge:

```bash
# Install from conda-forge
conda install -c conda-forge quantrs2

# Create conda environment
conda create -n quantrs2-env python=3.9
conda activate quantrs2-env
conda install -c conda-forge quantrs2
```

### 4. Docker Installation

Use our pre-built Docker images:

```bash
# Pull the latest image
docker pull quantrs2/quantrs2:latest

# Run interactive session
docker run -it --rm quantrs2/quantrs2:latest python

# Run with Jupyter notebook
docker run -p 8888:8888 quantrs2/quantrs2:jupyter
```

## GPU Support

### CUDA Installation

For GPU acceleration, ensure you have CUDA installed:

1. **Install CUDA Toolkit**: Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
2. **Verify CUDA**: Run `nvcc --version` to confirm installation
3. **Install QuantRS2 with GPU support**:

```bash
pip install quantrs2[gpu]
```

### Test GPU Support

```python
import quantrs2

# Check GPU availability
if quantrs2.gpu_available():
    print("GPU acceleration available!")
    
    # Create circuit and run on GPU
    circuit = quantrs2.Circuit(10)
    for i in range(10):
        circuit.h(i)
    
    # Run with GPU acceleration
    result = circuit.run(use_gpu=True)
    print("GPU simulation successful!")
else:
    print("GPU not available, using CPU simulation")
```

## IDE Integration

### VS Code Extension

Install the QuantRS2 VS Code extension:

1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X)
3. Search for "QuantRS2"
4. Click Install

Or install from command line:
```bash
code --install-extension quantrs2.quantrs2-vscode
```

### Jupyter Integration

QuantRS2 works seamlessly with Jupyter notebooks:

```bash
# Install Jupyter
pip install jupyter

# Install QuantRS2 Jupyter extension
pip install quantrs2[jupyter]

# Start Jupyter
jupyter notebook
```

Load the QuantRS2 magic commands:
```python
%load_ext quantrs2.jupyter_magic
```

## Troubleshooting

### Common Issues

#### Import Error: No module named '_quantrs2'

This indicates the native extension wasn't built properly:

```bash
# Reinstall with force rebuild
pip uninstall quantrs2
pip install quantrs2 --no-cache-dir
```

#### CUDA Not Found

If you have CUDA installed but QuantRS2 doesn't detect it:

```bash
# Check CUDA environment
echo $CUDA_HOME
echo $LD_LIBRARY_PATH

# Install with explicit CUDA path
CUDA_HOME=/usr/local/cuda pip install quantrs2[gpu]
```

#### Performance Issues

For better performance:

```bash
# Install with optimized BLAS
pip install quantrs2[performance]

# Use conda for optimized packages
conda install -c conda-forge quantrs2 mkl
```

### Getting Help

If you encounter issues:

1. **Check the FAQ**: [Frequently Asked Questions](../community/support.md#faq)
2. **Search Issues**: [GitHub Issues](https://github.com/quantrs/quantrs2/issues)
3. **Community Discord**: [Join our Discord](https://discord.gg/quantrs2)
4. **Stack Overflow**: Tag questions with `quantrs2`

## Environment Setup

### Virtual Environment (Recommended)

Always use a virtual environment for QuantRS2 projects:

```bash
# Using venv
python -m venv quantrs2-env
source quantrs2-env/bin/activate  # On Windows: quantrs2-env\Scripts\activate
pip install quantrs2

# Using conda
conda create -n quantrs2-env python=3.9
conda activate quantrs2-env
pip install quantrs2
```

### Development Environment

For QuantRS2 development:

```bash
# Clone and setup development environment
git clone https://github.com/quantrs/quantrs2.git
cd quantrs2

# Create development environment
python -m venv dev-env
source dev-env/bin/activate

# Install in development mode with all dependencies
pip install -e .[dev,test,docs]

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest tests/
```

## Next Steps

Now that you have QuantRS2 installed:

1. **[Quick Start Guide](quickstart.md)**: Learn the basics in 5 minutes
2. **[First Circuit](first-circuit.md)**: Build your first quantum circuit
3. **[Basic Examples](basic-examples.md)**: Explore practical examples
4. **[Tutorials](../tutorials/beginner/)**: Comprehensive learning path

## Dependencies

QuantRS2 has minimal required dependencies and many optional ones:

### Required Dependencies
- `numpy >= 1.19.0`
- `scipy >= 1.7.0`

### Optional Dependencies
- `matplotlib >= 3.3.0` (for visualization)
- `jupyter >= 1.0.0` (for notebook integration)
- `pandas >= 1.3.0` (for data analysis)
- `networkx >= 2.6` (for quantum networking)
- `qiskit >= 0.34.0` (for Qiskit integration)
- `cirq >= 0.14.0` (for Cirq integration)

### Development Dependencies
- `pytest >= 6.0`
- `black >= 21.0`
- `mypy >= 0.910`
- `pre-commit >= 2.15.0`

All dependencies are automatically managed when you install QuantRS2 with the appropriate extras.

---

**Ready to start building?** Continue to the [Quick Start Guide](quickstart.md) to create your first quantum circuit!