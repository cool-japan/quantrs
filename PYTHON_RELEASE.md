# Python Package Release Guide for QuantRS2

This document outlines the steps to build and release the Python package for QuantRS2 to PyPI.

## Prerequisites

- Python 3.8+ installed
- Required Python packages: `build`, `wheel`, `setuptools`, `twine`
- PyPI account with API token

## Release Process

### 1. Update Version Numbers

Update the version number in the following files:
- `/py/python/quantrs2/__init__.py`: Update `__version__` variable
- `/py/pyproject.toml`: Update `version` field 
- `/py/setup.py`: Update `version` field

### 2. Manual Build Process

The normal `maturin` build process doesn't include Python modules correctly. Use this manual build process instead:

```bash
# Navigate to project root
cd /path/to/quantrs

# Create a temporary build directory
mkdir -p dist/quantrs2 dist/_quantrs2

# Copy Python modules
cp -r py/python/quantrs2/* dist/quantrs2/

# Extract native module from maturin wheel
mkdir -p wheel_extract && cd wheel_extract
maturin build --release -m py/Cargo.toml
unzip ../target/wheels/quantrs2-*.whl
cp _quantrs2/_quantrs2.abi3.so ../dist/_quantrs2/
cp _quantrs2/__init__.py ../dist/_quantrs2/
cd ..

# Create setup.py for manual build
cat > dist/setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="quantrs2",
    version="VERSION_NUMBER",  # Update this
    description="Python bindings for the QuantRS2 quantum computing framework",
    long_description=open("../py/README.md").read(),
    long_description_content_type="text/markdown",
    author="QuantRS2 Contributors",
    author_email="noreply@example.com",
    url="https://github.com/cool-japan/quantrs",
    packages=["quantrs2", "_quantrs2"],
    package_data={
        "quantrs2": ["**/*.py"],
        "_quantrs2": ["**/*.py", "*.so", "*.dylib", "*.pyd"],
    },
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "ipython>=7.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=22.3.0",
            "flake8>=5.0.0",
            "isort>=5.10.0",
        ],
        "ml": [
            "scikit-learn>=1.0.0",
            "scipy>=1.7.0",
        ],
        "gpu": [
            "tabulate>=0.8.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Rust",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "License :: OSI Approved :: Apache Software License",
    ],
)
EOF

# Update the version number in the setup.py file
sed -i '' "s/VERSION_NUMBER/$(grep -o "__version__ = \"[^\"]*\"" py/python/quantrs2/__init__.py | cut -d'"' -f2)/g" dist/setup.py

# Build wheel package
cd dist
python -m build --wheel
```

### 3. Testing the Package Locally

Test the package locally before uploading to PyPI:

```bash
# Install the built wheel
pip install dist/quantrs2-*.whl --force-reinstall

# Test the basic functionality
python -c "import quantrs2; print(f'Version: {quantrs2.__version__}'); circuit = quantrs2.PyCircuit(2); circuit.h(0); circuit.cnot(0, 1); result = circuit.run(); print(result.state_probabilities())"
```

### 4. Uploading to PyPI

Use `twine` to upload the package to PyPI:

```bash
# For TestPyPI (optional testing step)
twine upload --repository testpypi dist/quantrs2-*.whl

# For production PyPI
twine upload dist/quantrs2-*.whl
```

You will be prompted for your PyPI credentials. Alternatively, you can use a PyPI API token:

```bash
twine upload dist/quantrs2-*.whl --username __token__ --password pypi-YOUR_API_TOKEN
```

### 5. Verify the Release

Once uploaded, verify the package can be installed from PyPI:

```bash
# Create a new virtual environment
python -m venv test_env
source test_env/bin/activate

# Install the package from PyPI
pip install quantrs2

# Test basic functionality
python -c "import quantrs2; print(quantrs2.__version__)"
```

## Troubleshooting

### Issue: Missing Python Modules

If the wheel file doesn't include Python modules, check the following:
- The `package_data` in `setup.py` includes all necessary file patterns
- The directory structure follows the Python package conventions
- The `__init__.py` files exist in all package directories

### Issue: Native Module Loading Error

If the native module fails to load:
- Ensure the `.so`/`.dylib`/`.pyd` file is included in the wheel
- Check platform compatibility (e.g., macOS arm64 vs x86_64)
- Verify that `_quantrs2/__init__.py` correctly imports the native module

### Issue: Version Mismatch

If the installed version doesn't match expected version:
- Check all version numbers in source files
- Make sure to rebuild both the Rust components and Python package
- Verify that no old installations are interfering

## Notes

- Remember to increment the version number for each release according to semantic versioning
- For macOS builds, ensure correct architecture targeting (arm64 for Apple Silicon)
- Consider using GitHub Actions to automate the release process