# This file is just a placeholder for pip discovery
# The actual build is handled by maturin through pyproject.toml

from setuptools import setup, find_namespace_packages

setup(
    name="quantrs2",
    version="0.1.0b2",
    description="Python bindings for the QuantRS2 quantum computing framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="QuantRS2 Contributors",
    author_email="noreply@example.com",
    url="https://github.com/cool-japan/quantrs",
    packages=find_namespace_packages(include=["quantrs2*", "_quantrs2*"], where="python"),
    package_dir={"": "python"},
    package_data={
        "quantrs2": ["**/*.py"],
        "_quantrs2": ["**/*.py"]
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