# This file is just a placeholder for pip discovery
# The actual build is handled by maturin through pyproject.toml

from setuptools import setup

setup(
    name="quantrs2",
    version="0.1.0a1",
    description="Python bindings for the QuantRS2 quantum computing framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="QuantRS2 Contributors",
    author_email="noreply@example.com",
    url="https://github.com/cool-japan/quantrs",
    packages=["quantrs2"],
    package_dir={"": "python"},
    zip_safe=False,
    python_requires=">=3.8",
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