# GPU Documentation

This directory contains documentation for GPU acceleration in QuantRS2.

## Contents

- [metal_backend.md](metal_backend.md) - Metal GPU backend documentation for macOS/Apple Silicon

## Overview

QuantRS2 supports multiple GPU backends for accelerated quantum simulation:
- CUDA (NVIDIA GPUs)
- OpenCL (Cross-platform)
- Metal (Apple Silicon)

All GPU backends are being migrated to use SciRS2's unified GPU abstractions for better maintainability and performance.