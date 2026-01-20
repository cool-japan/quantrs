# Platform Documentation

This directory contains documentation for platform-specific features and optimizations in QuantRS2.

## Contents

- [capabilities.md](capabilities.md) - Platform capabilities detection and adaptive optimization

## Overview

QuantRS2 includes sophisticated platform detection and adaptation mechanisms to ensure optimal performance across different hardware configurations. The platform module automatically detects:

- CPU features (SSE, AVX, AVX2, AVX-512)
- GPU availability and capabilities
- Memory hierarchy and cache sizes
- SIMD instruction support

This information is used to select the best implementation for quantum operations at runtime.