---
layout: page
title: Installation
permalink: /pages/installation/
---

# Installation Guide

## System Requirements

- Python 3.9 or higher
- 8GB RAM minimum (16GB recommended)
- Modern multi-core processor
- Linux, macOS, or Windows

## Dependencies

The system requires several scientific computing libraries:

- NumPy and SciPy for numerical computation
- CVXPY for optimization
- MOSEK for professional optimization solving
- Matplotlib and Plotly for visualization

## Installation Methods

### Method 1: Package Installation

```bash
# Install from PyPI (when available)
pip install spacecraft-drmpc

# Verify installation
spacecraft-drmpc --version
```

### Method 2: Source Installation

```bash
# Clone repository
git clone https://github.com/yourusername/spacecraft-docking-drmpc.git
cd spacecraft-docking-drmpc

# Install dependencies
make install

# Run tests
make test
```

### Method 3: Docker Installation

```bash
# Build container
docker build -t spacecraft-drmpc .

# Run simulation
docker run -p 8080:8080 spacecraft-drmpc
```

## Configuration

The system uses YAML configuration files located in the config directory. Basic configuration includes spacecraft parameters, mission scenarios, and solver settings.

## Verification

After installation, verify the system works correctly:

```bash
# Run basic test
spacecraft-drmpc simulate --scenario single_docking --test-mode

# Check system status
spacecraft-drmpc --help
```

## Troubleshooting

Common installation issues and solutions are documented in the main repository README file.