---
layout: page
title: Usage
permalink: /pages/usage/
---

# Usage Guide

## Command Line Interface

The system provides a professional command-line interface for all operations.

### Basic Commands

```bash
# Display help
spacecraft-drmpc --help

# Show version
spacecraft-drmpc --version

# Run simulation
spacecraft-drmpc simulate --scenario SCENARIO_NAME
```

## Simulation Scenarios

```bash
# Single spacecraft docking
spacecraft-drmpc simulate --scenario single_docking --visualize

# Formation flying with multiple spacecraft
spacecraft-drmpc simulate --scenario formation_flying --duration 3600

# Emergency abort procedures
spacecraft-drmpc simulate --scenario emergency_abort --realtime
```

## Monitoring and Analysis

```bash
# Start web dashboard
spacecraft-drmpc dashboard --port 8080

# Analyze simulation results
spacecraft-drmpc analyze --results simulation_results.h5

# Performance benchmarking
spacecraft-drmpc benchmark --cycles 100
```

## Configuration

### Mission Configuration

Mission parameters are specified in YAML files:

```yaml
mission:
  scenario: "single_docking"
  duration: 300
  timestep: 0.1

spacecraft:
  count: 2
  mass: 1000
  initial_position: [0, 0, 0]

control:
  horizon: 30
  frequency: 10
```

## Advanced Options

The system supports extensive customization through configuration files and command-line parameters for research and operational applications.

## Programming Interface

For custom applications, the system provides a Python API:

```python
from spacecraft_drmpc import SpacecraftSimulator

# Create simulator
sim = SpacecraftSimulator("single_docking")

# Run simulation
results = sim.run(duration=300, visualize=True)

# Analyze results
print(f"Final position error: {results.position_error}")
```

## Output Formats

Simulation results are available in multiple formats:

- HDF5 for numerical data
- JSON for configuration and metadata
- CSV for tabular analysis
- HTML for web-based reports