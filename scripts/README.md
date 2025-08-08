# Testing and Results Generation Scripts

This directory contains comprehensive testing and results generation scripts for the Spacecraft Docking System.

## Scripts Overview

### Core Testing Scripts

#### `generate_results.py`
Comprehensive testing suite that validates all system components and generates detailed performance metrics.

**Features:**
- 10 major test categories with 1000+ individual test cases
- Async testing framework for efficient execution
- Statistical analysis with confidence intervals
- JSON, CSV, and YAML output formats
- Executive summary generation

**Usage:**
```bash
python scripts/generate_results.py
```

**Test Categories:**
- DR-MPC Controller Performance
- Multi-Agent Coordination
- Formation Flying Capabilities
- Collision Avoidance System
- Fault Tolerance and FDIR
- Security Systems Validation
- Real-time Performance Analysis
- Scalability Benchmarking
- Accuracy and Precision Testing
- Robustness Under Uncertainty

#### `generate_plots.py`
Visualization system that creates clean, minimalistic plots for website integration.

**Features:**
- White background, black text only (Jekyll-compatible)
- High-resolution PNG output (300 DPI)
- 10 different plot types covering all performance aspects
- Matplotlib-based with professional styling

**Usage:**
```bash
python scripts/generate_plots.py
```

**Generated Plots:**
- Scalability Performance Analysis
- Real-time Performance Metrics
- Accuracy Comparison Charts
- Fault Tolerance Heatmaps
- Uncertainty Robustness Curves
- Formation Flying Success Rates
- Collision Avoidance Performance
- System Comparison Radar Charts
- DR-MPC Performance Under Uncertainty
- Security Metrics Dashboard

#### `jekyll_integration.py`
Converts test results into Jekyll-friendly formats for website integration.

**Features:**
- YAML data file generation for Jekyll tables
- CSV benchmark data for performance comparisons
- Navigation structure updates
- Performance metrics formatting
- System comparison data generation

**Usage:**
```bash
python scripts/jekyll_integration.py
```

### Execution Scripts

#### `run_all_tests.sh`
Complete test execution pipeline with dependency checking and error handling.

**Features:**
- Automatic dependency verification
- Colored output for status tracking
- Complete pipeline execution (tests → plots → Jekyll integration)
- Comprehensive error handling and reporting
- Execution summary generation

**Usage:**
```bash
./scripts/run_all_tests.sh
```

**Pipeline Steps:**
1. System dependency checks
2. Comprehensive test suite execution
3. Visualization plot generation
4. Jekyll data integration
5. Summary report creation

#### `quick_test.py`
Rapid validation and demonstration script for development purposes.

**Features:**
- Sub-second execution time
- Essential system validation
- Quick Jekyll data generation
- Development-friendly output

**Usage:**
```bash
python scripts/quick_test.py
```

## Output Structure

### Test Results (`docs/_data/results/`)
```
results/
├── detailed_results.json          # Complete test results
├── executive_summary.json         # Executive summary
├── system_comparison.json         # Performance comparison data
├── performance_metrics.csv        # Tabular performance data
├── scalability_benchmark.csv      # Scalability test results
├── accuracy_benchmark.csv         # Accuracy test results
└── execution_summary.txt          # Pipeline execution log
```

### Visualizations (`docs/assets/images/`)
```
images/
├── scalability_performance.png    # Fleet size vs performance
├── real_time_performance.png      # Control frequency analysis
├── accuracy_comparison.png        # Precision across scenarios
├── fault_tolerance_heatmap.png    # FDIR performance matrix
├── uncertainty_robustness.png     # Robustness curves
├── formation_flying_success.png   # Formation capabilities
├── collision_avoidance_performance.png  # Avoidance metrics
├── system_comparison.png          # Radar chart comparison
├── dr_mpc_performance.png         # Controller performance
└── security_metrics.png           # Security validation
```

### Jekyll Data (`docs/_data/`)
```
_data/
├── performance_metrics.yml        # System specifications table
├── test_results.yml              # Test status and scores
├── system_comparison.yml         # Comparison table data
├── specifications.yml            # Technical specifications
├── scalability_benchmark.csv     # Scalability data for tables
├── accuracy_benchmark.csv        # Accuracy data for tables
├── results_summary.yml           # Homepage results summary
└── navigation.yml                # Results page navigation
```

## Development Workflow

### Quick Development Testing
For rapid validation during development:

```bash
# Quick validation (< 1 second)
python scripts/quick_test.py

# Start Jekyll development server
cd docs && ./serve.sh

# View results at http://localhost:4000
```

### Complete System Validation
For comprehensive testing and production deployment:

```bash
# Full test suite (5-10 minutes)
./scripts/run_all_tests.sh

# Review detailed results
ls docs/_data/results/

# Deploy to production
git add . && git commit -m "Update test results" && git push
```

### Individual Script Execution
Run individual components as needed:

```bash
# Generate test results only
python scripts/generate_results.py

# Generate plots only (requires existing results)
python scripts/generate_plots.py

# Update Jekyll data only (requires existing results)
python scripts/jekyll_integration.py
```

## Dependencies

### Required Python Packages
```bash
pip install numpy scipy matplotlib plotly pandas pyyaml asyncio
```

### Optional Dependencies
```bash
# For enhanced plotting
pip install seaborn

# For memory profiling
pip install psutil

# For advanced statistics
pip install scikit-learn
```

### System Requirements
- Python 3.9 or higher
- 4GB RAM minimum (8GB recommended for full tests)
- 1GB free disk space
- Multi-core processor recommended

## Testing Parameters

### Performance Targets
- **Position Accuracy**: 0.1 meters
- **Attitude Accuracy**: 0.5 degrees
- **Control Frequency**: 100 Hz maximum
- **Fleet Size**: 50+ spacecraft
- **Collision Avoidance**: 95%+ success rate
- **Fault Recovery**: <30 seconds
- **Real-time Compliance**: 95%+ deadline adherence

### Test Scale
- **Total Test Cases**: 1000+ individual tests
- **Monte Carlo Trials**: 100-1000 per scenario
- **Uncertainty Levels**: 10%-50% tested
- **Fleet Sizes**: 1-100 spacecraft tested
- **Mission Scenarios**: 15+ different scenarios
- **Fault Types**: 6 major fault categories

## Troubleshooting

### Common Issues

**Memory Issues:**
```bash
# Reduce test scale in generate_results.py
# Modify loop ranges: range(100) → range(25)
```

**Dependency Errors:**
```bash
# Install all dependencies
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.9+
```

**Permission Errors:**
```bash
# Make scripts executable
chmod +x scripts/*.sh

# Check directory permissions
ls -la docs/_data/
```

**Jekyll Integration Issues:**
```bash
# Ensure Jekyll data directory exists
mkdir -p docs/_data

# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('docs/_data/performance_metrics.yml'))"
```

### Debug Mode
Enable verbose output for debugging:

```bash
# Set debug environment variable
export SPACECRAFT_DEBUG=1
python scripts/generate_results.py

# Or use Python debug flags
python -v scripts/generate_results.py
```

## Contributing

### Adding New Tests
1. Add test method to `ComprehensiveTestSuite` class
2. Update `run_all_tests()` method to include new test
3. Add corresponding visualization in `generate_plots.py`
4. Update Jekyll integration in `jekyll_integration.py`

### Modifying Plots
1. Edit plot methods in `ResultsVisualizer` class
2. Maintain white background, black text styling
3. Save as high-resolution PNG (300 DPI)
4. Update plot generation list in `generate_all_plots()`

### Jekyll Integration
1. Create new YAML data files in Jekyll format
2. Use consistent naming: `snake_case` for files, `Title Case` for display
3. Ensure all data is JSON-serializable
4. Test with Jekyll development server

For questions or issues, refer to the main project documentation or create an issue in the project repository.