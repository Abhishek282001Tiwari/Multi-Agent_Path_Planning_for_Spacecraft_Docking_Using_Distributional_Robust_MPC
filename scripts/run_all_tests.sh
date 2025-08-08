#!/bin/bash
# run_all_tests.sh - Complete test execution and results generation pipeline

set -e  # Exit on any error

echo "=================================================================="
echo "SPACECRAFT DOCKING SYSTEM - COMPREHENSIVE TEST PIPELINE"
echo "=================================================================="
echo "Starting complete testing and documentation generation pipeline..."
echo ""

# Configuration
PYTHON_CMD="python3"
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPTS_DIR")"
RESULTS_DIR="$PROJECT_ROOT/docs/_data/results"
IMAGES_DIR="$PROJECT_ROOT/docs/assets/images"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Python module is available
check_python_module() {
    local module=$1
    $PYTHON_CMD -c "import $module" 2>/dev/null
    return $?
}

# Dependency checks
print_status "Checking system dependencies..."

if ! command -v $PYTHON_CMD &> /dev/null; then
    print_error "Python 3 not found. Please install Python 3.9 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
print_status "Python version: $PYTHON_VERSION"

if ! check_python_module "numpy"; then
    print_warning "NumPy not found. Installing dependencies..."
    pip install -r "$PROJECT_ROOT/requirements.txt"
fi

# Create output directories
print_status "Creating output directories..."
mkdir -p "$RESULTS_DIR"
mkdir -p "$IMAGES_DIR"

# Step 1: Run comprehensive test suite
print_status "Step 1/4: Running comprehensive test suite..."
echo "This may take several minutes..."

cd "$PROJECT_ROOT"
if $PYTHON_CMD scripts/generate_results.py; then
    print_success "Comprehensive testing completed successfully"
else
    print_error "Testing failed. Check error messages above."
    exit 1
fi

# Step 2: Generate visualization plots
print_status "Step 2/4: Generating visualization plots..."

if $PYTHON_CMD scripts/generate_plots.py; then
    print_success "Visualization plots generated successfully"
else
    print_error "Plot generation failed. Check error messages above."
    exit 1
fi

# Step 3: Integrate results with Jekyll
print_status "Step 3/4: Integrating results with Jekyll website..."

if $PYTHON_CMD scripts/jekyll_integration.py; then
    print_success "Jekyll integration completed successfully"
else
    print_error "Jekyll integration failed. Check error messages above."
    exit 1
fi

# Step 4: Generate summary report
print_status "Step 4/4: Generating summary report..."

# Create execution summary
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
SUMMARY_FILE="$RESULTS_DIR/execution_summary.txt"

cat > "$SUMMARY_FILE" << EOF
SPACECRAFT DOCKING SYSTEM - TEST EXECUTION SUMMARY
==================================================

Execution Date: $TIMESTAMP
Python Version: $PYTHON_VERSION
Execution Host: $(hostname)
Project Root: $PROJECT_ROOT

Test Pipeline Status: COMPLETED SUCCESSFULLY

Generated Files:
$(find "$RESULTS_DIR" -name "*.json" -o -name "*.csv" -o -name "*.yml" | wc -l) data files created
$(find "$IMAGES_DIR" -name "*.png" | wc -l) visualization plots generated

Output Directories:
- Results Data: $RESULTS_DIR
- Visualization Plots: $IMAGES_DIR
- Jekyll Data: $PROJECT_ROOT/docs/_data

Next Steps:
1. Review results in $RESULTS_DIR
2. Check visualization plots in $IMAGES_DIR
3. Update Jekyll website with: cd docs && bundle exec jekyll serve
4. Deploy to GitHub Pages when ready

Test Categories Executed:
- DR-MPC Controller Performance Testing
- Multi-Agent Coordination Validation
- Formation Flying Capability Assessment
- Collision Avoidance System Testing
- Fault Tolerance and FDIR Validation
- Security Systems Assessment
- Real-time Performance Analysis
- Scalability Benchmarking
- Accuracy and Precision Testing
- Robustness Under Uncertainty Analysis

EOF

print_success "Summary report saved to: $SUMMARY_FILE"

# Final status report
echo ""
echo "=================================================================="
echo "TEST PIPELINE EXECUTION COMPLETED SUCCESSFULLY"
echo "=================================================================="
echo ""
echo "📊 Results Summary:"
echo "   - $(find "$RESULTS_DIR" -name "*.json" | wc -l | tr -d ' ') JSON result files"
echo "   - $(find "$RESULTS_DIR" -name "*.csv" | wc -l | tr -d ' ') CSV data files" 
echo "   - $(find "$RESULTS_DIR" -name "*.yml" | wc -l | tr -d ' ') YAML configuration files"
echo "   - $(find "$IMAGES_DIR" -name "*.png" | wc -l | tr -d ' ') visualization plots"
echo ""
echo "📁 Output Locations:"
echo "   - Test Results: $RESULTS_DIR"
echo "   - Visualizations: $IMAGES_DIR"
echo "   - Jekyll Data: $PROJECT_ROOT/docs/_data"
echo ""
echo "🚀 Next Steps:"
echo "   1. Review detailed results in the output directories"
echo "   2. Start Jekyll development server: cd docs && ./serve.sh"
echo "   3. View website at http://localhost:4000"
echo "   4. Commit changes and deploy to GitHub Pages"
echo ""
echo "✅ All systems tested and validated successfully!"
echo "=================================================================="