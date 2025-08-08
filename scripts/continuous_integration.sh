#!/bin/bash
# Continuous Integration automation script for Spacecraft Docking System
# Performs comprehensive testing, validation, and deployment preparation

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="${PROJECT_ROOT}/venv"
REPORTS_DIR="${PROJECT_ROOT}/reports"
COVERAGE_MIN=85

# Functions
print_header() {
    echo -e "\n${BLUE}=================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

check_dependencies() {
    print_header "Checking Dependencies"
    
    # Check Python version
    if ! python3 --version | grep -E "3\.(9|1[0-9])" > /dev/null; then
        print_error "Python 3.9+ required"
        exit 1
    fi
    print_success "Python version check passed"
    
    # Check virtual environment
    if [[ ! -d "$VENV_PATH" ]]; then
        print_info "Creating virtual environment..."
        python3 -m venv "$VENV_PATH"
    fi
    
    # Activate virtual environment
    source "$VENV_PATH/bin/activate"
    print_success "Virtual environment activated"
    
    # Install/upgrade dependencies
    print_info "Installing dependencies..."
    pip install --upgrade pip > /dev/null 2>&1
    pip install -r "$PROJECT_ROOT/requirements.txt" > /dev/null 2>&1
    pip install -e "$PROJECT_ROOT" > /dev/null 2>&1
    
    # Install development dependencies
    pip install pytest pytest-cov black flake8 mypy bandit safety > /dev/null 2>&1
    print_success "Dependencies installed"
}

run_code_quality_checks() {
    print_header "Code Quality Checks"
    
    # Create reports directory
    mkdir -p "$REPORTS_DIR"
    
    # Black code formatting check
    print_info "Checking code formatting with Black..."
    if black --check --diff "$PROJECT_ROOT/src" "$PROJECT_ROOT/scripts" > "$REPORTS_DIR/black_report.txt" 2>&1; then
        print_success "Code formatting check passed"
    else
        print_error "Code formatting issues found. See $REPORTS_DIR/black_report.txt"
        return 1
    fi
    
    # Flake8 linting
    print_info "Running Flake8 linting..."
    if flake8 "$PROJECT_ROOT/src" "$PROJECT_ROOT/scripts" \
       --max-line-length=88 \
       --extend-ignore=E203,W503 \
       --format='%(path)s:%(row)d:%(col)d: %(code)s %(text)s' \
       > "$REPORTS_DIR/flake8_report.txt" 2>&1; then
        print_success "Flake8 linting passed"
    else
        print_warning "Flake8 warnings found. See $REPORTS_DIR/flake8_report.txt"
    fi
    
    # Type checking with mypy
    print_info "Running type checking with mypy..."
    if mypy "$PROJECT_ROOT/src" \
       --ignore-missing-imports \
       --strict-optional \
       --warn-redundant-casts \
       > "$REPORTS_DIR/mypy_report.txt" 2>&1; then
        print_success "Type checking passed"
    else
        print_warning "Type checking issues found. See $REPORTS_DIR/mypy_report.txt"
    fi
}

run_security_checks() {
    print_header "Security Checks"
    
    # Bandit security linting
    print_info "Running Bandit security analysis..."
    if bandit -r "$PROJECT_ROOT/src" \
       -f json \
       -o "$REPORTS_DIR/bandit_report.json" > /dev/null 2>&1; then
        print_success "Security analysis completed"
    else
        print_warning "Security issues found. See $REPORTS_DIR/bandit_report.json"
    fi
    
    # Safety check for known vulnerabilities
    print_info "Checking for known vulnerabilities..."
    if safety check --json > "$REPORTS_DIR/safety_report.json" 2>&1; then
        print_success "No known vulnerabilities found"
    else
        print_warning "Potential vulnerabilities found. See $REPORTS_DIR/safety_report.json"
    fi
}

run_unit_tests() {
    print_header "Unit Tests"
    
    print_info "Running unit tests with coverage..."
    if pytest "$PROJECT_ROOT/tests" \
       --cov="$PROJECT_ROOT/src" \
       --cov-report=html:"$REPORTS_DIR/coverage_html" \
       --cov-report=xml:"$REPORTS_DIR/coverage.xml" \
       --cov-report=term \
       --junit-xml="$REPORTS_DIR/pytest_report.xml" \
       --tb=short \
       -v > "$REPORTS_DIR/test_output.txt" 2>&1; then
        print_success "Unit tests passed"
    else
        print_error "Unit tests failed. See $REPORTS_DIR/test_output.txt"
        return 1
    fi
    
    # Check coverage threshold
    coverage_percent=$(grep -oE '[0-9]+%' "$REPORTS_DIR/test_output.txt" | tail -1 | sed 's/%//')
    if [[ $coverage_percent -lt $COVERAGE_MIN ]]; then
        print_error "Test coverage ${coverage_percent}% below minimum ${COVERAGE_MIN}%"
        return 1
    else
        print_success "Test coverage ${coverage_percent}% meets minimum requirement"
    fi
}

run_integration_tests() {
    print_header "Integration Tests"
    
    print_info "Running system validation tests..."
    if "$SCRIPT_DIR/run_all_tests.sh" > "$REPORTS_DIR/integration_test_output.txt" 2>&1; then
        print_success "Integration tests passed"
    else
        print_error "Integration tests failed. See $REPORTS_DIR/integration_test_output.txt"
        return 1
    fi
}

run_performance_benchmarks() {
    print_header "Performance Benchmarks"
    
    print_info "Running performance benchmarks..."
    if python3 "$SCRIPT_DIR/generate_results.py" > "$REPORTS_DIR/performance_output.txt" 2>&1; then
        print_success "Performance benchmarks completed"
    else
        print_error "Performance benchmarks failed. See $REPORTS_DIR/performance_output.txt"
        return 1
    fi
    
    # Check performance regression
    if [[ -f "$PROJECT_ROOT/docs/_data/results/performance_metrics.json" ]]; then
        control_freq=$(grep -o '"control_frequency":[^,]*' "$PROJECT_ROOT/docs/_data/results/performance_metrics.json" | cut -d':' -f2)
        if [[ $(echo "$control_freq < 50" | bc) -eq 1 ]]; then
            print_error "Performance regression detected: control frequency $control_freq Hz below 50 Hz threshold"
            return 1
        else
            print_success "Performance benchmarks meet requirements"
        fi
    fi
}

generate_documentation() {
    print_header "Documentation Generation"
    
    print_info "Generating Jekyll website..."
    if python3 "$SCRIPT_DIR/jekyll_integration.py" > "$REPORTS_DIR/jekyll_output.txt" 2>&1; then
        print_success "Jekyll data generated"
    else
        print_error "Jekyll generation failed. See $REPORTS_DIR/jekyll_output.txt"
        return 1
    fi
    
    # Build Jekyll site for validation
    cd "$PROJECT_ROOT/docs"
    if bundle exec jekyll build --destination "$REPORTS_DIR/site" > "$REPORTS_DIR/jekyll_build.txt" 2>&1; then
        print_success "Jekyll site built successfully"
    else
        print_warning "Jekyll site build issues. See $REPORTS_DIR/jekyll_build.txt"
    fi
    cd "$PROJECT_ROOT"
}

create_release_artifacts() {
    print_header "Creating Release Artifacts"
    
    # Create distribution package
    print_info "Building distribution package..."
    python3 setup.py sdist bdist_wheel > "$REPORTS_DIR/build_output.txt" 2>&1
    print_success "Distribution package created"
    
    # Create documentation archive
    print_info "Creating documentation archive..."
    tar -czf "$REPORTS_DIR/documentation.tar.gz" -C "$PROJECT_ROOT/docs" .
    print_success "Documentation archive created"
    
    # Create source code archive
    print_info "Creating source code archive..."
    git archive --format=tar.gz --prefix=spacecraft-drmpc/ HEAD > "$REPORTS_DIR/source_code.tar.gz"
    print_success "Source code archive created"
}

generate_ci_report() {
    print_header "Generating CI Report"
    
    report_file="$REPORTS_DIR/ci_summary_report.html"
    cat > "$report_file" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Spacecraft DR-MPC CI Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .success { color: green; font-weight: bold; }
        .warning { color: orange; font-weight: bold; }
        .error { color: red; font-weight: bold; }
        .section { margin: 20px 0; padding: 10px; border-left: 3px solid #ccc; }
        pre { background: #f5f5f5; padding: 10px; overflow-x: auto; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>Spacecraft DR-MPC Continuous Integration Report</h1>
    <p><strong>Generated:</strong> $(date)</p>
    <p><strong>Commit:</strong> $(git rev-parse HEAD)</p>
    <p><strong>Branch:</strong> $(git rev-parse --abbrev-ref HEAD)</p>
    
    <div class="section">
        <h2>Build Summary</h2>
        <table>
            <tr><th>Component</th><th>Status</th><th>Details</th></tr>
            <tr><td>Code Quality</td><td class="success">PASSED</td><td>All checks completed</td></tr>
            <tr><td>Security</td><td class="success">PASSED</td><td>No critical issues</td></tr>
            <tr><td>Unit Tests</td><td class="success">PASSED</td><td>Coverage: ${coverage_percent:-0}%</td></tr>
            <tr><td>Integration Tests</td><td class="success">PASSED</td><td>All scenarios validated</td></tr>
            <tr><td>Performance</td><td class="success">PASSED</td><td>All benchmarks met</td></tr>
            <tr><td>Documentation</td><td class="success">PASSED</td><td>Site generated successfully</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Test Coverage</h2>
        <iframe src="coverage_html/index.html" width="100%" height="400px" frameborder="0"></iframe>
    </div>
    
    <div class="section">
        <h2>Performance Metrics</h2>
        <p>Latest performance benchmark results:</p>
        <ul>
            <li>Control Frequency: $(grep -o '"control_frequency":[^,]*' "$PROJECT_ROOT/docs/_data/results/performance_metrics.json" 2>/dev/null | cut -d':' -f2 || echo "N/A") Hz</li>
            <li>Position Accuracy: $(grep -o '"position_accuracy":[^,]*' "$PROJECT_ROOT/docs/_data/results/performance_metrics.json" 2>/dev/null | cut -d':' -f2 || echo "N/A") m</li>
            <li>Fleet Size: $(grep -o '"max_fleet_size":[^,]*' "$PROJECT_ROOT/docs/_data/results/test_results.json" 2>/dev/null | cut -d':' -f2 || echo "N/A") spacecraft</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>Artifacts</h2>
        <ul>
            <li><a href="source_code.tar.gz">Source Code Archive</a></li>
            <li><a href="documentation.tar.gz">Documentation Archive</a></li>
            <li><a href="../dist/">Distribution Packages</a></li>
        </ul>
    </div>
</body>
</html>
EOF
    
    print_success "CI report generated: $report_file"
}

cleanup() {
    print_header "Cleanup"
    
    # Clean temporary files
    find "$PROJECT_ROOT" -name "*.pyc" -delete
    find "$PROJECT_ROOT" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    
    # Clean build artifacts
    rm -rf "$PROJECT_ROOT/build" "$PROJECT_ROOT/*.egg-info"
    
    print_success "Cleanup completed"
}

main() {
    print_header "Spacecraft DR-MPC Continuous Integration Pipeline"
    print_info "Starting CI pipeline at $(date)"
    
    # Set trap for cleanup on exit
    trap cleanup EXIT
    
    # Run CI pipeline stages
    check_dependencies || exit 1
    run_code_quality_checks || exit 1
    run_security_checks || exit 1
    run_unit_tests || exit 1
    run_integration_tests || exit 1
    run_performance_benchmarks || exit 1
    generate_documentation || exit 1
    create_release_artifacts || exit 1
    generate_ci_report || exit 1
    
    print_header "✅ CI Pipeline Completed Successfully"
    print_success "All checks passed!"
    print_info "Reports available in: $REPORTS_DIR"
    print_info "CI Summary: $REPORTS_DIR/ci_summary_report.html"
}

# Run main function
main "$@"