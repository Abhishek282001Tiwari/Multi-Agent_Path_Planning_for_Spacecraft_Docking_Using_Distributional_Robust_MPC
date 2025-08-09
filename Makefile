.PHONY: help install dev-install test test-all lint format security clean build docs serve demo benchmark validate

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Python and pip executables
PYTHON := python3
PIP := pip
PYTEST := pytest

# Project configuration
PROJECT_NAME := spacecraft-drmpc
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs

# Default target
help: ## Show this help message
	@echo "$(BLUE)Multi-Agent Spacecraft Docking System - Development Commands$(NC)"
	@echo "=============================================================="
	@echo ""
	@echo "$(YELLOW)Quick Start:$(NC)"
	@echo "  $(GREEN)make install$(NC)     Install for end users"
	@echo "  $(GREEN)make dev$(NC)         Complete development setup"
	@echo "  $(GREEN)make demo$(NC)        Run quick demonstration"
	@echo ""
	@echo "$(YELLOW)Development:$(NC)"
	@echo "  $(GREEN)make test$(NC)        Run unit tests"
	@echo "  $(GREEN)make test-all$(NC)    Run all tests with coverage"
	@echo "  $(GREEN)make lint$(NC)        Check code quality"
	@echo "  $(GREEN)make format$(NC)      Format code"
	@echo "  $(GREEN)make security$(NC)    Security analysis"
	@echo ""
	@echo "$(YELLOW)Documentation:$(NC)"
	@echo "  $(GREEN)make docs$(NC)        Build documentation"
	@echo "  $(GREEN)make serve$(NC)       Serve docs locally"
	@echo ""
	@echo "$(YELLOW)Validation:$(NC)"
	@echo "  $(GREEN)make validate$(NC)    Full system validation"
	@echo "  $(GREEN)make benchmark$(NC)   Performance benchmarks"
	@echo ""

# Installation targets
install: ## Install package and dependencies for end users
	@echo "$(BLUE)Installing Spacecraft Docking System...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "$(GREEN)✅ Installation complete!$(NC)"

dev-install: ## Install development dependencies and setup
	@echo "$(BLUE)Setting up development environment...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "$(GREEN)✅ Development environment ready!$(NC)"

install-deps: ## Install only dependencies  
	@echo "$(YELLOW)Installing dependencies...$(NC)"
	pip install -r requirements.txt
	@echo "$(GREEN)Dependencies installed!$(NC)"

test: ## Run all tests
	@echo "$(YELLOW)Running test suite...$(NC)"
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "$(GREEN)Tests completed!$(NC)"

test-unit: ## Run unit tests only
	@echo "$(YELLOW)Running unit tests...$(NC)"
	pytest tests/unit/ -v
	@echo "$(GREEN)Unit tests completed!$(NC)"

test-integration: ## Run integration tests only
	@echo "$(YELLOW)Running integration tests...$(NC)"
	pytest tests/integration/ -v
	@echo "$(GREEN)Integration tests completed!$(NC)"

test-performance: ## Run performance tests
	@echo "$(YELLOW)Running performance tests...$(NC)"
	pytest tests/performance/ -v
	@echo "$(GREEN)Performance tests completed!$(NC)"

clean: ## Clean up build artifacts and cache files
	@echo "$(YELLOW)Cleaning up...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf logs/ outputs/ temp/
	@echo "$(GREEN)Cleanup completed!$(NC)"

format: ## Format code with black and ruff
	@echo "$(YELLOW)Formatting code...$(NC)"
	black src/ tests/
	ruff check src/ tests/ --fix
	@echo "$(GREEN)Code formatting completed!$(NC)"

lint: ## Run linting checks
	@echo "$(YELLOW)Running linting checks...$(NC)"
	flake8 src/ tests/
	ruff check src/ tests/
	mypy src/
	@echo "$(GREEN)Linting completed!$(NC)"

typecheck: ## Run type checking with mypy
	@echo "$(YELLOW)Running type checks...$(NC)"
	mypy src/
	@echo "$(GREEN)Type checking completed!$(NC)"

docs: ## Build documentation
	@echo "$(YELLOW)Building documentation...$(NC)"
	cd docs && make html
	@echo "$(GREEN)Documentation built! Open docs/_build/html/index.html$(NC)"

docker: ## Build and start Docker containers
	@echo "$(YELLOW)Building Docker containers...$(NC)"
	docker build -t spacecraft-drmpc .
	docker-compose up --build
	@echo "$(GREEN)Docker containers started!$(NC)"

docker-prod: ## Build production Docker image
	@echo "$(YELLOW)Building production Docker image...$(NC)"
	docker build -f Dockerfile.prod -t spacecraft-drmpc:prod .
	@echo "$(GREEN)Production image built!$(NC)"

simulation: ## Run a basic simulation
	@echo "$(YELLOW)Running basic simulation...$(NC)"
	python run_simulation.py --scenario single_docking --visualize
	@echo "$(GREEN)Simulation completed!$(NC)"

benchmark: ## Run performance benchmarks
	@echo "$(YELLOW)Running performance benchmarks...$(NC)"
	python scripts/performance_benchmark.py
	@echo "$(GREEN)Benchmarks completed!$(NC)"

security-scan: ## Run security vulnerability scan
	@echo "$(YELLOW)Running security scan...$(NC)"
	safety check
	bandit -r src/
	@echo "$(GREEN)Security scan completed!$(NC)"

pre-commit: format lint test ## Run pre-commit checks (format, lint, test)
	@echo "$(GREEN)Pre-commit checks passed!$(NC)"

dev-setup: ## Set up development environment
	@echo "$(YELLOW)Setting up development environment...$(NC)"
	python -m venv spacecraft_env
	@echo "$(BLUE)Activate with: source spacecraft_env/bin/activate$(NC)"
	@echo "$(BLUE)Then run: make install$(NC)"

ci: clean format lint test ## Run full CI pipeline
	@echo "$(GREEN)CI pipeline completed successfully!$(NC)"

# Development utilities
dev-serve: ## Start development server with hot reload
	@echo "$(YELLOW)Starting development server...$(NC)"
	python -m spacecraft_drmpc.visualization.dashboard.mission_control --debug

monitor: ## Start system monitoring
	@echo "$(YELLOW)Starting system monitoring...$(NC)"
	python -m spacecraft_drmpc.monitoring.system_logger

# Quick shortcuts
quick-test: ## Run quick smoke tests
	pytest tests/unit/test_agents.py -v

quick-sim: ## Run quick simulation test
	python -c "from spacecraft_drmpc.simulations.simulator import SpacecraftSimulator; sim = SpacecraftSimulator('single_docking'); print('Simulation initialized successfully!')"

# Demo and validation targets
demo: ## Run quick demonstration
	@echo "$(BLUE)🚀 Running Spacecraft Docking Demo...$(NC)"
	$(PYTHON) scripts/quick_test.py
	@echo "$(GREEN)✅ Demo complete! Check results in docs/_data/$(NC)"

benchmark: ## Run performance benchmarks  
	@echo "$(BLUE)Running comprehensive benchmarks...$(NC)"
	@mkdir -p reports
	$(PYTHON) scripts/generate_results.py
	$(PYTHON) scripts/generate_plots.py
	@echo "$(GREEN)✅ Benchmarks complete! Results in docs/_data/results/$(NC)"

validate: ## Run full system validation
	@echo "$(BLUE)Running full system validation...$(NC)"
	@echo "$(YELLOW)→ Quick validation...$(NC)"
	$(PYTHON) scripts/quick_test.py
	@echo "$(YELLOW)→ Running tests...$(NC)"
	make test
	@echo "$(GREEN)✅ System validation complete!$(NC)"

validate-quick: ## Run quick validation (30 seconds)
	@echo "$(BLUE)Running quick system validation...$(NC)"
	$(PYTHON) scripts/quick_test.py
	@echo "$(GREEN)✅ Quick validation passed!$(NC)"

serve: ## Serve documentation locally
	@echo "$(BLUE)Starting documentation server...$(NC)"
	$(PYTHON) scripts/jekyll_integration.py
	@if command -v bundle >/dev/null 2>&1; then \
		cd $(DOCS_DIR) && bundle install && bundle exec jekyll serve --host 0.0.0.0 --port 4000; \
	else \
		echo "$(YELLOW)⚠️  Jekyll not installed. Install with: gem install bundler jekyll$(NC)"; \
	fi

# Development workflow shortcuts
dev: dev-install validate-quick ## Full development setup and validation

quick: validate-quick serve ## Quick validation and serve docs

# Special targets for first-time users
first-run: ## Complete first-time setup and demo
	@echo "$(BLUE)🚀 Welcome to Spacecraft Docking System!$(NC)"
	@echo "$(YELLOW)Setting up for first-time use...$(NC)"
	make install
	make demo
	@echo "$(GREEN)🎉 Setup complete! Try 'make help' for more commands.$(NC)"