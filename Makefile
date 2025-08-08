.PHONY: install test clean docs docker format lint help

# Colors for output
YELLOW := \033[1;33m
GREEN := \033[1;32m
BLUE := \033[1;34m
RED := \033[1;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Spacecraft DRMPC - Available Commands:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'

install: ## Install the package and dependencies
	@echo "$(YELLOW)Installing spacecraft-drmpc package...$(NC)"
	pip install -e .
	pip install -e ".[dev]"
	@echo "$(GREEN)Installation completed!$(NC)"

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