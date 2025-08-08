#!/bin/bash
set -euo pipefail

# Production Entrypoint Script for Spacecraft Simulation
# This script handles initialization, health checks, and service startup

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        INFO)
            echo -e "${GREEN}[INFO]${NC} ${timestamp} - ${message}" ;;
        WARN)
            echo -e "${YELLOW}[WARN]${NC} ${timestamp} - ${message}" ;;
        ERROR)
            echo -e "${RED}[ERROR]${NC} ${timestamp} - ${message}" >&2 ;;
        DEBUG)
            if [[ "${LOG_LEVEL:-INFO}" == "DEBUG" ]]; then
                echo -e "${BLUE}[DEBUG]${NC} ${timestamp} - ${message}"
            fi
            ;;
    esac
}

# Error handler
error_handler() {
    local line_number=$1
    log ERROR "Script failed at line ${line_number}"
    exit 1
}

trap 'error_handler ${LINENO}' ERR

# Signal handlers for graceful shutdown
cleanup() {
    log INFO "Received shutdown signal, performing cleanup..."
    
    # Stop background processes
    if [[ -n "${BACKGROUND_PIDS:-}" ]]; then
        for pid in $BACKGROUND_PIDS; do
            if kill -0 "$pid" 2>/dev/null; then
                log INFO "Stopping process $pid"
                kill -TERM "$pid" || true
            fi
        done
    fi
    
    # Stop supervisor if running
    if pgrep supervisord >/dev/null; then
        log INFO "Stopping supervisord"
        supervisorctl shutdown || true
    fi
    
    log INFO "Cleanup completed"
    exit 0
}

trap cleanup SIGTERM SIGINT

# Environment validation
validate_environment() {
    log INFO "Validating environment variables..."
    
    local required_vars=(
        "DATABASE_URL"
        "REDIS_URL"
        "JWT_SECRET"
        "API_SECRET_KEY"
    )
    
    local missing_vars=()
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        log ERROR "Missing required environment variables: ${missing_vars[*]}"
        exit 1
    fi
    
    # Validate database URL format
    if [[ ! "$DATABASE_URL" =~ ^postgresql:// ]]; then
        log ERROR "Invalid DATABASE_URL format. Expected postgresql:// URL"
        exit 1
    fi
    
    # Validate Redis URL format
    if [[ ! "$REDIS_URL" =~ ^redis:// ]]; then
        log ERROR "Invalid REDIS_URL format. Expected redis:// URL"
        exit 1
    fi
    
    log INFO "Environment validation completed successfully"
}

# Database connectivity check
check_database() {
    log INFO "Checking database connectivity..."
    
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if python -c "
import psycopg2
import os
import sys
try:
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    conn.close()
    print('Database connection successful')
    sys.exit(0)
except Exception as e:
    print(f'Database connection failed: {e}')
    sys.exit(1)
" >/dev/null 2>&1; then
            log INFO "Database connection established"
            return 0
        else
            log WARN "Database connection attempt $attempt/$max_attempts failed, retrying in 5 seconds..."
            sleep 5
            ((attempt++))
        fi
    done
    
    log ERROR "Failed to connect to database after $max_attempts attempts"
    exit 1
}

# Redis connectivity check
check_redis() {
    log INFO "Checking Redis connectivity..."
    
    local max_attempts=15
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if python -c "
import redis
import os
import sys
try:
    r = redis.from_url(os.environ['REDIS_URL'])
    r.ping()
    print('Redis connection successful')
    sys.exit(0)
except Exception as e:
    print(f'Redis connection failed: {e}')
    sys.exit(1)
" >/dev/null 2>&1; then
            log INFO "Redis connection established"
            return 0
        else
            log WARN "Redis connection attempt $attempt/$max_attempts failed, retrying in 3 seconds..."
            sleep 3
            ((attempt++))
        fi
    done
    
    log ERROR "Failed to connect to Redis after $max_attempts attempts"
    exit 1
}

# System resource checks
check_system_resources() {
    log INFO "Checking system resources..."
    
    # Check available memory
    local available_memory_mb=$(free -m | awk '/^Mem:/{print $7}')
    local required_memory_mb=1000
    
    if [[ $available_memory_mb -lt $required_memory_mb ]]; then
        log WARN "Low available memory: ${available_memory_mb}MB (recommended: ${required_memory_mb}MB+)"
    else
        log INFO "Available memory: ${available_memory_mb}MB"
    fi
    
    # Check disk space
    local available_disk_gb=$(df /app | awk 'NR==2{print int($4/1024/1024)}')
    local required_disk_gb=2
    
    if [[ $available_disk_gb -lt $required_disk_gb ]]; then
        log WARN "Low disk space: ${available_disk_gb}GB (recommended: ${required_disk_gb}GB+)"
    else
        log INFO "Available disk space: ${available_disk_gb}GB"
    fi
    
    # Check CPU cores
    local cpu_cores=$(nproc)
    log INFO "Available CPU cores: $cpu_cores"
}

# License validation
check_licenses() {
    log INFO "Checking software licenses..."
    
    # Check MOSEK license
    if [[ -f "/opt/mosek/licenses/mosek.lic" ]]; then
        log INFO "MOSEK license found"
        
        # Validate license expiry (basic check)
        if python -c "
import mosek
try:
    with mosek.Env() as env:
        print('MOSEK license is valid')
except Exception as e:
    print(f'MOSEK license validation failed: {e}')
    exit(1)
" >/dev/null 2>&1; then
            log INFO "MOSEK license validation successful"
        else
            log WARN "MOSEK license validation failed, falling back to open-source solvers"
        fi
    else
        log WARN "MOSEK license not found, using open-source solvers only"
    fi
}

# Configuration setup
setup_configuration() {
    log INFO "Setting up application configuration..."
    
    # Create necessary directories
    local dirs=("/app/logs" "/app/data" "/app/results" "/app/tmp" "/var/log/spacecraft")
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log DEBUG "Created directory: $dir"
        fi
    done
    
    # Set up logging configuration
    export SPACECRAFT_LOG_FILE="/app/logs/spacecraft.log"
    export SPACECRAFT_ERROR_LOG="/app/logs/spacecraft_error.log"
    
    # Create log files with proper permissions
    touch "$SPACECRAFT_LOG_FILE" "$SPACECRAFT_ERROR_LOG"
    
    # Set up supervisor configuration if not exists
    if [[ ! -f "/etc/supervisor/conf.d/spacecraft.conf" ]]; then
        log WARN "Supervisor configuration not found, creating default"
        cat > /etc/supervisor/conf.d/spacecraft.conf << EOF
[program:spacecraft]
command=python -m src.main
directory=/app
user=spacecraft
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/app/logs/spacecraft.log
stdout_logfile_maxbytes=100MB
stdout_logfile_backups=5
environment=PYTHONPATH=/app
EOF
    fi
}

# Health check function
perform_health_check() {
    log INFO "Performing initial health check..."
    
    # Check if the application starts successfully
    timeout 30s python -c "
import sys
sys.path.append('/app')
try:
    from src.health import check_health
    result = check_health()
    if result['healthy']:
        print('Health check passed')
        sys.exit(0)
    else:
        print(f'Health check failed: {result}')
        sys.exit(1)
except Exception as e:
    print(f'Health check error: {e}')
    sys.exit(1)
" || {
        log ERROR "Initial health check failed"
        exit 1
    }
    
    log INFO "Initial health check passed"
}

# Service startup functions
start_spacecraft_app() {
    log INFO "Starting spacecraft application..."
    
    # Start with supervisor for process management
    if command -v supervisord >/dev/null; then
        log INFO "Starting application with supervisor"
        exec supervisord -n -c /etc/supervisor/supervisord.conf
    else
        log INFO "Starting application directly"
        exec python -m src.main
    fi
}

start_coordinator() {
    log INFO "Starting spacecraft coordinator service..."
    exec python -m src.coordination.coordinator
}

start_migration() {
    log INFO "Running database migrations..."
    python /app/migrate.py
    log INFO "Migrations completed"
}

start_backup() {
    log INFO "Starting backup service..."
    exec python -m src.backup.service
}

start_metrics() {
    log INFO "Starting metrics collection service..."
    exec python -m src.metrics.collector
}

# Main execution logic
main() {
    local service_type="${1:-spacecraft-app}"
    
    log INFO "Starting spacecraft simulation service: $service_type"
    log INFO "Build version: ${BUILD_VERSION:-unknown}"
    log INFO "Build commit: ${BUILD_COMMIT:-unknown}"
    log INFO "Environment: ${APP_ENV:-production}"
    
    # Common initialization for all services
    validate_environment
    check_system_resources
    setup_configuration
    
    # Service-specific initialization and startup
    case "$service_type" in
        spacecraft-app)
            check_database
            check_redis
            check_licenses
            perform_health_check
            start_spacecraft_app
            ;;
        coordinator)
            check_database
            check_redis
            start_coordinator
            ;;
        migration)
            check_database
            start_migration
            ;;
        backup)
            check_database
            start_backup
            ;;
        metrics)
            start_metrics
            ;;
        health-check)
            check_database
            check_redis
            perform_health_check
            log INFO "Health check completed successfully"
            exit 0
            ;;
        *)
            log ERROR "Unknown service type: $service_type"
            log INFO "Available services: spacecraft-app, coordinator, migration, backup, metrics, health-check"
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"