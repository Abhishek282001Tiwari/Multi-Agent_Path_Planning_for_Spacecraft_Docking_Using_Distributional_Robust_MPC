#!/usr/bin/env python3
"""
Production Health Check Script for Spacecraft Simulation
Comprehensive health monitoring for all application components
"""

import os
import sys
import time
import json
import logging
import requests
import psycopg2
import redis
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class HealthChecker:
    """Comprehensive health checker for spacecraft simulation components"""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        self.timeout = 30  # Total health check timeout
        
        # Configuration from environment
        self.database_url = os.environ.get('DATABASE_URL')
        self.redis_url = os.environ.get('REDIS_URL')
        self.app_port = os.environ.get('APP_PORT', '8080')
        self.grpc_port = os.environ.get('GRPC_PORT', '9090')
        self.metrics_port = os.environ.get('METRICS_PORT', '8081')
        
        # Health check thresholds
        self.max_memory_usage = float(os.environ.get('MAX_MEMORY_USAGE', '90.0'))
        self.max_cpu_usage = float(os.environ.get('MAX_CPU_USAGE', '95.0'))
        self.min_disk_space_gb = float(os.environ.get('MIN_DISK_SPACE_GB', '1.0'))
    
    def check_http_endpoint(self, endpoint: str, timeout: int = 5) -> Dict[str, Any]:
        """Check HTTP endpoint health"""
        try:
            response = requests.get(endpoint, timeout=timeout)
            return {
                'healthy': response.status_code == 200,
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'error': None
            }
        except requests.exceptions.RequestException as e:
            return {
                'healthy': False,
                'status_code': None,
                'response_time': None,
                'error': str(e)
            }
    
    def check_database(self) -> Dict[str, Any]:
        """Check database connectivity and health"""
        if not self.database_url:
            return {'healthy': False, 'error': 'DATABASE_URL not configured'}
        
        try:
            conn = psycopg2.connect(self.database_url)
            cursor = conn.cursor()
            
            # Basic connectivity test
            cursor.execute('SELECT 1;')
            result = cursor.fetchone()
            
            # Check database size and connection count
            cursor.execute("""
                SELECT 
                    pg_database_size(current_database()) as db_size,
                    count(*) as connection_count
                FROM pg_stat_activity 
                WHERE state = 'active';
            """)
            db_info = cursor.fetchone()
            
            # Check for long-running queries
            cursor.execute("""
                SELECT count(*) 
                FROM pg_stat_activity 
                WHERE state = 'active' 
                AND now() - query_start > interval '5 minutes';
            """)
            long_queries = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            return {
                'healthy': True,
                'database_size_bytes': db_info[0],
                'active_connections': db_info[1],
                'long_running_queries': long_queries,
                'error': None
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }
    
    def check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity and health"""
        if not self.redis_url:
            return {'healthy': False, 'error': 'REDIS_URL not configured'}
        
        try:
            r = redis.from_url(self.redis_url)
            
            # Basic connectivity test
            ping_result = r.ping()
            
            # Get Redis info
            info = r.info()
            
            # Check memory usage
            used_memory = info.get('used_memory', 0)
            max_memory = info.get('maxmemory', 0)
            memory_usage_percent = (used_memory / max_memory * 100) if max_memory > 0 else 0
            
            # Check connected clients
            connected_clients = info.get('connected_clients', 0)
            
            return {
                'healthy': ping_result,
                'used_memory_bytes': used_memory,
                'max_memory_bytes': max_memory,
                'memory_usage_percent': memory_usage_percent,
                'connected_clients': connected_clients,
                'redis_version': info.get('redis_version'),
                'error': None
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            # Memory check
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            
            mem_total = int([line for line in meminfo.split('\n') if 'MemTotal' in line][0].split()[1]) * 1024
            mem_available = int([line for line in meminfo.split('\n') if 'MemAvailable' in line][0].split()[1]) * 1024
            memory_usage_percent = ((mem_total - mem_available) / mem_total) * 100
            
            # CPU check (simplified)
            with open('/proc/loadavg', 'r') as f:
                load_avg = float(f.read().split()[0])
            
            cpu_count = os.cpu_count() or 1
            cpu_usage_percent = (load_avg / cpu_count) * 100
            
            # Disk space check
            statvfs = os.statvfs('/app')
            disk_free_bytes = statvfs.f_bavail * statvfs.f_frsize
            disk_total_bytes = statvfs.f_blocks * statvfs.f_frsize
            disk_usage_percent = ((disk_total_bytes - disk_free_bytes) / disk_total_bytes) * 100
            disk_free_gb = disk_free_bytes / (1024**3)
            
            # Process count
            try:
                process_count = int(subprocess.check_output(['ps', 'aux'], text=True).count('\n'))
            except:
                process_count = None
            
            # Check if resources are within acceptable limits
            memory_healthy = memory_usage_percent < self.max_memory_usage
            cpu_healthy = cpu_usage_percent < self.max_cpu_usage
            disk_healthy = disk_free_gb > self.min_disk_space_gb
            
            return {
                'healthy': memory_healthy and cpu_healthy and disk_healthy,
                'memory': {
                    'total_bytes': mem_total,
                    'available_bytes': mem_available,
                    'usage_percent': memory_usage_percent,
                    'healthy': memory_healthy
                },
                'cpu': {
                    'load_average': load_avg,
                    'cpu_count': cpu_count,
                    'usage_percent': cpu_usage_percent,
                    'healthy': cpu_healthy
                },
                'disk': {
                    'total_bytes': disk_total_bytes,
                    'free_bytes': disk_free_bytes,
                    'free_gb': disk_free_gb,
                    'usage_percent': disk_usage_percent,
                    'healthy': disk_healthy
                },
                'process_count': process_count,
                'error': None
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }
    
    def check_application_endpoints(self) -> Dict[str, Any]:
        """Check application-specific endpoints"""
        endpoints = {
            'health': f'http://localhost:{self.app_port}/health',
            'ready': f'http://localhost:{self.app_port}/ready',
            'metrics': f'http://localhost:{self.metrics_port}/metrics',
            'simulation_status': f'http://localhost:{self.app_port}/api/v1/status'
        }
        
        results = {}
        for name, url in endpoints.items():
            results[name] = self.check_http_endpoint(url)
        
        # Overall endpoint health
        all_healthy = all(result.get('healthy', False) for result in results.values())
        
        return {
            'healthy': all_healthy,
            'endpoints': results,
            'error': None if all_healthy else 'One or more endpoints unhealthy'
        }
    
    def check_simulation_health(self) -> Dict[str, Any]:
        """Check spacecraft simulation specific health metrics"""
        try:
            # Check if any simulations are stuck or consuming excessive resources
            response = requests.get(
                f'http://localhost:{self.app_port}/api/v1/simulations/health',
                timeout=5
            )
            
            if response.status_code == 200:
                simulation_data = response.json()
                
                # Check for stuck simulations
                stuck_simulations = simulation_data.get('stuck_simulations', [])
                active_simulations = simulation_data.get('active_simulations', 0)
                failed_simulations = simulation_data.get('failed_simulations', 0)
                
                # Check solver performance
                avg_solve_time = simulation_data.get('avg_solve_time_ms', 0)
                max_solve_time = simulation_data.get('max_solve_time_ms', 0)
                
                # Health criteria
                solver_healthy = avg_solve_time < 1000 and max_solve_time < 5000  # ms
                simulation_healthy = len(stuck_simulations) == 0 and failed_simulations < 5
                
                return {
                    'healthy': solver_healthy and simulation_healthy,
                    'active_simulations': active_simulations,
                    'failed_simulations': failed_simulations,
                    'stuck_simulations': len(stuck_simulations),
                    'avg_solve_time_ms': avg_solve_time,
                    'max_solve_time_ms': max_solve_time,
                    'solver_healthy': solver_healthy,
                    'simulation_healthy': simulation_healthy,
                    'error': None
                }
            else:
                return {
                    'healthy': False,
                    'error': f'Simulation health endpoint returned {response.status_code}'
                }
                
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check external dependencies"""
        checks = {}
        
        # MOSEK license check
        try:
            import mosek
            with mosek.Env() as env:
                checks['mosek'] = {'healthy': True, 'error': None}
        except Exception as e:
            checks['mosek'] = {'healthy': False, 'error': str(e)}
        
        # Check if required Python packages are available
        required_packages = ['numpy', 'scipy', 'cvxpy', 'psycopg2', 'redis', 'fastapi']
        for package in required_packages:
            try:
                __import__(package)
                checks[f'package_{package}'] = {'healthy': True, 'error': None}
            except ImportError as e:
                checks[f'package_{package}'] = {'healthy': False, 'error': str(e)}
        
        all_healthy = all(check.get('healthy', False) for check in checks.values())
        
        return {
            'healthy': all_healthy,
            'dependencies': checks,
            'error': None if all_healthy else 'One or more dependencies unhealthy'
        }
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks concurrently"""
        health_checks = {
            'database': self.check_database,
            'redis': self.check_redis,
            'system_resources': self.check_system_resources,
            'application_endpoints': self.check_application_endpoints,
            'simulation_health': self.check_simulation_health,
            'dependencies': self.check_dependencies
        }
        
        with ThreadPoolExecutor(max_workers=6) as executor:
            # Submit all health checks
            future_to_check = {
                executor.submit(check_func): check_name 
                for check_name, check_func in health_checks.items()
            }
            
            # Collect results with timeout
            results = {}
            for future in future_to_check:
                check_name = future_to_check[future]
                try:
                    results[check_name] = future.result(timeout=10)
                except FutureTimeoutError:
                    results[check_name] = {
                        'healthy': False,
                        'error': 'Health check timed out'
                    }
                except Exception as e:
                    results[check_name] = {
                        'healthy': False,
                        'error': str(e)
                    }
        
        # Calculate overall health
        all_checks_healthy = all(
            result.get('healthy', False) for result in results.values()
        )
        
        # Calculate execution time
        execution_time = time.time() - self.start_time
        
        return {
            'healthy': all_checks_healthy,
            'timestamp': datetime.utcnow().isoformat(),
            'execution_time_seconds': round(execution_time, 3),
            'checks': results,
            'summary': {
                'total_checks': len(results),
                'healthy_checks': sum(1 for r in results.values() if r.get('healthy', False)),
                'unhealthy_checks': sum(1 for r in results.values() if not r.get('healthy', False))
            }
        }

def main():
    """Main health check execution"""
    try:
        checker = HealthChecker()
        results = checker.run_health_checks()
        
        # Output results
        print(json.dumps(results, indent=2))
        
        # Exit with appropriate code
        if results['healthy']:
            logger.info("Health check passed")
            sys.exit(0)
        else:
            logger.error("Health check failed")
            
            # Log specific failures
            for check_name, check_result in results['checks'].items():
                if not check_result.get('healthy', False):
                    error = check_result.get('error', 'Unknown error')
                    logger.error(f"{check_name}: {error}")
            
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Health check execution failed: {e}")
        print(json.dumps({
            'healthy': False,
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e)
        }))
        sys.exit(1)

if __name__ == '__main__':
    main()