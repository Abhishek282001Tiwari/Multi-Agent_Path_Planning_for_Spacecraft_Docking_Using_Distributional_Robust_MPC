"""
System Health Monitoring and Alerting System

This module provides comprehensive health monitoring for all spacecraft simulation
components, including predictive health analysis, automated fault detection,
and intelligent alerting with recovery recommendations.
"""

import time
import threading
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
from enum import Enum
import json
import pickle
from pathlib import Path
import sqlite3
import statistics
from concurrent.futures import ThreadPoolExecutor
import queue
import psutil
import os
import subprocess
import socket
import requests
from contextlib import contextmanager
import weakref

# Machine learning imports for predictive health
try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from .system_logger import get_component_logger, ComponentType, PerformanceMetrics
from .performance_metrics import SystemMetrics, AlertLevel, PerformanceAlert


class HealthStatus(Enum):
    """Component health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"
    UNKNOWN = "unknown"


class ComponentStatus(Enum):
    """Component operational status"""
    ONLINE = "online"
    OFFLINE = "offline"
    STARTING = "starting"
    STOPPING = "stopping"
    MAINTENANCE = "maintenance"
    ERROR = "error"


class HealthMetricType(Enum):
    """Types of health metrics"""
    AVAILABILITY = "availability"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    EFFICIENCY = "efficiency"
    SAFETY = "safety"
    RESOURCE_USAGE = "resource_usage"


@dataclass
class HealthMetric:
    """Individual health metric"""
    name: str = ""
    value: float = 0.0
    unit: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    is_higher_better: bool = True
    component: str = ""
    metric_type: HealthMetricType = HealthMetricType.PERFORMANCE
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentHealth:
    """Health status of a single component"""
    component_id: str = ""
    component_type: ComponentType = ComponentType.SYSTEM
    status: ComponentStatus = ComponentStatus.ONLINE
    health_status: HealthStatus = HealthStatus.HEALTHY
    overall_health_score: float = 100.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Health metrics
    metrics: Dict[str, HealthMetric] = field(default_factory=dict)
    
    # Status information
    uptime: float = 0.0  # seconds
    error_count: int = 0
    warning_count: int = 0
    last_error_time: Optional[datetime] = None
    last_warning_time: Optional[datetime] = None
    
    # Predictive health
    predicted_failure_time: Optional[datetime] = None
    confidence_score: float = 0.0
    degradation_trend: float = 0.0  # negative indicates degrading health
    
    # Additional metadata
    version: str = ""
    configuration: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class HealthAlert:
    """Health-related alert"""
    alert_id: str = field(default_factory=lambda: str(time.time()))
    timestamp: datetime = field(default_factory=datetime.now)
    level: AlertLevel = AlertLevel.INFO
    component_id: str = ""
    component_type: ComponentType = ComponentType.SYSTEM
    metric_name: str = ""
    current_value: float = 0.0
    threshold: float = 0.0
    message: str = ""
    description: str = ""
    
    # Alert management
    acknowledged: bool = False
    acknowledged_by: str = ""
    acknowledged_time: Optional[datetime] = None
    resolved: bool = False
    resolved_time: Optional[datetime] = None
    resolution_notes: str = ""
    
    # Predictive information
    predicted_impact: str = ""
    recommended_actions: List[str] = field(default_factory=list)
    estimated_recovery_time: Optional[timedelta] = None
    
    # Context
    related_components: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    """Base class for component health checkers"""
    
    def __init__(self, component_id: str, component_type: ComponentType):
        self.component_id = component_id
        self.component_type = component_type
        self.logger = get_component_logger(ComponentType.MONITORING, f"health.{component_id}")
        self.enabled = True
        self.check_interval = 30.0  # seconds
        self.timeout = 10.0  # seconds
        
    def check_health(self) -> ComponentHealth:
        """Override this method to implement health checking logic"""
        raise NotImplementedError
    
    def is_healthy(self, health: ComponentHealth) -> bool:
        """Determine if component is healthy based on health data"""
        return health.health_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
    
    def calculate_health_score(self, metrics: Dict[str, HealthMetric]) -> float:
        """Calculate overall health score from individual metrics"""
        if not metrics:
            return 0.0
        
        scores = []
        for metric in metrics.values():
            if metric.threshold_critical is not None:
                if metric.is_higher_better:
                    if metric.value >= metric.threshold_critical:
                        score = 100.0
                    elif metric.threshold_warning and metric.value >= metric.threshold_warning:
                        score = 50.0
                    else:
                        score = 0.0
                else:
                    if metric.value <= metric.threshold_critical:
                        score = 100.0
                    elif metric.threshold_warning and metric.value <= metric.threshold_warning:
                        score = 50.0
                    else:
                        score = 0.0
                scores.append(score)
        
        return statistics.mean(scores) if scores else 100.0


class SystemHealthChecker(HealthChecker):
    """System-level health checker"""
    
    def __init__(self):
        super().__init__("system", ComponentType.SYSTEM)
        self.process = psutil.Process()
    
    def check_health(self) -> ComponentHealth:
        """Check overall system health"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1.0)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network connectivity
            network_healthy = self._check_network_health()
            
            # Process health
            process_healthy = self._check_process_health()
            
            # Create health metrics
            metrics = {
                'cpu_usage': HealthMetric(
                    name='cpu_usage',
                    value=cpu_percent,
                    unit='percent',
                    threshold_warning=80.0,
                    threshold_critical=95.0,
                    is_higher_better=False,
                    component=self.component_id,
                    metric_type=HealthMetricType.RESOURCE_USAGE
                ),
                'memory_usage': HealthMetric(
                    name='memory_usage',
                    value=memory.percent,
                    unit='percent',
                    threshold_warning=80.0,
                    threshold_critical=90.0,
                    is_higher_better=False,
                    component=self.component_id,
                    metric_type=HealthMetricType.RESOURCE_USAGE
                ),
                'disk_usage': HealthMetric(
                    name='disk_usage',
                    value=disk.percent,
                    unit='percent',
                    threshold_warning=85.0,
                    threshold_critical=95.0,
                    is_higher_better=False,
                    component=self.component_id,
                    metric_type=HealthMetricType.RESOURCE_USAGE
                ),
                'network_connectivity': HealthMetric(
                    name='network_connectivity',
                    value=100.0 if network_healthy else 0.0,
                    unit='boolean',
                    threshold_critical=1.0,
                    is_higher_better=True,
                    component=self.component_id,
                    metric_type=HealthMetricType.AVAILABILITY
                ),
                'process_health': HealthMetric(
                    name='process_health',
                    value=100.0 if process_healthy else 0.0,
                    unit='boolean',
                    threshold_critical=1.0,
                    is_higher_better=True,
                    component=self.component_id,
                    metric_type=HealthMetricType.RELIABILITY
                )
            }
            
            # Calculate overall health
            health_score = self.calculate_health_score(metrics)
            health_status = self._determine_health_status(health_score)
            
            # System uptime
            uptime = time.time() - psutil.boot_time()
            
            return ComponentHealth(
                component_id=self.component_id,
                component_type=self.component_type,
                status=ComponentStatus.ONLINE,
                health_status=health_status,
                overall_health_score=health_score,
                metrics=metrics,
                uptime=uptime
            )
            
        except Exception as e:
            self.logger.error(f"Error checking system health: {e}")
            return ComponentHealth(
                component_id=self.component_id,
                component_type=self.component_type,
                status=ComponentStatus.ERROR,
                health_status=HealthStatus.UNKNOWN,
                overall_health_score=0.0
            )
    
    def _check_network_health(self) -> bool:
        """Check network connectivity"""
        try:
            # Try to connect to a reliable external service
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            return True
        except (socket.timeout, socket.error):
            return False
    
    def _check_process_health(self) -> bool:
        """Check if critical processes are running"""
        try:
            # Check if current process is responsive
            return self.process.is_running() and self.process.status() != 'zombie'
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
    
    def _determine_health_status(self, health_score: float) -> HealthStatus:
        """Determine health status based on score"""
        if health_score >= 90:
            return HealthStatus.HEALTHY
        elif health_score >= 70:
            return HealthStatus.DEGRADED
        elif health_score >= 50:
            return HealthStatus.WARNING
        elif health_score >= 20:
            return HealthStatus.CRITICAL
        else:
            return HealthStatus.FAILED


class DatabaseHealthChecker(HealthChecker):
    """Database health checker"""
    
    def __init__(self, database_url: str):
        super().__init__("database", ComponentType.DATABASE)
        self.database_url = database_url
    
    def check_health(self) -> ComponentHealth:
        """Check database health"""
        try:
            import psycopg2
            import psycopg2.extras
            
            # Connection test
            start_time = time.time()
            conn = psycopg2.connect(self.database_url)
            connection_time = time.time() - start_time
            
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Basic query test
            start_time = time.time()
            cursor.execute("SELECT 1;")
            cursor.fetchone()
            query_time = time.time() - start_time
            
            # Database statistics
            cursor.execute("""
                SELECT 
                    pg_database_size(current_database()) as db_size,
                    count(*) as active_connections
                FROM pg_stat_activity 
                WHERE state = 'active';
            """)
            db_stats = cursor.fetchone()
            
            # Long-running queries
            cursor.execute("""
                SELECT count(*) as long_queries
                FROM pg_stat_activity 
                WHERE state = 'active' 
                AND now() - query_start > interval '1 minute';
            """)
            long_queries = cursor.fetchone()['long_queries']
            
            # Lock information
            cursor.execute("""
                SELECT count(*) as blocked_queries
                FROM pg_stat_activity 
                WHERE waiting = true;
            """)
            blocked_queries = cursor.fetchone()['blocked_queries']
            
            cursor.close()
            conn.close()
            
            # Create health metrics
            metrics = {
                'connection_time': HealthMetric(
                    name='connection_time',
                    value=connection_time * 1000,  # Convert to ms
                    unit='ms',
                    threshold_warning=1000.0,
                    threshold_critical=5000.0,
                    is_higher_better=False,
                    component=self.component_id,
                    metric_type=HealthMetricType.PERFORMANCE
                ),
                'query_time': HealthMetric(
                    name='query_time',
                    value=query_time * 1000,  # Convert to ms
                    unit='ms',
                    threshold_warning=100.0,
                    threshold_critical=1000.0,
                    is_higher_better=False,
                    component=self.component_id,
                    metric_type=HealthMetricType.PERFORMANCE
                ),
                'active_connections': HealthMetric(
                    name='active_connections',
                    value=db_stats['active_connections'],
                    unit='count',
                    threshold_warning=80.0,
                    threshold_critical=100.0,
                    is_higher_better=False,
                    component=self.component_id,
                    metric_type=HealthMetricType.RESOURCE_USAGE
                ),
                'database_size': HealthMetric(
                    name='database_size',
                    value=db_stats['db_size'] / (1024**3),  # Convert to GB
                    unit='GB',
                    threshold_warning=50.0,
                    threshold_critical=100.0,
                    is_higher_better=False,
                    component=self.component_id,
                    metric_type=HealthMetricType.RESOURCE_USAGE
                ),
                'long_running_queries': HealthMetric(
                    name='long_running_queries',
                    value=long_queries,
                    unit='count',
                    threshold_warning=5.0,
                    threshold_critical=10.0,
                    is_higher_better=False,
                    component=self.component_id,
                    metric_type=HealthMetricType.PERFORMANCE
                ),
                'blocked_queries': HealthMetric(
                    name='blocked_queries',
                    value=blocked_queries,
                    unit='count',
                    threshold_warning=1.0,
                    threshold_critical=5.0,
                    is_higher_better=False,
                    component=self.component_id,
                    metric_type=HealthMetricType.RELIABILITY
                )
            }
            
            health_score = self.calculate_health_score(metrics)
            health_status = self._determine_health_status(health_score)
            
            return ComponentHealth(
                component_id=self.component_id,
                component_type=self.component_type,
                status=ComponentStatus.ONLINE,
                health_status=health_status,
                overall_health_score=health_score,
                metrics=metrics
            )
            
        except Exception as e:
            self.logger.error(f"Error checking database health: {e}")
            return ComponentHealth(
                component_id=self.component_id,
                component_type=self.component_type,
                status=ComponentStatus.OFFLINE,
                health_status=HealthStatus.FAILED,
                overall_health_score=0.0
            )
    
    def _determine_health_status(self, health_score: float) -> HealthStatus:
        """Determine health status based on score"""
        if health_score >= 90:
            return HealthStatus.HEALTHY
        elif health_score >= 70:
            return HealthStatus.DEGRADED
        elif health_score >= 50:
            return HealthStatus.WARNING
        elif health_score >= 20:
            return HealthStatus.CRITICAL
        else:
            return HealthStatus.FAILED


class APIHealthChecker(HealthChecker):
    """API endpoint health checker"""
    
    def __init__(self, api_base_url: str):
        super().__init__("api", ComponentType.API)
        self.api_base_url = api_base_url.rstrip('/')
    
    def check_health(self) -> ComponentHealth:
        """Check API health"""
        try:
            endpoints = [
                '/health',
                '/api/v1/status',
                '/metrics'
            ]
            
            metrics = {}
            error_count = 0
            total_response_time = 0
            
            for endpoint in endpoints:
                url = f"{self.api_base_url}{endpoint}"
                
                try:
                    start_time = time.time()
                    response = requests.get(url, timeout=self.timeout)
                    response_time = time.time() - start_time
                    total_response_time += response_time
                    
                    endpoint_name = endpoint.replace('/', '_').strip('_') or 'root'
                    
                    # Response time metric
                    metrics[f'{endpoint_name}_response_time'] = HealthMetric(
                        name=f'{endpoint_name}_response_time',
                        value=response_time * 1000,  # Convert to ms
                        unit='ms',
                        threshold_warning=1000.0,
                        threshold_critical=5000.0,
                        is_higher_better=False,
                        component=self.component_id,
                        metric_type=HealthMetricType.PERFORMANCE
                    )
                    
                    # Status code metric
                    metrics[f'{endpoint_name}_status'] = HealthMetric(
                        name=f'{endpoint_name}_status',
                        value=1.0 if 200 <= response.status_code < 300 else 0.0,
                        unit='boolean',
                        threshold_critical=1.0,
                        is_higher_better=True,
                        component=self.component_id,
                        metric_type=HealthMetricType.AVAILABILITY
                    )
                    
                    if response.status_code >= 400:
                        error_count += 1
                        
                except requests.exceptions.RequestException as e:
                    error_count += 1
                    endpoint_name = endpoint.replace('/', '_').strip('_') or 'root'
                    
                    metrics[f'{endpoint_name}_status'] = HealthMetric(
                        name=f'{endpoint_name}_status',
                        value=0.0,
                        unit='boolean',
                        threshold_critical=1.0,
                        is_higher_better=True,
                        component=self.component_id,
                        metric_type=HealthMetricType.AVAILABILITY,
                        metadata={'error': str(e)}
                    )
            
            # Overall metrics
            availability = (len(endpoints) - error_count) / len(endpoints) * 100
            avg_response_time = total_response_time / len(endpoints) * 1000
            
            metrics['availability'] = HealthMetric(
                name='availability',
                value=availability,
                unit='percent',
                threshold_warning=95.0,
                threshold_critical=90.0,
                is_higher_better=True,
                component=self.component_id,
                metric_type=HealthMetricType.AVAILABILITY
            )
            
            metrics['average_response_time'] = HealthMetric(
                name='average_response_time',
                value=avg_response_time,
                unit='ms',
                threshold_warning=500.0,
                threshold_critical=2000.0,
                is_higher_better=False,
                component=self.component_id,
                metric_type=HealthMetricType.PERFORMANCE
            )
            
            health_score = self.calculate_health_score(metrics)
            health_status = self._determine_health_status(health_score)
            
            return ComponentHealth(
                component_id=self.component_id,
                component_type=self.component_type,
                status=ComponentStatus.ONLINE if error_count == 0 else ComponentStatus.ERROR,
                health_status=health_status,
                overall_health_score=health_score,
                metrics=metrics,
                error_count=error_count
            )
            
        except Exception as e:
            self.logger.error(f"Error checking API health: {e}")
            return ComponentHealth(
                component_id=self.component_id,
                component_type=self.component_type,
                status=ComponentStatus.OFFLINE,
                health_status=HealthStatus.FAILED,
                overall_health_score=0.0
            )
    
    def _determine_health_status(self, health_score: float) -> HealthStatus:
        """Determine health status based on score"""
        if health_score >= 95:
            return HealthStatus.HEALTHY
        elif health_score >= 80:
            return HealthStatus.DEGRADED
        elif health_score >= 60:
            return HealthStatus.WARNING
        elif health_score >= 30:
            return HealthStatus.CRITICAL
        else:
            return HealthStatus.FAILED


class HealthMonitor:
    """Central health monitoring system"""
    
    def __init__(self):
        self.logger = get_component_logger(ComponentType.MONITORING, "health_monitor")
        self.health_checkers: Dict[str, HealthChecker] = {}
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.active_alerts: Dict[str, HealthAlert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        
        # Monitoring control
        self.running = False
        self.worker_threads: List[threading.Thread] = []
        self.check_interval = 30.0  # seconds
        
        # Alert management
        self.alert_callbacks: List[Callable[[HealthAlert], None]] = []
        self.alert_suppression: Dict[str, datetime] = {}
        self.suppression_duration = timedelta(minutes=5)
        
        # Predictive health (if ML available)
        self.predictive_models: Dict[str, Any] = {}
        self.prediction_enabled = ML_AVAILABLE
        
        # Health data storage
        self.data_store = HealthDataStore()
        
        # Recovery actions
        self.recovery_actions: Dict[str, Callable] = {}
        
        # Initialize default health checkers
        self._initialize_default_checkers()
    
    def _initialize_default_checkers(self):
        """Initialize default health checkers"""
        # System health checker
        system_checker = SystemHealthChecker()
        self.register_health_checker(system_checker)
        
        # Database health checker (if URL available)
        database_url = os.environ.get('DATABASE_URL')
        if database_url:
            db_checker = DatabaseHealthChecker(database_url)
            self.register_health_checker(db_checker)
        
        # API health checker
        api_base_url = os.environ.get('API_BASE_URL', 'http://localhost:8080')
        api_checker = APIHealthChecker(api_base_url)
        self.register_health_checker(api_checker)
    
    def register_health_checker(self, checker: HealthChecker):
        """Register a health checker"""
        self.health_checkers[checker.component_id] = checker
        self.logger.info(f"Registered health checker: {checker.component_id}")
    
    def register_alert_callback(self, callback: Callable[[HealthAlert], None]):
        """Register callback for health alerts"""
        self.alert_callbacks.append(callback)
    
    def register_recovery_action(self, component_id: str, action: Callable):
        """Register recovery action for a component"""
        self.recovery_actions[component_id] = action
        self.logger.info(f"Registered recovery action for: {component_id}")
    
    def start_monitoring(self):
        """Start health monitoring"""
        if self.running:
            return
        
        self.running = True
        
        # Start monitoring threads
        self.worker_threads = [
            threading.Thread(target=self._monitoring_loop, daemon=True),
            threading.Thread(target=self._alert_management_loop, daemon=True),
            threading.Thread(target=self._predictive_analysis_loop, daemon=True)
        ]
        
        for thread in self.worker_threads:
            thread.start()
        
        self.logger.info("Started health monitoring")
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.running = False
        
        # Wait for threads to finish
        for thread in self.worker_threads:
            thread.join(timeout=5.0)
        
        self.logger.info("Stopped health monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Check health of all registered components
                for component_id, checker in self.health_checkers.items():
                    if not checker.enabled:
                        continue
                    
                    try:
                        health = checker.check_health()
                        self._process_health_update(health)
                    except Exception as e:
                        self.logger.error(f"Error checking health for {component_id}: {e}")
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def _process_health_update(self, health: ComponentHealth):
        """Process health update from a component"""
        component_id = health.component_id
        
        # Store health data
        self.health_history[component_id].append(health)
        self.data_store.store_health_data(health)
        
        # Check for alerts
        self._check_health_alerts(health)
        
        # Log health status changes
        previous_health = self._get_previous_health(component_id)
        if previous_health and previous_health.health_status != health.health_status:
            self.logger.info(
                f"Health status changed for {component_id}: "
                f"{previous_health.health_status.value} -> {health.health_status.value}"
            )
        
        # Trigger recovery actions if needed
        if health.health_status in [HealthStatus.CRITICAL, HealthStatus.FAILED]:
            self._trigger_recovery_action(component_id, health)
    
    def _get_previous_health(self, component_id: str) -> Optional[ComponentHealth]:
        """Get previous health status for a component"""
        history = self.health_history.get(component_id, deque())
        return history[-2] if len(history) >= 2 else None
    
    def _check_health_alerts(self, health: ComponentHealth):
        """Check health data for alert conditions"""
        component_id = health.component_id
        
        # Check each metric for alert conditions
        for metric_name, metric in health.metrics.items():
            alert_triggered = False
            alert_level = AlertLevel.INFO
            
            # Check critical threshold
            if (metric.threshold_critical is not None and 
                ((metric.is_higher_better and metric.value < metric.threshold_critical) or
                 (not metric.is_higher_better and metric.value > metric.threshold_critical))):
                alert_level = AlertLevel.CRITICAL
                alert_triggered = True
            
            # Check warning threshold
            elif (metric.threshold_warning is not None and
                  ((metric.is_higher_better and metric.value < metric.threshold_warning) or
                   (not metric.is_higher_better and metric.value > metric.threshold_warning))):
                alert_level = AlertLevel.WARNING
                alert_triggered = True
            
            # Component health status alerts
            if health.health_status in [HealthStatus.CRITICAL, HealthStatus.FAILED]:
                alert_level = AlertLevel.CRITICAL
                alert_triggered = True
            elif health.health_status == HealthStatus.WARNING:
                alert_level = AlertLevel.WARNING
                alert_triggered = True
            
            if alert_triggered:
                alert = HealthAlert(
                    level=alert_level,
                    component_id=component_id,
                    component_type=health.component_type,
                    metric_name=metric_name,
                    current_value=metric.value,
                    threshold=metric.threshold_critical or metric.threshold_warning or 0,
                    message=f"{metric_name} threshold exceeded for {component_id}",
                    description=f"{metric_name} is {metric.value:.2f} {metric.unit}",
                    recommended_actions=self._get_recommended_actions(component_id, metric_name, health)
                )
                
                self._trigger_alert(alert)
    
    def _get_recommended_actions(self, component_id: str, metric_name: str, 
                                health: ComponentHealth) -> List[str]:
        """Get recommended recovery actions"""
        actions = []
        
        if component_id == "system":
            if metric_name == "cpu_usage":
                actions = [
                    "Check for runaway processes",
                    "Consider scaling up compute resources",
                    "Review recent code changes for performance issues"
                ]
            elif metric_name == "memory_usage":
                actions = [
                    "Check for memory leaks",
                    "Restart memory-intensive processes",
                    "Scale up memory resources"
                ]
            elif metric_name == "disk_usage":
                actions = [
                    "Clean up temporary files",
                    "Archive old log files",
                    "Add more storage capacity"
                ]
        
        elif component_id == "database":
            if metric_name == "connection_time":
                actions = [
                    "Check database server load",
                    "Review network connectivity",
                    "Optimize database configuration"
                ]
            elif metric_name == "long_running_queries":
                actions = [
                    "Identify and optimize slow queries",
                    "Check for blocking locks",
                    "Consider query cancellation"
                ]
        
        elif component_id == "api":
            if metric_name == "availability":
                actions = [
                    "Check API server status",
                    "Review load balancer configuration",
                    "Verify network connectivity"
                ]
            elif metric_name == "average_response_time":
                actions = [
                    "Check server load",
                    "Review recent deployments",
                    "Optimize API endpoints"
                ]
        
        return actions
    
    def _trigger_alert(self, alert: HealthAlert):
        """Trigger a health alert"""
        # Check for alert suppression
        alert_key = f"{alert.component_id}:{alert.metric_name}:{alert.level.value}"
        
        if alert_key in self.alert_suppression:
            suppressed_until = self.alert_suppression[alert_key]
            if datetime.now() < suppressed_until:
                return  # Alert is suppressed
        
        # Store alert
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        
        # Set suppression
        self.alert_suppression[alert_key] = datetime.now() + self.suppression_duration
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
        
        # Log alert
        log_method = getattr(self.logger, alert.level.value.lower(), self.logger.info)
        log_method(
            f"Health alert: {alert.message}",
            component_id=alert.component_id,
            metric_name=alert.metric_name,
            current_value=alert.current_value,
            threshold=alert.threshold
        )
    
    def _trigger_recovery_action(self, component_id: str, health: ComponentHealth):
        """Trigger recovery action for a component"""
        if component_id in self.recovery_actions:
            try:
                self.logger.info(f"Triggering recovery action for {component_id}")
                action = self.recovery_actions[component_id]
                action(health)
            except Exception as e:
                self.logger.error(f"Error executing recovery action for {component_id}: {e}")
    
    def _alert_management_loop(self):
        """Alert management and resolution loop"""
        while self.running:
            try:
                time.sleep(60.0)  # Check every minute
                
                # Check for resolved alerts
                current_time = datetime.now()
                resolved_alerts = []
                
                for alert_id, alert in self.active_alerts.items():
                    # Auto-resolve old alerts
                    if (current_time - alert.timestamp).total_seconds() > 3600:  # 1 hour
                        alert.resolved = True
                        alert.resolved_time = current_time
                        alert.resolution_notes = "Auto-resolved due to age"
                        resolved_alerts.append(alert_id)
                
                # Remove resolved alerts
                for alert_id in resolved_alerts:
                    del self.active_alerts[alert_id]
                
                # Clean up old suppressions
                expired_suppressions = [
                    key for key, expiry in self.alert_suppression.items()
                    if current_time > expiry
                ]
                for key in expired_suppressions:
                    del self.alert_suppression[key]
                
            except Exception as e:
                self.logger.error(f"Error in alert management loop: {e}")
    
    def _predictive_analysis_loop(self):
        """Predictive health analysis loop"""
        if not self.prediction_enabled:
            return
        
        while self.running:
            try:
                time.sleep(300.0)  # Run every 5 minutes
                
                for component_id in self.health_checkers.keys():
                    self._run_predictive_analysis(component_id)
                
            except Exception as e:
                self.logger.error(f"Error in predictive analysis loop: {e}")
    
    def _run_predictive_analysis(self, component_id: str):
        """Run predictive analysis for a component"""
        if not ML_AVAILABLE:
            return
        
        history = list(self.health_history[component_id])
        if len(history) < 50:  # Need sufficient data
            return
        
        try:
            # Prepare data for analysis
            features = []
            health_scores = []
            
            for health in history:
                # Create feature vector from metrics
                feature_vector = []
                for metric in health.metrics.values():
                    feature_vector.append(metric.value)
                
                if feature_vector:  # Only if we have metrics
                    features.append(feature_vector)
                    health_scores.append(health.overall_health_score)
            
            if len(features) < 20:
                return
            
            features = np.array(features)
            health_scores = np.array(health_scores)
            
            # Train or update predictive model
            if component_id not in self.predictive_models:
                # Create new model
                self.predictive_models[component_id] = {
                    'scaler': StandardScaler(),
                    'model': RandomForestClassifier(n_estimators=100, random_state=42)
                }
            
            model_info = self.predictive_models[component_id]
            scaler = model_info['scaler']
            model = model_info['model']
            
            # Prepare labels (healthy vs unhealthy)
            labels = (health_scores >= 70).astype(int)  # Binary classification
            
            if len(np.unique(labels)) > 1:  # Need both classes
                # Scale features
                features_scaled = scaler.fit_transform(features)
                
                # Train model
                model.fit(features_scaled, labels)
                
                # Predict future health
                recent_features = features_scaled[-10:]  # Last 10 observations
                predictions = model.predict_proba(recent_features)[:, 0]  # Probability of unhealthy
                
                # Update component health with predictions
                if len(self.health_history[component_id]) > 0:
                    latest_health = self.health_history[component_id][-1]
                    
                    # Average probability of failure
                    failure_probability = np.mean(predictions)
                    
                    if failure_probability > 0.7:  # High probability of failure
                        # Estimate time to failure based on trend
                        trend = np.polyfit(range(len(health_scores[-10:])), health_scores[-10:], 1)[0]
                        
                        if trend < 0:  # Declining health
                            estimated_hours = max(1, abs(health_scores[-1] / trend))
                            predicted_failure_time = datetime.now() + timedelta(hours=estimated_hours)
                            
                            latest_health.predicted_failure_time = predicted_failure_time
                            latest_health.confidence_score = failure_probability
                            latest_health.degradation_trend = trend
                            
                            # Create predictive alert
                            predictive_alert = HealthAlert(
                                level=AlertLevel.WARNING,
                                component_id=component_id,
                                component_type=latest_health.component_type,
                                metric_name="predicted_failure",
                                current_value=failure_probability,
                                threshold=0.7,
                                message=f"Predictive model indicates potential failure for {component_id}",
                                description=f"Predicted failure in approximately {estimated_hours:.1f} hours",
                                predicted_impact="Component may become unavailable",
                                estimated_recovery_time=timedelta(hours=1),
                                recommended_actions=["Schedule preventive maintenance", "Monitor closely", "Prepare backup systems"]
                            )
                            
                            self._trigger_alert(predictive_alert)
        
        except Exception as e:
            self.logger.error(f"Error in predictive analysis for {component_id}: {e}")
    
    def get_health_status(self, component_id: Optional[str] = None) -> Union[ComponentHealth, Dict[str, ComponentHealth]]:
        """Get current health status"""
        if component_id:
            history = self.health_history.get(component_id, deque())
            return history[-1] if history else None
        else:
            return {
                comp_id: history[-1] if history else None
                for comp_id, history in self.health_history.items()
            }
    
    def get_active_alerts(self) -> List[HealthAlert]:
        """Get list of active alerts"""
        return list(self.active_alerts.values())
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_time = datetime.now()
            self.logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
    
    def resolve_alert(self, alert_id: str, resolution_notes: str = ""):
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_time = datetime.now()
            alert.resolution_notes = resolution_notes
            del self.active_alerts[alert_id]
            self.logger.info(f"Alert {alert_id} resolved: {resolution_notes}")


class HealthDataStore:
    """Persistent storage for health data"""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("health_data.db")
        self.lock = threading.Lock()
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database for health data"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS health_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component_id TEXT,
                    component_type TEXT,
                    timestamp TEXT,
                    health_status TEXT,
                    health_score REAL,
                    metrics TEXT,
                    metadata TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE,
                    component_id TEXT,
                    timestamp TEXT,
                    level TEXT,
                    message TEXT,
                    resolved INTEGER,
                    resolution_time TEXT
                )
            ''')
            
            conn.execute('CREATE INDEX IF NOT EXISTS idx_component_time ON health_data(component_id, timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_alert_component ON alerts(component_id, timestamp)')
    
    def store_health_data(self, health: ComponentHealth):
        """Store health data"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT INTO health_data 
                        (component_id, component_type, timestamp, health_status, health_score, metrics, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        health.component_id,
                        health.component_type.value,
                        health.last_updated.isoformat(),
                        health.health_status.value,
                        health.overall_health_score,
                        json.dumps({name: asdict(metric) for name, metric in health.metrics.items()}, default=str),
                        json.dumps(asdict(health), default=str)
                    ))
            except Exception as e:
                # Don't let storage errors break monitoring
                pass
    
    def store_alert(self, alert: HealthAlert):
        """Store alert data"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO alerts
                        (alert_id, component_id, timestamp, level, message, resolved, resolution_time)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        alert.alert_id,
                        alert.component_id,
                        alert.timestamp.isoformat(),
                        alert.level.value,
                        alert.message,
                        int(alert.resolved),
                        alert.resolved_time.isoformat() if alert.resolved_time else None
                    ))
            except Exception as e:
                # Don't let storage errors break monitoring
                pass


# Global health monitor instance
health_monitor = HealthMonitor()

# Convenience functions
def start_health_monitoring():
    """Start global health monitoring"""
    health_monitor.start_monitoring()

def stop_health_monitoring():
    """Stop global health monitoring"""
    health_monitor.stop_monitoring()

def get_health_status(component_id: Optional[str] = None):
    """Get current health status"""
    return health_monitor.get_health_status(component_id)

def get_active_alerts():
    """Get active health alerts"""
    return health_monitor.get_active_alerts()


# Example usage
if __name__ == "__main__":
    # Start health monitoring
    start_health_monitoring()
    
    # Register a custom alert callback
    def alert_callback(alert: HealthAlert):
        print(f"ALERT: {alert.level.value.upper()} - {alert.message}")
        if alert.recommended_actions:
            print("Recommended actions:")
            for action in alert.recommended_actions:
                print(f"  - {action}")
    
    health_monitor.register_alert_callback(alert_callback)
    
    # Monitor for a while
    try:
        time.sleep(60)  # Monitor for 1 minute
        
        # Get health status
        health_status = get_health_status()
        print("\nCurrent Health Status:")
        for component_id, health in health_status.items():
            if health:
                print(f"{component_id}: {health.health_status.value} ({health.overall_health_score:.1f}%)")
        
        # Get active alerts
        alerts = get_active_alerts()
        print(f"\nActive Alerts: {len(alerts)}")
        for alert in alerts:
            print(f"  {alert.level.value}: {alert.message}")
    
    finally:
        stop_health_monitoring()