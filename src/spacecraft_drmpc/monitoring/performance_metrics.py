"""
Real-time Performance Monitoring for Spacecraft Simulation

This module provides comprehensive performance monitoring capabilities including
system resources, algorithm performance, mission metrics, and real-time alerts.
"""

import time
import threading
import psutil
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
from contextlib import contextmanager
from enum import Enum
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import queue
import weakref
import gc
import tracemalloc
import os
import sys
from pathlib import Path

# Third-party monitoring integrations
try:
    import prometheus_client as prom
    from prometheus_client import Counter, Histogram, Gauge, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import GPUtil
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    GPU_MONITORING_AVAILABLE = False

from .system_logger import get_component_logger, ComponentType, PerformanceMetrics


class MetricType(Enum):
    """Types of performance metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class SystemMetrics:
    """System-level performance metrics"""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_available_gb: float = 0.0
    disk_usage_percent: float = 0.0
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0
    open_files: int = 0
    thread_count: int = 0
    process_count: int = 0
    load_average: Tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    
    # GPU metrics (if available)
    gpu_utilization: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_temperature: Optional[float] = None


@dataclass
class AlgorithmMetrics:
    """Algorithm-specific performance metrics"""
    timestamp: datetime = field(default_factory=datetime.now)
    algorithm_name: str = ""
    execution_time_ms: float = 0.0
    memory_allocated_mb: float = 0.0
    iterations: int = 0
    convergence_status: str = ""
    convergence_tolerance: float = 0.0
    objective_value: float = 0.0
    constraint_violations: int = 0
    solver_status: str = ""
    matrix_condition_number: Optional[float] = None
    sparsity_ratio: Optional[float] = None


@dataclass
class MissionMetrics:
    """Mission and simulation-specific metrics"""
    timestamp: datetime = field(default_factory=datetime.now)
    mission_id: str = ""
    simulation_id: str = ""
    simulation_time: float = 0.0
    real_time_factor: float = 1.0
    active_agents: int = 0
    completed_maneuvers: int = 0
    failed_maneuvers: int = 0
    fuel_consumption_total: float = 0.0
    average_position_error: float = 0.0
    maximum_position_error: float = 0.0
    collision_warnings: int = 0
    emergency_stops: int = 0
    mission_phase: str = ""
    completion_percentage: float = 0.0


@dataclass
class PerformanceAlert:
    """Performance alert data structure"""
    timestamp: datetime = field(default_factory=datetime.now)
    alert_id: str = ""
    level: AlertLevel = AlertLevel.INFO
    component: str = ""
    metric_name: str = ""
    current_value: float = 0.0
    threshold: float = 0.0
    message: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class MetricCollector:
    """Base class for metric collectors"""
    
    def __init__(self, name: str, collection_interval: float = 1.0):
        self.name = name
        self.collection_interval = collection_interval
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.data_queue = queue.Queue()
        self.subscribers: List[Callable] = []
        self.logger = get_component_logger(ComponentType.MONITORING, f"collector.{name}")
    
    def subscribe(self, callback: Callable[[Any], None]):
        """Subscribe to metric updates"""
        self.subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable[[Any], None]):
        """Unsubscribe from metric updates"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    def _notify_subscribers(self, metrics: Any):
        """Notify all subscribers of new metrics"""
        for callback in self.subscribers:
            try:
                callback(metrics)
            except Exception as e:
                self.logger.error(f"Error notifying subscriber: {e}")
    
    def start(self):
        """Start metric collection"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.thread.start()
        self.logger.info(f"Started metric collector: {self.name}")
    
    def stop(self):
        """Stop metric collection"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        self.logger.info(f"Stopped metric collector: {self.name}")
    
    def _collection_loop(self):
        """Main collection loop"""
        while self.running:
            try:
                metrics = self.collect_metrics()
                if metrics:
                    self.data_queue.put(metrics)
                    self._notify_subscribers(metrics)
                
                time.sleep(self.collection_interval)
            except Exception as e:
                self.logger.error(f"Error in collection loop: {e}")
                time.sleep(self.collection_interval)
    
    def collect_metrics(self) -> Any:
        """Override this method to implement specific metric collection"""
        raise NotImplementedError
    
    def get_latest_metrics(self) -> Optional[Any]:
        """Get the latest metrics without blocking"""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None


class SystemMetricsCollector(MetricCollector):
    """Collects system-level performance metrics"""
    
    def __init__(self, collection_interval: float = 1.0):
        super().__init__("system", collection_interval)
        self.process = psutil.Process()
        self.network_io_prev = psutil.net_io_counters() if hasattr(psutil, 'net_io_counters') else None
        self.disk_io_prev = psutil.disk_io_counters() if hasattr(psutil, 'disk_io_counters') else None
        self.last_collection_time = time.time()
    
    def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        current_time = time.time()
        time_delta = current_time - self.last_collection_time
        
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk_usage = psutil.disk_usage('/')
        
        # Network I/O
        network_io = psutil.net_io_counters() if hasattr(psutil, 'net_io_counters') else None
        network_sent_mb = 0.0
        network_recv_mb = 0.0
        if network_io and self.network_io_prev and time_delta > 0:
            network_sent_mb = (network_io.bytes_sent - self.network_io_prev.bytes_sent) / (1024 * 1024 * time_delta)
            network_recv_mb = (network_io.bytes_recv - self.network_io_prev.bytes_recv) / (1024 * 1024 * time_delta)
            self.network_io_prev = network_io
        
        # Disk I/O
        disk_io = psutil.disk_io_counters() if hasattr(psutil, 'disk_io_counters') else None
        disk_io_read_mb = 0.0
        disk_io_write_mb = 0.0
        if disk_io and self.disk_io_prev and time_delta > 0:
            disk_io_read_mb = (disk_io.read_bytes - self.disk_io_prev.read_bytes) / (1024 * 1024 * time_delta)
            disk_io_write_mb = (disk_io.write_bytes - self.disk_io_prev.write_bytes) / (1024 * 1024 * time_delta)
            self.disk_io_prev = disk_io
        
        # Process information
        try:
            open_files = len(self.process.open_files())
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            open_files = 0
        
        thread_count = threading.active_count()
        
        # Load average
        load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0.0, 0.0, 0.0)
        
        # GPU metrics (if available)
        gpu_utilization = None
        gpu_memory_percent = None
        gpu_memory_used_mb = None
        gpu_temperature = None
        
        if GPU_MONITORING_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    gpu_utilization = gpu.load * 100
                    gpu_memory_percent = gpu.memoryUtil * 100
                    gpu_memory_used_mb = gpu.memoryUsed
                    gpu_temperature = gpu.temperature
            except Exception:
                pass  # GPU monitoring failed, continue without it
        
        self.last_collection_time = current_time
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_available_gb=memory.available / (1024**3),
            disk_usage_percent=disk_usage.percent,
            disk_io_read_mb=disk_io_read_mb,
            disk_io_write_mb=disk_io_write_mb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb,
            open_files=open_files,
            thread_count=thread_count,
            process_count=len(psutil.pids()),
            load_average=load_avg,
            gpu_utilization=gpu_utilization,
            gpu_memory_percent=gpu_memory_percent,
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_temperature=gpu_temperature
        )


class AlgorithmProfiler:
    """Profiles algorithm performance with detailed metrics"""
    
    def __init__(self, algorithm_name: str):
        self.algorithm_name = algorithm_name
        self.logger = get_component_logger(ComponentType.MONITORING, f"profiler.{algorithm_name}")
        self.metrics_history: deque = deque(maxlen=1000)
        self.tracemalloc_enabled = False
        self.prometheus_metrics = {}
        
        # Initialize Prometheus metrics if available
        if PROMETHEUS_AVAILABLE:
            self.prometheus_metrics = {
                'execution_time': Histogram(
                    f'{algorithm_name}_execution_time_seconds',
                    f'Execution time for {algorithm_name} algorithm'
                ),
                'iterations': Histogram(
                    f'{algorithm_name}_iterations',
                    f'Number of iterations for {algorithm_name} algorithm'
                ),
                'memory_usage': Gauge(
                    f'{algorithm_name}_memory_usage_bytes',
                    f'Memory usage for {algorithm_name} algorithm'
                )
            }
    
    @contextmanager
    def profile_execution(self, **context):
        """Context manager for profiling algorithm execution"""
        start_time = time.time()
        start_memory = 0
        
        # Enable memory tracing if available
        if hasattr(tracemalloc, 'start'):
            try:
                tracemalloc.start()
                self.tracemalloc_enabled = True
            except RuntimeError:
                pass  # Already started
        
        if self.tracemalloc_enabled:
            snapshot = tracemalloc.take_snapshot()
            start_memory = sum(stat.size for stat in snapshot.statistics('lineno'))
        
        yield_data = {'start_time': start_time, 'context': context}
        
        try:
            yield yield_data
        finally:
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Calculate memory usage
            memory_allocated = 0
            if self.tracemalloc_enabled:
                current_snapshot = tracemalloc.take_snapshot()
                current_memory = sum(stat.size for stat in current_snapshot.statistics('lineno'))
                memory_allocated = max(0, current_memory - start_memory)
            
            # Create metrics object
            metrics = AlgorithmMetrics(
                algorithm_name=self.algorithm_name,
                execution_time_ms=execution_time * 1000,
                memory_allocated_mb=memory_allocated / (1024 * 1024),
                **context
            )
            
            # Store metrics
            self.metrics_history.append(metrics)
            
            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE and self.prometheus_metrics:
                self.prometheus_metrics['execution_time'].observe(execution_time)
                self.prometheus_metrics['memory_usage'].set(memory_allocated)
                if 'iterations' in context:
                    self.prometheus_metrics['iterations'].observe(context['iterations'])
            
            # Log performance
            self.logger.info(
                f"Algorithm {self.algorithm_name} completed",
                execution_time_ms=execution_time * 1000,
                memory_allocated_mb=memory_allocated / (1024 * 1024),
                **context
            )
    
    def get_statistics(self, window_size: int = 100) -> Dict[str, Any]:
        """Get performance statistics over a window of recent executions"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-window_size:]
        execution_times = [m.execution_time_ms for m in recent_metrics]
        memory_usage = [m.memory_allocated_mb for m in recent_metrics]
        
        stats = {
            'algorithm_name': self.algorithm_name,
            'sample_count': len(recent_metrics),
            'execution_time': {
                'mean': np.mean(execution_times),
                'std': np.std(execution_times),
                'min': np.min(execution_times),
                'max': np.max(execution_times),
                'p50': np.percentile(execution_times, 50),
                'p95': np.percentile(execution_times, 95),
                'p99': np.percentile(execution_times, 99)
            },
            'memory_usage': {
                'mean': np.mean(memory_usage),
                'std': np.std(memory_usage),
                'min': np.min(memory_usage),
                'max': np.max(memory_usage)
            }
        }
        
        return stats


class PerformanceMonitor:
    """Central performance monitoring system"""
    
    def __init__(self):
        self.collectors: Dict[str, MetricCollector] = {}
        self.profilers: Dict[str, AlgorithmProfiler] = {}
        self.alert_manager = AlertManager()
        self.metrics_store = MetricsStore()
        self.logger = get_component_logger(ComponentType.MONITORING, "performance_monitor")
        self.running = False
        
        # Performance thresholds for alerts
        self.thresholds = {
            'cpu_percent': 90.0,
            'memory_percent': 85.0,
            'disk_usage_percent': 90.0,
            'execution_time_ms': 1000.0,
            'gpu_utilization': 95.0,
            'gpu_temperature': 80.0
        }
        
        # Initialize Prometheus metrics server if available
        if PROMETHEUS_AVAILABLE:
            self._setup_prometheus_metrics()
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics server"""
        try:
            # Start Prometheus metrics server
            prom.start_http_server(8000)
            self.logger.info("Started Prometheus metrics server on port 8000")
        except Exception as e:
            self.logger.warning(f"Failed to start Prometheus server: {e}")
    
    def add_collector(self, collector: MetricCollector):
        """Add a metric collector"""
        self.collectors[collector.name] = collector
        collector.subscribe(self._handle_metrics)
        self.logger.info(f"Added metric collector: {collector.name}")
    
    def add_profiler(self, profiler: AlgorithmProfiler) -> AlgorithmProfiler:
        """Add an algorithm profiler"""
        self.profilers[profiler.algorithm_name] = profiler
        self.logger.info(f"Added algorithm profiler: {profiler.algorithm_name}")
        return profiler
    
    def get_profiler(self, algorithm_name: str) -> AlgorithmProfiler:
        """Get or create an algorithm profiler"""
        if algorithm_name not in self.profilers:
            self.profilers[algorithm_name] = AlgorithmProfiler(algorithm_name)
        return self.profilers[algorithm_name]
    
    def _handle_metrics(self, metrics: Any):
        """Handle incoming metrics from collectors"""
        # Store metrics
        self.metrics_store.store(metrics)
        
        # Check for alerts
        if isinstance(metrics, SystemMetrics):
            self._check_system_alerts(metrics)
        elif isinstance(metrics, AlgorithmMetrics):
            self._check_algorithm_alerts(metrics)
        elif isinstance(metrics, MissionMetrics):
            self._check_mission_alerts(metrics)
    
    def _check_system_alerts(self, metrics: SystemMetrics):
        """Check system metrics for alert conditions"""
        alerts = []
        
        if metrics.cpu_percent > self.thresholds['cpu_percent']:
            alerts.append(PerformanceAlert(
                level=AlertLevel.WARNING,
                component="system",
                metric_name="cpu_percent",
                current_value=metrics.cpu_percent,
                threshold=self.thresholds['cpu_percent'],
                message=f"High CPU usage: {metrics.cpu_percent:.1f}%"
            ))
        
        if metrics.memory_percent > self.thresholds['memory_percent']:
            alerts.append(PerformanceAlert(
                level=AlertLevel.CRITICAL,
                component="system",
                metric_name="memory_percent",
                current_value=metrics.memory_percent,
                threshold=self.thresholds['memory_percent'],
                message=f"High memory usage: {metrics.memory_percent:.1f}%"
            ))
        
        if metrics.disk_usage_percent > self.thresholds['disk_usage_percent']:
            alerts.append(PerformanceAlert(
                level=AlertLevel.WARNING,
                component="system",
                metric_name="disk_usage_percent",
                current_value=metrics.disk_usage_percent,
                threshold=self.thresholds['disk_usage_percent'],
                message=f"High disk usage: {metrics.disk_usage_percent:.1f}%"
            ))
        
        if (metrics.gpu_utilization and 
            metrics.gpu_utilization > self.thresholds['gpu_utilization']):
            alerts.append(PerformanceAlert(
                level=AlertLevel.WARNING,
                component="gpu",
                metric_name="gpu_utilization",
                current_value=metrics.gpu_utilization,
                threshold=self.thresholds['gpu_utilization'],
                message=f"High GPU utilization: {metrics.gpu_utilization:.1f}%"
            ))
        
        if (metrics.gpu_temperature and 
            metrics.gpu_temperature > self.thresholds['gpu_temperature']):
            alerts.append(PerformanceAlert(
                level=AlertLevel.CRITICAL,
                component="gpu",
                metric_name="gpu_temperature",
                current_value=metrics.gpu_temperature,
                threshold=self.thresholds['gpu_temperature'],
                message=f"High GPU temperature: {metrics.gpu_temperature:.1f}°C"
            ))
        
        for alert in alerts:
            self.alert_manager.trigger_alert(alert)
    
    def _check_algorithm_alerts(self, metrics: AlgorithmMetrics):
        """Check algorithm metrics for alert conditions"""
        if metrics.execution_time_ms > self.thresholds['execution_time_ms']:
            alert = PerformanceAlert(
                level=AlertLevel.WARNING,
                component="algorithm",
                metric_name="execution_time_ms",
                current_value=metrics.execution_time_ms,
                threshold=self.thresholds['execution_time_ms'],
                message=f"Slow algorithm execution: {metrics.algorithm_name} took {metrics.execution_time_ms:.1f}ms",
                context={'algorithm_name': metrics.algorithm_name}
            )
            self.alert_manager.trigger_alert(alert)
    
    def _check_mission_alerts(self, metrics: MissionMetrics):
        """Check mission metrics for alert conditions"""
        if metrics.collision_warnings > 0:
            alert = PerformanceAlert(
                level=AlertLevel.CRITICAL,
                component="mission",
                metric_name="collision_warnings",
                current_value=metrics.collision_warnings,
                threshold=0,
                message=f"Collision warnings detected: {metrics.collision_warnings}",
                context={'mission_id': metrics.mission_id, 'simulation_id': metrics.simulation_id}
            )
            self.alert_manager.trigger_alert(alert)
        
        if metrics.emergency_stops > 0:
            alert = PerformanceAlert(
                level=AlertLevel.EMERGENCY,
                component="mission",
                metric_name="emergency_stops",
                current_value=metrics.emergency_stops,
                threshold=0,
                message=f"Emergency stops triggered: {metrics.emergency_stops}",
                context={'mission_id': metrics.mission_id, 'simulation_id': metrics.simulation_id}
            )
            self.alert_manager.trigger_alert(alert)
    
    def start(self):
        """Start performance monitoring"""
        if self.running:
            return
        
        self.running = True
        
        # Add default system collector
        system_collector = SystemMetricsCollector()
        self.add_collector(system_collector)
        
        # Start all collectors
        for collector in self.collectors.values():
            collector.start()
        
        # Start alert manager
        self.alert_manager.start()
        
        self.logger.info("Performance monitoring started")
    
    def stop(self):
        """Stop performance monitoring"""
        self.running = False
        
        # Stop all collectors
        for collector in self.collectors.values():
            collector.stop()
        
        # Stop alert manager
        self.alert_manager.stop()
        
        self.logger.info("Performance monitoring stopped")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics from all collectors"""
        current_metrics = {}
        for name, collector in self.collectors.items():
            metrics = collector.get_latest_metrics()
            if metrics:
                current_metrics[name] = metrics
        return current_metrics
    
    def get_profiler_statistics(self, algorithm_name: Optional[str] = None) -> Dict[str, Any]:
        """Get profiler statistics"""
        if algorithm_name:
            if algorithm_name in self.profilers:
                return self.profilers[algorithm_name].get_statistics()
            return {}
        else:
            return {name: profiler.get_statistics() 
                   for name, profiler in self.profilers.items()}


class AlertManager:
    """Manages performance alerts and notifications"""
    
    def __init__(self):
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.subscribers: List[Callable] = []
        self.running = False
        self.logger = get_component_logger(ComponentType.MONITORING, "alert_manager")
    
    def subscribe(self, callback: Callable[[PerformanceAlert], None]):
        """Subscribe to alerts"""
        self.subscribers.append(callback)
    
    def trigger_alert(self, alert: PerformanceAlert):
        """Trigger a performance alert"""
        alert_key = f"{alert.component}:{alert.metric_name}"
        alert.alert_id = alert_key
        
        # Check if this is a duplicate alert
        if alert_key in self.active_alerts:
            existing_alert = self.active_alerts[alert_key]
            if existing_alert.level == alert.level:
                return  # Don't spam duplicate alerts
        
        # Store alert
        self.active_alerts[alert_key] = alert
        self.alert_history.append(alert)
        
        # Notify subscribers
        for callback in self.subscribers:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error notifying alert subscriber: {e}")
        
        # Log alert
        log_method = getattr(self.logger, alert.level.value.lower(), self.logger.info)
        log_method(f"Performance alert: {alert.message}", **alert.context)
    
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolution_time = datetime.now()
            del self.active_alerts[alert_id]
            self.logger.info(f"Alert resolved: {alert.message}")
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get list of active alerts"""
        return list(self.active_alerts.values())
    
    def start(self):
        """Start alert manager"""
        self.running = True
        self.logger.info("Alert manager started")
    
    def stop(self):
        """Stop alert manager"""
        self.running = False
        self.logger.info("Alert manager stopped")


class MetricsStore:
    """Stores and retrieves performance metrics"""
    
    def __init__(self, max_history: int = 100000):
        self.max_history = max_history
        self.metrics_data = defaultdict(deque)
        self.lock = threading.Lock()
        self.logger = get_component_logger(ComponentType.MONITORING, "metrics_store")
    
    def store(self, metrics: Any):
        """Store metrics data"""
        with self.lock:
            metric_type = type(metrics).__name__
            self.metrics_data[metric_type].append(metrics)
            
            # Maintain maximum history size
            if len(self.metrics_data[metric_type]) > self.max_history:
                self.metrics_data[metric_type].popleft()
    
    def get_metrics(self, metric_type: str, limit: int = 1000) -> List[Any]:
        """Get stored metrics of a specific type"""
        with self.lock:
            if metric_type in self.metrics_data:
                return list(self.metrics_data[metric_type])[-limit:]
            return []
    
    def get_time_series(self, metric_type: str, start_time: datetime, 
                       end_time: datetime) -> List[Any]:
        """Get metrics within a time range"""
        with self.lock:
            if metric_type not in self.metrics_data:
                return []
            
            return [m for m in self.metrics_data[metric_type]
                   if start_time <= m.timestamp <= end_time]


# Global performance monitor instance
performance_monitor = PerformanceMonitor()

# Convenience functions
def start_monitoring():
    """Start global performance monitoring"""
    performance_monitor.start()

def stop_monitoring():
    """Stop global performance monitoring"""
    performance_monitor.stop()

def get_profiler(algorithm_name: str) -> AlgorithmProfiler:
    """Get or create an algorithm profiler"""
    return performance_monitor.get_profiler(algorithm_name)

def profile_algorithm(algorithm_name: str):
    """Decorator for profiling algorithm performance"""
    def decorator(func):
        profiler = get_profiler(algorithm_name)
        
        def wrapper(*args, **kwargs):
            with profiler.profile_execution():
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    # Start monitoring
    start_monitoring()
    
    # Test algorithm profiling
    @profile_algorithm("test_algorithm")
    def test_algorithm():
        time.sleep(0.1)  # Simulate algorithm execution
        return np.random.randn(1000, 1000)
    
    # Run some tests
    for i in range(10):
        result = test_algorithm()
        time.sleep(0.5)
    
    # Get statistics
    stats = performance_monitor.get_profiler_statistics("test_algorithm")
    print("Algorithm Statistics:", json.dumps(stats, indent=2))
    
    # Get current system metrics
    current_metrics = performance_monitor.get_current_metrics()
    print("Current Metrics:", json.dumps(current_metrics, default=str, indent=2))
    
    # Stop monitoring
    time.sleep(5)
    stop_monitoring()