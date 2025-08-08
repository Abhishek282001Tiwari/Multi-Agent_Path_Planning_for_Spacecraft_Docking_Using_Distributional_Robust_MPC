"""
Structured Logging System for Spacecraft Simulation Components

This module provides comprehensive logging capabilities with structured output,
context management, and integration with external logging systems.
"""

import logging
import logging.handlers
import json
import sys
import threading
import traceback
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import os
from queue import Queue, Empty
import atexit

# Third-party imports for enhanced logging
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

try:
    from pythonjsonlogger import jsonlogger
    JSON_LOGGER_AVAILABLE = True
except ImportError:
    JSON_LOGGER_AVAILABLE = False


class LogLevel(Enum):
    """Log level enumeration with spacecraft-specific levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    MISSION_CRITICAL = "MISSION_CRITICAL"  # Highest priority for mission-critical events


class ComponentType(Enum):
    """Spacecraft simulation component types for categorized logging"""
    CONTROLLER = "controller"
    AGENT = "agent"
    COORDINATION = "coordination"
    SIMULATION = "simulation"
    COMMUNICATION = "communication"
    FAULT_TOLERANCE = "fault_tolerance"
    GUIDANCE = "guidance"
    NAVIGATION = "navigation"
    PROPULSION = "propulsion"
    POWER = "power"
    THERMAL = "thermal"
    ATTITUDE = "attitude"
    SENSORS = "sensors"
    DATABASE = "database"
    API = "api"
    MONITORING = "monitoring"
    SYSTEM = "system"


@dataclass
class LogContext:
    """Context information for structured logging"""
    mission_id: Optional[str] = None
    simulation_id: Optional[str] = None
    agent_id: Optional[str] = None
    component: Optional[ComponentType] = None
    phase: Optional[str] = None
    subsystem: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary, filtering None values"""
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                if isinstance(value, Enum):
                    result[key] = value.value
                else:
                    result[key] = value
        return result


@dataclass
class PerformanceMetrics:
    """Performance-related logging data"""
    execution_time: Optional[float] = None
    memory_usage: Optional[int] = None
    cpu_usage: Optional[float] = None
    solver_time: Optional[float] = None
    iterations: Optional[int] = None
    convergence_status: Optional[str] = None
    fuel_consumption: Optional[float] = None
    position_error: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary, filtering None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}


class SpacecraftLogRecord(logging.LogRecord):
    """Extended log record with spacecraft-specific fields"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mission_id = getattr(self, 'mission_id', None)
        self.simulation_id = getattr(self, 'simulation_id', None)
        self.agent_id = getattr(self, 'agent_id', None)
        self.component = getattr(self, 'component', None)
        self.metrics = getattr(self, 'metrics', None)
        self.context = getattr(self, 'context', None)
        self.trace_id = getattr(self, 'trace_id', str(uuid.uuid4()))


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hostname = os.uname().nodename if hasattr(os, 'uname') else 'unknown'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName,
            'process': record.process,
            'hostname': self.hostname,
            'trace_id': getattr(record, 'trace_id', str(uuid.uuid4()))
        }
        
        # Add spacecraft-specific fields
        if hasattr(record, 'mission_id') and record.mission_id:
            log_entry['mission_id'] = record.mission_id
        if hasattr(record, 'simulation_id') and record.simulation_id:
            log_entry['simulation_id'] = record.simulation_id
        if hasattr(record, 'agent_id') and record.agent_id:
            log_entry['agent_id'] = record.agent_id
        if hasattr(record, 'component') and record.component:
            log_entry['component'] = record.component.value if isinstance(record.component, ComponentType) else record.component
        if hasattr(record, 'context') and record.context:
            log_entry['context'] = record.context.to_dict() if isinstance(record.context, LogContext) else record.context
        if hasattr(record, 'metrics') and record.metrics:
            log_entry['metrics'] = record.metrics.to_dict() if isinstance(record.metrics, PerformanceMetrics) else record.metrics
        
        # Add exception information
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add stack trace for critical errors
        if record.levelno >= logging.ERROR and not record.exc_info:
            log_entry['stack_trace'] = traceback.format_stack()
        
        return json.dumps(log_entry, default=str)


class ElasticsearchHandler(logging.Handler):
    """Handler for sending logs to Elasticsearch"""
    
    def __init__(self, elasticsearch_url: str, index_pattern: str = "spacecraft-logs-%Y.%m.%d"):
        super().__init__()
        self.elasticsearch_url = elasticsearch_url
        self.index_pattern = index_pattern
        self.queue = Queue()
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        atexit.register(self.close)
    
    def emit(self, record: logging.LogRecord):
        """Add log record to queue for processing"""
        try:
            log_entry = json.loads(self.format(record))
            self.queue.put(log_entry)
        except Exception:
            self.handleError(record)
    
    def _worker(self):
        """Background worker to send logs to Elasticsearch"""
        try:
            import elasticsearch
            es = elasticsearch.Elasticsearch([self.elasticsearch_url])
            
            while not self.stop_event.is_set():
                try:
                    log_entry = self.queue.get(timeout=1.0)
                    index_name = datetime.now().strftime(self.index_pattern)
                    es.index(index=index_name, body=log_entry)
                    self.queue.task_done()
                except Empty:
                    continue
                except Exception as e:
                    print(f"Error sending log to Elasticsearch: {e}", file=sys.stderr)
        except ImportError:
            print("Elasticsearch library not available", file=sys.stderr)
    
    def close(self):
        """Close the handler and stop the worker thread"""
        self.stop_event.set()
        self.worker_thread.join(timeout=5.0)
        super().close()


class SpacecraftLogger:
    """Main logger class for spacecraft simulation components"""
    
    _instances: Dict[str, 'SpacecraftLogger'] = {}
    _lock = threading.Lock()
    
    def __init__(self, name: str, component: Optional[ComponentType] = None):
        self.name = name
        self.component = component
        self.logger = logging.getLogger(name)
        self.context_stack = threading.local()
        self.performance_stack = threading.local()
        
        # Set custom log record factory
        self.logger.makeRecord = self._make_record
        
        # Initialize context and performance stacks
        if not hasattr(self.context_stack, 'contexts'):
            self.context_stack.contexts = []
        if not hasattr(self.performance_stack, 'metrics'):
            self.performance_stack.metrics = []
    
    @classmethod
    def get_logger(cls, name: str, component: Optional[ComponentType] = None) -> 'SpacecraftLogger':
        """Get or create logger instance (singleton pattern)"""
        with cls._lock:
            if name not in cls._instances:
                cls._instances[name] = cls(name, component)
            return cls._instances[name]
    
    def _make_record(self, name, level, fn, lno, msg, args, exc_info, func=None, extra=None, sinfo=None):
        """Create custom log record with spacecraft-specific fields"""
        record = SpacecraftLogRecord(name, level, fn, lno, msg, args, exc_info, func, sinfo)
        
        # Add current context
        current_context = self._get_current_context()
        if current_context:
            record.context = current_context
            record.mission_id = current_context.mission_id
            record.simulation_id = current_context.simulation_id
            record.agent_id = current_context.agent_id
            record.component = current_context.component or self.component
        else:
            record.component = self.component
        
        # Add current performance metrics
        current_metrics = self._get_current_metrics()
        if current_metrics:
            record.metrics = current_metrics
        
        # Add extra fields
        if extra:
            for key, value in extra.items():
                setattr(record, key, value)
        
        return record
    
    def _get_current_context(self) -> Optional[LogContext]:
        """Get the current context from the stack"""
        if hasattr(self.context_stack, 'contexts') and self.context_stack.contexts:
            return self.context_stack.contexts[-1]
        return None
    
    def _get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the current performance metrics from the stack"""
        if hasattr(self.performance_stack, 'metrics') and self.performance_stack.metrics:
            return self.performance_stack.metrics[-1]
        return None
    
    @contextmanager
    def context(self, **kwargs):
        """Context manager for adding contextual information to logs"""
        context = LogContext(**kwargs)
        
        if not hasattr(self.context_stack, 'contexts'):
            self.context_stack.contexts = []
        
        self.context_stack.contexts.append(context)
        try:
            yield context
        finally:
            self.context_stack.contexts.pop()
    
    @contextmanager
    def metrics(self, **kwargs):
        """Context manager for adding performance metrics to logs"""
        metrics = PerformanceMetrics(**kwargs)
        
        if not hasattr(self.performance_stack, 'metrics'):
            self.performance_stack.metrics = []
        
        self.performance_stack.metrics.append(metrics)
        try:
            yield metrics
        finally:
            self.performance_stack.metrics.pop()
    
    def debug(self, msg: str, **kwargs):
        """Log debug message"""
        self.logger.debug(msg, extra=kwargs)
    
    def info(self, msg: str, **kwargs):
        """Log info message"""
        self.logger.info(msg, extra=kwargs)
    
    def warning(self, msg: str, **kwargs):
        """Log warning message"""
        self.logger.warning(msg, extra=kwargs)
    
    def error(self, msg: str, **kwargs):
        """Log error message"""
        self.logger.error(msg, extra=kwargs)
    
    def critical(self, msg: str, **kwargs):
        """Log critical message"""
        self.logger.critical(msg, extra=kwargs)
    
    def mission_critical(self, msg: str, **kwargs):
        """Log mission-critical message (highest priority)"""
        # Use critical level but with mission_critical flag
        kwargs['mission_critical'] = True
        self.logger.critical(f"[MISSION CRITICAL] {msg}", extra=kwargs)
    
    def exception(self, msg: str, **kwargs):
        """Log exception with traceback"""
        self.logger.exception(msg, extra=kwargs)
    
    def log_performance(self, msg: str, metrics: PerformanceMetrics, level: str = "INFO", **kwargs):
        """Log performance-related message with metrics"""
        kwargs['metrics'] = metrics
        getattr(self.logger, level.lower())(msg, extra=kwargs)
    
    def log_solver_performance(self, solver_name: str, solve_time: float, 
                             iterations: int, convergence_status: str, **kwargs):
        """Log solver performance metrics"""
        metrics = PerformanceMetrics(
            solver_time=solve_time,
            iterations=iterations,
            convergence_status=convergence_status
        )
        self.log_performance(
            f"Solver {solver_name} completed in {solve_time:.3f}s with {iterations} iterations",
            metrics, **kwargs
        )
    
    def log_mission_event(self, event: str, phase: str, **kwargs):
        """Log mission-related events with structured data"""
        with self.context(phase=phase):
            self.info(f"Mission event: {event}", **kwargs)
    
    def log_system_health(self, component: str, status: str, metrics: Dict[str, Any], **kwargs):
        """Log system health information"""
        health_metrics = PerformanceMetrics(**{k: v for k, v in metrics.items() 
                                              if k in PerformanceMetrics.__annotations__})
        with self.context(subsystem=component):
            self.log_performance(f"System health: {component} - {status}", health_metrics, **kwargs)


class LogManager:
    """Central log management and configuration"""
    
    def __init__(self):
        self.configured_loggers = set()
        self.handlers = {}
    
    def setup_logging(self, 
                     log_level: str = "INFO",
                     log_dir: Optional[Path] = None,
                     console_output: bool = True,
                     json_format: bool = True,
                     elasticsearch_url: Optional[str] = None,
                     max_file_size: int = 100 * 1024 * 1024,  # 100MB
                     backup_count: int = 10):
        """Configure global logging settings"""
        
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatters
        if json_format:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(component)s] '
                '[%(mission_id)s/%(simulation_id)s/%(agent_id)s] - %(message)s'
            )
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.addFilter(lambda record: record.levelno < logging.ERROR)
            root_logger.addHandler(console_handler)
            
            # Separate handler for errors to stderr
            error_handler = logging.StreamHandler(sys.stderr)
            error_handler.setFormatter(formatter)
            error_handler.setLevel(logging.ERROR)
            root_logger.addHandler(error_handler)
        
        # File handlers
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Main log file
            file_handler = logging.handlers.RotatingFileHandler(
                log_dir / "spacecraft.log",
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            
            # Error log file
            error_file_handler = logging.handlers.RotatingFileHandler(
                log_dir / "spacecraft_errors.log",
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            error_file_handler.setFormatter(formatter)
            error_file_handler.setLevel(logging.ERROR)
            root_logger.addHandler(error_file_handler)
            
            # Mission-critical log file
            critical_file_handler = logging.handlers.RotatingFileHandler(
                log_dir / "spacecraft_critical.log",
                maxBytes=max_file_size,
                backupCount=backup_count * 2  # Keep more critical logs
            )
            critical_file_handler.setFormatter(formatter)
            critical_file_handler.addFilter(lambda record: hasattr(record, 'mission_critical'))
            root_logger.addHandler(critical_file_handler)
        
        # Elasticsearch handler
        if elasticsearch_url:
            try:
                es_handler = ElasticsearchHandler(elasticsearch_url)
                es_handler.setFormatter(formatter)
                root_logger.addHandler(es_handler)
            except Exception as e:
                print(f"Failed to setup Elasticsearch logging: {e}", file=sys.stderr)
    
    def get_component_logger(self, component: ComponentType, name: Optional[str] = None) -> SpacecraftLogger:
        """Get a logger for a specific component"""
        logger_name = name or f"spacecraft.{component.value}"
        return SpacecraftLogger.get_logger(logger_name, component)
    
    def configure_component_logging(self, component: ComponentType, level: str = "INFO"):
        """Configure logging for a specific component"""
        logger_name = f"spacecraft.{component.value}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, level.upper()))


# Global log manager instance
log_manager = LogManager()

# Convenience functions
def get_logger(name: str, component: Optional[ComponentType] = None) -> SpacecraftLogger:
    """Get a logger instance"""
    return SpacecraftLogger.get_logger(name, component)

def get_component_logger(component: ComponentType, name: Optional[str] = None) -> SpacecraftLogger:
    """Get a logger for a specific component"""
    return log_manager.get_component_logger(component, name)

def setup_logging(**kwargs):
    """Setup global logging configuration"""
    log_manager.setup_logging(**kwargs)

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    setup_logging(
        log_level="DEBUG",
        log_dir=Path("logs"),
        console_output=True,
        json_format=True
    )
    
    # Create component loggers
    controller_logger = get_component_logger(ComponentType.CONTROLLER)
    agent_logger = get_component_logger(ComponentType.AGENT, "agent_001")
    
    # Test structured logging
    with controller_logger.context(mission_id="MISSION_001", simulation_id="SIM_001"):
        controller_logger.info("Mission started")
        
        with controller_logger.metrics(execution_time=0.045, memory_usage=1024*1024):
            controller_logger.info("Controller step completed")
        
        # Test performance logging
        metrics = PerformanceMetrics(
            solver_time=0.023,
            iterations=15,
            convergence_status="converged",
            position_error=0.001
        )
        controller_logger.log_performance("MPC solver completed", metrics)
        
        # Test mission critical logging
        controller_logger.mission_critical("Collision avoidance maneuver initiated")
    
    with agent_logger.context(agent_id="AGENT_001", phase="docking"):
        agent_logger.info("Docking phase initiated")
        agent_logger.warning("Low fuel warning")
        
        try:
            raise ValueError("Test exception")
        except Exception:
            agent_logger.exception("Error during docking maneuver")