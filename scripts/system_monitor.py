#!/usr/bin/env python3
"""
System monitoring and maintenance automation for Spacecraft DR-MPC System
Provides real-time monitoring, alerting, and automated maintenance capabilities
"""

import asyncio
import json
import logging
import os
import smtplib
import sys
import time
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse
import psutil
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemMonitor:
    """Comprehensive system monitoring and maintenance automation."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.project_root = Path(__file__).parent.parent
        self.config_path = config_path or self.project_root / "monitoring" / "config.yaml"
        self.config = self.load_config()
        self.monitoring_data = {}
        self.alerts_history = []
        self.last_maintenance = {}
        
    def load_config(self) -> Dict:
        """Load monitoring configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded monitoring configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self.default_config()
    
    def default_config(self) -> Dict:
        """Default monitoring configuration."""
        return {
            "monitoring": {
                "interval_seconds": 30,
                "metrics_retention_hours": 24,
                "alert_cooldown_minutes": 15
            },
            "thresholds": {
                "cpu_usage": {"warning": 70, "critical": 85},
                "memory_usage": {"warning": 75, "critical": 90},
                "disk_usage": {"warning": 80, "critical": 90},
                "response_time": {"warning": 1000, "critical": 5000},
                "error_rate": {"warning": 0.05, "critical": 0.10}
            },
            "alerts": {
                "email": {
                    "enabled": False,
                    "smtp_server": "smtp.example.com",
                    "smtp_port": 587,
                    "from_email": "noreply@spacecraft-system.com",
                    "to_emails": ["admin@spacecraft-system.com"]
                },
                "slack": {
                    "enabled": False,
                    "webhook_url": ""
                }
            },
            "maintenance": {
                "auto_cleanup": True,
                "log_rotation_days": 7,
                "data_retention_days": 30,
                "auto_restart_on_errors": True
            },
            "health_checks": {
                "endpoints": [
                    {"url": "http://localhost:8080/health", "timeout": 10},
                    {"url": "http://localhost:8080/metrics", "timeout": 5}
                ]
            }
        }
    
    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics."""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "system": {},
            "application": {},
            "network": {},
            "disk": {}
        }
        
        # System metrics
        metrics["system"] = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": dict(psutil.virtual_memory()._asdict()),
            "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0],
            "uptime": time.time() - psutil.boot_time()
        }
        
        # Disk metrics
        disk_usage = {}
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_usage[partition.mountpoint] = {
                    "total": usage.total,
                    "used": usage.used,
                    "free": usage.free,
                    "percent": usage.used / usage.total * 100
                }
            except PermissionError:
                continue
        metrics["disk"] = disk_usage
        
        # Network metrics
        net_io = psutil.net_io_counters()
        metrics["network"] = {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv
        }
        
        # Application-specific metrics
        metrics["application"] = await self.collect_application_metrics()
        
        return metrics
    
    async def collect_application_metrics(self) -> Dict[str, Any]:
        """Collect application-specific metrics."""
        app_metrics = {
            "processes": [],
            "log_errors": 0,
            "response_times": [],
            "active_connections": 0
        }
        
        # Find application processes
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                if 'spacecraft' in proc.info['name'].lower():
                    app_metrics["processes"].append({
                        "pid": proc.info['pid'],
                        "name": proc.info['name'],
                        "cpu_percent": proc.info['cpu_percent'],
                        "memory_percent": proc.info['memory_percent']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        
        # Check log files for errors
        log_dir = self.project_root / "logs"
        if log_dir.exists():
            error_count = 0
            for log_file in log_dir.glob("*.log"):
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()[-100:]  # Last 100 lines
                        error_count += sum(1 for line in lines if 'ERROR' in line.upper())
                except Exception:
                    continue
            app_metrics["log_errors"] = error_count
        
        # Health check response times
        response_times = []
        for endpoint in self.config.get("health_checks", {}).get("endpoints", []):
            try:
                import requests
                start_time = time.time()
                response = requests.get(endpoint["url"], timeout=endpoint.get("timeout", 10))
                response_time = (time.time() - start_time) * 1000  # Convert to ms
                response_times.append({
                    "url": endpoint["url"],
                    "response_time": response_time,
                    "status_code": response.status_code,
                    "healthy": response.status_code == 200
                })
            except Exception as e:
                response_times.append({
                    "url": endpoint["url"],
                    "response_time": -1,
                    "status_code": -1,
                    "healthy": False,
                    "error": str(e)
                })
        
        app_metrics["response_times"] = response_times
        
        return app_metrics
    
    def analyze_metrics(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze metrics against thresholds and generate alerts."""
        alerts = []
        thresholds = self.config.get("thresholds", {})
        
        # CPU usage alert
        cpu_usage = metrics["system"]["cpu_percent"]
        if cpu_usage >= thresholds.get("cpu_usage", {}).get("critical", 85):
            alerts.append({
                "level": "critical",
                "type": "cpu_usage",
                "message": f"Critical CPU usage: {cpu_usage:.1f}%",
                "value": cpu_usage,
                "threshold": thresholds["cpu_usage"]["critical"]
            })
        elif cpu_usage >= thresholds.get("cpu_usage", {}).get("warning", 70):
            alerts.append({
                "level": "warning",
                "type": "cpu_usage",
                "message": f"High CPU usage: {cpu_usage:.1f}%",
                "value": cpu_usage,
                "threshold": thresholds["cpu_usage"]["warning"]
            })
        
        # Memory usage alert
        memory_usage = metrics["system"]["memory"]["percent"]
        if memory_usage >= thresholds.get("memory_usage", {}).get("critical", 90):
            alerts.append({
                "level": "critical",
                "type": "memory_usage",
                "message": f"Critical memory usage: {memory_usage:.1f}%",
                "value": memory_usage,
                "threshold": thresholds["memory_usage"]["critical"]
            })
        elif memory_usage >= thresholds.get("memory_usage", {}).get("warning", 75):
            alerts.append({
                "level": "warning",
                "type": "memory_usage",
                "message": f"High memory usage: {memory_usage:.1f}%",
                "value": memory_usage,
                "threshold": thresholds["memory_usage"]["warning"]
            })
        
        # Disk usage alerts
        for mountpoint, usage in metrics["disk"].items():
            disk_percent = usage["percent"]
            if disk_percent >= thresholds.get("disk_usage", {}).get("critical", 90):
                alerts.append({
                    "level": "critical",
                    "type": "disk_usage",
                    "message": f"Critical disk usage on {mountpoint}: {disk_percent:.1f}%",
                    "value": disk_percent,
                    "threshold": thresholds["disk_usage"]["critical"]
                })
            elif disk_percent >= thresholds.get("disk_usage", {}).get("warning", 80):
                alerts.append({
                    "level": "warning",
                    "type": "disk_usage",
                    "message": f"High disk usage on {mountpoint}: {disk_percent:.1f}%",
                    "value": disk_percent,
                    "threshold": thresholds["disk_usage"]["warning"]
                })
        
        # Application health alerts
        for health_check in metrics["application"]["response_times"]:
            if not health_check["healthy"]:
                alerts.append({
                    "level": "critical",
                    "type": "health_check",
                    "message": f"Health check failed for {health_check['url']}",
                    "value": health_check.get("error", "Unknown error"),
                    "threshold": "healthy"
                })
            elif health_check["response_time"] >= thresholds.get("response_time", {}).get("critical", 5000):
                alerts.append({
                    "level": "critical",
                    "type": "response_time",
                    "message": f"Critical response time for {health_check['url']}: {health_check['response_time']:.0f}ms",
                    "value": health_check["response_time"],
                    "threshold": thresholds["response_time"]["critical"]
                })
        
        # Log error alerts
        error_count = metrics["application"]["log_errors"]
        if error_count > 10:
            alerts.append({
                "level": "warning",
                "type": "log_errors",
                "message": f"High error count in logs: {error_count} errors",
                "value": error_count,
                "threshold": 10
            })
        
        return alerts
    
    async def send_alert(self, alert: Dict[str, Any]):
        """Send alert notification."""
        alert_config = self.config.get("alerts", {})
        
        # Check alert cooldown
        alert_key = f"{alert['type']}_{alert['level']}"
        cooldown_minutes = self.config.get("monitoring", {}).get("alert_cooldown_minutes", 15)
        last_sent = self.last_maintenance.get(f"alert_{alert_key}")
        
        if last_sent:
            time_since = datetime.now() - datetime.fromisoformat(last_sent)
            if time_since < timedelta(minutes=cooldown_minutes):
                logger.debug(f"Alert {alert_key} in cooldown period")
                return
        
        # Log alert
        logger.warning(f"ALERT [{alert['level'].upper()}]: {alert['message']}")
        
        # Email alert
        if alert_config.get("email", {}).get("enabled", False):
            await self.send_email_alert(alert)
        
        # Slack alert
        if alert_config.get("slack", {}).get("enabled", False):
            await self.send_slack_alert(alert)
        
        # Record alert time
        self.last_maintenance[f"alert_{alert_key}"] = datetime.now().isoformat()
        self.alerts_history.append({
            **alert,
            "timestamp": datetime.now().isoformat()
        })
    
    async def send_email_alert(self, alert: Dict[str, Any]):
        """Send email alert."""
        try:
            email_config = self.config["alerts"]["email"]
            
            subject = f"Spacecraft System Alert: {alert['level'].upper()} - {alert['type']}"
            body = f"""
            Alert Level: {alert['level'].upper()}
            Alert Type: {alert['type']}
            Message: {alert['message']}
            Value: {alert['value']}
            Threshold: {alert['threshold']}
            Timestamp: {datetime.now()}
            
            This is an automated alert from the Spacecraft DR-MPC monitoring system.
            """
            
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = email_config['from_email']
            msg['To'] = ', '.join(email_config['to_emails'])
            
            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                server.starttls()
                if email_config.get('username'):
                    server.login(email_config['username'], email_config['password'])
                server.send_message(msg)
            
            logger.info("Email alert sent successfully")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    async def send_slack_alert(self, alert: Dict[str, Any]):
        """Send Slack alert."""
        try:
            import requests
            
            slack_config = self.config["alerts"]["slack"]
            webhook_url = slack_config["webhook_url"]
            
            color = {
                "warning": "#ffcc00",
                "critical": "#ff0000"
            }.get(alert["level"], "#cccccc")
            
            payload = {
                "attachments": [
                    {
                        "color": color,
                        "title": f"Spacecraft System Alert: {alert['level'].upper()}",
                        "fields": [
                            {"title": "Type", "value": alert["type"], "short": True},
                            {"title": "Level", "value": alert["level"], "short": True},
                            {"title": "Message", "value": alert["message"], "short": False},
                            {"title": "Value", "value": str(alert["value"]), "short": True},
                            {"title": "Threshold", "value": str(alert["threshold"]), "short": True}
                        ],
                        "timestamp": int(time.time())
                    }
                ]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info("Slack alert sent successfully")
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    def perform_maintenance(self):
        """Perform automated maintenance tasks."""
        maintenance_config = self.config.get("maintenance", {})
        
        # Log rotation
        if maintenance_config.get("auto_cleanup", True):
            self.rotate_logs()
            self.cleanup_old_data()
        
        # System health checks
        self.check_and_restart_services()
        
        logger.info("Automated maintenance completed")
    
    def rotate_logs(self):
        """Rotate log files."""
        try:
            retention_days = self.config.get("maintenance", {}).get("log_rotation_days", 7)
            log_dir = self.project_root / "logs"
            
            if not log_dir.exists():
                return
            
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            for log_file in log_dir.glob("*.log"):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    log_file.unlink()
                    logger.info(f"Rotated old log file: {log_file}")
            
        except Exception as e:
            logger.error(f"Log rotation failed: {e}")
    
    def cleanup_old_data(self):
        """Cleanup old data files."""
        try:
            retention_days = self.config.get("maintenance", {}).get("data_retention_days", 30)
            data_dirs = [
                self.project_root / "reports",
                self.project_root / "docs" / "_data" / "results"
            ]
            
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            for data_dir in data_dirs:
                if not data_dir.exists():
                    continue
                    
                for data_file in data_dir.glob("*"):
                    if data_file.is_file() and data_file.stat().st_mtime < cutoff_date.timestamp():
                        # Keep essential files
                        if data_file.name in ["test_results.json", "performance_metrics.json"]:
                            continue
                        data_file.unlink()
                        logger.info(f"Cleaned up old data file: {data_file}")
            
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
    
    def check_and_restart_services(self):
        """Check and restart services if needed."""
        if not self.config.get("maintenance", {}).get("auto_restart_on_errors", True):
            return
        
        # Check if any spacecraft processes are stuck or crashed
        spacecraft_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'status']):
            try:
                if 'spacecraft' in proc.info['name'].lower():
                    spacecraft_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Restart processes if needed
        for proc in spacecraft_processes:
            try:
                if proc.status() == psutil.STATUS_ZOMBIE:
                    logger.warning(f"Restarting zombie process {proc.pid}")
                    proc.terminate()
                    time.sleep(2)
                    # Here you would implement actual service restart logic
            except Exception as e:
                logger.error(f"Failed to restart process {proc.pid}: {e}")
    
    def generate_monitoring_report(self) -> str:
        """Generate comprehensive monitoring report."""
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "system_summary": {
                "monitoring_duration": len(self.monitoring_data),
                "total_alerts": len(self.alerts_history),
                "critical_alerts": len([a for a in self.alerts_history if a.get("level") == "critical"]),
                "warning_alerts": len([a for a in self.alerts_history if a.get("level") == "warning"])
            },
            "recent_metrics": list(self.monitoring_data.values())[-10:],  # Last 10 metrics
            "recent_alerts": self.alerts_history[-20:],  # Last 20 alerts
            "system_health": self.assess_system_health()
        }
        
        # Save report
        reports_dir = self.project_root / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        report_file = reports_dir / f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Monitoring report saved: {report_file}")
        return str(report_file)
    
    def assess_system_health(self) -> str:
        """Assess overall system health."""
        if not self.monitoring_data:
            return "unknown"
        
        latest_metrics = list(self.monitoring_data.values())[-1]
        recent_alerts = [a for a in self.alerts_history 
                        if datetime.fromisoformat(a["timestamp"]) > datetime.now() - timedelta(hours=1)]
        
        critical_alerts = [a for a in recent_alerts if a.get("level") == "critical"]
        warning_alerts = [a for a in recent_alerts if a.get("level") == "warning"]
        
        if critical_alerts:
            return "critical"
        elif warning_alerts:
            return "warning"
        elif latest_metrics["system"]["cpu_percent"] > 90 or latest_metrics["system"]["memory"]["percent"] > 95:
            return "degraded"
        else:
            return "healthy"
    
    async def monitoring_loop(self):
        """Main monitoring loop."""
        interval = self.config.get("monitoring", {}).get("interval_seconds", 30)
        
        logger.info(f"Starting monitoring loop with {interval}s interval")
        
        while True:
            try:
                # Collect metrics
                metrics = await self.collect_system_metrics()
                timestamp = metrics["timestamp"]
                self.monitoring_data[timestamp] = metrics
                
                # Analyze for alerts
                alerts = self.analyze_metrics(metrics)
                for alert in alerts:
                    await self.send_alert(alert)
                
                # Periodic maintenance (every hour)
                if len(self.monitoring_data) % (3600 // interval) == 0:
                    self.perform_maintenance()
                
                # Cleanup old monitoring data
                retention_hours = self.config.get("monitoring", {}).get("metrics_retention_hours", 24)
                cutoff_time = datetime.now() - timedelta(hours=retention_hours)
                self.monitoring_data = {
                    k: v for k, v in self.monitoring_data.items()
                    if datetime.fromisoformat(k) > cutoff_time
                }
                
                # Log system status
                system_health = self.assess_system_health()
                logger.info(f"System health: {system_health}, "
                          f"CPU: {metrics['system']['cpu_percent']:.1f}%, "
                          f"Memory: {metrics['system']['memory']['percent']:.1f}%")
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
            
            await asyncio.sleep(interval)

def main():
    parser = argparse.ArgumentParser(description="Spacecraft DR-MPC System Monitor")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--report", action="store_true", help="Generate monitoring report and exit")
    parser.add_argument("--health-check", action="store_true", help="Run health check and exit")
    parser.add_argument("--maintenance", action="store_true", help="Run maintenance tasks and exit")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    monitor = SystemMonitor(args.config)
    
    if args.report:
        report_file = monitor.generate_monitoring_report()
        print(f"Monitoring report generated: {report_file}")
        sys.exit(0)
    
    if args.health_check:
        health = monitor.assess_system_health()
        print(f"System health: {health}")
        sys.exit(0 if health in ["healthy", "warning"] else 1)
    
    if args.maintenance:
        monitor.perform_maintenance()
        print("Maintenance tasks completed")
        sys.exit(0)
    
    # Start continuous monitoring
    try:
        asyncio.run(monitor.monitoring_loop())
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
        report_file = monitor.generate_monitoring_report()
        logger.info(f"Final report saved: {report_file}")

if __name__ == "__main__":
    main()