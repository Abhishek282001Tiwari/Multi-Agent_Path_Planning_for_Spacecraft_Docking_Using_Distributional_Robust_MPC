"""
Real-time Dashboard System for Spacecraft Simulation

This package provides a comprehensive real-time dashboard interface for mission control,
including live data visualization, system health monitoring, and interactive controls.
"""

from .mission_control import MissionControlDashboard
from .data_visualizer import RealTimeVisualizer, PlotManager
from .health_display import HealthDashboard
from .interactive_3d import Interactive3DView
from .report_generator import ReportGenerator

__all__ = [
    'MissionControlDashboard',
    'RealTimeVisualizer', 
    'PlotManager',
    'HealthDashboard',
    'Interactive3DView',
    'ReportGenerator'
]