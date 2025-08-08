"""
Real-time Data Visualization Components

This module provides comprehensive data visualization capabilities for spacecraft
telemetry, including real-time plotting, interactive charts, and customizable displays.
"""

import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import threading
import queue
import json

# Plotting libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Circle, Rectangle
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from spacecraft_drmpc.monitoring.system_logger import get_component_logger, ComponentType
from spacecraft_drmpc.monitoring.telemetry_collector import TelemetryPoint, TelemetryType, SpacecraftState


class PlotType(Enum):
    """Types of plots supported by the visualizer"""
    LINE = "line"
    SCATTER = "scatter"
    BAR = "bar"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    SURFACE = "surface"
    GAUGE = "gauge"
    INDICATOR = "indicator"
    PIE = "pie"
    BOX = "box"
    VIOLIN = "violin"
    TRAJECTORY_3D = "trajectory_3d"
    VECTOR_FIELD = "vector_field"


class PlotStyle(Enum):
    """Plot styling options"""
    MINIMAL = "minimal"
    SCIENTIFIC = "scientific"
    MISSION_CONTROL = "mission_control"
    DARK = "dark"
    COLORFUL = "colorful"


@dataclass
class PlotConfig:
    """Configuration for individual plots"""
    plot_id: str = ""
    plot_type: PlotType = PlotType.LINE
    title: str = ""
    x_label: str = "Time"
    y_label: str = "Value"
    z_label: str = "Z"
    color_scheme: str = "viridis"
    style: PlotStyle = PlotStyle.MISSION_CONTROL
    
    # Data configuration
    max_points: int = 1000
    update_interval: float = 1.0  # seconds
    auto_scale: bool = True
    show_legend: bool = True
    show_grid: bool = True
    
    # Axis configuration
    x_range: Optional[Tuple[float, float]] = None
    y_range: Optional[Tuple[float, float]] = None
    z_range: Optional[Tuple[float, float]] = None
    
    # Styling
    width: int = 800
    height: int = 400
    background_color: str = "white"
    font_size: int = 12
    
    # Animation
    animated: bool = False
    animation_speed: float = 1.0
    trail_length: int = 50
    
    # Interactive features
    zoom_enabled: bool = True
    pan_enabled: bool = True
    crossfilter_enabled: bool = False


@dataclass
class PlotData:
    """Container for plot data"""
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    z: Optional[List[float]] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    colors: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class PlotManager:
    """Manages individual plot instances and their data"""
    
    def __init__(self, plot_config: PlotConfig):
        self.config = plot_config
        self.logger = get_component_logger(ComponentType.MONITORING, f"plot.{plot_config.plot_id}")
        
        # Data storage
        self.data_buffer: deque = deque(maxlen=plot_config.max_points)
        self.current_data = PlotData()
        
        # Update tracking
        self.last_update = datetime.now()
        self.update_queue = queue.Queue()
        self.subscribers: List[Callable] = []
        
        # Plot state
        self.figure = None
        self.needs_update = True
        
        # Initialize plot based on backend
        if PLOTLY_AVAILABLE:
            self._init_plotly()
        elif MATPLOTLIB_AVAILABLE:
            self._init_matplotlib()
        else:
            self.logger.error("No plotting backend available")
    
    def _init_plotly(self):
        """Initialize Plotly figure"""
        if self.config.plot_type == PlotType.LINE:
            self.figure = go.Figure()
            self.figure.add_trace(go.Scatter(x=[], y=[], mode='lines+markers', name=self.config.title))
        
        elif self.config.plot_type == PlotType.SCATTER:
            self.figure = go.Figure()
            self.figure.add_trace(go.Scatter(x=[], y=[], mode='markers', name=self.config.title))
        
        elif self.config.plot_type == PlotType.BAR:
            self.figure = go.Figure()
            self.figure.add_trace(go.Bar(x=[], y=[], name=self.config.title))
        
        elif self.config.plot_type == PlotType.HISTOGRAM:
            self.figure = go.Figure()
            self.figure.add_trace(go.Histogram(x=[], name=self.config.title))
        
        elif self.config.plot_type == PlotType.HEATMAP:
            self.figure = go.Figure()
            self.figure.add_trace(go.Heatmap(z=[[]], colorscale=self.config.color_scheme))
        
        elif self.config.plot_type == PlotType.SURFACE:
            self.figure = go.Figure()
            self.figure.add_trace(go.Surface(z=[[]], colorscale=self.config.color_scheme))
        
        elif self.config.plot_type == PlotType.GAUGE:
            self.figure = go.Figure()
            self.figure.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=0,
                title={'text': self.config.title}
            ))
        
        elif self.config.plot_type == PlotType.PIE:
            self.figure = go.Figure()
            self.figure.add_trace(go.Pie(labels=[], values=[], name=self.config.title))
        
        elif self.config.plot_type == PlotType.TRAJECTORY_3D:
            self.figure = go.Figure()
            self.figure.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines+markers', 
                                             name=self.config.title))
        
        # Configure layout
        self._configure_plotly_layout()
    
    def _init_matplotlib(self):
        """Initialize Matplotlib figure"""
        if self.config.plot_type == PlotType.TRAJECTORY_3D:
            self.figure, self.ax = plt.subplots(subplot_kw=dict(projection='3d'), 
                                               figsize=(self.config.width/100, self.config.height/100))
        else:
            self.figure, self.ax = plt.subplots(figsize=(self.config.width/100, self.config.height/100))
        
        # Configure matplotlib styling
        self._configure_matplotlib_style()
    
    def _configure_plotly_layout(self):
        """Configure Plotly layout and styling"""
        layout_config = {
            'title': self.config.title,
            'xaxis_title': self.config.x_label,
            'yaxis_title': self.config.y_label,
            'width': self.config.width,
            'height': self.config.height,
            'showlegend': self.config.show_legend,
            'plot_bgcolor': self.config.background_color,
            'font_size': self.config.font_size
        }
        
        # Style-specific configurations
        if self.config.style == PlotStyle.DARK:
            layout_config.update({
                'plot_bgcolor': '#1f1f1f',
                'paper_bgcolor': '#1f1f1f',
                'font_color': 'white'
            })
        elif self.config.style == PlotStyle.MISSION_CONTROL:
            layout_config.update({
                'plot_bgcolor': '#0f0f23',
                'paper_bgcolor': '#0f0f23',
                'font_color': '#00ff00'
            })
        
        # Grid configuration
        if self.config.show_grid:
            layout_config.update({
                'xaxis_showgrid': True,
                'yaxis_showgrid': True
            })
        
        # Range configuration
        if self.config.x_range:
            layout_config['xaxis_range'] = list(self.config.x_range)
        if self.config.y_range:
            layout_config['yaxis_range'] = list(self.config.y_range)
        
        # 3D specific configuration
        if self.config.plot_type == PlotType.TRAJECTORY_3D:
            layout_config.update({
                'scene': {
                    'xaxis_title': self.config.x_label,
                    'yaxis_title': self.config.y_label,
                    'zaxis_title': self.config.z_label,
                    'aspectmode': 'cube'
                }
            })
            if self.config.z_range:
                layout_config['scene']['zaxis_range'] = list(self.config.z_range)
        
        self.figure.update_layout(**layout_config)
    
    def _configure_matplotlib_style(self):
        """Configure Matplotlib styling"""
        if self.config.style == PlotStyle.DARK:
            plt.style.use('dark_background')
        elif self.config.style == PlotStyle.SCIENTIFIC:
            plt.style.use('seaborn-v0_8-paper')
        
        self.ax.set_title(self.config.title, fontsize=self.config.font_size + 2)
        self.ax.set_xlabel(self.config.x_label, fontsize=self.config.font_size)
        self.ax.set_ylabel(self.config.y_label, fontsize=self.config.font_size)
        
        if self.config.plot_type == PlotType.TRAJECTORY_3D:
            self.ax.set_zlabel(self.config.z_label, fontsize=self.config.font_size)
        
        if self.config.show_grid:
            self.ax.grid(True)
    
    def add_data(self, data: Union[PlotData, Dict[str, Any], List[float]]):
        """Add new data to the plot"""
        if isinstance(data, dict):
            plot_data = PlotData(**data)
        elif isinstance(data, list):
            plot_data = PlotData(y=data, x=list(range(len(data))))
        else:
            plot_data = data
        
        # Add to buffer
        self.data_buffer.append(plot_data)
        self.current_data = plot_data
        self.last_update = datetime.now()
        self.needs_update = True
        
        # Notify subscribers
        for callback in self.subscribers:
            try:
                callback(plot_data)
            except Exception as e:
                self.logger.error(f"Error in subscriber callback: {e}")
        
        # Queue update
        self.update_queue.put(plot_data)
    
    def update_plot(self):
        """Update the plot with latest data"""
        if not self.needs_update:
            return
        
        try:
            if PLOTLY_AVAILABLE and isinstance(self.figure, go.Figure):
                self._update_plotly()
            elif MATPLOTLIB_AVAILABLE:
                self._update_matplotlib()
            
            self.needs_update = False
            
        except Exception as e:
            self.logger.error(f"Error updating plot: {e}")
    
    def _update_plotly(self):
        """Update Plotly figure"""
        if not self.data_buffer:
            return
        
        # Collect all data from buffer
        all_x = []
        all_y = []
        all_z = []
        
        for data_point in self.data_buffer:
            all_x.extend(data_point.x)
            all_y.extend(data_point.y)
            if data_point.z:
                all_z.extend(data_point.z)
        
        # Update traces based on plot type
        if self.config.plot_type == PlotType.LINE:
            self.figure.data[0].x = all_x
            self.figure.data[0].y = all_y
        
        elif self.config.plot_type == PlotType.SCATTER:
            self.figure.data[0].x = all_x
            self.figure.data[0].y = all_y
        
        elif self.config.plot_type == PlotType.BAR:
            latest_data = self.data_buffer[-1]
            self.figure.data[0].x = latest_data.labels or list(range(len(latest_data.y)))
            self.figure.data[0].y = latest_data.y
        
        elif self.config.plot_type == PlotType.HISTOGRAM:
            self.figure.data[0].x = all_y  # Use y values for histogram
        
        elif self.config.plot_type == PlotType.GAUGE:
            if self.data_buffer:
                latest_value = self.data_buffer[-1].y[-1] if self.data_buffer[-1].y else 0
                self.figure.data[0].value = latest_value
        
        elif self.config.plot_type == PlotType.PIE:
            latest_data = self.data_buffer[-1]
            self.figure.data[0].labels = latest_data.labels
            self.figure.data[0].values = latest_data.y
        
        elif self.config.plot_type == PlotType.TRAJECTORY_3D:
            self.figure.data[0].x = all_x
            self.figure.data[0].y = all_y
            self.figure.data[0].z = all_z
        
        # Auto-scale if enabled
        if self.config.auto_scale and self.config.plot_type not in [PlotType.GAUGE, PlotType.PIE]:
            self.figure.update_layout(
                xaxis_autorange=True,
                yaxis_autorange=True
            )
    
    def _update_matplotlib(self):
        """Update Matplotlib figure"""
        if not self.data_buffer:
            return
        
        self.ax.clear()
        self._configure_matplotlib_style()
        
        # Collect data
        all_x = []
        all_y = []
        all_z = []
        
        for data_point in self.data_buffer:
            all_x.extend(data_point.x)
            all_y.extend(data_point.y)
            if data_point.z:
                all_z.extend(data_point.z)
        
        # Plot based on type
        if self.config.plot_type == PlotType.LINE:
            self.ax.plot(all_x, all_y, '-o', markersize=3)
        
        elif self.config.plot_type == PlotType.SCATTER:
            self.ax.scatter(all_x, all_y, alpha=0.6)
        
        elif self.config.plot_type == PlotType.BAR:
            latest_data = self.data_buffer[-1]
            labels = latest_data.labels or list(range(len(latest_data.y)))
            self.ax.bar(labels, latest_data.y)
        
        elif self.config.plot_type == PlotType.HISTOGRAM:
            self.ax.hist(all_y, bins=30, alpha=0.7)
        
        elif self.config.plot_type == PlotType.TRAJECTORY_3D:
            self.ax.plot(all_x, all_y, all_z, '-o', markersize=3)
        
        # Auto-scale if enabled
        if self.config.auto_scale:
            self.ax.relim()
            self.ax.autoscale_view()
        
        plt.tight_layout()
    
    def get_figure(self):
        """Get the current figure"""
        return self.figure
    
    def subscribe(self, callback: Callable[[PlotData], None]):
        """Subscribe to data updates"""
        self.subscribers.append(callback)
    
    def export_data(self, filename: str, format: str = 'csv'):
        """Export plot data to file"""
        if not self.data_buffer:
            self.logger.warning("No data to export")
            return
        
        # Collect all data
        data_records = []
        for i, data_point in enumerate(self.data_buffer):
            for j, (x, y) in enumerate(zip(data_point.x, data_point.y)):
                record = {
                    'point_index': i,
                    'data_index': j,
                    'x': x,
                    'y': y,
                    'timestamp': data_point.timestamp.isoformat()
                }
                if data_point.z and j < len(data_point.z):
                    record['z'] = data_point.z[j]
                data_records.append(record)
        
        # Export based on format
        if format.lower() == 'csv':
            df = pd.DataFrame(data_records)
            df.to_csv(filename, index=False)
        elif format.lower() == 'json':
            with open(filename, 'w') as f:
                json.dump(data_records, f, indent=2, default=str)
        elif format.lower() == 'html' and PLOTLY_AVAILABLE:
            self.figure.write_html(filename)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Exported plot data to {filename}")


class RealTimeVisualizer:
    """Main real-time visualization system"""
    
    def __init__(self):
        self.logger = get_component_logger(ComponentType.MONITORING, "visualizer")
        self.plot_managers: Dict[str, PlotManager] = {}
        self.data_processors: Dict[TelemetryType, Callable] = {}
        self.running = False
        
        # Update thread
        self.update_thread: Optional[threading.Thread] = None
        self.update_interval = 0.1  # seconds
        
        # Data sources
        self.telemetry_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.spacecraft_states: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Default plot configurations
        self._create_default_plots()
        
        # Register default data processors
        self._register_default_processors()
    
    def _create_default_plots(self):
        """Create default plot configurations"""
        default_plots = [
            PlotConfig(
                plot_id="position_3d",
                plot_type=PlotType.TRAJECTORY_3D,
                title="Spacecraft Trajectories",
                x_label="X (m)",
                y_label="Y (m)",
                z_label="Z (m)",
                animated=True,
                trail_length=100
            ),
            PlotConfig(
                plot_id="fuel_levels",
                plot_type=PlotType.LINE,
                title="Fuel Levels",
                x_label="Time (s)",
                y_label="Fuel Mass (kg)",
                color_scheme="reds"
            ),
            PlotConfig(
                plot_id="system_health",
                plot_type=PlotType.GAUGE,
                title="System Health",
                y_range=(0, 100),
                color_scheme="RdYlGn"
            ),
            PlotConfig(
                plot_id="velocity_profile",
                plot_type=PlotType.LINE,
                title="Velocity Profiles",
                x_label="Time (s)",
                y_label="Velocity (m/s)",
                color_scheme="blues"
            ),
            PlotConfig(
                plot_id="mission_progress",
                plot_type=PlotType.PIE,
                title="Mission Status Distribution",
                color_scheme="Set3"
            )
        ]
        
        for config in default_plots:
            self.add_plot(config)
    
    def _register_default_processors(self):
        """Register default data processors for telemetry types"""
        self.data_processors[TelemetryType.POSITION] = self._process_position_data
        self.data_processors[TelemetryType.VELOCITY] = self._process_velocity_data
        self.data_processors[TelemetryType.FUEL] = self._process_fuel_data
        self.data_processors[TelemetryType.SYSTEM_STATE] = self._process_system_state_data
    
    def add_plot(self, config: PlotConfig):
        """Add a new plot to the visualizer"""
        plot_manager = PlotManager(config)
        self.plot_managers[config.plot_id] = plot_manager
        self.logger.info(f"Added plot: {config.plot_id}")
    
    def remove_plot(self, plot_id: str):
        """Remove a plot from the visualizer"""
        if plot_id in self.plot_managers:
            del self.plot_managers[plot_id]
            self.logger.info(f"Removed plot: {plot_id}")
    
    def get_plot(self, plot_id: str) -> Optional[PlotManager]:
        """Get a specific plot manager"""
        return self.plot_managers.get(plot_id)
    
    def process_telemetry(self, telemetry_point: TelemetryPoint):
        """Process incoming telemetry data"""
        agent_id = telemetry_point.agent_id
        tel_type = telemetry_point.telemetry_type
        
        # Store in buffer
        self.telemetry_buffer[agent_id].append(telemetry_point)
        
        # Process with registered processor
        if tel_type in self.data_processors:
            try:
                self.data_processors[tel_type](telemetry_point)
            except Exception as e:
                self.logger.error(f"Error processing telemetry type {tel_type}: {e}")
    
    def process_spacecraft_state(self, state: SpacecraftState):
        """Process spacecraft state data"""
        agent_id = state.agent_id
        
        # Store in buffer
        self.spacecraft_states[agent_id].append(state)
        
        # Update relevant plots
        self._update_3d_trajectory(state)
        self._update_fuel_plot(state)
        self._update_health_plot(state)
    
    def _process_position_data(self, telemetry_point: TelemetryPoint):
        """Process position telemetry"""
        if len(telemetry_point.data) >= 3:
            plot_data = PlotData(
                x=[telemetry_point.data[0]],
                y=[telemetry_point.data[1]],
                z=[telemetry_point.data[2]],
                metadata={
                    'agent_id': telemetry_point.agent_id,
                    'mission_time': telemetry_point.mission_time
                }
            )
            
            if "position_3d" in self.plot_managers:
                self.plot_managers["position_3d"].add_data(plot_data)
    
    def _process_velocity_data(self, telemetry_point: TelemetryPoint):
        """Process velocity telemetry"""
        if isinstance(telemetry_point.data, list) and len(telemetry_point.data) >= 3:
            velocity_magnitude = np.linalg.norm(telemetry_point.data)
        else:
            velocity_magnitude = float(telemetry_point.data)
        
        plot_data = PlotData(
            x=[telemetry_point.mission_time],
            y=[velocity_magnitude],
            metadata={
                'agent_id': telemetry_point.agent_id,
                'velocity_vector': telemetry_point.data
            }
        )
        
        if "velocity_profile" in self.plot_managers:
            self.plot_managers["velocity_profile"].add_data(plot_data)
    
    def _process_fuel_data(self, telemetry_point: TelemetryPoint):
        """Process fuel telemetry"""
        plot_data = PlotData(
            x=[telemetry_point.mission_time],
            y=[float(telemetry_point.data)],
            metadata={
                'agent_id': telemetry_point.agent_id
            }
        )
        
        if "fuel_levels" in self.plot_managers:
            self.plot_managers["fuel_levels"].add_data(plot_data)
    
    def _process_system_state_data(self, telemetry_point: TelemetryPoint):
        """Process system state telemetry"""
        if isinstance(telemetry_point.data, dict):
            health_score = telemetry_point.data.get('health_score', 100.0)
        else:
            health_score = float(telemetry_point.data)
        
        plot_data = PlotData(
            x=[telemetry_point.mission_time],
            y=[health_score],
            metadata={
                'agent_id': telemetry_point.agent_id
            }
        )
        
        if "system_health" in self.plot_managers:
            self.plot_managers["system_health"].add_data(plot_data)
    
    def _update_3d_trajectory(self, state: SpacecraftState):
        """Update 3D trajectory plot"""
        plot_data = PlotData(
            x=[state.position[0]],
            y=[state.position[1]],
            z=[state.position[2]],
            metadata={
                'agent_id': state.agent_id,
                'mission_time': state.mission_time,
                'velocity': state.velocity.tolist(),
                'mission_phase': state.mission_phase.value
            }
        )
        
        if "position_3d" in self.plot_managers:
            self.plot_managers["position_3d"].add_data(plot_data)
    
    def _update_fuel_plot(self, state: SpacecraftState):
        """Update fuel level plot"""
        plot_data = PlotData(
            x=[state.mission_time],
            y=[state.fuel_mass],
            metadata={
                'agent_id': state.agent_id
            }
        )
        
        if "fuel_levels" in self.plot_managers:
            self.plot_managers["fuel_levels"].add_data(plot_data)
    
    def _update_health_plot(self, state: SpacecraftState):
        """Update system health plot"""
        plot_data = PlotData(
            x=[state.mission_time],
            y=[state.system_health],
            metadata={
                'agent_id': state.agent_id
            }
        )
        
        if "system_health" in self.plot_managers:
            self.plot_managers["system_health"].add_data(plot_data)
    
    def start(self):
        """Start the real-time visualization system"""
        if self.running:
            return
        
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        self.logger.info("Started real-time visualizer")
    
    def stop(self):
        """Stop the real-time visualization system"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5.0)
        self.logger.info("Stopped real-time visualizer")
    
    def _update_loop(self):
        """Main update loop for plot refresh"""
        while self.running:
            try:
                # Update all plots
                for plot_manager in self.plot_managers.values():
                    plot_manager.update_plot()
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in update loop: {e}")
                time.sleep(1.0)
    
    def create_mission_summary_plot(self, mission_data: Dict[str, Any]) -> go.Figure:
        """Create a comprehensive mission summary plot"""
        if not PLOTLY_AVAILABLE:
            self.logger.error("Plotly not available for mission summary plot")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Trajectory Overview', 'Fuel Consumption',
                'Velocity Profiles', 'System Health',
                'Mission Progress', 'Performance Metrics'
            ],
            specs=[
                [{"type": "scatter3d"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "pie"}, {"type": "bar"}]
            ]
        )
        
        # Add trajectory data (3D)
        if 'trajectories' in mission_data:
            for agent_id, trajectory in mission_data['trajectories'].items():
                fig.add_trace(
                    go.Scatter3d(
                        x=trajectory['x'],
                        y=trajectory['y'],
                        z=trajectory['z'],
                        mode='lines+markers',
                        name=f'{agent_id} Trajectory',
                        line=dict(width=3)
                    ),
                    row=1, col=1
                )
        
        # Add fuel consumption data
        if 'fuel_data' in mission_data:
            for agent_id, fuel_data in mission_data['fuel_data'].items():
                fig.add_trace(
                    go.Scatter(
                        x=fuel_data['time'],
                        y=fuel_data['fuel'],
                        mode='lines',
                        name=f'{agent_id} Fuel',
                        line=dict(width=2)
                    ),
                    row=1, col=2
                )
        
        # Add velocity profiles
        if 'velocity_data' in mission_data:
            for agent_id, vel_data in mission_data['velocity_data'].items():
                fig.add_trace(
                    go.Scatter(
                        x=vel_data['time'],
                        y=vel_data['velocity'],
                        mode='lines',
                        name=f'{agent_id} Velocity',
                        line=dict(width=2)
                    ),
                    row=2, col=1
                )
        
        # Add system health
        if 'health_data' in mission_data:
            for agent_id, health_data in mission_data['health_data'].items():
                fig.add_trace(
                    go.Scatter(
                        x=health_data['time'],
                        y=health_data['health'],
                        mode='lines',
                        name=f'{agent_id} Health',
                        line=dict(width=2)
                    ),
                    row=2, col=2
                )
        
        # Add mission progress pie chart
        if 'mission_status' in mission_data:
            status_data = mission_data['mission_status']
            fig.add_trace(
                go.Pie(
                    labels=list(status_data.keys()),
                    values=list(status_data.values()),
                    name="Mission Status"
                ),
                row=3, col=1
            )
        
        # Add performance metrics bar chart
        if 'performance_metrics' in mission_data:
            metrics = mission_data['performance_metrics']
            fig.add_trace(
                go.Bar(
                    x=list(metrics.keys()),
                    y=list(metrics.values()),
                    name="Performance Metrics"
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Mission Summary Dashboard",
            showlegend=True
        )
        
        return fig
    
    def export_all_plots(self, output_dir: str, format: str = 'html'):
        """Export all plots to files"""
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for plot_id, plot_manager in self.plot_managers.items():
            filename = output_path / f"{plot_id}.{format}"
            try:
                plot_manager.export_data(str(filename), format)
            except Exception as e:
                self.logger.error(f"Error exporting plot {plot_id}: {e}")
        
        self.logger.info(f"Exported all plots to {output_dir}")


# Example specialized plot types
class TrajectoryPlot(PlotManager):
    """Specialized trajectory plotting"""
    
    def __init__(self, plot_id: str = "trajectory"):
        config = PlotConfig(
            plot_id=plot_id,
            plot_type=PlotType.TRAJECTORY_3D,
            title="Spacecraft Trajectory",
            x_label="X Position (m)",
            y_label="Y Position (m)",
            z_label="Z Position (m)",
            animated=True,
            trail_length=200,
            color_scheme="viridis"
        )
        super().__init__(config)
        self.agent_colors = {}
        self.color_index = 0
        self.color_palette = px.colors.qualitative.Set1
    
    def add_trajectory_point(self, agent_id: str, position: Tuple[float, float, float], 
                           timestamp: datetime):
        """Add a trajectory point for a specific agent"""
        # Assign color to agent if not already assigned
        if agent_id not in self.agent_colors:
            self.agent_colors[agent_id] = self.color_palette[self.color_index % len(self.color_palette)]
            self.color_index += 1
        
        plot_data = PlotData(
            x=[position[0]],
            y=[position[1]], 
            z=[position[2]],
            metadata={
                'agent_id': agent_id,
                'timestamp': timestamp,
                'color': self.agent_colors[agent_id]
            }
        )
        
        self.add_data(plot_data)


class PerformancePlot(PlotManager):
    """Specialized performance monitoring plot"""
    
    def __init__(self, plot_id: str = "performance"):
        config = PlotConfig(
            plot_id=plot_id,
            plot_type=PlotType.LINE,
            title="Algorithm Performance",
            x_label="Time (s)",
            y_label="Execution Time (ms)",
            color_scheme="blues",
            max_points=500
        )
        super().__init__(config)
    
    def add_performance_data(self, algorithm_name: str, execution_time: float,
                           timestamp: datetime):
        """Add performance data for an algorithm"""
        plot_data = PlotData(
            x=[timestamp.timestamp()],
            y=[execution_time * 1000],  # Convert to milliseconds
            metadata={
                'algorithm': algorithm_name,
                'timestamp': timestamp
            }
        )
        
        self.add_data(plot_data)


# Example usage
if __name__ == "__main__":
    # Create visualizer
    visualizer = RealTimeVisualizer()
    visualizer.start()
    
    # Simulate some data
    import random
    from spacecraft_drmpc.monitoring.telemetry_collector import TelemetryPoint, TelemetryType, SpacecraftState, MissionPhase
    
    # Simulate telemetry data
    for i in range(100):
        # Position telemetry
        position_tel = TelemetryPoint(
            agent_id="TEST_AGENT",
            telemetry_type=TelemetryType.POSITION,
            data=[i * 0.1 + random.random(), i * 0.05 + random.random(), random.random()],
            mission_time=i * 0.1
        )
        visualizer.process_telemetry(position_tel)
        
        # Velocity telemetry  
        velocity_tel = TelemetryPoint(
            agent_id="TEST_AGENT",
            telemetry_type=TelemetryType.VELOCITY,
            data=[1.0 + 0.1 * random.random(), 0.5 + 0.1 * random.random(), 0.1 * random.random()],
            mission_time=i * 0.1
        )
        visualizer.process_telemetry(velocity_tel)
        
        # Spacecraft state
        state = SpacecraftState(
            agent_id="TEST_AGENT",
            mission_time=i * 0.1,
            position=np.array([i * 0.1 + random.random(), i * 0.05 + random.random(), random.random()]),
            velocity=np.array([1.0, 0.5, 0.1]),
            fuel_mass=100 - i * 0.5,
            system_health=100 - i * 0.1,
            mission_phase=MissionPhase.CRUISE
        )
        visualizer.process_spacecraft_state(state)
        
        time.sleep(0.1)
    
    # Export plots
    visualizer.export_all_plots("test_plots", "html")
    
    # Stop visualizer
    visualizer.stop()