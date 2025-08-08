"""
Mission Control Dashboard Interface

This module provides the main mission control interface with real-time monitoring,
interactive controls, and comprehensive mission status display.
"""

import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
from collections import defaultdict, deque

# Web framework and real-time communication
try:
    import dash
    from dash import dcc, html, Input, Output, State, callback, clientside_callback
    import dash_bootstrap_components as dbc
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

# WebSocket support for real-time updates
try:
    import websockets
    import socketio
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

from spacecraft_drmpc.monitoring.system_logger import get_component_logger, ComponentType
from spacecraft_drmpc.monitoring.telemetry_collector import TelemetryCollector, SpacecraftState, MissionPhase
from spacecraft_drmpc.monitoring.health_monitor import HealthMonitor, ComponentHealth, HealthStatus
from spacecraft_drmpc.monitoring.performance_metrics import PerformanceMonitor

from .data_visualizer import RealTimeVisualizer
from .health_display import HealthDashboard
from .interactive_3d import Interactive3DView


@dataclass
class MissionStatus:
    """Current mission status information"""
    mission_id: str = ""
    phase: MissionPhase = MissionPhase.INITIALIZATION
    start_time: datetime = None
    elapsed_time: float = 0.0
    active_agents: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    alerts_count: int = 0
    system_health: float = 100.0
    mission_progress: float = 0.0


@dataclass
class AgentStatus:
    """Individual agent status for dashboard"""
    agent_id: str = ""
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    fuel_level: float = 100.0
    health_status: str = "healthy"
    mission_phase: str = "initialization"
    target_distance: float = 0.0
    last_update: datetime = None


class MissionControlDashboard:
    """Main mission control dashboard application"""
    
    def __init__(self, mission_id: str, host: str = "localhost", port: int = 8050):
        self.mission_id = mission_id
        self.host = host
        self.port = port
        self.logger = get_component_logger(ComponentType.MONITORING, f"dashboard.{mission_id}")
        
        # Dashboard state
        self.running = False
        self.mission_status = MissionStatus(mission_id=mission_id, start_time=datetime.now())
        self.agent_states: Dict[str, AgentStatus] = {}
        self.telemetry_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # External system connections
        self.telemetry_collector: Optional[TelemetryCollector] = None
        self.health_monitor: Optional[HealthMonitor] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        
        # Dashboard components
        self.visualizer = RealTimeVisualizer()
        self.health_dashboard = HealthDashboard()
        self.interactive_3d = Interactive3DView()
        
        # Real-time update system
        self.update_callbacks: List[Callable] = []
        self.websocket_server = None
        self.connected_clients: set = set()
        
        # User interface state
        self.selected_agent = None
        self.view_mode = "overview"  # overview, detailed, 3d, health
        self.auto_refresh = True
        self.refresh_interval = 1.0  # seconds
        
        if not DASH_AVAILABLE:
            self.logger.error("Dash is not available. Dashboard functionality will be limited.")
            return
        
        # Initialize Dash application
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
            title=f"Mission Control - {mission_id}"
        )
        
        self._setup_layout()
        self._setup_callbacks()
        
        # Start background update thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
    
    def connect_telemetry_collector(self, collector: TelemetryCollector):
        """Connect telemetry data source"""
        self.telemetry_collector = collector
        collector.register_processor(self._process_telemetry_data)
        self.logger.info("Connected telemetry collector")
    
    def connect_health_monitor(self, monitor: HealthMonitor):
        """Connect health monitoring system"""
        self.health_monitor = monitor
        monitor.register_alert_callback(self._process_health_alert)
        self.logger.info("Connected health monitor")
    
    def connect_performance_monitor(self, monitor: PerformanceMonitor):
        """Connect performance monitoring system"""
        self.performance_monitor = monitor
        self.logger.info("Connected performance monitor")
    
    def _setup_layout(self):
        """Setup the dashboard layout"""
        if not DASH_AVAILABLE:
            return
        
        self.app.layout = dbc.Container([
            # Header
            self._create_header(),
            
            # Main content area
            dcc.Tabs(id="main-tabs", value="overview", children=[
                dcc.Tab(label="Mission Overview", value="overview", children=[
                    self._create_overview_tab()
                ]),
                dcc.Tab(label="Detailed View", value="detailed", children=[
                    self._create_detailed_tab()
                ]),
                dcc.Tab(label="3D Visualization", value="3d", children=[
                    self._create_3d_tab()
                ]),
                dcc.Tab(label="System Health", value="health", children=[
                    self._create_health_tab()
                ]),
                dcc.Tab(label="Performance", value="performance", children=[
                    self._create_performance_tab()
                ])
            ]),
            
            # Footer with controls
            self._create_footer(),
            
            # Interval components for auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=1000,  # Update every second
                n_intervals=0,
                disabled=False
            ),
            
            # Store components for state management
            dcc.Store(id='mission-store'),
            dcc.Store(id='agent-store'),
            dcc.Store(id='alert-store')
        ], fluid=True)
    
    def _create_header(self):
        """Create dashboard header"""
        return dbc.Row([
            dbc.Col([
                html.H1([
                    html.I(className="fas fa-rocket me-2"),
                    f"Mission Control - {self.mission_id}"
                ], className="text-primary mb-0"),
                html.P(f"Started: {self.mission_status.start_time.strftime('%Y-%m-%d %H:%M:%S')}", 
                      className="text-muted small")
            ], width=6),
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        dbc.Badge("ACTIVE", color="success", className="me-2"),
                        html.Span(id="mission-time", className="fw-bold")
                    ]),
                    dbc.Col([
                        dbc.ButtonGroup([
                            dbc.Button([
                                html.I(className="fas fa-play" if not self.auto_refresh else "fas fa-pause"),
                                " Auto Refresh"
                            ], id="refresh-btn", color="primary", size="sm"),
                            dbc.Button([
                                html.I(className="fas fa-download"),
                                " Export"
                            ], id="export-btn", color="secondary", size="sm"),
                            dbc.Button([
                                html.I(className="fas fa-cog"),
                                " Settings"
                            ], id="settings-btn", color="info", size="sm")
                        ])
                    ], width="auto")
                ], justify="end")
            ], width=6)
        ], className="mb-3 p-3 bg-light rounded")
    
    def _create_overview_tab(self):
        """Create mission overview tab"""
        return dbc.Container([
            # Mission status cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4([
                                html.I(className="fas fa-users text-primary me-2"),
                                "Active Agents"
                            ]),
                            html.H2(id="active-agents-count", children="0", className="text-primary"),
                            html.Small("Spacecraft Online", className="text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4([
                                html.I(className="fas fa-check-circle text-success me-2"),
                                "Operations"
                            ]),
                            html.H2(id="successful-ops", children="0", className="text-success"),
                            html.Small("Successful", className="text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4([
                                html.I(className="fas fa-exclamation-triangle text-warning me-2"),
                                "Alerts"
                            ]),
                            html.H2(id="alerts-count", children="0", className="text-warning"),
                            html.Small("Active Warnings", className="text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4([
                                html.I(className="fas fa-heartbeat text-info me-2"),
                                "System Health"
                            ]),
                            html.H2(id="system-health", children="100%", className="text-info"),
                            html.Small("Overall Status", className="text-muted")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            # Mission progress and timeline
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-tasks me-2"),
                                "Mission Progress"
                            ])
                        ]),
                        dbc.CardBody([
                            dbc.Progress(id="mission-progress", value=0, striped=True, animated=True),
                            html.Div(id="mission-phase", className="mt-2 text-center fw-bold")
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Live charts
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-chart-line me-2"),
                                "Real-time Telemetry"
                            ])
                        ]),
                        dbc.CardBody([
                            dcc.Graph(id="telemetry-chart", style={'height': '400px'})
                        ])
                    ])
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-list me-2"),
                                "Agent Status"
                            ])
                        ]),
                        dbc.CardBody([
                            html.Div(id="agent-status-list")
                        ])
                    ])
                ], width=4)
            ])
        ], fluid=True)
    
    def _create_detailed_tab(self):
        """Create detailed view tab"""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-search me-2"),
                                "Agent Selection"
                            ])
                        ]),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id="agent-selector",
                                placeholder="Select an agent for detailed view",
                                className="mb-3"
                            ),
                            html.Div(id="selected-agent-info")
                        ])
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-chart-area me-2"),
                                "Detailed Telemetry"
                            ])
                        ]),
                        dbc.CardBody([
                            dcc.Graph(id="detailed-telemetry-chart", style={'height': '400px'})
                        ])
                    ])
                ], width=8)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-map me-2"),
                                "Trajectory Plot"
                            ])
                        ]),
                        dbc.CardBody([
                            dcc.Graph(id="trajectory-plot", style={'height': '400px'})
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-tachometer-alt me-2"),
                                "Performance Metrics"
                            ])
                        ]),
                        dbc.CardBody([
                            html.Div(id="performance-metrics")
                        ])
                    ])
                ], width=6)
            ])
        ], fluid=True)
    
    def _create_3d_tab(self):
        """Create 3D visualization tab"""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-cube me-2"),
                                "3D Mission View"
                            ])
                        ]),
                        dbc.CardBody([
                            dcc.Graph(
                                id="3d-visualization",
                                style={'height': '600px'},
                                config={'displayModeBar': True}
                            )
                        ])
                    ])
                ], width=9),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-sliders-h me-2"),
                                "View Controls"
                            ])
                        ]),
                        dbc.CardBody([
                            html.Label("View Mode:"),
                            dcc.Dropdown(
                                id="3d-view-mode",
                                options=[
                                    {"label": "Orbital View", "value": "orbital"},
                                    {"label": "Formation View", "value": "formation"},
                                    {"label": "Docking View", "value": "docking"},
                                    {"label": "Free Camera", "value": "free"}
                                ],
                                value="orbital",
                                className="mb-3"
                            ),
                            
                            html.Label("Time Range:"),
                            dcc.RangeSlider(
                                id="3d-time-range",
                                min=0,
                                max=100,
                                value=[80, 100],
                                marks={i: f"{i}%" for i in range(0, 101, 20)},
                                className="mb-3"
                            ),
                            
                            html.Label("Display Options:"),
                            dbc.Checklist(
                                id="3d-display-options",
                                options=[
                                    {"label": "Show Trajectories", "value": "trajectories"},
                                    {"label": "Show Velocity Vectors", "value": "velocities"},
                                    {"label": "Show Target Zones", "value": "targets"},
                                    {"label": "Show Communication Links", "value": "comms"}
                                ],
                                value=["trajectories"],
                                className="mb-3"
                            ),
                            
                            html.Hr(),
                            dbc.Button([
                                html.I(className="fas fa-play me-2"),
                                "Play Animation"
                            ], id="3d-play-btn", color="success", className="w-100 mb-2"),
                            
                            dbc.Button([
                                html.I(className="fas fa-camera me-2"),
                                "Export View"
                            ], id="3d-export-btn", color="info", className="w-100")
                        ])
                    ])
                ], width=3)
            ])
        ], fluid=True)
    
    def _create_health_tab(self):
        """Create system health monitoring tab"""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-heartbeat me-2"),
                                "System Health Overview"
                            ])
                        ]),
                        dbc.CardBody([
                            html.Div(id="health-overview")
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-exclamation-circle me-2"),
                                "Active Alerts"
                            ])
                        ]),
                        dbc.CardBody([
                            html.Div(id="active-alerts-list")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-chart-pie me-2"),
                                "Component Health"
                            ])
                        ]),
                        dbc.CardBody([
                            dcc.Graph(id="health-pie-chart", style={'height': '300px'})
                        ])
                    ])
                ], width=6)
            ])
        ], fluid=True)
    
    def _create_performance_tab(self):
        """Create performance monitoring tab"""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-tachometer-alt me-2"),
                                "System Performance"
                            ])
                        ]),
                        dbc.CardBody([
                            dcc.Graph(id="performance-charts", style={'height': '400px'})
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-microchip me-2"),
                                "Resource Usage"
                            ])
                        ]),
                        dbc.CardBody([
                            html.Div(id="resource-usage")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-stopwatch me-2"),
                                "Algorithm Performance"
                            ])
                        ]),
                        dbc.CardBody([
                            html.Div(id="algorithm-performance")
                        ])
                    ])
                ], width=6)
            ])
        ], fluid=True)
    
    def _create_footer(self):
        """Create dashboard footer"""
        return dbc.Row([
            dbc.Col([
                html.Hr(),
                html.P([
                    html.Small([
                        "Spacecraft Simulation Dashboard | ",
                        html.Span(id="connection-status", className="text-success"),
                        " | Last Update: ",
                        html.Span(id="last-update-time")
                    ])
                ], className="text-center text-muted")
            ])
        ])
    
    def _setup_callbacks(self):
        """Setup Dash callbacks for interactivity"""
        if not DASH_AVAILABLE:
            return
        
        # Main update callback
        @self.app.callback(
            [Output('mission-time', 'children'),
             Output('active-agents-count', 'children'),
             Output('successful-ops', 'children'),
             Output('alerts-count', 'children'),
             Output('system-health', 'children'),
             Output('mission-progress', 'value'),
             Output('mission-phase', 'children'),
             Output('last-update-time', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_mission_status(n):
            if not self.running:
                return "00:00:00", "0", "0", "0", "100%", 0, "Stopped", ""
            
            # Calculate elapsed time
            elapsed = (datetime.now() - self.mission_status.start_time).total_seconds()
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            
            # Update mission status
            self.mission_status.elapsed_time = elapsed
            self.mission_status.active_agents = len(self.agent_states)
            
            # Get health information
            health_score = "100%"
            if self.health_monitor:
                health_status = self.health_monitor.get_health_status()
                if health_status:
                    avg_health = np.mean([h.overall_health_score for h in health_status.values() if h])
                    health_score = f"{avg_health:.1f}%"
            
            # Get alerts count
            alerts_count = 0
            if self.health_monitor:
                alerts_count = len(self.health_monitor.get_active_alerts())
            
            return (
                elapsed_str,
                str(self.mission_status.active_agents),
                str(self.mission_status.successful_operations),
                str(alerts_count),
                health_score,
                self.mission_status.mission_progress,
                self.mission_status.phase.value.title(),
                datetime.now().strftime('%H:%M:%S')
            )
        
        # Telemetry chart callback
        @self.app.callback(
            Output('telemetry-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_telemetry_chart(n):
            return self._create_telemetry_chart()
        
        # Agent status list callback
        @self.app.callback(
            Output('agent-status-list', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_agent_status(n):
            return self._create_agent_status_cards()
        
        # 3D visualization callback
        @self.app.callback(
            Output('3d-visualization', 'figure'),
            [Input('interval-component', 'n_intervals'),
             Input('3d-view-mode', 'value'),
             Input('3d-display-options', 'value')]
        )
        def update_3d_view(n, view_mode, display_options):
            return self._create_3d_visualization(view_mode, display_options)
        
        # Health overview callback
        @self.app.callback(
            Output('health-overview', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_health_overview(n):
            return self._create_health_overview()
        
        # Performance charts callback
        @self.app.callback(
            Output('performance-charts', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_performance_charts(n):
            return self._create_performance_charts()
    
    def _create_telemetry_chart(self):
        """Create real-time telemetry chart"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Position', 'Velocity', 'Fuel Level', 'System Health'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        if not self.agent_states:
            fig.add_annotation(text="No data available", xref="paper", yref="paper", 
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Add traces for each agent
        for agent_id, agent_status in self.agent_states.items():
            if agent_status.last_update is None:
                continue
            
            # Position trace
            fig.add_trace(
                go.Scatter(
                    x=[agent_status.last_update],
                    y=[np.linalg.norm(agent_status.position)],
                    mode='lines+markers',
                    name=f'{agent_id} Position',
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Velocity trace
            fig.add_trace(
                go.Scatter(
                    x=[agent_status.last_update],
                    y=[np.linalg.norm(agent_status.velocity)],
                    mode='lines+markers',
                    name=f'{agent_id} Velocity',
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # Fuel level trace
            fig.add_trace(
                go.Scatter(
                    x=[agent_status.last_update],
                    y=[agent_status.fuel_level],
                    mode='lines+markers',
                    name=f'{agent_id} Fuel',
                    showlegend=False
                ),
                row=2, col=1
            )
        
        fig.update_layout(height=400, showlegend=True, 
                         title_text="Real-time Telemetry Data")
        return fig
    
    def _create_agent_status_cards(self):
        """Create agent status cards"""
        if not self.agent_states:
            return dbc.Alert("No active agents", color="info")
        
        cards = []
        for agent_id, status in self.agent_states.items():
            # Determine status color
            if status.health_status == "healthy":
                color = "success"
                icon = "fas fa-check-circle"
            elif status.health_status == "warning":
                color = "warning"  
                icon = "fas fa-exclamation-triangle"
            else:
                color = "danger"
                icon = "fas fa-times-circle"
            
            card = dbc.Card([
                dbc.CardBody([
                    html.H6([
                        html.I(className=f"{icon} me-2"),
                        agent_id
                    ]),
                    html.P([
                        html.Small(f"Phase: {status.mission_phase.title()}", className="text-muted d-block"),
                        html.Small(f"Fuel: {status.fuel_level:.1f}%", className="text-muted d-block"),
                        html.Small(f"Distance: {status.target_distance:.1f}m", className="text-muted d-block")
                    ])
                ])
            ], color=color, outline=True, className="mb-2")
            
            cards.append(card)
        
        return cards
    
    def _create_3d_visualization(self, view_mode="orbital", display_options=None):
        """Create 3D visualization"""
        if display_options is None:
            display_options = ["trajectories"]
        
        fig = go.Figure()
        
        if not self.agent_states:
            fig.add_annotation(text="No data available", xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Add spacecraft positions
        for agent_id, status in self.agent_states.items():
            fig.add_trace(go.Scatter3d(
                x=[status.position[0]],
                y=[status.position[1]], 
                z=[status.position[2]],
                mode='markers',
                marker=dict(size=10, color='red' if status.health_status != "healthy" else 'blue'),
                name=agent_id,
                text=f"{agent_id}<br>Fuel: {status.fuel_level:.1f}%<br>Phase: {status.mission_phase}",
                hovertemplate="%{text}<extra></extra>"
            ))
            
            # Add velocity vectors if requested
            if "velocities" in display_options:
                end_pos = np.array(status.position) + np.array(status.velocity) * 10
                fig.add_trace(go.Scatter3d(
                    x=[status.position[0], end_pos[0]],
                    y=[status.position[1], end_pos[1]],
                    z=[status.position[2], end_pos[2]],
                    mode='lines',
                    line=dict(color='green', width=3),
                    name=f"{agent_id} Velocity",
                    showlegend=False
                ))
        
        # Add target zone if in docking mode
        if view_mode == "docking" and "targets" in display_options:
            # Add target sphere
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = 5 * np.outer(np.cos(u), np.sin(v))
            y = 5 * np.outer(np.sin(u), np.sin(v))
            z = 5 * np.outer(np.ones(np.size(u)), np.cos(v))
            
            fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                opacity=0.3,
                colorscale='Reds',
                showscale=False,
                name="Target Zone"
            ))
        
        fig.update_layout(
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='cube'
            ),
            height=600,
            title="3D Mission Visualization"
        )
        
        return fig
    
    def _create_health_overview(self):
        """Create health status overview"""
        if not self.health_monitor:
            return dbc.Alert("Health monitoring not connected", color="warning")
        
        health_status = self.health_monitor.get_health_status()
        if not health_status:
            return dbc.Alert("No health data available", color="info")
        
        components = []
        for component_id, health in health_status.items():
            if health is None:
                continue
            
            # Determine status badge color
            if health.health_status == HealthStatus.HEALTHY:
                badge_color = "success"
            elif health.health_status == HealthStatus.DEGRADED:
                badge_color = "warning"
            elif health.health_status == HealthStatus.WARNING:
                badge_color = "warning"
            elif health.health_status == HealthStatus.CRITICAL:
                badge_color = "danger"
            else:
                badge_color = "danger"
            
            component = dbc.Row([
                dbc.Col([
                    html.H6(component_id.title())
                ], width=4),
                dbc.Col([
                    dbc.Badge(health.health_status.value.title(), color=badge_color)
                ], width=4),
                dbc.Col([
                    f"{health.overall_health_score:.1f}%"
                ], width=4)
            ], className="mb-2")
            
            components.append(component)
        
        return components
    
    def _create_performance_charts(self):
        """Create performance monitoring charts"""
        if not self.performance_monitor:
            fig = go.Figure()
            fig.add_annotation(text="Performance monitoring not connected", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        current_metrics = self.performance_monitor.get_current_metrics()
        
        if not current_metrics:
            fig = go.Figure()
            fig.add_annotation(text="No performance data available",
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CPU Usage', 'Memory Usage', 'Network I/O', 'Disk I/O')
        )
        
        # Add performance traces
        if 'system' in current_metrics:
            system_metrics = current_metrics['system']
            
            # CPU usage
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=system_metrics.cpu_percent,
                    title={'text': "CPU %"},
                    gauge={'axis': {'range': [None, 100]},
                           'threshold': {'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75, 'value': 90}}
                ),
                row=1, col=1
            )
            
            # Memory usage
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=system_metrics.memory_percent,
                    title={'text': "Memory %"},
                    gauge={'axis': {'range': [None, 100]},
                           'threshold': {'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75, 'value': 85}}
                ),
                row=1, col=2
            )
        
        fig.update_layout(height=400)
        return fig
    
    def _update_loop(self):
        """Background update loop"""
        while True:
            try:
                if self.running and self.auto_refresh:
                    # Update mission status
                    self._update_mission_status()
                    
                    # Send updates to connected WebSocket clients
                    if self.connected_clients:
                        update_data = {
                            'mission_status': asdict(self.mission_status),
                            'agent_states': {k: asdict(v) for k, v in self.agent_states.items()},
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        # Broadcast to WebSocket clients
                        asyncio.run(self._broadcast_update(update_data))
                
                time.sleep(self.refresh_interval)
                
            except Exception as e:
                self.logger.error(f"Error in update loop: {e}")
                time.sleep(5.0)
    
    def _update_mission_status(self):
        """Update internal mission status"""
        # Update elapsed time
        elapsed = (datetime.now() - self.mission_status.start_time).total_seconds()
        self.mission_status.elapsed_time = elapsed
        
        # Calculate mission progress (example logic)
        if self.mission_status.phase == MissionPhase.COMPLETED:
            self.mission_status.mission_progress = 100.0
        else:
            # Simple progress based on elapsed time (customize as needed)
            self.mission_status.mission_progress = min(elapsed / 3600 * 100, 95.0)  # 1 hour = 95%
    
    def _process_telemetry_data(self, telemetry_point):
        """Process incoming telemetry data"""
        agent_id = telemetry_point.agent_id
        
        # Update agent state
        if agent_id not in self.agent_states:
            self.agent_states[agent_id] = AgentStatus(agent_id=agent_id)
        
        agent_status = self.agent_states[agent_id]
        agent_status.last_update = datetime.now()
        
        # Update based on telemetry type
        from spacecraft_drmpc.monitoring.telemetry_collector import TelemetryType
        
        if telemetry_point.telemetry_type == TelemetryType.POSITION:
            agent_status.position = tuple(telemetry_point.data)
        elif telemetry_point.telemetry_type == TelemetryType.VELOCITY:
            agent_status.velocity = tuple(telemetry_point.data)
        elif telemetry_point.telemetry_type == TelemetryType.FUEL:
            agent_status.fuel_level = telemetry_point.data
        
        # Store in telemetry buffer
        self.telemetry_buffer[agent_id].append(telemetry_point)
    
    def _process_spacecraft_state(self, spacecraft_state: SpacecraftState):
        """Process spacecraft state update"""
        agent_id = spacecraft_state.agent_id
        
        # Update agent status
        if agent_id not in self.agent_states:
            self.agent_states[agent_id] = AgentStatus(agent_id=agent_id)
        
        agent_status = self.agent_states[agent_id]
        agent_status.position = tuple(spacecraft_state.position)
        agent_status.velocity = tuple(spacecraft_state.velocity)
        agent_status.fuel_level = spacecraft_state.fuel_mass
        agent_status.mission_phase = spacecraft_state.mission_phase.value
        agent_status.target_distance = spacecraft_state.target_distance
        agent_status.last_update = datetime.now()
        
        # Determine health status
        if spacecraft_state.system_health >= 80:
            agent_status.health_status = "healthy"
        elif spacecraft_state.system_health >= 50:
            agent_status.health_status = "warning"
        else:
            agent_status.health_status = "critical"
        
        # Update mission phase if changed
        if spacecraft_state.mission_phase != self.mission_status.phase:
            self.mission_status.phase = spacecraft_state.mission_phase
            self.logger.info(f"Mission phase changed to: {spacecraft_state.mission_phase.value}")
    
    def _process_health_alert(self, alert):
        """Process health alert"""
        self.logger.warning(f"Health alert received: {alert.message}")
        # Alerts are handled by the callback system
    
    async def _broadcast_update(self, data):
        """Broadcast update to WebSocket clients"""
        if self.connected_clients:
            message = json.dumps(data, default=str)
            disconnected_clients = set()
            
            for client in self.connected_clients:
                try:
                    await client.send(message)
                except:
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            self.connected_clients -= disconnected_clients
    
    def start(self, debug: bool = False):
        """Start the dashboard server"""
        if not DASH_AVAILABLE:
            self.logger.error("Cannot start dashboard - Dash not available")
            return
        
        self.running = True
        self.logger.info(f"Starting dashboard server on {self.host}:{self.port}")
        
        try:
            self.app.run_server(
                host=self.host,
                port=self.port,
                debug=debug,
                use_reloader=False
            )
        except Exception as e:
            self.logger.error(f"Error starting dashboard server: {e}")
            self.running = False
    
    def stop(self):
        """Stop the dashboard server"""
        self.running = False
        self.logger.info("Dashboard server stopped")


# Example usage and testing
if __name__ == "__main__":
    import threading
    import time
    from spacecraft_drmpc.monitoring.telemetry_collector import TelemetryCollector, TelemetryPoint, TelemetryType, SpacecraftState
    from spacecraft_drmpc.monitoring.health_monitor import start_health_monitoring
    from spacecraft_drmpc.monitoring.performance_metrics import start_monitoring
    
    # Create dashboard
    dashboard = MissionControlDashboard("TEST_MISSION")
    
    # Start monitoring systems
    start_health_monitoring()
    start_monitoring()
    
    # Create and connect telemetry collector
    telemetry_collector = TelemetryCollector("TEST_MISSION")
    telemetry_collector.start_collection()
    dashboard.connect_telemetry_collector(telemetry_collector)
    
    # Simulate telemetry data in background
    def simulate_telemetry():
        for i in range(1000):
            # Simulate spacecraft state
            state = SpacecraftState(
                agent_id=f"AGENT_{i % 3 + 1}",
                mission_time=i * 0.1,
                position=np.random.randn(3) * 100,
                velocity=np.random.randn(3) * 5,
                fuel_mass=100 - i * 0.05,
                mission_phase=MissionPhase.CRUISE
            )
            telemetry_collector.collect_spacecraft_state(state)
            dashboard._process_spacecraft_state(state)
            time.sleep(1.0)
    
    # Start simulation thread
    sim_thread = threading.Thread(target=simulate_telemetry, daemon=True)
    sim_thread.start()
    
    # Start dashboard (this blocks)
    dashboard.start(debug=True)