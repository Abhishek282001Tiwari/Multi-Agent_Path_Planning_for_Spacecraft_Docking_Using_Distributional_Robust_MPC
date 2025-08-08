"""
Mission Data Collection and Telemetry Analysis System

This module provides comprehensive telemetry collection capabilities for spacecraft
missions, including real-time data aggregation, mission analysis, and reporting.
"""

import time
import threading
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, NamedTuple
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
from enum import Enum
from pathlib import Path
import json
import h5py
import sqlite3
import pickle
import zlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import weakref
import uuid

# Scientific and analysis libraries
import scipy.stats as stats
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

from .system_logger import get_component_logger, ComponentType
from .performance_metrics import PerformanceAlert, AlertLevel


class TelemetryType(Enum):
    """Types of telemetry data"""
    POSITION = "position"
    VELOCITY = "velocity"
    ACCELERATION = "acceleration"
    ATTITUDE = "attitude"
    ANGULAR_VELOCITY = "angular_velocity"
    THRUST = "thrust"
    TORQUE = "torque"
    FUEL = "fuel"
    POWER = "power"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    SENSOR_DATA = "sensor_data"
    CONTROL_INPUT = "control_input"
    SYSTEM_STATE = "system_state"
    MISSION_EVENT = "mission_event"
    ERROR_STATE = "error_state"
    PERFORMANCE = "performance"


class DataQuality(Enum):
    """Data quality indicators"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    INVALID = "invalid"


class MissionPhase(Enum):
    """Mission phase enumeration"""
    INITIALIZATION = "initialization"
    LAUNCH = "launch"
    CRUISE = "cruise"
    APPROACH = "approach"
    PROXIMITY = "proximity"
    DOCKING = "docking"
    ATTACHED = "attached"
    UNDOCKING = "undocking"
    DEPARTURE = "departure"
    EMERGENCY = "emergency"
    COMPLETED = "completed"
    ABORTED = "aborted"


@dataclass
class TelemetryPoint:
    """Single telemetry data point"""
    timestamp: datetime = field(default_factory=datetime.now)
    agent_id: str = ""
    telemetry_type: TelemetryType = TelemetryType.SYSTEM_STATE
    data: Union[float, List[float], Dict[str, Any]] = field(default_factory=dict)
    units: str = ""
    quality: DataQuality = DataQuality.GOOD
    source: str = ""
    mission_time: float = 0.0
    simulation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpacecraftState:
    """Complete spacecraft state at a given time"""
    timestamp: datetime = field(default_factory=datetime.now)
    agent_id: str = ""
    mission_time: float = 0.0
    
    # Orbital state
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [x, y, z] in meters
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [vx, vy, vz] in m/s
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [ax, ay, az] in m/s²
    
    # Attitude state
    attitude_quaternion: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))  # [w, x, y, z]
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [wx, wy, wz] in rad/s
    angular_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [alpha_x, alpha_y, alpha_z] in rad/s²
    
    # Control inputs
    thrust_vector: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [Fx, Fy, Fz] in N
    torque_vector: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [Mx, My, Mz] in N⋅m
    
    # System status
    fuel_mass: float = 0.0  # kg
    power_level: float = 100.0  # percentage
    temperature: float = 293.15  # Kelvin
    system_health: float = 100.0  # percentage
    
    # Mission-specific
    mission_phase: MissionPhase = MissionPhase.INITIALIZATION
    target_distance: float = float('inf')  # meters
    relative_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m/s
    
    # Quality indicators
    position_accuracy: float = 0.0  # meters (estimated error)
    attitude_accuracy: float = 0.0  # radians (estimated error)
    data_quality: DataQuality = DataQuality.GOOD


@dataclass
class MissionSummary:
    """Mission summary statistics and analysis"""
    mission_id: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration: float = 0.0  # seconds
    
    # Mission metrics
    total_agents: int = 0
    successful_dockings: int = 0
    failed_dockings: int = 0
    emergency_events: int = 0
    collision_warnings: int = 0
    
    # Performance metrics
    total_fuel_consumption: float = 0.0  # kg
    average_position_error: float = 0.0  # meters
    maximum_position_error: float = 0.0  # meters
    average_attitude_error: float = 0.0  # radians
    maximum_attitude_error: float = 0.0  # radians
    
    # Efficiency metrics
    fuel_efficiency: float = 0.0  # m/kg
    time_efficiency: float = 100.0  # percentage
    success_rate: float = 100.0  # percentage
    
    # Data quality
    data_completeness: float = 100.0  # percentage
    data_quality_score: float = 100.0  # overall quality score
    
    # Additional statistics
    statistics: Dict[str, Any] = field(default_factory=dict)


class TelemetryCollector:
    """Main telemetry collection and management system"""
    
    def __init__(self, mission_id: str, buffer_size: int = 100000):
        self.mission_id = mission_id
        self.buffer_size = buffer_size
        self.logger = get_component_logger(ComponentType.MONITORING, f"telemetry.{mission_id}")
        
        # Data storage
        self.telemetry_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=buffer_size))
        self.spacecraft_states: Dict[str, deque] = defaultdict(lambda: deque(maxlen=buffer_size))
        self.mission_events: deque = deque(maxlen=10000)
        
        # Real-time processing
        self.processors: List[Callable] = []
        self.analyzers: List[Callable] = []
        self.data_queue = queue.Queue()
        self.running = False
        self.worker_threads: List[threading.Thread] = []
        
        # Mission tracking
        self.mission_start_time = datetime.now()
        self.active_agents: set = set()
        self.mission_statistics = defaultdict(float)
        
        # Data export
        self.export_formats = ['hdf5', 'csv', 'json', 'sqlite']
        self.export_lock = threading.Lock()
        
        # Anomaly detection
        self.anomaly_detectors: Dict[str, Any] = {}
        self.anomaly_threshold = 2.0  # Standard deviations
        
        # Data quality monitoring
        self.quality_metrics = defaultdict(list)
        self.quality_thresholds = {
            'position_accuracy': 1.0,  # meters
            'attitude_accuracy': 0.1,  # radians
            'data_freshness': 5.0,  # seconds
            'completeness': 95.0  # percentage
        }
    
    def register_processor(self, processor: Callable[[TelemetryPoint], None]):
        """Register a real-time telemetry processor"""
        self.processors.append(processor)
        self.logger.info(f"Registered telemetry processor: {processor.__name__}")
    
    def register_analyzer(self, analyzer: Callable[[List[TelemetryPoint]], Any]):
        """Register a batch telemetry analyzer"""
        self.analyzers.append(analyzer)
        self.logger.info(f"Registered telemetry analyzer: {analyzer.__name__}")
    
    def start_collection(self):
        """Start telemetry collection"""
        if self.running:
            return
        
        self.running = True
        self.mission_start_time = datetime.now()
        
        # Start worker threads
        self.worker_threads = [
            threading.Thread(target=self._processing_loop, daemon=True),
            threading.Thread(target=self._analysis_loop, daemon=True),
            threading.Thread(target=self._quality_monitoring_loop, daemon=True),
            threading.Thread(target=self._anomaly_detection_loop, daemon=True)
        ]
        
        for thread in self.worker_threads:
            thread.start()
        
        self.logger.info(f"Started telemetry collection for mission {self.mission_id}")
    
    def stop_collection(self):
        """Stop telemetry collection"""
        self.running = False
        
        # Wait for worker threads to finish
        for thread in self.worker_threads:
            thread.join(timeout=5.0)
        
        self.logger.info(f"Stopped telemetry collection for mission {self.mission_id}")
    
    def collect_telemetry(self, telemetry: TelemetryPoint):
        """Collect a single telemetry point"""
        # Validate telemetry data
        if not self._validate_telemetry(telemetry):
            self.logger.warning(f"Invalid telemetry data received from {telemetry.agent_id}")
            return
        
        # Add to buffer
        self.telemetry_buffer[telemetry.agent_id].append(telemetry)
        self.active_agents.add(telemetry.agent_id)
        
        # Queue for processing
        self.data_queue.put(telemetry)
        
        # Update statistics
        self.mission_statistics['total_points'] += 1
        self.mission_statistics['last_update'] = time.time()
    
    def collect_spacecraft_state(self, state: SpacecraftState):
        """Collect complete spacecraft state"""
        # Validate state data
        if not self._validate_spacecraft_state(state):
            self.logger.warning(f"Invalid spacecraft state received from {state.agent_id}")
            return
        
        # Add to buffer
        self.spacecraft_states[state.agent_id].append(state)
        self.active_agents.add(state.agent_id)
        
        # Convert to telemetry points and queue
        telemetry_points = self._state_to_telemetry(state)
        for point in telemetry_points:
            self.data_queue.put(point)
        
        # Update mission tracking
        self._update_mission_tracking(state)
    
    def _validate_telemetry(self, telemetry: TelemetryPoint) -> bool:
        """Validate telemetry data"""
        if not telemetry.agent_id:
            return False
        
        if isinstance(telemetry.data, (list, np.ndarray)):
            if any(not np.isfinite(x) for x in np.asarray(telemetry.data).flatten()):
                telemetry.quality = DataQuality.INVALID
                return False
        elif isinstance(telemetry.data, (int, float)):
            if not np.isfinite(telemetry.data):
                telemetry.quality = DataQuality.INVALID
                return False
        
        return True
    
    def _validate_spacecraft_state(self, state: SpacecraftState) -> bool:
        """Validate spacecraft state data"""
        if not state.agent_id:
            return False
        
        # Check for NaN or infinite values
        arrays_to_check = [
            state.position, state.velocity, state.acceleration,
            state.attitude_quaternion, state.angular_velocity, state.angular_acceleration,
            state.thrust_vector, state.torque_vector, state.relative_velocity
        ]
        
        for arr in arrays_to_check:
            if not np.all(np.isfinite(arr)):
                state.data_quality = DataQuality.INVALID
                return False
        
        # Check quaternion normalization
        if not np.allclose(np.linalg.norm(state.attitude_quaternion), 1.0, rtol=1e-3):
            state.data_quality = DataQuality.POOR
        
        return True
    
    def _state_to_telemetry(self, state: SpacecraftState) -> List[TelemetryPoint]:
        """Convert spacecraft state to telemetry points"""
        points = []
        
        # Position telemetry
        points.append(TelemetryPoint(
            timestamp=state.timestamp,
            agent_id=state.agent_id,
            telemetry_type=TelemetryType.POSITION,
            data=state.position.tolist(),
            units="meters",
            quality=state.data_quality,
            mission_time=state.mission_time
        ))
        
        # Velocity telemetry
        points.append(TelemetryPoint(
            timestamp=state.timestamp,
            agent_id=state.agent_id,
            telemetry_type=TelemetryType.VELOCITY,
            data=state.velocity.tolist(),
            units="m/s",
            quality=state.data_quality,
            mission_time=state.mission_time
        ))
        
        # Attitude telemetry
        points.append(TelemetryPoint(
            timestamp=state.timestamp,
            agent_id=state.agent_id,
            telemetry_type=TelemetryType.ATTITUDE,
            data=state.attitude_quaternion.tolist(),
            units="quaternion",
            quality=state.data_quality,
            mission_time=state.mission_time
        ))
        
        # Fuel telemetry
        points.append(TelemetryPoint(
            timestamp=state.timestamp,
            agent_id=state.agent_id,
            telemetry_type=TelemetryType.FUEL,
            data=state.fuel_mass,
            units="kg",
            quality=state.data_quality,
            mission_time=state.mission_time
        ))
        
        # Power telemetry
        points.append(TelemetryPoint(
            timestamp=state.timestamp,
            agent_id=state.agent_id,
            telemetry_type=TelemetryType.POWER,
            data=state.power_level,
            units="percent",
            quality=state.data_quality,
            mission_time=state.mission_time
        ))
        
        return points
    
    def _update_mission_tracking(self, state: SpacecraftState):
        """Update mission tracking metrics"""
        agent_id = state.agent_id
        
        # Track mission phases
        if state.mission_phase == MissionPhase.DOCKING:
            if state.target_distance < 1.0:  # Successfully docked
                self.mission_statistics[f'{agent_id}_successful_dockings'] += 1
        
        # Track fuel consumption
        if f'{agent_id}_initial_fuel' not in self.mission_statistics:
            self.mission_statistics[f'{agent_id}_initial_fuel'] = state.fuel_mass
        
        fuel_consumed = (self.mission_statistics[f'{agent_id}_initial_fuel'] - state.fuel_mass)
        self.mission_statistics[f'{agent_id}_fuel_consumed'] = fuel_consumed
        
        # Track position accuracy
        if state.position_accuracy > 0:
            self.mission_statistics[f'{agent_id}_max_position_error'] = max(
                self.mission_statistics.get(f'{agent_id}_max_position_error', 0),
                state.position_accuracy
            )
    
    def _processing_loop(self):
        """Main processing loop for real-time telemetry"""
        while self.running:
            try:
                telemetry = self.data_queue.get(timeout=1.0)
                
                # Apply processors
                for processor in self.processors:
                    try:
                        processor(telemetry)
                    except Exception as e:
                        self.logger.error(f"Error in telemetry processor {processor.__name__}: {e}")
                
                self.data_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
    
    def _analysis_loop(self):
        """Background analysis loop"""
        while self.running:
            try:
                time.sleep(10.0)  # Run analysis every 10 seconds
                
                for agent_id in list(self.active_agents):
                    # Get recent telemetry data
                    recent_data = list(self.telemetry_buffer[agent_id])[-1000:]  # Last 1000 points
                    
                    if len(recent_data) < 10:
                        continue
                    
                    # Apply analyzers
                    for analyzer in self.analyzers:
                        try:
                            result = analyzer(recent_data)
                            if result:
                                self.logger.info(f"Analysis result from {analyzer.__name__}: {result}")
                        except Exception as e:
                            self.logger.error(f"Error in analyzer {analyzer.__name__}: {e}")
                
            except Exception as e:
                self.logger.error(f"Error in analysis loop: {e}")
    
    def _quality_monitoring_loop(self):
        """Data quality monitoring loop"""
        while self.running:
            try:
                time.sleep(30.0)  # Check quality every 30 seconds
                
                for agent_id in list(self.active_agents):
                    quality_metrics = self._calculate_quality_metrics(agent_id)
                    self.quality_metrics[agent_id].append(quality_metrics)
                    
                    # Check quality thresholds
                    self._check_quality_thresholds(agent_id, quality_metrics)
                
            except Exception as e:
                self.logger.error(f"Error in quality monitoring: {e}")
    
    def _anomaly_detection_loop(self):
        """Anomaly detection loop"""
        while self.running:
            try:
                time.sleep(60.0)  # Run anomaly detection every minute
                
                for agent_id in list(self.active_agents):
                    self._detect_anomalies(agent_id)
                
            except Exception as e:
                self.logger.error(f"Error in anomaly detection: {e}")
    
    def _calculate_quality_metrics(self, agent_id: str) -> Dict[str, float]:
        """Calculate data quality metrics for an agent"""
        recent_states = list(self.spacecraft_states[agent_id])[-100:]  # Last 100 states
        
        if not recent_states:
            return {}
        
        # Data freshness
        latest_time = recent_states[-1].timestamp
        data_freshness = (datetime.now() - latest_time).total_seconds()
        
        # Position accuracy
        position_accuracies = [s.position_accuracy for s in recent_states if s.position_accuracy > 0]
        avg_position_accuracy = np.mean(position_accuracies) if position_accuracies else 0
        
        # Attitude accuracy
        attitude_accuracies = [s.attitude_accuracy for s in recent_states if s.attitude_accuracy > 0]
        avg_attitude_accuracy = np.mean(attitude_accuracies) if attitude_accuracies else 0
        
        # Data completeness (percentage of expected data points)
        expected_points = len(recent_states) * 5  # 5 telemetry types per state
        actual_points = len([tp for tp in self.telemetry_buffer[agent_id] 
                            if tp.timestamp >= recent_states[0].timestamp])
        completeness = (actual_points / expected_points * 100) if expected_points > 0 else 0
        
        return {
            'data_freshness': data_freshness,
            'position_accuracy': avg_position_accuracy,
            'attitude_accuracy': avg_attitude_accuracy,
            'completeness': completeness
        }
    
    def _check_quality_thresholds(self, agent_id: str, metrics: Dict[str, float]):
        """Check data quality against thresholds"""
        for metric_name, value in metrics.items():
            if metric_name in self.quality_thresholds:
                threshold = self.quality_thresholds[metric_name]
                
                # Different logic for different metrics
                if metric_name == 'completeness' and value < threshold:
                    self.logger.warning(f"Low data completeness for {agent_id}: {value:.1f}%")
                elif metric_name in ['position_accuracy', 'attitude_accuracy', 'data_freshness'] and value > threshold:
                    self.logger.warning(f"Poor {metric_name} for {agent_id}: {value:.3f}")
    
    def _detect_anomalies(self, agent_id: str):
        """Detect anomalies in telemetry data"""
        recent_states = list(self.spacecraft_states[agent_id])[-1000:]  # Last 1000 states
        
        if len(recent_states) < 50:
            return
        
        # Extract numerical features for anomaly detection
        features = []
        for state in recent_states:
            feature_vector = np.concatenate([
                state.position,
                state.velocity,
                [state.fuel_mass, state.power_level, state.temperature, state.system_health]
            ])
            features.append(feature_vector)
        
        features = np.array(features)
        
        # Use Isolation Forest for anomaly detection
        if agent_id not in self.anomaly_detectors:
            self.anomaly_detectors[agent_id] = IsolationForest(
                contamination=0.1,  # Expect 10% anomalies
                random_state=42
            )
        
        detector = self.anomaly_detectors[agent_id]
        
        try:
            # Fit and predict
            anomaly_scores = detector.fit_predict(features)
            
            # Check for recent anomalies
            recent_anomalies = anomaly_scores[-10:]  # Last 10 points
            if np.any(recent_anomalies == -1):  # -1 indicates anomaly
                anomaly_count = np.sum(recent_anomalies == -1)
                self.logger.warning(
                    f"Anomalies detected for {agent_id}: {anomaly_count}/10 recent points",
                    agent_id=agent_id,
                    anomaly_count=int(anomaly_count)
                )
        
        except Exception as e:
            self.logger.error(f"Error in anomaly detection for {agent_id}: {e}")
    
    def get_mission_summary(self) -> MissionSummary:
        """Generate mission summary with statistics"""
        current_time = datetime.now()
        duration = (current_time - self.mission_start_time).total_seconds()
        
        # Calculate aggregate statistics
        total_fuel_consumption = sum(
            self.mission_statistics.get(f'{agent_id}_fuel_consumed', 0)
            for agent_id in self.active_agents
        )
        
        # Calculate position errors
        position_errors = []
        for agent_id in self.active_agents:
            states = list(self.spacecraft_states[agent_id])
            errors = [s.position_accuracy for s in states if s.position_accuracy > 0]
            position_errors.extend(errors)
        
        avg_position_error = np.mean(position_errors) if position_errors else 0
        max_position_error = np.max(position_errors) if position_errors else 0
        
        # Calculate data quality
        total_points = self.mission_statistics.get('total_points', 0)
        quality_scores = []
        for agent_metrics in self.quality_metrics.values():
            if agent_metrics:
                completeness_scores = [m.get('completeness', 100) for m in agent_metrics]
                quality_scores.extend(completeness_scores)
        
        data_quality_score = np.mean(quality_scores) if quality_scores else 100
        
        summary = MissionSummary(
            mission_id=self.mission_id,
            start_time=self.mission_start_time,
            end_time=current_time if not self.running else None,
            duration=duration,
            total_agents=len(self.active_agents),
            total_fuel_consumption=total_fuel_consumption,
            average_position_error=avg_position_error,
            maximum_position_error=max_position_error,
            data_quality_score=data_quality_score,
            statistics={
                'total_telemetry_points': total_points,
                'active_agents': len(self.active_agents),
                'mission_statistics': dict(self.mission_statistics)
            }
        )
        
        return summary
    
    def export_data(self, output_dir: Path, format: str = 'hdf5', 
                   agent_ids: Optional[List[str]] = None):
        """Export telemetry data to various formats"""
        with self.export_lock:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            agents_to_export = agent_ids or list(self.active_agents)
            
            if format == 'hdf5':
                self._export_hdf5(output_dir, agents_to_export)
            elif format == 'csv':
                self._export_csv(output_dir, agents_to_export)
            elif format == 'json':
                self._export_json(output_dir, agents_to_export)
            elif format == 'sqlite':
                self._export_sqlite(output_dir, agents_to_export)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            self.logger.info(f"Exported telemetry data to {output_dir} in {format} format")
    
    def _export_hdf5(self, output_dir: Path, agent_ids: List[str]):
        """Export data to HDF5 format"""
        output_file = output_dir / f"{self.mission_id}_telemetry.h5"
        
        with h5py.File(output_file, 'w') as f:
            # Mission metadata
            mission_group = f.create_group('mission')
            mission_group.attrs['mission_id'] = self.mission_id
            mission_group.attrs['start_time'] = self.mission_start_time.isoformat()
            mission_group.attrs['export_time'] = datetime.now().isoformat()
            
            # Telemetry data for each agent
            for agent_id in agent_ids:
                agent_group = f.create_group(f'agents/{agent_id}')
                
                # Spacecraft states
                states = list(self.spacecraft_states[agent_id])
                if states:
                    states_group = agent_group.create_group('states')
                    
                    # Time series data
                    times = [s.mission_time for s in states]
                    positions = [s.position for s in states]
                    velocities = [s.velocity for s in states]
                    
                    states_group.create_dataset('time', data=times)
                    states_group.create_dataset('position', data=positions)
                    states_group.create_dataset('velocity', data=velocities)
                
                # Raw telemetry
                telemetry = list(self.telemetry_buffer[agent_id])
                if telemetry:
                    telemetry_group = agent_group.create_group('telemetry')
                    
                    # Group by telemetry type
                    telemetry_by_type = defaultdict(list)
                    for tp in telemetry:
                        telemetry_by_type[tp.telemetry_type.value].append(tp)
                    
                    for tel_type, tel_points in telemetry_by_type.items():
                        type_group = telemetry_group.create_group(tel_type)
                        times = [tp.mission_time for tp in tel_points]
                        data = [tp.data if isinstance(tp.data, (int, float)) 
                               else tp.data for tp in tel_points]
                        
                        type_group.create_dataset('time', data=times)
                        try:
                            type_group.create_dataset('data', data=data)
                        except (TypeError, ValueError):
                            # Handle mixed data types
                            serialized_data = [json.dumps(d) if not isinstance(d, (int, float)) 
                                             else d for d in data]
                            type_group.create_dataset('data', data=serialized_data)
    
    def _export_csv(self, output_dir: Path, agent_ids: List[str]):
        """Export data to CSV format"""
        for agent_id in agent_ids:
            # Export spacecraft states
            states = list(self.spacecraft_states[agent_id])
            if states:
                states_df = pd.DataFrame([
                    {
                        'timestamp': s.timestamp,
                        'mission_time': s.mission_time,
                        'pos_x': s.position[0],
                        'pos_y': s.position[1],
                        'pos_z': s.position[2],
                        'vel_x': s.velocity[0],
                        'vel_y': s.velocity[1],
                        'vel_z': s.velocity[2],
                        'fuel_mass': s.fuel_mass,
                        'power_level': s.power_level,
                        'mission_phase': s.mission_phase.value
                    }
                    for s in states
                ])
                
                states_file = output_dir / f"{self.mission_id}_{agent_id}_states.csv"
                states_df.to_csv(states_file, index=False)
    
    def _export_json(self, output_dir: Path, agent_ids: List[str]):
        """Export data to JSON format"""
        for agent_id in agent_ids:
            agent_data = {
                'agent_id': agent_id,
                'mission_id': self.mission_id,
                'spacecraft_states': [
                    {
                        'timestamp': s.timestamp.isoformat(),
                        'mission_time': s.mission_time,
                        'position': s.position.tolist(),
                        'velocity': s.velocity.tolist(),
                        'attitude_quaternion': s.attitude_quaternion.tolist(),
                        'fuel_mass': s.fuel_mass,
                        'power_level': s.power_level,
                        'mission_phase': s.mission_phase.value
                    }
                    for s in list(self.spacecraft_states[agent_id])
                ],
                'telemetry': [
                    {
                        'timestamp': tp.timestamp.isoformat(),
                        'telemetry_type': tp.telemetry_type.value,
                        'data': tp.data,
                        'units': tp.units,
                        'quality': tp.quality.value
                    }
                    for tp in list(self.telemetry_buffer[agent_id])
                ]
            }
            
            output_file = output_dir / f"{self.mission_id}_{agent_id}.json"
            with open(output_file, 'w') as f:
                json.dump(agent_data, f, indent=2, default=str)
    
    def _export_sqlite(self, output_dir: Path, agent_ids: List[str]):
        """Export data to SQLite database"""
        db_file = output_dir / f"{self.mission_id}_telemetry.db"
        
        with sqlite3.connect(db_file) as conn:
            # Create tables
            conn.execute('''
                CREATE TABLE IF NOT EXISTS spacecraft_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT,
                    timestamp TEXT,
                    mission_time REAL,
                    pos_x REAL, pos_y REAL, pos_z REAL,
                    vel_x REAL, vel_y REAL, vel_z REAL,
                    fuel_mass REAL,
                    power_level REAL,
                    mission_phase TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT,
                    timestamp TEXT,
                    telemetry_type TEXT,
                    data TEXT,
                    units TEXT,
                    quality TEXT
                )
            ''')
            
            # Insert data
            for agent_id in agent_ids:
                # Insert states
                states = list(self.spacecraft_states[agent_id])
                for state in states:
                    conn.execute('''
                        INSERT INTO spacecraft_states 
                        (agent_id, timestamp, mission_time, pos_x, pos_y, pos_z,
                         vel_x, vel_y, vel_z, fuel_mass, power_level, mission_phase)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        agent_id, state.timestamp.isoformat(), state.mission_time,
                        state.position[0], state.position[1], state.position[2],
                        state.velocity[0], state.velocity[1], state.velocity[2],
                        state.fuel_mass, state.power_level, state.mission_phase.value
                    ))
                
                # Insert telemetry
                telemetry_points = list(self.telemetry_buffer[agent_id])
                for tp in telemetry_points:
                    conn.execute('''
                        INSERT INTO telemetry 
                        (agent_id, timestamp, telemetry_type, data, units, quality)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        agent_id, tp.timestamp.isoformat(), tp.telemetry_type.value,
                        json.dumps(tp.data), tp.units, tp.quality.value
                    ))


# Example telemetry processors and analyzers
def trajectory_analyzer(telemetry_points: List[TelemetryPoint]) -> Dict[str, Any]:
    """Analyze trajectory smoothness and efficiency"""
    position_points = [tp for tp in telemetry_points if tp.telemetry_type == TelemetryType.POSITION]
    
    if len(position_points) < 3:
        return {}
    
    # Calculate trajectory statistics
    positions = np.array([tp.data for tp in position_points])
    times = np.array([tp.mission_time for tp in position_points])
    
    # Velocity estimation
    dt = np.diff(times)
    velocities = np.diff(positions, axis=0) / dt[:, np.newaxis]
    
    # Smoothness metric (acceleration variance)
    if len(velocities) > 1:
        accelerations = np.diff(velocities, axis=0) / dt[1:][:, np.newaxis]
        smoothness = np.var(np.linalg.norm(accelerations, axis=1))
    else:
        smoothness = 0
    
    return {
        'trajectory_smoothness': smoothness,
        'total_distance': np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)),
        'average_speed': np.mean(np.linalg.norm(velocities, axis=1))
    }


def fuel_efficiency_analyzer(telemetry_points: List[TelemetryPoint]) -> Dict[str, Any]:
    """Analyze fuel efficiency"""
    fuel_points = [tp for tp in telemetry_points if tp.telemetry_type == TelemetryType.FUEL]
    thrust_points = [tp for tp in telemetry_points if tp.telemetry_type == TelemetryType.THRUST]
    
    if not fuel_points or not thrust_points:
        return {}
    
    # Calculate fuel consumption rate
    fuel_data = [(tp.mission_time, tp.data) for tp in fuel_points]
    fuel_data.sort(key=lambda x: x[0])
    
    if len(fuel_data) < 2:
        return {}
    
    fuel_consumption = fuel_data[0][1] - fuel_data[-1][1]  # Initial - final
    time_span = fuel_data[-1][0] - fuel_data[0][0]
    consumption_rate = fuel_consumption / time_span if time_span > 0 else 0
    
    # Calculate specific impulse estimate
    thrust_magnitudes = [np.linalg.norm(tp.data) if isinstance(tp.data, list) else tp.data 
                        for tp in thrust_points]
    average_thrust = np.mean(thrust_magnitudes)
    
    specific_impulse = average_thrust / consumption_rate if consumption_rate > 0 else 0
    
    return {
        'fuel_consumption_rate': consumption_rate,
        'total_fuel_consumed': fuel_consumption,
        'average_thrust': average_thrust,
        'specific_impulse_estimate': specific_impulse
    }


# Example usage
if __name__ == "__main__":
    # Create telemetry collector
    collector = TelemetryCollector("TEST_MISSION_001")
    
    # Register analyzers
    collector.register_analyzer(trajectory_analyzer)
    collector.register_analyzer(fuel_efficiency_analyzer)
    
    # Start collection
    collector.start_collection()
    
    # Simulate telemetry data
    for i in range(100):
        # Simulate spacecraft state
        state = SpacecraftState(
            agent_id="TEST_AGENT_001",
            mission_time=i * 0.1,
            position=np.random.randn(3) * 10,
            velocity=np.random.randn(3),
            fuel_mass=100 - i * 0.1,
            mission_phase=MissionPhase.CRUISE
        )
        collector.collect_spacecraft_state(state)
        
        time.sleep(0.01)
    
    # Get mission summary
    summary = collector.get_mission_summary()
    print("Mission Summary:")
    print(f"Duration: {summary.duration:.2f} seconds")
    print(f"Total agents: {summary.total_agents}")
    print(f"Fuel consumption: {summary.total_fuel_consumption:.2f} kg")
    
    # Export data
    collector.export_data(Path("telemetry_export"), format='json')
    
    # Stop collection
    collector.stop_collection()