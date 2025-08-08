---
layout: page
title: API Reference
permalink: /pages/api-reference/
---

# API Reference

Complete API reference for the Multi-Agent Spacecraft Docking System, providing programmatic interfaces for mission planning, real-time control, and system monitoring.

## Core Classes

### SpacecraftDRMPCController

Primary controller class implementing Distributionally Robust Model Predictive Control for spacecraft operations.

```python
class SpacecraftDRMPCController:
    def __init__(self, config: DRMPCConfig):
        """Initialize DR-MPC controller with configuration."""
        
    def set_target(self, position: np.ndarray, velocity: np.ndarray = None):
        """Set target position and optional velocity."""
        
    def update_state(self, current_state: SpacecraftState):
        """Update controller with current spacecraft state."""
        
    def compute_control(self) -> ControlCommand:
        """Compute optimal control action."""
        
    def get_prediction_horizon(self) -> TrajectoryPrediction:
        """Return predicted trajectory over control horizon."""
```

**Parameters:**
- `config`: Configuration object containing control parameters
- `position`: Target position as 3D numpy array [x, y, z]
- `velocity`: Optional target velocity [vx, vy, vz]
- `current_state`: Current spacecraft state including position, velocity, attitude

**Returns:**
- `ControlCommand`: Control forces and torques
- `TrajectoryPrediction`: Future state predictions

**Example Usage:**
```python
from spacecraft_drmpc import SpacecraftDRMPCController, DRMPCConfig

config = DRMPCConfig(
    horizon_length=15,
    control_frequency=100.0,
    uncertainty_level=0.3
)

controller = SpacecraftDRMPCController(config)
controller.set_target([100.0, 0.0, 0.0])

# Control loop
while mission_active:
    state = get_spacecraft_state()
    controller.update_state(state)
    control = controller.compute_control()
    apply_control(control)
    time.sleep(0.01)  # 100 Hz
```

### MultiAgentCoordinator

Coordination system for managing multiple spacecraft in formation flying and collaborative missions.

```python
class MultiAgentCoordinator:
    def __init__(self, num_agents: int, communication_graph: NetworkTopology):
        """Initialize multi-agent coordination system."""
        
    def add_spacecraft(self, spacecraft_id: str, controller: SpacecraftDRMPCController):
        """Add spacecraft to coordination system."""
        
    def set_formation(self, formation_type: str, parameters: dict):
        """Configure desired formation geometry."""
        
    def update_all_states(self, states: Dict[str, SpacecraftState]):
        """Update all spacecraft states for coordination."""
        
    def compute_coordinated_controls(self) -> Dict[str, ControlCommand]:
        """Compute coordinated control for all spacecraft."""
        
    def get_formation_error(self) -> FormationMetrics:
        """Return formation tracking errors and metrics."""
```

**Formation Types:**
- `"line"`: Linear formation with configurable spacing
- `"v-formation"`: V-shaped formation for fuel efficiency  
- `"diamond"`: Diamond pattern for maximum coverage
- `"custom"`: User-defined formation from waypoints

**Example Usage:**
```python
from spacecraft_drmpc import MultiAgentCoordinator, NetworkTopology

topology = NetworkTopology.create_fully_connected(num_agents=4)
coordinator = MultiAgentCoordinator(4, topology)

# Add spacecraft controllers
for i, controller in enumerate(controllers):
    coordinator.add_spacecraft(f"craft_{i}", controller)

# Set V-formation with 50m separation
coordinator.set_formation("v-formation", {"separation": 50.0, "angle": 30.0})

# Coordination loop
while formation_active:
    states = get_all_spacecraft_states()
    coordinator.update_all_states(states)
    controls = coordinator.compute_coordinated_controls()
    apply_all_controls(controls)
```

### SafetySystem

Comprehensive safety system providing collision avoidance, fault tolerance, and emergency response capabilities.

```python
class SafetySystem:
    def __init__(self, safety_config: SafetyConfig):
        """Initialize safety system with configuration."""
        
    def register_spacecraft(self, spacecraft_id: str, state_provider: Callable):
        """Register spacecraft for safety monitoring."""
        
    def check_collision_threats(self) -> List[CollisionThreat]:
        """Check for potential collision threats."""
        
    def compute_avoidance_maneuver(self, threat: CollisionThreat) -> ControlCommand:
        """Compute collision avoidance maneuver."""
        
    def detect_faults(self, spacecraft_id: str) -> List[FaultDetection]:
        """Detect system faults for specified spacecraft."""
        
    def execute_emergency_stop(self, spacecraft_id: str):
        """Execute emergency stop procedure."""
```

**Safety Features:**
- Real-time collision detection with configurable safety margins
- Automatic avoidance maneuver generation
- Fault detection covering actuators, sensors, and communication
- Emergency procedures with graceful degradation

**Example Usage:**
```python
from spacecraft_drmpc import SafetySystem, SafetyConfig

safety_config = SafetyConfig(
    collision_radius=15.0,
    approach_speed_limit=0.1,
    fault_threshold=3.0
)

safety = SafetySystem(safety_config)

# Register all spacecraft
for craft_id, state_func in spacecraft_states.items():
    safety.register_spacecraft(craft_id, state_func)

# Safety monitoring loop
while mission_active:
    threats = safety.check_collision_threats()
    if threats:
        for threat in threats:
            avoidance = safety.compute_avoidance_maneuver(threat)
            apply_emergency_control(threat.spacecraft_id, avoidance)
    
    # Check for faults
    for craft_id in spacecraft_ids:
        faults = safety.detect_faults(craft_id)
        if faults:
            handle_faults(craft_id, faults)
```

## Configuration Classes

### DRMPCConfig

Configuration parameters for DR-MPC controller.

```python
@dataclass
class DRMPCConfig:
    horizon_length: int = 15                    # Control horizon steps
    control_frequency: float = 100.0            # Hz, control update rate
    uncertainty_level: float = 0.3              # Model uncertainty tolerance
    wasserstein_radius: float = 0.2             # Ambiguity set radius
    solver_tolerance: float = 1e-6              # Optimization tolerance
    max_iterations: int = 100                   # Solver iteration limit
    position_weight: np.ndarray = None          # State cost weights Q
    control_weight: np.ndarray = None           # Control cost weights R
    constraint_margin: float = 0.1              # Constraint tightening
```

### SafetyConfig

Safety system configuration parameters.

```python
@dataclass
class SafetyConfig:
    collision_radius: float = 10.0              # Minimum separation distance
    approach_speed_limit: float = 0.1           # Maximum approach velocity
    emergency_deceleration: float = 2.0         # Emergency stop acceleration
    fault_detection_threshold: float = 3.0      # Fault detection sigma
    monitoring_frequency: float = 200.0         # Safety check frequency
    threat_prediction_horizon: float = 30.0     # Threat prediction time
```

## Data Structures

### SpacecraftState

Complete spacecraft state representation.

```python
@dataclass
class SpacecraftState:
    timestamp: float                            # State timestamp
    position: np.ndarray                        # [x, y, z] position
    velocity: np.ndarray                        # [vx, vy, vz] velocity
    attitude: Quaternion                        # Attitude quaternion
    angular_velocity: np.ndarray                # [wx, wy, wz] angular rates
    mass: float                                 # Spacecraft mass
    actuator_status: Dict[str, bool]            # Actuator health status
    sensor_status: Dict[str, bool]              # Sensor health status
```

### ControlCommand

Control command output from controllers.

```python
@dataclass
class ControlCommand:
    timestamp: float                            # Command timestamp
    forces: np.ndarray                          # [fx, fy, fz] control forces
    torques: np.ndarray                         # [tx, ty, tz] control torques
    thruster_commands: Dict[str, float]         # Individual thruster commands
    validity_duration: float                    # Command validity period
    priority: int                               # Command priority level
```

### CollisionThreat

Collision threat detection result.

```python
@dataclass
class CollisionThreat:
    threat_id: str                              # Unique threat identifier
    primary_spacecraft: str                     # First spacecraft ID
    secondary_spacecraft: str                   # Second spacecraft ID
    time_to_collision: float                    # Predicted collision time
    closest_approach_distance: float            # Minimum separation
    threat_level: ThreatLevel                   # LOW, MEDIUM, HIGH, CRITICAL
    recommended_action: str                     # Suggested response
```

## Utility Functions

### Mission Planning Utilities

```python
def create_mission_plan(waypoints: List[np.ndarray], 
                       timing: List[float]) -> MissionPlan:
    """Create mission plan from waypoints and timing."""

def validate_trajectory(trajectory: TrajectoryPrediction, 
                       constraints: Constraints) -> ValidationResult:
    """Validate trajectory against operational constraints."""

def optimize_fuel_consumption(mission_plan: MissionPlan) -> MissionPlan:
    """Optimize mission plan for minimum fuel consumption."""
```

### Analysis and Visualization

```python
def analyze_performance(mission_log: MissionLog) -> PerformanceReport:
    """Analyze mission performance from log data."""

def plot_trajectory(trajectory: TrajectoryPrediction, 
                   save_path: str = None) -> matplotlib.Figure:
    """Plot 3D trajectory visualization."""

def generate_formation_animation(formation_history: List[FormationState]) -> Animation:
    """Generate animation of formation flying maneuvers."""
```

## Real-Time Interfaces

### State Providers

```python
class StateProvider:
    """Abstract base class for state providers."""
    
    def get_current_state(self, spacecraft_id: str) -> SpacecraftState:
        """Get current spacecraft state."""
        raise NotImplementedError

class SimulationStateProvider(StateProvider):
    """State provider for simulation environments."""
    
class HardwareStateProvider(StateProvider):
    """State provider for real hardware interfaces."""
```

### Control Interfaces

```python
class ControlInterface:
    """Abstract base class for control interfaces."""
    
    def apply_control(self, spacecraft_id: str, command: ControlCommand):
        """Apply control command to spacecraft."""
        raise NotImplementedError

class SimulationControlInterface(ControlInterface):
    """Control interface for simulation environments."""
    
class HardwareControlInterface(ControlInterface):
    """Control interface for real hardware systems."""
```

## Event Handling

### Event System

```python
class EventHandler:
    """Event handling system for mission monitoring."""
    
    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for specific event type."""
        
    def emit_event(self, event: Event):
        """Emit event to registered handlers."""

# Event types
class EventTypes:
    MISSION_START = "mission_start"
    TARGET_REACHED = "target_reached"
    COLLISION_THREAT = "collision_threat"
    FAULT_DETECTED = "fault_detected"
    FORMATION_COMPLETE = "formation_complete"
    EMERGENCY_STOP = "emergency_stop"
```

### Custom Event Example

```python
def handle_collision_threat(event: Event):
    """Handle collision threat events."""
    threat = event.data
    logger.warning(f"Collision threat detected: {threat}")
    # Implement custom threat response

event_handler = EventHandler()
event_handler.register_callback(EventTypes.COLLISION_THREAT, handle_collision_threat)
```

## Error Handling

### Exception Classes

```python
class SpacecraftDRMPCException(Exception):
    """Base exception for spacecraft system."""

class ControllerException(SpacecraftDRMPCException):
    """Controller-related exceptions."""

class SafetyException(SpacecraftDRMPCException):
    """Safety system exceptions."""

class CommunicationException(SpacecraftDRMPCException):
    """Communication system exceptions."""

class OptimizationException(SpacecraftDRMPCException):
    """Optimization solver exceptions."""
```

## Performance Monitoring

### Metrics Collection

```python
class PerformanceMonitor:
    """Real-time performance monitoring system."""
    
    def start_monitoring(self):
        """Start performance data collection."""
        
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        
    def log_execution_time(self, function_name: str, execution_time: float):
        """Log function execution time."""

# Usage with decorator
@PerformanceMonitor.time_it
def compute_control():
    # Controller computation
    pass
```

---

This API reference provides complete documentation for programmatic interaction with the Multi-Agent Spacecraft Docking System. For implementation examples and tutorials, see the technical documentation and example code repository.

**API Version**: 2.1  
**Last Updated**: December 2024  
**Compatibility**: Python 3.9+