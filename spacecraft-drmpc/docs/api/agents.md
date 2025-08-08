# Spacecraft Agent API Documentation

This document provides comprehensive documentation for all spacecraft agent classes and their methods.

## Table of Contents
- [SpacecraftAgent (Base Class)](#spacecraftagent-base-class)
- [AdvancedSpacecraftAgent](#advancedspacecraftagent)
- [FaultTolerantSpacecraft](#faulttolerantspacecraft)
- [SecureSpacecraftAgent](#securespacecraftagent)
- [Code Examples](#code-examples)

---

## SpacecraftAgent (Base Class)

The base class for all spacecraft agents in the multi-agent system. Provides core functionality for state management, control, and communication.

### Class Definition
```python
class SpacecraftAgent(ABC):
    """
    Base class for all spacecraft agents in the multi-agent system
    
    Attributes:
        agent_id (str): Unique identifier for the agent
        state (np.ndarray): Current 13-element state vector [position, velocity, quaternion, angular_velocity]
        target_state (np.ndarray): Target 13-element state vector
        is_active (bool): Agent activity status
        mission_phase (str): Current mission phase
        performance_metrics (dict): Performance tracking data
    """
```

### Constructor
```python
def __init__(self, agent_id: str, config: Optional[Dict] = None)
```

**Parameters:**
- `agent_id` (str): Unique identifier for the spacecraft agent
- `config` (Optional[Dict]): Configuration dictionary with agent parameters

**Returns:** None

**Example:**
```python
agent = SpacecraftAgent('chaser-001', {'max_thrust': 10.0})
```

### Abstract Methods

#### update_control
```python
async def update_control(self, dt: float) -> np.ndarray
```
**Description:** Compute control inputs for the spacecraft (must be implemented by subclasses)

**Parameters:**
- `dt` (float): Time step in seconds

**Returns:** 
- `np.ndarray`: 6-element control vector [thrust_x, thrust_y, thrust_z, torque_x, torque_y, torque_z]

#### communicate
```python
async def communicate(self, message: Dict, target_agent: Optional[str] = None)
```
**Description:** Send message to other agents (must be implemented by subclasses)

**Parameters:**
- `message` (Dict): Message content to send
- `target_agent` (Optional[str]): Target agent ID (None for broadcast)

**Returns:** None

### State Management Methods

#### update_state
```python
def update_state(self, new_state: np.ndarray) -> None
```
**Description:** Update the agent's current state vector

**Parameters:**
- `new_state` (np.ndarray): 13-element state vector

**State Vector Format:**
```
[0:3]   - Position [x, y, z] in meters (LVLH frame)
[3:6]   - Velocity [vx, vy, vz] in m/s
[6:10]  - Attitude quaternion [qw, qx, qy, qz] 
[10:13] - Angular velocity [wx, wy, wz] in rad/s
```

#### set_target
```python
def set_target(self, target_state: np.ndarray) -> None
```
**Description:** Set the target state for the agent

**Parameters:**
- `target_state` (np.ndarray): 13-element target state vector

### State Access Methods

#### get_position
```python
def get_position() -> np.ndarray
```
**Description:** Get current position vector

**Returns:** 
- `np.ndarray`: 3-element position vector [x, y, z] in meters

#### get_velocity
```python
def get_velocity() -> np.ndarray
```
**Description:** Get current velocity vector

**Returns:** 
- `np.ndarray`: 3-element velocity vector [vx, vy, vz] in m/s

#### get_attitude
```python
def get_attitude() -> np.ndarray
```
**Description:** Get current attitude quaternion

**Returns:** 
- `np.ndarray`: 4-element quaternion [qw, qx, qy, qz]

#### get_angular_velocity
```python
def get_angular_velocity() -> np.ndarray
```
**Description:** Get current angular velocity

**Returns:** 
- `np.ndarray`: 3-element angular velocity vector [wx, wy, wz] in rad/s

### Communication Methods

#### receive_message
```python
async def receive_message(self, message: Dict, sender_id: str) -> None
```
**Description:** Receive and queue message from another agent

**Parameters:**
- `message` (Dict): Message content received
- `sender_id` (str): ID of the sending agent

#### process_messages
```python
async def process_messages() -> None
```
**Description:** Process all pending messages in the queue

### Performance & Monitoring

#### calculate_position_error
```python
def calculate_position_error() -> float
```
**Description:** Calculate Euclidean distance from current position to target

**Returns:** 
- `float`: Position error in meters

#### update_performance_metrics
```python
def update_performance_metrics(self, control_input: np.ndarray, dt: float) -> None
```
**Description:** Update performance tracking metrics

**Parameters:**
- `control_input` (np.ndarray): Applied control inputs
- `dt` (float): Time step

#### get_status
```python
def get_status() -> Dict
```
**Description:** Get comprehensive agent status

**Returns:** 
- `Dict`: Status dictionary with keys:
  - `agent_id`: Agent identifier
  - `position`: Current position
  - `velocity`: Current velocity  
  - `is_active`: Activity status
  - `mission_phase`: Current phase
  - `performance`: Performance metrics

---

## AdvancedSpacecraftAgent

Advanced spacecraft agent with integrated ML uncertainty prediction, formation control, fault tolerance, and security features.

### Class Definition
```python
class AdvancedSpacecraftAgent(SecureSpacecraftAgent):
    """
    Complete advanced spacecraft agent with all features integrated
    """
```

### Constructor
```python
def __init__(self, agent_id: str, config: dict)
```

**Configuration Parameters:**
```python
config = {
    'formation': {
        'pattern': str,          # Formation pattern ('line', 'triangle', 'circle', 'custom')
        'min_distance': float,   # Minimum inter-agent distance (meters)
        'type': str             # Formation type identifier
    },
    'ml_prediction': {
        'enabled': bool,         # Enable ML uncertainty prediction
        'model_path': str,       # Path to trained model (optional)
        'update_frequency': int  # Model update frequency (Hz)
    },
    'security': {
        'encryption_enabled': bool,      # Enable encrypted communications
        'key_rotation_interval': int     # Key rotation interval (seconds)
    },
    'fault_tolerance': {
        'fdir_enabled': bool,           # Enable FDIR system
        'redundancy_level': int         # Actuator redundancy level
    }
}
```

### Mission Execution

#### execute_mission
```python
async def execute_mission(self, mission_plan: dict) -> None
```
**Description:** Execute complete mission with all advanced features

**Parameters:**
- `mission_plan` (dict): Mission plan with phases and parameters

**Mission Plan Format:**
```python
mission_plan = {
    'phase': str,              # 'formation_approach', 'precision_docking', 'fault_recovery'
    'duration': float,         # Phase duration (seconds)
    'formation': dict,         # Formation configuration
    'target_center': list,     # Target center position [x, y, z]
    'num_spacecraft': int,     # Number of spacecraft in formation
    'safety_constraints': dict # Safety parameters
}
```

#### formation_approach_phase
```python
async def formation_approach_phase(self, mission_plan: dict) -> None
```
**Description:** Execute formation flying approach with ML optimization

**Parameters:**
- `mission_plan` (dict): Mission parameters for formation phase

#### precision_docking_phase
```python
async def precision_docking_phase(self, mission_plan: dict) -> None
```
**Description:** Execute high-precision docking maneuver

**Parameters:**
- `mission_plan` (dict): Mission parameters for docking phase

#### fault_recovery_phase
```python
async def fault_recovery_phase(self, mission_plan: dict) -> None
```
**Description:** Execute fault recovery procedures

**Parameters:**
- `mission_plan` (dict): Mission parameters for recovery phase

### Advanced Features

#### Formation Control Integration
- Automatic formation position calculation
- Leader-following and consensus-based control
- Formation reconfiguration capabilities
- Collision avoidance within formation

#### ML Uncertainty Prediction
- Real-time uncertainty estimation
- Adaptive model updating
- Performance improvement over time
- Integration with DR-MPC controller

#### Emergency Procedures
- Automatic fault detection and isolation
- Emergency abort sequences
- Safe mode operations
- Recovery strategy execution

---

## FaultTolerantSpacecraft

Spacecraft agent with comprehensive fault detection, isolation, and recovery (FDIR) capabilities.

### Class Definition
```python
class FaultTolerantSpacecraft(SpacecraftAgent):
    """
    Spacecraft agent with fault tolerance and FDIR capabilities
    """
```

### Constructor
```python
def __init__(self, agent_id: str)
```

### Fault Tolerance Methods

#### handle_faulty_actuator
```python
def handle_faulty_actuator(self, fault_status: FaultStatus) -> None
```
**Description:** Handle actuator fault with graceful degradation

**Parameters:**
- `fault_status` (FaultStatus): Fault information object

**FaultStatus Object:**
```python
@dataclass
class FaultStatus:
    type: FaultType              # Type of fault detected
    severity: float              # Fault severity (0.0 to 1.0)
    detected_time: float         # Time of fault detection
    estimated_impact: Dict       # Impact assessment
```

#### detect_thruster_faults
```python
def detect_thruster_faults(self) -> List[FaultStatus]
```
**Description:** Detect thruster performance anomalies

**Returns:** 
- `List[FaultStatus]`: List of detected faults

#### reconfigure_control_allocation
```python
def reconfigure_control_allocation(self, failed_thrusters: List[int]) -> np.ndarray
```
**Description:** Reconfigure control allocation matrix for failed thrusters

**Parameters:**
- `failed_thrusters` (List[int]): List of failed thruster indices

**Returns:** 
- `np.ndarray`: New control allocation matrix

### Fault Types

```python
class FaultType(Enum):
    THRUSTER_STUCK = "stuck"
    THRUSTER_DEGRADED = "degraded" 
    THRUSTER_COMPLETE_FAILURE = "complete_failure"
    SENSOR_DRIFT = "sensor_drift"
    COMMUNICATION_LOSS = "communication_loss"
```

---

## SecureSpacecraftAgent

Spacecraft agent with end-to-end encrypted communications and security features.

### Class Definition
```python
class SecureSpacecraftAgent(FaultTolerantSpacecraft):
    """
    Spacecraft agent with secure encrypted communications
    """
```

### Security Methods

#### secure_broadcast
```python
async def secure_broadcast(self, message: dict) -> None
```
**Description:** Broadcast encrypted message to all trusted agents

**Parameters:**
- `message` (dict): Message to broadcast securely

#### establish_secure_channel
```python
async def establish_secure_channel(self, target_agent: str) -> bool
```
**Description:** Establish encrypted communication channel with target agent

**Parameters:**
- `target_agent` (str): Target agent identifier

**Returns:** 
- `bool`: True if channel established successfully

#### rotate_encryption_keys
```python
async def rotate_encryption_keys(self) -> None
```
**Description:** Rotate encryption keys for enhanced security

### Security Features

- **AES-256 Encryption**: Military-grade symmetric encryption
- **RSA Key Exchange**: 2048-bit asymmetric key exchange
- **Message Authentication**: HMAC-SHA256 message authentication
- **Replay Protection**: Timestamp and nonce-based protection
- **Automatic Key Rotation**: Configurable key rotation intervals

---

## Code Examples

### Basic Agent Usage
```python
#!/usr/bin/env python3
"""Basic spacecraft agent usage example"""

import numpy as np
import asyncio
from src.agents.spacecraft_agent import SpacecraftAgent

class SimpleSpacecraftAgent(SpacecraftAgent):
    """Simple implementation of spacecraft agent"""
    
    async def update_control(self, dt):
        """Simple proportional control"""
        position_error = self.target_state[:3] - self.state[:3]
        velocity_error = self.target_state[3:6] - self.state[3:6]
        
        # PD control
        thrust = 2.0 * position_error + 0.5 * velocity_error
        torque = np.zeros(3)
        
        return np.concatenate([thrust, torque])
    
    async def communicate(self, message, target_agent=None):
        """Simple message logging"""
        print(f"Agent {self.agent_id} sending: {message}")

# Create and use agent
async def main():
    # Initialize agent
    agent = SimpleSpacecraftAgent('demo-agent')
    
    # Set initial state
    initial_state = np.zeros(13)
    initial_state[6] = 1.0  # Unit quaternion
    agent.update_state(initial_state)
    
    # Set target
    target = np.zeros(13)
    target[:3] = [10.0, 5.0, 0.0]  # Target position
    target[6] = 1.0  # Unit quaternion
    agent.set_target(target)
    
    # Control loop
    for i in range(100):
        control = await agent.update_control(0.1)
        print(f"Step {i}: Control = {control[:3]}, Error = {agent.calculate_position_error():.3f}m")
        
        # Simple integration (replace with proper dynamics)
        new_state = agent.state.copy()
        new_state[3:6] += control[:3] * 0.1 / 500.0  # F=ma integration
        new_state[:3] += new_state[3:6] * 0.1
        agent.update_state(new_state)
        
        if agent.calculate_position_error() < 0.1:
            print("Target reached!")
            break

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced Agent with Formation Control
```python
#!/usr/bin/env python3
"""Advanced agent with formation control example"""

import numpy as np
from src.agents.advanced_spacecraft_agent import AdvancedSpacecraftAgent

# Configuration for advanced features
advanced_config = {
    'formation': {
        'pattern': 'triangle',
        'min_distance': 10.0,
        'type': 'leader_following'
    },
    'security': {
        'encryption_enabled': True,
        'key_rotation_interval': 300
    },
    'fault_tolerance': {
        'fdir_enabled': True,
        'redundancy_level': 2
    }
}

# Create advanced agent
agent = AdvancedSpacecraftAgent('leader-001', advanced_config)

# Mission plan for formation flying
mission_plan = {
    'phase': 'formation_approach',
    'duration': 1800.0,
    'formation': {
        'type': 'triangle',
        'spacing': 15.0
    },
    'target_center': [100.0, 50.0, 0.0],
    'num_spacecraft': 3,
    'safety_constraints': {
        'min_separation': 8.0,
        'max_relative_velocity': 2.0
    }
}

# Execute mission
import asyncio
asyncio.run(agent.execute_mission(mission_plan))
```

### Fault Tolerance Example
```python
#!/usr/bin/env python3
"""Fault tolerance and recovery example"""

from src.agents.spacecraft_agent import SpacecraftAgent
from src.fault_tolerance.actuator_fdir import FaultStatus, FaultType

class FaultDemoAgent(SpacecraftAgent):
    def __init__(self, agent_id):
        super().__init__(agent_id)
        self.thruster_health = np.ones(12)  # 12 thrusters, all healthy
    
    async def update_control(self, dt):
        # Check for thruster faults
        faults = self.detect_faults()
        
        if faults:
            print(f"Faults detected: {len(faults)}")
            for fault in faults:
                self.handle_fault(fault)
        
        # Compute control with fault-aware allocation
        control = self.compute_fault_tolerant_control()
        return control
    
    def detect_faults(self):
        """Simulate fault detection"""
        faults = []
        
        # Check each thruster
        for i, health in enumerate(self.thruster_health):
            if health < 0.8:  # Degraded performance
                fault = FaultStatus(
                    type=FaultType.THRUSTER_DEGRADED,
                    severity=1.0 - health,
                    detected_time=time.time(),
                    estimated_impact={'thrust_reduction': (1.0 - health) * 100}
                )
                faults.append(fault)
        
        return faults
    
    def handle_fault(self, fault):
        """Handle detected fault"""
        print(f"Handling fault: {fault.type} with severity {fault.severity}")
        
        if fault.type == FaultType.THRUSTER_DEGRADED:
            # Increase thrust margins
            self.increase_control_authority(1.2)
        elif fault.type == FaultType.THRUSTER_COMPLETE_FAILURE:
            # Reconfigure control allocation
            self.reconfigure_actuators()
    
    async def communicate(self, message, target_agent=None):
        pass  # Implement as needed

# Simulate thruster degradation
agent = FaultDemoAgent('fault-demo')
agent.thruster_health[0] = 0.6  # Thruster 0 degraded to 60%
agent.thruster_health[5] = 0.0  # Thruster 5 completely failed
```

### Performance Monitoring
```python
#!/usr/bin/env python3
"""Performance monitoring and metrics example"""

import matplotlib.pyplot as plt
from src.agents.spacecraft_agent import SpacecraftAgent

# Create agent and run simulation
agent = SpacecraftAgent('monitor-demo')

# Simulation loop with performance tracking
performance_history = []
time_history = []

for step in range(1000):
    t = step * 0.1
    
    # Get control input
    control = await agent.update_control(0.1)
    
    # Update performance metrics
    agent.update_performance_metrics(control, 0.1)
    
    # Store history
    performance_history.append(agent.performance_metrics.copy())
    time_history.append(t)
    
    # Update state (simplified)
    agent.state[:3] += np.random.normal(0, 0.1, 3)  # Noisy position

# Plot performance metrics
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

# Fuel consumption
fuel_data = [p['fuel_consumption'] for p in performance_history]
ax1.plot(time_history, fuel_data)
ax1.set_ylabel('Fuel Consumption (kg)')
ax1.set_title('Performance Metrics Over Time')

# Position error
error_data = [p['position_error'] for p in performance_history]
ax2.plot(time_history, error_data)
ax2.set_ylabel('Position Error (m)')

# Computation time
comp_data = [p['computation_time'] for p in performance_history]
ax3.plot(time_history, comp_data)
ax3.set_ylabel('Computation Time (s)')
ax3.set_xlabel('Time (s)')

plt.tight_layout()
plt.show()
```

---

## Error Handling

All agent methods include comprehensive error handling:

```python
try:
    control = await agent.update_control(dt)
except ControlComputationError as e:
    logger.error(f"Control computation failed: {e}")
    # Fallback to safe control
    control = agent.get_safe_control()
except CommunicationError as e:
    logger.warning(f"Communication failed: {e}")
    # Continue with local control
except Exception as e:
    logger.critical(f"Unexpected error: {e}")
    # Emergency stop procedure
    await agent.emergency_stop()
```

## Thread Safety

All agent classes are designed to be thread-safe for multi-agent simulations:

- State updates use atomic operations
- Message queues are thread-safe
- Shared resources have proper locking
- Performance metrics use thread-local storage

## Memory Management

Agents implement efficient memory management:

- Circular buffers for trajectory history
- Automatic cleanup of old messages
- Configurable history lengths
- Memory pool for temporary calculations

---

*For more examples and advanced usage, see the [tutorials](../tutorials/) directory.*