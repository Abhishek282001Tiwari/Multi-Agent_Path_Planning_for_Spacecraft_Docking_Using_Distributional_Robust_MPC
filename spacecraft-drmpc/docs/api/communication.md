# Communication API Documentation

This document provides comprehensive documentation for inter-agent communication protocols, adaptive networking, and distributed coordination systems.

## Table of Contents
- [Communication Protocols](#communication-protocols)
- [Adaptive Networking](#adaptive-networking)
- [Message Types](#message-types)
- [Fault Tolerance](#fault-tolerance)
- [Code Examples](#code-examples)

---

## Communication Protocols

### AdaptiveCommunicationProtocol

Core communication protocol with adaptive timeout, retry mechanisms, and network optimization for spacecraft-to-spacecraft communications.

### Class Definition
```python
class AdaptiveCommunicationProtocol:
    """
    Adaptive timeout and retry mechanism for spacecraft communication
    
    Features:
    - Dynamic timeout adjustment based on network conditions
    - Exponential backoff with jitter for retry attempts
    - Bandwidth optimization and message prioritization
    - Quality of Service (QoS) management
    - Network topology awareness
    """
```

### Constructor
```python
def __init__(self, config: Optional[Dict] = None)
```

**Configuration Parameters:**
```python
config = {
    # Basic Parameters
    'base_timeout': float,          # Base timeout value (seconds)
    'max_timeout': float,           # Maximum timeout value (seconds)
    'min_timeout': float,           # Minimum timeout value (seconds)
    'max_retries': int,             # Maximum number of retry attempts
    'exponential_base': float,      # Exponential backoff base (default: 2.0)
    'jitter_factor': float,         # Jitter factor for timeout randomization
    
    # Adaptive Parameters
    'adaptation_rate': float,       # Rate of timeout adaptation (0.0-1.0)
    'latency_history_size': int,    # Size of latency history buffer
    'packet_loss_threshold': float, # Threshold for packet loss detection
    'rtt_smoothing_factor': float,  # RTT smoothing factor
    
    # QoS Parameters
    'enable_prioritization': bool,  # Enable message prioritization
    'bandwidth_limit': float,       # Bandwidth limit (bytes/second)
    'congestion_control': bool,     # Enable congestion control
    'flow_control': bool,          # Enable flow control
    
    # Network Parameters
    'network_topology': str,        # 'mesh', 'star', 'ring', 'hierarchy'
    'multi_path_routing': bool,     # Enable multi-path routing
    'load_balancing': bool,        # Enable load balancing
    'encryption_enabled': bool      # Enable message encryption
}
```

**Example Configuration:**
```python
config = {
    'base_timeout': 0.1,           # 100ms base timeout
    'max_timeout': 2.0,            # 2 second maximum
    'min_timeout': 0.05,           # 50ms minimum
    'max_retries': 3,              # Up to 3 retries
    'adaptation_rate': 0.1,        # 10% adaptation rate
    'enable_prioritization': True,
    'network_topology': 'mesh'
}

protocol = AdaptiveCommunicationProtocol(config)
```

### Core Methods

#### send_with_adaptive_timeout
```python
async def send_with_adaptive_timeout(self, message: Dict, destination: str,
                                   priority: int = 1, timeout_override: Optional[float] = None) -> Dict
```
**Description:** Send message with adaptive timeout and retry logic

**Parameters:**
- `message` (Dict): Message content to send
- `destination` (str): Target agent identifier
- `priority` (int): Message priority (1=highest, 5=lowest)
- `timeout_override` (Optional[float]): Override default timeout

**Returns:**
- `Dict`: Response message with metadata

**Message Format:**
```python
message = {
    'type': str,              # Message type identifier
    'sender': str,            # Sender agent ID
    'timestamp': float,       # Unix timestamp
    'sequence_id': int,       # Message sequence number
    'priority': int,          # Message priority
    'payload': Dict,          # Actual message content
    'requires_ack': bool,     # Whether acknowledgment is required
    'expires_at': float       # Message expiration time
}
```

**Example:**
```python
# High-priority control message
control_message = {
    'type': 'control_command',
    'payload': {
        'command': 'adjust_trajectory',
        'parameters': {
            'thrust_vector': [1.0, 0.0, 0.0],
            'duration': 5.0
        }
    },
    'requires_ack': True
}

response = await protocol.send_with_adaptive_timeout(
    control_message, 
    destination='follower-001',
    priority=1  # Highest priority
)

if response['status'] == 'success':
    print(f"Message delivered in {response['delivery_time']:.3f}s")
else:
    print(f"Delivery failed: {response['error']}")
```

#### update_network_statistics
```python
def update_network_statistics(self, destination: str, latency: float, 
                            success: bool, packet_size: int) -> None
```
**Description:** Update network performance statistics for adaptive algorithms

**Parameters:**
- `destination` (str): Target agent identifier
- `latency` (float): Measured round-trip time (seconds)
- `success` (bool): Whether the message was successfully delivered
- `packet_size` (int): Size of the transmitted packet (bytes)

#### get_optimal_timeout
```python
def get_optimal_timeout(self, destination: str, message_size: int) -> float
```
**Description:** Calculate optimal timeout value based on network conditions

**Parameters:**
- `destination` (str): Target agent identifier
- `message_size` (int): Message size in bytes

**Returns:**
- `float`: Optimal timeout value (seconds)

**Timeout Calculation:**
```
timeout = base_timeout + size_factor * message_size + network_factor * avg_latency
```

#### broadcast_with_confirmation
```python
async def broadcast_with_confirmation(self, message: Dict, 
                                    destinations: List[str],
                                    min_confirmations: int = None) -> Dict
```
**Description:** Broadcast message to multiple agents with confirmation tracking

**Parameters:**
- `message` (Dict): Message to broadcast
- `destinations` (List[str]): List of target agent IDs
- `min_confirmations` (int): Minimum required confirmations

**Returns:**
- `Dict`: Broadcast results with confirmation status

### Network Topology Management

#### NetworkTopology
```python
class NetworkTopology:
    """
    Network topology management for multi-agent communication
    """
    
    def __init__(self, topology_type: str, agents: List[str]):
        self.topology_type = topology_type
        self.agents = agents
        self.adjacency_matrix = self.build_adjacency_matrix()
        self.routing_table = self.build_routing_table()
    
    def build_adjacency_matrix(self) -> np.ndarray:
        """Build adjacency matrix based on topology type"""
        n = len(self.agents)
        matrix = np.zeros((n, n))
        
        if self.topology_type == 'mesh':
            # Full connectivity
            matrix = np.ones((n, n)) - np.eye(n)
            
        elif self.topology_type == 'star':
            # Hub-and-spoke with agent 0 as hub
            matrix[0, :] = 1
            matrix[:, 0] = 1
            matrix[0, 0] = 0
            
        elif self.topology_type == 'ring':
            # Ring topology
            for i in range(n):
                matrix[i, (i+1) % n] = 1
                matrix[i, (i-1) % n] = 1
                
        return matrix
    
    def find_shortest_path(self, source: str, destination: str) -> List[str]:
        """Find shortest communication path between agents"""
        # Implementation using Dijkstra's algorithm
        pass
    
    def get_neighbors(self, agent_id: str) -> List[str]:
        """Get direct neighbors of an agent"""
        agent_index = self.agents.index(agent_id)
        neighbor_indices = np.where(self.adjacency_matrix[agent_index] == 1)[0]
        return [self.agents[i] for i in neighbor_indices]
```

---

## Adaptive Networking

### AdaptiveNoiseFilter

Advanced filtering system for communication channels with adaptive noise suppression and signal enhancement.

### Class Definition
```python
class AdaptiveNoiseFilter:
    """
    Adaptive Extended Kalman Filter for spacecraft sensor spike mitigation
    and communication channel noise filtering
    """
```

### Constructor
```python
def __init__(self, state_dim: int = 13, measurement_dim: int = 6)
```

**Parameters:**
- `state_dim` (int): State vector dimension
- `measurement_dim` (int): Measurement vector dimension

### Methods

#### filter_message
```python
def filter_message(self, raw_message: Dict, channel_stats: Dict) -> Dict
```
**Description:** Filter incoming message for noise and corruption

**Parameters:**
- `raw_message` (Dict): Raw received message
- `channel_stats` (Dict): Channel statistics for filtering

**Returns:**
- `Dict`: Filtered message with confidence metrics

#### adapt_filter_parameters
```python
def adapt_filter_parameters(self, channel_quality: float, 
                           interference_level: float) -> None
```
**Description:** Adapt filter parameters based on channel conditions

### CommunicationStats

Data class for tracking communication performance metrics.

```python
@dataclass
class CommunicationStats:
    """Communication performance statistics"""
    
    avg_latency: float          # Average round-trip time (seconds)
    max_latency: float          # Maximum observed latency (seconds)
    min_latency: float          # Minimum observed latency (seconds)
    packet_loss_rate: float     # Packet loss rate (0.0-1.0)
    timeout_count: int          # Number of timeout events
    retry_count: int            # Number of retry attempts
    throughput: float           # Data throughput (bytes/second)
    jitter: float              # Latency variation (seconds)
    
    # Quality metrics
    signal_strength: float      # Signal strength indicator
    signal_to_noise_ratio: float # SNR in dB
    bit_error_rate: float      # Bit error rate
    
    def update(self, latency: float, success: bool, packet_size: int):
        """Update statistics with new measurement"""
        # Update moving averages and counters
        pass
    
    def get_quality_score(self) -> float:
        """Calculate overall communication quality score (0.0-1.0)"""
        # Weighted combination of metrics
        quality = (
            (1 - self.packet_loss_rate) * 0.3 +
            min(1.0, 1.0 / (1 + self.avg_latency)) * 0.3 +
            min(1.0, self.throughput / 1000.0) * 0.2 +
            min(1.0, self.signal_to_noise_ratio / 20.0) * 0.2
        )
        return max(0.0, min(1.0, quality))
```

---

## Message Types

### Standard Message Types

#### Control Messages
```python
class ControlMessage:
    """Control command messages between agents"""
    
    TYPE = "control_command"
    
    @staticmethod
    def create_thrust_command(thrust_vector: np.ndarray, duration: float) -> Dict:
        return {
            'type': ControlMessage.TYPE,
            'subtype': 'thrust_command',
            'payload': {
                'thrust_vector': thrust_vector.tolist(),
                'duration': duration,
                'coordinate_frame': 'body'
            },
            'priority': 1,  # High priority
            'requires_ack': True
        }
    
    @staticmethod
    def create_attitude_command(target_quaternion: np.ndarray, 
                              angular_velocity: np.ndarray) -> Dict:
        return {
            'type': ControlMessage.TYPE,
            'subtype': 'attitude_command',
            'payload': {
                'target_quaternion': target_quaternion.tolist(),
                'angular_velocity': angular_velocity.tolist()
            },
            'priority': 1,
            'requires_ack': True
        }
```

#### State Update Messages
```python
class StateUpdateMessage:
    """State information sharing messages"""
    
    TYPE = "state_update"
    
    @staticmethod
    def create_position_update(agent_id: str, state: np.ndarray, 
                             covariance: np.ndarray = None) -> Dict:
        message = {
            'type': StateUpdateMessage.TYPE,
            'subtype': 'position_update',
            'payload': {
                'agent_id': agent_id,
                'position': state[:3].tolist(),
                'velocity': state[3:6].tolist(),
                'attitude': state[6:10].tolist(),
                'angular_velocity': state[10:13].tolist(),
                'timestamp': time.time()
            },
            'priority': 2,  # Medium priority
            'requires_ack': False
        }
        
        if covariance is not None:
            message['payload']['covariance'] = covariance.tolist()
        
        return message
    
    @staticmethod
    def create_trajectory_update(agent_id: str, trajectory: np.ndarray,
                               time_vector: np.ndarray) -> Dict:
        return {
            'type': StateUpdateMessage.TYPE,
            'subtype': 'trajectory_update',
            'payload': {
                'agent_id': agent_id,
                'trajectory': trajectory.tolist(),
                'time_vector': time_vector.tolist(),
                'prediction_horizon': len(time_vector)
            },
            'priority': 3,
            'requires_ack': False
        }
```

#### Coordination Messages
```python
class CoordinationMessage:
    """Multi-agent coordination messages"""
    
    TYPE = "coordination"
    
    @staticmethod
    def create_formation_command(formation_type: str, parameters: Dict) -> Dict:
        return {
            'type': CoordinationMessage.TYPE,
            'subtype': 'formation_command',
            'payload': {
                'formation_type': formation_type,
                'parameters': parameters,
                'timestamp': time.time()
            },
            'priority': 2,
            'requires_ack': True
        }
    
    @staticmethod  
    def create_consensus_proposal(proposal_id: str, value: float,
                                round_number: int) -> Dict:
        return {
            'type': CoordinationMessage.TYPE,
            'subtype': 'consensus_proposal',
            'payload': {
                'proposal_id': proposal_id,
                'value': value,
                'round_number': round_number,
                'timestamp': time.time()
            },
            'priority': 2,
            'requires_ack': True
        }
    
    @staticmethod
    def create_conflict_resolution(conflict_id: str, resolution: Dict) -> Dict:
        return {
            'type': CoordinationMessage.TYPE,
            'subtype': 'conflict_resolution',
            'payload': {
                'conflict_id': conflict_id,
                'resolution': resolution,
                'authority_level': 1  # Higher numbers have more authority
            },
            'priority': 1,  # High priority
            'requires_ack': True
        }
```

#### Emergency Messages
```python
class EmergencyMessage:
    """Emergency and fault notification messages"""
    
    TYPE = "emergency"
    
    @staticmethod
    def create_fault_notification(fault_type: str, severity: float,
                                 affected_systems: List[str]) -> Dict:
        return {
            'type': EmergencyMessage.TYPE,
            'subtype': 'fault_notification',
            'payload': {
                'fault_type': fault_type,
                'severity': severity,  # 0.0-1.0
                'affected_systems': affected_systems,
                'detected_at': time.time(),
                'recovery_actions': []
            },
            'priority': 0,  # Highest priority
            'requires_ack': True,
            'broadcast': True
        }
    
    @staticmethod
    def create_collision_warning(other_agent: str, time_to_collision: float,
                               closest_approach: float) -> Dict:
        return {
            'type': EmergencyMessage.TYPE,
            'subtype': 'collision_warning',
            'payload': {
                'other_agent': other_agent,
                'time_to_collision': time_to_collision,
                'closest_approach': closest_approach,
                'warning_level': 'high' if time_to_collision < 60 else 'medium'
            },
            'priority': 0,
            'requires_ack': True
        }
    
    @staticmethod
    def create_emergency_stop(reason: str) -> Dict:
        return {
            'type': EmergencyMessage.TYPE,
            'subtype': 'emergency_stop',
            'payload': {
                'reason': reason,
                'stop_all_operations': True,
                'safe_mode_required': True
            },
            'priority': 0,
            'requires_ack': True,
            'broadcast': True
        }
```

### Message Validation

```python
class MessageValidator:
    """Validate message format and content"""
    
    @staticmethod
    def validate_message(message: Dict) -> Tuple[bool, str]:
        """Validate message structure and content"""
        
        # Check required fields
        required_fields = ['type', 'payload', 'priority']
        for field in required_fields:
            if field not in message:
                return False, f"Missing required field: {field}"
        
        # Validate priority
        if not isinstance(message['priority'], int) or not 0 <= message['priority'] <= 5:
            return False, "Priority must be integer between 0 and 5"
        
        # Validate timestamp if present
        if 'timestamp' in message:
            if not isinstance(message['timestamp'], (int, float)):
                return False, "Timestamp must be numeric"
            
            # Check if timestamp is reasonable (not too old or in future)
            current_time = time.time()
            if abs(message['timestamp'] - current_time) > 300:  # 5 minutes
                return False, "Timestamp too far from current time"
        
        # Type-specific validation
        if message['type'] == ControlMessage.TYPE:
            return MessageValidator._validate_control_message(message)
        elif message['type'] == StateUpdateMessage.TYPE:
            return MessageValidator._validate_state_message(message)
        elif message['type'] == EmergencyMessage.TYPE:
            return MessageValidator._validate_emergency_message(message)
        
        return True, "Valid"
    
    @staticmethod
    def _validate_control_message(message: Dict) -> Tuple[bool, str]:
        """Validate control message content"""
        payload = message['payload']
        
        if 'subtype' not in message:
            return False, "Control message missing subtype"
        
        if message['subtype'] == 'thrust_command':
            if 'thrust_vector' not in payload:
                return False, "Thrust command missing thrust_vector"
            
            thrust = payload['thrust_vector']
            if not isinstance(thrust, list) or len(thrust) != 3:
                return False, "Thrust vector must be 3-element list"
            
            # Check thrust magnitude limits
            thrust_magnitude = np.linalg.norm(thrust)
            if thrust_magnitude > 100.0:  # 100N limit
                return False, f"Thrust magnitude {thrust_magnitude:.1f}N exceeds limit"
        
        return True, "Valid"
    
    @staticmethod
    def _validate_state_message(message: Dict) -> Tuple[bool, str]:
        """Validate state update message content"""
        payload = message['payload']
        
        if 'agent_id' not in payload:
            return False, "State message missing agent_id"
        
        if message.get('subtype') == 'position_update':
            required_fields = ['position', 'velocity', 'attitude', 'angular_velocity']
            for field in required_fields:
                if field not in payload:
                    return False, f"Position update missing {field}"
                
                if field == 'attitude':
                    if len(payload[field]) != 4:
                        return False, "Attitude quaternion must have 4 elements"
                elif len(payload[field]) != 3:
                    return False, f"{field} must have 3 elements"
        
        return True, "Valid"
    
    @staticmethod
    def _validate_emergency_message(message: Dict) -> Tuple[bool, str]:
        """Validate emergency message content"""
        payload = message['payload']
        
        if message.get('subtype') == 'fault_notification':
            if 'fault_type' not in payload or 'severity' not in payload:
                return False, "Fault notification missing required fields"
            
            severity = payload['severity']
            if not isinstance(severity, (int, float)) or not 0 <= severity <= 1:
                return False, "Severity must be numeric between 0 and 1"
        
        return True, "Valid"
```

---

## Fault Tolerance

### Communication Fault Handling

```python
class CommunicationFaultHandler:
    """Handle communication failures and implement recovery strategies"""
    
    def __init__(self, config: Dict):
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay_base = config.get('retry_delay_base', 1.0)
        self.alternative_routes = config.get('alternative_routes', True)
        self.store_and_forward = config.get('store_and_forward', True)
        
        # Message queue for failed deliveries
        self.failed_messages = deque(maxlen=1000)
        self.retry_scheduler = {}
    
    async def handle_send_failure(self, message: Dict, destination: str, 
                                 error: Exception) -> bool:
        """Handle failed message transmission"""
        
        failure_info = {
            'message': message.copy(),
            'destination': destination,
            'error': str(error),
            'failure_time': time.time(),
            'retry_count': message.get('retry_count', 0)
        }
        
        # Check if we should retry
        if failure_info['retry_count'] < self.max_retries:
            return await self._schedule_retry(failure_info)
        else:
            return await self._handle_permanent_failure(failure_info)
    
    async def _schedule_retry(self, failure_info: Dict) -> bool:
        """Schedule message for retry with exponential backoff"""
        
        retry_count = failure_info['retry_count']
        retry_delay = self.retry_delay_base * (2 ** retry_count)
        
        # Add jitter to prevent thundering herd
        jitter = random.uniform(0.8, 1.2)
        retry_delay *= jitter
        
        # Schedule retry
        retry_time = time.time() + retry_delay
        message_id = f"{failure_info['destination']}_{retry_time}"
        
        self.retry_scheduler[message_id] = {
            'retry_time': retry_time,
            'failure_info': failure_info
        }
        
        print(f"Scheduled retry for message to {failure_info['destination']} "
              f"in {retry_delay:.2f}s (attempt {retry_count + 1})")
        
        return True
    
    async def _handle_permanent_failure(self, failure_info: Dict) -> bool:
        """Handle permanently failed message"""
        
        message = failure_info['message']
        
        # Try alternative routes if available
        if self.alternative_routes:
            alternative_sent = await self._try_alternative_routes(failure_info)
            if alternative_sent:
                return True
        
        # Store for later if store-and-forward is enabled
        if self.store_and_forward:
            self.failed_messages.append(failure_info)
            print(f"Stored failed message to {failure_info['destination']} "
                  f"for later delivery")
        
        # Check if this is a critical message
        if message.get('priority', 5) <= 1:  # High priority message
            await self._handle_critical_message_failure(failure_info)
        
        return False
    
    async def _try_alternative_routes(self, failure_info: Dict) -> bool:
        """Try sending message via alternative routes"""
        
        # This would implement multi-hop routing through other agents
        # For now, just a placeholder
        print(f"Attempting alternative routes for message to "
              f"{failure_info['destination']}")
        
        return False  # Not implemented in this example
    
    async def _handle_critical_message_failure(self, failure_info: Dict):
        """Handle failure of critical high-priority message"""
        
        # Generate emergency alert
        emergency_msg = EmergencyMessage.create_fault_notification(
            fault_type="communication_failure",
            severity=0.8,
            affected_systems=["communication", "coordination"]
        )
        
        # Broadcast emergency message
        print(f"CRITICAL: Communication failure to {failure_info['destination']} "
              f"for high-priority message")
        
        # Could trigger safe mode or emergency procedures
    
    async def process_retry_queue(self):
        """Process scheduled message retries"""
        
        current_time = time.time()
        ready_retries = []
        
        # Find messages ready for retry
        for message_id, retry_info in self.retry_scheduler.items():
            if current_time >= retry_info['retry_time']:
                ready_retries.append(message_id)
        
        # Process ready retries
        for message_id in ready_retries:
            retry_info = self.retry_scheduler.pop(message_id)
            failure_info = retry_info['failure_info']
            
            # Increment retry count
            failure_info['retry_count'] += 1
            failure_info['message']['retry_count'] = failure_info['retry_count']
            
            # Attempt to resend
            try:
                # This would call the actual send method
                print(f"Retrying message to {failure_info['destination']} "
                      f"(attempt {failure_info['retry_count']})")
                
                # success = await self.send_message(
                #     failure_info['message'], 
                #     failure_info['destination']
                # )
                
                # For demo, assume some retries succeed
                success = random.random() > 0.3
                
                if not success:
                    await self.handle_send_failure(
                        failure_info['message'],
                        failure_info['destination'],
                        Exception("Retry failed")
                    )
                else:
                    print(f"Retry successful for {failure_info['destination']}")
                    
            except Exception as e:
                await self.handle_send_failure(
                    failure_info['message'],
                    failure_info['destination'],
                    e
                )
```

### Network Partitioning Handling

```python
class NetworkPartitionHandler:
    """Handle network partitions and connectivity changes"""
    
    def __init__(self, topology: NetworkTopology):
        self.topology = topology
        self.partition_detector = PartitionDetector()
        self.bridge_agents = []  # Agents that can bridge partitions
        
    def detect_partition(self, connectivity_matrix: np.ndarray) -> List[List[str]]:
        """Detect network partitions and return connected components"""
        
        # Use graph algorithms to find connected components
        n_agents = len(self.topology.agents)
        visited = [False] * n_agents
        partitions = []
        
        for i in range(n_agents):
            if not visited[i]:
                partition = []
                self._dfs_partition(i, connectivity_matrix, visited, partition)
                partitions.append([self.topology.agents[j] for j in partition])
        
        return partitions
    
    def _dfs_partition(self, agent_idx: int, connectivity: np.ndarray, 
                      visited: List[bool], partition: List[int]):
        """Depth-first search to find connected component"""
        visited[agent_idx] = True
        partition.append(agent_idx)
        
        for neighbor_idx in range(len(connectivity)):
            if connectivity[agent_idx][neighbor_idx] and not visited[neighbor_idx]:
                self._dfs_partition(neighbor_idx, connectivity, visited, partition)
    
    async def handle_partition_detected(self, partitions: List[List[str]]):
        """Handle detected network partition"""
        
        print(f"Network partition detected: {len(partitions)} components")
        for i, partition in enumerate(partitions):
            print(f"  Partition {i+1}: {partition}")
        
        # Try to establish bridge connections
        await self._establish_bridges(partitions)
        
        # Reconfigure routing tables
        self._reconfigure_routing(partitions)
        
        # Notify agents about partition
        await self._notify_agents_of_partition(partitions)
    
    async def _establish_bridges(self, partitions: List[List[str]]):
        """Try to establish bridge connections between partitions"""
        
        for bridge_agent in self.bridge_agents:
            # Bridge agents with long-range communication capabilities
            # could try to connect different partitions
            pass
    
    def _reconfigure_routing(self, partitions: List[List[str]]):
        """Reconfigure routing tables for partitioned network"""
        
        # Update routing tables to avoid routes through broken connections
        for partition in partitions:
            # Create sub-topology for each partition
            sub_topology = self._create_sub_topology(partition)
            # Update routing tables for agents in this partition
```

---

## Code Examples

### Basic Communication Setup

```python
#!/usr/bin/env python3
"""Basic communication protocol setup and usage"""

import asyncio
import numpy as np
from src.communication.adaptive_timeout import AdaptiveCommunicationProtocol
from src.communication.message_types import ControlMessage, StateUpdateMessage

class CommunicatingAgent:
    """Example agent with communication capabilities"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.state = np.zeros(13)  # 13-element state vector
        
        # Initialize communication protocol
        comm_config = {
            'base_timeout': 0.1,
            'max_timeout': 2.0,
            'max_retries': 3,
            'enable_prioritization': True
        }
        self.comm_protocol = AdaptiveCommunicationProtocol(comm_config)
        
        # Message handlers
        self.message_handlers = {
            'control_command': self.handle_control_command,
            'state_update': self.handle_state_update,
            'coordination': self.handle_coordination_message
        }
    
    async def send_state_update(self, target_agents: List[str]):
        """Send state update to specified agents"""
        
        # Create state update message
        message = StateUpdateMessage.create_position_update(
            self.agent_id, 
            self.state
        )
        
        # Send to all target agents
        results = {}
        for target in target_agents:
            try:
                response = await self.comm_protocol.send_with_adaptive_timeout(
                    message, target, priority=2
                )
                results[target] = response
                print(f"State update sent to {target}: {response['status']}")
                
            except Exception as e:
                print(f"Failed to send state update to {target}: {e}")
                results[target] = {'status': 'error', 'error': str(e)}
        
        return results
    
    async def send_thrust_command(self, target_agent: str, 
                                 thrust_vector: np.ndarray, duration: float):
        """Send thrust command to another agent"""
        
        # Create control message
        message = ControlMessage.create_thrust_command(thrust_vector, duration)
        
        try:
            response = await self.comm_protocol.send_with_adaptive_timeout(
                message, target_agent, priority=1  # High priority
            )
            
            if response['status'] == 'success':
                print(f"Thrust command sent to {target_agent} successfully")
                return True
            else:
                print(f"Thrust command failed: {response.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"Communication error: {e}")
            return False
    
    async def handle_control_command(self, message: Dict, sender: str):
        """Handle incoming control command"""
        
        payload = message['payload']
        subtype = message.get('subtype', 'unknown')
        
        print(f"Received {subtype} from {sender}")
        
        if subtype == 'thrust_command':
            thrust_vector = np.array(payload['thrust_vector'])
            duration = payload['duration']
            
            # Apply thrust command
            print(f"Applying thrust {thrust_vector} for {duration}s")
            
            # In real implementation, would integrate with control system
            # self.apply_thrust(thrust_vector, duration)
            
        elif subtype == 'attitude_command':
            target_quat = np.array(payload['target_quaternion'])
            ang_vel = np.array(payload['angular_velocity'])
            
            print(f"Setting attitude target: {target_quat}")
            # self.set_attitude_target(target_quat, ang_vel)
    
    async def handle_state_update(self, message: Dict, sender: str):
        """Handle incoming state update"""
        
        payload = message['payload']
        
        if message.get('subtype') == 'position_update':
            sender_position = np.array(payload['position'])
            sender_velocity = np.array(payload['velocity'])
            
            print(f"Received state update from {sender}: "
                  f"pos={sender_position}, vel={sender_velocity}")
            
            # Store neighbor state for coordination
            # self.neighbor_states[sender] = {
            #     'position': sender_position,
            #     'velocity': sender_velocity,
            #     'timestamp': payload['timestamp']
            # }
    
    async def handle_coordination_message(self, message: Dict, sender: str):
        """Handle coordination messages"""
        
        subtype = message.get('subtype', 'unknown')
        print(f"Received coordination message ({subtype}) from {sender}")
        
        # Handle different coordination message types
        if subtype == 'formation_command':
            formation_type = message['payload']['formation_type']
            parameters = message['payload']['parameters']
            print(f"Formation command: {formation_type} with params {parameters}")
    
    async def message_loop(self):
        """Main message processing loop"""
        
        # Simulate receiving messages
        while True:
            # In real implementation, would receive from communication hardware
            await asyncio.sleep(0.1)
            
            # Process any pending messages
            # for message in self.receive_queue:
            #     await self.process_message(message)

# Usage example
async def main():
    # Create communicating agents
    leader = CommunicatingAgent('leader-001')
    follower1 = CommunicatingAgent('follower-001')
    follower2 = CommunicatingAgent('follower-002')
    
    # Set some initial states
    leader.state[:3] = np.array([0, 0, 0])      # At origin
    follower1.state[:3] = np.array([10, 0, 0])  # 10m in x
    follower2.state[:3] = np.array([5, 5, 0])   # 5m in x and y
    
    # Leader sends state updates to followers
    await leader.send_state_update(['follower-001', 'follower-002'])
    
    # Leader commands follower to apply thrust
    thrust_command = np.array([1.0, 0.0, 0.0])  # 1N in x direction
    await leader.send_thrust_command('follower-001', thrust_command, 5.0)
    
    # Simulate message handling
    control_msg = ControlMessage.create_thrust_command(thrust_command, 5.0)
    await follower1.handle_control_command(control_msg, 'leader-001')
    
    state_msg = StateUpdateMessage.create_position_update('leader-001', leader.state)
    await follower2.handle_state_update(state_msg, 'leader-001')

if __name__ == "__main__":
    asyncio.run(main())
```

### Fault-Tolerant Communication

```python
#!/usr/bin/env python3
"""Fault-tolerant communication with retry and alternative routing"""

import asyncio
import random
from src.communication.adaptive_timeout import AdaptiveCommunicationProtocol
from src.communication.fault_handling import CommunicationFaultHandler

class FaultTolerantCommAgent:
    """Agent with fault-tolerant communication capabilities"""
    
    def __init__(self, agent_id: str, neighbors: List[str]):
        self.agent_id = agent_id
        self.neighbors = neighbors
        self.online_neighbors = set(neighbors)
        
        # Communication components
        self.comm_protocol = AdaptiveCommunicationProtocol({
            'base_timeout': 0.5,
            'max_timeout': 5.0,
            'max_retries': 3
        })
        
        self.fault_handler = CommunicationFaultHandler({
            'max_retries': 3,
            'retry_delay_base': 1.0,
            'alternative_routes': True,
            'store_and_forward': True
        })
        
        # Network monitoring
        self.connectivity_monitor = ConnectivityMonitor(agent_id, neighbors)
        
    async def send_with_fault_tolerance(self, message: Dict, 
                                       destination: str) -> bool:
        """Send message with comprehensive fault tolerance"""
        
        try:
            # Check if destination is directly reachable
            if destination in self.online_neighbors:
                response = await self.comm_protocol.send_with_adaptive_timeout(
                    message, destination
                )
                
                if response['status'] == 'success':
                    return True
                else:
                    raise Exception(response.get('error', 'Send failed'))
            
            else:
                # Try multi-hop routing
                return await self.send_via_multihop(message, destination)
                
        except Exception as e:
            # Handle the failure
            success = await self.fault_handler.handle_send_failure(
                message, destination, e
            )
            return success
    
    async def send_via_multihop(self, message: Dict, destination: str) -> bool:
        """Send message via multi-hop routing"""
        
        # Find path to destination
        path = self.find_path_to_destination(destination)
        
        if not path:
            raise Exception(f"No path to destination {destination}")
        
        # Send via first hop with routing information
        next_hop = path[0]
        routing_message = {
            'type': 'routed_message',
            'final_destination': destination,
            'path': path[1:],  # Remaining path
            'original_message': message,
            'hop_count': 0,
            'max_hops': len(path)
        }
        
        return await self.send_with_fault_tolerance(routing_message, next_hop)
    
    def find_path_to_destination(self, destination: str) -> List[str]:
        """Find communication path to destination"""
        
        # Simple breadth-first search for shortest path
        # In practice, would use network topology information
        
        if destination in self.online_neighbors:
            return [destination]
        
        # For demo, return a multi-hop path if available
        for neighbor in self.online_neighbors:
            # Check if neighbor can reach destination
            # This would normally query neighbor's routing table
            if random.random() > 0.5:  # 50% chance neighbor has path
                return [neighbor, destination]
        
        return []  # No path found
    
    async def handle_connectivity_change(self, neighbor: str, is_online: bool):
        """Handle neighbor connectivity changes"""
        
        if is_online:
            self.online_neighbors.add(neighbor)
            print(f"Neighbor {neighbor} came online")
            
            # Try to deliver stored messages
            await self.retry_stored_messages()
            
        else:
            self.online_neighbors.discard(neighbor)
            print(f"Neighbor {neighbor} went offline")
            
            # Update routing tables
            self.update_routing_after_failure(neighbor)
    
    async def retry_stored_messages(self):
        """Retry messages that were stored due to delivery failures"""
        
        # Process any messages in the fault handler's queue
        await self.fault_handler.process_retry_queue()
        
        # Try to deliver store-and-forward messages
        for failure_info in list(self.fault_handler.failed_messages):
            destination = failure_info['destination']
            
            if destination in self.online_neighbors:
                # Destination is now reachable
                message = failure_info['message']
                success = await self.send_with_fault_tolerance(message, destination)
                
                if success:
                    self.fault_handler.failed_messages.remove(failure_info)
                    print(f"Successfully delivered stored message to {destination}")
    
    def update_routing_after_failure(self, failed_neighbor: str):
        """Update routing information after neighbor failure"""
        
        # Remove failed neighbor from routing tables
        # Recalculate paths that went through failed neighbor
        print(f"Updating routing after {failed_neighbor} failure")
        
        # In practice, would update routing tables and notify other agents
    
    async def network_monitoring_loop(self):
        """Monitor network connectivity continuously"""
        
        while True:
            # Check connectivity to all neighbors
            connectivity_results = await self.connectivity_monitor.check_all_neighbors()
            
            for neighbor, is_connected in connectivity_results.items():
                was_online = neighbor in self.online_neighbors
                
                if is_connected != was_online:
                    await self.handle_connectivity_change(neighbor, is_connected)
            
            await asyncio.sleep(5.0)  # Check every 5 seconds

class ConnectivityMonitor:
    """Monitor connectivity to neighboring agents"""
    
    def __init__(self, agent_id: str, neighbors: List[str]):
        self.agent_id = agent_id
        self.neighbors = neighbors
        
    async def check_all_neighbors(self) -> Dict[str, bool]:
        """Check connectivity to all neighbors"""
        
        results = {}
        
        for neighbor in self.neighbors:
            try:
                # Send ping message
                ping_message = {
                    'type': 'ping',
                    'payload': {'ping_id': f"{self.agent_id}_{time.time()}"},
                    'priority': 3
                }
                
                # Simulate connectivity check
                # In reality, would send actual ping message
                is_connected = await self.ping_neighbor(neighbor, ping_message)
                results[neighbor] = is_connected
                
            except Exception as e:
                print(f"Connectivity check failed for {neighbor}: {e}")
                results[neighbor] = False
        
        return results
    
    async def ping_neighbor(self, neighbor: str, ping_message: Dict) -> bool:
        """Ping specific neighbor to check connectivity"""
        
        try:
            # Simulate network conditions
            # 90% success rate normally, 30% during "outages"
            if random.random() < 0.1:  # 10% chance of temporary outage
                success_rate = 0.3
            else:
                success_rate = 0.9
            
            await asyncio.sleep(0.1)  # Simulate network delay
            
            return random.random() < success_rate
            
        except Exception:
            return False

# Usage example
async def main():
    # Create network of agents
    agents = {
        'agent_1': FaultTolerantCommAgent('agent_1', ['agent_2', 'agent_3']),
        'agent_2': FaultTolerantCommAgent('agent_2', ['agent_1', 'agent_3', 'agent_4']),
        'agent_3': FaultTolerantCommAgent('agent_3', ['agent_1', 'agent_2', 'agent_4']),
        'agent_4': FaultTolerantCommAgent('agent_4', ['agent_2', 'agent_3'])
    }
    
    # Start monitoring loops
    monitoring_tasks = []
    for agent in agents.values():
        task = asyncio.create_task(agent.network_monitoring_loop())
        monitoring_tasks.append(task)
    
    # Simulate communication with failures
    agent_1 = agents['agent_1']
    
    # Send messages with fault tolerance
    test_message = {
        'type': 'test_message',
        'payload': {'data': 'Hello, this is a test message'},
        'priority': 2
    }
    
    print("Sending messages with fault tolerance...")
    
    # Direct communication (should work)
    success = await agent_1.send_with_fault_tolerance(test_message, 'agent_2')
    print(f"Direct message to agent_2: {'Success' if success else 'Failed'}")
    
    # Multi-hop communication
    success = await agent_1.send_with_fault_tolerance(test_message, 'agent_4')
    print(f"Multi-hop message to agent_4: {'Success' if success else 'Failed'}")
    
    # Simulate network problems and recovery
    print("\\nSimulating network problems...")
    await agent_1.handle_connectivity_change('agent_2', False)  # agent_2 goes offline
    
    await asyncio.sleep(2)
    
    # Try sending message while agent_2 is offline
    success = await agent_1.send_with_fault_tolerance(test_message, 'agent_2')
    print(f"Message to offline agent_2: {'Success' if success else 'Failed'}")
    
    await asyncio.sleep(5)
    
    # agent_2 comes back online
    await agent_1.handle_connectivity_change('agent_2', True)
    
    await asyncio.sleep(2)
    
    # Cleanup
    for task in monitoring_tasks:
        task.cancel()

if __name__ == "__main__":
    asyncio.run(main())
```

### Distributed Consensus Communication

```python
#!/usr/bin/env python3
"""Distributed consensus algorithm with communication"""

import asyncio
import random
from typing import Dict, List, Optional
from src.communication.message_types import CoordinationMessage

class ConsensusAgent:
    """Agent implementing distributed consensus algorithm"""
    
    def __init__(self, agent_id: str, neighbors: List[str], initial_value: float):
        self.agent_id = agent_id
        self.neighbors = neighbors
        self.value = initial_value
        self.round_number = 0
        self.consensus_reached = False
        
        # Consensus state
        self.received_proposals = {}  # round -> {agent_id: value}
        self.convergence_threshold = 0.01
        
    async def start_consensus_round(self):
        """Start a new consensus round"""
        
        self.round_number += 1
        self.received_proposals[self.round_number] = {}
        
        # Send proposal to all neighbors
        proposal_msg = CoordinationMessage.create_consensus_proposal(
            f"{self.agent_id}_{self.round_number}",
            self.value,
            self.round_number
        )
        
        print(f"Agent {self.agent_id} Round {self.round_number}: "
              f"Proposing value {self.value:.4f}")
        
        # Send to all neighbors
        for neighbor in self.neighbors:
            await self.send_message(proposal_msg, neighbor)
        
        # Include own proposal
        self.received_proposals[self.round_number][self.agent_id] = self.value
    
    async def handle_consensus_proposal(self, message: Dict, sender: str):
        """Handle incoming consensus proposal"""
        
        payload = message['payload']
        round_num = payload['round_number']
        value = payload['value']
        
        # Store the proposal
        if round_num not in self.received_proposals:
            self.received_proposals[round_num] = {}
        
        self.received_proposals[round_num][sender] = value
        
        print(f"Agent {self.agent_id} received proposal from {sender}: "
              f"value={value:.4f}, round={round_num}")
        
        # Check if we have proposals from all neighbors for this round
        if self.have_all_proposals(round_num):
            await self.process_consensus_round(round_num)
    
    def have_all_proposals(self, round_num: int) -> bool:
        """Check if we have received all proposals for a round"""
        
        if round_num not in self.received_proposals:
            return False
        
        proposals = self.received_proposals[round_num]
        expected_agents = set(self.neighbors + [self.agent_id])
        received_agents = set(proposals.keys())
        
        return expected_agents == received_agents
    
    async def process_consensus_round(self, round_num: int):
        """Process consensus round after receiving all proposals"""
        
        proposals = self.received_proposals[round_num]
        values = list(proposals.values())
        
        # Average consensus algorithm
        new_value = sum(values) / len(values)
        
        # Check convergence
        value_change = abs(new_value - self.value)
        self.value = new_value
        
        print(f"Agent {self.agent_id} Round {round_num} complete: "
              f"New value={self.value:.4f}, Change={value_change:.4f}")
        
        if value_change < self.convergence_threshold:
            if not self.consensus_reached:
                print(f"Agent {self.agent_id} reached consensus: {self.value:.4f}")
                self.consensus_reached = True
        
        # Check if global consensus is reached
        values_array = np.array(values)
        value_std = np.std(values_array)
        
        if value_std < self.convergence_threshold:
            print(f"Global consensus reached in round {round_num}: "
                  f"Value={np.mean(values_array):.4f}")
            return
        
        # Start next round after delay
        await asyncio.sleep(1.0)
        if not self.consensus_reached:
            await self.start_consensus_round()
    
    async def send_message(self, message: Dict, destination: str):
        """Simulate sending message to another agent"""
        
        # Add sender information
        message['sender'] = self.agent_id
        message['timestamp'] = time.time()
        
        # Simulate network delay and potential message loss
        await asyncio.sleep(random.uniform(0.1, 0.3))
        
        if random.random() < 0.95:  # 95% success rate
            # In real implementation, would send via communication protocol
            print(f"  Message sent from {self.agent_id} to {destination}")
            return True
        else:
            print(f"  Message LOST from {self.agent_id} to {destination}")
            return False

# Consensus network simulation
async def run_consensus_simulation():
    """Run distributed consensus simulation"""
    
    # Create network of 5 agents with different initial values
    initial_values = [1.0, 2.5, 4.0, 3.2, 1.8]
    agent_ids = [f"agent_{i}" for i in range(5)]
    
    # Create fully connected network
    agents = {}
    for i, agent_id in enumerate(agent_ids):
        neighbors = [other_id for other_id in agent_ids if other_id != agent_id]
        agents[agent_id] = ConsensusAgent(agent_id, neighbors, initial_values[i])
    
    print("Starting distributed consensus simulation...")
    print("Initial values:", {aid: agent.value for aid, agent in agents.items()})
    print()
    
    # Start consensus algorithm
    consensus_tasks = []
    for agent in agents.values():
        task = asyncio.create_task(agent.start_consensus_round())
        consensus_tasks.append(task)
    
    # Wait for consensus to be reached
    await asyncio.gather(*consensus_tasks)
    
    print("\\nFinal values:", {aid: agent.value for aid, agent in agents.items()})
    
    # Simulate message handling
    # In practice, agents would have message receiving loops
    for round_num in range(1, 10):  # Simulate up to 10 rounds
        
        all_converged = True
        for agent in agents.values():
            if not agent.consensus_reached:
                all_converged = False
                break
        
        if all_converged:
            break
        
        # Simulate message exchanges
        for sender_agent in agents.values():
            for neighbor_id in sender_agent.neighbors:
                if neighbor_id in agents:
                    neighbor_agent = agents[neighbor_id]
                    
                    # Create mock proposal message
                    if sender_agent.round_number in sender_agent.received_proposals:
                        proposal_msg = CoordinationMessage.create_consensus_proposal(
                            f"{sender_agent.agent_id}_{sender_agent.round_number}",
                            sender_agent.value,
                            sender_agent.round_number
                        )
                        
                        await neighbor_agent.handle_consensus_proposal(
                            proposal_msg, sender_agent.agent_id
                        )
        
        await asyncio.sleep(2.0)  # Wait between rounds

if __name__ == "__main__":
    asyncio.run(run_consensus_simulation())
```

---

*For more advanced communication examples and networking protocols, see the [tutorials](../tutorials/) directory and [user manual](../user_manual.md).*