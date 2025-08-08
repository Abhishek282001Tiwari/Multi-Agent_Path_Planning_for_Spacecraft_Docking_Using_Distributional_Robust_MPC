# src/agents/spacecraft_agent.py
import numpy as np
from typing import Dict, List, Optional, Tuple
import asyncio
import logging
from abc import ABC, abstractmethod

class SpacecraftAgent(ABC):
    """
    Base class for all spacecraft agents in the multi-agent system
    """
    
    def __init__(self, agent_id: str, config: Optional[Dict] = None):
        self.agent_id = agent_id
        self.config = config or {}
        self.logger = logging.getLogger(f"Agent-{agent_id}")
        
        # Agent state
        self.state = np.zeros(13)  # [position, velocity, quaternion, angular_velocity]
        self.target_state = np.zeros(13)
        self.is_active = True
        self.mission_phase = 'initialization'
        
        # Communication
        self.message_queue = asyncio.Queue()
        self.neighbors = []
        
        # Performance tracking
        self.performance_metrics = {
            'fuel_consumption': 0.0,
            'position_error': 0.0,
            'computation_time': 0.0
        }
        
    @abstractmethod
    async def update_control(self, dt: float) -> np.ndarray:
        """
        Compute control inputs for the spacecraft
        
        Args:
            dt: Time step
            
        Returns:
            Control inputs [thrust_x, thrust_y, thrust_z, torque_x, torque_y, torque_z]
        """
        pass
    
    @abstractmethod
    async def communicate(self, message: Dict, target_agent: Optional[str] = None):
        """
        Send message to other agents
        
        Args:
            message: Message content
            target_agent: Target agent ID (None for broadcast)
        """
        pass
    
    def update_state(self, new_state: np.ndarray):
        """Update agent's current state"""
        self.state = new_state.copy()
        
    def set_target(self, target_state: np.ndarray):
        """Set target state for the agent"""
        self.target_state = target_state.copy()
        
    def get_position(self) -> np.ndarray:
        """Get current position [x, y, z]"""
        return self.state[:3]
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity [vx, vy, vz]"""
        return self.state[3:6]
        
    def get_attitude(self) -> np.ndarray:
        """Get current attitude quaternion [qw, qx, qy, qz]"""
        return self.state[6:10]
    
    def get_angular_velocity(self) -> np.ndarray:
        """Get current angular velocity [wx, wy, wz]"""
        return self.state[10:13]
    
    async def receive_message(self, message: Dict, sender_id: str):
        """Receive and process message from another agent"""
        await self.message_queue.put({
            'content': message,
            'sender': sender_id,
            'timestamp': asyncio.get_event_loop().time()
        })
    
    async def process_messages(self):
        """Process all pending messages"""
        while not self.message_queue.empty():
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=0.1)
                await self._handle_message(message)
            except asyncio.TimeoutError:
                break
    
    async def _handle_message(self, message: Dict):
        """Handle incoming message (override in subclasses)"""
        self.logger.debug(f"Received message from {message['sender']}: {message['content']}")
    
    def calculate_position_error(self) -> float:
        """Calculate position error from target"""
        return np.linalg.norm(self.get_position() - self.target_state[:3])
    
    def update_performance_metrics(self, control_input: np.ndarray, dt: float):
        """Update performance metrics"""
        # Fuel consumption (simplified)
        thrust_magnitude = np.linalg.norm(control_input[:3])
        self.performance_metrics['fuel_consumption'] += thrust_magnitude * dt
        
        # Position error
        self.performance_metrics['position_error'] = self.calculate_position_error()
    
    def get_status(self) -> Dict:
        """Get current agent status"""
        return {
            'agent_id': self.agent_id,
            'position': self.get_position().tolist(),
            'velocity': self.get_velocity().tolist(),
            'is_active': self.is_active,
            'mission_phase': self.mission_phase,
            'performance': self.performance_metrics.copy()
        }