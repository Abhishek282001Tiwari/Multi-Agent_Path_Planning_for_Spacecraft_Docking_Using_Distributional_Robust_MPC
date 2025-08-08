# src/communication/adaptive_timeout.py
import asyncio
import time
import numpy as np
from dataclasses import dataclass

from ..agents.spacecraft_agent import SpacecraftAgent
from ..filters.adaptive_noise_filter import AdaptiveNoiseFilter

@dataclass
class CommunicationStats:
    avg_latency: float
    max_latency: float
    packet_loss_rate: float
    timeout_count: int

class AdaptiveCommunicationProtocol:
    """Adaptive timeout and retry mechanism for spacecraft communication"""
    
    def __init__(self):
        self.base_timeout = 0.1  # 100ms
        self.max_timeout = 2.0   # 2s
        self.min_timeout = 0.05  # 50ms
        self.adaptation_factor = 1.2
        
        # Statistics tracking
        self.latency_history = []
        self.timeout_history = []
        self.packet_loss_history = []
        
    async def send_with_adaptive_timeout(self, message, destination):
        """Send message with dynamically adjusted timeout"""
        
        # Calculate current timeout
        timeout = self.calculate_optimal_timeout()
        
        try:
            start_time = time.perf_counter()
            
            # Send with current timeout
            response = await self.send_with_timeout(message, destination, timeout)
            
            # Update statistics
            latency = time.perf_counter() - start_time
            self.update_latency_stats(latency)
            
            return response
            
        except asyncio.TimeoutError:
            # Handle timeout - adjust parameters
            self.handle_timeout()
            raise
            
    def calculate_optimal_timeout(self):
        """Calculate optimal timeout based on recent performance"""
        
        if len(self.latency_history) < 5:
            return self.base_timeout
        
        # Use 95th percentile of recent latencies
        recent_latencies = self.latency_history[-20:]
        p95_latency = np.percentile(recent_latencies, 95)
        
        # Add safety margin
        optimal_timeout = min(
            max(p95_latency * 2.0, self.min_timeout),
            self.max_timeout
        )
        
        return optimal_timeout
    
    def update_latency_stats(self, latency):
        """Update latency tracking with exponential decay"""
        
        self.latency_history.append(latency)
        
        # Keep only recent history
        max_history = 100
        if len(self.latency_history) > max_history:
            self.latency_history.pop(0)
    
    def handle_timeout(self):
        """Handle communication timeout - adjust parameters"""
        
        self.timeout_history.append(time.time())
        
        # Increase timeout if frequent timeouts
        if len(self.timeout_history) > 3:
            recent_timeouts = [t for t in self.timeout_history 
                             if time.time() - t < 60]  # Last minute
            
            if len(recent_timeouts) > 3:
                # Too many timeouts - increase base timeout
                self.base_timeout = min(
                    self.base_timeout * self.adaptation_factor,
                    self.max_timeout
                )
                
                # Reset timeout history
                self.timeout_history = []
    
    def get_communication_stats(self):
        """Return current communication statistics"""
        
        if not self.latency_history:
            return CommunicationStats(0.1, 0.1, 0.0, 0)
        
        return CommunicationStats(
            avg_latency=np.mean(self.latency_history),
            max_latency=np.max(self.latency_history),
            packet_loss_rate=len(self.timeout_history) / max(len(self.latency_history), 1),
            timeout_count=len(self.timeout_history)
        )

# Integration with multi-agent system
class RobustSpacecraftAgent(SpacecraftAgent):
    def __init__(self, agent_id, dynamics_model, dr_mpc_controller):
        super().__init__(agent_id, dynamics_model, dr_mpc_controller)
        self.comm_protocol = AdaptiveCommunicationProtocol()
        self.noise_filter = AdaptiveNoiseFilter()
        
    async def broadcast_trajectory(self, trajectory):
        """Broadcast trajectory with robust communication"""
        
        try:
            response = await self.comm_protocol.send_with_adaptive_timeout(
                trajectory, "broadcast_channel"
            )
            return response
            
        except asyncio.TimeoutError:
            # Handle communication failure gracefully
            return self.handle_communication_failure(trajectory)
    
    def handle_communication_failure(self, local_trajectory):
        """Handle communication failure with autonomous operation"""
        
        # Continue with last known trajectory
        logging.warning(f"Agent {self.id}: Communication timeout, using local trajectory")
        
        # Reduce conservativeness in local planning
        self.dr_mpc_controller.increase_uncertainty_radius(1.5)
        
        return {"status": "local_mode", "trajectory": local_trajectory}