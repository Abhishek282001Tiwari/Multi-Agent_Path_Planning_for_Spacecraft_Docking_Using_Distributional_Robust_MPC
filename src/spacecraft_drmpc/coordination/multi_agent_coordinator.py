# src/coordination/multi_agent_coordinator.py
import numpy as np
from typing import List, Dict
import threading
import queue
import time

class MultiAgentCoordinator:
    def __init__(self, num_agents: int, station_position: np.ndarray):
        self.num_agents = num_agents
        self.station_position = station_position
        self.agents = {}
        self.consensus_algorithm = DistributedConsensus()
        self.collision_manager = CollisionManager()
        
    def register_agent(self, agent_id: str, agent_instance):
        """Register a new spacecraft agent"""
        self.agents[agent_id] = {
            'instance': agent_instance,
            'state': None,
            'trajectory': None,
            'priority': len(self.agents)
        }
    
    def coordinate_trajectories(self):
        """Main coordination loop"""
        while True:
            # Collect all agent states and intentions
            agent_data = self.collect_agent_data()
            
            # Run distributed consensus
            consensus_trajectories = self.consensus_algorithm.compute(
                agent_data, self.station_position
            )
            
            # Check and resolve collisions
            safe_trajectories = self.collision_manager.avoid_collisions(
                consensus_trajectories
            )
            
            # Distribute updated trajectories
            self.distribute_trajectories(safe_trajectories)
            
            time.sleep(0.1)  # 10Hz coordination rate
    
    def collect_agent_data(self) -> Dict:
        """Collect current state and trajectory from all agents"""
        data = {}
        for agent_id, agent_info in self.agents.items():
            data[agent_id] = {
                'state': agent_info['instance'].get_current_state(),
                'trajectory': agent_info['instance'].get_planned_trajectory(),
                'priority': agent_info['priority']
            }
        return data
    
    def distribute_trajectories(self, trajectories: Dict):
        """Send updated trajectories to all agents"""
        for agent_id, trajectory in trajectories.items():
            if agent_id in self.agents:
                self.agents[agent_id]['instance'].update_trajectory(trajectory)

class DistributedConsensus:
    """Distributed consensus algorithm for multi-agent coordination"""
    
    def __init__(self, max_iterations=100, tolerance=1e-3):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def compute(self, agent_data: Dict, station_position: np.ndarray) -> Dict:
        """Compute consensus trajectories using ADMM"""
        
        # Initialize local variables
        local_trajectories = {}
        shared_trajectories = {}
        
        for agent_id, data in agent_data.items():
            local_trajectories[agent_id] = self.initialize_trajectory(
                data['state'][:3], station_position
            )
            shared_trajectories[agent_id] = local_trajectories[agent_id].copy()
        
        # ADMM iterations
        for iteration in range(self.max_iterations):
            # Update local variables
            for agent_id in agent_data:
                local_trajectories[agent_id] = self.update_local(
                    agent_id, agent_data, shared_trajectories
                )
            
            # Update shared variables
            old_shared = shared_trajectories.copy()
            shared_trajectories = self.update_shared(
                local_trajectories, agent_data
            )
            
            # Check convergence
            if self.check_convergence(old_shared, shared_trajectories):
                break
        
        return local_trajectories
    
    def initialize_trajectory(self, start_pos, end_pos):
        """Initialize straight-line trajectory"""
        return np.linspace(start_pos, end_pos, 50)
    
    def update_local(self, agent_id, agent_data, shared):
        """Update local trajectory for one agent"""
        # Implement local optimization considering neighbors
        return shared[agent_id]  # Simplified
    
    def update_shared(self, local, agent_data):
        """Update shared consensus variables"""
        # Implement averaging over neighbors
        return local  # Simplified
    
    def check_convergence(self, old, new):
        """Check if consensus has converged"""
        max_diff = 0
        for agent_id in old:
            diff = np.linalg.norm(old[agent_id] - new[agent_id])
            max_diff = max(max_diff, diff)
        return max_diff < self.tolerance

class CollisionManager:
    """Manages collision avoidance between agents"""
    
    def __init__(self, safety_radius=5.0):
        self.safety_radius = safety_radius
    
    def avoid_collisions(self, trajectories: Dict) -> Dict:
        """Modify trajectories to avoid collisions"""
        
        # Check all pairs for potential collisions
        agent_ids = list(trajectories.keys())
        
        for i in range(len(agent_ids)):
            for j in range(i+1, len(agent_ids)):
                agent1, agent2 = agent_ids[i], agent_ids[j]
                
                collision_points = self.detect_collision(
                    trajectories[agent1], trajectories[agent2]
                )
                
                if collision_points:
                    trajectories = self.resolve_collision(
                        trajectories, agent1, agent2, collision_points
                    )
        
        return trajectories
    
    def detect_collision(self, traj1, traj2):
        """Detect collision between two trajectories"""
        distances = np.linalg.norm(traj1 - traj2, axis=1)
        collisions = distances < self.safety_radius
        
        if np.any(collisions):
            return np.where(collisions)[0]
        return None
    
    def resolve_collision(self, trajectories, agent1, agent2, points):
        """Resolve collision by adjusting trajectories"""
        # Implement collision resolution
        # This is a simplified version - full implementation uses velocity obstacles
        
        for point in points:
            # Shift trajectories apart
            direction = trajectories[agent2][point] - trajectories[agent1][point]
            direction = direction / np.linalg.norm(direction)
            
            shift = self.safety_radius * 1.2
            trajectories[agent1][point] -= direction * shift/2
            trajectories[agent2][point] += direction * shift/2
        
        return trajectories