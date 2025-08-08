# src/formation/distributed_formation_control.py
import numpy as np
from typing import List, Tuple, Dict

class FormationController:
    """
    Distributed formation flying control for spacecraft swarms
    Based on consensus algorithms with collision avoidance
    """
    
    def __init__(self, formation_config):
        self.formation_pattern = formation_config['pattern']
        self.inter_spacecraft_distance = formation_config['min_distance']
        self.formation_type = formation_config['type']  # 'line', 'triangle', 'circle', 'custom'
        
        # Formation parameters
        self.leader_id = None
        self.formation_tolerance = 0.5  # meters
        self.reconfiguration_time = 30.0  # seconds
        
    def calculate_formation_positions(self, center_position: np.ndarray, 
                                    num_spacecraft: int) -> List[np.ndarray]:
        """Calculate target positions for formation flying"""
        
        if self.formation_type == 'line':
            return self._line_formation(center_position, num_spacecraft)
        elif self.formation_type == 'triangle':
            return self._triangle_formation(center_position, num_spacecraft)
        elif self.formation_type == 'circle':
            return self._circle_formation(center_position, num_spacecraft)
        elif self.formation_type == 'custom':
            return self._custom_formation(center_position, num_spacecraft)
    
    def _line_formation(self, center: np.ndarray, n: int) -> List[np.ndarray]:
        """Generate line formation positions"""
        positions = []
        spacing = self.inter_spacecraft_distance
        
        for i in range(n):
            offset = np.array([(i - (n-1)/2) * spacing, 0, 0])
            positions.append(center + offset)
            
        return positions
    
    def _triangle_formation(self, center: np.ndarray, n: int) -> List[np.ndarray]:
        """Generate triangular formation positions"""
        positions = []
        
        # Create triangular lattice
        row = 0
        col = 0
        spacecraft_count = 0
        
        while spacecraft_count < n:
            for pos_in_row in range(row + 1):
                if spacecraft_count >= n:
                    break
                    
                x = (pos_in_row - row/2) * self.inter_spacecraft_distance
                y = row * self.inter_spacecraft_distance * np.sqrt(3)/2
                positions.append(center + np.array([x, y, 0]))
                
                spacecraft_count += 1
            row += 1
            
        return positions
    
    def _circle_formation(self, center: np.ndarray, n: int) -> List[np.ndarray]:
        """Generate circular formation positions"""
        positions = []
        radius = self.inter_spacecraft_distance * np.sqrt(n) / (2 * np.pi)
        
        for i in range(n):
            angle = 2 * np.pi * i / n
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            positions.append(center + np.array([x, y, 0]))
            
        return positions
    
    def compute_formation_error(self, current_positions: Dict[str, np.ndarray],
                              target_positions: List[np.ndarray]) -> float:
        """Compute total formation error"""
        
        errors = []
        spacecraft_ids = list(current_positions.keys())
        
        for i, spacecraft_id in enumerate(spacecraft_ids):
            if i < len(target_positions):
                error = np.linalg.norm(
                    current_positions[spacecraft_id] - target_positions[i]
                )
                errors.append(error)
        
        return np.mean(errors)
    
    def reconfigure_formation(self, new_formation_type: str, transition_time: float):
        """Reconfigure formation pattern dynamically"""
        
        self.formation_type = new_formation_type
        
        # Generate smooth transition trajectories
        transition_trajectories = self._generate_transition_trajectories(
            transition_time
        )
        
        return transition_trajectories
    
    def _generate_transition_trajectories(self, duration: float) -> Dict[str, np.ndarray]:
        """Generate smooth trajectories for formation reconfiguration"""
        
        # Use minimum jerk trajectories
        trajectories = {}
        
        for agent_id in self.spacecraft_positions:
            start_pos = self.spacecraft_positions[agent_id]
            end_pos = self.target_positions[agent_id]
            
            # Generate smooth trajectory
            t = np.linspace(0, duration, int(duration/0.1))
            trajectory = self._minimum_jerk_trajectory(start_pos, end_pos, t)
            
            trajectories[agent_id] = trajectory
            
        return trajectories
    
    def _minimum_jerk_trajectory(self, start: np.ndarray, end: np.ndarray, 
                               time_points: np.ndarray) -> np.ndarray:
        """Generate minimum jerk trajectory"""
        
        duration = time_points[-1]
        t_normalized = time_points / duration
        
        # Minimum jerk polynomial
        trajectory = start + (end - start) * (
            10 * t_normalized**3 - 15 * t_normalized**4 + 6 * t_normalized**5
        )
        
        return trajectory

# Formation flying integration
class FormationSpacecraftAgent(SpacecraftAgent):
    def __init__(self, agent_id, formation_controller):
        super().__init__(agent_id)
        self.formation_controller = formation_controller
        self.formation_role = 'follower'  # or 'leader'
        
    def execute_formation_maneuver(self, target_center, formation_type):
        """Execute formation flying maneuver"""
        
        # Calculate target positions
        target_positions = self.formation_controller.calculate_formation_positions(
            target_center, self.formation_controller.num_spacecraft
        )
        
        # Assign target position for this spacecraft
        my_target = target_positions[self.agent_index]
        
        # Update DR-MPC reference
        self.dr_mpc_controller.set_formation_target(my_target)
        
        return my_target