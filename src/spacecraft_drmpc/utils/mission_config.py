# src/utils/mission_config.py
import yaml
import numpy as np
from typing import Dict, List, Optional
import os

class MissionConfig:
    """Mission configuration loader and manager"""
    
    def __init__(self, scenario: str = 'three_spacecraft'):
        self.scenario = scenario
        self.config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config')
        self.load_config()
    
    def load_config(self):
        """Load configuration based on scenario"""
        
        # Default configurations
        self.spacecraft_configs = []
        self.mission_params = {}
        
        if self.scenario == 'single':
            self._setup_single_spacecraft()
        elif self.scenario == 'three_spacecraft':
            self._setup_three_spacecraft()
        elif self.scenario == 'formation_flying':
            self._setup_formation_flying()
        else:
            self._setup_default()
    
    def _setup_single_spacecraft(self):
        """Setup single spacecraft docking scenario"""
        self.spacecraft_configs = [{
            'agent_id': 'chaser-001',
            'initial_position': [0.0, 0.0, -50.0],
            'initial_velocity': [0.0, 0.0, 0.1],
            'target_position': [0.0, 0.0, 0.0],
            'mass': 500.0,
            'inertia': np.diag([100.0, 150.0, 120.0]).tolist()
        }]
        
        self.mission_params = {
            'duration': 1800.0,  # 30 minutes
            'time_step': 0.1,
            'target_position': [0.0, 0.0, 0.0],
            'docking_tolerance': 0.5
        }
    
    def _setup_three_spacecraft(self):
        """Setup three spacecraft cooperative docking"""
        self.spacecraft_configs = [
            {
                'agent_id': 'chaser-001',
                'initial_position': [-30.0, 0.0, -30.0],
                'initial_velocity': [0.05, 0.0, 0.05],
                'target_position': [-5.0, 0.0, 0.0],
                'mass': 500.0,
                'inertia': np.diag([100.0, 150.0, 120.0]).tolist()
            },
            {
                'agent_id': 'chaser-002',
                'initial_position': [30.0, 0.0, -30.0],
                'initial_velocity': [-0.05, 0.0, 0.05],
                'target_position': [5.0, 0.0, 0.0],
                'mass': 450.0,
                'inertia': np.diag([90.0, 140.0, 110.0]).tolist()
            },
            {
                'agent_id': 'target-001',
                'initial_position': [0.0, 0.0, 0.0],
                'initial_velocity': [0.0, 0.0, 0.0],
                'target_position': [0.0, 0.0, 0.0],
                'mass': 1000.0,
                'inertia': np.diag([200.0, 250.0, 220.0]).tolist()
            }
        ]
        
        self.mission_params = {
            'duration': 2400.0,  # 40 minutes
            'time_step': 0.1,
            'formation_distance': 10.0,
            'docking_tolerance': 0.5
        }
    
    def _setup_formation_flying(self):
        """Setup formation flying scenario"""
        num_spacecraft = 5
        formation_radius = 100.0
        
        self.spacecraft_configs = []
        for i in range(num_spacecraft):
            angle = 2 * np.pi * i / num_spacecraft
            x = formation_radius * np.cos(angle)
            y = formation_radius * np.sin(angle)
            
            self.spacecraft_configs.append({
                'agent_id': f'formation-{i+1:03d}',
                'initial_position': [x, y, 0.0],
                'initial_velocity': [0.0, 0.0, 0.0],
                'target_position': [x * 0.5, y * 0.5, 0.0],
                'mass': 400.0,
                'inertia': np.diag([80.0, 120.0, 100.0]).tolist()
            })
        
        self.mission_params = {
            'duration': 3600.0,  # 1 hour
            'time_step': 0.1,
            'formation_type': 'circle',
            'formation_radius': 50.0
        }
    
    def _setup_default(self):
        """Setup default configuration"""
        self._setup_three_spacecraft()
    
    def get_spacecraft_config(self, agent_id: str) -> Optional[Dict]:
        """Get configuration for specific spacecraft"""
        for config in self.spacecraft_configs:
            if config['agent_id'] == agent_id:
                return config
        return None
    
    def get_all_spacecraft_configs(self) -> List[Dict]:
        """Get all spacecraft configurations"""
        return self.spacecraft_configs.copy()
    
    def get_mission_params(self) -> Dict:
        """Get mission parameters"""
        return self.mission_params.copy()