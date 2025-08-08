# src/simulations/docking_simulator.py
import numpy as np
import asyncio
import logging
import time
from typing import Dict, List, Optional
import h5py

from ..agents.spacecraft_agent import SpacecraftAgent
from ..controllers.dr_mpc_controller import DRMPCController
from ..dynamics.spacecraft_dynamics import SpacecraftDynamics
from ..coordination.multi_agent_coordinator import MultiAgentCoordinator

class SimulationResults:
    """Container for simulation results"""
    
    def __init__(self):
        self.spacecraft_states = {}
        self.control_inputs = {}
        self.performance_metrics = {}
        self.timestamps = []
        
    def save(self, filename: str):
        """Save results to HDF5 file"""
        try:
            with h5py.File(filename, 'w') as f:
                # Save timestamps
                f.create_dataset('timestamps', data=self.timestamps)
                
                # Save spacecraft data
                for agent_id, states in self.spacecraft_states.items():
                    group = f.create_group(f'spacecraft/{agent_id}')
                    group.create_dataset('states', data=np.array(states))
                    
                    if agent_id in self.control_inputs:
                        group.create_dataset('controls', data=np.array(self.control_inputs[agent_id]))
                        
                    if agent_id in self.performance_metrics:
                        metrics_group = group.create_group('metrics')
                        # Handle list of dictionaries
                        if self.performance_metrics[agent_id]:
                            sample_metrics = self.performance_metrics[agent_id][0]
                            for metric_name in sample_metrics.keys():
                                values = [m[metric_name] for m in self.performance_metrics[agent_id]]
                                metrics_group.create_dataset(metric_name, data=values)
                            
            logging.info(f"Results saved to {filename}")
        except Exception as e:
            logging.error(f"Failed to save results: {e}")

class DockingSimulator:
    """Multi-agent spacecraft docking simulator with DR-MPC control"""
    
    def __init__(self, mission_config):
        self.config = mission_config
        self.logger = logging.getLogger('DockingSimulator')
        
        # Simulation state
        self.current_time = 0.0
        self.dt = mission_config.get_mission_params().get('time_step', 0.1)
        
        # Initialize components
        self.spacecraft_agents = {}
        self.dynamics = {}
        self.coordinator = None
        self.results = SimulationResults()
        
        self._initialize_simulation()
    
    def _initialize_simulation(self):
        """Initialize simulation components"""
        
        # Create spacecraft agents
        for sc_config in self.config.get_all_spacecraft_configs():
            agent_id = sc_config['agent_id']
            
            # Create agent with DR-MPC controller
            controller_config = {
                'prediction_horizon': 20,
                'time_step': self.dt,
                'wasserstein_radius': 0.1,
                'confidence_level': 0.95,
                'max_thrust': 10.0,
                'max_torque': 1.0,
                'safety_radius': 5.0
            }
            
            agent = self._create_agent(agent_id, controller_config)
            self.spacecraft_agents[agent_id] = agent
            
            # Initialize state
            initial_state = np.zeros(13)
            initial_state[:3] = sc_config['initial_position']
            initial_state[3:6] = sc_config['initial_velocity']
            initial_state[6] = 1.0  # Quaternion w component
            agent.update_state(initial_state)
            
            # Set target
            target_state = np.zeros(13)
            target_state[:3] = sc_config['target_position']
            target_state[6] = 1.0  # Quaternion w component
            agent.set_target(target_state)
            
            # Create dynamics model
            dynamics_config = {
                'initial_mass': sc_config['mass'],
                'inertia_matrix': sc_config['inertia'],
                'thruster_config': {'num_thrusters': 12},
                'orbital_rate': 0.0011
            }
            self.dynamics[agent_id] = SpacecraftDynamics(dynamics_config)
            
            # Initialize result storage
            self.results.spacecraft_states[agent_id] = []
            self.results.control_inputs[agent_id] = []
            self.results.performance_metrics[agent_id] = []
        
        # Create coordinator
        station_position = np.array([0.0, 0.0, 0.0])  # Default station position
        self.coordinator = MultiAgentCoordinator(len(self.spacecraft_agents), station_position)
        
        self.logger.info(f"Initialized simulation with {len(self.spacecraft_agents)} spacecraft")
    
    def _create_agent(self, agent_id: str, controller_config: Dict) -> SpacecraftAgent:
        """Create a spacecraft agent"""
        
        class SimulationSpacecraftAgent(SpacecraftAgent):
            def __init__(self, agent_id, config):
                super().__init__(agent_id, config)
                self.controller = DRMPCController(controller_config)
            
            async def update_control(self, dt):
                # Simple proportional control for now
                position_error = self.target_state[:3] - self.state[:3]
                velocity_error = self.target_state[3:6] - self.state[3:6]
                
                # Simple PD control
                thrust = 0.5 * position_error + 0.1 * velocity_error
                torque = np.zeros(3)  # No attitude control for now
                
                return np.concatenate([thrust, torque])
            
            async def communicate(self, message, target_agent=None):
                # Simple message logging
                self.logger.debug(f"Agent {self.agent_id} sending message: {message}")
        
        return SimulationSpacecraftAgent(agent_id, controller_config)
    
    def run(self, duration: float = None, realtime: bool = False) -> SimulationResults:
        """Run the simulation"""
        
        if duration is None:
            duration = self.config.get_mission_params().get('duration', 1800.0)
        
        self.logger.info(f"Starting simulation for {duration} seconds")
        
        start_time = time.time()
        
        while self.current_time < duration:
            # Run simulation step
            self._simulation_step()
            
            # Real-time synchronization
            if realtime:
                elapsed = time.time() - start_time
                if elapsed < self.current_time:
                    time.sleep(self.current_time - elapsed)
            
            # Check termination conditions
            if self._check_termination():
                self.logger.info("Simulation terminated due to mission completion")
                break
                
            self.current_time += self.dt
        
        self.logger.info(f"Simulation completed at t={self.current_time:.2f}s")
        return self.results
    
    def _simulation_step(self):
        """Execute one simulation step"""
        
        # Update all agents
        for agent_id, agent in self.spacecraft_agents.items():
            try:
                # Get control input (simplified synchronous call)
                control = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Placeholder
                
                # Update dynamics
                current_state = agent.state.copy()
                new_state = self.dynamics[agent_id].propagate_dynamics(
                    current_state, control, self.dt
                )
                agent.update_state(new_state)
                
                # Store results
                self.results.timestamps.append(self.current_time)
                self.results.spacecraft_states[agent_id].append(new_state.copy())
                self.results.control_inputs[agent_id].append(control.copy())
                
                # Update performance metrics
                agent.update_performance_metrics(control, self.dt)
                self.results.performance_metrics[agent_id].append(
                    agent.performance_metrics.copy()
                )
                
            except Exception as e:
                self.logger.error(f"Error updating agent {agent_id}: {e}")
    
    def _check_termination(self) -> bool:
        """Check if simulation should terminate early"""
        
        # Check if all agents reached their targets
        all_converged = True
        for agent in self.spacecraft_agents.values():
            position_error = agent.calculate_position_error()
            if position_error > 1.0:  # 1 meter tolerance
                all_converged = False
                break
        
        return all_converged
    
    def get_current_status(self) -> Dict:
        """Get current simulation status"""
        status = {
            'current_time': self.current_time,
            'spacecraft_count': len(self.spacecraft_agents),
            'spacecraft_status': {}
        }
        
        for agent_id, agent in self.spacecraft_agents.items():
            status['spacecraft_status'][agent_id] = agent.get_status()
        
        return status