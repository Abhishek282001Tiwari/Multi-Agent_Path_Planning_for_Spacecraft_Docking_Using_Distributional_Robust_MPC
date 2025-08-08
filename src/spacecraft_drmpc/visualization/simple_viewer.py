# src/visualization/simple_viewer.py
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Optional

class LiveViewer:
    """Simple 2D visualization for spacecraft docking simulation"""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.simulator = None
        self.logger = logging.getLogger('LiveViewer')
        
        # Plotting setup
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlabel('X Position (m)')
        self.ax.set_ylabel('Y Position (m)')
        self.ax.set_title('Spacecraft Docking Simulation')
        self.ax.grid(True)
        
        # Color map for different spacecraft
        self.colors = ['red', 'blue', 'green', 'orange', 'purple']
        
    def connect(self, simulator):
        """Connect viewer to simulator"""
        self.simulator = simulator
        self.logger.info("Viewer connected to simulator")
    
    def update_plot(self):
        """Update the visualization"""
        if self.simulator is None:
            return
        
        self.ax.clear()
        self.ax.set_xlabel('X Position (m)')
        self.ax.set_ylabel('Y Position (m)')
        self.ax.set_title(f'Spacecraft Docking Simulation (t={self.simulator.current_time:.1f}s)')
        self.ax.grid(True)
        
        # Plot spacecraft positions
        for i, (agent_id, agent) in enumerate(self.simulator.spacecraft_agents.items()):
            position = agent.get_position()
            target_position = agent.target_state[:3]
            
            color = self.colors[i % len(self.colors)]
            
            # Current position
            self.ax.scatter(position[0], position[1], 
                          c=color, s=100, marker='o', label=f'{agent_id} (current)')
            
            # Target position
            self.ax.scatter(target_position[0], target_position[1], 
                          c=color, s=100, marker='x', label=f'{agent_id} (target)')
            
            # Trajectory line
            self.ax.plot([position[0], target_position[0]], 
                        [position[1], target_position[1]], 
                        c=color, linestyle='--', alpha=0.5)
        
        self.ax.legend()
        self.ax.axis('equal')
        plt.draw()
        plt.pause(0.01)
    
    def show_final_results(self, results):
        """Show final simulation results"""
        if not results.spacecraft_states:
            return
        
        # Create trajectory plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Position trajectories
        for i, (agent_id, states) in enumerate(results.spacecraft_states.items()):
            if not states:
                continue
                
            states_array = np.array(states)
            color = self.colors[i % len(self.colors)]
            
            ax1.plot(states_array[:, 0], states_array[:, 1], 
                    c=color, label=f'{agent_id}', linewidth=2)
            
            # Mark start and end
            ax1.scatter(states_array[0, 0], states_array[0, 1], 
                       c=color, s=100, marker='o', edgecolor='black')
            ax1.scatter(states_array[-1, 0], states_array[-1, 1], 
                       c=color, s=100, marker='s', edgecolor='black')
        
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('Spacecraft Trajectories')
        ax1.grid(True)
        ax1.legend()
        ax1.axis('equal')
        
        # Position errors over time
        for i, (agent_id, metrics_list) in enumerate(results.performance_metrics.items()):
            if not metrics_list:
                continue
                
            errors = [m.get('position_error', 0) for m in metrics_list]
            timestamps = results.timestamps[:len(errors)]
            color = self.colors[i % len(self.colors)]
            
            ax2.plot(timestamps, errors, c=color, label=f'{agent_id}', linewidth=2)
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position Error (m)')
        ax2.set_title('Position Error vs Time')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        self.logger.info("Final results visualization displayed")
    
    def save_plot(self, filename: str):
        """Save current plot to file"""
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        self.logger.info(f"Plot saved to {filename}")