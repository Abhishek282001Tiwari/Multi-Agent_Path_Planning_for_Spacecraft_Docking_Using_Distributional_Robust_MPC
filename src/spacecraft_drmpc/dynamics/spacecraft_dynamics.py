# src/dynamics/spacecraft_dynamics.py
import numpy as np
from scipy.integrate import odeint
from dataclasses import dataclass

@dataclass
class SpacecraftState:
    position: np.ndarray  # [x, y, z] in LVLH frame (m)
    velocity: np.ndarray  # [vx, vy, vz] (m/s)
    attitude: np.ndarray  # Quaternion [q1, q2, q3, q4]
    angular_velocity: np.ndarray  # [wx, wy, wz] (rad/s)
    mass: float  # Current mass (kg)

class SpacecraftDynamics:
    def __init__(self, config):
        self.mass = config['initial_mass']
        self.inertia = np.array(config['inertia_matrix'])
        self.thruster_config = config['thruster_config']
        self.orbital_rate = config.get('orbital_rate', 0.0011)  # rad/s for LEO
        
    def hill_clohessy_wiltshire(self, state, t, control):
        """6-DOF Hill-Clohessy-Wiltshire equations with perturbations"""
        x, y, z, vx, vy, vz, q1, q2, q3, q4, wx, wy, wz = state
        
        n = self.orbital_rate
        
        # Translational dynamics (CW equations)
        dx = vx
        dy = vy
        dz = vz
        
        dvx = 3*n**2*x + 2*n*vy + control[0]/self.mass
        dvy = -2*n*vx + control[1]/self.mass
        dvz = -n**2*z + control[2]/self.mass
        
        # Rotational dynamics (quaternion kinematics)
        q = np.array([q1, q2, q3, q4])
        omega = np.array([wx, wy, wz])
        
        dq = 0.5 * self.quaternion_multiply(q, np.array([0, *omega]))
        
        # Euler's equation for rotation
        tau = control[3:6]  # Control torques
        domega = np.linalg.inv(self.inertia) @ (
            tau - np.cross(omega, self.inertia @ omega)
        )
        
        return np.array([dx, dy, dz, dvx, dvy, dvz, *dq, *domega])
    
    def quaternion_multiply(self, q1, q2):
        """Quaternion multiplication"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def fuel_consumption(self, thrust, dt):
        """Calculate fuel mass consumed based on thrust magnitude"""
        Isp = self.thruster_config.get('specific_impulse', 300)
        g0 = 9.81
        thrust_mag = np.linalg.norm(thrust[:3])
        if thrust_mag > 0:
            mdot = thrust_mag / (Isp * g0)
            return mdot * dt
        return 0.0
    
    def propagate_dynamics(self, state, control, dt):
        """Propagate spacecraft dynamics forward by dt seconds"""
        # Simple Euler integration for now
        def dynamics(s, t):
            return self.hill_clohessy_wiltshire(s, t, control)
        
        # Integrate over small time step
        t_span = [0, dt]
        result = odeint(dynamics, state, t_span)
        
        return result[-1]  # Return final state