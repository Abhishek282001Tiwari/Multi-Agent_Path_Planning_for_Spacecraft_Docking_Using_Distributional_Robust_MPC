# src/filters/adaptive_noise_filter.py
import numpy as np
from scipy.linalg import expm
from scipy.stats import chi2

class AdaptiveNoiseFilter:
    """
    Adaptive Extended Kalman Filter for spacecraft sensor spike mitigation
    Based on [^8^] adaptive filtering strategies for spacecraft
    """
    
    def __init__(self, state_dim=13, measurement_dim=6):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        
        # Adaptive parameters
        self.q_scale = 1.0  # Process noise scaling
        self.r_scale = 1.0  # Measurement noise scaling
        self.window_size = 10  # Adaptation window
        
        # Innovation monitoring
        self.innovation_history = []
        self.adaptation_threshold = 3.0
        
    def predict(self, x, P, dt):
        """State prediction with adaptive process noise"""
        
        # State transition matrix (discretized Hill-Clohessy-Wiltshire)
        n = 0.0011  # orbital rate
        F = np.array([
            [4-3*np.cos(n*dt), 0, 0, np.sin(n*dt)/n, 2*(1-np.cos(n*dt))/n, 0],
            [6*(np.sin(n*dt)-n*dt), 1, 0, 2*(np.cos(n*dt)-1)/n, (4*np.sin(n*dt)-3*n*dt)/n, 0],
            [0, 0, np.cos(n*dt), 0, 0, np.sin(n*dt)/n],
            [3*n*np.sin(n*dt), 0, 0, np.cos(n*dt), 2*np.sin(n*dt), 0],
            [6*n*(np.cos(n*dt)-1), 0, 0, -2*np.sin(n*dt), 4*np.cos(n*dt)-3, 0],
            [0, 0, -n*np.sin(n*dt), 0, 0, np.cos(n*dt)]
        ])
        
        # Adaptive process noise
        Q_adaptive = self.compute_adaptive_process_noise()
        
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q_adaptive
        
        return x_pred, P_pred
    
    def update(self, x_pred, P_pred, z, R_nominal):
        """Measurement update with spike detection and mitigation"""
        
        # Measurement matrix
        H = np.eye(self.measurement_dim, self.state_dim)
        
        # Innovation
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R_nominal * self.r_scale
        
        # Spike detection using Mahalanobis distance
        d = np.sqrt(y.T @ np.linalg.inv(S) @ y)
        
        # Adaptive measurement noise
        if d > self.adaptation_threshold:
            # Sensor spike detected - increase measurement noise
            scale_factor = min(d / self.adaptation_threshold, 10.0)
            R_adaptive = R_nominal * (scale_factor ** 2)
            S = H @ P_pred @ H.T + R_adaptive
        else:
            R_adaptive = R_nominal
        
        # Kalman gain
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        # State update
        x_update = x_pred + K @ y
        P_update = (np.eye(self.state_dim) - K @ H) @ P_pred
        
        # Online adaptation of noise statistics
        self.adapt_noise_statistics(y, S)
        
        return x_update, P_update
    
    def adapt_noise_statistics(self, innovation, innovation_cov):
        """Online adaptation of noise statistics based on [^8^]"""
        
        self.innovation_history.append(innovation)
        
        if len(self.innovation_history) > self.window_size:
            # Remove oldest
            self.innovation_history.pop(0)
            
            # Compute sample statistics
            innovations = np.array(self.innovation_history)
            sample_cov = np.cov(innovations.T)
            
            # Covariance matching
            expected_cov = innovation_cov
            ratio = np.trace(sample_cov) / np.trace(expected_cov)
            
            # Smooth adaptation
            if ratio > 1.5:
                self.r_scale = min(self.r_scale * 1.1, 5.0)
            elif ratio < 0.5:
                self.r_scale = max(self.r_scale * 0.9, 0.5)
    
    def compute_adaptive_process_noise(self):
        """Compute adaptive process noise based on current uncertainty"""
        
        # Base process noise
        Q_base = np.diag([
            1e-4, 1e-4, 1e-4,  # position
            1e-5, 1e-5, 1e-5,  # velocity
            1e-6, 1e-6, 1e-6,  # attitude
            1e-7, 1e-7, 1e-7,  # angular velocity
            1e-8                # mass
        ])
        
        return Q_base * self.q_scale

# Integration with spacecraft systems
class SpacecraftWithAdaptiveFiltering:
    def __init__(self, spacecraft_id):
        self.id = spacecraft_id
        self.filter = AdaptiveNoiseFilter()
        self.state = np.zeros(13)
        self.covariance = np.eye(13) * 0.01
        
    def process_measurement(self, raw_measurement):
        """Process sensor measurement with spike filtering"""
        
        # Prediction step
        self.state, self.covariance = self.filter.predict(
            self.state, self.covariance, dt=0.1
        )
        
        # Update step with spike mitigation
        R_measurement = np.diag([0.01**2, 0.01**2, 0.01**2,  # position
                               0.001**2, 0.001**2, 0.001**2])  # velocity
        
        self.state, self.covariance = self.filter.update(
            self.state, self.covariance, raw_measurement, R_measurement
        )
        
        return self.state