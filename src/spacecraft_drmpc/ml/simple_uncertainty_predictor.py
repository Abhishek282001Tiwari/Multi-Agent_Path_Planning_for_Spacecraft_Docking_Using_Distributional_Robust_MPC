# src/ml/simple_uncertainty_predictor.py
import numpy as np
from typing import Dict, List
from collections import deque

class MetaUncertaintyPredictor:
    """
    Simplified uncertainty predictor without ML dependencies
    Uses statistical methods for uncertainty estimation
    """
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.state_history = deque(maxlen=window_size)
        self.error_history = deque(maxlen=window_size)
        self.baseline_uncertainty = 0.1
        
    def update(self, state: np.ndarray, prediction_error: np.ndarray):
        """Update uncertainty predictor with new data"""
        self.state_history.append(state.copy())
        self.error_history.append(prediction_error.copy())
    
    def predict_uncertainty(self, state: np.ndarray) -> np.ndarray:
        """Predict uncertainty for given state"""
        if len(self.error_history) < 5:
            return np.full(state.shape[0], self.baseline_uncertainty)
        
        # Calculate moving statistics
        recent_errors = np.array(list(self.error_history)[-10:])
        uncertainty = np.std(recent_errors, axis=0) + self.baseline_uncertainty
        
        return uncertainty
    
    def get_confidence_bounds(self, prediction: np.ndarray, 
                            uncertainty: np.ndarray) -> tuple:
        """Get confidence bounds for prediction"""
        lower_bound = prediction - 1.96 * uncertainty
        upper_bound = prediction + 1.96 * uncertainty
        return lower_bound, upper_bound