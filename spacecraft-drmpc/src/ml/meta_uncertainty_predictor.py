# src/ml/meta_uncertainty_predictor.py
import torch
import torch.nn as nn
import numpy as np
from collections import deque

class MetaUncertaintyNetwork(nn.Module):
    """
    Meta-learning network for dynamic uncertainty prediction
    Based on LSTM architecture for sequential learning
    """
    
    def __init__(self, input_dim=18, hidden_dim=128, output_dim=13):
        super().__init__()
        
        # LSTM layers for temporal dynamics
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim//2, batch_first=True)
        
        # Uncertainty prediction heads
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim//2, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
        # Meta-learning adaptation layer
        self.meta_adapter = nn.Linear(output_dim, output_dim)
        
    def forward(self, sequence, context_state=None):
        """Forward pass for uncertainty prediction"""
        
        # Process temporal sequence
        lstm_out, _ = self.lstm1(sequence)
        lstm_out, (hidden, _) = self.lstm2(lstm_out)
        
        # Get last output
        final_hidden = lstm_out[:, -1, :]
        
        # Predict uncertainty
        base_uncertainty = self.uncertainty_head(final_hidden)
        
        # Meta-adapt based on context
        if context_state is not None:
            adapted_uncertainty = base_uncertainty * torch.sigmoid(
                self.meta_adapter(context_state)
            )
        else:
            adapted_uncertainty = base_uncertainty
            
        return adapted_uncertainty

class MetaUncertaintyPredictor:
    """Complete ML system for uncertainty prediction"""
    
    def __init__(self, model_path=None):
        self.network = MetaUncertaintyNetwork()
        self.memory = deque(maxlen=1000)  # Experience replay
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
        
        if model_path:
            self.load_model(model_path)
    
    def prepare_features(self, spacecraft_state, environment_data):
        """Prepare features for uncertainty prediction"""
        
        features = np.concatenate([
            spacecraft_state,  # 13D state
            environment_data['solar_pressure'],
            environment_data['atmospheric_density'],
            environment_data['magnetic_field'],
            environment_data['timestamp']
        ])
        
        return features
    
    def predict_uncertainty(self, state_sequence, environment_context):
        """Predict dynamic uncertainty bounds"""
        
        with torch.no_grad():
            sequence_tensor = torch.FloatTensor([state_sequence])
            context_tensor = torch.FloatTensor([environment_context])
            
            uncertainty = self.network(sequence_tensor, context_tensor)
            
        return uncertainty.numpy()[0]
    
    def online_update(self, true_state, predicted_uncertainty, actual_error):
        """Online learning from actual performance"""
        
        # Compute loss
        predicted_error = predicted_uncertainty
        actual_error_norm = np.linalg.norm(actual_error)
        
        loss = torch.nn.functional.mse_loss(
            torch.FloatTensor([predicted_error]),
            torch.FloatTensor([actual_error_norm])
        )
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Store experience
        self.memory.append({
            'state': true_state,
            'predicted': predicted_error,
            'actual': actual_error_norm
        })
    
    def save_model(self, path):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'memory': list(self.memory)
        }, path)
    
    def load_model(self, path):
        """Load trained model"""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.memory = deque(checkpoint['memory'])

# Real-time integration
class MLEnhancedSpacecraft(SpacecraftAgent):
    def __init__(self, agent_id, dynamics_model, dr_mpc_controller):
        super().__init__(agent_id, dynamics_model, dr_mpc_controller)
        self.uncertainty_predictor = MetaUncertaintyPredictor()
        self.state_history = deque(maxlen=50)
        
    def update_uncertainty_model(self, current_state, environment_data):
        """Update uncertainty model with ML predictions"""
        
        # Prepare features
        features = self.uncertainty_predictor.prepare_features(
            current_state, environment_data
        )
        
        # Add to history
        self.state_history.append(features)
        
        if len(self.state_history) >= 10:
            # Predict uncertainty
            uncertainty = self.uncertainty_predictor.predict_uncertainty(
                list(self.state_history),
                environment_data
            )
            
            # Update DR-MPC uncertainty bounds
            self.dr_mpc_controller.set_dynamic_uncertainty(uncertainty)