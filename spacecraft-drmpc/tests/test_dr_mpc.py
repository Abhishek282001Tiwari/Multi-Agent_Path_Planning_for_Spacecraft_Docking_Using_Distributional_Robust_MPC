# tests/test_dr_mpc.py
import pytest
import numpy as np
from src.controllers.dr_mpc_controller import DRMPCController

class TestDRMPC:
    @pytest.fixture
    def controller(self):
        config = {
            'prediction_horizon': 10,
            'time_step': 0.1,
            'wasserstein_radius': 0.1,
            'confidence_level': 0.95,
            'max_thrust': 10.0,
            'max_torque': 1.0,
            'safety_radius': 5.0
        }
        return DRMPCController(config)
    
    def test_nominal_case(self, controller):
        x0 = np.zeros(13)
        x0[0] = 100  # 100m away in x-direction
        reference = np.zeros(13)
        
        uncertainty = {
            'state_covariance': np.eye(13) * 0.01,
            'process_noise': np.eye(6) * 0.001
        }
        
        control, cost = controller.formulate_optimization(x0, reference, uncertainty)
        
        assert control.shape == (6,)
        assert np.isfinite(cost)
        assert np.all(np.abs(control[:3]) <= 10.0)