# src/testing/real_time_validator.py
import time
import numpy as np
from src.controllers.dr_mpc_controller import DRMPCController

class RealTimeValidator:
    def __init__(self):
        self.timing_stats = []
        self.success_count = 0
        
    def validate_mpc_timing(self, scenario):
        """Validate MPC solve times under real-time constraints"""
        
        for cycle in range(1000):
            start_time = time.perf_counter()
            
            # Run MPC for all spacecraft
            for spacecraft in scenario.spacecraft:
                control, success = spacecraft.mpc_controller.solve(
                    spacecraft.state, 
                    spacecraft.target,
                    spacecraft.uncertainty
                )
                
                if success:
                    self.success_count += 1
            
            solve_time = (time.perf_counter() - start_time) * 1000  # ms
            
            # Real-time check
            if solve_time > 50:  # 50ms deadline
                self.log_violation(cycle, solve_time)
            
            self.timing_stats.append(solve_time)
            
        # Calculate statistics
        avg_time = np.mean(self.timing_stats)
        success_rate = self.success_count / (1000 * 3)  # 3 spacecraft
        
        return {
            'average_solve_time': avg_time,
            'success_rate': success_rate,
            'max_time': np.max(self.timing_stats),
            'min_time': np.min(self.timing_stats)
        }

# Current validation results
validator = RealTimeValidator()
results = validator.validate_mpc_timing(current_scenario)
print(f"Average MPC solve time: {results['average_solve_time']:.1f}ms")
print(f"Real-time success rate: {results['success_rate']*100:.1f}%")