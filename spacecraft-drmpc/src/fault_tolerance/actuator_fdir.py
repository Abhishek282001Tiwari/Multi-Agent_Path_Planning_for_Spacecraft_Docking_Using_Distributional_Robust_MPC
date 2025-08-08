# src/fault_tolerance/actuator_fdir.py
import numpy as np
from enum import Enum
from dataclasses import dataclass
import threading
import time
from typing import Dict
import logging

from ..agents.spacecraft_agent import SpacecraftAgent

class FaultType(Enum):
    THRUSTER_STUCK = "stuck"
    THRUSTER_DEGRADED = "degraded"
    THRUSTER_COMPLETE_FAILURE = "complete_failure"
    SENSOR_DRIFT = "sensor_drift"
    COMMUNICATION_LOSS = "communication_loss"

@dataclass
class FaultStatus:
    type: FaultType
    severity: float  # 0.0 to 1.0
    detected_time: float
    estimated_impact: Dict[str, float]

class ActuatorFDIR:
    """Comprehensive fault tolerance system for spacecraft actuators"""
    
    def __init__(self, num_thrusters=12, redundancy_level=2):
        self.num_thrusters = num_thrusters
        self.redundancy_level = redundancy_level
        
        # Thruster configuration matrix
        self.thruster_config = self._initialize_thruster_config()
        
        # Fault detection parameters
        self.fault_threshold = 0.1
        self.detection_window = 5.0  # seconds
        self.health_history = deque(maxlen=100)
        
        # Monitoring threads
        self.monitoring_active = False
        self.monitoring_thread = None
        
    def _initialize_thruster_config(self):
        """Initialize thruster configuration matrix"""
        
        # 12 thrusters in standard configuration
        # [x, y, z, torque_x, torque_y, torque_z] for each thruster
        config = np.array([
            [1, 0, 0, 0, 0.1, 0],    # +X thruster
            [-1, 0, 0, 0, -0.1, 0],  # -X thruster
            [0, 1, 0, -0.1, 0, 0],    # +Y thruster
            [0, -1, 0, 0.1, 0, 0],    # -Y thruster
            [0, 0, 1, 0, 0, 0.1],     # +Z thruster
            [0, 0, -1, 0, 0, -0.1],   # -Z thruster
            # Additional thrusters for redundancy
            [0.707, 0.707, 0, -0.071, 0.071, 0],  # Diagonal thrusters
            [0.707, -0.707, 0, 0.071, 0.071, 0],
            [0.707, 0, 0.707, 0, -0.071, 0.071],
            [0.707, 0, -0.707, 0, 0.071, 0.071],
            [0, 0.707, 0.707, -0.071, 0, 0.071],
            [0, 0.707, -0.707, 0.071, 0, 0.071]
        ])
        
        return config
    
    def detect_faults(self, commanded_thrust, actual_thrust, sensor_data):
        """Real-time fault detection using residual analysis"""
        
        residuals = commanded_thrust - actual_thrust
        
        # Statistical fault detection
        fault_detected = False
        fault_type = None
        severity = 0.0
        
        # Thruster stuck fault
        if np.max(np.abs(residuals)) > self.fault_threshold * np.max(np.abs(commanded_thrust)):
            if np.std(actual_thrust) < 0.01:  # Very low variation
                fault_type = FaultType.THRUSTER_STUCK
                severity = np.max(np.abs(residuals)) / np.max(np.abs(commanded_thrust))
                fault_detected = True
        
        # Thruster degraded performance
        elif np.mean(np.abs(residuals)) > 0.05 * np.mean(np.abs(commanded_thrust)):
            fault_type = FaultType.THRUSTER_DEGRADED
            severity = np.mean(np.abs(residuals)) / np.mean(np.abs(commanded_thrust))
            fault_detected = True
        
        if fault_detected:
            return FaultStatus(
                type=fault_type,
                severity=severity,
                detected_time=time.time(),
                estimated_impact=self.estimate_fault_impact(fault_type, severity)
            )
        
        return None
    
    def estimate_fault_impact(self, fault_type: FaultType, severity: float) -> Dict[str, float]:
        """Estimate impact of detected fault on system performance"""
        
        impact = {
            'control_authority_loss': 0.0,
            'fuel_penalty': 0.0,
            'mission_success_probability': 1.0
        }
        
        if fault_type == FaultType.THRUSTER_STUCK:
            impact['control_authority_loss'] = severity * 0.1
            impact['fuel_penalty'] = 1.2  # 20% more fuel needed
            impact['mission_success_probability'] = 0.95
            
        elif fault_type == FaultType.THRUSTER_DEGRADED:
            impact['control_authority_loss'] = severity * 0.3
            impact['fuel_penalty'] = 1.1  # 10% more fuel needed
            impact['mission_success_probability'] = 0.98
            
        return impact
    
    def reconfigure_control_allocation(self, fault_status: FaultStatus):
        """Reconfigure control allocation matrix for fault tolerance"""
        
        if fault_status.type == FaultType.THRUSTER_STUCK:
            # Identify failed thruster
            failed_thruster = self.identify_failed_thruster()
            
            # Create reduced configuration matrix
            reduced_config = np.delete(self.thruster_config, failed_thruster, axis=0)
            
            # Compute pseudo-inverse for control allocation
            allocation_matrix = np.linalg.pinv(reduced_config.T @ reduced_config) @ reduced_config.T
            
            return allocation_matrix
            
        return self.thruster_config
    
    def start_monitoring(self):
        """Start continuous fault monitoring"""
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
    
    def _monitor_loop(self):
        """Continuous monitoring loop"""
        
        while self.monitoring_active:
            # Check thruster health
            health_status = self.check_thruster_health()
            
            # Update health history
            self.health_history.append(health_status)
            
            # Detect trends
            if len(self.health_history) >= 10:
                self.analyze_health_trends()
            
            time.sleep(0.1)  # 10Hz monitoring
    
    def check_thruster_health(self):
        """Check overall thruster health status"""
        
        # Placeholder for actual health checking
        return {
            'timestamp': time.time(),
            'thruster_status': np.ones(self.num_thrusters),
            'performance_metrics': np.random.rand(self.num_thrusters)
        }

# Fault-tolerant spacecraft integration
class FaultTolerantSpacecraft(SpacecraftAgent):
    def __init__(self, agent_id):
        super().__init__(agent_id)
        self.fdir_system = ActuatorFDIR()
        self.thruster_health = np.ones(12)
        self.redundancy_active = False
        
    def handle_faulty_actuator(self, fault_status):
        """Handle actuator fault with graceful degradation"""
        
        logging.warning(f"Fault detected: {fault_status}")
        
        # Reconfigure control allocation
        new_allocation = self.fdir_system.reconfigure_control_allocation(
            fault_status
        )
        
        # Update MPC constraints
        self.dr_mpc_controller.update_actuator_constraints(
            new_allocation,
            fault_status.estimated_impact
        )
        
        # Notify other spacecraft
        self.broadcast_fault_status(fault_status)
        
        return new_allocation