# Multi-Agent Spacecraft Docking User Manual

Welcome to the comprehensive user manual for the Multi-Agent Spacecraft Docking System with Distributionally Robust Model Predictive Control (DR-MPC). This manual provides step-by-step instructions for mission planning, system configuration, and operational procedures.

## Table of Contents
- [Mission Planning Guide](#mission-planning-guide)
- [Configuration Management](#configuration-management)
- [Scenario Setup Instructions](#scenario-setup-instructions)
- [System Operation](#system-operation)
- [Visualization and Monitoring](#visualization-and-monitoring)
- [Performance Tuning](#performance-tuning)
- [Common Use Cases](#common-use-cases)
- [Troubleshooting Guide](#troubleshooting-guide)

---

## Mission Planning Guide

### Step 1: Mission Objectives Definition

Before setting up any simulation, clearly define your mission objectives:

#### Single Spacecraft Missions
```python
mission_objectives = {
    'primary_objective': 'autonomous_docking',
    'target_spacecraft': 'ISS_station',
    'approach_corridor': 'v_bar',  # velocity vector approach
    'docking_port': 'nadir_port',
    'mission_duration': 1800,  # 30 minutes
    'precision_requirements': {
        'position_tolerance': 0.05,  # meters
        'attitude_tolerance': 0.5,   # degrees
        'velocity_tolerance': 0.01   # m/s
    }
}
```

#### Multi-Agent Missions
```python
formation_mission = {
    'primary_objective': 'formation_flying',
    'formation_type': 'triangular',
    'leader_spacecraft': 'alpha-leader',
    'follower_spacecraft': ['beta-001', 'gamma-002'],
    'formation_parameters': {
        'baseline_distance': 50.0,  # meters
        'formation_plane': 'orbital',
        'configuration_changes': [
            {'time': 600, 'formation': 'line'},
            {'time': 1200, 'formation': 'diamond'}
        ]
    }
}
```

### Step 2: Mission Timeline Development

Create a detailed mission timeline with key phases:

```python
class MissionTimeline:
    def __init__(self):
        self.phases = []
    
    def add_phase(self, name, start_time, duration, parameters):
        phase = {
            'name': name,
            'start_time': start_time,
            'duration': duration,
            'parameters': parameters
        }
        self.phases.append(phase)
    
    def create_docking_timeline(self):
        """Create standard docking mission timeline"""
        
        # Phase 1: Approach initiation
        self.add_phase('approach_initiation', 0, 300, {
            'target_distance': 1000.0,  # 1 km
            'approach_velocity': -0.5,   # m/s
            'guidance_mode': 'coarse'
        })
        
        # Phase 2: Far field approach
        self.add_phase('far_field_approach', 300, 600, {
            'target_distance': 200.0,   # 200 m
            'approach_velocity': -0.2,   # m/s
            'guidance_mode': 'refined'
        })
        
        # Phase 3: Close approach
        self.add_phase('close_approach', 900, 600, {
            'target_distance': 50.0,    # 50 m
            'approach_velocity': -0.1,   # m/s
            'guidance_mode': 'precision'
        })
        
        # Phase 4: Final approach
        self.add_phase('final_approach', 1500, 240, {
            'target_distance': 5.0,     # 5 m
            'approach_velocity': -0.05,  # m/s
            'guidance_mode': 'terminal'
        })
        
        # Phase 5: Contact and capture
        self.add_phase('contact_capture', 1740, 60, {
            'target_distance': 0.0,     # Contact
            'approach_velocity': -0.01,  # m/s
            'guidance_mode': 'contact'
        })

# Usage example
timeline = MissionTimeline()
timeline.create_docking_timeline()
```

### Step 3: Risk Assessment and Safety Planning

Identify potential risks and develop mitigation strategies:

```python
risk_assessment = {
    'collision_risk': {
        'probability': 'low',
        'impact': 'catastrophic',
        'mitigation': [
            'implement_collision_avoidance',
            'maintain_safe_corridors',
            'emergency_abort_capability'
        ]
    },
    'communication_failure': {
        'probability': 'medium',
        'impact': 'high',
        'mitigation': [
            'autonomous_operation_mode',
            'redundant_communication_links',
            'predefined_safe_trajectories'
        ]
    },
    'thruster_failure': {
        'probability': 'medium',
        'impact': 'medium',
        'mitigation': [
            'fault_tolerant_control',
            'actuator_redundancy',
            'graceful_degradation'
        ]
    }
}
```

---

## Configuration Management

### Master Configuration File Structure

The system uses a hierarchical configuration structure:

```yaml
# config/mission_config.yaml
mission:
  name: "ISS_Docking_Mission_001"
  type: "single_spacecraft_docking"
  duration: 1800  # seconds
  real_time: false
  
spacecraft:
  chaser:
    agent_id: "chaser-001"
    initial_conditions:
      position: [-1000.0, 0.0, 0.0]  # LVLH frame, meters
      velocity: [0.5, 0.0, 0.0]       # m/s
      attitude: [1.0, 0.0, 0.0, 0.0]  # quaternion [w,x,y,z]
      angular_velocity: [0.0, 0.0, 0.0]  # rad/s
    
    physical_properties:
      mass: 500.0  # kg
      inertia_matrix: [
        [100.0, 0.0, 0.0],
        [0.0, 150.0, 0.0],
        [0.0, 0.0, 120.0]
      ]  # kg⋅m²
      
    thrusters:
      configuration: "12_thruster_config"
      max_thrust_per_thruster: 10.0  # N
      min_pulse_width: 0.02  # seconds
      
  target:
    agent_id: "ISS-station"
    initial_conditions:
      position: [0.0, 0.0, 0.0]
      velocity: [0.0, 0.0, 0.0]
      attitude: [1.0, 0.0, 0.0, 0.0]
      angular_velocity: [0.0, 0.0, 0.0]
    
    docking_ports:
      - name: "nadir_port"
        position: [0.0, 0.0, -2.0]  # relative to spacecraft center
        orientation: [0.0, 0.0, -1.0]  # docking direction
        
controller:
  type: "DR_MPC"
  prediction_horizon: 20
  time_step: 0.1
  
  optimization:
    solver: "MOSEK"
    max_iterations: 1000
    tolerance: 1e-6
    warm_start: true
    
  uncertainty:
    wasserstein_radius: 0.1
    confidence_level: 0.95
    uncertainty_types: ["parametric", "environmental"]
    
  constraints:
    position_bounds: [-2000.0, 2000.0]  # meters
    velocity_bounds: [-5.0, 5.0]        # m/s
    thrust_bounds: [0.0, 100.0]         # N total
    
environment:
  reference_frame: "LVLH"  # Local Vertical Local Horizontal
  orbital_parameters:
    altitude: 408000  # meters (ISS altitude)
    inclination: 51.6  # degrees
    eccentricity: 0.0003
    
  disturbances:
    atmospheric_drag: true
    solar_radiation_pressure: true
    gravitational_perturbations: true
    
simulation:
  integrator: "RK45"
  absolute_tolerance: 1e-8
  relative_tolerance: 1e-6
  max_step_size: 0.1
  
  logging:
    save_trajectory: true
    save_controls: true
    save_performance_metrics: true
    output_format: "HDF5"
    
visualization:
  enabled: true
  real_time_plots: true
  plot_types: ["trajectory", "attitude", "controls"]
  update_frequency: 10  # Hz
```

### Spacecraft-Specific Configuration

Each spacecraft type can have its own configuration template:

```python
# config/spacecraft_templates/small_chaser.py
SMALL_CHASER_CONFIG = {
    'physical_properties': {
        'mass': 180.0,  # kg
        'inertia_matrix': np.diag([25.0, 30.0, 35.0]),  # kg⋅m²
        'drag_coefficient': 2.2,
        'reference_area': 4.0,  # m²
        'reflectivity_coefficient': 1.3
    },
    
    'propulsion_system': {
        'thruster_configuration': 'quad_config',
        'num_thrusters': 8,
        'max_thrust_per_thruster': 5.0,  # N
        'specific_impulse': 220.0,  # seconds
        'fuel_capacity': 20.0  # kg
    },
    
    'sensors': {
        'gps': {
            'position_accuracy': 0.5,  # meters
            'velocity_accuracy': 0.1   # m/s
        },
        'imu': {
            'gyro_noise': 1e-5,        # rad/s/√Hz
            'accel_noise': 1e-4        # m/s²/√Hz
        },
        'camera': {
            'resolution': [1024, 768],
            'field_of_view': 60,       # degrees
            'range': [1.0, 1000.0]     # meters
        }
    }
}
```

### Environment Configuration

Configure environmental conditions and disturbances:

```python
# config/environment_config.py
class EnvironmentConfig:
    def __init__(self, orbit_type='LEO'):
        self.orbit_type = orbit_type
        self.setup_orbit_parameters()
        self.setup_disturbances()
    
    def setup_orbit_parameters(self):
        if self.orbit_type == 'LEO':
            self.orbital_parameters = {
                'semi_major_axis': 6786000,  # meters (408 km altitude)
                'eccentricity': 0.0003,
                'inclination': 51.6,         # degrees
                'right_ascension': 0.0,      # degrees
                'argument_perigee': 0.0,     # degrees
                'true_anomaly': 0.0          # degrees
            }
        elif self.orbit_type == 'GEO':
            self.orbital_parameters = {
                'semi_major_axis': 42164000,  # meters
                'eccentricity': 0.0001,
                'inclination': 0.1,          # degrees
                'right_ascension': 0.0,
                'argument_perigee': 0.0,
                'true_anomaly': 0.0
            }
    
    def setup_disturbances(self):
        if self.orbit_type == 'LEO':
            self.disturbances = {
                'atmospheric_drag': {
                    'enabled': True,
                    'density_model': 'harris_priester',
                    'density_uncertainty': 0.3  # ±30%
                },
                'solar_radiation_pressure': {
                    'enabled': True,
                    'solar_constant': 1361,      # W/m²
                    'eclipse_model': True
                },
                'gravitational_perturbations': {
                    'enabled': True,
                    'degree_order': [20, 20],    # Earth gravity model
                    'third_body_sun': True,
                    'third_body_moon': True
                }
            }
        else:  # GEO
            self.disturbances = {
                'atmospheric_drag': {'enabled': False},
                'solar_radiation_pressure': {
                    'enabled': True,
                    'solar_constant': 1361,
                    'eclipse_model': True
                },
                'gravitational_perturbations': {
                    'enabled': True,
                    'degree_order': [8, 8],
                    'third_body_sun': True,
                    'third_body_moon': True
                }
            }
```

---

## Scenario Setup Instructions

### Scenario 1: Single Spacecraft Autonomous Docking

#### Step-by-Step Setup

1. **Create Mission Directory**
```bash
mkdir -p missions/iss_docking_demo
cd missions/iss_docking_demo
```

2. **Configure Mission Parameters**
```python
# missions/iss_docking_demo/mission_config.py
from src.utils.mission_config import MissionConfig

class ISSDocingDemo(MissionConfig):
    def __init__(self):
        super().__init__('single_spacecraft_docking')
        self._setup_iss_docking_scenario()
    
    def _setup_iss_docking_scenario(self):
        # Chaser spacecraft configuration
        self.add_spacecraft('chaser-001', {
            'type': 'cygnus_cargo',
            'initial_position': [-500.0, 0.0, 0.0],  # 500m behind ISS
            'initial_velocity': [0.2, 0.0, 0.0],     # approaching at 0.2 m/s
            'mass': 7500.0,  # kg (Cygnus cargo vehicle)
            'target_port': 'unity_nadir'
        })
        
        # ISS target configuration
        self.add_spacecraft('ISS-station', {
            'type': 'space_station',
            'initial_position': [0.0, 0.0, 0.0],
            'initial_velocity': [0.0, 0.0, 0.0],
            'mass': 420000.0,  # kg (ISS mass)
            'attitude_control': 'station_keeping'
        })
        
        # Mission timeline
        self.mission_phases = [
            {
                'name': 'far_approach',
                'duration': 900,  # 15 minutes
                'guidance_mode': 'coarse',
                'target_distance': 200.0
            },
            {
                'name': 'close_approach',
                'duration': 600,  # 10 minutes
                'guidance_mode': 'refined',
                'target_distance': 50.0
            },
            {
                'name': 'final_approach',
                'duration': 300,  # 5 minutes
                'guidance_mode': 'precision',
                'target_distance': 0.0
            }
        ]
```

3. **Launch Simulation**
```bash
# From project root directory
python3 main.py --config missions/iss_docking_demo/mission_config.py \
                --visualize \
                --duration 1800 \
                --output results/iss_docking_demo
```

### Scenario 2: Three Spacecraft Coordinated Operations

#### Configuration Setup

```python
# missions/three_spacecraft_demo/mission_config.py
class ThreeSpacecraftDemo(MissionConfig):
    def __init__(self):
        super().__init__('multi_spacecraft_coordination')
        self._setup_coordination_scenario()
    
    def _setup_coordination_scenario(self):
        # Primary chaser
        self.add_spacecraft('alpha-chaser', {
            'role': 'primary',
            'initial_position': [-300.0, -50.0, 0.0],
            'target_position': [-10.0, 0.0, 0.0],
            'approach_corridor': 'v_bar'
        })
        
        # Secondary chaser
        self.add_spacecraft('beta-chaser', {
            'role': 'secondary', 
            'initial_position': [-300.0, 50.0, 0.0],
            'target_position': [10.0, 0.0, 0.0],
            'approach_corridor': 'r_bar'
        })
        
        # Observer spacecraft
        self.add_spacecraft('gamma-observer', {
            'role': 'observer',
            'initial_position': [0.0, 0.0, 100.0],
            'target_position': [0.0, 0.0, 50.0],
            'mission': 'documentation'
        })
        
        # Target space station
        self.add_spacecraft('target-station', {
            'role': 'target',
            'initial_position': [0.0, 0.0, 0.0],
            'docking_ports': ['port_nadir', 'port_zenith']
        })
        
        # Coordination parameters
        self.coordination_config = {
            'communication_topology': 'mesh',
            'consensus_algorithm': 'distributed_averaging',
            'conflict_resolution': 'priority_based',
            'safety_margins': {
                'min_separation': 20.0,  # meters
                'communication_range': 1000.0,  # meters
                'emergency_abort_distance': 5.0  # meters
            }
        }
```

### Scenario 3: Formation Flying Mission

#### Advanced Formation Configuration

```python
# missions/formation_demo/formation_config.py
class FormationFlyingDemo(MissionConfig):
    def __init__(self):
        super().__init__('formation_flying')
        self._setup_formation_scenario()
    
    def _setup_formation_scenario(self):
        # Formation leader
        self.add_spacecraft('formation-leader', {
            'role': 'leader',
            'initial_position': [0.0, 0.0, 0.0],
            'control_authority': 'high',
            'communication_master': True
        })
        
        # Formation followers in triangular pattern
        follower_positions = self._generate_triangular_formation(
            center=[0.0, 0.0, 0.0],
            radius=50.0,
            num_spacecraft=3
        )
        
        for i, position in enumerate(follower_positions):
            self.add_spacecraft(f'follower-{i+1:02d}', {
                'role': 'follower',
                'initial_position': position,
                'formation_index': i,
                'leader_id': 'formation-leader'
            })
        
        # Formation maneuvers timeline
        self.formation_maneuvers = [
            {
                'time': 0,
                'maneuver': 'maintain_formation',
                'formation_type': 'triangle',
                'parameters': {'radius': 50.0}
            },
            {
                'time': 600,  # 10 minutes
                'maneuver': 'reconfigure_formation',
                'formation_type': 'line',
                'parameters': {'spacing': 30.0, 'orientation': 'along_track'}
            },
            {
                'time': 1200,  # 20 minutes
                'maneuver': 'precision_approach',
                'formation_type': 'stacked',
                'parameters': {'vertical_spacing': 10.0}
            }
        ]
    
    def _generate_triangular_formation(self, center, radius, num_spacecraft):
        """Generate triangular formation positions"""
        positions = []
        for i in range(num_spacecraft):
            angle = 2 * np.pi * i / num_spacecraft
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = center[2]
            positions.append([x, y, z])
        return positions
```

---

## System Operation

### Pre-Flight Checklist

Before running any simulation, complete this comprehensive checklist:

#### System Verification
- [ ] **Python Environment**: Version 3.9+ activated
- [ ] **Dependencies**: All packages installed (`pip list | grep -E "(numpy|scipy|cvxpy|matplotlib)"`)
- [ ] **Solver License**: MOSEK license valid (if using commercial solver)
- [ ] **Configuration Files**: Mission config validated
- [ ] **Output Directory**: Write permissions verified

#### Mission Configuration Review
- [ ] **Spacecraft Parameters**: Mass, inertia, thruster specifications reviewed
- [ ] **Initial Conditions**: Positions and velocities within reasonable bounds
- [ ] **Mission Timeline**: All phases properly sequenced
- [ ] **Safety Parameters**: Collision avoidance margins set appropriately
- [ ] **Performance Requirements**: Success criteria defined

#### Computational Resources
- [ ] **Memory**: Sufficient RAM available (4GB minimum, 16GB recommended)
- [ ] **CPU**: Multi-core processor available for real-time operations
- [ ] **Disk Space**: Adequate storage for trajectory and performance data
- [ ] **Network**: Communication ports available (for distributed simulations)

### Mission Execution Procedure

#### Phase 1: Initialization
```bash
# Navigate to project directory
cd /path/to/spacecraft-drmpc

# Activate environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Verify system status
python3 -c "
import sys
sys.path.insert(0, '.')
from src.utils.system_check import SystemCheck
checker = SystemCheck()
checker.run_full_system_check()
"
```

#### Phase 2: Mission Loading
```python
#!/usr/bin/env python3
"""Mission execution script"""

import sys
import logging
from src.utils.mission_config import MissionConfig
from src.simulations.docking_simulator import DockingSimulator
from src.visualization.mission_monitor import MissionMonitor

def execute_mission(config_file, visualization=True):
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('MissionController')
    
    try:
        # Load mission configuration
        logger.info("Loading mission configuration...")
        config = MissionConfig.load_from_file(config_file)
        logger.info(f"Mission: {config.mission_name}")
        logger.info(f"Duration: {config.mission_duration} seconds")
        logger.info(f"Spacecraft count: {len(config.spacecraft_configs)}")
        
        # Initialize simulator
        logger.info("Initializing simulation environment...")
        simulator = DockingSimulator(config)
        
        # Setup visualization (if enabled)
        monitor = None
        if visualization:
            logger.info("Initializing mission monitor...")
            monitor = MissionMonitor(config)
            monitor.setup_displays()
        
        # Pre-flight verification
        logger.info("Running pre-flight verification...")
        verification_result = simulator.run_preflight_check()
        if not verification_result.all_systems_nominal:
            logger.error("Pre-flight check failed!")
            for error in verification_result.errors:
                logger.error(f"  - {error}")
            return False
        
        logger.info("Pre-flight check passed ✓")
        
        # Mission execution
        logger.info("Starting mission execution...")
        results = simulator.run(
            duration=config.mission_duration,
            realtime=config.realtime_mode,
            monitor=monitor
        )
        
        # Mission completion analysis
        logger.info("Mission completed. Analyzing results...")
        success_metrics = analyze_mission_success(results, config)
        
        # Report results
        for agent_id, metrics in success_metrics.items():
            logger.info(f"{agent_id}: {metrics['status']}")
            if metrics['status'] == 'SUCCESS':
                logger.info(f"  Final position error: {metrics['position_error']:.3f} m")
                logger.info(f"  Final attitude error: {metrics['attitude_error']:.3f}°")
                logger.info(f"  Fuel consumption: {metrics['fuel_used']:.2f} kg")
        
        return True
        
    except Exception as e:
        logger.error(f"Mission execution failed: {e}")
        return False

def analyze_mission_success(results, config):
    """Analyze mission results against success criteria"""
    success_metrics = {}
    
    for agent_id, agent_results in results.spacecraft_states.items():
        final_state = agent_results.states[-1]
        target_state = agent_results.target_state
        
        # Calculate final errors
        position_error = np.linalg.norm(final_state[:3] - target_state[:3])
        attitude_error = calculate_attitude_error(
            final_state[6:10], target_state[6:10]
        )  # quaternion difference in degrees
        
        # Determine success status
        pos_tolerance = config.success_criteria.get('position_tolerance', 0.1)
        att_tolerance = config.success_criteria.get('attitude_tolerance', 1.0)
        
        if position_error <= pos_tolerance and attitude_error <= att_tolerance:
            status = 'SUCCESS'
        elif position_error <= pos_tolerance * 2 and attitude_error <= att_tolerance * 2:
            status = 'PARTIAL_SUCCESS'
        else:
            status = 'FAILURE'
        
        success_metrics[agent_id] = {
            'status': status,
            'position_error': position_error,
            'attitude_error': attitude_error,
            'fuel_used': agent_results.fuel_consumption[-1]
        }
    
    return success_metrics

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mission_executor.py <config_file> [--no-viz]")
        sys.exit(1)
    
    config_file = sys.argv[1]
    visualization = '--no-viz' not in sys.argv
    
    success = execute_mission(config_file, visualization)
    sys.exit(0 if success else 1)
```

#### Phase 3: Real-Time Monitoring

During mission execution, monitor key parameters:

```python
class RealTimeMonitor:
    def __init__(self, simulator):
        self.simulator = simulator
        self.alerts = []
        self.performance_history = []
    
    def monitor_mission_progress(self):
        """Monitor mission in real-time"""
        while self.simulator.is_running():
            current_status = self.simulator.get_current_status()
            
            # Check for alerts
            self._check_safety_violations(current_status)
            self._check_performance_degradation(current_status)
            self._check_communication_status(current_status)
            
            # Update displays
            self._update_telemetry_display(current_status)
            self._update_trajectory_plot(current_status)
            
            time.sleep(0.1)  # 10 Hz monitoring
    
    def _check_safety_violations(self, status):
        """Check for safety constraint violations"""
        for agent_id, agent_status in status.agents.items():
            # Inter-agent separation
            for other_id, other_status in status.agents.items():
                if agent_id != other_id:
                    distance = np.linalg.norm(
                        agent_status.position - other_status.position
                    )
                    if distance < 10.0:  # 10m safety margin
                        self._raise_alert(
                            'PROXIMITY_WARNING',
                            f"{agent_id} and {other_id} separation: {distance:.2f}m"
                        )
            
            # Velocity limits
            velocity_magnitude = np.linalg.norm(agent_status.velocity)
            if velocity_magnitude > 2.0:  # 2 m/s maximum
                self._raise_alert(
                    'VELOCITY_VIOLATION',
                    f"{agent_id} velocity: {velocity_magnitude:.2f} m/s"
                )
    
    def _raise_alert(self, alert_type, message):
        """Raise system alert"""
        alert = {
            'timestamp': time.time(),
            'type': alert_type,
            'message': message
        }
        self.alerts.append(alert)
        print(f"ALERT [{alert_type}]: {message}")
```

### Post-Mission Analysis

After mission completion, perform comprehensive analysis:

```python
class PostMissionAnalyzer:
    def __init__(self, results_file):
        self.results = self.load_results(results_file)
    
    def generate_mission_report(self):
        """Generate comprehensive mission analysis report"""
        
        report = {
            'executive_summary': self._generate_executive_summary(),
            'performance_metrics': self._analyze_performance(),
            'trajectory_analysis': self._analyze_trajectories(),
            'fuel_consumption': self._analyze_fuel_usage(),
            'control_performance': self._analyze_control_systems(),
            'safety_analysis': self._analyze_safety_margins(),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _analyze_performance(self):
        """Analyze overall mission performance"""
        metrics = {}
        
        for agent_id, agent_data in self.results.agents.items():
            # Position accuracy over time
            position_errors = [
                np.linalg.norm(state[:3] - target[:3])
                for state, target in zip(agent_data.states, agent_data.targets)
            ]
            
            # Control effort
            control_effort = [
                np.linalg.norm(control)
                for control in agent_data.controls
            ]
            
            metrics[agent_id] = {
                'final_position_error': position_errors[-1],
                'mean_position_error': np.mean(position_errors),
                'max_position_error': np.max(position_errors),
                'total_control_effort': np.sum(control_effort),
                'control_efficiency': self._calculate_control_efficiency(agent_data)
            }
        
        return metrics
    
    def generate_plots(self, output_dir):
        """Generate analysis plots"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot 1: Trajectories
        for agent_id, agent_data in self.results.agents.items():
            positions = np.array([state[:3] for state in agent_data.states])
            axes[0,0].plot(positions[:,0], positions[:,1], label=agent_id)
        axes[0,0].set_title('Spacecraft Trajectories')
        axes[0,0].set_xlabel('X Position (m)')
        axes[0,0].set_ylabel('Y Position (m)')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Plot 2: Position errors
        for agent_id, agent_data in self.results.agents.items():
            errors = [
                np.linalg.norm(state[:3] - target[:3])
                for state, target in zip(agent_data.states, agent_data.targets)
            ]
            axes[0,1].semilogy(self.results.time, errors, label=agent_id)
        axes[0,1].set_title('Position Errors')
        axes[0,1].set_xlabel('Time (s)')
        axes[0,1].set_ylabel('Position Error (m)')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # Plot 3: Control inputs
        for agent_id, agent_data in self.results.agents.items():
            thrust_magnitudes = [
                np.linalg.norm(control[:3])
                for control in agent_data.controls
            ]
            axes[0,2].plot(self.results.time[:-1], thrust_magnitudes, label=agent_id)
        axes[0,2].set_title('Thrust Magnitudes')
        axes[0,2].set_xlabel('Time (s)')
        axes[0,2].set_ylabel('Thrust (N)')
        axes[0,2].legend()
        axes[0,2].grid(True)
        
        # Plot 4: Fuel consumption
        for agent_id, agent_data in self.results.agents.items():
            axes[1,0].plot(self.results.time, agent_data.fuel_consumption, label=agent_id)
        axes[1,0].set_title('Fuel Consumption')
        axes[1,0].set_xlabel('Time (s)')
        axes[1,0].set_ylabel('Fuel Used (kg)')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Plot 5: Inter-agent distances
        agent_ids = list(self.results.agents.keys())
        if len(agent_ids) >= 2:
            for i, agent1 in enumerate(agent_ids[:-1]):
                for agent2 in agent_ids[i+1:]:
                    distances = [
                        np.linalg.norm(
                            state1[:3] - state2[:3]
                        )
                        for state1, state2 in zip(
                            self.results.agents[agent1].states,
                            self.results.agents[agent2].states
                        )
                    ]
                    axes[1,1].plot(self.results.time, distances, label=f"{agent1}-{agent2}")
        axes[1,1].set_title('Inter-Agent Distances')
        axes[1,1].set_xlabel('Time (s)')
        axes[1,1].set_ylabel('Distance (m)')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        # Plot 6: Computational performance
        solve_times = self.results.performance_data.get('solve_times', [])
        if solve_times:
            axes[1,2].plot(solve_times)
            axes[1,2].set_title('MPC Solve Times')
            axes[1,2].set_xlabel('Iteration')
            axes[1,2].set_ylabel('Solve Time (s)')
            axes[1,2].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/mission_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
```

---

## Visualization and Monitoring

### Real-Time Visualization Setup

The system provides multiple visualization options for monitoring mission progress:

#### Basic 2D Trajectory Visualization

```python
from src.visualization.simple_viewer import LiveViewer

# Initialize viewer
viewer = LiveViewer({
    'window_size': (1200, 800),
    'update_frequency': 10,  # Hz
    'plot_history_length': 500  # data points
})

# Setup plots
viewer.add_trajectory_plot('main', {
    'xlim': [-1000, 200],
    'ylim': [-200, 200],
    'title': 'Spacecraft Trajectories (LVLH Frame)'
})

viewer.add_telemetry_plot('telemetry', {
    'parameters': ['position_error', 'velocity', 'fuel_remaining'],
    'time_window': 300  # 5 minutes
})

# Start real-time monitoring
viewer.start_monitoring(simulator)
```

#### Advanced 3D Visualization

```python
from src.visualization.advanced_viewer import Advanced3DViewer

# Configure 3D viewer
viewer_config = {
    'renderer': 'OpenGL',
    'lighting': 'orbital',
    'camera_mode': 'free',
    'background': 'space',
    'spacecraft_models': {
        'chaser': 'models/cygnus.obj',
        'target': 'models/iss.obj'
    }
}

viewer = Advanced3DViewer(viewer_config)

# Add orbital mechanics visualization
viewer.add_orbit_visualization({
    'reference_orbit': True,
    'coordinate_frames': ['LVLH', 'ECI'],
    'ground_track': True
})

# Add formation flying visualization
viewer.add_formation_display({
    'formation_constraints': True,
    'communication_links': True,
    'safety_spheres': True
})
```

### Mission Monitoring Dashboard

Create comprehensive monitoring dashboard:

```python
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class MissionControlDashboard:
    def __init__(self, simulator):
        self.simulator = simulator
        self.root = tk.Tk()
        self.root.title("Mission Control Dashboard")
        self.root.geometry("1600x1000")
        
        self.setup_dashboard()
        self.start_monitoring()
    
    def setup_dashboard(self):
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Mission status panel
        status_frame = ttk.LabelFrame(main_frame, text="Mission Status")
        status_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        self.mission_status_label = ttk.Label(status_frame, text="Initializing...")
        self.mission_status_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.mission_time_label = ttk.Label(status_frame, text="T+ 00:00:00")
        self.mission_time_label.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # Agent status panel
        agents_frame = ttk.LabelFrame(main_frame, text="Spacecraft Status")
        agents_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        self.agent_status_tree = ttk.Treeview(agents_frame, columns=('Position', 'Velocity', 'Fuel', 'Status'))
        self.agent_status_tree.heading('#0', text='Agent')
        self.agent_status_tree.heading('Position', text='Position (m)')
        self.agent_status_tree.heading('Velocity', text='Velocity (m/s)')
        self.agent_status_tree.heading('Fuel', text='Fuel (%)')
        self.agent_status_tree.heading('Status', text='Status')
        self.agent_status_tree.pack(fill=tk.X, padx=5, pady=5)
        
        # Plots panel
        plots_frame = ttk.LabelFrame(main_frame, text="Real-Time Plots")
        plots_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5)
        
        # Create matplotlib figure
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, plots_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize plot data
        self.time_data = []
        self.position_data = {}
        self.velocity_data = {}
        self.fuel_data = {}
        
        # Alerts panel
        alerts_frame = ttk.LabelFrame(main_frame, text="System Alerts")
        alerts_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        self.alerts_text = tk.Text(alerts_frame, height=6, width=80)
        alerts_scrollbar = ttk.Scrollbar(alerts_frame, orient=tk.VERTICAL, command=self.alerts_text.yview)
        self.alerts_text.config(yscrollcommand=alerts_scrollbar.set)
        self.alerts_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        alerts_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def start_monitoring(self):
        """Start real-time monitoring loop"""
        self.update_dashboard()
        self.root.after(100, self.start_monitoring)  # Update every 100ms
    
    def update_dashboard(self):
        """Update dashboard with current data"""
        if not self.simulator.is_running():
            return
        
        current_status = self.simulator.get_current_status()
        
        # Update mission status
        self.mission_status_label.config(text=f"Status: {current_status.mission_phase}")
        mission_time = current_status.elapsed_time
        hours, remainder = divmod(mission_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.mission_time_label.config(text=f"T+ {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        
        # Update agent status tree
        for item in self.agent_status_tree.get_children():
            self.agent_status_tree.delete(item)
        
        for agent_id, agent_status in current_status.agents.items():
            pos_str = f"[{agent_status.position[0]:.1f}, {agent_status.position[1]:.1f}, {agent_status.position[2]:.1f}]"
            vel_str = f"{np.linalg.norm(agent_status.velocity):.3f}"
            fuel_str = f"{agent_status.fuel_remaining:.1f}"
            
            self.agent_status_tree.insert('', 'end', text=agent_id,
                values=(pos_str, vel_str, fuel_str, agent_status.status))
        
        # Update plots
        self.time_data.append(mission_time)
        
        for agent_id, agent_status in current_status.agents.items():
            if agent_id not in self.position_data:
                self.position_data[agent_id] = []
                self.velocity_data[agent_id] = []
                self.fuel_data[agent_id] = []
            
            self.position_data[agent_id].append(np.linalg.norm(agent_status.position))
            self.velocity_data[agent_id].append(np.linalg.norm(agent_status.velocity))
            self.fuel_data[agent_id].append(agent_status.fuel_remaining)
        
        # Keep only last 500 data points
        if len(self.time_data) > 500:
            self.time_data = self.time_data[-500:]
            for agent_id in self.position_data:
                self.position_data[agent_id] = self.position_data[agent_id][-500:]
                self.velocity_data[agent_id] = self.velocity_data[agent_id][-500:]
                self.fuel_data[agent_id] = self.fuel_data[agent_id][-500:]
        
        # Update matplotlib plots
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot 1: Positions
        for agent_id in self.position_data:
            self.axes[0,0].plot(self.time_data, self.position_data[agent_id], label=agent_id)
        self.axes[0,0].set_title('Distance from Origin')
        self.axes[0,0].set_ylabel('Distance (m)')
        self.axes[0,0].legend()
        self.axes[0,0].grid(True)
        
        # Plot 2: Velocities
        for agent_id in self.velocity_data:
            self.axes[0,1].plot(self.time_data, self.velocity_data[agent_id], label=agent_id)
        self.axes[0,1].set_title('Velocity Magnitude')
        self.axes[0,1].set_ylabel('Velocity (m/s)')
        self.axes[0,1].legend()
        self.axes[0,1].grid(True)
        
        # Plot 3: Fuel remaining
        for agent_id in self.fuel_data:
            self.axes[1,0].plot(self.time_data, self.fuel_data[agent_id], label=agent_id)
        self.axes[1,0].set_title('Fuel Remaining')
        self.axes[1,0].set_ylabel('Fuel (%)')
        self.axes[1,0].legend()
        self.axes[1,0].grid(True)
        
        # Plot 4: Control performance (if available)
        if hasattr(current_status, 'control_performance'):
            solve_times = current_status.control_performance.get('solve_times', [])
            if solve_times:
                self.axes[1,1].plot(solve_times[-100:])  # Last 100 solve times
                self.axes[1,1].set_title('MPC Solve Times')
                self.axes[1,1].set_ylabel('Solve Time (s)')
                self.axes[1,1].grid(True)
        
        self.canvas.draw()
        
        # Check for new alerts
        self.check_alerts(current_status)
    
    def check_alerts(self, status):
        """Check for system alerts and update display"""
        # Check inter-agent distances
        agent_ids = list(status.agents.keys())
        for i, agent1 in enumerate(agent_ids[:-1]):
            for agent2 in agent_ids[i+1:]:
                distance = np.linalg.norm(
                    status.agents[agent1].position - status.agents[agent2].position
                )
                if distance < 15.0:  # 15m warning threshold
                    self.add_alert(f"PROXIMITY: {agent1}-{agent2} separation {distance:.1f}m")
        
        # Check velocity limits
        for agent_id, agent_status in status.agents.items():
            vel_mag = np.linalg.norm(agent_status.velocity)
            if vel_mag > 1.5:  # 1.5 m/s warning threshold
                self.add_alert(f"VELOCITY: {agent_id} speed {vel_mag:.2f} m/s")
    
    def add_alert(self, message):
        """Add alert to alerts display"""
        timestamp = time.strftime('%H:%M:%S')
        alert_message = f"[{timestamp}] {message}\n"
        self.alerts_text.insert(tk.END, alert_message)
        self.alerts_text.see(tk.END)

# Usage example
if __name__ == "__main__":
    # Initialize simulator (example)
    simulator = DockingSimulator(mission_config)
    
    # Start dashboard
    dashboard = MissionControlDashboard(simulator)
    
    # Start simulation in separate thread
    import threading
    sim_thread = threading.Thread(target=simulator.run)
    sim_thread.start()
    
    # Run dashboard
    dashboard.root.mainloop()
```

---

## Performance Tuning

### Computational Performance Optimization

#### Real-Time Performance Configuration

For real-time operations requiring 10-100 Hz control rates:

```python
# config/performance/realtime_config.py
REALTIME_CONFIG = {
    'controller': {
        'prediction_horizon': 10,        # Reduce horizon for speed
        'time_step': 0.1,               # 10 Hz control rate
        'solver': 'OSQP',               # Fast QP solver
        'solver_settings': {
            'max_iter': 1000,
            'eps_abs': 1e-4,
            'eps_rel': 1e-4,
            'warm_start': True,
            'adaptive_rho': True
        },
        'warm_start': True,             # Essential for real-time
        'emergency_fallback': True      # Simple backup control
    },
    
    'optimization': {
        'approximation_level': 'medium',  # Balance accuracy vs speed
        'constraint_relaxation': 1e-6,   # Slight constraint relaxation
        'early_termination': True,       # Stop when "good enough"
        'parallel_scenarios': 4          # Parallel uncertainty scenarios
    },
    
    'system': {
        'parallel_processing': True,      # Multi-threading
        'num_threads': 4,                # CPU cores to use
        'memory_pool': True,             # Pre-allocated memory
        'sparse_matrices': True,         # Sparse linear algebra
        'vectorized_operations': True    # NumPy vectorization
    }
}
```

#### High-Accuracy Configuration

For offline analysis or critical missions requiring maximum precision:

```python
# config/performance/high_accuracy_config.py
HIGH_ACCURACY_CONFIG = {
    'controller': {
        'prediction_horizon': 50,        # Long horizon for accuracy
        'time_step': 0.02,              # 50 Hz for fine control
        'solver': 'MOSEK',              # Commercial high-accuracy solver
        'solver_settings': {
            'MSK_DPAR_INTPNT_TOL_PFEAS': 1e-10,
            'MSK_DPAR_INTPNT_TOL_DFEAS': 1e-10,
            'MSK_DPAR_INTPNT_TOL_INFEAS': 1e-12,
            'MSK_IPAR_INTPNT_MAX_ITERATIONS': 10000
        },
        'terminal_constraints': True,    # Enforce terminal constraints
        'robust_invariant_set': True    # Use robust terminal set
    },
    
    'uncertainty': {
        'wasserstein_radius': 0.2,      # Larger uncertainty set
        'scenario_sampling': 100,       # Many uncertainty scenarios
        'monte_carlo_samples': 10000,   # Detailed uncertainty propagation
        'adaptive_sampling': True       # Importance sampling
    },
    
    'integration': {
        'method': 'Radau',              # High-order implicit method
        'rtol': 1e-10,                  # Tight relative tolerance
        'atol': 1e-12,                  # Tight absolute tolerance
        'max_step': 0.01               # Small maximum step size
    }
}
```

### Memory Optimization Strategies

#### Large-Scale Multi-Agent Simulations

For simulations with 20+ spacecraft:

```python
class MemoryOptimizedSimulator:
    def __init__(self, config):
        self.config = config
        self.setup_memory_pools()
        self.setup_hierarchical_control()
    
    def setup_memory_pools(self):
        """Pre-allocate memory pools to avoid runtime allocation"""
        max_agents = self.config['max_agents']
        horizon = self.config['prediction_horizon']
        
        # Pre-allocate trajectory storage
        self.trajectory_pool = np.zeros((max_agents, horizon, 13))
        self.control_pool = np.zeros((max_agents, horizon-1, 6))
        
        # Pre-allocate optimization matrices
        self.optimization_matrices = {
            'A': np.zeros((max_agents, horizon*13, horizon*6)),
            'b': np.zeros((max_agents, horizon*13)),
            'Q': scipy.sparse.block_diag([np.eye(13)] * horizon),
            'R': scipy.sparse.block_diag([np.eye(6)] * (horizon-1))
        }
    
    def setup_hierarchical_control(self):
        """Setup hierarchical control to reduce coupling"""
        # Group agents into clusters to reduce communication overhead
        self.agent_clusters = self.create_agent_clusters()
        
        # Setup cluster leaders for coordination
        self.cluster_coordinators = {}
        for cluster_id, agents in self.agent_clusters.items():
            leader = agents[0]  # First agent is leader
            self.cluster_coordinators[cluster_id] = leader
    
    def create_agent_clusters(self):
        """Create agent clusters based on proximity"""
        # Use k-means clustering based on initial positions
        from sklearn.cluster import KMeans
        
        positions = np.array([
            agent.initial_position 
            for agent in self.config['agents']
        ])
        
        num_clusters = min(4, len(positions) // 5)  # ~5 agents per cluster
        kmeans = KMeans(n_clusters=num_clusters)
        cluster_labels = kmeans.fit_predict(positions)
        
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(self.config['agents'][i])
        
        return clusters
```

#### Efficient Data Structures

```python
class EfficientTrajectoryStorage:
    """Memory-efficient trajectory storage using circular buffers"""
    
    def __init__(self, max_length=1000):
        self.max_length = max_length
        self.data = np.zeros((max_length, 13))  # 13-DOF state
        self.controls = np.zeros((max_length, 6))  # 6-DOF control
        self.timestamps = np.zeros(max_length)
        self.current_index = 0
        self.is_full = False
    
    def append(self, state, control, timestamp):
        """Add new data point"""
        self.data[self.current_index] = state
        self.controls[self.current_index] = control
        self.timestamps[self.current_index] = timestamp
        
        self.current_index = (self.current_index + 1) % self.max_length
        if self.current_index == 0:
            self.is_full = True
    
    def get_recent_data(self, num_points=None):
        """Get most recent data points"""
        if num_points is None:
            num_points = self.get_length()
        
        if not self.is_full:
            # Buffer not yet full
            end_idx = self.current_index
            start_idx = max(0, end_idx - num_points)
            return {
                'states': self.data[start_idx:end_idx],
                'controls': self.controls[start_idx:end_idx],
                'timestamps': self.timestamps[start_idx:end_idx]
            }
        else:
            # Buffer is full, need to handle wraparound
            indices = [(self.current_index - i - 1) % self.max_length 
                      for i in range(min(num_points, self.max_length))]
            indices.reverse()
            
            return {
                'states': self.data[indices],
                'controls': self.controls[indices], 
                'timestamps': self.timestamps[indices]
            }
    
    def get_length(self):
        """Get current number of stored points"""
        return self.max_length if self.is_full else self.current_index
```

### Solver Performance Tuning

#### Solver Selection Guide

Choose the optimal solver based on problem characteristics:

```python
def select_optimal_solver(problem_characteristics):
    """Select best solver based on problem characteristics"""
    
    num_agents = problem_characteristics['num_agents']
    horizon = problem_characteristics['prediction_horizon']
    has_nonlinear_constraints = problem_characteristics['nonlinear_constraints']
    real_time_required = problem_characteristics['real_time']
    
    if real_time_required and num_agents <= 5:
        # Real-time with few agents
        return {
            'solver': 'OSQP',
            'settings': {
                'max_iter': 1000,
                'eps_abs': 1e-4,
                'eps_rel': 1e-4,
                'warm_start': True,
                'adaptive_rho': True,
                'polish': False  # Skip polishing for speed
            }
        }
    
    elif real_time_required and num_agents > 5:
        # Real-time with many agents - use distributed approach
        return {
            'solver': 'ADMM',  # Alternating Direction Method of Multipliers
            'settings': {
                'max_iter': 500,
                'tolerance': 1e-3,
                'rho': 1.0,
                'distributed': True,
                'num_parallel_solvers': min(4, num_agents)
            }
        }
    
    elif has_nonlinear_constraints:
        # Nonlinear optimization required
        return {
            'solver': 'IPOPT',
            'settings': {
                'max_iter': 3000,
                'tol': 1e-8,
                'acceptable_tol': 1e-6,
                'mu_strategy': 'adaptive'
            }
        }
    
    else:
        # High accuracy offline computation
        return {
            'solver': 'MOSEK',
            'settings': {
                'MSK_DPAR_INTPNT_TOL_PFEAS': 1e-9,
                'MSK_DPAR_INTPNT_TOL_DFEAS': 1e-9,
                'MSK_IPAR_INTPNT_MAX_ITERATIONS': 5000,
                'MSK_IPAR_PRESOLVE_USE': 1
            }
        }
```

#### Warm Starting Strategy

Implement efficient warm starting for real-time performance:

```python
class WarmStartManager:
    def __init__(self, prediction_horizon):
        self.horizon = prediction_horizon
        self.previous_solution = None
        self.solution_history = []
        self.max_history = 10
    
    def get_warm_start(self, current_state, target_state):
        """Generate warm start solution"""
        
        if self.previous_solution is None:
            # No previous solution - use simple initial guess
            return self._generate_initial_guess(current_state, target_state)
        
        # Shift previous solution and add new terminal point
        warm_start = self._shift_and_extend(
            self.previous_solution, current_state, target_state
        )
        
        return warm_start
    
    def _shift_and_extend(self, prev_solution, current_state, target_state):
        """Shift previous solution forward in time"""
        
        # Extract previous trajectory
        prev_states = prev_solution['states']
        prev_controls = prev_solution['controls']
        
        # Shift trajectory (remove first state, shift others)
        new_states = np.zeros((self.horizon + 1, 13))
        new_controls = np.zeros((self.horizon, 6))
        
        # Current state is known
        new_states[0] = current_state
        
        if len(prev_states) > 2:
            # Shift previous trajectory
            new_states[1:-1] = prev_states[2:]
            new_controls[:-1] = prev_controls[1:]
            
            # Extrapolate terminal state and control
            # Simple linear extrapolation
            if len(prev_states) >= 3:
                state_trend = prev_states[-1] - prev_states[-2]
                new_states[-1] = prev_states[-1] + state_trend
            else:
                new_states[-1] = target_state
            
            # Terminal control (simple proportional)
            position_error = target_state[:3] - new_states[-1][:3]
            new_controls[-1][:3] = 0.5 * position_error  # Proportional control
            new_controls[-1][3:] = np.zeros(3)  # No torque
        
        else:
            # Not enough history - generate fresh initial guess
            return self._generate_initial_guess(current_state, target_state)
        
        return {'states': new_states, 'controls': new_controls}
    
    def _generate_initial_guess(self, current_state, target_state):
        """Generate initial guess for optimization"""
        
        states = np.zeros((self.horizon + 1, 13))
        controls = np.zeros((self.horizon, 6))
        
        # Linear interpolation between current and target states
        for k in range(self.horizon + 1):
            alpha = k / self.horizon
            states[k] = (1 - alpha) * current_state + alpha * target_state
        
        # Simple bang-bang control
        for k in range(self.horizon):
            position_error = states[k+1][:3] - states[k][:3]
            controls[k][:3] = 2.0 * position_error / 0.1  # Assuming dt=0.1
            controls[k][3:] = np.zeros(3)  # No torque initially
        
        return {'states': states, 'controls': controls}
    
    def update_solution(self, new_solution):
        """Update with new optimal solution"""
        self.previous_solution = new_solution.copy()
        
        # Store in history for learning
        self.solution_history.append(new_solution)
        if len(self.solution_history) > self.max_history:
            self.solution_history.pop(0)
```

---

## Common Use Cases

### Use Case 1: ISS Cargo Resupply Mission

**Scenario:** Automated cargo vehicle approaches and dock with the International Space Station

**Mission Parameters:**
- **Vehicle:** Cygnus cargo spacecraft (7500 kg)
- **Target:** ISS Nadir port
- **Approach:** V-bar approach corridor
- **Duration:** 30 minutes
- **Key Requirements:** ±10 cm position accuracy, ±0.5° attitude accuracy

**Configuration Example:**
```python
# missions/iss_resupply/mission_config.py
class ISSResupplyMission(MissionConfig):
    def __init__(self):
        super().__init__('cargo_resupply')
        
        # Cygnus spacecraft configuration
        self.add_spacecraft('cygnus-ng18', {
            'type': 'cygnus_enhanced',
            'mass': 7500.0,
            'initial_position': [-1000.0, 0.0, -200.0],  # 1km behind, 200m below
            'initial_velocity': [0.3, 0.0, 0.05],        # Slow approach
            'fuel_capacity': 300.0,
            'docking_mechanism': 'cbers',
            'target_port': 'unity_nadir'
        })
        
        # ISS configuration  
        self.add_spacecraft('ISS', {
            'type': 'space_station',
            'mass': 420000.0,
            'initial_position': [0.0, 0.0, 0.0],
            'attitude_control': 'station_keeping',
            'docking_ports': {
                'unity_nadir': {
                    'position': [0.0, 0.0, -4.57],
                    'orientation': [0.0, 0.0, -1.0],
                    'type': 'cbers'
                }
            }
        })
        
        # Mission phases
        self.mission_phases = [
            {
                'name': 'approach_initiation',
                'duration': 600,
                'target_distance': 300.0,
                'approach_velocity': -0.2,
                'safety_ellipse': [50.0, 50.0, 25.0]
            },
            {
                'name': 'close_approach',
                'duration': 900,
                'target_distance': 30.0,
                'approach_velocity': -0.1,
                'keep_out_sphere': 10.0
            },
            {
                'name': 'final_approach',
                'duration': 300,
                'target_distance': 0.0,
                'approach_velocity': -0.03,
                'precision_mode': True
            }
        ]
        
        # Success criteria
        self.success_criteria = {
            'position_tolerance': 0.10,  # 10 cm
            'attitude_tolerance': 0.5,   # 0.5 degrees
            'velocity_tolerance': 0.02,  # 2 cm/s
            'fuel_efficiency': 0.85     # 85% fuel efficiency target
        }
```

### Use Case 2: Multi-Satellite Servicing Mission

**Scenario:** Three servicing satellites coordinate to simultaneously service a target satellite

**Mission Parameters:**
- **Servicers:** Three identical robotic satellites (500 kg each)
- **Target:** Communication satellite (2000 kg)
- **Operations:** Simultaneous fuel transfer, component replacement, and inspection
- **Duration:** 2 hours
- **Coordination:** Distributed consensus control

**Configuration Example:**
```python
# missions/satellite_servicing/mission_config.py
class SatelliteServicingMission(MissionConfig):
    def __init__(self):
        super().__init__('multi_satellite_servicing')
        
        # Target satellite
        self.add_spacecraft('geosat-target', {
            'type': 'communication_satellite',
            'mass': 2000.0,
            'initial_position': [0.0, 0.0, 0.0],
            'orbital_parameters': {
                'altitude': 35786000,  # GEO altitude
                'inclination': 0.1
            },
            'service_points': [
                {'name': 'fuel_port', 'position': [2.0, 0.0, 0.0]},
                {'name': 'antenna_panel', 'position': [0.0, 3.0, 0.0]},
                {'name': 'solar_array', 'position': [0.0, 0.0, 2.0]}
            ]
        })
        
        # Servicing satellites
        service_positions = [
            [-50.0, 0.0, 0.0],   # Fuel transfer satellite
            [0.0, -50.0, 0.0],   # Component replacement satellite
            [0.0, 0.0, -50.0]    # Inspection satellite
        ]
        
        service_roles = ['fuel_transfer', 'component_replacement', 'inspection']
        
        for i, (position, role) in enumerate(zip(service_positions, service_roles)):
            self.add_spacecraft(f'servicer-{i+1:02d}', {
                'type': 'robotic_servicer',
                'mass': 500.0,
                'initial_position': position,
                'service_role': role,
                'manipulator_config': 'dual_arm' if role != 'inspection' else 'sensor_suite'
            })
        
        # Coordination parameters
        self.coordination_config = {
            'formation_type': 'tetrahedral',
            'formation_spacing': 25.0,
            'consensus_algorithm': 'distributed_mpc',
            'communication_topology': 'all_to_all',
            'conflict_resolution': 'auction_based'
        }
        
        # Service operations timeline
        self.service_operations = [
            {
                'time': 1800,  # 30 minutes
                'operation': 'simultaneous_approach',
                'formation': 'approach_triangle',
                'safety_distance': 15.0
            },
            {
                'time': 3600,  # 60 minutes
                'operation': 'coordinated_servicing',
                'formations': {
                    'fuel_transfer': 'contact_position',
                    'component_replacement': 'maintenance_position',
                    'inspection': 'observation_orbit'
                }
            },
            {
                'time': 5400,  # 90 minutes
                'operation': 'formation_departure',
                'safety_distance': 50.0
            }
        ]
```

### Use Case 3: Deep Space Formation Flying

**Scenario:** Five spacecraft maintain precise formation for interferometric observations

**Mission Parameters:**
- **Formation:** Pentagon formation (50m baseline)
- **Mission:** Radio interferometry of distant galaxies
- **Environment:** L2 Lagrange point
- **Requirements:** ±1 cm position accuracy, formation reconfiguration
- **Duration:** 24 hours

**Configuration Example:**
```python
# missions/deep_space_interferometry/mission_config.py
class DeepSpaceInterferometry(MissionConfig):
    def __init__(self):
        super().__init__('formation_flying')
        
        # Formation center position (L2 Lagrange point)
        l2_position = [1.5e9, 0.0, 0.0]  # 1.5 million km from Earth
        
        # Generate pentagon formation
        formation_positions = self._generate_pentagon_formation(
            center=l2_position,
            radius=25.0,  # 25m baseline
            num_spacecraft=5
        )
        
        # Formation master
        self.add_spacecraft('interferometer-master', {
            'type': 'formation_leader',
            'mass': 300.0,
            'initial_position': l2_position,
            'role': 'formation_control_master',
            'high_precision_navigation': True,
            'formation_control_authority': True
        })
        
        # Formation elements
        for i, position in enumerate(formation_positions):
            self.add_spacecraft(f'interferometer-{i+1:02d}', {
                'type': 'interferometer_element',
                'mass': 150.0,
                'initial_position': position,
                'role': 'formation_element',
                'formation_index': i,
                'precision_requirements': {
                    'position_accuracy': 0.01,  # 1 cm
                    'attitude_accuracy': 0.01,  # 0.01 degrees
                    'time_synchronization': 1e-12  # picosecond timing
                }
            })
        
        # Formation reconfiguration schedule
        self.formation_reconfigurations = [
            {
                'time': 21600,  # 6 hours
                'new_formation': 'linear_array',
                'baseline_length': 100.0,
                'transition_time': 1800  # 30 minutes
            },
            {
                'time': 43200,  # 12 hours
                'new_formation': 'y_array',
                'arm_length': 75.0,
                'transition_time': 1800
            },
            {
                'time': 64800,  # 18 hours
                'new_formation': 'pentagon',
                'radius': 50.0,  # Return to larger pentagon
                'transition_time': 1800
            }
        ]
        
        # Deep space environment
        self.environment_config = {
            'location': 'L2_lagrange_point',
            'gravitational_bodies': ['earth', 'moon', 'sun'],
            'radiation_environment': 'deep_space',
            'disturbances': {
                'solar_radiation_pressure': True,
                'gravitational_perturbations': True,
                'atmospheric_drag': False  # No atmosphere at L2
            }
        }
    
    def _generate_pentagon_formation(self, center, radius, num_spacecraft=5):
        """Generate pentagon formation positions"""
        positions = []
        for i in range(num_spacecraft):
            angle = 2 * np.pi * i / num_spacecraft
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = center[2]
            positions.append([x, y, z])
        return positions
```

### Use Case 4: Emergency Debris Avoidance

**Scenario:** Multiple spacecraft coordinate emergency maneuvers to avoid space debris

**Mission Parameters:**
- **Alert:** Space debris detected on collision course
- **Response Time:** 30 seconds to initiate avoidance
- **Spacecraft:** Mixed constellation (ISS, cargo vehicle, servicing satellite)
- **Coordination:** Emergency consensus protocol
- **Constraints:** Fuel-limited emergency maneuvers

**Configuration Example:**
```python
# missions/debris_avoidance/emergency_config.py
class EmergencyDebrisAvoidance(MissionConfig):
    def __init__(self, debris_trajectory):
        super().__init__('emergency_coordination')
        self.debris_trajectory = debris_trajectory
        self.emergency_timeline = self._calculate_emergency_timeline()
        
    def _calculate_emergency_timeline(self):
        """Calculate time-critical emergency response timeline"""
        debris_tle = self.debris_trajectory['tle']
        
        # Predict debris trajectory
        from sgp4.earth_gravity import wgs84
        from sgp4.io import twoline2rv
        
        satellite = twoline2rv(debris_tle[0], debris_tle[1], wgs84)
        
        # Find closest approach times for each spacecraft
        emergency_timeline = {}
        
        for spacecraft_id in self.get_spacecraft_ids():
            closest_approach_time, minimum_distance = self._find_closest_approach(
                spacecraft_id, satellite
            )
            
            # Calculate required avoidance maneuver
            if minimum_distance < 1000.0:  # 1km safety threshold
                avoidance_start_time = closest_approach_time - 300  # 5 minutes before
                emergency_timeline[spacecraft_id] = {
                    'alert_time': closest_approach_time - 600,  # 10 minutes warning
                    'maneuver_start': avoidance_start_time,
                    'closest_approach': closest_approach_time,
                    'minimum_distance': minimum_distance,
                    'threat_level': self._assess_threat_level(minimum_distance)
                }
        
        return emergency_timeline
    
    def setup_emergency_coordination(self):
        """Setup emergency coordination protocol"""
        
        # Emergency communication protocol
        self.emergency_protocol = {
            'communication_priority': 'emergency',
            'message_frequency': 1.0,  # 1 Hz during emergency
            'consensus_algorithm': 'fast_consensus',
            'decision_timeout': 10.0,  # 10 seconds to reach consensus
            'fallback_strategy': 'independent_avoidance'
        }
        
        # Emergency maneuver constraints
        self.emergency_constraints = {
            'max_acceleration': 0.5,  # m/s² for emergency maneuvers
            'fuel_reserve': 0.1,      # Keep 10% fuel in reserve
            'safety_margin': 500.0,   # 500m minimum safe distance
            'coordination_radius': 5000.0  # Coordinate within 5km
        }
        
        # Spacecraft-specific emergency procedures
        for spacecraft_id in self.emergency_timeline.keys():
            emergency_data = self.emergency_timeline[spacecraft_id]
            
            if emergency_data['threat_level'] == 'critical':
                # Critical threat - immediate large maneuver
                self.add_emergency_maneuver(spacecraft_id, {
                    'maneuver_type': 'immediate_avoidance',
                    'delta_v_budget': 10.0,  # m/s
                    'maneuver_direction': 'optimal_avoidance',
                    'coordination_required': True
                })
            
            elif emergency_data['threat_level'] == 'high':
                # High threat - coordinated avoidance
                self.add_emergency_maneuver(spacecraft_id, {
                    'maneuver_type': 'coordinated_avoidance',
                    'delta_v_budget': 5.0,   # m/s
                    'coordination_required': True
                })
            
            else:
                # Monitor only
                self.add_emergency_maneuver(spacecraft_id, {
                    'maneuver_type': 'monitor_and_prepare',
                    'delta_v_budget': 1.0,   # Small reserve
                    'coordination_required': False
                })

# Emergency execution example
def execute_emergency_response():
    # Detect debris (simulated)
    debris_data = {
        'tle': [
            '1 99999U 23001A   23100.50000000  .00000000  00000-0  00000-0 0  9990',
            '2 99999  51.6000 123.4567   0.0000000   0.0000 180.0000 15.50000000000000'
        ],
        'detection_time': time.time(),
        'confidence': 0.95
    }
    
    # Initialize emergency coordination
    emergency_mission = EmergencyDebrisAvoidance(debris_data)
    emergency_mission.setup_emergency_coordination()
    
    # Execute emergency response
    simulator = EmergencyCoordinationSimulator(emergency_mission)
    results = simulator.run_emergency_response()
    
    # Analyze results
    for spacecraft_id, result in results.items():
        print(f"{spacecraft_id}: {result.status}")
        print(f"  Final safety margin: {result.final_safety_margin:.1f} m")
        print(f"  Delta-V used: {result.delta_v_used:.2f} m/s")
        print(f"  Coordination success: {result.coordination_success}")
```

---

## Troubleshooting Guide

### Common Installation Issues

#### Python Environment Problems

**Issue:** ImportError or ModuleNotFoundError
```
ImportError: No module named 'cvxpy'
```

**Solutions:**
1. **Verify Python Version:**
   ```bash
   python3 --version  # Should be 3.9+
   ```

2. **Check Virtual Environment:**
   ```bash
   which python3
   pip list | grep cvxpy
   ```

3. **Reinstall Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt --force-reinstall
   ```

4. **Platform-Specific Issues:**
   ```bash
   # macOS with Apple Silicon
   arch -x86_64 pip install cvxpy
   
   # Linux missing system dependencies
   sudo apt-get install python3-dev build-essential
   
   # Windows missing Visual C++
   # Download and install Visual Studio Build Tools
   ```

#### MOSEK License Issues

**Issue:** MOSEK license error
```
mosek.Error: License server is not responding (1001)
```

**Solutions:**
1. **Academic License:**
   - Request academic license from MOSEK website
   - Save license file to `~/mosek/mosek.lic`
   
2. **Commercial License:**
   ```bash
   export MOSEKLM_LICENSE_FILE=/path/to/license/file
   ```

3. **Fallback to Free Solver:**
   ```python
   # In configuration file
   config['controller']['solver'] = 'OSQP'  # Free alternative
   ```

### Runtime Issues

#### Memory Problems

**Issue:** Out of memory during simulation
```
MemoryError: Unable to allocate array with shape (10000, 10000)
```

**Solutions:**
1. **Reduce Problem Size:**
   ```python
   config = {
       'prediction_horizon': 15,  # Reduced from 30
       'max_agents': 5,           # Reduced from 20
       'history_length': 100      # Reduced from 1000
   }
   ```

2. **Enable Memory Optimization:**
   ```python
   config['system'] = {
       'memory_efficient': True,
       'sparse_matrices': True,
       'circular_buffers': True
   }
   ```

3. **Monitor Memory Usage:**
   ```python
   import psutil
   
   def monitor_memory():
       process = psutil.Process()
       print(f"Memory usage: {process.memory_info().rss / 1024**2:.1f} MB")
   ```

#### Performance Issues

**Issue:** Simulation running too slowly
```
Real-time factor: 0.1x (should be 1.0x or higher)
```

**Solutions:**
1. **Profile Performance:**
   ```bash
   python -m cProfile -o profile.stats main.py --scenario single
   python -c "
   import pstats
   p = pstats.Stats('profile.stats')
   p.sort_stats('cumulative').print_stats(10)
   "
   ```

2. **Optimization Settings:**
   ```python
   performance_config = {
       'solver': 'OSQP',              # Faster solver
       'max_iterations': 500,         # Limit iterations
       'warm_start': True,           # Essential for speed
       'parallel_processing': True,   # Use multiple cores
       'approximation_level': 'medium' # Balance accuracy vs speed
   }
   ```

3. **Hardware Optimization:**
   ```python
   # Check CPU utilization
   import psutil
   print(f"CPU usage: {psutil.cpu_percent()}%")
   print(f"Available cores: {psutil.cpu_count()}")
   
   # Enable all cores
   config['system']['num_threads'] = psutil.cpu_count()
   ```

#### Convergence Problems

**Issue:** Spacecraft not reaching target
```
Agent chaser-001: FAILURE - Final position error: 5.234 m (tolerance: 0.1 m)
```

**Solutions:**
1. **Check Initial Conditions:**
   ```python
   # Verify initial conditions are reasonable
   initial_distance = np.linalg.norm(
       initial_position - target_position
   )
   print(f"Initial distance: {initial_distance} m")
   
   # Check if trajectory is feasible
   required_delta_v = estimate_delta_v_requirement(
       initial_position, target_position, mission_duration
   )
   available_delta_v = spacecraft_config['max_thrust'] * mission_duration / spacecraft_config['mass']
   
   if required_delta_v > available_delta_v:
       print("WARNING: Mission may not be feasible with available thrust")
   ```

2. **Adjust Control Parameters:**
   ```python
   # Increase prediction horizon
   config['prediction_horizon'] = 30  # Increased from 20
   
   # Tighten convergence tolerance
   config['tolerance'] = 1e-8  # Increased precision
   
   # Adjust safety margins
   config['safety_radius'] = 2.0  # Reduced from 5.0 if too conservative
   ```

3. **Debug Control System:**
   ```python
   def debug_control_system(agent, target):
       position_error = target[:3] - agent.get_position()
       velocity_error = target[3:6] - agent.get_velocity()
       
       print(f"Position error: {np.linalg.norm(position_error):.3f} m")
       print(f"Velocity error: {np.linalg.norm(velocity_error):.3f} m/s")
       
       # Check if control is saturated
       control = agent.get_last_control()
       max_thrust = agent.config['max_thrust']
       thrust_utilization = np.linalg.norm(control[:3]) / max_thrust
       
       print(f"Thrust utilization: {thrust_utilization:.1%}")
       if thrust_utilization > 0.95:
           print("WARNING: Control saturated - consider increasing max_thrust")
   ```

### Communication and Coordination Issues

**Issue:** Agents not coordinating properly
```
WARNING: Consensus not reached after 50 iterations
```

**Solutions:**
1. **Check Communication Topology:**
   ```python
   def verify_communication_topology(agents):
       # Check if network is connected
       import networkx as nx
       
       G = nx.Graph()
       for agent in agents:
           G.add_node(agent.agent_id)
       
       # Add edges for communication links
       for agent1 in agents:
           for agent2 in agents:
               if agent1 != agent2:
                   distance = np.linalg.norm(
                       agent1.get_position() - agent2.get_position()
                   )
                   if distance <= agent1.communication_range:
                       G.add_edge(agent1.agent_id, agent2.agent_id)
       
       if not nx.is_connected(G):
           print("WARNING: Communication network is not connected")
           components = list(nx.connected_components(G))
           print(f"Network has {len(components)} components: {components}")
   ```

2. **Adjust Consensus Parameters:**
   ```python
   consensus_config = {
       'max_iterations': 100,        # Increased from 50
       'tolerance': 1e-4,           # Relaxed tolerance
       'step_size': 0.1,            # Smaller step for stability
       'communication_delay': 0.05   # Account for realistic delays
   }
   ```

3. **Enable Debug Logging:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   
   # Enable coordination debug logging
   logging.getLogger('coordination').setLevel(logging.DEBUG)
   logging.getLogger('consensus').setLevel(logging.DEBUG)
   ```

### Visualization Problems

**Issue:** No visualization appearing or slow updates
```
matplotlib backend error: No display available
```

**Solutions:**
1. **Check Display Environment:**
   ```bash
   echo $DISPLAY  # Should show display (Linux/macOS)
   
   # For headless systems
   export DISPLAY=:99
   Xvfb :99 -screen 0 1024x768x24 &
   ```

2. **Alternative Backends:**
   ```python
   import matplotlib
   matplotlib.use('Agg')  # For headless systems
   # or
   matplotlib.use('TkAgg')  # For interactive display
   ```

3. **Reduce Visualization Load:**
   ```python
   visualization_config = {
       'update_frequency': 5,      # Reduced from 10 Hz
       'plot_history_length': 200, # Reduced from 500
       'enable_3d_plots': False,   # Disable expensive 3D
       'realtime_updates': False   # Update only at end
   }
   ```

### Configuration File Issues

**Issue:** Configuration validation errors
```
ConfigurationError: Invalid parameter 'prediction_horizont' in controller config
```

**Solutions:**
1. **Use Configuration Validator:**
   ```python
   from src.utils.config_validator import ConfigValidator
   
   validator = ConfigValidator()
   errors = validator.validate_config(config)
   
   if errors:
       for error in errors:
           print(f"Configuration error: {error}")
   else:
       print("Configuration is valid")
   ```

2. **Use Configuration Templates:**
   ```python
   from src.utils.config_templates import get_template
   
   # Start with validated template
   config = get_template('single_spacecraft_docking')
   
   # Modify specific parameters
   config['controller']['prediction_horizon'] = 25
   config['spacecraft']['chaser']['mass'] = 500.0
   ```

3. **Check Parameter Ranges:**
   ```python
   def validate_parameters(config):
       # Check physical constraints
       if config['spacecraft']['mass'] <= 0:
           raise ValueError("Mass must be positive")
       
       if config['controller']['prediction_horizon'] < 5:
           raise ValueError("Prediction horizon too short")
       
       if config['controller']['time_step'] <= 0:
           raise ValueError("Time step must be positive")
       
       # Check solver compatibility
       solver = config['controller']['solver']
       if solver == 'MOSEK' and not check_mosek_license():
           print("WARNING: MOSEK license not found, switching to OSQP")
           config['controller']['solver'] = 'OSQP'
   ```

### Emergency Procedures

If the simulation encounters critical errors:

1. **Safe Shutdown:**
   ```python
   # Implement emergency stop
   def emergency_stop(simulator):
       print("EMERGENCY STOP INITIATED")
       
       # Save current state
       simulator.save_emergency_state('emergency_backup.h5')
       
       # Stop all agents safely
       for agent in simulator.agents:
           agent.emergency_stop()
       
       # Close all connections
       simulator.shutdown()
   
   # Use signal handler for Ctrl+C
   import signal
   
   def signal_handler(sig, frame):
       emergency_stop(simulator)
       sys.exit(0)
   
   signal.signal(signal.SIGINT, signal_handler)
   ```

2. **Recovery from Backup:**
   ```python
   def recover_from_backup(backup_file):
       print(f"Recovering from {backup_file}")
       
       with h5py.File(backup_file, 'r') as f:
           # Restore agent states
           for agent_id in f['agents'].keys():
               agent_state = f['agents'][agent_id]['state'][:]
               agent_config = dict(f['agents'][agent_id].attrs)
               
               # Reinitialize agent
               agent = create_agent(agent_id, agent_config)
               agent.update_state(agent_state)
   ```

3. **System Health Check:**
   ```python
   def run_system_health_check():
       """Comprehensive system health check"""
       health_status = {
           'python_version': sys.version_info >= (3, 9),
           'dependencies': check_dependencies(),
           'memory_available': psutil.virtual_memory().available > 1e9,  # 1GB
           'disk_space': psutil.disk_usage('.').free > 1e9,  # 1GB
           'solver_license': check_solver_licenses()
       }
       
       all_healthy = all(health_status.values())
       
       print("System Health Check:")
       for component, status in health_status.items():
           status_str = "✓ PASS" if status else "✗ FAIL"
           print(f"  {component}: {status_str}")
       
       return all_healthy
   ```

---

**🎯 Ready to plan your spacecraft mission? Start with the [Mission Planning Guide](#mission-planning-guide) and refer to the [API Documentation](api/) for detailed technical information!**