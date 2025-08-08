# Distributionally Robust MPC Controller API Documentation

This document provides comprehensive documentation for the DR-MPC controller and related control algorithms.

## Table of Contents
- [DRMPCController](#drmpccontroller)
- [Robust Solver Chain](#robust-solver-chain)
- [Performance Tuning](#performance-tuning)
- [Code Examples](#code-examples)

---

## DRMPCController

The core distributionally robust model predictive controller that handles uncertainty in spacecraft dynamics and environmental conditions.

### Class Definition
```python
class DRMPCController:
    """
    Distributionally Robust Model Predictive Controller for spacecraft
    
    Handles parametric uncertainties, environmental disturbances, and model
    uncertainties using Wasserstein distributionally robust optimization.
    """
```

### Constructor
```python
def __init__(self, config: Dict)
```

**Configuration Parameters:**
```python
config = {
    # Core MPC Parameters
    'prediction_horizon': int,      # Prediction horizon (N) - typically 10-50
    'time_step': float,            # Control time step (dt) - typically 0.05-0.5s
    'state_dimension': int,        # State vector dimension (default: 13)
    'control_dimension': int,      # Control vector dimension (default: 6)
    
    # Distributionally Robust Parameters
    'wasserstein_radius': float,   # Wasserstein ambiguity radius (0.05-0.3)
    'confidence_level': float,     # Confidence level (0.90-0.99)
    'uncertainty_type': str,       # 'wasserstein', 'moment_based', or 'scenario'
    
    # Physical Constraints
    'max_thrust': float,           # Maximum thrust per thruster (N)
    'max_torque': float,          # Maximum torque per axis (Nm)
    'max_velocity': float,        # Maximum velocity constraint (m/s)
    'max_angular_velocity': float, # Maximum angular velocity (rad/s)
    'safety_radius': float,       # Safety radius for collision avoidance (m)
    
    # Optimization Settings
    'solver': str,                # 'MOSEK', 'OSQP', 'CLARABEL', 'SCS'
    'solver_settings': dict,      # Solver-specific parameters
    'warm_start': bool,           # Enable warm starting
    'max_iterations': int,        # Maximum solver iterations
    'tolerance': float,           # Convergence tolerance
    
    # Advanced Features
    'terminal_constraint': bool,   # Enable terminal state constraints
    'soft_constraints': bool,     # Enable soft constraint formulation
    'adaptive_horizon': bool,     # Enable adaptive horizon length
    'robust_invariant_set': bool  # Use robust invariant terminal sets
}
```

**Example Configuration:**
```python
config = {
    'prediction_horizon': 20,
    'time_step': 0.1,
    'wasserstein_radius': 0.1,
    'confidence_level': 0.95,
    'max_thrust': 10.0,
    'max_torque': 1.0,
    'safety_radius': 5.0,
    'solver': 'MOSEK',
    'warm_start': True
}
controller = DRMPCController(config)
```

### Core Methods

#### formulate_optimization
```python
def formulate_optimization(self, x0: np.ndarray, reference: np.ndarray, 
                          uncertainty_set: Dict) -> Tuple[np.ndarray, float]
```
**Description:** Formulate and solve the distributionally robust MPC optimization problem

**Parameters:**
- `x0` (np.ndarray): Current state vector (13 elements)
- `reference` (np.ndarray): Reference trajectory (13 x N+1 matrix) 
- `uncertainty_set` (Dict): Uncertainty set description

**Returns:**
- `Tuple[np.ndarray, float]`: Optimal control sequence and cost

**Uncertainty Set Format:**
```python
uncertainty_set = {
    'type': 'wasserstein',          # Uncertainty type
    'radius': 0.1,                  # Wasserstein radius
    'nominal_distribution': {       # Nominal uncertainty distribution
        'mean': np.ndarray,         # Mean vector
        'covariance': np.ndarray,   # Covariance matrix
        'samples': List[np.ndarray] # Historical samples (optional)
    },
    'support': {                    # Uncertainty support constraints
        'lower_bounds': np.ndarray, # Lower bounds on uncertainty
        'upper_bounds': np.ndarray  # Upper bounds on uncertainty
    }
}
```

#### update_uncertainty_set
```python
def update_uncertainty_set(self, new_samples: List[np.ndarray], 
                          forgetting_factor: float = 0.95) -> Dict
```
**Description:** Update uncertainty set with new observations

**Parameters:**
- `new_samples` (List[np.ndarray]): New uncertainty realizations
- `forgetting_factor` (float): Exponential forgetting factor (0 < λ < 1)

**Returns:**
- `Dict`: Updated uncertainty set

#### compute_robust_control
```python
def compute_robust_control(self, current_state: np.ndarray, 
                          target_state: np.ndarray,
                          disturbance_forecast: Optional[np.ndarray] = None) -> np.ndarray
```
**Description:** Compute robust control input for current time step

**Parameters:**
- `current_state` (np.ndarray): Current spacecraft state (13 elements)
- `target_state` (np.ndarray): Desired target state (13 elements)
- `disturbance_forecast` (Optional[np.ndarray]): Predicted disturbances

**Returns:**
- `np.ndarray`: Control input vector (6 elements) [thrust_xyz, torque_xyz]

**Example:**
```python
# Current state: [position, velocity, quaternion, angular_velocity]
current_state = np.array([10, 5, 0, -0.1, -0.05, 0, 1, 0, 0, 0, 0, 0, 0])

# Target state: docking port at origin
target_state = np.zeros(13)
target_state[6] = 1.0  # Unit quaternion

# Compute control
control = controller.compute_robust_control(current_state, target_state)
print(f"Thrust: {control[:3]} N, Torque: {control[3:]} Nm")
```

### Constraint Handling

#### add_state_constraints
```python
def add_state_constraints(self, constraint_type: str, parameters: Dict) -> None
```
**Description:** Add state constraints to the MPC formulation

**Constraint Types:**
- `'box'`: Box constraints on states
- `'polytope'`: Polytopic constraints  
- `'ellipsoid'`: Ellipsoidal constraints
- `'collision_avoidance'`: Inter-agent collision avoidance
- `'approach_corridor'`: Docking approach corridor

**Parameters Examples:**
```python
# Box constraints
controller.add_state_constraints('box', {
    'lower_bounds': np.array([-100, -100, -100, -5, -5, -5, -1, -1, -1, -1, -2, -2, -2]),
    'upper_bounds': np.array([100, 100, 100, 5, 5, 5, 1, 1, 1, 1, 2, 2, 2])
})

# Collision avoidance
controller.add_state_constraints('collision_avoidance', {
    'other_agents': ['agent_1', 'agent_2'],
    'safety_radius': 5.0,
    'prediction_method': 'constant_velocity'
})

# Approach corridor for docking
controller.add_state_constraints('approach_corridor', {
    'corridor_width': 2.0,
    'corridor_axis': np.array([1, 0, 0]),  # Along x-axis
    'distance_threshold': 20.0
})
```

#### add_control_constraints
```python
def add_control_constraints(self, constraint_type: str, parameters: Dict) -> None
```
**Description:** Add control input constraints

**Examples:**
```python
# Thrust magnitude limits
controller.add_control_constraints('thrust_limits', {
    'max_thrust_per_axis': [10, 10, 10],  # N
    'max_total_thrust': 15.0              # N
})

# Thruster configuration constraints  
controller.add_control_constraints('thruster_config', {
    'num_thrusters': 12,
    'thruster_positions': thruster_positions,  # 3 x 12 matrix
    'thruster_directions': thruster_directions, # 3 x 12 matrix
    'min_on_time': 0.01,  # Minimum thruster on time (s)
    'max_duty_cycle': 0.8 # Maximum duty cycle
})
```

### Uncertainty Modeling

#### set_parametric_uncertainty
```python
def set_parametric_uncertainty(self, parameter_name: str, 
                               uncertainty_description: Dict) -> None
```
**Description:** Define parametric uncertainties in the spacecraft model

**Supported Parameters:**
- `'mass'`: Spacecraft mass uncertainty
- `'inertia'`: Inertia tensor uncertainty  
- `'center_of_mass'`: Center of mass location uncertainty
- `'thruster_efficiency'`: Thruster performance uncertainty
- `'aerodynamic_coefficients'`: Aerodynamic parameter uncertainty

**Example:**
```python
# Mass uncertainty (±10%)
controller.set_parametric_uncertainty('mass', {
    'type': 'uniform',
    'nominal': 500.0,  # kg
    'bounds': [450.0, 550.0]
})

# Inertia uncertainty (±15% diagonal elements)
controller.set_parametric_uncertainty('inertia', {
    'type': 'ellipsoidal',
    'nominal': np.diag([100, 150, 120]),  # kg⋅m²
    'uncertainty_matrix': np.diag([15, 22.5, 18])  # ±15%
})

# Thruster efficiency uncertainty
controller.set_parametric_uncertainty('thruster_efficiency', {
    'type': 'beta',
    'alpha': 2.0, 'beta': 2.0,  # Beta distribution parameters
    'support': [0.8, 1.0]        # 80-100% efficiency
})
```

#### set_disturbance_model
```python
def set_disturbance_model(self, disturbance_type: str, model_parameters: Dict) -> None
```
**Description:** Set environmental disturbance models

**Disturbance Types:**
- `'solar_radiation_pressure'`: Solar radiation effects
- `'atmospheric_drag'`: Atmospheric drag (LEO)
- `'gravitational_perturbations'`: Higher-order gravitational effects
- `'magnetic_torques'`: Magnetic field interactions

**Example:**
```python
# Solar radiation pressure
controller.set_disturbance_model('solar_radiation_pressure', {
    'solar_constant': 1361,      # W/m²
    'spacecraft_area': 10.0,     # m²
    'reflectivity_coefficient': 1.3,
    'eclipse_model': True,       # Account for Earth shadow
    'sun_vector_uncertainty': 0.05  # ±3° sun vector uncertainty
})

# Atmospheric drag  
controller.set_disturbance_model('atmospheric_drag', {
    'drag_coefficient': 2.2,
    'reference_area': 8.0,      # m²
    'atmosphere_model': 'harris_priester',
    'density_uncertainty': 0.3  # ±30% density uncertainty
})
```

### Performance Monitoring

#### get_optimization_statistics
```python
def get_optimization_statistics() -> Dict
```
**Description:** Get detailed statistics from the last optimization solve

**Returns:**
- `Dict`: Statistics including solve time, iterations, optimality gap, etc.

```python
stats = controller.get_optimization_statistics()
print(f"Solve time: {stats['solve_time']:.3f} s")
print(f"Iterations: {stats['iterations']}")
print(f"Optimal cost: {stats['optimal_cost']:.6f}")
print(f"Constraint violations: {stats['max_constraint_violation']:.6f}")
```

#### get_robustness_metrics
```python
def get_robustness_metrics() -> Dict
```
**Description:** Get metrics quantifying the robustness of the current solution

**Returns:**
- `Dict`: Robustness metrics including worst-case cost, constraint satisfaction probability

```python
robustness = controller.get_robustness_metrics()
print(f"Worst-case cost: {robustness['worst_case_cost']:.3f}")
print(f"Constraint satisfaction probability: {robustness['constraint_satisfaction_prob']:.3f}")
print(f"Expected performance degradation: {robustness['expected_degradation']:.3f}")
```

### Advanced Features

#### enable_learning_based_prediction
```python
def enable_learning_based_prediction(self, model_config: Dict) -> None
```
**Description:** Enable ML-based uncertainty prediction and model adaptation

**Configuration:**
```python
model_config = {
    'model_type': 'gaussian_process',  # 'neural_network', 'gaussian_process'
    'training_data': training_data,    # Historical data for training
    'update_frequency': 10,            # Update every N control steps
    'confidence_threshold': 0.8,       # Minimum confidence for predictions
    'adaptation_rate': 0.1             # Learning rate for online adaptation
}
controller.enable_learning_based_prediction(model_config)
```

#### set_terminal_ingredients
```python
def set_terminal_ingredients(self, terminal_set: Dict, terminal_cost: Dict) -> None
```
**Description:** Set terminal constraint set and cost for stability guarantees

**Parameters:**
```python
# Robust invariant terminal set
terminal_set = {
    'type': 'ellipsoidal',
    'center': target_state,
    'shape_matrix': P_terminal,     # Positive definite matrix
    'robustness_level': 0.95       # Probability of constraint satisfaction
}

# Terminal cost function  
terminal_cost = {
    'type': 'quadratic',
    'weight_matrix': Q_terminal,    # Positive semi-definite matrix
    'robustness_penalty': 100.0     # Penalty for robustness violations
}

controller.set_terminal_ingredients(terminal_set, terminal_cost)
```

---

## Robust Solver Chain

Advanced solver chain for handling complex distributionally robust optimization problems.

### Class Definition
```python
class RobustSolverChain:
    """
    Multi-stage solver chain for distributionally robust optimization
    
    Implements progressive refinement and solver switching strategies
    for improved computational efficiency and solution quality.
    """
```

### Constructor
```python
def __init__(self, solver_configs: List[Dict])
```

**Solver Chain Configuration:**
```python
solver_configs = [
    # Stage 1: Fast approximate solution
    {
        'solver': 'OSQP',
        'max_time': 0.01,           # 10ms time limit
        'tolerance': 1e-3,
        'use_as_warm_start': True,
        'approximation_level': 'high'  # High approximation for speed
    },
    # Stage 2: Medium accuracy solution  
    {
        'solver': 'CLARABEL',
        'max_time': 0.05,           # 50ms time limit
        'tolerance': 1e-4,
        'warm_start_from_previous': True,
        'approximation_level': 'medium'
    },
    # Stage 3: High accuracy solution
    {
        'solver': 'MOSEK',
        'max_time': 0.1,            # 100ms time limit
        'tolerance': 1e-6,
        'warm_start_from_previous': True,
        'approximation_level': 'low',  # Low approximation for accuracy
        'use_only_if_time_permits': True
    }
]

solver_chain = RobustSolverChain(solver_configs)
```

### Methods

#### solve_progressive
```python
def solve_progressive(self, problem: ConvexProblem, 
                     time_budget: float = 0.1) -> SolutionResult
```
**Description:** Solve optimization problem using progressive refinement

**Parameters:**
- `problem` (ConvexProblem): Distributionally robust optimization problem
- `time_budget` (float): Total time budget for solving (seconds)

**Returns:**
- `SolutionResult`: Best solution found within time budget

#### adapt_solver_chain
```python
def adapt_solver_chain(self, performance_history: List[Dict]) -> None
```
**Description:** Adapt solver chain based on historical performance

**Parameters:**
- `performance_history` (List[Dict]): Historical solver performance data

---

## Performance Tuning

### Computational Performance Optimization

#### Real-Time Configuration
```python
# Configuration optimized for real-time performance (10-50 Hz)
real_time_config = {
    'prediction_horizon': 10,          # Shorter horizon
    'time_step': 0.1,                 # Larger time step
    'wasserstein_radius': 0.05,       # Smaller uncertainty set
    'solver': 'OSQP',                 # Fast solver
    'warm_start': True,               # Essential for real-time
    'max_iterations': 500,            # Limit iterations
    'tolerance': 1e-4,                # Relaxed tolerance
    'adaptive_horizon': True,         # Reduce horizon when needed
    'emergency_fallback': True        # Simple control backup
}
```

#### High-Accuracy Configuration  
```python
# Configuration optimized for accuracy (offline/slow scenarios)
high_accuracy_config = {
    'prediction_horizon': 50,         # Longer horizon
    'time_step': 0.02,               # Smaller time step  
    'wasserstein_radius': 0.2,       # Larger uncertainty set
    'solver': 'MOSEK',               # High-accuracy solver
    'max_iterations': 10000,         # More iterations
    'tolerance': 1e-8,               # Tight tolerance
    'terminal_constraint': True,     # Enforce terminal constraints
    'robust_invariant_set': True     # Use robust terminal set
}
```

### Memory Optimization

#### Efficient Data Structures
```python
class EfficientDRMPCController(DRMPCController):
    """Memory-optimized version of DR-MPC controller"""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Pre-allocate matrices to avoid repeated allocation
        self.preallocate_optimization_matrices()
        
        # Use circular buffers for trajectory storage
        self.setup_circular_buffers()
        
        # Enable sparse matrix operations
        self.enable_sparse_mode()
    
    def preallocate_optimization_matrices(self):
        """Pre-allocate optimization matrices"""
        N = self.N  # Prediction horizon
        nx, nu = self.nx, self.nu  # State and control dimensions
        
        # Decision variables  
        self.X = np.zeros((nx, N+1))  # State trajectory
        self.U = np.zeros((nu, N))    # Control trajectory
        
        # Constraint matrices
        self.A_ineq = np.zeros((self.n_constraints, nx*(N+1) + nu*N))
        self.b_ineq = np.zeros(self.n_constraints)
        
        # Cost matrices
        self.Q_block = np.kron(np.eye(N+1), self.Q)
        self.R_block = np.kron(np.eye(N), self.R)
```

### Parallel Processing

#### Multi-Threading for Large Horizons
```python
class ParallelDRMPCController(DRMPCController):
    """Multi-threaded DR-MPC controller for improved performance"""
    
    def __init__(self, config):
        super().__init__(config)
        self.num_threads = config.get('num_threads', 4)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.num_threads)
    
    def compute_robust_control_parallel(self, current_state, target_state):
        """Compute control using parallel scenario evaluation"""
        
        # Generate uncertainty scenarios
        scenarios = self.generate_uncertainty_scenarios(self.num_threads * 10)
        
        # Split scenarios across threads
        scenario_chunks = np.array_split(scenarios, self.num_threads)
        
        # Parallel evaluation
        futures = []
        for chunk in scenario_chunks:
            future = self.thread_pool.submit(
                self.evaluate_scenarios_chunk, 
                current_state, target_state, chunk
            )
            futures.append(future)
        
        # Collect results
        results = [future.result() for future in futures]
        
        # Aggregate and solve master problem
        return self.solve_master_problem(results)
```

---

## Code Examples

### Basic DR-MPC Usage
```python
#!/usr/bin/env python3
"""Basic DR-MPC controller usage"""

import numpy as np
from src.controllers.dr_mpc_controller import DRMPCController

# Configure controller
config = {
    'prediction_horizon': 20,
    'time_step': 0.1,
    'wasserstein_radius': 0.1,
    'confidence_level': 0.95,
    'max_thrust': 10.0,
    'max_torque': 1.0,
    'safety_radius': 5.0
}

# Initialize controller
controller = DRMPCController(config)

# Add constraints
controller.add_state_constraints('box', {
    'position_bounds': [-50, 50],
    'velocity_bounds': [-5, 5]
})

controller.add_control_constraints('thrust_limits', {
    'max_thrust_per_axis': [10, 10, 10]
})

# Set uncertainty models
controller.set_parametric_uncertainty('mass', {
    'type': 'uniform',
    'nominal': 500.0,
    'bounds': [450, 550]
})

# Simulation loop
current_state = np.array([10, 5, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
target_state = np.zeros(13)
target_state[6] = 1.0

for step in range(100):
    # Compute robust control
    control = controller.compute_robust_control(current_state, target_state)
    
    # Apply control (integrate dynamics)
    current_state = integrate_dynamics(current_state, control, 0.1)
    
    # Monitor performance
    stats = controller.get_optimization_statistics()
    robustness = controller.get_robustness_metrics()
    
    print(f"Step {step}: Cost={stats['optimal_cost']:.3f}, "
          f"Solve time={stats['solve_time']:.3f}s, "
          f"Robustness={robustness['constraint_satisfaction_prob']:.3f}")
    
    # Check convergence
    if np.linalg.norm(current_state[:3] - target_state[:3]) < 0.1:
        print("Target reached!")
        break
```

### Multi-Agent Coordination with DR-MPC
```python
#!/usr/bin/env python3
"""Multi-agent coordination using DR-MPC controllers"""

import numpy as np
from src.controllers.dr_mpc_controller import DRMPCController
from concurrent.futures import ThreadPoolExecutor
import threading

class MultiAgentDRMPCCoordinator:
    """Coordinates multiple DR-MPC controllers for formation flying"""
    
    def __init__(self, agent_configs):
        self.agents = {}
        self.controllers = {}
        
        # Initialize controllers for each agent
        for agent_id, config in agent_configs.items():
            self.controllers[agent_id] = DRMPCController(config)
            self.agents[agent_id] = {
                'state': np.zeros(13),
                'target': np.zeros(13),
                'control': np.zeros(6)
            }
        
        self.coordination_lock = threading.Lock()
    
    def compute_coordinated_control(self, agent_states, formation_targets):
        """Compute coordinated control for all agents"""
        
        # Update agent states
        for agent_id, state in agent_states.items():
            self.agents[agent_id]['state'] = state
            self.agents[agent_id]['target'] = formation_targets[agent_id]
        
        # Add collision avoidance constraints
        self.update_collision_avoidance_constraints()
        
        # Solve in parallel
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            futures = {}
            
            for agent_id in self.agents.keys():
                future = executor.submit(
                    self.solve_single_agent, 
                    agent_id
                )
                futures[agent_id] = future
            
            # Collect results
            controls = {}
            for agent_id, future in futures.items():
                controls[agent_id] = future.result()
        
        return controls
    
    def solve_single_agent(self, agent_id):
        """Solve optimization for single agent"""
        controller = self.controllers[agent_id]
        agent = self.agents[agent_id]
        
        return controller.compute_robust_control(
            agent['state'], 
            agent['target']
        )
    
    def update_collision_avoidance_constraints(self):
        """Update collision avoidance constraints between agents"""
        
        with self.coordination_lock:
            for agent_id, controller in self.controllers.items():
                # Get other agents' positions and predicted trajectories
                other_agents = {
                    other_id: self.agents[other_id]['state'][:3] 
                    for other_id in self.agents.keys() 
                    if other_id != agent_id
                }
                
                # Update collision avoidance constraints
                controller.add_state_constraints('collision_avoidance', {
                    'other_agents_positions': other_agents,
                    'safety_radius': 5.0,
                    'prediction_method': 'constant_velocity'
                })

# Usage example
agent_configs = {
    'leader': {
        'prediction_horizon': 20,
        'time_step': 0.1,
        'wasserstein_radius': 0.1,
        'max_thrust': 12.0,
        'role': 'leader'
    },
    'follower_1': {
        'prediction_horizon': 15,
        'time_step': 0.1,
        'wasserstein_radius': 0.08,
        'max_thrust': 10.0,
        'role': 'follower'
    },
    'follower_2': {
        'prediction_horizon': 15,
        'time_step': 0.1,
        'wasserstein_radius': 0.08,
        'max_thrust': 10.0,
        'role': 'follower'
    }
}

# Initialize coordinator
coordinator = MultiAgentDRMPCCoordinator(agent_configs)

# Formation targets (triangle formation)
formation_targets = {
    'leader': np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
    'follower_1': np.array([10, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
    'follower_2': np.array([5, 8.66, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
}

# Simulation loop
agent_states = {
    'leader': np.array([50, 20, 0, -1, -0.5, 0, 1, 0, 0, 0, 0, 0, 0]),
    'follower_1': np.array([40, 30, 5, -0.8, -0.6, -0.1, 1, 0, 0, 0, 0, 0, 0]),
    'follower_2': np.array([60, 25, -5, -1.2, -0.4, 0.1, 1, 0, 0, 0, 0, 0, 0])
}

for step in range(200):
    # Compute coordinated control
    controls = coordinator.compute_coordinated_control(agent_states, formation_targets)
    
    # Integrate dynamics for each agent
    for agent_id in agent_states.keys():
        agent_states[agent_id] = integrate_dynamics(
            agent_states[agent_id], 
            controls[agent_id], 
            0.1
        )
    
    # Check formation convergence
    formation_error = calculate_formation_error(agent_states, formation_targets)
    print(f"Step {step}: Formation error = {formation_error:.3f} m")
    
    if formation_error < 1.0:
        print("Formation achieved!")
        break
```

### Adaptive Uncertainty Estimation
```python
#!/usr/bin/env python3
"""Adaptive uncertainty estimation and model updating"""

import numpy as np
from collections import deque
from scipy.stats import wasserstein_distance

class AdaptiveDRMPCController(DRMPCController):
    """DR-MPC controller with adaptive uncertainty estimation"""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Uncertainty estimation parameters
        self.uncertainty_history = deque(maxlen=100)
        self.model_mismatch_history = deque(maxlen=50)
        self.adaptation_rate = config.get('adaptation_rate', 0.1)
        self.uncertainty_threshold = config.get('uncertainty_threshold', 0.05)
        
        # Online learning components
        self.enable_online_learning = config.get('online_learning', True)
        self.learning_rate = config.get('learning_rate', 0.01)
    
    def update_uncertainty_estimate(self, predicted_state, actual_state, applied_control):
        """Update uncertainty estimate based on prediction error"""
        
        # Compute model mismatch
        prediction_error = actual_state - predicted_state
        self.model_mismatch_history.append(prediction_error)
        
        if len(self.model_mismatch_history) >= 10:
            # Estimate uncertainty distribution from recent errors
            recent_errors = np.array(list(self.model_mismatch_history)[-10:])
            
            # Update empirical distribution
            empirical_mean = np.mean(recent_errors, axis=0)
            empirical_cov = np.cov(recent_errors.T)
            
            # Compute Wasserstein distance to current nominal distribution
            if hasattr(self, 'nominal_uncertainty_mean'):
                distance = self.compute_wasserstein_distance(
                    empirical_mean, empirical_cov,
                    self.nominal_uncertainty_mean, self.nominal_uncertainty_cov
                )
                
                # Adapt uncertainty set if significant drift detected
                if distance > self.uncertainty_threshold:
                    self.adapt_uncertainty_set(empirical_mean, empirical_cov)
        
        # Store for history
        uncertainty_sample = {
            'prediction_error': prediction_error,
            'applied_control': applied_control.copy(),
            'timestamp': time.time()
        }
        self.uncertainty_history.append(uncertainty_sample)
    
    def adapt_uncertainty_set(self, new_mean, new_cov):
        """Adapt uncertainty set based on observed data"""
        
        # Exponential forgetting for mean and covariance
        if hasattr(self, 'nominal_uncertainty_mean'):
            self.nominal_uncertainty_mean = (
                (1 - self.adaptation_rate) * self.nominal_uncertainty_mean +
                self.adaptation_rate * new_mean
            )
            self.nominal_uncertainty_cov = (
                (1 - self.adaptation_rate) * self.nominal_uncertainty_cov +
                self.adaptation_rate * new_cov
            )
        else:
            self.nominal_uncertainty_mean = new_mean
            self.nominal_uncertainty_cov = new_cov
        
        # Adjust Wasserstein radius based on uncertainty growth
        uncertainty_growth = np.trace(new_cov) / np.trace(self.nominal_uncertainty_cov)
        if uncertainty_growth > 1.5:  # 50% increase in uncertainty
            self.wasserstein_radius = min(
                self.wasserstein_radius * 1.2, 
                0.3  # Maximum radius
            )
            print(f"Increased Wasserstein radius to {self.wasserstein_radius:.3f}")
    
    def compute_robust_control_adaptive(self, current_state, target_state):
        """Compute control with adaptive uncertainty estimation"""
        
        # Standard robust control computation
        control = self.compute_robust_control(current_state, target_state)
        
        # Predict next state for uncertainty update
        predicted_next_state = self.predict_next_state(current_state, control)
        
        # Store prediction for later uncertainty update
        self.last_prediction = {
            'predicted_state': predicted_next_state,
            'current_state': current_state.copy(),
            'control': control.copy()
        }
        
        return control
    
    def update_with_measurement(self, measured_state):
        """Update controller with new state measurement"""
        
        if hasattr(self, 'last_prediction'):
            # Update uncertainty estimate
            self.update_uncertainty_estimate(
                self.last_prediction['predicted_state'],
                measured_state,
                self.last_prediction['control']
            )
            
            # Online learning update (if enabled)
            if self.enable_online_learning:
                self.update_model_parameters(
                    self.last_prediction['current_state'],
                    self.last_prediction['control'],
                    measured_state
                )
    
    def update_model_parameters(self, prev_state, control, current_state):
        """Update model parameters using online learning"""
        
        # Simple gradient-based parameter update
        # This is a simplified example - real implementation would be more sophisticated
        
        predicted_next = self.dynamics_model.predict(prev_state, control)
        prediction_error = current_state - predicted_next
        
        # Update model parameters (simplified)
        parameter_gradient = self.compute_parameter_gradient(
            prev_state, control, prediction_error
        )
        
        self.dynamics_model.parameters += self.learning_rate * parameter_gradient

# Usage example
adaptive_config = {
    'prediction_horizon': 20,
    'time_step': 0.1,
    'wasserstein_radius': 0.1,
    'confidence_level': 0.95,
    'max_thrust': 10.0,
    'online_learning': True,
    'adaptation_rate': 0.1,
    'learning_rate': 0.01
}

controller = AdaptiveDRMPCController(adaptive_config)

# Simulation with uncertainty adaptation
current_state = np.array([10, 5, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
target_state = np.zeros(13)
target_state[6] = 1.0

for step in range(100):
    # Compute adaptive control
    control = controller.compute_robust_control_adaptive(current_state, target_state)
    
    # Simulate dynamics with unknown disturbances
    true_next_state = simulate_with_disturbances(current_state, control, 0.1)
    
    # Update controller with measurement
    controller.update_with_measurement(true_next_state)
    
    # Update current state
    current_state = true_next_state
    
    # Monitor adaptation
    if hasattr(controller, 'nominal_uncertainty_mean'):
        uncertainty_norm = np.linalg.norm(controller.nominal_uncertainty_mean)
        print(f"Step {step}: Uncertainty estimate norm = {uncertainty_norm:.4f}, "
              f"Wasserstein radius = {controller.wasserstein_radius:.3f}")
```

---

## Error Handling and Diagnostics

### Solver Failure Handling
```python
class RobustDRMPCController(DRMPCController):
    """DR-MPC controller with comprehensive error handling"""
    
    def compute_robust_control(self, current_state, target_state):
        try:
            # Attempt primary optimization
            control = super().compute_robust_control(current_state, target_state)
            
            # Validate solution
            if not self.validate_solution(control, current_state):
                raise ValueError("Invalid control solution")
                
            return control
            
        except Exception as e:
            self.logger.warning(f"Primary optimization failed: {e}")
            return self.handle_optimization_failure(current_state, target_state, e)
    
    def handle_optimization_failure(self, current_state, target_state, error):
        """Handle optimization failures with fallback strategies"""
        
        # Strategy 1: Reduce problem complexity
        try:
            simplified_config = self.config.copy()
            simplified_config['prediction_horizon'] //= 2
            simplified_config['wasserstein_radius'] /= 2
            
            temp_controller = DRMPCController(simplified_config)
            control = temp_controller.compute_robust_control(current_state, target_state)
            self.logger.info("Fallback with simplified problem succeeded")
            return control
            
        except Exception:
            pass
        
        # Strategy 2: Use deterministic MPC
        try:
            control = self.compute_deterministic_mpc(current_state, target_state)
            self.logger.info("Fallback with deterministic MPC succeeded")
            return control
            
        except Exception:
            pass
        
        # Strategy 3: Emergency control
        self.logger.error("All optimization strategies failed, using emergency control")
        return self.compute_emergency_control(current_state, target_state)
    
    def compute_emergency_control(self, current_state, target_state):
        """Compute simple emergency control (proportional control)"""
        
        position_error = target_state[:3] - current_state[:3]
        velocity_error = target_state[3:6] - current_state[3:6]
        
        # Simple PD control with safety limits
        thrust = 1.0 * position_error + 0.2 * velocity_error
        thrust = np.clip(thrust, -self.config['max_thrust']/2, self.config['max_thrust']/2)
        
        # No torque in emergency mode
        torque = np.zeros(3)
        
        return np.concatenate([thrust, torque])
```

---

*For more examples and advanced usage, see the [tutorials](../tutorials/) directory and [user manual](../user_manual.md).*