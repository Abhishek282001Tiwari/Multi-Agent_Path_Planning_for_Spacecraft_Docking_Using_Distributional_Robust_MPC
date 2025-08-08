# Spacecraft Dynamics API Documentation

This document provides comprehensive documentation for spacecraft dynamics models, state propagation, and orbital mechanics components.

## Table of Contents
- [SpacecraftDynamics](#spacecraftdynamics)
- [State Representation](#state-representation)
- [Orbital Mechanics](#orbital-mechanics)
- [Disturbance Models](#disturbance-models)
- [Code Examples](#code-examples)

---

## SpacecraftDynamics

High-fidelity 6-DOF spacecraft dynamics model with Hill-Clohessy-Wiltshire equations, environmental disturbances, and uncertainty propagation.

### Class Definition
```python
class SpacecraftDynamics:
    """
    Comprehensive spacecraft dynamics model for proximity operations
    
    Implements Hill-Clohessy-Wiltshire equations with:
    - 6-DOF translational and rotational dynamics
    - Environmental disturbances (drag, SRP, gravity gradients)
    - Parametric uncertainties (mass, inertia, CoM)
    - Actuator models (thrusters, reaction wheels)
    """
```

### Constructor
```python
def __init__(self, config: Dict)
```

**Configuration Parameters:**
```python
config = {
    # Basic Parameters
    'initial_mass': float,           # Spacecraft mass (kg)
    'inertia_matrix': np.ndarray,    # 3x3 inertia tensor (kg⋅m²)
    'thruster_config': Dict,         # Thruster configuration
    'orbital_rate': float,           # Orbital angular velocity (rad/s)
    
    # Advanced Parameters
    'center_of_mass': np.ndarray,    # Center of mass offset [x,y,z] (m)
    'aerodynamic_coefficients': Dict, # Drag coefficients
    'solar_radiation_parameters': Dict, # SRP parameters
    'gravitational_harmonics': bool, # Enable J2, J3 effects
    'fuel_consumption_model': str,   # 'ideal', 'realistic'
    
    # Uncertainty Parameters
    'mass_uncertainty': float,       # Mass uncertainty std dev (kg)
    'inertia_uncertainty': np.ndarray, # Inertia uncertainty (kg⋅m²)
    'com_uncertainty': np.ndarray,   # CoM uncertainty std dev (m)
    'disturbance_scale_factor': float, # Disturbance scaling (default: 1.0)
    
    # Integration Parameters
    'integration_method': str,       # 'runge_kutta_4', 'dormand_prince'
    'max_step_size': float,         # Maximum integration step (s)
    'relative_tolerance': float,     # Relative integration tolerance
    'absolute_tolerance': float      # Absolute integration tolerance
}
```

**Example Configuration:**
```python
config = {
    'initial_mass': 500.0,
    'inertia_matrix': np.diag([100.0, 150.0, 120.0]),
    'thruster_config': {
        'num_thrusters': 12,
        'max_thrust_per_thruster': 5.0,
        'specific_impulse': 220.0,
        'thruster_positions': thruster_positions,  # 3x12 matrix
        'thruster_directions': thruster_directions  # 3x12 matrix
    },
    'orbital_rate': 0.0011,  # ~90 min orbit
    'center_of_mass': np.array([0.0, 0.0, 0.1]),
    'mass_uncertainty': 25.0,  # ±25 kg
    'integration_method': 'runge_kutta_4'
}

dynamics = SpacecraftDynamics(config)
```

### Core Methods

#### propagate_dynamics
```python
def propagate_dynamics(self, state: np.ndarray, control: np.ndarray, 
                      dt: float, disturbances: Optional[Dict] = None) -> np.ndarray
```
**Description:** Propagate spacecraft state forward in time using high-fidelity dynamics

**Parameters:**
- `state` (np.ndarray): Current 13-element state vector
- `control` (np.ndarray): 6-element control input [thrust_xyz, torque_xyz]
- `dt` (float): Integration time step (seconds)
- `disturbances` (Optional[Dict]): External disturbances

**Returns:**
- `np.ndarray`: Propagated 13-element state vector

**State Vector Format:**
```
Elements [0:3]   - Position [x, y, z] in LVLH frame (m)
Elements [3:6]   - Velocity [vx, vy, vz] in LVLH frame (m/s)
Elements [6:10]  - Attitude quaternion [qw, qx, qy, qz] (unit quaternion)
Elements [10:13] - Angular velocity [wx, wy, wz] in body frame (rad/s)
```

**Control Vector Format:**
```
Elements [0:3] - Body-frame thrust commands [Fx, Fy, Fz] (N)
Elements [3:6] - Body-frame torque commands [Mx, My, Mz] (Nm)
```

**Example:**
```python
# Initial state: 50m behind target, approaching at 0.1 m/s
initial_state = np.array([
    -50.0, 0.0, 0.0,      # Position (LVLH)
     0.1, 0.0, 0.0,       # Velocity (LVLH)  
     1.0, 0.0, 0.0, 0.0,  # Quaternion (identity)
     0.0, 0.0, 0.0        # Angular velocity
])

# Control: 1N thrust in +x direction
control = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Propagate for 0.1 seconds
next_state = dynamics.propagate_dynamics(initial_state, control, 0.1)
print(f"New position: {next_state[:3]}")
print(f"New velocity: {next_state[3:6]}")
```

#### hill_clohessy_wiltshire
```python
def hill_clohessy_wiltshire(self, state: np.ndarray, t: float, 
                           control: np.ndarray, disturbances: Dict) -> np.ndarray
```
**Description:** Compute state derivatives using Hill-Clohessy-Wiltshire equations

**Parameters:**
- `state` (np.ndarray): Current state vector
- `t` (float): Current time (for time-varying disturbances)
- `control` (np.ndarray): Control inputs
- `disturbances` (Dict): Disturbance forces and torques

**Returns:**
- `np.ndarray`: State derivative vector

**Hill-Clohessy-Wiltshire Equations:**
```
ẍ - 2nẏ - 3n²x = Fx/m + dx
ÿ + 2nẋ = Fy/m + dy  
z̈ + n²z = Fz/m + dz

where:
- n = orbital angular velocity
- (Fx, Fy, Fz) = control forces in LVLH frame
- (dx, dy, dz) = disturbance accelerations
```

#### quaternion_multiply
```python
def quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray
```
**Description:** Multiply two quaternions (Hamilton product)

**Parameters:**
- `q1` (np.ndarray): First quaternion [qw, qx, qy, qz]
- `q2` (np.ndarray): Second quaternion [qw, qx, qy, qz]

**Returns:**
- `np.ndarray`: Product quaternion

### Physical Models

#### fuel_consumption
```python
def fuel_consumption(self, thrust: np.ndarray, dt: float, 
                    thruster_config: Optional[Dict] = None) -> float
```
**Description:** Calculate fuel mass consumed based on thrust and thruster performance

**Parameters:**
- `thrust` (np.ndarray): Thrust vector [Fx, Fy, Fz] (N)
- `dt` (float): Time duration (s)
- `thruster_config` (Optional[Dict]): Override default thruster parameters

**Returns:**
- `float`: Fuel mass consumed (kg)

**Rocket Equation:**
```
dm/dt = |F| / (Isp * g0)

where:
- |F| = thrust magnitude
- Isp = specific impulse (s)
- g0 = standard gravity (9.81 m/s²)
```

**Example:**
```python
# 5N thrust for 10 seconds
thrust = np.array([3.0, 4.0, 0.0])  # |F| = 5N
fuel_used = dynamics.fuel_consumption(thrust, 10.0)
print(f"Fuel consumed: {fuel_used:.6f} kg")
```

#### compute_inertia_tensor
```python
def compute_inertia_tensor(self, mass: float, geometry: Dict) -> np.ndarray
```
**Description:** Compute inertia tensor from spacecraft geometry

**Parameters:**
- `mass` (float): Spacecraft total mass (kg)
- `geometry` (Dict): Geometric description

**Geometry Types:**
```python
# Box geometry
geometry = {
    'type': 'box',
    'dimensions': [2.0, 1.5, 1.0],  # [length, width, height] (m)
    'mass_distribution': 'uniform'   # 'uniform', 'concentrated'
}

# Cylindrical geometry  
geometry = {
    'type': 'cylinder',
    'radius': 0.5,                  # (m)
    'height': 2.0,                  # (m)
    'axis': 'z'                     # Primary axis
}

# Custom point masses
geometry = {
    'type': 'point_masses',
    'masses': [50, 100, 75, 25],           # Mass of each component (kg)
    'positions': [[0,0,1], [1,0,0], ...], # Position of each mass (m)
}
```

**Returns:**
- `np.ndarray`: 3x3 inertia tensor (kg⋅m²)

### Uncertainty Propagation

#### propagate_with_uncertainty
```python
def propagate_with_uncertainty(self, state_distribution: Dict, control: np.ndarray,
                              dt: float, method: str = 'unscented') -> Dict
```
**Description:** Propagate state uncertainty through nonlinear dynamics

**Parameters:**
- `state_distribution` (Dict): State uncertainty description
- `control` (np.ndarray): Deterministic control input
- `dt` (float): Time step
- `method` (str): Propagation method ('unscented', 'monte_carlo', 'linearization')

**State Distribution Format:**
```python
state_distribution = {
    'mean': np.ndarray,           # Mean state (13 elements)
    'covariance': np.ndarray,     # Covariance matrix (13x13)
    'type': 'gaussian'            # Distribution type
}
```

**Returns:**
- `Dict`: Propagated state distribution

**Methods:**
- **Unscented Transform**: Sigma-point based propagation (accurate for moderate nonlinearities)
- **Monte Carlo**: Sample-based propagation (most accurate but computationally expensive)
- **Linearization**: First-order Taylor expansion (fastest but least accurate)

**Example:**
```python
# Initial state with uncertainty
state_dist = {
    'mean': np.array([10, 5, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
    'covariance': np.eye(13) * 0.1,  # 0.1 variance on each state
    'type': 'gaussian'
}

# Propagate uncertainty
next_dist = dynamics.propagate_with_uncertainty(state_dist, control, 0.1, 'unscented')

print(f"Propagated mean: {next_dist['mean'][:3]}")
print(f"Position uncertainty: {np.sqrt(np.diag(next_dist['covariance'][:3]))}")
```

### Coordinate Transformations

#### lvlh_to_inertial
```python
def lvlh_to_inertial(self, lvlh_vector: np.ndarray, orbit_state: Dict) -> np.ndarray
```
**Description:** Transform vector from LVLH (Local Vertical Local Horizontal) to inertial frame

**Parameters:**
- `lvlh_vector` (np.ndarray): Vector in LVLH coordinates
- `orbit_state` (Dict): Chief spacecraft orbital state

**Returns:**
- `np.ndarray`: Vector in inertial coordinates

#### inertial_to_lvlh
```python
def inertial_to_lvlh(self, inertial_vector: np.ndarray, orbit_state: Dict) -> np.ndarray
```
**Description:** Transform vector from inertial to LVLH frame

**Coordinate Frames:**
```
LVLH Frame:
- x-axis: Radial direction (Earth center to spacecraft)
- y-axis: Along-track direction (direction of orbital motion)  
- z-axis: Cross-track direction (orbital angular momentum)

Inertial Frame:
- Standard J2000 Earth-centered inertial coordinate system
```

#### body_to_lvlh
```python
def body_to_lvlh(self, body_vector: np.ndarray, quaternion: np.ndarray) -> np.ndarray
```
**Description:** Transform vector from spacecraft body frame to LVLH frame

**Parameters:**
- `body_vector` (np.ndarray): Vector in body-fixed coordinates
- `quaternion` (np.ndarray): Attitude quaternion [qw, qx, qy, qz]

**Returns:**
- `np.ndarray`: Vector in LVLH coordinates

---

## State Representation

### SpacecraftState Dataclass
```python
@dataclass
class SpacecraftState:
    """
    Structured representation of spacecraft state
    """
    position: np.ndarray          # [x, y, z] in LVLH frame (m)
    velocity: np.ndarray          # [vx, vy, vz] in LVLH frame (m/s)  
    attitude: np.ndarray          # Quaternion [qw, qx, qy, qz]
    angular_velocity: np.ndarray  # [wx, wy, wz] in body frame (rad/s)
    mass: float                   # Current mass (kg)
    
    def to_vector(self) -> np.ndarray:
        """Convert to 13-element state vector"""
        return np.concatenate([
            self.position, 
            self.velocity,
            self.attitude,
            self.angular_velocity
        ])
    
    @classmethod
    def from_vector(cls, state_vector: np.ndarray, mass: float) -> 'SpacecraftState':
        """Create from 13-element state vector"""
        return cls(
            position=state_vector[0:3],
            velocity=state_vector[3:6], 
            attitude=state_vector[6:10],
            angular_velocity=state_vector[10:13],
            mass=mass
        )
    
    def get_position_error(self, target: 'SpacecraftState') -> float:
        """Calculate position error from target"""
        return np.linalg.norm(self.position - target.position)
    
    def get_velocity_error(self, target: 'SpacecraftState') -> float:
        """Calculate velocity error from target"""
        return np.linalg.norm(self.velocity - target.velocity)
    
    def get_attitude_error(self, target: 'SpacecraftState') -> float:
        """Calculate attitude error (quaternion angle)"""
        q_error = self.quaternion_multiply(self.attitude, self.quaternion_conjugate(target.attitude))
        return 2 * np.arccos(np.abs(q_error[0]))  # Angle of rotation
```

### State Validation
```python
def validate_state(self, state: np.ndarray) -> bool:
    """Validate state vector for physical consistency"""
    
    # Check vector length
    if len(state) != 13:
        return False
    
    # Check quaternion normalization
    quat_norm = np.linalg.norm(state[6:10])
    if abs(quat_norm - 1.0) > 1e-6:
        return False
    
    # Check for NaN or infinite values
    if not np.all(np.isfinite(state)):
        return False
    
    # Check physical bounds
    position_magnitude = np.linalg.norm(state[0:3])
    velocity_magnitude = np.linalg.norm(state[3:6])
    angular_velocity_magnitude = np.linalg.norm(state[10:13])
    
    if position_magnitude > 10000:  # 10 km limit
        return False
    if velocity_magnitude > 100:    # 100 m/s limit  
        return False
    if angular_velocity_magnitude > 10:  # 10 rad/s limit
        return False
    
    return True

def normalize_quaternion(self, state: np.ndarray) -> np.ndarray:
    """Normalize quaternion component of state vector"""
    state_normalized = state.copy()
    quat = state_normalized[6:10]
    quat_normalized = quat / np.linalg.norm(quat)
    state_normalized[6:10] = quat_normalized
    return state_normalized
```

---

## Orbital Mechanics

### Orbital Elements Conversion
```python
class OrbitalMechanics:
    """Orbital mechanics utilities and conversions"""
    
    @staticmethod
    def cartesian_to_orbital_elements(r: np.ndarray, v: np.ndarray, 
                                    mu: float = 3.986004418e14) -> Dict:
        """Convert Cartesian state to orbital elements"""
        
        # Compute orbital elements
        h = np.cross(r, v)  # Specific angular momentum
        h_mag = np.linalg.norm(h)
        
        n = np.cross([0, 0, 1], h)  # Node vector
        n_mag = np.linalg.norm(n)
        
        e_vec = ((np.linalg.norm(v)**2 - mu/np.linalg.norm(r)) * r - 
                np.dot(r, v) * v) / mu  # Eccentricity vector
        e = np.linalg.norm(e_vec)
        
        # Semi-major axis
        energy = np.linalg.norm(v)**2/2 - mu/np.linalg.norm(r)
        a = -mu / (2 * energy)
        
        # Inclination
        i = np.arccos(h[2] / h_mag)
        
        # RAAN (Right Ascension of Ascending Node)
        if n_mag > 1e-10:
            raan = np.arccos(n[0] / n_mag)
            if n[1] < 0:
                raan = 2*np.pi - raan
        else:
            raan = 0
        
        # Argument of periapsis
        if n_mag > 1e-10 and e > 1e-10:
            arg_periapsis = np.arccos(np.dot(n, e_vec) / (n_mag * e))
            if e_vec[2] < 0:
                arg_periapsis = 2*np.pi - arg_periapsis
        else:
            arg_periapsis = 0
        
        # True anomaly
        if e > 1e-10:
            true_anomaly = np.arccos(np.dot(e_vec, r) / (e * np.linalg.norm(r)))
            if np.dot(r, v) < 0:
                true_anomaly = 2*np.pi - true_anomaly
        else:
            true_anomaly = 0
        
        return {
            'a': a,                    # Semi-major axis (m)
            'e': e,                    # Eccentricity
            'i': i,                    # Inclination (rad)
            'raan': raan,              # RAAN (rad)
            'arg_periapsis': arg_periapsis,  # Argument of periapsis (rad)
            'true_anomaly': true_anomaly     # True anomaly (rad)
        }
    
    @staticmethod
    def orbital_elements_to_cartesian(elements: Dict, mu: float = 3.986004418e14) -> Tuple[np.ndarray, np.ndarray]:
        """Convert orbital elements to Cartesian state"""
        
        a, e, i, raan, arg_periapsis, true_anomaly = (
            elements['a'], elements['e'], elements['i'],
            elements['raan'], elements['arg_periapsis'], elements['true_anomaly']
        )
        
        # Orbital radius
        r_mag = a * (1 - e**2) / (1 + e * np.cos(true_anomaly))
        
        # Position and velocity in perifocal frame
        r_pf = r_mag * np.array([np.cos(true_anomaly), np.sin(true_anomaly), 0])
        
        p = a * (1 - e**2)  # Semi-latus rectum
        v_pf = np.sqrt(mu/p) * np.array([
            -np.sin(true_anomaly),
            e + np.cos(true_anomaly),
            0
        ])
        
        # Rotation matrices
        R3_raan = np.array([
            [np.cos(raan), -np.sin(raan), 0],
            [np.sin(raan), np.cos(raan), 0], 
            [0, 0, 1]
        ])
        
        R1_i = np.array([
            [1, 0, 0],
            [0, np.cos(i), -np.sin(i)],
            [0, np.sin(i), np.cos(i)]
        ])
        
        R3_arg = np.array([
            [np.cos(arg_periapsis), -np.sin(arg_periapsis), 0],
            [np.sin(arg_periapsis), np.cos(arg_periapsis), 0],
            [0, 0, 1]
        ])
        
        # Transform to inertial frame
        R_total = R3_raan @ R1_i @ R3_arg
        r_inertial = R_total @ r_pf
        v_inertial = R_total @ v_pf
        
        return r_inertial, v_inertial
```

### Relative Motion
```python
class RelativeMotion:
    """Relative motion dynamics and transformations"""
    
    def __init__(self, chief_elements: Dict):
        self.chief_elements = chief_elements
        self.n = np.sqrt(3.986004418e14 / chief_elements['a']**3)  # Mean motion
    
    def absolute_to_relative_lvlh(self, chief_state: np.ndarray, 
                                 deputy_state: np.ndarray) -> np.ndarray:
        """Convert absolute states to relative LVLH coordinates"""
        
        chief_pos, chief_vel = chief_state[:3], chief_state[3:6]
        deputy_pos, deputy_vel = deputy_state[:3], deputy_state[3:6]
        
        # Relative position and velocity in inertial frame
        delta_r_inertial = deputy_pos - chief_pos
        delta_v_inertial = deputy_vel - chief_vel
        
        # Compute LVLH transformation matrix
        R_lvlh_to_inertial = self.compute_lvlh_matrix(chief_state)
        R_inertial_to_lvlh = R_lvlh_to_inertial.T
        
        # Transform to LVLH frame
        delta_r_lvlh = R_inertial_to_lvlh @ delta_r_inertial
        delta_v_lvlh = R_inertial_to_lvlh @ delta_v_inertial
        
        # Account for LVLH frame rotation
        omega_lvlh = np.array([0, 0, -self.n])  # LVLH angular velocity
        delta_v_lvlh -= np.cross(omega_lvlh, delta_r_lvlh)
        
        return np.concatenate([delta_r_lvlh, delta_v_lvlh])
    
    def compute_lvlh_matrix(self, chief_state: np.ndarray) -> np.ndarray:
        """Compute LVLH-to-inertial transformation matrix"""
        
        r_chief = chief_state[:3]
        v_chief = chief_state[3:6]
        
        # LVLH axes
        x_lvlh = r_chief / np.linalg.norm(r_chief)  # Radial
        z_lvlh = np.cross(r_chief, v_chief)         # Cross-track
        z_lvlh = z_lvlh / np.linalg.norm(z_lvlh)
        y_lvlh = np.cross(z_lvlh, x_lvlh)           # Along-track
        
        return np.column_stack([x_lvlh, y_lvlh, z_lvlh])
```

---

## Disturbance Models

### Atmospheric Drag
```python
class AtmosphericDrag:
    """Atmospheric drag model for LEO operations"""
    
    def __init__(self, config: Dict):
        self.drag_coefficient = config['drag_coefficient']
        self.reference_area = config['reference_area']
        self.atmosphere_model = config.get('atmosphere_model', 'exponential')
    
    def compute_drag_acceleration(self, position: np.ndarray, velocity: np.ndarray,
                                 spacecraft_mass: float, altitude: float = None) -> np.ndarray:
        """Compute atmospheric drag acceleration"""
        
        if altitude is None:
            altitude = np.linalg.norm(position) - 6371000  # Earth radius
        
        # Atmospheric density
        density = self.atmospheric_density(altitude)
        
        # Relative velocity (subtract Earth rotation)
        v_rel = velocity - self.earth_rotation_velocity(position)
        v_rel_magnitude = np.linalg.norm(v_rel)
        
        if v_rel_magnitude < 1e-10:
            return np.zeros(3)
        
        # Drag force
        drag_force = -0.5 * density * self.drag_coefficient * self.reference_area * v_rel_magnitude * v_rel
        
        return drag_force / spacecraft_mass
    
    def atmospheric_density(self, altitude: float) -> float:
        """Atmospheric density model"""
        
        if self.atmosphere_model == 'exponential':
            # Simple exponential model
            rho_0 = 1.225  # kg/m³ at sea level
            H = 8000       # Scale height (m)
            return rho_0 * np.exp(-altitude / H)
            
        elif self.atmosphere_model == 'harris_priester':
            # Harris-Priester model (more accurate)
            return self.harris_priester_density(altitude)
        
        else:
            raise ValueError(f"Unknown atmosphere model: {self.atmosphere_model}")
    
    def harris_priester_density(self, altitude: float) -> float:
        """Harris-Priester atmospheric density model"""
        
        # Altitude ranges and density values (simplified)
        altitude_km = altitude / 1000.0
        
        if altitude_km < 100:
            return 0.0  # Below atmosphere
        elif altitude_km < 200:
            return 5e-7 * np.exp(-(altitude_km - 175) / 40)
        elif altitude_km < 500:
            return 2e-11 * np.exp(-(altitude_km - 350) / 60)
        else:
            return 1e-15 * np.exp(-(altitude_km - 500) / 100)
```

### Solar Radiation Pressure
```python
class SolarRadiationPressure:
    """Solar radiation pressure disturbance model"""
    
    def __init__(self, config: Dict):
        self.solar_constant = config.get('solar_constant', 1361)  # W/m²
        self.spacecraft_area = config['spacecraft_area']
        self.reflectivity_coefficient = config.get('reflectivity_coefficient', 1.3)
        self.eclipse_model = config.get('eclipse_model', True)
    
    def compute_srp_acceleration(self, position: np.ndarray, sun_vector: np.ndarray,
                               spacecraft_mass: float, earth_position: np.ndarray = None) -> np.ndarray:
        """Compute solar radiation pressure acceleration"""
        
        # Check for eclipse
        if self.eclipse_model and self.in_earth_shadow(position, sun_vector, earth_position):
            return np.zeros(3)
        
        # Distance from sun (AU)
        sun_distance_au = 1.0  # Assume ~1 AU for Earth vicinity
        
        # Solar flux at spacecraft
        solar_flux = self.solar_constant / sun_distance_au**2
        
        # Unit vector from sun to spacecraft
        sun_direction = sun_vector / np.linalg.norm(sun_vector)
        
        # SRP acceleration
        pressure = solar_flux / 299792458  # Solar flux / speed of light
        force_magnitude = pressure * self.spacecraft_area * self.reflectivity_coefficient
        
        srp_acceleration = force_magnitude * sun_direction / spacecraft_mass
        
        return srp_acceleration
    
    def in_earth_shadow(self, position: np.ndarray, sun_vector: np.ndarray,
                       earth_position: np.ndarray = None) -> bool:
        """Check if spacecraft is in Earth's shadow"""
        
        if earth_position is None:
            earth_position = np.zeros(3)
        
        # Vector from Earth to spacecraft
        earth_to_spacecraft = position - earth_position
        
        # Vector from Earth to Sun
        earth_to_sun = -sun_vector  # Assuming sun_vector points from sun to spacecraft
        
        # Project spacecraft position onto Earth-Sun line
        projection_length = np.dot(earth_to_spacecraft, earth_to_sun) / np.linalg.norm(earth_to_sun)
        
        # If projection is negative, spacecraft is on sun side
        if projection_length < 0:
            return False
        
        # Distance from spacecraft to Earth-Sun line
        earth_sun_unit = earth_to_sun / np.linalg.norm(earth_to_sun)
        projection_vector = projection_length * earth_sun_unit
        perpendicular_distance = np.linalg.norm(earth_to_spacecraft - projection_vector)
        
        # Earth radius
        earth_radius = 6371000  # m
        
        # Check if perpendicular distance is less than Earth radius
        return perpendicular_distance < earth_radius
```

### Gravitational Perturbations
```python
class GravitationalPerturbations:
    """Higher-order gravitational perturbations (J2, J3, etc.)"""
    
    def __init__(self, config: Dict):
        self.include_j2 = config.get('include_j2', True)
        self.include_j3 = config.get('include_j3', False)
        self.earth_radius = 6378137.0  # WGS84 Earth radius (m)
        self.mu = 3.986004418e14       # Earth gravitational parameter
        self.j2 = 1.0826359e-3         # J2 coefficient
        self.j3 = -2.5323e-6           # J3 coefficient
    
    def compute_perturbation_acceleration(self, position: np.ndarray) -> np.ndarray:
        """Compute gravitational perturbation accelerations"""
        
        r = np.linalg.norm(position)
        x, y, z = position
        
        acceleration = np.zeros(3)
        
        if self.include_j2:
            acceleration += self.j2_acceleration(position, r)
        
        if self.include_j3:
            acceleration += self.j3_acceleration(position, r)
        
        return acceleration
    
    def j2_acceleration(self, position: np.ndarray, r: float) -> np.ndarray:
        """J2 perturbation acceleration"""
        
        x, y, z = position
        factor = -1.5 * self.j2 * self.mu * self.earth_radius**2 / r**5
        
        a_j2 = factor * np.array([
            x * (1 - 5*z**2/r**2),
            y * (1 - 5*z**2/r**2),
            z * (3 - 5*z**2/r**2)
        ])
        
        return a_j2
    
    def j3_acceleration(self, position: np.ndarray, r: float) -> np.ndarray:
        """J3 perturbation acceleration"""
        
        x, y, z = position
        factor = -2.5 * self.j3 * self.mu * self.earth_radius**3 / r**7
        
        a_j3 = factor * np.array([
            x * z * (3 - 7*z**2/r**2),
            y * z * (3 - 7*z**2/r**2), 
            6*z**2 - 7*z**4/r**2 - 3*r**2/5
        ])
        
        return a_j3
```

---

## Code Examples

### Basic Dynamics Propagation
```python
#!/usr/bin/env python3
"""Basic spacecraft dynamics propagation example"""

import numpy as np
import matplotlib.pyplot as plt
from src.dynamics.spacecraft_dynamics import SpacecraftDynamics

# Configure dynamics model
dynamics_config = {
    'initial_mass': 500.0,
    'inertia_matrix': np.diag([100.0, 150.0, 120.0]),
    'thruster_config': {
        'num_thrusters': 12,
        'max_thrust_per_thruster': 5.0,
        'specific_impulse': 220.0
    },
    'orbital_rate': 0.0011,
    'integration_method': 'runge_kutta_4'
}

dynamics = SpacecraftDynamics(dynamics_config)

# Initial conditions: approach trajectory
initial_state = np.array([
    -100.0, 0.0, 0.0,    # Start 100m behind target
    0.5, 0.0, 0.0,       # Approaching at 0.5 m/s
    1.0, 0.0, 0.0, 0.0,  # Identity quaternion
    0.0, 0.0, 0.0        # No initial rotation
])

# Simple proportional control
target_position = np.array([0.0, 0.0, 0.0])
Kp_position = 0.1
Kd_velocity = 0.5

# Simulation parameters
dt = 0.1
total_time = 200.0
steps = int(total_time / dt)

# Storage arrays
time_history = []
state_history = []
control_history = []
fuel_history = []

current_state = initial_state.copy()
total_fuel_used = 0.0

for step in range(steps):
    t = step * dt
    
    # Simple control law
    position_error = target_position - current_state[:3]
    velocity_error = -current_state[3:6]  # Want to stop at target
    
    # PD control for thrust
    thrust_command = Kp_position * position_error + Kd_velocity * velocity_error
    
    # No torque control for this example
    torque_command = np.zeros(3)
    
    control = np.concatenate([thrust_command, torque_command])
    
    # Propagate dynamics
    next_state = dynamics.propagate_dynamics(current_state, control, dt)
    
    # Calculate fuel consumption
    fuel_used = dynamics.fuel_consumption(thrust_command, dt)
    total_fuel_used += fuel_used
    
    # Update mass
    dynamics.mass -= fuel_used
    
    # Store history
    time_history.append(t)
    state_history.append(current_state.copy())
    control_history.append(control.copy())
    fuel_history.append(total_fuel_used)
    
    # Update state
    current_state = next_state
    
    # Print progress
    if step % 50 == 0:
        position_error_magnitude = np.linalg.norm(position_error)
        velocity_magnitude = np.linalg.norm(current_state[3:6])
        print(f"t={t:6.1f}s: Position error={position_error_magnitude:6.3f}m, "
              f"Velocity={velocity_magnitude:6.3f}m/s, Fuel used={total_fuel_used:.4f}kg")
    
    # Check convergence
    if np.linalg.norm(position_error) < 0.1 and np.linalg.norm(current_state[3:6]) < 0.01:
        print(f"\\nTarget reached at t={t:.1f}s!")
        print(f"Final position error: {np.linalg.norm(position_error):.4f}m")
        print(f"Total fuel used: {total_fuel_used:.4f}kg")
        break

# Plot results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Convert to numpy arrays
states = np.array(state_history)
controls = np.array(control_history)

# Position trajectory
ax1.plot(time_history, states[:, 0], 'r-', label='X')
ax1.plot(time_history, states[:, 1], 'g-', label='Y')
ax1.plot(time_history, states[:, 2], 'b-', label='Z')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Position (m)')
ax1.set_title('Position vs Time')
ax1.legend()
ax1.grid(True)

# Velocity trajectory
ax2.plot(time_history, states[:, 3], 'r-', label='Vx')
ax2.plot(time_history, states[:, 4], 'g-', label='Vy') 
ax2.plot(time_history, states[:, 5], 'b-', label='Vz')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Velocity (m/s)')
ax2.set_title('Velocity vs Time')
ax2.legend()
ax2.grid(True)

# Control inputs
ax3.plot(time_history, controls[:, 0], 'r-', label='Fx')
ax3.plot(time_history, controls[:, 1], 'g-', label='Fy')
ax3.plot(time_history, controls[:, 2], 'b-', label='Fz')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Thrust (N)')
ax3.set_title('Control Inputs')
ax3.legend()
ax3.grid(True)

# Fuel consumption
ax4.plot(time_history, fuel_history, 'k-')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Cumulative Fuel Used (kg)')
ax4.set_title('Fuel Consumption')
ax4.grid(True)

plt.tight_layout()
plt.show()
```

### Uncertainty Propagation Example
```python
#!/usr/bin/env python3
"""Uncertainty propagation through spacecraft dynamics"""

import numpy as np
from src.dynamics.spacecraft_dynamics import SpacecraftDynamics
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# Initialize dynamics
dynamics_config = {
    'initial_mass': 500.0,
    'inertia_matrix': np.diag([100.0, 150.0, 120.0]),
    'thruster_config': {'num_thrusters': 12, 'specific_impulse': 220.0},
    'orbital_rate': 0.0011,
    'mass_uncertainty': 25.0,      # ±25 kg mass uncertainty
    'integration_method': 'runge_kutta_4'
}

dynamics = SpacecraftDynamics(dynamics_config)

# Initial state distribution
mean_state = np.array([
    10.0, 5.0, 0.0,      # Position uncertainty
    0.1, 0.05, 0.0,      # Velocity uncertainty  
    1.0, 0.0, 0.0, 0.0,  # Attitude (identity quaternion)
    0.0, 0.0, 0.0        # Angular velocity
])

# Covariance matrix (uncertainties)
initial_covariance = np.eye(13) * 0.0
initial_covariance[0, 0] = 1.0    # 1m position uncertainty in x
initial_covariance[1, 1] = 0.5    # 0.5m position uncertainty in y
initial_covariance[2, 2] = 0.2    # 0.2m position uncertainty in z
initial_covariance[3, 3] = 0.01   # 0.1m/s velocity uncertainty in x
initial_covariance[4, 4] = 0.005  # 0.05m/s velocity uncertainty in y
initial_covariance[5, 5] = 0.002  # 0.02m/s velocity uncertainty in z

state_distribution = {
    'mean': mean_state,
    'covariance': initial_covariance,
    'type': 'gaussian'
}

# Control input
control = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 1N thrust in x

# Propagate uncertainty using different methods
dt = 0.1
methods = ['linearization', 'unscented', 'monte_carlo']
results = {}

for method in methods:
    print(f"Propagating uncertainty using {method}...")
    
    try:
        result = dynamics.propagate_with_uncertainty(state_distribution, control, dt, method)
        results[method] = result
        
        # Extract position uncertainties
        pos_std = np.sqrt(np.diag(result['covariance'][:3]))
        print(f"  Position uncertainties: [{pos_std[0]:.4f}, {pos_std[1]:.4f}, {pos_std[2]:.4f}] m")
        
    except Exception as e:
        print(f"  Error with {method}: {e}")

# Compare methods
if results:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot initial and propagated position uncertainties
    methods_to_plot = [m for m in methods if m in results]
    
    # Position uncertainty comparison
    ax = axes[0, 0]
    initial_pos_std = np.sqrt(np.diag(initial_covariance[:3]))
    ax.bar(range(3), initial_pos_std, alpha=0.7, label='Initial', width=0.2)
    
    for i, method in enumerate(methods_to_plot):
        propagated_std = np.sqrt(np.diag(results[method]['covariance'][:3]))
        ax.bar(np.arange(3) + 0.2*(i+1), propagated_std, alpha=0.7, 
               label=f'Propagated ({method})', width=0.2)
    
    ax.set_xlabel('Position Component')
    ax.set_ylabel('Standard Deviation (m)')
    ax.set_title('Position Uncertainty Propagation')
    ax.set_xticks(range(3))
    ax.set_xticklabels(['X', 'Y', 'Z'])
    ax.legend()
    ax.grid(True)
    
    # Velocity uncertainty comparison
    ax = axes[0, 1]
    initial_vel_std = np.sqrt(np.diag(initial_covariance[3:6]))
    ax.bar(range(3), initial_vel_std, alpha=0.7, label='Initial', width=0.2)
    
    for i, method in enumerate(methods_to_plot):
        propagated_std = np.sqrt(np.diag(results[method]['covariance'][3:6]))
        ax.bar(np.arange(3) + 0.2*(i+1), propagated_std, alpha=0.7,
               label=f'Propagated ({method})', width=0.2)
    
    ax.set_xlabel('Velocity Component') 
    ax.set_ylabel('Standard Deviation (m/s)')
    ax.set_title('Velocity Uncertainty Propagation')
    ax.set_xticks(range(3))
    ax.set_xticklabels(['Vx', 'Vy', 'Vz'])
    ax.legend()
    ax.grid(True)
    
    # Uncertainty ellipse (2D projection)
    if 'unscented' in results:
        ax = axes[1, 0]
        
        # Extract position covariances
        initial_cov_2d = initial_covariance[:2, :2]  # X-Y covariance
        propagated_cov_2d = results['unscented']['covariance'][:2, :2]
        
        # Plot uncertainty ellipses
        from matplotlib.patches import Ellipse
        
        # Initial ellipse (95% confidence)
        eigenvals, eigenvecs = np.linalg.eigh(initial_cov_2d)
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        width, height = 2 * 2.45 * np.sqrt(eigenvals)  # 95% confidence ellipse
        
        initial_ellipse = Ellipse(mean_state[:2], width, height, angle=angle,
                                facecolor='blue', alpha=0.3, label='Initial')
        ax.add_patch(initial_ellipse)
        
        # Propagated ellipse
        eigenvals, eigenvecs = np.linalg.eigh(propagated_cov_2d)
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        width, height = 2 * 2.45 * np.sqrt(eigenvals)
        
        prop_mean = results['unscented']['mean'][:2]
        prop_ellipse = Ellipse(prop_mean, width, height, angle=angle,
                             facecolor='red', alpha=0.3, label='Propagated')
        ax.add_patch(prop_ellipse)
        
        # Set axis limits
        ax.set_xlim(mean_state[0] - 5, prop_mean[0] + 5)
        ax.set_ylim(mean_state[1] - 5, prop_mean[1] + 5)
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('95% Confidence Ellipses')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
```

### Multi-Body Dynamics with Disturbances
```python
#!/usr/bin/env python3
"""Multi-body spacecraft dynamics with environmental disturbances"""

import numpy as np
from src.dynamics.spacecraft_dynamics import SpacecraftDynamics
from src.dynamics.disturbance_models import AtmosphericDrag, SolarRadiationPressure, GravitationalPerturbations

# Spacecraft configurations
spacecraft_configs = {
    'chaser': {
        'initial_mass': 500.0,
        'inertia_matrix': np.diag([100.0, 150.0, 120.0]),
        'drag_coefficient': 2.2,
        'reference_area': 8.0,
        'spacecraft_area': 12.0,
        'reflectivity_coefficient': 1.3
    },
    'target': {
        'initial_mass': 1000.0,
        'inertia_matrix': np.diag([200.0, 250.0, 220.0]),
        'drag_coefficient': 2.0,
        'reference_area': 15.0,
        'spacecraft_area': 20.0,
        'reflectivity_coefficient': 1.2
    }
}

# Initialize dynamics and disturbance models
spacecraft_dynamics = {}
drag_models = {}
srp_models = {}

for name, config in spacecraft_configs.items():
    # Dynamics model
    dynamics_config = {
        'initial_mass': config['initial_mass'],
        'inertia_matrix': config['inertia_matrix'],
        'thruster_config': {'num_thrusters': 12, 'specific_impulse': 220.0},
        'orbital_rate': 0.0011
    }
    spacecraft_dynamics[name] = SpacecraftDynamics(dynamics_config)
    
    # Drag model
    drag_config = {
        'drag_coefficient': config['drag_coefficient'],
        'reference_area': config['reference_area'],
        'atmosphere_model': 'harris_priester'
    }
    drag_models[name] = AtmosphericDrag(drag_config)
    
    # SRP model
    srp_config = {
        'spacecraft_area': config['spacecraft_area'],
        'reflectivity_coefficient': config['reflectivity_coefficient'],
        'eclipse_model': True
    }
    srp_models[name] = SolarRadiationPressure(srp_config)

# Gravitational perturbations
grav_perturbations = GravitationalPerturbations({'include_j2': True, 'include_j3': True})

# Initial states (LEO at 400 km altitude)
orbit_radius = 6771000  # Earth radius + 400 km
orbital_velocity = np.sqrt(3.986004418e14 / orbit_radius)

initial_states = {
    'chaser': np.array([
        -50.0, 0.0, 0.0,          # 50m behind target in LVLH
        0.1, 0.0, 0.0,            # Slow approach
        1.0, 0.0, 0.0, 0.0,       # Identity quaternion
        0.0, 0.0, 0.0             # No rotation
    ]),
    'target': np.array([
        0.0, 0.0, 0.0,            # Origin of LVLH frame
        0.0, 0.0, 0.0,            # Stationary in LVLH
        1.0, 0.0, 0.0, 0.0,       # Identity quaternion
        0.0, 0.0, 0.0             # No rotation
    ])
}

# Simulation parameters
dt = 1.0  # 1 second time step
simulation_time = 3600.0  # 1 hour
steps = int(simulation_time / dt)

# Sun vector (pointing from sun to Earth vicinity)
sun_vector = np.array([1.5e11, 0, 0])  # ~1 AU in x-direction

# Storage
results = {name: {'states': [], 'disturbances': []} for name in spacecraft_configs.keys()}
current_states = initial_states.copy()

print("Running multi-body simulation with disturbances...")

for step in range(steps):
    t = step * dt
    
    for name in spacecraft_configs.keys():
        state = current_states[name]
        mass = spacecraft_configs[name]['initial_mass']
        
        # Convert LVLH to inertial for disturbance calculations
        # (Simplified - assumes circular orbit for chief spacecraft)
        position_inertial = state[:3] + np.array([orbit_radius, 0, 0])
        velocity_inertial = state[3:6] + np.array([0, orbital_velocity, 0])
        altitude = np.linalg.norm(position_inertial) - 6371000
        
        # Compute disturbances
        disturbances = np.zeros(6)  # [force_xyz, torque_xyz]
        
        # Atmospheric drag
        if altitude < 500000:  # Below 500 km
            drag_acc = drag_models[name].compute_drag_acceleration(
                position_inertial, velocity_inertial, mass, altitude
            )
            disturbances[:3] += drag_acc * mass
        
        # Solar radiation pressure
        srp_acc = srp_models[name].compute_srp_acceleration(
            position_inertial, sun_vector, mass
        )
        disturbances[:3] += srp_acc * mass
        
        # Gravitational perturbations
        grav_acc = grav_perturbations.compute_perturbation_acceleration(position_inertial)
        disturbances[:3] += grav_acc * mass
        
        # Add small random torque disturbances (magnetic, solar pressure)
        disturbances[3:] += np.random.normal(0, 1e-6, 3)
        
        # Control (simple station-keeping for target, approach for chaser)
        if name == 'chaser':
            # Approach control
            position_error = -state[:3]  # Want to reach origin
            velocity_error = -state[3:6]  # Want to stop there
            control = np.concatenate([
                0.1 * position_error + 0.05 * velocity_error,  # Thrust
                np.zeros(3)  # No torque control
            ])
        else:
            # Station-keeping (minimal control)
            control = np.zeros(6)
        
        # Propagate dynamics with disturbances
        disturbance_dict = {
            'forces': disturbances[:3],
            'torques': disturbances[3:]
        }
        
        next_state = spacecraft_dynamics[name].propagate_dynamics(
            state, control, dt, disturbance_dict
        )
        
        # Store results
        results[name]['states'].append(state.copy())
        results[name]['disturbances'].append(disturbances.copy())
        
        # Update state
        current_states[name] = next_state
    
    # Print progress
    if step % 600 == 0:  # Every 10 minutes
        chaser_pos = current_states['chaser'][:3]
        distance = np.linalg.norm(chaser_pos)
        print(f"t={t/60:6.1f} min: Chaser distance = {distance:8.3f} m")

print("Simulation complete!")

# Analyze results
import matplotlib.pyplot as plt

# Convert to numpy arrays
for name in spacecraft_configs.keys():
    results[name]['states'] = np.array(results[name]['states'])
    results[name]['disturbances'] = np.array(results[name]['disturbances'])

time_vector = np.arange(steps) * dt / 60  # Convert to minutes

# Plot results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Relative position
chaser_positions = results['chaser']['states'][:, :3]
ax = axes[0, 0]
ax.plot(time_vector, chaser_positions[:, 0], 'r-', label='X')
ax.plot(time_vector, chaser_positions[:, 1], 'g-', label='Y')
ax.plot(time_vector, chaser_positions[:, 2], 'b-', label='Z')
ax.set_xlabel('Time (min)')
ax.set_ylabel('Relative Position (m)')
ax.set_title('Chaser Relative Position')
ax.legend()
ax.grid(True)

# Relative distance
distance_history = np.linalg.norm(chaser_positions, axis=1)
ax = axes[0, 1]
ax.plot(time_vector, distance_history, 'k-')
ax.set_xlabel('Time (min)')
ax.set_ylabel('Distance (m)')
ax.set_title('Chaser-Target Distance')
ax.grid(True)

# Disturbance forces on chaser
chaser_disturbances = results['chaser']['disturbances'][:, :3]
ax = axes[0, 2]
ax.plot(time_vector, chaser_disturbances[:, 0] * 1000, 'r-', label='Fx')
ax.plot(time_vector, chaser_disturbances[:, 1] * 1000, 'g-', label='Fy')
ax.plot(time_vector, chaser_disturbances[:, 2] * 1000, 'b-', label='Fz')
ax.set_xlabel('Time (min)')
ax.set_ylabel('Disturbance Force (mN)')
ax.set_title('Disturbance Forces on Chaser')
ax.legend()
ax.grid(True)

# 3D trajectory
ax = fig.add_subplot(2, 2, 3, projection='3d')
ax.plot(chaser_positions[:, 0], chaser_positions[:, 1], chaser_positions[:, 2], 'b-')
ax.scatter([0], [0], [0], color='red', s=100, label='Target')
ax.scatter([chaser_positions[0, 0]], [chaser_positions[0, 1]], [chaser_positions[0, 2]], 
           color='blue', s=50, label='Chaser Start')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Trajectory')
ax.legend()

# Disturbance magnitude comparison
disturbance_magnitudes = np.linalg.norm(chaser_disturbances, axis=1) * 1000  # mN
ax = axes[1, 1]
ax.plot(time_vector, disturbance_magnitudes, 'k-')
ax.set_xlabel('Time (min)')
ax.set_ylabel('Total Disturbance Magnitude (mN)')
ax.set_title('Total Disturbance on Chaser')
ax.grid(True)

plt.tight_layout()
plt.show()

# Summary statistics
final_distance = distance_history[-1]
max_distance = np.max(distance_history)
min_distance = np.min(distance_history)
avg_disturbance = np.mean(disturbance_magnitudes)
max_disturbance = np.max(disturbance_magnitudes)

print(f"\\nSimulation Summary:")
print(f"Final distance: {final_distance:.3f} m")
print(f"Distance range: {min_distance:.3f} - {max_distance:.3f} m") 
print(f"Average disturbance: {avg_disturbance:.3f} mN")
print(f"Maximum disturbance: {max_disturbance:.3f} mN")
```

---

*For more advanced dynamics examples and orbital mechanics utilities, see the [tutorials](../tutorials/) directory.*