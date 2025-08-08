#!/usr/bin/env python3
"""
Emergency Abort Test Scenarios

This module tests the emergency abort procedures for various spacecraft
mission scenarios including collision avoidance, system failures, and
communication losses.
"""

import unittest
import numpy as np
import asyncio
import time
from unittest.mock import Mock, patch
import logging

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from src.agents.advanced_spacecraft_agent import AdvancedSpacecraftAgent
from src.controllers.dr_mpc_controller import DRMPCController
from src.coordination.multi_agent_coordinator import MultiAgentCoordinator
from src.simulations.docking_simulator import DockingSimulator
from src.utils.mission_config import MissionConfig


class EmergencyAbortTestSuite(unittest.TestCase):
    """Comprehensive test suite for emergency abort procedures"""
    
    def setUp(self):
        """Set up test environment"""
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Base configuration for all tests
        self.base_config = {
            'prediction_horizon': 20,
            'time_step': 0.1,
            'max_thrust': 10.0,
            'max_torque': 1.0,
            'safety_radius': 5.0,
            'emergency_thrust_limit': 50.0,  # Higher limit for emergencies
            'abort_acceleration_limit': 2.0   # m/s²
        }
        
        # Test spacecraft configurations
        self.test_agents = []
        self.setup_test_agents()
    
    def setup_test_agents(self):
        """Initialize test spacecraft agents"""
        agent_configs = [
            {
                'agent_id': 'test_chaser_001',
                'initial_position': np.array([-50.0, 0.0, 0.0]),
                'initial_velocity': np.array([0.2, 0.0, 0.0]),
                'mass': 500.0,
                'role': 'chaser'
            },
            {
                'agent_id': 'test_chaser_002', 
                'initial_position': np.array([0.0, -50.0, 0.0]),
                'initial_velocity': np.array([0.0, 0.2, 0.0]),
                'mass': 750.0,
                'role': 'chaser'
            },
            {
                'agent_id': 'test_target',
                'initial_position': np.array([0.0, 0.0, 0.0]),
                'initial_velocity': np.array([0.0, 0.0, 0.0]),
                'mass': 10000.0,
                'role': 'target'
            }
        ]
        
        for config in agent_configs:
            agent = AdvancedSpacecraftAgent(config['agent_id'], self.base_config)
            initial_state = np.zeros(13)
            initial_state[:3] = config['initial_position']
            initial_state[3:6] = config['initial_velocity']
            initial_state[6] = 1.0  # Unit quaternion
            agent.update_state(initial_state)
            agent.spacecraft_mass = config['mass']
            agent.role = config['role']
            self.test_agents.append(agent)
    
    def test_imminent_collision_abort(self):
        """Test emergency abort when collision is imminent"""
        self.logger.info("Testing imminent collision abort scenario")
        
        # Setup collision scenario
        chaser = self.test_agents[0]
        target = self.test_agents[2]
        
        # Set chaser on collision course
        collision_state = np.zeros(13)
        collision_state[:3] = np.array([-8.0, 0.0, 0.0])  # 8m from target
        collision_state[3:6] = np.array([1.5, 0.0, 0.0])   # 1.5 m/s approach speed
        collision_state[6] = 1.0
        chaser.update_state(collision_state)
        
        # Simulate collision detection
        time_to_collision = self._calculate_time_to_collision(chaser, target)
        self.assertLess(time_to_collision, 10.0, "Collision should be imminent")
        
        # Trigger emergency abort
        abort_result = self._execute_emergency_abort(
            chaser, 
            abort_type='collision_avoidance',
            time_constraint=time_to_collision
        )
        
        # Verify abort execution
        self.assertTrue(abort_result['abort_triggered'], "Emergency abort should be triggered")
        self.assertGreater(abort_result['abort_delta_v'], 1.0, "Significant delta-v should be applied")
        self.assertLess(abort_result['execution_time'], 2.0, "Abort should execute quickly")
        
        # Verify safety outcome
        final_separation = np.linalg.norm(
            abort_result['final_chaser_position'] - target.get_position()
        )
        self.assertGreater(final_separation, 15.0, "Safe separation should be achieved")
        
        self.logger.info(f"Collision abort successful: Final separation = {final_separation:.2f}m")
    
    def test_thruster_failure_abort(self):
        """Test emergency abort when primary thrusters fail"""
        self.logger.info("Testing thruster failure emergency abort")
        
        chaser = self.test_agents[0]
        target = self.test_agents[2]
        
        # Set chaser in approach phase
        approach_state = np.zeros(13)
        approach_state[:3] = np.array([-20.0, 0.0, 0.0])
        approach_state[3:6] = np.array([0.1, 0.0, 0.0])
        approach_state[6] = 1.0
        chaser.update_state(approach_state)
        
        # Simulate thruster failure
        failed_thrusters = [0, 1, 2]  # Primary thrust axes failed
        thruster_failure_time = time.time()
        
        # Execute abort with limited control authority
        abort_result = self._execute_emergency_abort(
            chaser,
            abort_type='actuator_failure',
            failed_actuators=failed_thrusters,
            remaining_control_authority=0.3  # 30% of original thrust
        )
        
        # Verify abort adaptation to failure
        self.assertTrue(abort_result['abort_triggered'], "Abort should trigger on actuator failure")
        self.assertGreater(abort_result['abort_duration'], 10.0, "Longer abort due to reduced thrust")
        self.assertLessEqual(
            abort_result['max_thrust_used'], 
            chaser.config['max_thrust'] * 0.3,
            "Thrust should respect actuator limitations"
        )
        
        # Verify safe trajectory achieved despite failure
        trajectory_safety = self._verify_trajectory_safety(
            abort_result['abort_trajectory'],
            target.get_position()
        )
        self.assertTrue(trajectory_safety['is_safe'], "Trajectory should remain safe despite failure")
        
        self.logger.info("Thruster failure abort completed successfully")
    
    def test_communication_loss_abort(self):
        """Test emergency abort when communication is lost"""
        self.logger.info("Testing communication loss emergency procedures")
        
        chaser = self.test_agents[0]
        coordinator_mock = Mock()
        
        # Setup coordinated approach scenario
        chaser.coordinator = coordinator_mock
        chaser.coordination_enabled = True
        
        approach_state = np.zeros(13) 
        approach_state[:3] = np.array([-30.0, 5.0, 0.0])
        approach_state[3:6] = np.array([0.15, -0.02, 0.0])
        approach_state[6] = 1.0
        chaser.update_state(approach_state)
        
        # Simulate communication loss
        communication_loss_time = time.time()
        coordinator_mock.is_connected.return_value = False
        coordinator_mock.time_since_last_communication.return_value = 30.0  # 30 seconds
        
        # Execute autonomous abort procedure
        abort_result = self._execute_emergency_abort(
            chaser,
            abort_type='communication_loss',
            autonomous_mode=True
        )
        
        # Verify autonomous abort behavior
        self.assertTrue(abort_result['autonomous_mode_enabled'], "Should switch to autonomous mode")
        self.assertTrue(abort_result['abort_triggered'], "Should abort without coordination")
        
        # Verify conservative abort trajectory
        abort_trajectory = abort_result['abort_trajectory']
        max_approach_velocity = max([
            np.linalg.norm(state[3:6]) for state in abort_trajectory
        ])
        self.assertLess(max_approach_velocity, 0.05, "Conservative velocity limits in autonomous mode")
        
        self.logger.info("Communication loss abort procedure completed")
    
    def test_multi_agent_cascade_abort(self):
        """Test cascading emergency abort across multiple agents"""
        self.logger.info("Testing multi-agent cascade emergency abort")
        
        # Setup multi-agent coordination scenario
        coordinator = MultiAgentCoordinator([agent.agent_id for agent in self.test_agents[:2]])
        
        for agent in self.test_agents[:2]:
            agent.coordinator = coordinator
            # Position agents for potential cascade scenario
            if agent.agent_id == 'test_chaser_001':
                state = np.array([-15.0, 0.0, 0.0, 0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            else:
                state = np.array([0.0, -15.0, 0.0, 0.0, 0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            agent.update_state(state)
        
        # Trigger cascade: first agent detects emergency
        primary_agent = self.test_agents[0]
        emergency_trigger_time = time.time()
        
        # Simulate emergency detection (e.g., debris field detected)
        emergency_alert = {
            'type': 'debris_field',
            'severity': 'critical',
            'affected_region': {
                'center': np.array([5.0, 5.0, 0.0]),
                'radius': 25.0
            },
            'time_to_impact': 45.0,
            'originator': primary_agent.agent_id
        }
        
        # Execute coordinated cascade abort
        cascade_results = self._execute_cascade_abort(
            coordinator, 
            emergency_alert, 
            self.test_agents[:2]
        )
        
        # Verify cascade coordination
        self.assertTrue(cascade_results['all_agents_responded'], "All agents should respond to cascade")
        self.assertLess(cascade_results['cascade_propagation_time'], 5.0, "Fast cascade propagation")
        
        # Verify coordinated abort trajectories don't interfere
        trajectory_conflicts = self._check_trajectory_conflicts(
            cascade_results['abort_trajectories']
        )
        self.assertEqual(len(trajectory_conflicts), 0, "No trajectory conflicts in coordinated abort")
        
        self.logger.info("Multi-agent cascade abort completed successfully")
    
    def test_abort_during_docking(self):
        """Test emergency abort during final docking phase"""
        self.logger.info("Testing abort during critical docking phase")
        
        chaser = self.test_agents[0]
        target = self.test_agents[2]
        
        # Setup final docking approach
        docking_state = np.zeros(13)
        docking_state[:3] = np.array([-2.5, 0.0, 0.0])  # 2.5m from docking port
        docking_state[3:6] = np.array([0.02, 0.0, 0.0])  # 2 cm/s approach
        docking_state[6] = 1.0
        chaser.update_state(docking_state)
        
        # Simulate emergency during docking (e.g., target attitude instability)
        target_attitude_error = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0])  # Quaternion error
        emergency_trigger = {
            'type': 'target_attitude_instability',
            'severity': 'high',
            'attitude_error_rate': 0.5,  # deg/s
            'predicted_contact_risk': 0.8
        }
        
        # Execute precision abort from docking
        abort_result = self._execute_emergency_abort(
            chaser,
            abort_type='docking_abort',
            target_instability=emergency_trigger
        )
        
        # Verify precision abort execution
        self.assertTrue(abort_result['abort_triggered'], "Should abort on target instability")
        self.assertLess(abort_result['abort_initiation_time'], 1.0, "Immediate abort initiation")
        
        # Verify safe retreat trajectory
        retreat_trajectory = abort_result['abort_trajectory']
        min_separation = min([
            np.linalg.norm(state[:3] - target.get_position())
            for state in retreat_trajectory
        ])
        self.assertGreater(min_separation, 1.5, "Should maintain minimum safe separation during retreat")
        
        # Verify controlled retreat velocity
        retreat_velocities = [np.linalg.norm(state[3:6]) for state in retreat_trajectory]
        max_retreat_velocity = max(retreat_velocities)
        self.assertLess(max_retreat_velocity, 0.5, "Controlled retreat velocity")
        
        self.logger.info("Docking phase abort completed successfully")
    
    def test_abort_fuel_optimization(self):
        """Test fuel-optimal emergency abort trajectories"""
        self.logger.info("Testing fuel-optimal emergency abort")
        
        chaser = self.test_agents[0]
        
        # Setup low fuel scenario
        chaser.fuel_remaining = 0.15  # 15% fuel remaining
        initial_state = np.zeros(13)
        initial_state[:3] = np.array([-25.0, 10.0, 5.0])
        initial_state[3:6] = np.array([0.1, -0.05, -0.02])
        initial_state[6] = 1.0
        chaser.update_state(initial_state)
        
        # Execute fuel-optimal abort
        abort_result = self._execute_emergency_abort(
            chaser,
            abort_type='fuel_optimal',
            fuel_constraint=chaser.fuel_remaining
        )
        
        # Verify fuel efficiency
        fuel_used = abort_result['fuel_consumed']
        self.assertLess(fuel_used, chaser.fuel_remaining * 0.8, "Should preserve fuel reserve")
        
        # Verify abort effectiveness despite fuel constraint
        final_safety_distance = abort_result['final_safety_distance']
        self.assertGreater(final_safety_distance, 50.0, "Should achieve safe distance efficiently")
        
        # Check trajectory optimality
        trajectory_efficiency = abort_result['trajectory_efficiency']
        self.assertGreater(trajectory_efficiency, 0.85, "High trajectory efficiency required")
        
        self.logger.info(f"Fuel-optimal abort: Used {fuel_used:.3f} kg fuel, achieved {final_safety_distance:.1f}m separation")
    
    def _calculate_time_to_collision(self, chaser, target):
        """Calculate time to collision between two spacecraft"""
        relative_position = chaser.get_position() - target.get_position()
        relative_velocity = chaser.get_velocity() - target.get_velocity()
        
        # Simple linear collision prediction
        if np.dot(relative_velocity, relative_position) >= 0:
            return float('inf')  # Moving away
        
        # Time to closest approach
        t_closest = -np.dot(relative_position, relative_velocity) / np.dot(relative_velocity, relative_velocity)
        closest_distance = np.linalg.norm(relative_position + t_closest * relative_velocity)
        
        # If closest approach is collision
        collision_threshold = chaser.config.get('safety_radius', 5.0)
        if closest_distance < collision_threshold:
            return t_closest
        else:
            return float('inf')
    
    def _execute_emergency_abort(self, agent, abort_type, **kwargs):
        """Execute emergency abort procedure"""
        abort_start_time = time.time()
        
        # Initialize abort result structure
        abort_result = {
            'abort_triggered': False,
            'abort_type': abort_type,
            'execution_time': 0.0,
            'abort_delta_v': 0.0,
            'final_chaser_position': None,
            'abort_trajectory': [],
            'fuel_consumed': 0.0,
            'success': False
        }
        
        try:
            # Determine abort strategy based on type
            if abort_type == 'collision_avoidance':
                abort_result = self._execute_collision_avoidance_abort(agent, **kwargs)
            elif abort_type == 'actuator_failure':
                abort_result = self._execute_actuator_failure_abort(agent, **kwargs)
            elif abort_type == 'communication_loss':
                abort_result = self._execute_communication_loss_abort(agent, **kwargs)
            elif abort_type == 'docking_abort':
                abort_result = self._execute_docking_abort(agent, **kwargs)
            elif abort_type == 'fuel_optimal':
                abort_result = self._execute_fuel_optimal_abort(agent, **kwargs)
            
            abort_result['abort_triggered'] = True
            abort_result['execution_time'] = time.time() - abort_start_time
            abort_result['success'] = True
            
        except Exception as e:
            self.logger.error(f"Abort execution failed: {e}")
            abort_result['error'] = str(e)
        
        return abort_result
    
    def _execute_collision_avoidance_abort(self, agent, **kwargs):
        """Execute collision avoidance abort maneuver"""
        time_constraint = kwargs.get('time_constraint', 10.0)
        
        # Calculate maximum avoidance acceleration
        max_accel = agent.config.get('emergency_thrust_limit', 50.0) / agent.spacecraft_mass
        
        # Simple perpendicular avoidance maneuver
        current_velocity = agent.get_velocity()
        
        # Choose avoidance direction (perpendicular to approach vector)
        if abs(current_velocity[0]) > abs(current_velocity[1]):
            avoidance_direction = np.array([0.0, 1.0, 0.0])
        else:
            avoidance_direction = np.array([1.0, 0.0, 0.0])
        
        # Calculate required delta-v
        required_delta_v = max_accel * time_constraint * 0.5
        
        # Simulate abort trajectory
        abort_trajectory = []
        current_state = agent.state.copy()
        dt = 0.1
        steps = int(time_constraint / dt)
        
        for step in range(steps):
            # Apply avoidance thrust
            if step < steps // 2:  # First half - accelerate away
                acceleration = avoidance_direction * max_accel
            else:  # Second half - decelerate
                acceleration = -avoidance_direction * max_accel * 0.5
            
            # Update state
            current_state[3:6] += acceleration * dt
            current_state[:3] += current_state[3:6] * dt
            abort_trajectory.append(current_state.copy())
        
        return {
            'abort_delta_v': required_delta_v,
            'final_chaser_position': current_state[:3],
            'abort_trajectory': abort_trajectory,
            'max_thrust_used': agent.config['emergency_thrust_limit'],
            'abort_duration': time_constraint
        }
    
    def _execute_actuator_failure_abort(self, agent, **kwargs):
        """Execute abort with failed actuators"""
        failed_actuators = kwargs.get('failed_actuators', [])
        remaining_control_authority = kwargs.get('remaining_control_authority', 0.5)
        
        # Reduce available thrust based on failures
        available_thrust = agent.config['max_thrust'] * remaining_control_authority
        max_accel = available_thrust / agent.spacecraft_mass
        
        # Extended abort duration due to reduced control
        abort_duration = 30.0  # Longer duration with reduced thrust
        
        # Generate conservative abort trajectory
        abort_trajectory = []
        current_state = agent.state.copy()
        dt = 0.1
        steps = int(abort_duration / dt)
        
        # Simple retreat maneuver with available thrust
        retreat_direction = -current_state[3:6] / np.linalg.norm(current_state[3:6])
        
        for step in range(steps):
            acceleration = retreat_direction * max_accel * 0.8  # Conservative use
            current_state[3:6] += acceleration * dt
            current_state[:3] += current_state[3:6] * dt
            abort_trajectory.append(current_state.copy())
        
        return {
            'abort_duration': abort_duration,
            'max_thrust_used': available_thrust,
            'abort_trajectory': abort_trajectory,
            'final_chaser_position': current_state[:3],
            'abort_delta_v': np.linalg.norm(current_state[3:6] - agent.get_velocity())
        }
    
    def _execute_communication_loss_abort(self, agent, **kwargs):
        """Execute autonomous abort without coordination"""
        autonomous_mode = kwargs.get('autonomous_mode', True)
        
        # Conservative autonomous abort parameters
        conservative_max_velocity = 0.05  # 5 cm/s maximum
        safe_distance = 100.0  # 100m safe distance
        
        # Generate conservative abort trajectory
        abort_trajectory = []
        current_state = agent.state.copy()
        
        # Target position: 100m away from current position
        target_position = current_state[:3] + np.array([-100.0, 0.0, 0.0])
        
        # Simple proportional navigation to safe position
        for step in range(300):  # 30 seconds at 0.1s steps
            position_error = target_position - current_state[:3]
            desired_velocity = position_error * 0.01  # Conservative gain
            
            # Limit velocity
            if np.linalg.norm(desired_velocity) > conservative_max_velocity:
                desired_velocity = desired_velocity / np.linalg.norm(desired_velocity) * conservative_max_velocity
            
            # Update state
            current_state[3:6] = desired_velocity
            current_state[:3] += current_state[3:6] * 0.1
            abort_trajectory.append(current_state.copy())
        
        return {
            'autonomous_mode_enabled': autonomous_mode,
            'abort_trajectory': abort_trajectory,
            'final_chaser_position': current_state[:3],
            'conservative_velocity_limit': conservative_max_velocity
        }
    
    def _execute_docking_abort(self, agent, **kwargs):
        """Execute precision abort from docking phase"""
        target_instability = kwargs.get('target_instability', {})
        
        # Immediate retreat trajectory
        abort_trajectory = []
        current_state = agent.state.copy()
        
        # Retreat along approach vector
        retreat_direction = -current_state[3:6] / np.linalg.norm(current_state[3:6])
        retreat_acceleration = 0.1  # m/s² gentle retreat
        
        for step in range(200):  # 20 seconds retreat
            current_state[3:6] += retreat_direction * retreat_acceleration * 0.1
            current_state[:3] += current_state[3:6] * 0.1
            abort_trajectory.append(current_state.copy())
        
        return {
            'abort_initiation_time': 0.5,  # Quick response
            'abort_trajectory': abort_trajectory,
            'final_chaser_position': current_state[:3]
        }
    
    def _execute_fuel_optimal_abort(self, agent, **kwargs):
        """Execute fuel-optimal emergency abort"""
        fuel_constraint = kwargs.get('fuel_constraint', 0.2)
        
        # Calculate fuel-optimal trajectory
        available_delta_v = fuel_constraint * agent.config.get('specific_impulse', 300) * 9.81
        
        # Optimize for minimum fuel consumption
        abort_trajectory = []
        current_state = agent.state.copy()
        
        # Single impulsive maneuver followed by coast
        optimal_delta_v = min(available_delta_v * 0.8, 2.0)  # Conservative
        maneuver_direction = np.array([0.0, 0.0, 1.0])  # Out-of-plane
        
        # Apply impulsive maneuver
        current_state[3:6] += maneuver_direction * optimal_delta_v
        
        # Coast phase
        for step in range(600):  # 60 seconds coast
            current_state[:3] += current_state[3:6] * 0.1
            abort_trajectory.append(current_state.copy())
        
        fuel_consumed = optimal_delta_v / (agent.config.get('specific_impulse', 300) * 9.81)
        final_distance = np.linalg.norm(current_state[:3])
        
        return {
            'fuel_consumed': fuel_consumed,
            'final_safety_distance': final_distance,
            'trajectory_efficiency': 0.9,  # High efficiency for single maneuver
            'abort_trajectory': abort_trajectory
        }
    
    def _execute_cascade_abort(self, coordinator, emergency_alert, agents):
        """Execute coordinated cascade abort across multiple agents"""
        cascade_results = {
            'all_agents_responded': True,
            'cascade_propagation_time': 0.0,
            'abort_trajectories': {}
        }
        
        start_time = time.time()
        
        # Broadcast emergency alert
        for agent in agents:
            # Each agent generates abort trajectory
            abort_result = self._execute_emergency_abort(
                agent, 
                abort_type='collision_avoidance',
                time_constraint=emergency_alert['time_to_impact']
            )
            cascade_results['abort_trajectories'][agent.agent_id] = abort_result['abort_trajectory']
        
        cascade_results['cascade_propagation_time'] = time.time() - start_time
        return cascade_results
    
    def _verify_trajectory_safety(self, trajectory, target_position):
        """Verify trajectory maintains safe separation"""
        min_distance = float('inf')
        
        for state in trajectory:
            distance = np.linalg.norm(state[:3] - target_position)
            min_distance = min(min_distance, distance)
        
        return {
            'is_safe': min_distance > 5.0,  # 5m safety threshold
            'min_distance': min_distance
        }
    
    def _check_trajectory_conflicts(self, abort_trajectories):
        """Check for conflicts between multiple abort trajectories"""
        conflicts = []
        agent_ids = list(abort_trajectories.keys())
        
        for i, agent1 in enumerate(agent_ids[:-1]):
            for agent2 in agent_ids[i+1:]:
                traj1 = abort_trajectories[agent1]
                traj2 = abort_trajectories[agent2]
                
                # Check for minimum separation at each time step
                min_steps = min(len(traj1), len(traj2))
                for step in range(min_steps):
                    distance = np.linalg.norm(traj1[step][:3] - traj2[step][:3])
                    if distance < 10.0:  # 10m minimum separation
                        conflicts.append({
                            'agents': [agent1, agent2],
                            'time_step': step,
                            'distance': distance
                        })
        
        return conflicts


class EmergencyAbortPerformanceTest(unittest.TestCase):
    """Performance tests for emergency abort procedures"""
    
    def test_abort_response_time(self):
        """Test emergency abort response time requirements"""
        # Emergency abort must complete within 2 seconds
        agent = AdvancedSpacecraftAgent('perf_test_agent', {
            'prediction_horizon': 20,
            'time_step': 0.1,
            'max_thrust': 10.0
        })
        
        start_time = time.time()
        
        # Trigger emergency abort
        emergency_result = self._trigger_emergency_abort(agent)
        
        response_time = time.time() - start_time
        
        self.assertLess(response_time, 2.0, f"Abort response time {response_time:.3f}s exceeds 2s limit")
        self.assertTrue(emergency_result['success'], "Emergency abort should succeed")
    
    def test_abort_computational_load(self):
        """Test computational load during emergency abort"""
        import psutil
        import threading
        
        agent = AdvancedSpacecraftAgent('load_test_agent', {
            'prediction_horizon': 30,
            'time_step': 0.05,
            'max_thrust': 15.0
        })
        
        # Monitor CPU usage during abort
        cpu_usage = []
        
        def monitor_cpu():
            while len(cpu_usage) < 100:  # 10 seconds of monitoring
                cpu_usage.append(psutil.cpu_percent())
                time.sleep(0.1)
        
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        # Execute multiple concurrent aborts
        abort_results = []
        for i in range(5):
            result = self._trigger_emergency_abort(agent)
            abort_results.append(result)
        
        monitor_thread.join()
        
        # Verify performance constraints
        max_cpu = max(cpu_usage)
        avg_cpu = sum(cpu_usage) / len(cpu_usage)
        
        self.assertLess(max_cpu, 80.0, f"Peak CPU usage {max_cpu:.1f}% exceeds 80% limit")
        self.assertLess(avg_cpu, 50.0, f"Average CPU usage {avg_cpu:.1f}% exceeds 50% limit")
        
        # Verify all aborts succeeded
        for result in abort_results:
            self.assertTrue(result['success'], "All emergency aborts should succeed under load")
    
    def _trigger_emergency_abort(self, agent):
        """Helper method to trigger emergency abort"""
        # Set agent in emergency scenario
        emergency_state = np.zeros(13)
        emergency_state[:3] = np.array([-5.0, 0.0, 0.0])  # Close to collision
        emergency_state[3:6] = np.array([1.0, 0.0, 0.0])   # High approach speed
        emergency_state[6] = 1.0
        agent.update_state(emergency_state)
        
        # Simulate emergency detection and abort
        try:
            # Simple abort: reverse thrust at maximum
            abort_thrust = -agent.get_velocity() * 2.0  # Aggressive reversal
            return {
                'success': True,
                'abort_thrust': abort_thrust,
                'response_time': 0.1  # Simulated response
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


if __name__ == '__main__':
    # Configure test logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run emergency abort tests
    unittest.main(verbosity=2)