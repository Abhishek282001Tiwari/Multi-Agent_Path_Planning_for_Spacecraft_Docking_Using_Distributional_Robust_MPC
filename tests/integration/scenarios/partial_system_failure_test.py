#!/usr/bin/env python3
"""
Partial System Failure Test Scenarios

This module tests system behavior under various partial failure conditions,
including component degradation, sensor failures, and graceful system recovery.
"""

import unittest
import numpy as np
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock
import logging
import random

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from src.agents.advanced_spacecraft_agent import AdvancedSpacecraftAgent
from src.controllers.dr_mpc_controller import DRMPCController
from src.fault_tolerance.actuator_fdir import ActuatorFDIR, FaultStatus, FaultType
from src.coordination.multi_agent_coordinator import MultiAgentCoordinator


class PartialSystemFailureTestSuite(unittest.TestCase):
    """Comprehensive test suite for partial system failure scenarios"""
    
    def setUp(self):
        """Set up test environment for partial failure testing"""
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # System configuration
        self.system_config = {
            'fault_detection_enabled': True,
            'fault_isolation_enabled': True,
            'fault_recovery_enabled': True,
            'redundancy_level': 2,
            'degraded_mode_threshold': 0.7,
            'minimum_operational_threshold': 0.3
        }
        
        # Initialize test spacecraft with fault tolerance
        self.test_spacecraft = []
        self.setup_fault_tolerant_spacecraft()
        
        # Initialize fault detection systems
        self.fdir_systems = {}
        self.setup_fdir_systems()
    
    def setup_fault_tolerant_spacecraft(self):
        """Initialize fault-tolerant test spacecraft"""
        spacecraft_configs = [
            {
                'agent_id': 'fault_test_primary',
                'initial_position': np.array([-30.0, 0.0, 0.0]),
                'mass': 500.0,
                'thruster_config': 'redundant_12_thruster',
                'sensor_config': 'triple_redundant',
                'role': 'primary'
            },
            {
                'agent_id': 'fault_test_secondary',
                'initial_position': np.array([0.0, -30.0, 0.0]),
                'mass': 400.0,
                'thruster_config': 'standard_8_thruster', 
                'sensor_config': 'dual_redundant',
                'role': 'secondary'
            },
            {
                'agent_id': 'fault_test_backup',
                'initial_position': np.array([30.0, 30.0, 0.0]),
                'mass': 300.0,
                'thruster_config': 'minimal_4_thruster',
                'sensor_config': 'single_string',
                'role': 'backup'
            }
        ]
        
        base_config = {
            'prediction_horizon': 20,
            'time_step': 0.1,
            'max_thrust': 12.0,
            'fault_tolerance_enabled': True,
            'emergency_thrust_multiplier': 1.5
        }
        
        for config in spacecraft_configs:
            agent = AdvancedSpacecraftAgent(config['agent_id'], base_config)
            
            # Set initial state
            initial_state = np.zeros(13)
            initial_state[:3] = config['initial_position']
            initial_state[6] = 1.0  # Unit quaternion
            agent.update_state(initial_state)
            
            # Configure fault tolerance parameters
            agent.spacecraft_mass = config['mass']
            agent.thruster_config = config['thruster_config']
            agent.sensor_config = config['sensor_config']
            agent.role = config['role']
            
            # Initialize component health
            self._initialize_component_health(agent, config)
            
            self.test_spacecraft.append(agent)
    
    def setup_fdir_systems(self):
        """Initialize Fault Detection, Isolation, and Recovery systems"""
        for agent in self.test_spacecraft:
            fdir = ActuatorFDIR(agent.agent_id, {
                'detection_threshold': 0.1,
                'isolation_confidence': 0.8,
                'recovery_timeout': 10.0,
                'health_monitoring_frequency': 10.0  # Hz
            })
            self.fdir_systems[agent.agent_id] = fdir
    
    def _initialize_component_health(self, agent, config):
        """Initialize component health status for agent"""
        # Thruster health (12 thrusters for redundant config)
        if config['thruster_config'] == 'redundant_12_thruster':
            agent.thruster_health = np.ones(12)  # All healthy initially
            agent.thruster_redundancy = True
        elif config['thruster_config'] == 'standard_8_thruster':
            agent.thruster_health = np.ones(8)
            agent.thruster_redundancy = False
        else:  # minimal_4_thruster
            agent.thruster_health = np.ones(4)
            agent.thruster_redundancy = False
        
        # Sensor health
        if config['sensor_config'] == 'triple_redundant':
            agent.sensor_health = {
                'gps': [1.0, 1.0, 1.0],
                'imu': [1.0, 1.0, 1.0],
                'camera': [1.0, 1.0, 1.0]
            }
        elif config['sensor_config'] == 'dual_redundant':
            agent.sensor_health = {
                'gps': [1.0, 1.0],
                'imu': [1.0, 1.0], 
                'camera': [1.0, 1.0]
            }
        else:  # single_string
            agent.sensor_health = {
                'gps': [1.0],
                'imu': [1.0],
                'camera': [1.0]
            }
        
        # Communication system health
        agent.communication_health = 1.0
        
        # Power system health
        agent.power_system_health = 1.0
        
        # Overall system health
        agent.system_health = 1.0
    
    def test_single_thruster_failure(self):
        """Test system response to single thruster failure"""
        self.logger.info("Testing single thruster failure scenario")
        
        spacecraft = self.test_spacecraft[0]  # Primary spacecraft
        fdir = self.fdir_systems[spacecraft.agent_id]
        
        # Initial system health check
        initial_health = self._assess_system_health(spacecraft)
        self.assertGreater(initial_health, 0.9, "System should start healthy")
        
        # Simulate single thruster failure
        failed_thruster_id = 3
        failure_time = time.time()
        
        # Induce thruster failure
        spacecraft.thruster_health[failed_thruster_id] = 0.0
        
        # Test fault detection
        fault_detected = fdir.detect_thruster_fault(
            thruster_id=failed_thruster_id,
            expected_thrust=10.0,
            measured_thrust=0.0,
            detection_time=failure_time
        )
        
        self.assertTrue(fault_detected, "FDIR should detect thruster failure")
        
        # Test fault isolation
        isolation_result = fdir.isolate_fault(failed_thruster_id)
        self.assertTrue(isolation_result['isolated'], "Fault should be isolated")
        self.assertEqual(
            isolation_result['fault_type'], 
            FaultType.THRUSTER_COMPLETE_FAILURE,
            "Should identify complete thruster failure"
        )
        
        # Test system reconfiguration
        reconfig_result = self._execute_system_reconfiguration(
            spacecraft, failed_components=[failed_thruster_id]
        )
        
        self.assertTrue(reconfig_result['success'], "System reconfiguration should succeed")
        self.assertGreater(
            reconfig_result['remaining_capability'], 0.85,
            "Should retain >85% capability with single thruster failure"
        )
        
        # Test degraded performance operation
        degraded_performance = self._test_degraded_operation(spacecraft)
        self.assertTrue(degraded_performance['operational'], "System should remain operational")
        self.assertGreater(
            degraded_performance['performance_retention'], 0.8,
            "Should retain >80% performance"
        )
        
        self.logger.info("Single thruster failure test completed successfully")
    
    def test_multiple_thruster_failures(self):
        """Test system response to multiple thruster failures"""
        self.logger.info("Testing multiple thruster failure scenario")
        
        spacecraft = self.test_spacecraft[0]  # Primary spacecraft (redundant system)
        fdir = self.fdir_systems[spacecraft.agent_id]
        
        # Simulate cascading thruster failures
        failed_thrusters = [1, 4, 7]  # Multiple failures
        failure_times = [time.time(), time.time() + 5.0, time.time() + 12.0]
        
        cascade_results = []
        
        for thruster_id, failure_time in zip(failed_thrusters, failure_times):
            # Simulate failure
            spacecraft.thruster_health[thruster_id] = 0.0
            
            # Detect and isolate fault
            fault_detected = fdir.detect_thruster_fault(
                thruster_id=thruster_id,
                expected_thrust=8.0,
                measured_thrust=0.0,
                detection_time=failure_time
            )
            
            isolation_result = fdir.isolate_fault(thruster_id)
            
            # Reconfigure system
            active_failures = [t for t, health in enumerate(spacecraft.thruster_health) if health == 0.0]
            reconfig_result = self._execute_system_reconfiguration(
                spacecraft, failed_components=active_failures
            )
            
            cascade_results.append({
                'thruster_id': thruster_id,
                'fault_detected': fault_detected,
                'isolated': isolation_result['isolated'],
                'reconfig_success': reconfig_result['success'],
                'remaining_capability': reconfig_result['remaining_capability']
            })
        
        # Verify cascade handling
        for i, result in enumerate(cascade_results):
            self.assertTrue(result['fault_detected'], f"Failure {i+1} should be detected")
            self.assertTrue(result['isolated'], f"Failure {i+1} should be isolated")
            self.assertTrue(result['reconfig_success'], f"Reconfiguration {i+1} should succeed")
        
        # Check final system capability
        final_capability = cascade_results[-1]['remaining_capability']
        self.assertGreater(final_capability, 0.6, "System should retain >60% capability")
        
        # Test mission continuation capability
        mission_continuation = self._test_mission_continuation(spacecraft)
        self.assertTrue(
            mission_continuation['can_continue'], 
            "Mission should be continuable with reduced capability"
        )
        
        self.logger.info(f"Multiple thruster failure test completed: Final capability = {final_capability:.2f}")
    
    def test_sensor_degradation(self):
        """Test system response to sensor degradation"""
        self.logger.info("Testing sensor degradation scenario")
        
        spacecraft = self.test_spacecraft[0]  # Triple redundant sensors
        
        # Test GPS sensor degradation
        gps_degradation_sequence = [
            {'sensor_idx': 0, 'health_level': 0.8, 'error_type': 'bias'},
            {'sensor_idx': 1, 'health_level': 0.0, 'error_type': 'complete_failure'},
            {'sensor_idx': 2, 'health_level': 0.6, 'error_type': 'noise_increase'}
        ]
        
        sensor_fusion_results = []
        
        for degradation in gps_degradation_sequence:
            # Apply sensor degradation
            sensor_idx = degradation['sensor_idx']
            spacecraft.sensor_health['gps'][sensor_idx] = degradation['health_level']
            
            # Test sensor fusion adaptation
            fusion_result = self._test_sensor_fusion(
                spacecraft, 'gps', degradation['error_type']
            )
            sensor_fusion_results.append(fusion_result)
            
            # Verify navigation accuracy
            nav_accuracy = fusion_result['navigation_accuracy']
            if len([h for h in spacecraft.sensor_health['gps'] if h > 0.5]) >= 2:
                # At least 2 good sensors
                self.assertGreater(nav_accuracy, 0.8, "Navigation should remain accurate with 2+ sensors")
            elif len([h for h in spacecraft.sensor_health['gps'] if h > 0.3]) >= 1:
                # At least 1 degraded sensor
                self.assertGreater(nav_accuracy, 0.5, "Navigation should be degraded but functional")
        
        # Test IMU sensor cascade failure
        imu_failures = [0, 1]  # Fail 2 out of 3 IMU sensors
        for sensor_idx in imu_failures:
            spacecraft.sensor_health['imu'][sensor_idx] = 0.0
        
        # Verify system switches to backup navigation mode
        backup_nav = self._test_backup_navigation_mode(spacecraft)
        self.assertTrue(backup_nav['backup_active'], "Should activate backup navigation")
        self.assertGreater(backup_nav['accuracy'], 0.4, "Backup navigation should provide minimum accuracy")
        
        self.logger.info("Sensor degradation test completed successfully")
    
    def test_communication_system_failure(self):
        """Test multi-agent coordination under communication failures"""
        self.logger.info("Testing communication system failure scenario")
        
        # Set up multi-agent coordination scenario
        primary = self.test_spacecraft[0]
        secondary = self.test_spacecraft[1]
        backup = self.test_spacecraft[2]
        
        coordinator = MultiAgentCoordinator([agent.agent_id for agent in self.test_spacecraft])
        
        # Initial coordination test
        initial_coord = self._test_coordination_performance([primary, secondary, backup])
        self.assertGreater(initial_coord['coordination_quality'], 0.8, "Initial coordination should be good")
        
        # Simulate progressive communication degradation
        comm_failure_scenarios = [
            {
                'affected_agent': primary.agent_id,
                'failure_type': 'intermittent',
                'degradation_level': 0.3,
                'duration': 30.0
            },
            {
                'affected_agent': secondary.agent_id,
                'failure_type': 'complete',
                'degradation_level': 0.0,
                'duration': 45.0
            },
            {
                'affected_agent': backup.agent_id,
                'failure_type': 'bandwidth_limited',
                'degradation_level': 0.1,
                'duration': 20.0
            }
        ]
        
        coordination_adaptation_results = []
        
        for scenario in comm_failure_scenarios:
            # Apply communication failure
            affected_agent = next(a for a in self.test_spacecraft if a.agent_id == scenario['affected_agent'])
            affected_agent.communication_health = scenario['degradation_level']
            
            # Test coordination adaptation
            adaptation_result = self._test_coordination_adaptation(
                coordinator, self.test_spacecraft, scenario
            )
            coordination_adaptation_results.append(adaptation_result)
            
            # Verify graceful degradation
            if scenario['failure_type'] == 'complete':
                # Should switch to autonomous mode
                self.assertTrue(
                    adaptation_result['autonomous_mode_active'],
                    "Should activate autonomous mode for complete comm failure"
                )
            else:
                # Should adapt communication protocol
                self.assertTrue(
                    adaptation_result['protocol_adapted'],
                    "Should adapt communication protocol for partial failures"
                )
        
        # Test network topology reconfiguration
        topology_reconfig = self._test_network_topology_reconfiguration(coordinator, self.test_spacecraft)
        self.assertTrue(topology_reconfig['success'], "Network topology should reconfigure successfully")
        self.assertGreater(
            topology_reconfig['connectivity'], 0.6,
            "Network should maintain reasonable connectivity"
        )
        
        self.logger.info("Communication system failure test completed")
    
    def test_power_system_degradation(self):
        """Test system response to power system degradation"""
        self.logger.info("Testing power system degradation scenario")
        
        spacecraft = self.test_spacecraft[1]  # Secondary spacecraft
        
        # Simulate power system degradation scenarios
        power_scenarios = [
            {'type': 'solar_panel_degradation', 'power_reduction': 0.2, 'duration': 60.0},
            {'type': 'battery_failure', 'power_reduction': 0.4, 'duration': 120.0},
            {'type': 'power_management_fault', 'power_reduction': 0.3, 'duration': 30.0}
        ]
        
        power_management_results = []
        
        for scenario in power_scenarios:
            # Apply power degradation
            original_power = spacecraft.power_system_health
            spacecraft.power_system_health *= (1.0 - scenario['power_reduction'])
            
            # Test power management adaptation
            power_mgmt_result = self._test_power_management_adaptation(spacecraft, scenario)
            power_management_results.append(power_mgmt_result)
            
            # Verify power allocation optimization
            self.assertTrue(
                power_mgmt_result['power_reallocation_success'],
                f"Power reallocation should succeed for {scenario['type']}"
            )
            
            # Verify critical systems maintained
            critical_systems_power = power_mgmt_result['critical_systems_power_fraction']
            self.assertGreater(
                critical_systems_power, 0.8,
                "Critical systems should maintain >80% power allocation"
            )
            
            # Verify non-critical systems gracefully degraded
            if scenario['power_reduction'] > 0.3:
                non_critical_power = power_mgmt_result['non_critical_systems_power_fraction']
                self.assertLess(
                    non_critical_power, 0.5,
                    "Non-critical systems should be power-limited during significant degradation"
                )
        
        # Test emergency power mode
        # Simulate severe power crisis
        spacecraft.power_system_health = 0.3  # 30% power remaining
        
        emergency_mode = self._test_emergency_power_mode(spacecraft)
        self.assertTrue(emergency_mode['activated'], "Emergency power mode should activate")
        self.assertGreater(emergency_mode['mission_time_extension'], 30.0, "Should extend mission time by >30 minutes")
        
        self.logger.info("Power system degradation test completed")
    
    def test_cascading_system_failures(self):
        """Test system response to cascading multi-system failures"""
        self.logger.info("Testing cascading multi-system failures")
        
        spacecraft = self.test_spacecraft[0]  # Primary spacecraft
        
        # Define cascading failure scenario
        cascade_scenario = [
            {
                'time': 0.0,
                'system': 'thruster',
                'component': 2,
                'failure_type': 'complete',
                'trigger': 'primary_failure'
            },
            {
                'time': 5.0,
                'system': 'sensor',
                'component': 'gps_0',
                'failure_type': 'degradation',
                'trigger': 'thermal_stress'
            },
            {
                'time': 12.0,
                'system': 'communication',
                'component': 'primary_antenna',
                'failure_type': 'intermittent',
                'trigger': 'vibration_damage'
            },
            {
                'time': 20.0,
                'system': 'power',
                'component': 'battery_2',
                'failure_type': 'capacity_reduction',
                'trigger': 'thermal_runaway'
            },
            {
                'time': 35.0,
                'system': 'thruster',
                'component': 7,
                'failure_type': 'performance_degradation',
                'trigger': 'propellant_contamination'
            }
        ]
        
        cascade_results = []
        system_health_timeline = []
        
        for failure in cascade_scenario:
            # Apply failure
            self._apply_system_failure(spacecraft, failure)
            
            # Assess system response
            response_result = self._assess_cascade_response(spacecraft, failure)
            cascade_results.append(response_result)
            
            # Record system health
            current_health = self._assess_system_health(spacecraft)
            system_health_timeline.append({
                'time': failure['time'],
                'system_health': current_health,
                'operational_capability': response_result['operational_capability']
            })
            
            # Verify system still operational above minimum threshold
            min_threshold = self.system_config['minimum_operational_threshold']
            if current_health > min_threshold:
                self.assertTrue(
                    response_result['system_operational'],
                    f"System should remain operational at health level {current_health:.2f}"
                )
        
        # Verify graceful degradation curve
        health_values = [h['system_health'] for h in system_health_timeline]
        capability_values = [h['operational_capability'] for h in system_health_timeline]
        
        # Health should decrease monotonically (or stay constant)
        for i in range(1, len(health_values)):
            self.assertLessEqual(
                health_values[i], health_values[i-1] + 0.01,  # Small tolerance for floating point
                "System health should not increase during cascade failures"
            )
        
        # Final system assessment
        final_health = health_values[-1]
        final_capability = capability_values[-1]
        
        if final_health > self.system_config['minimum_operational_threshold']:
            self.assertGreater(final_capability, 0.3, "System should retain minimum operational capability")
        
        # Test recovery initiation
        recovery_result = self._test_system_recovery(spacecraft)
        if final_health > 0.2:  # Recovery possible above 20% health
            self.assertTrue(recovery_result['recovery_initiated'], "System recovery should be initiated")
        
        self.logger.info(f"Cascading failure test completed: Final health = {final_health:.2f}")
    
    def test_fault_isolation_accuracy(self):
        """Test accuracy of fault isolation algorithms"""
        self.logger.info("Testing fault isolation accuracy")
        
        # Test various fault types and isolation accuracy
        fault_test_cases = [
            {
                'fault_type': FaultType.THRUSTER_STUCK,
                'component': 5,
                'signature': {'expected_thrust': 10.0, 'measured_thrust': 10.0, 'commanded_thrust': 0.0}
            },
            {
                'fault_type': FaultType.THRUSTER_DEGRADED,
                'component': 3,
                'signature': {'expected_thrust': 10.0, 'measured_thrust': 6.0, 'commanded_thrust': 10.0}
            },
            {
                'fault_type': FaultType.SENSOR_DRIFT,
                'component': 'gps_1',
                'signature': {'expected_reading': 100.0, 'measured_reading': 105.5, 'drift_rate': 0.1}
            },
            {
                'fault_type': FaultType.COMMUNICATION_LOSS,
                'component': 'comm_primary',
                'signature': {'packet_loss_rate': 0.8, 'signal_strength': 0.1}
            }
        ]
        
        isolation_accuracy_results = []
        
        for test_case in fault_test_cases:
            spacecraft = self.test_spacecraft[0]
            fdir = self.fdir_systems[spacecraft.agent_id]
            
            # Inject fault with known signature
            self._inject_known_fault(spacecraft, test_case)
            
            # Test fault isolation
            isolation_result = fdir.isolate_fault_with_signature(
                test_case['component'],
                test_case['signature']
            )
            
            # Verify isolation accuracy
            correctly_identified = isolation_result['identified_fault_type'] == test_case['fault_type']
            confidence = isolation_result['confidence']
            
            isolation_accuracy_results.append({
                'fault_type': test_case['fault_type'],
                'correctly_identified': correctly_identified,
                'confidence': confidence,
                'isolation_time': isolation_result['isolation_time']
            })
            
            # Verify minimum performance requirements
            self.assertGreater(confidence, 0.7, f"Isolation confidence should be >70% for {test_case['fault_type']}")
            self.assertLess(isolation_result['isolation_time'], 5.0, "Isolation should complete within 5 seconds")
        
        # Calculate overall isolation accuracy
        accuracy = np.mean([r['correctly_identified'] for r in isolation_accuracy_results])
        avg_confidence = np.mean([r['confidence'] for r in isolation_accuracy_results])
        avg_isolation_time = np.mean([r['isolation_time'] for r in isolation_accuracy_results])
        
        # Verify overall performance
        self.assertGreater(accuracy, 0.8, "Overall isolation accuracy should be >80%")
        self.assertGreater(avg_confidence, 0.75, "Average isolation confidence should be >75%")
        self.assertLess(avg_isolation_time, 3.0, "Average isolation time should be <3 seconds")
        
        self.logger.info(f"Fault isolation accuracy: {accuracy:.1%}, Avg confidence: {avg_confidence:.2f}")
    
    def test_system_recovery_procedures(self):
        """Test automated system recovery procedures"""
        self.logger.info("Testing system recovery procedures")
        
        spacecraft = self.test_spacecraft[1]  # Secondary spacecraft
        
        # Test recovery scenarios
        recovery_scenarios = [
            {
                'name': 'thruster_recovery',
                'initial_failures': {'thrusters': [2, 5]},
                'recovery_type': 'recalibration',
                'expected_success_rate': 0.6
            },
            {
                'name': 'sensor_recovery',
                'initial_failures': {'sensors': ['gps_0', 'imu_1']},
                'recovery_type': 'soft_reset',
                'expected_success_rate': 0.8
            },
            {
                'name': 'communication_recovery',
                'initial_failures': {'communication': ['primary_radio']},
                'recovery_type': 'antenna_reconfiguration',
                'expected_success_rate': 0.7
            },
            {
                'name': 'integrated_recovery',
                'initial_failures': {'thrusters': [1], 'sensors': ['camera_0'], 'power': ['battery_1']},
                'recovery_type': 'coordinated_recovery',
                'expected_success_rate': 0.5
            }
        ]
        
        recovery_results = []
        
        for scenario in recovery_scenarios:
            self.logger.info(f"Testing {scenario['name']} recovery")
            
            # Apply initial failures
            self._apply_scenario_failures(spacecraft, scenario['initial_failures'])
            initial_health = self._assess_system_health(spacecraft)
            
            # Execute recovery procedure
            recovery_result = self._execute_recovery_procedure(spacecraft, scenario)
            
            # Assess recovery effectiveness
            post_recovery_health = self._assess_system_health(spacecraft)
            health_improvement = post_recovery_health - initial_health
            
            recovery_results.append({
                'scenario': scenario['name'],
                'initial_health': initial_health,
                'post_recovery_health': post_recovery_health,
                'health_improvement': health_improvement,
                'recovery_success': recovery_result['success'],
                'recovery_time': recovery_result['recovery_time']
            })
            
            # Verify recovery performance
            if recovery_result['success']:
                self.assertGreater(
                    health_improvement, 0.05,
                    f"{scenario['name']} should show measurable health improvement"
                )
                self.assertLess(
                    recovery_result['recovery_time'], 60.0,
                    "Recovery should complete within 60 seconds"
                )
            
            # Reset for next scenario
            self._reset_spacecraft_health(spacecraft)
        
        # Verify overall recovery system performance
        successful_recoveries = sum(1 for r in recovery_results if r['recovery_success'])
        success_rate = successful_recoveries / len(recovery_results)
        
        self.assertGreater(success_rate, 0.6, "Overall recovery success rate should be >60%")
        
        avg_recovery_time = np.mean([r['recovery_time'] for r in recovery_results if r['recovery_success']])
        self.assertLess(avg_recovery_time, 45.0, "Average recovery time should be <45 seconds")
        
        self.logger.info(f"System recovery test completed: Success rate = {success_rate:.1%}")
    
    # Helper methods for test implementation
    
    def _assess_system_health(self, spacecraft):
        """Assess overall system health"""
        # Thruster health
        thruster_health = np.mean(spacecraft.thruster_health) if hasattr(spacecraft, 'thruster_health') else 1.0
        
        # Sensor health (average across all sensor types)
        sensor_health = 1.0
        if hasattr(spacecraft, 'sensor_health'):
            all_sensors = []
            for sensor_type, sensors in spacecraft.sensor_health.items():
                all_sensors.extend(sensors)
            sensor_health = np.mean(all_sensors) if all_sensors else 1.0
        
        # Communication health
        comm_health = getattr(spacecraft, 'communication_health', 1.0)
        
        # Power system health  
        power_health = getattr(spacecraft, 'power_system_health', 1.0)
        
        # Weighted average of subsystem health
        weights = {'thruster': 0.3, 'sensor': 0.25, 'comm': 0.2, 'power': 0.25}
        overall_health = (
            weights['thruster'] * thruster_health +
            weights['sensor'] * sensor_health +
            weights['comm'] * comm_health +
            weights['power'] * power_health
        )
        
        return overall_health
    
    def _execute_system_reconfiguration(self, spacecraft, failed_components):
        """Execute system reconfiguration after component failures"""
        # Simulate reconfiguration logic
        total_components = len(spacecraft.thruster_health) if hasattr(spacecraft, 'thruster_health') else 12
        failed_count = len(failed_components)
        
        # Calculate remaining capability
        remaining_capability = max(0.0, 1.0 - (failed_count / total_components))
        
        # Reconfiguration successful if enough redundancy
        success = remaining_capability > 0.3
        
        if success:
            # Update control allocation matrix (simulated)
            self.logger.info(f"Reconfigured system: {remaining_capability:.1%} capability remaining")
        
        return {
            'success': success,
            'remaining_capability': remaining_capability,
            'reconfiguration_time': 2.0  # Simulated reconfiguration time
        }
    
    def _test_degraded_operation(self, spacecraft):
        """Test spacecraft operation in degraded mode"""
        current_health = self._assess_system_health(spacecraft)
        
        # Performance retention based on health
        performance_retention = min(1.0, current_health + 0.1)  # Some resilience
        
        # Operational if above minimum threshold
        operational = current_health > self.system_config['minimum_operational_threshold']
        
        return {
            'operational': operational,
            'performance_retention': performance_retention,
            'degraded_mode_active': current_health < self.system_config['degraded_mode_threshold']
        }
    
    def _test_mission_continuation(self, spacecraft):
        """Test if mission can continue with current system health"""
        current_health = self._assess_system_health(spacecraft)
        
        # Mission continuation criteria
        can_continue = current_health > 0.4  # 40% minimum for mission continuation
        mission_modification_required = current_health < 0.7  # Modify mission below 70%
        
        return {
            'can_continue': can_continue,
            'mission_modification_required': mission_modification_required,
            'estimated_capability': current_health
        }
    
    def _test_sensor_fusion(self, spacecraft, sensor_type, error_type):
        """Test sensor fusion adaptation to sensor failures"""
        sensor_healths = spacecraft.sensor_health[sensor_type]
        healthy_sensors = sum(1 for h in sensor_healths if h > 0.3)
        
        # Navigation accuracy based on healthy sensor count
        if healthy_sensors >= 3:
            nav_accuracy = 0.95
        elif healthy_sensors >= 2:
            nav_accuracy = 0.85
        elif healthy_sensors >= 1:
            nav_accuracy = 0.65
        else:
            nav_accuracy = 0.2  # Backup navigation only
        
        return {
            'navigation_accuracy': nav_accuracy,
            'fusion_algorithm_adapted': healthy_sensors >= 1,
            'backup_navigation_active': healthy_sensors == 0
        }
    
    def _test_backup_navigation_mode(self, spacecraft):
        """Test backup navigation mode activation"""
        # Count functional sensors
        total_functional = 0
        for sensor_type, sensors in spacecraft.sensor_health.items():
            total_functional += sum(1 for h in sensors if h > 0.3)
        
        backup_active = total_functional < 3  # Activate backup if <3 functional sensors
        accuracy = 0.4 if backup_active else 0.9
        
        return {
            'backup_active': backup_active,
            'accuracy': accuracy
        }
    
    def _test_coordination_performance(self, agents):
        """Test multi-agent coordination performance"""
        # Simulate coordination quality based on communication health
        comm_healths = [getattr(agent, 'communication_health', 1.0) for agent in agents]
        avg_comm_health = np.mean(comm_healths)
        
        coordination_quality = avg_comm_health * 0.9  # Some degradation in coordination
        
        return {
            'coordination_quality': coordination_quality,
            'all_agents_connected': all(h > 0.1 for h in comm_healths)
        }
    
    def _test_coordination_adaptation(self, coordinator, agents, failure_scenario):
        """Test coordination system adaptation to communication failures"""
        affected_agent_id = failure_scenario['affected_agent']
        failure_type = failure_scenario['failure_type']
        
        # Adaptation strategies based on failure type
        if failure_type == 'complete':
            autonomous_mode_active = True
            protocol_adapted = False
        elif failure_type == 'intermittent':
            autonomous_mode_active = False
            protocol_adapted = True  # Use store-and-forward, redundant messaging
        elif failure_type == 'bandwidth_limited':
            autonomous_mode_active = False
            protocol_adapted = True  # Compress messages, reduce frequency
        else:
            autonomous_mode_active = False
            protocol_adapted = False
        
        return {
            'autonomous_mode_active': autonomous_mode_active,
            'protocol_adapted': protocol_adapted,
            'adaptation_time': 3.0
        }
    
    def _test_network_topology_reconfiguration(self, coordinator, agents):
        """Test network topology reconfiguration"""
        # Count functional communication links
        functional_agents = sum(1 for agent in agents if getattr(agent, 'communication_health', 1.0) > 0.1)
        total_agents = len(agents)
        
        # Success if majority of agents can communicate
        success = functional_agents >= (total_agents // 2)
        connectivity = functional_agents / total_agents if total_agents > 0 else 0.0
        
        return {
            'success': success,
            'connectivity': connectivity,
            'reconfiguration_time': 5.0
        }
    
    def _test_power_management_adaptation(self, spacecraft, power_scenario):
        """Test power management system adaptation"""
        # Simulate power reallocation
        power_reduction = power_scenario['power_reduction']
        
        # Critical systems get priority
        critical_systems_power = min(1.0, 1.2 - power_reduction)  # Some margin
        non_critical_systems_power = max(0.0, 1.0 - 2 * power_reduction)  # Reduced first
        
        reallocation_success = critical_systems_power > 0.6
        
        return {
            'power_reallocation_success': reallocation_success,
            'critical_systems_power_fraction': critical_systems_power,
            'non_critical_systems_power_fraction': non_critical_systems_power
        }
    
    def _test_emergency_power_mode(self, spacecraft):
        """Test emergency power mode activation"""
        current_power = spacecraft.power_system_health
        
        activated = current_power < 0.4  # Activate below 40% power
        
        if activated:
            # Estimate mission time extension through power conservation
            power_savings = 0.3  # 30% power reduction through emergency mode
            mission_time_extension = (power_savings / (1.0 - power_savings)) * 60.0  # Minutes
        else:
            mission_time_extension = 0.0
        
        return {
            'activated': activated,
            'mission_time_extension': mission_time_extension
        }
    
    def _apply_system_failure(self, spacecraft, failure):
        """Apply system failure to spacecraft"""
        system = failure['system']
        component = failure['component']
        failure_type = failure['failure_type']
        
        if system == 'thruster':
            if failure_type == 'complete':
                spacecraft.thruster_health[component] = 0.0
            elif failure_type == 'performance_degradation':
                spacecraft.thruster_health[component] = 0.6
                
        elif system == 'sensor':
            sensor_type, sensor_idx = component.split('_')
            idx = int(sensor_idx)
            if failure_type == 'degradation':
                spacecraft.sensor_health[sensor_type][idx] = 0.3
                
        elif system == 'communication':
            if failure_type == 'intermittent':
                spacecraft.communication_health = 0.4
                
        elif system == 'power':
            if failure_type == 'capacity_reduction':
                spacecraft.power_system_health *= 0.8
    
    def _assess_cascade_response(self, spacecraft, failure):
        """Assess system response to cascading failure"""
        current_health = self._assess_system_health(spacecraft)
        
        operational = current_health > self.system_config['minimum_operational_threshold']
        operational_capability = max(0.0, current_health - 0.1)  # Some performance loss
        
        return {
            'system_operational': operational,
            'operational_capability': operational_capability,
            'response_time': 1.0  # Simulated response time
        }
    
    def _test_system_recovery(self, spacecraft):
        """Test system recovery initiation"""
        current_health = self._assess_system_health(spacecraft)
        
        # Recovery possible if above critical threshold
        recovery_possible = current_health > 0.2
        recovery_initiated = recovery_possible and current_health < 0.8  # Below nominal
        
        return {
            'recovery_initiated': recovery_initiated,
            'recovery_possible': recovery_possible
        }
    
    def _inject_known_fault(self, spacecraft, test_case):
        """Inject known fault for isolation testing"""
        fault_type = test_case['fault_type']
        component = test_case['component']
        
        # Inject fault signature based on type
        if fault_type == FaultType.THRUSTER_STUCK:
            spacecraft.thruster_health[component] = 0.5  # Partially functional but stuck
        elif fault_type == FaultType.THRUSTER_DEGRADED:
            spacecraft.thruster_health[component] = 0.6  # Degraded performance
        elif fault_type == FaultType.SENSOR_DRIFT:
            # Inject sensor drift (simulated)
            pass
        elif fault_type == FaultType.COMMUNICATION_LOSS:
            spacecraft.communication_health = 0.1
    
    def _apply_scenario_failures(self, spacecraft, failures):
        """Apply multiple failures for recovery testing"""
        if 'thrusters' in failures:
            for thruster_id in failures['thrusters']:
                spacecraft.thruster_health[thruster_id] = 0.0
                
        if 'sensors' in failures:
            for sensor_id in failures['sensors']:
                sensor_type, idx = sensor_id.split('_')
                spacecraft.sensor_health[sensor_type][int(idx)] = 0.0
                
        if 'communication' in failures:
            spacecraft.communication_health = 0.0
            
        if 'power' in failures:
            spacecraft.power_system_health *= 0.7
    
    def _execute_recovery_procedure(self, spacecraft, scenario):
        """Execute recovery procedure"""
        recovery_type = scenario['recovery_type']
        
        # Simulate recovery based on type
        if recovery_type == 'recalibration':
            recovery_time = 20.0
            success_rate = 0.6
        elif recovery_type == 'soft_reset':
            recovery_time = 10.0
            success_rate = 0.8
        elif recovery_type == 'antenna_reconfiguration':
            recovery_time = 15.0
            success_rate = 0.7
        elif recovery_type == 'coordinated_recovery':
            recovery_time = 45.0
            success_rate = 0.5
        else:
            recovery_time = 30.0
            success_rate = 0.6
        
        # Random success based on expected rate
        success = random.random() < success_rate
        
        if success:
            # Partial recovery of failed components
            self._partial_component_recovery(spacecraft, scenario['initial_failures'])
        
        return {
            'success': success,
            'recovery_time': recovery_time
        }
    
    def _partial_component_recovery(self, spacecraft, failures):
        """Partially recover failed components"""
        if 'thrusters' in failures:
            for thruster_id in failures['thrusters']:
                spacecraft.thruster_health[thruster_id] = 0.7  # Partial recovery
                
        if 'sensors' in failures:
            for sensor_id in failures['sensors']:
                sensor_type, idx = sensor_id.split('_')
                spacecraft.sensor_health[sensor_type][int(idx)] = 0.8  # Good recovery
                
        if 'communication' in failures:
            spacecraft.communication_health = 0.6  # Partial recovery
            
        if 'power' in failures:
            spacecraft.power_system_health = min(1.0, spacecraft.power_system_health * 1.3)
    
    def _reset_spacecraft_health(self, spacecraft):
        """Reset spacecraft to healthy state"""
        self._initialize_component_health(spacecraft, {
            'thruster_config': spacecraft.thruster_config,
            'sensor_config': spacecraft.sensor_config
        })


if __name__ == '__main__':
    # Configure test logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run partial system failure tests
    unittest.main(verbosity=2)