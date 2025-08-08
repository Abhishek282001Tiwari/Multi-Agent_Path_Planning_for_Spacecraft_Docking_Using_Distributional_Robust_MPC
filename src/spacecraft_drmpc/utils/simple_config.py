#!/usr/bin/env python3
"""
Simple configuration loader for testing
"""

class SimpleConfig:
    """Simple configuration for testing purposes"""
    def __init__(self, scenario="single_docking"):
        self.scenario = scenario
        self.duration = 1800
        self.num_spacecraft = 3
        self.visualization_enabled = False

def load_mission_config(scenario):
    """Load mission configuration for scenario"""
    return SimpleConfig(scenario)