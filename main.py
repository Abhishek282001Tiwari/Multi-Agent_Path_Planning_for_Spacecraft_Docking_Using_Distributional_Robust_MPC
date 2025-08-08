# main.py
#!/usr/bin/env python3
"""
Multi-Agent DR-MPC Spacecraft Docking System
Main execution entry point for the professional aerospace-grade system
"""
import argparse
import logging
import sys
from pathlib import Path

# Add the src directory to Python path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from spacecraft_drmpc.simulations.docking_simulator import DockingSimulator
from spacecraft_drmpc.utils.simple_config import load_mission_config

def main():
    parser = argparse.ArgumentParser(description='DR-MPC Spacecraft Docking System')
    parser.add_argument('--scenario', type=str, default='three_spacecraft',
                       choices=['single', 'three_spacecraft', 'formation_flying'])
    parser.add_argument('--visualize', action='store_true', help='Enable 3D visualization')
    parser.add_argument('--realtime', action='store_true', help='Real-time mode')
    parser.add_argument('--duration', type=float, default=1800, help='Simulation duration (s)')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load mission configuration
    config = load_mission_config(args.scenario)
    
    # Initialize simulator
    simulator = DockingSimulator(config)
    
    # Initialize viewer if requested
    if args.visualize:
        logger.info("Visualization mode enabled (simplified)")
    
    # Run simulation
    logger.info(f"Starting {args.scenario} scenario...")
    results = simulator.run(duration=args.duration, realtime=args.realtime)
    
    # Save results
    results.save('simulation_results.h5')
    logger.info("Simulation completed successfully!")

if __name__ == "__main__":
    main()