#!/usr/bin/env python3
"""
Command Line Interface for Spacecraft DRMPC System
Provides professional CLI access to all system capabilities
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .simulations.docking_simulator import DockingSimulator
from .utils.simple_config import load_mission_config
# from .visualization.live_viewer import LiveViewer
# from .monitoring.system_logger import SystemLogger


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup system-wide logging configuration"""
    log_level = getattr(logging, level.upper())
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the main CLI argument parser"""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Spacecraft Docking System with DR-MPC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  spacecraft-drmpc simulate --scenario single_docking --visualize
  spacecraft-drmpc simulate --scenario formation_flying --duration 3600
  spacecraft-drmpc analyze --results simulation_results.h5
  spacecraft-drmpc dashboard --config mission_config.yaml
        """
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version="spacecraft-drmpc 1.0.0"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Simulation command
    sim_parser = subparsers.add_parser("simulate", help="Run spacecraft simulation")
    sim_parser.add_argument(
        "--scenario",
        choices=["single_docking", "formation_flying", "multi_target", "emergency_abort"],
        default="single_docking",
        help="Simulation scenario to run"
    )
    sim_parser.add_argument(
        "--config",
        type=Path,
        help="Custom configuration file path"
    )
    sim_parser.add_argument(
        "--duration",
        type=float,
        default=1800.0,
        help="Simulation duration in seconds (default: 1800)"
    )
    sim_parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable 3D visualization"
    )
    sim_parser.add_argument(
        "--realtime",
        action="store_true", 
        help="Run in real-time mode"
    )
    sim_parser.add_argument(
        "--output",
        type=Path,
        default="simulation_results.h5",
        help="Output file for results (default: simulation_results.h5)"
    )
    
    # Analysis command
    analysis_parser = subparsers.add_parser("analyze", help="Analyze simulation results")
    analysis_parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Results file to analyze"
    )
    analysis_parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate analysis report"
    )
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Launch mission control dashboard")
    dashboard_parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Dashboard port (default: 8080)"
    )
    dashboard_parser.add_argument(
        "--host",
        default="localhost",
        help="Dashboard host (default: localhost)"
    )
    
    # Configuration validation
    config_parser = subparsers.add_parser("validate", help="Validate configuration files")
    config_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Configuration file to validate"
    )
    
    return parser


def handle_simulate(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Handle simulation command"""
    try:
        logger.info(f"Initializing {args.scenario} simulation scenario")
        
        # Load configuration
        config = load_mission_config(args.scenario)
        
        # Initialize simulator
        simulator = DockingSimulator(config)
        
        # Setup visualization if requested
        if args.visualize:
            logger.info("Visualization requested (simplified for testing)")
            # viewer = LiveViewer()
            # simulator.attach_viewer(viewer)
        
        # Run simulation
        logger.info(f"Starting simulation (duration: {args.duration}s)")
        results = simulator.run(
            duration=args.duration,
            realtime=args.realtime
        )
        
        # Save results
        logger.info(f"Saving results to {args.output}")
        results.save(args.output)
        
        logger.info("Simulation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        return 1


def handle_dashboard(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Handle dashboard command"""
    try:
        from .visualization.dashboard.mission_control import MissionControlDashboard
        
        logger.info(f"Starting mission control dashboard on {args.host}:{args.port}")
        dashboard = MissionControlDashboard(host=args.host, port=args.port)
        dashboard.start()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Dashboard shutdown requested")
        return 0
    except Exception as e:
        logger.error(f"Dashboard failed: {e}")
        return 1


def handle_analyze(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Handle analysis command"""
    try:
        from .utils.analysis_tools import ResultsAnalyzer
        
        logger.info(f"Analyzing results from {args.results}")
        analyzer = ResultsAnalyzer(args.results)
        
        # Run analysis
        analysis = analyzer.analyze()
        
        # Print summary
        analyzer.print_summary(analysis)
        
        # Generate report if requested
        if args.generate_report:
            report_path = args.results.with_suffix('.html')
            analyzer.generate_report(analysis, report_path)
            logger.info(f"Analysis report saved to {report_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


def handle_validate(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Handle configuration validation command"""
    try:
        from .utils.config_validator import ConfigValidator
        
        logger.info(f"Validating configuration file: {args.config}")
        validator = ConfigValidator()
        
        if validator.validate_file(args.config):
            logger.info("Configuration file is valid")
            return 0
        else:
            logger.error("Configuration file validation failed")
            return 1
            
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1


def main() -> int:
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Handle commands
    if args.command == "simulate":
        return handle_simulate(args, logger)
    elif args.command == "dashboard":
        return handle_dashboard(args, logger)
    elif args.command == "analyze":
        return handle_analyze(args, logger)
    elif args.command == "validate":
        return handle_validate(args, logger)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())