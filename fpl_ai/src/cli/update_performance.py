"""
CLI command to update model performance tracking with actual results.

This script compares predicted vs actual points for previous gameweeks
and updates performance tracking metrics.
"""

import argparse
import sys
from pathlib import Path

# Add the fpl_ai directory to Python path
fpl_ai_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(fpl_ai_dir))

from src.common.performance_tracker import ModelPerformanceTracker
from src.common.config import get_logger
from src.common.timeutil import get_current_gw

logger = get_logger(__name__)


def update_performance_for_gameweek(gameweek: int) -> None:
    """Update performance tracking for a completed gameweek."""
    logger.info(f"Updating performance tracking for GW{gameweek}")
    
    tracker = ModelPerformanceTracker()
    tracker.update_actual_results(gameweek)
    
    logger.info(f"Performance tracking updated for GW{gameweek}")


def update_all_completed_gameweeks() -> None:
    """Update performance tracking for all completed gameweeks."""
    current_gw = get_current_gw()
    tracker = ModelPerformanceTracker()
    
    logger.info(f"Current gameweek: {current_gw}")
    logger.info("Updating performance tracking for all completed gameweeks")
    
    # Update for gameweeks 1 through current-1 (completed gameweeks)
    updated_count = 0
    for gw in range(1, current_gw):
        try:
            logger.info(f"Processing GW{gw}")
            tracker.update_actual_results(gw)
            updated_count += 1
        except Exception as e:
            logger.error(f"Failed to update GW{gw}: {e}")
    
    logger.info(f"Updated performance tracking for {updated_count} gameweeks")


def show_performance_report() -> None:
    """Display current performance tracking report."""
    tracker = ModelPerformanceTracker()
    report = tracker.generate_performance_report()
    print(report)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Update FPL model performance tracking with actual results"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Update specific gameweek
    update_parser = subparsers.add_parser(
        'update', 
        help='Update performance tracking for a specific gameweek'
    )
    update_parser.add_argument(
        'gameweek', 
        type=int, 
        help='Gameweek number to update'
    )
    
    # Update all completed gameweeks
    update_all_parser = subparsers.add_parser(
        'update-all',
        help='Update performance tracking for all completed gameweeks'
    )
    
    # Show performance report
    report_parser = subparsers.add_parser(
        'report',
        help='Show current performance tracking report'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'update':
            update_performance_for_gameweek(args.gameweek)
        elif args.command == 'update-all':
            update_all_completed_gameweeks()
        elif args.command == 'report':
            show_performance_report()
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
