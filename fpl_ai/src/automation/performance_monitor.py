"""
Performance Tracking Monitor - Automated Updates

This script monitors for new gameweek data and automatically updates
performance tracking when new actual results become available.
"""

import time
import schedule
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import sys
import subprocess
import os

# Add the fpl_ai directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.performance_tracker import ModelPerformanceTracker
from src.common.config import get_logger, get_config
from src.common.timeutil import get_current_gw

logger = get_logger(__name__)


class PerformanceMonitor:
    """
    Monitors for new gameweek data and automatically updates performance tracking.
    """
    
    def __init__(self, vaastav_data_dir: str = None):
        if vaastav_data_dir is None:
            from src.common.timeutil import get_current_season
            vaastav_data_dir = f"data/vaastav/data/{get_current_season()}"
        self.vaastav_data_dir = Path(vaastav_data_dir)
        self.tracker = ModelPerformanceTracker()
        self.last_checked = datetime.now()
        self.processed_gameweeks: Set[int] = set()
        
        # Load existing processed gameweeks
        self._load_processed_gameweeks()
        
        logger.info(f"Performance monitor initialized")
        logger.info(f"Vaastav data directory: {self.vaastav_data_dir}")
        logger.info(f"Performance directory: {self.tracker.performance_dir}")
    
    def _load_processed_gameweeks(self) -> None:
        """Load list of already processed gameweeks."""
        try:
            summary = self.tracker.get_performance_summary()
            if summary:
                first_gw = summary.get('first_gameweek', 1)
                latest_gw = summary.get('latest_gameweek', 1)
                self.processed_gameweeks = set(range(first_gw, latest_gw + 1))
                logger.info(f"Loaded {len(self.processed_gameweeks)} processed gameweeks: {sorted(self.processed_gameweeks)}")
        except Exception as e:
            logger.warning(f"Could not load processed gameweeks: {e}")
            self.processed_gameweeks = set()
    
    def check_for_new_gameweek_data(self) -> List[int]:
        """Check for new gameweek data files."""
        new_gameweeks = []
        
        try:
            gws_dir = self.vaastav_data_dir / "gws"
            if not gws_dir.exists():
                logger.warning(f"Gameweeks directory not found: {gws_dir}")
                return new_gameweeks
            
            # Find all gameweek files
            gw_files = list(gws_dir.glob("gw*.csv"))
            
            for gw_file in gw_files:
                # Extract gameweek number from filename (e.g., gw5.csv -> 5)
                try:
                    gw_num = int(gw_file.stem[2:])  # Remove 'gw' prefix
                    
                    # Check if this gameweek has been processed
                    if gw_num not in self.processed_gameweeks:
                        # Check if we have predictions for this gameweek
                        prediction_file = self.tracker.performance_dir / f"predicted_team_gw{gw_num}.json"
                        
                        if prediction_file.exists():
                            # Check if the gameweek file is recent (modified in last 7 days)
                            file_age = datetime.now() - datetime.fromtimestamp(gw_file.stat().st_mtime)
                            
                            if file_age <= timedelta(days=7):
                                new_gameweeks.append(gw_num)
                                logger.info(f"Found new gameweek data: GW{gw_num}")
                            else:
                                logger.debug(f"GW{gw_num} data exists but is old, skipping")
                        else:
                            logger.debug(f"GW{gw_num} data found but no predictions available")
                    
                except (ValueError, IndexError) as e:
                    logger.warning(f"Could not parse gameweek from filename {gw_file}: {e}")
            
            new_gameweeks.sort()
            return new_gameweeks
            
        except Exception as e:
            logger.error(f"Error checking for new gameweek data: {e}")
            return []
    
    def update_gameweek_performance(self, gameweek: int) -> bool:
        """Update performance tracking for a specific gameweek."""
        try:
            logger.info(f"Updating performance tracking for GW{gameweek}")
            
            # Update performance tracking
            self.tracker.update_actual_results(gameweek, str(self.vaastav_data_dir))
            
            # Mark as processed
            self.processed_gameweeks.add(gameweek)
            
            logger.info(f"✅ Successfully updated performance tracking for GW{gameweek}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update performance tracking for GW{gameweek}: {e}")
            return False
    
    def run_performance_check(self) -> None:
        """Run a complete performance check and update cycle."""
        logger.info("🔍 Running performance check...")
        
        try:
            # Check for new gameweek data
            new_gameweeks = self.check_for_new_gameweek_data()
            
            if not new_gameweeks:
                logger.info("No new gameweek data found")
                return
            
            logger.info(f"Found {len(new_gameweeks)} new gameweeks to process: {new_gameweeks}")
            
            # Update each new gameweek
            updated_count = 0
            for gw in new_gameweeks:
                if self.update_gameweek_performance(gw):
                    updated_count += 1
                    
                    # Small delay between updates to avoid overwhelming the system
                    time.sleep(1)
            
            if updated_count > 0:
                logger.info(f"🎉 Updated performance tracking for {updated_count} gameweeks")
                
                # Generate and display performance report
                try:
                    report = self.tracker.generate_performance_report()
                    logger.info("📊 Performance Report:\n" + report)
                except Exception as e:
                    logger.warning(f"Could not generate performance report: {e}")
            
        except Exception as e:
            logger.error(f"Error during performance check: {e}")
    
    def start_monitoring(self, check_interval_hours: int = 6) -> None:
        """Start continuous monitoring for new gameweek data."""
        logger.info(f"🚀 Starting performance monitoring (checking every {check_interval_hours} hours)")
        
        # Schedule regular checks
        schedule.every(check_interval_hours).hours.do(self.run_performance_check)
        
        # Run initial check
        self.run_performance_check()
        
        # Start monitoring loop
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute for scheduled jobs
                
        except KeyboardInterrupt:
            logger.info("⏹️ Performance monitoring stopped by user")
        except Exception as e:
            logger.error(f"Performance monitoring error: {e}")
    
    def run_manual_update(self, gameweek: Optional[int] = None) -> None:
        """Run a manual performance update."""
        if gameweek:
            logger.info(f"Manual update for GW{gameweek}")
            self.update_gameweek_performance(gameweek)
        else:
            logger.info("Manual update for all new gameweeks")
            self.run_performance_check()


def setup_cron_job(script_path: str, log_file: str, check_interval_hours: int = 6) -> str:
    """Generate a cron job command for automated monitoring."""
    cron_command = f"0 */{check_interval_hours} * * * cd {Path(__file__).parent.parent.parent} && python -m src.automation.performance_monitor monitor >> {log_file} 2>&1"
    
    cron_setup = f"""
# Add this line to your crontab (run 'crontab -e' to edit):
{cron_command}

# Or run this command to add it automatically:
(crontab -l 2>/dev/null; echo "{cron_command}") | crontab -

# To view current cron jobs:
crontab -l

# To remove the job:
crontab -l | grep -v "performance_monitor" | crontab -
"""
    
    return cron_setup


def main():
    """Main entry point for the performance monitor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FPL Performance Tracking Monitor")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Monitor command (continuous)
    monitor_parser = subparsers.add_parser('monitor', help='Start continuous monitoring')
    monitor_parser.add_argument(
        '--interval', 
        type=int, 
        default=6, 
        help='Check interval in hours (default: 6)'
    )
    
    # Check command (one-time)
    check_parser = subparsers.add_parser('check', help='Run one-time check for new data')
    
    # Update command (manual)
    update_parser = subparsers.add_parser('update', help='Manual update for specific gameweek')
    update_parser.add_argument('gameweek', type=int, nargs='?', help='Gameweek to update (optional)')
    
    # Setup command (cron job)
    setup_parser = subparsers.add_parser('setup-cron', help='Generate cron job setup commands')
    setup_parser.add_argument(
        '--interval', 
        type=int, 
        default=6, 
        help='Check interval in hours (default: 6)'
    )
    setup_parser.add_argument(
        '--log-file', 
        type=str, 
        default='/tmp/fpl_performance_monitor.log', 
        help='Log file path'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create monitor instance
    monitor = PerformanceMonitor()
    
    try:
        if args.command == 'monitor':
            monitor.start_monitoring(args.interval)
        elif args.command == 'check':
            monitor.run_performance_check()
        elif args.command == 'update':
            monitor.run_manual_update(args.gameweek)
        elif args.command == 'setup-cron':
            cron_setup = setup_cron_job(
                str(Path(__file__)), 
                args.log_file, 
                args.interval
            )
            print(cron_setup)
    
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()



