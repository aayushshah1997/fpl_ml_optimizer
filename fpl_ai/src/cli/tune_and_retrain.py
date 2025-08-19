"""
CLI interface for hyperparameter tuning and model retraining.

Provides commands for:
- Running Optuna hyperparameter optimization
- Retraining models with optimized parameters
- Quick tuning sessions with reduced scope
- Loading and applying best found configurations
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any

from ..common.config import load_settings, get_logger
from ..common.logging_setup import setup_logging
from ..tuning.optuna_tune import (
    run_optuna_study,
    load_best_settings,
    get_study_leaderboard,
    quick_tune
)
from ..modeling.model_lgbm import LGBMTrainer
from ..features.builder import build_training_table
from ..common.timeutil import get_current_gw

logger = get_logger(__name__)


def run_hyperparameter_tuning(
    n_trials: Optional[int] = None,
    timeout_min: Optional[int] = None,
    quick: bool = False
) -> Dict[str, Any]:
    """
    Run hyperparameter tuning with Optuna.
    
    Args:
        n_trials: Number of trials to run (overrides config)
        timeout_min: Timeout in minutes (overrides config)
        quick: Whether to run quick tune with reduced scope
        
    Returns:
        Tuning results dictionary
    """
    logger.info("Starting hyperparameter tuning...")
    
    if quick:
        logger.info("Running quick tune mode")
        return quick_tune(
            n_trials=n_trials or 20,
            timeout_min=timeout_min or 30
        )
    
    # Load configuration and apply overrides
    if n_trials is not None or timeout_min is not None:
        cfg = load_settings()
        if n_trials is not None:
            cfg.setdefault("autotune", {})["n_trials"] = n_trials
        if timeout_min is not None:
            cfg.setdefault("autotune", {})["timeout_min"] = timeout_min
        
        # Temporarily save modified config
        temp_config_path = Path("fpl_ai/artifacts/tuning/temp_config.yaml")
        from ..common.config import save_settings_dict
        save_settings_dict(cfg, temp_config_path)
        
        logger.info(f"Running tuning with overrides: n_trials={n_trials}, timeout_min={timeout_min}")
    
    # Run full study
    return run_optuna_study()


def retrain_models(
    settings_override_path: Optional[str] = None,
    force: bool = False,
    current_gw: Optional[int] = None
) -> Dict[str, Any]:
    """
    Retrain models using best found parameters or current configuration.
    
    Args:
        settings_override_path: Path to settings file with optimized parameters
        force: Whether to force retraining even if models exist
        current_gw: Current gameweek for training context
        
    Returns:
        Training results dictionary
    """
    logger.info("Starting model retraining...")
    
    # Load configuration
    if settings_override_path:
        logger.info(f"Using optimized settings from {settings_override_path}")
        try:
            cfg = load_settings(settings_override_path)
        except Exception as e:
            logger.error(f"Failed to load settings from {settings_override_path}: {e}")
            logger.info("Falling back to default settings")
            cfg = load_settings()
    else:
        # Try to load best settings from tuning
        best_settings = load_best_settings()
        if best_settings != load_settings():  # Check if we got different settings
            logger.info("Using best settings from previous tuning")
            cfg = best_settings
        else:
            logger.info("Using current settings (no tuning results found)")
            cfg = load_settings()
    
    # Determine current gameweek
    if current_gw is None:
        try:
            current_gw = get_current_gw()
            logger.info(f"Auto-detected current gameweek: {current_gw}")
        except Exception:
            current_gw = 1
            logger.warning("Could not detect current gameweek, using GW 1")
    
    # Check if models already exist
    from ..common.config import get_config
    config = get_config()
    models_dir = config.models_dir
    
    existing_models = list(models_dir.glob("model_points_*.pkl"))
    
    if existing_models and not force:
        logger.info(f"Found {len(existing_models)} existing models")
        logger.info("Use --force to retrain anyway")
        return {
            "status": "skipped",
            "reason": "models_exist",
            "existing_models": [str(p) for p in existing_models]
        }
    
    # Build training data
    logger.info("Building training data...")
    training_data = build_training_table(next_gw=current_gw)
    
    if training_data.empty:
        logger.error("No training data available")
        return {
            "status": "error",
            "reason": "no_training_data"
        }
    
    logger.info(f"Training data built: {len(training_data):,} samples")
    
    # Train models
    trainer = LGBMTrainer()
    
    # Apply configuration overrides if using optimized settings
    if settings_override_path or best_settings != load_settings():
        # Update trainer configuration dynamically
        trainer.gbm_params = cfg.get("modeling", {}).get("gbm", trainer.gbm_params)
        
        # Handle per-position parameters if enabled
        per_position_config = cfg.get("modeling", {}).get("per_position", {})
        if per_position_config.get("enabled", False):
            logger.info("Using position-specific model parameters")
    
    training_results = trainer.train_models(
        training_data=training_data,
        mode="full",  # Use full mode for production retraining
        current_gw=current_gw
    )
    
    if not training_results:
        logger.error("Model training failed")
        return {
            "status": "error",
            "reason": "training_failed"
        }
    
    logger.info("Model retraining completed successfully")
    
    return {
        "status": "completed",
        "training_results": training_results,
        "training_samples": len(training_data),
        "models_trained": list(training_results.get("position_results", {}).keys()),
        "current_gw": current_gw,
        "settings_source": settings_override_path or "best_found" if best_settings != load_settings() else "default"
    }


def show_tuning_results(artifacts_dir: Optional[str] = None) -> None:
    """
    Display tuning results and leaderboard.
    
    Args:
        artifacts_dir: Optional path to tuning artifacts directory
    """
    if artifacts_dir is None:
        artifacts_dir = "fpl_ai/artifacts/tuning"
    
    artifacts_path = Path(artifacts_dir)
    
    if not artifacts_path.exists():
        print("No tuning results found.")
        return
    
    # Load study summary
    summary_path = artifacts_path / "study_summary.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        print("=" * 60)
        print("HYPERPARAMETER TUNING RESULTS")
        print("=" * 60)
        print(f"Study: {summary.get('study_name', 'N/A')}")
        print(f"Trials completed: {summary.get('n_complete', 0)}/{summary.get('n_trials', 0)}")
        print(f"Study time: {summary.get('study_time_seconds', 0):.1f} seconds")
        print(f"Objective: {summary.get('objective', 'N/A')}")
        print(f"Best trial: #{summary.get('best_trial', 'N/A')}")
        print(f"Best value: {summary.get('best_value', 'N/A'):.3f}")
        print()
    
    # Load and display leaderboard
    leaderboard = get_study_leaderboard(artifacts_dir)
    
    if leaderboard:
        print("TOP 10 TRIALS:")
        print("-" * 60)
        print(f"{'Trial':<6} {'Value':<8} {'Total Pts':<10} {'Risk Adj':<10} {'Success':<8}")
        print("-" * 60)
        
        for trial in leaderboard[:10]:
            print(f"{trial['trial']:<6} "
                  f"{trial['value']:<8.2f} "
                  f"{trial['total_points']:<10.1f} "
                  f"{trial['risk_adj_points']:<10.1f} "
                  f"{trial['success_rate']:<8.1%}")
        
        print()
        
        # Show best parameters
        if leaderboard:
            best_trial = leaderboard[0]
            print("BEST PARAMETERS:")
            print("-" * 30)
            for param, value in best_trial.get('params', {}).items():
                print(f"{param}: {value}")
    else:
        print("No trial results found.")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="FPL AI Hyperparameter Tuning and Model Retraining",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Tune command
    tune_parser = subparsers.add_parser("tune", help="Run hyperparameter tuning")
    tune_parser.add_argument(
        "--trials", type=int,
        help="Number of trials to run (overrides config)"
    )
    tune_parser.add_argument(
        "--timeout", type=int,
        help="Timeout in minutes (overrides config)"
    )
    tune_parser.add_argument(
        "--quick", action="store_true",
        help="Run quick tune with reduced scope"
    )
    
    # Retrain command
    retrain_parser = subparsers.add_parser("retrain", help="Retrain models")
    retrain_parser.add_argument(
        "--settings", type=str,
        help="Path to settings file with optimized parameters"
    )
    retrain_parser.add_argument(
        "--force", action="store_true",
        help="Force retraining even if models exist"
    )
    retrain_parser.add_argument(
        "--gw", type=int,
        help="Current gameweek for training context"
    )
    
    # Combined command
    both_parser = subparsers.add_parser("both", help="Run tuning then retrain with best settings")
    both_parser.add_argument(
        "--trials", type=int,
        help="Number of trials to run"
    )
    both_parser.add_argument(
        "--timeout", type=int,
        help="Timeout in minutes"
    )
    both_parser.add_argument(
        "--quick", action="store_true",
        help="Run quick tune"
    )
    both_parser.add_argument(
        "--force", action="store_true",
        help="Force retraining even if models exist"
    )
    
    # Results command
    results_parser = subparsers.add_parser("results", help="Show tuning results")
    results_parser.add_argument(
        "--artifacts-dir", type=str,
        help="Path to tuning artifacts directory"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == "tune":
            results = run_hyperparameter_tuning(
                n_trials=args.trials,
                timeout_min=args.timeout,
                quick=args.quick
            )
            print(json.dumps(results, indent=2, default=str))
            
        elif args.command == "retrain":
            results = retrain_models(
                settings_override_path=args.settings,
                force=args.force,
                current_gw=args.gw
            )
            print(json.dumps(results, indent=2, default=str))
            
        elif args.command == "both":
            # Run tuning first
            logger.info("Step 1: Running hyperparameter tuning...")
            tune_results = run_hyperparameter_tuning(
                n_trials=args.trials,
                timeout_min=args.timeout,
                quick=args.quick
            )
            
            if tune_results.get("status") == "completed":
                best_settings_path = tune_results.get("best_settings_path")
                
                logger.info("Step 2: Retraining with best settings...")
                retrain_results = retrain_models(
                    settings_override_path=best_settings_path,
                    force=args.force
                )
                
                combined_results = {
                    "tuning": tune_results,
                    "retraining": retrain_results
                }
            else:
                logger.error("Tuning failed, skipping retraining")
                combined_results = {"tuning": tune_results}
            
            print(json.dumps(combined_results, indent=2, default=str))
            
        elif args.command == "results":
            show_tuning_results(args.artifacts_dir)
            
        else:
            parser.print_help()
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
