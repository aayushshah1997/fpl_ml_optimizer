"""
Optuna-based hyperparameter tuning for FPL ML models.

Integrates with walk-forward backtesting to optimize:
- LightGBM hyperparameters (global and position-specific)
- Monte Carlo simulation parameters
- Captain selection policies
- Risk management parameters
"""

from __future__ import annotations
import optuna
import json
import time
import warnings
from pathlib import Path
from copy import deepcopy
from typing import Dict, Any, Optional

from ..common.config import load_settings, save_settings_dict, get_logger
from .walkforward import walk_forward_backtest, analyze_backtest_results

# Suppress Optuna warnings
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

logger = get_logger(__name__)


def apply_trial_overrides(cfg: Dict, trial: optuna.trial.Trial, search_space: Dict) -> Dict:
    """
    Apply Optuna trial suggestions to configuration.
    
    Args:
        cfg: Base configuration dictionary
        trial: Optuna trial object
        search_space: Search space configuration with dotted keys
        
    Returns:
        Modified configuration with trial suggestions
    """
    modified_cfg = deepcopy(cfg)
    
    for param_path, choices in search_space.items():
        # Suggest value from choices
        if isinstance(choices, list):
            if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in choices):
                # Numeric choices - use suggest_categorical to avoid rounding issues
                value = trial.suggest_categorical(param_path, choices)
            else:
                # Mixed or string choices
                value = trial.suggest_categorical(param_path, choices)
        else:
            logger.warning(f"Invalid search space format for {param_path}: {choices}")
            continue
        
        # Apply the value to the nested configuration
        keys = param_path.split(".")
        target_dict = modified_cfg
        
        # Navigate to the parent dictionary
        for key in keys[:-1]:
            if key not in target_dict:
                target_dict[key] = {}
            target_dict = target_dict[key]
        
        # Set the final value
        target_dict[keys[-1]] = value
        
        logger.debug(f"Trial {trial.number}: {param_path} = {value}")
    
    return modified_cfg


def run_optuna_study() -> Dict[str, Any]:
    """
    Run Optuna hyperparameter optimization study.
    
    Returns:
        Dictionary with optimization results
    """
    logger.info("Starting Optuna hyperparameter optimization study")
    
    # Load configuration
    cfg = load_settings()
    
    # Extract tuning configuration
    autotune_config = cfg.get("autotune", {})
    backtest_config = cfg.get("backtest", {})
    search_space = cfg.get("spaces", {})
    
    if not autotune_config.get("enabled", True):
        logger.info("Auto-tuning is disabled in configuration")
        return {"status": "disabled"}
    
    if not search_space:
        logger.warning("No search space defined in configuration")
        return {"status": "no_search_space"}
    
    # Study configuration
    study_name = autotune_config.get("study_name", "fpl_auto_tune")
    n_trials = int(autotune_config.get("n_trials", 60))
    timeout_minutes = int(autotune_config.get("timeout_min", 60))
    timeout_seconds = timeout_minutes * 60
    pruner_type = autotune_config.get("pruner", "median")
    
    # Backtest configuration
    seasons = backtest_config.get("seasons", ["2022-2023", "2023-2024", "2024-2025"])
    start_gw = int(backtest_config.get("start_gw", 2))
    end_gw = int(backtest_config.get("end_gw", 38))
    objective_name = backtest_config.get("objective", "risk_adj_points")
    
    logger.info(f"Study: {study_name}, {n_trials} trials, {timeout_minutes}min timeout")
    logger.info(f"Backtest: {len(seasons)} seasons, GW {start_gw}-{end_gw}")
    logger.info(f"Objective: {objective_name}")
    logger.info(f"Search space: {len(search_space)} parameters")
    
    # Create pruner
    if pruner_type == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=1
        )
    else:
        pruner = optuna.pruners.NopPruner()
    
    # Create artifacts directory
    artifacts_dir = Path("fpl_ai/artifacts/tuning")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Define objective function
    def objective(trial: optuna.trial.Trial) -> float:
        """Objective function for Optuna optimization."""
        try:
            trial_start_time = time.time()
            
            # Apply trial suggestions to configuration
            trial_cfg = apply_trial_overrides(cfg, trial, search_space)
            
            # Run walk-forward backtest
            logger.debug(f"Trial {trial.number}: Running backtest...")
            backtest_df, metrics = walk_forward_backtest(
                seasons=seasons,
                start_gw=start_gw,
                end_gw=end_gw,
                settings=trial_cfg
            )
            
            # Calculate objective score
            if objective_name == "risk_adj_points":
                score = metrics["risk_adj_points"]
            elif objective_name == "sharpe_like":
                score = metrics["sharpe_like"]
            elif objective_name == "total_points":
                score = metrics["total_points"]
            else:
                logger.warning(f"Unknown objective {objective_name}, using total_points")
                score = metrics["total_points"]
            
            trial_time = time.time() - trial_start_time
            
            # Save trial artifacts
            trial_dir = artifacts_dir / f"trial_{trial.number}"
            trial_dir.mkdir(exist_ok=True)
            
            # Save backtest results
            backtest_df.to_csv(trial_dir / "backtest_results.csv", index=False)
            
            # Save metrics
            with open(trial_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2, default=str)
            
            # Save trial configuration
            with open(trial_dir / "config.json", "w") as f:
                json.dump(trial_cfg, f, indent=2, default=str)
            
            # Store user attributes for later analysis
            trial.set_user_attr("metrics", metrics)
            trial.set_user_attr("trial_time", trial_time)
            trial.set_user_attr("successful_gws", metrics.get("successful_gws", 0))
            trial.set_user_attr("success_rate", metrics.get("success_rate", 0.0))
            
            logger.info(f"Trial {trial.number}: {objective_name}={score:.2f}, "
                       f"time={trial_time:.1f}s, success_rate={metrics.get('success_rate', 0):.1%}")
            
            return float(score)
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            # Return a very low score for failed trials
            return -1000.0
    
    # Create and run study
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        pruner=pruner
    )
    
    study_start_time = time.time()
    
    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout_seconds,
            show_progress_bar=False,  # We'll use our own logging
            callbacks=[_log_trial_callback]
        )
    except KeyboardInterrupt:
        logger.info("Study interrupted by user")
    except Exception as e:
        logger.error(f"Study failed: {e}")
        return {"status": "error", "error": str(e)}
    
    study_time = time.time() - study_start_time
    
    # Extract results
    if not study.trials:
        logger.error("No trials completed")
        return {"status": "no_trials"}
    
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value
    best_metrics = best_trial.user_attrs.get("metrics", {})
    
    logger.info(f"Study completed: {len(study.trials)} trials in {study_time:.1f}s")
    logger.info(f"Best trial: #{best_trial.number}, {objective_name}={best_value:.2f}")
    
    # Generate best configuration
    best_cfg = apply_trial_overrides(cfg, best_trial, search_space)
    
    # Save best configuration
    best_settings_path = Path(autotune_config.get("persist_best_to", "fpl_ai/artifacts/tuning/best_settings.yaml"))
    save_settings_dict(best_cfg, best_settings_path)
    logger.info(f"Best configuration saved to {best_settings_path}")
    
    # Optional: Write back to main settings
    if autotune_config.get("write_back_to_settings", False):
        save_settings_dict(best_cfg, Path("settings.yaml"))
        logger.info("Best configuration written back to settings.yaml")
    
    # Create leaderboard
    leaderboard = []
    for trial in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
        trial_metrics = trial.user_attrs.get("metrics", {})
        leaderboard.append({
            "trial": trial.number,
            "value": trial.value,
            "total_points": trial_metrics.get("total_points", 0),
            "risk_adj_points": trial_metrics.get("risk_adj_points", 0),
            "sharpe_like": trial_metrics.get("sharpe_like", 0),
            "success_rate": trial_metrics.get("success_rate", 0),
            "trial_time": trial.user_attrs.get("trial_time", 0),
            "params": trial.params
        })
    
    # Sort by objective value
    leaderboard.sort(key=lambda x: x["value"], reverse=True)
    
    # Save leaderboard
    leaderboard_path = artifacts_dir / "leaderboard.json"
    with open(leaderboard_path, "w") as f:
        json.dump(leaderboard, f, indent=2, default=str)
    
    # Save study summary
    study_summary = {
        "study_name": study_name,
        "n_trials": len(study.trials),
        "n_complete": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_failed": len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
        "study_time_seconds": study_time,
        "best_trial": best_trial.number,
        "best_value": best_value,
        "best_params": best_params,
        "objective": objective_name,
        "search_space": search_space,
        "backtest_config": {
            "seasons": seasons,
            "start_gw": start_gw,
            "end_gw": end_gw
        }
    }
    
    summary_path = artifacts_dir / "study_summary.json"
    with open(summary_path, "w") as f:
        json.dump(study_summary, f, indent=2, default=str)
    
    results = {
        "status": "completed",
        "study_summary": study_summary,
        "best_trial": best_trial.number,
        "best_value": best_value,
        "best_metrics": best_metrics,
        "best_params": best_params,
        "best_settings_path": str(best_settings_path),
        "leaderboard_path": str(leaderboard_path),
        "artifacts_dir": str(artifacts_dir),
        "n_trials_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    }
    
    logger.info("Optuna study completed successfully")
    return results


def _log_trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Callback to log trial progress."""
    if trial.state == optuna.trial.TrialState.COMPLETE:
        logger.info(f"Trial {trial.number} completed: value={trial.value:.3f}")
    elif trial.state == optuna.trial.TrialState.PRUNED:
        logger.info(f"Trial {trial.number} pruned")
    elif trial.state == optuna.trial.TrialState.FAIL:
        logger.warning(f"Trial {trial.number} failed")


def load_best_settings(settings_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load best settings from tuning results.
    
    Args:
        settings_path: Optional path to best settings file
        
    Returns:
        Best configuration dictionary
    """
    if settings_path is None:
        # Try to find best settings from default location
        default_path = Path("fpl_ai/artifacts/tuning/best_settings.yaml")
        if default_path.exists():
            settings_path = str(default_path)
        else:
            logger.warning("No best settings found, using default configuration")
            return load_settings()
    
    try:
        return load_settings(settings_path)
    except Exception as e:
        logger.error(f"Failed to load best settings from {settings_path}: {e}")
        return load_settings()


def get_study_leaderboard(artifacts_dir: Optional[str] = None) -> list[Dict[str, Any]]:
    """
    Get study leaderboard from artifacts.
    
    Args:
        artifacts_dir: Optional path to artifacts directory
        
    Returns:
        List of trial results sorted by performance
    """
    if artifacts_dir is None:
        artifacts_dir = "fpl_ai/artifacts/tuning"
    
    leaderboard_path = Path(artifacts_dir) / "leaderboard.json"
    
    if not leaderboard_path.exists():
        logger.warning(f"No leaderboard found at {leaderboard_path}")
        return []
    
    try:
        with open(leaderboard_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load leaderboard: {e}")
        return []


def quick_tune(n_trials: int = 20, timeout_min: int = 30) -> Dict[str, Any]:
    """
    Run a quick tuning session with reduced scope.
    
    Args:
        n_trials: Number of trials to run
        timeout_min: Timeout in minutes
        
    Returns:
        Tuning results
    """
    logger.info(f"Running quick tune: {n_trials} trials, {timeout_min}min timeout")
    
    # Load settings and modify for quick tune
    cfg = load_settings()
    
    # Reduce backtest scope for speed
    cfg["backtest"]["seasons"] = ["2023-2024", "2024-2025"]  # Only recent seasons
    cfg["backtest"]["start_gw"] = 10  # Skip early season
    cfg["backtest"]["end_gw"] = 25   # End mid-season
    
    # Update autotune settings
    cfg["autotune"]["n_trials"] = n_trials
    cfg["autotune"]["timeout_min"] = timeout_min
    
    # Reduce search space for speed (keep only key parameters)
    reduced_space = {}
    full_space = cfg.get("spaces", {})
    
    # Keep only core LightGBM parameters
    key_params = [
        "modeling.gbm.learning_rate",
        "modeling.gbm.num_leaves", 
        "modeling.gbm.n_estimators",
        "mc.cvar_alpha",
        "captain.policy"
    ]
    
    for param in key_params:
        if param in full_space:
            reduced_space[param] = full_space[param]
    
    cfg["spaces"] = reduced_space
    
    logger.info(f"Quick tune search space: {len(reduced_space)} parameters")
    
    # Save temporary config and run study
    temp_config_path = Path("fpl_ai/artifacts/tuning/quick_tune_config.yaml")
    save_settings_dict(cfg, temp_config_path)
    
    # Temporarily override global config
    import sys
    original_load_settings = sys.modules[__name__].load_settings
    sys.modules[__name__].load_settings = lambda: cfg
    
    try:
        results = run_optuna_study()
        results["mode"] = "quick_tune"
        return results
    finally:
        # Restore original function
        sys.modules[__name__].load_settings = original_load_settings
