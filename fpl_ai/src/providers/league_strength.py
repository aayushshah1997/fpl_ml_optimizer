"""
League strength provider for dynamic strength multipliers and uncertainty handling.

This module handles:
- Loading baseline league strength mappings from config
- Optional CSV overrides for manual league strength tuning  
- Logging all encountered leagues for audit/tuning purposes
- Sample weight multipliers based on league strength curves
- Uncertainty bumps for Monte Carlo simulation
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Set, Tuple
import pandas as pd
from ..common.config import get_config, get_logger

logger = get_logger(__name__)


def load_league_strength(settings: dict = None) -> dict:
    """
    Load league strength mappings from config and optional CSV overrides.
    
    Args:
        settings: Configuration dict (if None, loads from get_config())
        
    Returns:
        Dict mapping league_id (str) -> strength (float)
    """
    if settings is None:
        settings = get_config().get_settings()
    
    # Start with baseline settings
    base = dict(settings.get("training", {}).get("league_strength", {}))
    
    # Apply optional CSV overrides: columns [league_id, strength]
    csv_path = settings.get("io", {}).get("league_strength_overrides_csv")
    if csv_path and Path(csv_path).exists():
        try:
            df = pd.read_csv(csv_path)
            for _, r in df.iterrows():
                lid = str(int(r["league_id"]))
                strength = float(r["strength"])
                base[lid] = strength
                logger.debug(f"Applied league strength override: {lid} -> {strength}")
        except Exception as e:
            logger.warning(f"Failed to load league strength overrides from {csv_path}: {e}")
    
    return base


def log_seen_leagues(league_ids: Set[str], settings: dict = None) -> None:
    """
    Log encountered leagues to JSON file for audit and future tuning.
    
    Args:
        league_ids: Set of league IDs encountered in this run
        settings: Configuration dict (if None, loads from get_config())
    """
    if settings is None:
        settings = get_config().get_settings()
    
    if not league_ids:
        return
    
    log_path = Path(settings.get("io", {}).get("league_seen_log", "cache/league_seen.json"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing counts
    current_counts = {}
    if log_path.exists():
        try:
            current_counts = json.loads(log_path.read_text())
        except Exception as e:
            logger.warning(f"Failed to load existing league seen log: {e}")
            current_counts = {}
    
    # Update counts
    for lid in league_ids:
        lid_str = str(lid)
        current_counts[lid_str] = current_counts.get(lid_str, 0) + 1
    
    # Save updated counts
    try:
        log_path.write_text(json.dumps(current_counts, indent=2, sort_keys=True))
        logger.info(f"Updated league seen log with {len(league_ids)} leagues: {log_path}")
    except Exception as e:
        logger.error(f"Failed to save league seen log: {e}")


def strength_and_weight_mult(league_id: str, settings: dict = None) -> Tuple[float, float]:
    """
    Get league strength and sample weight multiplier for a given league.
    
    Args:
        league_id: League ID as string
        settings: Configuration dict (if None, loads from get_config())
        
    Returns:
        Tuple of (strength, sample_weight_multiplier)
    """
    if settings is None:
        settings = get_config().get_settings()
    
    lid = str(league_id)
    
    # Get league strength (with fallback to unknown_league_default)
    ls_map = load_league_strength(settings)
    training_config = settings.get("training", {})
    
    strength = float(ls_map.get(lid, training_config.get("unknown_league_default", 0.78)))
    
    # Derive sample weight multiplier from piecewise curve
    curve = training_config.get("sample_weight_curve", [])
    sw_mult = 1.0  # default if no curve defined
    
    for threshold, multiplier in curve:
        if strength <= threshold:
            sw_mult = multiplier
            break
    
    # Apply minimum multiplier floor
    min_mult = training_config.get("sample_weight_min_mult", 0.6)
    sw_mult = max(sw_mult, min_mult)
    
    return strength, sw_mult


def get_uncertainty_bump(league_id: str, settings: dict = None) -> float:
    """
    Get uncertainty bump for Monte Carlo simulation based on league tier.
    
    Args:
        league_id: League ID as string  
        settings: Configuration dict (if None, loads from get_config())
        
    Returns:
        Uncertainty bump (0.0 if league_strength >= 0.85, else lowtier_uncertainty_bump)
    """
    if settings is None:
        settings = get_config().get_settings()
    
    strength, _ = strength_and_weight_mult(league_id, settings)
    training_config = settings.get("training", {})
    
    if strength < 0.85:
        return training_config.get("lowtier_uncertainty_bump", 0.15)
    else:
        return 0.0


def is_lowtier_league(league_id: str, settings: dict = None) -> bool:
    """
    Check if league is considered low-tier (strength < 0.85).
    
    Args:
        league_id: League ID as string
        settings: Configuration dict (if None, loads from get_config())
        
    Returns:
        True if league strength < 0.85, False otherwise
    """
    if settings is None:
        settings = get_config().get_settings()
    
    strength, _ = strength_and_weight_mult(league_id, settings)
    return strength < 0.85


def get_new_leagues_report(settings: dict = None) -> Dict[str, Dict]:
    """
    Generate a report of newly seen leagues for audit purposes.
    
    Args:
        settings: Configuration dict (if None, loads from get_config())
        
    Returns:
        Dict with league analysis: {league_id: {count, strength, is_new, needs_override}}
    """
    if settings is None:
        settings = get_config().get_settings()
    
    log_path = Path(settings.get("io", {}).get("league_seen_log", "cache/league_seen.json"))
    
    if not log_path.exists():
        return {}
    
    try:
        seen_leagues = json.loads(log_path.read_text())
    except Exception:
        return {}
    
    ls_map = load_league_strength(settings)
    unknown_default = settings.get("training", {}).get("unknown_league_default", 0.78)
    
    report = {}
    for league_id, count in seen_leagues.items():
        strength = float(ls_map.get(league_id, unknown_default))
        is_using_default = league_id not in ls_map
        
        report[league_id] = {
            "count": count,
            "strength": strength,
            "is_using_default": is_using_default,
            "needs_override": is_using_default and count >= 5  # Suggest override if seen 5+ times
        }
    
    return report
