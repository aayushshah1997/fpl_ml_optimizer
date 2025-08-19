"""
Manager resolution module for FBR API-only manager discovery.

Provides functions to resolve current Premier League managers using the FBR API
with CSV override support and fail-soft behavior.
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path
from ..common.config import get_config, get_logger
from .fbrapi_client import FBRAPIClient

logger = get_logger(__name__)


def resolve_current_managers() -> pd.DataFrame:
    """
    Returns DataFrame: [team_id, team_name, manager] for the CURRENT season, using FBR staff,
    with optional overrides CSV.
    
    Returns:
        DataFrame with columns: team_id, team_name, manager
    """
    config = get_config()
    client = FBRAPIClient()
    
    season_curr = config.get("rotation_engine.season_curr", "2025-2026")
    overrides_path = config.get("managers.overrides_csv", "data/manager_overrides.csv")
    
    # Load overrides if available
    overrides_full_path = config.project_path / overrides_path
    if overrides_full_path.exists():
        overrides = pd.read_csv(overrides_full_path)
        logger.info(f"Loaded manager overrides from {overrides_full_path}")
    else:
        overrides = pd.DataFrame(columns=["team_id", "team_name", "manager"])
        logger.info("No manager overrides file found")
    
    # Get PL teams
    teams = client.get_pl_teams()
    if teams.empty:
        logger.error("Failed to retrieve PL teams from FBR API")
        return pd.DataFrame(columns=["team_id", "team_name", "manager"])
    
    logger.info(f"Retrieved {len(teams)} PL teams from FBR API")
    
    rows = []
    for _, team in teams.iterrows():
        tid = str(team.get("team_id", team.get("id", "")))
        tname = str(team.get("name", team.get("team_name", "")))
        mgr = None
        
        # Try to get current season staff from API
        try:
            staff = client.get_team_staff(tid, season_curr)
            if not staff.empty:
                # Look for manager or head coach role
                manager_candidates = staff[
                    staff.get("role", "").str.contains("manager|head coach", case=False, na=False)
                ]
                if not manager_candidates.empty:
                    mgr = str(manager_candidates.iloc[0]["name"])
                    logger.debug(f"Found manager {mgr} for team {tname} via API")
        except Exception as e:
            logger.warning(f"Failed to get staff for team {tname} (ID: {tid}): {e}")
        
        # Check for override
        if not overrides.empty and (overrides["team_id"].astype(str) == tid).any():
            override_mgr = str(overrides[overrides["team_id"].astype(str) == tid].iloc[0]["manager"])
            if override_mgr and override_mgr != "nan":
                mgr = override_mgr
                logger.info(f"Using override manager {mgr} for team {tname}")
        
        # Fall back to Unknown if no manager found
        if not mgr or mgr == "nan":
            mgr = "Unknown"
            logger.warning(f"No manager found for team {tname}, using 'Unknown'")
        
        rows.append({
            "team_id": int(tid) if tid.isdigit() else 0,
            "team_name": tname,
            "manager": mgr
        })
    
    result = pd.DataFrame(rows)
    logger.info(f"Resolved managers for {len(result)} teams")
    return result
