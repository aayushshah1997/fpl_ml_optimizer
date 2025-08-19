"""
Rotation prior feature module for pipeline integration.

Provides functions to add manager rotation priors to player datasets for use in
minutes modeling and Monte Carlo simulations.
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from ..common.config import get_config, get_logger

logger = get_logger(__name__)


def add_manager_rotation_prior(players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add manager rotation priors to player dataset.
    
    This function enriches player data with manager information and rotation priors
    computed from historical data. The blended_prior is used as the primary rotation
    risk indicator for minutes modeling.
    
    Args:
        players_df: DataFrame containing player data with team_id column
        
    Returns:
        DataFrame with added columns:
        - manager: Manager name for the player's team
        - manager_rotation_prior: Blended rotation prior for the manager
        - rotation_risk: Clipped rotation risk for modeling (same as manager_rotation_prior, capped at 0.5)
    """
    df = players_df.copy()
    config = get_config()
    
    # Get file paths
    rotation_path = config.project_path / config.get("managers.csv_path", "data/manager_rotation_overrides.csv")
    team_map_path = config.project_path / config.get("managers.team_manager_csv", "data/team_manager_map.csv")
    default_prior = float(config.get("managers.manager_rotation_default", 0.05))
    
    # Load data files
    try:
        if rotation_path.exists():
            rotation_df = pd.read_csv(rotation_path)
            logger.debug(f"Loaded rotation priors from {rotation_path}")
        else:
            rotation_df = pd.DataFrame(columns=["manager", "blended_prior"])
            logger.warning(f"Rotation priors file not found: {rotation_path}")
        
        if team_map_path.exists():
            team_map_df = pd.read_csv(team_map_path)
            logger.debug(f"Loaded team manager mapping from {team_map_path}")
        else:
            team_map_df = pd.DataFrame(columns=["team_id", "manager"])
            logger.warning(f"Team manager mapping file not found: {team_map_path}")
        
    except Exception as e:
        logger.error(f"Failed to load rotation data files: {e}")
        # Fallback to default values
        rotation_df = pd.DataFrame(columns=["manager", "blended_prior"])
        team_map_df = pd.DataFrame(columns=["team_id", "manager"])
    
    # Add manager information if not already present
    if "manager" not in df.columns and "team_id" in df.columns and not team_map_df.empty:
        logger.debug("Adding manager information from team mapping")
        df = df.merge(
            team_map_df[["team_id", "manager"]], 
            on="team_id", 
            how="left"
        )
        logger.info(f"Added manager info for {df['manager'].notna().sum()} players")
    
    # Add rotation priors if manager information is available
    if "manager" in df.columns and not rotation_df.empty:
        logger.debug("Adding rotation priors based on manager")
        
        # Merge with rotation data
        df = df.merge(
            rotation_df[["manager", "blended_prior"]], 
            on="manager", 
            how="left"
        )
        
        # Use blended_prior as the primary rotation prior
        df["manager_rotation_prior"] = df["blended_prior"].fillna(default_prior)
        
        # Drop the temporary blended_prior column
        if "blended_prior" in df.columns:
            df = df.drop(columns=["blended_prior"])
        
        logger.info(f"Added rotation priors for {df['manager_rotation_prior'].notna().sum()} players")
        
        # Log summary statistics
        if df["manager_rotation_prior"].notna().any():
            mean_prior = df["manager_rotation_prior"].mean()
            min_prior = df["manager_rotation_prior"].min()
            max_prior = df["manager_rotation_prior"].max()
            logger.info(f"Rotation prior stats - Mean: {mean_prior:.3f}, Range: {min_prior:.3f}-{max_prior:.3f}")
    else:
        # Fallback to default rotation prior
        logger.warning("No manager information or rotation data available, using default prior")
        df["manager_rotation_prior"] = default_prior
    
    # Create rotation_risk column for modeling (clip at 0.5 for stability)
    df["rotation_risk"] = df["manager_rotation_prior"].clip(0, 0.5)
    
    # Log final statistics
    unique_managers = df.get("manager", pd.Series()).nunique()
    players_with_priors = df["manager_rotation_prior"].notna().sum()
    
    logger.info(f"Rotation prior assignment complete:")
    logger.info(f"  - Players processed: {len(df)}")
    logger.info(f"  - Unique managers: {unique_managers}")
    logger.info(f"  - Players with rotation priors: {players_with_priors}")
    logger.info(f"  - Default prior used for: {len(df) - players_with_priors} players")
    
    return df


def get_manager_rotation_summary() -> pd.DataFrame:
    """
    Get a summary of current manager rotation priors for analysis.
    
    Returns:
        DataFrame with manager rotation summary data
    """
    config = get_config()
    rotation_path = config.project_path / config.get("managers.csv_path", "data/manager_rotation_overrides.csv")
    team_map_path = config.project_path / config.get("managers.team_manager_csv", "data/team_manager_map.csv")
    
    try:
        if not rotation_path.exists() or not team_map_path.exists():
            logger.warning("Required rotation data files not found")
            return pd.DataFrame()
        
        rotation_df = pd.read_csv(rotation_path)
        team_map_df = pd.read_csv(team_map_path)
        
        # Combine team and rotation data
        summary = team_map_df.merge(
            rotation_df[["manager", "blended_prior", "n_matches_prev", "n_matches_curr"]], 
            on="manager", 
            how="left"
        )
        
        # Add summary stats
        summary["data_quality"] = summary.apply(
            lambda row: "Excellent" if row.get("n_matches_curr", 0) >= 8
            else "Good" if row.get("n_matches_curr", 0) >= 4
            else "Limited" if row.get("n_matches_prev", 0) >= 8
            else "Poor", axis=1
        )
        
        return summary
        
    except Exception as e:
        logger.error(f"Failed to generate rotation summary: {e}")
        return pd.DataFrame()


def validate_rotation_priors() -> Dict[str, Any]:
    """
    Validate rotation prior data and return diagnostic information.
    
    Returns:
        Dictionary with validation results and diagnostics
    """
    config = get_config()
    rotation_path = config.project_path / config.get("managers.csv_path", "data/manager_rotation_overrides.csv")
    team_map_path = config.project_path / config.get("managers.team_manager_csv", "data/team_manager_map.csv")
    
    validation = {
        "files_exist": {},
        "data_quality": {},
        "coverage": {},
        "recommendations": []
    }
    
    # Check file existence
    validation["files_exist"]["rotation_data"] = rotation_path.exists()
    validation["files_exist"]["team_mapping"] = team_map_path.exists()
    
    if not all(validation["files_exist"].values()):
        validation["recommendations"].append("Run 'make rotation_update' to generate missing data files")
        return validation
    
    try:
        rotation_df = pd.read_csv(rotation_path)
        team_map_df = pd.read_csv(team_map_path)
        
        # Data quality checks
        validation["data_quality"]["num_managers"] = len(rotation_df)
        validation["data_quality"]["num_teams"] = len(team_map_df)
        validation["data_quality"]["unknown_managers"] = int((team_map_df["manager"] == "Unknown").sum())
        
        if "blended_prior" in rotation_df.columns:
            priors = rotation_df["blended_prior"].dropna()
            validation["data_quality"]["prior_range"] = [float(priors.min()), float(priors.max())]
            validation["data_quality"]["avg_prior"] = float(priors.mean())
        
        # Coverage analysis
        if "n_matches_curr" in rotation_df.columns:
            sufficient_data = (rotation_df["n_matches_curr"] >= 8).sum()
            validation["coverage"]["teams_with_sufficient_current_data"] = int(sufficient_data)
            validation["coverage"]["pct_sufficient_data"] = float(sufficient_data / len(rotation_df) * 100)
        
        # Recommendations
        if validation["data_quality"]["unknown_managers"] > 0:
            validation["recommendations"].append(
                f"Consider adding overrides for {validation['data_quality']['unknown_managers']} unknown managers"
            )
        
        if validation["coverage"].get("pct_sufficient_data", 0) < 70:
            validation["recommendations"].append(
                "Consider updating rotation data more frequently or adjusting min_matches_stable threshold"
            )
        
    except Exception as e:
        validation["error"] = str(e)
        validation["recommendations"].append("Check data file formats and re-run rotation engine update")
    
    return validation
