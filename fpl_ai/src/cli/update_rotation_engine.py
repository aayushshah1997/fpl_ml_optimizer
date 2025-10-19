"""
CLI tool for updating manager data and rotation priors.

This tool refreshes manager mappings and computes data-driven rotation priors
from previous season and current season to date using the FBR API.
"""

from __future__ import annotations
import argparse
import pandas as pd
from pathlib import Path
from ..common.config import get_config, get_logger
from ..providers.manager_resolver import resolve_current_managers
from ..features.rotation_metrics import compute_rotation_metrics_for_team, map_metrics_to_prior

logger = get_logger(__name__)


def main():
    """Main CLI entry point for rotation engine updates."""
    parser = argparse.ArgumentParser(
        description="Update manager mappings and rotation priors from FBR API data"
    )
    parser.add_argument(
        "--write", 
        action="store_true", 
        help="Write CSV outputs to data directory"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    config = get_config()
    
    # Get configuration
    season_prev = config.get("rotation_engine.season_prev", "2024-2025")
    season_curr = config.get("rotation_engine.season_curr", "2025-2026")
    min_stable = int(config.get("rotation_engine.min_matches_stable", 8))
    default_prior = float(config.get("managers.manager_rotation_default", 0.05))
    
    team_map_path = config.project_path / config.get("managers.team_manager_csv", "data/team_manager_map.csv")
    rotation_path = config.project_path / config.get("managers.csv_path", "data/manager_rotation_overrides.csv")
    
    logger.info("Starting rotation engine update...")
    logger.info(f"Previous season: {season_prev}")
    logger.info(f"Current season: {season_curr}")
    logger.info(f"Minimum stable matches: {min_stable}")
    
    # Step 1: Resolve current managers
    logger.info("Resolving current season managers...")
    try:
        team_managers = resolve_current_managers()
        if team_managers.empty:
            logger.error("Failed to resolve any managers")
            return 1
        
        logger.info(f"Resolved managers for {len(team_managers)} teams")
    except Exception as e:
        logger.error(f"Failed to resolve managers: {e}")
        return 1
    
    # Step 2: Compute rotation metrics and priors
    logger.info("Computing rotation metrics and priors...")
    records = []
    
    for _, team_row in team_managers.iterrows():
        team_id = str(team_row["team_id"])
        team_name = team_row["team_name"]
        manager = team_row["manager"]
        
        logger.info(f"Processing team: {team_name} (ID: {team_id}, Manager: {manager})")
        
        try:
            # Previous season metrics
            logger.debug(f"Computing previous season metrics for team {team_id}")
            metrics_prev = compute_rotation_metrics_for_team(team_id, season_prev)
            prior_prev = (
                map_metrics_to_prior(metrics_prev) 
                if metrics_prev["n_matches"] > 0 
                else default_prior
            )
            
            # Current season metrics
            logger.debug(f"Computing current season metrics for team {team_id}")
            metrics_curr = compute_rotation_metrics_for_team(team_id, season_curr)
            prior_curr = (
                map_metrics_to_prior(metrics_curr) 
                if metrics_curr["n_matches"] > 0 
                else None
            )
            
            # Compute blended prior
            if metrics_curr["n_matches"] >= min_stable:
                # Sufficient current season data - use it primarily
                blended_prior = prior_curr
                logger.debug(f"Using current season prior for {team_name}: {blended_prior:.3f}")
            elif prior_curr is not None:
                # Blend current and previous season
                blended_prior = 0.6 * prior_curr + 0.4 * prior_prev
                logger.debug(f"Blending priors for {team_name}: {blended_prior:.3f} (curr: {prior_curr:.3f}, prev: {prior_prev:.3f})")
            else:
                # Fall back to previous season or default
                blended_prior = prior_prev
                logger.debug(f"Using previous season prior for {team_name}: {blended_prior:.3f}")
            
            records.append({
                "team_id": int(team_id),
                "team_name": team_name,
                "manager": manager,
                "prior_prev_season": round(prior_prev, 3),
                "prior_curr_ytd": round(prior_curr, 3) if prior_curr is not None else None,
                "blended_prior": round(float(blended_prior), 3),
                "n_matches_prev": metrics_prev["n_matches"],
                "n_matches_curr": metrics_curr["n_matches"],
                # Additional metrics for debugging
                "xi_change_pct_prev": round(metrics_prev.get("xi_change_pct", 0.0), 3),
                "xi_change_pct_curr": round(metrics_curr.get("xi_change_pct", 0.0), 3),
                "starts_variance_prev": round(metrics_prev.get("starts_variance", 0.0), 3),
                "starts_variance_curr": round(metrics_curr.get("starts_variance", 0.0), 3)
            })
            
        except Exception as e:
            logger.error(f"Failed to process team {team_name}: {e}")
            # Add fallback record
            records.append({
                "team_id": int(team_id),
                "team_name": team_name,
                "manager": manager,
                "prior_prev_season": default_prior,
                "prior_curr_ytd": None,
                "blended_prior": default_prior,
                "n_matches_prev": 0,
                "n_matches_curr": 0,
                "xi_change_pct_prev": 0.0,
                "xi_change_pct_curr": 0.0,
                "starts_variance_prev": 0.0,
                "starts_variance_curr": 0.0
            })
    
    rotation_df = pd.DataFrame(records)
    
    # Display results
    print("\n" + "="*60)
    print("ROTATION ENGINE SUMMARY")
    print("="*60)
    
    print("\n--- Team Manager Mapping ---")
    team_display = team_managers[["team_id", "team_name", "manager"]].copy()
    print(team_display.to_string(index=False))
    
    print("\n--- Rotation Priors Summary ---")
    summary_cols = [
        "team_name", "manager", "blended_prior", 
        "n_matches_prev", "n_matches_curr"
    ]
    summary_df = rotation_df[summary_cols].copy()
    print(summary_df.to_string(index=False))
    
    print(f"\n--- Statistics ---")
    print(f"Teams processed: {len(rotation_df)}")
    print(f"Teams with current season data: {len(rotation_df[rotation_df['n_matches_curr'] > 0])}")
    print(f"Teams with stable current data (>= {min_stable} matches): {len(rotation_df[rotation_df['n_matches_curr'] >= min_stable])}")
    print(f"Average blended prior: {rotation_df['blended_prior'].mean():.3f}")
    print(f"Prior range: {rotation_df['blended_prior'].min():.3f} - {rotation_df['blended_prior'].max():.3f}")
    
    # Write output files if requested
    if args.write:
        try:
            # Ensure directories exist
            team_map_path.parent.mkdir(parents=True, exist_ok=True)
            rotation_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write team manager mapping
            team_managers.to_csv(team_map_path, index=False)
            logger.info(f"Wrote team manager mapping to {team_map_path}")
            
            # Write rotation priors (output format for pipeline consumption)
            output_cols = [
                "manager", "prior_prev_season", "prior_curr_ytd", 
                "blended_prior", "n_matches_prev", "n_matches_curr"
            ]
            rotation_output = rotation_df[output_cols].copy()
            rotation_output.to_csv(rotation_path, index=False)
            logger.info(f"Wrote rotation priors to {rotation_path}")
            
            print(f"\n[✓] Files written:")
            print(f"    Team managers: {team_map_path}")
            print(f"    Rotation priors: {rotation_path}")
            
        except Exception as e:
            logger.error(f"Failed to write output files: {e}")
            return 1
    else:
        print("\n[i] Dry run mode. Use --write to save CSV outputs.")
    
    logger.info("Rotation engine update completed successfully")
    return 0


if __name__ == "__main__":
    exit(main())

