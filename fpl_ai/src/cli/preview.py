"""
Preview tool for FPL AI predictions.

Quick preview and summary of prediction frames and model outputs.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from ..common.config import get_config, get_logger
from ..common.logging_setup import setup_logging
from ..common.timeutil import get_current_gw
from ..features.builder import build_prediction_frame

logger = get_logger(__name__)


def format_player_summary(player_row: pd.Series) -> str:
    """
    Format a single player summary.
    
    Args:
        player_row: Player data row
        
    Returns:
        Formatted player string
    """
    name = player_row.get('web_name', 'Unknown')
    position = player_row.get('position', '?')
    team = player_row.get('team_name', player_row.get('team_short', '?'))
    
    # Cost
    cost = player_row.get('now_cost', 0)
    if cost > 20:  # Convert from tenths
        cost /= 10
    cost_str = f"£{cost:.1f}M"
    
    # Projection
    proj = player_row.get('proj_points', 0)
    proj_str = f"{proj:.1f}pts"
    
    # Recent form (if available)
    recent_form = ""
    if 'r3_points_per_game' in player_row:
        r3_form = player_row.get('r3_points_per_game', 0)
        recent_form = f" (R3: {r3_form:.1f})"
    elif 'r3_points' in player_row:
        r3_total = player_row.get('r3_points', 0)
        recent_form = f" (R3: {r3_total:.0f})"
    
    # Expected minutes (if available)
    minutes_str = ""
    if 'expected_minutes' in player_row:
        exp_mins = player_row.get('expected_minutes', 0)
        minutes_str = f" [{exp_mins:.0f}min]"
    
    return f"{name:20} {position:3} {team:3} {cost_str:8} {proj_str:8}{recent_form}{minutes_str}"


def preview_predictions_frame(df: pd.DataFrame, num_players: int = 20) -> str:
    """
    Generate preview of predictions frame.
    
    Args:
        df: Predictions DataFrame
        num_players: Number of top players to show
        
    Returns:
        Formatted preview string
    """
    if df.empty:
        return "No prediction data available"
    
    preview = []
    preview.append("=" * 80)
    preview.append("FPL AI PREDICTIONS PREVIEW")
    preview.append("=" * 80)
    preview.append("")
    
    # Overview
    total_players = len(df)
    positions_count = df.get('position', pd.Series()).value_counts().to_dict()
    
    preview.append(f"Total Players: {total_players:,}")
    preview.append("Position Breakdown:")
    for position in ['GK', 'DEF', 'MID', 'FWD']:
        count = positions_count.get(position, 0)
        preview.append(f"  {position}: {count}")
    
    preview.append("")
    
    # Feature summary
    feature_categories = {
        "Basic": ["web_name", "position", "team_name", "now_cost"],
        "Projections": ["proj_points", "expected_minutes"],
        "Recent Form": [col for col in df.columns if col.startswith("r3_") or col.startswith("r5_")],
        "Advanced": [col for col in df.columns if any(x in col for x in ["xG", "xA", "BPS"])]
    }
    
    preview.append("Available Features:")
    for category, features in feature_categories.items():
        available = [f for f in features if f in df.columns]
        preview.append(f"  {category}: {len(available)}/{len(features)} features")
    
    preview.append("")
    
    # Top players overall
    if 'proj_points' in df.columns:
        top_overall = df.nlargest(num_players, 'proj_points')
        
        preview.append(f"TOP {num_players} PLAYERS BY PROJECTION:")
        preview.append("-" * 50)
        preview.append(f"{'Name':<20} {'Pos':<3} {'Team':<3} {'Cost':<8} {'Proj':<8} {'Form'}")
        preview.append("-" * 80)
        
        for _, player in top_overall.iterrows():
            preview.append(format_player_summary(player))
    
    preview.append("")
    
    # Top players by position
    for position in ['GK', 'DEF', 'MID', 'FWD']:
        pos_df = df[df.get('position') == position] if 'position' in df.columns else pd.DataFrame()
        
        if not pos_df.empty and 'proj_points' in pos_df.columns:
            top_pos = pos_df.nlargest(min(5, len(pos_df)), 'proj_points')
            
            preview.append(f"TOP {position}S:")
            preview.append("-" * 20)
            
            for _, player in top_pos.iterrows():
                preview.append(format_player_summary(player))
            
            preview.append("")
    
    preview.append("=" * 80)
    
    return "\n".join(preview)


def preview_saved_predictions(target_gw: int) -> str:
    """
    Preview saved predictions from file.
    
    Args:
        target_gw: Gameweek to preview
        
    Returns:
        Formatted preview string
    """
    config = get_config()
    artifacts_dir = Path(config.get("io", {}).get("out_dir", "artifacts"))
    pred_file = artifacts_dir / f"predictions_gw{target_gw}.csv"
    
    if not pred_file.exists():
        return f"No saved predictions found for GW {target_gw} at {pred_file}"
    
    try:
        df = pd.read_csv(pred_file)
        
        # Add Monte Carlo summary if available
        mc_cols = ['mean', 'p10', 'p90', 'std']
        mc_available = [col for col in mc_cols if col in df.columns]
        
        preview = preview_predictions_frame(df)
        
        if mc_available:
            preview += "\n\nMONTE CARLO SUMMARY:\n"
            preview += "-" * 20 + "\n"
            preview += f"Available MC metrics: {', '.join(mc_available)}\n"
            
            if 'mean' in df.columns and 'std' in df.columns:
                overall_mean = df['mean'].mean()
                overall_std = df['std'].mean()
                preview += f"Overall average projection: {overall_mean:.2f} ± {overall_std:.2f}\n"
        
        return preview
        
    except Exception as e:
        return f"Error loading predictions from {pred_file}: {str(e)}"


def preview_team_optimization(target_gw: int, budget: float = 100.0) -> str:
    """
    Preview optimal team selection.
    
    Args:
        target_gw: Gameweek to optimize for
        budget: Budget constraint
        
    Returns:
        Formatted team preview
    """
    try:
        # Load predictions
        config = get_config()
        artifacts_dir = Path(config.get("io", {}).get("out_dir", "artifacts"))
        pred_file = artifacts_dir / f"predictions_gw{target_gw}.csv"
        
        if not pred_file.exists():
            return f"No predictions available for GW {target_gw}"
        
        df = pd.read_csv(pred_file)
        
        # Quick optimal team selection (simplified)
        from ..optimize.optimizer import TeamOptimizer
        optimizer = TeamOptimizer()
        
        result = optimizer.optimize_team(df, budget=budget, objective="mean")
        
        if not result:
            return "Team optimization failed"
        
        squad = result.get('squad', [])
        expected_points = result.get('expected_points', 0)
        total_cost = result.get('total_cost', 0)
        
        preview = []
        preview.append("=" * 60)
        preview.append(f"OPTIMAL TEAM PREVIEW - GW {target_gw}")
        preview.append("=" * 60)
        preview.append("")
        preview.append(f"Expected Points: {expected_points:.1f}")
        preview.append(f"Total Cost: £{total_cost:.1f}M")
        preview.append(f"Remaining Budget: £{budget - total_cost:.1f}M")
        preview.append("")
        
        # Group by position
        by_position = {}
        for player in squad:
            pos = player.get('position', 'UNKNOWN')
            if pos not in by_position:
                by_position[pos] = []
            by_position[pos].append(player)
        
        for position in ['GK', 'DEF', 'MID', 'FWD']:
            if position in by_position:
                preview.append(f"{position}S:")
                preview.append("-" * 10)
                
                pos_players = by_position[position]
                pos_players.sort(key=lambda x: x.get('proj_points', 0), reverse=True)
                
                for player in pos_players:
                    player_series = pd.Series(player)
                    preview.append(f"  {format_player_summary(player_series)}")
                
                preview.append("")
        
        return "\n".join(preview)
        
    except Exception as e:
        return f"Error generating team preview: {str(e)}"


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="FPL AI Preview Tool")
    
    parser.add_argument("--gw", type=int, help="Target gameweek (default: current)")
    parser.add_argument("--mode", choices=["frame", "saved", "team"], default="saved",
                       help="Preview mode: frame (prediction frame), saved (saved predictions), team (optimal team)")
    parser.add_argument("--players", type=int, default=20, help="Number of top players to show")
    parser.add_argument("--budget", type=float, default=100.0, help="Budget for team optimization")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    
    # Determine target gameweek
    target_gw = args.gw
    if target_gw is None:
        try:
            target_gw = get_current_gw()
            logger.info(f"Auto-detected current gameweek: {target_gw}")
        except Exception:
            logger.error("Could not auto-detect gameweek, please specify --gw")
            sys.exit(1)
    
    # Generate preview
    try:
        if args.mode == "frame":
            logger.info(f"Building prediction frame for GW {target_gw}...")
            df = build_prediction_frame(next_gw=target_gw)
            preview_text = preview_predictions_frame(df, args.players)
            
        elif args.mode == "saved":
            preview_text = preview_saved_predictions(target_gw)
            
        elif args.mode == "team":
            preview_text = preview_team_optimization(target_gw, args.budget)
            
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)
        
        print(preview_text)
        logger.info("Preview completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Preview interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
