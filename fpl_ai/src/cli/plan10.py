"""
10-week transfer planner CLI for FPL AI.

Command-line interface for the multi-week transfer planning system with
GW1 baseline initialization and horizon optimization.
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List, Optional
from ..common.config import get_config, get_logger
from ..common.logging_setup import setup_logging
from ..common.timeutil import get_current_gw
from ..plan.multiweek_planner import MultiWeekPlanner
from ..features.builder import build_prediction_frame

logger = get_logger(__name__)


def load_predictions_for_horizon(start_gw: int, horizon: int) -> Dict[int, pd.DataFrame]:
    """
    Load predictions for the planning horizon.
    
    Args:
        start_gw: Starting gameweek
        horizon: Number of gameweeks ahead
        
    Returns:
        Dictionary mapping gameweek to predictions DataFrame
    """
    config = get_config()
    artifacts_dir = Path(config.get("io", {}).get("out_dir", "artifacts"))
    
    predictions_by_gw = {}
    
    for gw in range(start_gw, start_gw + horizon + 1):
        pred_file = artifacts_dir / f"predictions_gw{gw}.csv"
        
        if pred_file.exists():
            try:
                df = pd.read_csv(pred_file)
                predictions_by_gw[gw] = df
                logger.debug(f"Loaded predictions for GW {gw}: {len(df)} players")
            except Exception as e:
                logger.warning(f"Could not load predictions for GW {gw}: {e}")
        else:
            logger.warning(f"No predictions file found for GW {gw}")
    
    logger.info(f"Loaded predictions for {len(predictions_by_gw)} gameweeks")
    return predictions_by_gw


def generate_predictions_for_horizon(start_gw: int, horizon: int) -> Dict[int, pd.DataFrame]:
    """
    Generate predictions for the planning horizon.
    
    Args:
        start_gw: Starting gameweek
        horizon: Number of gameweeks ahead
        
    Returns:
        Dictionary mapping gameweek to predictions DataFrame
    """
    logger.info(f"Generating predictions for GW {start_gw}-{start_gw + horizon}")
    
    predictions_by_gw = {}
    
    for gw in range(start_gw, start_gw + horizon + 1):
        try:
            logger.debug(f"Building prediction frame for GW {gw}")
            df = build_prediction_frame(next_gw=gw)
            
            if not df.empty:
                predictions_by_gw[gw] = df
                logger.debug(f"Generated predictions for GW {gw}: {len(df)} players")
            else:
                logger.warning(f"Empty prediction frame for GW {gw}")
        
        except Exception as e:
            logger.error(f"Failed to generate predictions for GW {gw}: {e}")
    
    logger.info(f"Generated predictions for {len(predictions_by_gw)} gameweeks")
    return predictions_by_gw


def format_transfer_plan(plan_result: Dict[str, Any]) -> str:
    """
    Format transfer plan for display.
    
    Args:
        plan_result: Transfer plan result
        
    Returns:
        Formatted plan string
    """
    if "error" in plan_result:
        return f"Error: {plan_result['error']}"
    
    steps = plan_result.get('steps', [])
    final_state = plan_result.get('final_state', {})
    analysis = plan_result.get('analysis', {})
    
    output = []
    output.append("=" * 80)
    output.append("FPL AI 10-WEEK TRANSFER PLAN")
    output.append("=" * 80)
    output.append("")
    
    # Plan summary
    metadata = plan_result.get('metadata', {})
    output.append(f"Planning Period: GW {metadata.get('start_gw', '?')}-{metadata.get('start_gw', 0) + len(steps) - 1}")
    output.append(f"Total Steps: {len(steps)}")
    
    # Analysis summary
    if analysis:
        value_analysis = analysis.get('value_analysis', {})
        transfer_summary = analysis.get('transfer_summary', {})
        
        output.append(f"Expected Total Value: {value_analysis.get('total_expected_points', 0):.1f} points")
        output.append(f"Total Hit Cost: {value_analysis.get('total_hit_cost', 0)} points")
        output.append(f"Net Expected Value: {value_analysis.get('net_expected_value', 0):.1f} points")
        output.append("")
        
        output.append(f"Transfers Planned: {transfer_summary.get('single_transfers', 0)} single, {transfer_summary.get('double_transfers', 0)} double")
        output.append(f"Total Hits: {transfer_summary.get('total_hits', 0)}")
        output.append(f"Chips Used: {', '.join(transfer_summary.get('chips_used', [])) or 'None'}")
    
    output.append("")
    output.append("WEEKLY PLAN:")
    output.append("-" * 50)
    
    # Weekly steps
    for step in steps:
        gw = step.get('gameweek', '?')
        action = step.get('action', {})
        score = step.get('score', 0)
        gw_value = step.get('gw_value', 0)
        
        action_type = action.get('action', 'unknown')
        
        if action_type == 'roll':
            output.append(f"GW {gw}: ROLL FT (Expected: {gw_value:.1f} pts)")
        
        elif action_type == 'single_transfer':
            transfers = action.get('transfers', [])
            if transfers:
                transfer = transfers[0]
                out_name = transfer.get('out', {}).get('web_name', 'Unknown')
                in_name = transfer.get('in', {}).get('web_name', 'Unknown')
                points_gain = transfer.get('points_gain', 0)
                cost = transfer.get('cost', 0)
                hits = action.get('hits', 0)
                
                hit_text = f" (Hit: -{hits * 4})" if hits > 0 else ""
                output.append(f"GW {gw}: {out_name} → {in_name} (+{points_gain:.1f}, £{cost:+.1f}M){hit_text}")
        
        elif action_type == 'double_transfer':
            transfers = action.get('transfers', [])
            if len(transfers) >= 2:
                t1, t2 = transfers[0], transfers[1]
                out1_name = t1.get('out', {}).get('web_name', 'Unknown')
                in1_name = t1.get('in', {}).get('web_name', 'Unknown')
                out2_name = t2.get('out', {}).get('web_name', 'Unknown')
                in2_name = t2.get('in', {}).get('web_name', 'Unknown')
                total_gain = sum(t.get('points_gain', 0) for t in transfers)
                hits = action.get('hits', 0)
                
                hit_text = f" (Hit: -{hits * 4})" if hits > 0 else ""
                output.append(f"GW {gw}: {out1_name} → {in1_name}, {out2_name} → {in2_name} (+{total_gain:.1f}){hit_text}")
        
        elif action_type in ['free_hit', 'wildcard', 'triple_captain', 'bench_boost']:
            chip_value = action.get('chip_value', 0)
            output.append(f"GW {gw}: {action_type.upper().replace('_', ' ')} (+{chip_value:.1f} pts)")
        
        else:
            output.append(f"GW {gw}: {action_type} (Score: {score:.1f})")
    
    output.append("")
    
    # Final state summary
    if final_state:
        final_squad = final_state.get('squad', [])
        final_bank = final_state.get('bank', 0)
        final_fts = final_state.get('free_transfers', 0)
        
        output.append("FINAL STATE:")
        output.append("-" * 15)
        output.append(f"Squad Size: {len(final_squad)} players")
        output.append(f"Bank: £{final_bank:.1f}M")
        output.append(f"Free Transfers: {final_fts}")
        
        # Top final squad players
        if final_squad and len(final_squad) > 0:
            # Sort by projection if available
            squad_with_proj = []
            for player in final_squad:
                proj = player.get('proj_points', 0)
                squad_with_proj.append((proj, player))
            
            squad_with_proj.sort(reverse=True)
            
            output.append("\nTop Final Squad Players:")
            for i, (proj, player) in enumerate(squad_with_proj[:5]):
                name = player.get('web_name', 'Unknown')
                position = player.get('position', '?')
                cost = player.get('now_cost', 0)
                if cost > 20:
                    cost /= 10
                output.append(f"  {i+1}. {name} ({position}) - £{cost:.1f}M, {proj:.1f} pts")
    
    output.append("")
    output.append("=" * 80)
    
    return "\n".join(output)


def save_plan_json(plan_result: Dict[str, Any], output_file: Path) -> None:
    """
    Save transfer plan to JSON file.
    
    Args:
        plan_result: Transfer plan result
        output_file: Output file path
    """
    try:
        with open(output_file, 'w') as f:
            json.dump(plan_result, f, indent=2, default=str)
        
        logger.info(f"Transfer plan saved to {output_file}")
    
    except Exception as e:
        logger.error(f"Failed to save plan to {output_file}: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="FPL AI 10-Week Transfer Planner")
    
    # Planning parameters
    parser.add_argument("--gw", type=int, help="Starting gameweek (default: current)")
    parser.add_argument("--horizon", type=int, default=10, help="Planning horizon in gameweeks")
    parser.add_argument("--bank", type=float, default=0.0, help="Current bank balance")
    parser.add_argument("--fts", type=int, default=1, help="Current free transfers")
    
    # Team initialization
    parser.add_argument("--entry-id", type=int, help="FPL Entry ID for team initialization")
    parser.add_argument("--squad-csv", help="CSV file with current squad")
    parser.add_argument("--squad", help="Comma-separated list of element IDs")
    
    # Predictions
    parser.add_argument("--use-saved", action="store_true", 
                       help="Use saved predictions instead of generating new ones")
    
    # Output
    parser.add_argument("--output", "-o", help="Output JSON file for detailed plan")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    
    # Determine starting gameweek
    start_gw = args.gw
    if start_gw is None:
        try:
            start_gw = get_current_gw()
            logger.info(f"Auto-detected current gameweek: {start_gw}")
        except Exception:
            logger.error("Could not auto-detect gameweek, please specify --gw")
            sys.exit(1)
    
    try:
        # Initialize planner
        planner = MultiWeekPlanner()
        
        # Initialize team state
        team_state = None
        
        if args.entry_id:
            logger.info(f"Initializing team from FPL Entry ID: {args.entry_id}")
            team_state = planner.initialize_from_gw1_team(
                entry_id=args.entry_id, 
                bank=args.bank
            )
        elif args.squad_csv:
            logger.info(f"Loading team from CSV: {args.squad_csv}")
            team_state = planner.initialize_from_gw1_team(
                csv_path=args.squad_csv, 
                bank=args.bank
            )
        elif args.squad:
            # Parse squad element IDs
            try:
                element_ids = [int(x.strip()) for x in args.squad.split(',')]
                squad_data = [{'element_id': eid} for eid in element_ids]
                
                logger.info(f"Using provided squad: {len(element_ids)} players")
                team_state = planner.initialize_from_gw1_team(
                    gw1_team=squad_data, 
                    bank=args.bank
                )
            except ValueError:
                logger.error("Invalid squad format. Use comma-separated element IDs.")
                sys.exit(1)
        
        if not team_state:
            logger.warning("No team state provided. Will try to load from cache.")
        
        # Apply current state overrides
        state_overrides = {
            'gameweek': start_gw,
            'bank': args.bank,
            'free_transfers': args.fts
        }
        
        # Load or generate predictions
        if args.use_saved:
            logger.info("Loading saved predictions...")
            predictions_by_gw = load_predictions_for_horizon(start_gw, args.horizon)
        else:
            logger.info("Generating fresh predictions...")
            predictions_by_gw = generate_predictions_for_horizon(start_gw, args.horizon)
        
        if not predictions_by_gw:
            logger.error("No predictions available for planning")
            sys.exit(1)
        
        logger.info(f"Predictions available for {len(predictions_by_gw)} gameweeks")
        
        # Run transfer planning
        logger.info(f"Running {args.horizon}-week transfer plan from GW {start_gw}")
        
        plan_result = planner.plan_transfers(
            start_gw=start_gw,
            predictions_by_gw=predictions_by_gw,
            current_state=team_state,
            overrides=state_overrides
        )
        
        if "error" in plan_result:
            logger.error(f"Planning failed: {plan_result['error']}")
            sys.exit(1)
        
        # Display results
        plan_text = format_transfer_plan(plan_result)
        print(plan_text)
        
        # Save detailed results if requested
        if args.output:
            output_path = Path(args.output)
            save_plan_json(plan_result, output_path)
        
        logger.info("Transfer planning completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Planning interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
