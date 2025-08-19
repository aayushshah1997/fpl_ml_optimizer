"""
Main training and prediction pipeline for FPL AI.

Handles training models with staging, making predictions, and managing
the full ML pipeline from data loading to model output.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from typing import Optional, Dict, Any
from ..common.config import get_config, get_logger
from ..common.logging_setup import setup_logging
from ..common.timeutil import get_current_gw
from ..features.builder import build_training_table, build_prediction_frame
from ..modeling.minutes_model import train_minutes_model, predict_minutes
from ..modeling.model_lgbm import train_models, predict_points
from ..modeling.mc_sim import run_mc_simulation

logger = get_logger(__name__)


def detect_training_mode(current_gw: int) -> str:
    """
    Detect training mode based on current gameweek and configuration.
    
    Args:
        current_gw: Current gameweek
        
    Returns:
        Training mode: 'warm' or 'full'
    """
    config = get_config()
    staging_config = config.get("training", {}).get("staging", {})
    
    mode = staging_config.get("mode", "auto")
    warm_until_gw = staging_config.get("warm_until_gw", 8)
    
    if mode == "auto":
        detected_mode = "warm" if current_gw < warm_until_gw else "full"
        logger.info(f"Auto-detected training mode: {detected_mode} (GW {current_gw})")
        return detected_mode
    elif mode in ["warm", "full"]:
        logger.info(f"Using configured training mode: {mode}")
        return mode
    else:
        logger.warning(f"Unknown staging mode '{mode}', defaulting to 'warm'")
        return "warm"


def train_and_predict_pipeline(target_gw: int, force_retrain: bool = False) -> Dict[str, Any]:
    """
    Run full training and prediction pipeline.
    
    Args:
        target_gw: Gameweek to predict for
        force_retrain: Force retraining even if models exist
        
    Returns:
        Pipeline results
    """
    logger.info(f"Starting train_and_predict pipeline for GW {target_gw}")
    
    try:
        config = get_config()
        
        # Detect training mode
        training_mode = detect_training_mode(target_gw)
        
        # Check if models exist and retrain is not forced
        models_dir = Path(config.get("io", {}).get("out_dir", "artifacts")) / "models"
        models_exist = (
            (models_dir / "model_minutes.pkl").exists() and
            any(models_dir.glob("model_points_*.pkl"))
        )
        
        if models_exist and not force_retrain:
            logger.info("Models exist, skipping training. Use --force to retrain.")
            train_results = {"skipped": True}
        else:
            # Build training data
            logger.info("Building training data...")
            df_train = build_training_table(next_gw=target_gw)
            
            if df_train.empty:
                logger.error("No training data available")
                return {"error": "No training data"}
            
            logger.info(f"Training data built: {len(df_train):,} rows")
            
            # Train minutes model
            logger.info("Training minutes model...")
            minutes_results = train_minutes_model(df_train)
            
            # Train points models
            logger.info(f"Training points models ({training_mode} mode)...")
            points_results = train_models(df_train, mode=training_mode, current_gw=target_gw)
            
            train_results = {
                "training_mode": training_mode,
                "training_rows": len(df_train),
                "minutes_model": minutes_results,
                "points_models": points_results
            }
            
            logger.info("Training completed successfully")
        
        # Build prediction frame
        logger.info(f"Building prediction frame for GW {target_gw}...")
        df_pred = build_prediction_frame(next_gw=target_gw)
        
        if df_pred.empty:
            logger.error("No prediction data available")
            return {"error": "No prediction data", "train_results": train_results}
        
        logger.info(f"Prediction frame built: {len(df_pred):,} players")
        
        # Predict minutes
        logger.info("Predicting minutes...")
        minutes_pred = predict_minutes(df_pred)
        
        # Predict points
        logger.info("Predicting points...")
        points_pred = predict_points(df_pred)
        
        # Run Monte Carlo simulation
        logger.info("Running Monte Carlo simulation...")
        mc_results = run_mc_simulation(points_pred)
        
        # Save predictions
        artifacts_dir = Path(config.get("io", {}).get("out_dir", "artifacts"))
        artifacts_dir.mkdir(exist_ok=True)
        
        pred_file = artifacts_dir / f"predictions_gw{target_gw}.csv"
        mc_results.to_csv(pred_file, index=False)
        logger.info(f"Predictions saved to {pred_file}")
        
        return {
            "success": True,
            "target_gw": target_gw,
            "train_results": train_results,
            "prediction_results": {
                "players_predicted": len(mc_results),
                "minutes_model_used": minutes_pred is not None,
                "monte_carlo_scenarios": config.get("mc", {}).get("num_scenarios", 2000),
                "output_file": str(pred_file)
            },
            "predictions": mc_results
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return {"error": str(e)}


def predict_only_pipeline(target_gw: int) -> Dict[str, Any]:
    """
    Run prediction-only pipeline (no training).
    
    Args:
        target_gw: Gameweek to predict for
        
    Returns:
        Prediction results
    """
    logger.info(f"Starting predict-only pipeline for GW {target_gw}")
    
    try:
        config = get_config()
        
        # Check that models exist
        models_dir = Path(config.get("io", {}).get("out_dir", "artifacts")) / "models"
        
        required_models = ["model_minutes.pkl"]
        for position in ["GK", "DEF", "MID", "FWD"]:
            required_models.append(f"model_points_{position}.pkl")
        
        missing_models = []
        for model_file in required_models:
            if not (models_dir / model_file).exists():
                missing_models.append(model_file)
        
        if missing_models:
            logger.error(f"Missing required models: {missing_models}")
            return {"error": f"Missing models: {missing_models}"}
        
        # Build prediction frame
        logger.info(f"Building prediction frame for GW {target_gw}...")
        df_pred = build_prediction_frame(next_gw=target_gw)
        
        if df_pred.empty:
            logger.error("No prediction data available")
            return {"error": "No prediction data"}
        
        logger.info(f"Prediction frame built: {len(df_pred):,} players")
        
        # Predict minutes
        logger.info("Predicting minutes...")
        minutes_pred = predict_minutes(df_pred)
        
        # Predict points
        logger.info("Predicting points...")
        points_pred = predict_points(df_pred)
        
        # Run Monte Carlo simulation
        logger.info("Running Monte Carlo simulation...")
        mc_results = run_mc_simulation(points_pred)
        
        # Save predictions
        artifacts_dir = Path(config.get("io", {}).get("out_dir", "artifacts"))
        artifacts_dir.mkdir(exist_ok=True)
        
        pred_file = artifacts_dir / f"predictions_gw{target_gw}.csv"
        mc_results.to_csv(pred_file, index=False)
        logger.info(f"Predictions saved to {pred_file}")
        
        return {
            "success": True,
            "target_gw": target_gw,
            "prediction_results": {
                "players_predicted": len(mc_results),
                "minutes_model_used": minutes_pred is not None,
                "monte_carlo_scenarios": config.get("mc", {}).get("num_scenarios", 2000),
                "output_file": str(pred_file)
            },
            "predictions": mc_results
        }
        
    except Exception as e:
        logger.error(f"Prediction pipeline failed: {str(e)}", exc_info=True)
        return {"error": str(e)}


def wildcard_pipeline(target_gw: int) -> Dict[str, Any]:
    """
    Run wildcard optimization pipeline.
    
    Args:
        target_gw: Gameweek to optimize for
        
    Returns:
        Wildcard optimization results
    """
    logger.info(f"Starting wildcard pipeline for GW {target_gw}")
    
    try:
        # First run predictions
        pred_result = predict_only_pipeline(target_gw)
        
        if "error" in pred_result:
            return pred_result
        
        predictions = pred_result["predictions"]
        
        # Optimize wildcard team
        from ..optimize.optimizer import TeamOptimizer
        optimizer = TeamOptimizer()
        
        wc_result = optimizer.optimize_team(
            predictions,
            budget=100.0,
            objective="monte_carlo"
        )
        
        if not wc_result:
            return {"error": "Wildcard optimization failed"}
        
        # Optimize starting XI
        squad = wc_result.get("squad", [])
        xi_result = optimizer.optimize_starting_xi(
            squad, predictions, objective="monte_carlo"
        )
        
        return {
            "success": True,
            "target_gw": target_gw,
            "wildcard_squad": squad,
            "starting_xi": xi_result.get("starting_xi", []),
            "captain": xi_result.get("captain"),
            "vice_captain": xi_result.get("vice_captain"),
            "expected_points": wc_result.get("expected_points", 0),
            "total_cost": wc_result.get("total_cost", 0),
            "formation": xi_result.get("formation")
        }
        
    except Exception as e:
        logger.error(f"Wildcard pipeline failed: {str(e)}", exc_info=True)
        return {"error": str(e)}


def transfers_pipeline(target_gw: int, num_transfers: int = 1) -> Dict[str, Any]:
    """
    Run transfer optimization pipeline.
    
    Args:
        target_gw: Gameweek to optimize for
        num_transfers: Number of transfers to suggest
        
    Returns:
        Transfer optimization results
    """
    logger.info(f"Starting transfers pipeline for GW {target_gw}")
    
    try:
        # First run predictions
        pred_result = predict_only_pipeline(target_gw)
        
        if "error" in pred_result:
            return pred_result
        
        predictions = pred_result["predictions"]
        
        # Load current squad (would need to be implemented)
        # For now, return placeholder
        logger.warning("Transfer optimization not fully implemented")
        
        return {
            "success": True,
            "target_gw": target_gw,
            "message": "Transfer optimization placeholder - implement squad loading"
        }
        
    except Exception as e:
        logger.error(f"Transfers pipeline failed: {str(e)}", exc_info=True)
        return {"error": str(e)}


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="FPL AI Pipeline")
    
    parser.add_argument("--mode", choices=["train_and_predict", "predict", "wildcard", "transfers"], 
                       default="predict", help="Pipeline mode")
    parser.add_argument("--gw", type=int, help="Target gameweek (default: current)")
    parser.add_argument("--force", action="store_true", help="Force retrain models")
    parser.add_argument("--transfers", type=int, default=1, help="Number of transfers for transfers mode")
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
    
    # Run appropriate pipeline
    try:
        if args.mode == "train_and_predict":
            result = train_and_predict_pipeline(target_gw, force_retrain=args.force)
        elif args.mode == "predict":
            result = predict_only_pipeline(target_gw)
        elif args.mode == "wildcard":
            result = wildcard_pipeline(target_gw)
        elif args.mode == "transfers":
            result = transfers_pipeline(target_gw, args.transfers)
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)
        
        # Display results
        if "error" in result:
            logger.error(f"Pipeline failed: {result['error']}")
            sys.exit(1)
        else:
            logger.info("Pipeline completed successfully")
            
            # Print summary
            if args.mode in ["train_and_predict", "predict"]:
                pred_results = result.get("prediction_results", {})
                print(f"\n=== FPL AI Predictions GW {target_gw} ===")
                print(f"Players predicted: {pred_results.get('players_predicted', 0):,}")
                print(f"Output file: {pred_results.get('output_file', 'N/A')}")
                
                if "train_results" in result and not result["train_results"].get("skipped"):
                    train_results = result["train_results"]
                    print(f"Training mode: {train_results.get('training_mode', 'N/A')}")
                    print(f"Training rows: {train_results.get('training_rows', 0):,}")
            
            elif args.mode == "wildcard":
                print(f"\n=== Wildcard Team GW {target_gw} ===")
                print(f"Expected points: {result.get('expected_points', 0):.1f}")
                print(f"Total cost: £{result.get('total_cost', 0):.1f}M")
                print(f"Formation: {result.get('formation', 'N/A')}")
                
                captain = result.get("captain", {})
                if captain:
                    print(f"Captain: {captain.get('web_name', 'N/A')}")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
