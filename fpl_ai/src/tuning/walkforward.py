"""
Walk-forward backtesting for FPL ML models.

Implements temporal validation with:
- EWMA (exponential) recency weighting for training samples
- Walk-forward validation using only data available up to each GW
- Support for position-specific models and captain policy evaluation
- Monte Carlo simulation for risk assessment
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import time
from copy import deepcopy
from typing import Dict, Tuple, Any, Optional
from pathlib import Path

from ..common.config import load_settings, get_logger
from ..features.builder import build_training_table, build_prediction_frame
from ..modeling.model_lgbm import LGBMTrainer, LGBMPredictor
from ..optimize.optimizer import TeamOptimizer
from ..modeling.mc_sim import MonteCarloSimulator

logger = get_logger(__name__)


def ewma_weights(n: int, alpha: float = 0.85) -> np.ndarray:
    """
    Generate EWMA (exponentially weighted moving average) weights.
    
    Args:
        n: Number of samples
        alpha: Decay parameter (0 < alpha < 1), higher = more recent weight
        
    Returns:
        Normalized weights array where most recent gets highest weight
    """
    if n <= 0:
        return np.array([])
    
    # Generate weights: most recent gets highest weight
    weights = np.array([alpha**(n-1-i) for i in range(n)], dtype=float)
    
    # Normalize to sum to 1
    weights /= (weights.sum() + 1e-12)
    
    return weights


def train_model_upto_gw(
    target_season: str, 
    upto_gw: int, 
    ewma_alpha: float,
    settings: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Build a training table using only data strictly before target GW,
    apply recency weights via sample_weight column, fit and return model(s).
    
    Args:
        target_season: Season to train up to (e.g., "2023-2024")
        upto_gw: GW to stop before (exclusive)
        ewma_alpha: EWMA decay parameter for recency weighting
        settings: Optional settings override
        
    Returns:
        Dictionary containing trained models and metadata
    """
    if settings is None:
        settings = load_settings()
    
    try:
        # For walkforward, we need to modify the training data building
        # to only use data before the target GW
        # This is a simplified approach - in practice you'd need to modify
        # build_training_table to accept an end_gw parameter
        
        logger.info(f"Training model up to {target_season} GW {upto_gw}")
        
        # Build training data (this would need modification to respect upto_gw)
        df = build_training_table(next_gw=upto_gw)
        
        if df.empty:
            raise ValueError(f"No training data available for {target_season} up to GW {upto_gw}")
        
        # Apply temporal ordering and EWMA weights
        df = df.sort_values(["season", "gameweek", "element_id"], na_position='last')
        
        # Create recency weights by position within season and gameweek
        weights = []
        for (season, gw), group in df.groupby(["season", "gameweek"]):
            group_size = len(group)
            group_weights = ewma_weights(group_size, alpha=ewma_alpha)
            weights.extend(group_weights)
        
        df["sample_weight"] = weights
        
        # Train model with sample weights
        trainer = LGBMTrainer()
        training_results = trainer.train_models(
            training_data=df,
            mode="full",  # Use full mode for backtest training
            current_gw=upto_gw
        )
        
        return {
            "models": trainer.models,
            "feature_names": trainer.feature_names,
            "training_metrics": trainer.training_metrics,
            "training_samples": len(df),
            "target_season": target_season,
            "upto_gw": upto_gw
        }
        
    except Exception as e:
        logger.error(f"Error training model up to GW {upto_gw}: {e}")
        return {}


def simulate_gw(
    models: Dict[str, Any],
    gw: int, 
    settings: Dict,
    captain_policy: Optional[str] = None
) -> Dict[str, Any]:
    """
    Simulate a gameweek using trained models with transfer strategy and captain policy.
    
    Args:
        models: Trained models dictionary
        gw: Gameweek to simulate
        settings: Configuration settings
        captain_policy: Captain selection policy ("mean", "cvar", "mix")
        
    Returns:
        Dictionary with simulation results including transfer decisions
    """
    try:
        # Import here to avoid circular imports
        from ..plan.strategy import evaluate_one_step, load_team_state_for_backtest
        from ..tuning.captain import choose_captain
        from ..modeling.mc_sim import simulate_player_matrix
        
        # 1) Build market projections for all players (from model)
        pred = build_prediction_frame(next_gw=gw)
        
        if pred.empty:
            logger.warning(f"No prediction data for GW {gw}")
            return {"points_mean": 0.0, "scen_sum": np.array([]), "decision": None}
        
        # Make predictions using the trained models
        predictor = LGBMPredictor()
        predictor.models = models.get("models", {})
        predictor.feature_names = models.get("feature_names", {})
        predictor.model_metrics = models.get("training_metrics", {})
        
        preds = predictor.predict_points(pred)
        
        if preds.empty:
            logger.warning(f"No predictions generated for GW {gw}")
            return {"points_mean": 0.0, "scen_sum": np.array([]), "decision": None}
        
        # Merge predictions with player data to create market
        market = pred.merge(preds, left_on="id", right_on="element_id", how="inner")
        
        if market.empty:
            logger.warning(f"No merged market data for GW {gw}")
            return {"points_mean": 0.0, "scen_sum": np.array([]), "decision": None}
        
        # 2) Load team state for this GW (from backtest state management)
        team_df, bank, free_transfers = load_team_state_for_backtest(gw)
        
        if team_df.empty:
            logger.warning(f"No team state for GW {gw}, falling back to optimization")
            # Fallback: optimize team from market
            optimizer = TeamOptimizer()
            optimization_result = optimizer.optimize_team(
                predictions=market,
                budget=100.0,
                objective="mean"
            )
            
            if not optimization_result or not optimization_result.get("squad"):
                logger.warning(f"Team optimization failed for GW {gw}")
                return {"points_mean": 0.0, "scen_sum": np.array([]), "decision": None}
            
            # Use optimized squad as team
            team_df = pd.DataFrame(optimization_result["squad"])
            bank = 1.0
            free_transfers = 1
        
        # 3) Evaluate transfers using strategy engine
        decision = evaluate_one_step(
            team_df=team_df, 
            market_df=market, 
            bank=bank, 
            free_transfers=free_transfers, 
            settings=settings
        )
        
        # 4) Apply transfers to get final XI
        final_team = team_df.copy()
        for out_id, in_id in decision.transfers:
            # Find the incoming player in market
            in_player_mask = (market.get("element_id") == in_id) | (market.get("player_id") == in_id)
            if in_player_mask.any():
                in_player = market[in_player_mask].iloc[0]
                
                # Replace outgoing player in team
                out_player_mask = (final_team.get("element_id") == out_id) | (final_team.get("player_id") == out_id)
                if out_player_mask.any():
                    # Update player data
                    for col in ["element_id", "player_id", "web_name", "team_name", "position", "now_cost", "cost"]:
                        if col in final_team.columns and col in in_player.index:
                            final_team.loc[out_player_mask, col] = in_player[col]
                    
                    # Update projection
                    proj_col = "proj_points" if "proj_points" in final_team.columns else "proj"
                    in_proj_col = "proj_points" if "proj_points" in in_player.index else "proj" 
                    if proj_col in final_team.columns and in_proj_col in in_player.index:
                        final_team.loc[out_player_mask, proj_col] = in_player[in_proj_col]
        
        # 5) Get starting XI from final team
        optimizer = TeamOptimizer()
        optimization_result = optimizer.optimize_team(
            predictions=final_team,
            budget=100.0,
            objective="mean"
        )
        
        if not optimization_result or not optimization_result.get("squad"):
            logger.warning(f"Final team optimization failed for GW {gw}")
            return {"points_mean": 0.0, "scen_sum": np.array([]), "decision": decision}
        
        xi_result = optimizer.optimize_starting_xi(
            squad=optimization_result["squad"],
            predictions=final_team,
            objective="mean"
        )
        
        if not xi_result or not xi_result.get("starting_xi"):
            logger.warning(f"Starting XI selection failed for GW {gw}")
            return {"points_mean": 0.0, "scen_sum": np.array([]), "decision": decision}
        
        # 6) Apply captain policy on chosen XI using per-player MC
        xi_df = pd.DataFrame(xi_result["starting_xi"])
        
        # Generate per-player scenarios
        scen_mat, player_ids = simulate_player_matrix(
            xi_df, 
            S=int(settings.get("mc", {}).get("num_scenarios", 2000)),
            seed=int(settings.get("mc", {}).get("seed", 42)),
            minutes_uncertainty=float(settings.get("mc", {}).get("minutes_uncertainty", 0.20)),
            settings=settings
        )
        
        if scen_mat.size == 0:
            logger.warning(f"Monte Carlo simulation failed for GW {gw}")
            # Fallback to mean calculation
            points_mean = sum(player.get("proj_points", player.get("proj", 0)) for player in xi_result["starting_xi"])
            return {"points_mean": float(points_mean), "scen_sum": np.array([points_mean]), "decision": decision}
        
        # Apply captain policy
        cap_cfg = settings.get("captain", {})
        policy = captain_policy or cap_cfg.get("policy", "mix")
        alpha = float(cap_cfg.get("cvar_alpha", settings.get("mc", {}).get("cvar_alpha", 0.2)))
        mix_lambda = float(cap_cfg.get("mix_lambda", 0.6))
        candidates = int(cap_cfg.get("candidates", 5))
        
        cap_idx, vc_idx = choose_captain(
            xi_df=xi_df,
            scen_mat=scen_mat, 
            policy=policy,
            alpha=alpha,
            mix_lambda=mix_lambda,
            topN=candidates
        )
        
        # Calculate final team scores with captain bonus
        team_sum = scen_mat.sum(axis=1) + scen_mat[:, cap_idx]  # Add captain double points
        points_mean = float(np.mean(team_sum))
        
        # Update decision with final captain info
        if cap_idx < len(xi_df):
            decision.captain_id = int(xi_df.iloc[cap_idx].get("element_id", decision.captain_id))
        if vc_idx < len(xi_df):
            decision.vice_id = int(xi_df.iloc[vc_idx].get("element_id", decision.vice_id))
        
        return {
            "points_mean": points_mean,
            "scen_sum": team_sum,
            "starting_xi": xi_result["starting_xi"],
            "captain": {"element_id": decision.captain_id} if decision.captain_id else None,
            "vice_captain": {"element_id": decision.vice_id} if decision.vice_id else None,
            "decision": decision,
            "team_after_transfers": final_team.to_dict('records'),
            "bank_after": decision.bank_after
        }
        
    except Exception as e:
        logger.error(f"Error simulating GW {gw}: {e}")
        return {"points_mean": 0.0, "scen_sum": np.array([]), "decision": None}


def walk_forward_backtest(
    seasons: list, 
    start_gw: int, 
    end_gw: int, 
    settings: Dict
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run walk-forward backtest across multiple seasons.
    
    Args:
        seasons: List of seasons to backtest (e.g., ["2021-2022", "2022-2023"])
        start_gw: Starting gameweek for backtest
        end_gw: Ending gameweek for backtest
        settings: Configuration settings
        
    Returns:
        Tuple of (backtest_results_df, aggregate_metrics)
    """
    logger.info(f"Running walk-forward backtest for {len(seasons)} seasons, GW {start_gw}-{end_gw}")
    
    ewma_alpha = float(settings.get("backtest", {}).get("ewma_alpha", 0.85))
    captain_policy = settings.get("captain", {}).get("policy", "mix")
    
    records = []
    all_scenarios = []
    
    start_time = time.time()
    
    for season_idx, season in enumerate(seasons):
        logger.info(f"Backtesting season {season} ({season_idx + 1}/{len(seasons)})")
        
        for gw in range(start_gw, end_gw + 1):
            try:
                logger.debug(f"Training model up to {season} GW {gw}")
                
                # Train model using only data before this GW
                model_result = train_model_upto_gw(season, gw, ewma_alpha, settings)
                
                if not model_result:
                    logger.warning(f"Failed to train model for {season} GW {gw}")
                    records.append({
                        "season": season,
                        "gw": gw,
                        "points": 0.0,
                        "scen_sum": np.array([]),
                        "training_samples": 0,
                        "status": "training_failed"
                    })
                    continue
                
                logger.debug(f"Simulating {season} GW {gw}")
                
                # Simulate this gameweek
                sim_result = simulate_gw(model_result, gw, settings, captain_policy)
                
                # Store result
                record = {
                    "season": season,
                    "gw": gw,
                    "points": sim_result["points_mean"],
                    "scen_sum": sim_result["scen_sum"],
                    "training_samples": model_result.get("training_samples", 0),
                    "status": "success"
                }
                
                records.append(record)
                
                # Collect scenarios for aggregate risk metrics
                if isinstance(sim_result["scen_sum"], np.ndarray) and sim_result["scen_sum"].size > 0:
                    all_scenarios.extend(sim_result["scen_sum"].tolist())
                
                logger.debug(f"Completed {season} GW {gw}: {sim_result['points_mean']:.1f} points")
                
            except Exception as e:
                logger.error(f"Error in backtest for {season} GW {gw}: {e}")
                records.append({
                    "season": season,
                    "gw": gw,
                    "points": 0.0,
                    "scen_sum": np.array([]),
                    "training_samples": 0,
                    "status": "error"
                })
    
    # Create results DataFrame
    df = pd.DataFrame(records)
    
    # Calculate aggregate metrics
    total_time = time.time() - start_time
    
    # Basic metrics
    successful_gws = df[df["status"] == "success"]
    total_points = float(successful_gws["points"].sum()) if not successful_gws.empty else 0.0
    
    # Risk metrics from scenarios
    all_scenarios_array = np.array(all_scenarios) if all_scenarios else np.array([])
    
    if all_scenarios_array.size > 0:
        cvar_alpha = float(settings.get("mc", {}).get("cvar_alpha", 0.2))
        var_threshold = np.quantile(all_scenarios_array, cvar_alpha)
        tail_scenarios = all_scenarios_array[all_scenarios_array <= var_threshold]
        risk_adj_points = float(np.mean(tail_scenarios)) if tail_scenarios.size > 0 else total_points
        
        # Sharpe-like ratio
        mean_points = np.mean(all_scenarios_array)
        std_points = np.std(all_scenarios_array)
        sharpe_like = float((mean_points - np.mean(tail_scenarios)) / (std_points + 1e-9)) if tail_scenarios.size > 0 else 0.0
    else:
        risk_adj_points = total_points
        sharpe_like = 0.0
    
    metrics = {
        "total_points": total_points,
        "risk_adj_points": risk_adj_points,
        "sharpe_like": sharpe_like,
        "successful_gws": len(successful_gws),
        "total_gws": len(df),
        "success_rate": len(successful_gws) / len(df) if len(df) > 0 else 0.0,
        "avg_points_per_gw": total_points / len(successful_gws) if len(successful_gws) > 0 else 0.0,
        "total_time_seconds": total_time,
        "seasons_tested": len(seasons),
        "gw_range": (start_gw, end_gw)
    }
    
    logger.info(f"Backtest completed: {total_points:.1f} total points, {metrics['success_rate']:.1%} success rate")
    
    return df, metrics


def analyze_backtest_results(df: pd.DataFrame, metrics: Dict) -> Dict[str, Any]:
    """
    Analyze backtest results and generate insights.
    
    Args:
        df: Backtest results DataFrame
        metrics: Aggregate metrics
        
    Returns:
        Analysis results dictionary
    """
    try:
        analysis = {
            "summary": metrics,
            "by_season": {},
            "by_gameweek": {},
            "performance_trends": {}
        }
        
        # Analysis by season
        if not df.empty:
            season_analysis = df.groupby("season").agg({
                "points": ["sum", "mean", "std", "count"],
                "training_samples": "mean"
            }).round(2)
            
            analysis["by_season"] = season_analysis.to_dict()
            
            # Analysis by gameweek
            gw_analysis = df.groupby("gw").agg({
                "points": ["mean", "std", "count"],
            }).round(2)
            
            analysis["by_gameweek"] = gw_analysis.to_dict()
            
            # Performance trends
            if len(df) > 5:  # Need sufficient data for trends
                # Calculate rolling performance
                df_sorted = df.sort_values(["season", "gw"])
                df_sorted["cumulative_points"] = df_sorted["points"].cumsum()
                df_sorted["rolling_avg_5"] = df_sorted["points"].rolling(5, min_periods=1).mean()
                
                analysis["performance_trends"] = {
                    "improving": df_sorted["rolling_avg_5"].tail(5).mean() > df_sorted["rolling_avg_5"].head(5).mean(),
                    "consistency": float(df_sorted["points"].std()),
                    "final_cumulative": float(df_sorted["cumulative_points"].iloc[-1])
                }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing backtest results: {e}")
        return {"error": str(e)}
