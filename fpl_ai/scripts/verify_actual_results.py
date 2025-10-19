"""
Verify actual results data quality and compare with predictions.

This script validates the generated actual results files and provides
insights into predicted vs actual performance patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fpl_ai.src.common.config import get_logger

logger = get_logger(__name__)


def verify_actual_results(gw: int) -> dict:
    """Verify actual results for a specific gameweek."""
    actual_file = Path(f"fpl_ai/artifacts/gw{gw}_actual_results.csv")
    predictions_file = Path(f"fpl_ai/artifacts/predictions_gw{gw}.csv")
    
    if not actual_file.exists():
        return {"error": f"Actual results file not found for GW{gw}"}
    
    if not predictions_file.exists():
        return {"error": f"Predictions file not found for GW{gw}"}
    
    try:
        # Load data
        actual_df = pd.read_csv(actual_file)
        predictions_df = pd.read_csv(predictions_file)
        
        # Basic validation
        validation_results = {
            "gameweek": gw,
            "actual_players": len(actual_df),
            "predicted_players": len(predictions_df),
            "actual_points_stats": {
                "mean": actual_df['actual_points'].mean(),
                "std": actual_df['actual_points'].std(),
                "min": actual_df['actual_points'].min(),
                "max": actual_df['actual_points'].max(),
                "median": actual_df['actual_points'].median()
            },
            "data_quality_issues": []
        }
        
        # Check for data quality issues
        if actual_df['actual_points'].max() > 30:
            validation_results["data_quality_issues"].append("Suspiciously high points detected")
        
        if (actual_df['actual_points'] < 0).sum() > len(actual_df) * 0.1:
            validation_results["data_quality_issues"].append("Too many negative points")
        
        if actual_df['actual_minutes'].max() > 90:
            validation_results["data_quality_issues"].append("Minutes exceed 90")
        
        # Check for missing data
        missing_points = actual_df['actual_points'].isna().sum()
        if missing_points > 0:
            validation_results["data_quality_issues"].append(f"{missing_points} missing points")
        
        # Position-wise analysis
        position_stats = {}
        for position in ['GK', 'DEF', 'MID', 'FWD']:
            pos_data = actual_df[actual_df['position'] == position]
            if not pos_data.empty:
                position_stats[position] = {
                    "count": len(pos_data),
                    "avg_points": pos_data['actual_points'].mean(),
                    "max_points": pos_data['actual_points'].max()
                }
        
        validation_results["position_stats"] = position_stats
        
        # Compare with predictions if possible
        try:
            # Merge on element_id
            merged = actual_df.merge(
                predictions_df[['element_id', 'proj_points']],
                on='element_id',
                how='inner'
            )
            
            if not merged.empty:
                validation_results["comparison_stats"] = {
                    "matched_players": len(merged),
                    "predicted_avg": merged['proj_points'].mean(),
                    "actual_avg": merged['actual_points'].mean(),
                    "correlation": merged['proj_points'].corr(merged['actual_points']),
                    "mae": np.mean(np.abs(merged['proj_points'] - merged['actual_points'])),
                    "overprediction_ratio": merged['proj_points'].mean() / merged['actual_points'].mean() if merged['actual_points'].mean() > 0 else None
                }
        except Exception as e:
            validation_results["comparison_error"] = str(e)
        
        return validation_results
        
    except Exception as e:
        return {"error": f"Error processing GW{gw}: {e}"}


def verify_all_gameweeks():
    """Verify all gameweeks and generate summary report."""
    logger.info("Starting verification of all actual results files...")
    
    results = {}
    summary_stats = {
        "total_gameweeks": 0,
        "valid_gameweeks": 0,
        "total_players": 0,
        "avg_points_per_gw": [],
        "data_quality_issues": []
    }
    
    for gw in range(1, 8):
        logger.info(f"Verifying GW{gw}...")
        result = verify_actual_results(gw)
        results[gw] = result
        
        if "error" not in result:
            summary_stats["valid_gameweeks"] += 1
            summary_stats["total_players"] += result["actual_players"]
            summary_stats["avg_points_per_gw"].append(result["actual_points_stats"]["mean"])
            
            if result["data_quality_issues"]:
                summary_stats["data_quality_issues"].extend([f"GW{gw}: {issue}" for issue in result["data_quality_issues"]])
        
        summary_stats["total_gameweeks"] += 1
    
    # Generate summary
    logger.info("\n" + "="*60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total gameweeks processed: {summary_stats['total_gameweeks']}")
    logger.info(f"Valid gameweeks: {summary_stats['valid_gameweeks']}")
    logger.info(f"Total players: {summary_stats['total_players']}")
    
    if summary_stats["avg_points_per_gw"]:
        logger.info(f"Average points per gameweek: {np.mean(summary_stats['avg_points_per_gw']):.2f}")
        logger.info(f"Points range: {min(summary_stats['avg_points_per_gw']):.2f} - {max(summary_stats['avg_points_per_gw']):.2f}")
    
    if summary_stats["data_quality_issues"]:
        logger.info(f"\nData quality issues found: {len(summary_stats['data_quality_issues'])}")
        for issue in summary_stats["data_quality_issues"]:
            logger.warning(f"  - {issue}")
    else:
        logger.info("\nNo data quality issues detected!")
    
    # Detailed per-gameweek results
    logger.info("\n" + "="*60)
    logger.info("DETAILED RESULTS BY GAMEWEEK")
    logger.info("="*60)
    
    for gw, result in results.items():
        if "error" in result:
            logger.error(f"GW{gw}: {result['error']}")
        else:
            logger.info(f"GW{gw}: {result['actual_players']} players, "
                       f"avg points: {result['actual_points_stats']['mean']:.2f}, "
                       f"max: {result['actual_points_stats']['max']}")
            
            if "comparison_stats" in result:
                comp = result["comparison_stats"]
                logger.info(f"  Comparison: {comp['matched_players']} matched players, "
                           f"correlation: {comp['correlation']:.3f}, "
                           f"MAE: {comp['mae']:.2f}")
                
                if comp["overprediction_ratio"]:
                    ratio = comp["overprediction_ratio"]
                    if ratio > 1.1:
                        logger.warning(f"  ⚠️  Overprediction detected: {ratio:.2f}x")
                    elif ratio < 0.9:
                        logger.warning(f"  ⚠️  Underprediction detected: {ratio:.2f}x")
                    else:
                        logger.info(f"  ✅ Good calibration: {ratio:.2f}x")
    
    return results, summary_stats


def save_verification_report(results: dict, summary_stats: dict):
    """Save verification report to file."""
    import json
    from datetime import datetime
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": summary_stats,
        "detailed_results": results
    }
    
    report_file = Path("fpl_ai/artifacts/actual_results_verification_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Verification report saved to: {report_file}")


if __name__ == "__main__":
    results, summary_stats = verify_all_gameweeks()
    save_verification_report(results, summary_stats)
