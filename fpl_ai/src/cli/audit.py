"""
Audit and analysis tools for FPL AI.

Provides feature coverage analysis, data quality checks, and model diagnostics.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from ..common.config import get_config, get_logger
from ..common.logging_setup import setup_logging
from ..common.timeutil import get_current_gw
from ..features.builder import build_prediction_frame
from ..providers.league_strength import get_new_leagues_report

logger = get_logger(__name__)


def analyze_feature_coverage(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze feature coverage and data quality in prediction frame.
    
    Args:
        df: Prediction DataFrame
        
    Returns:
        Feature coverage analysis
    """
    if df.empty:
        return {"error": "Empty DataFrame"}
    
    total_players = len(df)
    coverage_analysis = {}
    
    # Basic coverage
    coverage_analysis["total_players"] = total_players
    coverage_analysis["total_features"] = len(df.columns)
    
    # Feature categories for analysis
    feature_categories = {
        "basic": ["element_id", "web_name", "position", "team_name", "now_cost"],
        "projections": ["proj_points", "proj_raw", "expected_minutes"],
        "recent_form": [col for col in df.columns if col.startswith("r3_") or col.startswith("r5_")],
        "attacking": [col for col in df.columns if any(x in col for x in ["goals", "assists", "shots", "xG", "xA"])],
        "defensive": [col for col in df.columns if any(x in col for x in ["clean_sheets", "goals_conceded", "saves"])],
        "creativity": [col for col in df.columns if any(x in col for x in ["key_pass", "big_chance"])],
        "penalty": [col for col in df.columns if "pen_" in col],
        "set_pieces": [col for col in df.columns if any(x in col for x in ["fk_", "corner_"])],
        "team_form": [col for col in df.columns if "team_" in col and "form" in col],
        "opponent": [col for col in df.columns if "opp_" in col],
        "odds": [col for col in df.columns if any(x in col for x in ["odds", "prob"])],
        "market": [col for col in df.columns if any(x in col for x in ["selected", "price", "ownership"])],
        "fixtures": [col for col in df.columns if any(x in col for x in ["difficulty", "home", "away"])],
        "availability": [col for col in df.columns if any(x in col for x in ["avail", "injury", "status"])]
    }
    
    # Analyze coverage by category
    category_coverage = {}
    for category, features in feature_categories.items():
        available_features = [f for f in features if f in df.columns]
        if available_features:
            # Calculate missing data percentage for each feature
            missing_data = {}
            for feature in available_features:
                missing_pct = (df[feature].isna().sum() / total_players) * 100
                missing_data[feature] = missing_pct
            
            category_coverage[category] = {
                "features_available": len(available_features),
                "features_total": len(features),
                "coverage_pct": (len(available_features) / len(features)) * 100,
                "missing_data": missing_data,
                "avg_missing_pct": np.mean(list(missing_data.values()))
            }
        else:
            category_coverage[category] = {
                "features_available": 0,
                "features_total": len(features),
                "coverage_pct": 0,
                "missing_data": {},
                "avg_missing_pct": 100
            }
    
    coverage_analysis["category_coverage"] = category_coverage
    
    # Position-specific analysis
    position_analysis = {}
    for position in ["GK", "DEF", "MID", "FWD"]:
        pos_df = df[df["position"] == position] if "position" in df.columns else pd.DataFrame()
        
        if not pos_df.empty:
            position_analysis[position] = {
                "player_count": len(pos_df),
                "avg_projection": pos_df.get("proj_points", pd.Series([0])).mean(),
                "projection_std": pos_df.get("proj_points", pd.Series([0])).std(),
                "cost_range": {
                    "min": pos_df.get("now_cost", pd.Series([0])).min(),
                    "max": pos_df.get("now_cost", pd.Series([0])).max(),
                    "avg": pos_df.get("now_cost", pd.Series([0])).mean()
                }
            }
    
    coverage_analysis["position_analysis"] = position_analysis
    
    # Data quality issues
    quality_issues = []
    
    # Check for missing essential data
    essential_cols = ["element_id", "position", "proj_points"]
    for col in essential_cols:
        if col not in df.columns:
            quality_issues.append(f"Missing essential column: {col}")
        elif df[col].isna().any():
            missing_count = df[col].isna().sum()
            quality_issues.append(f"Missing values in {col}: {missing_count} players")
    
    # Check for suspicious values
    if "proj_points" in df.columns:
        negative_proj = (df["proj_points"] < 0).sum()
        if negative_proj > 0:
            quality_issues.append(f"Negative projections: {negative_proj} players")
        
        extreme_proj = (df["proj_points"] > 20).sum()
        if extreme_proj > 0:
            quality_issues.append(f"Extremely high projections (>20): {extreme_proj} players")
    
    if "now_cost" in df.columns:
        zero_cost = (df["now_cost"] <= 0).sum()
        if zero_cost > 0:
            quality_issues.append(f"Zero/negative cost players: {zero_cost}")
    
    coverage_analysis["quality_issues"] = quality_issues
    
    return coverage_analysis


def analyze_model_performance() -> Dict[str, Any]:
    """
    Analyze model performance from saved metrics.
    
    Returns:
        Model performance analysis
    """
    config = get_config()
    models_dir = Path(config.get("io", {}).get("out_dir", "artifacts")) / "models"
    
    performance_analysis = {}
    
    # Check for CV results
    positions = ["GK", "DEF", "MID", "FWD"]
    
    for position in positions:
        cv_file = models_dir / f"cv_{position}.json"
        
        if cv_file.exists():
            try:
                import json
                with open(cv_file, 'r') as f:
                    cv_data = json.load(f)
                
                performance_analysis[position] = cv_data
            except Exception as e:
                logger.warning(f"Could not load CV data for {position}: {e}")
    
    # Load feature importance
    feature_importance = {}
    for position in positions:
        fi_file = models_dir / f"fi_{position}.csv"
        
        if fi_file.exists():
            try:
                fi_df = pd.read_csv(fi_file)
                if not fi_df.empty:
                    # Top 10 features
                    top_features = fi_df.head(10).to_dict('records')
                    feature_importance[position] = top_features
            except Exception as e:
                logger.warning(f"Could not load feature importance for {position}: {e}")
    
    performance_analysis["feature_importance"] = feature_importance
    
    # Load residual analysis
    residual_analysis = {}
    for position in positions:
        residuals_file = models_dir / f"residuals_{position}.csv"
        
        if residuals_file.exists():
            try:
                residuals_df = pd.read_csv(residuals_file)
                if not residuals_df.empty and "residual" in residuals_df.columns:
                    residual_analysis[position] = {
                        "mean_absolute_residual": residuals_df["residual"].abs().mean(),
                        "std_residual": residuals_df["residual"].std(),
                        "residual_range": {
                            "min": residuals_df["residual"].min(),
                            "max": residuals_df["residual"].max()
                        }
                    }
            except Exception as e:
                logger.warning(f"Could not load residuals for {position}: {e}")
    
    performance_analysis["residual_analysis"] = residual_analysis
    
    return performance_analysis


def generate_data_quality_report(df: pd.DataFrame) -> str:
    """
    Generate human-readable data quality report.
    
    Args:
        df: Prediction DataFrame
        
    Returns:
        Formatted report string
    """
    analysis = analyze_feature_coverage(df)
    
    if "error" in analysis:
        return f"Error: {analysis['error']}"
    
    report = []
    report.append("=" * 60)
    report.append("FPL AI DATA QUALITY REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Overview
    report.append(f"Total Players: {analysis['total_players']:,}")
    report.append(f"Total Features: {analysis['total_features']:,}")
    report.append("")
    
    # Position breakdown
    report.append("POSITION BREAKDOWN:")
    report.append("-" * 20)
    position_analysis = analysis.get("position_analysis", {})
    for position, data in position_analysis.items():
        report.append(f"{position}: {data['player_count']} players, "
                     f"avg projection: {data['avg_projection']:.1f}")
    report.append("")
    
    # Feature coverage by category
    report.append("FEATURE COVERAGE BY CATEGORY:")
    report.append("-" * 35)
    category_coverage = analysis.get("category_coverage", {})
    
    for category, data in category_coverage.items():
        coverage_pct = data["coverage_pct"]
        missing_pct = data["avg_missing_pct"]
        
        status = "✓" if coverage_pct > 80 and missing_pct < 20 else "⚠" if coverage_pct > 50 else "✗"
        
        report.append(f"{status} {category.upper()}: "
                     f"{data['features_available']}/{data['features_total']} features "
                     f"({coverage_pct:.0f}% coverage, {missing_pct:.1f}% missing)")
    
    report.append("")
    
    # Quality issues
    quality_issues = analysis.get("quality_issues", [])
    if quality_issues:
        report.append("DATA QUALITY ISSUES:")
        report.append("-" * 20)
        for issue in quality_issues:
            report.append(f"⚠ {issue}")
    else:
        report.append("✓ No major data quality issues detected")
    
    report.append("")
    report.append("=" * 60)
    
    return "\n".join(report)


def generate_model_performance_report() -> str:
    """
    Generate human-readable model performance report.
    
    Returns:
        Formatted report string
    """
    analysis = analyze_model_performance()
    
    report = []
    report.append("=" * 60)
    report.append("FPL AI MODEL PERFORMANCE REPORT")
    report.append("=" * 60)
    report.append("")
    
    # CV Performance
    positions_with_cv = [pos for pos in ["GK", "DEF", "MID", "FWD"] if pos in analysis]
    
    if positions_with_cv:
        report.append("CROSS-VALIDATION PERFORMANCE:")
        report.append("-" * 30)
        
        for position in positions_with_cv:
            cv_data = analysis[position]
            if isinstance(cv_data, dict) and "mean_mae" in cv_data:
                report.append(f"{position}: MAE {cv_data['mean_mae']:.3f} ± {cv_data.get('std_mae', 0):.3f}")
    else:
        report.append("No cross-validation results found")
    
    report.append("")
    
    # Feature Importance
    feature_importance = analysis.get("feature_importance", {})
    if feature_importance:
        report.append("TOP FEATURES BY POSITION:")
        report.append("-" * 25)
        
        for position, features in feature_importance.items():
            if features:
                report.append(f"\n{position}:")
                for i, feature in enumerate(features[:5], 1):
                    importance = feature.get("importance", 0)
                    feature_name = feature.get("feature", "unknown")
                    report.append(f"  {i}. {feature_name}: {importance:.0f}")
    
    report.append("")
    
    # Residual Analysis
    residual_analysis = analysis.get("residual_analysis", {})
    if residual_analysis:
        report.append("PREDICTION UNCERTAINTY (RESIDUALS):")
        report.append("-" * 35)
        
        for position, data in residual_analysis.items():
            mae = data["mean_absolute_residual"]
            std = data["std_residual"]
            report.append(f"{position}: MAE {mae:.2f}, Std {std:.2f}")
    
    report.append("")
    report.append("=" * 60)
    
    return "\n".join(report)


def audit_predictions(target_gw: int) -> Dict[str, Any]:
    """
    Run comprehensive audit of predictions for a gameweek.
    
    Args:
        target_gw: Gameweek to audit
        
    Returns:
        Audit results
    """
    logger.info(f"Running prediction audit for GW {target_gw}")
    
    try:
        # Build prediction frame
        df_pred = build_prediction_frame(next_gw=target_gw)
        
        if df_pred.empty:
            return {"error": "No prediction data available"}
        
        # Analyze feature coverage
        coverage_analysis = analyze_feature_coverage(df_pred)
        
        # Analyze model performance
        performance_analysis = analyze_model_performance()
        
        # Analyze league usage
        league_audit = audit_league_usage()
        
        # Generate reports
        data_quality_report = generate_data_quality_report(df_pred)
        model_performance_report = generate_model_performance_report()
        league_audit_report = generate_league_audit_report(league_audit)
        
        return {
            "success": True,
            "target_gw": target_gw,
            "coverage_analysis": coverage_analysis,
            "performance_analysis": performance_analysis,
            "league_audit": league_audit,
            "data_quality_report": data_quality_report,
            "model_performance_report": model_performance_report,
            "league_audit_report": league_audit_report
        }
        
    except Exception as e:
        logger.error(f"Audit failed: {str(e)}", exc_info=True)
        return {"error": str(e)}


def audit_league_usage() -> Dict[str, Any]:
    """
    Audit league usage and provide recommendations for overrides.
    
    Returns:
        League usage analysis and recommendations
    """
    logger.info("Auditing league usage and strength mappings")
    
    try:
        # Get league usage report
        config = get_config()
        settings = config.get_settings()
        league_report = get_new_leagues_report(settings)
        
        if not league_report:
            return {
                "total_leagues": 0,
                "new_leagues": 0,
                "needs_override": 0,
                "recommendations": []
            }
        
        # Analyze the report
        total_leagues = len(league_report)
        new_leagues = sum(1 for data in league_report.values() if data.get("is_using_default", False))
        needs_override = sum(1 for data in league_report.values() if data.get("needs_override", False))
        
        # Generate recommendations
        recommendations = []
        for league_id, data in league_report.items():
            if data.get("needs_override", False):
                count = data.get("count", 0)
                strength = data.get("strength", 0.78)
                recommendations.append(
                    f"League {league_id}: seen {count} times, using default strength {strength:.2f} "
                    f"- consider adding to 'data/league_strength_overrides.csv'"
                )
        
        # Sort recommendations by frequency
        recommendations.sort(key=lambda x: int(x.split("seen ")[1].split(" times")[0]), reverse=True)
        
        return {
            "total_leagues": total_leagues,
            "new_leagues": new_leagues,
            "needs_override": needs_override,
            "recommendations": recommendations,
            "league_details": league_report
        }
        
    except Exception as e:
        logger.error(f"League audit failed: {str(e)}")
        return {"error": str(e)}


def generate_league_audit_report(league_audit: Dict[str, Any]) -> str:
    """Generate league audit report text."""
    if "error" in league_audit:
        return f"League Audit Error: {league_audit['error']}"
    
    report_lines = [
        "=== LEAGUE USAGE AUDIT ===",
        "",
        f"Total leagues encountered: {league_audit['total_leagues']}",
        f"Leagues using default strength: {league_audit['new_leagues']}",
        f"Leagues needing override consideration: {league_audit['needs_override']}",
        ""
    ]
    
    if league_audit['recommendations']:
        report_lines.append("RECOMMENDATIONS:")
        for rec in league_audit['recommendations']:
            report_lines.append(f"  • {rec}")
        report_lines.append("")
        
        # Add sample CSV format
        report_lines.extend([
            "To add overrides, create/update 'data/league_strength_overrides.csv' with:",
            "league_id,strength",
            "# Example: Portuguese Liga (8),0.86",
            ""
        ])
    else:
        report_lines.append("✅ No league overrides needed at this time.")
        report_lines.append("")
    
    return "\n".join(report_lines)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="FPL AI Audit Tool")
    
    parser.add_argument("--gw", type=int, help="Target gameweek (default: current)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--output", "-o", help="Output file for detailed report")
    
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
    
    # Run audit
    try:
        result = audit_predictions(target_gw)
        
        if "error" in result:
            logger.error(f"Audit failed: {result['error']}")
            sys.exit(1)
        
        # Display reports
        print(result["data_quality_report"])
        print("\n")
        print(result["model_performance_report"])
        print("\n")
        print(result["league_audit_report"])
        
        # Save detailed report if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                f.write(result["data_quality_report"])
                f.write("\n\n")
                f.write(result["model_performance_report"])
                f.write("\n\n")
                f.write(result["league_audit_report"])
            
            logger.info(f"Detailed report saved to {output_path}")
        
        logger.info("Audit completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Audit interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
