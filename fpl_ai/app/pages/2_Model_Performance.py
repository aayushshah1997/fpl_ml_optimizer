"""
FPL AI Dashboard - Model Performance Page

Model diagnostics, feature importance, cross-validation metrics,
and data quality analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import json
from typing import Dict, Any

# Add src to path for imports
app_dir = Path(__file__).parent.parent
src_dir = app_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from app._utils import (
    load_model_performance, get_gameweek_selector, load_predictions,
    get_artifacts_dir, create_position_summary
)

# Page configuration
st.set_page_config(
    page_title="Model Performance - FPL AI",
    page_icon="📊",
    layout="wide"
)


def load_cv_metrics() -> Dict[str, Any]:
    """Load cross-validation metrics for all positions."""
    artifacts_dir = get_artifacts_dir()
    models_dir = artifacts_dir / "models"
    
    cv_metrics = {}
    positions = ["GK", "DEF", "MID", "FWD"]
    
    for position in positions:
        cv_file = models_dir / f"cv_{position}.json"
        
        if cv_file.exists():
            try:
                with open(cv_file, 'r') as f:
                    cv_data = json.load(f)
                cv_metrics[position] = cv_data
            except Exception as e:
                st.warning(f"Could not load CV metrics for {position}: {e}")
    
    return cv_metrics


def load_feature_importance() -> Dict[str, pd.DataFrame]:
    """Load feature importance for all positions."""
    artifacts_dir = get_artifacts_dir()
    models_dir = artifacts_dir / "models"
    
    feature_importance = {}
    positions = ["GK", "DEF", "MID", "FWD"]
    
    for position in positions:
        fi_file = models_dir / f"fi_{position}.csv"
        
        if fi_file.exists():
            try:
                fi_df = pd.read_csv(fi_file)
                if not fi_df.empty:
                    feature_importance[position] = fi_df
            except Exception as e:
                st.warning(f"Could not load feature importance for {position}: {e}")
    
    return feature_importance


def load_residual_analysis() -> Dict[str, Dict[str, float]]:
    """Load residual analysis for all positions."""
    artifacts_dir = get_artifacts_dir()
    models_dir = artifacts_dir / "models"
    
    residual_analysis = {}
    positions = ["GK", "DEF", "MID", "FWD"]
    
    for position in positions:
        residuals_file = models_dir / f"residuals_{position}.csv"
        
        if residuals_file.exists():
            try:
                residuals_df = pd.read_csv(residuals_file)
                
                if not residuals_df.empty and "residual" in residuals_df.columns:
                    residuals = residuals_df["residual"]
                    
                    residual_analysis[position] = {
                        "mae": residuals.abs().mean(),
                        "rmse": np.sqrt((residuals ** 2).mean()),
                        "std": residuals.std(),
                        "min": residuals.min(),
                        "max": residuals.max(),
                        "q25": residuals.quantile(0.25),
                        "median": residuals.median(),
                        "q75": residuals.quantile(0.75)
                    }
            except Exception as e:
                st.warning(f"Could not load residuals for {position}: {e}")
    
    return residual_analysis


def create_cv_metrics_chart(cv_metrics: Dict[str, Any]) -> go.Figure:
    """Create cross-validation metrics chart."""
    if not cv_metrics:
        return go.Figure()
    
    positions = list(cv_metrics.keys())
    mae_values = []
    rmse_values = []
    
    for position in positions:
        metrics = cv_metrics[position]
        mae_values.append(metrics.get("mean_mae", 0))
        rmse_values.append(metrics.get("mean_rmse", 0))
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='MAE',
        x=positions,
        y=mae_values,
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='RMSE',
        x=positions,
        y=rmse_values,
        marker_color='lightcoral'
    ))
    
    fig.update_layout(
        title='Cross-Validation Metrics by Position',
        xaxis_title='Position',
        yaxis_title='Error',
        barmode='group',
        height=400
    )
    
    return fig


def create_feature_importance_chart(fi_data: Dict[str, pd.DataFrame], position: str, top_n: int = 15) -> go.Figure:
    """Create feature importance chart for a position."""
    if position not in fi_data or fi_data[position].empty:
        return go.Figure()
    
    fi_df = fi_data[position].head(top_n)
    
    fig = go.Figure(go.Bar(
        x=fi_df.get('importance', []),
        y=fi_df.get('feature', []),
        orientation='h',
        marker_color='steelblue'
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Features - {position}',
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=500,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def create_residual_distribution_chart(residual_data: Dict[str, Dict[str, float]]) -> go.Figure:
    """Create residual distribution chart."""
    if not residual_data:
        return go.Figure()
    
    fig = go.Figure()
    
    positions = list(residual_data.keys())
    colors = ['blue', 'green', 'orange', 'red']
    
    for i, position in enumerate(positions):
        data = residual_data[position]
        
        # Create box plot data
        fig.add_trace(go.Box(
            name=position,
            q1=[data.get('q25', 0)],
            median=[data.get('median', 0)],
            q3=[data.get('q75', 0)],
            lowerfence=[data.get('min', 0)],
            upperfence=[data.get('max', 0)],
            marker_color=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        title='Prediction Residual Distribution by Position',
        yaxis_title='Residual (Actual - Predicted)',
        height=400
    )
    
    return fig


def analyze_feature_coverage(predictions_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze feature coverage in predictions."""
    if predictions_df.empty:
        return {}
    
    total_players = len(predictions_df)
    total_features = len(predictions_df.columns)
    
    # Feature categories
    feature_categories = {
        "Basic Info": ["element_id", "web_name", "position", "team_name", "now_cost"],
        "Projections": ["proj_points", "expected_minutes", "proj_raw"],
        "Recent Form": [col for col in predictions_df.columns if col.startswith(("r3_", "r5_", "r8_"))],
        "Attacking": [col for col in predictions_df.columns if any(x in col for x in ["goals", "assists", "shots", "xG", "xA"])],
        "Defensive": [col for col in predictions_df.columns if any(x in col for x in ["clean_sheets", "goals_conceded", "saves"])],
        "Set Pieces": [col for col in predictions_df.columns if any(x in col for x in ["pen_", "fk_", "corner_"])],
        "Team Form": [col for col in predictions_df.columns if "team_" in col and "form" in col],
        "Market": [col for col in predictions_df.columns if any(x in col for x in ["selected", "ownership", "price"])],
        "Advanced": [col for col in predictions_df.columns if any(x in col for x in ["BPS", "ICT", "creativity", "threat"])]
    }
    
    coverage_analysis = {}
    
    for category, features in feature_categories.items():
        available_features = [f for f in features if f in predictions_df.columns]
        
        if available_features:
            # Calculate missing data
            missing_percentages = {}
            for feature in available_features:
                missing_pct = (predictions_df[feature].isna().sum() / total_players) * 100
                missing_percentages[feature] = missing_pct
            
            coverage_analysis[category] = {
                "available": len(available_features),
                "total": len(features),
                "coverage_pct": (len(available_features) / len(features)) * 100,
                "avg_missing_pct": np.mean(list(missing_percentages.values())) if missing_percentages else 100,
                "features": available_features
            }
        else:
            coverage_analysis[category] = {
                "available": 0,
                "total": len(features),
                "coverage_pct": 0,
                "avg_missing_pct": 100,
                "features": []
            }
    
    return {
        "total_players": total_players,
        "total_features": total_features,
        "by_category": coverage_analysis
    }


def main():
    """Main model performance page."""
    
    st.title("📊 Model Performance")
    st.markdown("### Comprehensive Model Diagnostics & Analysis")
    
    # Get current gameweek
    target_gw = get_gameweek_selector()
    
    # Load data
    cv_metrics = load_cv_metrics()
    feature_importance = load_feature_importance()
    residual_analysis = load_residual_analysis()
    predictions_df = load_predictions(target_gw)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Model Metrics", "🔍 Feature Analysis", "📈 Data Quality", "⚙️ System Info"])
    
    with tab1:
        st.subheader("🎯 Cross-Validation Performance")
        
        if cv_metrics:
            # CV metrics overview
            col1, col2, col3, col4 = st.columns(4)
            
            positions = ["GK", "DEF", "MID", "FWD"]
            for i, position in enumerate(positions):
                with [col1, col2, col3, col4][i]:
                    if position in cv_metrics:
                        mae = cv_metrics[position].get("mean_mae", 0)
                        st.metric(f"{position} MAE", f"{mae:.3f}")
                    else:
                        st.metric(f"{position} MAE", "N/A")
            
            # CV chart
            cv_chart = create_cv_metrics_chart(cv_metrics)
            st.plotly_chart(cv_chart, use_container_width=True)
            
            # Detailed metrics table
            st.markdown("#### 📋 Detailed Metrics")
            
            cv_table_data = []
            for position in positions:
                if position in cv_metrics:
                    metrics = cv_metrics[position]
                    cv_table_data.append({
                        "Position": position,
                        "MAE": f"{metrics.get('mean_mae', 0):.3f} ± {metrics.get('std_mae', 0):.3f}",
                        "RMSE": f"{metrics.get('mean_rmse', 0):.3f} ± {metrics.get('std_rmse', 0):.3f}",
                        "R²": f"{metrics.get('mean_r2', 0):.3f}",
                        "CV Folds": metrics.get('n_splits', 'N/A')
                    })
            
            if cv_table_data:
                cv_df = pd.DataFrame(cv_table_data)
                st.dataframe(cv_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No cross-validation metrics available. Train models to generate metrics.")
        
        st.divider()
        
        # Residual analysis
        st.subheader("📊 Prediction Uncertainty")
        
        if residual_analysis:
            # Residual distribution chart
            residual_chart = create_residual_distribution_chart(residual_analysis)
            st.plotly_chart(residual_chart, use_container_width=True)
            
            # Residual statistics table
            st.markdown("#### 📈 Residual Statistics")
            
            residual_table_data = []
            for position, stats in residual_analysis.items():
                residual_table_data.append({
                    "Position": position,
                    "MAE": f"{stats.get('mae', 0):.2f}",
                    "RMSE": f"{stats.get('rmse', 0):.2f}",
                    "Std Dev": f"{stats.get('std', 0):.2f}",
                    "Range": f"[{stats.get('min', 0):.1f}, {stats.get('max', 0):.1f}]"
                })
            
            if residual_table_data:
                residual_df = pd.DataFrame(residual_table_data)
                st.dataframe(residual_df, use_container_width=True, hide_index=True)
                
                st.info("""
                **Residual Interpretation:**
                - Lower MAE/RMSE = Better prediction accuracy
                - Std Dev shows prediction uncertainty used in Monte Carlo
                - Range shows the spread of prediction errors
                """)
        else:
            st.warning("No residual analysis available.")
    
    with tab2:
        st.subheader("🔍 Feature Importance Analysis")
        
        if feature_importance:
            # Position selector
            position = st.selectbox(
                "Select Position",
                options=list(feature_importance.keys()),
                index=0
            )
            
            if position in feature_importance:
                fi_df = feature_importance[position]
                
                # Feature importance chart
                fi_chart = create_feature_importance_chart(feature_importance, position)
                st.plotly_chart(fi_chart, use_container_width=True)
                
                # Top features table
                st.markdown(f"#### 🏆 Top Features for {position}")
                
                top_features = fi_df.head(20)
                
                # Add feature categories
                if not top_features.empty:
                    feature_cats = []
                    for _, row in top_features.iterrows():
                        feature_name = row.get('feature', '')
                        
                        if any(x in feature_name for x in ['r3_', 'r5_', 'r8_']):
                            category = "Recent Form"
                        elif any(x in feature_name for x in ['goals', 'assists', 'shots', 'xG', 'xA']):
                            category = "Attacking"
                        elif any(x in feature_name for x in ['clean_sheets', 'goals_conceded', 'saves']):
                            category = "Defensive"
                        elif any(x in feature_name for x in ['pen_', 'fk_', 'corner_']):
                            category = "Set Pieces"
                        elif 'team_' in feature_name:
                            category = "Team Form"
                        elif any(x in feature_name for x in ['selected', 'ownership', 'price']):
                            category = "Market"
                        elif any(x in feature_name for x in ['BPS', 'ICT', 'creativity']):
                            category = "Advanced"
                        else:
                            category = "Other"
                        
                        feature_cats.append(category)
                    
                    top_features = top_features.copy()
                    top_features['Category'] = feature_cats
                    
                    # Display table
                    display_cols = ['feature', 'Category', 'importance']
                    available_cols = [col for col in display_cols if col in top_features.columns]
                    
                    if available_cols:
                        display_df = top_features[available_cols].copy()
                        display_df = display_df.rename(columns={
                            'feature': 'Feature',
                            'importance': 'Importance'
                        })
                        
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Feature category summary
                if 'Category' in top_features.columns:
                    st.markdown("#### 📊 Feature Category Distribution")
                    
                    category_counts = top_features['Category'].value_counts()
                    
                    # Create pie chart
                    fig = px.pie(
                        values=category_counts.values,
                        names=category_counts.index,
                        title=f"Top 20 Features by Category - {position}"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No feature importance data available. Train models to generate feature importance.")
    
    with tab3:
        st.subheader("📈 Data Quality Analysis")
        
        if predictions_df is not None and not predictions_df.empty:
            # Feature coverage analysis
            coverage_analysis = analyze_feature_coverage(predictions_df)
            
            if coverage_analysis:
                # Overview metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Players", f"{coverage_analysis['total_players']:,}")
                with col2:
                    st.metric("Total Features", coverage_analysis['total_features'])
                with col3:
                    # Calculate overall coverage
                    by_category = coverage_analysis['by_category']
                    avg_coverage = np.mean([cat['coverage_pct'] for cat in by_category.values()])
                    st.metric("Avg Coverage", f"{avg_coverage:.1f}%")
                
                # Coverage by category
                st.markdown("#### 📊 Feature Coverage by Category")
                
                coverage_data = []
                for category, data in coverage_analysis['by_category'].items():
                    status = "✅" if data['coverage_pct'] > 80 and data['avg_missing_pct'] < 20 else "⚠️" if data['coverage_pct'] > 50 else "❌"
                    
                    coverage_data.append({
                        "Status": status,
                        "Category": category,
                        "Features": f"{data['available']}/{data['total']}",
                        "Coverage": f"{data['coverage_pct']:.1f}%",
                        "Missing Data": f"{data['avg_missing_pct']:.1f}%"
                    })
                
                coverage_df = pd.DataFrame(coverage_data)
                st.dataframe(coverage_df, use_container_width=True, hide_index=True)
                
                # Position summary
                st.markdown("#### 📈 Position Summary")
                
                position_summary = create_position_summary(predictions_df)
                if not position_summary.empty:
                    st.dataframe(position_summary, use_container_width=True, hide_index=True)
                
                # Data quality issues
                st.markdown("#### ⚠️ Data Quality Issues")
                
                quality_issues = []
                
                # Check for missing essential data
                essential_cols = ["element_id", "position", "proj_points"]
                for col in essential_cols:
                    if col not in predictions_df.columns:
                        quality_issues.append(f"Missing essential column: {col}")
                    elif predictions_df[col].isna().any():
                        missing_count = predictions_df[col].isna().sum()
                        quality_issues.append(f"Missing values in {col}: {missing_count} players")
                
                # Check for suspicious values
                if "proj_points" in predictions_df.columns:
                    negative_proj = (predictions_df["proj_points"] < 0).sum()
                    if negative_proj > 0:
                        quality_issues.append(f"Negative projections: {negative_proj} players")
                    
                    extreme_proj = (predictions_df["proj_points"] > 20).sum()
                    if extreme_proj > 0:
                        quality_issues.append(f"Extremely high projections (>20): {extreme_proj} players")
                
                if quality_issues:
                    for issue in quality_issues:
                        st.warning(f"⚠️ {issue}")
                else:
                    st.success("✅ No major data quality issues detected")
        else:
            st.warning(f"No prediction data available for GW {target_gw}")
    
    with tab4:
        st.subheader("⚙️ System Information")
        
        # Model files status
        st.markdown("#### 🤖 Model Files Status")
        
        artifacts_dir = get_artifacts_dir()
        models_dir = artifacts_dir / "models"
        
        model_files = [
            "model_minutes.pkl",
            "model_points_GK.pkl",
            "model_points_DEF.pkl", 
            "model_points_MID.pkl",
            "model_points_FWD.pkl"
        ]
        
        model_status = []
        for model_file in model_files:
            file_path = models_dir / model_file
            exists = file_path.exists()
            
            if exists:
                try:
                    file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                    size_str = f"{file_size:.1f} MB"
                except:
                    size_str = "Unknown"
            else:
                size_str = "N/A"
            
            model_status.append({
                "Model": model_file,
                "Status": "✅ Available" if exists else "❌ Missing",
                "Size": size_str
            })
        
        model_status_df = pd.DataFrame(model_status)
        st.dataframe(model_status_df, use_container_width=True, hide_index=True)
        
        # Configuration info
        st.markdown("#### ⚙️ Configuration")
        
        from app._utils import load_config
        config = load_config()
        
        config_info = {
            "Artifacts Directory": str(artifacts_dir),
            "MC Scenarios": config.get("mc", {}).get("num_scenarios", "N/A"),
            "Risk Lambda": config.get("mc", {}).get("lambda_risk", "N/A"),
            "Warm Until GW": config.get("training", {}).get("staging", {}).get("warm_until_gw", "N/A"),
            "Training Seasons": config.get("training", {}).get("seasons_back", "N/A")
        }
        
        for key, value in config_info.items():
            st.markdown(f"**{key}:** {value}")
        
        # Performance recommendations
        st.markdown("#### 💡 Performance Recommendations")
        
        recommendations = []
        
        if not cv_metrics:
            recommendations.append("🎯 Train models to generate performance metrics")
        
        if not feature_importance:
            recommendations.append("🔍 Feature importance analysis unavailable - retrain models")
        
        if predictions_df is None:
            recommendations.append(f"📊 Generate predictions for GW {target_gw}")
        
        # Check if models are too old
        try:
            oldest_model = min([
                (models_dir / f).stat().st_mtime 
                for f in model_files 
                if (models_dir / f).exists()
            ])
            
            import time
            days_old = (time.time() - oldest_model) / (24 * 3600)
            
            if days_old > 7:
                recommendations.append(f"🔄 Models are {days_old:.1f} days old - consider retraining")
        except:
            pass
        
        if recommendations:
            for rec in recommendations:
                st.info(rec)
        else:
            st.success("✅ System appears to be running optimally")


if __name__ == "__main__":
    main()
