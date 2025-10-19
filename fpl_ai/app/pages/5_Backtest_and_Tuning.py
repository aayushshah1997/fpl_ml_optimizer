"""
FPL AI Dashboard - Backtest & Auto-Tuning Page

Walk-forward backtesting results, hyperparameter optimization leaderboard,
and parameter sensitivity analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import json
from typing import Dict, Any, Optional, List

# Add src to path for imports
app_dir = Path(__file__).parent.parent
src_dir = app_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from fpl_ai.app._utils import get_artifacts_dir

# Page configuration
st.set_page_config(
    page_title="Backtest & Auto-Tuning - FPL AI",
    page_icon="🔬",
    layout="wide"
)


def load_study_summary() -> Optional[Dict[str, Any]]:
    """Load Optuna study summary."""
    artifacts_dir = get_artifacts_dir()
    tuning_dir = artifacts_dir / "tuning"
    summary_file = tuning_dir / "study_summary.json"
    
    if summary_file.exists():
        try:
            with open(summary_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Failed to load study summary: {e}")
    
    return None


def load_leaderboard() -> pd.DataFrame:
    """Load tuning results leaderboard."""
    artifacts_dir = get_artifacts_dir()
    tuning_dir = artifacts_dir / "tuning"
    leaderboard_file = tuning_dir / "leaderboard.json"
    
    if leaderboard_file.exists():
        try:
            with open(leaderboard_file, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        except Exception as e:
            st.error(f"Failed to load leaderboard: {e}")
    
    return pd.DataFrame()


def load_best_settings() -> Optional[Dict[str, Any]]:
    """Load best settings from tuning."""
    artifacts_dir = get_artifacts_dir()
    tuning_dir = artifacts_dir / "tuning"
    best_settings_file = tuning_dir / "best_settings.yaml"
    
    if best_settings_file.exists():
        try:
            import yaml
            with open(best_settings_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            st.error(f"Failed to load best settings: {e}")
    
    return None


def load_walkforward_results(trial_number: Optional[int] = None) -> pd.DataFrame:
    """Load walk-forward backtest results."""
    artifacts_dir = get_artifacts_dir()
    tuning_dir = artifacts_dir / "tuning"
    
    if trial_number is not None:
        # Load specific trial results
        trial_dir = tuning_dir / f"trial_{trial_number}"
        wf_file = trial_dir / "backtest_results.csv"
    else:
        # Look for any walk-forward results
        wf_files = list(tuning_dir.glob("wf_trial_*.csv"))
        if wf_files:
            wf_file = wf_files[-1]  # Use most recent
        else:
            # Check for trial directories
            trial_dirs = list(tuning_dir.glob("trial_*"))
            if trial_dirs:
                trial_dir = sorted(trial_dirs)[-1]  # Most recent trial
                wf_file = trial_dir / "backtest_results.csv"
            else:
                return pd.DataFrame()
    
    if wf_file.exists():
        try:
            return pd.read_csv(wf_file)
        except Exception as e:
            st.error(f"Failed to load walkforward results: {e}")
    
    return pd.DataFrame()


def display_study_overview(summary: Dict[str, Any]):
    """Display study overview metrics."""
    st.header("📈 Study Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Trials Completed", 
            f"{summary.get('n_complete', 0)}/{summary.get('n_trials', 0)}"
        )
    
    with col2:
        study_time = summary.get('study_time_seconds', 0)
        if study_time > 3600:
            time_str = f"{study_time/3600:.1f}h"
        elif study_time > 60:
            time_str = f"{study_time/60:.1f}m"
        else:
            time_str = f"{study_time:.0f}s"
        st.metric("Study Time", time_str)
    
    with col3:
        st.metric("Best Trial", f"#{summary.get('best_trial', 'N/A')}")
    
    with col4:
        best_value = summary.get('best_value', 0)
        objective = summary.get('objective', 'unknown')
        st.metric(f"Best {objective.replace('_', ' ').title()}", f"{best_value:.2f}")
    
    # Backtest configuration
    backtest_config = summary.get('backtest_config', {})
    if backtest_config:
        st.subheader("Backtest Configuration")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            seasons = backtest_config.get('seasons', [])
            if seasons and len(seasons) >= 2:
                st.write(f"**Seasons:** {len(seasons)} ({seasons[0]} to {seasons[-1]})")
            elif seasons and len(seasons) == 1:
                st.write(f"**Seasons:** {len(seasons)} ({seasons[0]})")
            else:
                st.write("**Seasons:** N/A")
        
        with col2:
            start_gw = backtest_config.get('start_gw', 'N/A')
            end_gw = backtest_config.get('end_gw', 'N/A')
            st.write(f"**Gameweeks:** {start_gw} to {end_gw}")
        
        with col3:
            search_space = summary.get('search_space', {})
            st.write(f"**Parameters Tuned:** {len(search_space)}")


def display_leaderboard(df: pd.DataFrame):
    """Display trials leaderboard."""
    st.header("🏆 Trials Leaderboard")
    
    if df.empty:
        st.info("No trial results available. Run tuning to populate leaderboard.")
        return
    
    # Top trials table
    display_cols = [
        'trial', 'value', 'total_points', 'risk_adj_points', 
        'sharpe_like', 'success_rate', 'trial_time'
    ]
    
    available_cols = [col for col in display_cols if col in df.columns]
    
    if available_cols:
        top_df = df[available_cols].head(10).copy()
        
        # Format columns
        if 'success_rate' in top_df.columns:
            top_df['success_rate'] = top_df['success_rate'].apply(lambda x: f"{x:.1%}")
        if 'trial_time' in top_df.columns:
            top_df['trial_time'] = top_df['trial_time'].apply(lambda x: f"{x:.1f}s")
        
        # Rename columns for display
        column_names = {
            'trial': 'Trial #',
            'value': 'Objective Value',
            'total_points': 'Total Points',
            'risk_adj_points': 'Risk Adj Points',
            'sharpe_like': 'Sharpe-like',
            'success_rate': 'Success Rate',
            'trial_time': 'Time'
        }
        
        top_df = top_df.rename(columns=column_names)
        
        st.dataframe(
            top_df,
            use_container_width=True,
            hide_index=True
        )
    
    # Trial performance distribution
    if 'value' in df.columns and len(df) > 1:
        st.subheader("Trial Performance Distribution")
        
        fig = px.histogram(
            df, 
            x='value', 
            nbins=20,
            title="Distribution of Trial Objective Values",
            labels={'value': 'Objective Value', 'count': 'Number of Trials'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def display_parameter_analysis(df: pd.DataFrame):
    """Display parameter sensitivity analysis."""
    st.header("🔧 Parameter Analysis")
    
    if df.empty:
        st.info("No parameter data available.")
        return
    
    # Extract parameters from the params column
    if 'params' not in df.columns:
        st.warning("No parameter information found in trials data.")
        return
    
    # Parse parameters
    param_data = []
    for idx, row in df.iterrows():
        trial_params = row.get('params', {})
        if isinstance(trial_params, dict):
            for param_name, param_value in trial_params.items():
                param_data.append({
                    'trial': row['trial'],
                    'value': row['value'],
                    'parameter': param_name,
                    'param_value': param_value
                })
    
    if not param_data:
        st.warning("No parameter data could be parsed.")
        return
    
    params_df = pd.DataFrame(param_data)
    
    # Parameter importance (correlation with objective)
    st.subheader("Parameter Importance")
    
    param_correlations = []
    for param in params_df['parameter'].unique():
        param_subset = params_df[params_df['parameter'] == param]
        
        # Only analyze numeric parameters
        try:
            numeric_values = pd.to_numeric(param_subset['param_value'], errors='coerce')
            if not numeric_values.isna().all():
                correlation = numeric_values.corr(param_subset['value'])
                if not pd.isna(correlation):
                    param_correlations.append({
                        'parameter': param,
                        'correlation': abs(correlation),
                        'direction': 'positive' if correlation > 0 else 'negative'
                    })
        except:
            continue
    
    if param_correlations:
        corr_df = pd.DataFrame(param_correlations).sort_values('correlation', ascending=False)
        
        fig = px.bar(
            corr_df.head(10), 
            x='correlation', 
            y='parameter',
            color='direction',
            orientation='h',
            title="Top 10 Most Important Parameters (by correlation with objective)",
            labels={'correlation': 'Absolute Correlation', 'parameter': 'Parameter'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Parameter value distributions for top parameters
    if param_correlations:
        st.subheader("Parameter Value Distributions")
        
        top_params = [p['parameter'] for p in param_correlations[:4]]
        
        cols = st.columns(2)
        for i, param in enumerate(top_params):
            with cols[i % 2]:
                param_subset = params_df[params_df['parameter'] == param]
                
                # Check if parameter is numeric
                try:
                    numeric_values = pd.to_numeric(param_subset['param_value'], errors='coerce')
                    if not numeric_values.isna().all():
                        fig = px.scatter(
                            param_subset,
                            x='param_value',
                            y='value',
                            title=f"{param}",
                            labels={'param_value': 'Parameter Value', 'value': 'Objective Value'}
                        )
                    else:
                        # Categorical parameter
                        fig = px.box(
                            param_subset,
                            x='param_value',
                            y='value',
                            title=f"{param}",
                            labels={'param_value': 'Parameter Value', 'value': 'Objective Value'}
                        )
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.write(f"Could not analyze parameter: {param}")


def display_walkforward_analysis(df: pd.DataFrame):
    """Display walk-forward backtest analysis."""
    st.header("📊 Walk-Forward Backtest Analysis")
    
    if df.empty:
        st.info("No walk-forward results available. Run tuning to generate backtest data.")
        return
    
    # Overall performance metrics
    st.subheader("Performance Summary")
    
    successful_gws = df[df['status'] == 'success'] if 'status' in df.columns else df
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_points = successful_gws['points'].sum() if 'points' in df.columns else 0
        st.metric("Total Points", f"{total_points:.1f}")
    
    with col2:
        avg_points = successful_gws['points'].mean() if 'points' in df.columns else 0
        st.metric("Avg Points/GW", f"{avg_points:.2f}")
    
    with col3:
        success_rate = len(successful_gws) / len(df) if len(df) > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1%}")
    
    with col4:
        if 'points' in df.columns:
            volatility = successful_gws['points'].std()
            st.metric("Volatility (σ)", f"{volatility:.2f}")
    
    # Points per gameweek chart
    if 'points' in df.columns and 'gw' in df.columns:
        st.subheader("Points by Gameweek")
        
        # Aggregate by gameweek
        gw_summary = df.groupby('gw')['points'].agg(['mean', 'std', 'count']).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=gw_summary['gw'],
            y=gw_summary['mean'],
            mode='lines+markers',
            name='Average Points',
            line=dict(color='blue', width=2)
        ))
        
        # Add error bars if we have std data
        if 'std' in gw_summary.columns:
            fig.add_trace(go.Scatter(
                x=gw_summary['gw'],
                y=gw_summary['mean'] + gw_summary['std'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=gw_summary['gw'],
                y=gw_summary['mean'] - gw_summary['std'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0,100,80,0.2)',
                name='±1 Std Dev',
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title="Average Points by Gameweek",
            xaxis_title="Gameweek",
            yaxis_title="Points",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Season-by-season breakdown
    if 'season' in df.columns:
        st.subheader("Performance by Season")
        
        season_summary = df.groupby('season')['points'].agg(['sum', 'mean', 'count']).reset_index()
        season_summary.columns = ['Season', 'Total Points', 'Avg Points/GW', 'Games Played']
        
        # Format the dataframe
        season_summary['Total Points'] = season_summary['Total Points'].round(1)
        season_summary['Avg Points/GW'] = season_summary['Avg Points/GW'].round(2)
        
        st.dataframe(season_summary, use_container_width=True, hide_index=True)
        
        # Season performance chart
        fig = px.bar(
            season_summary,
            x='Season',
            y='Total Points',
            title="Total Points by Season",
            text='Total Points'
        )
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def display_best_settings(settings: Dict[str, Any]):
    """Display best found settings."""
    st.header("⚙️ Best Settings")
    
    if not settings:
        st.info("No optimized settings available.")
        return
    
    # Key optimized parameters
    st.subheader("Optimized Parameters")
    
    # Extract key sections
    modeling_params = settings.get('modeling', {})
    mc_params = settings.get('mc', {})
    captain_params = settings.get('captain', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**LightGBM Parameters:**")
        gbm_params = modeling_params.get('gbm', {})
        key_gbm_params = {
            'learning_rate': gbm_params.get('learning_rate'),
            'num_leaves': gbm_params.get('num_leaves'),
            'n_estimators': gbm_params.get('n_estimators'),
            'subsample': gbm_params.get('subsample'),
            'reg_alpha': gbm_params.get('reg_alpha'),
            'reg_lambda': gbm_params.get('reg_lambda')
        }
        
        for param, value in key_gbm_params.items():
            if value is not None:
                st.write(f"- {param}: {value}")
        
        # Position-specific parameters
        gbm_by_pos = modeling_params.get('gbm_by_pos', {})
        if gbm_by_pos:
            st.write("**Position-Specific Overrides:**")
            for pos, pos_params in gbm_by_pos.items():
                if pos_params:
                    st.write(f"- {pos}: {len(pos_params)} parameters")
    
    with col2:
        st.write("**Monte Carlo Parameters:**")
        key_mc_params = {
            'cvar_alpha': mc_params.get('cvar_alpha'),
            'lambda_risk': mc_params.get('lambda_risk'),
            'minutes_uncertainty': mc_params.get('minutes_uncertainty'),
            'team_correlation': mc_params.get('correlation', {}).get('team_level'),
            'opponent_correlation': mc_params.get('correlation', {}).get('opponent_level')
        }
        
        for param, value in key_mc_params.items():
            if value is not None:
                st.write(f"- {param}: {value}")
        
        st.write("**Captain Parameters:**")
        key_captain_params = {
            'policy': captain_params.get('policy'),
            'mix_lambda': captain_params.get('mix_lambda'),
            'candidates': captain_params.get('candidates')
        }
        
        for param, value in key_captain_params.items():
            if value is not None:
                st.write(f"- {param}: {value}")
    
    # Full settings as expandable JSON
    with st.expander("View Full Best Configuration"):
        st.json(settings)


def main():
    """Main dashboard function."""
    st.title("🔬 Backtest & Auto-Tuning")
    st.markdown("Walk-forward backtesting results and hyperparameter optimization analysis")
    
    # Load data
    summary = load_study_summary()
    leaderboard_df = load_leaderboard()
    best_settings = load_best_settings()
    
    # Check if any tuning data exists
    if summary is None and leaderboard_df.empty:
        st.info("⚠️ No tuning results found. Run hyperparameter tuning to populate this dashboard.")
        st.markdown("""
        **To get started:**
        
        1. Run tuning from the command line:
           ```bash
           make tune
           # or
           python -m src.cli.tune_and_retrain tune --quick
           ```
        
        2. Or run both tuning and retraining:
           ```bash
           make tune_and_retrain
           ```
        
        3. Refresh this page to see results
        """)
        return
    
    # Display sections
    if summary:
        display_study_overview(summary)
        st.divider()
    
    if not leaderboard_df.empty:
        display_leaderboard(leaderboard_df)
        st.divider()
        
        display_parameter_analysis(leaderboard_df)
        st.divider()
    
    # Walk-forward results
    wf_df = load_walkforward_results()
    if not wf_df.empty:
        display_walkforward_analysis(wf_df)
        st.divider()
    
    # Best settings
    if best_settings:
        display_best_settings(best_settings)
    
    # Footer with manual actions
    st.markdown("---")
    st.markdown("""
    **Manual Actions:**
    - **Run Tuning:** `make tune` or `python -m src.cli.tune_and_retrain tune`
    - **Quick Tune:** `python -m src.cli.tune_and_retrain tune --quick`
    - **Retrain Models:** `make retrain` or `python -m src.cli.tune_and_retrain retrain`
    - **View Results:** `python -m src.cli.tune_and_retrain results`
    """)


if __name__ == "__main__":
    main()
