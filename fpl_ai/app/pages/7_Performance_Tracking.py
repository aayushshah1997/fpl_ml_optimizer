"""
Performance Tracking Dashboard - FPL AI Model Performance Analysis

This page provides comprehensive analysis of model prediction accuracy
by comparing predicted vs actual player points over time.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add the parent directory to the path so we can import from src
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.common.performance_tracker import ModelPerformanceTracker
from src.common.config import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="📊 Performance Tracking",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_performance_data() -> Dict[str, Any]:
    """Load all performance tracking data."""
    try:
        # Use absolute path to artifacts directory
        artifacts_path = Path(__file__).parent.parent.parent / "artifacts"
        tracker = ModelPerformanceTracker(str(artifacts_path))
        
        # Load summary
        summary = tracker.get_performance_summary()
        
        # Load detailed data
        team_performance = pd.DataFrame()
        player_performance = pd.DataFrame()
        
        if tracker.team_performance_file.exists():
            team_performance = pd.read_csv(tracker.team_performance_file)
        
        if tracker.player_performance_file.exists():
            player_performance = pd.read_csv(tracker.player_performance_file)
        
        return {
            'summary': summary,
            'team_performance': team_performance,
            'player_performance': player_performance,
            'tracker': tracker
        }
    except Exception as e:
        logger.error(f"Error loading performance data: {e}")
        return {
            'summary': None,
            'team_performance': pd.DataFrame(),
            'player_performance': pd.DataFrame(),
            'tracker': None
        }

def create_accuracy_trend_chart(team_df: pd.DataFrame) -> go.Figure:
    """Create accuracy trend chart over gameweeks."""
    if team_df.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Squad Points: Predicted vs Actual',
            'Starting XI Points: Predicted vs Actual', 
            'Mean Absolute Error Trend',
            'Accuracy Percentage Trend'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Squad points comparison
    fig.add_trace(
        go.Scatter(
            x=team_df['gameweek'],
            y=team_df['squad_predicted'],
            mode='lines+markers',
            name='Squad Predicted',
            line=dict(color='blue'),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=team_df['gameweek'],
            y=team_df['squad_actual'],
            mode='lines+markers',
            name='Squad Actual',
            line=dict(color='red'),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    # Starting XI points comparison
    fig.add_trace(
        go.Scatter(
            x=team_df['gameweek'],
            y=team_df['xi_predicted'],
            mode='lines+markers',
            name='XI Predicted',
            line=dict(color='green'),
            marker=dict(size=8),
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=team_df['gameweek'],
            y=team_df['xi_actual'],
            mode='lines+markers',
            name='XI Actual',
            line=dict(color='orange'),
            marker=dict(size=8),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # MAE trend
    fig.add_trace(
        go.Scatter(
            x=team_df['gameweek'],
            y=team_df['squad_mae'],
            mode='lines+markers',
            name='Squad MAE',
            line=dict(color='purple'),
            marker=dict(size=8),
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=team_df['gameweek'],
            y=team_df['xi_mae'],
            mode='lines+markers',
            name='XI MAE',
            line=dict(color='brown'),
            marker=dict(size=8),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Accuracy percentage trend
    if 'squad_predicted' in team_df.columns and 'squad_actual' in team_df.columns:
        squad_accuracy = (1 - team_df['squad_mae'] / team_df['squad_actual'].abs()) * 100
        xi_accuracy = (1 - team_df['xi_mae'] / team_df['xi_actual'].abs()) * 100
        
        fig.add_trace(
            go.Scatter(
                x=team_df['gameweek'],
                y=squad_accuracy,
                mode='lines+markers',
                name='Squad Accuracy %',
                line=dict(color='darkblue'),
                marker=dict(size=8),
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=team_df['gameweek'],
                y=xi_accuracy,
                mode='lines+markers',
                name='XI Accuracy %',
                line=dict(color='darkgreen'),
                marker=dict(size=8),
                showlegend=False
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title_text="Model Performance Trends Over Time",
        height=600,
        showlegend=True
    )
    
    # Update x-axis titles
    fig.update_xaxes(title_text="Gameweek", row=2, col=1)
    fig.update_xaxes(title_text="Gameweek", row=2, col=2)
    
    # Update y-axis titles
    fig.update_yaxes(title_text="Points", row=1, col=1)
    fig.update_yaxes(title_text="Points", row=1, col=2)
    fig.update_yaxes(title_text="MAE", row=2, col=1)
    fig.update_yaxes(title_text="Accuracy %", row=2, col=2)
    
    return fig

def create_position_accuracy_chart(player_df: pd.DataFrame) -> go.Figure:
    """Create position-wise accuracy analysis."""
    if player_df.empty:
        return go.Figure()
    
    # Calculate position-wise metrics
    position_stats = player_df.groupby('position').agg({
        'predicted_points': 'mean',
        'actual_points': 'mean',
        'absolute_error': 'mean',
        'element_id': 'count'
    }).round(2)
    
    position_stats.columns = ['Avg Predicted', 'Avg Actual', 'Avg MAE', 'Player Count']
    position_stats = position_stats.reset_index()
    
    # Calculate accuracy percentage
    position_stats['Accuracy %'] = (1 - position_stats['Avg MAE'] / position_stats['Avg Actual'].abs()) * 100
    position_stats['Accuracy %'] = position_stats['Accuracy %'].round(1)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Average Points by Position', 'Accuracy by Position'),
        specs=[[{"secondary_y": False}, {"secondary_y": True}]]
    )
    
    # Average points comparison
    fig.add_trace(
        go.Bar(
            x=position_stats['position'],
            y=position_stats['Avg Predicted'],
            name='Predicted',
            marker_color='lightblue',
            text=position_stats['Avg Predicted'],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=position_stats['position'],
            y=position_stats['Avg Actual'],
            name='Actual',
            marker_color='lightcoral',
            text=position_stats['Avg Actual'],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Accuracy percentage
    fig.add_trace(
        go.Bar(
            x=position_stats['position'],
            y=position_stats['Accuracy %'],
            name='Accuracy %',
            marker_color='lightgreen',
            text=position_stats['Accuracy %'],
            textposition='auto',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Player count on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=position_stats['position'],
            y=position_stats['Player Count'],
            mode='lines+markers',
            name='Player Count',
            line=dict(color='red', width=3),
            marker=dict(size=10),
            yaxis='y2',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="Position-wise Performance Analysis",
        height=400,
        showlegend=True
    )
    
    # Update y-axis titles
    fig.update_yaxes(title_text="Average Points", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy %", row=1, col=2)
    fig.update_yaxes(title_text="Player Count", secondary_y=True, row=1, col=2)
    
    return fig

def create_captain_performance_chart(team_df: pd.DataFrame) -> go.Figure:
    """Create captain performance analysis."""
    if team_df.empty or 'captain_predicted' not in team_df.columns:
        return go.Figure()
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=team_df['gameweek'],
            y=team_df['captain_predicted'],
            mode='lines+markers',
            name='Captain Predicted',
            line=dict(color='gold', width=3),
            marker=dict(size=10)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=team_df['gameweek'],
            y=team_df['captain_actual'],
            mode='lines+markers',
            name='Captain Actual',
            line=dict(color='darkgoldenrod', width=3),
            marker=dict(size=10)
        )
    )
    
    # Add error bars
    fig.add_trace(
        go.Scatter(
            x=team_df['gameweek'],
            y=team_df['captain_error'],
            mode='lines+markers',
            name='Captain Error',
            line=dict(color='red', dash='dash'),
            marker=dict(size=8)
        )
    )
    
    fig.update_layout(
        title="Captain Performance Over Time",
        xaxis_title="Gameweek",
        yaxis_title="Points (with Captain Multiplier)",
        height=400
    )
    
    return fig

def create_gameweek_breakdown_table(team_df: pd.DataFrame) -> pd.DataFrame:
    """Create detailed gameweek breakdown table."""
    if team_df.empty:
        return pd.DataFrame()
    
    breakdown = team_df.copy()
    
    # Calculate additional metrics
    breakdown['Squad Error'] = breakdown['squad_actual'] - breakdown['squad_predicted']
    breakdown['XI Error'] = breakdown['xi_actual'] - breakdown['xi_predicted']
    breakdown['Squad Accuracy %'] = (1 - breakdown['squad_mae'] / breakdown['squad_actual'].abs()) * 100
    breakdown['XI Accuracy %'] = (1 - breakdown['xi_mae'] / breakdown['xi_actual'].abs()) * 100
    
    # Select and rename columns for display
    display_cols = {
        'gameweek': 'GW',
        'formation': 'Formation',
        'squad_predicted': 'Squad Pred',
        'squad_actual': 'Squad Actual',
        'Squad Error': 'Squad Err',
        'Squad Accuracy %': 'Squad Acc%',
        'xi_predicted': 'XI Pred',
        'xi_actual': 'XI Actual', 
        'XI Error': 'XI Err',
        'XI Accuracy %': 'XI Acc%',
        'captain_predicted': 'Capt Pred',
        'captain_actual': 'Capt Actual',
        'captain_error': 'Capt Err'
    }
    
    # Filter to only existing columns and rename
    existing_cols = {k: v for k, v in display_cols.items() if k in breakdown.columns}
    result = breakdown[list(existing_cols.keys())].rename(columns=existing_cols)
    
    # Round numeric columns
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    result[numeric_cols] = result[numeric_cols].round(1)
    
    return result

def show_performance_insights(summary: Dict[str, Any], team_df: pd.DataFrame):
    """Show key performance insights and recommendations."""
    if not summary:
        st.warning("No performance summary available")
        return
    
    st.subheader("🎯 Performance Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Overall Squad Accuracy",
            f"{(1 - summary['squad_metrics']['avg_mae'] / summary['squad_metrics']['avg_points_per_gw_actual']) * 100:.1f}%",
            f"MAE: {summary['squad_metrics']['avg_mae']:.1f} pts"
        )
    
    with col2:
        st.metric(
            "Starting XI Accuracy", 
            f"{(1 - summary['xi_metrics']['avg_mae'] / summary['xi_metrics']['avg_points_per_gw_actual']) * 100:.1f}%",
            f"MAE: {summary['xi_metrics']['avg_mae']:.1f} pts"
        )
    
    with col3:
        st.metric(
            "Captain Success Rate",
            f"{summary['captain_metrics']['avg_points_per_gw_actual']:.1f} pts",
            f"Error: {summary['captain_metrics']['avg_error']:+.1f} pts"
        )
    
    # Performance recommendations
    st.subheader("💡 Model Improvement Recommendations")
    
    squad_acc = (1 - summary['squad_metrics']['avg_mae'] / summary['squad_metrics']['avg_points_per_gw_actual']) * 100
    xi_acc = (1 - summary['xi_metrics']['avg_mae'] / summary['xi_metrics']['avg_points_per_gw_actual']) * 100
    
    recommendations = []
    
    if squad_acc < 80:
        recommendations.append("🔴 **Squad Accuracy Low**: Consider reviewing feature engineering or model parameters")
    elif squad_acc < 90:
        recommendations.append("🟡 **Squad Accuracy Moderate**: Good progress, fine-tune hyperparameters")
    else:
        recommendations.append("🟢 **Squad Accuracy Excellent**: Model is performing well!")
    
    if xi_acc < squad_acc - 10:
        recommendations.append("🔴 **Starting XI Selection**: Review formation optimization logic")
    
    if summary['captain_metrics']['avg_error'] < -5:
        recommendations.append("🔴 **Captain Selection**: Model overestimates captain points consistently")
    elif summary['captain_metrics']['avg_error'] > 5:
        recommendations.append("🟡 **Captain Selection**: Model underestimates captain points")
    
    if not team_df.empty and len(team_df) >= 3:
        recent_mae = team_df.tail(3)['squad_mae'].mean()
        older_mae = team_df.head(len(team_df)-3)['squad_mae'].mean() if len(team_df) > 3 else recent_mae
        
        if recent_mae > older_mae * 1.1:
            recommendations.append("🔴 **Model Degradation**: Recent accuracy is declining, retrain model")
        elif recent_mae < older_mae * 0.9:
            recommendations.append("🟢 **Model Improvement**: Accuracy is improving over time!")
    
    for rec in recommendations:
        st.markdown(rec)

def main():
    """Main performance tracking dashboard."""
    
    st.title("📊 FPL Model Performance Tracking")
    st.markdown("### Compare predicted vs actual player points to validate model accuracy")
    
    # Load data
    with st.spinner("Loading performance data..."):
        data = load_performance_data()
    
    summary = data['summary']
    team_df = data['team_performance']
    player_df = data['player_performance']
    tracker = data['tracker']
    
    if not summary:
        st.error("❌ No performance tracking data found")
        st.info("""
        **To start tracking performance:**
        1. Generate predictions using the dashboard or pipeline
        2. After gameweeks complete, run: `make perf-update`
        3. Return here to view performance analysis
        """)
        
        st.code("make perf-update  # Update all completed gameweeks")
        st.code("make perf-report  # View text report")
        return
    
    # Sidebar controls
    with st.sidebar:
        st.header("📊 Analysis Controls")
        
        # Gameweek filter
        if not team_df.empty:
            min_gw = int(team_df['gameweek'].min())
            max_gw = int(team_df['gameweek'].max())
            
            selected_gws = st.slider(
                "Gameweek Range",
                min_value=min_gw,
                max_value=max_gw,
                value=(min_gw, max_gw),
                help="Filter analysis to specific gameweeks"
            )
            
            # Filter data
            team_df = team_df[
                (team_df['gameweek'] >= selected_gws[0]) & 
                (team_df['gameweek'] <= selected_gws[1])
            ]
            player_df = player_df[
                (player_df['gameweek'] >= selected_gws[0]) & 
                (player_df['gameweek'] <= selected_gws[1])
            ]
        
        # Update performance data
        st.subheader("🔄 Data Management")
        
        if st.button("🔄 Update Performance Data", type="primary"):
            with st.spinner("Updating performance data..."):
                try:
                    if tracker:
                        # Get current gameweek range for updates
                        from src.common.timeutil import get_current_gw
                        current_gw = get_current_gw()
                        
                        updated_count = 0
                        for gw in range(1, current_gw):
                            try:
                                tracker.update_actual_results(gw)
                                updated_count += 1
                            except Exception as e:
                                logger.warning(f"Could not update GW{gw}: {e}")
                        
                        st.success(f"✅ Updated {updated_count} gameweeks")
                        st.rerun()
                    else:
                        st.error("Performance tracker not available")
                except Exception as e:
                    st.error(f"Update failed: {e}")
        
        if st.button("📄 Generate Report"):
            if tracker:
                report = tracker.generate_performance_report()
                st.text_area("Performance Report", report, height=400)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Performance Trends", 
        "📊 Position Analysis", 
        "👑 Captain Performance",
        "📋 Detailed Breakdown"
    ])
    
    with tab1:
        st.subheader("📈 Model Performance Trends")
        
        # Show key insights first
        show_performance_insights(summary, team_df)
        
        # Performance trends chart
        if not team_df.empty:
            trend_chart = create_accuracy_trend_chart(team_df)
            st.plotly_chart(trend_chart, use_container_width=True)
        else:
            st.info("No gameweek data available for trends")
    
    with tab2:
        st.subheader("📊 Position-wise Performance Analysis")
        
        if not player_df.empty:
            position_chart = create_position_accuracy_chart(player_df)
            st.plotly_chart(position_chart, use_container_width=True)
            
            # Position summary table
            st.subheader("Position Summary")
            position_stats = player_df.groupby('position').agg({
                'predicted_points': ['mean', 'std'],
                'actual_points': ['mean', 'std'],
                'absolute_error': 'mean',
                'element_id': 'count'
            }).round(2)
            
            position_stats.columns = ['Pred Mean', 'Pred Std', 'Actual Mean', 'Actual Std', 'MAE', 'Count']
            position_stats['Accuracy %'] = (1 - position_stats['MAE'] / position_stats['Actual Mean'].abs()) * 100
            position_stats = position_stats.round(1)
            
            st.dataframe(position_stats, use_container_width=True)
        else:
            st.info("No player data available for position analysis")
    
    with tab3:
        st.subheader("👑 Captain Performance Analysis")
        
        if not team_df.empty:
            captain_chart = create_captain_performance_chart(team_df)
            st.plotly_chart(captain_chart, use_container_width=True)
            
            # Captain summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_pred = team_df['captain_predicted'].mean()
                st.metric("Avg Predicted Captain Points", f"{avg_pred:.1f}")
            
            with col2:
                avg_actual = team_df['captain_actual'].mean()
                st.metric("Avg Actual Captain Points", f"{avg_actual:.1f}")
            
            with col3:
                avg_error = team_df['captain_error'].mean()
                st.metric("Avg Captain Error", f"{avg_error:+.1f}")
        else:
            st.info("No captain data available")
    
    with tab4:
        st.subheader("📋 Detailed Gameweek Breakdown")
        
        if not team_df.empty:
            breakdown_table = create_gameweek_breakdown_table(team_df)
            st.dataframe(breakdown_table, use_container_width=True)
            
            # Download option
            csv = breakdown_table.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"fpl_performance_breakdown_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No detailed breakdown data available")

if __name__ == "__main__":
    main()
