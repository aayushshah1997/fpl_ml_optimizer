"""
Team Analytics module for FPL AI Dashboard.

Implements advanced analytics and insights including:
- Transfer suggestions based on ML predictions
- Risk analysis and portfolio optimization
- Performance vs. predictions tracking
- Automated transfer recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Add src to path for imports
app_dir = Path(__file__).parent
src_dir = app_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from fpl_ai.src.common.timeutil import get_current_gw
from fpl_ai.src.providers.fpl_picks import FPLPicksClient
from fpl_ai.src.providers.fpl_api import FPLAPIClient
from fpl_ai.src.optimize.optimizer import TeamOptimizer
from fpl_ai.src.plan.multiweek_planner import MultiWeekPlanner
from fpl_ai.app._utils import load_predictions, enrich_predictions_with_fpl_data, get_saved_entry_id
from fpl_ai.app.fbref_integration import FBRefIntegration


class TeamAnalytics:
    """Advanced team analytics and insights."""
    
    def __init__(self):
        """Initialize team analytics."""
        self.fpl_api = FPLAPIClient()
        self.picks_client = FPLPicksClient()
        self.optimizer = TeamOptimizer()
        self.current_gw = get_current_gw()
    
    def get_user_team_performance(self, entry_id: int, weeks_back: int = 5) -> Dict[str, Any]:
        """Get user team performance over recent weeks."""
        try:
            performance_data = []
            
            for gw in range(max(1, self.current_gw - weeks_back), self.current_gw):
                try:
                    # Get team picks for this gameweek
                    picks = self.picks_client.get_user_picks(entry_id, gw)
                    if not picks:
                        continue
                    
                    # Calculate team performance
                    total_points = sum(pick.get('points', 0) for pick in picks)
                    captain_points = 0
                    vice_captain_points = 0
                    
                    for pick in picks:
                        if pick.get('is_captain'):
                            captain_points = pick.get('points', 0)
                        elif pick.get('is_vice_captain'):
                            vice_captain_points = pick.get('points', 0)
                    
                    performance_data.append({
                        'gameweek': gw,
                        'total_points': total_points,
                        'captain_points': captain_points,
                        'vice_captain_points': vice_captain_points,
                        'bench_points': sum(pick.get('points', 0) for pick in picks if pick.get('multiplier', 0) == 0),
                        'transfers_made': len([p for p in picks if p.get('is_transfer_in', False)]),
                        'hit_taken': pick.get('event_transfers_cost', 0) if picks else 0
                    })
                    
                except Exception as e:
                    st.warning(f"Could not load data for GW {gw}: {e}")
                    continue
            
            return {
                'success': True,
                'performance_data': performance_data,
                'weeks_analyzed': len(performance_data)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'performance_data': [],
                'weeks_analyzed': 0
            }
    
    def analyze_transfer_suggestions(self, entry_id: int, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze transfer suggestions based on ML predictions."""
        try:
            # Get current team
            current_team = self.picks_client.get_user_picks(entry_id, self.current_gw)
            if not current_team:
                return {'success': False, 'error': 'Could not load current team'}
            
            # Get optimized team
            optimization_result = self.optimizer.optimize_team_selection(
                predictions_df, 
                budget=100.0, 
                objective='mean'
            )
            
            if not optimization_result or not optimization_result.get('squad'):
                return {'success': False, 'error': 'Could not generate optimized team'}
            
            # Compare current vs optimized
            current_player_ids = {pick['element'] for pick in current_team}
            optimized_player_ids = {player['element_id'] for player in optimization_result['squad']}
            
            # Find transfers
            players_out = current_player_ids - optimized_player_ids
            players_in = optimized_player_ids - current_player_ids
            
            # Get player details for transfers
            transfers_out = []
            transfers_in = []
            
            for player_id in players_out:
                player_data = next((pick for pick in current_team if pick['element'] == player_id), None)
                if player_data:
                    transfers_out.append({
                        'element_id': player_id,
                        'web_name': player_data.get('web_name', 'Unknown'),
                        'position': player_data.get('position', 'Unknown'),
                        'now_cost': player_data.get('now_cost', 0) / 10.0,
                        'current_points': player_data.get('points', 0)
                    })
            
            for player_id in players_in:
                player_data = next((player for player in optimization_result['squad'] if player['element_id'] == player_id), None)
                if player_data:
                    transfers_in.append({
                        'element_id': player_id,
                        'web_name': player_data.get('web_name', 'Unknown'),
                        'position': player_data.get('position', 'Unknown'),
                        'now_cost': player_data.get('now_cost', 0),
                        'proj_points': player_data.get('proj_points', 0),
                        'value': player_data.get('proj_points', 0) / max(player_data.get('now_cost', 1), 0.1)
                    })
            
            # Calculate transfer impact
            total_cost_in = sum(t['now_cost'] for t in transfers_in)
            total_cost_out = sum(t['now_cost'] for t in transfers_out)
            net_cost = total_cost_in - total_cost_out
            
            projected_points_gain = sum(t['proj_points'] for t in transfers_in) - sum(t['current_points'] for t in transfers_out)
            
            return {
                'success': True,
                'transfers_out': transfers_out,
                'transfers_in': transfers_in,
                'net_cost': net_cost,
                'projected_points_gain': projected_points_gain,
                'total_transfers': len(transfers_out),
                'optimization_result': optimization_result
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def calculate_risk_metrics(self, predictions_df: pd.DataFrame, team_data: List[Dict]) -> Dict[str, Any]:
        """Calculate risk analysis and portfolio optimization metrics."""
        try:
            if not team_data:
                return {'success': False, 'error': 'No team data available'}
            
            # Get team player predictions
            team_player_ids = {player['element'] for player in team_data}
            team_predictions = predictions_df[predictions_df['element_id'].isin(team_player_ids)].copy()
            
            if team_predictions.empty:
                return {'success': False, 'error': 'No predictions available for team players'}
            
            # Calculate risk metrics
            total_proj_points = team_predictions['proj_points'].sum()
            total_uncertainty = team_predictions['std'].sum() if 'std' in team_predictions.columns else 0
            
            # Position diversification
            position_counts = team_predictions['position'].value_counts().to_dict()
            
            # Team diversification (max 3 players per team)
            team_counts = team_predictions['team_id'].value_counts().to_dict()
            over_represented_teams = {team: count for team, count in team_counts.items() if count > 3}
            
            # Value analysis
            team_predictions['value'] = team_predictions['proj_points'] / team_predictions['now_cost']
            avg_value = team_predictions['value'].mean()
            
            # Risk concentration
            top_3_players = team_predictions.nlargest(3, 'proj_points')
            concentration_risk = top_3_players['proj_points'].sum() / total_proj_points if total_proj_points > 0 else 0
            
            # Fixture difficulty analysis
            upcoming_difficulty = 3.0  # Default neutral
            if 'fixture_difficulty' in team_predictions.columns:
                upcoming_difficulty = team_predictions['fixture_difficulty'].mean()
            
            return {
                'success': True,
                'total_proj_points': total_proj_points,
                'total_uncertainty': total_uncertainty,
                'position_counts': position_counts,
                'team_counts': team_counts,
                'over_represented_teams': over_represented_teams,
                'avg_value': avg_value,
                'concentration_risk': concentration_risk,
                'upcoming_difficulty': upcoming_difficulty,
                'risk_score': self._calculate_risk_score(team_predictions)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _calculate_risk_score(self, team_predictions: pd.DataFrame) -> float:
        """Calculate overall risk score (0-100, higher = riskier)."""
        try:
            risk_factors = []
            
            # Uncertainty risk
            if 'std' in team_predictions.columns:
                uncertainty_risk = (team_predictions['std'].mean() / team_predictions['proj_points'].mean()) * 50
                risk_factors.append(min(uncertainty_risk, 50))
            
            # Concentration risk
            top_3_share = team_predictions.nlargest(3, 'proj_points')['proj_points'].sum() / team_predictions['proj_points'].sum()
            concentration_risk = (top_3_share - 0.3) * 100  # Penalty if top 3 > 30%
            risk_factors.append(max(concentration_risk, 0))
            
            # Team diversification risk
            team_counts = team_predictions['team_id'].value_counts()
            over_represented = (team_counts > 3).sum()
            diversification_risk = over_represented * 10
            risk_factors.append(min(diversification_risk, 30))
            
            # Value risk (penalty for low-value players)
            if 'value' in team_predictions.columns:
                low_value_players = (team_predictions['value'] < 0.5).sum()
                value_risk = (low_value_players / len(team_predictions)) * 20
                risk_factors.append(min(value_risk, 20))
            
            return min(sum(risk_factors), 100)
            
        except Exception:
            return 50.0  # Default moderate risk
    
    def track_performance_vs_predictions(self, entry_id: int, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """Track actual performance vs ML predictions."""
        try:
            # Get recent performance data
            performance_data = self.get_user_team_performance(entry_id, weeks_back=5)
            
            if not performance_data['success'] or not performance_data['performance_data']:
                return {'success': False, 'error': 'No performance data available'}
            
            # Get historical predictions for comparison
            comparison_data = []
            
            for gw_data in performance_data['performance_data']:
                gw = gw_data['gameweek']
                
                # Try to load predictions for this gameweek
                try:
                    gw_predictions = load_predictions(gw)
                    if gw_predictions is not None and not gw_predictions.empty:
                        # Get team picks for this GW
                        picks = self.picks_client.get_user_picks(entry_id, gw)
                        if picks:
                            team_player_ids = {pick['element'] for pick in picks}
                            team_predictions = gw_predictions[gw_predictions['element_id'].isin(team_player_ids)]
                            
                            if not team_predictions.empty:
                                predicted_points = team_predictions['proj_points'].sum()
                                actual_points = gw_data['total_points']
                                
                                comparison_data.append({
                                    'gameweek': gw,
                                    'predicted_points': predicted_points,
                                    'actual_points': actual_points,
                                    'difference': actual_points - predicted_points,
                                    'accuracy': 1 - abs(actual_points - predicted_points) / max(predicted_points, 1)
                                })
                except Exception:
                    continue
            
            if not comparison_data:
                return {'success': False, 'error': 'No comparison data available'}
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Calculate tracking metrics
            avg_accuracy = comparison_df['accuracy'].mean()
            avg_difference = comparison_df['difference'].mean()
            prediction_bias = comparison_df['difference'].mean()  # Positive = over-predicted
            
            return {
                'success': True,
                'comparison_data': comparison_data,
                'avg_accuracy': avg_accuracy,
                'avg_difference': avg_difference,
                'prediction_bias': prediction_bias,
                'weeks_tracked': len(comparison_data)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


def display_team_analytics(entry_id: int, predictions_df: pd.DataFrame):
    """Display comprehensive team analytics."""
    
    analytics = TeamAnalytics()
    
    # Create tabs for different analytics
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔄 Transfer Suggestions", 
        "📊 Risk Analysis", 
        "📈 Performance Tracking",
        "🎯 Automated Recommendations",
        "⚽ FBRef Analytics"
    ])
    
    with tab1:
        st.subheader("🔄 Transfer Suggestions")
        
        with st.spinner("Analyzing transfer opportunities..."):
            transfer_analysis = analytics.analyze_transfer_suggestions(entry_id, predictions_df)
        
        if transfer_analysis['success']:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Players to Transfer Out:**")
                if transfer_analysis['transfers_out']:
                    for player in transfer_analysis['transfers_out']:
                        st.markdown(f"• {player['web_name']} ({player['position']}) - {player['current_points']:.1f} pts")
                else:
                    st.info("No suggested transfers out")
            
            with col2:
                st.markdown("**Players to Transfer In:**")
                if transfer_analysis['transfers_in']:
                    for player in transfer_analysis['transfers_in']:
                        st.markdown(f"• {player['web_name']} ({player['position']}) - {player['proj_points']:.1f} pts (Value: {player['value']:.2f})")
                else:
                    st.info("No suggested transfers in")
            
            # Transfer impact
            st.markdown("**Transfer Impact:**")
            impact_col1, impact_col2, impact_col3 = st.columns(3)
            
            with impact_col1:
                st.metric("Net Cost", f"£{transfer_analysis['net_cost']:.1f}m")
            with impact_col2:
                st.metric("Projected Points Gain", f"{transfer_analysis['projected_points_gain']:.1f}")
            with impact_col3:
                st.metric("Total Transfers", transfer_analysis['total_transfers'])
        
        else:
            st.error(f"Transfer analysis failed: {transfer_analysis['error']}")
    
    with tab2:
        st.subheader("📊 Risk Analysis & Portfolio Optimization")
        
        with st.spinner("Calculating risk metrics..."):
            # Get current team data
            picks_client = FPLPicksClient()
            current_team = picks_client.get_user_picks(entry_id, analytics.current_gw)
            
            if current_team:
                risk_analysis = analytics.calculate_risk_metrics(predictions_df, current_team)
                
                if risk_analysis['success']:
                    # Risk overview
                    risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
                    
                    with risk_col1:
                        st.metric("Risk Score", f"{risk_analysis['risk_score']:.1f}/100")
                    with risk_col2:
                        st.metric("Total Uncertainty", f"±{risk_analysis['total_uncertainty']:.1f}")
                    with risk_col3:
                        st.metric("Concentration Risk", f"{risk_analysis['concentration_risk']:.1%}")
                    with risk_col4:
                        st.metric("Avg Value", f"{risk_analysis['avg_value']:.2f}")
                    
                    # Position distribution
                    st.markdown("**Position Distribution:**")
                    pos_data = risk_analysis['position_counts']
                    if pos_data:
                        pos_df = pd.DataFrame(list(pos_data.items()), columns=['Position', 'Count'])
                        fig = px.bar(pos_df, x='Position', y='Count', title="Players by Position")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Team diversification
                    st.markdown("**Team Diversification:**")
                    team_data = risk_analysis['team_counts']
                    if team_data:
                        team_df = pd.DataFrame(list(team_data.items()), columns=['Team', 'Players'])
                        team_df = team_df.sort_values('Players', ascending=False)
                        
                        # Color code over-represented teams
                        colors = ['red' if count > 3 else 'blue' for count in team_df['Players']]
                        fig = px.bar(team_df, x='Team', y='Players', color=colors, 
                                   title="Players per Team (Red = Over-represented)")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk warnings
                    if risk_analysis['over_represented_teams']:
                        st.warning("⚠️ Over-represented teams detected (max 3 players per team)")
                        for team, count in risk_analysis['over_represented_teams'].items():
                            st.write(f"• Team {team}: {count} players")
                
                else:
                    st.error(f"Risk analysis failed: {risk_analysis['error']}")
            else:
                st.error("Could not load current team data")
    
    with tab3:
        st.subheader("📈 Performance vs. Predictions Tracking")
        
        with st.spinner("Loading performance tracking data..."):
            tracking_data = analytics.track_performance_vs_predictions(entry_id, predictions_df)
        
        if tracking_data['success'] and tracking_data['comparison_data']:
            # Performance metrics
            perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
            
            with perf_col1:
                st.metric("Avg Accuracy", f"{tracking_data['avg_accuracy']:.1%}")
            with perf_col2:
                st.metric("Avg Difference", f"{tracking_data['avg_difference']:.1f} pts")
            with perf_col3:
                st.metric("Prediction Bias", f"{tracking_data['prediction_bias']:.1f} pts")
            with perf_col4:
                st.metric("Weeks Tracked", tracking_data['weeks_tracked'])
            
            # Performance chart
            comparison_df = pd.DataFrame(tracking_data['comparison_data'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=comparison_df['gameweek'],
                y=comparison_df['predicted_points'],
                mode='lines+markers',
                name='Predicted',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=comparison_df['gameweek'],
                y=comparison_df['actual_points'],
                mode='lines+markers',
                name='Actual',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title="Predicted vs Actual Points",
                xaxis_title="Gameweek",
                yaxis_title="Points",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Accuracy trend
            comparison_df['rolling_accuracy'] = comparison_df['accuracy'].rolling(window=3, min_periods=1).mean()
            
            fig2 = px.line(comparison_df, x='gameweek', y='rolling_accuracy', 
                          title="Rolling 3-GW Accuracy Trend")
            fig2.add_hline(y=0.8, line_dash="dash", line_color="green", 
                          annotation_text="80% Accuracy Target")
            st.plotly_chart(fig2, use_container_width=True)
        
        else:
            st.error(f"Performance tracking failed: {tracking_data.get('error', 'No data available')}")
    
    with tab4:
        st.subheader("🎯 Automated Transfer Recommendations")
        
        st.info("🤖 **AI-Powered Transfer Engine**")
        
        # Multi-week planning
        with st.expander("📅 10-Week Transfer Plan", expanded=True):
            st.markdown("""
            **Automated Transfer Strategy:**
            - Analyzes upcoming fixtures and form
            - Considers price changes and value opportunities  
            - Optimizes for long-term points accumulation
            - Balances risk vs reward across multiple gameweeks
            """)
            
            if st.button("Generate 10-Week Plan", type="primary"):
                with st.spinner("Generating comprehensive transfer plan..."):
                    try:
                        planner = MultiWeekPlanner()
                        plan_result = planner.run_transfer_planning(
                            entry_id=entry_id,
                            current_gw=analytics.current_gw,
                            horizon_gw=min(analytics.current_gw + 10, 38)
                        )
                        
                        if plan_result and plan_result.get('success'):
                            st.success("✅ 10-week transfer plan generated!")
                            
                            # Display plan summary
                            if 'summary' in plan_result:
                                st.json(plan_result['summary'])
                        else:
                            st.error("Failed to generate transfer plan")
                    except Exception as e:
                        st.error(f"Error generating plan: {e}")
        
        # Immediate recommendations
        with st.expander("⚡ Immediate Recommendations", expanded=True):
            st.markdown("""
            **This Week's Priority Actions:**
            - Based on current form and upcoming fixtures
            - Considers budget constraints and transfer costs
            - Optimizes for next 3 gameweeks
            """)
            
            if st.button("Get Immediate Recommendations", type="secondary"):
                with st.spinner("Analyzing immediate opportunities..."):
                    # This would integrate with the transfer analysis
                    st.info("Immediate recommendations would be generated here based on current analysis")
        
        # Risk alerts
        with st.expander("⚠️ Risk Alerts", expanded=True):
            st.markdown("""
            **Current Risk Factors:**
            - Monitor over-represented teams
            - Watch for fixture difficulty spikes
            - Track player form and injury status
            """)
            
            # This would show real-time risk alerts
            st.info("Risk monitoring system would display current alerts here")
    
    with tab5:
        st.subheader("⚽ FBRef Analytics")
        
        # Initialize FBRef integration
        fbref = FBRefIntegration()
        
        if fbref.fbref_data is not None:
            st.success(f"✅ FBRef data loaded: {len(fbref.fbref_data)} players")
            
            # Display FBRef analytics
            from fpl_ai.app.fbref_integration import display_fbref_analytics
            display_fbref_analytics(predictions_df)
        else:
            st.warning("FBRef data not available. Please ensure data integration is complete.")
            st.info("""
            **To enable FBRef analytics:**
            1. Ensure the FBRef Excel file is in the project root
            2. Run the data integration script
            3. Refresh the dashboard
            """)
