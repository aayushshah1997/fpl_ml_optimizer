"""
FBRef Data Integration for FPL AI Dashboard.

Integrates FBRef per-90 statistics with FPL predictions and team analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import sys

# Add src to path for imports
app_dir = Path(__file__).parent
src_dir = app_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from fpl_ai.app._utils import load_predictions, enrich_predictions_with_fpl_data, get_saved_entry_id


class FBRefIntegration:
    """Integrates FBRef per-90 data with FPL AI system."""
    
    def __init__(self):
        """Initialize FBRef integration."""
        self.fbref_data = None
        self.load_fbref_data()
    
    def load_fbref_data(self) -> bool:
        """Load FBRef data from CSV files."""
        try:
            # Try to load enhanced data first
            enhanced_path = Path(__file__).parent.parent / "fbref_enhanced.csv"
            if enhanced_path.exists():
                self.fbref_data = pd.read_csv(enhanced_path)
                st.success(f"✅ Loaded FBRef data: {len(self.fbref_data)} players")
                return True
            else:
                st.warning("FBRef enhanced data not found. Run data integration first.")
                return False
        except Exception as e:
            st.error(f"Error loading FBRef data: {e}")
            return False
    
    def get_player_fbref_stats(self, player_name: str, team: str = None) -> Optional[Dict]:
        """Get FBRef statistics for a specific player."""
        if self.fbref_data is None:
            return None
        
        # Search for player by name and optionally team
        mask = self.fbref_data['Player'].str.contains(player_name, case=False, na=False)
        if team:
            mask = mask & self.fbref_data['Squad'].str.contains(team, case=False, na=False)
        
        player_data = self.fbref_data[mask]
        
        if player_data.empty:
            return None
        
        # Return the first match
        player_row = player_data.iloc[0]
        
        return {
            'player_name': player_row['Player'],
            'team': player_row['Squad'],
            'position': player_row['FPL_Pos'],
            'minutes_90s': player_row['90s'],
            'goals': player_row['Gls'],
            'assists': player_row['Ast'],
            'goals_assists': player_row['G+A'],
            'expected_goals': player_row['xG'],
            'expected_assists': player_row['xAG'],
            'expected_goals_assists': player_row['xG+xAG'],
            'goals_per_90': player_row['Goals_per_90'],
            'assists_per_90': player_row['Assists_per_90'],
            'ga_per_90': player_row['G+A_per_90'],
            'xg_per_90': player_row['xG_per_90'],
            'xa_per_90': player_row['xA_per_90'],
            'xga_per_90': player_row['xG+xA_per_90']
        }
    
    def enhance_predictions_with_fbref(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Enhance FPL predictions with FBRef per-90 statistics."""
        if self.fbref_data is None or predictions_df.empty:
            return predictions_df
        
        enhanced_df = predictions_df.copy()
        
        # Create a mapping from FPL data to FBRef data
        fbref_stats = []
        
        for _, row in enhanced_df.iterrows():
            player_name = row.get('web_name', '')
            team_name = row.get('team_name', '')
            
            # Try to find matching FBRef data
            fbref_match = self.get_player_fbref_stats(player_name, team_name)
            
            if fbref_match:
                fbref_stats.append(fbref_match)
            else:
                # Add empty stats if no match found
                fbref_stats.append({
                    'player_name': player_name,
                    'team': team_name,
                    'position': row.get('position', 'Unknown'),
                    'minutes_90s': 0,
                    'goals': 0,
                    'assists': 0,
                    'goals_assists': 0,
                    'expected_goals': 0,
                    'expected_assists': 0,
                    'expected_goals_assists': 0,
                    'goals_per_90': 0,
                    'assists_per_90': 0,
                    'ga_per_90': 0,
                    'xg_per_90': 0,
                    'xa_per_90': 0,
                    'xga_per_90': 0
                })
        
        # Add FBRef columns to predictions
        fbref_df = pd.DataFrame(fbref_stats)
        
        # Merge with predictions
        for col in ['goals_per_90', 'assists_per_90', 'ga_per_90', 'xg_per_90', 'xa_per_90', 'xga_per_90']:
            if col in fbref_df.columns:
                enhanced_df[f'fbref_{col}'] = fbref_df[col]
        
        return enhanced_df
    
    def get_position_analysis(self) -> Dict[str, Any]:
        """Get position-based analysis from FBRef data."""
        if self.fbref_data is None:
            return {}
        
        analysis = {}
        
        for position in ['GK', 'DEF', 'MID', 'FWD']:
            pos_data = self.fbref_data[self.fbref_data['FPL_Pos'] == position]
            
            if not pos_data.empty:
                analysis[position] = {
                    'count': len(pos_data),
                    'avg_goals_per_90': pos_data['Goals_per_90'].mean(),
                    'avg_assists_per_90': pos_data['Assists_per_90'].mean(),
                    'avg_ga_per_90': pos_data['G+A_per_90'].mean(),
                    'avg_xg_per_90': pos_data['xG_per_90'].mean(),
                    'avg_xa_per_90': pos_data['xA_per_90'].mean(),
                    'avg_xga_per_90': pos_data['xG+xA_per_90'].mean(),
                    'top_scorer': pos_data.loc[pos_data['Goals_per_90'].idxmax(), 'Player'] if 'Goals_per_90' in pos_data.columns else 'N/A',
                    'top_assister': pos_data.loc[pos_data['Assists_per_90'].idxmax(), 'Player'] if 'Assists_per_90' in pos_data.columns else 'N/A'
                }
        
        return analysis
    
    def get_team_analysis(self) -> Dict[str, Any]:
        """Get team-based analysis from FBRef data."""
        if self.fbref_data is None:
            return {}
        
        team_stats = self.fbref_data.groupby('Squad').agg({
            'Player': 'count',
            'Goals_per_90': 'mean',
            'Assists_per_90': 'mean',
            'G+A_per_90': 'mean',
            'xG_per_90': 'mean',
            'xA_per_90': 'mean',
            'xG+xA_per_90': 'mean'
        }).round(3)
        
        return team_stats.to_dict('index')
    
    def get_top_performers(self, metric: str = 'ga_per_90', limit: int = 10) -> pd.DataFrame:
        """Get top performers by specified metric."""
        if self.fbref_data is None:
            return pd.DataFrame()
        
        metric_col = f'{metric}_per_90' if not metric.endswith('_per_90') else metric
        
        if metric_col not in self.fbref_data.columns:
            return pd.DataFrame()
        
        top_performers = self.fbref_data.nlargest(limit, metric_col)[
            ['Player', 'Squad', 'FPL_Pos', metric_col, '90s']
        ].copy()
        
        return top_performers


def display_fbref_analytics(predictions_df: pd.DataFrame):
    """Display FBRef analytics in the dashboard."""
    
    fbref = FBRefIntegration()
    
    if fbref.fbref_data is None:
        st.error("FBRef data not available. Please ensure data integration is complete.")
        return
    
    # Create tabs for different FBRef analytics
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Player Performance", 
        "⚽ Position Analysis", 
        "🏆 Team Comparison",
        "🎯 Enhanced Predictions"
    ])
    
    with tab1:
        st.subheader("📊 Top Performers (Per 90 Minutes)")
        
        # Top goal scorers
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top Goal Scorers:**")
            top_goals = fbref.get_top_performers('goals', 10)
            if not top_goals.empty:
                st.dataframe(top_goals, use_container_width=True)
        
        with col2:
            st.markdown("**Top Assist Providers:**")
            top_assists = fbref.get_top_performers('assists', 10)
            if not top_assists.empty:
                st.dataframe(top_assists, use_container_width=True)
        
        # Top expected contributors
        st.markdown("**Top Expected Goal Contributors:**")
        top_xg = fbref.get_top_performers('xg', 10)
        if not top_xg.empty:
            st.dataframe(top_xg, use_container_width=True)
    
    with tab2:
        st.subheader("⚽ Position Analysis")
        
        pos_analysis = fbref.get_position_analysis()
        
        if pos_analysis:
            # Create position comparison chart
            pos_data = []
            for pos, stats in pos_analysis.items():
                pos_data.append({
                    'Position': pos,
                    'Goals per 90': stats['avg_goals_per_90'],
                    'Assists per 90': stats['avg_assists_per_90'],
                    'G+A per 90': stats['avg_ga_per_90'],
                    'xG per 90': stats['avg_xg_per_90'],
                    'xA per 90': stats['avg_xa_per_90']
                })
            
            pos_df = pd.DataFrame(pos_data)
            
            # Goals and assists comparison
            fig = px.bar(pos_df, x='Position', y=['Goals per 90', 'Assists per 90'], 
                        title="Goals and Assists per 90 by Position",
                        barmode='group')
            st.plotly_chart(fig, use_container_width=True)
            
            # Expected goals and assists comparison
            fig2 = px.bar(pos_df, x='Position', y=['xG per 90', 'xA per 90'], 
                         title="Expected Goals and Assists per 90 by Position",
                         barmode='group')
            st.plotly_chart(fig2, use_container_width=True)
            
            # Position summary table
            st.markdown("**Position Summary:**")
            summary_data = []
            for pos, stats in pos_analysis.items():
                summary_data.append({
                    'Position': pos,
                    'Players': stats['count'],
                    'Avg Goals/90': f"{stats['avg_goals_per_90']:.3f}",
                    'Avg Assists/90': f"{stats['avg_assists_per_90']:.3f}",
                    'Top Scorer': stats['top_scorer'],
                    'Top Assister': stats['top_assister']
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    with tab3:
        st.subheader("🏆 Team Comparison")
        
        team_analysis = fbref.get_team_analysis()
        
        if team_analysis:
            # Create team comparison chart
            team_data = []
            for team, stats in team_analysis.items():
                team_data.append({
                    'Team': team,
                    'Players': stats['Player'],
                    'Goals per 90': stats['Goals_per_90'],
                    'Assists per 90': stats['Assists_per_90'],
                    'G+A per 90': stats['G+A_per_90']
                })
            
            team_df = pd.DataFrame(team_data)
            
            # Top 10 teams by G+A per 90
            top_teams = team_df.nlargest(10, 'G+A per 90')
            
            fig = px.bar(top_teams, x='Team', y='G+A per 90', 
                        title="Top 10 Teams by Goals + Assists per 90")
            st.plotly_chart(fig, use_container_width=True)
            
            # Team comparison table
            st.markdown("**Team Performance Summary:**")
            st.dataframe(team_df.sort_values('G+A per 90', ascending=False), 
                        use_container_width=True, hide_index=True)
    
    with tab4:
        st.subheader("🎯 Enhanced Predictions with FBRef Data")
        
        # Enhance predictions with FBRef data
        enhanced_predictions = fbref.enhance_predictions_with_fbref(predictions_df)
        
        if 'fbref_ga_per_90' in enhanced_predictions.columns:
            st.success("✅ Predictions enhanced with FBRef per-90 statistics")
            
            # Show enhanced predictions
            enhanced_cols = ['web_name', 'position', 'team_name', 'proj_points', 
                           'fbref_goals_per_90', 'fbref_assists_per_90', 'fbref_ga_per_90',
                           'fbref_xg_per_90', 'fbref_xa_per_90', 'fbref_xga_per_90']
            
            available_enhanced_cols = [col for col in enhanced_cols if col in enhanced_predictions.columns]
            
            if available_enhanced_cols:
                st.markdown("**Enhanced Predictions Sample:**")
                st.dataframe(enhanced_predictions[available_enhanced_cols].head(20), 
                           use_container_width=True, hide_index=True)
            
            # Correlation analysis
            st.markdown("**Correlation Analysis:**")
            numeric_cols = enhanced_predictions.select_dtypes(include=[np.number]).columns
            fbref_cols = [col for col in numeric_cols if col.startswith('fbref_')]
            
            if len(fbref_cols) > 1:
                corr_matrix = enhanced_predictions[fbref_cols].corr()
                
                fig = px.imshow(corr_matrix, 
                              title="FBRef Metrics Correlation Matrix",
                              color_continuous_scale='RdBu')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Could not enhance predictions with FBRef data")





