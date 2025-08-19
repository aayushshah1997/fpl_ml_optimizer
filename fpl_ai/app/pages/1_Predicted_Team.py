"""
FPL AI Dashboard - Predicted Team Page

Team prediction and optimization page with Monte Carlo analysis,
captain selection, and comparison with user team.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Optional, Any

# Add src to path for imports
app_dir = Path(__file__).parent.parent
src_dir = app_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from app._utils import (
    load_predictions, get_gameweek_selector, build_confidence_intervals,
    format_team_display, display_formation_grid, create_comparison_table,
    create_position_summary, load_config
)

# Page configuration
st.set_page_config(
    page_title="Predicted Team - FPL AI",
    page_icon="🎯",
    layout="wide"
)

def load_user_team(entry_id: Optional[int] = None) -> Optional[List[Dict]]:
    """Load user team from FPL API or uploaded data."""
    if entry_id:
        try:
            from src.providers.fpl_picks import get_user_picks
            gw = get_gameweek_selector()
            return get_user_picks(entry_id, gw)
        except Exception as e:
            st.error(f"Failed to load team from Entry ID {entry_id}: {e}")
    
    return None


def optimize_team_selection(predictions_df: pd.DataFrame, objective: str = "mean") -> Dict[str, Any]:
    """Optimize team selection using the optimizer."""
    try:
        from src.optimize.optimizer import TeamOptimizer
        
        optimizer = TeamOptimizer()
        
        # Optimize full squad
        result = optimizer.optimize_team(
            predictions_df,
            budget=100.0,
            objective=objective
        )
        
        if not result:
            return {}
        
        squad = result.get('squad', [])
        
        # Optimize starting XI
        xi_result = optimizer.optimize_starting_xi(
            squad, predictions_df, objective=objective
        )
        
        return {
            'squad': squad,
            'starting_xi': xi_result.get('starting_xi', []),
            'formation': xi_result.get('formation'),
            'captain': xi_result.get('captain'),
            'vice_captain': xi_result.get('vice_captain'),
            'expected_points': result.get('expected_points', 0),
            'total_cost': result.get('total_cost', 0),
            'bench': xi_result.get('bench', [])
        }
    
    except Exception as e:
        st.error(f"Team optimization failed: {e}")
        return {}


def main():
    """Main predicted team page."""
    
    st.title("🎯 Predicted Team")
    st.markdown("### AI-Optimized Team Selection with Risk Analysis")
    
    # Get current gameweek
    target_gw = get_gameweek_selector()
    
    # Load predictions
    predictions_df = load_predictions(target_gw)
    
    if predictions_df is None or predictions_df.empty:
        st.error(f"No predictions available for GW {target_gw}")
        st.info("Run the training pipeline to generate predictions")
        return
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Team Optimization")
        
        # Optimization objective
        objective = st.selectbox(
            "Optimization Objective",
            options=["mean", "risk_adjusted", "monte_carlo"],
            index=0,
            help="Choose optimization strategy"
        )
        
        # Monte Carlo settings
        st.subheader("🎲 Monte Carlo Settings")
        
        config = load_config()
        mc_config = config.get("mc", {})
        
        num_scenarios = st.number_input(
            "Number of Scenarios",
            min_value=100,
            max_value=10000,
            value=mc_config.get("num_scenarios", 2000),
            step=500,
            help="More scenarios = more accurate uncertainty estimates"
        )
        
        minutes_uncertainty = st.slider(
            "Minutes Uncertainty",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Uncertainty in minutes predictions (0.2 = 20%)"
        )
        
        # Risk settings
        st.subheader("⚖️ Risk Settings")
        
        risk_lambda = st.slider(
            "Risk Penalty (λ)",
            min_value=0.0,
            max_value=1.0,
            value=mc_config.get("lambda_risk", 0.2),
            step=0.05,
            help="Higher values favor lower-risk players"
        )
        
        # User team comparison
        st.subheader("👤 User Team Comparison")
        
        entry_id = st.number_input(
            "FPL Entry ID",
            min_value=1,
            value=None,
            help="Your FPL Entry ID for comparison"
        )
        
        compare_user_team = st.checkbox(
            "Compare with User Team",
            value=False,
            disabled=entry_id is None
        )
    
    # Build confidence intervals
    predictions_with_ci = build_confidence_intervals(predictions_df, minutes_uncertainty)
    
    # Training mode indicator
    config = load_config()
    staging_config = config.get("training", {}).get("staging", {})
    warm_until_gw = staging_config.get("warm_until_gw", 8)
    
    if target_gw < warm_until_gw:
        st.info(f"🌱 **Training Mode: Warm Start** (GW {target_gw} < {warm_until_gw})")
    else:
        st.success(f"🔥 **Training Mode: Full ML** (GW {target_gw} ≥ {warm_until_gw})")
    
    # Optimize team
    with st.spinner("Optimizing team selection..."):
        team_result = optimize_team_selection(predictions_with_ci, objective)
    
    if not team_result:
        st.error("Team optimization failed")
        return
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Optimal Team", "📊 Analysis", "🔍 Player Explorer", "⚖️ Comparison"])
    
    with tab1:
        # Team overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Expected Points",
                f"{team_result.get('expected_points', 0):.1f}",
                help="Total expected points for the squad"
            )
        
        with col2:
            st.metric(
                "Total Cost",
                f"£{team_result.get('total_cost', 0):.1f}M",
                help="Total squad cost"
            )
        
        with col3:
            remaining_budget = 100.0 - team_result.get('total_cost', 0)
            st.metric(
                "Remaining Budget",
                f"£{remaining_budget:.1f}M",
                help="Unused budget"
            )
        
        with col4:
            formation = team_result.get('formation', (0, 0, 0))
            st.metric(
                "Formation",
                f"{formation[0]}-{formation[1]}-{formation[2]}",
                help="Optimal formation"
            )
        
        st.divider()
        
        # Starting XI
        starting_xi = team_result.get('starting_xi', [])
        captain = team_result.get('captain', {})
        vice_captain = team_result.get('vice_captain', {})
        
        if starting_xi:
            # Formation display
            display_formation_grid(
                starting_xi,
                formation,
                captain.get('element_id'),
                vice_captain.get('element_id')
            )
            
            st.divider()
            
            # Captain analysis
            st.subheader("👑 Captain Selection")
            
            capt_col1, capt_col2 = st.columns(2)
            
            with capt_col1:
                st.markdown("**Captain:**")
                if captain:
                    capt_proj = captain.get('proj_points', 0)
                    capt_name = captain.get('web_name', 'Unknown')
                    st.markdown(f"🎯 **{capt_name}** - {capt_proj:.1f} pts (×2 = {capt_proj * 2:.1f})")
                    
                    # Confidence interval if available
                    if 'p10' in captain and 'p90' in captain:
                        p10, p90 = captain['p10'], captain['p90']
                        st.markdown(f"📈 Range: {p10 * 2:.1f} - {p90 * 2:.1f} pts (with captain)")
            
            with capt_col2:
                st.markdown("**Vice Captain:**")
                if vice_captain:
                    vc_proj = vice_captain.get('proj_points', 0)
                    vc_name = vice_captain.get('web_name', 'Unknown')
                    st.markdown(f"🥈 **{vc_name}** - {vc_proj:.1f} pts")
        
        st.divider()
        
        # Starting XI Table
        st.subheader("📋 Starting XI Details")
        
        if starting_xi:
            xi_df = pd.DataFrame(starting_xi)
            
            # Format for display
            display_cols = ['web_name', 'position', 'team_name', 'proj_points', 'now_cost']
            available_cols = [col for col in display_cols if col in xi_df.columns]
            
            if available_cols:
                xi_display = xi_df[available_cols].copy()
                
                # Add captain indicators
                if 'element_id' in xi_df.columns:
                    captain_id = captain.get('element_id')
                    vc_id = vice_captain.get('element_id')
                    
                    xi_display['Role'] = xi_df['element_id'].apply(
                        lambda x: '👑 Captain' if x == captain_id
                                else '🥈 Vice' if x == vc_id
                                else ''
                    )
                
                # Format cost
                if 'now_cost' in xi_display.columns:
                    xi_display['now_cost'] = xi_display['now_cost'].apply(
                        lambda x: f"£{x/10:.1f}M" if x > 20 else f"£{x:.1f}M"
                    )
                
                # Rename columns
                column_names = {
                    'web_name': 'Player',
                    'position': 'Pos',
                    'team_name': 'Team',
                    'proj_points': 'Projection',
                    'now_cost': 'Cost'
                }
                
                xi_display = xi_display.rename(columns=column_names)
                
                st.dataframe(
                    xi_display,
                    use_container_width=True,
                    hide_index=True
                )
        
        # Bench
        st.subheader("🪑 Bench")
        
        bench = team_result.get('bench', [])
        if bench:
            bench_df = pd.DataFrame(bench)
            
            display_cols = ['web_name', 'position', 'team_name', 'proj_points', 'now_cost']
            available_cols = [col for col in display_cols if col in bench_df.columns]
            
            if available_cols:
                bench_display = bench_df[available_cols].copy()
                
                # Format cost
                if 'now_cost' in bench_display.columns:
                    bench_display['now_cost'] = bench_display['now_cost'].apply(
                        lambda x: f"£{x/10:.1f}M" if x > 20 else f"£{x:.1f}M"
                    )
                
                bench_display = bench_display.rename(columns=column_names)
                
                st.dataframe(
                    bench_display,
                    use_container_width=True,
                    hide_index=True
                )
    
    with tab2:
        st.subheader("📊 Team Analysis")
        
        # Position summary
        position_summary = create_position_summary(predictions_with_ci)
        
        if not position_summary.empty:
            st.markdown("#### 📈 Position Summary")
            st.dataframe(position_summary, use_container_width=True, hide_index=True)
        
        # Recent form analysis
        st.markdown("#### 📈 Recent Form Analysis")
        
        form_cols = [col for col in predictions_with_ci.columns if col.startswith('r3_') or col.startswith('r5_')]
        
        if form_cols:
            st.info(f"Recent form metrics available: {len(form_cols)} features")
            
            # Show top recent performers
            if 'r3_points_per_game' in predictions_with_ci.columns:
                top_form = predictions_with_ci.nlargest(10, 'r3_points_per_game')
                
                st.markdown("**Top R3 Form Players:**")
                for _, player in top_form.iterrows():
                    name = player.get('web_name', 'Unknown')
                    r3_form = player.get('r3_points_per_game', 0)
                    proj = player.get('proj_points', 0)
                    st.markdown(f"• {name}: {r3_form:.1f} R3 PPG → {proj:.1f} projection")
        else:
            st.warning("No recent form data available")
        
        # Uncertainty analysis
        st.markdown("#### 🎲 Prediction Uncertainty")
        
        if 'std' in predictions_with_ci.columns:
            uncertainty_stats = {
                'Average Uncertainty': predictions_with_ci['std'].mean(),
                'Max Uncertainty': predictions_with_ci['std'].max(),
                'Min Uncertainty': predictions_with_ci['std'].min()
            }
            
            uncert_col1, uncert_col2, uncert_col3 = st.columns(3)
            
            with uncert_col1:
                st.metric("Avg Uncertainty", f"{uncertainty_stats['Average Uncertainty']:.2f}")
            with uncert_col2:
                st.metric("Max Uncertainty", f"{uncertainty_stats['Max Uncertainty']:.2f}")
            with uncert_col3:
                st.metric("Min Uncertainty", f"{uncertainty_stats['Min Uncertainty']:.2f}")
            
            # Most/least certain picks
            most_certain = predictions_with_ci.nsmallest(5, 'std')
            least_certain = predictions_with_ci.nlargest(5, 'std')
            
            cert_col1, cert_col2 = st.columns(2)
            
            with cert_col1:
                st.markdown("**Most Certain Picks:**")
                for _, player in most_certain.iterrows():
                    name = player.get('web_name', 'Unknown')
                    std = player.get('std', 0)
                    proj = player.get('proj_points', 0)
                    st.markdown(f"• {name}: {proj:.1f} ± {std:.2f}")
            
            with cert_col2:
                st.markdown("**Least Certain Picks:**")
                for _, player in least_certain.iterrows():
                    name = player.get('web_name', 'Unknown')
                    std = player.get('std', 0)
                    proj = player.get('proj_points', 0)
                    st.markdown(f"• {name}: {proj:.1f} ± {std:.2f}")
    
    with tab3:
        st.subheader("🔍 Player Explorer")
        
        # Filters
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            position_filter = st.selectbox(
                "Position",
                options=['All'] + ['GK', 'DEF', 'MID', 'FWD'],
                index=0
            )
        
        with filter_col2:
            min_cost = st.number_input(
                "Min Cost (£M)",
                min_value=3.5,
                max_value=15.0,
                value=4.0,
                step=0.5
            )
        
        with filter_col3:
            max_cost = st.number_input(
                "Max Cost (£M)",
                min_value=4.0,
                max_value=15.0,
                value=12.0,
                step=0.5
            )
        
        # Apply filters
        filtered_df = predictions_with_ci.copy()
        
        if position_filter != 'All':
            filtered_df = filtered_df[filtered_df.get('position') == position_filter]
        
        # Cost filtering (handle tenths conversion)
        cost_col = 'now_cost'
        if cost_col in filtered_df.columns:
            cost_values = filtered_df[cost_col]
            if cost_values.max() > 20:  # In tenths
                filtered_df = filtered_df[
                    (cost_values / 10 >= min_cost) & (cost_values / 10 <= max_cost)
                ]
            else:
                filtered_df = filtered_df[
                    (cost_values >= min_cost) & (cost_values <= max_cost)
                ]
        
        # Sort by projection
        if 'proj_points' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('proj_points', ascending=False)
        
        st.markdown(f"**{len(filtered_df)} players match your criteria**")
        
        # Display top players
        display_cols = ['web_name', 'position', 'team_name', 'proj_points', 'now_cost']
        if 'p10' in filtered_df.columns and 'p90' in filtered_df.columns:
            display_cols.extend(['p10', 'p90'])
        
        available_cols = [col for col in display_cols if col in filtered_df.columns]
        
        if available_cols and not filtered_df.empty:
            display_df = filtered_df[available_cols].head(20).copy()
            
            # Format cost
            if 'now_cost' in display_df.columns:
                display_df['now_cost'] = display_df['now_cost'].apply(
                    lambda x: f"£{x/10:.1f}M" if x > 20 else f"£{x:.1f}M"
                )
            
            # Rename columns
            column_names = {
                'web_name': 'Player',
                'position': 'Pos',
                'team_name': 'Team',
                'proj_points': 'Projection',
                'now_cost': 'Cost',
                'p10': 'P10',
                'p90': 'P90'
            }
            
            display_df = display_df.rename(columns=column_names)
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
    
    with tab4:
        st.subheader("⚖️ Team Comparison")
        
        if compare_user_team and entry_id:
            with st.spinner("Loading user team..."):
                user_team = load_user_team(entry_id)
            
            if user_team:
                st.success(f"✅ Loaded team from Entry ID {entry_id}")
                
                # Create comparison
                comparison_df = create_comparison_table(
                    team_result.get('squad', []),
                    user_team,
                    predictions_with_ci
                )
                
                if not comparison_df.empty:
                    # Summary metrics
                    ai_total = comparison_df[comparison_df['AI_Selected']]['Projection'].sum()
                    user_total = comparison_df[comparison_df['User_Selected']]['Projection'].sum()
                    both_count = (comparison_df['AI_Selected'] & comparison_df['User_Selected']).sum()
                    
                    comp_col1, comp_col2, comp_col3 = st.columns(3)
                    
                    with comp_col1:
                        st.metric("AI Team Expected", f"{ai_total:.1f} pts")
                    with comp_col2:
                        st.metric("User Team Expected", f"{user_total:.1f} pts")
                    with comp_col3:
                        st.metric("Overlapping Players", both_count)
                    
                    # Net gain
                    if ai_total > user_total:
                        st.success(f"🚀 **Net Gain**: +{ai_total - user_total:.1f} points with AI team")
                    elif user_total > ai_total:
                        st.warning(f"📉 User team projected {user_total - ai_total:.1f} points higher")
                    else:
                        st.info("🤝 Teams have similar projections")
                    
                    # Detailed comparison
                    st.markdown("#### 📊 Detailed Comparison")
                    
                    # Format for display
                    comp_display = comparison_df[['Name', 'Position', 'Team', 'Selection', 'Projection', 'Cost']].copy()
                    comp_display['Cost'] = comp_display['Cost'].apply(lambda x: f"£{x:.1f}M")
                    
                    st.dataframe(
                        comp_display,
                        use_container_width=True,
                        hide_index=True
                    )
                
            else:
                st.error("Failed to load user team")
        
        else:
            st.info("Enter your FPL Entry ID in the sidebar to compare teams")
            
            # Alternative: file upload
            st.markdown("#### 📤 Alternative: Upload Team CSV")
            
            uploaded_file = st.file_uploader(
                "Upload team CSV",
                type=['csv'],
                help="CSV with columns: element_id, web_name, position, etc."
            )
            
            if uploaded_file:
                try:
                    user_team_df = pd.read_csv(uploaded_file)
                    
                    if 'element_id' in user_team_df.columns:
                        user_team = user_team_df.to_dict('records')
                        
                        # Create comparison
                        comparison_df = create_comparison_table(
                            team_result.get('squad', []),
                            user_team,
                            predictions_with_ci
                        )
                        
                        if not comparison_df.empty:
                            st.success("✅ Team comparison created from uploaded file")
                            
                            comp_display = comparison_df[['Name', 'Position', 'Team', 'Selection', 'Projection', 'Cost']].copy()
                            comp_display['Cost'] = comp_display['Cost'].apply(lambda x: f"£{x:.1f}M")
                            
                            st.dataframe(
                                comp_display,
                                use_container_width=True,
                                hide_index=True
                            )
                    else:
                        st.error("CSV must contain 'element_id' column")
                
                except Exception as e:
                    st.error(f"Error processing uploaded file: {e}")


if __name__ == "__main__":
    main()
