"""
FPL AI Dashboard - 10-Week Transfer Planner Page

Multi-week transfer planning with GW1 baseline initialization,
chip optimization, and comprehensive transfer analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
from typing import Dict, Any, List, Optional

# Add src to path for imports
app_dir = Path(__file__).parent.parent
src_dir = app_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from app._utils import (
    get_gameweek_selector, load_predictions, get_artifacts_dir, load_config
)

# Page configuration
st.set_page_config(
    page_title="10-Week Planner - FPL AI",
    page_icon="📅",
    layout="wide"
)


def load_team_state(filename: str) -> Optional[Dict[str, Any]]:
    """Load team state from cache."""
    try:
        cache_dir = get_artifacts_dir() / "cache"
        filepath = cache_dir / filename
        
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Could not load team state: {e}")
    
    return None


def save_team_state(state: Dict[str, Any], filename: str) -> None:
    """Save team state to cache."""
    try:
        cache_dir = get_artifacts_dir() / "cache"
        cache_dir.mkdir(exist_ok=True)
        
        filepath = cache_dir / filename
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        st.success(f"Team state saved to {filename}")
    except Exception as e:
        st.error(f"Failed to save team state: {e}")


def initialize_team_from_entry_id(entry_id: int, bank: float = 0.0) -> Optional[Dict[str, Any]]:
    """Initialize team from FPL Entry ID."""
    try:
        from src.plan.multiweek_planner import MultiWeekPlanner
        
        planner = MultiWeekPlanner()
        team_state = planner.initialize_from_gw1_team(
            entry_id=entry_id,
            bank=bank
        )
        
        return team_state
    except Exception as e:
        st.error(f"Failed to initialize team from Entry ID {entry_id}: {e}")
        return None


def initialize_team_from_csv(uploaded_file, bank: float = 0.0) -> Optional[Dict[str, Any]]:
    """Initialize team from uploaded CSV."""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check required columns
        required_cols = ['element_id']
        if not all(col in df.columns for col in required_cols):
            st.error(f"CSV must contain columns: {required_cols}")
            return None
        
        team_data = df.to_dict('records')
        
        from src.plan.multiweek_planner import MultiWeekPlanner
        
        planner = MultiWeekPlanner()
        team_state = planner.initialize_from_gw1_team(
            gw1_team=team_data,
            bank=bank
        )
        
        return team_state
    except Exception as e:
        st.error(f"Failed to initialize team from CSV: {e}")
        return None


def run_transfer_planning(
    start_gw: int,
    horizon: int,
    team_state: Dict[str, Any],
    overrides: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Run transfer planning."""
    try:
        from src.plan.multiweek_planner import MultiWeekPlanner
        
        planner = MultiWeekPlanner()
        
        # Load predictions for horizon
        predictions_by_gw = {}
        for gw in range(start_gw, start_gw + horizon + 1):
            pred_df = load_predictions(gw)
            if pred_df is not None:
                predictions_by_gw[gw] = pred_df
        
        if not predictions_by_gw:
            st.error("No predictions available for planning horizon")
            return None
        
        # Run planning
        plan_result = planner.plan_transfers(
            start_gw=start_gw,
            predictions_by_gw=predictions_by_gw,
            current_state=team_state,
            overrides=overrides
        )
        
        return plan_result
    
    except Exception as e:
        st.error(f"Transfer planning failed: {e}")
        return None


def format_transfer_step(step: Dict[str, Any]) -> str:
    """Format a transfer planning step for display."""
    gw = step.get('gameweek', '?')
    action = step.get('action', {})
    score = step.get('score', 0)
    gw_value = step.get('gw_value', 0)
    
    action_type = action.get('action', 'unknown')
    
    if action_type == 'roll':
        return f"**GW {gw}:** Roll FT (Expected: {gw_value:.1f} pts)"
    
    elif action_type == 'single_transfer':
        transfers = action.get('transfers', [])
        if transfers:
            transfer = transfers[0]
            out_name = transfer.get('out', {}).get('web_name', 'Unknown')
            in_name = transfer.get('in', {}).get('web_name', 'Unknown')
            points_gain = transfer.get('points_gain', 0)
            cost = transfer.get('cost', 0)
            hits = action.get('hits', 0)
            
            hit_text = f" (Hit: -{hits * 4})" if hits > 0 else ""
            return f"**GW {gw}:** {out_name} → {in_name} (+{points_gain:.1f}, £{cost:+.1f}M){hit_text}"
    
    elif action_type == 'double_transfer':
        transfers = action.get('transfers', [])
        if len(transfers) >= 2:
            t1, t2 = transfers[0], transfers[1]
            out1_name = t1.get('out', {}).get('web_name', 'Unknown')
            in1_name = t1.get('in', {}).get('web_name', 'Unknown')
            out2_name = t2.get('out', {}).get('web_name', 'Unknown')
            in2_name = t2.get('in', {}).get('web_name', 'Unknown')
            total_gain = sum(t.get('points_gain', 0) for t in transfers)
            hits = action.get('hits', 0)
            
            hit_text = f" (Hit: -{hits * 4})" if hits > 0 else ""
            return f"**GW {gw}:** {out1_name} → {in1_name}, {out2_name} → {in2_name} (+{total_gain:.1f}){hit_text}"
    
    elif action_type in ['free_hit', 'wildcard', 'triple_captain', 'bench_boost']:
        chip_value = action.get('chip_value', 0)
        chip_name = action_type.replace('_', ' ').title()
        return f"**GW {gw}:** {chip_name} (+{chip_value:.1f} pts)"
    
    else:
        return f"**GW {gw}:** {action_type} (Score: {score:.1f})"


def display_team_state(state: Dict[str, Any]) -> None:
    """Display current team state."""
    if not state:
        st.warning("No team state available")
        return
    
    squad = state.get('squad', [])
    bank = state.get('bank', 0)
    fts = state.get('free_transfers', 0)
    gw = state.get('gameweek', 1)
    chips = state.get('chips_available', [])
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Squad Size", len(squad))
    with col2:
        st.metric("Bank", f"£{bank:.1f}M")
    with col3:
        st.metric("Free Transfers", fts)
    with col4:
        st.metric("Gameweek", gw)
    
    # Chips available
    if chips:
        st.markdown(f"**Chips Available:** {', '.join(chips)}")
    else:
        st.markdown("**Chips Available:** None")
    
    # Squad details
    if squad:
        st.markdown("#### 👥 Current Squad")
        
        # Group by position
        by_position = {'GK': [], 'DEF': [], 'MID': [], 'FWD': []}
        
        for player in squad:
            position = player.get('position', 'UNKNOWN')
            if position in by_position:
                by_position[position].append(player)
        
        for position in ['GK', 'DEF', 'MID', 'FWD']:
            if by_position[position]:
                st.markdown(f"**{position}s:**")
                
                for player in by_position[position]:
                    name = player.get('web_name', 'Unknown')
                    cost = player.get('now_cost', 0)
                    if cost > 20:
                        cost /= 10
                    
                    st.markdown(f"• {name} (£{cost:.1f}M)")


def main():
    """Main 10-week planner page."""
    
    st.title("📅 10-Week Transfer Planner")
    st.markdown("### Advanced Multi-Week Transfer Optimization")
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Planning Settings")
        
        # Planning parameters
        start_gw = st.number_input(
            "Starting Gameweek",
            min_value=1,
            max_value=38,
            value=get_gameweek_selector(),
            help="Starting gameweek for planning"
        )
        
        horizon = st.slider(
            "Planning Horizon",
            min_value=1,
            max_value=15,
            value=10,
            help="Number of gameweeks to plan ahead"
        )
        
        # Current state overrides
        st.subheader("💰 Current State")
        
        bank = st.number_input(
            "Bank Balance (£M)",
            min_value=0.0,
            max_value=50.0,
            value=0.0,
            step=0.1,
            help="Current bank balance"
        )
        
        free_transfers = st.number_input(
            "Free Transfers",
            min_value=0,
            max_value=2,
            value=1,
            help="Current free transfers available"
        )
        
        # Planning preferences
        st.subheader("⚙️ Transfer Strategy")
        
        config = load_config()
        planner_config = config.get("planner", {})
        strategy_config = config.get("strategy", {})
        
        max_transfers_per_gw = st.slider(
            "Max Transfers per GW",
            min_value=1,
            max_value=3,
            value=strategy_config.get("max_transfers_per_gw", 2),
            help="Maximum transfers per gameweek"
        )
        
        roll_threshold = st.slider(
            "Roll Threshold (points)",
            min_value=0.0,
            max_value=5.0,
            value=strategy_config.get("roll_threshold_points", 2.0),
            step=0.5,
            help="If expected gain < threshold, roll free transfer instead"
        )
        
        bank_future_weight = st.slider(
            "Bank Future Weight",
            min_value=0.0,
            max_value=0.5,
            value=strategy_config.get("bank_future_weight", 0.25),
            step=0.05,
            help="Utility weight for keeping money for future upgrades"
        )
        
        hit_cost = st.selectbox(
            "Hit Cost",
            options=[4, 8],
            index=0 if strategy_config.get("hit_cost", 4) == 4 else 1,
            help="Point penalty per hit (some leagues use 8)"
        )
        
        shortlist_per_pos = st.slider(
            "Candidates per Position",
            min_value=5,
            max_value=50,
            value=strategy_config.get("shortlist", {}).get("per_pos", 20),
            help="Size of candidate pool per position"
        )
        
        # Captain policy settings
        st.subheader("👑 Captain Policy")
        
        captain_config = config.get("captain", {})
        
        captain_policy = st.selectbox(
            "Captain Selection",
            options=["mean", "cvar", "mix"],
            index=["mean", "cvar", "mix"].index(captain_config.get("policy", "mix")),
            help="Strategy for captain selection: mean (highest expected), cvar (risk-adjusted), mix (weighted combination)"
        )
        
        captain_candidates = st.slider(
            "Captain Candidates",
            min_value=3,
            max_value=10,
            value=captain_config.get("candidates", 5),
            help="Top N players considered for captaincy"
        )
        
        if captain_policy == "mix":
            captain_mix_lambda = st.slider(
                "Mix Weight (Mean vs CVaR)",
                min_value=0.0,
                max_value=1.0,
                value=captain_config.get("mix_lambda", 0.6),
                step=0.1,
                help="Weight on mean vs CVaR in mixed policy (1.0 = pure mean, 0.0 = pure CVaR)"
            )
        else:
            captain_mix_lambda = 0.6
        
        # Risk and Monte Carlo settings
        st.subheader("🎲 Risk & Monte Carlo")
        
        mc_config = config.get("mc", {})
        
        risk_lambda = st.slider(
            "Risk Penalty (λ)",
            min_value=0.0,
            max_value=1.0,
            value=mc_config.get("lambda_risk", 0.2),
            step=0.05,
            help="Risk aversion parameter (higher = more conservative)"
        )
        
        cvar_alpha = st.slider(
            "CVaR Alpha",
            min_value=0.05,
            max_value=0.5,
            value=mc_config.get("cvar_alpha", 0.2),
            step=0.05,
            help="Tail probability for CVaR calculation (lower = more risk-averse)"
        )
        
        consider_roll = st.checkbox(
            "Consider Rolling FTs",
            value=strategy_config.get("allow_roll", True),
            help="Whether to consider rolling free transfers"
        )
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Team Setup", "📊 Planning Results", "🔄 Transfer History", "💡 Recommendations"])
    
    with tab1:
        st.subheader("🏠 Team Initialization")
        
        # Team initialization options
        init_method = st.radio(
            "Initialize Team From:",
            options=["Saved State", "FPL Entry ID", "Upload CSV", "Manual Entry"],
            index=0,
            help="Choose how to initialize your team"
        )
        
        team_state = None
        
        if init_method == "Saved State":
            # Load from saved state
            saved_states = []
            cache_dir = get_artifacts_dir() / "cache"
            
            if cache_dir.exists():
                for file in cache_dir.glob("team_state_*.json"):
                    saved_states.append(file.name)
            
            if saved_states:
                selected_state = st.selectbox(
                    "Select Saved State",
                    options=saved_states,
                    index=0 if "team_state_gw1.json" in saved_states else 0
                )
                
                if st.button("Load Team State"):
                    team_state = load_team_state(selected_state)
                    if team_state:
                        st.success(f"Team state loaded from {selected_state}")
                        st.session_state['team_state'] = team_state
            else:
                st.info("No saved team states found. Initialize using another method first.")
        
        elif init_method == "FPL Entry ID":
            # Initialize from FPL Entry ID
            entry_id = st.number_input(
                "FPL Entry ID",
                min_value=1,
                value=None,
                help="Your FPL Entry ID"
            )
            
            if entry_id and st.button("Initialize from Entry ID"):
                with st.spinner("Loading team from FPL..."):
                    team_state = initialize_team_from_entry_id(entry_id, bank)
                    
                    if team_state:
                        st.success("Team initialized from FPL Entry ID")
                        st.session_state['team_state'] = team_state
                        
                        # Save as GW1 baseline
                        save_team_state(team_state, "team_state_gw1.json")
        
        elif init_method == "Upload CSV":
            # Initialize from CSV
            uploaded_file = st.file_uploader(
                "Upload Team CSV",
                type=['csv'],
                help="CSV with columns: element_id, web_name, position, etc."
            )
            
            if uploaded_file and st.button("Initialize from CSV"):
                with st.spinner("Processing CSV..."):
                    team_state = initialize_team_from_csv(uploaded_file, bank)
                    
                    if team_state:
                        st.success("Team initialized from CSV")
                        st.session_state['team_state'] = team_state
                        
                        # Save as GW1 baseline
                        save_team_state(team_state, "team_state_gw1.json")
        
        elif init_method == "Manual Entry":
            # Manual team entry (simplified)
            st.info("Manual entry not yet implemented. Use FPL Entry ID or CSV upload.")
        
        # Display current team state
        if 'team_state' in st.session_state:
            team_state = st.session_state['team_state']
        
        if team_state:
            st.divider()
            st.subheader("📋 Current Team State")
            display_team_state(team_state)
    
    with tab2:
        st.subheader("📊 Transfer Planning Results")
        
        if 'team_state' not in st.session_state:
            st.warning("Initialize your team in the Team Setup tab first")
            return
        
        team_state = st.session_state['team_state']
        
        # Run planning button
        if st.button("🚀 Run Transfer Planning", type="primary"):
            # State overrides with strategy settings
            overrides = {
                'gameweek': start_gw,
                'bank': bank,
                'free_transfers': free_transfers,
                'strategy': {
                    'max_transfers_per_gw': max_transfers_per_gw,
                    'roll_threshold_points': roll_threshold,
                    'bank_future_weight': bank_future_weight,
                    'hit_cost': hit_cost,
                    'allow_roll': consider_roll,
                    'shortlist': {
                        'per_pos': shortlist_per_pos
                    }
                },
                'captain': {
                    'policy': captain_policy,
                    'candidates': captain_candidates,
                    'mix_lambda': captain_mix_lambda,
                    'cvar_alpha': cvar_alpha
                },
                'mc': {
                    'lambda_risk': risk_lambda,
                    'cvar_alpha': cvar_alpha
                }
            }
            
            with st.spinner(f"Planning transfers for {horizon} gameweeks..."):
                plan_result = run_transfer_planning(start_gw, horizon, team_state, overrides)
                
                if plan_result:
                    st.session_state['plan_result'] = plan_result
                    st.success("Transfer planning completed!")
        
        # Display planning results
        if 'plan_result' in st.session_state:
            plan_result = st.session_state['plan_result']
            
            # Plan summary
            analysis = plan_result.get('analysis', {})
            
            if analysis:
                value_analysis = analysis.get('value_analysis', {})
                transfer_summary = analysis.get('transfer_summary', {})
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Total Expected Value",
                        f"{value_analysis.get('total_expected_points', 0):.1f} pts"
                    )
                
                with col2:
                    st.metric(
                        "Total Hit Cost",
                        f"{value_analysis.get('total_hit_cost', 0)} pts"
                    )
                
                with col3:
                    st.metric(
                        "Net Expected Value",
                        f"{value_analysis.get('net_expected_value', 0):.1f} pts"
                    )
                
                with col4:
                    st.metric(
                        "Total Transfers",
                        f"{transfer_summary.get('single_transfers', 0) + transfer_summary.get('double_transfers', 0)}"
                    )
            
            st.divider()
            
            # Weekly plan with strategy decisions
            steps = plan_result.get('steps', [])
            
            if steps:
                st.markdown("#### 📅 Weekly Transfer Plan")
                
                for step in steps:
                    step_text = format_transfer_step(step)
                    st.markdown(step_text)
                    
                    # Show strategy decision details if available
                    decision = step.get('decision')
                    if decision:
                        with st.expander(f"Strategy Details - GW {step.get('gameweek', '?')}", expanded=False):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown("**Transfer Decision:**")
                                if decision.get('roll', False):
                                    st.markdown("🏦 Roll free transfer")
                                else:
                                    transfers = decision.get('transfers', [])
                                    if transfers:
                                        for out_id, in_id in transfers:
                                            st.markdown(f"🔄 Player {out_id} → Player {in_id}")
                                    else:
                                        st.markdown("📍 No transfers")
                                
                                hits = decision.get('hits', 0)
                                if hits > 0:
                                    st.markdown(f"💥 {hits} hit{'s' if hits != 1 else ''} taken")
                            
                            with col2:
                                st.markdown("**Expected Performance:**")
                                exp_points = decision.get('exp_points', 0)
                                st.markdown(f"📊 Expected: {exp_points:.1f} pts")
                                
                                risk_adj = decision.get('risk_adj', 0)
                                st.markdown(f"⚖️ Risk-adjusted: {risk_adj:.1f} pts")
                                
                                bank_after = decision.get('bank_after', 0)
                                st.markdown(f"💰 Bank after: £{bank_after:.1f}M")
                            
                            with col3:
                                st.markdown("**Captain/Strategy:**")
                                captain_id = decision.get('captain_id', 0)
                                vice_id = decision.get('vice_id', 0)
                                if captain_id:
                                    st.markdown(f"👑 Captain: Player {captain_id}")
                                if vice_id and vice_id != captain_id:
                                    st.markdown(f"🥈 Vice: Player {vice_id}")
                                
                                reason = decision.get('details', {}).get('reason', 'unknown')
                                reason_display = {
                                    'roll': '🏦 Roll FT',
                                    'roll_threshold': '📉 Below threshold',
                                    '1T_improve': '⬆️ Single transfer',
                                    '2T_improve': '⬆️⬆️ Double transfer',
                                    'baseline_failed': '❌ Baseline failed',
                                    'error': '🚨 Error occurred'
                                }.get(reason, f"❓ {reason}")
                                st.markdown(f"🎯 Strategy: {reason_display}")
                
                # Plan analysis
                st.divider()
                
                if analysis:
                    st.markdown("#### 📈 Plan Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Transfer Summary:**")
                        st.markdown(f"• Single transfers: {transfer_summary.get('single_transfers', 0)}")
                        st.markdown(f"• Double transfers: {transfer_summary.get('double_transfers', 0)}")
                        st.markdown(f"• Rolls: {transfer_summary.get('rolls', 0)}")
                        st.markdown(f"• Total hits: {transfer_summary.get('total_hits', 0)}")
                    
                    with col2:
                        st.markdown("**Value Analysis:**")
                        st.markdown(f"• Avg GW value: {value_analysis.get('average_gw_value', 0):.1f} pts")
                        
                        efficiency = analysis.get('efficiency_metrics', {})
                        st.markdown(f"• Value per transfer: {efficiency.get('value_per_transfer', 0):.1f} pts")
                        st.markdown(f"• Transfers per GW: {efficiency.get('transfers_per_gw', 0):.2f}")
                    
                    # Chips used
                    chips_used = transfer_summary.get('chips_used', [])
                    if chips_used:
                        st.markdown(f"**Chips Used:** {', '.join(chips_used)}")
                    else:
                        st.markdown("**Chips Used:** None")
                
                # Save plan
                if st.button("💾 Save Transfer Plan"):
                    plan_filename = f"transfer_plan_gw{start_gw}_{horizon}w.json"
                    
                    try:
                        cache_dir = get_artifacts_dir() / "cache"
                        cache_dir.mkdir(exist_ok=True)
                        
                        with open(cache_dir / plan_filename, 'w') as f:
                            json.dump(plan_result, f, indent=2, default=str)
                        
                        st.success(f"Transfer plan saved as {plan_filename}")
                    except Exception as e:
                        st.error(f"Failed to save plan: {e}")
    
    with tab3:
        st.subheader("🔄 Transfer History")
        
        # Load saved plans
        cache_dir = get_artifacts_dir() / "cache"
        saved_plans = []
        
        if cache_dir.exists():
            for file in cache_dir.glob("transfer_plan_*.json"):
                saved_plans.append(file.name)
        
        if saved_plans:
            st.markdown(f"**{len(saved_plans)} saved transfer plans found**")
            
            selected_plan = st.selectbox(
                "Select Plan to View",
                options=saved_plans,
                index=0
            )
            
            if selected_plan:
                try:
                    with open(cache_dir / selected_plan, 'r') as f:
                        plan_data = json.load(f)
                    
                    # Display plan summary
                    metadata = plan_data.get('metadata', {})
                    st.markdown(f"**Plan:** {selected_plan}")
                    st.markdown(f"**Created:** {metadata.get('planning_timestamp', 'Unknown')}")
                    st.markdown(f"**Period:** {metadata.get('analysis_period', 'Unknown')}")
                    
                    # Show steps
                    steps = plan_data.get('steps', [])
                    if steps:
                        st.markdown("**Transfer Steps:**")
                        for step in steps:
                            step_text = format_transfer_step(step)
                            st.markdown(step_text)
                
                except Exception as e:
                    st.error(f"Error loading plan: {e}")
        else:
            st.info("No saved transfer plans found. Create a plan first.")
    
    with tab4:
        st.subheader("💡 Recommendations")
        
        # General recommendations
        st.markdown("#### 🎯 Planning Best Practices")
        
        st.markdown("""
        **Transfer Planning Tips:**
        
        1. **Start with GW1 baseline** - Use your actual GW1 team for realistic planning
        2. **Consider fixtures** - Factor in upcoming fixture difficulty and double gameweeks
        3. **Risk management** - Higher λ values favor safer, lower-variance players
        4. **Chip timing** - Plan chip usage around favorable gameweeks (DGWs for BB/TC, BGWs for FH)
        5. **Rolling transfers** - Sometimes banking transfers provides better long-term value
        """)
        
        # Dynamic recommendations based on state
        if 'team_state' in st.session_state and 'plan_result' in st.session_state:
            plan_result = st.session_state['plan_result']
            analysis = plan_result.get('analysis', {})
            
            st.markdown("#### 🔍 Plan-Specific Recommendations")
            
            if analysis:
                transfer_summary = analysis.get('transfer_summary', {})
                value_analysis = analysis.get('value_analysis', {})
                efficiency = analysis.get('efficiency_metrics', {})
                
                # Check for issues
                recommendations = []
                
                total_hits = transfer_summary.get('total_hits', 0)
                if total_hits > horizon * 0.5:
                    recommendations.append("⚠️ High number of hits planned - consider more conservative approach")
                
                value_per_transfer = efficiency.get('value_per_transfer', 0)
                if value_per_transfer < 3:
                    recommendations.append("📉 Low value per transfer - focus on higher-impact moves")
                
                net_value = value_analysis.get('net_expected_value', 0)
                if net_value < 0:
                    recommendations.append("🚨 Negative net value - current plan may not be worthwhile")
                
                chips_used = transfer_summary.get('chips_used', [])
                if not chips_used:
                    recommendations.append("🎫 No chips used in plan - consider optimal chip timing")
                
                if recommendations:
                    for rec in recommendations:
                        st.markdown(f"• {rec}")
                else:
                    st.success("✅ Plan looks good! No major issues detected.")
        
        # System recommendations
        st.markdown("#### 🔧 System Recommendations")
        
        system_recs = []
        
        # Check predictions availability
        predictions_count = 0
        for gw in range(start_gw, start_gw + horizon + 1):
            if load_predictions(gw) is not None:
                predictions_count += 1
        
        if predictions_count < horizon:
            system_recs.append(f"📊 Only {predictions_count}/{horizon} gameweeks have predictions - generate more predictions")
        
        # Check team state
        if 'team_state' not in st.session_state:
            system_recs.append("🏠 Initialize your team in the Team Setup tab")
        
        if system_recs:
            for rec in system_recs:
                st.info(rec)
        else:
            st.success("🎉 System ready for optimal transfer planning!")


if __name__ == "__main__":
    main()
