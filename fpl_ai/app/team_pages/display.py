"""
Display utilities for the FPL AI dashboard.

Handles UI rendering helpers and display formatting.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Any, Tuple


def display_player_fixtures(player_data: Dict, current_gw: int):
    """Display player fixtures for the next few gameweeks."""
    if not player_data:
        return
    
    try:
        from ..data_loaders import load_team_fixtures, get_difficulty_color
        
        team_name = player_data.get('team_name', 'Unknown')
        player_name = player_data.get('web_name', 'Unknown')
        
        # Get team fixtures
        fixtures_df = load_team_fixtures(team_name, current_gw, current_gw + 5)
        
        if fixtures_df.empty:
            st.write(f"**{player_name}** ({team_name}) - No upcoming fixtures")
            return
        
        st.write(f"**{player_name}** ({team_name}) - Next 5 fixtures:")
        
        # Display fixtures with difficulty colors
        for _, fixture in fixtures_df.iterrows():
            opponent = fixture.get('opponent', 'Unknown')
            difficulty = fixture.get('difficulty', 3)
            is_home = fixture.get('is_home', True)
            
            # Get difficulty color
            color = get_difficulty_color(difficulty)
            
            # Format display
            venue = "vs" if is_home else "@"
            st.markdown(f"- {venue} **{opponent}** - <span style='color: {color}'>Difficulty: {difficulty}</span>", 
                       unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error displaying fixtures: {e}")


def display_team_summary(team_data: Dict[str, Any], title: str = "Team Summary"):
    """Display a summary of team data."""
    if not team_data or team_data.get("error"):
        st.error(f"Error in team data: {team_data.get('error', 'Unknown error')}")
        return
    
    st.subheader(title)
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Expected Points", f"{team_data.get('expected_points', 0):.1f}")
    
    with col2:
        st.metric("Total Cost", f"£{team_data.get('total_cost', 0):.1f}m")
    
    with col3:
        st.metric("Bank", f"£{team_data.get('bank', 0):.1f}m")
    
    with col4:
        formation = team_data.get('formation', (3, 4, 3))
        st.metric("Formation", f"{formation[0]}-{formation[1]}-{formation[2]}")


def display_formation_grid(starting_xi: List[Dict], formation: Tuple[int, int, int]):
    """Display the team formation in a grid layout."""
    if not starting_xi or not formation:
        st.error("No team data to display")
        return
    
    # Group players by position
    gk_players = [p for p in starting_xi if p.get('position') == 'GK']
    def_players = [p for p in starting_xi if p.get('position') == 'DEF']
    mid_players = [p for p in starting_xi if p.get('position') == 'MID']
    fwd_players = [p for p in starting_xi if p.get('position') == 'FWD']
    
    # Create formation display
    st.subheader(f"Formation: {formation[0]}-{formation[1]}-{formation[2]}")
    
    # Goalkeeper
    if gk_players:
        with st.container():
            st.markdown("**Goalkeeper**")
            for gk in gk_players[:1]:  # Only show first GK
                st.write(f"🛡️ {gk.get('web_name', 'Unknown')} ({gk.get('team_name', 'Unknown')}) - {gk.get('proj_points', 0):.1f} pts")
    
    # Defense
    if def_players:
        with st.container():
            st.markdown("**Defense**")
            cols = st.columns(min(len(def_players), 5))
            for i, defender in enumerate(def_players[:formation[0]]):
                with cols[i % 5]:
                    st.write(f"🛡️ {defender.get('web_name', 'Unknown')}")
                    st.write(f"({defender.get('team_name', 'Unknown')})")
                    st.write(f"{defender.get('proj_points', 0):.1f} pts")
    
    # Midfield
    if mid_players:
        with st.container():
            st.markdown("**Midfield**")
            cols = st.columns(min(len(mid_players), 5))
            for i, midfielder in enumerate(mid_players[:formation[1]]):
                with cols[i % 5]:
                    st.write(f"⚽ {midfielder.get('web_name', 'Unknown')}")
                    st.write(f"({midfielder.get('team_name', 'Unknown')})")
                    st.write(f"{midfielder.get('proj_points', 0):.1f} pts")
    
    # Forwards
    if fwd_players:
        with st.container():
            st.markdown("**Forwards**")
            cols = st.columns(min(len(fwd_players), 3))
            for i, forward in enumerate(fwd_players[:formation[2]]):
                with cols[i % 3]:
                    st.write(f"🎯 {forward.get('web_name', 'Unknown')}")
                    st.write(f"({forward.get('team_name', 'Unknown')})")
                    st.write(f"{forward.get('proj_points', 0):.1f} pts")


def display_captain_selection(captain: Dict, vice_captain: Dict, captain_value: Dict[str, Any]):
    """Display captain and vice-captain selection."""
    if not captain:
        st.warning("No captain selected")
        return
    
    st.subheader("Captain & Vice-Captain")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Captain**")
        st.write(f"👑 {captain.get('web_name', 'Unknown')} ({captain.get('team_name', 'Unknown')})")
        st.write(f"Expected: {captain.get('proj_points', 0) * 2:.1f} pts (×2)")
        
        if captain_value:
            st.write(f"Form Score: {captain_value.get('form_score', 0)}/5")
            st.write(f"Risk Level: {captain_value.get('risk_level', 'Unknown')}")
    
    with col2:
        st.markdown("**Vice-Captain**")
        if vice_captain:
            st.write(f"🔸 {vice_captain.get('web_name', 'Unknown')} ({vice_captain.get('team_name', 'Unknown')})")
            st.write(f"Expected: {vice_captain.get('proj_points', 0):.1f} pts")
        else:
            st.write("No vice-captain selected")


def display_transfer_suggestions(transfer_data: Dict[str, Any]):
    """Display transfer suggestions."""
    if not transfer_data or transfer_data.get("action") == "No data":
        st.info("No transfer suggestions available")
        return
    
    st.subheader("Transfer Suggestions")
    
    action = transfer_data.get("action", "Unknown")
    details = transfer_data.get("details", [])
    
    if action == "Save transfer":
        st.success("💾 **Save Transfer**")
        st.write("No beneficial transfers found. Consider saving your transfer.")
    elif action == "1 transfer":
        st.info("🔄 **Recommended Transfer**")
        st.write("Consider making this transfer:")
    else:
        st.write(f"**{action}**")
    
    # Display details
    for detail in details:
        st.write(f"• {detail}")


def create_position_summary(starting_xi: List[Dict]) -> Dict[str, Any]:
    """Create a summary of players by position."""
    if not starting_xi:
        return {}
    
    summary = {
        "GK": [],
        "DEF": [],
        "MID": [],
        "FWD": []
    }
    
    for player in starting_xi:
        position = player.get('position', 'MID')
        if position in summary:
            summary[position].append({
                "name": player.get('web_name', 'Unknown'),
                "team": player.get('team_name', 'Unknown'),
                "points": player.get('proj_points', 0),
                "cost": player.get('cost', 0)
            })
    
    return summary
