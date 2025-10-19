"""
FPL AI Dashboard - Home Page

Main landing page for the FPL AI Streamlit dashboard with overview
and navigation to different analysis pages.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import os

# Add src to path for imports
app_dir = Path(__file__).parent
src_dir = app_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from fpl_ai.app._utils import load_config, get_current_gw, check_predictions_available, get_saved_entry_id
from fpl_ai.src.common.timeutil import get_current_gw as get_gw_util
from fpl_ai.app.team_utils import get_team_manager, display_team_data, display_player_prices

# Page configuration
st.set_page_config(
    page_title="FPL AI Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main dashboard home page."""
    
    # Title and header
    st.title("⚽ FPL AI Dashboard")
    st.markdown("### Advanced Fantasy Premier League Analytics & Optimization")
    
    # Sidebar for global settings
    with st.sidebar:
        st.header("🎛️ Global Settings")

        # FPL Team ID Input Section
        st.subheader("👤 Your FPL Team")

        # Initialize session state for team data
        if 'fpl_team_id' not in st.session_state:
            st.session_state['fpl_team_id'] = ''
        if 'team_data' not in st.session_state:
            st.session_state['team_data'] = None
        if 'team_data_loaded' not in st.session_state:
            st.session_state['team_data_loaded'] = False

        # Team ID input
        saved_entry_id = get_saved_entry_id()
        default_value = str(saved_entry_id) if saved_entry_id else st.session_state['fpl_team_id']
        
        team_id_input = st.text_input(
            "FPL Team ID",
            value=default_value,
            placeholder="Enter your FPL team ID (e.g., 123456)",
            help="Your FPL team ID can be found in the URL when viewing your team on the FPL website"
        )

        # Load team data button
        if st.button("🔄 Load Team Data", use_container_width=True):
            if team_id_input.strip():
                st.session_state['fpl_team_id'] = team_id_input.strip()
                
                # Save the Entry ID for persistence
                try:
                    from fpl_ai.app._utils import set_saved_entry_id
                    set_saved_entry_id(int(team_id_input.strip()))
                except Exception:
                    pass
                
                team_manager = get_team_manager()
                team_data = team_manager.load_team_data(st.session_state['fpl_team_id'])

                if team_data:
                    st.session_state['team_data'] = team_data
                    st.session_state['team_data_loaded'] = True
                    st.success(f"✅ Team data loaded for: {team_data['team_name']}")
                    st.rerun()
                else:
                    st.session_state['team_data_loaded'] = False
            else:
                st.error("❌ Please enter a valid team ID")

        # Clear team data button
        if st.button("🗑️ Clear Team Data", use_container_width=True):
            st.session_state['fpl_team_id'] = ''
            st.session_state['team_data'] = None
            st.session_state['team_data_loaded'] = False
            st.rerun()

        # Display team data status
        if st.session_state.get('team_data_loaded') and st.session_state.get('team_data'):
            team_data = st.session_state['team_data']
            data_gw = team_data.get('data_gw', team_data.get('current_gw', 'Unknown'))
            st.success(f"📊 {team_data['team_name']} (ID: {team_data['team_id']}) - GW {data_gw} data")
            try:
                last_updated = pd.to_datetime(team_data['last_updated'])
                # Handle timezone-aware datetime
                if last_updated.tzinfo:
                    last_updated = last_updated.tz_convert(None)
                st.caption(f"Last updated: {last_updated.strftime('%H:%M:%S')}")
            except Exception as e:
                st.caption(f"Last updated: {team_data.get('last_updated', 'Unknown')}")

        st.divider()

        # Gameweek selection - default to GW7 for predictions
        try:
            current_gw = get_gw_util()
            gw_options = list(range(max(1, current_gw - 5), current_gw + 6))
            # Default to GW7 if available, otherwise current GW
            if 7 in gw_options:
                default_idx = gw_options.index(7)
            else:
                default_idx = gw_options.index(current_gw) if current_gw in gw_options else 0
        except:
            gw_options = list(range(1, 39))
            # Default to GW7 if available
            default_idx = gw_options.index(7) if 7 in gw_options else 0
            current_gw = 7

        selected_gw = st.selectbox(
            "Target Gameweek",
            options=gw_options,
            index=default_idx,
            help="Select the gameweek for predictions and analysis"
        )

        # Store in session state
        st.session_state['selected_gw'] = selected_gw

        # Configuration info
        st.divider()
        st.subheader("📊 System Status")

        # Check if predictions are available
        predictions_available = check_predictions_available(selected_gw)

        if predictions_available:
            st.success(f"✅ Predictions available for GW {selected_gw}")
        else:
            st.warning(f"⚠️ No predictions for GW {selected_gw}")
            st.info("Run training pipeline to generate predictions")

        # Training mode indicator
        config = load_config()
        staging_config = config.get("training", {}).get("staging", {})
        warm_until_gw = staging_config.get("warm_until_gw", 8)

        if selected_gw < warm_until_gw:
            training_mode = "🌱 Warm Start"
            mode_color = "orange"
        else:
            training_mode = "🔥 Full ML"
            mode_color = "green"

        st.markdown(f"**Training Mode:** :{mode_color}[{training_mode}]")

        # System info
        st.divider()
        st.caption("🤖 FPL AI v0.1.0")
    
    # Display team data if available
    if st.session_state.get('team_data_loaded') and st.session_state.get('team_data'):
        team_data = st.session_state['team_data']

        # Team Data Section
        st.markdown(f"## 👤 Your Team: {team_data['team_name']}")

        # Create tabs for different views
        team_tab1, team_tab2, team_tab3 = st.tabs(["📊 Overview", "💰 Player Prices", "📈 Analytics"])

        with team_tab1:
            display_team_data(team_data)

        with team_tab2:
            display_player_prices(team_data)

        with team_tab3:
            st.markdown("### 📈 Team Analytics")
            
            # Import and display team analytics
            try:
                from fpl_ai.app.team_analytics import display_team_analytics
                from fpl_ai.app._utils import load_predictions, get_current_gw
                
                # Get current gameweek and predictions
                current_gw = get_current_gw()
                predictions_df = load_predictions(current_gw)
                
                if predictions_df is not None and not predictions_df.empty:
                    # Enrich predictions with FPL data
                    from fpl_ai.app._utils import enrich_predictions_with_fpl_data
                    predictions_df = enrich_predictions_with_fpl_data(predictions_df)
                    
                    # Get saved entry ID
                    entry_id = get_saved_entry_id()
                    
                    if entry_id:
                        display_team_analytics(entry_id, predictions_df)
                    else:
                        st.info("Please enter your FPL Entry ID in the sidebar to view team analytics")
                else:
                    st.warning("No predictions available. Run the training pipeline to generate predictions.")
                    
            except ImportError as e:
                st.error(f"Could not load team analytics: {e}")
            except Exception as e:
                st.error(f"Error loading team analytics: {e}")

        st.divider()

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        # Welcome section
        if st.session_state.get('team_data_loaded') and st.session_state.get('team_data'):
            st.markdown("""
            ## 🚀 FPL AI Analysis

            Your comprehensive Fantasy Premier League analytics platform powered by machine learning.
            With your team loaded, you can now:
            """)
        else:
            st.markdown("""
            ## 🚀 Welcome to FPL AI

            Your comprehensive Fantasy Premier League analytics platform powered by machine learning.
            This dashboard provides:
            """)

        # Dynamic content based on team data
        if st.session_state.get('team_data_loaded') and st.session_state.get('team_data'):
            st.markdown("""
            ### 🎯 **AI-Powered Analysis for Your Team**

            **📈 Predicted Team** - Get AI-optimized suggestions based on your current squad
            **🔍 Model Performance** - See how well predictions match your players
            **📅 10-Week Planner** - Plan transfers with your actual team as baseline
            **🗃️ Player Database** - Browse all players with comprehensive metrics by position

            *💡 Your team data is being used to provide personalized insights!*
            """)
        else:
            st.markdown("""
            ### 🎯 **Core Features**

            **📈 Predicted Team** - AI-optimized team selection with:
            - Position-specific ML models with staging (warm vs full training)
            - Monte Carlo simulation with risk analysis
            - Recent form integration (R3/R5 snapshots)
            - Captain optimization with confidence intervals

            **🔍 Model Performance** - Comprehensive model diagnostics:
            - Cross-validation metrics by position
            - Feature importance analysis
            - Prediction uncertainty and calibration
            - Data coverage and quality reports

            **📅 10-Week Planner** - Advanced transfer planning:
            - GW1 baseline initialization from your actual team
            - Multi-week horizon optimization with CVaR risk adjustment
            - Chip timing optimization (TC, BB, FH, WC)
            - Budget constraints and selling price calculations
            
            **🗃️ Player Database** - Comprehensive player analysis:
            - All players organized by position (GK, DEF, MID, FWD)
            - Advanced filtering and sorting capabilities
            - Export functionality for external analysis
            - Detailed metrics including projections and current season stats
            """)
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        # First row of actions
        action_col1, action_col2 = st.columns(2)
        
        with action_col1:
            if st.button("🎯 View Predicted Team", use_container_width=True):
                st.switch_page("pages/1_Predicted_Team.py")
        
        with action_col2:
            if st.button("📊 Model Performance", use_container_width=True):
                st.switch_page("pages/2_Model_Performance.py")
        
        # Second row of actions
        action_col3, action_col4 = st.columns(2)
        
        with action_col3:
            if st.button("📅 10-Week Planner", use_container_width=True):
                st.switch_page("pages/3_10_Week_Planner.py")
        
        with action_col4:
            if st.button("🗃️ Player Database", use_container_width=True):
                st.switch_page("pages/6_Player_Database.py")
    
    with col2:
        # System overview card
        st.markdown("### 📋 System Overview")
        
        with st.container():
            # Data years info
            st.markdown("""
            **📊 Training Data:**
            - Last 3 full PL seasons + current
            - Multi-league priors (La Liga, Serie A, etc.)
            - Exponential recency weighting
            """)
            
            # Model info
            st.markdown("""
            **🤖 Models:**
            - Position-specific LightGBM
            - Minutes expectation model
            - Isotonic calibration
            - Monte Carlo uncertainty
            """)
            
            # Features info
            st.markdown("""
            **🔧 Features:**
            - Set-piece roles (penalties, free kicks)
            - Team/opponent form
            - Market sentiment & ownership
            - xG/xA and advanced metrics
            """)
            
            # Security note
            st.markdown("""
            **🔒 Security:**
            - All secrets loaded from `.env`
            - No credentials stored in code
            - API rate limiting implemented
            """)
        
        # Recent activity (placeholder)
        st.markdown("### 📈 Recent Activity")
        
        with st.expander("View Activity Log"):
            st.info("Activity logging not yet implemented")
            st.markdown("""
            Future features:
            - Model training history
            - Prediction accuracy tracking
            - Transfer plan performance
            """)
    
    # Footer with key explanations
    st.divider()
    
    with st.expander("ℹ️ Key Concepts & Explanations"):
        
        concept_col1, concept_col2 = st.columns(2)
        
        with concept_col1:
            st.markdown("""
            **🎯 Staging Mode:**
            - **Warm Start** (GW < 8): Lighter models, higher shrinkage for new signings
            - **Full ML** (GW ≥ 8): Complete feature set, full cross-validation
            
            **📊 Recency Weighting:**
            - Recent games weighted more heavily
            - Current season gets 1.3x boost
            - League strength adjustments for non-PL history
            
            **🔄 Monte Carlo Simulation:**
            - 2000 scenarios by default
            - Position-specific uncertainty
            - CVaR risk metrics (worst 20% scenarios)
            """)
        
        with concept_col2:
            st.markdown("""
            **🎲 New Signings:**
            - FBR API pulls prior league history
            - Per-90 stats adjusted by league strength
            - Cold-start shrinkage when minutes are scarce
            
            **💰 Transfer Planning:**
            - FPL selling value rules implemented
            - 10-week greedy horizon optimization
            - Mean return - λ × CVaR risk penalty
            
            **🏆 GW1 Baseline:**
            - All planning starts from your actual GW1 team
            - Purchase prices and bank balance tracked
            - No retrospective "perfect" team assumptions
            """)
    
    # Navigation help
    st.info("""
    💡 **Navigation Tip:** Use the sidebar to select your target gameweek, then explore the different pages using the navigation menu on the left.
    """)


if __name__ == "__main__":
    main()
