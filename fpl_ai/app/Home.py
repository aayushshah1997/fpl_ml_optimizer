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

from app._utils import load_config, get_current_gw, check_predictions_available
from src.common.timeutil import get_current_gw as get_gw_util

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
        
        # Gameweek selection
        try:
            current_gw = get_gw_util()
            gw_options = list(range(max(1, current_gw - 5), current_gw + 6))
            default_idx = gw_options.index(current_gw) if current_gw in gw_options else 0
        except:
            gw_options = list(range(1, 39))
            default_idx = 0
            current_gw = 1
        
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
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Welcome section
        st.markdown("""
        ## 🚀 Welcome to FPL AI
        
        Your comprehensive Fantasy Premier League analytics platform powered by machine learning.
        This dashboard provides:
        
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
        """)
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        action_col1, action_col2, action_col3 = st.columns(3)
        
        with action_col1:
            if st.button("🎯 View Predicted Team", use_container_width=True):
                st.switch_page("pages/1_Predicted_Team.py")
        
        with action_col2:
            if st.button("📊 Model Performance", use_container_width=True):
                st.switch_page("pages/2_Model_Performance.py")
        
        with action_col3:
            if st.button("📅 10-Week Planner", use_container_width=True):
                st.switch_page("pages/3_10_Week_Planner.py")
    
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
