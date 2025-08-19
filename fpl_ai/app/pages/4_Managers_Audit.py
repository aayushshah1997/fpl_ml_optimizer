"""
Managers & Rotation Risk Audit Dashboard Page.

Provides comprehensive view of manager mappings, rotation priors, and data quality
for the automatic rotation risk engine.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from fpl_ai.src.features.rotation_prior import get_manager_rotation_summary, validate_rotation_priors
except ImportError:
    # Fallback for direct file access
    pass

# Page config
st.set_page_config(
    page_title="Managers & Rotation Audit",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Managers & Rotation Risk Engine")
st.markdown("*Automated manager discovery and data-driven rotation priors from FBR API*")

# File paths
project_base = Path(__file__).parent.parent.parent / "fpl_ai"
rotation_path = project_base / "data" / "manager_rotation_overrides.csv"
team_map_path = project_base / "data" / "team_manager_map.csv"

# Check if files exist
files_exist = rotation_path.exists() and team_map_path.exists()

if not files_exist:
    st.error("🚨 Manager data not found!")
    st.markdown("""
    The rotation engine data files are missing. Please run:
    
    ```bash
    make rotation_update
    ```
    
    This will:
    - Discover current PL managers via FBR API
    - Compute rotation priors from historical match data
    - Generate the required CSV files
    """)
    st.stop()

# Load data
try:
    rotation_df = pd.read_csv(rotation_path)
    team_map_df = pd.read_csv(team_map_path)
    
    # Combine for full view
    if 'manager' in rotation_df.columns:
        full_df = team_map_df.merge(rotation_df, on='manager', how='left')
    else:
        full_df = team_map_df.copy()
        
    st.success(f"✅ Loaded data for {len(team_map_df)} teams and {len(rotation_df)} managers")
    
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview", 
    "👔 Team Managers", 
    "🔄 Rotation Priors",
    "🔍 Data Quality"
])

with tab1:
    st.header("System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Teams Tracked", len(team_map_df))
    
    with col2:
        unknown_count = int((team_map_df.get('manager', '') == 'Unknown').sum())
        st.metric("Unknown Managers", unknown_count)
    
    with col3:
        if 'blended_prior' in full_df.columns:
            avg_prior = full_df['blended_prior'].mean()
            st.metric("Avg Rotation Prior", f"{avg_prior:.3f}")
        else:
            st.metric("Avg Rotation Prior", "N/A")
    
    with col4:
        if 'n_matches_curr' in full_df.columns:
            sufficient_data = (full_df['n_matches_curr'] >= 8).sum()
            st.metric("Teams w/ Sufficient Data", sufficient_data)
        else:
            st.metric("Teams w/ Sufficient Data", "N/A")
    
    # Data flow diagram
    st.subheader("🔄 Data Flow")
    st.markdown("""
    1. **Manager Discovery**: Current PL managers via FBR API + CSV overrides
    2. **Match Data**: Historical lineups across all competitions (prev season + current YTD)
    3. **Metrics Computation**: XI changes, starts variance, minutes distribution, bench rates
    4. **Prior Mapping**: Weighted combination → rotation prior (0.03-0.30 range)
    5. **Blending**: Current season (if ≥8 matches) vs weighted blend with previous season
    6. **Pipeline Integration**: Used in minutes model and Monte Carlo simulations
    """)
    
    # Recent update info
    if rotation_path.exists():
        mod_time = rotation_path.stat().st_mtime
        import datetime
        last_update = datetime.datetime.fromtimestamp(mod_time)
        st.info(f"📅 Last updated: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")

with tab2:
    st.header("👔 Team Manager Mapping")
    
    # Display team managers
    display_df = team_map_df.copy()
    if 'manager' in display_df.columns:
        # Add status indicator
        display_df['Status'] = display_df['manager'].apply(
            lambda x: '❓ Unknown' if x == 'Unknown' else '✅ Resolved'
        )
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "team_id": "Team ID",
            "team_name": "Team Name", 
            "manager": "Manager",
            "Status": "Status"
        }
    )
    
    # Manager stats
    if 'manager' in team_map_df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Manager Resolution")
            manager_counts = team_map_df['manager'].value_counts()
            
            fig = px.pie(
                values=[
                    len(team_map_df[team_map_df['manager'] != 'Unknown']),
                    len(team_map_df[team_map_df['manager'] == 'Unknown'])
                ],
                names=['Resolved', 'Unknown'],
                title="Manager Resolution Status",
                color_discrete_map={'Resolved': '#00ff87', 'Unknown': '#ff6b6b'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🔧 Recommendations")
            unknown_teams = team_map_df[team_map_df['manager'] == 'Unknown']['team_name'].tolist()
            if unknown_teams:
                st.warning(f"**{len(unknown_teams)} teams** have unknown managers:")
                for team in unknown_teams:
                    st.write(f"- {team}")
                st.markdown("*Consider adding manual overrides in `data/manager_overrides.csv`*")
            else:
                st.success("✅ All managers successfully resolved!")

with tab3:
    st.header("🔄 Rotation Priors")
    
    if 'blended_prior' not in rotation_df.columns:
        st.warning("No rotation prior data available. Run `make rotation_update` to generate.")
    else:
        # Prior distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Prior Distribution")
            fig = px.histogram(
                rotation_df,
                x='blended_prior',
                nbins=20,
                title="Distribution of Rotation Priors",
                labels={'blended_prior': 'Blended Prior', 'count': 'Number of Managers'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Data Availability")
            if 'n_matches_curr' in rotation_df.columns and 'n_matches_prev' in rotation_df.columns:
                fig = px.scatter(
                    rotation_df,
                    x='n_matches_prev',
                    y='n_matches_curr',
                    size='blended_prior',
                    hover_name='manager',
                    title="Match Data Availability",
                    labels={
                        'n_matches_prev': 'Previous Season Matches',
                        'n_matches_curr': 'Current Season Matches',
                        'blended_prior': 'Rotation Prior'
                    }
                )
                fig.add_hline(y=8, line_dash="dash", line_color="red", 
                             annotation_text="Stability Threshold (8 matches)")
                st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.subheader("📋 Detailed Rotation Data")
        
        display_cols = ['manager', 'blended_prior', 'n_matches_prev', 'n_matches_curr']
        if 'prior_prev_season' in rotation_df.columns:
            display_cols.append('prior_prev_season')
        if 'prior_curr_ytd' in rotation_df.columns:
            display_cols.append('prior_curr_ytd')
        
        available_cols = [col for col in display_cols if col in rotation_df.columns]
        
        # Add data quality indicator
        display_rotation = rotation_df[available_cols].copy()
        if 'n_matches_curr' in display_rotation.columns:
            display_rotation['Data Quality'] = display_rotation['n_matches_curr'].apply(
                lambda x: '🟢 Excellent' if x >= 8 
                else '🟡 Good' if x >= 4 
                else '🔴 Limited'
            )
        
        st.dataframe(
            display_rotation.sort_values('blended_prior', ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={
                'blended_prior': st.column_config.NumberColumn('Blended Prior', format="%.3f"),
                'prior_prev_season': st.column_config.NumberColumn('Previous Season', format="%.3f"),
                'prior_curr_ytd': st.column_config.NumberColumn('Current YTD', format="%.3f"),
                'n_matches_prev': 'Prev Matches',
                'n_matches_curr': 'Curr Matches'
            }
        )

with tab4:
    st.header("🔍 Data Quality & Diagnostics")
    
    # Validation metrics
    try:
        validation = validate_rotation_priors()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📁 File Status")
            for file_type, exists in validation.get("files_exist", {}).items():
                status = "✅" if exists else "❌"
                st.write(f"{status} {file_type.replace('_', ' ').title()}")
        
        with col2:
            st.subheader("📊 Data Statistics")
            quality = validation.get("data_quality", {})
            for metric, value in quality.items():
                if isinstance(value, (int, float)):
                    st.metric(metric.replace('_', ' ').title(), value)
                elif isinstance(value, list) and len(value) == 2:
                    st.metric(f"{metric.replace('_', ' ').title()} Range", f"{value[0]:.3f} - {value[1]:.3f}")
        
        # Coverage analysis
        if "coverage" in validation:
            st.subheader("📈 Coverage Analysis")
            coverage = validation["coverage"]
            
            col1, col2 = st.columns(2)
            with col1:
                sufficient = coverage.get("teams_with_sufficient_current_data", 0)
                st.metric("Teams with Sufficient Current Data", sufficient)
            
            with col2:
                pct = coverage.get("pct_sufficient_data", 0)
                st.metric("% Sufficient Data", f"{pct:.1f}%")
                
                # Add progress bar
                st.progress(pct / 100)
        
        # Recommendations
        recommendations = validation.get("recommendations", [])
        if recommendations:
            st.subheader("💡 Recommendations")
            for rec in recommendations:
                st.info(f"💡 {rec}")
        else:
            st.success("✅ No issues detected - system is working optimally!")
            
    except Exception as e:
        st.error(f"Failed to run validation: {e}")
    
    # System configuration
    st.subheader("⚙️ System Configuration")
    
    config_info = {
        "Competition ID": "9 (Premier League)",
        "Previous Season": "2024-2025",
        "Current Season": "2025-2026",
        "Stability Threshold": "8 matches",
        "Default Prior": "0.05",
        "Prior Range": "0.03 - 0.30"
    }
    
    for key, value in config_info.items():
        st.write(f"**{key}**: {value}")

# Footer
st.markdown("---")
st.markdown("""
**Rotation Risk Engine** | *Powered by FBR API* | **No Web Scraping** ✅  
*Automatically updates rotation priors from comprehensive match data across all competitions*
""")
