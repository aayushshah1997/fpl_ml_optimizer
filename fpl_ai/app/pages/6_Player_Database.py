"""
FPL AI Dashboard - Player Database Page

Comprehensive player database with all metrics, organized by position.
Provides detailed player analysis, filtering, and export capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Optional, Any
import io

# Add src to path for imports
app_dir = Path(__file__).parent.parent
src_dir = app_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from fpl_ai.app._utils import (
    get_gameweek_selector, build_confidence_intervals, load_config
)
from fpl_ai.app.data_loaders import load_predictions_cached, enrich_with_current_fpl_data

# Page configuration
st.set_page_config(
    page_title="Player Database - FPL AI",
    page_icon="🗃️",
    layout="wide"
)


def format_player_metrics_table(df: pd.DataFrame, position: str) -> pd.DataFrame:
    """Format player metrics for display table."""
    if df.empty:
        return df
    
    # Select relevant columns based on what's available
    base_columns = ['web_name', 'team_name', 'now_cost']
    
    # Prediction columns
    prediction_columns = ['mean_points', 'std_points', 'p10', 'p50', 'p90', 
                         'prob_double_digits', 'prob_15_plus']
    
    # FPL current season columns
    fpl_columns = ['selected_by_percent', 'form', 'total_points', 'minutes', 
                  'goals_scored', 'assists', 'clean_sheets', 'saves', 'bonus']
    
    # Build column list based on what's available
    display_columns = []
    for col in base_columns + prediction_columns + fpl_columns:
        if col in df.columns:
            display_columns.append(col)
    
    # Create display dataframe
    display_df = df[display_columns].copy()
    
    # Format numerical columns
    if 'now_cost' in display_df.columns:
        # Convert cost to millions if needed
        cost_values = display_df['now_cost']
        if cost_values.max() > 20:  # In tenths
            display_df['now_cost'] = cost_values / 10
        display_df['now_cost'] = display_df['now_cost'].apply(lambda x: f"£{x:.1f}M")
    
    # Format prediction columns
    for col in ['mean_points', 'std_points', 'p10', 'p50', 'p90']:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
    
    # Format probability columns
    for col in ['prob_double_digits', 'prob_15_plus']:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
            display_df[col] = display_df[col].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A")
    
    # Format percentage columns
    if 'selected_by_percent' in display_df.columns:
        display_df['selected_by_percent'] = pd.to_numeric(display_df['selected_by_percent'], errors='coerce')
        display_df['selected_by_percent'] = display_df['selected_by_percent'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
    
    # Format other numerical columns
    for col in ['form', 'total_points', 'minutes', 'goals_scored', 'assists', 'clean_sheets', 'saves', 'bonus']:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
            if col in ['form']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
            else:
                display_df[col] = display_df[col].apply(lambda x: f"{int(x)}" if pd.notna(x) else "N/A")
    
    # Rename columns for better display
    column_mapping = {
        'web_name': 'Name',
        'team_name': 'Team',
        'now_cost': 'Cost',
        'mean_points': 'Projected Pts',
        'std_points': 'Std Dev',
        'p10': '10th %ile',
        'p50': 'Median',
        'p90': '90th %ile',
        'prob_double_digits': '10+ Pts %',
        'prob_15_plus': '15+ Pts %',
        'selected_by_percent': 'Ownership',
        'form': 'Form',
        'total_points': 'Total Pts',
        'minutes': 'Minutes',
        'goals_scored': 'Goals',
        'assists': 'Assists',
        'clean_sheets': 'Clean Sheets',
        'saves': 'Saves',
        'bonus': 'Bonus'
    }
    
    display_df = display_df.rename(columns=column_mapping)
    
    return display_df


def create_player_filters(df: pd.DataFrame, position: str) -> Dict[str, Any]:
    """Create position-specific filter controls."""
    with st.expander(f"🎛️ {position} Filters", expanded=False):
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        filters = {}
        
        with filter_col1:
            # Team filter
            if 'team_name' in df.columns:
                teams = ['All'] + sorted(df['team_name'].dropna().unique().tolist())
                filters['team'] = st.selectbox(
                    f"{position} Team",
                    options=teams,
                    index=0,
                    key=f"{position}_team_filter"
                )
            
            # Cost range
            if 'now_cost' in df.columns:
                cost_values = df['now_cost'].dropna()
                if cost_values.max() > 20:  # In tenths
                    cost_values = cost_values / 10
                
                min_cost = float(cost_values.min()) if len(cost_values) > 0 else 4.0
                max_cost = float(cost_values.max()) if len(cost_values) > 0 else 15.0
                
                filters['min_cost'] = st.number_input(
                    f"{position} Min Cost (£M)",
                    min_value=min_cost,
                    max_value=max_cost,
                    value=min_cost,
                    step=0.1,
                    key=f"{position}_min_cost"
                )
                
                filters['max_cost'] = st.number_input(
                    f"{position} Max Cost (£M)",
                    min_value=min_cost,
                    max_value=max_cost,
                    value=max_cost,
                    step=0.1,
                    key=f"{position}_max_cost"
                )
        
        with filter_col2:
            # Projection range
            if 'mean_points' in df.columns:
                proj_values = pd.to_numeric(df['mean_points'], errors='coerce').dropna()
                min_proj = float(proj_values.min()) if len(proj_values) > 0 else 0.0
                max_proj = float(proj_values.max()) if len(proj_values) > 0 else 20.0
                
                filters['min_projection'] = st.number_input(
                    f"{position} Min Projection",
                    min_value=min_proj,
                    max_value=max_proj,
                    value=min_proj,
                    step=0.5,
                    key=f"{position}_min_proj"
                )
                
                filters['max_projection'] = st.number_input(
                    f"{position} Max Projection",
                    min_value=min_proj,
                    max_value=max_proj,
                    value=max_proj,
                    step=0.5,
                    key=f"{position}_max_proj"
                )
            
            # Ownership range
            if 'selected_by_percent' in df.columns:
                own_values = pd.to_numeric(df['selected_by_percent'], errors='coerce').dropna()
                min_own = float(own_values.min()) if len(own_values) > 0 else 0.0
                max_own = float(own_values.max()) if len(own_values) > 0 else 100.0
                
                filters['max_ownership'] = st.number_input(
                    f"{position} Max Ownership (%)",
                    min_value=min_own,
                    max_value=max_own,
                    value=max_own,
                    step=1.0,
                    key=f"{position}_max_own"
                )
        
        with filter_col3:
            # Sort options
            sort_options = [
                'Projected Points (High to Low)',
                'Projected Points (Low to High)',
                'Cost (Low to High)',
                'Cost (High to Low)',
                'Name (A-Z)',
                'Ownership (High to Low)',
                'Form (High to Low)'
            ]
            
            if 'mean_points' in df.columns and 'now_cost' in df.columns:
                sort_options.append('Value (Points/Cost)')
            
            filters['sort_by'] = st.selectbox(
                f"{position} Sort By",
                options=sort_options,
                index=0,
                key=f"{position}_sort"
            )
            
            # Show top N
            filters['show_top'] = st.number_input(
                f"{position} Show Top N",
                min_value=10,
                max_value=200,
                value=50,
                step=10,
                key=f"{position}_show_top"
            )
    
    return filters


def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply filters to dataframe."""
    filtered_df = df.copy()
    
    # Team filter
    if filters.get('team') and filters['team'] != 'All':
        filtered_df = filtered_df[filtered_df.get('team_name') == filters['team']]
    
    # Cost filters
    if 'min_cost' in filters and 'max_cost' in filters and 'now_cost' in filtered_df.columns:
        cost_values = filtered_df['now_cost']
        if cost_values.max() > 20:  # In tenths
            filtered_df = filtered_df[
                (cost_values / 10 >= filters['min_cost']) & 
                (cost_values / 10 <= filters['max_cost'])
            ]
        else:
            filtered_df = filtered_df[
                (cost_values >= filters['min_cost']) & 
                (cost_values <= filters['max_cost'])
            ]
    
    # Projection filters
    if 'min_projection' in filters and 'max_projection' in filters and 'mean_points' in filtered_df.columns:
        proj_values = pd.to_numeric(filtered_df['mean_points'], errors='coerce')
        filtered_df = filtered_df[
            (proj_values >= filters['min_projection']) & 
            (proj_values <= filters['max_projection']) &
            proj_values.notna()
        ]
    
    # Ownership filter
    if 'max_ownership' in filters and 'selected_by_percent' in filtered_df.columns:
        own_values = pd.to_numeric(filtered_df['selected_by_percent'], errors='coerce')
        filtered_df = filtered_df[
            (own_values <= filters['max_ownership']) & own_values.notna()
        ]
    
    # Add value calculation if needed
    if filters.get('sort_by') == 'Value (Points/Cost)' and 'mean_points' in filtered_df.columns and 'now_cost' in filtered_df.columns:
        cost_for_value = filtered_df['now_cost'].copy()
        if cost_for_value.max() > 20:  # In tenths
            cost_for_value = cost_for_value / 10
        filtered_df['value'] = pd.to_numeric(filtered_df['mean_points'], errors='coerce') / cost_for_value
    
    # Sort
    sort_by = filters.get('sort_by', 'Projected Points (High to Low)')
    if 'Projected Points (High to Low)' in sort_by and 'mean_points' in filtered_df.columns:
        # Convert to numeric and sort by column name
        filtered_df['mean_points_numeric'] = pd.to_numeric(filtered_df['mean_points'], errors='coerce')
        filtered_df = filtered_df.sort_values('mean_points_numeric', ascending=False, na_position='last')
        filtered_df = filtered_df.drop(columns=['mean_points_numeric'])
    elif 'Projected Points (Low to High)' in sort_by and 'mean_points' in filtered_df.columns:
        # Convert to numeric and sort by column name
        filtered_df['mean_points_numeric'] = pd.to_numeric(filtered_df['mean_points'], errors='coerce')
        filtered_df = filtered_df.sort_values('mean_points_numeric', ascending=True, na_position='last')
        filtered_df = filtered_df.drop(columns=['mean_points_numeric'])
    elif 'Cost (Low to High)' in sort_by and 'now_cost' in filtered_df.columns:
        filtered_df = filtered_df.sort_values('now_cost', ascending=True, na_position='last')
    elif 'Cost (High to Low)' in sort_by and 'now_cost' in filtered_df.columns:
        filtered_df = filtered_df.sort_values('now_cost', ascending=False, na_position='last')
    elif 'Name (A-Z)' in sort_by and 'web_name' in filtered_df.columns:
        filtered_df = filtered_df.sort_values('web_name', ascending=True, na_position='last')
    elif 'Ownership (High to Low)' in sort_by and 'selected_by_percent' in filtered_df.columns:
        # Convert to numeric and sort by column name
        filtered_df['selected_by_percent_numeric'] = pd.to_numeric(filtered_df['selected_by_percent'], errors='coerce')
        filtered_df = filtered_df.sort_values('selected_by_percent_numeric', ascending=False, na_position='last')
        filtered_df = filtered_df.drop(columns=['selected_by_percent_numeric'])
    elif 'Form (High to Low)' in sort_by and 'form' in filtered_df.columns:
        # Convert to numeric and sort by column name
        filtered_df['form_numeric'] = pd.to_numeric(filtered_df['form'], errors='coerce')
        filtered_df = filtered_df.sort_values('form_numeric', ascending=False, na_position='last')
        filtered_df = filtered_df.drop(columns=['form_numeric'])
    elif 'Value (Points/Cost)' in sort_by and 'value' in filtered_df.columns:
        filtered_df = filtered_df.sort_values('value', ascending=False, na_position='last')
    
    # Limit results
    show_top = filters.get('show_top', 50)
    filtered_df = filtered_df.head(show_top)
    
    return filtered_df


def create_export_data(df: pd.DataFrame, position: str) -> bytes:
    """Create exportable CSV data."""
    # Don't format the export data - keep raw numbers
    export_df = df.copy()
    
    # Convert cost to millions if needed
    if 'now_cost' in export_df.columns:
        cost_values = export_df['now_cost']
        if cost_values.max() > 20:  # In tenths
            export_df['now_cost'] = cost_values / 10
    
    # Create CSV
    output = io.StringIO()
    export_df.to_csv(output, index=False)
    return output.getvalue().encode('utf-8')


def display_position_summary(df: pd.DataFrame, position: str) -> None:
    """Display summary statistics for position."""
    if df.empty:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Players", len(df))
    
    with col2:
        if 'mean_points' in df.columns:
            avg_proj = pd.to_numeric(df['mean_points'], errors='coerce').mean()
            st.metric("Avg Projection", f"{avg_proj:.1f}" if pd.notna(avg_proj) else "N/A")
    
    with col3:
        if 'now_cost' in df.columns:
            cost_values = df['now_cost']
            if cost_values.max() > 20:  # In tenths
                cost_values = cost_values / 10
            avg_cost = cost_values.mean()
            st.metric("Avg Cost", f"£{avg_cost:.1f}M" if pd.notna(avg_cost) else "N/A")
    
    with col4:
        if 'selected_by_percent' in df.columns:
            avg_own = pd.to_numeric(df['selected_by_percent'], errors='coerce').mean()
            st.metric("Avg Ownership", f"{avg_own:.1f}%" if pd.notna(avg_own) else "N/A")


def main():
    """Main player database page."""
    
    st.title("🗃️ Player Database")
    st.markdown("### Comprehensive Player Metrics by Position")
    
    # Get current gameweek
    target_gw = get_gameweek_selector()
    
    # Load predictions
    predictions_df = load_predictions_cached(target_gw)
    
    if predictions_df is None or predictions_df.empty:
        st.error(f"No predictions available for GW {target_gw}")
        st.info("Run the training pipeline to generate predictions")
        return
    
    # Enrich predictions with current FPL data
    predictions_df = enrich_with_current_fpl_data(predictions_df)
    
    # Build confidence intervals
    predictions_df = build_confidence_intervals(predictions_df)
    
    # Training mode indicator
    config = load_config()
    staging_config = config.get("training", {}).get("staging", {})
    warm_until_gw = staging_config.get("warm_until_gw", 8)
    
    if target_gw < warm_until_gw:
        st.info(f"🌱 **Training Mode: Warm Start** (GW {target_gw} < {warm_until_gw})")
    else:
        st.success(f"🔥 **Training Mode: Full ML** (GW {target_gw} ≥ {warm_until_gw})")
    
    # Overall statistics
    with st.container():
        st.markdown("#### 📊 Database Overview")
        
        overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
        
        with overview_col1:
            st.metric("Total Players", len(predictions_df))
        
        with overview_col2:
            unique_teams = predictions_df['team_name'].nunique() if 'team_name' in predictions_df.columns else 0
            st.metric("Teams", unique_teams)
        
        with overview_col3:
            avg_projection = pd.to_numeric(predictions_df.get('mean_points', [0]), errors='coerce').mean()
            st.metric("Avg Projection", f"{avg_projection:.1f}" if pd.notna(avg_projection) else "N/A")
        
        with overview_col4:
            if 'now_cost' in predictions_df.columns:
                cost_values = predictions_df['now_cost']
                if cost_values.max() > 20:  # In tenths
                    cost_values = cost_values / 10
                avg_cost = cost_values.mean()
                st.metric("Avg Cost", f"£{avg_cost:.1f}M" if pd.notna(avg_cost) else "N/A")
    
    st.divider()
    
    # Position-based tabs
    gk_tab, def_tab, mid_tab, fwd_tab, export_tab = st.tabs(["🥅 Goalkeepers", "🛡️ Defenders", "⚽ Midfielders", "🎯 Forwards", "📤 Export"])
    
    positions = {
        'GK': (gk_tab, "🥅 Goalkeepers"),
        'DEF': (def_tab, "🛡️ Defenders"), 
        'MID': (mid_tab, "⚽ Midfielders"),
        'FWD': (fwd_tab, "🎯 Forwards")
    }
    
    for position, (tab, title) in positions.items():
        with tab:
            st.subheader(title)
            
            # Filter data for position
            position_df = predictions_df[predictions_df.get('position') == position].copy()
            
            if position_df.empty:
                st.warning(f"No {position} players found in predictions")
                continue
            
            # Display summary
            display_position_summary(position_df, position)
            
            st.markdown("---")
            
            # Filters
            filters = create_player_filters(position_df, position)
            
            # Apply filters
            filtered_df = apply_filters(position_df, filters)
            
            st.markdown(f"**Showing {len(filtered_df)} {position} players** (filtered from {len(position_df)} total)")
            
            if not filtered_df.empty:
                # Format for display
                display_df = format_player_metrics_table(filtered_df, position)
                
                # Display table
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    height=600
                )
                
                # Quick stats for filtered data
                with st.expander(f"📈 {position} Statistics (Filtered Data)", expanded=False):
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    
                    with stats_col1:
                        if 'mean_points' in filtered_df.columns:
                            proj_values = pd.to_numeric(filtered_df['mean_points'], errors='coerce')
                            st.markdown(f"""
                            **Projections:**
                            - Highest: {proj_values.max():.1f}
                            - Average: {proj_values.mean():.1f}
                            - Lowest: {proj_values.min():.1f}
                            """)
                    
                    with stats_col2:
                        if 'now_cost' in filtered_df.columns:
                            cost_values = filtered_df['now_cost']
                            if cost_values.max() > 20:  # In tenths
                                cost_values = cost_values / 10
                            st.markdown(f"""
                            **Pricing:**
                            - Most Expensive: £{cost_values.max():.1f}M
                            - Average: £{cost_values.mean():.1f}M
                            - Cheapest: £{cost_values.min():.1f}M
                            """)
                    
                    with stats_col3:
                        if 'selected_by_percent' in filtered_df.columns:
                            own_values = pd.to_numeric(filtered_df['selected_by_percent'], errors='coerce')
                            st.markdown(f"""
                            **Ownership:**
                            - Highest: {own_values.max():.1f}%
                            - Average: {own_values.mean():.1f}%
                            - Lowest: {own_values.min():.1f}%
                            """)
            else:
                st.info(f"No {position} players match the current filters")
    
    with export_tab:
        st.subheader("📤 Export Player Data")
        st.markdown("Export player data in CSV format for external analysis")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            # Export options
            export_position = st.selectbox(
                "Select Position to Export",
                options=['All', 'GK', 'DEF', 'MID', 'FWD'],
                index=0
            )
            
            export_format = st.selectbox(
                "Export Format",
                options=['CSV', 'JSON'],
                index=0
            )
            
            include_raw_data = st.checkbox(
                "Include Raw Prediction Data",
                value=True,
                help="Include all prediction columns (p10, p50, p90, etc.)"
            )
        
        with export_col2:
            # Export preview
            if export_position == 'All':
                export_df = predictions_df.copy()
                export_title = "All Players"
            else:
                export_df = predictions_df[predictions_df.get('position') == export_position].copy()
                export_title = f"{export_position} Players"
            
            st.markdown(f"**Export Preview: {export_title}**")
            st.markdown(f"- Players: {len(export_df)}")
            st.markdown(f"- Columns: {len(export_df.columns)}")
            
            if not include_raw_data:
                # Remove detailed prediction columns for cleaner export
                columns_to_remove = ['std_points', 'p10', 'p50', 'p90', 'prob_double_digits', 'prob_15_plus']
                export_df = export_df.drop(columns=[col for col in columns_to_remove if col in export_df.columns])
                st.markdown(f"- Columns (filtered): {len(export_df.columns)}")
        
        st.markdown("---")
        
        # Generate export
        if st.button("📥 Generate Export", use_container_width=True):
            if not export_df.empty:
                try:
                    if export_format == 'CSV':
                        # Prepare CSV data
                        output = io.StringIO()
                        export_df.to_csv(output, index=False)
                        csv_data = output.getvalue().encode('utf-8')
                        
                        # Download button
                        filename = f"fpl_players_{export_position.lower()}_{target_gw}.csv"
                        st.download_button(
                            label=f"⬇️ Download {export_title} CSV",
                            data=csv_data,
                            file_name=filename,
                            mime='text/csv',
                            use_container_width=True
                        )
                        
                        st.success(f"✅ CSV export ready: {len(export_df)} players")
                    
                    elif export_format == 'JSON':
                        # Prepare JSON data
                        json_data = export_df.to_json(orient='records', indent=2)
                        
                        # Download button
                        filename = f"fpl_players_{export_position.lower()}_{target_gw}.json"
                        st.download_button(
                            label=f"⬇️ Download {export_title} JSON",
                            data=json_data,
                            file_name=filename,
                            mime='application/json',
                            use_container_width=True
                        )
                        
                        st.success(f"✅ JSON export ready: {len(export_df)} players")
                    
                except Exception as e:
                    st.error(f"Error generating export: {e}")
            else:
                st.warning("No data to export")
        
        # Export examples
        with st.expander("💡 Export Use Cases", expanded=False):
            st.markdown("""
            **Common use cases for exported data:**
            
            🔍 **External Analysis**
            - Import into Excel/Google Sheets for custom analysis
            - Use with external tools like Tableau or Power BI
            - Perform statistical analysis in R or Python
            
            📊 **Data Science Projects**
            - Train your own prediction models
            - Analyze player performance patterns
            - Compare with other FPL prediction sources
            
            🤝 **Sharing & Collaboration**
            - Share player data with league members
            - Create custom reports for mini-leagues
            - Backup data for historical analysis
            
            📈 **Portfolio Tracking**
            - Track player price changes over time
            - Monitor ownership percentage trends
            - Analyze prediction accuracy
            """)


if __name__ == "__main__":
    main()

