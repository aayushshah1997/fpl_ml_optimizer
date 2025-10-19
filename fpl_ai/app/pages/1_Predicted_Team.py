import streamlit as st
import pandas as pd
from pathlib import Path
from fpl_ai.app.data_loaders import load_predictions_cached, enrich_with_current_fpl_data, validate_actual_results

def load_actual_results(gw: int) -> pd.DataFrame:
    """Load actual FPL results for a gameweek."""
    try:
        # Try CSV format first
        results_file = Path(f"fpl_ai/artifacts/gw{gw}_actual_results.csv")
        if results_file.exists():
            df = pd.read_csv(results_file)
            
            # Validate data quality
            if not validate_actual_results(df):
                st.warning(f"⚠️ Data quality issues detected for GW{gw}")
            else:
                st.success(f"✅ Loaded {len(df)} verified actual results for GW{gw}")
            
            return df
        
        st.warning(f"No actual results found for GW{gw}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading actual results: {e}")
        return pd.DataFrame()

def calculate_team_performance(predicted_team, actual_results):
    """Calculate predicted vs actual performance."""
    if actual_results.empty:
        return None
    
    total_predicted = 0
    total_actual = 0
    player_comparisons = []
    
    # Debug: Print what we're processing
    st.write(f"🔍 Debug: Processing {len(predicted_team)} players")
    
    for _, player in predicted_team.iterrows():
        player_id = player['element_id']
        
        # Find actual results for this player
        actual_player = actual_results[actual_results['element_id'] == player_id]
        
        if not actual_player.empty:
            actual_points = actual_player.iloc[0]['actual_points']
            predicted_points = player['proj_points']
            
            total_predicted += predicted_points
            total_actual += actual_points
            
            player_comparisons.append({
                'name': player['web_name'],
                'position': player['position'],
                'predicted': predicted_points,
                'actual': actual_points,
                'difference': actual_points - predicted_points
            })
    
    # Debug: Show totals
    st.write(f"🔍 Debug: Total predicted = {total_predicted:.1f}, Total actual = {total_actual}")
    
    return {
        'total_predicted': total_predicted,
        'total_actual': total_actual,
        'difference': total_actual - total_predicted,
        'accuracy': len(player_comparisons) / len(predicted_team) * 100,
        'player_comparisons': player_comparisons
    }

def main():
    st.title("🎯 Predicted Team")
    st.markdown("### AI-Optimized Team Selection with Risk Analysis")
    
    # Gameweek selector
    gw = st.selectbox("Select Gameweek", [7, 6, 5, 4, 3, 2, 1], index=0)
    
    # Load predictions
    predictions_df = load_predictions_cached(gw)

    if predictions_df is None or predictions_df.empty:
        st.error(f"No predictions available for GW {gw}")
        st.info("Run the training pipeline to generate predictions")
        return

    # Enrich with FPL data
    predictions_df = enrich_with_current_fpl_data(predictions_df)
    
    # Proper FPL squad building with position limits
    from fpl_ai.src.optimize.team_builder import build_squad_from_predictions
    from fpl_ai.src.optimize.formations import FormationValidator
    
    try:
        # Build a proper 15-player squad following FPL rules
        squad_players = build_squad_from_predictions(
            predictions_df, 
            budget=100.0  # FPL budget
        )
        
        if not squad_players or len(squad_players) != 15:
            st.error(f"Team optimization failed: Expected 15 players, got {len(squad_players) if squad_players else 0}")
            # Fallback to simple selection with warning
            top_players = predictions_df.nlargest(15, 'proj_points')
            st.warning("⚠️ Using simple selection (may not follow FPL rules)")
        else:
            # Convert list of dicts to DataFrame
            top_players = pd.DataFrame(squad_players)
            st.success("✅ Optimized team built following FPL rules")
            
            # Debug: Show squad details
            st.write(f"🔍 Debug: Squad has {len(top_players)} players")
            st.write(f"🔍 Debug: Total predicted points = {top_players['proj_points'].sum():.1f}")
            
    except Exception as e:
        st.error(f"Team optimization error: {e}")
        # Fallback to simple selection
        top_players = predictions_df.nlargest(15, 'proj_points')
        st.warning("⚠️ Using simple selection due to optimization error")
    
    st.subheader("📊 Predicted Team")
    
    # Display squad composition
    gk_count = len(top_players[top_players['position'] == 'GK'])
    def_count = len(top_players[top_players['position'] == 'DEF'])
    mid_count = len(top_players[top_players['position'] == 'MID'])
    fwd_count = len(top_players[top_players['position'] == 'FWD'])
    
    st.write(f"**Squad Composition:** {gk_count} GK, {def_count} DEF, {mid_count} MID, {fwd_count} FWD")
    
    # Validate squad composition
    if gk_count < 2 or gk_count > 2:
        st.error(f"❌ Invalid goalkeeper count: {gk_count} (must be exactly 2)")
    if def_count < 3 or def_count > 5:
        st.error(f"❌ Invalid defender count: {def_count} (must be 3-5)")
    if mid_count < 3 or mid_count > 5:
        st.error(f"❌ Invalid midfielder count: {mid_count} (must be 3-5)")
    if fwd_count < 1 or fwd_count > 3:
        st.error(f"❌ Invalid forward count: {fwd_count} (must be 1-3)")
    
    # Display the team
    for i, (_, player) in enumerate(top_players.iterrows(), 1):
        st.write(f"{i}. {player['web_name']} ({player['position']}) - {player['proj_points']:.1f} points - £{player['now_cost']:.1f}M")
    
    # PREDICTED VS ACTUAL COMPARISON
    st.subheader("📈 Predicted vs Actual Performance")
    
    # Load actual results
    actual_results = load_actual_results(gw)
    
    if not actual_results.empty:
        # Calculate performance
        performance = calculate_team_performance(top_players, actual_results)
        
        if performance and performance['player_comparisons']:
            # Overall metrics
            col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                    "Predicted Points",
                    f"{performance['total_predicted']:.1f}",
                    delta=None
            )
        
        with col2:
            st.metric(
                    "Actual Points", 
                    f"{performance['total_actual']:.1f}",
                    delta=f"{performance['difference']:+.1f}"
                )
            
            with col3:
                st.metric(
                    "Prediction Accuracy",
                    f"{performance['accuracy']:.1f}%",
                    delta=None
                )
            
            with col4:
                st.metric(
                    "Model Error",
                    f"{abs(performance['difference']):.1f}",
                    delta="Overprediction" if performance['difference'] < 0 else "Underprediction"
                )
            
            # Player-by-player comparison
            st.subheader("👥 Player-by-Player Comparison")
            
            comparison_df = pd.DataFrame(performance['player_comparisons'])
            if not comparison_df.empty:
                st.dataframe(comparison_df, use_container_width=True)
                
                # Summary insights
                st.subheader("🔍 Key Insights")
                
                overperformers = comparison_df[comparison_df['difference'] > 0]
                underperformers = comparison_df[comparison_df['difference'] < 0]
                
                if not overperformers.empty:
                    best_player = overperformers.loc[overperformers['difference'].idxmax()]
                    st.success(f"🏆 Best Prediction: {best_player['name']} (+{best_player['difference']:.1f} points)")
                
                if not underperformers.empty:
                    worst_player = underperformers.loc[underperformers['difference'].idxmin()]
                    st.error(f"📉 Worst Prediction: {worst_player['name']} ({worst_player['difference']:.1f} points)")
            else:
                st.warning("Could not calculate performance comparison - no matching players found")
    else:
        st.info(f"Actual results for GW{gw} not yet available")
        
        # Show what we would expect
        total_predicted = top_players['proj_points'].sum()
        st.write(f"**Total Predicted Points:** {total_predicted:.1f}")
        st.write("Actual results will be available after the gameweek deadline")

if __name__ == "__main__":
    main()
