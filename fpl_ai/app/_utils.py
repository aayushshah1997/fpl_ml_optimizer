"""
Utility functions for the FPL AI Streamlit dashboard.

Common functions for loading data, formatting displays, and managing
session state across dashboard pages.
"""

import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import json
import yaml
import sys
from typing import Dict, Any, List, Optional, Tuple

# Add src to path for imports
app_dir = Path(__file__).parent
src_dir = app_dir.parent / "src"
sys.path.insert(0, str(src_dir))


@st.cache_data
def load_config() -> Dict[str, Any]:
    """Load configuration from settings.yaml."""
    try:
        config_path = Path(__file__).parent.parent / "settings.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
    except Exception:
        pass
    
    # Return default config if loading fails
    return {
        "io": {"out_dir": "artifacts"},
        "mc": {"num_scenarios": 2000, "lambda_risk": 0.2, "cvar_alpha": 0.2},
        "training": {"staging": {"warm_until_gw": 8}}
    }


@st.cache_data
def get_current_gw() -> int:
    """Get current gameweek (cached)."""
    try:
        from src.common.timeutil import get_current_gw
        return get_current_gw()
    except:
        return 1


def get_artifacts_dir() -> Path:
    """Get artifacts directory path."""
    config = load_config()
    return Path(config.get("io", {}).get("out_dir", "artifacts"))


@st.cache_data
def check_predictions_available(gameweek: int) -> bool:
    """Check if predictions are available for a gameweek."""
    artifacts_dir = get_artifacts_dir()
    pred_file = artifacts_dir / f"predictions_gw{gameweek}.csv"
    return pred_file.exists()


@st.cache_data
def load_predictions(gameweek: int) -> Optional[pd.DataFrame]:
    """Load predictions for a gameweek."""
    artifacts_dir = get_artifacts_dir()
    pred_file = artifacts_dir / f"predictions_gw{gameweek}.csv"
    
    if pred_file.exists():
        try:
            return pd.read_csv(pred_file)
        except Exception as e:
            st.error(f"Error loading predictions: {e}")
    
    return None


@st.cache_data
def load_model_performance() -> Dict[str, Any]:
    """Load model performance metrics."""
    artifacts_dir = get_artifacts_dir()
    models_dir = artifacts_dir / "models"
    
    performance = {}
    positions = ["GK", "DEF", "MID", "FWD"]
    
    # Load CV results
    for position in positions:
        cv_file = models_dir / f"cv_{position}.json"
        if cv_file.exists():
            try:
                with open(cv_file, 'r') as f:
                    cv_data = json.load(f)
                performance[f"cv_{position}"] = cv_data
            except Exception:
                pass
    
    # Load feature importance
    for position in positions:
        fi_file = models_dir / f"fi_{position}.csv"
        if fi_file.exists():
            try:
                fi_df = pd.read_csv(fi_file)
                performance[f"fi_{position}"] = fi_df
            except Exception:
                pass
    
    return performance


def format_player_display(
    player_row: pd.Series,
    include_form: bool = True,
    include_confidence: bool = True
) -> str:
    """Format player for display in tables."""
    name = player_row.get('web_name', 'Unknown')
    position = player_row.get('position', '?')
    team = player_row.get('team_name', player_row.get('team_short', '?'))
    
    # Cost
    cost = player_row.get('now_cost', 0)
    if cost > 20:  # Convert from tenths
        cost /= 10
    
    # Projection
    proj = player_row.get('proj_points', 0)
    
    # Build basic display
    display_parts = [f"{name} ({position}, {team[:3]})"]
    
    # Add league source badge if applicable
    # This indicates the player has prior data from lower-tier leagues
    # Tooltip: "Per-90s adjusted by league strength; MC uncertainty widened."
    if player_row.get('is_lowtier_league', False):
        uncertainty = player_row.get('prior_league_uncertainty', 0)
        if uncertainty > 0:
            display_parts.append("🌍 Prior: non-Top-5 (higher uncertainty)")
    
    display_parts.append(f"£{cost:.1f}M")
    display_parts.append(f"{proj:.1f}pts")
    
    # Recent form
    if include_form:
        if 'r3_points_per_game' in player_row:
            r3_form = player_row.get('r3_points_per_game', 0)
            display_parts.append(f"R3: {r3_form:.1f}")
        elif 'r3_points' in player_row:
            r3_total = player_row.get('r3_points', 0)
            display_parts.append(f"R3: {r3_total:.0f}")
    
    # Confidence intervals
    if include_confidence and 'p10' in player_row and 'p90' in player_row:
        p10 = player_row.get('p10', 0)
        p90 = player_row.get('p90', 0)
        display_parts.append(f"[{p10:.1f}-{p90:.1f}]")
    
    return " | ".join(display_parts)


def build_confidence_intervals(df: pd.DataFrame, minutes_uncertainty: float = 0.2) -> pd.DataFrame:
    """Build confidence intervals for predictions."""
    if df.empty:
        return df
    
    result_df = df.copy()
    
    # If Monte Carlo results are available, use them
    if all(col in df.columns for col in ['mean', 'std', 'p10', 'p90']):
        return result_df
    
    # Otherwise, build simple confidence intervals
    if 'proj_points' in df.columns:
        # Use position-based uncertainty if std not available
        position_uncertainty = {'GK': 1.8, 'DEF': 2.0, 'MID': 2.8, 'FWD': 3.0}
        
        if 'prediction_std' not in df.columns:
            result_df['prediction_std'] = result_df.get('position', 'MID').map(
                position_uncertainty
            ).fillna(2.5)
        
        # Build confidence intervals
        mean_pts = result_df['proj_points']
        std_pts = result_df['prediction_std']
        
        result_df['mean'] = mean_pts
        result_df['std'] = std_pts
        result_df['p10'] = mean_pts - 1.28 * std_pts  # 10th percentile
        result_df['p90'] = mean_pts + 1.28 * std_pts  # 90th percentile
        
        # Ensure non-negative
        for col in ['p10', 'p90', 'mean']:
            if col in result_df.columns:
                result_df[col] = result_df[col].clip(lower=0)
    
    return result_df


def create_position_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create position-wise summary statistics."""
    if df.empty or 'position' not in df.columns:
        return pd.DataFrame()
    
    summary_data = []
    
    for position in ['GK', 'DEF', 'MID', 'FWD']:
        pos_df = df[df['position'] == position]
        
        if not pos_df.empty:
            summary = {
                'Position': position,
                'Players': len(pos_df),
                'Avg Projection': pos_df.get('proj_points', pd.Series([0])).mean(),
                'Top Projection': pos_df.get('proj_points', pd.Series([0])).max(),
                'Avg Cost': pos_df.get('now_cost', pd.Series([5])).mean(),
                'Cost Range': f"£{pos_df.get('now_cost', pd.Series([5])).min():.1f}-{pos_df.get('now_cost', pd.Series([5])).max():.1f}M"
            }
            
            # Adjust cost if in tenths
            if summary['Avg Cost'] > 20:
                summary['Avg Cost'] /= 10
                cost_min = pos_df.get('now_cost', pd.Series([5])).min() / 10
                cost_max = pos_df.get('now_cost', pd.Series([5])).max() / 10
                summary['Cost Range'] = f"£{cost_min:.1f}-{cost_max:.1f}M"
            
            summary['Avg Cost'] = f"£{summary['Avg Cost']:.1f}M"
            
            summary_data.append(summary)
    
    return pd.DataFrame(summary_data)


def format_team_display(
    squad: List[Dict[str, Any]],
    predictions: pd.DataFrame,
    formation: Optional[Tuple[int, int, int]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Format team for display with formation."""
    if not squad:
        return pd.DataFrame(), {}
    
    # Convert squad to DataFrame
    squad_df = pd.DataFrame(squad)
    
    # Merge with predictions if available
    if not predictions.empty and 'element_id' in squad_df.columns:
        # Include league source columns if available
        pred_columns = ['element_id', 'proj_points', 'web_name', 'position', 'team_name', 'now_cost']
        if 'is_lowtier_league' in predictions.columns:
            pred_columns.extend(['is_lowtier_league', 'prior_league_uncertainty'])
        
        display_df = squad_df.merge(
            predictions[pred_columns],
            on='element_id',
            how='left',
            suffixes=('', '_pred')
        )
        
        # Use prediction data where available
        for col in ['proj_points', 'web_name', 'position', 'team_name', 'now_cost']:
            pred_col = f"{col}_pred"
            if pred_col in display_df.columns:
                display_df[col] = display_df[pred_col].fillna(display_df.get(col, ''))
    else:
        display_df = squad_df.copy()
    
    # Add display formatting
    if 'web_name' in display_df.columns:
        display_df['Display'] = display_df.apply(
            lambda row: format_player_display(row, include_form=False, include_confidence=False),
            axis=1
        )
    
    # Calculate team summary
    summary = {
        'total_players': len(squad),
        'total_cost': display_df.get('now_cost', pd.Series([0])).sum(),
        'expected_points': display_df.get('proj_points', pd.Series([0])).sum(),
        'formation': formation or (0, 0, 0)
    }
    
    # Adjust cost if in tenths
    if summary['total_cost'] > 200:
        summary['total_cost'] /= 10
    
    return display_df, summary


def display_confidence_chart(df: pd.DataFrame, top_n: int = 15) -> None:
    """Display confidence interval chart using Streamlit."""
    if df.empty or not all(col in df.columns for col in ['proj_points', 'web_name']):
        st.warning("Insufficient data for confidence chart")
        return
    
    # Get top players
    chart_df = df.nlargest(top_n, 'proj_points').copy()
    
    # Build chart data
    chart_data = pd.DataFrame({
        'Player': chart_df['web_name'],
        'Projection': chart_df['proj_points'],
        'P10': chart_df.get('p10', chart_df['proj_points'] - 2),
        'P90': chart_df.get('p90', chart_df['proj_points'] + 2)
    })
    
    # Create chart
    st.bar_chart(
        chart_data.set_index('Player')['Projection'],
        height=400
    )


def get_gameweek_selector() -> int:
    """Get gameweek from session state or default."""
    return st.session_state.get('selected_gw', get_current_gw())


def display_formation_grid(
    starting_xi: List[Dict[str, Any]],
    formation: Tuple[int, int, int],
    captain_id: Optional[int] = None,
    vice_captain_id: Optional[int] = None
) -> None:
    """Display starting XI in formation grid."""
    if not starting_xi or len(starting_xi) != 11:
        st.warning("Invalid starting XI for formation display")
        return
    
    # Group players by position
    by_position = {'GK': [], 'DEF': [], 'MID': [], 'FWD': []}
    
    for player in starting_xi:
        position = player.get('position', 'MID')
        if position in by_position:
            by_position[position].append(player)
    
    # Display in formation
    def format_player_card(player: Dict[str, Any]) -> str:
        name = player.get('web_name', 'Unknown')
        proj = player.get('proj_points', 0)
        player_id = player.get('element_id')
        
        # Captain indicators
        indicator = ""
        if player_id == captain_id:
            indicator = " (C)"
        elif player_id == vice_captain_id:
            indicator = " (VC)"
        
        return f"**{name}**{indicator}  \n{proj:.1f} pts"
    
    # Display formation rows
    st.markdown("### 🏟️ Formation View")
    
    # Forwards
    if by_position['FWD']:
        fwd_cols = st.columns(len(by_position['FWD']))
        for i, player in enumerate(by_position['FWD']):
            with fwd_cols[i]:
                st.markdown(format_player_card(player))
    
    # Midfielders
    if by_position['MID']:
        mid_cols = st.columns(len(by_position['MID']))
        for i, player in enumerate(by_position['MID']):
            with mid_cols[i]:
                st.markdown(format_player_card(player))
    
    # Defenders
    if by_position['DEF']:
        def_cols = st.columns(len(by_position['DEF']))
        for i, player in enumerate(by_position['DEF']):
            with def_cols[i]:
                st.markdown(format_player_card(player))
    
    # Goalkeeper
    if by_position['GK']:
        gk_col1, gk_col2, gk_col3 = st.columns([1, 1, 1])
        with gk_col2:
            st.markdown(format_player_card(by_position['GK'][0]))


def create_comparison_table(
    predicted_team: List[Dict[str, Any]],
    user_team: Optional[List[Dict[str, Any]]] = None,
    predictions_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Create team comparison table."""
    if not predicted_team:
        return pd.DataFrame()
    
    # Build comparison data
    comparison_data = []
    
    # Process predicted team
    for player in predicted_team:
        row = {
            'Name': player.get('web_name', 'Unknown'),
            'Position': player.get('position', '?'),
            'Team': player.get('team_name', '?'),
            'AI_Selected': True,
            'User_Selected': False,
            'Projection': player.get('proj_points', 0),
            'Cost': player.get('now_cost', 0)
        }
        
        if row['Cost'] > 20:
            row['Cost'] /= 10
        
        comparison_data.append(row)
    
    # Add user team if provided
    if user_team and predictions_df is not None:
        user_ids = {p.get('element_id') for p in user_team}
        pred_ids = {p.get('element_id') for p in predicted_team}
        
        # Mark overlapping players
        for i, row in enumerate(comparison_data):
            player_id = predicted_team[i].get('element_id')
            if player_id in user_ids:
                comparison_data[i]['User_Selected'] = True
        
        # Add user-only players
        for user_player in user_team:
            user_id = user_player.get('element_id')
            if user_id not in pred_ids:
                # Get prediction data
                pred_data = predictions_df[predictions_df['element_id'] == user_id]
                
                if not pred_data.empty:
                    pred_row = pred_data.iloc[0]
                    row = {
                        'Name': pred_row.get('web_name', user_player.get('web_name', 'Unknown')),
                        'Position': pred_row.get('position', user_player.get('position', '?')),
                        'Team': pred_row.get('team_name', user_player.get('team_name', '?')),
                        'AI_Selected': False,
                        'User_Selected': True,
                        'Projection': pred_row.get('proj_points', 0),
                        'Cost': pred_row.get('now_cost', user_player.get('now_cost', 0))
                    }
                    
                    if row['Cost'] > 20:
                        row['Cost'] /= 10
                    
                    comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Add selection indicators
    if 'AI_Selected' in df.columns and 'User_Selected' in df.columns:
        df['Selection'] = df.apply(
            lambda row: '✅ Both' if row['AI_Selected'] and row['User_Selected']
                       else '🤖 AI Only' if row['AI_Selected']
                       else '👤 User Only',
            axis=1
        )
    
    return df
