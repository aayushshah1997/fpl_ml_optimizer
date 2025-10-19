"""
Captain and vice-captain selection logic for the FPL AI dashboard.

Handles captain selection based on projected points and form.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple


def select_captain_and_vice_captain(starting_xi: List[Dict]) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Select captain and vice-captain from starting XI based on projected points."""
    if not starting_xi:
        return None, None
    
    # Sort players by projected points (descending)
    sorted_xi = sorted(
        starting_xi, 
        key=lambda x: x.get('proj_points', x.get('mean_points', 0)), 
        reverse=True
    )
    
    # Select top 2 as captain and vice-captain
    captain = sorted_xi[0] if len(sorted_xi) >= 1 else None
    vice_captain = sorted_xi[1] if len(sorted_xi) >= 2 else None
    
    return captain, vice_captain


def calculate_captain_value(captain: Dict, vice_captain: Dict) -> Dict[str, Any]:
    """Calculate the expected value of captain selection."""
    if not captain:
        return {"captain_points": 0, "vice_points": 0, "total_value": 0}
    
    captain_points = captain.get('proj_points', captain.get('mean_points', 0)) * 2
    vice_points = vice_captain.get('proj_points', vice_captain.get('mean_points', 0)) if vice_captain else 0
    
    # Captain gets double points, vice gets normal points
    total_value = captain_points + vice_points
    
    return {
        "captain_points": captain_points,
        "vice_points": vice_points,
        "total_value": total_value,
        "captain_name": captain.get('web_name', 'Unknown'),
        "vice_name": vice_captain.get('web_name', 'Unknown') if vice_captain else 'None'
    }


def get_captain_alternatives(starting_xi: List[Dict], exclude_captain: Optional[Dict] = None) -> List[Dict]:
    """Get alternative captain options from starting XI."""
    if not starting_xi:
        return []
    
    # Filter out current captain if specified
    candidates = [p for p in starting_xi if p != exclude_captain]
    
    # Sort by projected points
    candidates.sort(key=lambda x: x.get('proj_points', x.get('mean_points', 0)), reverse=True)
    
    # Return top 3 alternatives
    return candidates[:3]


def analyze_captain_form(captain: Dict, predictions_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze captain's recent form and consistency."""
    if not captain or predictions_df.empty:
        return {"form_score": 0, "consistency": 0, "risk_level": "Unknown"}
    
    try:
        # Get captain's data
        captain_data = predictions_df[
            predictions_df['element_id'] == captain['element_id']
        ]
        
        if captain_data.empty:
            return {"form_score": 0, "consistency": 0, "risk_level": "Unknown"}
        
        # Calculate form metrics (if historical data available)
        current_projection = captain.get('proj_points', captain.get('mean_points', 0))
        
        # Simple form scoring based on projection
        if current_projection >= 8:
            form_score = 5
            risk_level = "Low"
        elif current_projection >= 6:
            form_score = 4
            risk_level = "Medium"
        elif current_projection >= 4:
            form_score = 3
            risk_level = "Medium-High"
        else:
            form_score = 2
            risk_level = "High"
        
        # Consistency (based on minutes and availability)
        minutes = captain.get('minutes', 0)
        if minutes >= 80:
            consistency = 5
        elif minutes >= 60:
            consistency = 4
        elif minutes >= 30:
            consistency = 3
        else:
            consistency = 2
        
        return {
            "form_score": form_score,
            "consistency": consistency,
            "risk_level": risk_level,
            "current_projection": current_projection,
            "minutes": minutes
        }
        
    except Exception as e:
        return {
            "form_score": 0,
            "consistency": 0,
            "risk_level": "Unknown",
            "error": str(e)
        }
