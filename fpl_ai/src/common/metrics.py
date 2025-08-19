"""
Performance metrics and evaluation utilities for FPL AI system.

Provides functions for calculating prediction accuracy, model evaluation,
and performance visualization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .config import get_config, get_logger

logger = get_logger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate comprehensive prediction metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        sample_weight: Optional sample weights
        
    Returns:
        Dictionary of metrics
    """
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    weights_clean = sample_weight[mask] if sample_weight is not None else None
    
    if len(y_true_clean) == 0:
        logger.warning("No valid predictions for metric calculation")
        return {}
    
    metrics = {}
    
    # Basic regression metrics
    metrics['mae'] = mean_absolute_error(y_true_clean, y_pred_clean, sample_weight=weights_clean)
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean, sample_weight=weights_clean))
    metrics['r2'] = r2_score(y_true_clean, y_pred_clean, sample_weight=weights_clean)
    
    # Mean prediction and actual
    metrics['mean_pred'] = np.average(y_pred_clean, weights=weights_clean)
    metrics['mean_actual'] = np.average(y_true_clean, weights=weights_clean)
    
    # Bias (systematic over/under prediction)
    metrics['bias'] = metrics['mean_pred'] - metrics['mean_actual']
    
    # Variance metrics
    metrics['pred_std'] = np.sqrt(np.average((y_pred_clean - metrics['mean_pred'])**2, weights=weights_clean))
    metrics['actual_std'] = np.sqrt(np.average((y_true_clean - metrics['mean_actual'])**2, weights=weights_clean))
    
    # Correlation
    correlation_matrix = np.corrcoef(y_true_clean, y_pred_clean)
    metrics['correlation'] = correlation_matrix[0, 1] if correlation_matrix.shape == (2, 2) else 0.0
    
    # Percentage metrics
    metrics['mape'] = np.mean(np.abs((y_true_clean - y_pred_clean) / np.maximum(np.abs(y_true_clean), 0.1))) * 100
    
    # Median metrics (robust to outliers)
    metrics['median_abs_error'] = np.median(np.abs(y_true_clean - y_pred_clean))
    
    # Hit rate (for classification-like evaluation)
    # How often we predict the correct "bin" (0-2, 3-5, 6+)
    def get_points_bin(points):
        return np.where(points <= 2, 0, np.where(points <= 5, 1, 2))
    
    true_bins = get_points_bin(y_true_clean)
    pred_bins = get_points_bin(y_pred_clean)
    metrics['bin_accuracy'] = np.mean(true_bins == pred_bins)
    
    # Custom FPL metrics
    # Capture rate: how well we identify high scorers (6+ points)
    high_scorers_mask = y_true_clean >= 6
    if np.any(high_scorers_mask):
        # Precision: of predicted high scorers, how many actually scored high
        pred_high_mask = y_pred_clean >= 6
        if np.any(pred_high_mask):
            metrics['high_scorer_precision'] = np.mean(y_true_clean[pred_high_mask] >= 6)
        else:
            metrics['high_scorer_precision'] = 0.0
        
        # Recall: of actual high scorers, how many did we predict
        metrics['high_scorer_recall'] = np.mean(y_pred_clean[high_scorers_mask] >= 6)
        
        # F1 score
        if metrics['high_scorer_precision'] + metrics['high_scorer_recall'] > 0:
            metrics['high_scorer_f1'] = 2 * (metrics['high_scorer_precision'] * metrics['high_scorer_recall']) / \
                                       (metrics['high_scorer_precision'] + metrics['high_scorer_recall'])
        else:
            metrics['high_scorer_f1'] = 0.0
    else:
        metrics['high_scorer_precision'] = 0.0
        metrics['high_scorer_recall'] = 0.0
        metrics['high_scorer_f1'] = 0.0
    
    # Rank correlation (how well we order players)
    from scipy.stats import spearmanr
    rank_corr, _ = spearmanr(y_true_clean, y_pred_clean)
    metrics['rank_correlation'] = rank_corr if not np.isnan(rank_corr) else 0.0
    
    metrics['n_samples'] = len(y_true_clean)
    
    return metrics


def calculate_position_metrics(df: pd.DataFrame, position_col: str = 'position') -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics by position.
    
    Args:
        df: DataFrame with true/pred values and positions
        position_col: Column name for position
        
    Returns:
        Dictionary mapping position to metrics
    """
    position_metrics = {}
    
    for position in df[position_col].unique():
        pos_df = df[df[position_col] == position]
        
        if len(pos_df) > 0:
            position_metrics[position] = calculate_metrics(
                pos_df['y_true'].values,
                pos_df['y_pred'].values,
                pos_df.get('sample_weight', None)
            )
    
    return position_metrics


def plot_predictions(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    position: Optional[str] = None,
    title: Optional[str] = None
) -> go.Figure:
    """
    Create prediction vs actual scatter plot.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        position: Position for title
        title: Custom title
        
    Returns:
        Plotly figure
    """
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(text="No valid data for plotting", showarrow=False)
        return fig
    
    # Calculate metrics for subtitle
    metrics = calculate_metrics(y_true_clean, y_pred_clean)
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=y_true_clean,
        y=y_pred_clean,
        mode='markers',
        marker=dict(
            size=6,
            opacity=0.6,
            color='lightblue',
            line=dict(width=1, color='navy')
        ),
        name='Predictions'
    ))
    
    # Add perfect prediction line
    min_val = min(y_true_clean.min(), y_pred_clean.min())
    max_val = max(y_true_clean.max(), y_pred_clean.max())
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Perfect Prediction'
    ))
    
    # Customize layout
    pos_text = f" ({position})" if position else ""
    title_text = title or f"Predictions vs Actual{pos_text}"
    subtitle = f"MAE: {metrics.get('mae', 0):.2f} | RMSE: {metrics.get('rmse', 0):.2f} | R²: {metrics.get('r2', 0):.3f}"
    
    fig.update_layout(
        title=f"{title_text}<br><sub>{subtitle}</sub>",
        xaxis_title="Actual Points",
        yaxis_title="Predicted Points",
        showlegend=True,
        template="plotly_white",
        width=600,
        height=500
    )
    
    return fig


def plot_residuals(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    position: Optional[str] = None
) -> go.Figure:
    """
    Create residual plot for model diagnostics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        position: Position for title
        
    Returns:
        Plotly figure
    """
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No valid data for plotting", showarrow=False)
        return fig
    
    residuals = y_true_clean - y_pred_clean
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Residuals vs Predicted', 'Residual Distribution'),
        horizontal_spacing=0.1
    )
    
    # Residuals vs predicted
    fig.add_trace(
        go.Scatter(
            x=y_pred_clean,
            y=residuals,
            mode='markers',
            marker=dict(size=4, opacity=0.6),
            name='Residuals'
        ),
        row=1, col=1
    )
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    
    # Histogram of residuals
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=30,
            name='Distribution',
            showlegend=False
        ),
        row=1, col=2
    )
    
    pos_text = f" ({position})" if position else ""
    fig.update_layout(
        title=f"Residual Analysis{pos_text}",
        template="plotly_white",
        width=900,
        height=400
    )
    
    fig.update_xaxes(title_text="Predicted Points", row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=1, col=1)
    fig.update_xaxes(title_text="Residuals", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    
    return fig


def plot_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10
) -> go.Figure:
    """
    Create calibration plot to assess prediction reliability.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        n_bins: Number of bins for calibration
        
    Returns:
        Plotly figure
    """
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No valid data for plotting", showarrow=False)
        return fig
    
    # Create bins based on predicted values
    bin_edges = np.percentile(y_pred_clean, np.linspace(0, 100, n_bins + 1))
    bin_centers = []
    bin_means = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (y_pred_clean >= bin_edges[i]) & (y_pred_clean <= bin_edges[i + 1])
        if np.sum(mask) > 0:
            bin_centers.append(np.mean(y_pred_clean[mask]))
            bin_means.append(np.mean(y_true_clean[mask]))
            bin_counts.append(np.sum(mask))
    
    fig = go.Figure()
    
    # Add calibration line
    fig.add_trace(go.Scatter(
        x=bin_centers,
        y=bin_means,
        mode='markers+lines',
        marker=dict(
            size=[np.sqrt(count) for count in bin_counts],
            sizemode='diameter',
            sizeref=2.*max(bin_counts)/(40.**2),
            sizemin=4
        ),
        name='Calibration'
    ))
    
    # Add perfect calibration line
    min_val = min(min(bin_centers), min(bin_means))
    max_val = max(max(bin_centers), max(bin_means))
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Perfect Calibration'
    ))
    
    fig.update_layout(
        title="Model Calibration (Reliability Diagram)",
        xaxis_title="Mean Predicted Points",
        yaxis_title="Mean Actual Points",
        template="plotly_white",
        width=600,
        height=500
    )
    
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importance_scores: np.ndarray,
    max_features: int = 20
) -> go.Figure:
    """
    Plot feature importance scores.
    
    Args:
        feature_names: List of feature names
        importance_scores: Importance scores
        max_features: Maximum features to show
        
    Returns:
        Plotly figure
    """
    # Sort by importance
    sorted_idx = np.argsort(importance_scores)[-max_features:]
    
    fig = go.Figure(go.Bar(
        x=importance_scores[sorted_idx],
        y=[feature_names[i] for i in sorted_idx],
        orientation='h',
        marker=dict(color='lightblue')
    ))
    
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        template="plotly_white",
        width=800,
        height=max(400, len(sorted_idx) * 25)
    )
    
    return fig


def create_performance_dashboard(
    results_dict: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create comprehensive performance dashboard.
    
    Args:
        results_dict: Dictionary with position results
        save_path: Optional path to save figure
        
    Returns:
        Plotly figure with subplots
    """
    positions = list(results_dict.keys())
    n_positions = len(positions)
    
    fig = make_subplots(
        rows=2, cols=n_positions,
        subplot_titles=[f"{pos} - Predictions" for pos in positions] + 
                       [f"{pos} - Residuals" for pos in positions],
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )
    
    # Add prediction plots
    for i, position in enumerate(positions):
        data = results_dict[position]
        
        if 'y_true' in data and 'y_pred' in data:
            y_true = data['y_true']
            y_pred = data['y_pred']
            
            # Remove NaN values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_true_clean) > 0:
                # Predictions vs actual
                fig.add_trace(
                    go.Scatter(
                        x=y_true_clean,
                        y=y_pred_clean,
                        mode='markers',
                        marker=dict(size=4, opacity=0.6),
                        showlegend=False
                    ),
                    row=1, col=i+1
                )
                
                # Perfect prediction line
                min_val = min(y_true_clean.min(), y_pred_clean.min())
                max_val = max(y_true_clean.max(), y_pred_clean.max())
                
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        line=dict(color='red', dash='dash'),
                        showlegend=False
                    ),
                    row=1, col=i+1
                )
                
                # Residuals
                residuals = y_true_clean - y_pred_clean
                fig.add_trace(
                    go.Scatter(
                        x=y_pred_clean,
                        y=residuals,
                        mode='markers',
                        marker=dict(size=4, opacity=0.6),
                        showlegend=False
                    ),
                    row=2, col=i+1
                )
                
                # Zero line for residuals
                fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=i+1)
    
    fig.update_layout(
        title="Model Performance Dashboard",
        template="plotly_white",
        width=300 * n_positions,
        height=800
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig
