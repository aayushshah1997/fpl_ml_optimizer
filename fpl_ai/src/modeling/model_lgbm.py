"""
LightGBM models for FPL points prediction.

Implements per-position gradient boosting models with:
- Staged training (warm vs full modes)
- Time series cross-validation
- Comprehensive feature engineering
- Model calibration support
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
from ..common.config import get_config, get_logger
from ..common.cache import get_cache
from ..common.metrics import calculate_metrics, calculate_position_metrics

logger = get_logger(__name__)


class LGBMTrainer:
    """
    Trainer for per-position LightGBM models.
    """
    
    def __init__(self):
        """Initialize LGBM trainer."""
        self.config = get_config()
        self.cache = get_cache()
        
        # Model configuration
        self.positions = self.config.get_positions()
        self.gbm_params = self.config.get("modeling.gbm", {})
        
        # Training state
        self.models = {}
        self.feature_names = {}
        self.training_metrics = {}
        
        logger.info("LGBM trainer initialized")
    
    def train_models(
        self,
        training_data: pd.DataFrame,
        mode: str = "full",
        current_gw: int = 1,
        settings_override: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Train per-position models.
        
        Args:
            training_data: Training dataset
            mode: Training mode ('warm' or 'full')
            current_gw: Current gameweek for mode determination
            settings_override: Optional settings to override default config
            
        Returns:
            Training results and metrics
        """
        logger.info(f"Training LGBM models in {mode} mode for GW {current_gw}")
        
        if training_data.empty:
            logger.error("No training data provided")
            return {}
        
        # Apply settings override if provided
        if settings_override:
            # Temporarily update configuration
            original_gbm_params = self.gbm_params.copy()
            modeling_config = settings_override.get("modeling", {})
            
            # Update global gbm parameters
            if "gbm" in modeling_config:
                self.gbm_params.update(modeling_config["gbm"])
            
            # Store original config for restoration
            original_config_get = self.config.get
            
            def override_config_get(key: str, default=None):
                """Override config.get to use settings_override"""
                try:
                    # Navigate through the override settings
                    keys = key.split(".")
                    value = settings_override
                    for k in keys:
                        value = value[k]
                    return value
                except (KeyError, TypeError):
                    # Fall back to original config
                    return original_config_get(key, default)
            
            # Temporarily replace config.get
            self.config.get = override_config_get
            
            logger.info("Applied settings override for training")
        
        try:
            # Determine actual mode if auto
            actual_mode = self._determine_training_mode(mode, current_gw)
            
            # Train models for each position
            training_results = {}
            
            for position in self.positions:
                logger.info(f"Training {position} model...")
                
                # Filter data for this position
                pos_data = self._filter_position_data(training_data, position)
                
                if pos_data.empty:
                    logger.warning(f"No training data for position {position}")
                    continue
                
                # Train position model
                pos_results = self._train_position_model(pos_data, position, actual_mode)
                
                if pos_results and pos_results.get('model'):
                    self.models[position] = pos_results['model']
                    self.feature_names[position] = pos_results['features']
                    self.training_metrics[position] = pos_results['metrics']
                    
                    training_results[position] = pos_results
                    
                    logger.info(f"{position} model trained - MAE: {pos_results['metrics'].get('cv_mae', 0):.3f}")
            
            # Save models
            self._save_models()
            
            # Save training artifacts
            self._save_training_artifacts(training_results)
            
            overall_results = {
                'models_trained': list(training_results.keys()),
                'mode': actual_mode,
                'current_gw': current_gw,
                'position_results': training_results,
                'overall_metrics': self._calculate_overall_metrics(training_results)
            }
            
            logger.info(f"Training completed. Models: {len(training_results)}")
            return overall_results
            
        finally:
            # Restore original configuration if override was applied
            if settings_override:
                self.gbm_params = original_gbm_params
                self.config.get = original_config_get
                logger.debug("Restored original configuration")
    
    def _determine_training_mode(self, mode: str, current_gw: int) -> str:
        """Determine actual training mode."""
        if mode != "auto":
            return mode
        
        # Use config to determine mode
        return self.config.get_training_mode(current_gw)
    
    def _filter_position_data(self, df: pd.DataFrame, position: str) -> pd.DataFrame:
        """Filter training data for specific position."""
        if 'position' not in df.columns:
            logger.error("Position column not found in training data")
            return pd.DataFrame()
        
        pos_data = df[df['position'] == position].copy()
        
        # Remove rows with missing target
        target_col = self.config.get("training.target.name", "points_next")
        pos_data = pos_data.dropna(subset=[target_col])
        
        logger.debug(f"Position {position}: {len(pos_data)} training samples")
        return pos_data
    
    def _train_position_model(
        self,
        pos_data: pd.DataFrame,
        position: str,
        mode: str
    ) -> Dict[str, Any]:
        """Train model for specific position."""
        # Prepare training data
        X, y, sample_weights = self._prepare_position_training_data(pos_data, position)
        
        if len(X) == 0:
            logger.error(f"No valid training data for {position}")
            return {}
        
        # Get model parameters for this mode
        model_params = self._get_model_params(mode, position)
        
        # Train with cross-validation
        if mode == "warm":
            return self._train_warm_model(X, y, sample_weights, position, model_params)
        else:
            return self._train_full_model(X, y, sample_weights, position, model_params)
    
    def _prepare_position_training_data(
        self,
        pos_data: pd.DataFrame,
        position: str
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare features and target for position training."""
        target_col = self.config.get("training.target.name", "points_next")
        
        # Get position-specific features
        features = self._get_position_features(pos_data, position)
        
        # Build feature matrix
        X = pos_data[features].copy()
        
        # Fill missing values
        X = self._fill_missing_values(X, position)
        
        # Target and weights
        y = pos_data[target_col]
        sample_weights = pos_data.get('sample_weight', pd.Series(1.0, index=pos_data.index))
        
        logger.debug(f"{position} training data: {len(X)} samples, {len(features)} features")
        return X, y, sample_weights
    
    def _get_position_features(self, df: pd.DataFrame, position: str) -> List[str]:
        """Get feature list for specific position."""
        # Base features for all positions
        base_features = [
            # Recent form (rolling averages)
            'points_r3', 'points_r5', 'points_r8',
            'minutes_r3', 'minutes_r5', 'minutes_r8',
            'goals_r3', 'goals_r5', 'assists_r3', 'assists_r5',
            
            # Availability and rotation
            'avail_prob', 'rotation_risk', 'fixture_congestion',
            
            # Set pieces
            'pen_taker', 'fk_taker', 'corner_taker',
            
            # Team form
            'team_form_r3', 'team_form_r5', 'attack_strength_r3', 'defense_strength_r3',
            
            # Fixture context
            'fixture_difficulty', 'home_away_H',
            
            # Market signals
            'now_cost', 'selected_by_percent', 'transfers_in', 'transfers_out',
            'value_per_point', 'transfer_momentum',
            
            # H2H features
            'h2h_points_avg_shrunk', 'h2h_goals_avg_shrunk',
            
            # League strength and prior quality
            'league_strength_mult', 'is_lowtier_league', 'prior_league_uncertainty'
        ]
        
        # Position-specific features
        if position == 'GK':
            position_features = [
                'saves_r3', 'saves_r5', 'clean_sheets_r3', 'clean_sheets_r5',
                'goals_conceded_r3', 'goals_conceded_r5',
                'penalty_saves_r3', 'penalty_saves_r5',
                'gk_vs_high_scoring'
            ]
        elif position == 'DEF':
            position_features = [
                'clean_sheets_r3', 'clean_sheets_r5', 'goals_conceded_r3',
                'own_goals_r3', 'yellow_cards_r3', 'red_cards_r3',
                'tackles_r3', 'interceptions_r3', 'clearances_r3',
                'def_vs_strong_attack'
            ]
        elif position == 'MID':
            position_features = [
                'creativity_r3', 'creativity_r5', 'key_passes_r3', 'key_passes_r5',
                'big_chances_created_r3', 'expected_assists_r3',
                'passes_completed_r3', 'involvement_intensity_r3'
            ]
        else:  # FWD
            position_features = [
                'shots_r3', 'shots_r5', 'shots_on_target_r3',
                'big_chances_missed_r3', 'expected_goals_r3', 'expected_goals_r5',
                'att_vs_weak_defense', 'penalty_conversion_r5'
            ]
        
        # Combine features
        all_features = base_features + position_features
        
        # Filter to existing columns
        existing_features = [col for col in all_features if col in df.columns]
        
        # Add one-hot encoded features
        if 'home_away' in df.columns:
            df['home_away_H'] = (df['home_away'] == 'H').astype(int)
            if 'home_away_H' not in existing_features:
                existing_features.append('home_away_H')
        
        logger.debug(f"{position} features: {len(existing_features)} selected")
        return existing_features
    
    def _fill_missing_values(self, X: pd.DataFrame, position: str) -> pd.DataFrame:
        """Fill missing values with position-specific defaults."""
        X_filled = X.copy()
        
        # Position-specific defaults
        defaults = {
            'GK': {
                'points_r3': 2.5, 'points_r5': 2.5, 'points_r8': 2.5,
                'saves_r3': 1.5, 'clean_sheets_r3': 0.3,
                'avail_prob': 0.9, 'rotation_risk': 0.1
            },
            'DEF': {
                'points_r3': 2.8, 'points_r5': 2.8, 'points_r8': 2.8,
                'clean_sheets_r3': 0.25, 'goals_r3': 0.05,
                'avail_prob': 0.85, 'rotation_risk': 0.2
            },
            'MID': {
                'points_r3': 3.2, 'points_r5': 3.2, 'points_r8': 3.2,
                'assists_r3': 0.15, 'creativity_r3': 20,
                'avail_prob': 0.8, 'rotation_risk': 0.3
            },
            'FWD': {
                'points_r3': 3.8, 'points_r5': 3.8, 'points_r8': 3.8,
                'goals_r3': 0.25, 'shots_r3': 1.2,
                'avail_prob': 0.82, 'rotation_risk': 0.25
            }
        }
        
        pos_defaults = defaults.get(position, defaults['MID'])
        
        # Apply defaults
        for col in X_filled.columns:
            if col in pos_defaults:
                X_filled[col] = X_filled[col].fillna(pos_defaults[col])
            elif col.endswith('_r3') or col.endswith('_r5') or col.endswith('_r8'):
                X_filled[col] = X_filled[col].fillna(0)
            else:
                # General defaults
                if 'prob' in col or 'pct' in col:
                    X_filled[col] = X_filled[col].fillna(0.5)
                elif 'cost' in col or 'value' in col:
                    X_filled[col] = X_filled[col].fillna(X_filled[col].median())
                else:
                    X_filled[col] = X_filled[col].fillna(0)
        
        return X_filled
    
    def _get_model_params(self, mode: str, position: str) -> Dict[str, Any]:
        """Get LightGBM parameters for training mode and position."""
        base_params = self.gbm_params.copy()
        
        # Apply position-specific parameters from configuration if available
        gbm_by_pos = self.config.get("modeling.gbm_by_pos", {})
        if gbm_by_pos and position in gbm_by_pos:
            position_params = gbm_by_pos[position]
            # Only update parameters that are explicitly set (not empty/None)
            for key, value in position_params.items():
                if value is not None and value != "":
                    base_params[key] = value
            
            logger.debug(f"Applied position-specific parameters for {position}: {position_params}")
        
        # Mode-specific adjustments
        if mode == "warm":
            # Lighter model for warm start
            base_params.update({
                'num_leaves': min(31, base_params.get('num_leaves', 63)),
                'min_data_in_leaf': max(50, base_params.get('min_data_in_leaf', 25)),
                'learning_rate': base_params.get('learning_rate', 0.03) * 1.5,
                'feature_fraction': 0.7,
                'lambda_l2': base_params.get('reg_lambda', 0.3) * 2
            })
        
        # Fallback position-specific adjustments (only if not overridden by config)
        if 'reg_alpha' not in gbm_by_pos.get(position, {}) and position == 'GK':
            # Goalkeepers have less variance, can use more regularization
            base_params['reg_alpha'] = base_params.get('reg_alpha', 0.1) * 1.5
            base_params['reg_lambda'] = base_params.get('reg_lambda', 0.3) * 1.5
        elif 'num_leaves' not in gbm_by_pos.get(position, {}) and position == 'FWD':
            # Forwards have high variance, reduce overfitting
            base_params['num_leaves'] = min(base_params.get('num_leaves', 63), 31)
            base_params['min_data_in_leaf'] = max(base_params.get('min_data_in_leaf', 25), 40)
        
        return base_params
    
    def _train_warm_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series,
        position: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train warm start model with reduced complexity."""
        logger.debug(f"Training warm model for {position}")
        
        # Warm start uses fewer rounds and simpler CV
        params.update({
            'objective': 'regression',
            'metric': 'mae',
            'verbose': -1,
            'random_state': 42
        })
        
        try:
            # Simple 3-fold CV for warm start
            cv_folds = 3
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            
            oof_predictions = np.zeros(len(y))
            cv_scores = []
            feature_importance = np.zeros(len(X.columns))
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                w_train = sample_weights.iloc[train_idx]
                
                # Create datasets
                train_set = lgb.Dataset(X_train, label=y_train, weight=w_train)
                val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
                
                # Train with early stopping
                fold_model = lgb.train(
                    params,
                    train_set,
                    valid_sets=[val_set],
                    num_boost_round=300,  # Fewer rounds for warm start
                    callbacks=[lgb.early_stopping(15), lgb.log_evaluation(0)]
                )
                
                # Predictions
                val_pred = fold_model.predict(X_val, num_iteration=fold_model.best_iteration)
                oof_predictions[val_idx] = val_pred
                
                # Metrics
                fold_mae = mean_absolute_error(y_val, val_pred)
                cv_scores.append(fold_mae)
                
                # Feature importance
                feature_importance += fold_model.feature_importance(importance_type='gain')
                
                logger.debug(f"{position} warm fold {fold + 1}: MAE = {fold_mae:.3f}")
            
            # Train final model
            final_dataset = lgb.Dataset(X, label=y, weight=sample_weights)
            final_model = lgb.train(
                params,
                final_dataset,
                num_boost_round=int(np.mean([300] * cv_folds)),  # Use average rounds
                callbacks=[lgb.log_evaluation(0)]
            )
            
            # Calculate metrics
            metrics = calculate_metrics(y.values, oof_predictions, sample_weights.values)
            metrics.update({
                'cv_mae': np.mean(cv_scores),
                'cv_mae_std': np.std(cv_scores),
                'n_folds': cv_folds,
                'mode': 'warm'
            })
            
            # Feature importance
            feature_importance /= cv_folds
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            return {
                'model': final_model,
                'features': list(X.columns),
                'metrics': metrics,
                'oof_predictions': oof_predictions,
                'feature_importance': importance_df,
                'cv_scores': cv_scores
            }
            
        except Exception as e:
            logger.error(f"Error training warm model for {position}: {e}")
            return {}
    
    def _train_full_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series,
        position: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train full model with complete cross-validation."""
        logger.debug(f"Training full model for {position}")
        
        params.update({
            'objective': 'regression',
            'metric': 'mae',
            'verbose': -1,
            'random_state': 42
        })
        
        try:
            # Full 5-fold time series CV
            cv_folds = self.config.get("modeling.calibration.cv_folds", 5)
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            
            oof_predictions = np.zeros(len(y))
            cv_scores = []
            feature_importance = np.zeros(len(X.columns))
            fold_models = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                w_train = sample_weights.iloc[train_idx]
                
                # Create datasets
                train_set = lgb.Dataset(X_train, label=y_train, weight=w_train)
                val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
                
                # Train with early stopping
                fold_model = lgb.train(
                    params,
                    train_set,
                    valid_sets=[val_set],
                    num_boost_round=self.gbm_params.get('n_estimators', 1200),
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )
                
                fold_models.append(fold_model)
                
                # Predictions
                val_pred = fold_model.predict(X_val, num_iteration=fold_model.best_iteration)
                oof_predictions[val_idx] = val_pred
                
                # Metrics
                fold_mae = mean_absolute_error(y_val, val_pred)
                cv_scores.append(fold_mae)
                
                # Feature importance
                feature_importance += fold_model.feature_importance(importance_type='gain')
                
                logger.debug(f"{position} full fold {fold + 1}: MAE = {fold_mae:.3f}")
            
            # Train final model on all data
            final_dataset = lgb.Dataset(X, label=y, weight=sample_weights)
            avg_rounds = int(np.mean([m.best_iteration for m in fold_models]))
            
            final_model = lgb.train(
                params,
                final_dataset,
                num_boost_round=avg_rounds,
                callbacks=[lgb.log_evaluation(0)]
            )
            
            # Calculate comprehensive metrics
            metrics = calculate_metrics(y.values, oof_predictions, sample_weights.values)
            metrics.update({
                'cv_mae': np.mean(cv_scores),
                'cv_mae_std': np.std(cv_scores),
                'n_folds': cv_folds,
                'avg_rounds': avg_rounds,
                'mode': 'full'
            })
            
            # Feature importance
            feature_importance /= cv_folds
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            return {
                'model': final_model,
                'features': list(X.columns),
                'metrics': metrics,
                'oof_predictions': oof_predictions,
                'feature_importance': importance_df,
                'cv_scores': cv_scores,
                'fold_models': fold_models
            }
            
        except Exception as e:
            logger.error(f"Error training full model for {position}: {e}")
            return {}
    
    def _calculate_overall_metrics(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall metrics across all positions."""
        if not training_results:
            return {}
        
        overall_metrics = {}
        
        # Aggregate CV scores
        all_cv_scores = []
        position_maes = {}
        
        for position, results in training_results.items():
            if 'metrics' in results:
                cv_mae = results['metrics'].get('cv_mae', 0)
                all_cv_scores.append(cv_mae)
                position_maes[position] = cv_mae
        
        if all_cv_scores:
            overall_metrics.update({
                'overall_cv_mae': np.mean(all_cv_scores),
                'overall_cv_mae_std': np.std(all_cv_scores),
                'position_maes': position_maes,
                'best_position': min(position_maes.items(), key=lambda x: x[1])[0] if position_maes else None,
                'worst_position': max(position_maes.items(), key=lambda x: x[1])[0] if position_maes else None
            })
        
        return overall_metrics
    
    def _save_models(self) -> bool:
        """Save trained models to disk."""
        try:
            models_saved = 0
            
            for position, model in self.models.items():
                model_path = self.config.models_dir / f"model_points_{position}.pkl"
                
                model_data = {
                    'model': model,
                    'features': self.feature_names.get(position, []),
                    'metrics': self.training_metrics.get(position, {}),
                    'position': position
                }
                
                with open(model_path, 'wb') as f:
                    pickle.dump(model_data, f)
                
                models_saved += 1
                logger.debug(f"Saved {position} model to {model_path}")
            
            logger.info(f"Saved {models_saved} position models")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
            return False
    
    def _save_training_artifacts(self, training_results: Dict[str, Any]) -> bool:
        """Save training artifacts (metrics, feature importance, etc.)."""
        try:
            artifacts_dir = self.config.artifacts_dir
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            
            # Save CV metrics
            cv_metrics = {}
            for position, results in training_results.items():
                if 'metrics' in results:
                    cv_metrics[position] = results['metrics']
            
            cv_path = artifacts_dir / "cv_metrics.json"
            with open(cv_path, 'w') as f:
                json.dump(cv_metrics, f, indent=2, default=str)
            
            # Save feature importance
            for position, results in training_results.items():
                if 'feature_importance' in results:
                    fi_path = artifacts_dir / f"fi_{position}.csv"
                    results['feature_importance'].to_csv(fi_path, index=False)
            
            # Save residuals for calibration
            for position, results in training_results.items():
                if 'oof_predictions' in results and position in self.training_metrics:
                    # Reconstruct y_true from training (this is simplified)
                    residuals_data = {
                        'position': position,
                        'oof_available': len(results['oof_predictions']),
                        'mae': results['metrics'].get('mae', 0),
                        'rmse': results['metrics'].get('rmse', 0),
                        'std': np.std(results['oof_predictions']) if len(results['oof_predictions']) > 0 else 0
                    }
                    
                    residuals_path = artifacts_dir / f"residuals_{position}.json"
                    with open(residuals_path, 'w') as f:
                        json.dump(residuals_data, f, indent=2)
            
            logger.info("Saved training artifacts")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save training artifacts: {e}")
            return False


class LGBMPredictor:
    """
    Predictor using trained LightGBM models.
    """
    
    def __init__(self):
        """Initialize LGBM predictor."""
        self.config = get_config()
        self.cache = get_cache()
        
        # Model storage
        self.models = {}
        self.feature_names = {}
        self.model_metrics = {}
        
        logger.info("LGBM predictor initialized")
    
    def predict_points(
        self,
        prediction_data: pd.DataFrame,
        load_models: bool = True
    ) -> pd.DataFrame:
        """
        Predict points for all positions.
        
        Args:
            prediction_data: Data for prediction
            load_models: Whether to load models if not already loaded
            
        Returns:
            DataFrame with predictions
        """
        if prediction_data.empty:
            return pd.DataFrame()
        
        # Load models if needed
        if not self.models and load_models:
            self._load_models()
        
        if not self.models:
            logger.error("No trained models available")
            return pd.DataFrame()
        
        # Make predictions for each position
        all_predictions = []
        
        for position in self.config.get_positions():
            if position not in self.models:
                logger.warning(f"No model available for {position}")
                continue
            
            # Filter data for this position
            pos_data = prediction_data[prediction_data.get('position') == position].copy()
            
            if pos_data.empty:
                continue
            
            # Make predictions for this position
            pos_predictions = self._predict_position(pos_data, position)
            
            if not pos_predictions.empty:
                all_predictions.append(pos_predictions)
        
        if not all_predictions:
            logger.warning("No predictions generated")
            return pd.DataFrame()
        
        # Combine all predictions
        final_predictions = pd.concat(all_predictions, ignore_index=True)
        
        # Sort by predicted points (descending)
        final_predictions = final_predictions.sort_values('proj_points', ascending=False)
        
        logger.info(f"Generated predictions for {len(final_predictions)} players")
        return final_predictions
    
    def _predict_position(self, pos_data: pd.DataFrame, position: str) -> pd.DataFrame:
        """Make predictions for specific position."""
        try:
            model = self.models[position]
            features = self.feature_names[position]
            
            # Prepare features
            X = self._prepare_prediction_features(pos_data, features, position)
            
            if len(X) == 0:
                logger.warning(f"No valid features for {position} prediction")
                return pd.DataFrame()
            
            # Make predictions
            raw_predictions = model.predict(X, num_iteration=model.best_iteration)
            
            # Create results DataFrame
            results = pos_data[['element_id', 'web_name', 'team_name', 'position', 'now_cost']].copy()
            results['proj_points'] = raw_predictions
            
            # Add expected minutes if available (for per-90 calculations)
            if 'expected_minutes' in pos_data.columns:
                results['expected_minutes'] = pos_data['expected_minutes']
                # Calculate per-90 projection
                results['proj_per90'] = results['proj_points'] / (results['expected_minutes'] / 90)
            else:
                results['expected_minutes'] = 75  # Default
                results['proj_per90'] = results['proj_points'] / 0.83  # ~75 minutes
            
            # Add confidence metrics if available
            if position in self.model_metrics:
                mae = self.model_metrics[position].get('cv_mae', 2.0)
                results['prediction_std'] = mae * 1.25  # Rough uncertainty estimate
            else:
                results['prediction_std'] = 2.0  # Default uncertainty
            
            logger.debug(f"Predicted {len(results)} {position} players")
            return results
            
        except Exception as e:
            logger.error(f"Error predicting {position}: {e}")
            return pd.DataFrame()
    
    def _prepare_prediction_features(
        self,
        pos_data: pd.DataFrame,
        features: List[str],
        position: str
    ) -> pd.DataFrame:
        """Prepare features for prediction."""
        # Get available features
        available_features = []
        
        for feature in features:
            if feature in pos_data.columns:
                available_features.append(feature)
            else:
                logger.debug(f"Missing feature for {position}: {feature}")
        
        if not available_features:
            return pd.DataFrame()
        
        X = pos_data[available_features].copy()
        
        # Fill missing values using same logic as training
        trainer = LGBMTrainer()
        X = trainer._fill_missing_values(X, position)
        
        return X
    
    def _load_models(self) -> bool:
        """Load trained models from disk."""
        try:
            loaded_count = 0
            
            for position in self.config.get_positions():
                model_path = self.config.models_dir / f"model_points_{position}.pkl"
                
                if not model_path.exists():
                    logger.warning(f"Model file not found for {position}: {model_path}")
                    continue
                
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.models[position] = model_data['model']
                self.feature_names[position] = model_data['features']
                self.model_metrics[position] = model_data['metrics']
                
                loaded_count += 1
                logger.debug(f"Loaded {position} model")
            
            logger.info(f"Loaded {loaded_count} position models")
            return loaded_count > 0
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False


# Standalone functions for easy importing
def train(
    training_data: pd.DataFrame,
    mode: str = "full",
    current_gw: int = 1,
    gbm_params_by_pos: Optional[Dict[str, Dict]] = None,
    settings_override: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Train position-specific models (backward compatibility function).
    
    Args:
        training_data: Training dataset
        mode: Training mode ('warm' or 'full')
        current_gw: Current gameweek
        gbm_params_by_pos: Position-specific GBM parameters (legacy format)
        settings_override: Optional configuration override for tuning
        
    Returns:
        Training results dictionary
    """
    # Convert gbm_params_by_pos to settings_override format if provided
    if gbm_params_by_pos and not settings_override:
        settings_override = {
            "modeling": {
                "gbm_by_pos": gbm_params_by_pos
            }
        }
    elif gbm_params_by_pos and settings_override:
        # Merge with existing override
        if "modeling" not in settings_override:
            settings_override["modeling"] = {}
        if "gbm_by_pos" not in settings_override["modeling"]:
            settings_override["modeling"]["gbm_by_pos"] = {}
        settings_override["modeling"]["gbm_by_pos"].update(gbm_params_by_pos)
    
    trainer = LGBMTrainer()
    return trainer.train_models(training_data, mode, current_gw, settings_override)


def train_models(
    training_data: pd.DataFrame,
    mode: str = "full",
    current_gw: int = 1,
    settings_override: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Train position-specific models with optional settings override.
    
    Args:
        training_data: Training dataset
        mode: Training mode ('warm' or 'full')
        current_gw: Current gameweek
        settings_override: Optional configuration override for tuning
        
    Returns:
        Training results dictionary
    """
    trainer = LGBMTrainer()
    return trainer.train_models(training_data, mode, current_gw, settings_override)


def predict_points(
    prediction_data: pd.DataFrame, 
    model: Optional[Dict] = None,
    load_models: bool = True
) -> pd.DataFrame:
    """
    Predict points using trained models.
    
    Handles both per-position models (dictionary) and traditional single model.
    
    Args:
        prediction_data: Data for prediction
        model: Optional dictionary of per-position models or single model
        load_models: Whether to load models if not already loaded
        
    Returns:
        DataFrame with predictions
    """
    predictor = LGBMPredictor()
    
    # If model is provided as dictionary (per-position models)
    if isinstance(model, dict) and model:
        # Check if it's a dictionary of position models
        if any(pos in model for pos in ['GK', 'DEF', 'MID', 'FWD']):
            # Set the per-position models directly
            predictor.models = {}
            predictor.feature_names = {}
            predictor.model_metrics = {}
            
            for position, pos_model in model.items():
                if isinstance(pos_model, dict):
                    # Handle model dictionary format
                    predictor.models[position] = pos_model.get('model')
                    predictor.feature_names[position] = pos_model.get('features', [])
                    predictor.model_metrics[position] = pos_model.get('metrics', {})
                else:
                    # Handle direct model object
                    predictor.models[position] = pos_model
                    # Use default feature names if not provided
                    predictor.feature_names[position] = getattr(pos_model, 'feature_name_', [])
                    predictor.model_metrics[position] = {}
            
            return predictor.predict_points(prediction_data, load_models=False)
        
        # Handle single model wrapped in dictionary (legacy format)
        elif 'model' in model or 'models' in model:
            single_model = model.get('model') or model.get('models')
            if single_model:
                # For single model, predict for all positions using same model
                results = []
                
                for position in ['GK', 'DEF', 'MID', 'FWD']:
                    pos_data = prediction_data[prediction_data.get('position') == position].copy()
                    
                    if pos_data.empty:
                        continue
                    
                    # Use available features or fall back to basic ones
                    features = model.get('feature_names', model.get('features', []))
                    if not features:
                        # Basic feature set for fallback
                        features = [col for col in pos_data.columns if col.endswith('_r3') or col.endswith('_r5')]
                    
                    # Filter to available features
                    available_features = [f for f in features if f in pos_data.columns]
                    
                    if available_features:
                        X = pos_data[available_features].fillna(0)
                        try:
                            raw_pred = single_model.predict(X)
                            
                            result_df = pos_data[['element_id', 'web_name', 'team_name', 'position', 'now_cost']].copy()
                            result_df['proj_points'] = raw_pred
                            result_df['expected_minutes'] = 75
                            result_df['proj_per90'] = result_df['proj_points'] / 0.83
                            result_df['prediction_std'] = 2.0
                            
                            results.append(result_df)
                        except Exception as e:
                            logger.warning(f"Error predicting with single model for {position}: {e}")
                
                if results:
                    return pd.concat(results, ignore_index=True).sort_values('proj_points', ascending=False)
    
    # Default case: use standard prediction with model loading
    return predictor.predict_points(prediction_data, load_models)
