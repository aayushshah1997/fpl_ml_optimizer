# FPL AI - Elite Fantasy Premier League Analytics

A production-grade machine learning system for Fantasy Premier League (FPL) with multi-league data integration, staged training, and advanced optimization capabilities.

## 🚀 Key Features

- **Multi-League Data Integration**: Combines FPL data with FBRef API for comprehensive player history
- **Staged Training**: Intelligent training modes (Warm Start vs Full ML) based on gameweek progression
- **4-Year Walk-Forward Backtesting**: Temporal validation with EWMA weighting and no data leakage
- **Auto-Tuning with Optuna**: Hyperparameter optimization for LightGBM + Monte Carlo + captain policies + transfer strategy
- **Position-Specific Models**: Separate parameter tuning and modeling for GK/DEF/MID/FWD with position-aware optimization
- **Advanced Transfer Strategy**: Auto-optimization of when to roll vs make 0/1/2 transfers with hit cost analysis and bank utility
- **Intelligent Captain Policy**: Mean/CVaR/mixed captaincy selection using per-player Monte Carlo scenarios
- **Advanced Planning**: 10-week transfer planner with risk optimization, strategy tuning, and chip timing
- **Real-time Dashboard**: Beautiful Streamlit interface with performance analytics, strategy controls, and tuning results
- **Production Ready**: Comprehensive logging, caching, error handling, and monitoring

## 📊 Data Sources & Coverage

### How Many Years of Data?
- **Primary**: Last 3 full Premier League seasons + current season-to-date (configurable via `training.seasons_back: 4`)
- **Multi-League Priors**: Up to 2 seasons from La Liga, Serie A, Bundesliga, Ligue 1, Eredivisie for new signings
- **Recency Weighting**: Recent games weighted exponentially higher with season and league strength adjustments

### New Signings Support
- Automatic integration of FBRef API history from previous leagues
- Per-90 metrics adjusted by league strength coefficients
- Cold-start shrinkage to global means when minute data is scarce
- Manual player mapping via `data/fbr_player_map.csv`

### Recency & Form Tracking
- **Sample Weighting**: `w = exp(-λ * games_ago) × season_boost × league_strength_factor`
- **Recent Form**: Rolling 3/5/8 game windows displayed prominently in dashboard
- **Current Season Priority**: 1.3x boost vs 1.1x last season vs 0.7x older seasons

## 🔄 Rotation Risk Engine (API-Only, No Scraping)

### Automated Manager Discovery & Rotation Priors
- **Manager Resolution**: Current PL managers automatically discovered via FBR API
- **Historical Analysis**: Rotation patterns computed from previous season + current season YTD
- **All Competitions**: Includes European fixtures for comprehensive rotation modeling
- **Data-Driven Priors**: XI changes, starts variance, minutes distribution, bench rates
- **Fail-Soft**: CSV overrides for unknown managers, defaults to 0.05 until sufficient data

### Rotation Metrics Computed
1. **XI Change %**: Percentage of matches with ≥3 starting XI changes
2. **Starts Variance**: Normalized variance in player start frequencies  
3. **Minutes Shortfall**: `(90 - median_minutes)/90` for rotation depth
4. **Bench Rate**: Share of fit attackers with 0 minutes (rotation DNPs)

### Prior Blending Strategy
- **≥8 Current Matches**: Use current season data primarily
- **<8 Current Matches**: Weighted blend (60% current, 40% previous season)  
- **No Current Data**: Fall back to previous season or default (0.05)
- **Range**: Priors mapped to 0.03-0.30 scale for modeling stability

### Usage in Pipeline
```bash
# Update manager mappings and rotation priors
make rotation_update

# Outputs:
# - data/team_manager_map.csv (team_id, team_name, manager)  
# - data/manager_rotation_overrides.csv (manager, blended_prior, match_counts)
```

The rotation priors integrate seamlessly into:
- **Minutes Model**: `rotation_risk` feature for start probability
- **Monte Carlo**: Sigma inflation based on manager rotation tendency
- **Dashboard**: Manager audit page with data quality metrics

## 🛡️ Security & Privacy

- **Environment Variables**: All credentials stored in `.env` file only
- **No Hardcoded Secrets**: API keys and passwords never printed or logged
- **Local Processing**: All data processing happens locally on your machine

## ⚡ Quick Start

### 1. Initial Setup
```bash
# Clone and setup
git clone <your-repo>
cd fpl-ai
cp .env.example .env

# Add your credentials to .env
FBR_API_KEY=your_key_here  # Get from: curl -X POST https://fbrapi.com/generate_api_key
FPL_EMAIL=your_email@example.com
FPL_PASSWORD=your_password
FPL_ENTRY_ID=your_entry_id
```

### 2. Install Dependencies
```bash
make setup
```

### 3. First Training Run
```bash
# Train models (uses staging mode based on current GW)
make train

# Launch dashboard
make dash
```

### One-Step Full Data + Training
```bash
make bootstrap
```

This single command will:
1. Fetch all player and FBRef data
2. Build manager mappings and rotation priors  
3. Train all ML models with fresh data
4. Display cache creation summary

Perfect for first-time setup or when you want to refresh all data and models!

### Health Check
Run a quick repo audit:
```bash
make doctor
```

This will check your settings, environment, caches, and provide actionable recommendations for optimal FPL AI performance.

### 4. Set Your GW1 Baseline
The planner uses your actual GW1 team as the starting point. You can either:

**Option A: Automatic (recommended)**
- System will fetch your GW1 team using your Entry ID from `.env`

**Option B: Manual CSV Upload**
- Upload CSV via dashboard with columns: `element,purchase_price,selling_price,is_captain,bank`
- System saves this as `cache/team_state_gw1.json` for all future planning

## 🎯 Training Modes

### Warm Start (GW < 8)
- Lighter GBM models with fewer trees
- Higher shrinkage for non-Premier League data
- Reduced cross-validation folds for faster training
- Ideal for early season when data is limited

### Full ML (GW ≥ 8)  
- Full per-position gradient boosting models
- Complete 5-fold time series cross-validation
- Isotonic calibration on recent 8-game windows
- Maximum model complexity and accuracy

Mode is automatically selected based on current gameweek and `settings.yaml` configuration.

## 📈 Model Pipeline

### 1. Feature Engineering
- **Rolling Windows**: 3/5/8 game averages for all key metrics
- **Set Piece Roles**: Automated inference + community data + manual overrides
- **Team Form**: Rolling team and opponent strength metrics
- **Market Signals**: Price changes, ownership trends, transfer activity
- **H2H History**: Head-to-head shrunk features for fixture-specific insights

### 2. Minutes Prediction
- Separate Expected Minutes model using rotation patterns
- Accounts for fitness, congestion, tactical changes
- Critical for accurate per-90 projections

### 3. Points Prediction
- Per-position LightGBM models with 1200+ trees
- Features: attacking output, defensive metrics, set pieces, form, fixtures
- Isotonic calibration for well-calibrated probabilities

### 4. Risk Assessment
- Monte Carlo simulation with 2000+ scenarios
- Position-based uncertainty from historical residuals
- Team-level and opponent-level correlations
- CVaR (Conditional Value at Risk) optimization

## 🔧 Available Commands

```bash
# Core Pipeline
make train          # Full training + prediction
make predict        # Prediction only (requires trained models)
make audit          # Feature coverage analysis
make plan10         # 10-week transfer planning

# Dashboard & Analysis  
make dash           # Launch Streamlit dashboard
make cache-stats    # Check cache utilization

# Health Check
make doctor         # Repository audit and health check

# Development
make test           # Run test suite
make lint           # Code quality checks
make format         # Auto-format code
make clean          # Clear cache and artifacts
```

## 🔬 Backtest + Auto-Tuning

Run Optuna-powered hyperparameter optimization with 4-year walk-forward backtesting. Optimizes LightGBM parameters, Monte Carlo risk settings, and captain selection policies while maintaining strict temporal validation.

### Walk-Forward Backtesting
- **4-Season Window**: Tests on 2021-2025 with realistic data availability
- **Temporal Validation**: Uses only data available before each gameweek
- **EWMA Weighting**: Recent training samples weighted exponentially higher
- **Position-Specific Models**: Separate parameter tuning for GK/DEF/MID/FWD
- **Captain Policy Testing**: Evaluates mean/CVaR/mixed captain selection strategies

### Hyperparameter Optimization
- **Optuna Integration**: State-of-the-art Bayesian optimization
- **Multi-Objective**: Optimizes for total points, risk-adjusted points, or Sharpe-like ratios
- **Pruning**: MedianPruner stops unpromising trials early
- **Parallel-Safe**: Configurable trial parallelism for faster search

### Quick Commands
```bash
# Run full hyperparameter search (60 trials, 60min timeout)
make tune

# Quick tune for faster iteration (20 trials, 30min timeout)
make tune-quick

# Retrain models with best found parameters
make retrain

# Run tuning then retrain automatically
make tune_and_retrain

# View tuning results and leaderboard
make tune-results
```

### Advanced Usage
```bash
# Custom tuning session
python -m src.cli.tune_and_retrain tune --trials 100 --timeout 120

# Retrain with specific settings file
python -m src.cli.tune_and_retrain retrain --settings artifacts/tuning/best_settings.yaml --force

# View detailed results
python -m src.cli.tune_and_retrain results
```

### Tuning Search Space
The optimization explores:

**LightGBM Parameters:**
- `learning_rate`: [0.02, 0.03, 0.05]
- `num_leaves`: [31, 63, 95, 127]
- `n_estimators`: [600, 900, 1200, 1600]
- `subsample`, `colsample_bytree`: [0.7, 0.8, 0.9]
- `reg_alpha`, `reg_lambda`: [0.0, 0.1, 0.3, 0.6]

**Position-Specific Overrides:**
- Different optimal parameters for GK vs DEF vs MID vs FWD
- Automatically falls back to global parameters if not specified

**Monte Carlo & Risk:**
- `cvar_alpha`: [0.10, 0.15, 0.20, 0.25]
- `lambda_risk`: [0.10, 0.15, 0.20, 0.30]
- `minutes_uncertainty`: [0.12, 0.16, 0.20, 0.25]
- Team/opponent correlation levels

**Captain Strategy:**
- `policy`: ["mean", "cvar", "mix"]
- `mix_lambda`: [0.40, 0.50, 0.60, 0.70] (for mixed policy)

### Artifacts & Results
```
artifacts/tuning/
├── best_settings.yaml              # Best configuration found
├── leaderboard.json                # All trials ranked by performance  
├── study_summary.json              # Optimization session metadata
├── trial_X/                        # Per-trial results
│   ├── backtest_results.csv        # Walk-forward GW-by-GW performance
│   ├── metrics.json                # Aggregate performance metrics
│   └── config.json                 # Trial configuration
```

### Backtest Metrics
- **Total Points**: Raw cumulative points across all gameweeks
- **Risk-Adjusted Points**: CVaR-based downside protection measure
- **Sharpe-like**: Return/volatility ratio for consistency
- **Success Rate**: Percentage of gameweeks with successful predictions
- **Seasonal Breakdown**: Performance by individual season

The system respects temporal boundaries strictly - no future data leakage, only information available before each gameweek prediction.

## 📱 Dashboard Features

### 1. Predicted Team (`1_Predicted_Team.py`)
- **Best XI Optimization**: Risk-adjusted team selection with captain choice
- **Confidence Intervals**: P10/P50/P90 projections with Monte Carlo simulation
- **Recent Form Display**: Rolling 3/5 game form alongside projections  
- **Training Mode Indicator**: Shows whether using Warm Start or Full ML
- **Net Gain Analysis**: Compare against your current team

### 2. Model Performance (`2_Model_Performance.py`)
- **Cross-Validation Metrics**: MAE/RMSE by position from recent training
- **Feature Importance**: Top predictive features by position
- **Calibration Quality**: Reliability diagrams and residual analysis
- **Coverage Statistics**: Feature availability across player universe

### 3. 10-Week Planner (`3_10_Week_Planner.py`)
- **GW1 Baseline Integration**: Load your actual starting team
- **Auto-Transfer Strategy**: Backtests and tunes when to **roll**, take **0/1/2 transfers**, weigh **hit cost**, and place **bank utility** on future upgrades
- **Captaincy Policy**: Optimizes (C)/(VC) via **mean / CVaR / mixed** on per-player Monte Carlo outcomes
- **Per-Position Models**: Separate GBMs for GK/DEF/MID/FWD when enabled for position-specific optimization
- **Transfer Optimization**: Multi-week planning with chip timing and intelligent rolling thresholds
- **Risk Management**: Balance expected returns vs uncertainty with configurable CVaR parameters
- **Interactive Controls**: Adjust bank, transfers, starting gameweek, strategy settings, and captain policy

### 4. Managers Audit (`4_Managers_Audit.py`)
- **Rotation Risk Analysis**: Manager-specific rotation patterns and priors
- **Data Quality Metrics**: Coverage and reliability of rotation data
- **Team-by-Team Breakdown**: Current manager assignments and historical analysis

### 5. Backtest & Auto-Tuning (`5_Backtest_and_Tuning.py`)
- **Study Overview**: Optuna optimization session results and metrics
- **Trials Leaderboard**: All hyperparameter trials ranked by performance
- **Parameter Analysis**: Sensitivity analysis and parameter importance
- **Walk-Forward Results**: Temporal validation performance by season/gameweek
- **Best Settings Display**: Optimized configuration parameters

## 🏗️ Architecture

```
fpl_ai/
├── src/
│   ├── common/          # Config, caching, logging, utilities
│   ├── providers/       # Data APIs (FPL, FBRef, injuries, fixtures)
│   ├── features/        # Feature engineering pipeline
│   ├── plan/            # Transfer strategy optimization and planning
│   ├── tuning/          # Walk-forward backtesting and captain policies
│   ├── modeling/        # ML models, calibration, Monte Carlo
│   ├── optimize/        # Team optimization and formations
│   └── cli/             # Command-line interfaces
├── app/                 # Streamlit dashboard
├── data/                # Static data files and mappings
├── cache/               # API response caching
├── models/              # Trained model artifacts
└── artifacts/           # Training outputs and logs
```

## 🎛️ Configuration

Key settings in `settings.yaml`:

```yaml
training:
  seasons_back: 4                    # How many seasons to include
  extra_leagues: [12,11,20,13,10]   # Additional leagues for priors
  staging:
    warm_until_gw: 8                # When to switch to full ML

modeling:
  per_position:
    enabled: true                   # Enable position-specific models
  gbm:
    n_estimators: 1200              # Model complexity
  gbm_by_pos:                       # Position-specific overrides
    GK: {}                          # Goalkeeper parameters
    DEF: {}                         # Defender parameters
    MID: {}                         # Midfielder parameters  
    FWD: {}                         # Forward parameters
  calibration:
    window_gws: 8                   # Recent games for calibration

backtest:
  seasons: ["2021-2022","2022-2023","2023-2024","2024-2025"]
  start_gw: 2                       # First GW to backtest
  end_gw: 38                        # Last GW to backtest
  ewma_alpha: 0.85                  # Recency weighting
  objective: "risk_adj_points"      # Optimization target

autotune:
  enabled: true                     # Enable hyperparameter tuning
  n_trials: 60                      # Number of Optuna trials
  timeout_min: 60                   # Study timeout
  pruner: "median"                  # Trial pruning strategy

captain:
  policy: "mix"                     # Captain selection strategy
  mix_lambda: 0.60                  # Mean vs CVaR weight
  candidates: 5                     # Top N candidates to consider

mc:
  num_scenarios: 2000               # Monte Carlo scenarios
  lambda_risk: 0.20                 # Risk penalty weight

planner:
  horizon_gws: 10                   # Planning horizon
  max_transfers_per_gw: 2           # Transfer limits
```

## 🔍 Model Interpretability

- **Feature Importance**: Understand what drives predictions
- **SHAP Values**: Local explanations for individual players (planned)
- **Residual Analysis**: Identify model strengths and weaknesses
- **Calibration Plots**: Verify prediction reliability

## 📊 Performance Tracking

- **Cross-Validation**: Time series splits for realistic evaluation
- **Out-of-Sample Testing**: Never train on future data
- **Live Tracking**: Compare predictions vs actual points weekly
- **Feature Drift**: Monitor data quality and feature stability

## 🔄 Continuous Improvement

- **Weekly Retraining**: Models update with latest data
- **Feature Engineering**: Continuous addition of predictive signals
- **Hyperparameter Tuning**: Automated Optuna optimization with walk-forward validation
- **Position-Specific Tuning**: Separate optimization for goalkeeper, defense, midfield, forward models
- **Captain Policy Optimization**: Data-driven captain selection strategy tuning
- **Ensemble Methods**: Multiple model combinations (planned)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## ⚠️ Disclaimer

This tool is for educational and research purposes. Fantasy football involves risk, and past performance does not guarantee future results. Always do your own research and make informed decisions.

## 🙏 Acknowledgments

- Fantasy Premier League for providing the official API
- FBRef for comprehensive football statistics
- The FPL community for insights and data sharing
- Open source libraries that power this system

---

Built with ❤️ for the FPL community
