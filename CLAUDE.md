# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Premier League (Premier League) Monte Carlo simulation and forecasting application built with Streamlit. It predicts football match outcomes and season standings using statistical models (Poisson-based) and optionally integrates live betting odds for hybrid predictions.

## Development Commands

### Running the Application

```bash
# Run the Streamlit app locally
streamlit run app.py --server.port 5000

# Run with custom config (via environment variables)
DATABASE_URL=postgresql://... ODDS_API_KEY=... streamlit run app.py
```

### Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_odds_integration.py

# Run with verbose output
python -m pytest -v tests/
```

### Database Operations

The application supports both SQLite (default) and PostgreSQL:

```bash
# Set PostgreSQL connection
export DATABASE_URL="postgresql://user:password@host:port/dbname"

# Default SQLite database is automatically created at: data/db/premier_league.db
```

### Data Pipeline

Manual execution of key pipeline steps (normally done through the UI):

```bash
# From Python console:
from shl.data.scraper import Premier LeagueScraper
scraper = Premier LeagueScraper()
raw_data = scraper.scrape_matches(seasons=[2024, 2025])

# Clean data
from shl.data.cleaner import DataCleaner
cleaner = DataCleaner()
results, fixtures = cleaner.clean_data(raw_data)
```

## Architecture

The application follows a modular architecture under the `shl/` package:

### Core Modules

- **`data/`**: Data collection and processing
  - `scraper.py`: Web scraper for stats.swefootball.se (Premier League's official stats site)
  - `cleaner.py`: Data cleaning and validation
  - `odds_api.py`: Integration with The-Odds-API for live betting odds
  - `odds_schema.py`: Pydantic schemas for odds data
  - `strength.py`: Team strength calculator using historical data
  - `odds_strength_extractor.py`: Extract team strength signals from betting odds

- **`models/`**: Statistical prediction models
  - `poisson_model.py`: Poisson regression model with optional MLE and Dixon-Coles adjustments
  - `hybrid_model.py`: Hybrid model combining Poisson predictions with betting odds

- **`simulation/`**: Monte Carlo simulation engine
  - `simulator.py`: Runs Monte Carlo simulations to predict season outcomes
  - Supports both pure statistical and hybrid (odds-enhanced) simulations

- **`analysis/`**: Results aggregation and analysis
  - `aggregator.py`: Calculates position probabilities, championship/relegation odds, expected points

- **`visualization/`**: Dashboard and visualization
  - `dashboard.py`: Streamlit dashboard components

- **`database/`**: Data persistence
  - `db_manager.py`: Database abstraction supporting SQLite and PostgreSQL
  - Tables: matches, team_statistics, model_parameters, simulation_results

- **`utils/`**: Utility functions
  - `odds_converter.py`: Odds format conversion and margin removal
  - `column_standardizer.py`: Standardize column names across datasets
  - `helpers.py`: General helper functions

- **`config/`**: Configuration management
  - `config_loader.py`: Load configuration from YAML files (if they exist)

### Data Flow

1. **Scraping**: `Premier LeagueScraper` fetches match data from stats.swefootball.se
2. **Cleaning**: `DataCleaner` separates completed results from upcoming fixtures
3. **Strength Calculation**: `TeamStrengthCalculator` computes attack/defense ratings
4. **Model Training**: `PoissonModel` learns from historical data
5. **Odds Integration** (optional): `OddsAPI` fetches live odds, `HybridModel` combines with Poisson
6. **Simulation**: `MonteCarloSimulator` runs thousands of season simulations
7. **Analysis**: `ResultsAggregator` computes probabilities and expected outcomes
8. **Visualization**: Streamlit UI displays results and predictions

### Application Structure

`app.py` is the main Streamlit application with these pages:

1. **Data Collection**: Scrape and save match data
2. **Data Verification**: Review and filter collected data
3. **Model Training**: Calculate team strengths and train Poisson models
4. **Odds Integration**: Fetch live odds and create hybrid predictions
5. **Monte Carlo Simulation**: Run season simulations
6. **Fixture Predictions**: View individual match predictions
7. **Results Analysis**: Championship/relegation probabilities, position matrices
8. **Dashboard**: Interactive visualizations
9. **Database Management**: View and manage stored data

## Key Technical Details

### Poisson Model Variants

- **Fast Training**: Basic Poisson regression (faster, good for development)
- **Advanced Training**: Uses Maximum Likelihood Estimation (MLE) and Dixon-Coles correlation adjustment (slower, more accurate)

### Hybrid Model

The hybrid model dynamically weights Poisson predictions vs. betting odds based on season progress:
- Early season: Higher odds weight (70%) due to limited historical data
- Mid season: Balanced weighting (30-50%)
- Late season: Higher Poisson weight (90%) as statistical sample grows

Weight calculation considers:
- Games played
- Odds quality (margin/confidence)
- Season stage

### Column Name Standardization

Historical data may use different column naming conventions. The `ColumnStandardizer` utility normalizes:
- `Home_Team`/`home_team` → `HomeTeam`
- `Away_Team`/`away_team` → `AwayTeam`
- `FTHG`/`home_goals` → `FTHG` (Full Time Home Goals)
- `FTAG`/`away_goals` → `FTAG` (Full Time Away Goals)

### Database Schema

- **matches**: Raw match data (both results and fixtures)
- **team_statistics**: Computed attack/defense ratings for each team
- **model_parameters**: Saved Poisson model parameters
- **simulation_results**: Historical simulation runs
- **analysis_results**: Aggregated analysis outputs

### Configuration

The application looks for configuration in:
- Environment variables: `DATABASE_URL`, `ODDS_API_KEY`
- YAML config files (optional, loaded by `config_loader.py`)
- Defaults: SQLite database at `data/db/premier_league.db`

## Common Development Patterns

### Adding a New Statistical Feature

1. Add feature calculation to `TeamStrengthCalculator` in `shl/data/strength.py`
2. Update `team_stats` DataFrame columns
3. Modify `PoissonModel.fit()` to incorporate the new feature
4. Update database schema in `db_manager.py` if persisting

### Integrating a New Data Source

1. Create new scraper/fetcher in `shl/data/`
2. Define Pydantic schemas in a new or existing schema file
3. Update `DataCleaner` to handle the new data format
4. Add data source option to the UI in relevant page function in `app.py`

### Adding a New Model

1. Create model class in `shl/models/` following the interface pattern (fit, predict methods)
2. Add model selection option in `model_training_page()` in `app.py`
3. Update `MonteCarloSimulator` to support the new model
4. Save/load model parameters in `models/` directory

### Working with Streamlit Session State

The app uses session state for:
- `data_loaded`: Whether data has been successfully loaded
- `model_trained`: Whether model has been trained
- `simulation_complete`: Whether simulation has been run
- `db_manager`: Database connection instance
- `db_connected`: Database connection status
- `poisson_model`: Trained model instance
- `odds_data`: Fetched betting odds (OddsData instance)
- `odds_fetched`: Whether odds have been fetched
- `active_page`: Current navigation page

## File Paths and Conventions

### Data Directories

- `data/raw/`: Raw scraped data (if saved)
- `data/clean/`: Cleaned datasets
  - `results.csv`: Completed matches with scores
  - `fixtures.csv`: Upcoming matches without scores
  - `upcoming_fixtures.csv`: Authentic fixture schedule (preferred for simulations)
- `data/processed/`: Computed statistics
  - `team_stats.csv`: Team strength metrics
- `data/db/`: SQLite database files

### Model Storage

- `models/poisson_params.pkl`: Saved Poisson model parameters

### Reports

- `reports/simulations/sim_results.csv`: Monte Carlo simulation results
- `reports/simulations/fixture_predictions.csv`: Individual match predictions

## Important Notes

### Data Source

The scraper targets `stats.swefootball.se`, which is the official Premier League statistics website. The scraper:
- Identifies regular season matches (excludes playoffs, qualifiers)
- Handles both completed matches and upcoming fixtures
- Parses Swedish date/time formats
- Extracts period-by-period scores when available

### Odds API

Uses The-Odds-API (`api.the-odds-api.com`) for live betting odds. The free tier has request limits, so:
- Cache odds data in session state
- Only fetch when explicitly requested
- Check usage with the "Check API Usage" button before fetching

### Simulation Performance

Monte Carlo simulations can be computationally intensive:
- Use progress callbacks to update UI
- Default: 10,000 simulations (adjustable)
- Results are cached to `reports/simulations/`
- Use fewer simulations for development/testing

### Team Name Consistency

Team names must be consistent across all data sources:
- Scraper extracts official team names from stats.swefootball.se
- Odds API may use different team names (handle via mapping if needed)
- Column standardizer helps normalize column names but not team names themselves
