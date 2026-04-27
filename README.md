# Baseball Matchup Tracker

Real-time MLB matchup analysis dashboard built with Streamlit and Statcast data.

## Running the app

```bash
streamlit run dashboard/app.py
```

## Architecture

Five strict layers — each layer may only import from layers below it.

```
sources/       Layer 1 — network calls only (MLB Stats API, Baseball Savant)
storage/       Layer 2 — Parquet cache + DuckDB query layer
transforms/    Layer 3 — pure functions: DataFrame in, DataFrame out
metrics/       Layer 4 — composite scores (MASH, MISS, Stuff+, Command+, Deception+, Arsenal+)
dashboard/     Layer 5 — Streamlit UI only, no business logic
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full layer contract and data flow.

## Key scores

| Score | What it measures | Range |
|-------|-----------------|-------|
| **MASH** | Batter contact quality vs this pitcher | 0–100 (50 = neutral) |
| **MISS** | Swing-and-miss probability for this matchup | 0–100 (50 = neutral) |
| **Stuff+** | Raw pitch quality vs league average | 100 = avg, >100 = better |
| **Command+** | Pitch location precision vs league average | 100 = avg |
| **Deception+** | Hitter-decision difficulty vs league average | 100 = avg |
| **Arsenal+** | Usage-weighted Stuff+ across pitch mix | 100 = avg |

## Tests

```bash
pytest tests/
```

82 tests across storage, transforms, and metrics layers.

## Project layout

```
sources/           MLB Stats API + Baseball Savant fetchers
storage/           LocalParquetStorage, DuckDB query helper
transforms/        pitchers.py, batters.py, lineups.py
metrics/           mash.py, miss.py, stuff.py, command.py, deception.py, arsenal.py
                   features.py, weighting.py, shrinkage.py
dashboard/
  app.py           Streamlit entry point
  components/      live_charts.py, season_charts.py, matchup_cards.py
mlb_live/          Live game data pipeline (statsapi)
mlb_season/        Season Statcast pipeline (Baseball Savant CSV)
core/              Shared utilities (config, handedness, player lookup, game selector)
tests/             test_storage.py, test_transforms.py, test_metrics.py
archive/           Superseded files kept for reference
```
