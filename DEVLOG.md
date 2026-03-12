# DEVLOG

## 2026-03-12 - MVP 4-feature spatial split patch (03_model_training.ipynb)

### The Change
- Updated [03_model_training.ipynb](d:/projects/water-quality-prediction/notebooks/03_model_training.ipynb) to run from merged MVP parquet inputs instead of `master_train/master_test`.
- Restricted modeling to the 4 baseline features only: `swir22`, `NDMI`, `MNDWI`, `pet`.
- Replaced Region/LORO evaluation with spatial grouping:
  - KMeans-based `spatial_group` labels from latitude/longitude.
  - GroupKFold OOF evaluation across spatial groups.
  - Added pseudo-validation southeast holdout scoring (`holdout_r2`) derived from validation bbox overlap.
- Updated scout/full selection and manifest logic to use `holdout_r2` in scoring and logging.
- Updated diagnostic and validation-loading cells to match MVP merged files:
  - `../data/interim/water_quality_mvp_baseline.parquet`
  - `../data/interim/water_quality_mvp_validation.parquet`

### The Reasoning
- Random splits are optimistic under geographic shift. The new split strategy is meant to approximate unseen-area behavior while staying computationally simple and reproducible.
- Keeping only the 4 core features provides a controlled baseline while waiting for richer master data.
- `holdout_r2` adds a deployment-shaped signal during model selection rather than relying on global OOF alone.

### The Tech Debt
- The notebook still contains historical outputs/markdown from the prior Region-based version and should be re-executed end-to-end to refresh outputs.
- Pseudo-validation holdout currently uses a bbox+fallback heuristic; this should be replaced by a fixed, explicit spatial polygon/block definition.
- We still rely on notebook-centric orchestration; extracting this split/eval logic into reusable `src/` modules would reduce drift.

## 2026-03-12 - Group-level pseudo-holdout hardening (03_model_training.ipynb)

### The Change
- Refined [03_model_training.ipynb](d:/projects/water-quality-prediction/notebooks/03_model_training.ipynb) split strategy from row-level pseudo holdout to strict group-level pseudo holdout.
- Increased spatial partition granularity from `SPATIAL_N_CLUSTERS=10` to `16`.
- Added holdout controls:
  - `HOLDOUT_MIN_GROUPS = 3`
  - `HOLDOUT_MIN_FRAC = 0.08`
  - `HOLDOUT_MAX_FRAC = 0.15`
- Replaced mask heuristic with `select_pseudo_holdout_groups(...)` that selects entire spatial groups nearest validation footprint (bbox distance + center distance ranking).
- Updated grouped evaluation to enforce leakage guard:
  - Raises an error if train and holdout intersect on `spatial_group` during holdout scoring.

### The Reasoning
- Row-level pseudo holdout allowed mixed membership within the same spatial clusters and could still be optimistic under geographic shift.
- Whole-group exclusion better reflects unseen geography and yields a more defensible `holdout_r2` for model ranking.
- Increasing the number of clusters gives finer spatial control so holdout can cover more than one coarse cluster while keeping target fraction bounds.

### The Tech Debt
- Holdout group selection still uses centroid proximity heuristics; this should be replaced by a fixed spatial polygon/zone definition once deployment geography is finalized.
- Cluster count (`16`) is static and may need retuning if training distribution changes with incoming master data.
- Notebook outputs must be re-run to refresh stale printed results and avoid confusion.

## 2026-03-12 - Added benchmark-derived 03c notebook

### The Change
- Added [03c_model_training_benchmark_derived.ipynb](d:/projects/water-quality-prediction/notebooks/03c_model_training_benchmark_derived.ipynb).
- Notebook is derived from the benchmark approach and intentionally keeps a simple baseline flow:
  - 4 features only: `swir22`, `NDMI`, `MNDWI`, `pet`
  - random 70/30 split
  - `RandomForestRegressor` per target
  - submission generation from merged MVP parquet files

### The Reasoning
- Current strict spatial pipeline in `03_model_training.ipynb` is useful for robustness diagnostics but underperforming leaderboard outcomes.
- A clean benchmark-aligned anchor notebook gives a stable baseline to iterate from while waiting for new teammate data.

### The Tech Debt
- Random split is optimistic under geographic shift; this notebook is intentionally a baseline, not a final robustness protocol.
- No MLflow tracking or grouped CV in this notebook by design; if needed later, add progressively after baseline validation.

