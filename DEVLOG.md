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

## 2026-03-12 - 03c benchmark parity patch + master data sanity audit

### The Change
- Updated [03c_model_training_benchmark_derived.ipynb](d:/projects/water-quality-prediction/notebooks/03c_model_training_benchmark_derived.ipynb) to match benchmark inference behavior more closely:
  - Added `val_medians` and changed validation imputation from training medians to validation medians.
  - Un-commented submission generation and CSV save cells.
  - Added a template/validation row-order integrity guard (Longitude, Latitude, Sample Date) before writing submission.
- Ran a no-feature-change sanity audit comparing `water_quality_mvp_baseline` and `master_train` behavior:
  - Confirmed `master_train.parquet` does not contain `Latitude/Longitude/Sample Date`.
  - Confirmed large metric drop is primarily split-strategy sensitivity (random split vs grouped geographic split), not target corruption.

### The Reasoning
- Prior `03c` had one non-benchmark-equivalent behavior (validation imputation source) that could alter leaderboard predictions on null rows.
- Enabling submission cells plus explicit row-order checks reduces the chance of silent bad uploads.
- The master-vs-MVP audit was needed to separate merge-quality concerns from expected out-of-region generalization drop under stricter splits.

### The Tech Debt
- `03c` notebook outputs are now stale relative to the new code cells and should be re-run before sharing metrics.
- Integrity checks are notebook-local; this should eventually move into reusable helper code under `src/`.
- `master_train` lacks geo key columns, so it cannot support the same spatial split protocol as MVP without a separate key-join artifact.

## 2026-03-12 - 03c hyperparameter sweep and guarded multi-shot export

### The Change
- Refactored [03c_model_training_benchmark_derived.ipynb](d:/projects/water-quality-prediction/notebooks/03c_model_training_benchmark_derived.ipynb) from single fixed RF training into a controlled hyperparameter workflow while keeping the same 4 features (`swir22`, `NDMI`, `MNDWI`, `pet`).
- Added candidate sweep across `RandomForestRegressor` and `ExtraTreesRegressor` configurations.
- Added multi-seed scoring (`SEED_SWEEP = [42, 52, 62, 72]`) and robust ranking (`Robust_Score = mean_r2 - 0.25 * std_r2`).
- Added non-regression guard per target:
  - baseline candidate = `rf_baseline`
  - tuned candidate is selected only if it beats baseline by `IMPROVEMENT_MARGIN = 0.002` on mean test R2.
- Added full-data fit for both:
  - anchor baseline models
  - selected tuned models
- Added submission construction for 3 files:
  - `submission_*_03c_anchor_baseline.csv`
  - `submission_*_03c_tuned.csv`
  - `submission_*_03c_hedge.csv` (35% anchor + 65% tuned)
- Kept row-order integrity checks against template before prediction export.

### The Reasoning
- Prior 03c was a single-point baseline, which made it easy to get stuck around the same local metric.
- Multi-seed tuning reduces overreacting to one lucky/unlucky split.
- The baseline fallback guard is a pragmatic safety rail to avoid local regression while still letting tuned candidates win when improvement is consistent.
- Three-shot export supports practical leaderboard strategy (safe anchor, aggressive tuned, blended hedge) without changing features.

### The Tech Debt
- The sweep space is intentionally small for runtime; wider search (or Bayesian tuning) is deferred.
- Evaluation is still random-split based and not geography-robust; this notebook remains leaderboard-oriented, not a final robustness protocol.
- Notebook execution outputs are now stale and must be regenerated by running cells end-to-end.

## 2026-03-12 - 03c runtime optimization (parallel sweep + tqdm)

### The Change
- Updated [03c_model_training_benchmark_derived.ipynb](d:/projects/water-quality-prediction/notebooks/03c_model_training_benchmark_derived.ipynb) to improve tuning runtime visibility and throughput:
  - Added `tqdm` progress bars for target-level and candidate-level sweep tracking.
  - Added `joblib.Parallel` candidate-level parallel scoring.
  - Added `PARALLEL_JOBS` control (`max(1, min(n_candidates, cpu_count//2))`).
  - Switched per-estimator `n_jobs` from `-1` to `1` to avoid nested parallel oversubscription when running outer parallel jobs.
  - Added helper `evaluate_candidate_row(...)` for clean parallel execution units.

### The Reasoning
- The sweep now runs many model fits; serial execution is slow and gives poor feedback during waiting.
- Outer-loop parallelism gives a safer speedup than nested estimator parallelism for this notebook structure.
- `tqdm` gives immediate progress insight so runtime is predictable while iterating.

### The Tech Debt
- Parallel speedup depends on local CPU/core limits and process overhead; no adaptive runtime benchmark is stored yet.
- We still use notebook-local orchestration; moving sweep/eval code into `src/` would improve reuse and testability.

## 2026-03-12 - 03c confirmation patch: tqdm + parallel sweep active in source cells

### The Change
- Re-checked and patched [03c_model_training_benchmark_derived.ipynb](d:/projects/water-quality-prediction/notebooks/03c_model_training_benchmark_derived.ipynb) to ensure both requirements are active in executable source code:
  - Added imports: `from tqdm.auto import tqdm` and `from joblib import Parallel, delayed`.
  - Added `evaluate_candidate_row(...)` helper for candidate-level parallel scoring.
  - Added `PARALLEL_JOBS` and `Parallel(...)(delayed(...))` execution in the hyperparameter loop.
  - Added `tqdm` progress bars at target and candidate sweep levels.
  - Enforced `candidate['params']['n_jobs'] = 1` before outer parallel execution to avoid nested oversubscription.

### The Reasoning
- Prior notebook state still reflected non-parallel source execution in active cells.
- This patch guarantees that progress tracking and parallel scoring are both present where the notebook actually runs.

### The Tech Debt
- Notebook outputs still include historical traces from previous runs and should be refreshed by re-running end-to-end.

## 2026-03-12 - 03c model-family expansion while preserving RF baseline anchor

### The Change
- Updated [03c_model_training_benchmark_derived.ipynb](d:/projects/water-quality-prediction/notebooks/03c_model_training_benchmark_derived.ipynb) candidate pool from mostly RF/ET variants to a more diverse set:
  - Added `HistGradientBoostingRegressor` candidates.
  - Added regularized linear candidates (`Ridge`, `ElasticNet`) with log-target transformation.
  - Added log-target RF/ET variants via `TransformedTargetRegressor`.
- Kept RF baseline governance unchanged:
  - `BASELINE_ID = 'rf_baseline'`
  - all model comparisons still measured against baseline
  - fallback remains active when candidates fail to exceed baseline by `IMPROVEMENT_MARGIN`.

### The Reasoning
- Prior sweep stayed in one narrow model family and repeatedly selected baseline.
- Diversity in inductive bias is needed to test whether non-tree or transformed-target models can beat baseline on local validation without adding new features.
- Baseline-anchored fallback is preserved to avoid accidental regression.

### The Tech Debt
- Candidate list growth increases runtime; if needed, reduce candidate count per target after first comparative run.
- Legacy output cells may still show old run traces until notebook is re-executed end-to-end.

## 2026-03-12 - 03c hotfix for HGB parallel parameter compatibility

### The Change
- Fixed [03c_model_training_benchmark_derived.ipynb](d:/projects/water-quality-prediction/notebooks/03c_model_training_benchmark_derived.ipynb) parallelization guard so `n_jobs=1` is applied only to `rf` and `et` candidates.
- Removed unintended `n_jobs` injection into non-forest models (notably `HistGradientBoostingRegressor`), which caused runtime failure.

### The Reasoning
- `HistGradientBoostingRegressor` does not accept an `n_jobs` argument.
- The previous blanket assignment to all candidate param dicts created an invalid constructor argument path during parallel evaluation.

### The Tech Debt
- Candidate parameter sanitization is still implicit; adding a model-kind-specific param validator would make this safer.

## 2026-03-12 - Re-applied 03c HGB `n_jobs` hotfix after unsaved local state

### The Change
- Re-applied the `03c` tuning-cell guard in [03c_model_training_benchmark_derived.ipynb](d:/projects/water-quality-prediction/notebooks/03c_model_training_benchmark_derived.ipynb):
  - from blanket `candidate['params']['n_jobs'] = 1`
  - to conditional application only for `rf` and `et` candidate kinds.

### The Reasoning
- The notebook file in workspace still had the old blanket assignment (likely from an unsaved/reverted local state), which breaks `HistGradientBoostingRegressor`.
- Re-applying guarantees the candidate pool with `hgb` can run.

### The Tech Debt
- This repeated regression indicates notebook state drift risk; extracting sweep config to a Python module would reduce accidental loss.

## 2026-03-12 - 03c switched to baseline-only DRP hedge workflow

### The Change
- Updated [03c_model_training_benchmark_derived.ipynb](d:/projects/water-quality-prediction/notebooks/03c_model_training_benchmark_derived.ipynb) to disable hyperparameter sweep execution and use fixed `rf_baseline` training for all targets.
- Replaced candidate search loop with baseline-only repeated split diagnostics (`SEED_SWEEP = [42, 52, 62, 72]`) and full-data fit.
- Replaced submission variants with DRP hedge strategy while keeping TA/EC from baseline:
  - `A`: anchor baseline (`DRP` raw baseline prediction)
  - `B`: DRP shrink (`alpha=0.50`) toward training median with `[0, q99.5]` clipping
  - `C`: stronger DRP shrink (`alpha=0.35`) with same clipping
- Updated save cell output names to explicit `A/B/C` DRP hedge filenames.

### The Reasoning
- Hyperparameter expansion was not beating baseline and added instability/runtime.
- DRP was the main leaderboard failure mode, so target-specific post-processing is a safer short-term lever than broader model complexity changes.
- Keeping TA/EC untouched preserves what is already working while probing DRP risk-reduction variants.

### The Tech Debt
- Helper functions/imports from the prior sweep remain in notebook context and can be cleaned in a later pass.
- DRP hedge alphas are fixed heuristics (`0.50`, `0.35`); they should be tuned against a pseudo-spatial validation target once a stable proxy is finalized.

