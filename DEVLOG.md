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

## 2026-03-12 - 03_model_training switched to RF-only spatial sweep

### The Change
- Updated [03_model_training.ipynb](d:/projects/water-quality-prediction/notebooks/03_model_training.ipynb) model configuration to RF-only mode while preserving the existing spatial GroupKFold + pseudo-holdout workflow.
- Replaced mixed model bank (Ridge/Lasso/Elastic/XGB/RF) with:
  - `RF_n600_raw`
  - `RF_n600_Log`
- Updated target sweep so all targets (`Total Alkalinity`, `Electrical Conductance`, `Dissolved Reactive Phosphorus`) evaluate only those two RF variants.
- Updated `DRP_SAFE_MODELS` to RF-only list for manifest selection consistency.

### The Reasoning
- Current goal is to remove model-family variance and establish a stable spatial baseline before rebuilding complexity.
- Keeping only RF variants makes score movement easier to attribute to split strategy and post-processing rather than architecture churn.

### The Tech Debt
- Unused imports and legacy model-construction paths remain in notebook context and should be cleaned in a future hygiene pass.
- RF-only search space is intentionally narrow and may cap upside until new features arrive.

## 2026-03-12 - Cleared stale notebook outputs after RF-only patch

### The Change
- Cleared `execution_count` and `outputs` for all code cells in [03_model_training.ipynb](d:/projects/water-quality-prediction/notebooks/03_model_training.ipynb) so cached historical results no longer show pre-patch mixed-model artifacts.

### The Reasoning
- Old cached outputs (XGB/Ridge/Lasso traces) made it appear that RF-only changes were not applied.
- Clearing outputs ensures the visible notebook state matches current source logic and avoids confusion during reruns.

### The Tech Debt
- Notebook output history is now removed; rerun is required to regenerate diagnostics and tables.

## 2026-03-12 - Added dummy-baseline diagnostics to spatial RF evaluation

### The Change
- Updated [03_model_training.ipynb](d:/projects/water-quality-prediction/notebooks/03_model_training.ipynb) grouped spatial evaluation pipeline to compute a dummy baseline (target median predictor) on the exact same folds and pseudo-holdout split as the trained model.
- Extended `grouped_oof_eval(...)` outputs with:
  - `dummy_r2`, `dummy_rmse`, `dummy_mae`
  - `dummy_mean_fold_r2`, `dummy_min_fold_r2`, `dummy_holdout_r2`
  - deltas vs dummy: `delta_r2_vs_dummy`, `delta_min_fold_r2_vs_dummy`, `delta_holdout_r2_vs_dummy`
- Updated scout/full loops to:
  - log dummy and delta metrics to MLflow
  - include dummy/delta columns in `scout_df` and `full_df`
  - print model-vs-dummy comparisons inline for quick gating.
- Fixed generated scout/full print formatting issues and validated notebook code-cell syntax.

### The Reasoning
- Global/spatial metrics alone were hard to trust without a no-skill reference.
- Dummy baseline on identical splits provides a concrete floor; RF should consistently beat it before any further complexity is justified.

### The Tech Debt
- Current selection score still ranks by model metrics only; an explicit gate on `delta_r2_vs_dummy` / `delta_holdout_r2_vs_dummy` can be added next.

## 2026-03-12 - Added hard dummy-gate manifests and DRP fallback hedge for submission shots

### The Change
- Updated [03_model_training.ipynb](d:/projects/water-quality-prediction/notebooks/03_model_training.ipynb) finalist selection to use a hard gate:
  - candidates with `delta_holdout_r2_vs_dummy > 0` are preferred
  - if none pass, fallback to the best available row for that target
- Added `pick_with_gate(...)` and rebuilt `manifest_A` / `manifest_B` with target-wise RF preferences:
  - TA prefers `RF_n600_raw`
  - EC prefers `RF_n600_Log` for A, open best gated choice for B
  - DRP prefers `RF_n600_Log` but is still gate-controlled
- Added `DRP_DELTA_HOLDOUT` and `DRP_USE_MODEL` flags and wired them into submission generation:
  - when DRP fails gate, Shot A/B/C use train median for DRP
  - when DRP passes gate, Shot A uses model DRP, Shot B/C use model-to-median hedge blends
- Updated feature-availability checks so DRP model features are required only when `DRP_USE_MODEL=True`.

### The Reasoning
- DRP was the dominant failure mode and could destroy a full submission even when TA/EC were acceptable.
- Gating against dummy on pseudo-holdout gives a practical safety boundary for whether DRP model signal is trustworthy.
- Explicit fallback keeps the pipeline deterministic and avoids forcing weak DRP predictions into every submission variant.

### The Tech Debt
- Gate threshold is currently binary at `0`; it may still be noisy with limited pseudo-holdout representativeness.
- TA/EC/DRP target preferences are hardcoded and should eventually be parameterized for faster experimentation.
- The notebook still carries legacy cells/markdown from prior model families that can be pruned in a cleanup pass.

## 2026-03-12 - Optimized wide-feature EDA flow for 169-column interim data

### The Change
- Refactored [01_eda_and_discovery.ipynb](d:/projects/water-quality-prediction/notebooks/01_eda_and_discovery.ipynb) to run efficiently on wide tables (`~169` columns) by introducing EDA guardrails and prioritized plotting.
- Added notebook-level EDA controls in the setup cell:
  - sampling cap (`EDA_SAMPLE_ROWS`)
  - max plot column caps (`MAX_NUM_PLOTS`, `MAX_CAT_PLOTS`, `MAX_SCATTER_FEATURES_PER_TARGET`, `MAX_HEATMAP_FEATURES`)
  - compact preview cap (`MAX_PREVIEW_COLS`)
  - helper `sample_rows(...)`.
- Reworked heavy visualization cells to avoid plotting all columns:
  - Bird's-eye section now shows first+random preview columns and compact metadata instead of full chunk-by-chunk rendering.
  - Numeric and categorical distribution plots now run on prioritized subsets.
  - Correlation heatmap now uses selected candidate features (Spearman) rather than full-matrix plotting.
  - Missingness visualization now uses top-missing columns + sampled rows.
  - Feature-vs-target section now uses selected features per target with hexbin plots.
  - Outlier detection switched to vectorized IQR summary and top-feature charting.
  - VIF/multicollinearity now runs on a bounded candidate subset for tractable runtime.
- Added a new section/cell: `Train-Test Drift Check (Optimized)` using KS statistics on prioritized numeric columns.
- Validated notebook JSON and code-cell syntax after edits.

### The Reasoning
- Full-feature plotting on 169 columns creates noisy output and slow execution with low diagnostic value.
- Prioritization (missingness, variance, target correlation) preserves signal while cutting runtime and visual clutter.
- Adding lightweight drift diagnostics gives a direct sanity check before modeling without requiring full CV runs.

### The Tech Debt
- Prioritization heuristics are fixed and notebook-local; moving them to reusable utility functions under `src/` would reduce drift.
- Target correlation screening currently uses sampled Spearman correlation; robustness can be improved with repeated-seed summaries if needed.
- Geography and LOSO sections remain dependent on geo/date columns and will still be skipped/fail if those fields are absent in future interim snapshots.

## 2026-03-12 - Added SANLC-specific schema and coverage diagnostics in EDA

### The Change
- Updated [01_eda_and_discovery.ipynb](d:/projects/water-quality-prediction/notebooks/01_eda_and_discovery.ipynb) with a new section:
  - `### SANLC Coverage And Schema Diagnostics`
- Added a dedicated SANLC code cell (after validation inspection, before train-vs-valid comparison) that:
  - detects SANLC columns by prefix (`sanlc2020_pct_`, `sanlc2022_pct_`)
  - compares train SANLC schema vs validation SANLC schema
  - reports train-only and valid-only SANLC columns
  - constructs `df_validation_sanlc_aligned` by adding missing train SANLC columns to validation with `0.0`
  - summarizes SANLC per-column behavior (`non_null_pct`, `zero_pct`, `mean`, `p95`, `max`) for train and aligned validation
  - computes row-wise SANLC percentage-sum summaries for 2020 and 2022 blocks
  - prints a concrete feature-contract checklist for train/test alignment.
- Verified notebook code-cell syntax after insertion.

### The Reasoning
- Current modeling issues include train/validation feature-count mismatch likely driven by absent SANLC classes in validation geography.
- A SANLC-only diagnostic makes this mismatch explicit and provides a reproducible zero-fill alignment path before model training.
- Row-sum summaries provide a quick sanity check that aligned coverage behaves plausibly after column harmonization.

### The Tech Debt
- SANLC prefix matching is rule-based; if naming conventions change, this detection cell must be updated.
- Alignment is currently notebook-local (`df_validation_sanlc_aligned`); production training/inference should move this contract into shared preprocessing utilities.

## 2026-03-12 - Implemented feature-contract handoff (01 -> 02) and schema-aligned preprocessing

### The Change
- Updated [01_eda_and_discovery.ipynb](d:/projects/water-quality-prediction/notebooks/01_eda_and_discovery.ipynb) with a new export section:
  - `### Export Feature Contract For Preprocessing`
  - writes `../data/interim/feature_contract_master_iteration.txt` (feature columns, one per line)
  - writes `../data/interim/feature_contract_master_iteration_meta.json` (dtype map, targets, SANLC count, source metadata).
- Rewrote [02_preprocessing.ipynb](d:/projects/water-quality-prediction/notebooks/02_preprocessing.ipynb) into a contract-driven alignment pipeline:
  - loads train/test iteration parquet files
  - loads contract from TXT (or derives fallback from train)
  - enforces target-leakage guard (targets cannot appear in feature contract)
  - aligns test to train contract:
    - adds missing contract columns
    - fills missing SANLC columns with `0.0`
    - fills other missing contract columns with `NaN`
    - drops extra non-target test columns
    - reorders columns to exact contract order
    - attempts dtype harmonization to train dtypes
  - runs pre-model checks:
    - schema parity
    - missing-after-alignment summary
    - SANLC row-sum sanity summary
    - geo/time-key presence check for pseudo-spatial readiness
  - saves aligned artifacts:
    - `../data/interim/master_train_iteration_aligned.parquet`
    - `../data/interim/master_test_iteration_aligned.parquet`
    - `../data/interim/feature_alignment_report_master_iteration.csv`.

### The Reasoning
- You requested a direct handoff from EDA insights into preprocessing before modeling.
- Contract export in `01` creates a stable schema artifact you can reuse and paste/reference in `02`.
- Contract-driven alignment in `02` operationalizes the “best plan before 03”:
  - freeze schema
  - align test to train contract
  - run leakage/sanity gates
  - produce reproducible aligned datasets.

### The Tech Debt
- Non-SANLC missing test columns are currently `NaN`-filled by default; target-aware imputation policy should be added next if those columns exist.
- Contract currently includes all non-target train columns; if you want strict model feature subsets, add a second “model_contract” artifact.
- Pseudo-spatial split itself is not executed in `02`; this notebook only verifies geo/time key readiness for that next stage.

## 2026-03-12 - 03_model_training switched to aligned iteration data sources with contract guards

### The Change
- Updated [03_model_training.ipynb](d:/projects/water-quality-prediction/notebooks/03_model_training.ipynb) data loading to prioritize aligned iteration artifacts produced by preprocessing:
  - train candidates:
    - `../data/interim/master_train_iteration_aligned.parquet`
    - `../data/interim/master_train_iteration.parquet`
    - fallback `../data/interim/water_quality_mvp_baseline.parquet`
  - validation candidates:
    - `../data/interim/master_test_iteration_aligned.parquet`
    - `../data/interim/master_test_iteration.parquet`
    - fallback `../data/interim/water_quality_mvp_validation.parquet`
- Added path resolver helper (`pick_first_existing`) and explicit prints for selected train/validation inputs.
- Added optional feature-contract schema check via:
  - `../data/interim/feature_contract_master_iteration.txt`
  - verifies every contract column exists in both train and validation before training starts.
- Updated submission inference cell to read validation from `VALID_PATH` (same resolved source as split setup), removing hardcoded MVP validation path dependency.
- Updated the diagnostics summary cell to derive its feature list from active `FEATURE_SETS['C']` instead of a hardcoded 4-feature list.

### The Reasoning
- You requested proceeding with the current 03 workflow while making minimal adjustments for new data readiness.
- This keeps model logic stable (RF + gating + DRP hedge) while enforcing schema consistency with the new contract-aligned preprocessing outputs.
- Using the same resolved validation source across split setup and submission generation avoids accidental source mismatch.

### The Tech Debt
- 03 still uses a single feature-set recipe (`C`) by default; richer aligned features are available but intentionally not expanded in this patch to avoid uncontrolled variance.
- Contract guard is optional (only active when TXT exists); making it mandatory could be a next hardening step once pipeline order is fixed.

## 2026-03-12 - Added tqdm progress tracking to 03_model_training execution loops

### The Change
- Updated [03_model_training.ipynb](d:/projects/water-quality-prediction/notebooks/03_model_training.ipynb) to include `tqdm` progress indicators:
  - Added import: `from tqdm.auto import tqdm`
  - Added fold-level progress in `grouped_oof_eval(...)` with optional controls:
    - new args: `show_fold_progress=False`, `fold_desc=None`
    - GroupKFold loop now runs through `tqdm(..., total=n_splits, unit='fold')`
  - Added scout-level progress bars:
    - target loop: `tqdm(TARGET_COLS, desc='Scout targets')`
    - per-target model loop: `tqdm(recipes, desc=f'Scout {target}')`
  - Added full-stage progress bar:
    - finalist loop: `tqdm(finalist_df.iterrows(), total=len(finalist_df), desc='Full finalists')`
  - Enabled fold progress in scout/full `grouped_oof_eval(...)` calls via:
    - `show_fold_progress=True`
    - informative `fold_desc` labels.

### The Reasoning
- Training/evaluation runtime is multi-minute and users need real-time visibility into stage and fold progression.
- Progress bars were added without changing model-selection logic or scoring behavior.

### The Tech Debt
- Nested progress bars can appear busy in some notebook frontends; if output becomes noisy, fold-level tqdm can be toggled off via `show_fold_progress=False`.

## 2026-03-12 - Marked non-essential 03_model_training cells as optional for lean runs

### The Change
- Updated [03_model_training.ipynb](d:/projects/water-quality-prediction/notebooks/03_model_training.ipynb) to keep core training path intact while disabling non-essential diagnostics by default:
  - cell `9`: `df.info()` schema print (commented)
  - cell `25`: legacy constants (`TARGET_GLOBAL_R2_FLOOR`, `DRP_SAFE_MODELS`) currently unused by active manifest logic (commented)
  - cell `28`: feature/target describe diagnostic (commented)
  - cell `34`: post-submission diagnostic summary tables (commented)
- Added a `Lean Run Guide` note in the top markdown cell indicating these optional cells are pre-commented for faster execution.

### The Reasoning
- You asked to comment cells you do not need to run.
- This reduces notebook noise/runtime overhead while preserving the active RF + gating + submission workflow.
- It also makes clear that `FEATURE_SETS` is still actively used in scout/full loops and should not be removed.

### The Tech Debt
- Some top markdown text still references older region-based wording from earlier notebook lineage; wording cleanup can be done later for consistency with current spatial-group implementation.

## 2026-03-12 - Restored spatial_group materialization in 03_model_training split setup

### The Change
- Fixed [03_model_training.ipynb](d:/projects/water-quality-prediction/notebooks/03_model_training.ipynb) `spatial_group` KeyError path by restoring the execution block in the data-loading/split cell that:
  - applies `add_spatial_groups(...)` to `df`
  - selects pseudo-holdout groups via `select_pseudo_holdout_groups(...)`
  - creates `df['is_pseudo_valid']`
  - prints split summary and validates train/holdout group disjointness.
- Added an explicit guard in `grouped_oof_eval(...)`:
  - if `spatial_group` is missing, raise a clear runtime message instructing to run cell 8 first.

### The Reasoning
- The notebook still had helper function definitions for spatial grouping, but the materialization step was absent, causing `KeyError: 'spatial_group'` in scout/full evaluation.
- Restoring the split setup block and adding a targeted guard prevents silent failure and improves execution-order clarity.

### The Tech Debt
- This remains notebook order-dependent; extracting split setup into a dedicated reusable function/module would reduce recurrence risk.

## 2026-03-13 - Switched 03_model_training primary feature set from benchmark-4 to full numeric contract

### The Change
- Updated [03_model_training.ipynb](d:/projects/water-quality-prediction/notebooks/03_model_training.ipynb) feature-set configuration:
  - Kept legacy benchmark set `C` (`swir22`, `NDMI`, `MNDWI`, `pet`) as optional fallback.
  - Added `FULL_NUMERIC` feature set as primary, built from contract/data columns with exclusions:
    - exclude targets
    - exclude split/meta runtime columns (`spatial_group`, `is_pseudo_valid`)
    - keep only numeric dtypes to match current preprocessor.
  - Added `PRIMARY_FEATURE_SET = 'FULL_NUMERIC'`.
- Rewired `TARGET_SWEEP` to use `PRIMARY_FEATURE_SET` for all targets and both RF variants (`RF_n600_raw`, `RF_n600_Log`).
- Updated optional diagnostics fallback expression to reference `PRIMARY_FEATURE_SET` when uncommented.

### The Reasoning
- Using only 4 features defeated the purpose of loading expanded aligned datasets.
- This change allows immediate training with the wider feature space while preserving the existing RF + gating workflow.
- Numeric-only filtering avoids breakage because current preprocessing pipeline is numeric-only (`median imputer + scaler`).

### The Tech Debt
- Non-numeric contract columns are currently excluded; if full 166-column mixed-type training is required, preprocessing must be expanded to handle categorical/datetime features safely (e.g., explicit date transforms and one-hot encoding).

## 2026-03-13 - Added SANLC-specific imputation policy in 03_model_training preprocessor

### The Change
- Updated [03_model_training.ipynb](d:/projects/water-quality-prediction/notebooks/03_model_training.ipynb) `get_preprocessor(features_used)` to use targeted imputation rules:
  - SANLC percentage columns (`sanlc2020_pct_*`, `sanlc2022_pct_*`): `SimpleImputer(strategy='constant', fill_value=0.0)`
  - all other numeric selected features: `SimpleImputer(strategy='median')`
- Kept `StandardScaler()` in both branches and returned a combined `ColumnTransformer` with separate transformers for SANLC vs non-SANLC numeric blocks.
- Added guard for empty feature input to fail fast.

### The Reasoning
- In your pipeline, missing SANLC class columns/values represent absent land-cover share, where `0.0` is semantically correct.
- Median imputation remains safer for non-SANLC continuous variables.
- This operationalizes the SANLC assumptions discovered in EDA/contract alignment before model training.

### The Tech Debt
- Scaling is retained for consistency, though RF models do not require it; this could be simplified later for minor runtime gains.
- Rule-based SANLC detection depends on naming convention prefixes and should be centralized if schema naming evolves.

## 2026-03-13 - Added submission-stage guardrails in 03_model_training (Freeze Manifest A onward)

### The Change
- Updated [03_model_training.ipynb](d:/projects/water-quality-prediction/notebooks/03_model_training.ipynb) to harden post-training cells against common execution-order failures:
  - Restored active no-op `engineer_features(...)` definition and invocation in the feature-engineering cell so submission build can safely call `engineer_features(df_val)`.
  - Added defensive checks in manifest freeze cell:
    - explicit error if a target has no `full_df` rows
    - explicit error if gated candidate pool ends up empty.
  - Added artifact and feature validation in `predict_from_manifest_entry(...)`:
    - fail fast if `preproc_path` / `model_path` files are missing
    - fail fast with clear message if required validation features are missing.
  - Added pre-check in submission cell for validation/template row mismatch before prediction assignment.
- Re-ran notebook code-cell syntax validation after the patch.

### The Reasoning
- Recent runtime failures were caused by missing helper definitions and implicit assumptions about prior cell execution.
- These guards make failures deterministic and informative, reducing debug time under tight submission windows.

### The Tech Debt
- Notebook execution is still stateful; converting the freeze/submission path into idempotent functions or a script entrypoint would further reduce order-dependent breakage.

## 2026-03-13 - Added high-ROI engineered features in 03_model_training

### The Change
- Replaced no-op feature engineering in [03_model_training.ipynb](d:/projects/water-quality-prediction/notebooks/03_model_training.ipynb) with a concrete `engineer_features(...)` implementation and applied it to train data before feature-set construction.
- Implemented the requested high-ROI feature blocks:
  1. **Temporal cyclics** from `Sample_Date` / `Sample Date`:
     - `month_sin`, `month_cos`, `doy_sin`, `doy_cos`, `is_wet_season`
  2. **SANLC year deltas** for matched class pairs:
     - `sanlc_delta_<class> = sanlc2022_pct_<class> - sanlc2020_pct_<class>`
     - `sanlc_abs_delta_<class> = abs(delta)`
  3. **SANLC thematic aggregates** by year + delta:
     - theme shares for `urban`, `agri`, `wetland`, `water`, `mining`, `natural_veg`, `bare`
     - total SANLC share per year
     - per-theme and total deltas (`2022 - 2020`)
- Added summary printout of engineered-column count and sample names after train feature engineering is applied.

### The Reasoning
- With limited time and submission budget, feature signal expansion is higher expected ROI than broad model churn.
- SANLC 2020/2022 are especially suitable for change-based features, which can capture dynamics missed by static shares.
- Temporal cyclics introduce seasonal structure without introducing high-leakage transforms.

### The Tech Debt
- Theme assignment is keyword-based and heuristic; a maintained class-to-theme mapping table would be more robust.
- This introduces many additional columns; if runtime balloons, we may need quick pruning based on feature importance or stability gates.

## 2026-03-13 - Fixed engineered-feature inclusion in PRIMARY_FEATURE_SET

### The Change
- Updated [03_model_training.ipynb](d:/projects/water-quality-prediction/notebooks/03_model_training.ipynb) feature-set assembly (cell 13) to remove the contract-only bottleneck:
  - `FULL_NUMERIC` now builds from the union of:
    - contract-backed base columns (when available), and
    - all current dataframe columns after feature engineering (`df`), excluding reserved runtime/target fields.
- Kept numeric-only filtering for compatibility with the current numeric preprocessor.
- Added explicit diagnostic prints:
  - count of engineered columns included in `FULL_NUMERIC`
  - sample list of included engineered columns.

### The Reasoning
- Engineered columns were created in `engineer_features(...)` but could be excluded when `contract_cols` existed, causing a train-time feature disconnect.
- This patch ensures engineered features actually participate in model training without breaking schema safety.

### The Tech Debt
- The dataframe-wide union may include additional numeric columns beyond intended scope; if needed, we can tighten with an explicit allowlist in a follow-up patch.

## 2026-03-13 - Added fast FE ablation toggle (disable per-class SANLC deltas by default)

### The Change
- Updated [03_model_training.ipynb](d:/projects/water-quality-prediction/notebooks/03_model_training.ipynb) feature-engineering cell to support quick ablation:
  - added `FE_INCLUDE_CLASS_DELTAS = False` default
  - per-class SANLC delta generation (`sanlc_delta_*`, `sanlc_abs_delta_*`) now runs only when toggle is `True`
  - keeps temporal cyclic features and SANLC thematic aggregate features active.
- Added cleanup logic in `engineer_features(...)` to drop previously generated engineered columns when rerunning the cell in the same kernel, preventing accumulation/stale feature carry-over.
- Added explicit printout of FE toggle state for run traceability.

### The Reasoning
- The previous FE batch showed mixed performance and runtime increase; disabling high-cardinality per-class deltas is a fast way to reduce noise and collinearity while preserving the highest-ROI aggregate signals.
- Cleanup on rerun avoids hidden state artifacts during rapid notebook iteration.

### The Tech Debt
- Theme-based aggregate engineering remains heuristic; future refinement may need target-specific feature selection to avoid over-expansion.

## 2026-03-13 - Added ET/HGB challengers to 03_model_training with RF-anchored manifest policy

### The Change
- Updated [03_model_training.ipynb](d:/projects/water-quality-prediction/notebooks/03_model_training.ipynb) model bank (cell 17) from RF-only to a controlled comparison set:
  - `RF_n600_raw`, `RF_n600_Log` (anchor)
  - `ET_n700_raw`, `ET_n700_Log` (ExtraTrees challengers)
  - `HGB_n450_raw`, `HGB_n450_Log` (HistGradientBoosting challengers)
- Added model-specific default parameter blocks and extended `build_model_from_spec(...)` to handle `rf`, `et`, and `hgb` kinds.
- Rewired `TARGET_SWEEP` so each target evaluates the full ordered model set on `PRIMARY_FEATURE_SET`.
- Updated manifest freeze logic (cell 26):
  - `manifest_A` keeps RF-first preference (safety anchor) under dummy-holdout gate.
  - `manifest_B` takes best gated challenger (no model preference) for TA/EC.
  - DRP remains conservative with RF-first preference and existing gate behavior.
- Syntax-validated notebook cells after patching.

### The Reasoning
- You asked to add model comparison beyond RF with the goal of score improvement.
- This keeps deployment risk controlled by preserving RF as anchor while allowing challengers to win only when gated metrics justify it.
- Target-wise challenger selection improves chance of gains without forcing one global model family.

### The Tech Debt
- Sweep size is now larger and runtime will increase materially; if needed, reduce model list or finalists depth per target.
- Current challenger/default params are fixed; lightweight target-wise tuning may still be needed for best effect.


## 2026-03-13 - Switched 02_preprocessing pipeline paths from iteration to terra artifacts

### The Change
- Updated [02_preprocessing.ipynb](d:/projects/water-quality-prediction/notebooks/02_preprocessing.ipynb) path constants to Terra dataset naming:
  - `TRAIN_PATH` -> `../data/interim/master_train_terra.parquet`
  - `TEST_PATH` -> `../data/interim/master_test_terra.parquet`
  - `CONTRACT_TXT_PATH` -> `../data/interim/feature_contract_master_terra.txt`
  - `CONTRACT_META_PATH` -> `../data/interim/feature_contract_master_terra_meta.json`
  - `OUT_TRAIN_PATH` -> `../data/interim/master_train_terra_aligned.parquet`
  - `OUT_TEST_PATH` -> `../data/interim/master_test_terra_aligned.parquet`
  - `OUT_REPORT_PATH` -> `../data/interim/feature_alignment_report_master_terra.csv`
- Updated matching notebook text outputs/listed artifact paths to Terra names for consistency.

### The Reasoning
- Current EDA and incoming features are being run on Terra-enriched files, so preprocessing had to be aligned to the same source and output naming to avoid train/inference schema drift and accidental use of stale iteration artifacts.

### The Tech Debt
- [03_model_training.ipynb](d:/projects/water-quality-prediction/notebooks/03_model_training.ipynb) still points to `*_iteration*` candidates and should be updated next to consume `*_terra_aligned*` outputs consistently.
- Existing historical contract files from iteration runs remain in `data/interim`; cleanup policy is not yet standardized.

## 2026-03-13 - Updated 03_model_training input/contract paths to Terra-aligned artifacts

### The Change
- Updated [03_model_training.ipynb](d:/projects/water-quality-prediction/notebooks/03_model_training.ipynb) path constants/candidates to Terra naming:
  - `TRAIN_PATH_CANDIDATES` now prefers `../data/interim/master_train_terra_aligned.parquet` then `../data/interim/master_train_terra.parquet`
  - `VALID_PATH_CANDIDATES` now prefers `../data/interim/master_test_terra_aligned.parquet` then `../data/interim/master_test_terra.parquet`
  - `CONTRACT_TXT_PATH` now uses `../data/interim/feature_contract_master_terra.txt`
- Updated matching printed path outputs in notebook JSON so run logs stay consistent with Terra artifacts.

### The Reasoning
- Preprocessing was moved to Terra files; modeling had to read the same aligned outputs to avoid silently training on stale `iteration` artifacts and mismatched feature contracts.

### The Tech Debt
- Historical output text in notebook cells can still become stale between runs; run the path-selection cell first each session to verify active sources.
- We still rely on filename-based dataset switching; central config could reduce future path drift.

## 2026-03-13 - Added Terra-informed high-ROI feature engineering in 03_model_training

### The Change
- Updated [03_model_training.ipynb](d:/projects/water-quality-prediction/notebooks/03_model_training.ipynb) feature-engineering cell to add low-risk derived features from existing Terra-aligned columns:
  - Hydroclimate: `aridity_idx`, `water_balance`, `evap_eff`, `runoff_ratio`, `dry_heat`, `temp_range`, `rain_wind_event`
  - Soil/hydro/pop interactions: `soil_texture_balance`, `soil_fines`, `drainage_proxy`, `upstream_human_pressure`, `stream_power_proxy`, `river_discharge_per_width`, `hydro_pressure_index`, `pop_compaction_1km`, `pop_gradient`, `pop_sum_ratio_1_to_5km`
  - SANLC aggregates: thematic `share_2020/share_2022`, `ratio_2020/ratio_2022`, `*_delta`, plus `human_land_share_2020/2022/delta`
- Kept per-class SANLC deltas behind `FE_INCLUDE_CLASS_DELTAS = False` toggle for runtime/noise control.
- Preserved rerun hygiene by dropping previously engineered columns before rebuilding features.
- Updated engineered-feature reporting cell to include the new exact feature names and `sanlc_*_ratio_*` patterns when summarizing `FULL_NUMERIC` composition.

### The Reasoning
- Mapped XAI guidance into derivations that can be created immediately from current Terra schema without waiting for new extraction sources.
- Prioritized interaction and aggregate features over high-cardinality expansions to reduce instability under pseudo-spatial validation.

### The Tech Debt
- Feature set size continues to grow; a target-wise ablation/pruning pass is still needed to remove non-contributing engineered columns.
- Some thematic keyword mapping for SANLC is heuristic and should eventually be validated against class taxonomy docs.

## 2026-03-13 - Added OSM feature-engineering block and regularization ladder in 03_model_training

### The Change
- Updated [03_model_training.ipynb](d:/projects/water-quality-prediction/notebooks/03_model_training.ipynb) `engineer_features(...)` to include OSM-specific derived features:
  - Source transforms per `mine/farm/wastewater`: `osm_log_dist_*`, `osm_inv_dist_*`, `osm_near_*_lt1km`, `osm_near_*_lt5km`, `osm_log_count_*_1km`, `osm_log_count_*_5km`, `osm_local_ratio_*`, `osm_ring_count_*`, `osm_pressure_*`
  - Aggregates/interactions: `osm_total_pressure`, `farm_rain`, `farm_runoff`, `mine_runoff`, `ww_dry`, `ww_urban`, `farm_crop`, `mine_land`
  - Added rerun-cleanup for these engineered columns to avoid duplicate/stale features in notebook iterative runs.
- Updated engineered feature tracking in feature-set cell so new OSM engineered columns are counted in `engineered_cols_in_full`.
- Replaced simple RF/ET/HGB model set with a regularized mini-ladder in model cell:
  - Added RF regularized variants (`RF_regA_raw`, `RF_regB_raw`, `RF_regA_Log`)
  - Added ET regularized variant (`ET_regA_raw`)
  - Added HGB regularized variants (`HGB_regA_raw`, `HGB_regB_raw`) with `early_stopping`, `validation_fraction`, and `n_iter_no_change`
- Added compact target-wise model sweep (`TARGET_MODEL_ORDER`) to limit runtime while preserving baseline comparison and regularization coverage.

### The Reasoning
- OSM raw features are sparse/skewed; transforms and pressure-style composites make them easier for tree models to use under spatial shift.
- Previous challengers showed CV gains but occasional pseudo-holdout collapse; stronger regularization is needed to improve robustness without dropping RF baseline anchoring.
- A target-wise compact sweep keeps iteration speed practical under deadline constraints.

### The Tech Debt
- Feature count has increased; next cleanup step should be fold-stability pruning (drop engineered features with unstable fold importances/effects).
- Existing output cells in the notebook still contain historical render text from prior runs and may display stale examples until rerun.

## 2026-03-13 - Added Shot D submission hedge and optional spatial-group RFECV path

### The Change
- Updated [03_model_training.ipynb](d:/projects/water-quality-prediction/notebooks/03_model_training.ipynb) model/sweep cell to include an optional RFECV stage:
  - Added `RFECV_ENABLED` toggle (`False` by default), `RFECV_STEP`, `RFECV_MIN_FEATURES`, and `RFECV_N_JOBS`
  - RFECV uses `GroupKFold` on `spatial_group` and runs on train-only groups (`~is_pseudo_valid`) to avoid pseudo-holdout leakage into feature selection
  - Creates target-specific feature sets (`RFECV_<target_key>`) when enabled
  - `TARGET_SWEEP` now maps each target to either RFECV-selected features or `PRIMARY_FEATURE_SET` fallback.
- Expanded submission construction section from Shot A/B/C to Shot A/B/C/D:
  - Added **Shot D** (`D_drp_softhedge`) with robust DRP soft hedge
  - Shot D always includes a small DRP model contribution (even when DRP gate is negative):
    - `alpha=0.25` if DRP gate positive
    - `alpha=0.05` if DRP gate negative
  - Added path save/export and diagnostics placeholders for Shot D.
- Updated markdown headers and diagnostics comments to include Shot D upload path/tracker ordering.

### The Reasoning
- Recent leaderboard drop despite local positives pointed to DRP over-conservatism from hard median fallback. Shot D introduces a low-risk way to recover model signal without fully trusting unstable DRP models.
- RFECV is added as a controlled, optional tool so feature pruning can be tested under spatial-group CV constraints rather than random split assumptions.

### The Tech Debt
- RFECV runtime can be high on large feature sets; further acceleration may require staged prefiltering or larger `step` values for quick sweeps.
- Submission variants continue to grow; a small experiment registry table (run id -> shot -> upload result) should be persisted outside notebook output for cleaner tracking.

## 2026-03-13 - Added persistent leaderboard submission tracker CSV

### The Change
- Added [leaderboard_submission_tracker.csv](d:/projects/water-quality-prediction/reports/leaderboard_submission_tracker.csv) with a fixed schema for submission-level tracking:
  - `entry_id`, `submitted_at_local`, `leaderboard_score`, `submission_variant`, `description`, `notebook_path`, `data_source_tag`, `split_strategy_tag`, `manifest_tag`, `notes`
- Seeded the tracker with the current positive leaderboard run:
  - `2026-03-13 08:12:37`, score `0.0149`, variant `D`, description `Spatial-CV RF/ET + OSM/terra, DRP soft-hedge (dummy-gated)`.

### The Reasoning
- Notebook outputs and MLflow runs are useful internally but not ideal for quick submission-history decisions under tight slot budgets.
- A single CSV registry makes it easier to compare upload outcome vs pipeline variant before using remaining leaderboard submissions.

### The Tech Debt
- Tracker currently stores submission-level metrics only; it should later include linked local metrics (`r2`, `holdout_r2`, `delta_vs_dummy`) and file hashes for stronger reproducibility.
- No automation yet for appending new rows directly from notebook runs; entries are currently manual.

## 2026-03-13 - Patched 03_model_training to spatial v2 split with contiguous holdout + buffer exclusion

### The Change
- Updated [03_model_training.ipynb](d:/projects/water-quality-prediction/notebooks/03_model_training.ipynb) split configuration to `spatial v2`:
  - `SPLIT_STRATEGY = SpatialV2_GroupKFold+ContiguousHoldout+Buffer`
  - `GROUP_DEFINITION_VERSION = kmeans_latlon_v3_contiguous_holdout_buffer`
  - Added `HOLDOUT_BUFFER_DEG` and tightened holdout margin.
- Replaced the previous nearest-group holdout selector with a contiguous-growth selector:
  - seed from validation-footprint bbox overlap
  - grow holdout by nearest-centroid adjacency until `min_groups/min_frac` constraints are satisfied.
- Added `assign_spatial_v2_masks(...)`:
  - `is_pseudo_valid` for holdout groups
  - `is_buffer_excluded` for near-boundary non-holdout rows
  - `is_spatial_v2_active` for rows eligible in CV/selection.
- Wired downstream cells to respect spatial v2 masks:
  - `grouped_oof_eval(...)` now filters to `is_spatial_v2_active`
  - RFECV train mask now uses `(~is_pseudo_valid) & is_spatial_v2_active`
  - feature-set `RESERVED_COLS` now excludes `is_buffer_excluded` and `is_spatial_v2_active`.

### The Reasoning
- Prior pseudo-spatial split selected groups by proximity only and could still include boundary-adjacent rows that leak location signal.
- Contiguous holdout plus explicit buffer exclusion is a stricter proxy for unseen-area generalization and closer to your observed train/valid geography pattern.
- Keeping the rest of the modeling pipeline intact preserves comparability while upgrading only the validation geometry.

### The Tech Debt
- Contiguity is centroid-based and still heuristic; polygon/H3-based region masks would be more faithful if time allows.
- Buffer uses axis-aligned bbox expansion; geodesic distance buffering would be more precise.
- Historical notebook outputs are stale and should be refreshed by rerunning split + scout/full cells end-to-end.

## 2026-03-13 - Logged latest 03_model_training run into tracker and added structured local metrics columns

### The Change
- Updated [leaderboard_submission_tracker.csv](d:/projects/water-quality-prediction/reports/leaderboard_submission_tracker.csv):
  - Expanded schema to include per-target local metrics:
    - `local_ta_r2`, `local_ec_r2`, `local_drp_r2`
    - `local_ta_holdout_r2`, `local_ec_holdout_r2`, `local_drp_holdout_r2`
  - Added `status` column to distinguish submitted vs local-only runs.
- Added a new row for the latest executed notebook snapshot from [03_model_training.ipynb](d:/projects/water-quality-prediction/notebooks/03_model_training.ipynb):
  - `entry_id=2026-03-13_0924_local`
  - split tag: `SpatialV2_GroupKFold+ContiguousHoldout+Buffer`
  - status: `local_only_not_submitted`
  - captured local manifest-A metrics for TA/EC/DRP.
- Preserved prior submitted row (`2026-03-13_081237`, leaderboard `0.0149`) and backfilled its local metrics for direct run-to-run comparison.

### The Reasoning
- The earlier tracker was submission-centric only; it could not support apples-to-apples local metric comparisons against prior runs.
- Adding local metric columns enables fast go/no-go checks before consuming leaderboard slots.

### The Tech Debt
- Current comparison values are still manually derived from notebook output text; this should be auto-exported from the notebook run objects to avoid transcription drift.
- `submitted_at_local` for local-only runs may reflect notebook stamp granularity rather than exact cell finish time.

## 2026-03-13 - Enriched submission tracker with explicit run-to-run comparison fields

### The Change
- Updated [leaderboard_submission_tracker.csv](d:/projects/water-quality-prediction/reports/leaderboard_submission_tracker.csv) to include comparison-oriented columns:
  - `compare_to_entry_id`
  - `delta_ta_holdout_vs_compare`
  - `delta_ec_holdout_vs_compare`
  - `delta_drp_holdout_vs_compare`
  - `recommend_submit`
- Filled comparison values for `2026-03-13_0924_local` against the last positive submitted baseline `2026-03-13_081237`.
- Marked decision signal for the inspected local run as `recommend_submit=no` based on holdout degradation.

### The Reasoning
- We already inspect deltas verbally after runs; keeping them in the tracker avoids re-deriving the same decision context under submission-slot pressure.
- Explicit recommendation flags reduce accidental low-confidence submissions during rapid iteration.

### The Tech Debt
- `recommend_submit` is currently a manual flag; it should be computed from agreed gating rules to avoid subjective drift.
- Comparison currently targets one baseline entry id at a time; multi-baseline views (best LB, latest submitted, latest local) are not yet automated.

## 2026-03-13 - Logged latest 03_model_training local run (#2) and compared against submitted baseline

### The Change
- Updated [leaderboard_submission_tracker.csv](d:/projects/water-quality-prediction/reports/leaderboard_submission_tracker.csv) with a new local-only row:
  - `entry_id=2026-03-13_2009_local`
  - source from notebook outputs in [03_model_training.ipynb](d:/projects/water-quality-prediction/notebooks/03_model_training.ipynb) (`submission_20260313_2009_*` block / manifest A).
- Stored extracted local metrics for TA/EC/DRP and holdout metrics, then compared directly to submitted baseline `2026-03-13_081237` (~0.015 LB).
- Filled comparison deltas and set `recommend_submit=no` for this run.

### The Reasoning
- User requested explicit analysis of the latest notebook output and comparison versus the submission tracker while preserving status (`#2` not submitted).
- Keeping local run #2 in the tracker avoids repeated manual notebook parsing and makes slot decisions auditable.

### The Tech Debt
- Notebook-output parsing is still manual/text-based; this should be replaced by a structured export cell that writes manifest metrics directly to CSV.
- Current tracker cannot auto-calculate comparison deltas on append; those values are currently entered explicitly.

## 2026-03-13 - Added computed overall decision score to submission tracker

### The Change
- Updated [leaderboard_submission_tracker.csv](d:/projects/water-quality-prediction/reports/leaderboard_submission_tracker.csv) with new column:
  - `overall_local_decision_score`
- Backfilled scores for all tracked runs using:
  - `(local_ta_holdout_r2 + local_ec_holdout_r2 + local_drp_holdout_r2) / 3`
- Kept existing `recommend_submit` flags, now supported by a single scalar summary metric.

### The Reasoning
- User requested a compact computed signal to compare runs quickly before spending submission slots.
- Mean holdout R2 across targets is simple, transparent, and directly aligned with the tracker’s per-target holdout fields.

### The Tech Debt
- `recommend_submit` is still not auto-derived from `overall_local_decision_score` plus DRP safety gates; this should be automated to avoid manual inconsistency.
- Score aggregation is currently unweighted; if competition target weighting is known, we should switch to weighted aggregation.

## 2026-03-13 - Created simplified training notebook 03d with sentinel-first baseline flow

### The Change
- Added [03d_model_training_simple.ipynb](d:/projects/water-quality-prediction/notebooks/03d_model_training_simple.ipynb).
- Kept the same high-level section flow as `03_model_training.ipynb` (config, loading, FE, feature sets, preprocessing, models, grouped eval, scout/full stages, manifest, submission build).
- Simplified the modeling approach:
  - master sentinel aligned data as default input
  - lightweight feature engineering only
  - compact feature sets (`SPECTRAL_CORE`, `SPECTRAL_PLUS_COORDS`, `MIXED_LITE`)
  - small model family (RF/ET/HGB baseline variants)
  - simple pseudo-holdout grouped evaluation and two submission shots (A direct, B DRP hedge).
- Embedded target-specific realistic local ceiling ranges in the notebook config output.

### The Reasoning
- Current pipeline complexity made it harder to separate true signal from validation machinery.
- A clean baseline notebook allows faster iteration and clearer attribution of what actually improves score.
- Sentinel-first setup aligns with the current direction to prioritize transferable spectral signal.

### The Tech Debt
- The simplified notebook still uses heuristic pseudo-holdout construction and should be recalibrated if leaderboard behavior shifts.
- Manifest-to-tracker export is not automated in `03d`; run logging remains manual.

## 2026-03-13 - TA-focused feature-combo patch in 03d simple notebook

### The Change
- Updated [03d_model_training_simple.ipynb](d:/projects/water-quality-prediction/notebooks/03d_model_training_simple.ipynb) with TA-targeted feature logic:
  - Added engineered TA proxy features in `engineer_features(...)`:
    - `soil_fines_0_5cm`, `soil_texture_balance`
    - `residence_time_proxy`, `flatness_proxy`, `basin_flatness_proxy`
    - `rain_dilution_proxy`, `event_runoff_proxy`
    - `weathering_capacity_proxy`, `contact_weathering_proxy`
  - Added TA-specific frozen feature sets:
    - `TA_HYDRO_GEO`
    - `TA_RESIDENCE_SOIL`
  - Updated `TARGET_SWEEP['Total Alkalinity']` to use TA-specific sets instead of generic `MIXED_LITE`.
- Kept EC/DRP sweep definitions unchanged to isolate TA hypothesis impact.

### The Reasoning
- User observed TA not learning stable patterns under the simple setup.
- TA domain assumptions (bedrock/weathering residence, slope/flow speed, soil weathering surface, rainfall dilution) were translated into explicit proxy features and TA-only feature combinations.
- This preserves notebook simplicity while making TA model inputs more mechanistic and target-specific.

### The Tech Debt
- TA geology is still proxied from available covariates; no direct lithology/bedrock layer is included yet.
- Several proxies are correlated and may need post-run pruning to prevent redundant feature noise.
- The notebook still depends on heuristic pseudo-holdout and may under/over-penalize TA depending on split geometry.

## 2026-03-13 - Simplified 03d data loading to raw sentinel train/test (removed aligned/fallback path logic)

### The Change
- Updated [03d_model_training_simple.ipynb](d:/projects/water-quality-prediction/notebooks/03d_model_training_simple.ipynb) data-loading cell:
  - Removed candidate/fallback path resolution (`*_CANDIDATES` and `pick_first_existing`).
  - Hard-pinned input paths to:
    - `../data/interim/master_train_sentinel.parquet`
    - `../data/interim/master_test_sentinel.parquet`
  - Kept explicit path existence checks and contract-file guard (`feature_contract_master_sentinel.txt`).
- Updated data-loading markdown text to reflect direct raw sentinel usage.
- Cleared stale cached output in the data-loading cell so old aligned-path prints do not mislead future runs.

### The Reasoning
- User confirmed train/test schema already matches and requested removal of unnecessary aligned fallback complexity in the simple notebook.
- Direct path usage reduces branching and makes run behavior easier to reason about.

### The Tech Debt
- This notebook now assumes sentinel paths are present; if dataset source switches, the paths must be edited manually.
- Contract compatibility is still validated against TXT metadata; if contract file drifts from raw sentinel schema, this cell will fail fast (intentionally).

## 2026-03-13 - Extracted independent training notebook 03e from 03_model_training

### The Change
- Added [03e_model_training_independent.ipynb](d:/projects/water-quality-prediction/notebooks/03e_model_training_independent.ipynb) as an extracted standalone version of [03_model_training.ipynb](d:/projects/water-quality-prediction/notebooks/03_model_training.ipynb).
- Inlined environment/config behavior directly inside the notebook (no external `src/config_local.py` or `src/config_snowflake.py` imports):
  - `ENV` switch (`local` vs `snowflake`)
  - `MLFLOW_URI` assignment
  - `load_data()` and `save_artifacts(...)` function definitions for both environments.
- Removed aligned-fallback file-selection logic from the data-loading cell and set explicit sentinel paths:
  - `../data/interim/master_train_sentinel.parquet`
  - `../data/interim/master_test_sentinel.parquet`
- Kept the rest of the training/evaluation pipeline structure intact and cleared code-cell outputs in the new notebook.

### The Reasoning
- The user requested a fully independent notebook that can run without relying on config modules.
- Inlining the config branch logic inside the notebook preserves original behavior while making execution self-contained.
- Direct sentinel-path loading removes unnecessary fallback complexity now that the dataset pairing is stable.

### The Tech Debt
- The inlined `load_data()` local branch still mirrors legacy config behavior (`master_train.parquet`) and is currently not wired into the sentinel-loading cell; if desired, unify these paths in one source of truth.
- Snowflake validation/test loading is still local-path based in this notebook flow; a full Snowflake-native validation path is not implemented yet.

## 2026-03-13 - Updated markdown descriptions in 03 and 03e training notebooks

### The Change
- Updated descriptive markdown cells in [03_model_training.ipynb](d:/projects/water-quality-prediction/notebooks/03_model_training.ipynb) and [03e_model_training_independent.ipynb](d:/projects/water-quality-prediction/notebooks/03e_model_training_independent.ipynb).
- Replaced stale narrative that referenced `Region`, old XGB-focused shortlist text, and 3-shot outputs.
- Aligned descriptions with current implemented flow:
  - spatial-group CV (`spatial_group`, pseudo-holdout, buffer-exclusion)
  - tree-ensemble bank (RF/ET/HGB) and optional RFECV
  - Manifest A/B freezing logic
  - four submission variants (A/B/C/D)
- Added notebook-specific context:
  - `03_model_training.ipynb`: config imported from `src/config_local.py` / `src/config_snowflake.py`
  - `03e_model_training_independent.ipynb`: config logic inlined and direct sentinel file paths.

### The Reasoning
- User requested markdown description adjustment; previous text no longer matched the code path and could mislead reruns.
- Keeping notebook narrative synchronized with code reduces onboarding time and run mistakes during deadline iterations.

### The Tech Debt
- Some section headings remain generic and could be standardized across all `03*` notebooks via shared templates.
- Optional diagnostics are still documented as comments rather than toggles; adding parameterized run modes would improve consistency.
