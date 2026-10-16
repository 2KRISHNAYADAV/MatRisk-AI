# Notes on this handoff — notebooks 03-09

## What you got
All nine notebooks in `notebooks/`, each fully executed with real outputs and figures (not just source code):
`01_data_audit.ipynb`, `02_material_EDA.ipynb`, `03_financial_EDA.ipynb` (your uploads), plus
`03_feature_engineering.ipynb`, `04_material_property_models.ipynb`, `05_failure_prediction.ipynb`,
`06_financial_risk_engine.ipynb`, `07_scenario_stress_testing.ipynb`, `08_model_interpretability.ipynb`, and
`09_final_validation.ipynb` (built in this session). Supporting files:
- `data/processed/*.csv` — every feature store, model-ready table, risk score and prediction file the notebooks write.
- `data/processed/models/*.joblib` — the saved, fitted models 05 and 04 persist (K/G regressors, condition-rating
  regressor, case-control classifier), reused by 06, 07, 08 and 09 rather than retrained each time.
- `outputs/tables/*.csv` — every summary table: feature manifest, model performance scorecards, SHAP importance,
  stress-scenario results, the final consistency-check and assumptions-ledger tables from 09.
- `outputs/figures/*.png` — every figure embedded in every notebook (~33 total).
- `src/data/audit.py`, `src/materials/composition.py` — see caveat below.

## A real bug found and fixed along the way (05 → 06 → 07 → 08)
Building 07's stress scenarios surfaced a genuine problem: the first version of 05's case-control failure classifier
(`HistGradientBoostingClassifier`, no constraints) was **not monotonic in `corrosion_rate_mm_yr`** — for 45.6% of rows,
*lowering* corrosion rate predicted *higher* failure probability, a pure tree-ensemble artifact in sparse regions of
feature space. This surfaced concretely when SC-005 (a corrosion-*reducing* scenario) came out loss-increasing, which
made no sense. Fixed by refitting both the condition-rating model and the case-control classifier with `monotonic_cst`
constraints (sklearn's native support for this), re-verified directly in 08 with partial dependence plots, and
re-confirmed in 09's consistency checks. 06's numbers shift slightly as a result (EL $5.89M → $5.82M) — the fix was
about correctness under perturbation, not a material change to the baseline numbers themselves. This is exactly the
kind of thing worth telling you about rather than quietly fixing: if you rebuild any of this with your own code, test
for monotonicity before trusting a tree model's behavior under a shock/scenario input.

## PD calibration approach (06)
05's case-control classifier outputs a score trained on an artificial 16.8%-positive sample — not a real probability.
06 rescales it to an annual PD by **matching the population mean** to the known rate (1,011 DS6 bridge failures / 19
years / 5,000 DS3 bridges = 1.06%/yr), preserving the model's relative ranking while fixing its scale. This is simpler
and more robust than a King-Zeng-style pointwise prior correction (tried first; it didn't recover the population mean
reliably because the base classifier isn't well-calibrated to its own training prevalence). If you have a better-known
true population rate, swap the one constant (`TRUE_ANNUAL_PD` in 06) rather than re-deriving the whole calibration.

## Important caveat: the `src/` modules are reconstructed, not your originals
Your three notebooks `import from src.data import audit as A` and `from src.materials import composition as comp_mod`,
but only the notebooks and CSVs were uploaded — not your `src/` package. I rebuilt the subset of that API your notebooks
actually call (`load_csvs`, `load_xlsx_blocks`, `parse_elements`, `_poisson_from_KG`, `CSV_FILES`, `XLSX_FILE`,
`README_FILE`, `METAL_MATERIALS_DS3`, plus a new `young_from_KG` for this notebook) by reading your code's usage, not by
copying your implementation — because I don't have it.

**I validated this by re-deriving your own EDA numbers from scratch and checking they match:**

| Quantity | My notebook | Your EDA notebook |
|---|---|---|
| Density within ±10% of composition-implied | 4.7% | 4.7% (02, §A4) |
| Pugh ductile share (K/G>1.75) | 62.9% | 62.9% (02, §A5) |
| DS5 coverage of DS1 (element cost) | 32.6% | 32.6% (02, §A7 / 03, §C3) |
| Fatigue cycles within 10% of ADT·365·age/1e6 | 81.5% | 81.5% (02, §B3) |
| Workbook critical/original thickness ratio range | 0.417–0.500 | 0.417–0.500 (02, §B4) |
| Bridges below critical (high ratio) | 0.8% | 0.5–0.8% (02, §B4) |
| Median years to critical thickness | 436 | ≈436 (02, §B4) |
| AUC, corrosion rate alone (case-control) | 0.603 | 0.60 (02, §C5) |
| AUC, age alone (case-control) | 0.475 | 0.475 (02, §C5) |
| term_spread raw scale range | 4.9–161.8 | 4.9–161.8 (03, §A7) |

If your real `src/` modules differ in some edge case (e.g. a different xlsx-block-splitting heuristic), the numbers above
are how to check: re-run Part A/B of this notebook with your real modules swapped in and confirm the table still matches.

## Headline results across 04-09 (all out-of-fold / honestly re-verified, see each notebook for how)
- **04:** composition+category models beat the plain category-mean baseline on only 1 of 5 DS1 targets (density) — and
  even that win rides on a physically backwards correlation with cell volume (dug into directly in 08). `is_metal` is
  solved exactly (AUC≈1.0); `is_stable` tops out ≈0.71-0.72 AUC without `energy_above_hull_eV`.
- **05:** condition_rating is close to a 2-variable function of age+corrosion (R²≈0.79-0.81, richer features don't add
  much); the case-control classifier reaches AUC≈0.74 vs. a 0.60 single-feature baseline; `severity` is confirmed
  genuinely unpredictable (negative out-of-fold R²) — the right model there is the empirical frequency table, not a regression.
- **06:** portfolio credit EL ≈$5.8M/yr on a $1,877M book; premiums collected (~$56M/yr) run ~9.7x that, with near-zero
  correlation to it — consistent with 03's "premiums aren't risk-based" audit finding, now confirmed at the portfolio level.
- **07:** SC-003 (climate-accelerated corrosion) ≈triples credit EL; SC-005 (favourable alloy scenario) cuts it to
  ≈0.73x — both run through 05/06's actual fitted models, not a separate toy calculation.
- **08:** SHAP confirms `category` dominates the DS1 models (56-66% of total |SHAP|); partial dependence confirms both
  05 monotonic fixes actually produced monotonic predictions; `location` turned out to **deterministically** determine
  `corrosion_environment` (not just correlate with it) — a genuine simplification opportunity for a production version.
- **09:** 5/5 cross-notebook consistency checks pass; the full assumptions ledger (15 logged decisions across 6
  notebooks) is in `outputs/tables/final_assumptions_ledger.csv`.

## Naming note
I named the file `03_feature_engineering.ipynb` exactly as you asked for next. Your `03_financial_EDA.ipynb` already
uses the `03_` prefix, so there are now two different `03_...` notebooks — harmless, but it means alphabetical sorting
puts `03_feature_engineering.ipynb` *before* `03_financial_EDA.ipynb` (e < i), even though it depends on it. If you'd
rather keep strict run-order sorting, renaming this one to `04_feature_engineering.ipynb` (and shifting your planned
`04_material_property_models.ipynb` … `09_final_validation.ipynb` to `05`…`10`) would fix that — just say the word.

## Where to drop this in your real project
Copy `notebooks/03_feature_engineering.ipynb` into your existing project's `notebooks/` folder (or wherever 01-03 live).
It auto-discovers `ROOT` the same way your other notebooks do (walks up from `cwd` looking for a folder containing both
`src/` and `data/`), so as long as it sits next to your real `src/` and `data/raw/`, it should run as-is — just double
check your real `src.data.audit` module exposes the same function names used above (`load_csvs`, `load_xlsx_blocks`,
`parse_elements`, `_poisson_from_KG`), or swap in the ones here.
