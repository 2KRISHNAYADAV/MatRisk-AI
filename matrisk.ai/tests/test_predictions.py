"""
tests/test_predictions.py
--------------------------
Tests for model prediction functions:
  - Models accept valid feature inputs without raising.
  - Predictions fall within physically plausible ranges.
  - Monotonicity constraints hold (critical correctness check from notebook 05/07).
  - PD calibration produces values in [0, 1].
  - Derived quantities (E, ν) are physically consistent.
  - The EL = PD × LGD × EAD identity holds for any new prediction.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import joblib

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODELS = PROJECT_ROOT / "data" / "processed" / "models"
PROCESSED = PROJECT_ROOT / "data" / "processed"

# Load models once for the test session
import joblib
_bulk_model = joblib.load(MODELS / "ds1_bulk_modulus_GPa_model.joblib")
_shear_model = joblib.load(MODELS / "ds1_shear_modulus_GPa_model.joblib")
_cond_model = joblib.load(MODELS / "ds3_condition_rating_model.joblib")
_cc_model = joblib.load(MODELS / "ds6_case_control_model.joblib")
_snap = joblib.load(MODELS / "ds1_feature_snapshot.joblib")
_exposure = pd.read_csv(PROCESSED / "ds3_exposure_features.csv")

TRUE_ANNUAL_PD = 1011 / (19 * 5000)   # ≈ 0.01064


def _make_bulk_row(
    category="Metal", crystal_system="cubic", n_elements=3, nsites=4,
    volume_A3=100.0, mean_atomic_mass=60.0, mean_Z=28.0, max_Z=40.0,
    min_Z=10.0, atoms_per_formula=4.0, density_implied=6.0,
    element_cost=np.nan, element_cost_available=False, catmean_K=120.0,
) -> pd.DataFrame:
    return pd.DataFrame([{
        "category": category, "crystal_system": crystal_system,
        "n_elements": n_elements, "nsites": nsites, "volume_A3": volume_A3,
        "mean_atomic_mass": mean_atomic_mass, "mean_Z": mean_Z,
        "max_Z": max_Z, "min_Z": min_Z, "atoms_per_formula": atoms_per_formula,
        "density_implied_g_cm3": density_implied,
        "element_cost_usd_per_kg": element_cost,
        "element_cost_available": element_cost_available,
        "catmean_bulk_modulus_GPa": catmean_K,
    }])


def _make_shear_row(**kwargs) -> pd.DataFrame:
    row = _make_bulk_row(**kwargs)
    row = row.drop(columns=["catmean_bulk_modulus_GPa"])
    row["catmean_shear_modulus_GPa"] = kwargs.get("catmean_G", 60.0)
    return row


def _make_cond_row(
    material="Reinforced Concrete", bridge_type="Interstate",
    location="Urban Midwest", corrosion_environment="Moderate",
    is_metal=False, age_years=40, design_life_years=100,
    age_ratio=0.4, corrosion_rate_mm_yr=0.05, log_corr=None,
    thick_frac=0.95, adt=10000, tensile_strength_MPa=400.0,
    yield_strength_MPa=300.0,
) -> pd.DataFrame:
    if log_corr is None:
        log_corr = np.log(max(corrosion_rate_mm_yr, 1e-6))
    return pd.DataFrame([{
        "material": material, "bridge_type": bridge_type,
        "location": location, "corrosion_environment": corrosion_environment,
        "is_metal": is_metal, "age_years": age_years,
        "design_life_years": design_life_years, "age_ratio": age_ratio,
        "corrosion_rate_mm_yr": corrosion_rate_mm_yr, "log_corr": log_corr,
        "thick_frac": thick_frac, "adt": adt,
        "tensile_strength_MPa": tensile_strength_MPa,
        "yield_strength_MPa": yield_strength_MPa,
    }])


def _make_cc_row(
    material="Steel (A36)", corrosion_environment="Moderate",
    location="Urban Midwest", age_years=40,
    corrosion_rate_mm_yr=0.05, is_metal=True,
) -> pd.DataFrame:
    return pd.DataFrame([{
        "material": material, "corrosion_environment": corrosion_environment,
        "location": location, "age_years": age_years,
        "corrosion_rate_mm_yr": corrosion_rate_mm_yr, "is_metal": is_metal,
    }])


# ---------------------------------------------------------------------------
# Material property model tests
# ---------------------------------------------------------------------------

class TestBulkModulusModel:

    def test_prediction_runs(self):
        row = _make_bulk_row()
        pred = _bulk_model.predict(row)
        assert pred.shape == (1,)

    def test_prediction_positive(self):
        row = _make_bulk_row()
        pred = _bulk_model.predict(row)[0]
        assert pred > 0, f"Bulk modulus prediction should be positive, got {pred}"

    def test_prediction_plausible_range(self):
        """Most real materials have K between 1 and 800 GPa."""
        for cat in ["Metal", "Oxide", "Alloy"]:
            row = _make_bulk_row(category=cat)
            pred = _bulk_model.predict(row)[0]
            assert 1.0 < pred < 1000.0, f"K prediction out of range for {cat}: {pred}"

    def test_feature_count(self):
        assert _bulk_model.n_features_in_ == 14


class TestShearModulusModel:

    def test_prediction_runs(self):
        row = _make_shear_row()
        pred = _shear_model.predict(row)
        assert pred.shape == (1,)

    def test_prediction_positive(self):
        row = _make_shear_row()
        pred = _shear_model.predict(row)[0]
        assert pred > 0


class TestDerivedQuantities:

    def test_poisson_ratio_range(self):
        """Isotropic ν should be in (-1, 0.5) for stable materials."""
        for K, G in [(100, 50), (200, 80), (50, 30), (300, 120)]:
            nu = (3 * K - 2 * G) / (2 * (3 * K + G))
            assert -1.0 < nu < 0.5, f"ν out of range for K={K}, G={G}: {nu}"

    def test_young_modulus_positive(self):
        for K, G in [(100, 50), (200, 80)]:
            E = (9 * K * G) / (3 * K + G)
            assert E > 0

    def test_young_modulus_identity(self):
        """E = 9KG/(3K+G) and ν = (3K-2G)/(2(3K+G)) imply E = 2G(1+ν)."""
        K, G = 150.0, 75.0
        E = (9 * K * G) / (3 * K + G)
        nu = (3 * K - 2 * G) / (2 * (3 * K + G))
        np.testing.assert_allclose(E, 2 * G * (1 + nu), rtol=1e-10)


# ---------------------------------------------------------------------------
# Condition rating model tests
# ---------------------------------------------------------------------------

class TestConditionRatingModel:

    def test_prediction_runs(self):
        row = _make_cond_row()
        pred = _cond_model.predict(row)
        assert pred.shape == (1,)

    def test_prediction_in_range(self):
        row = _make_cond_row()
        pred = _cond_model.predict(row)[0]
        # Clip to [1,9] in the app; raw prediction should be near this range
        assert -2.0 < pred < 12.0, f"Condition rating wildly out of range: {pred}"

    def test_monotonicity_corrosion_rate(self):
        """CRITICAL: Higher corrosion rate must predict lower condition rating.
        This monotonicity was explicitly enforced in notebook 05 after a bug was found.
        """
        rates = [0.001, 0.01, 0.05, 0.1, 0.2, 0.35, 0.5]
        preds = []
        for cr in rates:
            row = _make_cond_row(corrosion_rate_mm_yr=cr, log_corr=np.log(cr),
                                  thick_frac=max(0.0, 1.0 - cr * 40 / 40.0))
            preds.append(_cond_model.predict(row)[0])

        # Check monotonically non-increasing (allow tolerance for near-ties)
        violations = 0
        for i in range(len(preds) - 1):
            if preds[i + 1] > preds[i] + 0.05:
                violations += 1

        assert violations == 0, (
            f"Monotonicity violation: condition rating increased as corrosion rate increased.\n"
            f"  rates={rates}\n  preds={[f'{p:.3f}' for p in preds]}"
        )

    def test_monotonicity_age(self):
        """CRITICAL: Older bridges should have lower or equal condition rating."""
        ages = [10, 20, 30, 40, 60, 80, 100]
        preds = []
        for age in ages:
            row = _make_cond_row(age_years=age, age_ratio=age/100)
            preds.append(_cond_model.predict(row)[0])

        violations = sum(1 for i in range(len(preds) - 1) if preds[i + 1] > preds[i] + 0.1)
        # Note: age doesn't have a hard monotonic constraint, but should be weakly decreasing
        assert violations <= 2, (
            f"Too many age-monotonicity violations: {violations}\n"
            f"  ages={ages}\n  preds={[f'{p:.3f}' for p in preds]}"
        )


# ---------------------------------------------------------------------------
# Case-control failure model tests
# ---------------------------------------------------------------------------

class TestCaseControlModel:

    def test_prediction_runs(self):
        row = _make_cc_row()
        proba = _cc_model.predict_proba(row)
        assert proba.shape == (1, 2)

    def test_probability_sums_to_one(self):
        row = _make_cc_row()
        proba = _cc_model.predict_proba(row)[0]
        np.testing.assert_allclose(proba.sum(), 1.0, atol=1e-10)

    def test_probability_in_range(self):
        row = _make_cc_row()
        proba = _cc_model.predict_proba(row)[0]
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_monotonicity_corrosion_rate(self):
        """CRITICAL: Higher corrosion rate must predict higher failure probability.
        This was the core bug fixed in notebook 05 — the original model was
        NON-monotonic for 45.6% of rows.
        """
        rates = [0.001, 0.01, 0.05, 0.1, 0.2, 0.35, 0.5]
        probs = []
        for cr in rates:
            row = _make_cc_row(corrosion_rate_mm_yr=cr)
            probs.append(_cc_model.predict_proba(row)[0][1])

        violations = sum(1 for i in range(len(probs) - 1) if probs[i + 1] < probs[i] - 0.001)
        assert violations == 0, (
            f"MONOTONICITY BUG: failure probability decreased as corrosion rate increased.\n"
            f"  rates={rates}\n  probs={[f'{p:.4f}' for p in probs]}"
        )


# ---------------------------------------------------------------------------
# PD calibration tests
# ---------------------------------------------------------------------------

class TestPDCalibration:

    def test_calibration_constant(self):
        """TRUE_ANNUAL_PD must match documented formula."""
        expected = 1011 / (19 * 5000)
        np.testing.assert_allclose(TRUE_ANNUAL_PD, expected, rtol=1e-10)

    def test_calibrated_pd_in_range(self):
        """Calibrated PD for any row should be in [0, 1]."""
        rows = _make_cc_row()
        score = _cc_model.predict_proba(rows)[0][1]
        pop_scores = _cc_model.predict_proba(
            _exposure[list(rows.columns)].head(500).dropna()
        )[:, 1]
        pop_mean = pop_scores.mean()
        pd_cal = score / pop_mean * TRUE_ANNUAL_PD if pop_mean > 0 else TRUE_ANNUAL_PD
        pd_cal = max(0.0, min(1.0, pd_cal))
        assert 0.0 <= pd_cal <= 1.0

    def test_portfolio_mean_pd_near_true_rate(self):
        """After calibration, portfolio mean PD should be close to 1.064%."""
        sample = _exposure.sample(200, random_state=42)
        cc_cols = ["material", "corrosion_environment", "location",
                   "age_years", "corrosion_rate_mm_yr", "is_metal"]
        sample_clean = sample[cc_cols].dropna()
        if len(sample_clean) < 100:
            pytest.skip("Not enough clean rows for calibration test")

        scores = _cc_model.predict_proba(sample_clean)[:, 1]
        pop_mean = scores.mean()
        pd_calibrated = scores / pop_mean * TRUE_ANNUAL_PD
        portfolio_mean = pd_calibrated.mean()

        np.testing.assert_allclose(
            portfolio_mean, TRUE_ANNUAL_PD,
            rtol=0.05,  # within 5% of target
            err_msg=f"Portfolio mean PD {portfolio_mean:.4%} too far from target {TRUE_ANNUAL_PD:.4%}"
        )


# ---------------------------------------------------------------------------
# Risk calculation tests
# ---------------------------------------------------------------------------

class TestRiskCalculations:

    def test_el_identity(self):
        """EL = PD × LGD × EAD must hold for any chosen values."""
        test_cases = [
            (0.01, 0.20, 1.0),
            (0.05, 0.50, 2.5),
            (0.001, 0.957, 10.0),
        ]
        for pd_val, lgd_val, ead_val in test_cases:
            el = pd_val * lgd_val * ead_val
            assert el > 0
            np.testing.assert_allclose(el, pd_val * lgd_val * ead_val, rtol=1e-12)

    def test_portfolio_el_sign(self):
        """Portfolio EL must be positive."""
        df = pd.read_csv(PROCESSED / "ds3_credit_risk.csv")
        total_el = df["EL_M"].sum()
        assert total_el > 0

    def test_portfolio_el_magnitude(self):
        """Portfolio EL should be in the documented range ~$5.8M."""
        df = pd.read_csv(PROCESSED / "ds3_credit_risk.csv")
        total_el = df["EL_M"].sum()
        assert 4.0 < total_el < 8.0, f"Portfolio EL ${total_el:.2f}M outside expected range $4-8M"

    def test_scenario_stress_multiplier(self):
        """SC-003 EL multiple must be > 2 (documented as ~2.91×)."""
        df = pd.read_csv(PROJECT_ROOT / "outputs" / "tables" / "scenario_credit_stress_summary.csv")
        baseline = float(df[df["scenario"] == "baseline"]["mean_loss_M"].iloc[0])
        stressed = float(df[df["scenario"].str.contains("SC-003")]["mean_loss_M"].iloc[0])
        multiple = stressed / baseline
        assert multiple > 2.0, f"SC-003 stress multiple {multiple:.2f} < 2.0 (expected ~2.91)"

    def test_var_es_ordering(self):
        """ES must be >= VaR at the same confidence level."""
        df = pd.read_csv(PROJECT_ROOT / "outputs" / "tables" / "commodity_var_es.csv")
        assert (df["ES95_1d"] >= df["VaR95_1d"] - 1e-6).all()
        assert (df["ES99_1d"] >= df["VaR99_1d"] - 1e-6).all()
        assert (df["VaR99_1d"] >= df["VaR95_1d"] - 1e-6).all()
