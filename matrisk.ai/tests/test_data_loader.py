"""
tests/test_data_loader.py
--------------------------
Tests that all required data files exist, load without error, and have
the expected shapes / columns / value ranges.

These tests do NOT use Streamlit — they call the underlying pandas/joblib
logic directly.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import joblib

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS = PROCESSED / "models"
TABLES = PROJECT_ROOT / "outputs" / "tables"
FIGURES = PROJECT_ROOT / "outputs" / "figures"


# ---------------------------------------------------------------------------
# Processed CSV files
# ---------------------------------------------------------------------------

class TestProcessedCSVs:

    def test_ds1_material_features_exists(self):
        assert (PROCESSED / "ds1_material_features.csv").exists()

    def test_ds1_material_features_shape(self):
        df = pd.read_csv(PROCESSED / "ds1_material_features.csv")
        assert df.shape[0] == 5500, f"Expected 5500 rows, got {df.shape[0]}"
        assert "category" in df.columns
        assert "bulk_modulus_GPa" in df.columns
        assert "density_g_cm3" in df.columns

    def test_ds1_model_predictions_exists(self):
        assert (PROCESSED / "ds1_model_predictions.csv").exists()

    def test_ds1_model_predictions_shape(self):
        df = pd.read_csv(PROCESSED / "ds1_model_predictions.csv")
        assert df.shape[0] == 5500
        required_cols = [
            "bulk_modulus_GPa_actual", "bulk_modulus_GPa_pred_oof",
            "shear_modulus_GPa_actual", "shear_modulus_GPa_pred_oof",
            "density_g_cm3_actual", "density_g_cm3_pred_oof",
            "K_pred_final", "G_pred_final", "nu_derived", "E_derived_GPa",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_ds2_financial_features_exists(self):
        assert (PROCESSED / "ds2_ds4_financial_features.csv").exists()

    def test_ds2_financial_features_shape(self):
        df = pd.read_csv(PROCESSED / "ds2_ds4_financial_features.csv")
        assert df.shape[0] > 20000
        assert "commodity" in df.columns
        assert "close" in df.columns
        assert "daily_return" in df.columns

    def test_ds2_commodities(self):
        df = pd.read_csv(PROCESSED / "ds2_ds4_financial_features.csv")
        expected = {"Aluminium", "Cobalt_Index", "Copper", "Iron_Ore",
                    "Lithium_ETF", "Nickel", "RareEarth_ETF", "Steel_HRC"}
        actual = set(df["commodity"].unique())
        assert actual == expected, f"Commodity mismatch: {actual ^ expected}"

    def test_ds3_credit_risk_exists(self):
        assert (PROCESSED / "ds3_credit_risk.csv").exists()

    def test_ds3_credit_risk_shape(self):
        df = pd.read_csv(PROCESSED / "ds3_credit_risk.csv")
        assert df.shape[0] == 5000
        assert "PD_annual" in df.columns
        assert "LGD" in df.columns
        assert "EAD_M" in df.columns
        assert "EL_M" in df.columns

    def test_ds3_credit_risk_pd_range(self):
        df = pd.read_csv(PROCESSED / "ds3_credit_risk.csv")
        assert df["PD_annual"].between(0, 1).all(), "PD values outside [0, 1]"

    def test_ds3_credit_risk_lgd_range(self):
        df = pd.read_csv(PROCESSED / "ds3_credit_risk.csv")
        assert df["LGD"].between(0, 1).all(), "LGD values outside [0, 1]"

    def test_ds3_el_equals_pd_lgd_ead(self):
        """EL = PD * LGD * EAD must hold (to floating-point tolerance)."""
        df = pd.read_csv(PROCESSED / "ds3_credit_risk.csv")
        computed = df["PD_annual"] * df["LGD"] * df["EAD_M"]
        np.testing.assert_allclose(
            df["EL_M"].values, computed.values,
            rtol=1e-4, err_msg="EL != PD * LGD * EAD"
        )

    def test_ds6_failure_features_exists(self):
        assert (PROCESSED / "ds6_failure_features.csv").exists()

    def test_ds6_failure_features_shape(self):
        df = pd.read_csv(PROCESSED / "ds6_failure_features.csv")
        assert df.shape[0] == 2000


# ---------------------------------------------------------------------------
# Trained model files
# ---------------------------------------------------------------------------

class TestModelFiles:

    def test_bulk_modulus_model_exists(self):
        assert (MODELS / "ds1_bulk_modulus_GPa_model.joblib").exists()

    def test_shear_modulus_model_exists(self):
        assert (MODELS / "ds1_shear_modulus_GPa_model.joblib").exists()

    def test_condition_rating_model_exists(self):
        assert (MODELS / "ds3_condition_rating_model.joblib").exists()

    def test_case_control_model_exists(self):
        assert (MODELS / "ds6_case_control_model.joblib").exists()

    def test_feature_snapshot_exists(self):
        assert (MODELS / "ds1_feature_snapshot.joblib").exists()

    def test_bulk_modulus_model_type(self):
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = joblib.load(MODELS / "ds1_bulk_modulus_GPa_model.joblib")
        assert isinstance(m, HistGradientBoostingRegressor)

    def test_shear_modulus_model_type(self):
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = joblib.load(MODELS / "ds1_shear_modulus_GPa_model.joblib")
        assert isinstance(m, HistGradientBoostingRegressor)

    def test_condition_rating_model_type(self):
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = joblib.load(MODELS / "ds3_condition_rating_model.joblib")
        assert isinstance(m, HistGradientBoostingRegressor)

    def test_case_control_model_type(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = joblib.load(MODELS / "ds6_case_control_model.joblib")
        assert isinstance(m, HistGradientBoostingClassifier)

    def test_feature_snapshot_type(self):
        snap = joblib.load(MODELS / "ds1_feature_snapshot.joblib")
        assert isinstance(snap, pd.DataFrame)
        assert snap.shape[0] == 5500

    def test_bulk_modulus_model_features(self):
        m = joblib.load(MODELS / "ds1_bulk_modulus_GPa_model.joblib")
        expected = {"category", "crystal_system", "n_elements", "nsites", "volume_A3",
                    "mean_atomic_mass", "mean_Z", "max_Z", "min_Z", "atoms_per_formula",
                    "density_implied_g_cm3", "element_cost_usd_per_kg",
                    "element_cost_available", "catmean_bulk_modulus_GPa"}
        actual = set(m.feature_names_in_)
        assert actual == expected

    def test_case_control_model_features(self):
        m = joblib.load(MODELS / "ds6_case_control_model.joblib")
        expected = {"material", "corrosion_environment", "location",
                    "age_years", "corrosion_rate_mm_yr", "is_metal"}
        actual = set(m.feature_names_in_)
        assert actual == expected


# ---------------------------------------------------------------------------
# Output tables
# ---------------------------------------------------------------------------

class TestOutputTables:

    def test_material_model_performance_exists(self):
        assert (TABLES / "material_model_performance.csv").exists()

    def test_failure_model_performance_exists(self):
        assert (TABLES / "failure_model_performance.csv").exists()

    def test_commodity_var_es_exists(self):
        assert (TABLES / "commodity_var_es.csv").exists()

    def test_shap_feature_importance_exists(self):
        assert (TABLES / "shap_feature_importance.csv").exists()

    def test_scenario_price_shock_exists(self):
        assert (TABLES / "scenario_price_shock_summary.csv").exists()

    def test_scenario_credit_stress_exists(self):
        assert (TABLES / "scenario_credit_stress_summary.csv").exists()

    def test_severity_lgd_exists(self):
        assert (TABLES / "severity_lgd_table.csv").exists()

    def test_final_assumptions_ledger_exists(self):
        assert (TABLES / "final_assumptions_ledger.csv").exists()

    def test_material_model_performance_beats_baseline(self):
        df = pd.read_csv(TABLES / "material_model_performance.csv")
        density_row = df[df["target"] == "density_g_cm3"]
        assert not density_row.empty
        assert bool(density_row["beats_baseline"].iloc[0]) is True

    def test_failure_model_auc(self):
        df = pd.read_csv(TABLES / "failure_model_performance.csv")
        auc_row = df[df["model"] == "case-control failure classifier"]
        assert not auc_row.empty
        auc = float(auc_row["value"].iloc[0])
        assert auc > 0.70, f"Expected AUC > 0.70, got {auc}"

    def test_severity_lgd_sum_to_one(self):
        df = pd.read_csv(TABLES / "severity_lgd_table.csv")
        total = df["empirical_share"].sum()
        np.testing.assert_allclose(total, 1.0, atol=0.01, err_msg="Severity shares don't sum to 1")

    def test_commodity_var_all_positive(self):
        df = pd.read_csv(TABLES / "commodity_var_es.csv")
        assert (df["VaR95_1d"] > 0).all(), "VaR95 values should be positive"
        assert (df["ES95_1d"] >= df["VaR95_1d"]).all(), "ES should be >= VaR"


# ---------------------------------------------------------------------------
# Output figures
# ---------------------------------------------------------------------------

class TestFigures:

    EXPECTED_FIGURES = [
        "08A_shap_bulk_modulus_GPa.png",
        "08A_shap_shear_modulus_GPa.png",
        "08B_shap_density.png",
        "08C_shap_condition.png",
        "08D_shap_case_control.png",
        "08C_pdp_corrosion.png",
        "08D_pdp_and_overlap.png",
        "A1_pred_vs_actual_KG.png",
    ]

    @pytest.mark.parametrize("fname", EXPECTED_FIGURES)
    def test_figure_exists(self, fname):
        p = FIGURES / fname
        assert p.exists(), f"Expected figure not found: {p}"
