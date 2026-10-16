"""
dashboard/data_loader.py
------------------------
Centralised, cached data-loading layer for the MatRisk AI dashboard.

All loaders use st.cache_data or st.cache_resource so that files are
read from disk exactly once per Streamlit session.  Paths are resolved
relative to PROJECT_ROOT, which is auto-detected at import time.

IMPORTANT: This module never fabricates data.  If a file is missing it
raises a descriptive FileNotFoundError rather than substituting a dummy.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    """Walk up from this file looking for a directory that contains both
    'src/' and 'data/' subdirectories."""
    candidate = Path(__file__).resolve().parent
    for _ in range(6):
        if (candidate / "src").is_dir() and (candidate / "data").is_dir():
            return candidate
        candidate = candidate.parent
    raise RuntimeError(
        f"Cannot locate project root (directory with both 'src/' and 'data/')."
        f"  Started from: {Path(__file__).resolve()}"
    )

PROJECT_ROOT: Path = _find_project_root()
PROCESSED: Path = PROJECT_ROOT / "data" / "processed"
MODELS: Path = PROCESSED / "models"
TABLES: Path = PROJECT_ROOT / "outputs" / "tables"
FIGURES: Path = PROJECT_ROOT / "outputs" / "figures"
RAW_DATA: Path = Path(__file__).resolve().parent.parent.parent / "totaldataset"


# ---------------------------------------------------------------------------
# Add src/ to Python path so that src.data.audit can be imported
# ---------------------------------------------------------------------------
_src = PROJECT_ROOT / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require(path: Path) -> Path:
    """Assert path exists and return it; raise a clear error if not."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            f"  (PROJECT_ROOT={PROJECT_ROOT})"
        )
    return path


# ---------------------------------------------------------------------------
# Processed CSVs
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_material_features() -> pd.DataFrame:
    """DS1 material features (5 500 rows × 42 cols) used by notebook 04."""
    return pd.read_csv(_require(PROCESSED / "ds1_material_features.csv"))


@st.cache_data(show_spinner=False)
def load_model_predictions() -> pd.DataFrame:
    """DS1 out-of-fold model predictions from notebook 04."""
    return pd.read_csv(_require(PROCESSED / "ds1_model_predictions.csv"))


@st.cache_data(show_spinner=False)
def load_financial_features() -> pd.DataFrame:
    """DS2+DS4 financial feature store (22 952 rows × 55 cols).
    'date' column is parsed as datetime.
    """
    df = pd.read_csv(_require(PROCESSED / "ds2_ds4_financial_features.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_credit_risk() -> pd.DataFrame:
    """DS3 credit risk table (5 000 bridges × 15 cols): PD, LGD, EAD, EL."""
    return pd.read_csv(_require(PROCESSED / "ds3_credit_risk.csv"))


@st.cache_data(show_spinner=False)
def load_risk_scores() -> pd.DataFrame:
    """DS3 risk score table (5 000 bridges × 11 cols)."""
    return pd.read_csv(_require(PROCESSED / "ds3_risk_scores.csv"))


@st.cache_data(show_spinner=False)
def load_exposure_features() -> pd.DataFrame:
    """DS3 full exposure feature table (5 000 rows × 35 cols)."""
    return pd.read_csv(_require(PROCESSED / "ds3_exposure_features.csv"))


@st.cache_data(show_spinner=False)
def load_failure_features() -> pd.DataFrame:
    """DS6 historical failure records (2 000 events × 22 cols)."""
    return pd.read_csv(_require(PROCESSED / "ds6_failure_features.csv"))


@st.cache_data(show_spinner=False)
def load_failure_case_control() -> pd.DataFrame:
    """DS6 case-control dataset (6 011 rows × 8 cols) used for classifier."""
    return pd.read_csv(_require(PROCESSED / "ds6_failure_case_control.csv"))


@st.cache_data(show_spinner=False)
def load_element_price_features() -> pd.DataFrame:
    """DS5 element price feature table."""
    return pd.read_csv(_require(PROCESSED / "ds5_element_price_features.csv"))


# ---------------------------------------------------------------------------
# Output tables
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_table(filename: str) -> pd.DataFrame:
    """Generic loader for any CSV in outputs/tables/."""
    return pd.read_csv(_require(TABLES / filename))


@st.cache_data(show_spinner=False)
def load_material_model_performance() -> pd.DataFrame:
    return load_table("material_model_performance.csv")


@st.cache_data(show_spinner=False)
def load_failure_model_performance() -> pd.DataFrame:
    return load_table("failure_model_performance.csv")


@st.cache_data(show_spinner=False)
def load_commodity_var_es() -> pd.DataFrame:
    return load_table("commodity_var_es.csv")


@st.cache_data(show_spinner=False)
def load_shap_importance() -> pd.DataFrame:
    return load_table("shap_feature_importance.csv")


@st.cache_data(show_spinner=False)
def load_scenario_price_shock() -> pd.DataFrame:
    return load_table("scenario_price_shock_summary.csv")


@st.cache_data(show_spinner=False)
def load_scenario_credit_stress() -> pd.DataFrame:
    return load_table("scenario_credit_stress_summary.csv")


@st.cache_data(show_spinner=False)
def load_severity_lgd() -> pd.DataFrame:
    return load_table("severity_lgd_table.csv")


@st.cache_data(show_spinner=False)
def load_financial_risk_engine_summary() -> pd.DataFrame:
    return load_table("financial_risk_engine_summary.csv")


@st.cache_data(show_spinner=False)
def load_credit_mc_summary() -> pd.DataFrame:
    return load_table("credit_monte_carlo_summary.csv")


@st.cache_data(show_spinner=False)
def load_assumptions_ledger() -> pd.DataFrame:
    return load_table("final_assumptions_ledger.csv")


@st.cache_data(show_spinner=False)
def load_final_scorecard() -> pd.DataFrame:
    return load_table("final_scorecard.csv")


@st.cache_data(show_spinner=False)
def load_final_signoff() -> pd.DataFrame:
    return load_table("final_signoff.csv")


@st.cache_data(show_spinner=False)
def load_reference_validation() -> pd.DataFrame:
    return load_table("A6_reference_material_validation.csv")


# ---------------------------------------------------------------------------
# Trained models  (st.cache_resource – objects not serialisable by arrow)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_bulk_modulus_model():
    """HistGradientBoostingRegressor for DS1 bulk modulus (GPa)."""
    return joblib.load(_require(MODELS / "ds1_bulk_modulus_GPa_model.joblib"))


@st.cache_resource(show_spinner=False)
def load_shear_modulus_model():
    """HistGradientBoostingRegressor for DS1 shear modulus (GPa)."""
    return joblib.load(_require(MODELS / "ds1_shear_modulus_GPa_model.joblib"))


@st.cache_resource(show_spinner=False)
def load_condition_rating_model():
    """HistGradientBoostingRegressor for DS3 bridge condition rating."""
    return joblib.load(_require(MODELS / "ds3_condition_rating_model.joblib"))


@st.cache_resource(show_spinner=False)
def load_case_control_model():
    """HistGradientBoostingClassifier for DS6 failure case-control."""
    return joblib.load(_require(MODELS / "ds6_case_control_model.joblib"))


@st.cache_resource(show_spinner=False)
def load_ds1_feature_snapshot() -> pd.DataFrame:
    """Reference feature snapshot for DS1 models (used for category means)."""
    return joblib.load(_require(MODELS / "ds1_feature_snapshot.joblib"))


# ---------------------------------------------------------------------------
# Figure paths  (returned as Path objects; callers use st.image)
# ---------------------------------------------------------------------------

def figure_path(filename: str) -> Optional[Path]:
    """Return the absolute Path to a saved figure, or None if not found."""
    p = FIGURES / filename
    return p if p.exists() else None


def list_figures() -> list[str]:
    """List all .png filenames in outputs/figures/."""
    if FIGURES.exists():
        return sorted(f.name for f in FIGURES.glob("*.png"))
    return []


# ---------------------------------------------------------------------------
# PD calibration constants  (documented in notebook 06 / NOTES_for_handoff.md)
# ---------------------------------------------------------------------------

# Known population rate: 1,011 DS6 bridge failures / 19 years / 5,000 bridges
TRUE_ANNUAL_PD: float = 1011 / (19 * 5000)   # ≈ 0.01064 = 1.064 %/yr

# Severity LGD mapping (from notebook 06 / severity_lgd_table.csv)
SEVERITY_LGD: Dict[str, float] = {
    "Minor": 0.0495,
    "Moderate": 0.2003,
    "Major": 0.5019,
    "Catastrophic": 0.8657,   # capped mean
}
