"""
pages/1_Material_Property_Explorer.py
--------------------------------------
Predict bulk and shear modulus for a user-supplied material composition
using the two trained HistGradientBoostingRegressor models from notebook 04.

What is real:
  - Both models are loaded from .joblib files — no retraining.
  - Category means (catmean_*) come from the feature snapshot saved in 04.
  - R² and MAE are from the honest out-of-fold evaluation in notebook 04.
  - Derived quantities (E, ν) use the isotropic identity from src.data.audit.

What is NOT claimed:
  - These models beat the category-mean baseline on K and G (they do not —
    see material_model_performance.csv).
  - Predictions are only valid within the feature distribution seen in DS1.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.components import (
    apply_global_css, hero_banner, warning_box, info_box,
    footer, plotly_scatter, plotly_bar, PALETTE, show_figure_or_missing,
)
from dashboard.data_loader import (
    load_bulk_modulus_model, load_shear_modulus_model,
    load_ds1_feature_snapshot, load_material_model_performance,
    load_model_predictions, PROJECT_ROOT,
)

st.set_page_config(page_title="Material Property Explorer · MatRisk", layout="wide", page_icon="⚗️")
apply_global_css()

hero_banner("⚗️ Material Property Explorer", "Live predictions via trained DS1 models · Notebook 04")

# ---------------------------------------------------------------------------
# Load assets (cached)
# ---------------------------------------------------------------------------
with st.spinner("Loading models and data..."):
    try:
        model_K = load_bulk_modulus_model()
        model_G = load_shear_modulus_model()
        snap = load_ds1_feature_snapshot()
        perf = load_material_model_performance()
        preds = load_model_predictions()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

# ---------------------------------------------------------------------------
# Model-performance callout
# ---------------------------------------------------------------------------
st.markdown("### 📊 Model Performance (Honest OOF Evaluation)")

warning_box(
    "The composition+category models beat the plain category-mean baseline on "
    "<strong>only 1 of 5 targets (density)</strong>. "
    "For bulk and shear modulus the baseline R² is higher. "
    "Predictions are shown to explore relative patterns — not as authoritative property estimates."
)

perf_display = perf.copy()
perf_display["Beats Baseline"] = perf_display["beats_baseline"].map(
    {True: "✅ Yes", False: "❌ No"}
)
perf_display = perf_display.rename(columns={
    "target": "Target", "model_R2_oof": "Model R² (OOF)",
    "catmean_baseline_R2": "Baseline R² (cat-mean)", "model_MAE": "MAE"
})
st.dataframe(
    perf_display[["Target", "Model R² (OOF)", "Baseline R² (cat-mean)", "MAE", "Beats Baseline"]],
    width='stretch', hide_index=True,
)

# ---------------------------------------------------------------------------
# Category / crystal system options
# ---------------------------------------------------------------------------
CATEGORIES = sorted(snap["category"].dropna().unique())
CRYSTAL_SYSTEMS = sorted(snap["crystal_system"].dropna().unique())

# Category means look-up table
cat_means = (
    snap.groupby("category")[
        ["catmean_bulk_modulus_GPa", "catmean_shear_modulus_GPa",
         "catmean_density_g_cm3", "catmean_melting_point_K",
         "catmean_formation_energy_per_atom_eV"]
    ].mean()
)

st.markdown("---")
st.markdown("### 🔬 Predict Properties for a New Material")
info_box(
    "Input ranges below are derived from the DS1 training distribution. "
    "Predictions outside these ranges extrapolate beyond the training data."
)

col_inp, col_out = st.columns([1, 1])

with col_inp:
    st.markdown("#### Input Features")

    cat = st.selectbox("Material category", CATEGORIES, index=CATEGORIES.index("Metal") if "Metal" in CATEGORIES else 0)
    cryst = st.selectbox("Crystal system", CRYSTAL_SYSTEMS)

    n_elements = st.slider("Number of elements (n_elements)", 1, 8, 3)
    nsites = st.slider("Number of sites in unit cell (nsites)", 1, 20, 4)
    volume_A3 = st.number_input("Unit cell volume (Å³)", min_value=5.0, max_value=3000.0, value=100.0, step=5.0)
    mean_atomic_mass = st.number_input("Mean atomic mass (amu)", min_value=1.0, max_value=240.0, value=60.0, step=1.0)
    mean_Z = st.number_input("Mean atomic number (mean_Z)", min_value=1.0, max_value=92.0, value=28.0, step=1.0)
    max_Z = st.number_input("Max atomic number (max_Z)", min_value=1.0, max_value=92.0, value=40.0, step=1.0)
    min_Z = st.number_input("Min atomic number (min_Z)", min_value=1.0, max_value=92.0, value=10.0, step=1.0)
    atoms_per_formula = st.number_input("Atoms per formula unit", min_value=1.0, max_value=50.0, value=4.0, step=1.0)

    # Derived implied density
    AMU_TO_G = 1.66053906660e-24
    density_implied = (nsites * mean_atomic_mass * AMU_TO_G) / (volume_A3 * 1e-24) if volume_A3 > 0 else np.nan

    st.markdown(f"*Implied density from cell parameters: **{density_implied:.3f} g/cm³***")

    has_cost = st.checkbox("Element cost data available?", value=False)
    element_cost = st.number_input(
        "Element cost (USD/kg)", min_value=0.0, max_value=100000.0, value=0.0, step=1.0,
        disabled=not has_cost,
    )

    # Get category means
    if cat in cat_means.index:
        cm = cat_means.loc[cat]
        catmean_K = float(cm["catmean_bulk_modulus_GPa"])
        catmean_G = float(cm["catmean_shear_modulus_GPa"])
    else:
        # fallback to overall mean
        catmean_K = float(snap["catmean_bulk_modulus_GPa"].mean())
        catmean_G = float(snap["catmean_shear_modulus_GPa"].mean())

    # Validate ranges
    valid = True
    if max_Z < min_Z:
        st.error("max_Z must be ≥ min_Z")
        valid = False

with col_out:
    st.markdown("#### Predictions")

    if valid:
        # Build feature row for K model
        row_K = pd.DataFrame([{
            "category": cat, "crystal_system": cryst, "n_elements": n_elements,
            "nsites": nsites, "volume_A3": volume_A3, "mean_atomic_mass": mean_atomic_mass,
            "mean_Z": mean_Z, "max_Z": max_Z, "min_Z": min_Z,
            "atoms_per_formula": atoms_per_formula,
            "density_implied_g_cm3": density_implied,
            "element_cost_usd_per_kg": element_cost if has_cost else np.nan,
            "element_cost_available": has_cost,
            "catmean_bulk_modulus_GPa": catmean_K,
        }])

        row_G = row_K.copy()
        row_G["catmean_shear_modulus_GPa"] = catmean_G
        row_G = row_G.drop(columns=["catmean_bulk_modulus_GPa"], errors="ignore")

        try:
            K_pred = float(model_K.predict(row_K)[0])
            G_pred = float(model_G.predict(row_G)[0])

            # Derived isotropic quantities
            nu_pred = (3 * K_pred - 2 * G_pred) / (2 * (3 * K_pred + G_pred)) if (3 * K_pred + G_pred) != 0 else np.nan
            E_pred = (9 * K_pred * G_pred) / (3 * K_pred + G_pred) if (3 * K_pred + G_pred) != 0 else np.nan
            pugh = K_pred / G_pred if G_pred != 0 else np.nan
            ductile = "Ductile (K/G > 1.75)" if pugh > 1.75 else "Brittle (K/G ≤ 1.75)"

            st.metric("Bulk Modulus K (GPa)", f"{K_pred:.1f}")
            st.metric("Shear Modulus G (GPa)", f"{G_pred:.1f}")
            st.metric("Derived Young's Modulus E (GPa)", f"{E_pred:.1f}" if not np.isnan(E_pred) else "N/A")
            st.metric("Derived Poisson's Ratio ν", f"{nu_pred:.3f}" if not np.isnan(nu_pred) else "N/A")
            st.metric("Pugh Ratio K/G", f"{pugh:.2f}" if not np.isnan(pugh) else "N/A",
                      delta=ductile)
            st.markdown(f"*Category-mean baseline K = {catmean_K:.1f} GPa, G = {catmean_G:.1f} GPa*")

            # Limitation callout
            warning_box(
                f"These are model predictions, <strong>not validated experimental values</strong>. "
                f"Category dominates SHAP importance (56–66% of |SHAP|). "
                f"Models were not validated against independent experimental literature except "
                f"for 5 reference materials in notebook 09."
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------------------------------------------------------------------
# Predicted vs Actual scatter (OOF)
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🎯 Predicted vs. Actual (Out-of-Fold, DS1 Training Set)")

target_choice = st.selectbox(
    "Select target",
    ["bulk_modulus_GPa", "shear_modulus_GPa", "density_g_cm3",
     "melting_point_K", "formation_energy_per_atom_eV"],
)

if target_choice in ["bulk_modulus_GPa", "shear_modulus_GPa", "density_g_cm3",
                      "melting_point_K", "formation_energy_per_atom_eV"]:
    actual_col = f"{target_choice}_actual"
    pred_col = f"{target_choice}_pred_oof"

    if actual_col in preds.columns and pred_col in preds.columns:
        plot_df = preds[[actual_col, pred_col, "category"]].dropna()
        fig = plotly_scatter(
            plot_df, x=actual_col, y=pred_col, color_col="category",
            title=f"OOF Predictions vs Actual — {target_choice}",
            hover_data=["category"], height=450, add_diagonal=True,
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("OOF prediction columns not found for this target.")

# ---------------------------------------------------------------------------
# Saved SHAP figure
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🔍 SHAP Feature Importance (Saved from Notebook 08)")

shap_fig_map = {
    "bulk_modulus_GPa": "08A_shap_bulk_modulus_GPa.png",
    "shear_modulus_GPa": "08A_shap_shear_modulus_GPa.png",
    "density_g_cm3": "08B_shap_density.png",
}

if target_choice in shap_fig_map:
    show_figure_or_missing(
        shap_fig_map[target_choice],
        f"SHAP summary plot for {target_choice} (generated in notebook 08)",
    )
else:
    st.info("No saved SHAP figure for this target.")

footer()
