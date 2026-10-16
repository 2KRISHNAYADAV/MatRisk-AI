"""
pages/5_Explainability_SHAP.py
--------------------------------
Displays SHAP feature importance and saved partial-dependence plots
from notebook 08.

What is shown:
  - Saved SHAP summary PNG figures (from outputs/figures/)
  - Interactive SHAP importance bar chart from shap_feature_importance.csv
  - Saved partial-dependence plot figures
  - Plain-language model explanations

What is NOT claimed:
  - Feature importance ≠ causation.
  - SHAP values explain the model's predictions, not the underlying physical mechanism.
  - The density model's reliance on density_implied_g_cm3 is a backwards correlation
    (cell volume predicts density because both are derived from the same crystal structure),
    not a new physical insight.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.components import (
    apply_global_css, hero_banner, warning_box, info_box,
    footer, PALETTE, PLOTLY_TEMPLATE, show_figure_or_missing,
)
from dashboard.data_loader import load_shap_importance

st.set_page_config(page_title="Explainability · MatRisk", layout="wide", page_icon="🔍")
apply_global_css()

hero_banner("🔍 Explainability (SHAP)", "Feature importance · Partial dependence · Model explanation · Notebook 08")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with st.spinner("Loading SHAP data..."):
    try:
        shap_df = load_shap_importance()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

# ---------------------------------------------------------------------------
# Caveat banner
# ---------------------------------------------------------------------------
warning_box(
    "<strong>Feature importance ≠ causation.</strong> "
    "SHAP values quantify each feature's contribution to the <em>model's</em> predictions — "
    "they do not prove that a feature physically causes the property. "
    "Importantly, the density model's top feature (<code>density_implied_g_cm3</code>) is a "
    "physically backwards correlation: cell volume and density are derived from the same crystal "
    "structure, so the model is essentially using one side of an identity to predict the other."
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_bar, tab_figures, tab_pdp, tab_explain = st.tabs([
    "📊 SHAP Importance Bar", "🖼️ SHAP Summary Plots", "📈 Partial Dependence", "💬 Plain Language"
])

# ── Tab 1: Interactive SHAP bar chart ─────────────────────────────────────
with tab_bar:
    st.markdown("#### SHAP Feature Importance (Mean |SHAP| from Notebook 08)")

    MODELS = sorted(shap_df["model"].unique())
    model_sel = st.selectbox(
        "Select model",
        MODELS,
        format_func=lambda m: m.replace("_", " ").title(),
    )

    sub = shap_df[shap_df["model"] == model_sel].sort_values("mean_abs_shap", ascending=True)

    fig = go.Figure(go.Bar(
        x=sub["mean_abs_shap"],
        y=sub["feature"],
        orientation="h",
        marker=dict(
            color=sub["mean_abs_shap"],
            colorscale=[[0, PALETTE["secondary"]], [1, PALETTE["primary"]]],
            showscale=False,
        ),
        text=sub["mean_abs_shap"].map("{:.4f}".format),
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Mean |SHAP| — {model_sel.replace('_', ' ').title()}",
        template=PLOTLY_TEMPLATE, height=max(350, len(sub) * 35 + 80),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=PALETTE["text"]),
        title_font=dict(size=14, color=PALETTE["primary"]),
        xaxis_title="Mean |SHAP|", yaxis_title="Feature",
        margin=dict(l=20, r=60, t=50, b=20),
    )
    st.plotly_chart(fig, width='stretch')

    # Key insight per model
    insights = {
        "bulk_modulus_GPa": (
            "**`category`** dominates (68.6 / ~103 total = **66%** of total |SHAP|). "
            "The composition features (mean_atomic_mass, volume_A3) add secondary signal. "
            "This confirms that the model is largely a category-mean smoother."
        ),
        "shear_modulus_GPa": (
            "**`category`** again dominates (0.597 / ~1.06 total = **56%** of |SHAP|). "
            "Similar story to K: the composition features add modest signal around the category mean."
        ),
        "density_g_cm3": (
            "**`density_implied_g_cm3`** is the top feature (1.53), but this is a **backwards correlation**: "
            "density_implied comes from the same unit-cell identity as the target. "
            "The model uses crystal geometry to recover density — physically valid, but not a "
            "new predictive discovery."
        ),
        "condition_rating": (
            "**`corrosion_rate_mm_yr`** (0.718) and **`age_years`** (0.365) dominate, "
            "confirming that condition rating is close to a 2-variable function of these two. "
            "The monotonic constraint on corrosion_rate was necessary and is confirmed here."
        ),
        "case_control_failure": (
            "**`corrosion_rate_mm_yr`** (3.124) and **`material`** (0.716) dominate. "
            "The monotonic constraint on corrosion_rate_mm_yr (increasing failure risk with "
            "increasing corrosion) is confirmed by partial dependence plots."
        ),
    }
    if model_sel in insights:
        info_box(insights[model_sel])

# ── Tab 2: Saved SHAP PNG figures ─────────────────────────────────────────
with tab_figures:
    st.markdown("#### Saved SHAP Summary Plots (from Notebook 08)")

    col_a, col_b = st.columns(2)
    with col_a:
        show_figure_or_missing("08A_shap_bulk_modulus_GPa.png", "SHAP — Bulk Modulus (K)")
    with col_b:
        show_figure_or_missing("08A_shap_shear_modulus_GPa.png", "SHAP — Shear Modulus (G)")

    col_c, col_d = st.columns(2)
    with col_c:
        show_figure_or_missing("08B_shap_density.png", "SHAP — Density")
    with col_d:
        show_figure_or_missing("08C_shap_condition.png", "SHAP — Condition Rating")

    show_figure_or_missing("08D_shap_case_control.png", "SHAP — Case-Control Failure Classifier")

# ── Tab 3: Partial Dependence ─────────────────────────────────────────────
with tab_pdp:
    st.markdown("#### Partial Dependence Plots (from Notebook 08)")
    info_box(
        "PDPs confirm that the monotonic constraints applied in notebook 05 "
        "actually produce monotonic responses — i.e. higher corrosion rate always predicts "
        "lower condition rating and higher failure probability."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        show_figure_or_missing("08C_pdp_corrosion.png",
                               "PDP: Corrosion Rate vs Condition Rating (monotonic confirmed)")
    with col_b:
        show_figure_or_missing("08D_pdp_and_overlap.png",
                               "PDP: Age and corrosion + overlap region")

    show_figure_or_missing("08B_density_volume_dependence.png",
                           "Density model: volume dependence (backwards correlation explained)")

# ── Tab 4: Plain language ─────────────────────────────────────────────────
with tab_explain:
    st.markdown("#### Plain-Language Model Explanations")

    explanations = {
        "Bulk & Shear Modulus Models (K, G)": """
**What the model does:** Predicts the elastic stiffness of a crystal given its composition,
crystal system, and unit-cell geometry.

**Why category matters so much:** Materials in the same category (e.g. "Metal", "Oxide") 
share similar bonding character, which strongly determines stiffness. The model essentially 
anchors its prediction at the category mean, then makes small adjustments based on composition details.

**Honest limitation:** The model does **not** beat the simple category-mean baseline on bulk or 
shear modulus (R² OOF: 0.44 vs baseline 0.51 for K). Use the category mean as your primary estimate; 
the model adds noise relative to that baseline for K and G.
        """,
        "Condition Rating Model (DS3)": """
**What the model does:** Predicts the FHWA condition rating (1–9 scale) of a bridge from its 
age, corrosion rate, material, and structural features.

**Key insight:** Two variables — `age_years` and `corrosion_rate_mm_yr` — explain nearly as 
much as the full feature set (2-var R² = 0.810 vs full R² = 0.783). Adding more features 
slightly *reduces* OOF R² — the additional features are not informative beyond these two.

**Monotonicity:** A monotonic constraint ensures the model always predicts lower condition 
ratings as corrosion rate increases. This is physically required and is confirmed by the PDP.
        """,
        "Case-Control Failure Classifier (DS6)": """
**What the model does:** Distinguishes bridges that experienced a failure event (DS6) from 
bridges in the general population (DS3). Outputs a failure risk score.

**AUC = 0.74 vs baseline 0.60:** The model beats the single-feature (corrosion-rate) baseline 
by 0.14 AUC points. This is modest but meaningful improvement.

**PD calibration:** The raw score is NOT a probability — it was trained on an artificial 
50/50 mix. It is calibrated so the portfolio mean PD matches the known failure rate 
(1.064%/yr from DS6). Individual asset PDs are relative estimates, not absolute probabilities.

**Monotonicity fix:** The original model (without monotonic constraint) predicted *lower* 
failure probability for *higher* corrosion rates for 45.6% of rows — a pure tree artifact 
that was caught and corrected in notebook 05/07.
        """,
    }

    for title, body in explanations.items():
        with st.expander(f"📘 {title}", expanded=False):
            st.markdown(body)

    warning_box(
        "None of the SHAP analysis above demonstrates causation. "
        "In particular, '`location` deterministically sets `corrosion_environment`' "
        "(found in notebook 08) means these two features carry redundant information — "
        "it is a data-encoding finding, not a physical law."
    )

footer()
