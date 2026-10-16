"""
pages/6_ESG_Inverse_Design.py
-------------------------------
ESG metrics and inverse material design.

STATUS: Partially implemented.
  - ESG metrics derived from DS2/DS4 features (carbon intensity, green premium,
    substitution elasticity, supply disruption risk, herfindahl index) ARE available
    and are displayed.
  - Inverse design (identify compositions with target K/G from DS1 model) is
    shown as a prototype search — find existing materials in DS1 predictions
    that satisfy user-specified property targets. This is a lookup/filter, not
    generative design.
  - Generative inverse design (propose new compositions) is NOT implemented.
    A clear "Not yet implemented" message is shown.

ENGINEERING HONESTY: No fake ESG scores are fabricated. All values come from
engineered features in the DS2/DS4 processed file.
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
    footer, not_implemented, PALETTE, PLOTLY_TEMPLATE,
)
from dashboard.data_loader import load_financial_features, load_model_predictions, load_material_features

st.set_page_config(page_title="ESG & Design · MatRisk", layout="wide", page_icon="🌿")
apply_global_css()

hero_banner("🌿 ESG & Inverse Material Design", "ESG signals from DS4 · Property-target material lookup · Notebook 03/08")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with st.spinner("Loading data..."):
    try:
        fin = load_financial_features()
        preds = load_model_predictions()
        mat_feat = load_material_features()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

# ---------------------------------------------------------------------------
# Status banners
# ---------------------------------------------------------------------------
info_box(
    "<strong>ESG signals</strong> (carbon intensity, green premium, substitution elasticity, "
    "supply disruption, Herfindahl index) are displayed from DS4 engineered features. "
    "<strong>Generative inverse design is not implemented</strong> — shown as 'not yet' below."
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_esg, tab_inv, tab_supply = st.tabs([
    "🌍 ESG Signals", "🔬 Inverse Design (Lookup)", "⛏️ Supply Risk"
])

# ── Tab 1: ESG signals ────────────────────────────────────────────────────
with tab_esg:
    st.markdown("#### Material ESG Signals (DS4 Engineered Features, Notebook 03)")
    info_box(
        "These are <strong>engineered features</strong> from the DS2+DS4 feature store — "
        "they represent modelled/estimated values, not certified ESG ratings. "
        "Carbon intensity values are from the DS4 dataset (virgin vs. recycled production routes)."
    )

    COMMODITIES = sorted(fin["commodity"].unique())
    # Get latest snapshot per commodity
    latest = fin.sort_values("date").groupby("commodity").last().reset_index()

    esg_cols = {
        "carbon_intensity_virgin": "Carbon Intensity (Virgin) [relative]",
        "carbon_intensity_recycled": "Carbon Intensity (Recycled) [relative]",
        "green_premium_per_kg": "Green Premium (USD/kg)",
        "substitution_elasticity": "Substitution Elasticity",
    }

    available_esg = {k: v for k, v in esg_cols.items() if k in latest.columns}

    if not available_esg:
        st.warning("ESG columns not found in the financial feature store.")
    else:
        col_a, col_b = st.columns(2)
        items = list(available_esg.items())

        for i, (col, label) in enumerate(items):
            target_col = col_a if i % 2 == 0 else col_b
            with target_col:
                fig = px.bar(
                    latest.sort_values(col),
                    x="commodity", y=col,
                    title=label,
                    template=PLOTLY_TEMPLATE, height=320,
                    color=col,
                    color_continuous_scale=[[0, PALETTE["success"]], [1, PALETTE["accent"]]],
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", color=PALETTE["text"]),
                    title_font=dict(size=12, color=PALETTE["primary"]),
                    margin=dict(l=10, r=10, t=45, b=20), coloraxis_showscale=False,
                    xaxis_tickangle=-30,
                )
                st.plotly_chart(fig, width='stretch')

    # Carbon intensity virgin vs recycled comparison
    if "carbon_intensity_virgin" in latest.columns and "carbon_intensity_recycled" in latest.columns:
        st.markdown("#### Virgin vs. Recycled Carbon Intensity")
        fig_ci = go.Figure()
        fig_ci.add_trace(go.Bar(
            name="Virgin", x=latest["commodity"], y=latest["carbon_intensity_virgin"],
            marker_color=PALETTE["accent"],
        ))
        fig_ci.add_trace(go.Bar(
            name="Recycled", x=latest["commodity"], y=latest["carbon_intensity_recycled"],
            marker_color=PALETTE["success"],
        ))
        fig_ci.update_layout(
            barmode="group", title="Carbon Intensity: Virgin vs. Recycled Production",
            template=PLOTLY_TEMPLATE, height=380,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color=PALETTE["text"]),
            title_font=dict(size=14, color=PALETTE["primary"]),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_ci, width='stretch')

    warning_box(
        "These ESG values are <strong>engineered estimates from DS4</strong>, not certified "
        "third-party ESG ratings. Do not use them as regulatory or investment-grade ESG scores."
    )

# ── Tab 2: Inverse Design (Lookup) ───────────────────────────────────────
with tab_inv:
    st.markdown("#### Inverse Design — Property-Target Material Lookup")
    info_box(
        "This is a <strong>filter / lookup</strong> over the DS1 dataset using the model's "
        "OOF predictions. Enter target property ranges and retrieve existing DS1 materials "
        "that meet the criteria. This is <strong>not generative inverse design</strong>."
    )

    col1, col2 = st.columns(2)
    with col1:
        k_min = st.number_input("Min Bulk Modulus K (GPa)", 0.0, 500.0, 100.0, 10.0)
        k_max = st.number_input("Max Bulk Modulus K (GPa)", 0.0, 1000.0, 200.0, 10.0)
        g_min = st.number_input("Min Shear Modulus G (GPa)", 0.0, 400.0, 50.0, 10.0)
        g_max = st.number_input("Max Shear Modulus G (GPa)", 0.0, 800.0, 150.0, 10.0)
    with col2:
        cat_filter = st.multiselect(
            "Filter by category (optional)",
            sorted(preds["category"].dropna().unique()),
            default=[],
        )
        sort_by = st.selectbox(
            "Sort by",
            ["bulk_modulus_GPa_actual", "shear_modulus_GPa_actual", "density_g_cm3_actual"],
            format_func=lambda c: c.replace("_actual", "").replace("_", " ").title(),
        )

    mask = (
        (preds["bulk_modulus_GPa_actual"] >= k_min) &
        (preds["bulk_modulus_GPa_actual"] <= k_max) &
        (preds["shear_modulus_GPa_actual"] >= g_min) &
        (preds["shear_modulus_GPa_actual"] <= g_max)
    )
    if cat_filter:
        mask &= preds["category"].isin(cat_filter)

    results = preds[mask].copy().sort_values(sort_by, ascending=False)

    st.markdown(f"**{len(results)} materials found** matching the criteria (using DS1 actual values).")

    if len(results) > 0:
        show_cols = ["material_id", "formula", "category",
                     "bulk_modulus_GPa_actual", "shear_modulus_GPa_actual",
                     "density_g_cm3_actual", "K_pred_final", "G_pred_final",
                     "nu_derived", "E_derived_GPa"]
        disp = results[show_cols].head(100).rename(columns={
            "bulk_modulus_GPa_actual": "K_actual", "shear_modulus_GPa_actual": "G_actual",
            "density_g_cm3_actual": "ρ_actual", "K_pred_final": "K_pred",
            "G_pred_final": "G_pred", "nu_derived": "ν", "E_derived_GPa": "E (GPa)",
        })
        st.dataframe(disp, width='stretch', hide_index=True)

        # Scatter K vs G coloured by category
        fig_inv = px.scatter(
            results.head(300),
            x="bulk_modulus_GPa_actual", y="shear_modulus_GPa_actual",
            color="category",
            hover_data=["material_id", "formula", "density_g_cm3_actual"],
            title="Matching Materials: K vs. G (actual DS1 values)",
            template=PLOTLY_TEMPLATE, height=420,
        )
        fig_inv.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color=PALETTE["text"]),
            title_font=dict(size=13, color=PALETTE["primary"]),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_inv, width='stretch')
    else:
        st.info("No materials match the specified criteria. Try widening the property ranges.")

    st.markdown("---")
    not_implemented(
        "Generative Inverse Design",
        "Proposing novel chemical compositions that satisfy target properties requires "
        "a generative model (VAE, GAN, or structure-property surrogate with optimisation). "
        "The necessary generative model or inverse surrogate has not been trained. "
        "This feature will be added in a future version."
    )

# ── Tab 3: Supply risk ─────────────────────────────────────────────────────
with tab_supply:
    st.markdown("#### Supply Risk Indicators (DS4 Features)")
    info_box(
        "Supply disruption probability and Herfindahl-Hirschman Index (HHI) are "
        "cross-domain risk features from DS4 — they represent estimated supply-chain "
        "concentration and disruption exposure."
    )

    latest_sup = fin.sort_values("date").groupby("commodity").last().reset_index()

    sup_cols = ["supply_disruption_prob", "herfindahl_index"]
    avail_sup = [c for c in sup_cols if c in latest_sup.columns]

    if avail_sup:
        for col in avail_sup:
            fig_sup = px.bar(
                latest_sup.sort_values(col, ascending=False),
                x="commodity", y=col,
                title=col.replace("_", " ").title(),
                template=PLOTLY_TEMPLATE, height=320,
                color=col,
                color_continuous_scale=[[0, PALETTE["success"]], [1, PALETTE["accent"]]],
            )
            fig_sup.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color=PALETTE["text"]),
                title_font=dict(size=13, color=PALETTE["primary"]),
                margin=dict(l=20, r=20, t=45, b=20), coloraxis_showscale=False,
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig_sup, width='stretch')
    else:
        st.info("Supply risk columns not found.")

footer()
