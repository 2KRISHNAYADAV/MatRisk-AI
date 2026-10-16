"""
pages/7_MatRisk_Lab.py
-----------------------
MatRisk Lab — Interactive Simulation Environment

STATUS: Architecture prepared. Full simulator not yet implemented.

This page provides:
  - The dashboard layout and navigation for the planned simulator.
  - A single working demonstration: run the condition-rating model and
    case-control model interactively to explore parameter sensitivity
    (this is a manual sensitivity analysis, NOT a validated simulator).
  - Clear "not yet implemented" stubs for:
    - Monte Carlo simulator with user-defined distribution inputs
    - Agent-based degradation simulator
    - Portfolio optimiser
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.components import (
    apply_global_css, hero_banner, warning_box, info_box,
    footer, not_implemented, PALETTE, PLOTLY_TEMPLATE,
)
from dashboard.data_loader import (
    load_condition_rating_model, load_case_control_model,
    load_exposure_features, TRUE_ANNUAL_PD,
)

st.set_page_config(page_title="MatRisk Lab · MatRisk", layout="wide", page_icon="🧪")
apply_global_css()

hero_banner("🧪 MatRisk Lab", "Parameter sensitivity · Architecture ready · Simulator pending implementation")

# ---------------------------------------------------------------------------
# Status banner
# ---------------------------------------------------------------------------
warning_box(
    "<strong>This page is under construction.</strong> "
    "The full interactive simulator is NOT yet functional. "
    "What IS working: a manual sensitivity analysis using the actual trained models. "
    "All other simulator features show a clear 'Not yet implemented' message."
)

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
with st.spinner("Loading models..."):
    try:
        model_cond = load_condition_rating_model()
        model_cc = load_case_control_model()
        exposure = load_exposure_features()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_sens, tab_mc, tab_agent, tab_opt = st.tabs([
    "🎛️ Parameter Sensitivity (Live)", "🎲 MC Simulator (Planned)", 
    "🤖 Degradation Agent (Planned)", "📐 Portfolio Optimiser (Planned)"
])

# ── Tab 1: Parameter Sensitivity (WORKING) ───────────────────────────────
with tab_sens:
    st.markdown("#### Live Parameter Sensitivity — Actual Trained Models")
    info_box(
        "This tool runs the actual fitted condition-rating regressor (DS3) and "
        "case-control failure classifier (DS6/DS3) with your chosen parameter values. "
        "Vary corrosion rate and age to see how condition rating and failure probability respond. "
        "PD is calibrated to the known population rate (1.064%/yr)."
    )

    MATERIALS = sorted(exposure["material"].dropna().unique())
    LOCATIONS = sorted(exposure["location"].dropna().unique())
    CORR_ENVS = sorted(exposure["corrosion_environment"].dropna().unique())
    BRIDGE_TYPES = sorted(exposure["bridge_type"].dropna().unique())

    col_inp, col_res = st.columns([1, 1])

    with col_inp:
        st.markdown("##### Asset Parameters")
        material = st.selectbox("Material", MATERIALS)
        bridge_type = st.selectbox("Bridge type", BRIDGE_TYPES)
        location = st.selectbox("Location", LOCATIONS)
        corr_env = st.selectbox("Corrosion environment", CORR_ENVS)
        is_metal = st.checkbox("Is metal?", value=True)
        age_years = st.slider("Age (years)", 1, 120, 40)
        design_life = st.slider("Design life (years)", 50, 150, 100)
        corrosion_rate = st.slider("Corrosion rate (mm/yr)", 0.001, 0.5, 0.05, 0.001, format="%.3f")
        original_thickness = st.number_input("Original thickness (mm)", 10.0, 100.0, 40.0, 1.0)
        adt = st.number_input("Average daily traffic (ADT)", 1000, 100000, 10000, 1000)
        tensile_strength = st.number_input("Tensile strength (MPa)", 100.0, 2000.0, 400.0, 10.0)
        yield_strength = st.number_input("Yield strength (MPa)", 50.0, 1500.0, 300.0, 10.0)

    with col_res:
        st.markdown("##### Model Predictions")

        # Derived features
        age_ratio = age_years / design_life
        remaining_thickness = max(original_thickness - corrosion_rate * age_years, 0.0)
        thick_frac = remaining_thickness / original_thickness if original_thickness > 0 else 0
        log_corr = np.log(max(corrosion_rate, 1e-6))

        # Condition rating
        row_cond = pd.DataFrame([{
            "material": material, "bridge_type": bridge_type,
            "location": location, "corrosion_environment": corr_env,
            "is_metal": is_metal, "age_years": age_years, "design_life_years": design_life,
            "age_ratio": age_ratio, "corrosion_rate_mm_yr": corrosion_rate,
            "log_corr": log_corr, "thick_frac": thick_frac,
            "adt": adt, "tensile_strength_MPa": tensile_strength,
            "yield_strength_MPa": yield_strength,
        }])

        # Case-control
        row_cc = pd.DataFrame([{
            "material": material, "corrosion_environment": corr_env,
            "location": location, "age_years": age_years,
            "corrosion_rate_mm_yr": corrosion_rate, "is_metal": is_metal,
        }])

        try:
            cond_pred = float(model_cond.predict(row_cond)[0])
            cond_pred = max(1.0, min(9.0, cond_pred))

            cc_score = float(model_cc.predict_proba(row_cc)[0][1])
            # Calibrate to population PD
            pop_mean_score = float(model_cc.predict_proba(
                pd.DataFrame(exposure[row_cc.columns].head(1000))
            )[:, 1].mean())
            pd_annual = cc_score / pop_mean_score * TRUE_ANNUAL_PD if pop_mean_score > 0 else TRUE_ANNUAL_PD
            pd_annual = max(0.0, min(1.0, pd_annual))

            # Risk colour
            if cond_pred >= 7:
                cond_color = PALETTE["success"]
            elif cond_pred >= 5:
                cond_color = PALETTE["warning"]
            else:
                cond_color = PALETTE["accent"]

            st.markdown(
                f"""
                <div style="background:{PALETTE['bg_card']};border:1px solid #30363D;
                            border-radius:12px;padding:1.5rem;margin-bottom:1rem;">
                    <div style="font-size:0.85rem;color:{PALETTE['muted']};">Condition Rating (1–9)</div>
                    <div style="font-size:2.5rem;font-weight:700;color:{cond_color};">{cond_pred:.2f}</div>
                    <div style="font-size:0.8rem;color:{PALETTE['muted']};">
                        {'Good (≥7)' if cond_pred >= 7 else 'Fair (5–6)' if cond_pred >= 5 else 'Poor (<5)'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.metric("Annual PD (calibrated)", f"{pd_annual*100:.3f}%",
                      help="Case-control score calibrated to population rate 1.064%/yr")
            st.metric("Derived Remaining Thickness", f"{thick_frac*100:.1f}% of original",
                      help=f"remaining = {remaining_thickness:.1f} mm")
            st.metric("Age / Design Life", f"{age_ratio*100:.1f}%")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.exception(e)

    # Sensitivity sweep: corrosion rate vs condition rating
    st.markdown("---")
    st.markdown("#### Sensitivity Sweep — Corrosion Rate vs. Condition Rating & PD")

    corr_range = np.linspace(0.001, 0.4, 60)
    preds_sweep = []
    pd_sweep = []

    # Precompute population mean score for calibration
    try:
        pop_scores = model_cc.predict_proba(
            exposure[row_cc.columns].dropna().head(500)
        )[:, 1]
        pop_mean = pop_scores.mean()
    except Exception:
        pop_mean = 0.1

    for cr in corr_range:
        rem = max(original_thickness - cr * age_years, 0.0)
        tf = rem / original_thickness if original_thickness > 0 else 0
        lr = np.log(max(cr, 1e-6))
        row = pd.DataFrame([{
            "material": material, "bridge_type": bridge_type,
            "location": location, "corrosion_environment": corr_env,
            "is_metal": is_metal, "age_years": age_years, "design_life_years": design_life,
            "age_ratio": age_ratio, "corrosion_rate_mm_yr": cr,
            "log_corr": lr, "thick_frac": tf,
            "adt": adt, "tensile_strength_MPa": tensile_strength,
            "yield_strength_MPa": yield_strength,
        }])
        row_cc_sw = pd.DataFrame([{
            "material": material, "corrosion_environment": corr_env,
            "location": location, "age_years": age_years,
            "corrosion_rate_mm_yr": cr, "is_metal": is_metal,
        }])
        try:
            cond = float(model_cond.predict(row)[0])
            preds_sweep.append(max(1.0, min(9.0, cond)))
            cc_s = float(model_cc.predict_proba(row_cc_sw)[0][1])
            pd_s = cc_s / pop_mean * TRUE_ANNUAL_PD if pop_mean > 0 else TRUE_ANNUAL_PD
            pd_sweep.append(min(1.0, pd_s) * 100)
        except Exception:
            preds_sweep.append(np.nan)
            pd_sweep.append(np.nan)

    sweep_df = pd.DataFrame({
        "corrosion_rate_mm_yr": corr_range,
        "condition_rating_pred": preds_sweep,
        "pd_annual_pct": pd_sweep,
    })

    col_sw1, col_sw2 = st.columns(2)

    with col_sw1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=sweep_df["corrosion_rate_mm_yr"], y=sweep_df["condition_rating_pred"],
            mode="lines", line=dict(color=PALETTE["primary"], width=2.5),
            name="Condition Rating",
        ))
        fig1.add_vline(x=corrosion_rate, line_dash="dash", line_color=PALETTE["accent"],
                       annotation_text="Current")
        fig1.update_layout(
            title="Condition Rating vs. Corrosion Rate (monotonic by construction)",
            template=PLOTLY_TEMPLATE, height=350,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color=PALETTE["text"]),
            title_font=dict(size=12, color=PALETTE["primary"]),
            xaxis_title="Corrosion Rate (mm/yr)", yaxis_title="Predicted Condition (1–9)",
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig1, width='stretch')

    with col_sw2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=sweep_df["corrosion_rate_mm_yr"], y=sweep_df["pd_annual_pct"],
            mode="lines", line=dict(color=PALETTE["accent"], width=2.5),
            name="Annual PD (%)",
        ))
        fig2.add_vline(x=corrosion_rate, line_dash="dash", line_color=PALETTE["warning"],
                       annotation_text="Current")
        fig2.update_layout(
            title="Calibrated Annual PD (%) vs. Corrosion Rate (monotonic by construction)",
            template=PLOTLY_TEMPLATE, height=350,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color=PALETTE["text"]),
            title_font=dict(size=12, color=PALETTE["primary"]),
            xaxis_title="Corrosion Rate (mm/yr)", yaxis_title="Annual PD (%)",
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig2, width='stretch')

# ── Tab 2: MC Simulator (Planned) ────────────────────────────────────────
with tab_mc:
    not_implemented(
        "Monte Carlo Portfolio Simulator",
        "The planned simulator will allow user-defined distribution parameters "
        "(PD, LGD, asset correlations) for full credit portfolio loss distribution simulation. "
        "The fixed-distribution Monte Carlo in notebook 06 is the current state. "
        "This interactive version requires additional validation before deployment."
    )

# ── Tab 3: Agent (Planned) ────────────────────────────────────────────────
with tab_agent:
    not_implemented(
        "Agent-Based Degradation Simulator",
        "Asset-level degradation propagation over time, incorporating inspection schedules, "
        "maintenance interventions, and corrosion evolution models. "
        "Requires a validated time-series degradation model and inspection event data "
        "not currently in the dataset."
    )

# ── Tab 4: Optimiser (Planned) ────────────────────────────────────────────
with tab_opt:
    not_implemented(
        "Portfolio Risk Optimiser",
        "Optimise inspection/maintenance budget allocation across the bridge portfolio "
        "to minimise expected credit loss. "
        "Requires a cost-of-inspection/maintenance dataset and a validated causal "
        "maintenance→condition model — not yet available."
    )

footer()
