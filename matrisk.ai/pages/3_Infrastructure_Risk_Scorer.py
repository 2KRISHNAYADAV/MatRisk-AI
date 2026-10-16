"""
pages/3_Infrastructure_Risk_Scorer.py
--------------------------------------
Displays asset-level and portfolio-level risk for the DS3 bridge portfolio.

PD CALIBRATION (documented in notebook 06 / NOTES_for_this_handoff.md):
  - The case-control classifier outputs a score trained on an artificial 16.8%-positive
    sample — NOT a real probability.
  - PD is calibrated by matching the population mean to the known rate:
    1,011 DS6 failures / 19 years / 5,000 bridges = 1.064%/yr
  - LGD is the empirical frequency-weighted mean from DS6 severity table.
  - EAD = loan_outstanding_M.
  - EL = PD × LGD × EAD.
  - These are CALIBRATED ESTIMATES, not externally validated probabilities.

No fabricated risk scores are shown. All numbers derive from the saved
processed CSV files written by notebooks 05 and 06.
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
    footer, metric_row, PALETTE, PLOTLY_TEMPLATE,
)
from dashboard.data_loader import (
    load_credit_risk, load_risk_scores, load_exposure_features,
    load_failure_model_performance, load_severity_lgd,
    load_financial_risk_engine_summary, load_credit_mc_summary,
    TRUE_ANNUAL_PD,
)

st.set_page_config(page_title="Infrastructure Risk · MatRisk", layout="wide", page_icon="🏗️")
apply_global_css()

hero_banner("🏗️ Infrastructure Risk Scorer", "Bridge portfolio · PD, LGD, EAD, EL · DS3 + DS6 · Notebooks 05 & 06")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with st.spinner("Loading infrastructure data..."):
    try:
        credit = load_credit_risk()
        exposure = load_exposure_features()
        fail_perf = load_failure_model_performance()
        severity_lgd = load_severity_lgd()
        fe_summary = load_financial_risk_engine_summary()
        mc_summary = load_credit_mc_summary()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

# ---------------------------------------------------------------------------
# PD calibration notice
# ---------------------------------------------------------------------------
st.markdown("### 📌 PD Calibration — Assumptions")
info_box(
    f"<strong>Population PD = {TRUE_ANNUAL_PD*100:.3f}%/yr</strong> "
    f"(1,011 DS6 failures ÷ 19 years ÷ 5,000 bridges). "
    "The case-control classifier rank-orders assets; PD is scaled so the portfolio mean matches "
    "this known rate. <strong>This is a calibrated estimate, not a validated probability.</strong> "
    "Source: notebooks 05 → 06, NOTES_for_this_handoff.md §PD calibration."
)

# ---------------------------------------------------------------------------
# Portfolio KPIs
# ---------------------------------------------------------------------------
st.markdown("### 💼 Portfolio Summary")

fe_idx = fe_summary.set_index("metric")["value"]
ead = fe_idx.get("portfolio EAD ($M)", credit["EAD_M"].sum())
el = fe_idx.get("portfolio annual EL ($M, closed-form)", credit["EL_M"].sum())
mc_el = float(mc_summary["mean_annual_loss_M"].iloc[0])
var95 = float(mc_summary["VaR95_M"].iloc[0])
var99 = float(mc_summary["VaR99_M"].iloc[0])

metric_row([
    {"label": "Portfolio EAD", "value": f"${ead:,.0f}M", "help": "Sum of loan_outstanding_M across 5,000 bridges"},
    {"label": "Annual EL (Closed-Form)", "value": f"${el:.2f}M", "help": "Basel II closed-form credit loss"},
    {"label": "MC Mean EL", "value": f"${mc_el:.2f}M", "help": "Monte Carlo mean across 10,000 simulations"},
    {"label": "MC VaR 95%", "value": f"${var95:.2f}M", "help": "95th percentile annual credit loss"},
    {"label": "MC VaR 99%", "value": f"${var99:.2f}M", "help": "99th percentile extreme credit loss"},
])

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_port, tab_asset, tab_lgd, tab_model = st.tabs([
    "📊 Portfolio View", "🔎 Asset Drill-Down", "📋 LGD Table", "🤖 Model Performance"
])

# ── Tab 1: Portfolio view ──────────────────────────────────────────────────
with tab_port:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("##### EL Distribution by Material")
        el_by_mat = credit.groupby("material")["EL_M"].sum().reset_index().sort_values("EL_M", ascending=False)
        fig = px.bar(
            el_by_mat, x="EL_M", y="material", orientation="h",
            title="Total Annual EL by Material Type",
            template=PLOTLY_TEMPLATE, height=380,
            color="EL_M", color_continuous_scale=["#00C6FF", "#7B2FF7"],
        )
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(family="Inter", color=PALETTE["text"]),
                          title_font=dict(size=13, color=PALETTE["primary"]),
                          margin=dict(l=20, r=20, t=50, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch')

    with col_b:
        st.markdown("##### EL Distribution by Location")
        el_by_loc = credit.groupby("location")["EL_M"].sum().reset_index().sort_values("EL_M", ascending=False)
        fig2 = px.bar(
            el_by_loc, x="EL_M", y="location", orientation="h",
            title="Total Annual EL by Location",
            template=PLOTLY_TEMPLATE, height=380,
            color="EL_M", color_continuous_scale=["#6BCB77", "#FF6B6B"],
        )
        fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           font=dict(family="Inter", color=PALETTE["text"]),
                           title_font=dict(size=13, color=PALETTE["primary"]),
                           margin=dict(l=20, r=20, t=50, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig2, width='stretch')

    st.markdown("##### PD Distribution (Annual)")
    fig_pd = px.histogram(
        credit["PD_annual"] * 100, nbins=60, title="Distribution of Annual PD (%) Across Portfolio",
        template=PLOTLY_TEMPLATE, height=340,
    )
    fig_pd.update_traces(marker_color=PALETTE["secondary"])
    fig_pd.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                         font=dict(family="Inter", color=PALETTE["text"]),
                         title_font=dict(size=13, color=PALETTE["primary"]),
                         xaxis_title="Annual PD (%)", showlegend=False,
                         margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_pd, width='stretch')

# ── Tab 2: Asset drill-down ───────────────────────────────────────────────
with tab_asset:
    st.markdown("#### Asset-Level Risk Explorer")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        mat_filter = st.multiselect("Filter by material", sorted(credit["material"].unique()),
                                    default=sorted(credit["material"].unique())[:3])
    with col_f2:
        loc_filter = st.multiselect("Filter by location", sorted(credit["location"].unique()),
                                    default=sorted(credit["location"].unique())[:3])

    filtered = credit.copy()
    if mat_filter:
        filtered = filtered[filtered["material"].isin(mat_filter)]
    if loc_filter:
        filtered = filtered[filtered["location"].isin(loc_filter)]

    # Sort by EL descending
    filtered = filtered.sort_values("EL_M", ascending=False)

    # Display top assets
    n_show = st.slider("Assets to display", 10, 200, 50)
    display_cols = ["bridge_id", "material", "location", "corrosion_environment",
                    "condition_rating", "PD_annual", "LGD", "EAD_M", "EL_M"]
    disp = filtered[display_cols].head(n_show).copy()
    disp["PD_annual"] = disp["PD_annual"].map("{:.4f}".format)
    disp["LGD"] = disp["LGD"].map("{:.4f}".format)
    disp["EAD_M"] = disp["EAD_M"].map("{:.3f}".format)
    disp["EL_M"] = disp["EL_M"].map("{:.4f}".format)
    st.dataframe(disp, width='stretch', hide_index=True)

    # Scatter: EAD vs EL
    fig_s = px.scatter(
        filtered.head(500), x="EAD_M", y="EL_M",
        color="material", hover_data=["bridge_id", "PD_annual", "LGD"],
        title="EAD vs. EL (top 500 by EL, filtered)",
        template=PLOTLY_TEMPLATE, height=420,
    )
    fig_s.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter", color=PALETTE["text"]),
                        title_font=dict(size=13, color=PALETTE["primary"]),
                        margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_s, width='stretch')

    warning_box(
        "Asset-level PD targeting is impossible with this data — only segment-level "
        "shocks are supported (see notebook 07 and scenario testing page). "
        "EL values are modelled estimates, not insurer quotes."
    )

# ── Tab 3: LGD table ─────────────────────────────────────────────────────
with tab_lgd:
    st.markdown("#### Empirical LGD by Severity (DS6, n=2,000 events)")
    info_box(
        "LGD is the empirical mean loss ratio from DS6 historical failures, grouped by severity. "
        "'Catastrophic' uses the capped mean (cap at 1.0) to avoid influence from loss ratios > 1. "
        "This is an empirical frequency table — not a regression model output."
    )

    lgd_display = severity_lgd.rename(columns={
        "severity": "Severity", "n": "N events",
        "mean_loss_ratio": "Mean LR", "p50_loss_ratio": "Median LR",
        "mean_lr_cap": "Mean LR (capped)", "empirical_share": "Empirical Share",
    })
    st.dataframe(lgd_display, width='stretch', hide_index=True)

    fig_lgd = px.bar(
        severity_lgd, x="severity", y="mean_lr_cap",
        title="Mean Loss Ratio by Severity (capped at 1.0)",
        template=PLOTLY_TEMPLATE, height=340,
        color="mean_lr_cap", color_continuous_scale=["#6BCB77", "#FF6B6B"],
    )
    fig_lgd.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(family="Inter", color=PALETTE["text"]),
                          title_font=dict(size=13, color=PALETTE["primary"]),
                          margin=dict(l=20, r=20, t=50, b=20), coloraxis_showscale=False,
                          yaxis_title="Mean Loss Ratio (capped)")
    st.plotly_chart(fig_lgd, width='stretch')

# ── Tab 4: Model performance ──────────────────────────────────────────────
with tab_model:
    st.markdown("#### Infrastructure Model Performance (Notebooks 05 & 09)")

    perf_display = fail_perf.rename(columns={"model": "Model", "metric": "Metric", "value": "Value"})
    perf_display["Value"] = perf_display["Value"].map("{:.4f}".format)
    st.dataframe(perf_display, width='stretch', hide_index=True)

    warning_box(
        "<strong>severity_ord R² = −0.318</strong> — severity is confirmed genuinely unpredictable "
        "by regression. The right model for severity is the empirical frequency table (LGD tab), "
        "not a regression. No severity prediction is shown in this dashboard."
    )
    info_box(
        "condition_rating: The full-feature model (R²=0.783) is only marginally better than the "
        "2-variable model (age + corrosion, R²=0.810 — slightly better!). "
        "Adding more features does not improve this model."
    )

footer()
