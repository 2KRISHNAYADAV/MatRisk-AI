"""
pages/4_Scenario_Stress_Testing.py
------------------------------------
Connects the actual stress-scenario results from notebook 07.

Supported scenarios (from scenario_price_shock_summary.csv and
scenario_credit_stress_summary.csv):
  SC-001  Critical minerals demand surge
  SC-002  Green-steel transition
  SC-003  Climate-accelerated corrosion  ← credit stress (triples EL)
  SC-004  Aluminium substitution
  SC-005  Alloy improvement (corrosion-reducing)

IMPORTANT LIMITATIONS (documented in final_signoff.csv):
  - Status: demonstration-grade
  - Mechanically sound and reuses notebooks 05/06 actual fitted models.
  - Asset-level targeting is impossible with this data; shocks are segment-level.
  - Scenario assumptions are documented in the assumptions ledger.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.components import (
    apply_global_css, hero_banner, warning_box, info_box,
    footer, PALETTE, PLOTLY_TEMPLATE,
)
from dashboard.data_loader import (
    load_scenario_price_shock, load_scenario_credit_stress,
    load_assumptions_ledger, load_final_signoff, load_credit_mc_summary,
)

st.set_page_config(page_title="Scenario & Stress · MatRisk", layout="wide", page_icon="⚡")
apply_global_css()

hero_banner("⚡ Scenario & Stress Testing", "5 scenarios · Price shocks · Credit stress · Notebooks 07 & 09")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with st.spinner("Loading scenario data..."):
    try:
        price_shocks = load_scenario_price_shock()
        credit_stress = load_scenario_credit_stress()
        assumptions = load_assumptions_ledger()
        signoff = load_final_signoff()
        mc_baseline = load_credit_mc_summary()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

# ---------------------------------------------------------------------------
# Status banner
# ---------------------------------------------------------------------------
sc_row = signoff[signoff["component"].str.contains("Scenario", case=False)]
if not sc_row.empty:
    note = sc_row.iloc[0]["note"]
    warning_box(f"<strong>Status: Demonstration-grade</strong> — {note}")

# ---------------------------------------------------------------------------
# Scenario descriptions
# ---------------------------------------------------------------------------
SCENARIO_INFO = {
    "SC-001": ("Critical Minerals Demand Surge",
               "Rare earth and lithium demand shock from EV/battery transition. "
               "Price effects on material cost. +179% Rare Earth, +25% Lithium."),
    "SC-002": ("Green-Steel Transition",
               "Steel demand shifts to recycled/electric-arc routes. "
               "Steel HRC −8%, Iron Ore −5%."),
    "SC-003": ("Climate-Accelerated Corrosion",
               "30-year climate scenario: median corrosion rate +50% for coastal/urban assets. "
               "Credit EL ~triples (×2.91). Runs through the actual fitted condition-rating model."),
    "SC-004": ("Aluminium Substitution",
               "Virgin aluminium −15%, recycled aluminium +25%. "
               "Carbon and cost implications for material selection."),
    "SC-005": ("Alloy Improvement (Favourable)",
               "Next-generation corrosion-resistant alloys reduce corrosion rate −20%. "
               "Credit EL reduces to ×0.73 of baseline. Runs through actual fitted models."),
}

# ---------------------------------------------------------------------------
# Sidebar: scenario selector
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("#### Scenario Selector")
    selected_scenarios = st.multiselect(
        "Select scenarios to compare",
        list(SCENARIO_INFO.keys()),
        default=["SC-001", "SC-003"],
        format_func=lambda k: f"{k}: {SCENARIO_INFO[k][0]}",
    )

if not selected_scenarios:
    st.warning("Select at least one scenario.")
    st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_desc, tab_price, tab_credit, tab_assump = st.tabs([
    "📖 Scenario Descriptions", "💰 Price Shock Effects", "🏗️ Credit Stress Effects", "📋 Assumptions Ledger"
])

# ── Tab 1: Descriptions ───────────────────────────────────────────────────
with tab_desc:
    for sc in selected_scenarios:
        name, desc = SCENARIO_INFO[sc]
        st.markdown(
            f"""
            <div style="background:{PALETTE['bg_card']};border:1px solid #30363D;border-radius:12px;
                        padding:1rem 1.25rem;margin-bottom:1rem;">
                <div style="color:{PALETTE['primary']};font-weight:700;font-size:1rem;">{sc} — {name}</div>
                <div style="color:{PALETTE['text']};margin-top:0.4rem;font-size:0.9rem;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Tab 2: Price shock effects ────────────────────────────────────────────
with tab_price:
    st.markdown("#### Commodity Price Shock Effects")
    info_box(
        "Price changes are simulated from historical volatility (×2–×3 std shock) "
        "and cross-correlation structure from DS2. Shown as % change distributions "
        "(mean and 5th/95th percentile). Source: notebook 07 / scenario_price_shock_summary.csv"
    )

    shock_sel = [s for s in selected_scenarios if s in price_shocks["scenario"].values]
    if not shock_sel:
        st.info("No price shock data for the selected scenarios. "
                "(SC-003 and SC-005 affect credit only, not price.)")
    else:
        shock_df = price_shocks[price_shocks["scenario"].isin(shock_sel)].copy()

        fig_shock = go.Figure()
        colors_sc = {sc: col for sc, col in zip(
            shock_sel,
            [PALETTE["primary"], PALETTE["accent"], PALETTE["warning"], PALETTE["success"], PALETTE["secondary"]]
        )}

        for sc in shock_sel:
            sub = shock_df[shock_df["scenario"] == sc]
            fig_shock.add_trace(go.Bar(
                name=f"{sc}: {SCENARIO_INFO[sc][0]}",
                x=sub["commodity"],
                y=sub["mean_pct"],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=(sub["p95_pct"] - sub["mean_pct"]).values,
                    arrayminus=(sub["mean_pct"] - sub["p05_pct"]).values,
                    color=PALETTE["muted"],
                ),
                marker_color=colors_sc[sc],
            ))

        fig_shock.update_layout(
            title="Price Change (%) — Mean and 5th/95th Percentile by Scenario",
            barmode="group", template=PLOTLY_TEMPLATE, height=420,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color=PALETTE["text"]),
            title_font=dict(size=14, color=PALETTE["primary"]),
            xaxis_title="Commodity", yaxis_title="Price Change (%)",
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_shock, width='stretch')

        # Table
        st.markdown("#### Detailed Price Shock Table")
        display_shock = shock_df.rename(columns={
            "scenario": "Scenario", "commodity": "Commodity",
            "mean_pct": "Mean Δ%", "p05_pct": "P05 Δ%", "p95_pct": "P95 Δ%",
        })
        for col in ["Mean Δ%", "P05 Δ%", "P95 Δ%"]:
            display_shock[col] = display_shock[col].map("{:+.2f}%".format)
        st.dataframe(display_shock, width='stretch', hide_index=True)

# ── Tab 3: Credit stress ──────────────────────────────────────────────────
with tab_credit:
    st.markdown("#### Credit Portfolio Stress Results")
    info_box(
        "Credit stress runs the actual fitted condition-rating (DS3) and case-control failure (DS6) "
        "models with perturbed inputs. Only the segment (material + location) targeted by the "
        "scenario is shocked — asset-level targeting is not possible with this data."
    )

    # Baseline MC
    mc_mean = float(mc_baseline["mean_annual_loss_M"].iloc[0])
    mc_var95 = float(mc_baseline["VaR95_M"].iloc[0])
    mc_var99 = float(mc_baseline["VaR99_M"].iloc[0])

    # Show credit stress table
    cs_display = credit_stress.copy()
    cs_display["EL multiple vs baseline"] = (cs_display["mean_loss_M"] / cs_display["mean_loss_M"].iloc[0]).map("{:.2f}×".format)
    cs_display = cs_display.rename(columns={
        "scenario": "Scenario", "mean_loss_M": "Mean EL ($M)",
        "VaR95_M": "VaR 95% ($M)", "VaR99_M": "VaR 99% ($M)",
    })
    cs_display["Mean EL ($M)"] = cs_display["Mean EL ($M)"].map("${:.3f}M".format)
    cs_display["VaR 95% ($M)"] = cs_display["VaR 95% ($M)"].map("${:.3f}M".format)
    cs_display["VaR 99% ($M)"] = cs_display["VaR 99% ($M)"].map("${:.3f}M".format)
    st.dataframe(cs_display, width='stretch', hide_index=True)

    # Waterfall-style comparison
    scenarios_with_credit = credit_stress["scenario"].tolist()
    labels = ["Baseline MC"] + [s for s in scenarios_with_credit if s != "baseline"]
    values = [mc_mean] + [
        float(credit_stress[credit_stress["scenario"] == s]["mean_loss_M"].iloc[0])
        for s in scenarios_with_credit if s != "baseline"
    ]
    bar_colors = [PALETTE["primary"]] + [
        PALETTE["accent"] if v > mc_mean else PALETTE["success"] for v in values[1:]
    ]

    fig_bar = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=bar_colors,
        text=[f"${v:.2f}M" for v in values],
        textposition="outside",
    ))
    fig_bar.update_layout(
        title="Mean Annual Credit EL — Baseline vs. Stressed",
        template=PLOTLY_TEMPLATE, height=380,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=PALETTE["text"]),
        title_font=dict(size=14, color=PALETTE["primary"]),
        yaxis_title="Mean Annual EL ($M)",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig_bar, width='stretch')

    # Key finding callout
    st.markdown(
        f"""
        <div style="background:{PALETTE['bg_card']};border:1px solid {PALETTE['accent']};
                    border-radius:12px;padding:1rem 1.25rem;margin:1rem 0;">
            <div style="color:{PALETTE['accent']};font-weight:700;">🔥 Key Finding (SC-003)</div>
            <div style="color:{PALETTE['text']};margin-top:0.4rem;">
                Climate-accelerated corrosion (SC-003) <strong>~triples credit EL</strong> 
                (baseline $4.19M → stressed $12.22M, ×2.91). 
                This runs through the actual fitted monotonic condition-rating and 
                case-control models — not a separate toy calculation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    warning_box(
        "These are <strong>segment-level shocks</strong> only. Asset-level stress targeting "
        "(e.g. shock only coastal steel bridges above age 40) is not possible without "
        "richer event-level data linking DS6 failures to DS3 assets."
    )

# ── Tab 4: Assumptions ────────────────────────────────────────────────────
with tab_assump:
    st.markdown("#### Full Assumptions Ledger (from Notebook 09)")
    info_box(
        "All modelling decisions and assumptions across notebooks 03–09 are logged here. "
        "Source: outputs/tables/final_assumptions_ledger.csv"
    )

    st.dataframe(
        assumptions.rename(columns={"dataset": "Dataset", "PROBLEM": "Problem",
                                    "DECISION": "Decision", "notebook": "Notebook"}),
        width='stretch', hide_index=True,
    )

footer()
