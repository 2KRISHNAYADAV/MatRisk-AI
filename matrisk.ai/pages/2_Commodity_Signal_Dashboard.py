"""
pages/2_Commodity_Signal_Dashboard.py
--------------------------------------
Loads the actual DS2+DS4 processed financial feature store and displays:
  - Price history and returns for 8 commodities (2014-2024)
  - Annualised volatility and rolling statistics
  - Historical VaR/ES from the saved outputs/tables/commodity_var_es.csv
  - Drawdown analysis
  - Macro-risk indicators (MQI, supply disruption, etc.)

What is NOT shown:
  - Predictive alpha / model-based return forecasts: no validated backtest exists,
    so no forward return predictions are displayed.
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
    footer, plotly_line, PALETTE, PLOTLY_TEMPLATE,
)
from dashboard.data_loader import load_financial_features, load_commodity_var_es

st.set_page_config(page_title="Commodity Signals · MatRisk", layout="wide", page_icon="📈")
apply_global_css()

hero_banner("📈 Commodity Signal Dashboard", "Historical prices · Volatility · VaR/ES · DS2+DS4 · 2014–2024")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with st.spinner("Loading commodity data..."):
    try:
        fin = load_financial_features()
        var_es = load_commodity_var_es()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

COMMODITIES = sorted(fin["commodity"].unique())
COMMODITY_DISPLAY = {c: c.replace("_", " ").replace("ETF", " ETF").replace("HRC", " HRC") for c in COMMODITIES}

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("#### Filters")
    selected = st.multiselect(
        "Commodities",
        options=COMMODITIES,
        default=COMMODITIES[:4],
        format_func=lambda c: COMMODITY_DISPLAY[c],
    )
    date_range = st.date_input(
        "Date range",
        value=[fin["date"].min().date(), fin["date"].max().date()],
        min_value=fin["date"].min().date(),
        max_value=fin["date"].max().date(),
    )

if not selected:
    st.warning("Select at least one commodity.")
    st.stop()

# Apply filters
mask = (
    fin["commodity"].isin(selected) &
    (fin["date"] >= pd.to_datetime(date_range[0])) &
    (fin["date"] <= pd.to_datetime(date_range[1]))
)
df = fin[mask].copy()

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tab_price, tab_vol, tab_risk, tab_drawdown, tab_macro = st.tabs([
    "💰 Price & Returns", "📉 Volatility", "🎲 VaR / ES", "📊 Drawdown", "🌍 Macro Indicators"
])

# ── Tab 1: Price & Returns ────────────────────────────────────────────────
with tab_price:
    st.markdown("#### Closing Price History")
    pivot_close = df.pivot_table(index="date", columns="commodity", values="close")
    # Normalise to base 100
    pivot_norm = (pivot_close / pivot_close.iloc[0] * 100).reset_index()

    fig = go.Figure()
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"],
              PALETTE["warning"], PALETTE["success"], "#A855F7", "#EC4899", "#14B8A6"]
    for i, c in enumerate(selected):
        if c in pivot_norm.columns:
            fig.add_trace(go.Scatter(
                x=pivot_norm["date"], y=pivot_norm[c],
                name=COMMODITY_DISPLAY[c], mode="lines",
                line=dict(color=colors[i % len(colors)], width=1.5),
            ))
    fig.update_layout(
        title="Normalised Closing Price (Base 100 at start of selected period)",
        template=PLOTLY_TEMPLATE, height=420,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=PALETTE["text"]),
        title_font=dict(size=14, color=PALETTE["primary"]),
        xaxis_title="Date", yaxis_title="Indexed Price (Base=100)",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, width='stretch')

    st.markdown("#### Daily Returns Distribution")
    col1, col2 = st.columns(2)
    with col1:
        commodity_ret = st.selectbox("Select commodity", selected, key="ret_sel",
                                     format_func=lambda c: COMMODITY_DISPLAY[c])
    sub = df[df["commodity"] == commodity_ret]["daily_return"].dropna()
    fig_hist = px.histogram(
        sub, nbins=80, title=f"Daily Return Distribution — {COMMODITY_DISPLAY[commodity_ret]}",
        template=PLOTLY_TEMPLATE, height=350,
    )
    fig_hist.update_traces(marker_color=PALETTE["primary"])
    fig_hist.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           font=dict(family="Inter", color=PALETTE["text"]),
                           title_font=dict(size=13, color=PALETTE["primary"]),
                           margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_hist, width='stretch')

    # Returns stats table
    st.markdown("#### Return Statistics")
    stats = (
        df.groupby("commodity")["daily_return"]
        .agg(["mean", "std", "min", "max"])
        .loc[selected]
        .rename(columns={"mean": "Mean Return", "std": "Std Dev", "min": "Min", "max": "Max"})
    )
    stats = stats.map(lambda x: f"{x:.4f}")
    st.dataframe(stats, width='stretch')

# ── Tab 2: Volatility ─────────────────────────────────────────────────────
with tab_vol:
    st.markdown("#### Rolling Annualised Volatility")
    vol_choice = st.selectbox(
        "Volatility window",
        ["volatility_5d_ann", "volatility_21d_ann", "volatility_63d_ann"],
        format_func=lambda x: x.replace("volatility_", "").replace("_ann", "").replace("d", "-day annualised"),
    )
    pivot_vol = df.pivot_table(index="date", columns="commodity", values=vol_choice).reset_index()
    fig_vol = go.Figure()
    for i, c in enumerate(selected):
        if c in pivot_vol.columns:
            fig_vol.add_trace(go.Scatter(x=pivot_vol["date"], y=pivot_vol[c],
                                         name=COMMODITY_DISPLAY[c], mode="lines",
                                         line=dict(color=colors[i % len(colors)])))
    fig_vol.update_layout(title="Rolling Annualised Volatility", template=PLOTLY_TEMPLATE, height=400,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(family="Inter", color=PALETTE["text"]),
                          title_font=dict(size=14, color=PALETTE["primary"]),
                          margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_vol, width='stretch')

    # Vol summary table
    vol_summary = (
        df.groupby("commodity")["volatility_21d_ann"]
        .agg(["mean", "median", "max"])
        .loc[[c for c in selected if c in df["commodity"].unique()]]
        .rename(columns={"mean": "Mean Vol (21d ann)", "median": "Median Vol", "max": "Peak Vol"})
    )
    vol_summary = vol_summary.map(lambda x: f"{x:.4f}")
    st.dataframe(vol_summary, width='stretch')

# ── Tab 3: VaR / ES ──────────────────────────────────────────────────────
with tab_risk:
    st.markdown("#### Historical Simulation VaR & ES (from Notebook 06)")
    info_box(
        "VaR and ES computed via historical simulation on the full DS2 price history. "
        "No distributional assumption is imposed. These are portfolio-level, not per-asset forecasts. "
        "No predictive model was trained for forward returns."
    )

    var_display = var_es.copy()
    var_display = var_display.rename(columns={
        "commodity": "Commodity", "VaR95_1d": "VaR 95% 1d",
        "ES95_1d": "ES 95% 1d", "VaR99_1d": "VaR 99% 1d", "ES99_1d": "ES 99% 1d",
        "ann_vol": "Ann. Vol", "VaR95_21d_actual": "VaR 95% 21d (actual)",
        "VaR95_21d_sqrtscaled": "VaR 95% 21d (√T)", "scaling_ratio": "Scaling Ratio",
    })
    num_cols = [c for c in var_display.columns if c != "Commodity"]
    for c in num_cols:
        var_display[c] = var_display[c].map("{:.4f}".format)
    st.dataframe(var_display, width='stretch', hide_index=True)

    # Bar chart: VaR95 1d
    fig_var = px.bar(
        var_es, x="commodity", y=["VaR95_1d", "ES95_1d"],
        barmode="group", title="1-Day VaR 95% and ES 95% by Commodity",
        template=PLOTLY_TEMPLATE, height=380,
        color_discrete_map={"VaR95_1d": PALETTE["primary"], "ES95_1d": PALETTE["accent"]},
    )
    fig_var.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(family="Inter", color=PALETTE["text"]),
                          title_font=dict(size=13, color=PALETTE["primary"]),
                          margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_var, width='stretch')

    warning_box(
        "No forward-looking return models are shown here. The VaR/ES numbers above are "
        "descriptive statistics from historical returns — they do not imply alpha or forecasting ability."
    )

# ── Tab 4: Drawdown ──────────────────────────────────────────────────────
with tab_drawdown:
    st.markdown("#### Rolling Drawdown from Peak")

    dd_sel = st.selectbox("Select commodity", selected, key="dd_sel",
                           format_func=lambda c: COMMODITY_DISPLAY[c])
    sub_dd = df[df["commodity"] == dd_sel][["date", "close"]].set_index("date").sort_index()
    rolling_max = sub_dd["close"].cummax()
    drawdown = (sub_dd["close"] - rolling_max) / rolling_max * 100

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        fill="tozeroy",
        fillcolor="rgba(255,107,107,0.2)",
        line=dict(color=PALETTE["accent"]),
        name="Drawdown %",
    ))
    fig_dd.update_layout(
        title=f"Drawdown from Peak — {COMMODITY_DISPLAY[dd_sel]}",
        template=PLOTLY_TEMPLATE, height=380,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=PALETTE["text"]),
        title_font=dict(size=14, color=PALETTE["primary"]),
        yaxis_title="Drawdown (%)", xaxis_title="Date",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_dd, width='stretch')

    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    st.metric("Maximum Drawdown", f"{max_dd:.2f}%", help=f"Worst peak-to-trough: {max_dd_date.date()}")

# ── Tab 5: Macro Indicators ───────────────────────────────────────────────
with tab_macro:
    st.markdown("#### Material Quality Index (MQI) and Supply Risk Signals")
    info_box(
        "MQI, supply_disruption_prob, and herfindahl_index are engineered features from DS4 "
        "(notebook 03). They are descriptive cross-domain signals — not model outputs."
    )

    macro_sel = st.selectbox("Select commodity", selected, key="macro_sel",
                              format_func=lambda c: COMMODITY_DISPLAY[c])
    sub_m = df[df["commodity"] == macro_sel].sort_values("date")

    macro_cols = ["mqi", "supply_disruption_prob", "herfindahl_index"]
    available = [c for c in macro_cols if c in sub_m.columns]

    if available:
        pivot_macro = sub_m[["date"] + available].set_index("date")
        fig_macro = go.Figure()
        for i, c in enumerate(available):
            fig_macro.add_trace(go.Scatter(
                x=pivot_macro.index, y=pivot_macro[c],
                name=c.replace("_", " ").title(),
                line=dict(color=colors[i % len(colors)]),
            ))
        fig_macro.update_layout(
            title=f"Macro/Risk Signals — {COMMODITY_DISPLAY[macro_sel]}",
            template=PLOTLY_TEMPLATE, height=380,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color=PALETTE["text"]),
            title_font=dict(size=14, color=PALETTE["primary"]),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_macro, width='stretch')
    else:
        st.info("Macro indicator columns not found in the financial features dataset.")

footer()
