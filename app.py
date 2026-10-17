"""NULLXES GAG CAP VC research terminal; reads only persisted real observations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import os


st.set_page_config(page_title="NULLXES / GAG CAP VC", layout="wide", page_icon="N")
st.markdown("""
<style>
.stApp { background: #0a0a0b; color: #f2f2f0; font-family: Inter, 'Segoe UI', sans-serif; }
[data-testid="stSidebar"] { background: #111113; border-right: 1px solid #29292c; }
.block-container { max-width: 1500px; padding-top: 2.2rem; }
.nx-brand { color: #f4f4f2; font: 600 15px ui-monospace, Consolas, monospace; letter-spacing: .16em; }
.nx-sub { color: #858589; font: 11px ui-monospace, Consolas, monospace; letter-spacing: .12em; }
div[data-testid="stMetric"] { background: #121214; border: 1px solid #29292c; border-radius: 5px; padding: 16px; }
div[data-testid="stMetricLabel"] { color: #96969a; font: 11px 'DM Mono', monospace; text-transform: uppercase; }
div[data-testid="stAlert"] { background: #171719; border: 1px solid #3a3a3e; }
hr { border-color: #29292c; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="nx-brand">NULLXES&nbsp; / &nbsp;GAG CAP VC</div>', unsafe_allow_html=True)
st.markdown('<div class="nx-sub">QUANTITATIVE INTELLIGENCE · RESEARCH ONLY · CPU FIRST</div>', unsafe_allow_html=True)
st.divider()

sections = ["Overview", "Market Explorer", "Asset Forecast", "Macro Intelligence",
            "Central Bank Monitor", "GEO-RISK", "Scenario Analysis", "Risk Analytics",
            "Backtest Laboratory", "Model Registry", "Data Sources"]
section = st.sidebar.radio("RESEARCH TERMINAL", sections)
store_path = Path(st.sidebar.text_input("PARQUET LAKE", os.environ.get("GAGCAP_LAKE_PATH", "data/lake/observations.parquet")))

if not store_path.is_file():
    st.warning("DATA_UNAVAILABLE · No persisted provider observations found at the configured path.")
    st.caption("Ingest real provider data from the Colab notebook before using research views.")
    st.stop()

try:
    frame = pd.read_parquet(store_path)
except (OSError, ValueError, ImportError) as exc:
    st.error(f"DATA_UNAVAILABLE · Could not read Parquet lake: {exc}")
    st.stop()
if frame.empty or not {"source", "series_id", "observation_date", "value", "unit"}.issubset(frame.columns):
    st.warning("DATA_UNAVAILABLE · Dataset is empty or does not match the observation schema.")
    st.stop()
frame["observation_date"] = pd.to_datetime(frame["observation_date"], errors="coerce")
frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
frame = frame.dropna(subset=["observation_date", "value"])
if frame.empty:
    st.warning("DATA_UNAVAILABLE · No valid observations are available.")
    st.stop()

st.subheader(section.upper())
if section in {"Overview", "Market Explorer", "Macro Intelligence", "Central Bank Monitor"}:
    ids = sorted(frame["series_id"].astype(str).unique().tolist())
    selected = st.selectbox("SERIES", ids)
    current = frame[frame["series_id"].astype(str) == selected].sort_values("observation_date")
    if current.empty:
        st.warning("DATA_UNAVAILABLE")
    else:
        st.caption(f"SOURCE: {current.iloc[-1]['source']} · UNIT: {current.iloc[-1]['unit']} · "
                   f"AS OF: {current.iloc[-1]['observation_date'].date()}")
        st.line_chart(current.set_index("observation_date")["value"], height=380)
        left, right = st.columns(2)
        left.metric("OBSERVATIONS", f"{len(current):,}")
        right.metric("LATEST PROVIDER VALUE", f"{current.iloc[-1]['value']:,.4f}")
elif section == "Asset Forecast":
    forecast_path = store_path.parent.parent / "forecasts" / "forecasts.parquet"
    if not forecast_path.is_file():
        st.warning("DATA_UNAVAILABLE · No persisted forecast records.")
    else:
        forecasts = pd.read_parquet(forecast_path)
        if forecasts.empty:
            st.warning("DATA_UNAVAILABLE · Forecast registry is empty.")
        else:
            st.dataframe(forecasts, use_container_width=True, hide_index=True)
elif section == "Data Sources":
    st.dataframe(frame.groupby(["source", "series_id", "unit"], as_index=False).agg(
        observations=("value", "count"), first_observation=("observation_date", "min"),
        latest_observation=("observation_date", "max")), use_container_width=True, hide_index=True)
else:
    st.info("DATA_UNAVAILABLE · This view requires separately persisted, validated records.")

st.divider()
st.caption("NULLXES GAG CAP VC · EXPERIMENTAL RESEARCH OUTPUTS · NO TRADING EXECUTION")
