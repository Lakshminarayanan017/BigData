"""
Phase 7: Streamlit Dashboard
==============================
Interactive logistics management dashboard for airline cargo optimization.

Sections:
  🏠 Overview         — KPI cards, pipeline summary
  ✈️  Flights          — Flight analytics, capacity map
  📦 Cargo Allocation — Optimization results, shipment table
  🌐 Route Analysis   — Route heatmap, cost comparison
  📡 Live Events      — Kafka flight event stream
  📊 Scalability      — Pandas vs Spark benchmark charts
  🏗️  Architecture     — System diagram and data flow

Run:
    streamlit run src/dashboard.py
"""

import os
import sys
import json
import time
import sqlite3
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ANALYTICS_DIR = os.path.join(BASE_DIR, "data", "analytics")
DB_PATH       = os.path.join(ANALYTICS_DIR, "aviation_cargo.db")
RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aviation Cargo Optimizer",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Aviation Cargo Logistics Optimization — Big Data Architecture Project"}
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark glassmorphism sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid rgba(99,102,241,0.3);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1a2744 50%, #0f172a 100%);
}

/* KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(139,92,246,0.1) 100%);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 16px;
    padding: 20px 24px;
    margin: 4px 0;
    backdrop-filter: blur(10px);
    transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(99,102,241,0.3);
}
.kpi-value {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}
.kpi-label {
    font-size: 0.8rem;
    color: #94a3b8;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 4px;
}
.kpi-delta {
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 6px;
}
.kpi-delta.positive { color: #34d399; }
.kpi-delta.warning  { color: #fbbf24; }
.kpi-delta.danger   { color: #f87171; }

/* Section headers */
.section-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 24px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid rgba(99,102,241,0.4);
}

/* Status badges */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-express  { background: rgba(239,68,68,0.2);  color: #f87171; border: 1px solid rgba(239,68,68,0.4); }
.badge-high     { background: rgba(251,146,60,0.2); color: #fb923c; border: 1px solid rgba(251,146,60,0.4); }
.badge-medium   { background: rgba(250,204,21,0.2); color: #fde047; border: 1px solid rgba(250,204,21,0.4); }
.badge-low      { background: rgba(74,222,128,0.2); color: #4ade80; border: 1px solid rgba(74,222,128,0.4); }

/* Alert boxes */
.alert-warning {
    background: rgba(251,146,60,0.1);
    border: 1px solid rgba(251,146,60,0.4);
    border-radius: 8px;
    padding: 12px 16px;
    color: #fb923c;
}
.alert-success {
    background: rgba(74,222,128,0.1);
    border: 1px solid rgba(74,222,128,0.4);
    border-radius: 8px;
    padding: 12px 16px;
    color: #4ade80;
}

/* Override Streamlit default element backgrounds */
.stPlotlyChart { background: transparent !important; }
div[data-testid="metric-container"] {
    background: rgba(30,41,59,0.8);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px;
    padding: 12px;
}

/* Dataframe styling */
.stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor":  "rgba(15,23,42,0.6)",
    "font":          {"family": "Inter", "color": "#e2e8f0"},
}

# ─────────────────────────────────────────────────────────────────────────────
# Data Loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def db_query(sql: str) -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql_query(sql, conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def get_kpis() -> dict:
    def q(sql):
        df = db_query(sql)
        return df.iloc[0, 0] if not df.empty else 0

    ship_count  = q("SELECT COUNT(*) FROM shipments")
    alloc_count = q("SELECT COUNT(*) FROM allocation")
    total_cost  = q("SELECT COALESCE(SUM(cost_usd), 0) FROM allocation")
    avg_util    = q("SELECT COALESCE(AVG(utilization_pct), 0) FROM flight_utilization")
    delayed     = q("SELECT COALESCE(MAX(delayed_flights),   0) FROM kafka_events")
    cancelled   = q("SELECT COALESCE(MAX(cancelled_flights), 0) FROM kafka_events")
    total_events = q("SELECT COALESCE(MAX(total_events),     0) FROM kafka_events")

    alloc_rate = (alloc_count / ship_count * 100) if ship_count > 0 else 0

    return {
        "total_shipments":   int(ship_count),
        "allocated":         int(alloc_count),
        "allocation_rate":   round(float(alloc_rate), 1),
        "total_cost_usd":    round(float(total_cost), 2),
        "avg_utilization":   round(float(avg_util), 1),
        "delayed_flights":   int(delayed),
        "cancelled_flights": int(cancelled),
        "total_events":      int(total_events),
    }


@st.cache_data(ttl=30)
def load_scalability() -> pd.DataFrame:
    path = os.path.join(ANALYTICS_DIR, "scalability_results.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=30)
def load_kafka_events() -> dict:
    path = os.path.join(ANALYTICS_DIR, "kafka_summary.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Airport Coordinates (for map rendering)
# ─────────────────────────────────────────────────────────────────────────────
AIRPORT_COORDS = {
    "MAA": (12.99, 80.17, "Chennai"),
    "DEL": (28.55, 77.09, "Delhi"),
    "BOM": (19.09, 72.87, "Mumbai"),
    "BLR": (13.19, 77.70, "Bangalore"),
    "HYD": (17.24, 78.43, "Hyderabad"),
    "CCU": (22.65, 88.45, "Kolkata"),
    "SIN": (1.36, 103.99, "Singapore"),
    "DXB": (25.25, 55.36, "Dubai"),
    "LHR": (51.47, -0.46, "London"),
    "FRA": (50.03, 8.57,  "Frankfurt"),
    "BKK": (13.69, 100.75, "Bangkok"),
    "KUL": (2.74, 101.70, "Kuala Lumpur"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px 0;'>
        <div style='font-size:2.5rem'>✈️</div>
        <div style='font-size:1.1rem; font-weight:700; color:#818cf8;'>Cargo Optimizer</div>
        <div style='font-size:0.7rem; color:#64748b; margin-top:2px;'>Big Data Architecture</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Overview",
         "✈️ Flight Analytics",
         "📦 Cargo Allocation",
         "🌐 Route Analysis",
         "📡 Live Events",
         "📊 Scalability Benchmarks",
         "🏗️ Architecture"],
        label_visibility="collapsed"
    )

    st.divider()

    # Re-run pipeline button
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("""
    <div style='margin-top:16px; padding: 12px; background: rgba(99,102,241,0.1);
                border-radius:8px; border: 1px solid rgba(99,102,241,0.3);'>
        <div style='font-size:0.7rem; color:#64748b; font-weight:600;'>TECH STACK</div>
        <div style='font-size:0.75rem; color:#94a3b8; margin-top:4px; line-height:1.6'>
            🟠 Apache Kafka<br>
            🟡 Apache Spark<br>
            🔵 OR-Tools MIP<br>
            🟢 SQLite / Parquet<br>
            🔴 Streamlit + Plotly
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div style='padding: 24px 0 8px 0;'>
        <h1 style='font-size:2.2rem; font-weight:800; color:#e2e8f0; margin:0;'>
            ✈️ Aviation Cargo Logistics Optimizer
        </h1>
        <p style='color:#64748b; margin-top:4px; font-size:0.95rem;'>
            Big Data Architecture — Real-Time Cargo Allocation & Route Optimization
        </p>
    </div>
    """, unsafe_allow_html=True)

    kpis = get_kpis()
    data_ready = kpis["total_shipments"] > 0

    if not data_ready:
        st.warning("""
        ⚠️ **No data found.** Run the pipeline first:
        ```bash
        python run_pipeline.py
        ```
        """)
        st.stop()

    # ── KPI Row 1 ───────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{kpis['total_shipments']:,}</div>
            <div class="kpi-label">📦 Total Shipments</div>
            <div class="kpi-delta positive">↑ Loaded in pipeline</div>
        </div>""", unsafe_allow_html=True)

    with k2:
        rate = kpis["allocation_rate"]
        color = "positive" if rate >= 80 else "warning" if rate >= 60 else "danger"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{rate}%</div>
            <div class="kpi-label">✅ Allocation Rate</div>
            <div class="kpi-delta {color}">{kpis['allocated']:,} shipments assigned</div>
        </div>""", unsafe_allow_html=True)

    with k3:
        cost_m = kpis["total_cost_usd"] / 1_000_000
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">${cost_m:.2f}M</div>
            <div class="kpi-label">💰 Total Cargo Cost</div>
            <div class="kpi-delta positive">Optimized by OR-Tools</div>
        </div>""", unsafe_allow_html=True)

    with k4:
        util = kpis["avg_utilization"]
        color = "positive" if util >= 70 else "warning"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{util:.1f}%</div>
            <div class="kpi-label">🛫 Avg Capacity Used</div>
            <div class="kpi-delta {color}">Flight utilization</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── KPI Row 2 ────────────────────────────────────────────────────────────
    k5, k6, k7, k8 = st.columns(4)

    with k5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{kpis['total_events']:,}</div>
            <div class="kpi-label">📡 Kafka Events</div>
            <div class="kpi-delta positive">Processed in stream</div>
        </div>""", unsafe_allow_html=True)

    with k6:
        d = kpis["delayed_flights"]
        color = "danger" if d > 20 else "warning" if d > 10 else "positive"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{d}</div>
            <div class="kpi-label">⏰ Delayed Flights</div>
            <div class="kpi-delta {color}">Active disruptions</div>
        </div>""", unsafe_allow_html=True)

    with k7:
        c = kpis["cancelled_flights"]
        color = "danger" if c > 5 else "warning" if c > 2 else "positive"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{c}</div>
            <div class="kpi-label">❌ Cancelled Flights</div>
            <div class="kpi-delta {color}">Needs rerouting</div>
        </div>""", unsafe_allow_html=True)

    with k8:
        unalloc = kpis["total_shipments"] - kpis["allocated"]
        color = "danger" if unalloc > 20 else "warning" if unalloc > 10 else "positive"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{unalloc}</div>
            <div class="kpi-label">🔴 Unallocated</div>
            <div class="kpi-delta {color}">Pending assignment</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts Row ────────────────────────────────────────────────────────────
    col_a, col_b = st.columns([1.2, 0.8])

    with col_a:
        st.markdown('<div class="section-header">📦 Shipments by Priority & Type</div>',
                    unsafe_allow_html=True)
        ship_df = db_query("""
            SELECT priority, cargo_type, COUNT(*) as count, SUM(weight_kg) as total_weight
            FROM shipments GROUP BY priority, cargo_type ORDER BY count DESC
        """)
        if not ship_df.empty:
            fig = px.bar(
                ship_df.head(20), x="cargo_type", y="count", color="priority",
                color_discrete_map={
                    "Express": "#f87171", "High": "#fb923c",
                    "Medium": "#fde047", "Low": "#4ade80"
                },
                labels={"count": "Shipments", "cargo_type": "Cargo Type"},
                **PLOTLY_THEME
            )
            fig.update_layout(height=300, showlegend=True,
                              legend=dict(orientation="h", y=1.08))
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">📊 Allocation Status</div>',
                    unsafe_allow_html=True)
        allocated   = kpis["allocated"]
        unallocated = kpis["total_shipments"] - allocated

        fig = go.Figure(go.Pie(
            labels=["Allocated", "Unallocated"],
            values=[allocated, unallocated],
            hole=0.65,
            marker_colors=["#818cf8", "#475569"],
            textinfo="percent",
        ))
        fig.update_layout(
            height=280,
            annotations=[dict(text=f"{kpis['allocation_rate']}%",
                              x=0.5, y=0.5, font_size=24, showarrow=False,
                              font_color="#818cf8", font_family="Inter")],
            **PLOTLY_THEME
        )
        fig.update_traces(textfont_size=11)
        st.plotly_chart(fig, use_container_width=True)

    # ── Pipeline Status ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔄 Pipeline Component Status</div>',
                unsafe_allow_html=True)

    pipe_cols = st.columns(6)
    components = [
        ("📂 Data Generated",     os.path.join(RAW_DIR, "shipments.csv")),
        ("⚡ Spark Pipeline",      os.path.join(ANALYTICS_DIR, "flight_analytics.parquet")),
        ("📡 Kafka Events",       os.path.join(ANALYTICS_DIR, "flight_events.jsonl")),
        ("🔧 Optimization",       os.path.join(ANALYTICS_DIR, "allocation_results.csv")),
        ("🗄️ Database",            DB_PATH),
        ("📊 Benchmarks",         os.path.join(ANALYTICS_DIR, "scalability_results.csv")),
    ]

    for col, (label, path) in zip(pipe_cols, components):
        exists = os.path.exists(path)
        icon   = "✅" if exists else "⭕"
        color  = "#4ade80" if exists else "#ef4444"
        col.markdown(f"""
        <div style='text-align:center; padding:12px; background:rgba(30,41,59,0.8);
                    border-radius:10px; border:1px solid rgba(99,102,241,0.2);'>
            <div style='font-size:1.5rem'>{icon}</div>
            <div style='font-size:0.7rem; color:{color}; margin-top:4px; font-weight:600;'>
                {label}
            </div>
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Flight Analytics
# ═════════════════════════════════════════════════════════════════════════════
elif page == "✈️ Flight Analytics":
    st.markdown("""
    <h1 style='font-size:1.8rem; font-weight:800; color:#e2e8f0; margin:0 0 4px 0;'>
        ✈️ Flight Analytics
    </h1>
    <p style='color:#64748b; margin-bottom:16px;'>Capacity utilization, delay risk, and airline performance</p>
    """, unsafe_allow_html=True)

    # Filters
    f1, f2, f3 = st.columns(3)
    with f1:
        airlines_df = db_query("SELECT DISTINCT airline FROM flights WHERE airline IS NOT NULL")
        airlines    = ["All"] + airlines_df["airline"].tolist() if not airlines_df.empty else ["All"]
        sel_airline = st.selectbox("Airline", airlines)

    with f2:
        origins_df = db_query("SELECT DISTINCT origin FROM flights ORDER BY origin")
        origins    = ["All"] + origins_df["origin"].tolist() if not origins_df.empty else ["All"]
        sel_origin = st.selectbox("Origin Airport", origins)

    with f3:
        risk_threshold = st.slider("Max Delay Risk Score", 0, 100, 100, 5)

    # Query
    where_clauses = ["1=1"]
    if sel_airline != "All":
        where_clauses.append(f"airline = '{sel_airline}'")
    if sel_origin != "All":
        where_clauses.append(f"origin = '{sel_origin}'")
    where_clauses.append(f"delay_risk_score <= {risk_threshold}")
    where_str = " AND ".join(where_clauses)

    flights_df = db_query(f"""
        SELECT f.*, COALESCE(u.utilization_pct, 0) as utilization_pct,
               COALESCE(u.loaded_kg, 0) as loaded_kg
        FROM flights f
        LEFT JOIN flight_utilization u ON f.flight_id = u.flight_id
        WHERE {where_str}
        ORDER BY f.delay_risk_score DESC
        LIMIT 500
    """)

    if flights_df.empty:
        st.info("No flight data found. Run the pipeline first.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Flights", f"{len(flights_df):,}")
        m2.metric("Avg Delay Risk", f"{flights_df['delay_risk_score'].mean():.1f}")
        m3.metric("Avg Capacity (kg)", f"{flights_df['cargo_capacity_kg'].mean():,.0f}")
        m4.metric("Avg Utilization", f"{flights_df['utilization_pct'].mean():.1f}%")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="section-header">🎯 Delay Risk Distribution</div>',
                        unsafe_allow_html=True)
            fig = px.histogram(
                flights_df, x="delay_risk_score", nbins=30,
                color_discrete_sequence=["#818cf8"],
                labels={"delay_risk_score": "Delay Risk Score", "count": "Flights"},
                **PLOTLY_THEME
            )
            fig.update_layout(height=280, showlegend=False)
            fig.update_traces(marker_line_color="#6366f1", marker_line_width=0.5)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown('<div class="section-header">🛫 Capacity Utilization by Airline</div>',
                        unsafe_allow_html=True)
            if "airline" in flights_df.columns:
                airline_util = (flights_df.groupby("airline")["utilization_pct"]
                               .mean().reset_index().sort_values("utilization_pct", ascending=True))
                fig = px.bar(
                    airline_util, x="utilization_pct", y="airline",
                    orientation="h",
                    color="utilization_pct",
                    color_continuous_scale=["#f87171", "#fde047", "#4ade80"],
                    range_color=[0, 100],
                    labels={"utilization_pct": "Utilization %", "airline": ""},
                    **PLOTLY_THEME
                )
                fig.update_layout(height=280, showlegend=False,
                                  coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

        # Scatter: Cost vs Delay Risk
        st.markdown('<div class="section-header">💸 Cost vs Delay Risk (per flight)</div>',
                    unsafe_allow_html=True)
        fig = px.scatter(
            flights_df.dropna(subset=["cost_per_kg", "delay_risk_score"]),
            x="delay_risk_score", y="cost_per_kg",
            color="utilization_pct",
            size="cargo_capacity_kg",
            size_max=20,
            hover_data=["flight_id", "origin", "destination", "airline"],
            color_continuous_scale="plasma",
            labels={"delay_risk_score": "Delay Risk Score",
                    "cost_per_kg": "Cost per kg (USD)",
                    "utilization_pct": "Utilization %"},
            **PLOTLY_THEME
        )
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

        # Flight table
        with st.expander("📋 Flight Details Table", expanded=False):
            show_cols = ["flight_id", "airline", "aircraft_type", "origin", "destination",
                         "cargo_capacity_kg", "cost_per_kg", "delay_risk_score",
                         "utilization_pct", "on_time_history_pct"]
            show_cols = [c for c in show_cols if c in flights_df.columns]
            st.dataframe(
                flights_df[show_cols].round(2),
                hide_index=True, use_container_width=True
            )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Cargo Allocation
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📦 Cargo Allocation":
    st.markdown("""
    <h1 style='font-size:1.8rem; font-weight:800; color:#e2e8f0; margin:0 0 4px 0;'>
        📦 Cargo Allocation Results
    </h1>
    <p style='color:#64748b; margin-bottom:16px;'>OR-Tools MIP optimization — shipment-to-flight assignments</p>
    """, unsafe_allow_html=True)

    alloc_df = db_query("SELECT * FROM allocation ORDER BY cost_usd DESC")
    util_df  = db_query("SELECT * FROM flight_utilization ORDER BY utilization_pct DESC")
    opt_df   = db_query("SELECT * FROM optimization_metrics ORDER BY id DESC LIMIT 1")

    if alloc_df.empty:
        st.info("No allocation results found. Run the optimization engine first.")
        st.code("python run_pipeline.py", language="bash")
        st.stop()

    # Optimization method info
    if not opt_df.empty:
        row = opt_df.iloc[0]
        m_col1, m_col2 = st.columns([2, 1])
        with m_col1:
            st.markdown(f"""
            <div class="alert-success">
                ✅ <strong>{row.get('method', 'Optimization')} completed</strong> in
                <strong>{row.get('solve_time_sec', 0):.2f}s</strong> |
                Total cost: <strong>${row.get('total_cost_usd', 0):,.2f}</strong> |
                Allocation rate: <strong>{row.get('allocation_rate', 0):.1f}%</strong>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header">💰 Cost by Priority</div>',
                    unsafe_allow_html=True)
        cost_by_priority = (alloc_df.groupby("priority")["cost_usd"]
                           .agg(["sum", "mean", "count"])
                           .reset_index()
                           .rename(columns={"sum": "total_cost", "mean": "avg_cost", "count": "shipments"}))
        fig = px.bar(
            cost_by_priority, x="priority", y="total_cost",
            color="priority",
            color_discrete_map={
                "Express": "#f87171", "High": "#fb923c",
                "Medium": "#fde047", "Low": "#4ade80"
            },
            text="shipments",
            labels={"total_cost": "Total Cost (USD)", "priority": "Priority"},
            **PLOTLY_THEME
        )
        fig.update_traces(texttemplate="%{text} shipments", textposition="outside")
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">🛫 Flight Capacity Utilization</div>',
                    unsafe_allow_html=True)
        if not util_df.empty:
            fig = px.bar(
                util_df.head(15),
                x="flight_id", y="utilization_pct",
                color="utilization_pct",
                color_continuous_scale=["#f87171", "#fde047", "#4ade80"],
                range_color=[0, 100],
                labels={"utilization_pct": "Utilization %", "flight_id": "Flight"},
                **PLOTLY_THEME
            )
            fig.update_layout(height=300, showlegend=False,
                              coloraxis_showscale=True, xaxis_tickangle=-45)
            fig.add_hline(y=75, line_dash="dash", line_color="#818cf8",
                          annotation_text="75% Target")
            st.plotly_chart(fig, use_container_width=True)

    # Weight distribution in allocations
    st.markdown('<div class="section-header">📊 Weight Distribution by Cargo Type</div>',
                unsafe_allow_html=True)
    fig = px.box(
        alloc_df.dropna(subset=["cargo_type", "weight_kg"]),
        x="cargo_type", y="weight_kg",
        color="priority",
        color_discrete_map={
            "Express": "#f87171", "High": "#fb923c",
            "Medium": "#fde047", "Low": "#4ade80"
        },
        labels={"weight_kg": "Weight (kg)", "cargo_type": "Cargo Type"},
        **PLOTLY_THEME
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    # Allocation table
    st.markdown('<div class="section-header">📋 Allocation Results</div>',
                unsafe_allow_html=True)

    # Filters
    prio_filter = st.multiselect(
        "Filter by Priority",
        ["Express", "High", "Medium", "Low"],
        default=["Express", "High", "Medium", "Low"]
    )

    filtered = alloc_df[alloc_df["priority"].isin(prio_filter)] if prio_filter else alloc_df
    show_cols = ["shipment_id", "flight_id", "origin", "destination",
                 "weight_kg", "priority", "cargo_type", "airline",
                 "departure_time", "cost_usd", "delay_risk_score"]
    show_cols = [c for c in show_cols if c in filtered.columns]

    st.dataframe(
        filtered[show_cols].round(2).head(500),
        hide_index=True, use_container_width=True
    )

    # Unallocated
    unalloc_path = os.path.join(ANALYTICS_DIR, "unallocated.csv")
    if os.path.exists(unalloc_path):
        unalloc = pd.read_csv(unalloc_path)
        if not unalloc.empty:
            with st.expander(f"⚠️ Unallocated Shipments ({len(unalloc)})", expanded=False):
                st.dataframe(unalloc, hide_index=True, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Route Analysis
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🌐 Route Analysis":
    st.markdown("""
    <h1 style='font-size:1.8rem; font-weight:800; color:#e2e8f0; margin:0 0 4px 0;'>
        🌐 Route Analysis
    </h1>
    <p style='color:#64748b; margin-bottom:16px;'>Network heatmap, cost comparison, and route performance</p>
    """, unsafe_allow_html=True)

    route_df = db_query("""
        SELECT origin, destination,
               COUNT(*) as total_flights,
               AVG(cost_per_kg) as avg_cost_per_kg,
               AVG(delay_risk_score) as avg_delay_risk,
               AVG(on_time_history_pct) as avg_on_time_pct,
               SUM(cargo_capacity_kg) as total_capacity_kg
        FROM flights
        GROUP BY origin, destination
        HAVING total_flights > 0
        ORDER BY total_flights DESC
    """)

    if route_df.empty:
        st.info("No route data available. Run the pipeline first.")
        st.stop()

    route_df["route"] = route_df["origin"] + " → " + route_df["destination"]

    # Route volume heatmap
    st.markdown('<div class="section-header">🔥 Route Traffic Heatmap</div>',
                unsafe_allow_html=True)

    pivot = route_df.pivot_table(
        index="origin", columns="destination",
        values="total_flights", aggfunc="sum"
    ).fillna(0)

    fig = px.imshow(
        pivot,
        color_continuous_scale="plasma",
        labels=dict(x="Destination", y="Origin", color="Flights"),
        **PLOTLY_THEME
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header">💸 Top Routes by Cost</div>',
                    unsafe_allow_html=True)
        top_cost = route_df.nlargest(12, "avg_cost_per_kg")
        fig = px.bar(
            top_cost, x="avg_cost_per_kg", y="route",
            orientation="h",
            color="avg_cost_per_kg",
            color_continuous_scale="reds",
            labels={"avg_cost_per_kg": "Avg Cost/kg (USD)", "route": ""},
            **PLOTLY_THEME
        )
        fig.update_layout(height=380, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">⚠️ Top Routes by Delay Risk</div>',
                    unsafe_allow_html=True)
        top_risk = route_df.nlargest(12, "avg_delay_risk")
        fig = px.bar(
            top_risk, x="avg_delay_risk", y="route",
            orientation="h",
            color="avg_delay_risk",
            color_continuous_scale=["#4ade80", "#fde047", "#f87171"],
            range_color=[0, 100],
            labels={"avg_delay_risk": "Avg Delay Risk Score", "route": ""},
            **PLOTLY_THEME
        )
        fig.update_layout(height=380, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # Globe route map
    st.markdown('<div class="section-header">🗺️ Route Network Map</div>',
                unsafe_allow_html=True)

    airport_df = db_query("SELECT * FROM flights LIMIT 1")
    top_routes = route_df.nlargest(20, "total_flights")

    fig = go.Figure()

    # Airport dots
    for code, (lat, lon, city) in AIRPORT_COORDS.items():
        fig.add_trace(go.Scattergeo(
            lon=[lon], lat=[lat],
            mode="markers+text",
            marker=dict(size=10, color="#818cf8",
                        line=dict(color="#c084fc", width=1)),
            text=[code], textposition="top right",
            textfont=dict(size=9, color="#e2e8f0"),
            name=city, showlegend=False,
            hovertemplate=f"<b>{code}</b><br>{city}<extra></extra>"
        ))

    # Route arcs
    for _, row in top_routes.iterrows():
        o, d = row["origin"], row["destination"]
        if o in AIRPORT_COORDS and d in AIRPORT_COORDS:
            o_lat, o_lon, _ = AIRPORT_COORDS[o]
            d_lat, d_lon, _ = AIRPORT_COORDS[d]
            intensity = min(1.0, row["total_flights"] / top_routes["total_flights"].max())
            fig.add_trace(go.Scattergeo(
                lon=[o_lon, d_lon], lat=[o_lat, d_lat],
                mode="lines",
                line=dict(width=max(1, int(intensity * 3)),
                          color=f"rgba(129,140,248,{0.3 + intensity * 0.5})"),
                showlegend=False,
                hovertemplate=f"{o} → {d}<br>Flights: {row['total_flights']}<extra></extra>"
            ))

    fig.update_layout(
        height=500,
        geo=dict(
            showland=True, landcolor="#1e293b",
            showocean=True, oceancolor="#0f172a",
            showlakes=True, lakecolor="#0f172a",
            showcountries=True, countrycolor="#334155",
            showcoastlines=True, coastlinecolor="#475569",
            bgcolor="rgba(0,0,0,0)",
            projection_type="natural earth",
        ),
        **PLOTLY_THEME
    )
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Live Events
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📡 Live Events":
    st.markdown("""
    <h1 style='font-size:1.8rem; font-weight:800; color:#e2e8f0; margin:0 0 4px 0;'>
        📡 Live Flight Events
    </h1>
    <p style='color:#64748b; margin-bottom:16px;'>Kafka event stream — real-time flight status updates</p>
    """, unsafe_allow_html=True)

    kafka_summary = load_kafka_events()
    events_file   = os.path.join(ANALYTICS_DIR, "flight_events.jsonl")

    if not kafka_summary:
        st.info("No Kafka events found. Run the pipeline to generate events.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("📊 Total Events",    kafka_summary.get("total_events", 0))
        k2.metric("⏰ Delayed Flights", kafka_summary.get("delayed_flights", 0))
        k3.metric("❌ Cancelled",        kafka_summary.get("cancelled_flights", 0))
        k4.metric("⏱️ Avg Delay",       f"{kafka_summary.get('avg_delay_minutes', 0):.1f} min")

        # Event type breakdown
        st.markdown('<div class="section-header">📊 Events by Type</div>',
                    unsafe_allow_html=True)

        ebt = kafka_summary.get("events_by_type", {})
        if ebt:
            df_events = pd.DataFrame(list(ebt.items()), columns=["Event Type", "Count"])
            df_events = df_events[df_events["Count"] > 0].sort_values("Count", ascending=True)

            color_map = {
                "FLIGHT_DELAYED":   "#f87171",
                "FLIGHT_CANCELLED": "#ef4444",
                "GATE_CHANGED":     "#fde047",
                "CARGO_LOADED":     "#4ade80",
                "CARGO_OFFLOADED":  "#fb923c",
                "FLIGHT_DEPARTED":  "#818cf8",
                "FLIGHT_ARRIVED":   "#34d399",
            }

            fig = px.bar(
                df_events, x="Count", y="Event Type",
                orientation="h",
                color="Event Type",
                color_discrete_map=color_map,
                **PLOTLY_THEME
            )
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Raw events viewer
        if os.path.exists(events_file):
            with st.expander("📋 Raw Event Stream (last 50 events)", expanded=False):
                events = []
                with open(events_file) as f:
                    for line in f:
                        try:
                            events.append(json.loads(line.strip()))
                        except Exception:
                            pass
                if events:
                    df_raw = pd.DataFrame(events[-50:])
                    st.dataframe(df_raw, hide_index=True, use_container_width=True)

    # Architecture note
    st.markdown("""
    <div style='margin-top:24px; padding:16px; background:rgba(99,102,241,0.1);
                border-radius:12px; border:1px solid rgba(99,102,241,0.3);'>
        <h4 style='color:#818cf8; margin:0 0 8px 0;'>📡 Kafka Integration Note</h4>
        <p style='color:#94a3b8; margin:0; font-size:0.85rem;'>
            In production, Apache Kafka runs as a distributed message broker.
            This project simulates Kafka events and writes them to a local JSONL file
            when a Kafka cluster is not available. The event schema, producer, and consumer
            logic are fully implemented and ready to connect to a real Kafka cluster
            by updating the <code>KAFKA_BOOTSTRAP</code> config in <code>kafka_simulator.py</code>.
        </p>
        <br>
        <p style='color:#94a3b8; margin:0; font-size:0.85rem;'>
            <strong>Topic:</strong> <code>aviation.flight.updates</code><br>
            <strong>Events:</strong> FLIGHT_DELAYED · FLIGHT_CANCELLED · CARGO_LOADED ·
                                     CARGO_OFFLOADED · FLIGHT_DEPARTED · FLIGHT_ARRIVED
        </p>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Scalability Benchmarks
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Scalability Benchmarks":
    st.markdown("""
    <h1 style='font-size:1.8rem; font-weight:800; color:#e2e8f0; margin:0 0 4px 0;'>
        📊 Scalability Benchmarks
    </h1>
    <p style='color:#64748b; margin-bottom:16px;'>Pandas vs PySpark performance at increasing dataset sizes</p>
    """, unsafe_allow_html=True)

    bench_df = load_scalability()

    if bench_df.empty:
        st.info("No benchmark results found. Run the pipeline first.")
        st.stop()

    # Convert types
    bench_df["dataset_size"]        = pd.to_numeric(bench_df["dataset_size"],        errors="coerce")
    bench_df["processing_time_sec"] = pd.to_numeric(bench_df["processing_time_sec"], errors="coerce")
    bench_df["throughput_rps"]      = pd.to_numeric(bench_df["throughput_rps"],       errors="coerce")
    bench_df["memory_mb"]           = pd.to_numeric(bench_df["memory_mb"],            errors="coerce")

    valid = bench_df.dropna(subset=["processing_time_sec"])

    m1, m2, m3 = st.columns(3)

    pandas_rows = valid[valid["engine"] == "Pandas"]
    spark_rows  = valid[valid["engine"] == "PySpark"]

    if not pandas_rows.empty:
        m1.metric("🐼 Max Pandas Throughput",
                  f"{pandas_rows['throughput_rps'].max():,.0f} rows/s")
    if not spark_rows.empty:
        m2.metric("⚡ Max Spark Throughput",
                  f"{spark_rows['throughput_rps'].max():,.0f} rows/s")
    if not pandas_rows.empty and not spark_rows.empty:
        # Compare at largest common size
        sizes = set(pandas_rows["dataset_size"]) & set(spark_rows["dataset_size"])
        if sizes:
            sz = max(sizes)
            p_time = pandas_rows[pandas_rows["dataset_size"] == sz]["processing_time_sec"].mean()
            s_time = spark_rows[spark_rows["dataset_size"] == sz]["processing_time_sec"].mean()
            if s_time > 0:
                ratio = p_time / s_time
                m3.metric(f"⚡ Spark Speedup at {sz:,}",
                          f"{ratio:.2f}x",
                          delta=f"vs Pandas at {sz:,} rows")

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header">⏱️ Processing Time</div>',
                    unsafe_allow_html=True)
        fig = px.line(
            valid, x="dataset_size", y="processing_time_sec",
            color="engine",
            markers=True,
            color_discrete_map={"Pandas": "#fde047", "PySpark": "#818cf8"},
            labels={"dataset_size": "Dataset Size (rows)",
                    "processing_time_sec": "Processing Time (sec)",
                    "engine": "Engine"},
            log_x=True,
            **PLOTLY_THEME
        )
        fig.update_traces(line_width=3, marker_size=8)
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">🚀 Throughput (rows/sec)</div>',
                    unsafe_allow_html=True)
        fig = px.line(
            valid, x="dataset_size", y="throughput_rps",
            color="engine",
            markers=True,
            color_discrete_map={"Pandas": "#fde047", "PySpark": "#818cf8"},
            labels={"dataset_size": "Dataset Size (rows)",
                    "throughput_rps": "Rows/Second",
                    "engine": "Engine"},
            log_x=True,
            **PLOTLY_THEME
        )
        fig.update_traces(line_width=3, marker_size=8)
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    # Memory chart
    st.markdown('<div class="section-header">💾 Peak Memory Usage</div>',
                unsafe_allow_html=True)
    mem_valid = valid.dropna(subset=["memory_mb"])
    if not mem_valid.empty:
        fig = px.bar(
            mem_valid, x="dataset_size", y="memory_mb",
            color="engine", barmode="group",
            color_discrete_map={"Pandas": "#fde047", "PySpark": "#818cf8"},
            labels={"dataset_size": "Dataset Size", "memory_mb": "Peak Memory (MB)"},
            **PLOTLY_THEME
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Raw table
    with st.expander("📋 Raw Benchmark Results", expanded=False):
        st.dataframe(valid.round(4), hide_index=True, use_container_width=True)

    # Interpretation
    st.markdown("""
    <div style='padding:16px; background:rgba(99,102,241,0.1); border-radius:12px;
                border:1px solid rgba(99,102,241,0.3); margin-top:12px;'>
        <h4 style='color:#818cf8; margin:0 0 8px 0;'>🔍 Analysis</h4>
        <ul style='color:#94a3b8; margin:0; font-size:0.85rem; line-height:1.8;'>
            <li><strong>Pandas</strong> is faster on small datasets (&lt;50K) due to zero startup overhead.</li>
            <li><strong>PySpark</strong> has a startup JVM overhead of ~10-20s, making it slower at small scale.</li>
            <li>At <strong>100K+ rows</strong>, Spark's parallelism starts compensating for overhead.</li>
            <li>At <strong>500K+ rows</strong>, Spark significantly outperforms Pandas on multi-core machines.</li>
            <li>In production with a <strong>cluster</strong>, Spark scales to billions of records.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Architecture
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🏗️ Architecture":
    st.markdown("""
    <h1 style='font-size:1.8rem; font-weight:800; color:#e2e8f0; margin:0 0 4px 0;'>
        🏗️ System Architecture
    </h1>
    <p style='color:#64748b; margin-bottom:16px;'>Big Data Architecture — components, data flow, and technology choices</p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:rgba(15,23,42,0.9); border:1px solid rgba(99,102,241,0.4);
                border-radius:16px; padding:24px; font-family: monospace; font-size:0.85rem;
                color:#94a3b8; line-height:1.8;'>
        <pre style='margin:0; color:#94a3b8;'>
┌─────────────────────────────────────────────────────────────────┐
│                    DATA SOURCES LAYER                           │
│  📄 shipments.csv   ✈️ flights.csv   🌤️ weather.csv             │
│  🏢 airports.csv    🔗 connections.csv                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    INGESTION LAYER                              │
│  📡 Apache Kafka (aviation.flight.updates topic)                │
│     ├── FLIGHT_DELAYED  ├── CARGO_LOADED  ├── FLIGHT_DEPARTED   │
│     └── FLIGHT_CANCELLED └── CARGO_OFFLOADED └── FLIGHT_ARRIVED│
│  📂 CSV Batch Ingestion (Apache Spark reader)                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│               DISTRIBUTED PROCESSING LAYER                      │
│                  ⚡ Apache Spark / PySpark                       │
│  ┌───────────────┐    ┌───────────────┐    ┌─────────────────┐  │
│  │ Data Cleaning │    │Feature Eng.   │    │ Dataset Joins   │  │
│  │ - Null removal│    │- Delay risk   │    │ Shipments ×     │  │
│  │ - Dedup       │    │- Priority     │    │ Flights ×       │  │
│  │ - Type cast   │    │  scoring      │    │ Airports ×      │  │
│  │ - Validation  │    │- Urgency calc │    │ Weather         │  │
│  └───────────────┘    └───────────────┘    └─────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    DATA LAKE LAYER                              │
│  📁 data/raw/       ← Original CSVs (Bronze Layer)              │
│  📁 data/processed/ ← Cleaned Parquet (Silver Layer)            │
│  📁 data/analytics/ ← Aggregated Parquet (Gold Layer)           │
│       Partitioned by: origin airport                            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                  OPTIMIZATION LAYER                             │
│  🔧 Google OR-Tools — Mixed Integer Programming                 │
│                                                                 │
│  Objective:  Minimize ΣΣ cost_ij × x_ij + Σ penalty_i × u_i   │
│  Subject to: Σ weight_i × x_ij ≤ capacity_j   (per flight)     │
│              Σ x_ij + u_i = 1                  (per shipment)   │
│              x_ij ∈ {0,1},  u_i ∈ {0,1}                        │
│                                                                 │
│  Fallback: Greedy priority-weighted allocation                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                   STORAGE LAYER                                 │
│  🗄️ SQLite Database (aviation_cargo.db)                         │
│     Tables: shipments · flights · allocation ·                  │
│             flight_utilization · route_stats · kafka_events     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                 PRESENTATION LAYER                              │
│  📊 Streamlit Dashboard + Plotly Visualizations                 │
│     Sections: Overview · Flights · Allocation ·                 │
│               Routes · Events · Benchmarks · Architecture       │
└─────────────────────────────────────────────────────────────────┘
        </pre>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Technology justification table
    st.markdown('<div class="section-header">🛠️ Technology Decisions</div>',
                unsafe_allow_html=True)

    tech_data = pd.DataFrame([
        ["Apache Kafka",   "Event Ingestion",       "Distributed pub-sub for real-time flight events; persistent, replayable stream"],
        ["Apache Spark",   "Distributed Processing","Handles 100K–10M+ records; columnar processing; fault-tolerant execution"],
        ["Parquet",        "Data Lake Storage",     "Columnar format; 4-10× compression vs CSV; predicate pushdown; Spark-native"],
        ["Google OR-Tools","Optimization Engine",   "Production-grade MIP solver; SCIP backend; optimal cargo assignments in seconds"],
        ["SQLite",         "Analytics Database",    "Zero-config relational DB; fast reads; sufficient for dashboard queries"],
        ["Streamlit",      "Dashboard",             "Rapid data app development; built-in caching; native Plotly integration"],
        ["Plotly",         "Visualizations",        "Interactive charts; dark theme; geo maps; responsive to user filters"],
        ["PyArrow",        "Parquet I/O",           "High-performance columnar I/O; bridges Pandas/Spark data formats"],
    ], columns=["Technology", "Component", "Justification"])

    st.dataframe(tech_data, hide_index=True, use_container_width=True)

    # Big Data characteristics
    st.markdown('<div class="section-header">📐 Big Data Characteristics</div>',
                unsafe_allow_html=True)

    v_data = pd.DataFrame([
        ["Volume",      "✅ Demonstrated",
         "10,000–500,000 shipment records; 420–700 flights over 7 days"],
        ["Velocity",    "✅ Demonstrated",
         "Kafka event stream simulating 500+ real-time flight status events"],
        ["Variety",     "✅ Demonstrated",
         "CSV files + JSON events + Parquet analytics + SQLite relational data"],
        ["Veracity",    "✅ Demonstrated",
         "Data validation, null handling, deduplication, type enforcement"],
        ["Value",       "✅ Demonstrated",
         "Cargo cost reduced via MIP optimization; capacity utilization maximized"],
        ["Scalability", "✅ Benchmarked",
         "Pandas vs Spark comparison at 10K / 100K / 500K rows with metrics"],
    ], columns=["Characteristic", "Status", "How It's Demonstrated"])

    st.dataframe(v_data, hide_index=True, use_container_width=True)
