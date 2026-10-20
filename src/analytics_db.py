"""
Phase 5: Analytics Database
=============================
Loads all processed & optimized results into SQLite for structured queries.
The Streamlit dashboard queries this DB for fast retrieval.

Tables created:
  - shipments       : Cleaned shipment records
  - flights         : Cleaned flight records with delay risk
  - allocation      : Cargo assignment results
  - flight_util     : Flight capacity utilization
  - route_stats     : Per-route aggregated metrics
  - kafka_events    : Flight status events summary
  - perf_metrics    : Scalability benchmark results
"""

import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
ANALYTICS_DIR = os.path.join(BASE_DIR, "data", "analytics")
DB_PATH       = os.path.join(ANALYTICS_DIR, "aviation_cargo.db")


# ─────────────────────────────────────────────────────────────────────────────
# Connection Helper
# ─────────────────────────────────────────────────────────────────────────────
def get_connection() -> sqlite3.Connection:
    os.makedirs(ANALYTICS_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# Schema Setup
# ─────────────────────────────────────────────────────────────────────────────
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS shipments (
    shipment_id        TEXT PRIMARY KEY,
    origin             TEXT,
    destination        TEXT,
    cargo_type         TEXT,
    weight_kg          REAL,
    volume_m3          REAL,
    priority           TEXT,
    priority_score     INTEGER,
    deadline           TEXT,
    hours_until_deadline REAL,
    declared_value_usd REAL,
    special_handling   INTEGER,
    status             TEXT,
    urgency_score      REAL
);

CREATE TABLE IF NOT EXISTS flights (
    flight_id              TEXT PRIMARY KEY,
    airline                TEXT,
    aircraft_type          TEXT,
    origin                 TEXT,
    destination            TEXT,
    departure_time         TEXT,
    arrival_time           TEXT,
    flight_date            TEXT,
    cargo_capacity_kg      REAL,
    cost_per_kg            REAL,
    effective_cost_per_kg  REAL,
    delay_risk_score       REAL,
    on_time_history_pct    REAL,
    distance_km            REAL,
    duration_hours         REAL,
    origin_weather_delay   REAL,
    status                 TEXT
);

CREATE TABLE IF NOT EXISTS allocation (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    shipment_id      TEXT,
    flight_id        TEXT,
    origin           TEXT,
    destination      TEXT,
    weight_kg        REAL,
    priority         TEXT,
    cargo_type       TEXT,
    airline          TEXT,
    departure_time   TEXT,
    arrival_time     TEXT,
    cost_usd         REAL,
    delay_risk_score REAL,
    on_time_pct      REAL,
    allocation_method TEXT
);

CREATE TABLE IF NOT EXISTS flight_utilization (
    flight_id          TEXT PRIMARY KEY,
    airline            TEXT,
    origin             TEXT,
    destination        TEXT,
    cargo_capacity_kg  REAL,
    loaded_kg          REAL,
    remaining_kg       REAL,
    utilization_pct    REAL,
    shipment_count     INTEGER
);

CREATE TABLE IF NOT EXISTS route_stats (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    origin          TEXT,
    destination     TEXT,
    total_flights   INTEGER,
    total_capacity_kg REAL,
    avg_cost_per_kg REAL,
    avg_delay_risk  REAL,
    avg_on_time_pct REAL
);

CREATE TABLE IF NOT EXISTS kafka_events (
    event_type      TEXT,
    event_count     INTEGER,
    total_events    INTEGER,
    delayed_flights INTEGER,
    cancelled_flights INTEGER,
    avg_delay_minutes REAL,
    generated_at    TEXT
);

CREATE TABLE IF NOT EXISTS perf_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_size    INTEGER,
    engine          TEXT,
    processing_time_sec REAL,
    throughput_rps  REAL,
    memory_mb       REAL,
    run_at          TEXT
);

CREATE TABLE IF NOT EXISTS optimization_metrics (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    method           TEXT,
    solve_time_sec   REAL,
    total_cost_usd   REAL,
    n_allocated      INTEGER,
    n_unallocated    INTEGER,
    allocation_rate  REAL,
    avg_utilization  REAL,
    generated_at     TEXT
);
"""


def create_schema(conn: sqlite3.Connection):
    conn.executescript(SCHEMA_SQL)
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────
def _safe_parquet(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception as e:
            print(f"  ⚠  Could not read {path}: {e}")
    return pd.DataFrame()


def _safe_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"  ⚠  Could not read {path}: {e}")
    return pd.DataFrame()


def _safe_json(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _upsert_df(conn: sqlite3.Connection, df: pd.DataFrame, table: str, replace: bool = True):
    if df.empty:
        return
    # Convert boolean to int for SQLite compatibility
    for col in df.select_dtypes(include="bool").columns:
        df[col] = df[col].astype(int)
    # Fill NaN
    df = df.where(pd.notna(df), None)
    mode = "replace" if replace else "append"
    df.to_sql(table, conn, if_exists=mode, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Main DB Builder
# ─────────────────────────────────────────────────────────────────────────────
def build_database(verbose: bool = True):
    conn = get_connection()
    create_schema(conn)

    # ── Shipments ─────────────────────────────────────────────────────────────
    ship = _safe_parquet(os.path.join(PROCESSED_DIR, "shipments_clean.parquet"))
    if not ship.empty:
        cols = ["shipment_id", "origin", "destination", "cargo_type", "weight_kg",
                "volume_m3", "priority", "priority_score", "deadline",
                "hours_until_deadline", "declared_value_usd", "special_handling",
                "status", "urgency_score"]
        ship_sub = ship[[c for c in cols if c in ship.columns]].copy()
        conn.execute("DELETE FROM shipments")
        _upsert_df(conn, ship_sub, "shipments", replace=False)
        if verbose:
            print(f"  ✓  Shipments loaded into DB: {len(ship_sub):,}")

    # ── Flights ────────────────────────────────────────────────────────────────
    fl = _safe_parquet(os.path.join(PROCESSED_DIR, "flights_clean.parquet"))
    if not fl.empty:
        cols = ["flight_id", "airline", "aircraft_type", "origin", "destination",
                "departure_time", "arrival_time", "flight_date", "cargo_capacity_kg",
                "cost_per_kg", "effective_cost_per_kg", "delay_risk_score",
                "on_time_history_pct", "distance_km", "duration_hours",
                "origin_weather_delay", "status"]
        fl_sub = fl[[c for c in cols if c in fl.columns]].copy()
        conn.execute("DELETE FROM flights")
        _upsert_df(conn, fl_sub, "flights", replace=False)
        if verbose:
            print(f"  ✓  Flights loaded into DB:   {len(fl_sub):,}")

    # ── Allocation Results ─────────────────────────────────────────────────────
    alloc = _safe_csv(os.path.join(ANALYTICS_DIR, "allocation_results.csv"))
    if not alloc.empty:
        conn.execute("DELETE FROM allocation")
        _upsert_df(conn, alloc, "allocation", replace=False)
        if verbose:
            print(f"  ✓  Allocations loaded:       {len(alloc):,}")

    # ── Flight Utilization ─────────────────────────────────────────────────────
    util = _safe_csv(os.path.join(ANALYTICS_DIR, "flight_utilization.csv"))
    if not util.empty:
        conn.execute("DELETE FROM flight_utilization")
        _upsert_df(conn, util, "flight_utilization", replace=False)

    # ── Route Stats ────────────────────────────────────────────────────────────
    routes = _safe_parquet(os.path.join(ANALYTICS_DIR, "route_stats.parquet"))
    if not routes.empty:
        conn.execute("DELETE FROM route_stats")
        _upsert_df(conn, routes, "route_stats", replace=False)

    # ── Kafka Events ──────────────────────────────────────────────────────────
    kafka = _safe_json(os.path.join(ANALYTICS_DIR, "kafka_summary.json"))
    if kafka:
        import datetime
        rows = []
        for etype, count in kafka.get("events_by_type", {}).items():
            rows.append({
                "event_type":       etype,
                "event_count":      count,
                "total_events":     kafka.get("total_events", 0),
                "delayed_flights":  kafka.get("delayed_flights", 0),
                "cancelled_flights": kafka.get("cancelled_flights", 0),
                "avg_delay_minutes": kafka.get("avg_delay_minutes", 0),
                "generated_at":     datetime.datetime.now().isoformat(),
            })
        if rows:
            conn.execute("DELETE FROM kafka_events")
            _upsert_df(conn, pd.DataFrame(rows), "kafka_events", replace=False)

    # ── Optimization Metrics ──────────────────────────────────────────────────
    opt = _safe_json(os.path.join(ANALYTICS_DIR, "optimization_metrics.json"))
    if opt:
        conn.execute("DELETE FROM optimization_metrics")
        util_val = 0
        util_df  = _safe_csv(os.path.join(ANALYTICS_DIR, "flight_utilization.csv"))
        if not util_df.empty and "utilization_pct" in util_df.columns:
            util_val = float(util_df["utilization_pct"].mean())
        row = {
            "method":          opt.get("method", ""),
            "solve_time_sec":  opt.get("solve_time_sec", 0),
            "total_cost_usd":  opt.get("total_cost_usd", 0),
            "n_allocated":     opt.get("n_allocated", 0),
            "n_unallocated":   opt.get("n_unallocated", 0),
            "allocation_rate": opt.get("allocation_rate", 0),
            "avg_utilization": util_val,
            "generated_at":    opt.get("generated_at", ""),
        }
        _upsert_df(conn, pd.DataFrame([row]), "optimization_metrics", replace=False)

    conn.commit()
    conn.close()
    if verbose:
        print(f"  ✓  Database built at: {DB_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# Query Helpers (used by dashboard)
# ─────────────────────────────────────────────────────────────────────────────
def query(sql: str, params=()) -> pd.DataFrame:
    conn = get_connection()
    try:
        df = pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()
    return df


def get_kpis() -> dict:
    conn = get_connection()
    try:
        ship_count = conn.execute("SELECT COUNT(*) FROM shipments").fetchone()[0]
        alloc_count = conn.execute("SELECT COUNT(*) FROM allocation").fetchone()[0]
        total_cost  = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM allocation").fetchone()[0]
        avg_util    = conn.execute(
            "SELECT COALESCE(AVG(utilization_pct), 0) FROM flight_utilization").fetchone()[0]
        delayed     = conn.execute(
            "SELECT COALESCE(SUM(delayed_flights), 0) FROM kafka_events LIMIT 1").fetchone()[0]
        cancelled   = conn.execute(
            "SELECT COALESCE(SUM(cancelled_flights), 0) FROM kafka_events LIMIT 1").fetchone()[0]
    finally:
        conn.close()

    alloc_rate = (alloc_count / ship_count * 100) if ship_count > 0 else 0

    return {
        "total_shipments":    ship_count,
        "allocated":          alloc_count,
        "allocation_rate":    round(alloc_rate, 1),
        "total_cost_usd":     round(total_cost, 2),
        "avg_utilization":    round(avg_util, 1),
        "delayed_flights":    delayed,
        "cancelled_flights":  cancelled,
    }


def upsert_perf_metric(size: int, engine: str, time_sec: float,
                        throughput: float, memory_mb: float):
    import datetime
    row = {
        "dataset_size":       size,
        "engine":             engine,
        "processing_time_sec": round(time_sec, 3),
        "throughput_rps":     round(throughput, 1),
        "memory_mb":          round(memory_mb, 1),
        "run_at":             datetime.datetime.now().isoformat(),
    }
    conn = get_connection()
    _upsert_df(conn, pd.DataFrame([row]), "perf_metrics", replace=False)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    print("\n🗄️   Aviation Cargo — Analytics Database Builder")
    print("=" * 50)
    build_database()
