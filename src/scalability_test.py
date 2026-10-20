"""
Phase 6: Scalability Benchmarking
====================================
Measures pipeline performance at different dataset sizes.
Compares Pandas vs PySpark throughput and processing time.

Experiments:
  Dataset Size | Engine   | Metric
  ──────────────────────────────────────────────────
  10,000       | Pandas   | Processing time, throughput
  10,000       | Spark    | Processing time, throughput
  100,000      | Pandas   | Processing time, throughput
  100,000      | Spark    | Processing time, throughput
  500,000      | Pandas   | Processing time, throughput
  500,000      | Spark    | Processing time, throughput

Results saved to:
  - data/analytics/scalability_results.csv
  - data/analytics/scalability_results.json
"""

import os
import sys
import time
import json
import tracemalloc
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYTICS_DIR = os.path.join(BASE_DIR, "data", "analytics")

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Data Generator (for benchmarks — no file I/O needed)
# ─────────────────────────────────────────────────────────────────────────────
AIRPORTS = ["MAA", "DEL", "BOM", "BLR", "HYD", "CCU", "SIN", "DXB", "LHR", "FRA"]
PRIORITIES = ["Low", "Medium", "High", "Express"]
CARGO_TYPES = ["General", "Perishable", "Pharmaceutical", "Electronics", "Machinery"]

def make_shipments(n: int) -> pd.DataFrame:
    """Generate synthetic shipment DataFrame of size n."""
    rng = np.random.default_rng(42)
    origins  = rng.choice(AIRPORTS, size=n)
    dests    = rng.choice(AIRPORTS, size=n)
    same     = origins == dests
    dests[same] = np.roll(AIRPORTS, 1)[rng.integers(0, len(AIRPORTS) - 1, size=same.sum())]

    priorities = rng.choice(PRIORITIES, size=n, p=[0.25, 0.40, 0.25, 0.10])
    weights    = np.clip(np.exp(rng.normal(5.5, 1.2, n)), 10, 20000).round(2)
    values     = (weights * rng.uniform(50, 500, n)).round(2)
    base       = datetime(2026, 10, 16)
    deadlines  = [base + timedelta(hours=int(h)) for h in rng.integers(6, 96, n)]

    return pd.DataFrame({
        "shipment_id":        [f"S{i:08d}" for i in range(n)],
        "origin":             origins,
        "destination":        dests,
        "cargo_type":         rng.choice(CARGO_TYPES, size=n),
        "weight_kg":          weights,
        "priority":           priorities,
        "deadline":           deadlines,
        "declared_value_usd": values,
    })


def make_flights(n: int) -> pd.DataFrame:
    """Generate synthetic flight DataFrame of size n."""
    rng = np.random.default_rng(99)
    origins  = rng.choice(AIRPORTS, size=n)
    dests    = rng.choice(AIRPORTS, size=n)
    same     = origins == dests
    dests[same] = np.roll(AIRPORTS, 2)[rng.integers(0, len(AIRPORTS) - 1, size=same.sum())]

    capacities  = rng.integers(10000, 137000, n).astype(float)
    costs       = rng.uniform(1.5, 15.0, n).round(2)
    on_time_pct = rng.uniform(70, 98, n).round(1)

    return pd.DataFrame({
        "flight_id":          [f"FL{i:06d}" for i in range(n)],
        "origin":             origins,
        "destination":        dests,
        "cargo_capacity_kg":  capacities,
        "cost_per_kg":        costs,
        "on_time_history_pct": on_time_pct,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark: Pandas Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_pandas_benchmark(n_shipments: int, n_flights: int = 500) -> dict:
    tracemalloc.start()
    t0 = time.perf_counter()

    # Generate data
    shipments = make_shipments(n_shipments)
    flights   = make_flights(n_flights)

    # Clean shipments
    shipments = shipments.dropna(subset=["weight_kg"])
    shipments = shipments[shipments["weight_kg"] > 0]
    shipments = shipments[shipments["origin"] != shipments["destination"]]
    shipments["priority_score"] = shipments["priority"].map(
        {"Express": 4, "High": 3, "Medium": 2, "Low": 1})
    now = datetime.now()
    shipments["hours_until_deadline"] = (
        pd.to_datetime(shipments["deadline"]) - now).dt.total_seconds() / 3600
    shipments = shipments[shipments["hours_until_deadline"] > 0]

    # Clean flights
    flights["delay_risk_score"] = np.clip(
        (100 - flights["on_time_history_pct"]) * 0.6, 0, 100)
    flights["effective_cost_per_kg"] = (
        flights["cost_per_kg"] * (1 + flights["delay_risk_score"] / 200))

    # Route stats
    route_stats = (flights.groupby(["origin", "destination"])
                   .agg(total_flights=("flight_id", "count"),
                        avg_cost_per_kg=("cost_per_kg", "mean"),
                        avg_delay_risk=("delay_risk_score", "mean"))
                   .reset_index())

    # Shipment summary
    ship_summary = (shipments.groupby(["origin", "destination", "priority"])
                    .agg(count=("shipment_id", "count"),
                         total_weight=("weight_kg", "sum"))
                    .reset_index())

    elapsed = time.perf_counter() - t0
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "dataset_size":       n_shipments,
        "engine":             "Pandas",
        "processing_time_sec": round(elapsed, 4),
        "throughput_rps":     round(n_shipments / elapsed, 1),
        "memory_mb":          round(peak_mem / 1024 / 1024, 2),
        "rows_processed":     len(shipments),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark: PySpark Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_spark_benchmark(n_shipments: int, n_flights: int = 500) -> dict:
    if not SPARK_AVAILABLE:
        return {"engine": "Spark", "dataset_size": n_shipments,
                "processing_time_sec": None, "throughput_rps": None,
                "memory_mb": None, "note": "Spark not installed"}

    spark = (SparkSession.builder
             .appName("AviationBenchmark")
             .config("spark.ui.showConsoleProgress", "false")
             .config("spark.sql.shuffle.partitions", "4")
             .master("local[*]")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    # Generate data via Pandas then convert
    shipments_pd = make_shipments(n_shipments)
    flights_pd   = make_flights(n_flights)

    tracemalloc.start()
    t0 = time.perf_counter()

    try:
        shipments = spark.createDataFrame(shipments_pd)
        flights   = spark.createDataFrame(flights_pd)

        # Clean
        shipments = (shipments
                     .filter(F.col("weight_kg") > 0)
                     .filter(F.col("origin") != F.col("destination"))
                     .withColumn("priority_score",
                                 F.when(F.col("priority") == "Express", 4)
                                  .when(F.col("priority") == "High",    3)
                                  .when(F.col("priority") == "Medium",  2)
                                  .otherwise(1)))

        flights = (flights
                   .withColumn("delay_risk_score",
                               F.least(F.lit(100.0),
                                       (100 - F.col("on_time_history_pct")) * 0.6))
                   .withColumn("effective_cost_per_kg",
                               F.col("cost_per_kg") * (1 + F.col("delay_risk_score") / 200)))

        # Route stats
        route_stats = (flights
                       .groupBy("origin", "destination")
                       .agg(
                           F.count("*").alias("total_flights"),
                           F.avg("cost_per_kg").alias("avg_cost_per_kg"),
                           F.avg("delay_risk_score").alias("avg_delay_risk"),
                       ))

        # Shipment summary
        ship_summary = (shipments
                        .groupBy("origin", "destination", "priority")
                        .agg(
                            F.count("*").alias("count"),
                            F.sum("weight_kg").alias("total_weight"),
                        ))

        # Trigger execution (Spark is lazy)
        n_processed = shipments.count()
        route_count = route_stats.count()
        summary_count = ship_summary.count()

    finally:
        spark.stop()

    elapsed = time.perf_counter() - t0
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "dataset_size":       n_shipments,
        "engine":             "PySpark",
        "processing_time_sec": round(elapsed, 4),
        "throughput_rps":     round(n_shipments / elapsed, 1),
        "memory_mb":          round(peak_mem / 1024 / 1024, 2),
        "rows_processed":     n_processed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Benchmarking Runner
# ─────────────────────────────────────────────────────────────────────────────
def run_scalability_tests(
    sizes: list = None,
    skip_spark: bool = False,
    verbose: bool = True
) -> pd.DataFrame:

    if sizes is None:
        sizes = [10_000, 100_000, 500_000]

    results = []

    for n in sizes:
        if verbose:
            print(f"\n  ── Dataset size: {n:,} ──────────────────────")

        # Pandas
        if verbose:
            print(f"    Running Pandas ...", end=" ", flush=True)
        try:
            r = run_pandas_benchmark(n)
            results.append(r)
            if verbose:
                print(f"✓  {r['processing_time_sec']}s  "
                      f"({r['throughput_rps']:,.0f} rows/s)  "
                      f"{r['memory_mb']} MB")
        except Exception as e:
            print(f"✗  {e}")

        # PySpark
        if not skip_spark and SPARK_AVAILABLE:
            if verbose:
                print(f"    Running PySpark ...", end=" ", flush=True)
            try:
                r = run_spark_benchmark(n)
                results.append(r)
                if verbose:
                    print(f"✓  {r['processing_time_sec']}s  "
                          f"({r['throughput_rps']:,.0f} rows/s)  "
                          f"{r['memory_mb']} MB")
            except Exception as e:
                print(f"✗  {e}")
        elif not skip_spark:
            results.append({
                "dataset_size":        n,
                "engine":              "PySpark",
                "processing_time_sec": None,
                "throughput_rps":      None,
                "memory_mb":           None,
                "note":                "Spark not installed",
            })

    df = pd.DataFrame(results)

    # Save results
    os.makedirs(ANALYTICS_DIR, exist_ok=True)
    df.to_csv( os.path.join(ANALYTICS_DIR, "scalability_results.csv"),  index=False)
    df.to_json(os.path.join(ANALYTICS_DIR, "scalability_results.json"), orient="records", indent=2)

    # Save to DB
    try:
        from src.analytics_db import upsert_perf_metric
        for _, row in df.iterrows():
            if row.get("processing_time_sec"):
                upsert_perf_metric(
                    size=int(row["dataset_size"]),
                    engine=row["engine"],
                    time_sec=float(row["processing_time_sec"]),
                    throughput=float(row["throughput_rps"] or 0),
                    memory_mb=float(row["memory_mb"] or 0),
                )
    except Exception:
        pass

    if verbose:
        print("\n  ── Summary ────────────────────────────────────")
        print(df[["dataset_size", "engine", "processing_time_sec",
                   "throughput_rps", "memory_mb"]].to_string(index=False))

    return df


if __name__ == "__main__":
    print("\n📊  Aviation Cargo — Scalability Benchmarks")
    print("=" * 50)
    run_scalability_tests(sizes=[10_000, 100_000, 500_000])
