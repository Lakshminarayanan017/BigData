"""
Phase 2: Apache Spark Big Data Pipeline
========================================
Reads raw CSVs, cleans & validates data, joins datasets,
computes analytics features, and writes Parquet to the data lake.

Stages:
  1. Raw Ingestion       — load CSVs from data/raw/
  2. Data Cleaning       — nulls, type casting, dedup, validation
  3. Feature Engineering — capacity metrics, cost estimates, delay risk
  4. Dataset Joins       — shipments ↔ flights ↔ airports ↔ weather
  5. Analytics           — aggregations per route, per flight, per day
  6. Data Lake Write     — save Parquet to data/processed/ & data/analytics/
"""

import os
import sys
import time
from datetime import datetime

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        StructType, StructField,
        StringType, DoubleType, IntegerType, BooleanType, TimestampType
    )
    from pyspark.sql.window import Window
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
ANALYTICS_DIR = os.path.join(BASE_DIR, "data", "analytics")


# ─────────────────────────────────────────────────────────────────────────────
# Hadoop & HDFS Cluster Configuration
# ─────────────────────────────────────────────────────────────────────────────
HADOOP_NAMENODE    = os.getenv("HADOOP_NAMENODE", "hdfs://localhost:9000")
HADOOP_YARN_MASTER = os.getenv("SPARK_MASTER", "local[*]")  # Set to "yarn" for Hadoop YARN cluster
USE_HDFS           = os.getenv("USE_HDFS", "false").lower() == "true"

# ─────────────────────────────────────────────────────────────────────────────
# Spark Session Factory (with Hadoop / HDFS Integration)
# ─────────────────────────────────────────────────────────────────────────────
def get_spark_session(app_name: str = "AviationCargoOptimization") -> "SparkSession":
    builder = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .master(HADOOP_YARN_MASTER)
    )

    # Connect to Apache Hadoop Distributed File System (HDFS)
    if USE_HDFS:
        builder = (
            builder
            .config("spark.hadoop.fs.defaultFS", HADOOP_NAMENODE)
            .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
        )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 & 2: Ingest + Clean
# ─────────────────────────────────────────────────────────────────────────────
def load_and_clean_shipments(spark, path: str):
    df = spark.read.option("header", True).option("inferSchema", True).csv(path)

    df = (df
          .dropDuplicates(["shipment_id"])
          .filter(F.col("weight_kg").isNotNull() & (F.col("weight_kg") > 0))
          .filter(F.col("origin") != F.col("destination"))
          .filter(F.col("priority").isin("Low", "Medium", "High", "Express"))
          .withColumn("created_at", F.to_timestamp("created_at"))
          .withColumn("deadline",   F.to_timestamp("deadline"))
          .withColumn("weight_kg",  F.round(F.col("weight_kg"), 2))
          .withColumn("priority_score",
                      F.when(F.col("priority") == "Express", 4)
                       .when(F.col("priority") == "High",    3)
                       .when(F.col("priority") == "Medium",  2)
                       .otherwise(1))
          .withColumn("hours_until_deadline",
                      (F.unix_timestamp("deadline") - F.unix_timestamp(F.lit(datetime.now().isoformat()))) / 3600)
          .filter(F.col("hours_until_deadline") > 0)   # drop already-expired
          )

    return df


def load_and_clean_flights(spark, path: str):
    df = spark.read.option("header", True).option("inferSchema", True).csv(path)

    df = (df
          .dropDuplicates(["flight_id"])
          .filter(F.col("cargo_capacity_kg").isNotNull() & (F.col("cargo_capacity_kg") > 0))
          .filter(F.col("origin") != F.col("destination"))
          .withColumn("departure_time", F.to_timestamp("departure_time"))
          .withColumn("arrival_time",   F.to_timestamp("arrival_time"))
          .withColumn("available_capacity_kg", F.col("cargo_capacity_kg"))
          .withColumn("utilization_pct", F.lit(0.0))
          .withColumn("cost_per_kg", F.round(F.col("cost_per_kg"), 2))
          )

    return df


def load_and_clean_airports(spark, path: str):
    df = spark.read.option("header", True).option("inferSchema", True).csv(path)
    return df.dropDuplicates(["airport_code"])


def load_and_clean_weather(spark, path: str):
    df = spark.read.option("header", True).option("inferSchema", True).csv(path)
    df = (df
          .withColumn("start_time", F.to_timestamp("start_time"))
          .withColumn("end_time",   F.to_timestamp("end_time"))
          .withColumn("is_active",
                      (F.to_timestamp(F.lit(datetime.now().isoformat())).between(
                          F.col("start_time"), F.col("end_time"))))
          )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
def engineer_flight_features(flights_df, weather_df):
    """Add delay risk, effective cost, and route features to flights."""

    # Aggregate weather impact per airport
    weather_agg = (weather_df
                   .filter(F.col("is_active") == True)
                   .groupBy("airport_code")
                   .agg(
                       F.max("expected_delay_min").alias("max_weather_delay_min"),
                       F.count("*").alias("active_weather_events"),
                   ))

    # Join weather at origin
    flights = (flights_df
               .join(weather_agg.withColumnRenamed("airport_code", "origin")
                                .withColumnRenamed("max_weather_delay_min", "origin_weather_delay")
                                .withColumnRenamed("active_weather_events", "origin_weather_events"),
                     on="origin", how="left")
               .fillna({"origin_weather_delay": 0, "origin_weather_events": 0})
               )

    # Delay risk score (0-100)
    flights = (flights
               .withColumn("delay_risk_score",
                           F.least(F.lit(100),
                                   F.col("origin_weather_delay") * 0.4 +
                                   (100 - F.col("on_time_history_pct")) * 0.6))
               .withColumn("effective_cost_per_kg",
                           F.col("cost_per_kg") * (1 + F.col("delay_risk_score") / 200))
               .withColumn("route_key",
                           F.concat_ws("_", F.col("origin"), F.col("destination")))
               )

    return flights


def engineer_shipment_features(shipments_df):
    """Add urgency, value density, and compatibility features."""
    return (shipments_df
            .withColumn("value_density",
                        F.col("declared_value_usd") / F.col("weight_kg"))
            .withColumn("urgency_score",
                        F.col("priority_score") * 25 +
                        F.when(F.col("hours_until_deadline") < 6,  50)
                         .when(F.col("hours_until_deadline") < 12, 30)
                         .when(F.col("hours_until_deadline") < 24, 15)
                         .otherwise(0))
            )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Joins
# ─────────────────────────────────────────────────────────────────────────────
def build_cargo_candidates(shipments_df, flights_df, airports_df):
    """
    Join shipments with compatible flights.
    A flight is compatible if:
      - origin matches
      - destination matches
      - departure is before deadline
      - cargo capacity exists
    """

    # Enrich shipments with airport metadata (origin)
    shipments_enriched = (
        shipments_df
        .join(airports_df.select("airport_code", "city", "country")
                         .withColumnRenamed("airport_code", "origin")
                         .withColumnRenamed("city", "origin_city")
                         .withColumnRenamed("country", "origin_country"),
              on="origin", how="left")
        .join(airports_df.select("airport_code", "city", "country")
                         .withColumnRenamed("airport_code", "destination")
                         .withColumnRenamed("city", "dest_city")
                         .withColumnRenamed("country", "dest_country"),
              on="destination", how="left")
    )

    # Build cross-join candidates (shipment × compatible flights)
    candidates = (
        shipments_enriched.alias("s")
        .join(flights_df.alias("f"),
              on=(
                  (F.col("s.origin")      == F.col("f.origin")) &
                  (F.col("s.destination") == F.col("f.destination")) &
                  (F.col("f.departure_time") <= F.col("s.deadline")) &
                  (F.col("f.cargo_capacity_kg") >= F.col("s.weight_kg"))
              ), how="inner")
        .select(
            F.col("s.shipment_id"),
            F.col("f.flight_id"),
            F.col("s.origin"),
            F.col("s.destination"),
            F.col("s.weight_kg"),
            F.col("s.priority"),
            F.col("s.priority_score"),
            F.col("s.urgency_score"),
            F.col("s.cargo_type"),
            F.col("s.deadline"),
            F.col("s.hours_until_deadline"),
            F.col("s.declared_value_usd"),
            F.col("f.airline"),
            F.col("f.departure_time"),
            F.col("f.arrival_time"),
            F.col("f.cargo_capacity_kg"),
            F.col("f.cost_per_kg"),
            F.col("f.effective_cost_per_kg"),
            F.col("f.delay_risk_score"),
            F.col("f.distance_km"),
            F.col("s.origin_city"),
            F.col("s.dest_city"),
            (F.col("s.weight_kg") * F.col("f.cost_per_kg")).alias("estimated_cost_usd"),
        )
    )

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: Analytics Aggregations
# ─────────────────────────────────────────────────────────────────────────────
def compute_route_analytics(flights_df, shipments_df) -> pd.DataFrame:
    route_stats = (
        flights_df
        .groupBy("origin", "destination")
        .agg(
            F.count("*").alias("total_flights"),
            F.sum("cargo_capacity_kg").alias("total_capacity_kg"),
            F.avg("cost_per_kg").alias("avg_cost_per_kg"),
            F.avg("delay_risk_score").alias("avg_delay_risk"),
            F.avg("on_time_history_pct").alias("avg_on_time_pct"),
        )
        .toPandas()
    )
    return route_stats


def compute_flight_analytics(flights_df) -> pd.DataFrame:
    return (
        flights_df
        .select("flight_id", "airline", "origin", "destination",
                "departure_time", "cargo_capacity_kg", "cost_per_kg",
                "delay_risk_score", "on_time_history_pct", "aircraft_type")
        .toPandas()
    )


def compute_shipment_summary(shipments_df) -> pd.DataFrame:
    return (
        shipments_df
        .groupBy("origin", "destination", "priority", "cargo_type")
        .agg(
            F.count("*").alias("shipment_count"),
            F.sum("weight_kg").alias("total_weight_kg"),
            F.avg("weight_kg").alias("avg_weight_kg"),
            F.avg("hours_until_deadline").alias("avg_hours_until_deadline"),
        )
        .toPandas()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6: Data Lake Write
# ─────────────────────────────────────────────────────────────────────────────
def write_parquet(df, path: str, partition_by: list = None):
    os.makedirs(path, exist_ok=True)
    writer = df.write.mode("overwrite")
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.parquet(path)


# ─────────────────────────────────────────────────────────────────────────────
# Pandas Fallback Pipeline (when Spark not available or for small tests)
# ─────────────────────────────────────────────────────────────────────────────
def run_pandas_pipeline():
    print("  ⚠  PySpark not available — running Pandas fallback pipeline")

    shipments = pd.read_csv(os.path.join(RAW_DIR, "shipments.csv"), parse_dates=["created_at", "deadline"])
    flights   = pd.read_csv(os.path.join(RAW_DIR, "flights.csv"),
                             parse_dates=["departure_time", "arrival_time"])
    airports  = pd.read_csv(os.path.join(RAW_DIR, "airports.csv"))
    weather   = pd.read_csv(os.path.join(RAW_DIR, "weather_events.csv"),
                             parse_dates=["start_time", "end_time"])

    now = datetime.now()

    # Clean shipments
    shipments = shipments.dropna(subset=["weight_kg", "origin", "destination"])
    shipments = shipments[shipments["weight_kg"] > 0]
    shipments = shipments[shipments["origin"] != shipments["destination"]]
    shipments["priority_score"] = shipments["priority"].map(
        {"Express": 4, "High": 3, "Medium": 2, "Low": 1})
    shipments["hours_until_deadline"] = (
        pd.to_datetime(shipments["deadline"]) - now).dt.total_seconds() / 3600
    shipments = shipments[shipments["hours_until_deadline"] > 0]

    # Clean flights
    flights = flights.dropna(subset=["cargo_capacity_kg"])
    flights = flights[flights["cargo_capacity_kg"] > 0]

    # Weather delay risk
    active_weather = weather[
        (weather["start_time"] <= now) & (weather["end_time"] >= now)
    ]
    weather_delay = (active_weather.groupby("airport_code")["expected_delay_min"]
                     .max().reset_index()
                     .rename(columns={"expected_delay_min": "origin_weather_delay"}))
    flights = flights.merge(weather_delay.rename(columns={"airport_code": "origin"}),
                            on="origin", how="left")
    flights["origin_weather_delay"] = flights["origin_weather_delay"].fillna(0)
    flights["delay_risk_score"] = np.clip(
        flights["origin_weather_delay"] * 0.4 +
        (100 - flights["on_time_history_pct"]) * 0.6, 0, 100)
    flights["effective_cost_per_kg"] = (
        flights["cost_per_kg"] * (1 + flights["delay_risk_score"] / 200))

    # Save processed
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    shipments.to_parquet(os.path.join(PROCESSED_DIR, "shipments_clean.parquet"), index=False)
    flights.to_parquet(os.path.join(PROCESSED_DIR, "flights_clean.parquet"),     index=False)
    airports.to_parquet(os.path.join(PROCESSED_DIR, "airports_clean.parquet"),   index=False)
    weather.to_parquet(os.path.join(PROCESSED_DIR, "weather_clean.parquet"),     index=False)

    # Route analytics
    route_stats = (flights.groupby(["origin", "destination"])
                   .agg(total_flights=("flight_id", "count"),
                        total_capacity_kg=("cargo_capacity_kg", "sum"),
                        avg_cost_per_kg=("cost_per_kg", "mean"),
                        avg_delay_risk=("delay_risk_score", "mean"),
                        avg_on_time_pct=("on_time_history_pct", "mean"))
                   .reset_index())

    os.makedirs(ANALYTICS_DIR, exist_ok=True)
    route_stats.to_parquet(os.path.join(ANALYTICS_DIR, "route_stats.parquet"), index=False)
    flights.to_parquet(os.path.join(ANALYTICS_DIR, "flight_analytics.parquet"), index=False)

    print(f"  ✓  Shipments processed: {len(shipments):,}")
    print(f"  ✓  Flights processed:   {len(flights):,}")

    return shipments, flights, airports, weather


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline Runner
# ─────────────────────────────────────────────────────────────────────────────
def run_spark_pipeline():
    spark = get_spark_session()
    print("  ✓  Spark session started")

    # Stage 1 & 2: Ingest + Clean
    shipments = load_and_clean_shipments(spark, os.path.join(RAW_DIR, "shipments.csv"))
    flights   = load_and_clean_flights(  spark, os.path.join(RAW_DIR, "flights.csv"))
    airports  = load_and_clean_airports( spark, os.path.join(RAW_DIR, "airports.csv"))
    weather   = load_and_clean_weather(  spark, os.path.join(RAW_DIR, "weather_events.csv"))

    print(f"  ✓  Shipments loaded: {shipments.count():,}")
    print(f"  ✓  Flights loaded:   {flights.count():,}")

    # Stage 3: Feature Engineering
    flights   = engineer_flight_features(flights, weather)
    shipments = engineer_shipment_features(shipments)

    # Stage 4: Build cargo candidates
    candidates = build_cargo_candidates(shipments, flights, airports)
    print(f"  ✓  Cargo candidates: {candidates.count():,}")

    # Stage 5: Analytics
    route_stats = compute_route_analytics(flights, shipments)
    flight_data = compute_flight_analytics(flights)
    ship_summary = compute_shipment_summary(shipments)

    # Stage 6: Write to Data Lake
    write_parquet(shipments,  os.path.join(PROCESSED_DIR, "shipments_clean"),  ["origin"])
    write_parquet(flights,    os.path.join(PROCESSED_DIR, "flights_clean"),    ["origin"])
    write_parquet(candidates, os.path.join(PROCESSED_DIR, "cargo_candidates"), ["origin"])

    os.makedirs(ANALYTICS_DIR, exist_ok=True)
    route_stats.to_parquet( os.path.join(ANALYTICS_DIR, "route_stats.parquet"),    index=False)
    flight_data.to_parquet( os.path.join(ANALYTICS_DIR, "flight_analytics.parquet"), index=False)
    ship_summary.to_parquet(os.path.join(ANALYTICS_DIR, "shipment_summary.parquet"), index=False)

    # Also save processed data as pandas parquet for optimization engine
    shipments.toPandas().to_parquet(os.path.join(PROCESSED_DIR, "shipments_clean.parquet"), index=False)
    flights.toPandas().to_parquet(  os.path.join(PROCESSED_DIR, "flights_clean.parquet"),   index=False)

    spark.stop()
    print("  ✓  Data lake written")


def run_pipeline(verbose: bool = True):
    t0 = time.time()
    if verbose:
        print("\n  Running Big Data pipeline...")

    if SPARK_AVAILABLE:
        try:
            run_spark_pipeline()
        except Exception as e:
            print(f"  ⚠  Spark error ({e}) — falling back to Pandas")
            run_pandas_pipeline()
    else:
        run_pandas_pipeline()

    elapsed = time.time() - t0
    if verbose:
        print(f"  ✓  Pipeline complete in {elapsed:.1f}s")
    return elapsed


if __name__ == "__main__":
    print("\n⚡  Aviation Cargo — Spark Pipeline")
    print("=" * 50)
    run_pipeline()
