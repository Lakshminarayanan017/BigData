"""
run_pipeline.py — Master Pipeline Runner
==========================================
Runs all phases of the Aviation Cargo Optimization project in sequence.

Phases:
  1. Data Generation     → Generate realistic datasets (CSV)
  2. Big Data Pipeline   → PySpark processing → Parquet data lake
  3. Kafka Simulation    → Real-time flight event stream
  4. Optimization        → OR-Tools cargo allocation
  5. Analytics Database  → Load results into SQLite
  6. Scalability Tests   → Pandas vs Spark benchmarks

Usage:
  python run_pipeline.py               # Run all phases
  python run_pipeline.py --skip-spark  # Skip PySpark (use Pandas only)
  python run_pipeline.py --small       # Small dataset (faster)
  python run_pipeline.py --large       # Large dataset (500K shipments)
  python run_pipeline.py --no-bench    # Skip scalability benchmarks
"""

import sys
import os
import time
import argparse

# Ensure stdout/stderr handles UTF-8 cleanly on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from colorama import Fore, Style, init
init(autoreset=True)

BANNER = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════════╗
║  ✈️   Aviation: Cargo Logistics Optimization for Airlines         ║
║       Big Data Architecture Project                              ║
╚══════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""

STEP_OK   = f"{Fore.GREEN}  ✓{Style.RESET_ALL}"
STEP_WARN = f"{Fore.YELLOW}  ⚠{Style.RESET_ALL}"
STEP_ERR  = f"{Fore.RED}  ✗{Style.RESET_ALL}"
STEP_INFO = f"{Fore.CYAN}  ▶{Style.RESET_ALL}"


def section(title: str):
    print(f"\n{Fore.BLUE}{'─' * 60}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}  {title}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'─' * 60}{Style.RESET_ALL}")


def run_phase(name: str, fn, **kwargs) -> float:
    section(name)
    t0 = time.time()
    try:
        fn(**kwargs)
        elapsed = time.time() - t0
        print(f"{STEP_OK}  Phase complete in {elapsed:.1f}s")
        return elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f"{STEP_ERR}  Phase failed after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return -1


# ─────────────────────────────────────────────────────────────────────────────
# Phase Runners
# ─────────────────────────────────────────────────────────────────────────────
def phase_generate(n_shipments: int, n_days: int):
    from src.data_generator import generate_all
    generate_all(n_shipments=n_shipments, n_days=n_days, verbose=True)


def phase_pipeline():
    from src.spark_pipeline import run_pipeline
    run_pipeline(verbose=True)


def phase_kafka():
    from src.kafka_simulator import simulate_events
    simulate_events(n_events=500, interval_ms=0, verbose=True)


def phase_optimize():
    from src.optimization_engine import run_optimization
    run_optimization(max_shipments=200, max_flights=50, verbose=True)


def phase_database():
    from src.analytics_db import build_database
    build_database(verbose=True)


def phase_scalability(skip_spark: bool = False, sizes=None):
    from src.scalability_test import run_scalability_tests
    if sizes is None:
        sizes = [10_000, 100_000, 500_000]
    run_scalability_tests(sizes=sizes, skip_spark=skip_spark, verbose=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Aviation Cargo Optimization — Big Data Pipeline"
    )
    parser.add_argument("--skip-spark",  action="store_true",
                        help="Use Pandas fallback instead of PySpark")
    parser.add_argument("--small",       action="store_true",
                        help="Small dataset: 1,000 shipments, 3 days")
    parser.add_argument("--large",       action="store_true",
                        help="Large dataset: 100,000 shipments, 14 days")
    parser.add_argument("--no-bench",    action="store_true",
                        help="Skip scalability benchmarks")
    parser.add_argument("--bench-only",  action="store_true",
                        help="Run only scalability benchmarks")
    parser.add_argument("--hdfs",        action="store_true",
                        help="Connect to Hadoop HDFS cluster (hdfs://localhost:9000)")
    args = parser.parse_args()

    if args.hdfs:
        os.environ["USE_HDFS"] = "true"
        print(f"{STEP_INFO}  Hadoop HDFS Mode: ENABLED (Connecting to Hadoop NameNode)")

    print(BANNER)

    # Dataset size configuration
    if args.small:
        n_shipments, n_days = 1_000, 3
        bench_sizes = [1_000, 5_000, 10_000]
        print(f"{STEP_INFO}  Mode: SMALL (1K shipments, 3 days)")
    elif args.large:
        n_shipments, n_days = 100_000, 14
        bench_sizes = [10_000, 100_000, 500_000]
        print(f"{STEP_INFO}  Mode: LARGE (100K shipments, 14 days)")
    else:
        n_shipments, n_days = 10_000, 7
        bench_sizes = [10_000, 100_000, 500_000]
        print(f"{STEP_INFO}  Mode: DEFAULT (10K shipments, 7 days)")

    pipeline_start = time.time()
    timings = {}

    if args.bench_only:
        t = run_phase("📊 Phase 6: Scalability Benchmarks",
                      phase_scalability,
                      skip_spark=args.skip_spark,
                      sizes=bench_sizes)
        timings["Benchmarks"] = t
    else:
        # ── Phase 1 ──
        t = run_phase("📂 Phase 1: Data Generation",
                      phase_generate,
                      n_shipments=n_shipments, n_days=n_days)
        timings["Data Generation"] = t

        # ── Phase 2 ──
        t = run_phase("⚡ Phase 2: Big Data Pipeline (Spark)",
                      phase_pipeline)
        timings["Spark Pipeline"] = t

        # ── Phase 3 ──
        t = run_phase("📡 Phase 3: Kafka Event Simulation",
                      phase_kafka)
        timings["Kafka Simulation"] = t

        # ── Phase 4 ──
        t = run_phase("🔧 Phase 4: Cargo Optimization (OR-Tools)",
                      phase_optimize)
        timings["Optimization"] = t

        # ── Phase 5 ──
        t = run_phase("🗄️  Phase 5: Analytics Database",
                      phase_database)
        timings["Database"] = t

        # ── Phase 6 ──
        if not args.no_bench:
            t = run_phase("📊 Phase 6: Scalability Benchmarks",
                          phase_scalability,
                          skip_spark=args.skip_spark,
                          sizes=bench_sizes)
            timings["Benchmarks"] = t
            +r

    # ── Summary ──────────────────────────────────────────────────────────────
    total_time = time.time() - pipeline_start

    section("📋 Pipeline Summary")
    for phase, t in timings.items():
        status = STEP_OK if t >= 0 else STEP_ERR
        time_str = f"{t:.1f}s" if t >= 0 else "FAILED"
        print(f"{status}  {phase:<25} {time_str}")

    print(f"\n{Fore.GREEN}  Total pipeline time: {total_time:.1f}s{Style.RESET_ALL}")

    print(f"""
{Fore.CYAN}╔═══════════════════════════════════════════════════════╗
║      Pipeline Complete  Launch the dashboard                     ║
║                                                             ║
║     streamlit run src/dashboard.py                          ║
║                                                             ║
║     Outputs:                                                ║
║     data/raw/          → Raw CSV datasets                        ║
║     data/processed/    → Parquet data lake                       ║
║     data/analytics/    → Results, metrics, benchmarks            ║
╚══════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
""")


if __name__ == "__main__":
    main()
