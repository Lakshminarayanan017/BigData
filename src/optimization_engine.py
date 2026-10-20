"""
Phase 4: Optimization Engine
==============================
Uses Google OR-Tools (CP-SAT / Linear Solver) to optimize cargo allocation.

Problem:
  - Assign shipments to flights
  - Minimize total transportation cost
  - Respect constraints: weight capacity, deadlines, priorities

Outputs:
  - allocation_results.csv  : Which shipment goes on which flight
  - unallocated.csv         : Shipments that could not be assigned
  - optimization_metrics.json : Cost, utilization, on-time statistics
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from ortools.linear_solver import pywraplp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
ANALYTICS_DIR = os.path.join(BASE_DIR, "data", "analytics")

# ─────────────────────────────────────────────────────────────────────────────
# Data Loader
# ─────────────────────────────────────────────────────────────────────────────
def load_optimization_data(max_shipments: int = 200, max_flights: int = 50):
    """
    Load shipments and flights for optimization.
    Limits size for tractable solve times; in production use decomposition.
    """
    ship_path   = os.path.join(PROCESSED_DIR, "shipments_clean.parquet")
    flight_path = os.path.join(PROCESSED_DIR, "flights_clean.parquet")

    if not os.path.exists(ship_path) or not os.path.exists(flight_path):
        raise FileNotFoundError(
            "Processed data not found. Run the pipeline first (run_pipeline.py)."
        )

    shipments = pd.read_parquet(ship_path)
    flights   = pd.read_parquet(flight_path)

    # Filter to a manageable window (next 24 hrs of flights)
    now = pd.Timestamp.now()
    if "departure_time" in flights.columns:
        flights["departure_time"] = pd.to_datetime(flights["departure_time"])
        flights = flights[flights["departure_time"] >= now]

    if "deadline" in shipments.columns:
        shipments["deadline"] = pd.to_datetime(shipments["deadline"])
        shipments = shipments[pd.to_datetime(shipments["deadline"]) > now]

    # Take top-priority shipments first
    if "priority_score" in shipments.columns:
        shipments = shipments.sort_values("priority_score", ascending=False)

    shipments = shipments.head(max_shipments).reset_index(drop=True)
    flights   = flights.head(max_flights).reset_index(drop=True)

    return shipments, flights


# ─────────────────────────────────────────────────────────────────────────────
# Feasibility Check
# ─────────────────────────────────────────────────────────────────────────────
def is_feasible(shipment: pd.Series, flight: pd.Series) -> bool:
    """Check if a shipment can physically go on a flight."""
    # Route match
    if shipment["origin"] != flight["origin"]:
        return False
    if shipment["destination"] != flight["destination"]:
        return False

    # Weight check
    if shipment["weight_kg"] > flight["cargo_capacity_kg"]:
        return False

    # Deadline check
    if "departure_time" in flight.index and "deadline" in shipment.index:
        dep = pd.to_datetime(flight["departure_time"])
        ddl = pd.to_datetime(shipment["deadline"])
        if dep > ddl:
            return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# OR-Tools MIP Optimizer
# ─────────────────────────────────────────────────────────────────────────────
class CargoOptimizer:
    """
    Mixed-Integer Programming model for cargo allocation.

    Decision variables:
      x[i][j] = 1  if shipment i is assigned to flight j, else 0
      u[i]    = 1  if shipment i is UNALLOCATED (penalty term)

    Objective:
      Minimize:
        sum_{i,j} cost_ij * x[i][j]
        + sum_i  penalty_i * u[i]

    Constraints:
      (1) Each shipment assigned to at most one flight
      (2) Each flight's weight capacity not exceeded
      (3) x[i][j] only set if route is feasible (origin, dest, deadline match)
    """

    def __init__(self, shipments: pd.DataFrame, flights: pd.DataFrame):
        self.shipments = shipments.reset_index(drop=True)
        self.flights   = flights.reset_index(drop=True)
        self.n_ship    = len(shipments)
        self.n_flight  = len(flights)
        self.feasible  = self._build_feasibility_matrix()

    def _build_feasibility_matrix(self) -> np.ndarray:
        mat = np.zeros((self.n_ship, self.n_flight), dtype=bool)
        for i, ship in self.shipments.iterrows():
            for j, flight in self.flights.iterrows():
                mat[i][j] = is_feasible(ship, flight)
        return mat

    def solve(self, time_limit_sec: int = 60) -> dict:
        if not ORTOOLS_AVAILABLE:
            print("  ⚠  OR-Tools not available — using greedy fallback")
            return self._greedy_solve()

        solver = pywraplp.Solver.CreateSolver("SCIP")
        if not solver:
            return self._greedy_solve()

        solver.SetTimeLimit(time_limit_sec * 1000)

        # ── Decision variables ──────────────────────────────────────────────
        x = {}
        for i in range(self.n_ship):
            for j in range(self.n_flight):
                if self.feasible[i][j]:
                    x[i, j] = solver.BoolVar(f"x_{i}_{j}")

        u = [solver.BoolVar(f"u_{i}") for i in range(self.n_ship)]

        # ── Objective ────────────────────────────────────────────────────────
        obj = solver.Objective()

        for i, ship in self.shipments.iterrows():
            weight = float(ship["weight_kg"])
            tier_bonus = {"Express": 50000.0, "High": 20000.0, "Medium": 10000.0, "Low": 5000.0}.get(ship.get("priority", "Low"), 5000.0)
            priority_penalty = max(50000.0, weight * 50.0 + tier_bonus)
            obj.SetCoefficient(u[i], priority_penalty)

            for j, flight in self.flights.iterrows():
                if (i, j) in x:
                    cost_ij = (weight *
                               flight.get("effective_cost_per_kg",
                                          flight.get("cost_per_kg", 5.0)))
                    obj.SetCoefficient(x[i, j], cost_ij)

        obj.SetMinimization()

        # ── Constraint 1: Each shipment assigned to at most 1 flight ────────
        # sum_{j} x[i, j] + u[i] == 1 (must be either assigned to flight or marked unallocated)
        for i in range(self.n_ship):
            ct = solver.Constraint(1.0, 1.0)
            for j in range(self.n_flight):
                if (i, j) in x:
                    ct.SetCoefficient(x[i, j], 1.0)
            ct.SetCoefficient(u[i], 1.0)

        # ── Constraint 2: Flight capacity ────────────────────────────────────
        for j, flight in self.flights.iterrows():
            cap = flight["cargo_capacity_kg"]
            ct  = solver.Constraint(0, float(cap))
            for i, ship in self.shipments.iterrows():
                if (i, j) in x:
                    ct.SetCoefficient(x[i, j], float(ship["weight_kg"]))

        # ── Solve ─────────────────────────────────────────────────────────────
        t0     = time.time()
        status = solver.Solve()
        elapsed = time.time() - t0

        if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            return self._extract_solution(x, u, solver.Objective().Value(), elapsed, "OR-Tools MIP")
        else:
            print("  ⚠  MIP solver couldn't find solution — using greedy fallback")
            return self._greedy_solve()

    # ─────────────────────────────────────────────────────────────────────────
    # Greedy Fallback (when OR-Tools unavailable or dataset too large)
    # ─────────────────────────────────────────────────────────────────────────
    def _greedy_solve(self) -> dict:
        """
        Priority-weighted greedy: assign high-priority, low-cost shipments first.
        """
        remaining_cap = self.flights["cargo_capacity_kg"].copy().values.astype(float)
        assignment    = [-1] * self.n_ship   # flight index for each shipment
        total_cost    = 0.0

        # Sort shipments: Express first, then by weight (heavier first for bin packing)
        priority_order = {"Express": 0, "High": 1, "Medium": 2, "Low": 3}
        order = self.shipments.copy()
        order["_pord"] = order["priority"].map(priority_order).fillna(3)
        sorted_idx = order.sort_values(["_pord", "weight_kg"],
                                        ascending=[True, False]).index.tolist()

        for i in sorted_idx:
            ship = self.shipments.iloc[i]
            best_j    = -1
            best_cost = float("inf")

            for j in range(self.n_flight):
                if not self.feasible[i][j]:
                    continue
                if remaining_cap[j] < ship["weight_kg"]:
                    continue

                flight = self.flights.iloc[j]
                cost   = ship["weight_kg"] * flight.get(
                    "effective_cost_per_kg", flight.get("cost_per_kg", 5.0))

                if cost < best_cost:
                    best_cost = cost
                    best_j    = j

            if best_j >= 0:
                assignment[i]        = best_j
                remaining_cap[best_j] -= ship["weight_kg"]
                total_cost           += best_cost

        return self._extract_solution_greedy(assignment, total_cost, "Greedy")

    def _extract_solution(self, x, u, obj_val, elapsed, method) -> dict:
        """Extract OR-Tools solution into a structured result."""
        allocations  = []
        unallocated  = []

        for i, ship in self.shipments.iterrows():
            assigned = False
            for j, flight in self.flights.iterrows():
                if (i, j) in x and x[i, j].solution_value() > 0.5:
                    cost = ship["weight_kg"] * flight.get(
                        "effective_cost_per_kg", flight.get("cost_per_kg", 5.0))
                    allocations.append({
                        "shipment_id":       ship["shipment_id"],
                        "flight_id":         flight["flight_id"],
                        "origin":            ship["origin"],
                        "destination":       ship["destination"],
                        "weight_kg":         ship["weight_kg"],
                        "priority":          ship.get("priority", ""),
                        "cargo_type":        ship.get("cargo_type", ""),
                        "airline":           flight.get("airline", ""),
                        "departure_time":    str(flight.get("departure_time", "")),
                        "arrival_time":      str(flight.get("arrival_time", "")),
                        "cost_usd":          round(cost, 2),
                        "delay_risk_score":  round(float(flight.get("delay_risk_score", 0)), 1),
                        "on_time_pct":       float(flight.get("on_time_history_pct", 0)),
                        "allocation_method": method,
                    })
                    assigned = True
                    break
            if not assigned:
                unallocated.append({
                    "shipment_id": ship["shipment_id"],
                    "origin":      ship["origin"],
                    "destination": ship["destination"],
                    "weight_kg":   ship["weight_kg"],
                    "priority":    ship.get("priority", ""),
                    "reason":      "No feasible flight found within deadline",
                })

        return self._build_result(allocations, unallocated, obj_val, elapsed, method)

    def _extract_solution_greedy(self, assignment, total_cost, method) -> dict:
        allocations = []
        unallocated = []

        for i, j in enumerate(assignment):
            ship   = self.shipments.iloc[i]
            if j >= 0:
                flight = self.flights.iloc[j]
                cost   = ship["weight_kg"] * flight.get(
                    "effective_cost_per_kg", flight.get("cost_per_kg", 5.0))
                allocations.append({
                    "shipment_id":       ship["shipment_id"],
                    "flight_id":         flight["flight_id"],
                    "origin":            ship["origin"],
                    "destination":       ship["destination"],
                    "weight_kg":         ship["weight_kg"],
                    "priority":          ship.get("priority", ""),
                    "cargo_type":        ship.get("cargo_type", ""),
                    "airline":           flight.get("airline", ""),
                    "departure_time":    str(flight.get("departure_time", "")),
                    "arrival_time":      str(flight.get("arrival_time", "")),
                    "cost_usd":          round(cost, 2),
                    "delay_risk_score":  round(float(flight.get("delay_risk_score", 0)), 1),
                    "on_time_pct":       float(flight.get("on_time_history_pct", 0)),
                    "allocation_method": method,
                })
            else:
                unallocated.append({
                    "shipment_id": ship["shipment_id"],
                    "origin":      ship["origin"],
                    "destination": ship["destination"],
                    "weight_kg":   ship["weight_kg"],
                    "priority":    ship.get("priority", ""),
                    "reason":      "No feasible flight (capacity or route mismatch)",
                })

        return self._build_result(allocations, unallocated, total_cost, 0, method)

    def _build_result(self, allocations, unallocated, total_cost, elapsed, method) -> dict:
        alloc_df   = pd.DataFrame(allocations)
        unalloc_df = pd.DataFrame(unallocated)

        n_allocated   = len(allocations)
        n_unallocated = len(unallocated)
        n_total       = n_allocated + n_unallocated

        # Capacity utilization per flight
        flight_utils = []
        if not alloc_df.empty:
            for j, flight in self.flights.iterrows():
                fid  = flight["flight_id"]
                mask = alloc_df["flight_id"] == fid
                loaded_kg = alloc_df.loc[mask, "weight_kg"].sum()
                cap_kg    = flight["cargo_capacity_kg"]
                util_pct  = (loaded_kg / cap_kg * 100) if cap_kg > 0 else 0
                flight_utils.append({
                    "flight_id":        fid,
                    "airline":          flight.get("airline", ""),
                    "origin":           flight["origin"],
                    "destination":      flight["destination"],
                    "cargo_capacity_kg": cap_kg,
                    "loaded_kg":        round(loaded_kg, 2),
                    "remaining_kg":     round(cap_kg - loaded_kg, 2),
                    "utilization_pct":  round(util_pct, 1),
                    "shipment_count":   int(mask.sum()),
                })

        return {
            "method":          method,
            "solve_time_sec":  round(elapsed, 2),
            "total_cost_usd":  round(total_cost, 2),
            "n_allocated":     n_allocated,
            "n_unallocated":   n_unallocated,
            "n_total":         n_total,
            "allocation_rate": round(n_allocated / n_total * 100, 1) if n_total else 0,
            "allocations":     alloc_df,
            "unallocated":     unalloc_df,
            "flight_utilization": pd.DataFrame(flight_utils),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main Runner
# ─────────────────────────────────────────────────────────────────────────────
def run_optimization(max_shipments: int = 200, max_flights: int = 50, verbose: bool = True):
    if verbose:
        print(f"\n  Loading data (max {max_shipments} shipments, {max_flights} flights)...")

    try:
        shipments, flights = load_optimization_data(max_shipments, max_flights)
    except FileNotFoundError as e:
        print(f"  ✗  {e}")
        return None

    if verbose:
        print(f"  ✓  Shipments: {len(shipments)}, Flights: {len(flights)}")
        print(f"  ✓  Running {'OR-Tools MIP' if ORTOOLS_AVAILABLE else 'Greedy'} optimizer...")

    optimizer = CargoOptimizer(shipments, flights)
    result    = optimizer.solve(time_limit_sec=30)

    # ── Save Results ─────────────────────────────────────────────────────────
    os.makedirs(ANALYTICS_DIR, exist_ok=True)

    if not result["allocations"].empty:
        result["allocations"].to_csv(
            os.path.join(ANALYTICS_DIR, "allocation_results.csv"), index=False)
        result["allocations"].to_parquet(
            os.path.join(ANALYTICS_DIR, "allocation_results.parquet"), index=False)

    if not result["unallocated"].empty:
        result["unallocated"].to_csv(
            os.path.join(ANALYTICS_DIR, "unallocated.csv"), index=False)

    if not result["flight_utilization"].empty:
        result["flight_utilization"].to_csv(
            os.path.join(ANALYTICS_DIR, "flight_utilization.csv"), index=False)

    # Metrics summary
    metrics = {
        "method":            result["method"],
        "solve_time_sec":    result["solve_time_sec"],
        "total_cost_usd":    result["total_cost_usd"],
        "n_allocated":       result["n_allocated"],
        "n_unallocated":     result["n_unallocated"],
        "n_total":           result["n_total"],
        "allocation_rate":   result["allocation_rate"],
        "avg_utilization_pct": (
            result["flight_utilization"]["utilization_pct"].mean()
            if not result["flight_utilization"].empty else 0
        ),
        "generated_at":      datetime.now().isoformat(),
    }

    with open(os.path.join(ANALYTICS_DIR, "optimization_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if verbose:
        print(f"\n  ── Optimization Results ──────────────────────")
        print(f"  Method:          {result['method']}")
        print(f"  Solve time:      {result['solve_time_sec']}s")
        print(f"  Total cost:      ${result['total_cost_usd']:,.2f}")
        print(f"  Allocated:       {result['n_allocated']:,} / {result['n_total']:,}")
        print(f"  Allocation rate: {result['allocation_rate']}%")
        if not result["flight_utilization"].empty:
            avg_util = result["flight_utilization"]["utilization_pct"].mean()
            print(f"  Avg utilization: {avg_util:.1f}%")

    return result


if __name__ == "__main__":
    print("\n🔧  Aviation Cargo — Optimization Engine")
    print("=" * 50)
    run_optimization()
