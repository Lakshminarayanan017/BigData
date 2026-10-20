"""
Phase 1: Data Generator
=======================
Generates realistic aviation cargo datasets for the Big Data pipeline.
Produces:
  - airports.csv       : Airport metadata
  - flights.csv        : Flight schedules and cargo capacity
  - shipments.csv      : Cargo shipment orders
  - weather_events.csv : Weather disruptions
  - connections.csv    : Multi-leg flight connections
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

random.seed(42)
np.random.seed(42)

# ── Fake company name generator (no Faker dependency) ─────────────────────────
_COMPANY_ADJ  = ["Global", "Pacific", "Premier", "Swift", "Apex", "Stellar",
                 "Horizon", "Allied", "Pinnacle", "Summit", "Delta", "Omega"]
_COMPANY_NOUN = ["Logistics", "Cargo", "Freight", "Trade", "Shipping",
                 "Commerce", "Supply", "Industries", "Enterprises", "Group"]
_COMPANY_SFXS = ["Ltd.", "Inc.", "Corp.", "Pvt. Ltd.", "LLC", "GmbH"]

def _fake_company() -> str:
    return (f"{random.choice(_COMPANY_ADJ)} "
            f"{random.choice(_COMPANY_NOUN)} "
            f"{random.choice(_COMPANY_SFXS)}")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")

AIRPORTS = {
    "MAA": {"name": "Chennai International",    "city": "Chennai",    "country": "India",     "lat": 12.99,  "lon": 80.17},
    "DEL": {"name": "Indira Gandhi International", "city": "Delhi",   "country": "India",     "lat": 28.55,  "lon": 77.09},
    "BOM": {"name": "Chhatrapati Shivaji",       "city": "Mumbai",    "country": "India",     "lat": 19.09,  "lon": 72.87},
    "BLR": {"name": "Kempegowda International",  "city": "Bangalore", "country": "India",     "lat": 13.19,  "lon": 77.70},
    "HYD": {"name": "Rajiv Gandhi International","city": "Hyderabad", "country": "India",     "lat": 17.24,  "lon": 78.43},
    "CCU": {"name": "Netaji Subhas Chandra",     "city": "Kolkata",   "country": "India",     "lat": 22.65,  "lon": 88.45},
    "SIN": {"name": "Changi Airport",            "city": "Singapore", "country": "Singapore", "lat": 1.36,   "lon": 103.99},
    "DXB": {"name": "Dubai International",       "city": "Dubai",     "country": "UAE",       "lat": 25.25,  "lon": 55.36},
    "LHR": {"name": "Heathrow Airport",          "city": "London",    "country": "UK",        "lat": 51.47,  "lon": -0.46},
    "FRA": {"name": "Frankfurt Airport",         "city": "Frankfurt", "country": "Germany",   "lat": 50.03,  "lon": 8.57},
    "BKK": {"name": "Suvarnabhumi Airport",      "city": "Bangkok",   "country": "Thailand",  "lat": 13.69,  "lon": 100.75},
    "KUL": {"name": "Kuala Lumpur International","city": "Kuala Lumpur","country": "Malaysia", "lat": 2.74,   "lon": 101.70},
}

AIRLINES = ["IndigoAir", "AirAsia Cargo", "SpiceJet Cargo", "Emirates SkyCargo",
            "Singapore Airlines Cargo", "Lufthansa Cargo", "British Airways World Cargo"]

CARGO_TYPES = ["General", "Perishable", "Pharmaceutical", "Electronics",
               "Automotive", "Textile", "Machinery", "Dangerous Goods"]

PRIORITY_LEVELS = ["Low", "Medium", "High", "Express"]
PRIORITY_WEIGHTS = [0.25, 0.40, 0.25, 0.10]

BASE_DATE = datetime(2026, 10, 16, 0, 0, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Airport Dataset
# ─────────────────────────────────────────────────────────────────────────────
def generate_airports() -> pd.DataFrame:
    records = []
    for code, info in AIRPORTS.items():
        records.append({
            "airport_code":  code,
            "airport_name":  info["name"],
            "city":          info["city"],
            "country":       info["country"],
            "latitude":      info["lat"],
            "longitude":     info["lon"],
            "handling_capacity_tonnes_per_day": random.randint(200, 2000),
            "hub_type":      random.choice(["Primary Hub", "Secondary Hub", "Regional"]),
        })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Flight Dataset
# ─────────────────────────────────────────────────────────────────────────────
def generate_flights(n_days: int = 7) -> pd.DataFrame:
    airport_codes = list(AIRPORTS.keys())
    records = []
    flight_num = 1000

    for day_offset in range(n_days):
        flight_date = BASE_DATE + timedelta(days=day_offset)
        n_flights = random.randint(60, 100)

        for _ in range(n_flights):
            origin = random.choice(airport_codes)
            dest_pool = [a for a in airport_codes if a != origin]
            destination = random.choice(dest_pool)

            # Flight duration based on rough distance estimate
            distance_km = _haversine(
                AIRPORTS[origin]["lat"], AIRPORTS[origin]["lon"],
                AIRPORTS[destination]["lat"], AIRPORTS[destination]["lon"]
            )
            duration_hrs = max(1.0, distance_km / 800)

            dep_hour = random.randint(0, 23)
            dep_min  = random.choice([0, 15, 30, 45])
            departure = flight_date.replace(hour=dep_hour, minute=dep_min)
            arrival   = departure + timedelta(hours=duration_hrs)

            aircraft_type = random.choice(["B777F", "B747-8F", "A330F", "B737F", "A350F"])
            max_capacity  = {"B777F": 103000, "B747-8F": 137756, "A330F": 70000,
                             "B737F": 20000, "A350F": 111000}[aircraft_type]

            cargo_capacity_kg = int(max_capacity * random.uniform(0.6, 0.95))
            base_cost_per_kg  = _route_cost_per_kg(distance_km)

            records.append({
                "flight_id":         f"FL{flight_num:05d}",
                "airline":           random.choice(AIRLINES),
                "aircraft_type":     aircraft_type,
                "origin":            origin,
                "destination":       destination,
                "departure_time":    departure.isoformat(),
                "arrival_time":      arrival.isoformat(),
                "flight_date":       flight_date.date().isoformat(),
                "distance_km":       round(distance_km, 1),
                "duration_hours":    round(duration_hrs, 2),
                "max_capacity_kg":   max_capacity,
                "cargo_capacity_kg": cargo_capacity_kg,
                "cost_per_kg":       round(base_cost_per_kg, 2),
                "status":            "Scheduled",
                "delay_minutes":     0,
                "on_time_history_pct": round(random.uniform(70, 98), 1),
            })
            flight_num += 1

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Shipment Dataset
# ─────────────────────────────────────────────────────────────────────────────
def generate_shipments(n_shipments: int = 10000) -> pd.DataFrame:
    airport_codes = list(AIRPORTS.keys())
    records = []

    for i in range(1, n_shipments + 1):
        origin = random.choice(airport_codes)
        dest_pool = [a for a in airport_codes if a != origin]
        destination = random.choice(dest_pool)

        priority  = np.random.choice(PRIORITY_LEVELS, p=PRIORITY_WEIGHTS)
        cargo_type = random.choice(CARGO_TYPES)

        # Weight distribution: most shipments are small
        weight_kg = round(np.random.lognormal(mean=5.5, sigma=1.2), 2)
        weight_kg = max(10.0, min(weight_kg, 20000.0))

        volume_m3 = round(weight_kg / random.uniform(100, 400), 3)

        # Deadline: High/Express = shorter window
        deadline_hours = {"Low": 96, "Medium": 48, "High": 24, "Express": 12}[priority]
        offset_hours   = random.randint(0, 48)
        created_at     = BASE_DATE + timedelta(hours=offset_hours)
        deadline       = created_at + timedelta(hours=deadline_hours)

        declared_value = round(weight_kg * random.uniform(50, 500), 2)
        special_handling = cargo_type in ["Perishable", "Pharmaceutical", "Dangerous Goods"]

        records.append({
            "shipment_id":        f"S{i:08d}",
            "origin":             origin,
            "destination":        destination,
            "cargo_type":         cargo_type,
            "weight_kg":          weight_kg,
            "volume_m3":          volume_m3,
            "priority":           priority,
            "created_at":         created_at.isoformat(),
            "deadline":           deadline.isoformat(),
            "declared_value_usd": declared_value,
            "special_handling":   special_handling,
            "shipper":            _fake_company(),
            "consignee":          _fake_company(),
            "status":             "Pending",
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Weather Events Dataset
# ─────────────────────────────────────────────────────────────────────────────
def generate_weather_events(n_events: int = 50) -> pd.DataFrame:
    airport_codes = list(AIRPORTS.keys())
    records = []

    for i in range(1, n_events + 1):
        airport    = random.choice(airport_codes)
        start_hour = random.randint(0, 168)
        event_start = BASE_DATE + timedelta(hours=start_hour)
        duration_hrs = random.randint(2, 24)
        event_end   = event_start + timedelta(hours=duration_hrs)
        severity    = random.choice(["Minor", "Moderate", "Severe"])
        event_type  = random.choice(["Thunderstorm", "Fog", "Cyclone", "High Winds", "Snow", "Hail"])

        delay_impact = {"Minor": (0, 30), "Moderate": (30, 90), "Severe": (90, 360)}[severity]
        expected_delay = random.randint(*delay_impact)

        records.append({
            "event_id":          f"WE{i:05d}",
            "airport_code":      airport,
            "event_type":        event_type,
            "severity":          severity,
            "start_time":        event_start.isoformat(),
            "end_time":          event_end.isoformat(),
            "expected_delay_min": expected_delay,
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Flight Connections (for multi-leg routing)
# ─────────────────────────────────────────────────────────────────────────────
def generate_connections() -> pd.DataFrame:
    airport_codes = list(AIRPORTS.keys())
    records = []
    conn_id = 1

    # Hub airports that act as transit points
    hubs = ["DEL", "BOM", "DXB", "SIN", "FRA"]

    for origin in airport_codes:
        for destination in airport_codes:
            if origin == destination:
                continue
            if random.random() < 0.3:   # only 30% of pairs have a direct connection
                continue
            # Add via-hub connections
            for hub in hubs:
                if hub != origin and hub != destination:
                    records.append({
                        "connection_id":  f"CN{conn_id:06d}",
                        "origin":         origin,
                        "destination":    destination,
                        "via_hub":        hub,
                        "min_transit_hr": round(random.uniform(1.5, 6.0), 1),
                        "extra_cost_pct": round(random.uniform(5, 25), 1),  # % surcharge
                    })
                    conn_id += 1

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────
def _haversine(lat1, lon1, lat2, lon2) -> float:
    """Compute great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _route_cost_per_kg(distance_km: float) -> float:
    """Approximate cost per kg based on distance."""
    if distance_km < 500:
        return random.uniform(1.5, 3.0)
    elif distance_km < 2000:
        return random.uniform(2.5, 5.0)
    elif distance_km < 5000:
        return random.uniform(4.0, 8.0)
    else:
        return random.uniform(7.0, 15.0)


# ─────────────────────────────────────────────────────────────────────────────
# Main Generator
# ─────────────────────────────────────────────────────────────────────────────
def generate_all(n_shipments: int = 10000, n_days: int = 7, verbose: bool = True):
    os.makedirs(DATA_DIR, exist_ok=True)

    datasets = {
        "airports.csv":       (generate_airports,      {}),
        "flights.csv":        (generate_flights,        {"n_days": n_days}),
        "shipments.csv":      (generate_shipments,      {"n_shipments": n_shipments}),
        "weather_events.csv": (generate_weather_events, {}),
        "connections.csv":    (generate_connections,    {}),
    }

    results = {}
    for filename, (fn, kwargs) in datasets.items():
        if verbose:
            print(f"  Generating {filename}...", end=" ", flush=True)
        df = fn(**kwargs)
        path = os.path.join(DATA_DIR, filename)
        df.to_csv(path, index=False)
        results[filename] = df
        if verbose:
            print(f"✓  ({len(df):,} rows)")

    return results


if __name__ == "__main__":
    print("\n🛫  Aviation Cargo — Data Generator")
    print("=" * 50)
    generate_all(n_shipments=10000, n_days=7)
    print("\n✅  All datasets written to data/raw/")
