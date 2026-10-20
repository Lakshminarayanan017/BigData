"""
Phase 3: Kafka Event Simulator
================================
Simulates real-time flight status updates via Apache Kafka.
Produces JSON events to the topic: aviation.flight.updates

Events include:
  - FLIGHT_DELAYED   : Flight departure pushed back
  - FLIGHT_CANCELLED : Flight cancelled, shipments need rerouting
  - GATE_CHANGED     : Gate logistics update
  - CARGO_LOADED     : Cargo successfully loaded on flight
  - CARGO_OFFLOADED  : Cargo removed from flight (overweight, priority)
  - FLIGHT_DEPARTED  : Flight has departed
  - FLIGHT_ARRIVED   : Flight landed at destination

If Kafka is not running, events are written to a local JSONL file.
"""

import os
import sys
import json
import time
import random
import threading
import pandas as pd
from datetime import datetime, timedelta

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import NoBrokersAvailable
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
EVENTS_DIR   = os.path.join(BASE_DIR, "data", "analytics")

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC_FLIGHT    = "aviation.flight.updates"
TOPIC_CARGO     = "aviation.cargo.events"

EVENT_TYPES = [
    "FLIGHT_DELAYED",
    "FLIGHT_CANCELLED",
    "GATE_CHANGED",
    "CARGO_LOADED",
    "CARGO_OFFLOADED",
    "FLIGHT_DEPARTED",
    "FLIGHT_ARRIVED",
]

EVENT_WEIGHTS = [0.20, 0.05, 0.10, 0.30, 0.10, 0.15, 0.10]


# ─────────────────────────────────────────────────────────────────────────────
# Event Generator
# ─────────────────────────────────────────────────────────────────────────────
class FlightEventGenerator:
    """Generates realistic flight status events."""

    def __init__(self, flights_path: str = None):
        self.flight_ids = []
        if flights_path and os.path.exists(flights_path):
            try:
                df = pd.read_parquet(flights_path)
                self.flight_ids = df["flight_id"].tolist()[:500]
            except Exception:
                pass

        if not self.flight_ids:
            # Fallback: generate synthetic flight IDs
            self.flight_ids = [f"FL{i:05d}" for i in range(1000, 1500)]

    def generate_event(self) -> dict:
        flight_id  = random.choice(self.flight_ids)
        event_type = random.choices(EVENT_TYPES, weights=EVENT_WEIGHTS, k=1)[0]
        now = datetime.now()

        event = {
            "event_id":    f"EVT_{int(time.time() * 1000)}_{random.randint(100, 999)}",
            "flight_id":   flight_id,
            "event_type":  event_type,
            "timestamp":   now.isoformat(),
            "source":      "ATC_SYSTEM",
        }

        if event_type == "FLIGHT_DELAYED":
            delay_min = random.choice([15, 30, 45, 60, 90, 120, 180])
            event.update({
                "delay_minutes": delay_min,
                "reason":        random.choice([
                    "Weather", "Technical", "Air Traffic Control",
                    "Crew", "Operational", "Ground Handling"
                ]),
                "new_departure":  (now + timedelta(minutes=delay_min)).isoformat(),
            })

        elif event_type == "FLIGHT_CANCELLED":
            event.update({
                "reason": random.choice(["Weather", "Technical", "Operational"]),
                "affected_cargo_kg": random.randint(500, 50000),
                "rebook_options": [
                    f"FL{random.randint(1000, 9999):05d}",
                    f"FL{random.randint(1000, 9999):05d}",
                ],
            })

        elif event_type == "GATE_CHANGED":
            event.update({
                "old_gate": f"{random.choice('ABCDE')}{random.randint(1, 40)}",
                "new_gate": f"{random.choice('ABCDE')}{random.randint(1, 40)}",
            })

        elif event_type in ("CARGO_LOADED", "CARGO_OFFLOADED"):
            event.update({
                "shipment_count": random.randint(1, 50),
                "total_weight_kg": round(random.uniform(100, 10000), 1),
                "reason": "Priority Override" if event_type == "CARGO_OFFLOADED" else None,
            })

        elif event_type in ("FLIGHT_DEPARTED", "FLIGHT_ARRIVED"):
            event.update({
                "actual_time": now.isoformat(),
                "cargo_weight_kg": round(random.uniform(5000, 100000), 1),
            })

        return event


# ─────────────────────────────────────────────────────────────────────────────
# Kafka Producer
# ─────────────────────────────────────────────────────────────────────────────
def create_kafka_producer():
    try:
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BOOTSTRAP],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
            retries=3,
            max_block_ms=5000,
        )
        return producer
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# File-based Fallback Producer
# ─────────────────────────────────────────────────────────────────────────────
class FileEventStore:
    """Stores events to a JSONL file when Kafka is unavailable."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._lock    = threading.Lock()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def write(self, event: dict):
        with self._lock:
            with open(self.filepath, "a") as f:
                f.write(json.dumps(event) + "\n")


def read_events_from_file(filepath: str, max_events: int = 1000) -> list:
    """Read events from the JSONL file."""
    events = []
    if not os.path.exists(filepath):
        return events
    with open(filepath) as f:
        for i, line in enumerate(f):
            if i >= max_events:
                break
            try:
                events.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return events


# ─────────────────────────────────────────────────────────────────────────────
# Event Stream Consumer / Analyzer
# ─────────────────────────────────────────────────────────────────────────────
class EventStreamAnalyzer:
    """Processes flight events and maintains aggregated state."""

    def __init__(self):
        self.delayed_flights   = {}
        self.cancelled_flights = set()
        self.total_events      = 0
        self.events_by_type    = {et: 0 for et in EVENT_TYPES}

    def process_event(self, event: dict):
        self.total_events += 1
        etype = event.get("event_type", "")
        if etype in self.events_by_type:
            self.events_by_type[etype] += 1

        fid = event.get("flight_id", "")
        if etype == "FLIGHT_DELAYED":
            self.delayed_flights[fid] = event.get("delay_minutes", 0)
        elif etype == "FLIGHT_CANCELLED":
            self.cancelled_flights.add(fid)
            self.delayed_flights.pop(fid, None)

    def get_summary(self) -> dict:
        return {
            "total_events":       self.total_events,
            "delayed_flights":    len(self.delayed_flights),
            "cancelled_flights":  len(self.cancelled_flights),
            "events_by_type":     self.events_by_type,
            "avg_delay_minutes":  (
                sum(self.delayed_flights.values()) / len(self.delayed_flights)
                if self.delayed_flights else 0
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main Simulator
# ─────────────────────────────────────────────────────────────────────────────
def simulate_events(n_events: int = 500, interval_ms: int = 10, verbose: bool = True):
    """
    Simulate n_events flight status events.
    Writes to Kafka if available, otherwise to a local JSONL file.
    """
    os.makedirs(EVENTS_DIR, exist_ok=True)
    events_file = os.path.join(EVENTS_DIR, "flight_events.jsonl")

    # Clear previous events
    if os.path.exists(events_file):
        os.remove(events_file)

    flights_path = os.path.join(PROCESSED_DIR, "flights_clean.parquet")
    generator   = FlightEventGenerator(flights_path)
    analyzer    = EventStreamAnalyzer()
    store       = FileEventStore(events_file)

    producer = None
    if KAFKA_AVAILABLE:
        producer = create_kafka_producer()
        if producer and verbose:
            print(f"  ✓  Kafka producer connected to {KAFKA_BOOTSTRAP}")
        elif verbose:
            print(f"  ⚠  Kafka not available — writing to {events_file}")
    else:
        if verbose:
            print(f"  ⚠  kafka-python not installed — writing to {events_file}")

    if verbose:
        print(f"  Generating {n_events:,} events...", end=" ", flush=True)

    for i in range(n_events):
        event = generator.generate_event()
        analyzer.process_event(event)

        if producer:
            producer.send(TOPIC_FLIGHT, value=event)
        else:
            store.write(event)

        if interval_ms > 0:
            time.sleep(interval_ms / 1000)

    if producer:
        producer.flush()

    summary = analyzer.get_summary()

    if verbose:
        print("✓")
        print(f"  ✓  Events generated:   {summary['total_events']:,}")
        print(f"  ✓  Delayed flights:    {summary['delayed_flights']}")
        print(f"  ✓  Cancelled flights:  {summary['cancelled_flights']}")
        print(f"  ✓  Avg delay:          {summary['avg_delay_minutes']:.1f} min")

    # Save summary
    summary_path = os.path.join(EVENTS_DIR, "kafka_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary, events_file


if __name__ == "__main__":
    print("\n📡  Aviation Cargo — Kafka Event Simulator")
    print("=" * 50)
    simulate_events(n_events=500, verbose=True)
