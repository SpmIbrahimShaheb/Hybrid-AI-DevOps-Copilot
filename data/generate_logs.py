"""
generate_logs.py
-----------------
Generates 100% synthetic application logs simulating realistic cascading
failure patterns. No real company/organizational data is used anywhere.

Produces 4 known incident scenarios (each triggering a cascade of related
log lines) plus random unrelated "noise" logs, and stores everything in
a SQLite database (logs.db) with a hidden ground-truth label column
(scenario_label) used ONLY for evaluating the clustering later -- it is
never fed into the ML model itself.
"""

import random
import sqlite3
import uuid
from datetime import datetime, timedelta

random.seed(42)  # reproducible results -- important for your report

DB_PATH = "logs.db"

SERVICES = [
    "auth-service", "payment-service", "checkout-service",
    "inventory-service", "notification-service", "order-service",
    "cache-service", "gateway-service",
]

SCENARIOS = {
    "db_connection_exhaustion": {
        "level": "ERROR",
        "templates": [
            "Connection pool exhausted for database 'orders_db', active=100 max=100",
            "Timeout acquiring DB connection after 30000ms",
            "PSQLException: FATAL: too many connections for role 'app_user'",
            "Query execution failed: connection reset by peer",
            "Retrying DB connection attempt {n}/5, pool still exhausted",
            "HTTP 503 returned to client due to downstream DB timeout",
            "Connection pool wait queue size exceeded threshold (200)",
        ],
        "cascade_size": (10, 15),
        "services": ["payment-service", "checkout-service", "order-service"],
    },
    "memory_leak": {
        "level": "WARN",
        "templates": [
            "Heap usage at {pct}% of allocated memory, approaching limit",
            "GC pause duration exceeded 2000ms, potential memory pressure",
            "OutOfMemoryError: Java heap space",
            "Container memory usage 950Mi / 1000Mi, nearing OOMKill threshold",
            "Pod restarted due to OOMKilled status",
            "Memory usage trend increasing steadily over last 6 hours",
        ],
        "cascade_size": (8, 12),
        "services": ["inventory-service", "cache-service"],
    },
    "disk_full": {
        "level": "ERROR",
        "templates": [
            "No space left on device: /var/log/app.log write failed",
            "Disk usage at 98% on volume /data, write operations failing",
            "Log rotation failed: insufficient disk space",
            "Failed to write checkpoint file: ENOSPC",
            "Disk cleanup job failed: unable to remove old files, disk full",
        ],
        "cascade_size": (6, 10),
        "services": ["gateway-service", "notification-service"],
    },
    "network_timeout": {
        "level": "ERROR",
        "templates": [
            "Upstream request to auth-service timed out after 5000ms",
            "Connection refused: unable to reach service at 10.0.0.{n}:8080",
            "Circuit breaker OPEN for downstream dependency 'auth-service'",
            "Retry attempt {n} failed: read timeout",
            "Gateway timeout (504) returned after 3 failed retries",
            "DNS resolution failed for internal service endpoint",
        ],
        "cascade_size": (8, 14),
        "services": ["auth-service", "gateway-service"],
    },
}

NOISE_MESSAGES = [
    "Health check passed for {svc}",
    "Scheduled backup completed successfully",
    "User login successful, session created",
    "Cache warm-up completed in {n}ms",
    "Deployment version v1.{n}.0 rolled out successfully",
    "Config reload triggered, no changes detected",
]


def fill_template(template: str) -> str:
    return template.format(
        n=random.randint(1, 999),
        pct=random.randint(85, 99),
        svc=random.choice(SERVICES),
    )


def generate_incident(scenario_name: str, start_time: datetime):
    cfg = SCENARIOS[scenario_name]
    count = random.randint(*cfg["cascade_size"])
    rows = []
    t = start_time
    for _ in range(count):
        t += timedelta(seconds=random.randint(1, 15))
        rows.append({
            "log_id": str(uuid.uuid4()),
            "timestamp": t.isoformat(),
            "service_name": random.choice(cfg["services"]),
            "log_level": cfg["level"],
            "message": fill_template(random.choice(cfg["templates"])),
            "scenario_label": scenario_name,
        })
    return rows, t


def generate_noise(n: int, start_time: datetime):
    rows = []
    t = start_time
    for _ in range(n):
        t += timedelta(minutes=random.randint(1, 20))
        rows.append({
            "log_id": str(uuid.uuid4()),
            "timestamp": t.isoformat(),
            "service_name": random.choice(SERVICES),
            "log_level": "INFO",
            "message": fill_template(random.choice(NOISE_MESSAGES)),
            "scenario_label": "noise",
        })
    return rows


def build_dataset():
    all_rows = []
    current_time = datetime.now() - timedelta(hours=6)

    scenario_order = list(SCENARIOS.keys())
    random.shuffle(scenario_order)

    for scenario_name in scenario_order:
        noise_rows = generate_noise(random.randint(3, 6), current_time)
        all_rows.extend(noise_rows)
        current_time = datetime.fromisoformat(noise_rows[-1]["timestamp"])

        incident_rows, current_time = generate_incident(scenario_name, current_time)
        all_rows.extend(incident_rows)

    all_rows.extend(generate_noise(5, current_time))

    return all_rows


def save_to_db(rows, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            log_id TEXT PRIMARY KEY,
            timestamp TEXT,
            service_name TEXT,
            log_level TEXT,
            message TEXT,
            scenario_label TEXT
        )
    """)
    cur.execute("DELETE FROM logs")
    cur.executemany(
        """INSERT INTO logs (log_id, timestamp, service_name, log_level, message, scenario_label)
           VALUES (:log_id, :timestamp, :service_name, :log_level, :message, :scenario_label)""",
        rows,
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    rows = build_dataset()
    save_to_db(rows)

    print(f"Generated {len(rows)} log lines -> saved to {DB_PATH}")
    print("\nBreakdown by scenario (ground truth -- hidden from clustering step):")
    from collections import Counter
    counts = Counter(r["scenario_label"] for r in rows)
    for label, count in counts.items():
        print(f"  {label:30s} {count}")
