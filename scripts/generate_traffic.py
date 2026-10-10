"""Send random prediction requests so the Grafana dashboard has data to show.

Usage: python scripts/generate_traffic.py [base_url] [n_requests]
"""
import json
import random
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def random_patient():
    return {
        "age": random.randint(29, 77), "sex": random.randint(0, 1), "cp": random.randint(1, 4),
        "trestbps": random.randint(94, 200), "chol": random.randint(126, 564),
        "fbs": random.randint(0, 1), "restecg": random.randint(0, 2),
        "thalach": random.randint(71, 202), "exang": random.randint(0, 1),
        "oldpeak": round(random.uniform(0, 6), 1), "slope": random.randint(1, 3),
        "ca": random.randint(0, 3), "thal": random.choice([3, 6, 7]),
    }


def main():
    base = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    ok = bad = 0
    for i in range(n):
        payload = random_patient()
        if i % 25 == 24:  # occasionally send an invalid request to show 4xx in the dashboard
            payload["age"] = 999
        req = Request(f"{base}/predict", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=10) as resp:  # nosec
                ok += resp.status == 200
        except HTTPError:
            bad += 1
        time.sleep(random.uniform(0.02, 0.15))
    print(f"Sent {n} requests: {ok} OK, {bad} rejected")


if __name__ == "__main__":
    main()
