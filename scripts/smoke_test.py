"""Call a running API (local container, Kubernetes service...) with a sample patient.

Usage: python scripts/smoke_test.py [base_url]     default http://localhost:8000
Exit code is non-zero on any failure, so it can be used in CI.
"""
import json
import sys
from urllib.request import Request, urlopen

SAMPLE = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2,
    "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
}


def call(url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=10) as resp:  # nosec - user-supplied local URL
        return resp.status, json.loads(resp.read().decode())


def main():
    base = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
    status, body = call(f"{base}/health")
    print("GET /health ->", status, body)
    assert status == 200 and body["status"] == "ok"

    status, body = call(f"{base}/predict", SAMPLE)
    print("POST /predict ->", status, json.dumps(body, indent=2))
    assert status == 200
    assert body["prediction"] in (0, 1) and 0.5 <= body["confidence"] <= 1.0
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print("SMOKE TEST FAILED:", exc)
        sys.exit(1)
