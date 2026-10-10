"""Draw docs/architecture.png (use it in the PDF report; Mermaid doesn't render in PDF)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

BOXES = {
    "uci": (0.3, 7.2, "UCI Heart Disease\ndata (download script)", "#e9ecef"),
    "prep": (3.3, 7.2, "Cleaning +\nEDA (src/data, src/eda)", "#dbeafe"),
    "train": (6.3, 7.2, "Training + tuning\n(sklearn Pipeline, CV)", "#dbeafe"),
    "mlflow": (9.3, 7.2, "MLflow tracking\n(params, metrics, plots)", "#fde68a"),
    "model": (6.3, 5.0, "Packaged model\nmodels/model.joblib", "#bbf7d0"),
    "git": (0.3, 5.0, "GitHub repo", "#e9ecef"),
    "ci": (3.3, 5.0, "GitHub Actions\nlint > test > train > docker", "#fecaca"),
    "docker": (6.3, 2.8, "Docker image\nFastAPI /predict", "#c7d2fe"),
    "k8s": (3.3, 2.8, "Kubernetes (local)\n2 pods + LoadBalancer", "#c7d2fe"),
    "user": (0.3, 2.8, "Client\n(curl / app)", "#e9ecef"),
    "prom": (3.3, 0.6, "Prometheus\n(scrapes /metrics)", "#fed7aa"),
    "graf": (6.3, 0.6, "Grafana\ndashboard", "#fed7aa"),
    "logs": (9.3, 2.8, "Structured JSON\nrequest logs", "#fed7aa"),
}
ARROWS = [
    ("uci", "prep"), ("prep", "train"), ("train", "mlflow"), ("train", "model"), ("git", "ci"),
    ("ci", "model"), ("model", "docker"), ("docker", "k8s"), ("user", "k8s"),
    ("k8s", "prom"), ("prom", "graf"), ("docker", "logs"),
]
W, H = 2.4, 1.2


def anchors(a, b):
    """Edge-to-edge anchor points for axis-aligned neighbours."""
    ax_, ay_, *_ = BOXES[a]
    bx_, by_, *_ = BOXES[b]
    if abs(ay_ - by_) < 1e-6:  # same row -> horizontal arrow
        if bx_ > ax_:
            return (ax_ + W, ay_ + H / 2), (bx_, by_ + H / 2)
        return (ax_, ay_ + H / 2), (bx_ + W, by_ + H / 2)
    if by_ < ay_:  # target below
        return (ax_ + W / 2, ay_), (bx_ + W / 2, by_ + H)
    return (ax_ + W / 2, ay_ + H), (bx_ + W / 2, by_)


fig, ax = plt.subplots(figsize=(13, 8))
ax.set_xlim(0, 12.5)
ax.set_ylim(0, 8.8)
ax.axis("off")
for key, (x, y, label, color) in BOXES.items():
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.05", fc=color, ec="#334155", lw=1.4))
    ax.text(x + W / 2, y + H / 2, label, ha="center", va="center", fontsize=9.5)
for a, b in ARROWS:
    start, end = anchors(a, b)
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                 lw=1.5, color="#475569", shrinkA=2, shrinkB=2))
ax.set_title("Heart Disease MLOps - architecture", fontsize=14, fontweight="bold")
out = Path(__file__).resolve().parents[1] / "docs" / "architecture.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
