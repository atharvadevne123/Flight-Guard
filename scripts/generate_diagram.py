"""Generate system architecture diagram for Flight-Guard."""

from __future__ import annotations

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    color: str,
    fontsize: int = 9,
) -> None:
    rect = mpatches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor="#333333",
        linewidth=1.2,
    )
    ax.add_patch(rect)
    ax.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        wrap=True,
        multialignment="center",
    )


def draw_arrow(
    ax: plt.Axes,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    label: str = "",
) -> None:
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5),
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.08, label, ha="center", va="bottom", fontsize=7, color="#555555")


def main() -> None:
    os.makedirs("screenshots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    # Title
    ax.text(
        8,
        8.5,
        "Flight-Guard — System Architecture",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#1a1a2e",
    )

    # Layer: Client
    draw_box(ax, 0.3, 7.0, 2.0, 0.8, "API Client\n(REST / SDK)", "#aed6f1")

    # Layer: FastAPI
    draw_box(
        ax,
        3.0,
        6.5,
        3.0,
        1.4,
        "FastAPI\n/api/v1/predict\n/health /metrics\n/drift /stats",
        "#85c1e9",
    )

    # Middleware stack
    draw_box(ax, 6.5, 7.0, 2.2, 0.8, "Middleware\nCorr-ID + Rate Limit", "#a9cce3")

    # Feature pipeline
    draw_box(
        ax,
        3.0,
        4.5,
        3.0,
        1.5,
        "Feature Pipeline\nCarrier Risk Encoder\nRoute + Temporal\nLag/Rolling + Scaler",
        "#a3e4d7",
    )

    # Model ensemble
    draw_box(
        ax,
        6.8,
        4.5,
        2.5,
        1.5,
        "Ensemble Model\nXGBoost + LightGBM\n+ RandomForest\nSoft Voting",
        "#a9dfbf",
    )

    # Monitoring
    draw_box(
        ax, 10.0, 4.5, 2.5, 1.5, "Monitoring\nKS-test Drift\nPSI Score\nPrediction Stats", "#f9e79f"
    )

    # Databases
    draw_box(ax, 3.0, 2.5, 2.5, 1.2, "SQLite / PostgreSQL\n(PredictionLog\nDriftReport)", "#fadbd8")
    draw_box(ax, 6.8, 2.5, 2.5, 1.2, "Model Storage\nmodel.joblib\nmetrics.json", "#d7bde2")
    draw_box(ax, 10.0, 2.5, 2.5, 1.2, "Airflow DAG\nWeekly Retrain\nAUC Gate ≥0.70", "#fdebd0")

    # Docker
    draw_box(
        ax,
        3.0,
        0.4,
        9.5,
        1.2,
        "Docker + docker-compose  ·  API Container + PostgreSQL Container",
        "#eaeded",
    )

    # Arrows
    draw_arrow(ax, 2.3, 7.4, 3.0, 7.2)
    draw_arrow(ax, 4.5, 6.5, 4.5, 6.0)
    draw_arrow(ax, 6.0, 5.2, 6.8, 5.2)
    draw_arrow(ax, 9.3, 5.2, 10.0, 5.2)
    draw_arrow(ax, 4.5, 4.5, 4.5, 3.7)
    draw_arrow(ax, 8.0, 4.5, 8.0, 3.7)
    draw_arrow(ax, 11.2, 4.5, 11.2, 3.7)
    draw_arrow(ax, 8.0, 2.5, 8.0, 1.6)

    # Legend
    legend_items = [
        mpatches.Patch(color="#85c1e9", label="API Layer"),
        mpatches.Patch(color="#a3e4d7", label="Feature Engineering"),
        mpatches.Patch(color="#a9dfbf", label="ML Model"),
        mpatches.Patch(color="#f9e79f", label="Monitoring"),
        mpatches.Patch(color="#fadbd8", label="Database"),
        mpatches.Patch(color="#fdebd0", label="Retraining"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig("screenshots/architecture.png", dpi=150, bbox_inches="tight")
    print("Architecture diagram saved to screenshots/architecture.png")


if __name__ == "__main__":
    main()
