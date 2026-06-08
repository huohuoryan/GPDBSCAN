from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from synthetic_external_metrics import (
    generate_double_boundary,
    generate_four_boundary,
    generate_nonconvex,
    generate_periodic_3d,
    run_methods,
)


OUTPUT_PATH = Path(__file__).resolve().parent / "figs" / "fig3_clustering_results_synthetic_labeled.png"

DATASETS = [
    ("2D Double-Boundary", generate_double_boundary, (-0.05, 1.05), (0.40, 0.60)),
    ("2D Four-Boundary", generate_four_boundary, (-0.05, 1.05), (-0.05, 1.05)),
    ("2D Non-convex", generate_nonconvex, (-0.05, 1.05), (-0.05, 1.05)),
    ("3D Periodic", generate_periodic_3d, (0.0, 1.0), (0.0, 1.0)),
]

METHODS = ["DBSCAN", "PBC-DBSCAN", "K-PBC", "GPDBSCAN"]
SUBPLOT_PREFIXES = ["a", "b", "c", "d"]
CLUSTER_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def plot_2d(ax, x, labels, xlim, ylim):
    for label in sorted(np.unique(labels)):
        mask = labels == label
        if label == -1:
            ax.scatter(x[mask, 0], x[mask, 1], c="black", s=5, alpha=0.82, linewidths=0)
        else:
            color = CLUSTER_COLORS[int(label) % len(CLUSTER_COLORS)]
            ax.scatter(x[mask, 0], x[mask, 1], c=[color], s=6, alpha=0.88, linewidths=0)

    ax.axvline(0, color="#8F8F8F", linestyle="--", linewidth=0.7, alpha=0.65)
    ax.axvline(1, color="#8F8F8F", linestyle="--", linewidth=0.7, alpha=0.65)
    if ylim[0] < 0:
        ax.axhline(0, color="#8F8F8F", linestyle="--", linewidth=0.7, alpha=0.65)
        ax.axhline(1, color="#8F8F8F", linestyle="--", linewidth=0.7, alpha=0.65)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x", fontsize=8)
    ax.set_ylabel("y", fontsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=7)


def plot_3d(ax, x, labels):
    for label in sorted(np.unique(labels)):
        mask = labels == label
        if label == -1:
            ax.scatter(x[mask, 0], x[mask, 1], x[mask, 2], c="black", s=4, alpha=0.80, linewidths=0)
        else:
            color = CLUSTER_COLORS[int(label) % len(CLUSTER_COLORS)]
            ax.scatter(x[mask, 0], x[mask, 1], x[mask, 2], c=[color], s=5, alpha=0.88, linewidths=0)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel("x", fontsize=8, labelpad=-6)
    ax.set_ylabel("y", fontsize=8, labelpad=-6)
    ax.set_zlabel("z", fontsize=8, labelpad=-12)
    ax.view_init(elev=24, azim=-58)
    ax.tick_params(labelsize=0, pad=-4)


def add_subplot_label(ax, label):
    text_func = ax.text2D if hasattr(ax, "text2D") else ax.text
    text_func(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
    )


def main():
    fig = plt.figure(figsize=(12.0, 12.8))
    axes = []

    for row_idx, (dataset_name, generator, xlim, ylim) in enumerate(DATASETS):
        spec = generator()
        x = spec["x"]
        method_labels = run_methods(x, spec["params"])
        row_axes = []

        for col_idx, method in enumerate(METHODS):
            subplot_index = row_idx * len(METHODS) + col_idx + 1
            if x.shape[1] == 3:
                ax = fig.add_subplot(4, 4, subplot_index, projection="3d")
                plot_3d(ax, x, method_labels[method])
            else:
                ax = fig.add_subplot(4, 4, subplot_index)
                plot_2d(ax, x, method_labels[method], xlim, ylim)

            if row_idx == 0:
                ax.set_title(method, fontsize=10, pad=6)
            add_subplot_label(ax, f"({SUBPLOT_PREFIXES[row_idx]}{col_idx + 1})")
            row_axes.append(ax)

        axes.append(row_axes)
        fig.text(
            0.015,
            0.88 - row_idx * 0.232,
            dataset_name,
            rotation=90,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="Predicted clusters", markerfacecolor=CLUSTER_COLORS[0], markersize=7),
        Line2D([0], [0], marker="o", color="w", label="Noise", markerfacecolor="black", markersize=7),
        Line2D([0], [0], color="#8F8F8F", linestyle="--", linewidth=1.0, label="Periodic boundary"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=True,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.text(
        0.5,
        0.006,
        "Cluster colors are assigned independently within each subplot.",
        ha="center",
        va="bottom",
        fontsize=12,
    )
    fig.subplots_adjust(left=0.055, right=0.965, top=0.972, bottom=0.075, hspace=0.35, wspace=0.32)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300)
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
