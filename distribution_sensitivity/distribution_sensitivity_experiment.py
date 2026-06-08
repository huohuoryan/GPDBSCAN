import csv
import itertools
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

from dbscan_pbc.dbscan_pbc_grid import GPDBSCAN


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = BASE_DIR / "distribution_sensitivity_results.csv"
OUTPUT_JSON = BASE_DIR / "distribution_sensitivity_results.json"
OUTPUT_MD = BASE_DIR / "distribution_sensitivity_table.md"
OUTPUT_FIG = BASE_DIR / "figs" / "distribution_sensitivity_speedup.png"

N_POINTS = 5000
DIMS = 2
EPS = 0.05
MIN_SAMPLES = 5
REPEATS = 5
PBC_LOWER = np.zeros(DIMS)
PBC_UPPER = np.ones(DIMS)


def boundary_mask(x, eps=EPS):
    return np.any((x < eps) | (x > 1.0 - eps), axis=1)


def make_interior_concentrated(n=N_POINTS, seed=42):
    rng = np.random.default_rng(seed)
    n_boundary = int(0.08 * n)
    n_interior = n - n_boundary

    interior = rng.normal(loc=[0.5, 0.5], scale=[0.09, 0.09], size=(n_interior, 2))
    interior = np.clip(interior, 0.18, 0.82)

    half = n_boundary // 2
    left = np.column_stack(
        [
            rng.uniform(0.0, EPS * 0.8, size=half),
            rng.normal(0.5, 0.08, size=half),
        ]
    )
    right = np.column_stack(
        [
            rng.uniform(1.0 - EPS * 0.8, 1.0, size=n_boundary - half),
            rng.normal(0.5, 0.08, size=n_boundary - half),
        ]
    )
    boundary = np.vstack([left, right])
    boundary[:, 1] = np.clip(boundary[:, 1], 0.18, 0.82)
    return np.vstack([interior, boundary])


def make_uniform(n=N_POINTS, seed=43):
    rng = np.random.default_rng(seed)
    return rng.random((n, 2))


def make_boundary_concentrated(n=N_POINTS, seed=44):
    rng = np.random.default_rng(seed)
    n_boundary = int(0.70 * n)
    n_interior = n - n_boundary

    interior = rng.uniform(0.18, 0.82, size=(n_interior, 2))
    sides = rng.integers(0, 4, size=n_boundary)
    boundary = rng.uniform(0.0, 1.0, size=(n_boundary, 2))
    low = rng.uniform(0.0, EPS * 0.8, size=n_boundary)
    high = rng.uniform(1.0 - EPS * 0.8, 1.0, size=n_boundary)

    boundary[sides == 0, 0] = low[sides == 0]
    boundary[sides == 1, 0] = high[sides == 1]
    boundary[sides == 2, 1] = low[sides == 2]
    boundary[sides == 3, 1] = high[sides == 3]
    return np.vstack([interior, boundary])


def full_replication_pbc_fit(x):
    shifts = []
    for dim in range(x.shape[1]):
        neg = np.zeros(x.shape[1])
        pos = np.zeros(x.shape[1])
        neg[dim] = -1.0
        pos[dim] = 1.0
        shifts.extend([neg, pos])

    replicas = [x]
    source_idx = [np.arange(len(x), dtype=int)]
    for shift in shifts:
        replicas.append(x + shift)
        source_idx.append(np.arange(len(x), dtype=int))

    padded = np.vstack(replicas)
    source_idx = np.concatenate(source_idx)
    model = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
    model.fit(padded)

    labels = model.labels_[: len(x)].copy()
    replica_labels = model.labels_[len(x) :]
    replica_sources = source_idx[len(x) :]
    return merge_replica_labels(labels, replica_labels, replica_sources)


def merge_replica_labels(labels, replica_labels, replica_sources):
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        if parent[a] != a:
            parent[a] = find(parent[a])
        return parent[a]

    def union(a, b):
        if a < 0 or b < 0:
            return
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for src, replica_label in zip(replica_sources, replica_labels):
        union(labels[src], replica_label)

    merged = labels.copy()
    for i, label in enumerate(merged):
        if label >= 0:
            merged[i] = find(label)

    unique = sorted(np.unique(merged[merged >= 0]))
    label_map = {old: new for new, old in enumerate(unique)}
    for old, new in label_map.items():
        merged[merged == old] = new
    return merged


def run_once(method, x):
    start = time.perf_counter()
    if method == "PBC-DBSCAN":
        labels = full_replication_pbc_fit(x.copy())
    elif method == "GPDBSCAN":
        model = GPDBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
        model.fit(x.copy(), pbc_lower=PBC_LOWER, pbc_upper=PBC_UPPER)
        labels = model.labels_
    else:
        raise ValueError(method)
    elapsed = time.perf_counter() - start
    n_clusters = len(np.unique(labels[labels >= 0]))
    n_noise = int(np.sum(labels == -1))
    return elapsed, n_clusters, n_noise


def summarize(values):
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def evaluate_distribution(name, x):
    k = int(np.sum(boundary_mask(x)))
    rows = []
    runtime_by_method = {}

    for method in ["PBC-DBSCAN", "GPDBSCAN"]:
        runtimes = []
        clusters = []
        noises = []
        for _ in range(REPEATS):
            elapsed, n_clusters, n_noise = run_once(method, x)
            runtimes.append(elapsed)
            clusters.append(n_clusters)
            noises.append(n_noise)

        mean_rt, std_rt = summarize(runtimes)
        runtime_by_method[method] = mean_rt
        rows.append(
            {
                "distribution": name,
                "n": len(x),
                "k": k,
                "k_over_n": k / len(x),
                "n_over_k": len(x) / k if k else float("inf"),
                "method": method,
                "runtime_mean_s": mean_rt,
                "runtime_std_s": std_rt,
                "clusters": int(round(float(np.mean(clusters)))),
                "noise_points": int(round(float(np.mean(noises)))),
            }
        )

    speedup = runtime_by_method["PBC-DBSCAN"] / runtime_by_method["GPDBSCAN"]
    for row in rows:
        row["observed_speedup"] = speedup
    return rows


def write_outputs(rows):
    fieldnames = [
        "distribution",
        "n",
        "k",
        "k_over_n",
        "n_over_k",
        "method",
        "runtime_mean_s",
        "runtime_std_s",
        "clusters",
        "noise_points",
        "observed_speedup",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    OUTPUT_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    OUTPUT_MD.write_text(build_markdown(rows), encoding="utf-8")
    draw_figure(rows)


def build_markdown(rows):
    lines = [
        "# Distribution Sensitivity of the Boundary Reduction Effect",
        "",
        "| Distribution | n | k | k/n | n/k | PBC-DBSCAN runtime (s) | GPDBSCAN runtime (s) | Observed speedup |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    grouped = {}
    for row in rows:
        grouped.setdefault(row["distribution"], {})[row["method"]] = row

    for distribution, methods in grouped.items():
        pbc = methods["PBC-DBSCAN"]
        gp = methods["GPDBSCAN"]
        lines.append(
            "| {distribution} | {n} | {k} | {k_over_n:.3f} | {n_over_k:.2f} | "
            "{pbc_mean:.4f} ± {pbc_std:.4f} | {gp_mean:.4f} ± {gp_std:.4f} | {speedup:.2f} |".format(
                distribution=distribution,
                n=pbc["n"],
                k=pbc["k"],
                k_over_n=pbc["k_over_n"],
                n_over_k=pbc["n_over_k"],
                pbc_mean=pbc["runtime_mean_s"],
                pbc_std=pbc["runtime_std_s"],
                gp_mean=gp["runtime_mean_s"],
                gp_std=gp["runtime_std_s"],
                speedup=pbc["observed_speedup"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def draw_figure(rows):
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["distribution"], {})[row["method"]] = row

    labels = list(grouped)
    k_ratios = [grouped[label]["PBC-DBSCAN"]["k_over_n"] for label in labels]
    speedups = [grouped[label]["PBC-DBSCAN"]["observed_speedup"] for label in labels]

    fig, ax1 = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(labels))
    bars = ax1.bar(
        x,
        k_ratios,
        width=0.46,
        label=r"Boundary-point ratio $k/n$",
        color="#4C78A8",
        edgecolor="#2F5F8F",
        linewidth=0.8,
        alpha=0.88,
        zorder=2,
    )
    ax1.set_ylabel(r"Boundary-point ratio $k/n$", fontsize=12)
    ax1.set_ylim(0, 0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)

    ax2 = ax1.twinx()
    line = ax2.plot(
        x,
        speedups,
        marker="o",
        markersize=6.5,
        linewidth=2.2,
        label=r"Observed speedup $T_{\mathrm{PBC}}/T_{\mathrm{GP}}$",
        color="#E45756",
        markerfacecolor="#E45756",
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=3,
    )
    ax2.set_ylabel(r"Observed speedup $T_{\mathrm{PBC}}/T_{\mathrm{GP}}$", fontsize=12)
    ax2.set_ylim(0, 3.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", frameon=True, fontsize=10)
    ax1.set_title("Distribution Sensitivity of Boundary Reduction", fontsize=14, pad=10)
    ax1.grid(axis="y", alpha=0.25, linestyle="-", linewidth=0.8, zorder=0)
    ax1.set_axisbelow(True)
    ax1.tick_params(axis="y", labelsize=10)
    ax2.tick_params(axis="y", labelsize=10)

    for rect, value in zip(bars, k_ratios):
        ax1.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#2F5F8F",
        )
    for xi, value in zip(x, speedups):
        ax2.text(
            xi,
            value + 0.08,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#B83A3F",
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_FIG, dpi=300)
    plt.close(fig)


def main():
    datasets = [
        ("Interior-concentrated", make_interior_concentrated()),
        ("Uniform", make_uniform()),
        ("Boundary-concentrated", make_boundary_concentrated()),
    ]

    rows = []
    for name, x in datasets:
        rows.extend(evaluate_distribution(name, x))

    write_outputs(rows)
    print(build_markdown(rows))
    print(f"Saved CSV: {OUTPUT_CSV}")
    print(f"Saved JSON: {OUTPUT_JSON}")
    print(f"Saved table: {OUTPUT_MD}")
    print(f"Saved figure: {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
