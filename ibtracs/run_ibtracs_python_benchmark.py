import csv
import json
import sys
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN


BASE_DIR = Path(__file__).resolve().parent.parent
PYTHON_PROJECT = BASE_DIR / "pythonProject"
if str(PYTHON_PROJECT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PROJECT))

from dbscan_pbc.dbscan_pbc import DBSCAN_PBC
from dbscan_pbc.dbscan_pbc_grid import GPDBSCAN


POINTS_CSV = BASE_DIR / "ibtracs" / "experiment" / "ibtracs_experiment_points.csv"
EXPERIMENT_DIR = BASE_DIR / "ibtracs" / "experiment"
TABLE5_CSV = EXPERIMENT_DIR / "table5_ibtracs_python_results.csv"
TABLE5_JSON = EXPERIMENT_DIR / "table5_ibtracs_python_results.json"
FIG6_PNG = EXPERIMENT_DIR / "ibtracs_dateline_overview_matplotlib.png"
FIG7_PNG = EXPERIMENT_DIR / "ibtracs_runtime_memory_scaling_pythonProject.png"

EPS = 5.0
MIN_SAMPLES = 2
TIME_SCALE = 0.20
REPEATS = 5
KEEP_COUNTS = [4, 6, 8, 10, 12, 14, 17]
METHODS = ["DBSCAN", "PBC-DBSCAN", "GPDBSCAN"]
COLORS = {"DBSCAN": "#1f77b4", "PBC-DBSCAN": "#ff7f0e", "GPDBSCAN": "#d62728"}
MARKERS = {"DBSCAN": "o", "PBC-DBSCAN": "s", "GPDBSCAN": "D"}


def load_points():
    points = []
    with POINTS_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append(
                {
                    "sid": row["sid"],
                    "name": row["name"],
                    "basin": row["basin"],
                    "iso_time": row["iso_time"],
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "lon_periodic": float(row["lon_periodic"]),
                    "time_hours": float(row["time_hours"]),
                    "true_label": int(row["true_label"]),
                }
            )
    points.sort(key=lambda row: row["time_hours"])
    return points


def score_storms_by_boundary_ratio(points, eps=EPS):
    by_storm = defaultdict(list)
    for point in points:
        by_storm[point["sid"]].append(point)

    scored = []
    for sid, rows in by_storm.items():
        boundary_count = sum(
            1 for row in rows if row["lon_periodic"] < eps or row["lon_periodic"] > (360.0 - eps)
        )
        scored.append(
            {
                "sid": sid,
                "rows": sorted(rows, key=lambda r: r["time_hours"]),
                "ratio": boundary_count / len(rows),
                "boundary_count": boundary_count,
                "n_points": len(rows),
            }
        )

    scored.sort(key=lambda item: (item["ratio"], item["boundary_count"], -item["n_points"]))
    return scored


def build_cumulative_subset(points, scored_storms, keep_count):
    selected_sids = {item["sid"] for item in scored_storms[:keep_count]}
    subset = [point for point in points if point["sid"] in selected_sids]
    subset.sort(key=lambda row: row["time_hours"])

    relabel = {}
    next_label = 0
    for point in subset:
        sid = point["sid"]
        if sid not in relabel:
            relabel[sid] = next_label
            next_label += 1

    for point in subset:
        point["subset_true_label"] = relabel[point["sid"]]
    return subset


def build_features(points):
    lon_periodic = np.array([point["lon_periodic"] for point in points], dtype=np.float32)
    lat = np.array([point["lat"] for point in points], dtype=np.float32)
    time_hours = np.array([point["time_hours"] for point in points], dtype=np.float32)
    tau = (time_hours - time_hours.min()) * TIME_SCALE
    x = np.column_stack([lon_periodic, lat, tau]).astype(np.float32)
    y_true = np.array([point["subset_true_label"] for point in points], dtype=np.int32)
    return x, y_true


def pbc_bounds(x):
    lat_min = float(np.min(x[:, 1]) - EPS * 10.0)
    lat_max = float(np.max(x[:, 1]) + EPS * 10.0)
    tau_min = float(np.min(x[:, 2]) - EPS * 10.0)
    tau_max = float(np.max(x[:, 2]) + EPS * 10.0)
    return [0.0, lat_min, tau_min], [360.0, lat_max, tau_max]


def summarize_labels(labels):
    labels = np.asarray(labels)
    valid = labels[labels >= 0]
    return {
        "clusters": int(np.unique(valid).size),
        "noise_points": int(np.sum(labels < 0)),
    }


def benchmark(factory, fit_callable, repeats=REPEATS):
    runtimes = []
    peaks = []
    labels = None
    for _ in range(repeats):
        model = factory()
        tracemalloc.start()
        t0 = time.perf_counter()
        fit_callable(model)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        runtimes.append(elapsed)
        peaks.append(peak / (1024 * 1024))
        labels = model.labels_.copy()
    return labels, float(np.mean(runtimes)), float(np.std(runtimes, ddof=0)), float(np.mean(peaks)), float(np.std(peaks, ddof=0))


def plot_fig6(points):
    by_storm = defaultdict(list)
    for point in points:
        by_storm[point["sid"]].append(point)
    for sid in by_storm:
        by_storm[sid].sort(key=lambda row: row["time_hours"])

    def shifted_lon(lon):
        return lon + 360.0 if lon < 0 else lon

    def split_segments(rows, jump_threshold=20.0):
        segments = [[]]
        for point in rows:
            current = {**point, "shifted_lon": shifted_lon(point["lon"])}
            prev = segments[-1][-1] if segments[-1] else None
            if prev is not None and abs(current["shifted_lon"] - prev["shifted_lon"]) > jump_threshold:
                segments.append([])
            segments[-1].append(current)
        return [seg for seg in segments if len(seg) >= 2]

    lats = [p["lat"] for p in points]
    lat_min = min(lats) - 2
    lat_max = max(lats) + 2

    fig, ax = plt.subplots(figsize=(11, 6.8), dpi=200)
    cmap = plt.get_cmap("tab20")
    shown = 0
    legend_handles = []

    for _, rows in by_storm.items():
        color = cmap(rows[0]["true_label"] % 20)
        for seg in split_segments(rows):
            ax.plot(
                [p["shifted_lon"] for p in seg],
                [p["lat"] for p in seg],
                color=color,
                linewidth=2.0,
                alpha=0.92,
            )
        sampled = rows[::6] if len(rows) > 6 else rows
        ax.scatter(
            [shifted_lon(p["lon"]) for p in sampled],
            [p["lat"] for p in sampled],
            s=10,
            color=[color],
            alpha=0.95,
            zorder=3,
        )
        if shown < 10:
            handle = plt.Line2D([0], [0], color=color, lw=2.5, label=f"{rows[0]['name']} ({len(rows)})")
            legend_handles.append(handle)
            shown += 1

    ax.axvline(180.0, color="#444444", linewidth=1.4, linestyle=(0, (5, 4)))
    ax.text(180.5, lat_max - 1.5, "dateline seam", color="#444444", fontsize=11, va="top")

    ax.set_xlim(160, 200)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xticks([160, 170, 180, 190, 200])
    ax.set_xticklabels(["160 deg E", "170 deg E", "180 deg", "170 deg W", "160 deg W"], fontsize=11)
    ax.set_yticks([-30, -20, -10, 0, 10, 20, 30, 40])
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="x", color="#e6e6e6", linewidth=1.0)
    ax.grid(axis="y", color="#f0f0f0", linewidth=1.0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)

    ax.set_title("IBTrACS Dateline-Crossing Storm Tracks", fontsize=18, pad=18)
    fig.text(
        0.5,
        0.93,
        "17 real storms, 1321 track points, longitude seam at the International Date Line",
        ha="center",
        va="center",
        fontsize=11,
        color="#555555",
    )
    ax.set_xlabel("longitude around the dateline", fontsize=12, labelpad=12)
    ax.set_ylabel("latitude", fontsize=12, labelpad=10)
    leg = ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=9.5,
        frameon=False,
        ncol=2,
        borderaxespad=0.6,
        handlelength=2.0,
        columnspacing=1.2,
    )
    for line in leg.get_lines():
        line.set_linewidth(2.5)

    plt.tight_layout(rect=[0.03, 0.04, 0.98, 0.90])
    fig.savefig(FIG6_PNG, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_fig7(rows):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))

    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        method_rows.sort(key=lambda row: row["n_points"])
        x = [row["n_points"] for row in method_rows]

        axes[0].errorbar(
            x,
            [row["runtime_mean_s"] for row in method_rows],
            yerr=[row["runtime_std_s"] for row in method_rows],
            marker=MARKERS[method],
            linewidth=2.0,
            markersize=6,
            capsize=3,
            color=COLORS[method],
            label=method,
        )
        axes[1].errorbar(
            x,
            [row["peak_memory_mb_mean"] for row in method_rows],
            yerr=[row["peak_memory_mb_std"] for row in method_rows],
            marker=MARKERS[method],
            linewidth=2.0,
            markersize=6,
            capsize=3,
            color=COLORS[method],
            label=method,
        )

    axes[0].set_title("Runtime vs. Subset Size", fontsize=16)
    axes[0].set_xlabel("Subset size n", fontsize=15)
    axes[0].set_ylabel("Runtime (s)", fontsize=15)

    axes[1].set_title("Peak Memory vs. Subset Size", fontsize=16)
    axes[1].set_xlabel("Subset size n", fontsize=15)
    axes[1].set_ylabel("Peak Python memory (MB)", fontsize=15)

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", direction="in", labelsize=13)
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.1)
        ax.legend(frameon=False, fontsize=12, loc="upper left")

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.90, wspace=0.25)
    fig.savefig(FIG7_PNG, dpi=300, facecolor="white")
    plt.close(fig)


def main():
    points = load_points()
    scored = score_storms_by_boundary_ratio(points)
    all_rows = []
    summary = {}

    for keep_count in KEEP_COUNTS:
        subset = build_cumulative_subset(points, scored, keep_count)
        x, y_true = build_features(subset)
        lower, upper = pbc_bounds(x)

        results = {
            "DBSCAN": benchmark(
                lambda: DBSCAN(eps=EPS, min_samples=MIN_SAMPLES),
                lambda model: model.fit(x),
            ),
            "PBC-DBSCAN": benchmark(
                lambda: DBSCAN_PBC(eps=EPS, min_samples=MIN_SAMPLES),
                lambda model: model.fit(x.copy(), pbc_lower=lower, pbc_upper=upper),
            ),
            "GPDBSCAN": benchmark(
                lambda: GPDBSCAN(eps=EPS, min_samples=MIN_SAMPLES),
                lambda model: model.fit(x.copy(), pbc_lower=lower, pbc_upper=upper),
            ),
        }

        subset_boundary_ratio = float(
            np.mean((x[:, 0] < EPS) | (x[:, 0] > 360.0 - EPS))
        )

        summary[keep_count] = {
            "n_points": int(len(subset)),
            "boundary_ratio": subset_boundary_ratio,
            "results": {},
        }

        for method, (labels, runtime_mean, runtime_std, peak_mean, peak_std) in results.items():
            label_stats = summarize_labels(labels)
            row = {
                "selected_storms": int(keep_count),
                "n_points": int(len(subset)),
                "boundary_ratio": subset_boundary_ratio,
                "method": method,
                "clusters": label_stats["clusters"],
                "noise_points": label_stats["noise_points"],
                "runtime_mean_s": runtime_mean,
                "runtime_std_s": runtime_std,
                "peak_memory_mb_mean": peak_mean,
                "peak_memory_mb_std": peak_std,
            }
            all_rows.append(row)
            summary[keep_count]["results"][method] = row

    fieldnames = [
        "selected_storms",
        "n_points",
        "boundary_ratio",
        "method",
        "clusters",
        "noise_points",
        "runtime_mean_s",
        "runtime_std_s",
        "peak_memory_mb_mean",
        "peak_memory_mb_std",
    ]
    with TABLE5_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    TABLE5_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_fig6(points)
    plot_fig7(all_rows)

    print(
        json.dumps(
            {
                "table5_csv": str(TABLE5_CSV),
                "table5_json": str(TABLE5_JSON),
                "fig6": str(FIG6_PNG),
                "fig7": str(FIG7_PNG),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
