import argparse
import csv
import gc
import json
import time
import tracemalloc
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FormatStrFormatter
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
    v_measure_score,
)

from dbscan_pbc.dbscan_pbc import DBSCAN_PBC
from dbscan_pbc.dbscan_pbc_grid import GPDBSCAN
from dbscan_pbc.kmeans_pbc import KMeansPBC


METHOD_ORDER = ["DBSCAN", "PBC-DBSCAN", "K-PBC", "GPDBSCAN"]
PLOT_STYLES = {
    "DBSCAN": {"marker": "o", "linewidth": 2.0},
    "PBC-DBSCAN": {"marker": "s", "linewidth": 2.0},
    "K-PBC": {"marker": "^", "linewidth": 2.0},
    "GPDBSCAN": {"marker": "D", "linewidth": 2.2},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run controlled synthetic scalability benchmarks for DBSCAN, "
            "PBC-DBSCAN, K-PBC, and GPDBSCAN using the pythonProject implementation."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "controlled_synthetic_benchmark"),
        help="Directory used to save CSV, JSON, and figure outputs.",
    )
    parser.add_argument("--n-values", nargs="+", type=int, default=[500, 1000, 2000, 5000])
    parser.add_argument("--d-values", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    parser.add_argument("--fixed-d", type=int, default=2)
    parser.add_argument("--fixed-n", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--box-length", type=float, default=1.0)
    parser.add_argument("--boundary-offset", type=float, default=0.02)
    parser.add_argument("--cluster-std", type=float, default=0.03)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--base-eps-2d", type=float, default=0.10)
    parser.add_argument("--dataset-shape", choices=["filament", "gaussian"], default="gaussian")
    parser.add_argument(
        "--center-fraction",
        type=float,
        default=0.95,
        help=(
            "Fraction of points assigned to the interior cluster. The remaining points "
            "are distributed evenly across the d boundary-spanning clusters."
        ),
    )
    return parser.parse_args()


def split_counts(total, parts):
    counts = [total // parts] * parts
    for idx in range(total % parts):
        counts[idx] += 1
    return counts


def split_boundary_center_counts(total, d, center_fraction):
    center_count = int(round(total * center_fraction))
    center_count = max(1, min(total - d, center_count))
    boundary_total = total - center_count
    boundary_counts = split_counts(boundary_total, d)
    return boundary_counts + [center_count]


def generate_periodic_dataset(
    n_points,
    d,
    box_length,
    boundary_offset,
    cluster_std,
    seed,
    dataset_shape,
    center_fraction,
):
    rng = np.random.default_rng(seed)
    true_cluster_count = d + 1
    cluster_sizes = split_boundary_center_counts(n_points, d, center_fraction)

    x_parts = []
    y_parts = []
    middle = np.full(d, box_length * 0.5, dtype=np.float32)

    if dataset_shape == "gaussian":
        for dim in range(d):
            cluster_size = cluster_sizes[dim]
            left_size = cluster_size // 2
            right_size = cluster_size - left_size

            left_center = middle.copy()
            right_center = middle.copy()
            left_center[dim] = boundary_offset * box_length
            right_center[dim] = (1.0 - boundary_offset) * box_length

            if left_size:
                left = rng.normal(loc=left_center, scale=cluster_std, size=(left_size, d))
                x_parts.append(left)
                y_parts.append(np.full(left_size, dim, dtype=np.int32))
            if right_size:
                right = rng.normal(loc=right_center, scale=cluster_std, size=(right_size, d))
                x_parts.append(right)
                y_parts.append(np.full(right_size, dim, dtype=np.int32))

        center_cluster = rng.normal(loc=middle, scale=cluster_std, size=(cluster_sizes[-1], d))
        x_parts.append(center_cluster)
        y_parts.append(np.full(cluster_sizes[-1], d, dtype=np.int32))
    else:
        line_half_span = 0.28 * box_length
        thickness = cluster_std * 0.35

        for dim in range(d):
            cluster_size = cluster_sizes[dim]
            left_size = cluster_size // 2
            right_size = cluster_size - left_size
            support_dim = (dim + 1) % d

            for size, seam_center in [
                (left_size, boundary_offset * box_length),
                (right_size, (1.0 - boundary_offset) * box_length),
            ]:
                if not size:
                    continue
                points = np.tile(middle, (size, 1))
                points += rng.normal(loc=0.0, scale=thickness, size=(size, d))
                points[:, dim] = seam_center + rng.normal(loc=0.0, scale=thickness, size=size)
                points[:, support_dim] = (
                    middle[support_dim]
                    + rng.uniform(-line_half_span, line_half_span, size=size)
                    + rng.normal(loc=0.0, scale=thickness, size=size)
                )
                x_parts.append(points)
                y_parts.append(np.full(size, dim, dtype=np.int32))

        center_size = cluster_sizes[-1]
        t = rng.uniform(-line_half_span, line_half_span, size=center_size)
        center_cluster = np.tile(middle, (center_size, 1))
        center_cluster += rng.normal(loc=0.0, scale=thickness, size=(center_size, d))
        center_cluster[:, 0] = middle[0] + t + rng.normal(loc=0.0, scale=thickness, size=center_size)
        if d > 1:
            center_cluster[:, 1] = middle[1] + 0.45 * t + rng.normal(
                loc=0.0, scale=thickness, size=center_size
            )
        x_parts.append(center_cluster)
        y_parts.append(np.full(center_size, d, dtype=np.int32))

    x = np.vstack(x_parts).astype(np.float32)
    x %= box_length
    y_true = np.concatenate(y_parts)
    order = rng.permutation(n_points)
    return x[order], y_true[order], true_cluster_count


def eps_for_dimension(base_eps_2d, d, dataset_shape):
    if dataset_shape == "filament":
        return float(base_eps_2d)
    return float(base_eps_2d * np.sqrt(d / 2.0))


def metrics_from_labels(y_true, y_pred):
    return {
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "V_measure": float(v_measure_score(y_true, y_pred)),
        "FMI": float(fowlkes_mallows_score(y_true, y_pred)),
    }


def summarize_labels(labels):
    labels = np.asarray(labels)
    valid = labels[labels >= 0]
    n_clusters = int(np.unique(valid).size)
    noise_points = int(np.sum(labels == -1))
    return {
        "n_clusters": n_clusters,
        "noise_points": noise_points,
        "noise_ratio": float(noise_points / labels.size),
    }


def benchmark_estimator(estimator_factory, fit_callable):
    warmup_estimator = estimator_factory()
    fit_callable(warmup_estimator)
    del warmup_estimator
    gc.collect()

    tracemalloc.start()
    start = time.perf_counter()
    estimator = estimator_factory()
    fit_callable(estimator)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return estimator.labels_.copy(), elapsed, peak / (1024 * 1024)


def aggregate_records(records):
    aggregated = []
    metrics = [
        "runtime_s",
        "peak_memory_mb",
        "ARI",
        "NMI",
        "V_measure",
        "FMI",
        "n_clusters",
        "noise_points",
        "noise_ratio",
    ]
    key_fields = [
        "experiment",
        "method",
        "n_points",
        "dimension",
        "eps",
        "min_samples",
        "true_clusters",
        "repeats",
    ]

    grouped = {}
    for record in records:
        key = tuple(record[field] for field in key_fields)
        grouped.setdefault(key, []).append(record)

    for key in sorted(grouped):
        rows = grouped[key]
        item = {field: key[idx] for idx, field in enumerate(key_fields)}
        for metric in metrics:
            values = np.array([row[metric] for row in rows], dtype=np.float64)
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(np.std(values, ddof=0))
        aggregated.append(item)

    return aggregated


def save_csv(rows, path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def style_axes(ax):
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=11)


def plot_runtime_vs_n(rows, output_path):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for method in METHOD_ORDER:
        method_rows = [row for row in rows if row["method"] == method]
        method_rows.sort(key=lambda row: row["n_points"])
        x = [row["n_points"] for row in method_rows]
        y = [row["runtime_s_mean"] for row in method_rows]
        yerr = [row["runtime_s_std"] for row in method_rows]
        ax.errorbar(x, y, yerr=yerr, capsize=3, label=method, **PLOT_STYLES[method])
    ax.set_xlabel("Dataset size n", fontsize=12)
    ax.set_ylabel("Runtime (s)", fontsize=12)
    ax.set_title("Runtime vs. Dataset Size", fontsize=14)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_memory_vs_n(rows, output_path):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for method in METHOD_ORDER:
        method_rows = [row for row in rows if row["method"] == method]
        method_rows.sort(key=lambda row: row["n_points"])
        x = [row["n_points"] for row in method_rows]
        y = [row["peak_memory_mb_mean"] for row in method_rows]
        yerr = [row["peak_memory_mb_std"] for row in method_rows]
        ax.errorbar(x, y, yerr=yerr, capsize=3, label=method, **PLOT_STYLES[method])
    ax.set_xlabel("Dataset size n", fontsize=12)
    ax.set_ylabel("Peak memory (MB)", fontsize=12)
    ax.set_title("Peak Memory vs. Dataset Size", fontsize=14)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_runtime_vs_d(rows, output_path):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for method in METHOD_ORDER:
        method_rows = [row for row in rows if row["method"] == method]
        method_rows.sort(key=lambda row: row["dimension"])
        x = [row["dimension"] for row in method_rows]
        y = [row["runtime_s_mean"] for row in method_rows]
        yerr = [row["runtime_s_std"] for row in method_rows]
        ax.errorbar(x, y, yerr=yerr, capsize=3, label=method, **PLOT_STYLES[method])
    ax.set_xlabel("Dimension d", fontsize=12)
    ax.set_ylabel("Runtime (s)", fontsize=12)
    ax.set_title("Runtime vs. Dimensionality", fontsize=14)
    dims = sorted({row["dimension"] for row in rows})
    ax.xaxis.set_major_locator(FixedLocator(dims))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    style_axes(ax)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_memory_vs_d(rows, output_path):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for method in METHOD_ORDER:
        method_rows = [row for row in rows if row["method"] == method]
        method_rows.sort(key=lambda row: row["dimension"])
        x = [row["dimension"] for row in method_rows]
        y = [row["peak_memory_mb_mean"] for row in method_rows]
        yerr = [row["peak_memory_mb_std"] for row in method_rows]
        ax.errorbar(x, y, yerr=yerr, capsize=3, label=method, **PLOT_STYLES[method])
    ax.set_xlabel("Dimension d", fontsize=12)
    ax.set_ylabel("Peak memory (MB)", fontsize=12)
    ax.set_title("Peak Memory vs. Dimensionality", fontsize=14)
    dims = sorted({row["dimension"] for row in rows})
    ax.xaxis.set_major_locator(FixedLocator(dims))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    style_axes(ax)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_configuration(
    experiment_name,
    n_points,
    d,
    repeats,
    seed,
    box_length,
    boundary_offset,
    cluster_std,
    base_eps_2d,
    min_samples,
    dataset_shape,
    center_fraction,
):
    eps = eps_for_dimension(base_eps_2d, d, dataset_shape)
    pbc_lower = [0.0] * d
    pbc_upper = [box_length] * d

    rows = []
    for repeat_idx in range(repeats):
        x, y_true, true_clusters = generate_periodic_dataset(
            n_points=n_points,
            d=d,
            box_length=box_length,
            boundary_offset=boundary_offset,
            cluster_std=cluster_std,
            seed=seed + repeat_idx,
            dataset_shape=dataset_shape,
            center_fraction=center_fraction,
        )

        methods = [
            ("DBSCAN", lambda: DBSCAN(eps=eps, min_samples=min_samples), lambda model: model.fit(x)),
            (
                "PBC-DBSCAN",
                lambda: DBSCAN_PBC(eps=eps, min_samples=min_samples),
                lambda model: model.fit(x.copy(), pbc_lower=pbc_lower, pbc_upper=pbc_upper),
            ),
            (
                "K-PBC",
                lambda: KMeansPBC(
                    n_clusters=true_clusters,
                    box_lengths=np.full(d, box_length, dtype=np.float32),
                    random_state=seed + repeat_idx,
                    max_iter=100,
                    tol=1e-4,
                ),
                lambda model: model.fit(x),
            ),
            (
                "GPDBSCAN",
                lambda: GPDBSCAN(eps=eps, min_samples=min_samples),
                lambda model: model.fit(x.copy(), pbc_lower=pbc_lower, pbc_upper=pbc_upper),
            ),
        ]

        for method_name, factory, fit_callable in methods:
            labels, runtime_s, peak_memory_mb = benchmark_estimator(factory, fit_callable)
            row = {
                "experiment": experiment_name,
                "repeat": repeat_idx,
                "method": method_name,
                "n_points": int(n_points),
                "dimension": int(d),
                "eps": float(eps),
                "min_samples": int(min_samples),
                "true_clusters": int(true_clusters),
                "repeats": int(repeats),
                "runtime_s": float(runtime_s),
                "peak_memory_mb": float(peak_memory_mb),
            }
            row.update(summarize_labels(labels))
            row.update(metrics_from_labels(y_true, labels))
            rows.append(row)

    return rows


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = []

    for n_points in args.n_values:
        raw_rows.extend(
            run_configuration(
                experiment_name="vary_n",
                n_points=n_points,
                d=args.fixed_d,
                repeats=args.repeats,
                seed=args.seed + 1000 * n_points,
                box_length=args.box_length,
                boundary_offset=args.boundary_offset,
                cluster_std=args.cluster_std,
                base_eps_2d=args.base_eps_2d,
                min_samples=args.min_samples,
                dataset_shape=args.dataset_shape,
                center_fraction=args.center_fraction,
            )
        )

    for d in args.d_values:
        raw_rows.extend(
            run_configuration(
                experiment_name="vary_d",
                n_points=args.fixed_n,
                d=d,
                repeats=args.repeats,
                seed=args.seed + 100000 * d,
                box_length=args.box_length,
                boundary_offset=args.boundary_offset,
                cluster_std=args.cluster_std,
                base_eps_2d=args.base_eps_2d,
                min_samples=args.min_samples,
                dataset_shape=args.dataset_shape,
                center_fraction=args.center_fraction,
            )
        )

    aggregated_rows = aggregate_records(raw_rows)
    vary_n_rows = [row for row in aggregated_rows if row["experiment"] == "vary_n"]
    vary_d_rows = [row for row in aggregated_rows if row["experiment"] == "vary_d"]

    save_csv(raw_rows, output_dir / "synthetic_benchmark_raw.csv")
    save_csv(vary_n_rows, output_dir / "synthetic_scaling_vs_n.csv")
    save_csv(vary_d_rows, output_dir / "synthetic_scaling_vs_d.csv")

    runtime_vs_n_path = figures_dir / "fig4a_runtime_vs_n.png"
    memory_vs_n_path = figures_dir / "fig4b_memory_vs_n.png"
    runtime_vs_d_path = figures_dir / "fig5_runtime_vs_d.png"
    memory_vs_d_path = figures_dir / "fig5_memory_vs_d.png"

    plot_runtime_vs_n(vary_n_rows, runtime_vs_n_path)
    plot_memory_vs_n(vary_n_rows, memory_vs_n_path)
    plot_runtime_vs_d(vary_d_rows, runtime_vs_d_path)
    plot_memory_vs_d(vary_d_rows, memory_vs_d_path)

    report = {
        "config": {
            "n_values": args.n_values,
            "d_values": args.d_values,
            "fixed_d": args.fixed_d,
            "fixed_n": args.fixed_n,
            "repeats": args.repeats,
            "seed": args.seed,
            "box_length": args.box_length,
            "boundary_offset": args.boundary_offset,
            "cluster_std": args.cluster_std,
            "base_eps_2d": args.base_eps_2d,
            "min_samples": args.min_samples,
            "dataset_shape": args.dataset_shape,
            "center_fraction": args.center_fraction,
            "eps_rule": (
                "eps(d) = base_eps_2d"
                if args.dataset_shape == "filament"
                else "eps(d) = base_eps_2d * sqrt(d / 2)"
            ),
            "true_cluster_rule": "d boundary-spanning clusters + 1 central cluster",
        },
        "files": {
            "raw_csv": str(output_dir / "synthetic_benchmark_raw.csv"),
            "vary_n_csv": str(output_dir / "synthetic_scaling_vs_n.csv"),
            "vary_d_csv": str(output_dir / "synthetic_scaling_vs_d.csv"),
            "fig4a": str(runtime_vs_n_path),
            "fig4b": str(memory_vs_n_path),
            "fig5": str(runtime_vs_d_path),
            "fig5b": str(memory_vs_d_path),
        },
        "vary_n": vary_n_rows,
        "vary_d": vary_d_rows,
    }

    report_path = output_dir / "synthetic_benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
