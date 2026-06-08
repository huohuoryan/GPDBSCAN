import json
from pathlib import Path

import numpy as np
import pandas as pd
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


OUTPUT_CSV = Path(__file__).with_name("synthetic_external_metrics.csv")
OUTPUT_JSON = Path(__file__).with_name("synthetic_external_metrics.json")
OUTPUT_MD = Path(__file__).with_name("synthetic_external_metrics_table.md")


def generate_double_boundary():
    np.random.seed(42)
    n = 500
    center1 = np.random.normal(loc=[0.5, 0.5], scale=0.07, size=(n, 2))
    center2 = np.random.normal(loc=[0.02, 0.5], scale=0.07, size=(n, 2))
    center3 = np.random.normal(loc=[0.98, 0.5], scale=0.07, size=(n, 2))
    x = np.vstack([center1, center2, center3])
    x = np.clip(x, 0, 1)
    y_true = np.concatenate(
        [
            np.zeros(n, dtype=int),
            np.ones(n, dtype=int),
            np.ones(n, dtype=int),
        ]
    )
    params = {
        "eps": 0.05,
        "min_samples": 3,
        "k_pbc_clusters": 2,
        "pbc_lower": [0.0, 0.0],
        "pbc_upper": [1.0, 1.0],
    }
    return {
        "dataset": "2D Double-Boundary",
        "x": x,
        "y_true": y_true,
        "params": params,
        "true_clusters": 2,
    }


def generate_four_boundary():
    np.random.seed(42)
    n = 200
    scale = 0.1
    center_left = np.random.normal(loc=[0.02, 0.5], scale=scale, size=(n, 2))
    center_right = np.random.normal(loc=[0.98, 0.5], scale=scale, size=(n, 2))
    center_bottom = np.random.normal(loc=[0.5, 0.02], scale=scale, size=(n, 2))
    center_top = np.random.normal(loc=[0.5, 0.98], scale=scale, size=(n, 2))
    center_middle = np.random.normal(loc=[0.5, 0.5], scale=scale, size=(n, 2))
    x = np.vstack([center_left, center_right, center_bottom, center_top, center_middle])
    x = np.clip(x, 0, 1)
    y_true = np.concatenate(
        [
            np.zeros(n, dtype=int),
            np.zeros(n, dtype=int),
            np.ones(n, dtype=int),
            np.ones(n, dtype=int),
            np.full(n, 2, dtype=int),
        ]
    )
    params = {
        "eps": 0.05,
        "min_samples": 5,
        "k_pbc_clusters": 3,
        "pbc_lower": [0.0, 0.0],
        "pbc_upper": [1.0, 1.0],
    }
    return {
        "dataset": "2D Four-Boundary",
        "x": x,
        "y_true": y_true,
        "params": params,
        "true_clusters": 3,
    }


def generate_nonconvex():
    np.random.seed(42)

    def generate_base_data(n=300, scale=0.04):
        centers = {
            "left": [0.02, 0.5],
            "right": [0.98, 0.5],
            "bottom": [0.5, 0.02],
            "top": [0.5, 0.98],
            "middle": [0.5, 0.5],
        }
        x = np.vstack(
            [
                np.random.normal(loc=centers["left"], scale=scale, size=(n, 2)),
                np.random.normal(loc=centers["right"], scale=scale, size=(n, 2)),
                np.random.normal(loc=centers["bottom"], scale=scale, size=(n, 2)),
                np.random.normal(loc=centers["top"], scale=scale, size=(n, 2)),
                np.random.normal(loc=centers["middle"], scale=scale, size=(n, 2)),
            ]
        )
        return np.clip(x, 0, 1)

    def transform_with_labels(x):
        center = np.array([0.5, 0.5])
        distances = np.linalg.norm(x - center, axis=1)
        mask = (distances > 0.2) & (distances < 0.3)

        y = np.concatenate(
            [
                np.zeros(300, dtype=int),
                np.zeros(300, dtype=int),
                np.ones(300, dtype=int),
                np.ones(300, dtype=int),
                np.full(300, 2, dtype=int),
            ]
        )

        density_zones = {
            "sparse": (0.1, 0.2),
            "dense": (0.03, 0.1),
        }

        for i, point in enumerate(x):
            if point[0] < 0.5 and point[1] < 0.5:
                scale = np.random.uniform(*density_zones["dense"])
            else:
                scale = np.random.uniform(*density_zones["sparse"])
            direction = point - center
            unit_vector = direction / (np.linalg.norm(direction) + 1e-8)
            x[i] += unit_vector * scale

        bridge_points = x[mask]
        if len(bridge_points) > 0:
            theta = np.linspace(0, 2 * np.pi, len(bridge_points))
            bridge_points[:, 0] = 0.4 + 0.1 * np.cos(theta)
            bridge_points[:, 1] = 0.4 + 0.1 * np.sin(theta)
            x[mask] = bridge_points

        # The bridge/ring points form the intended non-convex central structure.
        y[mask] = 2

        x = np.concatenate([x, x + [1, 0], x + [0, 1], x + [1, 1]])
        x = x % 1
        y = np.tile(y, 4)
        return x, y

    base_x = generate_base_data()
    x, y_true = transform_with_labels(base_x)
    params = {
        "eps": 0.035,
        "min_samples": 3,
        "k_pbc_clusters": 3,
        "pbc_lower": [0.0, 0.0],
        "pbc_upper": [1.0, 1.0],
    }
    return {
        "dataset": "2D Non-convex",
        "x": x,
        "y_true": y_true,
        "params": params,
        "true_clusters": 3,
    }


def generate_periodic_3d():
    np.random.seed(42)
    n = 500
    scale = 0.07
    centers = np.array(
        [
            [0.5, 0.5, 0.5],
            [0.02, 0.5, 0.5],
            [0.98, 0.5, 0.5],
            [0.5, 0.02, 0.5],
            [0.5, 0.98, 0.5],
            [0.5, 0.5, 0.02],
            [0.5, 0.5, 0.98],
        ]
    )
    points = []
    labels = []
    for idx, center in enumerate(centers):
        cluster = np.random.normal(loc=center, scale=scale, size=(n, 3))
        points.append(cluster)
        if idx == 0:
            labels.append(np.full(n, 0, dtype=int))
        elif idx in (1, 2):
            labels.append(np.full(n, 1, dtype=int))
        elif idx in (3, 4):
            labels.append(np.full(n, 2, dtype=int))
        else:
            labels.append(np.full(n, 3, dtype=int))

    x = np.vstack(points)
    x = np.clip(x, 0, 1)
    y_true = np.concatenate(labels)
    params = {
        "eps": 0.08,
        "min_samples": 5,
        "k_pbc_clusters": 4,
        "pbc_lower": [0.0, 0.0, 0.0],
        "pbc_upper": [1.0, 1.0, 1.0],
    }
    return {
        "dataset": "3D Periodic",
        "x": x,
        "y_true": y_true,
        "params": params,
        "true_clusters": 4,
    }


def run_methods(x, params):
    dims = x.shape[1]
    periodic_lower = params["pbc_lower"]
    periodic_upper = params["pbc_upper"]

    methods = {}

    dbscan = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
    dbscan.fit(x.copy())
    methods["DBSCAN"] = dbscan.labels_.copy()

    pbc = DBSCAN_PBC(eps=params["eps"], min_samples=params["min_samples"])
    pbc.fit(x.copy(), pbc_lower=periodic_lower, pbc_upper=periodic_upper)
    methods["PBC-DBSCAN"] = pbc.labels_.copy()

    k_pbc = KMeansPBC(
        n_clusters=params["k_pbc_clusters"],
        box_lengths=[1.0] * dims,
        random_state=42,
    )
    k_pbc.fit(x.copy())
    methods["K-PBC"] = k_pbc.labels_.copy()

    gp = GPDBSCAN(eps=params["eps"], min_samples=params["min_samples"])
    gp.fit(x.copy(), pbc_lower=periodic_lower, pbc_upper=periodic_upper)
    methods["GPDBSCAN"] = gp.labels_.copy()

    return methods


def evaluate_dataset(spec):
    x = spec["x"]
    y_true = spec["y_true"]
    methods = run_methods(x, spec["params"])
    rows = []

    for method, labels in methods.items():
        non_noise = labels[labels >= 0]
        rows.append(
            {
                "dataset": spec["dataset"],
                "n_points": int(x.shape[0]),
                "dimension": int(x.shape[1]),
                "true_clusters": int(spec["true_clusters"]),
                "method": method,
                "pred_clusters": int(len(np.unique(non_noise))),
                "noise_points": int(np.sum(labels == -1)),
                "ARI": float(adjusted_rand_score(y_true, labels)),
                "NMI": float(normalized_mutual_info_score(y_true, labels)),
                "V_measure": float(v_measure_score(y_true, labels)),
                "FMI": float(fowlkes_mallows_score(y_true, labels)),
                "eps": spec["params"]["eps"],
                "min_samples": spec["params"]["min_samples"],
                "k_pbc_clusters": spec["params"]["k_pbc_clusters"],
            }
        )

    return rows


def build_markdown_table(df):
    display_cols = [
        "dataset",
        "method",
        "n_points",
        "dimension",
        "true_clusters",
        "pred_clusters",
        "noise_points",
        "ARI",
        "NMI",
        "V_measure",
        "FMI",
    ]
    pretty = df[display_cols].copy()
    for metric in ["ARI", "NMI", "V_measure", "FMI"]:
        pretty[metric] = pretty[metric].map(lambda v: f"{v:.3f}")

    headers = list(pretty.columns)
    rows = [headers] + pretty.values.tolist()
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(headers))]

    def fmt_row(row):
        return "| " + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) + " |"

    separator = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    lines = [fmt_row(headers), separator]
    lines.extend(fmt_row(row) for row in pretty.values.tolist())
    return "\n".join(lines)


def main():
    datasets = [
        generate_double_boundary(),
        generate_four_boundary(),
        generate_nonconvex(),
        generate_periodic_3d(),
    ]

    rows = []
    for spec in datasets:
        rows.extend(evaluate_dataset(spec))

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    json_payload = {
        "datasets": [
            {
                "dataset": spec["dataset"],
                "n_points": int(spec["x"].shape[0]),
                "dimension": int(spec["x"].shape[1]),
                "true_clusters": int(spec["true_clusters"]),
                "params": spec["params"],
            }
            for spec in datasets
        ],
        "results": df.to_dict(orient="records"),
    }
    OUTPUT_JSON.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    markdown = "# External Clustering Metrics on the Representative Synthetic Datasets\n\n"
    markdown += build_markdown_table(df)
    markdown += "\n"
    OUTPUT_MD.write_text(markdown, encoding="utf-8")

    print(markdown)


if __name__ == "__main__":
    main()
