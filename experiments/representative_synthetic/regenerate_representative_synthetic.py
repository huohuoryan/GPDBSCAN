import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from dbscan_pbc.dbscan_pbc import DBSCAN_PBC
from dbscan_pbc.dbscan_pbc_grid import GPDBSCAN
from dbscan_pbc.kmeans_pbc import KMeansPBC
from synthetic_external_metrics import (
    evaluate_dataset,
    generate_double_boundary,
    generate_four_boundary,
    generate_nonconvex,
    generate_periodic_3d,
)


BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "representative_synthetic_outputs"


def dataset_specs():
    return [
        generate_double_boundary(),
        generate_four_boundary(),
        generate_nonconvex(),
        generate_periodic_3d(),
    ]


def build_parameter_table(specs):
    rows = []
    for spec in specs:
        params = spec["params"]
        rows.append(
            {
                "Dataset": spec["dataset"],
                "Points n": int(spec["x"].shape[0]),
                "Dimension d": int(spec["x"].shape[1]),
                "True clusters": int(spec["true_clusters"]),
                "eps": params["eps"],
                "min_samples": params["min_samples"],
                "K-PBC k": params["k_pbc_clusters"],
            }
        )
    return pd.DataFrame(rows)


def run_methods_once(x, params):
    dims = x.shape[1]
    periodic_lower = params["pbc_lower"]
    periodic_upper = params["pbc_upper"]

    rows = []

    methods = [
        (
            "DBSCAN",
            lambda: DBSCAN(eps=params["eps"], min_samples=params["min_samples"]).fit(x.copy()),
        ),
        (
            "PBC-DBSCAN",
            lambda: DBSCAN_PBC(eps=params["eps"], min_samples=params["min_samples"]).fit(
                x.copy(), pbc_lower=periodic_lower, pbc_upper=periodic_upper
            ),
        ),
        (
            "K-PBC",
            lambda: KMeansPBC(
                n_clusters=params["k_pbc_clusters"],
                box_lengths=[1.0] * dims,
                random_state=42,
            ).fit(x.copy()),
        ),
        (
            "GPDBSCAN",
            lambda: GPDBSCAN(eps=params["eps"], min_samples=params["min_samples"]).fit(
                x.copy(), pbc_lower=periodic_lower, pbc_upper=periodic_upper
            ),
        ),
    ]

    for method, runner in methods:
        runner()
        start = time.perf_counter()
        runner()
        elapsed = time.perf_counter() - start
        rows.append((method, elapsed))

    return rows


def build_runtime_table(specs, repeats=5):
    rows = []
    for spec in specs:
        per_method = {}
        for _ in range(repeats):
            for method, elapsed in run_methods_once(spec["x"], spec["params"]):
                per_method.setdefault(method, []).append(elapsed)

        for method, values in per_method.items():
            rows.append(
                {
                    "Dataset": spec["dataset"],
                    "Method": method,
                    "Runtime mean (s)": float(np.mean(values)),
                    "Runtime std (s)": float(np.std(values, ddof=1) if len(values) > 1 else 0.0),
                }
            )
    return pd.DataFrame(rows)


def to_markdown(df, float_formats=None):
    pretty = df.copy()
    float_formats = float_formats or {}
    for col, fmt in float_formats.items():
        if col in pretty.columns:
            pretty[col] = pretty[col].map(lambda v: fmt.format(v))

    headers = list(pretty.columns)
    rows = [headers] + pretty.astype(str).values.tolist()
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(headers))]

    def fmt_row(row):
        return "| " + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) + " |"

    separator = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    lines = [fmt_row(headers), separator]
    lines.extend(fmt_row(row) for row in pretty.astype(str).values.tolist())
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = dataset_specs()

    params_df = build_parameter_table(specs)
    metrics_rows = []
    for spec in specs:
        metrics_rows.extend(evaluate_dataset(spec))
    metrics_df = pd.DataFrame(metrics_rows)
    runtime_df = build_runtime_table(specs, repeats=5)

    params_csv = OUT_DIR / "table2_exact_parameter_settings.csv"
    metrics_csv = OUT_DIR / "table3_external_clustering_evaluation.csv"
    runtime_csv = OUT_DIR / "table4_running_times.csv"
    params_md = OUT_DIR / "table2_exact_parameter_settings.md"
    metrics_md = OUT_DIR / "table3_external_clustering_evaluation.md"
    runtime_md = OUT_DIR / "table4_running_times.md"
    payload_json = OUT_DIR / "representative_synthetic_summary.json"

    params_df.to_csv(params_csv, index=False)
    metrics_df.to_csv(metrics_csv, index=False)
    runtime_df.to_csv(runtime_csv, index=False)

    params_md.write_text(
        to_markdown(
            params_df,
            {
                "eps": "{:.3f}",
            },
        )
        + "\n",
        encoding="utf-8",
    )
    metrics_md.write_text(
        to_markdown(
            metrics_df[
                [
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
            ],
            {
                "ARI": "{:.3f}",
                "NMI": "{:.3f}",
                "V_measure": "{:.3f}",
                "FMI": "{:.3f}",
            },
        )
        + "\n",
        encoding="utf-8",
    )
    runtime_md.write_text(
        to_markdown(
            runtime_df,
            {
                "Runtime mean (s)": "{:.4f}",
                "Runtime std (s)": "{:.4f}",
            },
        )
        + "\n",
        encoding="utf-8",
    )

    payload = {
        "parameters": params_df.to_dict(orient="records"),
        "metrics": metrics_df.to_dict(orient="records"),
        "runtime": runtime_df.to_dict(orient="records"),
    }
    payload_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved: {params_csv}")
    print(f"Saved: {metrics_csv}")
    print(f"Saved: {runtime_csv}")


if __name__ == "__main__":
    main()
