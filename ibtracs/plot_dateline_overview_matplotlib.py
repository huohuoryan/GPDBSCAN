import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


BASE = Path(r"D:\广州工商学院\论文研究\我的论文撰写\基于网格化预处理的周期性边界DBSCAN算法\chathelp\ibtracs")
POINTS_CSV = BASE / "experiment" / "ibtracs_experiment_points.csv"
OUT_PNG = BASE / "experiment" / "ibtracs_dateline_overview_matplotlib.png"


def load_points():
    points = []
    with POINTS_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append(
                {
                    "sid": row["sid"],
                    "name": row["name"],
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "time_hours": float(row["time_hours"]),
                    "true_label": int(row["true_label"]),
                }
            )
    return points


def shifted_lon(lon):
    return lon + 360 if lon < 0 else lon


def split_segments(rows, jump_threshold=20):
    segments = [[]]
    for p in rows:
        current = {**p, "shifted_lon": shifted_lon(p["lon"])}
        prev = segments[-1][-1] if segments[-1] else None
        if prev is not None and abs(current["shifted_lon"] - prev["shifted_lon"]) > jump_threshold:
            segments.append([])
        segments[-1].append(current)
    return [seg for seg in segments if len(seg) >= 2]


def main():
    points = load_points()
    by_storm = defaultdict(list)
    for p in points:
        by_storm[p["sid"]].append(p)
    for sid in by_storm:
        by_storm[sid].sort(key=lambda x: x["time_hours"])

    lats = [p["lat"] for p in points]
    lat_min = min(lats) - 2
    lat_max = max(lats) + 2

    fig, ax = plt.subplots(figsize=(11, 6.8), dpi=200)
    cmap = plt.get_cmap("tab20")

    legend_handles = []
    shown = 0
    for sid, rows in by_storm.items():
        color = cmap(rows[0]["true_label"] % 20)
        segments = split_segments(rows)
        for seg in segments:
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
            legend_handles.append(Line2D([0], [0], color=color, lw=2.5, label=f"{rows[0]['name']} ({len(rows)})"))
            shown += 1

    ax.axvline(180, color="#444444", linewidth=1.4, linestyle=(0, (5, 4)))
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
    fig.savefig(OUT_PNG, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(str(OUT_PNG))


if __name__ == "__main__":
    main()
