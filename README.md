The proposed method is GPDBSCAN. It uses grid partitioning to identify boundary-adjacent regions, selectively generates periodic replicas, applies DBSCAN to the augmented data, and merges periodic-equivalent cluster labels.

## Code

The main implementation is:

```text
src/dbscan_pbc/dbscan_pbc_grid.py
```
## Requirements

The code was written in Python and requires:

```text
numpy
pandas
scikit-learn
matplotlib
```

## Experiments

### Representative synthetic datasets

This reproduces Table 2, Fig. 3, Table 3, and Table 4.

```bash
cd experiments/representative_synthetic
python regenerate_representative_synthetic.py
python plot_fig3_synthetic_labeled.py
```

Main outputs:

```text
outputs/table2_exact_parameter_settings.csv
outputs/table3_external_clustering_evaluation.csv
outputs/table4_running_times.csv
outputs/fig3_clustering_results_synthetic_labeled.png
```

### Controlled synthetic scalability benchmarks

This reproduces Fig. 4 and Fig. 5.

```bash
cd experiments/controlled_benchmark
python run_controlled_synthetic_benchmarks.py
```

Default settings:

```text
Dataset-size scaling: d = 2, n = 500, 1000, 2000, 5000
Dimensionality scaling: n = 1000, d = 2, 3, 4, 5, 6
Repeated runs: 5
```

Main outputs:

```text
outputs/synthetic_scaling_vs_n.csv
outputs/synthetic_scaling_vs_d.csv
outputs/figures/fig4a_runtime_vs_n.png
outputs/figures/fig4b_memory_vs_n.png
outputs/figures/fig5_runtime_vs_d.png
outputs/figures/fig5_memory_vs_d.png
```

### Distribution-sensitivity experiment

```bash
cd experiments/distribution_sensitivity
python distribution_sensitivity_experiment.py
```

Main outputs:

```text
outputs/distribution_sensitivity_results.csv
outputs/distribution_sensitivity_results.json
outputs/distribution_sensitivity_speedup.png
```

### IBTrACS experiment

This reproduces the real-world tropical cyclone experiment.

```bash
cd experiments/ibtracs
python run_ibtracs_python_benchmark.py
python plot_dateline_overview_matplotlib.py
```

Main outputs:

```text
outputs/table6_ibtracs_python_results.csv
outputs/table6_ibtracs_python_results.json
outputs/ibtracs_dateline_overview_matplotlib.png
outputs/ibtracs_runtime_memory_scaling_pythonProject.png
```

The IBTrACS data source and filtering procedure are described in:

```text
data/ibtracs/source_note.txt
```

## Notes

Runtime is measured using `time.perf_counter()`. Peak Python memory is measured using `tracemalloc`. Differences from the reported values may occur because of hardware, Python version, and operating-system differences.
