# Notebooks

This directory is reserved for Google Colab wrappers and exploratory
post-run analysis.

Authoritative experiment logic does not live in notebooks. Model
implementations, dataset construction, training and evaluation behavior, and
reproducibility utilities are maintained in:

```text
src/densenet_experiments/
```

Completed benchmark artifacts are tracked under:

```text
results/
```

Lightweight result-processing utilities are available under:

```text
scripts/
```

In particular:

```text
scripts/aggregate_results.py
scripts/plot_results.py
```

The current portfolio workflow uses Google Colab as the compute environment
for training and final checkpoint evaluation, while the reusable experiment
logic remains in the Python package.

Notebooks should not contain duplicated model implementations, authoritative
training loops, fabricated metrics, or hardcoded experimental results.