# Configuration Directory

This directory contains the machine-readable inputs to the experimental
protocol:

- `datasets/` defines dataset and preprocessing behavior.
- `splits/` stores verified split indices and checksums.
- `experiments/` defines the four controlled model ablation variants.
- `training/` defines smoke, pilot, and final training levels.

The four variants must use an identical training configuration and seed set.
Legacy accuracy values are historical artifacts and are unrelated to results
produced under these configurations.
