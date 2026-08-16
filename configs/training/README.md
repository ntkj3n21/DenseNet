# Training Configurations

The training configurations define four execution levels that share the same
dataset, model-ablation, optimizer, evaluation, and checkpoint-selection
principles.

## Execution levels

- `smoke.json`
  - 1 epoch
  - seed 42
  - pipeline integration check only
  - not used for performance conclusions

- `pilot.json`
  - 20 epochs
  - seed 42
  - completed
  - used as a preliminary benchmark and runtime estimate

- `extended_50.json`
  - 50 epochs
  - seed 42
  - completed
  - main portfolio benchmark reported in `docs/RESULTS.md`

- `final.json`
  - 200 epochs
  - seeds 42, 123, 2024, 3407, and 9999
  - originally designed as a larger research-scale protocol
  - not executed as part of the current portfolio study

All four ablation variants must use the same training configuration within a
given execution level.

Validation accuracy is the checkpoint-selection metric. The official CIFAR-10
test set is not used during epoch training or variant selection.

The completed pilot and extended benchmark results are stored under `results/`.
Raw result artifacts are preserved separately from derived summary tables and
figures.

The current portfolio conclusions are based on the completed 50-epoch,
single-seed benchmark. They should not be interpreted as multi-seed statistical
evidence.

Any future experiment that changes the training protocol should use a new,
explicitly versioned configuration rather than modifying completed benchmark
conditions retroactively.