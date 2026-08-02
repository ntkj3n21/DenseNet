# Training Configurations

The training configurations define three execution levels that share one
optimizer, scheduler, data, selection, and evaluation protocol.

- `smoke.json` runs one seed for one epoch and checks pipeline integration
  only. It does not produce research results.
- `pilot.json` runs one seed for 20 epochs to detect errors and estimate
  runtime. It does not produce official research results.
- `final.json` runs the five locked seeds for 200 epochs and is the only level
  intended to produce research results.

All four ablation variants must use the same selected training configuration
and seed set. Validation accuracy alone selects the best checkpoint. The
official test set is evaluated once, using that best checkpoint, after model
selection. AMP is currently disabled.

Once official final runs begin, this protocol must not change. Any subsequent
change requires a new explicitly versioned protocol and configuration set.
