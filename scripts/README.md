# Command-Line Scripts

This directory will contain thin command-line entry points for:

- `train`: run or resume one configured training experiment.
- `evaluate`: evaluate a selected checkpoint using the locked protocol.
- `aggregate_results`: combine verified run artifacts across variants and seeds.

The scripts will call reusable package code from `src/densenet_experiments`. No Python scripts are implemented at this stage.
