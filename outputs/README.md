# Experiment Outputs

Generated runs will follow this conceptual structure:

```text
outputs/<experiment-id>/
|-- config
|-- environment metadata
|-- metrics.csv
|-- best checkpoint
|-- last checkpoint
`-- test metrics
```

These artifacts must be produced by real runs. This directory must not contain fabricated, placeholder, or mock results.
