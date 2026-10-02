# Contributing

Start with a reproducible failure: an incident, the expected result and the model
assumption that failed. Synthetic cases are welcome; strip internal identifiers
and obtain permission before contributing real telemetry.

```bash
python -m pip install '.[bench]'
python -m unittest discover -s tests -v
```

For algorithm changes, compare the same held-out topologies and measurement
budget. Keep truth outside the inference input. Report degraded scenarios and
wrong high-confidence decisions, not only mean accuracy. Do not tune on the
published test incidents and present the result as held-out performance.

Keep the runtime dependency set small. Figures and benchmark packages are optional.
Changes to methods or defaults need updated experiment evidence; documentation
edits do not require a fresh timing benchmark.

Reinstall after editing source, or use an editable installation in an environment
that processes `.pth` files.

Contributions use the MIT license. Do not include credentials, private network
captures, customer topology or model weights without the right to distribute them.
