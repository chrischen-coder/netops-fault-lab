# Experiment protocol and results

Measured on 2026-10-02. All incidents are generated simulations of probe outcomes,
not packet simulation, network emulation or production telemetry.
[Raw results](../benchmarks/results/v0.1/results.json) · [Predictions](../benchmarks/results/v0.1/predictions.csv) · [Active traces](../benchmarks/results/v0.1/active-probes.csv).

## Data and splits

Seeds: 7, 17, 29. Each seed generates 48 connected undirected graphs with 8–12 nodes:
a random spanning tree plus independently sampled extra edges. Use 24 graphs for
training and 24 for testing, with 12 incidents per graph. Healthy probability is
0.15; otherwise one link is selected uniformly. Each seed trains its own model.

Graph groups use four-iteration Weisfeiler–Lehman hashes without node labels.
No hash crosses train/test within a run. Hash collisions conservatively group
additional graphs together. Train/test still use the same graph generator family;
this is not a test on unseen network architectures or field distributions.

Each incident uses 14 sampled shortest paths and up to 12 unused candidate paths.
The inference engine receives those routes explicitly. Node names and link IDs
are not learning features; truth is stored outside the inference input.

The compressed JSONL datasets, their uncompressed SHA-256 hashes, graph split
lists, Bayesian parameters and logistic coefficients are committed with results.

## Methods

- **Failed-path votes:** count failed paths through each link. Rank healthy first
  only when there are no failed paths. No learning or use of passed-path evidence.
- **Logistic regression:** candidate-wise binary classification, with standardized
  features: on/off-path fail/pass counts, covered fractions, healthy indicator and
  inverse link count. Use default L2 regularization (`C=1`), `max_iter=1000`, and
  equal total candidate weight per incident. Fit scaling and coefficients only on
  training data. Rank by positive-class score; scores are not treated as a
  normalized multiclass posterior.
- **Bayesian:** fit affected/background failure rates and healthy prior from the
  same training incidents with Beta(1,1) smoothing; use exact single-fault inference.

No test-set tuning. The decision threshold is fixed at 0.9. This nominal generator
matches the Bayesian likelihood family; it is a favorable model check, not an
independent test of real-world adequacy.

## Stress conditions

| Condition | Affected-path failure | Background failure | Missing probability | Copies per record |
| --- | ---: | ---: | ---: | ---: |
| Training / nominal test | 0.90 | 0.03 | 0.10 | 1 |
| Noise shift | 0.65 | 0.15 | 0.10 | 1 |
| Missing data | 0.90 | 0.03 | 0.55 | 1 |
| Copied alarms | 0.90 | 0.03 | 0.10 | 4 identical copies |

Hold topology, truth, route selection and random draws fixed across test conditions.
Models remain trained on nominal data. Copied alarms are intentionally passed
as separate record IDs; no new measurement is made. This tests the failure of
conditional independence, not robustness to correctly deduplicated input.

| Condition | Bayesian Top-1 | Decision coverage | Error among accepted decisions | NLL |
| --- | ---: | ---: | ---: | ---: |
| Nominal | 72.3% | 40.2% | 2.6% | 0.766 |
| Noise shift | 37.2% | 25.5% | 38.2% | 2.391 |
| 55% missing | 49.7% | 15.4% | 4.5% | 1.426 |
| Copied alarms | 72.1% | 67.7% | 12.6% | 1.484 |

Acceptance requires probability at least 0.9 **and** a unique observed signature.
High confidence can be wrong when noise assumptions change. A clean synthetic
result cannot establish a deployment threshold.

## Active-probe comparison

Start both policies with the first three probe records, including any missing
outcomes. Allow up to four additional probes from the same candidate pool, each
with cost 1. Sample one noisy outcome for each candidate once, then share those
outcomes across both policies. Additional outcomes use nominal noise and have
no missingness. The selector never receives truth or the future outcome table.

Both policies start at 31.4% Top-1. After four additional probes, information gain
reaches 72.5%, random selection 57.2%. The paired difference is 15.3 percentage
points; 1,000 topology-group bootstrap samples give 11.8–18.5 points (95%). This
interval describes the sampled synthetic topologies and fitted runs, not field uncertainty.

The traces preserve selected probe IDs, outcomes, query counts, ranks and entropy.
The policy may stop if no supplied candidate has positive expected information.
The demo's scripted extra outcome is separate from this randomized experiment.

## Metrics and timing

Top-1 and Top-3 include healthy incidents. Ties are broken by identifier ordering;
MRR uses the resulting rank. Nominal topologies and truth are generated independently
of that tie rule. The JSON includes 500-sample topology-bootstrap Top-1 intervals.

For Bayesian posteriors, report NLL with a `1e-15` floor, the sum of squared class
probability errors (multiclass Brier), and top-label ECE with 10 equal-width bins.
Brier and NLL mix calibration and discrimination; neither alone proves calibration.
See [scikit-learn's calibration guide](https://scikit-learn.org/stable/modules/calibration.html).

Per-case timing covers ranking only, excluding parsing, model fitting, report
rendering, posterior diagnostics and active selection. It is recorded for inspection,
not a speed comparison or production throughput claim.

## Reproduce

```bash
python -m pip install '.[bench]'
netops-fault-lab benchmark --output runs/local
```

Use a new output directory. For the recorded Python 3.12 package versions:

```bash
python -m pip install -r benchmarks/requirements-python312.txt
```

Runtime/library versions are in the results. Timings vary across hosts. Reports,
figures and screenshots can be regenerated after installing the optional tools:

```bash
python -m pip install '.[figures]'
python scripts/build_visuals.py
npm install --no-save --package-lock=false playwright
npx playwright install chromium
node scripts/capture_report.cjs
```

The plot reads committed numeric results. Screenshots come from the actual HTML
report; browser QA checks both diagnosis states, a mobile viewport and absence of
external requests. [UI results](../benchmarks/results/ui.json).
