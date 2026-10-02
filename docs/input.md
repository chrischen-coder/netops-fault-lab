# Input and output contract

## Incident JSON

```json
{
  "schema_version": 1,
  "id": "incident-1",
  "topology_id": "network-snapshot-1",
  "links": [
    {"id": "uplink", "source": "client", "target": "router"},
    {"id": "downlink", "source": "router", "target": "server"}
  ],
  "observations": [
    {"id": "end-to-end", "path": ["client", "router", "server"], "outcome": "fail"}
  ],
  "available_probes": [
    {"id": "check-uplink", "path": ["client", "router"], "cost": 1}
  ]
}
```

Links are undirected. IDs and endpoint pairs must be unique; self-links and
parallel links are unsupported. Each path is an ordered list of at least two
unique nodes, and every consecutive edge must exist. Paths describe the actual
route, which may differ from a shortest path.

Probe IDs are unique across observations and available candidates. Outcomes are
`pass`, `fail` or `missing`; a missing outcome is not evidence of success.
Candidate costs are finite and positive, defaulting to 1. Repeated measurements
must be independent under the model; copies of one event are not independent.

Identifiers are nonempty strings of at most 200 characters, without control or
unpaired surrogate characters. `__healthy__` is reserved. Unknown fields,
duplicate JSON keys, NaN and Infinity are rejected. Ground truth is never an
incident field. `available_probes` can be omitted.

## Training JSONL

Each line wraps a valid incident with a label:

```text
{"case": { ...incident fields... }, "truth": "uplink"}
```

The example above abbreviates the incident object; use complete valid JSON.
Labels are link IDs or `__healthy__`. Case IDs must be unique. Both `.jsonl` and
`.jsonl.gz` are supported. Training requires observed affected and unaffected
paths. This command writes estimated noise parameters and provenance:

```bash
netops-fault-lab fit train.jsonl --output model.json
netops-fault-lab diagnose incident.json --model model.json --format json
```

Training and incident files are loaded into memory. The bundled scale is small;
large-network throughput and bounded memory have not been established.

## Python API

```python
from netops_fault_lab import Case, NoiseModel, diagnose
from netops_fault_lab.schema import read_json

case = Case.from_dict(read_json("examples/ambiguous.json"))
model = NoiseModel.from_dict(read_json("model.json"))
result = diagnose(case, model, decision_threshold=0.9)
print(result["status"], result["next_probes"])
```

Use `Case.from_dict` for validated inputs. The low-level frozen data classes can
also be constructed directly by trusted callers, who are responsible for consistency.

## Report JSON, schema 1

- `ranking`: all link and healthy hypotheses with model probabilities, descending.
- `status`: `candidate`, `consistent_with_healthy`, `review`, `ambiguous` or `insufficient_evidence`.
- `indistinguishable_from_top`: hypotheses with the same observed path signature.
- `next_probes`: positive-information candidates, ordered by gain per cost.
- `evidence_comparison`: observed paths and log Bayes factors for the top two hypotheses.
- `model`: noise rates, prior and fit provenance; the CLI also records `input_sha256`.

The 0.9 threshold is a user-adjustable heuristic, not a calibrated service guarantee.
No probe is executed and no network device is changed. HTML embeds no external assets.
Reports contain input identifiers and paths; review their content before sharing them.

CLI exit `0` means a valid report was produced, including ambiguous results.
Exit `2` means invalid input, parameters, unavailable optional dependencies or I/O failure.
Output parents must exist; existing files are not overwritten.
