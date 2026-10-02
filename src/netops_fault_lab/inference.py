"""Single-fault Bayesian inference and one-step information-gain probe selection."""

from collections import defaultdict
import math

from . import __version__
from .schema import Case, HEALTHY, NoiseModel, Probe


def fit_noise(rows: list[tuple[Case, str]]) -> NoiseModel:
    """Beta(1,1) posterior means from labeled, independent training observations."""
    if not rows:
        raise ValueError("Training data is empty")
    counts = {"hit_fail": 0, "hit_total": 0, "background_fail": 0, "background_total": 0}
    healthy = 0
    for case, truth in rows:
        if truth not in case.hypotheses:
            raise ValueError("Training truth is not in the case hypotheses")
        healthy += truth == HEALTHY
        for probe in case.observed:
            key = "hit" if truth in probe.links else "background"
            counts[key + "_total"] += 1
            counts[key + "_fail"] += probe.outcome == "fail"
    if not counts["hit_total"] or not counts["background_total"]:
        raise ValueError("Training needs both affected and unaffected path observations")
    return NoiseModel(
        (counts["hit_fail"] + 1) / (counts["hit_total"] + 2),
        (counts["background_fail"] + 1) / (counts["background_total"] + 2),
        (healthy + 1) / (len(rows) + 2),
        {"source": "Beta(1,1) fit on labeled training data", "cases": len(rows),
         "topologies": len({case.topology_id for case, _ in rows}),
         "training_topology_ids": sorted({case.topology_id for case, _ in rows}), "counts": counts},
    )


def failure_probability(hypothesis: str, probe: Probe, model: NoiseModel) -> float:
    return model.hit_failure_probability if hypothesis in probe.links else model.background_failure_probability


def posterior(case: Case, model: NoiseModel) -> dict[str, float]:
    prior_link = (1 - model.healthy_prior) / len(case.links)
    scores = {}
    for hypothesis in case.hypotheses:
        value = math.log(model.healthy_prior if hypothesis == HEALTHY else prior_link)
        for probe in case.observed:
            q = failure_probability(hypothesis, probe, model)
            value += math.log(q) if probe.outcome == "fail" else math.log1p(-q)
        scores[hypothesis] = value
    maximum = max(scores.values())
    weights = {h: math.exp(s - maximum) for h, s in scores.items()}
    total = math.fsum(weights.values())
    return {h: w / total for h, w in weights.items()}


def entropy(probabilities) -> float:
    return -math.fsum(p * math.log2(p) for p in probabilities if p > 0)


def probe_information(probabilities: dict[str, float], probe: Probe, model: NoiseModel) -> dict:
    rates = {h: failure_probability(h, probe, model) for h in probabilities}
    prediction = math.fsum(probabilities[h] * rates[h] for h in probabilities)
    gain = entropy((prediction, 1 - prediction)) - math.fsum(
        probabilities[h] * entropy((q, 1 - q)) for h, q in rates.items())
    gain = max(0.0, gain)  # Floating-point cancellation can produce a tiny negative.
    return {"id": probe.id, "path": list(probe.path), "expected_gain_bits": gain,
            "cost": probe.cost, "gain_per_cost": gain / probe.cost,
            "predicted_failure_probability": prediction}


def diagnose(case: Case, model: NoiseModel | None = None, *, decision_threshold: float = 0.9) -> dict:
    if isinstance(decision_threshold, bool) or not isinstance(decision_threshold, (float, int)) or not 0 < decision_threshold < 1:
        raise ValueError("decision_threshold must be between 0 and 1")
    model = model or NoiseModel()
    probabilities = posterior(case, model)
    ranking = sorted(probabilities, key=lambda h: (-probabilities[h], h))
    signatures: dict[tuple[bool, ...], list[str]] = defaultdict(list)
    for hypothesis in case.hypotheses:
        signatures[tuple(hypothesis in p.links for p in case.observed)].append(hypothesis)
    equivalent = next(group for group in signatures.values() if ranking[0] in group)
    if not case.observed:
        status = "insufficient_evidence"
    elif len(equivalent) > 1:
        status = "ambiguous"
    elif probabilities[ranking[0]] < decision_threshold:
        status = "review"
    else:
        status = "consistent_with_healthy" if ranking[0] == HEALTHY else "candidate"
    recommendations = sorted((probe_information(probabilities, p, model) for p in case.available_probes),
                             key=lambda p: (-p["gain_per_cost"], p["id"]))
    recommendations = [p for p in recommendations if p["expected_gain_bits"] > 1e-12]
    evidence = []
    top = ranking[0]
    alternative = ranking[1]
    for probe in case.observed:
        qa, qb = (failure_probability(h, probe, model) for h in (top, alternative))
        factor = math.log(qa / qb) if probe.outcome == "fail" else math.log((1 - qa) / (1 - qb))
        evidence.append({"id": probe.id, "outcome": probe.outcome, "path": list(probe.path),
                         "supports_top_vs_runner_up_log_bayes_factor": factor})
    return {
        "schema_version": 1, "tool_version": __version__, "case_id": case.id,
        "topology_id": case.topology_id, "status": status,
        "decision_threshold": decision_threshold, "model": model.to_dict(),
        "observed_probes": len(case.observed), "missing_probes": len(case.observations) - len(case.observed),
        "entropy_bits": entropy(probabilities.values()),
        "ranking": [{"hypothesis": h, "probability": probabilities[h]} for h in ranking],
        "indistinguishable_from_top": sorted(equivalent),
        "equivalent_group_probability": sum(probabilities[h] for h in equivalent),
        "next_probes": recommendations,
        "evidence_comparison": {"top": top, "runner_up": alternative, "probes": evidence},
        "assumptions": ["At most one failed link, or healthy", "Known, unchanged, undirected probe paths",
                        "Conditionally independent probe outcomes with shared noise rates",
                        "Probabilities depend on this model; a ranking is not a verified root cause"],
    }
