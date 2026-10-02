"""Fit on training graphs; evaluate unchanged models on held-out graph structures."""

from collections import defaultdict
from dataclasses import replace
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import time

from . import __version__
from .inference import diagnose, entropy, fit_noise, posterior
from .schema import Case, HEALTHY
from .synthetic import SCENARIOS, incident, topologies


def candidate_features(case: Case, hypothesis: str) -> list[float]:
    fail = sum(p.outcome == 'fail' for p in case.observed)
    passed = len(case.observed) - fail
    hit_fail = sum(p.outcome == 'fail' and hypothesis in p.links for p in case.observed)
    hit_pass = sum(p.outcome == 'pass' and hypothesis in p.links for p in case.observed)
    return [hit_fail, hit_pass, fail - hit_fail, passed - hit_pass,
            hit_fail / max(fail, 1), hit_pass / max(passed, 1),
            float(hypothesis == HEALTHY), 1 / len(case.links)]


def fit_logistic(rows: list[tuple[Case, str]], seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    features, targets, weights = [], [], []
    for case, truth in rows:
        for hypothesis in case.hypotheses:
            features.append(candidate_features(case, hypothesis))
            targets.append(int(hypothesis == truth))
            weights.append(1 / len(case.hypotheses))
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    model.fit(features, targets, logisticregression__sample_weight=weights)
    return model


def rank(case: Case, method: str, model=None, logistic=None) -> list[str]:
    if method == 'bayes':
        scores = posterior(case, model)
    elif method == 'logistic':
        values = logistic.predict_proba([candidate_features(case, h) for h in case.hypotheses])[:, 1]
        scores = dict(zip(case.hypotheses, values))
    elif method == 'failed_path_votes':
        fail = [p for p in case.observed if p.outcome == 'fail']
        scores = {h: (sum(h in p.links for p in fail) if h != HEALTHY else float(not fail))
                  for h in case.hypotheses}
    else:
        raise ValueError('Unknown method')
    return sorted(scores, key=lambda h: (-scores[h], h))


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    return {'cases': n, 'top1': sum(r['correct'] for r in rows) / n,
            'top3': sum(r['rank'] <= 3 for r in rows) / n,
            'mrr': sum(1 / r['rank'] for r in rows) / n}


def probability_metrics(rows: list[dict]) -> dict:
    accepted = [r for r in rows if r['accepted']]
    bins = [[] for _ in range(10)]
    for row in rows:
        bins[min(int(row['confidence'] * 10), 9)].append(row)
    reliability = [{'count': len(bucket), 'mean_probability': sum(r['confidence'] for r in bucket) / len(bucket),
                    'accuracy': sum(r['correct'] for r in bucket) / len(bucket)}
                   for bucket in bins if bucket]
    return {'nll': sum(r['nll'] for r in rows) / len(rows),
            'brier_sum': sum(r['brier'] for r in rows) / len(rows),
            'ece_10_equal_width': sum(b['count'] * abs(b['mean_probability'] - b['accuracy']) for b in reliability) / len(rows),
            'coverage_at_0_9': len(accepted) / len(rows),
            'accepted_error_rate': sum(not r['correct'] for r in accepted) / len(accepted) if accepted else None,
            'reliability_bins': reliability}


def topology_bootstrap(rows: list[dict], seed: int = 20261002) -> list[float]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row['seed'], row['topology_id'])].append(row['correct'])
    values = list(groups.values())
    rng = random.Random(seed)
    estimates = []
    for _ in range(500):
        sample = [rng.choice(values) for _ in values]
        estimates.append(sum(sum(g) for g in sample) / sum(len(g) for g in sample))
    estimates.sort()
    return [estimates[12], estimates[487]]


def active_trial(case: Case, truth: str, model, seed: int, budget: int = 4) -> list[dict]:
    # Outcomes are sampled once per candidate and shared across both policies.
    stable_seed = int(hashlib.sha256(f'{seed}:{case.id}'.encode()).hexdigest()[:16], 16)
    rng = random.Random(stable_seed)
    outcomes = {p.id: ('fail' if rng.random() < (.9 if truth in p.links else .03) else 'pass')
                for p in case.available_probes}
    result = []
    for policy in ('information_gain', 'random'):
        current = replace(case, observations=case.observations[:3])
        chooser = random.Random(stable_seed + 1)
        used = 0
        chosen_id, chosen_outcome = '', ''
        for step in range(budget + 1):
            diagnosis = diagnose(current, model)
            ordering = [r['hypothesis'] for r in diagnosis['ranking']]
            result.append({'seed': seed, 'case_id': case.id, 'topology_id': case.topology_id,
                           'policy': policy, 'step': step, 'queries_used': used,
                           'correct': ordering[0] == truth, 'rank': ordering.index(truth) + 1,
                           'entropy_bits': diagnosis['entropy_bits'], 'selected_probe': chosen_id,
                           'selected_outcome': chosen_outcome})
            chosen_id, chosen_outcome = '', ''
            if step == budget or not current.available_probes:
                continue
            if policy == 'information_gain':
                if not diagnosis['next_probes']:
                    continue
                chosen_id = diagnosis['next_probes'][0]['id']
                chosen = next(p for p in current.available_probes if p.id == chosen_id)
            else:
                chosen = chooser.choice(current.available_probes)
                chosen_id = chosen.id
            chosen_outcome = outcomes[chosen_id]
            current = replace(current, observations=(*current.observations, replace(chosen, outcome=chosen_outcome)),
                              available_probes=tuple(p for p in current.available_probes if p.id != chosen_id))
            used += 1
    return result


def paired_active_interval(rows: list[dict]) -> dict:
    pairs = defaultdict(dict)
    for row in rows:
        if row['step'] == 4:
            pairs[(row['seed'], row['topology_id'], row['case_id'])][row['policy']] = int(row['correct'])
    grouped = defaultdict(list)
    for (seed, topology, _), scores in pairs.items():
        grouped[(seed, topology)].append(scores['information_gain'] - scores['random'])
    groups = list(grouped.values())
    rng = random.Random(20261002)
    estimates = []
    for _ in range(1000):
        sample = [rng.choice(groups) for _ in groups]
        estimates.append(sum(map(sum, sample)) / sum(map(len, sample)))
    estimates.sort()
    return {'additional_probe_budget': 4, 'samples': 1000,
            'difference_top1': sum(map(sum, groups)) / sum(map(len, groups)),
            'ci_95pct': [estimates[25], estimates[974]], 'seed': 20261002}


def _save_dataset(path: Path, rows: list[dict]) -> str:
    payload = ''.join(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n' for row in rows).encode()
    with path.open('xb') as stream:
        with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as compressed:
            compressed.write(payload)
    return hashlib.sha256(payload).hexdigest()


def _save_csv(path: Path, rows: list[dict]) -> None:
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(output: Path, *, seeds: list[int], train_topologies: int,
                  test_topologies: int, cases_per_topology: int) -> dict:
    import numpy as np
    if not seeds or len(seeds) != len(set(seeds)) or any(not 0 <= s < 2**32 for s in seeds):
        raise ValueError('Use distinct integer seeds between 0 and 2**32 - 1')
    if any(isinstance(n, bool) or n < 2 for n in (train_topologies, test_topologies, cases_per_topology)):
        raise ValueError('Use at least two train/test topologies and cases per topology')
    if output.exists():
        raise ValueError('Benchmark output directory already exists; choose a new directory')
    output.mkdir(parents=True)
    predictions, active, manifest = [], [], []
    for seed in seeds:
        graphs = topologies(train_topologies + test_topologies, seed)
        train_graphs, test_graphs = graphs[:train_topologies], graphs[train_topologies:]
        assert not {g['topology_id'] for g in train_graphs} & {g['topology_id'] for g in test_graphs}
        training = [incident(g, seed * 100000 + gi * 100 + j, f's{seed}-train-{gi}-{j}')
                    for gi, g in enumerate(train_graphs) for j in range(cases_per_topology)]
        train_hash = _save_dataset(output / f'train-{seed}.jsonl.gz', training)
        train_rows = [(Case.from_dict(r['case']), r['truth']) for r in training]
        model = fit_noise(train_rows)
        model.provenance['training_uncompressed_sha256'] = train_hash
        model.provenance['training_topology_ids'] = sorted({c.topology_id for c, _ in train_rows})
        (output / f'model-{seed}.json').write_text(json.dumps(model.to_dict(), indent=2) + '\n')
        logistic = fit_logistic(train_rows, seed)
        scaler, classifier = logistic.steps[0][1], logistic.steps[1][1]
        (output / f'logistic-{seed}.json').write_text(json.dumps({
            'feature_order': ['hit_fail', 'hit_pass', 'off_path_fail', 'off_path_pass',
                              'fraction_fail_covered', 'fraction_pass_covered', 'healthy_indicator', 'inverse_links'],
            'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(),
            'coefficients': classifier.coef_.tolist(), 'intercept': classifier.intercept_.tolist(),
            'classes': classifier.classes_.tolist(), 'iterations': classifier.n_iter_.tolist(),
        }, indent=2) + '\n')
        seed_manifest = {'seed': seed, 'train_sha256': train_hash,
                         'train_topology_ids': [g['topology_id'] for g in train_graphs],
                         'test_topology_ids': [g['topology_id'] for g in test_graphs], 'test_sha256': {}}
        for scenario in SCENARIOS:
            test = [incident(g, seed * 100000 + 50000 + gi * 100 + j,
                             f's{seed}-test-{gi}-{j}', scenario)
                    for gi, g in enumerate(test_graphs) for j in range(cases_per_topology)]
            seed_manifest['test_sha256'][scenario] = _save_dataset(output / f'test-{seed}-{scenario}.jsonl.gz', test)
            for raw in test:
                case, truth = Case.from_dict(raw['case']), raw['truth']
                for method in ('failed_path_votes', 'logistic', 'bayes'):
                    start = time.perf_counter()
                    ordering = rank(case, method, model, logistic)
                    duration = time.perf_counter() - start
                    row = {'seed': seed, 'scenario': scenario, 'case_id': case.id, 'topology_id': case.topology_id,
                           'method': method, 'truth': truth, 'prediction': ordering[0], 'rank': ordering.index(truth) + 1,
                           'correct': ordering[0] == truth, 'seconds': duration,
                           'confidence': '', 'accepted': '', 'nll': '', 'brier': ''}
                    if method == 'bayes':
                        distribution = posterior(case, model)
                        diagnosis = diagnose(case, model)
                        row.update(confidence=distribution[ordering[0]],
                                   accepted=diagnosis['status'] in ('candidate', 'consistent_with_healthy'),
                                   nll=-math.log(max(distribution[truth], 1e-15)),
                                   brier=math.fsum((p - int(h == truth)) ** 2 for h, p in distribution.items()))
                    predictions.append(row)
                if scenario == 'nominal':
                    active.extend(active_trial(case, truth, model, seed))
        manifest.append(seed_manifest)
    summary = {}
    for scenario in SCENARIOS:
        summary[scenario] = {}
        for method in ('failed_path_votes', 'logistic', 'bayes'):
            selected = [r for r in predictions if r['scenario'] == scenario and r['method'] == method]
            values = summarize(selected)
            values['top1_topology_bootstrap_95pct'] = topology_bootstrap(selected)
            values['median_rank_ms'] = float(np.median([r['seconds'] for r in selected]) * 1000)
            if method == 'bayes':
                values.update(probability_metrics(selected))
            summary[scenario][method] = values
    active_summary = {policy: {str(step): summarize([r for r in active if r['policy'] == policy and r['step'] == step])
                              for step in range(5)} for policy in ('information_gain', 'random')}
    _save_csv(output / 'predictions.csv', predictions)
    _save_csv(output / 'active-probes.csv', active)
    result = {'schema_version': 1, 'tool_version': __version__, 'synthetic': True,
              'parameters': {'seeds': seeds, 'train_topologies': train_topologies, 'test_topologies': test_topologies,
                             'cases_per_topology': cases_per_topology, 'healthy_probability': .15,
                             'scenarios': SCENARIOS, 'threshold': .9},
              'environment': {'python': platform.python_version(), 'system': platform.system(),
                              'machine': platform.machine(), **{p: importlib.metadata.version(p) for p in ('numpy','scikit-learn','networkx')}},
              'manifest': manifest, 'summary': summary, 'active_probes': active_summary,
              'active_paired_topology_bootstrap': paired_active_interval(active)}
    (output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
