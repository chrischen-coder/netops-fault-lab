"""Controlled path-probe simulations; not packets, emulation or field telemetry."""

import random
import warnings

from .schema import HEALTHY

SCENARIOS = {
    'nominal': {'hit': .9, 'background': .03, 'missing': .1, 'copies': 1},
    'noise_shift': {'hit': .65, 'background': .15, 'missing': .1, 'copies': 1},
    'missing_55pct': {'hit': .9, 'background': .03, 'missing': .55, 'copies': 1},
    'copied_alarms_x4': {'hit': .9, 'background': .03, 'missing': .1, 'copies': 4},
}


def topologies(count: int, seed: int) -> list[dict]:
    import networkx as nx
    rng = random.Random(seed)
    graphs, used = [], set()
    attempts = 0
    while len(graphs) < count:
        attempts += 1
        if attempts > count * 200:
            raise ValueError('Could not generate enough distinct topology hashes')
        size = rng.randint(8, 12)
        edges = {(i, rng.randrange(i)) for i in range(1, size)}
        edges = {tuple(sorted(edge)) for edge in edges}
        for left in range(size):
            for right in range(left + 1, size):
                if rng.random() < .12:
                    edges.add((left, right))
        graph = nx.Graph()
        graph.add_nodes_from(range(size))
        graph.add_edges_from(sorted(edges))
        # Treat equal hashes as one group, including any WL collisions.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='The hashes produced for graphs without node or edge attributes.*')
            fingerprint = nx.weisfeiler_lehman_graph_hash(graph, iterations=4)
        if fingerprint in used:
            continue
        used.add(fingerprint)
        links = [{'id': f'link-{i:02d}', 'source': f'n{u}', 'target': f'n{v}'}
                 for i, (u, v) in enumerate(sorted(edges))]
        routes = [[f'n{v}' for v in nx.shortest_path(graph, u, v)]
                  for u in range(size) for v in range(u + 1, size)]
        graphs.append({'topology_id': fingerprint, 'links': links, 'routes': routes})
    return graphs


def incident(graph: dict, seed: int, identifier: str, scenario: str = 'nominal') -> dict:
    settings = SCENARIOS[scenario]
    rng = random.Random(seed)
    truth = HEALTHY if rng.random() < .15 else rng.choice(graph['links'])['id']
    lookup = {frozenset((link['source'], link['target'])): link['id'] for link in graph['links']}
    paths = list(graph['routes'])
    rng.shuffle(paths)
    observations = []
    for index, path in enumerate(paths[:14]):
        edges = {lookup[frozenset(pair)] for pair in zip(path, path[1:])}
        outcome_draw, missing_draw = rng.random(), rng.random()
        rate = settings['hit'] if truth in edges else settings['background']
        outcome = 'fail' if outcome_draw < rate else 'pass'
        if missing_draw < settings['missing']:
            outcome = 'missing'
        for copy in range(settings['copies']):
            observations.append({'id': f'obs-{index}-{copy}', 'path': path, 'outcome': outcome})
    candidates = [{'id': f'next-{i}', 'path': path, 'cost': 1.0} for i, path in enumerate(paths[14:26])]
    return {'case': {'id': identifier, 'topology_id': graph['topology_id'],
                     'links': graph['links'], 'observations': observations, 'available_probes': candidates},
            'truth': truth}
