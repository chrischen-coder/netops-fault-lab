"""Create the documented synthetic two-candidate incident and one extra outcome."""

import copy
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
case = {
    'schema_version': 1,
    'id': 'two-links-one-symptom',
    'topology_id': 'demo-branch',
    'links': [
        {'id': 'access', 'source': 'api', 'target': 'r1'},
        {'id': 'core', 'source': 'r1', 'target': 'r2'},
        {'id': 'database', 'source': 'r2', 'target': 'db'},
        {'id': 'cache', 'source': 'r1', 'target': 'cache'},
        {'id': 'monitor', 'source': 'monitor', 'target': 'r1'},
    ],
    'observations': [
        {'id': 'api-db', 'path': ['api', 'r1', 'r2', 'db'], 'outcome': 'fail'},
        {'id': 'monitor-db', 'path': ['monitor', 'r1', 'r2', 'db'], 'outcome': 'fail'},
        {'id': 'api-cache', 'path': ['api', 'r1', 'cache'], 'outcome': 'pass'},
        {'id': 'monitor-cache', 'path': ['monitor', 'r1', 'cache'], 'outcome': 'pass'},
    ],
    'available_probes': [
        {'id': 'probe-core', 'path': ['r1', 'r2'], 'cost': 1},
        {'id': 'probe-database', 'path': ['r2', 'db'], 'cost': 2},
        {'id': 'probe-access', 'path': ['api', 'r1'], 'cost': 1},
    ],
}
(root/'examples/ambiguous.json').write_text(json.dumps(case, indent=2)+'\n')
after = copy.deepcopy(case)
after['id'] = 'after-core-probe'
after['observations'].append({'id': 'probe-core', 'path': ['r1','r2'], 'outcome': 'fail'})
after['available_probes'] = [p for p in after['available_probes'] if p['id'] != 'probe-core']
(root/'examples/after-probe.json').write_text(json.dumps(after, indent=2)+'\n')
# The next outcome is scripted, not a measurement on a real network.
(root/'examples/truth.json').write_text(json.dumps({'ambiguous.json':'core','after-probe.json':'core'},indent=2)+'\n')
